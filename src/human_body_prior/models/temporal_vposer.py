#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal VPoser: VPoser + Transformer for Sequential Human Motion Modeling
Combines VPoser's anatomical constraints with temporal modeling capabilities
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from human_body_prior.models.model_components import BatchFlatten
from human_body_prior.models.vposer_model import ContinousRotReprDecoder, NormalDistDecoder
from human_body_prior.tools.rotation_tools import matrot2aa


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        # Access registered buffer safely
        pe_buffer = getattr(self, 'pe')
        pe_slice = pe_buffer[:x.size(0), :].to(x.device)
        return x + pe_slice


class TemporalVPoserEncoder(nn.Module):
    """Temporal encoder: processes sequences of poses"""
    def __init__(self, model_ps):
        super(TemporalVPoserEncoder, self).__init__()
        
        num_neurons = model_ps.model_params.num_neurons
        self.latentD = model_ps.model_params.latentD
        self.num_joints = 21
        n_features = self.num_joints * 3  # 63
        
        # Per-frame encoder (same as original VPoser but without BatchFlatten)
        self.frame_encoder = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
        )
        
        # Distribution parameters
        self.mu = nn.Linear(num_neurons, self.latentD)
        self.logvar = nn.Linear(num_neurons, self.latentD)
    
    def forward(self, pose_sequence):
        """
        Args:
            pose_sequence: (batch_size, seq_len, 63)
        Returns:
            distributions: (batch_size, seq_len, latentD) - one distribution per frame
        """
        batch_size, seq_len, n_features = pose_sequence.shape
        
        # Reshape to process all frames together
        pose_flat = pose_sequence.reshape(batch_size * seq_len, n_features)
        
        # Encode each frame
        encoded = self.frame_encoder(pose_flat)
        
        # Get distribution parameters
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        
        # Reshape back to sequence format
        mu = mu.reshape(batch_size, seq_len, self.latentD)
        logvar = logvar.reshape(batch_size, seq_len, self.latentD)
        
        # Create distributions
        distributions = torch.distributions.normal.Normal(mu, F.softplus(logvar))
        
        return distributions


class TemporalVPoserDecoder(nn.Module):
    """Temporal decoder: processes sequences of latent codes"""
    def __init__(self, model_ps):
        super(TemporalVPoserDecoder, self).__init__()
        
        num_neurons = model_ps.model_params.num_neurons
        self.latentD = model_ps.model_params.latentD
        self.num_joints = 21
        
        # Per-frame decoder (same as original VPoser)
        self.frame_decoder = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        )
    
    def forward(self, latent_sequence):
        """
        Args:
            latent_sequence: (batch_size, seq_len, latentD)
        Returns:
            pose_sequence: (batch_size, seq_len, 63)
        """
        batch_size, seq_len, latentD = latent_sequence.shape
        
        # Reshape to process all frames together
        latent_flat = latent_sequence.reshape(batch_size * seq_len, latentD)
        
        # Decode each frame
        decoded = self.frame_decoder(latent_flat)  # (batch_size * seq_len, 21, 3, 3)
        
        # Convert to axis-angle and reshape
        pose_body = matrot2aa(decoded.reshape(-1, 3, 3)).reshape(batch_size, seq_len, -1)
        
        return {
            'pose_body': pose_body,
            'pose_body_matrot': decoded.reshape(batch_size, seq_len, -1)
        }


class TemporalTransformer(nn.Module):
    """Transformer for temporal modeling in latent space"""
    def __init__(self, model_ps):
        super(TemporalTransformer, self).__init__()
        
        self.latentD = model_ps.model_params.latentD
        
        # Transformer configuration (inspired by Learning to Listen)
        self.d_model = model_ps.temporal_params.get('d_model', 256)
        self.num_layers = model_ps.temporal_params.get('num_layers', 4)
        self.num_heads = model_ps.temporal_params.get('num_heads', 8)
        self.dim_feedforward = model_ps.temporal_params.get('dim_feedforward', 512)
        self.dropout = model_ps.temporal_params.get('dropout', 0.1)
        
        # Project latent to transformer dimension
        self.latent_proj = nn.Linear(self.latentD, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.latentD)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        # Transformer layers
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=False  # (seq_len, batch_size, d_model)
        )
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
    
    def forward(self, latent_sequence, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            latent_sequence: (batch_size, seq_len, latentD)
            src_mask: (seq_len, seq_len) - attention mask
            src_key_padding_mask: (batch_size, seq_len) - padding mask
        Returns:
            refined_sequence: (batch_size, seq_len, latentD)
        """
        batch_size, seq_len, latentD = latent_sequence.shape
        
        # Project to transformer dimension
        x = self.latent_proj(latent_sequence)  # (batch_size, seq_len, d_model)
        
        # Transpose for transformer: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Transpose back: (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)
        
        # Project back to latent dimension
        refined_sequence = self.output_proj(x)  # (batch_size, seq_len, latentD)
        
        return refined_sequence


class TemporalVPoser(nn.Module):
    """
    Temporal VPoser: Sequential human pose modeling with anatomical constraints
    
    Architecture:
    Input: (batch_size, seq_len, 63) pose sequences
    → Encoder: (batch_size, seq_len, latentD) latent distributions
    → Transformer: (batch_size, seq_len, latentD) temporally refined latents
    → Decoder: (batch_size, seq_len, 63) reconstructed pose sequences
    """
    
    def __init__(self, model_ps):
        super(TemporalVPoser, self).__init__()
        
        self.latentD = model_ps.model_params.latentD
        self.num_joints = 21
        
        # Components
        self.encoder = TemporalVPoserEncoder(model_ps)
        self.temporal_transformer = TemporalTransformer(model_ps)
        self.decoder = TemporalVPoserDecoder(model_ps)
        
        # KL divergence weight
        self.kl_weight = model_ps.temporal_params.get('kl_weight', 1.0)
    
    def encode(self, pose_sequence):
        """
        Args:
            pose_sequence: (batch_size, seq_len, 63)
        Returns:
            distributions: (batch_size, seq_len, latentD)
        """
        return self.encoder(pose_sequence)
    
    def decode(self, latent_sequence):
        """
        Args:
            latent_sequence: (batch_size, seq_len, latentD)
        Returns:
            pose_results: dict with pose_body and pose_body_matrot
        """
        return self.decoder(latent_sequence)
    
    def forward(self, pose_sequence, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            pose_sequence: (batch_size, seq_len, 63)
            src_mask: (seq_len, seq_len) - attention mask for autoregressive generation
            src_key_padding_mask: (batch_size, seq_len) - padding mask
        Returns:
            results: dict with reconstructed poses and latent information
        """
        batch_size, seq_len, _ = pose_sequence.shape
        
        # Encode to latent distributions
        q_z = self.encode(pose_sequence)
        
        # Sample from distributions
        z_sample = q_z.rsample()  # (batch_size, seq_len, latentD)
        
        # Apply temporal transformer
        z_refined = self.temporal_transformer(z_sample, src_mask, src_key_padding_mask)
        
        # Decode refined latents
        decode_results = self.decode(z_refined)
        
        # Compile results
        results = {
            'pose_body': decode_results['pose_body'],
            'pose_body_matrot': decode_results['pose_body_matrot'],
            'poZ_body_mean': q_z.mean,
            'poZ_body_std': q_z.scale,
            'q_z': q_z,
            'z_sample': z_sample,
            'z_refined': z_refined
        }
        
        return results
    
    def sample_poses(self, num_poses, seq_len, seed=None):
        """
        Sample random pose sequences
        
        Args:
            num_poses: number of sequences to generate
            seq_len: length of each sequence
            seed: random seed
        Returns:
            pose_results: dict with generated pose sequences
        """
        if seed is not None:
            np.random.seed(seed)
        
        some_weight = next(self.parameters())
        dtype = some_weight.dtype
        device = some_weight.device
        
        self.eval()
        with torch.no_grad():
            # Generate random latent sequences
            z_random = torch.randn(num_poses, seq_len, self.latentD, 
                                 dtype=dtype, device=device)
            
            # Apply temporal transformer for coherence
            z_refined = self.temporal_transformer(z_random)
            
            # Decode to poses
            results = self.decode(z_refined)
        
        return results
    
    def generate_sequence(self, initial_poses, future_len, temperature=1.0):
        """
        Generate future poses given initial poses (autoregressive)
        
        Args:
            initial_poses: (batch_size, init_len, 63)
            future_len: number of future frames to generate
            temperature: sampling temperature
        Returns:
            full_sequence: (batch_size, init_len + future_len, 63)
        """
        batch_size, init_len, _ = initial_poses.shape
        device = initial_poses.device
        
        # Create causal mask
        total_len = init_len + future_len
        causal_mask = torch.triu(torch.ones(total_len, total_len), diagonal=1).bool()
        causal_mask = causal_mask.to(device)
        
        # Start with initial poses
        current_sequence = initial_poses
        
        self.eval()
        with torch.no_grad():
            for i in range(future_len):
                # Encode current sequence
                q_z = self.encode(current_sequence)
                z_sample = q_z.mean  # Use mean for stable generation
                
                # Add temperature scaling
                if temperature != 1.0:
                    z_sample = z_sample / temperature
                
                # Apply transformer with causal mask
                current_len = current_sequence.shape[1]
                mask = causal_mask[:current_len, :current_len]
                z_refined = self.temporal_transformer(z_sample, src_mask=mask)
                
                # Decode to get next pose
                decode_results = self.decode(z_refined)
                
                # Take the last frame as the next pose
                next_pose = decode_results['pose_body'][:, -1:, :]  # (batch_size, 1, 63)
                
                # Append to sequence
                current_sequence = torch.cat([current_sequence, next_pose], dim=1)
        
        return current_sequence


def create_temporal_vposer_config(
    num_neurons=512,
    latentD=32,
    d_model=256,
    num_layers=4,
    num_heads=8,
    dim_feedforward=512,
    dropout=0.1,
    kl_weight=1.0
):
    """
    Create configuration for Temporal VPoser
    
    Args:
        num_neurons: hidden layer size for encoder/decoder
        latentD: latent space dimension
        d_model: transformer hidden dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
        dim_feedforward: transformer feedforward dimension
        dropout: dropout rate
        kl_weight: KL divergence loss weight
    """
    class Config:
        def __init__(self):
            self.model_params = type('ModelParams', (), {
                'num_neurons': num_neurons,
                'latentD': latentD
            })()
            
            self.temporal_params = {
                'd_model': d_model,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'dim_feedforward': dim_feedforward,
                'dropout': dropout,
                'kl_weight': kl_weight
            }
    
    return Config()


# Example usage
if __name__ == "__main__":
    # Create model configuration
    config = create_temporal_vposer_config(
        num_neurons=512,
        latentD=32,
        d_model=256,
        num_layers=4,
        num_heads=8
    )
    
    # Create model
    model = TemporalVPoser(config)
    
    # Test with sample data
    batch_size, seq_len = 4, 16
    sample_poses = torch.randn(batch_size, seq_len, 63)
    
    # Forward pass
    results = model(sample_poses)
    print(f"Input shape: {sample_poses.shape}")
    print(f"Output shape: {results['pose_body'].shape}")
    print(f"Latent shape: {results['z_refined'].shape}")
    
    # Generate random sequences
    generated = model.sample_poses(num_poses=2, seq_len=10)
    print(f"Generated shape: {generated['pose_body'].shape}")
    
    # Autoregressive generation
    initial = torch.randn(2, 5, 63)
    future_seq = model.generate_sequence(initial, future_len=5)
    print(f"Future sequence shape: {future_seq.shape}") 