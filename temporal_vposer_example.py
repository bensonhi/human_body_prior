#!/usr/bin/env python3
"""
Example script showing how to use Temporal VPoser with BEAT2 dataset
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

import pandas as pd

import sys
import os
sys.path.append('src')
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from human_body_prior.models.temporal_vposer import TemporalVPoser, create_temporal_vposer_config


class BEAT2SequenceDataset(Dataset):
    """Dataset for BEAT2 SMPL-X pose sequences"""
    
    def __init__(self, beat2_root, split_type='train', sequence_length=16, stride=8, 
                 speakers=None, max_files_per_speaker=None):
        """
        Args:
            beat2_root: Path to BEAT2 dataset root (e.g., 'BEAT2/beat_english_v2.0.0')
            split_type: 'train', 'val', 'test', or 'additional'
            sequence_length: Length of pose sequences
            stride: Stride between sequences
            speakers: List of speaker names to include (None for all)
            max_files_per_speaker: Maximum number of files per speaker (None for all)
        """
        self.beat2_root = Path(beat2_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.split_type = split_type
        
        # Load train/test split information
        split_file = self.beat2_root / 'train_test_split.csv'
        self.split_df = pd.read_csv(split_file)
        
        # Filter by split type
        self.file_ids = self.split_df[self.split_df['type'] == split_type]['id'].tolist()
        
        # Filter by speakers if specified
        if speakers is not None:
            self.file_ids = [fid for fid in self.file_ids 
                           if any(speaker in fid for speaker in speakers)]
        
        # Limit files per speaker if specified
        if max_files_per_speaker is not None:
            speaker_counts = {}
            filtered_file_ids = []
            for fid in self.file_ids:
                speaker = fid.split('_')[1]  # Extract speaker name
                if speaker not in speaker_counts:
                    speaker_counts[speaker] = 0
                if speaker_counts[speaker] < max_files_per_speaker:
                    filtered_file_ids.append(fid)
                    speaker_counts[speaker] += 1
            self.file_ids = filtered_file_ids
        
        # Load pose data from all files
        self.pose_sequences = []
        self.sequence_metadata = []
        
        smplx_dir = self.beat2_root / 'smplxflame_30'
        
        total_poses = 0
        loaded_files = 0
        
        for file_id in self.file_ids:
            npz_file = smplx_dir / f'{file_id}.npz'
            
            if not npz_file.exists():
                print(f"Warning: File not found: {npz_file}")
                continue
            
            try:
                data = np.load(npz_file)
                
                # Extract body poses (columns 3:66 from the 165D poses)
                full_poses = data['poses']  # Shape: (N, 165)
                body_poses = full_poses[:, 3:66]  # Shape: (N, 63)
                
                # Create sequences from this file
                file_sequences = []
                for i in range(0, len(body_poses) - sequence_length + 1, stride):
                    sequence = body_poses[i:i + sequence_length]
                    file_sequences.append(sequence)
                    
                    # Store metadata
                    speaker = file_id.split('_')[1]
                    self.sequence_metadata.append({
                        'file_id': file_id,
                        'speaker': speaker,
                        'start_frame': i,
                        'end_frame': i + sequence_length
                    })
                
                self.pose_sequences.extend(file_sequences)
                total_poses += len(body_poses)
                loaded_files += 1
                
                print(f"Loaded {file_id}: {len(body_poses)} poses -> {len(file_sequences)} sequences")
                
            except Exception as e:
                print(f"Error loading {file_id}: {e}")
                continue
        
        print(f"\n=== BEAT2 Dataset Loaded ===")
        print(f"Split: {split_type}")
        print(f"Total files: {loaded_files}")
        print(f"Total poses: {total_poses}")
        print(f"Total sequences: {len(self.pose_sequences)}")
        print(f"Sequence length: {sequence_length}")
        print(f"Stride: {stride}")
        
        # Print speaker distribution
        speakers_in_data = {}
        for meta in self.sequence_metadata:
            speaker = meta['speaker']
            speakers_in_data[speaker] = speakers_in_data.get(speaker, 0) + 1
        
        print(f"Speaker distribution:")
        for speaker, count in sorted(speakers_in_data.items()):
            print(f"  {speaker}: {count} sequences")
        
        # Convert to numpy array
        self.pose_sequences = np.array(self.pose_sequences, dtype=np.float32)
        print(f"Final dataset shape: {self.pose_sequences.shape}")
    
    def __len__(self):
        return len(self.pose_sequences)
    
    def __getitem__(self, idx):
        sequence = self.pose_sequences[idx]  # Shape: (seq_len, 63)
        metadata = self.sequence_metadata[idx]
        
        return {
            'poses': torch.FloatTensor(sequence),
            'metadata': metadata
        }


def temporal_vae_loss(recon_poses, target_poses, q_z, kl_weight=1.0):
    """
    Compute VAE loss for temporal sequences
    
    Args:
        recon_poses: Reconstructed poses (batch_size, seq_len, 63)
        target_poses: Target poses (batch_size, seq_len, 63)
        q_z: Latent distributions (batch_size, seq_len, latentD)
        kl_weight: Weight for KL divergence loss
    """
    # Reconstruction loss
    recon_loss = nn.MSELoss()(recon_poses, target_poses)
    
    # KL divergence loss (sum over sequence and latent dimensions)
    kl_loss = torch.distributions.kl.kl_divergence(
        q_z, 
        torch.distributions.Normal(0, 1)
    ).sum(dim=-1).mean()  # Mean over batch and sequence
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }


def train_temporal_vposer(model, train_loader, val_loader, num_epochs=10, lr=1e-3, kl_weight=1.0):
    """
    Training loop for Temporal VPoser with validation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            poses = batch['poses'].to(device)  # (batch_size, seq_len, 63)
            
            # Forward pass
            optimizer.zero_grad()
            results = model(poses)
            
            # Compute loss
            loss_dict = temporal_vae_loss(
                results['pose_body'], 
                poses, 
                results['q_z'], 
                kl_weight=kl_weight
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            # Accumulate losses
            train_loss += loss_dict['total_loss'].item()
            train_recon_loss += loss_dict['recon_loss'].item()
            train_kl_loss += loss_dict['kl_loss'].item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: '
                      f'Loss: {loss_dict["total_loss"].item():.6f}, '
                      f'Recon: {loss_dict["recon_loss"].item():.6f}, '
                      f'KL: {loss_dict["kl_loss"].item():.6f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                poses = batch['poses'].to(device)
                results = model(poses)
                
                loss_dict = temporal_vae_loss(
                    results['pose_body'], 
                    poses, 
                    results['q_z'], 
                    kl_weight=kl_weight
                )
                
                val_loss += loss_dict['total_loss'].item()
                val_recon_loss += loss_dict['recon_loss'].item()
                val_kl_loss += loss_dict['kl_loss'].item()
        
        # Average losses
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)
        
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train - Total: {train_loss:.6f}, Recon: {train_recon_loss:.6f}, KL: {train_kl_loss:.6f}')
        print(f'  Val   - Total: {val_loss:.6f}, Recon: {val_recon_loss:.6f}, KL: {val_kl_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'best_temporal_vposer.pth')
            print(f'  New best model saved! Val loss: {val_loss:.6f}')
        
        scheduler.step()
        print(f'  Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        print()


def main():
    """Main function demonstrating Temporal VPoser usage with BEAT2 dataset"""
    
    # Configuration
    config = create_temporal_vposer_config(
        num_neurons=512,      # Hidden layer size
        latentD=32,           # Latent space dimension
        d_model=256,          # Transformer hidden dimension
        num_layers=4,         # Number of transformer layers
        num_heads=8,          # Number of attention heads
        dim_feedforward=512,  # Transformer feedforward dimension
        dropout=0.1,          # Dropout rate
        kl_weight=1.0         # KL divergence weight
    )
    
    # Create model
    model = TemporalVPoser(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Dataset configuration
    beat2_root = "BEAT2/beat_english_v2.0.0"
    sequence_length = 16
    stride = 8
    batch_size = 8
    
    # Select specific speakers for faster training (optional)
    # Use None to include all speakers
    selected_speakers = ['miranda', 'sophie', 'carla']  # Start with a few speakers
    max_files_per_speaker = 10  # Limit files per speaker for faster training
    
    # Create datasets
    print("Loading training dataset...")
    train_dataset = BEAT2SequenceDataset(
        beat2_root=beat2_root,
        split_type='train',
        sequence_length=sequence_length,
        stride=stride,
        speakers=selected_speakers,
        max_files_per_speaker=max_files_per_speaker
    )
    
    print("\nLoading validation dataset...")
    val_dataset = BEAT2SequenceDataset(
        beat2_root=beat2_root,
        split_type='val',
        sequence_length=sequence_length,
        stride=stride,
        speakers=selected_speakers,
        max_files_per_speaker=max_files_per_speaker
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} sequences")
    print(f"  Validation: {len(val_dataset)} sequences")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Train model
    print("\nStarting training...")
    train_temporal_vposer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        lr=1e-3,
        kl_weight=0.1  # Start with lower KL weight
    )
    
    # Test model capabilities
    print("\nTesting model capabilities...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Ensure model is on correct device
    model.eval()
    with torch.no_grad():
        # Get a sample batch
        sample_batch = next(iter(val_loader))
        sample_poses = sample_batch['poses'].to(device)  # Move to device
        sample_metadata = sample_batch['metadata']
        
        # 1. Test reconstruction
        results = model(sample_poses)
        print(f"Reconstruction test - Input: {sample_poses.shape}, Output: {results['pose_body'].shape}")
        
        # Calculate reconstruction error
        recon_error = nn.MSELoss()(results['pose_body'], sample_poses)
        print(f"Reconstruction error: {recon_error.item():.6f}")
        
        # 2. Test sequence generation
        generated = model.sample_poses(num_poses=3, seq_len=20)
        print(f"Generated sequences: {generated['pose_body'].shape}")
        
        # 3. Test autoregressive generation
        initial_poses = sample_poses[:2, :5, :]  # First 5 frames from 2 sequences
        future_sequence = model.generate_sequence(initial_poses, future_len=10)
        print(f"Autoregressive generation - Initial: {initial_poses.shape}, Future: {future_sequence.shape}")
        
        # 4. Test with specific speakers
        print(f"\nSample metadata:")
        batch_size = len(sample_metadata['speaker'])
        for i in range(min(3, batch_size)):
            print(f"  Sequence {i}: Speaker {sample_metadata['speaker'][i]}, File {sample_metadata['file_id'][i]}")
    
    # Save final model
    torch.save(model.state_dict(), 'temporal_vposer_beat2_final.pth')
    print("\nFinal model saved to 'temporal_vposer_beat2_final.pth'")
    
    # Test with different speakers
    print("\nTesting with different speakers...")
    test_dataset = BEAT2SequenceDataset(
        beat2_root=beat2_root,
        split_type='test',
        sequence_length=sequence_length,
        stride=stride,
        speakers=['miranda'],  # Test with one speaker
        max_files_per_speaker=2
    )
    
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        test_batch = next(iter(test_loader))
        test_poses = test_batch['poses'].to(device)  # Move to device
        
        # Test generalization
        with torch.no_grad():
            test_results = model(test_poses)
            test_recon_error = nn.MSELoss()(test_results['pose_body'], test_poses)
            print(f"Test reconstruction error: {test_recon_error.item():.6f}")


if __name__ == "__main__":
    main() 