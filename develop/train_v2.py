"""
Improved training script with contrastive learning for cross-session generalization.
Based on STNDT and NDT paper insights.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import json

from config import (
    DEVICE, MONKEY_CONFIGS, TRAIN_CONFIG, MODEL_CONFIG, MODEL_CONFIG_V2,
    CHECKPOINT_DIR, LOG_DIR, INPUT_WINDOW, FORECAST_WINDOW, AUG_CONFIG
)
from data import NeuroForecastDataset, load_all_training_data
from models import SpatioTemporalForecasterV2, ContrastiveLoss


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience=15, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def apply_augmentation(input_seq, aug_config):
    """
    Apply augmentation to input sequence for contrastive learning.
    Returns augmented version of the input.
    """
    augmented = input_seq.clone()

    # Add Gaussian noise
    if aug_config.get('noise_std', 0) > 0:
        noise = torch.randn_like(augmented) * aug_config['noise_std']
        augmented = augmented + noise

    # Random channel dropout (different from training augmentation)
    if aug_config.get('channel_masking_prob', 0) > 0:
        B, T, C, F = augmented.shape
        mask = torch.rand(B, 1, C, 1, device=augmented.device) > aug_config['channel_masking_prob']
        augmented = augmented * mask.float()

    return augmented


class TrainerV2:
    """Training orchestrator with contrastive learning support."""

    def __init__(self, monkey_name, config=None, use_contrastive=True):
        self.monkey_name = monkey_name
        self.config = config or TRAIN_CONFIG
        self.monkey_config = MONKEY_CONFIGS[monkey_name]
        self.use_contrastive = use_contrastive

        # Create directories
        self.checkpoint_dir = Path(CHECKPOINT_DIR) / f'{monkey_name}_v2'
        self.log_dir = Path(LOG_DIR) / f'{monkey_name}_v2'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.contrastive_losses = []
        self.train_val_gaps = []

    def prepare_data(self):
        """Load and prepare datasets with sample-wise normalization."""
        print(f"\n{'='*60}")
        print(f"Preparing data for {self.monkey_name} (V2)")
        print(f"{'='*60}")

        # Load all training data
        train_data = load_all_training_data(self.monkey_config['train_files'])

        # Create full dataset WITH sample-wise normalization
        full_dataset = NeuroForecastDataset(
            train_data,
            is_train=True,
            augment=True,
            sample_wise_norm=True  # KEY: Use sample-wise normalization
        )

        # Save normalization statistics (for backward compatibility)
        mean, std = full_dataset.get_normalization_stats()
        stats_path = self.checkpoint_dir / f'normalization_stats.npz'
        np.savez(stats_path, mean=mean, std=std)
        print(f"Saved normalization stats to {stats_path}")

        # Split into train/val
        val_size = int(len(full_dataset) * self.config['val_split'])
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Note: sample_wise_norm applies to both train and val
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def build_model(self):
        """Build V2 model with contrastive learning support."""
        print(f"\n{'='*60}")
        print(f"Building model (V2 with contrastive learning)")
        print(f"{'='*60}")

        # Use V2 config with higher dropout
        model_config = MODEL_CONFIG_V2

        self.model = SpatioTemporalForecasterV2(
            num_channels=self.monkey_config['num_channels'],
            input_features=model_config['input_features'],
            feature_embed_dim=model_config['feature_embed_dim'],
            spatial_hidden_dim=model_config['spatial_hidden_dim'],
            spatial_num_heads=model_config['spatial_num_heads'],
            spatial_num_layers=model_config['spatial_num_layers'],
            spatial_dropout=model_config['spatial_dropout'],
            spatial_attention_dropout=model_config['spatial_attention_dropout'],
            temporal_hidden_dim=model_config['temporal_hidden_dim'],
            temporal_num_heads=model_config['temporal_num_heads'],
            temporal_num_layers=model_config['temporal_num_layers'],
            temporal_dropout=model_config['temporal_dropout'],
            forecast_hidden_dims=model_config['forecast_hidden_dims'],
            forecast_dropout=model_config['forecast_dropout'],
            input_window=INPUT_WINDOW,
            forecast_window=FORECAST_WINDOW,
            use_spectral_norm=model_config['use_spectral_norm'],
            contrastive_dim=model_config['contrastive_dim']
        ).to(DEVICE)

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

        # Loss functions
        self.mse_criterion = nn.MSELoss()
        if self.use_contrastive:
            self.contrastive_criterion = ContrastiveLoss(
                temperature=model_config['contrastive_temperature']
            )
            self.contrastive_weight = model_config['contrastive_weight']
            print(f"Contrastive learning enabled (weight={self.contrastive_weight})")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Learning rate scheduler with warmup
        warmup_epochs = self.config.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            def warmup_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (self.config['num_epochs'] - warmup_epochs)
                    return self.config['min_lr'] / self.config['learning_rate'] + \
                           (1 - self.config['min_lr'] / self.config['learning_rate']) * \
                           0.5 * (1 + np.cos(np.pi * progress))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=warmup_lambda
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=self.config['min_lr']
            )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['patience'],
            min_delta=self.config['min_delta']
        )

    def train_epoch(self):
        """Train for one epoch with contrastive learning."""
        self.model.train()
        epoch_mse_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_total_loss = 0.0
        grad_accum_steps = self.config.get('grad_accumulation_steps', 1)

        with tqdm(self.train_loader, desc=f'Epoch {self.epoch+1}/{self.config["num_epochs"]}') as pbar:
            for batch_idx, (input_seq, target_seq) in enumerate(pbar):
                # Move to device
                input_seq = input_seq.to(DEVICE)  # (B, T_in, C, F)
                target_seq = target_seq.to(DEVICE)  # (B, T_out, C)

                if self.use_contrastive:
                    # Create augmented view for contrastive learning
                    input_seq_aug = apply_augmentation(input_seq, AUG_CONFIG)

                    # Forward pass with embeddings
                    predictions, z1 = self.model(input_seq, return_embedding=True)
                    _, z2 = self.model(input_seq_aug, return_embedding=True)

                    # Compute losses
                    mse_loss = self.mse_criterion(predictions, target_seq)
                    contrastive_loss = self.contrastive_criterion(z1, z2)
                    loss = mse_loss + self.contrastive_weight * contrastive_loss

                    epoch_contrastive_loss += contrastive_loss.item()
                else:
                    # Standard forward pass
                    predictions = self.model(input_seq)
                    loss = self.mse_criterion(predictions, target_seq)
                    mse_loss = loss

                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps

                # Backward pass
                loss.backward()

                # Update weights every grad_accum_steps batches
                if (batch_idx + 1) % grad_accum_steps == 0:
                    if self.config['grad_clip'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['grad_clip']
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Update metrics
                epoch_mse_loss += mse_loss.item()
                epoch_total_loss += loss.item() * grad_accum_steps

                # Update progress bar
                pbar.set_postfix({
                    'mse': f'{mse_loss.item():.6f}',
                    'total': f'{loss.item() * grad_accum_steps:.6f}'
                })

        num_batches = len(self.train_loader)
        avg_mse_loss = epoch_mse_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        avg_contrastive_loss = epoch_contrastive_loss / num_batches if self.use_contrastive else 0

        self.train_losses.append(avg_total_loss)
        self.contrastive_losses.append(avg_contrastive_loss)

        return avg_total_loss, avg_mse_loss, avg_contrastive_loss

    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for input_seq, target_seq in self.val_loader:
                input_seq = input_seq.to(DEVICE)
                target_seq = target_seq.to(DEVICE)

                # Use standard forward pass for validation
                predictions = self.model(input_seq)
                loss = self.mse_criterion(predictions, target_seq)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)

        return avg_val_loss

    def save_checkpoint(self, val_loss, train_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'contrastive_losses': self.contrastive_losses,
            'train_val_gaps': self.train_val_gaps,
        }

        # Save latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            gap = val_loss - train_loss
            print(f"  Saved best model (val_loss: {val_loss:.6f}, gap: {gap:.6f})")

    def save_training_log(self):
        """Save training metrics."""
        log_data = {
            'monkey_name': self.monkey_name,
            'model_version': 'V2',
            'use_contrastive': self.use_contrastive,
            'num_epochs': len(self.train_losses),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'contrastive_losses': self.contrastive_losses,
            'train_val_gaps': self.train_val_gaps,
            'config': self.config,
            'model_config': MODEL_CONFIG_V2
        }

        log_path = self.log_dir / 'training_log.json'
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting V2 training for {self.monkey_name}")
        print(f"Device: {DEVICE}")
        print(f"Contrastive learning: {self.use_contrastive}")
        print(f"{'='*60}\n")

        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch

            # Train
            train_loss, mse_loss, contrastive_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Calculate train-val gap (relative, not absolute)
            # Use relative gap: (val - train) / train to compare with threshold
            train_val_gap_abs = val_loss - train_loss
            train_val_gap_rel = train_val_gap_abs / (train_loss + 1e-8)  # Relative gap
            self.train_val_gaps.append(train_val_gap_abs)

            # Learning rate schedule
            if self.scheduler is not None:
                self.scheduler.step()

            # Log
            current_lr = self.optimizer.param_groups[0]['lr']
            overfitting_threshold = self.config.get('overfitting_threshold', 0.5)
            overfitting_warning = " OVERFITTING!" if train_val_gap_rel > overfitting_threshold else ""

            contrastive_str = f"| Contrast: {contrastive_loss:.6f} " if self.use_contrastive else ""
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"MSE: {mse_loss:.6f} {contrastive_str}"
                  f"| Gap: {train_val_gap_rel*100:.1f}%{overfitting_warning} | "
                  f"LR: {current_lr:.2e}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1

            self.save_checkpoint(val_loss, train_loss, is_best=is_best)

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best model was at epoch {self.best_epoch} with val_loss: {self.best_val_loss:.6f}")
                break

            # Overfitting-based early stopping (only if severe - relative gap > 2x threshold)
            if train_val_gap_rel > overfitting_threshold * 2:
                print(f"\nSevere overfitting detected! Relative gap: {train_val_gap_rel*100:.1f}%")
                print(f"Stopping at epoch {epoch+1}")
                break

        # Save training log
        self.save_training_log()

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train V2 neural forecasting model with contrastive learning')
    parser.add_argument(
        '--monkey',
        type=str,
        required=True,
        choices=['affi', 'beignet', 'both'],
        help='Which monkey to train for'
    )
    parser.add_argument(
        '--no-contrastive',
        action='store_true',
        help='Disable contrastive learning'
    )
    args = parser.parse_args()

    monkeys = ['affi', 'beignet'] if args.monkey == 'both' else [args.monkey]
    use_contrastive = not args.no_contrastive

    for monkey_name in monkeys:
        trainer = TrainerV2(monkey_name, use_contrastive=use_contrastive)
        trainer.prepare_data()
        trainer.build_model()
        trainer.train()


if __name__ == '__main__':
    main()
