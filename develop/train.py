"""Training script for neural forecasting models."""

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
    DEVICE, MONKEY_CONFIGS, TRAIN_CONFIG, MODEL_CONFIG,
    CHECKPOINT_DIR, LOG_DIR, INPUT_WINDOW, FORECAST_WINDOW
)
from data import NeuroForecastDataset, load_all_training_data
from models import SpatioTemporalForecaster


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


class Trainer:
    """Training orchestrator."""

    def __init__(self, monkey_name, config=None):
        self.monkey_name = monkey_name
        self.config = config or TRAIN_CONFIG
        self.monkey_config = MONKEY_CONFIGS[monkey_name]

        # Create directories
        self.checkpoint_dir = Path(CHECKPOINT_DIR) / monkey_name
        self.log_dir = Path(LOG_DIR) / monkey_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_val_gaps = []  # Track overfitting metric

    def prepare_data(self):
        """Load and prepare datasets."""
        print(f"\n{'='*60}")
        print(f"Preparing data for {self.monkey_name}")
        print(f"{'='*60}")

        # Load all training data
        train_data = load_all_training_data(self.monkey_config['train_files'])

        # Create full dataset
        full_dataset = NeuroForecastDataset(
            train_data,
            is_train=True,
            augment=True
        )

        # Save normalization statistics
        mean, std = full_dataset.get_normalization_stats()
        stats_path = self.checkpoint_dir / f'normalization_stats.npz'
        np.savez(
            stats_path,
            mean=mean,
            std=std
        )
        print(f"Saved normalization stats to {stats_path}")

        # Split into train/val
        val_size = int(len(full_dataset) * self.config['val_split'])
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Disable augmentation for validation
        val_dataset.dataset.augment = False

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
        """Build model, optimizer, and scheduler."""
        print(f"\n{'='*60}")
        print(f"Building model")
        print(f"{'='*60}")

        self.model = SpatioTemporalForecaster(
            num_channels=self.monkey_config['num_channels'],
            input_features=MODEL_CONFIG['input_features'],
            feature_embed_dim=MODEL_CONFIG['feature_embed_dim'],
            spatial_hidden_dim=MODEL_CONFIG['spatial_hidden_dim'],
            spatial_num_heads=MODEL_CONFIG['spatial_num_heads'],
            spatial_num_layers=MODEL_CONFIG['spatial_num_layers'],
            spatial_dropout=MODEL_CONFIG['spatial_dropout'],
            spatial_attention_dropout=MODEL_CONFIG['spatial_attention_dropout'],
            temporal_hidden_dim=MODEL_CONFIG['temporal_hidden_dim'],
            temporal_num_heads=MODEL_CONFIG['temporal_num_heads'],
            temporal_num_layers=MODEL_CONFIG['temporal_num_layers'],
            temporal_dropout=MODEL_CONFIG['temporal_dropout'],
            forecast_hidden_dims=MODEL_CONFIG['forecast_hidden_dims'],
            forecast_dropout=MODEL_CONFIG['forecast_dropout'],
            input_window=INPUT_WINDOW,
            forecast_window=FORECAST_WINDOW,
            use_spectral_norm=MODEL_CONFIG['use_spectral_norm']
        ).to(DEVICE)

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

        # Loss function (MSE on forecast window)
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Learning rate scheduler with warmup
        if self.config['scheduler'] == 'cosine':
            # Create warmup scheduler
            warmup_epochs = self.config.get('warmup_epochs', 0)
            if warmup_epochs > 0:
                # Linear warmup
                def warmup_lambda(epoch):
                    if epoch < warmup_epochs:
                        return (epoch + 1) / warmup_epochs
                    else:
                        # Cosine annealing after warmup
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
        else:
            self.scheduler = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['patience'],
            min_delta=self.config['min_delta']
        )

    def train_epoch(self):
        """Train for one epoch with gradient accumulation support."""
        self.model.train()
        epoch_loss = 0.0
        grad_accum_steps = self.config.get('grad_accumulation_steps', 1)

        with tqdm(self.train_loader, desc=f'Epoch {self.epoch+1}/{self.config["num_epochs"]}') as pbar:
            for batch_idx, (input_seq, target_seq) in enumerate(pbar):
                # Move to device
                input_seq = input_seq.to(DEVICE)  # (B, T_in, C, F)
                target_seq = target_seq.to(DEVICE)  # (B, T_out, C)

                # Forward pass
                predictions = self.model(input_seq)  # (B, T_out, C)

                # Compute loss
                loss = self.criterion(predictions, target_seq)

                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps

                # Backward pass
                loss.backward()

                # Update weights every grad_accum_steps batches
                if (batch_idx + 1) % grad_accum_steps == 0:
                    # Gradient clipping
                    if self.config['grad_clip'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['grad_clip']
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Update metrics (unscaled loss for logging)
                epoch_loss += loss.item() * grad_accum_steps

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item() * grad_accum_steps:.6f}',
                    'avg_loss': f'{epoch_loss / (batch_idx + 1):.6f}'
                })

        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for input_seq, target_seq in self.val_loader:
                input_seq = input_seq.to(DEVICE)
                target_seq = target_seq.to(DEVICE)

                predictions = self.model(input_seq)
                loss = self.criterion(predictions, target_seq)

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
            print(f"✓ Saved best model at epoch {self.epoch+1} (val_loss: {val_loss:.6f}, train_loss: {train_loss:.6f}, gap: {gap:.6f})")

        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch+1}.pth'
        torch.save(checkpoint, epoch_path)

    def save_training_log(self):
        """Save training metrics."""
        log_data = {
            'monkey_name': self.monkey_name,
            'num_epochs': len(self.train_losses),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_val_gaps': self.train_val_gaps,
            'config': self.config,
            'model_config': MODEL_CONFIG
        }

        log_path = self.log_dir / 'training_log.json'
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def train(self):
        """Main training loop with overfitting detection."""
        print(f"\n{'='*60}")
        print(f"Starting training for {self.monkey_name}")
        print(f"Device: {DEVICE}")
        print(f"{'='*60}\n")

        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Calculate train-val gap (overfitting metric)
            train_val_gap = val_loss - train_loss
            self.train_val_gaps.append(train_val_gap)

            # Learning rate schedule
            if self.scheduler is not None:
                self.scheduler.step()

            # Log with overfitting detection
            current_lr = self.optimizer.param_groups[0]['lr']
            overfitting_threshold = self.config.get('overfitting_threshold', 0.15)
            overfitting_warning = " ⚠️ OVERFITTING!" if train_val_gap > overfitting_threshold else ""

            print(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Gap: {train_val_gap:.6f}{overfitting_warning} | "
                  f"LR: {current_lr:.2e}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1

            self.save_checkpoint(val_loss, train_loss, is_best=is_best)

            # Early stopping based on validation loss
            if self.early_stopping(val_loss):
                print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
                print(f"Best model was at epoch {self.best_epoch} with val_loss: {self.best_val_loss:.6f}")
                break

            # Additional overfitting-based early stopping
            if train_val_gap > overfitting_threshold * 1.5:
                print(f"\n⛔ Severe overfitting detected! Gap: {train_val_gap:.6f} > {overfitting_threshold * 1.5:.6f}")
                print(f"Stopping training at epoch {epoch+1}")
                print(f"Best model was at epoch {self.best_epoch} with val_loss: {self.best_val_loss:.6f}")
                break

        # Save training log
        self.save_training_log()

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        if len(self.train_val_gaps) > 0:
            final_gap = self.train_val_gaps[-1]
            print(f"Final train-val gap: {final_gap:.6f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train neural forecasting model')
    parser.add_argument(
        '--monkey',
        type=str,
        required=True,
        choices=['affi', 'beignet', 'both'],
        help='Which monkey to train for'
    )
    args = parser.parse_args()

    monkeys = ['affi', 'beignet'] if args.monkey == 'both' else [args.monkey]

    for monkey_name in monkeys:
        trainer = Trainer(monkey_name)
        trainer.prepare_data()
        trainer.build_model()
        trainer.train()


if __name__ == '__main__':
    main()
