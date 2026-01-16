"""Monitor training progress."""

import torch
from pathlib import Path
import time

from config import CHECKPOINT_DIR

def monitor_training(monkey_name='beignet'):
    checkpoint_dir = Path(CHECKPOINT_DIR) / monkey_name
    latest_checkpoint = checkpoint_dir / 'checkpoint_latest.pth'

    if not latest_checkpoint.exists():
        print(f"No checkpoints found for {monkey_name}")
        return

    checkpoint = torch.load(latest_checkpoint, map_location='cpu')

    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']

    print(f"\n{'='*60}")
    print(f"Training Progress: {monkey_name.upper()}")
    print(f"{'='*60}")
    print(f"Current epoch: {epoch + 1}")
    print(f"Current train loss: {train_losses[-1]:.6f}")
    print(f"Current val loss: {val_losses[-1]:.6f}")
    print(f"Best val loss: {min(val_losses):.6f} (epoch {val_losses.index(min(val_losses)) + 1})")

    # Show last 10 epochs
    print(f"\nLast 10 epochs:")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12}")
    print("-" * 40)
    start_idx = max(0, len(train_losses) - 10)
    for i in range(start_idx, len(train_losses)):
        marker = " *" if val_losses[i] == min(val_losses) else ""
        print(f"{i+1:<8} {train_losses[i]:<12.6f} {val_losses[i]:<12.6f}{marker}")

    # Check convergence
    if len(val_losses) >= 5:
        recent_improvement = val_losses[-5] - val_losses[-1]
        print(f"\nRecent improvement (last 5 epochs): {recent_improvement:.6f}")

        if recent_improvement < 0.0001:
            print("⚠️  Model may be converging (very small improvement)")


if __name__ == '__main__':
    import sys

    monkey = sys.argv[1] if len(sys.argv) > 1 else 'beignet'
    monitor_training(monkey)
