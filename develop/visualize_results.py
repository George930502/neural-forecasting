"""Visualize training results and model predictions."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from config import LOG_DIR, CHECKPOINT_DIR


def plot_training_curves(monkey_name):
    """Plot training and validation loss curves."""
    log_file = Path(LOG_DIR) / monkey_name / 'training_log.json'

    if not log_file.exists():
        print(f"No log file found for {monkey_name}")
        return

    with open(log_file, 'r') as f:
        log_data = json.load(f)

    train_losses = log_data['train_losses']
    val_losses = log_data['val_losses']
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title(f'Training Progress - {monkey_name.capitalize()}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = Path(LOG_DIR) / monkey_name / 'training_curve.png'
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"✓ Saved training curve to {output_file}")

    # Print summary statistics
    print(f"\n{monkey_name.upper()} Training Summary:")
    print(f"  Total epochs: {len(train_losses)}")
    print(f"  Best val loss: {log_data['best_val_loss']:.6f}")
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    print(f"  Final val loss: {val_losses[-1]:.6f}")


def plot_predictions(monkey_name, num_samples=4, num_channels=4):
    """Visualize model predictions vs ground truth."""
    import torch
    from data import load_all_training_data, NeuroForecastDataset
    from models import SpatioTemporalForecaster
    from config import MONKEY_CONFIGS, MODEL_CONFIG, DEVICE, INPUT_WINDOW, FORECAST_WINDOW

    print(f"\nGenerating prediction visualizations for {monkey_name}...")

    # Load data
    monkey_config = MONKEY_CONFIGS[monkey_name]
    data = load_all_training_data(monkey_config['train_files'])

    # Load normalization stats
    stats_path = Path(CHECKPOINT_DIR) / monkey_name / 'normalization_stats.npz'
    stats = np.load(stats_path)
    mean = stats['mean']
    std = stats['std']

    # Create dataset
    dataset = NeuroForecastDataset(data, mean=mean, std=std, is_train=False, augment=False)

    # Build and load model
    model = SpatioTemporalForecaster(
        num_channels=monkey_config['num_channels'],
        input_features=MODEL_CONFIG['input_features'],
        feature_embed_dim=MODEL_CONFIG['feature_embed_dim'],
        spatial_hidden_dim=MODEL_CONFIG['spatial_hidden_dim'],
        spatial_num_heads=MODEL_CONFIG['spatial_num_heads'],
        spatial_num_layers=MODEL_CONFIG['spatial_num_layers'],
        temporal_hidden_dim=MODEL_CONFIG['temporal_hidden_dim'],
        temporal_num_heads=MODEL_CONFIG['temporal_num_heads'],
        temporal_num_layers=MODEL_CONFIG['temporal_num_layers'],
        forecast_hidden_dims=MODEL_CONFIG['forecast_hidden_dims'],
        input_window=INPUT_WINDOW,
        forecast_window=FORECAST_WINDOW,
        dropout=MODEL_CONFIG['forecast_dropout']
    ).to(DEVICE)

    checkpoint_path = Path(CHECKPOINT_DIR) / monkey_name / 'checkpoint_best.pth'
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Select random samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    # Create figure
    fig, axes = plt.subplots(num_samples, num_channels, figsize=(16, 10))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    from data.dataset import denormalize_data

    for i, sample_idx in enumerate(sample_indices):
        # Get input and ground truth
        input_seq, target_seq = dataset[sample_idx]

        # Predict
        with torch.no_grad():
            pred = model(input_seq.unsqueeze(0).to(DEVICE))
            pred = pred.squeeze(0).cpu().numpy()  # (10, C)

        # Denormalize
        pred_denorm = denormalize_data(pred, mean, std)
        target_denorm = denormalize_data(target_seq.unsqueeze(0).numpy(), mean, std).squeeze(0)

        # Get input (original scale)
        input_denorm = data[sample_idx, :INPUT_WINDOW, :, 0]  # (10, C)

        # Select channels to plot
        total_channels = pred_denorm.shape[1]
        channel_indices = np.linspace(0, total_channels-1, num_channels, dtype=int)

        for j, ch_idx in enumerate(channel_indices):
            ax = axes[i, j]

            # Plot input window
            ax.plot(range(INPUT_WINDOW), input_denorm[:, ch_idx],
                   'o-', color='blue', label='Input', markersize=4, linewidth=2)

            # Plot ground truth forecast
            ax.plot(range(INPUT_WINDOW, INPUT_WINDOW + FORECAST_WINDOW),
                   target_denorm[:, ch_idx],
                   's-', color='green', label='Ground Truth', markersize=4, linewidth=2)

            # Plot predicted forecast
            ax.plot(range(INPUT_WINDOW, INPUT_WINDOW + FORECAST_WINDOW),
                   pred_denorm[:, ch_idx],
                   '^-', color='red', label='Predicted', markersize=4, linewidth=2, alpha=0.7)

            # MSE for this channel
            mse = np.mean((pred_denorm[:, ch_idx] - target_denorm[:, ch_idx]) ** 2)

            ax.axvline(INPUT_WINDOW, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f'Ch {ch_idx}, MSE={mse:.2f}', fontsize=10)
            ax.grid(True, alpha=0.3)

            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='upper left')

            if i == num_samples - 1:
                ax.set_xlabel('Timestep', fontsize=9)
            if j == 0:
                ax.set_ylabel('Signal', fontsize=9)

    plt.suptitle(f'{monkey_name.capitalize()} - Predictions vs Ground Truth',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = Path(LOG_DIR) / monkey_name / 'predictions.png'
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"✓ Saved predictions to {output_file}")


def main():
    print("="*60)
    print("Visualizing Results")
    print("="*60)

    for monkey_name in ['beignet', 'affi']:
        log_dir = Path(LOG_DIR) / monkey_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # Plot training curves
        plot_training_curves(monkey_name)

        # Plot predictions
        try:
            plot_predictions(monkey_name, num_samples=3, num_channels=4)
        except Exception as e:
            print(f"Could not generate predictions for {monkey_name}: {e}")

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
