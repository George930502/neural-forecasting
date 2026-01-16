"""Local evaluation script to test models before submission."""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from config import DEVICE, MONKEY_CONFIGS, INPUT_WINDOW, FORECAST_WINDOW, CHECKPOINT_DIR
from data import NeuroForecastDataset
from models import SpatioTemporalForecaster


def load_test_data(filepath):
    """Load test data from npz file."""
    data = np.load(filepath)['arr_0']
    return data


def load_ground_truth(filepath):
    """Load ground truth (unmasked test data)."""
    # For evaluation, we need the ground truth
    # This assumes you have access to unmasked test data locally
    data = np.load(filepath)['arr_0']
    return data[:, :, :, 0]  # Only feature 0


def calculate_mse(predictions, ground_truth):
    """
    Calculate MSE on forecast window (last 10 steps).

    Args:
        predictions: (N, T, C)
        ground_truth: (N, T, C)

    Returns:
        mse: float
    """
    # Only evaluate on forecast window (steps 10-19)
    pred_forecast = predictions[:, INPUT_WINDOW:, :]
    gt_forecast = ground_truth[:, INPUT_WINDOW:, :]

    mse = np.mean((pred_forecast - gt_forecast) ** 2)
    return mse


def evaluate_model(model, test_data, mean, std, monkey_name):
    """
    Evaluate model on test data.

    Args:
        model: trained model
        test_data: numpy array (N, T, C, F)
        mean: normalization mean
        std: normalization std
        monkey_name: 'affi' or 'beignet'

    Returns:
        predictions: numpy array (N, T, C)
    """
    model.eval()

    # Create dataset (normalize)
    dataset = NeuroForecastDataset(
        test_data,
        mean=mean,
        std=std,
        is_train=False,
        augment=False
    )

    # Predict
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f'Predicting {monkey_name}'):
            input_seq, _ = dataset[i]
            input_seq = input_seq.unsqueeze(0).to(DEVICE)  # (1, T_in, C, F)

            # Predict
            pred = model(input_seq)  # (1, T_out, C)
            pred = pred.squeeze(0).cpu().numpy()  # (T_out, C)

            predictions.append(pred)

    predictions = np.array(predictions)  # (N, T_out, C)

    # Denormalize predictions
    from data.dataset import denormalize_data
    predictions = denormalize_data(predictions, mean, std)

    # Combine with input window for complete sequence
    # Use the first 10 steps from original data (feature 0)
    input_steps = test_data[:, :INPUT_WINDOW, :, 0]  # (N, 10, C)

    # Concatenate: (N, 10, C) + (N, 10, C) = (N, 20, C)
    full_predictions = np.concatenate([input_steps, predictions], axis=1)

    return full_predictions


def main():
    print("="*60)
    print("Local Evaluation")
    print("="*60)

    # Test datasets (for evaluation, you need ground truth)
    # For now, we'll use training data as a sanity check
    # In production, replace with actual test data + ground truth

    results = {}

    for monkey_name in ['affi', 'beignet']:
        print(f"\n{'='*60}")
        print(f"Evaluating {monkey_name.upper()}")
        print(f"{'='*60}")

        monkey_config = MONKEY_CONFIGS[monkey_name]

        # Load model checkpoint
        checkpoint_dir = Path(CHECKPOINT_DIR) / monkey_name
        checkpoint_path = checkpoint_dir / 'checkpoint_best.pth'

        if not checkpoint_path.exists():
            print(f"Warning: No checkpoint found at {checkpoint_path}")
            print("Skipping evaluation for this monkey.")
            continue

        # Load normalization stats
        stats_path = checkpoint_dir / 'normalization_stats.npz'
        stats = np.load(stats_path)
        mean = stats['mean']
        std = stats['std']

        # Build model
        from config import MODEL_CONFIG
        model = SpatioTemporalForecaster(
            num_channels=monkey_config['num_channels'],
            input_features=MODEL_CONFIG['input_features'],
            feature_embed_dim=MODEL_CONFIG['feature_embed_dim'],
            spatial_hidden_dim=MODEL_CONFIG['spatial_hidden_dim'],
            spatial_num_heads=MODEL_CONFIG['spatial_num_heads'],
            spatial_num_layers=MODEL_CONFIG['spatial_num_layers'],
            spatial_dropout=MODEL_CONFIG['spatial_dropout'],
            spatial_attention_dropout=MODEL_CONFIG.get('spatial_attention_dropout', 0.1),
            temporal_hidden_dim=MODEL_CONFIG['temporal_hidden_dim'],
            temporal_num_heads=MODEL_CONFIG['temporal_num_heads'],
            temporal_num_layers=MODEL_CONFIG['temporal_num_layers'],
            temporal_dropout=MODEL_CONFIG['temporal_dropout'],
            forecast_hidden_dims=MODEL_CONFIG['forecast_hidden_dims'],
            input_window=INPUT_WINDOW,
            forecast_window=FORECAST_WINDOW,
            forecast_dropout=MODEL_CONFIG['forecast_dropout'],
            use_spectral_norm=MODEL_CONFIG.get('use_spectral_norm', False)
        ).to(DEVICE)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")

        # Evaluate on test files
        monkey_mse = []

        for test_file in monkey_config['test_files']:
            test_name = Path(test_file).stem

            # Load test data
            test_data = load_test_data(test_file)
            print(f"\nTest file: {test_name}")
            print(f"Shape: {test_data.shape}")

            # Predict
            predictions = evaluate_model(model, test_data, mean, std, monkey_name)

            print(f"Predictions shape: {predictions.shape}")
            print(f"Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")

            # For sanity check, compute MSE on training data
            # (In real evaluation, you need ground truth test data)
            if 'train' in test_file:
                # If evaluating on train data, we have ground truth
                ground_truth = test_data[:, :, :, 0]
                mse = calculate_mse(predictions, ground_truth)
                monkey_mse.append(mse)
                print(f"MSE (forecast window): {mse:.6f}")

        if monkey_mse:
            avg_mse = np.mean(monkey_mse)
            results[monkey_name] = avg_mse
            print(f"\nAverage MSE for {monkey_name}: {avg_mse:.6f}")

    # Overall results
    if results:
        print(f"\n{'='*60}")
        print("Overall Results")
        print(f"{'='*60}")
        for monkey_name, mse in results.items():
            print(f"{monkey_name}: {mse:.6f}")

        total_mse = np.mean(list(results.values()))
        print(f"\nTotal Average MSE: {total_mse:.6f}")


if __name__ == '__main__':
    main()
