"""Dataset and preprocessing for neural forecasting."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    NUM_TIMESTEPS, INPUT_WINDOW, FORECAST_WINDOW,
    NUM_FEATURES, NORM_CONFIG, AUG_CONFIG
)
from scipy.interpolate import CubicSpline


def compute_normalization_stats(data, method='robust', clip_std=4.0, eps=1e-8):
    """
    Compute robust normalization statistics across all sessions.

    Args:
        data: numpy array of shape (N, T, C, F)
        method: 'robust' or 'standard'
        clip_std: number of std deviations for clipping
        eps: epsilon for numerical stability

    Returns:
        mean, std: normalization statistics of shape (1, 1, C, F)
    """
    N, T, C, F = data.shape

    if method == 'robust':
        # Use median and IQR for robustness to outliers
        # Compute per-channel, per-feature statistics across N*T
        data_reshaped = data.reshape(N * T, C, F)

        # Median as center
        median = np.median(data_reshaped, axis=0, keepdims=True)  # (1, C, F)

        # IQR-based scale (more robust than std)
        q75 = np.percentile(data_reshaped, 75, axis=0, keepdims=True)
        q25 = np.percentile(data_reshaped, 25, axis=0, keepdims=True)
        iqr = q75 - q25 + eps

        # Convert IQR to equivalent std (IQR ≈ 1.35 * std for normal)
        scale = iqr / 1.35

        mean = median
        std = scale

    else:  # standard
        # Standard mean and std
        data_reshaped = data.reshape(N * T, C, F)
        mean = np.mean(data_reshaped, axis=0, keepdims=True)
        std = np.std(data_reshaped, axis=0, keepdims=True) + eps

    # Reshape to (1, 1, C, F) for broadcasting
    mean = mean.reshape(1, 1, C, F)
    std = std.reshape(1, 1, C, F)

    return mean, std


def compute_sample_wise_stats(data, eps=1e-8):
    """
    Compute normalization statistics per-sample for test-time adaptation.

    Args:
        data: numpy array of shape (N, T, C, F) or (T, C, F)
        eps: epsilon for numerical stability

    Returns:
        mean, std: arrays matching input shape for broadcasting
    """
    if data.ndim == 3:
        # Single sample (T, C, F) - compute over time
        median = np.median(data, axis=0, keepdims=True)  # (1, C, F)
        q75 = np.percentile(data, 75, axis=0, keepdims=True)
        q25 = np.percentile(data, 25, axis=0, keepdims=True)
        iqr = q75 - q25 + eps
        scale = iqr / 1.35
        return median, scale
    else:
        # Batch (N, T, C, F) - compute per sample over time
        N, T, C, F = data.shape
        median = np.median(data, axis=1, keepdims=True)  # (N, 1, C, F)
        q75 = np.percentile(data, 75, axis=1, keepdims=True)
        q25 = np.percentile(data, 25, axis=1, keepdims=True)
        iqr = q75 - q25 + eps
        scale = iqr / 1.35
        return median, scale


def normalize_data(data, mean, std, clip_std=4.0):
    """
    Normalize data using precomputed statistics.

    Args:
        data: numpy array of shape (N, T, C, F)
        mean: mean of shape (1, 1, C, F)
        std: std of shape (1, 1, C, F)
        clip_std: clip to mean ± clip_std * std

    Returns:
        normalized_data: z-scored and clipped data
    """
    # Z-score normalization
    normalized = (data - mean) / std

    # Clip to reasonable range
    normalized = np.clip(normalized, -clip_std, clip_std)

    return normalized


def denormalize_data(normalized_data, mean, std):
    """
    Denormalize data back to original scale.

    Args:
        normalized_data: normalized array of shape (N, T, C)
        mean: mean of shape (1, 1, C, F) - use only feature 0
        std: std of shape (1, 1, C, F) - use only feature 0

    Returns:
        denormalized_data: original scale data
    """
    # Extract feature 0 statistics
    mean_f0 = mean[..., 0]  # (1, 1, C)
    std_f0 = std[..., 0]    # (1, 1, C)

    # Denormalize
    denormalized = normalized_data * std_f0 + mean_f0

    return denormalized


def load_all_training_data(file_paths):
    """
    Load and concatenate all training data files.

    Args:
        file_paths: list of paths to .npz files

    Returns:
        combined_data: numpy array of shape (N_total, T, C, F)
    """
    data_arrays = []

    for path in file_paths:
        data = np.load(path)['arr_0']
        data_arrays.append(data)
        print(f"Loaded {path}: {data.shape}")

    combined = np.concatenate(data_arrays, axis=0)
    print(f"Combined training data: {combined.shape}")

    return combined


def apply_channel_masking(data, mask_prob=0.1):
    """
    Randomly mask some channels during training.

    Args:
        data: numpy array of shape (T, C, F)
        mask_prob: probability of masking each channel

    Returns:
        masked_data: data with some channels masked (set to 0)
    """
    T, C, F = data.shape
    mask = np.random.rand(C) > mask_prob  # True for channels to keep
    masked_data = data.copy()
    masked_data[:, ~mask, :] = 0  # Mask out selected channels
    return masked_data


def apply_time_warping(data, sigma=0.2):
    """
    Apply time warping augmentation using cubic spline interpolation.

    Args:
        data: numpy array of shape (T, C, F)
        sigma: strength of warping (std of random knot perturbations)

    Returns:
        warped_data: time-warped data
    """
    T, C, F = data.shape

    # Create original time points
    orig_time = np.arange(T)

    # Create warped time points with random perturbations
    # Add smooth random warping using cumulative sum of small perturbations
    perturbations = np.random.randn(T) * sigma
    perturbations = np.cumsum(perturbations)
    perturbations = perturbations - perturbations.mean()  # Center
    perturbations = perturbations / (perturbations.std() + 1e-8) * sigma  # Normalize scale

    # Warped time should still be monotonic and within bounds
    warped_time = orig_time + perturbations
    warped_time = np.clip(warped_time, 0, T - 1)

    # Apply cubic spline interpolation for each channel and feature
    warped_data = np.zeros_like(data)
    for c in range(C):
        for f in range(F):
            cs = CubicSpline(orig_time, data[:, c, f])
            warped_data[:, c, f] = cs(warped_time)

    return warped_data


def apply_mixup(data1, data2, alpha=0.2):
    """
    Apply mixup augmentation between two samples.

    Args:
        data1: first sample of shape (T, C, F)
        data2: second sample of shape (T, C, F)
        alpha: mixup parameter (beta distribution parameter)

    Returns:
        mixed_data: mixed sample
        lambda_: mixing coefficient
    """
    if alpha > 0:
        lambda_ = np.random.beta(alpha, alpha)
    else:
        lambda_ = 1.0

    mixed_data = lambda_ * data1 + (1 - lambda_) * data2
    return mixed_data, lambda_


class NeuroForecastDataset(Dataset):
    """
    PyTorch Dataset for neural forecasting.

    Input: First 10 timesteps with all 9 features
    Target: Last 10 timesteps with only feature 0
    """

    def __init__(
        self,
        data,
        mean=None,
        std=None,
        is_train=True,
        augment=False,
        sample_wise_norm=True  # NEW: Enable sample-wise normalization by default
    ):
        """
        Args:
            data: numpy array of shape (N, T, C, F)
            mean: normalization mean (if None, compute from data)
            std: normalization std (if None, compute from data)
            is_train: whether this is training data
            augment: whether to apply data augmentation
            sample_wise_norm: if True, normalize each sample independently (for cross-session generalization)
        """
        self.is_train = is_train
        self.augment = augment and is_train
        self.sample_wise_norm = sample_wise_norm

        # Store raw data for sample-wise normalization
        self.raw_data = data.copy()

        # Compute or use provided normalization stats (for backward compatibility)
        if mean is None or std is None:
            self.mean, self.std = compute_normalization_stats(
                data,
                method=NORM_CONFIG['method'],
                clip_std=NORM_CONFIG['clip_std'],
                eps=NORM_CONFIG['eps']
            )
        else:
            self.mean = mean
            self.std = std

        if sample_wise_norm:
            # For sample-wise normalization, we normalize each sample on-the-fly in __getitem__
            # Store raw data instead of pre-normalized data
            self.data = data
        else:
            # Global normalization (original behavior)
            self.data = normalize_data(
                data,
                self.mean,
                self.std,
                clip_std=NORM_CONFIG['clip_std']
            )

        self.N, self.T, self.C, self.F = self.data.shape

        assert self.T == NUM_TIMESTEPS, f"Expected {NUM_TIMESTEPS} timesteps, got {self.T}"
        assert self.F == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {self.F}"

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
        Returns:
            input_seq: tensor of shape (T_input, C, F) - first 10 steps, all features
            target_seq: tensor of shape (T_forecast, C) - last 10 steps, feature 0 only
        """
        sample = self.data[idx]  # (T, C, F)

        # Apply sample-wise normalization if enabled
        if self.sample_wise_norm:
            # Compute per-sample statistics
            sample_mean, sample_std = compute_sample_wise_stats(sample, eps=NORM_CONFIG['eps'])
            # Normalize this sample
            sample = normalize_data(
                sample,
                sample_mean,
                sample_std,
                clip_std=NORM_CONFIG['clip_std']
            )

        # Split into input and target
        input_seq = sample[:INPUT_WINDOW, :, :].copy()  # (10, C, F)
        target_seq = sample[INPUT_WINDOW:, :, 0].copy()  # (10, C) - feature 0 only

        # Data augmentation
        if self.augment and AUG_CONFIG['enabled']:
            # 1. Add Gaussian noise
            if AUG_CONFIG['noise_std'] > 0:
                noise = np.random.normal(
                    0,
                    AUG_CONFIG['noise_std'],
                    input_seq.shape
                )
                input_seq = input_seq + noise

            # 2. Channel masking (randomly drop some channels)
            if AUG_CONFIG.get('channel_masking_prob', 0) > 0:
                if np.random.rand() < 0.5:  # Apply 50% of the time
                    input_seq = apply_channel_masking(
                        input_seq,
                        mask_prob=AUG_CONFIG['channel_masking_prob']
                    )

            # 3. Time warping (temporal distortion)
            if AUG_CONFIG.get('time_warping_sigma', 0) > 0:
                if np.random.rand() < 0.3:  # Apply 30% of the time
                    input_seq = apply_time_warping(
                        input_seq,
                        sigma=AUG_CONFIG['time_warping_sigma']
                    )

            # 4. Mixup (blend with another random sample)
            if AUG_CONFIG.get('mixup_alpha', 0) > 0:
                if np.random.rand() < 0.2:  # Apply 20% of the time
                    # Get another random sample
                    other_idx = np.random.randint(0, self.N)
                    other_sample = self.data[other_idx]
                    other_input = other_sample[:INPUT_WINDOW, :, :]
                    other_target = other_sample[INPUT_WINDOW:, :, 0]

                    # Apply mixup
                    input_seq, lambda_ = apply_mixup(
                        input_seq,
                        other_input,
                        alpha=AUG_CONFIG['mixup_alpha']
                    )
                    target_seq = lambda_ * target_seq + (1 - lambda_) * other_target

            # 5. Temporal shift (small random shift in time)
            if AUG_CONFIG.get('temporal_shift', 0) > 0:
                shift = np.random.randint(
                    -AUG_CONFIG['temporal_shift'],
                    AUG_CONFIG['temporal_shift'] + 1
                )
                if shift != 0:
                    input_seq = np.roll(input_seq, shift, axis=0)

        # Convert to tensors
        input_seq = torch.from_numpy(input_seq).float()
        target_seq = torch.from_numpy(target_seq).float()

        return input_seq, target_seq

    def get_normalization_stats(self):
        """Return normalization statistics for saving."""
        return self.mean, self.std
