"""Create Codabench-compatible submission package."""

import os
import sys
import shutil
import numpy as np
import torch
from pathlib import Path

from config import (
    DEVICE, MONKEY_CONFIGS, MODEL_CONFIG, CHECKPOINT_DIR,
    SUBMISSION_DIR, INPUT_WINDOW, FORECAST_WINDOW
)
from models import SpatioTemporalForecaster


def create_submission_model_file():
    """Create the model.py file for submission."""

    model_code = '''"""
Codabench submission model for neural forecasting.
This file contains the Model class that will be used for evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SpatialAttention(nn.Module):
    """Spatial (channel-wise) attention."""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class TemporalTransformer(nn.Module):
    """Temporal transformer."""

    def __init__(self, hidden_dim, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

    def forward(self, x, mask=None):
        x = self.pos_encoder(x)
        x = self.transformer(x, mask=mask)
        return x


class SpatioTemporalForecaster(nn.Module):
    """Spatiotemporal forecasting model."""

    def __init__(
        self,
        num_channels,
        input_features=9,
        feature_embed_dim=64,
        spatial_hidden_dim=128,
        spatial_num_heads=4,
        spatial_num_layers=2,
        temporal_hidden_dim=256,
        temporal_num_heads=8,
        temporal_num_layers=3,
        forecast_hidden_dims=[256, 128],
        input_window=10,
        forecast_window=10,
        dropout=0.1
    ):
        super().__init__()

        self.num_channels = num_channels
        self.input_features = input_features
        self.input_window = input_window
        self.forecast_window = forecast_window

        # Feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(input_features, feature_embed_dim),
            nn.LayerNorm(feature_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Spatial attention layers
        self.spatial_layers = nn.ModuleList([
            SpatialAttention(
                feature_embed_dim,
                num_heads=spatial_num_heads,
                dropout=dropout
            )
            for _ in range(spatial_num_layers)
        ])

        self.spatial_to_temporal = nn.Linear(feature_embed_dim, temporal_hidden_dim)

        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            hidden_dim=temporal_hidden_dim,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
            dropout=dropout
        )

        # Forecasting head
        forecast_layers = []
        in_dim = temporal_hidden_dim

        for hidden_dim in forecast_hidden_dims:
            forecast_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        forecast_layers.append(nn.Linear(in_dim, forecast_window))
        self.forecast_head = nn.Sequential(*forecast_layers)

    def forward(self, x):
        B, T, C, F = x.shape

        # Feature embedding
        x = self.feature_embed(x)

        # Spatial attention
        spatial_outputs = []
        for t in range(T):
            x_t = x[:, t, :, :]
            for spatial_layer in self.spatial_layers:
                x_t = spatial_layer(x_t)
            spatial_outputs.append(x_t)

        x = torch.stack(spatial_outputs, dim=1)

        # Temporal modeling
        x = self.spatial_to_temporal(x)
        x_agg = x.mean(dim=2)
        x_agg = self.temporal_transformer(x_agg)

        # Forecasting
        x_agg_expanded = x_agg.unsqueeze(2).expand(-1, -1, C, -1)
        x_combined = x + x_agg_expanded
        x_last = x_combined[:, -1, :, :]
        forecasts = self.forecast_head(x_last)
        forecasts = forecasts.transpose(1, 2)

        return forecasts


class Model:
    """Codabench submission interface."""

    def __init__(self, monkey_name='beignet'):
        self.monkey_name = monkey_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configuration
        if monkey_name == 'affi':
            self.num_channels = 239
        elif monkey_name == 'beignet':
            self.num_channels = 89
        else:
            raise ValueError(f'Unknown monkey: {monkey_name}')

        # Build model
        self.model = SpatioTemporalForecaster(
            num_channels=self.num_channels,
            input_features=9,
            feature_embed_dim=64,
            spatial_hidden_dim=128,
            spatial_num_heads=4,
            spatial_num_layers=2,
            temporal_hidden_dim=256,
            temporal_num_heads=8,
            temporal_num_layers=3,
            forecast_hidden_dims=[256, 128],
            input_window=10,
            forecast_window=10,
            dropout=0.1
        ).to(self.device)

        # Load normalization stats
        base_dir = os.path.dirname(__file__)
        stats_path = os.path.join(
            base_dir,
            f'normalization_stats_{monkey_name}.npz'
        )

        stats = np.load(stats_path)
        self.mean = stats['mean']
        self.std = stats['std']

    def load(self):
        """Load trained weights."""
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(
            base_dir,
            f'model_{self.monkey_name}.pth'
        )

        state_dict = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def normalize(self, data):
        """Normalize data."""
        normalized = (data - self.mean) / self.std
        normalized = np.clip(normalized, -4.0, 4.0)
        return normalized

    def denormalize(self, normalized_data):
        """Denormalize predictions."""
        mean_f0 = self.mean[..., 0]
        std_f0 = self.std[..., 0]
        denormalized = normalized_data * std_f0 + mean_f0
        return denormalized

    def predict(self, x):
        """
        Make predictions.

        Args:
            x: numpy array of shape (N, 20, C, F)

        Returns:
            predictions: numpy array of shape (N, 20, C)
        """
        # Normalize
        x_norm = self.normalize(x)

        # Get input window
        input_window = x_norm[:, :10, :, :]  # (N, 10, C, F)

        # Convert to tensor
        input_tensor = torch.from_numpy(input_window).float().to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_tensor)  # (N, 10, C)
            predictions = predictions.cpu().numpy()

        # Denormalize
        predictions = self.denormalize(predictions)

        # Combine with input window
        input_steps = x[:, :10, :, 0]  # (N, 10, C)
        full_predictions = np.concatenate([input_steps, predictions], axis=1)

        return full_predictions
'''

    return model_code


def main():
    print("="*60)
    print("Creating Codabench Submission Package")
    print("="*60)

    # Create submission directory
    submission_dir = Path(SUBMISSION_DIR)
    submission_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSubmission directory: {submission_dir}")

    # Create model.py
    print("\n1. Creating model.py...")
    model_code = create_submission_model_file()
    model_file = submission_dir / 'model.py'
    with open(model_file, 'w') as f:
        f.write(model_code)
    print(f"   ✓ Created {model_file}")

    # Copy trained weights for each monkey
    for monkey_name in ['affi', 'beignet']:
        print(f"\n2. Processing {monkey_name}...")

        checkpoint_dir = Path(CHECKPOINT_DIR) / monkey_name

        # Load best checkpoint
        checkpoint_path = checkpoint_dir / 'checkpoint_best.pth'
        if not checkpoint_path.exists():
            print(f"   Warning: No checkpoint found at {checkpoint_path}")
            print(f"   Skipping {monkey_name}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Save model weights
        weights_file = submission_dir / f'model_{monkey_name}.pth'
        torch.save(checkpoint['model_state_dict'], weights_file)
        print(f"   ✓ Saved weights to {weights_file}")

        # Copy normalization stats
        stats_path = checkpoint_dir / 'normalization_stats.npz'
        if stats_path.exists():
            dest_stats = submission_dir / f'normalization_stats_{monkey_name}.npz'
            shutil.copy(stats_path, dest_stats)
            print(f"   ✓ Copied normalization stats to {dest_stats}")

    # Create requirements.txt
    print("\n3. Creating requirements.txt...")
    requirements = """torch>=2.0.0
numpy>=1.24.0
"""
    requirements_file = submission_dir / 'requirements.txt'
    with open(requirements_file, 'w') as f:
        f.write(requirements)
    print(f"   ✓ Created {requirements_file}")

    print("\n" + "="*60)
    print("Submission package created successfully!")
    print(f"Location: {submission_dir}")
    print("="*60)

    # List files
    print("\nPackage contents:")
    for file in sorted(submission_dir.iterdir()):
        size = file.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {file.name} ({size:.2f} MB)")

    print("\nNext steps:")
    print("1. Test the submission locally using the ingestion/scoring scripts")
    print("2. Zip the submission directory")
    print("3. Upload to Codabench")


if __name__ == '__main__':
    main()
