"""
Submission model for HDR Challenge Year 2 - Neural Forecasting.
Spatiotemporal Transformer for predicting neural activity from uECoG arrays.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


# Model Configuration
MODEL_CONFIG = {
    'input_features': 9,
    'feature_embed_dim': 64,
    'spatial_hidden_dim': 128,
    'spatial_num_heads': 4,
    'spatial_num_layers': 2,
    'spatial_dropout': 0.2,
    'spatial_attention_dropout': 0.15,
    'temporal_hidden_dim': 256,
    'temporal_num_heads': 8,
    'temporal_num_layers': 3,
    'temporal_dropout': 0.2,
    'forecast_hidden_dims': [256, 128],
    'forecast_dropout': 0.25,
    'use_spectral_norm': True,
}


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
    """Spatial (channel-wise) attention to model inter-electrode relationships."""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, attention_dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.attention_dropout(attn_out)
        x = self.norm(x + self.dropout(attn_out))
        return x


class TemporalTransformer(nn.Module):
    """Temporal transformer to capture temporal dynamics."""

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
    """Complete spatiotemporal forecasting model."""

    def __init__(
        self,
        num_channels,
        input_features=9,
        feature_embed_dim=64,
        spatial_hidden_dim=128,
        spatial_num_heads=4,
        spatial_num_layers=2,
        spatial_dropout=0.1,
        spatial_attention_dropout=0.1,
        temporal_hidden_dim=256,
        temporal_num_heads=8,
        temporal_num_layers=3,
        temporal_dropout=0.1,
        forecast_hidden_dims=[256, 128],
        forecast_dropout=0.1,
        input_window=10,
        forecast_window=10,
        use_spectral_norm=False
    ):
        super().__init__()

        self.num_channels = num_channels
        self.input_features = input_features
        self.input_window = input_window
        self.forecast_window = forecast_window
        self.use_spectral_norm = use_spectral_norm

        # Feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(input_features, feature_embed_dim),
            nn.LayerNorm(feature_embed_dim),
            nn.GELU(),
            nn.Dropout(spatial_dropout)
        )

        # Spatial attention layers
        self.spatial_layers = nn.ModuleList([
            SpatialAttention(
                feature_embed_dim,
                num_heads=spatial_num_heads,
                dropout=spatial_dropout,
                attention_dropout=spatial_attention_dropout
            )
            for _ in range(spatial_num_layers)
        ])

        # Project to temporal dimension
        self.spatial_to_temporal = nn.Linear(feature_embed_dim, temporal_hidden_dim)

        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            hidden_dim=temporal_hidden_dim,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout
        )

        # Forecasting head with optional spectral normalization
        forecast_layers = []
        in_dim = temporal_hidden_dim

        for hidden_dim in forecast_hidden_dims:
            linear = nn.Linear(in_dim, hidden_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)

            forecast_layers.extend([
                linear,
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(forecast_dropout)
            ])
            in_dim = hidden_dim

        final_linear = nn.Linear(in_dim, forecast_window)
        if use_spectral_norm:
            final_linear = nn.utils.spectral_norm(final_linear)
        forecast_layers.append(final_linear)

        self.forecast_head = nn.Sequential(*forecast_layers)

    def forward(self, x):
        B, T, C, F = x.shape

        # Feature embedding per channel
        x = self.feature_embed(x)

        # Spatial attention across channels (per timestep)
        spatial_outputs = []
        for t in range(T):
            x_t = x[:, t, :, :]

            for spatial_layer in self.spatial_layers:
                x_t = spatial_layer(x_t)

            spatial_outputs.append(x_t)

        x = torch.stack(spatial_outputs, dim=1)

        # Project to temporal dimension
        x = self.spatial_to_temporal(x)

        # Aggregate across channels
        x_agg = x.mean(dim=2)

        # Temporal transformer
        x_agg = self.temporal_transformer(x_agg)

        # Channel-specific forecasting
        x_agg_expanded = x_agg.unsqueeze(2).expand(-1, -1, C, -1)
        x_combined = x + x_agg_expanded

        # Use last timestep for forecasting
        x_last = x_combined[:, -1, :, :]
        forecasts = self.forecast_head(x_last)
        forecasts = forecasts.transpose(1, 2)

        return forecasts


def normalize_data(data, mean, std, clip_std=4.0, eps=1e-8):
    """Normalize data using robust normalization."""
    std_safe = np.where(std < eps, eps, std)
    normalized = (data - mean) / std_safe
    normalized = np.clip(normalized, -clip_std, clip_std)
    return normalized


def denormalize_data(normalized_data, mean, std):
    """Denormalize data back to original scale."""
    mean_f0 = mean[..., 0]
    std_f0 = std[..., 0]
    denormalized = normalized_data * std_f0 + mean_f0
    return denormalized


class Model(torch.nn.Module):
    """Wrapper class for submission with required interface."""

    def __init__(self, monkey_name='beignet'):
        super(Model, self).__init__()
        self.monkey_name = monkey_name

        if self.monkey_name == 'beignet':
            self.num_channels = 89
        elif self.monkey_name == 'affi':
            self.num_channels = 239
        else:
            raise ValueError(f'No such a monkey: {self.monkey_name}')

        # Build the model
        self.model = SpatioTemporalForecaster(
            num_channels=self.num_channels,
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
            input_window=10,
            forecast_window=10,
            use_spectral_norm=MODEL_CONFIG['use_spectral_norm']
        )

        # Load normalization stats
        base = os.path.dirname(__file__)
        try:
            stats_path = os.path.join(base, f'normalization_stats_{self.monkey_name}.npz')
            stats = np.load(stats_path)
            self.mean = stats['mean']
            self.std = stats['std']
        except FileNotFoundError:
            print(f"Warning: normalization_stats_{self.monkey_name}.npz not found.")
            self.mean = None
            self.std = None

    def forward(self, x):
        return self.model(x)

    def load(self):
        base = os.path.dirname(__file__)

        if self.monkey_name == 'beignet':
            path = os.path.join(base, "model_beignet.pth")
        elif self.monkey_name == 'affi':
            path = os.path.join(base, "model_affi.pth")
        else:
            raise ValueError(f'No such a monkey: {self.monkey_name}')

        state_dict = torch.load(
            path,
            map_location=torch.device(device),
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)

        # Move model to device (GPU if available)
        self.model.to(device)

    def predict(self, x):
        """
        Make predictions on input data.

        Args:
            x: numpy array of shape (N, T, C, F) - samples, timesteps, channels, features

        Returns:
            predictions: numpy array of shape (N, T, C) - full sequence predictions (feature 0)
        """
        # Normalize input data
        if self.mean is not None and self.std is not None:
            x_normalized = normalize_data(x, self.mean, self.std)
        else:
            x_normalized = x

        # Convert to tensor
        x_tensor = torch.tensor(x_normalized, dtype=torch.float32)

        # Get input window (first 10 timesteps)
        input_window = x_tensor[:, :10, :, :]

        # Make predictions
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(len(input_window)):
                sample = input_window[i:i+1].to(device)  # (1, 10, C, F)
                output = self.model(sample)  # (1, 10, C)
                predictions.append(output.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)  # (N, 10, C)

        # Denormalize predictions
        if self.mean is not None and self.std is not None:
            predictions = denormalize_data(predictions, self.mean, self.std)

        # Combine with input window for complete sequence
        # Use original (non-normalized) data for input portion, feature 0 only
        input_steps = x[:, :10, :, 0]  # (N, 10, C)

        # Concatenate: (N, 10, C) + (N, 10, C) = (N, 20, C)
        full_predictions = np.concatenate([input_steps, predictions], axis=1)

        return full_predictions
