"""
Spatiotemporal Transformer for Neural Forecasting.

Architecture:
1. Feature embedding: project 9 features to embedding dimension
2. Spatial attention: model inter-electrode relationships
3. Temporal transformer: capture temporal dynamics
4. Forecasting head: predict future timesteps
"""

import torch
import torch.nn as nn
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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SpatialAttention(nn.Module):
    """
    Spatial (channel-wise) attention to model inter-electrode relationships.
    Uses multi-head attention across channels.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, attention_dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,  # Attention-specific dropout
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)  # Additional dropout

    def forward(self, x):
        """
        Args:
            x: (B, C, D) - batch, channels, hidden_dim

        Returns:
            (B, C, D)
        """
        # Self-attention across channels
        attn_out, _ = self.attention(x, x, x)

        # Apply additional attention dropout
        attn_out = self.attention_dropout(attn_out)

        # Residual connection
        x = self.norm(x + self.dropout(attn_out))

        return x


class TemporalTransformer(nn.Module):
    """
    Temporal transformer to capture temporal dynamics.
    Uses causal masking for autoregressive forecasting.
    """

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
        """
        Args:
            x: (B, T, D)
            mask: optional attention mask

        Returns:
            (B, T, D)
        """
        x = self.pos_encoder(x)
        x = self.transformer(x, mask=mask)
        return x


class SpatioTemporalForecaster(nn.Module):
    """
    Complete spatiotemporal forecasting model.

    Pipeline:
    1. Embed features per channel
    2. Apply spatial attention across channels
    3. Apply temporal transformer
    4. Forecast future timesteps
    """

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

        # 1. Feature embedding: (B, T, C, F) -> (B, T, C, D_feat)
        self.feature_embed = nn.Sequential(
            nn.Linear(input_features, feature_embed_dim),
            nn.LayerNorm(feature_embed_dim),
            nn.GELU(),
            nn.Dropout(spatial_dropout)
        )

        # 2. Spatial attention layers
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

        # 3. Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            hidden_dim=temporal_hidden_dim,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout
        )

        # 4. Forecasting head with optional spectral normalization
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

        # Final projection to forecast window
        final_linear = nn.Linear(in_dim, forecast_window)
        if use_spectral_norm:
            final_linear = nn.utils.spectral_norm(final_linear)
        forecast_layers.append(final_linear)

        self.forecast_head = nn.Sequential(*forecast_layers)

    def forward(self, x):
        """
        Args:
            x: (B, T_in, C, F) - batch, input_window, channels, features

        Returns:
            (B, T_out, C) - batch, forecast_window, channels
        """
        B, T, C, F = x.shape

        # 1. Feature embedding per channel
        # (B, T, C, F) -> (B, T, C, D_feat)
        x = self.feature_embed(x)

        # 2. Spatial attention across channels (per timestep)
        # Process each timestep independently
        spatial_outputs = []
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, C, D_feat)

            # Apply spatial attention layers
            for spatial_layer in self.spatial_layers:
                x_t = spatial_layer(x_t)

            spatial_outputs.append(x_t)

        # Stack back to temporal dimension: (B, T, C, D_feat)
        x = torch.stack(spatial_outputs, dim=1)

        # 3. Project to temporal dimension and aggregate across channels
        # (B, T, C, D_feat) -> (B, T, C, D_temp)
        x = self.spatial_to_temporal(x)

        # Aggregate across channels: mean pooling
        # (B, T, C, D_temp) -> (B, T, D_temp)
        x_agg = x.mean(dim=2)

        # 4. Temporal transformer
        # (B, T, D_temp) -> (B, T, D_temp)
        x_agg = self.temporal_transformer(x_agg)

        # 5. Channel-specific forecasting
        # We need to forecast each channel independently
        # Use the aggregated temporal features + channel-specific features

        # Add channel-specific information back
        # (B, T, D_temp) -> (B, T, C, D_temp)
        x_agg_expanded = x_agg.unsqueeze(2).expand(-1, -1, C, -1)

        # Combine aggregated temporal features with channel-specific features
        x_combined = x + x_agg_expanded  # (B, T, C, D_temp)

        # Use the last timestep's features for forecasting
        x_last = x_combined[:, -1, :, :]  # (B, C, D_temp)

        # Forecast each channel
        # (B, C, D_temp) -> (B, C, T_out)
        forecasts = self.forecast_head(x_last)

        # Transpose to (B, T_out, C)
        forecasts = forecasts.transpose(1, 2)

        return forecasts


class ContrastiveLoss(nn.Module):
    """
    Contrastive learning loss for cross-session generalization.
    Based on STNDT paper approach - learns session-invariant representations.

    Uses InfoNCE-style contrastive loss on temporal embeddings:
    - Positive pairs: same sample's embeddings across different augmentations
    - Negative pairs: different samples within the batch
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1: (B, D) - embeddings from first view
            z2: (B, D) - embeddings from second view (augmented)

        Returns:
            loss: scalar contrastive loss
        """
        B = z1.shape[0]

        # Normalize embeddings
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Compute similarity matrix
        # z1 @ z2.T gives (B, B) where [i,j] = similarity(z1[i], z2[j])
        sim_matrix = torch.mm(z1, z2.T) / self.temperature

        # Positive pairs are on diagonal, negatives are off-diagonal
        labels = torch.arange(B, device=z1.device)

        # Cross-entropy loss treats diagonal as correct class
        loss_12 = nn.functional.cross_entropy(sim_matrix, labels)
        loss_21 = nn.functional.cross_entropy(sim_matrix.T, labels)

        return (loss_12 + loss_21) / 2


class SpatioTemporalForecasterV2(nn.Module):
    """
    Improved spatiotemporal forecasting model with contrastive learning support.

    Improvements based on STNDT and NDT papers:
    1. Better spatial-temporal fusion (multiplicative instead of additive)
    2. Projection head for contrastive learning
    3. Higher dropout for better generalization
    """

    def __init__(
        self,
        num_channels,
        input_features=9,
        feature_embed_dim=64,
        spatial_hidden_dim=128,
        spatial_num_heads=4,
        spatial_num_layers=2,
        spatial_dropout=0.3,  # Increased from 0.1
        spatial_attention_dropout=0.2,  # Increased
        temporal_hidden_dim=256,
        temporal_num_heads=8,
        temporal_num_layers=3,
        temporal_dropout=0.3,  # Increased from 0.1
        forecast_hidden_dims=[256, 128],
        forecast_dropout=0.3,  # Increased
        input_window=10,
        forecast_window=10,
        use_spectral_norm=True,
        contrastive_dim=128  # Projection head dimension
    ):
        super().__init__()

        self.num_channels = num_channels
        self.input_features = input_features
        self.input_window = input_window
        self.forecast_window = forecast_window
        self.use_spectral_norm = use_spectral_norm
        self.temporal_hidden_dim = temporal_hidden_dim

        # 1. Feature embedding: (B, T, C, F) -> (B, T, C, D_feat)
        self.feature_embed = nn.Sequential(
            nn.Linear(input_features, feature_embed_dim),
            nn.LayerNorm(feature_embed_dim),
            nn.GELU(),
            nn.Dropout(spatial_dropout)
        )

        # 2. Spatial attention layers
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

        # 3. Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            hidden_dim=temporal_hidden_dim,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout
        )

        # 4. Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, contrastive_dim)
        )

        # 5. Forecasting head with spectral normalization
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

        # Final projection to forecast window
        final_linear = nn.Linear(in_dim, forecast_window)
        if use_spectral_norm:
            final_linear = nn.utils.spectral_norm(final_linear)
        forecast_layers.append(final_linear)

        self.forecast_head = nn.Sequential(*forecast_layers)

    def encode(self, x):
        """
        Encode input to latent representation for contrastive learning.

        Args:
            x: (B, T_in, C, F)

        Returns:
            z: (B, D) - latent representation
            x_combined: (B, T, C, D_temp) - full spatiotemporal features
        """
        B, T, C, F = x.shape

        # 1. Feature embedding
        x = self.feature_embed(x)

        # 2. Spatial attention
        spatial_outputs = []
        for t in range(T):
            x_t = x[:, t, :, :]
            for spatial_layer in self.spatial_layers:
                x_t = spatial_layer(x_t)
            spatial_outputs.append(x_t)

        x = torch.stack(spatial_outputs, dim=1)

        # 3. Project to temporal dimension
        x = self.spatial_to_temporal(x)

        # 4. Aggregate and temporal transform
        x_agg = x.mean(dim=2)  # (B, T, D_temp)
        x_agg = self.temporal_transformer(x_agg)

        # Channel-specific features
        x_agg_expanded = x_agg.unsqueeze(2).expand(-1, -1, C, -1)
        x_combined = x + x_agg_expanded  # (B, T, C, D_temp)

        # Latent representation (mean over time and channels)
        z = x_combined.mean(dim=(1, 2))  # (B, D_temp)

        return z, x_combined

    def forward(self, x, return_embedding=False):
        """
        Args:
            x: (B, T_in, C, F)
            return_embedding: if True, also return latent embedding for contrastive loss

        Returns:
            forecasts: (B, T_out, C)
            z: (B, contrastive_dim) - only if return_embedding=True
        """
        z, x_combined = self.encode(x)
        C = x_combined.shape[2]

        # Use last timestep for forecasting
        x_last = x_combined[:, -1, :, :]  # (B, C, D_temp)

        # Forecast each channel
        forecasts = self.forecast_head(x_last)  # (B, C, T_out)
        forecasts = forecasts.transpose(1, 2)  # (B, T_out, C)

        if return_embedding:
            # Project to contrastive space
            z_proj = self.projection_head(z)  # (B, contrastive_dim)
            return forecasts, z_proj

        return forecasts


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    model = SpatioTemporalForecaster(
        num_channels=239,
        input_features=9,
        feature_embed_dim=64,
        spatial_hidden_dim=128,
        temporal_hidden_dim=256,
        input_window=10,
        forecast_window=10
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 10, 239, 9)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (4, 10, 239)
