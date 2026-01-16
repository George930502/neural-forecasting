"""Configuration for neural forecasting pipeline."""

import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data configuration
MONKEY_CONFIGS = {
    'affi': {
        'num_channels': 239,
        'train_files': [
            'dataset/train/train_data_affi.npz',
            'dataset/train/train_data_affi_2024-03-20_private.npz'
        ],
        'test_files': [
            'dataset/test/test_data_affi_masked.npz',
            'dataset/test/test_data_affi_2024-03-20_private_masked.npz'
        ]
    },
    'beignet': {
        'num_channels': 89,
        'train_files': [
            'dataset/train/train_data_beignet.npz',
            'dataset/train/train_data_beignet_2022-06-01_private.npz',
            'dataset/train/train_data_beignet_2022-06-02_private.npz'
        ],
        'test_files': [
            'dataset/test/test_data_beignet_masked.npz',
            'dataset/test/test_data_beignet_2022-06-01_private_masked.npz',
            'dataset/test/test_data_beignet_2022-06-02_private_masked.npz'
        ]
    }
}

# Data preprocessing
NUM_TIMESTEPS = 20
INPUT_WINDOW = 10  # First 10 timesteps
FORECAST_WINDOW = 10  # Last 10 timesteps
NUM_FEATURES = 9  # 9 frequency band features

# Model architecture
MODEL_CONFIG = {
    # Input/output
    'input_features': NUM_FEATURES,
    'output_features': 1,  # Predict only feature 0

    # Feature embedding
    'feature_embed_dim': 64,

    # Spatial (channel) attention
    'spatial_hidden_dim': 128,
    'spatial_num_heads': 4,
    'spatial_num_layers': 2,
    'spatial_dropout': 0.3,  # Increased for better generalization (STNDT/NDT guidance)
    'spatial_attention_dropout': 0.2,  # Increased for regularization

    # Temporal modeling
    'temporal_hidden_dim': 256,
    'temporal_num_heads': 8,
    'temporal_num_layers': 3,
    'temporal_dropout': 0.3,  # Increased for better generalization

    # Forecasting head
    'forecast_hidden_dims': [256, 128],
    'forecast_dropout': 0.35,  # Increased for better generalization
    'use_spectral_norm': True,  # Spectral normalization for forecast head

    # Contrastive learning (for V2 model)
    'contrastive_dim': 128,  # Projection head dimension
    'contrastive_weight': 0.1,  # Weight for contrastive loss
    'contrastive_temperature': 0.1,  # Temperature for contrastive loss
}

# V2 Model with higher dropout for cross-session generalization
MODEL_CONFIG_V2 = {
    **MODEL_CONFIG,
    'spatial_dropout': 0.4,  # Higher dropout based on NDT paper (0.2-0.6 range)
    'spatial_attention_dropout': 0.3,
    'temporal_dropout': 0.4,
    'forecast_dropout': 0.4,
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 5e-4,  # Reduced from 1e-3 per paper insights (Transformers sensitive to LR)
    'weight_decay': 1e-3,  # Increased from 1e-4 for stronger L2 regularization
    'grad_clip': 1.0,
    'grad_accumulation_steps': 1,  # New: gradient accumulation for stable training

    # Learning rate schedule
    'scheduler': 'cosine',
    'warmup_epochs': 10,  # Increased from 5 for more gradual warmup
    'min_lr': 1e-6,

    # Early stopping
    'patience': 15,  # Increased from 10 to allow more training
    'min_delta': 1e-4,
    'overfitting_threshold': 0.5,  # Relaxed from 0.15 to allow longer training

    # Validation
    'val_split': 0.15,
    'val_check_interval': 1,  # epochs

    # Checkpointing
    'save_top_k': 3,
    'monitor': 'val_loss',
}

# Normalization configuration
NORM_CONFIG = {
    'method': 'robust',  # 'robust' or 'standard'
    'clip_std': 4.0,  # Clip to mean ± 4*std
    'eps': 1e-8,
}

# Data augmentation (for training)
AUG_CONFIG = {
    'enabled': True,
    'noise_std': 0.05,  # Increased from 0.02 for stronger regularization
    'temporal_shift': 1,  # Small temporal shift for robustness
    'channel_masking_prob': 0.1,  # New: randomly mask 10% of channels
    'time_warping_sigma': 0.2,  # New: time-warping augmentation strength
    'mixup_alpha': 0.2,  # New: mixup augmentation parameter
}

# Paths
CHECKPOINT_DIR = 'develop/checkpoints'
SUBMISSION_DIR = 'develop/submission'
LOG_DIR = 'develop/logs'
