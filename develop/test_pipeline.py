"""Quick test to verify the pipeline works end-to-end."""

import numpy as np
import torch
from pathlib import Path

print("Testing Neural Forecasting Pipeline")
print("="*60)

# Test 1: Data loading
print("\n1. Testing data loading...")
from data import load_all_training_data, NeuroForecastDataset
from config import MONKEY_CONFIGS

beignet_files = MONKEY_CONFIGS['beignet']['train_files']
data = load_all_training_data(beignet_files)
print(f"   ✓ Loaded {data.shape[0]} samples")

# Test 2: Dataset creation
print("\n2. Testing dataset creation...")
dataset = NeuroForecastDataset(data, is_train=True, augment=False)
print(f"   ✓ Dataset size: {len(dataset)}")

input_seq, target_seq = dataset[0]
print(f"   ✓ Input shape: {input_seq.shape}")
print(f"   ✓ Target shape: {target_seq.shape}")

mean, std = dataset.get_normalization_stats()
print(f"   ✓ Normalization stats: mean={mean.shape}, std={std.shape}")

# Test 3: Model creation
print("\n3. Testing model creation...")
from models import SpatioTemporalForecaster
from config import MODEL_CONFIG, DEVICE

model = SpatioTemporalForecaster(
    num_channels=89,
    input_features=MODEL_CONFIG['input_features'],
    feature_embed_dim=MODEL_CONFIG['feature_embed_dim'],
    spatial_hidden_dim=MODEL_CONFIG['spatial_hidden_dim'],
    spatial_num_heads=MODEL_CONFIG['spatial_num_heads'],
    spatial_num_layers=MODEL_CONFIG['spatial_num_layers'],
    temporal_hidden_dim=MODEL_CONFIG['temporal_hidden_dim'],
    temporal_num_heads=MODEL_CONFIG['temporal_num_heads'],
    temporal_num_layers=MODEL_CONFIG['temporal_num_layers'],
    forecast_hidden_dims=MODEL_CONFIG['forecast_hidden_dims'],
    input_window=10,
    forecast_window=10,
    dropout=MODEL_CONFIG['forecast_dropout']
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   ✓ Model created with {num_params:,} parameters")

# Test 4: Forward pass
print("\n4. Testing forward pass...")
model.eval()
batch_input = input_seq.unsqueeze(0).to(DEVICE)
with torch.no_grad():
    output = model(batch_input)
print(f"   ✓ Input: {batch_input.shape} -> Output: {output.shape}")

# Test 5: Training step
print("\n5. Testing training step...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

optimizer.zero_grad()
output_train = model(batch_input)
batch_target = target_seq.unsqueeze(0).to(DEVICE)
loss = criterion(output_train, batch_target)
loss.backward()
optimizer.step()
print(f"   ✓ Training step successful, loss: {loss.item():.6f}")

# Test 6: DataLoader
print("\n6. Testing DataLoader...")
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=8, shuffle=True)
batch = next(iter(loader))
input_batch, target_batch = batch
print(f"   ✓ Batch input: {input_batch.shape}")
print(f"   ✓ Batch target: {target_batch.shape}")

# Test 7: Prediction on batch
print("\n7. Testing batch prediction...")
model.eval()
with torch.no_grad():
    batch_output = model(input_batch.to(DEVICE))
print(f"   ✓ Batch output: {batch_output.shape}")

# Test 8: Denormalization
print("\n8. Testing denormalization...")
from data.dataset import denormalize_data

pred_np = batch_output.cpu().numpy()
denorm_pred = denormalize_data(pred_np, mean, std)
print(f"   ✓ Denormalized predictions shape: {denorm_pred.shape}")
print(f"   ✓ Denormalized range: [{denorm_pred.min():.2f}, {denorm_pred.max():.2f}]")

print("\n" + "="*60)
print("All pipeline tests passed!")
print("="*60)
