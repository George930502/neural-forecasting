"""Quick data inspection script to understand the dataset structure."""
import numpy as np

# Load and inspect data
def inspect_npz(filepath, name):
    print(f"\n{'='*60}")
    print(f"Inspecting: {name}")
    print(f"{'='*60}")

    data = np.load(filepath)
    arr = data['arr_0']

    print(f"Shape: {arr.shape}")
    print(f"  N (samples): {arr.shape[0]}")
    print(f"  T (timesteps): {arr.shape[1]}")
    print(f"  C (channels): {arr.shape[2]}")
    print(f"  F (features): {arr.shape[3]}")

    print(f"\nData statistics:")
    print(f"  Mean: {arr.mean():.4f}")
    print(f"  Std: {arr.std():.4f}")
    print(f"  Min: {arr.min():.4f}")
    print(f"  Max: {arr.max():.4f}")

    # Check feature 0 vs others
    feat0 = arr[:, :, :, 0]
    feat_rest = arr[:, :, :, 1:]
    print(f"\nFeature 0 (target) stats:")
    print(f"  Mean: {feat0.mean():.4f}, Std: {feat0.std():.4f}")
    print(f"\nFeatures 1-8 stats:")
    for i in range(1, arr.shape[3]):
        f = arr[:, :, :, i]
        print(f"  Feature {i}: Mean={f.mean():.4f}, Std={f.std():.4f}")

    return arr.shape

# Inspect training data
affi_shape = inspect_npz('dataset/train/train_data_affi.npz', 'Affi Training')
beignet_shape = inspect_npz('dataset/train/train_data_beignet.npz', 'Beignet Training')

# Inspect private training data
inspect_npz('dataset/train/train_data_affi_2024-03-20_private.npz', 'Affi Private')
inspect_npz('dataset/train/train_data_beignet_2022-06-01_private.npz', 'Beignet Private 1')
inspect_npz('dataset/train/train_data_beignet_2022-06-02_private.npz', 'Beignet Private 2')

# Inspect test data (masked)
print("\n" + "="*60)
print("TEST DATA (Masked)")
print("="*60)
test_affi = np.load('dataset/test/test_data_affi_masked.npz')['arr_0']
print(f"Test Affi shape: {test_affi.shape}")
print(f"First 10 steps vs last 10 steps comparison:")
print(f"  Steps 0-9 mean: {test_affi[:, :10, :, 0].mean():.4f}")
print(f"  Steps 10-19 mean: {test_affi[:, 10:, :, 0].mean():.4f}")
print(f"  Are last 10 steps == first 10? {np.allclose(test_affi[:, :10], test_affi[:, 10:])}")

print("\nTotal training samples:")
print(f"  Affi: {affi_shape[0]} + 162 private = {affi_shape[0] + 162}")
print(f"  Beignet: {beignet_shape[0]} + 82 + 76 private = {beignet_shape[0] + 82 + 76}")
