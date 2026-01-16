"""Test submission package format and interface."""

import numpy as np
import sys
import os

def test_submission_interface():
    """Test that the submission interface matches Codabench requirements."""

    print("="*60)
    print("Testing Submission Interface")
    print("="*60)

    # First, create a mock submission
    print("\n1. Creating mock submission directory...")
    from create_submission import create_submission_model_file
    from pathlib import Path

    # Create temp submission directory
    test_sub_dir = Path('develop/test_submission')
    test_sub_dir.mkdir(parents=True, exist_ok=True)

    # Create model.py
    model_code = create_submission_model_file()
    model_file = test_sub_dir / 'model.py'
    with open(model_file, 'w') as f:
        f.write(model_code)
    print(f"   ✓ Created {model_file}")

    # Copy normalization stats (if available)
    from config import CHECKPOINT_DIR
    checkpoint_dir = Path(CHECKPOINT_DIR) / 'beignet'
    stats_src = checkpoint_dir / 'normalization_stats.npz'

    if stats_src.exists():
        import shutil
        stats_dst = test_sub_dir / 'normalization_stats_beignet.npz'
        shutil.copy(stats_src, stats_dst)

        # Also copy weights
        checkpoint_best = checkpoint_dir / 'checkpoint_best.pth'
        if checkpoint_best.exists():
            import torch
            checkpoint = torch.load(checkpoint_best, map_location='cpu')
            weights_dst = test_sub_dir / 'model_beignet.pth'
            torch.save(checkpoint['model_state_dict'], weights_dst)
            print(f"   ✓ Copied weights and stats")

    # Test 2: Import and instantiate
    print("\n2. Testing model import...")
    sys.path.insert(0, str(test_sub_dir))

    try:
        from model import Model

        # Test initialization
        m = Model('beignet')
        print(f"   ✓ Model initialized for beignet")
        print(f"   ✓ Num channels: {m.num_channels}")

    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return False

    # Test 3: Load weights
    print("\n3. Testing model.load()...")
    try:
        m.load()
        print(f"   ✓ Weights loaded successfully")
    except Exception as e:
        print(f"   ✗ Load failed: {e}")
        print(f"   (This is expected if training isn't complete)")

    # Test 4: Predict interface
    print("\n4. Testing model.predict()...")

    # Create dummy input matching expected format
    # Shape: (N, 20, C, F)
    N = 5
    T = 20
    C = 89  # beignet channels
    F = 9

    dummy_input = np.random.randn(N, T, C, F).astype(np.float32)
    dummy_input = dummy_input * 100 + 300  # Scale to realistic range

    try:
        predictions = m.predict(dummy_input)
        print(f"   ✓ Prediction successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {predictions.shape}")

        # Verify output shape
        expected_shape = (N, T, C)
        if predictions.shape == expected_shape:
            print(f"   ✓ Output shape correct: {expected_shape}")
        else:
            print(f"   ✗ Output shape mismatch!")
            print(f"      Expected: {expected_shape}")
            print(f"      Got: {predictions.shape}")
            return False

        # Verify output dtype
        if predictions.dtype == np.float32 or predictions.dtype == np.float64:
            print(f"   ✓ Output dtype correct: {predictions.dtype}")
        else:
            print(f"   ✗ Output dtype unexpected: {predictions.dtype}")

        # Verify output range (should be reasonable)
        pred_min, pred_max = predictions.min(), predictions.max()
        print(f"   Output range: [{pred_min:.2f}, {pred_max:.2f}]")

        if -10000 < pred_min < 10000 and -10000 < pred_max < 10000:
            print(f"   ✓ Output range reasonable")
        else:
            print(f"   ⚠️  Output range may be unusual")

    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Verify forecast window
    print("\n5. Verifying forecast window behavior...")

    # The first 10 steps should match input
    input_first_10 = dummy_input[:, :10, :, 0]
    pred_first_10 = predictions[:, :10, :]

    are_equal = np.allclose(input_first_10, pred_first_10, rtol=1e-4)
    if are_equal:
        print(f"   ✓ First 10 steps match input (as expected)")
    else:
        max_diff = np.max(np.abs(input_first_10 - pred_first_10))
        print(f"   ✗ First 10 steps don't match input!")
        print(f"      Max difference: {max_diff}")

    # The last 10 steps are the actual predictions
    pred_last_10 = predictions[:, 10:, :]
    print(f"   Forecast window (steps 10-19) stats:")
    print(f"      Mean: {pred_last_10.mean():.2f}")
    print(f"      Std: {pred_last_10.std():.2f}")

    # Test 6: Verify normalization/denormalization
    print("\n6. Testing normalization handling...")

    # Create two batches
    batch1 = m.predict(dummy_input[:2])
    batch2 = m.predict(dummy_input[2:4])

    # They should be different (not all same values)
    if not np.allclose(batch1, batch2):
        print(f"   ✓ Different inputs produce different outputs")
    else:
        print(f"   ⚠️  All predictions are identical (potential bug)")

    print("\n" + "="*60)
    print("Submission Format Test Complete!")
    print("="*60)

    return True


if __name__ == '__main__':
    success = test_submission_interface()

    if success:
        print("\n✓ All tests passed!")
        print("The submission interface is Codabench-compatible.")
    else:
        print("\n✗ Some tests failed.")
        print("Review the errors above before submitting.")

    sys.exit(0 if success else 1)
