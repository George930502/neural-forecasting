# Quick Start Guide

## 1. Train Models

```bash
# Train both monkeys (recommended)
python develop/train.py --monkey both

# Or individually
python develop/train.py --monkey beignet  # ~30-60 min on CPU
python develop/train.py --monkey affi     # ~30-60 min on CPU
```

## 2. Monitor Progress

```bash
# Check current status
python develop/monitor_training.py beignet
python develop/monitor_training.py affi

# Check saved checkpoints
ls develop/checkpoints/beignet/
ls develop/checkpoints/affi/
```

## 3. Create Submission

```bash
# Generate Codabench package
python develop/create_submission.py

# Verify format
python develop/test_submission_format.py

# Create zip file
cd develop/submission
zip -r submission.zip .
```

## 4. Submit to Codabench

1. Go to [Codabench Competition Page]
2. Upload `develop/submission/submission.zip`
3. Wait for results

## File Locations

```
develop/
├── train.py                    ← Training script
├── create_submission.py        ← Generate submission
├── checkpoints/                ← Trained models
│   ├── affi/
│   │   ├── checkpoint_best.pth
│   │   └── normalization_stats.npz
│   └── beignet/
│       ├── checkpoint_best.pth
│       └── normalization_stats.npz
└── submission/                 ← Ready for upload
    ├── model.py
    ├── model_affi.pth
    ├── model_beignet.pth
    └── *.npz
```

## Quick Commands

```bash
# Complete pipeline (all-in-one)
bash develop/run_complete_pipeline.sh

# Just test everything works
python develop/test_pipeline.py

# Visualize results
python develop/visualize_results.py
```

## Expected Output

**Training**:
- Beignet: ~858 training samples, MSE ~0.73
- Affi: ~1147 training samples, MSE ~0.65

**Submission**:
- `model.py`: ~220 lines
- `model_*.pth`: ~34 MB each
- `normalization_stats_*.npz`: ~13 KB each

## Troubleshooting

**Training fails**:
- Check data files exist: `ls dataset/train/`
- Reduce batch_size in `config.py`

**Out of memory**:
- Reduce batch_size from 32 to 16 or 8
- Train one monkey at a time

**Submission errors**:
- Run `python develop/test_submission_format.py`
- Check all files present in `develop/submission/`

## Performance Targets

- Validation MSE < 0.80 (good)
- Validation MSE < 0.70 (very good)
- Validation MSE < 0.60 (excellent)

## Data Summary

**Monkey Affi** (239 channels):
- Train: 985 + 162 = 1147 samples
- Test: 122 + 25 = 147 samples

**Monkey Beignet** (89 channels):
- Train: 700 + 82 + 76 = 858 samples
- Test: 65 + 16 + 15 = 96 samples

## Architecture

```
Input: (N, 10, C, 9)
  ↓ Feature Embedding
  ↓ Spatial Attention
  ↓ Temporal Transformer
  ↓ Forecasting Head
Output: (N, 10, C)
```

**Model Size**: ~2.5M parameters per monkey
