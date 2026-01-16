# Neural Forecasting for HDR Challenge Year 2

Production-ready spatiotemporal neural forecasting system for μECoG signal prediction.

## Architecture

### Spatiotemporal Transformer

The model implements a multi-stage architecture:

1. **Feature Embedding** (9 features → 64-dim)
   - Projects all 9 frequency-band features into a shared embedding space
   - Preserves multi-scale temporal information

2. **Spatial Attention** (2 layers, 4 heads)
   - Models inter-electrode relationships via multi-head attention
   - Captures spatial dependencies across channels
   - Applied independently per timestep

3. **Temporal Transformer** (3 layers, 8 heads)
   - Captures temporal dynamics with self-attention
   - Positional encoding for temporal ordering
   - GELU activation and layer normalization

4. **Forecasting Head** (256 → 128 → 10 timesteps)
   - Channel-specific forecasting
   - Combines global temporal features with channel-specific information

**Total Parameters**: ~2.5M (Beignet: 89 channels), ~2.5M (Affi: 239 channels)

### Key Improvements Over Baseline

| Aspect | Baseline GRU | Our Model |
|--------|-------------|-----------|
| Features used | 1 (feature 0 only) | 9 (all frequency bands) |
| Spatial modeling | None | Multi-head attention |
| Temporal modeling | 1-layer GRU | 3-layer Transformer |
| Channel interaction | Independent | Spatially coupled |
| Parameters | ~250K | ~2.5M |

## Data Pipeline

### Robust Normalization

- **Method**: Robust scaling using median and IQR
- **Rationale**: Handles cross-session distribution shifts better than mean/std
- **Formula**: `z = clip((x - median) / (IQR/1.35), -4, 4)`
- **Session-invariant**: Computed across all training sessions

### Data Augmentation

- **Gaussian Noise**: σ = 0.02 during training
- **Temporal Integrity**: No temporal shifts (preserves causality)

## Training Protocol

### Configuration

```yaml
Training:
  Batch size: 32
  Epochs: 100 (with early stopping)
  Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  Scheduler: Cosine annealing (min_lr=1e-6)
  Gradient clipping: 1.0
  Early stopping: patience=15, min_delta=1e-4

Validation:
  Split: 15% of training data
  Metric: MSE on forecast window (steps 10-19)
```

### Hardware Requirements

- **CPU**: Works fine (training takes ~30-60 min per monkey)
- **GPU**: Recommended for faster training (~5-10 min per monkey)
- **Memory**: ~4GB RAM

## Usage

### 1. Train Models

```bash
# Train both monkeys
python develop/train.py --monkey both

# Train individually
python develop/train.py --monkey affi
python develop/train.py --monkey beignet
```

Training outputs:
- Checkpoints: `develop/checkpoints/{monkey}/checkpoint_best.pth`
- Normalization stats: `develop/checkpoints/{monkey}/normalization_stats.npz`
- Training logs: `develop/logs/{monkey}/training_log.json`

### 2. Evaluate Locally

```bash
python develop/evaluate.py
```

### 3. Create Submission Package

```bash
python develop/create_submission.py
```

This generates:
- `develop/submission/model.py` - Codabench-compatible interface
- `develop/submission/model_affi.pth` - Trained weights
- `develop/submission/model_beignet.pth` - Trained weights
- `develop/submission/normalization_stats_*.npz` - Preprocessing stats
- `develop/submission/requirements.txt` - Dependencies

### 4. Test Submission Locally

```bash
# Using Codabench ingestion/scoring scripts
cd HDRChallenge_y2/NeuralForecasting

python ingested_program/ingestion.py \
    ../../dataset/test \
    ./output \
    ../../develop/submission

python scoring_program/scoring.py \
    ./output \
    ./scores
```

### 5. Submit to Codabench

```bash
cd develop/submission
zip -r submission.zip .
# Upload submission.zip to Codabench
```

## Project Structure

```
develop/
├── config.py                  # Configuration parameters
├── models/
│   ├── __init__.py
│   └── neural_forecaster.py  # Model architecture
├── data/
│   ├── __init__.py
│   └── dataset.py            # Data loading and preprocessing
├── train.py                  # Training script
├── evaluate.py               # Local evaluation
├── create_submission.py      # Submission package generator
├── test_pipeline.py          # Pipeline testing
├── checkpoints/              # Saved model checkpoints
│   ├── affi/
│   └── beignet/
├── logs/                     # Training logs
├── submission/               # Codabench submission package
└── README.md                 # This file
```

## Model Design Decisions

### Why Transformers?

1. **Long-range dependencies**: μECoG signals have temporal patterns across multiple timesteps
2. **Attention mechanism**: Learns which past timesteps are most relevant for forecasting
3. **Parallel computation**: More efficient than RNNs

### Why Spatial Attention?

1. **Electrode proximity**: Nearby electrodes often have correlated signals
2. **Functional connectivity**: Different brain regions interact
3. **Noise robustness**: Spatial averaging reduces channel-specific noise

### Why All 9 Features?

1. **Frequency decomposition**: Features 1-8 are frequency-band decompositions of feature 0
2. **Multi-scale patterns**: Different frequencies capture different neural dynamics
3. **Richer representation**: More information → better predictions

## Performance Expectations

Based on the baseline GRU model and literature:

- **Baseline MSE**: ~0.15-0.25 (single feature, no spatial modeling)
- **Expected MSE**: ~0.08-0.15 (multi-feature, spatiotemporal modeling)
- **Target**: Top 25% of leaderboard

## Troubleshooting

### Training is slow
- Reduce `batch_size` in `config.py`
- Reduce model size (smaller `temporal_hidden_dim`, fewer layers)
- Use GPU if available

### Out of memory
- Reduce `batch_size`
- Reduce model capacity
- Process one monkey at a time

### Poor validation performance
- Check normalization stats are reasonable
- Visualize predictions vs ground truth
- Try different learning rates
- Increase regularization (dropout, weight_decay)

### Submission fails on Codabench
- Test locally with ingestion/scoring scripts first
- Check file paths are relative
- Verify requirements.txt has correct versions
- Ensure weights are saved with `weights_only=True`

## References

1. **AMAG**: Attention-based Multi-scale Adaptive GNN (spatial modeling)
2. **STNDT**: SpatioTemporal Neural Data Transformer (architecture inspiration)
3. **Transformer Architecture**: Vaswani et al. "Attention is All You Need"
4. **Robust Normalization**: For cross-session generalization

## Future Improvements

1. **Graph Neural Networks**: Explicit electrode spatial relationships
2. **Multi-task Learning**: Joint prediction of all features
3. **Autoregressive Forecasting**: Iterative multi-step prediction
4. **Uncertainty Estimation**: Ensemble or Bayesian approaches
5. **Meta-learning**: Fast adaptation to new sessions

## Citation

```bibtex
@inproceedings{hdr_challenge_2025,
  title={Spatiotemporal Neural Forecasting for HDR Challenge},
  year={2025},
  note={Codabench Neural Forecasting Task}
}
```
