# Paper Summaries for Neural Forecasting Cross-Session Generalization

This document summarizes key insights from academic papers relevant to improving cross-session/cross-subject generalization in neural signal forecasting.

---

## 1. Neural Data Transformer (NDT)
**Paper**: arXiv 2108.01210
**Title**: Drop, Swap, and Generate: A Self-Supervised Approach for Generating Neural Activity

### Key Insights
- **Architecture**: Transformer-based model for neural population data
- **Self-supervised pre-training**: Uses masked token prediction similar to BERT
- **Dropout for regularization**: 0.2-0.6 dropout range recommended for neural data
- **Position encoding**: Sinusoidal positional encoding for temporal sequences

### Relevance to Our Problem
- Validates Transformer architecture for neural signals
- Dropout range guidance: our V2 model uses 0.3-0.4 dropout (within recommended range)
- Self-supervised pre-training could help with cross-session generalization

---

## 2. STNDT - Spatiotemporal Neural Data Transformer
**Paper**: arXiv 2206.04727
**Title**: A Spatiotemporal Neural Data Transformer for Accurate and Generalizable Brain-Computer Interfaces

### Key Insights
- **Spatial-temporal factorization**: Separate spatial (electrode) and temporal attention
- **Cross-session generalization**: Key focus of the paper
- **Contrastive learning**: Session-invariant representations via contrastive loss
- **Higher dropout**: Recommends increased dropout (0.3-0.6) for cross-session transfer

### Relevance to Our Problem
- **Directly applicable**: Our SpatioTemporalForecasterV2 implements this architecture
- **Contrastive learning**: InfoNCE-style loss helps learn session-invariant features
- **Dropout strategy**: Validates our higher dropout (0.3-0.4) in V2 model

---

## 3. Domain Generalization for Session-Independent BCI
**Paper**: arXiv 2012.03533
**Title**: Domain Generalization for Session-Independent Brain-Computer Interface

### Key Insights
- **Deeper models generalize better**: ResNet1D-18 (62.58% accuracy) outperformed shallower models
- **ERM beats explicit DG algorithms**: Empirical Risk Minimization outperformed DANN, GroupDRO, Mixup, IRM, and CORAL
- **Fine-tuning can hurt**: Subject-specific fine-tuning degraded performance due to inter-session variability
- **Training-domain validation**: Better predictor of test performance than leave-one-out CV

### Key Takeaways for Our Model
1. **Don't over-complicate**: Simple ERM with good architecture may beat complex domain adaptation
2. **Deeper is better**: Consider increasing model depth (more transformer layers)
3. **Avoid overfitting to sessions**: Our contrastive learning approach aligns with this finding
4. **Validation strategy matters**: Use cross-session validation, not random splits

---

## 4. Confidence-Aware Subject-to-Subject Transfer Learning
**Paper**: arXiv 2112.09243
**Title**: Confidence-Aware Subject-to-Subject Transfer Learning for Brain-Computer Interface

### Key Insights
- **Co-teaching algorithm**: Two networks teach each other, selecting confident samples
- **Small-loss trick**: Samples with small loss are more likely correctly labeled/reliable
- **R(T) schedule**: Gradually reduce percentage of samples used (forget rate)
- **Confidence-aware selection**: Not all source subjects equally useful for transfer

### Key Takeaways for Our Model
1. **Sample weighting**: Could weight training samples by prediction confidence
2. **Co-teaching**: Train two models, use consensus for more robust predictions
3. **Progressive curriculum**: Start with "easier" sessions, gradually add harder ones
4. **Subject selection**: Some training sessions may be more useful than others

---

## 5. Contrastive Learning in Time-Series Forecasting
**Paper**: arXiv 2306.12086
**Title**: What Constitutes Good Contrastive Learning in Time-Series Forecasting?

### Key Insights
- **Best configuration**: End-to-end training with Transformer + MSE + MoCo2 (SSCL)
- **End-to-end > Two-step**: Joint training outperforms frozen encoder approaches
- **MoCo2 is optimal**: Among SSCL methods, MoCo2 consistently best
- **Contrastive helps with**: Scale patterns and periodic patterns in time series
- **Learning rate critical**: Transformers sensitive to LR; 5e-4 often optimal

### Specific Results
| Method | MSE |
|--------|-----|
| Transformer + MSE only | 0.552 |
| Transformer + MSE + MoCo2 | **0.540** |
| Transformer + MSE + TS2Vec | 0.548 |
| Two-step (frozen) | 0.560+ |

### Key Takeaways for Our Model
1. **Keep end-to-end training**: Don't freeze encoder during forecasting
2. **MoCo2 over InfoNCE**: Consider switching to MoCo2 (momentum contrast)
3. **Joint optimization**: MSE + contrastive loss together (we already do this)
4. **Learning rate tuning**: Try 5e-4 learning rate for transformer

---

## 6. Transfer Learning Between Motor Imagery Datasets
**Paper**: arXiv 2311.16109
**Title**: Transfer Learning Across Heterogeneous Motor Imagery EEG Datasets

### Key Insights
- **EEGNet architecture**: Lightweight CNN effective for cross-dataset transfer
- **Pre-train + Fine-tune**: Two-phase approach for transfer learning
- **Linear evaluation protocol**: Freeze backbone, train only linear classifier
- **Donor dataset matters**: Some source datasets transfer better than others
- **Lee2019 best donor**: For left-hand vs right-hand classification

### Transfer Learning Framework
1. **Phase 1 - Pre-training**: Train on multiple source datasets
2. **Phase 2 - Fine-tuning**: Adapt to target with frozen backbone
3. **Linear probe**: Only train final linear layer for evaluation

### Key Takeaways for Our Model
1. **Multi-session pre-training**: Pre-train on all available sessions jointly
2. **Lightweight architectures**: EEGNet-style depthwise separable convolutions could help
3. **Linear probe evaluation**: Test cross-session transfer with frozen encoder
4. **Dataset quality**: Not all training sessions equally valuable

---

## Summary of Actionable Improvements

### Architecture
| Change | Source | Priority |
|--------|--------|----------|
| Increase model depth | Paper 3 | Medium |
| Try MoCo2 instead of InfoNCE | Paper 5 | High |
| Add depthwise separable convs | Paper 6 | Low |

### Training Strategy
| Change | Source | Priority |
|--------|--------|----------|
| End-to-end contrastive training | Paper 5 | Already implemented |
| Higher dropout (0.3-0.4) | Papers 1, 2 | Already implemented |
| Learning rate 5e-4 | Paper 5 | Medium |
| Sample confidence weighting | Paper 4 | Medium |
| Cross-session validation | Paper 3 | High |

### Regularization
| Change | Source | Priority |
|--------|--------|----------|
| Don't fine-tune on specific sessions | Paper 3 | High |
| Contrastive loss for invariance | Paper 2 | Already implemented |
| ERM may beat complex DG | Paper 3 | Consider simplifying |

---

## Next Steps

1. **Immediate**: Monitor V2 training with current contrastive learning setup
2. **If results plateau**: Consider MoCo2 implementation (momentum encoder)
3. **Validation**: Implement proper cross-session validation split
4. **Hyperparameter search**: Try LR=5e-4, different dropout values
5. **Simplification**: If V2 doesn't improve, try simpler ERM baseline

---

*Last updated: January 2026*
