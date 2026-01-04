# Training Run 20251201_144407

**Trained**: 2025-12-01 14:44:07

## Configuration

- Train samples: 20,000,000
- Test samples: 100,000
- Features: 22
- Attack types: DDoS, DoS, Reconnaissance

## Training Metrics

### Stage 1 (Binary Classification)

- Accuracy: 0.9926 (99.26%)
- Precision: 1.0000
- Recall: 0.9924
- F1-Score: 0.9962
- Training time: 30.6s

### Stage 2 (Multi-class Classification)

- Accuracy: 0.9758 (97.58%)
- Precision (weighted): 0.9791
- Recall (weighted): 0.9758
- F1-Score (weighted): 0.9764
- Training time: 58.7s

### Overall Pipeline

- **Overall Accuracy: 0.9719 (97.19%)**

## Files

- `models/` - Trained models and artifacts
- `metrics/` - Training metrics (JSON)
- `plots/` - Loss curves (run visualization script)

## Next Steps

1. Run evaluation: `python 04_evaluate_model.py`
2. Generate plots: `python 06_plot_training_curves.py`
