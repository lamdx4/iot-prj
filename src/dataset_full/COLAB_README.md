# 🚀 Google Colab Training Guide

## 📋 Quick Start (Recommended Method)

### ✅ **Using GitHub Repository** (EASIEST!)

**Colab Setup**:
- Runtime type: **Python 3**
- Hardware accelerator: **GPU (T4, V100, or A100)** ⚡
- Runtime shape: **High-RAM** (52GB)

---

## 🔧 Setup Instructions

### 1. Clone Repository

```python
# Clone the project from GitHub
!git clone https://github.com/lamdx4/iot-prj iot-prj
```

**Note**: Repository already includes:
- ✅ All training scripts
- ✅ Pre-processed batch files (in Data/Dataset/merged_batches/)
- ✅ Complete pipeline ready to run

---

### 2. Install Dependencies

```python
!pip install -q xgboost scikit-learn imbalanced-learn pandas numpy joblib psutil matplotlib seaborn
```

---

### 3. Run Training

```python
# Train the two-stage hierarchical model
!python ./iot-prj/src/dataset_full/scripts/03_train_colab_highmem.py
```

**Training Configuration**:
- Train: batch_01 + batch_04 (~20M records)
- Test: batch_02 sampled (300K records)
- RAM: ~33 GB peak usage
- GPU: Uses `tree_method="gpu_hist"` + `gpu_predictor` automatically
- Time: ~15-20 minutes (with GPU)

**Expected Output**:
```
================================================================================
TRAIN TWO-STAGE MODEL - COLAB PRO+ HIGH-RAM
================================================================================
🚀 GPU detected: Tesla T4, 15360 MiB
   GPU Memory: 15.0 GB

📂 Loading batch_01... ✅ 10,000,000 records
📂 Loading batch_04... ✅ 10,000,000 records
✅ Training data: 20,000,000 records

🚀 Training Stage 1... (Attack vs Normal)
   ✅ Best iteration: 127, Score: 0.0023

📊 Stage 1 Performance:
   Accuracy:  0.9987
   Precision: 0.9989
   Recall:    0.9996
   F1-Score:  0.9992

🚀 Training Stage 2... (DDoS, DoS, Reconnaissance)
   ✅ Best iteration: 156, Score: 0.0156

📊 Overall Accuracy: 0.9245 (92.45%)

✅ Models saved to: models/full_dataset/
```

---

### 4. Evaluate Models

```python
# Run evaluation on test set
!python ./iot-prj/src/dataset_full/scripts/04_evaluate_model.py
```

---

### 5. Generate Visualizations

```python
# Create charts and plots
!python ./iot-prj/src/dataset_full/scripts/05_visualize_results.py
```

This will generate 7 visualization files in `models/full_dataset/visualizations/`:
1. `01_confusion_matrix.png`
2. `02_confusion_matrix_normalized.png`
3. `03_per_category_metrics.png`
4. `04_stage_comparison.png`
5. `05_class_distribution.png`
6. `06_accuracy_vs_support.png`
7. `00_summary_dashboard.png`

---

### 6. Download Results

```python
# Zip models and evaluation results
!zip -r /content/models.zip /content/iot-prj/models/
!zip -r /content/evaluation.zip /content/iot-prj/models/full_dataset/visualizations/

# Download
from google.colab import files
files.download('/content/models.zip')
files.download('/content/evaluation.zip')
```

---

## 🚀 GPU Optimization

The training script **automatically detects and uses GPU** if available!

### GPU Detection Output:
```
🚀 GPU detected: Tesla T4, 15360 MiB
   GPU Memory: 15.0 GB
   Using: gpu_hist (GPU ⚡)
```

### Check GPU Availability:
```python
# Verify GPU is enabled
!nvidia-smi
```

### XGBoost GPU Parameters (Auto-configured):
```python
XGBClassifier(
    tree_method="gpu_hist",      # Use GPU for tree construction
    predictor="gpu_predictor",   # Use GPU for prediction
    n_jobs=1,                     # GPU handles parallelism internally
    max_bin=512,                  # More bins for GPU
    ...
)
```

### Speed Comparison:
| GPU Type | Training Time (20M records) |
|----------|----------------------------|
| **CPU only** | ~2-3 hours |
| **T4** | ~15-20 minutes |
| **V100** | ~10-15 minutes |
| **A100** | ~5-10 minutes |

---

## 💾 Memory Usage

The script uses **batch_01 + batch_04** (~20M records) to fit in 52GB High-RAM:

```
Loading batch_01... ✅ 10M records → RAM: ~5 GB
Loading batch_04... ✅ 10M records → RAM: ~8 GB
After preprocessing... → RAM: ~15-20 GB
During training... → RAM: ~25-33 GB (peak)
```

Memory is automatically managed with garbage collection after each major step.

---

## 🎯 Recommendations

### For Best Performance:
1. ✅ Use **Colab Pro+ High-RAM** (52 GB)
2. ✅ Enable **GPU** (T4 or better)
3. ✅ Clone repo directly (no manual uploads)
4. ✅ Let training run ~15-20 minutes
5. ✅ Expected accuracy: **90-95%**

---

## 🐛 Troubleshooting

### "Repository not found" or Clone fails
```python
# Make sure the repo is public or use authentication
!git clone https://github.com/lamdx4/iot-prj iot-prj
```

### "CUDA out of memory"
- Colab GPU has limited VRAM (15GB for T4)
- **Solution**: Script is already optimized for T4 GPU
- If still occurs, restart runtime and try again

### "RAM/Disk quota exceeded"
- Free Colab: ~12GB RAM ❌ Not enough
- Colab Pro: ~25GB RAM ⚠️ Might work
- **Colab Pro+ High-RAM: ~52GB** ✅ Recommended
- **Solution**: Upgrade to High-RAM runtime

### "Training very slow" (Taking hours)
Check if GPU is enabled:
```python
!nvidia-smi
```
If no output, GPU is not enabled.
- **Solution**: Runtime → Change runtime type → GPU

### Import errors
```python
# Reinstall packages
!pip install --upgrade xgboost scikit-learn imbalanced-learn matplotlib seaborn
```

### "File not found" errors
Make sure you cloned the repo correctly:
```python
# Check repo structure
!ls -lh iot-prj/
!ls -lh iot-prj/src/dataset_full/scripts/
!ls -lh iot-prj/Data/Dataset/merged_batches/
```

---

## 📁 File Structure After Clone

```
/content/
└── iot-prj/
    ├── Data/
    │   └── Dataset/
    │       └── merged_batches/        ← Pre-processed batch files
    │           ├── batch_01.csv  (~2.6 GB)
    │           ├── batch_02.csv  (~2.6 GB)
    │           ├── batch_04.csv  (~2.6 GB)
    │           └── batch_05.csv  (~2.6 GB)
    │
    ├── src/
    │   └── dataset_full/
    │       └── scripts/
    │           ├── 03_train_colab_highmem.py  ← Training script
    │           ├── 04_evaluate_model.py       ← Evaluation
    │           └── 05_visualize_results.py    ← Visualization
    │
    └── models/                        ← Created during training
        └── full_dataset/
            ├── stage1_*.pkl
            ├── stage2_*.pkl
            ├── encoders_*.pkl
            ├── mapping_*.pkl
            ├── features_*.pkl
            ├── training_metrics_*.json
            └── visualizations/        ← Created by step 5
                ├── 00_summary_dashboard.png
                ├── 01_confusion_matrix.png
                └── ...
```

---

## ⏱️ Estimated Time & Resources

| Step | Time | RAM Peak | GPU Usage |
|------|------|----------|-----------|
| 1. Clone repo | ~2 min | 1 GB | - |
| 2. Install deps | ~1 min | 1 GB | - |
| 3. Train models | ~15-20 min | 33 GB | High |
| 4. Evaluate | ~3 min | 10 GB | Medium |
| 5. Visualize | ~2 min | 5 GB | - |
| 6. Download | ~2 min | - | - |
| **Total** | **~25-30 min** | **33 GB** | **GPU required** |

---

## 🎓 Tips

1. **Keep tab active**: Colab may disconnect if idle during training
2. **Monitor progress**: Script prints detailed progress logs
3. **GPU utilization**: Check with `!nvidia-smi` in another cell
4. **Save checkpoints**: Models are saved automatically with timestamps
5. **Download immediately**: Runtime resets after 12 hours
6. **Batch files included**: No need to upload large CSV files manually!

---

## 📞 Quick Reference

### Check System Info:
```python
# GPU
!nvidia-smi

# RAM
!free -h

# Disk space
!df -h

# Python packages
!pip list | grep -E "xgboost|scikit|imbalanced"
```

### Verify Files:
```python
# Check batch files exist
!ls -lh iot-prj/Data/Dataset/merged_batches/*.csv

# Check scripts
!ls -lh iot-prj/src/dataset_full/scripts/*.py
```

---

## 🚀 All Commands in One Cell

```python
# Complete pipeline - just run this cell!

# 1. Clone repo
!git clone https://github.com/lamdx4/iot-prj iot-prj

# 2. Install dependencies
!pip install -q xgboost scikit-learn imbalanced-learn pandas numpy joblib psutil matplotlib seaborn

# 3. Train
!python ./iot-prj/src/dataset_full/scripts/03_train_colab_highmem.py

# 4. Evaluate
!python ./iot-prj/src/dataset_full/scripts/04_evaluate_model.py

# 5. Visualize
!python ./iot-prj/src/dataset_full/scripts/05_visualize_results.py

# 6. Download results
!zip -r /content/models.zip /content/iot-prj/models/
!zip -r /content/evaluation.zip /content/iot-prj/models/full_dataset/visualizations/

from google.colab import files
files.download('/content/models.zip')
files.download('/content/evaluation.zip')

print("\n✅ ALL DONE! Models and visualizations downloaded.")
```

---

**Good luck with training! 🚀**
