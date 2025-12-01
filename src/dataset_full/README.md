# Full Dataset Training Pipeline

Train Two-Stage Hierarchical Model trên toàn bộ Bot-IoT Dataset (74 files, ~16GB)

## 📁 Structure

```
dataset_full/
├── scripts/                          ← Training pipeline scripts
│   ├── 01_merge_files.py            ← Gộp 10 files → 1 batch
│   ├── 02_analyze_batches.py        ← Phân tích → JSON stats
│   ├── 03_train_hierarchical.py     ← Train models
│   └── 04_test_model.py             ← Test models
│
├── stats/                            ← Statistics files
│   ├── batch_statistics.json        ← Detailed stats
│   └── batch_summary.txt            ← Human-readable summary
│
└── README.md                         ← This file
```

## 🚀 Quick Start

### Step 1: Gộp Files (10 files → 1 batch)

```bash
cd /home/lamdx4/Projects/IOT\ prj/src/dataset_full/scripts
python 01_merge_files.py
```

**Output:**
- `Data/Dataset/merged_batches/batch_01.csv` (files 1-10)
- `Data/Dataset/merged_batches/batch_02.csv` (files 11-20)
- ...
- `Data/Dataset/merged_batches/batch_08.csv` (files 71-74)

**Time:** ~10-15 phút

---

### Step 2: Phân tích Batches → JSON

```bash
python 02_analyze_batches.py
```

**Output:**
- `stats/batch_statistics.json` - Chi tiết từng batch
- `stats/batch_summary.txt` - Tóm tắt dễ đọc

**Statistics include:**
- Number of records per batch
- Class distribution (Normal, DDoS, DoS, Reconnaissance)
- Missing values
- Time range (stime, ltime)
- Protocol distribution
- Training recommendations

**Time:** ~5-10 phút

---

### Step 3: Train Models

```bash
python 03_train_hierarchical.py
```

**Process:**
1. Đọc JSON statistics
2. Chọn best batches (prioritize batches with Normal samples)
3. Load và merge selected batches
4. Train Stage 1 (Binary: Attack vs Normal)
5. Train Stage 2 (Multi-class: DDoS, DoS, Recon)
6. Evaluate on test set (5% dataset)
7. Save models + metrics

**Output:**
- `models/full_dataset/stage1_binary_TIMESTAMP.pkl`
- `models/full_dataset/stage2_multiclass_TIMESTAMP.pkl`
- `models/full_dataset/label_encoder_TIMESTAMP.pkl`
- `models/full_dataset/attack_mapping_TIMESTAMP.pkl`
- `models/full_dataset/feature_columns_TIMESTAMP.pkl`
- `models/full_dataset/metrics_TIMESTAMP.json`

**Time:** ~10-20 phút (tùy số batches)

---

### Step 4: Test Models

```bash
python 04_test_model.py
```

**Tests:**
- Load latest trained models
- Predict on test set
- Detailed evaluation per class
- Error analysis
- Sample predictions with confidence

**Output:**
- `models/full_dataset/test_results_TIMESTAMP.json`
- Console output with detailed metrics

**Time:** ~2-3 phút

---

## 📊 Pipeline Flow

```
74 Raw Files (16GB)
    ↓
[01_merge_files.py]
    ↓
8 Batch Files (~2GB each)
    ↓
[02_analyze_batches.py]
    ↓
batch_statistics.json
    ↓
[03_train_hierarchical.py]
    ↓
2 Models (Stage 1 + Stage 2)
    ↓
[04_test_model.py]
    ↓
Test Results + Metrics
```

---

## 🎯 Key Features

### 1. Smart Batch Selection
- Đọc JSON stats để chọn batches
- Prioritize batches có nhiều Normal samples
- Avoid training trên toàn bộ dataset (memory efficient)

### 2. Imbalance Handling
- SMOTE cho Stage 1 (Normal samples)
- SMOTE cho Stage 2 (Reconnaissance minority)
- XGBoost scale_pos_weight

### 3. Comprehensive Metrics
- Per-stage metrics (Stage 1, Stage 2)
- Overall pipeline accuracy
- Confusion matrix
- Per-class accuracy
- Error analysis

### 4. Production-Ready
- Models saved với timestamp
- JSON metrics cho reproducibility
- Test script để verify model quality

---

## 📈 Expected Performance

### Dataset Stats (Full 74 files):
- Total records: ~40-50 million
- Normal samples: ~10,000-20,000 (0.02-0.05%)
- Attack samples: 99.95-99.98%
- Imbalance ratio: ~2000-5000:1

### After Merging (8 batches):
- Batch size: ~5-6 million records each
- Selected for training: Top 3-5 batches (by Normal count)
- Training size: ~15-30 million records

### Model Performance:
- Stage 1 (Binary): 99.5-99.9% accuracy
- Stage 2 (Multi-class): 98-99% accuracy
- Overall Pipeline: 98-99.5% accuracy

---

## ⚙️ Configuration

### Merge Settings (01_merge_files.py):
```python
BATCH_SIZE = 10  # Files per batch
```

### Training Settings (03_train_hierarchical.py):
```python
NUM_BATCHES_TO_USE = 5  # Number of batches to train on
```

Adjust based on:
- Available RAM (mỗi batch ~2GB)
- Training time requirements
- Performance needs

---

## 💡 Tips

### Memory Management:
- **8GB RAM**: Use 2-3 batches
- **16GB RAM**: Use 4-5 batches
- **32GB+ RAM**: Use all 8 batches

### Training Speed:
- **Quick test**: 1-2 batches (~5 phút)
- **Good performance**: 3-5 batches (~15 phút)
- **Best performance**: 6-8 batches (~30 phút)

### Batch Selection Strategy:
- Scripts tự động chọn batches với nhiều Normal nhất
- Normal samples critical cho evaluation
- Attack distribution tương tự nhau giữa batches

---

## 🔍 Troubleshooting

### Out of Memory:
```python
# Trong 03_train_hierarchical.py
NUM_BATCHES_TO_USE = 2  # Giảm xuống 2
```

### Training Too Slow:
```python
# Trong XGBClassifier
n_estimators=100  # Giảm từ 200 → 100
```

### Need More Normal Samples:
→ Scripts đã tự động chọn batches với most Normal
→ Check `stats/batch_summary.txt` để xem distribution

---

## 📂 Output Files

### Merged Batches:
```
Data/Dataset/merged_batches/
├── batch_01.csv  (~2GB, files 1-10)
├── batch_02.csv  (~2GB, files 11-20)
...
└── batch_08.csv  (~0.8GB, files 71-74)
```

### Statistics:
```
src/dataset_full/stats/
├── batch_statistics.json   (detailed stats)
└── batch_summary.txt        (human-readable)
```

### Models:
```
models/full_dataset/
├── stage1_binary_TIMESTAMP.pkl
├── stage2_multiclass_TIMESTAMP.pkl
├── label_encoder_TIMESTAMP.pkl
├── attack_mapping_TIMESTAMP.pkl
├── feature_columns_TIMESTAMP.pkl
├── metrics_TIMESTAMP.json
└── test_results_TIMESTAMP.json
```

---

## 🎓 Cho Đề Tài

### Báo cáo nên include:

1. **Dataset Description:**
   - Show batch statistics
   - Highlight extreme imbalance
   - Explain merge strategy

2. **Methodology:**
   - Two-stage hierarchical approach
   - Smart batch selection
   - Imbalance handling (SMOTE)

3. **Results:**
   - Per-stage performance
   - Overall accuracy
   - Confusion matrix
   - Error analysis

4. **Comparison:**
   - 5% dataset vs Full dataset
   - Show improvement in Normal detection
   - More reliable evaluation

---

## ✅ Advantages vs 5% Dataset

| Aspect | 5% Dataset | Full Dataset (Merged) |
|--------|------------|----------------------|
| **Normal samples** | 4 | 10,000-20,000 |
| **Evaluation reliability** | ❌ Poor | ✅ Good |
| **Training time** | 5 min | 15-30 min |
| **Memory usage** | 2GB | 8-16GB |
| **Accuracy** | 99.6% | 99.5-99.9% |
| **Production-ready** | ⚠️ Limited | ✅ Yes |

---

**Ready to train! 🚀**


