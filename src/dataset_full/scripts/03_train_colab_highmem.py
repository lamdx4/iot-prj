"""
Step 3: Train Two-Stage Model - COLAB PRO+ HIGH-RAM (52GB)
===========================================================

✅ OPTIMIZED FOR: Colab Pro+ High-RAM Runtime (52GB RAM)

Strategy:
  • Train: batch_01 + batch_04 + batch_05 (30M records)
  • Test: batch_02 sampled (1M records)
  • Features: ALL 35 features
  • Expected RAM: ~33 GB (with safety margin)
  • Expected Accuracy: 90%+

How to use in Colab:
  1. Runtime → Change runtime type → High-RAM
  2. Upload merged_batches/ folder to Colab
  3. Run this script

Author: Lambda Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
import joblib

# ============================================================================
# MEMORY MONITORING
# ============================================================================

def print_memory_usage():
    """Print current memory usage"""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024**3)
    print(f"   💾 Current RAM usage: {mem_gb:.2f} GB")

print("="*80)
print("TRAIN TWO-STAGE MODEL - COLAB PRO+ HIGH-RAM")
print("="*80)
print(f"XGBoost version: {xgb.__version__}")

# Check GPU availability
try:
    import subprocess
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']).decode()
    print(f"🚀 GPU detected: {gpu_info.strip()}")
    
    # Get GPU memory
    gpu_mem_gb = int(gpu_info.split(',')[1].strip().split()[0]) / 1024
    print(f"   GPU Memory: {gpu_mem_gb:.1f} GB")
    
    USE_GPU = True
    TREE_METHOD = "gpu_hist"
    
    # Optimize for GPU (especially A100)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Force XGBoost to use all GPU memory
    if 'A100' in gpu_info or 'V100' in gpu_info:
        print(f"   🚀 Detected high-end GPU! Optimizing for maximum throughput...")
        # For large datasets, increase GPU cache
        os.environ['XGBOOST_GPU_SINGLE_PRECISION_HISTOGRAM'] = '1'
    
except:
    print("⚠️  No GPU detected, using CPU")
    USE_GPU = False
    TREE_METHOD = "hist"

try:
    print_memory_usage()
except:
    print("   (psutil not available for memory monitoring)")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Auto-detect project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))

# Paths relative to project root (can be overridden via environment variables)
BATCH_DIR = os.getenv('BATCH_DIR', os.path.join(PROJECT_ROOT, "Data/Dataset/merged_batches"))
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(PROJECT_ROOT, "models/full_dataset"))
STATS_FILE = os.getenv('STATS_FILE', os.path.join(PROJECT_ROOT, "src/dataset_full/stats/batch_statistics.json"))

# For Colab: override with simple paths if not in project structure
if not os.path.exists(BATCH_DIR):
    BATCH_DIR = "merged_batches"  # Fallback for Colab
    MODEL_DIR = "models_full"

os.makedirs(MODEL_DIR, exist_ok=True)

print("\n" + "="*80)
print("1. CONFIGURATION")
print("="*80)

print(f"\n📂 Detected Paths:")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   Batch dir:    {BATCH_DIR}")
print(f"   Model dir:    {MODEL_DIR}")
print(f"   Stats file:   {STATS_FILE}")

print(f"\n💡 Strategy:")
print(f"   Train: batch_01 + batch_04 + batch_05 (30M records)")
print(f"   Test:  batch_02 sampled (1M records)")
print(f"   RAM:   Expected ~33 GB (fits in 52 GB)")

# ============================================================================
# 2. LOAD TRAINING BATCHES
# ============================================================================

print("\n" + "="*80)
print("2. LOADING TRAINING DATA")
print("="*80)

train_batches = ['batch_01', 'batch_04']  # batch_05 removed to save ~8GB RAM
dfs_train = []

for batch_name in train_batches:
    batch_file = os.path.join(BATCH_DIR, f"{batch_name}.csv")
    print(f"\n📂 Loading {batch_name}...", end=" ")
    df = pd.read_csv(batch_file, low_memory=False)
    dfs_train.append(df)
    print(f"✅ {len(df):,} records")
    
    try:
        print_memory_usage()
    except:
        pass

print(f"\n🔧 Merging {len(dfs_train)} batches...")
df_train = pd.concat(dfs_train, ignore_index=True)
del dfs_train
gc.collect()

print(f"✅ Training data: {len(df_train):,} records")
try:
    print_memory_usage()
except:
    pass

# ============================================================================
# 3. LOAD TEST DATA
# ============================================================================

print("\n" + "="*80)
print("3. LOADING TEST DATA")
print("="*80)

test_batch_file = os.path.join(BATCH_DIR, "batch_02.csv")
print(f"\n📂 Loading batch_02...", end=" ")
df_batch02 = pd.read_csv(test_batch_file, low_memory=False)
print(f"✅ {len(df_batch02):,} records")

# Sample 1M
print(f"\n🔧 Sampling 300K from batch_02 (RAM optimized)...")
if len(df_batch02) > 1000000:
    df_test, _ = train_test_split(
        df_batch02,
        train_size=300000,
        random_state=42,
        stratify=df_batch02['category']
    )
else:
    df_test = df_batch02

del df_batch02
gc.collect()

print(f"✅ Test data: {len(df_test):,} records")
try:
    print_memory_usage()
except:
    pass

# ============================================================================
# 4. DATA EXPLORATION
# ============================================================================

print("\n" + "="*80)
print("4. DATA EXPLORATION")
print("="*80)

print(f"\n📊 Training Distribution:")
for cat, count in df_train['category'].value_counts().items():
    pct = count / len(df_train) * 100
    print(f"   {cat:15s}: {count:,} ({pct:.2f}%)")

print(f"\n📊 Test Distribution:")
for cat, count in df_test['category'].value_counts().items():
    pct = count / len(df_test) * 100
    print(f"   {cat:15s}: {count:,} ({pct:.2f}%)")

# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("5. FEATURE ENGINEERING")
print("="*80)

cols_to_drop = ['pkSeqID', 'saddr', 'sport', 'daddr', 'dport', 
                'smac', 'dmac', 'soui', 'doui', 'sco', 'dco',
                'attack', 'category', 'subcategory']

feature_cols = [col for col in df_train.columns if col not in cols_to_drop]
print(f"\n✅ Features: {len(feature_cols)}")

df_train = df_train[['category'] + feature_cols]
df_test = df_test[['category'] + feature_cols]

# Missing values
print(f"\n🔧 Handling missing values...")
missing = df_train.isnull().sum().sum() + df_test.isnull().sum().sum()
if missing > 0:
    numeric_cols = df_train.select_dtypes(include=['number']).columns
    df_train[numeric_cols] = df_train[numeric_cols].fillna(df_train[numeric_cols].median())
    df_test[numeric_cols] = df_test[numeric_cols].fillna(df_test[numeric_cols].median())
    print(f"   ✅ Filled {missing:,} NaN values")

# Encode categorical
print(f"\n🔧 Encoding categorical features...")
cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()
cat_cols = [col for col in cat_cols if col != 'category']

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([df_train[col], df_test[col]]).unique()
    le.fit(combined)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le
    print(f"   ✅ {col}: {len(le.classes_)} classes")

feature_cols = [col for col in df_train.columns if col != 'category']
print(f"\n✅ Final features: {len(feature_cols)}")

gc.collect()
try:
    print_memory_usage()
except:
    pass

# ============================================================================
# STAGE 1: BINARY CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("STAGE 1: ATTACK vs NORMAL")
print("="*80)

df_train['is_attack'] = (df_train['category'] != 'Normal').astype(int)
df_test['is_attack'] = (df_test['category'] != 'Normal').astype(int)

print(f"\n📊 Distribution:")
print(f"   Train Normal: {(df_train['is_attack']==0).sum():,}")
print(f"   Train Attack: {(df_train['is_attack']==1).sum():,}")
print(f"   Test Normal:  {(df_test['is_attack']==0).sum():,}")
print(f"   Test Attack:  {(df_test['is_attack']==1).sum():,}")

X_train_full = df_train[feature_cols]
y_train_full = df_train['is_attack']
X_test = df_test[feature_cols]
y_test_binary = df_test['is_attack']

# Split
X_train_s1, X_val_s1, y_train_s1, y_val_s1 = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

print(f"\n📊 Split: Train={len(X_train_s1):,}, Val={len(X_val_s1):,}, Test={len(X_test):,}")

# SMOTE
print(f"\n🔧 Applying SMOTE...")
k = min((y_train_s1==0).sum() - 1, 5)
if k > 0:
    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=k)
    X_train_s1_res, y_train_s1_res = smote.fit_resample(X_train_s1, y_train_s1)
    print(f"   ✅ {len(y_train_s1):,} → {len(y_train_s1_res):,}")
else:
    X_train_s1_res, y_train_s1_res = X_train_s1, y_train_s1

# Train
print(f"\n🚀 Training Stage 1...")
print(f"   Using: {TREE_METHOD} {'(GPU)' if USE_GPU else '(CPU)'}")
scale_pos_weight = (y_train_s1_res==0).sum() / max((y_train_s1_res==1).sum(), 1)

model_s1 = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    tree_method=TREE_METHOD,
    predictor="gpu_predictor" if USE_GPU else "auto",
    random_state=42, eval_metric='logloss',
    early_stopping_rounds=20, 
    n_jobs=1 if USE_GPU else -1,
    # GPU optimization
    max_bin=512 if USE_GPU else 256,  # More bins = more GPU usage
    grow_policy='depthwise' if USE_GPU else 'depthwise',  # Better for GPU
)

model_s1.fit(X_train_s1_res, y_train_s1_res, 
             eval_set=[(X_val_s1, y_val_s1)], verbose=False)

print(f"   ✅ Best iteration: {model_s1.best_iteration}, Score: {model_s1.best_score:.4f}")

# Check GPU usage after training
if USE_GPU:
    try:
        gpu_usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode().strip()
        print(f"   💾 GPU Memory Used: {float(gpu_usage)/1024:.1f} GB")
    except:
        pass

# Evaluate
y_pred_s1 = model_s1.predict(X_test)
y_pred_s1_proba = model_s1.predict_proba(X_test)[:, 1]

print(f"\n📊 Stage 1 Performance:")
print(f"   Accuracy:  {accuracy_score(y_test_binary, y_pred_s1):.4f}")
print(f"   Precision: {precision_score(y_test_binary, y_pred_s1, zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_test_binary, y_pred_s1, zero_division=0):.4f}")
print(f"   F1-Score:  {f1_score(y_test_binary, y_pred_s1, zero_division=0):.4f}")
if len(np.unique(y_test_binary)) > 1:
    print(f"   ROC-AUC:   {roc_auc_score(y_test_binary, y_pred_s1_proba):.4f}")

gc.collect()
try:
    print_memory_usage()
except:
    pass

# ============================================================================
# STAGE 2: ATTACK TYPE CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("STAGE 2: ATTACK TYPE CLASSIFICATION")
print("="*80)

df_train_atk = df_train[df_train['is_attack'] == 1].copy()
df_test_atk = df_test[df_test['is_attack'] == 1].copy()

print(f"\n📊 Attacks: Train={len(df_train_atk):,}, Test={len(df_test_atk):,}")

attack_types = [c for c in df_train_atk['category'].unique() if c not in ['Normal', 'Theft']]
attack_mapping = {cat: idx for idx, cat in enumerate(sorted(attack_types))}

print(f"\n📊 Mapping:")
for cat, idx in sorted(attack_mapping.items(), key=lambda x: x[1]):
    print(f"   {cat:15s} → {idx}")

df_train_atk = df_train_atk[df_train_atk['category'].isin(list(attack_mapping.keys()))]
df_test_atk = df_test_atk[df_test_atk['category'].isin(list(attack_mapping.keys()))]

df_train_atk['attack_type'] = df_train_atk['category'].map(attack_mapping)
df_test_atk['attack_type'] = df_test_atk['category'].map(attack_mapping)

X_train_full_s2 = df_train_atk[feature_cols]
y_train_full_s2 = df_train_atk['attack_type']
X_test_s2 = df_test_atk[feature_cols]
y_test_s2 = df_test_atk['attack_type']

# Split
X_train_s2, X_val_s2, y_train_s2, y_val_s2 = train_test_split(
    X_train_full_s2, y_train_full_s2, test_size=0.15, random_state=42, stratify=y_train_full_s2
)

# SMOTE - Limited to avoid RAM issues
print(f"\n🔧 Applying SMOTE (limited sampling)...")
print(f"   Class distribution before SMOTE:")
for cls, count in sorted(y_train_s2.value_counts().items()):
    cls_name = [k for k, v in attack_mapping.items() if v == cls][0]
    print(f"      {cls_name}: {count:,}")

# Calculate safe sampling strategy (max 10% of majority class)
class_counts = y_train_s2.value_counts()
majority_count = class_counts.max()
safe_target = int(majority_count * 0.1)  # 10% of majority

# Only oversample if minority is VERY small
minority_count = class_counts.min()
if minority_count < safe_target and minority_count > 5:
    # Custom sampling strategy: bring minority up to 10% of majority
    sampling_dict = {}
    for cls in class_counts.index:
        if class_counts[cls] < safe_target:
            sampling_dict[cls] = safe_target
    
    k2 = min(minority_count - 1, 5)
    smote_s2 = SMOTE(sampling_strategy=sampling_dict, random_state=42, k_neighbors=k2)
    X_train_s2_res, y_train_s2_res = smote_s2.fit_resample(X_train_s2, y_train_s2)
    print(f"   ✅ Total: {len(y_train_s2):,} → {len(y_train_s2_res):,}")
    print(f"   New distribution:")
    for cls, count in sorted(pd.Series(y_train_s2_res).value_counts().items()):
        cls_name = [k for k, v in attack_mapping.items() if v == cls][0]
        print(f"      {cls_name}: {count:,}")
else:
    print(f"   ⚠️  Skipping SMOTE (minority too small or safe), using class weights instead")
    X_train_s2_res, y_train_s2_res = X_train_s2, y_train_s2

# Train with class weights
print(f"\n🚀 Training Stage 2...")
print(f"   Using: {TREE_METHOD} {'(GPU)' if USE_GPU else '(CPU)'}")

# Calculate class weights (inverse frequency)
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train_s2_res)
print(f"   Using balanced sample weights to handle class imbalance")

model_s2 = XGBClassifier(
    objective='multi:softmax', num_class=len(attack_mapping),
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    tree_method=TREE_METHOD,
    predictor="gpu_predictor" if USE_GPU else "auto",
    random_state=42, eval_metric='mlogloss',
    early_stopping_rounds=20, 
    n_jobs=1 if USE_GPU else -1,
    # GPU optimization
    max_bin=512 if USE_GPU else 256,  # More bins = more GPU usage
    grow_policy='depthwise' if USE_GPU else 'depthwise',  # Better for GPU
)

model_s2.fit(X_train_s2_res, y_train_s2_res,
             sample_weight=sample_weights,
             eval_set=[(X_val_s2, y_val_s2)], verbose=False)

print(f"   ✅ Best iteration: {model_s2.best_iteration}, Score: {model_s2.best_score:.4f}")

# Check GPU usage after training
if USE_GPU:
    try:
        gpu_usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode().strip()
        print(f"   💾 GPU Memory Used: {float(gpu_usage)/1024:.1f} GB")
    except:
        pass

# Evaluate
y_pred_s2 = model_s2.predict(X_test_s2)

print(f"\n📊 Stage 2 Performance:")
print(f"   Accuracy:  {accuracy_score(y_test_s2, y_pred_s2):.4f}")
print(f"   Precision: {precision_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0):.4f}")
print(f"   F1-Score:  {f1_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0):.4f}")

# ============================================================================
# COMBINED PIPELINE
# ============================================================================

print("\n" + "="*80)
print("COMBINED PIPELINE EVALUATION")
print("="*80)

# Vectorized approach - MUCH FASTER!
print(f"\n🔮 Combining predictions (vectorized)...")

# Initialize with 'Normal' for all predicted as non-attack
final_predictions = np.where(y_pred_s1 == 0, 'Normal', None).astype(object)

# Batch predict for all attacks
attack_mask = (y_pred_s1 == 1)
num_attacks = attack_mask.sum()

if num_attacks > 0:
    print(f"   Processing {num_attacks:,} detected attacks...")
    
    # Get attack samples
    if isinstance(X_test, pd.DataFrame):
        attack_samples = X_test[attack_mask]
    else:
        attack_samples = X_test[attack_mask]
    
    # Batch prediction - MUCH faster than loop!
    attack_types = model_s2.predict(attack_samples)
    
    # Map attack type labels to names
    reverse_mapping = {v: k for k, v in attack_mapping.items()}
    attack_names = np.array([reverse_mapping[t] for t in attack_types])
    
    # Fill in attack predictions
    final_predictions[attack_mask] = attack_names
    
    print(f"   ✅ Batch prediction completed")

# Convert to list and get true labels
final_predictions = list(final_predictions)
final_true_labels = df_test['category'].tolist()

overall_acc = accuracy_score(final_true_labels, final_predictions)
print(f"\n📊 Overall Accuracy: {overall_acc:.6f} ({overall_acc*100:.2f}%)")

all_categories = ['Normal'] + sorted([k for k in attack_mapping.keys()])

# Confusion Matrix
cm_overall = confusion_matrix(final_true_labels, final_predictions, labels=all_categories)

print(f"\n" + "="*80)
print("CONFUSION MATRIX (Detailed)")
print("="*80)
print(f"\n{'':15s} " + " ".join([f"{c:>10s}" for c in all_categories]))
print("─" * (15 + 11 * len(all_categories)))
for i, true_cat in enumerate(all_categories):
    row = f"{true_cat:15s} " + " ".join([f"{cm_overall[i,j]:10d}" for j in range(len(all_categories))])
    print(row)

# Per-Category Accuracy
print(f"\n" + "="*80)
print("PER-CATEGORY ACCURACY")
print("="*80)
for i, cat in enumerate(all_categories):
    total = cm_overall[i].sum()
    correct = cm_overall[i, i]
    acc = correct / total if total > 0 else 0
    print(f"   {cat:15s}: {acc:.6f} ({acc*100:6.2f}%) - {correct:,}/{total:,}")

# Per-Category Detailed Metrics
print(f"\n" + "="*80)
print("PER-CATEGORY DETAILED METRICS")
print("="*80)
for i, cat in enumerate(all_categories):
    total = cm_overall[i].sum()
    pred_total = cm_overall[:, i].sum()
    correct = cm_overall[i, i]
    
    precision = correct / pred_total if pred_total > 0 else 0
    recall = correct / total if total > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n   {cat}:")
    print(f"      Precision: {precision:.6f}")
    print(f"      Recall:    {recall:.6f}")
    print(f"      F1-Score:  {f1:.6f}")
    print(f"      Support:   {total:,}")

print(f"\n{'='*80}")
print("CLASSIFICATION REPORT")
print("="*80)
print("\n", classification_report(final_true_labels, final_predictions,
                                  labels=all_categories, zero_division=0, digits=4))

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

joblib.dump(model_s1, os.path.join(MODEL_DIR, f"stage1_{timestamp}.pkl"))
joblib.dump(model_s2, os.path.join(MODEL_DIR, f"stage2_{timestamp}.pkl"))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, f"encoders_{timestamp}.pkl"))
joblib.dump(attack_mapping, os.path.join(MODEL_DIR, f"mapping_{timestamp}.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, f"features_{timestamp}.pkl"))

print(f"\n✅ Models saved to: {MODEL_DIR}/")
print(f"   • stage1_{timestamp}.pkl")
print(f"   • stage2_{timestamp}.pkl")

# Save comprehensive training metrics
training_metrics = {
    'metadata': {
        'timestamp': timestamp,
        'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'train_samples': len(df_train),
        'test_samples': len(df_test),
        'num_features': len(feature_cols),
        'features': feature_cols,
        'attack_types': list(attack_mapping.keys())
    },
    'training_data': {
        'total_records': len(df_train),
        'normal_count': int((df_train['is_attack']==0).sum()),
        'attack_count': int((df_train['is_attack']==1).sum()),
        'category_distribution': df_train['category'].value_counts().to_dict()
    },
    'test_data': {
        'total_records': len(df_test),
        'normal_count': int((df_test['is_attack']==0).sum()),
        'attack_count': int((df_test['is_attack']==1).sum()),
        'category_distribution': df_test['category'].value_counts().to_dict()
    },
    'stage1': {
        'accuracy': float(accuracy_score(y_test_binary, y_pred_s1)),
        'precision': float(precision_score(y_test_binary, y_pred_s1, zero_division=0)),
        'recall': float(recall_score(y_test_binary, y_pred_s1, zero_division=0)),
        'f1_score': float(f1_score(y_test_binary, y_pred_s1, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test_binary, y_pred_s1_proba)) if len(np.unique(y_test_binary)) > 1 else None,
        'best_iteration': int(model_s1.best_iteration),
        'best_score': float(model_s1.best_score)
    },
    'stage2': {
        'accuracy': float(accuracy_score(y_test_s2, y_pred_s2)),
        'precision_weighted': float(precision_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0)),
        'best_iteration': int(model_s2.best_iteration),
        'best_score': float(model_s2.best_score)
    },
    'overall': {
        'accuracy': float(overall_acc),
        'confusion_matrix': cm_overall.tolist(),
        'per_category_metrics': {}
    }
}

# Add per-category metrics
for i, cat in enumerate(all_categories):
    total = cm_overall[i].sum()
    pred_total = cm_overall[:, i].sum()
    correct = cm_overall[i, i]
    
    precision = float(correct / pred_total) if pred_total > 0 else 0.0
    recall = float(correct / total) if total > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    training_metrics['overall']['per_category_metrics'][cat] = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': int(total),
        'correct': int(correct)
    }

# Save metrics
metrics_file = os.path.join(MODEL_DIR, f"training_metrics_{timestamp}.json")
with open(metrics_file, 'w') as f:
    json.dump(training_metrics, f, indent=2)

print(f"   • training_metrics_{timestamp}.json")

print("\n" + "="*80)
print("✅ TRAINING COMPLETED!")
print("="*80)
print(f"\n📊 Final Accuracy: {overall_acc:.4f}")
print(f"📂 Models: {MODEL_DIR}/")
try:
    print_memory_usage()
except:
    pass

print("\n💡 To generate visualizations (charts), run:")
print(f"   python {os.path.join(SCRIPT_DIR, '05_visualize_results.py')}")
print(f"   (This will read metrics from {metrics_file})")

print("\n" + "="*80)

