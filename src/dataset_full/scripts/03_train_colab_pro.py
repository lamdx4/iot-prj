"""
Step 3: Train Two-Stage Model - COLAB PRO OPTIMIZED (25GB)
===========================================================

✅ OPTIMIZED FOR: Colab Pro Standard Runtime (25GB RAM)

Strategy:
  • Train: batch_01 + batch_04 (20M records) - SKIP batch_05 to save RAM
  • Test: batch_02 sampled (1M records)
  • Features: ALL 35 features
  • Expected RAM: ~22 GB (fits in 25 GB)
  • Expected Accuracy: 85-90%
  • Aggressive garbage collection

Trade-off:
  ✅ Uses less RAM (22 GB vs 33 GB)
  ✅ Still covers all attack types (DDoS, DoS, Recon)
  ✅ Keeps 96% of Normal samples (7.8K vs 8K)
  ⚠️  Slightly less DDoS samples (only from batch_04)

How to use in Colab:
  1. Runtime → Change runtime type → Standard (or High-RAM for safety)
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
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Configure matplotlib for better display
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

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
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / (1024**3)
        print(f"   💾 RAM: {mem_gb:.2f} GB")
    except:
        pass

def force_gc():
    """Aggressive garbage collection"""
    gc.collect()
    gc.collect()  # Run twice
    gc.collect()

print("="*80)
print("TRAIN TWO-STAGE MODEL - COLAB PRO OPTIMIZED")
print("="*80)
print(f"XGBoost version: {xgb.__version__}")

# Check GPU
try:
    import subprocess
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']).decode()
    print(f"🚀 GPU: {gpu_info.strip()}")
    USE_GPU = True
    TREE_METHOD = "gpu_hist"
except:
    print("⚠️  No GPU, using CPU")
    USE_GPU = False
    TREE_METHOD = "hist"

print_memory_usage()

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

print(f"\n💡 Strategy (OPTIMIZED FOR 25GB RAM):")
print(f"   Train: batch_01 + batch_04 (20M records)")
print(f"   Test:  batch_02 sampled (1M records)")
print(f"   RAM:   Expected ~22 GB")
print(f"   ⚠️  Skipping batch_05 to save ~8GB RAM")

# ============================================================================
# 2. LOAD TRAINING BATCHES
# ============================================================================

print("\n" + "="*80)
print("2. LOADING TRAINING DATA")
print("="*80)

train_batches = ['batch_01', 'batch_04']  # Skip batch_05
dfs_train = []

for batch_name in train_batches:
    batch_file = os.path.join(BATCH_DIR, f"{batch_name}.csv")
    print(f"\n📂 Loading {batch_name}...", end=" ")
    df = pd.read_csv(batch_file, low_memory=False)
    dfs_train.append(df)
    print(f"✅ {len(df):,} records")
    print_memory_usage()

print(f"\n🔧 Merging...")
df_train = pd.concat(dfs_train, ignore_index=True)
print(f"✅ {len(df_train):,} records")

# Clear immediately
del dfs_train
force_gc()
print(f"   🧹 Cleared temp data")
print_memory_usage()

# ============================================================================
# 3. LOAD TEST DATA (with early sampling)
# ============================================================================

print("\n" + "="*80)
print("3. LOADING TEST DATA")
print("="*80)

test_batch_file = os.path.join(BATCH_DIR, "batch_02.csv")

# Load in chunks and sample early to save RAM
print(f"\n📂 Loading batch_02 with chunked sampling...")
chunks = []
target_sample = 1000000
samples_per_chunk = 100000  # Sample 100K per 1M chunk

for i, chunk in enumerate(pd.read_csv(test_batch_file, low_memory=False, chunksize=1000000), 1):
    if len(chunks) * samples_per_chunk < target_sample:
        # Sample from chunk
        if len(chunk) >= samples_per_chunk:
            try:
                chunk_sample = chunk.sample(n=samples_per_chunk, random_state=42)
            except:
                chunk_sample = chunk.iloc[:samples_per_chunk]
        else:
            chunk_sample = chunk
        
        chunks.append(chunk_sample)
        print(f"   Chunk {i}: sampled {len(chunk_sample):,} (total: {sum(len(c) for c in chunks):,})")
    
    if sum(len(c) for c in chunks) >= target_sample:
        break

df_test = pd.concat(chunks, ignore_index=True)
del chunks
force_gc()

print(f"✅ Test data: {len(df_test):,} records")
print_memory_usage()

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

force_gc()

# Missing values
print(f"\n🔧 Handling missing values...")
missing = df_train.isnull().sum().sum() + df_test.isnull().sum().sum()
if missing > 0:
    numeric_cols = df_train.select_dtypes(include=['number']).columns
    df_train[numeric_cols] = df_train[numeric_cols].fillna(df_train[numeric_cols].median())
    df_test[numeric_cols] = df_test[numeric_cols].fillna(df_test[numeric_cols].median())
    print(f"   ✅ Filled {missing:,} values")

# Encode categorical
print(f"\n🔧 Encoding categorical...")
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

feature_cols = [col for col in df_train.columns if col != 'category']
print(f"\n✅ Final features: {len(feature_cols)}")

force_gc()
print_memory_usage()

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

X_train_full = df_train[feature_cols]
y_train_full = df_train['is_attack']
X_test = df_test[feature_cols]
y_test_binary = df_test['is_attack']

# Split
X_train_s1, X_val_s1, y_train_s1, y_val_s1 = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

print(f"\n📊 Split: Train={len(X_train_s1):,}, Val={len(X_val_s1):,}")

# SMOTE
print(f"\n🔧 SMOTE...")
k = min((y_train_s1==0).sum() - 1, 5)
if k > 0:
    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=k)
    X_train_s1_res, y_train_s1_res = smote.fit_resample(X_train_s1, y_train_s1)
    print(f"   ✅ {len(y_train_s1):,} → {len(y_train_s1_res):,}")
else:
    X_train_s1_res, y_train_s1_res = X_train_s1, y_train_s1

force_gc()

# Train
print(f"\n🚀 Training Stage 1...")
print(f"   Using: {TREE_METHOD} {'(GPU ⚡)' if USE_GPU else '(CPU)'}")
scale_pos_weight = (y_train_s1_res==0).sum() / max((y_train_s1_res==1).sum(), 1)

model_s1 = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    tree_method=TREE_METHOD,
    predictor="gpu_predictor" if USE_GPU else "auto",
    random_state=42, eval_metric='logloss',
    early_stopping_rounds=20, 
    n_jobs=1 if USE_GPU else -1
)

model_s1.fit(X_train_s1_res, y_train_s1_res, 
             eval_set=[(X_val_s1, y_val_s1)], verbose=False)

print(f"   ✅ Done! Score: {model_s1.best_score:.4f}")

# Evaluate
y_pred_s1 = model_s1.predict(X_test)
y_pred_s1_proba = model_s1.predict_proba(X_test)[:, 1]

print(f"\n📊 Performance:")
print(f"   Accuracy:  {accuracy_score(y_test_binary, y_pred_s1):.4f}")
print(f"   Precision: {precision_score(y_test_binary, y_pred_s1, zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_test_binary, y_pred_s1, zero_division=0):.4f}")
print(f"   F1:        {f1_score(y_test_binary, y_pred_s1, zero_division=0):.4f}")
if len(np.unique(y_test_binary)) > 1:
    print(f"   ROC-AUC:   {roc_auc_score(y_test_binary, y_pred_s1_proba):.4f}")

force_gc()
print_memory_usage()

# ============================================================================
# STAGE 2: ATTACK TYPE
# ============================================================================

print("\n" + "="*80)
print("STAGE 2: ATTACK TYPE")
print("="*80)

df_train_atk = df_train[df_train['is_attack'] == 1].copy()
df_test_atk = df_test[df_test['is_attack'] == 1].copy()

attack_types = [c for c in df_train_atk['category'].unique() if c not in ['Normal', 'Theft']]
attack_mapping = {cat: idx for idx, cat in enumerate(sorted(attack_types))}

print(f"\n📊 Mapping:")
for cat, idx in sorted(attack_mapping.items(), key=lambda x: x[1]):
    print(f"   {cat} → {idx}")

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

# SMOTE - Limited (PRO version with less RAM)
print(f"\n🔧 SMOTE (limited for 25GB RAM)...")
print(f"   Class distribution:")
for cls, count in sorted(y_train_s2.value_counts().items()):
    cls_name = [k for k, v in attack_mapping.items() if v == cls][0]
    print(f"      {cls_name}: {count:,}")

# Safe sampling: max 5% of majority (more conservative for PRO)
class_counts = y_train_s2.value_counts()
majority_count = class_counts.max()
safe_target = int(majority_count * 0.05)  # 5% for PRO (less RAM)
minority_count = class_counts.min()

if minority_count < safe_target and minority_count > 5:
    sampling_dict = {}
    for cls in class_counts.index:
        if class_counts[cls] < safe_target:
            sampling_dict[cls] = safe_target
    
    k2 = min(minority_count - 1, 5)
    smote_s2 = SMOTE(sampling_strategy=sampling_dict, random_state=42, k_neighbors=k2)
    X_train_s2_res, y_train_s2_res = smote_s2.fit_resample(X_train_s2, y_train_s2)
    print(f"   ✅ {len(y_train_s2):,} → {len(y_train_s2_res):,}")
else:
    print(f"   ⚠️  Skipping SMOTE, using class weights")
    X_train_s2_res, y_train_s2_res = X_train_s2, y_train_s2

force_gc()

# Train with class weights
print(f"\n🚀 Training Stage 2...")
print(f"   Using: {TREE_METHOD} {'(GPU ⚡)' if USE_GPU else '(CPU)'}")

from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train_s2_res)
print(f"   Using balanced sample weights")

model_s2 = XGBClassifier(
    objective='multi:softmax', num_class=len(attack_mapping),
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    tree_method=TREE_METHOD,
    predictor="gpu_predictor" if USE_GPU else "auto",
    random_state=42, eval_metric='mlogloss',
    early_stopping_rounds=20, 
    n_jobs=1 if USE_GPU else -1
)

model_s2.fit(X_train_s2_res, y_train_s2_res,
             sample_weight=sample_weights,
             eval_set=[(X_val_s2, y_val_s2)], verbose=False)

print(f"   ✅ Done! Score: {model_s2.best_score:.4f}")

# Evaluate
y_pred_s2 = model_s2.predict(X_test_s2)

print(f"\n📊 Performance:")
print(f"   Accuracy:  {accuracy_score(y_test_s2, y_pred_s2):.4f}")
print(f"   Precision: {precision_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0):.4f}")
print(f"   F1:        {f1_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0):.4f}")

# ============================================================================
# COMBINED PIPELINE
# ============================================================================

print("\n" + "="*80)
print("COMBINED PIPELINE")
print("="*80)

final_predictions = []
final_true_labels = []

for i in range(len(X_test)):
    true_cat = df_test.iloc[i]['category']
    is_attack = y_pred_s1[i]
    
    if is_attack == 0:
        prediction = 'Normal'
    else:
        sample = X_test.iloc[i:i+1]
        attack_type = model_s2.predict(sample)[0]
        prediction = [k for k, v in attack_mapping.items() if v == attack_type][0]
    
    final_predictions.append(prediction)
    final_true_labels.append(true_cat)

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

joblib.dump(model_s1, os.path.join(MODEL_DIR, f"stage1_pro_{timestamp}.pkl"))
joblib.dump(model_s2, os.path.join(MODEL_DIR, f"stage2_pro_{timestamp}.pkl"))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, f"encoders_pro_{timestamp}.pkl"))
joblib.dump(attack_mapping, os.path.join(MODEL_DIR, f"mapping_pro_{timestamp}.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, f"features_pro_{timestamp}.pkl"))

# Save comprehensive training metrics
training_metrics = {
    'metadata': {
        'timestamp': timestamp,
        'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'version': 'pro_optimized_25gb',
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
metrics_file = os.path.join(MODEL_DIR, f"training_metrics_pro_{timestamp}.json")
with open(metrics_file, 'w') as f:
    json.dump(training_metrics, f, indent=2)

print(f"\n✅ Saved to: {MODEL_DIR}/")
print(f"   • Models: stage1_pro_{timestamp}.pkl, stage2_pro_{timestamp}.pkl")
print(f"   • Metrics: training_metrics_pro_{timestamp}.json")

"""
# ============================================================================
# VISUALIZATION
# ============================================================================

"""
print("⚠️  Visualization disabled to save RAM")

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

viz_dir = os.path.join(MODEL_DIR, f"visualizations_pro_{timestamp}")
os.makedirs(viz_dir, exist_ok=True)

# 1. Confusion Matrix Heatmap
print("\n📊 1. Confusion Matrix Heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues', 
            xticklabels=all_categories, yticklabels=all_categories,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Overall Performance', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '01_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Normalized Confusion Matrix (%)
print("📊 2. Normalized Confusion Matrix...")
cm_normalized = cm_overall.astype('float') / cm_overall.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
            xticklabels=all_categories, yticklabels=all_categories,
            vmin=0, vmax=1, cbar_kws={'label': 'Percentage'})
plt.title('Confusion Matrix (Normalized) - Percentage', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '02_confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Per-Category Metrics Bar Chart
print("📊 3. Per-Category Metrics...")
categories = list(training_metrics['overall']['per_category_metrics'].keys())
precisions = [training_metrics['overall']['per_category_metrics'][c]['precision'] for c in categories]
recalls = [training_metrics['overall']['per_category_metrics'][c]['recall'] for c in categories]
f1_scores = [training_metrics['overall']['per_category_metrics'][c]['f1_score'] for c in categories]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
bars2 = ax.bar(x, recalls, width, label='Recall', color='#2ecc71')
bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')

ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Category Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend(loc='lower right')
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '03_per_category_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Stage Comparison
print("📊 4. Stage Comparison...")
stages = ['Stage 1\n(Binary)', 'Stage 2\n(Multi-class)', 'Overall']
accuracies = [
    training_metrics['stage1']['accuracy'],
    training_metrics['stage2']['accuracy'],
    training_metrics['overall']['accuracy']
]

colors = ['#3498db', '#e74c3c', '#2ecc71']
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(stages, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Model Performance by Stage (PRO - 2 Batches)', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}\n({height*100:.2f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '04_stage_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Class Distribution (Train vs Test)
print("📊 5. Class Distribution...")
train_dist = training_metrics['training_data']['category_distribution']
test_dist = training_metrics['test_data']['category_distribution']

# Get all categories
all_cats = sorted(set(list(train_dist.keys()) + list(test_dist.keys())))
train_counts = [train_dist.get(c, 0) for c in all_cats]
test_counts = [test_dist.get(c, 0) for c in all_cats]

x = np.arange(len(all_cats))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Absolute counts
bars1 = ax1.bar(x - width/2, train_counts, width, label='Training', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, test_counts, width, label='Testing', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Category', fontsize=11, fontweight='bold')
ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
ax1.set_title('Class Distribution: Training vs Testing', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(all_cats, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_yscale('log')  # Log scale for better visibility

# Subplot 2: Percentage
train_total = sum(train_counts)
test_total = sum(test_counts)
train_pcts = [c/train_total*100 for c in train_counts]
test_pcts = [c/test_total*100 for c in test_counts]

bars3 = ax2.bar(x - width/2, train_pcts, width, label='Training', color='#3498db', alpha=0.8)
bars4 = ax2.bar(x + width/2, test_pcts, width, label='Testing', color='#e74c3c', alpha=0.8)

ax2.set_xlabel('Category', fontsize=11, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax2.set_title('Class Distribution: Percentage', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(all_cats, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '05_class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Per-Category Accuracy with Support
print("📊 6. Accuracy vs Support...")
categories = list(training_metrics['overall']['per_category_metrics'].keys())
accuracies_cat = []
supports = []

for i, cat in enumerate(all_categories):
    total = cm_overall[i].sum()
    correct = cm_overall[i, i]
    acc = correct / total if total > 0 else 0
    accuracies_cat.append(acc)
    supports.append(total)

fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = '#2ecc71'
ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color=color1)
bars = ax1.bar(categories, accuracies_cat, color=color1, alpha=0.7, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim([0, 1.1])
ax1.set_xticklabels(categories, rotation=45, ha='right')

# Add value labels
for i, (bar, acc, sup) in enumerate(zip(bars, accuracies_cat, supports)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2%}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Secondary y-axis for support
ax2 = ax1.twinx()
color2 = '#e74c3c'
ax2.set_ylabel('Support (samples)', fontsize=12, fontweight='bold', color=color2)
ax2.plot(categories, supports, color=color2, marker='o', linewidth=2, 
         markersize=8, label='Support', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_yscale('log')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

ax1.set_title('Per-Category Accuracy vs Support Size', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '06_accuracy_vs_support.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. Summary Dashboard
print("📊 7. Summary Dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Top: Confusion Matrix
ax_cm = fig.add_subplot(gs[0:2, 0:2])
sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='RdYlGn', 
            xticklabels=all_categories, yticklabels=all_categories,
            vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, ax=ax_cm)
ax_cm.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
ax_cm.set_ylabel('True Label')
ax_cm.set_xlabel('Predicted Label')

# Top Right: Key Metrics
ax_metrics = fig.add_subplot(gs[0:2, 2])
ax_metrics.axis('off')
metrics_text = f"""
KEY METRICS
[PRO VERSION - 2 Batches]

Overall Accuracy:
{training_metrics['overall']['accuracy']:.4f} ({training_metrics['overall']['accuracy']*100:.2f}%)

Stage 1 (Binary):
• Accuracy: {training_metrics['stage1']['accuracy']:.4f}
• ROC-AUC: {training_metrics['stage1'].get('roc_auc', 0):.4f}

Stage 2 (Multi-class):
• Accuracy: {training_metrics['stage2']['accuracy']:.4f}
• F1-Score: {training_metrics['stage2']['f1_weighted']:.4f}

Dataset:
• Train: {training_metrics['training_data']['total_records']:,} samples
• Test: {training_metrics['test_data']['total_records']:,} samples
• Features: {training_metrics['metadata']['num_features']}

Training Time:
• Timestamp: {training_metrics['metadata']['trained_at']}
"""
ax_metrics.text(0.1, 0.95, metrics_text, transform=ax_metrics.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Bottom: Per-Category Metrics
ax_bars = fig.add_subplot(gs[2, :])
x_pos = np.arange(len(categories))
width = 0.25
bars1 = ax_bars.bar(x_pos - width, precisions, width, label='Precision', color='#3498db', alpha=0.8)
bars2 = ax_bars.bar(x_pos, recalls, width, label='Recall', color='#2ecc71', alpha=0.8)
bars3 = ax_bars.bar(x_pos + width, f1_scores, width, label='F1-Score', color='#e74c3c', alpha=0.8)

ax_bars.set_xlabel('Category', fontweight='bold')
ax_bars.set_ylabel('Score', fontweight='bold')
ax_bars.set_title('Per-Category Performance', fontweight='bold')
ax_bars.set_xticks(x_pos)
ax_bars.set_xticklabels(categories, rotation=45, ha='right')
ax_bars.legend(loc='lower right')
ax_bars.set_ylim([0, 1.1])
ax_bars.grid(axis='y', alpha=0.3)

fig.suptitle('Hierarchical Classification - Training Summary (PRO)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(os.path.join(viz_dir, '00_summary_dashboard.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Visualizations saved to: {viz_dir}/")
print(f"   • 00_summary_dashboard.png (Overview)")
print(f"   • 01_confusion_matrix.png")
print(f"   • 02_confusion_matrix_normalized.png")
print(f"   • 03_per_category_metrics.png")
print(f"   • 04_stage_comparison.png")
print(f"   • 05_class_distribution.png")
print(f"   • 06_accuracy_vs_support.png")

print("\n" + "="*80)
print("✅ COMPLETED!")
print("="*80)
print(f"\n📊 Accuracy: {overall_acc:.4f}")
print(f"📂 Models: {MODEL_DIR}/")
print(f"📊 Visualizations: {viz_dir}/")
print_memory_usage()

print("\n💡 Note: This used 2 batches (batch_01 + batch_04) to fit in 25GB RAM.")
print("   For better accuracy, use High-RAM runtime with 3 batches.")
print("\n" + "="*80)

