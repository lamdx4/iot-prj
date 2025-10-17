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

BATCH_DIR = "merged_batches"
MODEL_DIR = "models_full"
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n" + "="*80)
print("1. CONFIGURATION")
print("="*80)

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

# SMOTE
print(f"\n🔧 SMOTE...")
k2 = min(y_train_s2.value_counts().min() - 1, 5)
if k2 > 0:
    smote_s2 = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=k2)
    X_train_s2_res, y_train_s2_res = smote_s2.fit_resample(X_train_s2, y_train_s2)
    print(f"   ✅ {len(y_train_s2):,} → {len(y_train_s2_res):,}")
else:
    X_train_s2_res, y_train_s2_res = X_train_s2, y_train_s2

force_gc()

# Train
print(f"\n🚀 Training Stage 2...")
print(f"   Using: {TREE_METHOD} {'(GPU ⚡)' if USE_GPU else '(CPU)'}")

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
print(f"\n📊 Overall Accuracy: {overall_acc:.4f}")

all_categories = ['Normal'] + sorted([k for k in attack_mapping.keys()])
print(f"\n{'='*80}")
print("CLASSIFICATION REPORT")
print("="*80)
print("\n", classification_report(final_true_labels, final_predictions,
                                  labels=all_categories, zero_division=0))

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

print(f"\n✅ Saved to: {MODEL_DIR}/")

print("\n" + "="*80)
print("✅ COMPLETED!")
print("="*80)
print(f"\n📊 Accuracy: {overall_acc:.4f}")
print(f"📂 Models: {MODEL_DIR}/")
print_memory_usage()

print("\n💡 Note: This used 2 batches (batch_01 + batch_04) to fit in 25GB RAM.")
print("   For better accuracy, use High-RAM runtime with 3 batches.")
print("\n" + "="*80)

