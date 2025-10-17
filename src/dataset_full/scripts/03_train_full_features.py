"""
Step 3: Train Two-Stage Model - Full Dataset with ALL Features
================================================================

Strategy based on JSON analysis:
  • Train: batch_01 + batch_04 + batch_05 (30M records)
    - Covers: DDoS, DoS, Reconnaissance, Normal
    - Normal: 8,095 samples (85% of total)
  
  • Test: Sample from batch_02 (1M from 10M)
    - Independent test set
    - Has: DoS + Normal
  
  • Features: ALL 35 features (no mismatch!)

Author: Lambda Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
import joblib

print("="*80)
print("TRAIN TWO-STAGE MODEL - FULL DATASET (ALL 35 FEATURES)")
print("="*80)
print(f"XGBoost version: {xgb.__version__}\n")

# ============================================================================
# 1. LOAD STATISTICS & DEFINE STRATEGY
# ============================================================================

# Auto-detect project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))

# Paths relative to project root
STATS_FILE = os.getenv('STATS_FILE', os.path.join(PROJECT_ROOT, "src/dataset_full/stats/batch_statistics.json"))
BATCH_DIR = os.getenv('BATCH_DIR', os.path.join(PROJECT_ROOT, "Data/Dataset/merged_batches"))
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(PROJECT_ROOT, "models/full_dataset"))

print("\n📂 Detected Paths:")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   Stats file:   {STATS_FILE}")
print(f"   Batch dir:    {BATCH_DIR}")
print(f"   Model dir:    {MODEL_DIR}")

os.makedirs(MODEL_DIR, exist_ok=True)

print("\n" + "="*80)
print("1. STRATEGY (Based on JSON Analysis)")
print("="*80)

with open(STATS_FILE, 'r') as f:
    stats = json.load(f)

print(f"\n📊 Overall Statistics:")
print(f"   Total: {stats['overall']['total_records']:,} records")
print(f"   Normal: {stats['overall']['total_normal']:,} ({stats['overall']['normal_percentage']:.3f}%)")
print(f"   Attack: {stats['overall']['total_attacks']:,}")
print(f"   Imbalance: {stats['overall']['imbalance_ratio']:.1f}:1")

print(f"\n💡 Training Strategy:")
print(f"   Train batches: batch_01, batch_04, batch_05")
print(f"     • batch_01: DoS (81.7%) + Recon (18.2%) + Normal (7,384)")
print(f"     • batch_04: DDoS (51.6%) + DoS (48.4%) + Normal (385)")
print(f"     • batch_05: DDoS (100%) + Normal (326)")
print(f"     → Total: 30M records, ~8K Normal, ALL attack types ✅")

print(f"\n   Test batch: batch_02 (sampled)")
print(f"     • DoS (100%) + Normal (338)")
print(f"     → Sample 1M from 10M for testing")

print(f"\n   Features: ALL 35 features")
print(f"     → No feature mismatch!")
print(f"     → Better DDoS vs DoS separation")

# ============================================================================
# 2. LOAD TRAINING BATCHES
# ============================================================================

print("\n" + "="*80)
print("2. LOADING TRAINING DATA")
print("="*80)

train_batches = ['batch_01', 'batch_04', 'batch_05']
dfs_train = []

for batch_name in train_batches:
    batch_file = os.path.join(BATCH_DIR, f"{batch_name}.csv")
    print(f"\n📂 Loading {batch_name}...", end=" ")
    df = pd.read_csv(batch_file, low_memory=False)
    dfs_train.append(df)
    print(f"✅ {len(df):,} records")

print(f"\n🔧 Merging {len(dfs_train)} batches...")
df_train = pd.concat(dfs_train, ignore_index=True)
print(f"✅ Training data: {len(df_train):,} records")

# ============================================================================
# 3. LOAD TEST DATA (Sample from batch_02)
# ============================================================================

print("\n" + "="*80)
print("3. LOADING TEST DATA")
print("="*80)

test_batch_file = os.path.join(BATCH_DIR, "batch_02.csv")
print(f"\n📂 Loading batch_02 (for sampling)...", end=" ")
df_batch02 = pd.read_csv(test_batch_file, low_memory=False)
print(f"✅ {len(df_batch02):,} records")

# Sample 1M for test (stratified by category)
print(f"\n🔧 Sampling 1M from batch_02 (stratified)...")

if len(df_batch02) > 1000000:
    df_test, _ = train_test_split(
        df_batch02,
        train_size=1000000,
        random_state=42,
        stratify=df_batch02['category']
    )
else:
    df_test = df_batch02

print(f"✅ Test data: {len(df_test):,} records")

# ============================================================================
# 4. DATA EXPLORATION
# ============================================================================

print("\n" + "="*80)
print("4. DATA EXPLORATION")
print("="*80)

print(f"\n📊 Training Distribution:")
train_dist = df_train['category'].value_counts()
for cat, count in train_dist.items():
    pct = count / len(df_train) * 100
    print(f"   {cat:15s}: {count:,} ({pct:.2f}%)")

print(f"\n📊 Test Distribution:")
test_dist = df_test['category'].value_counts()
for cat, count in test_dist.items():
    pct = count / len(df_test) * 100
    print(f"   {cat:15s}: {count:,} ({pct:.2f}%)")

print(f"\n✅ All columns in training data: {len(df_train.columns)}")
print(f"   Columns: {list(df_train.columns)}")

# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("5. FEATURE ENGINEERING")
print("="*80)

# Drop network identifiers and target columns
cols_to_drop = ['pkSeqID', 'saddr', 'sport', 'daddr', 'dport', 
                'smac', 'dmac', 'soui', 'doui', 'sco', 'dco',
                'attack', 'category', 'subcategory']

# Get feature columns
feature_cols = [col for col in df_train.columns if col not in cols_to_drop]

print(f"\n✅ Total features: {len(feature_cols)}")
print(f"   Features: {feature_cols}")

# Select features
df_train = df_train[['category'] + feature_cols]
df_test = df_test[['category'] + feature_cols]

# Handle missing values
print(f"\n🔧 Handling missing values...")
missing_train = df_train.isnull().sum().sum()
missing_test = df_test.isnull().sum().sum()
print(f"   Train: {missing_train:,} NaN values")
print(f"   Test:  {missing_test:,} NaN values")

if missing_train > 0 or missing_test > 0:
    numeric_cols = df_train.select_dtypes(include=['number']).columns
    df_train[numeric_cols] = df_train[numeric_cols].fillna(df_train[numeric_cols].median())
    df_test[numeric_cols] = df_test[numeric_cols].fillna(df_test[numeric_cols].median())
    print(f"   ✅ Filled with median")

# Encode categorical features
print(f"\n🔧 Encoding categorical features...")
cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()
cat_cols = [col for col in cat_cols if col != 'category']  # Keep category

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    
    # Combine train + test to fit
    combined_values = pd.concat([df_train[col], df_test[col]]).unique()
    le.fit(combined_values)
    
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    
    label_encoders[col] = le
    print(f"   ✅ Encoded: {col} ({len(le.classes_)} unique values)")

# Update feature_cols (now all numeric)
feature_cols = [col for col in df_train.columns if col != 'category']

print(f"\n✅ Final features: {len(feature_cols)}")

# ============================================================================
# STAGE 1: BINARY CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("STAGE 1: BINARY CLASSIFICATION - ATTACK vs NORMAL")
print("="*80)

# Create binary target
df_train['is_attack'] = (df_train['category'] != 'Normal').astype(int)
df_test['is_attack'] = (df_test['category'] != 'Normal').astype(int)

print(f"\n📊 Stage 1 Distribution:")
print(f"   Train Normal: {(df_train['is_attack']==0).sum():,} ({(df_train['is_attack']==0).sum()/len(df_train)*100:.3f}%)")
print(f"   Train Attack: {(df_train['is_attack']==1).sum():,} ({(df_train['is_attack']==1).sum()/len(df_train)*100:.2f}%)")
print(f"   Test Normal:  {(df_test['is_attack']==0).sum():,} ({(df_test['is_attack']==0).sum()/len(df_test)*100:.3f}%)")
print(f"   Test Attack:  {(df_test['is_attack']==1).sum():,} ({(df_test['is_attack']==1).sum()/len(df_test)*100:.2f}%)")

normal_train = (df_train['is_attack']==0).sum()
attack_train = (df_train['is_attack']==1).sum()
if normal_train > 0:
    print(f"\n   Imbalance Ratio: {attack_train/normal_train:.1f}:1 (Attack:Normal)")

# Prepare data
X_train_full = df_train[feature_cols]
y_train_full = df_train['is_attack']

X_test = df_test[feature_cols]
y_test_binary = df_test['is_attack']

# Split train into train/val
X_train_s1, X_val_s1, y_train_s1, y_val_s1 = train_test_split(
    X_train_full, y_train_full,
    test_size=0.15,
    random_state=42,
    stratify=y_train_full
)

print(f"\n📊 Stage 1 Split:")
print(f"   Train: {len(X_train_s1):,}")
print(f"   Val:   {len(X_val_s1):,}")
print(f"   Test:  {len(X_test):,}")

# SMOTE
print(f"\n🔧 Applying SMOTE for Stage 1...")
k_neighbors = min((y_train_s1==0).sum() - 1, 5)

if k_neighbors > 0:
    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=k_neighbors)
    X_train_s1_res, y_train_s1_res = smote.fit_resample(X_train_s1, y_train_s1)
    print(f"   ✅ SMOTE (k={k_neighbors}): {len(y_train_s1):,} → {len(y_train_s1_res):,}")
    print(f"      Normal: {(y_train_s1_res==0).sum():,} ({(y_train_s1_res==0).sum()/len(y_train_s1_res)*100:.1f}%)")
    print(f"      Attack: {(y_train_s1_res==1).sum():,} ({(y_train_s1_res==1).sum()/len(y_train_s1_res)*100:.1f}%)")
else:
    X_train_s1_res = X_train_s1
    y_train_s1_res = y_train_s1
    print(f"   ⚠️  SMOTE skipped")

# Train Stage 1
print(f"\n🚀 Training Stage 1 (Binary)...")

scale_pos_weight = (y_train_s1_res==0).sum() / (y_train_s1_res==1).sum() if (y_train_s1_res==1).sum() > 0 else 1.0

model_s1 = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=20,
    n_jobs=-1
)

model_s1.fit(X_train_s1_res, y_train_s1_res, eval_set=[(X_val_s1, y_val_s1)], verbose=False)

print(f"   ✅ Completed! Best iteration: {model_s1.best_iteration}, Score: {model_s1.best_score:.4f}")

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

# ============================================================================
# STAGE 2: MULTI-CLASS (Attack Types)
# ============================================================================

print("\n" + "="*80)
print("STAGE 2: MULTI-CLASS - ATTACK TYPE CLASSIFICATION")
print("="*80)

# Filter attacks
df_train_atk = df_train[df_train['is_attack'] == 1].copy()
df_test_atk = df_test[df_test['is_attack'] == 1].copy()

print(f"\n📊 Stage 2 - Attacks Only:")
print(f"   Train: {len(df_train_atk):,}")
print(f"   Test:  {len(df_test_atk):,}")

# Get attack types (exclude Normal, Theft)
attack_types_train = [c for c in df_train_atk['category'].unique() if c not in ['Normal', 'Theft']]
attack_types_test = [c for c in df_test_atk['category'].unique() if c not in ['Normal', 'Theft']]

print(f"\n📊 Attack types:")
print(f"   Train: {sorted(attack_types_train)}")
print(f"   Test:  {sorted(attack_types_test)}")

# Create mapping
attack_mapping = {cat: idx for idx, cat in enumerate(sorted(set(attack_types_train + attack_types_test)))}

print(f"\n📊 Attack Type Mapping:")
for cat, idx in sorted(attack_mapping.items(), key=lambda x: x[1]):
    print(f"   {cat:15s} → {idx}")

# Filter and encode
df_train_atk = df_train_atk[df_train_atk['category'].isin(list(attack_mapping.keys()))]
df_test_atk = df_test_atk[df_test_atk['category'].isin(list(attack_mapping.keys()))]

df_train_atk['attack_type'] = df_train_atk['category'].map(attack_mapping)
df_test_atk['attack_type'] = df_test_atk['category'].map(attack_mapping)

print(f"\n📊 Attack Distribution (Train):")
for cls in sorted(df_train_atk['attack_type'].unique()):
    name = [k for k, v in attack_mapping.items() if v == cls][0]
    count = (df_train_atk['attack_type'] == cls).sum()
    pct = count / len(df_train_atk) * 100
    print(f"   {name:15s} ({cls}): {count:,} ({pct:.2f}%)")

# Prepare data
X_train_full_s2 = df_train_atk[feature_cols]
y_train_full_s2 = df_train_atk['attack_type']

X_test_s2 = df_test_atk[feature_cols]
y_test_s2 = df_test_atk['attack_type']

# Split
X_train_s2, X_val_s2, y_train_s2, y_val_s2 = train_test_split(
    X_train_full_s2, y_train_full_s2,
    test_size=0.15,
    random_state=42,
    stratify=y_train_full_s2
)

print(f"\n📊 Stage 2 Split:")
print(f"   Train: {len(X_train_s2):,}")
print(f"   Val:   {len(X_val_s2):,}")
print(f"   Test:  {len(X_test_s2):,}")

# SMOTE
print(f"\n🔧 Applying SMOTE for Stage 2...")
min_samples = y_train_s2.value_counts().min()
k_neighbors_s2 = min(min_samples - 1, 5)

if k_neighbors_s2 > 0:
    smote_s2 = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=k_neighbors_s2)
    X_train_s2_res, y_train_s2_res = smote_s2.fit_resample(X_train_s2, y_train_s2)
    print(f"   ✅ SMOTE (k={k_neighbors_s2}): {len(y_train_s2):,} → {len(y_train_s2_res):,}")
else:
    X_train_s2_res = X_train_s2
    y_train_s2_res = y_train_s2
    print(f"   ⚠️  SMOTE skipped")

# Train Stage 2
print(f"\n🚀 Training Stage 2 (Multi-class)...")

num_classes = len(attack_mapping)
print(f"   Number of classes: {num_classes}")

model_s2 = XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    early_stopping_rounds=20,
    n_jobs=-1
)

model_s2.fit(X_train_s2_res, y_train_s2_res, eval_set=[(X_val_s2, y_val_s2)], verbose=False)

print(f"   ✅ Completed! Best iteration: {model_s2.best_iteration}, Score: {model_s2.best_score:.4f}")

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
print("COMBINED TWO-STAGE PIPELINE EVALUATION")
print("="*80)

print(f"\n🔄 Running full pipeline...")

final_predictions = []
final_true_labels = []

for i in range(len(X_test)):
    true_cat = df_test.iloc[i]['category']
    
    # Stage 1
    is_attack = y_pred_s1[i]
    
    if is_attack == 0:
        prediction = 'Normal'
    else:
        # Stage 2
        sample = X_test.iloc[i:i+1]
        attack_type = model_s2.predict(sample)[0]
        prediction = [k for k, v in attack_mapping.items() if v == attack_type][0]
    
    final_predictions.append(prediction)
    final_true_labels.append(true_cat)

# Evaluate
overall_acc = accuracy_score(final_true_labels, final_predictions)

print(f"\n📊 Overall Pipeline Accuracy: {overall_acc:.4f}")

# Classification report
all_categories = ['Normal'] + sorted([k for k in attack_mapping.keys()])
print(f"\n" + "="*80)
print("CLASSIFICATION REPORT - FULL PIPELINE")
print("="*80)
print("\n", classification_report(final_true_labels, final_predictions,
                                  labels=all_categories,
                                  zero_division=0))

# Confusion matrix
cm = confusion_matrix(final_true_labels, final_predictions, labels=all_categories)
print(f"\n📊 Confusion Matrix:")
print(f"{'':15s} " + " ".join([f"{c:>10s}" for c in all_categories]))
print("-" * 80)
for i, true_cat in enumerate(all_categories):
    row = f"{true_cat:15s} " + " ".join([f"{cm[i,j]:10d}" for j in range(len(all_categories))])
    print(row)

# Per-class accuracy
print(f"\n📊 Per-Class Accuracy:")
for i, cat in enumerate(all_categories):
    if cm[i].sum() > 0:
        acc = cm[i, i] / cm[i].sum()
        print(f"   {cat:15s}: {acc:.4f} ({cm[i,i]}/{cm[i].sum()})")

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS & ARTIFACTS")
print("="*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save models
model_s1_path = os.path.join(MODEL_DIR, f"stage1_full_features_{timestamp}.pkl")
joblib.dump(model_s1, model_s1_path)
print(f"\n✅ Stage 1: {os.path.basename(model_s1_path)}")

model_s2_path = os.path.join(MODEL_DIR, f"stage2_full_features_{timestamp}.pkl")
joblib.dump(model_s2, model_s2_path)
print(f"✅ Stage 2: {os.path.basename(model_s2_path)}")

# Save encoders
encoders_path = os.path.join(MODEL_DIR, f"encoders_{timestamp}.pkl")
joblib.dump(label_encoders, encoders_path)
print(f"✅ Encoders: {os.path.basename(encoders_path)}")

# Save mapping
mapping_path = os.path.join(MODEL_DIR, f"attack_mapping_{timestamp}.pkl")
joblib.dump(attack_mapping, mapping_path)
print(f"✅ Mapping: {os.path.basename(mapping_path)}")

# Save feature columns
features_path = os.path.join(MODEL_DIR, f"features_{timestamp}.pkl")
joblib.dump(feature_cols, features_path)
print(f"✅ Features: {os.path.basename(features_path)}")

# Save metrics
metrics = {
    'timestamp': timestamp,
    'strategy': {
        'train_batches': train_batches,
        'test_batch': 'batch_02 (sampled)',
        'num_features': len(feature_cols),
        'features': feature_cols
    },
    'data': {
        'train_records': len(df_train),
        'test_records': len(df_test),
        'train_normal': int((df_train['is_attack']==0).sum()),
        'test_normal': int((df_test['is_attack']==0).sum())
    },
    'performance': {
        'stage1': {
            'accuracy': float(accuracy_score(y_test_binary, y_pred_s1)),
            'precision': float(precision_score(y_test_binary, y_pred_s1, zero_division=0)),
            'recall': float(recall_score(y_test_binary, y_pred_s1, zero_division=0)),
            'f1': float(f1_score(y_test_binary, y_pred_s1, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test_binary, y_pred_s1_proba)) if len(np.unique(y_test_binary)) > 1 else None
        },
        'stage2': {
            'accuracy': float(accuracy_score(y_test_s2, y_pred_s2)),
            'precision_weighted': float(precision_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0))
        },
        'overall': {
            'accuracy': float(overall_acc)
        }
    },
    'confusion_matrix': cm.tolist()
}

metrics_path = os.path.join(MODEL_DIR, f"metrics_full_{timestamp}.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Metrics: {os.path.basename(metrics_path)}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ TRAINING COMPLETED!")
print("="*80)

print(f"\n📊 Summary:")
print(f"   Strategy: Train on batch_01 + batch_04 + batch_05")
print(f"             Test on batch_02 (sampled)")
print(f"   Features: {len(feature_cols)} (ALL features from full dataset)")
print(f"   Training records: {len(df_train):,}")
print(f"   Test records: {len(df_test):,}")

print(f"\n📊 Performance:")
print(f"   Stage 1 Accuracy: {metrics['performance']['stage1']['accuracy']:.4f}")
print(f"   Stage 2 Accuracy: {metrics['performance']['stage2']['accuracy']:.4f}")
print(f"   Overall Accuracy: {metrics['performance']['overall']['accuracy']:.4f}")

print(f"\n📂 Models saved to: {MODEL_DIR}/")
print(f"   • stage1_full_features_{timestamp}.pkl")
print(f"   • stage2_full_features_{timestamp}.pkl")
print(f"   • metrics_full_{timestamp}.json")

print("\n" + "="*80)



