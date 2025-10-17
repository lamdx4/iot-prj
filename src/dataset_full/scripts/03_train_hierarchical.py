"""
Step 3: Train Two-Stage Hierarchical Model on Full Dataset
===========================================================

Read statistics JSON to determine training strategy
Train Stage 1 (Binary) and Stage 2 (Multi-class)
Save models and comprehensive metrics

Author: Lambda Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
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

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("STEP 3: TRAIN TWO-STAGE HIERARCHICAL MODEL - FULL DATASET")
print("="*80)
print(f"XGBoost version: {xgb.__version__}")

# =============================================================================
# 1. LOAD STATISTICS AND DETERMINE STRATEGY
# =============================================================================

# Auto-detect project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))

# Paths relative to project root
STATS_FILE = os.getenv('STATS_FILE', os.path.join(PROJECT_ROOT, "src/dataset_full/stats/batch_statistics.json"))
BATCH_DIR = os.getenv('BATCH_DIR', os.path.join(PROJECT_ROOT, "Data/Dataset/merged_batches"))
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(PROJECT_ROOT, "models/full_dataset"))
TEST_PATH = os.getenv('TEST_PATH', os.path.join(PROJECT_ROOT, "Data/Dataset/5%/10-best features/split/UNSW_2018_IoT_Botnet_Final_10_Best_Testing.csv"))

print("\n📂 Detected Paths:")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   Stats file:   {STATS_FILE}")
print(f"   Batch dir:    {BATCH_DIR}")
print(f"   Model dir:    {MODEL_DIR}")
print(f"   Test path:    {TEST_PATH}")

os.makedirs(MODEL_DIR, exist_ok=True)

print("\n" + "="*80)
print("1. LOADING STATISTICS")
print("="*80)

with open(STATS_FILE, 'r') as f:
    stats = json.load(f)

print(f"\n✅ Loaded statistics from: {STATS_FILE}")
print(f"\n📊 Overall Statistics:")
print(f"   Total records: {stats['overall']['total_records']:,}")
print(f"   Normal: {stats['overall']['total_normal']:,} ({stats['overall']['normal_percentage']:.2f}%)")
print(f"   Attack: {stats['overall']['total_attacks']:,} ({stats['overall']['attack_percentage']:.2f}%)")

if stats['overall']['total_normal'] > 0:
    print(f"   Imbalance Ratio: {stats['overall']['imbalance_ratio']:.1f}:1 (Attack:Normal)")

# Determine which batches to use
print(f"\n📊 Available batches: {len(stats['batches'])}")

# Strategy: Use batch_01 only (has 77% of all Normal samples)
batches_with_normal = []
for batch_name, batch_stats in stats['batches'].items():
    if batch_stats['num_normal'] > 0:
        batches_with_normal.append((batch_name, batch_stats['num_normal'], batch_stats['num_records']))

batches_with_normal.sort(key=lambda x: x[1], reverse=True)  # Sort by Normal count

print(f"\n💡 Training Strategy:")
print(f"   Batches with Normal samples: {len(batches_with_normal)}")

if batches_with_normal:
    print(f"\n   Top batches by Normal count:")
    for batch_name, normal_count, total in batches_with_normal[:5]:
        pct = normal_count / stats['overall']['total_normal'] * 100
        print(f"     {batch_name}: {normal_count:,} Normal / {total:,} total ({pct:.1f}% of all Normal)")

# Decision: Use batch_01 + batch_04 to cover all attack types
# batch_01: Has most Normal (7,384) + DoS + Reconnaissance
# batch_04: Has DDoS + DoS + Normal (385)
# Combined: Covers all attack types (DDoS, DoS, Reconnaissance) + sufficient Normal

selected_batches = ['batch_01', 'batch_04']

print(f"\n✅ DECISION: Train on batch_01 + batch_04")
print(f"   Reason:")
print(f"     • batch_01: DoS + Recon + 7,384 Normal (most Normal!)")
print(f"     • batch_04: DDoS + DoS + 385 Normal")
print(f"     → Combined covers ALL attack types + sufficient Normal")

for batch in selected_batches:
    batch_info = stats['batches'][batch]
    print(f"\n   📊 {batch}:")
    print(f"      Records: {batch_info['num_records']:,}")
    print(f"      Normal: {batch_info['num_normal']:,}")
    print(f"      Categories: {list(batch_info['category_distribution'].keys())}")

# =============================================================================
# 2. LOAD TRAINING DATA
# =============================================================================

print("\n" + "="*80)
print("2. LOADING TRAINING DATA")
print("="*80)

dfs_train = []
total_train_records = 0

for batch_name in selected_batches:
    batch_file = os.path.join(BATCH_DIR, f"{batch_name}.csv")
    
    print(f"\n  📂 Loading {batch_name}...", end=" ")
    df = pd.read_csv(batch_file)
    dfs_train.append(df)
    total_train_records += len(df)
    print(f"✅ {len(df):,} records")

print(f"\n  🔧 Merging {len(dfs_train)} batches...")
df_train_full = pd.concat(dfs_train, ignore_index=True)

print(f"  ✅ Total training records: {len(df_train_full):,}")

# Load test data (5% dataset)
print(f"\n  📂 Loading test data from 5% dataset...")
df_test = pd.read_csv(TEST_PATH)
print(f"  ✅ Test records: {len(df_test):,}")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*80)
print("3. FEATURE ENGINEERING")
print("="*80)

# Columns to drop
cols_to_drop = ['saddr', 'sport', 'daddr', 'dport', 'attack', 'category', 'subcategory']

print(f"\n🔧 Feature Mapping:")
print(f"   Full dataset has: {list(df_train_full.columns[:15])}...")  # Show first 15
print(f"   Test set has: {list(df_test.columns)}")

# Common features between full dataset and test set (10 best)
# Test set features: proto, seq, stddev, N_IN_Conn_P_SrcIP, min, state_number, mean, N_IN_Conn_P_DstIP, drate, srate, max
# Full dataset features: proto, seq, stddev, min, mean, drate, srate, max, state (not state_number)
# N_IN_Conn_P_SrcIP and N_IN_Conn_P_DstIP are NOT in full dataset (custom features from 5% processing)

# Use only common features (proto will be separate as it needs encoding)
common_features_numeric = ['seq', 'stddev', 'min', 'mean', 'drate', 'srate', 'max']

print(f"\n✅ Common features (7 numeric + 1 categorical):")
print(f"   Categorical: proto")
print(f"   Numeric: {common_features_numeric}")
print(f"   ⚠️  Excluded (not in full dataset): N_IN_Conn_P_SrcIP, N_IN_Conn_P_DstIP, state_number")

# Select columns for training (don't duplicate proto)
df_train_full = df_train_full[['category', 'proto'] + common_features_numeric]
df_test = df_test[['category', 'proto'] + common_features_numeric]

feature_cols = ['proto'] + common_features_numeric

print(f"\n✅ Training will use {len(feature_cols)} features")
print(f"   Features: {feature_cols}")

# Handle missing values
print(f"\n🔧 Handling missing values...")
print(f"   Train - Before: {df_train_full.isnull().sum().sum():,} NaN")
print(f"   Test - Before: {df_test.isnull().sum().sum():,} NaN")

# Fill numeric
numeric_cols = df_train_full.select_dtypes(include=['number']).columns
df_train_full[numeric_cols] = df_train_full[numeric_cols].fillna(df_train_full[numeric_cols].median())

numeric_cols_test = df_test.select_dtypes(include=['number']).columns
df_test[numeric_cols_test] = df_test[numeric_cols_test].fillna(df_test[numeric_cols_test].median())

print(f"   Train - After: {df_train_full.isnull().sum().sum():,} NaN")
print(f"   Test - After: {df_test.isnull().sum().sum():,} NaN")

# Encode proto
print(f"\n🔧 Encoding 'proto' feature...")
le_proto = LabelEncoder()
df_train_full['proto'] = le_proto.fit_transform(df_train_full['proto'].astype(str))
df_test['proto'] = le_proto.transform(df_test['proto'].astype(str))

print(f"   ✅ Protocol encoding:")
for i, proto in enumerate(le_proto.classes_):
    print(f"      {proto:10s} → {i}")

# Class distribution
print(f"\n📊 Training Category Distribution:")
train_dist = df_train_full['category'].value_counts()
for cat, count in train_dist.items():
    pct = count / len(df_train_full) * 100
    print(f"   {cat:15s}: {count:,} ({pct:.2f}%)")

print(f"\n📊 Test Category Distribution:")
test_dist = df_test['category'].value_counts()
for cat, count in test_dist.items():
    pct = count / len(df_test) * 100
    print(f"   {cat:15s}: {count:,} ({pct:.2f}%)")

# =============================================================================
# STAGE 1: BINARY CLASSIFICATION (Attack vs Normal)
# =============================================================================

print("\n" + "="*80)
print("STAGE 1: BINARY CLASSIFICATION - ATTACK vs NORMAL")
print("="*80)

# Create binary target
df_train_full['is_attack'] = (df_train_full['category'] != 'Normal').astype(int)
df_test['is_attack'] = (df_test['category'] != 'Normal').astype(int)

# Prepare data
X_train_full_s1 = df_train_full[feature_cols]
y_train_full_s1 = df_train_full['is_attack']

X_test_s1 = df_test[feature_cols]
y_test_s1 = df_test['is_attack']

# Split train into train/val
X_train_s1, X_val_s1, y_train_s1, y_val_s1 = train_test_split(
    X_train_full_s1, y_train_full_s1,
    test_size=0.15,
    random_state=42,
    stratify=y_train_full_s1
)

print(f"\n📊 Stage 1 Data Split:")
print(f"   Train: {len(X_train_s1):,} samples")
print(f"   Val:   {len(X_val_s1):,} samples")
print(f"   Test:  {len(X_test_s1):,} samples")

print(f"\n📊 Stage 1 Distribution:")
print(f"   Train Normal: {(y_train_s1==0).sum():,} ({(y_train_s1==0).sum()/len(y_train_s1)*100:.2f}%)")
print(f"   Train Attack: {(y_train_s1==1).sum():,} ({(y_train_s1==1).sum()/len(y_train_s1)*100:.2f}%)")
print(f"   Test Normal:  {(y_test_s1==0).sum():,} ({(y_test_s1==0).sum()/len(y_test_s1)*100:.2f}%)")
print(f"   Test Attack:  {(y_test_s1==1).sum():,} ({(y_test_s1==1).sum()/len(y_test_s1)*100:.2f}%)")

if (y_train_s1==0).sum() > 0:
    imb_ratio = (y_train_s1==1).sum() / (y_train_s1==0).sum()
    print(f"\n   ⚠️  Imbalance Ratio: {imb_ratio:.1f}:1 (Attack:Normal)")

# SMOTE
print(f"\n🔧 Applying SMOTE for Stage 1...")
normal_count = (y_train_s1 == 0).sum()
k_neighbors_s1 = min(normal_count - 1, 5)

if k_neighbors_s1 > 0 and normal_count > 5:
    smote_s1 = SMOTE(
        sampling_strategy=0.1,
        random_state=42,
        k_neighbors=k_neighbors_s1
    )
    X_train_s1_resampled, y_train_s1_resampled = smote_s1.fit_resample(X_train_s1, y_train_s1)
    
    print(f"   ✅ SMOTE applied (k={k_neighbors_s1})")
    print(f"      Before: {len(y_train_s1):,} → After: {len(y_train_s1_resampled):,}")
    print(f"      Normal: {(y_train_s1_resampled==0).sum():,} ({(y_train_s1_resampled==0).sum()/len(y_train_s1_resampled)*100:.1f}%)")
    print(f"      Attack: {(y_train_s1_resampled==1).sum():,} ({(y_train_s1_resampled==1).sum()/len(y_train_s1_resampled)*100:.1f}%)")
else:
    X_train_s1_resampled = X_train_s1
    y_train_s1_resampled = y_train_s1
    print(f"   ⚠️  SMOTE skipped (not enough Normal samples)")

# Train Stage 1
print(f"\n🚀 Training Stage 1 Model (Binary)...")

scale_pos_weight = (y_train_s1_resampled==0).sum() / (y_train_s1_resampled==1).sum() if (y_train_s1_resampled==1).sum() > 0 else 1.0

model_stage1 = XGBClassifier(
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

model_stage1.fit(
    X_train_s1_resampled, y_train_s1_resampled,
    eval_set=[(X_val_s1, y_val_s1)],
    verbose=False
)

print(f"   ✅ Training completed!")
print(f"      Best iteration: {model_stage1.best_iteration}")
print(f"      Best score: {model_stage1.best_score:.4f}")

# Evaluate Stage 1
y_test_s1_pred = model_stage1.predict(X_test_s1)
y_test_s1_proba = model_stage1.predict_proba(X_test_s1)[:, 1]

print(f"\n📊 Stage 1 Performance (Test Set):")
print(f"   Accuracy:  {accuracy_score(y_test_s1, y_test_s1_pred):.4f}")
print(f"   Precision: {precision_score(y_test_s1, y_test_s1_pred, zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_test_s1, y_test_s1_pred, zero_division=0):.4f}")
print(f"   F1-Score:  {f1_score(y_test_s1, y_test_s1_pred, zero_division=0):.4f}")
if len(np.unique(y_test_s1)) > 1:
    print(f"   ROC-AUC:   {roc_auc_score(y_test_s1, y_test_s1_proba):.4f}")

# =============================================================================
# STAGE 2: MULTI-CLASS (Attack Types)
# =============================================================================

print("\n" + "="*80)
print("STAGE 2: MULTI-CLASS - ATTACK TYPE CLASSIFICATION")
print("="*80)

# Filter attacks only
df_train_attacks = df_train_full[df_train_full['is_attack'] == 1].copy()
df_test_attacks = df_test[df_test['is_attack'] == 1].copy()

print(f"\n📊 Stage 2 - Attacks Only:")
print(f"   Train: {len(df_train_attacks):,}")
print(f"   Test:  {len(df_test_attacks):,}")

# Get actual attack categories in training data (exclude Theft and Normal)
train_attack_categories = [cat for cat in df_train_attacks['category'].unique() if cat not in ['Normal', 'Theft']]
test_attack_categories = [cat for cat in df_test_attacks['category'].unique() if cat not in ['Normal', 'Theft']]

print(f"\n📊 Attack categories in data:")
print(f"   Train: {sorted(train_attack_categories)}")
print(f"   Test:  {sorted(test_attack_categories)}")

# IMPORTANT: Only train on categories present in training data
# batch_01 doesn't have DDoS, so we'll train only DoS and Reconnaissance
# Test set will be filtered to only these categories
print(f"\n⚠️  NOTE: batch_01 has DoS and Reconnaissance only (no DDoS)")
print(f"   Will train and test only on categories present in training data")

# Create attack type mapping based on TRAINING categories only
# Re-map from 0 to ensure XGBoost gets [0,1,...] without gaps
attack_type_mapping = {cat: idx for idx, cat in enumerate(sorted(train_attack_categories))}

print(f"\n📊 Attack Type Mapping:")
for cat, idx in sorted(attack_type_mapping.items(), key=lambda x: x[1]):
    print(f"   {cat:15s} → {idx}")

# Filter and map
df_train_attacks = df_train_attacks[df_train_attacks['category'].isin(list(attack_type_mapping.keys()))]
df_test_attacks = df_test_attacks[df_test_attacks['category'].isin(list(attack_type_mapping.keys()))]

df_train_attacks['attack_type'] = df_train_attacks['category'].map(attack_type_mapping)
df_test_attacks['attack_type'] = df_test_attacks['category'].map(attack_type_mapping)

print(f"\n📊 Attack Type Distribution (Train):")
for cls in sorted(df_train_attacks['attack_type'].unique()):
    attack_name = [k for k, v in attack_type_mapping.items() if v == cls][0]
    count = (df_train_attacks['attack_type'] == cls).sum()
    pct = count / len(df_train_attacks) * 100
    print(f"   {attack_name:15s} ({cls}): {count:,} ({pct:.2f}%)")

# Prepare Stage 2 data
X_train_full_s2 = df_train_attacks[feature_cols]
y_train_full_s2 = df_train_attacks['attack_type']

X_test_s2 = df_test_attacks[feature_cols]
y_test_s2 = df_test_attacks['attack_type']

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

# SMOTE for minority
print(f"\n🔧 Applying SMOTE for Stage 2...")
min_samples = y_train_s2.value_counts().min()
k_neighbors_s2 = min(min_samples - 1, 5)

if k_neighbors_s2 > 0 and min_samples > 5:
    smote_s2 = SMOTE(
        sampling_strategy='not majority',
        random_state=42,
        k_neighbors=k_neighbors_s2
    )
    X_train_s2_resampled, y_train_s2_resampled = smote_s2.fit_resample(X_train_s2, y_train_s2)
    
    print(f"   ✅ SMOTE applied (k={k_neighbors_s2})")
    print(f"      Before: {len(y_train_s2):,} → After: {len(y_train_s2_resampled):,}")
else:
    X_train_s2_resampled = X_train_s2
    y_train_s2_resampled = y_train_s2
    print(f"   ⚠️  SMOTE skipped")

# Train Stage 2
print(f"\n🚀 Training Stage 2 Model (Multi-class)...")

# Determine number of classes from actual data
num_attack_classes = len(attack_type_mapping)
print(f"   Number of attack classes: {num_attack_classes}")

model_stage2 = XGBClassifier(
    objective='multi:softmax',
    num_class=num_attack_classes,
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

model_stage2.fit(
    X_train_s2_resampled, y_train_s2_resampled,
    eval_set=[(X_val_s2, y_val_s2)],
    verbose=False
)

print(f"   ✅ Training completed!")
print(f"      Best iteration: {model_stage2.best_iteration}")
print(f"      Best score: {model_stage2.best_score:.4f}")

# Evaluate Stage 2
y_test_s2_pred = model_stage2.predict(X_test_s2)

print(f"\n📊 Stage 2 Performance (Test Set - Attacks Only):")
print(f"   Accuracy:  {accuracy_score(y_test_s2, y_test_s2_pred):.4f}")
print(f"   Precision: {precision_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0):.4f}")
print(f"   F1-Score:  {f1_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0):.4f}")

# =============================================================================
# COMBINED PIPELINE
# =============================================================================

print("\n" + "="*80)
print("COMBINED TWO-STAGE PIPELINE")
print("="*80)

print(f"\n🔄 Running full pipeline on test set...")

final_predictions = []
final_true_labels = []

for i in range(len(X_test_s1)):
    true_label = df_test.iloc[i]['category']
    
    # Stage 1
    is_attack = y_test_s1_pred[i]
    
    if is_attack == 0:
        prediction = 'Normal'
    else:
        # Stage 2
        sample = X_test_s1.iloc[i:i+1]
        attack_type = model_stage2.predict(sample)[0]
        prediction = [k for k, v in attack_type_mapping.items() if v == attack_type][0]
    
    final_predictions.append(prediction)
    final_true_labels.append(true_label)

# Evaluate
overall_accuracy = accuracy_score(final_true_labels, final_predictions)

print(f"\n📊 Overall Pipeline Performance:")
print(f"   Accuracy: {overall_accuracy:.4f}")

# Detailed report
all_categories = ['Normal', 'DDoS', 'DoS', 'Reconnaissance']
print(f"\n" + "="*80)
print("CLASSIFICATION REPORT - FULL PIPELINE")
print("="*80)
print("\n", classification_report(final_true_labels, final_predictions,
                                  labels=all_categories,
                                  zero_division=0))

# Confusion matrix
cm_full = confusion_matrix(final_true_labels, final_predictions, labels=all_categories)
print(f"\nConfusion Matrix:")
print(cm_full)

# =============================================================================
# SAVE MODELS
# =============================================================================

print("\n" + "="*80)
print("SAVING MODELS & ARTIFACTS")
print("="*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save models
model_s1_path = os.path.join(MODEL_DIR, f"stage1_binary_{timestamp}.pkl")
joblib.dump(model_stage1, model_s1_path)
print(f"\n✅ Stage 1 model: {os.path.basename(model_s1_path)}")

model_s2_path = os.path.join(MODEL_DIR, f"stage2_multiclass_{timestamp}.pkl")
joblib.dump(model_stage2, model_s2_path)
print(f"✅ Stage 2 model: {os.path.basename(model_s2_path)}")

# Save encoders
encoder_path = os.path.join(MODEL_DIR, f"label_encoder_{timestamp}.pkl")
joblib.dump(le_proto, encoder_path)
print(f"✅ Label encoder: {os.path.basename(encoder_path)}")

# Save mapping
mapping_path = os.path.join(MODEL_DIR, f"attack_mapping_{timestamp}.pkl")
joblib.dump(attack_type_mapping, mapping_path)
print(f"✅ Attack mapping: {os.path.basename(mapping_path)}")

# Save feature columns
features_path = os.path.join(MODEL_DIR, f"feature_columns_{timestamp}.pkl")
joblib.dump(feature_cols, features_path)
print(f"✅ Feature columns: {os.path.basename(features_path)}")

# Save comprehensive metrics
metrics = {
    'timestamp': timestamp,
    'training_info': {
        'num_batches_used': len(selected_batches),
        'batches': selected_batches,
        'total_train_records': len(df_train_full),
        'total_test_records': len(df_test)
    },
    'stage1': {
        'accuracy': float(accuracy_score(y_test_s1, y_test_s1_pred)),
        'precision': float(precision_score(y_test_s1, y_test_s1_pred, zero_division=0)),
        'recall': float(recall_score(y_test_s1, y_test_s1_pred, zero_division=0)),
        'f1': float(f1_score(y_test_s1, y_test_s1_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test_s1, y_test_s1_proba)) if len(np.unique(y_test_s1)) > 1 else None
    },
    'stage2': {
        'accuracy': float(accuracy_score(y_test_s2, y_test_s2_pred)),
        'precision_weighted': float(precision_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0))
    },
    'overall': {
        'accuracy': float(overall_accuracy)
    },
    'confusion_matrix': cm_full.tolist()
}

metrics_path = os.path.join(MODEL_DIR, f"metrics_{timestamp}.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Metrics: {os.path.basename(metrics_path)}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("✅ TRAINING COMPLETED!")
print("="*80)

print(f"\n📊 Training Summary:")
print(f"   Batches used: {len(selected_batches)}")
print(f"   Training records: {len(df_train_full):,}")
print(f"   Test records: {len(df_test):,}")

print(f"\n📊 Performance:")
print(f"   Stage 1 Accuracy: {metrics['stage1']['accuracy']:.4f}")
print(f"   Stage 2 Accuracy: {metrics['stage2']['accuracy']:.4f}")
print(f"   Overall Accuracy: {metrics['overall']['accuracy']:.4f}")

print(f"\n📂 Saved to: {MODEL_DIR}/")
print(f"   • stage1_binary_{timestamp}.pkl")
print(f"   • stage2_multiclass_{timestamp}.pkl")
print(f"   • metrics_{timestamp}.json")

print("\n" + "="*80)
print("NEXT: Run 04_test_model.py to test the trained models")
print("="*80)

