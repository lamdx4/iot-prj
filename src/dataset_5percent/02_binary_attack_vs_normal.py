"""
Phương án 1: Binary Classification - Attack vs Normal Detection
=================================================================

Đề tài: Phát hiện tấn công IoT bằng XGBoost
Problem: EXTREME Imbalance (3 Normal vs 29,345 Attack)
Solution: SMOTE + XGBoost với class_weight

Author: Lambda Team
Date: October 2025
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Imbalanced Data
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# Model Persistence
import joblib
from datetime import datetime
import os

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("="*80)
print("PHƯƠNG ÁN 1: BINARY CLASSIFICATION - ATTACK vs NORMAL")
print("="*80)
print(f"✅ XGBoost version: {xgb.__version__}")
print(f"✅ Pandas version: {pd.__version__}")

# =============================================================================
# 2. LOAD DATA
# =============================================================================

print("\n" + "="*80)
print("2. LOADING DATASET")
print("="*80)

TRAIN_PATH = "/home/lamdx4/Projects/IOT prj/Data/Dataset/5%/10-best features/split/UNSW_2018_IoT_Botnet_Final_10_Best_Training.csv"
TEST_PATH = "/home/lamdx4/Projects/IOT prj/Data/Dataset/5%/10-best features/split/UNSW_2018_IoT_Botnet_Final_10_Best_Testing.csv"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

print(f"\n✅ Training data loaded: {df_train.shape}")
print(f"✅ Testing data loaded: {df_test.shape}")
print(f"✅ Total samples: {len(df_train) + len(df_test):,}")

# Display sample
print("\nSample data:")
print(df_train.head(3))

# =============================================================================
# 3. CREATE BINARY TARGET (ATTACK vs NORMAL)
# =============================================================================

print("\n" + "="*80)
print("3. CREATE BINARY TARGET")
print("="*80)

# Binary target: 1 = Attack (all attack types), 0 = Normal
df_train['target'] = (df_train['category'] != 'Normal').astype(int)
df_test['target'] = (df_test['category'] != 'Normal').astype(int)

print("\nTarget Distribution:")
print("\nTraining Set:")
train_dist = df_train['target'].value_counts().sort_index()
for cls, count in train_dist.items():
    label = "Normal" if cls == 0 else "Attack"
    print(f"  {label} ({cls}): {count:,} ({count/len(df_train)*100:.4f}%)")

if 0 in train_dist.index and 1 in train_dist.index:
    imbalance_ratio = train_dist[1] / train_dist[0]
    print(f"\n⚠️  EXTREME IMBALANCE RATIO: {imbalance_ratio:.0f}:1 (Attack:Normal)")

print("\nTesting Set:")
test_dist = df_test['target'].value_counts().sort_index()
for cls, count in test_dist.items():
    label = "Normal" if cls == 0 else "Attack"
    print(f"  {label} ({cls}): {count:,} ({count/len(df_test)*100:.4f}%)")

# Visualize distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = ['Normal', 'Attack']
y = [train_dist.get(0, 0), train_dist.get(1, 0)]
colors = ['lightgreen', 'salmon']

bars = ax.bar(x, y, color=colors, edgecolor='black', linewidth=2)
ax.set_title('EXTREME CLASS IMBALANCE - Training Set', fontsize=16, fontweight='bold')
ax.set_xlabel('Class', fontsize=14)
ax.set_ylabel('Count (log scale)', fontsize=14)
ax.set_yscale('log')
ax.grid(alpha=0.3, axis='y')

# Add count labels
for i, (bar, count) in enumerate(zip(bars, y)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.5,
            f'{count:,}\n({count/len(df_train)*100:.2f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/lamdx4/Projects/IOT prj/models/binary_class_distribution.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved visualization: models/binary_class_distribution.png")
plt.show()

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*80)
print("4. FEATURE ENGINEERING")
print("="*80)

# Columns to drop
cols_to_drop = ['saddr', 'sport', 'daddr', 'dport', 'attack', 'category', 'subcategory', 'target']

# Get feature columns
feature_cols = [col for col in df_train.columns if col not in cols_to_drop]

print(f"\nTotal features: {len(feature_cols)}")
print("\nFeatures to use:")
for i, col in enumerate(feature_cols, 1):
    dtype = 'categorical' if df_train[col].dtype == 'object' else 'numeric'
    print(f"  {i:2d}. {col:20s} ({dtype})")

# Encode categorical features (proto)
print("\nEncoding categorical features...")
le_proto = LabelEncoder()
df_train_processed = df_train.copy()
df_test_processed = df_test.copy()

df_train_processed['proto'] = le_proto.fit_transform(df_train_processed['proto'])
df_test_processed['proto'] = le_proto.transform(df_test_processed['proto'])

print(f"\n✅ Protocol encoding:")
for i, proto in enumerate(le_proto.classes_):
    print(f"   {proto:10s} → {i}")

# =============================================================================
# 5. PREPARE TRAIN/VAL/TEST SPLIT
# =============================================================================

print("\n" + "="*80)
print("5. DATA SPLIT")
print("="*80)

# Prepare features and target
X_train_full = df_train_processed[feature_cols]
y_train_full = df_train_processed['target']

X_test = df_test_processed[feature_cols]
y_test = df_test_processed['target']

# Split train into train and validation (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.15,
    random_state=42,
    stratify=y_train_full
)

print(f"\nTraining Set:   {X_train.shape[0]:,} samples")
print(f"  Normal (0):  {(y_train == 0).sum():,}")
print(f"  Attack (1):  {(y_train == 1).sum():,}")

print(f"\nValidation Set: {X_val.shape[0]:,} samples")
print(f"  Normal (0):  {(y_val == 0).sum():,}")
print(f"  Attack (1):  {(y_val == 1).sum():,}")

print(f"\nTest Set:       {X_test.shape[0]:,} samples")
print(f"  Normal (0):  {(y_test == 0).sum():,}")
print(f"  Attack (1):  {(y_test == 1).sum():,}")

# =============================================================================
# 6. HANDLE EXTREME IMBALANCE WITH SMOTE
# =============================================================================

print("\n" + "="*80)
print("6. HANDLING EXTREME IMBALANCE")
print("="*80)

print(f"\n⚠️  Original Training Set:")
print(f"   Attack: {(y_train == 1).sum():,} samples")
print(f"   Normal: {(y_train == 0).sum():,} samples")
print(f"   Ratio:  {(y_train == 1).sum()/(y_train == 0).sum() if (y_train == 0).sum() > 0 else float('inf'):.0f}:1")

# Apply SMOTE
# Since we have very few minority samples, use k_neighbors carefully
k_neighbors = min((y_train == 0).sum() - 1, 5)

if k_neighbors > 0 and (y_train == 0).sum() > 1:
    print(f"\n🔧 Applying SMOTE with k_neighbors={k_neighbors}...")
    smote = SMOTE(
        sampling_strategy=0.1,  # Don't oversample to 100%, just to 10% of majority
        random_state=42,
        k_neighbors=k_neighbors
    )
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\n✅ SMOTE applied successfully!")
    print(f"\nAfter SMOTE:")
    print(f"   Total samples: {X_train_resampled.shape[0]:,}")
    print(f"   Normal (0):    {(y_train_resampled == 0).sum():,} ({(y_train_resampled == 0).sum()/len(y_train_resampled)*100:.2f}%)")
    print(f"   Attack (1):    {(y_train_resampled == 1).sum():,} ({(y_train_resampled == 1).sum()/len(y_train_resampled)*100:.2f}%)")
    print(f"   New Ratio:     {(y_train_resampled == 1).sum()/(y_train_resampled == 0).sum():.1f}:1")
else:
    print("\n⚠️  WARNING: Too few minority samples for SMOTE!")
    print("   Using original data with heavy class weights...")
    X_train_resampled = X_train
    y_train_resampled = y_train

# =============================================================================
# 7. TRAIN XGBOOST MODEL
# =============================================================================

print("\n" + "="*80)
print("7. TRAINING XGBOOST MODEL")
print("="*80)

# Calculate scale_pos_weight
if (y_train_resampled == 0).sum() > 0:
    scale_pos_weight = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum()
else:
    scale_pos_weight = 1.0

print(f"\nModel hyperparameters:")
print(f"  scale_pos_weight: {scale_pos_weight:.4f}")
print(f"  n_estimators: 200")
print(f"  max_depth: 6")
print(f"  learning_rate: 0.1")

# Initialize model
model = XGBClassifier(
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

print("\n🚀 Training model...")

# Train with validation for early stopping
model.fit(
    X_train_resampled, y_train_resampled,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"\n✅ Training completed!")
print(f"   Best iteration: {model.best_iteration}")
print(f"   Best score: {model.best_score:.4f}")

# =============================================================================
# 8. MODEL EVALUATION
# =============================================================================

print("\n" + "="*80)
print("8. MODEL EVALUATION")
print("="*80)

# Predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

y_test_pred_proba = model.predict_proba(X_test)[:, 1]

# Training performance
print("\n📊 Training Set:")
print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"  Precision: {precision_score(y_train, y_train_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_train, y_train_pred, zero_division=0):.4f}")
print(f"  F1-Score:  {f1_score(y_train, y_train_pred, zero_division=0):.4f}")

# Validation performance
print("\n📊 Validation Set:")
print(f"  Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"  Precision: {precision_score(y_val, y_val_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_val, y_val_pred, zero_division=0):.4f}")
print(f"  F1-Score:  {f1_score(y_val, y_val_pred, zero_division=0):.4f}")

# Test performance
print("\n📊 Test Set (Final Evaluation):")
print(f"  Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_test_pred, zero_division=0):.4f}")

if len(np.unique(y_test)) > 1:
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_test_pred_proba):.4f}")

# Classification report
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT - TEST SET")
print("="*80)
print("\n", classification_report(y_test, y_test_pred, 
                                  target_names=['Normal', 'Attack'],
                                  zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_title('Confusion Matrix - Binary Classification\\n(Attack vs Normal)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('/home/lamdx4/Projects/IOT prj/models/binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved confusion matrix: models/binary_confusion_matrix.png")
plt.show()

print("\nConfusion Matrix Breakdown:")
if cm.shape == (2, 2):
    print(f"  True Negatives (TN):  {cm[0,0]:,} (Correctly predicted Normal)")
    print(f"  False Positives (FP): {cm[0,1]:,} (Normal predicted as Attack)")
    print(f"  False Negatives (FN): {cm[1,0]:,} (Attack predicted as Normal)")
    print(f"  True Positives (TP):  {cm[1,1]:,} (Correctly predicted Attack)")

# ROC Curve
if len(np.unique(y_test)) > 1:
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Binary Classification', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/lamdx4/Projects/IOT prj/models/binary_roc_curve.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved ROC curve: models/binary_roc_curve.png")
    plt.show()

# =============================================================================
# 9. FEATURE IMPORTANCE
# =============================================================================

print("\n" + "="*80)
print("9. FEATURE IMPORTANCE ANALYSIS")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
y_pos = range(len(feature_importance))
ax.barh(y_pos, feature_importance['importance'], color='teal', edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_importance['feature'])
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('XGBoost Feature Importance - Binary Classification', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/lamdx4/Projects/IOT prj/models/binary_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved feature importance: models/binary_feature_importance.png")
plt.show()

# =============================================================================
# 10. SAVE MODEL
# =============================================================================

print("\n" + "="*80)
print("10. SAVING MODEL & ARTIFACTS")
print("="*80)

MODEL_DIR = "/home/lamdx4/Projects/IOT prj/models"
os.makedirs(MODEL_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save model
model_path = f"{MODEL_DIR}/xgboost_binary_attack_normal_{timestamp}.pkl"
joblib.dump(model, model_path)
print(f"\n✅ Model saved: {model_path}")

# Save label encoder
encoder_path = f"{MODEL_DIR}/label_encoder_proto_{timestamp}.pkl"
joblib.dump(le_proto, encoder_path)
print(f"✅ Label encoder saved: {encoder_path}")

# Save feature columns
features_path = f"{MODEL_DIR}/feature_columns_binary_{timestamp}.pkl"
joblib.dump(feature_cols, features_path)
print(f"✅ Feature columns saved: {features_path}")

# Save metrics summary
metrics_summary = {
    'model_type': 'Binary Classification (Attack vs Normal)',
    'timestamp': timestamp,
    'test_accuracy': accuracy_score(y_test, y_test_pred),
    'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
    'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
    'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
    'test_roc_auc': roc_auc_score(y_test, y_test_pred_proba) if len(np.unique(y_test)) > 1 else None,
    'imbalance_ratio': f"{(y_train == 1).sum()/(y_train == 0).sum():.0f}:1" if (y_train == 0).sum() > 0 else "inf",
    'smote_applied': k_neighbors > 0 and (y_train == 0).sum() > 1,
    'scale_pos_weight': scale_pos_weight
}

metrics_path = f"{MODEL_DIR}/metrics_binary_{timestamp}.pkl"
joblib.dump(metrics_summary, metrics_path)
print(f"✅ Metrics summary saved: {metrics_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("✅ PHƯƠNG ÁN 1 HOÀN THÀNH - BINARY CLASSIFICATION")
print("="*80)

print("\n📊 Final Results:")
print(f"  Problem Type:     Binary Classification (Attack vs Normal)")
print(f"  Algorithm:        XGBoost")
print(f"  Training Samples: {X_train.shape[0]:,} → {X_train_resampled.shape[0]:,} (after SMOTE)")
print(f"  Test Accuracy:    {metrics_summary['test_accuracy']:.4f}")
print(f"  Test F1-Score:    {metrics_summary['test_f1']:.4f}")
print(f"  Imbalance Ratio:  {metrics_summary['imbalance_ratio']}")

print("\n📂 Saved Files:")
print(f"  • Model: {os.path.basename(model_path)}")
print(f"  • Encoder: {os.path.basename(encoder_path)}")
print(f"  • Features: {os.path.basename(features_path)}")
print(f"  • Metrics: {os.path.basename(metrics_path)}")

print("\n🚀 Next Steps:")
print("  → Chạy script 03_multiclass_4classes.py để train phương án 2")
print("  → Deploy model cho real-time detection")

print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("="*80)

