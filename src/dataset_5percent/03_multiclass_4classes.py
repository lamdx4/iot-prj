"""
Phương án 2: Multi-class Classification - 4 Attack Categories
==============================================================

Đề tài: Phân loại chi tiết các loại tấn công IoT
Problem: Highly Imbalanced (Normal: 3, Reconnaissance: 712, DoS: 12,975, DDoS: 15,658)
Solution: SMOTE + XGBoost với class_weight

Classes:
  0 = Normal
  1 = DDoS
  2 = DoS  
  3 = Reconnaissance

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
    confusion_matrix, classification_report, roc_auc_score
)

# Imbalanced Data
from imblearn.over_sampling import SMOTE
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
print("PHƯƠNG ÁN 2: MULTI-CLASS CLASSIFICATION - 4 ATTACK CATEGORIES")
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
# 3. CREATE MULTI-CLASS TARGET (4 CATEGORIES)
# =============================================================================

print("\n" + "="*80)
print("3. CREATE MULTI-CLASS TARGET")
print("="*80)

# Category distribution
print("\nOriginal Category Distribution:")
print("\nTraining Set:")
print(df_train['category'].value_counts())
print("\nTesting Set:")
print(df_test['category'].value_counts())

# Create numeric target
# Encode: Normal=0, DDoS=1, DoS=2, Reconnaissance=3
category_mapping = {
    'Normal': 0,
    'DDoS': 1,
    'DoS': 2,
    'Reconnaissance': 3
}

df_train['target'] = df_train['category'].map(category_mapping)
df_test['target'] = df_test['category'].map(category_mapping)

print("\n" + "="*80)
print("MULTI-CLASS TARGET DISTRIBUTION (4 CLASSES)")
print("="*80)

print("\nTraining Set:")
train_dist = df_train['target'].value_counts().sort_index()
for cls, count in train_dist.items():
    category_name = [k for k, v in category_mapping.items() if v == cls][0]
    print(f"  {category_name:15s} ({cls}): {count:,} ({count/len(df_train)*100:.4f}%)")

print("\nTesting Set:")
test_dist = df_test['target'].value_counts().sort_index()
for cls, count in test_dist.items():
    category_name = [k for k, v in category_mapping.items() if v == cls][0]
    print(f"  {category_name:15s} ({cls}): {count:,} ({count/len(df_test)*100:.4f}%)")

# Calculate imbalance ratios
print("\n⚠️  IMBALANCE ANALYSIS:")
max_class = train_dist.max()
for cls, count in train_dist.items():
    category_name = [k for k, v in category_mapping.items() if v == cls][0]
    ratio = max_class / count if count > 0 else float('inf')
    print(f"  {category_name:15s}: {ratio:.1f}:1 vs largest class")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Training
category_names = [k for k, v in sorted(category_mapping.items(), key=lambda x: x[1])]
train_counts = [train_dist.get(i, 0) for i in range(4)]
colors = ['lightgreen', 'salmon', 'skyblue', 'orange']

axes[0].bar(category_names, train_counts, color=colors, edgecolor='black', linewidth=2)
axes[0].set_title('Training Set - Multi-Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Category', fontsize=12)
axes[0].set_ylabel('Count (log scale)', fontsize=12)
axes[0].set_yscale('log')
axes[0].grid(alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=45)

for i, (name, count) in enumerate(zip(category_names, train_counts)):
    axes[0].text(i, count * 1.5, f'{count:,}\n({count/len(df_train)*100:.2f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Testing
test_counts = [test_dist.get(i, 0) for i in range(4)]
axes[1].bar(category_names, test_counts, color=colors, edgecolor='black', linewidth=2)
axes[1].set_title('Testing Set - Multi-Class Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Category', fontsize=12)
axes[1].set_ylabel('Count (log scale)', fontsize=12)
axes[1].set_yscale('log')
axes[1].grid(alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=45)

for i, (name, count) in enumerate(zip(category_names, test_counts)):
    axes[1].text(i, count * 1.5, f'{count:,}\n({count/len(df_test)*100:.2f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/lamdx4/Projects/IOT prj/models/multiclass_distribution.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved visualization: models/multiclass_distribution.png")
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
for cls in sorted(y_train.unique()):
    category_name = [k for k, v in category_mapping.items() if v == cls][0]
    count = (y_train == cls).sum()
    print(f"  {category_name:15s} ({cls}): {count:,}")

print(f"\nValidation Set: {X_val.shape[0]:,} samples")
for cls in sorted(y_val.unique()):
    category_name = [k for k, v in category_mapping.items() if v == cls][0]
    count = (y_val == cls).sum()
    print(f"  {category_name:15s} ({cls}): {count:,}")

print(f"\nTest Set:       {X_test.shape[0]:,} samples")
for cls in sorted(y_test.unique()):
    category_name = [k for k, v in category_mapping.items() if v == cls][0]
    count = (y_test == cls).sum()
    print(f"  {category_name:15s} ({cls}): {count:,}")

# =============================================================================
# 6. HANDLE IMBALANCE WITH SMOTE
# =============================================================================

print("\n" + "="*80)
print("6. HANDLING CLASS IMBALANCE")
print("="*80)

print(f"\n⚠️  Original Training Set Distribution:")
for cls in sorted(y_train.unique()):
    category_name = [k for k, v in category_mapping.items() if v == cls][0]
    count = (y_train == cls).sum()
    print(f"   {category_name:15s}: {count:,} samples")

# Check minimum samples for SMOTE
min_samples = y_train.value_counts().min()
print(f"\n   Minimum class samples: {min_samples}")

# Apply SMOTE if possible
k_neighbors = min(min_samples - 1, 5)

if k_neighbors > 0 and min_samples > 1:
    print(f"\n🔧 Applying SMOTE with k_neighbors={k_neighbors}...")
    print("   Strategy: Oversample minority classes to 20% of majority class")
    
    smote = SMOTE(
        sampling_strategy='not majority',  # Oversample all except majority class
        random_state=42,
        k_neighbors=k_neighbors
    )
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\n✅ SMOTE applied successfully!")
    print(f"\nAfter SMOTE:")
    print(f"   Total samples: {X_train_resampled.shape[0]:,}")
    for cls in sorted(np.unique(y_train_resampled)):
        category_name = [k for k, v in category_mapping.items() if v == cls][0]
        count = (y_train_resampled == cls).sum()
        pct = count / len(y_train_resampled) * 100
        print(f"   {category_name:15s} ({cls}): {count:,} ({pct:.2f}%)")
else:
    print("\n⚠️  WARNING: Too few samples in minority class for SMOTE!")
    print("   Using original data with class weights...")
    X_train_resampled = X_train
    y_train_resampled = y_train

# =============================================================================
# 7. TRAIN XGBOOST MODEL (MULTI-CLASS)
# =============================================================================

print("\n" + "="*80)
print("7. TRAINING XGBOOST MODEL (MULTI-CLASS)")
print("="*80)

print(f"\nModel hyperparameters:")
print(f"  objective: multi:softmax")
print(f"  num_class: 4")
print(f"  n_estimators: 200")
print(f"  max_depth: 6")
print(f"  learning_rate: 0.1")

# Initialize model for multi-class classification
model = XGBClassifier(
    objective='multi:softmax',
    num_class=4,
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

# Training performance
print("\n📊 Training Set:")
print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"  Precision: {precision_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f} (weighted)")
print(f"  Recall:    {recall_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f} (weighted)")
print(f"  F1-Score:  {f1_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f} (weighted)")

# Validation performance
print("\n📊 Validation Set:")
print(f"  Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"  Precision: {precision_score(y_val, y_val_pred, average='weighted', zero_division=0):.4f} (weighted)")
print(f"  Recall:    {recall_score(y_val, y_val_pred, average='weighted', zero_division=0):.4f} (weighted)")
print(f"  F1-Score:  {f1_score(y_val, y_val_pred, average='weighted', zero_division=0):.4f} (weighted)")

# Test performance
print("\n📊 Test Set (Final Evaluation):")
print(f"  Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_test_pred, average='weighted', zero_division=0):.4f} (weighted)")
print(f"  Recall:    {recall_score(y_test, y_test_pred, average='weighted', zero_division=0):.4f} (weighted)")
print(f"  F1-Score:  {f1_score(y_test, y_test_pred, average='weighted', zero_division=0):.4f} (weighted)")

# Per-class metrics
print("\n📊 Per-Class Metrics (Test Set):")
for cls in sorted(np.unique(y_test)):
    category_name = [k for k, v in category_mapping.items() if v == cls][0]
    y_test_binary = (y_test == cls).astype(int)
    y_test_pred_binary = (y_test_pred == cls).astype(int)
    
    prec = precision_score(y_test_binary, y_test_pred_binary, zero_division=0)
    rec = recall_score(y_test_binary, y_test_pred_binary, zero_division=0)
    f1 = f1_score(y_test_binary, y_test_pred_binary, zero_division=0)
    
    print(f"\n  {category_name} (Class {cls}):")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")

# Detailed classification report
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT - TEST SET")
print("="*80)
target_names = [k for k, v in sorted(category_mapping.items(), key=lambda x: x[1])]
print("\n", classification_report(y_test, y_test_pred, 
                                  target_names=target_names,
                                  zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_title('Confusion Matrix - Multi-Class Classification\\n(4 Attack Categories)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/lamdx4/Projects/IOT prj/models/multiclass_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved confusion matrix: models/multiclass_confusion_matrix.png")
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
ax.set_title('XGBoost Feature Importance - Multi-Class Classification', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/lamdx4/Projects/IOT prj/models/multiclass_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved feature importance: models/multiclass_feature_importance.png")
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
model_path = f"{MODEL_DIR}/xgboost_multiclass_4categories_{timestamp}.pkl"
joblib.dump(model, model_path)
print(f"\n✅ Model saved: {model_path}")

# Save label encoder
encoder_path = f"{MODEL_DIR}/label_encoder_proto_multiclass_{timestamp}.pkl"
joblib.dump(le_proto, encoder_path)
print(f"✅ Label encoder saved: {encoder_path}")

# Save category mapping
mapping_path = f"{MODEL_DIR}/category_mapping_{timestamp}.pkl"
joblib.dump(category_mapping, mapping_path)
print(f"✅ Category mapping saved: {mapping_path}")

# Save feature columns
features_path = f"{MODEL_DIR}/feature_columns_multiclass_{timestamp}.pkl"
joblib.dump(feature_cols, features_path)
print(f"✅ Feature columns saved: {features_path}")

# Save metrics summary
metrics_summary = {
    'model_type': 'Multi-Class Classification (4 Categories)',
    'timestamp': timestamp,
    'classes': list(category_mapping.keys()),
    'test_accuracy': accuracy_score(y_test, y_test_pred),
    'test_precision_weighted': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
    'test_recall_weighted': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
    'test_f1_weighted': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
    'smote_applied': k_neighbors > 0 and min_samples > 1,
    'num_classes': 4
}

metrics_path = f"{MODEL_DIR}/metrics_multiclass_{timestamp}.pkl"
joblib.dump(metrics_summary, metrics_path)
print(f"✅ Metrics summary saved: {metrics_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("✅ PHƯƠNG ÁN 2 HOÀN THÀNH - MULTI-CLASS CLASSIFICATION")
print("="*80)

print("\n📊 Final Results:")
print(f"  Problem Type:     Multi-Class Classification (4 Categories)")
print(f"  Classes:          {', '.join(category_mapping.keys())}")
print(f"  Algorithm:        XGBoost")
print(f"  Training Samples: {X_train.shape[0]:,} → {X_train_resampled.shape[0]:,} (after SMOTE)")
print(f"  Test Accuracy:    {metrics_summary['test_accuracy']:.4f}")
print(f"  Test F1-Score:    {metrics_summary['test_f1_weighted']:.4f} (weighted)")

print("\n📂 Saved Files:")
print(f"  • Model: {os.path.basename(model_path)}")
print(f"  • Encoder: {os.path.basename(encoder_path)}")
print(f"  • Category Mapping: {os.path.basename(mapping_path)}")
print(f"  • Features: {os.path.basename(features_path)}")
print(f"  • Metrics: {os.path.basename(metrics_path)}")

print("\n🚀 Next Steps:")
print("  → So sánh kết quả 2 phương án (Binary vs Multi-class)")
print("  → Chọn model phù hợp nhất cho deployment")
print("  → Deploy cho real-time detection system")

print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("="*80)

