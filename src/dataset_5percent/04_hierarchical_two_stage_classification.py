"""
Phương án 3: Two-Stage Hierarchical Classification
===================================================

Đề tài: Phát hiện tấn công IoT - Hierarchical Approach

Stage 1: Binary Classification (Attack vs Normal)
  → Phát hiện có tấn công hay không?
  → Target: 0=Normal, 1=Attack
  → Xử lý EXTREME imbalance với SMOTE

Stage 2: Multi-class Classification (Only Attacks)
  → Nếu là Attack → Phân loại loại nào?
  → Target: 0=DDoS, 1=DoS, 2=Reconnaissance
  → Chỉ train trên Attack samples (không có Normal)

Ưu điểm:
  ✓ Giải quyết vấn đề Normal quá ít (3 samples)
  ✓ Stage 2 balanced hơn (không có Normal)
  ✓ Đúng với mọi đề bài
  ✓ Kết hợp tốt nhất của cả 2 phương án

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Imbalanced Data
from imblearn.over_sampling import SMOTE

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
print("PHƯƠNG ÁN 3: TWO-STAGE HIERARCHICAL CLASSIFICATION")
print("="*80)
print("\nStage 1: Attack vs Normal Detection")
print("Stage 2: Attack Type Classification (DDoS, DoS, Reconnaissance)")
print("\n" + "="*80)
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

# Check distribution
print("\nOriginal Category Distribution (Training):")
print(df_train['category'].value_counts())

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*80)
print("3. FEATURE ENGINEERING")
print("="*80)

# Columns to drop
cols_to_drop = ['saddr', 'sport', 'daddr', 'dport', 'attack', 'category', 'subcategory']

# Get feature columns
feature_cols = [col for col in df_train.columns if col not in cols_to_drop]

print(f"\nTotal features: {len(feature_cols)}")
print("Features:", feature_cols)

# Encode categorical features (proto)
le_proto = LabelEncoder()
df_train_processed = df_train.copy()
df_test_processed = df_test.copy()

df_train_processed['proto'] = le_proto.fit_transform(df_train_processed['proto'])
df_test_processed['proto'] = le_proto.transform(df_test_processed['proto'])

print(f"\n✅ Protocol encoding:")
for i, proto in enumerate(le_proto.classes_):
    print(f"   {proto:10s} → {i}")

# =============================================================================
# STAGE 1: BINARY CLASSIFICATION (ATTACK vs NORMAL)
# =============================================================================

print("\n" + "="*80)
print("STAGE 1: BINARY CLASSIFICATION - ATTACK vs NORMAL")
print("="*80)

# Create binary target
df_train_processed['is_attack'] = (df_train_processed['category'] != 'Normal').astype(int)
df_test_processed['is_attack'] = (df_test_processed['category'] != 'Normal').astype(int)

print("\nStage 1 Target Distribution:")
print("\nTraining Set:")
stage1_train_dist = df_train_processed['is_attack'].value_counts().sort_index()
for cls, count in stage1_train_dist.items():
    label = "Normal" if cls == 0 else "Attack"
    print(f"  {label:10s} ({cls}): {count:,} ({count/len(df_train_processed)*100:.4f}%)")

if 0 in stage1_train_dist.index and 1 in stage1_train_dist.index:
    imbalance = stage1_train_dist[1] / stage1_train_dist[0]
    print(f"\n⚠️  Imbalance Ratio: {imbalance:.0f}:1 (Attack:Normal)")

# Prepare Stage 1 data
X_train_full_s1 = df_train_processed[feature_cols]
y_train_full_s1 = df_train_processed['is_attack']

X_test_s1 = df_test_processed[feature_cols]
y_test_s1 = df_test_processed['is_attack']

# Split train into train and validation
X_train_s1, X_val_s1, y_train_s1, y_val_s1 = train_test_split(
    X_train_full_s1, y_train_full_s1,
    test_size=0.15,
    random_state=42,
    stratify=y_train_full_s1
)

print(f"\nStage 1 Data Split:")
print(f"  Training:   {X_train_s1.shape[0]:,} samples")
print(f"  Validation: {X_val_s1.shape[0]:,} samples")
print(f"  Test:       {X_test_s1.shape[0]:,} samples")

# Handle imbalance with SMOTE
print("\n🔧 Applying SMOTE for Stage 1...")
k_neighbors_s1 = min((y_train_s1 == 0).sum() - 1, 5)

if k_neighbors_s1 > 0 and (y_train_s1 == 0).sum() > 1:
    smote_s1 = SMOTE(
        sampling_strategy=0.1,  # Oversample Normal to 10% of Attack
        random_state=42,
        k_neighbors=k_neighbors_s1
    )
    X_train_s1_resampled, y_train_s1_resampled = smote_s1.fit_resample(X_train_s1, y_train_s1)
    
    print(f"✅ SMOTE applied (k_neighbors={k_neighbors_s1})")
    print(f"   Before: {X_train_s1.shape[0]:,} → After: {X_train_s1_resampled.shape[0]:,}")
    print(f"   Normal: {(y_train_s1_resampled == 0).sum():,} ({(y_train_s1_resampled == 0).sum()/len(y_train_s1_resampled)*100:.2f}%)")
    print(f"   Attack: {(y_train_s1_resampled == 1).sum():,} ({(y_train_s1_resampled == 1).sum()/len(y_train_s1_resampled)*100:.2f}%)")
else:
    X_train_s1_resampled = X_train_s1
    y_train_s1_resampled = y_train_s1
    print("⚠️  SMOTE skipped - using class weights")

# Train Stage 1 Model
print("\n🚀 Training Stage 1 Model (Binary Classifier)...")

scale_pos_weight_s1 = (y_train_s1_resampled == 0).sum() / (y_train_s1_resampled == 1).sum() if (y_train_s1_resampled == 1).sum() > 0 else 1.0

model_stage1 = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight_s1,
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

print(f"✅ Stage 1 training completed!")
print(f"   Best iteration: {model_stage1.best_iteration}")
print(f"   Best score: {model_stage1.best_score:.4f}")

# Evaluate Stage 1
y_test_s1_pred = model_stage1.predict(X_test_s1)
y_test_s1_proba = model_stage1.predict_proba(X_test_s1)[:, 1]

print("\n📊 Stage 1 Performance (Test Set):")
print(f"  Accuracy:  {accuracy_score(y_test_s1, y_test_s1_pred):.4f}")
print(f"  Precision: {precision_score(y_test_s1, y_test_s1_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test_s1, y_test_s1_pred, zero_division=0):.4f}")
print(f"  F1-Score:  {f1_score(y_test_s1, y_test_s1_pred, zero_division=0):.4f}")
if len(np.unique(y_test_s1)) > 1:
    print(f"  ROC-AUC:   {roc_auc_score(y_test_s1, y_test_s1_proba):.4f}")

# =============================================================================
# STAGE 2: MULTI-CLASS CLASSIFICATION (ATTACK TYPES ONLY)
# =============================================================================

print("\n" + "="*80)
print("STAGE 2: MULTI-CLASS - ATTACK TYPE CLASSIFICATION")
print("="*80)

# Filter only Attack samples for Stage 2
df_train_attacks = df_train_processed[df_train_processed['is_attack'] == 1].copy()
df_test_attacks = df_test_processed[df_test_processed['is_attack'] == 1].copy()

print(f"\nStage 2 focuses ONLY on Attack samples:")
print(f"  Training Attacks: {len(df_train_attacks):,}")
print(f"  Testing Attacks:  {len(df_test_attacks):,}")

# Create attack type mapping (exclude Normal)
attack_type_mapping = {
    'DDoS': 0,
    'DoS': 1,
    'Reconnaissance': 2
}

df_train_attacks['attack_type'] = df_train_attacks['category'].map(attack_type_mapping)
df_test_attacks['attack_type'] = df_test_attacks['category'].map(attack_type_mapping)

print("\nStage 2 Target Distribution (Training):")
stage2_dist = df_train_attacks['attack_type'].value_counts().sort_index()
for cls, count in stage2_dist.items():
    attack_name = [k for k, v in attack_type_mapping.items() if v == cls][0]
    print(f"  {attack_name:15s} ({cls}): {count:,} ({count/len(df_train_attacks)*100:.2f}%)")

# Prepare Stage 2 data
X_train_full_s2 = df_train_attacks[feature_cols]
y_train_full_s2 = df_train_attacks['attack_type']

X_test_s2 = df_test_attacks[feature_cols]
y_test_s2 = df_test_attacks['attack_type']

# Split for Stage 2
X_train_s2, X_val_s2, y_train_s2, y_val_s2 = train_test_split(
    X_train_full_s2, y_train_full_s2,
    test_size=0.15,
    random_state=42,
    stratify=y_train_full_s2
)

print(f"\nStage 2 Data Split:")
print(f"  Training:   {X_train_s2.shape[0]:,} samples")
print(f"  Validation: {X_val_s2.shape[0]:,} samples")
print(f"  Test:       {X_test_s2.shape[0]:,} samples")

# Handle imbalance in Stage 2
print("\n🔧 Applying SMOTE for Stage 2 (Reconnaissance is minority)...")
min_samples_s2 = y_train_s2.value_counts().min()
k_neighbors_s2 = min(min_samples_s2 - 1, 5)

if k_neighbors_s2 > 0 and min_samples_s2 > 1:
    smote_s2 = SMOTE(
        sampling_strategy='not majority',
        random_state=42,
        k_neighbors=k_neighbors_s2
    )
    X_train_s2_resampled, y_train_s2_resampled = smote_s2.fit_resample(X_train_s2, y_train_s2)
    
    print(f"✅ SMOTE applied (k_neighbors={k_neighbors_s2})")
    print(f"   Before: {X_train_s2.shape[0]:,} → After: {X_train_s2_resampled.shape[0]:,}")
    for cls in sorted(np.unique(y_train_s2_resampled)):
        attack_name = [k for k, v in attack_type_mapping.items() if v == cls][0]
        count = (y_train_s2_resampled == cls).sum()
        print(f"   {attack_name:15s}: {count:,} ({count/len(y_train_s2_resampled)*100:.2f}%)")
else:
    X_train_s2_resampled = X_train_s2
    y_train_s2_resampled = y_train_s2
    print("⚠️  SMOTE skipped - using original data")

# Train Stage 2 Model
print("\n🚀 Training Stage 2 Model (Multi-class Classifier)...")

model_stage2 = XGBClassifier(
    objective='multi:softmax',
    num_class=3,  # DDoS, DoS, Reconnaissance
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

print(f"✅ Stage 2 training completed!")
print(f"   Best iteration: {model_stage2.best_iteration}")
print(f"   Best score: {model_stage2.best_score:.4f}")

# Evaluate Stage 2
y_test_s2_pred = model_stage2.predict(X_test_s2)

print("\n📊 Stage 2 Performance (Test Set - Attacks Only):")
print(f"  Accuracy:  {accuracy_score(y_test_s2, y_test_s2_pred):.4f}")
print(f"  Precision: {precision_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0):.4f} (weighted)")
print(f"  Recall:    {recall_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0):.4f} (weighted)")
print(f"  F1-Score:  {f1_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0):.4f} (weighted)")

# =============================================================================
# COMBINED EVALUATION: TWO-STAGE PIPELINE
# =============================================================================

print("\n" + "="*80)
print("COMBINED PIPELINE EVALUATION")
print("="*80)

# Full pipeline prediction
print("\n🔄 Running full two-stage pipeline on test set...")

# Stage 1: Predict Attack vs Normal
y_test_stage1_pred = model_stage1.predict(X_test_s1)

# Stage 2: For predicted attacks, classify attack type
final_predictions = []
final_true_labels = []

for i in range(len(X_test_s1)):
    true_category = df_test_processed.iloc[i]['category']
    
    # Stage 1 prediction
    is_attack = y_test_stage1_pred[i]
    
    if is_attack == 0:
        # Predicted as Normal
        final_pred = 'Normal'
    else:
        # Predicted as Attack → go to Stage 2
        sample = X_test_s1.iloc[i:i+1]
        attack_type_pred = model_stage2.predict(sample)[0]
        attack_name = [k for k, v in attack_type_mapping.items() if v == attack_type_pred][0]
        final_pred = attack_name
    
    final_predictions.append(final_pred)
    final_true_labels.append(true_category)

# Convert to array
final_predictions = np.array(final_predictions)
final_true_labels = np.array(final_true_labels)

# Overall accuracy
overall_accuracy = accuracy_score(final_true_labels, final_predictions)
print(f"\n📊 Overall Two-Stage Pipeline Performance:")
print(f"  Overall Accuracy: {overall_accuracy:.4f}")

# Detailed report
all_categories = ['Normal', 'DDoS', 'DoS', 'Reconnaissance']
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT - FULL PIPELINE")
print("="*80)
print("\n", classification_report(final_true_labels, final_predictions,
                                  labels=all_categories,
                                  zero_division=0))

# Confusion Matrix
cm_full = confusion_matrix(final_true_labels, final_predictions, labels=all_categories)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues',
            xticklabels=all_categories,
            yticklabels=all_categories,
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_title('Confusion Matrix - Two-Stage Hierarchical Classification\\n(Full Pipeline)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/lamdx4/Projects/IOT prj/models/hierarchical_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: models/hierarchical_confusion_matrix.png")
plt.show()

# =============================================================================
# VISUALIZATION: TWO-STAGE ARCHITECTURE
# =============================================================================

print("\n" + "="*80)
print("VISUALIZATIONS")
print("="*80)

# Architecture diagram (text-based)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Stage 1 Confusion Matrix
cm_s1 = confusion_matrix(y_test_s1, y_test_s1_pred)
sns.heatmap(cm_s1, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            cbar_kws={'label': 'Count'}, ax=axes[0, 0])
axes[0, 0].set_title('Stage 1: Binary Classification\\n(Attack vs Normal)', fontweight='bold')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# Stage 2 Confusion Matrix
cm_s2 = confusion_matrix(y_test_s2, y_test_s2_pred)
attack_names = [k for k, v in sorted(attack_type_mapping.items(), key=lambda x: x[1])]
sns.heatmap(cm_s2, annot=True, fmt='d', cmap='Oranges',
            xticklabels=attack_names,
            yticklabels=attack_names,
            cbar_kws={'label': 'Count'}, ax=axes[0, 1])
axes[0, 1].set_title('Stage 2: Multi-class Classification\\n(Attack Types)', fontweight='bold')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')

# Stage 1 ROC Curve
if len(np.unique(y_test_s1)) > 1:
    fpr, tpr, _ = roc_curve(y_test_s1, y_test_s1_proba)
    roc_auc = roc_auc_score(y_test_s1, y_test_s1_proba)
    axes[1, 0].plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axes[1, 0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('Stage 1: ROC Curve', fontweight='bold')
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(alpha=0.3)

# Feature Importance (Stage 1)
feat_imp_s1 = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_stage1.feature_importances_
}).sort_values('importance', ascending=True).tail(10)

axes[1, 1].barh(range(len(feat_imp_s1)), feat_imp_s1['importance'], color='teal', edgecolor='black')
axes[1, 1].set_yticks(range(len(feat_imp_s1)))
axes[1, 1].set_yticklabels(feat_imp_s1['feature'])
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('Top 10 Features (Stage 1)', fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/lamdx4/Projects/IOT prj/models/hierarchical_overview.png', dpi=300, bbox_inches='tight')
print("✅ Saved: models/hierarchical_overview.png")
plt.show()

# =============================================================================
# SAVE MODELS
# =============================================================================

print("\n" + "="*80)
print("SAVING MODELS & ARTIFACTS")
print("="*80)

MODEL_DIR = "/home/lamdx4/Projects/IOT prj/models"
os.makedirs(MODEL_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save Stage 1 model
model_s1_path = f"{MODEL_DIR}/xgboost_stage1_binary_{timestamp}.pkl"
joblib.dump(model_stage1, model_s1_path)
print(f"\n✅ Stage 1 model saved: {model_s1_path}")

# Save Stage 2 model
model_s2_path = f"{MODEL_DIR}/xgboost_stage2_multiclass_{timestamp}.pkl"
joblib.dump(model_stage2, model_s2_path)
print(f"✅ Stage 2 model saved: {model_s2_path}")

# Save encoders and mappings
encoder_path = f"{MODEL_DIR}/label_encoder_proto_hierarchical_{timestamp}.pkl"
joblib.dump(le_proto, encoder_path)
print(f"✅ Label encoder saved: {encoder_path}")

mapping_path = f"{MODEL_DIR}/attack_type_mapping_{timestamp}.pkl"
joblib.dump(attack_type_mapping, mapping_path)
print(f"✅ Attack type mapping saved: {mapping_path}")

features_path = f"{MODEL_DIR}/feature_columns_hierarchical_{timestamp}.pkl"
joblib.dump(feature_cols, features_path)
print(f"✅ Feature columns saved: {features_path}")

# Save metrics
metrics = {
    'model_type': 'Two-Stage Hierarchical Classification',
    'timestamp': timestamp,
    'stage1': {
        'accuracy': accuracy_score(y_test_s1, y_test_s1_pred),
        'precision': precision_score(y_test_s1, y_test_s1_pred, zero_division=0),
        'recall': recall_score(y_test_s1, y_test_s1_pred, zero_division=0),
        'f1': f1_score(y_test_s1, y_test_s1_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test_s1, y_test_s1_proba) if len(np.unique(y_test_s1)) > 1 else None
    },
    'stage2': {
        'accuracy': accuracy_score(y_test_s2, y_test_s2_pred),
        'precision_weighted': precision_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test_s2, y_test_s2_pred, average='weighted', zero_division=0)
    },
    'overall': {
        'accuracy': overall_accuracy
    }
}

metrics_path = f"{MODEL_DIR}/metrics_hierarchical_{timestamp}.pkl"
joblib.dump(metrics, metrics_path)
print(f"✅ Metrics saved: {metrics_path}")

# =============================================================================
# CREATE INFERENCE FUNCTION
# =============================================================================

def predict_two_stage(sample, model_s1, model_s2, attack_mapping, feature_cols):
    """
    Two-stage prediction function
    
    Args:
        sample: DataFrame with features
        model_s1: Stage 1 binary classifier
        model_s2: Stage 2 multi-class classifier
        attack_mapping: Dict mapping attack types to labels
        feature_cols: List of feature column names
    
    Returns:
        prediction: 'Normal' or attack type name
        confidence: Probability/confidence score
    """
    # Stage 1: Attack vs Normal
    sample_features = sample[feature_cols]
    is_attack = model_s1.predict(sample_features)[0]
    attack_proba = model_s1.predict_proba(sample_features)[0, 1]
    
    if is_attack == 0:
        return 'Normal', 1 - attack_proba
    else:
        # Stage 2: Attack type classification
        attack_type_idx = model_s2.predict(sample_features)[0]
        attack_type_proba = model_s2.predict_proba(sample_features)[0]
        
        # Get attack name
        attack_name = [k for k, v in attack_mapping.items() if v == attack_type_idx][0]
        confidence = attack_type_proba[attack_type_idx]
        
        return attack_name, confidence

# Save inference function
inference_path = f"{MODEL_DIR}/predict_function_hierarchical_{timestamp}.pkl"
joblib.dump(predict_two_stage, inference_path)
print(f"✅ Inference function saved: {inference_path}")

# =============================================================================
# DEMO PREDICTION
# =============================================================================

print("\n" + "="*80)
print("DEMO: TWO-STAGE PREDICTION")
print("="*80)

# Test with a few samples
print("\nTesting pipeline with sample predictions:")
for i in range(min(5, len(X_test_s1))):
    sample = X_test_s1.iloc[i:i+1]
    true_label = df_test_processed.iloc[i]['category']
    
    prediction, confidence = predict_two_stage(
        sample, model_stage1, model_stage2, attack_type_mapping, feature_cols
    )
    
    print(f"\nSample {i+1}:")
    print(f"  True Label:  {true_label}")
    print(f"  Predicted:   {prediction}")
    print(f"  Confidence:  {confidence:.4f}")
    print(f"  Correct:     {'✓' if prediction == true_label else '✗'}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("✅ TWO-STAGE HIERARCHICAL CLASSIFICATION COMPLETED!")
print("="*80)

print("\n📊 Architecture Summary:")
print("  Stage 1: Binary Classifier (Attack vs Normal)")
print(f"    → Accuracy: {metrics['stage1']['accuracy']:.4f}")
print(f"    → F1-Score: {metrics['stage1']['f1']:.4f}")
print(f"    → ROC-AUC:  {metrics['stage1']['roc_auc']:.4f if metrics['stage1']['roc_auc'] else 'N/A'}")

print("\n  Stage 2: Multi-class Classifier (DDoS, DoS, Reconnaissance)")
print(f"    → Accuracy: {metrics['stage2']['accuracy']:.4f}")
print(f"    → F1-Score: {metrics['stage2']['f1_weighted']:.4f} (weighted)")

print(f"\n  Overall Pipeline Accuracy: {metrics['overall']['accuracy']:.4f}")

print("\n📂 Saved Files:")
print(f"  • Stage 1 Model: {os.path.basename(model_s1_path)}")
print(f"  • Stage 2 Model: {os.path.basename(model_s2_path)}")
print(f"  • Metrics: {os.path.basename(metrics_path)}")
print(f"  • Inference Function: {os.path.basename(inference_path)}")

print("\n✨ Advantages:")
print("  ✓ Xử lý EXTREME imbalance (3 Normal vs 29K Attack)")
print("  ✓ Stage 2 không bị ảnh hưởng bởi Normal samples")
print("  ✓ Kết hợp tốt nhất của Binary và Multi-class")
print("  ✓ Phù hợp với mọi đề bài: 'Phát hiện' + 'Phân loại'")
print("  ✓ Dễ deploy và maintain")

print("\n🚀 Usage:")
print("  1. Load models:")
print(f"     model_s1 = joblib.load('{os.path.basename(model_s1_path)}')")
print(f"     model_s2 = joblib.load('{os.path.basename(model_s2_path)}')")
print("  2. Predict:")
print("     prediction, confidence = predict_two_stage(sample, model_s1, model_s2, ...)")

print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("="*80)

