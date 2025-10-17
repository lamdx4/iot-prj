"""
Step 4: Test Trained Two-Stage Hierarchical Model
==================================================

Load trained models and test on various data:
- Test set from 5% dataset
- Sample predictions with confidence scores
- Detailed error analysis

Author: Lambda Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import joblib
import json
import glob
import os
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

print("="*80)
print("STEP 4: TEST TWO-STAGE HIERARCHICAL MODEL")
print("="*80)

# =============================================================================
# 1. LOAD MODELS
# =============================================================================

MODEL_DIR = "/home/lamdx4/Projects/IOT prj/models/full_dataset"
TEST_PATH = "/home/lamdx4/Projects/IOT prj/Data/Dataset/5%/10-best features/split/UNSW_2018_IoT_Botnet_Final_10_Best_Testing.csv"

print("\n" + "="*80)
print("1. LOADING MODELS")
print("="*80)

# Find latest models
stage1_models = sorted(glob.glob(os.path.join(MODEL_DIR, "stage1_binary_*.pkl")))
stage2_models = sorted(glob.glob(os.path.join(MODEL_DIR, "stage2_multiclass_*.pkl")))
encoder_files = sorted(glob.glob(os.path.join(MODEL_DIR, "label_encoder_*.pkl")))
mapping_files = sorted(glob.glob(os.path.join(MODEL_DIR, "attack_mapping_*.pkl")))
feature_files = sorted(glob.glob(os.path.join(MODEL_DIR, "feature_columns_*.pkl")))

if not stage1_models or not stage2_models:
    print("\n❌ No trained models found!")
    print(f"   Please run 03_train_hierarchical.py first")
    exit(1)

# Use latest
model_s1_path = stage1_models[-1]
model_s2_path = stage2_models[-1]
encoder_path = encoder_files[-1]
mapping_path = mapping_files[-1]
features_path = feature_files[-1]

print(f"\n📂 Loading models:")
print(f"   Stage 1: {os.path.basename(model_s1_path)}")
print(f"   Stage 2: {os.path.basename(model_s2_path)}")

model_stage1 = joblib.load(model_s1_path)
model_stage2 = joblib.load(model_s2_path)
le_proto = joblib.load(encoder_path)
attack_type_mapping = joblib.load(mapping_path)
feature_cols = joblib.load(features_path)

print(f"\n✅ Models loaded successfully!")
print(f"   Features: {len(feature_cols)}")
print(f"   Attack types: {list(attack_type_mapping.keys())}")

# =============================================================================
# 2. LOAD TEST DATA
# =============================================================================

print("\n" + "="*80)
print("2. LOADING TEST DATA")
print("="*80)

df_test = pd.read_csv(TEST_PATH)
print(f"\n✅ Test data loaded: {len(df_test):,} samples")

print(f"\n📊 Test Distribution:")
for cat, count in df_test['category'].value_counts().items():
    pct = count / len(df_test) * 100
    print(f"   {cat:15s}: {count:,} ({pct:.2f}%)")

# =============================================================================
# 3. PREPROCESS TEST DATA
# =============================================================================

print("\n" + "="*80)
print("3. PREPROCESSING")
print("="*80)

# Handle missing values
print(f"\n🔧 Handling missing values...")
numeric_cols = df_test.select_dtypes(include=['number']).columns
df_test[numeric_cols] = df_test[numeric_cols].fillna(df_test[numeric_cols].median())

# Encode proto
print(f"🔧 Encoding 'proto' feature...")
df_test['proto'] = le_proto.transform(df_test['proto'].astype(str))

# Create binary target
df_test['is_attack'] = (df_test['category'] != 'Normal').astype(int)

print(f"✅ Preprocessing completed")

# =============================================================================
# 4. PREDICTIONS
# =============================================================================

print("\n" + "="*80)
print("4. MAKING PREDICTIONS")
print("="*80)

X_test = df_test[feature_cols]
y_test_binary = df_test['is_attack']
y_test_category = df_test['category']

# Stage 1: Binary predictions
print(f"\n🔮 Stage 1: Binary Classification...")
y_pred_binary = model_stage1.predict(X_test)
y_pred_binary_proba = model_stage1.predict_proba(X_test)[:, 1]

stage1_acc = accuracy_score(y_test_binary, y_pred_binary)
print(f"   Accuracy: {stage1_acc:.4f}")

# Stage 2: Multi-class (only for attacks)
print(f"\n🔮 Stage 2: Multi-class Classification (attacks only)...")
attack_mask = df_test['is_attack'] == 1
X_test_attacks = X_test[attack_mask]
y_test_attacks = df_test[attack_mask]['category'].map(attack_type_mapping)

y_pred_attacks = model_stage2.predict(X_test_attacks)
stage2_acc = accuracy_score(y_test_attacks, y_pred_attacks)
print(f"   Accuracy: {stage2_acc:.4f}")

# Combined pipeline
print(f"\n🔮 Combined Pipeline...")

def predict_two_stage(sample):
    """Two-stage prediction"""
    # Stage 1
    is_attack = model_stage1.predict(sample)[0]
    attack_proba = model_stage1.predict_proba(sample)[0, 1]
    
    if is_attack == 0:
        return 'Normal', 1 - attack_proba
    else:
        # Stage 2
        attack_type = model_stage2.predict(sample)[0]
        attack_type_proba = model_stage2.predict_proba(sample)[0]
        attack_name = [k for k, v in attack_type_mapping.items() if v == attack_type][0]
        confidence = attack_type_proba[attack_type]
        return attack_name, confidence

final_predictions = []
final_confidences = []

for i in range(len(X_test)):
    sample = X_test.iloc[i:i+1]
    prediction, confidence = predict_two_stage(sample)
    final_predictions.append(prediction)
    final_confidences.append(confidence)

overall_acc = accuracy_score(y_test_category, final_predictions)
print(f"   Overall Accuracy: {overall_acc:.4f}")

# =============================================================================
# 5. DETAILED EVALUATION
# =============================================================================

print("\n" + "="*80)
print("5. DETAILED EVALUATION")
print("="*80)

# Classification report
all_categories = ['Normal', 'DDoS', 'DoS', 'Reconnaissance']
print(f"\n📊 Classification Report:")
print("="*80)
print(classification_report(y_test_category, final_predictions,
                          labels=all_categories,
                          zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test_category, final_predictions, labels=all_categories)
print(f"\n📊 Confusion Matrix:")
print("="*80)
print(f"{'':15s} " + " ".join([f"{c:10s}" for c in all_categories]))
print("-"*80)
for i, true_cat in enumerate(all_categories):
    row = f"{true_cat:15s} " + " ".join([f"{cm[i,j]:10d}" for j in range(len(all_categories))])
    print(row)

# Per-class accuracy
print(f"\n📊 Per-Class Accuracy:")
print("="*80)
for i, cat in enumerate(all_categories):
    if cm[i].sum() > 0:
        class_acc = cm[i, i] / cm[i].sum()
        total = cm[i].sum()
        correct = cm[i, i]
        print(f"   {cat:15s}: {class_acc:.4f} ({correct}/{total})")

# =============================================================================
# 6. ERROR ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("6. ERROR ANALYSIS")
print("="*80)

# Find misclassified samples
errors = []
for i in range(len(y_test_category)):
    if final_predictions[i] != y_test_category.iloc[i]:
        errors.append({
            'index': i,
            'true': y_test_category.iloc[i],
            'predicted': final_predictions[i],
            'confidence': final_confidences[i]
        })

print(f"\n❌ Total errors: {len(errors)} / {len(y_test_category)} ({len(errors)/len(y_test_category)*100:.2f}%)")

if errors:
    print(f"\n❌ Error breakdown:")
    error_types = {}
    for err in errors:
        key = f"{err['true']} → {err['predicted']}"
        error_types[key] = error_types.get(key, 0) + 1
    
    for err_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   {err_type:30s}: {count:4d} errors")
    
    # Show first 5 errors
    print(f"\n❌ Sample Errors (first 5):")
    for i, err in enumerate(errors[:5], 1):
        print(f"\n   Error {i}:")
        print(f"      True:       {err['true']}")
        print(f"      Predicted:  {err['predicted']}")
        print(f"      Confidence: {err['confidence']:.4f}")

# =============================================================================
# 7. SAMPLE PREDICTIONS
# =============================================================================

print("\n" + "="*80)
print("7. SAMPLE PREDICTIONS")
print("="*80)

print(f"\n🔮 Testing with random samples:")

# Sample from each category
samples_per_cat = 2

for cat in all_categories:
    cat_indices = df_test[df_test['category'] == cat].index
    
    if len(cat_indices) == 0:
        print(f"\n   ⚠️  No {cat} samples in test set")
        continue
    
    sample_indices = np.random.choice(cat_indices, min(samples_per_cat, len(cat_indices)), replace=False)
    
    print(f"\n   Category: {cat}")
    print("   " + "-"*70)
    
    for idx in sample_indices:
        sample = X_test.iloc[idx:idx+1]
        true_label = df_test.iloc[idx]['category']
        
        prediction, confidence = predict_two_stage(sample)
        
        status = "✓" if prediction == true_label else "✗"
        print(f"      {status} True: {true_label:15s} | Predicted: {prediction:15s} | Conf: {confidence:.4f}")

# =============================================================================
# 8. SAVE TEST RESULTS
# =============================================================================

print("\n" + "="*80)
print("8. SAVING TEST RESULTS")
print("="*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

test_results = {
    'timestamp': timestamp,
    'model_files': {
        'stage1': os.path.basename(model_s1_path),
        'stage2': os.path.basename(model_s2_path)
    },
    'test_data': {
        'file': TEST_PATH,
        'num_samples': len(df_test),
        'distribution': df_test['category'].value_counts().to_dict()
    },
    'performance': {
        'stage1_accuracy': float(stage1_acc),
        'stage2_accuracy': float(stage2_acc),
        'overall_accuracy': float(overall_acc)
    },
    'confusion_matrix': cm.tolist(),
    'errors': {
        'total': len(errors),
        'percentage': float(len(errors) / len(y_test_category) * 100),
        'breakdown': error_types if errors else {}
    }
}

results_path = os.path.join(MODEL_DIR, f"test_results_{timestamp}.json")
with open(results_path, 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\n✅ Test results saved: {os.path.basename(results_path)}")

# =============================================================================
# 9. SUMMARY
# =============================================================================

print("\n" + "="*80)
print("✅ TESTING COMPLETED!")
print("="*80)

print(f"\n📊 Summary:")
print(f"   Test samples: {len(df_test):,}")
print(f"   Stage 1 Accuracy: {stage1_acc:.4f}")
print(f"   Stage 2 Accuracy: {stage2_acc:.4f}")
print(f"   Overall Accuracy: {overall_acc:.4f}")
print(f"   Total errors: {len(errors)}")

print(f"\n📂 Results saved to:")
print(f"   {results_path}")

print("\n" + "="*80)
print("Model is ready for deployment!")
print("="*80)


