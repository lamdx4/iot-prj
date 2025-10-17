"""
Step 4: Comprehensive Model Evaluation
=======================================

Detailed evaluation with:
- Confusion matrices (detailed numbers)
- Per-class metrics (precision, recall, F1 for each attack type)
- Training statistics
- Inference time analysis
- JSON export for easy review
- Visualization support

Author: Lambda Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib

print("="*80)
print("STEP 4: COMPREHENSIVE MODEL EVALUATION")
print("="*80)

# Auto-detect project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))

# Paths
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(PROJECT_ROOT, "models/full_dataset"))
EVAL_DIR = os.getenv('EVAL_DIR', os.path.join(PROJECT_ROOT, "results/evaluation"))
BATCH_DIR = os.getenv('BATCH_DIR', os.path.join(PROJECT_ROOT, "Data/Dataset/merged_batches"))

os.makedirs(EVAL_DIR, exist_ok=True)

print(f"\n📂 Paths:")
print(f"   Model dir: {MODEL_DIR}")
print(f"   Eval dir:  {EVAL_DIR}")
print(f"   Batch dir: {BATCH_DIR}")

# ============================================================================
# 1. LOAD LATEST MODELS
# ============================================================================

print("\n" + "="*80)
print("1. LOADING MODELS")
print("="*80)

# Find latest model files
import glob
model_files = glob.glob(os.path.join(MODEL_DIR, "stage1_*.pkl"))
if not model_files:
    print("❌ No models found! Please run training first.")
    print(f"   Expected directory: {MODEL_DIR}")
    print(f"   Files in directory:")
    try:
        for f in os.listdir(MODEL_DIR):
            print(f"      {f}")
    except:
        print(f"      Directory does not exist!")
    exit(1)

# Get latest timestamp - FIXED!
latest_stage1 = sorted(model_files)[-1]
# Extract timestamp: stage1_20251017_123456.pkl -> 20251017_123456
basename = os.path.basename(latest_stage1)  # stage1_20251017_123456.pkl
timestamp = basename.replace('stage1_', '').replace('.pkl', '')  # 20251017_123456

print(f"\n📂 Loading models with timestamp: {timestamp}")
print(f"   Latest model: {os.path.basename(latest_stage1)}")

model_s1 = joblib.load(os.path.join(MODEL_DIR, f"stage1_{timestamp}.pkl"))
model_s2 = joblib.load(os.path.join(MODEL_DIR, f"stage2_{timestamp}.pkl"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, f"encoders_{timestamp}.pkl"))
attack_mapping = joblib.load(os.path.join(MODEL_DIR, f"mapping_{timestamp}.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, f"features_{timestamp}.pkl"))

print(f"✅ Loaded all model artifacts")
print(f"   Features: {len(feature_cols)}")
print(f"   Attack types: {len(attack_mapping)}")

# ============================================================================
# 2. LOAD TEST DATA
# ============================================================================

print("\n" + "="*80)
print("2. LOADING TEST DATA")
print("="*80)

# Use batch_02 for testing (same as training)
test_file = os.path.join(BATCH_DIR, "batch_02.csv")

print(f"\n📂 Loading test data from batch_02...")
df_test = pd.read_csv(test_file, low_memory=False)
print(f"✅ Loaded {len(df_test):,} records")

# Sample if too large (for faster evaluation)
if len(df_test) > 1000000:
    print(f"\n🔧 Sampling 1M for evaluation...")
    from sklearn.model_selection import train_test_split
    df_test, _ = train_test_split(df_test, train_size=1000000, random_state=42, stratify=df_test['category'])
    print(f"✅ Sampled to {len(df_test):,} records")

# ============================================================================
# 3. PREPROCESS TEST DATA
# ============================================================================

print("\n" + "="*80)
print("3. PREPROCESSING TEST DATA")
print("="*80)

# Drop unwanted columns
cols_to_drop = ['pkSeqID', 'saddr', 'sport', 'daddr', 'dport', 
                'smac', 'dmac', 'soui', 'doui', 'sco', 'dco',
                'attack', 'category', 'subcategory']

available_features = [col for col in feature_cols if col in df_test.columns]
df_test_features = df_test[['category'] + available_features].copy()

print(f"✅ Selected {len(available_features)} features")

# Handle missing values
missing = df_test_features.isnull().sum().sum()
if missing > 0:
    numeric_cols = df_test_features.select_dtypes(include=['number']).columns
    df_test_features[numeric_cols] = df_test_features[numeric_cols].fillna(df_test_features[numeric_cols].median())
    print(f"✅ Filled {missing:,} missing values")

# Encode categorical features
cat_cols = df_test_features.select_dtypes(include=['object']).columns.tolist()
cat_cols = [col for col in cat_cols if col != 'category']

for col in cat_cols:
    if col in label_encoders:
        le = label_encoders[col]
        # Handle unseen labels
        df_test_features[col] = df_test_features[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

print(f"✅ Encoded {len(cat_cols)} categorical features")

# Prepare X, y
X_test = df_test_features[available_features]
y_test_true = df_test_features['category']

# ============================================================================
# 4. STAGE 1 EVALUATION - ATTACK vs NORMAL
# ============================================================================

print("\n" + "="*80)
print("4. STAGE 1 EVALUATION - ATTACK vs NORMAL")
print("="*80)

# Create binary labels
y_test_binary = (y_test_true != 'Normal').astype(int)

# Predict
print(f"\n🚀 Running Stage 1 predictions...")
start_time = time.time()
y_pred_s1 = model_s1.predict(X_test)
y_pred_s1_proba = model_s1.predict_proba(X_test)[:, 1]
inference_time_s1 = time.time() - start_time

print(f"✅ Stage 1 inference: {inference_time_s1:.2f}s ({len(X_test)/inference_time_s1:.0f} samples/sec)")

# Metrics
acc_s1 = accuracy_score(y_test_binary, y_pred_s1)
prec_s1 = precision_score(y_test_binary, y_pred_s1, zero_division=0)
rec_s1 = recall_score(y_test_binary, y_pred_s1, zero_division=0)
f1_s1 = f1_score(y_test_binary, y_pred_s1, zero_division=0)

if len(np.unique(y_test_binary)) > 1:
    auc_s1 = roc_auc_score(y_test_binary, y_pred_s1_proba)
else:
    auc_s1 = None

print(f"\n📊 Stage 1 Metrics:")
print(f"   Accuracy:  {acc_s1:.6f} ({acc_s1*100:.2f}%)")
print(f"   Precision: {prec_s1:.6f}")
print(f"   Recall:    {rec_s1:.6f}")
print(f"   F1-Score:  {f1_s1:.6f}")
if auc_s1:
    print(f"   ROC-AUC:   {auc_s1:.6f}")

# Confusion matrix
cm_s1 = confusion_matrix(y_test_binary, y_pred_s1)
tn, fp, fn, tp = cm_s1.ravel()

print(f"\n📊 Stage 1 Confusion Matrix:")
print(f"                    Predicted")
print(f"                Normal  Attack")
print(f"   Actual Normal  {tn:6d}  {fp:6d}  (Total: {tn+fp:,})")
print(f"   Actual Attack  {fn:6d}  {tp:6d}  (Total: {fn+tp:,})")

# Per-class metrics
print(f"\n📊 Stage 1 Per-Class Metrics:")
print(f"   Normal:")
print(f"      Precision: {tn/(tn+fn) if (tn+fn)>0 else 0:.6f}")
print(f"      Recall:    {tn/(tn+fp) if (tn+fp)>0 else 0:.6f}")
print(f"      Support:   {tn+fp:,}")
print(f"   Attack:")
print(f"      Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.6f}")
print(f"      Recall:    {tp/(tp+fn) if (tp+fn)>0 else 0:.6f}")
print(f"      Support:   {fn+tp:,}")

# ============================================================================
# 5. STAGE 2 EVALUATION - ATTACK TYPE CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("5. STAGE 2 EVALUATION - ATTACK TYPE")
print("="*80)

# Filter attack samples
df_test_attacks = df_test_features[df_test_features['category'] != 'Normal'].copy()
df_test_attacks = df_test_attacks[df_test_attacks['category'].isin(list(attack_mapping.keys()))]

X_test_s2 = df_test_attacks[available_features]
y_test_s2_cat = df_test_attacks['category']
y_test_s2 = y_test_s2_cat.map(attack_mapping)

print(f"\n📊 Test data (attacks only): {len(X_test_s2):,} samples")

# Predict
print(f"\n🚀 Running Stage 2 predictions...")
start_time = time.time()
y_pred_s2 = model_s2.predict(X_test_s2)
inference_time_s2 = time.time() - start_time

print(f"✅ Stage 2 inference: {inference_time_s2:.2f}s ({len(X_test_s2)/inference_time_s2:.0f} samples/sec)")

# Metrics
acc_s2 = accuracy_score(y_test_s2, y_pred_s2)
prec_s2 = precision_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0)
rec_s2 = recall_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0)
f1_s2 = f1_score(y_test_s2, y_pred_s2, average='weighted', zero_division=0)

print(f"\n📊 Stage 2 Metrics (Weighted):")
print(f"   Accuracy:  {acc_s2:.6f} ({acc_s2*100:.2f}%)")
print(f"   Precision: {prec_s2:.6f}")
print(f"   Recall:    {rec_s2:.6f}")
print(f"   F1-Score:  {f1_s2:.6f}")

# Confusion matrix
cm_s2 = confusion_matrix(y_test_s2, y_pred_s2)
attack_classes = sorted(list(attack_mapping.keys()))

print(f"\n📊 Stage 2 Confusion Matrix:")
print(f"{'':15s} " + " ".join([f"{c:>10s}" for c in attack_classes]))
print("─" * 80)
for i, true_class in enumerate(attack_classes):
    row = f"{true_class:15s} " + " ".join([f"{cm_s2[i,j]:10d}" for j in range(len(attack_classes))])
    print(row)

# Per-class metrics
print(f"\n📊 Stage 2 Per-Class Metrics:")
for i, cls in enumerate(attack_classes):
    true_count = (y_test_s2 == i).sum()
    pred_count = (y_pred_s2 == i).sum()
    correct = cm_s2[i, i]
    
    precision = correct / pred_count if pred_count > 0 else 0
    recall = correct / true_count if true_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n   {cls}:")
    print(f"      Precision: {precision:.6f}")
    print(f"      Recall:    {recall:.6f}")
    print(f"      F1-Score:  {f1:.6f}")
    print(f"      Support:   {true_count:,}")

# ============================================================================
# 6. COMBINED PIPELINE EVALUATION
# ============================================================================

print("\n" + "="*80)
print("6. COMBINED PIPELINE EVALUATION")
print("="*80)

print(f"\n🚀 Running full pipeline...")
start_time = time.time()

final_predictions = []
final_true_labels = []

for i in range(len(X_test)):
    true_cat = y_test_true.iloc[i]
    is_attack = y_pred_s1[i]
    
    if is_attack == 0:
        prediction = 'Normal'
    else:
        # Only predict attack type if we have this sample
        if true_cat in attack_mapping:
            sample = X_test.iloc[i:i+1]
            attack_type = model_s2.predict(sample)[0]
            prediction = [k for k, v in attack_mapping.items() if v == attack_type][0]
        else:
            prediction = 'Attack'  # Generic
    
    final_predictions.append(prediction)
    final_true_labels.append(true_cat)

inference_time_total = time.time() - start_time

print(f"✅ Full pipeline inference: {inference_time_total:.2f}s ({len(X_test)/inference_time_total:.0f} samples/sec)")

# Overall metrics
all_categories = ['Normal'] + sorted([k for k in attack_mapping.keys()])
overall_acc = accuracy_score(final_true_labels, final_predictions)

print(f"\n📊 Overall Pipeline Metrics:")
print(f"   Overall Accuracy: {overall_acc:.6f} ({overall_acc*100:.2f}%)")

# Confusion matrix
cm_overall = confusion_matrix(final_true_labels, final_predictions, labels=all_categories)

print(f"\n📊 Overall Confusion Matrix:")
print(f"{'':15s} " + " ".join([f"{c:>10s}" for c in all_categories]))
print("─" * 100)
for i, true_cat in enumerate(all_categories):
    row = f"{true_cat:15s} " + " ".join([f"{cm_overall[i,j]:10d}" for j in range(len(all_categories))])
    print(row)

# Per-category accuracy
print(f"\n📊 Per-Category Accuracy:")
for i, cat in enumerate(all_categories):
    total = cm_overall[i].sum()
    correct = cm_overall[i, i]
    acc = correct / total if total > 0 else 0
    print(f"   {cat:15s}: {acc:.6f} ({acc*100:6.2f}%) - {correct:,}/{total:,}")

# Classification report
print(f"\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT")
print("="*80)
print("\n", classification_report(final_true_labels, final_predictions, 
                                  labels=all_categories, zero_division=0, digits=4))

# ============================================================================
# 7. SAVE EVALUATION RESULTS
# ============================================================================

print("\n" + "="*80)
print("7. SAVING EVALUATION RESULTS")
print("="*80)

eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Prepare comprehensive results
results = {
    'metadata': {
        'evaluated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_timestamp': timestamp,
        'test_samples': len(X_test),
        'features': len(available_features)
    },
    'stage1': {
        'accuracy': float(acc_s1),
        'precision': float(prec_s1),
        'recall': float(rec_s1),
        'f1_score': float(f1_s1),
        'roc_auc': float(auc_s1) if auc_s1 else None,
        'inference_time_sec': float(inference_time_s1),
        'samples_per_sec': float(len(X_test)/inference_time_s1),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    },
    'stage2': {
        'accuracy': float(acc_s2),
        'precision_weighted': float(prec_s2),
        'recall_weighted': float(rec_s2),
        'f1_weighted': float(f1_s2),
        'inference_time_sec': float(inference_time_s2),
        'samples_per_sec': float(len(X_test_s2)/inference_time_s2) if len(X_test_s2) > 0 else 0,
        'confusion_matrix': cm_s2.tolist(),
        'per_class_metrics': {}
    },
    'overall': {
        'accuracy': float(overall_acc),
        'total_inference_time_sec': float(inference_time_total),
        'samples_per_sec': float(len(X_test)/inference_time_total),
        'avg_latency_ms': float(inference_time_total / len(X_test) * 1000),
        'confusion_matrix': cm_overall.tolist(),
        'per_category_accuracy': {}
    }
}

# Add per-class metrics for Stage 2
for i, cls in enumerate(attack_classes):
    true_count = int((y_test_s2 == i).sum())
    pred_count = int((y_pred_s2 == i).sum())
    correct = int(cm_s2[i, i])
    
    precision = float(correct / pred_count) if pred_count > 0 else 0.0
    recall = float(correct / true_count) if true_count > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    results['stage2']['per_class_metrics'][cls] = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': true_count
    }

# Add per-category accuracy for overall
for i, cat in enumerate(all_categories):
    total = int(cm_overall[i].sum())
    correct = int(cm_overall[i, i])
    acc = float(correct / total) if total > 0 else 0.0
    
    results['overall']['per_category_accuracy'][cat] = {
        'accuracy': acc,
        'correct': correct,
        'total': total
    }

# Save JSON
eval_json_file = os.path.join(EVAL_DIR, f"evaluation_{eval_timestamp}.json")
with open(eval_json_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved evaluation results:")
print(f"   {eval_json_file}")

# Save summary text
summary_file = os.path.join(EVAL_DIR, f"evaluation_summary_{eval_timestamp}.txt")
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MODEL EVALUATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Evaluated at: {results['metadata']['evaluated_at']}\n")
    f.write(f"Model timestamp: {timestamp}\n")
    f.write(f"Test samples: {len(X_test):,}\n")
    f.write(f"Features: {len(available_features)}\n\n")
    
    f.write("="*80 + "\n")
    f.write("STAGE 1 - ATTACK vs NORMAL\n")
    f.write("="*80 + "\n")
    f.write(f"Accuracy:  {acc_s1:.6f} ({acc_s1*100:.2f}%)\n")
    f.write(f"Precision: {prec_s1:.6f}\n")
    f.write(f"Recall:    {rec_s1:.6f}\n")
    f.write(f"F1-Score:  {f1_s1:.6f}\n")
    if auc_s1:
        f.write(f"ROC-AUC:   {auc_s1:.6f}\n")
    f.write(f"Inference: {len(X_test)/inference_time_s1:.0f} samples/sec\n\n")
    
    f.write("="*80 + "\n")
    f.write("STAGE 2 - ATTACK TYPE\n")
    f.write("="*80 + "\n")
    f.write(f"Accuracy:  {acc_s2:.6f} ({acc_s2*100:.2f}%)\n")
    f.write(f"Precision: {prec_s2:.6f}\n")
    f.write(f"Recall:    {rec_s2:.6f}\n")
    f.write(f"F1-Score:  {f1_s2:.6f}\n")
    f.write(f"Inference: {len(X_test_s2)/inference_time_s2:.0f} samples/sec\n\n")
    
    f.write("="*80 + "\n")
    f.write("OVERALL PIPELINE\n")
    f.write("="*80 + "\n")
    f.write(f"Overall Accuracy: {overall_acc:.6f} ({overall_acc*100:.2f}%)\n")
    f.write(f"Avg Latency: {inference_time_total / len(X_test) * 1000:.2f} ms/sample\n")
    f.write(f"Throughput: {len(X_test)/inference_time_total:.0f} samples/sec\n\n")
    
    f.write("="*80 + "\n")
    f.write("PER-CATEGORY ACCURACY\n")
    f.write("="*80 + "\n")
    for cat, metrics in results['overall']['per_category_accuracy'].items():
        f.write(f"{cat:15s}: {metrics['accuracy']:.6f} ({metrics['accuracy']*100:6.2f}%) - {metrics['correct']:,}/{metrics['total']:,}\n")

print(f"   {summary_file}")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ EVALUATION COMPLETED")
print("="*80)

print(f"\n📊 KEY METRICS:")
print(f"   Stage 1 Accuracy:  {acc_s1:.4f} ({acc_s1*100:.2f}%)")
print(f"   Stage 2 Accuracy:  {acc_s2:.4f} ({acc_s2*100:.2f}%)")
print(f"   Overall Accuracy:  {overall_acc:.4f} ({overall_acc*100:.2f}%)")
print(f"   Avg Latency:       {inference_time_total / len(X_test) * 1000:.2f} ms/sample")
print(f"   Throughput:        {len(X_test)/inference_time_total:.0f} samples/sec")

print(f"\n📂 Results saved to:")
print(f"   {EVAL_DIR}/")
print(f"   • evaluation_{eval_timestamp}.json")
print(f"   • evaluation_summary_{eval_timestamp}.txt")

print("\n" + "="*80)

