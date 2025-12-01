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

# Find latest run directory OR use specified run
RUN_NAME = os.getenv('RUN_NAME', None)  # Can specify specific run

if RUN_NAME:
    RUN_DIR = os.path.join(PROJECT_ROOT, f"results/{RUN_NAME}")
else:
    # Find latest run
    import glob
    runs = glob.glob(os.path.join(PROJECT_ROOT, "results/run_*"))
    if not runs:
        print("❌ No training runs found!")
        print(f"   Expected directory: {PROJECT_ROOT}/results/run_*")
        exit(1)
    RUN_DIR = sorted(runs)[-1]

MODEL_DIR = os.path.join(RUN_DIR, "models")
METRICS_DIR = os.path.join(RUN_DIR, "metrics")
BATCH_DIR = os.getenv('BATCH_DIR', os.path.join(PROJECT_ROOT, "Data/Dataset"))

print(f"\n📂 Paths:")
print(f"   Run: {os.path.basename(RUN_DIR)}")
print(f"   Models: {MODEL_DIR}")
print(f"   Metrics: {METRICS_DIR}")

# ============================================================================
# 1. LOAD LATEST MODELS
# ============================================================================

print("\n" + "="*80)
print("1. LOADING MODELS")
print("="*80)

print(f"\n📂 Loading from: {os.path.basename(RUN_DIR)}")

# Extract timestamp from run directory name (e.g., run_20251201_143848)
timestamp = os.path.basename(RUN_DIR).replace('run_', '')

# Load models (no timestamp suffix)
print(f"\n📂 Loading model files...")
model_s1 = joblib.load(os.path.join(MODEL_DIR, "stage1.pkl"))
print(f"   ✅ Loaded: stage1.pkl")

model_s2 = joblib.load(os.path.join(MODEL_DIR, "stage2.pkl"))
print(f"   ✅ Loaded: stage2.pkl")

label_encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
print(f"   ✅ Loaded: encoders.pkl")

attack_mapping = joblib.load(os.path.join(MODEL_DIR, "mapping.pkl"))
print(f"   ✅ Loaded: mapping.pkl")

feature_cols = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
print(f"   ✅ Loaded: features.pkl")

print(f"\n✅ Loaded all model artifacts")
print(f"   Features: {len(feature_cols)}")
print(f"   Attack types: {len(attack_mapping)}")

# ============================================================================
# 2. LOAD TEST DATA
# ============================================================================

print("\n" + "="*80)
print("2. LOADING TEST DATA")
print("="*80)

# Use balanced test set (created by 00_create_test_set.py)
TEST_FILE = os.getenv('TEST_FILE', os.path.join(PROJECT_ROOT, "Data/Dataset/test_balanced_100k.csv"))

print(f"\n📂 Loading balanced test set...")
print(f"   File: {TEST_FILE}")

df_test = pd.read_csv(TEST_FILE, low_memory=False)
print(f"✅ Loaded {len(df_test):,} records")

# Show distribution
print(f"\n📊 Test set distribution:")
for cat, count in df_test['category'].value_counts().items():
    pct = count / len(df_test) * 100
    print(f"   {cat:15s}: {count:6,} ({pct:5.2f}%)")


# ============================================================================
# 3. CREATE SOURCE DIVERSITY FEATURES (same as training)
# ============================================================================

print("\n" + "="*80)
print("3. CREATE SOURCE DIVERSITY FEATURES")
print("="*80)

print("\n🔧 Creating source diversity features...")
print("   Using saddr aggregation (same as training)")

from scipy.stats import entropy as scipy_entropy
import numpy as np

WINDOW_SIZE = 30  # seconds (same as training)

def calculate_source_diversity(group):
    """Calculate source diversity metrics - same as training"""
    if len(group) == 0:
        return pd.Series({
            'unique_src_count': 1,
            'src_entropy': 0.0,
            'top_src_ratio': 1.0
        })
    
    unique_count = group['saddr'].nunique()
    
    src_counts = group['saddr'].value_counts()
    if len(src_counts) > 1:
        src_probs = src_counts / src_counts.sum()
        src_entropy_val = scipy_entropy(src_probs, base=2)
    else:
        src_entropy_val = 0.0
    
    top_src_ratio = src_counts.iloc[0] / len(group) if len(src_counts) > 0 else 1.0
    
    return pd.Series({
        'unique_src_count': unique_count,
        'src_entropy': src_entropy_val,
        'top_src_ratio': top_src_ratio
    })

# Process test set
print(f"   Processing test data...")
df_test['time_window'] = (df_test['stime'] // WINDOW_SIZE).astype(int)

try:
    test_diversity = df_test.groupby(['time_window', 'daddr']).apply(
        calculate_source_diversity
    ).reset_index()
    
    df_test = df_test.merge(test_diversity, on=['time_window', 'daddr'], how='left')
    df_test['unique_src_count'].fillna(1, inplace=True)
    df_test['src_entropy'].fillna(0.0, inplace=True)
    df_test['top_src_ratio'].fillna(1.0, inplace=True)
    
    print(f"      ✅ Test: {len(df_test):,} records")
    print(f"      unique_src_count range: {df_test['unique_src_count'].min():.0f} - {df_test['unique_src_count'].max():.0f}")
    
except Exception as e:
    print(f"      ⚠️  Error: {e}")
    print(f"      Using default values")
    df_test['unique_src_count'] = 1
    df_test['src_entropy'] = 0.0
    df_test['top_src_ratio'] = 1.0

# Drop temporary column
df_test.drop('time_window', axis=1, inplace=True, errors='ignore')

print(f"\n   ✅ Added 3 source diversity features")

# ============================================================================
# 4. PREPROCESS TEST DATA
# ============================================================================

print("\n" + "="*80)
print("4. PREPROCESSING TEST DATA")
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

if label_encoders is not None:
    # Check if label_encoders is a dict or single encoder
    if isinstance(label_encoders, dict):
        # New format: dictionary of encoders
        print(f"🔧 Encoding categorical features (dict format)...")
        for col in cat_cols:
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unseen labels
                df_test_features[col] = df_test_features[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                print(f"   ✅ Encoded: {col}")
    else:
        # Old format: single encoder (assume it's for 'proto')
        print(f"🔧 Encoding categorical features (single encoder for 'proto')...")
        if 'proto' in cat_cols:
            le = label_encoders
            df_test_features['proto'] = df_test_features['proto'].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
            print(f"   ✅ Encoded: proto")
        # Remove 'proto' from cat_cols if exists
        cat_cols = [col for col in cat_cols if col != 'proto']
        
        # For other categorical columns, use simple label encoding
        from sklearn.preprocessing import LabelEncoder
        for col in cat_cols:
            le_temp = LabelEncoder()
            df_test_features[col] = le_temp.fit_transform(df_test_features[col].astype(str))
            print(f"   ✅ Encoded: {col} (fit on test)")
    
    print(f"✅ Encoded {len(cat_cols) + (1 if 'proto' in df_test_features.columns else 0)} categorical features")
else:
    # No encoders, skip encoding
    print(f"⚠️  No encoders available, skipping categorical encoding")

# Prepare X, y
X_test = df_test_features[available_features]
y_test_true = df_test_features['category']

# ============================================================================
# 5. STAGE 1 EVALUATION - ATTACK vs NORMAL
# ============================================================================

print("\n" + "="*80)
print("5. STAGE 1 EVALUATION - ATTACK vs NORMAL")
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
# 6. STAGE 2 EVALUATION - ATTACK TYPE CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("6. STAGE 2 EVALUATION - ATTACK TYPE")
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
# 7. COMBINED PIPELINE EVALUATION
# ============================================================================

print("\n" + "="*80)
print("7. COMBINED PIPELINE EVALUATION")
print("="*80)

print(f"\n🚀 Running full pipeline (vectorized)...")
start_time = time.time()

# Vectorized approach - MUCH FASTER!
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
    
    # Batch prediction!
    attack_types = model_s2.predict(attack_samples)
    
    # Map to names
    reverse_mapping = {v: k for k, v in attack_mapping.items()}
    attack_names = np.array([reverse_mapping[t] for t in attack_types])
    final_predictions[attack_mask] = attack_names
    
    print(f"   ✅ Batch prediction completed")

# Convert to list
final_predictions = list(final_predictions)
final_true_labels = y_test_true.tolist()

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
# 8. SAVE EVALUATION RESULTS
# ============================================================================

print("\n" + "="*80)
print("8. SAVING EVALUATION RESULTS")
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

# Save evaluation results (no timestamp in filename)
eval_json_file = os.path.join(METRICS_DIR, "evaluation.json")
with open(eval_json_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved evaluation results:")
print(f"   {eval_json_file}")

# Save summary text
summary_file = os.path.join(METRICS_DIR, "evaluation_summary.txt")
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
print(f"   {METRICS_DIR}/")
print(f"   • evaluation.json")
print(f"   • evaluation_summary.txt")

print("\n" + "="*80)

