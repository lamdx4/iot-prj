"""
Model training and evaluation module
"""
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with validation set"""
    print("\nStarting XGBoost training...")
    
    model = XGBClassifier(
        max_depth=7,
        learning_rate=0.1,
        n_estimators=100,
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    print("Training completed!")
    
    results = model.evals_result()
    training_history = {
        'train_loss': results['validation_0']['mlogloss'],
        'val_loss': results['validation_1']['mlogloss']
    }
    
    return model, training_history

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate model and calculate metrics"""
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"\nEvaluation results:")
    print(f"   Accuracy:  {accuracy:.6f} ({accuracy*100:.4f}%)")
    print(f"   Precision: {precision:.6f} ({precision*100:.4f}%)")
    print(f"   Recall:    {recall:.6f} ({recall*100:.4f}%)")
    print(f"   F1-Score:  {f1:.6f} ({f1*100:.4f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification Report
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Per-class metrics
    per_class_metrics = {}
    for i, cls in enumerate(label_encoder.classes_):
        per_class_metrics[cls] = {
            'precision': report[cls]['precision'],
            'recall': report[cls]['recall'],
            'f1-score': report[cls]['f1-score'],
            'support': report[cls]['support']
        }
    
    return metrics, cm, report, y_pred, y_pred_proba, per_class_metrics

def measure_inference_speed(model, X_test, num_runs=10):
    """Measure inference speed (flows/second)"""
    print("\nMeasuring inference speed...")
    times = []
    
    for _ in range(num_runs):
        start = time.time()
        _ = model.predict(X_test)
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    flows_per_sec = len(X_test) / avg_time
    std_flows = std_time * flows_per_sec / avg_time if avg_time > 0 else 0
    
    print(f"   Avg Flows/s: {flows_per_sec:.0f} +/- {std_flows:.0f}")
    
    if flows_per_sec > 1681:
        print(f"   Capable of real-time processing (>1681 flows/s)")
    else:
        print(f"   Needs optimization (<1681 flows/s)")
    
    return {
        'avg_flows_per_sec': flows_per_sec,
        'std_flows_per_sec': std_flows
    }