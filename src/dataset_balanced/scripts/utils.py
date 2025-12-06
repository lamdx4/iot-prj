"""
Utility functions for saving results
"""
import pickle
import json
from datetime import datetime
import os

def save_results(model, label_encoder, metrics, training_history, 
                 feature_cols, cm, report, speed_metrics, save_dir='./model'):
    """Save model and results"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving model and results...")
    
    # Save model
    model.save_model(f'{save_dir}/xgboost_model.json')
    print(f"   Saved: xgboost_model.json")
    
    # Save label encoder
    with open(f'{save_dir}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"   Saved: label_encoder.pkl")
    
    # Save training results
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'training_history': training_history,
        'feature_cols': feature_cols,
        'classes': label_encoder.classes_.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'time_performance': speed_metrics
    }
    
    with open(f'{save_dir}/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved: training_results.json")
    
    print(f"All files saved to: {save_dir}")