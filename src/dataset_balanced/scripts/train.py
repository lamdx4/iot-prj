"""
Main training script for XGBoost DDoS classifier
"""
from data_loader import load_data, prepare_data
from model_trainer import train_model, evaluate_model, measure_inference_speed
from visualization import plot_results
from utils import save_results
from sklearn.model_selection import train_test_split

def main(data_file):
    """Main training pipeline"""
    print("="*70)
    print("XGBOOST NETWORK TRAFFIC CLASSIFIER")
    print("="*70)
    
    # Load data
    df = load_data(data_file)
    
    # Prepare data
    X, y, label_encoder, feature_cols = prepare_data(df)
    
    # Split data: 80/10/10
    print("\nSplitting data: Train/Validation/Test (80/10/10)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   Train:      {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Train model
    model, training_history = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("VALIDATION SET EVALUATION")
    print("="*70)
    metrics_val, cm_val, report_val, y_pred_val, y_pred_proba_val, per_class_val = \
        evaluate_model(model, X_val, y_val, label_encoder)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("TEST SET EVALUATION (UNSEEN DATA)")
    print("="*70)
    metrics_test, cm_test, report_test, y_pred_test, y_pred_proba_test, per_class_test = \
        evaluate_model(model, X_test, y_test, label_encoder)
    
    # Time performance
    print("\n" + "="*70)
    print("TIME PERFORMANCE EVALUATION")
    print("="*70)
    speed_metrics = measure_inference_speed(model, X_test)
    
    # Get feature importance
    feature_importance = model.feature_importances_
    
    # Plot all results
    plot_results(
        training_history, cm_test, feature_importance, 
        feature_cols, label_encoder, metrics_test, per_class_test,
        y_test, y_pred_proba_test, X_train, y_train
    )
    
    # Save results
    all_metrics = {
        'validation': metrics_val,
        'test': metrics_test
    }
    save_results(
        model, label_encoder, all_metrics, training_history,
        feature_cols, cm_test, report_test, speed_metrics
    )
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print("\nFINAL RESULTS:")
    print(f"\n   VALIDATION SET:")
    print(f"   - Accuracy:  {metrics_val['accuracy']*100:.4f}%")
    print(f"   - Precision: {metrics_val['precision']*100:.4f}%")
    print(f"   - Recall:    {metrics_val['recall']*100:.4f}%")
    print(f"   - F1-Score:  {metrics_val['f1_score']*100:.4f}%")
    
    print(f"\n   TEST SET (Final):")
    print(f"   - Accuracy:  {metrics_test['accuracy']*100:.4f}%")
    print(f"   - Precision: {metrics_test['precision']*100:.4f}%")
    print(f"   - Recall:    {metrics_test['recall']*100:.4f}%")
    print(f"   - F1-Score:  {metrics_test['f1_score']*100:.4f}%")
    
    print(f"\n   TIME PERFORMANCE:")
    print(f"   - Avg Flows/s: {speed_metrics['avg_flows_per_sec']:.0f} +/- {speed_metrics['std_flows_per_sec']:.0f}")
    
    print("\nGenerated files:")
    print("   model/")
    print("      - xgboost_model.json")
    print("      - label_encoder.pkl")
    print("      - training_results.json")
    print("   plots/")
    print("      - class_distribution.png")
    print("      - training_history.png")
    print("      - confusion_matrix.png")
    print("      - feature_importance.png")
    print("      - correlation_matrix.png")
    print("      - overall_metrics.png")
    print("      - per_class_metrics.png")
    print("      - roc_curves.png")

if __name__ == "__main__":
    DATA_FILE = "balanced_ddos_only.csv"
    main(DATA_FILE)