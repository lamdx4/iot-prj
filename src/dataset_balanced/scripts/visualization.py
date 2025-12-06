"""
Visualization module for plotting results
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_results(training_history, cm, feature_importance, feature_cols, 
                 label_encoder, metrics, per_class_metrics, y_test, y_pred_proba,
                 X_train, y_train, save_dir='./plots'):
    """Plot all results in paper style"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nGenerating plots...")
    
    classes = label_encoder.classes_
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'][:len(classes)]
    
    # 1. Class Distribution
    plot_class_distribution(y_train, classes, colors, save_dir)
    
    # 2. Training History
    plot_training_history(training_history, save_dir)
    
    # 3. Confusion Matrix
    plot_confusion_matrix(cm, classes, save_dir)
    
    # 4. Feature Importance
    plot_feature_importance(feature_importance, feature_cols, save_dir)
    
    # 5. Overall Metrics
    plot_overall_metrics(metrics, save_dir)
    
    # 6. Per-Class Performance
    plot_per_class_metrics(per_class_metrics, colors, save_dir)
    
    # 7. Correlation Matrix
    plot_correlation_matrix(X_train, save_dir)
    
    # 8. ROC Curves
    plot_roc_curves(y_test, y_pred_proba, classes, colors, save_dir)
    
    print(f"\nAll plots saved to: {save_dir}")

def plot_class_distribution(y_train, classes, colors, save_dir):
    """Plot class distribution"""
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(y_train, return_counts=True)
    class_names = [classes[i] for i in unique]
    
    bars = plt.bar(class_names, counts, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Data Distribution Plot for Multiclass Classification', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Class', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=13, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: class_distribution.png")
    plt.close()

def plot_training_history(training_history, save_dir):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(training_history['train_loss']) + 1)
    plt.plot(epochs, training_history['train_loss'], 'b-', label='Train Loss', 
             linewidth=2, marker='o', markersize=4, markevery=5)
    plt.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', 
             linewidth=2, marker='s', markersize=4, markevery=5)
    plt.title('Training History', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Loss (Log Loss)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: training_history.png")
    plt.close()

def plot_confusion_matrix(cm, classes, save_dir):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes,
                yticklabels=classes,
                cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='gray',
                annot_kws={'size': 11, 'weight': 'bold'})
    
    plt.title('Confusion Matrix for Multiclass Classification', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xticks(fontsize=11, rotation=45, ha='right')
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: confusion_matrix.png")
    plt.close()

def plot_feature_importance(feature_importance, feature_cols, save_dir):
    """Plot feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    plt.barh(importance_df['feature'], importance_df['importance'], 
             color=colors_grad, edgecolor='black', linewidth=0.8)
    plt.title('Feature Importance', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Importance Score', fontsize=13, fontweight='bold')
    plt.ylabel('Feature', fontsize=13, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: feature_importance.png")
    plt.close()

def plot_overall_metrics(metrics, save_dir):
    """Plot overall performance metrics"""
    plt.figure(figsize=(10, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [metrics[k] * 100 for k in ['accuracy', 'precision', 'recall', 'f1_score']]
    colors_metrics = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
    
    bars = plt.bar(metric_names, metric_values, color=colors_metrics, 
                   edgecolor='black', linewidth=1.5)
    plt.title('Overall Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score (%)', fontsize=13, fontweight='bold')
    plt.ylim(0, 105)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=11)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/overall_metrics.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: overall_metrics.png")
    plt.close()

def plot_per_class_metrics(per_class_metrics, colors, save_dir):
    """Plot per-class performance metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    metric_titles = ['Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
        ax = axes[idx // 2, idx % 2]
        
        class_list = list(per_class_metrics.keys())
        values = [per_class_metrics[cls][metric] * 100 for cls in class_list]
        
        bars = ax.bar(class_list, values, color=colors[:len(class_list)], 
                     edgecolor='black', linewidth=1.2)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    # Support
    ax = axes[1, 1]
    class_list = list(per_class_metrics.keys())
    supports = [per_class_metrics[cls]['support'] for cls in class_list]
    
    bars = ax.bar(class_list, supports, color=colors[:len(class_list)], 
                 edgecolor='black', linewidth=1.2)
    ax.set_title('Support (Sample Count)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', 
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/per_class_metrics.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: per_class_metrics.png")
    plt.close()

def plot_correlation_matrix(X_train, save_dir):
    """Plot correlation matrix"""
    plt.figure(figsize=(14, 12))
    
    correlation_matrix = X_train.corr()
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                linecolor='white',
                cbar_kws={
                    'label': 'Correlation Coefficient', 
                    'shrink': 0.8
                },
                vmin=-1.0,
                vmax=1.0,
                annot_kws={'size': 8})
    
    plt.title('Feature Correlation Matrix Heatmap', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: correlation_matrix.png")
    plt.close()

def plot_roc_curves(y_test, y_pred_proba, classes, colors, save_dir):
    """Plot ROC curves for multiclass classification"""
    try:
        n_classes = len(classes)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(10, 8))
        colors_roc = cycle(colors)
        
        for i, color in zip(range(n_classes), colors_roc):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{classes[i]} (AUC = {roc_auc[i]:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        plt.title('ROC Curves - Multiclass Classification', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: roc_curves.png")
        plt.close()
        
    except Exception as e:
        print(f"   Warning: Could not generate ROC curves: {e}")