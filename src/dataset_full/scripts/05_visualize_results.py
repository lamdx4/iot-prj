"""
Step 5: Generate Visualizations from Training Metrics
======================================================

Create visualization charts from saved training metrics JSON file.
Run this AFTER training to avoid RAM issues during training.

Usage:
    python 05_visualize_results.py
    
    Or specify metrics file:
    python 05_visualize_results.py path/to/training_metrics_20251017_123456.json

Author: Lambda Team
Date: October 2025
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

print("="*80)
print("GENERATE VISUALIZATIONS FROM TRAINING METRICS")
print("="*80)

# ============================================================================
# 1. LOAD METRICS
# ============================================================================

print("\n" + "="*80)
print("1. LOADING METRICS")
print("="*80)

# Auto-detect paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(PROJECT_ROOT, "models/full_dataset"))

# Find metrics file
if len(sys.argv) > 1:
    metrics_file = sys.argv[1]
else:
    # Find latest metrics file
    import glob
    metrics_files = sorted(glob.glob(os.path.join(MODEL_DIR, "training_metrics_*.json")))
    if not metrics_files:
        print("\n❌ No training metrics found!")
        print(f"   Please train model first or specify metrics file path")
        sys.exit(1)
    metrics_file = metrics_files[-1]

print(f"\n📂 Loading: {os.path.basename(metrics_file)}")

with open(metrics_file, 'r') as f:
    metrics = json.load(f)

print(f"✅ Metrics loaded")
print(f"   Timestamp: {metrics['metadata']['timestamp']}")
print(f"   Train samples: {metrics['training_data']['total_records']:,}")
print(f"   Test samples: {metrics['test_data']['total_records']:,}")
print(f"   Overall accuracy: {metrics['overall']['accuracy']:.4f}")

# ============================================================================
# 2. PREPARE DATA
# ============================================================================

print("\n" + "="*80)
print("2. PREPARING DATA")
print("="*80)

# Extract data
cm_overall = np.array(metrics['overall']['confusion_matrix'])
all_categories = list(metrics['overall']['per_category_metrics'].keys())

categories = all_categories
precisions = [metrics['overall']['per_category_metrics'][c]['precision'] for c in categories]
recalls = [metrics['overall']['per_category_metrics'][c]['recall'] for c in categories]
f1_scores = [metrics['overall']['per_category_metrics'][c]['f1_score'] for c in categories]
supports = [metrics['overall']['per_category_metrics'][c]['support'] for c in categories]

print(f"✅ Data prepared")
print(f"   Categories: {', '.join(categories)}")

# Create output directory
timestamp = metrics['metadata']['timestamp']
viz_dir = os.path.join(MODEL_DIR, f"visualizations_{timestamp}")
os.makedirs(viz_dir, exist_ok=True)

print(f"   Output dir: {viz_dir}")

# ============================================================================
# 3. GENERATE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("3. GENERATING VISUALIZATIONS")
print("="*80)

# 1. Confusion Matrix Heatmap
print("\n📊 1. Confusion Matrix Heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues', 
            xticklabels=all_categories, yticklabels=all_categories,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Overall Performance', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '01_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 01_confusion_matrix.png")

# 2. Normalized Confusion Matrix (%)
print("📊 2. Normalized Confusion Matrix...")
cm_normalized = cm_overall.astype('float') / cm_overall.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
            xticklabels=all_categories, yticklabels=all_categories,
            vmin=0, vmax=1, cbar_kws={'label': 'Percentage'})
plt.title('Confusion Matrix (Normalized) - Percentage', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '02_confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 02_confusion_matrix_normalized.png")

# 3. Per-Category Metrics Bar Chart
print("📊 3. Per-Category Metrics...")
x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
bars2 = ax.bar(x, recalls, width, label='Recall', color='#2ecc71')
bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')

ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Category Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend(loc='lower right')
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '03_per_category_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 03_per_category_metrics.png")

# 4. Stage Comparison
print("📊 4. Stage Comparison...")
stages = ['Stage 1\n(Binary)', 'Stage 2\n(Multi-class)', 'Overall']
accuracies = [
    metrics['stage1']['accuracy'],
    metrics['stage2']['accuracy'],
    metrics['overall']['accuracy']
]

colors = ['#3498db', '#e74c3c', '#2ecc71']
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(stages, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Model Performance by Stage', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}\n({height*100:.2f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '04_stage_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 04_stage_comparison.png")

# 5. Class Distribution (Train vs Test)
print("📊 5. Class Distribution...")
train_dist = metrics['training_data']['category_distribution']
test_dist = metrics['test_data']['category_distribution']

# Get all categories
all_cats = sorted(set(list(train_dist.keys()) + list(test_dist.keys())))
train_counts = [train_dist.get(c, 0) for c in all_cats]
test_counts = [test_dist.get(c, 0) for c in all_cats]

x = np.arange(len(all_cats))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Absolute counts
bars1 = ax1.bar(x - width/2, train_counts, width, label='Training', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, test_counts, width, label='Testing', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Category', fontsize=11, fontweight='bold')
ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
ax1.set_title('Class Distribution: Training vs Testing', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(all_cats, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_yscale('log')  # Log scale for better visibility

# Subplot 2: Percentage
train_total = sum(train_counts)
test_total = sum(test_counts)
train_pcts = [c/train_total*100 for c in train_counts]
test_pcts = [c/test_total*100 for c in test_counts]

bars3 = ax2.bar(x - width/2, train_pcts, width, label='Training', color='#3498db', alpha=0.8)
bars4 = ax2.bar(x + width/2, test_pcts, width, label='Testing', color='#e74c3c', alpha=0.8)

ax2.set_xlabel('Category', fontsize=11, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax2.set_title('Class Distribution: Percentage', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(all_cats, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '05_class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 05_class_distribution.png")

# 6. Per-Category Accuracy with Support
print("📊 6. Accuracy vs Support...")
accuracies_cat = []

for i, cat in enumerate(all_categories):
    total = cm_overall[i].sum()
    correct = cm_overall[i, i]
    acc = correct / total if total > 0 else 0
    accuracies_cat.append(acc)

fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = '#2ecc71'
ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color=color1)
bars = ax1.bar(categories, accuracies_cat, color=color1, alpha=0.7, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim([0, 1.1])
ax1.set_xticklabels(categories, rotation=45, ha='right')

# Add value labels
for i, (bar, acc, sup) in enumerate(zip(bars, accuracies_cat, supports)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2%}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Secondary y-axis for support
ax2 = ax1.twinx()
color2 = '#e74c3c'
ax2.set_ylabel('Support (samples)', fontsize=12, fontweight='bold', color=color2)
ax2.plot(categories, supports, color=color2, marker='o', linewidth=2, 
         markersize=8, label='Support', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_yscale('log')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

ax1.set_title('Per-Category Accuracy vs Support Size', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, '06_accuracy_vs_support.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 06_accuracy_vs_support.png")

# 7. Summary Dashboard
print("📊 7. Summary Dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Top: Confusion Matrix
ax_cm = fig.add_subplot(gs[0:2, 0:2])
sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='RdYlGn', 
            xticklabels=all_categories, yticklabels=all_categories,
            vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, ax=ax_cm)
ax_cm.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
ax_cm.set_ylabel('True Label')
ax_cm.set_xlabel('Predicted Label')

# Top Right: Key Metrics
ax_metrics = fig.add_subplot(gs[0:2, 2])
ax_metrics.axis('off')
metrics_text = f"""
KEY METRICS

Overall Accuracy:
{metrics['overall']['accuracy']:.4f} ({metrics['overall']['accuracy']*100:.2f}%)

Stage 1 (Binary):
• Accuracy: {metrics['stage1']['accuracy']:.4f}
• ROC-AUC: {metrics['stage1'].get('roc_auc', 0):.4f}

Stage 2 (Multi-class):
• Accuracy: {metrics['stage2']['accuracy']:.4f}
• F1-Score: {metrics['stage2']['f1_weighted']:.4f}

Dataset:
• Train: {metrics['training_data']['total_records']:,} samples
• Test: {metrics['test_data']['total_records']:,} samples
• Features: {metrics['metadata']['num_features']}

Training Time:
• Timestamp: {metrics['metadata']['trained_at']}
"""
ax_metrics.text(0.1, 0.95, metrics_text, transform=ax_metrics.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Bottom: Per-Category Metrics
ax_bars = fig.add_subplot(gs[2, :])
x_pos = np.arange(len(categories))
width = 0.25
bars1 = ax_bars.bar(x_pos - width, precisions, width, label='Precision', color='#3498db', alpha=0.8)
bars2 = ax_bars.bar(x_pos, recalls, width, label='Recall', color='#2ecc71', alpha=0.8)
bars3 = ax_bars.bar(x_pos + width, f1_scores, width, label='F1-Score', color='#e74c3c', alpha=0.8)

ax_bars.set_xlabel('Category', fontweight='bold')
ax_bars.set_ylabel('Score', fontweight='bold')
ax_bars.set_title('Per-Category Performance', fontweight='bold')
ax_bars.set_xticks(x_pos)
ax_bars.set_xticklabels(categories, rotation=45, ha='right')
ax_bars.legend(loc='lower right')
ax_bars.set_ylim([0, 1.1])
ax_bars.grid(axis='y', alpha=0.3)

fig.suptitle('Hierarchical Classification - Training Summary Dashboard', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(os.path.join(viz_dir, '00_summary_dashboard.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 00_summary_dashboard.png")

# ============================================================================
# 4. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ VISUALIZATION COMPLETED!")
print("="*80)

print(f"\n📊 Generated 7 visualizations:")
print(f"   • 00_summary_dashboard.png (Overview)")
print(f"   • 01_confusion_matrix.png")
print(f"   • 02_confusion_matrix_normalized.png")
print(f"   • 03_per_category_metrics.png")
print(f"   • 04_stage_comparison.png")
print(f"   • 05_class_distribution.png")
print(f"   • 06_accuracy_vs_support.png")

print(f"\n📂 Saved to: {viz_dir}/")

print("\n" + "="*80)

