"""
Step 6: Plot Training Loss Curves
==================================

Visualize training and validation loss curves from training metrics.

Generates:
- Stage 1 loss curve (logloss)
- Stage 2 loss curve (mlogloss)
- Combined plot
- Convergence analysis

Author: Lambda Team
Date: December 2025
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("="*80)
print("STEP 6: PLOT TRAINING LOSS CURVES")
print("="*80)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(PROJECT_ROOT, "models/full_dataset"))
PLOT_DIR = os.getenv('PLOT_DIR', os.path.join(PROJECT_ROOT, "results/plots"))

os.makedirs(PLOT_DIR, exist_ok=True)

print(f"\n📂 Paths:")
print(f"   Model dir: {MODEL_DIR}")
print(f"   Plot dir:  {PLOT_DIR}")

# ============================================================================
# 1. LOAD LATEST TRAINING METRICS
# ============================================================================

print("\n" + "="*80)
print("1. LOADING TRAINING METRICS")
print("="*80)

import glob
metrics_files = glob.glob(os.path.join(MODEL_DIR, "training_metrics_*.json"))

if not metrics_files:
    print("❌ No training metrics found!")
    print(f"   Expected directory: {MODEL_DIR}")
    exit(1)

latest_metrics = sorted(metrics_files)[-1]
print(f"\n📂 Latest metrics: {os.path.basename(latest_metrics)}")

with open(latest_metrics, 'r') as f:
    metrics = json.load(f)

timestamp = metrics['metadata']['timestamp']

# Check if training_history exists
if 'training_history' not in metrics['stage1']:
    print("❌ No training history found in metrics!")
    print("   Please re-run training to capture loss curves.")
    exit(1)

print(f"✅ Loaded metrics from: {metrics['metadata']['trained_at']}")

# ============================================================================
# 2. PLOT STAGE 1 LOSS CURVE
# ============================================================================

print("\n" + "="*80)
print("2. PLOTTING STAGE 1 LOSS CURVE")
print("="*80)

history_s1 = metrics['stage1']['training_history']
train_loss_s1 = history_s1['train_loss']
val_loss_s1 = history_s1['val_loss']
best_iter_s1 = metrics['stage1']['best_iteration']

plt.figure(figsize=(12, 6))
plt.plot(train_loss_s1, label='Training Loss', linewidth=2, alpha=0.8, color='#2E86AB')
plt.plot(val_loss_s1, label='Validation Loss', linewidth=2, alpha=0.8, color='#A23B72')
plt.axvline(x=best_iter_s1, color='#F18F01', linestyle='--', 
            label=f'Best Iteration ({best_iter_s1})', alpha=0.7, linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Log Loss', fontsize=12)
plt.title('Stage 1: Binary Classification (Attack vs Normal)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

plot_file_s1 = os.path.join(PLOT_DIR, f"stage1_loss_curve_{timestamp}.png")
plt.savefig(plot_file_s1, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {os.path.basename(plot_file_s1)}")
plt.close()

# ============================================================================
# 3. PLOT STAGE 2 LOSS CURVE
# ============================================================================

print("\n" + "="*80)
print("3. PLOTTING STAGE 2 LOSS CURVE")
print("="*80)

history_s2 = metrics['stage2']['training_history']
train_loss_s2 = history_s2['train_loss']
val_loss_s2 = history_s2['val_loss']
best_iter_s2 = metrics['stage2']['best_iteration']

plt.figure(figsize=(12, 6))
plt.plot(train_loss_s2, label='Training Loss', linewidth=2, alpha=0.8, color='#2E86AB')
plt.plot(val_loss_s2, label='Validation Loss', linewidth=2, alpha=0.8, color='#A23B72')
plt.axvline(x=best_iter_s2, color='#F18F01', linestyle='--',
            label=f'Best Iteration ({best_iter_s2})', alpha=0.7, linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Multi-class Log Loss', fontsize=12)
plt.title('Stage 2: Multi-class Classification (DDoS/DoS/Recon)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

plot_file_s2 = os.path.join(PLOT_DIR, f"stage2_loss_curve_{timestamp}.png")
plt.savefig(plot_file_s2, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {os.path.basename(plot_file_s2)}")
plt.close()

# ============================================================================
# 4. COMBINED PLOT (Side by Side)
# ============================================================================

print("\n" + "="*80)
print("4. CREATING COMBINED PLOT")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Stage 1
ax1.plot(train_loss_s1, label='Training', linewidth=2, alpha=0.8, color='#2E86AB')
ax1.plot(val_loss_s1, label='Validation', linewidth=2, alpha=0.8, color='#A23B72')
ax1.axvline(x=best_iter_s1, color='#F18F01', linestyle='--', alpha=0.7, linewidth=2)
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Log Loss', fontsize=11)
ax1.set_title('Stage 1: Binary Classification', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')

# Stage 2
ax2.plot(train_loss_s2, label='Training', linewidth=2, alpha=0.8, color='#2E86AB')
ax2.plot(val_loss_s2, label='Validation', linewidth=2, alpha=0.8, color='#A23B72')
ax2.axvline(x=best_iter_s2, color='#F18F01', linestyle='--', alpha=0.7, linewidth=2)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Multi-class Log Loss', fontsize=11)
ax2.set_title('Stage 2: Multi-class Classification', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plot_file_combined = os.path.join(PLOT_DIR, f"combined_loss_curves_{timestamp}.png")
plt.savefig(plot_file_combined, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {os.path.basename(plot_file_combined)}")
plt.close()

# ============================================================================
# 5. CONVERGENCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("5. CONVERGENCE ANALYSIS")
print("="*80)

# Stage 1 analysis
final_train_loss_s1 = train_loss_s1[-1]
final_val_loss_s1 = val_loss_s1[-1]
min_val_loss_s1 = min(val_loss_s1)
overfit_gap_s1 = final_val_loss_s1 - final_train_loss_s1

print(f"\nStage 1:")
print(f"   Final train loss: {final_train_loss_s1:.6f}")
print(f"   Final val loss:   {final_val_loss_s1:.6f}")
print(f"   Min val loss:     {min_val_loss_s1:.6f}")
print(f"   Overfit gap:      {overfit_gap_s1:.6f}")
print(f"   Best iteration:   {best_iter_s1} / {len(train_loss_s1)}")

if overfit_gap_s1 < 0.05:
    print(f"   ✅ No significant overfitting detected")
elif overfit_gap_s1 < 0.1:
    print(f"   ⚠️  Mild overfitting detected")
else:
    print(f"   ❌ Significant overfitting detected")

# Stage 2 analysis
final_train_loss_s2 = train_loss_s2[-1]
final_val_loss_s2 = val_loss_s2[-1]
min_val_loss_s2 = min(val_loss_s2)
overfit_gap_s2 = final_val_loss_s2 - final_train_loss_s2

print(f"\nStage 2:")
print(f"   Final train loss: {final_train_loss_s2:.6f}")
print(f"   Final val loss:   {final_val_loss_s2:.6f}")
print(f"   Min val loss:     {min_val_loss_s2:.6f}")
print(f"   Overfit gap:      {overfit_gap_s2:.6f}")
print(f"   Best iteration:   {best_iter_s2} / {len(train_loss_s2)}")

if overfit_gap_s2 < 0.05:
    print(f"   ✅ No significant overfitting detected")
elif overfit_gap_s2 < 0.1:
    print(f"   ⚠️  Mild overfitting detected")
else:
    print(f"   ❌ Significant overfitting detected")

# ============================================================================
# 6. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ PLOTTING COMPLETED")
print("="*80)

print(f"\n📂 Generated plots:")
print(f"   • {os.path.basename(plot_file_s1)}")
print(f"   • {os.path.basename(plot_file_s2)}")
print(f"   • {os.path.basename(plot_file_combined)}")

print(f"\n📂 Saved to: {PLOT_DIR}/")

print(f"\n📊 Training Summary:")
print(f"   Stage 1: {len(train_loss_s1)} iterations, best at {best_iter_s1}")
print(f"   Stage 2: {len(train_loss_s2)} iterations, best at {best_iter_s2}")
print(f"   Training time: Stage 1 {metrics['stage1']['training_time_sec']:.1f}s, Stage 2 {metrics['stage2']['training_time_sec']:.1f}s")

print("\n" + "="*80)
