"""
Step 2: Analyze batch files and create statistics JSON
=======================================================

Analyze each batch file:
- Number of records
- Class distribution (Normal, DDoS, DoS, Reconnaissance)
- Attack distribution
- Missing values
- Feature statistics
- Time range (stime, ltime)

Output: JSON file with all statistics for training strategy

Author: Lambda Team  
Date: October 2025
"""

import pandas as pd
import numpy as np
import glob
import os
import json
from datetime import datetime

print("="*80)
print("STEP 2: ANALYZE BATCH FILES")
print("="*80)

# Auto-detect project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))

# Paths relative to project root
BATCH_DIR = os.getenv('BATCH_DIR', os.path.join(PROJECT_ROOT, "Data/Dataset/merged_batches"))
STATS_DIR = os.getenv('STATS_DIR', os.path.join(PROJECT_ROOT, "src/dataset_full/stats"))

print(f"\n📂 Detected Paths:")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   Batch dir:    {BATCH_DIR}")
print(f"   Stats dir:    {STATS_DIR}")

os.makedirs(STATS_DIR, exist_ok=True)

# Get all batch files
batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, "batch_*.csv")))

print(f"\n✅ Found {len(batch_files)} batch files")

# Statistics container
all_stats = {
    'metadata': {
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_batches': len(batch_files),
        'batch_dir': BATCH_DIR
    },
    'batches': {},
    'overall': {}
}

# Analyze each batch
overall_records = 0
overall_categories = {}
overall_normal = 0
overall_attacks = 0

for batch_file in batch_files:
    batch_name = os.path.basename(batch_file).replace('.csv', '')
    
    print(f"\n{'='*80}")
    print(f"ANALYZING: {batch_name}")
    print(f"{'='*80}")
    
    # Load batch
    print(f"  📂 Loading {batch_name}...")
    df = pd.read_csv(batch_file)
    
    print(f"  ✅ Loaded: {len(df):,} records, {len(df.columns)} columns")
    
    # Basic stats
    batch_stats = {
        'file': batch_name,
        'num_records': len(df),
        'num_features': len(df.columns),
        'file_size_mb': os.path.getsize(batch_file) / (1024 * 1024)
    }
    
    # Category distribution
    print(f"\n  📊 Category Distribution:")
    category_dist = df['category'].value_counts().to_dict()
    batch_stats['category_distribution'] = category_dist
    
    for cat, count in category_dist.items():
        pct = count / len(df) * 100
        print(f"     {cat:15s}: {count:,} ({pct:.2f}%)")
        
        # Update overall
        overall_categories[cat] = overall_categories.get(cat, 0) + count
    
    # Attack vs Normal
    num_normal = category_dist.get('Normal', 0)
    num_attacks = len(df) - num_normal
    
    batch_stats['num_normal'] = num_normal
    batch_stats['num_attacks'] = num_attacks
    batch_stats['normal_percentage'] = (num_normal / len(df) * 100) if len(df) > 0 else 0
    batch_stats['attack_percentage'] = (num_attacks / len(df) * 100) if len(df) > 0 else 0
    
    if num_normal > 0:
        batch_stats['imbalance_ratio'] = num_attacks / num_normal
    else:
        batch_stats['imbalance_ratio'] = float('inf')
    
    print(f"\n  ⚖️  Normal vs Attack:")
    print(f"     Normal:  {num_normal:,} ({batch_stats['normal_percentage']:.2f}%)")
    print(f"     Attack:  {num_attacks:,} ({batch_stats['attack_percentage']:.2f}%)")
    if num_normal > 0:
        print(f"     Imbalance Ratio: {batch_stats['imbalance_ratio']:.1f}:1 (Attack:Normal)")
    
    overall_normal += num_normal
    overall_attacks += num_attacks
    
    # Missing values
    missing_total = df.isnull().sum().sum()
    batch_stats['missing_values'] = {
        'total': int(missing_total),
        'percentage': float(missing_total / (len(df) * len(df.columns)) * 100)
    }
    
    print(f"\n  🔍 Missing Values:")
    print(f"     Total: {missing_total:,} ({batch_stats['missing_values']['percentage']:.2f}%)")
    
    # Time range (flow data)
    if 'stime' in df.columns and 'ltime' in df.columns:
        batch_stats['time_range'] = {
            'stime_min': float(df['stime'].min()),
            'stime_max': float(df['stime'].max()),
            'ltime_min': float(df['ltime'].min()),
            'ltime_max': float(df['ltime'].max()),
            'duration_seconds': float(df['stime'].max() - df['stime'].min())
        }
        
        print(f"\n  ⏱️  Time Range:")
        print(f"     Start: {df['stime'].min():.2f}")
        print(f"     End:   {df['stime'].max():.2f}")
        print(f"     Duration: {batch_stats['time_range']['duration_seconds']:.2f} seconds")
    
    # Feature statistics (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    batch_stats['numeric_features'] = len(numeric_cols)
    
    # Categorical features
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    batch_stats['categorical_features'] = len(cat_cols)
    
    print(f"\n  📈 Features:")
    print(f"     Numeric: {batch_stats['numeric_features']}")
    print(f"     Categorical: {batch_stats['categorical_features']}")
    
    # Protocol distribution
    if 'proto' in df.columns:
        proto_dist = df['proto'].value_counts().to_dict()
        batch_stats['protocol_distribution'] = proto_dist
        
        print(f"\n  🌐 Protocol Distribution:")
        for proto, count in list(proto_dist.items())[:5]:  # Top 5
            pct = count / len(df) * 100
            print(f"     {proto:10s}: {count:,} ({pct:.2f}%)")
    
    # Save batch stats
    all_stats['batches'][batch_name] = batch_stats
    overall_records += len(df)

# Overall statistics
print(f"\n{'='*80}")
print(f"OVERALL STATISTICS")
print(f"{'='*80}")

all_stats['overall'] = {
    'total_records': overall_records,
    'total_normal': overall_normal,
    'total_attacks': overall_attacks,
    'normal_percentage': (overall_normal / overall_records * 100) if overall_records > 0 else 0,
    'attack_percentage': (overall_attacks / overall_records * 100) if overall_records > 0 else 0,
    'imbalance_ratio': (overall_attacks / overall_normal) if overall_normal > 0 else float('inf'),
    'category_distribution': overall_categories
}

print(f"\n📊 Total Records: {overall_records:,}")
print(f"\n📊 Category Distribution:")
for cat, count in overall_categories.items():
    pct = count / overall_records * 100
    print(f"   {cat:15s}: {count:,} ({pct:.2f}%)")

print(f"\n⚖️  Overall Normal vs Attack:")
print(f"   Normal:  {overall_normal:,} ({all_stats['overall']['normal_percentage']:.2f}%)")
print(f"   Attack:  {overall_attacks:,} ({all_stats['overall']['attack_percentage']:.2f}%)")
if overall_normal > 0:
    print(f"   Imbalance Ratio: {all_stats['overall']['imbalance_ratio']:.1f}:1 (Attack:Normal)")

# Recommendations for training
print(f"\n{'='*80}")
print(f"TRAINING RECOMMENDATIONS")
print(f"{'='*80}")

recommendations = []

# Check imbalance
if all_stats['overall']['imbalance_ratio'] > 100:
    recommendations.append({
        'category': 'Imbalance',
        'severity': 'HIGH',
        'message': f"Extreme imbalance ({all_stats['overall']['imbalance_ratio']:.0f}:1). Use SMOTE or class weights.",
        'action': 'Apply SMOTE with sampling_strategy=0.1 or use scale_pos_weight in XGBoost'
    })

# Check Normal samples
if overall_normal < 1000:
    recommendations.append({
        'category': 'Normal Samples',
        'severity': 'HIGH',
        'message': f"Very few Normal samples ({overall_normal}). Evaluation may not be reliable.",
        'action': 'Consider using all Normal samples for validation/test. Use stratified split.'
    })

# Dataset size
if overall_records > 10000000:  # > 10M
    recommendations.append({
        'category': 'Dataset Size',
        'severity': 'MEDIUM',
        'message': f"Large dataset ({overall_records:,} records). May need sampling or incremental learning.",
        'action': 'Consider training on subset or use incremental learning with multiple epochs.'
    })

# Batch recommendation
normal_batches = []
for batch_name, stats in all_stats['batches'].items():
    if stats['num_normal'] > 0:
        normal_batches.append((batch_name, stats['num_normal']))

if normal_batches:
    recommendations.append({
        'category': 'Batch Selection',
        'severity': 'INFO',
        'message': f"{len(normal_batches)} batches contain Normal samples.",
        'action': f"Priority batches: {', '.join([b[0] for b in sorted(normal_batches, key=lambda x: x[1], reverse=True)[:3]])}"
    })

all_stats['recommendations'] = recommendations

for rec in recommendations:
    print(f"\n⚠️  {rec['severity']}: {rec['category']}")
    print(f"   {rec['message']}")
    print(f"   💡 Action: {rec['action']}")

# Save JSON
output_json = os.path.join(STATS_DIR, "batch_statistics.json")
print(f"\n{'='*80}")
print(f"SAVING STATISTICS")
print(f"{'='*80}")

with open(output_json, 'w') as f:
    json.dump(all_stats, f, indent=2)

print(f"\n✅ Statistics saved to: {output_json}")

# Also save a readable summary
summary_file = os.path.join(STATS_DIR, "batch_summary.txt")
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("BATCH STATISTICS SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Created: {all_stats['metadata']['created_at']}\n")
    f.write(f"Number of batches: {all_stats['metadata']['num_batches']}\n\n")
    
    f.write("OVERALL STATISTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Total records: {all_stats['overall']['total_records']:,}\n")
    f.write(f"Normal: {all_stats['overall']['total_normal']:,} ({all_stats['overall']['normal_percentage']:.2f}%)\n")
    f.write(f"Attack: {all_stats['overall']['total_attacks']:,} ({all_stats['overall']['attack_percentage']:.2f}%)\n")
    if all_stats['overall']['total_normal'] > 0:
        f.write(f"Imbalance Ratio: {all_stats['overall']['imbalance_ratio']:.1f}:1\n")
    
    f.write("\nCATEGORY DISTRIBUTION\n")
    f.write("-"*80 + "\n")
    for cat, count in all_stats['overall']['category_distribution'].items():
        pct = count / all_stats['overall']['total_records'] * 100
        f.write(f"{cat:15s}: {count:,} ({pct:.2f}%)\n")
    
    f.write("\nBATCH DETAILS\n")
    f.write("-"*80 + "\n")
    for batch_name in sorted(all_stats['batches'].keys()):
        stats = all_stats['batches'][batch_name]
        f.write(f"\n{batch_name}:\n")
        f.write(f"  Records: {stats['num_records']:,}\n")
        f.write(f"  Normal:  {stats['num_normal']:,} ({stats['normal_percentage']:.2f}%)\n")
        f.write(f"  Attack:  {stats['num_attacks']:,} ({stats['attack_percentage']:.2f}%)\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("TRAINING RECOMMENDATIONS\n")
    f.write("="*80 + "\n")
    for rec in recommendations:
        f.write(f"\n[{rec['severity']}] {rec['category']}\n")
        f.write(f"  {rec['message']}\n")
        f.write(f"  Action: {rec['action']}\n")

print(f"✅ Summary saved to: {summary_file}")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETED!")
print("="*80)
print(f"\n📊 Statistics JSON: {output_json}")
print(f"📄 Summary TXT: {summary_file}")
print("\n" + "="*80)
print("NEXT: Run 03_train_hierarchical.py to train model")
print("="*80)


