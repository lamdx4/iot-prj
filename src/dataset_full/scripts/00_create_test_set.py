"""
Step 0: Create Balanced Test Set
==================================

Create a balanced test set with all 4 attack categories from multiple batch files.

Current problem:
- Test set (batch_02) only has DoS (99.99%) and Normal (0.003%)
- Missing: DDoS and Reconnaissance
- Cannot evaluate Stage 2 model properly

Solution:
- Sample from multiple batches to get all 4 categories
- Target: 100K samples
- Distribution: DDoS 35%, DoS 50%, Recon 13%, Normal 2%

Output:
- Data/Dataset/test_balanced_100k.csv
- src/dataset_full/stats/test_set_statistics.json
- src/dataset_full/stats/test_set_summary.txt

Author: Lambda Team
Date: December 2025
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

print("="*80)
print("STEP 0: CREATE BALANCED TEST SET")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Auto-detect project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))

# Paths
BATCH_DIR = os.path.join(PROJECT_ROOT, "Data/Dataset/merged_batches")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Data/Dataset")
STATS_DIR = os.path.join(PROJECT_ROOT, "src/dataset_full/stats")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

print(f"\n📂 Configuration:")
print(f"   Batch dir:  {BATCH_DIR}")
print(f"   Output dir: {OUTPUT_DIR}")
print(f"   Stats dir:  {STATS_DIR}")

# ============================================================================
# SAMPLING PLAN
# ============================================================================

SAMPLING_PLAN = {
    'batch_01': {
        'Reconnaissance': 13_000,  # Only batch with Recon
        'DoS': 15_000,
        'Normal': 1_500,           # Most Normal samples
    },
    'batch_02': {
        'DoS': 15_000,
        'Normal': 100,
    },
    'batch_03': {
        'DoS': 10_000,
        'Normal': 100,
    },
    'batch_04': {
        'DDoS': 20_000,            # Has both DDoS and DoS
        'DoS': 10_000,
        'Normal': 100,
    },
    'batch_05': {
        'DDoS': 8_000,
        'Normal': 50,
    },
    'batch_06': {
        'DDoS': 7_000,
        'Normal': 50,
    },
    'batch_07': {
        'Normal': 50,
    },
    'batch_08': {
        'Normal': 50,
    },
}

# Calculate totals
total_samples = 0
category_totals = {}
for batch, categories in SAMPLING_PLAN.items():
    for category, count in categories.items():
        total_samples += count
        category_totals[category] = category_totals.get(category, 0) + count

print(f"\n📊 Sampling Plan:")
print(f"   Total samples: {total_samples:,}")
print(f"   Distribution:")
for cat in sorted(category_totals.keys()):
    count = category_totals[cat]
    pct = count / total_samples * 100
    print(f"      {cat:15s}: {count:6,} ({pct:5.2f}%)")

# ============================================================================
# SAMPLE FROM BATCHES
# ============================================================================

print(f"\n{'='*80}")
print("SAMPLING FROM BATCHES")
print("="*80)

sampled_dfs = []
batch_stats = {}

for batch_name, categories in SAMPLING_PLAN.items():
    print(f"\n{'='*80}")
    print(f"BATCH: {batch_name}")
    print("="*80)
    
    batch_file = os.path.join(BATCH_DIR, f"{batch_name}.csv")
    
    if not os.path.exists(batch_file):
        print(f"   ⚠️  File not found: {batch_file}")
        continue
    
    print(f"   📂 Loading {batch_name}...")
    df_batch = pd.read_csv(batch_file, low_memory=False)
    print(f"   ✅ Loaded: {len(df_batch):,} records")
    
    batch_samples = []
    
    for category, n_samples in categories.items():
        print(f"\n   🎯 Sampling {category}: {n_samples:,} samples")
        
        # Filter by category
        df_cat = df_batch[df_batch['category'] == category]
        available = len(df_cat)
        
        print(f"      Available: {available:,}")
        
        if available == 0:
            print(f"      ❌ No samples available for {category}")
            continue
        
        if available < n_samples:
            print(f"      ⚠️  Only {available} available, taking all")
            sampled = df_cat
        else:
            # Random sample
            sampled = df_cat.sample(n=n_samples, random_state=42)
            print(f"      ✅ Sampled: {len(sampled):,}")
        
        batch_samples.append(sampled)
    
    if batch_samples:
        df_batch_sampled = pd.concat(batch_samples, ignore_index=True)
        sampled_dfs.append(df_batch_sampled)
        
        # Store batch stats
        batch_stats[batch_name] = {
            'total_sampled': len(df_batch_sampled),
            'category_distribution': df_batch_sampled['category'].value_counts().to_dict()
        }
        
        print(f"\n   ✅ Total from {batch_name}: {len(df_batch_sampled):,}")

# ============================================================================
# MERGE AND SHUFFLE
# ============================================================================

print(f"\n{'='*80}")
print("MERGING AND SHUFFLING")
print("="*80)

print(f"\n🔧 Merging {len(sampled_dfs)} batch samples...")
df_test = pd.concat(sampled_dfs, ignore_index=True)

print(f"✅ Total samples: {len(df_test):,}")

print(f"\n🔀 Shuffling...")
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✅ Shuffled")

# ============================================================================
# VERIFY DISTRIBUTION
# ============================================================================

print(f"\n{'='*80}")
print("DISTRIBUTION VERIFICATION")
print("="*80)

category_counts = df_test['category'].value_counts()

print(f"\n📊 Final Distribution:")
for cat in sorted(category_counts.index):
    count = category_counts[cat]
    pct = count / len(df_test) * 100
    expected = category_totals.get(cat, 0)
    diff = count - expected
    status = "✅" if abs(diff) <= 10 else "⚠️"
    print(f"   {status} {cat:15s}: {count:6,} ({pct:5.2f}%) - Expected: {expected:,} (diff: {diff:+d})")

# ============================================================================
# SAVE TEST SET
# ============================================================================

print(f"\n{'='*80}")
print("SAVING TEST SET")
print("="*80)

output_file = os.path.join(OUTPUT_DIR, "test_balanced_100k.csv")
print(f"\n💾 Saving to: {output_file}")

df_test.to_csv(output_file, index=False)

file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"✅ Saved! Size: {file_size_mb:.2f} MB")

# ============================================================================
# EXPORT STATISTICS
# ============================================================================

print(f"\n{'='*80}")
print("EXPORTING STATISTICS")
print("="*80)

# Calculate detailed statistics
stats = {
    'metadata': {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_records': len(df_test),
        'file_path': output_file,
        'file_size_mb': round(file_size_mb, 2),
    },
    'category_distribution': {
        cat: {
            'count': int(count),
            'percentage': round(count / len(df_test) * 100, 4)
        }
        for cat, count in category_counts.items()
    },
    'sampling_plan': {
        batch: {
            cat: count for cat, count in categories.items()
        }
        for batch, categories in SAMPLING_PLAN.items()
    },
    'batch_contributions': batch_stats,
    'feature_info': {
        'num_features': len(df_test.columns),
        'feature_names': list(df_test.columns),
        'numeric_features': len(df_test.select_dtypes(include=['number']).columns),
        'categorical_features': len(df_test.select_dtypes(include=['object']).columns),
    },
    'missing_values': {
        'total': int(df_test.isnull().sum().sum()),
        'percentage': round(df_test.isnull().sum().sum() / (len(df_test) * len(df_test.columns)) * 100, 4)
    }
}

# Save JSON
stats_json_file = os.path.join(STATS_DIR, "test_set_statistics.json")
with open(stats_json_file, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"\n✅ Statistics JSON saved to: {stats_json_file}")

# Save human-readable summary
stats_txt_file = os.path.join(STATS_DIR, "test_set_summary.txt")
with open(stats_txt_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("TEST SET STATISTICS SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Created: {stats['metadata']['created_at']}\n")
    f.write(f"File: {output_file}\n")
    f.write(f"Size: {file_size_mb:.2f} MB\n\n")
    
    f.write("DISTRIBUTION\n")
    f.write("-"*80 + "\n")
    f.write(f"Total records: {len(df_test):,}\n\n")
    
    for cat in sorted(category_counts.index):
        count = category_counts[cat]
        pct = count / len(df_test) * 100
        f.write(f"{cat:15s}: {count:6,} ({pct:5.2f}%)\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("SAMPLING SOURCES\n")
    f.write("-"*80 + "\n\n")
    
    for batch, categories in SAMPLING_PLAN.items():
        f.write(f"{batch}:\n")
        for cat, count in categories.items():
            f.write(f"  {cat:15s}: {count:,}\n")
        f.write("\n")
    
    f.write("-"*80 + "\n")
    f.write("FEATURES\n")
    f.write("-"*80 + "\n\n")
    f.write(f"Total features: {len(df_test.columns)}\n")
    f.write(f"Numeric: {len(df_test.select_dtypes(include=['number']).columns)}\n")
    f.write(f"Categorical: {len(df_test.select_dtypes(include=['object']).columns)}\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("MISSING VALUES\n")
    f.write("-"*80 + "\n\n")
    missing_total = df_test.isnull().sum().sum()
    missing_pct = missing_total / (len(df_test) * len(df_test.columns)) * 100
    f.write(f"Total: {missing_total:,} ({missing_pct:.2f}%)\n")

print(f"✅ Summary TXT saved to: {stats_txt_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("✅ TEST SET CREATION COMPLETED!")
print("="*80)

print(f"\n📊 Created balanced test set:")
print(f"   • Records: {len(df_test):,}")
print(f"   • Size: {file_size_mb:.2f} MB")
print(f"   • File: {output_file}")

print(f"\n📈 Distribution:")
for cat in sorted(category_counts.index):
    count = category_counts[cat]
    pct = count / len(df_test) * 100
    print(f"   • {cat:15s}: {count:6,} ({pct:5.2f}%)")

print(f"\n📁 Statistics exported:")
print(f"   • JSON: {stats_json_file}")
print(f"   • TXT:  {stats_txt_file}")

print(f"\n{'='*80}")
print("NEXT: Update 03_train_colab_highmem.py to use test_balanced_100k.csv")
print("="*80)
