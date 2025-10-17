"""
Step 1: Merge 10 files into 1 batch file
=========================================

74 files → 8 batch files (10 files per batch, last batch has 4 files)
Sort files correctly (1,2,3... not 1,10,11...)
Sort by stime (flow data)

Author: Lambda Team
Date: October 2025
"""

import pandas as pd
import glob
import os
from pathlib import Path
import re

print("="*80)
print("STEP 1: MERGE FILES INTO BATCHES")
print("="*80)

# Auto-detect project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')))

# Paths relative to project root
DATA_DIR = os.getenv('DATA_DIR', os.path.join(PROJECT_ROOT, "Data/Dataset/Entire Dataset"))
OUTPUT_DIR = os.getenv('OUTPUT_DIR', os.path.join(PROJECT_ROOT, "Data/Dataset/merged_batches"))

print(f"\n📂 Detected Paths:")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   Data dir:     {DATA_DIR}")
print(f"   Output dir:   {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all files
all_files = glob.glob(os.path.join(DATA_DIR, "UNSW_2018_IoT_Botnet_Dataset_*.csv"))
all_files = [f for f in all_files if "Feature_Names" not in f]

# Sort by number (extract number from filename)
def extract_number(filepath):
    """Extract number from filename for correct sorting"""
    filename = os.path.basename(filepath)
    # Extract: UNSW_2018_IoT_Botnet_Dataset_42.csv → 42
    match = re.search(r'Dataset_(\d+)\.csv', filename)
    if match:
        return int(match.group(1))
    return 0

all_files = sorted(all_files, key=extract_number)

print(f"\n✅ Found {len(all_files)} files")
print(f"📊 First file: {os.path.basename(all_files[0])}")
print(f"📊 Last file:  {os.path.basename(all_files[-1])}")

# Load column names
feature_names_file = os.path.join(DATA_DIR, "UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv")
with open(feature_names_file, 'r') as f:
    column_names = f.read().strip().split(',')

print(f"\n✅ Loaded {len(column_names)} column names")

# Batch size
BATCH_SIZE = 10
num_batches = (len(all_files) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"\n📊 Batch configuration:")
print(f"  Files per batch: {BATCH_SIZE}")
print(f"  Total batches:   {num_batches}")

# Merge files in batches
for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(all_files))
    batch_files = all_files[start_idx:end_idx]
    
    print(f"\n{'='*80}")
    print(f"BATCH {batch_idx + 1}/{num_batches}")
    print(f"{'='*80}")
    print(f"Files {start_idx + 1} to {end_idx} ({len(batch_files)} files)")
    
    # Load and merge
    dfs = []
    total_records = 0
    
    for i, filepath in enumerate(batch_files, 1):
        filename = os.path.basename(filepath)
        file_num = extract_number(filepath)
        
        print(f"  [{i}/{len(batch_files)}] Loading file #{file_num}: {filename}...", end=" ")
        
        try:
            df = pd.read_csv(filepath, header=None, names=column_names)
            dfs.append(df)
            total_records += len(df)
            print(f"✅ {len(df):,} records")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Concatenate
    print(f"\n  🔧 Merging {len(dfs)} files...")
    merged_df = pd.concat(dfs, ignore_index=True)
    
    print(f"  ✅ Total records: {len(merged_df):,}")
    
    # Sort by stime (flow data)
    print(f"  🔧 Sorting by stime (flow data)...")
    merged_df = merged_df.sort_values('stime').reset_index(drop=True)
    
    # Save
    output_file = os.path.join(OUTPUT_DIR, f"batch_{batch_idx + 1:02d}.csv")
    print(f"  💾 Saving to: {os.path.basename(output_file)}...")
    merged_df.to_csv(output_file, index=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  ✅ Saved! Size: {file_size_mb:.2f} MB")
    
    # Show statistics
    print(f"\n  📊 Batch {batch_idx + 1} Statistics:")
    print(f"     Category distribution:")
    cat_dist = merged_df['category'].value_counts()
    for cat, count in cat_dist.items():
        pct = count / len(merged_df) * 100
        print(f"       {cat:15s}: {count:,} ({pct:.2f}%)")

print("\n" + "="*80)
print("✅ MERGING COMPLETED!")
print("="*80)

# Summary
batch_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "batch_*.csv")))
print(f"\n📊 Summary:")
print(f"  Created batches: {len(batch_files)}")
print(f"  Location: {OUTPUT_DIR}")
print(f"\n  Batch files:")
for bf in batch_files:
    size_mb = os.path.getsize(bf) / (1024 * 1024)
    print(f"    {os.path.basename(bf):20s} - {size_mb:7.2f} MB")

total_size = sum(os.path.getsize(bf) for bf in batch_files) / (1024 * 1024)
print(f"\n  Total size: {total_size:.2f} MB")

print("\n" + "="*80)
print("NEXT: Run 02_analyze_batches.py to create statistics JSON")
print("="*80)


