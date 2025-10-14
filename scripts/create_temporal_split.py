#!/usr/bin/env python3
"""
Create temporal train/val/test split from entire Bot-IoT dataset
Ensures no temporal leakage by splitting chronologically
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import yaml
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_multiple_files
from src.data.schema import TARGET_COL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_all_files(data_dir: Path, sample_frac: float = 1.0, nrows_per_file: int = None) -> pd.DataFrame:
    """
    Load all dataset files
    
    Args:
        data_dir: Directory containing CSV files
        sample_frac: Fraction of data to load (0.0-1.0)
        nrows_per_file: Rows per file (if set, overrides sample_frac)
        
    Returns:
        Concatenated DataFrame
    """
    logger.info(f"Loading data from {data_dir}...")
    
    file_pattern = str(data_dir / "UNSW_2018_IoT_Botnet_Dataset_*.csv")
    
    # Determine how many rows to load
    if nrows_per_file:
        rows_to_load = nrows_per_file
        logger.info(f"  Loading {rows_to_load:,} rows per file")
    elif sample_frac < 1.0:
        rows_to_load = int(1_000_000 * sample_frac)  # Each file has ~1M rows
        logger.info(f"  Sampling {sample_frac:.1%} = {rows_to_load:,} rows per file")
    else:
        rows_to_load = None
        logger.info(f"  Loading all rows (full dataset)")
    
    df = load_multiple_files(
        file_pattern,
        nrows_per_file=rows_to_load
    )
    
    logger.info(f"✓ Loaded {len(df):,} total rows")
    
    # Convert stime to datetime
    logger.info("Parsing timestamps...")
    df['stime_dt'] = pd.to_datetime(df['stime'], unit='s', errors='coerce')
    
    # Sort by time (CRITICAL for temporal split)
    logger.info("Sorting by time...")
    df = df.sort_values('stime_dt').reset_index(drop=True)
    
    logger.info(f"✓ Time range: {df['stime_dt'].min()} to {df['stime_dt'].max()}")
    
    return df


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple:
    """
    Split data temporally (chronologically)
    
    Args:
        df: DataFrame sorted by time
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        
    Returns:
        train_df, val_df, test_df
    """
    logger.info(f"Splitting temporally ({train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%})...")
    
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split at boundaries
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train+n_val].copy()
    test_df = df.iloc[n_train+n_val:].copy()
    
    logger.info(f"✓ Train: {len(train_df):,} rows ({len(train_df)/n_total:.1%})")
    logger.info(f"✓ Val:   {len(val_df):,} rows ({len(val_df)/n_total:.1%})")
    logger.info(f"✓ Test:  {len(test_df):,} rows ({len(test_df)/n_total:.1%})")
    
    return train_df, val_df, test_df


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Verify no temporal leakage between splits
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
    """
    logger.info("Verifying no temporal leakage...")
    
    train_max = train_df['stime_dt'].max()
    val_min = val_df['stime_dt'].min()
    val_max = val_df['stime_dt'].max()
    test_min = test_df['stime_dt'].min()
    
    logger.info(f"  Train: {train_df['stime_dt'].min()} to {train_max}")
    logger.info(f"  Val:   {val_min} to {val_max}")
    logger.info(f"  Test:  {test_min} to {test_df['stime_dt'].max()}")
    
    # Check for overlap
    if train_max > val_min:
        raise ValueError(f"❌ TEMPORAL LEAKAGE: Train overlaps Val! train_max={train_max} > val_min={val_min}")
    
    if val_max > test_min:
        raise ValueError(f"❌ TEMPORAL LEAKAGE: Val overlaps Test! val_max={val_max} > test_min={test_min}")
    
    logger.info("✓ No temporal leakage detected!")


def verify_class_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Verify class distribution in each split
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
    """
    logger.info("\nClass distribution:")
    
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if TARGET_COL not in df.columns:
            continue
        
        # Convert to numeric to ensure consistent types
        df_attack = pd.to_numeric(df[TARGET_COL], errors='coerce')
        dist = df_attack.value_counts()
        pct = df_attack.value_counts(normalize=True) * 100
        
        logger.info(f"  {name}:")
        for cls in sorted(dist.index):
            logger.info(f"    Class {int(cls)}: {dist[cls]:,} ({pct[cls]:.2f}%)")


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path):
    """
    Save splits to CSV files
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving splits to {output_dir}...")
    
    # Drop temporary datetime columns
    for df in [train_df, val_df, test_df]:
        if 'stime_dt' in df.columns:
            df.drop('stime_dt', axis=1, inplace=True)
        if 'ltime_dt' in df.columns:
            df.drop('ltime_dt', axis=1, inplace=True)
    
    # Save
    train_file = output_dir / "train.csv"
    val_file = output_dir / "val.csv"
    test_file = output_dir / "test.csv"
    
    train_df.to_csv(train_file, index=False)
    logger.info(f"  ✓ Train: {train_file} ({train_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    val_df.to_csv(val_file, index=False)
    logger.info(f"  ✓ Val:   {val_file} ({val_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    test_df.to_csv(test_file, index=False)
    logger.info(f"  ✓ Test:  {test_file} ({test_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    logger.info(f"\n✓ Saved splits successfully!")


def main():
    parser = argparse.ArgumentParser(description='Create temporal train/val/test split')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing dataset files')
    parser.add_argument('--output-dir', type=str, default='Data/processed/temporal_split',
                        help='Output directory for splits')
    parser.add_argument('--sample-frac', type=float, default=1.0,
                        help='Fraction of data to use (0.0-1.0, default: 1.0 = all data)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Temporal Split Creation")
    logger.info("="*60)
    
    # Load data
    try:
        df = load_all_files(
            Path(args.data_dir),
            sample_frac=args.sample_frac
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Split
    try:
        train_df, val_df, test_df = temporal_split(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        return 1
    
    # Verify no leakage
    try:
        verify_no_leakage(train_df, val_df, test_df)
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    # Verify class distribution
    verify_class_distribution(train_df, val_df, test_df)
    
    # Save
    try:
        save_splits(train_df, val_df, test_df, Path(args.output_dir))
    except Exception as e:
        logger.error(f"Failed to save splits: {e}")
        return 1
    
    logger.info("\n" + "="*60)
    logger.info("✓ Temporal split complete!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info(f"  1. Update config/base_config.yaml:")
    logger.info(f"     data_dir: '{args.output_dir}'")
    logger.info(f"  2. Proceed to temporal feature engineering")
    
    return 0


if __name__ == '__main__':
    exit(main())
