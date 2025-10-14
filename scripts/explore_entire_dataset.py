#!/usr/bin/env python3
"""
Explore the entire Bot-IoT dataset
Load sample data, analyze structure, and document findings
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import json
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_multiple_files
from src.data.schema import ALL_COLUMNS, TEMPORAL_COLS, TARGET_COL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_timestamps(df: pd.DataFrame) -> dict:
    """
    Analyze timestamp columns
    
    Args:
        df: DataFrame with timestamp columns
        
    Returns:
        Dictionary of timestamp statistics
    """
    logger.info("Analyzing timestamps...")
    
    results = {}
    
    for col in ['stime', 'ltime', 'dur']:
        if col not in df.columns:
            logger.warning(f"Column {col} not found")
            continue
        
        if col in ['stime', 'ltime']:
            # Convert to datetime
            df[f'{col}_dt'] = pd.to_datetime(df[col], unit='s', errors='coerce')
            
            results[col] = {
                'min': str(df[f'{col}_dt'].min()),
                'max': str(df[f'{col}_dt'].max()),
                'range_days': (df[f'{col}_dt'].max() - df[f'{col}_dt'].min()).days,
                'null_count': int(df[f'{col}_dt'].isnull().sum()),
                'format': 'unix_timestamp'
            }
        else:  # dur
            results[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'null_count': int(df[col].isnull().sum())
            }
    
    logger.info(f"✓ Timestamp analysis complete")
    return results


def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Comprehensive dataset analysis
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary of analysis results
    """
    logger.info("Analyzing dataset...")
    
    results = {
        'shape': {
            'rows': len(df),
            'columns': len(df.columns)
        },
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Class distribution
    if TARGET_COL in df.columns:
        class_dist = df[TARGET_COL].value_counts().to_dict()
        results['class_distribution'] = {
            'counts': class_dist,
            'percentages': {k: v/len(df)*100 for k, v in class_dist.items()}
        }
    
    # Category distribution (if exists)
    if 'category' in df.columns:
        cat_dist = df['category'].value_counts().to_dict()
        results['category_distribution'] = dict(list(cat_dist.items())[:10])  # Top 10
    
    # Numerical statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results['numerical_stats'] = {}
    for col in numeric_cols[:10]:  # First 10 numeric columns
        results['numerical_stats'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }
    
    logger.info(f"✓ Dataset analysis complete")
    return results


def main():
    parser = argparse.ArgumentParser(description='Explore entire Bot-IoT dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing dataset files')
    parser.add_argument('--n-rows', type=int, default=100000,
                        help='Number of rows to load per file')
    parser.add_argument('--n-files', type=int, default=10,
                        help='Number of files to load')
    parser.add_argument('--output', type=str, default='results/exploration/entire_dataset_exploration.json',
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Bot-IoT Dataset Exploration")
    logger.info("="*60)
    
    # Load data
    logger.info(f"\nLoading data from: {args.data_dir}")
    logger.info(f"  - Max {args.n_files} files")
    logger.info(f"  - {args.n_rows:,} rows per file")
    
    file_pattern = f"{args.data_dir}/UNSW_2018_IoT_Botnet_Dataset_*.csv"
    
    try:
        df = load_multiple_files(
            file_pattern,
            nrows_per_file=args.n_rows,
            max_files=args.n_files
        )
        
        logger.info(f"\n✓ Loaded {len(df):,} rows from {args.n_files} files")
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Analyze timestamps
    logger.info("\n" + "="*60)
    timestamp_results = analyze_timestamps(df)
    
    logger.info("\nTimestamp Summary:")
    if 'stime' in timestamp_results:
        logger.info(f"  Time range: {timestamp_results['stime']['min']} to {timestamp_results['stime']['max']}")
        logger.info(f"  Duration: {timestamp_results['stime']['range_days']} days")
    
    # Analyze dataset
    logger.info("\n" + "="*60)
    dataset_results = analyze_dataset(df)
    
    logger.info("\nDataset Summary:")
    logger.info(f"  Rows: {dataset_results['shape']['rows']:,}")
    logger.info(f"  Columns: {dataset_results['shape']['columns']}")
    logger.info(f"  Memory: {dataset_results['memory_mb']:.2f} MB")
    
    if 'class_distribution' in dataset_results:
        logger.info("\nClass Distribution:")
        for cls, pct in dataset_results['class_distribution']['percentages'].items():
            logger.info(f"  Class {cls}: {pct:.2f}%")
    
    # Missing values
    missing = {k: v for k, v in dataset_results['missing_values'].items() if v > 0}
    if missing:
        logger.info(f"\nMissing Values: {len(missing)} columns have missing data")
        for col, count in list(missing.items())[:5]:
            logger.info(f"  {col}: {count:,} ({dataset_results['missing_pct'][col]:.2f}%)")
    
    # Combine results
    final_results = {
        'timestamp': str(datetime.now()),
        'data_source': args.data_dir,
        'sample_info': {
            'n_files': args.n_files,
            'n_rows_per_file': args.n_rows,
            'total_rows': len(df)
        },
        'timestamps': timestamp_results,
        'dataset': dataset_results
    }
    
    # Save results
    logger.info(f"\nSaving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Exploration complete!")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())
