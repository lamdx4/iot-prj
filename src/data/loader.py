"""
Data loading utilities for Bot-IoT dataset
Handles CSV loading with chunking, column selection, and memory optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Iterator
import logging

from src.data.schema import ALL_COLUMNS, DTYPES

logger = logging.getLogger(__name__)


def load_data(
    filepath: Union[str, Path],
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    parse_times: bool = False,
    chunksize: Optional[int] = None
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Load Bot-IoT dataset from CSV
    
    Args:
        filepath: Path to CSV file
        nrows: Number of rows to load (None = all)
        usecols: Columns to load (None = all)
        parse_times: Parse timestamp columns to datetime
        chunksize: If set, return iterator of chunks
        
    Returns:
        DataFrame or iterator of DataFrames
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Read CSV (no header in files, must provide names)
    logger.info(f"Loading {filepath.name}...")
    
    df_or_iterator = pd.read_csv(
        filepath,
        names=ALL_COLUMNS,  # CSV has no header, provide column names
        header=None,        # No header row
        nrows=nrows,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False
    )
    
    # If chunked, return iterator
    if chunksize:
        return df_or_iterator
    
    df = df_or_iterator
    
    # Parse timestamps if requested
    if parse_times and 'stime' in df.columns:
        logger.info("Parsing timestamps...")
        df['stime'] = pd.to_datetime(df['stime'], unit='s', errors='coerce')
        if 'ltime' in df.columns:
            df['ltime'] = pd.to_datetime(df['ltime'], unit='s', errors='coerce')
    
    logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def load_multiple_files(
    file_pattern: str,
    nrows_per_file: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    max_files: Optional[int] = None
) -> pd.DataFrame:
    """
    Load multiple CSV files matching a pattern
    
    Args:
        file_pattern: Glob pattern (e.g., "Data/*.csv")
        nrows_per_file: Limit rows per file
        usecols: Columns to load
        max_files: Maximum number of files to load
        
    Returns:
        Concatenated DataFrame
    """
    from glob import glob
    
    files = sorted(glob(file_pattern))
    
    if max_files:
        files = files[:max_files]
    
    logger.info(f"Found {len(files)} files matching {file_pattern}")
    
    if not files:
        raise ValueError(f"No files found matching: {file_pattern}")
    
    dfs = []
    for filepath in files:
        try:
            df = load_data(filepath, nrows=nrows_per_file, usecols=usecols)
            dfs.append(df)
            logger.info(f"  ✓ Loaded {filepath}: {len(df):,} rows")
        except Exception as e:
            logger.error(f"  ✗ Failed to load {filepath}: {e}")
    
    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"✓ Concatenated {len(result):,} total rows from {len(dfs)} files")
    
    return result


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting dtypes
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        # Optimize integers
        if col_type == 'int64':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        # Optimize floats
        elif col_type == 'float64':
            df[col] = df[col].astype(np.float32)
        
        # Optimize objects to category
        elif col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Low cardinality
                df[col] = df[col].astype('category')
    
    return df
