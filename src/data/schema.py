"""
Dataset schema definition for Bot-IoT dataset
Single source of truth for columns, dtypes, and targets
"""

# All feature columns from the entire dataset
ALL_COLUMNS = [
    'pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport',
    'pkts', 'bytes', 'state', 'ltime', 'seq', 'dur', 'mean', 'stddev',
    'smac', 'dmac', 'sum', 'min', 'max', 'soui', 'doui', 'sco', 'dco',
    'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate',
    'attack', 'category', 'subcategory'
]

# Temporal columns
TEMPORAL_COLS = ['stime', 'ltime', 'dur']

# Categorical columns
CATEGORICAL_COLS = ['proto', 'state', 'flgs', 'category', 'subcategory']

# Numerical columns (excluding temporal and target)
NUMERICAL_COLS = [
    'sport', 'dport', 'pkts', 'bytes', 'seq', 'mean', 'stddev',
    'sum', 'min', 'max', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'srate', 'drate'
]

# IP address columns (special handling)
IP_COLS = ['saddr', 'daddr']

# MAC address columns
MAC_COLS = ['smac', 'dmac']

# Target columns
TARGET_COL = 'attack'  # Binary: 0=Normal, 1=Attack
CATEGORY_COL = 'category'  # Multi-class attack type
SUBCATEGORY_COL = 'subcategory'  # Detailed attack type

# Feature columns for 10-best dataset (from previous work)
BEST_10_COLS = [
    'proto', 'saddr', 'sport', 'daddr', 'dport',
    'seq', 'stddev', 'N_IN_Conn_P_SrcIP', 'min', 'state_number',
    'mean', 'N_IN_Conn_P_DstIP', 'drate', 'srate', 'max',
    'attack', 'category', 'subcategory'
]

# Data types (flexible - will be inferred and optimized later)
DTYPES = {
    'pkSeqID': 'int64',
    'stime': 'float64',  # Unix timestamp
    'flgs': 'object',
    'proto': 'object',
    'saddr': 'object',
    'sport': 'int64',
    'daddr': 'object',
    'dport': 'int64',
    'pkts': 'int64',
    'bytes': 'int64',
    'state': 'object',
    'ltime': 'float64',  # Unix timestamp
    'seq': 'int64',
    'dur': 'float64',
    'mean': 'float64',
    'stddev': 'float64',
    'smac': 'object',
    'dmac': 'object',
    'sum': 'float64',
    'min': 'float64',
    'max': 'float64',
    'soui': 'object',
    'doui': 'object',
    'sco': 'object',
    'dco': 'object',
    'spkts': 'int64',
    'dpkts': 'int64',
    'sbytes': 'int64',
    'dbytes': 'int64',
    'rate': 'float64',
    'srate': 'float64',
    'drate': 'float64',
    'attack': 'int64',
    'category': 'object',
    'subcategory': 'object'
}
