"""
Data loading and preparation module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Load data from CSV/TXT file"""
    print(f"Loading data from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
    
    if ',' in first_line:
        print("Detected CSV format, processing...")
        col_names = first_line.split(',')
        num_cols = len(col_names)
        data_values = second_line.split(',')
        num_values = len(data_values)
        
        if num_cols == 1 and num_values > 1:
            print(f"Header merged ({num_cols} cols), data separated ({num_values} values)")
            col_names = first_line.split(',')[0].split(',') if ',' in first_line else first_line.split()
        
        df = pd.read_csv(filepath, names=col_names, skiprows=1)
    else:
        try:
            df = pd.read_csv(filepath, sep=r'\s+')
        except:
            try:
                df = pd.read_csv(filepath, sep=',')
            except:
                df = pd.read_csv(filepath, sep='\t')
    
    print(f"Loaded {len(df)} rows")
    print(f"Number of columns: {len(df.columns)}")
    
    return df

def prepare_data(df):
    """Prepare data for training"""
    print("\nPreparing data...")
    
    # Feature Set
    feature_cols = [
        "pkts", "bytes", "seq", "dur", "mean", "stddev", "sum", "min", "max",
        "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate"
    ]
    
    # Check if features exist
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        feature_cols = [f for f in feature_cols if f in df.columns]
    
    # Get features
    X = df[feature_cols].copy()
    
    # Convert to numeric
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values and infinite values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Encode labels
    label_col = 'subcategory' if 'subcategory' in df.columns else 'category'
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    
    # Reorder classes: Normal, HTTP, TCP, UDP
    class_order = []
    if 'Normal' in le.classes_:
        class_order.append('Normal')
    for cls in ['DDoS-HTTP', 'DoS-HTTP', 'HTTP']:
        if cls in le.classes_:
            class_order.append(cls)
            break
    for cls in ['DDoS-TCP', 'DoS-TCP', 'TCP']:
        if cls in le.classes_:
            class_order.append(cls)
            break
    for cls in ['DDoS-UDP', 'DoS-UDP', 'UDP']:
        if cls in le.classes_:
            class_order.append(cls)
            break
    
    # Add any remaining classes
    for cls in le.classes_:
        if cls not in class_order:
            class_order.append(cls)
    
    # Create mapping for reordering
    class_mapping = {old_idx: class_order.index(cls) for old_idx, cls in enumerate(le.classes_)}
    y_reordered = np.array([class_mapping[label] for label in y])
    
    # Update label encoder
    le_new = LabelEncoder()
    le_new.classes_ = np.array(class_order)
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Classes: {list(le_new.classes_)}")
    print(f"Samples: {len(X)}")
    
    print(f"\nClass distribution:")
    for i, cls in enumerate(le_new.classes_):
        count = np.sum(y_reordered == i)
        print(f"   {cls}: {count} samples ({count/len(y_reordered)*100:.1f}%)")
    
    return X, y_reordered, le_new, feature_cols