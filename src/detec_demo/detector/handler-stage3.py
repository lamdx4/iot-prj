import pandas as pd
import glob

# ====== CONFIG ======
FEATURE_NAMES_FILE = "dataset/UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv"
FULL_PATTERN = "dataset/UNSW_2018_IoT_Botnet_Dataset_*.csv"
TRAIN_BALANCED_FILE = "balanced_ddos_only.csv"

OUT_FILE = "demo_ddos3_scenario.csv"
CHUNKSIZE = 200_000

# Block sizes
NEEDED = {"HTTP": 150, "TCP": 150, "UDP": 150}

COLUMNS = [
    "pkSeqID","stime","ltime","seq",
    "pkts","bytes","dur","mean","stddev","sum","min","max",
    "spkts","dpkts","sbytes","dbytes",
    "rate","srate","drate",
    "proto","category","subcategory","attack"
]

# ====== LOAD FEATURE NAMES ======
feat = pd.read_csv(FEATURE_NAMES_FILE, header=None)
colnames = [c.strip() for c in feat.iloc[0].tolist()]

files = sorted(glob.glob(FULL_PATTERN))
files = [f for f in files if "Feature_Names" not in f]
print("Files:", len(files))

# ====== LOAD TRAIN IDS (avoid leakage) ======
train_ids = set(pd.read_csv(TRAIN_BALANCED_FILE, usecols=["pkSeqID"])["pkSeqID"].astype("int64").tolist())
print("Train balanced IDs loaded:", len(train_ids))

def exclude_train(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["pkSeqID"].astype("int64").isin(train_ids)]

def take_ddos_block(subcat: str, n: int) -> pd.DataFrame:
    """Take n consecutive rows from FULL for DDoS subcategory, excluding train pkSeqID."""
    got_parts = []
    got = 0

    for f in files:
        print(f"Scanning for {subcat} in:", f)
        for chunk in pd.read_csv(
            f, header=None, names=colnames, usecols=COLUMNS, chunksize=CHUNKSIZE, low_memory=False
        ):
            part = chunk[(chunk["category"] == "DDoS") & (chunk["subcategory"] == subcat)]
            if len(part) == 0:
                continue

            part = exclude_train(part)
            if len(part) == 0:
                continue

            need = n - got
            take = part.iloc[:need].copy()   # consecutive within this chunk
            got_parts.append(take)
            got += len(take)

            if got >= n:
                break
        if got >= n:
            break

    if got < n:
        raise RuntimeError(f"Không đủ mẫu DDoS-{subcat}: got={got}, need={n} (sau khi loại trùng train).")

    blk = pd.concat(got_parts, ignore_index=True)
    blk = blk.iloc[:n].reset_index(drop=True)
    return blk

# ====== BUILD SCENARIO (3-class only) ======
print("\n=== BUILD DEMO 3-CLASS SCENARIO ===")
blocks = [
    take_ddos_block("HTTP", NEEDED["HTTP"]),
    take_ddos_block("TCP",  NEEDED["TCP"]),
    take_ddos_block("UDP",  NEEDED["UDP"]),
]

demo = pd.concat(blocks, ignore_index=True)

# ensure labels are clean for 3-class
demo["subcategory"] = demo["subcategory"].astype(str).str.upper()
demo.loc[demo["subcategory"].isin(["DDOS-HTTP","DOS-HTTP"]), "subcategory"] = "HTTP"
demo.loc[demo["subcategory"].isin(["DDOS-TCP","DOS-TCP"]), "subcategory"] = "TCP"
demo.loc[demo["subcategory"].isin(["DDOS-UDP","DOS-UDP"]), "subcategory"] = "UDP"

demo.to_csv(OUT_FILE, index=False)
print("Saved:", OUT_FILE)
print("Final size:", len(demo))
print(demo["subcategory"].value_counts())
print("\nDONE ✓")
import os
import time
import argparse
import pickle
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

FEATURES_16 = [
    "pkts", "bytes", "seq", "dur", "mean", "stddev", "sum", "min", "max",
    "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate"
]

def normalize_subcat(series: pd.Series) -> pd.Series:
    y = series.astype(str).str.strip().str.upper()
    return y.replace({
        "DDOS-HTTP": "HTTP",
        "DOS-HTTP": "HTTP",
        "DDOS-TCP": "TCP",
        "DOS-TCP": "TCP",
        "DDOS-UDP": "UDP",
        "DOS-UDP": "UDP",
        "HTTP": "HTTP",
        "TCP": "TCP",
        "UDP": "UDP",
    })

def load_label_encoder(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model(path_json: str) -> XGBClassifier:
    m = XGBClassifier()
    m.load_model(path_json)
    return m

def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES_16 if c not in df.columns]
    if missing:
        raise ValueError(f"Demo CSV thiếu features: {missing}")

    X = df[FEATURES_16].copy()
    for c in FEATURES_16:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    return X

def encode_y(df: pd.DataFrame, label_encoder):
    if "subcategory" not in df.columns:
        raise ValueError("Demo CSV thiếu cột 'subcategory'.")

    y_str = normalize_subcat(df["subcategory"])
    # check unseen labels
    unseen = set(y_str.unique()) - set(label_encoder.classes_)
    if unseen:
        raise ValueError(f"Demo label có giá trị không có trong encoder: {unseen}. "
                         f"Encoder classes={list(label_encoder.classes_)}")

    y = label_encoder.transform(y_str.astype(str))
    return y, y_str

def evaluate(model, X, y, label_encoder):
    pred = model.predict(X)

    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, average="weighted", zero_division=0)
    rec = recall_score(y, pred, average="weighted", zero_division=0)
    f1 = f1_score(y, pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y, pred)
    report = classification_report(y, pred, target_names=label_encoder.classes_)

    return acc, prec, rec, f1, cm, report, pred

def measure_throughput(model, X, runs=15):
    # warmup
    _ = model.predict(X.iloc[:min(500, len(X))])

    times = []
    for _ in range(runs):
        t0 = time.time()
        _ = model.predict(X)
        t1 = time.time()
        times.append(t1 - t0)

    avg_t = float(np.mean(times))
    rows_s = (len(X) / avg_t) if avg_t > 0 else 0.0
    return rows_s, avg_t

def realtime_print_demo(model, X, y, label_encoder, show=60, sleep=0.10):
    n = min(show, len(X))
    correct = 0

    print("\n" + "="*70)
    print("REALTIME DEMO (row-by-row)")
    print("="*70)

    for i in range(n):
xi = X.iloc[i:i+1]
        pred_i = int(model.predict(xi)[0])

        pred_name = label_encoder.inverse_transform([pred_i])[0]
        true_name = label_encoder.inverse_transform([int(y[i])])[0]

        if pred_i == int(y[i]):
            correct += 1

        live_acc = correct / (i + 1)
        print(f"[{i+1:04d}] true={true_name:<4}  pred={pred_name:<4}  live_acc={live_acc*100:6.2f}%")

        if sleep > 0:
            time.sleep(sleep)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description="Run 3-class DDoS variant demo + evaluation")
    ap.add_argument("--data", default=os.path.join(script_dir, "demo_ddos3_scenario.csv"),
                    help="Path to demo dataset CSV (from FULL dataset)")
    ap.add_argument("--model", default=os.path.join(script_dir, "model_ddos_only", "xgboost_model.json"),
                    help="Path to trained 3-class model json")
    ap.add_argument("--encoder", default=os.path.join(script_dir, "model_ddos_only", "label_encoder.pkl"),
                    help="Path to label encoder pkl")
    ap.add_argument("--show", type=int, default=60, help="How many rows to print in realtime demo")
    ap.add_argument("--sleep", type=float, default=0.10, help="Sleep between rows (seconds)")
    ap.add_argument("--no-realtime", action="store_true", help="Skip realtime printing")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Không thấy demo dataset: {args.data}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Không thấy model: {args.model}")
    if not os.path.exists(args.encoder):
        raise FileNotFoundError(f"Không thấy encoder: {args.encoder}")

    print("="*70)
    print("DDoS 3-CLASS DEMO + EVALUATION (HTTP/TCP/UDP)")
    print("="*70)
    print("Data   :", args.data)
    print("Model  :", args.model)
    print("Encoder:", args.encoder)

    df = pd.read_csv(args.data, low_memory=False)
    label_encoder = load_label_encoder(args.encoder)
    model = load_model(args.model)

    X = prepare_X(df)
    y, y_str = encode_y(df, label_encoder)

    # ---- Metrics ----
    acc, prec, rec, f1, cm, report, pred = evaluate(model, X, y, label_encoder)

    print("\n" + "="*70)
    print("OFFLINE EVALUATION (on demo data)")
    print("="*70)
    print(f"Accuracy : {acc:.6f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.6f}")
    print(f"Recall   : {rec:.6f}")
    print(f"F1-score : {f1:.6f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("Classes:", list(label_encoder.classes_))
    print(cm)

    print("\nClassification Report:")
    print(report)

    # ---- Throughput ----
    rows_s, avg_t = measure_throughput(model, X, runs=15)
    print("\n" + "="*70)
    print("THROUGHPUT")
    print("="*70)
    print(f"Batch size       : {len(X)} rows")
    print(f"Avg predict time : {avg_t:.4f}s")
    print(f"Throughput       : {rows_s:.0f} rows/s")
# ---- Realtime demo ----
    if not args.no_realtime:
        realtime_print_demo(model, X, y, label_encoder, show=args.show, sleep=args.sleep)

    print("\nDONE ✓")

if __name__ == "__main__":
    main()