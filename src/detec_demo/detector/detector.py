import argparse
import logging
import math
import os
import re 
import time

from prometheus_client import Counter as PromCounter, Gauge, start_http_server
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# -----------------------
# Setup logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# -----------------------
# Prometheus Metrics
# -----------------------

ATTACK_WINDOWS = PromCounter(
    "ddos_attack_windows_total",
    "Total detected attack windows",
    ["attack_type", "ddos_variant", "dst_ip"]
)

NORMAL_WINDOWS = PromCounter(
    "ddos_normal_windows_total",
    "Total detected normal windows",
    ["dst_ip"]
)

PACKET_RATE = Gauge(
    "ddos_packet_rate",
    "Packet rate per detection window",
    ["dst_ip"]
)

SRC_ENTROPY = Gauge(
    "ddos_src_entropy",
    "Source IP entropy per detection window",
    ["dst_ip"]
)


# -----------------------
# Constants / Defaults
# -----------------------
WINDOW_SIZE = 30
DEFAULT_DUR = float(WINDOW_SIZE)

# For SYN-heavy traffic, bytes are hard to infer from Argus meta; keep a conservative estimate
DEFAULT_BYTES_PER_PKT = 60  # ~ minimal TCP/IP headers order of magnitude


# -----------------------
# Utilities
# -----------------------
def shannon_entropy_from_counts(counts: Iterable[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return ent


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, (int, np.integer)):
            return int(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return int(float(s))
    except Exception:
        return default


def normalize_port(v: Any) -> int:
    """
    Argus output sometimes prints service name (http/https).
    Make ports numeric because your downstream code may need it for sanity checks.
    (Ports are not part of 22 features, but keeping it clean avoids confusion.)
    """
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s == "http":
        return 80
    if s == "https":
        return 443
    return safe_int(s, 0)


def parse_hms_micro_to_epoch_seconds(hms: str) -> float:
    """
    Argus '-c' often prints only time-of-day: HH:MM:SS.xxxxxx (no date).
    We'll anchor it to today's date. Since you only need relative windows,
    any consistent anchor is fine.
    """
    today = date.today().isoformat()
    # Accept both with/without microseconds
    if "." in hms:
        dt = datetime.strptime(f"{today} {hms}", "%Y-%m-%d %H:%M:%S.%f")
    else:
        dt = datetime.strptime(f"{today} {hms}", "%Y-%m-%d %H:%M:%S")
    return dt.timestamp()


# -----------------------
# Parsing flows_c (-c)
# -----------------------
TIME_RE = re.compile(r"(?P<t>\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
IP_RE = re.compile(r"(?P<ip>\d{1,3}(?:\.\d{1,3}){3})")
# Note: Argus compact format doesn't have word boundaries, so match tcp/udp/icmp anywhere
PROTO_RE = re.compile(r"(tcp|udp|icmp)", re.IGNORECASE)

# In your '-c' output the delimiter appears to be "s" (compact output)
# Example:
# 12:35:00.289552s          stcps172.19.0.5s35238ss172.19.0.3shttpssINT
# We'll parse robustly by extracting time + proto + ips + ports when present.
PORT_RE = re.compile(r"\b(\d{1,5}|http|https)\b", re.IGNORECASE)


@dataclass
class EventC:
    t_epoch: float
    saddr: str
    daddr: str
    proto: str


def load_events_from_flows_c(path: str, limit_lines: Optional[int] = None) -> List[EventC]:
    events: List[EventC] = []
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit_lines is not None and i >= limit_lines:
                break
            line = line.strip()
            if not line:
                continue
            # skip header / management
            if "StartTime" in line or "Proto" in line:
                continue
            if " man" in line or line.endswith("STA") or "sman" in line:
                continue

            mtime = TIME_RE.search(line)
            mproto = PROTO_RE.search(line)
            ips = IP_RE.findall(line)

            if not (mtime and mproto and len(ips) >= 2):
                continue

            t_epoch = parse_hms_micro_to_epoch_seconds(mtime.group("t"))
            proto = mproto.group(1).lower()
            saddr, daddr = ips[0], ips[1]

            events.append(EventC(t_epoch=t_epoch, saddr=saddr, daddr=daddr, proto=proto))
    return events


# -----------------------
# Parsing flows_s (-s)
# -----------------------
@dataclass
class MetaS:
    saddr: str
    daddr: str
    sport: int
    dport: int
    proto: str
    state: str
    flgs: str
    pkts: int
    bytes: int
    dur: float


def load_meta_from_flows_s(path: str, limit_lines: Optional[int] = None) -> List[MetaS]:
    """
    flows_s.txt is usually whitespace formatted.
    We'll parse by splitting, ignoring header-like lines.

    Expected typical columns:
      SrcAddr DstAddr Sport Dport Proto State Flgs TotPkts TotBytes Dur
    Some exports may omit Flgs or have blanks -> handle defensively.
    """
    metas: List[MetaS] = []

    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit_lines is not None and i >= limit_lines:
                break
            line = line.strip()
            if not line:
                continue
            # Skip header-ish lines
            if line.lower().startswith("srcaddr") or "DstAddr" in line or "TotPkts" in line:
                continue

            parts = line.split()
            # We need at minimum: saddr daddr sport dport proto state dur (len ~ 7+)
            if len(parts) < 7:
                continue

            saddr = parts[0]
            daddr = parts[1]
            sport = normalize_port(parts[2])
            dport = normalize_port(parts[3])
            proto = str(parts[4]).lower()
            state = str(parts[5])

            # Flgs may be missing/blank; Argus sometimes shifts columns.
            # Try to detect pkts/bytes/dur at the end.
            # We'll read last token as dur, previous as bytes, previous as pkts (if numeric).
            dur = safe_float(parts[-1], 0.0)
            bytes_ = safe_int(parts[-2], 0) if len(parts) >= 2 else 0
            pkts = safe_int(parts[-3], 0) if len(parts) >= 3 else 0

            # flgs is what's between state and pkts if present
            flgs = ""
            # candidate middle segment:
            mid = parts[6:-3]
            if mid:
                # join (rarely multiple)
                flgs = " ".join(mid)

            # Basic proto filter
            if proto not in {"tcp", "udp", "icmp"} and proto != "man":
                # keep unknown but mark
                pass

            metas.append(
                MetaS(
                    saddr=saddr,
                    daddr=daddr,
                    sport=sport,
                    dport=dport,
                    proto=proto,
                    state=state,
                    flgs=flgs,
                    pkts=pkts,
                    bytes=bytes_,
                    dur=dur,
                )
            )
    return metas


# -----------------------
# Feature building (22 features)
# -----------------------
def build_22_features(
    events: List[EventC],
    metas: List[MetaS],
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Training logic (your docs):
      - time_window = floor(stime / 30)
      - group by (time_window, daddr)
      - compute unique_src_count, src_entropy, top_src_ratio
    We'll do the same with RELATIVE time derived from event timestamps.
    """
    if not events:
        return pd.DataFrame()

    # Convert to DataFrame
    df_e = pd.DataFrame([e.__dict__ for e in events])
    # relative time to mimic stime scale; only windows matter
    t0 = df_e["t_epoch"].min()
    df_e["stime"] = df_e["t_epoch"] - t0  # seconds since start capture
    df_e["time_window"] = (df_e["stime"] // float(window_size)).astype(int)

    # metas: use them to enrich proto/state/flgs and (if non-zero) pkts/bytes/dur
    df_m = pd.DataFrame([m.__dict__ for m in metas]) if metas else pd.DataFrame()
    if not df_m.empty:
        # normalize proto to lowercase
        df_m["proto"] = df_m["proto"].astype(str).str.lower()
        df_m["time_window"] = -1  # no timestamps in -s; we will use daddr-level mode only

    rows: List[Dict[str, Any]] = []

    for (tw, daddr), g in df_e.groupby(["time_window", "daddr"], sort=True):
        g = g.sort_values("stime")
        stimes = g["stime"].to_numpy(dtype=float)

        # Inter-arrival times within the group (window + daddr)
        if len(stimes) >= 2:
            iat = np.diff(stimes)
        else:
            iat = np.array([0.0], dtype=float)

        mean_iat = float(np.mean(iat))
        std_iat = float(np.std(iat))
        sum_iat = float(np.sum(iat))
        min_iat = float(np.min(iat))
        max_iat = float(np.max(iat))

        # Packet/bytes: event-level gives you a "packet-like" count
        pkts = int(len(g))
        bytes_ = int(pkts * DEFAULT_BYTES_PER_PKT)

        # Duration: in training you kept 'dur' and used windowing on 'stime'.
        # For inference, safest is window size; optionally use span if longer.
        dur_span = float(stimes[-1] - stimes[0]) if len(stimes) >= 2 else 0.0
        dur = max(DEFAULT_DUR, dur_span) if dur_span > 0 else DEFAULT_DUR

        # Directional metrics: with sniffed SYN-heavy traffic, dst replies are typically absent
        spkts = pkts
        dpkts = 0
        sbytes = bytes_
        dbytes = 0

        rate = pkts / dur if dur > 0 else 0.0
        srate = spkts / dur if dur > 0 else 0.0
        drate = dpkts / dur if dur > 0 else 0.0

        # Diversity features
        src_counter = Counter(g["saddr"].tolist())
        unique_src_count = int(len(src_counter))
        src_entropy = float(shannon_entropy_from_counts(src_counter.values()))
        top_src_ratio = float(max(src_counter.values()) / pkts) if pkts > 0 else 1.0

        # Defaults per your doc: fill NaN with (1, 0.0, 1.0) for (unique, entropy, top_ratio)
        if unique_src_count <= 0:
            unique_src_count = 1
        if not np.isfinite(src_entropy):
            src_entropy = 0.0
        if not np.isfinite(top_src_ratio) or top_src_ratio <= 0:
            top_src_ratio = 1.0

        # proto: use mode from events first
        proto = str(g["proto"].mode().iloc[0]).lower() if not g["proto"].mode().empty else "tcp"

        # state/flgs/dport: enrich from metas if available for same daddr
        # NOTE: flows_s does NOT have timestamp → we can only match by daddr
        # This is acceptable because flows_s is ONLY used for categorical enrichment
        state = ""
        flgs = ""
        dport = ""

        if not df_m.empty:
            dm = df_m[df_m["daddr"] == daddr]
            if not dm.empty:
                # Enrich categorical features from flows_s
                state_mode = dm["state"].dropna().astype(str)
                flgs_mode = dm["flgs"].dropna().astype(str)
                proto_mode = dm["proto"].dropna().astype(str)
                dport_mode = dm["dport"].dropna().astype(str)

                if not proto_mode.empty:
                    proto = proto_mode.mode().iloc[0].lower()
                if not state_mode.empty:
                    state = state_mode.mode().iloc[0]
                if not flgs_mode.empty:
                    # flgs sometimes blank; choose the most common non-blank
                    non_blank = flgs_mode[flgs_mode.str.strip() != ""]
                    if not non_blank.empty:
                        flgs = non_blank.mode().iloc[0]
                if not dport_mode.empty:
                    dport = dport_mode.mode().iloc[0]

                # IMPORTANT: DO NOT override pkts/bytes/dur from flows_s
                # Reason:
                # 1. flows_c (event-level) is MORE ACCURATE than flows_s (flow summary)
                # 2. SYN flood → flows_s often has pkts=0 (incomplete flows)
                # 3. Training used event-level counts → inference must match
                # 4. Overriding can cause INCORRECT feature values
                #
                # We keep pkts/bytes/dur computed from flows_c (event counts)

        # seq: TCP sequence number not available from Argus summaries
        # Training also used seq=0 for aggregated flows, so this is consistent
        seq = 0

        row = {
            # 22 features
            "flgs": flgs,
            "proto": proto,
            "pkts": pkts,
            "bytes": bytes_,
            "state": state,
            "seq": seq,
            "dur": float(dur),
            "mean": mean_iat,
            "stddev": std_iat,
            "sum": sum_iat,
            "min": min_iat,
            "max": max_iat,
            "spkts": spkts,
            "dpkts": dpkts,
            "sbytes": sbytes,
            "dbytes": dbytes,
            "rate": float(rate),
            "srate": float(srate),
            "drate": float(drate),
            "unique_src_count": unique_src_count,
            "src_entropy": src_entropy,
            "top_src_ratio": top_src_ratio,
            # extras to help debug and fallback
            "_time_window": int(tw),
            "_daddr": str(daddr),
            "Dport": dport,  # Add Dport for fallback logic
        }
        rows.append(row)

    df_feat = pd.DataFrame(rows)
    
    # Fill missing values with defaults (matching training pipeline)
    # Training used median fill, but for real-time inference we use safe defaults
    numeric_cols = [
        "pkts", "bytes", "dur", "mean", "stddev", "sum", "min", "max",
        "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate",
        "unique_src_count", "src_entropy", "top_src_ratio", "seq"
    ]
    
    for col in numeric_cols:
        if col in df_feat.columns:
            # Replace NaN and inf with 0.0 (conservative default)
            df_feat[col] = df_feat[col].replace([np.inf, -np.inf], np.nan)
            df_feat[col] = df_feat[col].fillna(0.0)
    
    # Categorical columns: fill empty strings with ""
    categorical_cols = ["flgs", "proto", "state"]
    for col in categorical_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna("")
    
    return df_feat


# -----------------------
# Encoding + Align features
# -----------------------
def apply_label_encoders_unknown_minus_one(df: pd.DataFrame, encoders: Dict[str, Any]) -> pd.DataFrame:
    """
    Per your training: LabelEncoder fit on train; inference unknown => -1. :contentReference[oaicite:4]{index=4}
    encoders.pkl is expected to be a dict: {col_name: LabelEncoder}
    """
    out = df.copy()
    for col, enc in encoders.items():
        if col not in out.columns:
            continue
        values = out[col].astype(str).fillna("")
        # Build class->index mapping from encoder.classes_
        try:
            classes = list(enc.classes_)
            map_idx = {str(c): int(i) for i, c in enumerate(classes)}
        except Exception:
            # If encoder is not sklearn LabelEncoder-like, try transform directly with safe fallback
            map_idx = {}

        encoded = []
        for v in values.tolist():
            key = str(v)
            if key in map_idx:
                encoded.append(map_idx[key])
            else:
                encoded.append(-1)
        out[col] = np.array(encoded, dtype=int)
    return out


def align_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_list:
        if c not in out.columns:
            # create missing columns with 0
            out[c] = 0
    # Keep only features and in correct order
    return out[feature_list]


def map_predictions(pred: np.ndarray, mapping: Any) -> List[Any]:
    """
    mapping.pkl described as "Attack type mapping dictionary".
    It can be {0:'DDoS',1:'DoS',2:'Recon'} or {'DDoS':0,'DoS':1,'Recon':2}
    We need to handle both formats and return the name strings.
    """
    if mapping is None:
        return pred.tolist()

    # If mapping is dict
    if isinstance(mapping, dict):
        # Check if it's {name: index} format or {index: name} format
        first_key = next(iter(mapping.keys()))
        first_val = next(iter(mapping.values()))
        
        if isinstance(first_key, str) and isinstance(first_val, (int, np.integer)):
            # Format: {'DDoS': 0, 'DoS': 1, ...} - invert to {0: 'DDoS', 1: 'DoS', ...}
            mapping = {v: k for k, v in mapping.items()}
        
        # Now use index -> name format
        return [mapping.get(int(x), str(x)) for x in pred]

    # If mapping is list
    if isinstance(mapping, list):
        return [mapping[int(x)] if 0 <= int(x) < len(mapping) else x for x in pred]

    return pred.tolist()


def extract_stage3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 16 features required for Stage 3 (DDoS variant classification).
    
    Stage 3 uses a subset of the 22 features:
    ['pkts', 'bytes', 'seq', 'dur', 'mean', 'stddev', 'sum', 'min', 'max',
     'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate']
    
    Args:
        df: DataFrame with all 22 features
        
    Returns:
        DataFrame with 16 features (preserves column names for XGBoost)
    """
    stage3_features = ['pkts', 'bytes', 'seq', 'dur', 'mean', 'stddev', 
                       'sum', 'min', 'max', 'spkts', 'dpkts', 'sbytes', 
                       'dbytes', 'rate', 'srate', 'drate']
    
    # Ensure all features exist
    missing = [f for f in stage3_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing Stage 3 features: {missing}")
    
    return df[stage3_features]


def check_if_http_port(dport_value) -> bool:
    """
    Check if Dport indicates HTTP traffic.
    
    Args:
        dport_value: Destination port (can be string name or numeric)
        
    Returns:
        True if port indicates HTTP/HTTPS traffic, False otherwise
    """
    if pd.isna(dport_value):
        return False
    
    # Check symbolic names
    if isinstance(dport_value, str):
        dport_lower = dport_value.lower().strip()
        if dport_lower in ['http', 'https', 'http-alt', 'www', 'www-http']:
            return True
    
    # Check numeric ports
    try:
        port_num = int(float(dport_value))
        if port_num in [80, 443, 8080, 8443, 8000, 8888]:
            return True
    except (ValueError, TypeError):
        pass
    
    return False


# -----------------------
# Main
# -----------------------
def main():
    #promethues metric server
    start_http_server(8000)
    logger.info("[PROM] Metrics exposed at http://localhost:8000/metrics")

    # Auto-detect script directory to handle running from different locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    ap = argparse.ArgumentParser(description="DDoS Detection with Two-Stage ML Pipeline")
    ap.add_argument("--flows-c", default=os.path.join(script_dir, "flows/flows_c.txt"), 
                    help="Path to flows_c.txt (Argus -c output)")
    ap.add_argument("--flows-s", default=os.path.join(script_dir, "flows/flows_s.txt"), 
                    help="Path to flows_s.txt (Argus -s output)")
    ap.add_argument("--encoders", default=os.path.join(script_dir, "results/models/encoders.pkl"), 
                    help="Path to encoders.pkl")
    ap.add_argument("--features", default=os.path.join(script_dir, "results/models/features.pkl"), 
                    help="Path to features.pkl")
    ap.add_argument("--mapping", default=os.path.join(script_dir, "results/models/mapping.pkl"), 
                    help="Path to mapping.pkl")
    ap.add_argument("--stage1", default=os.path.join(script_dir, "results/models/stage1.pkl"), 
                    help="Path to stage1.pkl")
    ap.add_argument("--stage2", default=os.path.join(script_dir, "results/models/stage2.pkl"), 
                    help="Path to stage2.pkl")
    ap.add_argument("--stage3", default=os.path.join(script_dir, "results/models/stage3.json"), 
                    help="Path to stage3.json")
    ap.add_argument("--label-encoder", default=os.path.join(script_dir, "results/models/label_encoder.pkl"), 
                    help="Path to label_encoder.pkl for Stage 3")
    ap.add_argument("--window", type=int, default=WINDOW_SIZE, help=f"Time window size in seconds (default: {WINDOW_SIZE})")
    ap.add_argument("--limit-c", type=int, default=0, help="Limit lines for flows_c (0 = no limit)")
    ap.add_argument("--limit-s", type=int, default=0, help="Limit lines for flows_s (0 = no limit)")
    ap.add_argument("--show", type=int, default=20, help="Show top N results")
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = ap.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info("DDoS DETECTOR - Two-Stage ML Pipeline")
    logger.info("="*60)

    # Load models/assets
    logger.info(f"Loading Stage 1 model from: {args.stage1}")
    stage1 = joblib.load(args.stage1)
    
    logger.info(f"Loading Stage 2 model from: {args.stage2}")
    stage2 = joblib.load(args.stage2)

    logger.info(f"Loading Stage 3 model from: {args.stage3}")
    stage3 = xgb.Booster()
    stage3.load_model(args.stage3)
    
    logger.info(f"Loading label encoder for Stage 3 from: {args.label_encoder}")
    label_encoder = joblib.load(args.label_encoder)

    logger.info(f"Loading encoders from: {args.encoders}")
    encoders = joblib.load(args.encoders)
    
    logger.info(f"Loading features list from: {args.features}")
    feature_list = joblib.load(args.features)
    
    logger.info(f"Loading attack mapping from: {args.mapping}")
    mapping = joblib.load(args.mapping)

    # Sanity checks
    if not isinstance(encoders, dict):
        raise ValueError("encoders.pkl must be a dict: {col: LabelEncoder}")
    
    logger.debug(f"Encoders loaded for columns: {list(encoders.keys())}")
    logger.debug(f"Feature list ({len(feature_list)} features): {feature_list}")
    logger.debug(f"Attack mapping: {mapping}")

    # Load data
    limit_c = None if args.limit_c == 0 else args.limit_c
    limit_s = None if args.limit_s == 0 else args.limit_s

    logger.info(f"Loading flows_c from: {args.flows_c}" + (f" (limit: {limit_c})" if limit_c else ""))
    events = load_events_from_flows_c(args.flows_c, limit_lines=limit_c)
    
    logger.info(f"Loading flows_s from: {args.flows_s}" + (f" (limit: {limit_s})" if limit_s else ""))
    metas = load_meta_from_flows_s(args.flows_s, limit_lines=limit_s)

    if not events:
        logger.error("No events parsed from flows_c. Check flows_c format / path.")
        return
    
    logger.info(f"Parsed {len(events):,} events from flows_c")
    logger.info(f"Parsed {len(metas):,} flows from flows_s")

    # Build 22 features
    logger.info(f"Building 22 features with window_size={args.window}s...")
    df_feat = build_22_features(events, metas, window_size=args.window)

    if df_feat.empty:
        logger.error("Feature table empty. Nothing to predict.")
        return
    
    logger.info(f"Generated {len(df_feat):,} time windows for analysis")

    # Keep debug cols for later display
    debug_cols = ["_time_window", "_daddr"] if "_time_window" in df_feat.columns else []
    df_X = df_feat.copy()

    # Encode categorical
    logger.info("Encoding categorical features (unknown → -1)...")
    df_X = apply_label_encoders_unknown_minus_one(df_X, encoders)

    # Align 22 features
    logger.info(f"Aligning features to match training order ({len(feature_list)} features)...")
    X = align_features(df_X, feature_list)
    
    logger.debug(f"Final feature matrix shape: {X.shape}")

    # Stage 1: binary Attack vs Normal
    logger.info("Running Stage 1: Binary Classification (Attack vs Normal)...")
    y1 = stage1.predict(X)
    y1 = np.asarray(y1)

    # Normalize to a boolean mask
    attack_mask = None
    if y1.dtype.kind in {"i", "u", "b"}:
        attack_mask = y1.astype(int) == 1
    else:
        # string labels
        attack_mask = np.array([str(v).lower() not in {"0", "normal", "benign"} for v in y1], dtype=bool)

    # Stage 2: classify attack types
    y2 = np.array(["-"] * len(X), dtype=object)
    if attack_mask.any():
        logger.info(f"Running Stage 2: Multi-class Classification ({attack_mask.sum():,} attacks)...")
        y2_pred = stage2.predict(X[attack_mask])
        y2_pred = np.asarray(y2_pred)
        logger.debug(f"[DEBUG] y2_pred raw (first 5): {y2_pred[:5]}, dtype: {y2_pred.dtype}")
        y2_labels = map_predictions(y2_pred, mapping)
        logger.debug(f"[DEBUG] y2_labels after map (first 5): {y2_labels[:5]}")
        y2[attack_mask] = np.array(y2_labels, dtype=object)
        logger.debug(f"[DEBUG] y2 after assignment (first 5): {y2[:5]}")
    else:
        logger.info("Stage 2 skipped: No attacks detected by Stage 1")

    # Stage 3: DDoS variant classification (only for DDoS attacks)
    y3 = np.array(["unknown"] * len(X), dtype=object)
    ddos_mask = None
    
    if attack_mask.any():
        # Identify DDoS attacks from Stage 2 output
        ddos_mask = np.array([str(v).lower() == "ddos" for v in y2], dtype=bool)
        
        if ddos_mask.any():
            logger.info(f"Running Stage 3: DDoS Variant Classification ({ddos_mask.sum():,} DDoS attacks)...")
            
            # Extract 16 features for Stage 3
            X_stage3 = extract_stage3_features(df_feat)
            X_stage3_ddos = X_stage3[ddos_mask]
            
            # Run Stage 3 prediction (preserve feature names to satisfy XGBoost)
            dmat_stage3 = xgb.DMatrix(
                X_stage3_ddos,
                feature_names=list(X_stage3_ddos.columns)
            )
            y3_pred = stage3.predict(dmat_stage3)
            y3_pred = np.asarray(y3_pred, dtype=int)
            logger.debug(f"[DEBUG] y3_pred raw (first 5): {y3_pred[:5]}")
            
            # Convert numeric predictions to variant names using label_encoder
            y3_variants = label_encoder.inverse_transform(y3_pred)
            logger.debug(f"[DEBUG] y3_variants after inverse_transform (first 5): {list(y3_variants[:5])}")
            
            # Map variant names: keep 'Normal' as 'normal', 'HTTP' → 'http', etc.
            # All 4 classes from training: Normal, HTTP, TCP, UDP
            y3_labels = []
            for variant in y3_variants:
                variant_str = str(variant).lower()
                y3_labels.append(variant_str)  # normal, http, tcp, udp
            
            # Apply rule-based fallback for Normal predictions that should be HTTP
            # Check if Dport indicates HTTP traffic when model predicts Normal
            fallback_count = 0
            ddos_indices = np.where(ddos_mask)[0]
            
            for i, label in enumerate(y3_labels):
                if label == 'normal':
                    # Get original window index
                    window_idx = ddos_indices[i]
                    
                    # Get Dport from the window data
                    if 'Dport' in df_feat.columns:
                        dport = df_feat.iloc[window_idx]['Dport']
                        
                        # If Dport indicates HTTP, override prediction
                        if check_if_http_port(dport):
                            y3_labels[i] = 'http'
                            fallback_count += 1
                            logger.debug(f"[FALLBACK] Window {window_idx}: 'normal' → 'http' (Dport={dport})")
            
            if fallback_count > 0:
                logger.info(f"Rule-based fallback applied: {fallback_count} windows corrected from 'normal' to 'http' based on Dport")
            
            y3[ddos_mask] = np.array(y3_labels, dtype=object)
            logger.debug(f"[DEBUG] y3 after assignment (first 5): {list(y3[:5])}")
        else:
            logger.info("Stage 3 skipped: No DDoS attacks detected by Stage 2")
    else:
        logger.info("Stage 3 skipped: No attacks detected by Stage 1")

    # Build output table
    out = df_feat.copy()
    out["stage1_pred"] = y1
    out["stage2_pred"] = y2
    out["stage3_pred"] = y3

    #metric
    for idx, row in out.iterrows():
        dst_ip = row["_daddr"]

        # ===== Feature-level metrics =====
        if np.isfinite(row["rate"]):
            PACKET_RATE.labels(dst_ip=dst_ip).set(row["rate"])

        if np.isfinite(row["src_entropy"]):
            SRC_ENTROPY.labels(dst_ip=dst_ip).set(row["src_entropy"])

        # ===== Prediction-level metrics =====
        # Stage 1: Normal vs Attack
        if row["stage1_pred"] == 1:
            # Get attack type from Stage 2 and variant from Stage 3
            attack_type_raw = row["stage2_pred"]
            attack_type = str(attack_type_raw).strip().lower()  # ddos / dos / reconnaissance
            
            # Get DDoS variant from Stage 3 (only for DDoS attacks)
            ddos_variant = "unknown"  # default
            if attack_type == "ddos":
                variant_raw = row["stage3_pred"]
                ddos_variant = str(variant_raw).strip().lower()  # http / tcp / udp / unknown

            ATTACK_WINDOWS.labels(
                attack_type=attack_type,           # ddos / dos / reconnaissance
                ddos_variant=ddos_variant,         # http / tcp / udp / unknown
                dst_ip=dst_ip
            ).inc()
        else:
            NORMAL_WINDOWS.labels(dst_ip=dst_ip).inc()


    # Print summary
    logger.info("="*60)
    logger.info("DETECTION RESULTS")
    logger.info("="*60)
    print(f"\n[+] Windows analyzed: {len(out):,} (window={args.window}s)")
    print(f"[+] Stage 1 distribution: {dict(Counter(out['stage1_pred'].tolist()))}")
    if attack_mask.any():
        print(f"[+] Stage 2 distribution: {dict(Counter(out.loc[attack_mask, 'stage2_pred'].tolist()))}")
        # Show Stage 3 distribution if there are DDoS attacks
        if ddos_mask is not None and ddos_mask.any():
            print(f"[+] Stage 3 distribution (DDoS variants): {dict(Counter(out.loc[ddos_mask, 'stage3_pred'].tolist()))}")
    else:
        print("[+] Stage 2: No attacks detected")

    # Show top N
    show_n = max(1, int(args.show))
    cols_to_show = []
    if debug_cols:
        cols_to_show += debug_cols
    cols_to_show += ["pkts", "rate", "unique_src_count", "src_entropy", "top_src_ratio", "proto", "state", "flgs", "stage1_pred", "stage2_pred", "stage3_pred"]
    cols_to_show = [c for c in cols_to_show if c in out.columns]

    print("\n" + "="*60)
    print(f"SAMPLE RESULTS (Top {show_n})")
    print("="*60)
    print(out[cols_to_show].head(show_n).to_string(index=False))
    print("="*60)
    
    logger.info("Detection completed successfully!")

    logger.info("[PROM] Waiting for Prometheus scrape...")
    while True:
        time.sleep(5)


if __name__ == "__main__":
    main()
