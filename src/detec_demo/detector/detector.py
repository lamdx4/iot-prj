import argparse
import logging
import math
import os
import re 
import time
import queue
import socket
from multiprocessing import Queue

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

# Scapy import with fallback (for PCAP parsing)
try:
    from scapy.all import PcapReader
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning("Scapy not available. PCAP parsing will be disabled. Install: pip install scapy")
    # Dummy imports for type checking (never used at runtime if SCAPY_AVAILABLE is False)
    PcapReader = None  # type: ignore[assignment]
    IP = None  # type: ignore[assignment]
    TCP = None  # type: ignore[assignment]
    UDP = None  # type: ignore[assignment]
    ICMP = None  # type: ignore[assignment]


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


@dataclass
class EventC:
    t_epoch: float
    saddr: str
    daddr: str
    proto: str
    bytes: int = 0  # Packet size in bytes
    sport: int = 0  # Source port
    dport: int = 0  # Destination port (critical for HTTP detection)
    tcp_flags: int = 0  # TCP flags bitmask (0x02=SYN, 0x10=ACK, 0x01=FIN, 0x04=RST, etc.)


# -----------------------
# Scapy Packet Conversion (For Queue Mode)
# -----------------------
def scapy_packets_to_events(packets: List, global_t0: Optional[float] = None) -> Tuple[List[EventC], float]:
    """
    Convert Scapy packet objects (from victim queue) to EventC format
    This allows processing in-memory packets without file I/O
    
    Args:
        packets: List of Scapy packet objects
        global_t0: Optional global time reference (for consistency)
        
    Returns:
        Tuple of (events list, global_t0)
    """
    if not SCAPY_AVAILABLE:
        raise ImportError("Scapy is required. Install: pip install scapy")
    
    events: List[EventC] = []
    
    logger.info(f"Converting {len(packets)} Scapy packets to EventC format")
    
    skipped_count = 0
    
    for pkt in packets:
        try:
            # Check if packet has IP layer
            if not pkt.haslayer(IP):
                skipped_count += 1
                continue
            
            ip = pkt[IP]
            
            # Extract timestamp
            t_epoch = float(pkt.time)
            
            # Set GLOBAL_T0 from first packet if not provided
            if global_t0 is None:
                global_t0 = t_epoch
            
            # Extract addresses and size
            saddr = ip.src
            daddr = ip.dst
            pkt_bytes = len(pkt)
            
            # Protocol detection
            sport = 0
            dport = 0
            proto = 'tcp'  # Default to 'tcp' (most common, always in training)
            tcp_flags = 0
            
            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                proto = 'tcp'
                sport = tcp.sport
                dport = tcp.dport
                tcp_flags = int(tcp.flags)  # Extract TCP flags bitmask
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                proto = 'udp'
                sport = udp.sport
                dport = udp.dport
            elif pkt.haslayer(ICMP):
                proto = 'icmp'
            # Note: Other protocols (ARP, IPv6, etc.) default to 'tcp' to match training data
            
            # Create EventC object
            events.append(EventC(
                t_epoch=t_epoch,
                saddr=saddr,
                daddr=daddr,
                proto=proto,
                bytes=pkt_bytes,
                sport=sport,
                dport=dport,
                tcp_flags=tcp_flags
            ))
            
        except Exception as e:
            skipped_count += 1
            continue
    
    logger.info(f"Conversion complete:")
    logger.info(f"  Valid IP events: {len(events):,}")
    logger.info(f"  Skipped (non-IP): {skipped_count:,}")
    logger.info(f"  GLOBAL_T0: {global_t0}")
    
    if not events:
        logger.warning("No valid IP packets found!")
    
    # Ensure global_t0 is always float
    if global_t0 is None:
        global_t0 = 0.0
        logger.warning("No valid packets found, using default T0=0.0")
    
    return events, global_t0


# -----------------------
# PCAP Parsing (Direct Read)
# -----------------------
def load_events_from_pcap(
    path: str, 
    max_packets: Optional[int] = None
) -> Tuple[List[EventC], float]:
    
    if not SCAPY_AVAILABLE:
        raise ImportError("Scapy is required for PCAP parsing. Install: pip install scapy")
    
    events: List[EventC] = []
    global_t0 = None
    
    logger.info(f"Opening PCAP file: {path}")
    if max_packets:
        logger.info(f"  Packet limit: {max_packets:,}")
    
    packet_count = 0
    skipped_count = 0
    
    try:
        with PcapReader(path) as reader:  # type: ignore[misc]
            for pkt in reader:
                packet_count += 1
                
                # Check limit first (fast path)
                if max_packets and len(events) >= max_packets:
                    break
                
                # Extract timestamp from PCAP header (exact microsecond precision)
                t_epoch = float(pkt.time)
                
                # Set GLOBAL_T0 from first packet (for consistent time windows)
                if global_t0 is None:
                    global_t0 = t_epoch
                
                # Fast IP layer check using hasattr (faster than haslayer)
                try:
                    ip = pkt.getlayer('IP')
                    if ip is None:
                        skipped_count += 1
                        continue
                    
                    saddr = ip.src
                    daddr = ip.dst
                    pkt_bytes = len(pkt)
                    
                    # Fast protocol detection using getlayer (faster than haslayer)
                    sport = 0
                    dport = 0
                    proto = 'tcp'  # Default to 'tcp' (most common, always in training)
                    tcp_flags = 0
                    
                    tcp = pkt.getlayer('TCP')
                    if tcp:
                        proto = 'tcp'
                        sport = tcp.sport
                        dport = tcp.dport
                        tcp_flags = int(tcp.flags)  # Extract TCP flags bitmask
                    else:
                        udp = pkt.getlayer('UDP')
                        if udp:
                            proto = 'udp'
                            sport = udp.sport
                            dport = udp.dport
                        elif pkt.getlayer('ICMP'):
                            proto = 'icmp'
                        # Note: Other protocols (ARP, IPv6, etc.) default to 'tcp' to match training data
                
                except (AttributeError, IndexError):
                    skipped_count += 1
                    continue
                
                # Create event with exact timestamp, bytes, and ports
                events.append(EventC(
                    t_epoch=t_epoch,
                    saddr=saddr,
                    daddr=daddr,
                    proto=proto,
                    bytes=pkt_bytes,
                    sport=sport,
                    dport=dport,
                    tcp_flags=tcp_flags
                ))
        
        logger.info(f"PCAP parsing complete:")
        logger.info(f"  Total packets read: {packet_count:,}")
        logger.info(f"  Valid IP events: {len(events):,}")
        logger.info(f"  Skipped (non-IP): {skipped_count:,}")
        logger.info(f"  GLOBAL_T0: {global_t0}")
        
        if not events:
            logger.warning("No valid IP packets found in PCAP file!")
        
        # Ensure global_t0 is always float (fallback to 0.0 if None)
        if global_t0 is None:
            global_t0 = 0.0
            logger.warning("No valid packets found, using default T0=0.0")
        
        return events, global_t0
    
    except FileNotFoundError:
        logger.error(f"PCAP file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error reading PCAP file: {e}")
        raise


# -----------------------
# Feature Engineering (22 features)
# -----------------------
def build_22_features(
    events: List[EventC],
    window_size: int = WINDOW_SIZE,
    global_t0: Optional[float] = None,
) -> pd.DataFrame:
    
    if not events:
        return pd.DataFrame()

    # Convert events to DataFrame
    df_e = pd.DataFrame([e.__dict__ for e in events])
    
    # Set GLOBAL_T0 for consistent time windows
    if global_t0 is not None:
        t0 = global_t0  # Use provided T0 (for multi-file consistency)
        logger.debug(f"Using provided GLOBAL_T0: {t0}")
    else:
        t0 = df_e["t_epoch"].min()  # Use first packet as T0
        logger.debug(f"Computed GLOBAL_T0 from first packet: {t0}")
    
    # Calculate relative time and time windows
    df_e["stime"] = df_e["t_epoch"] - t0  # seconds since T0
    df_e["time_window"] = (df_e["stime"] // float(window_size)).astype(int)

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

        # Packet count and bytes from PCAP
        pkts = int(len(g))
        bytes_ = int(g["bytes"].sum())  # Exact bytes from PCAP packets

        # Duration: calculate from packet timestamps
        # Use actual time span or default to window size
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

        # Protocol: use mode from packets in this window
        proto = str(g["proto"].mode().iloc[0]).lower() if not g["proto"].mode().empty else "tcp"

        # Destination port: extract from PCAP packets (critical for HTTP detection)
        dport = ""
        if "dport" in g.columns:
            dport_values = g["dport"].dropna()
            if not dport_values.empty and dport_values.sum() > 0:
                # Use mode (most common port) for this window
                dport_mode = dport_values.mode()
                if not dport_mode.empty:
                    dport = str(int(dport_mode.iloc[0]))
        
        # TCP Flags: Aggregate from packets in window
        flgs = ""
        if proto == "tcp" and "tcp_flags" in g.columns:
            # Collect unique TCP flags from all packets
            flag_set = set()
            for flags in g["tcp_flags"]:
                if flags > 0:
                    if flags & 0x02: flag_set.add('S')  # SYN
                    if flags & 0x10: flag_set.add('A')  # ACK
                    if flags & 0x01: flag_set.add('F')  # FIN
                    if flags & 0x04: flag_set.add('R')  # RST
                    if flags & 0x08: flag_set.add('P')  # PSH
            # Sort for consistency (e.g., "AFPS")
            flgs = ''.join(sorted(flag_set)) if flag_set else ""
        
        # State: Approximate from TCP flags + packet pattern
        state = ""
        if proto == "tcp":
            # Derive state from flags and bidirectional traffic
            if 'R' in flgs:
                state = "RST"  # Reset connection
            elif 'F' in flgs:
                state = "FIN"  # Connection finishing
            elif 'S' in flgs and 'A' not in flgs:
                state = "REQ"  # SYN only (request without response)
            elif dpkts > 0:  # Bidirectional traffic
                state = "CON"  # Established connection (with responses)
            else:  # Only source packets (typical for attacks)
                state = "INT"  # Interrupted/one-way (no response from destination)
        elif proto == "udp":
            # UDP is connectionless
            state = "CON" if dpkts > 0 else "INT"
        elif proto == "icmp":
            state = "CON" if dpkts > 0 else "INT"
        else:
            state = "INT"  # Default for other protocols

        # TCP sequence number: not available from packet headers without stateful tracking
        # For packet-level analysis, we set to 0 (matches training with aggregated data)
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
            # extras to help debug
            "_time_window": int(tw),
            "_daddr": str(daddr),
            "Dport": dport,  # Destination port for analysis
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


# -----------------------
# Queue-based Detection (For Victim Integration)
# -----------------------
def detector_server_main(input_queue: Queue, models_path: str = "./results/models") -> None:
    """
    Main entry point for detector server running in queue mode.
    Continuously receives packets from victim via queue, processes them through 3-stage pipeline.
    
    Args:
        input_queue: Multiprocessing Queue receiving data from victim
        models_path: Path to model files directory
    """
    logger.info("="*60)
    logger.info("DETECTOR SERVER - Queue Mode")
    logger.info("="*60)
    
    # Start Prometheus metrics
    start_http_server(8000)
    logger.info("[PROM] Metrics exposed at http://localhost:8000/metrics")
    
    # Load models
    logger.info(f"Loading models from: {models_path}")
    try:
        stage1 = joblib.load(os.path.join(models_path, "stage1.pkl"))
        stage2 = joblib.load(os.path.join(models_path, "stage2.pkl"))
        stage3 = xgb.Booster()
        stage3.load_model(os.path.join(models_path, "stage3.json"))
        label_encoder = joblib.load(os.path.join(models_path, "label_encoder.pkl"))
        encoders = joblib.load(os.path.join(models_path, "encoders.pkl"))
        feature_list = joblib.load(os.path.join(models_path, "features.pkl"))
        mapping = joblib.load(os.path.join(models_path, "mapping.pkl"))
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    
    logger.info("Detector server started. Waiting for packets from victim...")
    
    batch_count = 0
    global_t0 = None
    
    while True:
        try:
            # Get batch from queue (block with timeout)
            try:
                batch_data = input_queue.get(timeout=5)
            except queue.Empty:
                continue
            
            batch_count += 1
            batch_id = batch_data.get('batch_id', batch_count)
            packets = batch_data.get('packets', [])
            packet_count = batch_data.get('packet_count', len(packets))
            
            logger.info(f"Processing batch {batch_id}: {packet_count} packets")
            
            if not packets:
                logger.warning(f"Batch {batch_id}: Empty packet list")
                continue
            
            # Convert Scapy packets to EventC
            events, batch_t0 = scapy_packets_to_events(packets, global_t0)
            
            # Update global_t0 for time window consistency
            if global_t0 is None:
                global_t0 = batch_t0
            
            if not events:
                logger.warning(f"Batch {batch_id}: No valid IP packets")
                continue
            
            # Extract 22 features (for Stage 1 & 2)
            logger.debug(f"Extracting 22 features...")
            df_feat = build_22_features(events, window_size=WINDOW_SIZE, global_t0=global_t0)
            df_feat.to_csv(f"features_batch_{batch_id}.csv", sep="\t", index=False)
            if df_feat.empty:
                logger.warning(f"Batch {batch_id}: Feature extraction failed")
                continue
            
            logger.info(f"Extracted features: {len(df_feat)} windows")
            
            # Note: build_22_features() already fills missing values
            
            # Apply encoders
            df_feat = apply_label_encoders_unknown_minus_one(df_feat, encoders)
            
            # Align features
            df_feat = align_features(df_feat, feature_list)
            
            # Stage 1: Attack vs Normal
            logger.debug("Running Stage 1 (Attack vs Normal)...")
            X_feat = df_feat[feature_list].values
            y1 = stage1.predict(X_feat)
            
            attack_mask = (y1 == 1)
            normal_count = np.sum(~attack_mask)
            attack_count = np.sum(attack_mask)
            
            logger.info(f"Stage 1: Normal={normal_count}, Attack={attack_count}")
            
            # Stage 2: Attack type classification
            y2 = np.full(len(y1), "", dtype=object)
            if attack_count > 0:
                logger.debug("Running Stage 2 (Attack Type)...")
                X_attack = X_feat[attack_mask]
                y2_pred = stage2.predict(X_attack)
                y2_mapped = map_predictions(y2_pred, mapping)
                y2[attack_mask] = y2_mapped
                
                # Count attack types
                attack_types = Counter(y2_mapped)
                logger.info(f"Stage 2: {dict(attack_types)}")
            
            # Stage 3: DDoS variant classification
            y3 = np.full(len(y1), "", dtype=object)
            ddos_mask = (y2 == "ddos")
            
            if np.any(ddos_mask):
                logger.debug("Running Stage 3 (DDoS Variants)...")
                df_feat_stage3 = extract_stage3_features(df_feat)
                X_ddos = df_feat_stage3.loc[ddos_mask].values
                
                dmatrix = xgb.DMatrix(X_ddos, feature_names=df_feat_stage3.columns.tolist())
                y3_pred = stage3.predict(dmatrix)
                
                # Map numeric predictions to variant names
                y3_variants = label_encoder.inverse_transform(y3_pred.astype(int))
                y3[ddos_mask] = y3_variants
                
                variant_counts = Counter(y3_variants)
                logger.info(f"Stage 3: {dict(variant_counts)}")
            
            # Update Prometheus metrics
            for idx in range(len(df_feat)):
                dst_ip = df_feat.iloc[idx].get("_daddr", "unknown")
                
                # Feature-level metrics
                if np.isfinite(df_feat.iloc[idx].get("rate", 0)):
                    PACKET_RATE.labels(dst_ip=dst_ip).set(df_feat.iloc[idx]["rate"])
                
                if np.isfinite(df_feat.iloc[idx].get("src_entropy", 0)):
                    SRC_ENTROPY.labels(dst_ip=dst_ip).set(df_feat.iloc[idx]["src_entropy"])
                
                # Prediction-level metrics
                if y1[idx] == 1:  # Attack
                    attack_type = str(y2[idx]).lower() if y2[idx] else "unknown"
                    ddos_variant = str(y3[idx]).lower() if y3[idx] else "unknown"
                    
                    ATTACK_WINDOWS.labels(
                        attack_type=attack_type,
                        ddos_variant=ddos_variant,
                        dst_ip=dst_ip
                    ).inc()
                    
                    # Log alerts
                    if attack_type == "ddos" and ddos_variant:
                        logger.warning(f"🚨 ALERT: DDoS-{ddos_variant.upper()} detected (dst={dst_ip})")
                    elif attack_type:
                        logger.warning(f"🚨 ALERT: {attack_type.upper()} detected (dst={dst_ip})")
                else:
                    NORMAL_WINDOWS.labels(dst_ip=dst_ip).inc()
            
            logger.info(f"Batch {batch_id} processing complete\n")
            
        except KeyboardInterrupt:
            logger.info("Detector server interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            continue
    
    logger.info("Detector server stopped")


# -----------------------
# Main
# -----------------------
def main():
    #promethues metric server
    start_http_server(8000)
    logger.info("[PROM] Metrics exposed at http://localhost:8000/metrics")

    # Auto-detect script directory to handle running from different locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    ap = argparse.ArgumentParser(
        description="DDoS Detection with Three-Stage ML Pipeline (PCAP Direct Read)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage:
  python detector.py --pcap capture.pcap
  
  # With packet limit (recommended for large files):
  python detector.py --pcap merged.pcap --max-packets 500000
  
  # With debug output:
  python detector.py --pcap capture.pcap --max-packets 100000 --debug
  
  # Custom window size:
  python detector.py --pcap capture.pcap --window 60
        """
    )
    
    # Input source arguments
    input_group = ap.add_argument_group('Input (required)')
    input_group.add_argument("--pcap", type=str, required=True,
                             help="Path to PCAP file (required). Can be single capture or merged file.")
    input_group.add_argument("--max-packets", type=int, default=0, 
                             help="Limit packets to read (0 = no limit). "
                                  "Use to control memory: 100K packets ≈ 20MB, 500K ≈ 100MB, 1M ≈ 200MB RAM")
    
    # Model arguments
    model_group = ap.add_argument_group('Model paths')
    model_group.add_argument("--encoders", default=os.path.join(script_dir, "results/models/encoders.pkl"), 
                             help="Path to encoders.pkl")
    model_group.add_argument("--features", default=os.path.join(script_dir, "results/models/features.pkl"), 
                             help="Path to features.pkl")
    model_group.add_argument("--mapping", default=os.path.join(script_dir, "results/models/mapping.pkl"), 
                             help="Path to mapping.pkl")
    model_group.add_argument("--stage1", default=os.path.join(script_dir, "results/models/stage1.pkl"), 
                             help="Path to stage1.pkl")
    model_group.add_argument("--stage2", default=os.path.join(script_dir, "results/models/stage2.pkl"), 
                             help="Path to stage2.pkl")
    model_group.add_argument("--stage3", default=os.path.join(script_dir, "results/models/stage3.json"), 
                             help="Path to stage3.json")
    model_group.add_argument("--label-encoder", default=os.path.join(script_dir, "results/models/label_encoder.pkl"), 
                             help="Path to label_encoder.pkl for Stage 3")
    
    # Processing arguments
    proc_group = ap.add_argument_group('Processing options')
    proc_group.add_argument("--window", type=int, default=WINDOW_SIZE, 
                            help=f"Time window size in seconds (default: {WINDOW_SIZE}s, matches training)")
    proc_group.add_argument("--show", type=int, default=20, 
                            help="Number of sample results to display (default: 20)")
    proc_group.add_argument("--debug", action="store_true", 
                            help="Enable debug logging (shows detailed processing info)")
    
    args = ap.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info("DDoS DETECTOR - Three-Stage ML Pipeline")
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

    # Check Scapy availability
    if not SCAPY_AVAILABLE:
        logger.error("="*60)
        logger.error("ERROR: Scapy is not installed!")
        logger.error("="*60)
        logger.error("Scapy is required for PCAP parsing.")
        logger.error("Install with: pip install scapy")
        logger.error("")
        return
    
    # Load events from PCAP file
    logger.info("="*60)
    logger.info("PCAP DIRECT READ MODE")
    logger.info("="*60)
    
    limit_packets = None if args.max_packets == 0 else args.max_packets
    logger.info(f"Input file: {args.pcap}")
    if limit_packets:
        logger.info(f"Packet limit: {limit_packets:,}")
    else:
        logger.info("Packet limit: None (reading full file)")
    
    try:
        events, global_t0 = load_events_from_pcap(args.pcap, max_packets=limit_packets)
    except Exception as e:
        logger.error(f"Failed to load PCAP file: {e}")
        return
    
    if not events:
        logger.error("No IP packets found in PCAP file. Check file format.")
        return
    
    # Display PCAP info
    time_span = events[-1].t_epoch - events[0].t_epoch
    logger.info(f"")
    logger.info(f"Parsed {len(events):,} packets from PCAP")
    logger.info(f"Time range: {events[0].t_epoch:.3f} to {events[-1].t_epoch:.3f}")
    logger.info(f"Duration: {time_span:.1f}s ({time_span/60:.1f} minutes)")
    logger.info(f"GLOBAL_T0: {global_t0}")
    
    # Build 22 features from PCAP events
    logger.info(f"")
    logger.info(f"Building 22 features with {args.window}s time windows...")
    df_feat = build_22_features(events, window_size=args.window, global_t0=global_t0)

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
            
            # Convert to lowercase for consistency
            y3_labels = [str(variant).lower() for variant in y3_variants]
            
            # Assign predictions to output array
            y3[ddos_mask] = np.array(y3_labels, dtype=object)
            logger.debug(f"[DEBUG] Assigned {len(y3_labels)} Stage 3 predictions")
            logger.debug(f"[DEBUG] First 5 DDoS predictions: {y3_labels[:5]}")
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
        print(f"[+] Stage 2 distribution: {dict(Counter(out[attack_mask]['stage2_pred'].tolist()))}")
        # Show Stage 3 distribution if there are DDoS attacks
        if ddos_mask is not None and ddos_mask.any():
            print(f"[+] Stage 3 distribution (DDoS variants): {dict(Counter(out[ddos_mask]['stage3_pred'].tolist()))}")
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
