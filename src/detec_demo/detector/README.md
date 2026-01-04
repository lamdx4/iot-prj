# DDoS Detector - Three-Stage ML Pipeline with Prometheus Metrics

## 📖 Overview

Real-time DDoS attack detection system using **Three-Stage Hierarchical Machine Learning model** with Prometheus metrics export and Grafana visualization.

**Stage 1**: Binary Classification (Attack vs Normal) - 99.26% accuracy
**Stage 2**: Multi-class Classification (DDoS vs DoS vs Reconnaissance) - 97.58% accuracy
**Stage 3**: DDoS Variant Classification (HTTP, TCP, UDP, Normal)
**Overall**: 97.19% accuracy

## 🏗️ Architecture

```
flows_c.txt (Argus -c) ──┐
                         ├──► Windowing 30s ──► Feature Engineering (22 features)
flows_s.txt (Argus -s) ──┘                      
                                                ↓
                                         Stage 1: Attack/Normal (99.26%)
                                                ↓
                                         Stage 2: DDoS/DoS/Recon (97.58%)
                                                ↓
                                    Stage 3: DDoS Variants (HTTP/TCP/UDP/Normal)
                                    ├─ ML Prediction (XGBoost)
                                                ↓
                                         Prometheus Metrics Export
                                         http://localhost:8000/metrics
                                                ↓
                                         Prometheus Scraper
                                         http://localhost:9090
                                                ↓
                                         Grafana Dashboard
                                         http://localhost:3000
```

## 🚀 Quick Start with Demo Script

### Automated Demo (Recommended)

```bash
cd /home/dngnguyen/Documents/kỳ_154/Iot/Iot-demo/detector

# Make script executable
chmod +x demo_pcap_replay.sh

# Run demo (will start containers automatically)
./demo_pcap_replay.sh

# Script will: (883 lines)
├── demo_pcap_replay.sh         # Automated demo script
├── flows_c.txt                 # Argus -c output (event-level, 929k lines)
├── flows_s.txt                 # Argus -s output (flow-level, 929k lines)
└── results/models/
    ├── encoders.pkl            # LabelEncoders for categorical features
    ├── features.pkl            # 22 feature names in correct order
    ├── mapping.pkl             # Attack type mapping (0→DDoS, 1→DoS, 2→Recon)
    ├── stage1.pkl              # Binary classifier (XGBoost)
    ├── stage2.pkl              # Multi-class classifier (XGBoost)
    ├── stage3.json             # DDoS variant classifier (XGBoost)
    └── label_encoder.pkl       # Encoder for Stage 3 classes: {0:Normal, 1:HTTP, 2:TCP, 3:UDP}
# Access during demo:
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### Manual Mode

```bash
# If you prefer to run without automation
python detector.py --limit-c 100000 --limit-s 100000 --show 20

# View metrics at: http://localhost:8000/metrics
```

## 🚀 Quick Start (Manual)

```bash
# Python 3.10+ with packages:
pip install numpy pandas scikit-learn joblib xgboost
```

### 2. File Structure

```
detector/
├── detector.py                 # Main detection script
├── flows_c.txt                 # Argus -c output (event-level)
├── flows_s.txt                 # Argus -s output (flow-level)
└── results/models/
    ├── encoders.pkl            # LabelEncoders for categorical features
    ├── features.pkl            # 22 feature names in correct order
    ├── mapping.pkl             # Attack type mapping (0→DDoS, 1→DoS, 2→Recon)
    ├── stage1.pkl              # Binary classifier model
    └── stage2.pkl              # Multi-class classifier model
```

##Automated demo (recommended)
./demo_pcap_replay.sh

# Basic usage (analyze all data, ~45 minutes)
python detector.py

# Quick test with limited data (~2-3 minutes)
python detector.py --limit-c 100000 --limit-s 100000 --show 20

# With debug logging
python detector.py --debug

# Custom window size
python detector.py --window 60
```

### 4. View Metrics (when running detector)

**Prometheus endpoint:**
```
http://localhost:8000/metrics
```

**Available metrics:**
- `ddos_attack_windows_total` - Counter: attack windows by type and variant
- `ddos_normal_windows_total` - Counter: normal windows
- `ddos_packet_rate` - Gauge: packets per second
- `ddos_src_entropy` - Gauge: Shannon entropy of source IPs

# Custom window size
python detector.py --window 60
```

## 📊 Features

### 22 Features Extracted

**Traffic Volume** (4):
- `pkts`, `bytes`, `dur`, `seq`

**Inter-Arrival Time** (5):
- `mean`, `stddev`, `sum`, `min`, `max`

**Directional** (4):
- `spkts`, `dpkts`, `sbytes`, `dbytes`

**Rate** (3):
- `rate`, `srate`, `drate`

**Source Diversity** (3) - **Key for DDoS detection**:
- `unique_src_count`: Number of unique source IPs
- `src_entropy`: Shannon entropy of source distribution
- `top_src_ratio`: Ratio of top source to total packets

**Categorical** (3):
- `proto`: Protocol (tcp/udp/icmp)
- `state`: Connection state (INT/CON/REQ/...)
- `flgs`: TCP flags (S/SA/...)

## 🎯 How It Works

### Step 1: Data Loading
- Parse `flows_c.txt` for event-level data (timestamps, IPs, protocols)
- Parse `flows_s.txt` for flow metadata (state, flags)

### Step 2: Windowing
```python
time_window = floor(stime / 30)  # 30-second windows
```

### Step 3: Grouping
```python
group by (time_window, daddr)  # Per victim IP per window
```

### Step 4: Feature Engineering
- Calculate 22 features for each window-victim pair
- **Source Diversity** features distinguish DDoS (many sources) from DoS (few sources)

### Step 5: Inference
- **Stage 1**: Predict Attack (1) or Normal (0)
- **Stage 2**: If Attack, classify as DDoS/DoS/Reconnaissance

## 📈 Performance

| Metric | Stage 1 | Stage 2 | Overall |
|--------|---------|---------|---------|
| Accuracy | 99.26% | 97.58% | 97.19% |
| Precision | 99.99% | 97.91% | 97.35% |
| Recall | 99.24% | 97.58% | 97.19% |
| F1-Score | 99.62% | 97.64% | 97.17% |
| Training Time | 31s (GPU) | 59s (GPU) | 90s |
| Inference Speed | 1M samples/s | 2.9M samples/s | 1.2M samples/s |

## 🔧 Command-Line Options

```
--flows-c PATH        Path to flows_c.txt (default: flows_c.txt)
--flows-s PATH        Path to flows_s.txt (default: flows_s.txt)
--encoders PATH       Path to encoders.pkl (default: results/models/encoders.pkl)
--features PATH       Path to features.pkl (default: results/models/features.pkl)
--mapping PATH        Path to mapping.pkl (default: results/models/mapping.pkl)
--stage1 PATH         Path to stage1.pkl (default: results/models/stage1.pkl)
--stage2 PATH         Path to stage2.pkl (default: results/models/stage2.pkl)
--window SECONDS      Time window size (default: 30)
--limit-c N           Limit flows_c lines (0=no limit, default: 0)
--limit-s N           Limit flows_s lines (0=no limit, default: 0)
--show N              Show top N results (default: 20)
--debug               Enable debug logging
```

## 📝 Example Output

```
============================================================
DDoS DETECTOR - Two-Stage ML Pipeline
============================================================
[12:34:56] INFO: Loading Stage 1 model from: results/models/stage1.pkl
[12:34:56] INFO: Loading Stage 2 model from: results/models/stage2.pkl
[12:34:56] INFO: Parsed 13,873 events from flows_c
[12:34:56] INFO: Parsed 13,954 flows from flows_s
[12:34:57] INFO: Generated 123 time windows for analysis
[12:34:57] INFO: Running Stage 1: Binary Classification
[12:34:57] INFO: Running Stage 2: Multi-class Classification (120 attacks)
============================================================
DETECTION RESULTS
============================================================
[+] Windows analyzed: 123 (window=30s)
[+] Stage 1 distribution: {'Attack': 120, 'Normal': 3}
[+] Stage 2 distribution: {'DDoS': 80, 'DoS': 35, 'Recon': 5}
============================================================
SAMPLE RESULTS (Top 20)
============================================================
_time_window  _daddr      pkts   rate  unique_src_count  src_entropy  top_src_ratio  proto  state  flgs  stage1_pred  stage2_pred
           0  172.19.0.3  1000  33.33                 1         0.00           1.00    tcp    INT          Attack       DoS
           1  172.19.0.3  5000 166.67               500         8.97           0.02    tcp    INT     S    Attack       DDoS
           2  172.19.0.3  8000 266.67              1000         9.97           0.01    tcp    INT     S    Attack       DDoS
...
============================================================
[12:34:58] INFO: Detection completed successfully!
```

## 🧪 Testing

### Unit Test (manual)
```bash
# Test with 100 lines
python detector.py --limit-c 100 --limit-s 100 --show 5
```

### Validation
```bash
# Verify all models load correctly
python -c "
import joblib
print('✅ encoders:', type(joblib.load('results/models/encoders.pkl')))
print('✅ features:', type(joblib.load('results/models/features.pkl')))
print('✅ mapping:', joblib.load('results/models/mapping.pkl'))
print('✅ stage1:', type(joblib.load('results/models/stage1.pkl')))
print('✅ stage2:', type(joblib.load('results/models/stage2.pkl')))
"
```

## 🐛 Troubleshooting

### Error: "No module named 'joblib'"
```bash
pip install joblib numpy pandas scikit-learn xgboost
```

### Error: "No events parsed from flows_c"
- Check file path: `--flows-c /correct/path/to/flows_c.txt`
- Verify file format (Argus `-c` output)
- Try with `--debug` flag for detailed error messages

### Error: "FileNotFoundError: results/models/stage1.pkl"
- Ensure models are in `results/models/` directory
- Or specify custom paths: `--stage1 /path/to/stage1.pkl`

### Low accuracy on your data
- Verify Argus output format matches training data
- Check if window size matches training (default: 30s)
-  📈 Performance

| Metric | Stage 1 | Stage 2 | Stage 3 | Overall |
|--------|---------|---------|---------|---------|
| Accuracy | 99.26% | 97.58% | ~95% | 97.19% |
| Precision | 99.99% | 97.91% | ~96% | 97.35% |
| Recall | 99.24% | 97.58% | ~94% | 97.19% |
| F1-Score | 99.62% | 97.64% | ~95% | 97.17% |

*Stage 3: ML 

## 🔧 Command-Line Options

```
--flows-c PATH        Path to flows_c.txt (default: flows_c.txt)
--flows-s PATH        Path to flows_s.txt (default: flows_s.txt)
--encoders PATH       Path to encoders.pkl (default: results/models/encoders.pkl)
--features PATH       Path to features.pkl (default: results/models/features.pkl)
--mapping PATH        Path to mapping.pkl (default: results/models/mapping.pkl)
--stage1 PATH         Path to stage1.pkl (default: results/models/stage1.pkl)
--stage2 PATH         Path to stage2.pkl (default: results/models/stage2.pkl)
--stage3 PATH         Path to stage3.json (default: results/models/stage3.json)
--label-encoder PATH  Path to label_encoder.pkl (default: results/models/label_encoder.pkl)
--window SECONDS      Time window size (default: 30)
--limit-c N           Limit flows_c lines (0=no limit, default: 0)
--limit-s N           Limit flows_s lines (0=no limit, default: 0)
--show N              Show top N results (default: 20)
--debug               Enable debug logging
```

## 📝 Example Output

```
============================================================
DDoS DETECTOR - Three-Stage ML Pipeline
============================================================
[11:04:52] INFO: Loading Stage 1 model from: results/models/stage1.pkl
[11:04:52] INFO: Loading Stage 2 model from: results/models/stage2.pkl
[11:04:52] INFO: Loading Stage 3 model from: results/models/stage3.json
[11:04:52] INFO: Parsed 99,995 events from flows_c (limit: 100000)
[11:04:56] INFO: Parsed 99,999 flows from flows_s (limit: 100000)
[11:04:56] INFO: Building 22 features with window_size=30s...
[11:09:35] INFO: Generated 19,174 time windows for analysis
[11:09:35] INFO: Running Stage 1: Binary Classification (Attack vs Normal)...
[11:09:35] INFO: Running Stage 2: Multi-class Classification (18,547 attacks)...
[11:09:35] INFO: Running Stage 3: DDoS Variant Classification (426 DDoS attacks)...
============================================================
DETECTION RESULTS
============================================================
[+] Windows analyzed: 19,174 (window=30s)
[+] Stage 1 distribution: {1: 18547, 0: 627}  (Attack/Normal)
[+] Stage 2 distribution: {'DoS': 17851, 'DDoS': 426, 'Reconnaissance': 270}
[+] Stage 3 distribution (DDoS variants): {'http': 82, 'tcp': 224, 'udp': 133, 'normal': 212}
============================================================
SAMPLE RESULTS (Top 20)
============================================================
_time_window  _daddr       pkts   rate  unique_src_count  src_entropy  top_src_ratio  proto state flgs stage1_pred  stage2_pred  stage3_pred
           0 192.168.100.3   90  3.00                 3      1.585        0.333       tcp  REQ  e s           1         DDoS           http
           1 192.168.100.3   90  3.00                 3      1.585        0.333       tcp  REQ  e s           1         DDoS           http
           2 192.168.100.3   89  2.97                 3      1.585        0.337       tcp  REQ  e s           1         DoS           normal
...
============================================================
[11:09:38] INFO: Detection completed successfully!
[11:09:38] INFO: [PROM] Metrics exposed at http://localhost:8000/metrics
[11:09:38] INFO: [PROM] Waiting for Prometheus scrape...
```

## 📊 Grafana Dashboard

### Recommended Panels

**1. Attack Type Distribution (Pie Chart)**
```promql
sum by (attack_type) (ddos_attack_windows_total)
```

**2. DDoS Variants Breakdown (Pie Chart)**
```promql
sum by (ddos_variant) (ddos_attack_windows_total{attack_type="ddos"})
```
Expected: http (82), tcp (224), udp (133), normal (212)

**3. Source IP Entropy (Bar Gauge - Top 15)**
```promql
topk(15, max(ddos_src_entropy) by (dst_ip))
```
Thresholds: 0-1.5 (green/normal), 1.5-3 (yellow/moderate), >3 (red/high DDoS)

**4. Packet Rate Distribution (Bar Gauge - Top 20)**
```promql
topk(20, avg(ddos_packet_rate) by (dst_ip) > 0.5)
```
Thresholds: <0.5 (green/normal), 0.5-2 (yellow/DoS), 2-10 (orange/DDoS), >10 (red/high-rate)

### Accessing Grafana

```
URL: http://localhost:3000
Username: admin
Password: admin
```

## 📚 Technical Details

### Source Diversity Algorithm (DDoS vs DoS)
```python
# DDoS characteristics: Many sources, high entropy, low top ratio
unique_src_count = len(set(sources))       # 100+ for DDoS
src_entropy = -Σ(pᵢ × log₂(pᵢ))            # 5-10 for DDoS
top_src_ratio = max(src_counts) / total   # 0.01-0.05 for DDoS

# DoS characteristics: Few sources, low entropy, high top ratio  
unique_src_count = 1-10                    # <10 for DoS
src_entropy = 0.0-2.0                      # <2 for DoS
top_src_ratio = 0.8-1.0                    # >0.8 for DoS
```

### Stage 3 Hybrid Approach

**ML Model (XGBoost):**
- Trained on Bot-IoT balanced dataset (7,635 samples per class)
- Classes: Normal (0), HTTP (1), TCP (2), UDP (3)
- 16 features optimized for variant discrimination

### Encoding Strategy
- **Categorical features**: LabelEncoder fitted on training data
- **Unknown values**: Encoded as `-1` (safe handling for inference)
- **Missing values**: Filled with `0.0` for numeric, `""` for categorical

### Model Architecture
- **Stage 1**: XGBoost Binary Classifier (200 trees, depth=6, GPU)
- **Stage 2**: XGBoost Multi-class Classifier (200 trees, depth=6, GPU)
- **Stage 3**: XGBoost Multi-class Classifier (200 trees, depth=6, JSON format
src_entropy = -Σ(pᵢ × log₂(pᵢ))            # High for DDoS
top_src_ratio = max(src_counts) / total   # Low for DDoS

# DoS: Few sources, low entropy, high top ratio
unique_src_count = 1-10                    # Low for DoS
src_entropy = 0.0-2.0                      # Low for DoS
top_src_ratio = 0.8-1.0                    # High for DoS
```

### Encoding Strategy
- **Categorical features**: LabelEncoder fitted on training data
- **Unknown values**: Encoded as `-1` (safe handling for inference)
- **Missing values**: Filled with `0.0` for numeric, `""` for categorical

### Model Architecture
- **Stage 1**: XGBoost Binary Classifier (200 trees, depth=6)
- **Stage 2**: XGBoost Multi-class Classifier (200 trees, depth=6)
- **GPU accelerated**: Training on Tesla T4, 160x faster than CPU

## 📖 References

See `doc/` folder for detailed documentation:
- `chuong_2_mo_hinh_de_xuat.md`: Model architecture and training
- `chuong_3_thuc_nghiem_va_thao_luan.md`: Experiments and results
- `IMPROVEMENTS.md`: Recent improvements and changelog

## 📄 License

Research project - UNSW Bot-IoT Dataset

## 👤 Author

Graduate student project - IoT Security
