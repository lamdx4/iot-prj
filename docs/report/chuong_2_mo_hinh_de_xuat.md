# CHƯƠNG 2: MÔ HÌNH ĐỀ XUẤT

## 2.1. Kiến trúc tổng thể hệ thống

### 2.1.1. Sơ đồ tổng quan

Hệ thống phát hiện và phân loại botnet IoT được thiết kế theo kiến trúc phân cấp hai giai đoạn (Two-Stage Hierarchical Model), trong đó mỗi giai đoạn chuyên biệt hóa cho một tác vụ phân loại cụ thể. Kiến trúc này cho phép tối ưu hóa hiệu năng của từng giai đoạn một cách độc lập và giảm độ phức tạp của bài toán tổng thể.

```
Luồng Dữ Liệu Tổng Thể:

Raw Data (74 CSV files, ~16GB)
         ↓
[Merge & Batch] → 8 Batches (~2GB each)
         ↓
[Feature Engineering] → 22 features
         ↓
┌─────────────────────────────────────┐
│    STAGE 1: Binary Classification   │
│    (Attack vs Normal)                │
│    Input: 22 features                │
│    Output: is_attack (0/1)           │
│    Model: XGBoost Binary Classifier  │
└─────────────────────────────────────┘
         ↓
   Decision Point
         ↓
    ┌────────┴────────┐
    │                 │
 Normal          Attack
(Output)            ↓
         ┌─────────────────────────────────────┐
         │  STAGE 2: Multi-class Classification│
         │  (DDoS vs DoS vs Reconnaissance)    │
         │  Input: 22 features                 │
         │  Output: attack_type (0/1/2)        │
         │  Model: XGBoost Multi-class         │
         └─────────────────────────────────────┘
                     ↓
             Final Classification
         (Normal / DDoS / DoS / Reconnaissance)
```

### 2.1.2. Các thành phần chính

Hệ thống bao gồm năm thành phần chính:

**Thành phần 1 - Tiền xử lý dữ liệu (Data Preprocessing)**:

- Chức năng: Gộp 74 file CSV thành 8 batch files, phân tích thống kê và chọn batch phù hợp cho huấn luyện.
- Input: 74 file CSV gốc từ tập dữ liệu Bot-IoT
- Output: 8 batch files và file thống kê JSON
- Công nghệ: Pandas DataFrame operations, JSON serialization

**Thành phần 2 - Kỹ thuật đặc trưng (Feature Engineering)**:

- Chức năng: Trích xuất 22 đặc trưng từ dữ liệu thô, bao gồm 3 đặc trưng source diversity đặc biệt để phân biệt DDoS và DoS.
- Input: Raw network flows với 35+ columns
- Output: 22 engineered features
- Công nghệ: Scipy entropy, NumPy aggregation, time-window analysis

**Thành phần 3 - Stage 1: Binary Classifier:**

- Chức năng: Phân loại nhị phân (Attack vs Normal)
- Input: 22 features
- Output: is_attack (0 = Normal, 1 = Attack)
- Mô hình: XGBoost Binary Classifier với SMOTE và scale_pos_weight
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

**Thành phần 4 - Stage 2: Multi-class Classifier:**

- Chức năng: Phân loại đa lớp cho các mẫu được xác định là tấn công
- Input: 22 features (chỉ attack samples)
- Output: attack_type (0 = DDoS, 1 = DoS, 2 = Reconnaissance)
- Mô hình: XGBoost Multi-class Classifier với SMOTE và balanced sample weights
- Metrics: Accuracy, Precision, Recall, F1-Score (weighted average)

**Thành phần 5 - Pipeline Integration:**

- Chức năng: Kết hợp dự đoán từ hai giai đoạn để đưa ra quyết định cuối cùng
- Logic: Nếu Stage 1 dự đoán Normal → Output "Normal", nếu Attack → truyền qua Stage 2 để xác định loại tấn công cụ thể
- Output: Final classification (Normal / DDoS / DoS / Reconnaissance)

## 2.2. Các thuật toán và mô-đun chính

### 2.2.1. Thuật toán XGBoost (Extreme Gradient Boosting)

XGBoost là thuật toán học máy dựa trên gradient boosting [2], [15], xây dựng một ensemble của nhiều cây quyết định yếu (weak learners) theo cách tuần tự, trong đó mỗi cây mới được huấn luyện để sửa lỗi của các cây trước đó.

**Hàm mục tiêu**:

```
L(φ) = Σ l(ŷᵢ, yᵢ) + Σ Ω(fₖ)
```

Trong đó:

- `l(ŷᵢ, yᵢ)`: Loss function (logloss cho binary, mlogloss cho multi-class)
- `Ω(fₖ)`: Regularization term để tránh overfitting
- `φ`: Tập hợp các tham số của model

**Regularization term**:

```
Ω(f) = γT + (λ/2) Σ wⱼ²
```

Trong đó:

- `T`: Số lượng lá (leaves) trong cây
- `wⱼ`: Trọng số của lá thứ j
- `γ`: Complexity control parameter
- `λ`: L2 regularization parameter

### 2.2.2. Thuật toán SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE được sử dụng để xử lý mất cân bằng dữ liệu bằng cách tạo ra các mẫu tổng hợp cho lớp thiểu số [3].

**Giả mã thuật toán SMOTE**:

```
Algorithm: SMOTE(X_minority, N, k)
Input:
  - X_minority: Tập mẫu lớp thiểu số
  - N: Tỷ lệ oversampling (ví dụ: 200 = tăng gấp đôi)
  - k: Số lân cận gần nhất (thường k=5)
Output:
  - X_synthetic: Tập mẫu tổng hợp

1: FOR mỗi mẫu xᵢ ∈ X_minority:
2:   Tìm k nearest neighbors của xᵢ (sử dụng Euclidean distance)
3:   FOR j = 1 TO (N/100):
4:     Chọn ngẫu nhiên một neighbor xₙ từ k neighbors
5:     Tạo mẫu tổng hợp:
        x_new = xᵢ + rand(0,1) × (xₙ - xᵢ)
6:     Thêm x_new vào X_synthetic
7:   END FOR
8: END FOR
9: RETURN X_minority ∪ X_synthetic
```

**Ứng dụng trong hệ thống**:

- Stage 1: Tăng số lượng mẫu Normal từ ~7,769 lên khoảng 2 triệu (sampling_strategy=0.1)
- Stage 2: Tăng số lượng mẫu Reconnaissance (lớp thiểu số) lên 10% của lớp đa số

### 2.2.3. Thuật toán tính Source Diversity Features

Đây là thuật toán quan trọng nhất để phân biệt giữa DDoS (nhiều nguồn) và DoS (ít nguồn).

**Giả mã thuật toán**:

```
Algorithm: CalculateSourceDiversity(flows, window_size)
Input:
  - flows: Tập luồng mạng với (stime, saddr, daddr)
  - window_size: Kích thước cửa sổ thời gian (seconds)
Output:
  - diversity_features: (unique_src_count, src_entropy, top_src_ratio)

1: Tạo time_window = floor(stime / window_size)

2: FOR mỗi (time_window, target_daddr):
3:   Group = flows WHERE time_window=tw AND daddr=target_daddr
4:
5:   // Feature 1: Unique source count
6:   unique_src_count = |{saddr ∈ Group}|
7:
8:   // Feature 2: Source entropy
9:   src_counts = frequency(saddr) FOR saddr ∈ Group
10:  src_probs = src_counts / Σ(src_counts)
11:  src_entropy = -Σ(pᵢ × log₂(pᵢ)) FOR pᵢ ∈ src_probs
12:
13:  // Feature 3: Top source ratio
14:  top_src_count = max(src_counts)
15:  top_src_ratio = top_src_count / |Group|
16:
17:  ASSIGN diversity_features TO all flows IN Group
18: END FOR

19: RETURN flows WITH diversity_features
```

**Ý nghĩa các features**:

- `unique_src_count`: DDoS có giá trị cao (hàng nghìn nguồn), DoS có giá trị thấp (< 10 nguồn)
- `src_entropy`: DDoS có entropy cao (phân phối đều), DoS có entropy thấp (tập trung)
- `top_src_ratio`: DDoS có tỷ lệ thấp (< 0.1), DoS có tỷ lệ cao (> 0.8)

### 2.2.4. Mô-đun GPU Optimization

Để tăng tốc độ huấn luyện, hệ thống sử dụng GPU (nếu có) thông qua XGBoost GPU support [2], [24].

**Thuật toán phát hiện và cấu hình GPU**:

```
Algorithm: ConfigureGPU()
Output: (USE_GPU, DEVICE, TREE_METHOD)

1: TRY:
2:   Chạy lệnh: nvidia-smi --query-gpu=name,memory.total
3:   IF lệnh thành công:
4:     gpu_name, gpu_memory = parse output
5:     PRINT "GPU detected: {gpu_name}, {gpu_memory}"
6:     USE_GPU = True
7:     DEVICE = 'cuda'
8:     TREE_METHOD = 'hist'  // XGBoost 3.x uses 'hist' with device='cuda'
9:   ELSE:
10:    USE_GPU = False
11:    DEVICE = 'cpu'
12:    TREE_METHOD = 'hist'
13: EXCEPT:
14:   USE_GPU = False
15:   DEVICE = 'cpu'
16:   TREE_METHOD = 'hist'
17: RETURN (USE_GPU, DEVICE, TREE_METHOD)
```

**Cấu hình XGBoost với GPU** (XGBoost 3.x):

```python
XGBClassifier(
    tree_method='hist',        # Histogram-based algorithm
    device='cuda',             # Use GPU (cuda:0 for specific GPU)
    n_jobs=1,                  # GPU handles parallelism internally
    max_bin=512,               # More bins for GPU (default=256)
    ...
)
```

## 2.3. Quy trình hoạt động chi tiết

### 2.3.1. Tổng quan quy trình

Hệ thống hoạt động theo 8 bước chính, từ tiền xử lý dữ liệu đến đánh giá kết quả cuối cùng.

### 2.3.2. Step 0: Merge và phân tích batches

**Mục tiêu**: Giảm số lần đọc file và chọn batch tốt nhất cho huấn luyện.

**Input**:

- 74 file CSV gốc (khoảng 230MB mỗi file)
- Tổng dung lượng: ~16GB

**Quy trình**:

1. Chia 74 files thành 8 groups (10 files/group, group cuối 4 files)
2. Merge mỗi group thành 1 batch file
3. Phân tích thống kê cho mỗi batch:
   - Total records
   - Category distribution (Normal, DDoS, DoS, Reconnaissance, Theft)
   - Missing values
   - Protocol distribution
4. Lưu thống kê vào `batch_statistics.json`

**Output**:

- 8 batch files: batch_01.csv đến batch_08.csv (~2-2.6GB mỗi file)
- File thống kê: batch_statistics.json

**Lựa chọn batch cho training**: Chọn batch_01 và batch_04 vì có số lượng Normal samples cao nhất (cần thiết cho Stage 1).

### 2.3.3. Step 1: Tạo balanced test set

**Mục tiêu**: Tạo tập test cân bằng để đánh giá công bằng.

**Input**: batch_02.csv (chọn làm nguồn test vì không dùng cho training)

**Quy trình**:

1. Sample từ batch_02:
   - Normal: 2,000 samples
   - DDoS: 35,000 samples
   - DoS: 50,000 samples
   - Reconnaissance: 13,000 samples
2. Shuffle và lưu thành `test_balanced_100k.csv`

**Output**: Tập test cân bằng với 100,000 records

**Lý do**: Tập test gốc có tỷ lệ mất cân bằng 2000:1, khiến accuracy không phản ánh đúng khả năng phát hiện Normal.

### 2.3.4. Step 2: Load training data

**Input**:

- batch_01.csv (~10M records)
- batch_04.csv (~10M records)

**Quy trình**:

1. Đọc batch_01.csv với `pd.read_csv(low_memory=False)`
2. Đọc batch_04.csv
3. Concatenate 2 DataFrames: `df_train = pd.concat([df1, df2])`
4. Garbage collection: `del df1, df2; gc.collect()`

**Output**: df_train với 20 triệu records

**Tối ưu bộ nhớ**: Chỉ load 2 batches thay vì 3 để tiết kiệm ~8GB RAM.

### 2.3.5. Step 3: Load test data

**Input**: test_balanced_100k.csv

**Quy trình**:

1. Đọc file test: `df_test = pd.read_csv(TEST_FILE)`
2. Kiểm tra distribution để đảm bảo cân bằng

**Output**: df_test với 100,000 records (cân bằng)

### 2.3.6. Step 4: Feature Engineering

Đây là bước quan trọng nhất, quyết định chất lượng của mô hình.

**Bước 4.1: Tạo Source Diversity Features**

Input: df_train và df_test với columns [stime, saddr, daddr, ...]

Quy trình:

1. Tạo time_window = floor(stime / 30) // Window 30 giây
2. Group by (time_window, daddr)
3. Với mỗi group:
   - Tính unique_src_count = số lượng saddr duy nhất
   - Tính src_entropy = -Σ(pᵢ log₂ pᵢ)
   - Tính top_src_ratio = max_count / total_count
4. Merge features trở lại df_train và df_test
5. Fill NaN với giá trị mặc định (1, 0.0, 1.0)

**Quan trọng**: Train và test PHẢI xử lý RIÊNG BIỆT để tránh data leakage.

Output: df_train và df_test có thêm 3 cột mới

**Bước 4.2: Feature Selection**

Columns cần loại bỏ:

- Identifiers: pkSeqID, saddr, sport, daddr, dport
- MAC addresses: smac, dmac, soui, doui
- Organizationally unique identifiers: sco, dco
- Labels: attack, category, subcategory
- Timestamps: stime, ltime (giữ lại 'dur' - duration)

Features cuối cùng (22 features):

```
1. flgs          - TCP flags
2. proto         - Protocol (TCP/UDP/ICMP)
3. pkts          - Total packets
4. bytes         - Total bytes
5. state         - Connection state
6. seq           - Sequence number
7. dur           - Duration
8. mean          - Mean of inter-arrival time
9. stddev        - Standard deviation
10. sum          - Sum of inter-arrival time
11. min          - Minimum inter-arrival time
12. max          - Maximum inter-arrival time
13. spkts        - Source packets
14. dpkts        - Destination packets
15. sbytes       - Source bytes
16. dbytes       - Destination bytes
17. rate         - Packet rate
18. srate        - Source rate
19. drate        - Destination rate
20. unique_src_count  - Đặc trưng mới (DDoS detector)
21. src_entropy       - Đặc trưng mới (Diversity measure)
22. top_src_ratio     - Đặc trưng mới (Concentration measure)
```

**Bước 4.3: Xử lý Missing Values**

Quy trình:

1. Tính median cho mỗi numeric column từ TRAIN set
2. Fill NaN trong TRAIN set bằng train medians
3. Fill NaN trong TEST set bằng train medians (KHÔNG dùng test medians → tránh leakage)

**Bước 4.4: Encoding Categorical Features**

Categorical columns: flgs, proto, state

Quy trình:

```python
FOR each categorical column:
    1. Fit LabelEncoder ONLY on train data
    2. Transform train data
    3. Transform test data với unknown handling:
       - Nếu value có trong train → transform bình thường
       - Nếu value KHÔNG có trong train → gán -1
    4. Save encoder để dùng cho inference
```

**Critical**: Encoder chỉ được fit trên train data để tránh data leakage.

### 2.3.7. Step 5: Train Stage 1 (Binary Classification)

**Mục tiêu**: Phân biệt Attack vs Normal

**Bước 5.1: Tạo binary labels**

```python
df_train['is_attack'] = (df_train['category'] != 'Normal').astype(int)
df_test['is_attack'] = (df_test['category'] != 'Normal').astype(int)
```

**Bước 5.2: Train-Validation Split**

```
Stratified split (85%-15%):
- Train: 17M records
- Validation: 3M records
- Stratify by 'is_attack' để giữ tỷ lệ
```

**Bước 5.3: SMOTE Oversampling**

Input: X_train (17M), y_train (99.96% attack, 0.04% normal)

Quy trình:

```
1. Tính k = min(normal_count - 1, 5) = 5
2. SMOTE(sampling_strategy=0.1, k_neighbors=5)
   - Tăng Normal từ ~7,000 lên ~2M (10% của Attack)
3. Output: 19M resampled records
```

**Bước 5.4: Huấn luyện XGBoost**

Cấu hình:

```python
XGBClassifier(
    n_estimators=200,             # Số cây
    max_depth=6,                  # Độ sâu tối đa
    learning_rate=0.1,            # Tốc độ học
    subsample=0.8,                # Tỷ lệ sample cho mỗi cây
    colsample_bytree=0.8,         # Tỷ lệ features cho mỗi cây
    scale_pos_weight=2574,        # (Normal_count / Attack_count)
    tree_method='hist',
    device='cuda',                # GPU
    eval_metric='logloss',
    early_stopping_rounds=20
)
```

Huấn luyện:

```python
model_s1.fit(
    X_train_resampled, y_train_resampled,
    eval_set=[(X_train_original, y_train_original),
              (X_val, y_val)],
    verbose=False
)
```

**Bước 5.5: Đánh giá Stage 1**

Metrics trên test set:

- Accuracy: 99.26%
- Precision: 99.99%
- Recall: 99.24%
- F1-Score: 99.62%
- ROC-AUC: 99.99%

**Thời gian huấn luyện**: ~30 giây (với GPU T4)

### 2.3.8. Step 6: Train Stage 2 (Multi-class Classification)

**Mục tiêu**: Phân loại loại tấn công (DDoS, DoS, Reconnaissance)

**Bước 6.1: Filter attack samples**

```python
df_train_attacks = df_train[df_train['is_attack'] == 1]
df_test_attacks = df_test[df_test['is_attack'] == 1]
```

**Bước 6.2: Attack mapping**

```
DDoS            → 0
DoS             → 1
Reconnaissance  → 2
(Theft bị loại bỏ vì quá ít: 1,587 samples)
```

**Bước 6.3: Train-Validation Split**

```
Stratified split (85%-15%):
- Train: 17M attacks
- Validation: 3M attacks
```

**Bước 6.4: SMOTE (Limited)**

Phân phối trước SMOTE:

- DoS: 13M (65%)
- DDoS: 5.2M (26%)
- Reconnaissance: 1.8M (9%)

Strategy:

```
Target = 10% của DoS = 1.3M
Chỉ tăng Reconnaissance từ 1.8M → 1.3M (không cần thiết)
→ Bỏ qua SMOTE, dùng balanced sample weights
```

**Bước 6.5: Huấn luyện XGBoost Multi-class**

Cấu hình:

```python
XGBClassifier(
    objective='multi:softmax',    # Multi-class classification
    num_class=3,                  # 3 loại tấn công
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    device='cuda',
    eval_metric='mlogloss',
    early_stopping_rounds=20
)
```

Huấn luyện với sample weights:

```python
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight('balanced', y_train)

model_s2.fit(
    X_train, y_train,
    sample_weight=weights,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)
```

**Bước 6.6: Đánh giá Stage 2**

Metrics trên attack samples:

- Accuracy: 97.58%
- Precision (weighted): 97.91%
- Recall (weighted): 97.58%
- F1-Score (weighted): 97.64%

**Thời gian huấn luyện**: ~59 giây (với GPU T4)

### 2.3.9. Step 7: Combined Pipeline Evaluation

**Mục tiêu**: Đánh giá hiệu năng tổng thể của hệ thống hai giai đoạn.

**Quy trình kết hợp dự đoán**:

```
Algorithm: CombinedPrediction(X_test, model_s1, model_s2)

1: // Stage 1: Binary prediction
2: y_pred_binary = model_s1.predict(X_test)  // shape: (100k,)
3:
4: // Initialize final predictions
5: final_pred = []
6:
7: FOR i in range(len(X_test)):
8:   IF y_pred_binary[i] == 0:  // Predicted as Normal
9:     final_pred[i] = 'Normal'
10:  ELSE:  // Predicted as Attack
11:    // Stage 2: Attack type classification
12:    attack_type_idx = model_s2.predict(X_test[i])
13:    attack_name = reverse_mapping[attack_type_idx]
14:    final_pred[i] = attack_name
15:  END IF
16: END FOR
17:
18: RETURN final_pred
```

**Tối ưu hóa vectorized** (tăng tốc 1000x):

```python
# Thay vì loop, dùng vectorized operations
final_pred = np.where(y_pred_s1 == 0, 'Normal', None)
attack_mask = (y_pred_s1 == 1)
final_pred[attack_mask] = model_s2.predict(X_test[attack_mask])
```

**Kết quả Overall**:

- Overall Accuracy: 97.19%
- Confusion Matrix 4x4 (Normal, DDoS, DoS, Reconnaissance)

### 2.3.10. Step 8: Save models và metrics

**Artifacts được lưu**:

1. `stage1.pkl`: Binary classifier model
2. `stage2.pkl`: Multi-class classifier model
3. `encoders.pkl`: Label encoders cho categorical features
4. `mapping.pkl`: Attack type mapping dictionary
5. `features.pkl`: Danh sách 22 features
6. `training_metrics.json`: Metrics chi tiết (26KB, 953 dòng)

**Cấu trúc training_metrics.json**:

```json
{
  "metadata": {...},
  "training_data": {...},
  "test_data": {...},
  "stage1": {
    "accuracy": 0.99255,
    "training_history": {
      "train_loss": [0.5998, ..., 0.0001204],
      "val_loss": [0.5998, ..., 0.0001408],
      "num_iterations": 200
    },
    "training_time_sec": 30.57
  },
  "stage2": {
    "accuracy": 0.97578,
    "training_history": {...},
    "training_time_sec": 58.74
  },
  "overall": {
    "accuracy": 0.97192,
    "confusion_matrix": [[...], ...],
    "per_category_metrics": {...}
  }
}
```

## 2.4. Công cụ, công nghệ và nền tảng triển khai

### 2.4.1. Ngôn ngữ lập trình và thư viện

**Ngôn ngữ**: Python 3.10

**Thư viện chính**:

| Thư viện             | Phiên bản | Chức năng                                 |
| -------------------- | --------- | ----------------------------------------- |
| **pandas**           | 2.x       | Xử lý dữ liệu dạng bảng, đọc CSV          |
| **numpy**            | 1.24+     | Tính toán số học, xử lý mảng              |
| **xgboost**          | 3.0+      | Mô hình gradient boosting với GPU support |
| **scikit-learn**     | 1.3+      | Preprocessing, metrics, train-test split  |
| **imbalanced-learn** | 0.11+     | SMOTE và các kỹ thuật xử lý imbalance     |
| **scipy**            | 1.11+     | Entropy calculation, thống kê             |
| **joblib**           | 1.3+      | Serialize/deserialize models              |
| **psutil**           | 5.9+      | Monitor RAM usage                         |
| **matplotlib**       | 3.7+      | Visualization (training curves)           |
| **seaborn**          | 0.13+     | Statistical plots                         |

### 2.4.2. Môi trường huấn luyện

**Nền tảng**: Google Colab Pro+

**Cấu hình**:

- Runtime type: High-RAM
- RAM: 52 GB (actual usage: ~33 GB peak)
- GPU: Tesla T4 (16GB VRAM) / V100 (32GB VRAM) / A100 (40GB VRAM)
- Disk: 100+ GB (để lưu batch files)
- Session timeout: 12 hours

**Lý do chọn Colab**:

- Miễn phí (Colab Free) hoặc chi phí thấp (Colab Pro+ ~$50/tháng)
- Không cần đầu tư phần cứng
- GPU mạnh mẽ sẵn có
- Phù hợp với môi trường giáo dục và nghiên cứu

### 2.4.3. Pipeline công cụ

**Development & Version Control**:

- Git/GitHub: Version control và collaboration
- Jupyter Notebook: Prototyping và visualization
- VS Code: Code editing

**Data Processing**:

- Pandas: DataFrame operations (merge, groupby, aggregation)
- NumPy: Vectorized operations (100-1000x faster than loops)

**Machine Learning**:

- XGBoost 3.x: Model training với GPU acceleration
- SMOTE: Oversampling cho class imbalance
- LabelEncoder: Categorical encoding

**Evaluation**:

- scikit-learn metrics: accuracy, precision, recall, f1, confusion_matrix
- Custom scripts: Per-category analysis

**Deployment** (future):

- Docker: Containerization
- Flask/FastAPI: REST API server
- ONNX: Model export cho cross-platform inference

## 2.5. Phân tích độ phức tạp và đánh giá tối ưu hóa

### 2.5.1. Độ phức tạp về thời gian (Time Complexity)

**Step 1: Merge files**

- Độ phức tạp: O(N × M) với N = số files, M = số records/file
- Thực tế: 74 files × 500k records ≈ 37M operations
- Thời gian: ~10-15 phút (I/O bound)

**Step 2: Feature Engineering**

Source diversity calculation:

```
Độ phức tạp: O(N × log(G)) với N = số records, G = số groups
- Sorting để group: O(N log N)
- Aggregation: O(N)
- Merge back: O(N)
Total: O(N log N)
```

Thực tế với 20M records:

- Sorting: ~2-3 phút
- Aggregation: ~1 phút
- Total: ~5 phút

**Step 3: SMOTE**

```
Độ phức tạp: O(N_minority × k × d)
- N_minority: Số mẫu lớp thiểu số
- k: Số neighbors (thường k=5)
- d: Số dimensions (features)

Thực tế Stage 1:
O(7,000 × 5 × 22) ≈ 770,000 operations
Thời gian: ~10 giây
```

**Step 4: XGBoost Training**

```
Độ phức tạp cho 1 cây: O(N × D × K)
- N: Số samples
- D: Số features
- K: Số bins (histogram buckets, thường 256)

Với T cây (n_estimators=200):
Total: O(T × N × D × K)

Thực tế Stage 1 (CPU):
O(200 × 20M × 22 × 256) ≈ 2.25 × 10^13 operations
Thời gian CPU: ~2-3 giờ

Với GPU (parallelization):
Speedup: 100-200x
Thời gian GPU: ~30 giây ✅
```

**Step 5: Inference (Combined Pipeline)**

```
Độ phức tạp: O(N_test × T × D)
- N_test: Số test samples (100k)
- T: Số cây (200)
- D: Độ sâu trung bình (~6)

Với vectorization:
O(100k × 200 × 6) ≈ 120M operations
Thời gian: ~2 giây
```

### 2.5.2. Độ phức tạp về không gian (Space Complexity)

**Training Data Storage**:

```
RAM cho DataFrames:
- df_train: 20M rows × 35 columns × 8 bytes ≈ 5.6 GB
- df_test: 100k rows × 35 columns × 8 bytes ≈ 28 MB
- After feature engineering: ~6 GB
```

**SMOTE Overhead**:

```
Stage 1 SMOTE:
- Original: 17M × 22 features × 8 bytes ≈ 3 GB
- After SMOTE: 19M × 22 features × 8 bytes ≈ 3.35 GB
- Overhead: +350 MB
```

**XGBoost Model Size**:

```
Binary model (Stage 1):
- 200 trees × ~10KB/tree ≈ 2 MB

Multi-class model (Stage 2):
- 200 trees × 3 classes × ~10KB ≈ 6 MB

Total models: ~8 MB (rất nhỏ!)
```

**Peak RAM Usage**:

```
Breakdown:
- DataFrames: 6 GB
- SMOTE working memory: 3.5 GB
- XGBoost training buffers: 8 GB
- GPU VRAM: 5 GB (riêng biệt)
- OS + Python overhead: 3 GB
- Safety margin: 5 GB

Total: ~26 GB (an toàn trong 52GB limit)
Peak measured: 33 GB ✅
```

### 2.5.3. Tối ưu hóa hiệu năng

**Optimization 1: Batch Processing**

Trước:

```
Load 74 files individually → 74 disk I/O operations
Thời gian: ~30 phút
```

Sau:

```
Merge thành 8 batches → 8 disk I/O operations
Thời gian: ~5 phút (giảm 83%)
```

**Optimization 2: Vectorization**

Combined prediction trước (loop):

```python
for i in range(100000):
    if pred_s1[i] == 0:
        final[i] = 'Normal'
    else:
        final[i] = model_s2.predict(X_test[i:i+1])
```

Thời gian: ~15 phút

Sau (vectorized):

```python
attack_mask = (pred_s1 == 1)
final[attack_mask] = model_s2.predict(X_test[attack_mask])
```

Thời gian: ~2 giây (giảm 99.8%)

**Optimization 3: GPU Acceleration**

XGBoost training với CPU:

- Stage 1: ~2.5 giờ
- Stage 2: ~1.5 giờ
- Total: ~4 giờ

Với GPU T4:

- Stage 1: ~30 giây (300x speedup)
- Stage 2: ~59 giây (91x speedup)
- Total: ~1.5 phút (160x speedup overall)

**Optimization 4: Memory-efficient Loading**

```python
# Trước: Load toàn bộ vào RAM
df = pd.read_csv(file)  # 2.6 GB per batch

# Sau: Low memory mode + chunking nếu cần
df = pd.read_csv(file, low_memory=False)  # Tối ưu dtype
gc.collect()  # Giải phóng bộ nhớ ngay sau mỗi step
```

**Optimization 5: Early Stopping**

```python
XGBClassifier(
    n_estimators=200,  # Maximum
    early_stopping_rounds=20  # Stop nếu không cải thiện sau 20 iterations
)
```

Kết quả:

- Stage 1: Stopped at iteration 199 (train đầy đủ nhưng có thể stop sớm)
- Stage 2: Stopped at iteration 199
- Tiết kiệm: 0-30% thời gian huấn luyện tùy dataset

### 2.5.4. Đánh giá tổng thể về hiệu năng

**Training Time Breakdown** (với GPU T4):

```
1. Load data:              ~5 min
2. Feature engineering:    ~5 min
3. SMOTE Stage 1:          ~10 sec
4. Train Stage 1:          ~31 sec
5. SMOTE Stage 2:          ~skip
6. Train Stage 2:          ~59 sec
7. Evaluation:             ~10 sec
8. Save models:            ~5 sec

Total: ~12 minút ✅
Target: 15-20 phút (đạt được và vượt mục tiêu)
```

**Inference Time** (real-time deployment):

```
Single sample:
- Stage 1 prediction: ~0.5 ms
- Stage 2 prediction: ~0.5 ms (if attack)
- Total: ~1 ms per sample

Batch (1000 samples):
- Stage 1: ~10 ms
- Stage 2: ~10 ms
- Total: ~20 ms for 1000 samples
- Throughput: 50,000 samples/second ✅
```

**Accuracy vs Speed Trade-off**:

| Configuration        | Training Time | Accuracy  | Notes                       |
| -------------------- | ------------- | --------- | --------------------------- |
| n_estimators=50      | ~3 min        | 95.5%     | Nhanh nhưng kém chính xác   |
| n_estimators=100     | ~6 min        | 96.8%     | Cân bằng tốt                |
| **n_estimators=200** | **~12 min**   | **97.2%** | Lựa chọn tối ưu ✅          |
| n_estimators=500     | ~30 min       | 97.4%     | Tốn thời gian, cải thiện ít |

**Resource Utilization**:

- RAM: 33GB / 52GB (63% usage) ✅
- GPU VRAM: 6GB / 16GB (38% usage) ✅
- Disk: 20GB / 100GB (20% usage) ✅
- CPU: Minimal (GPU handles most computation)

### 2.5.5. Scalability Analysis

**Khả năng mở rộng với dữ liệu lớn hơn**:

Giả sử tăng dataset lên 100M records (5x):

```
Training time ước tính:
- Load data: 25 min
- Feature engineering: 25 min
- SMOTE: 50 sec
- Stage 1 training: ~2.5 min (GPU scales tốt)
- Stage 2 training: ~5 min
Total: ~58 min (vẫn chấp nhận được)

RAM requirement:
- DataFrames: 30 GB
- SMOTE overhead: 2 GB
- XGBoost buffers: 15 GB
- Peak: ~50 GB (vừa đủ trong 52GB limit)
```

**Khả năng mở rộng với nhiều features**:

Giả sử tăng features lên 50 (từ 22):

```
Training time:
- XGBoost: Tăng ~2.3x (tuyến tính với số features)
- Predicted: ~28 min

RAM:
- DataFrames: Tăng ~2.3x
- Predicted peak: ~55 GB (vượt quá limit!)
→ Cần giảm số batches hoặc nâng cấp RAM
```

**Kết luận về tối ưu hóa**:

Hệ thống đã được tối ưu hóa tốt với:

- ✅ Thời gian huấn luyện: 12 phút (mục tiêu 15-20 phút)
- ✅ RAM usage: 33GB (mục tiêu < 52GB)
- ✅ Accuracy: 97.19% (mục tiêu > 90%)
- ✅ Throughput: 50k samples/second (real-time capable)
- ✅ Scalability: Có thể mở rộng lên 5x dữ liệu với cùng phần cứng

---

**Kết thúc Chương 2**
