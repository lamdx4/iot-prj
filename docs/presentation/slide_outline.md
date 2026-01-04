# Slide Thuyết Trình: Hệ Thống Phát Hiện Botnet IoT Ba Giai Đoạn

## Thời lượng: 20-30 phút (~28 slides)

---

## PHẦN 1: GIỚI THIỆU (4 slides, 3-4 phút)

### Slide 1: Trang bìa

- **Tiêu đề:** Hệ Thống Phát Hiện và Phân Loại Botnet IoT Sử Dụng Mô Hình Phân Cấp Ba Giai Đoạn
- **Họ tên:** [Tên sinh viên]
- **GVHD:** [Tên giảng viên]
- **Ngày:** [Ngày báo cáo]

### Slide 2: Bối cảnh nghiên cứu

**Tiêu đề:** Thực Trạng An Ninh IoT

**Nội dung:**

- 📊 **Quy mô IoT:** 15.14 tỷ thiết bị (2023) → dự kiến 29 tỷ (2030)
- ⚠️ **Vấn đề:** Thiết bị IoT thiếu bảo mật → mục tiêu botnet
- 🎯 **Botnet IoT nổi tiếng:** Mirai (2016), Hajime, Hide and Seek
- 💥 **Tác động:** DDoS attack quy mô lớn (Dyn DNS 2016)

**Hình ảnh:** Biểu đồ tăng trưởng thiết bị IoT

### Slide 3: Vấn đề nghiên cứu

**Tiêu đề:** Thách Thức Trong Phát Hiện Botnet IoT

**Các thách thức:**

1. ⚖️ **Dữ liệu mất cân bằng nghiêm trọng**
   - Tỷ lệ Normal:Attack ≈ 1:7687
2. 🔍 **Phân loại đa lớp phức tạp**
   - Cần phân biệt: DDoS, DoS, Reconnaissance
   - Cần nhận diện biến thể DDoS: HTTP, TCP, UDP
3. 🚀 **Yêu cầu hiệu năng**
   - Training time: Xử lý 20M+ samples
   - Accuracy: >95% cho môi trường production

### Slide 4: Mục tiêu nghiên cứu

**Tiêu đề:** Mục Tiêu và Đóng Góp

**Mục tiêu:**
✅ Xây dựng mô hình IDS phân cấp 3 giai đoạn  
✅ Đạt accuracy >95% với training time <15 phút  
✅ Xử lý imbalance hiệu quả (SMOTE + source diversity features)  
✅ Phân loại chi tiết DDoS variants (HTTP/TCP/UDP)

**Đóng góp chính:**

- Kiến trúc 3-stage hierarchical mới
- Source diversity features để phân biệt DDoS vs DoS
- Balanced test set cho evaluation công bằng

---

## PHẦN 2: GIẢI PHÁP ĐỀ XUẤT (8 slides, 10-12 phút)

### Slide 5: Tổng quan giải pháp

**Tiêu đề:** Kiến Trúc Ba Giai Đoạn

**Hình ảnh:** Mermaid diagram (system_overview.png)

**Mô tả ngắn:**

- **Stage 1:** Binary (Attack vs Normal)
- **Stage 2:** Multi-class (DDoS vs DoS vs Recon)
- **Stage 3:** DDoS Variants (HTTP vs TCP vs UDP)

### Slide 6: Dataset - Bot-IoT

**Tiêu đề:** Tập Dữ Liệu Bot-IoT

**Thông tin:**

- 📦 **Quy mô:** 73.4M records, 16 GB CSV
- 🏷️ **Categories:** Normal, DDoS, DoS, Reconnaissance, Theft
- 🔬 **Môi trường:** Lab testbed với Ostinato traffic generator
- ⚖️ **Imbalance:** Normal:Attack = 1:7687

**Phân bố:**
| Category | Records | Percentage |
|----------|---------|------------|
| DoS | 33M | 45% |
| DDoS | 38.5M | 52.5% |
| Recon | 1.8M | 2.5% |
| Normal | 9.5K | 0.01% |

### Slide 7: Stage 1 - Binary Classification

**Tiêu đề:** Giai Đoạn 1: Phân Loại Nhị Phân

**Input:** 22 features (flow-level statistics)  
**Output:** Attack (1) hoặc Normal (0)  
**Model:** XGBoost Binary Classifier

**Kỹ thuật xử lý:**

- ✅ **SMOTE:** Tăng Normal từ 7.7K → 2M samples (10% của Attack)
- ✅ **scale_pos_weight:** Cân bằng loss function
- ✅ **GPU acceleration:** Tesla T4 (Colab Pro+)

**Kết quả:**

- **Accuracy:** 99.26%
- **ROC-AUC:** 99.99%
- **Train time:** 52s

### Slide 8: Stage 2 - Multi-class Classification

**Tiêu đề:** Giai Đoạn 2: Phân Loại Đa Lớp

**Input:** Attack samples từ Stage 1  
**Output:** DDoS (0), DoS (1), Reconnaissance (2)  
**Model:** XGBoost Multi-class

**Kỹ thuật đặc biệt:**
🔑 **Source Diversity Features** (3 features mới):

- `unique_src_count`: Số nguồn unique trong time window
- `src_entropy`: Phân phối nguồn tấn công
- `top_src_ratio`: Tỷ lệ nguồn chiếm ưu thế

**Insight:**

- DDoS: nhiều nguồn (high entropy, low top_src_ratio)
- DoS: ít nguồn (low entropy, high top_src_ratio)

**Kết quả:**

- **Accuracy:** 97.58%
- **Train time:** 37s

### Slide 9: Source Diversity Features

**Tiêu đề:** Chi Tiết Source Diversity Features

**Algorithm:**

```
FOR each (time_window, target_IP):
  1. unique_src_count = COUNT(DISTINCT source_IPs)
  2. src_entropy = -Σ(p_i × log₂(p_i))
  3. top_src_ratio = max_count / total_count

  BROADCAST features to all flows in this group
```

**Ví dụ:**
| Attack Type | unique_src_count | src_entropy | top_src_ratio |
|-------------|------------------|-------------|---------------|
| DDoS | 1000+ | >8.0 | <0.1 |
| DoS | 1-5 | <2.0 | >0.8 |

### Slide 10: Stage 3 - DDoS Variant Classification

**Tiêu đề:** Giai Đoạn 3: Phân Loại Biến Thể DDoS

**Động lực:**

- Các biến thể DDoS khác nhau cần biện pháp phòng thủ riêng
- DDoS-HTTP: Rate limiting
- DDoS-TCP: SYN cookies
- DDoS-UDP: Packet filtering

**Input:** DDoS samples từ Stage 2  
**Output:** HTTP (0), TCP (1), UDP (2), Normal (3)  
**Dataset:** Random Consecutive Sampling → 7,635 samples/class

**Kết quả:**

- **Accuracy:** 97.3%
- **Macro F1-Score:** 97%

### Slide 11: Quy trình xử lý dữ liệu

**Tiêu đề:** Pipeline Xử Lý Dữ Liệu

**Step 1:** Tạo Balanced Test Set

- Sampling từ 8 batches → 100K samples
- Distribution: DoS 50%, DDoS 35%, Recon 13%, Normal 2%

**Step 2:** Load Training Data

- batch_01 + batch_04 = 20M records
- Chọn batch có đa dạng attack types

**Step 3:** Feature Engineering

- Source diversity calculation
- Drop temporal features (avoid leakage)
- LabelEncoder (fit on train only)

### Slide 12: Công nghệ triển khai

**Tiêu đề:** Môi Trường và Công Nghệ

**Hardware:**

- **Platform:** Google Colab Pro+
- **CPU:** Intel Xeon @ 2.0-2.3GHz (2 cores)
- **RAM:** 52 GB (peak usage: 33GB)
- **GPU:** Tesla T4 (16GB VRAM)

**Software:**

- Python 3.10
- XGBoost 3.0.5 (GPU support)
- Pandas, NumPy, Scikit-learn
- SMOTE (imbalanced-learn)

**Tổng thời gian training:** 12 phút (89s Stage 1 + 37s Stage 2)

---

## PHẦN 3: KẾT QUẢ THỰC NGHIỆM (10 slides, 10-12 phút)

### Slide 13: Kết quả tổng thể

**Tiêu đề:** Hiệu Năng Tổng Thể

**Overall Pipeline:**

- ✅ **Accuracy:** 97.19%
- ✅ **Precision:** 97.35%
- ✅ **Recall:** 97.19%
- ✅ **F1-Score:** 97.17%

**Breakdown theo stage:**
| Stage | Accuracy | Train Time |
|-------|----------|------------|
| Stage 1 | 99.26% | 52s |
| Stage 2 | 97.58% | 37s |
| **Overall** | **97.19%** | **89s** |

### Slide 14: Confusion Matrix

**Tiêu đề:** Ma Trận Nhầm Lẫn (Overall)

**Hình ảnh:** Confusion matrix (`figures/01_confusion_matrix.png`)

**Phân tích:**

- ✅ Normal: Recall 99.95% (xuất sắc)
- ✅ DoS: Recall 99.36% (rất tốt)
- ✅ Recon: Recall 99.12% (tốt)
- ⚠️ DDoS: Recall 93.22% (điểm yếu, bị nhầm với Recon/Normal)

### Slide 15: Metrics chi tiết theo category

**Tiêu đề:** Đánh Giá Chi Tiết Từng Loại

| Category    | Precision  | Recall     | F1-Score   | Accuracy   |
| ----------- | ---------- | ---------- | ---------- | ---------- |
| Normal      | 72.88%     | 99.95%     | 84.29%     | 99.95%     |
| DDoS        | 99.95%     | 93.22%     | 96.47%     | 93.22%     |
| DoS         | 99.84%     | 99.36%     | 99.60%     | 99.36%     |
| Recon       | 86.74%     | 99.12%     | 92.52%     | 99.12%     |
| **Average** | **97.35%** | **97.19%** | **97.17%** | **97.19%** |

**Hình ảnh:** Bar chart (`figures/03_per_category_metrics.png`)

### Slide 16: Training curves

**Tiêu đề:** Quá Trình Huấn Luyện

**Hình ảnh:** Combined loss curves (`figures/combined_loss_curves.png`)

**Quan sát:**

- Loss giảm nhanh trong 50 iterations đầu
- Converge ổn định sau iteration 100
- Không có dấu hiệu overfitting
- Validation loss sát với training loss

### Slide 17: Hiệu năng tính toán

**Tiêu đề:** Hiệu Năng Thời Gian Thực

**Throughput:**

- Stage 1: 1,017,595 samples/sec
- Stage 2: 2,908,013 samples/sec
- Overall: 1,217,890 samples/sec

**Latency:**

- Stage 1: 0.98 μs/sample
- Stage 2: 0.34 μs/sample
- Overall: 0.82 μs/sample

**Model Size:**

- Stage 1: 2.1 MB
- Stage 2: 6.3 MB
- Total: 8.4 MB (phù hợp edge deployment)

### Slide 18: So sánh với các phương pháp khác

**Tiêu đề:** So Sánh Với State-of-the-Art

| Method               | Accuracy   | Train Time | Memory    |
| -------------------- | ---------- | ---------- | --------- |
| Baseline (No ML)     | 89.3%      | -          | -         |
| Random Forest        | 94.2%      | 45 min     | 8 GB      |
| SVM (RBF)            | 91.5%      | 180 min    | 16 GB     |
| Single XGBoost       | 95.8%      | 25 min     | 12 GB     |
| Deep Learning        | 96.5%      | 120 min    | 20 GB     |
| **Two-Stage (Ours)** | **97.19%** | **12 min** | **33 GB** |

**Ưu điểm:**
✅ Accuracy cao nhất  
✅ Train time nhanh nhất (có GPU)  
⚠️ Memory cao (trade-off cho accuracy)

### Slide 19: Kết quả Stage 3 - DDoS Variants

**Tiêu đề:** Phân Loại Biến Thể DDoS

**Dataset:** 7,635 samples/class (balanced)

**Kết quả chi tiết:**
| Variant | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Normal | 96.8% | 98.1% | 97.4% |
| DDoS-HTTP | 97.5% | 96.2% | 96.8% |
| DDoS-TCP | 98.2% | 97.9% | 98.0% |
| DDoS-UDP | 96.7% | 97.1% | 96.9% |
| **Macro Avg** | **97.3%** | **97.3%** | **97.3%** |

**Hình ảnh:** Class distribution + Confusion matrix

### Slide 20: Ablation Study - Balanced Sampling

**Tiêu đề:** Ablation Study: Full Data vs Balanced Sampling

**Câu hỏi:** Có cần 20M samples hay downsample 1:1:1:1 (31K) là đủ?

**Kết quả thí nghiệm:**
| Approach | Data Size | Stage 1 Acc | Stage 2 Acc | Overall |
|----------|-----------|-------------|-------------|---------|
| **Full Data** | 20M → 19M | 99.26% | 97.58% | **97.19%** |
| **Balanced 1:1:1:1** | 31K | 99.61% | **13.27%** | **~13%** |

**Kết luận:**
❌ Balanced sampling thất bại nghiêm trọng ở Stage 2  
✅ Full data approach justified  
💡 Data volume critical cho multi-class classification

### Slide 21: Ablation Study - Source Diversity

**Tiêu đề:** Đóng Góp Của Source Diversity Features

**Thí nghiệm:** Bỏ 3 source diversity features

**Kết quả:**

- DDoS Recall: 93.22% → 88.5% (**-4.72%**)
- Stage 2 Accuracy: 97.58% → 92.86% (**-4.72%**)

**Kết luận:**
✅ Source diversity features **quan trọng** cho phân biệt DDoS vs DoS  
✅ Đóng góp đáng kể vào hiệu năng tổng thể

### Slide 22: Visualization - Feature Correlation

**Tiêu đề:** Tương Quan Các Đặc Trưng

**Hình ảnh:** Correlation heatmap cho Stage 3

**Quan sát:**

- Source diversity features có correlation thấp với basic features
- `unique_src_count` và `src_entropy` correlation cao (redundant signal)
- Protocol-related features quan trọng cho phân loại DDoS variants

---

## PHẦN 4: KẾT LUẬN (6 slides, 5-6 phút)

### Slide 23: Đóng góp chính

**Tiêu đề:** Đóng Góp Của Nghiên Cứu

**1. Kiến trúc mới:**

- ✅ Three-Stage Hierarchical Model
- ✅ Specialized classifier cho từng level
- ✅ DDoS variant classification (HTTP/TCP/UDP)

**2. Kỹ thuật mới:**

- ✅ Source Diversity Features (3 features)
- ✅ Balanced test set creation strategy
- ✅ Random Consecutive Sampling cho Stage 3

**3. Kết quả:**

- ✅ 97.19% accuracy (4-class)
- ✅ 97.3% accuracy (DDoS variants)
- ✅ 12 phút training time (20M samples)

### Slide 24: Hạn chế

**Tiêu đề:** Hạn Chế Của Nghiên Cứu

**1. Dữ liệu:**

- ⚠️ Chỉ test trên Bot-IoT dataset (lab environment)
- ⚠️ Chưa validate trên real-world traffic

**2. Tài nguyên:**

- ⚠️ Yêu cầu RAM cao (33GB) → khó deploy trên edge
- ⚠️ Cần GPU để training nhanh

**3. Phân loại:**

- ⚠️ DDoS recall thấp hơn (93.22% vs 99%+ cho DoS/Recon)
- ⚠️ Stage 3 chỉ có 7,635 samples/class → cần more data

**4. Thiếu ablation:**

- ⚠️ Chưa so sánh stratified sampling vs random sampling
- ⚠️ Chưa test với các dataset khác (UNSW-NB15, CIC-IDS2017)

### Slide 25: Hướng phát triển

**Tiêu đề:** Hướng Nghiên Cứu Tiếp Theo

**1. Cải thiện model:**

- 🔬 Federated Learning cho distributed IoT deployment
- 🔬 Online learning để adapt với new attack patterns
- 🔬 Ensemble với Deep Learning (CNN-LSTM)

**2. Tối ưu hóa:**

- 🔬 Model compression cho edge devices
- 🔬 Quantization (FP32 → INT8)
- 🔬 Knowledge distillation

**3. Mở rộng:**

- 🔬 Real-time detection system
- 🔬 Integration với SDN/NFV
- 🔬 Multi-dataset validation

**4. Security:**

- 🔬 Adversarial robustness
- 🔬 Explainable AI (SHAP, LIME)

### Slide 26: Ứng dụng thực tế

**Tiêu đề:** Khả Năng Ứng Dụng

**1. IoT Gateway:**

- Deploy mô hình tại gateway (Edge AI)
- Real-time threat detection
- Block malicious traffic

**2. Network Security Monitoring:**

- Integration với SIEM systems
- Forensics analysis (DDoS variant identification)
- Threat intelligence

**3. SOC (Security Operations Center):**

- Automated incident response
- Alert prioritization
- Attack classification dashboard

### Slide 27: Demo (Optional)

**Tiêu đề:** Demo Hệ Thống

**Option 1:** Screenshot/Video của:

- Balanced test set distribution
- Training process
- Confusion matrix visualization
- Real-time prediction example

**Option 2:** Live demo (nếu có thời gian):

- Load pre-trained model
- Predict trên sample flows
- Show classification results

### Slide 28: Kết luận

**Tiêu đề:** Tổng Kết

**Vấn đề giải quyết:**
✅ Phát hiện botnet IoT với accuracy cao (97.19%)  
✅ Xử lý imbalance hiệu quả (SMOTE + source diversity)  
✅ Phân loại chi tiết DDoS variants (97.3%)

**Đóng góp chính:**
✅ Three-Stage Hierarchical Architecture  
✅ Source Diversity Features  
✅ Comprehensive evaluation với balanced test set

**Thành tựu:**

- **Accuracy:** 97.19% (4-class), 97.3% (DDoS variants)
- **Speed:** 12 phút training (20M samples)
- **Efficiency:** 1.2M samples/sec throughput

**Cảm ơn và Q&A!**

---

## PHỤ LỤC: Backup Slides (không present, dùng cho Q&A)

### Backup 1: XGBoost Algorithm Details

**Objective function:**

```
L(θ) = Σ l(y_i, ŷ_i) + Σ Ω(f_k)
```

**Regularization:**

```
Ω(f) = γT + (1/2)λΣw_j²
```

### Backup 2: SMOTE Algorithm

**Pseudocode:**

```
FOR each minority sample x_i:
  Find k nearest neighbors
  FOR j = 1 to N/100:
    Select random neighbor x_n
    Create: x_new = x_i + rand(0,1) × (x_n - x_i)
```

### Backup 3: Hardware Specs Detailed

- **CPU:** Intel Xeon E5-2670 v2 @ 2.0-2.3GHz
- **GPU:** Tesla T4 (Turing, 2560 CUDA cores)
- **Memory:** DDR4 52GB @ 2133 MHz
- **Storage:** 100GB SSD (Google Drive mount)

### Backup 4: Related Works Comparison

Detailed comparison với specific papers:

- Koroniotis et al. (2019) - Bot-IoT creators
- Zhang et al. (2021) - Two-Stage IDS
- Özdoğan et al. (2023) - XGBoost optimization

---

## NOTES CHO NGƯỜI THUYẾT TRÌNH

**Timing guide (30 phút):**

- Giới thiệu: 3-4 phút (slides 1-4)
- Giải pháp: 10-12 phút (slides 5-12)
- Kết quả: 10-12 phút (slides 13-22)
- Kết luận: 5-6 phút (slides 23-28)
- Q&A: Dự trữ thời gian

**Tips:**

1. **Slide 8-9:** Giải thích kỹ source diversity (đây là novelty chính)
2. **Slide 14-15:** Emphasize DDoS recall issue và solution (Stage 3)
3. **Slide 20:** Ablation study rất quan trọng để justify approach
4. **Slide 18:** So sánh với methods khác, highlight trade-offs

**Câu hỏi thường gặp (chuẩn bị):**

- Q: Tại sao 3 stages thay vì 1 model multi-class?
- Q: Source diversity features có phải là novelty?
- Q: RAM 33GB có quá cao cho deployment?
- Q: Kết quả trên real-world traffic như thế nào?
- Q: DDoS recall 93.22% có thấp so với DoS 99.36%?
