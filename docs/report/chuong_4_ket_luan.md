# CHƯƠNG 4: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 4.1. Tóm tắt kết quả nghiên cứu

Nghiên cứu đã thành công trong việc xây dựng hệ thống phát hiện và phân loại botnet IoT dựa trên mô hình học máy phân cấp hai giai đoạn (Two-Stage Hierarchical Model), đạt được các kết quả đáng chú ý sau:

### 4.1.1. Hiệu năng tổng thể

Hệ thống đạt **Overall Accuracy 97.19%** trên tập dữ liệu Bot-IoT với 100,000 samples cân bằng, vượt trội so với các phương pháp hiện có:

- Random Forest: 94.2% (+3%)
- SVM với RBF kernel: 91.5% (+5.7%)
- Single-stage XGBoost: 95.8% (+1.4%)
- Deep Learning CNN-LSTM: 96.5% (+0.7%)

**Stage 1 (Binary Classification)** đạt accuracy 99.26% và ROC-AUC 99.99%, cho thấy khả năng phân biệt lưu lượng tấn công và bình thường gần như hoàn hảo. **Stage 2 (Multi-class Classification)** đạt accuracy 97.58% trên các loại tấn công, với F1-score weighted 97.64%.

### 4.1.2. Hiệu năng theo từng loại tấn công

Hệ thống thể hiện khả năng phát hiện xuất sắc trên hầu hết các loại tấn công:

- **Normal**: Recall 99.95% (chỉ bỏ sót 1/2,000 samples)
- **DoS**: Recall 99.36%, Precision 99.84% (hiệu năng tốt nhất)
- **Reconnaissance**: Recall 99.12%, Precision 86.74%
- **DDoS**: Recall 93.22%, Precision 99.95% (điểm yếu cần cải thiện)

### 4.1.3. Hiệu năng tính toán

Hệ thống đạt được hiệu năng tính toán ấn tượng, phù hợp cho triển khai thực tế:

- **Training time**: 12 phút (với GPU T4) vs 4 giờ (CPU only) - tăng tốc 20x
- **Inference throughput**: 1.2 triệu samples/giây
- **Latency**: 0.82 microseconds/sample
- **Model size**: 8.4 MB (compact, dễ deploy)
- **RAM inference**: ~1 GB (thấp, phù hợp edge devices)

Khả năng xử lý này cho phép hệ thống scale lên 121,000 IoT devices với single instance, vượt xa yêu cầu thực tế.

## 4.2. Đóng góp chính của nghiên cứu

### 4.2.1. Đóng góp về mô hình

**Kiến trúc Two-Stage Hierarchical Model**

Nghiên cứu đề xuất kiến trúc phân cấp hai giai đoạn, trong đó:

- **Stage 1** tập trung vào binary classification (Attack vs Normal) để lọc ra lưu lượng bình thường
- **Stage 2** chuyên sâu vào multi-class classification (DDoS vs DoS vs Reconnaissance) chỉ trên các samples được xác định là tấn công

Kiến trúc này cho phép tối ưu hóa riêng biệt cho từng bài toán con, khai thác đặc thù của từng giai đoạn, và đạt accuracy cao hơn 1.4% so với single-stage XGBoost.

**Tích hợp XGBoost với GPU Acceleration**

Sử dụng XGBoost 3.x với `tree_method='hist'` và `device='cuda'` để tận dụng tối đa GPU, đạt training time chỉ 12 phút so với 4 giờ khi sử dụng CPU, cho phép rapid experimentation và dễ dàng retrain model khi có dữ liệu mới.

### 4.2.2. Đóng góp về đặc trưng

**Source Diversity Features**

Nghiên cứu thiết kế 3 đặc trưng mới dựa trên phân tích đặc tính nguồn tấn công:

1. **unique_src_count**: Số lượng địa chỉ IP nguồn duy nhất trong time window
2. **src_entropy**: Entropy của phân phối địa chỉ nguồn (Shannon entropy)
3. **top_src_ratio**: Tỷ lệ của địa chỉ nguồn xuất hiện nhiều nhất

Các features này giúp phân biệt hiệu quả giữa:

- **DDoS attacks**: Nhiều nguồn → high unique_src_count, high entropy, low top_src_ratio
- **DoS attacks**: Ít nguồn → low unique_src_count, low entropy, high top_src_ratio

Ablation study chứng minh đóng góp đáng kể: bỏ source diversity features làm DDoS recall giảm 4.72% (từ 93.22% → 88.5%).

### 4.2.3. Đóng góp về xử lý mất cân bằng dữ liệu

**Extreme Imbalance Handling (2000:1)**

Nghiên cứu áp dụng kết hợp hai kỹ thuật để xử lý mất cân bằng nghiêm trọng (imbalance ratio 2000:1):

1. **SMOTE (Synthetic Minority Over-sampling Technique)**: Tạo synthetic samples cho lớp thiểu số (Normal, Reconnaissance), tăng số lượng từ 7,769 lên ~2 triệu samples
2. **scale_pos_weight trong XGBoost**: Cân bằng trọng số trong hàm loss để tăng ảnh hưởng của lớp thiểu số

Kết quả:

- Normal recall đạt 99.95% (chỉ miss 1/2000)
- Reconnaissance recall đạt 99.12% dù chỉ chiếm 9% training data
- Không bị bias vào lớp đa số như các phương pháp truyền thống

### 4.2.4. Đóng góp về quy trình và triển khai

**Scalable Data Processing Pipeline**

Thiết kế quy trình xử lý hiệu quả cho Big Data (74 files CSV, 16GB, 40-50M records):

- **Batch merging strategy**: Gộp 74 files thành 8 batches để giảm I/O overhead
- **Balanced test set creation**: Tạo tập test cân bằng để đánh giá công bằng
- **Memory optimization**: Sử dụng batch loading và garbage collection để control RAM usage (~33GB peak)

**Production-Ready Architecture**

Hệ thống được thiết kế với các đặc điểm phù hợp triển khai thực tế:

- Model size nhỏ gọn (8.4 MB)
- RAM inference thấp (~1 GB)
- Throughput cao (1.2M samples/sec)
- Dễ dàng containerize (Docker) và orchestrate (Kubernetes)

### 4.2.5. Đóng góp về nghiên cứu và giáo dục

**Open Source và Reproducibility**

- Code được tổ chức rõ ràng, có chú thích chi tiết
- Sử dụng nền tảng giáo dục (Google Colab Pro+) với chi phí hợp lý (~$50/tháng)
- Quy trình có thể tái sử dụng cho các datasets IoT khác
- Cung cấp baseline cho các nghiên cứu tiếp theo về IoT botnet detection

**Hướng dẫn thực tiễn**

Nghiên cứu cung cấp kinh nghiệm thực tế về:

- Xử lý imbalanced data nghiêm trọng (2000:1)
- Tối ưu GPU cho XGBoost 3.x
- Feature engineering cho network traffic analysis
- Trade-offs giữa accuracy, speed, và memory

## 4.3. Hạn chế và tồn tại

### 4.3.1. Hạn chế về hiệu năng

**DDoS Detection Performance**

Mặc dù Overall Accuracy đạt 97.19%, DDoS recall chỉ đạt 93.22%, thấp hơn so với các loại tấn công khác:

- Bỏ sót 2,372 DDoS samples trong 35,000 samples (6.78%)
- 513 samples bị Stage 1 phân loại nhầm là Normal
- 1,830 samples bị Stage 2 phân loại nhầm là Reconnaissance

**Nguyên nhân**:

- DDoS có đặc điểm đa dạng, một số "low-rate DDoS" giống Normal traffic
- Khó phân biệt DDoS và Reconnaissance do cả hai đều có "nhiều nguồn"
- Thiếu temporal features để capture time-series patterns

**Tác động**: Trong môi trường production, 6.78% DDoS bị bỏ sót có thể gây rủi ro bảo mật nghiêm trọng.

**Theft Category bị loại bỏ**

Loại tấn công Theft (data exfiltration) chỉ có 1,587 samples trong dataset (0.003%), quá ít để huấn luyện model hiệu quả:

- SMOTE không hiệu quả với dataset quá nhỏ (<5,000 samples)
- Hệ thống không thể phát hiện Theft attacks
- Đây là loại tấn công nguy hiểm (đánh cắp dữ liệu nhạy cảm)

### 4.3.2. Hạn chế về tài nguyên

**Memory Requirement cao**

Training cần 33GB RAM, cao nhất trong các phương pháp so sánh:

- Không phù hợp với consumer laptops (thường 8-16GB)
- Yêu cầu cloud instances với High-RAM (Colab Pro+, AWS r5.xlarge)
- Chi phí cao hơn nếu muốn train locally

**Nguyên nhân**:

- Dữ liệu lớn: 20M records × 35 features × 8 bytes ≈ 5.6 GB
- SMOTE tạo thêm 2M samples → +1.5 GB
- XGBoost training buffers: ~8 GB
- Python overhead và safety margin: ~8 GB

**Storage cho Dataset**

Dataset gốc 16GB (74 files) đòi hỏi storage đáng kể, không phù hợp cho môi trường resource-constrained.

### 4.3.3. Hạn chế về triển khai

**Chưa có Real-time Deployment**

Hệ thống mới được đánh giá offline trên tập test tĩnh:

- Chưa test với live network traffic
- Chưa có infrastructure để deploy (API server, monitoring pipeline)
- Chưa implement feedback loop để retrain model
- Chưa đánh giá performance degradation over time (concept drift)

**Chưa validate trên datasets khác**

Nghiên cứu chỉ đánh giá trên Bot-IoT dataset:

- Chưa cross-validate trên CICIDS, NSL-KDD, UNSW-NB15
- Không biết model có generalize tốt cho traffic từ môi trường IoT khác không
- Chưa so sánh với commercial IDS solutions (Snort, Suricata, Zeek)

### 4.3.4. Hạn chế về phương pháp

**Thiếu Temporal Analysis**

Model chỉ sử dụng features tĩnh từ individual flows:

- Không capture time-series patterns (sequence of flows over time)
- Không phát hiện được "slow attacks" (low-and-slow DDoS)
- Thiếu context về network behavior evolution

**No Ensemble Approach**

Chỉ sử dụng XGBoost single model:

- Chưa kết hợp với Deep Learning models (CNN-LSTM) để improve DDoS detection
- Chưa implement voting mechanism từ multiple models
- Có thể cải thiện accuracy thêm 0.5-1% với ensemble

## 4.4. Hướng phát triển tương lai

### 4.4.1. Cải thiện DDoS Detection

**Temporal Feature Engineering**

Thêm time-series features để capture attack evolution:

- Flow rate over time (packets/second trong sliding windows)
- Inter-arrival time statistics (mean, std, min, max)
- Burstiness metrics (coefficient of variation)
- Session duration distribution

**Ensemble với Deep Learning**

Kết hợp XGBoost với CNN-LSTM để leverage cả statistical và temporal patterns:

- **XGBoost**: Học statistical features (mean, std, entropy)
- **CNN-LSTM**: Học temporal dependencies (sequential patterns)
- **Voting mechanism**: Weighted average hoặc stacking

**Ước lượng cải thiện**: DDoS recall từ 93.22% → 96-97% (+3-4%)

### 4.4.2. Real-time Deployment

**API Development**

Xây dựng REST API với Flask hoặc FastAPI:

- Endpoint `/predict` để nhận network flows và trả về predictions
- Endpoint `/retrain` để trigger incremental learning
- Endpoint `/metrics` để monitoring performance
- Rate limiting và authentication để security

**Edge Deployment**

Deploy model trên IoT gateway devices:

- **Raspberry Pi 4**: 8GB RAM, ARM CPU (cần model optimization)
- **NVIDIA Jetson Nano**: 4GB RAM, GPU 128-core (phù hợp XGBoost)
- **Industrial gateways**: Cisco IR829, Advantech ARK series

**Monitoring và Alerting**

Implement real-time monitoring system:

- Prometheus + Grafana để visualize metrics (accuracy, latency, throughput)
- Alerting rules khi phát hiện attacks hoặc performance degradation
- Logging để audit và forensic analysis

**Continuous Learning Pipeline**

Xây dựng pipeline để retrain model định kỳ:

- Collect labeled data từ Security Operations Center (SOC)
- Incremental training hàng tuần/hàng tháng
- A/B testing để validate model mới trước khi deploy
- Rollback mechanism nếu performance giảm

### 4.4.3. Mở rộng Dataset và Cross-validation

**Thu thập dữ liệu Theft**

Giải quyết vấn đề thiếu Theft samples:

- Collaborate với security companies để thu thập data exfiltration logs
- Generate synthetic Theft samples từ normal data + adversarial perturbations
- Transfer learning từ models trained trên datasets khác (CICIDS có Infiltration class)

**Cross-dataset Validation**

Validate model trên multiple IoT datasets:

- **CICIDS 2017/2018**: General network traffic (not IoT-specific)
- **NSL-KDD**: Benchmark IDS dataset
- **UNSW-NB15**: Modern network traffic với các attacks mới
- **N-BaIoT**: IoT botnet traffic từ 9 commercial devices

**Online Learning**

Implement online learning để adapt với attacks mới:

- Incremental updates khi có labeled data mới
- Warm-start training từ pretrained model
- Adaptive learning rate để balance stability và plasticity

### 4.4.4. Tối ưu hóa Resource

**Model Compression**

Giảm memory và storage requirements:

- **Quantization**: Convert model từ float32 → int8 (giảm 4x size)
- **Pruning**: Remove low-importance features và trees
- **Knowledge Distillation**: Train smaller "student model" từ large "teacher model"

**Ước lượng**: Giảm model size từ 8.4 MB → 2-3 MB, RAM inference từ 1 GB → 256 MB

**Incremental Learning**

Giảm training memory requirement:

- Chia 20M records thành mini-batches (1M records/batch)
- Train iteratively với warm-start
- Gradient accumulation để simulate large batch size

**Ước lượng**: Giảm training RAM từ 33 GB → 10-12 GB

**Feature Selection**

Sử dụng Feature Importance từ XGBoost để chỉ giữ lại top features:

- Hiện tại: 22 features
- Giảm xuống: 12-15 features (chỉ giữ top contributors)
- Trade-off: Accuracy giảm ~0.5% nhưng RAM giảm 30-40%

**Batched Inference**

Tối ưu inference với vectorization và batching:

- Process multiple samples cùng lúc (batch size 1000-10000)
- Leverage CPU/GPU parallelism
- Reduce per-sample overhead

### 4.4.5. Tích hợp Defense Mechanisms

**Active Response**

Không chỉ detect mà còn tự động respond:

- **Auto-blocking**: Tự động thêm IP vào firewall blacklist khi phát hiện attack
- **Rate limiting**: Throttle traffic từ suspicious sources
- **Traffic shaping**: Prioritize legitimate traffic, drop attack packets
- **Quarantine**: Isolate infected IoT devices khỏi network

**Integration với Security Infrastructure**

Tích hợp với các công cụ security hiện có:

- **Firewall**: iptables, pfSense, Cisco ASA
- **IPS**: Snort, Suricata, Zeek
- **SIEM**: Splunk, ELK Stack, QRadar
- **Orchestration**: SOAR platforms (Phantom, Demisto)

**Collaborative Defense**

Multiple gateways chia sẻ threat intelligence:

- **Federated Learning**: Train model trên distributed data mà không share raw data
- **Threat Sharing**: Gateways report attack IPs/patterns lên central server
- **Consensus Mechanism**: Multiple gateways vote để confirm attacks
- **Honeypot Integration**: Deploy honeypots để collect attack samples

**Adaptive Defense**

Tự động adjust defense strategy dựa trên attack patterns:

- Tăng detection threshold khi có attack spike
- Switch sang aggressive mode khi phát hiện coordinated attack
- Gradual relaxation khi attack subsides

## 4.5. Kết luận

Nghiên cứu đã thành công trong việc xây dựng hệ thống phát hiện và phân loại botnet IoT dựa trên mô hình Two-Stage Hierarchical Model, đạt Overall Accuracy 97.19%, vượt trội so với các phương pháp hiện tại. Các đóng góp chính bao gồm:

1. **Kiến trúc phân cấp hai giai đoạn** cho phép tối ưu riêng Binary và Multi-class classification, đạt accuracy cao hơn 1.4% so với single-stage
2. **Source Diversity Features** giúp phân biệt hiệu quả DDoS và DoS, cải thiện recall +4.72%
3. **Xử lý extreme imbalance (2000:1)** với kết hợp SMOTE và scale_pos_weight, đạt Normal recall 99.95%
4. **GPU acceleration** giảm training time 20x (12 phút vs 4 giờ), cho phép rapid development
5. **Scalable pipeline** với throughput 1.2M samples/sec, phù hợp triển khai thực tế

Mặc dù còn tồn tại một số hạn chế như DDoS recall chỉ 93.22%, memory requirement cao (33GB), và chưa validate trên datasets khác, nghiên cứu đã đặt nền móng vững chắc cho các hướng phát triển tiếp theo. Với 5 hướng phát triển được đề xuất (temporal features, real-time deployment, dataset expansion, resource optimization, defense integration), hệ thống có tiềm năng trở thành giải pháp toàn diện cho bảo mật IoT trong tương lai.

Kết quả nghiên cứu không chỉ đạt được mục tiêu ban đầu về độ chính xác và hiệu năng, mà còn đóng góp kiến thức và kinh nghiệm thực tiễn cho cộng đồng nghiên cứu về IoT security và machine learning. Hy vọng rằng nghiên cứu này sẽ là bước đệm cho các công trình tiếp theo trong lĩnh vực phát hiện và phòng chống botnet IoT.

---

**Kết thúc Báo cáo**
