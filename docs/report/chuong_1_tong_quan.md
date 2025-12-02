# CHƯƠNG 1: TỔNG QUAN

## 1.1. Bối cảnh và vấn đề nghiên cứu

### 1.1.1. Thực trạng an ninh mạng trong hệ sinh thái IoT

Trong những năm gần đây, Internet of Things (IoT) đã trở thành một phần không thể thiếu trong cuộc sống hiện đại, từ các thiết bị gia dụng thông minh, hệ thống y tế, đến các ứng dụng công nghiệp quy mô lớn. Theo dự báo của Statista, số lượng thiết bị IoT toàn cầu dự kiến đạt 75 tỷ vào năm 2025, tạo ra một hệ sinh thái số hóa rộng lớn nhưng cũng đi kèm với những thách thức lớn về bảo mật.

Tuy nhiên, phần lớn thiết bị IoT được thiết kế với tài nguyên tính toán hạn chế, thiếu cơ chế bảo mật mạnh mẽ, và thường sử dụng mật khẩu mặc định dễ đoán. Những điểm yếu này khiến chúng trở thành mục tiêu lý tưởng cho các cuộc tấn công mạng, đặc biệt là các botnet IoT – mạng lưới thiết bị bị nhiễm mã độc và bị kiểm soát từ xa để thực hiện các hành vi phá hoại.

### 1.1.2. Các cuộc tấn công botnet IoT điển hình

Một trong những sự kiện đáng chú ý nhất là cuộc tấn công DDoS (Distributed Denial of Service) từ botnet Mirai vào năm 2016, đã làm tê liệt hàng loạt website lớn như Twitter, Netflix, Reddit bằng cách khai thác hơn 600.000 thiết bị IoT bị nhiễm mã độc. Sự kiện này đã gióng lên hồi chuông cảnh báo về mức độ nghiêm trọng của mối đe dọa từ botnet IoT.

Các dạng tấn công phổ biến từ botnet IoT bao gồm:

- **DDoS (Distributed Denial of Service)**: Tấn công từ nhiều nguồn khác nhau nhằm làm quá tải hệ thống mục tiêu.
- **DoS (Denial of Service)**: Tấn công từ một hoặc vài nguồn để ngăn chặn dịch vụ hợp pháp.
- **Reconnaissance (Trinh sát)**: Thu thập thông tin về mạng và thiết bị để chuẩn bị cho các cuộc tấn công tiếp theo.
- **Theft (Trộm cắp dữ liệu)**: Đánh cắp thông tin nhạy cảm từ thiết bị bị xâm nhập.

### 1.1.3. Thách thức trong phát hiện và phòng chống

Việc phát hiện và phân loại các cuộc tấn công botnet IoT gặp phải nhiều thách thức lớn:

**Thách thức về dữ liệu**: Tập dữ liệu trong thực tế thường có sự mất cân bằng nghiêm trọng (class imbalance), trong đó lưu lượng tấn công chiếm tỷ lệ áp đảo so với lưu lượng bình thường. Ví dụ, trong tập dữ liệu Bot-IoT được sử dụng trong nghiên cứu này, lưu lượng tấn công chiếm tới 99.95% trong khi lưu lượng bình thường chỉ chiếm khoảng 0.05%. Sự mất cân bằng này khiến các mô hình học máy truyền thống dễ bị thiên lệch về lớp đa số, dẫn đến khả năng phát hiện lưu lượng bình thường kém.

**Thách thức về độ phức tạp**: Các cuộc tấn công ngày càng tinh vi với nhiều kỹ thuật ẩn náu và mã hóa, khiến việc phân biệt giữa lưu lượng tấn công và lưu lượng bình thường trở nên khó khăn hơn. Đặc biệt, việc phân biệt giữa DDoS (nhiều nguồn tấn công) và DoS (ít nguồn tấn công) đòi hỏi phân tích sâu về đặc trưng nguồn tấn công (source diversity).

**Thách thức về quy mô**: Với hàng tỷ thiết bị IoT hoạt động đồng thời, khối lượng dữ liệu lưu lượng mạng cần phân tích là rất lớn. Điều này đặt ra yêu cầu cao về khả năng mở rộng (scalability) và hiệu năng xử lý của hệ thống phát hiện xâm nhập.

**Thách thức về tài nguyên**: Nhiều giải pháp hiện tại yêu cầu tài nguyên tính toán lớn, không phù hợp với các tổ chức nhỏ hoặc môi trường giáo dục có ngân sách hạn chế.

## 1.2. Tính cấp thiết của đề tài

### 1.2.1. Khoảng trống nghiên cứu

Mặc dù đã có nhiều nghiên cứu về phát hiện tấn công mạng sử dụng học máy, vẫn tồn tại những khoảng trống quan trọng cần được giải quyết:

**Xử lý mất cân bằng dữ liệu nghiêm trọng**: Hầu hết các nghiên cứu hiện tại chỉ xử lý mất cân bằng ở mức độ trung bình hoặc sử dụng tập dữ liệu đã được cân bằng trước. Tuy nhiên, trong thực tế, tỷ lệ mất cân bằng có thể lên tới 2000:1 hoặc cao hơn, đòi hỏi các kỹ thuật xử lý chuyên sâu hơn như SMOTE (Synthetic Minority Over-sampling Technique) và cân bằng trọng số trong mô hình.

**Phân loại đa cấp hiệu quả**: Nhiều nghiên cứu chỉ tập trung vào phân loại nhị phân (tấn công/bình thường) hoặc phân loại đa lớp trực tiếp. Trong khi đó, mô hình phân cấp hai giai đoạn (hierarchical two-stage) có tiềm năng cải thiện độ chính xác bằng cách chia nhỏ bài toán phức tạp thành các bước đơn giản hơn.

**Tối ưu hóa tài nguyên**: Các giải pháp thường yêu cầu phần cứng chuyên dụng hoặc máy chủ mạnh mẽ. Nghiên cứu này hướng tới giải pháp có thể triển khai trên các nền tảng điện toán đám mây miễn phí hoặc chi phí thấp như Google Colab, phù hợp với môi trường giáo dục và nghiên cứu.

### 1.2.2. Hạn chế của các phương pháp hiện tại

Các phương pháp phát hiện xâm nhập truyền thống như signature-based detection chỉ có thể phát hiện các cuộc tấn công đã biết, trong khi anomaly-based detection thường có tỷ lệ false positive cao. Các mô hình học máy đơn giản như Decision Tree, Random Forest tuy có độ chính xác tốt nhưng gặp khó khăn khi xử lý dữ liệu mất cân bằng nghiêm trọng.

Mô hình Deep Learning như CNN, LSTM mặc dù cho kết quả tốt nhưng yêu cầu tài nguyên tính toán lớn, thời gian huấn luyện dài, và khó giải thích được quyết định của mô hình (lack of interpretability). Điều này khiến chúng không phù hợp cho các ứng dụng thời gian thực hoặc môi trường có tài nguyên hạn chế.

## 1.3. Mục tiêu nghiên cứu

### 1.3.1. Mục tiêu tổng quát

Xây dựng hệ thống phát hiện và phân loại botnet IoT dựa trên mô hình học máy phân cấp hai giai đoạn (Two-Stage Hierarchical Model), có khả năng xử lý hiệu quả dữ liệu mất cân bằng nghiêm trọng, đạt độ chính xác cao và có thể triển khai trên nền tảng điện toán đám mây với tài nguyên hợp lý.

### 1.3.2. Mục tiêu cụ thể

Để đạt được mục tiêu tổng quát, nghiên cứu đặt ra các mục tiêu cụ thể sau:

1. **Xây dựng pipeline xử lý dữ liệu quy mô lớn**: Phát triển quy trình xử lý, gộp và phân tích tập dữ liệu Bot-IoT (74 file CSV, khoảng 16GB) một cách hiệu quả, tận dụng tối đa thông tin từ dữ liệu thực tế.

2. **Thiết kế mô hình phân cấp hai giai đoạn**:

   - Giai đoạn 1: Phân loại nhị phân (Binary Classification) để phân biệt lưu lượng tấn công và lưu lượng bình thường.
   - Giai đoạn 2: Phân loại đa lớp (Multi-class Classification) để xác định loại tấn công cụ thể (DDoS, DoS, Reconnaissance).

3. **Xử lý mất cân bằng dữ liệu**: Áp dụng kỹ thuật SMOTE để tạo dữ liệu tổng hợp cho lớp thiểu số và sử dụng cân bằng trọng số (scale_pos_weight) trong XGBoost để cải thiện khả năng phát hiện lưu lượng bình thường.

4. **Tối ưu hóa đặc trưng**: Thiết kế các đặc trưng phân biệt nguồn tấn công (source diversity features) bao gồm unique_src_count, src_entropy, và top_src_ratio để phân biệt hiệu quả giữa DDoS và DoS.

5. **Đánh giá hiệu năng toàn diện**: Thực hiện đánh giá mô hình trên nhiều chỉ số khác nhau (accuracy, precision, recall, F1-score) cho cả hai giai đoạn và đánh giá hiệu năng tổng thể của pipeline.

6. **Tối ưu hóa tài nguyên và thời gian**: Đảm bảo hệ thống có thể chạy trên Google Colab Pro+ với cấu hình 52GB RAM và GPU T4/V100, hoàn thành quá trình huấn luyện trong khoảng 15-20 phút.

## 1.4. Phạm vi nghiên cứu và đối tượng nghiên cứu

### 1.4.1. Phạm vi nghiên cứu

Nghiên cứu tập trung vào các khía cạnh sau:

**Phạm vi về dữ liệu**: Sử dụng tập dữ liệu Bot-IoT được công bố bởi UNSW Canberra Cyber, bao gồm 74 file CSV với tổng dung lượng khoảng 16GB, chứa khoảng 40-50 triệu bản ghi lưu lượng mạng. Tập dữ liệu này bao gồm cả lưu lượng bình thường và các loại tấn công botnet IoT thực tế.

**Phạm vi về phương pháp**: Tập trung vào kỹ thuật học máy gradient boosting, cụ thể là XGBoost, kết hợp với các kỹ thuật xử lý mất cân bằng dữ liệu như SMOTE và scale_pos_weight. Không nghiên cứu các phương pháp Deep Learning phức tạp do hạn chế về tài nguyên.

**Phạm vi về loại tấn công**: Nghiên cứu tập trung vào ba loại tấn công chính: DDoS, DoS và Reconnaissance. Loại tấn công Theft có số lượng mẫu quá ít sẽ được loại bỏ khỏi quá trình huấn luyện.

**Phạm vi về môi trường triển khai**: Hệ thống được thiết kế và tối ưu hóa cho nền tảng Google Colab Pro+ với cấu hình High-RAM (52GB) và GPU (T4/V100/A100), phù hợp với môi trường giáo dục và nghiên cứu.

### 1.4.2. Đối tượng nghiên cứu

**Đối tượng chính**: Các luồng lưu lượng mạng (network flows) trong môi trường IoT, được biểu diễn dưới dạng các đặc trưng thống kê như số lượng gói tin, số byte, thời lượng kết nối, các giá trị thống kê (mean, stddev, min, max), và các đặc trưng về nguồn tấn công.

**Đối tượng phụ**: Các kỹ thuật xử lý mất cân bằng dữ liệu, phương pháp tối ưu hóa siêu tham số, và chiến lược tận dụng GPU để tăng tốc quá trình huấn luyện.

## 1.5. Hướng tiếp cận và phương pháp thực hiện

### 1.5.1. Hướng tiếp cận tổng quát

Nghiên cứu áp dụng hướng tiếp cận phân cấp hai giai đoạn (hierarchical two-stage approach) với các nguyên tắc chính:

**Chia nhỏ bài toán phức tạp**: Thay vì xây dựng một mô hình phân loại đa lớp trực tiếp, nghiên cứu chia thành hai giai đoạn:

- Giai đoạn 1 tập trung vào việc phân biệt có tấn công hay không (binary classification).
- Giai đoạn 2 chỉ xử lý các mẫu được xác định là tấn công để phân loại loại tấn công cụ thể (multi-class classification).

**Tối ưu hóa từng giai đoạn**: Mỗi giai đoạn được tối ưu hóa riêng biệt với các siêu tham số và kỹ thuật xử lý mất cân bằng phù hợp, giúp tăng hiệu năng tổng thể của hệ thống.

**Tận dụng GPU**: Sử dụng XGBoost với cấu hình tree_method="gpu_hist" và predictor="gpu_predictor" để tận dụng tối đa GPU, giảm thời gian huấn luyện từ hàng giờ xuống còn 15-20 phút.

### 1.5.2. Phương pháp thực hiện

Nghiên cứu thực hiện theo các bước chính:

**Bước 1 - Tiền xử lý dữ liệu**:

- Gộp 74 file CSV thành 8 batch file (mỗi batch khoảng 10 file), giảm số lần đọc/ghi file.
- Phân tích thống kê từng batch để xác định phân phối lớp và chọn batch phù hợp cho huấn luyện.
- Tạo tập test cân bằng (balanced test set) để đánh giá mô hình một cách công bằng.

**Bước 2 - Kỹ thuật đặc trưng (Feature Engineering)**:

- Trích xuất các đặc trưng cơ bản từ lưu lượng mạng (protocol, flags, state, packets, bytes, duration).
- Thiết kế đặc trưng phân biệt nguồn (source diversity features) dựa trên thống kê địa chỉ nguồn trong cửa sổ thời gian:
  - unique_src_count: Số lượng địa chỉ IP nguồn duy nhất
  - src_entropy: Entropy của phân phối địa chỉ nguồn
  - top_src_ratio: Tỷ lệ của địa chỉ nguồn xuất hiện nhiều nhất

**Bước 3 - Xử lý mất cân bằng**:

- Áp dụng SMOTE cho lớp thiểu số trong cả hai giai đoạn.
- Sử dụng scale_pos_weight trong XGBoost để tăng trọng số cho lớp thiểu số.

**Bước 4 - Huấn luyện mô hình**:

- Huấn luyện Stage 1 với XGBoost để phân loại nhị phân (Attack vs Normal).
- Huấn luyện Stage 2 với XGBoost để phân loại đa lớp (DDoS vs DoS vs Reconnaissance).
- Sử dụng early stopping để tránh overfitting.

**Bước 5 - Đánh giá và tối ưu**:

- Đánh giá mô hình trên tập test cân bằng.
- Phân tích ma trận nhầm lẫn (confusion matrix) để xác định điểm mạnh và điểm yếu.
- Tạo các biểu đồ trực quan hóa để đánh giá hiệu năng.

## 1.6. Ý nghĩa khoa học và thực tiễn

### 1.6.1. Ý nghĩa khoa học

**Đóng góp về mô hình**: Nghiên cứu đề xuất kiến trúc phân cấp hai giai đoạn kết hợp với kỹ thuật xử lý mất cân bằng đa cấp, có thể được áp dụng cho các bài toán phân loại có mất cân bằng nghiêm trọng tương tự trong lĩnh vực an ninh mạng.

**Đóng góp về đặc trưng**: Các đặc trưng source diversity (unique_src_count, src_entropy, top_src_ratio) được thiết kế dựa trên phân tích đặc tính tấn công DDoS và DoS, có thể được sử dụng để cải thiện khả năng phân biệt giữa hai loại tấn công này trong các nghiên cứu khác.

**Đóng góp về phương pháp**: Quy trình xử lý dữ liệu quy mô lớn (74 file, 16GB) một cách hiệu quả trên nền tảng có tài nguyên hạn chế, cung cấp hướng dẫn thực tiễn cho các nhà nghiên cứu gặp phải vấn đề tương tự.

### 1.6.2. Ý nghĩa thực tiễn

**Cho lĩnh vực giáo dục**: Hệ thống có thể được sử dụng làm công cụ giảng dạy và học tập về an ninh mạng IoT, giúp sinh viên hiểu rõ hơn về các kỹ thuật phát hiện xâm nhập hiện đại. Code được tổ chức rõ ràng, có chú thích chi tiết, dễ dàng tái sử dụng và mở rộng.

**Cho các tổ chức nhỏ và vừa**: Giải pháp có thể triển khai trên nền tảng điện toán đám mây với chi phí thấp, không yêu cầu đầu tư phần cứng chuyên dụng, phù hợp với các tổ chức có ngân sách hạn chế nhưng vẫn muốn có hệ thống phát hiện tấn công hiệu quả.

**Cho nghiên cứu tiếp theo**: Kết quả của nghiên cứu này có thể là nền tảng để phát triển các hệ thống phát hiện xâm nhập thời gian thực, tích hợp vào các thiết bị IoT gateway hoặc các giải pháp Network Security Monitoring (NSM).

**Cho cộng đồng mã nguồn mở**: Toàn bộ code và quy trình được công khai trên GitHub, cho phép cộng đồng nghiên cứu tái sử dụng, kiểm chứng và cải thiện, góp phần thúc đẩy sự phát triển của lĩnh vực an ninh IoT.

## 1.7. Cấu trúc báo cáo

Báo cáo được tổ chức thành năm chương chính:

**Chương 1 - Tổng quan**: Trình bày bối cảnh nghiên cứu, vấn đề cần giải quyết, tính cấp thiết, mục tiêu, phạm vi nghiên cứu, hướng tiếp cận tổng quát và ý nghĩa của đề tài.

**Chương 2 - Cơ sở lý thuyết**: Giới thiệu các khái niệm nền tảng về botnet IoT, các loại tấn công mạng, tập dữ liệu Bot-IoT, kỹ thuật học máy XGBoost, phương pháp xử lý mất cân bằng dữ liệu SMOTE, và các nghiên cứu liên quan.

**Chương 3 - Phương pháp nghiên cứu**: Mô tả chi tiết kiến trúc hệ thống Two-Stage Hierarchical Model, quy trình xử lý dữ liệu, kỹ thuật đặc trưng source diversity, các siêu tham số của mô hình, và môi trường thực nghiệm trên Google Colab.

**Chương 4 - Thực nghiệm và kết quả**: Trình bày kết quả huấn luyện, đánh giá hiệu năng của từng giai đoạn và toàn hệ thống, phân tích ma trận nhầm lẫn, so sánh với các phương pháp khác, và thảo luận về ưu điểm, hạn chế.

**Chương 5 - Kết luận và hướng phát triển**: Tóm tắt các đóng góp chính của nghiên cứu, đánh giá mức độ đạt được các mục tiêu đã đề ra, chỉ ra các hạn chế cần khắc phục, và đề xuất các hướng nghiên cứu tiếp theo.

---

**Kết thúc Chương 1**
