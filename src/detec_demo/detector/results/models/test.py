import json
import joblib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Nối với tên file pkl (giả sử file pkl nằm CÙNG thư mục với file code)
file_path = os.path.join(current_dir, 'label_encoder.pkl')


le = joblib.load(file_path) # {'flgs': LabelEncoder(), 'proto': LabelEncoder(), 'state': LabelEncoder()}

print(le.classes_)

le_proto = le['state'] 
print("Các loại giao thức:", le_proto.classes_)

# Ví dụ: Muốn xem encoder của cột 'flgs' (để giải mã e, s...)
le_flgs = le['flgs']
print("Các loại cờ (flags):", le_flgs.classes_)

# ['flgs', 'proto', 'pkts', 'bytes', 'state', 'seq', 'dur', 'mean', 'stddev', 'sum', 'min', 'max', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate', 'unique_src_count', 'src_entropy', 'top_src_ratio']

# LabelEncoder()

# ['HTTP' 'TCP' 'UDP']