import time
import joblib
import pandas as pd
import xgboost as xgb
from datetime import datetime

# ===== CONFIG =====
MODEL_PATH = "model/xgboost_model.json"
ENCODER_PATH = "model/label_encoder.pkl"
DATA_PATH = "demo_scenario.csv"

SLEEP_TIME = 0.2  

# ===== LOAD MODEL & ENCODER =====
print("Loading model...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

label_encoder = joblib.load(ENCODER_PATH)

print("Model & encoder loaded ✓")

# ===== FEATURES USED =====
FEATURES = [
    "seq","pkts","bytes","dur",
    "mean","stddev","sum","min","max",
    "spkts","dpkts","sbytes","dbytes",
    "rate","srate","drate"
]

# ===== LOAD DEMO DATA =====
df = pd.read_csv(DATA_PATH)

X = df[FEATURES]

# ===== REALTIME DEMO =====
print("\n=== REALTIME DDoS DETECTION DEMO ===\n")

for i, row in X.iterrows():
    x = row.values.reshape(1, -1)

    pred = model.predict(x)[0]
    label = label_encoder.inverse_transform([pred])[0]

    ts = datetime.now().strftime("%H:%M:%S")

    print(f"[{ts}] Flow {i:05d} → Prediction: {label}")

    time.sleep(SLEEP_TIME)

print("\nFinished")
