"""
============================================================
 LSTM-Based Electricity Consumption Prediction
------------------------------------------------------------
 Author : Arung Tirto Nusantara
 Year   : 2026

 Description:
 - Predict monthly electricity consumption (kWh)
 - Input features:
     1) kWh previous month
     2) Number of holidays in the target month
 - Model: Lightweight LSTM
 - Output: Predicted kWh and estimated electricity bill

 Note:
 - Dataset file is not included in this repository
 - This script is provided for academic and publication purposes
============================================================
"""

# =========================
# 0) SETUP & UTILITIES
# =========================
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 120

MODEL_DIR = "saved_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Metrics ----------
def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / np.maximum(1e-8, y))) * 100)

# =========================
# 1) LOAD DATA (PLACEHOLDER)
# =========================
CSV_PATH = "your_dataset.csv"   # <-- dataset tidak dipublikasikan
df_raw = pd.read_csv(CSV_PATH)

cols = ["kWh_bulan_sebelumnya", "hari_libur_bulan_ini", "kWh_bulan_ini"]
for c in cols:
    df_raw[c] = df_raw[c].astype(str).str.replace(",", ".").astype(float)

df = df_raw[cols].copy()

# =========================
# 2) SPLIT & SCALING
# =========================
X = df[cols[:-1]].values
y = df[cols[-1]].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train)
X_test  = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_train = X_train.reshape(-1, 1, X_train.shape[1])
X_test  = X_test.reshape(-1, 1, X_test.shape[1])

# =========================
# 3) MODEL DEFINITION
# =========================
tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    layers.Input(shape=(1, X_train.shape[2])),
    layers.LSTM(16, activation="tanh"),
    layers.Dropout(0.1),
    layers.Dense(8, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="mse"
)

model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# 4) EVALUATION
# =========================
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

mae  = mean_absolute_error(y_test, y_pred)
rmse_v = rmse(y_test, y_pred)
mape_v = mape(y_test, y_pred)

print(f"MAE  : {mae:.2f} kWh")
print(f"RMSE : {rmse_v:.2f} kWh")
print(f"MAPE : {mape_v:.2f} %")

# =========================
# 5) SAVE ARTIFACTS
# =========================
model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(MODEL_DIR, "scaler_y.pkl"))

print("Model and scalers saved to:", MODEL_DIR)
