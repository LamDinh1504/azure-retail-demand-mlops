from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)


# ========================================
# 1. CẤU HÌNH
# ========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "store",
    "item",
    "day",
    "day_of_week",
    "month",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]

TARGET = "sales"


# ========================================
# 2. ĐỌC DỮ LIỆU
# ========================================

print("Đang đọc dữ liệu...")

train_df = pd.read_csv(
    DATA_DIR / "train_processed.csv"
)

val_df = pd.read_csv(
    DATA_DIR / "val_processed.csv"
)

print("Train ban đầu:", train_df.shape)
print("Validation:", val_df.shape)


# ========================================
# 3. LẤY MẪU ĐỂ CHẠY THỬ
# ========================================

train_df = train_df.sample(
    n=min(100_000, len(train_df)),
    random_state=42,
)

print("Train sau khi lấy mẫu:", train_df.shape)


# ========================================
# 4. TÁCH FEATURES VÀ TARGET
# ========================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]


# ========================================
# 5. KHỞI TẠO RANDOM FOREST
# ========================================

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=14,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)


# ========================================
# 6. HUẤN LUYỆN MÔ HÌNH
# ========================================

print("\nĐang huấn luyện Random Forest...")

model.fit(X_train, y_train)

print("Huấn luyện thành công!")


# ========================================
# 7. DỰ ĐOÁN TRÊN VALIDATION
# ========================================

predictions = model.predict(X_val)

# Doanh số không được âm
predictions = np.maximum(predictions, 0)


# ========================================
# 8. ĐÁNH GIÁ MÔ HÌNH
# ========================================

mae = mean_absolute_error(
    y_val,
    predictions,
)

rmse = root_mean_squared_error(
    y_val,
    predictions,
)

denominator = np.maximum(
    np.abs(y_val.to_numpy()) + np.abs(predictions),
    1e-8,
)

smape = 200 * np.mean(
    np.abs(y_val.to_numpy() - predictions)
    / denominator
)

print("\n===== RANDOM FOREST =====")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"sMAPE: {smape:.2f}%")


# ========================================
# 9. ĐÁNH GIÁ BASELINE
# ========================================

baseline_predictions = val_df["lag_7"]

baseline_mae = mean_absolute_error(
    y_val,
    baseline_predictions,
)

print("\n===== BASELINE =====")
print(f"MAE: {baseline_mae:.4f}")


# ========================================
# 10. LƯU MODEL
# ========================================

model_path = MODEL_DIR / "random_forest.joblib"

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
    },
    model_path,
)

print("\nĐã lưu mô hình tại:", model_path)