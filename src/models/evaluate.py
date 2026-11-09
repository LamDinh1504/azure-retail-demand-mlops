from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

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
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ========================================
# 2. ĐỌC DỮ LIỆU
# ========================================

print("Đang đọc Validation và Test...")

val_df = pd.read_csv(
    DATA_DIR / "val_processed.csv"
)

test_df = pd.read_csv(
    DATA_DIR / "test_processed.csv"
)


# ========================================
# 3. HÀM ĐÁNH GIÁ
# ========================================

def evaluate(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.maximum(0, np.asarray(y_pred))

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = root_mean_squared_error(
        y_true,
        y_pred
    )

    denominator = np.maximum(
        np.abs(y_true) + np.abs(y_pred),
        1e-8
    )

    smape = 200 * np.mean(
        np.abs(y_true - y_pred)
        / denominator
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "sMAPE": float(smape)
    }


# ========================================
# 4. DANH SÁCH MÔ HÌNH
# ========================================

model_files = {
    "RandomForest": MODEL_DIR / "random_forest.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib"
}

results = []
loaded_models = {}


# ========================================
# 5. ĐÁNH GIÁ VALIDATION
# ========================================

for name, path in model_files.items():

    print(f"\nĐang đánh giá {name}...")

    bundle = joblib.load(path)

    model = bundle["model"]
    features = bundle["features"]

    X_val = val_df[features]
    y_val = val_df["sales"]

    predictions = model.predict(X_val)

    metrics = evaluate(y_val, predictions)

    results.append({
        "Model": name,
        **metrics
    })

    loaded_models[name] = bundle


# ========================================
# 6. SO SÁNH KẾT QUẢ
# ========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\n===== MODEL COMPARISON =====")

print(results_df.to_string(index=False))


# ========================================
# 7. CHỌN MÔ HÌNH TỐT NHẤT
# ========================================

best_name = results_df.iloc[0]["Model"]

best_bundle = loaded_models[best_name]

best_model = best_bundle["model"]
best_features = best_bundle["features"]

print("\nMô hình tốt nhất:", best_name)


# ========================================
# 8. ĐÁNH GIÁ TRÊN TEST
# ========================================

X_test = test_df[best_features]
y_test = test_df["sales"]

test_predictions = best_model.predict(X_test)

test_metrics = evaluate(
    y_test,
    test_predictions
)

print("\n===== FINAL TEST RESULTS =====")

print("Model:", best_name)

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")


# ========================================
# 9. LƯU MÔ HÌNH TỐT NHẤT
# ========================================

best_model_path = MODEL_DIR / "best_model.joblib"

joblib.dump(
    best_bundle,
    best_model_path
)

print("\nĐã lưu best_model.joblib")


# ========================================
# 10. XUẤT BÁO CÁO
# ========================================

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

report = {
    "selected_model": best_name,
    "validation_results": results_df.to_dict(
        orient="records"
    ),
    "test_results": test_metrics,
    "training_mode": "100000_sample_rows"
}

with open(
    REPORT_DIR / "final_evaluation.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )

print("\nHoàn thành đánh giá mô hình!")