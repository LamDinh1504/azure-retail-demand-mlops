from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ========================================
# 1. CẤU HÌNH
# ========================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

TEST_PATH = (
    BASE_DIR / "data" / "processed" / "test_processed.csv"
)


# ========================================
# 2. TẢI MÔ HÌNH
# ========================================

print("Đang tải mô hình...")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]

print("Tải mô hình thành công!")


# ========================================
# 3. ĐỌC DỮ LIỆU KIỂM THỬ
# ========================================

test_df = pd.read_csv(TEST_PATH)

# Lấy 5 dòng đầu để dự báo
sample = test_df.head(5).copy()

X_sample = sample[features]


# ========================================
# 4. THỰC HIỆN DỰ BÁO
# ========================================

predictions = model.predict(X_sample)

# Không cho phép dự báo âm
predictions = np.maximum(predictions, 0)


# ========================================
# 5. HIỂN THỊ KẾT QUẢ
# ========================================

result = pd.DataFrame({
    "date": sample["date"],
    "store": sample["store"],
    "item": sample["item"],
    "actual_sales": sample["sales"],
    "predicted_sales": np.round(predictions, 2),
})

result["absolute_error"] = np.abs(
    result["actual_sales"] - result["predicted_sales"]
)

print("\n===== KẾT QUẢ DỰ BÁO =====")
print(result.to_string(index=False))

print("\nDự báo hoàn thành!")