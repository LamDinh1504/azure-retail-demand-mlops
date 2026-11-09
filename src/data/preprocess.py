from pathlib import Path
import pandas as pd


# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "raw" / "train.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. ĐỌC DỮ LIỆU
# ==========================================

print("Đang đọc dữ liệu...")

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["date"]
)

print("Số dòng ban đầu:", len(df))


# ==========================================
# 3. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU
# ==========================================

required_columns = ["date", "store", "item", "sales"]

if df[required_columns].isnull().any().any():
    raise ValueError("Dữ liệu có giá trị thiếu!")

if (df["sales"] < 0).any():
    raise ValueError("Dữ liệu sales có giá trị âm!")

if df.duplicated(
    subset=["date", "store", "item"]
).any():
    raise ValueError("Dữ liệu có bản ghi trùng!")


# ==========================================
# 4. SẮP XẾP DỮ LIỆU
# ==========================================

df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)


# ==========================================
# 5. TẠO ĐẶC TRƯNG THỜI GIAN
# ==========================================

df["day_of_week"] = df["date"].dt.day_of_week

df["month"] = df["date"].dt.month

df["day"] = df["date"].dt.day


# ==========================================
# 6. TẠO LAG FEATURES
# ==========================================

group = df.groupby(
    ["store", "item"]
)["sales"]

for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = group.shift(lag)


# ==========================================
# 7. TẠO ROLLING FEATURES
# ==========================================

df["rolling_mean_7"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

df["rolling_mean_28"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(
        lambda x: x.shift(1).rolling(28).mean()
    )
)


# ==========================================
# 8. XÓA DÒNG THIẾU LAG FEATURES
# ==========================================

feature_columns = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

df = df.dropna(subset=feature_columns)

print("Số dòng sau xử lý:", len(df))


# ==========================================
# 9. CHIA DỮ LIỆU THEO THỜI GIAN
# ==========================================

train_df = df[
    df["date"] < "2017-10-01"
]

val_df = df[
    (df["date"] >= "2017-10-01")
    & (df["date"] < "2017-12-01")
]

test_df = df[
    df["date"] >= "2017-12-01"
]


# ==========================================
# 10. LƯU DỮ LIỆU
# ==========================================

train_df.to_csv(
    OUTPUT_DIR / "train_processed.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "val_processed.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test_processed.csv",
    index=False
)


# ==========================================
# 11. HIỂN THỊ KẾT QUẢ
# ==========================================

print("\n===== KẾT QUẢ TIỀN XỬ LÝ =====")

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

print("\n5 dòng dữ liệu sau xử lý:")
print(df.head())

print("\nHoàn thành tiền xử lý dữ liệu!")