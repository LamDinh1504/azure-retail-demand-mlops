import pandas as pd

# Đọc dữ liệu
df = pd.read_csv(
    "data/raw/train.csv",
    parse_dates=["date"]
)

print("===== 5 DÒNG ĐẦU TIÊN =====")
print(df.head())

print("\n===== KÍCH THƯỚC DỮ LIỆU =====")
print("Số dòng:", df.shape[0])
print("Số cột:", df.shape[1])

print("\n===== CÁC CỘT DỮ LIỆU =====")
print(df.columns.tolist())

print("\n===== KIỂU DỮ LIỆU =====")
print(df.dtypes)

print("\n===== GIÁ TRỊ THIẾU =====")
print(df.isnull().sum())

print("\n===== DỮ LIỆU TRÙNG LẶP =====")
print(
    df.duplicated(
        subset=["date", "store", "item"]
    ).sum()
)

print("\n===== THÔNG TIN BÁN HÀNG =====")
print("Số cửa hàng:", df["store"].nunique())
print("Số sản phẩm:", df["item"].nunique())

print("\n===== KHOẢNG THỜI GIAN =====")
print("Ngày bắt đầu:", df["date"].min())
print("Ngày kết thúc:", df["date"].max())

print("\n===== THỐNG KÊ SALES =====")
print(df["sales"].describe())