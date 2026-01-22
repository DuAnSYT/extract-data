import pandas as pd
import os
import math

# Cấu hình
file_path = 'data_traintestsplit\google_24_04_train_part1.xlsx'  # Thay bằng đường dẫn file của bạn
num_parts = 2               # Số phần muốn chia

# Đọc file
df = pd.read_excel(file_path, engine='openpyxl')

# Tính số dòng mỗi phần
total_rows = len(df)
rows_per_part = math.ceil(total_rows / num_parts)

# Tách và lưu từng phần
file_name, file_ext = os.path.splitext(file_path)
for i in range(num_parts):
    start_idx = i * rows_per_part
    end_idx = start_idx + rows_per_part
    df_part = df.iloc[start_idx:end_idx].copy()
    part_path = f"{file_name}_part{i+1}{file_ext}"
    df_part.to_excel(part_path, index=False, engine='openpyxl')
    print(f"Đã lưu: {part_path}")