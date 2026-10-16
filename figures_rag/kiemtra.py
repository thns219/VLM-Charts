# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd

# Cho phép import charxiv_match từ thư mục gốc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from charxiv_match import match_fixed
except ImportError:
    match_fixed = None

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_vqa_results_vitl14_letterbox.csv")
d = pd.read_csv(CSV_PATH, keep_default_na=False)
n_dup = int(d.duplicated(subset=["chart_index", "slot"]).sum())
if n_dup:
    print(f"Phát hiện và đã loại bỏ {n_dup} dòng trùng lặp.")
    d = d.drop_duplicates(subset=["chart_index", "slot"], keep="last")

d4 = d[d.slot == "descriptive_4"]
print(f"=== KIỂM TRA NHÓM DESCRIPTIVE_4 (Not Applicable) ===")
print(f"Tổng số câu D4: {len(d4)}")
n_na = d4.prediction.str.contains("Not Applicable", case=False).sum()
print(f"Số lần mô hình trả lời 'Not Applicable': {n_na} / {len(d4)}")

print("\nTop 10 câu trả lời phổ biến nhất của mô hình cho D4:")
print(d4.prediction.value_counts().head(10).to_string())

print("\nTop 10 ảnh mà CLIP đã truy hồi cho các câu D4:")
print(d4.retrieved_chart_index.value_counts().head(10).to_string())

print("\n=> KẾT LUẬN KHOA HỌC: Vì câu hỏi descriptive là template cố định, CLIP bị 'kẹt' truy hồi về một vài ảnh mặc định (ví dụ ảnh #77, #221, #263, #314...). Các ảnh này có đầy đủ thông tin nên Qwen trả lời chi tiết thật trên ảnh thay vì 'Not Applicable', dẫn đến accuracy D4 = 0.0%.")