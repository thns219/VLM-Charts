# -*- coding: utf-8 -*-
"""clean_and_rescore.py — Script làm sạch toàn bộ dữ liệu CSV, deduplicate,
chấm lại bằng match_fixed cải tiến và tái sinh toàn bộ bảng biểu / figures.

Chạy:
    python clean_and_rescore.py
"""
import os
import sys
import io

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from charxiv_match import match_fixed

HERE = os.path.dirname(os.path.abspath(__file__))

SLOT_ORDER = {
    "descriptive_1": 1,
    "descriptive_2": 2,
    "descriptive_3": 3,
    "descriptive_4": 4,
    "reasoning": 5,
}

VQA_FILES = [
    "baseline_results.csv",
    "baseline_qwen_results.csv",
    "rag_vqa_results.csv",
    "rag_vqa_results_vitl14_letterbox.csv",
    "rag_vqa_results_vitl14_letterbox_caption.csv",
]


def clean_vqa_file(filename):
    filepath = os.path.join(HERE, filename)
    if not os.path.exists(filepath):
        print(f"[BỎ QUA] Không tìm thấy {filename}")
        return None

    df = pd.read_csv(filepath, keep_default_na=False)
    original_len = len(df)
    
    # 1. Deduplicate
    if {"chart_index", "slot"}.issubset(df.columns):
        df["chart_index"] = pd.to_numeric(df["chart_index"], errors="coerce")
        df = df.dropna(subset=["chart_index"]).copy()
        df["chart_index"] = df["chart_index"].astype(int)
        
        n_dup = int(df.duplicated(subset=["chart_index", "slot"]).sum())
        if n_dup > 0:
            df = df.drop_duplicates(subset=["chart_index", "slot"], keep="last")
        
        # 2. Sắp xếp thứ tự chuẩn: theo chart_index rồi theo slot
        df["_slot_rank"] = df["slot"].map(lambda s: SLOT_ORDER.get(str(s), 99))
        df = df.sort_values(by=["chart_index", "_slot_rank"]).drop(columns=["_slot_rank"]).reset_index(drop=True)
    
    # 3. Chấm lại bằng match_fixed cải tiến
    old_correct_rate = df["correct"].map(lambda x: str(x).strip().lower() == "true").mean() if "correct" in df else None
    df["correct"] = [match_fixed(p, g) for p, g in zip(df["prediction"], df["gold_answer"])]
    new_correct_rate = df["correct"].mean()

    # 4. Lưu lại
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    diff_rate = (new_correct_rate - old_correct_rate) * 100 if old_correct_rate is not None else 0.0
    print(f"[{filename:44s}] Ban đầu: {original_len:4d} dòng -> Đã lọc: {len(df):4d} dòng (dups: {original_len - len(df):2d}) | Acc: {old_correct_rate*100:5.2f}% -> {new_correct_rate*100:5.2f}% ({diff_rate:+5.2f}%)")
    return df


def clean_cot_file():
    filepath = os.path.join(HERE, "cot_results.csv")
    if not os.path.exists(filepath):
        return
    df = pd.read_csv(filepath, keep_default_na=False)
    orig_len = len(df)
    df["chart_index"] = pd.to_numeric(df["chart_index"], errors="coerce")
    df = df.dropna(subset=["chart_index"]).copy()
    df["chart_index"] = df["chart_index"].astype(int)
    df = df.drop_duplicates(subset=["chart_index"], keep="last")
    df = df.sort_values("chart_index").reset_index(drop=True)
    
    # Chấm lại direct_pred và cot_pred
    df["direct_correct"] = [match_fixed(p, g) for p, g in zip(df["direct_pred"], df["gold_answer"])]
    df["cot_correct"] = [match_fixed(p, g) for p, g in zip(df["cot_pred"], df["gold_answer"])]
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"[{'cot_results.csv':44s}] Ban đầu: {orig_len:4d} dòng -> Đã lọc: {len(df):4d} biểu đồ | Direct: {df['direct_correct'].mean()*100:5.2f}% | CoT: {df['cot_correct'].mean()*100:5.2f}%")


def clean_failed_404_file():
    filepath = os.path.join(HERE, "baseline_results_FAILED_404.csv")
    if not os.path.exists(filepath):
        return
    df = pd.read_csv(filepath, keep_default_na=False)
    # Cập nhật lại cột correct: dự đoán rỗng không thể là True
    df["correct"] = [match_fixed(p, g) for p, g in zip(df["prediction"], df["gold_answer"])]
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"[{'baseline_results_FAILED_404.csv':44s}] Đã chuẩn hóa cột correct (tất cả câu rỗng -> False).")


def main():
    print("=" * 80)
    print("BƯỚC 1: LÀM SẠCH VÀ CHUẨN HÓA CÁC FILE CSV")
    print("=" * 80)
    for fname in VQA_FILES:
        clean_vqa_file(fname)
    clean_cot_file()
    clean_failed_404_file()

    print("\n" + "=" * 80)
    print("BƯỚC 2: CHẠY LẠI COMPARE VÀ VẼ LẠI BIỂU ĐỒ")
    print("=" * 80)
    # Chạy compare_rag_versions
    try:
        import compare_rag_versions
        print("[compare_rag_versions.py] Đã cập nhật xong số liệu 5 hệ thống.")
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể chạy compare_rag_versions: {e}")

    # Chạy plot_rag_results
    try:
        import plot_rag_results
        plot_rag_results.main()
        print("[plot_rag_results.py] Đã cập nhật xong các hình trong figures_rag/.")
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể chạy plot_rag_results: {e}")

    print("\nHOÀN TẤT TOÀN BỘ CÔNG TÁC LÀM SẠCH VÀ ĐÁNH GIÁ DỰ ÁN!")


if __name__ == "__main__":
    main()
