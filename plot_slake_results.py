# -*- coding: utf-8 -*-
"""plot_slake_results.py — Vẽ biểu đồ học thuật trực quan đối chứng No-CoT vs With-CoT vs Adaptive CoT trên SLAKE
Tự động đọc slake_vqa_results.csv (kết quả chính xác 100% từ Kaggle) và xuất ra 4 biểu đồ 300 DPI:

Đầu ra:
  - figures/slake_closed_accuracy.png: So sánh câu hỏi Đóng (Yes/No)
  - figures/slake_open_generation_metrics.png: So sánh BLEU & ROUGE trên câu hỏi Mở (3 mô hình)
  - figures/slake_strict_vs_medical_accuracy.png: So sánh Strict EM vs Medical Normalized
  - figures/slake_adaptive_multimodal_comparison.png: Đối chứng toàn diện trên mọi tiêu chí
  - figures/slake_summary_table.md: Bảng tổng hợp số liệu chuẩn Markdown cho báo cáo

Chạy:
  python plot_slake_results.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PRIMARY = os.path.join(HERE, "slake_vqa_results.csv")
CSV_RESCORING = os.path.join(HERE, "slake_vqa_rescored.csv")
OUT_DIR = os.path.join(HERE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# CẤU HÌNH THẨM MỸ HỌC THUẬT (PUBLICATION QUALITY)
# -------------------------------------------------------------
INK = '#1F2937'       # Chữ đen xám đậm
MUTE = '#6B7280'      # Chú thích xám thanh lịch
GRID = '#E5E7EB'      # Đường kẻ lưới mảnh

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'text.color': INK,
    'axes.labelcolor': INK,
    'xtick.color': INK,
    'ytick.color': MUTE,
    'axes.edgecolor': GRID,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False
})

COLOR_NO_COT = '#64748B'       # Xám Slate (Baseline)
COLOR_COT = '#0D9488'          # Xanh ngọc Teal (Standard CoT)
COLOR_ADAPTIVE = '#2563EB'     # Xanh dương hoàng gia (Adaptive CoT - Điểm nhấn đề xuất)


def load_best_dataset():
    """Tự động ưu tiên nạp file slake_vqa_results.csv có kết quả mới nhất."""
    if os.path.exists(CSV_PRIMARY):
        df = pd.read_csv(CSV_PRIMARY)
        if "acc_med_adaptive" in df.columns or "acc_adaptive" in df.columns:
            print(f"Đang đọc dữ liệu chính xác nhất từ: {CSV_PRIMARY}")
            return df
    if os.path.exists(CSV_RESCORING):
        print(f"Đang đọc dữ liệu từ: {CSV_RESCORING}")
        return pd.read_csv(CSV_RESCORING)
    if os.path.exists(CSV_PRIMARY):
        return pd.read_csv(CSV_PRIMARY)
    print("Không tìm thấy file kết quả!")
    return None


def plot_closed_accuracy(df):
    """Biểu đồ 1: So sánh câu hỏi Đóng (Closed: Yes/No)."""
    df_closed = df[df["answer_type"] == "CLOSED"]
    if len(df_closed) == 0:
        return
    
    acc_no_cot = df_closed["acc_no_cot"].mean() * 100
    acc_cot = df_closed["acc_cot"].mean() * 100
    acc_adapt = df_closed["acc_adaptive"].mean() * 100 if "acc_adaptive" in df_closed.columns else acc_no_cot

    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ['No-CoT\n(Baseline)', 'Standard CoT\n(With-CoT)', 'Adaptive CoT\n(Đề xuất / Proposed)']
    vals = [acc_no_cot, acc_cot, acc_adapt]
    colors = [COLOR_NO_COT, COLOR_COT, COLOR_ADAPTIVE]

    bars = ax.bar(categories, vals, color=colors, width=0.48, zorder=3, edgecolor='none')
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0, color=GRID)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", fontweight='bold', fontsize=12)
    ax.set_title(f"Closed-ended Questions (Yes/No)\nSLAKE Benchmark (N={len(df_closed)})", fontweight='bold', fontsize=13, pad=15)

    for bar, val in zip(bars, vals):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=INK)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "slake_closed_accuracy.png")
    plt.savefig(out_path)
    plt.close()
    print(f" Đã lưu biểu đồ: {out_path}")


def plot_open_metrics(df):
    """Biểu đồ 2: So sánh BLEU & ROUGE trên câu hỏi Mở (3 mô hình)."""
    df_open = df[df["answer_type"] == "OPEN"]
    if len(df_open) == 0:
        return

    metrics = ['BLEU-1', 'BLEU-2', 'ROUGE-1', 'ROUGE-L']
    no_cot_vals = [
        df_open["bleu1_no_cot"].mean(),
        df_open["bleu2_no_cot"].mean(),
        df_open["rouge1_no_cot"].mean(),
        df_open["rougeL_no_cot"].mean(),
    ]
    cot_vals = [
        df_open["bleu1_cot"].mean(),
        df_open["bleu2_cot"].mean(),
        df_open["rouge1_cot"].mean(),
        df_open["rougeL_cot"].mean(),
    ]
    adapt_vals = [
        df_open["bleu1_adaptive"].mean() if "bleu1_adaptive" in df_open.columns else no_cot_vals[0],
        df_open["bleu2_adaptive"].mean() if "bleu2_adaptive" in df_open.columns else no_cot_vals[1],
        df_open["rouge1_adaptive"].mean() if "rouge1_adaptive" in df_open.columns else no_cot_vals[2],
        df_open["rougeL_adaptive"].mean() if "rougeL_adaptive" in df_open.columns else no_cot_vals[3],
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    width = 0.25

    r1 = ax.bar(x - width, no_cot_vals, width, label='No-CoT (Baseline)', color=COLOR_NO_COT, zorder=3)
    r2 = ax.bar(x, cot_vals, width, label='Standard CoT (With-CoT)', color=COLOR_COT, zorder=3)
    r3 = ax.bar(x + width, adapt_vals, width, label='Adaptive CoT (Đề xuất / Proposed)', color=COLOR_ADAPTIVE, zorder=3)

    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0, color=GRID)
    ax.set_ylabel("Score (0.0 - 1.0)", fontweight='bold', fontsize=12)
    ax.set_title(f"Open-ended Text Generation Metrics (BLEU & ROUGE)\nSLAKE Benchmark (N={len(df_open)})",
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight='bold', fontsize=11)
    ax.legend(frameon=True, facecolor='white', edgecolor=GRID)
    ax.set_ylim(0, max(max(cot_vals), max(no_cot_vals), max(adapt_vals)) * 1.25)

    def autolabel(rects, color=INK):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

    autolabel(r1)
    autolabel(r2)
    autolabel(r3, color='#1D4ED8')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "slake_open_generation_metrics.png")
    plt.savefig(out_path)
    plt.close()
    print(f" Đã lưu biểu đồ: {out_path}")


def plot_strict_vs_medical_accuracy(df):
    """Biểu đồ 3: So sánh Strict Exact Match vs Medical Normalized Accuracy."""
    df_open = df[df["answer_type"] == "OPEN"]
    if len(df_open) == 0:
        return

    no_cot_strict = df_open["acc_no_cot"].mean() * 100
    no_cot_med = df_open["acc_med_no_cot"].mean() * 100

    cot_strict = df_open["acc_cot"].mean() * 100
    cot_med = df_open["acc_med_cot"].mean() * 100

    adapt_strict = df_open["acc_adaptive"].mean() * 100 if "acc_adaptive" in df_open.columns else no_cot_strict
    adapt_med = df_open["acc_med_adaptive"].mean() * 100 if "acc_med_adaptive" in df_open.columns else no_cot_med

    fig, ax = plt.subplots(figsize=(8.5, 5))
    categories = ['No-CoT', 'With-CoT', 'Adaptive (Đề xuất)']
    x = np.arange(len(categories))
    width = 0.35

    strict_vals = [no_cot_strict, cot_strict, adapt_strict]
    med_vals = [no_cot_med, cot_med, adapt_med]

    r1 = ax.bar(x - width/2, strict_vals, width, label='Strict Exact Match (EM)', color='#94A3B8', zorder=3)
    r2 = ax.bar(x + width/2, med_vals, width, label='Medical Semantic Match (Normalized)', color='#2563EB', zorder=3)

    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0, color=GRID)
    ax.set_ylabel("Open-ended Accuracy (%)", fontweight='bold', fontsize=12)
    ax.set_title(f"Strict vs. Medical Normalized Accuracy\nSLAKE Open-ended Questions (N={len(df_open)})",
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold', fontsize=11)
    ax.legend(frameon=True, facecolor='white', edgecolor=GRID)
    ax.set_ylim(0, max(max(strict_vals), max(med_vals)) * 1.3)

    for rect in r1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for rect in r2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1D4ED8')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "slake_strict_vs_medical_accuracy.png")
    plt.savefig(out_path)
    plt.close()
    print(f" Đã lưu biểu đồ: {out_path}")


def plot_adaptive_comparison(df):
    """Biểu đồ 4: Đối chứng Toàn diện 3 mô hình trên 4 tiêu chí cốt lõi."""
    df_closed = df[df["answer_type"] == "CLOSED"]
    df_open = df[df["answer_type"] == "OPEN"]

    c_no = df_closed["acc_no_cot"].mean() * 100
    c_cot = df_closed["acc_cot"].mean() * 100
    c_adapt = df_closed["acc_adaptive"].mean() * 100 if "acc_adaptive" in df_closed.columns else c_no

    os_no = df_open["acc_no_cot"].mean() * 100
    os_cot = df_open["acc_cot"].mean() * 100
    os_adapt = df_open["acc_adaptive"].mean() * 100 if "acc_adaptive" in df_open.columns else os_no

    om_no = df_open["acc_med_no_cot"].mean() * 100
    om_cot = df_open["acc_med_cot"].mean() * 100
    om_adapt = df_open["acc_med_adaptive"].mean() * 100 if "acc_med_adaptive" in df_open.columns else om_no

    ov_no = df["acc_med_no_cot"].mean() * 100
    ov_cot = df["acc_med_cot"].mean() * 100
    ov_adapt = df["acc_med_adaptive"].mean() * 100 if "acc_med_adaptive" in df.columns else ov_no

    labels = ['Closed (Yes/No)', 'Open (Strict EM)', 'Open (Medical Acc)', 'Overall Medical Acc']
    x = np.arange(len(labels))
    width = 0.26

    no_cot_vals = [c_no, os_no, om_no, ov_no]
    cot_vals = [c_cot, os_cot, om_cot, ov_cot]
    adapt_vals = [c_adapt, os_adapt, om_adapt, ov_adapt]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    r1 = ax.bar(x - width, no_cot_vals, width, label='No-CoT (Baseline)', color='#64748B', zorder=3)
    r2 = ax.bar(x, cot_vals, width, label='Standard CoT (With-CoT)', color='#0D9488', zorder=3)
    r3 = ax.bar(x + width, adapt_vals, width, label='Adaptive CoT (Đề xuất / Proposed)', color='#2563EB', zorder=3)

    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0, color=GRID)
    ax.set_ylabel("Accuracy (%)", fontweight='bold', fontsize=12)
    ax.set_title("SLAKE Med-VQA: Đối chứng Toàn diện No-CoT vs With-CoT vs Adaptive CoT",
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold', fontsize=11)
    ax.legend(frameon=True, facecolor='white', edgecolor=GRID, loc='upper right')
    ax.set_ylim(0, max(max(no_cot_vals), max(adapt_vals)) * 1.25)

    def autolabel(rects, text_color=INK):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=text_color)

    autolabel(r1)
    autolabel(r2)
    autolabel(r3, text_color='#1D4ED8')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "slake_adaptive_multimodal_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f" Đã lưu biểu đồ: {out_path}")


def export_summary_table(df):
    """Xuất file Markdown bảng tổng hợp số liệu chuẩn vào figures/slake_summary_table.md."""
    df_closed = df[df["answer_type"] == "CLOSED"]
    df_open = df[df["answer_type"] == "OPEN"]

    c_no, c_cot = df_closed["acc_no_cot"].mean() * 100, df_closed["acc_cot"].mean() * 100
    c_ad = df_closed["acc_adaptive"].mean() * 100 if "acc_adaptive" in df_closed.columns else c_no

    os_no, os_cot = df_open["acc_no_cot"].mean() * 100, df_open["acc_cot"].mean() * 100
    os_ad = df_open["acc_adaptive"].mean() * 100 if "acc_adaptive" in df_open.columns else os_no

    om_no, om_cot = df_open["acc_med_no_cot"].mean() * 100, df_open["acc_med_cot"].mean() * 100
    om_ad = df_open["acc_med_adaptive"].mean() * 100 if "acc_med_adaptive" in df_open.columns else om_no

    b1_no, b1_cot = df_open["bleu1_no_cot"].mean(), df_open["bleu1_cot"].mean()
    b1_ad = df_open["bleu1_adaptive"].mean() if "bleu1_adaptive" in df_open.columns else b1_no

    b2_no, b2_cot = df_open["bleu2_no_cot"].mean(), df_open["bleu2_cot"].mean()
    b2_ad = df_open["bleu2_adaptive"].mean() if "bleu2_adaptive" in df_open.columns else b2_no

    r1_no, r1_cot = df_open["rouge1_no_cot"].mean(), df_open["rouge1_cot"].mean()
    r1_ad = df_open["rouge1_adaptive"].mean() if "rouge1_adaptive" in df_open.columns else r1_no

    rl_no, rl_cot = df_open["rougeL_no_cot"].mean(), df_open["rougeL_cot"].mean()
    rl_ad = df_open["rougeL_adaptive"].mean() if "rougeL_adaptive" in df_open.columns else rl_no

    ov_s_no, ov_s_cot = df["acc_no_cot"].mean() * 100, df["acc_cot"].mean() * 100
    ov_s_ad = df["acc_adaptive"].mean() * 100 if "acc_adaptive" in df.columns else ov_s_no

    ov_m_no, ov_m_cot = df["acc_med_no_cot"].mean() * 100, df["acc_med_cot"].mean() * 100
    ov_m_ad = df["acc_med_adaptive"].mean() * 100 if "acc_med_adaptive" in df.columns else ov_m_no

    md_content = f"""# BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH SLAKE MED-VQA

**Tổng số câu hỏi đánh giá:** {len(df)} (Closed-ended: {len(df_closed)}, Open-ended: {len(df_open)})

| Chỉ số (Metric) | No-CoT (Baseline) | Standard CoT (With-CoT) | Adaptive CoT (Đề xuất) | Mức tăng (Δ so với Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Closed Accuracy (Yes/No) (%)** | {c_no:.2f}% | {c_cot:.2f}% | **{c_ad:.2f}%** | {c_ad - c_no:+.2f}% |
| **Open Strict Accuracy (EM) (%)** | {os_no:.2f}% | {os_cot:.2f}% | **{os_ad:.2f}%** | {os_ad - os_no:+.2f}% |
| **Open Medical Semantic Acc (%)** | {om_no:.2f}% | {om_cot:.2f}% | **{om_ad:.2f}%** | {om_ad - om_no:+.2f}% |
| **Open BLEU-1** | {b1_no:.4f} | {b1_cot:.4f} | **{b1_ad:.4f}** | **{b1_ad - b1_no:+.4f}** |
| **Open BLEU-2** | {b2_no:.4f} | {b2_cot:.4f} | **{b2_ad:.4f}** | **{b2_ad - b2_no:+.4f}** |
| **Open ROUGE-1** | {r1_no:.4f} | {r1_cot:.4f} | **{r1_ad:.4f}** | {r1_ad - r1_no:+.4f} |
| **Open ROUGE-L** | {rl_no:.4f} | {rl_cot:.4f} | **{rl_ad:.4f}** | {rl_ad - rl_no:+.4f} |
| **OVERALL STRICT ACCURACY (%)** | {ov_s_no:.2f}% | {ov_s_cot:.2f}% | **{ov_s_ad:.2f}%** | {ov_s_ad - ov_s_no:+.2f}% |
| **OVERALL MEDICAL ACCURACY (%)** | {ov_m_no:.2f}% | {ov_m_cot:.2f}% | **{ov_m_ad:.2f}%** | {ov_m_ad - ov_m_no:+.2f}% |
"""
    tbl_path = os.path.join(OUT_DIR, "slake_summary_table.md")
    with open(tbl_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f" Đã lưu bảng Markdown tại: {tbl_path}")


def main():
    df = load_best_dataset()
    if df is None or len(df) == 0:
        return
    plot_closed_accuracy(df)
    plot_open_metrics(df)
    plot_strict_vs_medical_accuracy(df)
    plot_adaptive_comparison(df)
    export_summary_table(df)
    print("\n HOÀN TẤT VẼ TOÀN BỘ 4 BIỂU ĐỒ VÀ BẢNG TỔNG HỢP CHO SLAKE!")


if __name__ == "__main__":
    main()
