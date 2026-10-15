# -*- coding: utf-8 -*-
"""plot_multimodal_comparison.py — Vẽ biểu đồ học thuật trực quan đối chứng toàn diện:
Direct Baseline, Standard CoT, RAG-VQA, LoRA Fine-Tuned CoT và Adaptive LoRA trên 500 câu suy luận CharXiv.

Đầu ra: Lưu 4 biểu đồ độ phân giải cao (300 DPI) và bảng tổng hợp CSV vào thư mục figures/.
Chạy: python plot_multimodal_comparison.py
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

from charxiv_match import match_fixed

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = HERE
OUT_DIR = os.path.join(HERE, 'figures')
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

COLORS = {
    'Gemini': '#2563EB',          # Xanh dương hoàng gia
    'Qwen_Direct': '#0D9488',     # Xanh ngọc đậm (Teal)
    'Standard_CoT': '#94A3B8',    # Xám Slate
    'LoRA_CoT': '#8B5CF6',        # Tím LoRA
    'Adaptive_LoRA': '#EA580C',   # Cam đậm nổi bật (Điểm nhấn đề tài)
    'RAG_VQA': '#EF4444'          # Đỏ cam RAG
}

LABELS = {
    'Gemini': 'Gemini 2.5 Flash\n(Oracle / Closed API)',
    'Qwen_Direct': 'Qwen2.5-VL-7B\n(Direct Baseline)',
    'Standard_CoT': 'Standard CoT\n(2-Stage CoT)',
    'LoRA_CoT': 'LoRA CoT\n(Domain Grounded)',
    'Adaptive_LoRA': 'Adaptive LoRA CoT\n(Đề xuất / Proposed)',
    'RAG_VQA': 'RAG-VQA\n(CLIP + Qwen)'
}


def load_dataset_scored(filepath, pred_col='prediction', gold_col='gold_answer'):
    p = os.path.join(DATA_DIR, filepath)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Không tìm thấy file: {p}")
    df = pd.read_csv(p, keep_default_na=False)
    if 'slot' in df.columns:
        df = df[df.slot == 'reasoning'].copy().reset_index(drop=True)
    df['correct'] = [match_fixed(pr, g) for pr, g in zip(df[pred_col], df[gold_col])]
    return df


def boot_ci(df, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    ch = df.groupby('chart_index').correct.agg(['sum', 'count'])
    s = ch['sum'].values
    c = ch['count'].values
    k = len(ch)
    if k == 0:
        return 0.0, 0.0
    idx = rng.integers(0, k, (n, k))
    v = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(v, [2.5, 97.5]) * 100


def style_axis(ax, ymax=100, step=10):
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=0)
    ticks = list(range(0, int(ymax) + 1, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{x}%' for x in ticks])
    ax.set_ylim(0, ymax * 1.05)


def add_title(fig, main_title, subtitle=None):
    fig.text(0.06, 0.96, main_title, fontsize=16, fontweight='bold', ha='left', va='top')
    if subtitle:
        fig.text(0.06, 0.91, subtitle, fontsize=10.5, color=MUTE, ha='left', va='top')


def add_bar_label(ax, x, y, text, color=INK, dy=1.2, size=11, bold=True):
    ax.text(x, y + dy, text, ha='center', va='bottom', fontsize=size, fontweight='bold' if bold else 'normal', color=color)


# -------------------------------------------------------------
# 1. TẢI VÀ CHẤM TOÀN BỘ 5 MÔ HÌNH TRÊN 500 CÂU CHARXIV
# -------------------------------------------------------------
print("Đang nạp và chấm điểm toàn bộ các file kết quả...")
data = {
    'Gemini': load_dataset_scored('baseline_results.csv', 'prediction', 'gold_answer'),
    'Qwen_Direct': load_dataset_scored('baseline_qwen_results.csv', 'prediction', 'gold_answer'),
    'Standard_CoT': load_dataset_scored('cot_results.csv', 'cot_pred', 'gold_answer'),
    'LoRA_CoT': load_dataset_scored('structured_cot_lora_results.csv', 'prediction', 'gold_answer'),
    'Adaptive_LoRA': load_dataset_scored('adaptive_lora_results.csv', 'prediction', 'gold_answer'),
}

if os.path.exists(os.path.join(DATA_DIR, 'rag_vqa_results.csv')):
    data['RAG_VQA'] = load_dataset_scored('rag_vqa_results.csv', 'prediction', 'gold_answer')

# -------------------------------------------------------------
# 2. XUẤT BẢNG TỔNG HỢP HỌC THUẬT (CSV)
# -------------------------------------------------------------
subtypes = ['Number-in-Chart', 'Number-in-General', 'Text-in-Chart', 'Text-in-General']
summary_rows = []

for k, df in data.items():
    overall_acc = df.correct.mean() * 100
    lo, hi = boot_ci(df)
    row = {
        'Hệ thống': LABELS[k].replace('\n', ' '),
        'Số câu': len(df),
        'Tổng thể (%)': round(overall_acc, 2),
        'KTC 95% Thấp': round(lo, 2),
        'KTC 95% Cao': round(hi, 2),
    }
    for s in subtypes:
        sub_df = df[df.sub_type == s]
        row[s] = round(sub_df.correct.mean() * 100, 2) if len(sub_df) > 0 else 0.0
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, 'summary_multimodal_comparison.csv')
summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
print("\n=== BẢNG TỔNG HỢP KẾT QUẢ ĐỐI CHỨNG ===")
print(summary_df.to_string(index=False))
print(f"Đã lưu bảng tổng hợp tại: {summary_path}\n")


# -------------------------------------------------------------
# 3. BIỂU ĐỒ 1: SO SÁNH ĐỘ CHÍNH XÁC TỔNG THỂ (CÓ VẠCH KTC 95%)
# -------------------------------------------------------------
models_fig1 = ['RAG_VQA', 'Standard_CoT', 'LoRA_CoT', 'Adaptive_LoRA', 'Qwen_Direct', 'Gemini']
models_fig1 = [m for m in models_fig1 if m in data]

fig, ax = plt.subplots(figsize=(10.5, 6.2))
fig.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.96)
add_title(fig, 'So sánh độ chính xác tổng thể trên 500 câu suy luận CharXiv',
          'Vạch sai số biểu thị Khoảng Tin Cậy 95% (Bootstrap 2.000 lượt) · Chấm bằng match_fixed chuẩn')

x = np.arange(len(models_fig1))
accs = []
for i, m in enumerate(models_fig1):
    df = data[m]
    acc = df.correct.mean() * 100
    lo, hi = boot_ci(df)
    accs.append(acc)
    bar = ax.bar(i, acc, width=0.52, color=COLORS[m], zorder=3)
    ax.errorbar(i, acc, yerr=[[acc - lo], [hi - acc]], color=INK, capsize=5, lw=1.6, zorder=4)
    add_bar_label(ax, i, hi, f'{acc:.1f}%', COLORS[m], dy=1.2, size=13)

ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in models_fig1], fontsize=10.5)
style_axis(ax, ymax=60, step=10)

# Mũi tên làm nổi bật bước nhảy từ CoT lên Adaptive LoRA
idx_cot = models_fig1.index('Standard_CoT')
idx_adapt = models_fig1.index('Adaptive_LoRA')
ax.annotate('', xy=(idx_adapt, accs[idx_adapt] + 4), xytext=(idx_cot, accs[idx_cot] + 4),
            arrowprops=dict(arrowstyle='->', color=COLORS['Adaptive_LoRA'], lw=2, connectionstyle='arc3,rad=-0.15'), zorder=5)
ax.text((idx_cot + idx_adapt) / 2, accs[idx_adapt] + 6.5,
        f'+{accs[idx_adapt] - accs[idx_cot]:.1f}% (Khắc phục CoT)',
        ha='center', fontsize=10.5, color=COLORS['Adaptive_LoRA'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=COLORS['Adaptive_LoRA'], lw=1.2), zorder=6)

fig1_path = os.path.join(OUT_DIR, 'fig_so_sanh_tong_the_multimodal.png')
fig.savefig(fig1_path)
plt.close(fig)
print(f"Đã lưu: {fig1_path}")


# -------------------------------------------------------------
# 4. BIỂU ĐỒ 2: ĐỘT PHÁ Ở NHÓM CÂU HỎI SỐ HỌC (HERO FIGURE)
# -------------------------------------------------------------
# Minh chứng bước nhảy vọt của Number-in-General và Number-in-Chart
num_types = ['Number-in-General\n(So sánh, tính toán số)', 'Number-in-Chart\n(Đọc giá trị trục số)']
num_keys = ['Number-in-General', 'Number-in-Chart']
compare_models = ['Standard_CoT', 'Qwen_Direct', 'Adaptive_LoRA']

fig, ax = plt.subplots(figsize=(9.5, 6.0))
fig.subplots_adjust(top=0.81, bottom=0.18, left=0.08, right=0.96)
add_title(fig, 'Đột phá ở bài toán Số học: LoRA hóa giải Visual CoT Paradox',
          'So sánh tỷ lệ trả lời đúng ở 2 nhóm câu hỏi số học phức tạp nhất trên CharXiv')

x = np.arange(len(num_types))
w = 0.25
for j, m in enumerate(compare_models):
    vals = [data[m][data[m].sub_type == k].correct.mean() * 100 for k in num_keys]
    pos = x + (j - 1) * w
    bars = ax.bar(pos, vals, width=w * 0.9, color=COLORS[m], zorder=3, label=LABELS[m].split('\n')[0])
    for i, v in enumerate(vals):
        add_bar_label(ax, pos[i], v, f'{v:.1f}%', COLORS[m], size=11.5)

ax.set_xticks(x)
ax.set_xticklabels(num_types, fontsize=11.5, fontweight='bold')
style_axis(ax, ymax=55, step=10)
ax.legend(loc='upper left', frameon=False, fontsize=11)

fig2_path = os.path.join(OUT_DIR, 'fig_dot_pha_so_hoc_number.png')
fig.savefig(fig2_path)
plt.close(fig)
print(f"Đã lưu: {fig2_path}")


# -------------------------------------------------------------
# 5. BIỂU ĐỒ 3: PHÂN RÃ TOÀN DIỆN THEO 4 KIỂU ĐÁP ÁN (GROUPED BARS)
# -------------------------------------------------------------
st_labels = {
    'Text-in-Chart': 'Văn bản trục/nhãn\n(Text-in-Chart, n=213)',
    'Text-in-General': 'Văn bản suy diễn\n(Text-in-General, n=46)',
    'Number-in-Chart': 'Số trên biểu đồ\n(Number-in-Chart, n=115)',
    'Number-in-General': 'Số tính toán\n(Number-in-General, n=126)',
}
four_models = ['Gemini', 'Qwen_Direct', 'Standard_CoT', 'Adaptive_LoRA']

fig, ax = plt.subplots(figsize=(12.5, 6.4))
fig.subplots_adjust(top=0.81, bottom=0.20, left=0.06, right=0.98)
add_title(fig, 'Phân rã độ chính xác suy luận theo 4 kiểu đáp án CharXiv',
          'Direct mạnh ở câu hỏi văn bản (tra cứu), Adaptive LoRA bứt phá ở câu hỏi số học và đo đạc biểu đồ')

x = np.arange(len(subtypes))
w = 0.20
for j, m in enumerate(four_models):
    vals = [data[m][data[m].sub_type == s].correct.mean() * 100 for s in subtypes]
    pos = x + (j - 1.5) * w
    short_labels = {
        'Gemini': 'Gemini 2.5 Flash (Oracle)',
        'Qwen_Direct': 'Qwen2.5-VL-7B (Direct)',
        'Standard_CoT': 'Standard CoT',
        'Adaptive_LoRA': 'Adaptive LoRA (Đề xuất)'
    }
    ax.bar(pos, vals, width=w * 0.9, color=COLORS[m], zorder=3, label=short_labels[m])
    for i, v in enumerate(vals):
        add_bar_label(ax, pos[i], v, f'{v:.0f}%', COLORS[m], size=10)

ax.set_xticks(x)
ax.set_xticklabels([st_labels[s] for s in subtypes], fontsize=10.5)
style_axis(ax, ymax=70, step=10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=10.0)

fig3_path = os.path.join(OUT_DIR, 'fig_phan_ra_4_kieu_dap_an.png')
fig.savefig(fig3_path)
plt.close(fig)
print(f"Đã lưu: {fig3_path}")


# -------------------------------------------------------------
# 6. BIỂU ĐỒ 4: ABLATION STUDY — HÀNH TRÌNH TIẾN HÓA CỦA COT
# -------------------------------------------------------------
ablation_stages = [
    ('CoT Ban đầu\n(Chưa chuẩn hóa)', 29.20, '#CBD5E1'),
    ('+ Chuẩn hóa\nWord-to-Num', 36.20, '#94A3B8'),
    ('+ LoRA Alignment\n(Grounded CoT)', 34.80, '#8B5CF6'),
    ('+ Adaptive Routing\n(Đề xuất cuối cùng)', 37.40, '#EA580C'),
    ('Baseline Direct\n(Đối chứng mở)', 38.80, '#0D9488'),
]

fig, ax = plt.subplots(figsize=(10.0, 5.8))
fig.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.96)
add_title(fig, 'Hành trình phát triển Lập luận Đa thể thức (Ablation Progression)',
          'Từ thất bại sụp đổ ban đầu (29.2%) đến hệ thống Adaptive LoRA hoàn thiện (+41 câu đúng)')

x = np.arange(len(ablation_stages))
for i, (name, val, col) in enumerate(ablation_stages):
    ax.bar(i, val, width=0.48, color=col, zorder=3)
    add_bar_label(ax, i, val, f'{val:.1f}%', col, dy=1.2, size=12.5)

# Nối đường xu hướng
vals = [v for _, v, _ in ablation_stages[:4]]
ax.plot(range(4), vals, color='#475569', ls='--', marker='o', lw=1.8, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels([name for name, _, _ in ablation_stages], fontsize=10.5)
style_axis(ax, ymax=45, step=10)

fig4_path = os.path.join(OUT_DIR, 'fig_ablation_tien_trinh_cot.png')
fig.savefig(fig4_path)
plt.close(fig)
print(f"Đã lưu: {fig4_path}")

print("\nToàn bộ 4 biểu đồ và bảng CSV đã được tạo thành công trong thư mục: figures/")
