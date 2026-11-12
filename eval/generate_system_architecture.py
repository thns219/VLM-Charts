# -*- coding: utf-8 -*-
"""generate_system_architecture.py — Tái tạo CHÍNH XÁC 1:1 sơ đồ hệ thống Adaptive-VLM-Charts
theo đúng cấu trúc, vị trí, nhãn và thẩm mỹ của hình thiết kế chuẩn từ người dùng.

Bố cục 3 Panel:
  1. Panel trái: "Adaptive-VLM-Charts: input pipeline"
     - Bộ dữ liệu (CharXiv / SLAKE) [nền xanh nhạt]
     - Mẫu đầu vào: (I, Q), I hoặc Q [Image I (tùy chọn nét đứt), Question Q (tùy chọn nét đứt) -> Phân loại D1-D4 / R1-R4]
     - Minh họa Bar chart xanh/vàng trong khung nét đứt màu cam + Question
  2. Panel trên phải: "Adaptive-VLM-Charts: retrieval, routing & reasoning pipeline"
     - Input (I và/hoặc Q) -> Hybrid RAG (ChromaDB, CLIP, BM25)
     - Đường Oracle (cận trên) / Direct Bypass [nét đứt] đi vào Prompt D
     - Prompt D -> Lightweight Classifier & Dynamic Router (A: Visual/Label, B: Numerical/Arithmetic)
     - VLM Reasoning Engine (Qwen2.5-VL-7B) [khung vàng nét đứt]:
         (a) Direct Zero-Shot (bypass CoT)
         (b) Standard CoT (baseline không routing) [nét đứt]
         (c) Adaptive LoRA CoT (3-stage alignment) [nền vàng cam]
     - Hội tụ về Unified Output: Answer A (* Rationale R nếu có)
  3. Panel dưới phải: "Adaptive-VLM-Charts: evaluation pipeline"
     - Answer A (+ Rationale R) -> Chuẩn hóa (Word-to-Number, Bỏ LaTeX, Medical Synonyms)
     - Khối CharXiv (chấm điểm theo 4 nhóm: Text/Number in Chart/General với match_fixed)
     - Khối SLAKE (Closed Strict/Medical Acc, Open NLG BLEU/ROUGE, Open Strict/Medical Sem Acc)
  4. Caption học thuật đầy đủ ở dưới cùng.
"""

import os
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE) if os.path.basename(HERE) == "eval" else HERE
OUT_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# CẤU HÌNH THẨM MỸ HỌC THUẬT (CHUẨN MINH HỌA BÁO CÁO)
# -------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.facecolor': 'white'
})

# Kích thước khung vẽ (16:9 widescreen rộng thoáng)
fig = plt.figure(figsize=(18.0, 9.8), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Màu sắc chuẩn mực theo thiết kế
C_FRAME = "#0F172A"       # Viền khung chính màu đen xám đậm
C_BOX_BORDER = "#1E293B"  # Viền các hộp con
C_BLUE_BG = "#E0F2FE"     # Nền xanh nhạt cho Bộ dữ liệu
C_BLUE_BORDER = "#38BDF8" # Viền xanh cho Bộ dữ liệu
C_GOLD = "#F59E0B"        # Cam vàng cho khung nét đứt & LoRA CoT
C_YELLOW_BG = "#FEF3C7"   # Vàng nhạt cho LoRA CoT
C_BAR_BLUE = "#2563EB"    # Cột xanh biểu đồ
C_BAR_YELLOW = "#FBBF24"  # Cột vàng biểu đồ


def draw_box(x, y, w, h, bg_color="white", border_color=C_BOX_BORDER, radius=0.8, lw=1.1, ls="-", zorder=2):
    rect = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.15,rounding_size={radius}",
                          facecolor=bg_color, edgecolor=border_color, linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(rect)
    return rect


def draw_arrow(x1, y1, x2, y2, color="#0F172A", style="-|>", lw=1.2, ls="-", zorder=5):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=ArrowStyle(style, head_length=3.5, head_width=2.2),
                          color=color, linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(arr)
    return arr


# ==============================================================================
# 1. PANEL 1: ADAPTIVE-VLM-CHARTS: INPUT PIPELINE (BÊN TRÁI)
# ==============================================================================
# Khung bao Panel 1
draw_box(2.0, 8.5, 27.5, 87.5, bg_color="white", border_color=C_FRAME, radius=1.2, lw=1.3, zorder=1)
ax.text(15.75, 93.0, "Adaptive-VLM-Charts: input pipeline", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="#0F172A", zorder=3)

# Khối: Bộ dữ liệu (CharXiv / SLAKE)
draw_box(3.5, 81.5, 24.5, 8.5, bg_color=C_BLUE_BG, border_color=C_BLUE_BORDER, radius=0.6, lw=1.1, zorder=2)
ax.text(15.75, 86.8, "Bộ dữ liệu", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0F172A", zorder=3)
ax.text(15.75, 83.8, "(CharXiv / SLAKE)", ha="center", va="center", fontsize=8.2, color="#334155", zorder=3)

# Mũi tên từ Bộ dữ liệu xuống Mẫu đầu vào
draw_arrow(15.75, 81.5, 15.75, 75.5)

# Khối lớn: Mẫu đầu vào: (I, Q), I hoặc Q
draw_box(3.5, 52.0, 24.5, 23.5, bg_color="white", border_color=C_BOX_BORDER, radius=0.6, lw=1.1, zorder=2)
ax.text(15.75, 72.5, "Mẫu đầu vào: (I, Q), I hoặc Q", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#0F172A", zorder=3)

# Ô con 1: Image I (tùy chọn) - Nét đứt
draw_box(4.8, 62.5, 9.8, 8.0, bg_color="white", border_color="#475569", radius=0.4, ls="--", lw=1.1, zorder=3)
ax.text(9.7, 67.5, "Image I", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A", zorder=4)
ax.text(9.7, 64.5, "(tùy chọn)", ha="center", va="center", fontsize=7.2, style="italic", color="#64748B", zorder=4)

# Ô con 2: Question Q (tùy chọn) - Nét đứt
draw_box(4.8, 53.5, 9.8, 8.0, bg_color="white", border_color="#475569", radius=0.4, ls="--", lw=1.1, zorder=3)
ax.text(9.7, 58.5, "Question Q", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A", zorder=4)
ax.text(9.7, 55.5, "(tùy chọn)", ha="center", va="center", fontsize=7.2, style="italic", color="#64748B", zorder=4)

# Dấu móc ngoặc / mũi tên gom từ I & Q sang Phân loại
ax.plot([15.2, 16.0, 16.0], [66.5, 66.5, 62.0], color="#475569", lw=1.0, zorder=4)
ax.plot([15.2, 16.0, 16.0], [57.5, 57.5, 62.0], color="#475569", lw=1.0, zorder=4)
draw_arrow(16.0, 62.0, 17.5, 62.0, color="#475569", lw=1.0)

# Ô con 3: Phân loại D1–D4 / R1–R4
draw_box(17.5, 53.5, 9.2, 17.0, bg_color="white", border_color=C_BOX_BORDER, radius=0.4, lw=1.1, zorder=3)
ax.text(22.1, 65.5, "Phân loại", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0F172A", zorder=4)
ax.text(22.1, 60.5, "D1–D4", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1E293B", zorder=4)
ax.text(22.1, 56.5, "R1–R4", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1E293B", zorder=4)

# Khung nét đứt cam/vàng: Minh họa Biểu đồ & Câu hỏi
draw_box(3.5, 10.5, 24.5, 39.5, bg_color="white", border_color=C_GOLD, radius=0.6, ls="--", lw=1.2, zorder=2)

# Trục tọa độ và các cột Bar Chart
ax.plot([5.5, 26.0], [24.0, 24.0], color="#94A3B8", lw=1.0, zorder=3) # Trục X
ax.plot([5.5, 5.5], [24.0, 46.0], color="#94A3B8", lw=1.0, zorder=3) # Trục Y

# Cột 1
ax.bar([8.5], [11.0], width=2.2, bottom=24.0, color=C_BAR_BLUE, edgecolor="#1D4ED8", lw=0.7, zorder=4)
ax.bar([11.0], [7.0], width=2.2, bottom=24.0, color=C_BAR_YELLOW, edgecolor="#D97706", lw=0.7, zorder=4)

# Cột 2
ax.bar([14.8], [17.0], width=2.2, bottom=24.0, color=C_BAR_BLUE, edgecolor="#1D4ED8", lw=0.7, zorder=4)
ax.bar([17.3], [14.0], width=2.2, bottom=24.0, color=C_BAR_YELLOW, edgecolor="#D97706", lw=0.7, zorder=4)

# Cột 3
ax.bar([21.2], [9.5], width=2.2, bottom=24.0, color=C_BAR_BLUE, edgecolor="#1D4ED8", lw=0.7, zorder=4)
ax.bar([23.7], [6.5], width=2.2, bottom=24.0, color=C_BAR_YELLOW, edgecolor="#D97706", lw=0.7, zorder=4)

# Câu hỏi minh họa phía dưới biểu đồ
ax.text(15.75, 17.0, "Question: Hiệu giữa tổng các cột xanh và tổng\ncác cột vàng là bao nhiêu?",
        ha="center", va="center", fontsize=7.8, fontweight="bold", color="#0F172A", zorder=4)


# ==============================================================================
# 2. PANEL 2: RETRIEVAL, ROUTING & REASONING PIPELINE (TRÊN BÊN PHẢI)
# ==============================================================================
# Khung bao Panel 2
draw_box(31.0, 52.0, 67.0, 44.0, bg_color="white", border_color=C_FRAME, radius=1.2, lw=1.3, zorder=1)
ax.text(64.5, 93.0, "Adaptive-VLM-Charts: retrieval, routing & reasoning pipeline", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="#0F172A", zorder=3)

# Mũi tên nối từ Panel 1 sang Panel 2: Từ Mẫu đầu vào sang Input (I và/hoặc Q)
draw_arrow(28.0, 63.75, 32.5, 68.0, color="#0F172A", lw=1.3)

# Khối: Input (I và/hoặc Q)
draw_box(32.5, 63.5, 6.2, 9.0, bg_color="white", border_color=C_BOX_BORDER, radius=0.4, lw=1.1, zorder=2)
ax.text(35.6, 69.2, "Input", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A", zorder=3)
ax.text(35.6, 66.0, "I và/hoặc Q", ha="center", va="center", fontsize=7.2, color="#334155", zorder=3)

# Đường Oracle (cận trên) / Direct Bypass (nét đứt) chạy vòng lên trên RAG
ax.plot([35.6, 35.6, 56.5], [72.5, 85.0, 85.0], color="#334155", lw=1.1, ls="--", zorder=4)
draw_arrow(56.5, 85.0, 56.5, 74.0, color="#334155", style="-|>", lw=1.1, ls="--")
ax.text(46.0, 86.5, "Oracle (cận trên) / Direct Bypass", ha="center", va="bottom",
        fontsize=7.8, style="italic", fontweight="bold", color="#1E293B", zorder=5)

# Mũi tên từ Input vào Hybrid RAG
draw_arrow(38.7, 68.0, 40.5, 68.0)

# Khối: Hybrid RAG
draw_box(40.5, 57.5, 11.5, 21.0, bg_color="white", border_color=C_BOX_BORDER, radius=0.5, lw=1.1, zorder=2)
ax.text(46.25, 75.5, "Hybrid RAG", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0F172A", zorder=3)
ax.text(46.25, 71.8, "ChromaDB", ha="center", va="center", fontsize=7.8, color="#1E293B", zorder=3)
ax.text(46.25, 68.5, "CLIP ViT-L/14", ha="center", va="center", fontsize=7.6, color="#1E293B", zorder=3)
ax.text(46.25, 65.5, "(Letterbox)", ha="center", va="center", fontsize=7.2, color="#475569", zorder=3)
ax.text(46.25, 62.5, "BM25 Caption", ha="center", va="center", fontsize=7.6, color="#1E293B", zorder=3)
ax.text(46.25, 59.5, "(Query: text Q/I)", ha="center", va="center", fontsize=6.8, style="italic", color="#64748B", zorder=3)

# Mũi tên từ Hybrid RAG sang Prompt D
draw_arrow(52.0, 68.0, 54.0, 68.0)

# Khối: Prompt D
draw_box(54.0, 64.0, 5.0, 8.0, bg_color="white", border_color=C_BOX_BORDER, radius=0.4, lw=1.1, zorder=2)
ax.text(56.5, 69.2, "Prompt", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A", zorder=3)
ax.text(56.5, 66.0, "D", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#0F172A", zorder=3)

# Mũi tên từ Prompt D sang Router
draw_arrow(59.0, 68.0, 61.0, 68.0)

# Khối: Lightweight Classifier & Dynamic Router
draw_box(61.0, 57.5, 13.0, 21.0, bg_color="white", border_color=C_BOX_BORDER, radius=0.5, lw=1.2, zorder=2)
ax.text(67.5, 75.8, "Lightweight Classifier", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#0F172A", zorder=3)
ax.text(67.5, 73.2, "& Dynamic Router", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#0F172A", zorder=3)
ax.text(67.5, 69.5, "Input: câu hỏi Q (trong D)", ha="center", va="center", fontsize=6.8, color="#475569", zorder=3)
ax.text(67.5, 66.2, "A → Visual / Label reading", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#0D9488", zorder=3)
ax.text(67.5, 63.8, "(hoặc Image-only)", ha="center", va="center", fontsize=6.5, style="italic", color="#0D9488", zorder=3)
ax.text(67.5, 60.5, "B → Numerical / Arithmetic", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#C2410C", zorder=3)

# Khung nét đứt cam/vàng: VLM Reasoning Engine (Qwen2.5-VL-7B)
draw_box(76.0, 54.0, 13.5, 33.0, bg_color="white", border_color=C_GOLD, radius=0.6, ls="--", lw=1.2, zorder=2)
ax.text(82.75, 84.5, "VLM Reasoning Engine", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0F172A", zorder=3)
ax.text(82.75, 82.2, "(Qwen2.5-VL-7B)", ha="center", va="center", fontsize=7.2, color="#475569", zorder=3)

# (a) Direct Zero-Shot (bypass CoT)
draw_box(77.0, 74.0, 11.5, 6.5, bg_color="white", border_color=C_BOX_BORDER, radius=0.4, lw=1.1, zorder=3)
ax.text(82.75, 78.2, "(a) Direct Zero-Shot", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#0F172A", zorder=4)
ax.text(82.75, 75.8, "bypass CoT", ha="center", va="center", fontsize=6.8, color="#475569", zorder=4)

# (b) Standard CoT (baseline không routing) - Khung nét đứt
draw_box(77.0, 64.5, 11.5, 6.5, bg_color="white", border_color="#64748B", radius=0.4, ls="--", lw=1.0, zorder=3)
ax.text(82.75, 68.7, "(b) Standard CoT", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#475569", zorder=4)
ax.text(82.75, 66.3, "baseline (không routing)", ha="center", va="center", fontsize=6.8, color="#64748B", zorder=4)

# (c) Adaptive LoRA CoT (3-stage alignment) - Nền vàng cam
draw_box(77.0, 55.5, 11.5, 7.0, bg_color=C_YELLOW_BG, border_color="#D97706", radius=0.4, lw=1.2, zorder=3)
ax.text(82.75, 60.5, "(c) Adaptive LoRA", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#9A3412", zorder=4)
ax.text(82.75, 58.5, "CoT", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#9A3412", zorder=4)
ax.text(82.75, 56.7, "3-stage alignment", ha="center", va="center", fontsize=6.8, color="#9A3412", zorder=4)

# Hai mũi tên điều phối từ Router:
# Nhánh A lên (a) Direct Zero-Shot
draw_arrow(74.0, 71.0, 77.0, 77.0, color="#0D9488", lw=1.3)
# Nhánh B xuống (c) Adaptive LoRA CoT
draw_arrow(74.0, 65.0, 77.0, 59.0, color="#C2410C", lw=1.3)

# Khối: Unified Output (Answer A, * Rationale R nếu có)
draw_box(91.5, 63.0, 5.5, 15.0, bg_color="white", border_color=C_BOX_BORDER, radius=0.4, lw=1.1, zorder=2)
ax.text(94.25, 74.5, "Unified Output", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0F172A", zorder=3)
ax.text(94.25, 70.0, "Answer A", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#1E293B", zorder=3)
ax.text(94.25, 66.0, "(* Rationale R\nnếu có)", ha="center", va="center", fontsize=6.6, color="#475569", zorder=3)

# Mũi tên từ 3 chế độ hội tụ về Unified Output
draw_arrow(88.5, 77.0, 91.5, 73.0, color="#0F172A", lw=1.0)
draw_arrow(88.5, 67.75, 91.5, 70.0, color="#64748B", ls="--", lw=0.9)
draw_arrow(88.5, 59.0, 91.5, 67.0, color="#0F172A", lw=1.0)


# ==============================================================================
# 3. PANEL 3: EVALUATION PIPELINE (DƯỚI BÊN PHẢI)
# ==============================================================================
# Khung bao Panel 3
draw_box(31.0, 8.5, 67.0, 41.5, bg_color="white", border_color=C_FRAME, radius=1.2, lw=1.3, zorder=1)
ax.text(64.5, 47.5, "Adaptive-VLM-Charts: evaluation pipeline", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="#0F172A", zorder=3)

# Mũi tên từ Unified Output của Panel 2 chạy xuống Panel 3
ax.plot([94.25, 94.25, 36.5, 36.5], [63.0, 48.0, 48.0, 37.0], color="#0F172A", lw=1.2, zorder=4)
draw_arrow(36.5, 37.0, 36.5, 33.5, color="#0F172A", lw=1.2)

# Khối: Answer A (+ Rationale R)
draw_box(32.5, 23.5, 8.0, 10.0, bg_color="white", border_color=C_BOX_BORDER, radius=0.4, lw=1.1, zorder=2)
ax.text(36.5, 29.5, "Answer A", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A", zorder=3)
ax.text(36.5, 26.5, "(+ Rationale R)", ha="center", va="center", fontsize=7.2, color="#475569", zorder=3)

# Mũi tên sang Chuẩn hóa
draw_arrow(40.5, 28.5, 42.5, 28.5)

# Khối: Chuẩn hóa (Word-to-Number, Bỏ LaTeX, Medical Synonyms)
draw_box(42.5, 21.0, 12.0, 15.0, bg_color="white", border_color=C_BOX_BORDER, radius=0.5, lw=1.1, zorder=2)
ax.text(48.5, 32.5, "Chuẩn hóa", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0F172A", zorder=3)
ax.text(48.5, 29.0, "Word-to-Number", ha="center", va="center", fontsize=7.5, color="#1E293B", zorder=3)
ax.text(48.5, 26.2, "Bỏ LaTeX", ha="center", va="center", fontsize=7.5, color="#1E293B", zorder=3)
ax.text(48.5, 23.5, "Medical Synonyms", ha="center", va="center", fontsize=7.2, color="#1E293B", zorder=3)

# Mũi tên rẽ từ Chuẩn hóa sang 2 khung đánh giá (CharXiv và SLAKE)
ax.plot([54.5, 56.5, 56.5], [28.5, 28.5, 36.5], color="#0F172A", lw=1.1, zorder=4)
draw_arrow(56.5, 36.5, 58.0, 36.5, color="#0F172A", lw=1.1)

ax.plot([54.5, 56.5, 56.5], [28.5, 28.5, 18.5], color="#0F172A", lw=1.1, zorder=4)
draw_arrow(56.5, 18.5, 58.0, 18.5, color="#0F172A", lw=1.1)

# -------------------------------------------------------------
# KHỐI CHARXIV: CHẤM ĐIỂM THEO 4 NHÓM CÂU HỎI
# -------------------------------------------------------------
draw_box(58.0, 27.0, 38.5, 18.5, bg_color="white", border_color=C_BOX_BORDER, radius=0.5, lw=1.1, zorder=2)
ax.text(77.25, 43.2, "CharXiv (biểu đồ khoa học): chấm điểm theo 4 nhóm câu hỏi",
        ha="center", va="center", fontsize=8.2, fontweight="bold", color="#0F172A", zorder=3)

# match_fixed / Bootstrap 95% CI
draw_box(59.2, 28.8, 11.5, 11.5, bg_color="white", border_color=C_BOX_BORDER, radius=0.4, lw=1.0, zorder=3)
ax.text(64.95, 35.8, "match_fixed", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#0F172A", zorder=4)
ax.text(64.95, 32.5, "Bootstrap 95% CI", ha="center", va="center", fontsize=7.2, color="#475569", zorder=4)

# Mũi tên từ match_fixed rẽ sang 4 nhóm
ax.plot([70.7, 72.5, 72.5], [34.5, 34.5, 37.0], color="#0F172A", lw=1.0, zorder=4)
draw_arrow(72.5, 37.0, 74.0, 37.0, color="#0F172A", lw=1.0)
ax.plot([70.7, 72.5, 72.5], [34.5, 34.5, 31.5], color="#0F172A", lw=1.0, zorder=4)
draw_arrow(72.5, 31.5, 74.0, 31.5, color="#0F172A", lw=1.0)

# 4 ô con nhóm câu hỏi: Text-in-Chart, Text-in-General, Number-in-Chart, Number-in-General
draw_box(74.0, 34.8, 10.5, 4.8, bg_color="white", border_color=C_BOX_BORDER, radius=0.3, lw=0.9, zorder=3)
ax.text(79.25, 37.2, "Text-in-Chart", ha="center", va="center", fontsize=7.5, color="#1E293B", zorder=4)

draw_box(85.2, 34.8, 10.2, 4.8, bg_color="white", border_color=C_BOX_BORDER, radius=0.3, lw=0.9, zorder=3)
ax.text(90.3, 37.2, "Text-in-General", ha="center", va="center", fontsize=7.5, color="#1E293B", zorder=4)

draw_box(74.0, 29.0, 10.5, 4.8, bg_color="white", border_color=C_BOX_BORDER, radius=0.3, lw=0.9, zorder=3)
ax.text(79.25, 31.4, "Number-in-Chart", ha="center", va="center", fontsize=7.5, color="#1E293B", zorder=4)

draw_box(85.2, 29.0, 10.2, 4.8, bg_color="white", border_color=C_BOX_BORDER, radius=0.3, lw=0.9, zorder=3)
ax.text(90.3, 31.4, "Number-in-General", ha="center", va="center", fontsize=7.5, color="#1E293B", zorder=4)

# -------------------------------------------------------------
# KHỐI SLAKE: ACCURACY & NLG
# -------------------------------------------------------------
draw_box(58.0, 10.5, 38.5, 15.0, bg_color="white", border_color=C_BOX_BORDER, radius=0.5, lw=1.1, zorder=2)
ax.text(77.25, 23.6, "SLAKE (Y khoa): Accuracy & NLG",
        ha="center", va="center", fontsize=8.2, fontweight="bold", color="#0F172A", zorder=3)

# 3 ô con SLAKE: Closed, Open NLG, Open
draw_box(59.2, 12.2, 11.8, 8.8, bg_color="white", border_color=C_BOX_BORDER, radius=0.3, lw=0.9, zorder=3)
ax.text(65.1, 18.0, "Closed:", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#0F172A", zorder=4)
ax.text(65.1, 15.0, "Strict Acc, Medical Acc", ha="center", va="center", fontsize=6.8, color="#334155", zorder=4)

draw_box(71.8, 12.2, 12.4, 8.8, bg_color="white", border_color=C_BOX_BORDER, radius=0.3, lw=0.9, zorder=3)
ax.text(78.0, 18.0, "Open NLG:", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#0F172A", zorder=4)
ax.text(78.0, 15.0, "BLEU-1, 2 | ROUGE-1, L", ha="center", va="center", fontsize=6.8, color="#334155", zorder=4)

draw_box(85.0, 12.2, 10.4, 8.8, bg_color="white", border_color=C_BOX_BORDER, radius=0.3, lw=0.9, zorder=3)
ax.text(90.2, 18.0, "Open:", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#0F172A", zorder=4)
ax.text(90.2, 15.0, "Strict EM, Medical Sem. Acc", ha="center", va="center", fontsize=6.4, color="#334155", zorder=4)


# ==============================================================================
# 4. CAPTION BÀI BÁO TẠP CHÍ Ở DƯỚI CÙNG (CHÍNH XÁC NGUYÊN VĂN TỪ HÌNH MẪU)
# ==============================================================================
caption_text = (
    "Hình 1. Tổng quan Adaptive-VLM-Charts. Đầu vào gồm ảnh I và/hoặc câu hỏi Q (nét đứt: tùy chọn); đầu vào chỉ có Q được truy hồi ảnh qua Hybrid RAG. "
    "Router nhận văn bản câu hỏi Q trong prompt D: câu hỏi số học → Adaptive LoRA CoT; câu hỏi đọc nhãn hoặc đầu vào chỉ có ảnh → Direct Zero-Shot "
    "(Standard CoT là baseline; đường Oracle / Direct Bypass là thiết lập cận trên). Đầu ra thống nhất được chuẩn hóa rồi chấm điểm match_fixed (CharXiv) "
    "và BLEU, ROUGE (SLAKE); Adaptive Dynamic Routing đạt 39.60% (CharXiv) và 52.59% (SLAKE)."
)

ax.text(50.0, 3.2, caption_text, ha="center", va="center",
        fontsize=7.5, color="#1E293B", linespacing=1.35, zorder=3)


# Lưu file đầu ra
out_png = os.path.join(OUT_DIR, "system_architecture_redesign.png")
out_svg = os.path.join(OUT_DIR, "system_architecture_redesign.svg")

plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.savefig(out_svg, format='svg', bbox_inches='tight', pad_inches=0.08)
plt.close(fig)

print("ĐÃ VẼ LẠI CHÍNH XÁC 1:1 THEO HÌNH ẢNH MẪU CỦA BẠN:")
print(f"1. PNG: {out_png}")
print(f"2. SVG: {out_svg}")
