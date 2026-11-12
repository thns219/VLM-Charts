# -*- coding: utf-8 -*-
"""generate_system_architecture.py — Tái tạo CHÍNH XÁC 1:1 cấu trúc sơ đồ hệ thống
Adaptive-VLM-Charts theo chuẩn tạp chí khoa học quốc tế (IEEE/ACM style).

Thiết kế tối ưu:
  - Pipeline 1: Input Pipeline (Dữ liệu CQA, mẫu (I, Q), phân loại, biểu đồ thanh mini)
  - Pipeline 2: Retrieval & Reasoning Pipeline (Hybrid RAG, Oracle Bypass, VLM 3 chế độ suy luận)
  - Pipeline 3: Evaluation Pipeline tinh gọn theo cơ chế (Chuẩn hóa, Đánh giá CharXiv, Đánh giá SLAKE)
  - Khung ví dụ minh họa 3 bước và Caption chuẩn mực ở phía dưới.
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
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.facecolor': 'white'
})

# Bảng màu trang nhã, thanh thoát đúng chất tạp chí
C_BORDER = "#94A3B8"         # Viền xám trung tính mảnh
C_BOX_BG = "#FFFFFF"         # Trắng tinh khiết
C_HEADER_BG = "#E2E8F0"      # Xám sáng thanh lịch cho thanh tiêu đề 3 pipeline
C_TEXT_MAIN = "#1E293B"      # Chữ đen xám đậm dễ đọc
C_TEXT_MUTED = "#64748B"     # Chữ xám phụ

# Màu điểm nhấn ngữ nghĩa
C_BLUE = "#2563EB"           # Xanh cột Right / CharXiv
C_YELLOW = "#EAB308"         # Vàng cột Centre
C_ORANGE = "#EA580C"         # Cam cho Adaptive LoRA CoT (Đề xuất)
C_TEAL = "#0D9488"           # Xanh ngọc cho Oracle Bypass
C_PURPLE = "#7C3AED"         # Tím cho Hybrid RAG
C_GREEN = "#059669"          # Xanh lục cho SLAKE Med-VQA

fig = plt.figure(figsize=(18, 9.8), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')


def draw_box(x, y, w, h, bg_color=C_BOX_BG, border_color=C_BORDER, radius=0.6, lw=1.1, ls="-", zorder=2):
    rect = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.15,rounding_size={radius}",
                          facecolor=bg_color, edgecolor=border_color, linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(rect)
    return rect


def draw_arrow(x1, y1, x2, y2, color="#475569", style="-|>", lw=1.3, ls="-", zorder=5):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=ArrowStyle(style, head_length=3.5, head_width=2.2),
                          color=color, linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(arr)
    return arr


# =============================================================
# 1. PIPELINE 1: ADAPTIVE-VLM-CHARTS: INPUT PIPELINE (TRÁI)
# =============================================================
draw_box(2.0, 15.5, 29.0, 78.5, bg_color="#F8FAFC", border_color="#CBD5E1", radius=0.8, lw=1.2, zorder=1)

# Thanh Header Panel 1
draw_box(2.8, 88.5, 27.4, 4.2, bg_color=C_HEADER_BG, border_color="#CBD5E1", radius=0.4, zorder=2)
ax.text(16.5, 90.6, "Adaptive-VLM-Charts: input pipeline", ha="center", va="center",
        fontsize=11, fontweight="bold", color=C_TEXT_MAIN, zorder=3)

# Khối: Bộ dữ liệu CQA (Scientific Chart / SLAKE)
draw_box(4.5, 78.5, 24.0, 7.5, bg_color="white", border_color=C_BORDER, radius=0.5, zorder=2)
ax.text(16.5, 83.2, "Bộ dữ liệu CQA", ha="center", va="center", fontsize=10.5, fontweight="bold", color=C_TEXT_MAIN, zorder=3)
ax.text(16.5, 80.5, "(Scientific Chart / SLAKE)", ha="center", va="center", fontsize=9, color=C_TEXT_MUTED, zorder=3)

# Mũi tên xuống Mẫu đầu vào
draw_arrow(16.5, 78.5, 16.5, 74.5)

# Khối: Mẫu đầu vào (I, Q)
draw_box(4.5, 61.5, 24.0, 13.0, bg_color="white", border_color=C_BORDER, radius=0.5, zorder=2)
ax.text(16.5, 71.8, "Mẫu đầu vào (I, Q)", ha="center", va="center", fontsize=10, fontweight="bold", color=C_TEXT_MAIN, zorder=3)

# 2 ô con: Image I & Question Q
draw_box(6.0, 63.2, 10.0, 5.8, bg_color="#F1F5F9", border_color="#94A3B8", radius=0.3, zorder=3)
ax.text(11.0, 66.1, "Image I", ha="center", va="center", fontsize=9, fontweight="bold", color=C_TEXT_MAIN, zorder=4)

draw_box(17.0, 63.2, 10.0, 5.8, bg_color="#F1F5F9", border_color="#94A3B8", radius=0.3, zorder=3)
ax.text(22.0, 66.1, "Question Q", ha="center", va="center", fontsize=9, fontweight="bold", color=C_TEXT_MAIN, zorder=4)

# Mũi tên xuống Phân loại
draw_arrow(16.5, 61.5, 16.5, 57.5)

# Khối: Phân loại D1–D4, R1–R4
draw_box(6.5, 52.0, 20.0, 5.5, bg_color="white", border_color=C_BORDER, radius=0.4, zorder=2)
ax.text(10.5, 54.75, "Phân loại", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_TEXT_MAIN, zorder=3)
draw_box(14.5, 52.8, 5.2, 3.8, bg_color="#E0E7FF", border_color="#818CF8", radius=0.3, zorder=3)
ax.text(17.1, 54.7, "D1–D4", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#3730A3", zorder=4)
draw_box(20.3, 52.8, 5.2, 3.8, bg_color="#EDE9FE", border_color="#A78BFA", radius=0.3, zorder=3)
ax.text(22.9, 54.7, "R1–R4", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#5B21B6", zorder=4)

# Khối: Minh họa biểu đồ Bar Chart
draw_box(4.5, 17.5, 24.0, 32.0, bg_color="white", border_color="#CBD5E1", radius=0.5, zorder=2)
ax.text(16.5, 47.0, "■ xanh (Right)   □ vàng (Centre)", ha="center", va="center",
        fontsize=8.8, fontweight="bold", color=C_TEXT_MAIN, zorder=3)

# Vẽ 3 cặp cột minh họa
ax.plot([7.0, 26.0], [28.0, 28.0], color="#94A3B8", lw=1.1, zorder=3)
ax.plot([7.0, 7.0], [28.0, 44.0], color="#94A3B8", lw=1.1, zorder=3)

# Cặp cột 1: Xanh 32, Vàng 55
ax.bar([10.0], [7.0], width=1.5, bottom=28.0, color=C_BLUE, zorder=4)
ax.bar([11.8], [12.0], width=1.5, bottom=28.0, color=C_YELLOW, edgecolor="#CA8A04", lw=0.8, zorder=4)

# Cặp cột 2: Xanh 38, Vàng 27
ax.bar([15.5], [8.5], width=1.5, bottom=28.0, color=C_BLUE, zorder=4)
ax.bar([17.3], [6.0], width=1.5, bottom=28.0, color=C_YELLOW, edgecolor="#CA8A04", lw=0.8, zorder=4)

# Cặp cột 3: Xanh 30, Vàng 18
ax.bar([21.0], [6.6], width=1.5, bottom=28.0, color=C_BLUE, zorder=4)
ax.bar([22.8], [4.0], width=1.5, bottom=28.0, color=C_YELLOW, edgecolor="#CA8A04", lw=0.8, zorder=4)

# Câu hỏi minh họa phía dưới biểu đồ
draw_box(5.5, 19.0, 22.0, 7.5, bg_color="#FEF3C7", border_color="#F59E0B", radius=0.3, zorder=3)
ax.text(16.5, 22.75, "Question: Hiệu giữa tổng các cột xanh\nvà tổng các cột vàng là bao nhiêu?",
        ha="center", va="center", fontsize=8.2, fontweight="bold", color="#92400E", zorder=4)


# =============================================================
# 2. PIPELINE 2: RETRIEVAL & REASONING PIPELINE (Ở GIỮA)
# =============================================================
draw_box(32.5, 15.5, 39.5, 78.5, bg_color="#F8FAFC", border_color="#CBD5E1", radius=0.8, lw=1.2, zorder=1)

# Thanh Header Panel 2
draw_box(33.3, 88.5, 37.9, 4.2, bg_color=C_HEADER_BG, border_color="#CBD5E1", radius=0.4, zorder=2)
ax.text(52.25, 90.6, "Adaptive-VLM-Charts: retrieval & reasoning pipeline", ha="center", va="center",
        fontsize=11, fontweight="bold", color=C_TEXT_MAIN, zorder=3)

# Mũi tên từ Panel 1 sang Panel 2: Mẫu đầu vào (I, Q) -> Input (I, Q)
draw_arrow(28.5, 68.0, 35.0, 68.0, color=C_BLUE, lw=1.6)

# Khối: Input (I, Q)
draw_box(35.0, 64.5, 9.5, 7.0, bg_color="white", border_color=C_BORDER, radius=0.4, zorder=2)
ax.text(39.75, 68.0, "Input\n(I, Q)", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_TEXT_MAIN, zorder=3)

# Nhánh 1: Rẽ lên Hybrid RAG
draw_arrow(44.5, 68.0, 48.0, 74.0, color="#475569", lw=1.3)

# Khung nét đứt lớn: Hybrid RAG
draw_box(47.5, 65.5, 23.0, 20.0, bg_color="#FAF5FF", border_color=C_PURPLE, ls="--", lw=1.3, radius=0.6, zorder=2)
ax.text(59.0, 83.2, "Hybrid RAG", ha="center", va="center", fontsize=10, fontweight="bold", color=C_PURPLE, zorder=3)

# 3 thành phần bên trong Hybrid RAG: ChromaDB, CLIP, BM25
draw_box(49.0, 75.0, 20.0, 5.5, bg_color="white", border_color="#C084FC", radius=0.4, zorder=3)
ax.text(59.0, 77.75, "ChromaDB", ha="center", va="center", fontsize=9, fontweight="bold", color=C_TEXT_MAIN, zorder=4)

draw_box(49.0, 68.0, 9.5, 5.5, bg_color="white", border_color="#C084FC", radius=0.4, zorder=3)
ax.text(53.75, 70.75, "CLIP ViT-L/14\n(Letterbox)", ha="center", va="center", fontsize=7.8, fontweight="bold", color=C_TEXT_MAIN, zorder=4)

draw_box(59.5, 68.0, 9.5, 5.5, bg_color="white", border_color="#C084FC", radius=0.4, zorder=3)
ax.text(64.25, 70.75, "BM25\nCaption", ha="center", va="center", fontsize=8.2, fontweight="bold", color=C_TEXT_MAIN, zorder=4)

# Mũi tên từ Hybrid RAG đi xuống Prompt D
draw_arrow(59.0, 65.5, 59.0, 60.5, color=C_PURPLE, lw=1.4)

# Nhánh 2: Đường Oracle / Direct Bypass (Rẽ xuống vòng qua Hybrid RAG)
ax.plot([39.75, 39.75, 54.0], [64.5, 58.0, 58.0], color=C_TEAL, lw=1.4, ls="-", zorder=4)
draw_arrow(54.0, 58.0, 55.5, 58.0, color=C_TEAL, lw=1.4)
ax.text(46.0, 59.3, "Oracle / Direct Bypass", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=C_TEAL, zorder=5)

# Khối: Prompt D
draw_box(55.5, 55.5, 8.5, 5.0, bg_color="#FEF9C3", border_color="#EAB308", radius=0.4, zorder=3)
ax.text(59.75, 58.0, "Prompt D", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#854D0E", zorder=4)

# Mũi tên từ Prompt D xuống VLM
draw_arrow(59.75, 55.5, 59.75, 51.0, color="#475569", lw=1.4)

# Khung lớn: VLM Qwen2.5-VL-7B
draw_box(35.0, 17.5, 35.0, 33.5, bg_color="white", border_color=C_BORDER, radius=0.6, lw=1.2, zorder=2)
ax.text(52.5, 48.0, "VLM", ha="center", va="center", fontsize=11, fontweight="bold", color=C_TEXT_MAIN, zorder=3)
ax.text(52.5, 45.2, "Qwen2.5-VL-7B", ha="center", va="center", fontsize=9.2, color=C_TEXT_MUTED, zorder=3)

# 3 chế độ VLM
draw_box(37.0, 36.5, 31.0, 6.2, bg_color="#F8FAFC", border_color="#CBD5E1", radius=0.4, zorder=3)
ax.text(52.5, 39.6, "(a) Direct Zero-Shot", ha="center", va="center", fontsize=9, fontweight="bold", color=C_TEXT_MAIN, zorder=4)

draw_box(37.0, 28.5, 31.0, 6.2, bg_color="#F8FAFC", border_color="#CBD5E1", radius=0.4, zorder=3)
ax.text(52.5, 31.6, "(b) Standard CoT (2-stage)", ha="center", va="center", fontsize=9, fontweight="bold", color=C_TEXT_MAIN, zorder=4)

draw_box(37.0, 19.5, 31.0, 7.2, bg_color="#FFF7ED", border_color=C_ORANGE, lw=1.4, radius=0.4, zorder=3)
ax.text(52.5, 23.1, "(c) Adaptive LoRA CoT (Đề xuất)", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=C_ORANGE, zorder=4)


# =============================================================
# 3. PIPELINE 3: EVALUATION PIPELINE (BÊN PHẢI - THIẾT KẾ TINH GỌN CHUẨN MỰC)
# =============================================================
draw_box(73.5, 15.5, 24.5, 78.5, bg_color="#F8FAFC", border_color="#CBD5E1", radius=0.8, lw=1.2, zorder=1)

# Thanh Header Panel 3
draw_box(74.3, 88.5, 22.9, 4.2, bg_color=C_HEADER_BG, border_color="#CBD5E1", radius=0.4, zorder=2)
ax.text(85.75, 90.6, "Adaptive-VLM-Charts: evaluation pipeline", ha="center", va="center",
        fontsize=11, fontweight="bold", color=C_TEXT_MAIN, zorder=3)

# Mũi tên từ Panel 2 sang Panel 3:
# Đường Rationale R (từ CoT)
draw_arrow(68.0, 32.0, 76.5, 80.5, color="#64748B", lw=1.3)
# Đường Answer A (từ VLM)
draw_arrow(70.0, 23.0, 76.5, 71.0, color=C_ORANGE, lw=1.5)

# Khối Rationale R & Answer A
draw_box(76.5, 78.0, 18.5, 5.2, bg_color="white", border_color=C_BORDER, radius=0.4, zorder=2)
ax.text(85.75, 80.6, "Rationale R", ha="center", va="center", fontsize=9.2, fontweight="bold", color=C_TEXT_MAIN, zorder=3)

draw_arrow(85.75, 78.0, 85.75, 74.5)

draw_box(76.5, 68.0, 18.5, 6.5, bg_color="white", border_color=C_BORDER, radius=0.4, zorder=2)
ax.text(85.75, 71.25, "Answer A", ha="center", va="center", fontsize=9.8, fontweight="bold", color=C_TEXT_MAIN, zorder=3)

# Mũi tên xuống Chuẩn hóa
draw_arrow(85.75, 68.0, 85.75, 64.5)

# -------------------------------------------------------------
# KHỐI 1: CHUẨN HÓA (NORMALIZATION ENGINE)
# -------------------------------------------------------------
draw_box(75.5, 53.5, 20.5, 11.0, bg_color="white", border_color=C_BORDER, radius=0.5, zorder=2)
ax.text(85.75, 62.5, "Chuẩn hóa (Normalization)", ha="center", va="center", fontsize=9.2, fontweight="bold", color=C_TEXT_MAIN, zorder=3)

draw_box(76.8, 57.8, 17.9, 3.2, bg_color="#F1F5F9", border_color="#CBD5E1", radius=0.2, zorder=3)
ax.text(85.75, 59.4, "Word-to-Number   •   Bỏ LaTeX", ha="center", va="center", fontsize=7.6, color=C_TEXT_MAIN, zorder=4)

draw_box(76.8, 54.5, 17.9, 2.8, bg_color="#F1F5F9", border_color="#CBD5E1", radius=0.2, zorder=3)
ax.text(85.75, 55.9, "Medical Semantic Normalization", ha="center", va="center", fontsize=7.2, color=C_TEXT_MAIN, zorder=4)

# Mũi tên xuống Đánh giá CharXiv
draw_arrow(85.75, 53.5, 85.75, 50.0)

# -------------------------------------------------------------
# KHỐI 2: ĐÁNH GIÁ CHARXIV (KHỚP MẪU & SỐ HỌC)
# -------------------------------------------------------------
draw_box(75.5, 34.5, 20.5, 15.5, bg_color="white", border_color="#2563EB", lw=1.2, radius=0.5, zorder=2)
ax.text(85.75, 47.8, "Đánh giá CharXiv (Khớp mẫu & Số học)", ha="center", va="center",
        fontsize=8.2, fontweight="bold", color="#1D4ED8", zorder=3)
ax.plot([76.5, 95.0], [46.0, 46.0], color="#DBEAFE", lw=0.8, zorder=3)

ax.text(85.75, 44.0, "match_fixed (±5%)   |   Bootstrap 95% CI", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color="#1E40AF", zorder=3)

ax.text(85.75, 41.2, "Phân rã theo 4 nhóm câu hỏi:", ha="center", va="center",
        fontsize=7.6, fontweight="bold", color=C_TEXT_MAIN, zorder=3)
ax.text(77.5, 38.2, "• Number-in-Chart", fontsize=7.3, color=C_TEXT_MAIN, zorder=3)
ax.text(87.5, 38.2, "• Number-in-Gen", fontsize=7.3, color=C_TEXT_MAIN, zorder=3)
ax.text(77.5, 35.8, "• Text-in-Chart", fontsize=7.3, color=C_TEXT_MAIN, zorder=3)
ax.text(87.5, 35.8, "• Text-in-Gen", fontsize=7.3, color=C_TEXT_MAIN, zorder=3)

# Mũi tên xuống Đánh giá SLAKE
draw_arrow(85.75, 34.5, 85.75, 31.0)

# -------------------------------------------------------------
# KHỐI 3: ĐÁNH GIÁ SLAKE (Y KHOA & SINH NGÔN NGỮ NLG)
# -------------------------------------------------------------
draw_box(75.5, 17.5, 20.5, 13.5, bg_color="#F0FDF4", border_color=C_GREEN, lw=1.2, radius=0.5, zorder=2)
ax.text(85.75, 29.2, "Đánh giá SLAKE (Y khoa & Sinh NLG)", ha="center", va="center",
        fontsize=8.2, fontweight="bold", color="#166534", zorder=3)
ax.plot([76.5, 95.0], [27.6, 27.6], color="#BBF7D0", lw=0.8, zorder=3)

ax.text(85.75, 25.6, "• Closed: Strict Accuracy & Medical Acc", ha="center", va="center",
        fontsize=7.4, color=C_TEXT_MAIN, zorder=3)

# Ô làm nổi bật BLEU / ROUGE
draw_box(76.5, 20.5, 18.5, 3.4, bg_color="white", border_color="#86EFAC", radius=0.3, zorder=3)
ax.text(85.75, 22.2, "Open NLG: BLEU-1, 2  |  ROUGE-1, L", ha="center", va="center",
        fontsize=7.4, fontweight="bold", color="#15803D", zorder=4)

ax.text(85.75, 18.8, "• Open Strict EM & Medical Semantic Acc", ha="center", va="center",
        fontsize=7.1, color=C_TEXT_MAIN, zorder=3)


# =============================================================
# 4. HỘP VÍ DỤ MINH HỌA PHÍA DƯỚI (BOTTOM EXEMPLAR BOX)
# =============================================================
draw_box(2.0, 6.8, 96.0, 7.2, bg_color="#FEFCE8", border_color="#EAB308", radius=0.6, lw=1.2, zorder=2)
ax.text(50.0, 10.4, "Ví dụ Adaptive LoRA CoT: Step 1 (Target): cột xanh = nhãn \"Right\", cột vàng = nhãn \"Centre\". "
                       "Step 2 (Grounding): xanh = 32 + 38 + 30 = 100; vàng = 55 + 27 + 18 = 100. "
                       "Step 3 (Deduction): 100 − 100 = 0.  Answer: 0",
        ha="center", va="center", fontsize=8.8, color="#854D0E", fontweight="bold", zorder=3)


# =============================================================
# 5. CAPTION BÀI BÁO TẠP CHÍ Ở DƯỚI CÙNG
# =============================================================
ax.text(50.0, 3.0, "Hình 1. Tổng quan hệ thống Adaptive-VLM-Charts. Pipeline đầu vào tạo mẫu (I, Q); pipeline truy hồi & lập luận kết hợp Hybrid RAG "
                   "với ba chế độ suy luận của VLM (đường Oracle đi thẳng vào prompt); pipeline đánh giá chuẩn hóa đáp án, đo lường khớp mẫu và phân rã 4 nhóm câu hỏi "
                   "bằng match_fixed (CharXiv) song song với hệ chỉ số sinh ngôn ngữ BLEU / ROUGE (SLAKE Med-VQA).",
        ha="center", va="center", fontsize=8.2, color="#334155", style="italic", zorder=3)

# Lưu tệp hình ảnh
out_png = os.path.join(OUT_DIR, "system_architecture_redesign.png")
out_svg = os.path.join(OUT_DIR, "system_architecture_redesign.svg")

plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.savefig(out_svg, format='svg', bbox_inches='tight', pad_inches=0.08)
plt.close(fig)

print("ĐÃ CẬP NHẬT HOÀN TẤT THEO ĐÚNG THIẾT KẾ TINH GỌN CHUẨN XUẤT BẢN:")
print(f"1. PNG: {out_png}")
print(f"2. SVG: {out_svg}")
