# -*- coding: utf-8 -*-
"""plot_rag_results.py — Vẽ biểu đồ từ các file CSV kết quả. KHÔNG cần GPU,
chạy được trên máy cá nhân (VS Code) hoặc Colab runtime CPU.

CÀI ĐẶT (nếu chưa có trong .venv):
    pip install pandas numpy matplotlib

CHUẨN BỊ: tải 5 file này từ Drive (thư mục VQA_project) về cùng 1 thư mục
(hoặc sửa DATA_DIR bên dưới):
    rag_vqa_results.csv                               (RAG gốc)
    rag_vqa_results_vitl14_letterbox.csv              (+ crop/model)
    rag_vqa_results_vitl14_letterbox_caption.csv      (+ caption)
    retrieval_eval_results_v2.csv                     (Hit@k 4 cấu hình)
    caption_retrieval_hitk.csv                        (Hit@k caption)

CHẠY:  python plot_rag_results.py
Ảnh lưu vào thư mục figures_rag/ (PNG, 200 dpi).

CÁCH CHẤM: nếu có charxiv_match.py cùng thư mục thì script chấm lại cả 3 file
bằng match_fixed (cùng hàm với báo cáo), nên số khớp với compare_rag_versions.py.
Nếu không có, script dùng cột "correct" gốc (simple_match) và in cảnh báo.
Lưu ý: "Mô tả" trong các hình này gồm cả 4 câu mô tả (kể cả D4 "Not Applicable"),
còn bảng báo cáo tách D1-D3 và D4 riêng.
"""

import os
import sys
import io

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chấm lại bằng match_fixed (cùng hàm với báo cáo) nếu có charxiv_match.py cùng thư mục.
try:
    from charxiv_match import match_fixed
except ImportError:
    match_fixed = None
    print("CẢNH BÁO: không thấy charxiv_match.py -> dùng cột 'correct' có sẵn trong CSV "
          "(do simple_match chấm, có thể lệch với số trong báo cáo).")

DATA_DIR = "."
OUT_DIR = "figures_rag"
os.makedirs(OUT_DIR, exist_ok=True)

VQA_FILES = [
    ("RAG gốc\n(B/32 center-crop)", "rag_vqa_results.csv"),
    ("+ L/14 letterbox", "rag_vqa_results_vitl14_letterbox.csv"),
    ("+ L/14 letterbox\n+ caption", "rag_vqa_results_vitl14_letterbox_caption.csv"),
]
COLORS = ["#9AA5B1", "#4C8BF5", "#1E9E6A"]
REQUIRED_COLS = {"chart_index", "question_type", "slot", "retrieval_correct", "prediction", "gold_answer"}
D13 = ["descriptive_1", "descriptive_2", "descriptive_3"]   # câu mô tả trả lời được
D4 = "descriptive_4"                                       # câu "Not Applicable"

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def to_bool(series):
    return series.map(lambda x: str(x).strip().lower() == "true")


def load_vqa(path):
    df = pd.read_csv(os.path.join(DATA_DIR, path), keep_default_na=False)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path} thiếu cột: {missing}")
    if {"chart_index", "slot"}.issubset(df.columns):
        n_dup = int(df.duplicated(subset=["chart_index", "slot"]).sum())
        if n_dup:
            df = df.drop_duplicates(subset=["chart_index", "slot"], keep="last")
    if match_fixed is not None:
        df["correct"] = [match_fixed(p, g) for p, g in zip(df.prediction, df.gold_answer)]
    elif "correct" in df.columns:
        df["correct"] = to_bool(df["correct"])
    else:
        raise ValueError(f"{path}: không có charxiv_match.py và cũng không có cột 'correct'")
    df["retrieval_correct"] = to_bool(df["retrieval_correct"])
    return df


def boot_ci(df, n=2000, seed=42):
    """Bootstrap theo biểu đồ (resample cả biểu đồ, giống cách làm ở báo cáo)."""
    if len(df) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    g = df.groupby("chart_index")["correct"].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), size=(n, len(g)))
    accs = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return lo * 100, hi * 100


def acc_with_ci(df):
    if len(df) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = boot_ci(df)
    return df["correct"].mean() * 100, lo, hi


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("Đã lưu:", path)


# ---------------------------------------------------------------------------
# Hình 1: độ chính xác 3 phiên bản RAG (tổng / mô tả / suy luận) + KTC 95%
# ---------------------------------------------------------------------------
def fig_overall(dfs):
    groups = [("Tổng thể", lambda d: d),
              ("Mô tả D1–D3", lambda d: d[d.slot.isin(D13)]),
              ("Suy luận", lambda d: d[d.slot == "reasoning"]),
              ("Không thể trả lời (D4)*", lambda d: d[d.slot == D4])]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    width = 0.26
    x = np.arange(len(groups))
    for k, ((label, _), df) in enumerate(zip(VQA_FILES, dfs)):
        vals, errs = [], []
        for _, pick in groups:
            m, lo, hi = acc_with_ci(pick(df))
            vals.append(m)
            errs.append([m - lo, hi - m])
        errs = np.array(errs).T
        pos = x + (k - 1) * width
        bars = ax.bar(pos, vals, width, yerr=errs, capsize=3, color=COLORS[k],
                      label=label.replace("\n", " "))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Độ chính xác (%)")
    ax.set_xlabel("* D4: ở 2 bản không dùng caption, mỗi mẫu câu hỏi luôn truy hồi về cùng 1 ảnh, nên điểm D4 phụ thuộc "
                  "ảnh 'mặc định' đó,\nkhông phản ánh chất lượng truy hồi hay khả năng nhận ra câu không trả lời được.",
                  fontsize=8, color="#6B7280", loc="left")
    ax.set_title("Độ chính xác RAG-VQA qua từng cải tiến retrieval\n(thanh sai số: KTC 95% bootstrap theo biểu đồ)")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(30, ax.get_ylim()[1]))
    savefig(fig, "hinh_A_tong_the_3_phien_ban.png")


# ---------------------------------------------------------------------------
# Hình 2: Hit@1/5/10 của 4 cấu hình CLIP (mô tả vs suy luận) + caption
# ---------------------------------------------------------------------------
def fig_hitk():
    path = os.path.join(DATA_DIR, "retrieval_eval_results_v2.csv")
    if not os.path.exists(path):
        print("Bỏ qua Hình B: không thấy retrieval_eval_results_v2.csv")
        return
    ev = pd.read_csv(path)
    names = {"vitb32_centercrop": "B/32\ncenter-crop", "vitb32_letterbox": "B/32\nletterbox",
             "vitb32_tiling": "B/32\ntiling", "vitl14_letterbox": "L/14\nletterbox"}
    order = [c for c in names if c in set(ev.config_name)]

    cap_path = os.path.join(DATA_DIR, "caption_retrieval_hitk.csv")
    cap = pd.read_csv(cap_path).iloc[0] if os.path.exists(cap_path) else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    ks = [("hit_at_1", "Hit@1", "#1E9E6A"), ("hit_at_5", "Hit@5", "#4C8BF5"), ("hit_at_10", "Hit@10", "#9AA5B1")]
    for ax, qt, title in zip(axes, ["reasoning", "descriptive"],
                              ["Suy luận (truy vấn = câu hỏi riêng của từng biểu đồ)",
                               "Mô tả (truy vấn = 19 câu mẫu lặp lại)"]):
        labels = [names[c] for c in order]
        data = {k: [ev[(ev.config_name == c) & (ev.query_type == qt)][k].values[0] * 100 for c in order]
                for k, _, _ in ks}
        if qt == "descriptive" and cap is not None:
            labels.append("L/14 letterbox\n+ CAPTION")
            for k, _, _ in ks:
                data[k].append(float(cap[k]) * 100)
        x = np.arange(len(labels))
        w = 0.26
        for i, (k, lab, col) in enumerate(ks):
            bars = ax.bar(x + (i - 1) * w, data[k], w, color=col, label=lab)
            for b, v in zip(bars, data[k]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Tỷ lệ truy hồi đúng ảnh (%)")
    axes[0].legend(frameon=False)
    fig.suptitle("Chất lượng truy hồi ảnh (CLIP) theo cấu hình — pool 500 biểu đồ", fontsize=12)
    savefig(fig, "hinh_B_hit_at_k.png")


# ---------------------------------------------------------------------------
# Hình 3: Accuracy VQA khi retrieval đúng vs sai (bản cuối)
# ---------------------------------------------------------------------------
def fig_by_retrieval(df_final):
    rows = []
    for pick, qt_label in [(df_final.slot.isin(D13), "Mô tả"), (df_final.slot == "reasoning", "Suy luận")]:
        for rc, rc_label in [(True, "Truy hồi ĐÚNG"), (False, "Truy hồi SAI")]:
            sub = df_final[pick & (df_final.retrieval_correct == rc)]
            rows.append((qt_label, rc_label, sub["correct"].mean() * 100 if len(sub) else np.nan, len(sub)))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(2)
    w = 0.35
    for i, (rc_label, col) in enumerate([("Truy hồi ĐÚNG", "#1E9E6A"), ("Truy hồi SAI", "#D95C5C")]):
        vals = [r[2] for r in rows if r[1] == rc_label]
        ns = [r[3] for r in rows if r[1] == rc_label]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=rc_label)
        for b, v, n in zip(bars, vals, ns):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%\n(n={n})", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Mô tả D1–D3", "Suy luận"])
    ax.set_ylabel("Độ chính xác VQA (%)")
    ax.set_title("Accuracy phụ thuộc vào việc truy hồi đúng ảnh\n(bản cuối: L/14 letterbox + caption; không tính D4)")
    ax.legend(frameon=False)
    ax.set_ylim(0, 100)
    savefig(fig, "hinh_C_dung_sai_truy_hoi.png")


# ---------------------------------------------------------------------------
# Hình 4: Suy luận theo kiểu đáp án, 3 phiên bản
# ---------------------------------------------------------------------------
def fig_subtype(dfs):
    if "sub_type" not in dfs[0].columns:
        print("Bỏ qua Hình D: không có cột sub_type")
        return
    types = ["Number-in-Chart", "Number-in-General", "Text-in-Chart", "Text-in-General"]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    w = 0.26
    x = np.arange(len(types))
    for k, ((label, _), df) in enumerate(zip(VQA_FILES, dfs)):
        r = df[df.question_type == "reasoning"]
        vals = [r[r.sub_type == t]["correct"].mean() * 100 if (r.sub_type == t).any() else np.nan for t in types]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=COLORS[k], label=label.replace("\n", " "))
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
    n_each = dfs[-1][dfs[-1].question_type == "reasoning"].groupby("sub_type").size()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={n_each.get(t, 0)})" for t in types], fontsize=9)
    ax.set_ylabel("Độ chính xác suy luận (%)")
    ax.set_title("Suy luận theo kiểu đáp án — mẫu mỗi nhóm nhỏ, chỉ mang tính tham khảo")
    ax.legend(frameon=False)
    savefig(fig, "hinh_D_kieu_dap_an_suy_luan.png")


def main():
    dfs = [load_vqa(p) for _, p in VQA_FILES]
    print("Số dòng mỗi file:", [len(d) for d in dfs])
    print("\n=== Tóm tắt (đã chấm lại bằng match_fixed nếu có charxiv_match.py) ===")
    for (label, _), df in zip(VQA_FILES, dfs):
        d = df[df.slot.isin(D13)]["correct"].mean() * 100
        d4 = df[df.slot == D4]["correct"].mean() * 100
        r = df[df.slot == "reasoning"]["correct"].mean() * 100
        print(f"{label.replace(chr(10), ' '):38s} tổng {df['correct'].mean()*100:5.1f}%  "
              f"D1-D3 {d:5.1f}%  suy luận {r:5.1f}%  D4 {d4:5.1f}%  "
              f"retrieval đúng {df['retrieval_correct'].mean()*100:5.1f}%")
    fig_overall(dfs)
    fig_hitk()
    fig_by_retrieval(dfs[-1])
    fig_subtype(dfs)
    print(f"\nXong. Mở thư mục {OUT_DIR}/ để xem ảnh.")


if __name__ == "__main__":
    main()