# -*- coding: utf-8 -*-
"""compare_rag_versions.py — chạy trên máy (CPU, không cần GPU).

Đặt file này CÙNG THƯ MỤC với charxiv_match.py và 5 file CSV:
    baseline_results.csv                              (Gemini)
    baseline_qwen_results.csv                         (Qwen oracle)
    rag_vqa_results.csv                               (RAG gốc)
    rag_vqa_results_vitl14_letterbox.csv              (+ L/14 letterbox)
    rag_vqa_results_vitl14_letterbox_caption.csv      (+ caption)
rồi chạy:  python compare_rag_versions.py

Script làm 5 việc (tất cả chấm bằng match_fixed, cùng quy tắc "tập chung" như
rescore_and_plot.py: chỉ giữ câu Gemini có phản hồi):
  1. Kiểm tra RAG và baseline có chạy trên CÙNG 500 biểu đồ không.
  2. Bảng 5 hệ thống (tổng + KTC 95%, D1-D3, suy luận, không thể trả lời).
  3. Bootstrap GHÉP CẶP (resample biểu đồ) cho hiệu giữa các phiên bản RAG.
  4. Accuracy theo truy hồi đúng/sai (bản cuối), chấm lại bằng match_fixed.
  5. Chẩn đoán truy hồi câu mô tả: mỗi mẫu câu hỏi truy hồi về bao nhiêu ảnh khác nhau.
Kết quả in ra màn hình và lưu vào thư mục figures/.
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
DATA_DIR = HERE
OUT_DIR = os.path.join(HERE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "Gemini": "baseline_results.csv",
    "Qwen": "baseline_qwen_results.csv",
    "RAG gốc": "rag_vqa_results.csv",
    "RAG + L/14 letterbox": "rag_vqa_results_vitl14_letterbox.csv",
    "RAG + L/14 letterbox + caption": "rag_vqa_results_vitl14_letterbox_caption.csv",
}
RAG_NAMES = ["RAG gốc", "RAG + L/14 letterbox", "RAG + L/14 letterbox + caption"]
N_BOOT = 5000
rng = np.random.default_rng(42)


def load(name):
    path = os.path.join(DATA_DIR, FILES[name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không thấy '{path}'.")
    d = pd.read_csv(path, keep_default_na=False)
    d["k"] = d.chart_index.astype(str) + "|" + d.slot.astype(str)
    n_dup = int(d.duplicated("k").sum())
    if n_dup:
        print(f"[{name}] CẢNH BÁO: {n_dup} dòng trùng (chart_index, slot), giữ dòng cuối.")
        d = d.drop_duplicates("k", keep="last")
    d["correct"] = [match_fixed(p, g) for p, g in zip(d.prediction, d.gold_answer)]
    if "retrieval_correct" in d:
        d["retrieval_correct"] = d.retrieval_correct.astype(str) == "True"
    if "retrieved_chart_index" in d:
        d["retrieved_chart_index"] = pd.to_numeric(d.retrieved_chart_index, errors="coerce")
    return d


raw = {n: load(n) for n in FILES}

# ---------- 1. Kiểm tra cùng tập biểu đồ ----------
print("=" * 70)
print("1. KIỂM TRA RAG VÀ BASELINE CÓ CÙNG 500 BIỂU ĐỒ KHÔNG")
print("=" * 70)


def same_chart_rate(a, b):
    ra = a[a.slot == "reasoning"].set_index("chart_index")
    rb = b[b.slot == "reasoning"].set_index("chart_index")
    col = "question" if ("question" in ra and "question" in rb) else "gold_answer"
    common = ra.index.intersection(rb.index)
    if len(common) == 0:
        return col, 0, np.nan
    same = (ra.loc[common, col].astype(str).str.strip() == rb.loc[common, col].astype(str).str.strip()).mean()
    return col, len(common), same


for n in ["Qwen"] + RAG_NAMES:
    col, m, rate = same_chart_rate(raw["Gemini"], raw[n])
    verdict = "CÙNG biểu đồ" if rate > 0.95 else "KHÁC biểu đồ (cần xử lý!)"
    print(f"Gemini vs {n:34s}: so cột '{col}' trên {m} biểu đồ chung -> trùng {rate:.1%}  => {verdict}")
print("Ghi chú: nếu 'KHÁC biểu đồ', chart_index của các file không cùng nghĩa; "
      "không so sánh trực tiếp các hệ thống đó với nhau.")

# ---------- Tập chung ----------
valid = set(raw["Gemini"][raw["Gemini"].prediction.astype(str).str.strip() != ""].k)
common_keys = valid.copy()
for n in FILES:
    common_keys &= set(raw[n].k)
D = {n: d[d.k.isin(common_keys)].sort_values("k").reset_index(drop=True) for n, d in raw.items()}
print(f"\nTập chung: {len(common_keys)} câu hợp lệ (giao của 5 file và có phản hồi Gemini).")

M = pd.DataFrame({"k": D["Gemini"].k, "chart": D["Gemini"].chart_index.values, "slot": D["Gemini"].slot.values})
for n in FILES:
    assert (D[n].k.values == M.k.values).all(), f"Khóa của {n} không khớp thứ tự"
    M[n] = D[n].correct.astype(int).values

MASKS = {
    "Tất cả câu": np.ones(len(M), bool),
    "Mô tả D1-D3": M.slot.isin(["descriptive_1", "descriptive_2", "descriptive_3"]).values,
    "Suy luận": (M.slot == "reasoning").values,
    "Không thể trả lời (D4)": (M.slot == "descriptive_4").values,
}


def boot_ci(col, mask, n=N_BOOT):
    sub = M[mask]
    g = sub.groupby("chart")[col].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), (n, len(g)))
    return np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5]) * 100


# ---------- 2. Bảng 5 hệ thống ----------
print("\n" + "=" * 70)
print("2. BẢNG 5 HỆ THỐNG (chấm bằng match_fixed, tập chung)")
print("=" * 70)
rows = []
for n in FILES:
    lo, hi = boot_ci(n, MASKS["Tất cả câu"])
    rows.append({
        "Hệ thống": n, "n câu": len(M),
        "Tổng (%)": round(M[n].mean() * 100, 1), "KTC95 thấp": round(lo, 1), "KTC95 cao": round(hi, 1),
        "Mô tả D1-D3 (%)": round(M[n][MASKS["Mô tả D1-D3"]].mean() * 100, 1),
        "Suy luận (%)": round(M[n][MASKS["Suy luận"]].mean() * 100, 1),
        "Không thể trả lời (%)": round(M[n][MASKS["Không thể trả lời (D4)"]].mean() * 100, 1),
    })
table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(os.path.join(OUT_DIR, "ket_qua_5_he_thong.csv"), index=False, encoding="utf-8-sig")

# ---------- 3. Bootstrap ghép cặp ----------
print("\n" + "=" * 70)
print(f"3. BOOTSTRAP GHÉP CẶP ({N_BOOT} lần, resample biểu đồ) — hiệu = B - A, đơn vị điểm %")
print("=" * 70)
pairs = [("RAG gốc", "RAG + L/14 letterbox"),
         ("RAG gốc", "RAG + L/14 letterbox + caption"),
         ("RAG + L/14 letterbox", "RAG + L/14 letterbox + caption")]
prow = []
for a, b in pairs:
    for mname in ["Tất cả câu", "Mô tả D1-D3", "Suy luận"]:
        mask = MASKS[mname]
        sub = M[mask]
        g = sub.groupby("chart")
        sa, sb, c = g[a].sum().values, g[b].sum().values, g[a].count().values
        obs = (sb.sum() - sa.sum()) / c.sum() * 100
        idx = rng.integers(0, len(c), (N_BOOT, len(c)))
        diff = (sb[idx].sum(1) - sa[idx].sum(1)) / c[idx].sum(1) * 100
        lo, hi = np.percentile(diff, [2.5, 97.5])
        only_b = int(((sub[b] == 1) & (sub[a] == 0)).sum())
        only_a = int(((sub[a] == 1) & (sub[b] == 0)).sum())
        prow.append({"A": a, "B": b, "Nhóm câu": mname, "Hiệu B-A (điểm %)": round(obs, 2),
                     "KTC95 thấp": round(lo, 2), "KTC95 cao": round(hi, 2),
                     "KTC chứa 0?": "có" if lo <= 0 <= hi else "không",
                     "Chỉ B đúng": only_b, "Chỉ A đúng": only_a})
paired = pd.DataFrame(prow)
print(paired.to_string(index=False))
paired.to_csv(os.path.join(OUT_DIR, "paired_bootstrap.csv"), index=False, encoding="utf-8-sig")
print("Cách đọc: 'KTC chứa 0? = không' nghĩa là chênh lệch khó là do dao động chọn biểu đồ. "
      "Chưa loại được dao động giữa các lần chạy (mỗi bản chỉ chạy 1 lần).")

# ---------- 4. Accuracy theo truy hồi đúng/sai (bản cuối) ----------
print("\n" + "=" * 70)
print("4. ACCURACY THEO TRUY HỒI ĐÚNG/SAI — bản cuối (chấm bằng match_fixed)")
print("=" * 70)
fin = D["RAG + L/14 letterbox + caption"].copy()
fin["nhom"] = np.where(fin.slot == "reasoning", "Suy luận", "Mô tả (4 câu)")
t4 = (fin.groupby(["nhom", "retrieval_correct"]).correct.agg(["mean", "count"])
      .rename(columns={"mean": "accuracy", "count": "n"}))
t4["accuracy"] = (t4["accuracy"] * 100).round(1)
print(t4.to_string())
t4.to_csv(os.path.join(OUT_DIR, "acc_theo_truy_hoi.csv"), encoding="utf-8-sig")

# ---------- 5. Chẩn đoán truy hồi câu mô tả ----------
print("\n" + "=" * 70)
print("5. CHẨN ĐOÁN TRUY HỒI CÂU MÔ TẢ — mỗi mẫu câu hỏi truy hồi về bao nhiêu ảnh khác nhau?")
print("=" * 70)
drows = []
for n in RAG_NAMES:
    d = D[n]
    d = d[d.slot.str.startswith("descriptive")]
    per_t = d.groupby("question").retrieved_chart_index.nunique()
    wrong = d[~d.retrieval_correct]
    drows.append({
        "Phiên bản": n, "Số mẫu câu hỏi": len(per_t),
        "Ảnh khác nhau / mẫu (min-trung vị-max)": f"{int(per_t.min())}-{int(per_t.median())}-{int(per_t.max())}",
        "Tổng ảnh khác nhau (toàn bộ câu mô tả)": int(d.retrieved_chart_index.nunique()),
        "Retrieval đúng (%)": round(d.retrieval_correct.mean() * 100, 1),
        "Accuracy khi truy hồi SAI (%)": round(wrong.correct.mean() * 100, 1),
        "n sai": len(wrong),
    })
diag = pd.DataFrame(drows)
print(diag.to_string(index=False))
diag.to_csv(os.path.join(OUT_DIR, "chan_doan_truy_hoi_mo_ta.csv"), index=False, encoding="utf-8-sig")
print("Cách đọc: ở 2 bản dùng câu hỏi làm truy vấn, nếu 'ảnh khác nhau / mẫu' = 1 thì mỗi mẫu câu hỏi "
      "luôn lấy về đúng 1 ảnh cho mọi biểu đồ (do cách xây truy vấn). Khi đó accuracy mô tả của hai bản "
      "đó phản ánh ảnh 'mặc định' ấy tình cờ khớp đáp án đến đâu, không phải chất lượng truy hồi.")

# ---------- 6. Đối chiếu Hình B (Hit@1) với file VQA, và RAG khi truy hồi đúng vs Qwen oracle ----------
print("\n" + "=" * 70)
print("6. ĐỐI CHIẾU HÌNH B (Hit@1 đo riêng) VỚI TỈ LỆ TRUY HỒI ĐÚNG TRONG FILE VQA")
print("=" * 70)


def read_opt(name):
    p = os.path.join(DATA_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


ev = read_opt("retrieval_eval_results_v2.csv")
cap = read_opt("caption_retrieval_hitk.csv")
if ev is None:
    print("Không thấy retrieval_eval_results_v2.csv -> bỏ qua mục 6 (Hình B).")
else:
    cfg_of = {"RAG gốc": "vitb32_centercrop",
              "RAG + L/14 letterbox": "vitl14_letterbox",
              "RAG + L/14 letterbox + caption": "vitl14_letterbox"}
    rows6 = []
    for n in RAG_NAMES:
        for grp, qt in [("Suy luận", "reasoning"), ("Mô tả", "descriptive")]:
            e = ev[(ev.config_name == cfg_of[n]) & (ev.query_type == qt)]
            hit = float(e.hit_at_1.iloc[0]) * 100 if len(e) else np.nan
            if n.endswith("caption") and qt == "descriptive" and cap is not None:
                hit = float(cap.hit_at_1.iloc[0]) * 100
            d = raw[n]
            d = d[d.slot == "reasoning"] if qt == "reasoning" else d[d.slot.str.startswith("descriptive")]
            obs = d.retrieval_correct.mean() * 100
            rows6.append({"Phiên bản": n, "Nhóm": grp, "Hit@1 Hình B (%)": round(hit, 1),
                          "Truy hồi đúng trong file VQA (%)": round(obs, 1),
                          "Lệch (điểm %)": round(obs - hit, 1)})
    t6 = pd.DataFrame(rows6)
    print(t6.to_string(index=False))
    t6.to_csv(os.path.join(OUT_DIR, "doi_chieu_hinh_B.csv"), index=False, encoding="utf-8-sig")
    print("Cách đọc: hai cột nên gần nhau (lệch cỡ 1-2 biểu đồ, tức ~0.2-0.4 điểm %, do tìm kiếm gần đúng của "
          "ChromaDB). Lệch lớn hơn nghĩa là pipeline VQA và phép đo Hình B không dùng cùng truy vấn/collection.")
    print("Lưu ý: ở 2 bản không dùng caption, Hit@1 Hình B của nhóm mô tả luôn bằng đúng 1/500 = 0.2% "
          "(Hit@5 = 1.0%, Hit@10 = 2.0%) vì phép đo dùng MỘT câu hỏi giống nhau cho mọi biểu đồ nên danh sách "
          "top-k giống hệt nhau. Đây là hệ quả của cách đo, khớp với giải thích 'ảnh mặc định' nhưng không phải "
          "bằng chứng độc lập; mục 5 mới là kiểm tra trực tiếp trên dữ liệu pipeline.")

print("\n" + "=" * 70)
print("7. RAG KHI TRUY HỒI ĐÚNG so với QWEN ORACLE trên CÙNG câu hỏi (tập chung, match_fixed)")
print("=" * 70)
ALLDESC = M.slot.str.startswith("descriptive").values
rows7 = []
for n in RAG_NAMES:
    ok_mask = D[n].retrieval_correct.values
    for grp, gmask in [("Suy luận", MASKS["Suy luận"]), ("Mô tả (4 câu)", ALLDESC)]:
        m = ok_mask & gmask
        k = int(m.sum())
        rows7.append({"Phiên bản": n, "Nhóm": grp, "n câu truy hồi đúng": k,
                      "RAG (%)": round(M[n][m].mean() * 100, 1) if k else np.nan,
                      "Qwen oracle, cùng câu (%)": round(M["Qwen"][m].mean() * 100, 1) if k else np.nan,
                      "Ghi chú": "n quá nhỏ, bỏ qua" if k < 30 else ""})
t7 = pd.DataFrame(rows7)
print(t7.to_string(index=False))
t7.to_csv(os.path.join(OUT_DIR, "rag_dung_vs_qwen_oracle.csv"), index=False, encoding="utf-8-sig")
print("Cách đọc: nếu RAG gần Qwen oracle trên các câu truy hồi đúng thì khâu suy luận của VLM không phải nút thắt. "
      "Các biểu đồ truy hồi đúng có thể vốn dễ hơn (chọn lọc), nên chỉ nói đây là tương quan.")

print(f"\nXong. Các bảng đã lưu trong: {OUT_DIR}")
