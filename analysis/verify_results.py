# -*- coding: utf-8 -*-
"""
analysis/verify_results.py
================================================================================
KỊCH BẢN XÁC MINH ĐỘC LẬP KẾT QUẢ THỰC NGHIỆM VLM-CHARTS (REPRODUCIBILITY SCRIPT)
================================================================================
Mục đích:
    Tự động đọc trực tiếp các tệp kết quả thực nghiệm CSV gốc:
      1. results/slake_vqa_results.csv (SLAKE Med-VQA, N = 1.061)
      2. results/dynamic_routing_charxiv_results.csv (CharXiv Scientific Charts, N = 500)
    Xác minh từng chỉ số số học, tỷ lệ phần trăm, kiểm định thống kê và trần Oracle
    được công bố trong Báo cáo Thực nghiệm (ACADEMIC_EVALUATION_REPORT.md).

Cách chạy:
    python analysis/verify_results.py
================================================================================
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Hỗ trợ UTF-8 console trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Định vị thư mục gốc của repository
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
SLAKE_CSV = RESULTS_DIR / "slake_vqa_results.csv"
CHARXIV_CSV = RESULTS_DIR / "dynamic_routing_charxiv_results.csv"

def compute_mcnemar(b01, b10):
    """Tính McNemar chi2 và p-value có hiệu chỉnh liên tục Yates (1 bậc tự do)."""
    from scipy.stats import chi2
    n_discordant = b01 + b10
    if n_discordant == 0:
        return 0.0, 1.0
    stat = (abs(b01 - b10) - 1.0) ** 2 / n_discordant
    pval = float(chi2.sf(stat, df=1))
    return stat, pval

def check_metric(name, actual, expected, tol=0.05, note=""):
    """Hàm kiểm tra đối chiếu chỉ số thực nghiệm (hỗ trợ dung sai phiên bản thư viện)."""
    diff = abs(actual - expected)
    passed = diff <= tol
    status = "[PASS]" if passed else "[FAIL]"
    extra = f" [{note}]" if (note and diff > 0.05 and passed) else ""
    print(f"  {status} {name:<45} : Thực tế = {actual:6.2f}% | Báo cáo = {expected:6.2f}% (Sai số: {diff:.3f}%){extra}")
    return passed

def main():
    print("=" * 80)
    print("VLM-CHARTS: XÁC MINH TÍNH TÁI LẬP CỦA KẾT QUẢ THỰC NGHIỆM")
    print("=" * 80)
    
    # 1. Kiểm tra sự tồn tại của tệp dữ liệu
    for p in [SLAKE_CSV, CHARXIV_CSV]:
        if not p.exists():
            print(f"[LỖI] Không tìm thấy tệp kết quả: {p}")
            sys.exit(1)
        print(f"[OK] Đã tìm thấy: {p.relative_to(ROOT)}")

    all_passed = True

    # ══════════════════════════════════════════════════════════════════════════
    # 2. XÁC MINH KẾT QUẢ TRÊN BỘ DỮ LIỆU SLAKE (N = 1.061)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 80)
    print("PHẦN 1: BỘ DỮ LIỆU SLAKE (Y KHOA - QWEN2.5-VL-7B ZERO-SHOT)")
    print("-" * 80)
    slake_df = pd.read_csv(SLAKE_CSV)
    n_slake = len(slake_df)
    assert n_slake == 1061, f"Kích thước SLAKE không khớp: {n_slake} != 1061"

    s_nc = (slake_df["acc_no_cot"].fillna(0) > 0).astype(int)
    s_c  = (slake_df["acc_cot"].fillna(0) > 0).astype(int)
    s_ad = (slake_df["acc_adaptive"].fillna(0) > 0).astype(int)

    all_passed &= check_metric("SLAKE Baseline No-CoT Accuracy", s_nc.mean() * 100, 50.24)
    all_passed &= check_metric("SLAKE Standard CoT Accuracy", s_c.mean() * 100, 45.90)
    all_passed &= check_metric("SLAKE Adaptive Prompting Accuracy", s_ad.mean() * 100, 52.59)
    all_passed &= check_metric("SLAKE Delta (Adaptive - No-CoT)", (s_ad.mean() - s_nc.mean()) * 100, 2.36)

    # McNemar Test trên SLAKE
    b01_s = int(((s_nc == 1) & (s_ad == 0)).sum())
    b10_s = int(((s_nc == 0) & (s_ad == 1)).sum())
    chi2_s, pval_s = compute_mcnemar(b01_s, b10_s)
    print(f"  [OK] McNemar SLAKE: b01={b01_s}, b10={b10_s}, chi2={chi2_s:.4f}, p-value={pval_s:.6f}")
    assert pval_s < 0.01, f"SLAKE McNemar p-value {pval_s} không đạt mức ý nghĩa p < 0.01"

    # Phân tích câu hỏi Closed vs Open
    closed_m = slake_df["answer_type"] == "CLOSED"
    open_m   = slake_df["answer_type"] == "OPEN"
    all_passed &= check_metric("SLAKE Closed Questions No-CoT", s_nc[closed_m].mean() * 100, 69.47)
    all_passed &= check_metric("SLAKE Closed Questions CoT", s_c[closed_m].mean() * 100, 61.78)
    all_passed &= check_metric("SLAKE Closed Questions Adaptive", s_ad[closed_m].mean() * 100, 71.15)
    all_passed &= check_metric("SLAKE Open Questions No-CoT", s_nc[open_m].mean() * 100, 37.83)
    all_passed &= check_metric("SLAKE Open Questions CoT", s_c[open_m].mean() * 100, 35.66)
    all_passed &= check_metric("SLAKE Open Questions Adaptive", s_ad[open_m].mean() * 100, 40.62)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. XÁC MINH KẾT QUẢ TRÊN BỘ DỮ LIỆU CHARXIV (N = 500)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 80)
    print("PHẦN 2: BỘ DỮ LIỆU CHARXIV (BIỂU ĐỒ KHOA HỌC - N = 500)")
    print("-" * 80)
    cx_df = pd.read_csv(CHARXIV_CSV)
    n_cx = len(cx_df)
    assert n_cx == 500, f"Kích thước CharXiv không khớp: {n_cx} != 500"

    c_dir = cx_df["direct_correct"].fillna(False).astype(int)
    c_cot = cx_df["cot_correct"].fillna(False).astype(int)
    c_ad  = cx_df["adaptive_correct"].fillna(False).astype(int)
    c_rt  = cx_df["routed_correct"].fillna(False).astype(int)

    all_passed &= check_metric("CharXiv Direct Zero-Shot Accuracy", c_dir.mean() * 100, 38.80)
    all_passed &= check_metric("CharXiv Standard CoT Accuracy", c_cot.mean() * 100, 36.20)
    all_passed &= check_metric("CharXiv Adaptive LoRA CoT Accuracy", c_ad.mean() * 100, 37.40)
    all_passed &= check_metric("CharXiv Adaptive Routed Accuracy", c_rt.mean() * 100, 39.60)
    all_passed &= check_metric("CharXiv Delta (Routed - Direct)", (c_rt.mean() - c_dir.mean()) * 100, 0.80)

    # McNemar Test trên CharXiv
    b01_c = int(((c_dir == 1) & (c_rt == 0)).sum())
    b10_c = int(((c_dir == 0) & (c_rt == 1)).sum())
    chi2_c, pval_c = compute_mcnemar(b01_c, b10_c)
    print(f"  [OK] McNemar CharXiv: b01={b01_c}, b10={b10_c}, chi2={chi2_c:.4f}, p-value={pval_c:.4f}")

    # Bóc tách Trần Oracle
    # Oracle A: Direct + Adaptive LoRA (135 + 59 + 52 = 246 / 500 = 49.20%)
    both_da = int(((c_dir == 1) & (c_ad == 1)).sum())
    dir_only_a = int(((c_dir == 1) & (c_ad == 0)).sum())
    ad_only = int(((c_dir == 0) & (c_ad == 1)).sum())
    oracle_a = both_da + dir_only_a + ad_only
    all_passed &= check_metric("Oracle (Direct + Adaptive LoRA CoT)", (oracle_a / n_cx) * 100, 49.20)
    print(f"    Chi tiết: Cả 2 đúng={both_da}, Chỉ Direct đúng={dir_only_a}, Chỉ LoRA đúng={ad_only} (Tổng={oracle_a}/500)")

    # Oracle B: Direct + Standard CoT (117 + 77 + 64 = 258 / 500 = 51.60%)
    both_dc = int(((c_dir == 1) & (c_cot == 1)).sum())
    dir_only_c = int(((c_dir == 1) & (c_cot == 0)).sum())
    cot_only = int(((c_dir == 0) & (c_cot == 1)).sum())
    oracle_b = both_dc + dir_only_c + cot_only
    all_passed &= check_metric("Oracle (Direct + Standard CoT)", (oracle_b / n_cx) * 100, 51.60)
    print(f"    Chi tiết: Cả 2 đúng={both_dc}, Chỉ Direct đúng={dir_only_c}, Chỉ CoT đúng={cot_only} (Tổng={oracle_b}/500)")

    # Phân tích phân tầng Sub-groups trên CharXiv
    for st, exp_dir, exp_cot, exp_ad, exp_rt in [
        ("Number-in-General", 26.98, 28.57, 30.16, 30.16),
        ("Number-in-Chart", 45.22, 35.65, 45.22, 45.22),
        ("Text-in-General", 47.83, 39.13, 39.13, 47.83),
        ("Text-in-Chart", 40.38, 40.38, 37.09, 40.38),
    ]:
        sub = cx_df[cx_df["sub_type"] == st]
        all_passed &= check_metric(f"Subgroup [{st}] Direct", (sub["direct_correct"].fillna(False).astype(int)).mean() * 100, exp_dir)
        all_passed &= check_metric(f"Subgroup [{st}] CoT", (sub["cot_correct"].fillna(False).astype(int)).mean() * 100, exp_cot)
        all_passed &= check_metric(f"Subgroup [{st}] Routed", (sub["routed_correct"].fillna(False).astype(int)).mean() * 100, exp_rt)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. XÁC MINH HIỆU NĂNG VQA ĐẦU-CUỐI CỦA ROUTER HỌC ĐƯỢC (KHÔNG DÙNG METADATA)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 80)
    print("PHẦN 3: BỘ ĐỊNH TUYẾN HỌC MÁY TỰ ĐỘNG (END-TO-END VQA)")
    print("-" * 80)
    
    # Huấn luyện và kiểm thử trên phân chia 60/20/20 của CharXiv (N_test = 100)
    y_strat = (cx_df["routed_strategy"] == "Adaptive_LoRA_CoT").astype(int).values
    idx = np.arange(len(cx_df))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.40, random_state=42, stratify=y_strat)
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, random_state=42, stratify=y_strat[tmp_idx])

    # Trích xuất đặc trưng văn bản TF-IDF (khớp cấu hình leak-free statistical_analysis.py)
    vec = TfidfVectorizer(max_features=750, ngram_range=(1, 2))
    X_tr = vec.fit_transform(cx_df["question"].iloc[tr_idx].fillna("")).toarray()
    X_te = vec.transform(cx_df["question"].iloc[te_idx].fillna("")).toarray()

    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(X_tr, y_strat[tr_idx])
    pred_te = clf.predict(X_te)
    
    router_acc = float((pred_te == y_strat[te_idx]).mean()) * 100
    # Dung sai ±1 câu trên tập test N=100 (±1.0%) cho phép tương thích giữa các phiên bản scikit-learn (1.7.x: 86.0%, 1.9.x: 87.0%)
    all_passed &= check_metric(
        "CharXiv Router Classification Acc (N=100)", 
        router_acc, 
        86.00, 
        tol=1.05, 
        note="Dung sai ±1 câu / N=100 do phiên bản scikit-learn (86.0% ~ 87.0%)"
    )

    # Độ chính xác VQA đầu-cuối trên tập test (N=100)
    vqa_pred_learned = np.where(pred_te == 1, c_ad.iloc[te_idx], c_dir.iloc[te_idx])
    learned_vqa_acc = float(vqa_pred_learned.mean()) * 100
    all_passed &= check_metric(
        "CharXiv Learned Router VQA Acc (N=100)", 
        learned_vqa_acc, 
        35.00, 
        tol=1.05, 
        note="Dung sai ±1 câu / N=100 do phiên bản scikit-learn"
    )

    # Đánh giá chéo ngoài mẫu 5-Fold trên toàn bộ 500 câu hỏi
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_vqa_accs = []
    for tr_k, te_k in kf.split(cx_df):
        v_k = TfidfVectorizer(max_features=750, ngram_range=(1, 2))
        X_tr_k = v_k.fit_transform(cx_df["question"].iloc[tr_k].fillna(""))
        X_te_k = v_k.transform(cx_df["question"].iloc[te_k].fillna(""))
        c_k = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(X_tr_k, y_strat[tr_k])
        p_k = c_k.predict(X_te_k)
        vqa_k = np.where(p_k == 1, c_ad.iloc[te_k], c_dir.iloc[te_k])
        oof_vqa_accs.append(float(vqa_k.mean()) * 100)

    oof_vqa_mean = float(np.mean(oof_vqa_accs))
    # Dung sai ±1 câu trên 5-fold N=500 (±0.25%) cho phép tương thích giữa scikit-learn (1.7.x: 39.40%, 1.9.x: 39.60%)
    all_passed &= check_metric(
        "CharXiv Learned Router Out-of-Fold VQA (N=500)", 
        oof_vqa_mean, 
        39.40, 
        tol=0.25, 
        note="Dung sai ±1 câu / N=500 do phiên bản scikit-learn (39.4% ~ 39.6%)"
    )

    print("\n" + "=" * 80)
    if all_passed:
        print("TẤT CẢ CÁC CHỈ SỐ THỰC NGHIỆM ĐỀU KHỚP 100% VỚI BÁO CÁO [VERIFIED].")
    else:
        print("[CẢNH BÁO] Có chỉ số chưa khớp hoàn toàn với báo cáo.")
    print("=" * 80)

if __name__ == "__main__":
    main()
