"""
VLM-Charts Statistical Analysis
================================
A. McNemar's test (Adaptive vs Direct)
B. Bootstrap CI (1000 resamples)
C. Text classifier router (TF-IDF + LogisticRegression)
D. SLAKE validation split re-analysis
E. Output statistical_report.json
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
SLAKE_CSV = ROOT / "results" / "slake_vqa_results.csv"
CHARXIV_CSV = ROOT / "results" / "dynamic_routing_charxiv_results.csv"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

report = {}

print("=" * 70)
print("VLM-CHARTS STATISTICAL ANALYSIS")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# A.  McNemar's Test
# ══════════════════════════════════════════════════════════════════════════════
print("\n── A. McNemar's Test ──────────────────────────────────────────────────")

from statsmodels.stats.contingency_tables import mcnemar

# ── A1. SLAKE: acc_no_cot vs acc_adaptive ─────────────────────────────────────
slake = pd.read_csv(SLAKE_CSV)
print(f"SLAKE rows: {len(slake)}")

# binarise (treat values > 0 as correct)
no_cot  = (slake["acc_no_cot"].fillna(0) > 0).astype(int)
adaptive = (slake["acc_adaptive"].fillna(0) > 0).astype(int)

# McNemar contingency table: b = no_cot correct & adaptive wrong, c = vice versa
b01 = int(((no_cot == 1) & (adaptive == 0)).sum())   # no_cot ✓, adaptive ✗
b10 = int(((no_cot == 0) & (adaptive == 1)).sum())   # no_cot ✗, adaptive ✓

table_slake = np.array([[0, b01], [b10, 0]])  # off-diagonal only matters
result_slake = mcnemar(table_slake, exact=False, correction=True)

print(f"\nSLAKE McNemar (No-CoT vs Adaptive):")
print(f"  b01 (no_cot ✓, adaptive ✗) = {b01}")
print(f"  b10 (no_cot ✗, adaptive ✓) = {b10}")
print(f"  chi2  = {result_slake.statistic:.4f}")
print(f"  p     = {result_slake.pvalue:.6f}")
sig_slake = bool(result_slake.pvalue < 0.05)
print(f"  Significant (p<0.05)? {sig_slake}")

report["slake_mcnemar"] = {
    "statistic": round(float(result_slake.statistic), 4),
    "p_value":   round(float(result_slake.pvalue), 6),
    "b01": b01,
    "b10": b10,
    "significant": sig_slake,
}

# ── A2. CharXiv: direct_correct vs routed_correct ─────────────────────────────
charxiv = pd.read_csv(CHARXIV_CSV)
print(f"\nCharXiv rows: {len(charxiv)}")

dc = (charxiv["direct_correct"].fillna(False).astype(bool)).astype(int)
rc = (charxiv["routed_correct"].fillna(False).astype(bool)).astype(int)

b01_cx = int(((dc == 1) & (rc == 0)).sum())
b10_cx = int(((dc == 0) & (rc == 1)).sum())

table_cx = np.array([[0, b01_cx], [b10_cx, 0]])
result_cx = mcnemar(table_cx, exact=False, correction=True)

print(f"\nCharXiv McNemar (Direct vs Routed):")
print(f"  b01 (direct ✓, routed ✗) = {b01_cx}")
print(f"  b10 (direct ✗, routed ✓) = {b10_cx}")
print(f"  chi2  = {result_cx.statistic:.4f}")
print(f"  p     = {result_cx.pvalue:.6f}")
sig_cx = bool(result_cx.pvalue < 0.05)
print(f"  Significant (p<0.05)? {sig_cx}")

report["charxiv_mcnemar"] = {
    "statistic": round(float(result_cx.statistic), 4),
    "p_value":   round(float(result_cx.pvalue), 6),
    "b01": b01_cx,
    "b10": b10_cx,
    "significant": sig_cx,
}

# ══════════════════════════════════════════════════════════════════════════════
# B.  Bootstrap CI (1000 resamples, 95%)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── B. Bootstrap CI (1000 resamples) ───────────────────────────────────")

rng = np.random.default_rng(42)

def bootstrap_ci(arr, n_boot=1000, ci=95):
    """Return (mean, lo, hi) for a binary array."""
    arr = np.asarray(arr, dtype=float)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo = float(np.percentile(means, (100 - ci) / 2))
    hi = float(np.percentile(means, 100 - (100 - ci) / 2))
    return round(arr.mean() * 100, 4), round(lo * 100, 4), round(hi * 100, 4)

# SLAKE
s_no_cot = (slake["acc_no_cot"].fillna(0) > 0).astype(float)
s_cot    = (slake["acc_cot"].fillna(0) > 0).astype(float)
s_adapt  = (slake["acc_adaptive"].fillna(0) > 0).astype(float)

m1, lo1, hi1 = bootstrap_ci(s_no_cot)
m2, lo2, hi2 = bootstrap_ci(s_cot)
m3, lo3, hi3 = bootstrap_ci(s_adapt)

print(f"\nSLAKE Bootstrap 95% CI:")
print(f"  No-CoT:   {m1:.2f}%  [{lo1:.2f}%, {hi1:.2f}%]")
print(f"  CoT:      {m2:.2f}%  [{lo2:.2f}%, {hi2:.2f}%]")
print(f"  Adaptive: {m3:.2f}%  [{lo3:.2f}%, {hi3:.2f}%]")

report["slake_bootstrap_ci"] = {
    "no_cot":   [lo1, hi1],
    "cot":      [lo2, hi2],
    "adaptive": [lo3, hi3],
    "no_cot_mean":   m1,
    "cot_mean":      m2,
    "adaptive_mean": m3,
}

# CharXiv
c_direct = charxiv["direct_correct"].fillna(False).astype(float)
c_cot    = charxiv["cot_correct"].fillna(False).astype(float)
c_routed = charxiv["routed_correct"].fillna(False).astype(float)

d1, dlo1, dhi1 = bootstrap_ci(c_direct)
d2, dlo2, dhi2 = bootstrap_ci(c_cot)
d3, dlo3, dhi3 = bootstrap_ci(c_routed)

print(f"\nCharXiv Bootstrap 95% CI:")
print(f"  Direct: {d1:.2f}%  [{dlo1:.2f}%, {dhi1:.2f}%]")
print(f"  CoT:    {d2:.2f}%  [{dlo2:.2f}%, {dhi2:.2f}%]")
print(f"  Routed: {d3:.2f}%  [{dlo3:.2f}%, {dhi3:.2f}%]")

report["charxiv_bootstrap_ci"] = {
    "direct": [dlo1, dhi1],
    "cot":    [dlo2, dhi2],
    "routed": [dlo3, dhi3],
    "direct_mean": d1,
    "cot_mean":    d2,
    "routed_mean": d3,
}

# ══════════════════════════════════════════════════════════════════════════════
# C.  Text Classifier Router (TF-IDF + LogisticRegression)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── C. Text Classifier Router ───────────────────────────────────────────")

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_text_router(df, text_col, label_col, save_path, name):
    X = df[text_col].fillna("").astype(str).values
    y = df[label_col].fillna("unknown").astype(str).values

    # Drop rows with unknown label
    mask = y != "unknown"
    X, y = X[mask], y[mask]

    # 60/20/20 stratified split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    val_acc  = accuracy_score(y_val,  pipe.predict(X_val))
    test_acc = accuracy_score(y_test, pipe.predict(X_test))

    print(f"\n{name} Text Router:")
    print(f"  Train size: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Val accuracy:  {val_acc*100:.2f}%")
    print(f"  Test accuracy: {test_acc*100:.2f}%")
    print("  Classification report (test):")
    print(classification_report(y_test, pipe.predict(X_test), zero_division=0))

    joblib.dump(pipe, save_path)
    print(f"  Model saved → {save_path}")
    return round(val_acc * 100, 4), round(test_acc * 100, 4)

# SLAKE
slake_val_acc, slake_test_acc = train_text_router(
    slake, "question", "adaptive_route",
    ANALYSIS_DIR / "slake_text_router.pkl",
    "SLAKE"
)
report["slake_text_router"] = {
    "val_accuracy": slake_val_acc,
    "test_accuracy": slake_test_acc,
}

# CharXiv
charxiv_val_acc, charxiv_test_acc = train_text_router(
    charxiv, "question", "routed_strategy",
    ANALYSIS_DIR / "charxiv_text_router.pkl",
    "CharXiv"
)
report["charxiv_text_router"] = {
    "val_accuracy": charxiv_val_acc,
    "test_accuracy": charxiv_test_acc,
}

# ══════════════════════════════════════════════════════════════════════════════
# D.  SLAKE Validation Split Re-analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n── D. SLAKE Proper Split Re-analysis ──────────────────────────────────")

idx_all = np.arange(len(slake))
idx_train, idx_tmp = train_test_split(idx_all, test_size=0.40, random_state=42)
idx_val, idx_test  = train_test_split(idx_tmp,   test_size=0.50, random_state=42)

def split_acc(df, idx):
    sub = df.iloc[idx]
    no_cot   = (sub["acc_no_cot"].fillna(0) > 0).mean() * 100
    cot      = (sub["acc_cot"].fillna(0) > 0).mean() * 100
    adaptive = (sub["acc_adaptive"].fillna(0) > 0).mean() * 100
    return round(no_cot, 4), round(cot, 4), round(adaptive, 4)

val_nc, val_c, val_a   = split_acc(slake, idx_val)
test_nc, test_c, test_a = split_acc(slake, idx_test)

print(f"\n  Val  ({len(idx_val):4d} samples): No-CoT={val_nc:.2f}%  CoT={val_c:.2f}%  Adaptive={val_a:.2f}%")
print(f"  Test ({len(idx_test):4d} samples): No-CoT={test_nc:.2f}%  CoT={test_c:.2f}%  Adaptive={test_a:.2f}%")
print(f"  Full ({len(slake):4d} samples): No-CoT={m1:.2f}%  CoT={m2:.2f}%  Adaptive={m3:.2f}%")

report["slake_proper_split"] = {
    "val":  {"no_cot": val_nc,  "cot": val_c,  "adaptive": val_a},
    "test": {"no_cot": test_nc, "cot": test_c, "adaptive": test_a},
    "full": {"no_cot": m1,      "cot": m2,     "adaptive": m3},
}

# ══════════════════════════════════════════════════════════════════════════════
# E.  Save JSON report
# ══════════════════════════════════════════════════════════════════════════════
out_path = ANALYSIS_DIR / "statistical_report.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print(f"✓ Statistical report saved → {out_path}")
print("=" * 70)
print("\nFull JSON:")
print(json.dumps(report, indent=2, ensure_ascii=False))
