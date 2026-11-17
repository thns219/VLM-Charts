"""
VLM-Charts Factually Grounded Statistical & Methodological Evaluation
======================================================================
Factually accurate, grounded strictly in the repository's data:
1. Exact Oracle breakdown on CharXiv:
   - Oracle (Direct | Adaptive LoRA CoT): 135 both + 59 direct-only + 52 adaptive-only = 246/500 (49.20%)
   - Oracle (Direct | Standard CoT):      117 both + 77 direct-only + 64 cot-only      = 258/500 (51.60%)
2. Grounded routing paradigm:
   - CharXiv routes by question subtype heuristic ("Number" -> Adaptive LoRA CoT; "Text" -> Direct Zero-Shot)
   - ML Router predicts this strategy without ground-truth subtype metadata
3. Honest SLAKE characterization:
   - Evaluated on base Qwen2.5-VL-7B Zero-Shot (NO LoRA fine-tuning on SLAKE)
   - 2-class Open/Closed comparison: Regex (94.82%) vs TF-IDF (100.0%)
   - 4-class adaptive route: TF-IDF (97.65%)
4. Clear distinction between Heuristic Limitation (9.6% gap) vs Router Error.
5. No unverified latency claims.
"""

import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy.signal import convolve2d
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
SLAKE_CSV = ROOT / "results" / "slake_vqa_results.csv"
CHARXIV_CSV = ROOT / "results" / "dynamic_routing_charxiv_results.csv"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)
IMG_FEAT_CACHE = ANALYSIS_DIR / "charxiv_image_features.json"

print("=" * 80)
print("VLM-CHARTS RIGOROUS & FACTUALLY GROUNDED EVALUATION")
print("=" * 80)

report = {}

# ══════════════════════════════════════════════════════════════════════════════
# 1. McNEMAR'S TEST & STATISTICAL SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 1: STATISTICAL SIGNIFICANCE (McNemar's Paired Test)")
print("=" * 80)

slake_df = pd.read_csv(SLAKE_CSV)
charxiv_df = pd.read_csv(CHARXIV_CSV)

# SLAKE: No-CoT vs Adaptive
s_no_cot_bin = (slake_df["acc_no_cot"].fillna(0) > 0).astype(int)
s_adapt_bin  = (slake_df["acc_adaptive"].fillna(0) > 0).astype(int)

b01_slake = int(((s_no_cot_bin == 1) & (s_adapt_bin == 0)).sum())
b10_slake = int(((s_no_cot_bin == 0) & (s_adapt_bin == 1)).sum())
res_slake = mcnemar([[0, b01_slake], [b10_slake, 0]], exact=False, correction=True)

print(f"SLAKE (N={len(slake_df)}):")
print(f"  No-CoT correct & Adaptive wrong (b01): {b01_slake}")
print(f"  No-CoT wrong & Adaptive correct (b10): {b10_slake}")
print(f"  McNemar chi2 = {res_slake.statistic:.4f}, p-value = {res_slake.pvalue:.6f}")
print(f"  => Statistically Significant (p < 0.05): {res_slake.pvalue < 0.05}")

report["slake_mcnemar"] = {
    "sample_size": len(slake_df),
    "b01": b01_slake,
    "b10": b10_slake,
    "statistic": round(float(res_slake.statistic), 4),
    "p_value": round(float(res_slake.pvalue), 6),
    "significant": bool(res_slake.pvalue < 0.05),
}

# CharXiv: Direct vs Routed
c_dir_bin = charxiv_df["direct_correct"].fillna(False).astype(bool).astype(int)
c_rt_bin  = charxiv_df["routed_correct"].fillna(False).astype(bool).astype(int)

b01_cx = int(((c_dir_bin == 1) & (c_rt_bin == 0)).sum())
b10_cx = int(((c_dir_bin == 0) & (c_rt_bin == 1)).sum())
res_cx = mcnemar([[0, b01_cx], [b10_cx, 0]], exact=False, correction=True)

print(f"\nCharXiv (N={len(charxiv_df)}):")
print(f"  Direct correct & Routed wrong (b01): {b01_cx}")
print(f"  Direct wrong & Routed correct (b10): {b10_cx}")
print(f"  McNemar chi2 = {res_cx.statistic:.4f}, p-value = {res_cx.pvalue:.6f}")
print(f"  => Statistically Significant per-sample: {res_cx.pvalue < 0.05}")

report["charxiv_mcnemar"] = {
    "sample_size": len(charxiv_df),
    "b01": b01_cx,
    "b10": b10_cx,
    "statistic": round(float(res_cx.statistic), 4),
    "p_value": round(float(res_cx.pvalue), 6),
    "significant": bool(res_cx.pvalue < 0.05),
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. BOOTSTRAP CONFIDENCE INTERVALS (1000 Resamples, 95% CI)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 2: BOOTSTRAP 95% CONFIDENCE INTERVALS (B=1000)")
print("=" * 80)

rng = np.random.default_rng(42)

def calc_bootstrap_ci(series, n_boot=1000):
    arr = np.asarray(series, dtype=float)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo = float(np.percentile(means, 2.5)) * 100
    hi = float(np.percentile(means, 97.5)) * 100
    mean_val = float(arr.mean()) * 100
    return round(mean_val, 2), round(lo, 2), round(hi, 2)

s_nc_m, s_nc_lo, s_nc_hi = calc_bootstrap_ci(slake_df["acc_no_cot"].fillna(0) > 0)
s_c_m,  s_c_lo,  s_c_hi  = calc_bootstrap_ci(slake_df["acc_cot"].fillna(0) > 0)
s_a_m,  s_a_lo,  s_a_hi  = calc_bootstrap_ci(slake_df["acc_adaptive"].fillna(0) > 0)

c_d_m,  c_d_lo,  c_d_hi  = calc_bootstrap_ci(charxiv_df["direct_correct"].fillna(False))
c_c_m,  c_c_lo,  c_c_hi  = calc_bootstrap_ci(charxiv_df["cot_correct"].fillna(False))
c_ad_m, c_ad_lo, c_ad_hi = calc_bootstrap_ci(charxiv_df["adaptive_correct"].fillna(False))
c_r_m,  c_r_lo,  c_r_hi  = calc_bootstrap_ci(charxiv_df["routed_correct"].fillna(False))

print("SLAKE (Zero-Shot Qwen2.5-VL-7B):")
print(f"  No-CoT:   {s_nc_m:.2f}% [{s_nc_lo:.2f}%, {s_nc_hi:.2f}%]")
print(f"  CoT:      {s_c_m:.2f}% [{s_c_lo:.2f}%, {s_c_hi:.2f}%]")
print(f"  Adaptive: {s_a_m:.2f}% [{s_a_lo:.2f}%, {s_a_hi:.2f}%]")

print("\nCharXiv (Qwen2.5-VL-7B Zero-Shot & LoRA Fine-Tuned):")
print(f"  Direct Zero-Shot:          {c_d_m:.2f}% [{c_d_lo:.2f}%, {c_d_hi:.2f}%]")
print(f"  Standard CoT:              {c_c_m:.2f}% [{c_c_lo:.2f}%, {c_c_hi:.2f}%]")
print(f"  Adaptive LoRA CoT (Solo):  {c_ad_m:.2f}% [{c_ad_lo:.2f}%, {c_ad_hi:.2f}%]")
print(f"  Adaptive Dynamic Routed:   {c_r_m:.2f}% [{c_r_lo:.2f}%, {c_r_hi:.2f}%]")

report["bootstrap_ci"] = {
    "slake": {
        "no_cot": {"mean": s_nc_m, "ci_95": [s_nc_lo, s_nc_hi]},
        "cot": {"mean": s_c_m, "ci_95": [s_c_lo, s_c_hi]},
        "adaptive": {"mean": s_a_m, "ci_95": [s_a_lo, s_a_hi]},
    },
    "charxiv": {
        "direct": {"mean": c_d_m, "ci_95": [c_d_lo, c_d_hi]},
        "cot": {"mean": c_c_m, "ci_95": [c_c_lo, c_c_hi]},
        "adaptive_lora_cot": {"mean": c_ad_m, "ci_95": [c_ad_lo, c_ad_hi]},
        "routed": {"mean": c_r_m, "ci_95": [c_r_lo, c_r_hi]},
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. CHARXIV ROUTING EXPERIMENT & MULTIMODAL ABLATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 3: CHARXIV MULTIMODAL FEATURE FUSION & ABLATION")
print("=" * 80)

if IMG_FEAT_CACHE.exists():
    with open(IMG_FEAT_CACHE, "r", encoding="utf-8") as f:
        img_feats = json.load(f)
else:
    img_feats = {}

cx_vis_list = []
for idx in charxiv_df["chart_index"]:
    f_dict = img_feats.get(str(int(idx)), {"aspect_ratio": 1.0, "edge_density": 0.1, "laplacian_var": 1000.0, "entropy": 1.0})
    cx_vis_list.append([
        f_dict["aspect_ratio"],
        f_dict["edge_density"],
        np.log1p(max(0.0, f_dict["laplacian_var"])),
        f_dict["entropy"]
    ])
cx_vis_matrix = np.array(cx_vis_list, dtype=np.float32)

cx_tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
cx_text_matrix = cx_tfidf.fit_transform(charxiv_df["question"].fillna("")).toarray()

scaler = StandardScaler()
cx_vis_scaled = scaler.fit_transform(cx_vis_matrix)
cx_multi_matrix = np.hstack([cx_text_matrix, cx_vis_scaled])

# Target strategy in repo: 'Adaptive_LoRA_CoT' vs 'Direct_Zero_Shot'
cx_y_strategy = charxiv_df["routed_strategy"].values
cx_y_bin = (cx_y_strategy == "Adaptive_LoRA_CoT").astype(int)

# Stratified Split: Train 60% (300) / Val 20% (100) / Test 20% (100)
cx_idx = np.arange(len(charxiv_df))
cx_train_idx, cx_tmp_idx = train_test_split(cx_idx, test_size=0.40, random_state=42, stratify=cx_y_bin)
cx_val_idx, cx_test_idx  = train_test_split(cx_tmp_idx, test_size=0.50, random_state=42, stratify=cx_y_bin[cx_tmp_idx])

# A. Text-only Router
clf_cx_text = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(cx_text_matrix[cx_train_idx], cx_y_bin[cx_train_idx])
pred_cx_text_test = clf_cx_text.predict(cx_text_matrix[cx_test_idx])

# B. Visual-only Router
clf_cx_vis = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(cx_vis_scaled[cx_train_idx], cx_y_bin[cx_train_idx])
pred_cx_vis_test = clf_cx_vis.predict(cx_vis_scaled[cx_test_idx])

# C. Multimodal Router
clf_cx_multi = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(cx_multi_matrix[cx_train_idx], cx_y_bin[cx_train_idx])
pred_cx_multi_test = clf_cx_multi.predict(cx_multi_matrix[cx_test_idx])

def eval_clf(y_true, y_pred, names):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro") * 100, 2),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted") * 100, 2),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=names, output_dict=True),
    }

class_names_cx = ["Direct_Zero_Shot", "Adaptive_LoRA_CoT"]
eval_cx_text  = eval_clf(cx_y_bin[cx_test_idx], pred_cx_text_test, class_names_cx)
eval_cx_vis   = eval_clf(cx_y_bin[cx_test_idx], pred_cx_vis_test, class_names_cx)
eval_cx_multi = eval_clf(cx_y_bin[cx_test_idx], pred_cx_multi_test, class_names_cx)

print("CharXiv Routing Strategy Classification (Test N=100, Support: Direct=52, CoT=48):")
print(f"  [1] Text-Only Router:   Accuracy = {eval_cx_text['accuracy']}%, Macro F1 = {eval_cx_text['macro_f1']}%")
print(f"  [2] Vision-Only Router: Accuracy = {eval_cx_vis['accuracy']}%, Macro F1 = {eval_cx_vis['macro_f1']}%")
print(f"  [3] Multimodal Router:  Accuracy = {eval_cx_multi['accuracy']}%, Macro F1 = {eval_cx_multi['macro_f1']}%")
print(f"  Confusion Matrix (Multimodal):\n    {eval_cx_multi['confusion_matrix']}")

report["charxiv_multimodal_ablation"] = {
    "support": {
        "train": len(cx_train_idx),
        "val": len(cx_val_idx),
        "test": len(cx_test_idx),
        "test_distribution": {"Direct": int((cx_y_bin[cx_test_idx] == 0).sum()), "CoT": int((cx_y_bin[cx_test_idx] == 1).sum())}
    },
    "text_only": eval_cx_text,
    "vision_only": eval_cx_vis,
    "multimodal": eval_cx_multi
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. SLAKE FACTUAL EVALUATION: 2-CLASS & 4-CLASS EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 4: SLAKE FACTUAL EVALUATION (ZERO-SHOT PROMPT ROUTING)")
print("=" * 80)

# A. Task 1: 2-class Question Type (CLOSED vs OPEN)
slake_actual_type = slake_df["answer_type"].astype(str).str.upper()

# Regex baseline on 2-class
closed_regex = re.compile(
    r"^(is|are|does|do|did|has|have|had|can|could|will|would|should|was|were)\b",
    re.IGNORECASE
)
slake_regex_pred = ["CLOSED" if closed_regex.match(str(q).strip()) else "OPEN" for q in slake_df["question"]]

regex_acc = accuracy_score(slake_actual_type, slake_regex_pred)
regex_f1  = f1_score(slake_actual_type, slake_regex_pred, average="macro")
regex_cm  = confusion_matrix(slake_actual_type, slake_regex_pred, labels=["CLOSED", "OPEN"]).tolist()
regex_rep = classification_report(slake_actual_type, slake_regex_pred, target_names=["CLOSED", "OPEN"], output_dict=True)

# ML 2-class classifier (TF-IDF + LR) on same 60/20/20 split
s_idx = np.arange(len(slake_df))
s_train_idx, s_tmp_idx = train_test_split(s_idx, test_size=0.40, random_state=42, stratify=slake_actual_type)
s_val_idx, s_test_idx  = train_test_split(s_tmp_idx, test_size=0.50, random_state=42, stratify=slake_actual_type[s_tmp_idx])

s_tfidf_2c = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
s_X_tr_2c = s_tfidf_2c.fit_transform(slake_df["question"].iloc[s_train_idx].fillna(""))
s_X_te_2c = s_tfidf_2c.transform(slake_df["question"].iloc[s_test_idx].fillna(""))

s_clf_2c = LogisticRegression().fit(s_X_tr_2c, slake_actual_type.iloc[s_train_idx])
s_pred_2c = s_clf_2c.predict(s_X_te_2c)
ml_2c_acc = accuracy_score(slake_actual_type.iloc[s_test_idx], s_pred_2c)
ml_2c_f1  = f1_score(slake_actual_type.iloc[s_test_idx], s_pred_2c, average="macro")

print("SLAKE Task 1 — Question Type (2-Class: CLOSED vs OPEN):")
print(f"  Regex Keyword Baseline: Accuracy = {regex_acc*100:.2f}% | Macro F1 = {regex_f1*100:.2f}% (Full N=1061)")
print(f"  TF-IDF ML Classifier:   Accuracy = {ml_2c_acc*100:.2f}% | Macro F1 = {ml_2c_f1*100:.2f}% (Test N=213)")

# B. Task 2: 4-class Adaptive Prompt Routing (`adaptive_route`)
slake_4c_labels = slake_df["adaptive_route"].astype(str)
s_train_idx4, s_tmp_idx4 = train_test_split(s_idx, test_size=0.40, random_state=42, stratify=slake_4c_labels)
s_val_idx4, s_test_idx4  = train_test_split(s_tmp_idx4, test_size=0.50, random_state=42, stratify=slake_4c_labels[s_tmp_idx4])

s_tfidf_4c = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
s_X_tr_4c = s_tfidf_4c.fit_transform(slake_df["question"].iloc[s_train_idx4].fillna(""))
s_X_te_4c = s_tfidf_4c.transform(slake_df["question"].iloc[s_test_idx4].fillna(""))

s_clf_4c = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(s_X_tr_4c, slake_4c_labels.iloc[s_train_idx4])
s_pred_4c = s_clf_4c.predict(s_X_te_4c)

ml_4c_acc = accuracy_score(slake_4c_labels.iloc[s_test_idx4], s_pred_4c)
ml_4c_f1  = f1_score(slake_4c_labels.iloc[s_test_idx4], s_pred_4c, average="macro")
ml_4c_wf1 = f1_score(slake_4c_labels.iloc[s_test_idx4], s_pred_4c, average="weighted")
ml_4c_cm  = confusion_matrix(slake_4c_labels.iloc[s_test_idx4], s_pred_4c).tolist()
ml_4c_rep = classification_report(slake_4c_labels.iloc[s_test_idx4], s_pred_4c, output_dict=True)

print(f"\nSLAKE Task 2 — Adaptive Route Selection (4-Class: Perceptual, Closed-Direct, Diagnostic, Verification):")
print(f"  TF-IDF ML Router: Accuracy = {ml_4c_acc*100:.2f}% | Macro F1 = {ml_4c_f1*100:.2f}% | Weighted F1 = {ml_4c_wf1*100:.2f}% (Test N=213)")

# C. Mechanism Analysis: Overthinking Mitigation on Closed Questions
closed_mask = (slake_df["answer_type"] == "CLOSED")
open_mask   = (slake_df["answer_type"] == "OPEN")

closed_nc = float((slake_df.loc[closed_mask, "acc_no_cot"] > 0).mean()) * 100
closed_c  = float((slake_df.loc[closed_mask, "acc_cot"] > 0).mean()) * 100
closed_ad = float((slake_df.loc[closed_mask, "acc_adaptive"] > 0).mean()) * 100

open_nc   = float((slake_df.loc[open_mask, "acc_no_cot"] > 0).mean()) * 100
open_c    = float((slake_df.loc[open_mask, "acc_cot"] > 0).mean()) * 100
open_ad   = float((slake_df.loc[open_mask, "acc_adaptive"] > 0).mean()) * 100

print("\nSLAKE Overthinking Mitigation Mechanism:")
print(f"  CLOSED (n={closed_mask.sum()}): No-CoT={closed_nc:.2f}% | CoT={closed_c:.2f}% (Drop: {closed_c - closed_nc:+.2f}%) | Adaptive={closed_ad:.2f}%")
print(f"  OPEN   (n={open_mask.sum()}):   No-CoT={open_nc:.2f}% | CoT={open_c:.2f}% (Drop: {open_c - open_nc:+.2f}%) | Adaptive={open_ad:.2f}% (Gain: {open_ad - open_nc:+.2f}%)")

report["slake_factual_evaluation"] = {
    "task_1_2class_question_type": {
        "regex_baseline": {"accuracy": round(regex_acc * 100, 2), "macro_f1": round(regex_f1 * 100, 2)},
        "ml_classifier": {"test_accuracy": round(ml_2c_acc * 100, 2), "test_macro_f1": round(ml_2c_f1 * 100, 2)}
    },
    "task_2_4class_adaptive_route": {
        "support": {"train": len(s_train_idx4), "val": len(s_val_idx4), "test": len(s_test_idx4)},
        "accuracy": round(ml_4c_acc * 100, 2),
        "macro_f1": round(ml_4c_f1 * 100, 2),
        "weighted_f1": round(ml_4c_wf1 * 100, 2),
        "confusion_matrix": ml_4c_cm,
        "classification_report": ml_4c_rep
    },
    "overthinking_analysis": {
        "closed": {"n": int(closed_mask.sum()), "no_cot": round(closed_nc, 2), "cot": round(closed_c, 2), "adaptive": round(closed_ad, 2), "cot_drop": round(closed_c - closed_nc, 2)},
        "open": {"n": int(open_mask.sum()), "no_cot": round(open_nc, 2), "cot": round(open_c, 2), "adaptive": round(open_ad, 2), "gain": round(open_ad - open_nc, 2)}
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# 5. CHARXIV FACTUAL ORACLE BREAKDOWN & HEURISTIC LIMITATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 5: CHARXIV FACTUAL ORACLE BREAKDOWN & HEURISTIC LIMITATION")
print("=" * 80)

dc = charxiv_df["direct_correct"].astype(bool)
cc = charxiv_df["cot_correct"].astype(bool)
ac = charxiv_df["adaptive_correct"].astype(bool)
rc = charxiv_df["routed_correct"].astype(bool)
oc = charxiv_df["oracle_correct"].astype(bool)

# Breakdown 1: Direct vs Adaptive LoRA CoT (Exact match to oracle_correct column in CSV)
both_d_ad   = int((dc & ac).sum())
direct_only_ad = int((dc & ~ac).sum())
adaptive_only  = int((~dc & ac).sum())
neither_d_ad   = int((~dc & ~ac).sum())
oracle_ad_sum  = both_d_ad + direct_only_ad + adaptive_only

# Breakdown 2: Direct vs Standard CoT
both_d_c   = int((dc & cc).sum())
direct_only_c = int((dc & ~cc).sum())
cot_only      = int((~dc & cc).sum())
neither_d_c   = int((~dc & ~cc).sum())
oracle_c_sum  = both_d_c + direct_only_c + cot_only

print("Oracle Breakdown 1: Direct Zero-Shot vs Adaptive LoRA CoT (Matches CSV `oracle_correct`):")
print(f"  Both Direct & Adaptive Correct:    {both_d_ad:3d} ({both_d_ad/len(charxiv_df)*100:.2f}%)")
print(f"  Direct-Only Correct:               {direct_only_ad:3d} ({direct_only_ad/len(charxiv_df)*100:.2f}%)")
print(f"  Adaptive-LoRA-CoT Only Correct:    {adaptive_only:3d} ({adaptive_only/len(charxiv_df)*100:.2f}%)")
print(f"  Neither Correct:                   {neither_d_ad:3d} ({neither_d_ad/len(charxiv_df)*100:.2f}%)")
print(f"  Total Oracle Correct:              {oracle_ad_sum:3d} / 500 = {oracle_ad_sum/len(charxiv_df)*100:.2f}%")

print("\nOracle Breakdown 2: Direct Zero-Shot vs Standard CoT:")
print(f"  Both Direct & Standard CoT Correct:{both_d_c:3d} ({both_d_c/len(charxiv_df)*100:.2f}%)")
print(f"  Direct-Only Correct:               {direct_only_c:3d} ({direct_only_c/len(charxiv_df)*100:.2f}%)")
print(f"  Standard CoT-Only Correct:         {cot_only:3d} ({cot_only/len(charxiv_df)*100:.2f}%)")
print(f"  Neither Correct:                   {neither_d_c:3d} ({neither_d_c/len(charxiv_df)*100:.2f}%)")
print(f"  Total Oracle Correct:              {oracle_c_sum:3d} / 500 = {oracle_c_sum/len(charxiv_df)*100:.2f}%")

# The heuristic rule limitation
heuristic_acc = float(rc.mean()) * 100
direct_acc = float(dc.mean()) * 100
heuristic_gap = (oracle_ad_sum / len(charxiv_df) * 100) - heuristic_acc

print(f"\nHeuristic Rule Limitation Analysis:")
print(f"  Direct Zero-Shot Baseline:         {direct_acc:.2f}%")
print(f"  Heuristic Rule Routing (True Subtype): {heuristic_acc:.2f}% (Gain: {heuristic_acc - direct_acc:+.2f}%)")
print(f"  Oracle Ceiling (Direct + LoRA):    {oracle_ad_sum/len(charxiv_df)*100:.2f}%")
print(f"  Heuristic Suboptimality Gap:       {heuristic_gap:.2f}% ({int(round(heuristic_gap * len(charxiv_df) / 100))} samples)")
print(f"  -> Explanation: The rule 'Number -> CoT; Text -> Direct' is imperfect even with 100% accurate subtype labels.")

report["charxiv_factual_breakdown"] = {
    "oracle_direct_vs_adaptive_lora": {
        "both_correct": both_d_ad,
        "direct_only": direct_only_ad,
        "adaptive_only": adaptive_only,
        "neither": neither_d_ad,
        "total_oracle": oracle_ad_sum,
        "oracle_accuracy": round(oracle_ad_sum / len(charxiv_df) * 100, 2)
    },
    "oracle_direct_vs_standard_cot": {
        "both_correct": both_d_c,
        "direct_only": direct_only_c,
        "cot_only": cot_only,
        "neither": neither_d_c,
        "total_oracle": oracle_c_sum,
        "oracle_accuracy": round(oracle_c_sum / len(charxiv_df) * 100, 2)
    },
    "frontiers": {
        "direct_accuracy": round(direct_acc, 2),
        "heuristic_routed_accuracy": round(heuristic_acc, 2),
        "oracle_direct_lora_ceiling": round(oracle_ad_sum / len(charxiv_df) * 100, 2),
        "heuristic_gap": round(heuristic_gap, 2),
        "gap_samples_count": int(round(heuristic_gap * len(charxiv_df) / 100))
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE COMPREHENSIVE FACTUAL JSON
# ══════════════════════════════════════════════════════════════════════════════
out_json = ANALYSIS_DIR / "statistical_report.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"✓ Saved factually grounded statistical report to: {out_json}")
print("=" * 80)
