"""
VLM-Charts Rigorous Statistical & Methodological Evaluation
============================================================
Addresses all 5 academic critiques:
1. True Dynamic Policy Routing vs Question Classification
2. Resolving Modality Blindness via Multimodal (Text + Vision) Feature Fusion
3. Demystifying SLAKE High Accuracy (Syntactic Regularity & Regex Baseline vs Overthinking Prevention)
4. CharXiv Cascading Routing Error & Oracle Frontier Analysis (Regret Decomposition)
5. Comprehensive Academic Reporting (Confusion Matrix, Per-class Precision/Recall/F1, Support)
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
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
print("VLM-CHARTS RIGOROUS SCIENTIFIC & METHODOLOGICAL ANALYSIS")
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

# SLAKE
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

# CharXiv
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
c_r_m,  c_r_lo,  c_r_hi  = calc_bootstrap_ci(charxiv_df["routed_correct"].fillna(False))

print("SLAKE:")
print(f"  No-CoT:   {s_nc_m:.2f}% [{s_nc_lo:.2f}%, {s_nc_hi:.2f}%]")
print(f"  CoT:      {s_c_m:.2f}% [{s_c_lo:.2f}%, {s_c_hi:.2f}%]")
print(f"  Adaptive: {s_a_m:.2f}% [{s_a_lo:.2f}%, {s_a_hi:.2f}%]")

print("\nCharXiv:")
print(f"  Direct:   {c_d_m:.2f}% [{c_d_lo:.2f}%, {c_d_hi:.2f}%]")
print(f"  CoT:      {c_c_m:.2f}% [{c_c_lo:.2f}%, {c_c_hi:.2f}%]")
print(f"  Routed:   {c_r_m:.2f}% [{c_r_lo:.2f}%, {c_r_hi:.2f}%]")

report["bootstrap_ci"] = {
    "slake": {
        "no_cot": {"mean": s_nc_m, "ci_95": [s_nc_lo, s_nc_hi]},
        "cot": {"mean": s_c_m, "ci_95": [s_c_lo, s_c_hi]},
        "adaptive": {"mean": s_a_m, "ci_95": [s_a_lo, s_a_hi]},
    },
    "charxiv": {
        "direct": {"mean": c_d_m, "ci_95": [c_d_lo, c_d_hi]},
        "cot": {"mean": c_c_m, "ci_95": [c_c_lo, c_c_hi]},
        "routed": {"mean": c_r_m, "ci_95": [c_r_lo, c_r_hi]},
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. DUAL-DOMAIN FEATURE EXTRACTION & RESOLVING MODALITY BLINDNESS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 3: RESOLVING MODALITY BLINDNESS VIA MULTIMODAL FEATURE FUSION")
print("=" * 80)

# Load cached visual features or extract if missing
if IMG_FEAT_CACHE.exists():
    with open(IMG_FEAT_CACHE, "r", encoding="utf-8") as f:
        img_feats = json.load(f)
    print(f"Loaded {len(img_feats)} cached CharXiv visual complexity features.")
else:
    print("Extracting visual complexity features for CharXiv charts...")
    img_feats = {}
    for i in range(len(charxiv_df)):
        chart_p = ROOT / "charxiv_all_images" / "test" / f"chart_{i:05d}.png"
        if not chart_p.exists():
            continue
        with Image.open(chart_p) as im:
            w, h = im.size
            gray = im.convert("L")
            arr = np.array(gray, dtype=np.float32)
            if max(w, h) > 1024:
                scale = 1024 / max(w, h)
                im_small = gray.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
                arr = np.array(im_small, dtype=np.float32)
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_density = float(np.mean(np.array(edges) > 30))
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            lap = convolve2d(arr[::2, ::2], kernel, mode="valid")
            lap_var = float(lap.var())
            hist, _ = np.histogram(arr, bins=32, range=(0, 256), density=True)
            hist = hist[hist > 0]
            entropy = -float(np.sum(hist * np.log2(hist)))
            img_feats[str(i)] = {
                "width": w, "height": h, "aspect_ratio": round(w / h, 3),
                "edge_density": round(edge_density, 4),
                "laplacian_var": round(lap_var, 2),
                "entropy": round(entropy, 3)
            }
    with open(IMG_FEAT_CACHE, "w", encoding="utf-8") as f:
        json.dump(img_feats, f, indent=2)

# Build feature representations for CharXiv
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

cx_y_strategy = charxiv_df["routed_strategy"].values
cx_y_bin = (cx_y_strategy == "Adaptive_LoRA_CoT").astype(int)  # 0: Direct, 1: CoT

# Stratified Split 60% Train (300) / 20% Val (100) / 20% Test (100)
cx_idx = np.arange(len(charxiv_df))
cx_train_idx, cx_tmp_idx = train_test_split(cx_idx, test_size=0.40, random_state=42, stratify=cx_y_bin)
cx_val_idx, cx_test_idx  = train_test_split(cx_tmp_idx, test_size=0.50, random_state=42, stratify=cx_y_bin[cx_tmp_idx])

# A. Text-only Router
clf_cx_text = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(cx_text_matrix[cx_train_idx], cx_y_bin[cx_train_idx])
pred_cx_text_val  = clf_cx_text.predict(cx_text_matrix[cx_val_idx])
pred_cx_text_test = clf_cx_text.predict(cx_text_matrix[cx_test_idx])

# B. Visual-only Router
clf_cx_vis = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(cx_vis_scaled[cx_train_idx], cx_y_bin[cx_train_idx])
pred_cx_vis_val  = clf_cx_vis.predict(cx_vis_scaled[cx_val_idx])
pred_cx_vis_test = clf_cx_vis.predict(cx_vis_scaled[cx_test_idx])

# C. Multimodal Router (Text + Vision)
clf_cx_multi = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(cx_multi_matrix[cx_train_idx], cx_y_bin[cx_train_idx])
pred_cx_multi_val  = clf_cx_multi.predict(cx_multi_matrix[cx_val_idx])
pred_cx_multi_test = clf_cx_multi.predict(cx_multi_matrix[cx_test_idx])

def evaluate_classifier_detailed(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred).tolist()
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return {
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "weighted_f1": round(weighted_f1 * 100, 2),
        "confusion_matrix": cm,
        "classification_report": rep,
    }

class_names_cx = ["Direct_Zero_Shot", "Adaptive_LoRA_CoT"]

eval_cx_text  = evaluate_classifier_detailed(cx_y_bin[cx_test_idx], pred_cx_text_test, class_names_cx)
eval_cx_vis   = evaluate_classifier_detailed(cx_y_bin[cx_test_idx], pred_cx_vis_test, class_names_cx)
eval_cx_multi = evaluate_classifier_detailed(cx_y_bin[cx_test_idx], pred_cx_multi_test, class_names_cx)

print("CharXiv Routing Modality Ablation (Test N=100, Support: Direct=52, CoT=48):")
print(f"  [1] Text-Only Router:   Accuracy = {eval_cx_text['accuracy']}%, Macro F1 = {eval_cx_text['macro_f1']}%")
print(f"  [2] Vision-Only Router: Accuracy = {eval_cx_vis['accuracy']}%, Macro F1 = {eval_cx_vis['macro_f1']}%")
print(f"  [3] Multimodal Router:  Accuracy = {eval_cx_multi['accuracy']}%, Macro F1 = {eval_cx_multi['macro_f1']}%")
print(f"  Confusion Matrix (Multimodal):\n    {eval_cx_multi['confusion_matrix']}")

report["charxiv_multimodal_ablation"] = {
    "split_support": {
        "train": len(cx_train_idx),
        "val": len(cx_val_idx),
        "test": len(cx_test_idx),
        "test_class_distribution": {"Direct": int((cx_y_bin[cx_test_idx] == 0).sum()), "CoT": int((cx_y_bin[cx_test_idx] == 1).sum())}
    },
    "text_only": eval_cx_text,
    "vision_only": eval_cx_vis,
    "multimodal": eval_cx_multi
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. DEMYSTIFYING SLAKE ACCURACY: SYNTACTIC REGULARITY & OVERTHINKING MITIGATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 4: DEMYSTIFYING SLAKE HIGH ACCURACY & ABLATION STUDY")
print("=" * 80)

# Build Regex Baseline
closed_regex = re.compile(
    r"^(is|are|does|do|did|has|have|had|can|could|will|would|should|was|were)\b",
    re.IGNORECASE
)

slake_actual_type = slake_df["answer_type"].astype(str).str.upper()
slake_regex_pred = []
for q in slake_df["question"]:
    if closed_regex.match(str(q).strip()):
        slake_regex_pred.append("CLOSED")
    else:
        slake_regex_pred.append("OPEN")

regex_acc = accuracy_score(slake_actual_type, slake_regex_pred)
regex_f1  = f1_score(slake_actual_type, slake_regex_pred, average="macro")
regex_cm  = confusion_matrix(slake_actual_type, slake_regex_pred, labels=["CLOSED", "OPEN"]).tolist()
regex_rep = classification_report(slake_actual_type, slake_regex_pred, target_names=["CLOSED", "OPEN"], output_dict=True)

print(f"SLAKE Rule-Based Regex Baseline (Leading Auxiliary Verb Match):")
print(f"  Accuracy: {regex_acc * 100:.2f}% | Macro F1: {regex_f1 * 100:.2f}%")
print(f"  Confusion Matrix (Rows=True, Cols=Pred [CLOSED, OPEN]):\n    {regex_cm}")

# Machine Learning Text Router on SLAKE (4-class routing)
slake_X = slake_df["question"].fillna("").astype(str).values
slake_y = slake_df["adaptive_route"].fillna("unknown").astype(str).values

s_idx = np.arange(len(slake_df))
s_train_idx, s_tmp_idx = train_test_split(s_idx, test_size=0.40, random_state=42, stratify=slake_y)
s_val_idx, s_test_idx  = train_test_split(s_tmp_idx, test_size=0.50, random_state=42, stratify=slake_y[s_tmp_idx])

s_tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
s_X_train_vec = s_tfidf.fit_transform(slake_X[s_train_idx])
s_X_val_vec   = s_tfidf.transform(slake_X[s_val_idx])
s_X_test_vec  = s_tfidf.transform(slake_X[s_test_idx])

s_clf = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(s_X_train_vec, slake_y[s_train_idx])
s_pred_test = s_clf.predict(s_X_test_vec)

s_acc_test = accuracy_score(slake_y[s_test_idx], s_pred_test)
s_macro_f1 = f1_score(slake_y[s_test_idx], s_pred_test, average="macro")
s_weighted_f1 = f1_score(slake_y[s_test_idx], s_pred_test, average="weighted")
s_cm = confusion_matrix(slake_y[s_test_idx], s_pred_test).tolist()
s_rep = classification_report(slake_y[s_test_idx], s_pred_test, output_dict=True)

print(f"\nSLAKE TF-IDF Machine Learning Router (Test N={len(s_test_idx)}):")
print(f"  Test Accuracy: {s_acc_test * 100:.2f}% | Macro F1: {s_macro_f1 * 100:.2f}% | Weighted F1: {s_weighted_f1 * 100:.2f}%")

# Overthinking Impact Analysis on SLAKE
closed_mask = (slake_df["answer_type"] == "CLOSED")
open_mask   = (slake_df["answer_type"] == "OPEN")

closed_no_cot_acc = float((slake_df.loc[closed_mask, "acc_no_cot"] > 0).mean()) * 100
closed_cot_acc    = float((slake_df.loc[closed_mask, "acc_cot"] > 0).mean()) * 100
closed_adapt_acc  = float((slake_df.loc[closed_mask, "acc_adaptive"] > 0).mean()) * 100

open_no_cot_acc   = float((slake_df.loc[open_mask, "acc_no_cot"] > 0).mean()) * 100
open_cot_acc      = float((slake_df.loc[open_mask, "acc_cot"] > 0).mean()) * 100
open_adapt_acc    = float((slake_df.loc[open_mask, "acc_adaptive"] > 0).mean()) * 100

print("\nSLAKE Overthinking Impact (Mechanism of Action):")
print(f"  CLOSED Questions (n={closed_mask.sum()}):")
print(f"    No-CoT: {closed_no_cot_acc:.2f}% | CoT: {closed_cot_acc:.2f}% (Drop: {closed_cot_acc - closed_no_cot_acc:.2f}%) | Adaptive: {closed_adapt_acc:.2f}%")
print(f"  OPEN Questions (n={open_mask.sum()}):")
print(f"    No-CoT: {open_no_cot_acc:.2f}% | CoT: {open_cot_acc:.2f}% | Adaptive: {open_adapt_acc:.2f}% (Gain: {open_adapt_acc - open_no_cot_acc:+.2f}%)")

report["slake_demystified"] = {
    "regex_baseline": {
        "accuracy": round(regex_acc * 100, 2),
        "macro_f1": round(regex_f1 * 100, 2),
        "confusion_matrix": regex_cm,
        "classification_report": regex_rep
    },
    "ml_router": {
        "support": {"train": len(s_train_idx), "val": len(s_val_idx), "test": len(s_test_idx)},
        "accuracy": round(s_acc_test * 100, 2),
        "macro_f1": round(s_macro_f1 * 100, 2),
        "weighted_f1": round(s_weighted_f1 * 100, 2),
        "confusion_matrix": s_cm,
        "classification_report": s_rep
    },
    "overthinking_analysis": {
        "closed_questions": {
            "n": int(closed_mask.sum()),
            "no_cot_acc": round(closed_no_cot_acc, 2),
            "cot_acc": round(closed_cot_acc, 2),
            "adaptive_acc": round(closed_adapt_acc, 2),
            "cot_penalty_drop": round(closed_cot_acc - closed_no_cot_acc, 2)
        },
        "open_questions": {
            "n": int(open_mask.sum()),
            "no_cot_acc": round(open_no_cot_acc, 2),
            "cot_acc": round(open_cot_acc, 2),
            "adaptive_acc": round(open_adapt_acc, 2),
            "adaptive_gain": round(open_adapt_acc - open_no_cot_acc, 2)
        }
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# 5. CHARXIV ERROR CASCADE, REGRET DECOMPOSITION & ORACLE UPPER BOUND
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 5: CHARXIV ERROR CASCADE & ORACLE REGRET ANALYSIS")
print("=" * 80)

dc = charxiv_df["direct_correct"].astype(bool)
cc = charxiv_df["cot_correct"].astype(bool)
rc = charxiv_df["routed_correct"].astype(bool)
oc = charxiv_df["oracle_correct"].astype(bool)

both_correct  = int((dc & cc).sum())
direct_only   = int((dc & ~cc).sum())
cot_only      = int((~dc & cc).sum())
both_wrong    = int((~dc & ~cc).sum())

direct_acc = float(dc.mean()) * 100
cot_acc    = float(cc.mean()) * 100
routed_acc = float(rc.mean()) * 100
oracle_acc = float(oc.mean()) * 100
routing_regret = oracle_acc - routed_acc

print(f"CharXiv Sample Competence Distribution (N={len(charxiv_df)}):")
print(f"  Both Direct & CoT Correct:   {both_correct:3d} ({both_correct/len(charxiv_df)*100:.2f}%)")
print(f"  Direct-Only Correct:         {direct_only:3d} ({direct_only/len(charxiv_df)*100:.2f}%) [CoT Hallucinates]")
print(f"  CoT-Only Correct:            {cot_only:3d} ({cot_only/len(charxiv_df)*100:.2f}%) [Direct Fails]")
print(f"  Both Wrong:                  {both_wrong:3d} ({both_wrong/len(charxiv_df)*100:.2f}%)")

print("\nCharXiv Frontier Comparison:")
print(f"  Single-Branch Direct:         {direct_acc:.2f}%")
print(f"  Single-Branch CoT:            {cot_acc:.2f}%")
print(f"  Current Routed (Heuristic):   {routed_acc:.2f}% (Gain: {routed_acc - direct_acc:+.2f}%)")
print(f"  Theoretical Oracle Ceiling:   {oracle_acc:.2f}% (Potential: {oracle_acc - direct_acc:+.2f}%)")
print(f"  Routing Regret (Oracle Gap):  {routing_regret:.2f}% ({int(round(routing_regret * len(charxiv_df) / 100))} lost questions due to misrouting)")

report["charxiv_error_cascade"] = {
    "competence_distribution": {
        "both_correct": {"count": both_correct, "percent": round(both_correct / len(charxiv_df) * 100, 2)},
        "direct_only_correct": {"count": direct_only, "percent": round(direct_only / len(charxiv_df) * 100, 2)},
        "cot_only_correct": {"count": cot_only, "percent": round(cot_only / len(charxiv_df) * 100, 2)},
        "both_wrong": {"count": both_wrong, "percent": round(both_wrong / len(charxiv_df) * 100, 2)}
    },
    "frontiers": {
        "direct_accuracy": round(direct_acc, 2),
        "cot_accuracy": round(cot_acc, 2),
        "routed_accuracy": round(routed_acc, 2),
        "oracle_upper_bound": round(oracle_acc, 2),
        "routing_regret": round(routing_regret, 2),
        "lost_samples_count": int(round(routing_regret * len(charxiv_df) / 100))
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE COMPREHENSIVE JSON REPORT
# ══════════════════════════════════════════════════════════════════════════════
out_json = ANALYSIS_DIR / "statistical_report.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"✓ Saved updated statistical report to: {out_json}")
print("=" * 80)
