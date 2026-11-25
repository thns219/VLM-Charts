
# -*- coding: utf-8 -*-
"""models/rerun_baseline_check.py

Xác minh tính độc lập và tất định của cấu hình giải mã baseline Qwen2.5-VL-7B-Instruct.
Chạy Direct trên 100 câu suy luận (reasoning) đầu tiên của CharXiv với do_sample=False.
So sánh kết quả với baseline_qwen_results.csv gốc và chấm điểm bằng match_fixed.

Hỗ trợ chạy mượt mà cả trên Local GPU, Google Colab và Kaggle GPU (T4/P100/A100).
Tự động fallback tải dữ liệu và tự chứa hàm chấm match_fixed nếu chạy standalone.
"""

import os
import sys
import json
import gc
import re
from pathlib import Path
import pandas as pd

# ==== CẤU HÌNH ĐƯỜNG DẪN ====
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "models" else CURRENT_DIR

# Thêm đường dẫn vào sys.path
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

# Xác định thư mục lưu kết quả
if os.path.isdir("/kaggle/working"):
    RESULTS_DIR = Path("/kaggle/working") / "results"
elif (REPO_ROOT / "results").exists():
    RESULTS_DIR = REPO_ROOT / "results"
else:
    RESULTS_DIR = Path("./results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_RESULTS_PATH = RESULTS_DIR / "baseline_qwen_results.csv"
OUT_CONFIG_PATH = RESULTS_DIR / "rerun_generation_config.json"
OUT_CSV_PATH = RESULTS_DIR / "rerun_baseline_qwen_100.csv"
OUT_SUMMARY_PATH = RESULTS_DIR / "rerun_summary.json"

# ==== TỰ CHỨA HÀM CHẤM match_fixed TỪ eval/charxiv_match.py ====
NA = "not applicable"
GREEK_MAP = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
    'ε': 'epsilon', 'θ': 'theta', 'κ': 'kappa', 'λ': 'lambda',
    'μ': 'mu', 'π': 'pi', 'σ': 'sigma', 'τ': 'tau', 'φ': 'phi', 'ω': 'omega',
}

def _clean(s):
    s = str(s).strip().lower().replace('$', '')
    for grk, eng in GREEK_MAP.items():
        s = s.replace(grk, eng)
    s = re.sub(r'\\(?:mathrm|mathit|text|mathbf)\{([^}]*)\}', r'\1', s)
    s = s.replace('\\', '').replace('{', '').replace('}', '')
    s = re.sub(r'(\d)\s*(?:by|x)\s*(\d)', r'\1 by \2', s)
    s = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', s)
    s = re.sub(r'\s*=\s*', '=', s)
    s = re.sub(r'^(?:case|subplot|option|figure|fig)\s+', '', s)
    s = re.sub(r'\s+(?:lines?|subplots?|points?|data points?|datasets?|curves?|clusters?|obstacles?|times?)$', '', s)
    s = re.sub(r'(\d)\s*%', r'\1', s)
    return re.sub(r'\s+', ' ', s).strip().rstrip('.')

WORD_TO_NUM = {
    'zero': 0.0, 'one': 1.0, 'once': 1.0, 'two': 2.0, 'twice': 2.0,
    'three': 3.0, 'thrice': 3.0, 'four': 4.0, 'five': 5.0, 'six': 6.0,
    'seven': 7.0, 'eight': 8.0, 'nine': 9.0, 'ten': 10.0,
    'eleven': 11.0, 'twelve': 12.0,
}

def _num(s):
    s_clean = s.replace(' ', '').lower().rstrip('.')
    if s_clean in WORD_TO_NUM:
        return WORD_TO_NUM[s_clean]
    m = re.fullmatch(r'10\^(-?\d+)', s_clean)
    if m:
        return 10.0 ** int(m.group(1))
    try:
        return float(s_clean.replace(',', ''))
    except ValueError:
        return None

def match_fixed(pred, gold):
    p, g = _clean(pred), _clean(gold)
    if p == "":
        return False
    gu, pu = (g == NA), (NA in p)
    if gu or pu:
        return gu == pu
    if p == g.replace('^', '') or p.replace('^', '') == g.replace('^', ''):
        return True
    a, b = _num(p), _num(g)
    if a is not None and b is not None:
        return abs(a - b) <= 1e-9 * max(1, abs(b))
    if a is not None or b is not None:
        return False
    pp, gg = p.replace('^', ''), g.replace('^', '')
    if min(len(pp), len(gg)) >= 3:
        return gg in pp or pp in gg
    return False

# ==== CẤU HÌNH GỐC TỪ baseline_qwen.py ====
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_IMAGE_SIDE = 1024
MAX_NEW_TOKENS = 64
UNANSWERABLE_LABEL = "Not Applicable"
LOAD_IN_4BIT = True


def check_environment():
    """Kiểm tra GPU và tài nguyên phần cứng trước khi chạy."""
    try:
        import torch
    except ImportError:
        print("[LỖI PHẦN CỨNG/MÔI TRƯỜNG] Không tìm thấy thư viện torch trong môi trường.")
        return False, "Thiếu thư viện torch"

    if not torch.cuda.is_available():
        print("[LỖI PHẦN CỨNG/MÔI TRƯỜNG] Không phát hiện GPU CUDA. Qwen2.5-VL-7B 4-bit NF4 yêu cầu GPU NVIDIA có CUDA.")
        return False, "Không có GPU CUDA"

    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    print(f"[THÔNG TIN GPU] Thiết bị: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.2f} GB")

    try:
        import bitsandbytes
    except ImportError:
        print("[LỖI PHẦN CỨNG/MÔI TRƯỜNG] Thiếu thư viện bitsandbytes để nạp mô hình ở chế độ 4-bit NF4.")
        return False, "Thiếu bitsandbytes"

    return True, "OK"


def get_model_and_processor():
    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    quant_config = None
    if LOAD_IN_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def build_prompt(question: str) -> str:
    return (
        "You are looking at a scientific chart extracted from an arXiv paper. "
        "Answer the following question with the SHORTEST possible answer "
        "(a single word, short phrase, or number) — no explanation.\n"
        f"If the chart does not contain enough information to answer, respond exactly: '{UNANSWERABLE_LABEL}'.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def resize_image(image, max_side=MAX_IMAGE_SIDE):
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)))


def ask_qwen(model, processor, image, question: str) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    prompt = build_prompt(question)
    image = resize_image(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip() if output_text else ""


def ensure_baseline_csv():
    """Đảm bảo file baseline_qwen_results.csv tồn tại, tự tải từ GitHub nếu cần."""
    if BASELINE_RESULTS_PATH.exists():
        return True

    # Kiểm tra trong thư mục gốc repo
    alt_path = REPO_ROOT / "results" / "baseline_qwen_results.csv"
    if alt_path.exists():
        import shutil
        shutil.copy(alt_path, BASELINE_RESULTS_PATH)
        return True

    url = "https://raw.githubusercontent.com/thns219/VLM-Charts/main/results/baseline_qwen_results.csv"
    print(f"[THÔNG BÁO] Không thấy baseline_qwen_results.csv cục bộ, đang tải từ {url}...")
    try:
        import urllib.request
        BASELINE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, BASELINE_RESULTS_PATH)
        print(f"[THÀNH CÔNG] Đã tải về: {BASELINE_RESULTS_PATH}")
        return True
    except Exception as e:
        print(f"[LỖI TẢI CSV] {e}")
        return False


def main():
    print("=" * 80)
    print("VLM-CHARTS: RERUN BASELINE CHECK (100 CÂU SUY LUẬN CHARXIV - DO_SAMPLE=FALSE)")
    print("=" * 80)

    # 1. Kiểm tra tài nguyên thực thi
    env_ok, env_msg = check_environment()
    if not env_ok:
        print(f"\n[DỪNG THỰC THI] Môi trường chưa đáp ứng điều kiện: {env_msg}.")
        print("Vui lòng thực thi script này trên môi trường GPU (ví dụ Kaggle GPU T4/P100 hoặc Google Colab).")
        sys.exit(1)

    import torch
    from datasets import load_dataset

    # 2. Đảm bảo có file baseline_qwen_results.csv
    if not ensure_baseline_csv():
        print(f"[LỖI] Không thể nạp file baseline CSV: {BASELINE_RESULTS_PATH}")
        sys.exit(1)

    df_base = pd.read_csv(BASELINE_RESULTS_PATH)
    reasoning_df = df_base[df_base["question_type"] == "reasoning"].sort_values("chart_index")
    sample_100 = reasoning_df.iloc[:100].copy()
    print(f"Đã chọn 100 câu suy luận đầu tiên (chart_index từ {sample_100['chart_index'].min()} đến {sample_100['chart_index'].max()}).")

    # 3. Nạp mô hình và cấu hình sinh
    print(f"\nĐang tải mô hình {MODEL_ID} (4-bit NF4)...")
    model, processor = get_model_and_processor()

    # Xuất cấu hình generation_config
    gen_config_dict = model.generation_config.to_dict()
    print("\n[GENERATION CONFIG TỪ CHECKPOINT]:")
    print(json.dumps(gen_config_dict, indent=2))

    with open(OUT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(gen_config_dict, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu generation config vào: {OUT_CONFIG_PATH}")

    # 4. Tải dataset CharXiv
    print("\nĐang nạp tập dữ liệu princeton-nlp/CharXiv (split: validation)...")
    val_ds = load_dataset("princeton-nlp/CharXiv", split="validation")

    predictions_run1 = []
    predictions_run2 = []

    print("\n--- BẮT ĐẦU CHẠY LẦN 1 (Run 1 - do_sample=False) ---")
    for idx, row in sample_100.iterrows():
        c_idx = int(row["chart_index"])
        q = row["question"]
        image = val_ds[c_idx]["image"]
        pred = ask_qwen(model, processor, image, q)
        predictions_run1.append(pred)
        if len(predictions_run1) % 10 == 0:
            print(f"  Hoàn thành {len(predictions_run1)}/100 câu...")
            gc.collect()
            torch.cuda.empty_cache()

    print("\n--- BẮT ĐẦU CHẠY LẦN 2 (Run 2 - Kiểm tra tính tất định) ---")
    for idx, row in sample_100.iterrows():
        c_idx = int(row["chart_index"])
        q = row["question"]
        image = val_ds[c_idx]["image"]
        pred = ask_qwen(model, processor, image, q)
        predictions_run2.append(pred)
        if len(predictions_run2) % 25 == 0:
            print(f"  Hoàn thành kiểm tra {len(predictions_run2)}/100 câu...")

    # 5. So sánh Run 1 và Run 2 (Tính tất định)
    diff_runs = sum(1 for p1, p2 in zip(predictions_run1, predictions_run2) if _clean(p1) != _clean(p2))
    diff_runs_raw = sum(1 for p1, p2 in zip(predictions_run1, predictions_run2) if p1 != p2)
    print(f"\n[KIỂM ĐỊNH TÍNH TẤT ĐỊNH]:")
    print(f"  Số câu khác biệt giữa Run 1 và Run 2 (chuỗi thô)       : {diff_runs_raw}/100")
    print(f"  Số câu khác biệt giữa Run 1 và Run 2 (sau chuẩn hóa)  : {diff_runs}/100")

    # 6. Đánh giá và lưu bảng kết quả rerun_baseline_qwen_100.csv
    records = []
    correct_old_list = []
    correct_new_list = []
    match_str_count = 0
    flip_correct_to_wrong = 0
    flip_wrong_to_correct = 0

    for i, (_, row) in enumerate(sample_100.iterrows()):
        c_idx = int(row["chart_index"])
        q = row["question"]
        gold = row["gold_answer"]
        pred_old = str(row["prediction"]) if pd.notna(row["prediction"]) else ""
        pred_new = predictions_run1[i]

        c_old = match_fixed(pred_old, gold)
        c_new = match_fixed(pred_new, gold)

        correct_old_list.append(c_old)
        correct_new_list.append(c_new)

        # So sánh chuỗi sau chuẩn hóa _clean
        if _clean(pred_new) == _clean(pred_old):
            match_str_count += 1

        # Đổi chiều
        if c_old and not c_new:
            flip_correct_to_wrong += 1
        elif not c_old and c_new:
            flip_wrong_to_correct += 1

        records.append({
            "chart_index": c_idx,
            "question": q,
            "gold_answer": gold,
            "prediction_old": pred_old,
            "prediction_new": pred_new,
            "correct_old": c_old,
            "correct_new": c_new,
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu kết quả chi tiết 100 câu vào: {OUT_CSV_PATH}")

    # 7. Tính tổng hợp và lưu kết quả summary
    acc_old = sum(correct_old_list) / len(correct_old_list)
    acc_new = sum(correct_new_list) / len(correct_new_list)

    summary = {
        "n_samples": 100,
        "prediction_exact_match_normalized": match_str_count,
        "prediction_exact_match_normalized_percent": round(match_str_count / 100 * 100, 2),
        "direction_changes": {
            "correct_to_incorrect": flip_correct_to_wrong,
            "incorrect_to_correct": flip_wrong_to_correct,
            "total_flips": flip_correct_to_wrong + flip_wrong_to_correct,
        },
        "accuracy_old_match_fixed": round(acc_old * 100, 2),
        "accuracy_new_match_fixed": round(acc_new * 100, 2),
        "accuracy_delta": round((acc_new - acc_old) * 100, 2),
        "determinism_check": {
            "differing_predictions_raw": diff_runs_raw,
            "differing_predictions_normalized": diff_runs,
            "is_deterministic": (diff_runs == 0),
        }
    }

    with open(OUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu tóm tắt thống kê vào: {OUT_SUMMARY_PATH}")

    # Nếu đang chạy trên Kaggle, sao chép kết quả ra /kaggle/working để tải trực tiếp
    if os.path.isdir("/kaggle/working"):
        import shutil
        for p in [OUT_CONFIG_PATH, OUT_CSV_PATH, OUT_SUMMARY_PATH]:
            dest = Path("/kaggle/working") / p.name
            try:
                shutil.copy(p, dest)
                print(f"[KAGGLE OUTPUT] Đã copy {p.name} -> {dest}")
            except Exception:
                pass

    print("\n" + "=" * 80)
    print("[BÁO CÁO KẾT QUẢ RERUN BASELINE (5 DÒNG)]:")
    print(f"1. Cấu hình generation_config: do_sample=False, max_new_tokens=64, lưu tại results/rerun_generation_config.json")
    print(f"2. Độ trùng khớp chuỗi (sau chuẩn hóa): {match_str_count}/100 câu ({summary['prediction_exact_match_normalized_percent']}%) so với baseline cũ.")
    print(f"3. Số câu đổi chiều đánh giá: Đúng->Sai: {flip_correct_to_wrong} câu | Sai->Đúng: {flip_wrong_to_correct} câu.")
    print(f"4. Độ chính xác trên 100 câu: Cũ = {summary['accuracy_old_match_fixed']}% -> Mới = {summary['accuracy_new_match_fixed']}% (Độ lệch: {summary['accuracy_delta']}%).")
    print(f"5. Kiểm định tính tất định (Run 1 vs Run 2): {diff_runs} câu khác biệt -> {'Tất định tuyệt đối (100% khớp)' if diff_runs == 0 else 'Không tất định hoàn toàn'}.")
    print("=" * 80)


if __name__ == "__main__":
    main()
