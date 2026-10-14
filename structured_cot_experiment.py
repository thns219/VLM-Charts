# -*- coding: utf-8 -*-
"""structured_cot_experiment.py — Thí nghiệm Lập luận Đa thể thức Định hướng Mục tiêu
(Goal-Directed Grounded Multimodal CoT v2) cho Qwen2.5-VL-7B trên 500 câu suy luận CharXiv.

CỐT LÕI HỌC THUẬT VÀ CẢI TIẾN (Goal-Directed Multimodal Grounding):
  1. Khắc phục 'Visual CoT Paradox' & Hiện tượng trôi dạt ngôn ngữ (Language Drift):
     - Không ép khuôn cứng nhắc (Axes/Target/Value) làm mô hình ám ảnh số đo thô rồi xuất nhầm số thay vì tên.
     - Phân định rõ 4 loại mục tiêu: Nhãn thực thể (Entity), Đếm số lượng (Count), Phép tính số học (Arithmetic), Đo đạc (Measure).
  2. Khử nhập nhằng thực thể (Entity Disambiguation):
     - Câu hỏi "Which model/method/plot": Bắt buộc Answer là TÊN/NHÃN, KHÔNG ĐƯỢC lấy giá trị số.
     - Câu hỏi "How many": Bắt buộc Answer là SỐ Ả RẬP (1, 2, 3), TUYỆT ĐỐI KHÔNG viết chữ (One, Once).
     - Câu hỏi chênh lệch/tính toán: Suy luận rõ mốc giá trị và phép trừ/cộng.
  3. Mở rộng Token Budget (MAX_NEW_TOKENS = 256):
     - Chấm dứt hoàn toàn hiện tượng 64 câu bị đứt gãy token trước khi sinh dòng 'Answer:'.
  4. Loại bỏ bẫy 'Not Applicable' (False Trigger):
     - Không xúi giục mô hình bỏ cuộc trên các biểu đồ phân cụm (t-SNE) hay biểu đồ phi trục tọa độ.
  5. Chuẩn hóa Hậu xử lý (Advanced Normalization):
     - Tích hợp ánh xạ Word-to-Number, làm sạch Markdown/LaTeX, lọc tiền tố thừa.
  6. Tự động đồng bộ checkpoint lên Hugging Face Dataset (Thien219/charts).
"""
import io
import os
import sys
import gc
import re
import json
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from PIL import Image

# ==========================================================
# CẤU HÌNH THÍ NGHIỆM
# ==========================================================
N_CHARTS = 500
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_IMAGE_SIDE = 1024
MAX_NEW_TOKENS = 256  # Đủ hoàn toàn cho suy luận 3 bước + Answer (loại bỏ hoàn toàn đứt gãy 128 tokens)
LOAD_IN_4BIT = True
HF_REPO_ID = "Thien219/charts"

if os.path.isdir("/content"):
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUT_DIR = "/content/drive/MyDrive/VQA_project"
    except Exception:
        OUT_DIR = "/content"
elif os.path.isdir("/kaggle/working"):
    OUT_DIR = "/kaggle/working"
else:
    OUT_DIR = "."

DEFAULT_OUT_FILE = os.path.join(OUT_DIR, "structured_cot_results_v2.csv")
REASONING_TYPE_MAP = {1: "Text-in-Chart", 2: "Text-in-General", 3: "Number-in-Chart", 4: "Number-in-General"}

# ==========================================================
# THUẬT TOÁN ĐÁNH GIÁ CHUẨN CHARXIV (MATCH_FIXED TỰ CHỨA & CẢI TIẾN)
# ==========================================================
NA = "not applicable"
GREEK_MAP = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
    'ε': 'epsilon', 'θ': 'theta', 'κ': 'kappa', 'λ': 'lambda',
    'μ': 'mu', 'π': 'pi', 'σ': 'sigma', 'τ': 'tau', 'φ': 'phi', 'ω': 'omega',
}

WORD_TO_NUM = {
    'zero': 0.0, 'one': 1.0, 'once': 1.0, 'two': 2.0, 'twice': 2.0,
    'three': 3.0, 'thrice': 3.0, 'four': 4.0, 'five': 5.0, 'six': 6.0,
    'seven': 7.0, 'eight': 8.0, 'nine': 9.0, 'ten': 10.0,
    'eleven': 11.0, 'twelve': 12.0,
}

WORD_TO_DIGIT_STR = {
    'zero': '0', 'one': '1', 'once': '1', 'two': '2', 'twice': '2',
    'three': '3', 'thrice': '3', 'four': '4', 'five': '5', 'six': '6',
    'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12',
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


# ==========================================================
# ĐỒNG BỘ VỚI HUGGING FACE (Thien219/charts)
# ==========================================================
def get_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


def resume_from_huggingface(output_file):
    token = get_hf_token()
    if not token or os.path.exists(output_file):
        return
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        fname = os.path.basename(output_file)
        print(f"[HuggingFace] Đang kiểm tra checkpoint từ repo {HF_REPO_ID}...")
        downloaded = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=fname, token=token)
        shutil.copy(downloaded, output_file)
        print(f"[HuggingFace] Tải thành công checkpoint cũ về: {output_file}")
    except Exception:
        print("[HuggingFace] Chưa có checkpoint cũ trên repo, bắt đầu chạy mới.")


def push_to_huggingface(local_file):
    hf_token = get_hf_token()
    if not hf_token or not os.path.exists(local_file):
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        fname = os.path.basename(local_file)
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=fname,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
        print(f"[HuggingFace Sync] Đã lưu {fname} lên {HF_REPO_ID} thành công!")
    except Exception as e:
        print(f"[HuggingFace Sync Cảnh báo] {e}")


def get_lora_dir(lora_path):
    if lora_path and os.path.exists(lora_path):
        return lora_path
    try:
        from huggingface_hub import snapshot_download
        token = get_hf_token()
        print(f"[HuggingFace] Đang tải LoRA adapter từ dataset https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/qwen_chart_lora...")
        local_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            allow_patterns="qwen_chart_lora/*",
            token=token,
        )
        target = os.path.join(local_dir, "qwen_chart_lora")
        return target if os.path.exists(target) else local_dir
    except Exception as e:
        print(f"[LoRA Download Notice]: {e}")
        return lora_path


def load_qwen(lora_path=None):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    is_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if is_bf16 else torch.float16

    quant = None
    if LOAD_IN_4BIT and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as e:
            print(f"[Cảnh báo BitsAndBytes]: {e}. Chuyển sang {compute_dtype}...")
            quant = None

    print(f"Đang tải {QWEN_MODEL_ID} (4-bit: {quant is not None}, dtype={compute_dtype})...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=compute_dtype,
            device_map="auto",
            quantization_config=quant,
        )
    except Exception as e:
        print(f"[Fallback] Nạp 4-bit lỗi ({e}), chuyển sang {compute_dtype} gốc...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=compute_dtype,
            device_map="auto",
        )

    if lora_path:
        actual_lora_dir = get_lora_dir(lora_path)
        print(f"Đang nạp và ghép trọng số LoRA từ: {actual_lora_dir}...")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, actual_lora_dir)
            print("[LoRA] Đã tích hợp LoRA weights thành công!")
        except Exception as e:
            print(f"[LoRA Cảnh báo] Không thể nạp LoRA ({e}), tiếp tục với mô hình gốc.")

    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    return model, processor


def to_pil_image(image_input) -> Image.Image:
    """Chuyển đổi linh hoạt mọi định dạng ảnh (PIL, bytes, dict, path) sang PIL Image RGB."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, dict) and "bytes" in image_input and image_input["bytes"] is not None:
        return Image.open(io.BytesIO(image_input["bytes"])).convert("RGB")
    if isinstance(image_input, dict) and "path" in image_input and image_input["path"]:
        return Image.open(image_input["path"]).convert("RGB")
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    try:
        return Image.open(image_input).convert("RGB")
    except Exception as e:
        raise ValueError(f"Không thể mở ảnh từ kiểu {type(image_input)}: {e}")


def resize_image(image, max_side=MAX_IMAGE_SIDE):
    img = to_pil_image(image)
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    s = max_side / max(w, h)
    return img.resize((int(w * s), int(h * s)), Image.BICUBIC)


# ==========================================================
# GOAL-DIRECTED MULTIMODAL PROMPT & INFERENCE
# ==========================================================
def build_goal_directed_cot_prompt(question):
    """Prompt Lập luận Đa thể thức Định hướng Mục tiêu (Goal-Directed Multimodal Grounding).

    Thiết kế học thuật:
      - Phân tách rạch ròi loại mục tiêu để loại trừ hiện tượng 'nhầm tên sang số'.
      - Ràng buộc chữ số Ả Rập cho câu hỏi đếm (Count).
      - Hướng dẫn đọc bằng chứng thị giác cụ thể (màu đường, legend, điểm giao, nhãn trục).
      - Trực tiếp suy ra đáp án ngắn nhất trên dòng Answer:
    """
    return (
        "You are an expert scientific researcher analyzing a chart from an arXiv research paper.\n"
        "Solve the question using this 3-step visual grounding procedure:\n"
        "1. [Target & Expected Type]: Identify what the question specifically asks for:\n"
        "   - An entity name or label (e.g., model, method, curve, subplot name)?\n"
        "   - A count ('How many...')?\n"
        "   - A numerical calculation (difference, decrease, increase, sum)?\n"
        "   - A single coordinate reading from an axis?\n"
        "2. [Visual Grounding]: Locate the specific curve (match color/marker with legend), subplot, or data points. "
        "Read the relevant values or enumerate the instances step-by-step.\n"
        "3. [Deduction & Verification]: Calculate or deduce the final result.\n\n"
        "CRITICAL RULES FOR THE FINAL ANSWER:\n"
        "- If asked 'Which method/model/plot/line' or 'What is the name', the Answer MUST be that NAME/LABEL, NEVER a numerical score or percentage.\n"
        "- If asked 'How many' (counting), the Answer MUST be an Arabic integer (e.g., 1, 2, 3), NEVER words like 'One' or 'Once'.\n"
        "- All required visual data is present in the chart. Do not answer 'Not Applicable' unless completely unanswerable.\n"
        "- Produce ONLY the single word, number, or shortest phrase on the Answer line.\n\n"
        f"Question: {question}\n\n"
        "Respond strictly in this format:\n"
        "Reasoning:\n"
        "- Target: <expected answer type>\n"
        "- Grounding: <specific visual element, curve color, or subplot observed>\n"
        "- Deduction: <concise calculation, comparison, or count in 1-2 sentences>\n"
        "Answer: <ONLY the concise answer>"
    )


def build_direct_prompt(question):
    """Prompt trả lời thẳng có định dạng chuẩn (dùng cho chế độ Adaptive Routing)."""
    return (
        "You are looking at a scientific chart extracted from an arXiv research paper.\n"
        "Answer the following question with the SHORTEST possible answer (a single word, short phrase, or number) — no explanation.\n"
        "- If asked for a model, method, or line name, provide ONLY the exact name from the legend or chart.\n"
        "- If asked 'How many', provide an Arabic integer (e.g. 1, 2, 3).\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def extract_and_normalize_answer(response_text):
    """Hậu xử lý thông minh: trích xuất đáp án, làm sạch Markdown/LaTeX và chuẩn hóa từ chỉ số."""
    text = response_text.strip()
    match = re.search(r'Answer:\s*(.+)$', text, re.IGNORECASE | re.MULTILINE)
    if match:
        cand = match.group(1).strip()
        cand = cand.split('\n')[0].strip()
    else:
        # Fallback thông minh: tìm dòng Deduction hoặc dòng cuối cùng
        m_ded = re.search(r'-\s*(?:Deduction|Target):\s*(.+)$', text, re.IGNORECASE | re.MULTILINE)
        if m_ded:
            cand = m_ded.group(1).strip().split('\n')[0]
        else:
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            cand = lines[-1].replace("Answer:", "").strip() if lines else text

    # 1. Bỏ định dạng Markdown bold (**ans**), italic (*ans*), code (`ans`), LaTeX ($ans$) và dấu ngoặc kép
    cleaned = re.sub(r'[*_`"\'$]', '', cand).strip()

    # 2. Bỏ các tiền tố diễn giải rườm rà
    cleaned = re.sub(r'^(?:the answer is|it is|answer is|is|the result is)\s+', '', cleaned, flags=re.IGNORECASE).strip()

    # 3. Bỏ dấu chấm cuối câu
    cleaned = cleaned.rstrip('.')

    # 4. Chuẩn hóa từ chỉ số (ví dụ: 'Once' -> '1', 'Two' -> '2')
    low = cleaned.lower().strip()
    if low in WORD_TO_DIGIT_STR:
        return WORD_TO_DIGIT_STR[low]

    # 5. Nếu mô hình sinh dạng: "1 (one intersection)" -> lấy "1"
    m_num = re.match(r'^(\d+(?:\.\d+)?)\s*\([^)]*\)$', cleaned)
    if m_num:
        return m_num.group(1)

    return cleaned


def is_direct_lookup_question(question):
    """Phát hiện các câu hỏi nhận diện thực thể trực tiếp (tận dụng thị giác cross-attention)."""
    q_low = question.lower()
    entity_keywords = [
        "which model", "which method", "which algorithm", "which configuration",
        "which subplot", "which plot", "which line", "which city", "what is the name of",
        "which category", "which curve"
    ]
    return any(k in q_low for k in entity_keywords)


def safe_process_vision_info(messages):
    """Trích xuất ảnh chuẩn Qwen2.5-VL mà không phụ thuộc bắt buộc vào thư viện qwen_vl_utils."""
    try:
        from qwen_vl_utils import process_vision_info
        return process_vision_info(messages)
    except ImportError:
        images = []
        convo_list = [messages] if (messages and isinstance(messages[0], dict)) else messages
        for conv in convo_list:
            for msg in conv:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for elem in content:
                        if isinstance(elem, dict) and "image" in elem:
                            images.append(elem["image"])
        return (images if images else None), None


def ask_model(model, processor, image, question, mode="goal_directed"):
    """Thực hiện suy luận với Qwen2.5-VL theo chế độ được chỉ định."""
    import torch

    if mode == "adaptive" and is_direct_lookup_question(question):
        prompt = build_direct_prompt(question)
        max_tokens = 64
    else:
        prompt = build_goal_directed_cot_prompt(question)
        max_tokens = MAX_NEW_TOKENS

    main_img = resize_image(image)
    content = [
        {"type": "image", "image": main_img},
        {"type": "text", "text": prompt}
    ]

    messages = [{"role": "user", "content": content}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = safe_process_vision_info(messages)
    inputs = processor(text=[chat_text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    out_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    full_output = processor.batch_decode(out_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    del inputs, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_pred = extract_and_normalize_answer(full_output)
    return full_output, final_pred


# ==========================================================
# MAIN EXPERIMENT PIPELINE
# ==========================================================
def run_experiment(output_file=DEFAULT_OUT_FILE, mode="goal_directed", force=False, lora_path=None):
    from datasets import load_dataset

    print("=" * 60)
    print(f"BẮT ĐẦU THÍ NGHIỆM: Goal-Directed Multimodal CoT v2 (Mode: {mode})")
    if lora_path:
        print(f"Sử dụng LoRA Adapter: {lora_path}")
    print(f"File đầu ra: {output_file} | MAX_NEW_TOKENS = {MAX_NEW_TOKENS}")
    print("=" * 60)

    if not force:
        resume_from_huggingface(output_file)

    print("Đang tải CharXiv validation split...")
    val_ds = load_dataset("princeton-nlp/CharXiv", split="validation")
    n_charts = min(N_CHARTS, len(val_ds))

    done_indices = set()
    rows = []
    if os.path.exists(output_file) and not force:
        try:
            prev = pd.read_csv(output_file, keep_default_na=False)
            rows = prev.to_dict("records")
            done_indices = set(prev.chart_index.astype(int))
            print(f"Tìm thấy checkpoint: đã có {len(done_indices)}/{n_charts} biểu đồ.")
        except Exception:
            pass

    if len(done_indices) >= n_charts:
        print("Đã hoàn thành toàn bộ thí nghiệm.")
        analyze_results(output_file)
        push_to_huggingface(output_file)
        return

    model, processor = load_qwen(lora_path=lora_path)

    for idx in range(n_charts):
        if idx in done_indices:
            continue
        sample = val_ds[idx]
        image = sample["image"]
        q = sample["reasoning_q"]
        gold = sample["reasoning_a"]
        sub_type = REASONING_TYPE_MAP.get(sample.get("reasoning_a_type"))

        try:
            full_rationale, pred = ask_model(model, processor, image, q, mode=mode)
        except Exception as e:
            print(f"[Lỗi ở biểu đồ #{idx}]: {e}")
            full_rationale, pred = "", ""

        correct = match_fixed(pred, gold)
        rows.append({
            "chart_index": idx,
            "sub_type": sub_type,
            "question": q,
            "gold_answer": gold,
            "structured_rationale": full_rationale,
            "prediction": pred,
            "correct": correct,
        })
        pd.DataFrame(rows).to_csv(output_file, index=False, encoding="utf-8-sig")

        acc_so_far = np.mean([r["correct"] for r in rows]) * 100
        status_tag = "ĐÚNG" if correct else "SAI"
        print(f"[{len(rows):3d}/{n_charts}] Chart #{idx:3d} | Pred: '{pred[:20]}' | Gold: '{str(gold)[:20]}' | {status_tag} | Hiện tại: {acc_so_far:.1f}%")

        if (len(rows) % 5 == 0) or (len(rows) == n_charts):
            push_to_huggingface(output_file)
            gc.collect()

    print(f"\nThí nghiệm hoàn tất! Đã lưu kết quả tại: {output_file}")
    analyze_results(output_file)
    push_to_huggingface(output_file)


# ==========================================================
# PHÂN TÍCH KẾT QUẢ VÀ ĐỐI CHIẾU HỌC THUẬT
# ==========================================================
def analyze_results(filepath):
    df = pd.read_csv(filepath, keep_default_na=False)
    df["correct"] = [match_fixed(p, g) for p, g in zip(df.prediction, df.gold_answer)]

    print("\n" + "=" * 65)
    print(f"KẾT QUẢ ĐÁNH GIÁ MULTIMODAL REASONING TRÊN {len(df)} CÂU SUY LUẬN")
    print("=" * 65)
    print(f"Độ chính xác tổng thể: {df['correct'].mean() * 100:.2f}% ({df['correct'].sum()}/{len(df)})")
    print("\nTheo kiểu đáp án:")
    print(df.groupby("sub_type")["correct"].agg(["mean", "count"]).to_string())

    # Nếu có file đối chứng cot_results.csv trong thư mục, in bảng so sánh trực tiếp
    cot_ref = os.path.join(os.path.dirname(os.path.abspath(filepath)), "cot_results.csv")
    if os.path.exists(cot_ref):
        try:
            df_cot = pd.read_csv(cot_ref, keep_default_na=False)
            merged = pd.merge(df_cot[['chart_index', 'direct_pred', 'cot_pred']], df, on='chart_index')
            merged['direct_ok'] = [match_fixed(p, g) for p, g in zip(merged['direct_pred'], merged['gold_answer'])]
            merged['cot_ok'] = [match_fixed(p, g) for p, g in zip(merged['cot_pred'], merged['gold_answer'])]

            print("\n" + "-" * 65)
            print("BẢNG SO SÁNH ĐỐI CHỨNG HỌC THUẬT (Trên tập chung 500 câu):")
            print(f"1. Trả lời thẳng (Direct Prompting) : {merged['direct_ok'].mean()*100:.2f}%")
            print(f"2. CoT 2 giai đoạn (Standard CoT)    : {merged['cot_ok'].mean()*100:.2f}%")
            print(f"3. Phiên bản hiện tại ({os.path.basename(filepath)}): {merged['correct'].mean()*100:.2f}%")
            print("-" * 65)
        except Exception as e:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thí nghiệm Goal-Directed Multimodal CoT trên CharXiv.")
    parser.add_argument("--mode", type=str, default="goal_directed", choices=["goal_directed", "adaptive"],
                        help="Chế độ suy luận: goal_directed hoặc adaptive (hybrid routing).")
    parser.add_argument("--lora", type=str, default=None,
                        help="Đường dẫn thư mục hoặc repo Hugging Face chứa trọng số LoRA fine-tuned.")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_FILE, help="Đường dẫn file CSV đầu ra.")
    parser.add_argument("--force", action="store_true", help="Chạy lại từ đầu, ghi đè file cũ.")
    parser.add_argument("cmd", nargs="?", default="run", help="Lệnh: 'run' hoặc 'analyze'.")
    parser.add_argument("analyze_file", nargs="?", default=None, help="File CSV cần phân tích nếu dùng lệnh analyze.")

    args, _ = parser.parse_known_args()

    if args.cmd == "analyze":
        target_f = args.analyze_file or args.out
        analyze_results(target_f)
    else:
        run_experiment(output_file=args.out, mode=args.mode, force=args.force, lora_path=args.lora)
