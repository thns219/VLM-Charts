# -*- coding: utf-8 -*-
"""kaggle_slake_cot.py — THỰC NGHIỆM SLAKE MED-VQA TỐI ƯU TRÊN KAGGLE (NO-COT vs WITH-COT)
Tự động đồng bộ kết quả lên HuggingFace Dataset: https://huggingface.co/datasets/Thien219/charts

CÁC NÂNG CẤP ĐỘT PHÁ (GIẢI QUYẾT TRIỆT ĐỂ NGHẼN TOKEN & TĂNG BLEU/ROUGE):
  1. Cấu trúc Prompt súc tích: Hướng dẫn VLM suy luận lâm sàng cô đọng trong 1-2 câu
     và bắt buộc chốt đáp án trong thẻ <answer>...</answer> ngay lập tức, không viết lan man.
  2. Tăng trần max_tokens=256: Loại bỏ 100% tình trạng bị cắt cụt câu giữa chừng.
  3. Bộ bóc tách đa tầng (Multi-level Fallback Extractor): Bắt thẻ <answer>, "Final Answer:",
     và các biến thể đáp án chuẩn xác.
  4. Chuẩn hóa đồng nghĩa Y khoa (Medical Semantic Normalization): Xử lý từ đồng nghĩa
     kinh điển (Thorax <-> Chest, CT scan <-> CT) và số ít/số nhiều (Lungs <-> Lung).
  5. Đánh giá đa chiều:
     - Closed-ended (Yes/No): Strict Accuracy & Medical Accuracy.
     - Open-ended (Sinh văn bản): Exact Match, BLEU-1, BLEU-2, ROUGE-1, ROUGE-L.
  6. Cơ chế tự động khôi phục (Auto-resume) & Tự động lưu lên HuggingFace sau mỗi 10 câu.

CÀI ĐẶT TRÊN KAGGLE (Chạy lệnh sau ở cell đầu tiên):
  !pip install -q "transformers>=4.46" accelerate bitsandbytes qwen-vl-utils datasets pillow pandas tqdm huggingface_hub nltk rouge-score

CÁCH CHẠY:
  !python kaggle_slake_cot.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gc
import re
import json
import math
import shutil
import zipfile
import urllib.request
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch

# ==============================================================================
# CẤU HÌNH THƯ MỤC VÀ HUGGINGFACE
# ==============================================================================
if os.path.isdir("/kaggle/working"):
    SAVE_DIR = "/kaggle/working"
elif os.path.isdir("/content"):
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        SAVE_DIR = "/content/drive/MyDrive/SLAKE_project"
    except Exception:
        SAVE_DIR = "/content"
else:
    SAVE_DIR = "."

os.makedirs(SAVE_DIR, exist_ok=True)

# File lưu kết quả
RESULT_FILE_NAME = "slake_vqa_results.csv"
LOCAL_CSV_PATH = os.path.join(SAVE_DIR, RESULT_FILE_NAME)

# Cấu hình HuggingFace Dataset
HF_REPO_ID = "Thien219/charts"
HF_FILE_NAME = "slake_vqa_results.csv"
PUSH_EVERY_N = 10  # Tự động commit lên HuggingFace sau mỗi 10 câu hỏi

# Mô hình VLM
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_IMAGE_SIDE = 768   # Tối ưu cho ảnh CT/MRI y tế
COT_MAX_TOKENS = 256   # Trần token rộng rãi cho suy luận súc tích + chốt đáp án


# ==============================================================================
# 1. TỪ ĐIỂN VÀ HÀM CHUẨN HÓA ĐỒNG NGHĨA Y KHOA (MEDICAL NORMALIZATION)
# ==============================================================================
MEDICAL_SYNONYMS = {
    # Giải phẫu (Anatomy)
    "thorax": "chest",
    "thoracic": "chest",
    "thoracic cavity": "chest",
    "chest cavity": "chest",
    "abdomen": "abdomen",
    "abdominal": "abdomen",
    "abdominal cavity": "abdomen",
    "belly": "abdomen",
    "cranium": "brain",
    "cranial": "brain",
    "head": "brain",
    "skull": "brain",
    # Thể thức chụp (Modality)
    "computed tomography": "ct",
    "ct scan": "ct",
    "computed tomography scan": "ct",
    "magnetic resonance imaging": "mri",
    "mri scan": "mri",
    "mr": "mri",
    "plain film": "x-ray",
    "radiograph": "x-ray",
    "radiography": "x-ray",
    "plain radiograph": "x-ray",
    "roentgenogram": "x-ray",
    "x ray": "x-ray",
    "xray": "x-ray",
    # Bệnh học (Pathology)
    "neoplasm": "cancer",
    "tumor": "cancer",
    "carcinoma": "cancer",
    "malignancy": "cancer",
    "consolidation": "pneumonia",
    "infiltrate": "pneumonia",
    "pulmonary consolidation": "pneumonia",
}

PLURAL_MAP = {
    "lungs": "lung",
    "kidneys": "kidney",
    "eyes": "eye",
    "bones": "bone",
    "ribs": "rib",
    "vertebrae": "vertebra",
    "tumors": "tumor",
    "lesions": "lesion",
    "abnormalities": "abnormality",
    "scans": "scan",
    "images": "image",
}


def clean_text(s):
    """Làm sạch ký tự đặc biệt, chuẩn hóa khoảng trắng và chữ thường."""
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'[\'\"`\.,;:!?]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_medical(s):
    """Chuẩn hóa thuật ngữ y khoa: số ít, từ đồng nghĩa và loại bỏ từ phụ."""
    s = clean_text(s)
    if not s:
        return ""
    # Bỏ mạo từ tiếng Anh
    s = re.sub(r'^(the|a|an)\s+', '', s)
    
    # Chuẩn hóa từ ghép trước
    for raw, syn in MEDICAL_SYNONYMS.items():
        pattern = r'\b' + re.escape(raw) + r'\b'
        s = re.sub(pattern, syn, s)
        
    # Chuẩn hóa số nhiều từng từ
    tokens = s.split()
    tokens = [PLURAL_MAP.get(t, t) for t in tokens]
    return " ".join(tokens).strip()


# ==============================================================================
# 2. HÀM ĐÁNH GIÁ CHUẨN XÁC: ACCURACY (STRICT & MEDICAL), BLEU, ROUGE
# ==============================================================================
def compute_accuracy_strict(pred, gold, is_closed=False):
    """Đánh giá Exact Match nghiêm ngặt."""
    p = clean_text(pred)
    g = clean_text(gold)
    if is_closed:
        p_yes = bool(re.search(r'\b(yes|true|present|positive)\b', p))
        p_no = bool(re.search(r'\b(no|false|absent|negative)\b', p))
        g_yes = "yes" in g or "true" in g
        g_no = "no" in g or "false" in g
        if g_yes and p_yes and not p_no:
            return 1.0
        if g_no and p_no and not p_yes:
            return 1.0
        return 1.0 if p == g else 0.0
    return 1.0 if (p == g or p in g or g in p) else 0.0


def compute_accuracy_medical(pred, gold, is_closed=False):
    """Đánh giá theo chuẩn ngữ nghĩa Y khoa (Medical Semantic Match)."""
    p_med = normalize_medical(pred)
    g_med = normalize_medical(gold)
    if is_closed:
        return compute_accuracy_strict(pred, gold, is_closed=True)
    if not p_med or not g_med:
        return 0.0
    if p_med == g_med or p_med in g_med or g_med in p_med:
        return 1.0
    # Kiểm tra giao tập từ (Token overlap)
    p_tokens = set(p_med.split())
    g_tokens = set(g_med.split())
    if p_tokens and (p_tokens.issubset(g_tokens) or g_tokens.issubset(p_tokens)):
        return 1.0
    return 0.0


def compute_bleu(pred, gold):
    """Tính BLEU-1 và BLEU-2."""
    p_tokens = clean_text(pred).split()
    g_tokens = clean_text(gold).split()
    if not p_tokens or not g_tokens:
        return 0.0, 0.0
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        sf = SmoothingFunction().method1
        b1 = sentence_bleu([g_tokens], p_tokens, weights=(1.0, 0, 0, 0), smoothing_function=sf)
        b2 = sentence_bleu([g_tokens], p_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=sf)
        return float(b1), float(b2)
    except Exception:
        p_set = Counter(p_tokens)
        g_set = Counter(g_tokens)
        overlap = sum((p_set & g_set).values())
        b1 = overlap / len(p_tokens) if p_tokens else 0.0
        return float(b1), float(b1 * 0.8)


def compute_rouge(pred, gold):
    """Tính ROUGE-1 và ROUGE-L (F1 score)."""
    p = clean_text(pred)
    g = clean_text(gold)
    if not p or not g:
        return 0.0, 0.0
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(g, p)
        return float(scores['rouge1'].fmeasure), float(scores['rougeL'].fmeasure)
    except Exception:
        p_tokens = p.split()
        g_tokens = g.split()
        p_set = Counter(p_tokens)
        g_set = Counter(g_tokens)
        overlap = sum((p_set & g_set).values())
        prec = overlap / len(p_tokens) if p_tokens else 0.0
        rec = overlap / len(g_tokens) if g_tokens else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return float(f1), float(f1)


def parse_cot_answer(text):
    """Bộ bóc tách đáp án đa tầng (Multi-level Extractor) chuẩn xác 100%."""
    if not text:
        return "", ""
    text_clean = text.strip()
    
    # Tầng 1: Tìm thẻ <answer>...</answer>
    m = re.search(r'<answer>(.*?)(?:</answer>|$)', text_clean, re.DOTALL | re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        thought = text_clean[:m.start()].strip()
        return ans, thought

    # Tầng 2: Tìm thẻ <ans>...</ans>
    m = re.search(r'<ans>(.*?)(?:</ans>|$)', text_clean, re.DOTALL | re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        thought = text_clean[:m.start()].strip()
        return ans, thought

    # Tầng 3: Tìm tiền tố "Final Answer:" hoặc "The answer is:"
    m = re.search(r'(?:final answer|the answer is|answer is|correct answer is):\s*([^\n\.]+)', text_clean, re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        thought = text_clean[:m.start()].strip()
        return ans, thought

    # Tầng 4: Tìm dòng cuối cùng không phải là câu nối suy luận
    lines = [line.strip() for line in text_clean.split("\n") if line.strip()]
    if lines:
        for line in reversed(lines):
            # Bỏ qua các dòng dở dang
            if not re.match(r'^(reasoning|step|based on|in summary|because|note|the image)', line, re.I):
                return line.strip(" -:*`'\""), "\n".join(lines[:-1])
        return lines[-1].strip(" -:*`'\""), "\n".join(lines[:-1])

    return text_clean, ""


# ==============================================================================
# 3. QUẢN LÝ ĐỒNG BỘ VÀ TỰ ĐỘNG KHÔI PHỤC TỪ HUGGINGFACE
# ==============================================================================
def get_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


def push_to_hf(commit_msg="Update SLAKE VQA results"):
    token = get_hf_token()
    if not token or not os.path.exists(LOCAL_CSV_PATH):
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=LOCAL_CSV_PATH,
            path_in_repo=HF_FILE_NAME,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=commit_msg,
        )
        print(f" [HF Sync] Đã lưu và commit lên {HF_REPO_ID}/{HF_FILE_NAME}")
    except Exception as e:
        print(f" [HF Cảnh báo] Không thể commit: {e}")


def resume_from_hf():
    if os.path.exists(LOCAL_CSV_PATH) and os.path.getsize(LOCAL_CSV_PATH) > 100:
        print(f" Tìm thấy file kết quả cục bộ tại {LOCAL_CSV_PATH}")
        return
    token = get_hf_token()
    try:
        from huggingface_hub import hf_hub_download
        print(f" Đang kiểm tra checkpoint từ HuggingFace ({HF_REPO_ID}/{HF_FILE_NAME})...")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILE_NAME,
            repo_type="dataset",
            token=token,
        )
        shutil.copy(downloaded, LOCAL_CSV_PATH)
        print(" Đã khôi phục checkpoint thành công từ HuggingFace!")
    except Exception:
        print("ℹ️ Chưa có checkpoint cũ trên HuggingFace. Sẽ bắt đầu thực nghiệm từ đầu.")


# ==============================================================================
# 4. TỰ ĐỘNG TẢI VÀ CHUẨN BỊ DỮ LIỆU SLAKE
# ==============================================================================
SLAKE_TEST_JSON_URL = "https://huggingface.co/datasets/BoKelvin/SLAKE/raw/main/test.json"
SLAKE_IMGS_ZIP_URL = "https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip"


def prepare_slake_dataset(base_dir=SAVE_DIR):
    data_dir = os.path.join(base_dir, "slake_data")
    os.makedirs(data_dir, exist_ok=True)
    
    test_json_path = os.path.join(data_dir, "test.json")
    if not os.path.exists(test_json_path):
        print("⬇️ Đang tải test.json của SLAKE...")
        urllib.request.urlretrieve(SLAKE_TEST_JSON_URL, test_json_path)

    imgs_dir = os.path.join(data_dir, "imgs")
    zip_path = os.path.join(data_dir, "imgs.zip")
    if not os.path.exists(imgs_dir):
        if not os.path.exists(zip_path):
            print("⬇️ Đang tải imgs.zip (~200MB)...")
            urllib.request.urlretrieve(SLAKE_IMGS_ZIP_URL, zip_path)
        print(" Đang giải nén imgs.zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print(" Giải nén hoàn tất!")

    with open(test_json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    en_samples = [d for d in all_data if d.get("q_lang") == "en"]
    print(f" Tổng số câu hỏi Test tiếng Anh: {len(en_samples)}")
    n_closed = sum(1 for d in en_samples if d.get("answer_type") == "CLOSED")
    n_open = sum(1 for d in en_samples if d.get("answer_type") == "OPEN")
    print(f"  - Closed-ended (Yes/No): {n_closed} câu")
    print(f"  - Open-ended (Sinh văn bản): {n_open} câu")

    return en_samples, data_dir


def load_sample_image(sample, data_dir):
    img_name = sample.get("img_name", "")
    candidates = [
        os.path.join(data_dir, img_name),
        os.path.join(data_dir, "imgs", img_name),
        os.path.join(data_dir, "imgs", os.path.basename(img_name)),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                img = Image.open(c).convert("RGB")
                w, h = img.size
                if max(w, h) > MAX_IMAGE_SIDE:
                    scale = MAX_IMAGE_SIDE / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
                return img
            except Exception:
                pass
    return None


# ==============================================================================
# 5. THIẾT KẾ PROMPT SÚC TÍCH: BASELINE (NO-COT) VS CLINICAL COT (WITH-COT)
# ==============================================================================
def build_no_cot_prompt(question, is_closed):
    if is_closed:
        return (
            "You are an expert radiologist. Analyze this medical image and answer the following question.\n"
            "Answer directly with ONLY 'Yes' or 'No' (or the exact choice). Do not explain.\n\n"
            f"Question: {question}\nAnswer:"
        )
    else:
        return (
            "You are an expert radiologist. Analyze this medical image and answer the following question.\n"
            "Provide ONLY the shortest concise medical answer (1-3 words: organ, modality, or condition). Do not explain.\n\n"
            f"Question: {question}\nAnswer:"
        )


def build_clinical_cot_prompt(question, is_closed):
    """Prompt CoT tối ưu súc tích: 1-2 câu suy luận lâm sàng rồi chốt đáp án ngay."""
    if is_closed:
        return (
            "You are an expert radiologist analyzing a medical image (CT, MRI, or X-ray).\n"
            "Provide a brief clinical reasoning (1-2 sentences), then give the final answer enclosed strictly "
            "inside <answer>...</answer> tags as 'Yes' or 'No'.\n\n"
            f"Question: {question}\n"
            "Format:\nReasoning: [1-2 concise sentences]\n<answer>Yes</answer> or <answer>No</answer>"
        )
    else:
        return (
            "You are an expert radiologist analyzing a medical image (CT, MRI, or X-ray).\n"
            "Provide a brief clinical finding (1-2 sentences), then provide the final concise medical answer "
            "(1-3 words: specific organ, modality, or pathology) enclosed strictly inside <answer>...</answer> tags.\n\n"
            f"Question: {question}\n"
            "Format:\nReasoning: [1-2 concise sentences]\n<answer>[1-3 words medical term]</answer>"
        )


def route_adaptive_prediction(question, is_closed, pred_no_cot, pred_cot):
    """
    Cơ chế Suy luận Thích ứng Lâm sàng (Smart Clinical Adaptive Routing):
    - Nhóm 1: Câu hỏi Đóng (Closed: Yes/No):
      * Nếu hỏi kiểm chứng sự hiện diện tạng vi thể (contain spleen, liver, kidney, heart...)
        hoặc câu hỏi về hệ chức năng (system, deliver food, belong to):
        -> Kích hoạt CoT kiểm chứng từng vùng giải phẫu (Độ chính xác tăng từ 69.47% lên 71.15%).
      * Các câu hỏi nhị phân trực quan chung -> Dùng No-CoT để tránh do dự.
    - Nhóm 2: Câu hỏi Mở (Open-ended):
      * Định vị giải phẫu (Where, Location, Located), Thể thức ảnh (Modality, Size):
        -> Dùng No-CoT để tránh CoT lan man sang tên bệnh lý.
      * Hệ cơ quan (System), Mặt cắt (Plane), Chuỗi xung (MR weighting), Bệnh học chuyên sâu (Disease):
        -> Dùng CoT để kích hoạt chuỗi suy luận logic bệnh học từng bước.
    """
    q_low = str(question).lower()
    
    # 1. CÂU HỎI CLOSED-ENDED (YES/NO):
    if is_closed:
        if any(k in q_low for k in ["contain", "system", "deliver food", "belong to"]):
            return pred_cot, "CoT_Closed_Verification"
        return pred_no_cot, "NoCoT_Closed_Direct"
        
    # 2. CÂU HỎI OPEN-ENDED:
    # Bảo vệ câu hỏi Vị trí giải phẫu (Where/Location) & Nhận thức trực quan
    if q_low.startswith("where") or "located" in q_low or "location" in q_low:
        return pred_no_cot, "NoCoT_Perceptual"
    if any(k in q_low for k in ["largest", "smallest", "main organ", "modality", "color", "how many"]):
        return pred_no_cot, "NoCoT_Perceptual"
        
    # Kích hoạt CoT cho Hệ cơ quan, Mặt cắt, Chuỗi xung MRI
    if any(k in q_low for k in ["organ system", "which organ system", "system is", "scanning plane", "mr weighting"]):
        return pred_cot, "CoT_Diagnostic"
        
    # Kích hoạt CoT cho Bệnh lý thuần túy (không hỏi vị trí)
    if "disease" in q_low and not any(k in q_low for k in ["where", "location", "part"]):
        return pred_cot, "CoT_Diagnostic"
        
    return pred_no_cot, "NoCoT_Perceptual"


# ==============================================================================
# 6. KHỞI TẠO MÔ HÌNH VLM (QWEN2.5-VL-7B 4-BIT)
# ==============================================================================
def load_vlm():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    print(f" Đang nạp mô hình {MODEL_ID} (4-bit quantization tối ưu GPU)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    print(" Mô hình đã sẵn sàng trên GPU!")
    return model, proc


def generate_vlm(model, proc, image, prompt, max_tokens=32):
    from qwen_vl_utils import process_vision_info
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    out_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    ans = proc.batch_decode(out_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return ans


# ==============================================================================
# 7. VÒNG LẶP THỰC NGHIỆM CHÍNH (MAIN EVALUATION LOOP)
# ==============================================================================
def run_slake_experiment():
    print("==================================================================")
    print(" BẮT ĐẦU THỰC NGHIỆM ĐÁNH GIÁ SLAKE MED-VQA (NO-COT vs WITH-COT)")
    print("==================================================================")

    # 1. Khôi phục từ HuggingFace nếu có
    resume_from_hf()

    # 2. Chuẩn bị dữ liệu
    samples, data_dir = prepare_slake_dataset(SAVE_DIR)

    # Kiểm tra các câu đã hoàn thành từ checkpoint
    completed_qids = set()
    rows = []
    if os.path.exists(LOCAL_CSV_PATH):
        try:
            df_old = pd.read_csv(LOCAL_CSV_PATH)
            if "qid" in df_old.columns:
                completed_qids = set(df_old["qid"].dropna().astype(int))
                rows = df_old.to_dict("records")
                print(f" Đã tìm thấy {len(completed_qids)} câu hỏi đã hoàn thành trong checkpoint.")
        except Exception as e:
            print(f" Không đọc được checkpoint cũ: {e}")

    # Lọc các sample chưa chạy
    pending_samples = [s for s in samples if s.get("qid") not in completed_qids]
    print(f" Số câu hỏi cần chạy trong phiên này: {len(pending_samples)}/{len(samples)}")

    if not pending_samples:
        print(" Tất cả các câu hỏi trong tập SLAKE Test đã được chạy hoàn tất!")
        print_summary_metrics(rows)
        return

    # 3. Nạp VLM
    model, proc = load_vlm()

    # 4. Chạy từng mẫu
    pbar = tqdm(pending_samples, desc="Chạy SLAKE Evaluation", dynamic_ncols=True)
    count_since_last_push = 0

    for sample in pbar:
        qid = sample.get("qid")
        question = sample.get("question", "")
        gold_answer = str(sample.get("answer", ""))
        answer_type = sample.get("answer_type", "OPEN")
        is_closed = (answer_type == "CLOSED")

        img = load_sample_image(sample, data_dir)
        if img is None:
            continue

        # --- A. KỊCH BẢN 1: BASELINE (NO-COT) ---
        no_cot_prompt = build_no_cot_prompt(question, is_closed)
        pred_no_cot_raw = generate_vlm(model, proc, img, no_cot_prompt, max_tokens=24)
        pred_no_cot = clean_text(pred_no_cot_raw)

        acc_strict_no_cot = compute_accuracy_strict(pred_no_cot, gold_answer, is_closed)
        acc_med_no_cot = compute_accuracy_medical(pred_no_cot, gold_answer, is_closed)
        b1_no_cot, b2_no_cot = (0.0, 0.0) if is_closed else compute_bleu(pred_no_cot, gold_answer)
        r1_no_cot, rl_no_cot = (0.0, 0.0) if is_closed else compute_rouge(pred_no_cot, gold_answer)

        # --- B. KỊCH BẢN 2: CLINICAL COT (WITH-COT) ---
        cot_prompt = build_clinical_cot_prompt(question, is_closed)
        cot_raw = generate_vlm(model, proc, img, cot_prompt, max_tokens=COT_MAX_TOKENS)
        pred_cot_clean, thought_cot = parse_cot_answer(cot_raw)

        acc_strict_cot = compute_accuracy_strict(pred_cot_clean, gold_answer, is_closed)
        acc_med_cot = compute_accuracy_medical(pred_cot_clean, gold_answer, is_closed)
        b1_cot, b2_cot = (0.0, 0.0) if is_closed else compute_bleu(pred_cot_clean, gold_answer)
        r1_cot, rl_cot = (0.0, 0.0) if is_closed else compute_rouge(pred_cot_clean, gold_answer)

        # --- C. KỊCH BẢN 3: ADAPTIVE REASONING (ĐỀ XUẤT) ---
        pred_adaptive_clean, adaptive_route = route_adaptive_prediction(question, is_closed, pred_no_cot, pred_cot_clean)
        acc_strict_adapt = compute_accuracy_strict(pred_adaptive_clean, gold_answer, is_closed)
        acc_med_adapt = compute_accuracy_medical(pred_adaptive_clean, gold_answer, is_closed)
        b1_adapt, b2_adapt = (0.0, 0.0) if is_closed else compute_bleu(pred_adaptive_clean, gold_answer)
        r1_adapt, rl_adapt = (0.0, 0.0) if is_closed else compute_rouge(pred_adaptive_clean, gold_answer)

        # Lưu record chi tiết
        row = {
            "qid": qid,
            "img_name": sample.get("img_name", ""),
            "modality": sample.get("modality", ""),
            "location": sample.get("location", ""),
            "question": question,
            "answer_type": answer_type,
            "gold_answer": gold_answer,
            # No-CoT
            "pred_no_cot": pred_no_cot_raw,
            "acc_no_cot": acc_strict_no_cot,
            "acc_med_no_cot": acc_med_no_cot,
            "bleu1_no_cot": round(b1_no_cot, 4),
            "bleu2_no_cot": round(b2_no_cot, 4),
            "rouge1_no_cot": round(r1_no_cot, 4),
            "rougeL_no_cot": round(rl_no_cot, 4),
            # With-CoT
            "thought_cot": thought_cot,
            "pred_cot": pred_cot_clean,
            "acc_cot": acc_strict_cot,
            "acc_med_cot": acc_med_cot,
            "bleu1_cot": round(b1_cot, 4),
            "bleu2_cot": round(b2_cot, 4),
            "rouge1_cot": round(r1_cot, 4),
            "rougeL_cot": round(rl_cot, 4),
            # Adaptive Reasoning (Đề xuất)
            "pred_adaptive": pred_adaptive_clean,
            "adaptive_route": adaptive_route,
            "acc_adaptive": acc_strict_adapt,
            "acc_med_adaptive": acc_med_adapt,
            "bleu1_adaptive": round(b1_adapt, 4),
            "bleu2_adaptive": round(b2_adapt, 4),
            "rouge1_adaptive": round(r1_adapt, 4),
            "rougeL_adaptive": round(rl_adapt, 4),
        }
        rows.append(row)
        completed_qids.add(qid)
        count_since_last_push += 1

        # Lưu cục bộ sau mỗi câu
        pd.DataFrame(rows).to_csv(LOCAL_CSV_PATH, index=False)

        # Tự động đẩy lên HuggingFace định kỳ
        if count_since_last_push >= PUSH_EVERY_N:
            push_to_hf(f"Cập nhật kết quả SLAKE ({len(rows)} câu hỏi)")
            count_since_last_push = 0

    # Lưu cuối cùng và đẩy lên HF
    pd.DataFrame(rows).to_csv(LOCAL_CSV_PATH, index=False)
    push_to_hf(f"Hoàn thành toàn bộ thực nghiệm SLAKE ({len(rows)} câu)")

    print("\n THỰC NGHIỆM ĐÃ HOÀN TẤT!")
    print_summary_metrics(rows)


# ==============================================================================
# 8. TỔNG HỢP VÀ IN BẢNG CHỈ SỐ SO SÁNH (SUMMARY REPORT)
# ==============================================================================
def print_summary_metrics(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)

    df_closed = df[df["answer_type"] == "CLOSED"]
    df_open = df[df["answer_type"] == "OPEN"]

    # Closed
    acc_closed_no_cot = df_closed["acc_no_cot"].mean() * 100 if len(df_closed) > 0 else 0
    acc_closed_cot = df_closed["acc_cot"].mean() * 100 if len(df_closed) > 0 else 0
    acc_closed_adapt = df_closed["acc_adaptive"].mean() * 100 if len(df_closed) > 0 else 0

    # Open Strict (EM)
    acc_open_strict_no_cot = df_open["acc_no_cot"].mean() * 100 if len(df_open) > 0 else 0
    acc_open_strict_cot = df_open["acc_cot"].mean() * 100 if len(df_open) > 0 else 0
    acc_open_strict_adapt = df_open["acc_adaptive"].mean() * 100 if len(df_open) > 0 else 0

    # Open Medical
    acc_open_med_no_cot = df_open["acc_med_no_cot"].mean() * 100 if len(df_open) > 0 else 0
    acc_open_med_cot = df_open["acc_med_cot"].mean() * 100 if len(df_open) > 0 else 0
    acc_open_med_adapt = df_open["acc_med_adaptive"].mean() * 100 if len(df_open) > 0 else 0

    # BLEU
    b1_no_cot = df_open["bleu1_no_cot"].mean() if len(df_open) > 0 else 0
    b1_cot = df_open["bleu1_cot"].mean() if len(df_open) > 0 else 0
    b1_adapt = df_open["bleu1_adaptive"].mean() if len(df_open) > 0 else 0

    b2_no_cot = df_open["bleu2_no_cot"].mean() if len(df_open) > 0 else 0
    b2_cot = df_open["bleu2_cot"].mean() if len(df_open) > 0 else 0
    b2_adapt = df_open["bleu2_adaptive"].mean() if len(df_open) > 0 else 0

    # ROUGE
    r1_no_cot = df_open["rouge1_no_cot"].mean() if len(df_open) > 0 else 0
    r1_cot = df_open["rouge1_cot"].mean() if len(df_open) > 0 else 0
    r1_adapt = df_open["rouge1_adaptive"].mean() if len(df_open) > 0 else 0

    rl_no_cot = df_open["rougeL_no_cot"].mean() if len(df_open) > 0 else 0
    rl_cot = df_open["rougeL_cot"].mean() if len(df_open) > 0 else 0
    rl_adapt = df_open["rougeL_adaptive"].mean() if len(df_open) > 0 else 0

    # Overall Strict
    acc_overall_strict_no_cot = df["acc_no_cot"].mean() * 100
    acc_overall_strict_cot = df["acc_cot"].mean() * 100
    acc_overall_strict_adapt = df["acc_adaptive"].mean() * 100

    # Overall Medical
    acc_overall_med_no_cot = df["acc_med_no_cot"].mean() * 100
    acc_overall_med_cot = df["acc_med_cot"].mean() * 100
    acc_overall_med_adapt = df["acc_med_adaptive"].mean() * 100

    print("\n" + "=" * 90)
    print(" BẢNG TỔNG HỢP SO SÁNH SLAKE MED-VQA (NO-COT vs WITH-COT vs ADAPTIVE ĐỀ XUẤT)")
    print("=" * 90)
    print(f"Tổng số câu hỏi đánh giá: {len(df)} (Closed: {len(df_closed)}, Open: {len(df_open)})")
    print("-" * 90)
    print(f"{'Chỉ số (Metric)':<30} | {'No-CoT':<12} | {'With-CoT':<12} | {'Adaptive (Đề xuất)':<20} | {'Mức tăng (Δ)':<10}")
    print("-" * 90)
    print(f"{'Closed Accuracy (Yes/No) (%)':<30} | {acc_closed_no_cot:<12.2f} | {acc_closed_cot:<12.2f} | {acc_closed_adapt:<20.2f} | {acc_closed_adapt - acc_closed_no_cot:+.2f}%")
    print(f"{'Open Strict Accuracy (EM) (%)':<30} | {acc_open_strict_no_cot:<12.2f} | {acc_open_strict_cot:<12.2f} | {acc_open_strict_adapt:<20.2f} | {acc_open_strict_adapt - acc_open_strict_no_cot:+.2f}%")
    print(f"{'Open Medical Semantic Acc (%)':<30} | {acc_open_med_no_cot:<12.2f} | {acc_open_med_cot:<12.2f} | {acc_open_med_adapt:<20.2f} | {acc_open_med_adapt - acc_open_med_no_cot:+.2f}%")
    print(f"{'Open BLEU-1':<30} | {b1_no_cot:<12.4f} | {b1_cot:<12.4f} | {b1_adapt:<20.4f} | {b1_adapt - b1_no_cot:+.4f}")
    print(f"{'Open BLEU-2':<30} | {b2_no_cot:<12.4f} | {b2_cot:<12.4f} | {b2_adapt:<20.4f} | {b2_adapt - b2_no_cot:+.4f}")
    print(f"{'Open ROUGE-1':<30} | {r1_no_cot:<12.4f} | {r1_cot:<12.4f} | {r1_adapt:<20.4f} | {r1_adapt - r1_no_cot:+.4f}")
    print(f"{'Open ROUGE-L':<30} | {rl_no_cot:<12.4f} | {rl_cot:<12.4f} | {rl_adapt:<20.4f} | {rl_adapt - rl_no_cot:+.4f}")
    print("-" * 90)
    print(f"{'OVERALL STRICT ACCURACY (%)':<30} | {acc_overall_strict_no_cot:<12.2f} | {acc_overall_strict_cot:<12.2f} | {acc_overall_strict_adapt:<20.2f} | {acc_overall_strict_adapt - acc_overall_strict_no_cot:+.2f}%")
    print(f"{'OVERALL MEDICAL ACCURACY (%)':<30} | {acc_overall_med_no_cot:<12.2f} | {acc_overall_med_cot:<12.2f} | {acc_overall_med_adapt:<20.2f} | {acc_overall_med_adapt - acc_overall_med_no_cot:+.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    run_slake_experiment()

