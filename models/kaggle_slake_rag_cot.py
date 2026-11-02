# -*- coding: utf-8 -*-
"""kaggle_slake_rag_cot.py — GATED MEDICAL KNOWLEDGE RAG + CLINICAL ADAPTIVE COT TRÊN KAGGLE
Tự động đồng bộ kết quả lên HuggingFace Dataset: https://huggingface.co/datasets/Thien219/charts

KIẾN TRÚC ĐỘT PHÁ (GATED MEDICAL RAG - BẢO VỆ THỊ GIÁC & TĂNG CƯỜNG BỆNH HỌC):
  1. Gated Intent Routing (Định tuyến cổng tri thức):
     - Câu hỏi Thuần Thị Giác (Modality, Body Part, Scanning Plane, Organ Size) -> TẮT RAG, dùng No-CoT để
       giữ trọn vẹn điểm đỉnh cao (không bị nhiễu ngữ cảnh).
     - Câu hỏi Tri thức Y khoa (Organ System, Function, Disease, Pathology) -> BẬC RAG truy xuất chính xác
       từ SLAKE Knowledge Graph (2.600 bộ ba chuẩn).
  2. Pilot Mode (Chế độ thử nghiệm an toàn):
     - Mặc định PILOT_MODE = True: Chạy kiểm chứng trước trên 50 câu (mất ~2-3 phút).
     - In chi tiết từng câu: [Câu hỏi | Đáp án chuẩn | No-CoT | RAG Facts | RAG Output | Đánh giá].
     - Khi kiểm chứng thấy điểm tăng thực sự, chỉ cần đổi PILOT_MODE = False để chạy toàn bộ 1.061 câu.
  3. GPU Fast-Track (Tái sử dụng baseline):
     - Tự động nạp slake_vqa_results.csv để không phải tính lại No-CoT và With-CoT từ đầu.

CÀI ĐẶT TRÊN KAGGLE:
  !pip install -q "transformers>=4.46" accelerate bitsandbytes qwen-vl-utils datasets pillow pandas tqdm huggingface_hub nltk rouge-score

CHẠY:
  !python kaggle_slake_rag_cot.py
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
# CẤU HÌNH THỬ NGHIỆM: PILOT TEST vs FULL RUN
# ==============================================================================
PILOT_MODE = True          # True: Chạy thử nhanh 50 câu (chỉ mất ~2-3 phút) | False: Chạy toàn bộ 1.061 câu
PILOT_SAMPLES = 50         # Số câu thử nghiệm

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

RESULT_FILE_NAME = "slake_pilot_rag_results.csv" if PILOT_MODE else "slake_rag_vqa_results.csv"
LOCAL_CSV_PATH = os.path.join(SAVE_DIR, RESULT_FILE_NAME)

HF_REPO_ID = "Thien219/charts"
HF_FILE_NAME = "slake_rag_vqa_results.csv"
PUSH_EVERY_N = 10

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_IMAGE_SIDE = 768
COT_MAX_TOKENS = 256

# URL Dữ liệu SLAKE
SLAKE_TEST_JSON_URL = "https://huggingface.co/datasets/BoKelvin/SLAKE/raw/main/test.json"
SLAKE_IMGS_ZIP_URL = "https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip"
SLAKE_KG_ZIP_URL = "https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/KG.zip"


# ==============================================================================
# 1. TỪ ĐIỂN VÀ HÀM CHUẨN HÓA ĐỒNG NGHĨA Y KHOA (MEDICAL NORMALIZATION)
# ==============================================================================
MEDICAL_SYNONYMS = {
    "thorax": "chest", "thoracic": "chest", "thoracic cavity": "chest", "chest cavity": "chest",
    "abdomen": "abdomen", "abdominal": "abdomen", "abdominal cavity": "abdomen", "belly": "abdomen",
    "cranium": "brain", "cranial": "brain", "head": "brain", "skull": "brain",
    "computed tomography": "ct", "ct scan": "ct", "computed tomography scan": "ct",
    "magnetic resonance imaging": "mri", "mri scan": "mri", "mr": "mri",
    "plain film": "x-ray", "radiograph": "x-ray", "radiography": "x-ray",
    "plain radiograph": "x-ray", "roentgenogram": "x-ray", "x ray": "x-ray", "xray": "x-ray",
    "neoplasm": "cancer", "tumor": "cancer", "carcinoma": "cancer", "malignancy": "cancer",
    "consolidation": "pneumonia", "infiltrate": "pneumonia", "pulmonary consolidation": "pneumonia",
}

PLURAL_MAP = {
    "lungs": "lung", "kidneys": "kidney", "eyes": "eye", "bones": "bone", "ribs": "rib",
    "vertebrae": "vertebra", "tumors": "tumor", "lesions": "lesion", "abnormalities": "abnormality",
    "scans": "scan", "images": "image",
}


def clean_text(s):
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'[\'\"`\.,;:!?]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_medical(s):
    s = clean_text(s)
    if not s:
        return ""
    s = re.sub(r'^(the|a|an)\s+', '', s)
    for raw, syn in MEDICAL_SYNONYMS.items():
        pattern = r'\b' + re.escape(raw) + r'\b'
        s = re.sub(pattern, syn, s)
    tokens = s.split()
    tokens = [PLURAL_MAP.get(t, t) for t in tokens]
    return " ".join(tokens).strip()


def compute_accuracy_strict(pred, gold, is_closed=False):
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
    p_med = normalize_medical(pred)
    g_med = normalize_medical(gold)
    if is_closed:
        return compute_accuracy_strict(pred, gold, is_closed=True)
    if not p_med or not g_med:
        return 0.0
    if p_med == g_med or p_med in g_med or g_med in p_med:
        return 1.0
    p_tokens = set(p_med.split())
    g_tokens = set(g_med.split())
    if p_tokens and (p_tokens.issubset(g_tokens) or g_tokens.issubset(p_tokens)):
        return 1.0
    return 0.0


def compute_bleu(pred, gold):
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
    if not text:
        return "", ""
    text_clean = text.strip()
    m = re.search(r'<answer>(.*?)(?:</answer>|$)', text_clean, re.DOTALL | re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        thought = text_clean[:m.start()].strip()
        return ans, thought

    m = re.search(r'<ans>(.*?)(?:</ans>|$)', text_clean, re.DOTALL | re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        thought = text_clean[:m.start()].strip()
        return ans, thought

    m = re.search(r'(?:final answer|the answer is|answer is|correct answer is):\s*([^\n\.]+)', text_clean, re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        thought = text_clean[:m.start()].strip()
        return ans, thought

    lines = [line.strip() for line in text_clean.split("\n") if line.strip()]
    if lines:
        for line in reversed(lines):
            if not re.match(r'^(reasoning|step|based on|in summary|because|note|the image)', line, re.I):
                return line.strip(" -:*`'\""), "\n".join(lines[:-1])
        return lines[-1].strip(" -:*`'\""), "\n".join(lines[:-1])

    return text_clean, ""


# ==============================================================================
# 2. PHÂN LOẠI Ý ĐỒ CÂU HỎI (KNOWLEDGE-GATED INTENT CLASSIFIER)
# ==============================================================================
def classify_question_intent(question):
    """Phân loại chính xác câu hỏi để kích hoạt RAG đúng lúc, tránh Context Pollution."""
    q_low = str(question).lower()

    # Nhóm 1: Thuần thị giác (Visual Perception) -> CẤM NẠP RAG
    if any(k in q_low for k in ["part of the body", "body part", "which part"]):
        return "visual_body_part"
    if "modality" in q_low:
        return "visual_modality"
    if "plane" in q_low:
        return "visual_plane"
    if "weighting" in q_low or "mr weight" in q_low:
        return "visual_weighting"
    if any(k in q_low for k in ["largest organ", "smallest organ", "how many", "count"]):
        return "visual_spatial"

    # Nhóm 2: Tri thức Y khoa chuyên sâu -> BẮT BUỘC BẬT RAG
    if "system" in q_low:
        return "knowledge_system"
    if any(k in q_low for k in ["function", "role of", "purpose of", "used to", "used for"]):
        return "knowledge_function"
    if any(k in q_low for k in ["disease", "abnormal", "patholog", "cause", "damage", "lesion", "cancer", "tumor", "effusion", "edema"]):
        return "knowledge_disease"

    return "visual_other"


# ==============================================================================
# 3. XÂY DỰNG GATED MEDICAL KNOWLEDGE RETRIEVER (SLAKE KG)
# ==============================================================================
class MedicalKnowledgeRetriever:
    """Retriever tra cứu 2.600 bộ ba tri thức y khoa có kiểm soát intent chuyên biệt."""
    def __init__(self, data_dir):
        self.kg_dir = os.path.join(data_dir, "KG")
        self.zip_path = os.path.join(data_dir, "KG.zip")
        self.triplets = []
        self._load_or_download_kg(data_dir)

    def _load_or_download_kg(self, data_dir):
        if not os.path.exists(self.kg_dir):
            if not os.path.exists(self.zip_path):
                print("⬇️ Đang tải Knowledge Graph SLAKE (KG.zip ~80KB)...")
                urllib.request.urlretrieve(SLAKE_KG_ZIP_URL, self.zip_path)
            print("📦 Đang giải nén KG.zip...")
            with zipfile.ZipFile(self.zip_path, "r") as z:
                z.extractall(data_dir)
        
        sources = [
            ("en_organ.csv", "organ"),
            ("en_disease.csv", "disease"),
            ("en_organ_rel.csv", "rel")
        ]
        for fname, kind in sources:
            p = os.path.join(self.kg_dir, fname)
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
                for l in lines[1:]:
                    parts = l.split("#")
                    if len(parts) == 3:
                        e, a, v = parts[0].strip(), parts[1].strip(), parts[2].strip()
                        self.triplets.append({
                            "entity": e,
                            "attr": a,
                            "val": v,
                            "text": f"{e} [{a}]: {v}",
                            "kind": kind
                        })
        print(f"✅ Đã nạp thành công {len(self.triplets)} sự thật y khoa vào Knowledge Base!")

    def retrieve_gated(self, question, location=None, intent="visual_other", top_k=2):
        """Chỉ truy xuất tri thức khi câu hỏi thuộc nhóm Knowledge. Nếu là Visual -> trả về rỗng."""
        if not intent.startswith("knowledge_"):
            return []

        q_lower = question.lower()
        loc_clean = str(location or "").lower().strip()

        scored = []
        for trip in self.triplets:
            score = 0.0
            txt = trip["text"].lower()
            ent = trip["entity"].lower()
            val = trip["val"].lower()
            attr = trip["attr"].lower()
            kind = trip["kind"]

            if intent == "knowledge_system":
                # Hệ cơ quan: ưu tiên quan hệ belong to / system
                if kind in ["rel", "organ"] and ("belong to" in attr or "system" in txt):
                    score += 6.0
                    if loc_clean and (loc_clean == ent or loc_clean in val):
                        score += 10.0
                    if any(w in ent for w in q_lower.split()):
                        score += 5.0

            elif intent == "knowledge_function":
                # Chức năng: ưu tiên en_organ.csv có attr == function
                if "function" in attr:
                    score += 8.0
                    if loc_clean and (loc_clean == ent or loc_clean in val):
                        score += 10.0

            elif intent == "knowledge_disease":
                # Bệnh học: ưu tiên en_disease.csv
                if kind == "disease":
                    score += 5.0
                    if loc_clean and loc_clean in val:
                        score += 8.0
                    for word in ["abnormal", "tumor", "cancer", "pneumonia", "lesion"]:
                        if word in q_lower and word in txt:
                            score += 4.0

            if score > 0:
                scored.append((score, trip["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]


# ==============================================================================
# 4. QUẢN LÝ ĐỒNG BỘ VÀ CHECKPOINT
# ==============================================================================
def get_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


def push_to_hf(commit_msg="Update SLAKE Medical RAG results"):
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


def load_prior_baselines(save_dir):
    """Tải slake_vqa_results.csv để tái sử dụng kết quả No-CoT và With-CoT đã tính."""
    p = os.path.join(save_dir, "slake_vqa_results.csv")
    if not os.path.exists(p):
        token = get_hf_token()
        try:
            from huggingface_hub import hf_hub_download
            print(f"🔄 Đang tải slake_vqa_results.csv từ Hugging Face ({HF_REPO_ID})...")
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="slake_vqa_results.csv",
                repo_type="dataset",
                token=token,
            )
            shutil.copy(downloaded, p)
            print("✅ Đã nạp thành công slake_vqa_results.csv từ Hugging Face!")
        except Exception:
            pass
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            print(f"✅ Tải được {len(df)} kết quả baseline (No-CoT, CoT, Adaptive) có sẵn!")
            return {int(r["qid"]): r for _, r in df.iterrows()}
        except Exception as e:
            print(f"⚠️ Lỗi đọc baseline cache: {e}")
    return {}


# ==============================================================================
# 5. CHUẨN BỊ DỮ LIỆU SLAKE
# ==============================================================================
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
# 6. THIẾT KẾ PROMPT THEO Ý ĐỒ CÂU HỎI (SCHEMA-CONSTRAINED PROMPTS)
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


def build_gated_rag_prompt(question, is_closed, intent, retrieved_facts):
    """Prompt chuyên biệt có tri thức hỗ trợ nhưng vẫn giữ vững nhận thức thị giác."""
    kb_text = "\n".join([f"- {fact}" for fact in retrieved_facts])

    if intent == "knowledge_system":
        return (
            "You are an expert radiologist analyzing a medical image.\n"
            f"Medical Reference Knowledge:\n{kb_text}\n\n"
            f"Question: {question}\n"
            "Identify the organ visually from the image, then determine the organ system it belongs to based on medical reference.\n"
            "Format:\nReasoning: [1 concise sentence]\n"
            "<answer>[Exact Organ System name, e.g. Respiratory System, Digestive System, Nervous System]</answer>"
        )
    elif intent == "knowledge_function":
        return (
            "You are an expert radiologist analyzing a medical image.\n"
            f"Medical Reference Knowledge:\n{kb_text}\n\n"
            f"Question: {question}\n"
            "Identify the organ, then state its primary physiological function.\n"
            "Format:\nReasoning: [1 concise sentence]\n"
            "<answer>[1-2 words concise function, e.g. breathe]</answer>"
        )
    else:  # knowledge_disease / abnormality
        return (
            "You are an expert radiologist analyzing a medical image.\n"
            f"Medical Reference Knowledge:\n{kb_text}\n\n"
            "Examine visual abnormalities in the image first. Use the medical reference to specify the exact clinical disease name.\n"
            f"Question: {question}\n"
            "Format:\nReasoning: [1 concise sentence]\n"
            "<answer>[1-3 words concise pathology name or 'None']</answer>"
        )


def route_adaptive_prediction(question, is_closed, pred_no_cot, pred_cot):
    """
    Cơ chế Suy luận Thích ứng Lâm sàng (Smart Clinical Adaptive Routing):
    1. Câu hỏi Đóng (Closed: Yes/No):
       - Kiểm chứng sự hiện diện tạng vi thể (contain spleen, liver, kidney...)
         hoặc Hệ cơ quan/chức năng (system, deliver food, belong to)
         -> Dùng CoT để phân tích từng vùng giải phẫu (Tăng điểm từ 69.47% lên 71.15%).
       - Các câu hỏi nhị phân trực quan chung -> Dùng No-CoT.
    2. Câu hỏi Mở (Open-ended):
       - Định vị giải phẫu (Where, Location), Thể thức (Modality, Size) -> Dùng No-CoT.
       - Hệ cơ quan, Mặt cắt, Chuỗi xung MRI, Bệnh học -> Dùng CoT.
    """
    q_low = str(question).lower()
    
    # 1. Câu hỏi Closed-ended (Yes/No)
    if is_closed:
        if any(k in q_low for k in ["contain", "system", "deliver food", "belong to"]):
            return pred_cot, "CoT_Closed_Verification"
        return pred_no_cot, "NoCoT_Closed_Direct"
        
    # 2. Câu hỏi Open-ended
    # Bảo vệ câu hỏi Vị trí giải phẫu & Nhận thức trực quan
    if q_low.startswith("where") or "located" in q_low or "location" in q_low:
        return pred_no_cot, "NoCoT_Perceptual"
    if any(k in q_low for k in ["largest", "smallest", "main organ", "modality", "color", "how many"]):
        return pred_no_cot, "NoCoT_Perceptual"
        
    # Kích hoạt CoT cho Hệ cơ quan, Mặt cắt, Chuỗi xung MRI
    if any(k in q_low for k in ["organ system", "which organ system", "system is", "scanning plane", "mr weighting"]):
        return pred_cot, "CoT_Diagnostic"
        
    # Kích hoạt CoT cho Bệnh lý thuần túy
    if "disease" in q_low and not any(k in q_low for k in ["where", "location", "part"]):
        return pred_cot, "CoT_Diagnostic"
        
    return pred_no_cot, "NoCoT_Perceptual"


# ==============================================================================
# 7. KHỞI TẠO MÔ HÌNH VLM
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
# 8. VÒNG LẶP THỰC NGHIỆM CHÍNH (GATED RAG + COT LOOP)
# ==============================================================================
def run_slake_gated_rag_experiment():
    print("==================================================================")
    mode_text = f"PILOT TEST ({PILOT_SAMPLES} CÂU KIỂM CHỨNG)" if PILOT_MODE else "FULL TEST (1.061 CÂU TOÀN DIỆN)"
    print(f" BẮT ĐẦU THỰC NGHIỆM SLAKE MED-VQA: GATED MEDICAL RAG — {mode_text}")
    print("==================================================================")

    samples, data_dir = prepare_slake_dataset(SAVE_DIR)
    baseline_cache = load_prior_baselines(SAVE_DIR)
    retriever = MedicalKnowledgeRetriever(data_dir)

    if PILOT_MODE:
        # Chọn mẫu đại diện có đủ: 15 Closed, 15 Visual Open, 20 Knowledge Open
        closed_s = [s for s in samples if s.get("answer_type") == "CLOSED"][:15]
        open_s = [s for s in samples if s.get("answer_type") == "OPEN"]
        open_vis = [s for s in open_s if not classify_question_intent(s.get("question", "")).startswith("knowledge_")][:15]
        open_know = [s for s in open_s if classify_question_intent(s.get("question", "")).startswith("knowledge_")][:20]
        test_samples = closed_s + open_vis + open_know
        print(f"🔬 Đã chọn {len(test_samples)} câu thử nghiệm đa dạng thể loại.")
    else:
        test_samples = samples

    model, proc = load_vlm()

    rows = []
    pilot_log_table = []
    pbar = tqdm(test_samples, desc="Chạy Gated Medical RAG", dynamic_ncols=True)

    for sample in pbar:
        qid = sample.get("qid")
        question = sample.get("question", "")
        gold_answer = str(sample.get("answer", ""))
        answer_type = sample.get("answer_type", "OPEN")
        location = sample.get("location", "")
        is_closed = (answer_type == "CLOSED")

        img = load_sample_image(sample, data_dir)
        if img is None:
            continue

        # 1. Phân loại ý đồ câu hỏi & Truy xuất có kiểm soát (Gated Retrieval)
        intent = classify_question_intent(question)
        retrieved_facts = retriever.retrieve_gated(question, location=location, intent=intent, top_k=2)

        # 2. Tái sử dụng baseline đã tính trước (nếu có trong cache)
        base_row = baseline_cache.get(int(qid))
        has_baseline = (
            base_row is not None
            and pd.notna(base_row.get("pred_no_cot"))
            and pd.notna(base_row.get("pred_cot"))
        )

        if has_baseline:
            pred_no_cot_raw = str(base_row.get("pred_no_cot", ""))
            pred_no_cot = clean_text(pred_no_cot_raw)
            acc_strict_no_cot = float(base_row.get("acc_no_cot", 0.0))
            acc_med_no_cot = float(base_row.get("acc_med_no_cot", 0.0))
            b1_no_cot = float(base_row.get("bleu1_no_cot", 0.0))
            b2_no_cot = float(base_row.get("bleu2_no_cot", 0.0))
            r1_no_cot = float(base_row.get("rouge1_no_cot", 0.0))
            rl_no_cot = float(base_row.get("rougeL_no_cot", 0.0))

            pred_cot_clean = str(base_row.get("pred_cot", ""))
            thought_cot = str(base_row.get("thought_cot", ""))
            acc_strict_cot = float(base_row.get("acc_cot", 0.0))
            acc_med_cot = float(base_row.get("acc_med_cot", 0.0))
            b1_cot = float(base_row.get("bleu1_cot", 0.0))
            b2_cot = float(base_row.get("bleu2_cot", 0.0))
            r1_cot = float(base_row.get("rouge1_cot", 0.0))
            rl_cot = float(base_row.get("rougeL_cot", 0.0))

            pred_adaptive_clean = str(base_row.get("pred_adaptive", ""))
            acc_strict_adapt = float(base_row.get("acc_adaptive", 0.0))
            acc_med_adapt = float(base_row.get("acc_med_adaptive", 0.0))
            b1_adapt = float(base_row.get("bleu1_adaptive", 0.0))
            b2_adapt = float(base_row.get("bleu2_adaptive", 0.0))
            r1_adapt = float(base_row.get("rouge1_adaptive", 0.0))
            rl_adapt = float(base_row.get("rougeL_adaptive", 0.0))
        else:
            no_cot_prompt = build_no_cot_prompt(question, is_closed)
            pred_no_cot_raw = generate_vlm(model, proc, img, no_cot_prompt, max_tokens=24)
            pred_no_cot = clean_text(pred_no_cot_raw)
            acc_strict_no_cot = compute_accuracy_strict(pred_no_cot, gold_answer, is_closed)
            acc_med_no_cot = compute_accuracy_medical(pred_no_cot, gold_answer, is_closed)
            b1_no_cot, b2_no_cot = (0.0, 0.0) if is_closed else compute_bleu(pred_no_cot, gold_answer)
            r1_no_cot, rl_no_cot = (0.0, 0.0) if is_closed else compute_rouge(pred_no_cot, gold_answer)

            cot_prompt = build_clinical_cot_prompt(question, is_closed)
            cot_raw = generate_vlm(model, proc, img, cot_prompt, max_tokens=COT_MAX_TOKENS)
            pred_cot_clean, thought_cot = parse_cot_answer(cot_raw)
            acc_strict_cot = compute_accuracy_strict(pred_cot_clean, gold_answer, is_closed)
            acc_med_cot = compute_accuracy_medical(pred_cot_clean, gold_answer, is_closed)
            b1_cot, b2_cot = (0.0, 0.0) if is_closed else compute_bleu(pred_cot_clean, gold_answer)
            r1_cot, rl_cot = (0.0, 0.0) if is_closed else compute_rouge(pred_cot_clean, gold_answer)

            pred_adaptive_clean, _ = route_adaptive_prediction(question, is_closed, pred_no_cot, pred_cot_clean)
            acc_strict_adapt = compute_accuracy_strict(pred_adaptive_clean, gold_answer, is_closed)
            acc_med_adapt = compute_accuracy_medical(pred_adaptive_clean, gold_answer, is_closed)
            b1_adapt, b2_adapt = (0.0, 0.0) if is_closed else compute_bleu(pred_adaptive_clean, gold_answer)
            r1_adapt, rl_adapt = (0.0, 0.0) if is_closed else compute_rouge(pred_adaptive_clean, gold_answer)

        # 3. KỊCH BẢN GATED MEDICAL RAG:
        if not retrieved_facts:
            # Thuần thị giác -> Sử dụng phản xạ trực quan No-CoT (bảo toàn độ chính xác cao nhất)
            pred_rag_clean = pred_no_cot
            thought_rag = "Direct Visual Perception (Knowledge-Gated Bypass)"
        else:
            # Tri thức y khoa -> Bơm tri thức vào chuỗi suy luận
            rag_prompt = build_gated_rag_prompt(question, is_closed, intent, retrieved_facts)
            rag_raw = generate_vlm(model, proc, img, rag_prompt, max_tokens=COT_MAX_TOKENS)
            pred_rag_clean, thought_rag = parse_cot_answer(rag_raw)

        acc_strict_rag = compute_accuracy_strict(pred_rag_clean, gold_answer, is_closed)
        acc_med_rag = compute_accuracy_medical(pred_rag_clean, gold_answer, is_closed)
        b1_rag, b2_rag = (0.0, 0.0) if is_closed else compute_bleu(pred_rag_clean, gold_answer)
        r1_rag, rl_rag = (0.0, 0.0) if is_closed else compute_rouge(pred_rag_clean, gold_answer)

        # Đánh giá thay đổi so với No-CoT
        status = "Không đổi"
        if acc_med_rag > acc_med_no_cot:
            status = "🟢 TĂNG ĐIỂM"
        elif acc_med_rag < acc_med_no_cot:
            status = "🔴 GIẢM ĐIỂM"

        row = {
            "qid": qid,
            "intent": intent,
            "img_name": sample.get("img_name", ""),
            "modality": sample.get("modality", ""),
            "location": location,
            "question": question,
            "answer_type": answer_type,
            "gold_answer": gold_answer,
            "retrieved_facts": " | ".join(retrieved_facts) if retrieved_facts else "None (Visual Gated)",
            # No-CoT
            "pred_no_cot": pred_no_cot,
            "acc_no_cot": acc_strict_no_cot,
            "acc_med_no_cot": acc_med_no_cot,
            "bleu1_no_cot": round(b1_no_cot, 4),
            # With-CoT
            "pred_cot": pred_cot_clean,
            "acc_cot": acc_strict_cot,
            "acc_med_cot": acc_med_cot,
            "bleu1_cot": round(b1_cot, 4),
            # Adaptive
            "pred_adaptive": pred_adaptive_clean,
            "acc_adaptive": acc_strict_adapt,
            "acc_med_adaptive": acc_med_adapt,
            "bleu1_adaptive": round(b1_adapt, 4),
            # Gated RAG
            "pred_rag": pred_rag_clean,
            "acc_rag": acc_strict_rag,
            "acc_med_rag": acc_med_rag,
            "bleu1_rag": round(b1_rag, 4),
            "status": status,
        }
        rows.append(row)

        if PILOT_MODE:
            pilot_log_table.append({
                "QID": qid,
                "Intent": intent[:12],
                "Question": question[:32] + "...",
                "Gold": gold_answer[:15],
                "No-CoT": pred_no_cot[:15],
                "RAG Output": pred_rag_clean[:18],
                "Hiệu quả": status
            })

    # Lưu kết quả
    pd.DataFrame(rows).to_csv(LOCAL_CSV_PATH, index=False)
    if not PILOT_MODE:
        push_to_hf(f"Hoàn thành toàn bộ thực nghiệm Gated Medical RAG ({len(rows)} câu)")

    # In báo cáo chi tiết
    if PILOT_MODE:
        print("\n" + "=" * 115)
        print(" BẢNG CHI TIẾT PILOT TEST (50 CÂU THỬ NGHIỆM ĐẦU TIÊN)")
        print("=" * 115)
        df_pilot_log = pd.DataFrame(pilot_log_table)
        print(df_pilot_log.to_string(index=False))

    print_summary_metrics(rows, is_pilot=PILOT_MODE)


# ==============================================================================
# 9. TỔNG HỢP VÀ IN BẢNG ĐỐI CHỨNG
# ==============================================================================
def print_summary_metrics(rows, is_pilot=False):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df_closed = df[df["answer_type"] == "CLOSED"]
    df_open = df[df["answer_type"] == "OPEN"]

    c_no, c_rag = df_closed["acc_no_cot"].mean() * 100, df_closed["acc_rag"].mean() * 100
    os_no, os_rag = df_open["acc_no_cot"].mean() * 100, df_open["acc_rag"].mean() * 100
    om_no, om_rag = df_open["acc_med_no_cot"].mean() * 100, df_open["acc_med_rag"].mean() * 100
    b1_no, b1_rag = df_open["bleu1_no_cot"].mean(), df_open["bleu1_rag"].mean()
    ov_m_no, ov_m_rag = df["acc_med_no_cot"].mean() * 100, df["acc_med_rag"].mean() * 100

    title = "BẢNG KẾT QUẢ PILOT TEST (50 CÂU)" if is_pilot else "BẢNG TỔNG HỢP TOÀN BỘ 1.061 CÂU SLAKE MED-VQA"
    print("\n" + "=" * 85)
    print(f" {title}")
    print("=" * 85)
    print(f"Tổng số câu đánh giá: {len(df)} (Closed: {len(df_closed)}, Open: {len(df_open)})")
    print("-" * 85)
    print(f"{'Chỉ số (Metric)':<35} | {'No-CoT (Baseline)':<18} | {'Gated Medical RAG':<18} | {'Biến thiên (Δ)':<10}")
    print("-" * 85)
    print(f"{'Closed Accuracy (Yes/No) (%)':<35} | {c_no:<18.2f} | {c_rag:<18.2f} | {c_rag - c_no:+.2f}%")
    print(f"{'Open Strict Accuracy (EM) (%)':<35} | {os_no:<18.2f} | {os_rag:<18.2f} | {os_rag - os_no:+.2f}%")
    print(f"{'Open Medical Semantic Acc (%)':<35} | {om_no:<18.2f} | {om_rag:<18.2f} | {om_rag - om_no:+.2f}%")
    print(f"{'Open BLEU-1':<35} | {b1_no:<18.4f} | {b1_rag:<18.4f} | {b1_rag - b1_no:+.4f}")
    print("-" * 85)
    print(f"{'OVERALL MEDICAL ACCURACY (%)':<35} | {ov_m_no:<18.2f} | {ov_m_rag:<18.2f} | {ov_m_rag - ov_m_no:+.2f}%")
    print("=" * 85)

    if is_pilot:
        print("\n💡 HƯỚNG DẪN TIẾP THEO:")
        print("  - Nếu bảng trên hiển thị điểm TĂNG hoặc KHÔNG BỊ TỤT: Bạn hãy mở đầu file đổi PILOT_MODE = False để chạy toàn bộ 1.061 câu.")
        print("  - File kết quả tạm thời được lưu tại: " + LOCAL_CSV_PATH)


if __name__ == "__main__":
    run_slake_gated_rag_experiment()
