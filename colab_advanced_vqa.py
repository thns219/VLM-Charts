# -*- coding: utf-8 -*-
"""colab_advanced_vqa.py — SCRIPT HYBRID RAG + MULTIMODAL REASONING COT TỐI ƯU TỐC ĐỘ (~2 TIẾNG)
VÀ TỰ ĐỘNG ĐỒNG BỘ LÊN HUGGINGFACE (https://huggingface.co/datasets/Thien219/charts)

CÁC ĐIỂM CỐT LÕI (BẢO TOÀN 100% Ý CHÍNH CỦA ĐỀ TÀI):
  1. HYBRID RETRIEVAL: CLIP ViT-L/14 + BM25 trên Caption/Metadata (giải quyết nút thắt tìm sai ảnh).
  2. LẬP LUẬN ĐA THỂ THỨC (Multimodal Reasoning): Câu suy luận DÙNG STRUCTURED VISUAL COT 4 BƯỚC.
  3. NHẬN BIẾT TRỰC QUAN (Descriptive): 4 câu mô tả trả lời thẳng súc tích (tiết kiệm thời gian, tránh nói nhảm).
  4. TỰ ĐỘNG ĐẨY LÊN HUGGINGFACE: Đẩy file rag_vqa_results_hybrid.csv lên Thien219/charts sau mỗi 5 biểu đồ.
  5. ĐỘC LẬP 100%: Nhúng sẵn match_fixed và BM25, không cần tải thêm bất kỳ file nào khác.

CÀI ĐẶT TRÊN KAGGLE (chạy 1 dòng ở cell đầu):
    !pip install -q "transformers>=4.46" accelerate bitsandbytes qwen-vl-utils datasets pillow pandas tqdm chromadb huggingface_hub

CHẠY:
    !python colab_advanced_vqa.py
"""
import os
import sys
import gc
import re
import math
import json
import urllib.request
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image

# Xác định thư mục lưu dữ liệu an toàn trên Kaggle / Colab
if os.path.isdir("/kaggle/working"):
    SAVE_DIR = "/kaggle/working"
elif os.path.isdir("/content"):
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        SAVE_DIR = "/content/drive/MyDrive/VQA_project"
    except Exception:
        SAVE_DIR = "/content"
else:
    SAVE_DIR = "."

os.makedirs(SAVE_DIR, exist_ok=True)
CHECKPOINT_HYBRID = os.path.join(SAVE_DIR, "rag_vqa_results_hybrid.csv")
CACHE_CLIP_FILE = os.path.join(SAVE_DIR, "clip_vitl14_cache.npy")
IMAGE_METADATA_URL = "https://raw.githubusercontent.com/princeton-nlp/CharXiv/main/data/image_metadata_val.json"

# Cấu hình HuggingFace của bạn
HF_REPO_ID = "Thien219/charts"
HF_FILE_NAME = "rag_vqa_results_hybrid.csv"
PUSH_EVERY_N_CHARTS = 5  # Tự động commit lên HuggingFace sau mỗi 5 biểu đồ

N_CHARTS = 500
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_IMAGE_SIDE = 1024
UNANSWERABLE_LABEL = "Not Applicable"


# ==============================================================================
# 1. ĐỒNG BỘ VỚI HUGGINGFACE TỰ ĐỘNG
# ==============================================================================
def get_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


def push_to_hf(commit_msg):
    token = get_hf_token()
    if not token or not os.path.exists(CHECKPOINT_HYBRID):
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=CHECKPOINT_HYBRID,
            path_in_repo=HF_FILE_NAME,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=commit_msg,
        )
    except Exception as e:
        print(f"[Cảnh báo HF] Chưa đẩy được commit: {e}")


def resume_from_hf():
    """Nếu bắt đầu phiên mới, tự động kéo bản checkpoint mới nhất từ HF về."""
    if os.path.exists(CHECKPOINT_HYBRID):
        return
    token = get_hf_token()
    try:
        from huggingface_hub import hf_hub_download
        print(f"Đang kiểm tra checkpoint từ HuggingFace ({HF_REPO_ID})...")
        downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILE_NAME, repo_type="dataset", token=token)
        import shutil
        shutil.copy(downloaded, CHECKPOINT_HYBRID)
        print("Đã khôi phục checkpoint thành công từ HuggingFace!")
    except Exception:
        print("Chưa có checkpoint cũ trên HuggingFace, bắt đầu từ đầu.")


# ==============================================================================
# 2. HÀM CHẤM ĐIỂM CHUẨN HOÁ
# ==============================================================================
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

def _num(s):
    s = s.replace(' ', '')
    m = re.fullmatch(r'10\^(-?\d+)', s)
    if m:
        return 10.0 ** int(m.group(1))
    try:
        return float(s.replace(',', ''))
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


# ==============================================================================
# 3. BỘ TÌM KIẾM LAI BM25 + CLIP
# ==============================================================================
def tokenize(text):
    return re.findall(r'[a-z0-9_\-]+', str(text).lower())

class BM25Retriever:
    def __init__(self, corpus_texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_texts)
        self.tokenized_corpus = [tokenize(doc) for doc in corpus_texts]
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / (self.corpus_size or 1)
        self.doc_freqs = Counter()
        self.inverted_index = []
        for doc in self.tokenized_corpus:
            counts = Counter(doc)
            self.inverted_index.append(counts)
            for token in counts.keys():
                self.doc_freqs[token] += 1
        self.idf = {
            t: math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for t, freq in self.doc_freqs.items()
        }

    def get_scores(self, query_text):
        query_tokens = tokenize(query_text)
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if not query_tokens or self.corpus_size == 0:
            return scores
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for doc_idx, doc_counts in enumerate(self.inverted_index):
                freq = doc_counts.get(token, 0)
                if freq == 0:
                    continue
                doc_len = self.doc_lens[doc_idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += idf_val * (freq * (self.k1 + 1.0)) / denom
        return scores

def min_max_normalize(scores):
    arr = np.array(scores, dtype=np.float32)
    s_min, s_max = arr.min(), arr.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(arr)
    return (arr - s_min) / (s_max - s_min)

class HybridChartRetriever:
    def __init__(self, clip_embeddings, metadata_texts, alpha=0.4):
        self.clip_embeddings = clip_embeddings
        self.bm25 = BM25Retriever(metadata_texts)
        self.alpha = alpha
        self.n_charts = len(metadata_texts)

    def retrieve(self, query_text, query_clip_embedding=None, top_k=3):
        bm25_raw = self.bm25.get_scores(query_text)
        bm25_norm = min_max_normalize(bm25_raw)
        if query_clip_embedding is not None and self.clip_embeddings is not None:
            q_emb = query_clip_embedding / (np.linalg.norm(query_clip_embedding) + 1e-9)
            dense_raw = np.dot(self.clip_embeddings, q_emb)
            dense_norm = min_max_normalize(dense_raw)
            combined_scores = self.alpha * dense_norm + (1.0 - self.alpha) * bm25_norm
        else:
            combined_scores = bm25_norm
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        return [(int(idx), float(combined_scores[idx])) for idx in top_indices]


# ==============================================================================
# 4. TIỀN XỬ LÝ & TẢI MÔ HÌNH VLM
# ==============================================================================
DESCRIPTIVE_QMAP = {
    1: "What is the title of the plot?", 2: "What is the label of the x-axis?",
    3: "What is the label of the y-axis?", 4: "What is the leftmost labeled tick on the x-axis?",
    5: "What is the rightmost labeled tick on the x-axis?", 6: "What is the spatially lowest labeled tick on the y-axis?",
    7: "What is the spatially highest labeled tick on the y-axis?", 8: "What is difference between consecutive numerical tick values on the x-axis?",
    9: "What is difference between consecutive numerical tick values on the y-axis?", 10: "How many lines are there?",
    11: "Do any lines intersect?", 12: "How many discrete labels are there in the legend?",
    13: "What are the names of the labels in the legend? (from top to bottom, then left to right)",
    14: "What is the difference between the maximum and minimum values of the tick labels on the continuous legend (i.e., colorbar)?",
    15: "What is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?",
    16: "What is the general trend of data from left to right?", 17: "What is the total number of explicitly labeled ticks across all axes?",
    18: "What is the layout of the subplots?", 19: "What is the number of subplots?",
}
REASONING_TYPE_MAP = {1: "Text-in-Chart", 2: "Text-in-General", 3: "Number-in-Chart", 4: "Number-in-General"}

def letterbox_image(image, size=224):
    image = image.convert("RGB")
    w, h = image.size
    scale = size / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = image.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas

def load_captions(n_charts, val_ds):
    local_meta = os.path.join(SAVE_DIR, "image_metadata_val.json")
    if not os.path.exists(local_meta):
        print("Đang tải image_metadata_val.json...")
        urllib.request.urlretrieve(IMAGE_METADATA_URL, local_meta)
    with open(local_meta, encoding="utf-8") as f:
        meta = json.load(f)
    caption_map = {entry["figure_path"]: entry.get("caption", "") for entry in meta.values()}
    return [caption_map.get(val_ds[i].get("original_figure_path"), f"chart_{i}") for i in range(n_charts)]

def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang tải {CLIP_MODEL_NAME} ({device})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return model, proc, device

def load_qwen():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    quant = None
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    except Exception:
        quant = None

    print(f"Đang tải {QWEN_MODEL_ID} (4-bit={quant is not None})...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quant
        )
    except Exception as e:
        print(f"[Tự động xử lý] Bỏ qua 4-bit do lỗi môi trường ({e}), nạp mô hình ở dạng bfloat16 gốc (nét hơn)...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
    proc = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    return model, proc

def build_clip_embeddings(val_ds, clip_model, clip_proc, device, n_charts):
    import torch
    if os.path.exists(CACHE_CLIP_FILE):
        arr = np.load(CACHE_CLIP_FILE)
        if len(arr) >= n_charts:
            print(f"Tái sử dụng cache CLIP embeddings: {CACHE_CLIP_FILE}")
            return arr[:n_charts]
    print(f"Tính toán CLIP embeddings cho {n_charts} ảnh...")
    embeds = []
    for i in range(0, n_charts, 16):
        batch_imgs = [letterbox_image(val_ds[j]["image"]) for j in range(i, min(i + 16, n_charts))]
        inputs = clip_proc(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
            feats = clip_model.visual_projection(vision_out.pooler_output)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            embeds.append(feats.cpu().numpy())
    all_embeds = np.vstack(embeds)
    np.save(CACHE_CLIP_FILE, all_embeds)
    return all_embeds

def embed_text_clip(clip_model, clip_proc, device, text):
    import torch
    inputs = clip_proc(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = clip_model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        feats = clip_model.text_projection(out.pooler_output)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


# ==============================================================================
# 5. CHIẾN LƯỢC PROMPT TỐI ƯU TỐC ĐỘ & GIỮ NGUYÊN LẬP LUẬN ĐA THỂ THỨC
# ==============================================================================
def ask_qwen_descriptive(model, proc, image, question):
    """4 câu mô tả: Trả lời trực tiếp súc tích (10 tokens) -> CỰC NHANH (~1.5s/câu)."""
    import torch
    from qwen_vl_utils import process_vision_info
    image = image.convert("RGB")
    w, h = image.size
    scale = MAX_IMAGE_SIDE / max(w, h)
    img_resized = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC) if max(w, h) > MAX_IMAGE_SIDE else image

    prompt = (
        "You are looking at a scientific chart from an arXiv paper. "
        "Answer with the SHORTEST possible answer (a single number, word, or short phrase) — no explanation.\n"
        f"If the chart does not contain enough information, respond exactly: '{UNANSWERABLE_LABEL}'.\n\n"
        f"Question: {question}\nAnswer:"
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": img_resized}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    out_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    ans = proc.batch_decode(out_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return ans.strip().rstrip('.')


def ask_qwen_reasoning_cot(model, proc, image, question):
    """1 câu suy luận: GIỮ NGUYÊN LẬP LUẬN ĐA THỂ THỨC (STRUCTURED VISUAL COT 4 BƯỚC)."""
    import torch
    from qwen_vl_utils import process_vision_info
    image = image.convert("RGB")
    w, h = image.size
    scale = MAX_IMAGE_SIDE / max(w, h)
    img_resized = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC) if max(w, h) > MAX_IMAGE_SIDE else image

    prompt = (
        "You are an expert scientific data analyst performing multimodal reasoning on an arXiv chart.\n"
        "Follow this 4-step visual grounding procedure:\n"
        "1. Identify the relevant axes, lines, bars, or legends.\n"
        "2. Locate the specific value or label asked in the question.\n"
        "3. Read the exact numerical/text value directly from pixels.\n"
        "4. Output the final concise decision.\n\n"
        f"If the chart does not contain enough information, respond exactly: '{UNANSWERABLE_LABEL}'.\n\n"
        f"Question: {question}\n"
        "Format:\n"
        "Reasoning: <brief step-by-step grounding note>\n"
        "Answer: <ONLY the single word, number, or short phrase>"
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": img_resized}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    out_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    decoded = proc.batch_decode(out_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    match = re.search(r'Answer:\s*(.+)$', decoded, re.IGNORECASE | re.MULTILINE)
    ans = match.group(1).strip().rstrip('.') if match else decoded.split('\n')[-1].strip()
    return ans


# ==============================================================================
# 6. CHẠY TOÀN BỘ PIPELINE
# ==============================================================================
def main():
    from datasets import load_dataset
    from tqdm import tqdm

    resume_from_hf()

    print("Đang tải CharXiv validation split...")
    val_ds = load_dataset("princeton-nlp/CharXiv", split="validation")
    n_charts = min(N_CHARTS, len(val_ds))

    captions = load_captions(n_charts, val_ds)
    clip_model, clip_proc, device = load_clip()
    clip_embeds = build_clip_embeddings(val_ds, clip_model, clip_proc, device, n_charts)
    retriever = HybridChartRetriever(clip_embeddings=clip_embeds, metadata_texts=captions, alpha=0.4)

    done_keys = set()
    results = []
    if os.path.exists(CHECKPOINT_HYBRID):
        prev = pd.read_csv(CHECKPOINT_HYBRID, keep_default_na=False)
        done_keys = set(zip(prev.chart_index.astype(int), prev.slot))
        results = prev.to_dict("records")
        print(f"Tiếp tục từ checkpoint: {len(done_keys)}/{n_charts * 5} câu.")

    if len(done_keys) < n_charts * 5:
        qwen_model, qwen_proc = load_qwen()

        for idx in tqdm(range(n_charts), desc="Hybrid RAG + Multimodal CoT"):
            sample = val_ds[idx]
            cap = captions[idx]
            q_list = []
            for i in range(1, 5):
                qid = sample[f"descriptive_q{i}"]
                q_list.append({
                    "slot": f"descriptive_{i}", "question_type": "descriptive",
                    "question": DESCRIPTIVE_QMAP.get(qid, ""), "gold_answer": sample[f"descriptive_a{i}"],
                    "query_text": DESCRIPTIVE_QMAP.get(qid, ""),
                })
            q_list.append({
                "slot": "reasoning", "question_type": "reasoning",
                "sub_type": REASONING_TYPE_MAP.get(sample.get("reasoning_a_type")),
                "question": sample["reasoning_q"], "gold_answer": sample["reasoning_a"],
                "query_text": sample["reasoning_q"],
            })

            for qa in q_list:
                key = (idx, qa["slot"])
                if key in done_keys:
                    continue

                q_emb = embed_text_clip(clip_model, clip_proc, device, qa["query_text"])
                top_hits = retriever.retrieve(query_text=qa["query_text"], query_clip_embedding=q_emb, top_k=3)
                retrieved_idx = top_hits[0][0]
                retrieved_img = val_ds[retrieved_idx]["image"]

                try:
                    if qa["question_type"] == "descriptive":
                        pred = ask_qwen_descriptive(qwen_model, qwen_proc, retrieved_img, qa["question"])
                    else:
                        pred = ask_qwen_reasoning_cot(qwen_model, qwen_proc, retrieved_img, qa["question"])
                except Exception as e:
                    if idx < 3:
                        print(f"\n[LỖI QWEN tại chart {idx}]: {e}")
                    pred = ""

                correct = match_fixed(pred, qa["gold_answer"])
                results.append({
                    "chart_index": idx, "slot": qa["slot"], "question_type": qa["question_type"],
                    "sub_type": qa.get("sub_type"), "question": qa["question"], "gold_answer": qa["gold_answer"],
                    "retrieved_chart_index": retrieved_idx, "retrieval_correct": (retrieved_idx == idx),
                    "prediction": pred, "correct": correct,
                })
                pd.DataFrame(results).to_csv(CHECKPOINT_HYBRID, index=False, encoding="utf-8-sig")

            # Đẩy lên HuggingFace định kỳ mỗi 5 biểu đồ
            if (idx + 1) % PUSH_EVERY_N_CHARTS == 0 or idx == n_charts - 1:
                push_to_hf(f"Checkpoint: {idx + 1}/{n_charts} charts ({len(results)}/{n_charts*5} questions)")

            if idx % 10 == 0:
                gc.collect()

    push_to_hf("Hoàn tất: rag_vqa_results_hybrid.csv")
    df = pd.DataFrame(results).drop_duplicates(subset=["chart_index", "slot"], keep="last")
    print("\n" + "=" * 70)
    print(f"KẾT QUẢ ĐỘT PHÁ CỦA HYBRID RAG + MULTIMODAL COT ({len(df)} CÂU)")
    print("=" * 70)
    print(f"Retrieval đúng (Hit@1): {df['retrieval_correct'].mean() * 100:.2f}%")
    print(f"VQA Độ chính xác tổng : {df['correct'].mean() * 100:.2f}%")
    for qt in ["descriptive", "reasoning"]:
        sub = df[df["question_type"] == qt]
        print(f"  - {qt:12s}: VQA={sub['correct'].mean()*100:.2f}%, Retrieval={sub['retrieval_correct'].mean()*100:.2f}%")
    print(f"\nFile kết quả đã được lưu tại: {CHECKPOINT_HYBRID}")
    print(f"Và đã đẩy lên HuggingFace: https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    main()
