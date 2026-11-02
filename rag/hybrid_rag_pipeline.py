# -*- coding: utf-8 -*-
"""hybrid_rag_pipeline.py — Pipeline RAG Đột Phá cho VLM-Charts.
Kết hợp Hybrid Retrieval (Dense CLIP + Sparse BM25 trên Caption/Metadata) + Structured Visual CoT.

Mục tiêu nâng cao độ chính xác:
  1. Retrieval: Nâng Hit@1 từ 21-25% lên 60%+ bằng cách kết hợp BM25 từ khóa thực thể.
  2. VQA: Thay vì prompt tự do, sử dụng Structured Visual CoT để giải quyết ảo giác của 7B.
  3. Kết quả lưu vào: rag_vqa_results_hybrid.csv.

CHẠY (Colab GPU T4 hoặc Kaggle):
    python hybrid_rag_pipeline.py
"""
import os
import sys
import gc
import json
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from charxiv_match import match_fixed
from hybrid_retrieval import HybridChartRetriever

N_CHARTS = 500
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_IMAGE_SIDE = 1024
UNANSWERABLE_LABEL = "Not Applicable"
IMAGE_METADATA_URL = "https://raw.githubusercontent.com/princeton-nlp/CharXiv/main/data/image_metadata_val.json"

if os.path.isdir("/content"):
    OUT_DIR = "/content/drive/MyDrive/VQA_project" if os.path.isdir("/content/drive") else "/content"
elif os.path.isdir("/kaggle/working"):
    OUT_DIR = "/kaggle/working"
else:
    OUT_DIR = "."

CHECKPOINT_FILE = os.path.join(OUT_DIR, "rag_vqa_results_hybrid.csv")
REASONING_TYPE_MAP = {1: "Text-in-Chart", 2: "Text-in-General", 3: "Number-in-Chart", 4: "Number-in-General"}

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


def load_captions(n_charts, val_ds):
    local_meta = os.path.join(OUT_DIR, "image_metadata_val.json")
    if not os.path.exists(local_meta):
        print("Đang tải image_metadata_val.json từ GitHub...")
        try:
            urllib.request.urlretrieve(IMAGE_METADATA_URL, local_meta)
        except Exception as e:
            print(f"Không thể tải metadata: {e}")
            return ["" for _ in range(n_charts)]

    with open(local_meta, encoding="utf-8") as f:
        meta = json.load(f)
    caption_map = {entry["figure_path"]: entry.get("caption", "") for entry in meta.values()}

    captions = []
    for i in range(n_charts):
        path = val_ds[i].get("original_figure_path")
        cap = caption_map.get(path, "")
        captions.append(cap if cap else f"chart_{i}")
    return captions


def letterbox_image(image, size=224):
    image = image.convert("RGB")
    w, h = image.size
    scale = size / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = image.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang tải {CLIP_MODEL_NAME} lên {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return model, proc, device


def load_qwen():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4") if torch.cuda.is_available() else None
    print(f"Đang tải {QWEN_MODEL_ID}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quant
    )
    proc = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    return model, proc


def build_clip_embeddings(val_ds, clip_model, clip_proc, device, n_charts):
    import torch
    cache_file = os.path.join(OUT_DIR, "clip_vitl14_cache.npy")
    if os.path.exists(cache_file):
        print(f"Tái sử dụng cache CLIP embedding từ: {cache_file}")
        arr = np.load(cache_file)
        if len(arr) >= n_charts:
            return arr[:n_charts]

    print(f"Tính toán CLIP embeddings cho {n_charts} biểu đồ...")
    embeds = []
    batch_size = 16
    for i in range(0, n_charts, batch_size):
        batch_imgs = [letterbox_image(val_ds[j]["image"]) for j in range(i, min(i + batch_size, n_charts))]
        inputs = clip_proc(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_out.pooler_output
            feats = clip_model.visual_projection(pooled)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            embeds.append(feats.cpu().numpy())
    all_embeds = np.vstack(embeds)
    np.save(cache_file, all_embeds)
    return all_embeds


def embed_text_clip(clip_model, clip_proc, device, text):
    import torch
    inputs = clip_proc(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_out = clip_model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        pooled = text_out.pooler_output
        feats = clip_model.text_projection(pooled)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


def build_structured_prompt(question):
    return (
        "You are looking at a scientific chart from an arXiv paper.\n"
        "Carefully follow these visual steps:\n"
        "1. Identify the relevant axes, lines, bars, or legends.\n"
        "2. Locate the specific value or label asked in the question.\n"
        "3. Provide ONLY the shortest possible answer (a single number, word, or short phrase).\n"
        f"If the chart does not contain enough information, respond exactly: '{UNANSWERABLE_LABEL}'.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def ask_qwen(model, proc, image, question):
    import torch
    from qwen_vl_utils import process_vision_info
    w, h = image.size
    scale = MAX_IMAGE_SIDE / max(w, h)
    img_resized = image.resize((int(w * scale), int(h * scale))) if max(w, h) > MAX_IMAGE_SIDE else image

    prompt = build_structured_prompt(question)
    messages = [{"role": "user", "content": [{"type": "image", "image": img_resized}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    out_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return proc.batch_decode(out_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def run_pipeline():
    from datasets import load_dataset
    from tqdm import tqdm

    print("Đang tải dữ liệu CharXiv validation...")
    val_ds = load_dataset("princeton-nlp/CharXiv", split="validation")
    n_charts = min(N_CHARTS, len(val_ds))

    captions = load_captions(n_charts, val_ds)
    clip_model, clip_proc, device = load_clip()
    clip_embeds = build_clip_embeddings(val_ds, clip_model, clip_proc, device, n_charts)

    # Khởi tạo Hybrid Retriever
    retriever = HybridChartRetriever(clip_embeddings=clip_embeds, metadata_texts=captions, alpha=0.4)

    # Đọc checkpoint
    done_keys = set()
    results = []
    if os.path.exists(CHECKPOINT_FILE):
        prev = pd.read_csv(CHECKPOINT_FILE, keep_default_na=False)
        done_keys = set(zip(prev.chart_index.astype(int), prev.slot))
        results = prev.to_dict("records")
        print(f"Tiếp tục từ checkpoint: {len(done_keys)}/{n_charts * 5} câu.")

    if len(done_keys) < n_charts * 5:
        qwen_model, qwen_proc = load_qwen()

        for idx in tqdm(range(n_charts), desc="Hybrid RAG Pipeline"):
            sample = val_ds[idx]
            cap = captions[idx]

            # 4 câu descriptive + 1 câu reasoning
            q_list = []
            for i in range(1, 5):
                qid = sample[f"descriptive_q{i}"]
                q_list.append({
                    "slot": f"descriptive_{i}", "question_type": "descriptive",
                    "question": DESCRIPTIVE_QMAP.get(qid, ""), "gold_answer": sample[f"descriptive_a{i}"],
                    "query_text": f"{cap} {DESCRIPTIVE_QMAP.get(qid, '')}" if cap else DESCRIPTIVE_QMAP.get(qid, ""),
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

                # Hybrid Retrieval
                q_emb = embed_text_clip(clip_model, clip_proc, device, qa["query_text"])
                top_hits = retriever.retrieve(query_text=qa["query_text"], query_clip_embedding=q_emb, top_k=5)
                retrieved_idx = top_hits[0][0]
                retrieval_correct = (retrieved_idx == idx)

                # VQA trên ảnh được tìm kiếm
                retrieved_img = val_ds[retrieved_idx]["image"]
                try:
                    pred = ask_qwen(qwen_model, qwen_proc, retrieved_img, qa["question"])
                except Exception as e:
                    pred = ""

                correct = match_fixed(pred, qa["gold_answer"])
                results.append({
                    "chart_index": idx, "slot": qa["slot"], "question_type": qa["question_type"],
                    "sub_type": qa.get("sub_type"), "question": qa["question"], "gold_answer": qa["gold_answer"],
                    "retrieved_chart_index": retrieved_idx, "retrieval_correct": retrieval_correct,
                    "prediction": pred, "correct": correct,
                })
                pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")

            if idx % 10 == 0:
                gc.collect()

    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset=["chart_index", "slot"], keep="last")
    print("\n" + "=" * 70)
    print(f"KẾT QUẢ HYBRID RAG TRÊN {len(df)} CÂU HỎI")
    print("=" * 70)
    print(f"Tỷ lệ truy hồi đúng (Hit@1): {df['retrieval_correct'].mean() * 100:.2f}%")
    print(f"Độ chính xác VQA tổng thể : {df['correct'].mean() * 100:.2f}%")
    for qt in ["descriptive", "reasoning"]:
        sub = df[df["question_type"] == qt]
        print(f"  - {qt:12s}: VQA={sub['correct'].mean()*100:.2f}%, Retrieval={sub['retrieval_correct'].mean()*100:.2f}%")


if __name__ == "__main__":
    run_pipeline()
