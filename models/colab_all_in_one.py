# -*- coding: utf-8 -*-
"""colab_all_in_one.py — GỘP TOÀN BỘ các bước đã làm vào 1 file, chạy 1 lệnh:

  GIAI ĐOẠN 1: Hit@k cho 4 cấu hình CLIP (centercrop/letterbox/tiling x B32/L14)
  GIAI ĐOẠN 2: VQA full 2.500 câu với cấu hình tốt nhất (vitl14_letterbox),
               truy vấn = câu hỏi gốc (giống baseline, chỉ đổi ảnh/model)
  GIAI ĐOẠN 3: Hit@k descriptive dùng CAPTION THẬT (tải từ GitHub) + VQA full
               2.500 câu, descriptive dùng caption, reasoning dùng câu hỏi gốc

AN TOÀN — KHÔNG MẤT DỮ LIỆU CŨ
================================
Mọi tên file checkpoint GIỐNG HỆT các lần chạy trước (retrieval_eval_results_v2.csv,
rag_vqa_results_vitl14_letterbox.csv, caption_retrieval_hitk.csv). Vì mỗi giai
đoạn đều kiểm tra "đã làm đủ chưa" trước khi làm, nếu bạn đã chạy Giai đoạn 1-2
từ trước, script sẽ NHẬN RA NGAY và BỎ QUA (0 giây), chỉ thật sự tốn thời gian
ở phần CÒN THIẾU (Giai đoạn 3 — caption, phần duy nhất chưa chạy trên Colab).

CÀI ĐẶT:
    !pip install -q "transformers>=4.46" accelerate bitsandbytes qwen-vl-utils chromadb datasets pillow pandas tqdm

CHẠY:
    !python colab_all_in_one.py
"""

import os
import gc
import json
import urllib.request

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import chromadb
from transformers import CLIPModel, CLIPProcessor

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

DRIVE_ROOT = "/content/drive/MyDrive/VQA_project"
os.makedirs(DRIVE_ROOT, exist_ok=True)
os.makedirs(f"{DRIVE_ROOT}/chroma_eval", exist_ok=True)

# ==== CÔNG TẮC từng giai đoạn ====
RUN_STAGE1_HITK = True          # Hit@k 4 cấu hình — đã xong trước đó, sẽ tự bỏ qua
RUN_STAGE2_VQA_BEST_CONFIG = True  # VQA với crop/model tốt nhất — đã xong, tự bỏ qua
RUN_STAGE3_CAPTION = True       # Hit@k + VQA dùng caption — PHẦN MỚI, sẽ thực sự chạy

BEST_CONFIG_NAME = "vitl14_letterbox"
BEST_CLIP_MODEL = "openai/clip-vit-large-patch14"

N_CHARTS = 500
IMG_SIZE = 224
TILE_GRID = (2, 2)
BATCH_SIZE = 16
DESCRIPTIVE_SAMPLE_QUESTION = "What is the title of the plot?"
IMAGE_METADATA_URL = "https://raw.githubusercontent.com/princeton-nlp/CharXiv/main/data/image_metadata_val.json"

# Tên file — GIỮ NGUYÊN như các lần chạy trước để không mất dữ liệu cũ.
EVAL_RESULTS_FILE = f"{DRIVE_ROOT}/retrieval_eval_results_v2.csv"
STAGE2_CHECKPOINT = f"{DRIVE_ROOT}/rag_vqa_results_{BEST_CONFIG_NAME}.csv"
CAPTION_HIT_RESULTS_FILE = f"{DRIVE_ROOT}/caption_retrieval_hitk.csv"
STAGE3_CHECKPOINT = f"{DRIVE_ROOT}/rag_vqa_results_{BEST_CONFIG_NAME}_caption.csv"

CONFIGS = [
    {"name": "vitb32_centercrop", "model": "openai/clip-vit-base-patch32", "preprocess": "centercrop"},
    {"name": "vitb32_letterbox",  "model": "openai/clip-vit-base-patch32", "preprocess": "letterbox"},
    {"name": "vitb32_tiling",     "model": "openai/clip-vit-base-patch32", "preprocess": "tiling"},
    {"name": "vitl14_letterbox",  "model": "openai/clip-vit-large-patch14", "preprocess": "letterbox"},
]


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Tiền xử lý ảnh
# ---------------------------------------------------------------------------
def letterbox_image(image, size=IMG_SIZE, fill=(255, 255, 255)):
    image = image.convert("RGB")
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def get_tiles(image, grid=TILE_GRID, size=IMG_SIZE):
    image = image.convert("RGB")
    w, h = image.size
    gw, gh = grid
    tile_w, tile_h = w / gw, h / gh
    tiles = []
    for i in range(gh):
        for j in range(gw):
            box = (j * tile_w, i * tile_h, (j + 1) * tile_w, (i + 1) * tile_h)
            tiles.append(image.crop(box).resize((size, size), Image.BICUBIC))
    return tiles


# ---------------------------------------------------------------------------
# Embedding CLIP
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_images_raw(clip_model, clip_processor, device, images):
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
    feats = clip_model.visual_projection(vision_outputs.pooler_output)
    return (feats / feats.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()


@torch.no_grad()
def embed_texts(clip_model, clip_processor, device, texts):
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    text_outputs = clip_model.text_model(
        input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
    )
    feats = clip_model.text_projection(text_outputs.pooler_output)
    return (feats / feats.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()


def embed_one_image_for_config(clip_model, clip_processor, device, image, preprocess):
    if preprocess == "centercrop":
        return embed_images_raw(clip_model, clip_processor, device, [image.convert("RGB")])[0]
    elif preprocess == "letterbox":
        return embed_images_raw(clip_model, clip_processor, device, [letterbox_image(image)])[0]
    elif preprocess == "tiling":
        tile_feats = embed_images_raw(clip_model, clip_processor, device, get_tiles(image))
        combined = tile_feats.mean(axis=0)
        return combined / (combined ** 2).sum() ** 0.5
    raise ValueError(f"preprocess không hợp lệ: {preprocess}")


def build_index_for_config(cfg, val_ds, device, n_charts):
    chroma_dir = f"{DRIVE_ROOT}/chroma_eval/{cfg['name']}"
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name="images", metadata={"hnsw:space": "cosine"})
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
    todo = [i for i in range(n_charts) if str(i) not in existing_ids]
    if not todo:
        print(f"  [{cfg['name']}] Index đã đủ {n_charts} ảnh, bỏ qua.")
        return collection

    print(f"  [{cfg['name']}] Đang tải CLIP ({cfg['model']})...")
    clip_model = CLIPModel.from_pretrained(cfg["model"]).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(cfg["model"])
    print(f"  [{cfg['name']}] Đang mã hoá {len(todo)} ảnh còn thiếu...")
    for start in tqdm(range(0, len(todo), BATCH_SIZE)):
        batch_idx = todo[start:start + BATCH_SIZE]
        embeddings = [
            embed_one_image_for_config(clip_model, clip_processor, device, val_ds[i]["image"], cfg["preprocess"]).tolist()
            for i in batch_idx
        ]
        collection.add(ids=[str(i) for i in batch_idx], embeddings=embeddings,
                        metadatas=[{"chart_index": i} for i in batch_idx])
    del clip_model, clip_processor
    gc.collect()
    torch.cuda.empty_cache()
    return collection


# ---------------------------------------------------------------------------
# GIAI ĐOẠN 1 — Hit@k 4 cấu hình
# ---------------------------------------------------------------------------
def run_stage1(val_ds, device, n_charts):
    done, results = set(), []
    if os.path.exists(EVAL_RESULTS_FILE):
        try:
            prev = pd.read_csv(EVAL_RESULTS_FILE)
            done = set(zip(prev["config_name"], prev["query_type"]))
            results = prev.to_dict("records")
        except (pd.errors.EmptyDataError, KeyError):
            pass

    for cfg in CONFIGS:
        needed = {"descriptive", "reasoning"} - {qt for (cn, qt) in done if cn == cfg["name"]}
        if not needed:
            print(f"[GIAI ĐOẠN 1][{cfg['name']}] Đã đánh giá đủ, bỏ qua.")
            continue
        print(f"\n=== GIAI ĐOẠN 1 — Cấu hình: {cfg['name']} ===")
        collection = build_index_for_config(cfg, val_ds, device, n_charts)

        clip_model = CLIPModel.from_pretrained(cfg["model"]).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(cfg["model"])
        for query_type, get_q in [("descriptive", lambda s: DESCRIPTIVE_SAMPLE_QUESTION),
                                   ("reasoning", lambda s: s["reasoning_q"])]:
            if query_type not in needed:
                continue
            hit1 = hit5 = hit10 = 0
            for chart_idx in tqdm(range(n_charts), desc=f"  Hit@k ({query_type})"):
                query_emb = embed_texts(clip_model, clip_processor, device, [get_q(val_ds[chart_idx])])[0]
                r = collection.query(query_embeddings=[query_emb.tolist()], n_results=10)
                ids = [int(x) for x in r["ids"][0]]
                hit1 += chart_idx in ids[:1]; hit5 += chart_idx in ids[:5]; hit10 += chart_idx in ids[:10]
            results.append({"config_name": cfg["name"], "model": cfg["model"], "preprocess": cfg["preprocess"],
                             "n_charts": n_charts, "query_type": query_type,
                             "hit_at_1": hit1/n_charts, "hit_at_5": hit5/n_charts, "hit_at_10": hit10/n_charts})
        del clip_model, clip_processor
        gc.collect(); torch.cuda.empty_cache()
        pd.DataFrame(results).to_csv(EVAL_RESULTS_FILE, index=False, encoding="utf-8-sig")

    print("\n=== GIAI ĐOẠN 1 HOÀN TẤT ===")
    print(pd.DataFrame(results).pivot_table(index=["config_name"], columns="query_type",
          values=["hit_at_1", "hit_at_5", "hit_at_10"]).to_string())


# ---------------------------------------------------------------------------
# Câu hỏi / VQA dùng chung cho Giai đoạn 2 và 3
# ---------------------------------------------------------------------------
DESCRIPTIVE_QMAP = {
    1: "What is the title of the plot?", 2: "What is the label of the x-axis?",
    3: "What is the label of the y-axis?", 4: "What is the leftmost labeled tick on the x-axis?",
    5: "What is the rightmost labeled tick on the x-axis?",
    6: "What is the spatially lowest labeled tick on the y-axis?",
    7: "What is the spatially highest labeled tick on the y-axis?",
    8: "What is difference between consecutive numerical tick values on the x-axis?",
    9: "What is difference between consecutive numerical tick values on the y-axis?",
    10: "How many lines are there?", 11: "Do any lines intersect?",
    12: "How many discrete labels are there in the legend?",
    13: "What are the names of the labels in the legend? (from top to bottom, then left to right)",
    14: "What is the difference between the maximum and minimum values of the tick labels on the continuous legend (i.e., colorbar)?",
    15: "What is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?",
    16: "What is the general trend of data from left to right?",
    17: "What is the total number of explicitly labeled ticks across all axes?",
    18: "What is the layout of the subplots?", 19: "What is the number of subplots?",
}
REASONING_TYPE_MAP = {1: "Text-in-Chart", 2: "Text-in-General", 3: "Number-in-Chart", 4: "Number-in-General"}
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_IMAGE_SIDE = 1024
MAX_NEW_TOKENS = 64
UNANSWERABLE_LABEL = "Not Applicable"


def get_qwen_model_and_processor():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quant_config)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    return model, processor


def build_prompt(question):
    return ("You are looking at a scientific chart extracted from an arXiv paper. "
            "Answer the following question with the SHORTEST possible answer "
            "(a single word, short phrase, or number) — no explanation.\n"
            f"If the chart does not contain enough information to answer, respond exactly: '{UNANSWERABLE_LABEL}'.\n\n"
            f"Question: {question}\nAnswer:")


def resize_image(image, max_side=MAX_IMAGE_SIDE):
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)))


def ask_qwen(qwen_model, qwen_processor, image, question):
    from qwen_vl_utils import process_vision_info
    prompt = build_prompt(question)
    image = resize_image(image)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = qwen_processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0].strip() if output_text else ""


def simple_match(pred, gold):
    if gold is None:
        return None
    p, g = pred.strip().lower(), str(gold).strip().lower()
    gold_unans = (g == UNANSWERABLE_LABEL.lower())
    pred_unans = (UNANSWERABLE_LABEL.lower() in p)
    if gold_unans or pred_unans:
        return gold_unans == pred_unans
    return g in p or p in g


def iter_questions(sample):
    for i in range(1, 5):
        qid = sample[f"descriptive_q{i}"]
        yield {"slot": f"descriptive_{i}", "question_type": "descriptive",
               "question": DESCRIPTIVE_QMAP.get(qid, f"[Không rõ template id={qid}]"),
               "gold_answer": sample[f"descriptive_a{i}"]}
    yield {"slot": "reasoning", "question_type": "reasoning",
           "sub_type": REASONING_TYPE_MAP.get(sample.get("reasoning_a_type")),
           "question": sample["reasoning_q"], "gold_answer": sample["reasoning_a"]}


def run_vqa(val_ds, device, n_charts, collection, clip_model, clip_processor,
            checkpoint_file, caption_lookup=None, stage_label=""):
    done_keys = set()
    results = []
    if os.path.exists(checkpoint_file):
        prev = pd.read_csv(checkpoint_file)
        done_keys = set(zip(prev["chart_index"], prev["slot"]))
        results = prev.to_dict("records")
    if len(done_keys) >= n_charts * 5:
        print(f"[{stage_label}] Đã xong đủ {n_charts*5} câu, bỏ qua.")
        return pd.DataFrame(results)
    print(f"[{stage_label}] Tìm thấy {len(done_keys)} câu đã làm — chạy tiếp phần còn thiếu.")

    print(f"Đang tải {QWEN_MODEL_ID}...")
    qwen_model, qwen_processor = get_qwen_model_and_processor()

    for chart_idx in tqdm(range(n_charts), desc=stage_label):
        sample = val_ds[chart_idx]
        caption = caption_lookup[chart_idx] if caption_lookup else None
        for qa in iter_questions(sample):
            key = (chart_idx, qa["slot"])
            if key in done_keys:
                continue
            query_text = caption if (qa["question_type"] == "descriptive" and caption) else qa["question"]
            try:
                query_emb = embed_texts(clip_model, clip_processor, device, [query_text])[0]
                r = collection.query(query_embeddings=[query_emb.tolist()], n_results=1)
                retrieved_idx = int(r["ids"][0][0])
                retrieval_correct = (retrieved_idx == chart_idx)
                pred = ask_qwen(qwen_model, qwen_processor, val_ds[retrieved_idx]["image"], qa["question"])
            except Exception as e:
                print(f"[Lỗi ở biểu đồ {chart_idx}, {qa['slot']}]: {e}")
                pred, retrieved_idx, retrieval_correct = "", None, False
            results.append({
                "chart_index": chart_idx, "slot": qa["slot"], "question_type": qa["question_type"],
                "sub_type": qa.get("sub_type"), "question": qa["question"], "gold_answer": qa["gold_answer"],
                "query_source": "caption" if (caption_lookup and query_text == caption) else "question",
                "retrieved_chart_index": retrieved_idx, "retrieval_correct": retrieval_correct,
                "prediction": pred, "correct": simple_match(pred, qa["gold_answer"]),
            })
            pd.DataFrame(results).to_csv(checkpoint_file, index=False, encoding="utf-8-sig")
        if chart_idx % 10 == 0:
            gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print(f"\n=== {stage_label} HOÀN TẤT ===")
    print(f"Tổng thể: {df['correct'].mean():.2%}")
    for qt in ["descriptive", "reasoning"]:
        sub = df[df["question_type"] == qt]
        print(f"{qt}: {sub['correct'].mean():.2%}")

    del qwen_model, qwen_processor
    gc.collect(); torch.cuda.empty_cache()
    return df


# ---------------------------------------------------------------------------
# GIAI ĐOẠN 3 — caption
# ---------------------------------------------------------------------------
def clean_caption(raw):
    text = raw.replace("\\\\", " ").replace("\\n", " ").replace("\n", " ")
    text = text.replace("\\emph", "").replace("\\footnotesize", "").replace("\\raggedright", "")
    return " ".join(text.split())[:300]


def load_caption_map():
    local_path = f"{DRIVE_ROOT}/image_metadata_val.json"
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(IMAGE_METADATA_URL, local_path)
    with open(local_path) as f:
        meta = json.load(f)
    return {entry["figure_path"]: entry["caption"] for entry in meta.values()}


def match_captions_to_val_ds(val_ds, n_charts, caption_by_path):
    captions = []
    matched = 0
    for i in range(n_charts):
        original_path = val_ds[i].get("original_figure_path")
        raw_caption = caption_by_path.get(original_path) if original_path else None
        if raw_caption:
            captions.append(clean_caption(raw_caption))
            matched += 1
        else:
            captions.append(None)
    print(f"Khớp được caption cho {matched}/{n_charts} biểu đồ ({matched/n_charts:.1%}).")
    return captions


def run_stage3_hitk(val_ds, device, n_charts, collection, captions):
    if os.path.exists(CAPTION_HIT_RESULTS_FILE):
        print(f"[GIAI ĐOẠN 3 - Hit@k] Đã có kết quả, bỏ qua: {CAPTION_HIT_RESULTS_FILE}")
        return
    clip_model = CLIPModel.from_pretrained(BEST_CLIP_MODEL).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(BEST_CLIP_MODEL)
    hit1 = hit5 = hit10 = n_eval = 0
    for chart_idx in tqdm(range(n_charts), desc="Hit@k (descriptive, caption)"):
        caption = captions[chart_idx]
        if caption is None:
            continue
        n_eval += 1
        query_emb = embed_texts(clip_model, clip_processor, device, [caption])[0]
        r = collection.query(query_embeddings=[query_emb.tolist()], n_results=10)
        ids = [int(x) for x in r["ids"][0]]
        hit1 += chart_idx in ids[:1]; hit5 += chart_idx in ids[:5]; hit10 += chart_idx in ids[:10]
    row = {"query_source": "caption", "n_evaluated": n_eval,
           "hit_at_1": hit1/n_eval if n_eval else None, "hit_at_5": hit5/n_eval if n_eval else None,
           "hit_at_10": hit10/n_eval if n_eval else None}
    pd.DataFrame([row]).to_csv(CAPTION_HIT_RESULTS_FILE, index=False, encoding="utf-8-sig")
    print("=== GIAI ĐOẠN 3 - Hit@k HOÀN TẤT ===")
    print(row)
    del clip_model, clip_processor
    gc.collect(); torch.cuda.empty_cache()


def main():
    device = get_device()
    print(f"Device: {device}")
    val_ds = load_dataset("princeton-nlp/CharXiv", split="validation")
    n_charts = min(N_CHARTS, len(val_ds))

    if RUN_STAGE1_HITK:
        run_stage1(val_ds, device, n_charts)

    client = chromadb.PersistentClient(path=f"{DRIVE_ROOT}/chroma_eval/{BEST_CONFIG_NAME}")
    collection = client.get_or_create_collection(name="images", metadata={"hnsw:space": "cosine"})
    if collection.count() < n_charts:
        print(f"CẢNH BÁO: collection '{BEST_CONFIG_NAME}' chưa đủ {n_charts} ảnh — "
              f"cần RUN_STAGE1_HITK = True ít nhất 1 lần.")
        return

    if RUN_STAGE2_VQA_BEST_CONFIG:
        clip_model = CLIPModel.from_pretrained(BEST_CLIP_MODEL).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(BEST_CLIP_MODEL)
        run_vqa(val_ds, device, n_charts, collection, clip_model, clip_processor,
                STAGE2_CHECKPOINT, caption_lookup=None, stage_label="GIAI ĐOẠN 2 (crop/model cải thiện)")
        del clip_model, clip_processor
        gc.collect(); torch.cuda.empty_cache()

    if RUN_STAGE3_CAPTION:
        caption_by_path = load_caption_map()
        captions = match_captions_to_val_ds(val_ds, n_charts, caption_by_path)
        run_stage3_hitk(val_ds, device, n_charts, collection, captions)

        clip_model = CLIPModel.from_pretrained(BEST_CLIP_MODEL).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(BEST_CLIP_MODEL)
        run_vqa(val_ds, device, n_charts, collection, clip_model, clip_processor,
                STAGE3_CHECKPOINT, caption_lookup=captions, stage_label="GIAI ĐOẠN 3 (caption)")

    print("\n=== TẤT CẢ GIAI ĐOẠN ĐÃ CHẠY XONG (hoặc đã có sẵn, bỏ qua) ===")
    print(f"- Hit@k 4 cấu hình:      {EVAL_RESULTS_FILE}")
    print(f"- VQA crop/model tốt nhất: {STAGE2_CHECKPOINT}")
    print(f"- Hit@k caption:         {CAPTION_HIT_RESULTS_FILE}")
    print(f"- VQA caption:           {STAGE3_CHECKPOINT}")


if __name__ == "__main__":
    main()
