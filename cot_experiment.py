# -*- coding: utf-8 -*-
"""cot_experiment.py — so sánh "trả lời thẳng" với "suy luận hai giai đoạn" (theo ý tưởng
Multimodal-CoT) trên 500 câu SUY LUẬN của CharXiv, dùng Qwen2.5-VL-7B với ảnh ĐÚNG (oracle).

  Giai đoạn 1: ảnh + câu hỏi  -> viết lý do (rationale)
  Giai đoạn 2: ảnh + câu hỏi + lý do -> đáp án ngắn
  Đối chứng : ảnh + câu hỏi  -> đáp án ngắn (trả lời thẳng, cùng mô hình, cùng prompt style)

CHẠY (cần GPU, Colab hoặc Kaggle):
    !pip install -q "transformers>=4.46" accelerate bitsandbytes qwen-vl-utils datasets pillow pandas tqdm huggingface_hub
    !python cot_experiment.py
Kết quả lưu từng biểu đồ vào cot_results.csv (ngắt giữa chừng thì chạy lại, tự tiếp tục).

PHÂN TÍCH (chạy trên máy, không cần GPU, cần charxiv_match.py cùng thư mục):
    python cot_experiment.py analyze cot_results.csv

LƯU Ý: đây là áp dụng ý tưởng hai giai đoạn bằng prompt lên Qwen có sẵn, KHÔNG phải tái hiện bài
báo (bài fine-tune T5 nhỏ + ViT). Giải mã tham lam (do_sample=False) để kết quả lặp lại được.
"""
import os
import sys
import gc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

N_CHARTS = 500
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_IMAGE_SIDE = 1024
MAX_TOKENS_RATIONALE = 200
MAX_TOKENS_ANSWER = 64
UNANSWERABLE_LABEL = "Not Applicable"
REASONING_TYPE_MAP = {1: "Text-in-Chart", 2: "Text-in-General", 3: "Number-in-Chart", 4: "Number-in-General"}

if os.path.isdir("/content"):                      # Colab
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUT_DIR = "/content/drive/MyDrive/VQA_project"
    except Exception:
        OUT_DIR = "/content"
elif os.path.isdir("/kaggle/working"):             # Kaggle
    OUT_DIR = "/kaggle/working"
else:
    OUT_DIR = "."
OUT_FILE = os.path.join(OUT_DIR, "cot_results.csv")

# ---------------------------------------------------------------------------
# Đồng bộ với HuggingFace (dùng khi chạy Kaggle: ổ đĩa mất khi phiên kết thúc)
# Cần secret HF_TOKEN (quyền Write). Không có token -> bỏ qua, chỉ lưu cục bộ.
# ---------------------------------------------------------------------------
HF_REPO_ID = "Thien219/charts"
HF_FILE = "cot_results.csv"
PUSH_EVERY = 5          # đẩy lên HF sau mỗi 5 biểu đồ (không đẩy sau mỗi câu để tránh spam commit)


def get_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


HF_TOKEN = get_hf_token()


def resume_from_hf():
    """Nếu trên HF đã có kết quả từ phiên trước, tải về (giữ bản có nhiều dòng hơn)."""
    if not HF_TOKEN:
        print("Không có HF_TOKEN -> chỉ lưu cục bộ, không đồng bộ HuggingFace.")
        return
    try:
        import shutil
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=HF_FILE, token=HF_TOKEN)
        n_hf = len(pd.read_csv(got, keep_default_na=False))
        n_local = len(pd.read_csv(OUT_FILE, keep_default_na=False)) if os.path.exists(OUT_FILE) else 0
        if n_hf > n_local:
            shutil.copy(got, OUT_FILE)
            print(f"Đã tải kết quả cũ từ HuggingFace ({n_hf} biểu đồ) để tiếp tục.")
    except Exception as e:
        print(f"Chưa có kết quả cũ trên HuggingFace (hoặc không tải được): {type(e).__name__}. Chạy từ đầu.")


def push_to_hf(msg):
    """Đẩy cot_results.csv lên HF. Lỗi mạng chỉ cảnh báo, không làm dừng thí nghiệm."""
    if not HF_TOKEN or not os.path.exists(OUT_FILE):
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(path_or_fileobj=OUT_FILE, path_in_repo=HF_FILE,
                                          repo_id=HF_REPO_ID, repo_type="dataset", commit_message=msg)
    except Exception as e:
        print(f"[Cảnh báo] Chưa đẩy được lên HuggingFace ({type(e).__name__}); dữ liệu vẫn nằm trong {OUT_FILE}.")


# ---------------------------------------------------------------------------
# Qwen
# ---------------------------------------------------------------------------
def load_qwen():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quant)
    return model, AutoProcessor.from_pretrained(QWEN_MODEL_ID)


def resize_image(image, max_side=MAX_IMAGE_SIDE):
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    s = max_side / max(w, h)
    return image.resize((int(w * s), int(h * s)))


def generate(model, processor, image, prompt, max_new_tokens):
    import torch
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [{"type": "image", "image": resize_image(image)},
                                              {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def prompt_direct(q):
    return ("You are looking at a scientific chart extracted from an arXiv paper. Answer the following question "
            "with the SHORTEST possible answer (a single word, short phrase, or number) — no explanation.\n"
            f"If the chart does not contain enough information to answer, respond exactly: '{UNANSWERABLE_LABEL}'.\n\n"
            f"Question: {q}\nAnswer:")


def prompt_rationale(q):
    return ("You are looking at a scientific chart extracted from an arXiv paper. Think step by step about the "
            "question below, describing what you see in the chart that is relevant. Keep it under 120 words. "
            "Do NOT state the final answer yet.\n\n"
            f"Question: {q}\nReasoning:")


def prompt_answer(q, rationale):
    return ("You are looking at a scientific chart extracted from an arXiv paper.\n"
            f"Question: {q}\nReasoning so far: {rationale}\n\n"
            "Using the chart and the reasoning above, give the final answer with the SHORTEST possible answer "
            "(a single word, short phrase, or number) — no explanation.\n"
            f"If the chart does not contain enough information, respond exactly: '{UNANSWERABLE_LABEL}'.\nAnswer:")


# ---------------------------------------------------------------------------
# Chạy thí nghiệm
# ---------------------------------------------------------------------------
def run():
    from datasets import load_dataset
    from tqdm import tqdm
    import torch

    val = load_dataset("princeton-nlp/CharXiv", split="validation")
    n = min(N_CHARTS, len(val))

    resume_from_hf()
    rows = []
    done = set()
    if os.path.exists(OUT_FILE):
        prev = pd.read_csv(OUT_FILE, keep_default_na=False)
        rows = prev.to_dict("records")
        done = set(prev.chart_index.astype(int))
        print(f"Tiếp tục: đã có {len(done)}/{n} biểu đồ.")
    if len(done) >= n:
        print("Đã xong hết, không cần chạy.")
        return

    model, processor = load_qwen()
    for i in tqdm(range(n)):
        if i in done:
            continue
        s = val[i]
        q = s["reasoning_q"]
        try:
            direct = generate(model, processor, s["image"], prompt_direct(q), MAX_TOKENS_ANSWER)
            rationale = generate(model, processor, s["image"], prompt_rationale(q), MAX_TOKENS_RATIONALE)
            cot = generate(model, processor, s["image"], prompt_answer(q, rationale), MAX_TOKENS_ANSWER)
        except Exception as e:
            print(f"[Lỗi biểu đồ {i}]: {e}")
            direct, rationale, cot = "", "", ""
        rows.append({"chart_index": i, "sub_type": REASONING_TYPE_MAP.get(s.get("reasoning_a_type")),
                     "question": q, "gold_answer": s["reasoning_a"],
                     "direct_pred": direct, "rationale": rationale, "cot_pred": cot})
        pd.DataFrame(rows).to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
        if len(rows) % PUSH_EVERY == 0:
            push_to_hf(f"cot checkpoint: {len(rows)}/{n} biểu đồ")
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    push_to_hf("cot: hoàn tất")
    print(f"Xong. Kết quả: {OUT_FILE}")
    if HF_TOKEN:
        print(f"Bản trên HuggingFace: https://huggingface.co/datasets/{HF_REPO_ID}/blob/main/{HF_FILE}")


# ---------------------------------------------------------------------------
# Phân tích (chạy trên máy: python cot_experiment.py analyze cot_results.csv)
# ---------------------------------------------------------------------------
def analyze(path):
    from charxiv_match import match_fixed
    df = pd.read_csv(path, keep_default_na=False)
    df["direct_ok"] = [match_fixed(p, g) for p, g in zip(df.direct_pred, df.gold_answer)]
    df["cot_ok"] = [match_fixed(p, g) for p, g in zip(df.cot_pred, df.gold_answer)]
    n = len(df)
    n_err = int((df.rationale.astype(str).str.strip() == "").sum())
    print(f"Số câu: {n} (số câu bị lỗi/rỗng khi sinh: {n_err})")
    print(f"Trả lời thẳng : {df.direct_ok.mean()*100:.1f}%")
    print(f"Hai giai đoạn : {df.cot_ok.mean()*100:.1f}%")

    rng = np.random.default_rng(42)
    d = df.cot_ok.astype(int).values - df.direct_ok.astype(int).values
    boots = d[rng.integers(0, n, (5000, n))].mean(1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    only_cot = int(((df.cot_ok) & (~df.direct_ok)).sum())
    only_dir = int(((df.direct_ok) & (~df.cot_ok)).sum())
    print(f"Hiệu (hai giai đoạn - thẳng): {d.mean()*100:+.1f} điểm %, KTC 95% [{lo:+.1f}, {hi:+.1f}] "
          f"({'chứa 0' if lo <= 0 <= hi else 'không chứa 0'})")
    print(f"Chỉ hai giai đoạn đúng: {only_cot} câu | chỉ trả lời thẳng đúng: {only_dir} câu")

    print("\nTheo kiểu đáp án (mẫu nhỏ, chỉ tham khảo):")
    t = df.groupby("sub_type").agg(n=("direct_ok", "size"), thang=("direct_ok", "mean"), hai_gd=("cot_ok", "mean"))
    t[["thang", "hai_gd"]] = (t[["thang", "hai_gd"]] * 100).round(1)
    print(t.to_string())

    wrong = df[~df.cot_ok & (df.rationale.astype(str).str.strip() != "")]
    sample = wrong.sample(min(50, len(wrong)), random_state=42)[
        ["chart_index", "sub_type", "question", "gold_answer", "rationale", "cot_pred"]].copy()
    sample["loai_loi (tu dien)"] = ""
    out = os.path.join(os.path.dirname(os.path.abspath(path)), "cot_wrong_samples.csv")
    sample.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu {len(sample)} câu sai kèm lý do để bạn đọc và phân loại thủ công: {out}")
    print("Gợi ý cột loai_loi: 'lý do mâu thuẫn với ảnh' / 'lý do đúng nhưng đáp án sai' / 'đọc sai số hoặc chữ' / khác.")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "analyze":
        analyze(sys.argv[2])
    else:
        run()
