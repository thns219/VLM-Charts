"""
Baseline: dùng Gemini API để trả lời câu hỏi trên biểu đồ CharXiv  (BẢN ĐÃ SỬA)
------------------------------------------------------------------
Cài đặt trước khi chạy:
    pip install google-genai datasets pillow pandas tqdm

Thiết lập API key (KHÔNG hardcode key vào code):
    PowerShell:  $env:GEMINI_API_KEY = "dán_key_của_bạn_vào_đây"

Cần có cạnh file này: subset_chart_indices.json, charxiv_match.py

Chạy:
    python baseline_gemini.py

Thay đổi so với bản cũ:
  1. Chấm bằng match_fixed (charxiv_match.py) thay cho simple_match: dự đoán rỗng = sai,
     số so theo giá trị, bỏ LaTeX.
  2. Không bao giờ lưu dự đoán rỗng. Nếu gọi API thất bại, dòng đó KHÔNG được ghi vào CSV,
     nên lần chạy sau sẽ tự thử lại. Lý do lỗi được ghi vào gemini_errors.jsonl.
  3. Retry cho cả lỗi 429 lẫn 5xx/mạng, backoff tăng dần tới 60s.
  4. Gặp hạn mức THEO NGÀY (RPD) thì dừng hẳn và báo, không đốt tiếp các câu còn lại.
  5. Đọc checkpoint bằng keep_default_na=False (đáp án như "None"/"NA" không bị thành NaN),
     và coi các dòng rỗng cũ trong checkpoint là CHƯA LÀM -> tự chạy lại luôn 378 câu rỗng.

CẤU TRÚC DỮ LIỆU CharXiv: mỗi hàng = 1 biểu đồ + 4 câu descriptive (ID trỏ tới 19 câu mẫu,
"Not Applicable" = không trả lời được) + 1 câu reasoning (text đầy đủ).
"""

import os
import re
import json
import time

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from google import genai
from google.genai import errors as genai_errors

from charxiv_match import match_fixed

# ==== Bảng ánh xạ ID -> câu hỏi mẫu (nguyên văn từ src/constants.py của princeton-nlp/CharXiv) ====
DESCRIPTIVE_QMAP = {
    1: "What is the title of the plot?",
    2: "What is the label of the x-axis?",
    3: "What is the label of the y-axis?",
    4: "What is the leftmost labeled tick on the x-axis?",
    5: "What is the rightmost labeled tick on the x-axis?",
    6: "What is the spatially lowest labeled tick on the y-axis?",
    7: "What is the spatially highest labeled tick on the y-axis?",
    8: "What is difference between consecutive numerical tick values on the x-axis?",
    9: "What is difference between consecutive numerical tick values on the y-axis?",
    10: "How many lines are there?",
    11: "Do any lines intersect?",
    12: "How many discrete labels are there in the legend?",
    13: "What are the names of the labels in the legend? (from top to bottom, then left to right)",
    14: "What is the difference between the maximum and minimum values of the tick labels on the continuous legend (i.e., colorbar)?",
    15: "What is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?",
    16: "What is the general trend of data from left to right?",
    17: "What is the total number of explicitly labeled ticks across all axes?",
    18: "What is the layout of the subplots?",
    19: "What is the number of subplots?",
}

REASONING_TYPE_MAP = {
    1: "Text-in-Chart",
    2: "Text-in-General",
    3: "Number-in-Chart",
    4: "Number-in-General",
}

# ==== CẤU HÌNH ====
MODEL_NAME = "gemini-3.5-flash-lite"   # free tier: 30 RPM, 1.500 RPD
SUBSET_FILE = "subset_chart_indices.json"
SLEEP_BETWEEN_CALLS = 2.2
MAX_RETRIES = 6
MAX_IMAGE_SIDE = 1024
CHECKPOINT_FILE = "baseline_results.csv"
ERROR_LOG = "gemini_errors.jsonl"
UNANSWERABLE_LABEL = "Not Applicable"


class DailyQuotaExceeded(Exception):
    """Hết hạn mức theo ngày: dừng chương trình, chạy lại hôm sau."""


def load_chart_subset():
    if not os.path.exists(SUBSET_FILE):
        raise RuntimeError(
            f"Không tìm thấy '{SUBSET_FILE}'. File này chứa danh sách chart_index "
            "cố định (chung với Qwen/RAG/Retrieval) để đảm bảo so sánh công bằng. "
            "Copy file này vào cùng thư mục trước khi chạy."
        )
    with open(SUBSET_FILE) as f:
        return json.load(f)


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Chưa thấy biến môi trường GEMINI_API_KEY. "
            "Thiết lập bằng: $env:GEMINI_API_KEY = 'your_key' (PowerShell) rồi chạy lại."
        )
    return genai.Client(api_key=api_key)


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


def log_error(chart_idx, slot, reason):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"chart_index": int(chart_idx), "slot": slot, "reason": reason,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")


def ask_gemini(client, image, question: str):
    """Trả về (text, lý_do_lỗi). text == "" nghĩa là THẤT BẠI (không được lưu vào CSV)."""
    prompt = build_prompt(question)
    image = resize_image(image)
    last = ""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=[prompt, image])
            text = (response.text or "").strip()
            if text:
                return text, ""
            # HTTP 200 nhưng không có text: ghi lại lý do (bị chặn an toàn, finish_reason...)
            reasons = [str(getattr(c, "finish_reason", "")) for c in (response.candidates or [])]
            last = f"empty_text finish={reasons} feedback={getattr(response, 'prompt_feedback', None)}"
        except genai_errors.APIError as e:
            msg, code = str(e), getattr(e, "code", None)
            if code == 429 and re.search(r"per.?day", msg, re.I):
                raise DailyQuotaExceeded(msg[:300])
            if code in (429, 500, 502, 503, 504):
                last = f"{code}: {msg[:200]}"
            else:                                  # 400/403...: thử lại cũng vô ích
                return "", f"{code}: {msg[:200]}"
        except Exception as e:                     # lỗi mạng, timeout...
            last = f"{type(e).__name__}: {str(e)[:200]}"

        wait = min(60, 3 * 2 ** attempt)           # 3, 6, 12, 24, 48, 60s
        print(f"  Lỗi ({last[:80]}), chờ {wait}s rồi thử lại (lần {attempt + 1}/{MAX_RETRIES})...")
        time.sleep(wait)
    return "", last


def iter_questions(sample):
    """Sinh ra 5 câu hỏi (4 descriptive + 1 reasoning) từ 1 hàng dữ liệu (1 biểu đồ)."""
    for i in range(1, 5):
        qid = sample[f"descriptive_q{i}"]
        answer = sample[f"descriptive_a{i}"]
        question_text = DESCRIPTIVE_QMAP.get(qid, f"[Không rõ template id={qid}]")
        yield {
            "slot": f"descriptive_{i}",
            "question_type": "descriptive",
            "question": question_text,
            "gold_answer": answer,
        }

    yield {
        "slot": "reasoning",
        "question_type": "reasoning",
        "sub_type": REASONING_TYPE_MAP.get(sample.get("reasoning_a_type")),
        "question": sample["reasoning_q"],
        "gold_answer": sample["reasoning_a"],
    }


def save_checkpoint(results):
    tmp = CHECKPOINT_FILE + ".tmp"
    pd.DataFrame(results).to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, CHECKPOINT_FILE)


def run_baseline():
    print("Đang tải split 'validation' của CharXiv...")
    val_ds = load_dataset("princeton-nlp/CharXiv", split="validation")

    chart_indices = load_chart_subset()
    print(f"Chạy baseline trên {len(chart_indices)} biểu đồ đã chọn trước "
          f"(subset seed=42, {len(chart_indices) * 5} câu hỏi)...")

    results, done_keys = [], set()
    if os.path.exists(CHECKPOINT_FILE):
        # keep_default_na=False: giữ "" là "", đáp án "None"/"NA" không bị đổi thành NaN
        prev = pd.read_csv(CHECKPOINT_FILE, keep_default_na=False)
        n_empty = int((prev["prediction"].astype(str).str.strip() == "").sum())
        # Dòng có dự đoán rỗng coi như CHƯA LÀM -> bỏ khỏi checkpoint để chạy lại
        prev = prev[prev["prediction"].astype(str).str.strip() != ""]
        done_keys = set(zip(prev["chart_index"], prev["slot"]))
        results = prev.to_dict("records")
        print(f"Checkpoint: {len(done_keys)} câu hợp lệ; {n_empty} câu rỗng cũ sẽ được chạy lại.")

    client = get_client()
    failed = 0

    try:
        for chart_idx in tqdm(chart_indices):
            sample = val_ds[chart_idx]
            image = sample["image"]

            for qa in iter_questions(sample):
                key = (chart_idx, qa["slot"])
                if key in done_keys:
                    continue

                pred, why = ask_gemini(client, image, qa["question"])
                if not pred:
                    # KHÔNG lưu dòng rỗng: lần chạy sau sẽ tự thử lại câu này
                    failed += 1
                    log_error(chart_idx, qa["slot"], why)
                    print(f"[Bỏ qua biểu đồ {chart_idx}, {qa['slot']}]: {why[:100]}")
                    continue

                results.append({
                    "chart_index": chart_idx,
                    "slot": qa["slot"],
                    "question_type": qa["question_type"],
                    "sub_type": qa.get("sub_type"),
                    "question": qa["question"],
                    "gold_answer": qa["gold_answer"],
                    "prediction": pred,
                    "correct": match_fixed(pred, qa["gold_answer"]),
                })
                save_checkpoint(results)
                time.sleep(SLEEP_BETWEEN_CALLS)

    except DailyQuotaExceeded as e:
        print(f"\nHẾT HẠN MỨC THEO NGÀY — dừng. Chạy lại script này vào hôm sau (tiến độ đã lưu).\n{e}")
    except KeyboardInterrupt:
        print("\nĐã dừng tay, tiến độ đã được lưu.")

    save_checkpoint(results)
    df = pd.DataFrame(results)
    total = len(chart_indices) * 5
    print(f"\nĐã có {len(df)}/{total} câu hợp lệ; {failed} câu lỗi lần này (xem '{ERROR_LOG}').")

    if len(df) and df["correct"].notna().any():
        print(f"\n=== Accuracy (match_fixed) trên {len(df)} câu ===")
        print(f"Overall: {df['correct'].mean():.2%}")
        print("\nTheo loại câu hỏi:")
        print(df.groupby("question_type")["correct"].mean().to_string())
        reasoning_df = df[df["question_type"] == "reasoning"]
        if reasoning_df["sub_type"].notna().any():
            print("\nTheo phân loại reasoning:")
            print(reasoning_df.groupby("sub_type")["correct"].mean().to_string())

    print(f"\nĐã lưu chi tiết vào '{CHECKPOINT_FILE}'")
    return df


if __name__ == "__main__":
    run_baseline()