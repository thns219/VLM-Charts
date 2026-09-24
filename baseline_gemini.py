"""
Baseline: dùng Gemini API để trả lời câu hỏi trên biểu đồ CharXiv
------------------------------------------------------------------
Cài đặt trước khi chạy:
    pip install google-genai datasets pillow pandas tqdm

Thiết lập API key (KHÔNG hardcode key vào code):
    PowerShell:  $env:GEMINI_API_KEY = "dán_key_của_bạn_vào_đây"

Chạy:
    python baseline_gemini.py

CẤU TRÚC DỮ LIỆU THẬT của CharXiv (đã xác nhận từ dataset + repo gốc):
    Mỗi HÀNG trong dataset = 1 biểu đồ, đi kèm 5 câu hỏi:
      - descriptive_q1..q4: SỐ NGUYÊN, là ID trỏ tới 1 trong 19 câu hỏi MẪU cố định
        (bảng ánh xạ DESCRIPTIVE_QMAP bên dưới, lấy từ src/constants.py của repo gốc)
      - descriptive_a1..a4: đáp án dạng text tương ứng; "Not Applicable" = câu không
        trả lời được (đúng 1 trong 4 câu descriptive luôn là loại này)
      - reasoning_q: câu hỏi suy luận dạng TEXT ĐẦY ĐỦ (không cần tra bảng)
      - reasoning_a: đáp án của câu suy luận
      - reasoning_a_type: 1=Text-in-Chart, 2=Text-in-General, 3=Number-in-Chart,
        4=Number-in-General (chỉ để phân loại, không ảnh hưởng nội dung câu hỏi)

Script này duyệt qua từng biểu đồ, sinh ra 5 câu hỏi cho mỗi biểu đồ, gọi Gemini
cho từng câu, rồi tổng hợp accuracy — tách riêng theo loại "descriptive" và
"reasoning" vì độ khó của 2 loại rất khác nhau.
"""

import os
import time

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from google import genai
from google.genai import errors as genai_errors

# ==== Bảng ánh xạ ID -> câu hỏi mẫu (lấy nguyên văn từ src/constants.py của
#      repo gốc princeton-nlp/CharXiv, KHÔNG được tự đoán/sửa lại) ====
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
MODEL_NAME = "gemini-flash-latest"
N_CHARTS = 20                          # số biểu đồ thử nghiệm (mỗi biểu đồ = 5 câu hỏi)
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 5
MAX_IMAGE_SIDE = 1024
CHECKPOINT_FILE = "baseline_results.csv"
UNANSWERABLE_LABEL = "Not Applicable"   # đúng chuẩn CharXiv, giữ nguyên tiếng Anh để so khớp với đáp án gốc


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


def ask_gemini(client, image, question: str) -> str:
    prompt = build_prompt(question)
    image = resize_image(image)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, image],
            )
            return response.text.strip() if response.text else ""
        except genai_errors.ClientError as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  Rate limit, chờ {wait}s rồi thử lại (lần {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            raise
    return ""


def simple_match(pred: str, gold: str) -> bool:
    """So khớp thô: chuẩn hoá chữ thường, kiểm tra bao hàm.
    Xử lý riêng trường hợp 'Not Applicable' (unanswerable).
    Đây chỉ để có con số nhanh — báo cáo chính thức nên dùng script chấm điểm
    gốc của CharXiv (src/evaluate.py, chấm bằng GPT-4o theo đúng rubric)."""
    if gold is None:
        return None
    p = pred.strip().lower()
    g = str(gold).strip().lower()

    gold_unanswerable = (g == UNANSWERABLE_LABEL.lower())
    pred_unanswerable = (UNANSWERABLE_LABEL.lower() in p)
    if gold_unanswerable or pred_unanswerable:
        return gold_unanswerable == pred_unanswerable

    return g in p or p in g


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


def run_baseline():
    print("Đang tải split 'validation' của CharXiv...")
    val_ds = load_dataset("princeton-nlp/CharXiv", split="validation")

    n_charts = min(N_CHARTS, len(val_ds))
    print(f"Chạy baseline trên {n_charts} biểu đồ đầu tiên ({n_charts * 5} câu hỏi)...")

    done_keys = set()
    if os.path.exists(CHECKPOINT_FILE):
        prev = pd.read_csv(CHECKPOINT_FILE)
        done_keys = set(zip(prev["chart_index"], prev["slot"]))
        results = prev.to_dict("records")
        print(f"Tìm thấy checkpoint cũ với {len(done_keys)} câu đã xử lý — chạy tiếp phần còn thiếu.")
    else:
        results = []

    client = get_client()

    for chart_idx in tqdm(range(n_charts)):
        sample = val_ds[chart_idx]
        image = sample["image"]

        for qa in iter_questions(sample):
            key = (chart_idx, qa["slot"])
            if key in done_keys:
                continue

            try:
                pred = ask_gemini(client, image, qa["question"])
            except Exception as e:
                print(f"[Lỗi ở biểu đồ {chart_idx}, {qa['slot']}]: {e}")
                pred = ""

            correct = simple_match(pred, qa["gold_answer"])

            results.append({
                "chart_index": chart_idx,
                "slot": qa["slot"],
                "question_type": qa["question_type"],
                "sub_type": qa.get("sub_type"),
                "question": qa["question"],
                "gold_answer": qa["gold_answer"],
                "prediction": pred,
                "correct": correct,
            })

            pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")
            time.sleep(SLEEP_BETWEEN_CALLS)

    df = pd.DataFrame(results)

    if df["correct"].notna().any():
        overall_acc = df["correct"].mean()
        print(f"\n=== Accuracy thô (simple_match) trên {len(df)} câu hỏi ===")
        print(f"Overall: {overall_acc:.2%}")
        print("\nTheo loại câu hỏi (descriptive vs reasoning):")
        print(df.groupby("question_type")["correct"].mean().to_string())

        reasoning_df = df[df["question_type"] == "reasoning"]
        if reasoning_df["sub_type"].notna().any():
            print("\nTheo phân loại reasoning (Text/Number-in-Chart/General):")
            print(reasoning_df.groupby("sub_type")["correct"].mean().to_string())
    else:
        print("\nKhông tính được accuracy — kiểm tra lại dữ liệu.")

    print(f"\nĐã lưu chi tiết kết quả vào '{CHECKPOINT_FILE}'")
    return df


if __name__ == "__main__":
    run_baseline()