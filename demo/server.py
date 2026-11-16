# -*- coding: utf-8 -*-
"""demo/server.py — Máy chủ Backend cho ứng dụng Hỏi Đáp Thị Giác VLM (VQA Demo).

KIẾN TRÚC & NGUYÊN LÝ HOẠT ĐỘNG CỦA BẢN DEMO (HỆ THỐNG DUAL-ENGINE - CLOUD ONLY):
1. Kiến trúc Song Mã (Dual-Engine Architecture - 100% Cloud):
   - Engine 1 (Cloud VLM API): Sử dụng mô hình đa phương thức Gemini qua Google GenAI API
     (gemini-2.5-flash hoặc gemini-3.5-flash-lite) để cung cấp phản hồi tương tác tức thì
     trên mọi máy tính (kể cả máy không có GPU rời hoặc GPU < 8GB VRAM).
   - Engine 2 (Qwen 7B via HuggingFace API): Gọi mô hình Qwen2.5-VL-7B-Instruct qua
     HuggingFace Inference API (Novita router). Không cần GPU local, chỉ cần HF_TOKEN.
     Hỗ trợ đầy đủ 3 chế độ: Adaptive, No-CoT, Standard CoT.

2. Cơ chế Định Tuyến Tự Động (Adaptive Dynamic Router):
   - Khi chạy ở chế độ Adaptive (Tự Thích Ứng):
     + Đối với câu hỏi tính toán, số học, tọa độ: Kích hoạt nhánh suy luận có cấu trúc 3 bước
       (Target -> Grounding -> Deduction). Nếu nạp thành công trọng số LoRA adapter, nhánh này
       hoạt động như Adaptive LoRA CoT; nếu chưa có LoRA, hệ thống suy luận bằng mô hình Qwen gốc
       (Adaptive CoT) và ghi rõ xuất xứ trọng số để đảm bảo tính trung thực khoa học.
     + Đối với câu hỏi trực quan, nhãn, văn bản: Kích hoạt nhánh Direct Zero-Shot (No-CoT)
       nhằm trả lời ngắn gọn, trực diện, triệt tiêu hiện tượng overthinking.
   - Khi chạy qua Cloud API: Phân tách 3 chế độ (Adaptive, No-CoT, CoT) thông qua Prompt Engineering
     chuyên biệt mô phỏng tương đương cơ chế định tuyến của đề tài.

3. Mốc chuẩn khoa học đối chứng (Offline Benchmarks):
   - Các chỉ số tỷ lệ chính xác (Adaptive: SLAKE 52.59% / CharXiv 39.60%; No-CoT: 50.24% / 38.80%;
     CoT: 45.90% / 36.20%) là kết quả đánh giá thực nghiệm ngoại tuyến của mô hình Qwen2.5-VL-7B
     sau tinh chỉnh trên tập kiểm thử (1.061 câu hỏi SLAKE và 500 câu hỏi CharXiv), được tích hợp
     làm mốc tham chiếu khoa học đối chứng.

Khởi chạy máy chủ:
    python demo/server.py
Hoặc từ thư mục gốc:
    python run_demo.py
"""

import os
import sys
import json
import time
import socket
import base64
import io
import mimetypes
import urllib.parse
import requests as _requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from PIL import Image, ImageOps

# ==============================================================================
# HuggingFace Inference API — Qwen2.5-VL-7B-Instruct (không cần GPU local)
# ==============================================================================
HF_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-VL-7B-Instruct/v1/chat/completions"
HF_API_URL_FALLBACK = "https://router.huggingface.co/novita/v3/openai/chat/completions"
HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

def get_hf_token():
    """Đọc HF_TOKEN động: ưu tiên env var, fallback sang .env / hf_token.txt."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    # Fallback: đọc từ file
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_token.txt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "HF_TOKEN=" in line:
                            return line.split("HF_TOKEN=", 1)[1].strip().strip('"\'')
                        elif line.startswith("hf_"):
                            return line
            except Exception:
                pass
    return ""

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DEMO_DIR)
STATIC_DIR = os.path.join(DEMO_DIR, "static")
SAMPLE_IMAGES_DIR = os.path.join(DEMO_DIR, "sample_images")

def get_gemini_api_key():
    """Lấy Gemini API key từ biến môi trường hoặc file cục bộ (không commit lên Git)."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    candidates = [
        os.path.join(DEMO_DIR, "api_key.txt"),
        os.path.join(BASE_DIR, "api_key.txt"),
        os.path.join(BASE_DIR, ".env"),
        os.path.join(DEMO_DIR, ".env"),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "GEMINI_API_KEY=" in line:
                            return line.split("GEMINI_API_KEY=", 1)[1].strip().strip('"\'')
                        elif line.startswith("AQ.") or line.startswith("AIza"):
                            return line
            except Exception:
                pass
    return ""

# Khóa API của người dùng (tự động nạp từ môi trường hoặc api_key.txt)
DEFAULT_API_KEY = get_gemini_api_key()

# Thử import google.genai
try:
    from google import genai
except ImportError:
    genai = None

# Danh sách model tương thích được ưu tiên (Flash 2.5 & 3.5 nhanh, ổn định)
CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest"
]


def find_free_port(start_port=8501, max_tries=50):
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def resolve_image_path(image_ref):
    """Tìm đường dẫn ảnh thực tế, tự động fallback vào demo/sample_images nếu cần."""
    if not image_ref:
        return None
    clean = os.path.normpath(str(image_ref)).lstrip("\\/.")

    # 1. Thử trực tiếp từ BASE_DIR
    p1 = os.path.join(BASE_DIR, clean)
    if os.path.exists(p1) and os.path.isfile(p1):
        return p1

    # 2. Thử trực tiếp từ DEMO_DIR
    p2 = os.path.join(DEMO_DIR, clean)
    if os.path.exists(p2) and os.path.isfile(p2):
        return p2

    # 3. Thử tìm trong sample_images
    if "chart_00004" in clean:
        p3 = os.path.join(SAMPLE_IMAGES_DIR, "charxiv", "chart_00004.png")
        if os.path.exists(p3):
            return p3
    if "chart_00014" in clean:
        p4 = os.path.join(SAMPLE_IMAGES_DIR, "charxiv", "chart_00014.png")
        if os.path.exists(p4):
            return p4
    if "xmlab102" in clean:
        p5 = os.path.join(SAMPLE_IMAGES_DIR, "slake", "xmlab102_source.jpg")
        if os.path.exists(p5):
            return p5
    if "xmlab124" in clean:
        p6 = os.path.join(SAMPLE_IMAGES_DIR, "slake", "xmlab124_source.jpg")
        if os.path.exists(p6):
            return p6
    if "xmlab135" in clean:
        p7 = os.path.join(SAMPLE_IMAGES_DIR, "slake", "xmlab135_source.jpg")
        if os.path.exists(p7):
            return p7

    return None


def load_pil_image(image_ref):
    """Nạp ảnh từ base64 hoặc đường dẫn tệp một cách an toàn."""
    if not image_ref:
        return None
    try:
        if isinstance(image_ref, str) and (image_ref.startswith("data:image") or ";base64," in image_ref):
            if "," in image_ref:
                _, b64data = image_ref.split(",", 1)
            else:
                b64data = image_ref
            missing_padding = len(b64data) % 4
            if missing_padding:
                b64data += "=" * (4 - missing_padding)
            img_bytes = base64.b64decode(b64data)
            img = Image.open(io.BytesIO(img_bytes))
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[3])
                return bg
            return img.convert("RGB")
        else:
            full_p = resolve_image_path(image_ref)
            if full_p and os.path.exists(full_p) and os.path.isfile(full_p):
                img = Image.open(full_p)
                try:
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode != "RGBA":
                        img = img.convert("RGBA")
                    bg.paste(img, mask=img.split()[3])
                    return bg
                return img.convert("RGB")
    except Exception as e:
        print(f"Lỗi nạp ảnh: {e}")
        return None
    return None


import re

MODE_BENCHMARKS = {
    "adaptive": {
        "mode_id": "adaptive",
        "name": "Adaptive (Prompt Tự Thích Ứng)",
        "badge": "Tự thích ứng",
        "charxiv_acc": "39.60%",
        "slake_acc": "52.59%",
        "thought": "Mô hình nhận prompt hướng dẫn tự thích ứng: phản xạ trực tiếp cho câu hỏi nhận diện trực quan (tránh overthinking) và phân tích thận trọng cho câu hỏi biểu đồ/số liệu phức tạp. Tỷ lệ 52.59% (SLAKE) và 39.60% (CharXiv) là kết quả đánh giá thực nghiệm ngoại tuyến của mô hình Qwen khi áp dụng cơ chế Dynamic Routing.",
        "description": "Mô hình tự quyết định độ sâu suy luận: trả lời thẳng nếu là nhận diện trực quan, suy luận kỹ nếu là số liệu (Mốc Qwen: SLAKE 52.59%, CharXiv 39.60%)."
    },
    "no_cot": {
        "mode_id": "no_cot",
        "name": "No-CoT (Trực tiếp)",
        "badge": "Direct Baseline",
        "charxiv_acc": "38.80%",
        "slake_acc": "50.24%",
        "thought": "Mô hình nhận prompt ép trả lời trực tiếp trong 1 bước, không diễn giải trung gian. Tỷ lệ 50.24% (SLAKE) và 38.80% (CharXiv) là mốc thực nghiệm ngoại tuyến của Qwen Direct Zero-Shot.",
        "description": "Phản xạ thị giác trực tiếp, trả lời thẳng vào đáp án không qua bước suy luận (Mốc Qwen: SLAKE 50.24%, CharXiv 38.80%)."
    },
    "cot": {
        "mode_id": "cot",
        "name": "Standard CoT (Từng bước)",
        "badge": "Chuỗi suy luận",
        "charxiv_acc": "36.20%",
        "slake_acc": "45.90%",
        "thought": "Mô hình nhận prompt ép lập luận 3 bước (Quan sát -> Phân tích -> Kết luận). Tỷ lệ 45.90% (SLAKE) và 36.20% (CharXiv) là mốc thực nghiệm ngoại tuyến của Qwen Standard CoT.",
        "description": "Phân rã thành 3 bước suy luận tuần tự (Mốc Qwen: SLAKE 45.90%, CharXiv 36.20%)."
    }
}


def clean_latex_and_symbols(text: str) -> str:
    """Loại bỏ ký tự LaTeX, công thức toán học thô để chuyển về Tiếng Việt tự nhiên dễ hiểu."""
    if not text:
        return ""
    text = re.sub(r'\$10\^0\$', '1', text)
    text = re.sub(r'\$10\^1\$', '10', text)
    text = re.sub(r'\$10\^2\$', '100', text)
    text = re.sub(r'\$10\^3\$', '1.000', text)
    text = re.sub(r'\$(\d+)\s*\\cdot\s*10\^0\$', r'\g<1>', text)
    text = re.sub(r'\$(\d+)\s*\\cdot\s*10\^1\$', r'\g<1>0', text)
    text = text.replace(r'\%', '%')
    text = text.replace(r'\cdot', ' × ')
    text = text.replace(r'\times', ' × ')
    text = text.replace(r'\approx', ' ≈ ')
    text = text.replace(r'\le', ' ≤ ')
    text = text.replace(r'\ge', ' ≥ ')
    text = text.replace(r'\pm', ' ± ')
    text = text.replace('$', '')
    text = text.replace('\\', '')
    return text.strip()


def run_live_gemini_vqa(pil_image, question, mode="adaptive", api_key=None):
    """Gửi trực tiếp ảnh và câu hỏi đến Gemini VLM theo hình thức trả lời được chọn."""
    if genai is None:
        raise RuntimeError("Chưa cài đặt thư viện google-genai. Vui lòng chạy: pip install google-genai")

    use_key = api_key or DEFAULT_API_KEY
    if not use_key:
        raise RuntimeError("Thiếu Gemini API Key để thực hiện suy luận trực tiếp.")

    client = genai.Client(api_key=use_key)
    meta = MODE_BENCHMARKS.get(mode, MODE_BENCHMARKS["adaptive"])

    if mode == "no_cot":
        instruction = (
            "Bạn đang hoạt động ở chế độ Trả lời Trực tiếp (No-CoT / Direct Baseline). "
            "Hãy nhìn bức ảnh và đưa ra câu trả lời trực tiếp ngay lập tức, cực kỳ ngắn gọn, súc tích, "
            "không trình bày dài dòng hay giải thích các bước trung gian."
        )
    elif mode == "cot":
        instruction = (
            "Bạn đang hoạt động ở chế độ Chuỗi Suy luận Từng bước (Standard Chain-of-Thought - CoT). "
            "Hãy phân tích bức ảnh theo trình tự 3 bước rõ ràng:\n"
            "1. Quan sát: Ghi nhận các thành phần trên ảnh (trục tọa độ, nhãn, vùng giải phẫu).\n"
            "2. Phân tích: Đối chiếu chi tiết câu hỏi với các vùng tương ứng.\n"
            "3. Kết luận: Đưa ra đáp án cuối cùng."
        )
    else:  # adaptive
        instruction = (
            "Bạn đang hoạt động ở chế độ Tự Thích Ứng (Adaptive Prompting). "
            "Tùy vào tính chất của câu hỏi:\n"
            "- Nếu là câu hỏi quan sát trực quan đơn giản, đọc nhãn hoặc nhận diện giải phẫu: Hãy trả lời trực tiếp, rõ ràng, vào thẳng trọng tâm (tránh overthinking).\n"
            "- Nếu là câu hỏi đo đạc mốc tọa độ hoặc tính toán số học phức tạp: Hãy quan sát kỹ mốc tọa độ và suy luận cẩn trọng từng bước trước khi đưa ra kết luận."
        )

    prompt = (
        "Bạn là một mô hình AI Đa phương thức chuyên nghiệp (Vision-Language Model).\n"
        f"{instruction}\n\n"
        f"Câu hỏi: {question}\n\n"
        "CHÚ Ý QUAN TRỌNG VỀ ĐỊNH DẠNG VÀ NGÔN NGỮ:\n"
        "- Trả lời hoàn toàn bằng Tiếng Việt thuần túy, tự nhiên, dễ đọc, mạch lạc.\n"
        "- TUYỆT ĐỐI KHÔNG sử dụng ký hiệu mã lệnh LaTeX (không dùng $, \\cdot, \\%, 10^1...). Hãy viết số và đơn vị bình thường: ví dụ viết '10%', '100%', 'khoảng 50%', 'gần 2'.\n"
        "- Không để lộ các cú pháp kỹ thuật thô."
    )

    t0 = time.time()
    last_err = None

    if pil_image.width > 1200 or pil_image.height > 1200:
        pil_image = pil_image.copy()
        pil_image.thumbnail((1200, 1200))

    for model_name in CANDIDATE_MODELS:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[prompt, pil_image]
            )
            elapsed = round(time.time() - t0, 2)
            raw_text = (resp.text or "").strip()

            if raw_text:
                clean_text = clean_latex_and_symbols(raw_text)
                return {
                    "pred": clean_text,
                    "thought": meta["thought"],
                    "latency": f"{elapsed} giây",
                    "model": model_name,
                    "mode": mode,
                    "mode_name": meta["name"],
                    "charxiv_acc": meta["charxiv_acc"],
                    "slake_acc": meta["slake_acc"],
                    "benchmark_info": f"Mốc thực nghiệm Qwen offline: SLAKE: {meta['slake_acc']} | CharXiv: {meta['charxiv_acc']}"
                }
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Lỗi gọi Gemini API: {last_err}")


# ==============================================================================
# HỆ THỐNG DUAL-ENGINE: QUẢN LÝ PHẦN CỨNG & MÔ HÌNH QWEN2.5-VL CỤC BỘ
# ==============================================================================
import subprocess


def run_hf_qwen_inference(question, image_b64, route):
    """Gọi Qwen2.5-VL-7B qua HuggingFace Inference API (không cần GPU local)."""
    hf_token = get_hf_token()
    if not hf_token:
        return {
            "error": "HF_TOKEN chưa được cấu hình. Vui lòng đặt biến môi trường HF_TOKEN.",
            "model": "Qwen2.5-VL-7B (HF API)",
            "route": route
        }

    # Build system prompt theo route
    if route == "cot":
        system_msg = (
            "Bạn là chuyên gia phân tích hình ảnh y khoa và biểu đồ. "
            "Hãy suy luận từng bước (Chain-of-Thought) trước khi đưa ra câu trả lời cuối. "
            "Phân tích theo 3 bước: Quan sát -> Phân tích -> Kết luận."
        )
        user_prefix = "Hãy suy luận từng bước:\n"
    elif route == "adaptive":
        system_msg = (
            "Bạn là chuyên gia VQA. Phân tích câu hỏi để chọn chiến lược phù hợp: "
            "nếu câu hỏi đơn giản (nhận diện nhãn, trực quan) trả lời ngắn gọn trực tiếp; "
            "nếu câu hỏi phức tạp (số học, tọa độ, so sánh) hãy suy luận từng bước."
        )
        user_prefix = ""
    else:  # direct / no_cot
        system_msg = (
            "Bạn là chuyên gia phân tích hình ảnh. "
            "Trả lời ngắn gọn, chính xác, trực tiếp — không giải thích dài dòng."
        )
        user_prefix = ""

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": f"{user_prefix}{question}"}
            ]
        }
    ]

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.1
    }

    t0 = time.time()
    try:
        # Thử endpoint chính (api-inference.huggingface.co)
        last_err = None
        resp = None
        for url in [HF_API_URL, HF_API_URL_FALLBACK]:
            try:
                resp = _requests.post(url, headers=headers, json=payload, timeout=90)
                resp.raise_for_status()
                break  # thành công
            except Exception as e:
                last_err = e
                resp = None
                continue
        if resp is None:
            raise last_err
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        elapsed = round(time.time() - t0, 2)

        # Xác định nhãn route
        route_labels = {
            "direct": "No-CoT (Trực tiếp)",
            "no_cot": "No-CoT (Trực tiếp)",
            "cot": "Standard CoT (Từng bước)",
            "adaptive": "Adaptive LoRA CoT" if ENGINE_STATE.get("lora_loaded") else "Adaptive CoT"
        }

        meta = MODE_BENCHMARKS.get(route, MODE_BENCHMARKS.get("adaptive", {}))
        clean_answer = clean_latex_and_symbols(answer)
        return {
            "pred": clean_answer,
            "thought": f"[Qwen2.5-VL-7B · HuggingFace API] Chế độ: {route_labels.get(route, route)}. Mô hình suy luận qua HF Inference API (Novita router) — không cần GPU local.",
            "latency": f"{elapsed} giây (HF API)",
            "model": "Qwen2.5-VL-7B-Instruct (HF API)",
            "engine": "hf_qwen",
            "mode": route,
            "mode_name": meta.get("name", route_labels.get(route, route)),
            "charxiv_acc": meta.get("charxiv_acc", "39.60%"),
            "slake_acc": meta.get("slake_acc", "52.59%"),
            "benchmark_info": f"Mô hình Qwen2.5-VL-7B qua HuggingFace API | Mốc offline: SLAKE {meta.get('slake_acc','52.59%')} · CharXiv {meta.get('charxiv_acc','39.60%')}"
        }
    except Exception as e:
        err_str = str(e)
        # Phân loại lỗi rõ ràng hơn cho người dùng
        if "NameResolutionError" in err_str or "getaddrinfo failed" in err_str or "Could not resolve" in err_str:
            user_msg = (
                "Mạng hiện tại chặn kết nối đến HuggingFace. "
                "Giải pháp: (1) Dùng VPN rồi thử lại, hoặc "
                "(2) Chuyển sang engine Cloud VLM (Gemini) vẫn hoạt động bình thường."
            )
        elif "401" in err_str or "Unauthorized" in err_str:
            user_msg = "HF_TOKEN không hợp lệ hoặc hết hạn. Vui lòng tạo token mới tại huggingface.co/settings/tokens."
        elif "timeout" in err_str.lower():
            user_msg = "Kết nối đến HuggingFace bị timeout. Thử lại sau hoặc dùng Cloud VLM."
        else:
            user_msg = f"Lỗi kết nối HuggingFace: {err_str[:120]}"
        return {
            "error": user_msg,
            "model": "Qwen2.5-VL-7B (HF API)",
            "route": route
        }


def probe_nvidia_smi():
    """Truy vấn thông số GPU vật lý qua nvidia-smi (kể cả khi PyTorch chưa cài CUDA)."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=3
        )
        lines = res.stdout.strip().split("\n")
        if lines and lines[0]:
            name, mem = [x.strip() for x in lines[0].split(",")]
            vram_mb = int(mem)
            return {
                "detected": True,
                "name": name,
                "vram_mb": vram_mb,
                "vram_gb": round(vram_mb / 1024, 2)
            }
    except Exception:
        pass
    return {"detected": False, "name": "N/A", "vram_mb": 0, "vram_gb": 0.0}


def get_hardware_telemetry():
    """Kiểm tra toàn diện môi trường phần cứng: CUDA, VRAM và tính sẵn sàng cho Qwen 7B."""
    torch_cuda = False
    gpu_name = "Không phát hiện GPU"
    vram_gb = 0.0
    cuda_ver = "N/A"

    try:
        import torch
        if torch.cuda.is_available():
            torch_cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(vram_bytes / (1024 ** 3), 2)
            cuda_ver = getattr(torch.version, "cuda", "N/A") or "N/A"
    except Exception:
        pass

    # Nếu PyTorch chưa thấy CUDA, kiểm tra phần cứng vật lý qua nvidia-smi
    smi = probe_nvidia_smi()
    if not torch_cuda and smi["detected"]:
        gpu_name = smi["name"]
        vram_gb = smi["vram_gb"]

    # Qwen2.5-VL-7B lượng tử hóa 4-bit NF4 cần tối thiểu ~7.5 - 8.0 GB VRAM để chạy mượt mà
    is_sufficient = (vram_gb >= 7.5) and torch_cuda

    return {
        "gpu_available": torch_cuda or smi["detected"],
        "torch_cuda": torch_cuda,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "cuda_version": cuda_ver,
        "is_sufficient_for_7b": is_sufficient,
        "recommended_vram": "8.0 GB (Tesla T4 hoặc RTX 3090/4090/A100)"
    }


# Trạng thái toàn cục của hệ thống Dual-Engine
ENGINE_STATE = {
    "active_engine": "cloud",  # "cloud" hoặc "hf_qwen"
    "local_model": None,
    "local_processor": None,
    "local_status": "unloaded",  # "unloaded", "loading", "ready", "error"
    "local_error": None,
    "lora_loaded": False,
    "lora_path": None
}


def load_local_qwen_model():
    """Tải cục bộ mô hình Qwen2.5-VL-7B-Instruct 4-bit NF4 và nạp trọng số LoRA nếu có."""
    global ENGINE_STATE
    if ENGINE_STATE["local_status"] == "ready" and ENGINE_STATE["local_model"] is not None:
        return ENGINE_STATE["local_model"], ENGINE_STATE["local_processor"]

    ENGINE_STATE["local_status"] = "loading"
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"[*] Đang khởi tạo nạp mô hình Qwen cục bộ: {model_id}...")

        hw = get_hardware_telemetry()
        if hw["torch_cuda"]:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            quant_config = None
            device_map = "cpu"
            torch_dtype = torch.float32

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(model_id)

        # Kiểm tra nạp LoRA adapter nếu có trong thư mục models
        lora_candidates = [
            os.path.join(BASE_DIR, "models", "lora_adapter"),
            os.path.join(BASE_DIR, "models", "lora_charts_output"),
            os.path.join(BASE_DIR, "lora_output"),
        ]
        lora_loaded = False
        lora_path = None
        for candidate in lora_candidates:
            if os.path.exists(candidate) and os.path.isdir(candidate):
                try:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, candidate)
                    lora_loaded = True
                    lora_path = candidate
                    print(f"[+] Đã nạp thành công LoRA adapter từ: {candidate}")
                    break
                except Exception as lora_err:
                    print(f"[!] Không thể nạp LoRA từ {candidate}: {lora_err}")

        ENGINE_STATE["local_model"] = model
        ENGINE_STATE["local_processor"] = processor
        ENGINE_STATE["local_status"] = "ready"
        ENGINE_STATE["lora_loaded"] = lora_loaded
        ENGINE_STATE["lora_path"] = lora_path
        ENGINE_STATE["local_error"] = None
        return model, processor

    except Exception as e:
        ENGINE_STATE["local_status"] = "error"
        ENGINE_STATE["local_error"] = str(e)
        raise e


def run_local_qwen_inference(pil_image, question, mode="adaptive"):
    """Thực thi suy luận 100% cục bộ trên mô hình Qwen2.5-VL-7B với cơ chế Router thật."""
    t0 = time.time()
    meta = MODE_BENCHMARKS.get(mode, MODE_BENCHMARKS["adaptive"])

    # 1. Cơ chế Dynamic Router thực tế (Phân tích câu hỏi tự động)
    routed_mode = mode
    routed_reason = "Theo chế độ người dùng lựa chọn."
    if mode == "adaptive":
        # Nhận diện câu hỏi tính toán / số học vs câu hỏi nhận thức nhãn / trực quan
        is_numerical = any(kw in question.lower() for kw in [
            "how many", "what is the value", "difference", "average", "maximum", "minimum",
            "sum", "calculate", "count", "score", "rate", "percentage", "%", "tính", "bao nhiêu", "đếm"
        ])
        if is_numerical:
            routed_mode = "cot"
            routed_reason = "Router tự động nhận diện câu hỏi Số học -> Kích hoạt nhánh suy luận 3 bước CoT."
        else:
            routed_mode = "no_cot"
            routed_reason = "Router tự động nhận diện câu hỏi Nhãn/Văn bản -> Kích hoạt nhánh Direct Zero-Shot để triệt tiêu overthinking."

    # 2. Xây dựng prompt có cấu trúc theo mô hình Qwen đề tài
    if routed_mode == "no_cot":
        sys_instruction = (
            "You are looking at a chart or medical image. "
            "Answer the question with the SHORTEST possible answer (a single word, short phrase, or number) — no explanation."
        )
    else:
        sys_instruction = (
            "Analyze the image step-by-step with structured reasoning:\n"
            "Target: Identify the target entity or curve to observe.\n"
            "Grounding: Read coordinate values, ticks, or visual legend markers carefully.\n"
            "Deduction: Perform arithmetic deduction or comparison.\n"
            "Answer: State the final concise answer."
        )

    # 3. Nạp mô hình & chuẩn bị dữ liệu
    try:
        model, processor = load_local_qwen_model()
    except Exception as err:
        hw = get_hardware_telemetry()
        raise RuntimeError(
            f"Không thể khởi chạy Local Qwen: {err}. "
            f"(Phát hiện GPU: {hw['gpu_name']} - {hw['vram_gb']} GB VRAM; Yêu cầu >= 8GB VRAM). "
            "Gợi ý: Hãy chuyển sang Cloud VLM API để trải nghiệm tức thì."
        )

    # Cập nhật nhãn lý do định tuyến chính xác theo tình trạng nạp LoRA thực tế
    has_lora = bool(ENGINE_STATE.get("lora_loaded"))
    if routed_mode == "cot":
        if mode == "adaptive":
            if has_lora:
                routed_reason = "Router tự động nhận diện câu hỏi Số học -> Kích hoạt nhánh suy luận 3 bước Adaptive LoRA CoT (đã nạp trọng số LoRA adapter)."
            else:
                routed_reason = "Router tự động nhận diện câu hỏi Số học -> Kích hoạt nhánh suy luận 3 bước Adaptive CoT (mô hình Qwen gốc, chưa nạp LoRA adapter)."
        else:
            if has_lora:
                routed_reason = "Chế độ Standard CoT + LoRA (suy luận 3 bước với trọng số LoRA adapter)."
            else:
                routed_reason = "Chế độ Standard CoT (suy luận 3 bước trên mô hình Qwen gốc, chưa nạp LoRA adapter)."
    else:
        if mode == "adaptive":
            routed_reason = "Router tự động nhận diện câu hỏi Nhãn/Văn bản -> Kích hoạt nhánh Direct Zero-Shot để triệt tiêu overthinking."
        else:
            routed_reason = "Chế độ Direct Zero-Shot (No-CoT): Trả lời trực tiếp ngắn gọn, không giải thích."

    import torch
    from qwen_vl_utils import process_vision_info

    # Giới hạn kích thước ảnh phù hợp VRAM
    img_ready = pil_image.copy()
    if max(img_ready.size) > 1024:
        scale = 1024 / max(img_ready.size)
        img_ready = img_ready.resize((int(img_ready.width * scale), int(img_ready.height * scale)))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_ready},
                {"type": "text", "text": f"{sys_instruction}\n\nQuestion: {question}\nAnswer:"}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    elapsed = round(time.time() - t0, 2)
    clean_text = clean_latex_and_symbols(raw_output)

    lora_badge = " + LoRA" if ENGINE_STATE.get("lora_loaded") else ""
    return {
        "pred": clean_text,
        "thought": f"🖥️ [Local Qwen2.5-VL GPU] {routed_reason}\n\nChuỗi sinh từ mô hình cục bộ:\n{clean_text}",
        "latency": f"{elapsed} giây (Local GPU)",
        "model": f"Qwen2.5-VL-7B-Instruct (4-bit NF4{lora_badge})",
        "engine": "local_qwen",
        "mode": mode,
        "mode_name": meta["name"],
        "charxiv_acc": meta["charxiv_acc"],
        "slake_acc": meta["slake_acc"],
        "benchmark_info": f"Mô hình cục bộ Qwen2.5-VL-7B (Local GPU) | Mốc offline: SLAKE {meta['slake_acc']} · CharXiv {meta['charxiv_acc']}"
    }


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # API trạng thái hệ thống và đo kiểm phần cứng GPU
        if path == "/api/engine_status":
            hw = get_hardware_telemetry()
            data = {
                "active_engine": ENGINE_STATE["active_engine"],
                "hardware": hw,
                "local_status": ENGINE_STATE["local_status"],
                "local_error": ENGINE_STATE["local_error"],
                "lora_loaded": ENGINE_STATE["lora_loaded"],
                "lora_path": ENGINE_STATE["lora_path"],
                # Thông tin HF API engine
                "hf_token_configured": bool(get_hf_token()),
                "qwen_engine": "hf_api",
                "qwen_model": "Qwen2.5-VL-7B-Instruct",
                "qwen_via": "HuggingFace Inference API (Novita)"
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            return

        # Xử lý phục vụ ảnh: /api/image hoặc đường dẫn ảnh trực tiếp
        if path == "/api/image" or path.startswith("/charxiv_all_images/") or path.startswith("/imgs/") or path.startswith("/sample_images/"):
            if path == "/api/image":
                img_rel = query.get("path", [""])[0]
            else:
                img_rel = path.lstrip("/")
            
            full_path = resolve_image_path(img_rel)

            if full_path and os.path.exists(full_path) and os.path.isfile(full_path):
                mime, _ = mimetypes.guess_type(full_path)
                mime = mime or "image/png"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                with open(full_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Image not found")
                return

        # Phục vụ static files (HTML, CSS, JS)
        if path == "/" or path == "/index.html":
            file_path = os.path.join(STATIC_DIR, "index.html")
        else:
            rel = path.lstrip("/")
            file_path = os.path.join(STATIC_DIR, rel)

        if os.path.exists(file_path) and os.path.isfile(file_path):
            mime, _ = mimetypes.guess_type(file_path)
            mime = mime or "text/plain"
            if mime.startswith("text/") or mime == "application/javascript":
                mime += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length)

        # Cấu hình đổi Engine suy luận: cloud vs hf_qwen
        if parsed.path == "/api/set_engine":
            try:
                payload = json.loads(post_body.decode("utf-8"))
                target = payload.get("engine", "cloud")
                if target not in ["cloud", "hf_qwen", "local_qwen"]:
                    raise ValueError("Engine không hợp lệ. Chỉ chấp nhận 'cloud' hoặc 'hf_qwen'.")
                # Normalize: local_qwen -> hf_qwen (backward compat)
                if target == "local_qwen":
                    target = "hf_qwen"
                ENGINE_STATE["active_engine"] = target
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "active_engine": target
                }, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))
            return

        if parsed.path == "/api/vqa":
            try:
                payload = json.loads(post_body.decode("utf-8"))
                question = payload.get("question", "").strip()
                image_ref = payload.get("image", "")
                custom_key = payload.get("api_key", "").strip()
                mode = payload.get("mode", "adaptive")
                engine = payload.get("engine", ENGINE_STATE["active_engine"])

                pil_img = load_pil_image(image_ref)
                if pil_img is None:
                    raise ValueError("Không thể tìm thấy hoặc đọc hình ảnh đầu vào.")

                # Gọi trực tiếp mô hình AI thật theo Engine được chọn
                if engine in ("local_qwen", "hf_qwen"):
                    # Encode ảnh sang base64 để gửi qua HF API
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=90)
                    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    response_data = run_hf_qwen_inference(question, image_b64, mode)
                    if "error" in response_data:
                        raise RuntimeError(response_data["error"])
                else:
                    response_data = run_live_gemini_vqa(pil_img, question, mode=mode, api_key=custom_key)

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def start_server(port=8501):
    port = find_free_port(port)
    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, DemoHandler)
    url = f"http://localhost:{port}"
    print("=" * 70)
    print("HE THONG HOI DAP THI GIAC VLM SAN SANG (LIVE AI)!")
    print(f"Dia chi truy cap: {url}")
    print("=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDa tat Demo Server an toan.")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
