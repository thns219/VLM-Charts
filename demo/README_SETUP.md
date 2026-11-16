# Demo Setup Guide — VLM-Charts VQA Demo

## Giới thiệu

Demo hỏi đáp thị giác (VQA) cho đề tài VLM-Charts với **hai engine hoàn toàn chạy trên Cloud** — không cần GPU local.

| Engine | Model | Yêu cầu | GPU cần thiết |
|--------|-------|---------|---------------|
| **Cloud VLM** | Gemini 2.0/2.5 Flash | `GEMINI_API_KEY` | ❌ Không |
| **Qwen 7B (HF API)** | Qwen2.5-VL-7B-Instruct | `HF_TOKEN` | ❌ Không |

> **Lưu ý:** RTX 3050 Laptop (4GB VRAM) KHÔNG đủ để chạy Qwen2.5-VL-7B local (cần ≥ 8GB VRAM).  
> Thay vào đó, engine **Qwen 7B (HF API)** gọi mô hình qua HuggingFace Inference API — miễn phí với HF Token.

---

## Yêu cầu hệ thống

- Python 3.10+
- Kết nối Internet
- `GEMINI_API_KEY` — bắt buộc cho Cloud VLM (Gemini)
- `HF_TOKEN` — tùy chọn, để dùng Qwen 7B qua HuggingFace API

---

## Cài đặt

```bash
cd demo
pip install -r requirements.txt
```

Hoặc từ thư mục gốc:

```bash
pip install pillow google-genai requests
```

---

## Khởi động

### Chỉ Cloud VLM (Gemini):

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="AIza..."
python demo/server.py
```

```bash
# Linux / macOS
GEMINI_API_KEY=AIza... python demo/server.py
```

### Cloud VLM + Qwen 7B qua HuggingFace API:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="AIza..."
$env:HF_TOKEN="hf_xxx..."
python demo/server.py
```

```bash
# Linux / macOS
GEMINI_API_KEY=AIza... HF_TOKEN=hf_xxx... python demo/server.py
```

---

## Mở browser

Truy cập: **http://localhost:8501** (hoặc port hiện ra trong terminal)

---

## Lấy HF Token (miễn phí)

1. Đăng ký / đăng nhập tại [huggingface.co](https://huggingface.co)
2. Vào **Settings → Access Tokens**
3. Tạo token mới (Read hoặc Write)
4. Đặt `HF_TOKEN=hf_xxx...` và khởi động lại server

---

## Engines & Chế độ suy luận

### Engine 1: Cloud VLM (Gemini API)
- Sử dụng Gemini 2.5 Flash hoặc 2.5 Flash Lite
- Nhanh, không cần GPU, không giới hạn VRAM
- Hỗ trợ 3 chế độ: Adaptive / No-CoT / Standard CoT

### Engine 2: Qwen 7B (HF API)
- Gọi `Qwen/Qwen2.5-VL-7B-Instruct` qua HuggingFace Inference API (Novita router)
- Không cần GPU local, không cần cài `transformers` / `bitsandbytes`
- Chỉ cần `HF_TOKEN` (miễn phí)
- Hỗ trợ 3 chế độ: Adaptive / No-CoT / Standard CoT

### Chế độ suy luận

| Chế độ | Mô tả | Mốc SLAKE | Mốc CharXiv |
|--------|-------|-----------|-------------|
| **Adaptive** | Tự thích ứng: trả lời thẳng hoặc CoT tùy câu hỏi | 52.59% | 39.60% |
| **No-CoT** | Trả lời trực tiếp, không suy luận trung gian | 50.24% | 38.80% |
| **Standard CoT** | Suy luận 3 bước: Quan sát → Phân tích → Kết luận | 45.90% | 36.20% |

> Các mốc trên là kết quả thực nghiệm ngoại tuyến trên mô hình Qwen2.5-VL-7B sau fine-tuning.

---

## Cấu trúc thư mục

```
demo/
├── server.py          # Backend server (Python HTTP)
├── README_SETUP.md    # File hướng dẫn này
├── static/
│   ├── index.html     # Giao diện chính
│   ├── app.js         # Logic frontend
│   └── style.css      # CSS styling
└── sample_images/
    ├── charxiv/       # Ảnh biểu đồ mẫu (CharXiv)
    └── slake/         # Ảnh y khoa mẫu (SLAKE)
```
