# -*- coding: utf-8 -*-
"""run_demo.py — Trình khởi chạy hệ thống Demo Hỏi Đáp Thị Giác VLM (VQA Demo).

Khởi chạy nhanh từ thư mục gốc:
    python run_demo.py
Hoặc chạy trực tiếp từ thư mục demo:
    python demo/server.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(BASE_DIR, "demo")

if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from server import start_server

if __name__ == "__main__":
    start_server()
