# -*- coding: utf-8 -*-
"""charxiv_match.py — Hàm chấm match_fixed dùng chung cho toàn bộ dự án VLM-Charts.
Chuyển tiếp tự động đến eval/charxiv_match.py.
"""
import os
import sys

_EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from charxiv_match import *
