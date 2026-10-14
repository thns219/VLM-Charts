# -*- coding: utf-8 -*-
r"""charxiv_match.py — hàm chấm match_fixed dùng chung cho toàn bộ dự án VLM-Charts.
Import ở mọi script:  from charxiv_match import match_fixed

Cải tiến chuẩn hóa:
  1. Ký tự Hy Lạp (\sigma -> sigma, σ -> sigma).
  2. Bố cục subplot: "1x2" <-> "1 by 2".
  3. Dải số có dấu gạch ngang: "1 - 2" <-> "1-2".
  4. Tiền tố phân loại: "Case 7" <-> "7", "Subplot (b)" <-> "(b)".
  5. Hậu tố đếm / đơn vị: "7 lines" <-> "7", "10%" <-> "10".
  6. Dấu bằng và dấu phẩy: "t = 5" <-> "t=5".
"""
import re

NA = "not applicable"

GREEK_MAP = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
    'ε': 'epsilon', 'θ': 'theta', 'κ': 'kappa', 'λ': 'lambda',
    'μ': 'mu', 'π': 'pi', 'σ': 'sigma', 'τ': 'tau', 'φ': 'phi', 'ω': 'omega',
}


def _clean(s):
    s = str(s).strip().lower().replace('$', '')
    # Chuyển ký tự Hy Lạp Unicode về tên tiếng Anh để khớp với LaTeX (\sigma -> sigma, σ -> sigma)
    for grk, eng in GREEK_MAP.items():
        s = s.replace(grk, eng)
    s = re.sub(r'\\(?:mathrm|mathit|text|mathbf)\{([^}]*)\}', r'\1', s)
    s = s.replace('\\', '').replace('{', '').replace('}', '')
    # Chuẩn hóa bố cục subplot: "1x2" hoặc "1 x 2" -> "1 by 2"
    s = re.sub(r'(\d)\s*(?:by|x)\s*(\d)', r'\1 by \2', s)
    # Chuẩn hóa dải số: "1 - 2" -> "1-2"
    s = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', s)
    # Chuẩn hóa khoảng trắng quanh dấu bằng: "t = 5" -> "t=5"
    s = re.sub(r'\s*=\s*', '=', s)
    # Bỏ tiền tố hình/mục phổ biến: "case 7" -> "7", "subplot (b)" -> "(b)"
    s = re.sub(r'^(?:case|subplot|option|figure|fig)\s+', '', s)
    # Bỏ hậu tố đơn vị/đếm phổ biến: "7 lines" -> "7", "2 points" -> "2"
    s = re.sub(r'\s+(?:lines?|subplots?|points?|data points?|datasets?|curves?|clusters?|obstacles?|times?)$', '', s)
    # Bỏ ký hiệu % nếu đi liền sau số
    s = re.sub(r'(\d)\s*%', r'\1', s)
    return re.sub(r'\s+', ' ', s).strip().rstrip('.')


WORD_TO_NUM = {
    'zero': 0.0, 'one': 1.0, 'once': 1.0, 'two': 2.0, 'twice': 2.0,
    'three': 3.0, 'thrice': 3.0, 'four': 4.0, 'five': 5.0, 'six': 6.0,
    'seven': 7.0, 'eight': 8.0, 'nine': 9.0, 'ten': 10.0,
    'eleven': 11.0, 'twelve': 12.0,
}


def _num(s):
    s_clean = s.replace(' ', '').lower().rstrip('.')
    if s_clean in WORD_TO_NUM:
        return WORD_TO_NUM[s_clean]
    m = re.fullmatch(r'10\^(-?\d+)', s_clean)
    if m:
        return 10.0 ** int(m.group(1))
    try:
        return float(s_clean.replace(',', ''))
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