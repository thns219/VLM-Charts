# -*- coding: utf-8 -*-
"""charxiv_match.py — hàm chấm match_fixed dùng chung (copy nguyên từ rescore_and_plot.py).
Import ở mọi script:  from charxiv_match import match_fixed"""
import re

NA = "not applicable"


def _clean(s):
    s = str(s).strip().lower().replace('$', '')
    s = re.sub(r'\\(?:mathrm|mathit|text|mathbf)\{([^}]*)\}', r'\1', s)
    s = s.replace('\\', '').replace('{', '').replace('}', '')
    return re.sub(r'\s+', ' ', s).strip().rstrip('.')


def _num(s):
    s = s.replace(' ', '')
    m = re.fullmatch(r'10\^(-?\d+)', s)
    if m:
        return 10.0 ** int(m.group(1))
    try:
        return float(s.replace(',', ''))
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
