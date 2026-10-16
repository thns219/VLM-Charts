# -*- coding: utf-8 -*-
"""rescore_and_plot.py — chấm lại 3 file kết quả bằng match_fixed + vẽ 5 biểu đồ.

Cách dùng (Windows): đặt file này CÙNG THƯ MỤC với 4 file CSV và charxiv_match.py, rồi chạy
    python rescore_and_plot.py
Biểu đồ và bảng tổng hợp được lưu vào thư mục con 'figures'.

Cần có trong thư mục:
    baseline_results.csv, baseline_qwen_results.csv, rag_vqa_results.csv,
    retrieval_eval_results.csv, charxiv_match.py
Muốn đổi nơi đọc/ghi thì sửa DATA_DIR / OUT_DIR bên dưới.
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')                                   # không cần cửa sổ hiển thị, chỉ lưu file
import matplotlib.pyplot as plt

from charxiv_match import match_fixed

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = HERE                                  # nơi chứa các file CSV
OUT_DIR = os.path.join(HERE, 'figures')          # nơi lưu hình
os.makedirs(OUT_DIR, exist_ok=True)


def P(folder, name):
    return os.path.join(folder, name)


def load(name):
    path = P(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không thấy '{path}'. Copy file vào cùng thư mục với script hoặc sửa DATA_DIR.")
    d = pd.read_csv(path, keep_default_na=False)          # giữ "" là "", không thành NaN
    d['correct'] = [match_fixed(p, g_) for p, g_ in zip(d.prediction, d.gold_answer)]
    d['key'] = list(zip(d.chart_index, d.slot))
    if 'retrieval_correct' in d:
        d['retrieval_correct'] = d.retrieval_correct.astype(str) == 'True'
    return d


gem = load('baseline_results.csv')
qw = load('baseline_qwen_results.csv')
rag = load('rag_vqa_results.csv')
h = pd.read_csv(P(DATA_DIR, 'retrieval_eval_results.csv'))
O = OUT_DIR

# Tập câu hỏi chung: chỉ giữ câu Gemini có phản hồi (dự đoán rỗng = lỗi API, không phải câu trả lời).
valid = set(gem[gem.prediction.astype(str).str.strip() != ''].key)
N_VALID = len(valid)
N_CHART = gem.chart_index.nunique()
N_POOL = rag.chart_index.nunique()                       # số ảnh ứng viên khi truy hồi
fmt_n = lambda n: f'{n:,}'.replace(',', '.')
print(f'Tập chung: {fmt_n(N_VALID)} câu hợp lệ trên {N_CHART} biểu đồ.')

C = {'Gemini': '#2F5BEA', 'Qwen': '#14B8A6', 'RAG': '#F0603A'}
NAMES = {'Gemini': 'Gemini\n(baseline, oracle)', 'Qwen': 'Qwen2.5-VL-7B\n(baseline, oracle)', 'RAG': 'RAG-VQA\n(CLIP + Qwen)'}
INK = '#1F2937'; MUTE = '#6B7280'; GRID = '#E5E7EB'
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': MUTE, 'axes.edgecolor': GRID, 'figure.dpi': 100, 'savefig.dpi': 300,
    'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False})
D = {k: v[v.key.isin(valid)] for k, v in {'Gemini': gem, 'Qwen': qw, 'RAG': rag}.items()}   # tập chung cho Fig 1, 2, 5
rng = np.random.default_rng(42)
keys = ['Gemini', 'Qwen', 'RAG']
w = 0.26


def boot_ci(df, n=2000):
    ch = df.groupby('chart_index').correct.agg(['sum', 'count'])
    s = ch['sum'].values; c = ch['count'].values; k = len(ch)
    idx = rng.integers(0, k, (n, k)); v = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(v, [2.5, 97.5]) * 100


def style(ax, ymax=100, step=20):
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(axis='both', length=0)
    t = list(range(0, int(ymax) + 1, step)); ax.set_yticks(t); ax.set_yticklabels([f'{x}%' for x in t]); ax.set_ylim(0, ymax * 1.04)


def nice_ymax(v, step, headroom=1.25):
    return max(step, int(np.ceil(v * headroom / step) * step))


def title(fig, t, sub=None):
    fig.text(0.06, 0.955, t, fontsize=17, fontweight='bold', ha='left', va='top')
    if sub:
        fig.text(0.06, 0.905, sub, fontsize=11, color=MUTE, ha='left', va='top')


def label(ax, x, y, txt, color=INK, dy=1.5, size=12, bold=True):
    ax.text(x, y + dy, txt, ha='center', va='bottom', fontsize=size, fontweight='bold' if bold else 'normal', color=color)


# ---------- Số liệu dùng chung cho Fig 2 và bảng tổng hợp ----------
groups = [('Mô tả\n(trả lời được, D1–D3)', lambda d: d[d.slot.isin(['descriptive_1', 'descriptive_2', 'descriptive_3'])]),
          ('Suy luận\n(reasoning)', lambda d: d[d.slot == 'reasoning']),
          ('Không thể trả lời\n(đáp án: Not Applicable)', lambda d: d[d.slot == 'descriptive_4'])]
acc_group = {k: [f(D[k]).correct.mean() * 100 for _, f in groups] for k in keys}

# ---------- Bảng tổng hợp (in ra + lưu CSV để đưa vào báo cáo) ----------
rows = []
for k in keys:
    lo, hi = boot_ci(D[k])
    rows.append({'Mô hình': k, 'n câu': len(D[k]), 'Tổng (%)': round(D[k].correct.mean() * 100, 1),
                 'KTC95 thấp': round(lo, 1), 'KTC95 cao': round(hi, 1),
                 'Mô tả D1-D3 (%)': round(acc_group[k][0], 1), 'Suy luận (%)': round(acc_group[k][1], 1),
                 'Không thể trả lời (%)': round(acc_group[k][2], 1)})
summary = pd.DataFrame(rows)
print('\n=== BẢNG TỔNG HỢP (tập chung) ===\n' + summary.to_string(index=False))
summary.to_csv(P(O, 'summary_results.csv'), index=False, encoding='utf-8-sig')
sub_tab = pd.DataFrame({k: D[k][D[k].slot == 'reasoning'].groupby('sub_type').correct.mean() * 100 for k in keys}).round(1)
print('\n=== SUY LUẬN THEO KIỂU ĐÁP ÁN (%) ===\n' + sub_tab.to_string())
sub_tab.to_csv(P(O, 'summary_reasoning_subtype.csv'), encoding='utf-8-sig')

# ---------- Fig 1: overall ----------
fig, ax = plt.subplots(figsize=(9, 6.2)); fig.subplots_adjust(top=0.80, bottom=0.16, left=0.09, right=0.97)
title(fig, 'Độ chính xác tổng thể của 3 mô hình',
      f'{N_CHART} biểu đồ CharXiv (validation) · {fmt_n(N_VALID)} câu hỏi có phản hồi hợp lệ · vạch đen = KTC 95% (bootstrap)')
acc = []
for i, k in enumerate(keys):
    d = D[k]; a = d.correct.mean() * 100; lo, hi = boot_ci(d); acc.append(a)
    ax.bar(i, a, width=0.46, color=C[k], zorder=3)
    ax.errorbar(i, a, yerr=[[a - lo], [hi - a]], color=INK, capsize=5, lw=1.6, zorder=4)
    label(ax, i, hi, f'{a:.1f}%', C[k], dy=1.2, size=15)
ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[k] for k in keys], fontsize=11)
style(ax)
for i, (a0, a1) in enumerate([(acc[0], acc[1]), (acc[1], acc[2])]):
    ax.annotate('', xy=(i + 0.76, a1 + 3), xytext=(i + 0.24, a0 - 3),
                arrowprops=dict(arrowstyle='-|>', color=MUTE, lw=1.3, connectionstyle='arc3,rad=-0.1'), zorder=5)
    ax.text(i + 0.5, (a0 + a1) / 2 + 11, f'−{abs(a1 - a0):.1f} điểm %', ha='center', fontsize=10, color=MUTE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=GRID), zorder=6)
fig.savefig(P(O, 'fig1_do_chinh_xac_tong_the.png')); plt.close(fig)

# ---------- Fig 2: by task ----------
fig, ax = plt.subplots(figsize=(11, 6.4)); fig.subplots_adjust(top=0.80, bottom=0.24, left=0.07, right=0.98)
title(fig, 'Độ chính xác theo loại câu hỏi',
      f'Cùng một tập câu hỏi cho cả 3 mô hình. Ở câu "không thể trả lời", Qwen bịa đáp án ở ~{100 - acc_group["Qwen"][2]:.0f}% '
      'trường hợp thay vì trả về Not Applicable')
for j, k in enumerate(keys):
    for i, (nm, f) in enumerate(groups):
        v = acc_group[k][i]; x = i + (j - 1) * w
        ax.bar(x, v, width=w * 0.92, color=C[k], zorder=3, label=NAMES[k].replace('\n', ' ') if i == 0 else None)
        label(ax, x, v, f'{v:.1f}' + ('*' if (k == 'RAG' and i == 2) else ''), C[k], size=11)
ax.set_xticks(range(3)); ax.set_xticklabels([g_[0] for g_ in groups], fontsize=11); style(ax)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=10.5)
fig.text(0.07, 0.025, f'* Điểm {acc_group["RAG"][2]:.0f}% của RAG ở nhóm "không thể trả lời" chủ yếu do truy hồi sai ảnh → '
         'mô hình trả "Not Applicable" (chưa phải khả năng hiệu chỉnh thật).', fontsize=9, color=MUTE)
fig.savefig(P(O, 'fig2_theo_loai_cau_hoi.png')); plt.close(fig)

# ---------- Fig 3: retrieval Hit@k ----------
ks = [1, 3, 5, 10]; hv = [h[f'hit@{k}'].mean() * 100 for k in ks]; rnd = [k / N_POOL * 100 for k in ks]
fig, ax = plt.subplots(figsize=(9, 6.2)); fig.subplots_adjust(top=0.80, bottom=0.13, left=0.09, right=0.97)
title(fig, 'Chất lượng truy hồi (CLIP ViT-B/32 + ChromaDB)',
      f'Hit@k trên {len(h)} câu hỏi suy luận · {N_POOL} ảnh ứng viên · so với truy hồi ngẫu nhiên')
x = np.arange(4)
ax.bar(x - 0.19, hv, 0.36, color=C['RAG'], zorder=3, label='CLIP retrieval')
ax.bar(x + 0.19, rnd, 0.36, color='#CBD5E1', zorder=3, label=f'Ngẫu nhiên (k/{N_POOL})')
for i in range(4):
    label(ax, i - 0.19, hv[i], f'{hv[i]:.1f}%', C['RAG'], size=12)
    label(ax, i + 0.19, rnd[i], f'{rnd[i]:.1f}%', MUTE, size=10, bold=False)
ymax3 = nice_ymax(max(hv), 10)
ax.set_xticks(x); ax.set_xticklabels([f'Hit@{k}' for k in ks], fontsize=12); style(ax, ymax3, 10)
ax.legend(frameon=False, loc='upper left', fontsize=11)
ax.text(3.45, ymax3 * 0.92, f'Hit@10 chỉ ~{hv[3]:.0f}%: ngay cả top-10\nvẫn bỏ sót ~{round((100 - hv[3]) / 10)}/10 ảnh đúng',
        ha='right', va='top', fontsize=10.5, color=MUTE)
fig.savefig(P(O, 'fig3_truy_hoi_hit_at_k.png')); plt.close(fig)

# ---------- Fig 4: RAG diagnosis ----------
rr = rag[rag.question_type == 'reasoning']
ok = rr[rr.retrieval_correct]; bad = rr[~rr.retrieval_correct]
q_ok = qw[(qw.slot == 'reasoning') & qw.chart_index.isin(ok.chart_index)].correct.mean() * 100
rag_ok, rag_bad = ok.correct.mean() * 100, bad.correct.mean() * 100
n_desc = int((rag.question_type == 'descriptive').sum())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={'width_ratios': [1, 1.25]})
fig.subplots_adjust(top=0.78, bottom=0.17, left=0.06, right=0.98, wspace=0.22)
title(fig, 'Vì sao RAG-VQA tụt điểm? Nút thắt nằm ở bước truy hồi',
      f'Truy hồi đúng: RAG {rag_ok:.1f}% so với Qwen oracle {q_ok:.1f}% · truy hồi sai: RAG chỉ còn {rag_bad:.1f}%')
rt = [rag[rag.question_type == 'descriptive'].retrieval_correct.mean() * 100, rr.retrieval_correct.mean() * 100]
a1.bar([0, 1], rt, 0.55, color=['#FCA58C', C['RAG']], zorder=3)
for i, v in enumerate(rt):
    label(a1, i, v, f'{v:.1f}%', C['RAG'], size=14)
a1.set_xticks([0, 1])
a1.set_xticklabels([f'Câu hỏi mô tả\n({fmt_n(n_desc)} câu · chỉ 19 mẫu\ncâu hỏi lặp lại)', f'Câu hỏi suy luận\n({len(rr)} câu)'], fontsize=10.5)
style(a1, nice_ymax(max(rt), 10), 10)
a1.set_title('(a) Tỉ lệ truy hồi đúng ảnh', loc='left', fontsize=12.5, fontweight='bold', pad=12)
vals = [rag_bad, rag_ok, q_ok]
a2.bar(range(3), vals, 0.58, color=['#F8B4A0', C['RAG'], C['Qwen']], zorder=3)
for i, v in enumerate(vals):
    label(a2, i, v, f'{v:.1f}%', [C['RAG'], C['RAG'], C['Qwen']][i], size=14)
a2.set_xticks(range(3))
a2.set_xticklabels([f'RAG · truy hồi SAI\n(n={len(bad)})', f'RAG · truy hồi ĐÚNG\n(n={len(ok)})', f'Qwen oracle\n(cùng {len(ok)} biểu đồ)'], fontsize=10.5)
style(a2, nice_ymax(max(vals), 20), 20)
a2.set_title('(b) Độ chính xác câu hỏi suy luận, tách theo kết quả truy hồi', loc='left', fontsize=12.5, fontweight='bold', pad=12)
fig.savefig(P(O, 'fig4_chan_doan_rag.png')); plt.close(fig)

# ---------- Fig 5: reasoning by answer type ----------
st = ['Text-in-Chart', 'Text-in-General', 'Number-in-Chart', 'Number-in-General']
lab = {'Text-in-Chart': 'Văn bản\ntrong biểu đồ', 'Text-in-General': 'Văn bản\nsuy diễn chung',
       'Number-in-Chart': 'Số\ntrong biểu đồ', 'Number-in-General': 'Số\nsuy diễn chung'}
lab1 = {s: lab[s].replace('\n', ' ') for s in st}
cnt = D['Gemini'][D['Gemini'].slot == 'reasoning'].sub_type.value_counts()
qacc = {s: D['Qwen'][(D['Qwen'].slot == 'reasoning') & (D['Qwen'].sub_type == s)].correct.mean() * 100 for s in st}
rag_r = D['RAG'][D['RAG'].slot == 'reasoning']
worst_q = min(qacc, key=qacc.get)
fig, ax = plt.subplots(figsize=(11, 6.4)); fig.subplots_adjust(top=0.80, bottom=0.17, left=0.07, right=0.98)
title(fig, 'Câu hỏi suy luận theo kiểu đáp án',
      f'Qwen thấp nhất ở nhóm "{lab1[worst_q]}" ({qacc[worst_q]:.0f}%); RAG thấp ở mọi nhóm vì thiếu ảnh đúng '
      f'(cao nhất {max(rag_r[rag_r.sub_type == s].correct.mean() * 100 for s in st):.0f}%)')
for j, k in enumerate(keys):
    d = D[k]; d = d[d.slot == 'reasoning']
    for i, s in enumerate(st):
        v = d[d.sub_type == s].correct.mean() * 100; x = i + (j - 1) * w
        ax.bar(x, v, width=w * 0.92, color=C[k], zorder=3, label=NAMES[k].replace('\n', ' ') if i == 0 else None)
        label(ax, x, v, f'{v:.0f}', C[k], size=11)
ax.set_xticks(range(4)); ax.set_xticklabels([f'{lab[s]}\n(n={cnt.get(s, 0)})' for s in st], fontsize=10.5); style(ax)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=10.5)
fig.savefig(P(O, 'fig5_suy_luan_theo_kieu_dap_an.png')); plt.close(fig)

print(f"\nok — hình và bảng tổng hợp đã lưu vào: {O}")