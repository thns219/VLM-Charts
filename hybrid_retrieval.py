# -*- coding: utf-8 -*-
"""hybrid_retrieval.py — Công cụ tìm kiếm ảnh lai (Hybrid Retrieval: Dense CLIP + Sparse BM25)
và Re-ranking cho bài toán VLM-Charts / CharXiv.

Tại sao Hybrid Retrieval giải quyết được nút thắt:
  1. Dense CLIP giỏi hiểu ngữ nghĩa trừu tượng (loại biểu đồ, bố cục màu),
     nhưng kém với từ khóa riêng lẻ (tên phương pháp "JeVois", "ResNet", ký hiệu "IoU").
  2. Sparse BM25 tìm chính xác 100% các từ khóa thực thể xuất hiện trong câu hỏi
     và caption / metadata của biểu đồ.
  3. Kết hợp chuẩn hóa min-max:
     Score(Chart) = alpha * DenseScore + (1 - alpha) * BM25Score
"""
import re
import math
from collections import Counter
import numpy as np


def tokenize(text):
    """Tách từ đơn giản, loại bỏ ký tự đặc biệt và đưa về chữ thường."""
    text = str(text).lower()
    # Giữ lại từ và số (ví dụ: "vitl14", "408mhz", "iou")
    tokens = re.findall(r'[a-z0-9_\-]+', text)
    return tokens


class BM25Retriever:
    """Bộ tìm kiếm từ khóa BM25 thuần Python (không cần thư viện ngoài)."""
    def __init__(self, corpus_texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_texts)
        self.tokenized_corpus = [tokenize(doc) for doc in corpus_texts]
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / (self.corpus_size or 1)
        
        self.doc_freqs = Counter()
        self.inverted_index = []
        for doc in self.tokenized_corpus:
            counts = Counter(doc)
            self.inverted_index.append(counts)
            for token in counts.keys():
                self.doc_freqs[token] += 1
                
        # Tính IDF chuẩn của Okapi BM25
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_text):
        query_tokens = tokenize(query_text)
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if not query_tokens or self.corpus_size == 0:
            return scores

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for doc_idx, doc_counts in enumerate(self.inverted_index):
                freq = doc_counts.get(token, 0)
                if freq == 0:
                    continue
                doc_len = self.doc_lens[doc_idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += idf_val * (freq * (self.k1 + 1.0)) / denom
        return scores


def min_max_normalize(scores):
    """Chuẩn hóa điểm số về khoảng [0, 1] để phối hợp điểm số đa phương thức."""
    arr = np.array(scores, dtype=np.float32)
    s_min, s_max = arr.min(), arr.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(arr)
    return (arr - s_min) / (s_max - s_min)


class HybridChartRetriever:
    """Bộ tìm kiếm lai kết hợp CLIP Embedding và BM25 Metadata."""
    def __init__(self, clip_embeddings, metadata_texts, alpha=0.5):
        """
        Args:
            clip_embeddings: ma trận numpy (N, D) chứa đặc trưng ảnh đã chuẩn hóa L2
            metadata_texts: danh sách N chuỗi (caption, tên bài báo, text OCR của từng ảnh)
            alpha: trọng số Dense CLIP (1.0 = chỉ dùng CLIP, 0.0 = chỉ dùng BM25, 0.5 = kết hợp cân bằng)
        """
        self.clip_embeddings = clip_embeddings
        self.bm25 = BM25Retriever(metadata_texts)
        self.alpha = alpha
        self.n_charts = len(metadata_texts)

    def retrieve(self, query_text, query_clip_embedding=None, top_k=5):
        """
        Trả về danh sách (chart_index, combined_score) được sắp xếp giảm dần.
        """
        # 1. Tính điểm Sparse (BM25)
        bm25_raw = self.bm25.get_scores(query_text)
        bm25_norm = min_max_normalize(bm25_raw)

        # 2. Tính điểm Dense (CLIP) nếu có embedding câu hỏi
        if query_clip_embedding is not None and self.clip_embeddings is not None:
            # query_clip_embedding: (D,), clip_embeddings: (N, D)
            q_emb = query_clip_embedding / (np.linalg.norm(query_clip_embedding) + 1e-9)
            dense_raw = np.dot(self.clip_embeddings, q_emb)
            dense_norm = min_max_normalize(dense_raw)
            combined_scores = self.alpha * dense_norm + (1.0 - self.alpha) * bm25_norm
        else:
            combined_scores = bm25_norm

        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        return [(int(idx), float(combined_scores[idx])) for idx in top_indices]


def rerank_with_vlm_verifier(vlm_ask_fn, candidate_indices, val_ds, question):
    """
    Xếp hạng lại (Re-rank) danh sách Top-K bằng một câu hỏi xác thực siêu ngắn.
    Hỏi VLM: Biểu đồ này có chứa thông tin trả lời câu hỏi X không? (Yes/No).
    """
    for chart_idx in candidate_indices:
        image = val_ds[chart_idx]["image"]
        verification_prompt = (
            f"Question to answer: '{question}'.\n"
            "Does this chart contain the relevant variables, labels, or data to answer this question?\n"
            "Answer strictly with either 'Yes' or 'No'."
        )
        answer = vlm_ask_fn(image, verification_prompt).strip().lower()
        if "yes" in answer:
            return chart_idx
    # Nếu không ảnh nào được xác nhận tuyệt đối, giữ nguyên ứng viên số 1
    return candidate_indices[0] if candidate_indices else None
