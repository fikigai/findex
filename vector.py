from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from findex.corpus import iter_documents
from findex.index import Index, NumpyIndexView
from findex.ranking import SearchResult, generate_snippet
from findex.tokenizer import tokenize

SearchMode = Literal["lexical", "semantic", "hybrid"]

# Семантичні кластери для легкого вбудованого енкодера (коли sentence-transformers не встановлено)
_CONCEPT_CLUSTERS: tuple[tuple[str, ...], ...] = (
    ("python", "asyncio", "coroutine", "coro", "await", "event", "loop", "async", "nonblocking", "concurrent"),
    ("index", "inverted", "posting", "postings", "vocabulary", "dictionary", "token", "tokens", "lexicon"),
    ("rank", "ranking", "bm25", "tfidf", "tf", "idf", "score", "relevance", "weight", "retrieval"),
    ("crawl", "crawler", "spider", "http", "https", "url", "robots", "fetch", "scrape", "html", "link"),
    ("web", "api", "fastapi", "rest", "endpoint", "server", "uvicorn", "asgi", "json", "http", "route"),
    ("docker", "container", "image", "deploy", "deployment", "cloud", "render", "service", "production"),
    ("parallel", "multiprocess", "multiprocessing", "process", "thread", "threading", "worker", "gil", "cpu"),
    ("vector", "semantic", "embedding", "embeddings", "cosine", "similarity", "matrix", "numpy", "dense"),
)


def _topk_partition(
    doc_ids: NDArray[np.int32],
    scores: NDArray[np.float64],
    idx_to_doc_id: tuple[str, ...],
    top_k: int,
) -> list[tuple[str, float]]:
    """Відбирає top-k результатів за O(N) через np.argpartition із детермінованим tie-breaking."""
    n = len(scores)
    if n == 0 or top_k <= 0:
        return []

    k = min(top_k, n)
    if k < n:
        part_idx = np.argpartition(-scores, k - 1)[:k]
    else:
        part_idx = np.arange(n)

    # Для однакових score зберігаємо лексикографічний порядок doc_id (як у heapq.nlargest у Лабі 3)
    candidates = [
        (float(scores[i]), idx_to_doc_id[int(doc_ids[i])])
        for i in part_idx
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    # Якщо є нічиї на межі top_k, перевіряємо всі елементи з таким самим score
    if k < n:
        min_kept_score = candidates[-1][0]
        tied_mask = np.isclose(scores, min_kept_score, rtol=1e-12, atol=1e-12)
        if int(np.sum(tied_mask)) > 1:
            all_tied = [
                (float(scores[i]), idx_to_doc_id[int(doc_ids[i])])
                for i in np.flatnonzero(scores >= min_kept_score - 1e-12)
            ]
            all_tied.sort(key=lambda item: (-item[0], item[1]))
            candidates = all_tied[:k]

    return [(doc_str, sc) for sc, doc_str in candidates]


def vector_score_bm25(
    view: NumpyIndexView,
    terms: Sequence[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Векторизований підрахунок BM25 над масивами NumPy int32."""
    unique_terms = [t for t in dict.fromkeys(terms) if t in view.term_doc_ids]
    if not unique_terms:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    # Перетин відсортованих int32 масивів doc_ids (Boolean AND)
    common_docs = view.term_doc_ids[unique_terms[0]]
    for term in unique_terms[1:]:
        common_docs = np.intersect1d(common_docs, view.term_doc_ids[term], assume_unique=True)
        if len(common_docs) == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    n_docs = len(view.idx_to_doc_id)
    avgdl = view.avg_doc_length if view.avg_doc_length > 0.0 else 1.0
    dl = view.doc_lengths[common_docs].astype(np.float64)
    norm_factor = k1 * (1.0 - b + b * (dl / avgdl))

    total_scores = np.zeros(len(common_docs), dtype=np.float64)
    for term in unique_terms:
        t_docs = view.term_doc_ids[term]
        t_freqs = view.term_freqs[term]
        df = len(t_docs)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        pos = np.searchsorted(t_docs, common_docs)
        tf = t_freqs[pos].astype(np.float64)
        total_scores += idf * ((tf * (k1 + 1.0)) / (tf + norm_factor))

    return common_docs, total_scores


def vector_score_tfidf(
    view: NumpyIndexView,
    terms: Sequence[str],
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Векторизований підрахунок TF-IDF над масивами NumPy int32."""
    unique_terms = [t for t in dict.fromkeys(terms) if t in view.term_doc_ids]
    if not unique_terms:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    common_docs = view.term_doc_ids[unique_terms[0]]
    for term in unique_terms[1:]:
        common_docs = np.intersect1d(common_docs, view.term_doc_ids[term], assume_unique=True)
        if len(common_docs) == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

    n_docs = len(view.idx_to_doc_id)
    dl = np.maximum(view.doc_lengths[common_docs].astype(np.float64), 1.0)
    total_scores = np.zeros(len(common_docs), dtype=np.float64)

    for term in unique_terms:
        t_docs = view.term_doc_ids[term]
        t_freqs = view.term_freqs[term]
        df = len(t_docs)
        idf = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        pos = np.searchsorted(t_docs, common_docs)
        tf = t_freqs[pos].astype(np.float64) / dl
        total_scores += tf * idf

    return common_docs, total_scores


def vector_search(
    index: Index,
    view: NumpyIndexView,
    terms: Sequence[str],
    corpus_dir: Path,
    scorer_name: Literal["bm25", "tfidf"] = "bm25",
    top_k: int = 5,
) -> list[SearchResult]:
    """Виконує векторизований пошук та відбір top-k через np.argpartition."""
    if scorer_name == "tfidf":
        doc_ids, scores = vector_score_tfidf(view, terms)
    else:
        doc_ids, scores = vector_score_bm25(view, terms)

    ranked = _topk_partition(doc_ids, scores, view.idx_to_doc_id, top_k=top_k)
    results: list[SearchResult] = []
    for doc_id, score in ranked:
        doc_file = corpus_dir / doc_id
        text = doc_file.read_text(encoding="utf-8", errors="ignore") if doc_file.is_file() else ""
        snippet = generate_snippet(text, terms)
        results.append(SearchResult(score=score, doc_id=doc_id, snippet=snippet))
    return results


class SemanticEncoder:
    """L2-нормалізований енкодер тексту (all-MiniLM-L6-v2 або вбудований концепт-векторизатор)."""

    def __init__(self, dim: int = 256, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.dim = dim
        self.model_name = model_name
        self._st_model: object | None = None
        self._token_to_cluster: dict[str, int] = {}
        for cid, words in enumerate(_CONCEPT_CLUSTERS):
            for w in words:
                self._token_to_cluster[w] = cid

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self._st_model = SentenceTransformer(model_name)
        except Exception:  # noqa: BLE001
            self._st_model = None

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self._st_model is not None:
            raw = self._st_model.encode(list(texts), normalize_embeddings=True)  # type: ignore[attr-defined]
            return np.asarray(raw, dtype=np.float32)

        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        n_clusters = len(_CONCEPT_CLUSTERS)
        for row_idx, text in enumerate(texts):
            tokens = list(tokenize(text))
            for tok in tokens:
                # 1. Концепт-кластер (забезпечує близькість синонімів без спільних слів)
                cid = self._token_to_cluster.get(tok)
                if cid is not None:
                    mat[row_idx, cid] += 2.5
                # 2. Символьні триграми та хеш токена у решті вимірів
                grams = [tok] + [tok[i : i + 3] for i in range(max(1, len(tok) - 2))]
                for g in grams:
                    digest = hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest()
                    bucket = n_clusters + (int.from_bytes(digest, "little") % (self.dim - n_clusters))
                    mat[row_idx, bucket] += 1.0

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        return (mat / norms).astype(np.float32)


def chunk_document(text: str, chunk_words: int = 80, overlap: int = 20) -> list[str]:
    """Ділить текст документа на фрагменти (чанки) із перекриттям."""
    words = re.findall(r"\S+", text)
    if not words:
        return [""]
    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(1, chunk_words - overlap)
    for start in range(0, len(words), step):
        part = words[start : start + chunk_words]
        chunks.append(" ".join(part))
        if start + chunk_words >= len(words):
            break
    return chunks


@dataclass(slots=True)
class SemanticIndex:
    """Матриця нормалізованих ембедингів (.npy) та метадані фрагментів документів."""

    embeddings: NDArray[np.float32]
    chunk_doc_ids: list[str]
    chunk_texts: list[str]
    encoder: SemanticEncoder

    @classmethod
    def build_from_corpus(
        cls,
        corpus_dir: str | Path,
        chunk_words: int = 80,
        overlap: int = 20,
        encoder: SemanticEncoder | None = None,
    ) -> "SemanticIndex":
        enc = encoder if encoder is not None else SemanticEncoder()
        chunk_doc_ids: list[str] = []
        chunk_texts: list[str] = []

        for doc_id, text in iter_documents(corpus_dir):
            for chunk in chunk_document(text, chunk_words=chunk_words, overlap=overlap):
                chunk_doc_ids.append(doc_id)
                chunk_texts.append(chunk)

        matrix = enc.encode(chunk_texts)
        return cls(
            embeddings=matrix,
            chunk_doc_ids=chunk_doc_ids,
            chunk_texts=chunk_texts,
            encoder=enc,
        )

    def save(self, npy_path: str | Path) -> None:
        out_npy = Path(npy_path)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, self.embeddings)
        meta_path = out_npy.with_suffix(".meta.json")
        meta = {
            "dim": int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else self.encoder.dim,
            "chunk_doc_ids": self.chunk_doc_ids,
            "chunk_texts": self.chunk_texts,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, npy_path: str | Path) -> "SemanticIndex":
        in_npy = Path(npy_path)
        meta_path = in_npy.with_suffix(".meta.json")
        if not in_npy.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Файли ембедингів не знайдено: {in_npy}")
        matrix = np.load(in_npy).astype(np.float32)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dim = int(meta.get("dim", matrix.shape[1] if matrix.ndim == 2 else 256))
        return cls(
            embeddings=matrix,
            chunk_doc_ids=list(meta["chunk_doc_ids"]),
            chunk_texts=list(meta["chunk_texts"]),
            encoder=SemanticEncoder(dim=dim),
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Множить матрицю ембедингів на вектор запиту та зараховує документу найкращий фрагмент."""
        if len(self.chunk_doc_ids) == 0:
            return []

        q_vec = self.encoder.encode([query])[0]
        sims = self.embeddings @ q_vec

        best_per_doc: dict[str, tuple[float, str]] = {}
        for i, doc_id in enumerate(self.chunk_doc_ids):
            score = float(sims[i])
            prev = best_per_doc.get(doc_id)
            if prev is None or score > prev[0]:
                best_per_doc[doc_id] = (score, self.chunk_texts[i])

        ranked = sorted(
            best_per_doc.items(),
            key=lambda kv: (-kv[1][0], kv[0]),
        )[:top_k]

        terms = list(tokenize(query))
        results: list[SearchResult] = []
        for doc_id, (score, best_chunk) in ranked:
            snippet = generate_snippet(best_chunk, terms) if terms else best_chunk[:160]
            results.append(SearchResult(score=score, doc_id=doc_id, snippet=snippet))
        return results


def reciprocal_rank_fusion(
    lexical_results: Sequence[SearchResult],
    semantic_results: Sequence[SearchResult],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Зливає лексичний (BM25) та семантичний рейтинги через Reciprocal Rank Fusion (RRF)."""
    rrf_scores: dict[str, float] = {}
    snippets: dict[str, str] = {}

    for rank, hit in enumerate(lexical_results, start=1):
        rrf_scores[hit.doc_id] = rrf_scores.get(hit.doc_id, 0.0) + 1.0 / (rrf_k + rank)
        snippets[hit.doc_id] = hit.snippet

    for rank, hit in enumerate(semantic_results, start=1):
        rrf_scores[hit.doc_id] = rrf_scores.get(hit.doc_id, 0.0) + 1.0 / (rrf_k + rank)
        if hit.doc_id not in snippets:
            snippets[hit.doc_id] = hit.snippet

    merged = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [
        SearchResult(score=score, doc_id=doc_id, snippet=snippets.get(doc_id, ""))
        for doc_id, score in merged
    ]