import heapq
import json
import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal

from findex.decorators import timed
from findex.index import Index, NumpyIndexView, Posting
from findex.query import parse_query
from findex.ranking import BM25Scorer, Scorer, SearchResult, TFIDFScorer, generate_snippet
from findex.vector import vector_search

logger = logging.getLogger("findex")

EngineName = Literal["merge", "set", "numpy"]


class SearchEngine:
    def __init__(
        self,
        index: Index,
        scorer: Scorer | None = None,
        corpus_dir: str | Path = "data/corpus",
    ) -> None:
        self.index = index
        self.scorer: Scorer = scorer if scorer is not None else BM25Scorer()
        self.corpus_dir = Path(corpus_dir)
        self._numpy_view: NumpyIndexView | None = None
        self._cached_lookup = lru_cache(maxsize=128)(self._execute_cached)

    @property
    def numpy_view(self) -> NumpyIndexView:
        if self._numpy_view is None:
            self._numpy_view = NumpyIndexView.from_index(self.index)
        return self._numpy_view

    def load_doc_text(self, doc_id: str) -> str:
        if self.corpus_dir.is_file() and self.corpus_dir.suffix == ".jsonl":
            with open(self.corpus_dir, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("url") == doc_id or rec.get("doc_id") == doc_id:
                        return f"{rec.get('title', '')} {rec.get('text', '')}".strip()
            return ""

        doc_file = self.corpus_dir / doc_id
        if doc_file.exists() and doc_file.is_file():
            return doc_file.read_text(encoding="utf-8", errors="ignore")
        return ""

    @staticmethod
    def search_merge(index: Index, terms: Sequence[str]) -> list[str]:
        """Двовказівниковий Boolean AND-перетин за відсортованими списками постінгів."""
        if not terms:
            return []
        postings_lists = [index.index.get(term, []) for term in terms]
        if any(not p_list for p_list in postings_lists):
            return []

        result_docs = [p.doc_id for p in postings_lists[0]]
        for other_list in postings_lists[1:]:
            result_docs = SearchEngine._intersect(result_docs, other_list)
            if not result_docs:
                break
        return result_docs

    @staticmethod
    def _intersect(doc_ids: Sequence[str], postings: Sequence[Posting]) -> list[str]:
        i, j = 0, 0
        res: list[str] = []
        p_len = len(postings)
        while i < len(doc_ids) and j < p_len:
            d_id = doc_ids[i]
            p_id = postings[j].doc_id
            if d_id == p_id:
                res.append(d_id)
                i += 1
                j += 1
            elif d_id < p_id:
                i += 1
            else:
                j += 1
        return res

    @staticmethod
    def search_set(index: Index, terms: Sequence[str]) -> list[str]:
        """Boolean AND-перетин через операції над множинами set."""
        if not terms:
            return []
        sets = [{p.doc_id for p in index.index.get(term, [])} for term in terms]
        if any(not s for s in sets):
            return []
        return sorted(sets[0].intersection(*sets[1:]))

    @timed
    def search_cached(
        self, query_str: str, top_k: int = 5
    ) -> tuple[tuple[str, float, str], ...]:
        """Кешований пошук через внутрішній lru_cache."""
        return self._cached_lookup(query_str, top_k)

    def _execute_cached(
        self, query_str: str, top_k: int
    ) -> tuple[tuple[str, float, str], ...]:
        results = self.search(query_str, top_k=top_k)
        return tuple((res.doc_id, res.score, res.snippet) for res in results)

    def search(
        self,
        query_str: str,
        top_k: int = 5,
        engine: EngineName = "numpy",
    ) -> list[SearchResult]:
        ast = parse_query(query_str)
        if ast is None:
            return []

        terms = ast.extract_terms()
        is_simple_and = all(
            tok not in ("OR", "NOT") and not tok.startswith('"')
            for tok in query_str.split()
        )
        if is_simple_and and terms and engine == "numpy" and self.corpus_dir.is_dir():
            scorer_name: Literal["bm25", "tfidf"] = (
                "tfidf" if isinstance(self.scorer, TFIDFScorer) else "bm25"
            )
            return vector_search(
                self.index,
                self.numpy_view,
                terms,
                self.corpus_dir,
                scorer_name=scorer_name,
                top_k=top_k,
            )

        if is_simple_and and terms:
            if engine == "merge":
                doc_candidates = set(self.search_merge(self.index, terms))
            else:
                doc_candidates = set(self.search_set(self.index, terms))
        else:
            doc_candidates = ast.evaluate(self.index)

        if not terms:
            terms = [
                t.casefold()
                for t in query_str.split()
                if t.upper() not in ("AND", "OR", "NOT")
            ]

        scored_results: list[SearchResult] = []
        for doc_id in doc_candidates:
            doc_meta = self.index.docs.get(doc_id)
            if doc_meta is None:
                continue
            score = self.scorer.score(self.index, terms, doc_id, doc_meta.length)
            text = self.load_doc_text(doc_id)
            snippet = generate_snippet(text, terms)
            scored_results.append(SearchResult(score=score, doc_id=doc_id, snippet=snippet))

        logger.debug("Знайдено %d кандидатів для '%s'", len(scored_results), query_str)
        return heapq.nlargest(top_k, scored_results)