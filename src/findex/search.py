import argparse
from functools import lru_cache
import heapq
from pathlib import Path
from typing import List, Tuple
from findex.decorators import timed
from findex.index import Index, open_index
from findex.query import parse_query
from findex.ranking import BM25Scorer, Scorer, SearchResult, TfIdfScorer, generate_snippet


class SearchEngine:
    def __init__(self, index: Index, scorer: Scorer | None = None):
        self.index = index
        self.scorer = scorer or BM25Scorer()

    @timed
    @lru_cache(maxsize=128)
    def search_cached(self, query_str: str, top_k: int = 5) -> Tuple[Tuple[str, float, str], ...]:
        results = self.search(query_str, top_k=top_k)
        return tuple((res.doc_id, res.score, res.snippet) for res in results)

    def search(self, query_str: str, top_k: int = 5) -> List[SearchResult]:
        ast = parse_query(query_str)
        if ast is None:
            return []

        doc_candidates = ast.evaluate(self.index)
        terms = ast.extract_terms()
        if not terms:
            terms = [t.casefold() for t in query_str.split() if t.upper() not in ("AND", "OR", "NOT")]

        scored_results = []
        for doc_id in doc_candidates:
            doc_meta = self.index.docs.get(doc_id)
            if not doc_meta:
                continue
            score = self.scorer.score(self.index, terms, doc_id, doc_meta.length)
            try:
                text = Path("data/corpus", doc_id).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            snippet = generate_snippet(text, terms)
            scored_results.append(SearchResult(score=score, doc_id=doc_id, snippet=snippet))

        return heapq.nlargest(top_k, scored_results)


def main():
    parser = argparse.ArgumentParser(description="Ранжований пошук по індексу findex (Lab 3)")
    parser.add_argument("index_file", type=str, help="Шлях до файлу індексу")
    parser.add_argument("query", type=str, help="Пошуковий запит")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25", help="Модель ранжування")
    parser.add_argument("--top", type=int, default=5, help="Кількість топ-результатів")
    args = parser.parse_args()

    scorer = BM25Scorer() if args.scorer == "bm25" else TfIdfScorer()

    with open_index(args.index_file) as index:
        engine = SearchEngine(index, scorer=scorer)
        print(f"--- Запит: '{args.query}' (Рушій: {args.scorer.upper()}) ---")

        print("\n[Перший виклик — холодний кеш]:")
        results = engine.search_cached(args.query, top_k=args.top)

        print("\n[Другий виклик — гарячий кеш / cache hit]:")
        _ = engine.search_cached(args.query, top_k=args.top)

        print(f"\nТоп-{args.top} результатів ранжування:")
        for rank, (doc_id, score, snippet) in enumerate(results, 1):
            print(f"{rank}. [Score: {score:.4f}] Doc: {doc_id}")
            print(f"   Snippet: {snippet}\n")


if __name__ == "__main__":
    main()