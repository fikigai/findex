import argparse
import time
import tracemalloc
from typing import List
from findex.index import DocMeta, InvertedIndex, Posting  # <--- Додано DocMeta сюди


class SearchEngine:

    @staticmethod
    def search_merge(index: InvertedIndex, terms: List[str]) -> List[str]:
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
    def _intersect(doc_ids: List[str], postings: List[Posting]) -> List[str]:
        i, j = 0, 0
        res = []
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
    def search_set(index: InvertedIndex, terms: List[str]) -> List[str]:
        if not terms:
            return []
        sets = [
            set(p.doc_id for p in index.index.get(term, [])) for term in terms
        ]
        if any(not s for s in sets):
            return []
        return sorted(list(sets[0].intersection(*sets[1:])))


def main():
    parser = argparse.ArgumentParser(description="Пошук по індексу findex")
    parser.add_argument("index_file", type=str, help="Шлях до файлу індексу")
    parser.add_argument("query", type=str, help="Пошуковий запит (терми через пробіл)")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Рушій пошуку")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.time()

    index = InvertedIndex.load_pickle(args.index_file)
    terms = args.query.lower().split()

    if args.engine == "merge":
        results = SearchEngine.search_merge(index, terms)
    else:
        results = SearchEngine.search_set(index, terms)

    elapsed = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Знайдено документів ({args.engine}): {results}")
    print(f"Кількість результатів: {len(results)}")
    print(f"Час пошуку: {elapsed:.6f} сек")
    print(f"Пікова пам'ять: {peak_mem / (1024 * 1024):.4f} MB")


if __name__ == "__main__":
    main()