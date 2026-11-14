import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import pickle
import time
import tracemalloc
from typing import Dict, List
from findex.corpus import iter_documents
from findex.tokenizer import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: str
    freq: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: str
    length: int


class InvertedIndex:

    def __init__(self):
        self.index: Dict[str, List[Posting]] = {}
        self.docs: Dict[str, DocMeta] = {}

    @classmethod
    def build_from_corpus(
        cls, corpus_dir: str | Path, limit: int | None = None
    ) -> "InvertedIndex":
        inv_index = cls()
        raw_index = defaultdict(lambda: defaultdict(int))

        doc_stream = iter_documents(corpus_dir)
        if limit is not None:
            doc_stream = islice(doc_stream, limit)

        for doc_id, text in doc_stream:
            doc_tokens = list(tokenize(text))
            inv_index.docs[doc_id] = DocMeta(doc_id=doc_id, length=len(doc_tokens))

            term_freqs = defaultdict(int)
            for token in doc_tokens:
                term_freqs[token] += 1

            for term, freq in term_freqs.items():
                raw_index[term][doc_id] = freq

        for term, doc_freqs in raw_index.items():
            sorted_postings = [
                Posting(doc_id=d_id, freq=f)
                for d_id, f in sorted(doc_freqs.items())
            ]
            inv_index.index[term] = sorted_postings

        return inv_index

    def save_pickle(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(path: str | Path) -> "InvertedIndex":
        """УВАГА: pickle.load небезпечний для завантаження файлів з ненадійних джерел (RCE)."""
        with open(path, "rb") as f:
            return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу findex")
    parser.add_argument("corpus_dir", type=str, help="Шлях до корпусу")
    parser.add_argument("--out", type=str, default="index.bin", help="Файл збереження індексу")
    parser.add_argument("--limit", type=int, default=None, help="Ліміт документів")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.time()

    index = InvertedIndex.build_from_corpus(args.corpus_dir, limit=args.limit)
    index.save_pickle(args.out)

    elapsed = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Індекс успішно збережено у {args.out}")
    print(f"Документів: {len(index.docs)} | Унікальних термів: {len(index.index)}")
    print(f"Час побудови: {elapsed:.4f} сек")
    print(f"Пікова пам'ять: {peak_mem / (1024 * 1024):.4f} MB")


if __name__ == "__main__":
    main()