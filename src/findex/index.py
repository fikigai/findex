import argparse
from collections import abc, defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path
import pickle
from typing import Dict, Iterator, List, Tuple
from findex.corpus import iter_documents
from findex.decorators import timed
from findex.tokenizer import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: str
    freq: int
    positions: Tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: str
    length: int


class Index(abc.Mapping):
    """Об'єкт індексу, що реалізує протокол Mapping (як dict у Python)."""

    def __init__(
        self,
        index_dict: Dict[str, List[Posting]] = None,
        docs_dict: Dict[str, DocMeta] = None,
    ):
        self.index = index_dict or {}
        self.docs = docs_dict or {}

    def __getitem__(self, term: str) -> List[Posting]:
        return self.index[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return f"<Index terms={len(self.index)} docs={len(self.docs)}>"

    @property
    def num_docs(self) -> int:
        return len(self.docs)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.docs:
            return 0.0
        total_len = sum(meta.length for meta in self.docs.values())
        return total_len / len(self.docs)

    def df(self, term: str) -> int:
        return len(self.index.get(term, []))

    @classmethod
    @timed
    def build_from_corpus(
        cls, corpus_dir: str | Path, limit: int | None = None
    ) -> "Index":
        raw_index = defaultdict(
            lambda: defaultdict(lambda: {"freq": 0, "positions": []})
        )
        docs_dict = {}

        doc_stream = iter_documents(corpus_dir)
        if limit is not None:
            doc_stream = islice(doc_stream, limit)

        for doc_id, text in doc_stream:
            tokens = list(tokenize(text))
            docs_dict[doc_id] = DocMeta(doc_id=doc_id, length=len(tokens))

            for pos, token in enumerate(tokens):
                raw_index[token][doc_id]["freq"] += 1
                raw_index[token][doc_id]["positions"].append(pos)

        processed_index = {}
        for term, doc_data in raw_index.items():
            sorted_postings = [
                Posting(
                    doc_id=d_id,
                    freq=data["freq"],
                    positions=tuple(data["positions"]),
                )
                for d_id, data in sorted(doc_data.items())
            ]
            processed_index[term] = sorted_postings

        return cls(processed_index, docs_dict)

    def save_pickle(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    @timed
    def load_pickle(path: str | Path) -> "Index":
        with open(path, "rb") as f:
            return pickle.load(f)


InvertedIndex = Index


class open_index:
    """Контекстний менеджер для безпечного відкриття та очищення ресурсів індексу."""

    def __init__(self, path: str | Path):
        self.path = path
        self.index: Index | None = None

    def __enter__(self) -> Index:
        self.index = Index.load_pickle(self.path)
        return self.index

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.index = None
        return False


def main():
    from findex.index import Index as PackageIndex

    parser = argparse.ArgumentParser(
        description="Побудова інвертованого індексу findex (Lab 3)"
    )
    parser.add_argument("corpus_dir", type=str, help="Шлях до корпусу")
    parser.add_argument(
        "--out", type=str, default="index.bin", help="Файл збереження індексу"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Ліміт документів"
    )
    args = parser.parse_args()

    index = PackageIndex.build_from_corpus(args.corpus_dir, limit=args.limit)
    index.save_pickle(args.out)

    print(f"Індекс збережено у {args.out}: {repr(index)}")
    print(f"Середня довжина документа: {index.avg_doc_length:.2f} токенів")


if __name__ == "__main__":
    main()