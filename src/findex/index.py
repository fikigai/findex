import json
import logging
import pickle
from collections import abc, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from pathlib import Path
from types import TracebackType
from typing import Any

from findex.corpus import iter_documents
from findex.decorators import timed
from findex.tokenizer import tokenize

logger = logging.getLogger("findex")


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: str
    freq: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: str
    length: int


@dataclass
class _TermAccumulator:
    freq: int = 0
    positions: list[int] = field(default_factory=lambda: list[int]())

class Index(abc.Mapping[str, list[Posting]]):
    """Інвертований індекс, що реалізує протокол Mapping[str, list[Posting]]."""

    def __init__(
        self,
        index_dict: dict[str, list[Posting]] | None = None,
        docs_dict: dict[str, DocMeta] | None = None,
    ) -> None:
        self.index: dict[str, list[Posting]] = index_dict if index_dict is not None else {}
        self.docs: dict[str, DocMeta] = docs_dict if docs_dict is not None else {}

    def __getitem__(self, term: str) -> list[Posting]:
        return self.index[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return f"<Index terms={len(self.index)} docs={len(self.docs)}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return self.index == other.index and self.docs == other.docs

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
        cls,
        corpus_dir: str | Path,
        limit: int | None = None,
        keep_positions: bool = True,
        progress_cb: Callable[[str], None] | None = None,
    ) -> "Index":
        raw_index: dict[str, dict[str, _TermAccumulator]] = defaultdict(
            lambda: defaultdict(_TermAccumulator)
        )
        docs_dict: dict[str, DocMeta] = {}

        doc_stream = iter_documents(corpus_dir)
        if limit is not None:
            doc_stream = islice(doc_stream, limit)

        for doc_id, text in doc_stream:
            tokens = list(tokenize(text))
            docs_dict[doc_id] = DocMeta(doc_id=doc_id, length=len(tokens))

            for pos, token in enumerate(tokens):
                acc = raw_index[token][doc_id]
                acc.freq += 1
                if keep_positions:
                    acc.positions.append(pos)

            if progress_cb is not None:
                progress_cb(doc_id)

        processed_index: dict[str, list[Posting]] = {}
        for term, doc_data in raw_index.items():
            sorted_postings = [
                Posting(
                    doc_id=d_id,
                    freq=acc.freq,
                    positions=tuple(acc.positions) if keep_positions else (),
                )
                for d_id, acc in sorted(doc_data.items())
            ]
            processed_index[term] = sorted_postings

        logger.info(
            "Побудовано індекс: %d документів, %d термів",
            len(docs_dict),
            len(processed_index),
        )
        return cls(processed_index, docs_dict)

    @timed
    def save_pickle(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    @timed
    def load_pickle(path: str | Path) -> "Index":
        """Завантажує індекс із pickle (небезпечно для чужих файлів через RCE)."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл індексу не знайдено: {file_path}")
        with open(file_path, "rb") as f:
            loaded: Any = pickle.load(f)
            if not isinstance(loaded, Index):
                raise TypeError("Пошкоджений формат файлу індексу")
            return loaded

    @timed
    def save_json(self, path: str | Path) -> None:
        data = {
            "docs": {d_id: meta.length for d_id, meta in self.docs.items()},
            "index": {
                term: [[p.doc_id, p.freq, list(p.positions)] for p in postings]
                for term, postings in self.index.items()
            },
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    @timed
    def load_json(path: str | Path) -> "Index":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл індексу не знайдено: {file_path}")
        raw: dict[str, Any] = json.loads(file_path.read_text(encoding="utf-8"))
        docs = {
            str(d_id): DocMeta(doc_id=str(d_id), length=int(length))
            for d_id, length in raw["docs"].items()
        }
        index_dict: dict[str, list[Posting]] = {}
        for term, plist in raw["index"].items():
            index_dict[str(term)] = [
                Posting(
                    doc_id=str(item[0]),
                    freq=int(item[1]),
                    positions=tuple(int(x) for x in item[2]),
                )
                for item in plist
            ]
        return Index(index_dict, docs)


InvertedIndex = Index


class open_index:
    """Контекстний менеджер для відкриття та автоматичного очищення ресурсів індексу."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.index: Index | None = None

    def __enter__(self) -> Index:
        if self.path.suffix == ".json":
            self.index = Index.load_json(self.path)
        else:
            self.index = Index.load_pickle(self.path)
        return self.index

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self.index = None
        return False