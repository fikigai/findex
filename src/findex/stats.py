import logging
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.decorators import timed
from findex.tokenizer import tokenize

logger = logging.getLogger("findex")


@dataclass(frozen=True, slots=True)
class CorpusStats:
    doc_count: int
    token_count: int
    vocab_size: int
    top_terms: list[tuple[str, int]]


@timed
def compute_stats(
    corpus_dir: str | Path,
    limit: int | None = None,
    top_n: int = 10,
) -> CorpusStats:
    """Обчислює статистику корпусу за один потоковий прохід."""
    counter: Counter[str] = Counter()
    doc_count = 0
    token_count = 0

    stream = iter_documents(corpus_dir)
    if limit is not None:
        stream = islice(stream, limit)

    for _, text in stream:
        doc_count += 1
        for token in tokenize(text):
            token_count += 1
            counter[token] += 1

    logger.info("Статистику зібрано: docs=%d, tokens=%d", doc_count, token_count)
    return CorpusStats(
        doc_count=doc_count,
        token_count=token_count,
        vocab_size=len(counter),
        top_terms=counter.most_common(top_n),
    )