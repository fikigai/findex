from collections import Counter
from pathlib import Path
import time
import tracemalloc
from findex.corpus import iter_documents
from findex.tokenizer import tokenize


def collect_stats_stream(corpus_dir: str | Path) -> dict:
    """Збирає статистику корпусу в один прохід за допомогою генераторів (економія пам'яті)."""
    tracemalloc.start()
    start_time = time.time()

    doc_count = 0
    total_tokens = 0
    term_frequencies = Counter()

    for doc_id, text in iter_documents(corpus_dir):
        doc_count += 1
        for token in tokenize(text):
            total_tokens += 1
            term_frequencies[token] += 1

    elapsed_time = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "doc_count": doc_count,
        "total_tokens": total_tokens,
        "vocab_size": len(term_frequencies),
        "top_50": term_frequencies.most_common(50),
        "time_sec": elapsed_time,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }


def collect_stats_greedy(corpus_dir: str | Path) -> dict:
    """Жадібна версія: завантажує всі документи та списки токенів у пам'ять для порівняння."""
    tracemalloc.start()
    start_time = time.time()

    docs = list(iter_documents(corpus_dir))
    doc_count = len(docs)

    all_tokens = []
    for doc_id, text in docs:
        tokens = list(tokenize(text))
        all_tokens.extend(tokens)

    total_tokens = len(all_tokens)
    term_frequencies = Counter(all_tokens)

    elapsed_time = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "doc_count": doc_count,
        "total_tokens": total_tokens,
        "vocab_size": len(term_frequencies),
        "time_sec": elapsed_time,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }