from dataclasses import dataclass, field
import math
import re
from typing import List, Protocol
from findex.index import Index


class Scorer(Protocol):
    def score(self, index: Index, terms: List[str], doc_id: str, doc_len: int) -> float:
        ...


@dataclass(order=True)
class SearchResult:
    score: float
    doc_id: str = field(compare=False)
    snippet: str = field(compare=False, default="")


class TfIdfScorer:
    def score(self, index: Index, terms: List[str], doc_id: str, doc_len: int) -> float:
        score = 0.0
        for term in terms:
            postings = index.index.get(term, [])
            posting = next((p for p in postings if p.doc_id == doc_id), None)
            if posting and doc_len > 0:
                tf = 1.0 + math.log(posting.freq)
                df = index.df(term)
                idf = math.log(1.0 + index.num_docs / (df + 1.0))
                score += (tf * idf) / math.sqrt(doc_len)
        return score


class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, index: Index, terms: List[str], doc_id: str, doc_len: int) -> float:
        score = 0.0
        avg_dl = index.avg_doc_length
        if avg_dl == 0:
            return 0.0

        for term in terms:
            postings = index.index.get(term, [])
            posting = next((p for p in postings if p.doc_id == doc_id), None)
            if posting:
                freq = posting.freq
                df = index.df(term)
                idf = math.log((index.num_docs - df + 0.5) / (df + 0.5) + 1.0)
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl))
                score += idf * (numerator / denominator)
        return score


def generate_snippet(text: str, terms: List[str], window: int = 80) -> str:
    """Генерує вікно ±80 символів навколо найкращого входження та підсвічує терми."""
    if not text:
        return ""
    lower_text = text.casefold()
    best_pos = -1

    for term in terms:
        pos = lower_text.find(term.casefold())
        if pos != -1:
            best_pos = pos
            break

    if best_pos == -1:
        snippet = text[: window * 2]
        return snippet + ("..." if len(text) > window * 2 else "")

    start = max(0, best_pos - window)
    end = min(len(text), best_pos + window)
    snippet = text[start:end]

    for term in sorted(set(terms), key=len, reverse=True):
        if term:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            snippet = pattern.sub(lambda m: f"**{m.group(0)}**", snippet)

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet