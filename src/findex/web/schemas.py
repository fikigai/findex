from typing import Literal

from pydantic import BaseModel, Field


class SearchHitOut(BaseModel):
    rank: int
    doc_id: str
    score: float
    snippet: str


class SearchResponseOut(BaseModel):
    query: str
    mode: Literal["lexical", "semantic", "hybrid"] = "lexical"
    scorer: Literal["bm25", "tfidf"]
    page: int
    page_size: int
    total_hits: int
    total_pages: int
    elapsed_ms: float
    results: list[SearchHitOut]


class DocumentOut(BaseModel):
    doc_id: str
    length: int
    content: str


class StatsOut(BaseModel):
    num_docs: int
    vocab_size: int
    avg_doc_length: float


class HealthOut(BaseModel):
    status: Literal["ok", "unavailable"]
    index_loaded: bool
    num_docs: int = Field(ge=0)