from pathlib import Path

import pytest

from findex.index import Index


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.txt").write_text(
        "Python async event loop makes network programming fast and clean.",
        encoding="utf-8",
    )
    (corpus_dir / "doc2.txt").write_text(
        "Python data structures and inverted index search engine in python.",
        encoding="utf-8",
    )
    (corpus_dir / "doc3.txt").write_text(
        "Java virtual machine and enterprisethreads without event loop.",
        encoding="utf-8",
    )
    return corpus_dir


@pytest.fixture
def sample_index(sample_corpus: Path) -> Index:
    return Index.build_from_corpus(sample_corpus)