import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.index import DocMeta, Index, Posting, open_index
from findex.query import (
    AndNode,
    NotNode,
    OrNode,
    PhraseNode,
    QuerySyntaxError,
    TermNode,
    parse_query,
)
from findex.ranking import BM25Scorer, Scorer, generate_snippet, make_scorer
from findex.search import SearchEngine
from findex.stats import compute_stats
from findex.tokenizer import tokenize

runner = CliRunner()


# --- 1. Параметризовані тести токенізатора (10 тестів) ---
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello WORLD", ["hello", "world"]),
        ("Findex 2026!", ["findex", "2026"]),
        ("Пошукова система України", ["пошукова", "система", "україни"]),
        ("state-of-the-art", ["state-of-the-art"]),
        ("don't stop", ["don't", "stop"]),
        ("пам’ять комп'ютера", ["пам’ять", "комп'ютера"]),
        ("caf\u00e9 and cafe\u0301", ["café", "and", "café"]),
        ("Straße", ["strasse"]),
        ("   ...,,,!!!   ", []),
        ("Python_3.12 test", ["python", "3", "12", "test"]),
    ],
)
def test_tokenizer_parametrized(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


# --- 2. Корпус, статистика та модель Index (6 тестів) ---
def test_corpus_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_documents(tmp_path / "nonexistent"))


def test_compute_stats(sample_corpus: Path) -> None:
    stats = compute_stats(sample_corpus, limit=2, top_n=3)
    assert stats.doc_count == 2
    assert stats.token_count > 0
    assert stats.vocab_size > 0
    assert len(stats.top_terms) <= 3


def test_index_mapping_protocol(sample_index: Index) -> None:
    assert len(sample_index) > 0
    assert "python" in sample_index
    assert sample_index.num_docs == 3
    assert sample_index.avg_doc_length > 0
    assert sample_index.df("python") == 2
    assert "Index terms=" in repr(sample_index)
    assert list(iter(sample_index)) == list(sample_index.index.keys())
    assert len(sample_index["python"]) == 2


def test_save_and_load_pickle(sample_index: Index, tmp_path: Path) -> None:
    out_file = tmp_path / "idx.bin"
    sample_index.save_pickle(out_file)
    with open_index(out_file) as loaded:
        assert loaded == sample_index


def test_save_and_load_json(sample_index: Index, tmp_path: Path) -> None:
    out_file = tmp_path / "idx.json"
    sample_index.save_json(out_file)
    with open_index(out_file) as loaded:
        assert loaded == sample_index


def test_open_index_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError), open_index(tmp_path / "missing.bin"):
        pass


# --- 3. Парсер запитів та булеве AST (7 тестів) ---
def test_query_parser_complex_tree() -> None:
    ast = parse_query('python AND (async OR await) NOT java "event loop"')
    expected = AndNode(
        AndNode(
            AndNode(
                TermNode("python"),
                OrNode(TermNode("async"), TermNode("await")),
            ),
            NotNode(TermNode("java")),
        ),
        PhraseNode(["event", "loop"]),
    )
    assert ast == expected


def test_query_dsl_operators(sample_index: Index) -> None:
    node = TermNode("python") & (TermNode("async") | ~TermNode("java"))
    assert node.evaluate(sample_index) == {"doc1.txt", "doc2.txt"}


def test_phrase_evaluation(sample_index: Index) -> None:
    node = PhraseNode(["event", "loop"])
    assert node.evaluate(sample_index) == {"doc1.txt", "doc3.txt"}
    assert PhraseNode(["loop", "event"]).evaluate(sample_index) == set()


@pytest.mark.parametrize(
    "bad_query",
    [
        'python "unclosed phrase',
        "python AND (async OR await",
        "python AND ()",
        "AND python",
    ],
)
def test_query_parser_syntax_errors(bad_query: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse_query(bad_query)


# --- 4. Протокол Scorer та 3 перевірки ранжування (5 тестів) ---
@pytest.mark.parametrize("scorer_name", ["bm25", "tfidf"])
def test_scorer_sanity_checks(scorer_name: str) -> None:
    scorer: Scorer = make_scorer(scorer_name)  # type: ignore[arg-type] # перевірка фабрики з str
    index = Index(
        index_dict={
            "common": [
                Posting("d1", 1, (0,)),
                Posting("d2", 1, (0,)),
                Posting("d3", 1, (0,)),
            ],
            "rare": [Posting("d1", 1, (1,))],
            "rep": [Posting("d1", 1, (0,)), Posting("d2", 19, (0,)), Posting("d3", 20, (0,))],
        },
        docs_dict={
            "d1": DocMeta("d1", 10),
            "d2": DocMeta("d2", 10),
            "d3": DocMeta("d3", 100),
        },
    )
    # 1. Рідкий термін вище за частий
    assert scorer.score(index, ["rare"], "d1", 10) > scorer.score(index, ["common"], "d1", 10)
    # 2. Короткий документ вище за довгий при однаковій частоті
    assert scorer.score(index, ["common"], "d2", 10) > scorer.score(index, ["common"], "d3", 100)
    # 3. 20-те повторення додає значно менше, ніж перехід від 1 до 19
    s1 = scorer.score(index, ["rep"], "d1", 20)
    s19 = scorer.score(index, ["rep"], "d2", 20)
    s20 = scorer.score(index, ["rep"], "d3", 20)
    assert (s20 - s19) < (s19 - s1)


def test_snippet_highlighting() -> None:
    snippet = generate_snippet("Розробка системи на Python є швидкою.", ["python"], window=20)
    assert "**Python**" in snippet


def test_search_engine_cached(sample_index: Index, sample_corpus: Path) -> None:
    engine = SearchEngine(sample_index, scorer=BM25Scorer(), corpus_dir=sample_corpus)
    res1 = engine.search_cached("python", top_k=2)
    res2 = engine.search_cached("python", top_k=2)
    assert res1 == res2
    assert len(res1) == 2


# --- 5. Властивості Hypothesis (3 property-based тести) ---
_DOC_ID_STRATEGY = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=6
).map(lambda s: f"{s}.txt")
_TERM_STRATEGY = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)


@pytest.mark.slow
@settings(max_examples=25)
@given(
    raw_docs=st.dictionaries(
        keys=_DOC_ID_STRATEGY,
        values=st.lists(_TERM_STRATEGY, min_size=1, max_size=15),
        min_size=1,
        max_size=6,
    )
)
def test_hypothesis_roundtrip_and_sorted_postings(
    raw_docs: dict[str, list[str]], tmp_path_factory: pytest.TempPathFactory
) -> None:
    corpus_dir = tmp_path_factory.mktemp("hyp_corpus")
    for doc_id, tokens in raw_docs.items():
        (corpus_dir / doc_id).write_text(" ".join(tokens), encoding="utf-8")

    idx = Index.build_from_corpus(corpus_dir)

    # Властивість 1: постінги завжди відсортовані за doc_id
    for postings in idx.values():
        doc_ids = [p.doc_id for p in postings]
        assert doc_ids == sorted(doc_ids)

    # Властивість 2: load(save(index)) дає ідентичний індекс
    out_bin = corpus_dir / "index.bin"
    idx.save_pickle(out_bin)
    loaded = Index.load_pickle(out_bin)
    assert loaded == idx


@settings(max_examples=40)
@given(
    term_docs=st.dictionaries(
        keys=_TERM_STRATEGY,
        values=st.sets(_DOC_ID_STRATEGY, min_size=1, max_size=8),
        min_size=1,
        max_size=6,
    ),
    query_terms=st.lists(_TERM_STRATEGY, min_size=1, max_size=3),
)
def test_hypothesis_merge_matches_set(
    term_docs: dict[str, set[str]], query_terms: list[str]
) -> None:
    index_dict = {
        term: [Posting(doc_id=d, freq=1) for d in sorted(docs)]
        for term, docs in term_docs.items()
    }
    all_docs = {d: DocMeta(d, 10) for docs in term_docs.values() for d in docs}
    idx = Index(index_dict, all_docs)

    # Властивість 3: двовказівниковий merge завжди збігається з перетином set
    assert SearchEngine.search_merge(idx, query_terms) == SearchEngine.search_set(idx, query_terms)


# --- 6. Тести CLI через typer.testing.CliRunner (6 тестів) ---
def test_cli_index_and_search_rich(sample_corpus: Path, tmp_path: Path) -> None:
    out_idx = tmp_path / "cli_index.bin"
    res_idx = runner.invoke(app, ["-v", "index", str(sample_corpus), "--out", str(out_idx)])
    assert res_idx.exit_code == 0
    assert out_idx.exists()

    res_search = runner.invoke(
        app,
        ["search", str(out_idx), "python", "--corpus", str(sample_corpus), "--scorer", "tfidf"],
    )
    assert res_search.exit_code == 0
    assert "doc1.txt" in res_search.stdout


def test_cli_search_json_stdout(sample_corpus: Path, tmp_path: Path) -> None:
    out_idx = tmp_path / "cli_index.bin"
    runner.invoke(app, ["index", str(sample_corpus), "--out", str(out_idx)])

    res = runner.invoke(
        app,
        ["-vv", "search", str(out_idx), "python", "--corpus", str(sample_corpus), "--json"],
    )
    assert res.exit_code == 0
    data: list[dict[str, object]] = json.loads(res.stdout)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["doc_id"] in ("doc1.txt", "doc2.txt")


def test_cli_stats(sample_corpus: Path) -> None:
    res = runner.invoke(app, ["stats", str(sample_corpus)])
    assert res.exit_code == 0
    assert "Документів" in res.stdout


def test_cli_missing_index_error(tmp_path: Path) -> None:
    res = runner.invoke(app, ["search", str(tmp_path / "no_file.bin"), "python"])
    assert res.exit_code != 0
    assert "Traceback" not in res.stdout
    assert "Traceback" not in res.stderr


def test_cli_broken_query_error(sample_corpus: Path, tmp_path: Path) -> None:
    out_idx = tmp_path / "cli_index.bin"
    runner.invoke(app, ["index", str(sample_corpus), "--out", str(out_idx)])

    res = runner.invoke(app, ["search", str(out_idx), 'python "unclosed'])
    assert res.exit_code != 0
    assert "Traceback" not in res.stdout