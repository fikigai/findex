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


    # --- 7. Тести Лабораторної №5: build_partial, merge та паралельні пули ---
def test_merge_single_partial_equals_full_build(sample_corpus: Path) -> None:
    from findex.corpus import list_corpus_files
    from findex.index import build_partial, merge_indices

    files = list_corpus_files(sample_corpus)
    tasks = [(f.relative_to(sample_corpus).as_posix(), f) for f in files]

    expected = Index.build_from_corpus(sample_corpus, executor="serial", workers=1)
    actual = merge_indices([build_partial(tasks)])
    assert actual == expected


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_parallel_executors_match_serial(sample_corpus: Path, executor: str) -> None:
    serial_idx = Index.build_from_corpus(sample_corpus, executor="serial", workers=1)
    parallel_idx = Index.build_from_corpus(
        sample_corpus,
        executor=executor,  # type: ignore[arg-type] # тестування параметризованих режимів
        workers=2,
    )
    assert parallel_idx == serial_idx


def test_worker_crash_propagates_exception(tmp_path: Path) -> None:
    from findex.index import build_partial

    missing_file = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        build_partial([("missing.txt", missing_file)])


def test_cli_index_with_workers_and_executor(sample_corpus: Path, tmp_path: Path) -> None:
    out_idx = tmp_path / "parallel.bin"
    res = runner.invoke(
        app,
        [
            "index",
            str(sample_corpus),
            "--out",
            str(out_idx),
            "--workers",
            "2",
            "--executor",
            "threads",
        ],
    )
    assert res.exit_code == 0
    assert out_idx.exists()

    # --- 8. Тести Лабораторної №6: Асинхронний краулер з httpx.MockTransport ---
def test_canonicalize_url() -> None:
    from findex.crawler import canonicalize_url

    raw = "HTTPS://Example.COM:443/docs/page/?b=2&a=1#section-3"
    assert canonicalize_url(raw) == "https://example.com/docs/page?a=1&b=2"


def test_fetch_retries_429_and_503_with_retry_after() -> None:
    import asyncio

    import httpx

    from findex.crawler import fetch

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"}, text="Rate limited")
        if call_count == 2:
            return httpx.Response(503, text="Temporary overload")
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<h1>OK</h1>")

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(client, "http://test.local/item", base_backoff=0.01)
            assert res.status == 200
            assert res.attempts == 3

    asyncio.run(_run())
    assert call_count == 3


def test_fetch_does_not_retry_404() -> None:
    import asyncio

    import httpx

    from findex.crawler import fetch

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(404, text="Not Found")

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(client, "http://test.local/missing", base_backoff=0.01)
            assert res.status == 404
            assert res.attempts == 1

    asyncio.run(_run())
    assert call_count == 1


def test_robots_cache_disallows_blocked_path() -> None:
    import asyncio

    import httpx

    from findex.crawler import RobotsCache

    robots_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_calls
        if request.url.path == "/robots.txt":
            robots_calls += 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        return httpx.Response(200, text="ok")

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            rc = RobotsCache(client)
            assert await rc.can_fetch("http://test.local/public/page") is True
            assert await rc.can_fetch("http://test.local/private/secret") is False
            # Повторний виклик береться з кешу без нового HTTP-запиту
            assert await rc.can_fetch("http://test.local/public/other") is True

    asyncio.run(_run())
    assert robots_calls == 1


def test_crawl_respects_max_pages_seen_and_robots() -> None:
    import asyncio

    import httpx

    from findex.crawler import CrawledPage, crawl

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /blocked\n")
        html = f"""
        <html>
          <head><title>Page {path}</title></head>
          <body>
            <p>Asyncio python crawler content on {path}</p>
            <a href="/page1#frag">Page 1</a>
            <a href="/page2?b=2&a=1">Page 2</a>
            <a href="/page2?a=1&b=2">Page 2 duplicate</a>
            <a href="/page3">Page 3</a>
            <a href="/blocked">Blocked by robots</a>
          </body>
        </html>
        """
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

    async def _run() -> list[CrawledPage]:
        transport = httpx.MockTransport(handler)
        pages: list[CrawledPage] = []
        async for page in crawl(
            "http://test.local/start",
            max_pages=3,
            concurrency=2,
            min_host_interval=0.0,
            transport=transport,
        ):
            pages.append(page)
        return pages

    collected = asyncio.run(_run())
    assert len(collected) == 3
    urls = {p.url for p in collected}
    assert "http://test.local/blocked" not in urls


def test_jsonl_corpus_indexing_and_search(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "crawled.jsonl"
    records = [
        {
            "url": "http://docs.local/page1",
            "status": 200,
            "title": "Asyncio Event Loop",
            "text": "Asyncio event loop runs coroutines and tasks concurrently.",
        },
        {
            "url": "http://docs.local/page2",
            "status": 200,
            "title": "BM25 Ranking",
            "text": "Inverted index with BM25 ranking scores documents.",
        },
    ]
    jsonl_file.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    idx_out = tmp_path / "crawled.bin"
    res_idx = runner.invoke(app, ["index", str(jsonl_file), "--out", str(idx_out)])
    assert res_idx.exit_code == 0

    res_search = runner.invoke(
        app,
        ["search", str(idx_out), "asyncio", "--corpus", str(jsonl_file), "--json"],
    )
    assert res_search.exit_code == 0
    results: list[dict[str, object]] = json.loads(res_search.stdout)
    assert len(results) == 1
    assert results[0]["doc_id"] == "http://docs.local/page1"

   # --- 9. Тести Лабораторної №7: FastAPI Web Search через TestClient і dependency_overrides ---
def test_web_health_503_when_index_missing() -> None:
    from typing import Any

    from fastapi.testclient import TestClient

    from findex.web.app import create_app, get_optional_index

    web_app = create_app()
    web_app.dependency_overrides[get_optional_index] = lambda: None
    client: Any = TestClient(web_app)

    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["index_loaded"] is False


def test_web_search_pagination_and_stats(sample_index: Index, sample_corpus: Path) -> None:
    from typing import Any

    from fastapi.testclient import TestClient

    from findex.web.app import create_app, get_app_settings, get_optional_index
    from findex.web.config import Settings

    web_app = create_app()
    web_app.dependency_overrides[get_optional_index] = lambda: sample_index
    web_app.dependency_overrides[get_app_settings] = lambda: Settings(corpus_dir=sample_corpus)
    client: Any = TestClient(web_app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["num_docs"] == 3

    stats = client.get("/stats")
    assert stats.status_code == 200
    assert stats.json()["num_docs"] == 3

    search_resp = client.get("/search", params={"q": "python", "k": 1, "page": 1, "scorer": "bm25"})
    assert search_resp.status_code == 200
    data = search_resp.json()
    assert data["total_hits"] == 2
    assert data["total_pages"] == 2
    assert len(data["results"]) == 1
    assert "X-Request-ID" in search_resp.headers


def test_web_search_422_on_invalid_params(sample_index: Index) -> None:
    from typing import Any

    from fastapi.testclient import TestClient

    from findex.web.app import create_app, get_optional_index

    web_app = create_app()
    web_app.dependency_overrides[get_optional_index] = lambda: sample_index
    client: Any = TestClient(web_app)

    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "python", "scorer": "unknown"}).status_code == 422
    assert client.get("/search", params={"q": "AND OR"}).status_code == 422


def test_web_document_404_and_200(sample_index: Index, sample_corpus: Path) -> None:
    from typing import Any

    from fastapi.testclient import TestClient

    from findex.web.app import create_app, get_app_settings, get_optional_index
    from findex.web.config import Settings

    web_app = create_app()
    web_app.dependency_overrides[get_optional_index] = lambda: sample_index
    web_app.dependency_overrides[get_app_settings] = lambda: Settings(corpus_dir=sample_corpus)
    client: Any = TestClient(web_app)

    ok_doc = client.get("/docs/doc1.txt")
    assert ok_doc.status_code == 200
    assert "Python" in ok_doc.json()["content"]

    missing_doc = client.get("/docs/non_existent.txt")
    assert missing_doc.status_code == 404


def test_web_html_ui_renders(sample_index: Index, sample_corpus: Path) -> None:
    from typing import Any

    from fastapi.testclient import TestClient

    from findex.web.app import create_app, get_app_settings, get_optional_index
    from findex.web.config import Settings

    web_app = create_app()
    web_app.dependency_overrides[get_optional_index] = lambda: sample_index
    web_app.dependency_overrides[get_app_settings] = lambda: Settings(corpus_dir=sample_corpus)
    client: Any = TestClient(web_app)

    html_resp = client.get("/", params={"q": "python", "doc": "doc1.txt"})
    assert html_resp.status_code == 200
    assert "<mark>" in html_resp.text
    assert "doc1.txt" in html_resp.text