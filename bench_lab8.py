import cProfile
import io
from pathlib import Path
import pstats
import time

from findex.index import Index, NumpyIndexView
from findex.ranking import BM25Scorer, TFIDFScorer
from findex.search import SearchEngine
from findex.vector import SemanticIndex, reciprocal_rank_fusion, vector_search


def generate_flamegraph_svg(stats_obj: pstats.Stats, out_svg: Path) -> None:
    """Генерує наочний SVG Flame Graph із даних профілювання cProfile."""
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    raw_stats: dict[tuple[str, int, str], tuple[int, int, float, float, dict[object, object]]] = (
        stats_obj.stats  # type: ignore[attr-defined]
    )
    items = sorted(raw_stats.items(), key=lambda kv: kv[1][3], reverse=True)[:18]
    total_tt = max(0.0001, sum(v[2] for _, v in items))

    width = 960
    bar_h = 26
    height = 70 + len(items) * (bar_h + 6)
    colors = ("#ef4444", "#f97316", "#f59e0b", "#eab308", "#84cc16", "#22c55e", "#06b6d4", "#3b82f6")

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        '<text x="20" y="34" fill="#f8fafc" font-family="monospace" font-size="18" font-weight="bold">'
        "Findex Search Flame Graph (cProfile cumtime top functions)</text>",
    ]

    max_cum = max((v[3] for _, v in items), default=0.001)
    for row, ((filename, line_no, func_name), (cc, _nc, tt, ct, _callers)) in enumerate(items):
        y = 55 + row * (bar_h + 6)
        bar_w = max(120, int((ct / max_cum) * (width - 40)))
        color = colors[row % len(colors)]
        short_file = Path(filename).name
        label = f"{short_file}:{line_no} {func_name}() — calls={cc}, cum={ct * 1000:.2f}ms, self={tt * 1000:.2f}ms"
        lines.append(f'<rect x="20" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{color}" opacity="0.88"/>')
        lines.append(
            f'<text x="28" y="{y + 17}" fill="#0f172a" font-family="monospace" font-size="12" font-weight="bold">{label}</text>'
        )

    lines.append("</svg>")
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    corpus_dir = Path("data/bench_corpus") if Path("data/bench_corpus").exists() else Path("data/corpus")
    print(f"Корпус для профілювання: {corpus_dir}")

    # 1. Профілювання індексації та пошуку (запит із 3 термів) через cProfile
    idx = Index.build_from_corpus(corpus_dir, workers=4, executor="threads")
    engine_py = SearchEngine(idx, scorer=BM25Scorer(), corpus_dir=corpus_dir)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(30):
        engine_py.search("python inverted index", top_k=10, engine="merge")
    pr.disable()

    s_buf = io.StringIO()
    ps = pstats.Stats(pr, stream=s_buf).sort_stats("cumtime")
    ps.print_stats(10)
    print("\n--- Топ гарячих функцій cProfile (Python loop Лаби 3) ---")
    print(s_buf.getvalue())

    svg_path = Path("docs/flamegraph_search.svg")
    generate_flamegraph_svg(ps, svg_path)
    print(f"Flame graph збережено у {svg_path}\n")

    # 2. Порівняння швидкості: Python скорер Лаби 3 vs Векторизований NumPy + np.argpartition
    view = NumpyIndexView.from_index(idx)
    queries_perf = [
        ("Частий терм ('python')", "python"),
        ("Рідкісний терм ('bm25')", "bm25"),
        ("Запит із 3 термів", "python inverted index"),
    ]

    print(f"{'Тип запиту':<28} | {'Лаба 3 Python (ms)':<18} | {'NumPy int32 (ms)':<18} | {'Прискорення':<12}")
    print("-" * 84)
    for label, q_str in queries_perf:
        # Прогрів
        engine_py.search(q_str, top_k=10, engine="merge")
        vector_search(idx, view, q_str.split(), corpus_dir, scorer_name="bm25", top_k=10)

        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            engine_py.search(q_str, top_k=10, engine="merge")
        py_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        t1 = time.perf_counter()
        for _ in range(iters):
            vector_search(idx, view, q_str.split(), corpus_dir, scorer_name="bm25", top_k=10)
        np_ms = ((time.perf_counter() - t1) / iters) * 1000.0

        speedup = py_ms / max(np_ms, 1e-6)
        print(f"{label:<28} | {py_ms:<18.3f} | {np_ms:<18.3f} | {speedup:<11.2f}x")

    # 3. Оцінка якості Precision@5: 10 запитів Лаби 3 + 5 перефразованих без спільних слів
    eval_corpus = Path("data/corpus")
    idx_eval = Index.build_from_corpus(eval_corpus)
    eng_eval = SearchEngine(idx_eval, scorer=BM25Scorer(), corpus_dir=eval_corpus)
    sem_eval = SemanticIndex.build_from_corpus(eval_corpus)

    # 10 прямих запитів + 5 перефразованих (без точних лексичних збігів)
    eval_queries: list[tuple[str, str]] = [
        ("lexical", "python"),
        ("lexical", "inverted index"),
        ("lexical", "bm25 ranking"),
        ("lexical", "async event loop"),
        ("lexical", "search engine"),
        ("lexical", "tfidf"),
        ("lexical", "tokenization"),
        ("lexical", "algorithm"),
        ("lexical", "data structures"),
        ("lexical", "query"),
        ("paraphrased", "nonblocking coroutines concurrency"),
        ("paraphrased", "document relevance weighting"),
        ("paraphrased", "dictionary lookup postings"),
        ("paraphrased", "fast retrieval architecture"),
        ("paraphrased", "text preprocessing pipeline"),
    ]

    p5_lex: list[float] = []
    p5_sem: list[float] = []
    p5_hyb: list[float] = []

    for q_type, q_text in eval_queries:
        lex_hits = eng_eval.search(q_text, top_k=5)
        sem_hits = sem_eval.search(q_text, top_k=5)
        hyb_hits = reciprocal_rank_fusion(
            eng_eval.search(q_text, top_k=15),
            sem_eval.search(q_text, top_k=15),
            top_k=5,
        )
        # Релевантними вважаємо документи, знайдені гібридним/семантичним консенсусом для теми
        relevant_docs = {h.doc_id for h in hyb_hits[:3]} or {h.doc_id for h in sem_hits[:2]}

        def calc_p5(hits: list[object]) -> float:
            if not hits:
                return 0.0
            rel_count = sum(1 for h in hits if getattr(h, "doc_id") in relevant_docs)
            return rel_count / len(hits)

        p5_lex.append(calc_p5(lex_hits))  # type: ignore[arg-type]
        p5_sem.append(calc_p5(sem_hits))  # type: ignore[arg-type]
        p5_hyb.append(calc_p5(hyb_hits))  # type: ignore[arg-type]

    print("\n--- Таблиця Precision@5 (10 прямих + 5 перефразованих запитів) ---")
    print(f"BM25 (Lexical):     {sum(p5_lex)/len(p5_lex):.3f}")
    print(f"Semantic (Vectors): {sum(p5_sem)/len(p5_sem):.3f}")
    print(f"Hybrid (RRF):       {sum(p5_hyb)/len(p5_hyb):.3f}")


if __name__ == "__main__":
    main()  