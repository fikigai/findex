import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.crawler import CrawlMetrics, crawl
from findex.index import ExecutorType, Index, open_index
from findex.query import QuerySyntaxError
from findex.ranking import ScorerName, SearchResult, make_scorer
from findex.search import EngineName, SearchEngine
from findex.stats import compute_stats

app = typer.Typer(
    name="findex",
    help="Findex — швидка пошукова система з інвертованим індексом та ранжуванням BM25/TF-IDF.",
    add_completion=False,
)

stdout_console = Console()
stderr_console = Console(stderr=True)


def setup_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname)s [%(name)s]: %(message)s",
        stream=sys.stderr,
        force=True,
    )


@app.callback()
def main_callback(
    verbose: Annotated[
        int,
        typer.Option("-v", "--verbose", count=True, help="Підвищити детальність логів (-v, -vv)"),
    ] = 0,
) -> None:
    """Налаштовує рівень логування у stderr для всіх підкоманд."""
    setup_logging(verbose)


def _render_crawl_dashboard(metrics: CrawlMetrics, max_pages: int) -> Table:
    table = Table(title="Findex Async Crawler — Жива статистика")
    table.add_column("Зібрано сторінок", justify="right", style="green")
    table.add_column("Сторінок / сек", justify="right", style="cyan")
    table.add_column("Активних запитів", justify="right", style="yellow")
    table.add_column("Помилок", justify="right", style="red")
    table.add_column("Глибина черги", justify="right", style="magenta")

    table.add_row(
        f"{metrics.pages_collected} / {max_pages}",
        f"{metrics.pages_per_sec:.2f}",
        str(metrics.in_flight),
        str(metrics.errors),
        str(metrics.queue_depth),
    )
    return table

@app.command("crawl")
def crawl_cmd(
    seed_url: Annotated[str, typer.Argument(help="Початковий URL для обходу")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Вихідний JSONL-файл")
    ] = Path("crawled_pages.jsonl"),
    max_pages: Annotated[
        int, typer.Option("--max-pages", "-m", help="Максимальна кількість сторінок")
    ] = 200,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", help="Загальна кількість одночасних запитів")
    ] = 10,
    per_host: Annotated[
        int, typer.Option("--per-host", help="Ліміт одночасних запитів на один хост")
    ] = 5,
    delay: Annotated[
        float, typer.Option("--delay", help="Мінімальна пауза між запитами до хоста (сек)")
    ] = 0.01,
    log_file: Annotated[
        Path, typer.Option("--log-file", help="Лог-файл статусів кожного URL")
    ] = Path("crawl.log"),
    debug_loop: Annotated[
        bool, typer.Option("--debug-loop", help="Увімкнути debug-режим циклу подій asyncio")
    ] = False,
) -> None:
    """Асинхронно обійти сайт і потоково записати зібрані сторінки у JSONL."""
    import httpx

    crawler_logger = logging.getLogger("findex.crawler")
    crawler_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    crawler_logger.addHandler(file_handler)

    metrics = CrawlMetrics()

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with (
            open(out, "w", encoding="utf-8") as f_out,
            Live(
                _render_crawl_dashboard(metrics, max_pages),
                console=stderr_console,
                refresh_per_second=10,
                transient=False,
            ) as live,
        ):

            def update_ui(m: CrawlMetrics) -> None:
                live.update(_render_crawl_dashboard(m, max_pages))

            async def _run() -> None:
                async for page in crawl(
                    seed_url=seed_url,
                    max_pages=max_pages,
                    concurrency=concurrency,
                    per_host_concurrency=per_host,
                    min_host_interval=delay,
                    metrics=metrics,
                    on_update=update_ui,
                ):
                    record = {
                        "url": page.url,
                        "status": page.status,
                        "title": page.title,
                        "text": page.text,
                        "elapsed": round(page.elapsed, 4),
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    update_ui(metrics)

            asyncio.run(_run(), debug=debug_loop)

        stdout_console.print(
            f"[green]Обхід завершено:[/green] збережено {metrics.pages_collected} сторінок у {out} "
            f"({metrics.pages_per_sec:.2f} стор/с, помилок: {metrics.errors}, лог: {log_file})"
        )
    except (OSError, ValueError, httpx.HTTPError) as exc:
        stderr_console.print(f"[red]Помилка краулера:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from None
    finally:
        crawler_logger.removeHandler(file_handler)
        file_handler.close()
        
@app.command("index")
def index_cmd(
    corpus_dir: Annotated[Path, typer.Argument(help="Шлях до директорії корпусу або JSONL-файлу")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Файл збереження індексу")] = Path(
        "index.bin"
    ),
    limit: Annotated[int | None, typer.Option("--limit", help="Обмеження документів")] = None,
    workers: Annotated[
        int, typer.Option("--workers", "-w", help="Кількість воркерів")
    ] = os.cpu_count() or 1,
    executor: Annotated[
        ExecutorType,
        typer.Option("--executor", "-e", help="Режим виконання: serial, threads, processes"),
    ] = "processes",
) -> None:
    """Побудувати інвертований індекс із корпусу та зберегти на диск."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=stderr_console,
            transient=True,
        ) as progress:
            task = progress.add_task("Індексація документів...", total=None)

            def on_doc(doc_id: str) -> None:
                progress.update(task, description=f"Індексація: {doc_id}")

            idx = Index.build_from_corpus(
                corpus_dir,
                limit=limit,
                progress_cb=on_doc,
                workers=workers,
                executor=executor,
            )
            if out.suffix == ".json":
                idx.save_json(out)
            else:
                idx.save_pickle(out)

        stdout_console.print(
            f"[green]Індекс збережено у {out}[/green] "
            f"(документів: {idx.num_docs}, термів: {len(idx)}, "
            f"executor: {executor}, workers: {workers})"
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        stderr_console.print(f"[red]Помилка:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from None


@app.command("search")
def search_cmd(
    index_file: Annotated[Path, typer.Argument(help="Шлях до файлу індексу")],
    query: Annotated[str, typer.Argument(help="Пошуковий запит")],
    scorer: Annotated[ScorerName, typer.Option("--scorer", help="Модель ранжування")] = "bm25",
    engine: Annotated[EngineName, typer.Option("--engine", help="Рушій перетину")] = "merge",
    top: Annotated[int, typer.Option("--top", "-k", help="Кількість результатів")] = 5,
    corpus_dir: Annotated[
        Path, typer.Option("--corpus", help="Директорія корпусу або JSONL-файл для сніпетів")
    ] = Path("data/corpus"),
    as_json: Annotated[
        bool, typer.Option("--json", help="Вивести результати у форматі JSON в stdout")
    ] = False,
) -> None:
    """Виконати ранжований пошук по збереженому індексу."""
    results: list[SearchResult] = []
    try:
        with open_index(index_file) as idx:
            search_engine = SearchEngine(
                idx, scorer=make_scorer(scorer), corpus_dir=corpus_dir
            )
            results = search_engine.search(query, top_k=top, engine=engine)
    except (FileNotFoundError, QuerySyntaxError, ValueError, TypeError) as exc:
        stderr_console.print(f"[red]Помилка:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from None

    if as_json:
        payload = [
            {"rank": i, "doc_id": r.doc_id, "score": round(r.score, 6), "snippet": r.snippet}
            for i, r in enumerate(results, 1)
        ]
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return

    table = Table(title=f"Результати пошуку: '{query}' ({scorer.upper()})")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Doc ID", style="magenta")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Snippet", style="white")

    for rank, res in enumerate(results, 1):
        table.add_row(str(rank), res.doc_id, f"{res.score:.4f}", res.snippet)

    stdout_console.print(table)


@app.command("stats")
def stats_cmd(
    corpus_dir: Annotated[Path, typer.Argument(help="Шлях до директорії корпусу або JSONL")],
    limit: Annotated[int | None, typer.Option("--limit", help="Обмеження документів")] = None,
    top: Annotated[int, typer.Option("--top", help="Кількість топ-термів")] = 10,
) -> None:
    """Показати статистику корпусу у вигляді таблиці Rich."""
    try:
        stats = compute_stats(corpus_dir, limit=limit, top_n=top)
    except FileNotFoundError as exc:
        stderr_console.print(f"[red]Помилка:[/red] {exc}", highlight=False)
        raise typer.Exit(code=1) from None

    table = Table(title=f"Статистика корпусу ({corpus_dir})")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значення", justify="right", style="green")
    table.add_row("Документів", str(stats.doc_count))
    table.add_row("Усього токенів", str(stats.token_count))
    table.add_row("Розмір словника", str(stats.vocab_size))
    stdout_console.print(table)


if __name__ == "__main__":
    app()