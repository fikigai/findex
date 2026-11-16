import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

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


@app.command("index")
def index_cmd(
    corpus_dir: Annotated[Path, typer.Argument(help="Шлях до директорії корпусу")],
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
        Path, typer.Option("--corpus", help="Директорія корпусу для сніпетів")
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
    corpus_dir: Annotated[Path, typer.Argument(help="Шлях до директорії корпусу")],
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