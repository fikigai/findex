import json
import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger("findex")


def list_corpus_files(corpus_dir: str | Path, limit: int | None = None) -> list[Path]:
    """Повертає відсортований список шляхів до текстових файлів корпусу."""
    path = Path(corpus_dir)
    if not path.exists():
        raise FileNotFoundError(f"Шлях корпусу не знайдено: {path}")
    if path.is_file():
        return [path]

    files = sorted(p for p in path.rglob("*.txt") if p.is_file())
    if limit is not None:
        return files[:limit]
    return files


def _iter_jsonl_file(jsonl_path: Path) -> Iterator[tuple[str, str]]:
    """Потоково читає документи з JSONL-файлу краулера по одному рядку."""
    with open(jsonl_path, encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            doc_id = str(record.get("url") or record.get("doc_id") or f"line_{idx}")
            title = str(record.get("title", ""))
            text = str(record.get("text", ""))
            full_text = f"{title}\n{text}".strip() if title else text
            yield doc_id, full_text


def iter_documents(corpus_dir: str | Path) -> Iterator[tuple[str, str]]:
    """Лінивий генератор, що читає документи з директорії (.txt/.jsonl) або файлу .jsonl."""
    path = Path(corpus_dir)
    if not path.exists():
        raise FileNotFoundError(f"Директорію або файл корпусу не знайдено: {path}")

    if path.is_file() and path.suffix == ".jsonl":
        yield from _iter_jsonl_file(path)
        return

    if not path.is_dir():
        raise FileNotFoundError(f"Директорію корпусу не знайдено: {path}")

    jsonl_files = sorted(p for p in path.rglob("*.jsonl") if p.is_file())
    for jf in jsonl_files:
        yield from _iter_jsonl_file(jf)

    for file_path in list_corpus_files(path):
        doc_id = file_path.relative_to(path).as_posix()
        logger.debug("Читання документа: %s", doc_id)
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        yield doc_id, text