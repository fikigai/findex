import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger("findex")


def list_corpus_files(corpus_dir: str | Path, limit: int | None = None) -> list[Path]:
    """Повертає відсортований список шляхів до текстових файлів корпусу."""
    path = Path(corpus_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Директорію корпусу не знайдено: {path}")

    files = sorted(p for p in path.rglob("*.txt") if p.is_file())
    if limit is not None:
        return files[:limit]
    return files


def iter_documents(corpus_dir: str | Path) -> Iterator[tuple[str, str]]:
    """Лінивий генератор, що читає текстові документи з директорії по одному."""
    path = Path(corpus_dir)
    for file_path in list_corpus_files(path):
        doc_id = file_path.relative_to(path).as_posix()
        logger.debug("Читання документа: %s", doc_id)
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        yield doc_id, text