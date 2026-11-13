from pathlib import Path
from typing import Generator, Tuple


def iter_documents(corpus_dir: str | Path) -> Generator[Tuple[str, str], None, None]:
    """Генератор, який рекурсивно обходить корпус і повертає документи по одному.

    Повертає кортеж: (doc_id, text)
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        return

    # Шукаємо всі текстові файли в корпусі
    for file_path in corpus_path.rglob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            doc_id = str(file_path.relative_to(corpus_path))
            yield doc_id, text
        except Exception as e:
            print(f"Помилка читання файлу {file_path}: {e}")