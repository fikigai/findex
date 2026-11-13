import re
import unicodedata
from typing import Iterator


def tokenize(text: str) -> Iterator[str]:
    """Потоковий токенізатор: нормалізує Unicode, переводить у нижній регістр

    та виділяє терми за допомогою регулярного виразу.
    """
    # 1. Нормалізація Unicode (форма NFKC)
    normalized_text = unicodedata.normalize("NFKC", text)

    # 2. Переводимо в нижній регістр
    lower_text = normalized_text.lower()

    # 3. Виправляємо регулярний вираз: екрануємо дефіс (\-) та коректно вказуємо апострофи
    token_pattern = re.compile(
        r"\b[a-zа-яіїєґ0-9]+(?:['’\-][a-zа-яіїєґ0-9]+)*\b", re.UNICODE
    )

    for match in token_pattern.finditer(lower_text):
        yield match.group()