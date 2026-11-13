import re
import unicodedata
from typing import Iterator


def tokenize(text: str) -> Iterator[str]:
    """Потоковий токенізатор:

    - Нормалізація Unicode (форма NFC)
    - Регістронезалежне зведення через casefold()
    - Виділення термів з урахуванням дефісів, апострофів та цифр.
    """
    # 1. Нормалізація Unicode (форма NFC за вимогами курсу)
    normalized_text = unicodedata.normalize("NFC", text)

    # 2. Aggressive case-folding замість звичайного lower()
    folded_text = normalized_text.casefold()

    # 3. Регулярний вираз для слів, цифр, дефісів та апострофів
    token_pattern = re.compile(
        r"\b[a-zа-яіїєґ0-9]+(?:['’\-][a-zа-яіїєґ0-9]+)*\b", re.UNICODE
    )

    for match in token_pattern.finditer(folded_text):
        yield match.group()