import re
import unicodedata
from typing import Iterator

# [^\W_] відповідає будь-якій Unicode-літері або цифрі (включно з кирилицею, латиницею та акцентами як é)
_TOKEN_PATTERN = re.compile(
    r"\b[^\W_]+(?:['’\-][^\W_]+)*\b", re.UNICODE
)


def tokenize(text: str) -> Iterator[str]:
    """Потоковий токенізатор:

    - Нормалізація Unicode (форма NFC)
    - Регістронезалежне зведення через casefold()
    - Виділення термів з урахуванням дефісів, апострофів та цифр.
    """
    normalized_text = unicodedata.normalize("NFC", text)
    folded_text = normalized_text.casefold()

    for match in _TOKEN_PATTERN.finditer(folded_text):
        yield match.group()