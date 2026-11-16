import re
import unicodedata
from collections.abc import Iterator

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Потоковий токенізатор із NFC-нормалізацією та casefold()."""
    normalized_text = unicodedata.normalize("NFC", text)
    folded_text = normalized_text.casefold()

    for match in _TOKEN_PATTERN.finditer(folded_text):
        yield match.group()