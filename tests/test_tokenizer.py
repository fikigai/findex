import unicodedata
from findex.tokenizer import tokenize


def test_tokenize_comprehensive():
    # 1. Базові слова та регістр (casefold)
    text1 = "Hello FINDEX 2026!"
    assert list(tokenize(text1)) == ["hello", "findex", "2026"]

    # 2. Українська кирилиця
    text2 = "Пошукова система findex — українська мова."
    tokens2 = list(tokenize(text2))
    assert "пошукова" in tokens2
    assert "система" in tokens2
    assert "українська" in tokens2

    # 3. Комбінуючий знак (Combining-mark case: é як e + \u0301)
    combining_text = "café"  # може містити комбінуючий символ
    normalized_combining = unicodedata.normalize("NFC", combining_text)
    assert list(tokenize(normalized_combining)) == ["café"]

    # 4. Апострофи та дефіси (політика токенізатора)
    text3 = "Wi-Fi роутер та пам'ять студента"
    tokens3 = list(tokenize(text3))
    assert "wi-fi" in tokens3
    assert "пам'ять" in tokens3

    # 5. Порожній рядок
    assert list(tokenize("")) == []