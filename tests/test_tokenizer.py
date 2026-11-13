from findex.tokenizer import tokenize


def test_tokenize_basic():
    text = "Пошукова система findex. Розробка 2026 року."
    tokens = list(tokenize(text))
    assert "пошукова" in tokens
    assert "система" in tokens
    assert "findex" in tokens
    assert "2026" in tokens


def test_tokenize_unicode_normalization():
    # Перевірка нормалізації Unicode (наприклад, латинські/кириличні літери)
    text = "Wi-Fi та пам'ять"
    tokens = list(tokenize(text))
    assert "wi-fi" in tokens
    assert "пам'ять" in tokens or "пам" in tokens