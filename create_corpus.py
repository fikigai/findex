from pathlib import Path

# Визначаємо шлях до папки корпусу
corpus_dir = Path("data/corpus")
corpus_dir.mkdir(parents=True, exist_ok=True)

# Кілька тестових документів різною мовою з особливими символами (апострофи, дефіси, цифри)
documents = {
    "doc1.txt": (
        "Пошукова система findex. "
        "Розробка програмного забезпечення на Python — це захоплюючий процес. "
        "Пам'ять, швидкість та ефективність алгоритмів грають ключову роль. "
        "Wi-Fi роутер працює на частоті 2.4 ГГц."
    ),
    "doc2.txt": (
        "Project Gutenberg offers thousands of free ebooks. "
        "Reading classic literature helps improve vocabulary and grammar. "
        "Test-system should handle hyphens, apostrophes (don't, let's), and numbers like 2026 correctly."
    ),
    "doc3.txt": (
        "Українська мова має багату лексику. "
        "Комп'ютерні технології стрімко розвиваються. "
        "Студенти групи ПІ-223Б вивчають інженерію програмного забезпечення."
    ),
}

# Записуємо документи у файли
for filename, text in documents.items():
    file_path = corpus_dir / filename
    file_path.write_text(text, encoding="utf-8")
    print(f"Створено файл: {file_path}")

print("\nКорпус успішно створено! Тепер можна протестувати генератор.")