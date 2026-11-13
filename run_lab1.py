from pathlib import Path
from findex.stats import collect_stats_greedy, collect_stats_stream


def main():
    corpus_dir = Path("data/corpus")

    if not corpus_dir.exists():
        print(
            "Спочатку запустіть створення корпусу: uv run python create_corpus.py"
        )
        return

    print("=== Збір статистики потоковим методом (Stream / Generator) ===")
    stream_stats = collect_stats_stream(corpus_dir)
    print(f"Документів: {stream_stats['doc_count']}")
    print(f"Всього токенів: {stream_stats['total_tokens']}")
    print(f"Розмір словника (vocab size): {stream_stats['vocab_size']}")
    print(f"Час виконання: {stream_stats['time_sec']:.6f} сек")
    print(f"Пікова пам'ять: {stream_stats['peak_memory_mb']:.4f} MB")
    print(f"Топ-10 термів: {stream_stats['top_50'][:10]}")

    print("\n" + "=" * 50 + "\n")

    print("=== Збір статистики жадібним методом (Greedy / List) ===")
    greedy_stats = collect_stats_greedy(corpus_dir)
    print(f"Час виконання: {greedy_stats['time_sec']:.6f} сек")
    print(f"Пікова пам'ять: {greedy_stats['peak_memory_mb']:.4f} MB")


if __name__ == "__main__":
    main()