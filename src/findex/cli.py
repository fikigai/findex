import argparse
from pathlib import Path
import time
import tracemalloc
from findex.index import InvertedIndex


def main_index():
    parser = argparse.ArgumentParser(description="Будування інвертованого індексу findex")
    parser.add_argument("corpus_dir", type=str, help="Шлях до корпусу текстів")
    parser.add_argument("--out", type=str, default="index.bin", help="Файл збереження індексу")
    parser.add_argument("--limit", type=int, default=None, help="Обмежити кількість документів")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.time()

    print(f"Будуємо індекс з корпусу: {args.corpus_dir}...")
    index = InvertedIndex.build_from_corpus(args.corpus_dir, limit=args.limit)
    index.save_pickle(args.out)

    elapsed = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Індекс успішно збережено у {args.out}")
    print(f"Документів в індексі: {len(index.docs)}")
    print(f"Унікальних термів (словник): {len(index.index)}")
    print(f"Час побудови: {elapsed:.4f} сек")
    print(f"Пікова пам'ять: {peak_mem / (1024 * 1024):.2f} MB")


def main_search():
    parser = argparse.ArgumentParser(description="Пошук по збереженому індексу findex")
    parser.add_argument("index_file", type=str, help="Шлях до файлу індексу")
    parser.add_argument("query", type=str, help="Пошуковий запит (терми через пробіл)")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Рушій пошуку")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.time()

    index = InvertedIndex.load_pickle(args.index_file)
    terms = args.query.lower().split()

    if args.engine == "merge":
        results = index.search_merge(terms)
    else:
        results = index.search_set(terms)

    elapsed = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Результати пошуку ({args.engine}): {results}")
    print(f"Знайдено документів: {len(results)}")
    print(f"Час пошуку: {elapsed:.6f} сек")
    print(f"Пікова пам'ять: {peak_mem / (1024 * 1024):.2f} MB")