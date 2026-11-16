from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import threading
import time

try:
    import psutil  # type: ignore[import-not-found]
except ImportError:
    psutil = None

from findex.corpus import list_corpus_files
from findex.index import _chunk_tasks, build_partial, merge_indices

RESULTS_313T_FILE = Path("bench_313t_results.json")


def get_rss_mb() -> float:
    """Повертає RSS поточного процесу в МБ без використання tracemalloc (щоб не блокувати потоки)."""
    if psutil is not None:
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    if sys.platform == "win32":

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return float(pmc.WorkingSetSize) / (1024 * 1024)
    return 0.0


def ensure_bench_corpus(corpus_dir: Path, num_docs: int = 320, lines_per_doc: int = 550) -> None:
    marker = corpus_dir / f".marker_{num_docs}_{lines_per_doc}"
    if marker.exists():
        return
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    base_words = [
        "python", "concurrency", "parallelism", "gil", "threads", "processes",
        "amdahl", "pickle", "serialization", "inverted", "index", "search",
        "bm25", "tfidf", "ranking", "tokenizer", "unicode", "generator",
        "memory", "performance", "benchmark", "algorithm", "structure",
    ]
    for i in range(num_docs):
        lines = [
            f"Doc {i} row {j} analyzes {base_words[(i + j) % len(base_words)]} "
            f"and {base_words[(i * 3 + j) % len(base_words)]} token_{(i + j) % 40}."
            for j in range(lines_per_doc)
        ]
        (corpus_dir / f"doc_{i:04d}.txt").write_text("\n".join(lines), encoding="utf-8")
    marker.write_text("ok", encoding="utf-8")


def run_single_measurement(
    tasks: list[tuple[str, Path]],
    executor: str,
    workers: int,
    pool: ThreadPoolExecutor | ProcessPoolExecutor | None = None,
) -> tuple[float, float, float, float]:
    chunks = _chunk_tasks(tasks, workers if executor != "serial" else 1)

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    if executor == "serial":
        partials = [build_partial(c) for c in chunks]
    elif pool is not None:
        partials = list(pool.map(build_partial, chunks))
    else:
        raise ValueError(executor)

    t_merge0 = time.perf_counter()
    _ = merge_indices(partials)
    merge_time = time.perf_counter() - t_merge0

    wall_time = time.perf_counter() - t0_wall
    cpu_time = time.process_time() - t0_cpu
    rss_mb = get_rss_mb()
    return wall_time, cpu_time, rss_mb, merge_time


def demo_race_condition() -> None:
    unsafe_counter = {"count": 0}
    safe_counter = {"count": 0}
    lock = threading.Lock()

    def worker_unsafe(n: int) -> None:
        for _ in range(n):
            val = unsafe_counter["count"]
            unsafe_counter["count"] = val + 1

    def worker_safe(n: int) -> None:
        for _ in range(n):
            with lock:
                safe_counter["count"] += 1

    threads = [threading.Thread(target=worker_unsafe, args=(50_000,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    threads_safe = [threading.Thread(target=worker_safe, args=(50_000,)) for _ in range(4)]
    for t in threads_safe:
        t.start()
    for t in threads_safe:
        t.join()

    print(
        f"[Race Demo] Очікувалось: 200000 | Без Lock: {unsafe_counter['count']} "
        f"| З Lock: {safe_counter['count']}"
    )


def main() -> None:
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    cpu_count = os.cpu_count() or 4
    print(f"Python: {sys.version.split()[0]} | sys._is_gil_enabled() = {gil_enabled}")
    print(f"CPU: {platform.processor()} | Логічних ядер (os.cpu_count()): {cpu_count}\n")

    corpus_dir = Path("data/bench_corpus")
    ensure_bench_corpus(corpus_dir)
    files = list_corpus_files(corpus_dir)
    tasks = [(f.relative_to(corpus_dir).as_posix(), f) for f in files]

    worker_counts = sorted({1, 2, 4, 8, cpu_count})
    results: dict[str, dict[int, tuple[float, float, float, float, float]]] = {
        "threads": {},
        "processes": {},
    }

    serial_runs = [run_single_measurement(tasks, "serial", 1) for _ in range(3)]
    base_wall = statistics.median(r[0] for r in serial_runs)
    base_cpu = statistics.median(r[1] for r in serial_runs)
    base_rss = statistics.median(r[2] for r in serial_runs)
    base_merge = statistics.median(r[3] for r in serial_runs)
    print(
        f"Serial (1 worker): Wall={base_wall:.4f}s | CPU={base_cpu:.4f}s | "
        f"RSS={base_rss:.1f}MB | Merge={base_merge:.4f}s | Speedup=1.00x\n"
    )

    print(
        f"{'Executor':<12} | {'Workers':<7} | {'Wall (s)':<9} | {'CPU (s)':<9} | "
        f"{'RSS (MB)':<9} | {'Merge (s)':<9} | {'Speedup':<8}"
    )
    print("-" * 78)

    for executor in ("threads", "processes"):
        for w in worker_counts:
            pool_cls = ThreadPoolExecutor if executor == "threads" else ProcessPoolExecutor
            with pool_cls(max_workers=w) as pool:
                # Прогрів пулу воркерів (особливо важливо для spawn на Windows)
                _ = list(pool.map(build_partial, _chunk_tasks(tasks[:w], w)))
                runs = [run_single_measurement(tasks, executor, w, pool=pool) for _ in range(3)]

            med_wall = statistics.median(r[0] for r in runs)
            med_cpu = statistics.median(r[1] for r in runs)
            med_rss = statistics.median(r[2] for r in runs)
            med_merge = statistics.median(r[3] for r in runs)
            speedup = base_wall / med_wall if med_wall > 0 else 0.0
            results[executor][w] = (med_wall, med_cpu, med_rss, med_merge, speedup)
            print(
                f"{executor:<12} | {w:<7} | {med_wall:<9.4f} | {med_cpu:<9.4f} | "
                f"{med_rss:<9.1f} | {med_merge:<9.4f} | {speedup:<7.2f}x"
            )

    print()
    demo_race_condition()

    if not gil_enabled:
        RESULTS_313T_FILE.write_text(
            json.dumps({str(w): results["threads"][w][4] for w in worker_counts}),
            encoding="utf-8",
        )
        print(f"\nДані прискорення 3.13t збережено у {RESULTS_313T_FILE}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        xs = worker_counts
        plt.plot(xs, xs, "--", color="gray", label="Ідеальне лінійне прискорення (y = x)")
        plt.plot(
            xs,
            [results["threads"][w][4] for w in xs],
            "-o",
            label="Threads (3.12, GIL=True)",
        )
        plt.plot(
            xs,
            [results["processes"][w][4] for w in xs],
            "-s",
            label="Processes (spawn + pickle)",
        )
        if RESULTS_313T_FILE.exists():
            data_313t = json.loads(RESULTS_313T_FILE.read_text(encoding="utf-8"))
            plt.plot(
                xs,
                [float(data_313t[str(w)]) for w in xs],
                "-^",
                label="Threads (3.13t, GIL=False)",
            )
        plt.xlabel("Кількість воркерів (Workers)")
        plt.ylabel("Прискорення (Speedup = T1 / TN)")
        plt.title(f"Прискорення індексації Findex (AMD64, Ядер: {cpu_count})")
        plt.xticks(xs)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("speedup_lab5.png", dpi=150)
        print("\nГрафік збережено у speedup_lab5.png")
    except ImportError:
        print("\nMatplotlib не встановлено у цьому оточенні, пропуск побудови графіка.")


if __name__ == "__main__":
    main()