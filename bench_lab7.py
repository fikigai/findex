import asyncio
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx
from findex.index import Index


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = math.ceil((p / 100.0) * len(sorted_vals)) - 1
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def run_load_test(
    base_url: str,
    endpoint: str = "/search?q=python&k=5",
    total_requests: int = 300,
    concurrency: int = 20,
) -> tuple[float, float, float, float]:
    latencies_ms: list[float] = []
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url, timeout=15.0) as client:
        # Прогрів з'єднань
        for _ in range(5):
            await client.get(endpoint)

        t0 = time.perf_counter()

        async def one_req() -> None:
            async with sem:
                ts = time.perf_counter()
                resp = await client.get(endpoint)
                resp.raise_for_status()
                latencies_ms.append((time.perf_counter() - ts) * 1000.0)

        await asyncio.gather(*(one_req() for _ in range(total_requests)))
        total_sec = time.perf_counter() - t0

    latencies_ms.sort()
    rps = total_requests / total_sec
    return (
        rps,
        percentile(latencies_ms, 50),
        percentile(latencies_ms, 95),
        percentile(latencies_ms, 99),
    )


async def wait_until_healthy(base_url: str, timeout_s: float = 15.0) -> None:
    deadline = time.perf_counter() + timeout_s
    async with httpx.AsyncClient(base_url=base_url, timeout=2.0) as client:
        while time.perf_counter() < deadline:
            try:
                r = await client.get("/health")
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError(f"Сервер {base_url} не став healthy за {timeout_s}с")


def start_uvicorn_server(port: int, workers: int, index_path: Path, corpus_dir: Path) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["INDEX_PATH"] = str(index_path)
    env["CORPUS_DIR"] = str(corpus_dir)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "findex.web.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            str(workers),
            "--log-level",
            "error",
        ],
        env=env,
    )


async def measure_async_vs_def_event_loop_lag() -> tuple[float, float]:
    """Порівнює блокування циклу подій для 20 одночасних запитів: async def (CPU у loop) vs def (threadpool)."""
    from fastapi import FastAPI

    corpus_dir = Path("data/bench_corpus") if Path("data/bench_corpus").exists() else Path("data/corpus")
    idx = Index.build_from_corpus(corpus_dir, workers=4, executor="threads")

    mini = FastAPI()

    @mini.get("/search_async")
    async def search_async_blocking(q: str = "python") -> dict[str, int]:
        # Синхронний CPU-bound пошук усередині async def блокує event loop
        from findex.search import SearchEngine

        res = SearchEngine(idx, corpus_dir=corpus_dir).search(q, top_k=10)
        return {"count": len(res)}

    @mini.get("/search_def")
    def search_def_threadpool(q: str = "python") -> dict[str, int]:
        # Звичайний def автоматично виноситься FastAPI у threadpool
        from findex.search import SearchEngine

        res = SearchEngine(idx, corpus_dir=corpus_dir).search(q, top_k=10)
        return {"count": len(res)}

    transport = httpx.ASGITransport(app=mini)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(client.get("/search_async?q=python") for _ in range(20)))
        async_time_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        await asyncio.gather(*(client.get("/search_def?q=python") for _ in range(20)))
        def_time_ms = (time.perf_counter() - t1) * 1000.0

    return async_time_ms, def_time_ms


async def main() -> None:
    corpus_dir = Path("data/bench_corpus") if Path("data/bench_corpus").exists() else Path("data/corpus")
    index_path = Path("index.bin")
    if not index_path.exists():
        idx = Index.build_from_corpus(corpus_dir)
        idx.save_pickle(index_path)

    async_ms, def_ms = await measure_async_vs_def_event_loop_lag()
    print(f"20 одночасних запитів (async def у loop vs def у threadpool): async={async_ms:.1f}ms, def={def_ms:.1f}ms\n")

    print(f"{'Конфігурація':<32} | {'RPS (req/s)':<12} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10}")
    print("-" * 84)

    # 1. Локально: 1 воркер (def + threadpool)
    p1 = get_free_port()
    proc1 = start_uvicorn_server(p1, workers=1, index_path=index_path, corpus_dir=corpus_dir)
    try:
        url1 = f"http://127.0.0.1:{p1}"
        await wait_until_healthy(url1)
        rps, p50, p95, p99 = await run_load_test(url1, total_requests=300, concurrency=20)
        print(f"{'1 воркер (def / threadpool)':<32} | {rps:<12.1f} | {p50:<10.2f} | {p95:<10.2f} | {p99:<10.2f}")
    finally:
        proc1.terminate()
        proc1.wait()

    # 2. Локально: 4 воркери (мультипроцесність Uvicorn)
    p4 = get_free_port()
    proc4 = start_uvicorn_server(p4, workers=4, index_path=index_path, corpus_dir=corpus_dir)
    try:
        url4 = f"http://127.0.0.1:{p4}"
        await wait_until_healthy(url4)
        rps, p50, p95, p99 = await run_load_test(url4, total_requests=300, concurrency=20)
        print(f"{'4 воркери (Uvicorn processes)':<32} | {rps:<12.1f} | {p50:<10.2f} | {p95:<10.2f} | {p99:<10.2f}")
    finally:
        proc4.terminate()
        proc4.wait()

    # 3. Якщо передано публічний URL розгорнутого сервісу — міряємо і його
    if len(sys.argv) > 1:
        deployed_url = sys.argv[1].rstrip("/")
        rps, p50, p95, p99 = await run_load_test(deployed_url, total_requests=150, concurrency=15)
        print(f"{'Розгорнутий сервіс (Public)':<32} | {rps:<12.1f} | {p50:<10.2f} | {p95:<10.2f} | {p99:<10.2f}")


if __name__ == "__main__":
    asyncio.run(main())