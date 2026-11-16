import asyncio
import json
import logging
import os
from pathlib import Path
import time
import psutil
from findex.crawler import CrawlMetrics, crawl


async def start_local_docs_mirror() -> tuple[asyncio.Server, int]:
    """Підіймає локальне дзеркало документації (250 сторінок) з мережевою затримкою 25 мс."""
    seen_429_once: set[str] = set()

    async def handle_client(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            line = await reader.readline()
            if not line:
                writer.close()
                return
            parts = line.decode("latin-1").split()
            path = parts[1] if len(parts) > 1 else "/"
            while True:
                hdr = await reader.readline()
                if not hdr or hdr in (b"\r\n", b"\n"):
                    break

            await asyncio.sleep(0.025)  # імітація мережевого RTT 25 мс

            if path == "/robots.txt":
                body = b"User-agent: *\nDisallow: /admin\nAllow: /\n"
                resp = (
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                    + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
                writer.write(resp)
                await writer.drain()
                return

            if path == "/page_0042" and path not in seen_429_once:
                seen_429_once.add(path)
                body = b"Too Many Requests"
                resp = (
                    b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 0.05\r\n"
                    + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
                writer.write(resp)
                await writer.drain()
                return

            idx = 0
            if path.startswith("/page_"):
                try:
                    idx = int(path.split("_")[1].split("?")[0])
                except ValueError:
                    idx = 0

            links_html = "\n".join(
                f'<a href="/page_{(idx * 5 + k) % 250:04d}">Doc {(idx * 5 + k) % 250}</a>'
                for k in range(1, 8)
            )
            html = (
                f"<html><head><title>Findex Docs Page {idx}</title></head>"
                f"<body><h1>Documentation Section {idx}</h1>"
                f"<p>Asyncio event loop coroutines httpx inverted index bm25 page {idx}.</p>"
                f"{links_html}</body></html>"
            ).encode("utf-8")

            header = (
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                + f"Content-Length: {len(html)}\r\nConnection: close\r\n\r\n".encode()
            )
            writer.write(header + html)
            await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


async def run_benchmark(f_out_path: Path, proc: psutil.Process) -> None:
    server, port = await start_local_docs_mirror()
    seed_url = f"http://127.0.0.1:{port}/page_0000"
    max_pages = 200

    print(f"Локальне дзеркало запущено на {seed_url} (max_pages={max_pages})")
    print(
        f"{'Concurrency':<12} | {'Wall time (s)':<14} | {'Pages/sec':<12} | "
        f"{'Errors':<8} | {'Peak RSS (MB)':<14}"
    )
    print("-" * 70)

    async with server:
        for conc in (1, 5, 20):
            metrics = CrawlMetrics()
            t0 = time.perf_counter()
            peak_rss = proc.memory_info().rss / (1024 * 1024)
            lines_buf: list[str] = []

            async for page in crawl(
                seed_url=seed_url,
                max_pages=max_pages,
                concurrency=conc,
                per_host_concurrency=conc,
                min_host_interval=0.0,
                metrics=metrics,
            ):
                lines_buf.append(
                    json.dumps(
                        {
                            "url": page.url,
                            "status": page.status,
                            "title": page.title,
                            "text": page.text,
                        },
                        ensure_ascii=False,
                    )
                )

            cur_rss = proc.memory_info().rss / (1024 * 1024)
            if cur_rss > peak_rss:
                peak_rss = cur_rss

            wall = time.perf_counter() - t0
            pps = metrics.pages_collected / wall
            await asyncio.to_thread(
                f_out_path.write_text, "\n".join(lines_buf) + "\n", "utf-8"
            )
            print(
                f"{conc:<12} | {wall:<14.4f} | {pps:<12.2f} | "
                f"{metrics.errors:<8} | {peak_rss:<14.1f}"
            )


def main() -> None:
    log_path = Path("crawl.log")
    out_jsonl = Path("crawled_pages.jsonl")
    crawler_logger = logging.getLogger("findex.crawler")
    crawler_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8", delay=True)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    crawler_logger.addHandler(fh)
    proc = psutil.Process(os.getpid())

    try:
        asyncio.run(run_benchmark(out_jsonl, proc), debug=True)
    finally:
        crawler_logger.removeHandler(fh)
        fh.close()
    print(f"\nЛог записано у {log_path}, зібраний корпус у {out_jsonl}")


if __name__ == "__main__":
    main()