import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger("findex.crawler")

DEFAULT_USER_AGENT = "FindexBot/0.6 (+https://github.com/fikigai/findex; mailto:bunca@kai.edu.ua)"


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int
    content: bytes
    content_type: str
    elapsed: float
    attempts: int


@dataclass(frozen=True, slots=True)
class CrawledPage:
    url: str
    status: int
    title: str
    text: str
    links: tuple[str, ...]
    elapsed: float


@dataclass
class CrawlMetrics:
    pages_collected: int = 0
    in_flight: int = 0
    errors: int = 0
    queue_depth: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    @property
    def pages_per_sec(self) -> float:
        elapsed = max(0.001, time.perf_counter() - self.start_time)
        return self.pages_collected / elapsed


def canonicalize_url(url: str) -> str:
    """Канонізує URL: нижній регістр схеми/хоста, без #фрагмента, відсортовані query, єдиний слеш."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if (scheme == "http" and netloc.endswith(":80")) or (
        scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    query_pairs = sorted(parse_qsl(parts.query, keep_blank_values=True))
    query = urlencode(query_pairs)
    return urlunsplit((scheme, netloc, path, query, ""))


class _HTMLContentExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self._ignore_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript"):
            self._ignore_depth += 1
        elif t == "title":
            self._in_title = True
        elif t == "a":
            for k, v in attrs:
                if k.lower() == "href" and v:
                    resolved = urljoin(self.base_url, v.strip())
                    parsed = urlsplit(resolved)
                    if parsed.scheme in ("http", "https") and parsed.netloc:
                        self.links.append(canonicalize_url(resolved))

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript") and self._ignore_depth > 0:
            self._ignore_depth -= 1
        elif t == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignore_depth == 0:
            cleaned = data.strip()
            if cleaned:
                if self._in_title:
                    self.title_parts.append(cleaned)
                self.text_parts.append(cleaned)


def extract_html(base_url: str, html_text: str) -> tuple[str, str, list[str]]:
    """Повертає (title, visible_text, canonical_links) зі сторінки HTML."""
    parser = _HTMLContentExtractor(base_url)
    parser.feed(html_text)
    title = " ".join(parser.title_parts).strip()
    text = " ".join(parser.text_parts).strip()
    return title, text, parser.links


class RobotsCache:
    """Асинхронний кеш правил robots.txt окремо для кожного хоста."""

    def __init__(self, client: httpx.AsyncClient, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self._client = client
        self._user_agent = user_agent
        self._cache: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def can_fetch(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme.lower()}://{parts.netloc.lower()}"
        if origin not in self._locks:
            self._locks[origin] = asyncio.Lock()

        async with self._locks[origin]:
            if origin not in self._cache:
                rp = RobotFileParser()
                robots_url = f"{origin}/robots.txt"
                try:
                    async with asyncio.timeout(5.0):
                        resp = await self._client.get(robots_url)
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
                    elif resp.status_code in (401, 403):
                        rp.parse(["User-agent: *", "Disallow: /"])
                    else:
                        rp.parse(["User-agent: *", "Disallow:"])
                except (TimeoutError, httpx.HTTPError, OSError):
                    rp.parse(["User-agent: *", "Disallow:"])
                self._cache[origin] = rp

            return self._cache[origin].can_fetch(self._user_agent, url)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    timeout: float = 5.0,
    max_retries: int = 3,
    base_backoff: float = 0.2,
) -> FetchResult:
    """Виконує HTTP GET із asyncio.timeout, exponential backoff + jitter і Retry-After."""
    attempts = 0
    while True:
        attempts += 1
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url, follow_redirects=True)
            elapsed = time.perf_counter() - t0
            status = resp.status_code
            content = resp.content
            ctype = resp.headers.get("content-type", "")

            logger.info(
                "FETCH url=%s status=%d bytes=%d time=%.4fs attempt=%d",
                url,
                status,
                len(content),
                elapsed,
                attempts,
            )

            if (status == 429 or 500 <= status < 600) and attempts <= max_retries:
                retry_after_hdr = resp.headers.get("Retry-After")
                if retry_after_hdr is not None:
                    try:
                        delay = float(retry_after_hdr)
                    except ValueError:
                        delay = base_backoff * (2 ** (attempts - 1))
                else:
                    delay = base_backoff * (2 ** (attempts - 1))
                jitter = random.uniform(0.0, delay * 0.25)
                wait_sec = delay + jitter
                logger.warning(
                    "RETRY url=%s status=%d retry_in=%.3fs (attempt %d/%d)",
                    url,
                    status,
                    wait_sec,
                    attempts,
                    max_retries,
                )
                await asyncio.sleep(wait_sec)
                continue

            return FetchResult(
                url=str(resp.url),
                status=status,
                content=content,
                content_type=ctype,
                elapsed=elapsed,
                attempts=attempts,
            )

        except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
            elapsed = time.perf_counter() - t0
            logger.warning(
                "FETCH_ERR url=%s error=%s time=%.4fs attempt=%d",
                url,
                type(exc).__name__,
                elapsed,
                attempts,
            )
            if attempts > max_retries:
                raise
            delay = base_backoff * (2 ** (attempts - 1))
            wait_sec = delay + random.uniform(0.0, delay * 0.25)
            logger.warning(
                "RETRY url=%s reason=%s retry_in=%.3fs (attempt %d/%d)",
                url,
                type(exc).__name__,
                wait_sec,
                attempts,
                max_retries,
            )
            await asyncio.sleep(wait_sec)


class _HostLimiter:
    """Обмежує одночасні запити та забезпечує мінімальну паузу між запитами до одного хоста."""

    def __init__(self, per_host_concurrency: int, min_interval: float) -> None:
        self._per_host = max(1, per_host_concurrency)
        self._min_interval = max(0.0, min_interval)
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._rate_locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    async def acquire_and_wait(self, host: str) -> asyncio.Semaphore:
        if host not in self._semaphores:
            self._semaphores[host] = asyncio.Semaphore(self._per_host)
            self._rate_locks[host] = asyncio.Lock()

        sem = self._semaphores[host]
        await sem.acquire()
        try:
            if self._min_interval > 0.0:
                async with self._rate_locks[host]:
                    now = time.perf_counter()
                    last = self._last_request.get(host, 0.0)
                    wait_for = self._min_interval - (now - last)
                    if wait_for > 0.0:
                        await asyncio.sleep(wait_for)
                    self._last_request[host] = time.perf_counter()
        except BaseException:
            sem.release()
            raise
        return sem


async def crawl(
    seed_url: str,
    max_pages: int = 50,
    concurrency: int = 10,
    per_host_concurrency: int = 5,
    min_host_interval: float = 0.01,
    timeout: float = 5.0,
    max_retries: int = 3,
    base_backoff: float = 0.1,
    same_host_only: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    transport: httpx.AsyncBaseTransport | None = None,
    metrics: CrawlMetrics | None = None,
    on_update: Callable[[CrawlMetrics], None] | None = None,
) -> AsyncIterator[CrawledPage]:
    """Асинхронний генератор обходу сайту на базі TaskGroup, черги та семафорів."""
    if max_pages <= 0:
        return

    stats = metrics if metrics is not None else CrawlMetrics()
    start_canon = canonicalize_url(seed_url)
    seed_host = urlsplit(start_canon).netloc.lower()

    seen: set[str] = {start_canon}
    url_queue: asyncio.Queue[str] = asyncio.Queue()
    url_queue.put_nowait(start_canon)
    stats.queue_depth = url_queue.qsize()

    results_queue: asyncio.Queue[CrawledPage | None] = asyncio.Queue()
    global_sem = asyncio.Semaphore(max(1, concurrency))
    host_limiter = _HostLimiter(per_host_concurrency, min_host_interval)
    stop_event = asyncio.Event()
    active_workers = 0

    headers = {"User-Agent": user_agent}
    async with httpx.AsyncClient(headers=headers, transport=transport, verify=False) as client:
        robots = RobotsCache(client, user_agent=user_agent)

        async def worker() -> None:
            nonlocal active_workers
            while not stop_event.is_set():
                try:
                    current_url = await asyncio.wait_for(url_queue.get(), timeout=0.05)
                except TimeoutError:
                    if url_queue.empty() and active_workers == 0:
                        stop_event.set()
                        break
                    continue

                active_workers += 1
                stats.queue_depth = url_queue.qsize()
                try:
                    if stop_event.is_set():
                        continue

                    if not await robots.can_fetch(current_url):
                        logger.info("ROBOTS_DISALLOW url=%s", current_url)
                        continue

                    host = urlsplit(current_url).netloc.lower()
                    async with global_sem:
                        if stop_event.is_set():
                            continue
                        host_sem = await host_limiter.acquire_and_wait(host)
                        stats.in_flight += 1
                        if on_update:
                            on_update(stats)
                        try:
                            res = await fetch(
                                client,
                                current_url,
                                timeout=timeout,
                                max_retries=max_retries,
                                base_backoff=base_backoff,
                            )
                        except (TimeoutError, httpx.HTTPError, OSError, ValueError) as exc:
                            stats.errors += 1
                            logger.error("FAIL url=%s err=%s", current_url, exc)
                            if on_update:
                                on_update(stats)
                            continue
                        finally:
                            stats.in_flight -= 1
                            host_sem.release()

                    if res.status >= 400:
                        stats.errors += 1
                        if on_update:
                            on_update(stats)
                        continue

                    if "html" not in res.content_type.lower() and res.content_type != "":
                        continue

                    if stop_event.is_set() or stats.pages_collected >= max_pages:
                        stop_event.set()
                        continue

                    html_str = res.content.decode("utf-8", errors="ignore")
                    title, text, links = extract_html(res.url, html_str)
                    page = CrawledPage(
                        url=canonicalize_url(res.url),
                        status=res.status,
                        title=title,
                        text=text,
                        links=tuple(links),
                        elapsed=res.elapsed,
                    )

                    stats.pages_collected += 1
                    await results_queue.put(page)

                    if stats.pages_collected >= max_pages:
                        stop_event.set()
                    else:
                        for link in links:
                            link_host = urlsplit(link).netloc.lower()
                            if same_host_only and link_host != seed_host:
                                continue
                            if link not in seen:
                                seen.add(link)
                                url_queue.put_nowait(link)

                    stats.queue_depth = url_queue.qsize()
                    if on_update:
                        on_update(stats)
                finally:
                    active_workers -= 1
                    url_queue.task_done()
                    if url_queue.empty() and active_workers == 0:
                        stop_event.set()

        async def supervisor() -> None:
            tasks: list[asyncio.Task[None]] = []
            try:
                async with asyncio.TaskGroup() as tg:
                    for _ in range(max(1, concurrency)):
                        tasks.append(tg.create_task(worker()))
                    await stop_event.wait()
                    for t in tasks:
                        t.cancel()
            except* asyncio.CancelledError:
                pass
            finally:
                await results_queue.put(None)

        sup_task = asyncio.create_task(supervisor())
        try:
            while True:
                item = await results_queue.get()
                if item is None:
                    break
                yield item
        finally:
            stop_event.set()
            if not sup_task.done():
                sup_task.cancel()
            await asyncio.gather(sup_task, return_exceptions=True)