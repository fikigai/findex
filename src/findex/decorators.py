import logging
import time
import tracemalloc
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

logger = logging.getLogger("findex")

P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор для вимірювання часу виконання та пікової пам'яті з логуванням."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start_time
            _, peak_mem = tracemalloc.get_traced_memory()
            if not was_tracing:
                tracemalloc.stop()
            logger.debug(
                "[TIMED] %s виконано за %.6f сек | Пікова пам'ять: %.4f MB",
                func.__name__,
                elapsed,
                peak_mem / (1024 * 1024),
            )

    return wrapper