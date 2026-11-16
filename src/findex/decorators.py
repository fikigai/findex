from functools import wraps
import time
import tracemalloc


def timed(func):
    """Декоратор для вимірювання часу виконання та пікової пам'яті з @functools.wraps."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        elapsed = time.perf_counter() - start_time
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"[TIMED] {func.__name__} виконано за {elapsed:.6f} сек | Пікова пам'ять: {peak_mem / (1024 * 1024):.4f} MB")
        return result
    return wrapper