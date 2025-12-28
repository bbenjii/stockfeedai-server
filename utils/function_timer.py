import time
from functools import wraps
import logging
from utils.logger import logger

def function_timer(func):
    """
    A decorator that prints the time a function takes to execute.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter() # Use perf_counter for high-resolution timing
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"\n=== '{func.__name__}' completed in {elapsed_time:.4f} s ===")
        logger.info(f"=== '{func.__name__}' completed in {elapsed_time:.4f} s ===")
        return result
    return wrapper
    
def function_timer2(func):
    """
    A decorator that prints the time a function takes to execute.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter() # Use perf_counter for high-resolution timing
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"\n=== '{func.__name__}' completed in {elapsed_time:.4f} s ===")
        logger.info(f"=== '{func.__name__}' completed in {elapsed_time:.4f} s ===")
        return result, elapsed_time
    return wrapper