"""Retry utilities with exponential backoff.

Matches cloud-gpu-shopper's polling patterns:
- Initial interval: 15 seconds
- Multiplier: 1.5
- Max interval: 60 seconds
"""

import asyncio
import logging
from typing import Callable, TypeVar, Optional, Type, Tuple
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default backoff parameters (match shopper's provisioner/service.go)
DEFAULT_INITIAL_INTERVAL = 15.0  # DefaultSSHCheckInterval
DEFAULT_MAX_INTERVAL = 60.0  # DefaultSSHMaxInterval
DEFAULT_MULTIPLIER = 1.5  # DefaultSSHBackoffMultiplier
DEFAULT_MAX_ELAPSED_TIME = 8 * 60  # DefaultSSHVerifyTimeout (8 minutes)


async def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: Optional[int] = None,
    max_elapsed_time: Optional[float] = DEFAULT_MAX_ELAPSED_TIME,
    initial_interval: float = DEFAULT_INITIAL_INTERVAL,
    max_interval: float = DEFAULT_MAX_INTERVAL,
    multiplier: float = DEFAULT_MULTIPLIER,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> T:
    """Execute a function with exponential backoff retry.

    Args:
        func: Async function to execute (no arguments)
        max_attempts: Maximum number of attempts (None for unlimited)
        max_elapsed_time: Maximum total time in seconds (None for unlimited)
        initial_interval: Initial wait between retries
        max_interval: Maximum wait between retries
        multiplier: Backoff multiplier
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback(exception, attempt, next_interval)

    Returns:
        Result of the function

    Raises:
        The last exception if all retries fail
    """
    import time

    start_time = time.time()
    attempt = 0
    interval = initial_interval
    last_exception: Optional[Exception] = None

    while True:
        attempt += 1

        # Check max attempts
        if max_attempts is not None and attempt > max_attempts:
            if last_exception:
                raise last_exception
            raise RuntimeError("Max attempts exceeded")

        # Check max elapsed time
        elapsed = time.time() - start_time
        if max_elapsed_time is not None and elapsed >= max_elapsed_time:
            if last_exception:
                raise last_exception
            raise TimeoutError(f"Max elapsed time ({max_elapsed_time}s) exceeded")

        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except retryable_exceptions as e:
            last_exception = e

            # Check if we should continue
            if max_attempts is not None and attempt >= max_attempts:
                raise

            remaining_time = (
                max_elapsed_time - elapsed if max_elapsed_time else float("inf")
            )
            if remaining_time <= 0:
                raise

            # Calculate next interval
            wait_time = min(interval, max_interval, remaining_time)

            if on_retry:
                on_retry(e, attempt, wait_time)
            else:
                logger.debug(
                    f"Attempt {attempt} failed: {e}. Retrying in {wait_time:.1f}s"
                )

            await asyncio.sleep(wait_time)
            interval = min(interval * multiplier, max_interval)


def with_retry(
    max_attempts: int = 3,
    initial_interval: float = 1.0,
    max_interval: float = 30.0,
    multiplier: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator for retry with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        initial_interval: Initial wait between retries
        max_interval: Maximum wait between retries
        multiplier: Backoff multiplier
        retryable_exceptions: Tuple of exception types to retry on
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_attempts=max_attempts,
                initial_interval=initial_interval,
                max_interval=max_interval,
                multiplier=multiplier,
                retryable_exceptions=retryable_exceptions,
            )

        return wrapper

    return decorator


# Rate limit specific retry
async def retry_on_rate_limit(
    func: Callable[[], T],
    max_attempts: int = 5,
    base_delay: float = 5.0,
) -> T:
    """Retry with exponential backoff specifically for rate limiting.

    Uses doubling delays: 5s, 10s, 20s, 40s, 80s
    """
    from .errors import RateLimitError

    delay = base_delay

    for attempt in range(1, max_attempts + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except RateLimitError as e:
            if attempt >= max_attempts:
                raise

            # Use retry_after if provided, otherwise use exponential backoff
            wait_time = e.retry_after if e.retry_after else delay
            logger.warning(
                f"Rate limited (attempt {attempt}/{max_attempts}). "
                f"Waiting {wait_time}s..."
            )
            await asyncio.sleep(wait_time)
            delay *= 2

    raise RuntimeError("Should not reach here")
