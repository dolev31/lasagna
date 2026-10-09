"""Retry utilities for Knowledge SDK.

Provides retry decorators with exponential backoff for handling
transient failures in async operations.

Example:
    ```python
    from lasagna.sdk.utils.retry import with_retry
    from lasagna.sdk.exceptions import BackendError

    @with_retry(max_attempts=3, exceptions=(BackendError, ConnectionError))
    async def search_with_retry(query: str) -> SearchResults:
        return await backend.search(query)
    ```
"""

import asyncio
from functools import wraps
from typing import Type, Tuple, Callable, TypeVar, Any, Optional
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """Retry decorator for async functions with exponential backoff.

    Retries a function on failure with configurable attempts and delays.
    Uses exponential backoff to increase delay between retries.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
        backoff: Multiplier for delay after each retry (default: 2.0)
        max_delay: Maximum delay between retries (default: 60.0)
        exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called on each retry with (exception, attempt)

    Returns:
        Decorated async function with retry logic

    Example:
        ```python
        @with_retry(max_attempts=3, exceptions=(ConnectionError,))
        async def fetch_data():
            return await api.get_data()

        # With custom callback
        def log_retry(exc, attempt):
            logger.warning(f"Retry {attempt}: {exc}")

        @with_retry(on_retry=log_retry)
        async def fetch_with_logging():
            return await api.get_data()
        ```
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts:
                        # Log retry attempt
                        logger.debug(f"Retry {attempt}/{max_attempts} for {func.__name__}: {e}")

                        # Call retry callback if provided
                        if on_retry:
                            on_retry(e, attempt)

                        # Wait before retry
                        await asyncio.sleep(current_delay)

                        # Increase delay with backoff, capped at max_delay
                        current_delay = min(current_delay * backoff, max_delay)
                    else:
                        logger.warning(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            # All attempts failed, raise the last exception
            raise last_exception

        return wrapper

    return decorator


def with_retry_sync(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """Retry decorator for sync functions with exponential backoff.

    Same as with_retry but for synchronous functions.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called on each retry

    Returns:
        Decorated sync function with retry logic
    """
    import time

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts:
                        logger.debug(f"Retry {attempt}/{max_attempts} for {func.__name__}: {e}")

                        if on_retry:
                            on_retry(e, attempt)

                        time.sleep(current_delay)
                        current_delay = min(current_delay * backoff, max_delay)
                    else:
                        logger.warning(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper

    return decorator


class RetryConfig:
    """Configuration for retry behavior.

    Provides a reusable configuration object for retry settings.

    Example:
        ```python
        config = RetryConfig(max_attempts=5, delay=0.5)

        @config.decorator()
        async def my_func():
            ...
        ```
    """

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        max_delay: float = 60.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        """Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            delay: Initial delay between retries in seconds
            backoff: Multiplier for delay after each retry
            max_delay: Maximum delay between retries
            exceptions: Tuple of exception types to retry on
        """
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.max_delay = max_delay
        self.exceptions = exceptions

    def decorator(
        self,
        on_retry: Optional[Callable[[Exception, int], None]] = None,
    ) -> Callable:
        """Create a retry decorator with this configuration.

        Args:
            on_retry: Optional callback for retry events

        Returns:
            Retry decorator
        """
        return with_retry(
            max_attempts=self.max_attempts,
            delay=self.delay,
            backoff=self.backoff,
            max_delay=self.max_delay,
            exceptions=self.exceptions,
            on_retry=on_retry,
        )

    def sync_decorator(
        self,
        on_retry: Optional[Callable[[Exception, int], None]] = None,
    ) -> Callable:
        """Create a sync retry decorator with this configuration.

        Args:
            on_retry: Optional callback for retry events

        Returns:
            Sync retry decorator
        """
        return with_retry_sync(
            max_attempts=self.max_attempts,
            delay=self.delay,
            backoff=self.backoff,
            max_delay=self.max_delay,
            exceptions=self.exceptions,
            on_retry=on_retry,
        )


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# Default retry config for backend operations
RETRY_BACKEND = RetryConfig(
    max_attempts=3,
    delay=0.5,
    backoff=2.0,
    max_delay=10.0,
)

# Retry config for search operations
RETRY_SEARCH = RetryConfig(
    max_attempts=2,
    delay=0.2,
    backoff=2.0,
    max_delay=5.0,
)

# Aggressive retry for critical operations
RETRY_CRITICAL = RetryConfig(
    max_attempts=5,
    delay=1.0,
    backoff=2.0,
    max_delay=30.0,
)

# Quick retry for fast operations
RETRY_QUICK = RetryConfig(
    max_attempts=2,
    delay=0.1,
    backoff=2.0,
    max_delay=1.0,
)


__all__ = [
    "with_retry",
    "with_retry_sync",
    "RetryConfig",
    "RETRY_BACKEND",
    "RETRY_SEARCH",
    "RETRY_CRITICAL",
    "RETRY_QUICK",
]
