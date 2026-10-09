"""Token estimation utilities for RAG system.

This module provides accurate token counting for context window management.
Uses tiktoken when available for accurate OpenAI model token counts,
with a fallback heuristic for other scenarios.

Why accurate token counting matters:
1. Context window limits are hard limits - exceeding them causes errors
2. Rough estimates can be off by 20-50% for certain text (code, non-English)
3. RAG systems need to fit retrieved chunks within token budgets
"""

from functools import lru_cache
from loguru import logger

# Try to import tiktoken for accurate token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.debug("tiktoken not installed. Using heuristic token estimation.")


@lru_cache(maxsize=4)
def _get_tokenizer(model: str = "cl100k_base"):
    """Get a cached tokenizer encoder.

    Args:
        model: Tokenizer model name. Common options:
            - "cl100k_base": GPT-4, GPT-3.5-turbo, text-embedding-ada-002
            - "p50k_base": Codex, text-davinci-002/003
            - "r50k_base": GPT-3 (davinci, curie, etc.)

    Returns:
        tiktoken Encoding object, or None if unavailable
    """
    if not TIKTOKEN_AVAILABLE:
        return None

    try:
        return tiktoken.get_encoding(model)
    except Exception as e:
        logger.warning(f"Failed to load tiktoken encoding '{model}': {e}")
        return None


def estimate_tokens(text: str, model: str = "cl100k_base") -> int:
    """Estimate token count for a text string.

    Uses tiktoken for accurate counts when available, falling back to
    a character-based heuristic.

    Args:
        text: Input text to count tokens for
        model: Tokenizer model (default cl100k_base for GPT-4/3.5)

    Returns:
        Estimated token count

    Examples:
        >>> estimate_tokens("Hello, world!")
        4  # Accurate with tiktoken
        >>> estimate_tokens("Hello, world!")  # Without tiktoken
        3  # Heuristic: len("Hello, world!") // 4 = 13 // 4 = 3
    """
    if not text:
        return 0

    # Try accurate counting with tiktoken
    encoder = _get_tokenizer(model)
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception as e:
            logger.debug(f"tiktoken encoding failed, using heuristic: {e}")

    # Fallback: ~4 characters per token for English text
    # This is a rough approximation that works reasonably for most English text
    return len(text) // 4


def estimate_tokens_batch(texts: list[str], model: str = "cl100k_base") -> list[int]:
    """Estimate token counts for multiple texts efficiently.

    Args:
        texts: List of input texts
        model: Tokenizer model

    Returns:
        List of token counts
    """
    encoder = _get_tokenizer(model)

    if encoder is not None:
        try:
            return [len(encoder.encode(text)) if text else 0 for text in texts]
        except Exception as e:
            logger.debug(f"tiktoken batch encoding failed, using heuristic: {e}")

    # Fallback
    return [len(text) // 4 if text else 0 for text in texts]


def count_tokens_for_messages(messages: list[dict], model: str = "gpt-4") -> int:
    """Count tokens for a list of chat messages.

    Accounts for message formatting overhead used by OpenAI's chat API.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Model name for token counting rules

    Returns:
        Total token count including formatting overhead
    """
    if not TIKTOKEN_AVAILABLE:
        # Rough estimate: sum content lengths + overhead per message
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += len(content) // 4 + 4  # ~4 tokens overhead per message
        return total + 3  # +3 for reply priming

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Token overhead varies by model
    if "gpt-4" in model or "gpt-3.5-turbo" in model:
        tokens_per_message = 3  # <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = 1
    else:
        tokens_per_message = 4
        tokens_per_name = -1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value:
                num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # Reply priming
    return num_tokens


def fits_in_context(
    text: str, max_tokens: int, model: str = "cl100k_base", safety_margin: float = 0.95
) -> bool:
    """Check if text fits within a token limit.

    Args:
        text: Text to check
        max_tokens: Maximum token limit
        model: Tokenizer model
        safety_margin: Safety margin (0.95 = use 95% of limit)

    Returns:
        True if text fits within the limit
    """
    effective_limit = int(max_tokens * safety_margin)
    return estimate_tokens(text, model) <= effective_limit


def truncate_to_token_limit(
    text: str, max_tokens: int, model: str = "cl100k_base", truncation_suffix: str = "..."
) -> str:
    """Truncate text to fit within a token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum token limit
        model: Tokenizer model
        truncation_suffix: Suffix to append if truncated

    Returns:
        Truncated text that fits within the limit
    """
    if not text:
        return text

    encoder = _get_tokenizer(model)
    suffix_tokens = len(truncation_suffix) // 4  # Rough estimate

    if encoder is not None:
        try:
            tokens = encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text

            # Reserve space for suffix
            suffix_tokens = len(encoder.encode(truncation_suffix))
            truncated_tokens = tokens[: max_tokens - suffix_tokens]
            return encoder.decode(truncated_tokens) + truncation_suffix
        except Exception as e:
            logger.debug(f"tiktoken truncation failed, using heuristic: {e}")

    # Fallback: estimate 4 chars per token
    char_limit = max_tokens * 4
    if len(text) <= char_limit:
        return text

    return text[: char_limit - len(truncation_suffix)] + truncation_suffix
