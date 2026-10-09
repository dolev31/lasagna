"""Security validation helpers for RAG Backend.

Input validation functions to prevent injection attacks, path traversal,
and other security issues in user-provided data.
"""

import os
import re


def is_valid_session_id(session_id: str) -> bool:
    """Validate session ID format to prevent injection attacks.

    Valid format: alphanumeric characters, hyphens, and underscores only.
    Maximum length: 128 characters.

    Args:
        session_id: Session ID to validate

    Returns:
        True if valid format, False otherwise
    """
    if not session_id or not isinstance(session_id, str):
        return False
    # Allow alphanumeric, hyphen, underscore, and period (for UUIDs and common formats)
    # Maximum 128 characters to prevent DoS
    pattern = r"^[a-zA-Z0-9_\-\.]{1,128}$"
    return bool(re.match(pattern, session_id))


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    Removes:
    - Path separators (/, \\)
    - Null bytes
    - Parent directory references (..)
    - Leading dots (hidden files)

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem operations
    """
    if not filename:
        return "unnamed_file"

    # Remove null bytes and path separators
    sanitized = re.sub(r"[\x00/\\]", "", filename)

    # Remove parent directory references
    sanitized = sanitized.replace("..", "")

    # Remove leading dots (but keep extension dots)
    sanitized = sanitized.lstrip(".")

    # Replace any remaining problematic characters with underscore
    sanitized = re.sub(r'[<>:"|?*]', "_", sanitized)

    # Ensure we have a valid filename
    if not sanitized or sanitized.isspace():
        return "unnamed_file"

    # Limit length to prevent filesystem issues
    max_length = 255
    if len(sanitized) > max_length:
        # Preserve extension if present
        name, ext = os.path.splitext(sanitized)
        if ext:
            sanitized = name[: max_length - len(ext)] + ext
        else:
            sanitized = sanitized[:max_length]

    return sanitized


def escape_milvus_string(value: str) -> str:
    """Escape a string value for safe use in Milvus filter expressions.

    Escapes double quotes and backslashes to prevent injection.

    Args:
        value: String value to escape

    Returns:
        Escaped string safe for Milvus filter expressions
    """
    if not value:
        return ""
    # Escape backslashes first, then double quotes
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return escaped


def is_valid_document_id(document_id: str) -> bool:
    """Validate document ID matches the auto-generated format.

    Valid format: doc_<16 hex chars> (e.g., doc_abc123def456789a).
    Use this only for checking if an ID was auto-generated.
    For general ID validation, use ``is_safe_document_id`` instead.

    Args:
        document_id: Document ID to validate

    Returns:
        True if valid auto-generated format, False otherwise
    """
    if not document_id or not isinstance(document_id, str):
        return False
    # Match format: doc_ followed by exactly 16 hex characters
    pattern = r"^doc_[a-f0-9]{16}$"
    return bool(re.match(pattern, document_id))


def is_safe_document_id(document_id: str) -> bool:
    """Check that a document ID is safe for use in storage and queries.

    Permits custom IDs (e.g. benchmark qrels like ``TREC-2020-doc-42``)
    while rejecting values that could cause path traversal, injection,
    or DoS across *all* storage backends (local, S3, fsspec).

    Rejected patterns:
    - Empty, non-string, or excessively long (>512 chars)
    - Path separators (``/``, ``\\``) — document_id is used as a
      directory/prefix segment in every storage backend
    - Parent-directory traversal (``..``)
    - Null bytes

    Args:
        document_id: Document ID to validate

    Returns:
        True if safe, False otherwise
    """
    if not document_id or not isinstance(document_id, str):
        return False
    if len(document_id) > 512:
        return False
    # Reject path separators and traversal sequences — document_id is
    # used as a directory/prefix in all storage backends.
    if "/" in document_id or "\\" in document_id:
        return False
    if ".." in document_id:
        return False
    if "\0" in document_id:
        return False
    return True
