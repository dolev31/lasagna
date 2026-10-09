"""The package version, read from installed distribution metadata.

Single-sourced from ``pyproject.toml``. It was a hard-coded literal here *and*
in ``pyproject.toml`` *and* in ``CITATION.cff``, with nothing keeping the three
in step -- and ``core/provenance.py`` read this literal, so every results bundle
recorded the string in this file rather than the version actually installed. A
wheel built at one version and this file edited to another would have misreported
provenance without a single test noticing.

Falls back to the editable-install case, where the distribution may not be
registered yet.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lasagna-rag")
except PackageNotFoundError:  # pragma: no cover - source checkout, not installed
    __version__ = "0.0.0+unknown"


def _as_info(raw: str) -> tuple:
    """``"0.1.0a1"`` -> ``(0, 1, 0)``. Pre-release suffixes are dropped.

    Kept tolerant on purpose: this is a convenience, and a version scheme it
    cannot parse must not break an import.
    """
    parts = []
    for chunk in raw.split(".")[:3]:
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


__version_info__ = _as_info(__version__)
