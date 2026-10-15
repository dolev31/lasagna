"""No default, example or doc table in the package names a model that is gone.

The package defaulted to ``claude-sonnet-4-20250514`` (retired June 15, 2026),
``gemini-2.0-flash`` (shut down June 1, 2026), ``deepseek-chat`` (retired July
24, 2026) and ``anthropic/claude-3.5-sonnet`` on OpenRouter, and to
``gpt-4o-mini``, which OpenAI shuts down on January 14, 2027. A user who set
only an API key got a 404 or 400 on the first LLM call, and the docs told them
to copy the same ids.

Two lists. A retired id fails today and may appear nowhere in ``src/``. A
deprecated id still answers until its shutdown date, so it may appear only
where the package has to recognize it: the table that builds its requests and
the price table that bills it. Sources, read 2026-10-15:
https://platform.claude.com/docs/en/about-claude/model-deprecations,
https://developers.openai.com/api/docs/deprecations,
https://ai.google.dev/gemini-api/docs/deprecations,
https://api-docs.deepseek.com/news/news260424,
https://console.groq.com/docs/deprecations,
https://docs.together.ai/docs/deprecations,
https://docs.cohere.com/docs/deprecations.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "lasagna"

#: Shipped text: code, settings and the docs that ship in the wheel.
SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".json"}

_END = r"(?![\w.-])"
_DATE = r"(?:-\d{4}-\d{2}-\d{2})?"

#: Requests to these fail today.
RETIRED = {
    "Claude Sonnet 4 / Opus 4 / Opus 4.1": r"claude-(?:sonnet-4|opus-4|opus-4-1)-2025\d{4}",
    "Claude 3.x": r"claude-3[-.]\d*[\w.-]*",
    "OpenRouter Claude 3.5 Sonnet": r"anthropic/claude-3\.5-sonnet",
    "Gemini 2.0 Flash / 1.5": r"gemini-(?:2\.0-flash|1\.5-(?:pro|flash))[\w.-]*",
    "DeepSeek chat / reasoner": r"deepseek-(?:chat|reasoner)" + _END,
}

#: These answer until a published shutdown date (or, for Groq, Together and
#: Cohere, no longer answer new or free-tier accounts).
DEPRECATED = {
    "gpt-4o / gpt-4o-mini": r"gpt-4o(?:-mini)?" + _DATE + _END,
    "gpt-4.1 / gpt-4.1-mini": r"gpt-4\.1(?:-mini)?" + _DATE + _END,
    "gpt-5.1": r"gpt-5\.1" + _DATE + _END,
    "gpt-5.4-nano": r"gpt-5\.4-nano" + _DATE + _END,
    "Claude Sonnet 4.5": r"claude-sonnet-4-5-20250929",
    "Groq Llama 3.3 70B": r"llama-3\.3-70b-versatile",
    "Together Llama 3.3 70B": r"Llama-3\.3-70B-Instruct-Turbo",
    "Cohere Command R+": r"command-r-plus" + _END,
}

#: Where a deprecated id is needed, and why.
MAY_NAME_DEPRECATED = {
    "providers/model_capabilities.py": "builds the requests of models still served",
    "knowledge_evaluation_sdk/cost.py": "prices models until they shut down",
}


def _shipped_files():
    for path in sorted(SRC.rglob("*")):
        if path.suffix in SUFFIXES and path.is_file() and "__pycache__" not in path.parts:
            yield path


def _hits(patterns):
    compiled = {name: re.compile(r"(?<![\w.-])" + p) for name, p in patterns.items()}
    for path in _shipped_files():
        rel = path.relative_to(SRC).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for name, pattern in compiled.items():
                match = pattern.search(line)
                if match:
                    yield rel, f"{rel}:{lineno}: {match.group(0)} ({name})"


def test_no_retired_model_is_named_anywhere():
    offenders = [hit for _, hit in _hits(RETIRED)]
    assert not offenders, "retired models, whose requests fail:\n  " + "\n  ".join(offenders)


def test_deprecated_models_appear_only_where_they_must():
    offenders = [hit for rel, hit in _hits(DEPRECATED) if rel not in MAY_NAME_DEPRECATED]
    assert not offenders, (
        "deprecated models named as a default or an example; use a current one:\n  "
        + "\n  ".join(offenders)
    )
