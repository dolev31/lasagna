"""The README BibTeX entry must name the same paper as CITATION.cff.

The entry used the key `levy2026compositionality` while the paper is titled
"Retrieval Augmented Generation Compositionality" and the key convention is
the first title word. The key is now `levy2026retrieval`; the title and
authors must keep matching `preferred-citation` in CITATION.cff.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _bibtex() -> str:
    match = re.search(r"```bibtex\n(.*?)```", (ROOT / "README.md").read_text(), re.S)
    assert match, "README.md has no bibtex block"
    return match.group(1)


def test_the_bibtex_key_is_year_and_first_title_word():
    entry = _bibtex()
    key = re.match(r"@misc\{([^,]+),", entry).group(1)
    title = re.search(r"title\s*=\s*\{(.*?)\}", entry).group(1)
    assert key == f"levy2026{title.split()[0].lower()}"


def test_the_bibtex_title_matches_the_citation_file():
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text())["preferred-citation"]
    title = re.search(r"title\s*=\s*\{(.*?)\}", _bibtex()).group(1)
    assert title == cff["title"]
