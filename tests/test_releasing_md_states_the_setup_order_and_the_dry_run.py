"""RELEASING.md must not claim an order the steps do not need, and must say
the manual release.yml run never publishes (the workflow's publish job runs
only for a published release)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / "RELEASING.md").read_text(encoding="utf-8")
WORKFLOW = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")


def test_the_setup_does_not_claim_a_strict_order():
    assert "in this order" not in TEXT
    assert "deployment tag rules on a private repository need a paid plan" in TEXT
    assert "--visibility public" in TEXT


def test_the_dry_run_is_documented_as_never_publishing():
    assert "if: github.event_name == 'release'" in WORKFLOW
    assert "never publishes" in TEXT


def test_the_constraints_paragraph_is_split_in_two():
    assert "docling-core requires typer<0.27.\n\nUpdate the version in" in TEXT


def test_the_release_notes_have_no_relative_links(tmp_path):
    """The notes are cut from CHANGELOG.md, whose links are relative to the
    repository; on a release page those open nothing. Run the documented
    command and check every link it writes is absolute."""
    import re
    import shutil
    import subprocess

    import pytest

    if not (shutil.which("bash") and shutil.which("perl")):
        pytest.skip("needs bash and perl, as the documented command does")
    start = TEXT.index("{ awk '/^## \\[0\\.1\\.0\\] /")
    end = TEXT.index("> /tmp/notes-0.1.0.md", start)
    out = tmp_path / "notes.md"
    command = TEXT[start:end] + f"> {out}"
    subprocess.run(["bash", "-c", command], cwd=ROOT, check=True)
    notes = out.read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)\s]+)\)", notes)
    assert links, "the 0.1.0 section has no links; the check would prove nothing"
    relative = [link for link in links if not link.startswith(("https://", "http://"))]
    assert relative == []
    assert all("/blob/v0.1.0/" in link for link in links if "dolev31/lasagna" in link)
