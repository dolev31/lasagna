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
