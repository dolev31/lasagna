"""`scripts/check_release.py` gates the PyPI upload, so it has to work before
the one run that matters: CI cannot rehearse a real release."""

import importlib.util
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_release", REPO_ROOT / "scripts" / "check_release.py"
)
check_release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_release)

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

VERSION = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
    "version"
]
TAG = f"v{VERSION}"


def _copy_tree(tmp_path: Path) -> Path:
    for name in ("pyproject.toml", "CHANGELOG.md", "CITATION.cff", "README.md"):
        shutil.copy(REPO_ROOT / name, tmp_path / name)
    for name in check_release.NOTEBOOK_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / name, tmp_path / name)
    (tmp_path / "constraints").mkdir()
    shutil.copy(REPO_ROOT / "constraints" / f"{VERSION}.txt", tmp_path / "constraints")
    return tmp_path


def _stamp(root: Path, changelog_date: str, citation_date: str) -> None:
    log = root / "CHANGELOG.md"
    text = log.read_text(encoding="utf-8")
    date = check_release._changelog_date(text, VERSION)
    log.write_text(
        text.replace(f"## [{VERSION}] - {date}", f"## [{VERSION}] - {changelog_date}"),
        encoding="utf-8",
    )
    cff = root / "CITATION.cff"
    old = check_release._cff_field(cff.read_text(encoding="utf-8"), "date-released")
    cff.write_text(
        cff.read_text(encoding="utf-8").replace(
            f'date-released: "{old}"', f'date-released: "{citation_date}"'
        ),
        encoding="utf-8",
    )


def test_the_tree_agrees_with_its_own_version():
    """Every version statement in the tree names the version pyproject declares.
    The date may still be the placeholder until release day."""
    assert check_release.problems(TAG, allow_placeholder=True) == []


def test_a_tag_for_another_version_is_refused():
    found = check_release.problems("v9.9.9", allow_placeholder=True)
    assert any("does not match pyproject.toml" in p for p in found), found


def test_the_placeholder_date_is_refused_at_release(tmp_path):
    root = _copy_tree(tmp_path)
    _stamp(root, check_release.PLACEHOLDER, "2026-01-01")
    found = check_release.problems(TAG, root=root)
    assert any("stamp the date" in p for p in found), found


def test_changelog_and_citation_dates_must_agree(tmp_path):
    root = _copy_tree(tmp_path)
    _stamp(root, "2026-11-02", "2026-11-01")
    found = check_release.problems(TAG, root=root)
    assert found == [
        "CITATION.cff date-released is '2026-11-01', CHANGELOG.md says '2026-11-02'"
    ], found

    _stamp(root, "2026-11-02", "2026-11-02")
    assert check_release.problems(TAG, root=root) == []


def test_the_notebook_installs_the_release_it_documents(tmp_path):
    """The notebook installed whatever PyPI had newest, so its pinned output
    ("check bundle": "VALID") would have stopped matching with the next release."""
    for name in check_release.NOTEBOOK_FILES:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert f"lasagna-rag[eval]=={VERSION}" in text, name

    root = _copy_tree(tmp_path)
    script = root / check_release.NOTEBOOK_FILES[1]
    script.write_text(
        script.read_text(encoding="utf-8").replace(
            f"lasagna-rag[eval]=={VERSION}", "lasagna-rag[eval]"
        ),
        encoding="utf-8",
    )
    found = check_release.problems(TAG, root=root, allow_placeholder=True)
    assert found == [
        f"{check_release.NOTEBOOK_FILES[1]} does not install lasagna-rag[eval]=={VERSION}"
    ], found
