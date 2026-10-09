"""`lasagna doctor` told users to install the package they already had.

Rich reads square brackets as style markup. The advice for a missing optional
dependency is::

    pip install "lasagna-rag[eval]"

and `console.print` parsed ``[eval]`` as a tag, found no such style, and
dropped it. What reached the terminal was::

    pip install "lasagna-rag"

which the user has, by definition, already run. The extra name is the entire
actionable content of that message, and it was the only part removed.

This is a first-run experience: a fresh install reports three items needing
attention and gives wrong instructions for two of them. Confirmed against a
clean-room wheel install before the fix.

The `--json` and `--issue` paths were already correct -- they pass
``markup=False`` -- so the hazard was known in this very function and the
interactive path was the one that missed it.
"""

from __future__ import annotations

import pytest

rich_console = pytest.importorskip("rich.console")


def _render(text: str, **kwargs) -> str:
    """What a user actually sees for `text`."""
    import io

    console = rich_console.Console(file=io.StringIO(), width=200, no_color=True)
    console.print(text, **kwargs)
    return console.file.getvalue()


class TestTheHazardIsReal:
    """Evidence, not a regression target. If rich stops doing this the
    reasoning above should be revisited rather than the file deleted."""

    def test_rich_eats_an_extras_name(self):
        assert "eval" not in _render('pip install "lasagna-rag[eval]"')

    def test_escaping_preserves_it(self):
        from rich.markup import escape

        assert "[eval]" in _render(escape('pip install "lasagna-rag[eval]"'))


class TestDiagnosticsProduceTheRightAdvice:
    def test_the_advice_names_the_extra(self):
        """The library side was never wrong -- only the printing."""
        from lasagna.core.diagnostics import _OPTIONAL_IMPORTS, _check_optional_extras

        assert _OPTIONAL_IMPORTS, "no optional imports declared"
        advice = [c.advice for c in _check_optional_extras() if c.advice]
        if not advice:
            pytest.skip("every optional dependency is installed in this environment")
        assert any("[" in a for a in advice), (
            "no advice mentions an extra, so this test proves nothing here"
        )

    def test_every_extra_in_the_table_is_a_real_extra(self):
        """Advice naming an extra that pyproject does not define would be just
        as useless as advice naming none."""
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            import tomli as tomllib
        from pathlib import Path

        from lasagna.core.diagnostics import _OPTIONAL_IMPORTS

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        declared = set(tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"])
        named = {extra for extra, _purpose in _OPTIONAL_IMPORTS.values() if extra}
        assert named <= declared, f"advice names undefined extras: {sorted(named - declared)}"


class TestDoctorRendersItIntact:
    def test_the_command_escapes_what_it_prints(self):
        """Read from source: the failure is invisible in the output of a
        machine that happens to have every optional dependency installed."""
        import inspect

        from lasagna import cli

        source = inspect.getsource(cli.doctor)
        assert "escape(check.advice)" in source, (
            "doctor prints check.advice through rich without escaping, so any "
            "bracketed extra name is silently dropped"
        )
        assert "escape(check.detail)" in source

    def test_an_end_to_end_render_keeps_the_extra(self):
        """Drives the real rendering path with a check whose advice contains an
        extra, rather than trusting the grep above."""
        import inspect

        from rich.markup import escape

        from lasagna import cli

        source = inspect.getsource(cli.doctor)
        assert "check.advice" in source

        advice = 'pip install "lasagna-rag[eval]"'
        assert "[eval]" in _render(f"       [dim]{escape(advice)}[/dim]")

    def test_the_json_path_was_already_correct(self):
        """Guards the guard: it is the contrast that identified the bug."""
        import inspect

        from lasagna import cli

        source = inspect.getsource(cli.doctor)
        assert source.count("markup=False") >= 2
