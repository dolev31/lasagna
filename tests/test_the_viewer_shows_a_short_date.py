"""The results page shows when a run was made as a date a person reads.

The header showed ``2026-10-15T18:51:38.818843+00:00``. It shows
``2026-10-15 18:51 UTC``, and the recorded timestamp stays on the page, as the
element's ``title`` (shown on hover), so nothing the run recorded is lost.
"""

from __future__ import annotations

import json
import re

from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION
from lasagna.knowledge_evaluation_sdk.viewer import _date, render_html

RAW = "2026-10-15T18:51:38.818843+00:00"


def _page(tmp_path, created_at):
    task = {
        "task_key": "bench|control|h|s|live|run_0",
        "run_index": 0,
        "benchmark_name": "bench",
        "config_name": "control",
        "metrics": {"ndcg@10": 0.5},
        "per_sample_scores": [{"_sample_id": "q0", "ndcg@10": 0.5}],
        "error": None,
        "config_overrides": {},
    }
    config = {"experiment_name": "dated", "metrics_version": METRICS_VERSION}
    if created_at is not None:
        config["created_at"] = created_at
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"experiment_config": config, "results": [task]}))
    return render_html(path)


def test_the_header_date_is_short_and_keeps_the_timestamp(tmp_path):
    page = _page(tmp_path, RAW)
    assert f'<dt>Date</dt><dd title="{RAW}">2026-10-15 18:51 UTC</dd>' in page
    assert re.sub(r'title="[^"]*"', "", page).count(RAW) == 0


def test_other_offsets_and_unparseable_values_still_show(tmp_path):
    assert _date("2026-10-15T21:51:38+03:00") == "2026-10-15 21:51 +0300"
    assert _date("2026-10-15T18:51:38") == "2026-10-15 18:51"
    assert _date("last tuesday") == "last tuesday"
    assert "<dt>Date</dt>" not in _page(tmp_path, None)
