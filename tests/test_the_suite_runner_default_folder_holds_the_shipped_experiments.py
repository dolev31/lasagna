"""SuiteRunner() and ExperimentSupervisor() with no folder argument looked in
private_eval/experiments, which the wheel does not ship, so the examples in
their docstrings found no experiments. They now default to the shipped folder
the command line uses."""

from lasagna.knowledge_evaluation_sdk.private_eval.suite_runner import SuiteRunner
from lasagna.knowledge_evaluation_sdk.private_eval.supervisor import ExperimentSupervisor


def test_the_suite_runner_finds_the_shipped_experiments_by_default():
    files = SuiteRunner()._discover_files()
    assert files, "no experiment YAML found in the default folder"
    assert any(f.endswith("beir_sweep.yaml") for f in files)


def test_the_supervisor_scans_the_shipped_experiments_by_default():
    assert ExperimentSupervisor()._scan_yaml_specs()
