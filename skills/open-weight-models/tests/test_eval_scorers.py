"""Offline tests for the TEDDY eval scorers (eval/scorers.py) and the frozen 2026-10 record.

The replicated checks in eval/scorers.py must reproduce, sub-check for sub-check,
the scores recorded in 2026-10 from the notebooks recorded in 2026-10. Needs
mlflow-skinny (tests/requirements.txt); no workspace, no network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
PREVIOUS = EVAL_DIR / "previous_runs_2026-10"
OLD_TO_NEW = {"oss-001": "teddy-001", "oss-002": "teddy-002"}
RECORDED = {"baseline": "baseline_scores.json", "with_skill": "with_skill_scores.json"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scorers():
    return _load("owm_eval_scorers", EVAL_DIR / "scorers.py")


@pytest.fixture(scope="module")
def old_scorers():
    return _load("owm_previous_scorers", PREVIOUS / "scorers.py")


def _recorded(arm: str) -> dict:
    return json.loads((PREVIOUS / RECORDED[arm]).read_text())


def _source(arm: str, old_id: str) -> str:
    return (PREVIOUS / "results" / arm / f"{old_id}.txt").read_text()


@pytest.mark.parametrize("arm", RECORDED)
@pytest.mark.parametrize("old_id", OLD_TO_NEW)
def test_replicated_checks_reproduce_2026_10_scores(scorers, arm, old_id):
    task = scorers.TASKS[OLD_TO_NEW[old_id]]
    assert scorers.sub_check_scores(task, _source(arm, old_id)) == _recorded(arm)[old_id]


@pytest.mark.parametrize("arm", RECORDED)
@pytest.mark.parametrize("old_id", OLD_TO_NEW)
def test_frozen_scorers_still_reproduce_their_record(old_scorers, arm, old_id):
    tasks = {t["task_id"]: t for t in json.loads((PREVIOUS / "evalset_combined_original.json").read_text())["tasks"]}
    flat = {}
    for phase, result in old_scorers.score_task(tasks[old_id], _source(arm, old_id)).items():
        checks = result.get("checks", {})
        flat.update({f"{phase}/{c}": bool(v) for c, v in checks.items()} if checks else {phase: bool(result["pass"])})
    assert flat == _recorded(arm)[old_id]


def test_previous_split_files_match_combined_original():
    combined = {t["task_id"]: t for t in json.loads((PREVIOUS / "evalset_combined_original.json").read_text())["tasks"]}
    requests = json.loads((PREVIOUS / "evalset.json").read_text())
    answers = {t["task_id"]: t for t in json.loads((PREVIOUS / "expectations.json").read_text())}
    assert [r["task_id"] for r in requests] == list(answers)
    for request in requests:
        assert {**request, **answers[request["task_id"]]} == combined[request["task_id"]]


def test_retest_queries_identical_to_2026_10(scorers):
    combined = {t["task_id"]: t for t in json.loads((PREVIOUS / "evalset_combined_original.json").read_text())["tasks"]}
    for old_id, new_id in OLD_TO_NEW.items():
        assert scorers.TASKS[new_id]["query"] == combined[old_id]["query"]


def test_evalset_holds_no_answer_key():
    for request in json.loads((EVAL_DIR / "evalset.json").read_text()):
        assert not {"difficulty", "expectations", "deterministic_checks", "new_checks"} & set(request)


def test_metric_names_are_valid_mlflow_names(scorers):
    import re

    for arm in RECORDED:
        for key in _recorded(arm)["oss-002"]:
            name = scorers.mlflow_metric_name(key)
            assert re.fullmatch(r"[A-Za-z0-9_.\-/]+", name) and "//" not in name, name
    assert scorers.mlflow_metric_name('forbidden_patterns/workload_type="GPU') == "forbidden_patterns/workload_type_GPU"
    assert scorers.mlflow_metric_name("pyfunc_quality/load_context") == "pyfunc_quality/load_context"


def test_search_flag_from_capability(scorers):
    assert scorers.TASKS["teddy-001"]["vs"] is False
    assert scorers.TASKS["teddy-002"]["vs"] is True


def test_new_checks_are_separate_and_listed_per_task(scorers):
    task = scorers.TASKS["teddy-002"]
    source = (
        "from databricks.sdk.service.serving import AiGatewayConfig, AiGatewayInferenceTableConfig, "
        "AiGatewayUsageTrackingConfig\n"
        "mlflow.set_registry_uri('databricks-uc')\n"
        "mlflow.set_tags({'license': 'Apache-2.0', 'hf_revision': REV})\n"
        "w.serving_endpoints.get(NAME)\n"
        "w.serving_endpoints.create(name=NAME)\n"
        "spark.catalog.tableExists(T)\n"
        "df.write.saveAsTable(T)\n"
        "vsc.create_delta_sync_index(...)\n"
    )
    scores = scorers.new_check_scores(task, source)
    assert all(k.startswith(scorers.NEW_CHECK_PREFIX) for k in scores)
    assert scores == {
        "new_checks/ai_gateway/inference_table": True,
        "new_checks/ai_gateway/usage_tracking": True,
        "new_checks/uc_provenance/uc_registration": True,
        "new_checks/uc_provenance/license_tag": True,
        "new_checks/uc_provenance/hf_revision_tag": True,
        "new_checks/existing_resources/endpoint": True,
        "new_checks/existing_resources/table": True,
        "new_checks/existing_resources/index": False,  # created without looking first
    }
    assert not set(scores) & set(scorers.sub_check_scores(task, source))


def test_existing_resource_check_must_come_first(scorers):
    late = "w.serving_endpoints.delete(NAME)\nw.serving_endpoints.get(NAME)\n"
    assert scorers.existing_resources(late, {})["endpoint"] is False
    commented = "# w.serving_endpoints.get(NAME)\nw.serving_endpoints.create(name=NAME)\n"
    assert scorers.existing_resources(commented, {})["endpoint"] is False


def test_exports(scorers):
    assert scorers.scorers == scorers.replicated_scorers + scorers.new_check_scorers
    assert len(scorers.replicated_scorers) == 8 and len(scorers.new_check_scorers) == 3
