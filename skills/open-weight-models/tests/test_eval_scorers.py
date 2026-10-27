"""Offline tests for the TEDDY eval scorers (eval/scorers.py) and the frozen 2026-10 record.

The replicated checks in eval/scorers.py must reproduce, sub-check for sub-check,
the scores recorded in 2026-10 from the notebooks recorded in 2026-10. Needs
mlflow-skinny (tests/requirements.txt); no workspace, no network.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = SKILL_DIR / "eval"
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


def test_evalset_is_canonical():
    for request in json.loads((EVAL_DIR / "evalset.json").read_text()):
        assert set(request) == {"task_id", "dataset", "query"}
    for answer in json.loads((EVAL_DIR / "expectations.json").read_text()):
        assert {"model_family", "variant", "capability", "difficulty"} <= set(answer)


def test_metric_names_are_valid_mlflow_names(scorers):
    for arm in RECORDED:
        for key in _recorded(arm)["oss-002"]:
            name = scorers.mlflow_metric_name(key)
            assert re.fullmatch(r"[A-Za-z0-9_.\-/]+", name) and "//" not in name, name
    assert scorers.mlflow_metric_name('forbidden_patterns/workload_type="GPU') == "forbidden_patterns/workload_type_GPU"
    assert scorers.mlflow_metric_name("pyfunc_quality/load_context") == "pyfunc_quality/load_context"


def test_search_flag_from_capability(scorers):
    assert scorers.TASKS["teddy-001"]["vs"] is False
    assert scorers.TASKS["teddy-002"]["vs"] is True


GOOD_SOURCE = """
from databricks.sdk.service.serving import AiGatewayConfig, AiGatewayInferenceTableConfig, AiGatewayUsageTrackingConfig
mlflow.set_registry_uri("databricks-uc")
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.teddy_70m"
mlflow.register_model(model_uri, UC_MODEL_NAME)
mlflow.set_tags({"license": "Apache-2.0", "hf_revision": REV})
w.serving_endpoints.get(NAME)
gateway = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(catalog_name=CATALOG, schema_name=SCHEMA, enabled=True),
    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
)
w.serving_endpoints.create(name=NAME, config=cfg, ai_gateway=gateway)
spark.catalog.tableExists(T)
df.write.saveAsTable(T)
vsc.create_delta_sync_index(endpoint_name=E, index_name=I)
"""


def test_new_checks_are_separate_and_listed_per_task(scorers):
    task = scorers.TASKS["teddy-002"]
    scores = scorers.new_check_scores(task, GOOD_SOURCE)
    assert all(k.startswith(scorers.NEW_CHECK_PREFIX) for k in scores)
    assert scores == {
        "new_checks/ai_gateway/inference_table": True,
        "new_checks/ai_gateway/usage_tracking": True,
        "new_checks/uc_provenance/uc_registry": True,
        "new_checks/uc_provenance/provenance_tags": True,
        "new_checks/existing_resources/endpoint": True,
        "new_checks/existing_resources/table": True,
        "new_checks/existing_resources/index": False,  # created without looking first
    }
    assert not set(scores) & set(scorers.sub_check_scores(task, GOOD_SOURCE))


def _gateway(scorers, source):
    result = scorers.ai_gateway_logging(source, {})
    return result["inference_table"][0], result["usage_tracking"][0]


def _nb(*cells):
    """Exported-notebook text from Python cells."""
    return "# Databricks notebook source\n" + "\n\n# COMMAND ----------\n\n".join(cells) + "\n"


def test_ai_gateway_imports_alone_fail(scorers):
    imports_only = (
        "from databricks.sdk.service.serving import AiGatewayConfig, AiGatewayInferenceTableConfig, "
        "AiGatewayUsageTrackingConfig\nw.serving_endpoints.create(name=NAME, config=cfg)\n"
    )
    assert _gateway(scorers, imports_only) == (False, False)


def test_ai_gateway_needs_enabled_and_attached(scorers):
    not_attached = GOOD_SOURCE.replace(", ai_gateway=gateway", "")
    assert _gateway(scorers, not_attached) == (False, False)
    usage_off = GOOD_SOURCE.replace("AiGatewayUsageTrackingConfig(enabled=True)", "AiGatewayUsageTrackingConfig()")
    assert _gateway(scorers, usage_off) == (True, False)
    flag_false = _nb("ENABLED = False",
                     "gw = AiGatewayConfig(inference_table_config=AiGatewayInferenceTableConfig(enabled=ENABLED),\n"
                     "    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=ENABLED))",
                     "w.serving_endpoints.create(name=N, ai_gateway=gw)")
    assert _gateway(scorers, flag_false) == (False, False)
    # a REST body that is never sent to a serving-endpoints URL is not an attachment
    unsent = 'body = {"ai_gateway": {"inference_table_config": {"enabled": True}, "usage_tracking_config": {"enabled": True}}}\n'
    assert _gateway(scorers, unsent) == (False, False)


def test_ai_gateway_teddy_md_pattern(scorers):
    teddy = (SKILL_DIR / "references" / "models" / "teddy.md").read_text()
    block = next(b for b in re.findall(r"```python\n(.*?)```", teddy, re.S) if "ai_gateway = AiGatewayConfig(" in b)
    assert _gateway(scorers, _nb(block)) == (True, True)


@pytest.mark.parametrize("cells", [
    # annotated assignment
    ("gateway: AiGatewayConfig = AiGatewayConfig(\n"
     "    inference_table_config=AiGatewayInferenceTableConfig(catalog_name=C, schema_name=S, enabled=True),\n"
     "    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))",
     "w.serving_endpoints.create(name=N, config=cfg, ai_gateway=gateway)"),
    # boolean constant in a variable
    ("ENABLED = True",
     "itc = AiGatewayInferenceTableConfig(catalog_name=C, schema_name=S, enabled=ENABLED)\n"
     "gw = AiGatewayConfig(inference_table_config=itc, usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=ENABLED))",
     "w.serving_endpoints.create_and_wait(name=N, ai_gateway=gw)"),
    # REST body with the gateway built earlier
    ('gateway = {"inference_table_config": {"catalog_name": C, "enabled": True},\n'
     '           "usage_tracking_config": {"enabled": True}}',
     'payload = {"name": N, "config": cfg, "ai_gateway": gateway}',
     'requests.post(f"{HOST}/api/2.0/serving-endpoints", headers=H, json=payload)'),
    # REST via dict(...) and the SDK api client
    ('gw = dict(inference_table_config=dict(enabled=True), usage_tracking_config=dict(enabled=True))',
     'w.api_client.do("POST", "/api/2.0/serving-endpoints", body=dict(name=N, ai_gateway=gw))'),
    # PUT .../ai-gateway: the body is the gateway
    ('requests.put(f"{HOST}/api/2.0/serving-endpoints/{N}/ai-gateway",\n'
     '             json={"inference_table_config": {"enabled": True}, "usage_tracking_config": {"enabled": True}})',),
    # positional dataclass arguments
    ("gw = AiGatewayConfig(None, None, AiGatewayInferenceTableConfig(C, True, S), None, AiGatewayUsageTrackingConfig(True))",
     "w.serving_endpoints.create(N, ai_gateway=gw)"),
    # from_dict
    ('gw = AiGatewayConfig.from_dict({"inference_table_config": {"enabled": True}, "usage_tracking_config": {"enabled": True}})',
     "w.serving_endpoints.create(name=N, ai_gateway=gw)"),
    # put_ai_gateway on an existing endpoint
    ("w.serving_endpoints.put_ai_gateway(name=N,\n"
     "    inference_table_config=AiGatewayInferenceTableConfig(enabled=True),\n"
     "    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))",),
])
def test_ai_gateway_legitimate_patterns_pass(scorers, cells):
    assert _gateway(scorers, _nb(*cells)) == (True, True)


def test_ai_gateway_unparsable_cell_uses_regex_fallback(scorers):
    source = _nb(
        "%pip install x\nthis is ( not python",
        "gw = AiGatewayConfig(inference_table_config=AiGatewayInferenceTableConfig(enabled=True),\n"
        "    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))\n"
        "w.serving_endpoints.create(name=N, ai_gateway=gw)\nif broken syntax here",
    )
    result = scorers.ai_gateway_logging(source, {})
    assert result["inference_table"][0] and result["usage_tracking"][0]
    assert "regex" in result["inference_table"][1]


def _uc(scorers, source):
    return scorers.uc_provenance(source, {})["uc_registry"]


def test_uc_registry_uri_alone_fails(scorers):
    ok, reason = _uc(scorers, 'mlflow.set_registry_uri("databricks-uc")\n')
    assert ok is False and "alone" in reason


def test_uc_registry_two_part_name_fails(scorers):
    two_part = 'mlflow.pyfunc.log_model(name="m", python_model=M(), registered_model_name="skills.teddy_70m")\n'
    assert _uc(scorers, two_part)[0] is False
    # Unity Catalog rejects names that are not catalog.schema.model, so the URI does not rescue it
    assert _uc(scorers, 'mlflow.set_registry_uri("databricks-uc")\n' + two_part)[0] is False


@pytest.mark.parametrize("source", [
    'mlflow.pyfunc.log_model(name="m", python_model=M(), registered_model_name=f"{CATALOG}.skills.teddy_70m")\n',
    'CATALOG = "main"\nNAME = f"{CATALOG}.{SCHEMA}.teddy_70m"\nmlflow.register_model(f"runs:/{run_id}/model", NAME)\n',
    'MlflowClient().create_model_version("cat.skills.teddy_70m", source=uri, run_id=rid)\n',
    'client = MlflowClient()\nclient.create_registered_model(name="cat.skills.teddy_70m")\n',
])
def test_uc_registry_three_part_registration_passes(scorers, source):
    assert _uc(scorers, source)[0] is True


def test_uc_registry_uri_plus_runtime_name_passes(scorers):
    source = 'mlflow.set_registry_uri("databricks-uc")\nname = dbutils.widgets.get("model")\nmlflow.register_model(uri, name)\n'
    ok, reason = _uc(scorers, source)
    assert ok is True and "run time" in reason
    assert _uc(scorers, source.replace('mlflow.set_registry_uri("databricks-uc")\n', ""))[0] is False


def test_provenance_tags_need_hf_revision_and_report_kind(scorers):
    no_revision = 'mlflow.set_tags({"license": "Apache-2.0", "hf_source": URL})\n'
    ok, reason = scorers.uc_provenance(no_revision, {})["provenance_tags"]
    assert ok is False and "hf_revision: not recorded as a tag" in reason and "license: run tags" in reason
    uc_tags = (
        'client.set_model_version_tag(NAME, VERSION, "license", "Apache-2.0")\n'
        'client.set_registered_model_tag(NAME, "hf_revision", REV)\n'
    )
    ok, reason = scorers.uc_provenance(uc_tags, {})["provenance_tags"]
    assert ok is True and "UC model tags" in reason
    # license in a provenance JSON file, not a tag, does not count
    not_a_tag = 'provenance = {"license": "Apache-2.0", "revision": REV}\njson.dump(provenance, f)\n'
    assert scorers.uc_provenance(not_a_tag, {})["provenance_tags"][0] is False


def test_existing_resource_check_must_come_first(scorers):
    late = "w.serving_endpoints.delete(NAME)\nw.serving_endpoints.get(NAME)\n"
    assert scorers.existing_resources(late, {})["endpoint"][0] is False
    commented = "# w.serving_endpoints.get(NAME)\nw.serving_endpoints.create(name=NAME)\n"
    assert scorers.existing_resources(commented, {})["endpoint"][0] is False


def test_exports(scorers):
    assert scorers.scorers == scorers.replicated_scorers + scorers.new_check_scorers
    assert len(scorers.replicated_scorers) == 8 and len(scorers.new_check_scorers) == 3


def _recorded_prompt(path: Path) -> str:
    lines = path.read_text().splitlines()
    k = next(i for i, line in enumerate(lines) if "arm prompt (this notebook)" in line) + 1
    while not lines[k].startswith("# MAGIC >"):
        k += 1
    out = []
    while k < len(lines) and lines[k].startswith("# MAGIC >"):
        line = lines[k][len("# MAGIC >"):]
        out.append(line[1:] if line.startswith(" ") else line)
        k += 1
    return "\n".join(out)


@pytest.mark.parametrize("run, arm", [("old_prompt_baseline", "baseline"), ("old_prompt_with_skill", "with_skill")])
def test_old_prompts_are_recorded_text_except_paths(scorers, run, arm):
    blocks = scorers.run_prompts()[run]
    assert set(blocks) == set(OLD_TO_NEW.values())
    for old_id, new_id in OLD_TO_NEW.items():
        restored = blocks[new_id].replace(
            "/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/",
            "/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/.assistant/skills/oss-models/",
        )
        restored = re.sub(
            rf"/Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/{run}/{new_id}\.txt",
            "<export>", restored)
        recorded = re.sub(
            r"/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/results/\w+/oss-00\d\.txt",
            "<export>", _recorded_prompt(PREVIOUS / "results" / arm / f"{old_id}.txt"))
        assert restored == recorded


def test_run_prompts_standard_pair_uses_evalset_query(scorers):
    prompts = scorers.run_prompts()
    assert set(prompts) == set(scorers.RUN_IDS)
    for run in ("baseline", "with_skill"):
        assert prompts[run] == {tid: task["query"] for tid, task in scorers.TASKS.items()}
