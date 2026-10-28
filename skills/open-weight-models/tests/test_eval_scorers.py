"""Offline tests for the TEDDY eval scorers (eval/scorers.py) and the frozen Sep–Oct 2026 record.

The replicated checks in eval/scorers.py must reproduce, sub-check for sub-check,
the scores recorded in Sep–Oct 2026 from the notebooks recorded in Sep–Oct 2026. Needs
mlflow-skinny (tests/requirements.txt); no workspace, no network.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = SKILL_DIR / "eval"
PREVIOUS = EVAL_DIR / "models" / "teddy" / "runs_20260924"
# Run folders (exported notebooks, run records, per-run scores) are not kept in the repository: they are
# git-ignored and stay with whoever ran the round. Tests that replay them run where the folders are
# present and are skipped elsewhere, e.g. on CI.
NEEDS_PREVIOUS = pytest.mark.skipif(not PREVIOUS.is_dir(),
                                 reason=f"run records not in this checkout ({PREVIOUS.name} is kept outside the repository)")
OLD_TO_NEW = {"oss-001": "teddy-001", "oss-002": "teddy-002"}
RECORDED = {"baseline": "baseline_scores.json", "with_skill": "with_skill_scores.json"}


def _load(name: str, path: Path):
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))  # eval/scorers.py imports family_settings from its folder
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


@NEEDS_PREVIOUS
@pytest.mark.parametrize("arm", RECORDED)
@pytest.mark.parametrize("old_id", OLD_TO_NEW)
def test_replicated_checks_reproduce_sep_oct_2026_scores(scorers, arm, old_id):
    task = scorers.TASKS[OLD_TO_NEW[old_id]]
    assert scorers.sub_check_scores(task, _source(arm, old_id)) == _recorded(arm)[old_id]


@NEEDS_PREVIOUS
@pytest.mark.parametrize("arm", RECORDED)
@pytest.mark.parametrize("old_id", OLD_TO_NEW)
def test_frozen_scorers_still_reproduce_their_record(old_scorers, arm, old_id):
    tasks = {t["task_id"]: t for t in json.loads((PREVIOUS / "evalset_combined_original.json").read_text())["tasks"]}
    flat = {}
    for phase, result in old_scorers.score_task(tasks[old_id], _source(arm, old_id)).items():
        checks = result.get("checks", {})
        flat.update({f"{phase}/{c}": bool(v) for c, v in checks.items()} if checks else {phase: bool(result["pass"])})
    assert flat == _recorded(arm)[old_id]


@NEEDS_PREVIOUS
def test_previous_split_files_match_combined_original():
    combined = {t["task_id"]: t for t in json.loads((PREVIOUS / "evalset_combined_original.json").read_text())["tasks"]}
    requests = json.loads((PREVIOUS / "evalset.json").read_text())
    answers = {t["task_id"]: t for t in json.loads((PREVIOUS / "expectations.json").read_text())}
    assert [r["task_id"] for r in requests] == list(answers)
    for request in requests:
        assert {**request, **answers[request["task_id"]]} == combined[request["task_id"]]


FIRST_RUN_SENTENCE = (  # the exact sentence used for the first rerun (baseline, teddy-002)
    "Use these names: model `<catalog>.skills_owm_202610.owm_teddy_model_002_baseline`, serving endpoint "
    "`owm_teddy_ep_002_baseline`, Vector Search endpoint `owm_teddy_vs_002_baseline`, embeddings table "
    "`<catalog>.skills_owm_202610.owm_teddy_cells_002_baseline`, index "
    "`<catalog>.skills_owm_202610.owm_teddy_index_002_baseline`, and store downloaded files under "
    "`/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/002_baseline/`."
)


@NEEDS_PREVIOUS
def test_retest_queries_are_sep_oct_2026_queries_plus_schema_and_names(scorers):
    """Sep–Oct 2026 wording kept; only the schema changed and ONE names sentence added after it."""
    import family_settings as fs

    settings = fs.load_family("teddy")
    combined = {t["task_id"]: t for t in json.loads((PREVIOUS / "evalset_combined_original.json").read_text())["tasks"]}
    for old_id, new_id in OLD_TO_NEW.items():
        query = scorers.TASKS[new_id]["query"]
        old = combined[old_id]["query"]
        head, tail = old.split("Use catalog <catalog>, schema skills.")
        assert query.startswith(head + "Use catalog <catalog>, schema skills_owm_202610. Use these names: ")
        assert query.endswith(". " + tail.lstrip()) or query.endswith("." + tail)
        for run in ("baseline", "with_skill"):
            rendered = fs.render_query(query, settings, run, new_id)
            assert "{run}" not in rendered
            for value in fs.resource_names(settings, run, new_id).values():
                assert f"`{value}`" in rendered
    first = fs.run_prompts("teddy")["baseline"]["teddy-002"]
    assert FIRST_RUN_SENTENCE in first
    assert first.index("schema skills_owm_202610.") < first.index(FIRST_RUN_SENTENCE)


def test_evalset_is_canonical():
    for request in json.loads((EVAL_DIR / "evalset.json").read_text()):
        assert set(request) == {"task_id", "dataset", "query"}
    for answer in json.loads((EVAL_DIR / "expectations.json").read_text()):
        assert {"model_family", "variant", "capability", "difficulty"} <= set(answer)


def test_metric_names_are_valid_mlflow_names(scorers):
    for task in scorers.TASKS.values():
        for key in scorers.sub_check_scores(task, ""):      # every replicated sub-check key of every task
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
    assert scorers.scorers == (scorers.replicated_scorers + scorers.new_check_scorers + scorers.outcome_scorers
                               + scorers.convention_scorers)
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


@NEEDS_PREVIOUS
@pytest.mark.parametrize("run, arm", [("old_prompt_baseline", "baseline"), ("old_prompt_with_skill", "with_skill")])
def test_old_prompts_are_recorded_text_plus_documented_substitutions(scorers, run, arm):
    """Applying family.json's substitutions (names/paths), the skill-folder change and the
    export-path change to the recorded Sep–Oct 2026 prompt gives exactly the block in the file."""
    import family_settings as fs

    settings = fs.load_family("teddy")
    blocks = scorers.run_prompts()[run]
    assert set(blocks) == set(OLD_TO_NEW.values())
    for old_id, new_id in OLD_TO_NEW.items():
        recorded = _recorded_prompt(PREVIOUS / "results" / arm / f"{old_id}.txt")
        assert blocks[new_id] == fs.rewrite_old_prompt(recorded, settings, run, new_id)
        names = fs.resource_names(settings, run, new_id)
        for key in ("uc_model", "serving_endpoint", "files"):
            assert f"`{names[key]}`" in blocks[new_id]
        if new_id == "teddy-002":
            for key in ("delta_table", "vector_search_index", "vector_search_endpoint"):
                assert f"`{names[key]}`" in blocks[new_id]
        assert "skills_owm_202610" in blocks[new_id] and "`skills`" not in blocks[new_id]


def test_run_prompts_standard_pair_renders_evalset_query(scorers):
    import family_settings as fs

    settings = fs.load_family("teddy")
    prompts = scorers.run_prompts()
    assert set(prompts) == set(scorers.RUN_IDS)
    for run in ("baseline", "with_skill"):
        assert prompts[run] == {tid: fs.render_query(t["query"], settings, run, tid) for tid, t in scorers.TASKS.items()}
    # teddy-001 with_skill uses suffix skill_on: baseline attempt 1 (the standard teddy-001 request)
    # used the `_001_with_skill` names from the run guide before per-run names were finalised
    assert "owm_teddy_ep_001_skill_on" in prompts["with_skill"]["teddy-001"]
    assert "_with_skill" not in prompts["with_skill"]["teddy-001"]
    assert "owm_teddy_ep_002_with_skill" in prompts["with_skill"]["teddy-002"]
    assert prompts["baseline"]["teddy-002"] != prompts["with_skill"]["teddy-002"]
    assert prompts["baseline"]["teddy-002"].replace("_baseline", "_with_skill") == prompts["with_skill"]["teddy-002"]


# ---- severity, environment dependence, outcome scorer ----------------------------

REPO_ROOT = SKILL_DIR.parent.parent


def _all_metric_names(scorers, task_id: str) -> set:
    task = scorers.TASKS[task_id]
    names = {scorers.mlflow_metric_name(k) for k in scorers.sub_check_scores(task, GOOD_SOURCE)}
    names |= set(scorers.new_check_scores(task, GOOD_SOURCE))
    return names | set(task["outcome_checks"]) | set(task["convention_checks"])


def _quote_lines(path: Path, quote: str) -> list:
    """1-based numbers of the lines of `path` that contain `quote`."""
    return [i for i, text in enumerate(path.read_text().splitlines(), 1) if quote in text]


def test_citations_match_on_the_quote_not_the_line_number(tmp_path):
    cited = tmp_path / "SKILL.md"
    cited.write_text("intro\nuse usage_tracking_config here\n")
    assert _quote_lines(cited, "usage_tracking_config") == [2]
    cited.write_text("intro\nnew paragraph\nuse usage_tracking_config here\n")   # the skill was edited
    assert _quote_lines(cited, "usage_tracking_config") == [3]                   # still found, line reported
    assert _quote_lines(cited, "removed text") == []


@pytest.mark.parametrize("task_id", ["teddy-001", "teddy-002"])
def test_every_check_has_severity_and_cited_source(scorers, task_id):
    meta = scorers.TASKS[task_id]["checks"]
    assert set(meta) == _all_metric_names(scorers, task_id)
    for name, m in meta.items():
        assert m["severity"] in scorers.SEVERITIES, name
        assert m["depends_on"] in ("none", "model", "compute"), name
        if m["depends_on"] != "none":
            assert m.get("assumes"), name
        if m["severity"] in ("critical", "major"):
            path, line = m["source"].rsplit(":", 1)
            target = SKILL_DIR / path
            assert target.is_file(), m["source"]
            # the cited file must say what the check relies on. Matched on the quote, not the
            # line number, so the answer key survives edits to the skill; a drifted line number
            # is reported (warning) with the line(s) where the quote is now.
            assert m.get("quote"), name
            found = _quote_lines(target, m["quote"])
            assert found, (name, m["source"], m["quote"])
            if int(line) not in found:
                warnings.warn(f"{name}: quote {m['quote']!r} cited at {m['source']} is now at {path}:{found}")
            assert m.get("why"), name


def test_environment_dependence_examples(scorers):
    meta = scorers.TASKS["teddy-001"]["checks"]
    assert meta["download_staging/hf_xet_disabled"]["depends_on"] == "model"
    assert meta["download_staging/uses_tmp"]["depends_on"] == "compute"
    assert meta["download_staging/deps_pinned"]["depends_on"] == "model"
    assert scorers.TASKS["teddy-001"]["environment_assumed"]["runtime"] == "AI Runtime v5"


def test_failed_by_severity_and_table(scorers):
    metrics = {"corrected/download_staging/deps_pinned": False, "corrected/pyfunc_quality/sys_modules_purge": False,
               "download_staging/uses_tmp": False,           # verbatim: not counted next to corrected ones
               "new_checks/ai_gateway/usage_tracking": False, "outcome/endpoint_ready": True,
               "judge/guidelines": False}
    failed = scorers.failed_by_severity("teddy-001", metrics)
    assert failed == {"critical": ["corrected/download_staging/deps_pinned"],
                      "major": ["new_checks/ai_gateway/usage_tracking"],
                      "minor": ["corrected/pyfunc_quality/sys_modules_purge"]}
    with pytest.raises(ValueError, match="no severity"):
        scorers.failed_by_severity("teddy-001", {"corrected/made_up/check": False})
    table = scorers.severity_table({"baseline": {"teddy-001": metrics}, "with_skill": {"teddy-001": {}}},
                                   [("baseline", "with_skill")])
    assert "| baseline / with_skill | baseline | teddy-001 | 1 | 1 | 1 | yes | - |" in table
    assert "| baseline / with_skill | with_skill | teddy-001 | 0 | 0 | 0 | no | - |" in table
    assert "%" not in table


def test_severity_lookup_normalises_raw_sep_oct_keys(scorers):
    raw = {"forbidden_patterns//local_disk0": False, "forbidden_patterns/transformers>=": False}
    assert scorers.failed_by_severity("teddy-001", raw, groups=("replicated",))["critical"] == list(raw)


def test_verbatim_metrics_count_only_when_asked(scorers):
    """Verbatim (Sep–Oct 2026 logic) metrics alone raise by default instead of silently dropping out
    of the count; groups=("replicated",) gives the numbers as originally reported."""
    verbatim = {"download_staging/uses_tmp": False, "smoke_tests/endpoint_query": False}
    with pytest.raises(ValueError, match="without corrected ones"):
        scorers.failed_by_severity("teddy-001", verbatim)
    assert scorers.failed_by_severity("teddy-001", verbatim, groups=("replicated",)) == {
        "critical": ["download_staging/uses_tmp"], "major": ["smoke_tests/endpoint_query"], "minor": []}


def test_metric_group(scorers):
    assert [scorers.metric_group(n) for n in ("corrected/smoke_tests/endpoint_query",
                                              "smoke_tests/endpoint_query", "new_checks/ai_gateway/usage_tracking",
                                              "outcome/index_ready", "convention/shared_single_cell_format",
                                              "judge/guidelines")] == list(scorers.GROUPS)


def test_outcome_scorer_reads_recorded_outcome(scorers):
    checks = scorers.TASKS["teddy-001"]["outcome_checks"]
    measured = {c: True for c in checks} | {"outcome/returns_valid_embeddings": False}
    feedbacks = scorers.outcome_checks(inputs={"task_id": "teddy-001"}, outputs={"outcome": measured})
    assert {f.name: f.value for f in feedbacks} == measured
    with pytest.raises(ValueError, match="no outcome"):
        scorers.outcome_checks(inputs={"task_id": "teddy-001"}, outputs={"notebook_source": ""})
    with pytest.raises(ValueError, match="not measured"):
        scorers.outcome_checks(inputs={"task_id": "teddy-001"}, outputs={"outcome": {checks[0]: True}})
    assert scorers.outcome_checks in scorers.scorers
    convention = scorers.convention_checks(inputs={"task_id": "teddy-001"},
                                           outputs={"outcome": {"convention/shared_single_cell_format": False}})
    assert [(f.name, f.value) for f in convention] == [("convention/shared_single_cell_format", False)]


def test_convention_checks_are_outside_the_severity_counts(scorers):
    metrics = {"outcome/endpoint_ready": True, "convention/shared_single_cell_format": False}
    assert scorers.failed_by_severity("teddy-001", metrics) == {"critical": [], "major": [], "minor": []}
    scores = {"baseline": {"teddy-001": metrics}, "baseline attempt 1": {"teddy-001": metrics},
              "with_skill": {"teddy-001": {"outcome/endpoint_ready": True}}}
    table = scorers.severity_table(scores, [("baseline", "with_skill")]).splitlines()
    assert "| baseline / with_skill | baseline | teddy-001 | 0 | 0 | 0 | yes | no |" in table
    assert table.index("| baseline / with_skill | baseline attempt 1 | teddy-001 | 0 | 0 | 0 | yes | no |") == 3
    assert "| baseline / with_skill | with_skill | teddy-001 | 0 | 0 | 0 | yes | - |" in table


def test_guideline_judge_is_optional_and_binary(scorers):
    assert not any(getattr(s, "name", "").startswith(scorers.JUDGE_PREFIX) for s in scorers.scorers)
    (judge,) = scorers.guideline_judges()
    assert judge.name == "judge/guidelines"


@pytest.mark.parametrize("key, source", [
    ("model_registration/pip_reqs_from_source", 'pip_requirements=["torch>=2.3.0", "transformers==4.41.0"]\n'),
    ("pyfunc_quality/real_gene_ids", "tok = GeneTokenizer.from_pretrained(model_dir)\n"),
    ("download_staging/uses_tmp", "d = tempfile.mkdtemp()\n"),
    ("forbidden_patterns//local_disk0",
     'try:\n    os.makedirs("/local_disk0/tmp")\n    TMP = "/local_disk0/tmp"\nexcept OSError:\n    TMP = "/tmp"\n'),
    ("endpoint_deployment/ai_gateway_config", 'body = {"ai_gateway": {"inference_table_config": {"enabled": True}}}\n'),
    ("endpoint_deployment/sdk_enums", 'body = {"served_entities": [{"workload_type": "GPU_SMALL"}]}\n'),
    ("ai_search/graceful_skip", "try:\n    df = spark.table(T)\nexcept AnalysisException:\n    df = None\n"),
    ("endpoint_deployment/run_go_gate", 'DEPLOY = dbutils.widgets.get("deploy") == "yes"\n'),
    ("model_registration/infer_signature", 'sig = ModelSignature(inputs=Schema([ColSpec(Array(DataType.string), "g")]))\n'),
    ("download_staging/deps_pinned", 'pip_requirements = ["torch==2.5.1", "safetensors==0.4.5"]\n'),
])
def test_known_false_negatives(scorers, key, source):
    """The README's 'Known false negatives' table: valid alternatives the replicated checks fail."""
    assert scorers.sub_check_scores(scorers.TASKS["teddy-002"], source)[key] is False


def test_teddy_md_subprocess_download_is_not_a_false_negative(scorers):
    teddy = (SKILL_DIR / "references" / "models" / "teddy.md").read_text().splitlines()
    block = "\n".join(teddy[136:154])
    assert "snapshot_download" in block
    assert scorers.sub_check_scores(scorers.TASKS["teddy-001"], block)["download_staging/hf_xet_disabled"] is True



# ---- receivers and per-config flags (re-review of 2776cf9) -------------------------

def test_ai_gateway_flags_are_not_combined_across_endpoints(scorers):
    two_endpoints = _nb(
        "a = AiGatewayConfig(inference_table_config=AiGatewayInferenceTableConfig(enabled=True))",
        "b = AiGatewayConfig(usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))",
        "w.serving_endpoints.create(name='A', ai_gateway=a)\nw.serving_endpoints.create(name='B', ai_gateway=b)",
    )
    result = scorers.ai_gateway_logging(two_endpoints, {})
    assert (result["inference_table"][0], result["usage_tracking"][0]) == (True, False)
    assert "no single attached config enables both" in result["usage_tracking"][1]


@pytest.mark.parametrize("source", [
    # a helper that is not the SDK's serving_endpoints
    "helper.put_ai_gateway(name=N, inference_table_config=AiGatewayInferenceTableConfig(enabled=True),\n"
    "    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))\n",
    "gw = AiGatewayConfig(inference_table_config=AiGatewayInferenceTableConfig(enabled=True),\n"
    "    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))\nhelper.create(name=N, ai_gateway=gw)\n",
    # a client visibly built from something other than WorkspaceClient
    "w = MyFakeClient()\n"
    "w.serving_endpoints.put_ai_gateway(name=N, inference_table_config=AiGatewayInferenceTableConfig(enabled=True),\n"
    "    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))\n",
])
def test_ai_gateway_receiver_must_be_serving_endpoints(scorers, source):
    assert _gateway(scorers, source) == (False, False)


@pytest.mark.parametrize("source", [
    "from databricks.sdk import WorkspaceClient\nw = WorkspaceClient()\nse = w.serving_endpoints\n"
    "se.put_ai_gateway(name=N, inference_table_config=AiGatewayInferenceTableConfig(enabled=True),\n"
    "    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))\n",
    "WorkspaceClient(profile='p').serving_endpoints.put_ai_gateway(name=N,\n"
    "    inference_table_config=AiGatewayInferenceTableConfig(enabled=True),\n"
    "    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))\n",
    # ai_gateway passed through **kwargs
    "gw = AiGatewayConfig(inference_table_config=AiGatewayInferenceTableConfig(enabled=True),\n"
    "    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True))\n"
    "kw = {'name': N, 'ai_gateway': gw}\nw.serving_endpoints.create(**kw)\n",
])
def test_ai_gateway_sdk_receivers_pass(scorers, source):
    assert _gateway(scorers, source) == (True, True)


@pytest.mark.parametrize("source", [
    'custom.log_model(name="m", registered_model_name="cat.skills.teddy_70m")\n',
    'factory.create_registered_model("cat.skills.teddy_70m")\n',
    'client = SomethingElse()\nclient.create_model_version("cat.skills.teddy_70m", source=uri)\n',
    'mlflow.set_registry_uri("databricks-uc")\nhelper.register_model(uri, name)\n',
])
def test_uc_registry_needs_an_mlflow_receiver(scorers, source):
    assert _uc(scorers, source)[0] is False


@pytest.mark.parametrize("source", [
    'import mlflow\nmlflow.transformers.log_model(model, name="m", registered_model_name="cat.skills.teddy_70m")\n',
    'from mlflow import pyfunc\npyfunc.log_model(name="m", python_model=p, registered_model_name="cat.skills.t")\n',
    'from mlflow.tracking import MlflowClient\nc = MlflowClient()\nc.create_registered_model("cat.skills.teddy_70m")\n',
    'import mlflow\nmlflow.MlflowClient().create_model_version(name="cat.skills.teddy_70m", source=uri)\n',
])
def test_uc_registry_mlflow_receivers_pass(scorers, source):
    assert _uc(scorers, source)[0] is True


def test_uc_registry_kwargs(scorers):
    resolved = 'kw = {"python_model": p, "registered_model_name": f"{CATALOG}.skills.teddy_70m"}\n' \
               'mlflow.pyfunc.log_model(name="m", **kw)\n'
    assert _uc(scorers, resolved)[0] is True
    two_part = resolved.replace('f"{CATALOG}.skills.teddy_70m"', '"skills.teddy_70m"')
    assert _uc(scorers, two_part)[0] is False
    # **kwargs built elsewhere: the name cannot be seen -> indeterminate, not passed (even with the UC URI)
    opaque = 'mlflow.pyfunc.log_model(name="m", **build_kwargs())\n'
    for source in (opaque, 'mlflow.set_registry_uri("databricks-uc")\n' + opaque):
        ok, reason = _uc(scorers, source)
        assert ok is False and reason.startswith("indeterminate")


def test_self_referencing_assignment_does_not_recurse_forever(scorers):
    """runs_20261026 baseline/teddy-002 has `x = x + self.position_embeddings(pos)`; following
    `x` through its own assignment used to raise RecursionError in the AI Gateway check."""
    source = ('x = "https://host/api/2.0/serving-endpoints"\n'
              'x = x + "/ep"\n'
              'requests.put(x, json={"ai_gateway": {"inference_table_config": {"enabled": True}}})\n')
    details = scorers.new_check_details(scorers.TASKS["teddy-001"], source, groups={"ai_gateway"})
    assert details["new_checks/ai_gateway/inference_table"][0] is True
    assert details["new_checks/ai_gateway/usage_tracking"][0] is False


def test_conditional_gateway_config_is_followed(scorers):
    """runs_20261026 with_skill/teddy-002 builds `ai_gateway = AiGatewayConfig(...) if FLAG else None`
    and passes it to serving_endpoints.create; the conditional used to hide the config."""
    source = ("from databricks.sdk import WorkspaceClient\n"
              "from databricks.sdk.service.serving import AiGatewayConfig, AiGatewayInferenceTableConfig, "
              "AiGatewayUsageTrackingConfig\n"
              "w = WorkspaceClient()\n"
              "ai_gateway = AiGatewayConfig(\n"
              "    inference_table_config=AiGatewayInferenceTableConfig(catalog_name=C, enabled=True),\n"
              "    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),\n"
              ") if ENABLE else None\n"
              "w.serving_endpoints.create(name=N, config=cfg, ai_gateway=ai_gateway)\n")
    details = scorers.new_check_details(scorers.TASKS["teddy-002"], source, groups={"ai_gateway"})
    assert details["new_checks/ai_gateway/inference_table"][0] and details["new_checks/ai_gateway/usage_tracking"][0]


# ---- skill-eval summary pair (eval/baseline_scores.json, eval/with_skill_scores.json) ----------------
ROUND_2 = EVAL_DIR / "models" / "teddy" / "runs_20261027"


def test_counted_scores_keep_only_the_counted_groups(scorers):
    scores = {"teddy-001": {"corrected/pyfunc_quality/load_context": True, "pyfunc_quality/load_context": False,
                            "new_checks/ai_gateway/inference_table": False, "outcome/endpoint_ready": True,
                            "convention/shared_single_cell_format": True}}
    assert scorers.counted_scores(scores) == {"teddy-001": {"corrected/pyfunc_quality/load_context": True,
                                                            "new_checks/ai_gateway/inference_table": False,
                                                            "outcome/endpoint_ready": True}}


def test_update_summary_replaces_scored_tasks_and_keeps_other_families(scorers, tmp_path):
    (tmp_path / "baseline_scores.json").write_text(json.dumps({
        "geneformer-001": {"outcome/endpoint_ready": True}, "teddy-001": {"outcome/endpoint_ready": True}}))
    path = scorers.update_summary(tmp_path, "baseline", {"teddy-001": {"outcome/endpoint_ready": False,
                                                                       "pyfunc_quality/load_context": True}})
    assert json.loads(path.read_text()) == {"geneformer-001": {"outcome/endpoint_ready": True},
                                            "teddy-001": {"outcome/endpoint_ready": False}}


def test_committed_summary_gives_the_reported_verdict():
    """The repo's summary pair alone reproduces the report: the skill wins both TEDDY tasks, no regressions."""
    sys.path.insert(0, str(SKILL_DIR.parent / "skill-eval" / "scripts"))
    import compare_runs

    comp = compare_runs.compare(compare_runs.load_scores(EVAL_DIR / "baseline_scores.json"),
                                compare_runs.load_scores(EVAL_DIR / "with_skill_scores.json"),
                                difficulty=compare_runs.load_difficulty(EVAL_DIR / "expectations.json"))
    assert comp.counts()["win"] == 2 and comp.counts()["regression"] == 0
    assert comp.gate_passed()


@pytest.mark.skipif(not ROUND_2.is_dir(),
                    reason="run records not in this checkout (runs_20261027 is kept outside the repository)")
def test_committed_summary_matches_round_2_scores(scorers):
    for run, name in scorers.SUMMARY_FILES.items():
        full = json.loads((ROUND_2 / "scores" / name).read_text())
        assert json.loads((EVAL_DIR / name).read_text()) == scorers.counted_scores(full), name


# ---- per-round scores kept in the repo (eval/models/teddy/scores.json) ------------------------------
TEDDY_DIR = EVAL_DIR / "models" / "teddy"
ROUND_1 = TEDDY_DIR / "runs_20261026"


def test_update_family_scores_adds_a_round_and_keeps_the_others(scorers, tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"rounds": {"runs_20990101": {"label": "old", "samples": {}}}}))
    scorers.update_family_scores(path, "runs_20991231", {"baseline": {"teddy-001": {
        "outcome/endpoint_ready": True, "pyfunc_quality/load_context": False}}})
    body = json.loads(path.read_text())
    assert list(body["rounds"]) == ["runs_20991231", "runs_20990101"]            # newest first, old round kept
    assert body["rounds"]["runs_20991231"] == {"notes": "ROUNDS.md",
                                                "samples": {"baseline": {"teddy-001": {"outcome/endpoint_ready": True}}}}
    assert body["counted_groups"] == list(scorers.COUNTED_GROUPS)


def test_latest_round_scores_agree_with_the_summary_pair():
    samples = json.loads((TEDDY_DIR / "scores.json").read_text())["rounds"]["runs_20261027"]["samples"]
    for run in ("baseline", "with_skill"):
        assert json.loads((EVAL_DIR / f"{run}_scores.json").read_text()) == samples[run], run


def test_every_round_in_scores_json_has_its_notes():
    """Each round's "notes" points to its folded section in ROUNDS.md (file#anchor) and that section exists."""
    for name, entry in json.loads((TEDDY_DIR / "scores.json").read_text())["rounds"].items():
        file, _, anchor = entry["notes"].partition("#")
        assert (TEDDY_DIR / file).is_file(), name
        assert f'<a id="{anchor}"></a>' in (TEDDY_DIR / file).read_text(), name


@pytest.mark.skipif(not (ROUND_2.is_dir() and ROUND_1.is_dir()),
                    reason="run records not in this checkout (round folders are kept outside the repository)")
def test_round_scores_match_their_round_folders(scorers):
    rounds = json.loads((TEDDY_DIR / "scores.json").read_text())["rounds"]
    for folder, labels in ((ROUND_2, ["baseline", "with_skill"]), (ROUND_1, ["baseline", "baseline_attempt_1", "with_skill"])):
        samples = rounds[folder.name]["samples"]
        for label in labels:
            full = json.loads((folder / "scores" / f"{label}_scores.json").read_text())
            corrected = json.loads((folder / "scores" / f"{label}_corrected.json").read_text())
            merged = {t: {**full[t], **corrected.get(t, {})} for t in full}   # round 1 scored corrected checks later
            assert samples[label.replace("_attempt_", " attempt ")] == scorers.counted_scores(merged), (folder.name, label)


def test_eval_docs_link_only_to_files_in_the_repo():
    """Relative links in the eval docs resolve to files that are tracked (round folders are git-ignored)."""
    import subprocess

    tracked = set(subprocess.run(["git", "ls-files"], cwd=SKILL_DIR, capture_output=True, text=True).stdout.split())
    docs = [EVAL_DIR / "eval_report.md", EVAL_DIR / "README.md", TEDDY_DIR / "ROUNDS.md"]
    for doc in docs:
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", doc.read_text()):
            if target.startswith(("http://", "https://", "mailto:")):
                assert "eval-records" not in target, f"{doc.name}: links to another branch: {target}"
                continue
            resolved = (doc.parent / target).resolve()
            rel = resolved.relative_to(SKILL_DIR.resolve()).as_posix()
            assert resolved.is_dir() or rel in tracked, f"{doc.name}: {target} is not a tracked file"
