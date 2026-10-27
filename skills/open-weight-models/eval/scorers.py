"""MLflow scorers for the open-weight-models TEDDY retest (teddy-001, teddy-002).

Two groups, reported separately:

  Replicated set — the 8 phase checks used for the 2026-10 runs, with the check
  logic copied unchanged from previous_runs_2026-10/scorers.py so old and new
  sub-check counts can be compared:
    1. download_staging      — temp paths, XET, deps
    2. pyfunc_quality        — class structure, imports, sys.modules, io.StringIO
    3. model_registration    — signature, input_example, pip reqs, variant
    4. endpoint_deployment   — SDK enums, AI Gateway inference table, scale-to-zero
    5. smoke_tests           — registration smoke, endpoint query, output shape
    6. ai_search             — index creation, embedding dim, graceful skip
    7. edge_handling         — template fallback (N/A for TEDDY tasks)
    8. forbidden_patterns    — anti-pattern safety net
  Each sub-check is one metric, named "<phase>/<check>" as in the 2026-10 score
  files. MLflow metric names cannot contain characters such as '"' or '>=', so
  forbidden-pattern names are made safe by mlflow_metric_name(); apply the same
  function to an old score key to compare it.

  New checks — added for the retest, metric names start with "new_checks/":
    ai_gateway           — inference table AND usage tracking on the endpoint
    uc_provenance        — Unity Catalog registration with license + HF revision tags
    existing_resources   — looks for an existing endpoint/table/index before
                           creating or deleting it
  Which sub-checks apply to a task is set by "new_checks" in expectations.json.

Task metadata (model_family, variant, capability, difficulty, checks) is read
from evalset.json + expectations.json next to this file, keyed by task_id. Each
eval row's outputs must carry the exported notebook text as "notebook_source".

Exports: `scorers` (both groups), `replicated_scorers`, `new_check_scorers`,
NEW_CHECK_PREFIX.
"""
import json
import re
from pathlib import Path

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer as mlflow_scorer

EVAL_DIR = Path(__file__).resolve().parent
NEW_CHECK_PREFIX = "new_checks/"
SEARCH_CAPABILITY = "nearest_neighbor_search"


def load_tasks(eval_dir: Path = EVAL_DIR) -> dict:
    """{task_id: task} merged from evalset.json (request) + expectations.json (answer key).

    `vs` is derived from the request's capability so the replicated ai_search
    check reads the same flag it read in 2026-10.
    """
    with open(eval_dir / "evalset.json", encoding="utf-8") as f:
        requests = {t["task_id"]: t for t in json.load(f)}
    with open(eval_dir / "expectations.json", encoding="utf-8") as f:
        answers = {t["task_id"]: t for t in json.load(f)}
    if set(requests) != set(answers):
        raise ValueError(
            f"evalset.json and expectations.json task_ids differ: {sorted(set(requests) ^ set(answers))}"
        )
    tasks = {}
    for tid, request in requests.items():
        task = {**request, **answers[tid]}
        task["vs"] = SEARCH_CAPABILITY in task.get("capability", "")
        tasks[tid] = task
    return tasks


TASKS = load_tasks()


# ---- helpers ---------------------------------------------------------------

def _result(checks: dict, na_reason: str = None) -> dict:
    """Build a scorer result from a checks dict."""
    if na_reason:
        return {"pass": True, "rationale": na_reason, "checks": {}}
    missing = [k for k, v in checks.items() if not v]
    return {
        "pass": len(missing) == 0,
        "rationale": f"Missing: {missing}" if missing else None,
        "checks": checks,
    }


def _is_edge(task: dict) -> bool:
    return task.get("difficulty") == "edge"


def _strip_comments(text: str) -> str:
    """Strip comment lines, inline comments, and markdown cells from exported
    notebook source.  Prevents false positives when forbidden patterns appear
    in comments or negative examples (e.g. '# no /local_disk0')."""
    lines = []
    in_md = False
    for line in text.splitlines():
        s = line.lstrip()
        if s.startswith("# MAGIC %md"):
            in_md = True
            continue
        if in_md:
            if s.startswith("# COMMAND"):
                in_md = False
            continue
        if s.startswith("#"):
            continue
        # Strip inline comments (  # ... ) — simple heuristic: two+ spaces
        # before '#' followed by a space.  Won't mis-strip '#' inside strings
        # in practice because string-embedded '#' is rarely preceded by 2+ spaces.
        line = re.sub(r'\s{2,}#\s.*$', '', line)
        lines.append(line)
    return "\n".join(lines)


# ---- Phase 1: Download & Staging ------------------------------------------

def download_staging(response: str, task: dict) -> dict:
    """Correct download paths, XET workaround, dep pinning, notebook split."""
    if _is_edge(task):
        return _result({}, "N/A — edge task")
    is_teddy = task.get("model_family") == "teddy"
    return _result({
        "uses_tmp":         bool(re.search(r'(?<!/local_disk0)/tmp', response)),
        # TEDDY-specific: XET backend issue only affects TEDDY repo
        "hf_xet_disabled":  bool(re.search(r"HF_HUB_DISABLE_XET", response)) if is_teddy else True,
        # TEDDY-specific: transformers==4.41.0 pin (5.x breaks TeddyGModel)
        "deps_pinned":      (bool(re.search(r"transformers==\d", response))
                            and not bool(re.search(r"transformers>=", response))) if is_teddy else True,
        # INFORMATIONAL ONLY — no prompt asks for split, so this doesn't affect pass/fail.
        # Kept for signal logging: does the skill encourage notebook separation?
        # "cpu_gpu_split" is excluded from pass/fail via _INFORMATIONAL_CHECKS below.
    })


# ---- Phase 2: Custom PyFunc Quality ---------------------------------------

def pyfunc_quality(response: str, task: dict) -> dict:
    """PyFunc class structure, imports, module hygiene, I/O patterns."""
    if _is_edge(task):
        return _result({}, "N/A — edge task")
    cm = re.search(r"class\s+\w+.*?PythonModel", response)
    has_class = cm is not None
    imports_in_class = bool(re.search(r"^\s{4,}import\s", response[cm.start():], re.M)) if cm else False
    # Also accept module-level imports when using file-based logging (python_model=path)
    file_logging = bool(re.search(r"python_model\s*=|\.py['\"]", response))
    has_module_imports = bool(re.search(r"^(?:import|from)\s+\w+", response, re.M))
    # Widened: accept module-level imports (not just indented inside class methods)
    imports_ok = imports_in_class or has_module_imports
    # io.StringIO — only required if pd.read_json present
    stringio_ok = "StringIO" in response if "pd.read_json" in response else True
    # artifacts_declared — log_model must include artifacts= for pyfunc to find weights
    artifacts_ok = bool(re.search(r"log_model\(.*artifacts\s*=", response, re.DOTALL))
    return _result({
        "pyfunc_class":         has_class,
        "imports_inside_class": imports_ok,
        "load_context":         "load_context" in response,
        "sys_modules_purge":    "sys.modules" in response,
        "io_stringio":          stringio_ok,
        "artifacts_declared":   artifacts_ok,
        # TEDDY-specific: vocab.txt is TEDDY's tokenizer
        "real_gene_ids":        ("vocab.txt" in response) if task.get("model_family") == "teddy" else True,
    })


# ---- Phase 3: Model Registration ------------------------------------------

def model_registration(response: str, task: dict) -> dict:
    """Signature, input/output examples, pip reqs, variant markers."""
    if _is_edge(task):
        return _result({}, "N/A — edge task")
    checks = {
        "infer_signature":  "infer_signature" in response,
        "input_example":    "input_example" in response,
        "pip_requirements":  bool(re.search(r"pip_requirements|requirements", response, re.I)),
        # Widened: accept explicit dep-file references OR documented evidence of
        # source-based dependency analysis (TEDDY's HF repo lacks a clean dep spec,
        # so hardcoding from code analysis + documenting the rationale is acceptable).
        "pip_reqs_from_source": bool(re.search(
            r"pyproject\.toml|requirements\.txt|setup\.(cfg|py)"   # explicit dep file
            r"|(?:read|parsed|derived|sourced|examined).*(?:dependenc|pip.req)"  # documented analysis
            , response, re.I)),
    }
    v = task.get("variant", "")
    if v == "400M":
        checks["mentions_1024"] = "1024" in response
        checks["mentions_400M"] = "400M" in response
    elif v == "70M":
        checks["mentions_70M"] = "70M" in response
    return _result(checks)


# ---- Phase 4: Endpoint Deployment -----------------------------------------

def endpoint_deployment(response: str, task: dict) -> dict:
    """SDK enums, AI Gateway inference table, scale-to-zero."""
    if _is_edge(task):
        return _result({}, "N/A — edge task")
    return _result({
        "sdk_enums":            "ServingModelWorkloadType" in response
                                and not bool(re.search(r'workload_type\s*=\s*["\']GPU', response)),
        "ai_gateway_config":    "AiGatewayInferenceTableConfig" in response
                                and "AutoCaptureConfigInput" not in _strip_comments(response),
        "scale_to_zero":        bool(re.search(r"scale_to_zero", response, re.I)),
        "run_go_gate":          bool(re.search(r"run_go|widget.*gate|SKIP_DEPLOY|RUN_GO", response, re.I)),
    })


# ---- Phase 5: Smoke Tests --------------------------------------------------

def smoke_tests(response: str, task: dict) -> dict:
    """Post-registration and post-deployment validation."""
    if _is_edge(task):
        return _result({}, "N/A — edge task")
    # Registration smoke: load from UC + predict
    reg_smoke = bool(re.search(
        r"(load_model|pyfunc\.load).*models:/.*predict", response, re.DOTALL
    )) or bool(re.search(
        r"(smoke\s*test|registration.*test|validate.*registered)", response, re.I
    ))
    # Endpoint query smoke: actually call the endpoint after creation
    endpoint_smoke = bool(re.search(
        r"(serving_endpoints\.query|requests\.post.*endpoint|curl.*endpoint"
        r"|test.*endpoint|verify.*endpoint|query.*endpoint)",
        response, re.I
    ))
    # Output shape validation: check embedding dimension or output shape
    v = task.get("variant", "")
    if v == "400M":
        shape_ok = bool(re.search(r"(1024|embedding_dimension|output.*shape|assert.*len)", response, re.I))
    elif v == "70M":
        shape_ok = bool(re.search(r"(512|embedding_dimension|output.*shape|assert.*len)", response, re.I))
    else:
        shape_ok = True  # no variant = N/A
    return _result({
        "registration_smoke":   reg_smoke,
        "endpoint_query":       endpoint_smoke,
        "output_shape_check":   shape_ok,
    })


# ---- Phase 6: AI Search ---------------------------------------------------

def ai_search(response: str, task: dict) -> dict:
    """AI Search index creation, embedding dim, graceful skip. Conditional on task.vs."""
    if not task.get("vs"):
        return _result({}, "N/A — task does not require AI Search")
    return _result({
        "index_creation":   bool(re.search(
            r"(DeltaSyncVectorIndexSpecRequest|create_index|vector_search_indexes)", response)),
        "embedding_dim":    bool(re.search(
            r"(embedding_dimension|EmbeddingVectorColumn)", response)),
        "graceful_skip":    bool(re.search(
            r"(tableExists|table.*not.*exist|skip.*VS|skip.*AI.Search|Phase 1b.*optional)",
            response, re.I)),
    })


# ---- Phase 7: Edge Handling ------------------------------------------------

def edge_handling(response: str, task: dict) -> dict:
    """Unsupported model fallback. Conditional on edge tasks."""
    if not _is_edge(task):
        return _result({}, "N/A — not an edge task")
    has_tpl = "model-template" in response.lower() or "model_template" in response.lower()
    ack = any(p in response.lower() for p in
              ["not in", "no reference", "no existing", "does not have", "unknown", "not found"])
    return _result({
        "uses_template":        has_tpl,
        "acknowledges_unknowns": has_tpl or ack,
    })


# ---- Phase 8: Forbidden Patterns (safety net) ------------------------------

def forbidden_patterns(response: str, task: dict) -> dict:
    """Catch-all: no anti-patterns from the task's forbidden list."""
    pats = task.get("deterministic_checks", {}).get("forbidden_patterns", [])
    # Strip comments/markdown to avoid false positives from negative examples
    code_only = _strip_comments(response)
    found = [p for p in pats if re.search(re.escape(p), code_only, re.I)]
    return {
        "pass": len(found) == 0,
        "rationale": f"Found: {found}" if found else None,
        "checks": {p: (p not in found) for p in pats},
    }



# ---- Replicated set: plain registry + flattening ----------------------------

REPLICATED_PHASES = {
    "download_staging":     download_staging,
    "pyfunc_quality":       pyfunc_quality,
    "model_registration":   model_registration,
    "endpoint_deployment":  endpoint_deployment,
    "smoke_tests":          smoke_tests,
    "ai_search":            ai_search,
    "edge_handling":        edge_handling,
    "forbidden_patterns":   forbidden_patterns,
}


def score_task(task: dict, response: str) -> dict:
    """Score a response against all replicated phases. Returns {phase: {pass, rationale, checks}}."""
    return {name: fn(response, task) for name, fn in REPLICATED_PHASES.items()}


def _flatten(phase: str, result: dict) -> dict:
    """One phase result -> {"<phase>/<check>": bool}, or {phase: bool} when it has no checks."""
    checks = result.get("checks", {})
    if not checks:
        return {phase: bool(result["pass"])}
    return {f"{phase}/{check}": bool(ok) for check, ok in checks.items()}


def sub_check_scores(task: dict, response: str) -> dict:
    """Replicated sub-check scores keyed exactly as the 2026-10 score files."""
    out = {}
    for phase, result in score_task(task, response).items():
        out.update(_flatten(phase, result))
    return out


def mlflow_metric_name(key: str) -> str:
    """Make a sub-check key a valid MLflow metric name.

    "forbidden_patterns/transformers>=" -> "forbidden_patterns/transformers"; keys
    that are already valid (all non-forbidden-pattern checks) are unchanged.
    """
    head, sep, check = key.partition("/")
    if not sep:
        return key
    return f"{head}/{re.sub(r'[^A-Za-z0-9_.-]+', '_', check).strip('_')}"


# ---- New checks (reported separately) ---------------------------------------

def _first(pattern: str, text: str):
    m = re.search(pattern, text)
    return m.start() if m else None


def ai_gateway_logging(response: str, task: dict) -> dict:
    """Inference table AND usage tracking set on the endpoint's AI Gateway config
    (SDK classes, or the same field names in a REST body / put_ai_gateway call)."""
    code = _strip_comments(response)
    return {
        "inference_table": bool(re.search(r"AiGatewayInferenceTableConfig|inference_table_config", code)),
        "usage_tracking":  bool(re.search(r"AiGatewayUsageTrackingConfig|usage_tracking_config", code)),
    }


def uc_provenance(response: str, task: dict) -> dict:
    """Registered to Unity Catalog with license and Hugging Face revision recorded as tags."""
    code = _strip_comments(response)
    tagging = r"(set_registered_model_tag|set_model_version_tag|set_tag|tags\s*=|[\"']tags[\"']\s*:)"
    has_tagging = bool(re.search(tagging, code))
    return {
        "uc_registration": bool(re.search(r"databricks-uc|registered_model_name\s*=|register_model\(", code)),
        "license_tag":     has_tagging and bool(re.search(r"[\"'][\w.]*license[\w.]*[\"']", code, re.I)),
        "hf_revision_tag": has_tagging and bool(re.search(
            r"[\"'][\w.]*(revision|commit|sha)[\w.]*[\"']", code, re.I)),
    }


# Per resource: (existence check, first create/delete call). The check must appear
# before the first create/delete in the notebook source.
_RESOURCE_PATTERNS = {
    "endpoint": (
        r"serving_endpoints\.(get|list)\(",
        r"serving_endpoints\.(create|create_and_wait|delete)\(",
    ),
    "table": (
        r"tableExists\(|tables\.(exists|get)\(|SHOW TABLES|DESCRIBE TABLE|IF NOT EXISTS",
        r"saveAsTable\(|CREATE (OR REPLACE )?TABLE|DROP TABLE|\.writeTo\(",
    ),
    "index": (
        r"get_index\(|list_indexes\(|index_exists",
        r"create_delta_sync_index\(|create_direct_access_index\(|create_index\(|delete_index\(",
    ),
}


def existing_resources(response: str, task: dict) -> dict:
    """Looked for an existing resource before creating or deleting it."""
    code = _strip_comments(response)
    out = {}
    for resource, (check_pat, mutate_pat) in _RESOURCE_PATTERNS.items():
        check_at, mutate_at = _first(check_pat, code), _first(mutate_pat, code)
        out[resource] = check_at is not None and (mutate_at is None or check_at < mutate_at)
    return out


NEW_CHECKS = {
    "ai_gateway":         ai_gateway_logging,
    "uc_provenance":      uc_provenance,
    "existing_resources": existing_resources,
}


def new_check_scores(task: dict, response: str, groups=None) -> dict:
    """{"new_checks/<group>/<check>": bool} for the sub-checks listed in the task's new_checks.

    `groups` limits scoring to those new-check groups (default: all listed).
    """
    out = {}
    for group, wanted in task.get("new_checks", {}).items():
        if groups is not None and group not in groups:
            continue
        results = NEW_CHECKS[group](response, task)
        unknown = set(wanted) - set(results)
        if unknown:
            raise ValueError(f"{task['task_id']}: unknown {group} checks {sorted(unknown)}")
        out.update({f"{NEW_CHECK_PREFIX}{group}/{c}": bool(results[c]) for c in wanted})
    return out


# ---- MLflow @scorer wrappers -------------------------------------------------
#
# One scorer per phase / new-check group; each returns one Feedback per
# sub-check so every sub-check is its own boolean metric. A phase that is N/A
# for the task returns one Feedback named after the phase (value True), as in
# the 2026-10 score files.

def _task(inputs) -> dict:
    return TASKS[inputs["task_id"]]


def _source(outputs) -> str:
    return outputs["notebook_source"]


def _phase_feedbacks(phase: str, inputs, outputs) -> list:
    result = REPLICATED_PHASES[phase](_source(outputs), _task(inputs))
    rationale = result.get("rationale") or "pass"
    if not result.get("checks"):
        label = f"[N/A] {rationale}" if str(rationale).startswith("N/A") else rationale
        return [Feedback(name=phase, value=bool(result["pass"]), rationale=label)]
    return [
        Feedback(name=mlflow_metric_name(key), value=ok, rationale="pass" if ok else "fail")
        for key, ok in _flatten(phase, result).items()
    ]


def _group_feedbacks(group: str, inputs, outputs) -> list:
    scores = new_check_scores(_task(inputs), _source(outputs), groups={group})
    return [Feedback(name=k, value=ok, rationale="pass" if ok else "fail") for k, ok in scores.items()]


@mlflow_scorer
def download_staging_checks(inputs, outputs):
    """Download paths, XET, deps."""
    return _phase_feedbacks("download_staging", inputs, outputs)

@mlflow_scorer
def pyfunc_quality_checks(inputs, outputs):
    """PyFunc class structure, imports, module hygiene."""
    return _phase_feedbacks("pyfunc_quality", inputs, outputs)

@mlflow_scorer
def model_registration_checks(inputs, outputs):
    """Signature, input_example, pip reqs, variant."""
    return _phase_feedbacks("model_registration", inputs, outputs)

@mlflow_scorer
def endpoint_deployment_checks(inputs, outputs):
    """SDK enums, AI Gateway inference table, scale-to-zero, run gate."""
    return _phase_feedbacks("endpoint_deployment", inputs, outputs)

@mlflow_scorer
def smoke_tests_checks(inputs, outputs):
    """Registration smoke, endpoint query, output shape."""
    return _phase_feedbacks("smoke_tests", inputs, outputs)

@mlflow_scorer
def ai_search_checks(inputs, outputs):
    """AI Search index creation, embedding dim, graceful skip."""
    return _phase_feedbacks("ai_search", inputs, outputs)

@mlflow_scorer
def edge_handling_checks(inputs, outputs):
    """Template fallback, acknowledges unknowns."""
    return _phase_feedbacks("edge_handling", inputs, outputs)

@mlflow_scorer
def forbidden_patterns_checks(inputs, outputs):
    """Anti-pattern safety net."""
    return _phase_feedbacks("forbidden_patterns", inputs, outputs)

@mlflow_scorer
def ai_gateway_new_checks(inputs, outputs):
    """New: inference table + usage tracking on the endpoint's AI Gateway."""
    return _group_feedbacks("ai_gateway", inputs, outputs)

@mlflow_scorer
def uc_provenance_new_checks(inputs, outputs):
    """New: UC registration with license + HF revision tags."""
    return _group_feedbacks("uc_provenance", inputs, outputs)

@mlflow_scorer
def existing_resources_new_checks(inputs, outputs):
    """New: checks for an existing endpoint/table/index before create/delete."""
    return _group_feedbacks("existing_resources", inputs, outputs)


replicated_scorers = [
    download_staging_checks,
    pyfunc_quality_checks,
    model_registration_checks,
    endpoint_deployment_checks,
    smoke_tests_checks,
    ai_search_checks,
    edge_handling_checks,
    forbidden_patterns_checks,
]
new_check_scorers = [
    ai_gateway_new_checks,
    uc_provenance_new_checks,
    existing_resources_new_checks,
]
scorers = replicated_scorers + new_check_scorers
