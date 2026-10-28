"""MLflow scorers for the open-weight-models TEDDY retest (teddy-001, teddy-002).

Groups, reported separately (plus an optional judge):

  Replicated set ("verbatim") — the 8 phase checks used for the Sep–Oct 2026 runs, with the
  check logic copied unchanged from models/teddy/runs_20260924/scorers.py. Kept as a historical record
  only: it reproduces the Sep–Oct 2026 numbers as originally reported. Several sub-checks are
  flawed (literal-text matches); corrected_checks.py judges their intent properly, and those
  corrected results ("corrected/<phase>/<check>", the scorer of record) are what the severity
  table counts. Sub-checks:
    1. download_staging      — temp paths, XET, deps
    2. pyfunc_quality        — class structure, imports, sys.modules, io.StringIO
    3. model_registration    — signature, input_example, pip reqs, variant
    4. endpoint_deployment   — SDK enums, AI Gateway inference table, scale-to-zero
    5. smoke_tests           — registration smoke, endpoint query, output shape
    6. ai_search             — index creation, embedding dim, graceful skip
    7. edge_handling         — template fallback (N/A for TEDDY tasks)
    8. forbidden_patterns    — anti-pattern safety net
  Each sub-check is one metric, named "<phase>/<check>" as in the Sep–Oct 2026 score
  files. MLflow metric names cannot contain characters such as '"' or '>=', so
  forbidden-pattern names are made safe by mlflow_metric_name(); apply the same
  function to an old score key to compare it.

  New checks — added for the retest, metric names start with "new_checks/":
    ai_gateway           — inference table AND usage tracking enabled in an AI Gateway
                           config attached to the serving endpoint (parsed with `ast`)
    uc_provenance        — uc_registry (an actual Unity Catalog registration with a
                           three-part name, parsed with `ast`) and provenance_tags
                           (license + Hugging Face revision recorded as tags; the
                           rationale says whether run, logged-model or UC model tags)
    existing_resources   — looks for an existing endpoint/table/index before
                           creating or deleting it
  Which sub-checks apply to a task is set by "new_checks" in expectations.json.

Task metadata is read from evalset.json (task_id, dataset, query) and
expectations.json (difficulty, model_family, variant, capability, checks) next to
this file, keyed by task_id; each task's outcome checks come from its family settings
(models/<family>/family.json, via family_settings.py). Each
eval row's outputs must carry the exported notebook text as "notebook_source".

  Outcome group — metric names start with "outcome/": what the run built actually works
  (measured live by check_outcome.py before teardown, read here from outputs["outcome"]),
  with the request sent in the endpoint's own declared input format.
  Convention group — metric names start with "convention/": whether the endpoint uses the
  shared single-cell input format of the reference notes (teddy.md, scgpt.md). Read from the
  same recorded outcome, reported on its own and NOT counted in the severity table.
  Optional, off by default: guideline_judges() — an LLM judge over the task guidelines,
  metric "judge/guidelines".

Every check has a severity (critical / major / minor), the source line it rests on, and
whether it depends on the model or the compute, in expectations.json "checks" (a corrected
metric takes the severity of the sub-check it corrects); failed_by_severity() /
severity_table() report raw failed counts per severity over COUNTED_GROUPS (corrected, new
checks, outcome), or over the verbatim set when asked (`groups=("replicated",)`).

Exports: `scorers` (replicated + new checks + outcome + convention; the corrected scorer is
corrected_checks.corrected_scorers, kept there to avoid a circular import), `replicated_scorers`,
`new_check_scorers`, `outcome_scorers`, `convention_scorers`, guideline_judges(), the metric prefixes,
run_prompts() (the prompt pasted in each run), failed_by_severity(), severity_table().
"""
import ast
import json
import re
from pathlib import Path

import family_settings  # same folder: per-family names, prompts, outcome settings
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer as mlflow_scorer

EVAL_DIR = Path(__file__).resolve().parent
CORRECTED_PREFIX = "corrected/"
NEW_CHECK_PREFIX = "new_checks/"
OUTCOME_PREFIX = "outcome/"
CONVENTION_PREFIX = "convention/"
JUDGE_PREFIX = "judge/"
SEARCH_CAPABILITY = "nearest_neighbor_search"


def load_tasks(eval_dir: Path = EVAL_DIR) -> dict:
    """{task_id: task} merged from evalset.json (request) + expectations.json (answer key),
    plus the task's outcome checks from its family settings (models/<family>/family.json).

    `vs` is derived from the capability so the replicated ai_search check reads the same
    flag it read in Sep–Oct 2026.
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
        settings = family_settings.load_family(task["model_family"], eval_dir)
        task["outcome_checks"] = family_settings.outcome_checks(settings, tid)
        task["convention_checks"] = family_settings.convention_checks(settings)
        tasks[tid] = task
    return tasks


TASKS = load_tasks()

RUN_IDS = family_settings.RUN_IDS
run_prompts = family_settings.run_prompts   # {run_id: {task_id: prompt pasted in that run}}


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
    """Replicated sub-check scores keyed exactly as the Sep–Oct 2026 score files."""
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
#
# Each returns {check: (passed, reason)}. They read code only (comments and
# markdown stripped) and look for real calls, not imports or names.

def _first(pattern: str, text: str):
    m = re.search(pattern, text)
    return m.start() if m else None


def _balanced(text: str, open_at: int) -> str:
    """Text between the bracket at `open_at` and its matching close bracket (to the end if unclosed)."""
    pairs = {"(": ")", "{": "}", "[": "]"}
    stack = []
    for i in range(open_at, len(text)):
        ch = text[i]
        if ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return text[open_at + 1:i]
    return text[open_at + 1:]


def _call_args(code: str, callee: str) -> list:
    """Argument text of every call to `callee` (a regex for the name before '(')."""
    return [_balanced(code, m.end() - 1) for m in re.finditer(rf"{callee}\s*\(", code)]


def _assigned(code: str, name: str) -> str:
    """Right-hand side of every `name = ...` assignment; a bracketed value or call is taken in full."""
    out = []
    for m in re.finditer(rf"^\s*{re.escape(name)}\s*=(?!=)\s*", code, re.M):
        start = m.end()
        opener = re.match(r"[\w.]*\s*([({\[])", code[start:])
        if opener:
            open_at = start + opener.start(1)
            close = {"(": ")", "{": "}", "[": "]"}[code[open_at]]
            out.append(code[start:open_at + 1] + _balanced(code, open_at) + close)
        else:
            end = code.find("\n", start)
            out.append(code[start:end if end != -1 else len(code)])
    return "\n".join(out)


def _with_names(code: str, args: str) -> str:
    """`args` plus the assigned values of identifiers used in it (one level)."""
    names = set(re.findall(r"(?<![\w.\"'])([A-Za-z_]\w*)(?![\w(\"'])", args))
    return args + "\n" + "\n".join(_assigned(code, n) for n in names)


_ENABLED_TRUE = r"enabled\s*=\s*True|[\"']enabled[\"']\s*:\s*True"


# ---- Python-code view of a notebook (for the AI Gateway and UC registry checks)
#
# Each Python cell is parsed with `ast`; names are followed through plain and
# annotated assignments anywhere in the notebook, so a config built in one cell
# and attached in another is found. A cell that does not parse is checked with
# the regex fallback instead (`_regex_*`).

def code_cells(response: str) -> list:
    """Python code of each notebook cell: `# MAGIC` lines (markdown, %pip, %sql, %sh cells),
    the notebook header and line magics (`%...`, `!...`) removed; empty cells dropped."""
    cells = []
    for cell in re.split(r"^# COMMAND -+[ \t]*$", response, flags=re.M):
        lines = [line for line in cell.splitlines()
                 if not line.lstrip().startswith(("# MAGIC", "# Databricks notebook source", "%", "!"))]
        text = "\n".join(lines)
        if text.strip():
            cells.append(text)
    return cells


def _call_name(call: ast.Call) -> str:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


# Dataclass field order in databricks-sdk 0.157.0, for positional constructor arguments.
_POSITIONAL_FIELDS = {
    "AiGatewayConfig": ["fallback_config", "guardrails", "inference_table_config", "rate_limits",
                        "usage_tracking_config"],
    "AiGatewayInferenceTableConfig": ["catalog_name", "enabled", "schema_name", "table_name_prefix"],
    "AiGatewayUsageTrackingConfig": ["enabled"],
}


class NotebookCode:
    """Parsed Python cells of one notebook, with a name -> assigned-values table."""

    def __init__(self, response: str):
        self.trees, self.unparsed = [], []
        for cell in code_cells(response):
            try:
                self.trees.append(ast.parse(cell))
            except SyntaxError:
                self.unparsed.append(cell)
        self.assigned = {}
        self.imports = {}                              # local name -> dotted module path
        self.calls = []
        for tree in self.trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        local = alias.asname or alias.name.split(".")[0]
                        self.imports[local] = alias.name if alias.asname else alias.name.split(".")[0]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        self.imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.assigned.setdefault(target.id, []).append(node.value)
                elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
                    self.assigned.setdefault(node.target.id, []).append(node.value)
                elif isinstance(node, ast.Call):
                    self.calls.append(node)

    def values(self, node, depth: int = 0) -> list:
        """Candidate values of `node`: a simple name is followed through its assignments, and
        a conditional expression (`a if cond else b`) stands for both branches."""
        if depth < 8 and isinstance(node, ast.Name) and node.id in self.assigned:
            return [v for value in self.assigned[node.id] for v in self.values(value, depth + 1)]
        if depth < 8 and isinstance(node, ast.IfExp):
            return self.values(node.body, depth + 1) + self.values(node.orelse, depth + 1)
        return [node]

    def dotted(self, node, depth: int = 0) -> list:
        """Where an expression comes from, as dotted paths with imports and simple names
        resolved: `w.serving_endpoints` with `w = WorkspaceClient()` and
        `from databricks.sdk import WorkspaceClient` -> ["databricks.sdk.WorkspaceClient().serving_endpoints"].
        An unresolvable name stays as itself; anything else becomes "?"."""
        if depth > 8:
            return ["?"]
        if isinstance(node, ast.Name):
            if node.id in self.assigned:
                return [d for v in self.assigned[node.id] for d in self.dotted(v, depth + 1)]
            return [self.imports.get(node.id, node.id)]
        if isinstance(node, ast.Attribute):
            return [f"{d}.{node.attr}" for d in self.dotted(node.value, depth + 1)]
        if isinstance(node, ast.Call):
            return [f"{d}()" for d in self.dotted(node.func, depth + 1)]
        return ["?"]

    def keyword_maps(self, call: ast.Call) -> list:
        """Possible {keyword: value} maps of a call: explicit keywords plus `**kwargs` dicts
        that resolve to a dict literal or dict(...). A `**` that does not resolve adds the
        key "**" (value None) so callers can treat its contents as unknown."""
        explicit = {k.arg: k.value for k in call.keywords if k.arg}
        maps = [{}]
        for k in call.keywords:
            if k.arg is None:
                dicts = [v for v in self.values(k.value)
                         if isinstance(v, ast.Dict) or (isinstance(v, ast.Call) and _call_name(v) == "dict")]
                options = [o for v in dicts for o in self.members(v)] or [{"**": None}]
                maps = [{**m, **o} for m in maps for o in options]
        return [{**m, **explicit} for m in maps]

    def is_true(self, node) -> bool:
        return node is not None and any(isinstance(v, ast.Constant) and v.value is True for v in self.values(node))

    def members(self, node) -> list:
        """Field -> value maps `node` can stand for: a dict literal, dict(...), an SDK
        constructor (keyword or positional), Cls.from_dict({...}) or json.dumps({...})."""
        out = []
        for v in self.values(node):
            if isinstance(v, ast.Dict):
                out.append({k.value: val for k, val in zip(v.keys, v.values)
                            if isinstance(k, ast.Constant) and isinstance(k.value, str)})
            elif isinstance(v, ast.Call):
                name = _call_name(v)
                if name in ("from_dict", "dumps") and v.args:
                    out += self.members(v.args[0])
                    continue
                fields = dict(zip(_POSITIONAL_FIELDS.get(name, []), v.args))
                if name == "dict" and v.args:
                    for m in self.members(v.args[0]):
                        fields.update(m)
                fields.update({k.arg: k.value for k in v.keywords if k.arg})
                out.append(fields)
        return out

    def strings(self, node, depth: int = 0) -> list:
        """Possible string values of `node`; an unresolved f-string part becomes '{}'.
        `depth` bounds self-referencing assignments such as `x = x + y`."""
        if depth > 8:
            return []
        out = []
        for v in self.values(node):
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.append(v.value)
            elif isinstance(v, ast.JoinedStr):
                parts = [""]
                for piece in v.values:
                    if isinstance(piece, ast.Constant):
                        options = [str(piece.value)]
                    else:
                        options = self.strings(piece.value, depth + 1) or ["{}"]
                    parts = [p + o for p in parts for o in options]
                out += parts
            elif isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add):
                out += [a + b for a in self.strings(v.left, depth + 1) or ["{}"]
                        for b in self.strings(v.right, depth + 1) or ["{}"]]
        return out


def _site_flags(nb: NotebookCode, gateway: dict) -> tuple:
    """(inference table enabled, usage tracking enabled) for ONE attached AI Gateway config."""
    def enabled(field: str) -> bool:
        return gateway.get(field) is not None and any(nb.is_true(m.get("enabled")) for m in nb.members(gateway[field]))
    return enabled("inference_table_config"), enabled("usage_tracking_config")


def _gateway_flags(nb: NotebookCode, gateway_maps: list) -> tuple:
    """Flags of the single attached config that enables the most (first wins on a tie).
    Flags are never combined across configs: one endpoint with only the inference table
    and another with only usage tracking does not pass both."""
    flags = [_site_flags(nb, g) for g in gateway_maps]
    return max(flags, key=sum) if flags else (False, False)


# A WorkspaceClient's serving_endpoints: `WorkspaceClient(...).serving_endpoints`, or
# `<name>.serving_endpoints` where <name> is not visibly assigned to something else.
_SERVING_RECEIVER = re.compile(r"^(?:(?:[\w.]*\.)?WorkspaceClient\(\)|[A-Za-z_]\w*)\.serving_endpoints$")


def _is_serving_endpoints(nb: NotebookCode, receiver) -> bool:
    return any(_SERVING_RECEIVER.match(d) for d in nb.dotted(receiver))


def _gateway_attachments(nb: NotebookCode) -> list:
    """AI Gateway field maps attached to a serving endpoint, one per attachment site."""
    sites = []
    for call in nb.calls:
        name = _call_name(call)
        receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
        if name in ("create", "create_and_wait", "put_ai_gateway"):
            if receiver is None or not _is_serving_endpoints(nb, receiver):
                continue                                # e.g. helper.put_ai_gateway(...): not the SDK
            for kwargs in nb.keyword_maps(call):
                if name == "put_ai_gateway":
                    sites.append(kwargs)                # the call's keywords are the gateway fields
                elif kwargs.get("ai_gateway") is not None:
                    sites += nb.members(kwargs["ai_gateway"])  # ai_gateway is keyword-only in the SDK
            continue
        # REST: any call that mentions a serving-endpoints URL and sends a body
        args = list(call.args) + [v for m in nb.keyword_maps(call) for v in m.values() if v is not None]
        urls = [u for a in args for u in nb.strings(a) if "serving-endpoints" in u]
        if not urls:
            continue
        bodies = [m for a in args for m in nb.members(a)]
        if any("ai-gateway" in u for u in urls):
            sites += bodies                             # PUT .../ai-gateway: the body is the gateway
        sites += [g for body in bodies if body.get("ai_gateway") is not None for g in nb.members(body["ai_gateway"])]
    return sites


def _regex_ai_gateway(code: str) -> tuple:
    """Regex fallback for cells that do not parse: (inference enabled, usage enabled, attached)."""
    sdk_attach = [_with_names(code, a) for a in _call_args(code, r"serving_endpoints\.create(?:_and_wait)?")
                  if re.search(r"\bai_gateway\s*=", a)]
    put_attach = [_with_names(code, a) for a in _call_args(code, r"serving_endpoints\.put_ai_gateway")]
    rest_attach = [_balanced(code, m.end() - 1) for m in re.finditer(r"[\"']ai_gateway[\"']\s*:\s*\{", code)]
    attached = sdk_attach + put_attach + rest_attach

    def enabled(site: str, cls: str, field: str) -> bool:
        configs = _call_args(site, cls) + [
            _balanced(site, m.end() - 1) for m in re.finditer(rf"[\"']{field}[\"']\s*:\s*\{{", site)
        ]
        return any(re.search(_ENABLED_TRUE, c) for c in configs)

    # per attached config, never combined across configs (as in _gateway_flags)
    flags = [(enabled(site, "AiGatewayInferenceTableConfig", "inference_table_config"),
              enabled(site, "AiGatewayUsageTrackingConfig", "usage_tracking_config"))
             for site in (_with_names(code, a) for a in attached)]
    best = max(flags, key=sum) if flags else (False, False)
    return best[0], best[1], bool(attached)


def ai_gateway_logging(response: str, task: dict) -> dict:
    """Inference table AND usage tracking enabled in an AI Gateway config that is attached to
    a serving endpoint: `ai_gateway=` on serving_endpoints.create / create_and_wait, a
    put_ai_gateway(...) call, or an "ai_gateway" key in a REST body sent to a
    serving-endpoints URL (or the body of PUT .../ai-gateway). Imports alone never count."""
    nb = NotebookCode(response)
    sites = _gateway_attachments(nb)
    inference, usage = _gateway_flags(nb, sites)
    attached = bool(sites)
    how = "parsed code"
    if nb.unparsed and not (inference and usage):
        r_inf, r_use, r_att = _regex_ai_gateway(_strip_comments("\n".join(nb.unparsed)))
        if r_inf + r_use > inference + usage:          # take the better single config, never mix
            inference, usage = r_inf, r_use
        attached = attached or r_att
        how = f"parsed code + regex on {len(nb.unparsed)} unparsable cell(s)"
    if not attached:
        reason = f"no AI Gateway config attached to a serving endpoint ({how})"
        return {"inference_table": (False, reason), "usage_tracking": (False, reason)}
    single = "" if inference == usage else "; no single attached config enables both"
    return {
        "inference_table": (inference, f"{'enabled' if inference else 'attached config does not enable the inference table'}{single} ({how})"),
        "usage_tracking":  (usage, f"{'enabled' if usage else 'attached config does not enable usage tracking'}{single} ({how})"),
    }


# Registration sites: call name -> (keyword, position) of the model-name argument, and the
# MLflow receiver it must resolve to. create_registered_model and create_model_version are
# MlflowClient methods only (there is no mlflow.create_registered_model); register_model is
# mlflow.register_model; log_model is a flavor's mlflow.<flavor>.log_model.
_MLFLOW_CLIENT = r"(?:mlflow(?:\.tracking|\.client)?\.)?MlflowClient\(\)"
_REGISTRATION_SITES = {
    "log_model": ("registered_model_name", None, re.compile(r"^mlflow\.\w+\.log_model$")),
    "register_model": ("name", 1, re.compile(r"^mlflow\.register_model$")),
    "create_registered_model": ("name", 0, re.compile(rf"^{_MLFLOW_CLIENT}\.create_registered_model$")),
    "create_model_version": ("name", 0, re.compile(rf"^{_MLFLOW_CLIENT}\.create_model_version$")),
}


def _is_three_part(name: str) -> bool:
    parts = name.split(".")
    return len(parts) == 3 and all(parts)


def _is_unresolved(name: str) -> bool:
    return not name.replace("{}", "").strip()


KWARGS_UNKNOWN = "{**kwargs}"


def _registered_names(nb: NotebookCode, call: ast.Call) -> list | None:
    """Model names passed at an MLflow registration call; None if the call is not one.

    `**kwargs` is followed when it resolves to a dict literal or dict(...) in the notebook;
    a `**` that does not resolve (and no explicit name) gives KWARGS_UNKNOWN: the name cannot
    be seen, so the check is indeterminate (scored as not passed)."""
    spec = _REGISTRATION_SITES.get(_call_name(call))
    if not spec:
        return None
    keyword, position, receiver = spec
    if not any(receiver.match(d) for d in nb.dotted(call.func)):
        return None                                     # e.g. custom.log_model(...): not MLflow
    names = []
    for kwargs in nb.keyword_maps(call):
        node = kwargs.get(keyword)
        if node is None and position is not None and len(call.args) > position:
            node = call.args[position]
        if node is not None:
            names += nb.strings(node) or ["{}"]
        elif "**" in kwargs:
            names.append(KWARGS_UNKNOWN)
    return names


def _uc_registry(nb: NotebookCode) -> tuple:
    """(passed, reason) from parsed code."""
    uc_uri = any(_call_name(c) == "set_registry_uri" and c.args and "databricks-uc" in nb.strings(c.args[0])
                 and any(d == "mlflow.set_registry_uri" for d in nb.dotted(c.func)) for c in nb.calls)
    names, sites = [], 0
    for call in nb.calls:
        found = _registered_names(nb, call)
        if found:
            sites += 1
            names += found
    if any(_is_three_part(n) for n in names):
        return True, "registration with a three-part catalog.schema.model name"
    if uc_uri and any(_is_unresolved(n) for n in names if n != KWARGS_UNKNOWN):
        return True, "databricks-uc registry URI + registration whose name is built at run time"
    if KWARGS_UNKNOWN in names:
        return False, ("indeterminate: the registered name is passed through **kwargs that are not "
                       "visible in the notebook (scored as not passed; read the notebook by hand)")
    if not sites:
        return False, "no MLflow model registration call" + (" (databricks-uc registry URI alone is not a registration)" if uc_uri else "")
    return False, f"registered name(s) not three-part: {sorted(set(names))}"


def _regex_uc_registry(code: str) -> bool:
    """Regex fallback for cells that do not parse."""
    uc_uri = bool(re.search(r"mlflow\.set_registry_uri\(\s*[\"']databricks-uc[\"']\s*\)", code))
    name_sites = [_with_names(code, a) for a in _call_args(code, r"\bmlflow\.register_model")]
    if "MlflowClient(" in code:
        name_sites += [_with_names(code, a) for a in _call_args(code, r"\.(?:create_registered_model|create_model_version)")]
    name_sites += [_with_names(code, a) for a in _call_args(code, r"\bmlflow\.\w+\.log_model")
                   if re.search(r"registered_model_name\s*=", a)]
    three_part = any(re.search(_THREE_PART, s) for s in name_sites)
    return three_part or (uc_uri and bool(name_sites))


_THREE_PART = r"[\"'](?=[^\"'\s]*\.[^\"'\s]*\.)[^\"'\s.]+\.[^\"'\s.]+\.[^\"'\s.]+[\"']"

# Where a tag can be recorded: (kind, call name regex, look only at its tags= argument?)
# create_registered_model is an MlflowClient method only.
_TAG_SITES = [
    ("run tags", r"\bmlflow\.set_tags?", False),
    ("run tags", r"\bstart_run", True),
    ("logged-model tags", r"\bset_logged_model_tags", False),
    ("logged-model tags", r"\blog_model", True),
    ("UC model tags", r"\bset_registered_model_tag|\bset_model_version_tag", False),
    ("UC model tags", r"\bregister_model|\bcreate_registered_model", True),
]
_TAG_KEYS = {
    "license": r"[\"'][\w.-]*license[\w.-]*[\"']\s*[:,]",
    "hf_revision": r"[\"'][\w.-]*(revision|commit|sha)[\w.-]*[\"']\s*[:,]",
}


def provenance_tag_sources(code: str) -> dict:
    """{"license"|"hf_revision": [kinds of tag where it was recorded]} for comment-stripped code."""
    found = {key: [] for key in _TAG_KEYS}
    for kind, callee, tags_kwarg_only in _TAG_SITES:
        for args in _call_args(code, rf"(?:{callee})"):
            if tags_kwarg_only:
                m = re.search(r"\btags\s*=\s*", args)
                if not m:
                    continue
                args = args[m.end():]
            text = _with_names(code, args)
            for key, pattern in _TAG_KEYS.items():
                if re.search(pattern, text, re.I) and kind not in found[key]:
                    found[key].append(kind)
    return found


def uc_provenance(response: str, task: dict) -> dict:
    """(a) uc_registry: an actual registration — log_model(registered_model_name=...),
    mlflow.register_model(...), or MlflowClient().create_registered_model / create_model_version —
    whose model name is three-part (catalog.schema.model); or the databricks-uc registry URI
    plus a registration whose name is only known at run time. The URI alone, or a
    registration under a two-part name, fails (Unity Catalog requires three-part names).
    (b) provenance_tags: license AND Hugging Face revision recorded as tags (run tags,
    logged-model tags or UC model tags all count; the reason names which)."""
    nb = NotebookCode(response)
    registry, registry_reason = _uc_registry(nb)
    if not registry and nb.unparsed and _regex_uc_registry(_strip_comments("\n".join(nb.unparsed))):
        registry, registry_reason = True, f"regex on {len(nb.unparsed)} unparsable cell(s)"

    sources = provenance_tag_sources(_strip_comments(response))
    tags_ok = all(sources.values())
    tags_reason = "; ".join(f"{k}: {', '.join(v) if v else 'not recorded as a tag'}" for k, v in sources.items())
    return {
        "uc_registry":     (registry, registry_reason),
        "provenance_tags": (tags_ok, tags_reason),
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
        ok = check_at is not None and (mutate_at is None or check_at < mutate_at)
        reason = ("checked before first create/delete" if ok else
                  "no existence check" if check_at is None else "existence check comes after first create/delete")
        out[resource] = (ok, reason)
    return out


NEW_CHECKS = {
    "ai_gateway":         ai_gateway_logging,
    "uc_provenance":      uc_provenance,
    "existing_resources": existing_resources,
}


def new_check_details(task: dict, response: str, groups=None) -> dict:
    """{"new_checks/<group>/<check>": (passed, reason)} for the sub-checks listed in the
    task's new_checks. `groups` limits scoring to those groups (default: all listed)."""
    out = {}
    for group, wanted in task.get("new_checks", {}).items():
        if groups is not None and group not in groups:
            continue
        results = NEW_CHECKS[group](response, task)
        unknown = set(wanted) - set(results)
        if unknown:
            raise ValueError(f"{task['task_id']}: unknown {group} checks {sorted(unknown)}")
        out.update({f"{NEW_CHECK_PREFIX}{group}/{c}": (bool(results[c][0]), results[c][1]) for c in wanted})
    return out


def new_check_scores(task: dict, response: str, groups=None) -> dict:
    """{"new_checks/<group>/<check>": bool} — new_check_details without the reasons."""
    return {k: ok for k, (ok, _) in new_check_details(task, response, groups).items()}


# ---- MLflow @scorer wrappers -------------------------------------------------
#
# One scorer per phase / new-check group; each returns one Feedback per
# sub-check so every sub-check is its own boolean metric. A phase that is N/A
# for the task returns one Feedback named after the phase (value True), as in
# the Sep–Oct 2026 score files.

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
    details = new_check_details(_task(inputs), _source(outputs), groups={group})
    return [Feedback(name=k, value=ok, rationale=reason) for k, (ok, reason) in details.items()]


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


# ---- Outcome group -------------------------------------------------------------
#
# Did what the run built actually work? Measured live by check_outcome.py after each
# run and before teardown, stored in the run record, and passed to the scorer as
# outputs["outcome"] = {"outcome/<check>": bool}. Nothing is re-measured here.

def _recorded_feedbacks(task: dict, outputs, names: list) -> list:
    measured = outputs.get("outcome")
    if not measured:
        raise ValueError(f"{task['task_id']}: no outcome in outputs — run check_outcome.py before teardown")
    missing = [c for c in names if c not in measured]
    if missing:
        raise ValueError(f"{task['task_id']}: outcome checks not measured: {missing}")
    return [Feedback(name=c, value=bool(measured[c]), rationale="pass" if measured[c] else "fail")
            for c in names]


@mlflow_scorer
def outcome_checks(inputs, outputs):
    """Outcome: endpoint ready + serving the recorded version, valid embeddings in its own
    declared format, takes expression input, search works."""
    task = _task(inputs)
    return _recorded_feedbacks(task, outputs, task["outcome_checks"])


@mlflow_scorer
def convention_checks(inputs, outputs):
    """Convention: the endpoint uses the shared single-cell input format (reported on its own)."""
    task = _task(inputs)
    return _recorded_feedbacks(task, outputs, task["convention_checks"])


outcome_scorers = [outcome_checks]
convention_scorers = [convention_checks]
scorers = replicated_scorers + new_check_scorers + outcome_scorers + convention_scorers


# ---- Optional LLM judge over the task guidelines (off by default) ----------------

def guideline_judges(model: str = "databricks:/databricks-gpt-5-mini") -> list:
    """One binary judge (skill-eval scorer-pack.md: make_judge, explicit yes/no
    feedback_value_type) that grades the notebook against the task's `guidelines` from
    expectations.json. Metric name "judge/guidelines". Not part of `scorers`; the judge
    model must differ from the model that wrote the notebooks."""
    from typing import Literal

    from mlflow.genai.judges import make_judge

    return [make_judge(
        name=f"{JUDGE_PREFIX}guidelines",
        instructions="""
You are grading a Databricks notebook that an agent wrote to deploy an open-weight model.

Task given to the agent: {{ inputs }}
Notebook source written by the agent: {{ outputs }}
Rules the notebook must follow (the "guidelines" list): {{ expectations }}

Grading rules:
- Check each guideline against the notebook source only. Do not assume code that is not shown.
- A guideline that the source does not clearly satisfy counts as failed.
- Ignore every other field of the expectations; grade only the "guidelines" list.

Answer with exactly one word: 'yes' if every guideline is satisfied, otherwise 'no'.
""",
        feedback_value_type=Literal["yes", "no"],
        model=model,
    )]


# ---- Grouping + severity report (pure) ---------------------------------------------

GROUPS = ("corrected", "replicated", "new_checks", "outcome", "convention", "judge")
# Groups counted in the severity table: the corrected replicated checks (corrected_checks.py),
# the new checks and the outcome. "replicated" (the verbatim Sep–Oct 2026 logic) is counted only
# when asked for, to show Sep–Oct numbers as originally reported.
COUNTED_GROUPS = ("corrected", "new_checks", "outcome")


def metric_group(name: str) -> str:
    """Which reporting group a metric belongs to."""
    for group, prefix in (("corrected", CORRECTED_PREFIX), ("new_checks", NEW_CHECK_PREFIX), ("outcome", OUTCOME_PREFIX),
                          ("convention", CONVENTION_PREFIX), ("judge", JUDGE_PREFIX)):
        if name.startswith(prefix):
            return group
    return "replicated"


# skill-eval's summary pair (skills/skill-eval/SKILL.md "Outputs"): eval/baseline_scores.json and
# eval/with_skill_scores.json, {task_id: {metric: bool}}, read by compare_runs.py. Task IDs are unique
# across model families (teddy-001, geneformer-001, ...), so every family's latest round shares the two files.
SUMMARY_FILES = {"baseline": "baseline_scores.json", "with_skill": "with_skill_scores.json"}


def counted_scores(task_scores: dict) -> dict:
    """Keep only the counted groups (COUNTED_GROUPS) of {task_id: {metric: bool}}."""
    return {task_id: {m: v for m, v in metrics.items() if metric_group(m) in COUNTED_GROUPS}
            for task_id, metrics in task_scores.items()}


def update_summary(eval_dir, run: str, task_scores: dict) -> Path:
    """Write one standard run's counted scores into eval/<run>_scores.json.

    The tasks scored now replace their earlier entries; other tasks (other model families) are kept.
    """
    path = Path(eval_dir) / SUMMARY_FILES[run]
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary.update(counted_scores(task_scores))
    path.write_text(json.dumps(dict(sorted(summary.items())), indent=2) + "\n", encoding="utf-8")
    return path


def update_family_scores(path, round_name: str, scores_by_sample: dict, notes: str | None = None) -> Path:
    """Put one round's counted scores, every sample (baseline, with_skill, extra attempts), into the family's
    models/<family>/scores.json under "rounds" -> round_name; the other rounds in the file are kept.

    The round folder itself (exported notebooks, run records, per-group score files) is git-ignored; this file
    and the round's notes (folded under "Round details" in models/<family>/ROUNDS.md) are what the repo keeps
    of each round. `notes` names that section, e.g. "ROUNDS.md#round-2"; it defaults to ROUNDS.md.
    """
    path = Path(path)
    body = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rounds": {}}
    body["counted_groups"] = list(COUNTED_GROUPS)
    entry = body["rounds"].get(round_name, {})
    entry.update({"notes": notes or entry.get("notes", "ROUNDS.md"),
                  "samples": {label: counted_scores(s) for label, s in scores_by_sample.items()}})
    body["rounds"][round_name] = entry
    body["rounds"] = dict(sorted(body["rounds"].items(), reverse=True))        # newest round first
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


SEVERITIES = ("critical", "major", "minor")


def failed_by_severity(task_id: str, metrics: dict, tasks: dict = None, groups: tuple = COUNTED_GROUPS) -> dict:
    """{"critical"|"major"|"minor": [failed metric names]} for one (run, task), from the
    `checks` severities in expectations.json; a corrected metric has the severity of the
    replicated sub-check it corrects. Only metrics of `groups` are counted (judge metrics have no
    severity and convention metrics are reported in their own column). Verbatim replicated
    metrics without corrected ones raise unless `groups` asks for "replicated": they would
    otherwise drop out of the count silently."""
    meta = (tasks or TASKS)[task_id]["checks"]
    by_group = {}
    for name in metrics:
        by_group.setdefault(metric_group(name), []).append(name)
    if "replicated" in by_group and "corrected" not in by_group and "replicated" not in groups:
        raise ValueError(f"{task_id}: verbatim replicated metrics without corrected ones; score them with "
                         "corrected_checks.py, or pass groups=('replicated', ...) for the numbers as originally reported")
    out = {sev: [] for sev in SEVERITIES}
    for name, ok in metrics.items():
        if metric_group(name) not in groups:
            continue
        key = name[len(CORRECTED_PREFIX):] if name.startswith(CORRECTED_PREFIX) else name
        if metric_group(name) in ("corrected", "replicated"):
            key = mlflow_metric_name(key)   # a raw Sep–Oct 2026 key ("forbidden_patterns//local_disk0")
        if key not in meta:
            raise ValueError(f"{task_id}: no severity for metric {name!r} in expectations.json")
        if not ok:
            out[meta[key]["severity"]].append(name)
    return out


def severity_table(scores_by_run: dict, pairs: list, tasks: dict = None, groups: tuple = COUNTED_GROUPS) -> str:
    """Markdown: per pair, per run (or run sample), per task — raw counts of failed checks by
    severity, whether the outcome group passed, and the convention checks on their own
    ("-" if none were scored). No percentages, no single score. `scores_by_run` may hold
    extra samples of a run under "<run> <label>" keys (e.g. "baseline attempt 1"); each
    is listed under its pair right after the run's primary sample."""
    lines = ["| Pair | Run | Task | Critical failed | Major failed | Minor failed | Outcome passed "
             "| Shared input format |",
             "|---|---|---|---|---|---|---|---|"]
    for pair in pairs:
        for run in pair:
            samples = [run] + sorted(k for k in scores_by_run if k.startswith(f"{run} "))
            for sample in samples:
                for task_id, metrics in sorted(scores_by_run.get(sample, {}).items()):
                    failed = failed_by_severity(task_id, metrics, tasks, groups)
                    outcome = [ok for name, ok in metrics.items() if name.startswith(OUTCOME_PREFIX)]
                    convention = [ok for name, ok in metrics.items() if name.startswith(CONVENTION_PREFIX)]
                    lines.append(f"| {pair[0]} / {pair[1]} | {sample} | {task_id} | "
                                 + " | ".join(str(len(failed[s])) for s in SEVERITIES)
                                 + f" | {'yes' if outcome and all(outcome) else 'no'}"
                                 + f" | {('yes' if all(convention) else 'no') if convention else '-'} |")
    return "\n".join(lines)
