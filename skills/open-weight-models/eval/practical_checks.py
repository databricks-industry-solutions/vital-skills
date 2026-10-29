#!/usr/bin/env python3
"""Practical checks for the open-weight-models TEDDY eval — TRIAL, not adopted.

Not wired into score_open_weight_models.py and not part of `scorers.scorers`. Each check
scores whether a GOAL of the task was reached, accepting any valid way of reaching it,
instead of matching strings from teddy.md. Code goals are read by parsing the notebook's
Python cells with `ast` (scorers.NotebookCode), following values across cells; behaviour
goals read the outcome measured before teardown and kept in the run record
(run_records.json "outcome" / "endpoint_config"). Metric names start with "practical/".
Every check's goal, severity and accepted alternatives are in CHECKS.

Reported apart from the practical score (like the shared-format convention check):
GUIDELINES — task guidelines the prompts do not ask for, read from the run record only
(metric names "guideline/..."; not counted as failures).

models/teddy/runs_20261026/PRACTICAL_TRIAL.md compares these checks with the current ones
on the 2026-10-27 runs.

  python3 skills/open-weight-models/eval/practical_checks.py [--round runs_20261026]

prints each run's practical and guideline results (reason per check, with notebook line
numbers). Needs mlflow-skinny (scorers.py imports it); no workspace, no network.
"""

from __future__ import annotations

import argparse
import ast
import re
import shlex
import sys
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import scorers  # noqa: E402 — same folder; reused: NotebookCode, AI Gateway, UC registry, existing-resource logic
from scorers import NotebookCode, _call_name  # noqa: E402

PRACTICAL_PREFIX = "practical/"
GUIDELINE_PREFIX = "guideline/"
SEVERITIES = scorers.SEVERITIES

# Python package of each family's upstream code: a call into it is an upstream reference.
UPSTREAM_PACKAGES = {"teddy": ("teddy",)}

CHECKS = {
    "download_location_ok": {
        "severity": "critical",
        "goal": "Model files are downloaded and staged where serverless compute can write.",
        "accepts": ["a /tmp path", "tempfile.mkdtemp() / TemporaryDirectory() (both under /tmp)",
                    "a Unity Catalog Volume path (/Volumes/...)",
                    "the same, as the destination of a download run in a subprocess or shell command",
                    "/local_disk0 when compute-adaptive: the path variable has one of the above assigned in "
                    "the try handler or other if branch, and that variable is what the download / copy uses "
                    "(followed into a subprocess script's text)"],
        "fails": "/local_disk0 without such a fallback, as a download destination or as the destination of "
                 "a copy/move of the files (staging); a download destination that is none of the above, including the "
                 "default Hugging Face cache (not stated in the code). Other /local_disk0 strings "
                 "(messages, unrelated paths) do not count",
        "evidence": "code",
    },
    "xet_download_ok": {
        "severity": "critical",
        "goal": "The Hugging Face download of the TEDDY repository (Xet storage) completes.",
        "accepts": ["(every way below needs a disabling value: 1 / true / yes / on, any case)",
                    "the download runs in a subprocess whose environment sets HF_HUB_DISABLE_XET "
                    "(env= dict) or whose script sets it before importing huggingface_hub "
                    "(teddy.md, 'Hugging Face download')",
                    "a %sh / ! `hf download` command with HF_HUB_DISABLE_XET set on that line or "
                    "exported earlier in the cell",
                    "in the notebook: HF_HUB_DISABLE_XET set before the first huggingface_hub import "
                    "and before the download (teddy.md warns this is not reliable on AI Runtime)",
                    "recorded evidence the download completed: a later cell loads files from the "
                    "download's destination or a copy of it (from_pretrained, safetensors load_file, "
                    "load_state_dict, log_model artifacts; paths followed through assignments, joins, "
                    "copies and function parameters, never line order alone) "
                    "and the run record shows the endpoint READY serving the registered version"],
        "fails": "no download call; HF_HUB_DISABLE_XET set only after the download or import, or to a "
                 "value that does not disable Xet ('0', 'false'); a subprocess without it; and no "
                 "recorded evidence",
        "evidence": "code + record",
    },
    "model_loads_with_pinned_deps": {
        "severity": "critical",
        "goal": "The served model loads: transformers 5.x is excluded (it breaks TeddyGModel) or not "
                "used, and the endpoint reached READY.",
        "accepts": ["transformers pinned so 5.x is excluded (==4.41.0, ==4.x, <5, ~=4.x) in the serving "
                    "requirements, or in %pip when the serving pin is the installed version "
                    "(transformers=={transformers.__version__})",
                    "a model path that does not use transformers (plain-PyTorch re-implementation: "
                    "no transformers or upstream-package import, none in the serving requirements)"],
        "fails": "a transformers spec that allows 5.x, transformers used but never pinned, or the "
                 "endpoint not READY (run record outcome/endpoint_ready)",
        "evidence": "code + record",
    },
    "fidelity_verified": {
        "severity": "major",
        "goal": "The wrapper's or endpoint's embeddings were compared against an upstream reference.",
        "accepts": ["a comparison (cosine, allclose / isclose, largest absolute difference) whose one "
                    "side is derived from upstream code — a call into the upstream package (e.g. "
                    "TEDDY's tokenizer or tokenization functions), or a direct forward pass of the "
                    "upstream model (`<x>.model(...)`) in a notebook that loads the upstream package "
                    "and defines no nn.Module of its own — and whose other side comes from the "
                    "candidate (a `.predict(...)` of the wrapper / logged model, or an endpoint "
                    "query). Values are followed through assignments, loops and notebook functions "
                    "across cells"],
        "fails": "no such comparison; comparisons of two candidate outputs (endpoint vs the same "
                 "wrapper); a re-implementation (own nn.Module) compared only with itself (a riskier "
                 "valid alternative: scientific-risk check); an upstream call that does not feed a "
                 "side of the comparison",
        "evidence": "code",
    },
    "signature_and_example": {
        "severity": "critical",
        "goal": "The logged model carries a signature and an input example (Unity Catalog needs the "
                "signature; the example grounds it in a real request).",
        "accepts": ["mlflow.<flavor>.log_model(signature=..., input_example=...) with the signature "
                    "built by infer_signature(...) or ModelSignature(...)",
                    "another signature expression, if the run record shows the endpoint READY serving "
                    "the registered version (the registration, which needs a signature, succeeded)"],
        "fails": "no log_model call with both; a signature that is None, or unrecognisable without "
                 "that recorded evidence",
        "evidence": "code (+ record)",
    },
    "registered_in_uc": {
        "severity": "major",
        "goal": "The model is registered in Unity Catalog.",
        "accepts": ["scorers.uc_provenance()['uc_registry'] (the new check's logic, unchanged)"],
        "fails": "as that check",
        "evidence": "code",
    },
    "serves_valid_embeddings": {
        "severity": "critical",
        "goal": "The endpoint returns valid embeddings (one finite vector of the model size per cell, "
                "in its own declared input format).",
        "accepts": ["run record outcome/returns_valid_embeddings"],
        "fails": "not passed or not measured",
        "evidence": "record",
    },
    "serves_expression_input": {
        "severity": "major",
        "goal": "The endpoint embeds a cell from its expression (not pre-tokenized token IDs).",
        "accepts": ["run record outcome/accepts_expression_input"],
        "fails": "not passed or not measured",
        "evidence": "record",
    },
    "endpoint_smoke_tested": {
        "severity": "major",
        "goal": "The run queried its own endpoint and checked the result.",
        "accepts": ["an endpoint query (serving_endpoints.query, or a POST to .../invocations) whose "
                    "result feeds an assert, a comparison (==, <, >, ...) or a numeric comparison "
                    "(cosine, allclose, absolute difference); values followed across cells and "
                    "through notebook functions"],
        "fails": "no endpoint query, or its result only printed / displayed",
        "evidence": "code",
    },
    "existing_resources_checked": {
        "severity": "major",
        "goal": "The run looked for an existing endpoint / table / index before creating or deleting it.",
        "accepts": ["scorers.existing_resources() (the new check's logic, unchanged) for the resources "
                    "listed for the task in expectations.json new_checks"],
        "fails": "as that check, for any listed resource",
        "evidence": "code",
    },
    "governance_logging": {
        "severity": "major",
        "goal": "Requests are governed: AI Gateway inference table AND usage tracking enabled.",
        "accepts": ["scorers.ai_gateway_logging(): both enabled on one attached AI Gateway config "
                    "(the new checks' logic, unchanged)"],
        "fails": "as those checks",
        "evidence": "code (run record endpoint_config quoted in the reason)",
    },
    "provenance_recorded": {
        "severity": "major",
        "goal": "The upstream revision is pinned to a commit and the licence is recorded in a "
                "persisted record.",
        "accepts": ["revision: the Hugging Face download uses a 40-character hexadecimal commit SHA, or "
                    "the commit resolved at download time (`model_info(...).sha`) when that same value "
                    "is recorded under a revision/commit/sha key in a persisted sink — followed from the "
                    "resolved variable, or back from a subprocess through its printed output or a file "
                    "it wrote (also after a shutil copy) — or a 40-character SHA literal is recorded",
                    "licence: a non-empty licence value (not unknown / none / n/a) under a licence key "
                    "in a persisted sink — MLflow params or tags (set_tag(s), log_param(s), "
                    "start_run / log_model tags, log_dict / log_text), registered-model or "
                    "model-version tags, or a manifest file (json.dump to a file opened for writing, "
                    "Path(...).write_text)"],
        "fails": "no revision, a branch or tag name ('main', 'dev', 'v1'), an unresolved expression; a "
                 "resolved commit whose recorded value is something else (e.g. 'main'); a licence that "
                 "is missing, empty or unknown, only in comments, markdown or a model description, or "
                 "json.dump-ed to something that is not a file (io.StringIO())",
        "evidence": "code",
    },
    "similarity_search_works": {
        "severity": "critical",
        "goal": "teddy-002 only: the index is ready and a nearest-neighbour query returns a cell-type "
                "column.",
        "accepts": ["run record outcome/index_ready AND outcome/nn_returns_cell_type"],
        "fails": "either not passed or not measured",
        "evidence": "record",
        "search_only": True,
    },
}

GUIDELINES = {
    "gpu_serving": {
        "goal": "Served on a GPU workload, as the task guidelines suggest (GPU_SMALL). The prompts do "
                "not ask for a GPU and CPU endpoints worked, so this is reported on its own, not "
                "counted as a failure.",
        "accepts": ["run record endpoint_config: every served entity has a GPU_* workload type "
                    "(recorded settings only; code is never read)"],
        "context": "CPU endpoints were slower per cell in the recorded runs (round 1 baseline/teddy-002: "
                   "about 15 s per cell; round 2 baselines: about 1.7-3.3 s per cell, run record notes)",
        "evidence": "record",
    },
}


def checks_for(task: dict) -> list:
    """Practical check names that apply to a task (similarity_search_works: search tasks only)."""
    return [name for name, meta in CHECKS.items() if not meta.get("search_only") or task.get("vs")]


# ---- Notebook code with file line numbers ------------------------------------------------

_CELL_SEPARATOR = re.compile(r"^# COMMAND -+[ \t]*$")
_NOT_PYTHON = ("# MAGIC", "# Databricks notebook source", "%", "!")   # the prefixes scorers.code_cells drops


def _line_preserving(response: str) -> str:
    """The notebook rewritten so that scorers.code_cells() keeps every line in place: non-Python
    lines are blanked instead of dropped and each cell is padded with blank lines, so `ast`
    line numbers in the parsed cells are line numbers of the exported file."""
    cells, body, start = [], [], 1
    for number, line in enumerate(response.splitlines(), 1):
        if _CELL_SEPARATOR.match(line):
            cells.append((start, body))
            body, start = [], number + 1
        else:
            body.append("" if line.lstrip().startswith(_NOT_PYTHON) else line)
    cells.append((start, body))
    # re.split in code_cells leaves the newline after each separator at the start of the next cell
    return "\n# COMMAND ----------\n".join(
        "\n" * (first - (1 if i == 0 else 2)) + "\n".join(lines) for i, (first, lines) in enumerate(cells))


def _target_names(target) -> list:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for elt in target.elts for n in _target_names(elt)]
    return []


RUNTIME = "<value known at run time>"


class PracticalCode(NotebookCode):
    """NotebookCode whose node line numbers are notebook file lines, plus what NotebookCode does
    not follow: `with ... as x` targets, loop / comprehension / augmented-assignment /
    tuple-unpacking sources and the notebook's own functions."""

    def __init__(self, response: str):
        super().__init__(_line_preserving(response))
        self.with_values, self.sources, self.functions = {}, {}, {}
        for tree in self.trees:
            for node in ast.walk(tree):
                if isinstance(node, (ast.With, ast.AsyncWith)):
                    for item in node.items:
                        if isinstance(item.optional_vars, ast.Name):
                            self.with_values.setdefault(item.optional_vars.id, []).append(item.context_expr)
                elif isinstance(node, (ast.For, ast.comprehension)):
                    for name in _target_names(node.target):
                        self.sources.setdefault(name, []).append(node.iter)
                elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    self.sources.setdefault(node.target.id, []).append(node.value)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, (ast.Tuple, ast.List)):     # a, b = ...: both from the value
                            for name in _target_names(target):
                                self.sources.setdefault(name, []).append(node.value)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.functions[node.name] = node

    def nodes(self, kind):
        return [node for tree in self.trees for node in ast.walk(tree) if isinstance(node, kind)]

    def string_constants(self) -> list:
        return [n for n in self.nodes(ast.Constant) if isinstance(n.value, str)]

    def imports_package(self, packages: tuple) -> bool:
        return any(path.split(".")[0] in packages for path in self.imports.values())

    def value_texts(self, node) -> list:
        """What a recorded value can be: its string values, "" for None, RUNTIME for a value
        only known when the notebook runs (e.g. a commit read from the download)."""
        out = []
        for value in self.values(node):
            if isinstance(value, ast.Constant):
                out.append("" if value.value is None else str(value.value))
            else:
                out += [s for s in self.strings(value) if s.replace("{}", "").strip()] or [RUNTIME]
        return out


# ---- Where a value comes from (across cells) -------------------------------------------------

UPSTREAM, PREDICT, ENDPOINT = "upstream", "candidate predict", "endpoint query"


_HTTP_CALLS = ("post", "do", "request", "urlopen")   # requests.post, api_client.do, urllib urlopen, ...


def _is_endpoint_query(nb: PracticalCode, call: ast.Call) -> bool:
    """serving_endpoints.query(...), or an HTTP call (not e.g. print) to a .../invocations URL."""
    if any(d.endswith("serving_endpoints.query") for d in nb.dotted(call.func)):
        return True
    if _call_name(call) not in _HTTP_CALLS:
        return False
    args = list(call.args) + [k.value for k in call.keywords]
    return any("/invocations" in s for a in args for s in nb.strings(a))


def _reaching(nb: PracticalCode, name: str, line: int) -> list:
    """Values a name can hold when used at `line`: its latest assignment (or loop / comprehension /
    augmented assignment / with / tuple-unpacking source) up to that line. Inside a notebook
    function, a name with no earlier source is a global set later: all its sources count."""
    found = nb.assigned.get(name, []) + nb.sources.get(name, []) + nb.with_values.get(name, [])
    before = [v for v in found if getattr(v, "lineno", 0) <= line]
    if not before:
        in_function = any(f.lineno <= line <= f.end_lineno for f in nb.functions.values())
        return found if in_function else []
    latest = max(v.lineno for v in before)
    return [v for v in before if v.lineno == latest]


def flow_nodes(nb: PracticalCode, node, _seen=None):
    """Every AST node a value is derived from. Names are followed through assignments, loop /
    augmented-assignment sources and `with` targets anywhere in the notebook (the latest one
    before the use, see _reaching); a call to a notebook function also yields its body."""
    seen = set() if _seen is None else _seen
    for n in ast.walk(node):
        yield n
        if isinstance(n, ast.Call):
            name = n.func.id if isinstance(n.func, ast.Name) else None
            if name in nb.functions and ("def", name) not in seen:
                seen.add(("def", name))
                yield from flow_nodes(nb, nb.functions[name], seen)
        elif isinstance(n, ast.Name) and (n.id, n.lineno) not in seen:
            seen.add((n.id, n.lineno))
            for value in _reaching(nb, n.id, n.lineno):
                yield from flow_nodes(nb, value, seen)


def origins(nb: PracticalCode, node, packages: tuple, upstream_model: bool) -> set:
    """What a value is derived from (see flow_nodes): UPSTREAM (a call into the upstream
    package, or a direct forward pass `<x>.model(...)` when `upstream_model`), PREDICT (a
    `.predict(...)` of the wrapper / logged model) and/or ENDPOINT (a serving-endpoint query)."""
    out = set()
    for n in flow_nodes(nb, node):
        if not isinstance(n, ast.Call):
            continue
        called = nb.dotted(n.func)
        if any(d.split(".")[0] in packages and "." in d for d in called):
            out.add(UPSTREAM)
        elif upstream_model and any(d.endswith(".model") for d in called):
            out.add(UPSTREAM)
        if any(d.endswith(".predict") for d in called):
            out.add(PREDICT)
        if _is_endpoint_query(nb, n):
            out.add(ENDPOINT)
    return out


_COMPARE_CALLS = ("allclose", "assert_allclose", "isclose", "assert_close", "cosine_similarity")


def _product_sides(node):
    """The two operands of the first product (a @ b, a * b, dot(a, b)) inside a cosine numerator."""
    for n in ast.walk(node):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.MatMult, ast.Mult)):
            return [n.left, n.right]
        if isinstance(n, ast.Call) and _call_name(n) == "dot" and len(n.args) >= 2:
            return list(n.args[:2])
    return None


def numeric_comparisons(nb: PracticalCode) -> list:
    """(line, [side a, side b]) for every output comparison: allclose-style calls, a cosine
    (product / norm) and an absolute difference abs(a - b)."""
    out = []
    for node in nb.nodes((ast.Call, ast.BinOp)):
        if isinstance(node, ast.Call) and _call_name(node) in _COMPARE_CALLS and len(node.args) >= 2:
            out.append((node.lineno, list(node.args[:2])))
        elif (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
              and any(isinstance(n, ast.Call) and _call_name(n) == "norm" for n in ast.walk(node.right))):
            sides = _product_sides(node.left)
            if sides:
                out.append((node.lineno, sides))
        elif (isinstance(node, ast.Call) and _call_name(node) == "abs" and node.args
              and isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].op, ast.Sub)):
            out.append((node.lineno, [node.args[0].left, node.args[0].right]))
    return sorted(out, key=lambda item: item[0])


# ---- Environment: HF_HUB_DISABLE_XET -----------------------------------------------------------

XET_VAR = "HF_HUB_DISABLE_XET"
_TRUTHY = {"1", "true", "yes", "on"}        # values huggingface_hub reads as "disable"


def _is_truthy_text(text: str) -> bool:
    return text.strip().strip("\"'").lower() in _TRUTHY


def _disables(nb: PracticalCode, node) -> bool:
    """The value set for XET_VAR is a disabling one ("1", "true", "yes", "on", any case; True; 1)."""
    if node is None:
        return False
    return any((isinstance(v, ast.Constant) and (v.value is True or v.value == 1
                                                  or (isinstance(v.value, str) and _is_truthy_text(v.value))))
               for v in nb.values(node))


def _mapping_disables(nb: PracticalCode, node) -> bool:
    """A dict / dict(...) value whose XET_VAR entry disables Xet."""
    return any(XET_VAR in m and _disables(nb, m[XET_VAR]) for v in nb.values(node) for m in nb.members(v))


def _xet_set_lines(nb: PracticalCode, receiver) -> list:
    """Lines where XET_VAR is set to a disabling value on an environment mapping: `receiver(node)`
    says whether an expression is that mapping (os.environ, or an env dict for a subprocess)."""
    lines = []
    for node in nb.nodes(ast.Assign):
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and receiver(target.value)
                    and XET_VAR in nb.strings(target.slice) and _disables(nb, node.value)):
                lines.append(node.lineno)
    for call in nb.calls:
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "update" and receiver(func.value):
            if any(_mapping_disables(nb, a) for a in call.args) or any(
                    k.arg == XET_VAR and _disables(nb, k.value) for k in call.keywords):
                lines.append(call.lineno)
        elif (isinstance(func, ast.Attribute) and func.attr == "setdefault" and receiver(func.value)
              or any(d.endswith("putenv") for d in nb.dotted(func))):
            if len(call.args) >= 2 and XET_VAR in nb.strings(call.args[0]) and _disables(nb, call.args[1]):
                lines.append(call.lineno)
    return sorted(lines)


def _is_os_environ(nb: PracticalCode):
    return lambda node: any(d.endswith("environ") for d in nb.dotted(node))


def _in_process_xet_lines(nb: PracticalCode, response: str = "") -> list:
    lines = _xet_set_lines(nb, _is_os_environ(nb))
    lines += [i for i, text in enumerate(response.splitlines(), 1)
              if (m := re.match(rf"^\s*(?:# MAGIC\s*)?%env\s+{XET_VAR}\s*=\s*(\S+)", text))
              and _is_truthy_text(m.group(1))]
    return sorted(lines)


def _first_hf_import(nb: PracticalCode):
    lines = [n.lineno for n in nb.nodes((ast.Import, ast.ImportFrom))
             if (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "huggingface_hub")
             or (isinstance(n, ast.Import) and any(a.name.split(".")[0] == "huggingface_hub" for a in n.names))]
    return min(lines) if lines else None


def _set_before_import(nb: PracticalCode, response: str, before: int):
    """(ok, reason) for XET_VAR set in-process before the first huggingface_hub import and `before`."""
    sets = _in_process_xet_lines(nb, response)
    first_import = _first_hf_import(nb)
    limit = min(x for x in (first_import, before) if x is not None)
    early = [line for line in sets if line < limit]
    if early:
        return True, f"set at line {early[0]}, before the huggingface_hub import (line {first_import})"
    if sets:
        return False, f"set at line {sets[0]}, after the huggingface_hub import / download (line {limit})"
    return False, "not set to a disabling value (1 / true / yes / on)"


# ---- Hugging Face downloads -------------------------------------------------------------

_HF_DOWNLOADS = ("snapshot_download", "hf_hub_download")
_SUBPROCESS = re.compile(r"^(?:subprocess\.(?:run|call|check_call|check_output|Popen)|os\.system)$")
_CLI_DOWNLOAD = re.compile(r"\b(?:huggingface-cli|hf)\s+download\b(?P<rest>.*)")
_SHA = re.compile(r"^[0-9a-f]{40}$")
ACCEPTED_LOCATIONS = {"tmp", "volume", "tempfile"}


class Download:
    """One Hugging Face download: line, how it ran, destination kinds, revision, Xet setting."""

    def __init__(self, line: int, how: str, kinds: set, revision: tuple, xet: tuple, resolved=None,
                 paths=(), node=None):
        self.line, self.how, self.kinds = line, how, kinds
        self.paths = [p for p in paths if p.replace("{}", "").strip("/")]   # destination strings
        self.node = node              # the download call in the notebook (its return value is the path)
        self.revision = revision      # (kind, text): kind "sha" | "resolved" | None
        self.xet = xet                # (ok, reason) for HF_HUB_DISABLE_XET
        # for a resolved commit: is a notebook node the source of that commit? (see _resolved_sources)
        self.resolved = resolved or (lambda node: False)

    def describe(self) -> str:
        return f"line {self.line} ({self.how}): {', '.join(sorted(self.kinds))}"


def _classify(path: str) -> str:
    if path.startswith("/local_disk0"):
        return "local_disk0"
    if path.startswith("/tmp"):
        return "tmp"
    if path.startswith("/Volumes/"):
        return "volume"
    return "other"


def _path_kinds(nb: PracticalCode, node, depth: int = 0) -> set:
    """Kinds of location a path expression can stand for: tmp, volume, tempfile, local_disk0,
    other or unknown (the leading part of an f-string, a + / Path / os.path.join decides)."""
    if depth > 8:
        return {"unknown"}
    if isinstance(node, ast.Name) and node.id not in nb.assigned and node.id in nb.with_values:
        candidates = nb.with_values[node.id]
    else:
        candidates = nb.values(node)
    kinds = set()
    for value in candidates:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            kinds.add(_classify(value.value))
        elif isinstance(value, ast.JoinedStr) and value.values:
            head = value.values[0]
            kinds |= ({_classify(str(head.value))} if isinstance(head, ast.Constant)
                      else _path_kinds(nb, head.value, depth + 1))
        elif isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Div)):
            kinds |= _path_kinds(nb, value.left, depth + 1)
        elif isinstance(value, ast.Call):
            called = nb.dotted(value.func)
            if any(d.endswith(("mkdtemp", "TemporaryDirectory", "gettempdir")) for d in called):
                kinds.add("tempfile")
            elif any(d.endswith(("path.join", "Path")) for d in called) and value.args:
                kinds |= _path_kinds(nb, value.args[0], depth + 1)
            else:
                kinds.add("unknown")
        elif isinstance(value, ast.Attribute) and value.attr == "name":   # TemporaryDirectory().name
            kinds |= _path_kinds(nb, value.value, depth + 1)
        else:
            kinds.add("unknown")
    return kinds or {"unknown"}


def _revision(nb: PracticalCode, kwargs: dict) -> tuple:
    """(kind, text) of a download's revision=: ("sha", sha) for a 40-hex commit, ("resolved", how)
    for the commit resolved with model_info(...).sha, else (None, why)."""
    node = kwargs.get("revision")
    if node is None:
        return None, "no revision"
    texts = [t for t in nb.strings(node) if "{}" not in t]
    if texts and all(_SHA.match(t) for t in texts):
        return "sha", texts[0]
    if any(d.endswith("model_info().sha") for v in nb.values(node) for d in nb.dotted(v)):
        return "resolved", "commit resolved with model_info(...).sha"
    return None, f"revision {texts[0]!r} is not a commit SHA" if texts else "revision is an unresolved expression"


def _direct_downloads(nb: PracticalCode, response: str, how: str = "notebook", line: int | None = None,
                      xet=None) -> list:
    out = []
    for call in nb.calls:
        name = _call_name(call)
        if name not in _HF_DOWNLOADS:
            continue
        if not any(d == name or d.startswith("huggingface_hub") for d in nb.dotted(call.func)):
            continue
        for kwargs in nb.keyword_maps(call):
            dest = kwargs.get("local_dir") or kwargs.get("cache_dir")
            kinds = _path_kinds(nb, dest) if dest is not None else {"hf_default_cache"}
            setting = xet or _set_before_import(nb, response, call.lineno)
            sources = _resolved_nodes(nb, kwargs.get("revision"))
            out.append(Download(line or call.lineno, how, kinds, _revision(nb, kwargs), setting,
                                lambda node, s=sources: any(node is x for x in s),
                                paths=nb.strings(dest) if dest is not None else [],
                                node=None if line else call))
    return out


def _resolved_nodes(nb: PracticalCode, revision) -> list:
    """The `model_info(...).sha` expressions a download's revision= comes from."""
    if revision is None:
        return []
    return [v for v in nb.values(revision) if any(d.endswith("model_info().sha") for d in nb.dotted(v))]


def _derives(nb: PracticalCode, node, test) -> bool:
    return any(test(n) for n in flow_nodes(nb, node))


def _written_paths(nb: PracticalCode, test) -> set:
    """Paths of files the code writes a value matching `test` to: json.dump(x, open(P, "w")),
    open(P, "w").write(x), Path(P).write_text(x)."""
    paths = set()
    for call in nb.calls:
        name = _call_name(call)
        if name == "dump" and len(call.args) >= 2 and _derives(nb, call.args[0], test):
            paths |= {s for v in _file_targets(nb, call.args[1]) for s in nb.strings(v.args[0])}
        elif name in ("write", "write_text") and call.args and _derives(nb, call.args[0], test):
            holder = call.func.value if isinstance(call.func, ast.Attribute) else None
            for v in (nb.values(holder) if holder is not None else []):
                if isinstance(v, ast.Call) and v.args and _call_name(v) in ("open", "Path"):
                    paths |= set(nb.strings(v.args[0]))
    return paths


def _resolved_sources(nb: PracticalCode, call: ast.Call, script_nb: PracticalCode, download) -> callable:
    """For a download in a subprocess script with a resolved commit: notebook nodes that carry
    that commit back — the subprocess call itself (its output) when the script prints it, and a
    read of a file the script wrote it to (directly or through a shutil copy)."""
    in_script = download.resolved
    prints = any(_call_name(c) == "print" and any(_derives(script_nb, a, in_script) for a in c.args)
                 for c in script_nb.calls)
    files = _written_paths(script_nb, in_script)
    for copy in [c for c in nb.calls if _call_name(c) in _STAGING_CALLS and len(c.args) >= 2]:
        if set(nb.strings(copy.args[0])) & files:
            files |= set(nb.strings(copy.args[1]))

    def carries(node) -> bool:
        if prints and node is call:
            return True
        return (isinstance(node, ast.Call) and _call_name(node) in ("open", "Path", "read_text") and node.args
                and bool(set(nb.strings(node.args[0])) & files))
    return carries


def _subprocess_xet(nb: PracticalCode, call: ast.Call, script_nb: PracticalCode, script: str) -> tuple:
    """(ok, reason): XET_VAR in the subprocess env= mapping, or set in the script before its
    huggingface_hub import."""
    env = next((k.value for k in call.keywords if k.arg == "env"), None)
    if env is not None:
        in_dict = _mapping_disables(nb, env)
        names = {env.id} if isinstance(env, ast.Name) else set()
        on_name = _xet_set_lines(nb, lambda node: isinstance(node, ast.Name) and node.id in names)
        if in_dict or on_name:
            return True, f"set in the subprocess env (line {on_name[0] if on_name else call.lineno})"
    ok, why = _set_before_import(script_nb, script, 10 ** 9)
    return (True, "set in the subprocess script before its huggingface_hub import") if ok else (
        False, f"subprocess without it (script: {why}; env: {'not given' if env is None else 'no disabling ' + XET_VAR})")


def _subprocess_downloads(nb: PracticalCode) -> list:
    """Downloads in a Python script run with subprocess / os.system. A destination the script
    takes from its own argv is read from the path arguments of the subprocess call."""
    out = []
    for call in nb.calls:
        if not any(_SUBPROCESS.match(d) for d in nb.dotted(call.func)):
            continue
        elements = []
        for arg in list(call.args) + [k.value for k in call.keywords if k.arg == "args"]:
            for value in nb.values(arg):
                elements += value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
        texts = list(dict.fromkeys(t for e in elements for t in nb.strings(e)))   # unique, in order
        # path-like arguments, for a script that reads its destination from sys.argv
        path_args = set().union(*[_path_kinds(nb, e) for e in elements]) & (ACCEPTED_LOCATIONS | {"local_disk0"})
        for script in [t for t in texts if re.search(r"\b(?:snapshot_download|hf_hub_download)\s*\(", t)]:
            script_nb = PracticalCode(script)
            xet = _subprocess_xet(nb, call, script_nb, script)
            arg_paths = [t for e in elements for t in nb.strings(e) if t.startswith("/")]
            for download in _direct_downloads(script_nb, script, "subprocess", call.lineno, xet):
                if download.kinds <= {"unknown"} and path_args:
                    download.kinds = path_args
                download.paths = download.paths + [t for t in arg_paths if t.replace("{}", "").strip("/")]
                download.resolved = _resolved_sources(nb, call, script_nb, download)
                out.append(download)
        for command in texts:
            match = _CLI_DOWNLOAD.search(command)
            if match:
                on_line = _shell_disables(command)
                xet = (on_line, "set on the command line" if on_line else "not set to a disabling value")
                out.append(_cli_download(match.group("rest"), call.lineno, xet))
    return out


def _cli_download(rest: str, line: int, xet: tuple) -> Download:
    try:
        words = shlex.split(rest)
    except ValueError:
        words = rest.split()
    dest = next((words[i + 1] for i, w in enumerate(words[:-1]) if w in ("--local-dir", "--cache-dir")), None)
    revision = next((words[i + 1] for i, w in enumerate(words[:-1]) if w == "--revision"), None)
    kinds = {_classify(dest)} if dest else {"hf_default_cache"}
    rev = ("sha", revision) if revision and _SHA.match(revision) else (
        None, f"revision {revision!r} is not a commit SHA" if revision else "no revision")
    return Download(line, "shell", kinds, rev, xet, paths=[dest] if dest else [])


def _shell_disables(command: str) -> bool:
    """`HF_HUB_DISABLE_XET=<disabling value>` given in a shell command (before the download)."""
    return any(_is_truthy_text(v) for v in re.findall(rf"\b{XET_VAR}=(\S+)", command))


def _shell_downloads(response: str) -> list:
    """`%sh` / `!` cells: `huggingface-cli download` or `hf download` (run in their own process)."""
    out, in_sh, exported = [], False, False
    for number, line in enumerate(response.splitlines(), 1):
        if _CELL_SEPARATOR.match(line):
            in_sh = exported = False
            continue
        magic = re.match(r"^\s*# MAGIC\s?(.*)", line)
        text = magic.group(1) if magic else line
        if magic and text.lstrip().startswith("%"):
            in_sh = text.lstrip().startswith("%sh")     # a cell's magic decides its language
        if not ((in_sh and magic) or text.lstrip().startswith("!")):
            continue
        export = re.search(rf"\bexport\s+{XET_VAR}=(\S+)", text)
        if export:
            exported = _is_truthy_text(export.group(1))
        match = _CLI_DOWNLOAD.search(text)
        if match:
            on_line = _shell_disables(text[:match.start()])
            xet = (exported or on_line, "exported earlier in the cell" if exported else
                   "set on the command line" if on_line else "not set")
            out.append(_cli_download(match.group("rest"), number, xet))
    return out


def downloads(nb: PracticalCode, response: str) -> list:
    return sorted(_direct_downloads(nb, response) + _subprocess_downloads(nb) + _shell_downloads(response),
                  key=lambda d: d.line)


# ---- Persisted records (MLflow, registry, manifest) ---------------------------------------------

_LICENCE_KEY = re.compile(r"licen[cs]e", re.I)
_REVISION_KEY = re.compile(r"revision|commit|\bsha\b|_sha$", re.I)
_EMPTY = {"", "unknown", "none", "null", "n/a", "na", "tbd", "todo", "?"}
# Persisted sinks: call name -> position of the key argument (keyed) or the arguments that hold
# a dict (None = all). A model description is not a sink: it is free text, not a recorded field.
_KEYED_SINKS = {"set_tag": 0, "log_param": 0, "set_registered_model_tag": 1, "set_model_version_tag": 2}
_DICT_SINKS = {"set_tags": None, "log_params": None, "set_logged_model_tags": None, "log_dict": None,
               "log_text": None, "dump": None, "write_text": None, "log_model": ("tags", "metadata"),
               "start_run": ("tags",)}
_FILE_MODES = re.compile(r"[wax]")


def _file_targets(nb: PracticalCode, node) -> list:
    """open(path, "w"/"a"/"x") calls a json.dump target can stand for (a name or `with` target)."""
    candidates = nb.values(node) + (nb.with_values.get(node.id, []) if isinstance(node, ast.Name) else [])
    out = []
    for v in candidates:
        if isinstance(v, ast.Call) and _call_name(v) == "open" and v.args:
            mode = v.args[1] if len(v.args) > 1 else next((k.value for k in v.keywords if k.arg == "mode"), None)
            if mode is not None and any(_FILE_MODES.search(m) for m in nb.strings(mode)):
                out.append(v)
    return out


def _persists(nb: PracticalCode, call: ast.Call) -> bool:
    """json.dump writes to a file opened for writing (not e.g. io.StringIO()); write_text is
    called on a Path; the other sinks are MLflow / registry calls."""
    name = _call_name(call)
    if name == "dump":
        return len(call.args) >= 2 and bool(_file_targets(nb, call.args[1]))
    if name == "write_text":
        holder = call.func.value if isinstance(call.func, ast.Attribute) else None
        return holder is not None and any(d.endswith("Path()") or "Path()" in d for d in nb.dotted(holder))
    return True


def _filled(texts: list) -> bool:
    """A non-empty value: a string that is not empty / unknown, or a value known at run time."""
    return any(t.strip().lower() not in _EMPTY for t in texts)


def _keyed_values(nb: PracticalCode, node, key: re.Pattern, depth: int = 0) -> list:
    """String values under keys matching `key` in a dict / constructor value (nested dicts too);
    a nested dict under a matching key contributes all its string values."""
    out = []
    for fields in nb.members(node):
        for name, value in fields.items():
            if value is None:
                continue
            if key.search(name):
                out += nb.value_texts(value) + [s for m in nb.members(value) for v in m.values()
                                                  if v is not None for s in nb.value_texts(v)]
            elif depth < 4:
                out += _keyed_values(nb, value, key, depth + 1)
    return out


def _keyed_nodes(nb: PracticalCode, node, key: re.Pattern, depth: int = 0) -> list:
    """Value nodes under keys matching `key` (nested dicts too), for following where they came from."""
    out = []
    for fields in nb.members(node):
        for name, value in fields.items():
            if value is None:
                continue
            if key.search(name):
                out.append(value)
            elif depth < 4:
                out += _keyed_nodes(nb, value, key, depth + 1)
    return out


def persisted_records(nb: PracticalCode, key: re.Pattern) -> list:
    """(line, sink, values, value nodes) for every persisted record under a key matching `key`
    (the nodes are the recorded values themselves, for following where they came from)."""
    out = []
    for call in nb.calls:
        name = _call_name(call)
        if name in _KEYED_SINKS:
            position = _KEYED_SINKS[name]
            args = list(call.args)
            if len(args) > position and any(key.search(s) for s in nb.strings(args[position])):
                value = args[position + 1] if len(args) > position + 1 else next(
                    (k.value for k in call.keywords if k.arg == "value"), None)
                out.append((call.lineno, name, nb.value_texts(value) if value is not None else [""],
                            [value] if value is not None else []))
        elif name in _DICT_SINKS and _persists(nb, call):
            wanted = _DICT_SINKS[name]
            nodes = list(call.args) + [k.value for k in call.keywords] if wanted is None else [
                k.value for k in call.keywords if k.arg in wanted]
            values = [v for n in nodes for v in _keyed_values(nb, n, key)]
            if values or any(key.search(k) for n in nodes for m in nb.members(n) for k in m):
                out.append((call.lineno, {"dump": "json.dump manifest file", "write_text": "manifest file"}
                            .get(name, name), values, [v for n in nodes for v in _keyed_nodes(nb, n, key)]))
    # for k, v in {...}.items(): <keyed sink>(..., k, v)
    for loop in nb.nodes(ast.For):
        it = loop.iter
        if not (isinstance(it, ast.Call) and _call_name(it) == "items" and isinstance(it.func, ast.Attribute)):
            continue
        sinks = [c for stmt in loop.body for c in ast.walk(stmt)
                 if isinstance(c, ast.Call) and _call_name(c) in _KEYED_SINKS]
        values = _keyed_values(nb, it.func.value, key)
        if sinks and values:
            out.append((sinks[0].lineno, _call_name(sinks[0]), values, _keyed_nodes(nb, it.func.value, key)))
    return sorted(out, key=lambda item: item[0])


# ---- Run-record evidence ---------------------------------------------------------------

def _measured(record: dict | None, check: str):
    """passed (True/False) of an outcome check in the run record, or None if not measured."""
    entry = ((record or {}).get("outcome") or {}).get(check)
    return None if entry is None else entry.get("passed")


def _record_checks(record, names: list) -> tuple:
    values = {n: _measured(record, n) for n in names}
    missing = [n for n, v in values.items() if v is None]
    if missing:
        return False, f"not measured in the run record: {missing}"
    failed = [n for n, v in values.items() if not v]
    return not failed, ("run record: " + ", ".join(f"{n} {'pass' if v else 'fail'}" for n, v in values.items()))


def _endpoint_serves_registration(record) -> bool:
    return bool(_measured(record, "outcome/endpoint_ready") and _measured(record, "outcome/serves_recorded_version"))


# ---- The checks --------------------------------------------------------------------------
# Each takes (nb, response, record, task) and returns (passed, reason).

_STAGING_CALLS = ("copytree", "copy", "copy2", "copyfile", "move")


def assigned_values(stmts: list) -> dict:
    """{name: [value nodes]} for plain assignments anywhere inside `stmts`."""
    out = {}
    for stmt in stmts:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        out.setdefault(t.id, []).append(n.value)
    return out


def local_disk0_fallbacks(nb) -> dict:
    """{name: (line of the try / if)} for a path variable set to /local_disk0 on one side and to an
    accepted location (/tmp, tempfile, Volume) on the other: try body vs a handler, or if vs else.
    These are the compute-adaptive paths: the alternative is assigned to the same name."""
    out = {}
    pairs = [(t.body, [s for h in t.handlers for s in h.body] + t.orelse, t.lineno) for t in nb.nodes(ast.Try)]
    pairs += [(i.body, i.orelse, i.lineno) for i in nb.nodes(ast.If)] + [(i.orelse, i.body, i.lineno) for i in nb.nodes(ast.If)]
    for first, other, line in pairs:
        a, b = assigned_values(first), assigned_values(other)
        for name in set(a) & set(b):
            if (any("local_disk0" in _path_kinds(nb, v) for v in a[name])
                    and any(_path_kinds(nb, v) & ACCEPTED_LOCATIONS for v in b[name])):
                out[name] = line
    return out


def uses_fallback(nb, node, fallbacks: dict) -> bool:
    return node is not None and any(isinstance(n, ast.Name) and n.id in fallbacks for n in flow_nodes(nb, node))


def download_site(nb, d):
    """The notebook node a download's destination flows from: its local_dir / cache_dir keyword,
    or, for a download inside a subprocess script, the subprocess call's arguments (the script
    text and argv), whose formatted values carry the notebook's path variables. None if unknown."""
    if d.how == "subprocess":
        call = next((c for c in nb.calls if c.lineno == d.line
                     and any(_SUBPROCESS.match(x) for x in nb.dotted(c.func))), None)
        return ast.Tuple(elts=list(call.args), ctx=ast.Load()) if call is not None else None
    if d.node is None:
        return None
    return next((m.get("local_dir") or m.get("cache_dir") for m in nb.keyword_maps(d.node)
                 if m.get("local_dir") or m.get("cache_dir")), None)


def staging_dst(call: ast.Call):
    """The destination argument of a shutil copy / move (positional or dst=), or None."""
    return call.args[1] if len(call.args) > 1 else next((k.value for k in call.keywords if k.arg == "dst"), None)


def local_disk0_staging(nb) -> list:
    """shutil copy / move calls whose destination can be under /local_disk0."""
    return [c for c in nb.calls if _call_name(c) in _STAGING_CALLS
            and any(d.startswith("shutil.") for d in nb.dotted(c.func))
            and staging_dst(c) is not None and "local_disk0" in _path_kinds(nb, staging_dst(c))]


def download_location_ok(nb, response, record, task):
    found = downloads(nb, response)
    if not found:
        return False, "no Hugging Face download call found"
    # /local_disk0 counts only when compute-adaptive: the path variable has an accepted alternative
    # (try handler / other if branch) that is then used, so serverless gets the alternative
    fallbacks = local_disk0_fallbacks(nb)
    staging = [c.lineno for c in local_disk0_staging(nb) if not uses_fallback(nb, staging_dst(c), fallbacks)]
    bad = [d for d in found if d.kinds - ACCEPTED_LOCATIONS - {"local_disk0"}
           or ("local_disk0" in d.kinds and not uses_fallback(nb, download_site(nb, d), fallbacks))]
    where = "; ".join(d.describe() for d in found)
    if staging:
        return False, f"files staged to /local_disk0 (lines {staging}); downloads: {where}"
    if bad:
        return False, f"destination not an accepted location: {'; '.join(d.describe() for d in bad)}"
    adaptive = f" (/local_disk0 only with a used fallback: {', '.join(sorted(fallbacks))})" if any(
        "local_disk0" in d.kinds for d in found) or local_disk0_staging(nb) else ""
    return True, f"downloads: {where}{adaptive}"


_WEIGHT_LOADS = ("from_pretrained", "load_file", "load_state_dict")


def _under(path: str, base: str) -> bool:
    base = base.rstrip("/")
    return bool(base.replace("{}", "").strip("/")) and (path == base or path.startswith(base + "/"))


def _combine(parts: list, sep: str = "") -> list:
    out = [""]
    for options in parts:
        out = [a + (sep if a and sep else "") + b for a in out for b in (options or ["{}"])][:20]
    return out


def _parameter_values(nb: PracticalCode, name: ast.Name) -> list:
    """Arguments passed for `name` when it is a parameter of a notebook function: one per call site
    (a method's self / cls is skipped when the call is `x.method(...)`)."""
    out = []
    for f in nb.functions.values():
        params = [a.arg for a in f.args.args]
        if name.id not in params or not (f.lineno <= name.lineno <= f.end_lineno):
            continue
        for call in nb.calls:
            if _call_name(call) != f.name:
                continue
            index = params.index(name.id) - (1 if params[:1] in (["self"], ["cls"]) and isinstance(call.func, ast.Attribute) else 0)
            if 0 <= index < len(call.args):
                out.append(call.args[index])
            out += [k.value for k in call.keywords if k.arg == name.id]
    return out


def path_strings(nb: PracticalCode, node, depth: int = 0) -> list:
    """Possible string values of a path expression: names followed through assignments and
    notebook-function parameters (to the call sites), f-strings, `+`, `/` (pathlib),
    os.path.join(...) and Path(...); an unresolved part becomes '{}'."""
    if depth > 6 or node is None:
        return []
    out = []
    for v in nb.values(node):
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            out.append(v.value)
        elif isinstance(v, ast.JoinedStr):
            out += _combine([[str(p.value)] if isinstance(p, ast.Constant) else path_strings(nb, p.value, depth + 1)
                             for p in v.values])
        elif isinstance(v, ast.BinOp) and isinstance(v.op, (ast.Add, ast.Div)):
            out += _combine([path_strings(nb, v.left, depth + 1), path_strings(nb, v.right, depth + 1)],
                            "/" if isinstance(v.op, ast.Div) else "")
        elif isinstance(v, ast.Call) and _call_name(v) in ("join", "Path", "PurePath") and v.args:
            out += _combine([path_strings(nb, a, depth + 1) for a in v.args], "/")
        elif isinstance(v, ast.Name):
            out += [t for arg in _parameter_values(nb, v) for t in path_strings(nb, arg, depth + 1)]
    return list(dict.fromkeys(out))[:50]


def _destination_names(nb: PracticalCode, download) -> set:
    """Notebook variables a direct download writes into: the names in its local_dir / cache_dir."""
    if download.node is None or download.how != "notebook":
        return set()
    return {n.id for m in nb.keyword_maps(download.node) for key in ("local_dir", "cache_dir")
            if m.get(key) is not None for n in ast.walk(m[key]) if isinstance(n, ast.Name)}


def _staged_paths(nb: PracticalCode, download) -> list:
    """The download's destination plus where shutil copies / moves it: copytree(local, DEST), or
    a per-file copy whose source derives from the destination variable (an os.walk over it); the
    copy target's trailing per-file parts ("/{}") are dropped, so the target folder counts."""
    paths, names = list(download.paths), _destination_names(nb, download)
    for call in sorted(nb.calls, key=lambda c: c.lineno):
        if _call_name(call) in _STAGING_CALLS and len(call.args) >= 2 and any(
                d.startswith("shutil.") for d in nb.dotted(call.func)):
            src = call.args[0]
            if (any(_under(t, b) for t in path_strings(nb, src) for b in paths)
                    or (download.node is not None and _derives(nb, src, lambda n: n is download.node))
                    or (names and _derives(nb, src, lambda n: isinstance(n, ast.Name) and n.id in names))):
                paths += [re.sub(r"(?:/\{\})+$", "", t) for t in path_strings(nb, call.args[1])]
    return paths


def reads_download(nb: PracticalCode, node, download) -> bool:
    """`node` (a path argument) reads from the download's destination (or a copy of it): one of its
    string values is that path or a path under it, or it is derived from the download call's
    return value. Line order alone never links a file use to a download."""
    if any(_under(t, base) for t in path_strings(nb, node) for base in _staged_paths(nb, download)):
        return True
    return download.node is not None and _derives(nb, node, lambda n: n is download.node)


def _load_paths(call: ast.Call, nb: PracticalCode) -> list:
    """Path arguments of a weight load: the first argument / pretrained_model_name_or_path /
    filename of from_pretrained, load_file and load_state_dict, or the values of a log_model
    artifacts= dict and code_paths= list."""
    name = _call_name(call)
    if name in _WEIGHT_LOADS:
        return list(call.args[:1]) + [k.value for k in call.keywords
                                      if k.arg in ("pretrained_model_name_or_path", "filename", "f", "path")]
    if name == "log_model":
        out = []
        for k in call.keywords:
            if k.arg == "artifacts":
                out += [v for m in nb.members(k.value) for v in m.values() if v is not None]
            elif k.arg == "code_paths":
                out += [e for v in nb.values(k.value) if isinstance(v, (ast.List, ast.Tuple)) for e in v.elts]
        return out
    return []


def _later_weight_load(nb: PracticalCode, download):
    """(line, how) of the first weight load after the download that reads from its destination
    (data flow / path, not line order alone), or None."""
    for call in sorted(nb.calls, key=lambda c: c.lineno):
        if call.lineno <= download.line:
            continue
        if any(reads_download(nb, arg, download) for arg in _load_paths(call, nb)):
            name = _call_name(call)
            return call.lineno, name if name in _WEIGHT_LOADS else "log_model(artifacts=...)"
    return None


def xet_download_ok(nb, response, record, task):
    found = downloads(nb, response)
    if not found:
        return False, "no Hugging Face download call found"
    if all(d.xet[0] for d in found):
        return True, "; ".join(f"line {d.line} ({d.how}): {XET_VAR} {d.xet[1]}" for d in found)
    unset = [d for d in found if not d.xet[0]]
    uses = [(d, _later_weight_load(nb, d)) for d in unset]
    if all(used for _, used in uses) and _endpoint_serves_registration(record):
        return True, "recorded evidence the download completed: " + "; ".join(
            f"line {used[0]} ({used[1]}) loads files from the destination of the download at line {d.line}"
            for d, used in uses) + ", and the run record shows the endpoint READY serving the registered version"
    settings = "; ".join(f"line {d.line} ({d.how}): {XET_VAR} {d.xet[1]}" for d in found)
    return False, (f"{settings}; no recorded evidence the download completed (load from its destination: "
                   f"{[used and used[0] for _, used in uses]}; endpoint READY + registered version in run "
                   f"record: {_endpoint_serves_registration(record)})")


def _pip_specs(response: str) -> list:
    """(line, spec) for every transformers requirement in a %pip install line."""
    out = []
    for number, line in enumerate(response.splitlines(), 1):
        match = re.search(r"%pip\s+install\b(.*)", re.sub(r"^\s*# MAGIC\s?", "", line))
        if not match:
            continue
        try:
            words = shlex.split(match.group(1))
        except ValueError:
            words = match.group(1).split()
        out += [(number, w) for w in words if re.match(r"transformers(?![\w-])", w, re.I)]
    return out


def _mlflow_log_model_calls(nb: PracticalCode) -> list:
    return [c for c in nb.calls if _call_name(c) == "log_model"
            and any(re.match(r"^mlflow\.\w+\.log_model$", d) for d in nb.dotted(c.func))]


def _serving_specs(nb: PracticalCode) -> list:
    """(line, spec) for every transformers requirement passed to an MLflow log_model call."""
    out = []
    for call in _mlflow_log_model_calls(nb):
        for kwargs in nb.keyword_maps(call):
            for key in ("pip_requirements", "extra_pip_requirements"):
                node = kwargs.get(key)
                for value in nb.values(node) if node is not None else []:
                    items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else []
                    out += [(call.lineno, t) for item in items for t in nb.strings(item)
                            if re.match(r"transformers(?![\w-])", t, re.I)]
    return out


def _excludes_5(spec: str):
    """True if a transformers requirement excludes every 5.x version, False if it allows one,
    "installed" for a pin to the version installed at run time (transformers=={...})."""
    rest = re.sub(r"^transformers(\[[^\]]*\])?", "", spec.strip(), flags=re.I).strip()
    if not rest:
        return False
    if "{}" in rest:
        return "installed"
    try:
        allowed = SpecifierSet(rest)
    except InvalidSpecifier:
        return False
    return not any(allowed.contains(v, prereleases=True) for v in ("5.0.0", "5.1.0", "5.99.0"))


def _references_upstream(nb: PracticalCode, packages: tuple) -> bool:
    """The upstream package is imported in the parsed code or in a source string (a wrapper file)."""
    return nb.imports_package(packages) or any(
        re.search(rf"\bfrom\s+{p}\.|\bimport\s+{p}\b", s.value) for p in packages for s in nb.string_constants())


def _served_classes(nb: PracticalCode) -> list:
    """The served model's code in the notebook: mlflow PythonModel subclasses, plus the notebook
    functions and classes they reference by name (followed recursively)."""
    classes = {c.name: c for c in nb.nodes(ast.ClassDef)}
    defs = {**nb.functions, **classes}
    todo = [c for c in classes.values() if any(d.endswith("PythonModel") for b in c.bases for d in nb.dotted(b))]
    seen = {}
    while todo:
        node = todo.pop()
        if node.name in seen:
            continue
        seen[node.name] = node
        todo += [defs[n.id] for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in defs and n.id not in seen]
    return list(seen.values())


def _served_uses(nb: PracticalCode, packages: tuple):
    """Whether the served model needs transformers or the upstream package: the served classes
    import one or use a name the notebook imports from one; a wrapper written as a source file
    imports one; a log_model code_paths entry ships the upstream package; or the model is logged
    with the mlflow.transformers flavor. None when the notebook defines no PythonModel class
    (then the notebook-wide rule applies)."""
    roots = packages + ("transformers",)
    served = _served_classes(nb)
    calls = _mlflow_log_model_calls(nb)
    if any(".transformers." in d for c in calls for d in nb.dotted(c.func)):
        return True
    wrapper_files = [t.value for t in nb.string_constants() if "PythonModel" in t.value]
    if any(re.search(rf"\bfrom\s+{p}\b|\bimport\s+{p}\b", t) for p in roots for t in wrapper_files):
        return True
    shipped = [t for c in calls for k in c.keywords if k.arg == "code_paths"
               for v in nb.values(k.value) for e in (v.elts if isinstance(v, (ast.List, ast.Tuple)) else [v])
               for t in path_strings(nb, e)]
    if any(re.search(rf"(?:^|/){p}(?:/|$)", t) for p in packages for t in shipped):
        return True
    if not served and not wrapper_files:
        return None
    for node in served:
        for n in ast.walk(node):
            if isinstance(n, ast.Import) and any(a.name.split(".")[0] in roots for a in n.names):
                return True
            if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in roots:
                return True
            if isinstance(n, ast.Name) and nb.imports.get(n.id, "").split(".")[0] in roots:
                return True
    return False


def transformers_pin(nb: PracticalCode, response: str, task: dict) -> tuple:
    """(ok, reason): transformers 5.x is excluded from the serving requirements (or %pip when
    there is no serving pin), or the served model does not use transformers at all — judged on the
    served code (PythonModel classes and what they use, wrapper files, code_paths), so an upstream
    import used only to compare against upstream does not count. Shared by
    model_loads_with_pinned_deps and the corrected download_staging/deps_pinned check."""
    packages = UPSTREAM_PACKAGES.get(task.get("model_family"), ())
    pip, serving = _pip_specs(response), _serving_specs(nb)
    served = _served_uses(nb, packages)
    if served is None:
        served = nb.imports_package(packages + ("transformers",)) or _references_upstream(nb, packages)
    if not (served or serving):
        return True, ("the served model does not use transformers (no transformers or upstream-package "
                      "use in the served code, none in the serving requirements)")
    specs = serving or pip
    source = "serving requirements" if serving else "%pip (no serving pin)"
    if not specs:
        return False, "transformers is used but never pinned"
    verdicts = [(line, spec, _excludes_5(spec)) for line, spec in specs]
    pip_excludes = any(_excludes_5(spec) is True for _, spec in pip)
    ok = all(v is True or (v == "installed" and pip_excludes) for _, _, v in verdicts)
    return ok, f"{source}: " + "; ".join(
        f"line {line} {spec!r} " + ("excludes 5.x" if v is True else
                                    "pins the installed version" + (", which %pip limits to <5"
                                                                    if pip_excludes else ", not limited by %pip")
                                    if v == "installed" else "allows 5.x")
        for line, spec, v in verdicts)


def model_loads_with_pinned_deps(nb, response, record, task):
    ready = _measured(record, "outcome/endpoint_ready")
    deps_ok, deps = transformers_pin(nb, response, task)
    ok = deps_ok and bool(ready)
    return ok, f"{deps}; endpoint READY in run record: {ready}"


def _defines_own_module(nb: PracticalCode) -> bool:
    """A torch nn.Module subclass defined in the notebook: a re-implementation of the model."""
    return any(any(d.endswith(("nn.Module", "torch.nn.Module")) or d == "Module"
                   for base in cls.bases for d in nb.dotted(base))
               for cls in nb.nodes(ast.ClassDef))


def fidelity_verified(nb, response, record, task):
    packages = UPSTREAM_PACKAGES.get(task.get("model_family"), ())
    own_module = _defines_own_module(nb)
    upstream_model = _references_upstream(nb, packages) and not own_module
    seen = []
    for line, sides in numeric_comparisons(nb):
        a, b = (origins(nb, side, packages, upstream_model) for side in sides)
        seen.append(f"line {line}: {sorted(a) or ['-']} vs {sorted(b) or ['-']}")
        for ref, cand in ((a, b), (b, a)):
            # the reference side must not itself be a candidate output
            if UPSTREAM in ref and not ref & {PREDICT, ENDPOINT} and cand & {PREDICT, ENDPOINT}:
                return True, (f"line {line} compares a candidate output ({', '.join(sorted(cand & {PREDICT, ENDPOINT}))}) "
                              f"with a reference derived from upstream code")
    why = ("the notebook defines its own nn.Module (re-implementation) and never compares it with "
           "upstream code" if own_module else "no comparison of a candidate output with upstream code")
    return False, why + (f"; comparisons found: {'; '.join(seen)}" if seen else "")


_SIGNATURE_BUILDERS = ("infer_signature", "ModelSignature")


def logged_signatures(nb: PracticalCode) -> list:
    """One entry per MLflow log_model call (and **kwargs option): {"line", "builders" (the
    signature builders used: infer_signature / ModelSignature), "has_signature" (a signature
    that is not None), "has_example" (input_example given), "text" (for reasons)}."""
    out = []
    for call in _mlflow_log_model_calls(nb):
        for kwargs in nb.keyword_maps(call):
            signature, example = kwargs.get("signature"), kwargs.get("input_example")
            values = [v for v in nb.values(signature) if not (isinstance(v, ast.Constant) and v.value is None)] \
                if signature is not None else []
            builders = sorted({_call_name(v) for v in values if isinstance(v, ast.Call)} & set(_SIGNATURE_BUILDERS))
            out.append({"line": call.lineno, "builders": builders, "has_signature": bool(values),
                        "has_example": example is not None,
                        "text": f"line {call.lineno}: signature {builders or ('other expression' if values else 'missing')}, "
                                f"input_example {'given' if example is not None else 'missing'}"})
    return out


def signature_and_example(nb, response, record, task):
    seen = logged_signatures(nb)
    for entry in seen:
        if not entry["has_signature"] or not entry["has_example"]:
            continue
        if entry["builders"]:
            return True, entry["text"]
        if _endpoint_serves_registration(record):
            return True, entry["text"] + "; run record: endpoint READY serving the registered version"
    return False, "; ".join(e["text"] for e in seen) or "no MLflow log_model call"


def registered_in_uc(nb, response, record, task):
    return scorers.uc_provenance(response, task)["uc_registry"]


def serves_valid_embeddings(nb, response, record, task):
    return _record_checks(record, ["outcome/returns_valid_embeddings"])


def serves_expression_input(nb, response, record, task):
    return _record_checks(record, ["outcome/accepts_expression_input"])


def endpoint_smoke_tested(nb, response, record, task):
    queries = sorted(c.lineno for c in nb.calls if _is_endpoint_query(nb, c))
    if not queries:
        return False, "the endpoint is never queried"
    packages = UPSTREAM_PACKAGES.get(task.get("model_family"), ())
    checks = [(n.lineno, "assert") for n in nb.nodes(ast.Assert)
              if ENDPOINT in origins(nb, n.test, packages, False)]
    checks += [(n.lineno, "comparison") for n in nb.nodes(ast.Compare)
               if ENDPOINT in origins(nb, n, packages, False)]
    checks += [(line, "numeric comparison") for line, sides in numeric_comparisons(nb)
               if any(ENDPOINT in origins(nb, s, packages, False) for s in sides)]
    if not checks:
        return False, f"endpoint queried (line {queries[0]}) but the result is never checked"
    line, how = min(checks)
    return True, f"endpoint queried (line {queries[0]}); result checked by an {how} at line {line}" \
        if how == "assert" else f"endpoint queried (line {queries[0]}); result checked by a {how} at line {line}"


def existing_resources_checked(nb, response, record, task):
    wanted = task.get("new_checks", {}).get("existing_resources", [])
    results = scorers.existing_resources(response, task)
    failed = [r for r in wanted if not results[r][0]]
    return not failed, "; ".join(f"{r}: {results[r][1]}" for r in wanted)


def governance_logging(nb, response, record, task):
    result = scorers.ai_gateway_logging(response, task)
    (inference, why_inf), (usage, why_use) = result["inference_table"], result["usage_tracking"]
    config = (record or {}).get("endpoint_config") or {}
    measured = (f"; run record: inference table {config.get('ai_gateway_inference_table')}, usage tracking "
                f"{config.get('ai_gateway_usage_tracking')}" if config else "")
    return inference and usage, f"inference table: {why_inf}; usage tracking: {why_use}{measured}"


def provenance_recorded(nb, response, record, task):
    found = downloads(nb, response)
    revisions = persisted_records(nb, _REVISION_KEY)

    def recorded(download) -> list:
        """Lines where the commit a download resolved is persisted: a value derived from it, or
        a 40-character SHA literal."""
        out = []
        for line, _, values, nodes in revisions:
            if any(_SHA.match(t) for t in values) or any(_derives(nb, v, download.resolved) for v in nodes):
                out.append(line)
        return out

    pinned = [(d, recorded(d)) for d in found if d.revision[0] == "sha" or d.revision[0] == "resolved"]
    pinned = [(d, lines) for d, lines in pinned if d.revision[0] == "sha" or lines]
    licences = [(line, sink) for line, sink, values, _ in persisted_records(nb, _LICENCE_KEY) if _filled(values)]
    if pinned:
        d, lines = pinned[0]
        revision = (f"revision pinned at line {d.line} ({d.revision[1]})" if d.revision[0] == "sha" else
                    f"revision pinned at line {d.line} ({d.revision[1]}; that commit recorded at line {lines[0]})")
    elif found:
        revision = "no download pinned to a commit: " + "; ".join(f"line {d.line} {d.revision[1]}" for d in found)
        if any(d.revision[0] == "resolved" for d in found):
            revision += " (the resolved commit itself is never recorded)"
    else:
        revision = "no Hugging Face download call found"
    where = (f"licence recorded: {', '.join(f'line {line} {sink}' for line, sink in licences)}" if licences
             else "no non-empty licence in a persisted record")
    return bool(pinned) and bool(licences), f"{revision}; {where}"


def similarity_search_works(nb, response, record, task):
    return _record_checks(record, ["outcome/index_ready", "outcome/nn_returns_cell_type"])


CHECK_FUNCTIONS = {
    "download_location_ok": download_location_ok,
    "xet_download_ok": xet_download_ok,
    "model_loads_with_pinned_deps": model_loads_with_pinned_deps,
    "fidelity_verified": fidelity_verified,
    "signature_and_example": signature_and_example,
    "registered_in_uc": registered_in_uc,
    "serves_valid_embeddings": serves_valid_embeddings,
    "serves_expression_input": serves_expression_input,
    "endpoint_smoke_tested": endpoint_smoke_tested,
    "existing_resources_checked": existing_resources_checked,
    "governance_logging": governance_logging,
    "provenance_recorded": provenance_recorded,
    "similarity_search_works": similarity_search_works,
}


def gpu_serving(record) -> tuple:
    """(True / False / None, reason) from the recorded endpoint settings only."""
    entities = ((record or {}).get("endpoint_config") or {}).get("served_entities")
    if not entities:
        return None, "no endpoint_config in the run record (not measured)"
    types = [str(e.get("workload_type")) for e in entities]
    ok = all(t.upper().startswith("GPU") for t in types)
    return ok, f"run record workload_type: {types}" + ("" if ok else f" ({GUIDELINES['gpu_serving']['context']})")


GUIDELINE_FUNCTIONS = {"gpu_serving": gpu_serving}


def practical_details(task: dict, response: str, record: dict | None) -> dict:
    """{"practical/<check>": (passed, reason)} for the checks that apply to the task."""
    nb = PracticalCode(response)
    out = {}
    for name in checks_for(task):
        passed, reason = CHECK_FUNCTIONS[name](nb, response, record, task)
        out[f"{PRACTICAL_PREFIX}{name}"] = (bool(passed), reason)
    return out


def practical_scores(task: dict, response: str, record: dict | None) -> dict:
    """{"practical/<check>": bool} — practical_details without the reasons."""
    return {k: ok for k, (ok, _) in practical_details(task, response, record).items()}


def guideline_details(record: dict | None) -> dict:
    """{"guideline/<check>": (True / False / None, reason)} — reported on its own, never counted."""
    return {f"{GUIDELINE_PREFIX}{name}": fn(record) for name, fn in GUIDELINE_FUNCTIONS.items()}


def failed_by_severity(scores: dict) -> dict:
    """{"critical"|"major"|"minor": [failed practical metric names]} (guideline metrics are skipped)."""
    out = {sev: [] for sev in SEVERITIES}
    for name, ok in scores.items():
        if name.startswith(PRACTICAL_PREFIX) and not ok:
            out[CHECKS[name[len(PRACTICAL_PREFIX):]]["severity"]].append(name)
    return out


# ---- CLI: score one round's exports --------------------------------------------------------

def round_samples(run_dir: Path) -> list:
    """[(label, task_id, record, notebook source)] for every record that is not discarded;
    an earlier attempt is labelled "<run> attempt <n>", as in score_open_weight_models.py."""
    from install_skill_for_run import export_file, load_records, run_samples

    records = load_records(run_dir / "run_records.json")
    out = []
    for run in sorted({r["run_id"] for r in records}):
        for task_id in sorted({r["task_id"] for r in records}):
            found = run_samples(records, run, task_id)
            for i, record in enumerate(found):
                label = run if i == len(found) - 1 else f"{run} attempt {record.get('attempt') or record['timestamp']}"
                out.append((label, task_id, record, (run_dir / export_file(record)).read_text(encoding="utf-8")))
    return out


def main(argv=None) -> int:
    import family_settings

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default=family_settings.DEFAULT_FAMILY)
    parser.add_argument("--round", help="runs_<date> folder (default: family.json current_round)")
    args = parser.parse_args(argv)
    run_dir = family_settings.round_dir(args.family, args.round, EVAL_DIR)
    for label, task_id, record, source in round_samples(run_dir):
        details = practical_details(scorers.TASKS[task_id], source, record)
        failed = failed_by_severity({k: ok for k, (ok, _) in details.items()})
        passed = sum(ok for ok, _ in details.values())
        print(f"\n## {label} / {task_id}: {passed} / {len(details)} passed; failed critical "
              f"{len(failed['critical'])}, major {len(failed['major'])}, minor {len(failed['minor'])}")
        for name, (ok, reason) in details.items():
            print(f"- {name}: {'pass' if ok else 'FAIL'} — {reason}")
        for name, (ok, reason) in guideline_details(record).items():
            print(f"- {name} (guideline, not counted): {'-' if ok is None else 'yes' if ok else 'no'} — {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
