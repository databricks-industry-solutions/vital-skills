#!/usr/bin/env python3
"""Practical checks for the open-weight-models TEDDY eval — TRIAL, not adopted.

Not wired into score_open_weight_models.py and not part of `scorers.scorers`. Each check
scores whether a GOAL of the task was reached, accepting any valid way of reaching it,
instead of matching strings from teddy.md. Code goals are read by parsing the notebook's
Python cells with `ast` (scorers.NotebookCode); behaviour goals read the outcome measured
before teardown and kept in the run record (run_records.json "outcome" / "endpoint_config").
Metric names start with "practical/". Every check's goal, severity and accepted
alternatives are in CHECKS; models/teddy/runs_20261026/PRACTICAL_TRIAL.md compares these
checks with the current ones on the 2026-10-27 runs.

  python3 skills/open-weight-models/eval/practical_checks.py [--round runs_20261026]

prints each run's practical results (reason per check, with notebook line numbers).
Needs mlflow-skinny (scorers.py imports it); no workspace, no network.
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

import scorers  # noqa: E402 — same folder; reused: NotebookCode, AI Gateway and UC registry logic
from scorers import NotebookCode, _call_name, _strip_comments  # noqa: E402

PRACTICAL_PREFIX = "practical/"
SEVERITIES = scorers.SEVERITIES

# Python package of each family's upstream code: a call into it is an upstream reference.
UPSTREAM_PACKAGES = {"teddy": ("teddy",)}

CHECKS = {
    "download_location_ok": {
        "severity": "critical",
        "goal": "Model files are downloaded to a location that exists on serverless compute.",
        "accepts": ["a /tmp path", "tempfile.mkdtemp() / TemporaryDirectory() (both under /tmp)",
                    "a Unity Catalog Volume path (/Volumes/...)",
                    "the same, as the destination of a download run in a subprocess or shell command"],
        "fails": "any /local_disk0 path in the code; a download whose destination is none of the above "
                 "(including the default Hugging Face cache, which is not stated in the code)",
        "evidence": "code",
    },
    "xet_download_ok": {
        "severity": "critical",
        "goal": "The Hugging Face download of the TEDDY repository (Xet storage) completes.",
        "accepts": ["HF_HUB_DISABLE_XET set anywhere in the code (env var, env dict, subprocess script)",
                    "the download run in a subprocess / clean environment (including a %sh or ! "
                    "`hf download` / `huggingface-cli download` command)",
                    "evidence the download completed: a later cell loads files it wrote "
                    "(from_pretrained, safetensors load_file, load_state_dict, log_model artifacts) "
                    "and the endpoint built from them reached READY (run record)"],
        "fails": "no download call, or none of the above",
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
        "goal": "The model's (or endpoint's) embeddings were compared against an upstream reference.",
        "accepts": ["a cell that compares outputs (cosine, allclose, max absolute difference) and "
                    "computes the reference with the upstream package's own code (e.g. TEDDY's "
                    "tokenizer or tokenization functions)",
                    "a cell that compares the wrapper with a direct forward pass of the upstream model "
                    "(`<x>.model(...)`), in a notebook that loads the upstream package and defines no "
                    "torch nn.Module of its own"],
        "fails": "no such comparison; a re-implementation (own nn.Module) compared only with itself "
                 "(a riskier valid alternative: scientific-risk check); endpoint-vs-notebook checks of "
                 "the same wrapper",
        "evidence": "code",
    },
    "signature_and_example": {
        "severity": "critical",
        "goal": "The logged model carries a signature and an input example (Unity Catalog needs the "
                "signature; the example grounds it in a real request).",
        "accepts": ["mlflow.<flavor>.log_model(signature=..., input_example=...) with the signature from "
                    "infer_signature(...), an explicit ModelSignature(...), or any other value"],
        "fails": "no log_model call that passes both",
        "evidence": "code",
    },
    "registered_in_uc": {
        "severity": "major",
        "goal": "The model is registered in Unity Catalog.",
        "accepts": ["scorers.uc_provenance()['uc_registry'] (the new check's logic, unchanged)"],
        "fails": "as that check",
        "evidence": "code",
    },
    "serves_real_cell_data": {
        "severity": "critical",
        "goal": "The endpoint embeds a cell from its expression (not pre-tokenized IDs) and returns "
                "valid embeddings.",
        "accepts": ["run record outcome/accepts_expression_input AND outcome/returns_valid_embeddings, "
                    "each in the endpoint's own declared input format"],
        "fails": "either not passed or not measured",
        "evidence": "record",
    },
    "gpu_serving": {
        "severity": "major",
        "goal": "The model is served on a GPU workload (task guidelines: GPU_SMALL).",
        "accepts": ["run record endpoint_config: every served entity has a GPU_* workload type",
                    "if the record has no endpoint_config: ServingModelWorkloadType.GPU_* in the code"],
        "fails": "a CPU workload (baseline/teddy-002 measured about 15 s per cell on CPU)",
        "evidence": "record (code fallback)",
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
        "goal": "The upstream revision is pinned and the licence is recorded durably.",
        "accepts": ["revision: the Hugging Face download uses an explicit revision (a commit SHA "
                    "constant, or the SHA resolved from model_info), not 'main'",
                    "licence: a licence key or value in ANY durable record — MLflow tags or params, "
                    "logged-model tags/metadata, registered-model or model-version tags, the "
                    "registered-model description, or a provenance manifest (json.dump / log_dict)"],
        "fails": "no pinned revision, or no licence in any durable record",
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


class PracticalCode(NotebookCode):
    """NotebookCode whose node line numbers are notebook file lines, plus `with ... as x` targets."""

    def __init__(self, response: str):
        super().__init__(_line_preserving(response))
        self.with_values = {}
        for tree in self.trees:
            for node in ast.walk(tree):
                if isinstance(node, (ast.With, ast.AsyncWith)):
                    for item in node.items:
                        if isinstance(item.optional_vars, ast.Name):
                            self.with_values.setdefault(item.optional_vars.id, []).append(item.context_expr)

    def nodes(self, kind):
        return [node for tree in self.trees for node in ast.walk(tree) if isinstance(node, kind)]

    def string_constants(self) -> list:
        return [n for n in self.nodes(ast.Constant) if isinstance(n.value, str)]

    def imports_package(self, packages: tuple) -> bool:
        return any(path.split(".")[0] in packages for path in self.imports.values())


# ---- Hugging Face downloads -------------------------------------------------------------

_HF_DOWNLOADS = ("snapshot_download", "hf_hub_download")
_SUBPROCESS = re.compile(r"^(?:subprocess\.(?:run|call|check_call|check_output|Popen)|os\.system)$")
_CLI_DOWNLOAD = re.compile(r"\b(?:huggingface-cli|hf)\s+download\b(?P<rest>.*)")
ACCEPTED_LOCATIONS = {"tmp", "volume", "tempfile"}


class Download:
    """One Hugging Face download: where in the notebook, how it ran, destination kinds, revision."""

    def __init__(self, line: int, how: str, kinds: set, revision):
        self.line, self.how, self.kinds, self.revision = line, how, kinds, revision

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


def _revision(nb: PracticalCode, kwargs: dict):
    """The revision= of a download as text: a string constant, 'resolved at run time' for any
    other expression, or None when not given (or None / 'main')."""
    node = kwargs.get("revision")
    if node is None:
        return None
    texts = nb.strings(node)
    if texts and all(t in ("main", "") for t in texts):
        return None
    if any(isinstance(v, ast.Constant) and v.value is None for v in nb.values(node)):
        return None
    return texts[0] if texts and "{}" not in texts[0] else "resolved at run time"


def _direct_downloads(nb: PracticalCode, how: str = "notebook", line: int | None = None) -> list:
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
            out.append(Download(line or call.lineno, how, kinds, _revision(nb, kwargs)))
    return out


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
            for download in _direct_downloads(PracticalCode(script), "subprocess", call.lineno):
                if download.kinds <= {"unknown"} and path_args:
                    download.kinds = path_args
                out.append(download)
        for command in texts:
            match = _CLI_DOWNLOAD.search(command)
            if match:
                out.append(_cli_download(match.group("rest"), call.lineno))
    return out


def _cli_download(rest: str, line: int) -> Download:
    try:
        words = shlex.split(rest)
    except ValueError:
        words = rest.split()
    dest = next((words[i + 1] for i, w in enumerate(words[:-1]) if w in ("--local-dir", "--cache-dir")), None)
    revision = next((words[i + 1] for i, w in enumerate(words[:-1]) if w == "--revision"), None)
    kinds = {_classify(dest)} if dest else {"hf_default_cache"}
    return Download(line, "subprocess", kinds, None if revision in (None, "main") else revision)


def _shell_downloads(response: str) -> list:
    """`%sh` / `!` cells: `huggingface-cli download` or `hf download` (run in their own process)."""
    out, in_sh = [], False
    for number, line in enumerate(response.splitlines(), 1):
        if _CELL_SEPARATOR.match(line):
            in_sh = False
            continue
        magic = re.match(r"^\s*# MAGIC\s?(.*)", line)
        text = magic.group(1) if magic else line
        if magic and text.lstrip().startswith("%"):
            in_sh = text.lstrip().startswith("%sh")     # a cell's magic decides its language
        if (in_sh and magic) or text.lstrip().startswith("!"):
            match = _CLI_DOWNLOAD.search(text)
            if match:
                out.append(_cli_download(match.group("rest"), number))
    return out


def downloads(nb: PracticalCode, response: str) -> list:
    return sorted(_direct_downloads(nb) + _subprocess_downloads(nb) + _shell_downloads(response),
                  key=lambda d: d.line)


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


# ---- The checks --------------------------------------------------------------------------
# Each takes (nb, response, record, task) and returns (passed, reason).

def download_location_ok(nb, response, record, task):
    disk0 = sorted({n.lineno for n in nb.string_constants() if n.value.startswith("/local_disk0")})
    found = downloads(nb, response)
    if not found:
        return False, "no Hugging Face download call found"
    bad = [d for d in found if not d.kinds <= ACCEPTED_LOCATIONS]
    where = "; ".join(d.describe() for d in found)
    if disk0:
        return False, f"/local_disk0 path in the code (lines {disk0}); downloads: {where}"
    if bad:
        return False, f"destination not an accepted location: {'; '.join(d.describe() for d in bad)}"
    return True, f"downloads: {where}"


_WEIGHT_LOADS = ("from_pretrained", "load_file", "load_state_dict")


def _later_weight_load(nb: PracticalCode, after: int):
    for call in sorted(nb.calls, key=lambda c: c.lineno):
        if call.lineno <= after:
            continue
        name = _call_name(call)
        if name in _WEIGHT_LOADS:
            return call.lineno, name
        if name == "log_model" and any(k.arg in ("artifacts", "code_paths") for k in call.keywords):
            return call.lineno, "log_model(artifacts=...)"
    return None


def xet_download_ok(nb, response, record, task):
    found = downloads(nb, response)
    if not found:
        return False, "no Hugging Face download call found"
    code = _strip_comments(response)
    if "HF_HUB_DISABLE_XET" in code:
        line = next(i for i, text in enumerate(response.splitlines(), 1)
                    if "HF_HUB_DISABLE_XET" in text and not text.lstrip().startswith("#"))
        return True, f"HF_HUB_DISABLE_XET set (line {line})"
    isolated = [d for d in found if d.how == "subprocess"]
    if isolated:
        return True, f"download run in a subprocess (line {isolated[0].line})"
    first = found[0].line
    used = _later_weight_load(nb, first)
    ready = _measured(record, "outcome/endpoint_ready")
    if used and ready:
        return True, (f"download completed: line {used[0]} ({used[1]}) uses files from the download at line "
                      f"{first}, and the endpoint built from them reached READY (run record)")
    return False, ("no Xet workaround and no evidence the download completed "
                   f"(later file use: {used and used[0]}; endpoint READY in run record: {ready})")


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


def _serving_specs(nb: PracticalCode) -> list:
    """(line, spec) for every transformers requirement passed to an MLflow log_model call."""
    out = []
    for call in nb.calls:
        if _call_name(call) != "log_model" or not any(re.match(r"^mlflow\.\w+\.log_model$", d)
                                                      for d in nb.dotted(call.func)):
            continue
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


def model_loads_with_pinned_deps(nb, response, record, task):
    ready = _measured(record, "outcome/endpoint_ready")
    packages = UPSTREAM_PACKAGES.get(task.get("model_family"), ())
    pip, serving = _pip_specs(response), _serving_specs(nb)
    uses_transformers = (nb.imports_package(packages + ("transformers",)) or bool(serving)
                         or any(re.search(rf"\bfrom\s+{p}\.|\bimport\s+{p}\b", s.value)
                                for p in packages for s in nb.string_constants()))
    if not uses_transformers:
        deps_ok, deps = True, ("model path does not use transformers (no transformers or upstream-package "
                               "import, none in the serving requirements)")
    else:
        specs = serving or pip
        source = "serving requirements" if serving else "%pip (no serving pin)"
        if not specs:
            deps_ok, deps = False, "transformers is used but never pinned"
        else:
            verdicts = [(line, spec, _excludes_5(spec)) for line, spec in specs]
            pip_excludes = any(_excludes_5(spec) is True for _, spec in pip)
            deps_ok = all(v is True or (v == "installed" and pip_excludes) for _, _, v in verdicts)
            deps = f"{source}: " + "; ".join(
                f"line {line} {spec!r} " + ("excludes 5.x" if v is True else
                                            "pins the installed version" + (", which %pip limits to <5"
                                                                            if pip_excludes else ", not limited by %pip")
                                            if v == "installed" else "allows 5.x")
                for line, spec, v in verdicts)
    ok = deps_ok and bool(ready)
    return ok, f"{deps}; endpoint READY in run record: {ready}"


_COMPARE_CALLS = ("allclose", "assert_allclose", "isclose", "assert_close", "cosine_similarity")


def _contains(node, test) -> bool:
    return any(test(n) for n in ast.walk(node))


def _comparison_lines(tree) -> list:
    """Lines of output comparisons in one cell: allclose-style calls, a cosine
    (product / norm), or an absolute difference abs(a - b)."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in _COMPARE_CALLS:
            lines.append(node.lineno)
        elif (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
              and _contains(node.right, lambda n: isinstance(n, ast.Call) and _call_name(n) == "norm")
              and _contains(node.left, lambda n: (isinstance(n, ast.BinOp) and isinstance(n.op, (ast.MatMult, ast.Mult)))
                            or (isinstance(n, ast.Call) and _call_name(n) == "dot"))):
            lines.append(node.lineno)
        elif (isinstance(node, ast.Call) and _call_name(node) == "abs" and node.args
              and isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].op, ast.Sub)):
            lines.append(node.lineno)
    return sorted(set(lines))


def _defines_own_module(nb: PracticalCode) -> bool:
    """A torch nn.Module subclass defined in the notebook: a re-implementation of the model."""
    return any(any(d.endswith(("nn.Module", "torch.nn.Module")) or d == "Module"
                   for base in cls.bases for d in nb.dotted(base))
               for cls in nb.nodes(ast.ClassDef))


def _upstream_reference_lines(nb: PracticalCode, tree, packages: tuple, upstream_model: bool) -> list:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = nb.dotted(node.func)
        if any(d.split(".")[0] in packages and "." in d for d in called):
            out.append((node.lineno, "upstream code"))
        elif upstream_model and any(d.endswith(".model") for d in called):
            out.append((node.lineno, "direct upstream-model forward"))
    return out


def fidelity_verified(nb, response, record, task):
    packages = UPSTREAM_PACKAGES.get(task.get("model_family"), ())
    loads_upstream = nb.imports_package(packages) or any(
        re.search(rf"\bfrom\s+{p}\.|\bimport\s+{p}\b", s.value) for p in packages for s in nb.string_constants())
    own_module = _defines_own_module(nb)
    upstream_model = loads_upstream and not own_module
    compared = []
    for tree in nb.trees:
        lines = _comparison_lines(tree)
        refs = _upstream_reference_lines(nb, tree, packages, upstream_model) if lines else []
        if lines and refs:
            return True, (f"line {lines[0]} compares outputs with a reference from {refs[0][1]} "
                          f"(line {refs[0][0]})")
        compared += lines
    why = ("the notebook defines its own nn.Module (re-implementation) and never compares it with "
           "upstream code" if own_module else "no comparison against upstream code")
    return False, why + (f" (comparisons without an upstream reference at lines {compared})" if compared else "")


def signature_and_example(nb, response, record, task):
    seen = []
    for call in nb.calls:
        if _call_name(call) != "log_model" or not any(re.match(r"^mlflow\.\w+\.log_model$", d)
                                                      for d in nb.dotted(call.func)):
            continue
        for kwargs in nb.keyword_maps(call):
            signature, example = kwargs.get("signature"), kwargs.get("input_example")
            values = [v for v in nb.values(signature) if not (isinstance(v, ast.Constant) and v.value is None)] \
                if signature is not None else []
            built = sorted({_call_name(v) if isinstance(v, ast.Call) else type(v).__name__ for v in values})
            seen.append(f"line {call.lineno}: signature {built or 'missing'}, input_example "
                        f"{'given' if example is not None else 'missing'}")
            if values and example is not None:
                return True, seen[-1]
    return False, "; ".join(seen) or "no MLflow log_model call"


def registered_in_uc(nb, response, record, task):
    return scorers.uc_provenance(response, task)["uc_registry"]


def serves_real_cell_data(nb, response, record, task):
    return _record_checks(record, ["outcome/accepts_expression_input", "outcome/returns_valid_embeddings"])


def gpu_serving(nb, response, record, task):
    entities = ((record or {}).get("endpoint_config") or {}).get("served_entities")
    if entities:
        types = [str(e.get("workload_type")) for e in entities]
        ok = all(t.upper().startswith("GPU") for t in types)
        note = "" if ok else " (on CPU, baseline/teddy-002's endpoint took about 15 s per cell)"
        return ok, f"run record endpoint_config workload_type: {types}{note}"
    gpu = [n.lineno for n in nb.nodes(ast.Attribute) if n.attr.startswith("GPU_")
           and any(d.endswith("ServingModelWorkloadType") for d in nb.dotted(n.value))]
    return bool(gpu), (f"no endpoint_config in run record; code: ServingModelWorkloadType.GPU_* at lines {gpu}"
                       if gpu else "no endpoint_config in run record and no GPU workload type in the code")


def governance_logging(nb, response, record, task):
    result = scorers.ai_gateway_logging(response, task)
    (inference, why_inf), (usage, why_use) = result["inference_table"], result["usage_tracking"]
    config = (record or {}).get("endpoint_config") or {}
    measured = (f"; run record: inference table {config.get('ai_gateway_inference_table')}, usage tracking "
                f"{config.get('ai_gateway_usage_tracking')}" if config else "")
    return inference and usage, f"inference table: {why_inf}; usage tracking: {why_use}{measured}"


_LICENCE_KEY = re.compile(r"licen[cs]e", re.I)
_LICENCE_TEXT = re.compile(r"licen[cs]e|apache-2\.0|\bmit\b|cc-by|bsd-\d", re.I)
# Durable sinks: call name -> which arguments hold the record (None = all arguments).
_KEYED_SINKS = {"set_tag": 0, "log_param": 0, "set_registered_model_tag": 1, "set_model_version_tag": 2}
_DICT_SINKS = {"set_tags": None, "log_params": None, "set_logged_model_tags": None, "log_dict": None,
               "dump": None, "log_model": ("tags", "metadata"), "start_run": ("tags",)}
_DESCRIPTION_SINKS = ("update_registered_model", "create_registered_model", "update_model_version")


def _keys(nb: PracticalCode, node, depth: int = 0) -> list:
    """Keys of a dict / constructor value, nested dicts included."""
    out = []
    for fields in nb.members(node):
        for key, value in fields.items():
            out.append(key)
            if depth < 4 and value is not None:
                out += _keys(nb, value, depth + 1)
    return out


def licence_records(nb: PracticalCode) -> list:
    """(line, where) for every durable record of the licence."""
    out = []
    for call in nb.calls:
        name = _call_name(call)
        if name in _KEYED_SINKS:
            position = _KEYED_SINKS[name]
            key = call.args[position] if len(call.args) > position else next(
                (k.value for k in call.keywords if k.arg in ("key",)), None)
            if key is not None and any(_LICENCE_KEY.search(s) for s in nb.strings(key)):
                out.append((call.lineno, name))
        elif name in _DICT_SINKS:
            wanted = _DICT_SINKS[name]
            nodes = list(call.args) + [k.value for k in call.keywords] if wanted is None else [
                k.value for k in call.keywords if k.arg in wanted]
            if any(_LICENCE_KEY.search(k) for n in nodes for k in _keys(nb, n)):
                out.append((call.lineno, "json.dump manifest" if name == "dump" else name))
        elif name in _DESCRIPTION_SINKS:
            texts = [t for k in call.keywords if k.arg == "description" for t in nb.strings(k.value)]
            if any(_LICENCE_TEXT.search(t) for t in texts):
                out.append((call.lineno, f"{name}(description=...)"))
    # for k, v in {...}.items(): <tag sink>(..., k, v)
    for loop in nb.nodes(ast.For):
        it = loop.iter
        if not (isinstance(it, ast.Call) and _call_name(it) == "items" and isinstance(it.func, ast.Attribute)):
            continue
        body_sinks = [c for stmt in loop.body for c in ast.walk(stmt)
                      if isinstance(c, ast.Call) and _call_name(c) in _KEYED_SINKS]
        if body_sinks and any(_LICENCE_KEY.search(k) for k in _keys(nb, it.func.value)):
            out.append((body_sinks[0].lineno, _call_name(body_sinks[0])))
    return sorted(set(out))


def provenance_recorded(nb, response, record, task):
    pinned = [d for d in downloads(nb, response) if d.revision]
    licence = licence_records(nb)
    revision = (f"revision pinned at line {pinned[0].line} ({pinned[0].revision})" if pinned
                else "no download pinned to a revision")
    where = (f"licence recorded: {', '.join(f'line {line} {w}' for line, w in licence)}" if licence
             else "licence not in any durable record")
    return bool(pinned) and bool(licence), f"{revision}; {where}"


def similarity_search_works(nb, response, record, task):
    return _record_checks(record, ["outcome/index_ready", "outcome/nn_returns_cell_type"])


CHECK_FUNCTIONS = {
    "download_location_ok": download_location_ok,
    "xet_download_ok": xet_download_ok,
    "model_loads_with_pinned_deps": model_loads_with_pinned_deps,
    "fidelity_verified": fidelity_verified,
    "signature_and_example": signature_and_example,
    "registered_in_uc": registered_in_uc,
    "serves_real_cell_data": serves_real_cell_data,
    "gpu_serving": gpu_serving,
    "governance_logging": governance_logging,
    "provenance_recorded": provenance_recorded,
    "similarity_search_works": similarity_search_works,
}


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


def failed_by_severity(scores: dict) -> dict:
    """{"critical"|"major"|"minor": [failed practical metric names]}."""
    out = {sev: [] for sev in SEVERITIES}
    for name, ok in scores.items():
        if not ok:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
