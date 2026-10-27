#!/usr/bin/env python3
"""Corrected replicated checks for the open-weight-models TEDDY eval — the scorer of record.

The replicated checks in scorers.py are the Sep–Oct 2026 logic copied unchanged. Several of
them are flawed: they match literal text from teddy.md, so a notebook that reaches the same
goal another valid way fails (a false negative), and a mention in a comment or an unrelated
string can pass (a false positive). This module judges the SAME INTENT of each flawed
sub-check properly and is the result reported from now on (headline counts and severity).
The unchanged ("verbatim") logic is kept only as a historical record: it shows how the
Sep–Oct 2026 numbers were produced (tests/test_eval_scorers.py reproduces them), and where it
disagrees with the corrected result the report shows it as a ⚑ flag with the reason.

  corrected sub-check  = CORRECTIONS[key]["check"] for a flawed key, else the verbatim result
  metric name          = "corrected/<phase>/<check>" (scorers.mlflow_metric_name of the key)

Every correction lists the flaw of the verbatim check, what now passes, and its severity
(the replicated key's severity in expectations.json, unchanged; see "severity_note").
Code is read with practical_checks.PracticalCode (Python cells parsed with `ast`, values
followed across cells); two corrections may also use the run record's measured outcome
(endpoint READY serving the registered version). The Sep–Oct 2026 runs have no run records,
so those alternatives can only be credited for runs from 2026-10-27 on.

  python3 skills/open-weight-models/eval/corrected_checks.py [--round runs_20261026] [--write]
  python3 skills/open-weight-models/eval/corrected_checks.py --sep-oct [--write]

prints every sample's corrected results and ⚑ flags; --write writes the corrected score
files (scores/<run>_corrected.json for a round; runs_20260924/scores_corrected/ for Sep–Oct,
never touching the recorded Sep–Oct files). Needs mlflow-skinny and packaging; no workspace,
no network.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import family_settings  # noqa: E402 — same folder
import practical_checks as pc  # noqa: E402 — reused, reviewed goal logic (download, Xet, pins, signature)
import scorers  # noqa: E402 — the verbatim replicated checks and the new checks' AI Gateway logic
from mlflow.entities import Feedback  # noqa: E402
from mlflow.genai.scorers import scorer as mlflow_scorer  # noqa: E402
from scorers import _call_name  # noqa: E402

CORRECTED_PREFIX = scorers.CORRECTED_PREFIX
SEP_OCT_DIR = EVAL_DIR / "runs_20260924"
SEP_OCT_TASKS = {"oss-001": "teddy-001", "oss-002": "teddy-002"}
SEP_OCT_ARMS = {"baseline": "baseline_scores.json", "with_skill": "with_skill_scores.json"}


def _is_teddy(task: dict) -> bool:
    return task.get("model_family") == "teddy"


# ---- corrected checks: each takes (nb, response, record, task) -> (passed, reason) -----------

def uses_tmp(nb, response, record, task):
    return pc.download_location_ok(nb, response, record, task)


def hf_xet_disabled(nb, response, record, task):
    if not _is_teddy(task):
        return True, "not a TEDDY task (the Xet workaround is TEDDY-specific)"
    return pc.xet_download_ok(nb, response, record, task)


def deps_pinned(nb, response, record, task):
    if not _is_teddy(task):
        return True, "not a TEDDY task (the transformers pin is TEDDY-specific)"
    return pc.transformers_pin(nb, response, task)


def transformers_open_range(nb, response, record, task):
    """forbidden_patterns/transformers>= : no transformers requirement that allows 5.x."""
    pip, serving = pc._pip_specs(response), pc._serving_specs(nb)
    if not pip and not serving:
        return True, "no transformers requirement"
    pip_excludes = any(pc._excludes_5(spec) is True for _, spec in pip)
    bad, seen = [], []
    for where, specs in (("%pip", pip), ("serving requirements", serving)):
        for line, spec in specs:
            verdict = pc._excludes_5(spec)
            ok = verdict is True or (verdict == "installed" and pip_excludes)
            seen.append(f"line {line} {where} {spec!r}")
            if not ok:
                bad.append(f"line {line} {where} {spec!r}" + (
                    " (the version installed at run time; %pip does not exclude 5.x)" if verdict == "installed" else ""))
    if bad:
        return False, "allows transformers 5.x: " + "; ".join(bad)
    return True, "every transformers requirement excludes 5.x: " + "; ".join(seen)


def infer_signature(nb, response, record, task):
    """model_registration/infer_signature : the logged model has a signature."""
    seen = pc.logged_signatures(nb)
    for entry in seen:
        if entry["builders"]:
            return True, entry["text"]
    for entry in seen:
        if entry["has_signature"] and pc._endpoint_serves_registration(record):
            return True, entry["text"] + "; run record: endpoint READY serving the registered version"
    return False, "; ".join(e["text"] for e in seen) or "no MLflow log_model call"


_ENSEMBL = re.compile(r"\bENSG\d{11}\b")
_REAL_DATA_FILES = ("vocab.txt", ".h5ad")


def _real_gene_source(nb, node, packages: tuple):
    """What makes a value real gene data, or None: the model vocabulary (vocab.txt), a real
    single-cell file (.h5ad, read_h5ad), the upstream tokenizer (a call into the upstream
    package whose name mentions token), or an Ensembl gene ID literal (not ENSG00000FAKE)."""
    for n in pc.flow_nodes(nb, node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            text = n.value
            if any(f in text for f in _REAL_DATA_FILES):
                return f"line {n.lineno}: reads {text!r}"
            if _ENSEMBL.search(text) and "FAKE" not in text:
                return f"line {n.lineno}: Ensembl gene ID {_ENSEMBL.search(text).group(0)}"
        elif isinstance(n, ast.Call):
            for d in nb.dotted(n.func):
                if d.endswith("read_h5ad"):
                    return f"line {n.lineno}: {d}"
                if d.split(".")[0] in packages and "token" in d.lower():
                    return f"line {n.lineno}: upstream tokenizer {d}"
    return None


_RANDOM = re.compile(r"(?:^|\.)(?:randint|rand|randn|random_integers|choice|integers)\(\)")


def _random_source(nb, node) -> bool:
    return any(isinstance(n, ast.Call) and any(_RANDOM.search(d) for d in nb.dotted(n.func))
               for n in pc.flow_nodes(nb, node))


def real_gene_ids(nb, response, record, task):
    """pyfunc_quality/real_gene_ids : the input example (or the payload sent to the endpoint)
    is built from real gene data, not random or made-up token IDs."""
    if not _is_teddy(task):
        return True, "not a TEDDY task (vocab.txt is TEDDY's tokenizer)"
    packages = pc.UPSTREAM_PACKAGES.get(task.get("model_family"), ())
    examples = [(c.lineno, kwargs["input_example"]) for c in pc._mlflow_log_model_calls(nb)
                for kwargs in nb.keyword_maps(c) if kwargs.get("input_example") is not None]
    payloads = [(c.lineno, a) for c in nb.calls if pc._is_endpoint_query(nb, c)
                for a in list(c.args) + [k.value for k in c.keywords]]
    for kind, items in (("input example", examples), ("endpoint payload", payloads)):
        for line, node in items:
            source = _real_gene_source(nb, node, packages)
            if source:
                return True, f"{kind} at line {line} built from real gene data ({source})"
    # The upstream tokenizer run on real data, its output read back from files (no data flow
    # to follow), with no random generator behind the example.
    tokenizer = [c.lineno for c in nb.calls if any(d.split(".")[0] in packages and "token" in d.lower()
                                                   for d in nb.dotted(c.func))]
    random = [line for line, node in examples + payloads if _random_source(nb, node)]
    if tokenizer and examples and not random:
        return True, (f"the notebook runs the upstream tokenizer (line {tokenizer[0]}) and its input example "
                      f"has no random source")
    return False, (f"no input example or endpoint payload built from real gene data "
                   f"({len(examples)} input example(s), {len(payloads)} endpoint payload argument(s) checked)")


_DEPENDENCY_FILES = ("pyproject.toml", "requirements.txt", "setup.cfg", "setup.py")
_FILE_READS = ("open", "read_text", "read_bytes")


def _reads_dependency_file(nb) -> str | None:
    for call in nb.calls:
        name = _call_name(call)
        holder = call.func.value if isinstance(call.func, ast.Attribute) else None
        nodes = list(call.args[:1]) if name == "open" else (
            [a for v in nb.values(holder) if isinstance(v, ast.Call) for a in v.args[:1]]
            if name in _FILE_READS and holder is not None else [])
        for node in nodes:
            for text in nb.strings(node):
                if text.rsplit("/", 1)[-1] in _DEPENDENCY_FILES:
                    return f"line {call.lineno} reads {text!r}"
    return None


def serving_requirements(nb) -> list:
    """(line, requirement text) for every requirement passed to an MLflow log_model call."""
    out = []
    for call in pc._mlflow_log_model_calls(nb):
        for kwargs in nb.keyword_maps(call):
            for key in ("pip_requirements", "extra_pip_requirements"):
                node = kwargs.get(key)
                for value in nb.values(node) if node is not None else []:
                    items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else []
                    out += [(call.lineno, t) for item in items for t in nb.strings(item)]
    return out


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pip_reqs_from_source(nb, response, record, task):
    """model_registration/pip_reqs_from_source : the serving requirements come from the model's
    own dependency file — the notebook reads that file, or the serving pins agree with it."""
    read = _reads_dependency_file(nb)
    if read:
        return True, f"the notebook reads the model's dependency file ({read})"
    upstream = (family_settings.load_family(task["model_family"], EVAL_DIR).get("upstream_dependencies")
                or {}) if task.get("model_family") else {}
    pins = {_normalise(k): v for k, v in (upstream.get("pins") or {}).items()}
    if not pins:
        return False, "no dependency file read, and no upstream pins recorded for this model family"
    agree, disagree, unknown, unpinned = [], [], [], []
    for line, text in serving_requirements(nb):
        try:
            req = Requirement(text.replace("{}", "0+runtime") if "{}" in text else text)
        except InvalidRequirement:
            continue
        name = _normalise(req.name)
        if name not in pins:
            continue
        if "{}" in text:
            unknown.append(f"line {line} {text!r} (version known only at run time)")
        elif req.specifier.contains(pins[name], prereleases=True):
            (agree if str(req.specifier) else unpinned).append(f"line {line} {text!r}")
        else:
            disagree.append(f"line {line} {text!r} (upstream {name}=={pins[name]})")
    where = f"upstream {upstream.get('source', 'dependency file')} @ {upstream.get('revision', '?')[:7]}"
    if disagree or unknown or not agree:
        found = "; ".join(disagree + unknown) or "no pinned serving requirement for a package the upstream file pins"
        return False, f"serving pins do not follow the {where}: {found}"
    return True, f"serving pins agree with the {where}: {'; '.join(agree)}"


# Calls that write to a path: (call name, position of the path argument).
_WRITE_CALLS = {"makedirs": 0, "mkdir": None, "copytree": 1, "copy": 1, "copy2": 1, "copyfile": 1,
                "move": 1, "write_text": None, "write_bytes": None, "save_model": 1}


def _inside_try(nb, line: int) -> bool:
    return any(t.body and t.body[0].lineno <= line <= t.body[-1].end_lineno and t.handlers
               for t in nb.nodes(ast.Try))


def local_disk0(nb, response, record, task):
    """forbidden_patterns//local_disk0 : nothing is written to /local_disk0 (not writable on
    serverless) unless that use is compute-adaptive (inside a try with a fallback, or the
    path can also be a /tmp / Volume / tempfile location). Other strings do not count."""
    sites = []
    for d in pc.downloads(nb, response):
        sites.append((d.line, f"download ({d.how})", d.kinds))
    for call in nb.calls:
        name = _call_name(call)
        if name not in _WRITE_CALLS and name != "open":
            continue
        if name == "open":
            mode = call.args[1] if len(call.args) > 1 else next((k.value for k in call.keywords if k.arg == "mode"), None)
            if mode is None or not any(pc._FILE_MODES.search(m) for m in nb.strings(mode)) or not call.args:
                continue
            node = call.args[0]
        elif _WRITE_CALLS[name] is None:
            holder = call.func.value if isinstance(call.func, ast.Attribute) else None
            node = holder
        else:
            position = _WRITE_CALLS[name]
            node = call.args[position] if len(call.args) > position else next(
                (k.value for k in call.keywords if k.arg in ("dst", "path", "name")), None)
        if node is not None:
            sites.append((call.lineno, name, pc._path_kinds(nb, node)))
    bad = [(line, how) for line, how, kinds in sites
           if kinds == {"local_disk0"} and not _inside_try(nb, line)]
    adaptive = [line for line, _, kinds in sites if "local_disk0" in kinds and (len(kinds) > 1 or _inside_try(nb, line))]
    if bad:
        return False, "writes to /local_disk0: " + "; ".join(f"line {line} {how}" for line, how in bad)
    if adaptive:
        return True, f"/local_disk0 used only with a fallback (lines {adaptive})"
    return True, f"no write to /local_disk0 ({len(sites)} write / download site(s) checked)"


_SDK_SERVED = ("ServedEntityInput", "ServedModelInput")


def sdk_enums(nb, response, record, task):
    """endpoint_deployment/sdk_enums : the SDK gets ServingModelWorkloadType, not a plain string.
    A string workload type in a REST body is valid for the REST API."""
    sdk, bad, rest = [], [], []
    for call in nb.calls:
        if _call_name(call) in _SDK_SERVED:
            for k in call.keywords:
                if k.arg != "workload_type":
                    continue
                values = nb.values(k.value)
                plain = [v for v in values if isinstance(v, (ast.Constant, ast.JoinedStr))]
                (bad if plain else sdk).append(call.lineno)
    for node in nb.nodes(ast.Dict):
        if any(isinstance(key, ast.Constant) and key.value == "workload_type" for key in node.keys):
            rest.append(node.lineno)
    if bad:
        return False, f"plain string workload_type passed to the SDK at lines {bad}"
    if sdk or rest:
        return True, "; ".join(x for x in (sdk and f"SDK enum at lines {sdk}",
                                           rest and f"REST body / dict at lines {rest}") if x)
    return True, "no workload type set (the default is used; GPU use is reported separately)"


def ai_gateway_config(nb, response, record, task):
    """endpoint_deployment/ai_gateway_config : an AI Gateway inference table is enabled on a
    config attached to the endpoint (SDK or REST body), and the deprecated
    AutoCaptureConfigInput is not used."""
    inference, why = scorers.ai_gateway_logging(response, task)["inference_table"]
    deprecated = [c.lineno for c in nb.calls if _call_name(c) == "AutoCaptureConfigInput"]
    if deprecated:
        return False, f"deprecated AutoCaptureConfigInput at lines {deprecated}"
    return inference, f"inference table: {why}"


_TABLE_EXISTS = re.compile(r"(?:^|\.)(?:tableExists|table_exists|exists)$")
def _is_table_read(nb, call) -> bool:
    """spark.table(...) / spark.read.table(...) / read_table(...), or spark.sql("SELECT ...")
    (a DROP / CREATE statement is not a read)."""
    name = _call_name(call)
    if name in ("table", "read_table"):
        return True
    return name == "sql" and bool(call.args) and any(re.match(r"\s*(?:SELECT|WITH)\b", t, re.I)
                                                     for t in nb.strings(call.args[0]))


def graceful_skip(nb, response, record, task):
    """ai_search/graceful_skip : the search step copes with a missing source table — an
    existence check, or a table read inside try/except. Creating the table (CREATE TABLE IF NOT
    EXISTS, DROP + write) is not a guard: the step then never meets a missing table."""
    checks = [c.lineno for c in nb.calls if any(_TABLE_EXISTS.search(d.rstrip("()")) for d in nb.dotted(c.func))
              and any("table" in d.lower() for d in nb.dotted(c.func))]
    if checks:
        return True, f"table existence check at line {checks[0]}"
    for t in nb.nodes(ast.Try):
        reads = [c.lineno for stmt in t.body for c in ast.walk(stmt) if isinstance(c, ast.Call) and _is_table_read(nb, c)]
        if reads and t.handlers:
            return True, f"table read at line {reads[0]} inside try/except"
    return False, "no table existence check and no table read inside try/except"


_DEPLOY_CALLS = ("create", "create_and_wait", "update_config", "update_config_and_wait", "put_ai_gateway",
                 "create_endpoint", "update_endpoint")


def _is_flag(nb, test) -> str | None:
    """The `if` test derives from a widget value or a user-set True/False flag (_user_flags)."""
    for n in pc.flow_nodes(nb, test):
        if isinstance(n, ast.Call) and any(d.endswith("widgets.get") for d in nb.dotted(n.func)):
            return f"widget at line {n.lineno}"
    for n in ast.walk(test):
        if isinstance(n, ast.Name) and n.id in _user_flags(nb):
            return f"flag {n.id}"
    return None


def _user_flags(nb) -> set:
    """Names set only by top-level `NAME = True/False` statements: a switch the user sets.
    A name also assigned inside an if / try / loop / function is derived from state (e.g.
    SKIP_DEPLOY = True when the endpoint is already READY): idempotency, not a gate."""
    top, nested = {}, set()
    for tree in nb.trees:
        top_level = {id(stmt) for stmt in tree.body}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for name in [t.id for t in targets if isinstance(t, ast.Name)]:
                constant = isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool)
                if id(node) in top_level and constant:
                    top[name] = True
                else:
                    nested.add(name)
    return set(top) - nested


def run_go_gate(nb, response, record, task):
    """endpoint_deployment/run_go_gate : the costly deploy step is behind a flag the user sets
    (a widget or a user-set True/False notebook constant), under any name — an `if` whose either branch
    holds the deploy call, or an early exit before it."""
    for node in nb.nodes(ast.If):
        flag = _is_flag(nb, node.test)
        if not flag:
            continue
        # either branch: `if RUN_GO: deploy` or `if not RUN_GO: print(...) else: deploy`
        body_calls = [c for stmt in node.body + node.orelse for c in ast.walk(stmt) if isinstance(c, ast.Call)]
        deploys = [c.lineno for c in body_calls if _call_name(c) in _DEPLOY_CALLS
                   and any("serving_endpoints" in d or "deploy" in d for d in nb.dotted(c.func))]
        exits = [c.lineno for c in body_calls if any(d.endswith("notebook.exit") for d in nb.dotted(c.func))]
        if deploys:
            return True, f"deploy call at line {deploys[0]} runs only if {flag} (if at line {node.lineno})"
        if exits:
            return True, f"notebook exits at line {exits[0]} unless {flag} (if at line {node.lineno})"
    return False, "no deploy call behind a widget or flag"


# ---- registry ---------------------------------------------------------------------------------
# key (as in the Sep–Oct 2026 score files) -> the flaw of the verbatim check, what now passes,
# and the corrected check. Severities stay those of expectations.json (see severity_note).

CORRECTIONS = {
    "download_staging/uses_tmp": {
        "flaw": "passes on any '/tmp' text (even a comment-free print or an unrelated path) and fails "
                "tempfile.mkdtemp() or a Volume, which also work on serverless",
        "accepts": "the Hugging Face download (and any staging copy) goes to /tmp, a tempfile folder or a "
                   "Unity Catalog Volume, never /local_disk0 (practical_checks.download_location_ok)",
        "check": uses_tmp,
    },
    "download_staging/hf_xet_disabled": {
        "flaw": "passes on the name HF_HUB_DISABLE_XET anywhere, even in markdown saying it was NOT set; "
                "fails a download that completed without it",
        "accepts": "Xet disabled with a disabling value before the download (in-process, subprocess or "
                   "shell), or recorded evidence the download completed (practical_checks.xet_download_ok)",
        "check": hf_xet_disabled,
    },
    "download_staging/deps_pinned": {
        "flaw": "requires the literal 'transformers==<digit>' and fails any '>=', so a range that excludes "
                "5.x or a model path without transformers fails",
        "accepts": "transformers 5.x excluded from the serving requirements, or no transformers on the model "
                   "path (practical_checks.transformers_pin)",
        "check": deps_pinned,
    },
    "forbidden_patterns/transformers>=": {
        "flaw": "fails the text 'transformers>=' even in a range that excludes 5.x ('>=4.41,<5')",
        "accepts": "no transformers requirement (%pip or serving) that allows 5.x",
        "check": transformers_open_range,
    },
    "model_registration/infer_signature": {
        "flaw": "looks for the word infer_signature, so an explicit ModelSignature(...) fails and a "
                "mention in a comment passes",
        "accepts": "log_model is given a signature built by infer_signature(...) or ModelSignature(...), or "
                   "another signature with recorded evidence the registration was served",
        "check": infer_signature,
    },
    "pyfunc_quality/real_gene_ids": {
        "flaw": "looks for the text 'vocab.txt', so an input built with TEDDY's own tokenizer fails and a "
                "mention in a comment passes",
        "accepts": "the input example or endpoint payload is built from real gene data: vocab.txt, a "
                   ".h5ad file, the upstream tokenizer, or Ensembl gene IDs (not ENSG00000FAKE); or the "
                   "notebook runs the upstream tokenizer and the input example has no random source",
        "check": real_gene_ids,
    },
    "model_registration/pip_reqs_from_source": {
        "flaw": "looks for the text 'pyproject.toml' / 'requirements.txt' anywhere (a download filter "
                "passes) and fails pins copied from the file without naming it",
        "accepts": "the notebook reads the model's dependency file, or the serving requirements pin at least "
                   "one package the upstream file pins and every such pin admits the upstream version "
                   "(family.json upstream_dependencies; a pin to the version installed at run time does not "
                   "count — it records the environment, not the source)",
        "check": pip_reqs_from_source,
    },
    "forbidden_patterns//local_disk0": {
        "flaw": "fails the text '/local_disk0' anywhere in code, including messages and compute-adaptive "
                "code that falls back to /tmp",
        "accepts": "no download, copy or file write to /local_disk0, except with a fallback",
        "check": local_disk0,
    },
    "endpoint_deployment/sdk_enums": {
        "flaw": "requires the name ServingModelWorkloadType, so a REST body with \"workload_type\": "
                "\"GPU_SMALL\" (valid for the REST API) fails",
        "accepts": "no plain-string workload_type passed to the SDK (enum, REST body, or none set)",
        "check": sdk_enums,
    },
    "endpoint_deployment/ai_gateway_config": {
        "flaw": "passes on the class name AiGatewayInferenceTableConfig (even an import) and fails an "
                "inference table set in a REST body",
        "accepts": "an inference table enabled on an AI Gateway config attached to the endpoint "
                   "(scorers.ai_gateway_logging), and no AutoCaptureConfigInput",
        "check": ai_gateway_config,
    },
    "ai_search/graceful_skip": {
        "flaw": "looks for words such as tableExists or 'table ... not ... exist' anywhere, so the task "
                "prompt quoted in the first cell or 'CREATE TABLE IF NOT EXISTS' passes, and fails "
                "try/except around the table read",
        "accepts": "a table existence check, or the table read inside try/except",
        "check": graceful_skip,
    },
    "endpoint_deployment/run_go_gate": {
        "flaw": "looks for the names run_go / RUN_GO / SKIP_DEPLOY anywhere (comments count; SKIP_DEPLOY is "
                "usually 'skip because the endpoint is already READY', not a user gate) and fails a "
                "widget guard with another name",
        "accepts": "the deploy call is behind a widget or a user-set True/False flag (any name; a flag the "
                   "notebook sets from endpoint state does not count), or the notebook exits before it "
                   "unless the flag is set",
        "check": run_go_gate,
    },
}

SEVERITY_NOTE = (
    "Severities are unchanged: each corrected check keeps its expectations.json severity because the "
    "corrected intent has the same consequence when it fails — a download that does not complete or "
    "lands where serverless cannot write, a model that loads transformers 5.x, a model with no "
    "signature (Unity Catalog refuses it) or a plain-string workload type the SDK rejects is critical; "
    "a made-up input example or an endpoint without an inference table is major; dependency "
    "provenance, the run gate and the missing-table guard are minor."
)


def record_from_outcome(outcome: dict | None) -> dict | None:
    """A run record built from outputs["outcome"] ({"outcome/<check>": bool}), for the two
    corrections that may read the measured outcome."""
    if not outcome:
        return None
    return {"outcome": {k: {"passed": bool(v)} for k, v in outcome.items()}}


def corrected_details(task: dict, response: str, record: dict | None = None) -> dict:
    """{key: (passed, reason, verbatim)} for every replicated sub-check of the task, keyed as
    the Sep–Oct 2026 score files: the corrected result for a flawed key, else the verbatim one."""
    verbatim = scorers.sub_check_scores(task, response)
    nb = pc.PracticalCode(response)
    out = {}
    for key, value in verbatim.items():
        if key in CORRECTIONS:
            passed, reason = CORRECTIONS[key]["check"](nb, response, record, task)
            out[key] = (bool(passed), reason, value)
        else:
            out[key] = (value, "same logic as Sep–Oct 2026 (no known flaw)", value)
    return out


def corrected_scores(task: dict, response: str, record: dict | None = None) -> dict:
    """{"corrected/<phase>/<check>": bool} — MLflow metric names."""
    return {CORRECTED_PREFIX + scorers.mlflow_metric_name(k): ok
            for k, (ok, _, _) in corrected_details(task, response, record).items()}


def flags(task: dict, response: str, record: dict | None = None) -> dict:
    """{key: {"verbatim", "corrected", "reason", "flaw"}} where the verbatim result disagrees."""
    return {k: {"verbatim": v, "corrected": ok, "reason": why, "flaw": CORRECTIONS[k]["flaw"]}
            for k, (ok, why, v) in corrected_details(task, response, record).items() if ok != v}


@mlflow_scorer
def corrected_replicated_checks(inputs, outputs):
    """The replicated Sep–Oct 2026 sub-checks, judged by intent (the scorer of record)."""
    task = scorers.TASKS[inputs["task_id"]]
    record = record_from_outcome(outputs.get("outcome"))
    return [Feedback(name=CORRECTED_PREFIX + scorers.mlflow_metric_name(k), value=ok,
                     rationale=reason + ("" if ok == v else f" [⚑ verbatim: {'pass' if v else 'fail'}]"))
            for k, (ok, reason, v) in corrected_details(task, outputs["notebook_source"], record).items()]


corrected_scorers = [corrected_replicated_checks]


# ---- CLI ----------------------------------------------------------------------------------------

def sep_oct_samples() -> list:
    """[(arm, old task id, task id, notebook source)] for the recorded Sep–Oct 2026 exports that the
    recorded score files were computed from (results/<arm>/<old id>.txt)."""
    return [(arm, old, new, (SEP_OCT_DIR / "results" / arm / f"{old}.txt").read_text(encoding="utf-8"))
            for arm in SEP_OCT_ARMS for old, new in SEP_OCT_TASKS.items()]


def _print(label: str, task_id: str, details: dict) -> None:
    passed = sum(ok for ok, _, _ in details.values())
    verbatim = sum(v for _, _, v in details.values())
    print(f"\n## {label} / {task_id}: corrected {passed} / {len(details)} passed "
          f"(verbatim {verbatim} / {len(details)})")
    for key, (ok, reason, v) in details.items():
        if key in CORRECTIONS or ok != v:
            mark = "" if ok == v else f"  ⚑ verbatim {'pass' if v else 'fail'}"
            print(f"- {key}: {'pass' if ok else 'FAIL'}{mark} — {reason}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default=family_settings.DEFAULT_FAMILY)
    parser.add_argument("--round", help="runs_<date> folder (default: family.json current_round)")
    parser.add_argument("--sep-oct", action="store_true", help="re-score the recorded Sep–Oct 2026 exports")
    parser.add_argument("--write", action="store_true", help="write the corrected score files")
    args = parser.parse_args(argv)
    written = {}
    if args.sep_oct:
        for arm, old, new, source in sep_oct_samples():
            details = corrected_details(scorers.TASKS[new], source, None)
            _print(f"Sep–Oct {arm}", f"{old} ({new})", details)
            written.setdefault(SEP_OCT_DIR / "scores_corrected" / f"{arm}_corrected.json", {})[old] = {
                CORRECTED_PREFIX + scorers.mlflow_metric_name(k): ok for k, (ok, _, _) in details.items()}
    else:
        run_dir = family_settings.round_dir(args.family, args.round, EVAL_DIR)
        for label, task_id, record, source in pc.round_samples(run_dir):
            details = corrected_details(scorers.TASKS[task_id], source, record)
            _print(label, task_id, details)
            written.setdefault(run_dir / "scores" / f"{label.replace(' ', '_')}_corrected.json", {})[task_id] = {
                CORRECTED_PREFIX + scorers.mlflow_metric_name(k): ok for k, (ok, _, _) in details.items()}
    if args.write:
        for path, scores in written.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path.relative_to(EVAL_DIR) if path.is_relative_to(EVAL_DIR) else path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
