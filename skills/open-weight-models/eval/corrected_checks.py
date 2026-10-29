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
files (scores/<run>_corrected.json for a round; models/teddy/runs_20260924/scores_corrected/ for Sep–Oct,
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
SEP_OCT_DIR = EVAL_DIR / "models" / "teddy" / "runs_20260924"
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


_REAL_DATA_FILES = ("vocab.txt", ".h5ad")
_OUTPUT_KEYS = re.compile(r"save|output|out_dir|^out$|dest", re.I)
_FILE_READ_CALLS = ("glob", "load_dataset", "load_from_disk", "from_file", "read_parquet", "read_csv",
                    "read_h5ad", "open", "load", "read_table")


def _tokenizer_calls(nb, packages: tuple) -> list:
    """Calls into the upstream package whose dotted name mentions token (GeneTokenizer(...),
    tokenize(...), _build_batch_tensors(...))."""
    return [c for c in nb.calls if any(d.split(".")[0] in packages and "token" in d.lower() and "." in d
                                       for d in nb.dotted(c.func))]


def _tokenizer_outputs(nb, packages: tuple) -> list:
    """Paths an upstream tokenizer call writes to: values under save / output / dest keys of its
    keyword arguments or of a dict passed to it (tokenize(..., tokenization_args={"save_dir": P}))."""
    out = []
    for call in _tokenizer_calls(nb, packages):
        for k in call.keywords:
            if k.arg and _OUTPUT_KEYS.search(k.arg):
                out += pc.path_strings(nb, k.value)
            out += [t for key_node in pc._keyed_nodes(nb, k.value, _OUTPUT_KEYS) for t in pc.path_strings(nb, key_node)]
        for a in call.args:
            out += [t for key_node in pc._keyed_nodes(nb, a, _OUTPUT_KEYS) for t in pc.path_strings(nb, key_node)]
    return [t for t in dict.fromkeys(out) if t.replace("{}", "").strip("/")]


def _real_gene_source(nb, node, packages: tuple):
    """What the scored value's gene IDs derive from (data flow), or None: the model vocabulary
    (vocab.txt), a real single-cell file (.h5ad / read_h5ad), an upstream tokenizer call, or a file
    the upstream tokenizer wrote (read back from its save / output path)."""
    tokenizers = {id(c) for c in _tokenizer_calls(nb, packages)}
    outputs = _tokenizer_outputs(nb, packages)
    for n in pc.flow_nodes(nb, node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            if any(f in n.value for f in _REAL_DATA_FILES):
                return f"line {n.lineno}: reads {n.value!r}"
        elif isinstance(n, ast.Call):
            if id(n) in tokenizers:
                return f"line {n.lineno}: upstream tokenizer {nb.dotted(n.func)[0]}"
            if any(d.endswith("read_h5ad") for d in nb.dotted(n.func)):
                return f"line {n.lineno}: {nb.dotted(n.func)[0]}"
            if outputs and _call_name(n) in _FILE_READ_CALLS:
                for a in list(n.args) + [k.value for k in n.keywords]:
                    hit = next((t for t in pc.path_strings(nb, a) for base in outputs if pc._under(t, base)), None)
                    if hit:
                        return f"line {n.lineno}: reads {hit!r}, written by the upstream tokenizer"
    return None


def real_gene_ids(nb, response, record, task):
    """pyfunc_quality/real_gene_ids : the input example logged with the model (or the payload sent
    to the endpoint) derives — by data flow, not presence — from real gene data: the model
    vocabulary, a real .h5ad file, the upstream tokenizer, or files the upstream tokenizer wrote."""
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
                return True, f"{kind} at line {line} derives from real gene data ({source})"
    return False, (f"no input example or endpoint payload derives from real gene data "
                   f"({len(examples)} input example(s), {len(payloads)} endpoint payload argument(s) checked)")


_DEPENDENCY_FILES = ("pyproject.toml", "requirements.txt", "setup.cfg", "setup.py")
_FILE_READS = ("open", "read_text", "read_bytes")


def _dependency_reads(nb) -> list:
    """(call, path) for every read of a file named like a dependency spec."""
    out = []
    for call in nb.calls:
        name = _call_name(call)
        holder = call.func.value if isinstance(call.func, ast.Attribute) else None
        nodes = list(call.args[:1]) if name == "open" else (
            [a for v in nb.values(holder) if isinstance(v, ast.Call) for a in v.args[:1]]
            if name in _FILE_READS and holder is not None else [])
        for node in nodes:
            for text in pc.path_strings(nb, node):
                if text.rsplit("/", 1)[-1] in _DEPENDENCY_FILES:
                    out.append((call, text))
    return out


def _model_dependency_read(nb, response):
    """A read of the MODEL's dependency file (under the download destination, or a copy of it)
    whose parsed contents flow into the serving requirements; (line, path) or None."""
    found = pc.downloads(nb, response)
    model_reads = [(call, path) for call, path in _dependency_reads(nb)
                   if any(pc._under(path, base) for d in found for base in pc._staged_paths(nb, d))]
    if not model_reads:
        return None
    reqs = [node for call in pc._mlflow_log_model_calls(nb) for kwargs in nb.keyword_maps(call)
            for key in ("pip_requirements", "extra_pip_requirements") if (node := kwargs.get(key)) is not None]
    for call, path in model_reads:
        # followed through `with open(P) as f: tomllib.load(f)` (with-targets are flow sources)
        if any(pc._derives(nb, req, lambda n, c=call: n is c) for req in reqs):
            return call.lineno, path
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
    own dependency file — the notebook reads that file from the downloaded model (or a copy of it)
    and the parsed pins flow into pip_requirements, or the serving pins agree with it."""
    read = _model_dependency_read(nb, response)
    if read:
        return True, (f"line {read[0]} reads the model's dependency file {read[1]!r} and its contents flow "
                      f"into the serving requirements")
    upstream = (family_settings.load_family(task["model_family"], EVAL_DIR).get("upstream_dependencies")
                or {}) if task.get("model_family") else {}
    pins = {_normalise(k): v for k, v in (upstream.get("pins") or {}).items()}
    if not pins:
        return False, "no model dependency file read into the requirements, and no upstream pins recorded"
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


def _probe(nb, call, fallbacks: dict):
    """A /local_disk0 write inside a try body whose handler assigns a fallback path that the
    notebook then uses (the README's makedirs probe); the try's line, or None."""
    for t in nb.nodes(ast.Try):
        if _contains(t.body, call) and t.handlers:
            handled = pc.assigned_values([s for h in t.handlers for s in h.body])
            if any(name in fallbacks for name in handled):
                return t.lineno
    return None


def local_disk0(nb, response, record, task):
    """forbidden_patterns//local_disk0 : nothing is written to /local_disk0 (not writable on
    serverless) unless it is compute-adaptive: the path variable has an alternative (/tmp,
    tempfile, Volume) assigned in the try handler or the other if branch, and that same
    variable is what the download / write uses; or the write is the probe inside such a try.
    A try with any handler (even `except: pass`) is not enough. Other strings do not count."""
    fallbacks = pc.local_disk0_fallbacks(nb)
    sites = []
    for d in pc.downloads(nb, response):
        sites.append((d.line, f"download ({d.how})", d.kinds, pc.download_site(nb, d), d.node))
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
            node = call.func.value if isinstance(call.func, ast.Attribute) else None
        else:
            position = _WRITE_CALLS[name]
            node = call.args[position] if len(call.args) > position else next(
                (k.value for k in call.keywords if k.arg in ("dst", "path", "name")), None)
        if node is not None:
            sites.append((call.lineno, name, pc._path_kinds(nb, node), node, call))
    bad, adaptive = [], []
    for line, how, kinds, node, call in sites:
        if "local_disk0" not in kinds:
            continue
        if node is not None and pc.uses_fallback(nb, node, fallbacks):
            adaptive.append(line)
        elif call is not None and _probe(nb, call, fallbacks):
            adaptive.append(line)
        else:
            bad.append(f"line {line} {how}")
    if bad:
        return False, "writes to /local_disk0 without a used fallback: " + "; ".join(bad)
    if adaptive:
        return True, f"/local_disk0 used only with a fallback path that is then used (lines {adaptive})"
    return True, f"no write to /local_disk0 ({len(sites)} write / download site(s) checked)"


def endpoint_query(nb, response, record, task):
    """smoke_tests/endpoint_query : the notebook sends a request to the serving endpoint, through
    the SDK's serving_endpoints.query or an HTTP call to its /invocations URL (api_client.do,
    requests), in code that runs: at top level, or in a notebook function the notebook calls."""
    queries = [c for c in nb.calls if pc._is_endpoint_query(nb, c)]
    if not queries:
        return False, "no request sent to a serving endpoint (SDK query or HTTP call to /invocations)"
    for call in queries:
        owners = [f for f in nb.functions.values() if f.lineno <= call.lineno <= f.end_lineno]
        if not owners:
            return True, f"endpoint queried at line {call.lineno}"
        owner = max(owners, key=lambda f: f.lineno)
        callers = [c.lineno for c in nb.calls if _call_name(c) == owner.name
                   and not owner.lineno <= c.lineno <= owner.end_lineno]
        if callers:
            return True, f"endpoint queried at line {call.lineno} in {owner.name}(), called at line {callers[0]}"
    return False, (f"endpoint query only inside functions the notebook never calls "
                   f"(lines {[c.lineno for c in queries]})")


_SDK_SERVED = ("ServedEntityInput", "ServedModelInput")
_HTTP_SENDS = ("do", "post", "put", "patch", "request")


def _rest_bodies(nb) -> list:
    """Body arguments of HTTP calls to a serving-endpoints URL (api_client.do, requests.post /
    put / patch, ...): (line, node)."""
    out = []
    for call in nb.calls:
        if _call_name(call) not in _HTTP_SENDS:
            continue
        args = list(call.args) + [k.value for k in call.keywords]
        if not any("serving-endpoints" in t for a in args for t in nb.strings(a)):
            continue
        out += [(call.lineno, k.value) for k in call.keywords if k.arg in ("body", "json", "data")]
        out += [(call.lineno, a) for a in call.args[2:]]
    return out


def _is_plain_string(nb, node) -> bool:
    return any(isinstance(v, (ast.Constant, ast.JoinedStr)) and not (isinstance(v, ast.Constant) and v.value is None)
               for v in nb.values(node))


def sdk_enums(nb, response, record, task):
    """endpoint_deployment/sdk_enums : the SDK gets ServingModelWorkloadType, not a plain string —
    for ServedEntityInput / ServedModelInput constructors, keywords and **kwargs alike. A string is
    valid only in a body actually sent to the REST API, or through the SDK's from_dict (which
    converts it)."""
    sdk, bad = [], []
    for call in nb.calls:
        if _call_name(call) not in _SDK_SERVED:
            continue
        for kwargs in nb.keyword_maps(call):
            node = kwargs.get("workload_type")
            if node is None:
                continue
            (bad if _is_plain_string(nb, node) else sdk).append(call.lineno)
    rest = []
    for line, body in _rest_bodies(nb):
        if any(isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == "workload_type" for k in n.keys)
               for n in pc.flow_nodes(nb, body)):
            rest.append(line)
    if bad:
        return False, f"plain string workload_type passed to the SDK at lines {sorted(set(bad))}"
    if sdk or rest:
        return True, "; ".join(x for x in (sdk and f"SDK enum at lines {sorted(set(sdk))}",
                                           rest and f"REST body sent at lines {rest}") if x)
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
_INDEX_SOURCE_KEYS = ("source_table_name", "source_table")


def _index_sources(nb) -> list:
    """Table names a Vector Search index is built from (create_delta_sync_index(source_table_name=...),
    DeltaSyncVectorIndexSpecRequest(source_table=...), or a REST / dict body with those keys)."""
    out = []
    for call in nb.calls:
        for kwargs in nb.keyword_maps(call):
            for key in _INDEX_SOURCE_KEYS:
                if kwargs.get(key) is not None:
                    out += nb.strings(kwargs[key])
    for node in nb.nodes(ast.Dict):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value in _INDEX_SOURCE_KEYS:
                out += nb.strings(v)
    return [t for t in dict.fromkeys(out) if t.replace("{}", "").strip(".")]


def _is_table_read(nb, call) -> bool:
    """spark.table(...) / spark.read.table(...) / read_table(...), or spark.sql("SELECT ...")
    (a DROP / CREATE statement is not a read)."""
    name = _call_name(call)
    if name in ("table", "read_table"):
        return True
    return name == "sql" and bool(call.args) and any(re.match(r"\s*(?:SELECT|WITH)\b", t, re.I)
                                                     for t in nb.strings(call.args[0]))


def _raises(stmts: list) -> bool:
    return any(isinstance(n, ast.Raise) for stmt in stmts for n in ast.walk(stmt))


def graceful_skip(nb, response, record, task):
    """ai_search/graceful_skip : the search step copes with its source table missing — an
    existence check ON THE TABLE THE INDEX IS BUILT FROM whose missing branch skips or creates it
    (does not raise), or the read of that table inside try/except whose handler does not re-raise.
    Creating the table (CREATE TABLE IF NOT EXISTS, DROP + write) is not a guard."""
    sources = _index_sources(nb)
    if not sources:
        return False, "no Vector Search index source table found"
    names = lambda node: set(nb.strings(node)) & set(sources)  # noqa: E731
    for node in nb.nodes(ast.If):
        checks = [c for c in ast.walk(node.test) if isinstance(c, ast.Call)
                  and any(_TABLE_EXISTS.search(d.rstrip("()")) for d in nb.dotted(c.func))
                  and any("table" in d.lower() for d in nb.dotted(c.func)) and c.args and names(c.args[0])]
        if not checks:
            continue
        negated = isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
        missing = node.body if negated else node.orelse
        if not _raises(missing):
            return True, (f"line {node.lineno} checks that the index source table exists and the missing case "
                          f"{'skips' if not missing else 'is handled without raising'}")
    for t in nb.nodes(ast.Try):
        reads = [c.lineno for stmt in t.body for c in ast.walk(stmt) if isinstance(c, ast.Call)
                 and _is_table_read(nb, c) and c.args and names(c.args[0])]
        if reads and t.handlers and not any(_raises(h.body) for h in t.handlers):
            return True, f"read of the index source table at line {reads[0]} inside try/except that does not re-raise"
    return False, (f"no existence check or non-raising try/except on the index source table "
                   f"({', '.join(sources)})")


_DEPLOY_CALLS = ("create", "create_and_wait", "update_config", "update_config_and_wait", "put_ai_gateway",
                 "create_endpoint", "update_endpoint")


_WIDGET_DEFINE = ("widgets.dropdown", "widgets.text", "widgets.combobox", "widgets.multiselect")
_PARAM_READS = ("widgets.get", "getArgument", "environ.get", "getenv")
_UNKNOWN = object()


def _is_flag(nb, test) -> str | None:
    """The `if` test derives from a widget / notebook parameter / environment value, or a
    user-set True/False flag (_user_flags)."""
    for n in pc.flow_nodes(nb, test):
        if isinstance(n, ast.Call) and any(d.endswith(_PARAM_READS) for d in nb.dotted(n.func)):
            return f"{'widget' if any(d.endswith('widgets.get') for d in nb.dotted(n.func)) else 'parameter'} at line {n.lineno}"
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


def _widget_default(nb, name: str):
    """Default of the widget `name` from its dbutils.widgets.dropdown / text / ... definition
    (second argument or defaultValue=); _UNKNOWN if it is not defined in the notebook or its
    definitions disagree."""
    found = set()
    for call in nb.calls:
        if not any(d.endswith(_WIDGET_DEFINE) for d in nb.dotted(call.func)):
            continue
        name_node = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "name"), None)
        if name_node is None or name not in nb.strings(name_node):
            continue
        node = call.args[1] if len(call.args) > 1 else next(
            (k.value for k in call.keywords if k.arg in ("defaultValue", "default_value", "default")), None)
        values = nb.strings(node) if node is not None else []
        if len(values) != 1:
            return _UNKNOWN
        found.add(values[0])
    return found.pop() if len(found) == 1 else _UNKNOWN


def _default_value(nb, node, line: int, depth: int = 0):
    """The value an expression has when the notebook runs with every widget, notebook parameter
    and environment variable at its default; _UNKNOWN when that cannot be read from the code."""
    if depth > 12:
        return _UNKNOWN
    value = lambda n: _default_value(nb, n, line, depth + 1)  # noqa: E731
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        reaching = pc._reaching(nb, node.id, line)
        if not reaching:
            return _UNKNOWN
        results = {repr(value(v)) if value(v) is not _UNKNOWN else "?" for v in reaching}
        return value(reaching[0]) if len(results) == 1 and "?" not in results else _UNKNOWN
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = value(node.operand)
        return _UNKNOWN if inner is _UNKNOWN else not inner
    if isinstance(node, ast.BoolOp):
        parts = [value(v) for v in node.values]
        if any(p is _UNKNOWN for p in parts):
            return _UNKNOWN
        return all(parts) if isinstance(node.op, ast.And) else any(parts)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = value(node.left), value(node.comparators[0])
        if left is _UNKNOWN or right is _UNKNOWN:
            return _UNKNOWN
        op = node.ops[0]
        try:
            return {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                    ast.In: lambda: left in right, ast.NotIn: lambda: left not in right,
                    ast.Is: lambda: left is right, ast.IsNot: lambda: left is not right}[type(op)]()
        except (KeyError, TypeError):
            return _UNKNOWN
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items = [value(e) for e in node.elts]
        return _UNKNOWN if any(i is _UNKNOWN for i in items) else tuple(items)
    if isinstance(node, ast.Call):
        dotted = nb.dotted(node.func)
        args = node.args
        if any(d.endswith("widgets.get") for d in dotted) and args:
            names = nb.strings(args[0])
            return _widget_default(nb, names[0]) if len(names) == 1 else _UNKNOWN
        if any(d.endswith("getArgument") for d in dotted) and args:
            if len(args) > 1:
                return value(args[1])
            names = nb.strings(args[0])
            return _widget_default(nb, names[0]) if len(names) == 1 else _UNKNOWN
        if any(d.endswith(("environ.get", "getenv")) for d in dotted) and args:
            return value(args[1]) if len(args) > 1 else None      # unset variable -> None
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("lower", "upper", "strip") and not args:
            inner = value(node.func.value)
            return getattr(inner, node.func.attr)() if isinstance(inner, str) else _UNKNOWN
        if isinstance(node.func, ast.Name) and node.func.id in ("bool", "str", "int") and len(args) == 1:
            inner = value(args[0])
            if inner is _UNKNOWN:
                return _UNKNOWN
            try:
                return {"bool": bool, "str": str, "int": int}[node.func.id](inner)
            except (TypeError, ValueError):
                return _UNKNOWN
    return _UNKNOWN


def _contains(stmts: list, node) -> bool:
    return any(n is node for stmt in stmts for n in ast.walk(stmt))


def _deploy_sites(nb) -> list:
    """Costly deploy calls (serving-endpoint create / update): the call itself, or, for a call
    inside a notebook function, each call of that function (where the deploy really happens)."""
    calls = [c for c in nb.calls if _call_name(c) in _DEPLOY_CALLS
             and any("serving_endpoints" in d or "deploy" in d for d in nb.dotted(c.func))]
    sites = []
    for call in calls:
        owner = next((f for f in nb.functions.values() if _contains(f.body, call)), None)
        if owner is None:
            sites.append(call)
        else:
            sites += [c for c in nb.calls if _call_name(c) == owner.name and not _contains(owner.body, c)]
    return sorted(set(sites), key=lambda c: c.lineno)


def _top_level(nb) -> set:
    return {id(stmt) for tree in nb.trees for stmt in tree.body}


def _gate_for(nb, site, top: set):
    """(flag, if/assert node, side) of a gate that the deploy `site` needs: side "body" / "orelse" when the
    site is inside that branch of a flag `if`, "exit-body" / "exit-orelse" for a top-level flag `if`
    BEFORE the site whose branch stops the notebook (dbutils.notebook.exit / sys.exit / raise), and
    "assert" for a top-level `assert FLAG` before the site (it stops the notebook while the flag is off).
    The innermost enclosing gate wins; None when nothing gates the site."""
    enclosing = []
    for node in nb.nodes(ast.If):
        flag = _is_flag(nb, node.test)
        if not flag:
            continue
        if _contains(node.body, site):
            enclosing.append((node, flag, "body"))
        elif _contains(node.orelse, site):
            enclosing.append((node, flag, "orelse"))
    if enclosing:
        node, flag, side = max(enclosing, key=lambda e: e[0].lineno)
        return flag, node, side
    for node in sorted(nb.nodes(ast.If) + nb.nodes(ast.Assert), key=lambda n: n.lineno):
        if id(node) not in top or node.end_lineno >= site.lineno:
            continue
        flag = _is_flag(nb, node.test)
        if not flag:
            continue
        if isinstance(node, ast.Assert):
            return flag, node, "assert"
        for side in ("body", "orelse"):
            if any(_stops(nb, n) for stmt in getattr(node, side) for n in ast.walk(stmt)):
                return flag, node, f"exit-{side}"
    return None


def _stops(nb, n) -> bool:
    if isinstance(n, ast.Raise):
        return True
    return isinstance(n, ast.Call) and any(d.endswith(("notebook.exit", "sys.exit")) for d in nb.dotted(n.func))


def run_go_gate(nb, response, record, task):
    """endpoint_deployment/run_go_gate : every costly deploy call (serving-endpoint create / update)
    is reachable only when a flag the user sets is on — a widget, a notebook / environment parameter
    or a user-set True/False notebook constant, under any name: the call sits in the flag-on branch
    of an `if`, or a top-level `if` before it stops the notebook in the flag-off branch — AND the
    gate defaults to off: with every widget, parameter and environment variable at its default, the
    deploy does not run. A default that cannot be read from the code fails."""
    sites = _deploy_sites(nb)
    if not sites:
        return False, "no serving-endpoint deploy call"
    top, ok, problems = _top_level(nb), [], []
    for site in sites:
        where = f"deploy call at line {site.lineno}"
        gate = _gate_for(nb, site, top)
        if gate is None:
            problems.append(f"{where} is not behind a widget or flag")
            continue
        flag, node, side = gate
        kind = "assert" if side == "assert" else "if"
        default = _default_value(nb, node.test, node.lineno)
        if default is _UNKNOWN:
            problems.append(f"{where} is gated by {flag} ({kind} at line {node.lineno}), but the gate's default "
                            f"cannot be determined from the code")
            continue
        test = bool(default)
        # does the deploy run at the defaults?
        runs = {"body": test, "orelse": not test, "exit-body": not test, "exit-orelse": test, "assert": test}[side]
        if runs:
            problems.append(f"{where} is gated by {flag} ({kind} at line {node.lineno}), but the gate defaults to on "
                            f"(the deploy runs with the defaults)")
        else:
            ok.append(f"{where} runs only when {flag} is turned on ({kind} at line {node.lineno}; off by default)")
    if problems:
        return False, "; ".join(problems)
    return True, "; ".join(ok)


# ---- registry ---------------------------------------------------------------------------------
# key (as in the Sep–Oct 2026 score files) -> the flaw of the verbatim check, what now passes,
# and the corrected check. Severities stay those of expectations.json (see severity_note).

CORRECTIONS = {
    "download_staging/uses_tmp": {
        "flaw": "passes on any '/tmp' text (even a comment-free print or an unrelated path) and fails "
                "tempfile.mkdtemp() or a Volume, which also work on serverless",
        "accepts": "the Hugging Face download (and any staging copy) goes to /tmp, a tempfile folder or a "
                   "Unity Catalog Volume (practical_checks.download_location_ok); /local_disk0 only when "
                   "compute-adaptive, with the fallback rule of forbidden_patterns//local_disk0 (added after "
                   "round 2: a subprocess download's path is followed into the script text)",
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
        "accepts": "the logged input example (or endpoint payload) derives by data flow from real gene "
                   "data: the model vocabulary (vocab.txt), a real .h5ad file, an upstream tokenizer call, "
                   "or files the upstream tokenizer wrote (read back from its save / output path)",
        "check": real_gene_ids,
    },
    "model_registration/pip_reqs_from_source": {
        "flaw": "looks for the text 'pyproject.toml' / 'requirements.txt' anywhere (a download filter "
                "passes) and fails pins copied from the file without naming it",
        "accepts": "the notebook reads the model's dependency file from the downloaded model (or a copy) "
                   "and the parsed contents flow into pip_requirements, or the serving requirements pin at least "
                   "one package the upstream file pins and every such pin admits the upstream version "
                   "(family.json upstream_dependencies; a pin to the version installed at run time does not "
                   "count — it records the environment, not the source)",
        "check": pip_reqs_from_source,
    },
    "forbidden_patterns//local_disk0": {
        "flaw": "fails the text '/local_disk0' anywhere in code, including messages and compute-adaptive "
                "code that falls back to /tmp",
        "accepts": "no download, copy or file write to /local_disk0, unless the path variable has an "
                   "alternative (/tmp, tempfile, Volume) assigned in the try handler or other if branch and "
                   "that variable is what is used (or the write is the probe inside that try)",
        "check": local_disk0,
    },
    "smoke_tests/endpoint_query": {
        "flaw": "passes on text such as 'test ... endpoint' anywhere, including comments and markdown, and "
                "fails a request posted to /invocations with w.api_client.do, the way the skill sends "
                "inference controls (found on round 2, with-skill teddy-001)",
        "accepts": "a request sent to the serving endpoint (serving_endpoints.query or an HTTP call to its "
                   "/invocations URL) in code that runs: top level or a notebook function that is called",
        "check": endpoint_query,
    },
    "endpoint_deployment/sdk_enums": {
        "flaw": "requires the name ServingModelWorkloadType, so a REST body with \"workload_type\": "
                "\"GPU_SMALL\" (valid for the REST API) fails",
        "accepts": "no plain-string workload_type passed to an SDK constructor (keywords or **kwargs); "
                   "a string only in a body sent to the REST API",
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
        "accepts": "an existence check on the table the index is built from whose missing branch does not "
                   "raise, or the read of that table inside try/except that does not re-raise",
        "check": graceful_skip,
    },
    "endpoint_deployment/run_go_gate": {
        "flaw": "looks for the names run_go / RUN_GO / SKIP_DEPLOY anywhere (comments count; SKIP_DEPLOY is "
                "usually 'skip because the endpoint is already READY', not a user gate) and fails a "
                "widget guard with another name",
        "accepts": "every serving-endpoint deploy call is reachable only when a flag the user sets is on "
                   "(a widget, a notebook / environment parameter or a user-set True/False constant, any name; "
                   "a flag the notebook sets from endpoint state does not count): the call is in the flag-on "
                   "branch, a top-level `if` before it stops the notebook in the flag-off branch, or a top-level "
                   "`assert FLAG` before it — and the gate defaults to off (widget default 'false', constant "
                   "False, unset parameter), so nothing costly runs until the user turns it on; a default that "
                   "cannot be read from the code fails. Not scored: how the agent read a request such as "
                   "\"Deploy … Write the code.\" (switching the flag on for the run itself, or stopping to ask, are "
                   "both fine if it says so before spending); that is reported per run, not counted",
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
