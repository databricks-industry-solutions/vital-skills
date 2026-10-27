#!/usr/bin/env python3
"""Outcome checks for one eval run: does what the run built actually work?

Model-agnostic: the fixed cells, the expected embedding dimension and which tasks have
Vector Search come from the family settings, models/<family>/family.json ("outcome").

Fair to any input format: the request is built in the format the endpoint DECLARES — the
input signature of the registered model version it serves — not in the skill's format.
Recognised formats (input_format()): the shared single-cell format of teddy.md / scgpt.md
(adata_sparsematrix + adata_obs + adata_var), one row per cell of gene identifiers + values,
and one row per cell of pre-tokenized integer token IDs. Whether the endpoint uses the
shared format is reported separately, as a convention check, never as "broken".

Run after a run is exported and `resources_created` is filled in its record, and
BEFORE tearing anything down (see eval/README.md). From a laptop:

  python3 skills/open-weight-models/eval/check_outcome.py --profile <profile> \\
      --run-id <run_id> --task-id <task_id> --catalog <catalog> [--reference-endpoint <name>]

or from a notebook (the repo's eval/ folder on sys.path):

  from check_outcome import run_outcome_checks
  from family_settings import load_family, outcome_checks
  checks = outcome_checks(load_family("teddy"), "teddy-002")
  results = run_outcome_checks(WorkspaceClient(), resources, checks, family="teddy")
  # signature= the served version's input signature, e.g. load_signature(uc_model, version)

Checks (metric names start with "outcome/"):
  endpoint_ready              serving endpoint state is READY
  serves_recorded_version     it serves the recorded UC model + version
  returns_valid_embeddings    the fixed cells, sent in the endpoint's declared format (cells
                              of different lengths in one request), return one finite,
                              non-zero embedding per cell of the family's dimension (TEDDY 70M: 512)
  accepts_expression_input    the endpoint takes a cell's expression + gene identifiers, not
                              pre-tokenized token IDs (tokenization left to the client)
  vs_endpoint_online          search tasks: the Vector Search endpoint is ONLINE
  index_ready                 search tasks: the index reports ready
  nn_returns_cell_type        search tasks: a nearest-neighbour query with a fixture
                              embedding, asking only for columns the index holds, returns
                              rows that include a cell-type column
Convention check (metric name starts with "convention/", reported on its own):
  shared_single_cell_format   the declared input is the shared single-cell format
Optional, informational only (no pass/fail): reference_cosine — per-cell cosine similarity
of the fixture embeddings against another endpoint (e.g. a reference TEDDY deployment).
A check whose request cannot be built (no signature, unrecognised format) is recorded as
null with the reason; the scorers then refuse the run until it is checked by hand.

Results are written into the run record under "outcome" as
{"<prefix>/<check>": {"passed": true|false|null, "detail": "..."}}; null = not measured /
informational. The endpoint's serving settings (workload type and size, scale to zero, AI
Gateway inference table and usage tracking) go into the record's "endpoint_config".
Needs databricks-sdk; the signature is read with mlflow (mlflow-skinny is enough).
Requests are POSTed to /serving-endpoints/<name>/invocations with `w.api_client.do`, inference
controls under "params" (see _query). SDK names verified against databricks-sdk 0.157.0,
mlflow names against mlflow 3.17.2.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import family_settings  # noqa: E402 — same folder

PREFIX = "outcome/"
CONVENTION_PREFIX = "convention/"
SHARED_FORMAT_CHECK = f"{CONVENTION_PREFIX}shared_single_cell_format"
CELL_TYPE_COLUMN = re.compile(r"cell[_ ]?type", re.I)


# ---- request building (pure, unit-tested) -----------------------------------

def _split_json(column: str, values: list) -> str:
    """A one-column DataFrame as pandas `to_json(orient="split")` writes it."""
    return json.dumps({"columns": [column], "index": list(range(len(values))),
                       "data": [[v] for v in values]}, separators=(",", ":"))


def build_teddy_request(genes: list, expression: list, cell_ids: list) -> dict:
    """One `dataframe_records` record per the TEDDY serving contract (teddy.md:19-99):
    a dense cells x genes matrix, cell metadata and gene metadata as JSON orient=split."""
    if len(expression) != len(cell_ids):
        raise ValueError("one expression row per cell id")
    if any(len(row) != len(genes) for row in expression):
        raise ValueError("every expression row needs one value per gene")
    return {
        "adata_sparsematrix": [[float(x) for x in row] for row in expression],
        "adata_obs": _split_json("cell_id", list(cell_ids)),
        "adata_var": _split_json("index", list(genes)),
    }


def load_fixture(family: str = family_settings.DEFAULT_FAMILY, eval_dir: Path = EVAL_DIR) -> dict:
    """The family's fixed request (family.json outcome.fixture), with `expected.embedding_dim`
    taken from the family settings so there is one source for it."""
    settings = family_settings.load_family(family, eval_dir)
    path = family_settings.family_dir(family, eval_dir) / settings["outcome"]["fixture"]
    with open(path, encoding="utf-8") as handle:
        fixture = json.load(handle)
    if fixture["expected"]["embedding_dim"] != settings["outcome"]["embedding_dim"]:
        raise ValueError(f"{path}: embedding_dim differs from {family}/family.json")
    return fixture


def fill_catalog(resources: dict, catalog: str | None) -> dict:
    """Resource names with the "<catalog>" placeholder replaced (run records keep the placeholder)."""
    if any("<catalog>" in str(v) for v in resources.values()):
        if not catalog:
            raise ValueError("resource names contain <catalog>: pass --catalog")
        return {k: str(v).replace("<catalog>", catalog) for k, v in resources.items()}
    return dict(resources)


# ---- declared input format -> request (pure, unit-tested) --------------------------

SHARED_FORMAT = "shared single-cell format"     # teddy.md:19-99, scgpt.md: adata_* columns
GENES_AND_VALUES = "gene identifiers + values per cell"
TOKEN_IDS = "pre-tokenized token IDs per cell"
SHARED_COLUMNS = ("adata_sparsematrix", "adata_obs", "adata_var")
_INTEGER_TYPES = {"integer", "long"}
_FLOAT_TYPES = {"float", "double"}


def _array_type(column: dict) -> str | None:
    """Element type of a one-level array column of a signature (Schema.to_dict()), else None."""
    items = column.get("items")
    return items.get("type") if column.get("type") == "array" and isinstance(items, dict) else None


def input_format(inputs: list | None) -> tuple:
    """(format, {role: column}) from a model signature's inputs, as `Schema.to_dict()` writes
    them ([{"name", "type", "items", "required"}]); (None, {}) if not recognised.

    Gene identifiers + values: exactly one array-of-string column and one numeric array
    column, and no other required column. Token IDs: exactly one integer array column, no
    string or float array, no other required column."""
    inputs = inputs or []
    names = {c.get("name") for c in inputs}
    if set(SHARED_COLUMNS) <= names:
        return SHARED_FORMAT, {c: c for c in SHARED_COLUMNS}
    strings = [c["name"] for c in inputs if _array_type(c) == "string"]
    floats = [c["name"] for c in inputs if _array_type(c) in _FLOAT_TYPES]
    integers = [c["name"] for c in inputs if _array_type(c) in _INTEGER_TYPES]
    required = {c["name"] for c in inputs if c.get("required", True)}
    numeric = floats + integers
    if len(strings) == 1 and len(numeric) == 1 and required <= {strings[0], numeric[0]}:
        return GENES_AND_VALUES, {"genes": strings[0], "values": numeric[0],
                                  "integer_values": numeric[0] in integers}
    if len(integers) == 1 and not strings and not floats and required <= {integers[0]}:
        return TOKEN_IDS, {"token_ids": integers[0]}
    return None, {}


def fixture_cells(fixture: dict) -> list:
    """Per fixture cell, its expressed genes as (vocab position, gene, value), highest value
    first (ties: vocab order). Zero-valued genes are left out, so cells differ in length.
    The fixture genes are the first lines of the family's vocab.txt (fixture "description"),
    so a gene's position in fixture["genes"] is its 0-based vocab line."""
    cells = []
    for row in fixture["expression"]:
        expressed = [(i, fixture["genes"][i], float(v)) for i, v in enumerate(row) if v]
        cells.append(sorted(expressed, key=lambda e: (-e[2], e[0])))
    return cells


def build_request(fmt: str, columns: dict, fixture: dict) -> dict:
    """{"dataframe_records": [...], "params": {...} | None} for the fixed cells in the
    declared format. Shared format: the fixture's own request (one record holding all cells,
    with the fixture's default params). Own formats: one record per cell, no params.
    Token IDs are 0-based vocab.txt line numbers ranked by expression; this tests that the
    endpoint returns valid embeddings, not that the client-side tokenization is TEDDY's."""
    if fmt == SHARED_FORMAT:
        return {"dataframe_records": fixture["dataframe_records"], "params": fixture["params"]}
    cells = fixture_cells(fixture)
    if fmt == GENES_AND_VALUES:
        cast = int if columns["integer_values"] else float
        records = [{columns["genes"]: [g for _, g, _ in cell], columns["values"]: [cast(v) for _, _, v in cell]}
                   for cell in cells]
    elif fmt == TOKEN_IDS:
        records = [{columns["token_ids"]: [i for i, _, _ in cell]} for cell in cells]
    else:
        raise ValueError(f"no request builder for input format {fmt!r}")
    return {"dataframe_records": records, "params": None}


_PARAM_CASTS = {"long": int, "integer": int, "float": float, "double": float, "string": str,
                "boolean": lambda v: str(v).lower() == "true" if isinstance(v, str) else bool(v)}


def declared_params(params: dict | None, declared: list | None) -> dict | None:
    """The request's params limited to those the signature declares (ParamSchema.to_dict():
    [{"name", "type", "default", "shape"}]), each cast to its declared type; None if none
    remain. A param the model does not declare would be ignored by the scoring server."""
    types = {d["name"]: d.get("type") for d in declared or []}
    kept = {k: _PARAM_CASTS.get(types[k], lambda v: v)(v) for k, v in (params or {}).items() if k in types}
    return kept or None


def format_checks(fmt: str | None, inputs: list | None) -> dict:
    """accepts_expression_input and the shared-format convention check, from the declared format."""
    declared = [f"{c.get('name')}: {_array_type(c) and f'array<{_array_type(c)}>' or c.get('type')}"
                for c in inputs or []]
    if fmt is None:
        reason = (f"declared inputs {declared} not recognised: check by hand" if inputs
                  else "no input signature on the served model version: check by hand")
        return {f"{PREFIX}accepts_expression_input": (None, reason), SHARED_FORMAT_CHECK: (None, reason)}
    expression = fmt != TOKEN_IDS
    return {
        f"{PREFIX}accepts_expression_input": (
            expression, f"{fmt} {declared}" + ("" if expression else ": tokenization left to the client")),
        SHARED_FORMAT_CHECK: (fmt == SHARED_FORMAT, f"{fmt} {declared}"),
    }


# ---- result validation (pure, unit-tested) ------------------------------------

def _get(obj, *names):
    """Attribute (SDK object) or key (dict) lookup along a path; None if missing."""
    for name in names:
        if obj is None:
            return None
        obj = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return obj


def _value(enum_or_str):
    return getattr(enum_or_str, "value", enum_or_str)


def check_endpoint(endpoint, uc_model: str, uc_model_version) -> dict:
    """ServingEndpointDetailed -> endpoint_ready, serves_recorded_version. Reads
    config.served_entities (entity_name / entity_version) and, for endpoints created with the
    legacy field, config.served_models (model_name / model_version)."""
    ready = _value(_get(endpoint, "state", "ready"))
    served = [(_get(e, "entity_name"), str(_get(e, "entity_version")))
              for e in _get(endpoint, "config", "served_entities") or []]
    served += [(_get(m, "model_name"), str(_get(m, "model_version")))
               for m in _get(endpoint, "config", "served_models") or []]
    want = (uc_model, str(uc_model_version))
    return {
        f"{PREFIX}endpoint_ready": (ready == "READY", f"state.ready={ready}"),
        f"{PREFIX}serves_recorded_version": (want in served, f"serves {served}, recorded {want}"),
    }


def embedding_key(item: dict) -> str | None:
    """The embedding field of one prediction: "embedding" if present, else the first field
    whose name contains "emb" and whose value is a list (e.g. TEDDY's own "cell_emb")."""
    if "embedding" in item:
        return "embedding"
    return next((k for k, v in item.items() if "emb" in k.lower() and isinstance(v, list)), None)


def embeddings_from(predictions) -> list:
    """Embedding vectors from a response's `predictions`: [{"embedding": [...]}, ...],
    [{"cell_emb": [...], ...}, ...] or bare vectors."""
    out = []
    for item in predictions or []:
        if isinstance(item, dict):
            key = embedding_key(item)
            item = item.get(key) if key else None
        out.append(item)
    return out


def check_embeddings(predictions, n_cells: int, dim: int) -> dict:
    """returns_valid_embeddings: one finite, non-zero vector of dimension `dim` per cell."""
    vectors = embeddings_from(predictions)
    per_cell = len(vectors) == n_cells
    finite = bool(vectors) and all(
        isinstance(v, list) and v and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
        and any(x != 0 for x in v)
        for v in vectors
    )
    dims = sorted({len(v) for v in vectors if isinstance(v, list)})
    sized = bool(vectors) and dims == [dim]
    keys = sorted({embedding_key(p) for p in predictions or [] if isinstance(p, dict)} - {None})
    detail = (f"{len(vectors)} embeddings for {n_cells} cells; "
              f"{'all finite, none all-zero' if finite else 'non-finite, empty or all-zero'}; "
              f"dimensions {dims}, expected {dim}" + (f"; field {keys}" if keys else ""))
    return {f"{PREFIX}returns_valid_embeddings": (per_cell and finite and sized, detail)}


def check_index(vs_endpoint, index) -> dict:
    state = _value(_get(vs_endpoint, "endpoint_status", "state"))
    ready = _get(index, "status", "ready") is True
    rows = _get(index, "status", "indexed_row_count")
    return {
        f"{PREFIX}vs_endpoint_online": (state == "ONLINE", f"endpoint state={state}"),
        f"{PREFIX}index_ready": (ready, f"ready={ready}, indexed rows={rows}"),
    }


def check_nn_response(response) -> dict:
    """nn_returns_cell_type: at least one row, and a cell-type column among the returned columns."""
    columns = [_get(c, "name") for c in _get(response, "manifest", "columns") or []]
    rows = _get(response, "result", "data_array") or []
    has_type = any(CELL_TYPE_COLUMN.search(c or "") for c in columns)
    return {f"{PREFIX}nn_returns_cell_type": (len(rows) > 0 and has_type, f"{len(rows)} rows, columns {columns}")}


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else float("nan")


def index_query_columns(index, scan) -> list:
    """Columns to request from the index — only columns the index holds: the fields of one
    scanned row (`scan_index`), else the spec's columns_to_sync, else the primary key alone;
    embedding columns dropped, the primary key first. (The source table is NOT used: it can
    hold columns that were never synced, and asking for one fails the whole query.)"""
    spec = _get(index, "delta_sync_index_spec") or _get(index, "direct_access_index_spec")
    embedding_cols = {_get(c, "name") for c in _get(spec, "embedding_vector_columns") or []}
    rows = _get(scan, "data") or []
    held = [_get(f, "key") for f in _get(rows[0], "fields") or []] if rows else []
    columns = held or list(_get(spec, "columns_to_sync") or [])
    primary = _get(index, "primary_key")
    ordered = [primary] + [c for c in columns if c != primary] if primary else list(columns)
    return [c for c in ordered if c and c not in embedding_cols]


def endpoint_config(endpoint) -> dict:
    """Serving settings of each served entity and the AI Gateway logging flags."""
    gateway = _get(endpoint, "ai_gateway")
    return {
        "served_entities": [
            {"name": _get(e, "entity_name"), "version": str(_get(e, "entity_version")),
             "workload_type": _value(_get(e, "workload_type")), "workload_size": _get(e, "workload_size"),
             "scale_to_zero": _get(e, "scale_to_zero_enabled")}
            for e in _get(endpoint, "config", "served_entities") or []],
        "ai_gateway_inference_table": bool(_get(gateway, "inference_table_config", "enabled")),
        "ai_gateway_usage_tracking": bool(_get(gateway, "usage_tracking_config", "enabled")),
    }


def _failed(names, detail) -> dict:
    return {f"{PREFIX}{n}": (False, detail) for n in names}


# ---- live checks (SDK calls; tested with a fake client) --------------------------

def _query(w, name: str, request: dict) -> dict:
    """POST the request to the endpoint's invocations URL with inference controls under the
    body key "params" — the key the MLflow scoring server reads. (`serving_endpoints.query`
    sends them as "extra_params", which a custom model endpoint silently drops; verified
    2026-10-27.)"""
    body = {"dataframe_records": request["dataframe_records"]}
    if request.get("params"):
        body["params"] = request["params"]
    return w.api_client.do("POST", f"/serving-endpoints/{name}/invocations", body=body)


def run_outcome_checks(w, resources: dict, checks: list, fixture: dict | None = None,
                       reference_endpoint: str | None = None, family: str = family_settings.DEFAULT_FAMILY,
                       signature: dict | None = None) -> dict:
    """{"<prefix>/<check>": (passed | None, detail)} for the listed check names.

    `w` is a databricks.sdk.WorkspaceClient. `signature` is the served model version's
    signature as {"inputs": Schema.to_dict()} (load_signature()); the request is built in
    the format it declares. Every SDK call is wrapped so one failing resource fails its own
    checks instead of stopping the rest.
    """
    fixture = fixture or load_fixture(family)
    want = set(checks)
    out = {}
    name = resources.get("serving_endpoint")
    try:
        out.update(check_endpoint(w.serving_endpoints.get(name), resources.get("uc_model"),
                                  resources.get("uc_model_version")))
    except Exception as err:  # noqa: BLE001 — record any SDK error as a failed check
        out.update(_failed(["endpoint_ready", "serves_recorded_version"], f"get endpoint {name!r}: {err}"))

    inputs = (signature or {}).get("inputs")
    fmt, columns = input_format(inputs)
    out.update(format_checks(fmt, inputs))
    vectors, request = [], None
    if fmt is None:
        out[f"{PREFIX}returns_valid_embeddings"] = (None, out[f"{PREFIX}accepts_expression_input"][1])
    else:
        request = build_request(fmt, columns, fixture)
        request["params"] = declared_params(request["params"], (signature or {}).get("params"))
        try:
            predictions = _get(_query(w, name, request), "predictions")
            vectors = embeddings_from(predictions)
            out.update(check_embeddings(predictions, fixture["expected"]["n_cells"],
                                        fixture["expected"]["embedding_dim"]))
        except Exception as err:  # noqa: BLE001
            out.update(_failed(["returns_valid_embeddings"], f"query {name!r} ({fmt}): {err}"))

    if want & {f"{PREFIX}{c}" for c in ("vs_endpoint_online", "index_ready", "nn_returns_cell_type")}:
        index_name = resources.get("vector_search_index")
        try:
            vs_endpoint = w.vector_search_endpoints.get_endpoint(resources.get("vector_search_endpoint"))
            index = w.vector_search_indexes.get_index(index_name)
            out.update(check_index(vs_endpoint, index))
        except Exception as err:  # noqa: BLE001
            index = None
            out.update(_failed(["vs_endpoint_online", "index_ready"], f"get index {index_name!r}: {err}"))
        try:
            if not vectors or not isinstance(vectors[0], list):
                raise ValueError("no fixture embedding to query with")
            columns = index_query_columns(index, w.vector_search_indexes.scan_index(index_name, num_results=1))
            response = w.vector_search_indexes.query_index(index_name, columns, query_vector=vectors[0],
                                                           num_results=5)
            out.update(check_nn_response(response))
        except Exception as err:  # noqa: BLE001
            out.update(_failed(["nn_returns_cell_type"], f"query index: {err}"))

    if reference_endpoint and vectors:
        try:
            # the reference endpoint is assumed to take the same declared format
            ref = embeddings_from(_get(_query(w, reference_endpoint, request), "predictions"))
            values = [round(cosine(a, b), 4) for a, b in zip(vectors, ref)]
            out[f"{PREFIX}reference_cosine"] = (None, f"per-cell cosine vs {reference_endpoint}: {values}")
        except Exception as err:  # noqa: BLE001
            out[f"{PREFIX}reference_cosine"] = (None, f"reference query failed: {err}")

    missing = want - set(out)
    if missing:
        raise ValueError(f"no result for checks {sorted(missing)}")
    return {k: v for k, v in out.items() if k in want or (v[0] is None and k == f"{PREFIX}reference_cosine")}


def load_signature(uc_model: str, version, profile: str | None = None) -> dict | None:
    """{"inputs": [...], "params": [...] | None} of a Unity Catalog model version's MLflow
    signature, or None if it has none. Reads the version's MLmodel with mlflow."""
    import mlflow
    from mlflow.models import get_model_info

    mlflow.set_registry_uri(f"databricks-uc://{profile}" if profile else "databricks-uc")
    signature = get_model_info(f"models:/{uc_model}/{version}").signature
    if signature is None:
        return None
    return {"inputs": signature.inputs.to_dict() if signature.inputs else None,
            "params": signature.params.to_dict() if signature.params else None}


def outcome_scores(record: dict) -> dict:
    """{"<prefix>/<check>": bool} from a run record's "outcome" (outcome/ and convention/
    checks; informational and not-measured entries dropped)."""
    outcome = record.get("outcome")
    if not outcome:
        raise ValueError(f"{record.get('run_id')}/{record.get('task_id')}: no outcome recorded — run check_outcome.py")
    return {k: v["passed"] for k, v in outcome.items() if v["passed"] is not None}


# ---- CLI ----------------------------------------------------------------------------

def main(argv=None) -> int:
    from install_skill_for_run import load_records  # noqa: E402 — same folder

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--family", default=family_settings.DEFAULT_FAMILY)
    parser.add_argument("--round", help="runs_<date> folder (default: family.json current_round)")
    parser.add_argument("--records", help="default: models/<family>/<round>/run_records.json")
    parser.add_argument("--catalog", help="real catalog name, substituted for <catalog> in the recorded names")
    parser.add_argument("--reference-endpoint", help="optional: report cosine similarity against this endpoint")
    args = parser.parse_args(argv)
    args.records = args.records or str(family_settings.round_dir(args.family, args.round) / "run_records.json")

    settings = family_settings.load_family(args.family)
    checks = family_settings.outcome_checks(settings, args.task_id) + family_settings.convention_checks(settings)
    records = load_records(Path(args.records))
    index = max(i for i, r in enumerate(records) if r["run_id"] == args.run_id and r["task_id"] == args.task_id)
    resources = records[index].get("resources_created") or {}
    if not resources.get("serving_endpoint") or not resources.get("uc_model_version"):
        parser.error("fill resources_created (incl. uc_model_version) in the run record first (README step 8)")
    resources = fill_catalog(resources, args.catalog)

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile)
    signature = load_signature(resources["uc_model"], resources["uc_model_version"], args.profile)
    results = run_outcome_checks(w, resources, checks, reference_endpoint=args.reference_endpoint,
                                 family=args.family, signature=signature)
    records[index]["outcome"] = {k: {"passed": ok, "detail": detail} for k, (ok, detail) in results.items()}
    records[index]["endpoint_config"] = endpoint_config(w.serving_endpoints.get(resources["serving_endpoint"]))
    records[index]["outcome_checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(args.records, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")
    for k, (ok, detail) in results.items():
        print(f"{'PASS' if ok else 'n/a ' if ok is None else 'FAIL'}  {k}: {detail}")
    return 0 if all(ok is not False for ok, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
