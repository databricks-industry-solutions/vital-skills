#!/usr/bin/env python3
"""Outcome checks for one TEDDY eval run: does what the run built actually work?

Run after a run is exported and `resources_created` is filled in its record, and
BEFORE tearing anything down (see eval/README.md). From a laptop:

  python3 skills/open-weight-models/eval/check_outcome.py --profile <profile> \\
      --run-id <run_id> --task-id <task_id> [--reference-endpoint <name>]

or from a notebook (the repo's eval/ folder on sys.path):

  from check_outcome import run_outcome_checks
  results = run_outcome_checks(WorkspaceClient(), resources, "teddy-002")

Checks (metric names start with "outcome/"; which apply comes from expectations.json):
  endpoint_ready              serving endpoint state is READY
  serves_recorded_version     it serves the recorded UC model + version
  one_embedding_per_cell      the fixed request (fixtures/teddy_request.json) returns one
                              prediction per cell
  embeddings_finite           every embedding value is a finite number, no all-zero vector
  embedding_dim               every embedding has the expected dimension (512 for 70M)
  vs_endpoint_online          teddy-002: the Vector Search endpoint is ONLINE
  index_ready                 teddy-002: the index reports ready
  nn_query_returns_rows       teddy-002: a nearest-neighbour query with a fixture
                              embedding returns at least one row
  nn_has_cell_type_column     teddy-002: the returned columns include a cell-type column
Optional, informational only (no pass/fail): reference_cosine — per-cell cosine similarity
of the fixture embeddings against another endpoint (e.g. a reference TEDDY deployment).

Results are written into the run record under "outcome" as
{"outcome/<check>": {"passed": true|false|null, "detail": "..."}}; null = informational.
SDK names verified against databricks-sdk 0.157.0.
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
FIXTURE = EVAL_DIR / "fixtures" / "teddy_request.json"
DEFAULT_RECORDS = EVAL_DIR / "outputs" / "run_records.json"
PREFIX = "outcome/"
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


def load_fixture(path: Path = FIXTURE) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


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
    """ServingEndpointDetailed -> endpoint_ready, serves_recorded_version."""
    ready = _value(_get(endpoint, "state", "ready"))
    entities = _get(endpoint, "config", "served_entities") or []
    served = [(_get(e, "entity_name"), str(_get(e, "entity_version"))) for e in entities]
    want = (uc_model, str(uc_model_version))
    return {
        f"{PREFIX}endpoint_ready": (ready == "READY", f"state.ready={ready}"),
        f"{PREFIX}serves_recorded_version": (want in served, f"serves {served}, recorded {want}"),
    }


def embeddings_from(predictions) -> list:
    """Embedding vectors from the `predictions` of a TEDDY response ([{"embedding": [...]}, ...])."""
    out = []
    for item in predictions or []:
        vector = item.get("embedding") if isinstance(item, dict) else item
        out.append(vector)
    return out


def check_embeddings(predictions, n_cells: int, dim: int) -> dict:
    vectors = embeddings_from(predictions)
    per_cell = len(vectors) == n_cells
    finite = bool(vectors) and all(
        isinstance(v, list) and v and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
        and any(x != 0 for x in v)
        for v in vectors
    )
    dims = sorted({len(v) for v in vectors if isinstance(v, list)})
    return {
        f"{PREFIX}one_embedding_per_cell": (per_cell, f"{len(vectors)} embeddings for {n_cells} cells"),
        f"{PREFIX}embeddings_finite": (finite, "all finite, none all-zero" if finite else "non-finite, empty or all-zero"),
        f"{PREFIX}embedding_dim": (bool(vectors) and dims == [dim], f"dimensions {dims}, expected {dim}"),
    }


def check_index(vs_endpoint, index) -> dict:
    state = _value(_get(vs_endpoint, "endpoint_status", "state"))
    ready = _get(index, "status", "ready") is True
    rows = _get(index, "status", "indexed_row_count")
    return {
        f"{PREFIX}vs_endpoint_online": (state == "ONLINE", f"endpoint state={state}"),
        f"{PREFIX}index_ready": (ready, f"ready={ready}, indexed rows={rows}"),
    }


def check_nn_response(response) -> dict:
    columns = [_get(c, "name") for c in _get(response, "manifest", "columns") or []]
    rows = _get(response, "result", "data_array") or []
    return {
        f"{PREFIX}nn_query_returns_rows": (len(rows) > 0, f"{len(rows)} rows"),
        f"{PREFIX}nn_has_cell_type_column": (any(CELL_TYPE_COLUMN.search(c or "") for c in columns),
                                              f"columns {columns}"),
    }


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else float("nan")


def index_query_columns(index, table) -> list:
    """Columns to request from the index: columns_to_sync, else the source table's columns,
    minus embedding columns; the primary key first."""
    spec = _get(index, "delta_sync_index_spec")
    embedding_cols = {_get(c, "name") for c in _get(spec, "embedding_vector_columns") or []}
    columns = _get(spec, "columns_to_sync") or [_get(c, "name") for c in _get(table, "columns") or []]
    primary = _get(index, "primary_key")
    ordered = [primary] + [c for c in columns if c != primary] if primary else list(columns)
    return [c for c in ordered if c and c not in embedding_cols]


def _failed(names, detail) -> dict:
    return {f"{PREFIX}{n}": (False, detail) for n in names}


# ---- live checks (SDK calls; tested with a fake client) --------------------------

def run_outcome_checks(w, resources: dict, checks: list, fixture: dict | None = None,
                       reference_endpoint: str | None = None) -> dict:
    """{"outcome/<check>": (passed | None, detail)} for the listed check names.

    `w` is a databricks.sdk.WorkspaceClient. Every SDK call is wrapped so one failing
    resource fails its own checks instead of stopping the rest.
    """
    fixture = fixture or load_fixture()
    want = set(checks)
    out = {}
    name = resources.get("serving_endpoint")
    try:
        out.update(check_endpoint(w.serving_endpoints.get(name), resources.get("uc_model"),
                                  resources.get("uc_model_version")))
    except Exception as err:  # noqa: BLE001 — record any SDK error as a failed check
        out.update(_failed(["endpoint_ready", "serves_recorded_version"], f"get endpoint {name!r}: {err}"))

    vectors = []
    try:
        response = w.serving_endpoints.query(name, dataframe_records=fixture["dataframe_records"],
                                             extra_params=fixture["extra_params"])
        predictions = _get(response, "predictions")
        vectors = embeddings_from(predictions)
        out.update(check_embeddings(predictions, fixture["expected"]["n_cells"],
                                    fixture["expected"]["embedding_dim"]))
    except Exception as err:  # noqa: BLE001
        out.update(_failed(["one_embedding_per_cell", "embeddings_finite", "embedding_dim"],
                           f"query {name!r}: {err}"))

    if want & {f"{PREFIX}{c}" for c in ("vs_endpoint_online", "index_ready", "nn_query_returns_rows",
                                         "nn_has_cell_type_column")}:
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
            table = None if _get(index, "delta_sync_index_spec", "columns_to_sync") else \
                w.tables.get(resources.get("delta_table"))
            columns = index_query_columns(index, table)
            response = w.vector_search_indexes.query_index(index_name, columns, query_vector=vectors[0],
                                                           num_results=5)
            out.update(check_nn_response(response))
        except Exception as err:  # noqa: BLE001
            out.update(_failed(["nn_query_returns_rows", "nn_has_cell_type_column"], f"query index: {err}"))

    if reference_endpoint and vectors:
        try:
            ref = embeddings_from(_get(w.serving_endpoints.query(
                reference_endpoint, dataframe_records=fixture["dataframe_records"],
                extra_params=fixture["extra_params"]), "predictions"))
            values = [round(cosine(a, b), 4) for a, b in zip(vectors, ref)]
            out[f"{PREFIX}reference_cosine"] = (None, f"per-cell cosine vs {reference_endpoint}: {values}")
        except Exception as err:  # noqa: BLE001
            out[f"{PREFIX}reference_cosine"] = (None, f"reference query failed: {err}")

    missing = want - set(out)
    if missing:
        raise ValueError(f"no result for checks {sorted(missing)}")
    return {k: v for k, v in out.items() if k in want or v[0] is None}


def outcome_scores(record: dict) -> dict:
    """{"outcome/<check>": bool} from a run record's "outcome" (informational entries dropped)."""
    outcome = record.get("outcome")
    if not outcome:
        raise ValueError(f"{record.get('run_id')}/{record.get('task_id')}: no outcome recorded — run check_outcome.py")
    return {k: v["passed"] for k, v in outcome.items() if v["passed"] is not None}


# ---- CLI ----------------------------------------------------------------------------

def main(argv=None) -> int:
    sys.path.insert(0, str(EVAL_DIR))
    from install_skill_for_run import load_records  # noqa: E402 — same folder

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--reference-endpoint", help="optional: report cosine similarity against this endpoint")
    args = parser.parse_args(argv)

    with open(EVAL_DIR / "expectations.json", encoding="utf-8") as handle:
        checks = next(t["outcome_checks"] for t in json.load(handle) if t["task_id"] == args.task_id)
    records = load_records(Path(args.records))
    index = max(i for i, r in enumerate(records) if r["run_id"] == args.run_id and r["task_id"] == args.task_id)
    resources = records[index].get("resources_created") or {}
    if not resources.get("serving_endpoint"):
        parser.error("fill resources_created in the run record first (README step 7)")

    from databricks.sdk import WorkspaceClient

    results = run_outcome_checks(WorkspaceClient(profile=args.profile), resources, checks,
                                 reference_endpoint=args.reference_endpoint)
    records[index]["outcome"] = {k: {"passed": ok, "detail": detail} for k, (ok, detail) in results.items()}
    records[index]["outcome_checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(args.records, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")
    for k, (ok, detail) in results.items():
        print(f"{'PASS' if ok else 'info' if ok is None else 'FAIL'}  {k}: {detail}")
    return 0 if all(ok is not False for ok, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
