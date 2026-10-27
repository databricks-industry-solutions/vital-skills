"""Offline tests for eval/check_outcome.py: request building and result validation.
No workspace: the SDK client is a fake object."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

import check_outcome as co  # noqa: E402

import family_settings as fs  # noqa: E402

SETTINGS = fs.load_family("teddy")
CHECKS = {tid: fs.outcome_checks(SETTINGS, tid) for tid in ("teddy-001", "teddy-002")}


def test_request_matches_pandas_split_format_and_contract():
    record = co.build_teddy_request(["ENSG00000000003", "ENSG00000000005"], [[1, 0], [2, 3]], ["c0", "c1"])
    assert set(record) == {"adata_sparsematrix", "adata_obs", "adata_var"}  # teddy.md:23-29
    assert record["adata_sparsematrix"] == [[1.0, 0.0], [2.0, 3.0]]
    # exactly what pandas DataFrame.to_json(orient="split") writes for these frames
    assert record["adata_obs"] == '{"columns":["cell_id"],"index":[0,1],"data":[["c0"],["c1"]]}'
    assert record["adata_var"] == '{"columns":["index"],"index":[0,1],"data":[["ENSG00000000003"],["ENSG00000000005"]]}'
    with pytest.raises(ValueError):
        co.build_teddy_request(["g"], [[1, 2]], ["c0"])


def test_fixture_is_built_from_its_own_inputs():
    fixture = co.load_fixture()
    assert fixture["dataframe_records"] == [
        co.build_teddy_request(fixture["genes"], fixture["expression"], fixture["cell_ids"])]
    assert len(fixture["genes"]) == 100 and all(re.fullmatch(r"ENSG\d{11}", g) for g in fixture["genes"])
    assert fixture["expected"] == {**fixture["expected"], "n_cells": 3, "embedding_dim": 512}
    assert fixture["source"]["config_d_model"] == 512
    assert fixture["params"] == {"max_seq_len": "2048", "pooling": "mean"}   # defaults, cast per signature


def _endpoint(ready="READY", name="cat.skills.teddy_70m", version="2"):
    return NS(state=NS(ready=NS(value=ready)),
              config=NS(served_entities=[NS(entity_name=name, entity_version=version)]))


def test_check_endpoint():
    ok = co.check_endpoint(_endpoint(), "cat.skills.teddy_70m", 2)
    assert ok["outcome/endpoint_ready"][0] and ok["outcome/serves_recorded_version"][0]
    bad = co.check_endpoint(_endpoint(ready="NOT_READY", version="1"), "cat.skills.teddy_70m", "2")
    assert not bad["outcome/endpoint_ready"][0] and not bad["outcome/serves_recorded_version"][0]


VALID = "outcome/returns_valid_embeddings"


def _valid(predictions, n=3, dim=512):
    return co.check_embeddings(predictions, n, dim)[VALID][0]


def test_check_embeddings():
    good = [{"embedding": [0.1] * 512} for _ in range(3)]
    assert _valid(good)
    assert not _valid(good[:2])                                        # one per cell
    assert not _valid([{"embedding": [0.1] * 256}] * 3)                # dimension
    assert not _valid(good[:2] + [{"embedding": [0.1] * 768}])         # mixed dimensions
    assert not _valid([{"embedding": [math.nan] * 512}] * 3)           # finite
    assert not _valid([{"embedding": [0.0] * 512}] * 3)                # not all-zero
    assert not _valid(None)
    # TEDDY's own output field (the baseline endpoints): cell_emb next to other fields
    own = [{"cell_emb": [0.1] * 512, "cell_emb_mean": [0.2] * 512, "n_tokens": 85} for _ in range(3)]
    ok, detail = co.check_embeddings(own, 3, 512)[VALID]
    assert ok and "cell_emb" in detail
    assert co.embedding_key({"n_tokens": 3, "cell_emb_first": [1.0]}) == "cell_emb_first"
    assert co.embedding_key({"n_tokens": 3}) is None


# Declared input signatures of the 2026-10-27 endpoints (Schema.to_dict(), from the
# recorded schema-enforcement errors in runs_20261026/run_records.json).
SHARED_SIG = [{"type": "array", "items": {"type": "array", "items": {"type": "double"}}, "name": "adata_sparsematrix",
               "required": True}, {"type": "string", "name": "adata_obs", "required": True},
              {"type": "string", "name": "adata_var", "required": True}]
GENES_COUNTS_SIG = [{"type": "array", "items": {"type": "string"}, "name": "gene_ids", "required": True},
                    {"type": "array", "items": {"type": "double"}, "name": "counts", "required": True}]
TOKEN_SIG = [{"type": "array", "items": {"type": "long"}, "name": "gene_ids", "required": True}]


def test_input_format_recognises_the_three_declared_formats():
    assert co.input_format(SHARED_SIG)[0] == co.SHARED_FORMAT
    fmt, cols = co.input_format(GENES_COUNTS_SIG)
    assert fmt == co.GENES_AND_VALUES and cols == {"genes": "gene_ids", "values": "counts", "integer_values": False}
    assert co.input_format(TOKEN_SIG) == (co.TOKEN_IDS, {"token_ids": "gene_ids"})
    assert co.input_format(None) == (None, {})
    extra_required = GENES_COUNTS_SIG + [{"type": "string", "name": "species", "required": True}]
    assert co.input_format(extra_required) == (None, {})
    optional = GENES_COUNTS_SIG + [{"type": "string", "name": "species", "required": False}]
    assert co.input_format(optional)[0] == co.GENES_AND_VALUES


def test_requests_are_built_in_the_declared_format_with_mixed_lengths():
    fixture = co.load_fixture()
    shared = co.build_request(co.SHARED_FORMAT, {}, fixture)
    assert shared == {"dataframe_records": fixture["dataframe_records"], "params": fixture["params"]}

    lengths = [sum(1 for v in row if v) for row in fixture["expression"]]
    assert len(set(lengths)) > 1              # the fixed cells differ in length (padding is exercised)
    own = co.build_request(*co.input_format(GENES_COUNTS_SIG), fixture)
    assert own["params"] is None and [len(r["gene_ids"]) for r in own["dataframe_records"]] == lengths
    first = own["dataframe_records"][0]
    assert first["counts"] == sorted(first["counts"], reverse=True) and 0.0 not in first["counts"]
    assert all(g in fixture["genes"] for g in first["gene_ids"])

    tokens = co.build_request(*co.input_format(TOKEN_SIG), fixture)
    ids = tokens["dataframe_records"][0]["gene_ids"]
    assert [fixture["genes"][i] for i in ids] == first["gene_ids"]   # same genes, as vocab positions
    integer_counts = [GENES_COUNTS_SIG[0], {**GENES_COUNTS_SIG[1], "items": {"type": "long"}}]
    assert all(isinstance(v, int) for v in co.build_request(*co.input_format(integer_counts), fixture)
               ["dataframe_records"][0]["counts"])


def test_format_checks_keep_the_convention_apart():
    def checks(sig):
        return {k: v[0] for k, v in co.format_checks(co.input_format(sig)[0], sig).items()}
    assert checks(SHARED_SIG) == {"outcome/accepts_expression_input": True, co.SHARED_FORMAT_CHECK: True}
    assert checks(GENES_COUNTS_SIG) == {"outcome/accepts_expression_input": True, co.SHARED_FORMAT_CHECK: False}
    assert checks(TOKEN_SIG) == {"outcome/accepts_expression_input": False, co.SHARED_FORMAT_CHECK: False}
    assert checks(None) == {"outcome/accepts_expression_input": None, co.SHARED_FORMAT_CHECK: None}


def test_index_and_nn_checks():
    assert all(ok for ok, _ in co.check_index(NS(endpoint_status=NS(state=NS(value="ONLINE"))),
                                               NS(status=NS(ready=True, indexed_row_count=10))).values())
    assert not co.check_index(NS(endpoint_status=NS(state=NS(value="PROVISIONING"))),
                              NS(status=NS(ready=False)))["outcome/index_ready"][0]
    nn = "outcome/nn_returns_cell_type"
    response = NS(manifest=NS(columns=[NS(name="cell_id"), NS(name="cell_type"), NS(name="score")]),
                  result=NS(data_array=[["c1", "T cell", 0.9]]))
    assert co.check_nn_response(response)[nn][0]
    no_rows = NS(manifest=NS(columns=[NS(name="cell_id"), NS(name="cell_type")]), result=NS(data_array=[]))
    assert not co.check_nn_response(no_rows)[nn][0]
    no_type = NS(manifest=NS(columns=[NS(name="cell_id")]), result=NS(data_array=[["c1", 0.9]]))
    assert not co.check_nn_response(no_type)[nn][0]


def _scan(*keys):
    return NS(data=[NS(fields=[NS(key=k, value=NS(string_value="x")) for k in keys])])


def test_index_query_columns_are_the_columns_the_index_holds():
    index = NS(primary_key="cell_id", delta_sync_index_spec=NS(
        columns_to_sync=None, embedding_vector_columns=[NS(name="embedding")]))
    assert co.index_query_columns(index, _scan("cell_type", "embedding", "cell_id")) == ["cell_id", "cell_type"]
    synced = NS(primary_key="cell_id", delta_sync_index_spec=NS(
        columns_to_sync=["cell_id", "cell_type", "tissue_general"], embedding_vector_columns=[NS(name="embedding")]))
    assert co.index_query_columns(synced, NS(data=[])) == ["cell_id", "cell_type", "tissue_general"]
    assert co.index_query_columns(NS(primary_key="cell_id", delta_sync_index_spec=None), None) == ["cell_id"]


def test_with_skill_teddy_002_index_no_longer_asks_for_unsynced_table_columns():
    """runs_20261026 with_skill/teddy-002: the table had dataset_id, n_genes_used, census_version,
    model_version and model_name, the index synced only five columns, and the old check asked
    for the table's columns, so the query failed. Only the index's own columns are asked for now."""
    synced = ["cell_id", "cell_type", "cell_type_ontology_term_id", "tissue_general", "disease"]
    w = FakeWorkspace(columns_to_sync=None, index_holds=synced + ["embedding"])
    results = co.run_outcome_checks(w, RESOURCES, CHECKS["teddy-002"], signature={"inputs": SHARED_SIG})
    assert results["outcome/nn_returns_cell_type"][0]
    (asked,) = [c[2] for c in w.calls if c[0] == "query_index"]
    assert asked == synced


def test_endpoint_config():
    endpoint = NS(config=NS(served_entities=[NS(entity_name="cat.s.m", entity_version="1",
                                                workload_type=NS(value="GPU_SMALL"), workload_size="Small",
                                                scale_to_zero_enabled=True)]),
                  ai_gateway=NS(inference_table_config=NS(enabled=True), usage_tracking_config=NS(enabled=True)))
    assert co.endpoint_config(endpoint) == {
        "served_entities": [{"name": "cat.s.m", "version": "1", "workload_type": "GPU_SMALL",
                             "workload_size": "Small", "scale_to_zero": True}],
        "ai_gateway_inference_table": True, "ai_gateway_usage_tracking": True}
    no_gateway = co.endpoint_config(NS(config=NS(served_entities=[]), ai_gateway=None))
    assert not no_gateway["ai_gateway_inference_table"] and not no_gateway["ai_gateway_usage_tracking"]


class FakeWorkspace:
    """Stands in for databricks.sdk.WorkspaceClient; records calls, never touches a workspace."""

    def __init__(self, dim=512, fail_endpoint=False, columns_to_sync=("cell_id", "cell_type"),
                 index_holds=("cell_id", "cell_type", "embedding"), field="embedding"):
        self.calls = []

        def do(method, path, body=None, **kw):
            assert method == "POST" and path.startswith("/serving-endpoints/") and path.endswith("/invocations")
            name = path[len("/serving-endpoints/"):-len("/invocations")]
            self.calls.append(("query", name, body))
            n = len(body["dataframe_records"]) if "adata_obs" not in body["dataframe_records"][0] else 3
            return {"predictions": [{field: [0.01 * (i + 1)] * dim} for i in range(n)]}

        def query(*a, **kw):
            pytest.fail("serving_endpoints.query sends extra_params, which the scoring server drops")

        def get(name):
            if fail_endpoint:
                raise RuntimeError("RESOURCE_DOES_NOT_EXIST")
            return _endpoint()

        self.serving_endpoints = NS(get=get, query=query)
        self.api_client = NS(do=do)
        self.vector_search_endpoints = NS(get_endpoint=lambda n: NS(endpoint_status=NS(state=NS(value="ONLINE"))))
        index = NS(primary_key="cell_id", status=NS(ready=True, indexed_row_count=5), delta_sync_index_spec=NS(
            columns_to_sync=list(columns_to_sync) if columns_to_sync else None,
            embedding_vector_columns=[NS(name="embedding")]))

        def query_index(n, columns, **kw):
            self.calls.append(("query_index", n, columns))
            if set(columns) - set(index_holds):
                raise RuntimeError(f"Requested columns to fetch are not present in index: "
                                   f"{','.join(sorted(set(columns) - set(index_holds)))}")
            return NS(manifest=NS(columns=[NS(name=c) for c in columns] + [NS(name="score")]),
                      result=NS(data_array=[["c9", "B cell", 0.8]]))

        self.vector_search_indexes = NS(get_index=lambda n: index, query_index=query_index,
                                        scan_index=lambda n, **kw: _scan(*index_holds))
        self.tables = NS(get=lambda n: pytest.fail("the source table must not decide the query columns"))


RESOURCES = {"serving_endpoint": "teddy-70m-with_skill", "uc_model": "cat.skills.teddy_70m", "uc_model_version": "2",
             "delta_table": "cat.skills.teddy_cells", "vector_search_endpoint": "vs", "vector_search_index": "cat.skills.idx"}


ALL = {tid: checks + fs.convention_checks(SETTINGS) for tid, checks in CHECKS.items()}


@pytest.mark.parametrize("task_id", ["teddy-001", "teddy-002"])
def test_run_outcome_checks_all_pass_on_a_working_deployment(task_id):
    w = FakeWorkspace()
    results = co.run_outcome_checks(w, RESOURCES, ALL[task_id], signature={"inputs": SHARED_SIG, "params": PARAMS_SIG})
    assert set(results) == set(ALL[task_id]) and all(ok for ok, _ in results.values())
    query = next(c for c in w.calls if c[0] == "query")
    fixture = co.load_fixture()
    # inference controls go under the body key "params" (read by the scoring server), cast per signature
    assert query[2] == {"dataframe_records": fixture["dataframe_records"],
                        "params": {"max_seq_len": 2048, "pooling": "mean"}}


PARAMS_SIG = [{"name": "max_seq_len", "type": "long", "default": 2048, "shape": None},
              {"name": "pooling", "type": "string", "default": "mean", "shape": None}]


def test_declared_params():
    assert co.declared_params({"max_seq_len": "2048", "pooling": "mean"}, PARAMS_SIG) == \
        {"max_seq_len": 2048, "pooling": "mean"}
    assert co.declared_params({"max_seq_len": "2048", "pooling": "mean"}, PARAMS_SIG[1:]) == {"pooling": "mean"}
    assert co.declared_params({"max_seq_len": "2048"}, None) is None    # model declares no params: send none
    flag = [{"name": "normalize", "type": "boolean"}]
    assert co.declared_params({"normalize": "false"}, flag) == {"normalize": False}


@pytest.mark.parametrize("sig, expression, field", [(GENES_COUNTS_SIG, True, "cell_emb"), (TOKEN_SIG, False, "cell_emb")])
def test_own_declared_format_is_not_penalised(sig, expression, field):
    """An endpoint with its own valid format (the 2026-10-27 baselines) gets valid embeddings;
    only the separate checks say whether it takes expression input / the shared format."""
    w = FakeWorkspace(field=field)
    results = co.run_outcome_checks(w, RESOURCES, ALL["teddy-002"], signature={"inputs": sig})
    assert results["outcome/returns_valid_embeddings"][0] and results["outcome/nn_returns_cell_type"][0]
    assert results["outcome/accepts_expression_input"][0] is expression
    assert results[co.SHARED_FORMAT_CHECK][0] is False
    query = next(c for c in w.calls if c[0] == "query")
    assert set(query[2]) == {"dataframe_records"} and len(query[2]["dataframe_records"]) == 3   # one record per cell


def test_no_signature_is_not_measured_and_the_scorer_refuses_it():
    w = FakeWorkspace()
    results = co.run_outcome_checks(w, RESOURCES, ALL["teddy-001"], signature=None)
    assert results["outcome/returns_valid_embeddings"][0] is None and "check by hand" in \
        results["outcome/returns_valid_embeddings"][1]
    assert not any(c[0] == "query" for c in w.calls)
    record = {"outcome": {k: {"passed": v[0], "detail": v[1]} for k, v in results.items()}}
    assert "outcome/returns_valid_embeddings" not in co.outcome_scores(record)


def test_run_outcome_checks_wrong_dim_and_missing_endpoint_fail_their_checks_only():
    results = co.run_outcome_checks(FakeWorkspace(dim=1024, fail_endpoint=True), RESOURCES,
                                    CHECKS["teddy-002"], signature={"inputs": SHARED_SIG})
    assert not results["outcome/endpoint_ready"][0] and "RESOURCE_DOES_NOT_EXIST" in results["outcome/endpoint_ready"][1]
    assert not results["outcome/returns_valid_embeddings"][0]
    assert results["outcome/index_ready"][0] and results["outcome/nn_returns_cell_type"][0]


def test_reference_cosine_is_informational():
    results = co.run_outcome_checks(FakeWorkspace(), RESOURCES, CHECKS["teddy-001"],
                                    reference_endpoint="gwb-teddy", signature={"inputs": SHARED_SIG})
    ok, detail = results["outcome/reference_cosine"]
    assert ok is None and "gwb-teddy" in detail and "1.0" in detail
    assert co.outcome_scores({"outcome": {k: {"passed": v[0], "detail": v[1]} for k, v in results.items()}}) == {
        k: v[0] for k, v in results.items() if v[0] is not None}
    with pytest.raises(ValueError, match="no outcome"):
        co.outcome_scores({"run_id": "baseline", "task_id": "teddy-001", "outcome": None})


def test_cosine():
    assert co.cosine([1, 0], [1, 0]) == 1.0 and co.cosine([1, 0], [0, 1]) == 0.0
    assert math.isnan(co.cosine([0, 0], [1, 1]))


def test_fixture_split_json_equals_pandas():
    pd = pytest.importorskip("pandas")
    fixture = co.load_fixture()
    record = fixture["dataframe_records"][0]
    assert record["adata_obs"] == pd.DataFrame({"cell_id": fixture["cell_ids"]}).to_json(orient="split")
    assert record["adata_var"] == pd.DataFrame({"index": fixture["genes"]}).to_json(orient="split")


def test_check_endpoint_reads_legacy_served_models():
    legacy = NS(state=NS(ready=NS(value="READY")),
                config=NS(served_entities=None, served_models=[NS(model_name="cat.skills.teddy_70m", model_version="2")]))
    assert co.check_endpoint(legacy, "cat.skills.teddy_70m", 2)["outcome/serves_recorded_version"][0]
    assert not co.check_endpoint(legacy, "cat.skills.teddy_70m", 3)["outcome/serves_recorded_version"][0]


def test_docstring_notebook_example_passes_check_names():
    doc = co.__doc__
    assert 'checks = outcome_checks(load_family("teddy"), "teddy-002")' in doc
    assert 'run_outcome_checks(WorkspaceClient(), resources, checks, family="teddy")' in doc


def test_family_outcome_settings():
    assert SETTINGS["outcome"]["embedding_dim"] == 512 == co.load_fixture("teddy")["expected"]["embedding_dim"]
    assert SETTINGS["outcome"]["vector_search_tasks"] == ["teddy-002"]
    assert "outcome/index_ready" in CHECKS["teddy-002"] and "outcome/index_ready" not in CHECKS["teddy-001"]


def test_fill_catalog():
    filled = co.fill_catalog({"uc_model": "<catalog>.skills_owm_202610.m", "serving_endpoint": "ep"}, "main")
    assert filled == {"uc_model": "main.skills_owm_202610.m", "serving_endpoint": "ep"}
    with pytest.raises(ValueError, match="--catalog"):
        co.fill_catalog({"uc_model": "<catalog>.s.m"}, None)
    assert co.fill_catalog({"serving_endpoint": "ep"}, None) == {"serving_endpoint": "ep"}


def test_family_settings_prompt_cli(capsys):
    assert fs.main(["prompt", "--run-id", "baseline", "--task-id", "teddy-002"]) == 0
    out = capsys.readouterr().out
    assert "owm_teddy_ep_002_baseline" in out and "{run}" not in out
    assert fs.main(["names", "--run-id", "old_prompt_baseline", "--task-id", "teddy-001"]) == 0
    assert json.loads(capsys.readouterr().out)["serving_endpoint"] == "owm_teddy_ep_001_old_baseline"
    assert fs.round_dir("teddy") == EVAL_DIR / "models" / "teddy" / "runs_20261026"
