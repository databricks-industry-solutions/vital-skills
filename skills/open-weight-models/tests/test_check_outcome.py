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

EXPECTATIONS = {t["task_id"]: t for t in json.loads((EVAL_DIR / "expectations.json").read_text())}


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
    assert all(isinstance(v, str) for v in fixture["extra_params"].values())  # SDK extra_params are strings


def _endpoint(ready="READY", name="cat.skills.teddy_70m", version="2"):
    return NS(state=NS(ready=NS(value=ready)),
              config=NS(served_entities=[NS(entity_name=name, entity_version=version)]))


def test_check_endpoint():
    ok = co.check_endpoint(_endpoint(), "cat.skills.teddy_70m", 2)
    assert ok["outcome/endpoint_ready"][0] and ok["outcome/serves_recorded_version"][0]
    bad = co.check_endpoint(_endpoint(ready="NOT_READY", version="1"), "cat.skills.teddy_70m", "2")
    assert not bad["outcome/endpoint_ready"][0] and not bad["outcome/serves_recorded_version"][0]


def test_check_embeddings():
    good = [{"embedding": [0.1] * 512} for _ in range(3)]
    assert all(ok for ok, _ in co.check_embeddings(good, 3, 512).values())
    assert not co.check_embeddings(good[:2], 3, 512)["outcome/one_embedding_per_cell"][0]
    assert not co.check_embeddings([{"embedding": [0.1] * 256}] * 3, 3, 512)["outcome/embedding_dim"][0]
    mixed = good[:2] + [{"embedding": [0.1] * 768}]
    assert not co.check_embeddings(mixed, 3, 512)["outcome/embedding_dim"][0]
    assert not co.check_embeddings([{"embedding": [math.nan] * 512}] * 3, 3, 512)["outcome/embeddings_finite"][0]
    assert not co.check_embeddings([{"embedding": [0.0] * 512}] * 3, 3, 512)["outcome/embeddings_finite"][0]
    assert not any(ok for ok, _ in co.check_embeddings(None, 3, 512).values())


def test_index_and_nn_checks():
    assert all(ok for ok, _ in co.check_index(NS(endpoint_status=NS(state=NS(value="ONLINE"))),
                                               NS(status=NS(ready=True, indexed_row_count=10))).values())
    assert not co.check_index(NS(endpoint_status=NS(state=NS(value="PROVISIONING"))),
                              NS(status=NS(ready=False)))["outcome/index_ready"][0]
    response = NS(manifest=NS(columns=[NS(name="cell_id"), NS(name="cell_type"), NS(name="score")]),
                  result=NS(data_array=[["c1", "T cell", 0.9]]))
    assert all(ok for ok, _ in co.check_nn_response(response).values())
    no_type = NS(manifest=NS(columns=[NS(name="cell_id")]), result=NS(data_array=[]))
    assert not any(ok for ok, _ in co.check_nn_response(no_type).values())


def test_index_query_columns():
    index = NS(primary_key="cell_id", delta_sync_index_spec=NS(
        columns_to_sync=None, embedding_vector_columns=[NS(name="embedding")]))
    table = NS(columns=[NS(name="embedding"), NS(name="cell_type"), NS(name="cell_id")])
    assert co.index_query_columns(index, table) == ["cell_id", "cell_type"]
    synced = NS(primary_key="cell_id", delta_sync_index_spec=NS(
        columns_to_sync=["cell_id", "cell_type", "tissue_general"], embedding_vector_columns=[NS(name="embedding")]))
    assert co.index_query_columns(synced, None) == ["cell_id", "cell_type", "tissue_general"]


class FakeWorkspace:
    """Stands in for databricks.sdk.WorkspaceClient; records calls, never touches a workspace."""

    def __init__(self, dim=512, fail_endpoint=False):
        self.calls = []
        predictions = [{"embedding": [0.01 * (i + 1)] * dim} for i in range(3)]

        def query(name, **kw):
            self.calls.append(("query", name, kw))
            return NS(predictions=predictions)

        def get(name):
            if fail_endpoint:
                raise RuntimeError("RESOURCE_DOES_NOT_EXIST")
            return _endpoint()

        self.serving_endpoints = NS(get=get, query=query)
        self.vector_search_endpoints = NS(get_endpoint=lambda n: NS(endpoint_status=NS(state=NS(value="ONLINE"))))
        index = NS(primary_key="cell_id", status=NS(ready=True, indexed_row_count=5), delta_sync_index_spec=NS(
            columns_to_sync=["cell_id", "cell_type"], embedding_vector_columns=[NS(name="embedding")]))
        self.vector_search_indexes = NS(
            get_index=lambda n: index,
            query_index=lambda n, columns, **kw: NS(
                manifest=NS(columns=[NS(name=c) for c in columns] + [NS(name="score")]),
                result=NS(data_array=[["c9", "B cell", 0.8]])))
        self.tables = NS(get=lambda n: pytest.fail("columns_to_sync is set; table lookup not needed"))


RESOURCES = {"serving_endpoint": "teddy-70m-with_skill", "uc_model": "cat.skills.teddy_70m", "uc_model_version": "2",
             "delta_table": "cat.skills.teddy_cells", "vector_search_endpoint": "vs", "vector_search_index": "cat.skills.idx"}


@pytest.mark.parametrize("task_id", ["teddy-001", "teddy-002"])
def test_run_outcome_checks_all_pass_on_a_working_deployment(task_id):
    w = FakeWorkspace()
    checks = EXPECTATIONS[task_id]["outcome_checks"]
    results = co.run_outcome_checks(w, RESOURCES, checks)
    assert set(results) == set(checks) and all(ok for ok, _ in results.values())
    query = next(c for c in w.calls if c[0] == "query")
    fixture = co.load_fixture()
    assert query[2] == {"dataframe_records": fixture["dataframe_records"], "extra_params": fixture["extra_params"]}


def test_run_outcome_checks_wrong_dim_and_missing_endpoint_fail_their_checks_only():
    results = co.run_outcome_checks(FakeWorkspace(dim=1024, fail_endpoint=True), RESOURCES,
                                    EXPECTATIONS["teddy-002"]["outcome_checks"])
    assert not results["outcome/endpoint_ready"][0] and "RESOURCE_DOES_NOT_EXIST" in results["outcome/endpoint_ready"][1]
    assert not results["outcome/embedding_dim"][0]
    assert results["outcome/index_ready"][0] and results["outcome/nn_query_returns_rows"][0]


def test_reference_cosine_is_informational():
    results = co.run_outcome_checks(FakeWorkspace(), RESOURCES, EXPECTATIONS["teddy-001"]["outcome_checks"],
                                    reference_endpoint="gwb-teddy")
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
    assert 'run_outcome_checks(WorkspaceClient(), resources, checks)' in doc and '["outcome_checks"]' in doc
