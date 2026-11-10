"""scripts/nn_annotate.py on made-up data and a mocked WorkspaceClient."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nn_annotate as nn  # noqa: E402


def _types(counts):
    return [t for t, n in counts.items() for _ in range(n)]


# ---- request format and endpoint checks ------------------------------------------------------------
def test_invoke_puts_controls_under_params_not_extra_params():
    w = mock.MagicMock()
    nn.invoke(w, "ep", [{"a": 1}], params={"max_seq_len": "2048"})
    method, path = w.api_client.do.call_args.args
    body = w.api_client.do.call_args.kwargs["body"]
    assert (method, path) == ("POST", "/serving-endpoints/ep/invocations")
    assert body == {"dataframe_records": [{"a": 1}], "params": {"max_seq_len": "2048"}}
    w.serving_endpoints.query.assert_not_called()


def _endpoint(entities, routes=None):
    served = [NS(name=n, entity_name=m, entity_version=v, workload_type=NS(value=t)) for n, m, v, t in entities]
    traffic = NS(routes=[NS(served_entity_name=n, served_model_name=None, traffic_percentage=p) for n, p in routes]) \
        if routes is not None else None
    return NS(config=NS(served_entities=served, served_models=[], traffic_config=traffic))


def test_serving_endpoint_check_requires_one_version_with_all_traffic():
    w = mock.MagicMock()
    w.serving_endpoints.get.return_value = _endpoint([("s", "C.S.M", "2", "GPU_SMALL")], [("s", 100)])
    assert nn.serving_endpoint_check(w, "ep", uc_model="c.s.m", uc_model_version=2)[1] == "GPU_SMALL"
    with pytest.raises(RuntimeError, match="not the approved"):
        nn.serving_endpoint_check(w, "ep", uc_model="c.s.m", uc_model_version=3)
    w.serving_endpoints.get.return_value = _endpoint([("s", "c.s.m", "2", "GPU_SMALL")], [("s", 50), ("t", 50)])
    with pytest.raises(RuntimeError, match="routes traffic"):
        nn.serving_endpoint_check(w, "ep", uc_model="c.s.m", uc_model_version=2)
    w.serving_endpoints.get.return_value = _endpoint([("s", "c.s.m", "2", "x"), ("t", "c.s.m", "3", "x")])
    with pytest.raises(RuntimeError, match="serves 2 entities"):
        nn.serving_endpoint_check(w, "ep", uc_model="c.s.m", uc_model_version=2)


DATA_COLUMNS = {"adata_sparsematrix", "adata_obs", "adata_var"}


class FakeEndpoint:
    """Embedding depends on how many expressed genes max_seq_len lets through; can fail on big batches.

    Controls arrive as optional record columns or under the body "params" key; the column wins
    (integration-contract.md §Standard). echo=True returns the settings used in every prediction.
    """

    def __init__(self, fail_over=None, wrong_id=False, echo=False, force=None):
        self.fail_over, self.wrong_id, self.echo, self.force, self.calls = fail_over, wrong_id, echo, force, []

    def do(self, method, path, body):
        rec = body["dataframe_records"][0]
        rows = np.array(rec["adata_sparsematrix"])
        ids = list(pd.read_json(__import__("io").StringIO(rec["adata_obs"]), orient="split")["cell_id"])
        used = {"max_seq_len": "2048", **(body.get("params") or {}),
                **{k: v for k, v in rec.items() if k not in DATA_COLUMNS}, **(self.force or {})}
        self.calls.append((len(ids), used))
        if self.fail_over and len(ids) > self.fail_over:
            raise RuntimeError("CUDA out of memory")
        cap = int(used["max_seq_len"])
        preds = [{"embedding": [float(min((r > 0).sum(), cap)), 50.0 - min((r > 0).sum(), cap), 1.0]} for r in rows]
        if self.echo:
            for p in preds:
                p.update(used)
        if self.wrong_id:
            preds[0]["cell_id"] = "someone-else"
        return {"predictions": preds}


def test_embed_via_endpoint_batches_and_retries_one_cell_at_a_time():
    fake = FakeEndpoint(fail_over=3)
    w = NS(api_client=fake)
    X = np.ones((7, 5))
    emb = nn.embed_via_endpoint(w, "ep", X, [f"c{i}" for i in range(7)], [f"ENSG{i}" for i in range(5)],
                                params={"max_seq_len": "2048"}, tier="GPU_SMALL", batch_size=5)
    assert emb.shape == (7, 3)
    assert [n for n, _ in fake.calls] == [5, 1, 1, 1, 1, 1, 2]
    with pytest.raises(ValueError, match="exceeds"):
        nn.embed_via_endpoint(w, "ep", X, list("abcdefg"), list("abcde"), params={}, tier="GPU_SMALL", batch_size=11)


@pytest.mark.parametrize("route", ["column", "params"])
def test_embed_via_endpoint_sends_controls_by_either_route(route):
    fake = FakeEndpoint(echo=True)
    nn.embed_via_endpoint(NS(api_client=fake), "ep", np.ones((2, 40)), ["a", "b"], [f"g{i}" for i in range(40)],
                          params={"max_seq_len": 32}, route=route)
    assert fake.calls[0][1]["max_seq_len"] == "32"                # sent as a string, and it arrived
    with pytest.raises(ValueError, match="route must be one of"):
        nn.embed_via_endpoint(NS(api_client=fake), "ep", np.ones((1, 2)), ["a"], ["g0", "g1"], params={},
                              route="extra_params")


def test_embed_via_endpoint_stops_when_the_echo_shows_another_setting():
    fake = FakeEndpoint(echo=True, force={"pooling": "first"})    # the endpoint ignores the pooling sent
    with pytest.raises(nn.ControlNotApplied, match="pooling"):
        nn.embed_via_endpoint(NS(api_client=fake), "ep", np.ones((3, 2)), ["a", "b", "c"], ["g0", "g1"],
                              params={"pooling": "mean"})
    assert len(fake.calls) == 1                                   # a settings problem is not retried per cell


def test_embed_via_endpoint_rejects_a_mismatched_cell_id():
    w = NS(api_client=FakeEndpoint(wrong_id=True))
    with pytest.raises(RuntimeError, match="someone-else"):
        nn.embed_via_endpoint(w, "ep", np.ones((1, 2)), ["c0"], ["g0", "g1"], params={"max_seq_len": "2048"})


def test_check_controls_applied():
    w = NS(api_client=FakeEndpoint())
    X = np.ones((2, 40))
    cos = nn.check_controls_applied(w, "ep", X, ["a", "b"], [f"g{i}" for i in range(40)],
                                    params={"max_seq_len": "2048"})
    assert cos < 0.99

    assert nn.check_controls_applied(w, "ep", X, ["a", "b"], [f"g{i}" for i in range(40)],
                                     params={"max_seq_len": "2048"}, route="params") < 0.99

    class Ignores(FakeEndpoint):                    # what extra_params looks like: controls never arrive
        def do(self, method, path, body):
            recs = [{k: v for k, v in r.items() if k in DATA_COLUMNS} for r in body["dataframe_records"]]
            return super().do(method, path, {"dataframe_records": recs})
    for route in ("column", "params"):
        with pytest.raises(RuntimeError, match="not reaching"):
            nn.check_controls_applied(NS(api_client=Ignores()), "ep", X, ["a", "b"], [f"g{i}" for i in range(40)],
                                      params={"max_seq_len": "2048"}, route=route)


# ---- hold-out split (same cases as the workflow recipes test) ------------------------------------------
def test_split_holdout_stratified_with_rare_types_reported():
    types = _types({"T cell": 50, "B cell": 10, "NK cell": 5, "platelet": 4, "mast cell": 1})
    held, rare = nn.split_holdout(types)
    s = pd.Series(types)
    assert s[held].value_counts().to_dict() == {"T cell": 10, "B cell": 2, "NK cell": 2}
    assert dict(rare) == {"mast cell": 1, "platelet": 4}


def test_split_holdout_is_seeded():
    types = _types({"a": 30, "b": 20})
    assert (nn.split_holdout(types, seed=7)[0] == nn.split_holdout(types, seed=7)[0]).all()
    assert not (nn.split_holdout(types, seed=7)[0] == nn.split_holdout(types, seed=8)[0]).all()


# ---- neighbours and vote ------------------------------------------------------------------------------
def _w_with_hits(rows):
    w = mock.MagicMock()
    w.vector_search_indexes.query_index.return_value = NS(result=NS(data_array=rows))
    return w


def test_query_neighbours_excludes_the_query_cell_and_asks_for_one_more():
    rows = [["self", "T", 1.0]] + [[f"n{i}", "T" if i < 6 else "B", 0.9 - i / 100] for i in range(10)]
    w = _w_with_hits(rows)
    hits = nn.query_neighbours(w, "c.s.i", [0.1, 0.2], columns=["cell_id", "cell_type"], k=10, exclude_ids={"self"})
    assert len(hits) == 10 and all(h["cell_id"] != "self" for h in hits)
    assert w.vector_search_indexes.query_index.call_args.kwargs["num_results"] == 11
    v = nn.vote(hits)
    assert v["predicted"] == "T" and v["agreement"] == 0.6 and v["top1_score"] == pytest.approx(0.9)


def test_annotate_builds_one_row_per_cell():
    rows = [[f"n{i}", "T", 0.9] for i in range(10)]
    df = nn.annotate(_w_with_hits(rows), "c.s.i", [[0.0], [1.0]], columns=["cell_id", "cell_type"], ids=["a", "b"])
    assert list(df["cell_id"]) == ["a", "b"] and (df["agreement"] == 1.0).all()


# ---- calibration ---------------------------------------------------------------------------------------
def test_wilson_bounds():
    assert nn.wilson_lower(0, 0) == 0.0
    assert nn.wilson_lower(8, 10) == pytest.approx(0.4902, abs=1e-4)
    assert nn.wilson_lower(34, 34) < 0.90 <= nn.wilson_lower(35, 35)
    lo, hi = nn.wilson_interval(8, 10)
    assert lo == pytest.approx(nn.wilson_lower(8, 10)) and hi == pytest.approx(0.9433, abs=1e-4)


def test_calibrate_vote_threshold_refuses_tiny_or_weak_evidence():
    cal = nn.calibrate_vote_threshold
    assert cal([1.0], [True])[0] is None
    assert cal([1.0] * 29, [True] * 29, min_accepted=1)[0] is None
    assert cal([1.0] * 35, [True] * 35)[0] == 1.0
    assert cal([1.0] * 100, [True] * 80 + [False] * 20)[0] is None


def test_calibrate_vote_threshold_picks_lowest_supported_cutoff():
    rng = np.random.default_rng(0)
    conf = np.round(rng.uniform(0.1, 1.0, 2000), 1)
    correct = rng.uniform(size=2000) < conf
    threshold, table = nn.calibrate_vote_threshold(conf, correct)
    chosen = table[table["threshold"] == threshold].iloc[0]
    assert chosen["accepted"] >= nn.MIN_ACCEPTED and chosen["lower_bound"] >= nn.TARGET
    lower = table[table["threshold"] < threshold]
    assert not ((lower["accepted"] >= nn.MIN_ACCEPTED) & (lower["lower_bound"] >= nn.TARGET)).any()


def test_score_floor_and_flags():
    assert nn.score_floor([0.5] * 10) is None
    floor = nn.score_floor(np.linspace(0, 1, 101))
    assert floor == pytest.approx(0.05)
    df = pd.DataFrame({"agreement": [1.0, 0.5, 1.0], "top1_score": [0.9, 0.9, 0.01]})
    flags = nn.flag_cells(df, threshold=0.8, floor=floor)["flag"].tolist()
    assert flags == ["", "low confidence", "far from the reference"]
    assert nn.flag_cells(df, threshold=None, floor=None)["flag"].tolist() == ["not calibrated"] * 3


def test_calibrate_flag_rule_keeps_coverage():
    rng = np.random.default_rng(1)
    a, s = rng.uniform(size=500), rng.uniform(size=500)
    rule = nn.calibrate_flag_rule(a, s, coverage=0.95)
    flagged = (a < rule["agreement_threshold"]) | (s < rule["score_threshold"])
    assert flagged.mean() <= 0.05 and rule["heldout_flagged"] == flagged.sum()
    assert nn.calibrate_flag_rule(a[:10], s[:10]) is None


def test_compare_with_classifier_on_separable_clusters():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    centres = {"A": np.r_[1.0, 0, 0, 0], "B": np.r_[0, 1.0, 0, 0], "C": np.r_[0, 0, 1.0, 0]}
    def cells(n):
        labels = rng.choice(list(centres), size=n)
        return np.array([centres[l] + rng.normal(scale=0.05, size=4) for l in labels]), labels
    ref, ref_y = cells(150)
    held, held_y = cells(60)
    table, pred, _ = nn.compare_with_classifier(ref, ref_y, held, held_y, held_vote=held_y)
    assert list(table["method"]) == [f"neighbour vote (k={nn.K})", "classifier on frozen embeddings"]
    assert (table["accuracy"] > 0.95).all() and (table["accuracy_lo"] <= table["accuracy"]).all()
    assert len(pred) == 60
