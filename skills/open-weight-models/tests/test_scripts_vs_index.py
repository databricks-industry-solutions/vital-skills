"""scripts/vs_index.py on a mocked WorkspaceClient: names, guards, waits, creation gate, never deleting."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vs_index as vx  # noqa: E402
from databricks.sdk.errors import NotFound, PermissionDenied  # noqa: E402
from databricks.sdk.service.pipelines import UpdateInfoState  # noqa: E402
from databricks.sdk.service.vectorsearch import EndpointStatusState, EndpointType  # noqa: E402


def test_uc_names_compare_part_by_part():
    assert vx.uc_name_parts("`my-cat`.Sch.`ta``b`") == ("my-cat", "sch", "ta`b")
    assert vx.same_uc_name("MY-CAT.sch.t", "`my-cat`.`sch`.`t`")
    assert not vx.same_uc_name(None, "a.b.c") and not vx.same_uc_name("a.b", "a.b.c")
    assert vx.quote_uc_name("a.b-c.d") == "`a`.`b-c`.`d`"
    with pytest.raises(ValueError):
        vx.uc_name_parts("a.b.c.d")


def test_on_existing_stops_without_a_choice():
    with pytest.raises(RuntimeError, match="Ask the user"):
        vx.on_existing("Index", "a.b.c", "stop")
    with pytest.raises(ValueError):
        vx.on_existing("Index", "a.b.c", "yes")
    assert vx.on_existing("Index", "a.b.c", "reuse") == "reuse"


def _index(source="c.s.t", endpoint="vs", pk="cell_id", dim=512, synced=("cell_id", "cell_type"), pipeline="p1",
           ready=True, rows=10):
    spec = NS(source_table=source, embedding_vector_columns=[NS(name="embedding", embedding_dimension=dim)],
              columns_to_sync=list(synced) if synced is not None else None, pipeline_id=pipeline)
    return NS(delta_sync_index_spec=spec, endpoint_name=endpoint, primary_key=pk,
              status=NS(ready=ready, indexed_row_count=rows, message="m"))


def test_index_mismatches_lists_every_difference():
    ok = _index()
    assert vx.index_mismatches(ok, table="C.S.T", vs_endpoint="vs", emb_dim=512, required_columns=["cell_type"]) == []
    bad = _index(source="c.s.other", endpoint="vs2", pk="id", dim=1024, synced=("cell_id",))
    problems = vx.index_mismatches(bad, table="c.s.t", vs_endpoint="vs", emb_dim=512, required_columns=["cell_type"])
    assert len(problems) == 5
    assert vx.index_mismatches(NS(delta_sync_index_spec=None), table="c.s.t", vs_endpoint="vs", emb_dim=1) == [
        "not a Delta Sync index"]


def test_sleep_until_never_passes_the_deadline():
    with mock.patch.object(vx.time, "time", return_value=100.0), mock.patch.object(vx.time, "sleep") as sleep:
        vx.sleep_until(103.0, 15, "x")
        sleep.assert_called_once_with(3.0)
        with pytest.raises(TimeoutError):
            vx.sleep_until(100.0, 15, "x")


def _updates(*states):
    return NS(updates=[NS(update_id=f"u{i}", state=s) for i, s in enumerate(states)], next_page_token=None)


def test_sync_and_wait_tracks_the_update_it_started():
    w = mock.MagicMock()
    w.vector_search_indexes.get_index.return_value = _index()
    done = UpdateInfoState.COMPLETED
    w.pipelines.list_updates.side_effect = [
        _updates(done),                              # step 1: nothing running
        _updates(done),                              # step 2: "before" snapshot
        _updates(done, UpdateInfoState.RUNNING),     # step 3: ours is running
        _updates(done, done),                        # step 3: ours completed
    ]
    with mock.patch.object(vx.time, "sleep"):
        status = vx.sync_and_wait(w, "c.s.i", deadline=vx.time.time() + 60, poll_s=0)
    w.vector_search_indexes.sync_index.assert_called_once_with("c.s.i")
    assert status.ready


def test_sync_and_wait_raises_when_our_sync_fails():
    w = mock.MagicMock()
    w.vector_search_indexes.get_index.return_value = _index()
    done = UpdateInfoState.COMPLETED
    w.pipelines.list_updates.side_effect = [_updates(done), _updates(done), _updates(done, UpdateInfoState.FAILED)]
    with pytest.raises(RuntimeError, match="did not complete"):
        vx.sync_and_wait(w, "c.s.i", deadline=vx.time.time() + 60, poll_s=0)


def test_pipeline_permission_error_is_explained():
    w = mock.MagicMock()
    w.pipelines.list_updates.side_effect = PermissionDenied("no")
    with pytest.raises(RuntimeError, match="view access"):
        vx.pipeline_updates(w, "p1")


def _online(name="vs", kind=EndpointType.STANDARD):
    return NS(name=name, endpoint_type=kind, endpoint_status=NS(state=EndpointStatusState.ONLINE))


def test_ensure_endpoint_needs_run_go_to_create():
    w = mock.MagicMock()
    w.vector_search_endpoints.get_endpoint.side_effect = NotFound("missing")
    with pytest.raises(RuntimeError, match="run_go=True"):
        vx.ensure_endpoint(w, "vs")
    w.vector_search_endpoints.create_endpoint_and_wait.assert_not_called()
    w.vector_search_endpoints.create_endpoint_and_wait.return_value = _online()
    assert vx.ensure_endpoint(w, "vs", run_go=True).name == "vs"


def test_ensure_endpoint_reuses_but_never_replaces_and_checks_type():
    w = mock.MagicMock()
    w.vector_search_endpoints.get_endpoint.return_value = _online(kind=EndpointType.STORAGE_OPTIMIZED)
    with pytest.raises(ValueError, match="requested type"):
        vx.ensure_endpoint(w, "vs", choice="reuse")
    w.vector_search_endpoints.get_endpoint.return_value = _online()
    assert vx.ensure_endpoint(w, "vs", choice="replace").name == "vs"     # replace -> reuse for shared compute
    w.vector_search_endpoints.delete_endpoint.assert_not_called()


def test_ensure_index_creates_only_with_run_go_and_never_deletes():
    w = mock.MagicMock()
    w.vector_search_indexes.get_index.side_effect = NotFound("missing")
    kwargs = dict(endpoint=_online(), table="c.s.t", emb_dim=512, expected_rows=10, columns_to_sync=["cell_id"])
    with pytest.raises(RuntimeError, match="run_go=True"):
        vx.ensure_index(w, "c.s.i", **kwargs)
    w.vector_search_indexes.create_index.assert_not_called()

    w.vector_search_indexes.get_index.side_effect = [NotFound("missing"), _index(), _index()]
    with mock.patch.object(vx.time, "sleep"):
        status = vx.ensure_index(w, "c.s.i", run_go=True, poll_s=0, **kwargs)
    assert status.indexed_row_count == 10
    spec = w.vector_search_indexes.create_index.call_args.kwargs["delta_sync_index_spec"]
    assert spec.columns_to_sync == ["cell_id"] and spec.embedding_vector_columns[0].embedding_dimension == 512

    w2 = mock.MagicMock()
    w2.vector_search_indexes.get_index.return_value = _index()
    with pytest.raises(RuntimeError, match="delete it explicitly"):
        vx.ensure_index(w2, "c.s.i", choice="replace", **kwargs)
    w2.vector_search_indexes.delete_index.assert_not_called()


def test_storage_optimized_index_omits_columns_to_sync():
    w = mock.MagicMock()
    w.vector_search_indexes.get_index.side_effect = [NotFound("missing"), _index(synced=None), _index(synced=None)]
    with mock.patch.object(vx.time, "sleep"):
        vx.ensure_index(w, "c.s.i", endpoint=_online(kind=EndpointType.STORAGE_OPTIMIZED), table="c.s.t",
                        emb_dim=512, expected_rows=10, columns_to_sync=["cell_id"], run_go=True, poll_s=0)
    assert w.vector_search_indexes.create_index.call_args.kwargs["delta_sync_index_spec"].columns_to_sync is None


def test_grant_select_quotes_names():
    sql = []
    vx.grant_select(sql.append, ["c.s.t", "c.s.i"], "workshop-group")
    assert sql == ["GRANT SELECT ON TABLE `c`.`s`.`t` TO `workshop-group`",
                   "GRANT SELECT ON TABLE `c`.`s`.`i` TO `workshop-group`"]


def test_index_sanity_check():
    w = mock.MagicMock()
    rows = [["a", 1.0]] + [[f"b{i}", 0.9 - i / 100] for i in range(9)]
    w.vector_search_indexes.query_index.return_value = NS(result=NS(data_array=rows))
    assert vx.index_sanity_check(w, "c.s.i", columns=["cell_id"], cell_id="a", vector=[0.0], k=10)[0] == 1.0
    w.vector_search_indexes.query_index.return_value = NS(result=NS(data_array=rows[1:]))
    with pytest.raises(RuntimeError, match="top-1"):
        vx.index_sanity_check(w, "c.s.i", columns=["cell_id"], cell_id="a", vector=[0.0], k=10)


def _not_ready():
    return RuntimeError("Vector index c.s.i is not ready")


def test_wait_until_queryable_needs_answers_in_a_row():
    w = mock.MagicMock()
    ok = NS(result=NS(data_array=[["a", 1.0]]))
    w.vector_search_indexes.query_index.side_effect = [ok, _not_ready(), ok, ok, ok]
    with mock.patch.object(vx.time, "sleep"):
        vx.wait_until_queryable(w, "c.s.i", emb_dim=2, deadline=vx.time.time() + 60, poll_s=0, confirm_gap_s=0)
    assert w.vector_search_indexes.query_index.call_count == 5    # the "not ready" reset the count


def test_wait_until_queryable_other_errors_and_deadline():
    w = mock.MagicMock()
    w.vector_search_indexes.query_index.side_effect = PermissionDenied("no access")
    with pytest.raises(PermissionDenied):
        vx.wait_until_queryable(w, "c.s.i", emb_dim=2, deadline=vx.time.time() + 60, poll_s=0)
    assert w.vector_search_indexes.query_index.call_count == 1
    w.vector_search_indexes.query_index.side_effect = _not_ready()
    with pytest.raises(TimeoutError, match="does not answer queries yet"):
        vx.wait_until_queryable(w, "c.s.i", emb_dim=2, deadline=vx.time.time() - 1, poll_s=0)


def test_index_sanity_check_retries_not_ready_only_with_a_deadline():
    rows = [["a", 1.0]] + [[f"b{i}", 0.9 - i / 100] for i in range(9)]
    good = NS(result=NS(data_array=rows))
    kwargs = dict(columns=["cell_id"], cell_id="a", vector=[0.0], k=10)
    w = mock.MagicMock()
    w.vector_search_indexes.query_index.side_effect = [_not_ready(), _not_ready(), good]
    with mock.patch.object(vx.time, "sleep"):
        assert vx.index_sanity_check(w, "c.s.i", deadline=vx.time.time() + 60, poll_s=0, **kwargs)[0] == 1.0
    assert w.vector_search_indexes.query_index.call_count == 3
    w.vector_search_indexes.query_index.side_effect = [_not_ready()]
    with pytest.raises(RuntimeError, match="not ready"):
        vx.index_sanity_check(w, "c.s.i", **kwargs)                     # no deadline: as before
    w.vector_search_indexes.query_index.side_effect = [NS(result=NS(data_array=rows[1:] + [["z", 0.1]]))]
    with pytest.raises(RuntimeError, match="top-1"):
        vx.index_sanity_check(w, "c.s.i", deadline=vx.time.time() + 60, poll_s=0, **kwargs)
    assert w.vector_search_indexes.query_index.call_count == 5          # a failed check is not retried

