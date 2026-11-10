"""Create or reuse a Databricks AI Search endpoint and Delta Sync index for per-sample embeddings.

Databricks AI (artificial intelligence) Search is the product formerly called Vector Search; the
`databricks-sdk` still uses the `vector_search_*` names. These helpers are the tested code behind
`references/workflows/embeddings-to-vector-search.md` steps 0b, 3 and 4; the workflow page explains when
and why to call each one.

Rules this module keeps:

- **Read-only unless told otherwise.** Nothing is created unless the caller passes `run_go=True`
  (the user's confirmation, step 0b of the workflow).
- **Never deletes.** If the user chose *replace* for an existing index, the caller deletes it
  explicitly first (the workflow shows that one line); `ensure_index` refuses to touch it.
- **No guessing about sync state.** An existing index reports `ready=True` while it still serves its
  previous snapshot, so `sync_and_wait` tracks the sync pipeline update it started.
- **A new index can flap.** It can answer one query and say "not ready" to the next, so
  `wait_until_queryable` needs several answers in a row and `index_sanity_check` retries "not ready"
  when given a deadline.

Importable without Spark, PyTorch or a GPU: every function takes the `WorkspaceClient` (`w`) and, where
SQL is needed, a `run_sql` callable such as `spark.sql`. Checked against `databricks-sdk` 0.122.0
(serverless environment v6) and the names used by the 0.67.0 / 0.157.0 runs the workflow cites.
"""
import time

from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.sdk.service.pipelines import UpdateInfoState
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    EndpointStatusState,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

_DONE = {UpdateInfoState.COMPLETED}
_FAILED = {UpdateInfoState.FAILED, UpdateInfoState.CANCELED}
CHOICES = ("stop", "reuse", "replace")


# ---------------------------------------------------------------------------------------------------
# Unity Catalog (UC) names
# ---------------------------------------------------------------------------------------------------
def uc_name_parts(name):
    """Split a UC three-part name into lowercase parts, unquoting `...` identifiers (`` = one backtick)."""
    parts, buf, quoted, i = [], [], False, 0
    while i < len(name):
        ch = name[i]
        if ch == "`":
            if quoted and name[i + 1:i + 2] == "`":     # doubled backtick inside a quoted identifier
                buf.append("`")
                i += 2
                continue
            quoted = not quoted
        elif ch == "." and not quoted:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    if quoted or len(parts) != 3 or not all(parts):
        raise ValueError(f"Not a catalog.schema.name identifier: {name!r}")
    return tuple(p.lower() for p in parts)                 # UC identifiers are case-insensitive


def same_uc_name(api_name, ours):
    """Compare a name returned by the API with our configured name, component by component."""
    try:
        theirs = uc_name_parts(api_name)
    except (TypeError, ValueError):                         # None, or not a UC three-part name
        return False
    return theirs == uc_name_parts(ours)


def quote_uc_name(name):
    """Backtick-quote each part of a UC name for SQL (so hyphens and odd characters are safe)."""
    return ".".join("`" + p.replace("`", "``") + "`" for p in uc_name_parts(name))


# ---------------------------------------------------------------------------------------------------
# Step 0b — what already exists, and the one guard every later step goes through
# ---------------------------------------------------------------------------------------------------
def existing_resources(w, *, uc_model=None, table=None, index=None, vs_endpoint=None, table_rows=None):
    """Read-only lookup of resources under the proposed names. Returns a list of plain-language lines.

    table_rows: optional callable(table) -> row count (e.g. lambda t: spark.table(t).count()).
    """
    found = []
    if uc_model:
        # served_entities is the current field; served_models is the older one some endpoints still report.
        for e in w.serving_endpoints.list():
            cfg = e.config
            served = [(s.entity_name, s.entity_version) for s in (cfg.served_entities or [])] if cfg else []
            served += [(s.model_name, s.model_version) for s in (cfg.served_models or [])] if cfg else []
            for name, version in served:
                if same_uc_name(name, uc_model):
                    found.append(f"serving endpoint {e.name}: serves {name} v{version} "
                                 f"({len(served)} served entities), ready={e.state.ready if e.state else None}")
    if table and w.tables.exists(table).table_exists:
        rows = f"{table_rows(table)} rows, " if table_rows else ""
        cols = [c.name for c in (w.tables.get(table).columns or [])]
        found.append(f"table {table}: {rows}columns={cols}")
    if index:
        try:
            idx = w.vector_search_indexes.get_index(index)
            spec = idx.delta_sync_index_spec
            cols = (spec.embedding_vector_columns or []) if spec else []
            found.append(f"index {index}: endpoint={idx.endpoint_name}, "
                         f"source_table={spec.source_table if spec else None}, primary_key={idx.primary_key}, "
                         f"vector columns={[(c.name, c.embedding_dimension) for c in cols]}, "
                         f"ready={idx.status.ready if idx.status else None}")
        except NotFound:
            pass
    if vs_endpoint:
        try:
            ep = w.vector_search_endpoints.get_endpoint(vs_endpoint)
            n = len(list(w.vector_search_indexes.list_indexes(vs_endpoint)))
            found.append(f"AI Search endpoint {vs_endpoint}: {ep.endpoint_type}, "
                         f"{ep.endpoint_status.state if ep.endpoint_status else None}, {n} indexes")
        except NotFound:
            pass
    return found


def on_existing(kind, name, choice, details=""):
    """Return the user's choice for an existing resource; stop if none was made.

    choice: "stop" (default — raise and ask), "reuse" or "replace", set ONLY from the user's answer.
    """
    if choice not in CHOICES:
        raise ValueError(f"on_existing choice must be one of {CHOICES}, got {choice!r}")
    if choice == "stop":
        raise RuntimeError(f"{kind} {name} already exists {details}. Ask the user: reuse it, use a new "
                           "name, or replace it (workflow step 0b) — then pass their choice.")
    print(f"{kind} {name} exists → {choice} (confirmed by the user)")
    return choice


def index_mismatches(idx, *, table, vs_endpoint, emb_dim, primary_key="cell_id", vector_column="embedding",
                     required_columns=None):
    """Why an existing index cannot be reused for this run (an empty list means it can).

    required_columns: columns later queries ask for; on a STANDARD endpoint they must be in columns_to_sync.
    """
    spec = idx.delta_sync_index_spec
    if spec is None:
        return ["not a Delta Sync index"]
    cols = spec.embedding_vector_columns or []
    problems = [] if same_uc_name(spec.source_table, table) else [
        f"source_table={spec.source_table!r}, expected {table!r}"]
    want = {"endpoint": vs_endpoint, "primary_key": primary_key,
            "vector column (name, dim)": [(vector_column, emb_dim)]}
    got = {"endpoint": idx.endpoint_name, "primary_key": idx.primary_key,
           "vector column (name, dim)": [(c.name, c.embedding_dimension) for c in cols]}
    problems += [f"{k}={got[k]!r}, expected {want[k]!r}" for k in want if got[k] != want[k]]
    synced = spec.columns_to_sync
    if required_columns and synced is not None and not set(required_columns) <= set(synced) | {primary_key}:
        problems.append(f"columns_to_sync={synced!r} lacks {sorted(set(required_columns) - set(synced))}")
    return problems


def existing_dim(idx):
    """Embedding dimension of an existing index (None if it has no vector column)."""
    cols = (idx.delta_sync_index_spec.embedding_vector_columns or []) if idx.delta_sync_index_spec else []
    return cols[0].embedding_dimension if cols else None


# ---------------------------------------------------------------------------------------------------
# Waiting — every wait shares one absolute deadline (time.time() seconds)
# ---------------------------------------------------------------------------------------------------
def sleep_until(deadline, poll_s, what):
    """Sleep at most poll_s and never past `deadline`; raise once it has passed."""
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError(f"Deadline reached: {what}")
    time.sleep(min(poll_s, remaining))


def is_not_ready(e):
    """True for the "index is not ready" query error (the SDK raises different classes for it)."""
    return "not ready" in str(e).lower()


def wait_until_ready(w, index_name, deadline, poll_s=15):
    """Poll until the index reports ready; raise at `deadline` instead."""
    while True:
        status = w.vector_search_indexes.get_index(index_name).status
        if status and status.ready:
            return status
        sleep_until(deadline, poll_s, f"{index_name} not ready: {status.message if status else 'no status'}")


def wait_until_queryable(w, index_name, emb_dim, deadline, primary_key="cell_id", poll_s=15,
                         confirmations=3, confirm_gap_s=10):
    """Poll with a 1-result query until the index answers `confirmations` times in a row; raise at `deadline`.

    status.ready (and the row count) can be reported before queries work: a run on 2026-11-02 got
    "index is not ready" from query_index right after get_index said ready with every row. One answer
    is not enough either: on 2026-11-08 (a new index on a new STANDARD endpoint) the probe
    answered, then the next query, seconds later, got "not ready" again. A "not ready" resets the
    count. Only errors that say the index is not ready are retried; anything else propagates.
    """
    probe = [1.0] * emb_dim
    in_a_row = 0
    while True:
        try:
            w.vector_search_indexes.query_index(index_name=index_name, columns=[primary_key],
                                                query_vector=probe, num_results=1)
            in_a_row += 1
            if in_a_row >= confirmations:
                return
            sleep_until(deadline, confirm_gap_s, f"{index_name} answered {in_a_row} of {confirmations} probes")
        except Exception as e:
            if not is_not_ready(e):
                raise
            in_a_row = 0
            sleep_until(deadline, poll_s, f"{index_name} reports ready but does not answer queries yet")


def pipeline_updates(w, pipeline_id):
    """Every update of the index's sync pipeline (no order is documented, so read all pages)."""
    updates, token = [], None
    while True:
        try:
            page = w.pipelines.list_updates(pipeline_id, page_token=token)
        except PermissionDenied as e:
            raise RuntimeError(
                f"Confirming a fresh sync needs view access to the index's sync pipeline {pipeline_id}. "
                "Ask the index owner or the facilitator to grant it. Not falling back to status.ready, "
                "which can describe the previous snapshot.") from e
        updates += page.updates or []
        token = page.next_page_token
        if not token:
            return updates


def wait_for_row_count(w, index_name, expected, deadline, poll_s=15):
    """Poll until the ready index holds exactly `expected` rows; raise at `deadline`."""
    while True:
        status = w.vector_search_indexes.get_index(index_name).status
        if status and status.ready and status.indexed_row_count == expected:
            return status
        sleep_until(deadline, poll_s, f"{index_name} has {status.indexed_row_count if status else None} "
                                      f"rows, the table has {expected}")


def sync_and_wait(w, index_name, deadline, poll_s=15):
    """Request a sync and return only after THAT sync has completed and the index is ready.

    status.ready alone is not proof: an existing index reports ready=True while it still serves the
    previous snapshot. The SDK's index status has no sync timestamp, and the error for "a sync is
    already running" is not documented, so: wait for any running update to finish, let every
    sync_index error propagate, then track the pipeline update our request started until it completes.
    """
    pipeline_id = w.vector_search_indexes.get_index(index_name).delta_sync_index_spec.pipeline_id
    if not pipeline_id:
        raise RuntimeError(f"{index_name} reports no sync pipeline, so a fresh sync cannot be confirmed — "
                           "ask the user how to proceed")

    # 1. A sync that is already running may predate the rows just written: let it finish first.
    while any(u.state not in _DONE | _FAILED for u in pipeline_updates(w, pipeline_id)):
        sleep_until(deadline, poll_s, f"{index_name}: an earlier sync is still running")

    # 2. Request the sync. No error is suppressed.
    before = {u.update_id for u in pipeline_updates(w, pipeline_id)}
    w.vector_search_indexes.sync_index(index_name)
    print(f"Sync requested on {index_name}")

    # 3. Wait for the update this request started and require it to complete.
    while True:
        new = [u for u in pipeline_updates(w, pipeline_id) if u.update_id not in before]
        failed = [(u.update_id, u.state) for u in new if u.state in _FAILED]
        if failed:
            raise RuntimeError(f"{index_name}: requested sync did not complete: {failed}")
        if new and all(u.state in _DONE for u in new):
            break
        sleep_until(deadline, poll_s, f"{index_name}: the requested sync has not completed")

    return wait_until_ready(w, index_name, deadline)


# ---------------------------------------------------------------------------------------------------
# Steps 3 and 4 — create or reuse (creation only with run_go=True; never deletes)
# ---------------------------------------------------------------------------------------------------
def ensure_endpoint(w, name, *, endpoint_type="STANDARD", emb_dim=None, choice="stop", run_go=False):
    """Return an ONLINE AI Search endpoint `name`, creating it only when run_go is True.

    An existing endpoint is shared compute: it is reused (after the user's choice), never deleted or
    replaced here, and refused if it is the other endpoint type (the index settings depend on it).
    """
    try:
        requested = EndpointType(endpoint_type)                # enum, not a string
    except ValueError:
        raise ValueError(f"This databricks-sdk has no EndpointType {endpoint_type!r}; "
                         f"available: {[t.value for t in EndpointType]}. Upgrade with %pip.") from None
    if requested != EndpointType.STANDARD and emb_dim is not None and emb_dim % 16:
        raise ValueError(f"Storage-optimized endpoints need a dimension divisible by 16, got {emb_dim}")
    try:
        ep = w.vector_search_endpoints.get_endpoint(name)
        on_existing("AI Search endpoint", name, "reuse" if choice == "replace" else choice,
                    f"({ep.endpoint_type}, {ep.endpoint_status.state if ep.endpoint_status else None})")
    except NotFound:
        if not run_go:
            raise RuntimeError(f"AI Search endpoint {name} does not exist. Creating it starts billable compute "
                               "once it hosts an index; confirm with the user, then pass run_go=True.") from None
        print(f"Creating {requested.value} AI Search endpoint {name} (takes several minutes)...")
        ep = w.vector_search_endpoints.create_endpoint_and_wait(name=name, endpoint_type=requested)
    if ep.endpoint_type != requested:
        raise ValueError(f"{name} is {ep.endpoint_type}, but the requested type is {requested.value}. "
                         "Use a matching endpoint or change the type; the index spec depends on it.")
    if ep.endpoint_status is None or ep.endpoint_status.state != EndpointStatusState.ONLINE:
        ep = w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(name)
    print(f"AI Search endpoint {name} ONLINE")
    return ep


def ensure_index(w, index, *, endpoint, table, emb_dim, expected_rows, columns_to_sync, choice="stop",
                 run_go=False, primary_key="cell_id", vector_column="embedding", timeout_s=1800, poll_s=15):
    """Create (or reuse and re-sync) a triggered Delta Sync index; return its status once it holds every row.

    endpoint: the EndpointInfo from ensure_endpoint (its type decides whether columns_to_sync is allowed).
    expected_rows: the source table's row count, read by the caller after writing it.
    The index is never deleted here: for "replace", delete it first (the workflow shows how), then call again.
    """
    deadline = time.time() + timeout_s                  # one absolute deadline for every wait below
    try:
        idx = w.vector_search_indexes.get_index(index)
    except NotFound:
        idx = None

    if idx is not None:
        chosen = on_existing("Index", index, choice, f"(endpoint={idx.endpoint_name}, dim={existing_dim(idx)})")
        if chosen == "replace":
            raise RuntimeError(f"{index} still exists. The user chose replace: delete it explicitly first "
                               "(w.vector_search_indexes.delete_index), wait until it is gone, then call again.")
        problems = index_mismatches(idx, table=table, vs_endpoint=endpoint.name, emb_dim=emb_dim,
                                    primary_key=primary_key, vector_column=vector_column,
                                    required_columns=columns_to_sync)
        if problems:
            raise RuntimeError(f"{index} cannot be reused: {problems}. Ask the user for a new index name or replace.")
        sync_and_wait(w, index, deadline, poll_s)      # existing index: wait for the sync requested here
    else:
        if not run_go:
            raise RuntimeError(f"Index {index} does not exist. Confirm creating it with the user, then pass "
                               "run_go=True.")
        w.vector_search_indexes.create_index(
            name=index,
            endpoint_name=endpoint.name,
            primary_key=primary_key,
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=table,
                embedding_vector_columns=[EmbeddingVectorColumn(name=vector_column, embedding_dimension=emb_dim)],
                pipeline_type=PipelineType.TRIGGERED,
                # Storage-optimized endpoints reject columns_to_sync and sync every source column instead.
                columns_to_sync=list(columns_to_sync) if endpoint.endpoint_type == EndpointType.STANDARD else None,
            ),
        )
        print(f"Created {index}; the initial sync runs as part of creation")
        wait_until_ready(w, index, deadline, poll_s)   # a new index has no earlier snapshot to confuse

    status = wait_for_row_count(w, index, expected_rows, deadline, poll_s)
    wait_until_queryable(w, index, emb_dim, deadline, primary_key=primary_key, poll_s=poll_s)
    print(f"Index ready and answering queries: {status.indexed_row_count} rows, matching the table")
    return status


def grant_select(run_sql, objects, principal):
    """GRANT SELECT on tables and indexes to a user or group (idempotent).

    Querying an AI Search index needs USE CATALOG, USE SCHEMA and SELECT on the index itself; an index is
    a UC table-type securable, so the statement is GRANT ... ON TABLE <index>. run_sql: e.g. spark.sql.
    """
    for obj in objects:
        run_sql(f"GRANT SELECT ON TABLE {quote_uc_name(obj)} TO `{principal.replace('`', '``')}`")
        print(f"GRANT SELECT ON TABLE {obj} TO {principal}")


def index_sanity_check(w, index, *, columns, cell_id, vector, k=10, deadline=None, poll_s=15):
    """After a rebuild: a cell already in the table must find itself first, with k rows, scores not increasing.

    columns: columns to return; must start with the primary key column (the score is appended last).
    deadline: optional absolute time.time(); if given, "index is not ready" errors are retried until then
    (a new index can still flap right after wait_until_queryable). Other errors, and a failed check,
    are never retried.
    """
    while True:
        try:
            hits = w.vector_search_indexes.query_index(index_name=index, columns=list(columns),
                                                       query_vector=[float(v) for v in vector], num_results=k)
            break
        except Exception as e:
            if deadline is None or not is_not_ready(e):
                raise
            sleep_until(deadline, poll_s, f"{index} not ready for the sanity check")
    rows = (hits.result.data_array or []) if hits.result else []
    scores = [float(r[-1]) for r in rows]
    problems = []
    if len(rows) != k:
        problems.append(f"{len(rows)} rows returned, expected {k}")
    if rows and str(rows[0][0]) != str(cell_id):
        problems.append(f"top-1 is {rows[0][0]!r}, not the query cell {cell_id!r}")
    if any(b > a + 1e-9 for a, b in zip(scores, scores[1:])):
        problems.append(f"scores increase down the list: {scores}")
    if problems:
        raise RuntimeError(f"{index} failed the sanity check: {problems}")
    print(f"Sanity check passed: {cell_id} is its own top-1 (score {scores[0]:.4f}), {k} rows, scores not increasing")
    return scores
