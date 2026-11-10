# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Check: embeddings → Delta table → Databricks AI Search
# MAGIC
# MAGIC Dev-only notebook. Runs the code blocks of
# MAGIC `references/workflows/embeddings-to-vector-search.md` end to end on a **small** cell set
# MAGIC (default 100 cells) so the reference can be verified in a workspace before a workshop.
# MAGIC Lives in `tests/`, so it is excluded from the published skill bundle.
# MAGIC
# MAGIC **Prerequisites** (reference §Step 0): a READY TEDDY Model Serving endpoint (for
# MAGIC `embed_mode=serving`) or the registered UC model (for `embed_mode=local_model`, GPU
# MAGIC compute), and the TEDDY checkpoint directory (`config.json`, `vocab.txt`) in a Volume.
# MAGIC
# MAGIC **Confirm before building** (reference §Step 0b). Run with `run_go=false` first: the
# MAGIC settings-check cell prints every resolved choice (marking defaults), lists resources that
# MAGIC already exist under the chosen names, and stops. Confirm them, then set `run_go=true`.
# MAGIC
# MAGIC **Existing resources** — widget `on_existing`:
# MAGIC * `stop` (default): raise, listing what exists and how to proceed.
# MAGIC * `reuse`: write to the existing table, sync the existing index, use the existing endpoint.
# MAGIC * `replace`: the **only** path that deletes anything — it prints exactly what it will delete
# MAGIC   (table, index) before doing it. Endpoints and UC model versions are never deleted.
# MAGIC
# MAGIC **Creates** (only when `run_go=true`): a Delta table, an AI Search index, and — if it does not
# MAGIC exist — the AI Search endpoint. With `teardown=true` the index is deleted at the end, also when
# MAGIC a check fails.
# MAGIC
# MAGIC The code under test is the skill's `scripts/` (`vs_index.py`, `nn_annotate.py`, `visualize.py`); set the
# MAGIC `skill_dir` widget to the installed skill. Inference controls go to the endpoint under the REST body key `"params"` (`nn.invoke()`, as in
# MAGIC `teddy.md` §Sending inference controls), never `serving_endpoints.query(extra_params=...)`, which
# MAGIC custom PyFunc endpoints silently ignore; step 1 proves a non-default value changes the output.
# MAGIC
# MAGIC Cells follow the reference's steps; each prints a ✓ line or raises.
# MAGIC
# MAGIC The next two cells install dependencies with `%pip`, then restart Python:
# MAGIC the TEDDY pins from `teddy.md` §Dependencies (exact `transformers`; needed for
# MAGIC `embed_mode=local_model`; same order as the validated oss-002 v2 run), then
# MAGIC `cellxgene-census` (also provides `anndata`) and `umap-learn` (reference §Interpret and visualise
# MAGIC the results) — `%pip` + restart, never a subprocess (`teddy.md` §Dependencies and the botocore
# MAGIC conflict).

# COMMAND ----------

# MAGIC %pip install -q "transformers==4.41.0" "numpy>=1.26.4,<2.0" "pandas>=2.2.2,<3.0" mlflow

# COMMAND ----------

# MAGIC %pip install -q cellxgene-census umap-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widgets and configuration (reference §Step 0)
import hashlib
import json
import os
import re
import sys

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

RESOLVED = {}  # widget -> (value, "default" or "set by user"), printed by the settings check


def _text(name, default):
    dbutils.widgets.text(name, default)
    value = dbutils.widgets.get(name).strip()
    RESOLVED[name] = (value, "default" if value == default else "set by user")
    return value


def _dropdown(name, default, choices):
    dbutils.widgets.dropdown(name, default, choices)
    value = dbutils.widgets.get(name)
    RESOLVED[name] = (value, "default" if value == default else "set by user")
    return value


# The tested helpers live once, in the skill's scripts/ folder (reference §Status). Point skill_dir at the
# installed skill (personal .assistant/skills or the workspace-wide one) or a repo checkout of it.
SKILL_DIR = _text("skill_dir", "/Workspace/Users/<user>/.assistant/skills/open-weight-models")
sys.path.insert(0, f"{SKILL_DIR}/scripts")
import nn_annotate as nn  # noqa: E402
import visualize as viz  # noqa: E402
import vs_index as vx  # noqa: E402

CATALOG = _text("catalog", "<catalog>")
SCHEMA = _text("schema", "<schema>")
VOLUME = _text("volume", "<volume>")
VS_ENDPOINT = _text("vs_endpoint", "<vs_endpoint>")
ENDPOINT_TYPE = _dropdown("endpoint_type", "STANDARD", ["STANDARD", "STORAGE_OPTIMIZED"])
SERVING_ENDPOINT = _text("serving_endpoint", "<serving_endpoint>")
VARIANT = _text("variant", "70m").lower()
_model_dir = _text("model_dir", "")  # blank = teddy.md §Checkpoint directory layout under the volume
_uc_model = _text("uc_model", "")  # blank = teddy.md naming: <catalog>.<schema>.teddy_<variant>
_table = _text("table_name", "")  # blank = derived (per-user if per_user_names=true)
_index = _text("index_name", "")  # blank = derived (per-user if per_user_names=true)
N_CELLS = int(_text("n_cells", "100"))
CENSUS_VERSION = _text("census_version", "2025-01-30")  # the release the 2026-10-27 teddy-002 run used
H5AD_PATH = _text("h5ad_path", "")  # optional: pre-staged .h5ad in a Volume (no internet needed)
EMBED_MODE = _dropdown("embed_mode", "serving", ["serving", "local_model"])
UC_MODEL_VERSION = _text("uc_model_version", "1")
# GPU the model runs on: the serving endpoint's workload type, or the notebook's GPU for local_model
# (GPU_SMALL = T4, GPU_MEDIUM = A10G). It sets the TEDDY batch cap (teddy.md §Compute requirements).
GPU_TIER = _dropdown("gpu_tier", "GPU_SMALL", ["GPU_SMALL", "GPU_MEDIUM"])
_batch_widget = _text("batch_size", "")  # blank = the gpu_tier cap
PER_USER = _dropdown("per_user_names", "true", ["true", "false"]) == "true"
ON_EXISTING = _dropdown("on_existing", "stop", ["stop", "reuse", "replace"])
RUN_GO = _dropdown("run_go", "false", ["false", "true"]) == "true"
TEARDOWN = _dropdown("teardown", "false", ["false", "true"]) == "true"

for _name, _val in [("catalog", CATALOG), ("schema", SCHEMA), ("volume", VOLUME), ("vs_endpoint", VS_ENDPOINT)]:
    assert _val and not _val.startswith("<"), f"Set the '{_name}' widget"
if EMBED_MODE == "serving":
    assert SERVING_ENDPOINT and not SERVING_ENDPOINT.startswith("<"), "Set the 'serving_endpoint' widget"
assert 2 <= N_CELLS <= 1000, "Keep this check small: 2-1000 cells"

# Inference controls as strings, matching the signature's string defaults (teddy.md §Sending inference controls)
TEDDY_PARAMS = {"max_seq_len": "2048", "pooling": "mean"}

# TEDDY batch caps per GPU tier (nn.MAX_BATCH; teddy.md §Compute requirements; §Census generation pipeline, step 7)
BATCH_SIZE = int(_batch_widget) if _batch_widget else nn.MAX_BATCH[GPU_TIER]
if not 1 <= BATCH_SIZE <= nn.MAX_BATCH[GPU_TIER]:
    raise ValueError(f"batch_size {BATCH_SIZE} is outside 1..{nn.MAX_BATCH[GPU_TIER]} for {GPU_TIER}")

w = WorkspaceClient()

MODEL_DIR = _model_dir or f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/snapshot/teddy/models/teddy_g/{VARIANT.upper()}"
UC_MODEL = _uc_model or f"{CATALOG}.{SCHEMA}.teddy_{VARIANT}"

# Reference §Shared workspace: per-participant names. The readable part is lossy (a-b@ and a_b@ both
# become a_b), so a hash of the FULL user name keeps them unique; the "u_" prefix starts with a letter.
if PER_USER:
    USER_NAME = w.current_user.me().user_name
    _readable = re.sub(r"[^a-z0-9]", "_", USER_NAME.split("@")[0].lower())[:20]
    _digest = hashlib.sha1(USER_NAME.encode()).hexdigest()[:8]
    _stem = f"u_{_readable}_{_digest}_teddy"
else:
    _stem = "teddy"
TABLE = _table or f"{CATALOG}.{SCHEMA}.{_stem}_cells_{VARIANT}"
INDEX = _index or f"{CATALOG}.{SCHEMA}.{_stem}_cell_index_{VARIANT}"


INDEX_COLUMNS = ["cell_id", "cell_type", "tissue_general", "disease"]  # columns the index returns (step 5)
same_uc_name = vx.same_uc_name


def on_existing(kind, name, details=""):
    """vx.on_existing bound to the on_existing widget (reference §Step 0b)."""
    return vx.on_existing(kind, name, ON_EXISTING, details)


def index_mismatches(idx):
    """vx.index_mismatches bound to this run's table, endpoint, dimension and synced columns."""
    return vx.index_mismatches(idx, table=TABLE, vs_endpoint=VS_ENDPOINT, emb_dim=EMB_DIM,
                               required_columns=INDEX_COLUMNS)


def teardown_index():
    """Delete this run's index (never the shared endpoint). Safe to call more than once."""
    try:
        w.vector_search_indexes.delete_index(INDEX)
        print(f"✓ Deleted {INDEX}")
    except NotFound:
        print(f"{INDEX} already gone")

# COMMAND ----------

# DBTITLE 1,Embedding dimension and vocabulary from the checkpoint (reference §Step 0)
for _f in ("config.json", "vocab.txt"):
    assert os.path.isfile(os.path.join(MODEL_DIR, _f)), (
        f"{_f} not found in {MODEL_DIR} — point the model_dir widget at the checkpoint directory"
    )

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    EMB_DIM = int(json.load(f)["d_model"])

with open(os.path.join(MODEL_DIR, "vocab.txt")) as f:
    VOCAB_GENES = {t.strip() for t in f if t.strip().startswith("ENSG")}

print(f"✓ EMB_DIM={EMB_DIM}, vocab Ensembl IDs={len(VOCAB_GENES):,}")

# COMMAND ----------

# DBTITLE 1,Settings check — resolved choices and existing resources (read-only; reference §Step 0b)
from databricks.sdk.service.serving import EndpointStateReady
from databricks.sdk.service.vectorsearch import EndpointType

# 1. Resolved choices, marking defaults so the user can confirm them
print("Resolved choices (confirm before run_go=true):")
for _name, (_val, _origin) in RESOLVED.items():
    print(f"  {_name:<17} = {_val!r:<45} [{_origin}]")
_derived = [
    ("model_dir", MODEL_DIR, bool(_model_dir)), ("uc_model", UC_MODEL, bool(_uc_model)),
    ("table_name", TABLE, bool(_table)), ("index_name", INDEX, bool(_index)),
]
for _name, _val, _was_set in _derived:
    if not _was_set:
        print(f"  {_name:<17} → {_val}   [default applied: derived from naming convention]")
print(f"  batch_size        → {BATCH_SIZE}   [{'set by user' if _batch_widget else 'default applied: ' + GPU_TIER + ' cap'}]")
print(f"  embedding dim     → {EMB_DIM}   [read from {MODEL_DIR}/config.json]")
print(f"  data source       → {'h5ad ' + H5AD_PATH if H5AD_PATH else 'CELLxGENE Census ' + CENSUS_VERSION}, {N_CELLS} cells")
print(f"  teardown          → {'delete this run’s index at the end' if TEARDOWN else 'keep everything'}")

# 2. Settings that must be compatible before anything is created
try:
    REQUESTED_TYPE = EndpointType(ENDPOINT_TYPE)
except ValueError:
    raise ValueError(f"This databricks-sdk has no EndpointType {ENDPOINT_TYPE!r}; "
                     f"available: {[t.value for t in EndpointType]}. Upgrade with %pip.") from None
if REQUESTED_TYPE != EndpointType.STANDARD:
    assert EMB_DIM % 16 == 0, f"Storage-optimized endpoints need a dimension divisible by 16, got {EMB_DIM}"

if EMBED_MODE == "serving":
    print("  serving endpoint (traffic and exact versions):")
    _sep, _workload = nn.serving_endpoint_check(w, SERVING_ENDPOINT, uc_model=UC_MODEL,
                                                uc_model_version=UC_MODEL_VERSION)
    assert _sep.state.ready == EndpointStateReady.READY, f"{SERVING_ENDPOINT} not ready: {_sep.state}"
    assert _workload == GPU_TIER, (
        f"{SERVING_ENDPOINT} runs on {_workload}, but gpu_tier is {GPU_TIER}"
    )
else:
    import torch

    assert torch.cuda.is_available(), "embed_mode=local_model needs GPU compute"
    _gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if GPU_TIER == "GPU_MEDIUM":
        assert _gpu_gb >= 20, f"gpu_tier=GPU_MEDIUM needs an A10G-class GPU; this one has {_gpu_gb:.1f} GB"
    print(f"  local GPU         → {torch.cuda.get_device_name(0)}, {_gpu_gb:.1f} GB")

# 3. Existing resources under the chosen names (and endpoints already serving the same model)
FOUND = []           # (kind, name, details)
WILL_DELETE = []     # what on_existing=replace would delete

for _e in w.serving_endpoints.list():
    _cfg = _e.config
    _served = [(s.entity_name, s.entity_version) for s in (_cfg.served_entities or [])] if _cfg else []
    _served += [(s.model_name, s.model_version) for s in (_cfg.served_models or [])] if _cfg else []
    for _m, _v in _served:
        if same_uc_name(_m, UC_MODEL):
            FOUND.append(("Serving endpoint", _e.name,
                          f"serves {_m} v{_v} ({len(_served)} served entities), "
                          f"ready={_e.state.ready if _e.state else None} (never deleted here)"))

EXPECTED_COLUMNS = ["cell_id", "embedding", "cell_type", "tissue_general", "disease"]
TABLE_EXISTS = spark.catalog.tableExists(TABLE)
if TABLE_EXISTS:
    _cols = spark.table(TABLE).columns
    _props = {r["key"]: r["value"] for r in spark.sql(f"SHOW TBLPROPERTIES {TABLE}").collect()}
    FOUND.append(("Table", TABLE, f"{spark.table(TABLE).count()} rows, columns={_cols}, "
                                  f"CDF={_props.get('delta.enableChangeDataFeed', 'false')}"))
    WILL_DELETE.append(f"DROP TABLE {TABLE}")
    if ON_EXISTING == "reuse" and _cols != EXPECTED_COLUMNS:
        raise RuntimeError(f"{TABLE} has columns {_cols}, not {EXPECTED_COLUMNS}: it cannot be reused. "
                           "Use a new table_name or on_existing=replace.")


try:
    _idx = w.vector_search_indexes.get_index(INDEX)
    _spec = _idx.delta_sync_index_spec
    FOUND.append(("Index", INDEX, f"endpoint={_idx.endpoint_name}, source_table={_spec.source_table if _spec else None}, "
                                  f"primary_key={_idx.primary_key}, dim={vx.existing_dim(_idx)}, "
                                  f"ready={_idx.status.ready if _idx.status else None}, "
                                  f"rows={_idx.status.indexed_row_count if _idx.status else None}"))
    WILL_DELETE.append(f"delete_index({INDEX})")
    _problems = index_mismatches(_idx)
    if ON_EXISTING == "reuse" and _problems:
        raise RuntimeError(f"{INDEX} cannot be reused: {_problems}. Use a new index_name or on_existing=replace.")
except NotFound:
    pass

try:
    _ep = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT)
    _n_idx = len(list(w.vector_search_indexes.list_indexes(VS_ENDPOINT)))
    FOUND.append(("AI Search endpoint", VS_ENDPOINT,
                  f"{_ep.endpoint_type}, {_ep.endpoint_status.state}, {_n_idx} indexes (shared; never deleted here)"))
    if _ep.endpoint_type != REQUESTED_TYPE:
        raise ValueError(f"{VS_ENDPOINT} is {_ep.endpoint_type}, but endpoint_type is {REQUESTED_TYPE.value}")
except NotFound:
    pass

print("\nExisting resources:")
for _kind, _name, _details in FOUND:
    print(f"  {_kind:<18} {_name}: {_details}")
if not FOUND:
    print("  none under the chosen names")
if ON_EXISTING == "replace" and WILL_DELETE:
    print("\non_existing=replace WILL DELETE, in order:")
    for _item in WILL_DELETE:
        print(f"  - {_item}")

# 4. Stop here unless the user confirmed everything above
_summary = "; ".join(f"{k} {n}" for k, n, _ in FOUND) or "nothing"
if not RUN_GO:
    dbutils.notebook.exit(f"run_go=false — review the choices above (existing: {_summary}), then set run_go=true.")
if FOUND and ON_EXISTING == "stop":
    raise RuntimeError(
        f"Resources already exist: {_summary}. To proceed, either set on_existing=reuse (use them), "
        "change table_name / index_name / vs_endpoint (create new ones), or set on_existing=replace "
        f"(deletes: {', '.join(WILL_DELETE) or 'nothing'})."
    )
print(f"\n✓ Settings check passed: {REQUESTED_TYPE.value} endpoint, {GPU_TIER} with batch_size={BATCH_SIZE}, "
      f"on_existing={ON_EXISTING}")

# COMMAND ----------

# DBTITLE 1,Load a small cell set (Census, or a pre-staged .h5ad)
import numpy as np
import pandas as pd
import scipy.sparse


def file_digest(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:12]


if H5AD_PATH:
    import anndata

    adata = anndata.read_h5ad(H5AD_PATH)
    assert adata.n_obs >= 2, (f"{H5AD_PATH} has {adata.n_obs} cell(s); need at least 2 "
                              "(one is held out as the query cell)")
    if "feature_id" in adata.var.columns:
        adata.var_names = adata.var["feature_id"].values
    adata.var_names_make_unique()
    if adata.n_obs > N_CELLS:
        adata = adata[np.random.default_rng(42).choice(adata.n_obs, N_CELLS, replace=False)].copy()
    # obs_names recur across files: namespace with a hash of the file's contents
    _dataset = file_digest(H5AD_PATH)
    _source_ids = [f"h5ad_{_dataset}_{n}" for n in adata.obs_names]
else:
    # teddy.md §Census generation pipeline (same calls as the validated oss-002 v2 run)
    import cellxgene_census
    import tiledbsoma

    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
    experiment = census["census_data"]["homo_sapiens"]  # lowercase key
    obs_df = experiment.obs.read(
        value_filter='is_primary_data == True and assay == "10x 3\' v3"',
        column_names=["soma_joinid", "cell_type", "tissue_general", "disease"],
    ).concat().to_pandas()
    assert len(obs_df) >= 2, (f"Census {CENSUS_VERSION} matched {len(obs_df)} cell(s); need at least 2 "
                              "(one is held out as the query cell)")
    sampled = obs_df.sample(n=min(N_CELLS, len(obs_df)), random_state=42)
    with experiment.axis_query(
        measurement_name="RNA",
        obs_query=tiledbsoma.AxisQuery(coords=(sampled["soma_joinid"].tolist(),)),
    ) as q:
        adata = q.to_anndata(X_name="raw")
    census.close()
    assert adata.n_obs >= 2, f"Census returned {adata.n_obs} cell(s) for the sample; need at least 2"
    adata.var_names = adata.var["feature_id"].values  # Ensembl IDs, not numeric indices
    adata.var_names_make_unique()
    # soma_joinid is unique only within one Census release: namespace with the release
    _source_ids = [f"census_{CENSUS_VERSION}_{j}" for j in adata.obs["soma_joinid"]]

assert all(_source_ids) and len(set(_source_ids)) == len(_source_ids), "cell IDs must be non-empty and unique"

for _col in ("cell_type", "tissue_general", "disease"):
    if _col not in adata.obs.columns:
        adata.obs[_col] = "unknown"

# Reference §Step 1: filter to vocabulary genes, stable IDs, metadata
_mask = np.array([g in VOCAB_GENES for g in adata.var_names])
assert _mask.sum() >= 50, f"Only {_mask.sum()} genes overlap the vocabulary — check gene ID format"
adata = adata[:, _mask]
GENES = list(adata.var_names)
X_ALL = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
X_ALL = X_ALL.astype(np.float32)
IDS_ALL = _source_ids
META_ALL = adata.obs[["cell_type", "tissue_general", "disease"]].astype(str).reset_index(drop=True)

# Reference §Step 1c: seeded stratified hold-out (never indexed); rare types stay in the reference.
_held, RARE_TYPES = nn.split_holdout(META_ALL["cell_type"])
if not _held.any():
    raise RuntimeError(f"No cell type has {nn.MIN_HELD_PER_TYPE + nn.MIN_REF_PER_TYPE}+ cells, so nothing can be held "
                       f"out for step 5: raise n_cells (now {N_CELLS}). Rare types: {RARE_TYPES}")
X, X_HELD = X_ALL[~_held], X_ALL[_held]
CELL_IDS = [c for c, h in zip(IDS_ALL, _held) if not h]
HELD_IDS = [c for c, h in zip(IDS_ALL, _held) if h]
META = META_ALL[~_held].reset_index(drop=True)
HELD_META = META_ALL[_held].reset_index(drop=True)
# The first held-out cell is the "new cell" of step 5b.
X_new, NEW_ID, NEW_META = X_HELD[:1], HELD_IDS[0], HELD_META.iloc[0]
print(f"✓ {len(CELL_IDS)} reference cells + {len(HELD_IDS)} held-out cells; {len(GENES):,} vocabulary genes; "
      f"{META['cell_type'].nunique()} cell types")
if RARE_TYPES:
    print(f"  not measured (fewer than {nn.MIN_HELD_PER_TYPE + nn.MIN_REF_PER_TYPE} cells, reference only): {RARE_TYPES}")

# COMMAND ----------

# DBTITLE 1,Step 1 — Embeddings (1a serving endpoint or 1b local model)
VAR_JSON = pd.DataFrame({"index": GENES}).to_json(orient="split")


def _check_batch(batch_size):
    if batch_size > nn.MAX_BATCH[GPU_TIER]:
        raise ValueError(f"batch_size {batch_size} exceeds the {GPU_TIER} limit of {nn.MAX_BATCH[GPU_TIER]}")


if EMBED_MODE == "serving":
    def embed(X, cell_ids, batch_size=BATCH_SIZE):
        # nn.embed_via_endpoint: body "params", tier batch cap, one-cell retry, no zero vectors
        return nn.embed_via_endpoint(w, SERVING_ENDPOINT, X, cell_ids, GENES, params=TEDDY_PARAMS,
                                     tier=GPU_TIER, batch_size=batch_size)
else:
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    model = mlflow.pyfunc.load_model(f"models:/{UC_MODEL}/{UC_MODEL_VERSION}")

    def embed(X, cell_ids, batch_size=BATCH_SIZE):
        _check_batch(batch_size)
        out = []
        for start in range(0, len(cell_ids), batch_size):
            ids = cell_ids[start:start + batch_size]
            res = model.predict(
                pd.DataFrame({
                    "adata_sparsematrix": [X[start:start + batch_size].tolist()],
                    "adata_obs": [pd.DataFrame({"cell_id": ids}).to_json(orient="split")],
                    "adata_var": [VAR_JSON],
                }),
                params=TEDDY_PARAMS,  # same mechanism and types as the endpoint
            )
            records = res.to_dict("records") if isinstance(res, pd.DataFrame) else list(res)
            out.extend([float(v) for v in r["embedding"]] for r in records)
        return out

EMBEDDINGS = embed(X, CELL_IDS)
assert len(EMBEDDINGS) == len(CELL_IDS), f"{len(EMBEDDINGS)} embeddings for {len(CELL_IDS)} cells"
assert all(len(e) == EMB_DIM for e in EMBEDDINGS), "Embedding length != config.d_model"
_norms = np.linalg.norm(np.array(EMBEDDINGS), axis=1)
assert (_norms > 0).all(), "Zero-length embedding returned"
assert np.ptp(np.array(EMBEDDINGS), axis=0).max() > 0, "All embeddings identical — check gene IDs"
print(f"✓ {len(EMBEDDINGS)} embeddings, dim={EMB_DIM}, norm range {_norms.min():.3f}–{_norms.max():.3f}")

HELD_EMB = embed(X_HELD, HELD_IDS)
assert len(HELD_EMB) == len(HELD_IDS) and all(len(e) == EMB_DIM for e in HELD_EMB)
print(f"✓ {len(HELD_EMB)} held-out embeddings")

# teddy.md §Validation checklist, item 8: the controls must be declared and must change the output.
import mlflow
from mlflow.types import DataType

_params_schema = mlflow.models.get_model_info(f"models:/{UC_MODEL}/{UC_MODEL_VERSION}").signature.params
_declared = {p.name: p.dtype for p in (_params_schema.params if _params_schema else [])}
_undeclared = [k for k in TEDDY_PARAMS if _declared.get(k) != DataType.string]
assert not _undeclared, (f"{UC_MODEL} v{UC_MODEL_VERSION} declares params {_declared}; {_undeclared} must be "
                         "declared as strings or they are dropped / rejected")
if EMBED_MODE == "serving":
    nn.check_controls_applied(w, SERVING_ENDPOINT, X, CELL_IDS, GENES, params=TEDDY_PARAMS, tier=GPU_TIER)
else:
    _i = int(np.argmax((X > 0).sum(axis=1)))
    assert (X[_i] > 0).sum() > 32, "Need a cell with more than 32 expressed genes for the control check"
    _rec = lambda p: model.predict(pd.DataFrame({
        "adata_sparsematrix": [X[_i:_i + 1].tolist()],
        "adata_obs": [pd.DataFrame({"cell_id": [CELL_IDS[_i]]}).to_json(orient="split")],
        "adata_var": [VAR_JSON],
    }), params=p)
    _to_vec = lambda r: np.array((r.to_dict("records") if isinstance(r, pd.DataFrame) else list(r))[0]["embedding"],
                                 dtype=float)
    _a, _b = _to_vec(_rec(TEDDY_PARAMS)), _to_vec(_rec({**TEDDY_PARAMS, "max_seq_len": "32"}))
    _cos = float(_a @ _b / (np.linalg.norm(_a) * np.linalg.norm(_b)))
    assert _cos < 0.99, f"max_seq_len=32 gave the same embedding (cosine {_cos:.4f}): controls are not reaching predict()"
    print(f"✓ Inference controls applied (cosine default vs max_seq_len=32: {_cos:.3f})")

# COMMAND ----------

# DBTITLE 1,Step 2 — Delta table with Change Data Feed (guarded MERGE upsert)
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

if spark.catalog.tableExists(TABLE):
    if on_existing("Table", TABLE) == "replace":
        print(f"Replacing: DROP TABLE {TABLE}")
        spark.sql(f"DROP TABLE {TABLE}")
    # reuse: columns were checked by the settings check; MERGE updates rows with the same cell_id

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        cell_id        STRING NOT NULL,
        embedding      ARRAY<FLOAT>,
        cell_type      STRING,
        tissue_general STRING,
        disease        STRING
    )
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
spark.sql(f"ALTER TABLE {TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

schema = StructType([
    StructField("cell_id", StringType(), False),
    StructField("embedding", ArrayType(FloatType()), False),
    StructField("cell_type", StringType(), True),
    StructField("tissue_general", StringType(), True),
    StructField("disease", StringType(), True),
])

# MERGE matches on cell_id only: a duplicate or empty ID would silently overwrite another row.
assert all(CELL_IDS) and len(set(CELL_IDS)) == len(CELL_IDS), "cell IDs must be non-empty and unique"
assert len(CELL_IDS) == len(EMBEDDINGS) == len(META)
rows = [
    (cid, [float(v) for v in emb], META.loc[i, "cell_type"], META.loc[i, "tissue_general"], META.loc[i, "disease"])
    for i, (cid, emb) in enumerate(zip(CELL_IDS, EMBEDDINGS))
]
spark.createDataFrame(rows, schema=schema).createOrReplaceTempView("new_cells")
spark.sql(f"""
    MERGE INTO {TABLE} AS t USING new_cells AS s ON t.cell_id = s.cell_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

_props = {r["key"]: r["value"] for r in spark.sql(f"SHOW TBLPROPERTIES {TABLE}").collect()}
assert _props.get("delta.enableChangeDataFeed") == "true", "CDF is not enabled on the source table"
_n = spark.sql(f"SELECT COUNT(*) AS n FROM {TABLE} WHERE cell_id IN (SELECT cell_id FROM new_cells)").first()["n"]
assert _n == len(CELL_IDS), f"Expected {len(CELL_IDS)} merged rows, found {_n}"
print(f"✓ {TABLE}: {spark.table(TABLE).count()} rows total, CDF on")

# COMMAND ----------

# DBTITLE 1,Step 3 — Create or reuse the AI Search endpoint (type must match; never deleted)
from databricks.sdk.service.vectorsearch import EndpointStatusState

# run_go was checked by the settings check; vx.ensure_endpoint creates only when it is True, never deletes,
# and refuses an endpoint of the other type.
ep = vx.ensure_endpoint(w, VS_ENDPOINT, endpoint_type=ENDPOINT_TYPE, emb_dim=EMB_DIM, choice=ON_EXISTING,
                        run_go=RUN_GO)
assert ep.endpoint_status.state == EndpointStatusState.ONLINE
print(f"✓ AI Search endpoint {VS_ENDPOINT} ONLINE ({ep.endpoint_type})")

# COMMAND ----------

# DBTITLE 1,Step 4 — Create, sync or replace the Delta Sync index; wait until ready or raise
import time

try:
    # on_existing=replace is the only path that deletes anything; the settings check printed it.
    if ON_EXISTING == "replace":
        try:
            w.vector_search_indexes.get_index(INDEX)
            print(f"Replacing: delete_index({INDEX})")
            w.vector_search_indexes.delete_index(INDEX)
            _deadline = time.time() + 900
            while True:                                    # wait until it is really gone
                try:
                    w.vector_search_indexes.get_index(INDEX)
                except NotFound:
                    break
                vx.sleep_until(_deadline, 10, f"{INDEX} still exists after deletion")
        except NotFound:
            pass

    # vx.ensure_index: create (run_go) and wait for ready, or reuse and wait for the sync it requested,
    # then require the row count to match the table — one 30-minute deadline for every wait.
    status = vx.ensure_index(w, INDEX, endpoint=ep, table=TABLE, emb_dim=EMB_DIM,
                             expected_rows=spark.table(TABLE).count(), columns_to_sync=INDEX_COLUMNS,
                             choice="reuse" if ON_EXISTING == "replace" else ON_EXISTING, run_go=RUN_GO)
    assert not index_mismatches(w.vector_search_indexes.get_index(INDEX)), "Index settings differ from this run"
    print(f"✓ Index ready: {status.indexed_row_count} rows indexed")
except BaseException:
    if TEARDOWN:
        teardown_index()
    raise

# COMMAND ----------

# DBTITLE 1,Step 5 — Sanity checks and nearest neighbours of the held-out cell (teardown in finally)
from collections import Counter

NUM_RESULTS = min(nn.K, len(CELL_IDS))
COLUMNS = ["cell_id", "cell_type", "tissue_general"]
assert set(COLUMNS) <= set(INDEX_COLUMNS), "query only columns the index syncs (step 4)"

try:
    assert w.vector_search_indexes.get_index(INDEX).status.ready, "Index not ready — re-run step 4"

    # 5a. teddy.md §AI Search eval scorer: self-retrieval, result count, scores non-increasing.
    # Step 4 waited for status.ready, so every failure here is an error (no soft warnings).
    for cid, emb in list(zip(CELL_IDS, EMBEDDINGS))[:3]:
        vx.index_sanity_check(w, INDEX, columns=COLUMNS, cell_id=cid, vector=emb, k=NUM_RESULTS)
    print("✓ Self-retrieval, count and ordering checks passed")

    # 5b. Reference §Step 5: embed the held-out cell and look up its neighbours
    query_vec = embed(X_new, [NEW_ID])[0]
    hits = w.vector_search_indexes.query_index(
        index_name=INDEX, columns=COLUMNS, query_vector=query_vec, num_results=NUM_RESULTS,
    ).result.data_array or []
    for cell_id, cell_type, tissue, score in hits:
        print(f"  {float(score):.4f}  {cell_type:<40} {tissue:<15} {cell_id}")
    assert hits, "No neighbours returned for the held-out cell"
    if NEW_ID in [h[0] for h in hits]:
        print(f"⚠ {NEW_ID} is already in the index (indexed by an earlier run with a different n_cells)")

    label, n = Counter(h[1] for h in hits).most_common(1)[0]
    print(f"✓ Suggested cell type: {label} ({n}/{len(hits)} neighbours agree); "
          f"known label: {NEW_META['cell_type']} — a technical lookup, not a validated annotation")

    # 5c. Dictionary filter — STANDARD endpoints only (storage-optimized use SQL-like filter strings)
    if ep.endpoint_type == EndpointType.STANDARD:
        _tissue = NEW_META["tissue_general"]
        filtered = w.vector_search_indexes.query_index(
            index_name=INDEX, columns=COLUMNS, query_vector=query_vec, num_results=NUM_RESULTS,
            filters_json=json.dumps({"tissue_general": _tissue}),
        ).result.data_array or []
        assert all(h[2] == _tissue for h in filtered), "Filter returned rows from another tissue"
        print(f"✓ Filter tissue_general={_tissue!r}: {len(filtered)} rows, all matching")
    else:
        print("  Filter check skipped: storage-optimized endpoints use SQL-like filter strings")

    # 5d. Reference §Interpret and visualise the results: held-out accuracy, calibrated threshold, flags
    from pyspark.sql import functions as F

    assert spark.table(TABLE).where(F.col("cell_id").isin(HELD_IDS)).count() == 0, (
        "Held-out cells are in the table (an earlier run indexed them): use a new table_name / index_name")
    held = nn.annotate(w, INDEX, HELD_EMB, columns=COLUMNS, k=NUM_RESULTS)
    held["true"] = HELD_META["cell_type"].to_numpy()
    held["correct"] = held["predicted"] == held["true"]
    THRESHOLD, CUTOFFS = nn.calibrate_vote_threshold(held["agreement"], held["correct"])
    SCORE_FLOOR = nn.score_floor(held["top1_score"])
    print(CUTOFFS.round(3).to_string(index=False))
    QUERY_EMB = np.asarray([query_vec])
    new = nn.flag_cells(nn.annotate(w, INDEX, QUERY_EMB, columns=COLUMNS, k=NUM_RESULTS), THRESHOLD, SCORE_FLOOR)
    assert held["agreement"].between(0, 1).all() and len(new) == len(QUERY_EMB)
    print(f"✓ Held-out accuracy {held['correct'].mean():.1%} on {len(held)} cells "
          f"(chance about {1 / META['cell_type'].nunique():.1%}); threshold (Wilson lower bound >= {nn.TARGET:.0%}, "
          f">= {nn.MIN_ACCEPTED} cells): {THRESHOLD}; "
          f"score floor {SCORE_FLOOR}; new cell → {new.loc[0, 'predicted']} "
          f"({new.loc[0, 'agreement']:.0%}) {new.loc[0, 'flag'] or 'not flagged'}")

    # 5e. Reference §Optional — adapt the model: a classifier on the frozen embeddings vs the vote
    COMPARISON, _clf_pred, _ = nn.compare_with_classifier(EMBEDDINGS, META["cell_type"], HELD_EMB,
                                                          HELD_META["cell_type"], held_vote=held["predicted"])
    print(COMPARISON.round(3).to_string(index=False))
finally:
    if TEARDOWN:
        teardown_index()

# COMMAND ----------

# DBTITLE 1,Figures (reference §Interpret and visualise the results; local data only)
OUT = "/tmp/check_embeddings_figures"
ref_xy, new_xy, method = viz.map_2d(EMBEDDINGS, QUERY_EMB)
top = list(META["cell_type"].value_counts().index[:4])
for name, fig in [
    ("reference_map", viz.plot_reference_map(ref_xy, META["cell_type"], top, new_xy, method=method,
                                             note=f"{len(ref_xy)} reference cells, {len(new_xy)} new cell(s)")),
    ("heldout_confidence", viz.plot_vote_confidence(held["agreement"], held["correct"], k=NUM_RESULTS,
                                                    threshold=THRESHOLD)),
    ("heldout_confusion", viz.plot_confusion(*viz.confusion_fraction(held["true"], held["predicted"]))),
    ("vote_vs_classifier", viz.plot_accuracy_comparison(COMPARISON)),
]:
    viz.save_figure(fig, OUT, name)
    display(fig)        # helpers return figures detached from pyplot, so plt.show() would draw nothing
print(held.groupby("true")["correct"].agg(accuracy="mean", cells="size").sort_values("accuracy"))
print("✓ Figures drawn")

# COMMAND ----------

# DBTITLE 1,Step 6 — What is left in place
if not TEARDOWN:
    print(f"teardown=false — keeping {INDEX}; call teardown_index() or set teardown=true to remove it")

remaining = [i.name for i in w.vector_search_indexes.list_indexes(VS_ENDPOINT)]
print(f"Indexes still on {VS_ENDPOINT}: {remaining}")
print("The AI Search endpoint, the serving endpoint and the Delta table are left in place; this notebook "
      "never deletes them (reference §Step 6).")
