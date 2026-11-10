# Workflow: open-weight model embeddings → Databricks AI Search

A step-by-step reference for turning per-sample embeddings from an open-weight model into a searchable
index on Databricks, then finding the nearest neighbours of a new sample. **Databricks AI (artificial
intelligence) Search** is the product formerly called **Databricks Vector Search**; the Python software
development kit (SDK) still uses the `vector_search_*` names.

The worked example is **TEDDY-70M** single-cell embeddings (one vector per cell, looked up by cell type).
TEDDY stands for Transformer Encoder for DNA and scRNA-seq — deoxyribonucleic acid (DNA) and single-cell
ribonucleic acid sequencing (scRNA-seq) — and 70M is its 70-million-parameter variant (`teddy.md`
§Identity).
The same steps apply to any model in this skill that returns one vector per sample — the skill lists
TEDDY, Geneformer, Scimilarity and Midnight (`SKILL.md` §Key Concepts → Retrieval-augmented deployment).
Only the model-specific inputs change; those live in the model's reference file, never in this one.

**Plain-language decoder**

| Term | Meaning |
|---|---|
| Embedding | A fixed-length list of numbers the model produces for one sample (here: one cell). Similar samples get nearby lists. |
| Nearest neighbours | The stored samples whose embeddings are closest to a query embedding. |
| Unity Catalog (UC) | Databricks' governed namespace for tables, volumes and models: `<catalog>.<schema>.<name>`. |
| Volume | A UC folder for files, e.g. model weights: `/Volumes/<catalog>/<schema>/<volume>/…`. |
| Change Data Feed (CDF) | A Delta table setting that records row-level changes so the index can sync only what changed. |
| AI Search endpoint | The compute that hosts one or more indexes. Shared; can hold up to 50 indexes. |
| Delta Sync index | An index that copies rows (and their embeddings) from a Delta table and re-syncs when you ask it to. |
| Graphics processing unit (GPU) | Needed to run the model at a usable speed. NVIDIA T4 = 16 gigabytes (GB) of memory, NVIDIA A10G = 24 GB. A central processing unit (CPU) is the ordinary processor. |
| `GPU_SMALL` / `GPU_MEDIUM` | Model Serving GPU tiers: `GPU_SMALL` is a T4, `GPU_MEDIUM` an A10G (`teddy.md` §Compute requirements). |
| SDK | The `databricks-sdk` Python package (`WorkspaceClient`). |
| Identifier (ID) | The value that names one row, here one cell. |
| Out of memory (OOM) | The GPU ran out of memory — usually the batch was too large. CUDA is NVIDIA's GPU programming layer; its errors say `CUDA out of memory`. |
| Structured Query Language (SQL) | The query language used for Delta tables in Databricks. |
| L2 distance | Straight-line (Euclidean) distance between two embeddings. |
| Ensembl gene ID | A stable gene identifier from the Ensembl database; human ones start with `ENSG`. |
| Hypertext Transfer Protocol (HTTP) | The web request protocol a Model Serving endpoint is called over. |
| Inference controls | Per-request settings such as `max_seq_len`, sent as optional columns on the record (the standard) or under the request body's `"params"` key. |
| Held-out cells | Labelled cells kept out of the index, used to measure how often the neighbour vote is right. |
| Uniform Manifold Approximation and Projection (UMAP) | A method that draws high-dimensional embeddings as a 2-D map; nearby points have similar embeddings. |
| Wilson lower bound | The accuracy you can be 95% sure is not an overstatement, given how many cells it rests on. |
| Frozen embeddings | Embeddings from the model as published, with no training; a small classifier can still learn on top of them. |
| Fine-tuning | Training some of the model's own layers on your labels; it produces a different model. |
| Databricks AI Runtime | Serverless GPU training jobs submitted with the `databricks air` command-line interface. |

> **Status.** The code below follows the patterns of the validated TEDDY run
> (`eval/runs_20260924/oss-002_TEDDY-70M+VS_Deploy_withSkills_v2.py`), the canonical notebook
> (`eval/ground_truth/gt_teddy.py`) and the successful 2026-10-27 with-skill teddy-002
> evaluation run (round folder `eval/models/teddy/runs_20261026/`), with SDK names checked against `databricks-sdk`
> 0.67.0 (the AI Runtime v5 default), 0.122.0 (serverless environment v6) and 0.157.0. This page itself has
> **not yet been run end to end in a workspace**.
> The reusable code lives in this skill's **`scripts/`** folder — `vs_index.py` (AI Search endpoint and index),
> `nn_annotate.py` (embedding requests, neighbour vote, calibration, frozen-embedding classifier),
> `sample_checks.py` (donor cell-type mix, antibody-gene clone check) and `visualize.py` (figures) —
> unit-tested on made-up data in `tests/test_scripts_*.py`. The blocks below call
> them rather than repeating them. The dev-only check notebook `tests/check_embeddings_to_vector_search.py`
> runs these steps on a small cell set; run it before relying on this page for a workshop.

## What you build

```
model weights (UC Volume)
      │
      ▼
embeddings ── (a) Model Serving endpoint, small batches
      │    └─ (b) notebook / Job on a GPU, larger sets
      ▼
Delta table  <catalog>.<schema>.<model>_cells   (id, embedding, metadata; CDF on)
      │
      ▼
AI Search endpoint  <vs_endpoint>   (shared; create once, reuse)
      │
      ▼
Delta Sync index  <catalog>.<schema>.<model>_cell_index   (triggered sync)
      │
      ▼
query a new sample's embedding → top-k neighbours → e.g. majority cell type
```

## What you need from the model reference

Look these up in `references/models/<model>.md` before writing code. For TEDDY:

| Needed fact | TEDDY value | Source (`references/models/teddy.md`) |
|---|---|---|
| Request format | Three columns: `adata_sparsematrix`, `adata_obs`, `adata_var` — all required | §Serving contract |
| Inference controls | `{"max_seq_len": "2048", "pooling": "mean"}` as optional columns on the record (`route="column"`, the default; works with `serving_endpoints.query(dataframe_records=...)`), or under the REST body key `"params"` (`route="params"`, `nn.invoke()`) for endpoints logged before that standard — values as strings; the response echoes the values used. **Not** `serving_endpoints.query(extra_params=...)`, which custom PyFunc endpoints silently ignore | §Serving contract; §Sending inference controls |
| Response format | `predictions` → one `{"embedding": [...]}` per cell | §Output |
| Embedding dimension | `config.d_model` from `config.json` in the checkpoint directory (512 for 70M) | §Output; §AI Search index creation (dimension check) |
| Gene IDs | Ensembl IDs that are in `vocab.txt`; filter genes to the vocabulary before sending | §Census generation pipeline, steps 4–6 |
| Endpoint batch size | 10 cells per request on `GPU_SMALL` (T4) and on `GPU_MEDIUM` (A10G) for 70M at 2,048 genes; 8 for 400M (50 ran out of memory on an A10G) | §Compute requirements; §Deploy with Vector Search |
| Request size limit | 16 megabytes (MB) per request — another reason to filter genes | §Census generation pipeline, step 6 |
| Metadata worth indexing | `cell_type`, `tissue_general`, `disease` | §Census generation pipeline, step 8 |
| Naming | Tables/models/endpoints carry the variant suffix — variants are not interchangeable | §Table naming (variant-aware) |

For another model, fill the same table from its reference. If a fact is missing there, stop and find it
upstream; do not guess a dimension, batch size or input field.

## Step 0 — Prerequisites and compute

- **Workspace:** Unity Catalog enabled and serverless compute enabled (AI Search requirements,
  [AI Search docs](https://docs.databricks.com/aws/en/ai-search/ai-search/)).
- **Model already deployed or registered.** This page starts *after* the model is packaged. For TEDDY,
  `teddy.md` §Notebook architecture covers download → register → deploy. Path (a) needs a READY Model
  Serving endpoint; path (b) needs the registered UC model and the checkpoint directory in a Volume.
  If you still have to deploy, follow `teddy.md` §Deployment recommendation and §Registration and
  observability: the 70M variant fits `GPU_SMALL` with scale-to-zero, the endpoint gets an AI Gateway
  inference table plus usage tracking, and `EndpointCoreConfigInput` needs `name=` (required from
  `databricks-sdk` 0.71.0).
- **Privileges:** `USE CATALOG` / `USE SCHEMA`, `CREATE TABLE` on the target schema, `CAN_QUERY` on the
  serving endpoint, and permission to create indexes on the AI Search endpoint (see
  [Shared workspace](#shared-workspace-and-workshop-setup)).

**Compute for the notebook** (`eval/README_environments.md`):

| Option | What it is | When |
|---|---|---|
| Serverless GPU (A10G, AI Runtime v5) | What the validated runs used: Python 3.12, `databricks-sdk` 0.67.0, `mlflow` 3.16.1, `torch` 2.9.0 | Default. Needed for path (b); path (a) only needs CPU but runs fine here. |
| Classic GPU cluster (single-node A10G, Databricks Runtime for Machine Learning (ML) 15.x+) | Listed as an alternative, **not used in the eval runs** | When you need init scripts or a fixed runtime. |
| Serverless CPU | No GPU | Fine for path (a) and steps 2–6 if the embeddings come from an endpoint. |

Write temporary files to `/tmp`, not `/local_disk0` (`SKILL.md` §Best Practices 10).

**Configuration used by every block below** — set once, at the top of the notebook. Every value here is
a **proposal until the user confirms it** in step 0b; the defaults follow the `teddy.md` naming
convention (§Registration and observability) but must not be assumed.

```python
import sys
from databricks.sdk import WorkspaceClient

# This skill's scripts/ folder: personal install, or the workspace-wide one under /Workspace/.assistant/skills/.
SKILL_DIR = "/Workspace/Users/<user>/.assistant/skills/open-weight-models"
sys.path.insert(0, f"{SKILL_DIR}/scripts")
import vs_index as vx, nn_annotate as nn, visualize as viz

w = WorkspaceClient()

CATALOG          = "<catalog>"
SCHEMA           = "<schema>"
VOLUME           = "<volume>"
VS_ENDPOINT      = "<vs_endpoint>"        # AI Search endpoint (shared)
SERVING_ENDPOINT = "<serving_endpoint>"   # e.g. teddy-70m-embedder (teddy.md §Registration)
VARIANT          = "70m"
ENDPOINT_TYPE    = "STANDARD"             # AI Search endpoint type: "STANDARD" or "STORAGE_OPTIMIZED" (step 3)
GPU_TIER         = "GPU_SMALL"            # GPU running the model: "GPU_SMALL" (T4) or "GPU_MEDIUM" (A10G) (step 1)
CENSUS_VERSION   = "2025-01-30"           # CELLxGENE Census release (the 2026-10-27 teddy-002 run used it)
# Inference controls, as strings to match the signature's string defaults (teddy.md §Sending inference
# controls). If the registered version declares other types, send those (check its params schema).
TEDDY_PARAMS     = {"max_seq_len": "2048", "pooling": "mean"}

# Directory holding config.json + vocab.txt (layout: teddy.md §Checkpoint directory).
# The exact path depends on where the snapshot was downloaded (the oss-002 v2 run had an extra
# teddy/ level: .../snapshot/teddy/teddy/models/...). Check that config.json exists here.
MODEL_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/snapshot/teddy/models/teddy_g/70M"

UC_MODEL = f"{CATALOG}.{SCHEMA}.teddy_{VARIANT}"            # teddy.md §Registration and observability
UC_MODEL_VERSION = "<version>"                                 # the approved UC model version (step 0b, item 1)
TABLE    = f"{CATALOG}.{SCHEMA}.teddy_cells_{VARIANT}"
INDEX    = f"{CATALOG}.{SCHEMA}.teddy_cell_index_{VARIANT}"

# What to do with resources that already exist under these names. Set ONLY from the user's answer in
# step 0b: "stop" (default — raise and ask), "reuse", or "replace" (the only path that deletes anything).
ON_EXISTING = "stop"
# Creating the AI Search endpoint and index costs money: saved as False (SKILL.md Troubleshooting #14).
# If the request asked to build them, say so, set it to True for this run, then proceed.
RUN_GO = False
TEARDOWN_INDEX = False   # True only if the user asked for the index to be deleted at the end (step 6)
```

Read the embedding dimension from the checkpoint rather than hard-coding it (`teddy.md` §Output):

```python
import json, os

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    EMB_DIM = int(json.load(f)["d_model"])

with open(os.path.join(MODEL_DIR, "vocab.txt")) as f:
    VOCAB_GENES = {t.strip() for t in f if t.strip().startswith("ENSG")}
print(f"EMB_DIM={EMB_DIM}, vocab Ensembl IDs={len(VOCAB_GENES):,}")
```

## Step 0b — Confirm before building

**If uncertain, ask the user and wait; when proceeding with a default the user approved, state that
assumption explicitly.** Go through this list with the user before creating anything, asking only about
items you cannot determine from the request or the workspace:

| # | Confirm | Options and what it implies |
|---|---|---|
| 1 | Model, variant and version | The UC model **version** to use, and TEDDY 70M (validated) vs 160M / 400M (same reference, not yet tested; may need `GPU_MEDIUM` — `teddy.md` §Deployment recommendation). The variant fixes the embedding dimension. |
| 2 | Embedding route | An existing Model Serving endpoint (step 1a, small sets) vs batch in a notebook or Job (step 1b, larger sets, needs GPU compute). |
| 3 | Compute and cost | Serverless GPU vs classic GPU cluster; `GPU_SMALL` (T4) vs `GPU_MEDIUM` (A10G); both take 10 cells per request for 70M (8 for 400M). GPU compute costs more than CPU while it runs, and a GPU serving endpoint is the most expensive item to leave running (step 6). |
| 4 | Data source and size | CELLxGENE Census (which release; all tissues or a subset) vs a pre-staged `.h5ad` in a Volume; number of cells (`teddy.md` §Sizing guide: ~5 min per 1,000 cells on `GPU_MEDIUM`). |
| 5 | Names | UC model, serving endpoint, Delta table, AI Search endpoint, index. Offer the `teddy.md` defaults (`teddy_<variant>`, `teddy-<variant>-embedder`, `teddy_cells_<variant>`); there is no default AI Search endpoint name. |
| 6 | Existing resources | Report what the lookup below finds; for each, the user chooses **reuse**, **create under a new name**, or **replace**. |
| 7 | Teardown | What to delete at the end (step 6) and what to keep. |

**Look for existing resources (read-only).** A serving endpoint may already serve the same UC model
under another name; tables, indexes and AI Search endpoints may already exist under the proposed names.

```python
found = vx.existing_resources(w, uc_model=UC_MODEL, table=TABLE, index=INDEX, vs_endpoint=VS_ENDPOINT,
                              table_rows=lambda t: spark.table(t).count())
print("\n".join(found) or "Nothing exists under the proposed names.")
```

Report the list to the user and ask, per resource: **reuse** it, **create under a new name** (change the
name in step 0 and re-run the lookup), or **replace** it. Then set `ON_EXISTING` from the answer.
**Never delete, drop, overwrite or replace an existing endpoint, index, table or model version without the
user's explicit confirmation.** This workflow never deletes a Model Serving endpoint or a UC model
version, and never deletes a shared AI Search endpoint while it still hosts indexes.

Every later step goes through one guard, `vx.on_existing(kind, name, ON_EXISTING, details)`, so an existing
resource can never be changed by accident: with `ON_EXISTING = "stop"` it raises and tells you to ask the
user. An existing index is also checked with `vx.index_mismatches(...)`, which lists every reason it cannot
be reused (an empty list means it can):

```python
if spark.catalog.tableExists(TABLE):            # how step 2 uses it: raises unless the user made a choice
    vx.on_existing("Table", TABLE, ON_EXISTING, f"({spark.table(TABLE).count()} rows)")
```

An existing index is only reusable if it is backed by the same table, keyed on `cell_id`, and stores the
same vector column with the same dimension on the same endpoint, and (on a standard endpoint) syncs every
column later queries ask for (`VectorIndex.primary_key`, `delta_sync_index_spec.source_table`,
`.embedding_vector_columns` and `.columns_to_sync`, present in `databricks-sdk` 0.67.0, 0.122.0 and 0.157.0).
Otherwise the user chooses a new name or *replace*. Unity Catalog names are compared part by part with
`vx.same_uc_name()`, so a backtick-quoted name such as `` `my-catalog`.`schema`.`table` `` matches the
unquoted form the API returns.

## Step 1 — Get embeddings

Both paths start from the same input: an expression matrix with cells as rows, Ensembl gene IDs as
columns, filtered to the vocabulary, plus a stable ID and metadata per cell. With CELLxGENE Census as the
source, follow `teddy.md` §Census generation pipeline (organism key `homo_sapiens`, remap
`var_names` to `adata.var["feature_id"]`, intersect with the vocabulary, and install `cellxgene-census`
with `%pip` + `restartPython()`, not a subprocess). The check notebook has a working loader.

```python
import hashlib
import numpy as np, scipy.sparse

# adata: AnnData (annotated data matrix) with var_names = Ensembl IDs; obs has cell_type, tissue_general, disease
adata = adata[:, [g in VOCAB_GENES for g in adata.var_names]]
GENES = list(adata.var_names)
X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
X = X.astype(np.float32)

# Stable, globally unique IDs. Step 2 merges rows on cell_id alone, so an ID must never recur across
# sources — prefix it with the dataset it came from. Never use the row position.
#   Census: soma_joinid is only unique within one Census release, so include the release.
CELL_IDS = [f"census_{CENSUS_VERSION}_{j}" for j in adata.obs["soma_joinid"]]

#   A .h5ad file: obs_names recur across files, so include a hash of the file's contents.
def file_digest(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:12]
# CELL_IDS = [f"h5ad_{file_digest(H5AD_PATH)}_{n}" for n in adata.obs_names]

assert all(CELL_IDS) and len(set(CELL_IDS)) == len(CELL_IDS), "cell IDs must be non-empty and unique"
META = adata.obs[["cell_type", "tissue_general", "disease"]].astype(str).reset_index(drop=True)
```

### 1a — From an existing Model Serving endpoint (small sets)

Use this for up to a few thousand cells, or when the participant should not need a GPU. Confirm the
endpoint is ready first; a scaled-to-zero endpoint can take several minutes to wake up.

The batch size is capped by the endpoint's GPU tier: 10 cells per request on `GPU_SMALL` (T4) and on
`GPU_MEDIUM` (A10G) for 70M at 2,048 genes, 8 for 400M; larger batches run out of GPU memory (50 failed on an
A10G, 2026-10-29; `teddy.md` §Compute requirements). The block checks that the endpoint really runs on the tier you configured, and that it serves **exactly
the approved model version with 100% of traffic**: an endpoint splitting traffic between versions would
mix two embedding spaces in one index. Endpoints that report the older `served_models` field instead of
`served_entities` are checked the same way.

```python
from databricks.sdk.service.serving import EndpointStateReady

ep_info, workload = nn.serving_endpoint_check(w, SERVING_ENDPOINT, uc_model=UC_MODEL,
                                              uc_model_version=UC_MODEL_VERSION)
assert ep_info.state.ready == EndpointStateReady.READY, f"Endpoint not ready: {ep_info.state}"
assert workload == GPU_TIER, f"{SERVING_ENDPOINT} runs on {workload}, but GPU_TIER is {GPU_TIER}"

# One {"embedding": [...]} per cell, in order (teddy.md §Output). Controls go as optional columns on the record
# (route="column", the default; integration-contract.md §Standard); use route="params" only for endpoints logged
# before that standard. A response echo that differs from what was sent raises.
EMBEDDINGS = nn.embed_via_endpoint(w, SERVING_ENDPOINT, X, CELL_IDS, GENES, params=TEDDY_PARAMS, tier=GPU_TIER)
assert EMBEDDINGS.shape == (len(CELL_IDS), EMB_DIM)
```

`embed_via_endpoint` sends at most the tier's batch size (`nn.MAX_BATCH`), retries a failed batch one cell
at a time (for example after `CUDA out of memory`) and raises if a cell still fails, if the endpoint returns
a different number of predictions, or if an embedding is missing, non-finite or all zeros.

**Prove the controls reach the model.** A successful call does not show that a control was applied —
`serving_endpoints.query(extra_params=...)` also succeeds while being silently ignored (`SKILL.md`
Troubleshooting #13). Send one non-default value and require a different embedding, once per route
(`teddy.md` §Validation checklist, item 8, which also checks the version's params schema first):

```python
for route in ("column", "params"):   # integration-contract.md asks for both routes once per deployment
    nn.check_controls_applied(w, SERVING_ENDPOINT, X, CELL_IDS, GENES, params=TEDDY_PARAMS, tier=GPU_TIER,
                              route=route)
```

It embeds the cell with the most expressed genes twice — default controls, then `max_seq_len="32"` — and
raises if the two embeddings have cosine similarity of 0.99 or more. If the response echoes a control that
differs from what was sent, `embed_via_endpoint` raises `nn.ControlNotApplied` at once (no per-cell retry:
it is a settings problem, not a batch-size one); try the other route or check the signature.

Do not substitute a zero vector for a failed cell: it would be indexed and returned as a neighbour of
every other zero vector. Fix the failure or drop the cell and record it.

### 1b — Batch embedding in a notebook or Job (larger sets)

For tens of thousands of cells, or when no endpoint exists, load the registered model directly on a
GPU — the pattern used by `eval/ground_truth/gt_teddy.py` (Census cell). This needs the model's
pinned dependencies in the notebook environment (`teddy.md` §Dependencies: `transformers==4.41.0`,
`numpy<2`, `pandas<3`) and `mlflow`, installed with `%pip` followed by `dbutils.library.restartPython()`.

The same per-GPU limits apply when the model runs in your notebook. Set `GPU_TIER` to the class of GPU
the notebook has: `GPU_MEDIUM` needs an A10G-class GPU (24 GB; serverless GPU reports about
22 GB usable, `eval/README_environments.md`), otherwise use `GPU_SMALL`. Both are 10 cells per batch for
70M at 2,048 genes (`nn.MAX_BATCH`); use 8 for 400M.

```python
import mlflow, pandas as pd, torch

assert torch.cuda.is_available(), "Path 1b needs GPU compute"
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if GPU_TIER == "GPU_MEDIUM":
    assert gpu_gb >= 20, f"GPU_MEDIUM batches need an A10G-class GPU; this one has {gpu_gb:.1f} GB"
BATCH_SIZE = nn.MAX_BATCH[GPU_TIER]   # 10; up to 16 fits in a notebook A10G if no other model is loaded
                                      # (3,750 cells in 167 s at 16, 2026-10-30)

mlflow.set_registry_uri("databricks-uc")
model = mlflow.pyfunc.load_model(f"models:/{UC_MODEL}/{UC_MODEL_VERSION}")   # the approved version
VAR_JSON = pd.DataFrame({"index": GENES}).to_json(orient="split")

EMBEDDINGS = []
for start in range(0, len(CELL_IDS), BATCH_SIZE):
    ids = CELL_IDS[start:start + BATCH_SIZE]
    out = model.predict(
        pd.DataFrame({
            "adata_sparsematrix": [X[start:start + BATCH_SIZE].tolist()],   # one row = one batch
            "adata_obs": [pd.DataFrame({"cell_id": ids}).to_json(orient="split")],
            "adata_var": [VAR_JSON],
        }),
        params=TEDDY_PARAMS,   # same params mechanism as the endpoint, same types as the signature
    )
    # The wrapper may return a DataFrame or a list of dicts; normalise to a list of dicts.
    records = out.to_dict("records") if isinstance(out, pd.DataFrame) else list(out)
    EMBEDDINGS.extend([float(v) for v in r["embedding"]] for r in records)

EMBEDDINGS = np.asarray(EMBEDDINGS)        # cells x EMB_DIM, the same shape path 1a returns
assert EMBEDDINGS.shape == (len(CELL_IDS), EMB_DIM)
```

To run it unattended, schedule the same notebook as a Job task on GPU compute. Larger runs: `teddy.md`
§Sizing guide estimates ~45 minutes for 10,000 cells on a GPU; above that, split the input and write
each part to the table in turn (step 2 is safe to repeat).

### 1c — Hold out labelled cells

Keep some labelled cells **out of the index** so you can measure how well the neighbour lookup recovers
a known cell type and calibrate a confidence threshold ([Interpret and visualise the
results](#interpret-and-visualise-the-results)). The split is random but seeded (the same cells every
run) and **stratified**: every cell type with enough cells gives up the same share, as the 2026-10-27
with-skill teddy-002 run did. A type needs at least `nn.MIN_HELD_PER_TYPE + nn.MIN_REF_PER_TYPE` cells (5 by
default): fewer than that and it could not both be measured and stay findable. Such **rare types** stay
entirely in the reference — new cells can still be labelled with them — but they are left out of the
accuracy check and listed, so you know which labels are unmeasured. Only the reference cells go on to
step 2.

```python
# nn.HOLDOUT_FRACTION = 0.2 (share of each type held out), nn.MIN_HELD_PER_TYPE = 2, nn.MIN_REF_PER_TYPE = 3
_held, RARE_TYPES = nn.split_holdout(META["cell_type"])
HELD_IDS  = [c for c, h in zip(CELL_IDS, _held) if h]
HELD_EMB  = EMBEDDINGS[_held]
HELD_META = META[_held].reset_index(drop=True)
CELL_IDS   = [c for c, h in zip(CELL_IDS, _held) if not h]       # reference cells: written in step 2
EMBEDDINGS = EMBEDDINGS[~_held]
META       = META[~_held].reset_index(drop=True)
print(f"{len(CELL_IDS)} reference cells; {len(HELD_IDS)} held out across {HELD_META['cell_type'].nunique()} types")
if RARE_TYPES:
    print(f"Not measured (fewer than {nn.MIN_HELD_PER_TYPE + nn.MIN_REF_PER_TYPE} cells, reference only): {RARE_TYPES}")
```

## Step 2 — Write embeddings and metadata to a Delta table

The table is the source of truth; the index is a copy of it. Requirements and choices:

- **Change Data Feed on** — required for Delta Sync on standard endpoints (AI Search docs, Requirements).
- **Embedding column as an array of floats** — the documented format for pre-computed embeddings
  ([Create an AI Search index](https://docs.databricks.com/aws/en/ai-search/create-ai-search)).
- **Non-null, unique ID column** — it becomes the index primary key.
- **Upsert with `MERGE`**, so re-running adds or updates rows instead of dropping the table. `MERGE`
  still **overwrites rows that share a `cell_id`**, so an existing table is only written to after the
  user chose *reuse* (columns must match) or *replace* (drop and recreate — which also means rebuilding
  any index on it) in step 0b.

```python
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

EXPECTED_COLUMNS = ["cell_id", "embedding", "cell_type", "tissue_general", "disease"]
if spark.catalog.tableExists(TABLE):
    choice = vx.on_existing("Table", TABLE, ON_EXISTING, f"({spark.table(TABLE).count()} rows)")
    if choice == "reuse" and spark.table(TABLE).columns != EXPECTED_COLUMNS:
        raise RuntimeError(f"{TABLE} has columns {spark.table(TABLE).columns}, not {EXPECTED_COLUMNS}; "
                           "it cannot be reused — ask the user for a new name or replace")
    if choice == "replace":
        print(f"Replacing: DROP TABLE {TABLE}")
        spark.sql(f"DROP TABLE {TABLE}")

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
spark.sql(f"ALTER TABLE {TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")  # idempotent

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
print(spark.table(TABLE).count(), "rows in", TABLE)
```

**Optional — cosine similarity.** AI Search ranks by L2 distance. If you want cosine
similarity, normalise every embedding to length 1 before writing it **and** normalise each query vector
the same way; on unit-length vectors the two rankings are identical (AI Search docs, "How does AI Search
work?"). The validated TEDDY runs did not normalise.

## Step 3 — Create or reuse an AI Search endpoint

An endpoint is shared compute. In a shared workspace, **reuse** one rather than creating one per user:
each endpoint holds up to 50 indexes and a workspace up to 500 endpoints (AI Search docs, "Resource and
data size limits").

The two endpoint types need different index settings, so decide the type up front and refuse to reuse
an endpoint of the other type:

| | `STANDARD` | `STORAGE_OPTIMIZED` |
|---|---|---|
| Use for | Low-latency queries; the workshop default | Far more vectors at lower cost |
| `columns_to_sync` (step 4) | Supported | **Not supported** — every source column is synced |
| Sync mode | Triggered or continuous | Triggered only |
| Embedding dimension | Any (up to 4,096) | Must be divisible by 16 (512 is) |
| Query filters (step 5) | Dictionary, via `filters_json` | SQL-like strings |
| `databricks-sdk` 0.67.0 (AI Runtime v5 default) | Has `EndpointType.STANDARD` | No `EndpointType.STORAGE_OPTIMIZED` (0.157.0 has it) — upgrade with `%pip` |

Sources: [AI Search docs](https://docs.databricks.com/aws/en/ai-search/ai-search/), "Storage-optimized endpoints limitations" and "Resource and data size limits";
"Query an AI Search index" for filters.

```python
# Creates the endpoint only if it is missing AND RUN_GO is True; an existing one is reused (never deleted),
# and refused if it is the other endpoint type.
ep = vx.ensure_endpoint(w, VS_ENDPOINT, endpoint_type=ENDPOINT_TYPE, emb_dim=EMB_DIM, choice=ON_EXISTING,
                        run_go=RUN_GO)
```

## Step 4 — Create (or sync) the Delta Sync index

The index declares which column holds the vectors and their dimension. The dimension must equal the
model's embedding size; an existing index with a different dimension (for example after switching TEDDY
variants) cannot be reused (`teddy.md` §AI Search index creation, step 2) — the block stops and the user
chooses a new index name or *replace*. It never deletes an index the user has not agreed to replace.

**Triggered** sync means the index only refreshes when you call `sync_index` — predictable cost, good for
a reference set that changes in batches, and the only mode storage-optimized endpoints support. The block
**raises rather than query a stale or unready index**:

- A new index is created, then the block waits for `status.ready`.
- An existing index already reports `ready=True` while it serves its *previous* snapshot, so `ready` alone
  does not prove the new rows are in. `vx.sync_and_wait()` instead tracks the sync pipeline behind the index
  (`delta_sync_index_spec.pipeline_id`, read with `w.pipelines.list_updates`): it waits for any sync that
  is already running, requests a sync, and returns only once the pipeline update that request started
  has `COMPLETED`.
- No `sync_index` error is suppressed. The error returned when a sync is already running is not
  documented, and the SDK's index status has no sync timestamp (`VectorIndexStatus` has only `ready`,
  `indexed_row_count`, `message`, `index_url` in `databricks-sdk` 0.67.0, 0.122.0 and 0.157.0); the
  `databricks-ai-search` client's own wait helper reads a raw `status.detailed_state` that the SDK does
  not expose. Waiting for running updates first avoids that case instead of guessing at its message.
- Reading the pipeline's updates needs view access to the pipeline, which the index's creator has; this
  has not been tested for other users. Without it the block raises a `PermissionDenied` explanation
  rather than falling back to `status.ready`.
- Finally the block polls until the index's row count equals the table's, within the same 30-minute
  deadline (`vx.ensure_index(..., timeout_s=1800)`).
- `vx.ensure_index` creates nothing unless `RUN_GO` is True and never deletes: for *replace* the block
  deletes the index explicitly first, so the only delete is visible in the notebook.

```python
import time
from databricks.sdk.errors import NotFound

INDEX_COLUMNS = ["cell_id", "cell_type", "tissue_general", "disease"]   # columns the index returns (step 5)

# Replace = the user explicitly confirmed deleting the existing index (step 0b). The only delete here.
if ON_EXISTING == "replace":
    try:
        w.vector_search_indexes.get_index(INDEX)
        print(f"Replacing (confirmed by the user): delete_index({INDEX})")
        w.vector_search_indexes.delete_index(INDEX)
        deadline = time.time() + 900
        while True:                                      # wait until it is really gone
            try:
                w.vector_search_indexes.get_index(INDEX)
            except NotFound:
                break
            vx.sleep_until(deadline, 10, f"{INDEX} still exists after deletion")
    except NotFound:
        pass

status = vx.ensure_index(
    w, INDEX, endpoint=ep, table=TABLE, emb_dim=EMB_DIM, expected_rows=spark.table(TABLE).count(),
    columns_to_sync=INDEX_COLUMNS, choice="reuse" if ON_EXISTING == "replace" else ON_EXISTING, run_go=RUN_GO,
)
```

On a standard endpoint, only columns in `columns_to_sync` can be returned or filtered on (the primary key
and embedding column are always synced). When the user chose *replace*, the block deletes the index and
waits until it is really gone (`get_index` raises `NotFound`) before recreating it under the same name — the
canonical notebook waited a fixed 30 seconds instead. The validated run did not call `sync_index` after
`create_index`; it polled `status.ready` until the initial sync finished, which is what `ensure_index` does
for a new index.

## Step 5 — Query nearest neighbours and interpret them

Embed the new cell exactly as in step 1 (same model, gene filter and `TEDDY_PARAMS`, sent by the same
route: optional columns on the record by default, or the body `"params"` key with `route="params"`), then
query with the vector. Ask only for columns the index holds — on a `STANDARD`
endpoint that is `INDEX_COLUMNS` from step 4 (the 2026-10-27 teddy-002 run limited its query columns the
same way). Results come back as rows in `result.data_array`, columns in the order you asked for, with the
similarity **score appended as the last column**; there is no column-name attribute (`teddy.md` §AI
Search eval scorer, SDK note).

```python
assert w.vector_search_indexes.get_index(INDEX).status.ready, "Index not ready — run step 4 first"

# X_new: shape (1, len(GENES)) — same genes, same order as the table
query_vec = nn.embed_via_endpoint(w, SERVING_ENDPOINT, X_new, ["query_cell"], GENES,
                                  params=TEDDY_PARAMS, tier=GPU_TIER)[0]      # or the 1b loop for one cell

QUERY_COLUMNS = ["cell_id", "cell_type", "tissue_general"]
assert set(QUERY_COLUMNS) <= set(INDEX_COLUMNS), "query only columns the index syncs (step 4)"

hits = nn.query_neighbours(w, INDEX, query_vec, columns=QUERY_COLUMNS, k=10)   # dicts, plus "score"
for h in hits:
    print(f"{float(h['score']):.4f}  {h['cell_type']:<40} {h['tissue_general']:<15} {h['cell_id']}")
v = nn.vote(hits)
print(f"Suggested cell type: {v['predicted']} ({v['agreement']:.0%} of {len(hits)} neighbours agree)")
```

How to read the result:

- **Score:** higher means closer. It is **1 / (1 + squared L2 distance)**, not a cosine similarity; an exact
  self-match scores 1.0 (`teddy.md` §AI Search eval scorer). For unit-length embeddings the squared distance is
  2 − 2 × cosine, so score = 1 / (3 − 2 × cosine) and cosine = (3 − 1 / score) / 2 — a score threshold of 0.883
  is a cosine of about 0.93. (Checked 2026-10-28 in the TEDDY workshop test: neighbours recomputed by cosine on
  the saved embeddings, converted this way, matched the index's scores to within 0.00002.) The ranking is the
  same either way. Compare scores within one query, not across models.
- **Cell-type lookup:** a strong majority among the top 10 is a reasonable *suggested* label. A split
  vote, or low scores overall, means the reference set may not contain this cell type. This is a
  technical similarity lookup, not a validated annotation (`SKILL.md` §Key Concepts → Technical vs
  scientific validation).
- **Narrowing the search:** on a `STANDARD` endpoint, pass a dictionary filter on a synced column, e.g.
  `nn.query_neighbours(..., filters_json=json.dumps({"tissue_general": "<tissue>"}))`. Storage-optimized endpoints take SQL-like
  filter strings instead (AI Search docs, "Query an AI Search index").
- **Sanity check after every rebuild** (`teddy.md` §AI Search eval scorer): query with a cell that is
  already in the table, asking for at least 10 results (`num_results=1` missed the exact self-match for 3 of
  50 cells: approximate search explores fewer candidates); the top-1 result should be that cell, you should get exactly `num_results` rows,
  and scores should not increase down the list. `teddy.md` tolerates failures while the index is still
  syncing; this workflow waits for `status.ready` first (step 4), so any failure other than
  "not ready" is an error. A new index can still say "not ready" right after it first answered
  (2026-11-08), so pass a deadline and those are retried:
  `vx.index_sanity_check(w, INDEX, columns=QUERY_COLUMNS, cell_id=CELL_IDS[0], vector=EMBEDDINGS[0],
  deadline=time.time() + 900)`.

The same index is also queryable from SQL with the `vector_search()` function (AI Search docs, "Query an
AI Search index").

## Interpret and visualise the results

Turn neighbours into a label you can act on: the **vote label** is the most common cell type among the
top `nn.K` (10) neighbours, its **confidence** is the share of neighbours that agree (a vote share, not a
probability), and the top-1 score says how close the nearest reference cell is (higher is closer).
Calibrate both on the held-out cells from step 1c, then flag new cells that fall below them. Not yet run
in a workspace; the calibration, flag and map functions are exercised on made-up data by
`tests/test_scripts_nn_annotate.py` and `tests/test_scripts_visualize.py`.

**How the threshold is chosen.** Accuracy measured on a handful of cells is unreliable — one correct cell
is "100% accurate". So for each possible threshold the block counts the held-out cells at or above it
and computes the **Wilson lower bound** of their accuracy: the accuracy you can be 95% sure is not an
overstatement, given how many cells it rests on. The threshold is the lowest vote share whose lower bound
reaches `nn.TARGET` (0.90), and it must rest on at least `nn.MIN_ACCEPTED` (30) cells. Even with every cell correct, a 0.90
lower bound needs 35 accepted cells, so small runs usually end with **no threshold**: every new label is
then flagged as low confidence, which is the honest answer.

```python
from pyspark.sql import functions as F

# Held-out cells must not be in the index, or they find themselves and the accuracy is meaningless.
assert spark.table(TABLE).where(F.col("cell_id").isin(HELD_IDS)).count() == 0, "held-out cells are indexed"
held = nn.annotate(w, INDEX, HELD_EMB, columns=QUERY_COLUMNS, k=nn.K)   # predicted, agreement, top1_score, ...
held["true"] = HELD_META["cell_type"].to_numpy()
held["correct"] = held["predicted"] == held["true"]
print(f"Held-out accuracy: {held['correct'].mean():.1%} on {len(held)} cells "
      f"(chance about {1 / META['cell_type'].nunique():.1%}); not measured: {[t for t, _ in RARE_TYPES]}")

# nn.TARGET = 0.90 (accuracy as a 95% lower bound), nn.MIN_ACCEPTED = 30 held-out cells at least
THRESHOLD, CUTOFFS = nn.calibrate_vote_threshold(held["agreement"], held["correct"])
# "Far": the nearest reference match scores lower (is less close) than it does for 95% of held-out cells.
SCORE_FLOOR = nn.score_floor(held["top1_score"])
print(CUTOFFS.round(3).to_string(index=False))
print(f"Threshold: {THRESHOLD} (needs >= {nn.MIN_ACCEPTED} accepted cells and a lower bound >= {nn.TARGET:.0%}); "
      f"score floor: {SCORE_FLOOR}")

# Label new cells. QUERY_EMB: their embeddings, made as in step 5.
new = nn.flag_cells(nn.annotate(w, INDEX, QUERY_EMB, columns=QUERY_COLUMNS, k=nn.K), THRESHOLD, SCORE_FLOOR)
print(new["flag"].replace("", "trusted").value_counts())
```

Read it with care: held-out cells come from the same datasets as the reference, so their accuracy is
an upper bound for cells from another lab, assay or tissue. A "far" cell may be a cell type the reference
lacks, a doublet or a low-quality cell — inspect it rather than trusting its label. Labels of rare types
were never measured. Report the threshold together with the number of cells it rests on (`CUTOFFS`).

**Figures** (matplotlib, via `visualize.py`). The UMAP needs `%pip install umap-learn`, then
`dbutils.library.restartPython()`. `umap-learn` 0.5.12 raises on fewer than 4 reference cells (measured), so
`viz.map_2d` draws a principal component analysis (PCA) map instead below 10. The map is fitted on the
reference only and new cells are placed onto it, so they never move the reference. Colours follow the
`scientific-figures` skill (`references/color.md`: at most 6 hues, the rest grey); each figure is saved as
PDF (for reports) and PNG (for slides). The helpers return figures detached from pyplot, so in a notebook show
each with `display(fig)` (it then appears once; `plt.show()` would draw nothing):

```python
OUT = "/tmp/figures"

# 1. Map of the reference with the new cells on top. Colour the types that matter for the question (here the
#    new cells' predicted types, at most 5; the rest grey). Not "the most common": on a balanced reference
#    that is an arbitrary tie-break. To colour more types, pass families= (related types as shades of one
#    colour, e.g. {"navy": [4 interneuron subtypes], "orange": ["oligodendrocyte", "oligodendrocyte precursor cell"]}).
ref_xy, new_xy, method = viz.map_2d(EMBEDDINGS, QUERY_EMB)
shown = list(new["predicted"].value_counts().index[:5])   # the new cells' most frequent predicted types
other = sorted(set(META["cell_type"]) - set(shown))
fig = viz.plot_reference_map(ref_xy, META["cell_type"], shown, new_xy, method=method,
                             note=f"{len(ref_xy)} reference cells, {len(new_xy)} new cells; map fitted on the reference only. "
                                  f"Grey: {', '.join(other)}")
viz.save_figure(fig, OUT, "reference_map")

# 2. Held-out confidence, correct vs wrong, with the calibrated threshold (None draws no line)
fig = viz.plot_vote_confidence(held["agreement"], held["correct"], k=nn.K, threshold=THRESHOLD,
                               note=f"{len(held)} held-out cells; threshold rests on CUTOFFS above")
viz.save_figure(fig, OUT, "heldout_confidence")

# 3. Held-out confusion matrix, rows normalised: the diagonal is per-type accuracy
cm, labels = viz.confusion_fraction(held["true"], held["predicted"])
fig = viz.plot_confusion(cm, labels, note="Rows sum to 1; off-diagonal cells are mix-ups")
viz.save_figure(fig, OUT, "heldout_confusion")
print(held.groupby("true")["correct"].agg(accuracy="mean", cells="size").sort_values("accuracy"))
```

For figure sizing, fonts and export, use the `scientific-figures` skill (`references/matplotlib-seaborn.md`).
For follow-up single-cell analysis of the labelled cells — clustering, marker genes, comparing
conditions with scanpy — use the `sc-rnaseq` skill (`references/scanpy-workflow.md`).

## Check the sample, not just the cells

The flag rule asks one question per cell: "does it look like some reference cell?". Two kinds of unusual pass
it, so check the **donors** too (`scripts/sample_checks.py`, unit-tested in `tests/test_scripts_sample_checks.py`):

- **An unusual mix.** Every cell looks normal, but a donor has far more (or fewer) cells of one type than any
  healthy donor. Compare each new donor's share of each broad cell type with healthy donors at their **natural
  proportions** — the reference is capped per cell type, so it cannot be used for this; read the healthy
  donors' labels from the source dataset (for CELLxGENE Census, `cellxgene_census.get_obs`, no expression
  needed). An unusual mix is a reason to look, not a finding: in the TEDDY workshop's blood sample one donor's
  monocyte excess (56% vs at most 35% in 29 healthy donors) is ordinary disease biology, while another donor's
  B-cell excess turned out to be one clone.
- **What the model cannot see.** A model reads only the genes in its vocabulary. TEDDY-70M's vocabulary holds
  3 of the 330 antibody variable genes and none of the 17 antibody constant genes in that dataset, so one
  clone of B cells (all making the same antibody) embeds as ordinary B cells: 47 of 55 such cells passed the
  flag rule. The raw counts still hold those genes. A clone shows **one heavy-chain variable (V) gene in most
  cells and only one light-chain type** (kappa or lambda); a normal mix of B cells shows many V genes (the most
  common one in about 12% of cells) and both light-chain types.

```python
import sample_checks as sx

RULES = [("monocyte", ["monocyte"]), ("dendritic cell", ["dendritic"]), ("B / plasma cell", ["b cell", "plasma"]),
         ("T cell", ["t cell"])]                      # ordered keywords, matched at a word start; dataset-specific
# new: the labelled new cells from above. QUERY_DONORS: each new cell's donor (same order). HEALTHY_OBS: one row
# per healthy cell of the source dataset (donor_id, cell_type). RAW_COUNTS / GENE_SYMBOLS: the new cells' raw
# counts (all genes, same order) and the gene symbol of each column.
broad = [sx.broad_label(l, RULES) for l in new["predicted"]]
new_mix = sx.composition(QUERY_DONORS, broad)
healthy_mix = sx.composition(HEALTHY_OBS["donor_id"], [sx.broad_label(l, RULES) for l in HEALTHY_OBS["cell_type"]])
check = sx.donor_composition_check(new_mix, healthy_mix, min_cells=50)   # "above/below healthy range", "too few cells"
display(check[check["unusual"] != ""])

# Antibody genes from the raw counts, for the cells voted into the B lineage, per donor
b = np.array(broad) == "B / plasma cell"
usage = sx.antibody_usage(RAW_COUNTS[b], GENE_SYMBOLS, np.asarray(QUERY_DONORS)[b])
display(usage)          # top_heavy_v_share near 1 and kappa_share 0 or 1 = one clone
fig = viz.plot_antibody_usage(usage, highlight=list(usage.query("top_heavy_v_share > 0.5")["group"]))
```

`viz.plot_donor_composition(check)` draws each donor's shares against the healthy range. When comparing one
donor's cells with another donor's, set aside genes that mostly separate donors (`sx.donor_effect_genes`:
ribosomal, mitochondrial, sex-linked) before reading the marker list.

## Optional — adapt the model: a classifier on frozen embeddings, then fine-tuning

The neighbour vote needs no training. Two ways to adapt the model to the reference labels, from cheapest to
most expensive — always measured on the **same held-out cells** as the vote:

**1. A classifier on the frozen embeddings** (central processing unit (CPU), seconds). The encoder is not
changed ("frozen"); a logistic-regression layer learns the labels from the reference embeddings:

```python
table, clf_pred, clf = nn.compare_with_classifier(EMBEDDINGS, META["cell_type"], HELD_EMB,
                                                  HELD_META["cell_type"], held_vote=held["predicted"])
print(table)                     # method, cells, correct, accuracy and its 95% Wilson interval
fig = viz.plot_accuracy_comparison(table, note="Same held-out cells; bars show 95% Wilson intervals")
viz.save_figure(fig, OUT, "vote_vs_classifier")
```

If the two intervals overlap, the data do not show one method is better. The classifier gives one label
per cell with no "far from the reference" signal, so keep the vote's flags for triage.

**2. Fine-tuning TEDDY on Databricks AI Runtime** (one A10 GPU; **not upstream-validated**). TEDDY's
repository says its models can be fine-tuned and ships a classification head
(`teddy/models/classification_heads.py:70-88`, attached with `TeddyGModel.add_classification_head`,
`teddy/models/teddy_g/model.py:158-163`, revision `8c880aa`) but no fine-tuning script, so the training loop
in `scripts/finetune_teddy/train_teddy_classifier.py` is this skill's own. It uses upstream's
preprocessing, tokenization, model class, weights and head unchanged; checks that a cell's frozen embedding
is the same alone or padded in a batch (cosine >= 0.9999); trains the head plus the last `unfreeze_layers`
encoder layers; and reports, on the same held-out donors, the neighbour vote and logistic regression on
frozen embeddings next to the fine-tuned accuracy. Submit it with the AI Runtime command-line interface
(`databricks air`, the `databricks-ai-runtime` skill):

```bash
cd scripts/finetune_teddy                      # or a copy next to your notebook; fill the <placeholders>
python train_teddy_classifier.py --params params.yaml --smoke --no-mlflow    # optional CPU smoke test
databricks air run --file workload.yaml --dry-run -o json -p <profile>        # validates, launches nothing
databricks air run --file workload.yaml --idempotency-key <user>-teddy-finetune-<YYYYMMDD>-v1 -o json -p <profile>
```

Submit only after the user confirms (GPU time costs money); `max_retries: 0` means a failure stops and is
read from `databricks air logs <run_id>` rather than retried. A fine-tuned model is a **different model**:
register it (if at all) under its own Unity Catalog name, and never write its embeddings into an index
built from the original model. Training time on an A10 has not been measured.

## Write it up — linking skills into one workflow

This page is one step of a longer scientific workflow. The repo's other skills pick up where it ends:

| Step | Skill | What it adds |
|---|---|---|
| Embed, index, label, flag (this page) | `open-weight-models` | Embeddings, AI Search index, neighbour vote, calibrated flags |
| Look at the flagged cells | `sc-rnaseq` | Quality control (doublets, mitochondrial share), marker genes with scanpy |
| What the marker genes mean | `pathway-enrichment-analysis` | Over-representation analysis (Enrichr) or GSEA on the marker lists |
| Publication-ready figures | `scientific-figures` | Sizes, fonts, colour rules, PDF + PNG export (`visualize.py` already follows them) |
| A written report | `report-builder` | LaTeX methods/results/discussion compiled to PDF (downloads the Tectonic engine — asks first) |
| Slides | `presentation-builder` | A shareable HTML deck whose numbers are pulled from Unity Catalog tables by its `ground.py` |

To make the last two work, save every result you want to cite to a Unity Catalog table (held-out accuracy,
calibration, labels and flags of the new cells, triage decisions) and the figures to a Volume. The deck
builder only puts numbers on a slide that it read from a table, and asks about protected health information
(PHI) when its de-identification partner skill is not installed — say in the request that the data are public,
non-patient data when that is true (for example CELLxGENE Census).

## Step 6 — Teardown and cost

| Resource | Costs while it exists? | Action |
|---|---|---|
| Model Serving endpoint (GPU) | Yes — highest cost; scale-to-zero only reduces idle cost | Delete after the session, or rely on `scale_to_zero_enabled=True` (`teddy.md` §Compute requirements) |
| AI Search index | Counts toward the endpoint's size | Delete when done |
| AI Search endpoint | Billed only once it hosts an index; stops billing 24 hours after its last index is deleted ([AI Search cost management](https://docs.databricks.com/aws/en/ai-search/cost-management)) | Delete only if no one else's indexes use it |
| Delta table, UC model, Volume files | Storage only | Usually keep |

Delete only what the user confirmed in step 0b (item 7).

```python
from databricks.sdk.errors import NotFound

if TEARDOWN_INDEX:                       # set in step 0 only from the user's answer to item 7
    print(f"Teardown (confirmed by the user): delete_index({INDEX})")
    try:
        w.vector_search_indexes.delete_index(INDEX)
        print(f"Deleted {INDEX}")
    except NotFound:
        print(f"{INDEX} already gone")
else:
    print(f"Keeping {INDEX} (teardown not confirmed)")

# The AI Search endpoint is shared: delete it only when no indexes remain AND the user confirms.
remaining = [i.name for i in w.vector_search_indexes.list_indexes(VS_ENDPOINT)]
print(f"Indexes still on {VS_ENDPOINT}: {remaining}")
# if not remaining and <user confirmed deleting the shared endpoint>:
#     w.vector_search_endpoints.delete_endpoint(VS_ENDPOINT)
```

## Shared workspace and workshop setup

For a hands-on session (for example 15–30 participants on serverless GPU) the aim is that every
participant does steps 2–5 themselves, while the slow, internet-dependent and costly parts are done once
by the instructor.

**Facilitators pre-create the shared resources; participants reuse them.** The facilitator creates the
AI Search endpoint `<vs_endpoint>` and the Model Serving endpoint before the session and confirms their
names once (step 0b). Participants run with `ON_EXISTING = "reuse"` for these shared resources — the
step 0b lookup will find them, and *reuse* is the agreed answer, so no one is asked again or offered
*replace*. Per-participant tables and indexes still use the collision-safe names below.

**One shared AI Search endpoint, one index per participant.** One endpoint holds up to 50
indexes, so 30 participants fit on one; above 50, create a second endpoint and assign half the room to
it. Give each participant their own table and index name so nobody overwrites anyone else:

```python
import hashlib, re

USER_NAME = w.current_user.me().user_name
_readable = re.sub(r"[^a-z0-9]", "_", USER_NAME.split("@")[0].lower())[:20]   # for humans only
_digest = hashlib.sha1(USER_NAME.encode()).hexdigest()[:8]                     # uniqueness
USER_PREFIX = f"u_{_readable}_{_digest}"    # always starts with a letter
TABLE = f"{CATALOG}.{SCHEMA}.{USER_PREFIX}_teddy_cells_{VARIANT}"
INDEX = f"{CATALOG}.{SCHEMA}.{USER_PREFIX}_teddy_cell_index_{VARIANT}"
```

The readable part alone is not unique — `a-b@…` and `a_b@…`, or the same name at two email domains,
normalise to the same text — so the hash of the **full** user name is what keeps names apart. Index names
cannot contain spaces, periods or slashes beyond the `catalog.schema.name` separators (AI Search docs,
"Create an AI Search index"); the prefix uses only lowercase letters, digits and underscores.

**Pre-staged inputs in a Unity Catalog Volume** (needed when the workspace has restricted internet
access — the Hugging Face download and Census both need it):

| Item | Prepared by the instructor | Participants |
|---|---|---|
| Model weights + `vocab.txt` + `config.json` | Download once into `/Volumes/<catalog>/<schema>/<volume>/snapshot/...` (`teddy.md` §Hugging Face download) | Read only (`READ VOLUME`) |
| Model Serving endpoint | Deploy once; warm it up before the session | `CAN_QUERY` |
| Reference cells | Pre-embed a reference set once into a shared table, **or** save a small AnnData file (`.h5ad`) of cells to the Volume | Copy the shared table into their own (`CREATE TABLE … AS SELECT`, then enable CDF), or embed a small sample (50–200 cells) themselves via path 1a |
| AI Search endpoint | Create once (step 3); grant the participant group `CAN_USE` | Create their own index on it |

Endpoint permissions can be set in the workspace user interface (UI) or with the SDK. The permission
methods exist in `databricks-sdk` 0.122.0 and 0.157.0 but **not** in 0.67.0 (the AI Runtime v5 default), so run this
from an upgraded SDK (`%pip install -U databricks-sdk`, then restart Python) or use the UI:

```python
from databricks.sdk.service.vectorsearch import (
    VectorSearchEndpointAccessControlRequest,
    VectorSearchEndpointPermissionLevel,
)

ep = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT)
w.vector_search_endpoints.update_permissions(   # adds to existing grants; set_permissions replaces them
    ep.id,
    access_control_list=[
        VectorSearchEndpointAccessControlRequest(
            group_name="<participant_group>",
            permission_level=VectorSearchEndpointPermissionLevel.CAN_USE,
        )
    ],
)
```

**Letting others query an index.** Endpoint `CAN_USE` is not enough: querying an index needs `USE CATALOG`,
`USE SCHEMA` and `SELECT` on the index itself, which is a Unity Catalog table-type securable, so the grant is
`GRANT SELECT ON TABLE <index>` ([Query an AI Search index](https://docs.databricks.com/aws/en/ai-search/query-ai-search)).
`vx.grant_select(spark.sql, [TABLE, INDEX], "<participant_group>")` grants both (idempotent).

**Load on the shared serving endpoint.** Thirty people embedding at once all hit one GPU endpoint, which
processes 10 cells per request on `GPU_SMALL`. Keep participant embedding small (a few hundred cells
each), pre-embed the bulk of the reference set, and keep the endpoint warm during the session (a
scaled-to-zero endpoint can take minutes to wake). Throughput limits for this case have not been
measured.

**Clean up at the end:** each participant deletes their index (step 6); the instructor deletes the
serving endpoint and, once all indexes are gone, the AI Search endpoint.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` from the endpoint | Batch too large for the GPU tier | 10 cells per request on T4 and on A10G for 70M at 2,048 genes, 8 for 400M (`nn.MAX_BATCH`; `teddy.md` §Census generation pipeline, step 7) |
| Request rejected as too large | All genes sent | Filter to vocabulary genes first (16 MB limit, same section, step 6) |
| "controls are not reaching predict()" in step 1a, or changing `max_seq_len` changes nothing | Controls sent with `serving_endpoints.query(extra_params=...)`, or missing from the signature (optional columns and params schema) | Send them by either route of `nn.embed_via_endpoint()`: as optional columns on the record (`route="column"`, default) or under the body `"params"` key (`route="params"`, `nn.invoke()`); run `nn.check_controls_applied` once per route (`teddy.md` §Sending inference controls); declare every control in the signature (`SKILL.md` Troubleshooting #13). `nn.ControlNotApplied` means the response echo showed a different value: try the other route |
| Step 1a stops with "runs on … but GPU_TIER is …" | `GPU_TIER` does not match the endpoint | Set `GPU_TIER` to the endpoint's workload type |
| Step 1a stops with "serves N entities", "not the approved …" or "routes traffic as …" | Endpoint serves another model or version, or splits traffic | Confirm the model version with the user; use an endpoint that sends 100% of traffic to that version |
| Step 4 stops with "cannot be reused: …" | Existing index has another source table, key, vector column or endpoint | Ask the user for a new index name or *replace* |
| Step 4 raises about view access to the sync pipeline | No permission to read the index's pipeline updates | Ask the index owner or facilitator for access |
| Step 3 stops with "is … but ENDPOINT_TYPE is …" | Reused endpoint is the other type | Pick a matching endpoint, or change `ENDPOINT_TYPE` |
| Step 4 raises `TimeoutError` | Index still syncing after 30 minutes | Check `status.message`; re-run step 4 to keep waiting |
| Embeddings all look the same | Gene IDs did not match the vocabulary (all unknown tokens) | Use Ensembl IDs; check the overlap count (`teddy.md` §Input example) |
| Index creation fails on the source table | Change Data Feed not enabled | `ALTER TABLE … SET TBLPROPERTIES (delta.enableChangeDataFeed = true)` |
| Query fails with a dimension error, or a variant switch breaks the index | Index dimension ≠ embedding length | Ask the user: new index name, or replace (step 0b, step 4) |
| `RuntimeError: … already exists` | A resource exists and no choice was made | Report it to the user; set `ON_EXISTING` from their answer (step 0b) |
| Query returns fewer rows than asked, or self-match is not top-1 | Index not ready, or duplicate embeddings in the table | Confirm `status.ready`; check for identical embeddings |
| New rows do not appear | Triggered index not re-synced | Run step 4 again with `ON_EXISTING = "reuse"` (`vx.ensure_index` calls `vx.sync_and_wait`, which waits for the requested sync to complete) |
| `RuntimeError: … does not exist … pass run_go=True` | Step 3 or 4 would create a billable resource and `RUN_GO` is False | Confirm with the user (or, if the request asked to build it, say so), set `RUN_GO = True` for this run, re-run the step |
| `ModuleNotFoundError: No module named 'vs_index'` | `scripts/` is not on `sys.path` | Set `SKILL_DIR` to where the skill is installed (step 0) and re-run the configuration cell |
| Fine-tuning run fails on AI Runtime | Config, dependency or data error | `databricks air logs <run_id>`; fix and resubmit with a new idempotency key (`max_retries: 0` never retries by itself) |
| Step 4 raises from `sync_index` or "requested sync did not complete" | The sync was rejected or failed | Read the error / pipeline update; nothing was suppressed — ask the user before retrying |
| `AttributeError: 'str' object has no attribute 'value'` when passing `"STANDARD"` / `"DELTA_SYNC"` | SDK expects enums | Use `EndpointType.STANDARD`, `VectorIndexType.DELTA_SYNC`, `PipelineType.TRIGGERED` (`SKILL.md` §Best Practices 13) |

## Sources

- `references/models/teddy.md` — §Serving contract, §Output, §HTTP payload, §Sending inference
  controls, §Validation checklist, §Dependencies, §Compute requirements, §Deployment recommendation,
  §Registration and observability, §Table naming, §Census generation pipeline, §Sizing guide, §AI
  Search index creation, §AI Search eval scorer.
- `SKILL.md` Troubleshooting #13 — why `extra_params` never reaches a custom PyFunc; #14 — the `run_go` gate.
- `scripts/vs_index.py`, `scripts/nn_annotate.py`, `scripts/sample_checks.py`, `scripts/visualize.py`,
  `scripts/finetune_teddy/` — the code
  this page calls, with CPU unit tests in `tests/test_scripts_*.py` and `tests/test_finetune_teddy.py`.
- TEDDY upstream at Hugging Face revision `8c880aa67674042ccdc254bccb5de081092a48bf` (read-only):
  `teddy/models/teddy_g/model.py:158-163, 197-264, 347-350` (classification head, forward, `cell_emb`),
  `teddy/models/classification_heads.py:70-88`, `README.md:83, 284-286` (fine-tuning mentioned, no script).
- [Databricks AI Runtime command-line interface](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/yaml-config) —
  `workload.yaml` fields (checked with `databricks air run -h config`, CLI v1.20.0).
- `eval/runs_20260924/oss-002_TEDDY-70M+VS_Deploy_withSkills_v2.py` — validated end-to-end run
  (endpoint batch embedding, Delta table, AI Search endpoint + index, evaluation).
- `eval/ground_truth/gt_teddy.py` — canonical notebook (local batch embedding, index
  create/sync with dimension check, teardown).
- `eval/models/teddy/runs_20261026/` — the evaluation round named 2026-10-26; its with-skill teddy-002
  run (2026-10-27) used Census 2025-01-30, `GPU_SMALL` serving with an inference table and usage
  tracking, and a held-out nearest-neighbour cell-type accuracy check.
- `eval/README_environments.md` — compute and package versions.
- [Databricks AI Search](https://docs.databricks.com/aws/en/ai-search/ai-search/) ·
  [Create an AI Search index](https://docs.databricks.com/aws/en/ai-search/create-ai-search) ·
  [Query an AI Search index](https://docs.databricks.com/aws/en/ai-search/query-ai-search) ·
  [AI Search cost management](https://docs.databricks.com/aws/en/ai-search/cost-management).
