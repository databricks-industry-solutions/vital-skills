# TEDDY model reference

## Identity

* Model family: TEDDY (Transformer Encoder for DNA and scRNA-seq)
* Upstream code and weights URL: https://huggingface.co/Merck/TEDDY
* Model card: https://huggingface.co/Merck/TEDDY
* Paper: arXiv:2503.03485
* Reviewed date: 2026-09-05
* Upstream revision: pinned via `snapshot_download` — record the commit SHA from `.gitattributes` after download
* License (checked read-only at HF revision `8c880aa67674042ccdc254bccb5de081092a48bf`): the model card (`README.md` front matter) declares `license: apache-2.0`; the bundled Python package's `pyproject.toml` declares `license = "MIT"`; the repo has no `LICENSE` file. Both are permissive.
* License handling: record **both** declarations plus the revision in the provenance manifest and MLflow tags, and ask the user to confirm with the model owner before redistribution — do not state which one governs.
* Variants: 70M, 160M, 400M (parameter count)

## What this model does

TEDDY-G is a transformer encoder foundation model pre-trained on 116 million single-cell RNA-seq (scRNA-seq) cells. It produces per-cell embedding vectors from gene expression profiles. Upstream preprocessing scales each cell to 10,000 total counts and divides each gene by its TEDDY gene median; the model then keeps only vocabulary genes, selects the top-K most expressed genes per cell, rank-encodes their expression values as linearly spaced floats from 1.0 (highest) to −1.0 (lowest), and passes the sequence through a transformer encoder. The resulting embeddings are used for cell-type annotation, batch correction, perturbation prediction, and cross-dataset transfer. This reference covers **online inference** (bounded per-cell queries to a Model Serving endpoint); for large-batch embedding of AnnData objects use a Job.

## Inputs and outputs

### Serving contract

The model does not accept a raw AnnData object. Transport as three DataFrame columns per row; all three are **required** by the MLflow signature and must be present in every request payload, even if `adata_obs` is not used by the model internals.

| Column | Type | Description |
|--------|------|-------------|
| `adata_sparsematrix` | `list[list[float]]` | Dense expression matrix, shape `(cells, genes)`. One sublist per cell; values are **raw counts** — the wrapper applies TEDDY's normalisation (see [Top-K selection and rank encoding](#top-k-selection-and-rank-encoding)). |
| `adata_obs` | `str` — JSON orient=`split` | Cell metadata DataFrame. Required by the signature; not consumed by `predict`. Include at minimum a `cell_id` column. |
| `adata_var` | `str` — JSON orient=`split` | Gene metadata DataFrame. Must have an `index` column containing Ensembl gene IDs (`ENSGXXXXXXXXXXX`) that match `vocab.txt`. |

Inference controls go in the REST request body's `"params"` field (endpoint) or `params=` (MLflow local `predict`). Do **not** use the SDK's `serving_endpoints.query(extra_params=...)`: the endpoint silently ignores it (see [Sending inference controls](#sending-inference-controls) and SKILL.md Troubleshooting #13).

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `max_seq_len` | int (sent as `str`; declare a `str` default in the signature, cast in `predict`) | 2048 | Top-K genes selected per cell. |
| `pooling` | str | `"mean"` | `"mean"` pools over all token positions; `"cls"` uses the CLS token (only valid when `config.add_cls=True`). The 70M forward pass returns `cell_emb`, already one vector per cell (the encoder output at the first position), so the wrapper below skips pooling and this control has no effect there (see [Wrapper boundary](#wrapper-boundary)). |

### Output

```json
{
  "predictions": [
    {"embedding": ["<float>", "..."]}
  ]
}
```

One dict per input cell. Embedding dimension equals `config.d_model` (read from `config.json` in the checkpoint directory).

### Input example (Python — use real vocab gene IDs)

```json
{
  "adata_sparsematrix": [[12.0, 5.0, 3.0]],
  "adata_obs": "<json-string-orient=split-cell-metadata>",
  "adata_var": "<json-string-orient=split-gene-metadata-with-index>"
}
```

```python
import pandas as pd
import numpy as np

# Read real Ensembl IDs from vocab.txt — do NOT invent gene names
vocab_path = f"{model_dir}/vocab.txt"
with open(vocab_path) as f:
    vocab_tokens = [t.strip() for t in f if t.strip()]
real_genes = [t for t in vocab_tokens if not t.startswith("<")][:100]  # skip special tokens

n_cells, n_genes = 5, len(real_genes)
expr = np.random.default_rng(42).poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)

obs_df = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_cells)]})
var_df = pd.DataFrame({"index": real_genes})

input_example = pd.DataFrame({
    "adata_sparsematrix": [expr.tolist()],           # one row = one batch
    "adata_obs":          [obs_df.to_json(orient="split")],
    "adata_var":          [var_df.to_json(orient="split")],
})
```

**Critical**: always use real Ensembl IDs from `vocab.txt` when constructing test payloads. Synthetic gene names are not in the vocabulary, so upstream TEDDY drops them (see [Out-of-vocabulary genes](#out-of-vocabulary-oov-genes-drop-them-as-upstream-does)) and nothing real is tokenised — the test cannot validate the tokenisation path.

### HTTP payload (Databricks SDK)

Send inference controls under the request-body key `"params"`, **not** with `w.serving_endpoints.query(extra_params=...)`: the SDK writes `extra_params` to a body key `"extra_params"` that the MLflow scoring server drops, so the call succeeds but `predict()` runs with defaults (measured 2026-10-27 on a TEDDY endpoint: `extra_params` gave embeddings identical to the default, cosine 1.0; body `"params"` changed them, cosine 0.53–0.64).

```python
records = [{
    "adata_sparsematrix": expr.tolist(),
    "adata_obs":          obs_df.to_json(orient="split"),
    "adata_var":          var_df.to_json(orient="split"),
}]
response = invoke(w, ENDPOINT_NAME, records, params={"max_seq_len": "2048", "pooling": "mean"})  # w = WorkspaceClient(); invoke() is defined in "Sending inference controls"
embeddings = [p["embedding"] for p in response["predictions"]]
```

`invoke()`, the equivalent HTTP request, and the type rules are in [Sending inference controls](#sending-inference-controls). Values are strings; the PyFunc wrapper casts them: `int(params.get("max_seq_len", 2048))`, `str(params.get("pooling", "mean"))`.

## Artifacts and runtime

### Checkpoint directory (`model_dir`)

Path: `<VOLUME_PATH>/snapshot/teddy/models/teddy_g/<VARIANT>/`

| File | Size (70M) | Purpose |
|------|------------|----------|
| `model.safetensors` | ~285 MB | Model weights |
| `vocab.txt` | ~756 KB | 43,808 lines (70M, HF revision `8c880aa`): 25,424 Ensembl gene IDs plus special tokens, one per line; used by `GeneTokenizer.from_pretrained()` |
| `config.json` | — | Architecture config (`d_model`, `add_cls`, `cls_token_id`, …) |
| `tokenizer_config.json` | — | Tokenizer init config |
| `added_tokens.json` | — | Special tokens (`<pad>`, `<mask>`, `<unk>`) |
| `special_tokens_map.json` | — | Maps special-token roles |

### Code bundle (`teddy_pkg_parent`)

Stage a copy of `<VOLUME_PATH>/snapshot/teddy/` **without weights** to `/tmp/teddy_code` using `shutil.copytree` with `ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.ckpt", "*.pt")`. This produces `~30 MB` of Python source. Pass `clean_code_dir = "/tmp/teddy_code"` as the `teddy_pkg_parent` artifact so `load_context` can insert it into `sys.path`.

### Key source files (read before writing `load_context`)

| File | Exports used in wrapper |
|------|------------------------|
| `teddy/models/model_directory.py` | `get_architecture(model_dir)`, `model_dict` |
| `teddy/models/teddy_g/model.py` | `TeddyGConfig`, `TeddyGModel` (via `model_dict`) |
| `teddy/tokenizer/gene_tokenizer.py` | `GeneTokenizer` — `from_pretrained(model_dir)` reads `vocab.txt` from the CHECKPOINT dir, not `teddy/tokenizer/` |
| `teddy/data_processing/preprocessing/preprocess.py` | Not imported by the wrapper; the reference for preprocessing and the [upstream-parity test](#upstream-parity-test): `normalize_data_inplace(matrix_csr, norm_value)`, `scale_columns_by_median_dict(layer, data, median_dict_path, median_column)` |
| `teddy/data_processing/tokenization/tokenization.py` | Not imported by the wrapper; the reference for tokenization and the parity test: `_check_genes_in_tokenizer(data, gene_id_column, tokenizer)`, `_build_batch_tensors(X_batch, token_array, token_args, data=None, obs_indices=None)` |
| `teddy/data_processing/utils/medians/data/teddy_gene_medians.json` | Per-gene medians (Ensembl ID → median) used by preprocessing; ships in the code bundle |

### Hugging Face download

The TEDDY repo uses HF's **Xet/CAS storage backend**. Standard HTTP download fails with:
```
RuntimeError: CAS service error: IO Error: Illegal seek (os error 29)
```

**Preferred fix — subprocess isolation** (works on all AI Runtime versions, v5+, v6+):

Databricks AI base environments (v5, v6, and likely future versions) pre-enable `hf_transfer` via `HF_HUB_ENABLE_HF_TRANSFER=1` and import `huggingface_hub` early. Setting env vars in-process after import has no effect. The robust workaround is to run the download in a **subprocess** with a clean environment:

```python
import subprocess, sys, os
env = {**os.environ, "HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
subprocess.check_call(
    [sys.executable, "-c", f'''
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Merck/TEDDY", local_dir="{local_dir}")
'''],
    env=env,
)
```

This guarantees a fresh Python process with no pre-imported `huggingface_hub`. Do **not** rely on the simpler env-var-only pattern (`os.environ["HF_HUB_DISABLE_XET"] = "1"` before import) — it only works if the notebook kernel has never imported `huggingface_hub`, which is not guaranteed on AI Runtime.

### Compute requirements

| Task | Compute |
|------|---------|
| Download from HF | CPU (Serverless or classic). Write to `/tmp`, not `/local_disk0`. |
| Dry-load test, log model, register | GPU (Serverless GPU or GPU classic). |
| Census cell generation (1K cells) | GPU recommended (~5 min). CPU works but ~20 min for 1K cells (model forward pass is the bottleneck). |
| Model Serving endpoint | `GPU_SMALL` = **T4 (16 GB VRAM)**, not A10G. Sufficient for 70M with batch cap ≤10 cells. For 160M/400M or higher throughput, use `GPU_MEDIUM` (A10G, 24 GB VRAM). H100 is not required. `scale_to_zero_enabled=True`. |

### Dependencies (from `pyproject.toml`)

```python
pip_requirements = [
    "torch>=2.3.0",
    "transformers==4.41.0",   # EXACT — 5.x breaks TeddyGModel (all_tied_weights_keys)
    "numpy>=1.26.4,<2.0",    # numpy 2.x has C-ABI breaking changes
    "pandas>=2.2.2,<3.0",    # pandas 3.x has breaking changes
    # anndata — NOT in the serving path; do not include
]
```

`transformers` must be pinned exactly to `4.41.0`. In transformers 5.x, `PreTrainedModel._move_missing_keys_from_meta_to_device` calls `self.all_tied_weights_keys.keys()` on a dict; `TeddyGModel` only implements `_tied_weights_keys` (a list from 4.x). An unpinned `>=4.40` resolves to 5.x in the serving container and fails with `AttributeError: 'TeddyGModel' object has no attribute 'all_tied_weights_keys'`.

## Notebook architecture

A **single combined notebook** (gt_teddy) handles download through deploy, with **three `restartPython()` boundaries** that isolate lightweight, heavy, and Census dependencies:

| Phase | Cells | Installs | Why separate |
|---|---|---|---|
| **Download** | 5-9 | `huggingface_hub` only | Lightweight, no GPU deps |
| *restart 1* | Cell 5 | `restartPython()` | |
| **Register + Deploy** | 14-18 | `scanpy torch transformers mlflow` | Heavy deps for PyFunc + model loading |
| *restart 2* | Cell 14 | `restartPython()` | |
| **Census + AI Search** | 20-24 | `cellxgene-census` via `%pip` | Census deps conflict with base botocore |
| *restart 3* | Cell 20 | `restartPython()` | |

Cells 6, 15, and 21 re-read widgets after each restart (widgets persist, Python vars don't).

> **⚠️ Census deps must use `%pip install`, not subprocess.** Installing `cellxgene-census` via `subprocess.check_call` on serverless breaks `botocore.compat` (the base environment's botocore at `/opt/databricks-environments/...` is read-only and cannot be patched). The `%pip` + `restartPython()` approach cleanly installs to the ephemeral environment and avoids the conflict.

Alternatively, **two separate notebooks** (download CPU + register/deploy GPU) can be used to avoid wasting GPU time on downloads.

### Notebook phases (gt_teddy combined)

| Phase | What | Cells |
|-------|------|-------|
| **0. Setup** | Widgets, Volume check, compute detection, idempotent cleanup of resources this run created (ask before touching anything else) | 2‒4, 12 |
| **Download** | HF snapshot to UC Volume (idempotent, sentinel file) | 5–9 |
| **1. Register** | Wrap `TEDDYEmbedder` PyFunc, `input_example` + `signature`, register to UC as `teddy_{variant}` | 14–18 |
| **1b. Census** | Generate `teddy_cells_{variant}` from CELLxGENE Census (real cells, metadata) | 20–22 |
| **1b. AI Search index** | Delta Sync index over `teddy_cells_{variant}` (optional for oss-001) | 24 |
| **1b. Viz** | 5-panel: Heatmap, PCA, t-SNE, L2 norms, Cosine sim (colored by cell type) | 23 |
| **2. Deploy** | GPU Model Serving endpoint with inference table + AI Gateway + scale-to-zero | 26 |
| **3. Score / Eval** | 3a: Endpoint smoke test. 3b: AI Search eval scorer (self-retrieval, monotonic distances) | 28–29 |
| **4. Teardown** | Delete this run's endpoint + optionally its AI Search resources. Inference table + UC model preserved | 31 |

Gate costly steps (endpoint, deployment, AI Search endpoint and index) behind a `run_go` widget whose saved default is always `"false"` — `dbutils.widgets.dropdown("run_go", "false", ["false", "true"], "Run costly steps")`; never save it as `"true"` (SKILL.md Troubleshooting #14):

- **Request asks to deploy**: say in one short line that the gate is off by default and that, because they asked, you will switch `run_go` on for this run and create the endpoint (name, GPU tier) and any AI Search endpoint/index unless they stop you — then proceed. If you cannot set the widget, tell the user to set `run_go` to `true` and re-run from the gated cell.
- **Request says both "deploy" and "write the code"** (e.g. "Deploy X … Write the code."): treat it as a deploy request — the code is written either way — and say so in the same one line before switching `run_go` on.
- **Request asks only for code**: leave the gate off and say how to turn it on.
- **Anything beyond the request** (larger GPU tier, extra endpoints, deleting or replacing existing resources): stop and ask first.

### GWB lineage

| GWB notebook | Status | This notebook |
|---|---|---|
| `01_register_teddy.py` + `teddy_wrapper.py` | Active | Phase 1 (standalone, no GWB wheel) |
| `03_reembed_reference.py` | Active | Phase 1b (CELLxGENE → Delta) |
| `04_create_teddy_vs_index.py` | Active | Phase 1b (AI Search index) |
| `02_import_model_gwb.py` | Active (needs GWB wheel) | **Replaced** by Phase 2 (SDK-only) |

### AI Search reference corpus (Census-based)

The AI Search index needs a Delta table of pre-embedded cells (`teddy_cells_{variant}`) as the reference corpus. The gt_teddy notebook generates this from **CELLxGENE Census** (real single-cell data with metadata), not synthetic data.

#### Table naming (variant-aware)

```
{catalog}.{schema}.teddy_cells_{variant}   # e.g. <catalog>.skills.teddy_cells_70m
{catalog}.{schema}.teddy_{variant}          # UC model name
teddy-{variant}-embedder                   # endpoint name
```

Embeddings are NOT interchangeable between variants (different dimensionality), so tables, models, and endpoints are all variant-suffixed.

#### Census generation pipeline

1. Query Census obs metadata (`is_primary_data=True`, `assay="10x 3' v3"`) using organism key `"homo_sapiens"` (lowercase underscore — **not** `"Homo sapiens"`; the Census Python API changed to lowercase keys for all versions including older census snapshots like 2024-07-01) → ~19M cells (10x 3' v3 primary)
2. Sample `census_n_cells` (default 1000) with random seed for reproducibility
3. Fetch expression matrix via `tiledbsoma.AxisQuery.to_anndata(X_name="raw")`
4. **Remap var_names**: Census default `var_names` = numeric indices (0, 1, 2...), NOT gene symbols. Must use `adata.var["feature_id"]` for Ensembl IDs:
   ```python
   adata.var_names = adata.var["feature_id"].values  # CRITICAL
   adata.var_names_make_unique()
   ```
5. Intersect with TEDDY vocab: ~22K overlap out of 25K Ensembl IDs (86.6%)
6. **Filter AnnData to TEDDY vocab genes** before embedding: `adata = adata[:, mask]`. Census delivers ~60k genes per cell; sending all of them to the serving endpoint exceeds the 16 MB request-size limit. Filtering to the ~22k Census genes that are in the TEDDY vocab cuts the payload by >50% and stays within limits.
7. Embed in batches via the serving endpoint. **Batch size depends on GPU tier**: 10 cells for `GPU_SMALL` (T4, 16 GB); 50 for `GPU_MEDIUM` (A10G, 24 GB). Larger batches on T4 cause `CUDA out of memory` — the 70M model's activations at 2048 seq_len × 512 hidden dim saturate 14.5 GB of available VRAM with >10 concurrent cells.
8. Write `(cell_id, embedding, cell_type, tissue_general, disease)` to Delta **without overwriting an existing table**: if `spark.catalog.tableExists(CELLS_TABLE)`, check its embedding dimension matches this variant, then `MERGE` on `cell_id` (or append); overwrite only after the user confirms the table name (SKILL.md Best Practices §15):
   ```python
   from delta.tables import DeltaTable
   if spark.catalog.tableExists(CELLS_TABLE):
       (DeltaTable.forName(spark, CELLS_TABLE).alias("t")
           .merge(cells_df.alias("s"), "t.cell_id = s.cell_id")
           .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())
   else:
       cells_df.write.saveAsTable(CELLS_TABLE)
   ```
9. Enable CDF for Delta Sync AI Search index

#### Sizing guide

| Cells | Time (GPU) | Time (CPU) | Use case |
|-------|-----------|-----------|----------|
| 200 | ~1 min | ~4 min | Marginal for viz |
| **1,000** | **~5 min (GPU_MEDIUM, batch 50)** / **~2 min endpoint** | **~20 min** | **GT default, good for PCA/t-SNE** |
| 10,000 | ~45 min | hours | Production |

#### Gene ID compatibility (verified)

- TEDDY vocab: 43,804 tokens; 25,424 Ensembl IDs (`ENSG*`)
- Non-Ensembl tokens: cell type labels (`contractile_cell`, etc.) + special tokens
- Census `feature_id`: 61,497 Ensembl IDs
- Overlap: 22,029 (86.6%) — viable for all variants
- Expression: 88.3% sparse, mean nonzero 3.8 (typical scRNA-seq)

#### Dependencies and the botocore conflict

`cellxgene-census` pulls `tiledbsoma`, `anndata`, and transitive AWS deps (`botocore`, `s3transfer`). On serverless, the base environment's `botocore` at `/opt/databricks-environments/...` is read-only. A `subprocess.check_call` pip install upgrades botocore in the user site-packages but the base copy is found first on `sys.path`, causing:
```
ImportError: cannot import name 'EC' from 'botocore.compat'
```
**Fix**: use `%pip install -q cellxgene-census` + `dbutils.library.restartPython()` in a separate cell. The restart cleanly resolves the module search path.

#### AI Search index creation (Cell 24)

After the `teddy_cells_{variant}` table is written, Cell 24 creates a Delta Sync AI Search index:

1. **Endpoint**: confirm the AI Search endpoint name with the user and hold it in a variable (`VS_ENDPOINT = "<vector-search-endpoint>"`). Reuse an existing endpoint only if the user confirms it; otherwise create a `STANDARD` endpoint under the agreed name and wait for ONLINE status. Never assume another project's endpoint (for example one created by a Genesis Workbench deployment) exists or may be shared.
2. **Dimension check**: if an index with this name already exists and has a different dimension (e.g. 1024 from 400M vs 512 from 70M after a variant switch), **stop and ask the user** before deleting it — it may serve another project. Offer to create a new index under a variant-suffixed name (e.g. `teddy_cell_index_{variant}`) instead; delete and recreate the existing index only with the user's explicit confirmation. Embeddings are not interchangeable between variants.
3. **Index creation**:
   ```python
   w.vector_search_indexes.create_index(
       name=f"{CATALOG}.{SCHEMA}.teddy_cell_index",
       endpoint_name=VS_ENDPOINT,  # name confirmed with the user
       primary_key="cell_id",
       index_type=VectorIndexType.DELTA_SYNC,
       delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
           source_table=source_table,
           embedding_vector_columns=[
               EmbeddingVectorColumn(name="embedding", embedding_dimension=EMB_DIM)
           ],
           pipeline_type=PipelineType.TRIGGERED,
           columns_to_sync=["cell_id", "cell_type", "tissue_general"],
       ),
   )
   ```
4. **CDF required**: `delta.enableChangeDataFeed = true` must be set on the source table (Cell 22 does this automatically after writing).
5. **Sync**: if the index already exists and dimensions match, confirm with the user that it is the index to reuse, then trigger a sync instead of recreating.

The default index name is `{catalog}.{schema}.teddy_cell_index` (not variant-suffixed, so it can collide across runs and variants; a dimension mismatch means ask the user — never recreate it automatically).

#### AI Search eval scorer (Cell 29)

Validates the index after creation/sync:

| Scorer | What | Pass condition |
|--------|------|----------------|
| Self-retrieval | Query a known cell's embedding | Top-1 result = same cell (score \~1.0) |
| Result count | Requested 10 results | Got exactly 10 |
| Monotonic | Scores descending | `scores[i] >= scores[i+1]` for all i |

During sync, failures are soft warnings (expected). Once the index reports `ready=True`, failures become hard assertions.

**SDK note**: `QueryVectorIndexResponse.result` is a `ResultData` with `data_array` and `row_count` — there is no `column_names` attribute. Columns appear in `data_array` in the order specified in the `columns=` argument, with a trailing `score` column appended. Use positional indexing:
```python
top_id = data_array[0][0]            # first requested column
score  = float(data_array[0][-1])     # score is always last
```

### Snapshot path resolution

The register/deploy notebook resolves the snapshot directory with a 2-tier fallback:

```python
_gwb_path = f"{BASE_DIR}/snapshots/{HF_REVISION}/teddy"  # GWB convention
_flat_path = f"{BASE_DIR}/snapshot/teddy"                  # gt_teddy_01 flat layout
if os.path.isdir(_gwb_path):
    snapshot_dir = f"{BASE_DIR}/snapshots/{HF_REVISION}"
elif os.path.isdir(_flat_path):
    snapshot_dir = f"{BASE_DIR}/snapshot"
```

### File-based model logging (`python_model=path`)

> **⚠️ REQUIRED — never use `python_model=<instance>` for TEDDY.**
> The `teddy` package is shipped via `code_paths`, not pip-installed.
> Cloudpickle captures `import teddy` references at pickle time, but the
> serving container resolves `cloudpickle.load()` **before** `code_paths`
> are placed on `sys.path` → `ModuleNotFoundError: No module named 'teddy'`.
> Always write the wrapper class to a `.py` file, include
> `mlflow.models.set_model(TEDDYEmbedder())` at module level, and pass
> `python_model=<path_to_wrapper.py>` to `log_model`.

TEDDY uses file-based logging with `teddy_wrapper.py`, not cloudpickle:

```python
# Stage clean code bundle (source without weights)
clean_code_dir = "/tmp/teddy_code"
shutil.copytree(
    teddy_pkg_dir, f"{clean_code_dir}/teddy",
    ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.ckpt", "*.pt", "__pycache__"),
)
shutil.copy2(gwb_wrapper_path, f"{clean_code_dir}/teddy_wrapper.py")

# Log with code_paths (avoids cloudpickle serialization issues)
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=teddy_wrapper_path,  # file path, not instance
    code_paths=[clean_code_dir],
    artifacts={"model_dir": model_dir, "teddy_pkg_parent": clean_code_dir},
    signature=signature,
    input_example=(input_example, default_params),
    pip_requirements=[...],
)
```

## Wrapper boundary

### Imports inside the PyFunc class cell

Every symbol used inside class methods must be imported **in the same cell** as the class definition. The serving container deserialises the class in a fresh Python process with no notebook globals. At minimum:

```python
import sys, os, io, json, inspect
import mlflow, pandas as pd, numpy as np, torch
```

Do **not** rely on `sys`, `os`, or `io` being available from a notebook-level `import os, sys` — they will not exist in the serving container.

### `load_context` pattern

```python
def load_context(self, context):
    import sys  # explicit — not available from notebook globals in serving container
    teddy_pkg_parent = context.artifacts["teddy_pkg_parent"]
    if teddy_pkg_parent not in sys.path:
        sys.path.insert(0, teddy_pkg_parent)

    from teddy.models.model_directory import get_architecture, model_dict
    from teddy.tokenizer.gene_tokenizer import GeneTokenizer

    model_dir = context.artifacts["model_dir"]
    arch = get_architecture(model_dir)
    config_cls = model_dict[arch]["config_cls"]
    model_cls  = model_dict[arch]["model_cls"]

    self.config    = config_cls.from_pretrained(model_dir)
    self.model     = model_cls.from_pretrained(model_dir, config=self.config)
    self.device    = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(self.device).eval()
    self.tokenizer = GeneTokenizer.from_pretrained(model_dir)  # reads vocab.txt from model_dir
    self._forward_params = set(inspect.signature(self.model.forward).parameters.keys())
    self.add_cls     = bool(getattr(self.config, "add_cls", False))
    self.cls_token_id = int(getattr(self.config, "cls_token_id", 0))
    self.d_model     = int(getattr(self.config, "d_model", 0))
    self._use_bf16   = (self.device == "cuda")
    # Per-gene medians for upstream preprocessing; shipped in the code bundle
    with open(f"{teddy_pkg_parent}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json") as f:
        self.medians = json.load(f)
```

### Forward output key: `cell_emb` (already 2D)

> **Verified in oss-002 v2 (2026-10-01).** The 70M variant's `forward()` returns `{"cell_emb": tensor(cells, d_model)}` — a **2D embedding** (the encoder output at the first position, `transformer_output[:, 0, :]` in `teddy/models/teddy_g/model.py:261`; with `add_cls` false that is the top-ranked gene's position), NOT token-level 3D hidden states. The forward signature accepts `gene_ids` and `attention_mask` but **not `gene_values`**; rank encoding is handled internally by the model. Passing `gene_values` in `fwd_kwargs` is harmless (filtered out by `_forward_params` check) but unnecessary.
>
> Forward params (70M): `{'labels', 'return_outputs', 'gene_ids', 'kwargs', 'annotations', 'annotation_attention_mask', 'annotation_labels', 'attention_mask', 'position_ids'}`.

### Tensor-boolean safety in `_predict_batch`

TEDDY's forward pass returns a dict. The **primary key is `cell_emb`** (first-position output, shape `(cells, d_model)`). Fallback keys `all_embs`, `last_hidden_state`, and `hidden_states` are retained for forward-compatibility with other variants. **Never use Python `or` or truthy evaluation on these tensors** — `bool(tensor)` raises `RuntimeError: Boolean value of Tensor with more than one element is ambiguous`. Always use explicit `is None` checks:

```python
# WRONG — triggers tensor boolean ambiguity
token_embeddings = outputs.get("all_embs") or outputs.get("last_hidden_state")

# CORRECT — explicit None checks, cell_emb first (primary key for 70M)
token_embeddings = outputs.get("cell_emb")       # already 2D — preferred
if token_embeddings is None:
    token_embeddings = outputs.get("all_embs")    # fallback
if token_embeddings is None:
    token_embeddings = outputs.get("last_hidden_state")
if token_embeddings is None and outputs.get("hidden_states") is not None:
    token_embeddings = outputs["hidden_states"][-1]

# If already pooled (2D), skip external pooling
if token_embeddings.dim() == 2:
    embeddings = token_embeddings
else:
    # 3D token-level: apply mean or CLS pooling
    ...
```

This applies to all extraction branches (dict outputs, named-tuple outputs, hidden-states fallback).

### Out-of-vocabulary (OOV) genes: drop them, as upstream does

Upstream TEDDY never feeds `<unk>` for genes it does not know. Tokenization keeps only genes found in the vocabulary (`_check_genes_in_tokenizer`, `teddy/data_processing/tokenization/tokenization.py:129-146`), slices the expression matrix to those genes before choosing the top-K (`tokenization.py:230`), and raises an error when fewer than 10% of the genes are in the vocabulary (`tokenization.py:142-145`). Do the same in the wrapper: drop OOV genes, and reject a request with no vocabulary genes with a clear error rather than returning an embedding.

Do not rely on `convert_tokens_to_ids` for OOV detection: `<unk>` is an added token (`added_tokens.json`), not a line of `vocab.txt`, so `GeneTokenizer._convert_token_to_id` (`teddy/tokenizer/gene_tokenizer.py:126-128`) returns `None` for an unknown gene instead of an ID. Test membership in `self.tokenizer.vocab` (the `vocab.txt` dictionary) instead:

```python
keep = [i for i, g in enumerate(gene_names) if g in self.tokenizer.vocab]
if not keep:
    raise ValueError("No gene IDs in adata_var match the TEDDY vocabulary (expects Ensembl IDs, e.g. ENSG...)")
token_array = torch.tensor([self.tokenizer.vocab[gene_names[i]] for i in keep])
```

File and line references in this section and the next are to the upstream source at HF revision `8c880aa67674042ccdc254bccb5de081092a48bf`, checked read-only.

### Top-K selection and rank encoding

TEDDY does not consume raw expression counts. The wrapper reproduces upstream preprocessing, then upstream tokenization, in this order — the settings are the ones the Hugging Face model card says were used for the pretraining corpus (`README.md:136-155`: `normalized_total: 10000`, `median_dict: teddy_gene_medians.json`, `log1p: False`; tokenization `README.md:171-190`: `max_seq_len: 2048`, `add_cls: False`, `continuous_rank: True`, `include_zero_genes: False`):

1. **Normalise each cell to 10,000 total counts** (row L1-normalise × 10,000, `normalize_data_inplace`, `teddy/data_processing/preprocessing/preprocess.py:272-281`).
2. **Divide each gene by its TEDDY median** (`scale_columns_by_median_dict`, `preprocess.py:284-306`; genes without a median keep factor 1.0). No log transform.
3. **Drop OOV genes** (see above).
4. **Top-K by value, highest first** (`torch.topk(..., sorted=True)` in `_build_batch_tensors`, `tokenization.py:149-181`); K = `max_seq_len` (minus one if `add_cls`).
5. **Drop zero-expression genes** from the selected K (`include_zero_genes=False`, `tokenization.py:261-264`) — in a batch, pad them with `<pad>` and mask them out rather than feeding zero-valued genes.

Put these steps in one wrapper method, `_tokenize(X, gene_names, max_seq_len)` returning `(gene_ids, attention_mask)`, so the [upstream-parity test](#upstream-parity-test) can compare the wrapper's tokens with upstream's.

```python
# 1-2. Upstream preprocessing on raw counts X (cells x genes), gene_names = Ensembl IDs
tot = X.sum(axis=1, keepdims=True)
tot[tot == 0] = 1.0
X = X / tot * 1e4
X = X * np.array([1.0 / self.medians[g] if g in self.medians else 1.0 for g in gene_names],
                 dtype=np.float32)[None, :]
# 3. keep vocabulary genes only (see OOV section) -> keep, token_array
X_t = torch.from_numpy(np.ascontiguousarray(X[:, keep], dtype=np.float32))
# 4. top-K, highest expression first
k = min(max_seq_len, X_t.shape[1])
top_vals, top_idx = torch.topk(X_t, k=k, largest=True, sorted=True)
gene_ids = token_array[top_idx]                               # (cells, k)
# 5. drop zero-expression genes: pad + mask (HF convention, 1 = attend; the model inverts it)
expressed = top_vals > 0
gene_ids[~expressed] = self.tokenizer.pad_token_id
attention_mask = expressed.long()
```

Rank values: upstream's `continuous_rank` writes `torch.linspace(-1, 1, steps=n).flip(0)` over each cell's expressed genes (`_rank_continuous`, `tokenization.py:89-100`) — 1.0 for the highest, −1.0 for the lowest. The 70M `forward()` does not take them: it has no `gene_values` argument (extra keyword arguments are ignored, `model.py:197-208`) and adds position embeddings by sequence index (`model.py:223-226`), so the order of `gene_ids` carries the rank. Do not pass a values tensor; the `_forward_params` filter would drop it anyway. Only `gene_ids` and `attention_mask` are passed.

### `predict` — `pd.read_json` on newer pandas

Pandas 2.1+ deprecates passing a literal JSON string directly; always wrap:
```python
var_df = pd.read_json(io.StringIO(row["adata_var"]), orient="split")
```

### Stale module cache (out-of-order cell execution)

In a long-lived kernel, a prior partial import of the `teddy` package can persist in `sys.modules` and cause `ModuleNotFoundError` for submodules on re-run. **Purge in two places:**

**1. At cell level** — before any `from teddy import ...` in the notebook (prevents stale imports across re-runs):

```python
# Flush cached teddy modules so re-runs get fresh class objects
for _k in [k for k in sys.modules if k.startswith("teddy")]:
    del sys.modules[_k]
```

**2. Inside `load_context`** — the serving container has no prior kernel state, but the dry-load test cell does. Adding the purge inside `load_context` makes the wrapper safe for both local testing and production:

```python
def load_context(self, context):
    # Purge stale teddy modules (safety for re-runs and dry-load tests)
    for _m in list(sys.modules):
        if _m == "teddy" or _m.startswith("teddy."):
            del sys.modules[_m]

    teddy_pkg_parent = context.artifacts["teddy_pkg_parent"]
    if teddy_pkg_parent not in sys.path:
        sys.path.insert(0, teddy_pkg_parent)
    # ... rest of load_context
```

> **Why both?** The cell-level purge catches notebook re-execution issues. The `load_context` purge catches the dry-load test cell (which imports teddy in the same kernel as the class definition cell). The serving container starts clean, so the `load_context` purge is a no-op there — but it costs nothing and prevents a class of hard-to-debug failures.

## Deployment recommendation

**Model Serving** is appropriate for the 70M variant and bounded per-cell queries:

* Checkpoint fits on `GPU_SMALL` with headroom; `scale_to_zero_enabled=True` reduces cost.
* Larger variants (160M, 400M) may require `GPU_MEDIUM`; verify with a local dry-load.
* For full-AnnData batch embedding (thousands to millions of cells), prefer a Job that reads the Volume snapshot directly and writes embeddings back to Unity Catalog.
* **Always use SDK enum classes**, not string literals, for endpoint configuration (see SKILL.md §13):

```python
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

served_entity = ServedEntityInput(
    name=f"teddy-{VARIANT.lower()}-v{version}",
    entity_name=UC_MODEL_NAME,
    entity_version=str(version),
    workload_type=ServingModelWorkloadType.GPU_SMALL,  # not "GPU_SMALL"
    workload_size="Small",
    scale_to_zero_enabled=True,
)
```

## Registration and observability

Register to Unity Catalog as `<catalog>.<schema>.teddy_<variant>` (e.g. `<catalog>.skills.teddy_70m`). Log both declared licenses (model card Apache-2.0, package `pyproject.toml` MIT — see [Identity](#identity)), the HF revision, paper DOI, HF source URL, and variant as MLflow tags. The Census table follows the same convention: `teddy_cells_<variant>` (e.g. `teddy_cells_70m`), and the endpoint: `teddy-<variant>-embedder` (e.g. `teddy-70m-embedder`).

**Inference tables** are a best-practice default (see SKILL.md §AI Gateway and inference tables). Enable them in the `serving_endpoints.create()` call. The `adata_sparsematrix` and `adata_var` payloads contain gene expression profiles — classify before logging; prefer logging only `cell_id`, embedding dimension, and variant tag. Confirm the inference-table catalog is backed by external storage.

```python
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    AiGatewayUsageTrackingConfig,
)

ai_gateway = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        catalog_name=CATALOG,
        schema_name=SCHEMA,
        # Timestamped prefix avoids name conflicts from prior failed deploys
        table_name_prefix=f"{ENDPOINT_NAME}_{datetime.now():%Y%m%d%H%M}",
        enabled=True,
    ),
    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
)

w.serving_endpoints.create(
    name=ENDPOINT_NAME,
    config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity]),  # name: Optional in databricks-sdk <=0.70.0, required from 0.71.0; passing it works on both
    ai_gateway=ai_gateway,
)
```

> **Do not use** the deprecated `AutoCaptureConfigInput` — it was replaced by `AiGatewayInferenceTableConfig` in the current SDK.

## Validation checklist

1. **Vocabulary smoke test**: load `vocab.txt`, verify the line count (70M at HF revision `8c880aa`: 43,808 lines, of which 25,424 are Ensembl IDs), check that `ENSG00000000003` is present.
2. **Import test**: in the serving container environment, `from teddy.models.model_directory import get_architecture, model_dict` and `from teddy.tokenizer.gene_tokenizer import GeneTokenizer` succeed.
3. **Dry-load test**: `load_context` completes, prints `TEDDY-G 70M loaded on cuda, d_model=…`.
4. **Signature test**: build the signature with `infer_signature(input_example, output_example, params=default_params)` using string defaults (see [Sending inference controls](#sending-inference-controls)); check the input schema has all three columns (`adata_sparsematrix`, `adata_obs`, `adata_var`) as required fields and that `{p.name: p.dtype for p in signature.params.params}` lists every control as `DataType.string`.
5. **Real-gene payload test**: test payload uses Ensembl IDs from `vocab.txt`; embeddings are non-degenerate (not all the same value, norm > 0).
6. **All-OOV payload test**: all gene names set to random strings → every gene is dropped as out-of-vocabulary → the endpoint returns a clear error naming the vocabulary mismatch, not an embedding (upstream raises in the same case; see [Out-of-vocabulary genes](#out-of-vocabulary-oov-genes-drop-them-as-upstream-does)).
7. **Missing-field test**: omit `adata_obs` from the request → endpoint returns a schema-validation error (confirming the signature enforces all three fields).
8. **Inference-control test**: call the endpoint twice through `invoke()` ([Sending inference controls](#sending-inference-controls)) on the same real-gene payload — once with defaults, once with a non-default `{"max_seq_len": "32"}` (values as strings; each cell must have more than 32 expressed genes, or truncation changes nothing) — and assert the embeddings differ (cosine clearly below 1.0). Before the calls, confirm the registered version's params schema (`mlflow.models.get_model_info(model_uri).signature.params`) lists `max_seq_len` and `pooling` as strings — an undeclared control is dropped. Identical output means the control never reached `predict()`; a successful call alone proves nothing. Use `max_seq_len`, not `pooling`, for 70M: its pre-pooled output makes `pooling` a no-op in this wrapper.
9. **Embedding sanity**: for 70M, embedding dimension should match `config.d_model` (read from `config.json`).

10. **AI Search self-retrieval test**: query the index with a known cell's embedding; top-1 result must be the same cell (score ~1.0). Results must be monotonically ordered by similarity.
11. **Census gene ID test**: verify `adata.var_names` are Ensembl IDs (start with `ENSG`), not numeric indices, before intersecting with TEDDY vocab. Overlap should be ≥80% (22K+).
12. **AI Search dimension match test**: if switching variants, the index dimension must match `EMB_DIM` for the current variant. On a dim mismatch (e.g. 1024 from 400M vs 512 from 70M), stop and ask the user: use a new variant-suffixed index name, or replace the existing index only with their confirmation.
13. **Upstream-parity test** (SKILL.md Workflow step 7): run a few real cells through TEDDY's own preprocessing and tokenization and through the wrapper; the selected token sets must match and the per-cell embedding cosine must be at least 0.9999. Comparing the endpoint only with the same wrapper run in the notebook does not count. See [Upstream-parity test](#upstream-parity-test).

## Upstream-parity test

Run in the notebook after the dry-load test, before registering — not in the serving container. Upstream `preprocess.py` and `tokenization.py` import `anndata`, `scanpy`, `scikit-learn`, `datasets` and `tqdm` at module level, so install those in the notebook only. `wrapper` is the dry-loaded `TEDDYEmbedder` instance, `default_params` comes from [Sending inference controls](#sending-inference-controls), and the full snapshot (with `teddy/data_processing/` and the upstream sample data `data/sample_data.h5ad`) is under `<VOLUME_PATH>/snapshot`.

```python
import sys, types, numpy as np, pandas as pd, torch, anndata as ad
from scipy.sparse import csr_matrix

SNAP = f"{VOLUME_PATH}/snapshot"
for _k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.")]:
    del sys.modules[_k]
sys.path.insert(0, SNAP)
from teddy.data_processing.preprocessing.preprocess import normalize_data_inplace, scale_columns_by_median_dict
from teddy.data_processing.tokenization.tokenization import _check_genes_in_tokenizer, _build_batch_tensors
from teddy.tokenizer.gene_tokenizer import GeneTokenizer

# A few real cells as raw counts (upstream preprocessing starts from raw integer counts)
src = ad.read_h5ad(f"{SNAP}/data/sample_data.h5ad")
src = src.raw.to_adata() if src.raw is not None else src
cells = src[:5].copy()
counts = cells.X.toarray() if hasattr(cells.X, "toarray") else np.asarray(cells.X)
genes = list(cells.var_names)                                    # Ensembl IDs

# 1. Upstream preprocessing with the pretraining settings
layer = csr_matrix(counts.astype(np.float32))
normalize_data_inplace(layer, 10000)
scale_columns_by_median_dict(
    layer, cells, f"{SNAP}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json", "index")

# 2. Upstream tokenization: vocabulary genes only, top-K, zeros dropped below
tok = GeneTokenizer.from_pretrained(model_dir)
in_vocab, coding_genes, _ = _check_genes_in_tokenizer(cells, "index", tok)
token_array = torch.tensor(tok.encode(coding_genes.tolist(), add_special_tokens=False))
args = types.SimpleNamespace(max_seq_len=2048, add_cls=False, random_genes=False)
up_ids, up_vals, _, _ = _build_batch_tensors(
    torch.tensor(layer.toarray()[:, in_vocab], dtype=torch.float), token_array, args)

# 3. The wrapper on the same raw counts
request = pd.DataFrame({
    "adata_sparsematrix": [counts.tolist()],
    "adata_obs": [pd.DataFrame({"cell_id": list(cells.obs_names)}).to_json(orient="split")],
    "adata_var": [pd.DataFrame({"index": genes}).to_json(orient="split")],
})
E = np.array(wrapper.predict(None, request, params=default_params)["embedding"].tolist())
w_ids, w_mask = wrapper._tokenize(counts.astype(np.float32), genes, 2048)    # the Top-K recipe above

# 4. The same model on the upstream tokens; compare per cell
for i in range(len(cells)):
    ids = up_ids[i][up_vals[i] != 0].unsqueeze(0).to(wrapper.device)   # include_zero_genes=False
    up_row, w_row = ids[0].cpu(), w_ids[i][w_mask[i].bool()].cpu()
    same_genes = set(w_row.tolist()) == set(up_row.tolist())              # preprocessing parity: membership
    moved = int((w_row != up_row).sum()) if len(w_row) == len(up_row) else None   # information only
    with torch.no_grad():
        ref = wrapper.model(gene_ids=ids, attention_mask=torch.ones_like(ids))["cell_emb"][0].float().cpu().numpy()
    cos = float(ref @ E[i] / (np.linalg.norm(ref) * np.linalg.norm(E[i])))
    print(f"cell {i}: tokens upstream={len(up_row)} wrapper={len(w_row)}  same genes={same_genes}  "
          f"positions in a different order={moved}  cosine(upstream, wrapper)={cos:.9f}")
    assert same_genes and cos >= 0.9999, "wrapper does not reproduce upstream TEDDY on this cell"
```

Pass = the same set of genes and cosine >= 0.9999 on every cell. Token **order** is reported, not asserted: the wrapper's dense NumPy preprocessing and upstream's in-place sparse preprocessing differ in the last float32 digits (up to about 2e-6), which can swap genes with nearly tied values. On the upstream sample data at revision `8c880aa` (5 cells, CPU, float32), 3 of 5 cells had 2–8 positions in a different order, all 5 had the same gene sets, and cosine was 0.99999984 or higher. The 0.9999 threshold sits far above what a real preprocessing error gives: with the median step removed, cosine fell to 0.22–0.92 on the same cells — and on cells with fewer than 2,048 expressed genes the gene set still matched, so the cosine check is what catches it. A failure means the wrapper's preprocessing or token selection differs from upstream — compare the token lists before touching anything else. If the wrapper casts to bf16 on GPU, run this test with the wrapper in float32.

## Sending inference controls

Custom PyFunc inference controls must arrive under the request-body key `"params"`, which the MLflow scoring server passes to `predict(..., params=...)`. `w.serving_endpoints.query()` has no `params` argument; its `extra_params` argument is written to a body key `"extra_params"`, which the server drops without an error (SKILL.md Troubleshooting #13). Post the body yourself:

```python
def invoke(w, endpoint_name, records, params=None):
    """POST to the endpoint with inference controls under the body "params" key."""
    body = {"dataframe_records": records}
    if params:
        body["params"] = params
    return w.api_client.do("POST", f"/serving-endpoints/{endpoint_name}/invocations", body=body)
```

Equivalent HTTP request:

```http
POST /serving-endpoints/<endpoint-name>/invocations
{
  "dataframe_records": [{"adata_sparsematrix": [[...]], "adata_obs": "<json split>", "adata_var": "<json split>"}],
  "params": {"max_seq_len": "2048", "pooling": "mean"}
}
```

Type rules: log the signature with string defaults, because MLflow rejects a string sent for an integer-typed param; cast in the wrapper:

```python
import json
from mlflow.models import infer_signature
from mlflow.types import DataType

# input_example, model_dir and n_cells come from "Input example" above.
# output_example mirrors the documented response: one `embedding` of d_model floats per cell.
with open(f"{model_dir}/config.json") as f:
    d_model = int(json.load(f)["d_model"])  # 512 for 70M
output_example = pd.DataFrame({"embedding": [[0.0] * d_model for _ in range(n_cells)]})

default_params = {"max_seq_len": "2048", "pooling": "mean"}  # strings, matching what invoke() sends
signature = infer_signature(input_example, output_example, params=default_params)
assert {p.name: p.dtype for p in signature.params.params} == {
    "max_seq_len": DataType.string, "pooling": DataType.string}
```

A control missing from the signature's params schema is dropped with only a log warning. A successful call does not prove a control was applied — run Validation checklist item 8.

## Open questions

* Whether HF_HUB_DISABLE_XET will remain necessary as HF updates the Xet client. The subprocess workaround is runtime-version-agnostic (tested on AI v5; expected to work on v6+).
* Whether the `transformers==4.41.0` exact pin will need to be updated for future TEDDY releases.
* Optimal `max_seq_len` for 160M and 400M variants (may differ from 70M default of 2048).
* Whether per-cell `adata_obs` metadata (batch label, donor ID) should be echoed back in the response to aid downstream correlation.
* Whether future Databricks GPU_SMALL tiers will provision A10G instead of T4 — monitor release notes.
* ~~What output key the model forward pass uses~~ **RESOLVED (oss-002 v2, 2026-10-01):** 70M returns `{"cell_emb": tensor(cells, d_model)}` (first-position output, 2D). Forward accepts `gene_ids` + `attention_mask` but NOT `gene_values`. Verify whether 160M/400M variants use the same output key or different ones.
