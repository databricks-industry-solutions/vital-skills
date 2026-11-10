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

TEDDY-G is a transformer encoder foundation model pre-trained on 116 million single-cell RNA-seq (scRNA-seq) cells. It produces per-cell embedding vectors from gene expression profiles. Upstream preprocessing scales each cell to 10,000 total counts and divides each gene by its TEDDY gene median; the model then keeps only vocabulary genes, selects the top-K most expressed genes per cell, rank-encodes their expression values as linearly spaced floats from 1.0 (highest) to −1.0 (lowest), and passes the sequence through a transformer encoder. The resulting embeddings are used for cell-type annotation, batch correction, perturbation prediction, and cross-dataset transfer. This reference covers **online inference** (bounded per-cell queries to a Model Serving endpoint); for large-batch embedding of AnnData objects use a Job. For a deployment with a Vector Search (AI Search) index, see [Deploy with Vector Search](#deploy-with-vector-search).

## Intended use and reference test

* Intended use: zero-shot cell embeddings for clustering and neighbour search; fine-tuned classification (held-out donors, diseases). The paper's zero-shot embeddings average the output gene embeddings (arXiv:2503.03485v2, "Zero-shot embeddings produce coherent clusters").
* Training data: CELLxGENE downloaded 2024-06-11; cells with < 225 counts or > 10% mitochondrial reads and 10x v1 assays removed; cell-type, disease, tissue and sex labels used as training targets. Census 2024-07-01 is the closest proxy for "possibly in training data". Upstream `preprocess.py` compares the mito fraction (0-1) to `max_mitochondrial_prop=10`, so as coded it removes nothing: use 0.10, from Census `raw_sum` and all 37 mito genes (only 13 are in the vocabulary).
* Reference test: scib-style zero-shot clustering (Leiden at the NMI-best resolution 0.1-2.0, NMI / ARI / ASW). Paper: TEDDY-G 400M on Thomas et al. 2024 (gut, not in Census), coarse NMI 0.838 / ARI 0.715. Stand-in used: 10,000 cells of Pelka et al. 2021 colon (Census 2025-11-08, dataset `16023185-de21-4c0d-a9c8-73abdd52d142`, after the cut-off), 7 compartments.
* Baseline: log-normalised counts, 2,000 most variable genes, 50 PCs. On the stand-in, 70M with mean pooling matches it; first-token output scores far lower.

### Validation record

What has been checked, so users know how far to trust a deployment from this file. Guidance only; nothing here blocks use.

| Layer | Variant, revision | Result | Date |
| --- | --- | --- | --- |
| 1: fidelity | 70M, `8c880aa`, wrapper v2 (`pooling` control) | Pass: `first` and `mean` vs TEDDY's own code, min cosine 0.99999 (25 Census cells); `tutorial` vs the tutorial's steps, 0.99999 (64 cells) | 2026-10-30 |
| 2: contract, live endpoint (`GPU_MEDIUM`) | same | Pass, 11 of 11: column and `params` routes used and echoed, column wins, bad values rejected, batch invariance 1.0. `serving_endpoints.query(extra_params=...)` was dropped | 2026-10-30 |
| 3: index (Delta Sync, 3,001 cells) | same | Pass: recall@10 vs exact search 0.9999; every cell its own top-1 at `num_results=10` | 2026-10-30 |
| 4: paper claim | - | Not possible for 70M: the paper's zero-shot numbers are for 400M only; its 70M numbers need fine-tuning | 2026-10-30 |
| 4: tutorial recipe (`tutorials/model_tutorial.ipynb`, `data/sample_data.h5ad`, 3,538 cells) | same | Reproduced (fidelity above; UMAP redrawn). Reported, `Cell.Types` NMI: `tutorial` 0.72, `mean` 0.88, PCA baseline 0.95 | 2026-10-30 |
| 4: stand-in (Pelka 2021 colon, 10,000 cells, reported) | same | Coarse NMI: `mean` 0.95, `first` 0.51, PCA baseline 0.94; fine NMI: `mean` 0.72, PCA 0.73 (paper 400M on Thomas et al.: 0.84 / 0.88) | 2026-10-30 |
| 5: field-style (blood, Census 2025-11-08, reported) | same | Leave one collection out (16): 10-neighbour vote 46% vs PCA 36%. Held-out cells from seen studies: 75% vs PCA 77% | 2026-10-30 |
| 2: express deployment (`env_pack`, `GPU_MEDIUM`) | 70M, `8c880aa`, wrapper v2 | Pass, 5 of 5, twice (run 1, then a clean in-order re-test with order guards): registered in 25.8 s; READY 2.5 / 2.79 min after create (event log; standard: 15-25 min), 0 container-build events; served = notebook wrapper, min cosine 1.000000 (20 cells x 3 poolings); controls reach the model by both routes. A one-cell request in the serving UI's format matched the wrapper (cosine 1.000000) | 2026-11-01 |
| 1-3, earlier wrapper | 70M, `8c880aa`, first-token only | Passed, but served the most-expressed gene's output (layer 4 above); superseded by wrapper v2 | 2026-10-29 |

## Inputs and outputs

### Serving contract

The model does not accept a raw AnnData object. Transport as three DataFrame columns per row; all three are **required** by the MLflow signature and must be present in every request payload, even if `adata_obs` is not used by the model internals.

| Column | Type | Description |
|--------|------|-------------|
| `adata_sparsematrix` | `list[list[float]]` | Dense expression matrix, shape `(cells, genes)`. One sublist per cell; values are **raw counts** — the wrapper applies TEDDY's normalisation (see [Top-K selection and rank encoding](#top-k-selection-and-rank-encoding)). |
| `adata_obs` | `str` — JSON orient=`split` | Cell metadata DataFrame. Required by the signature; not consumed by `predict`. Include at minimum a `cell_id` column. |
| `adata_var` | `str` — JSON orient=`split` | Gene metadata DataFrame. Must have an `index` column containing Ensembl gene IDs (`ENSGXXXXXXXXXXX`) that match `vocab.txt`. |

Inference controls follow the standard in `../integration-contract.md`: **optional input columns** on the same row (one row = one batch of cells), used when present; else the same name under the request's `"params"`; else the default. Every output row echoes the values used. The column route works with any client, including `w.serving_endpoints.query(dataframe_records=...)`; `params` reaches `predict` only via a REST body with a top-level `"params"` key or a loaded model's `.predict(df, params=...)` (not `serving_endpoints.query(extra_params=...)`, not `mlflow.models.predict`). See [Sending inference controls](#sending-inference-controls).

| Control (optional column, `str`) | Default | Allowed | Notes |
|-----------|------|---------|-------|
| `max_seq_len` | `"2048"` | positive integer as a string | Top-K genes selected per cell; cast with `int()` in `predict`. |
| `pooling` | `"mean"` | `"mean"`, `"tutorial"`, `"first"` | `"mean"`: average the encoder outputs over expressed genes (padding excluded). `"tutorial"`: pad to 2,048 and average all positions, as in the official tutorial. `"first"`: the output at position 0, which for 70M (`add_cls=False`) is the most-expressed gene, not a cell token. Use `"mean"` for zero-shot use; see [Pooling for zero-shot use](#pooling-for-zero-shot-use). The control must actually switch the output (Validation checklist item 8). |

### Output

```json
{
  "predictions": [
    {"embedding": ["<float>", "..."], "pooling": "mean", "max_seq_len": "2048"}
  ]
}
```

One dict per input cell, with the controls actually used. Embedding dimension equals `config.d_model` (read from `config.json` in the checkpoint directory).

`cell_id` is **not** echoed: predictions come back one per cell in request order, so match them to cells by
position (check the count). Send requests with `dataframe_records=[...]` as in the example below; the SDK's
`dataframe_split` expects its typed object, and a plain dict fails with `'dict' object has no attribute
'as_dict'` (both hit in a 2026-11-02 run).

### Input example (Python — use real vocab gene IDs)

```json
{
  "adata_sparsematrix": [[12.0, 5.0, 3.0]],
  "adata_obs": "<json-string-orient=split-cell-metadata>",
  "adata_var": "<json-string-orient=split-gene-metadata-with-index>",
  "pooling": "mean",
  "max_seq_len": "2048"
}
```

`pooling` and `max_seq_len` are optional; omit them to get the defaults.

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
    "pooling":            ["mean"],                  # optional controls, shown so users see them
    "max_seq_len":        ["2048"],
})
```

**Critical**: always use real Ensembl IDs from `vocab.txt` when constructing test payloads. Synthetic gene names are not in the vocabulary, so upstream TEDDY drops them (see [Out-of-vocabulary genes](#out-of-vocabulary-oov-genes-drop-them-as-upstream-does)) and nothing real is tokenised — the test cannot validate the tokenisation path.

### HTTP payload (Databricks SDK)

Put controls in the record itself; this works with the SDK's `query()`. Never use `w.serving_endpoints.query(extra_params=...)`: the SDK writes it to a body key the MLflow scoring server drops, so the call succeeds with defaults (measured 2026-10-27 on a TEDDY endpoint: cosine 1.0 to the default).

```python
records = [{
    "adata_sparsematrix": expr.tolist(),
    "adata_obs":          obs_df.to_json(orient="split"),
    "adata_var":          var_df.to_json(orient="split"),
    "pooling":            "mean",                    # optional; omit for the default
}]
response = w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=records)  # w = WorkspaceClient()
preds = response.predictions                            # typed response: use the attribute, not .get()
embeddings = [p["embedding"] for p in preds]
assert {p["pooling"] for p in preds} == {"mean"}        # the echo shows what was used
```

Callers that send `"params"` instead (REST body, `invoke()` in [Sending inference controls](#sending-inference-controls)) still work; a column on the row wins over `params`. Values are strings; the wrapper casts them.

## Artifacts and runtime

### Checkpoint directory (`model_dir`)

Path: `<VOLUME_PATH>/snapshot/teddy/models/teddy_g/<VARIANT>/` (e.g. `.../snapshot/teddy/models/teddy_g/70M/`).
The weights sit inside the `teddy/` package folder, not at `snapshot/models/...`. Set `model_dir` from
this path and assert that `model_dir/config.json` and `model_dir/model.safetensors` exist before loading;
a run on 2026-11-02 first set it one folder too shallow and lost a retry.

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

**Download the full pinned snapshot when possible** (all three sizes, `tutorials/`, `data/sample_data.h5ad`; about 2.6 GB at `8c880aa`). The reference check needs the tutorial and its sample data, and comparing sizes needs every set of weights. Plain HTTPS from `https://huggingface.co/Merck/TEDDY/resolve/<revision>/<path>`, checking each file against the LFS SHA-256 from `https://huggingface.co/api/models/Merck/TEDDY/revision/<revision>?blobs=true`, avoids the Xet error below and gives a per-file checksum manifest (tested 2026-10-30: 74 files, 2.58 GB).

**When the user asks for one size only** (e.g. a short demo of 70M): download the same file list but skip
the other sizes' folders (`teddy/models/teddy_g/160M/` and `.../400M/` when serving 70M). Keep the code
(`teddy/`, including the medians JSON), `tutorials/` and `data/sample_data.h5ad`: the fidelity check needs
them. Record in the manifest which paths were skipped.

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
| Batch embedding in a notebook | Serverless A10 (22 GB usable): attention at 2,048 genes needs ~0.4-0.5 GB per cell, so use 16-20 cells per batch and free other models first (50 ran out of memory, 2026-10-29). |
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

## Deploy with Vector Search

When the request includes an index (endpoint plus nearest-neighbour search), follow `../workflows/embeddings-to-vector-search.md` for the steps and use its tested helpers (`scripts/vs_index.py`, `scripts/nn_annotate.py`). Index lifecycle mechanics come from the generic `vector-search` skill. This section holds only the TEDDY-specific facts.

| Topic | TEDDY rule |
| --- | --- |
| Vector size | `config.d_model`: 70M = 512, 160M = 768, 400M = 1,024. One index per model size; a size mismatch means a new index name, not a replacement (Validation checklist item 12). |
| Pooling | Embed reference cells with the default `pooling="mean"`. Queries must use the same model version and pooling as the index ([Pooling for zero-shot use](#pooling-for-zero-shot-use)). |
| Table columns | `cell_id` (primary key, Census release in the ID), `embedding`, labels (`cell_type`, `disease`, `dataset_id`, `donor_id`), `n_genes`, plus `model_version` and `pooling` per row. Change Data Feed on, for a Delta Sync index. |
| Writes | Append or `MERGE` on `cell_id`; never overwrite an existing table or index (SKILL.md Best Practices #15). |
| Embedding the reference | Faster on a notebook GPU with the registered model than through the endpoint; then check that the endpoint gives the same vectors (Upstream-parity test, Validation checklist item 13). |
| Endpoint batches | GPU_MEDIUM (A10G): about 10 cells per request at 2,048 genes for 70M, 8 for 400M; GPU_SMALL (T4): 10 for 70M. |
| Checks | Self-retrieval and reachability (Validation checklist item 10), dimension match (item 12), and recall@10 against exact search on held-out cells (workflow). |
| Query cells and "same type" scores | Pick held-out query cells from labels that are in the index, and avoid umbrella labels whose subtypes are also in it (Census brain: plain `neuron` next to `glutamatergic neuron` etc.). Neighbour agreement compares exact `cell_type` strings, so an umbrella-label query scores its subtype neighbours as misses (2026-11-02: 83/100 overall, with both misses from `neuron` queries). Report chance as that label's share of the index. Also say which reference types have no query cells: their accuracy is untested, and 10 of 10 can hide it (2026-11-09 demo: query cells covered 5 of 9 types, none of the 4 GABAergic interneuron types). For any vote below 10 of 10, show the neighbours' types; a split between related types (excitatory neuron vs pvalb interneuron) is expected, not a pipeline error. To check untested types without new data, run leave-one-out on reference cells (query with the cell's own embedding, k=11, drop self, vote over 10) and label it in-index (demo: 20 of 20 interneurons, agreement 7-10, misses pvalb vs sst). Reference map for the 9-type brain reference: `visualize.plot_reference_map(..., families={"red": ["L2/3-6 intratelencephalic projecting glutamatergic neuron"], "navy": ["VIP GABAergic cortical interneuron", "pvalb GABAergic cortical interneuron", "lamp5 GABAergic cortical interneuron", "sst GABAergic cortical interneuron"], "orange": ["oligodendrocyte", "oligodendrocyte precursor cell"], "teal": ["astrocyte", "microglial cell"]})`; this order gives pvalb a light and sst the darkest navy, so the pair most often confused stays distinguishable. |

### Short runs: do these, they cost a test run time (2026-11-02)

1. **One install cell, one restart, first.** Put every package in a single `%pip` cell before the config
   cell, e.g. `%pip install -q "transformers==4.41.0" "numpy>=1.26.4,<2.0" "pandas>=2.2.2,<3.0"
   "mlflow>=3.12" anndata scipy cellxgene-census`, then `dbutils.library.restartPython()` once.
   Installing Census later forced a second restart, which wiped `w` and the config.
2. **Config and `w = WorkspaceClient()` in one cell after the restart.** No re-defining config in later
   cells, no checkpoint or crash-recovery code.
3. **Census: filter in the query, before sampling.** Put every filter in `value_filter` and wrap values
   that contain `'` in double quotes:
   `'is_primary_data == True and tissue_general == "brain" and assay == "10x 3\' v3"'`. Read only the
   columns you need. Reading all primary brain cells and filtering in pandas took about 7 minutes.
4. **h5ad files: write to `/tmp`, then copy to the volume** (`adata.write_h5ad("/tmp/x.h5ad")`, then
   `shutil.copy`). Writing h5ad straight to `/Volumes/...` crashed the kernel (SIGSEGV).
5. **Vector Search: use `scripts/vs_index.py`, don't hand-write wait loops.** The endpoint status is an
   enum (`EndpointStatusState.ONLINE`); a loop comparing it with the string `"ONLINE"` never ends.
6. **Model signature types are attributes:** `ColSpec(DataType.string, "pooling")`, not
   `DataType[string]`.
7. **A new Vector Search endpoint's ONLINE status doesn't mean it is ready.** A new STANDARD endpoint
   reports ONLINE within seconds, but its compute only starts when its first index is created. That
   first index then takes 16-29 min to answer queries (2026-11-02: 16 and 18.6 min; 2026-11-08: 28.6 min), of which only
   about 1 min is the sync of about 1,000 rows. A new index on an endpoint that already holds one was
   ready in 0.7 min. For a short or live run, reuse an endpoint that already holds an index (confirm
   the name with the user), and report the time from index create to answering queries, not
   endpoint ONLINE.
8. **A new index can answer once and then say "not ready".** 2026-11-08 (a new index on a new
   STANDARD endpoint): the probe in `ensure_index` answered at 28.6 min, then the first
   self-retrieval query seconds later got "Vector index ... is not ready"; all 20 checks passed at
   29.9 min. `vs_index.py` now needs 3 answers in a row and `index_sanity_check(..., deadline=...)`
   retries "not ready"; pass a deadline in self-retrieval loops.

The subsections below ([AI Search reference corpus](#ai-search-reference-corpus-census-based) onward) are older notes from the `gt_teddy` notebook; their cell numbers refer to that notebook. Where they disagree with this section or the workflow, follow this section.

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
7. Embed in batches via the serving endpoint. **Batch size depends on GPU tier and model size**: 10 cells per request for 70M on `GPU_SMALL` (T4, 16 GB) and on `GPU_MEDIUM` (A10G, 24 GB); 8 for 400M. 50 ran out of memory on an A10G at 2,048 genes (2026-10-29). Larger batches on T4 cause `CUDA out of memory` — the 70M model's activations at 2048 seq_len × 512 hidden dim saturate 14.5 GB of available VRAM with >10 concurrent cells.
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
| **1,000** | **~1 min (notebook A10G, batch 16: 3,750 cells in 167 s, 2026-10-30)** / **~2 min endpoint** | **~20 min** | **GT default, good for PCA/t-SNE** |
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

### Pooling for zero-shot use

`cell_emb` is the output at position 0. With `add_cls=False` (70M), that is the most-expressed gene's output, so neighbours mostly share that gene rather than a cell type (measured 2026-10-30: 92% of 15 nearest neighbours shared it; zero-shot clustering NMI 0.51 vs 0.95 with mean pooling). Serve mean pooling instead: set `model.return_all_embs = True` and average `outputs["all_embs"]` over the attention mask. Don't use upstream's `cell_emb_mean`: it averages padding positions too, so a cell's vector depends on the other cells in its batch.

```python
# Controls: optional column on the row, else params, else default (integration-contract.md)
CONTROLS = {"pooling": ("mean", {"mean", "tutorial", "first"}), "max_seq_len": ("2048", None)}

def resolve_controls(row, columns, params):
    used = {}
    for name, (default, allowed) in CONTROLS.items():
        v = row[name] if name in columns and not pd.isna(row[name]) else (params or {}).get(name, default)
        v = str(v)
        if allowed and v not in allowed:
            raise ValueError(f"{name}={v!r} is not supported; use one of {sorted(allowed)}")
        used[name] = v
    return used                                    # echoed in each output dict

# load_context: once
model.return_all_embs = True

# _predict_batch: pad to 2,048 for "tutorial" (the tutorial's collate pads every cell to max_seq_len);
# "mean" and "first" don't depend on padding (cosine 0.9999999 and 1.0 alone vs in a batch).
H = outputs["all_embs"]                            # (cells, tokens, d_model)
M = attention_mask.unsqueeze(-1).to(H.dtype)       # 1 = expressed gene
if pooling == "mean":
    emb = (H * M).sum(1) / M.sum(1)
elif pooling == "tutorial":
    emb = H.mean(1)                                # all 2,048 positions, padding included
else:                                              # "first"
    emb = outputs["cell_emb"]
```

Prototype checked 2026-10-30 (MLflow 3.13, loaded model): column absent gives the defaults; a column value is used and echoed; `params` is used when the column is absent; the column wins when both are set; `"median"` raises the error above. Also checked through a live `GPU_MEDIUM` endpoint (2026-10-30): all of the above hold by both routes, and `serving_endpoints.query(extra_params=...)` was dropped (the echo showed the default).

**Default and options.** Default to `"mean"` and say so in one line; ask only if the user wants to reproduce earlier results or upstream's default output. Record the default as a model-version tag.

* `"mean"` (default): zero-shot search, clustering, or annotation by neighbour votes; the paper's method.
* `"tutorial"`: pad to 2,048 and average all positions, as in `tutorials/model_tutorial.ipynb`; for checking against the tutorial. A cell with 2,048 or more expressed genes has no padding, so `"tutorial"` equals `"mean"` for it. The tutorial embeds the integer counts in `.raw.X` of `data/sample_data.h5ad`; its `.X` is already normalised (max about 7.2) and gives different vectors.
* `"first"`: upstream's default `cell_emb` (what earlier deployments served).

Serve one model with a working `pooling` control, not several models: a request can override the default. The reference table and index must use the same model version and pooling as the queries, so store both in the table.

### Tensor-boolean safety in `_predict_batch`

TEDDY's forward pass returns a dict. For `pooling="mean"` read `all_embs` and pool over the mask ([Pooling for zero-shot use](#pooling-for-zero-shot-use)); the lookup below is for `pooling="first"`, where the key is `cell_emb` (first-position output, shape `(cells, d_model)`). Fallback keys `all_embs`, `last_hidden_state`, and `hidden_states` are retained for forward-compatibility with other variants. **Never use Python `or` or truthy evaluation on these tensors** — `bool(tensor)` raises `RuntimeError: Boolean value of Tensor with more than one element is ambiguous`. Always use explicit `is None` checks:

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

### Standard vs express deployment

Docs: [Express deployments for model serving endpoints](https://docs.databricks.com/aws/en/machine-learning/model-serving/express-deployments/) (formerly "serverless optimized deployments"). Express packages the notebook's environment when the model is registered, so the endpoint skips the container image build; the served environment is the notebook's.

| | Standard | Express |
| --- | --- | --- |
| Environment built | At endpoint creation (container image) | At registration (`env_pack`) |
| Served environment | From `pip_requirements`; can differ from the notebook | Identical to the registering notebook |
| TEDDY-70M, `GPU_MEDIUM` (measured) | 15-25 min to READY (2026-10-30) | Registration +25.8 s; 2.5 min to READY (2026-11-01); 2026-11-09: READY at the first check, 9.2 min after create (upper bound, not polled earlier) |
| Typical failure | Image build, dependency resolution | That class removed; the notebook environment must already work |
| Price | No separate express price in the docs; billing is for the serving compute either way | same |

**Use express (default when its requirements hold):** registering from a serverless GPU notebook (AI environment, version 3 or later) into Unity Catalog, `mlflow>=3.12`, `databricks-sdk>=0.102.0`, environment under 200 GB, serving on GPU (register on GPU to serve on GPU; CPU to CPU). Express is for **custom models** only, which TEDDY's PyFunc is.

**Use standard when:** registering from classic compute (Databricks Runtime ML) or into the workspace model registry; the serving environment must differ from the notebook (for example a pinned CUDA 12.x torch for an older driver); the compute types differ; or a model version already exists without `env_pack` (express needs a new registration, so a new version).

**Express pattern (tested 2026-11-01).** Log without `registered_model_name` (that path ignores `env_pack`), then register separately:

```python
from mlflow.utils.env_pack import EnvPackConfig

info = mlflow.pyfunc.log_model(name="model", python_model=WRAPPER_PATH,
                               artifacts={"model_dir": MODEL_DIR, "teddy_pkg_parent": CODE_DIR},
                               signature=signature, input_example=input_example, pip_requirements=PIP_REQS)
mv = mlflow.register_model(info.model_uri, UC_MODEL_NAME,
                           env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=True))
# If install_dependencies=True fails (custom code, no internet), retry once with False, as the docs suggest.
```

Then create the endpoint exactly as in standard (same `ServedEntityInput`, AI Gateway, tags); add the tag `deployment=express` to the model version and endpoint.

**Express caveats:**

* Everything installed in the notebook session is packaged. Keep the install cell minimal and pinned: `mlflow>=3.12`, `databricks-sdk>=0.102.0`, `transformers==4.41.0`, `anndata` (reading cells only). torch comes from the AI environment (tested: torch 2.11 with CUDA 13 on AI v6 served on `GPU_MEDIUM`); numpy 2.x from the AI environment worked, so the `numpy<2.0` pin in [Dependencies](#dependencies-from-pyprojecttoml) applies to standard only.
* The served environment is tied to the AI environment version registered from; re-registering on a newer version can change it, so re-run the fidelity check after re-registering.
* `pip_requirements` pin the notebook's own versions; drop torch's local build tag (`+cu130`) so the pin is installable.
* Run steps in order: verify the registered version loads and matches the notebook wrapper before creating the endpoint, then run the endpoint contract checks (served = wrapper, controls by both routes) as for standard.
* **Serving UI:** the Query endpoint box pre-fills from the model's `serving_input_example.json`. With a 3-cell x 22,029-gene example (1.73 MB) the box was empty on the express endpoint (seen 2026-11-01; size is the likely cause, and standard models logged the same way would carry the same example, not checked). Log a **one-cell example with only that cell's expressed genes** (about 40 kB): it gives the same embedding (cosine 0.999999), because zero genes change neither the total count nor top-K. Paste the same JSON (`{"dataframe_records": [ {...} ]}`) into the box to demo a query.
* **Inference table:** `<endpoint>_payload` appears at once with only `databricks_request_id` and 0 rows; full columns and rows arrive minutes later. Read it by printing its columns rather than assuming names.
* **Scaled-to-zero endpoint: small requests only.** 2026-11-09 (GPU_MEDIUM, about 2 h
  idle): a request of 10 cells x 61,497 genes (all genes, dense) was rejected with "Max request
  size exceeded for a scale to zero endpoint". The same request size worked while the endpoint was
  warm. One cell per request with only its expressed genes worked and woke the endpoint; it
  gives the same embedding (zero genes change neither the total count nor top-K). Send expressed
  genes only by default, at least for the first request after idle. The size limit was not measured.
* Not yet tested: 160M/400M, CPU serving.

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
4. **Signature test**: build the signature with `infer_signature(input_example, output_example, params=default_params)` using string defaults (see [Sending inference controls](#sending-inference-controls)); check the input schema has the three data columns (`adata_sparsematrix`, `adata_obs`, `adata_var`) as required, the controls (`pooling`, `max_seq_len`) as optional strings, and that `{p.name: p.dtype for p in signature.params.params}` lists the same controls as `DataType.string`.
5. **Real-gene payload test**: test payload uses Ensembl IDs from `vocab.txt`; embeddings are non-degenerate (not all the same value, norm > 0).
6. **All-OOV payload test**: all gene names set to random strings → every gene is dropped as out-of-vocabulary → the endpoint returns a clear error naming the vocabulary mismatch, not an embedding (upstream raises in the same case; see [Out-of-vocabulary genes](#out-of-vocabulary-oov-genes-drop-them-as-upstream-does)).
7. **Missing-field test**: omit `adata_obs` from the request → endpoint returns a schema-validation error (confirming the signature enforces all three fields).
8. **Inference-control test** (report the result; don't block): on the same real-gene payload, a non-default value must change the output and be echoed, through **both** routes — as a column via `w.serving_endpoints.query(dataframe_records=...)`, and under `"params"` via `invoke()`. Check each `pooling` value and a non-default `max_seq_len` such as `"32"` (values as strings; each cell must have more than 32 expressed genes, or truncation changes nothing) — and assert the embeddings differ (cosine clearly below 1.0). Before the calls, confirm the registered version's params schema (`mlflow.models.get_model_info(model_uri).signature.params`) lists `max_seq_len` and `pooling` as strings — an undeclared control is dropped. Identical output means the control never reached `predict()`; a successful call alone proves nothing. Repeat with `{"pooling": "first"}`: an unchanged output means the control is ignored. Check that `"tutorial"` differs from `"mean"` only on cells with fewer than 2,048 expressed genes: with no padding the two are equal by design.
9. **Embedding sanity**: for 70M, embedding dimension should match `config.d_model` (read from `config.json`).

10. **AI Search self-retrieval test**: query the index with a known cell's embedding, asking for at least 10 results (`num_results=1` missed the exact match for 3 of 50 cells, 2026-10-30); top-1 result must be the same cell (score ~1.0). Results must be monotonically ordered by similarity.
11. **Census gene ID test**: verify `adata.var_names` are Ensembl IDs (start with `ENSG`), not numeric indices, before intersecting with TEDDY vocab. Overlap should be ≥80% (22K+).
12. **AI Search dimension match test**: if switching variants, the index dimension must match `EMB_DIM` for the current variant. On a dim mismatch (e.g. 1024 from 400M vs 512 from 70M), stop and ask the user: use a new variant-suffixed index name, or replace the existing index only with their confirmation.
13. **Upstream-parity test** (SKILL.md Workflow step 7): run a few real cells through TEDDY's own preprocessing and tokenization and through the wrapper; the selected token sets must match and the per-cell embedding cosine must be at least 0.9999. Comparing the endpoint only with the same wrapper run in the notebook does not count. See [Upstream-parity test](#upstream-parity-test).
14. **Batch invariance**: a cell embedded alone and inside a batch of other cells gives the same vector (cosine >= 0.9999). Fails if padding leaks into pooling.
15. **Reference test** (`evaluation.md` layer 4): run [Intended use and reference test](#intended-use-and-reference-test) on the served output; report it next to the paper value and the PCA baseline.

## Upstream-parity test

Run in the notebook after the dry-load test, before registering — not in the serving container. Upstream `preprocess.py` and `tokenization.py` import `anndata`, `scanpy`, `scikit-learn`, `datasets` and `tqdm` at module level, so install those in the notebook only. With an express deployment (`env_pack`) the notebook environment is the serving environment, so don't add scanpy to the `%pip` cell: if `scanpy` is missing, install it for this test only into a side folder — `pip install -q --no-deps --target /tmp/<run>/parity_pkgs scanpy==1.11.2 legacy-api-wrap session-info2 natsort` (via `subprocess`), then `sys.path.insert(0, "/tmp/<run>/parity_pkgs")`. No Python restart, and nothing is packed with the model (tested 2026-11-06, AI environment v6: 5 cells, same gene sets, min cosine 0.9999992). `wrapper` is the dry-loaded `TEDDYEmbedder` instance, `default_params` comes from [Sending inference controls](#sending-inference-controls), and the full snapshot (with `teddy/data_processing/` and the upstream sample data `data/sample_data.h5ad`) is under `<VOLUME_PATH>/snapshot`.

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

Preferred: send controls as optional columns on the record (works with `w.serving_endpoints.query`; see [HTTP payload](#http-payload-databricks-sdk)). For callers that use `"params"`, they must arrive under the request-body key `"params"`, which the MLflow scoring server passes to `predict(..., params=...)`. `w.serving_endpoints.query()` has no `params` argument; its `extra_params` argument is written to a body key `"extra_params"`, which the server drops without an error (SKILL.md Troubleshooting #13). For `params`, post the body yourself:

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
output_example = output_example.assign(pooling="mean", max_seq_len="2048")  # documented echo
sig = infer_signature(input_example, output_example, params=default_params)

# Make the control columns optional (infer_signature marks every example column required)
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
signature = ModelSignature(
    inputs=Schema([ColSpec(c.type, c.name, required=c.name not in default_params) for c in sig.inputs.inputs]),
    outputs=sig.outputs, params=sig.params)
assert {p.name: p.dtype for p in signature.params.params} == {
    "max_seq_len": DataType.string, "pooling": DataType.string}
```

A control missing from the signature's params schema is dropped with only a log warning. A successful call does not prove a control was applied — run Validation checklist item 8.

## Open questions

* Whether HF_HUB_DISABLE_XET will remain necessary as HF updates the Xet client. The subprocess workaround is runtime-version-agnostic (tested on AI v5; expected to work on v6+).
* Whether the `transformers==4.41.0` exact pin will need to be updated for future TEDDY releases.
* Optimal `max_seq_len` for 160M and 400M variants (may differ from 70M default of 2048).
* Whether 160M/400M have `add_cls=True` (then `cell_emb` is a real cell token) and how first-token vs mean pooling compare on the reference test.
* Whether per-cell `adata_obs` metadata (batch label, donor ID) should be echoed back in the response to aid downstream correlation.
* Whether future Databricks GPU_SMALL tiers will provision A10G instead of T4 — monitor release notes.
* ~~What output key the model forward pass uses~~ **RESOLVED (oss-002 v2, 2026-10-01):** 70M returns `{"cell_emb": tensor(cells, d_model)}` (first-position output, 2D). Forward accepts `gene_ids` + `attention_mask` but NOT `gene_values`. Verify whether 160M/400M variants use the same output key or different ones.
