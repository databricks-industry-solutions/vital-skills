# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # TEDDY-70M Cell Embedding Serving and Cell-Type Search
# MAGIC
# MAGIC Downloads TEDDY-G 70M ([Merck/TEDDY](https://huggingface.co/Merck/TEDDY), Apache-2.0, [arXiv 2503.03485](https://arxiv.org/abs/2503.03485)), wraps raw-count preprocessing + rank tokenization + embedding in an MLflow pyfunc, registers it in Unity Catalog, deploys it to Model Serving, and builds a Vector Search index over reference cell embeddings for nearest-neighbor cell-type lookup.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# h5py: read the sample .h5ad without anndata/scanpy; uv: isolated serving-environment check before registration
%pip install -q h5py uv
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration
CATALOG = "<catalog>"
SCHEMA = "skills_owm_202610"

MODEL_NAME = f"{CATALOG}.{SCHEMA}.owm_teddy_model_002_baseline_r2"
SERVING_ENDPOINT = "owm_teddy_ep_002_baseline_r2"
VS_ENDPOINT = "owm_teddy_vs_002_baseline_r2"
EMB_TABLE = f"{CATALOG}.{SCHEMA}.owm_teddy_cells_002_baseline_r2"
VS_INDEX = f"{CATALOG}.{SCHEMA}.owm_teddy_index_002_baseline_r2"
FILES_DIR = "/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/002_baseline_r2"

HF_REPO = "Merck/TEDDY"
MODEL_SUBDIR = "teddy/models/teddy_g/70M"
MAX_SEQ_LEN = 2048      # TEDDY tokenization max_seq_len
NORM_TOTAL = 10_000     # TEDDY preprocessing normalized_total
EMB_DIM = 512           # TEDDY-G 70M d_model
QUERY_FRACTION = 0.2    # share of sample cells held out as queries (not used as neighbors)
K_NEIGHBORS = 10
SEED = 42

# Creating the serving endpoint and the Vector Search endpoint provisions billable compute.
# Set to True only after reviewing the planned specs printed by the workload-selection cell.
CONFIRM_CREATE_ENDPOINTS = True  # confirmed by user 2026-10-27

# COMMAND ----------

# DBTITLE 1,Download TEDDY-70M files to the Volume
import json, os, shutil, tempfile
from huggingface_hub import HfApi, snapshot_download

# Pin the exact commit so the files in the Volume are reproducible
HF_REVISION = HfApi().model_info(HF_REPO).sha
patterns = [
    "README.md",
    f"{MODEL_SUBDIR}/*",                       # weights, config, vocab, special tokens
    "teddy/__init__.py", "teddy/models/*.py", "teddy/models/teddy_g/*.py",  # reference model code (parity check)
    "teddy/data_processing/utils/medians/data/teddy_gene_medians.json",       # per-gene medians used in preprocessing
    "data/sample_data.h5ad", "data/sample_data_metadata.json",               # reference cells
]
tmp_dir = tempfile.mkdtemp()
snapshot_download(HF_REPO, revision=HF_REVISION, allow_patterns=patterns, local_dir=tmp_dir)

# Copy into the Volume (download locally first: HF lock files do not work on FUSE paths)
os.makedirs(FILES_DIR, exist_ok=True)
for root, _, files in os.walk(tmp_dir):
    if ".cache" in root.split(os.sep):
        continue
    for fn in files:
        src = os.path.join(root, fn)
        dst = os.path.join(FILES_DIR, os.path.relpath(src, tmp_dir))
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
with open(f"{FILES_DIR}/SOURCE.json", "w") as f:
    json.dump({"repo": HF_REPO, "revision": HF_REVISION, "license": "apache-2.0", "paper": "arXiv:2503.03485"}, f, indent=2)

MODEL_DIR = f"{FILES_DIR}/{MODEL_SUBDIR}"
ARTIFACTS = {
    "weights": f"{MODEL_DIR}/model.safetensors",
    "config": f"{MODEL_DIR}/config.json",
    "vocab": f"{MODEL_DIR}/vocab.txt",
    "added_tokens": f"{MODEL_DIR}/added_tokens.json",
    "medians": f"{FILES_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json",
}
SAMPLE_H5AD = f"{FILES_DIR}/data/sample_data.h5ad"

print("revision:", HF_REVISION)
for k, p in {**ARTIFACTS, "sample": SAMPLE_H5AD}.items():
    print(f"{k:13s} {os.path.getsize(p)/1e6:9.2f} MB  {p}")

# COMMAND ----------

# DBTITLE 1,TEDDY embedder (preprocess + tokenize + encode) as MLflow pyfunc
import json
import numpy as np
import pandas as pd
import torch
import mlflow
from torch import nn


class TeddyGEncoder(nn.Module):
    """Inference-only copy of the TeddyGModel encoder path (teddy/models/teddy_g/model.py):
    gene-token embedding + learned position embedding -> nlayers x nn.TransformerEncoderLayer.
    Plain torch, so serving does not depend on the transformers 4.41 API the original class needs."""

    def __init__(self, cfg):
        super().__init__()
        self.embeddings = nn.Embedding(cfg["ntoken"], cfg["d_model"])
        self.position_embeddings = nn.Embedding(cfg["max_position_embeddings"], cfg["d_model"])
        layer = nn.TransformerEncoderLayer(
            cfg["d_model"], cfg["nheads"], cfg["d_hid"], cfg.get("dropout", 0.0),
            batch_first=True, norm_first=cfg.get("pre_norm", False),
            activation=cfg.get("layer_activation", "relu"),
        )
        self.encoder = nn.TransformerEncoder(layer, cfg["nlayers"], enable_nested_tensor=False)

    def forward(self, gene_ids, attention_mask):
        pos = torch.arange(gene_ids.shape[1], device=gene_ids.device)
        x = self.embeddings(gene_ids) + self.position_embeddings(pos)
        pad_mask = ~attention_mask.bool()  # torch convention: True = ignore
        for layer in self.encoder.layers:
            x = layer(src=x, src_key_padding_mask=pad_mask)
        return x


def tokenize_cell(gene_ids, counts, vocab, medians, max_len=2048, norm_total=10_000):
    """TEDDY-G preprocessing + rank tokenization for one cell (raw counts in, token ids out):
    normalize to `norm_total`, divide by per-gene TEDDY medians, keep vocab genes with value > 0,
    sort descending, truncate to `max_len` (no CLS, no zero genes - the released 70M settings)."""
    genes = np.asarray(gene_ids, dtype=object)
    x = np.asarray(counts, dtype=np.float64)
    total = x.sum()
    if total <= 0:
        return np.empty(0, dtype=np.int64)
    x = x / total * norm_total
    x = x / np.array([medians.get(g, 1.0) for g in genes])
    tok = np.array([vocab.get(g, -1) for g in genes], dtype=np.int64)
    keep = (tok >= 0) & (x > 0)
    tok, x = tok[keep], x[keep]
    order = np.argsort(-x, kind="stable")[:max_len]
    return tok[order]


class TeddyEmbedder(mlflow.pyfunc.PythonModel):
    """Input: one row per cell with `gene_ids` (Ensembl IDs) and `counts` (raw UMI counts, same order).
    Output: `embedding` = L2-normalised masked-mean of TEDDY-G 70M token embeddings (512-d), `n_tokens`."""

    def load_context(self, context):
        from safetensors.torch import load_file

        a = context.artifacts
        with open(a["config"]) as f:
            cfg = json.load(f)
        with open(a["vocab"]) as f:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(f) if line.strip()}
        with open(a["added_tokens"]) as f:
            self.pad_id = json.load(f)["<pad>"]
        with open(a["medians"]) as f:
            self.medians = json.load(f)
        self.max_len = min(MAX_LEN_DEFAULT, cfg["max_position_embeddings"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32 if self.device.type == "cuda" else 4
        state = {k: v for k, v in load_file(a["weights"]).items() if not k.startswith("decoder_")}  # drop MLM head
        self.model = TeddyGEncoder(cfg)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def tokenize(self, gene_ids, counts):
        return tokenize_cell(gene_ids, counts, self.vocab, self.medians, self.max_len, NORM_TOTAL_DEFAULT)

    @torch.inference_mode()
    def embed_tokens(self, token_lists):
        out = []
        for s in range(0, len(token_lists), self.batch_size):
            batch = token_lists[s:s + self.batch_size]
            L = max(1, max(len(t) for t in batch))
            ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
            for i, t in enumerate(batch):
                ids[i, :len(t)] = torch.as_tensor(t, dtype=torch.long)
            mask = (ids != self.pad_id).long()
            ids, mask = ids.to(self.device), mask.to(self.device)
            h = self.model(ids, mask)
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict(self, context, model_input, params=None):
        df = pd.DataFrame(model_input)
        tokens = [self.tokenize(g, c) for g, c in zip(df["gene_ids"], df["counts"])]
        emb = self.embed_tokens(tokens)
        n_tok = [len(t) for t in tokens]
        emb = [e.tolist() if n > 0 else None for e, n in zip(emb, n_tok)]  # empty cells -> null embedding
        return pd.DataFrame({"embedding": emb, "n_tokens": n_tok})


# Module-level constants captured by cloudpickle with the class
MAX_LEN_DEFAULT = MAX_SEQ_LEN
NORM_TOTAL_DEFAULT = NORM_TOTAL

# COMMAND ----------

# DBTITLE 1,Load reference cells from the sample h5ad (raw counts)
import h5py
import scipy.sparse as sp


def _strings(ds):
    return np.array([v.decode() if isinstance(v, bytes) else v for v in ds[:]], dtype=object)


def _obs_col(obs, name):
    g = obs[name]
    if isinstance(g, h5py.Group):  # categorical
        return _strings(g["categories"])[g["codes"][:]]
    return _strings(g)


with h5py.File(SAMPLE_H5AD, "r") as f:
    raw = f["raw"]["X"]  # raw UMI counts (adata.X is already log-normalised)
    X = sp.csr_matrix((raw["data"][:], raw["indices"][:], raw["indptr"][:]), shape=tuple(raw.attrs["shape"]))
    var_ids = _strings(f["raw"]["var"]["_index"])
    obs = f["obs"]
    cells_pdf = pd.DataFrame({
        "cell_id": _strings(obs["_index"]),
        "cell_subtype": _obs_col(obs, "Cell.Types"),       # author-annotated interneuron subtype
        "cell_type": _obs_col(obs, "cell_type"),
        "cell_type_ontology_term_id": _obs_col(obs, "cell_type_ontology_term_id"),
        "donor_id": _obs_col(obs, "donor_id"),
        "disease": _obs_col(obs, "disease"),
        "tissue": _obs_col(obs, "tissue"),
    })

# Sparse rows -> (gene_ids, counts) lists of expressed genes; totals are unchanged because zeros add nothing
cells_pdf["gene_ids"] = [var_ids[X.indices[X.indptr[i]:X.indptr[i + 1]]].tolist() for i in range(X.shape[0])]
cells_pdf["counts"] = [X.data[X.indptr[i]:X.indptr[i + 1]].astype(float).tolist() for i in range(X.shape[0])]

# Stratified hold-out of query cells (excluded from neighbor candidates at search time)
rng = np.random.default_rng(SEED)
cells_pdf["split"] = "reference"
for _, idx in cells_pdf.groupby("cell_subtype").groups.items():
    idx = np.array(idx)
    cells_pdf.loc[rng.choice(idx, size=int(round(len(idx) * QUERY_FRACTION)), replace=False), "split"] = "query"

assert cells_pdf["cell_id"].is_unique
print(X.shape, "| integer counts:", np.allclose(X.data, np.round(X.data)))
display(cells_pdf.groupby(["cell_subtype", "split"]).size().unstack(fill_value=0))

# COMMAND ----------

# DBTITLE 1,Instantiate embedder locally and check parity with the official TEDDY code
import sys
from types import SimpleNamespace
from safetensors.torch import load_file

embedder = TeddyEmbedder()
embedder.load_context(SimpleNamespace(artifacts=ARTIFACTS))
print("device:", embedder.device, "| vocab:", len(embedder.vocab), "| pad_id:", embedder.pad_id,
      "| genes in vocab:", f"{np.mean([g in embedder.vocab for g in var_ids]):.1%}",
      "| genes with median:", f"{np.mean([g in embedder.medians for g in var_ids]):.1%}")

# Reference: the repo's own TeddyGModel.forward (its HF init hook is incompatible with transformers 5, so skip it)
sys.path.insert(0, FILES_DIR)
from teddy.models.teddy_g.model import TeddyGConfig, TeddyGModel
TeddyGModel.init_weights = lambda self: None
ref = TeddyGModel(TeddyGConfig.from_pretrained(MODEL_DIR))
ref.load_state_dict(load_file(ARTIFACTS["weights"]), strict=True)
ref.return_all_embs = True
ref.to(embedder.device).eval()

toks = [embedder.tokenize(g, c) for g, c in zip(cells_pdf["gene_ids"][:4], cells_pdf["counts"][:4])]
L = max(len(t) for t in toks)
ids = torch.full((len(toks), L), embedder.pad_id, dtype=torch.long)
for i, t in enumerate(toks):
    ids[i, :len(t)] = torch.as_tensor(t)
mask = (ids != embedder.pad_id).long()
with torch.inference_mode():
    ours = embedder.model(ids.to(embedder.device), mask.to(embedder.device))
    theirs = ref(gene_ids=ids.to(embedder.device), attention_mask=mask.to(embedder.device), return_outputs=True)["all_embs"]
m = mask.bool().to(embedder.device)
max_diff = (ours[m] - theirs[m]).abs().max().item()
print("tokens per cell:", [len(t) for t in toks], "| max |diff| on real tokens:", max_diff)
assert max_diff < 1e-4, "encoder does not match the official TeddyGModel forward"
del ref

# COMMAND ----------

# DBTITLE 1,Log the model with signature and input example
import cloudpickle, safetensors
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")

input_example = cells_pdf[["gene_ids", "counts"]].head(2).reset_index(drop=True)
example_out = embedder.predict(None, input_example)
signature = infer_signature(input_example, example_out)
print(signature)

pip_requirements = [
    f"mlflow=={mlflow.__version__}",
    f"cloudpickle=={cloudpickle.__version__}",
    f"torch=={torch.__version__.split('+')[0]}",
    f"safetensors=={safetensors.__version__}",
    f"numpy=={np.__version__}",
    f"pandas=={pd.__version__}",
]

with mlflow.start_run(run_name="teddy_g_70m_embedder_002_baseline_r2") as run:
    mlflow.log_params({"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "variant": "TEDDY-G 70M",
                       "max_seq_len": MAX_SEQ_LEN, "normalized_total": NORM_TOTAL,
                       "pooling": "masked_mean_l2norm", "embedding_dim": EMB_DIM})
    model_info = mlflow.pyfunc.log_model(
        name="teddy_embedder",
        python_model=TeddyEmbedder(),
        artifacts=ARTIFACTS,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
    )
print(model_info.model_uri)

# COMMAND ----------

# DBTITLE 1,Validate the artifact in an isolated environment
# Installs pip_requirements from scratch (as the serving container will) and runs the stored input example
loaded = mlflow.pyfunc.load_model(model_info.model_uri)
assert loaded.metadata.signature is not None, "missing signature"
representative_input = loaded.input_example
assert representative_input is not None, "missing input example"

mlflow.models.predict(
    model_uri=model_info.model_uri,
    input_data=representative_input,
    env_manager="uv",
)

# COMMAND ----------

# DBTITLE 1,Register the validated model in Unity Catalog
registered_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=MODEL_NAME,
    await_registration_for=300,
)
model_version = registered_version.version
print(f"{MODEL_NAME} version {model_version} -> models:/{MODEL_NAME}/{model_version}")

# COMMAND ----------

# DBTITLE 1,Embed reference cells and write the embeddings table
import time
from pyspark.sql import types as T

t0 = time.time()
cell_tokens = [embedder.tokenize(g, c) for g, c in zip(cells_pdf["gene_ids"], cells_pdf["counts"])]
cell_emb = embedder.embed_tokens(cell_tokens)
print(f"embedded {len(cell_emb)} cells in {time.time() - t0:.0f}s on {embedder.device}")

emb_pdf = cells_pdf.drop(columns=["gene_ids", "counts"]).copy()
emb_pdf["n_tokens"] = [len(t) for t in cell_tokens]
emb_pdf["embedding"] = [e.astype(np.float32).tolist() for e in cell_emb]
emb_pdf = emb_pdf[emb_pdf["n_tokens"] > 0].reset_index(drop=True)

if spark.catalog.tableExists(EMB_TABLE):
    raise RuntimeError(f"{EMB_TABLE} already exists - confirm before overwriting it")

spark.sql(f"""
CREATE TABLE {EMB_TABLE} (
  cell_id STRING NOT NULL,
  split STRING,
  cell_subtype STRING,
  cell_type STRING,
  cell_type_ontology_term_id STRING,
  donor_id STRING,
  disease STRING,
  tissue STRING,
  n_tokens INT,
  embedding ARRAY<FLOAT>,
  CONSTRAINT owm_teddy_cells_002_baseline_r2_pk PRIMARY KEY (cell_id)
)
COMMENT 'TEDDY-G 70M cell embeddings (L2-normalised masked-mean, {EMB_DIM}-d) for Merck/TEDDY sample_data.h5ad @ {HF_REVISION}'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
schema = T.StructType([
    T.StructField("cell_id", T.StringType(), False),
    *[T.StructField(c, T.StringType()) for c in ["split", "cell_subtype", "cell_type", "cell_type_ontology_term_id",
                                                 "donor_id", "disease", "tissue"]],
    T.StructField("n_tokens", T.IntegerType()),
    T.StructField("embedding", T.ArrayType(T.FloatType())),
])
spark.createDataFrame(emb_pdf[[f.name for f in schema.fields]], schema=schema).write.mode("append").saveAsTable(EMB_TABLE)
display(spark.sql(f"SELECT split, cell_subtype, count(*) n, min(size(embedding)) dmin, max(size(embedding)) dmax, "
                  f"round(avg(n_tokens)) avg_tokens FROM {EMB_TABLE} GROUP BY ALL ORDER BY ALL"))

# COMMAND ----------

# DBTITLE 1,Sanity check: local exact kNN cell-type accuracy
# Exact cosine kNN in numpy - independent baseline for what the Vector Search index should reproduce
ref_mask = (emb_pdf["split"] == "reference").to_numpy()
E = np.stack(emb_pdf["embedding"].to_numpy())
E_ref, E_q = E[ref_mask], E[~ref_mask]
y_ref, y_q = emb_pdf["cell_subtype"].to_numpy()[ref_mask], emb_pdf["cell_subtype"].to_numpy()[~ref_mask]
nn_idx = np.argsort(-(E_q @ E_ref.T), axis=1)[:, :K_NEIGHBORS]
pred = np.array([pd.Series(y_ref[r]).mode().iloc[0] for r in nn_idx])

majority = pd.Series(y_ref).mode().iloc[0]
print(f"kNN (k={K_NEIGHBORS}) subtype accuracy: {np.mean(pred == y_q):.3f} | "
      f"majority-class baseline ('{majority}'): {np.mean(y_q == majority):.3f} | n_query={len(y_q)}")
display(pd.crosstab(pd.Series(y_q, name="true"), pd.Series(pred, name="predicted")))

# COMMAND ----------

# DBTITLE 1,Select serving workload
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
workload_response = w.api_client.do("GET", "/api/2.0/serving-endpoints:workload-configs")
supported_workloads = {
    c["workload_type"]: [s["key"] for s in c.get("workload_sizes", [])]
    for c in workload_response.get("workload_configs", [])
}
print("host:", w.config.host)
print("supported:", supported_workloads)

# TEDDY-70M: 285 MB fp32 weights, no CUDA-only code (loads on CPU, uses CUDA only if present).
# Peak attention memory for a 4-cell x 2048-token CPU batch is ~1 GB -> fits CPU (4 GB per concurrency, AWS docs).
workload_type = "CPU"
assert workload_type in supported_workloads, f"{workload_type} not offered in this workspace"
workload_size = "Small" if "Small" in supported_workloads[workload_type] else supported_workloads[workload_type][0]
scale_to_zero_enabled = True  # avoids idle cost; first request after inactivity has a cold start

print(f"""Planned serving endpoint
  name:           {SERVING_ENDPOINT}
  model:          {MODEL_NAME} v{model_version}
  workload:       {workload_type} / {workload_size}
  scale_to_zero:  {scale_to_zero_enabled}
Planned Vector Search
  endpoint:       {VS_ENDPOINT} (STANDARD)
  index:          {VS_INDEX} (Delta Sync, self-managed {EMB_DIM}-d embeddings, TRIGGERED)""")

# COMMAND ----------

# DBTITLE 1,Create the Model Serving endpoint
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType

assert CONFIRM_CREATE_ENDPOINTS, "Review the planned specs above, then set CONFIRM_CREATE_ENDPOINTS = True"

served_entity = ServedEntityInput(
    entity_name=MODEL_NAME,
    entity_version=str(model_version),
    workload_type=ServingModelWorkloadType(workload_type),
    workload_size=workload_size,
    scale_to_zero_enabled=scale_to_zero_enabled,
)

try:
    existing_endpoint = w.serving_endpoints.get(SERVING_ENDPOINT)
except NotFound:
    existing_endpoint = None

if existing_endpoint is None:
    w.serving_endpoints.create(name=SERVING_ENDPOINT, config=EndpointCoreConfigInput(name=SERVING_ENDPOINT, served_entities=[served_entity]))
    print(f"Creating {SERVING_ENDPOINT} (typically ~15 min)")
else:
    current = [e.entity_name for e in (existing_endpoint.config.served_entities or [])] if existing_endpoint.config else []
    if current != [MODEL_NAME]:
        raise ValueError(f"Refusing to replace {SERVING_ENDPOINT}, which serves {current}")
    w.serving_endpoints.update_config(name=SERVING_ENDPOINT, served_entities=[served_entity])
    print(f"Updating {SERVING_ENDPOINT} to v{model_version}")

# COMMAND ----------

# DBTITLE 1,Wait for serving endpoint readiness
import time
from databricks.sdk.service.serving import EndpointStateConfigUpdate, EndpointStateReady


def wait_for_endpoint_ready(name, timeout_s=2400, poll_s=30):
    deadline = time.time() + timeout_s
    failed = {EndpointStateConfigUpdate.UPDATE_FAILED, EndpointStateConfigUpdate.UPDATE_CANCELED}
    while time.time() < deadline:
        state = w.serving_endpoints.get(name).state
        if state.ready == EndpointStateReady.READY and state.config_update == EndpointStateConfigUpdate.NOT_UPDATING:
            return
        if state.config_update in failed:
            raise RuntimeError(f"{name} deployment failed: {state.config_update.value}")
        time.sleep(poll_s)
    raise TimeoutError(f"{name} not ready after {timeout_s}s")


wait_for_endpoint_ready(SERVING_ENDPOINT)
print(SERVING_ENDPOINT, "is READY")

# COMMAND ----------

# DBTITLE 1,Create the Vector Search endpoint
import time
from databricks.sdk.errors import NotFound
from databricks.sdk.service import vectorsearch

assert CONFIRM_CREATE_ENDPOINTS, "Review the planned specs above, then set CONFIRM_CREATE_ENDPOINTS = True"

try:
    w.vector_search_endpoints.get_endpoint(VS_ENDPOINT)
    print(f"{VS_ENDPOINT} already exists - reusing it")
except NotFound:
    w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT, endpoint_type=vectorsearch.EndpointType.STANDARD)
    print(f"Creating {VS_ENDPOINT}")

deadline = time.time() + 1800
while True:
    st = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT).endpoint_status
    if st.state == vectorsearch.EndpointStatusState.ONLINE:
        break
    if st.state == vectorsearch.EndpointStatusState.OFFLINE or time.time() > deadline:
        raise RuntimeError(f"{VS_ENDPOINT}: {st.state} {st.message}")
    time.sleep(30)
print(VS_ENDPOINT, "is ONLINE")

# COMMAND ----------

# DBTITLE 1,Create and sync the Delta Sync index
INDEX_COLUMNS = ["cell_id", "split", "cell_subtype", "cell_type", "cell_type_ontology_term_id",
                 "donor_id", "disease", "tissue", "n_tokens", "embedding"]

try:
    if w.vector_search_indexes.get_index(VS_INDEX).status.ready:
        print(f"{VS_INDEX} already exists - triggering sync")
        w.vector_search_indexes.sync_index(VS_INDEX)
    else:
        print(f"{VS_INDEX} already exists and is still provisioning - waiting")
except NotFound:
    w.vector_search_indexes.create_index(
        name=VS_INDEX,
        endpoint_name=VS_ENDPOINT,
        primary_key="cell_id",
        index_type=vectorsearch.VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=vectorsearch.DeltaSyncVectorIndexSpecRequest(
            source_table=EMB_TABLE,
            embedding_vector_columns=[vectorsearch.EmbeddingVectorColumn(name="embedding", embedding_dimension=EMB_DIM)],
            pipeline_type=vectorsearch.PipelineType.TRIGGERED,
            columns_to_sync=INDEX_COLUMNS,
        ),
    )
    print(f"Creating {VS_INDEX}")

n_expected = spark.table(EMB_TABLE).count()
deadline = time.time() + 2400
while True:
    st = w.vector_search_indexes.get_index(VS_INDEX).status
    if st.ready and (st.indexed_row_count or 0) >= n_expected:
        break
    if time.time() > deadline:
        raise TimeoutError(f"{VS_INDEX}: {st.message}")
    time.sleep(30)
print(f"{VS_INDEX} ready: {st.indexed_row_count} rows indexed (table has {n_expected})")

# COMMAND ----------

# DBTITLE 1,Nearest-neighbor cell-type search via serving endpoint + Vector Search
from collections import Counter


def embed_via_endpoint(records, batch_size=4):
    """records: list of {'gene_ids': [...Ensembl...], 'counts': [...raw counts...]}.
    Small batches keep each request well under the serving timeout (~3 s/cell on CPU)."""
    out = []
    for s in range(0, len(records), batch_size):
        resp = w.serving_endpoints.query(name=SERVING_ENDPOINT, dataframe_records=records[s:s + batch_size])
        out += [p["embedding"] for p in resp.predictions]
    return out


def search_cell_type(embedding, k=K_NEIGHBORS, reference_only=True):
    res = w.vector_search_indexes.query_index(
        index_name=VS_INDEX,
        columns=["cell_id", "cell_subtype", "cell_type", "donor_id"],
        query_vector=embedding,
        num_results=k,
        filters_json=json.dumps({"split": "reference"}) if reference_only else None,
    )
    cols = [c.name for c in res.manifest.columns]
    hits = pd.DataFrame(res.result.data_array or [], columns=cols)
    votes = Counter(hits["cell_subtype"])
    label, n = votes.most_common(1)[0]
    return label, n / len(hits), hits


# End-to-end on held-out query cells: raw counts -> endpoint embedding -> ANN neighbors -> majority vote
queries = cells_pdf[cells_pdf["split"] == "query"].groupby("cell_subtype").head(3).reset_index(drop=True)
served = embed_via_endpoint(queries[["gene_ids", "counts"]].to_dict("records"))

local = embedder.predict(None, queries[["gene_ids", "counts"]])["embedding"]
cos = [float(np.dot(a, b)) for a, b in zip(served, local)]
print(f"served vs local embedding cosine: min={min(cos):.6f}")

rows = []
for (_, q), e in zip(queries.iterrows(), served):
    label, conf, hits = search_cell_type(e)
    rows.append({"cell_id": q["cell_id"], "true_subtype": q["cell_subtype"], "predicted_subtype": label,
                 "vote_share": round(conf, 2), "top_hit": hits["cell_id"].iloc[0], "top_score": hits["score"].iloc[0]})
result_pdf = pd.DataFrame(rows)
print(f"accuracy on {len(result_pdf)} query cells: {(result_pdf.true_subtype == result_pdf.predicted_subtype).mean():.3f}")
display(result_pdf)