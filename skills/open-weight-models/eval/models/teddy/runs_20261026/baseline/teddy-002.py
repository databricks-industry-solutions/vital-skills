# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # TEDDY-70M Serving and Cell-Type Nearest-Neighbor Search
# MAGIC
# MAGIC Downloads Merck TEDDY-G 70M (Apache-2.0, [arXiv 2503.03485](https://arxiv.org/abs/2503.03485)) from Hugging Face, wraps it as an MLflow pyfunc that maps raw single-cell expression to 512-d cell embeddings, registers it in Unity Catalog, serves it, and builds a Vector Search index over reference cell embeddings for nearest-neighbor cell-type lookup.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q anndata huggingface_hub safetensors "databricks-sdk>=0.40" uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration
CATALOG = "<catalog>"
SCHEMA = "skills_owm_202610"

MODEL_NAME = f"{CATALOG}.{SCHEMA}.owm_teddy_model_002_baseline"
SERVING_ENDPOINT = "owm_teddy_ep_002_baseline"
VS_ENDPOINT = "owm_teddy_vs_002_baseline"
CELLS_TABLE = f"{CATALOG}.{SCHEMA}.owm_teddy_cells_002_baseline"
VS_INDEX = f"{CATALOG}.{SCHEMA}.owm_teddy_index_002_baseline"

VOL_DIR = "/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/002_baseline"
REPO_DIR = f"{VOL_DIR}/TEDDY"          # mirror of the HF repo subset
MODEL_DIR = f"{REPO_DIR}/teddy/models/teddy_g/70M"
MEDIANS_JSON = f"{REPO_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json"
SAMPLE_H5AD = f"{REPO_DIR}/data/sample_data.h5ad"

HF_REPO = "Merck/TEDDY"
MAX_SEQ_LEN = 2048   # tokenizer max_seq_len used for TEDDY pretraining
EMB_DIM = 512        # d_model of TEDDY-G 70M

# COMMAND ----------

# DBTITLE 1,Download TEDDY-70M files to the Volume
import os, shutil, tempfile
from huggingface_hub import snapshot_download

# Only the 70M weights, the teddy python package, tokenizer/medians/mappings, and sample data
allow = [
    "README.md", "pyproject.toml",
    "teddy/**/*.py",
    "teddy/models/teddy_g/70M/*",
    "teddy/tokenizer/*",
    "teddy/data_processing/utils/medians/data/*",
    "teddy/data_processing/utils/bio_annotations/data/mappings/*",
    "data/*",
]

if not os.path.exists(f"{MODEL_DIR}/model.safetensors"):
    tmp = tempfile.mkdtemp()
    local = snapshot_download(HF_REPO, local_dir=f"{tmp}/TEDDY", allow_patterns=allow)
    os.makedirs(VOL_DIR, exist_ok=True)
    shutil.copytree(local, REPO_DIR, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".cache", ".DS_Store"))

for root, _, files in os.walk(MODEL_DIR):
    for f in files:
        p = os.path.join(root, f)
        print(f"{os.path.getsize(p)/1e6:8.2f} MB  {p}")

# COMMAND ----------

# DBTITLE 1,Define TEDDY-G encoder and inspect sample data
import json, torch
from torch import nn
import anndata as ad
from safetensors.torch import load_file


class TeddyGEncoder(nn.Module):
    """Pure-PyTorch re-implementation of the TeddyGModel inference path (same parameter names).

    The upstream class subclasses transformers.PreTrainedModel for transformers 4.41 and fails to
    initialise on current transformers, so inference uses this equivalent module instead.
    Mirrors teddy/models/teddy_g/model.py: token + position embeddings -> TransformerEncoder (post-norm),
    cell embedding = first-token hidden state (return_cell_embs_first_token=True).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.embeddings = nn.Embedding(cfg["ntoken"], cfg["d_model"])
        self.position_embeddings = nn.Embedding(cfg["max_position_embeddings"], cfg["d_model"])
        layer = nn.TransformerEncoderLayer(
            cfg["d_model"], cfg["nheads"], cfg["d_hid"], cfg.get("dropout", 0.0),
            batch_first=True, norm_first=cfg.get("pre_norm", False),
            activation=cfg.get("layer_activation", "relu"),
        )
        self.encoder = nn.TransformerEncoder(layer, cfg["nlayers"], enable_nested_tensor=False)

    @classmethod
    def from_dir(cls, model_dir, device="cpu"):
        cfg = json.load(open(f"{model_dir}/config.json"))
        m = cls(cfg)
        sd = load_file(f"{model_dir}/model.safetensors")
        # drop MLM decoder weights (not needed for embeddings)
        sd = {k: v for k, v in sd.items() if not k.startswith("decoder_")}
        m.load_state_dict(sd, strict=True)
        return m.to(device).eval()

    @torch.no_grad()
    def forward(self, gene_ids, attention_mask):
        x = self.embeddings(gene_ids.long())
        pos = torch.arange(gene_ids.shape[1], device=gene_ids.device)
        x = x + self.position_embeddings(pos)
        pad_mask = ~attention_mask.bool()  # PyTorch convention: True = ignore
        for layer in self.encoder.layers:
            x = layer(src=x, src_key_padding_mask=pad_mask)
        return x[:, 0, :], x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = TeddyGEncoder.from_dir(MODEL_DIR, device)
print("device:", device, "| params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))
print("safetensors keys not used:", [k for k in load_file(f"{MODEL_DIR}/model.safetensors") if k.startswith("decoder_")])

vocab = [l.rstrip("\n") for l in open(f"{MODEL_DIR}/vocab.txt")]
print("vocab size:", len(vocab), "first tokens:", vocab[:8], "...", vocab[1000:1003])
print("special tokens:", json.load(open(f"{MODEL_DIR}/special_tokens_map.json")))

medians = json.load(open(MEDIANS_JSON))
print("medians:", len(medians), list(medians.items())[:3])

adata = ad.read_h5ad(SAMPLE_H5AD)
print(adata)
print(adata.var.head(3))
print(adata.obs.head(3))

# COMMAND ----------

# DBTITLE 1,TEDDY preprocessing and tokenization
import numpy as np

added = json.load(open(f"{MODEL_DIR}/added_tokens.json"))
PAD_ID = added["<pad>"]
gene2tok = {g: i for i, g in enumerate(vocab)}


def tokenize_cell(gene_ids, counts, gene2tok, medians, max_len=MAX_SEQ_LEN):
    """TEDDY-G recipe for one cell (raw counts, Ensembl IDs):
    normalize total to 10,000 -> divide by TEDDY corpus gene medians -> keep genes in vocab with
    non-zero value -> rank descending -> take top max_len tokens (no CLS, rank encodes position).
    Genes missing from the vocab or medians table are dropped."""
    g = np.asarray(gene_ids)
    c = np.asarray(counts, dtype=np.float64)
    keep = (c > 0) & np.array([x in gene2tok and x in medians for x in g], dtype=bool)
    g, c = g[keep], c[keep]
    if len(c) == 0:
        return [], 0
    total = c.sum()
    v = c / total * 10_000 / np.array([medians[x] for x in g])
    order = np.argsort(-v, kind="stable")[:max_len]
    return [gene2tok[x] for x in g[order]], len(c)


def pad_batch(token_lists, pad_id=PAD_ID):
    L = max(1, max(len(t) for t in token_lists))
    ids = np.full((len(token_lists), L), pad_id, dtype=np.int64)
    mask = np.zeros((len(token_lists), L), dtype=bool)
    for i, t in enumerate(token_lists):
        ids[i, : len(t)] = t
        mask[i, : len(t)] = True
    return torch.from_numpy(ids), torch.from_numpy(mask)


def embed_tokens(model, token_lists, device, batch_size=32):
    """Returns (first-token cell_emb, masked mean-pooled emb), both [n, 512] float32."""
    first, mean = [], []
    for s in range(0, len(token_lists), batch_size):
        ids, mask = pad_batch(token_lists[s : s + batch_size])
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            cls_emb, hidden = model(ids, mask)
        m = mask.unsqueeze(-1).float()
        mean.append(((hidden.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
        first.append(cls_emb.float().cpu())
    return torch.cat(first).numpy(), torch.cat(mean).numpy()


# Use raw counts (adata.raw) per TEDDY preprocessing; adata.X in the sample is log-normalized
raw = adata.raw.to_adata()
var_ids = raw.var_names.to_numpy()
X = raw.X.tocsr()
tokens, n_expr = [], []
for i in range(X.shape[0]):
    row = X.getrow(i)
    t, n = tokenize_cell(var_ids[row.indices], row.data, gene2tok, medians)
    tokens.append(t); n_expr.append(n)
lens = np.array([len(t) for t in tokens])
print(f"cells: {len(tokens)} | tokens/cell min/median/max: {lens.min()}/{int(np.median(lens))}/{lens.max()}")
print(f"genes in vocab & medians: {np.mean([g in gene2tok and g in medians for g in var_ids]):.1%}")

# COMMAND ----------

# DBTITLE 1,Embed sample cells and compare pooling via held-out-donor kNN
import pandas as pd

emb_first, emb_mean = embed_tokens(model, tokens, device)
print(emb_first.shape, emb_mean.shape, "NaNs:", np.isnan(emb_first).sum(), np.isnan(emb_mean).sum())

obs = adata.obs
labels = obs["Cell.Types"].astype(str).to_numpy()   # fine interneuron subtypes
donors = obs["donor_id"].astype(str).to_numpy()


def loo_donor_knn_acc(E, labels, donors, k=10):
    """Leave-one-donor-out cosine kNN majority-vote accuracy (what the Vector Search lookup does)."""
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    correct = 0
    for d in np.unique(donors):
        q, r = donors == d, donors != d
        nn_idx = np.argsort(-(En[q] @ En[r].T), axis=1)[:, :k]
        ref = labels[r]
        pred = [pd.Series(ref[row]).mode()[0] for row in nn_idx]
        correct += (np.array(pred) == labels[q]).sum()
    return correct / len(labels)


results = pd.DataFrame({
    "pooling": ["first_token (upstream default)", "mean_over_genes"],
    "loo_donor_knn10_acc": [loo_donor_knn_acc(emb_first, labels, donors), loo_donor_knn_acc(emb_mean, labels, donors)],
})
results["majority_class_baseline"] = pd.Series(labels).value_counts(normalize=True).iloc[0]
display(results)

# COMMAND ----------

# DBTITLE 1,Define MLflow pyfunc and log the model
import mlflow, safetensors
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, Array


class TeddyEmbedder(mlflow.pyfunc.PythonModel):
    """Raw single-cell counts -> TEDDY-G 70M cell embeddings.

    Input  (one row per cell): gene_ids array<string> (Ensembl IDs), counts array<double> (raw UMI counts,
           non-zero genes only is fine).
    Output: cell_emb array<float>[512] (mean over gene tokens; used for Vector Search),
            cell_emb_first array<float>[512] (first-token embedding, upstream default), n_tokens int.
    """

    def load_context(self, context):
        import json, torch
        a = context.artifacts
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TeddyGEncoder.from_dir(a["model_dir"], self.device)
        vocab = [l.rstrip("\n") for l in open(f"{a['model_dir']}/vocab.txt")]
        self.gene2tok = {g: i for i, g in enumerate(vocab)}
        self.pad_id = json.load(open(f"{a['model_dir']}/added_tokens.json"))["<pad>"]
        self.medians = json.load(open(a["medians"]))
        self.max_len = 2048

    def _tokenize(self, gene_ids, counts):
        import numpy as np
        g = np.asarray(gene_ids)
        c = np.asarray(counts, dtype=np.float64)
        keep = (c > 0) & np.array([x in self.gene2tok and x in self.medians for x in g], dtype=bool)
        g, c = g[keep], c[keep]
        if len(c) == 0:
            return []
        v = c / c.sum() * 10_000 / np.array([self.medians[x] for x in g])
        order = np.argsort(-v, kind="stable")[: self.max_len]
        return [self.gene2tok[x] for x in g[order]]

    def predict(self, context, model_input, params=None):
        import numpy as np, pandas as pd, torch
        toks = [self._tokenize(g, c) for g, c in zip(model_input["gene_ids"], model_input["counts"])]
        bs = 32
        first, mean = [], []
        for s in range(0, len(toks), bs):
            chunk = [t if t else [self.pad_id] for t in toks[s : s + bs]]
            L = max(len(t) for t in chunk)
            ids = torch.full((len(chunk), L), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), L), dtype=torch.bool)
            for i, t in enumerate(chunk):
                ids[i, : len(t)] = torch.tensor(t)
                mask[i, : len(t)] = True
            ids, mask = ids.to(self.device), mask.to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device == "cuda")):
                cls_emb, hidden = self.model(ids, mask)
            m = mask.unsqueeze(-1).float()
            mean.append(((hidden.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
            first.append(cls_emb.float().cpu())
        mean = torch.cat(mean).numpy().astype(np.float32)
        first = torch.cat(first).numpy().astype(np.float32)
        return pd.DataFrame({
            "cell_emb": list(mean),
            "cell_emb_first": list(first),
            "n_tokens": [len(t) for t in toks],
        })


# Input example: 2 real sample cells (non-zero genes only)
def cell_row(i):
    r = X.getrow(i)
    return {"gene_ids": var_ids[r.indices].tolist(), "counts": r.data.astype(float).tolist()}

input_example = pd.DataFrame([cell_row(0), cell_row(1)])

signature = ModelSignature(
    inputs=Schema([ColSpec(Array("string"), "gene_ids"), ColSpec(Array("double"), "counts")]),
    outputs=Schema([ColSpec(Array("float"), "cell_emb"), ColSpec(Array("float"), "cell_emb_first"),
                    ColSpec("long", "n_tokens")]),
)

# Pin to the versions in this notebook (strip local build tags like +cu126)
base = lambda v: v.split("+")[0]
pip_reqs = [
    f"mlflow=={mlflow.__version__}",
    f"torch=={base(torch.__version__)}",
    f"safetensors=={safetensors.__version__}",
    f"numpy=={np.__version__}",
    f"pandas=={pd.__version__}",
]
print(pip_reqs)

# In-notebook sanity check before logging
_m = TeddyEmbedder()
class _Ctx: artifacts = {"model_dir": MODEL_DIR, "medians": MEDIANS_JSON}
_m.load_context(_Ctx())
_out = _m.predict(None, input_example)
print("max |diff| vs notebook embeddings:", float(np.abs(np.stack(_out.cell_emb) - emb_mean[:2]).max()))

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name="teddy_g_70m_embedder_002_baseline"):
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=TeddyEmbedder(),
        artifacts={"model_dir": MODEL_DIR, "medians": MEDIANS_JSON},
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_reqs,
    )
    mlflow.log_params({"hf_repo": HF_REPO, "variant": "teddy_g/70M", "license": "apache-2.0",
                       "paper": "arXiv:2503.03485", "pooling_for_vs": "mean", "max_seq_len": MAX_SEQ_LEN})
print(model_info.model_uri)

# COMMAND ----------

# DBTITLE 1,Isolated serving-readiness check
# Rebuilds the declared pip_requirements from scratch with uv (same as a serving container)
pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
assert pyfunc_model.metadata.signature is not None and pyfunc_model.input_example is not None

mlflow.models.predict(
    model_uri=model_info.model_uri,
    input_data=pyfunc_model.input_example,
    env_manager="uv",
)

# COMMAND ----------

# DBTITLE 1,Register the validated model in Unity Catalog
mlflow.set_registry_uri("databricks-uc")
registered_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=MODEL_NAME,
    await_registration_for=600,
)
MODEL_VERSION = registered_version.version
print(f"Registered models:/{MODEL_NAME}/{MODEL_VERSION}")

# COMMAND ----------

# DBTITLE 1,Write reference cell embeddings table
from pyspark.sql import types as T

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CELLS_TABLE} (
  cell_id        STRING NOT NULL COMMENT 'AnnData obs index',
  cell_type      STRING COMMENT 'CELLxGENE cell_type label',
  cell_subtype   STRING COMMENT 'Fine interneuron subtype (obs Cell.Types)',
  cell_type_ontology_term_id STRING,
  tissue         STRING,
  disease        STRING,
  donor_id       STRING,
  n_tokens       INT    COMMENT 'Number of gene tokens fed to TEDDY (max 2048)',
  source         STRING COMMENT 'Source h5ad',
  embedding      ARRAY<FLOAT> COMMENT 'TEDDY-G 70M mean-pooled cell embedding (512-d)',
  CONSTRAINT owm_teddy_cells_002_baseline_pk PRIMARY KEY (cell_id)
)
COMMENT 'TEDDY-G 70M reference cell embeddings for Vector Search cell-type lookup'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

pdf = pd.DataFrame({
    "cell_id": obs.index.astype(str),
    "cell_type": obs["cell_type"].astype(str).values,
    "cell_subtype": obs["Cell.Types"].astype(str).values,
    "cell_type_ontology_term_id": obs["cell_type_ontology_term_id"].astype(str).values,
    "tissue": obs["tissue"].astype(str).values,
    "disease": obs["disease"].astype(str).values,
    "donor_id": obs["donor_id"].astype(str).values,
    "n_tokens": lens.astype("int32"),
    "source": "Merck/TEDDY data/sample_data.h5ad",
    "embedding": [e.astype(np.float32).tolist() for e in emb_mean],
})
schema = spark.table(CELLS_TABLE).schema
cells_sdf = spark.createDataFrame(pdf, schema=T.StructType([T.StructField(f.name, f.dataType, True) for f in schema]))
cells_sdf.write.insertInto(CELLS_TABLE, overwrite=True)

display(spark.sql(f"SELECT cell_subtype, count(*) n, min(size(embedding)) dim FROM {CELLS_TABLE} GROUP BY 1 ORDER BY 2 DESC"))

# COMMAND ----------

# DBTITLE 1,Create the Model Serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType

w = WorkspaceClient()

# 48.7M-param fp32 model (~200 MB weights) with no CUDA requirement (device is auto-detected), so the
# smallest CPU type (4 GB per concurrency, AWS Custom models overview) fits. Switch to GPU_SMALL for lower latency.
WORKLOAD_TYPE, WORKLOAD_SIZE, SCALE_TO_ZERO = "CPU", "Small", True

supported = {c["workload_type"]: [s["key"] for s in c.get("workload_sizes", [])]
             for c in w.api_client.do("GET", "/api/2.0/serving-endpoints:workload-configs").get("workload_configs", [])}
assert WORKLOAD_SIZE in supported.get(WORKLOAD_TYPE, []), supported

served_entity = ServedEntityInput(
    entity_name=MODEL_NAME,
    entity_version=str(MODEL_VERSION),
    workload_type=ServingModelWorkloadType(WORKLOAD_TYPE),
    workload_size=WORKLOAD_SIZE,
    scale_to_zero_enabled=SCALE_TO_ZERO,
)

try:
    existing = w.serving_endpoints.get(SERVING_ENDPOINT)
except NotFound:
    existing = None

if existing is None:
    w.serving_endpoints.create(name=SERVING_ENDPOINT, config=EndpointCoreConfigInput(name=SERVING_ENDPOINT, served_entities=[served_entity]))
    print(f"Creating {SERVING_ENDPOINT} -> {MODEL_NAME} v{MODEL_VERSION}")
else:
    names = [e.entity_name for e in (existing.config.served_entities or [])] if existing.config else []
    if names != [MODEL_NAME]:
        raise ValueError(f"Refusing to replace {SERVING_ENDPOINT}, which serves {names}")
    w.serving_endpoints.update_config(name=SERVING_ENDPOINT, served_entities=[served_entity])
    print(f"Updating {SERVING_ENDPOINT} -> {MODEL_NAME} v{MODEL_VERSION}")

# COMMAND ----------

# DBTITLE 1,Wait for serving endpoint readiness
import time
from databricks.sdk.service.serving import EndpointStateConfigUpdate, EndpointStateReady


def wait_for_endpoint_ready(name, timeout_s=2700, poll_s=30):
    deadline = time.time() + timeout_s
    failed = {EndpointStateConfigUpdate.UPDATE_FAILED, EndpointStateConfigUpdate.UPDATE_CANCELED}
    while time.time() < deadline:
        st = w.serving_endpoints.get(name).state
        if st.ready == EndpointStateReady.READY and st.config_update == EndpointStateConfigUpdate.NOT_UPDATING:
            print(f"{name} is READY")
            return
        if st.config_update in failed:
            raise RuntimeError(f"{name} deployment failed: {st.config_update.value}")
        time.sleep(poll_s)
    raise TimeoutError(f"{name} not ready after {timeout_s}s")


wait_for_endpoint_ready(SERVING_ENDPOINT)

# COMMAND ----------

# DBTITLE 1,Create Vector Search endpoint and Delta Sync index
import time
from databricks.sdk.service import vectorsearch as vs

# Standard endpoint: small reference set, low-latency lookups
try:
    w.vector_search_endpoints.get_endpoint(VS_ENDPOINT)
    print(f"VS endpoint {VS_ENDPOINT} exists")
except NotFound:
    w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT, endpoint_type=vs.EndpointType.STANDARD)
    print(f"Creating VS endpoint {VS_ENDPOINT}")

while True:
    state = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT).endpoint_status.state
    if state == vs.EndpointStatusState.ONLINE:
        break
    if state in (vs.EndpointStatusState.OFFLINE,):
        raise RuntimeError(f"VS endpoint state {state}")
    time.sleep(30)
print(f"{VS_ENDPOINT}: ONLINE")

# Self-managed embeddings (TEDDY vectors already in the table), triggered sync
try:
    idx = w.vector_search_indexes.get_index(VS_INDEX)
    if idx.status and idx.status.ready:
        print(f"Index {VS_INDEX} exists; triggering sync")
        w.vector_search_indexes.sync_index(VS_INDEX)
    else:
        print(f"Index {VS_INDEX} exists; initial sync still running")
except NotFound:
    w.vector_search_indexes.create_index(
        name=VS_INDEX,
        endpoint_name=VS_ENDPOINT,
        primary_key="cell_id",
        index_type=vs.VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=vs.DeltaSyncVectorIndexSpecRequest(
            source_table=CELLS_TABLE,
            embedding_vector_columns=[vs.EmbeddingVectorColumn(name="embedding", embedding_dimension=EMB_DIM)],
            pipeline_type=vs.PipelineType.TRIGGERED,
            columns_to_sync=["cell_id", "cell_type", "cell_subtype", "cell_type_ontology_term_id",
                             "tissue", "disease", "donor_id", "embedding"],
        ),
    )
    print(f"Creating index {VS_INDEX}")

while True:
    st = w.vector_search_indexes.get_index(VS_INDEX).status
    if st.ready:
        break
    time.sleep(30)
print(f"{VS_INDEX} ready, indexed rows: {st.indexed_row_count}")

# COMMAND ----------

# DBTITLE 1,End-to-end: serving endpoint embedding to nearest-neighbor cell type
from collections import Counter


def predict_cell_type(cells_df, k=10, exclude_donor=None, cells_per_request=2):
    """cells_df: columns gene_ids (Ensembl IDs) and counts (raw UMIs). Embeds via the serving
    endpoint, then majority-votes cell_subtype over the k nearest reference cells.
    CPU serving takes ~5-10 s per 2,048-token cell, so requests are kept small to stay under the timeout."""
    recs = cells_df.to_dict("records")
    preds_all = []
    for s in range(0, len(recs), cells_per_request):
        preds_all += w.serving_endpoints.query(name=SERVING_ENDPOINT,
                                               dataframe_records=recs[s : s + cells_per_request]).predictions
    out = []
    for pred in preds_all:
        res = w.vector_search_indexes.query_index(
            index_name=VS_INDEX,
            columns=["cell_id", "cell_subtype", "cell_type", "donor_id"],
            query_vector=pred["cell_emb"],
            num_results=k,
            # held-out-donor check: exclude the query donor's own cells from the neighbor pool
            filters_json=json.dumps({"donor_id NOT": exclude_donor}) if exclude_donor else None,
        )
        cols = [c.name for c in res.manifest.columns]
        hits = [dict(zip(cols, r)) for r in res.result.data_array]
        votes = Counter(h["cell_subtype"] for h in hits)
        label, n = votes.most_common(1)[0]
        out.append({"predicted_subtype": label, "vote_frac": n / len(hits),
                    "top_hit": hits[0]["cell_id"], "top_score": hits[0]["score"]})
    return pd.DataFrame(out)


# Demo on 8 cells from one donor, excluding that donor from the neighbor pool
demo_donor = donors[0]
demo_idx = np.where(donors == demo_donor)[0][:8]
q = pd.DataFrame([cell_row(i) for i in demo_idx])
preds = predict_cell_type(q, k=10, exclude_donor=demo_donor)
preds.insert(0, "true_subtype", labels[demo_idx])
preds.insert(0, "cell_id", obs.index[demo_idx])
print("accuracy:", (preds.true_subtype == preds.predicted_subtype).mean())
display(preds)