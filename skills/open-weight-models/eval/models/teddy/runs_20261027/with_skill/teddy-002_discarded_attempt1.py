# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # TEDDY-70M Serving and Cell-Type Nearest-Neighbor Search
# MAGIC
# MAGIC Downloads Merck TEDDY-G 70M (Apache-2.0, [arXiv 2503.03485](https://arxiv.org/abs/2503.03485)) from Hugging Face, packages it as an MLflow pyfunc that turns raw single-cell counts into a 512-d cell embedding, registers it in Unity Catalog, deploys it to Model Serving, and indexes reference-cell embeddings in Vector Search for nearest-neighbor cell-type lookup.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q huggingface_hub safetensors anndata torch transformers==4.46.3 uv databricks-vectorsearch "databricks-sdk>=0.40"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration
import os

CATALOG = "<catalog>"
SCHEMA = "skills_owm_202610"
VOLUME = "owm_teddy_files"

MODEL_NAME = f"{CATALOG}.{SCHEMA}.owm_teddy_model_002_with_skill_r2"
SERVING_ENDPOINT = "owm_teddy_ep_002_with_skill_r2"
VS_ENDPOINT = "owm_teddy_vs_002_with_skill_r2"
EMB_TABLE = f"{CATALOG}.{SCHEMA}.owm_teddy_cells_002_with_skill_r2"
VS_INDEX = f"{CATALOG}.{SCHEMA}.owm_teddy_index_002_with_skill_r2"

FILES_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/teddy-eval-2026-10-27/002_with_skill"
HF_REPO = "Merck/TEDDY"
HF_DIR = f"{FILES_DIR}/hf_Merck_TEDDY"           # snapshot of the needed repo files
CODE_DIR = HF_DIR                                  # contains the `teddy` python package
TEDDY_PKG_DIR = f"{HF_DIR}/teddy"
MODEL_DIR = f"{HF_DIR}/teddy/models/teddy_g/70M"   # config.json, model.safetensors, vocab.txt
MEDIANS_PATH = f"{HF_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json"
SAMPLE_H5AD = f"{HF_DIR}/data/sample_data.h5ad"    # reference cells (CELLxGENE, 3,538 inhibitory neurons)

# Embedding settings (match TEDDY-G pretraining tokenization: top-2048 genes by median-normalized expression)
MAX_SEQ_LEN = 2048
EMB_DIM = 512
MAX_CELLS = None   # set e.g. 500 for a quicker first run; None = all reference cells

# COMMAND ----------

# DBTITLE 1,Download TEDDY-70M files from Hugging Face to the volume
from huggingface_hub import snapshot_download

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
os.makedirs(HF_DIR, exist_ok=True)

# Only the 70M checkpoint, the minimal model code, the gene-median table, and the sample reference data
snapshot_download(
    repo_id=HF_REPO,
    local_dir=HF_DIR,
    allow_patterns=[
        "README.md",
        "teddy/__init__.py",
        "teddy/models/__init__.py",
        "teddy/models/classification_heads.py",
        "teddy/models/model_directory.py",
        "teddy/models/teddy_g/__init__.py",
        "teddy/models/teddy_g/model.py",
        "teddy/models/teddy_g/70M/*",
        "teddy/data_processing/utils/medians/data/teddy_gene_medians.json",
        "data/sample_data.h5ad",
        "data/sample_data_metadata.json",
    ],
)
for root, _, files in os.walk(HF_DIR):
    for f in files:
        p = os.path.join(root, f)
        if ".cache" not in p:
            print(f"{os.path.getsize(p)/1e6:9.2f} MB  {p.replace(HF_DIR + '/', '')}")

# COMMAND ----------

# DBTITLE 1,TEDDY embedder (raw counts -> 512-d cell embedding)
import sys, json
import numpy as np
import pandas as pd


class TeddyEmbedder:
    """Raw counts -> TEDDY-G cell embedding.

    Replicates the TEDDY preprocessing + tokenization used for pretraining:
      1. normalize total counts to 10,000
      2. divide each gene by its TEDDY corpus median (teddy_gene_medians.json; missing genes -> 1.0)
      3. keep non-zero genes present in the vocab, rank descending, truncate to 2,048 tokens
    The cell embedding is the attention-masked mean of the final hidden states, L2-normalized
    (so Vector Search L2 distance is equivalent to cosine similarity).
    """

    PAD_TOKEN = "<pad>"

    def __init__(self, model_dir, medians_path, code_dir=None, max_seq_len=2048, batch_size=8, num_threads=None):
        if code_dir and code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        import torch
        from teddy.models.teddy_g.model import TeddyGConfig, TeddyGModel

        if num_threads:
            torch.set_num_threads(num_threads)
        self.torch = torch
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        with open(f"{model_dir}/vocab.txt") as f:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}
        with open(f"{model_dir}/added_tokens.json") as f:
            self.pad_id = json.load(f)[self.PAD_TOKEN]
        with open(medians_path) as f:
            self.medians = json.load(f)
        config = TeddyGConfig.from_pretrained(model_dir)
        self.model = TeddyGModel.from_pretrained(model_dir, config=config).eval()
        self.model.return_all_embs = True

    def tokenize(self, genes, counts):
        genes = np.asarray(list(genes), dtype=object)
        counts = np.asarray(list(counts), dtype=np.float64)
        total = counts.sum()
        if len(genes) == 0 or total <= 0:
            return []
        vals = counts / total * 10000.0
        vals = vals / np.array([self.medians.get(g, 1.0) for g in genes])
        keep = (vals > 0) & np.array([g in self.vocab for g in genes])
        g_keep, v_keep = genes[keep], vals[keep]
        order = np.argsort(-v_keep, kind="stable")[: self.max_seq_len]
        return [self.vocab[g] for g in g_keep[order]]

    def embed(self, gene_lists, count_lists):
        torch = self.torch
        tokens = [self.tokenize(g, c) for g, c in zip(gene_lists, count_lists)]
        out = [None] * len(tokens)
        valid = [i for i, t in enumerate(tokens) if len(t) > 0]
        for s in range(0, len(valid), self.batch_size):
            idx = valid[s : s + self.batch_size]
            L = max(len(tokens[i]) for i in idx)
            ids = torch.full((len(idx), L), self.pad_id, dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, : len(tokens[i])] = torch.tensor(tokens[i], dtype=torch.long)
            mask = (ids != self.pad_id).long()
            with torch.no_grad():
                h = self.model(gene_ids=ids, attention_mask=mask)["all_embs"]
            m = mask.unsqueeze(-1).float()
            emb = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)
            for r, i in enumerate(idx):
                out[i] = emb[r].tolist()
        return out, [len(t) for t in tokens]


# Load reference cells and smoke-test locally on a few cells
import anndata as ad

adata = ad.read_h5ad(SAMPLE_H5AD)
X_raw = (adata.raw.X if adata.raw is not None else adata.X).tocsr()   # TEDDY preprocessing starts from raw counts
var_ids = np.asarray(adata.raw.var_names if adata.raw is not None else adata.var_names)

def cell_profile(i):
    row = X_raw[i]
    return var_ids[row.indices].tolist(), row.data.astype(float).tolist()

embedder = TeddyEmbedder(MODEL_DIR, MEDIANS_PATH, code_dir=CODE_DIR, max_seq_len=MAX_SEQ_LEN)
profiles = [cell_profile(i) for i in range(3)]
vecs, n_used = embedder.embed([p[0] for p in profiles], [p[1] for p in profiles])
print("genes used per cell:", n_used, "| dim:", len(vecs[0]), "| norms:", [round(float(np.linalg.norm(v)), 4) for v in vecs])
print(adata.obs[["cell_type", "Cell.Types", "disease", "donor_id"]].head(3))

# COMMAND ----------

# DBTITLE 1,Log TEDDY as an MLflow pyfunc
import mlflow, torch, transformers, safetensors
from packaging.version import Version
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, Array


class TeddyCellEmbeddingModel(mlflow.pyfunc.PythonModel):
    """Input: one row per cell with `gene_ids` (Ensembl IDs) and raw `counts` (same order; zeros optional).
    Output: `embedding` (512 floats, L2-normalized) and `n_genes_used` (tokens fed to the model)."""

    def load_context(self, context):
        # `teddy` package is shipped via code_paths and is on sys.path in the serving container
        self.embedder = TeddyEmbedder(
            context.artifacts["model_dir"], context.artifacts["medians"], max_seq_len=2048, batch_size=8
        )

    def predict(self, context, model_input, params=None):
        vecs, n_used = self.embedder.embed(list(model_input["gene_ids"]), list(model_input["counts"]))
        return pd.DataFrame({"embedding": vecs, "n_genes_used": n_used})


signature = ModelSignature(
    inputs=Schema([ColSpec(Array("string"), "gene_ids"), ColSpec(Array("double"), "counts")]),
    outputs=Schema([ColSpec(Array("float"), "embedding"), ColSpec("long", "n_genes_used")]),
)
g0, c0 = cell_profile(0)
input_example = pd.DataFrame({"gene_ids": [g0], "counts": [c0]})

pip_requirements = [
    f"mlflow=={mlflow.__version__}",
    f"torch=={torch.__version__.split('+')[0]}",
    f"transformers=={transformers.__version__}",
    f"safetensors=={safetensors.__version__}",
    f"numpy=={np.__version__}",
    f"pandas=={pd.__version__}",
]
name_kw = {"name": "teddy_g_70m"} if Version(mlflow.__version__) >= Version("3.0") else {"artifact_path": "teddy_g_70m"}

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name="teddy_g_70m_pyfunc") as run:
    mlflow.log_params({"hf_repo": HF_REPO, "variant": "TEDDY-G 70M", "max_seq_len": MAX_SEQ_LEN, "pooling": "masked_mean_l2"})
    model_info = mlflow.pyfunc.log_model(
        python_model=TeddyCellEmbeddingModel(),
        artifacts={"model_dir": MODEL_DIR, "medians": MEDIANS_PATH},
        code_paths=[TEDDY_PKG_DIR],
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        **name_kw,
    )
print(model_info.model_uri)
print(pip_requirements)

# COMMAND ----------

# DBTITLE 1,Validate the artifact in an isolated environment
# Serving installs pip_requirements from scratch; this check reproduces that before registration.
pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
assert pyfunc_model.metadata.signature is not None, "missing signature"
representative_input = pyfunc_model.input_example
assert representative_input is not None, "missing input example"

mlflow.models.predict(
    model_uri=model_info.model_uri,
    input_data=representative_input,
    env_manager="uv",
)

# COMMAND ----------

# DBTITLE 1,Register the model in Unity Catalog
registered_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=MODEL_NAME,
    await_registration_for=600,
)
model_version = registered_version.version
print(f"Registered {MODEL_NAME} version {model_version}  (models:/{MODEL_NAME}/{model_version})")

# COMMAND ----------

# DBTITLE 1,Choose a supported serving workload
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
workload_response = w.api_client.do("GET", "/api/2.0/serving-endpoints:workload-configs")
supported_workloads = {
    cfg["workload_type"]: [s["key"] for s in cfg.get("workload_sizes", [])]
    for cfg in workload_response.get("workload_configs", [])
}
print(supported_workloads)

# TEDDY-G 70M: ~285 MB fp32 weights, loaded with map to CPU by default (no CUDA calls in the code path);
# the TEDDY tutorial runs the 70M variant on CPU. CPU workload is sufficient.
workload_type = "CPU"
assert workload_type in supported_workloads, f"{workload_type} not supported here: {list(supported_workloads)}"
workload_size = "Small" if "Small" in supported_workloads[workload_type] else supported_workloads[workload_type][0]
scale_to_zero_enabled = True   # avoids idle cost; first request after inactivity has a cold start
print(workload_type, workload_size, scale_to_zero_enabled)

# COMMAND ----------

# DBTITLE 1,Create or update the serving endpoint (~15 min)
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType

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
    w.serving_endpoints.create(name=SERVING_ENDPOINT, config=EndpointCoreConfigInput(served_entities=[served_entity]))
    print(f"Creating endpoint {SERVING_ENDPOINT}")
else:
    current = [e.entity_name for e in (existing_endpoint.config.served_entities or [])] if existing_endpoint.config else []
    if current != [MODEL_NAME]:
        raise ValueError(f"Refusing to replace {SERVING_ENDPOINT}, which serves {current}. Choose another name or confirm replacement.")
    w.serving_endpoints.update_config(name=SERVING_ENDPOINT, served_entities=[served_entity])
    print(f"Updating endpoint {SERVING_ENDPOINT} to version {model_version}")

# COMMAND ----------

# DBTITLE 1,Wait for endpoint readiness
import time
from databricks.sdk.service.serving import EndpointStateConfigUpdate, EndpointStateReady


def wait_for_endpoint_ready(name, timeout_s=2400, poll_s=30):
    deadline = time.time() + timeout_s
    failed = {EndpointStateConfigUpdate.UPDATE_FAILED, EndpointStateConfigUpdate.UPDATE_CANCELED}
    while time.time() < deadline:
        state = w.serving_endpoints.get(name).state
        if state.ready == EndpointStateReady.READY and state.config_update == EndpointStateConfigUpdate.NOT_UPDATING:
            print(f"{name} is READY")
            return
        if state.config_update in failed:
            raise RuntimeError(f"{name} deployment failed: {state.config_update.value}")
        time.sleep(poll_s)
    raise TimeoutError(f"{name} not ready after {timeout_s}s")


wait_for_endpoint_ready(SERVING_ENDPOINT)

# COMMAND ----------

# DBTITLE 1,Smoke-test the serving endpoint
def embed_via_endpoint(gene_ids, counts):
    resp = w.serving_endpoints.query(
        name=SERVING_ENDPOINT,
        dataframe_split={"columns": ["gene_ids", "counts"], "data": [[list(gene_ids), list(counts)]]},
    )
    pred = resp.predictions[0]
    return pred["embedding"], pred["n_genes_used"]


ep_vec, ep_n = embed_via_endpoint(g0, c0)
local_vec = vecs[0]
print("dim:", len(ep_vec), "| genes used:", ep_n, "| cosine(endpoint, local):", float(np.dot(ep_vec, local_vec)))

# COMMAND ----------

# DBTITLE 1,Compute reference-cell embeddings and write the Delta table
import pandas as pd
from pyspark.sql import types as T

# Reference-cell metadata (small) from the h5ad obs
n_cells = adata.n_obs if MAX_CELLS is None else min(MAX_CELLS, adata.n_obs)
obs = adata.obs.iloc[:n_cells]
meta_pdf = pd.DataFrame({
    "row_idx": np.arange(n_cells, dtype=np.int64),
    "cell_id": obs.index.astype(str),
    "cell_type": obs["cell_type"].astype(str).values,        # CELLxGENE ontology label
    "cell_subtype": obs["Cell.Types"].astype(str).values,    # author-annotated interneuron subtype
    "disease": obs["disease"].astype(str).values,
    "tissue": obs["tissue"].astype(str).values,
    "sex": obs["sex"].astype(str).values,
    "donor_id": obs["donor_id"].astype(str).values,
    "source": "Merck/TEDDY data/sample_data.h5ad",
})

# Distributed inference: each task reads the h5ad and TEDDY files from the volume and embeds its slice
_model_dir, _medians, _code_dir, _h5ad, _max_len = MODEL_DIR, MEDIANS_PATH, CODE_DIR, SAMPLE_H5AD, MAX_SEQ_LEN

def embed_partition(batches):
    import anndata as ad
    import numpy as np
    import pandas as pd
    emb = TeddyEmbedder(_model_dir, _medians, code_dir=_code_dir, max_seq_len=_max_len, batch_size=8, num_threads=4)
    a = ad.read_h5ad(_h5ad)
    X = (a.raw.X if a.raw is not None else a.X).tocsr()
    genes_all = np.asarray(a.raw.var_names if a.raw is not None else a.var_names)
    for pdf in batches:
        idx = pdf["row_idx"].tolist()
        g = [genes_all[X[i].indices].tolist() for i in idx]
        c = [X[i].data.astype(float).tolist() for i in idx]
        v, n = emb.embed(g, c)
        yield pd.DataFrame({"row_idx": idx, "embedding": v, "n_genes_used": n})

out_schema = T.StructType([
    T.StructField("row_idx", T.LongType()),
    T.StructField("embedding", T.ArrayType(T.FloatType())),
    T.StructField("n_genes_used", T.LongType()),
])
n_parts = max(1, min(64, n_cells // 25))
emb_sdf = (
    spark.range(n_cells).withColumnRenamed("id", "row_idx")
    .repartition(n_parts)
    .mapInPandas(embed_partition, schema=out_schema)
)
cells_sdf = spark.createDataFrame(meta_pdf).join(emb_sdf, "row_idx").drop("row_idx")

# Self-managed-embedding Delta Sync source: primary key + Change Data Feed
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {EMB_TABLE} (
  cell_id STRING NOT NULL,
  cell_type STRING,
  cell_subtype STRING,
  disease STRING,
  tissue STRING,
  sex STRING,
  donor_id STRING,
  source STRING,
  n_genes_used BIGINT,
  embedding ARRAY<FLOAT>,
  CONSTRAINT owm_teddy_cells_002_pk PRIMARY KEY (cell_id)
)
COMMENT 'TEDDY-G 70M cell embeddings (512-d, L2-normalized masked-mean pooling) for nearest-neighbor cell-type search'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Upsert (non-destructive: existing rows are updated by cell_id, nothing is deleted)
cells_sdf.createOrReplaceTempView("teddy_new_cells")
spark.sql(f"""
MERGE INTO {EMB_TABLE} t
USING teddy_new_cells s
ON t.cell_id = s.cell_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

display(spark.sql(f"""
SELECT count(*) AS n_cells, count_if(embedding IS NULL) AS null_embeddings,
       min(size(embedding)) AS min_dim, max(size(embedding)) AS max_dim, avg(n_genes_used) AS avg_genes_used
FROM {EMB_TABLE}
"""))

# COMMAND ----------

# DBTITLE 1,Create the Vector Search endpoint
from databricks.sdk.service import vectorsearch

# Standard endpoint: low-latency queries; a few thousand 512-d vectors is well within capacity
existing_vs = {e.name for e in w.vector_search_endpoints.list_endpoints()}
if VS_ENDPOINT not in existing_vs:
    w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT, endpoint_type=vectorsearch.EndpointType.STANDARD)
    print(f"Creating Vector Search endpoint {VS_ENDPOINT}")

deadline = time.time() + 1800
while True:
    st = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT).endpoint_status
    state = st.state.value if st and st.state else None
    if state == "ONLINE":
        print(f"{VS_ENDPOINT} is ONLINE")
        break
    if state in ("OFFLINE", "RED_STATE") or time.time() > deadline:
        raise RuntimeError(f"{VS_ENDPOINT} state={state}: {st.message if st else ''}")
    time.sleep(30)

# COMMAND ----------

# DBTITLE 1,Create or sync the Delta Sync index (self-managed embeddings)
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

try:
    w.vector_search_indexes.get_index(VS_INDEX)
    index_exists = True
except (NotFound, ResourceDoesNotExist):
    index_exists = False

if not index_exists:
    w.vector_search_indexes.create_index(
        name=VS_INDEX,
        endpoint_name=VS_ENDPOINT,
        primary_key="cell_id",
        index_type=vectorsearch.VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=vectorsearch.DeltaSyncVectorIndexSpecRequest(
            source_table=EMB_TABLE,
            embedding_vector_columns=[vectorsearch.EmbeddingVectorColumn(name="embedding", embedding_dimension=EMB_DIM)],
            pipeline_type=vectorsearch.PipelineType.TRIGGERED,
            columns_to_sync=["cell_id", "cell_type", "cell_subtype", "disease", "tissue", "sex", "donor_id", "n_genes_used", "embedding"],
        ),
    )
    print(f"Creating index {VS_INDEX}")
else:
    w.vector_search_indexes.sync_index(VS_INDEX)
    print(f"Triggered sync for {VS_INDEX}")

expected_rows = spark.table(EMB_TABLE).count()
deadline = time.time() + 3600
while True:
    idx = w.vector_search_indexes.get_index(VS_INDEX)
    ready = bool(idx.status and idx.status.ready)
    indexed = (idx.status.indexed_row_count or 0) if idx.status else 0
    if ready and indexed >= expected_rows:
        print(f"{VS_INDEX} ready with {indexed} rows")
        break
    if time.time() > deadline:
        raise TimeoutError(f"Index not ready: {idx.status}")
    time.sleep(30)

# COMMAND ----------

# DBTITLE 1,Nearest-neighbor cell-type search
from collections import Counter

RESULT_COLS = ["cell_id", "cell_type", "cell_subtype", "disease", "donor_id"]


def search_neighbors(query_vector, k=10, exclude_cell_id=None, filters_json=None):
    res = w.vector_search_indexes.query_index(
        index_name=VS_INDEX,
        columns=RESULT_COLS,
        query_vector=[float(x) for x in query_vector],
        num_results=k + (1 if exclude_cell_id else 0),
        filters_json=filters_json,
    )
    cols = [c.name for c in res.manifest.columns]   # RESULT_COLS + 'score'
    df = pd.DataFrame(res.result.data_array or [], columns=cols)
    if exclude_cell_id is not None:
        df = df[df["cell_id"] != exclude_cell_id]
    return df.head(k)


def predict_cell_type(gene_ids, counts, k=10, label_col="cell_subtype"):
    """Raw counts -> TEDDY endpoint embedding -> k nearest reference cells -> majority-vote label."""
    vec, n_used = embed_via_endpoint(gene_ids, counts)
    nbrs = search_neighbors(vec, k=k)
    votes = Counter(nbrs[label_col])
    label, n = votes.most_common(1)[0]
    return {"predicted": label, "vote_share": n / len(nbrs), "n_genes_used": n_used}, nbrs


# End-to-end demo: raw profile of a reference cell -> serving endpoint -> Vector Search
q = 5
qg, qc = cell_profile(q)
pred, nbrs = predict_cell_type(qg, qc, k=10)
print("query cell:", adata.obs_names[q], "| true subtype:", adata.obs["Cell.Types"].iloc[q], "| prediction:", pred)
display(nbrs)

# COMMAND ----------

# DBTITLE 1,Sanity check: leave-one-out kNN subtype accuracy
# Uses stored embeddings (no endpoint calls); compares kNN majority vote with the majority-class baseline
sample = spark.table(EMB_TABLE).select("cell_id", "cell_subtype", "embedding").orderBy("cell_id").limit(2000) \
    .toPandas().sample(n=min(100, spark.table(EMB_TABLE).count()), random_state=0)

hits = 0
for _, r in sample.iterrows():
    nb = search_neighbors(r["embedding"], k=10, exclude_cell_id=r["cell_id"])
    hits += int(Counter(nb["cell_subtype"]).most_common(1)[0][0] == r["cell_subtype"])

baseline = spark.table(EMB_TABLE).groupBy("cell_subtype").count().agg({"count": "max"}).first()[0] / spark.table(EMB_TABLE).count()
print(f"kNN(k=10) subtype accuracy on {len(sample)} cells: {hits/len(sample):.1%}  | majority-class baseline: {baseline:.1%}")

# COMMAND ----------

# DBTITLE 1,Notes
# MAGIC %md
# MAGIC **Notes**
# MAGIC * Embedding = attention-masked mean of TEDDY-G final hidden states, L2-normalized. The official tutorial mean-pools over the padded 2,048 positions; masking excludes padding so short cells are not diluted.
# MAGIC * Endpoint input: one row per cell, `gene_ids` (Ensembl IDs, e.g. `ENSG00000141510`) and raw integer `counts` in the same order. Zero-count genes can be omitted. Non-Ensembl symbols must be mapped first.
# MAGIC * The reference set is the repo's sample data: 3,538 prefrontal-cortex inhibitory interneurons (one CELLxGENE `cell_type`), so neighbor search is most informative on the author `cell_subtype` labels (7 subtypes). Add more reference cells (any CELLxGENE h5ad) by embedding them into the same table and re-running the index sync cell.
# MAGIC * CPU inference is about seconds per cell at 2,048 tokens; for large reference sets run the embedding cell on GPU compute or set `MAX_CELLS`.
# MAGIC * Monitor and manage the endpoint under **Serving** in the left navigation.