# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # TEDDY-G 70M on Model Serving
# MAGIC
# MAGIC Downloads Merck's TEDDY-G 70M single-cell foundation model (Apache-2.0, arXiv 2503.03485) from Hugging Face into a UC volume, packages it as an MLflow pyfunc cell-embedding model (raw counts in, 512-d embeddings out), validates it against the official TEDDY pipeline, registers it in Unity Catalog and deploys it to a Model Serving endpoint.

# COMMAND ----------

# DBTITLE 1,Install validation-only tools
# Notebook-only tools: uv (isolated serving-env check) and anndata/scanpy (to run the official
# TEDDY preprocessing/tokenization as a reference). None of these go into the served model.
%pip install -q uv anndata scanpy
%restart_python

# COMMAND ----------

# DBTITLE 1,Configuration
CATALOG = "<catalog>"
SCHEMA = "skills_owm_202610"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.owm_teddy_model_001_baseline_r2"
ENDPOINT_NAME = "owm_teddy_ep_001_baseline_r2"
FILES_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/owm_teddy_files/teddy-eval-2026-10-27/001_baseline"

HF_REPO = "Merck/TEDDY"
HF_REVISION = "8c880aa67674042ccdc254bccb5de081092a48bf"  # pinned commit for reproducibility
MODEL_SUBDIR = "teddy/models/teddy_g/70M"

# COMMAND ----------

# DBTITLE 1,Download TEDDY files to the UC volume
import os
from huggingface_hub import HfApi, snapshot_download

# Skip the 160M/400M checkpoints (~2.2 GB) and repo clutter; keep 70M weights, model/tokenizer code,
# gene medians (needed for normalization), and the sample data used for validation.
IGNORE = ["teddy/models/teddy_g/160M/*", "teddy/models/teddy_g/400M/*", "*.DS_Store", "poetry.lock"]

local_repo = snapshot_download(HF_REPO, revision=HF_REVISION, local_dir=FILES_DIR, ignore_patterns=IGNORE)

# Verify every downloaded file against Hugging Face's size metadata
info = HfApi().model_info(HF_REPO, revision=HF_REVISION, files_metadata=True)
expected = {s.rfilename: s.size for s in info.siblings}
import fnmatch
wanted = {f: sz for f, sz in expected.items() if not any(fnmatch.fnmatch(f, p) for p in IGNORE)}
mismatch = {f: (sz, os.path.getsize(os.path.join(FILES_DIR, f)) if os.path.exists(os.path.join(FILES_DIR, f)) else None)
            for f, sz in wanted.items() if not os.path.exists(os.path.join(FILES_DIR, f)) or os.path.getsize(os.path.join(FILES_DIR, f)) != sz}
assert not mismatch, mismatch
print(f"{len(wanted)} files verified in {FILES_DIR}")
MODEL_DIR = os.path.join(FILES_DIR, MODEL_SUBDIR)
MEDIANS_PATH = os.path.join(FILES_DIR, "teddy/data_processing/utils/medians/data/teddy_gene_medians.json")
for f in sorted(os.listdir(MODEL_DIR)):
    print(f"{f:28s} {os.path.getsize(os.path.join(MODEL_DIR, f))/1e6:9.1f} MB")

# COMMAND ----------

# DBTITLE 1,Serving wrapper: TEDDY-G pyfunc embedder
import mlflow
import numpy as np
import pandas as pd


class TeddyGEmbedder(mlflow.pyfunc.PythonModel):
    """Raw scRNA-seq counts per cell -> TEDDY-G 70M cell embeddings.

    Input (one row per cell):
      ensembl_ids: list[str]   human Ensembl gene IDs (ENSG...)
      counts:      list[float] raw UMI counts aligned with ensembl_ids (non-zero genes are enough)
    Output:
      cell_embedding:      list[float] (512) first-token embedding (TeddyGModel's default `cell_emb`)
      cell_embedding_mean: list[float] (512) mean over non-padded tokens
      n_genes_tokenized:   int  number of gene tokens fed to the model (<= 2048)

    Server-side preprocessing mirrors the official TEDDY pipeline (tutorial config):
    total-count normalize to 1e4 -> divide by TEDDY gene medians -> keep vocab genes with value > 0
    -> rank descending -> truncate to 2048 tokens (no CLS token, no bio-annotation tokens).
    """

    def load_context(self, context):
        import json
        import torch
        from torch import nn
        from safetensors.torch import load_file

        a = context.artifacts
        with open(a["config"]) as f:
            cfg = json.load(f)
        with open(a["vocab"]) as f:
            vocab = [line.strip() for line in f if line.strip()]
        self.token_of = {g: i for i, g in enumerate(vocab)}  # vocab.txt line index == token id
        with open(a["medians"]) as f:
            self.medians = json.load(f)
        with open(a["added_tokens"]) as f:
            self.pad_id = json.load(f)["<pad>"]
        self.max_len = cfg["max_position_embeddings"]
        self.batch_size = 8

        # Pure-PyTorch re-implementation of TeddyGModel's inference path. Parameter names match
        # the checkpoint exactly, so load_state_dict(strict=True) proves the architecture matches.
        class TeddyGEncoder(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.embeddings = nn.Embedding(c["ntoken"], c["d_model"])
                self.position_embeddings = nn.Embedding(c["max_position_embeddings"], c["d_model"])
                layer = nn.TransformerEncoderLayer(
                    c["d_model"], c["nheads"], c["d_hid"], c["dropout"], batch_first=True,
                    norm_first=c.get("pre_norm", False), activation=c["layer_activation"],
                )
                self.encoder = nn.TransformerEncoder(layer, c["nlayers"], enable_nested_tensor=False)
                # MLM head is unused for embeddings; kept only so strict loading succeeds.
                self.decoder_head = nn.Linear(c["d_model"], c["ntoken"], bias=False)
                self.decoder_bias = nn.Parameter(torch.zeros(c["ntoken"]))

            def forward(self, gene_ids, pad_mask):
                x = self.embeddings(gene_ids) + self.position_embeddings(
                    torch.arange(gene_ids.shape[1], device=gene_ids.device))
                for layer in self.encoder.layers:
                    x = layer(src=x, src_key_padding_mask=pad_mask)
                return x

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TeddyGEncoder(cfg)
        model.load_state_dict(load_file(a["weights"]), strict=True)
        self.model = model.to(self.device).eval()

    def _tokenize(self, ensembl_ids, counts):
        genes = np.asarray(ensembl_ids, dtype=object)
        x = np.asarray(counts, dtype=np.float64)
        if genes.shape != x.shape:
            raise ValueError("ensembl_ids and counts must have the same length")
        total = x.sum()
        if total <= 0:
            raise ValueError("cell has no counts")
        x = x / total * 1e4
        x = x / np.array([self.medians.get(g, 1.0) for g in genes])
        tok = np.array([self.token_of.get(g, -1) for g in genes])
        keep = (tok >= 0) & (x > 0)
        tok, x = tok[keep], x[keep]
        order = np.argsort(-x, kind="stable")[: self.max_len]
        return tok[order]

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import torch

        token_seqs = [self._tokenize(g, c) for g, c in zip(model_input["ensembl_ids"], model_input["counts"])]
        first, mean = [], []
        for i in range(0, len(token_seqs), self.batch_size):
            chunk = token_seqs[i : i + self.batch_size]
            L = max(max(len(t) for t in chunk), 1)
            ids = torch.full((len(chunk), L), self.pad_id, dtype=torch.long)
            for j, t in enumerate(chunk):
                ids[j, : len(t)] = torch.as_tensor(t, dtype=torch.long)
            ids = ids.to(self.device)
            pad_mask = ids == self.pad_id
            with torch.no_grad():
                h = self.model(ids, pad_mask)
            keep = (~pad_mask).unsqueeze(-1).float()
            first.append(h[:, 0, :].float().cpu().numpy())
            mean.append(((h * keep).sum(1) / keep.sum(1).clamp(min=1)).float().cpu().numpy())
        first, mean = np.concatenate(first), np.concatenate(mean)
        return pd.DataFrame({
            "cell_embedding": [r.astype(np.float32).tolist() for r in first],
            "cell_embedding_mean": [r.astype(np.float32).tolist() for r in mean],
            "n_genes_tokenized": np.array([len(t) for t in token_seqs], dtype=np.int64),
        })


ARTIFACTS = {
    "weights": f"{MODEL_DIR}/model.safetensors",
    "config": f"{MODEL_DIR}/config.json",
    "vocab": f"{MODEL_DIR}/vocab.txt",
    "added_tokens": f"{MODEL_DIR}/added_tokens.json",
    "medians": MEDIANS_PATH,
}

# COMMAND ----------

# DBTITLE 1,Reference: official TEDDY preprocess + tokenize on sample data
# Run the official TEDDY code (tutorial config) on the bundled sample data. Outputs go to /tmp,
# not the volume. This is the ground truth the serving wrapper is checked against.
import os, sys, glob
sys.path.insert(0, FILES_DIR)
_prev_cwd = os.getcwd()
os.chdir(FILES_DIR)  # official code resolves some helper files relative to repo root
from teddy.data_processing.preprocessing.preprocess import preprocess
from teddy.data_processing.tokenization.tokenization import tokenize

REF_DIR = "/tmp/teddy_ref"
maps = f"{FILES_DIR}/teddy/data_processing/utils/bio_annotations/data/mappings"
preprocess(
    data_path=f"{FILES_DIR}/data/sample_data.h5ad",
    metadata_path=f"{FILES_DIR}/data/sample_data_metadata.json",
    hyperparameters={
        "min_gene_counts": None, "remove_assays": [], "max_mitochondrial_prop": None, "remove_cell_types": [],
        "hvg_method": None, "normalized_total": 10000, "median_dict": MEDIANS_PATH, "log1p": False,
        "compute_medians": False, "median_column": "index", "reference_id_only": False,
        "load_dir": f"{FILES_DIR}/data", "save_dir": f"{REF_DIR}/processed",
    },
)
tokenize(
    data_path=f"{REF_DIR}/processed/sample_data.h5ad",
    metadata_path=f"{REF_DIR}/processed/sample_data_metadata.json",
    tokenization_args={
        "tokenizer_name_or_path": MODEL_DIR, "gene_id_column": "index", "bio_annotations": True,
        "disease_mapping": f"{maps}/all_filtered_disease_mapping.json",
        "tissue_mapping": f"{maps}/all_filtered_tissue_mapping.json",
        "cell_mapping": f"{maps}/all_filtered_cell_mapping.json",
        "sex_mapping": f"{maps}/all_filtered_sex_mapping.json",
        "max_shard_samples": 500, "max_seq_len": 2048, "pad_length": 2048, "add_cls": False, "bins": 0,
        "continuous_rank": True, "truncation_method": "max", "add_disease_annotation": False,
        "include_zero_genes": False, "load_dir": f"{REF_DIR}/processed", "save_dir": f"{REF_DIR}/tokenized",
    },
)
os.chdir(_prev_cwd)  # leave /Volumes: uv (used later) fails when the cwd is under /Volumes
from datasets import load_dataset
ref_ds = load_dataset("arrow", data_files={"train": sorted(glob.glob(f"{REF_DIR}/tokenized/sample_data/*.arrow"))})["train"]
print(ref_ds)

# COMMAND ----------

# DBTITLE 1,Build request payloads from raw sample counts
import anndata as ad
import scipy.sparse as sp

raw = ad.read_h5ad(f"{FILES_DIR}/data/sample_data.h5ad")
# Same raw-count source the official preprocess uses: .raw, else layers['counts'], else X
X = raw.raw.X if raw.raw is not None else raw.layers.get("counts", raw.X)
var_names = (raw.raw.var_names if raw.raw is not None else raw.var_names).to_numpy()
X = sp.csr_matrix(X)
print(raw.shape, "cells in raw:", X.shape[0], "| cells in official tokenized output:", len(ref_ds))

def cell_payload(i):
    row = X.getrow(i)  # send only non-zero genes; zeros affect neither normalization nor ranking
    return var_names[row.indices].tolist(), row.data.astype(float).tolist()

requests_df = pd.DataFrame([cell_payload(i) for i in range(X.shape[0])], columns=["ensembl_ids", "counts"])
input_example = requests_df.head(2)
print(requests_df.assign(n=requests_df.ensembl_ids.str.len()).n.describe())

# COMMAND ----------

# DBTITLE 1,Log the pyfunc model to MLflow
import torch, safetensors
from mlflow.models import infer_signature

# Local, in-process instance only to produce the output schema for the signature
_probe = TeddyGEmbedder()
_probe.load_context(type("Ctx", (), {"artifacts": ARTIFACTS})())
signature = infer_signature(input_example, _probe.predict(None, input_example))
print(signature)

pip_requirements = [
    f"mlflow=={mlflow.__version__}",
    f"torch=={torch.__version__.split('+')[0]}",
    f"safetensors=={safetensors.__version__}",
    f"numpy=={np.__version__}",
    f"pandas=={pd.__version__}",
]

with mlflow.start_run(run_name="teddy_g_70m_baseline_r2") as run:
    mlflow.log_params({"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "variant": "TEDDY-G 70M",
                       "max_seq_len": 2048, "license": "apache-2.0", "paper": "arXiv:2503.03485"})
    model_info = mlflow.pyfunc.log_model(
        name="teddy_g_70m",
        python_model=TeddyGEmbedder(),
        artifacts=ARTIFACTS,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
    )
print(model_info.model_uri)
print(pip_requirements)

# COMMAND ----------

# DBTITLE 1,Validate logged model against official TEDDY code
import time
from teddy.models.teddy_g.model import TeddyGConfig, TeddyGModel
from safetensors.torch import load_file

loaded = mlflow.pyfunc.load_model(model_info.model_uri)
impl = loaded.unwrap_python_model()
PAD = impl.pad_id

def ref_tokens(i):  # official arrow output stores each token as a 1-element list
    return [t[0] if isinstance(t, list) else t for t in ref_ds[i]["gene_ids"] if (t[0] if isinstance(t, list) else t) != PAD]

# 1) Tokenization: every sample cell vs official tokenize() output.
# Official code ranks with torch.topk, whose order among tied values is unspecified, so compare the
# sequence of normalized expression values (identical => any difference is only tie ordering).
def norm_vals(i):
    g = np.asarray(requests_df.ensembl_ids[i], dtype=object); x = np.asarray(requests_df.counts[i], float)
    x = x / x.sum() * 1e4 / np.array([impl.medians.get(k, 1.0) for k in g])
    return {impl.token_of[k]: v for k, v in zip(g, x) if k in impl.token_of and v > 0}

exact = ties_only = first_same = 0
for i in range(len(requests_df)):
    ours, ref = list(impl._tokenize(requests_df.ensembl_ids[i], requests_df.counts[i])), ref_tokens(i)
    v = norm_vals(i)
    exact += ours == ref
    ties_only += np.allclose([v[t] for t in ours], [v[t] for t in ref], rtol=1e-5)
    first_same += ours[0] == ref[0]
n = len(requests_df)
print(f"cells: {n} | identical token order: {exact} | identical value sequence (diff only in ties): {ties_only} "
      f"| same first token: {first_same}")
assert ties_only == n and first_same == n

# 2) Embeddings: logged model vs official TeddyGModel class on the official tokens (tutorial-style 2048 padding)
TeddyGModel.all_tied_weights_keys = {}  # notebook-only shim: official class predates transformers 5
ref_model = TeddyGModel(TeddyGConfig.from_pretrained(MODEL_DIR))
ref_model.load_state_dict(load_file(f"{MODEL_DIR}/model.safetensors"), strict=True)
ref_model = ref_model.to(impl.device).eval()
ref_model.return_cell_embs_all_tokens_mean = False

N = 32
ids = torch.full((N, 2048), PAD, dtype=torch.long)  # tutorial collate: pad to 2048
for i in range(N):
    t = ref_tokens(i); ids[i, : len(t)] = torch.tensor(t)
ids = ids.to(impl.device)
with torch.no_grad():
    ref_emb = ref_model(gene_ids=ids, attention_mask=(ids != PAD).long())["cell_emb"].cpu().numpy()
ours_df = loaded.predict(requests_df.head(N))
ours_emb = np.stack(ours_df.cell_embedding.to_numpy())
cos = (ours_emb * ref_emb).sum(1) / (np.linalg.norm(ours_emb, axis=1) * np.linalg.norm(ref_emb, axis=1))
print(f"cell_embedding vs official cell_emb over {N} cells: min cosine={cos.min():.6f}, "
      f"max abs diff={np.abs(ours_emb - ref_emb).max():.2e}  (end-to-end; includes tie-order differences)")

# Same official tokens through our encoder with per-batch (not 2048) padding -> isolates the architecture
L = int((ids != PAD).sum(1).max())
with torch.no_grad():
    same_tok = impl.model(ids[:, :L], ids[:, :L] == PAD)[:, 0, :].cpu().numpy()
print(f"encoder-only on identical tokens: max abs diff={np.abs(same_tok - ref_emb).max():.2e}")
display(ours_df.head(5).assign(dim=ours_df.cell_embedding.str.len(),
                               first4=ours_df.cell_embedding.map(lambda v: np.round(v[:4], 3).tolist()))
        [["n_genes_tokenized", "dim", "first4"]])

# 3) CPU latency, to inform the CPU vs GPU workload choice
impl_cpu = TeddyGEmbedder(); impl_cpu.load_context(type("Ctx", (), {"artifacts": ARTIFACTS})())
impl_cpu.device = torch.device("cpu"); impl_cpu.model = impl_cpu.model.cpu()
t = time.time(); impl_cpu.predict(None, requests_df.head(8)); cpu_s = (time.time() - t) / 8
print(f"CPU latency: {cpu_s:.2f} s/cell (torch threads={torch.get_num_threads()})")

# COMMAND ----------

# DBTITLE 1,Isolated serving-environment check (uv)
# Rebuilds the model's pip_requirements from scratch in a fresh venv, then loads and predicts,
# which is what the serving container does. Only this proves the artifact is installable/servable.
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

# DBTITLE 1,Register validated model in Unity Catalog
mlflow.set_registry_uri("databricks-uc")
registered_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=REGISTERED_MODEL_NAME,
    await_registration_for=300,
)
model_version = registered_version.version
print(f"Registered models:/{REGISTERED_MODEL_NAME}/{model_version}")

# COMMAND ----------

# DBTITLE 1,Endpoint configuration
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType

w = WorkspaceClient()

# CPU: the artifact loads on CPU (safetensors, device auto-selected), uses ~1.1 GB RSS, fits CPU's
# 4 GB/concurrency. Measured ~1.7 s/cell on CPU; switch to GPU_SMALL (T4) if latency matters --
# the same model version runs on GPU without re-logging.
WORKLOAD_TYPE = "CPU"
WORKLOAD_SIZE = "Small"
SCALE_TO_ZERO = True

supported = {c["workload_type"]: [s["key"] for s in c.get("workload_sizes", [])]
             for c in w.api_client.do("GET", "/api/2.0/serving-endpoints:workload-configs").get("workload_configs", [])}
assert WORKLOAD_SIZE in supported.get(WORKLOAD_TYPE, []), supported

served_entity = ServedEntityInput(
    entity_name=REGISTERED_MODEL_NAME,
    entity_version=str(model_version),
    workload_type=ServingModelWorkloadType(WORKLOAD_TYPE),
    workload_size=WORKLOAD_SIZE,
    scale_to_zero_enabled=SCALE_TO_ZERO,
)
endpoint_config = EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity])

try:
    existing_endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
except NotFound:
    existing_endpoint = None
if existing_endpoint is not None:
    names = [e.entity_name for e in (existing_endpoint.config.served_entities or [])] if existing_endpoint.config else []
    if names != [REGISTERED_MODEL_NAME]:
        raise ValueError(f"Refusing to replace endpoint {ENDPOINT_NAME!r}, which serves {names}.")
print("existing endpoint:", existing_endpoint is not None, "| plan:", served_entity)

# COMMAND ----------

# DBTITLE 1,Create or update the serving endpoint (~15 min)
if existing_endpoint is None:
    w.serving_endpoints.create(name=ENDPOINT_NAME, config=endpoint_config)
else:
    w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=[served_entity])
print(f"Deployment of {ENDPOINT_NAME} started")

# COMMAND ----------

# DBTITLE 1,Wait for endpoint readiness
import time
from databricks.sdk.service.serving import EndpointStateConfigUpdate, EndpointStateReady


def wait_for_endpoint_ready(name, timeout_s=1800, poll_s=30):
    deadline = time.time() + timeout_s
    failure_states = {EndpointStateConfigUpdate.UPDATE_FAILED, EndpointStateConfigUpdate.UPDATE_CANCELED}
    while time.time() < deadline:
        state = w.serving_endpoints.get(name).state
        if state.ready == EndpointStateReady.READY and state.config_update == EndpointStateConfigUpdate.NOT_UPDATING:
            return
        if state.config_update in failure_states:
            raise RuntimeError(f"{name} deployment failed with config update state {state.config_update.value}")
        time.sleep(poll_s)
    raise TimeoutError(f"{name} not ready after {timeout_s}s")


wait_for_endpoint_ready(ENDPOINT_NAME)
print(f"{ENDPOINT_NAME} is READY")

# COMMAND ----------

# DBTITLE 1,Smoke-test the endpoint
# Send 2 sample cells and compare with the official-reference embeddings computed above
resp = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_records=requests_df.head(2).to_dict(orient="records"),
)
served = np.array([p["cell_embedding"] for p in resp.predictions])
cos = (served * ref_emb[:2]).sum(1) / (np.linalg.norm(served, axis=1) * np.linalg.norm(ref_emb[:2], axis=1))
print("embedding dim:", served.shape[1], "| n_genes_tokenized:", [p["n_genes_tokenized"] for p in resp.predictions],
      "| cosine vs official reference:", np.round(cos, 6))