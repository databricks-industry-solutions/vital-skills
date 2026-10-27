# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # TEDDY-70M Model Serving + Cell-Type Nearest-Neighbor Search
# MAGIC
# MAGIC Downloads TEDDY-G 70M ([Merck/TEDDY](https://huggingface.co/Merck/TEDDY), Apache-2.0, arXiv:2503.03485) to a UC Volume, wraps it as an MLflow PyFunc, registers it in Unity Catalog, deploys a GPU Model Serving endpoint, embeds a CELLxGENE Census reference atlas into a Delta table, and builds a Vector Search index for nearest-neighbor cell-type lookup. Costly steps (endpoint, Vector Search) are gated by the `run_go` widget.

# COMMAND ----------

# DBTITLE 1,Install pinned dependencies
# skill-proof-teddy: 56b8c5a1
# TEDDY's pyproject pins transformers==4.41.0 (5.x breaks TeddyGModel: missing all_tied_weights_keys).
# The AI v6 base env ships transformers 5.x, so pin a notebook-scoped override. torch comes from the AI base env.
# uv is required for the isolated serving-environment check; cellxgene-census supplies the reference atlas.
%pip install -q "transformers==4.41.0" "huggingface_hub>=0.23,<1.0" uv cellxgene-census

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration and run gate
# skill-proof-teddy: 56b8c5a1
import os, sys, json, shutil, hashlib, subprocess, datetime
from databricks.sdk import WorkspaceClient

# Run gate for costly side effects (serving endpoint, Vector Search endpoint/index)
dbutils.widgets.dropdown("run_go", "false", ["false", "true"], "Run costly steps")
dbutils.widgets.dropdown("enable_inference_table", "true", ["true", "false"], "Inference table")
RUN_GO = dbutils.widgets.get("run_go") == "true"
ENABLE_INFERENCE_TABLE = dbutils.widgets.get("enable_inference_table") == "true"

CATALOG = "<catalog>"
SCHEMA = "skills_owm_202610"
VARIANT = "70M"

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.owm_teddy_model_002_with_skill"
ENDPOINT_NAME = "owm_teddy_ep_002_with_skill"
VS_ENDPOINT_NAME = "owm_teddy_vs_002_with_skill"
CELLS_TABLE = f"{CATALOG}.{SCHEMA}.owm_teddy_cells_002_with_skill"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.owm_teddy_index_002_with_skill"

HF_REPO = "Merck/TEDDY"
BASE_DIR = "/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/002_with_skill"
SNAPSHOT_DIR = f"{BASE_DIR}/snapshot"
MODEL_DIR = f"{SNAPSHOT_DIR}/teddy/models/teddy_g/{VARIANT}"
MANIFEST_PATH = f"{BASE_DIR}/manifest.json"
REGISTRATION_PATH = f"{BASE_DIR}/registration.json"

# Serverless: /local_disk0 is unavailable, use /tmp for scratch
TMP_DIR = "/tmp/teddy_002"
CODE_DIR = f"{TMP_DIR}/teddy_code"          # teddy/ source without weights (shipped via code_paths + artifact)
WRAPPER_PATH = f"{TMP_DIR}/teddy_wrapper.py" # file-based PyFunc (avoids cloudpickle ModuleNotFoundError)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

# Serving request settings
DEFAULT_PARAMS = {"max_seq_len": "2048", "pooling": "mean", "normalize": "true"}  # strings: SDK extra_params are str
PIP_REQUIREMENTS = [
    "torch>=2.3.0",
    "transformers==4.41.0",
    "numpy>=1.26.4,<2.0",
    "pandas>=2.2.2,<3.0",
]

w = WorkspaceClient()
print(f"RUN_GO={RUN_GO}  inference_table={ENABLE_INFERENCE_TABLE}")
print(f"model={UC_MODEL_NAME}\nendpoint={ENDPOINT_NAME}\nvs_endpoint={VS_ENDPOINT_NAME}\ntable={CELLS_TABLE}\nindex={VS_INDEX_NAME}\nfiles={BASE_DIR}")

# COMMAND ----------

# DBTITLE 1,Download TEDDY snapshot to UC Volume
# Idempotent: pinned-SHA snapshot -> /tmp -> UC Volume, sentinel + provenance manifest.
# TEDDY uses HF Xet storage; run the download in a clean subprocess with Xet/hf_transfer disabled.
SENTINEL = f"{BASE_DIR}/.snapshot_complete"

def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

if os.path.exists(SENTINEL):
    manifest = json.load(open(MANIFEST_PATH))
    print("Snapshot already present (sentinel found). Revision:", manifest["code"]["revision"])
else:
    local_snapshot = f"{TMP_DIR}/hf_snapshot"
    script = r'''
import os, sys, json
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import HfApi, snapshot_download
repo, local_dir = sys.argv[1], sys.argv[2]
sha = HfApi().model_info(repo).sha
snapshot_download(
    repo_id=repo, revision=sha, local_dir=local_dir,
    ignore_patterns=["teddy/models/teddy_g/160M/*", "teddy/models/teddy_g/400M/*", "*.DS_Store"],
)
print(json.dumps({"sha": sha}))
'''
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_XET")}
    env.update({"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    res = subprocess.run([sys.executable, "-c", script, HF_REPO, local_snapshot],
                         env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"HF download failed:\n{res.stderr[-3000:]}")
    hf_sha = json.loads(res.stdout.strip().splitlines()[-1])["sha"]

    shutil.rmtree(f"{local_snapshot}/.cache", ignore_errors=True)
    shutil.copytree(local_snapshot, SNAPSHOT_DIR, dirs_exist_ok=True)

    ckpt = f"{local_snapshot}/teddy/models/teddy_g/{VARIANT}"
    manifest = {
        "model": {"name": f"TEDDY-G {VARIANT}", "upstream_revision": hf_sha,
                  "paper": "arXiv:2503.03485", "reviewed_at": str(datetime.date.today())},
        "code": {"source_url": f"https://huggingface.co/{HF_REPO}", "revision": hf_sha},
        "weights": {"source_url": f"https://huggingface.co/{HF_REPO}", "revision_or_record": hf_sha,
                    "path": f"teddy/models/teddy_g/{VARIANT}/model.safetensors",
                    "sha256": sha256(f"{ckpt}/model.safetensors")},
        "license": {"code": "Apache-2.0", "weights": "Apache-2.0", "databases": "not-applicable"},
        "runtime": {"python": sys.version.split()[0], "accelerator": "gpu (cpu fallback)",
                    "network_required_at_runtime": False, "pip_requirements": PIP_REQUIREMENTS},
        "artifacts": [{"name": n, "location": f"{MODEL_DIR}/{n}", "sha256": sha256(f"{ckpt}/{n}")}
                      for n in ["config.json", "vocab.txt", "tokenizer_config.json"]],
        "acquired_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    open(SENTINEL, "w").write(hf_sha)
    print("Downloaded revision", hf_sha)

HF_REVISION = manifest["code"]["revision"]

# Sanity checks on the checkpoint
cfg = json.load(open(f"{MODEL_DIR}/config.json"))
vocab_tokens = [t.rstrip("\n") for t in open(f"{MODEL_DIR}/vocab.txt")]
ensg = [t for t in vocab_tokens if t.startswith("ENSG")]
assert "ENSG00000000003" in set(vocab_tokens), "vocab missing ENSG00000000003"
print(f"d_model={cfg['d_model']} layers={cfg['nlayers']} max_pos={cfg['max_position_embeddings']} "
      f"vocab={len(vocab_tokens)} ensembl={len(ensg)}")
print("weights sha256:", manifest["weights"]["sha256"][:16], "...")
display(dbutils.fs.ls(MODEL_DIR))

# COMMAND ----------

# DBTITLE 1,Stage code bundle and write PyFunc wrapper
# Stage the teddy/ source without weights; the wrapper adds it to sys.path in load_context.
shutil.rmtree(CODE_DIR, ignore_errors=True)
shutil.copytree(
    f"{SNAPSHOT_DIR}/teddy", f"{CODE_DIR}/teddy",
    ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.ckpt", "*.pt", "__pycache__", ".DS_Store"),
)
GENE_MEDIANS_PATH = f"{CODE_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json"

# Local copy of the checkpoint for fast artifact logging (Volume copy remains the governed source)
LOCAL_MODEL_DIR = f"{TMP_DIR}/checkpoint_{VARIANT}"
shutil.rmtree(LOCAL_MODEL_DIR, ignore_errors=True)
shutil.copytree(MODEL_DIR, LOCAL_MODEL_DIR)

WRAPPER_SRC = r'''
"""TEDDY-G PyFunc wrapper. Logged file-based (python_model=<path>) because the teddy package
is shipped as an artifact, not pip-installed.

Request row = one batch of cells:
  adata_sparsematrix: list[list[float]]  dense counts, shape (cells, genes)
  adata_obs:          JSON (orient=split) cell metadata; optional `cell_id` column is echoed back
  adata_var:          JSON (orient=split) gene metadata with an `index` column of Ensembl IDs
Params (strings): max_seq_len (<=2048), pooling ("mean" | "first"), normalize ("true" | "false")
Preprocessing mirrors upstream: normalize_total(1e4) -> divide by TEDDY gene medians ->
keep expressed in-vocab genes -> rank by expression -> top-K gene tokens.
"""
import sys
import os
import io
import json
import inspect

import numpy as np
import pandas as pd
import torch
import mlflow
from mlflow.pyfunc import PythonModel


class TEDDYEmbedder(PythonModel):
    MAX_CELLS_PER_REQUEST = 64
    FORWARD_BATCH = 8

    def load_context(self, context):
        # Serving container replaces stdout with a logger lacking isatty()
        for stream in (sys.stdout, sys.stderr):
            if stream is not None and not hasattr(stream, "isatty"):
                stream.isatty = lambda: False
        # Purge stale teddy modules (safe for re-runs / in-kernel dry-load)
        for _m in list(sys.modules):
            if _m == "teddy" or _m.startswith("teddy."):
                del sys.modules[_m]
        pkg_parent = context.artifacts["teddy_pkg_parent"]
        while pkg_parent in sys.path:
            sys.path.remove(pkg_parent)
        sys.path.insert(0, pkg_parent)

        from teddy.models.model_directory import get_architecture, model_dict
        from teddy.tokenizer.gene_tokenizer import GeneTokenizer

        model_dir = context.artifacts["model_dir"]
        arch = get_architecture(model_dir)
        self.config = model_dict[arch]["config_cls"].from_pretrained(model_dir)
        self.model = model_dict[arch]["model_cls"].from_pretrained(model_dir, config=self.config)
        self.model.return_all_embs = True              # token-level states for masked mean pooling
        self.model.return_cell_embs_first_token = True # pre-pooled first-token embedding ("cell_emb")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

        self.tokenizer = GeneTokenizer.from_pretrained(model_dir)  # reads vocab.txt from checkpoint dir
        self.vocab = dict(self.tokenizer.vocab)
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.unk_id = int(self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token))
        self.d_model = int(self.config.d_model)
        self.max_positions = int(self.config.max_position_embeddings)
        self.forward_params = set(inspect.signature(self.model.forward).parameters)
        with open(context.artifacts["gene_medians"]) as f:
            self.medians = json.load(f)
        print(f"TEDDY-G loaded on {self.device}, d_model={self.d_model}, vocab={len(self.vocab)}")

    @staticmethod
    def _matrix(value):
        rows = [np.asarray(r, dtype=np.float32).ravel() for r in value]
        if not rows:
            raise ValueError("adata_sparsematrix is empty")
        return np.vstack(rows)

    @staticmethod
    def _read_split_json(s):
        if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
            return pd.DataFrame()
        return pd.read_json(io.StringIO(str(s)), orient="split", dtype=False, convert_dates=False)

    def _tokenize(self, X, genes, max_seq_len, normalize):
        token_ids = np.array([self.vocab.get(g, -1) for g in genes], dtype=np.int64)
        in_vocab = token_ids >= 0
        X = X.astype(np.float64)
        if normalize:
            totals = X.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1.0
            X = X / totals * 1e4
            factors = np.array([1.0 / self.medians[g] if self.medians.get(g) else 1.0 for g in genes])
            X = X * factors[None, :]
        seqs, n_used = [], []
        for i in range(X.shape[0]):
            idx = np.where(in_vocab & (X[i] > 0))[0]
            order = idx[np.argsort(-X[i, idx], kind="stable")][:max_seq_len]
            ids = token_ids[order]
            n_used.append(int(ids.size))
            if ids.size == 0:  # all-OOV or all-zero cell -> single <unk> token
                ids = np.array([self.unk_id], dtype=np.int64)
            seqs.append(ids)
        return seqs, n_used

    @torch.no_grad()
    def _embed(self, seqs, pooling):
        out = []
        for s in range(0, len(seqs), self.FORWARD_BATCH):
            chunk = seqs[s:s + self.FORWARD_BATCH]
            L = max(len(x) for x in chunk)
            ids = torch.full((len(chunk), L), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, x in enumerate(chunk):
                ids[j, :len(x)] = torch.from_numpy(x)
                mask[j, :len(x)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            kwargs = {"gene_ids": ids, "attention_mask": mask}
            outputs = self.model(**{k: v for k, v in kwargs.items() if k in self.forward_params})
            emb = None
            if pooling == "mean":
                tok = outputs.get("all_embs")
                if tok is None:
                    tok = outputs.get("last_hidden_state")
                if tok is not None and tok.dim() == 3:
                    m = mask.unsqueeze(-1).to(tok.dtype)
                    emb = (tok * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
                elif tok is not None:
                    emb = tok
            if emb is None:
                emb = outputs.get("cell_emb")
            if emb is None:
                raise RuntimeError(f"No embedding in model outputs: {list(outputs.keys())}")
            out.append(emb.float().cpu().numpy())
        return np.vstack(out)

    def predict(self, context, model_input, params=None):
        params = params or {}
        max_seq_len = max(1, min(int(params.get("max_seq_len", 2048)), self.max_positions))
        pooling = str(params.get("pooling", "mean")).lower()
        if pooling not in ("mean", "first"):
            raise ValueError("pooling must be 'mean' or 'first'")
        normalize = str(params.get("normalize", "true")).lower() in ("true", "1", "yes")
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])

        results = []
        for _, row in model_input.iterrows():
            X = self._matrix(row["adata_sparsematrix"])
            var_df = self._read_split_json(row["adata_var"])
            obs_df = self._read_split_json(row.get("adata_obs"))
            genes = (var_df["index"] if "index" in var_df.columns else var_df.index).astype(str).tolist()
            if X.shape[1] != len(genes):
                raise ValueError(f"matrix has {X.shape[1]} genes but adata_var has {len(genes)}")
            if X.shape[0] > self.MAX_CELLS_PER_REQUEST:
                raise ValueError(f"max {self.MAX_CELLS_PER_REQUEST} cells per request row, got {X.shape[0]}")
            if "cell_id" in obs_df.columns and len(obs_df) == X.shape[0]:
                cell_ids = obs_df["cell_id"].astype(str).tolist()
            else:
                cell_ids = [f"cell_{i}" for i in range(X.shape[0])]
            seqs, n_used = self._tokenize(X, genes, max_seq_len, normalize)
            embs = self._embed(seqs, pooling)
            for cid, e, n in zip(cell_ids, embs, n_used):
                results.append({"cell_id": cid, "embedding": e.astype(np.float64).tolist(), "n_genes_used": n})
        return pd.DataFrame(results, columns=["cell_id", "embedding", "n_genes_used"])


mlflow.models.set_model(TEDDYEmbedder())
'''
with open(WRAPPER_PATH, "w") as f:
    f.write(WRAPPER_SRC)

ARTIFACTS = {"model_dir": LOCAL_MODEL_DIR, "teddy_pkg_parent": CODE_DIR, "gene_medians": GENE_MEDIANS_PATH}
bundle_mb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(CODE_DIR) for f in fs) / 1e6
print(f"code bundle {bundle_mb:.1f} MB; wrapper -> {WRAPPER_PATH}")

# COMMAND ----------

# DBTITLE 1,Dry-load test and input example
# Load the wrapper in-kernel with a stand-in context, inspect forward outputs, and build the input example.
import importlib.util, types
import numpy as np, pandas as pd, torch

for _k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.")]:
    del sys.modules[_k]
spec = importlib.util.spec_from_file_location("teddy_wrapper", WRAPPER_PATH)
teddy_wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teddy_wrapper)

embedder = teddy_wrapper.TEDDYEmbedder()
embedder.load_context(types.SimpleNamespace(artifacts=ARTIFACTS))
print("forward params:", sorted(embedder.forward_params))
print("pad/unk ids:", embedder.pad_id, embedder.unk_id, "| medians:", len(embedder.medians))

# Input example with REAL Ensembl IDs from vocab.txt (synthetic names would all map to <unk>)
real_genes = [t for t in vocab_tokens if t.startswith("ENSG")][:200]
rng = np.random.default_rng(42)
expr = rng.poisson(2.0, size=(5, len(real_genes))).astype(np.float32)
obs_df = pd.DataFrame({"cell_id": [f"example_cell_{i}" for i in range(5)]})
var_df = pd.DataFrame({"index": real_genes})
input_example = pd.DataFrame({
    "adata_sparsematrix": [expr.tolist()],
    "adata_obs": [obs_df.to_json(orient="split")],
    "adata_var": [var_df.to_json(orient="split")],
})

# Raw forward diagnostic (output keys/shapes)
with torch.no_grad():
    ids = torch.tensor([[embedder.vocab[g] for g in real_genes[:16]]], device=embedder.device)
    o = embedder.model(gene_ids=ids, attention_mask=torch.ones_like(ids))
print({k: tuple(v.shape) for k, v in o.items() if hasattr(v, "shape")})

example_output = embedder.predict(None, input_example, params=DEFAULT_PARAMS)
E = np.array(example_output["embedding"].tolist())
print("example output:", E.shape, "| norms:", np.round(np.linalg.norm(E, axis=1), 3),
      "| genes used:", example_output["n_genes_used"].tolist())
assert E.shape == (5, cfg["d_model"]) and np.isfinite(E).all() and (np.linalg.norm(E, axis=1) > 0).all()
assert np.std(E, axis=0).max() > 0, "degenerate embeddings: all cells identical"

# Negative / edge tests
oov = input_example.copy()
oov["adata_var"] = [pd.DataFrame({"index": [f"FAKE{i}" for i in range(len(real_genes))]}).to_json(orient="split")]
oov_out = embedder.predict(None, oov, params=DEFAULT_PARAMS)
print("all-OOV ->", np.array(oov_out["embedding"].tolist()).shape, "genes used", oov_out["n_genes_used"].tolist())
first_out = embedder.predict(None, input_example, params={**DEFAULT_PARAMS, "pooling": "first"})
print("pooling=first ->", np.array(first_out["embedding"].tolist()).shape)
display(example_output.assign(embedding=lambda d: d["embedding"].map(lambda v: np.round(v[:5], 4).tolist())))

# COMMAND ----------

# DBTITLE 1,Log PyFunc model (file-based)
import mlflow
from mlflow.models import infer_signature

mlflow.autolog(disable=True)
mlflow.set_registry_uri("databricks-uc")

signature = infer_signature(input_example, example_output, params=DEFAULT_PARAMS)
print(signature)

with mlflow.start_run(run_name=f"teddy_{VARIANT.lower()}_002_with_skill") as run:
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=WRAPPER_PATH,          # file path, NOT an instance (cloudpickle would break teddy imports)
        code_paths=[CODE_DIR],
        artifacts=ARTIFACTS,
        signature=signature,
        input_example=(input_example, DEFAULT_PARAMS),
        pip_requirements=PIP_REQUIREMENTS,
    )
    mlflow.set_tags({
        "model_family": "TEDDY-G", "variant": VARIANT, "license": "Apache-2.0",
        "paper": "arXiv:2503.03485", "hf_source": f"https://huggingface.co/{HF_REPO}",
        "hf_revision": HF_REVISION, "embedding_dim": str(cfg["d_model"]),
    })
    mlflow.log_dict(manifest, "provenance/manifest.json")

print("model_uri:", model_info.model_uri)

# COMMAND ----------

# DBTITLE 1,Isolated serving-environment check (uv)
# Installs pip_requirements from scratch in a fresh env and runs predict on the stored input example.
# This is the gate for registration: a notebook import does not prove the serving container can load the model.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
assert shutil.which("uv"), "uv not found on PATH"

loaded = mlflow.pyfunc.load_model(model_info.model_uri)
assert loaded.metadata.signature is not None, "no signature"
representative_input = loaded.input_example
assert representative_input is not None, "no input example"

mlflow.models.predict(
    model_uri=model_info.model_uri,
    input_data=input_example,   # params fall back to signature defaults (DEFAULT_PARAMS)
    env_manager="uv",
)
ISOLATED_CHECK_PASSED = True
print("Isolated uv check passed")

# COMMAND ----------

# DBTITLE 1,Register model version in Unity Catalog
assert globals().get("ISOLATED_CHECK_PASSED"), "Run the isolated check first; do not register an unvalidated artifact."
from mlflow import MlflowClient

registered_version = mlflow.register_model(
    model_uri=model_info.model_uri, name=UC_MODEL_NAME, await_registration_for=600,
)
MODEL_VERSION = registered_version.version

client = MlflowClient(registry_uri="databricks-uc")
client.update_registered_model(
    UC_MODEL_NAME,
    description=(f"TEDDY-G {VARIANT} single-cell foundation model (Merck/TEDDY @ {HF_REVISION}, Apache-2.0, "
                 "arXiv:2503.03485). Input: dense counts + Ensembl gene IDs; output: 512-d cell embeddings."),
)
for k, v in {"hf_revision": HF_REVISION, "license": "Apache-2.0", "variant": VARIANT,
             "weights_sha256": manifest["weights"]["sha256"], "isolated_check": "uv-passed"}.items():
    client.set_model_version_tag(UC_MODEL_NAME, MODEL_VERSION, k, v)

# Persist the exact version so later cells never pick "latest" implicitly
json.dump({"model": UC_MODEL_NAME, "version": str(MODEL_VERSION), "model_uri": model_info.model_uri,
           "hf_revision": HF_REVISION}, open(REGISTRATION_PATH, "w"), indent=2)
print(f"Registered {UC_MODEL_NAME} version {MODEL_VERSION} -> models:/{UC_MODEL_NAME}/{MODEL_VERSION}")

# COMMAND ----------

# DBTITLE 1,Create serving endpoint (gated)
from datetime import datetime
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType,
    AiGatewayConfig, AiGatewayInferenceTableConfig, AiGatewayUsageTrackingConfig,
)
from mlflow import MlflowClient

reg = json.load(open(REGISTRATION_PATH))
MODEL_VERSION = reg["version"]
MlflowClient(registry_uri="databricks-uc").get_model_version(UC_MODEL_NAME, MODEL_VERSION)  # verify exact version

# 70M (~285 MB weights) fits GPU_SMALL (T4 16 GB on AWS) with <=10-cell batches; scale-to-zero to avoid idle cost
WORKLOAD_TYPE = ServingModelWorkloadType.GPU_SMALL
WORKLOAD_SIZE = "Small"
SCALE_TO_ZERO = True
supported = {c["workload_type"]: [s["key"] for s in c.get("workload_sizes", [])]
             for c in w.api_client.do("GET", "/api/2.0/serving-endpoints:workload-configs")["workload_configs"]}
assert WORKLOAD_SIZE in supported.get(WORKLOAD_TYPE.value, []), f"{WORKLOAD_TYPE.value}/{WORKLOAD_SIZE} unsupported: {supported}"

SERVED_ENTITY_NAME = f"teddy-{VARIANT.lower()}-v{MODEL_VERSION}"
served_entity = ServedEntityInput(
    name=SERVED_ENTITY_NAME,
    entity_name=UC_MODEL_NAME,
    entity_version=str(MODEL_VERSION),
    workload_type=WORKLOAD_TYPE,
    workload_size=WORKLOAD_SIZE,
    scale_to_zero_enabled=SCALE_TO_ZERO,
)

# Inference table (HLS default). Payloads contain gene-expression profiles: review classification/retention
# before sharing the table. Timestamped prefix avoids conflicts with tables left by earlier failed deploys.
ai_gateway = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        catalog_name=CATALOG, schema_name=SCHEMA,
        table_name_prefix=f"{ENDPOINT_NAME}_{datetime.now():%Y%m%d%H%M}", enabled=True,
    ),
    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
) if ENABLE_INFERENCE_TABLE else None

print(f"Plan: endpoint={ENDPOINT_NAME} model={UC_MODEL_NAME} v{MODEL_VERSION} "
      f"workload={WORKLOAD_TYPE.value}/{WORKLOAD_SIZE} scale_to_zero={SCALE_TO_ZERO} inference_table={ENABLE_INFERENCE_TABLE}")

try:
    existing_endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
except NotFound:
    existing_endpoint = None

if not RUN_GO:
    print("run-gate off: set widget run_go=true to create/update the endpoint (~15-25 min for GPU).")
elif existing_endpoint is None:
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity]),
        ai_gateway=ai_gateway,
    )
    print("Endpoint creation started.")
else:
    current = [e.entity_name for e in (existing_endpoint.config.served_entities or [])] if existing_endpoint.config else []
    if current != [UC_MODEL_NAME]:
        raise ValueError(f"Refusing to replace endpoint {ENDPOINT_NAME!r} serving {current}; confirm replacement first.")
    w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=[served_entity])
    print("Endpoint config update started.")

# COMMAND ----------

# DBTITLE 1,Wait for endpoint readiness
import time
from databricks.sdk.service.serving import EndpointStateConfigUpdate, EndpointStateReady, ServedModelStateDeployment

def wait_for_endpoint_ready(name, timeout_s=1800, poll_s=30):
    deadline = time.time() + timeout_s
    failed = {EndpointStateConfigUpdate.UPDATE_FAILED, EndpointStateConfigUpdate.UPDATE_CANCELED}
    while time.time() < deadline:
        ep = w.serving_endpoints.get(name)
        st = ep.state
        if st.ready == EndpointStateReady.READY and st.config_update == EndpointStateConfigUpdate.NOT_UPDATING:
            return ep
        if st.config_update in failed:
            raise RuntimeError(f"{name} update failed: {st.config_update.value}. Check service logs: "
                               f"/api/2.0/serving-endpoints/{name}/served-entities/{SERVED_ENTITY_NAME}/logs")
        cfg_ = ep.pending_config or ep.config
        for se in (cfg_.served_entities or []) if cfg_ else []:
            if se.state and se.state.deployment == ServedModelStateDeployment.DEPLOYMENT_FAILED:
                raise RuntimeError(f"{se.name} deployment failed: {se.state.deployment_state_message}")
        print(f"{datetime.now():%H:%M:%S} ready={st.ready.value if st.ready else None} "
              f"config_update={st.config_update.value if st.config_update else None}")
        time.sleep(poll_s)
    raise TimeoutError(f"{name} not ready after {timeout_s}s")

ENDPOINT_READY = False
if RUN_GO:
    wait_for_endpoint_ready(ENDPOINT_NAME)
    ENDPOINT_READY = True
    print(f"{ENDPOINT_NAME} is READY")
else:
    print("run-gate off: skipping readiness wait.")

# COMMAND ----------

# DBTITLE 1,Endpoint smoke tests
# Payload built from the signature: all three columns required; params passed as string extra_params.
import numpy as np

if not ENDPOINT_READY:
    print("Endpoint not ready / run-gate off: skipping smoke tests.")
else:
    rec = {
        "adata_sparsematrix": expr.tolist(),
        "adata_obs": obs_df.to_json(orient="split"),
        "adata_var": var_df.to_json(orient="split"),
    }
    resp = w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=[rec], extra_params=DEFAULT_PARAMS)
    preds = resp.predictions  # typed response: attribute access
    E_ep = np.array([p["embedding"] for p in preds])
    E_loc = np.array(example_output["embedding"].tolist())
    cos = (E_ep * E_loc).sum(1) / (np.linalg.norm(E_ep, axis=1) * np.linalg.norm(E_loc, axis=1))
    print("endpoint embeddings:", E_ep.shape, "| cosine vs notebook:", np.round(cos, 5))
    assert E_ep.shape == (5, cfg["d_model"]) and cos.min() > 0.999

    # Negative test: missing required column must be rejected by schema enforcement
    try:
        w.serving_endpoints.query(name=ENDPOINT_NAME,
                                  dataframe_records=[{k: v for k, v in rec.items() if k != "adata_obs"}])
        raise AssertionError("missing adata_obs was accepted")
    except AssertionError:
        raise
    except Exception as e:
        print("missing-field request rejected as expected:", type(e).__name__)

# COMMAND ----------

# DBTITLE 1,Build CELLxGENE Census reference atlas
# Stratified sample of real human cells (10x 3' v3, primary data) across common cell types.
# Reference cells -> Delta table / index; held-out query cells -> nearest-neighbor evaluation.
import cellxgene_census

CENSUS_VERSION = "2025-01-30"   # pinned LTS release
N_TYPES, REF_PER_TYPE, QUERY_PER_TYPE, SEED = 40, 25, 3, 42

with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
    obs_all = cellxgene_census.get_obs(
        census, "homo_sapiens",
        value_filter="is_primary_data == True and assay == \"10x 3' v3\"",
        column_names=["soma_joinid", "cell_type", "cell_type_ontology_term_id", "tissue_general", "disease", "dataset_id"],
    )
    obs_all = obs_all[~obs_all["cell_type"].isin(["unknown", "native cell", "cell"])]
    top_types = obs_all["cell_type"].value_counts().head(N_TYPES).index
    sampled = (obs_all[obs_all["cell_type"].isin(top_types)]
               .groupby("cell_type", group_keys=False, observed=True)
               .apply(lambda g: g.sample(REF_PER_TYPE + QUERY_PER_TYPE, random_state=SEED)))
    sampled["split"] = (sampled.groupby("cell_type", observed=True).cumcount() < REF_PER_TYPE).map({True: "ref", False: "query"})
    adata = cellxgene_census.get_anndata(
        census, organism="Homo sapiens", X_name="raw",
        obs_coords=np.sort(sampled["soma_joinid"].to_numpy()),
        obs_column_names=["soma_joinid", "cell_type", "cell_type_ontology_term_id", "tissue_general", "disease", "dataset_id"],
        var_column_names=["feature_id", "feature_name"],
    )
print(f"candidate cells={len(obs_all):,}; sampled={adata.n_obs} x {adata.n_vars} genes")

# Census var_names are numeric indices -> remap to Ensembl IDs, then keep TEDDY-vocab genes only
adata.var_names = adata.var["feature_id"].astype(str).values
adata.var_names_make_unique()
assert adata.var_names[0].startswith("ENSG")
vocab_set = set(vocab_tokens)
in_vocab = np.array([g in vocab_set for g in adata.var_names])
print(f"Census genes in TEDDY vocab: {in_vocab.sum():,}/{adata.n_vars:,} ({in_vocab.mean():.1%})")
adata = adata[:, in_vocab].copy()

split_map = dict(zip(sampled["soma_joinid"], sampled["split"]))
adata.obs["split"] = adata.obs["soma_joinid"].map(split_map).values
adata.obs["cell_id"] = [f"census_{CENSUS_VERSION}_{j}" for j in adata.obs["soma_joinid"]]
print(adata.obs["split"].value_counts().to_dict(), "| cell types:", adata.obs["cell_type"].nunique())

# COMMAND ----------

# DBTITLE 1,Embed reference cells with the registered model
# Bulk embedding uses the exact registered version loaded on the notebook GPU (same artifact the endpoint serves).
import time
import scipy.sparse as sp

teddy_model = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}/{MODEL_VERSION}")
GENE_IDS = list(adata.var_names)
VAR_JSON = pd.DataFrame({"index": GENE_IDS}).to_json(orient="split")

def to_request_rows(X, cell_ids, batch=50):
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return pd.DataFrame([{
        "adata_sparsematrix": X[i:i + batch].astype(np.float32).tolist(),
        "adata_obs": pd.DataFrame({"cell_id": cell_ids[i:i + batch]}).to_json(orient="split"),
        "adata_var": VAR_JSON,
    } for i in range(0, X.shape[0], batch)])

def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

t0 = time.time()
emb_df = teddy_model.predict(to_request_rows(adata.X, adata.obs["cell_id"].tolist()), params=DEFAULT_PARAMS)
print(f"embedded {len(emb_df)} cells in {time.time() - t0:.0f}s")

cells = adata.obs[["cell_id", "cell_type", "cell_type_ontology_term_id", "tissue_general", "disease", "dataset_id", "split"]] \
    .astype(str).reset_index(drop=True).merge(emb_df, on="cell_id", how="inner")
assert len(cells) == adata.n_obs, "cell_id mismatch between request and response"
cells["embedding"] = list(unit(np.array(cells["embedding"].tolist())).astype(np.float32))  # unit-norm: L2 ranking == cosine
print("median genes used per cell:", int(cells["n_genes_used"].median()))

ref_cells = cells[cells["split"] == "ref"].reset_index(drop=True)
query_cells = cells[cells["split"] == "query"].reset_index(drop=True)
query_X = adata[adata.obs["split"] == "query"].X
query_ids = adata.obs.loc[adata.obs["split"] == "query", "cell_id"].tolist()
print(f"reference={len(ref_cells)} query={len(query_cells)}")

# COMMAND ----------

# DBTITLE 1,Write embeddings Delta table
# Source table for a Delta Sync index: PRIMARY KEY on cell_id + Change Data Feed enabled.
# Never overwrites an existing table (confirm before replacing).
if spark.catalog.tableExists(CELLS_TABLE):
    print(f"{CELLS_TABLE} already exists with {spark.table(CELLS_TABLE).count()} rows; not overwriting.")
else:
    spark.sql(f"""
        CREATE TABLE {CELLS_TABLE} (
          cell_id STRING NOT NULL,
          embedding ARRAY<FLOAT>,
          cell_type STRING,
          cell_type_ontology_term_id STRING,
          tissue_general STRING,
          disease STRING,
          dataset_id STRING,
          n_genes_used INT,
          census_version STRING,
          model_name STRING,
          model_version STRING,
          CONSTRAINT owm_teddy_cells_002_with_skill_pk PRIMARY KEY (cell_id)
        )
        COMMENT 'TEDDY-G 70M unit-normalized (512-d) embeddings of CELLxGENE Census {CENSUS_VERSION} reference cells'
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    out = ref_cells.drop(columns=["split"]).assign(
        embedding=lambda d: d["embedding"].map(lambda v: [float(x) for x in v]),
        n_genes_used=lambda d: d["n_genes_used"].astype("int32"),
        census_version=CENSUS_VERSION, model_name=UC_MODEL_NAME, model_version=str(MODEL_VERSION),
    )
    spark.createDataFrame(out).select(spark.table(CELLS_TABLE).columns) \
        .write.mode("append").saveAsTable(CELLS_TABLE)
    print(f"wrote {len(out)} rows to {CELLS_TABLE}")

display(spark.sql(f"""SELECT count(*) n_cells, count(DISTINCT cell_id) n_ids, count(DISTINCT cell_type) n_types,
                       min(size(embedding)) min_dim, max(size(embedding)) max_dim FROM {CELLS_TABLE}"""))

# COMMAND ----------

# DBTITLE 1,Create Vector Search endpoint and index (gated)
from datetime import timedelta
from databricks.sdk.service.vectorsearch import (
    EndpointType, VectorIndexType, DeltaSyncVectorIndexSpecRequest, EmbeddingVectorColumn, PipelineType,
)

EMB_DIM = int(cfg["d_model"])
VS_READY = False

if not RUN_GO:
    print("run-gate off: set widget run_go=true to create the Vector Search endpoint and index.")
else:
    # 1) Endpoint (STANDARD: low-latency ANN, ample capacity for this atlas)
    try:
        w.vector_search_endpoints.get_endpoint(VS_ENDPOINT_NAME)
        print("VS endpoint exists; waiting for ONLINE")
        w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(VS_ENDPOINT_NAME, timeout=timedelta(minutes=30))
    except NotFound:
        print("Creating VS endpoint ...")
        w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type=EndpointType.STANDARD) \
            .result(timeout=timedelta(minutes=30))
    print(f"{VS_ENDPOINT_NAME} ONLINE")

    # 2) Delta Sync index over self-managed TEDDY embeddings
    try:
        idx = w.vector_search_indexes.get_index(VS_INDEX_NAME)
        dims = [c.embedding_dimension for c in idx.delta_sync_index_spec.embedding_vector_columns]
        if dims != [EMB_DIM]:
            raise ValueError(f"Existing index has dim {dims}, expected {EMB_DIM}; confirm before dropping/recreating.")
        print("Index exists; triggering sync")
        w.vector_search_indexes.sync_index(VS_INDEX_NAME)
    except NotFound:
        w.vector_search_indexes.create_index(
            name=VS_INDEX_NAME,
            endpoint_name=VS_ENDPOINT_NAME,
            primary_key="cell_id",
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=CELLS_TABLE,
                embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=EMB_DIM)],
                pipeline_type=PipelineType.TRIGGERED,
                columns_to_sync=["cell_id", "cell_type", "cell_type_ontology_term_id", "tissue_general", "disease"],
            ),
        )
        print("Index creation started")

    # 3) Wait until the index is ready and fully indexed
    n_src = spark.table(CELLS_TABLE).count()
    deadline = time.time() + 2400
    while time.time() < deadline:
        st = w.vector_search_indexes.get_index(VS_INDEX_NAME).status
        n_idx = st.indexed_row_count or 0
        print(f"{datetime.now():%H:%M:%S} ready={st.ready} indexed={n_idx}/{n_src} {(st.message or '')[:120]}")
        if st.ready and n_idx >= n_src:
            VS_READY = True
            break
        time.sleep(30)
    if not VS_READY:
        raise TimeoutError("Vector Search index not ready in time")

# COMMAND ----------

# DBTITLE 1,Nearest-neighbor cell-type search and evaluation
# End-to-end: expression -> TEDDY embedding (serving endpoint) -> unit-normalize -> Vector Search kNN -> cell-type vote.
from collections import Counter

K = 10
VS_COLS = ["cell_id", "cell_type", "tissue_general"]

def knn(vec, k=K):
    r = w.vector_search_indexes.query_index(index_name=VS_INDEX_NAME, columns=VS_COLS,
                                            query_vector=[float(x) for x in vec], num_results=k)
    rows = (r.result.data_array or []) if r.result else []
    return [dict(zip(VS_COLS + ["score"], row)) for row in rows]  # score is always the last column

def embed_query_cells(X, ids):
    """Embed via the serving endpoint (10-cell requests for GPU_SMALL); fall back to the local registered model."""
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    if ENDPOINT_READY:
        out = []
        for i in range(0, X.shape[0], 10):
            rec = to_request_rows(X[i:i + 10], ids[i:i + 10], batch=10).iloc[0].to_dict()
            out += w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=[rec],
                                             extra_params=DEFAULT_PARAMS).predictions
        df = pd.DataFrame(out)
    else:
        df = teddy_model.predict(to_request_rows(X, ids), params=DEFAULT_PARAMS)
    return dict(zip(df["cell_id"], unit(np.array(df["embedding"].tolist()))))

if not VS_READY:
    print("Vector Search index not ready / run-gate off: skipping search evaluation.")
else:
    # 1) Self-retrieval on reference cells: top-1 is the same cell, score ~1.0, 10 results, scores descending
    for _, r in ref_cells.sample(10, random_state=SEED).iterrows():
        hits = knn(r["embedding"])
        scores = [float(h["score"]) for h in hits]
        assert hits[0]["cell_id"] == r["cell_id"] and scores[0] > 0.999, (r["cell_id"], hits[:2])
        assert len(hits) == K and all(a >= b for a, b in zip(scores, scores[1:]))
    print("self-retrieval: 10/10 passed (top-1 = query cell, monotonic scores)")

    # 2) Held-out query cells: embed through the endpoint, annotate by majority vote of K neighbors
    q_emb = embed_query_cells(query_X, query_ids)
    if ENDPOINT_READY:  # endpoint vs notebook embedding consistency
        loc = dict(zip(query_cells["cell_id"], query_cells["embedding"]))
        cons = np.array([float(np.dot(q_emb[c], loc[c])) for c in q_emb])
        print(f"endpoint vs local cosine: min={cons.min():.5f}")
    rows = []
    for _, r in query_cells.iterrows():
        hits = knn(q_emb[r["cell_id"]])
        vote, n = Counter(h["cell_type"] for h in hits).most_common(1)[0]
        rows.append({"cell_id": r["cell_id"], "true_cell_type": r["cell_type"], "predicted_cell_type": vote,
                     "vote_fraction": n / K, "top1_cell_type": hits[0]["cell_type"],
                     "top1_score": round(float(hits[0]["score"]), 4)})
    knn_results = pd.DataFrame(rows)
    acc = (knn_results["true_cell_type"] == knn_results["predicted_cell_type"]).mean()
    print(f"{K}-NN cell-type accuracy on {len(knn_results)} held-out cells: {acc:.1%} "
          f"(chance ~ {1 / ref_cells['cell_type'].nunique():.1%}, {ref_cells['cell_type'].nunique()} types)")
    display(knn_results)