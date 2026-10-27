# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # TEDDY-70M Model Serving Deployment
# MAGIC
# MAGIC Downloads TEDDY (Merck/TEDDY, Apache-2.0, arXiv:2503.03485) from Hugging Face into a UC Volume, wraps the 70M TEDDY-G variant as an MLflow PyFunc cell embedder, registers it to Unity Catalog, and deploys it to a GPU Model Serving endpoint with inference tables.

# COMMAND ----------

# DBTITLE 1,Configuration
# Central config. Re-run this cell after any restartPython() (widgets persist, Python vars do not).
dbutils.widgets.dropdown("run_go", "true", ["false", "true"], "Run costly steps (deploy)")
RUN_GO = dbutils.widgets.get("run_go") == "true"

CATALOG = "<catalog>"
SCHEMA = "skills_owm_202610"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.owm_teddy_model_001_skill_on"
ENDPOINT_NAME = "owm_teddy_ep_001_skill_on"
BASE_DIR = "/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/001_skill_on"

HF_REPO_ID = "Merck/TEDDY"
VARIANT = "70M"
SNAPSHOT_DIR = f"{BASE_DIR}/snapshot"          # full HF repo snapshot (code + weights)
SENTINEL = f"{SNAPSHOT_DIR}/.snapshot_complete"
MANIFEST_PATH = f"{BASE_DIR}/manifest.json"
TMP_DIR = "/tmp/teddy_001"                       # /local_disk0 is unavailable on Serverless

print(f"RUN_GO={RUN_GO}\nmodel={UC_MODEL_NAME}\nendpoint={ENDPOINT_NAME}\nbase={BASE_DIR}")

# COMMAND ----------

# DBTITLE 1,Download TEDDY snapshot to UC Volume
# Build-time download (never at request time). Runs in a subprocess so HF Xet / hf_transfer can be
# disabled before huggingface_hub is imported (TEDDY uses Xet storage, which fails on Serverless).
import os, sys, json, shutil, subprocess, hashlib, datetime

os.makedirs(BASE_DIR, exist_ok=True)

if os.path.exists(SENTINEL):
    print("Snapshot already present, skipping download:", SNAPSHOT_DIR)
else:
    local_dir = f"{TMP_DIR}/hf_snapshot"
    shutil.rmtree(local_dir, ignore_errors=True)
    env = {**os.environ, "HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    script = f'''
import os, json
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import HfApi, snapshot_download
info = HfApi().model_info("{HF_REPO_ID}")
sha = info.sha                       # pin to an immutable commit, not 'main'
snapshot_download(repo_id="{HF_REPO_ID}", revision=sha, local_dir="{local_dir}")
json.dump({{"sha": sha}}, open("{TMP_DIR}/hf_revision.json", "w"))
'''
    subprocess.check_call([sys.executable, "-c", script], env=env)
    # Copy to the governed Volume (drop the HF local cache dir)
    shutil.rmtree(SNAPSHOT_DIR, ignore_errors=True)
    shutil.copytree(local_dir, SNAPSHOT_DIR, ignore=shutil.ignore_patterns(".cache"))
    shutil.copy2(f"{TMP_DIR}/hf_revision.json", f"{SNAPSHOT_DIR}/.hf_revision.json")
    open(SENTINEL, "w").write(datetime.datetime.utcnow().isoformat())
    print("Downloaded to", SNAPSHOT_DIR)

HF_REVISION = json.load(open(f"{SNAPSHOT_DIR}/.hf_revision.json"))["sha"]
print("HF revision (commit SHA):", HF_REVISION)

# COMMAND ----------

# DBTITLE 1,Provenance manifest
# Record provenance next to the weights: source, immutable revision, licenses, checksums, date.
import hashlib, json, os, datetime

MODEL_DIR = f"{SNAPSHOT_DIR}/teddy/models/teddy_g/{VARIANT}"

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

manifest = {
    "model": {"name": "TEDDY-G", "variant": VARIANT, "paper": "arXiv:2503.03485",
              "upstream_revision": HF_REVISION, "reviewed_at": datetime.date.today().isoformat()},
    "code": {"source_url": f"https://huggingface.co/{HF_REPO_ID}", "revision": HF_REVISION},
    "weights": {"source_url": f"https://huggingface.co/{HF_REPO_ID}", "revision": HF_REVISION,
                "path": f"{MODEL_DIR}/model.safetensors", "sha256": sha256(f"{MODEL_DIR}/model.safetensors")},
    "license": {"model_card": "apache-2.0",
                "code_pyproject_field": "MIT (as declared in pyproject.toml; model card states Apache-2.0)"},
    "artifacts": [{"name": f, "sha256": sha256(f"{MODEL_DIR}/{f}")}
                  for f in ["config.json", "vocab.txt", "tokenizer_config.json", "added_tokens.json",
                            "special_tokens_map.json"]]
                 + [{"name": "teddy_gene_medians.json",
                     "sha256": sha256(f"{SNAPSHOT_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json")}],
    "runtime": {"accelerator": "GPU (serving: GPU_SMALL)", "network_required_at_runtime": False,
                "pins": "transformers==4.41.0 (from upstream pyproject.toml)"},
}
json.dump(manifest, open(MANIFEST_PATH, "w"), indent=2)
print(json.dumps(manifest, indent=2)[:1500])

# COMMAND ----------

# DBTITLE 1,Install serving-path dependencies
# transformers is pinned to the upstream pyproject version (4.41.0); 5.x breaks TeddyGModel.
# Installed here only for the local dry-load; the serving env gets its own pinned pip_requirements.
%pip install -q transformers==4.41.0 tokenizers==0.19.1 "mlflow>=3.1" anndata uv
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Re-load configuration after restart
import os, sys, json, shutil
RUN_GO = dbutils.widgets.get("run_go") == "true"
CATALOG = "<catalog>"
SCHEMA = "skills_owm_202610"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.owm_teddy_model_001_skill_on"
ENDPOINT_NAME = "owm_teddy_ep_001_skill_on"
BASE_DIR = "/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/001_skill_on"
HF_REPO_ID, VARIANT = "Merck/TEDDY", "70M"
SNAPSHOT_DIR = f"{BASE_DIR}/snapshot"
MANIFEST_PATH = f"{BASE_DIR}/manifest.json"
MODEL_DIR = f"{SNAPSHOT_DIR}/teddy/models/teddy_g/{VARIANT}"
MEDIANS_PATH = f"{SNAPSHOT_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json"
HF_REVISION = json.load(open(f"{SNAPSHOT_DIR}/.hf_revision.json"))["sha"]
TMP_DIR = "/tmp/teddy_001"
CODE_DIR = f"{TMP_DIR}/teddy_code"            # code bundle (no weights)
WRAPPER_PATH = f"{TMP_DIR}/teddy_wrapper.py"

import transformers
assert transformers.__version__ == "4.41.0", transformers.__version__
print("revision", HF_REVISION, "| transformers", transformers.__version__)

# COMMAND ----------

# DBTITLE 1,Write PyFunc wrapper file
# File-based PyFunc (python_model=<path>) so the bundled, non-pip `teddy` package is never cloudpickled.
# All imports live inside the file: the serving container has no notebook globals.
WRAPPER_SRC = r'''
import io, os, sys, json, inspect
import numpy as np
import pandas as pd
import torch
import mlflow
from mlflow.pyfunc import PythonModel

MAX_CELLS_PER_REQUEST = 64


class TEDDYEmbedder(PythonModel):
    """TEDDY-G cell embedder. Input: raw counts (cells x genes) + gene Ensembl IDs.
    Mirrors upstream preprocessing: normalize_total(1e4) -> divide by TEDDY gene medians ->
    keep vocab genes -> top-k expressed genes (desc), zeros dropped -> TeddyGModel."""

    def load_context(self, context):
        # Serving containers replace stdout with a logger lacking isatty(); transformers calls it.
        for s in (sys.stdout, sys.stderr):
            if s is not None and not hasattr(s, "isatty"):
                s.isatty = lambda: False
        # Purge any stale teddy modules, then put the bundled code first on sys.path.
        for m in list(sys.modules):
            if m == "teddy" or m.startswith("teddy."):
                del sys.modules[m]
        pkg_parent = context.artifacts["teddy_pkg_parent"]
        if pkg_parent in sys.path:
            sys.path.remove(pkg_parent)
        sys.path.insert(0, pkg_parent)

        from teddy.models.model_directory import get_architecture, model_dict
        from teddy.tokenizer.gene_tokenizer import GeneTokenizer

        model_dir = context.artifacts["model_dir"]
        arch = get_architecture(model_dir)
        config = model_dict[arch]["config_cls"].from_pretrained(model_dir)
        self.model = model_dict[arch]["model_cls"].from_pretrained(model_dir, config=config)
        self.model.return_all_embs = True             # token-level states for mean pooling
        self.model.return_cell_embs_first_token = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

        self.tokenizer = GeneTokenizer.from_pretrained(model_dir)
        self.vocab = dict(self.tokenizer.vocab)       # Ensembl ID -> token id (special tokens excluded)
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.max_pos = int(config.max_position_embeddings)
        self.d_model = int(config.d_model)
        with open(context.artifacts["gene_medians"]) as f:
            self.medians = json.load(f)
        self._fwd = set(inspect.signature(self.model.forward).parameters)
        print(f"TEDDY-G loaded on {self.device}, d_model={self.d_model}, vocab={len(self.vocab)}")

    # ---------- helpers ----------
    @staticmethod
    def _matrix(cell):
        rows = [np.asarray(r, dtype=np.float32) for r in cell]
        if not rows:
            raise ValueError("adata_sparsematrix is empty")
        if len({len(r) for r in rows}) != 1:
            raise ValueError("adata_sparsematrix rows must all have the same length (n_genes)")
        return np.stack(rows)

    def _embed(self, X, genes, max_seq_len, pooling, preprocess, batch_size):
        if preprocess:
            tot = X.sum(axis=1, keepdims=True)
            tot[tot == 0] = 1.0
            X = X / tot * 1e4
            X = X * np.array([1.0 / self.medians[g] if g in self.medians else 1.0 for g in genes],
                             dtype=np.float32)[None, :]
        keep = [i for i, g in enumerate(genes) if g in self.vocab]
        if not keep:
            raise ValueError("No gene IDs in adata_var match the TEDDY vocabulary (expects Ensembl IDs, e.g. ENSG...)")
        tok = torch.tensor([self.vocab[genes[i]] for i in keep], dtype=torch.long)
        Xv = torch.from_numpy(np.ascontiguousarray(X[:, keep]))
        k = min(max_seq_len, Xv.shape[1])
        vals, idx = torch.topk(Xv, k=k, dim=1, largest=True, sorted=True)
        ids = tok[idx]
        mask = vals > 0                                # drop zero-expression genes (include_zero_genes=False)
        ids[~mask] = self.pad_id
        n_used = mask.sum(1)

        out = []
        for s in range(0, ids.shape[0], batch_size):
            b_ids, b_mask, b_n = ids[s:s + batch_size], mask[s:s + batch_size], n_used[s:s + batch_size]
            L = max(1, int(b_n.max()))
            b_ids, b_mask = b_ids[:, :L].to(self.device), b_mask[:, :L].to(self.device)
            safe_mask = b_mask.clone()
            safe_mask[:, 0] = True                     # avoid all-masked rows (NaN); flagged below
            kw = {"gene_ids": b_ids, "attention_mask": safe_mask.long()}
            kw = {k_: v for k_, v in kw.items() if k_ in self._fwd}
            with torch.no_grad():
                o = self.model(**kw)
            if pooling == "first_token":
                emb = o.get("cell_emb")
            else:
                h = o.get("all_embs")
                m = b_mask.unsqueeze(-1).to(h.dtype)
                emb = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            emb = emb.float().cpu().numpy()
            for e, n in zip(emb, b_n.tolist()):
                out.append((e.tolist() if n > 0 else [], int(n)))
        return out

    # ---------- inference ----------
    def predict(self, context, model_input, params=None):
        params = params or {}
        max_seq_len = min(int(params.get("max_seq_len", 2048)), self.max_pos)
        pooling = str(params.get("pooling", "mean"))
        preprocess = str(params.get("preprocess", "true")).lower() == "true"
        batch_size = max(1, int(params.get("batch_size", 8)))
        if pooling not in ("mean", "first_token"):
            raise ValueError("pooling must be 'mean' or 'first_token'")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")

        records = []
        for _, row in model_input.iterrows():
            X = self._matrix(row["adata_sparsematrix"])
            obs = pd.read_json(io.StringIO(row["adata_obs"]), orient="split", dtype=False)
            var = pd.read_json(io.StringIO(row["adata_var"]), orient="split", dtype=False)
            genes = [str(g) for g in (var["index"] if "index" in var.columns else var.index)]
            if X.shape[1] != len(genes):
                raise ValueError(f"matrix has {X.shape[1]} genes but adata_var has {len(genes)}")
            if len(obs) != X.shape[0]:
                raise ValueError(f"matrix has {X.shape[0]} cells but adata_obs has {len(obs)} rows")
            if X.shape[0] > MAX_CELLS_PER_REQUEST:
                raise ValueError(f"max {MAX_CELLS_PER_REQUEST} cells per request; use a Job for bulk embedding")
            if not np.isfinite(X).all() or (X < 0).any():
                raise ValueError("expression values must be finite and non-negative raw counts")
            cell_ids = [str(c) for c in (obs["cell_id"] if "cell_id" in obs.columns else obs.index)]
            for cid, (emb, n) in zip(cell_ids, self._embed(X, genes, max_seq_len, pooling, preprocess, batch_size)):
                records.append({"cell_id": cid, "embedding": emb, "n_genes_used": n})
        return pd.DataFrame(records, columns=["cell_id", "embedding", "n_genes_used"])


mlflow.models.set_model(TEDDYEmbedder())
'''

os.makedirs(TMP_DIR, exist_ok=True)
open(WRAPPER_PATH, "w").write(WRAPPER_SRC)
shutil.copy2(WRAPPER_PATH, f"{BASE_DIR}/teddy_wrapper.py")   # keep a governed copy next to the weights

# Code bundle: only the packages the wrapper imports (models + tokenizer), no weights / variant dirs.
shutil.rmtree(CODE_DIR, ignore_errors=True)
shutil.copytree(
    f"{SNAPSHOT_DIR}/teddy", f"{CODE_DIR}/teddy",
    ignore=shutil.ignore_patterns("data_processing", "70M", "160M", "400M", "*.safetensors", "*.bin",
                                  "*.ckpt", "*.pt", "__pycache__", ".DS_Store", "*.ipynb"),
)
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(CODE_DIR) for f in fs)
print(f"wrapper: {WRAPPER_PATH}\ncode bundle: {CODE_DIR} ({size/1e6:.1f} MB)")

# COMMAND ----------

# DBTITLE 1,Dry-load wrapper and build real-cell input example
# Dry-load the wrapper in-kernel and build the input_example from REAL cells (raw counts) in the
# upstream sample_data.h5ad, so gene IDs are real Ensembl IDs that hit the vocab (not <unk>).
import importlib.util, types
import numpy as np, pandas as pd, anndata as ad

ARTIFACTS = {"model_dir": MODEL_DIR, "teddy_pkg_parent": CODE_DIR, "gene_medians": MEDIANS_PATH}
for _k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.") or k == "teddy_wrapper"]:
    del sys.modules[_k]
spec = importlib.util.spec_from_file_location("teddy_wrapper", WRAPPER_PATH)
tw = importlib.util.module_from_spec(spec); spec.loader.exec_module(tw)
embedder = tw.TEDDYEmbedder()
embedder.load_context(types.SimpleNamespace(artifacts=ARTIFACTS))

adata = ad.read_h5ad(f"{SNAPSHOT_DIR}/data/sample_data.h5ad")
raw = adata.raw.to_adata() if adata.raw is not None else adata   # upstream preprocess starts from raw counts
N_EX = 3
X_ex = raw.X[:N_EX].toarray().astype(np.float32)
assert np.all(np.mod(X_ex, 1) == 0), "expected integer raw counts"
nz_genes = np.where(X_ex.sum(0) > 0)[0]            # dropping all-zero genes leaves totals/ranks unchanged
X_ex = X_ex[:, nz_genes]
genes_ex = raw.var_names[nz_genes].tolist()
obs_ex = pd.DataFrame({"cell_id": raw.obs_names[:N_EX].astype(str),
                       "cell_type": raw.obs["cell_type"].astype(str).values[:N_EX]})
var_ex = pd.DataFrame({"index": genes_ex})

input_example = pd.DataFrame({
    "adata_sparsematrix": [X_ex.tolist()],          # one row = one batch of cells (dense counts)
    "adata_obs": [obs_ex.to_json(orient="split")],
    "adata_var": [var_ex.to_json(orient="split")],
})
DEFAULT_PARAMS = {"max_seq_len": 2048, "pooling": "mean", "preprocess": "true", "batch_size": 8}

pred = embedder.predict(None, input_example, DEFAULT_PARAMS)
E = np.array(pred["embedding"].tolist())
print(f"genes in example: {len(genes_ex)}, in vocab: {sum(g in embedder.vocab for g in genes_ex)}")
print(pred[["cell_id", "n_genes_used"]].assign(dim=E.shape[1], norm=np.linalg.norm(E, axis=1).round(3),
                                               cell_type=obs_ex["cell_type"].values))
assert E.shape == (N_EX, embedder.d_model) and np.isfinite(E).all() and (np.linalg.norm(E, axis=1) > 0).all()
assert not np.allclose(E[0], E[1]), "embeddings degenerate (identical across cells)"

# COMMAND ----------

# DBTITLE 1,Known-answer check vs upstream tokenization
# Cross-check the wrapper's token selection against the UPSTREAM tokenizer code path
# (teddy.data_processing.tokenization._build_batch_tensors) on the same cells.
import torch
from scipy.sparse import csr_matrix
# The wrapper bundle (CODE_DIR) omits data_processing; import upstream code from the full snapshot instead.
for _k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.")]:
    del sys.modules[_k]
sys.path.insert(0, SNAPSHOT_DIR)
from teddy.tokenizer.gene_tokenizer import GeneTokenizer
from teddy.tokenizer.tokenization_args import TokenizationArgs
from teddy.data_processing.tokenization.tokenization import _build_batch_tensors

tok = GeneTokenizer.from_pretrained(MODEL_DIR)
medians = json.load(open(MEDIANS_PATH))
# Upstream preprocess: row L1-normalize * 1e4, then scale columns by 1/median (1.0 if missing)
Xp = X_ex / X_ex.sum(1, keepdims=True) * 1e4
Xp = Xp * np.array([1.0 / medians[g] if g in medians else 1.0 for g in genes_ex], dtype=np.float32)
in_vocab = np.where([g in tok.vocab for g in genes_ex])[0]
token_array = torch.tensor(tok.encode([genes_ex[i] for i in in_vocab], add_special_tokens=False))
args = types.SimpleNamespace(max_seq_len=2048, add_cls=False, random_genes=False)
up_ids, up_vals, _, _ = _build_batch_tensors(torch.tensor(Xp[:, in_vocab]), token_array, args)

# Upstream-style forward on the upstream token ids (zeros dropped, as include_zero_genes=False)
model = embedder.model
for i in range(N_EX):
    nz = up_vals[i] != 0
    ids_i = up_ids[i][nz].unsqueeze(0).to(embedder.device)
    with torch.no_grad():
        h = model(gene_ids=ids_i, attention_mask=torch.ones_like(ids_i))["all_embs"]
    ref = h.mean(1).float().cpu().numpy()[0]
    cos = float(ref @ E[i] / (np.linalg.norm(ref) * np.linalg.norm(E[i])))
    print(f"cell {i}: upstream n_tokens={int(nz.sum())} wrapper n={pred.n_genes_used[i]}  cosine(upstream, wrapper)={cos:.6f}")
    assert int(nz.sum()) == pred.n_genes_used[i] and cos > 0.9999

# COMMAND ----------

# DBTITLE 1,Log and register model to Unity Catalog
import mlflow, importlib.metadata as md
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
me = spark.sql("select current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/owm_teddy_001_skill_on")

# Params are declared as strings: the Databricks SDK sends extra_params as Dict[str, str]; the wrapper casts.
SERVING_PARAMS = {k: str(v) for k, v in DEFAULT_PARAMS.items()}
signature = infer_signature(input_example, pred, params=SERVING_PARAMS)
print(signature)

# Pins come from upstream pyproject.toml (transformers/tokenizers/safetensors/numpy/pandas) and torch ^2.3.
# torch is pinned to a cu12x wheel line rather than ">=2.3" so the serving image does not float to a new major/CUDA.
pip_requirements = [
    f"mlflow=={md.version('mlflow')}",
    "torch==2.3.1",
    "transformers==4.41.0",
    "tokenizers==0.19.1",
    "safetensors==0.4.3",
    "huggingface-hub==0.23.1",
    "numpy==1.26.4",
    "pandas==2.2.2",
]

with mlflow.start_run(run_name=f"teddy_{VARIANT.lower()}_register") as run:
    mlflow.log_params({"variant": VARIANT, "hf_repo": HF_REPO_ID, "hf_revision": HF_REVISION})
    mlflow.set_tags({"license": "apache-2.0", "paper": "arXiv:2503.03485",
                     "source_url": f"https://huggingface.co/{HF_REPO_ID}", "hf_revision": HF_REVISION})
    mlflow.log_artifact(MANIFEST_PATH)
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=WRAPPER_PATH,                       # file-based: no cloudpickle of teddy imports
        artifacts=ARTIFACTS,                             # weights+vocab, code bundle, gene medians
        signature=signature,
        input_example=(input_example, SERVING_PARAMS),
        pip_requirements=pip_requirements,
        registered_model_name=UC_MODEL_NAME,
    )

MODEL_VERSION = model_info.registered_model_version
client = mlflow.MlflowClient()
client.update_registered_model(UC_MODEL_NAME, description=(
    "TEDDY-G 70M single-cell foundation model (Merck/TEDDY, Apache-2.0, arXiv:2503.03485). "
    "Input: raw counts + Ensembl gene IDs; output: 512-d per-cell embeddings."))
for k, v in {"hf_revision": HF_REVISION, "variant": VARIANT, "manifest": MANIFEST_PATH}.items():
    client.set_model_version_tag(UC_MODEL_NAME, str(MODEL_VERSION), k, v)
client.set_registered_model_alias(UC_MODEL_NAME, "candidate", MODEL_VERSION)   # promoted to champion after smoke test
print(f"Registered {UC_MODEL_NAME} v{MODEL_VERSION}  ({model_info.model_uri})")

# COMMAND ----------

# DBTITLE 1,Pre-deployment validation in an isolated serving-like env
# Rebuild the model's pinned environment from scratch (uv venv) and score the serving payload.
# This catches missing/incompatible deps and cloudpickle/sys.path issues a same-kernel dry-load cannot.
from mlflow.models import convert_input_example_to_serving_input

serving_payload = convert_input_example_to_serving_input((input_example, SERVING_PARAMS))
in_path, out_path = f"{TMP_DIR}/serving_input.json", f"{TMP_DIR}/serving_output.json"
open(in_path, "w").write(serving_payload)
mlflow.models.predict(
    model_uri=f"models:/{UC_MODEL_NAME}/{MODEL_VERSION}",
    input_path=in_path, output_path=out_path,
    content_type="json",
    env_manager="uv",
)
# Parity: pinned env (torch 2.3.1 / numpy 1.26) vs. in-kernel dry-load
E_iso = np.array([p["embedding"] for p in json.load(open(out_path))["predictions"]])
cos = (E_iso * E).sum(1) / (np.linalg.norm(E_iso, axis=1) * np.linalg.norm(E, axis=1))
print("cosine(isolated env, dry-load) per cell:", cos.round(6))
assert E_iso.shape == E.shape and (cos > 0.999).all()

# COMMAND ----------

# DBTITLE 1,Deploy GPU Model Serving endpoint
# GPU_SMALL is sufficient for 70M (285 MB weights); scale-to-zero limits idle cost.
# Inference tables + usage tracking via AI Gateway (timestamped prefix avoids *_payload name conflicts).
# Payloads contain gene-expression profiles: access to the inference table should follow your data-governance rules.
import time
from datetime import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType,
    AiGatewayConfig, AiGatewayInferenceTableConfig, AiGatewayUsageTrackingConfig,
    EndpointStateReady, EndpointStateConfigUpdate,
)

w = WorkspaceClient()
SERVED_NAME = f"teddy-{VARIANT.lower()}-v{MODEL_VERSION}"
served_entity = ServedEntityInput(
    name=SERVED_NAME,
    entity_name=UC_MODEL_NAME,
    entity_version=str(MODEL_VERSION),
    workload_type=ServingModelWorkloadType.GPU_SMALL,
    workload_size="Small",
    scale_to_zero_enabled=True,
)

if not RUN_GO:
    print("run_go=false: skipping endpoint create/update")
else:
    existing = {e.name for e in w.serving_endpoints.list()}
    if ENDPOINT_NAME in existing:
        print("Endpoint exists -> updating served entity")
        w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=[served_entity])
    else:
        w.serving_endpoints.create(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity]),
            ai_gateway=AiGatewayConfig(
                inference_table_config=AiGatewayInferenceTableConfig(
                    catalog_name=CATALOG, schema_name=SCHEMA,
                    table_name_prefix=f"{ENDPOINT_NAME}_{datetime.now():%Y%m%d%H%M}",
                    enabled=True),
                usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
            ),
        )
        print("Endpoint creation submitted:", ENDPOINT_NAME)

    # Poll up to 30 min; fail fast if the served entity deployment fails.
    t0 = time.time()
    while True:
        ep = w.serving_endpoints.get(ENDPOINT_NAME)
        ready, upd = ep.state.ready, ep.state.config_update
        ents = (ep.pending_config.served_entities if ep.pending_config else ep.config.served_entities) or []
        dep = [(e.name, e.state.deployment.value if e.state and e.state.deployment else None,
                (e.state.deployment_state_message or "")[:200] if e.state else "") for e in ents]
        print(f"[{int(time.time()-t0)}s] ready={ready.value if ready else None} update={upd.value if upd else None} {dep}")
        if ready == EndpointStateReady.READY and upd == EndpointStateConfigUpdate.NOT_UPDATING:
            break
        if upd == EndpointStateConfigUpdate.UPDATE_FAILED or any(d[1] == "DEPLOYMENT_FAILED" for d in dep):
            raise RuntimeError(f"Deployment failed: {dep}. Check service logs for {SERVED_NAME}.")
        if time.time() - t0 > 1800:
            raise TimeoutError("Endpoint not ready after 30 min")
        time.sleep(60)
    print("Endpoint READY:", ENDPOINT_NAME)

# COMMAND ----------

# DBTITLE 1,Endpoint smoke, parity and negative tests
# Query with 5 UNSEEN real cells (not in input_example), compare to the local dry-load, then negative tests.
from databricks.sdk.errors import DatabricksError

sl = slice(10, 15)
Xq = raw.X[sl].toarray().astype(np.float32)
gq = np.where(Xq.sum(0) > 0)[0]
Xq, genes_q = Xq[:, gq], raw.var_names[gq].tolist()
obs_q = pd.DataFrame({"cell_id": raw.obs_names[sl].astype(str)})
record = {"adata_sparsematrix": Xq.tolist(),
          "adata_obs": obs_q.to_json(orient="split"),
          "adata_var": pd.DataFrame({"index": genes_q}).to_json(orient="split")}

# NOTE: verified on this endpoint that SDK `extra_params` are merged at the top level of the body and are
# silently IGNORED by the MLflow scoring server. Inference params must go under the "params" key.
def invoke(records, params=None):
    body = {"dataframe_records": records}
    if params:
        body["params"] = {k: str(v) for k, v in params.items()}
    return w.api_client.do("POST", f"/serving-endpoints/{ENDPOINT_NAME}/invocations", body=body)

resp = w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=[record])   # SDK path, default params
preds = resp.predictions                                    # typed response: attribute access
E_ep = np.array([p["embedding"] for p in preds])
E_loc = np.array(embedder.predict(None, pd.DataFrame([record]), DEFAULT_PARAMS)["embedding"].tolist())
cos = (E_ep * E_loc).sum(1) / (np.linalg.norm(E_ep, axis=1) * np.linalg.norm(E_loc, axis=1))
print(pd.DataFrame({"cell_id": [p["cell_id"] for p in preds], "n_genes_used": [p["n_genes_used"] for p in preds],
                    "dim": E_ep.shape[1], "cos_vs_local": cos.round(6)}))
assert E_ep.shape == (5, 512) and (cos > 0.999).all()
assert [p["cell_id"] for p in preds] == obs_q.cell_id.tolist()

# Params path (REST body "params"): first_token pooling must differ from mean and match local
r2 = invoke([record], {"pooling": "first_token", "max_seq_len": 2048})["predictions"]
E_ft = np.array([p["embedding"] for p in r2])
E_ft_loc = np.array(embedder.predict(None, pd.DataFrame([record]), {**DEFAULT_PARAMS, "pooling": "first_token"})["embedding"].tolist())
cos_ft = (E_ft * E_ft_loc).sum(1) / (np.linalg.norm(E_ft, axis=1) * np.linalg.norm(E_ft_loc, axis=1))
cos_mf = (E_ft * E_ep).sum(1) / (np.linalg.norm(E_ft, axis=1) * np.linalg.norm(E_ep, axis=1))
print("first_token vs local:", cos_ft.round(6), "| first_token vs mean:", cos_mf.round(3))
assert (cos_ft > 0.999).all() and (cos_mf < 0.999).all(), "params not applied"

def expect_error(label, records, params=None):
    try:
        invoke(records, params)
        raise AssertionError(f"{label}: expected an error")
    except DatabricksError as e:
        print(f"{label}: rejected as expected -> {str(e)[:150]}")

expect_error("missing adata_obs", [{k: v for k, v in record.items() if k != "adata_obs"}])
expect_error("all-OOV gene IDs", [{**record, "adata_var": pd.DataFrame(
    {"index": [f"FAKE{i}" for i in range(len(genes_q))]}).to_json(orient="split")}])
expect_error("bad pooling", [record], {"pooling": "max"})

# Promote the validated version
mlflow.MlflowClient().set_registered_model_alias(UC_MODEL_NAME, "champion", MODEL_VERSION)
print(f"Alias 'champion' -> v{MODEL_VERSION}")

# COMMAND ----------

# DBTITLE 1,Request contract
# MAGIC %md
# MAGIC **Request contract** (`POST /serving-endpoints/owm_teddy_ep_001_skill_on/invocations`)
# MAGIC
# MAGIC * `dataframe_records`: one record per batch of up to 64 cells, with `adata_sparsematrix` (dense raw counts, cells x genes), `adata_obs` (JSON `orient=split`, with a `cell_id` column), and `adata_var` (JSON `orient=split`, with an `index` column of Ensembl IDs).
# MAGIC * `params` (optional, string values): `pooling` (`mean` | `first_token`), `max_seq_len` (<= 2048), `preprocess` (`true` applies TEDDY normalize-total + median scaling; use `false` only for already-processed input), and `batch_size`.
# MAGIC * Send params under the `params` key. The SDK `extra_params` argument is silently ignored by this endpoint.
# MAGIC * Response: one prediction per cell with `cell_id`, `embedding` (512 floats), and `n_genes_used`. An empty embedding means the cell had no expressed vocabulary genes.
# MAGIC * For bulk AnnData embedding, use a Job instead of this endpoint.