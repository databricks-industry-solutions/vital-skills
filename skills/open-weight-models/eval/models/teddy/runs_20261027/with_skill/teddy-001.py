# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # TEDDY-70M Model Serving Deployment
# MAGIC
# MAGIC Downloads Merck/TEDDY (pinned HF revision) to a UC Volume, wraps TEDDY-G 70M as a file-based MLflow PyFunc cell embedder, validates it against upstream TEDDY preprocessing/tokenization, registers it to Unity Catalog, and deploys a GPU Model Serving endpoint with an inference table. Costly steps are gated by the `run_go` widget (saved default `false`).

# COMMAND ----------

# DBTITLE 1,Install pinned TEDDY dependencies
# Pins come from TEDDY's own pyproject.toml (transformers==4.41.0 exact; 5.x breaks TeddyGModel).
# anndata/scanpy are only for the upstream-parity test in this notebook, not the serving container.
%pip install -q transformers==4.41.0 tokenizers==0.19.1 numpy==1.26.4 anndata==0.10.9 scanpy==1.10.4
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration and run_go gate
# Configuration. Names are fixed by the request; HF revision is pinned to an immutable commit SHA.
import os, datetime

dbutils.widgets.dropdown("run_go", "false", ["false", "true"], "Run costly steps")  # saved default stays "false"
RUN_GO = dbutils.widgets.get("run_go") == "true"

CATALOG  = "<catalog>"
SCHEMA   = "skills_owm_202610"
VOLUME   = "owm_teddy_files"
BASE_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/teddy-eval-2026-10-27/001_with_skill"

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.owm_teddy_model_001_with_skill_r2"
ENDPOINT_NAME = "owm_teddy_ep_001_with_skill_r2"

HF_REPO     = "Merck/TEDDY"
HF_REVISION = "8c880aa67674042ccdc254bccb5de081092a48bf"  # immutable commit (== main on 2026-10-27)
VARIANT     = "70M"

SNAPSHOT_DIR = f"{BASE_DIR}/snapshot"                         # full repo (minus other variants' weights)
MODEL_DIR    = f"{SNAPSHOT_DIR}/teddy/models/teddy_g/{VARIANT}"  # checkpoint + vocab
SENTINEL     = f"{SNAPSHOT_DIR}/.snapshot_complete"

# /local_disk0 is not available on Serverless; fall back to /tmp
try:
    os.makedirs("/local_disk0/tmp", exist_ok=True); TMP_DIR = "/local_disk0/tmp"
except (PermissionError, OSError):
    TMP_DIR = "/tmp"

print(f"RUN_GO={RUN_GO}  BASE_DIR={BASE_DIR}  TMP_DIR={TMP_DIR}")

# COMMAND ----------

# DBTITLE 1,Download pinned TEDDY snapshot to the UC Volume
# Build-time download (never at request time). Runs in a subprocess so HF Xet / hf_transfer
# settings take effect even if huggingface_hub was already imported by the base environment.
# Only the 70M weights are fetched (160M/400M safetensors skipped, ~2.2 GB saved).
import subprocess, sys, shutil, hashlib, json, yaml

if os.path.exists(SENTINEL):
    print(f"Snapshot already present (sentinel found): {SNAPSHOT_DIR}")
else:
    local_dir = f"{TMP_DIR}/teddy_snapshot"
    shutil.rmtree(local_dir, ignore_errors=True)
    dl = f'''
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download
snapshot_download(repo_id="{HF_REPO}", revision="{HF_REVISION}", local_dir="{local_dir}",
                  ignore_patterns=["teddy/models/teddy_g/160M/*.safetensors",
                                   "teddy/models/teddy_g/400M/*.safetensors", "*.DS_Store"])
'''
    env = {**os.environ, "HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    subprocess.check_call([sys.executable, "-c", dl], env=env)
    shutil.rmtree(f"{local_dir}/.cache", ignore_errors=True)
    os.makedirs(BASE_DIR, exist_ok=True)
    shutil.copytree(local_dir, SNAPSHOT_DIR, dirs_exist_ok=True)
    open(SENTINEL, "w").write(datetime.datetime.utcnow().isoformat())
    print("Downloaded and copied to", SNAPSHOT_DIR)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# Licenses: model card front matter vs bundled package pyproject (both recorded, neither asserted as governing)
readme = open(f"{SNAPSHOT_DIR}/README.md").read()
card_license = next((l.split(":", 1)[1].strip() for l in readme.splitlines() if l.startswith("license:")), "unknown")
pyproject = open(f"{SNAPSHOT_DIR}/pyproject.toml").read()
pkg_license = next((l.split("=", 1)[1].strip().strip('"') for l in pyproject.splitlines() if l.strip().startswith("license")), "unknown")

manifest = {
    "model": {"name": "TEDDY-G", "variant": VARIANT, "upstream_revision": HF_REVISION,
              "paper": "arXiv:2503.03485", "reviewed_at": "2026-10-27"},
    "code":    {"source_url": f"https://huggingface.co/{HF_REPO}", "revision": HF_REVISION},
    "weights": {"source_url": f"https://huggingface.co/{HF_REPO}", "revision_or_record": HF_REVISION,
                "sha256": sha256(f"{MODEL_DIR}/model.safetensors")},
    "license": {"model_card": card_license, "python_package_pyproject": pkg_license, "license_file": "none in repo",
                "note": "Both declarations recorded; confirm with model owner before redistribution."},
    "runtime": {"python": sys.version.split()[0], "accelerator": "GPU (serving: GPU_SMALL)",
                "network_required_at_runtime": False},
    "artifacts": [{"name": n, "location": f"{MODEL_DIR}/{n}", "sha256": sha256(f"{MODEL_DIR}/{n}")}
                  for n in ["vocab.txt", "config.json", "tokenizer_config.json"]] +
                 [{"name": "teddy_gene_medians.json",
                   "location": f"{SNAPSHOT_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json",
                   "sha256": sha256(f"{SNAPSHOT_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json")}],
    "acquired_at": open(SENTINEL).read().strip(),
}
with open(f"{BASE_DIR}/provenance_manifest.yaml", "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)
print(yaml.safe_dump(manifest, sort_keys=False))

# COMMAND ----------

# DBTITLE 1,Stage code bundle and write file-based PyFunc wrapper
# The teddy package is NOT pip-installable, so the wrapper is logged from a .py file
# (python_model=<path>) rather than a pickled instance: cloudpickle.load() runs before code_paths
# are on sys.path in the serving container and would fail with ModuleNotFoundError: teddy.
import shutil

CODE_DIR = f"{TMP_DIR}/teddy_code"
shutil.rmtree(CODE_DIR, ignore_errors=True)
shutil.copytree(
    f"{SNAPSHOT_DIR}/teddy", f"{CODE_DIR}/teddy",
    ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.ckpt", "*.pt", "__pycache__", ".DS_Store",
                                  # gene-mapping tables (~50 MB) are not used by the serving path
                                  "2407_*", "human_mapping.txt", "mouse_to_human_orthologs*"),
)

WRAPPER_SRC = r'''
"""TEDDY-G cell embedder (MLflow file-based PyFunc).

Request (one DataFrame row = one batch of cells):
  adata_sparsematrix : list[list[float]]  raw counts, shape (cells, genes)
  adata_obs          : str, pandas JSON orient=split; optional `cell_id` column
  adata_var          : str, pandas JSON orient=split; `index` column = Ensembl gene IDs
Params (REST body "params"): max_seq_len (str, default "2048")
Response: one row per cell -> {cell_id, embedding[d_model]}
"""
# Every symbol used below is imported here: the serving container has no notebook globals.
import sys, os, io, json, inspect
import numpy as np
import pandas as pd
import torch
import mlflow

MAX_CELLS_PER_FORWARD = 10   # GPU_SMALL (T4 16 GB) headroom at seq_len 2048
MIN_VOCAB_FRACTION = 0.10    # upstream TEDDY raises below 10% vocabulary coverage


class TEDDYEmbedder(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # Serving replaces stdout with a logger lacking isatty(); transformers calls it.
        for stream in (sys.stdout, sys.stderr):
            if stream is not None and not hasattr(stream, "isatty"):
                stream.isatty = lambda: False
        # Purge stale teddy modules (safe for notebook dry-loads; no-op in a fresh container)
        for _m in list(sys.modules):
            if _m == "teddy" or _m.startswith("teddy."):
                del sys.modules[_m]
        pkg_parent = context.artifacts["teddy_pkg_parent"]
        if pkg_parent in sys.path:
            sys.path.remove(pkg_parent)
        sys.path.insert(0, pkg_parent)

        from teddy.models.model_directory import get_architecture, model_dict
        from teddy.tokenizer.gene_tokenizer import GeneTokenizer

        model_dir = context.artifacts["model_dir"]
        arch = get_architecture(model_dir)
        self.config = model_dict[arch]["config_cls"].from_pretrained(model_dir)
        self.model = model_dict[arch]["model_cls"].from_pretrained(model_dir, config=self.config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()            # float32 throughout (70M is small; keeps upstream parity)
        self.tokenizer = GeneTokenizer.from_pretrained(model_dir)   # vocab.txt from the checkpoint dir
        self.vocab = self.tokenizer.vocab
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.add_cls = bool(getattr(self.config, "add_cls", False))
        self.d_model = int(getattr(self.config, "d_model", 0))
        self.max_pos = int(getattr(self.config, "max_position_embeddings", 2048))
        self._forward_params = set(inspect.signature(self.model.forward).parameters.keys())
        with open(f"{pkg_parent}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json") as f:
            self.medians = json.load(f)
        print(f"TEDDY-G loaded on {self.device}, arch={arch}, d_model={self.d_model}, vocab={len(self.vocab)}")

    # ---- upstream-equivalent preprocessing + tokenization ----
    def _tokenize(self, X, gene_names, max_seq_len):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(gene_names):
            raise ValueError(f"adata_sparsematrix shape {X.shape} does not match {len(gene_names)} genes in adata_var")
        # 1. normalise each cell to 10,000 total counts
        tot = X.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        X = X / tot * 1e4
        # 2. divide by TEDDY gene medians (genes without a median keep factor 1.0); no log1p
        X = X * np.array([1.0 / self.medians[g] if g in self.medians else 1.0 for g in gene_names],
                         dtype=np.float32)[None, :]
        # 3. keep vocabulary genes only (upstream drops OOV genes instead of using <unk>)
        keep = [i for i, g in enumerate(gene_names) if g in self.vocab]
        if not keep:
            raise ValueError("No gene IDs in adata_var match the TEDDY vocabulary (expects Ensembl IDs, e.g. ENSG00000000003)")
        if len(keep) / len(gene_names) < MIN_VOCAB_FRACTION:
            raise ValueError(f"Only {len(keep)}/{len(gene_names)} genes are in the TEDDY vocabulary (<10%)")
        token_array = torch.tensor([self.vocab[gene_names[i]] for i in keep], dtype=torch.long)
        X_t = torch.from_numpy(np.ascontiguousarray(X[:, keep], dtype=np.float32))
        # 4. top-K by value, highest first (order carries the rank via position embeddings)
        k = min(max_seq_len - (1 if self.add_cls else 0), X_t.shape[1])
        top_vals, top_idx = torch.topk(X_t, k=k, largest=True, sorted=True)
        gene_ids = token_array[top_idx].clone()
        # 5. drop zero-expression genes: pad + mask (HF convention 1 = attend; model inverts it)
        expressed = top_vals > 0
        gene_ids[~expressed] = self.pad_id
        return gene_ids, expressed.long()

    def _embed(self, gene_ids, attention_mask):
        out = []
        for s in range(0, gene_ids.shape[0], MAX_CELLS_PER_FORWARD):
            ids = gene_ids[s:s + MAX_CELLS_PER_FORWARD].to(self.device)
            am = attention_mask[s:s + MAX_CELLS_PER_FORWARD].to(self.device)
            kw = {k: v for k, v in {"gene_ids": ids, "attention_mask": am}.items() if k in self._forward_params}
            with torch.no_grad():
                o = self.model(**kw)
            emb = o.get("cell_emb") if isinstance(o, dict) else None   # explicit None checks, never `or` on tensors
            if emb is None and isinstance(o, dict):
                emb = o.get("all_embs")
            if emb is None:
                raise RuntimeError(f"Unexpected TEDDY forward output keys: {list(o.keys()) if isinstance(o, dict) else type(o)}")
            if emb.dim() == 3:                       # token-level fallback -> masked mean
                m = am.unsqueeze(-1).to(emb.dtype)
                emb = (emb * m).sum(1) / m.sum(1).clamp(min=1.0)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict(self, context, model_input, params=None):
        params = params or {}
        max_seq_len = int(params.get("max_seq_len", 2048))
        if not 1 <= max_seq_len <= self.max_pos:
            raise ValueError(f"max_seq_len must be in [1, {self.max_pos}], got {max_seq_len}")
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        rows = []
        for _, row in model_input.iterrows():
            var_df = pd.read_json(io.StringIO(row["adata_var"]), orient="split", dtype=False)
            genes = [str(g) for g in (var_df["index"] if "index" in var_df.columns else var_df.index)]
            obs_df = pd.read_json(io.StringIO(row["adata_obs"]), orient="split", dtype=False)
            X = np.array([np.asarray(r, dtype=np.float32) for r in row["adata_sparsematrix"]])
            if len(obs_df) not in (0, X.shape[0]):
                raise ValueError(f"adata_obs has {len(obs_df)} rows but adata_sparsematrix has {X.shape[0]} cells")
            cell_ids = ([str(c) for c in obs_df["cell_id"]] if "cell_id" in obs_df.columns
                        else [str(i) for i in range(X.shape[0])])
            gene_ids, am = self._tokenize(X, genes, max_seq_len)
            emb = self._embed(gene_ids, am)
            rows += [{"cell_id": c, "embedding": e.tolist()} for c, e in zip(cell_ids, emb)]
        return pd.DataFrame(rows, columns=["cell_id", "embedding"])


mlflow.models.set_model(TEDDYEmbedder())
'''
WRAPPER_PATH = f"{CODE_DIR}/teddy_wrapper.py"
with open(WRAPPER_PATH, "w") as f:
    f.write(WRAPPER_SRC)
shutil.copy2(WRAPPER_PATH, f"{BASE_DIR}/teddy_wrapper.py")   # keep a governed copy next to the manifest
size_mb = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(CODE_DIR) for x in fs) / 1e6
print(f"Code bundle staged at {CODE_DIR} ({size_mb:.1f} MB); wrapper at {WRAPPER_PATH}")

# COMMAND ----------

# DBTITLE 1,Dry-load wrapper and build signature
# Dry-load the wrapper exactly as MLflow will (import the file, call load_context with artifact paths),
# then build a signature from a REAL-gene input example (Ensembl IDs read from vocab.txt).
import sys, json, importlib.util, types
import numpy as np, pandas as pd
from mlflow.models import infer_signature
from mlflow.types import DataType

for _k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.") or k == "teddy_wrapper"]:
    del sys.modules[_k]
spec = importlib.util.spec_from_file_location("teddy_wrapper", WRAPPER_PATH)
teddy_wrapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(teddy_wrapper)

ARTIFACTS = {"model_dir": MODEL_DIR, "teddy_pkg_parent": CODE_DIR}
wrapper = teddy_wrapper.TEDDYEmbedder()
wrapper.load_context(types.SimpleNamespace(artifacts=ARTIFACTS))

# Vocabulary smoke test
vocab_tokens = [t.strip() for t in open(f"{MODEL_DIR}/vocab.txt") if t.strip()]
ens = [t for t in vocab_tokens if t.startswith("ENSG")]
assert len(vocab_tokens) == 43808 and len(ens) == 25424 and "ENSG00000000003" in ens, (len(vocab_tokens), len(ens))

# Input example: 5 cells x 100 real vocabulary genes, Poisson counts
real_genes = ens[:100]
n_cells = 5
expr = np.random.default_rng(42).poisson(2.0, size=(n_cells, len(real_genes))).astype(np.float32)
obs_df = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_cells)]})
var_df = pd.DataFrame({"index": real_genes})
input_example = pd.DataFrame({
    "adata_sparsematrix": [expr.tolist()],
    "adata_obs": [obs_df.to_json(orient="split")],
    "adata_var": [var_df.to_json(orient="split")],
})

default_params = {"max_seq_len": "2048"}   # string default; the wrapper casts to int
out = wrapper.predict(None, input_example, params=default_params)
E = np.array(out["embedding"].tolist())
print("output:", out.shape, "embedding dim:", E.shape[1], "norms:", np.round(np.linalg.norm(E, axis=1), 3))
assert E.shape == (n_cells, wrapper.d_model) == (5, 512)
assert np.all(np.isfinite(E)) and np.all(np.linalg.norm(E, axis=1) > 0) and E.std(axis=0).mean() > 0

signature = infer_signature(input_example, out, params=default_params)
assert {p.name: p.dtype for p in signature.params.params} == {"max_seq_len": DataType.string}
assert set(signature.inputs.input_names()) == {"adata_sparsematrix", "adata_obs", "adata_var"}
assert all(c.required for c in signature.inputs.inputs)
print(signature)

# COMMAND ----------

# DBTITLE 1,Upstream-parity test (wrapper vs TEDDY's own preprocessing/tokenization)
# The wrapper reimplements TEDDY preprocessing + tokenization; prove it matches upstream code on real cells
# from the upstream sample data. Pass = same gene set per cell and cosine >= 0.9999.
import types, torch, anndata as ad
from scipy.sparse import csr_matrix

for _k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.")]:
    del sys.modules[_k]
sys.path.insert(0, SNAPSHOT_DIR)   # full snapshot (includes gene-mapping data used by preprocess.py imports)
from teddy.data_processing.preprocessing.preprocess import normalize_data_inplace, scale_columns_by_median_dict
from teddy.data_processing.tokenization.tokenization import _check_genes_in_tokenizer, _build_batch_tensors
from teddy.tokenizer.gene_tokenizer import GeneTokenizer

src = ad.read_h5ad(f"{SNAPSHOT_DIR}/data/sample_data.h5ad")
src = src.raw.to_adata() if src.raw is not None else src
cells = src[:5].copy()
counts = cells.X.toarray() if hasattr(cells.X, "toarray") else np.asarray(cells.X)
genes = list(cells.var_names)
print("sample cells:", cells.shape, "first genes:", genes[:3], "integer counts:", bool(np.allclose(counts, np.round(counts))))

# 1. upstream preprocessing (pretraining settings: normalized_total=10000, median scaling, no log1p)
layer = csr_matrix(counts.astype(np.float32))
normalize_data_inplace(layer, 10000)
scale_columns_by_median_dict(layer, cells, f"{SNAPSHOT_DIR}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json", "index")

# 2. upstream tokenization (vocab genes only, top-K=2048, add_cls False; zeros dropped below)
tok = GeneTokenizer.from_pretrained(MODEL_DIR)
in_vocab, coding_genes, _ = _check_genes_in_tokenizer(cells, "index", tok)
token_array = torch.tensor(tok.encode(coding_genes.tolist(), add_special_tokens=False))
args = types.SimpleNamespace(max_seq_len=2048, add_cls=False, random_genes=False)
up_ids, up_vals, _, _ = _build_batch_tensors(torch.tensor(layer.toarray()[:, in_vocab], dtype=torch.float), token_array, args)

# 3. wrapper on the same raw counts
request = pd.DataFrame({
    "adata_sparsematrix": [counts.tolist()],
    "adata_obs": [pd.DataFrame({"cell_id": list(cells.obs_names)}).to_json(orient="split")],
    "adata_var": [pd.DataFrame({"index": genes}).to_json(orient="split")],
})
E_w = np.array(wrapper.predict(None, request, params=default_params)["embedding"].tolist())
w_ids, w_mask = wrapper._tokenize(counts.astype(np.float32), genes, 2048)

# 4. same model on upstream tokens; compare per cell
parity = []
for i in range(len(cells)):
    ids = up_ids[i][up_vals[i] != 0].unsqueeze(0).to(wrapper.device)
    up_row, w_row = ids[0].cpu(), w_ids[i][w_mask[i].bool()].cpu()
    same_genes = set(w_row.tolist()) == set(up_row.tolist())
    moved = int((w_row != up_row).sum()) if len(w_row) == len(up_row) else None
    with torch.no_grad():
        ref = wrapper.model(gene_ids=ids, attention_mask=torch.ones_like(ids))["cell_emb"][0].float().cpu().numpy()
    cos = float(ref @ E_w[i] / (np.linalg.norm(ref) * np.linalg.norm(E_w[i])))
    parity.append((i, len(up_row), len(w_row), same_genes, moved, cos))
    print(f"cell {i}: tokens upstream={len(up_row)} wrapper={len(w_row)} same genes={same_genes} "
          f"positions reordered={moved} cosine={cos:.9f}")
assert all(p[3] and p[5] >= 0.9999 for p in parity), "wrapper does not reproduce upstream TEDDY"
sys.path.remove(SNAPSHOT_DIR)
print("Upstream parity: PASS")

# COMMAND ----------

# DBTITLE 1,Log and register model to Unity Catalog
# File-based logging (python_model=<path>) + code bundle; weights packaged as an artifact so the
# serving container needs no network access. Dependencies pinned from TEDDY's pyproject.toml.
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
PIP_REQUIREMENTS = [
    "mlflow==3.13.0",
    "torch==2.4.1",            # pyproject: ^2.3.0; pinned to a cu121 build for broad serving-driver compatibility
    "transformers==4.41.0",    # EXACT (5.x breaks TeddyGModel: all_tied_weights_keys)
    "tokenizers==0.19.1",
    "safetensors==0.4.3",
    "numpy==1.26.4",
    "pandas==2.2.2",
]

with mlflow.start_run(run_name="teddy_70m_register") as run:
    mlflow.log_params({"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "variant": VARIANT, "d_model": wrapper.d_model})
    mlflow.set_tags({"license_model_card": manifest["license"]["model_card"],
                     "license_python_package": manifest["license"]["python_package_pyproject"],
                     "paper": "arXiv:2503.03485", "source_url": f"https://huggingface.co/{HF_REPO}"})
    mlflow.log_artifact(f"{BASE_DIR}/provenance_manifest.yaml")
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=WRAPPER_PATH,                       # file path, not an instance
        code_paths=[CODE_DIR],
        artifacts=ARTIFACTS,                              # model_dir (weights+vocab) and teddy_pkg_parent (code)
        signature=signature,
        input_example=(input_example, default_params),
        pip_requirements=PIP_REQUIREMENTS,
        registered_model_name=UC_MODEL_NAME,
    )

MODEL_VERSION = model_info.registered_model_version
client = MlflowClient()
for k, v in {"hf_revision": HF_REVISION, "variant": VARIANT, "license_model_card": manifest["license"]["model_card"],
             "license_python_package": manifest["license"]["python_package_pyproject"],
             "weights_sha256": manifest["weights"]["sha256"], "upstream_parity": "pass_cos>=0.9999_5_cells",
             "manifest": f"{BASE_DIR}/provenance_manifest.yaml"}.items():
    client.set_model_version_tag(UC_MODEL_NAME, MODEL_VERSION, k, str(v))
client.update_registered_model(UC_MODEL_NAME, description=(
    "TEDDY-G 70M single-cell embedder (Merck/TEDDY @ " + HF_REVISION[:7] + "). Input: raw counts + Ensembl gene IDs; "
    "output: 512-d cell embedding. Technical validation only; not clinically validated."))
print(f"Registered {UC_MODEL_NAME} version {MODEL_VERSION}  (run {run.info.run_id})")
print("params schema:", mlflow.models.get_model_info(model_info.model_uri).signature.params)

# COMMAND ----------

# DBTITLE 1,Validate the logged model in an isolated environment
# Rebuild the model's pinned environment from scratch (as the serving container will) and score the
# input example. Catches dependency/import errors before spending ~20 min on endpoint provisioning.
mlflow.models.predict(
    model_uri=f"models:/{UC_MODEL_NAME}/{MODEL_VERSION}",
    input_data=input_example,   # signature default params (max_seq_len="2048") apply
    env_manager="uv",
)

# COMMAND ----------

# DBTITLE 1,Deploy GPU Model Serving endpoint (gated by run_go)
# GPU_SMALL is sufficient for the 70M variant (wrapper caps forward batches at 10 cells).
# Inference table + usage tracking via AI Gateway (timestamped prefix avoids clashes with old payload tables).
# Typed SDK enums, not strings; never replaces an existing endpoint.
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType,
    AiGatewayConfig, AiGatewayInferenceTableConfig, AiGatewayUsageTrackingConfig,
    EndpointStateReady, EndpointStateConfigUpdate, ServedModelStateDeployment,
)

w = WorkspaceClient()
RUN_GO = dbutils.widgets.get("run_go") == "true"

if not RUN_GO:
    print('run_go is "false": skipped endpoint deployment. Set the run_go widget to "true" and re-run this cell to run it.')
else:
    try:
        existing = w.serving_endpoints.get(ENDPOINT_NAME)
        print(f"Endpoint {ENDPOINT_NAME} already exists (state={existing.state.ready}); not modifying it.")
    except ResourceDoesNotExist:
        served_entity = ServedEntityInput(
            name=f"teddy-70m-v{MODEL_VERSION}",
            entity_name=UC_MODEL_NAME,
            entity_version=str(MODEL_VERSION),
            workload_type=ServingModelWorkloadType.GPU_SMALL,
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
        ai_gateway = AiGatewayConfig(
            inference_table_config=AiGatewayInferenceTableConfig(
                catalog_name=CATALOG, schema_name=SCHEMA,
                table_name_prefix=f"{ENDPOINT_NAME}_{datetime.datetime.now():%Y%m%d%H%M}",
                enabled=True,
            ),
            usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        )
        w.serving_endpoints.create(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity]),
            ai_gateway=ai_gateway,
        )
        print(f"Creating {ENDPOINT_NAME} ...")

    # Poll up to 30 min; fail fast on DEPLOYMENT_FAILED
    deadline = time.time() + 1800
    while True:
        ep = w.serving_endpoints.get(ENDPOINT_NAME)
        ready, upd = ep.state.ready, ep.state.config_update
        pending = ep.pending_config.served_entities if ep.pending_config else []
        for se in pending or []:
            if se.state and se.state.deployment == ServedModelStateDeployment.DEPLOYMENT_FAILED:
                raise RuntimeError(f"Deployment failed: {se.state.deployment_state_message}. Check service logs for "
                                   f"config_version={ep.pending_config.config_version}.")
        if ready == EndpointStateReady.READY and upd != EndpointStateConfigUpdate.IN_PROGRESS:
            print(f"Endpoint READY: {ENDPOINT_NAME}"); break
        if upd == EndpointStateConfigUpdate.UPDATE_FAILED:
            raise RuntimeError(f"Endpoint config update failed: {ep.state}")
        if time.time() > deadline:
            raise TimeoutError(f"Endpoint not ready after 30 min: {ep.state}")
        print(f"{time.strftime('%H:%M:%S')} ready={ready} config_update={upd}")
        time.sleep(60)

# COMMAND ----------

# DBTITLE 1,Endpoint smoke, negative, and inference-control tests
# Controls must be sent under the REST body "params" key; SDK query(extra_params=...) is silently dropped
# by custom PyFunc endpoints, so post the body directly.
from databricks.sdk.errors import DatabricksError

def invoke(w, endpoint_name, records, params=None):
    body = {"dataframe_records": records}
    if params:
        body["params"] = params
    return w.api_client.do("POST", f"/serving-endpoints/{endpoint_name}/invocations", body=body)

def cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

if not RUN_GO:
    print('run_go is "false": skipped endpoint tests. Set the run_go widget to "true" and re-run this cell to run it.')
else:
    # Real upstream cells (raw counts, Ensembl IDs) restricted to vocabulary genes to keep the payload small
    vgenes = [i for i, g in enumerate(genes) if g in wrapper.vocab]
    rec = {"adata_sparsematrix": counts[:, vgenes].tolist(),
           "adata_obs": pd.DataFrame({"cell_id": list(cells.obs_names)}).to_json(orient="split"),
           "adata_var": pd.DataFrame({"index": [genes[i] for i in vgenes]}).to_json(orient="split")}

    # 1. Smoke test with defaults; compare to the upstream-validated local embeddings (E_w)
    r = invoke(w, ENDPOINT_NAME, [rec])
    E_ep = np.array([p["embedding"] for p in r["predictions"]])
    c_local = [cos(E_ep[i], E_w[i]) for i in range(len(E_ep))]
    print("1. smoke: shape", E_ep.shape, "cos(endpoint, upstream-validated local) min =", round(min(c_local), 6))
    assert E_ep.shape == (5, 512) and min(c_local) > 0.999

    # 2. Inference control reaches predict(): max_seq_len=32 must change the embedding
    r32 = invoke(w, ENDPOINT_NAME, [rec], params={"max_seq_len": "32"})
    c32 = [cos(a["embedding"], b) for a, b in zip(r32["predictions"], E_ep)]
    print("2. params max_seq_len=32: cos vs default =", np.round(c32, 3))
    assert max(c32) < 0.99, "params did not reach predict()"

    # 3. All-OOV genes -> clear error, not an embedding
    bad = dict(rec, adata_var=pd.DataFrame({"index": [f"FAKE{i}" for i in vgenes]}).to_json(orient="split"))
    try:
        invoke(w, ENDPOINT_NAME, [bad]); raise AssertionError("OOV payload returned an embedding")
    except DatabricksError as e:
        print("3. all-OOV rejected:", str(e)[:160])

    # 4. Missing required field -> schema error
    try:
        invoke(w, ENDPOINT_NAME, [{k: v for k, v in rec.items() if k != "adata_obs"}])
        raise AssertionError("missing adata_obs was accepted")
    except DatabricksError as e:
        print("4. missing adata_obs rejected:", str(e)[:160])
    print("Endpoint tests: PASS")