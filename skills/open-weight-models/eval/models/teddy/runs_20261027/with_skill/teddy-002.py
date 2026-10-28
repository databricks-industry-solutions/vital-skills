# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # TEDDY-70M Serving Endpoint and Cell-Type Vector Search
# MAGIC
# MAGIC Downloads TEDDY-G 70M (Merck/TEDDY, pinned HF revision), registers it to Unity Catalog as a file-based PyFunc, builds a reference atlas of cell embeddings, deploys a GPU Model Serving endpoint, and creates a Vector Search index for nearest-neighbor cell-type lookup. Costly steps are gated by the `run_go` widget (saved default `false`).

# COMMAND ----------

# DBTITLE 1,Install pinned dependencies
# transformers pinned EXACTLY to TEDDY's pyproject (5.x breaks TeddyGModel); anndata/scanpy only for the upstream-parity test;
# cellxgene-census provides real, labelled reference cells (must be %pip, not subprocess, on serverless)
%pip install -q "transformers==4.41.0" "anndata>=0.10" "scanpy" "databricks-vectorsearch" "cellxgene-census" "databricks-sdk>=0.71.0"

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration and run_go gate
import os, sys, json

# Costly-step gate: saved default MUST stay "false"
dbutils.widgets.dropdown("run_go", "false", ["false", "true"], "Run costly steps")
RUN_GO = dbutils.widgets.get("run_go") == "true"

CATALOG, SCHEMA = "<catalog>", "skills_owm_202610"
VARIANT         = "70M"
HF_REPO         = "Merck/TEDDY"
HF_REVISION     = "8c880aa67674042ccdc254bccb5de081092a48bf"   # immutable commit SHA
UC_MODEL_NAME   = f"{CATALOG}.{SCHEMA}.owm_teddy_model_002_with_skill_r2"
ENDPOINT_NAME   = "owm_teddy_ep_002_with_skill_r2"
VS_ENDPOINT     = "owm_teddy_vs_002_with_skill_r2"
CELLS_TABLE     = f"{CATALOG}.{SCHEMA}.owm_teddy_cells_002_with_skill_r2"
VS_INDEX        = f"{CATALOG}.{SCHEMA}.owm_teddy_index_002_with_skill_r2"
VOLUME_PATH     = "/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/teddy-eval-2026-10-27/002_with_skill"
SNAP            = f"{VOLUME_PATH}/snapshot"
MODEL_DIR       = f"{SNAP}/teddy/models/teddy_g/{VARIANT}"
TMP_DIR         = "/tmp/teddy_002"   # Serverless: /tmp, not /local_disk0
os.makedirs(VOLUME_PATH, exist_ok=True); os.makedirs(TMP_DIR, exist_ok=True)

print(f"run_go={RUN_GO}\nmodel={UC_MODEL_NAME}\nendpoint={ENDPOINT_NAME}\nvs={VS_ENDPOINT} / {VS_INDEX}\ncells={CELLS_TABLE}\nfiles={VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,Download TEDDY snapshot to UC Volume (idempotent)
import subprocess, shutil, hashlib, datetime

SENTINEL = f"{SNAP}/.snapshot_complete"
if os.path.exists(SENTINEL):
    print("Snapshot already present:", open(SENTINEL).read())
else:
    local_dir = f"{TMP_DIR}/hf_snapshot"
    # Subprocess isolation: TEDDY uses HF Xet storage; disable Xet/hf_transfer in a clean process
    env = {**os.environ, "HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    subprocess.check_call([sys.executable, "-c", f'''
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download
snapshot_download(repo_id="{HF_REPO}", revision="{HF_REVISION}", local_dir="{local_dir}",
                  ignore_patterns=["teddy/models/teddy_g/160M/*", "teddy/models/teddy_g/400M/*", "*.DS_Store", ".cache/*"])
'''], env=env)
    shutil.copytree(local_dir, SNAP, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache"))
    with open(SENTINEL, "w") as f:
        f.write(f"{HF_REPO}@{HF_REVISION} {datetime.datetime.utcnow().isoformat()}Z")
    print("Downloaded to", SNAP)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

print(sorted(os.listdir(MODEL_DIR)))
vocab_lines = [l.strip() for l in open(f"{MODEL_DIR}/vocab.txt") if l.strip()]
n_ens = sum(t.startswith("ENSG") for t in vocab_lines)
print(f"vocab lines={len(vocab_lines)} ensembl={n_ens} has ENSG00000000003={'ENSG00000000003' in vocab_lines}")
print("config:", json.load(open(f"{MODEL_DIR}/config.json")))

# COMMAND ----------

# DBTITLE 1,Provenance manifest
MANIFEST_PATH = f"{VOLUME_PATH}/provenance_manifest.json"
if not os.path.exists(MANIFEST_PATH):
    manifest = {
        "model": {"name": "TEDDY-G", "variant": VARIANT, "upstream_revision": HF_REVISION,
                  "paper": "arXiv:2503.03485", "reviewed_at": str(datetime.date.today())},
        "code": {"source_url": f"https://huggingface.co/{HF_REPO}", "revision": HF_REVISION},
        "weights": {"source_url": f"https://huggingface.co/{HF_REPO}", "revision_or_record": HF_REVISION,
                    "sha256": sha256(f"{MODEL_DIR}/model.safetensors")},
        # Two license declarations upstream; confirm with model owner before redistribution
        "license": {"model_card": "apache-2.0", "python_package_pyproject": "MIT", "databases": "not-applicable"},
        "runtime": {"python": sys.version.split()[0], "accelerator": "GPU (serving: GPU_SMALL)",
                    "network_required_at_runtime": False},
        "artifacts": [{"name": n, "location": f"{MODEL_DIR}/{n}", "sha256": sha256(f"{MODEL_DIR}/{n}")}
                      for n in ["vocab.txt", "config.json"]]
                     + [{"name": "teddy_gene_medians.json",
                         "location": f"{SNAP}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json",
                         "sha256": sha256(f"{SNAP}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json")}],
        "acquired_at": open(SENTINEL).read().split()[-1],
    }
    json.dump(manifest, open(MANIFEST_PATH, "w"), indent=2)
manifest = json.load(open(MANIFEST_PATH))
print(json.dumps(manifest, indent=2)[:1500])

# COMMAND ----------

# DBTITLE 1,Write PyFunc wrapper file (file-based logging)
# The teddy package ships via code_paths (not pip), so the wrapper is logged from a FILE
# (python_model=<path>) to avoid cloudpickle capturing `import teddy` before code_paths are on sys.path.
WRAPPER_SRC = r'''
import sys, os, io, json, inspect
import numpy as np
import pandas as pd
import torch
import mlflow
from mlflow.pyfunc import PythonModel


class TEDDYEmbedder(PythonModel):
    """TEDDY-G per-cell embeddings. Input: raw counts (cells x genes) + Ensembl gene IDs."""
    CHUNK = 8  # cells per forward pass (safe on T4 16 GB at 2048 tokens)

    def load_context(self, context):
        import sys, json, inspect, torch
        for stream in (sys.stdout, sys.stderr):           # serving StreamToLogger lacks isatty()
            if stream is not None and not hasattr(stream, "isatty"):
                stream.isatty = lambda: False
        for _m in list(sys.modules):                       # purge stale teddy modules
            if _m == "teddy" or _m.startswith("teddy."):
                del sys.modules[_m]
        pkg = context.artifacts["teddy_pkg_parent"]
        if pkg in sys.path:
            sys.path.remove(pkg)
        sys.path.insert(0, pkg)

        from teddy.models.model_directory import get_architecture, model_dict
        from teddy.tokenizer.gene_tokenizer import GeneTokenizer

        model_dir = context.artifacts["model_dir"]
        arch = get_architecture(model_dir)
        self.config = model_dict[arch]["config_cls"].from_pretrained(model_dir)
        self.model = model_dict[arch]["model_cls"].from_pretrained(model_dir, config=self.config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()                  # float32 (parity with upstream)
        self.tokenizer = GeneTokenizer.from_pretrained(model_dir)   # vocab.txt from checkpoint dir
        self.vocab = self.tokenizer.vocab
        self.pad_id = int(self.tokenizer.pad_token_id)
        self._forward_params = set(inspect.signature(self.model.forward).parameters.keys())
        self.add_cls = bool(getattr(self.config, "add_cls", False))
        self.d_model = int(getattr(self.config, "d_model", 0))
        with open(f"{pkg}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json") as f:
            self.medians = json.load(f)
        print(f"TEDDY-G loaded on {self.device}, arch={arch}, d_model={self.d_model}")

    # Upstream preprocessing + tokenization (README pretraining settings)
    def _tokenize(self, X, gene_names, max_seq_len):
        X = np.asarray(X, dtype=np.float32)
        tot = X.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        X = X / tot * 1e4                                                     # 1. normalize_total 1e4
        scale = np.array([1.0 / self.medians[g] if (g in self.medians and self.medians[g]) else 1.0
                          for g in gene_names], dtype=np.float32)
        X = X * scale[None, :]                                                # 2. divide by gene median
        keep = [i for i, g in enumerate(gene_names) if g in self.vocab]       # 3. drop OOV genes
        if not keep:
            raise ValueError("No gene IDs in adata_var match the TEDDY vocabulary "
                             "(expects Ensembl IDs such as ENSG00000000003).")
        token_array = torch.tensor([self.vocab[gene_names[i]] for i in keep], dtype=torch.long)
        X_t = torch.from_numpy(np.ascontiguousarray(X[:, keep], dtype=np.float32))
        k = min(max_seq_len - (1 if self.add_cls else 0), X_t.shape[1])
        top_vals, top_idx = torch.topk(X_t, k=k, largest=True, sorted=True)  # 4. top-K highest first
        gene_ids = token_array[top_idx]
        expressed = top_vals > 0                                              # 5. drop zero genes
        gene_ids[~expressed] = self.pad_id
        return gene_ids, expressed.long()

    def _embed(self, gene_ids, attention_mask):
        n_tok = attention_mask.sum(dim=1)
        if bool((n_tok == 0).any()):
            bad = torch.nonzero(n_tok == 0).flatten().tolist()
            raise ValueError(f"Cells {bad} have no expressed vocabulary genes; cannot embed.")
        out = []
        for s in range(0, gene_ids.shape[0], self.CHUNK):
            ids, am = gene_ids[s:s + self.CHUNK], attention_mask[s:s + self.CHUNK]
            L = int(am.sum(dim=1).max())                    # zeros/pads are at the tail -> trim
            kw = {"gene_ids": ids[:, :L].to(self.device), "attention_mask": am[:, :L].to(self.device)}
            kw = {k: v for k, v in kw.items() if k in self._forward_params}
            with torch.no_grad():
                o = self.model(**kw)
            emb = o.get("cell_emb") if isinstance(o, dict) else None      # 70M: (cells, d_model)
            if emb is None and isinstance(o, dict):
                emb = o.get("all_embs")
            if emb is None and isinstance(o, dict):
                emb = o.get("last_hidden_state")
            if emb is None:
                raise RuntimeError(f"Unexpected TEDDY forward output: {type(o)}")
            if emb.dim() == 3:                                # token-level fallback: masked mean
                m = kw["attention_mask"].unsqueeze(-1).to(emb.dtype)
                emb = (emb * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(emb.float().cpu())
        return torch.cat(out, dim=0).numpy()

    @staticmethod
    def _to_matrix(v):
        if isinstance(v, str):
            v = json.loads(v)
        rows = [np.asarray(r, dtype=np.float32).ravel() for r in v]
        return np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)

    def predict(self, context, model_input, params=None):
        params = params or {}
        max_seq_len = int(params.get("max_seq_len", 2048))
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])
        embs = []
        for _, row in model_input.iterrows():
            X = self._to_matrix(row["adata_sparsematrix"])
            var_df = pd.read_json(io.StringIO(row["adata_var"]), orient="split", dtype=False)
            genes = (var_df["index"] if "index" in var_df.columns else var_df.index).astype(str).tolist()
            if X.shape[1] != len(genes):
                raise ValueError(f"adata_sparsematrix has {X.shape[1]} genes but adata_var has {len(genes)}")
            ids, am = self._tokenize(X, genes, max_seq_len)
            embs.extend(self._embed(ids, am).tolist())
        return pd.DataFrame({"embedding": embs})


mlflow.models.set_model(TEDDYEmbedder())
'''
WRAPPER_PATH = f"{TMP_DIR}/teddy_wrapper.py"
with open(WRAPPER_PATH, "w") as f:
    f.write(WRAPPER_SRC)
print("wrote", WRAPPER_PATH)

# COMMAND ----------

# DBTITLE 1,Stage code bundle, dry-load, and build signature
import importlib.util, types, numpy as np, pandas as pd, torch, mlflow
from mlflow.models import infer_signature
from mlflow.types import DataType

# Code bundle = snapshot/teddy WITHOUT weights
CLEAN_CODE_DIR = f"{TMP_DIR}/teddy_code"
shutil.rmtree(CLEAN_CODE_DIR, ignore_errors=True)
shutil.copytree(f"{SNAP}/teddy", f"{CLEAN_CODE_DIR}/teddy",
                ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.ckpt", "*.pt", "__pycache__"))
shutil.rmtree(f"{CLEAN_CODE_DIR}/teddy/models/teddy_g/{VARIANT}", ignore_errors=True)  # weights live in model_dir artifact

# Dry-load the wrapper exactly as MLflow will (from the file)
for _k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.") or k == "teddy_wrapper"]:
    del sys.modules[_k]
spec = importlib.util.spec_from_file_location("teddy_wrapper", WRAPPER_PATH)
tw = importlib.util.module_from_spec(spec); spec.loader.exec_module(tw)
ARTIFACTS = {"model_dir": MODEL_DIR, "teddy_pkg_parent": CLEAN_CODE_DIR}
wrapper = tw.TEDDYEmbedder()
wrapper.load_context(types.SimpleNamespace(artifacts=ARTIFACTS))
print("forward params:", sorted(wrapper._forward_params))

# Runtime diagnostic of forward output keys
with torch.no_grad():
    _ids = torch.tensor([[wrapper.vocab["ENSG00000000003"]]], device=wrapper.device)
    _o = wrapper.model(gene_ids=_ids, attention_mask=torch.ones_like(_ids))   # mask is required (config has no pad_token_id)
print({k: tuple(v.shape) for k, v in _o.items() if hasattr(v, "shape")})

# Input example from REAL vocab Ensembl IDs
real_genes = [t for t in vocab_lines if t.startswith("ENSG")][:100]
n_cells = 5
expr = np.random.default_rng(42).poisson(2.0, size=(n_cells, len(real_genes))).astype(np.float32)
obs_df = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_cells)]})
var_df = pd.DataFrame({"index": real_genes})
input_example = pd.DataFrame({
    "adata_sparsematrix": [expr.tolist()],
    "adata_obs": [obs_df.to_json(orient="split")],
    "adata_var": [var_df.to_json(orient="split")],
})
D_MODEL = int(json.load(open(f"{MODEL_DIR}/config.json"))["d_model"])
default_params = {"max_seq_len": "2048", "pooling": "mean"}   # strings; wrapper casts
out = wrapper.predict(None, input_example, params=default_params)
E = np.array(out["embedding"].tolist())
print("embeddings:", E.shape, "norms:", np.linalg.norm(E, axis=1).round(3), "distinct:", len({tuple(r.round(5)) for r in E}))
assert E.shape == (n_cells, D_MODEL) and np.all(np.linalg.norm(E, axis=1) > 0)

signature = infer_signature(input_example, out, params=default_params)
assert {p.name: p.dtype for p in signature.params.params} == {"max_seq_len": DataType.string, "pooling": DataType.string}
print(signature)

# COMMAND ----------

# DBTITLE 1,Upstream-parity test (wrapper vs TEDDY's own preprocessing/tokenization)
import anndata as ad
from scipy.sparse import csr_matrix

for _k in [k for k in sys.modules if k == "teddy" or k.startswith("teddy.")]:
    del sys.modules[_k]
sys.path.insert(0, SNAP)   # full upstream snapshot (includes data_processing)
from teddy.data_processing.preprocessing.preprocess import normalize_data_inplace, scale_columns_by_median_dict
from teddy.data_processing.tokenization.tokenization import _check_genes_in_tokenizer, _build_batch_tensors
from teddy.tokenizer.gene_tokenizer import GeneTokenizer

src = ad.read_h5ad(f"{SNAP}/data/sample_data.h5ad")
src = src.raw.to_adata() if src.raw is not None else src         # raw integer counts
cells = src[:5].copy()
counts = cells.X.toarray() if hasattr(cells.X, "toarray") else np.asarray(cells.X)
genes = list(cells.var_names)

layer = csr_matrix(counts.astype(np.float32))
normalize_data_inplace(layer, 10000)
scale_columns_by_median_dict(layer, cells, f"{SNAP}/teddy/data_processing/utils/medians/data/teddy_gene_medians.json", "index")

tok = GeneTokenizer.from_pretrained(MODEL_DIR)
in_vocab, coding_genes, ratio = _check_genes_in_tokenizer(cells, "index", tok)
token_array = torch.tensor(tok.encode(coding_genes.tolist(), add_special_tokens=False))
args = types.SimpleNamespace(max_seq_len=2048, add_cls=False, random_genes=False)
up_ids, up_vals, _, _ = _build_batch_tensors(torch.tensor(layer.toarray()[:, in_vocab], dtype=torch.float), token_array, args)

request = pd.DataFrame({
    "adata_sparsematrix": [counts.tolist()],
    "adata_obs": [pd.DataFrame({"cell_id": list(cells.obs_names)}).to_json(orient="split")],
    "adata_var": [pd.DataFrame({"index": genes}).to_json(orient="split")],
})
E_w = np.array(wrapper.predict(None, request, params=default_params)["embedding"].tolist())
w_ids, w_mask = wrapper._tokenize(counts.astype(np.float32), genes, 2048)

print(f"vocab overlap ratio={ratio:.3f}")
for i in range(len(cells)):
    ids = up_ids[i][up_vals[i] != 0].unsqueeze(0).to(wrapper.device)
    up_row, w_row = ids[0].cpu(), w_ids[i][w_mask[i].bool()].cpu()
    same_genes = set(w_row.tolist()) == set(up_row.tolist())
    moved = int((w_row != up_row).sum()) if len(w_row) == len(up_row) else None
    with torch.no_grad():
        ref = wrapper.model(gene_ids=ids, attention_mask=torch.ones_like(ids))["cell_emb"][0].float().cpu().numpy()
    cos = float(ref @ E_w[i] / (np.linalg.norm(ref) * np.linalg.norm(E_w[i])))
    print(f"cell {i}: tokens upstream={len(up_row)} wrapper={len(w_row)} same genes={same_genes} "
          f"reordered={moved} cosine={cos:.9f}")
    assert same_genes and cos >= 0.9999, "wrapper does not reproduce upstream TEDDY on this cell"
sys.path.remove(SNAP)
print("Upstream parity: PASS")

# COMMAND ----------

# DBTITLE 1,Log and register model to Unity Catalog
from mlflow import MlflowClient
mlflow.set_registry_uri("databricks-uc")
user = spark.sql("select current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/owm_teddy_002_with_skill_r2")

PIP_REQS = [
    "torch>=2.3.0",
    "transformers==4.41.0",      # EXACT (pyproject); 5.x breaks TeddyGModel
    "numpy>=1.26.4,<2.0",
    "pandas>=2.2.2,<3.0",
    "safetensors>=0.4.3",
    f"mlflow=={mlflow.__version__}",
]
TAGS = {"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "variant": VARIANT, "paper": "arXiv:2503.03485",
        "license_model_card": "apache-2.0", "license_pyproject": "MIT", "embedding_dim": str(D_MODEL)}

with mlflow.start_run(run_name="teddy_70m_002_with_skill_r2") as run:
    mlflow.set_tags(TAGS)
    mlflow.log_artifact(MANIFEST_PATH, artifact_path="provenance")
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=WRAPPER_PATH,                 # file path, NOT an instance
        code_paths=[CLEAN_CODE_DIR],
        artifacts=ARTIFACTS,
        signature=signature,
        input_example=(input_example, default_params),
        pip_requirements=PIP_REQS,
        registered_model_name=UC_MODEL_NAME,
    )
MODEL_VERSION = info.registered_model_version
client = MlflowClient()
for k, v in TAGS.items():
    client.set_model_version_tag(UC_MODEL_NAME, MODEL_VERSION, k, v)
client.set_registered_model_alias(UC_MODEL_NAME, "champion", MODEL_VERSION)
print(f"Registered {UC_MODEL_NAME} v{MODEL_VERSION} (alias champion)")
print("params schema:", mlflow.models.get_model_info(f"models:/{UC_MODEL_NAME}/{MODEL_VERSION}").signature.params)

# COMMAND ----------

# DBTITLE 1,Validate the registered model loads and matches the dry-loaded wrapper
loaded = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}/{MODEL_VERSION}")
E_reg = np.array(loaded.predict(input_example, params=default_params)["embedding"].tolist())
cos = (E_reg * E).sum(1) / (np.linalg.norm(E_reg, axis=1) * np.linalg.norm(E, axis=1))
print("registered vs dry-load cosine:", cos.round(6))
assert np.all(cos > 0.9999)
# Inference control must change output (non-default max_seq_len)
E_32 = np.array(loaded.predict(input_example, params={"max_seq_len": "32", "pooling": "mean"})["embedding"].tolist())
c32 = (E_32 * E_reg).sum(1) / (np.linalg.norm(E_32, axis=1) * np.linalg.norm(E_reg, axis=1))
print("max_seq_len=32 vs default cosine:", c32.round(3)); assert np.all(c32 < 0.999)
# All-OOV payload must raise
bad = input_example.copy(); bad["adata_var"] = [pd.DataFrame({"index": [f"FAKE{i}" for i in range(len(real_genes))]}).to_json(orient="split")]
try:
    loaded.predict(bad, params=default_params); accepted = True
except ValueError as e:
    accepted = False; print("OOV payload correctly rejected:", str(e)[:120])
assert not accepted, "OOV payload did not raise"
del loaded

# COMMAND ----------

# DBTITLE 1,Build reference atlas from CELLxGENE Census
import cellxgene_census

CENSUS_VERSION = "2024-07-01"     # pinned LTS release
CENSUS_N_CELLS = 2000
SEED = 42
vocab_set = set(wrapper.vocab)

with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
    hs = census["census_data"]["homo_sapiens"]
    obs = hs.obs.read(
        value_filter="is_primary_data == True and assay == \"10x 3' v3\"",
        column_names=["soma_joinid", "cell_type", "tissue_general", "disease", "dataset_id"],
    ).concat().to_pandas()
    print(f"candidate cells: {len(obs):,}")
    sample = obs.sample(n=CENSUS_N_CELLS, random_state=SEED).sort_values("soma_joinid")
    adata = cellxgene_census.get_anndata(
        census, organism="Homo sapiens", X_name="raw",
        obs_coords=sample["soma_joinid"].to_numpy(),
        obs_column_names=["soma_joinid", "cell_type", "tissue_general", "disease", "dataset_id"],
    )

# Census var_names are numeric indices -> use Ensembl feature_id
adata.var_names = adata.var["feature_id"].astype(str).values
adata.var_names_make_unique()
assert adata.var_names[0].startswith("ENSG"), adata.var_names[:3]
mask = np.array([g in vocab_set for g in adata.var_names])
print(f"genes: census={adata.n_vars:,} in TEDDY vocab={mask.sum():,} ({mask.sum()/n_ens:.1%} of vocab Ensembl IDs)")
adata = adata[:, mask].copy()          # keep vocab genes only (payload size)
print(adata)
print("distinct cell types:", adata.obs["cell_type"].nunique())
display(adata.obs["cell_type"].value_counts().head(15).rename_axis("cell_type").reset_index())

# COMMAND ----------

# DBTITLE 1,Embed reference cells with the registered wrapper and write Delta table (additive MERGE)
from delta.tables import DeltaTable
from pyspark.sql import types as T

genes_ref = list(adata.var_names)
var_json = pd.DataFrame({"index": genes_ref}).to_json(orient="split")
X_all = adata.X.tocsr() if hasattr(adata.X, "tocsr") else adata.X
embs, BATCH = [], 64
for s in range(0, adata.n_obs, BATCH):
    Xb = X_all[s:s + BATCH].toarray().astype(np.float32)
    req = pd.DataFrame({"adata_sparsematrix": [Xb.tolist()], "adata_obs": ["{}"], "adata_var": [var_json]})
    embs.extend(wrapper.predict(None, req, params=default_params)["embedding"].tolist())
E_ref = np.asarray(embs, dtype=np.float32)
assert E_ref.shape == (adata.n_obs, D_MODEL) and np.isfinite(E_ref).all()
print("embedded:", E_ref.shape)

meta = adata.obs.reset_index(drop=True)
cells_pdf = pd.DataFrame({
    "cell_id": "census_" + meta["soma_joinid"].astype(str),
    "embedding": [list(map(float, e)) for e in E_ref],
    "cell_type": meta["cell_type"].astype(str),
    "tissue_general": meta["tissue_general"].astype(str),
    "disease": meta["disease"].astype(str),
    "dataset_id": meta["dataset_id"].astype(str),
})
cells_pdf["census_version"] = CENSUS_VERSION
cells_pdf["model_name"] = UC_MODEL_NAME
cells_pdf["model_version"] = str(MODEL_VERSION)
schema = T.StructType([
    T.StructField("cell_id", T.StringType(), False),
    T.StructField("embedding", T.ArrayType(T.FloatType()), False),
    *[T.StructField(c, T.StringType(), True) for c in
      ["cell_type", "tissue_general", "disease", "dataset_id", "census_version", "model_name", "model_version"]],
])
cells_df = spark.createDataFrame(cells_pdf, schema=schema)

# Additive write: create with PK + CDF if new; otherwise check dimension and MERGE on cell_id
if spark.catalog.tableExists(CELLS_TABLE):
    dim = spark.sql(f"SELECT size(embedding) d FROM {CELLS_TABLE} LIMIT 1").first()
    assert dim is None or dim.d == D_MODEL, f"{CELLS_TABLE} has dim {dim.d}, expected {D_MODEL}; ask before replacing"
else:
    spark.sql(f"""
      CREATE TABLE {CELLS_TABLE} (
        cell_id STRING NOT NULL, embedding ARRAY<FLOAT> NOT NULL,
        cell_type STRING, tissue_general STRING, disease STRING, dataset_id STRING,
        census_version STRING, model_name STRING, model_version STRING,
        CONSTRAINT owm_teddy_cells_002_r2_pk PRIMARY KEY (cell_id))
      COMMENT 'TEDDY-G 70M embeddings of CELLxGENE Census {CENSUS_VERSION} reference cells'
      TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
(DeltaTable.forName(spark, CELLS_TABLE).alias("t")
    .merge(cells_df.alias("s"), "t.cell_id = s.cell_id")
    .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())

display(spark.sql(f"""SELECT count(*) n, count(DISTINCT cell_id) n_ids, min(size(embedding)) dmin, max(size(embedding)) dmax,
                     count(DISTINCT cell_type) n_types FROM {CELLS_TABLE}"""))

# COMMAND ----------

# DBTITLE 1,Offline sanity check: leave-one-out kNN cell-type agreement
# Technical sanity only (not scientific validation): do L2 nearest neighbours share the query's cell type
# more often than chance? Uses L2, the same metric the Vector Search index uses.
labels = cells_pdf["cell_type"].to_numpy()
Et = torch.from_numpy(E_ref)
D = torch.cdist(Et, Et); D.fill_diagonal_(float("inf"))
nn_idx = D.topk(10, largest=False).indices.numpy()
known = labels != "unknown"
top1 = (labels[nn_idx[:, 0]] == labels)[known].mean()
maj = np.array([pd.Series(labels[r]).mode()[0] for r in nn_idx])
k10 = (maj == labels)[known].mean()
baseline = pd.Series(labels[known]).value_counts(normalize=True).iloc[0]
print(f"top-1 NN same cell type: {top1:.3f} | 10-NN majority vote: {k10:.3f} | majority-class baseline: {baseline:.3f}")
assert top1 > 2 * baseline, "embeddings do not separate cell types better than chance"

# COMMAND ----------

# DBTITLE 1,Deploy GPU Model Serving endpoint (gated by run_go)
import time
from datetime import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    AiGatewayConfig, AiGatewayInferenceTableConfig, AiGatewayUsageTrackingConfig,
    EndpointCoreConfigInput, EndpointStateConfigUpdate, EndpointStateReady,
    ServedEntityInput, ServedModelStateDeployment, ServingModelWorkloadType,
)
w = WorkspaceClient()

if not RUN_GO:
    print('run_go is "false": skipped endpoint deployment. Set the run_go widget to "true" and re-run this cell to run it.')
else:
    served = ServedEntityInput(
        name=f"teddy-70m-v{MODEL_VERSION}",
        entity_name=UC_MODEL_NAME, entity_version=str(MODEL_VERSION),
        workload_type=ServingModelWorkloadType.GPU_SMALL,   # T4 16 GB; enough for 70M
        workload_size="Small", scale_to_zero_enabled=True,
    )
    try:
        ep = w.serving_endpoints.get(ENDPOINT_NAME)
        served_now = [(e.entity_name, e.entity_version) for e in (ep.config.served_entities if ep.config else [])]
        if (UC_MODEL_NAME, str(MODEL_VERSION)) not in served_now:
            raise RuntimeError(f"Endpoint {ENDPOINT_NAME} already exists serving {served_now}. "
                               "Not modifying it without confirmation.")
        print(f"Endpoint {ENDPOINT_NAME} already serves v{MODEL_VERSION}; reusing.")
    except NotFound:
        # Inference table captures request payloads (gene-expression profiles): review data classification/retention.
        ai_gateway = AiGatewayConfig(
            inference_table_config=AiGatewayInferenceTableConfig(
                catalog_name=CATALOG, schema_name=SCHEMA, enabled=True,
                table_name_prefix=f"{ENDPOINT_NAME}_{datetime.now():%Y%m%d%H%M}"),
            usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        )
        w.serving_endpoints.create(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served]),
            ai_gateway=ai_gateway,
        )
        print(f"Creating endpoint {ENDPOINT_NAME} (GPU_SMALL, scale-to-zero)...")

    t0 = time.time()
    while True:
        ep = w.serving_endpoints.get(ENDPOINT_NAME)
        if ep.state.ready == EndpointStateReady.READY and ep.state.config_update != EndpointStateConfigUpdate.IN_PROGRESS:
            break
        cfg = ep.pending_config or ep.config
        for se in (cfg.served_entities or []) if cfg else []:
            if se.state and se.state.deployment == ServedModelStateDeployment.DEPLOYMENT_FAILED:
                ver = ep.pending_config.config_version if ep.pending_config else 0
                raise RuntimeError(f"Deployment failed: {se.state.deployment_state_message}. Service logs: "
                                   f"GET /api/2.0/serving-endpoints/{ENDPOINT_NAME}/served-entities/{se.name}/logs?config_version={ver}")
        if ep.state.config_update == EndpointStateConfigUpdate.UPDATE_FAILED:
            raise RuntimeError(f"Endpoint config update failed: {ep.state}")
        if time.time() - t0 > 1800:
            raise TimeoutError("Endpoint not ready after 30 min")
        print(f"[{int(time.time()-t0)}s] ready={ep.state.ready} config_update={ep.state.config_update}")
        time.sleep(60)
    print(f"Endpoint {ENDPOINT_NAME} READY after {int(time.time()-t0)}s")

# COMMAND ----------

# DBTITLE 1,Create Vector Search endpoint and Delta Sync index (gated by run_go)
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest, EmbeddingVectorColumn, EndpointStatusState,
    EndpointType, PipelineType, VectorIndexType,
)

if not RUN_GO:
    print('run_go is "false": skipped Vector Search endpoint/index. Set the run_go widget to "true" and re-run this cell to run it.')
else:
    try:
        w.vector_search_endpoints.get_endpoint(VS_ENDPOINT)
        print(f"Using existing Vector Search endpoint {VS_ENDPOINT}")
    except NotFound:
        w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT, endpoint_type=EndpointType.STANDARD)
        print(f"Creating Vector Search endpoint {VS_ENDPOINT} (STANDARD)...")
    t0 = time.time()
    while (st := w.vector_search_endpoints.get_endpoint(VS_ENDPOINT).endpoint_status.state) != EndpointStatusState.ONLINE:
        if st == EndpointStatusState.OFFLINE or time.time() - t0 > 1800:
            raise RuntimeError(f"VS endpoint state {st}")
        print(f"[{int(time.time()-t0)}s] VS endpoint {st}"); time.sleep(30)

    try:
        idx = w.vector_search_indexes.get_index(VS_INDEX)
        dim = idx.delta_sync_index_spec.embedding_vector_columns[0].embedding_dimension
        if dim != D_MODEL:
            raise RuntimeError(f"Index {VS_INDEX} has dim {dim} != {D_MODEL}; not replacing without confirmation.")
        w.vector_search_indexes.sync_index(VS_INDEX)
        print(f"Index {VS_INDEX} exists (dim {dim}); triggered sync.")
    except NotFound:
        w.vector_search_indexes.create_index(
            name=VS_INDEX, endpoint_name=VS_ENDPOINT, primary_key="cell_id",
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=CELLS_TABLE,
                embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=D_MODEL)],
                pipeline_type=PipelineType.TRIGGERED,
                columns_to_sync=["cell_id", "cell_type", "tissue_general", "disease", "dataset_id"],
            ),
        )
        print(f"Creating index {VS_INDEX}...")
    t0 = time.time()
    while not (idx := w.vector_search_indexes.get_index(VS_INDEX)).status.ready:
        if time.time() - t0 > 1800:
            raise TimeoutError(f"Index not ready: {idx.status.message}")
        print(f"[{int(time.time()-t0)}s] {idx.status.message}"); time.sleep(30)
    print(f"Index READY, indexed rows: {idx.status.indexed_row_count}")

# COMMAND ----------

# DBTITLE 1,Endpoint smoke tests and nearest-neighbor cell-type search (gated by run_go)
def invoke(w, endpoint_name, records, params=None):
    """POST with inference controls under the body 'params' key (serving_endpoints.query(extra_params=) is ignored)."""
    body = {"dataframe_records": records}
    if params:
        body["params"] = params
    return w.api_client.do("POST", f"/serving-endpoints/{endpoint_name}/invocations", body=body)

def records_for(X, genes, cell_ids):
    return [{"adata_sparsematrix": np.asarray(X, dtype=np.float32).tolist(),
             "adata_obs": pd.DataFrame({"cell_id": list(cell_ids)}).to_json(orient="split"),
             "adata_var": pd.DataFrame({"index": list(genes)}).to_json(orient="split")}]

def cell_type_search(X, genes, cell_ids, k=10):
    """Embed raw-count cells via the endpoint, then return the k nearest reference cells for each."""
    preds = invoke(w, ENDPOINT_NAME, records_for(X, genes, cell_ids))["predictions"]
    out = []
    for cid, p in zip(cell_ids, preds):
        r = w.vector_search_indexes.query_index(
            index_name=VS_INDEX, columns=["cell_id", "cell_type", "tissue_general"],
            query_vector=p["embedding"], num_results=k)
        for rank, row in enumerate(r.result.data_array):
            out.append({"query": cid, "rank": rank + 1, "cell_id": row[0], "cell_type": row[1],
                        "tissue_general": row[2], "score": float(row[-1])})
    return pd.DataFrame(out)

if not RUN_GO:
    print('run_go is "false": skipped endpoint and search tests. Set the run_go widget to "true" and re-run this cell to run it.')
else:
    # 1. Endpoint reproduces the reference embeddings (real Census cells, vocab genes)
    Xq = X_all[:5].toarray(); qids = cells_pdf["cell_id"][:5].tolist()
    E_ep = np.array([p["embedding"] for p in invoke(w, ENDPOINT_NAME, records_for(Xq, genes_ref, qids))["predictions"]])
    c = (E_ep * E_ref[:5]).sum(1) / (np.linalg.norm(E_ep, axis=1) * np.linalg.norm(E_ref[:5], axis=1))
    print("endpoint vs notebook cosine:", c.round(6)); assert np.all(c > 0.999)
    # 2. Inference control reaches predict()
    E_32 = np.array([p["embedding"] for p in invoke(w, ENDPOINT_NAME, records_for(Xq, genes_ref, qids), {"max_seq_len": "32"})["predictions"]])
    c32 = (E_32 * E_ep).sum(1) / (np.linalg.norm(E_32, axis=1) * np.linalg.norm(E_ep, axis=1))
    print("max_seq_len=32 vs default cosine:", c32.round(3)); assert np.all(c32 < 0.999)
    # 3. Schema enforcement: missing adata_obs is rejected
    r = records_for(Xq, genes_ref, qids); r[0].pop("adata_obs")
    try:
        invoke(w, ENDPOINT_NAME, r); accepted = True
    except Exception as e:
        accepted = False; print("missing field rejected:", str(e)[:100])
    assert not accepted, "missing adata_obs was accepted"
    # 4. Vector Search self-retrieval + monotonic scores
    hits = cell_type_search(Xq, genes_ref, qids)
    for q, g in hits.groupby("query"):
        assert g.iloc[0]["cell_id"] == q, f"self-retrieval failed for {q}"
        assert len(g) == 10 and g["score"].is_monotonic_decreasing
    print("self-retrieval: PASS (top-1 = query cell, 10 results, scores descending)")
    # 5. Demo: annotate unseen cells (TEDDY sample data, prefrontal-cortex inhibitory interneurons)
    keep_g = [i for i, g in enumerate(genes) if g in vocab_set]
    demo = cell_type_search(counts[:, keep_g], [genes[i] for i in keep_g], list(cells.obs_names))
    display(demo.groupby("query")["cell_type"].agg(lambda s: s.value_counts().index[0]).rename("predicted_cell_type")
            .reset_index().assign(true_label=list(cells.obs["Cell.Types"])))
    display(demo.head(20))