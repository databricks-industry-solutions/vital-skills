# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # TEDDY-G 70M on Databricks Model Serving
# MAGIC
# MAGIC Downloads the TEDDY-G 70M single-cell foundation model ([Merck/TEDDY](https://huggingface.co/Merck/TEDDY), Apache-2.0, [arXiv 2503.03485](https://arxiv.org/abs/2503.03485)) to a UC Volume, wraps it as an MLflow pyfunc cell-embedding model, validates it in an isolated environment, registers it to Unity Catalog, and serves it on a Model Serving endpoint.
# MAGIC
# MAGIC **Endpoint contract:** input = one row per cell with `gene_ids` (TEDDY rank-tokenized gene token IDs, highest-expressed first, as produced by `teddy.data_processing.tokenization.tokenize`). Output = `cell_emb` (first-token embedding, 512-d), `cell_emb_mean` (attention-masked mean over tokens, 512-d), `n_tokens`.

# COMMAND ----------

# DBTITLE 1,Install notebook-only tools
# Notebook-only tools: uv for the isolated serving-env check; anndata/scanpy only to run TEDDY's
# official preprocessing/tokenization on the sample data (they are NOT serving dependencies).
# transformers<5: TEDDY's TeddyGModel was written for transformers 4.41 and never calls post_init(),
# which transformers 5 requires (fails with "no attribute 'all_tied_weights_keys'"). The same 4.x
# version is pinned in the serving requirements below.
%pip install -q uv anndata scanpy "transformers>=4.41,<5"
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration
CATALOG = "<catalog>"
SCHEMA = "skills_owm_202610"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.owm_teddy_model_001_baseline"
ENDPOINT_NAME = "owm_teddy_ep_001_baseline"
VOLUME_DIR = "/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/001_baseline"

HF_REPO = "Merck/TEDDY"
# Pin the exact HF commit so the served artifact is reproducible even if the repo changes.
HF_REVISION = "8c880aa67674042ccdc254bccb5de081092a48bf"
MODEL_SUBDIR = "teddy/models/teddy_g/70M"
MODEL_DIR = f"{VOLUME_DIR}/{MODEL_SUBDIR}"

# COMMAND ----------

# DBTITLE 1,Download TEDDY files to the UC Volume
import os
from huggingface_hub import snapshot_download

os.makedirs(VOLUME_DIR, exist_ok=True)
# Only the 70M weights are needed; skip the 160M/400M checkpoints (~2.2 GB) and macOS junk files.
snapshot_download(
    repo_id=HF_REPO,
    revision=HF_REVISION,
    local_dir=VOLUME_DIR,
    allow_patterns=["teddy/*", "data/*", "tutorials/*", "README.md", "pyproject.toml"],
    ignore_patterns=["teddy/models/teddy_g/160M/*", "teddy/models/teddy_g/400M/*", "*.DS_Store"],
)

weights = f"{MODEL_DIR}/model.safetensors"
assert os.path.getsize(weights) == 284_887_424, "Unexpected 70M weights size"
for f in sorted(os.listdir(MODEL_DIR)):
    print(f"{f:28s} {os.path.getsize(os.path.join(MODEL_DIR, f)):>12,d} bytes")

# COMMAND ----------

# DBTITLE 1,Build a real input example with TEDDY's own pipeline
import glob, shutil, sys
from datasets import load_dataset

sys.path.insert(0, VOLUME_DIR)  # make the downloaded `teddy` package importable in the notebook
from teddy.data_processing.preprocessing.preprocess import preprocess
from teddy.data_processing.tokenization.tokenization import tokenize

# Work on a local copy so preprocessing outputs don't clutter the Volume (fresh dir, nothing deleted).
import tempfile
WORK_DIR = tempfile.mkdtemp(prefix="teddy_work_")
os.makedirs(f"{WORK_DIR}/data")
for f in ["sample_data.h5ad", "sample_data_metadata.json"]:
    shutil.copy(f"{VOLUME_DIR}/data/{f}", f"{WORK_DIR}/data/{f}")

U = f"{VOLUME_DIR}/teddy/data_processing/utils"
# Same settings as the official tutorial (tutorials/model_tutorial.ipynb), with absolute paths.
preprocess(
    data_path=f"{WORK_DIR}/data/sample_data.h5ad",
    metadata_path=f"{WORK_DIR}/data/sample_data_metadata.json",
    hyperparameters={
        "min_gene_counts": None, "remove_assays": [], "max_mitochondrial_prop": None,
        "remove_cell_types": [], "hvg_method": None, "normalized_total": 10000,
        "median_dict": f"{U}/medians/data/teddy_gene_medians.json", "log1p": False,
        "compute_medians": False, "median_column": "index", "reference_id_only": False,
        "load_dir": f"{WORK_DIR}/data", "save_dir": f"{WORK_DIR}/data/processed",
    },
)
tokenize(
    data_path=f"{WORK_DIR}/data/processed/sample_data.h5ad",
    metadata_path=f"{WORK_DIR}/data/processed/sample_data_metadata.json",
    tokenization_args={
        "tokenizer_name_or_path": MODEL_DIR, "gene_id_column": "index", "bio_annotations": True,
        "disease_mapping": f"{U}/bio_annotations/data/mappings/all_filtered_disease_mapping.json",
        "tissue_mapping": f"{U}/bio_annotations/data/mappings/all_filtered_tissue_mapping.json",
        "cell_mapping": f"{U}/bio_annotations/data/mappings/all_filtered_cell_mapping.json",
        "sex_mapping": f"{U}/bio_annotations/data/mappings/all_filtered_sex_mapping.json",
        "max_shard_samples": 500, "max_seq_len": 2048, "pad_length": 2048, "add_cls": False,
        "bins": 0, "continuous_rank": True, "truncation_method": "max",
        "add_disease_annotation": False, "include_zero_genes": False,
        "load_dir": f"{WORK_DIR}/data/processed", "save_dir": f"{WORK_DIR}/data/tokenized",
    },
)

arrow_files = glob.glob(f"{WORK_DIR}/data/tokenized/**/*.arrow", recursive=True)
ds = load_dataset("arrow", data_files={"train": arrow_files})["train"]
print(ds)
lens = [len(g) for g in ds.select(range(min(200, len(ds))))["gene_ids"]]
print(f"cells={len(ds)}  tokens/cell (first 200): min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")

# COMMAND ----------

# DBTITLE 1,Stage minimal TEDDY model code for the artifact
# Package only the Python modules needed to build TeddyGModel. Passing the whole `teddy/` folder as
# code_paths would duplicate the 285 MB weights and ~60 MB of preprocessing data into the artifact.
import tempfile
CODE_STAGE = tempfile.mkdtemp(prefix="teddy_code_")  # fresh dir each run; nothing is deleted
for rel in [
    "teddy/__init__.py",
    "teddy/models/__init__.py",
    "teddy/models/model_directory.py",
    "teddy/models/classification_heads.py",
    "teddy/models/teddy_g/__init__.py",
    "teddy/models/teddy_g/model.py",
]:
    os.makedirs(os.path.dirname(f"{CODE_STAGE}/{rel}"), exist_ok=True)
    shutil.copy(f"{VOLUME_DIR}/{rel}", f"{CODE_STAGE}/{rel}")
TEDDY_CODE_PATH = f"{CODE_STAGE}/teddy"
print(sorted(glob.glob(f"{CODE_STAGE}/**/*.py", recursive=True)))

# COMMAND ----------

# DBTITLE 1,Define the pyfunc cell-embedding wrapper
import json
import mlflow
import numpy as np
import pandas as pd


class TeddyGEmbedder(mlflow.pyfunc.PythonModel):
    """Serves TEDDY-G as a cell-embedding model.

    Input : DataFrame with column `gene_ids` -> list[int] per cell (TEDDY rank tokens, highest
            expressed gene first, as produced by TEDDY's tokenizer; truncated to 2048 tokens).
    Output: `cell_emb` (first-token embedding, the model's native `cell_emb`), `cell_emb_mean`
            (attention-masked mean over real tokens), `n_tokens`.
    """

    BATCH_SIZE = 8  # bounds activation memory: 8 x 2048 x 512 fp32 per layer

    def load_context(self, context):
        import torch
        from teddy.models.model_directory import get_architecture, model_dict

        path = context.artifacts["model_dir"]
        arch = get_architecture(path)
        config = model_dict[arch]["config_cls"].from_pretrained(path)
        model = model_dict[arch]["model_cls"].from_pretrained(path, config=config)
        model.return_cell_embs_first_token = True
        model.return_all_embs = True  # needed for the masked mean
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        with open(f"{path}/added_tokens.json") as f:
            self.pad_id = json.load(f)["<pad>"]
        self.n_token = config.ntoken
        self.max_len = config.max_position_embeddings

    def _validate(self, seqs):
        out = []
        for i, s in enumerate(seqs):
            s = np.asarray(s if s is not None else [], dtype=np.int64)[: self.max_len]
            if s.size == 0:
                raise ValueError(f"row {i}: gene_ids is empty")
            if s.min() < 0 or s.max() >= self.n_token:
                raise ValueError(f"row {i}: token ids must be in [0, {self.n_token})")
            out.append(s)
        return out

    def predict(self, context, model_input, params=None):
        import torch

        seqs = self._validate(list(model_input["gene_ids"]))
        cls_out, mean_out = [], []
        with torch.inference_mode():
            for start in range(0, len(seqs), self.BATCH_SIZE):
                batch = seqs[start : start + self.BATCH_SIZE]
                L = max(len(s) for s in batch)
                ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
                for j, s in enumerate(batch):
                    ids[j, : len(s)] = torch.from_numpy(s)
                mask = torch.zeros_like(ids)
                for j, s in enumerate(batch):
                    mask[j, : len(s)] = 1
                ids, mask = ids.to(self.device), mask.to(self.device)
                out = self.model(gene_ids=ids, attention_mask=mask)
                m = mask.unsqueeze(-1).to(out["all_embs"].dtype)
                mean = (out["all_embs"] * m).sum(1) / m.sum(1)
                cls_out.append(out["cell_emb"].float().cpu().numpy())
                mean_out.append(mean.float().cpu().numpy())
        cls_arr, mean_arr = np.concatenate(cls_out), np.concatenate(mean_out)
        return pd.DataFrame({
            "cell_emb": [r.tolist() for r in cls_arr],
            "cell_emb_mean": [r.tolist() for r in mean_arr],
            "n_tokens": [int(len(s)) for s in seqs],
        })

# COMMAND ----------

# DBTITLE 1,Local sanity check against the raw model
import torch
from types import SimpleNamespace

sys.path.insert(0, CODE_STAGE)
# Two cells with different lengths so the padding/masking path is exercised.
example_ids = [ds[0]["gene_ids"], ds[1]["gene_ids"]]
input_example = pd.DataFrame({"gene_ids": example_ids})
print("example lengths:", [len(x) for x in example_ids])

wrapper = TeddyGEmbedder()
wrapper.load_context(SimpleNamespace(artifacts={"model_dir": MODEL_DIR}))
pred = wrapper.predict(None, input_example)

# Reference: run each cell alone (no padding) through the raw model; batched+padded results must match.
for j, s in enumerate(example_ids):
    ids = torch.tensor([s], device=wrapper.device)
    with torch.inference_mode():
        ref = wrapper.model(gene_ids=ids, attention_mask=torch.ones_like(ids))
    d_cls = np.abs(np.array(pred.cell_emb[j]) - ref["cell_emb"][0].cpu().numpy()).max()
    d_mean = np.abs(np.array(pred.cell_emb_mean[j]) - ref["all_embs"][0].mean(0).cpu().numpy()).max()
    print(f"cell {j}: max|diff| cls={d_cls:.2e} mean={d_mean:.2e}")
    assert d_cls < 1e-3 and d_mean < 1e-3
print("device:", wrapper.device, "| emb dim:", len(pred.cell_emb[0]))
display(pred.assign(cell_emb=pred.cell_emb.map(lambda v: v[:5]), cell_emb_mean=pred.cell_emb_mean.map(lambda v: v[:5])))

# COMMAND ----------

# DBTITLE 1,Log the model to MLflow
import cloudpickle, huggingface_hub, safetensors, tokenizers, transformers
from mlflow.models import ModelSignature
from mlflow.types import DataType
from mlflow.types.schema import Array, ColSpec, Schema

# Explicit signature: list-of-int input per cell, fixed-width float embeddings out.
signature = ModelSignature(
    inputs=Schema([ColSpec(Array(DataType.long), "gene_ids")]),
    outputs=Schema([
        ColSpec(Array(DataType.double), "cell_emb"),
        ColSpec(Array(DataType.double), "cell_emb_mean"),
        ColSpec(DataType.long, "n_tokens"),
    ]),
)

# Exact pins of what was validated above. torch's local suffix (e.g. +cu130) is stripped so the
# serving container resolves the standard PyPI wheel.
# (Module __version__ is used because the runtime ships some packages without dist metadata.)
pip_requirements = [
    f"mlflow=={mlflow.__version__}",
    f"cloudpickle=={cloudpickle.__version__}",
    f"torch=={torch.__version__.split('+')[0]}",
    f"transformers=={transformers.__version__}",
    f"tokenizers=={tokenizers.__version__}",
    f"huggingface-hub=={huggingface_hub.__version__}",
    f"safetensors=={safetensors.__version__}",
    f"numpy=={np.__version__}",
    f"pandas=={pd.__version__}",
]
print(pip_requirements)

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name="owm_teddy_g_70m_001_baseline") as run:
    mlflow.log_params({"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "variant": "TEDDY-G 70M",
                       "license": "apache-2.0", "paper": "arXiv:2503.03485"})
    model_info = mlflow.pyfunc.log_model(
        name="teddy_g_70m",
        python_model=TeddyGEmbedder(),
        artifacts={"model_dir": MODEL_DIR},  # weights + config copied into the artifact
        code_paths=[TEDDY_CODE_PATH],         # upstream TeddyGModel source, unmodified
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
    )
print(model_info.model_uri)

# COMMAND ----------

# DBTITLE 1,Validate in an isolated serving-like environment
# Builds a fresh uv venv from pip_requirements only (like the serving container), loads the artifact
# and predicts on the stored input example. CUDA is hidden to mimic the CPU endpoint.
pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
assert pyfunc_model.metadata.signature is not None, "missing signature"
representative_input = pyfunc_model.input_example
assert representative_input is not None, "missing input example"

mlflow.models.predict(
    model_uri=model_info.model_uri,
    input_data=representative_input,
    env_manager="uv",
    extra_envs={"CUDA_VISIBLE_DEVICES": ""},
)

# COMMAND ----------

# DBTITLE 1,Register the validated artifact in Unity Catalog
# Only runs after the isolated check above passed. Each call creates a NEW version, so the exact
# returned version is carried forward instead of looking up "latest".
mlflow.set_registry_uri("databricks-uc")
registered_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=REGISTERED_MODEL_NAME,
    await_registration_for=300,
)
model_version = registered_version.version
print(f"Registered {REGISTERED_MODEL_NAME} v{model_version} -> models:/{REGISTERED_MODEL_NAME}/{model_version}")

# COMMAND ----------

# DBTITLE 1,Endpoint configuration (CPU, Small, scale-to-zero)
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType

w = WorkspaceClient()
supported = {c["workload_type"]: [s["key"] for s in c.get("workload_sizes", [])]
             for c in w.api_client.do("GET", "/api/2.0/serving-endpoints:workload-configs").get("workload_configs", [])}

# CPU: weights are ~285 MB fp32, so torch + model + activations for an 8-cell x 2048-token batch fit
# well inside CPU's documented 4 GB per concurrency, and the code runs on CPU (isolated check above).
workload_type, workload_size, scale_to_zero_enabled = "CPU", "Small", True
assert workload_size in supported.get(workload_type, []), supported

served_entity = ServedEntityInput(
    entity_name=REGISTERED_MODEL_NAME,
    entity_version=str(model_version),
    workload_type=ServingModelWorkloadType(workload_type),
    workload_size=workload_size,
    scale_to_zero_enabled=scale_to_zero_enabled,
)
endpoint_config = EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity])

try:
    existing_endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
except NotFound:
    existing_endpoint = None
if existing_endpoint is not None:
    names = [e.entity_name for e in (existing_endpoint.config.served_entities or [])]
    if names != [REGISTERED_MODEL_NAME]:
        raise ValueError(f"Refusing to replace {ENDPOINT_NAME!r}, which serves {names}.")
print(f"endpoint={ENDPOINT_NAME} exists={existing_endpoint is not None} model={REGISTERED_MODEL_NAME} v{model_version} "
      f"workload={workload_type}/{workload_size} scale_to_zero={scale_to_zero_enabled}")

# COMMAND ----------

# DBTITLE 1,Create or update the serving endpoint
if existing_endpoint is None:
    w.serving_endpoints.create(name=ENDPOINT_NAME, config=endpoint_config)
else:
    w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=[served_entity])
print(f"Submitted deployment of {ENDPOINT_NAME}; this typically takes ~15 minutes.")

# COMMAND ----------

# DBTITLE 1,Wait for endpoint readiness
import time
from databricks.sdk.service.serving import EndpointStateConfigUpdate, EndpointStateReady


def wait_for_endpoint_ready(name, timeout_s=2400, poll_s=30):
    deadline = time.time() + timeout_s
    failure_states = {EndpointStateConfigUpdate.UPDATE_FAILED, EndpointStateConfigUpdate.UPDATE_CANCELED}
    while time.time() < deadline:
        state = w.serving_endpoints.get(name).state
        if state.ready == EndpointStateReady.READY and state.config_update == EndpointStateConfigUpdate.NOT_UPDATING:
            print(f"{name} is READY")
            return
        if state.config_update in failure_states:
            raise RuntimeError(f"{name} deployment failed: {state.config_update.value}")
        time.sleep(poll_s)
    raise TimeoutError(f"{name} not ready after {timeout_s}s")


wait_for_endpoint_ready(ENDPOINT_NAME)

# COMMAND ----------

# DBTITLE 1,Smoke-test the endpoint
# Notebook credentials work because the endpoint is not route-optimized.
resp = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_records=[{"gene_ids": list(map(int, g))} for g in example_ids],
)
served = pd.DataFrame(resp.predictions)
print("n_tokens:", served.n_tokens.tolist(), "| emb dim:", len(served.cell_emb[0]))
# Compare with the local (GPU) result; small differences are expected from CPU vs GPU kernels.
print("max|diff| vs local:", float(np.abs(np.array(served.cell_emb.tolist()) - np.array(pred.cell_emb.tolist())).max()))