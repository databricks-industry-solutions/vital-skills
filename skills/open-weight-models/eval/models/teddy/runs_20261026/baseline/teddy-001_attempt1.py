# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Deploy TEDDY-70M as a Model Serving Endpoint
# MAGIC
# MAGIC This notebook downloads the TEDDY-70M foundation model for single-cell biology (Merck/TEDDY, Apache-2.0, arxiv 2503.03485) from HuggingFace, logs it to MLflow as a custom PyFunc model with the `teddy` package bundled as artifacts, validates the artifact in an isolated environment, registers it in Unity Catalog, and deploys it as a Databricks Model Serving endpoint.

# COMMAND ----------

# DBTITLE 1,Configuration
# ── Configuration ──────────────────────────────────────────────────────
# Unity Catalog target
CATALOG = "<catalog>"
SCHEMA = "skills_owm_202610"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.owm_teddy_model_001_with_skill"

# Serving endpoint
ENDPOINT_NAME = "owm_teddy_ep_001_with_skill"

# UC Volume for downloaded files
VOLUME_PATH = "/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/001_with_skill/"

# HuggingFace repo
HF_REPO_ID = "Merck/TEDDY"
HF_MODEL_SUBDIR = "teddy/models/teddy_g/70M"  # 70M variant

# Package root (the teddy Python package) inside the repo
TEDDY_PACKAGE_SUBDIR = "teddy"

print(f"Registered model : {REGISTERED_MODEL_NAME}")
print(f"Endpoint name    : {ENDPOINT_NAME}")
print(f"Volume path      : {VOLUME_PATH}")
print(f"HF repo          : {HF_REPO_ID}")
print(f"Model variant    : 70M")


# COMMAND ----------

# DBTITLE 1,Install huggingface_hub
# MAGIC %pip install huggingface_hub transformers==4.41.0 torch safetensors -q

# COMMAND ----------

# DBTITLE 1,Restart Python kernel
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Download TEDDY files to UC Volume
# ── Download TEDDY repo files from HuggingFace to UC Volume ────────
# We download the full repo so the teddy package code, model weights,
# tokenizer files, and data utilities are all stored in the volume.

import os
from huggingface_hub import snapshot_download

os.makedirs(VOLUME_PATH, exist_ok=True)

print("Downloading TEDDY repo from HuggingFace...")
local_path = snapshot_download(
    repo_id=HF_REPO_ID,
    repo_type="model",
    local_dir=VOLUME_PATH,
)
print(f"Downloaded to: {local_path}")

# Verify the 70M model files exist
model_dir = os.path.join(VOLUME_PATH, HF_MODEL_SUBDIR)
model_files = os.listdir(model_dir)
print(f"\n70M model directory contents: {model_files}")

# Verify the teddy package exists
teddy_pkg_dir = os.path.join(VOLUME_PATH, TEDDY_PACKAGE_SUBDIR)
print(f"teddy package exists: {os.path.isdir(teddy_pkg_dir)}")


# COMMAND ----------

# DBTITLE 1,Test model loading
# ── Test model loading from the volume ──────────────────────────────
# Verify we can load the model and run a forward pass before packaging it.

import sys
import os
import torch

# Add the volume root to sys.path so `import teddy` works
# (the teddy package lives at VOLUME_PATH/teddy/)
teddy_root = VOLUME_PATH.rstrip("/")
if teddy_root not in sys.path:
    sys.path.insert(0, teddy_root)

from teddy.models.model_directory import get_architecture, model_dict

model_dir = os.path.join(VOLUME_PATH, HF_MODEL_SUBDIR)
arch = get_architecture(model_dir)
print(f"Architecture: {arch}")

config_cls = model_dict[arch]["config_cls"]
model_cls = model_dict[arch]["model_cls"]

config = config_cls.from_pretrained(model_dir)
model = model_cls.from_pretrained(model_dir, config=config)
model.eval()
print(f"Model loaded: {type(model).__name__}")
print(f"  ntoken:       {config.ntoken}")
print(f"  d_model:      {config.d_model}")
print(f"  nlayers:      {config.nlayers}")
print(f"  nheads:       {config.nheads}")
print(f"  max_pos_emb:  {config.max_position_embeddings}")

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"  parameters:   {n_params:,} (~{n_params/1e6:.0f}M)")

# Test forward pass with dummy input
# gene_ids: [batch_size, seq_len] - token ids for gene expression ranks
# We pass explicit annotations (empty) and attention_mask to avoid a
# torch.cat dtype-mismatch bug in the model's forward when annotations is None.
batch_size, seq_len = 2, 512
dummy_gene_ids = torch.randint(2, config.ntoken, (batch_size, seq_len))
empty_annotations = torch.empty((batch_size, 0), dtype=torch.long)
empty_ann_mask = torch.ones((batch_size, 0), dtype=torch.bool)
attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)  # True = valid
with torch.no_grad():
    outputs = model(
        gene_ids=dummy_gene_ids,
        annotations=empty_annotations,
        annotation_attention_mask=empty_ann_mask,
        attention_mask=attention_mask,
    )
print(f"\nForward pass outputs:")
for k, v in outputs.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")


# COMMAND ----------

# DBTITLE 1,Log model with MLflow
# ── Define custom PyFunc model and log with MLflow ──────────────────
# TEDDY is NOT a standard HuggingFace transformers pipeline — it uses
# custom model code (TeddyGModel extends PreTrainedModel) and a custom
# tokenizer. The `teddy` package is not on PyPI, so we bundle it as
# MLflow artifacts and add it to sys.path at load time.

import os
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
import mlflow
from mlflow.models import infer_signature

# Ensure teddy is importable for the logging cell itself
teddy_root = VOLUME_PATH.rstrip("/")
if teddy_root not in sys.path:
    sys.path.insert(0, teddy_root)
from teddy.models.model_directory import get_architecture, model_dict


class TeddyModelWrapper(mlflow.pyfunc.PythonModel):
    """Custom PyFunc wrapper for TEDDY-70M.
    
    Input  : DataFrame with a single column ``gene_ids``; each row is a
             list of integer token IDs representing a cell's gene expression
             profile (rank-based encoding).
    Output : DataFrame with a single column ``cell_emb``; each row is the
             model's 512-dimensional cell embedding as a list of floats.
    """

    def load_context(self, context):
        import sys
        import os
        import torch

        # The teddy package is bundled as artifacts. The artifact path
        # points to the teddy/ directory itself, so we add its PARENT
        # to sys.path so that `import teddy` resolves correctly.
        teddy_pkg_path = context.artifacts["teddy_package"]
        teddy_parent = os.path.dirname(teddy_pkg_path)
        if teddy_parent not in sys.path:
            sys.path.insert(0, teddy_parent)

        from teddy.models.model_directory import get_architecture, model_dict

        model_dir = context.artifacts["model_dir"]
        arch = get_architecture(model_dir)
        config_cls = model_dict[arch]["config_cls"]
        model_cls = model_dict[arch]["model_cls"]

        config = config_cls.from_pretrained(model_dir)
        model = model_cls.from_pretrained(model_dir, config=config)
        model.eval()
        self.model = model
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import torch
        import numpy as np

        # gene_ids column contains lists of token IDs
        gene_ids_list = model_input["gene_ids"].tolist()

        # Pad to same length within the batch
        batch_size = len(gene_ids_list)
        max_len = max(len(ids) for ids in gene_ids_list)
        pad_value = self.model.config.pad_value  # -100 from TeddyGConfig
        padded = []
        attention_mask = []
        for ids in gene_ids_list:
            pad_count = max_len - len(ids)
            padded.append(list(ids) + [pad_value] * pad_count)
            # True = valid position, False = padding (model inverts internally)
            attention_mask.append([True] * len(ids) + [False] * pad_count)

        gene_ids_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.bool, device=self.device)

        # Pass explicit empty annotations and attention_mask to avoid a
        # torch.cat dtype-mismatch bug in the model's forward when annotations is None.
        empty_annotations = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
        empty_ann_mask = torch.ones((batch_size, 0), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            outputs = self.model(
                gene_ids=gene_ids_tensor,
                annotations=empty_annotations,
                annotation_attention_mask=empty_ann_mask,
                attention_mask=attention_mask_tensor,
            )

        cell_emb = outputs["cell_emb"].cpu().numpy()
        return pd.DataFrame({"cell_emb": list(cell_emb)})


# Prepare artifacts — copy to a clean temp directory
artifacts_dir = tempfile.mkdtemp(prefix="teddy_artifacts_")
artifacts = {
    "teddy_package": os.path.join(artifacts_dir, "teddy"),
    "model_dir": os.path.join(artifacts_dir, "model_70M"),
}

# Copy the teddy Python package
shutil.copytree(
    src=os.path.join(VOLUME_PATH, "teddy"),
    dst=artifacts["teddy_package"],
)

# Copy the 70M model files
shutil.copytree(
    src=os.path.join(VOLUME_PATH, HF_MODEL_SUBDIR),
    dst=artifacts["model_dir"],
)

print(f"Artifacts prepared at: {artifacts_dir}")
print(f"  teddy_package: {os.listdir(artifacts['teddy_package'])[:5]}...")
print(f"  model_dir:     {os.listdir(artifacts['model_dir'])}")

# Build a representative input example and signature
# Use small dummy gene_id sequences (token IDs 2..43839, pad=-100)
sample_gene_ids = [
    [5, 10, 15, 20, 25, 30, 35, 40],
    [100, 200, 300, 400, 500, 600, 700, 800],
]
sample_input = pd.DataFrame({"gene_ids": sample_gene_ids})

# Generate the corresponding output for the signature
model_dir_local = os.path.join(VOLUME_PATH, HF_MODEL_SUBDIR)
arch_local = get_architecture(model_dir_local)
config_cls_local = model_dict[arch_local]["config_cls"]
model_cls_local = model_dict[arch_local]["model_cls"]
config_local = config_cls_local.from_pretrained(model_dir_local)
model_local = model_cls_local.from_pretrained(model_dir_local, config=config_local)
model_local.eval()
batch_size_local = len(sample_gene_ids)
seq_len_local = len(sample_gene_ids[0])
with torch.no_grad():
    test_tensor = torch.tensor(sample_gene_ids, dtype=torch.long)
    test_ann = torch.empty((batch_size_local, 0), dtype=torch.long)
    test_ann_mask = torch.ones((batch_size_local, 0), dtype=torch.bool)
    test_attn_mask = torch.ones((batch_size_local, seq_len_local), dtype=torch.bool)
    test_output = model_local(
        gene_ids=test_tensor,
        annotations=test_ann,
        annotation_attention_mask=test_ann_mask,
        attention_mask=test_attn_mask,
    )
sample_output = pd.DataFrame({"cell_emb": list(test_output["cell_emb"].numpy())})

signature = infer_signature(model_input=sample_input, model_output=sample_output)
print(f"\nSignature:\n{signature}")
print(f"Sample input:\n{sample_input}")
print(f"Sample output shape: {sample_output.shape}")

# Log the model with MLflow
mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="teddy-70m-deploy") as run:
    model_info = mlflow.pyfunc.log_model(
        name="teddy_70m",
        python_model=TeddyModelWrapper(),
        artifacts=artifacts,
        input_example=sample_input,
        signature=signature,
        pip_requirements=[
            "torch>=2.0.1",
            "transformers==4.41.0",
            "safetensors>=0.4.0",
            "numpy",
            "pandas",
        ],
    )
    print(f"\nMLflow run ID: {run.info.run_id}")
    print(f"Model URI: {model_info.model_uri}")


# COMMAND ----------

# DBTITLE 1,Validate MLflow artifact
# ── Validate the MLflow artifact in an isolated environment ────────
# This is the critical serving-readiness gate: it installs pip_requirements
# from scratch in a fresh virtualenv (via uv) and runs a prediction.
# A clean notebook run proves nothing — only this check proves the
# serving container will be able to load and run the model.

%pip install uv -q

import mlflow

pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
if pyfunc_model.metadata.signature is None:
    raise ValueError("The logged model has no signature; fix the artifact before registration.")

representative_input = pyfunc_model.input_example
if representative_input is None:
    raise ValueError("The logged model has no input example; fix the artifact before registration.")

print("Running isolated validation (this may take a few minutes)...")
mlflow.models.predict(
    model_uri=model_info.model_uri,
    input_data=representative_input,
    env_manager="uv",
)
print("✓ Validation passed — the artifact loads and predicts in a fresh environment.")


# COMMAND ----------

# DBTITLE 1,Register in Unity Catalog
# ── Register the validated model in Unity Catalog ──────────────────

import mlflow

mlflow.set_registry_uri("databricks-uc")

registered_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=REGISTERED_MODEL_NAME,
    await_registration_for=300,
)
model_version = registered_version.version
print(f"Registered: {REGISTERED_MODEL_NAME} version {model_version}")
print(f"Model URI:  models:/{REGISTERED_MODEL_NAME}/{model_version}")


# COMMAND ----------

# DBTITLE 1,Select workload
# ── Select workload type and size ────────────────────────────────────
# TEDDY-70M is a float32 PyTorch model with ~70M parameters (~280 MB on
# disk). The forward pass uses standard nn.TransformerEncoder layers —
# no CUDA-specific ops, no .cuda() calls, no GPU-only dependencies.
# Therefore a CPU workload is appropriate.

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Fetch supported workload configurations
workload_response = w.api_client.do(
    "GET", "/api/2.0/serving-endpoints:workload-configs"
)
supported_workloads = {
    config["workload_type"]: [size["key"] for size in config.get("workload_sizes", [])]
    for config in workload_response.get("workload_configs", [])
}
print("Supported workloads:")
for wtype, sizes in supported_workloads.items():
    print(f"  {wtype}: {sizes}")

# Select CPU workload — the model is ~280 MB on disk and expands to
# ~500 MB in memory with PyTorch overhead. The smallest CPU size that
# fits should suffice; use "Small" for reasonable concurrency.
workload_type = "CPU"
if workload_type not in supported_workloads:
    raise ValueError(f"{workload_type} workload not supported in this workspace")

# Prefer "Small" if available, otherwise first available size
available_sizes = supported_workloads[workload_type]
workload_size = "Small" if "Small" in available_sizes else available_sizes[0]
scale_to_zero_enabled = True

print(f"\nSelected: type={workload_type}, size={workload_size}, scale_to_zero={scale_to_zero_enabled}")


# COMMAND ----------

# DBTITLE 1,Create serving endpoint
# ── Create the serving endpoint ─────────────────────────────────────
# Summary:
#   Endpoint name    : owm_teddy_ep_001_with_skill
#   Registered model : <catalog>.skills_owm_202610.owm_teddy_model_001_with_skill
#   Model version    : (from registration above)
#   Workload type    : CPU
#   Workload size    : Small
#   Scale to zero    : True (cold start on first request after idle)
#
# Endpoint creation typically takes ~15 minutes.

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

served_entity = ServedEntityInput(
    entity_name=REGISTERED_MODEL_NAME,
    entity_version=str(model_version),
    workload_type=ServingModelWorkloadType(workload_type),
    workload_size=workload_size,
    scale_to_zero_enabled=scale_to_zero_enabled,
)
endpoint_config = EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity])

# Check if endpoint already exists
try:
    existing_endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
except NotFound:
    existing_endpoint = None

if existing_endpoint is not None:
    current_entities = list(
        existing_endpoint.config.served_entities
        if existing_endpoint.config and existing_endpoint.config.served_entities
        else []
    )
    current_entity_names = [entity.entity_name for entity in current_entities]
    if len(current_entities) != 1 or current_entity_names[0] != REGISTERED_MODEL_NAME:
        raise ValueError(
            f"Refusing to replace endpoint {ENDPOINT_NAME!r}, which currently serves "
            f"{current_entity_names}. Choose a different name or explicitly confirm replacement."
        )
    print(f"Updating existing endpoint {ENDPOINT_NAME}...")
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME,
        served_entities=[served_entity],
    )
else:
    print(f"Creating endpoint {ENDPOINT_NAME}...")
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=endpoint_config,
    )

print(f"Endpoint {ENDPOINT_NAME} deployment initiated.")


# COMMAND ----------

# DBTITLE 1,Wait for readiness
# ── Wait for endpoint readiness ──────────────────────────────────────

import time
from databricks.sdk.service.serving import EndpointStateConfigUpdate, EndpointStateReady


def wait_for_endpoint_ready(name, timeout_s=2400, poll_s=30):
    deadline = time.time() + timeout_s
    failure_states = {
        EndpointStateConfigUpdate.UPDATE_FAILED,
        EndpointStateConfigUpdate.UPDATE_CANCELED,
    }
    while time.time() < deadline:
        state = w.serving_endpoints.get(name).state
        print(f"  state: ready={state.ready}, config_update={state.config_update}")
        if (
            state.ready == EndpointStateReady.READY
            and state.config_update == EndpointStateConfigUpdate.NOT_UPDATING
        ):
            return
        if state.config_update in failure_states:
            raise RuntimeError(
                f"{name} deployment failed with config update state {state.config_update.value}"
            )
        time.sleep(poll_s)
    raise TimeoutError(f"{name} not ready after {timeout_s}s")


print(f"Waiting for {ENDPOINT_NAME} to become ready...")
wait_for_endpoint_ready(ENDPOINT_NAME)
print(f"✓ Endpoint {ENDPOINT_NAME} is READY.")

# Print endpoint details
endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
for entity in endpoint.config.served_entities:
    print(f"  Served entity: {entity.entity_name} v{entity.entity_version}")
    print(f"  Workload:      {entity.workload_type} / {entity.workload_size}")
    print(f"  Scale to zero: {entity.scale_to_zero_enabled}")
print(f"\nEndpoint is ready to serve at: /serving-endpoints/{ENDPOINT_NAME}/invocations")
