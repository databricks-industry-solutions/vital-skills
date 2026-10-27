# Databricks notebook source
# DBTITLE 1,oss-005: Geneformer V1-10M Deploy
# MAGIC %md
# MAGIC # oss-005: Geneformer V1-10M Deploy
# MAGIC
# MAGIC **Eval protocol**: Each arm uses its own prompt (below). Paste the matching one into a fresh Genie Code chat on the corresponding notebook.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Skill arm prompt (this notebook)
# MAGIC
# MAGIC Paste into a **fresh** Genie Code chat on **this** notebook → save response to `results/with_skill/oss-005.txt`
# MAGIC
# MAGIC > Deploy Geneformer (V1-10M variant) as a Model Serving endpoint on Databricks for single-cell embedding. The model is at https://huggingface.co/ctheodoris/Geneformer. Use catalog `<catalog>`, schema `skills`. Store downloaded artifacts under `/Volumes/<catalog>/skills/test_with/models/geneformer_v1`. Register the model as `<catalog>.skills.geneformer_v1_10m` and name the endpoint `geneformer-v1-10m`. Make the notebook fully re-runnable: on first run, delete any pre-existing endpoint and UC model versions; on subsequent runs, skip cleanup if the endpoint is already in READY state (only tear down if it's in a FAILED state or doesn't exist). If a deployment fails, diagnose the error from endpoint events/logs, fix the root cause, and retry — don't leave the notebook in a broken state. Enable `scale_to_zero_enabled=True` on the served entity to avoid idle GPU charges. Write the code and explain your decisions.
# MAGIC >
# MAGIC > **Before writing any code**, read the project's HLS model-deployment skill and Geneformer model reference using `readAssetById` (type `file`) at these workspace paths:
# MAGIC > - `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/.assistant/skills/oss-models/SKILL.md`
# MAGIC > - `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/.assistant/skills/oss-models/references/models/geneformer.md`
# MAGIC >
# MAGIC > **Treat these skill references as authoritative.** When a skill specifies a concrete implementation pattern (e.g., a particular MLflow logging method, a dependency version, a wrapper structure), use that exact pattern — do not substitute your own approach even if it seems simpler or more familiar. Before writing each code cell, re-check the relevant skill section to ensure your implementation matches. If you deviate from a skill recommendation, state why explicitly in a code comment.
# MAGIC >
# MAGIC > **Important**: Do NOT read, search, or reference any notebook or file whose name starts with `gt_` in this project — those are ground-truth references and reading them would contaminate the evaluation.
# MAGIC >
# MAGIC > At the end, list every skill from the Skill Registry that you loaded or consulted while developing this notebook.
# MAGIC >
# MAGIC > **Development log (required)**: Before the export step, add a markdown cell titled "Development Log" that documents every code fix, debug iteration, and reversal you made during development. For each fix, record: (1) the cell number, (2) the error message, (3) the root cause, and (4) how many attempts it took to resolve. Include a summary table with: total unique bugs encountered, total fix iterations, number of cells requiring fixes, model versions created, and any known limitations.
# MAGIC >
# MAGIC > **Runtime capture (required)**: Before the export step, add a code cell titled "Runtime and Environment Info" that prints: Python version, platform, DBR version (`DATABRICKS_RUNTIME_VERSION`), compute type (serverless vs classic — check if DBR version starts with `client.` or pyspark contains `databricks.connect`), environment (check `DATABRICKS_AI_ENV` + `DATABRICKS_ENV_VERSION` env vars for AI Runtime version), key package versions (databricks-sdk, mlflow, transformers, torch, numpy, pandas, pyspark), and GPU info (device name, VRAM) if available. This is eval metadata for reproducibility — do not skip it.
# MAGIC >
# MAGIC > **Final step**: Export this notebook's source (Databricks `.py` format) to `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/results/with_skill/oss-005.txt` using the Workspace REST API (`GET /api/2.0/workspace/export` with `format=SOURCE`, then base64-decode and write to the output path).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Baseline arm prompt (other notebook)
# MAGIC
# MAGIC Paste into a **fresh** Genie Code chat on **[oss-005_Geneformer_V1_Baseline](#notebook-3383427230010119)** → save to `results/baseline/oss-005.txt`
# MAGIC
# MAGIC > Deploy Geneformer (V1-10M variant) as a Model Serving endpoint on Databricks for single-cell embedding. The model is at https://huggingface.co/ctheodoris/Geneformer. Use catalog `<catalog>`, schema `skills`. Store downloaded artifacts under `/Volumes/<catalog>/skills/test_without/models/geneformer_v1`. Register the model as `<catalog>.skills.geneformer_v1_10m_baseline` and name the endpoint `geneformer-v1-10m-baseline`. Make the notebook fully re-runnable: on first run, delete any pre-existing endpoint and UC model versions; on subsequent runs, skip cleanup if the endpoint is already in READY state (only tear down if it's in a FAILED state or doesn't exist). If a deployment fails, diagnose the error from endpoint events/logs, fix the root cause, and retry — don't leave the notebook in a broken state. Enable `scale_to_zero_enabled=True` on the served entity to avoid idle GPU charges. Write the code and explain your decisions.
# MAGIC >
# MAGIC > **Important**: Do NOT read, search, or reference any notebook or file whose name starts with `gt_` in this project — those are ground-truth references and reading them would contaminate the evaluation.
# MAGIC >
# MAGIC > At the end, list every skill from the Skill Registry that you loaded or consulted while developing this notebook.
# MAGIC >
# MAGIC > **Development log (required)**: Before the export step, add a markdown cell titled "Development Log" that documents every code fix, debug iteration, and reversal you made during development. For each fix, record: (1) the cell number, (2) the error message, (3) the root cause, and (4) how many attempts it took to resolve. Include a summary table with: total unique bugs encountered, total fix iterations, number of cells requiring fixes, model versions created, and any known limitations.
# MAGIC >
# MAGIC > **Runtime capture (required)**: Before the export step, add a code cell titled "Runtime and Environment Info" that prints: Python version, platform, DBR version (`DATABRICKS_RUNTIME_VERSION`), compute type (serverless vs classic — check if DBR version starts with `client.` or pyspark contains `databricks.connect`), environment (check `DATABRICKS_AI_ENV` + `DATABRICKS_ENV_VERSION` env vars for AI Runtime version), key package versions (databricks-sdk, mlflow, transformers, torch, numpy, pandas, pyspark), and GPU info (device name, VRAM) if available. This is eval metadata for reproducibility — do not skip it.
# MAGIC >
# MAGIC > **Final step**: Export this notebook's source (Databricks `.py` format) to `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/results/baseline/oss-005.txt` using the Workspace REST API (`GET /api/2.0/workspace/export` with `format=SOURCE`, then base64-decode and write to the output path).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## What's different between the prompts
# MAGIC
# MAGIC | | Skill arm | Baseline arm |
# MAGIC |---|---|---|
# MAGIC | **Skill reference** | Explicit `readAssetById` paths to `SKILL.md` + `geneformer.md` | None — built-in knowledge only |
# MAGIC | **Volume path** | `test_with/models/geneformer_v1` | `test_without/models/geneformer_v1` |
# MAGIC | **UC model name** | `geneformer_v1_10m` | `geneformer_v1_10m_baseline` |
# MAGIC | **Endpoint name** | `geneformer-v1-10m` | `geneformer-v1-10m-baseline` |
# MAGIC | **Everything else** | Identical | Identical |
# MAGIC
# MAGIC ## What the prompt does NOT prescribe (skill should decide)
# MAGIC
# MAGIC * Compute type (GPU\_SMALL? GPU\_MEDIUM?)
# MAGIC * How to handle HF repo restructure (subfolder selection)
# MAGIC * Pickle token dict loading strategy
# MAGIC * Dependency versions (especially `transformers` pinning)
# MAGIC * Serving contract shape (gene IDs in, embeddings out)
# MAGIC * Whether to enable inference tables / AI Gateway
# MAGIC * Download method, staging path, sentinel idempotency
# MAGIC * Notebook structure (one vs two notebooks, CPU/GPU split)
# MAGIC
# MAGIC > **Scoring rubric**: see `03_eval_rubric_and_compare` Cell 2 for the full sub-check rubric.
# MAGIC
# MAGIC > **Before pasting the prompt:** run Cell 2 below to confirm which arm is active.

# COMMAND ----------

# DBTITLE 1,Check Skill Symlink and SKILL.md Status
import os

# --- Paths ---
WS_HOME = "/Workspace/Users/<workspace-user>"
SYMLINK_PATH = f"{WS_HOME}/.assistant/skills/oss-models"
SOURCE_DIR = f"{WS_HOME}/PROJECTS/hls-skills-tests/<project-folder>/.assistant/skills/oss-models"

# --- Symlink ---
print("=== Symlink ===")
print(f"Path:    {SYMLINK_PATH}")
print(f"Exists:  {os.path.exists(SYMLINK_PATH)}")
print(f"Is link: {os.path.islink(SYMLINK_PATH)}")
if os.path.islink(SYMLINK_PATH):
    print(f"Target:  {os.readlink(SYMLINK_PATH)}")

# --- Source directory ---
print(f"\n=== Source ===")
print(f"Path:   {SOURCE_DIR}")
print(f"Exists: {os.path.exists(SOURCE_DIR)}")

# --- SKILL.md status ---
md  = os.path.exists(os.path.join(SOURCE_DIR, "SKILL.md"))
off = os.path.exists(os.path.join(SOURCE_DIR, "SKILL.md.off"))
print(f"\n=== Registry arm ===")
print(f"SKILL.md:     {'FOUND' if md else 'missing'}")
print(f"SKILL.md.off: {'FOUND' if off else 'missing'}")
status = "ACTIVE (skill arm)" if md else "DISABLED (baseline arm)" if off else "MISSING"
print(f"Status:       {status}")