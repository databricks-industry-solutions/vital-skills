# Changelog — open-weight-models skill

All notable changes to the `open-weight-models` skill are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/). The skill's
frontmatter `version` is the source of truth for releases.

## [Unreleased]

### Added
- `references/models/midnight.md` for the `kaiko-ai/midnight` pathology tile
  encoder.
- Evaluation prompt-template library (now
  `eval/previous_runs_2026-10/02_eval_prompt_templates.py`) and
  runtime-environment reference (now `eval/README_environments.md`).
- TEDDY retest harness in the standard skill-eval `eval/` layout: `evalset.json` and
  `expectations.json` (teddy-001, teddy-002, queries identical to 2026-10), new
  `scorers.py` (the 2026-10 checks unchanged, plus separately reported new checks:
  AI Gateway inference table + usage tracking enabled on the endpoint and Unity Catalog
  registration under a three-part name, both read by parsing the notebook's Python
  cells; license + Hugging Face revision tags; checking for existing resources), `score_open_weight_models.py` (standard pair `baseline` vs `with_skill`;
  replication pair `old_prompt_baseline` vs `old_prompt_with_skill`), run protocol
  (`README.md`), `answer_sheet.md`, `old_prompt_baseline.md`,
  `old_prompt_with_skill.md`, and `install_skill_for_run.py` (switches the skill
  off/on, hides codes that show which skill files were read, backs up before replacing
  an existing install, records each run's environment, restores the user's pre-eval
  install at the end), `check_outcome.py` + `fixtures/teddy_request.json` (outcome
  group, measured on the live deployment before teardown: endpoint ready and serving
  the recorded version, one finite 512-dimension embedding per cell, working
  nearest-neighbour search for teddy-002), per-check severity / source / environment
  dependence in `expectations.json`, an optional guideline LLM judge, and a README list
  of known false negatives in the replicated checks. Unit tests in
  `tests/test_install_skill_for_run.py` and `tests/test_eval_scorers.py`.
- `eval/previous_runs_2026-10/`: the 2026-10 TEDDY evaluation kept as a frozen record
  in the same layout, plus the Geneformer oss-005/oss-006 notebooks and the oss-005
  with-skill output from a workspace backup (user name, project folder and catalog
  replaced by placeholders).

### Changed
- Renamed the skill from `oss-models` to `open-weight-models`.
- Corrected Geneformer status to serving tested, with evaluation still in
  progress.
- Scrubbed concrete catalog identifiers from examples.
- Eval material moved out of `tests/` (which now holds only CI unit tests) into
  `eval/`. Task IDs renamed: oss-001 → teddy-001, oss-002 → teddy-002, oss-005 →
  geneformer-001, oss-006 → geneformer-002 (`SKILL.md` mentions updated; recorded
  2026-10 files keep the old IDs).

## [0.0.1] — 2026-10-01

Initial contribution — a Health & Life Sciences (HLS) guide for packaging,
registering, validating, and deploying open-weight scientific/foundation models on
Databricks (acquire weights → MLflow PyFunc → Unity Catalog → Model Serving / Jobs).

### Added
- Guide-style `SKILL.md` with Overview, When to Use, Key Concepts, Decision
  Framework, Best Practices, Troubleshooting, Guardrails, and References.
- Model references for six families (single-cell transcriptomics; protein /
  biomolecular structure):
  - **Geneformer** — two deployment paths (serving exercised on workspace; end-to-end eval validation WIP): V1-10M
    (embedding dim 256) and NVIDIA BioNeMo / TransformerEngine V2-316M
    (embedding dim 1152). The TransformerEngine checkpoints load via pip (no
    Docker container required), keeping the standard PyFunc → Unity Catalog →
    Model Serving flow.
  - **TEDDY** (`Merck/TEDDY`, arXiv:2503.03485) — single-cell scRNA-seq
    embeddings; code and weights Apache-2.0.
  - **scGPT** and **SCimilarity** — expanded references, flagged
    work-in-progress pending further workspace testing.
  - **AlphaFold/OpenFold** and **Boltz** — protein / biomolecular structure
    prediction (Jobs or hybrid deployment bias).
- Shared references: integration contract, model template, evaluation,
  maintenance, and a per-family index.
- Best Practices and Troubleshooting expanded from hands-on workspace-deployment
  lessons (dependency pinning, Serverless `/tmp`, file-based model logging,
  signature-based test payloads, inference-table name conflicts, TransformerEngine
  serving, and more).
- Evaluation harness (`tests/`) — a with-skills vs. no-skills Genie Code
  performance assessment. TEDDY and TEDDY + Vector Search comparisons complete;
  Geneformer ground truth validated with and without the NVIDIA engine; the
  skills-vs-no-skills comparison for Geneformer is still to follow.

### Attribution
- Incorporates @yenlow's TEDDY contribution (originally submitted as PR #7).

### Notes
- `version` is intentionally `0.0.1` — an initial, early-development release;
  scGPT and SCimilarity remain work-in-progress.
