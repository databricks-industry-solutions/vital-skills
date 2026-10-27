# Changelog — open-weight-models skill

All notable changes to the `open-weight-models` skill are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/). The skill's
frontmatter `version` is the source of truth for releases.

## [Unreleased]

### Added
- `references/models/midnight.md` for the `kaiko-ai/midnight` pathology tile
  encoder.
- Evaluation prompt-template library (now
  `eval/runs_20260924/02_eval_prompt_templates.py`) and
  runtime-environment reference (now `eval/README_environments.md`).
- Evaluation harness in the standard skill-eval `eval/` layout, shared across model
  families: `evalset.json` / `expectations.json` (teddy-001, teddy-002: the Sep–Oct 2026
  queries with schema `skills_owm_202610` and one sentence naming the run's resources),
  `scorers.py` (the Sep–Oct 2026 checks unchanged, plus separately reported new checks
  — AI Gateway inference table + usage tracking on one attached config and Unity Catalog
  registration under a three-part name, both read by parsing the notebook's Python
  cells; license + Hugging Face revision tags; checking for existing resources — and an
  outcome group), `score_open_weight_models.py` (standard pair `baseline` vs
  `with_skill`; replication pair `old_prompt_baseline` vs `old_prompt_with_skill`),
  `install_skill_for_run.py` (skill off/on, optional hidden codes, backups,
  run records with environment, pre-filled resource names and protocol fields),
  `check_outcome.py` (outcome checks on the live deployment before teardown, with the
  request built in the endpoint's own declared input format),
  `family_settings.py` (per-family names, prompt rendering, outcome settings), per-check
  severity / source / environment dependence, an optional guideline LLM judge, and a
  README run protocol with known false negatives of the replicated checks.
- `eval/models/teddy/`: TEDDY settings (`family.json`: names per run, embedding
  dimension 512, which tasks have Vector Search), fixed request
  (`fixtures/teddy_request.json`), `answer_sheet.md`, the Sep–Oct 2026 prompts with this
  round's names (`old_prompt_baseline.md`, `old_prompt_with_skill.md`), and the
  2026-10-26 round folder `runs_20261026/` (deviations from plan recorded in its
  README; run records can override names with `resources_created` / `prompt_used`). Unit tests in
  `tests/test_install_skill_for_run.py`, `tests/test_eval_scorers.py` and
  `tests/test_check_outcome.py`.
- `eval/models/teddy/runs_20261026/`: the 2026-10-27 TEDDY runs (five exported notebooks,
  catalog name replaced by `<catalog>`), filled run records, score files and the results
  (`RESULTS.md`); `eval_report.md` points to it. The first baseline/teddy-001 attempt
  is kept as a second sample: the prompt is the standard teddy-001 request, identical word
  for word; only the resource names and download path differ — they carried the
  `_001_with_skill` suffix from the run guide before per-run names were finalised. The
  skill was off, so this is a valid baseline sample.
- Convention check `convention/shared_single_cell_format` (shared `adata_*` input format),
  scored and reported apart from the outcome group; outcome checks
  `returns_valid_embeddings`, `accepts_expression_input`, `nn_returns_cell_type`.
- `eval/runs_20260924/`: the Sep 24 – Oct 7, 2026 TEDDY evaluation kept as a frozen record
  in the same layout, plus the Geneformer oss-005/oss-006 notebooks and the oss-005
  with-skill output from a workspace backup (user name, project folder and catalog
  replaced by placeholders).

### Changed
- Renamed the skill from `oss-models` to `open-weight-models`.
- Corrected Geneformer status to serving tested, with evaluation still in
  progress.
- Scrubbed concrete catalog identifiers from examples.
- Eval: hidden-code markers are off by default (`--hidden-codes` opts in) and no longer
  count as proof of skill use — a planted marker instruction is a prompt-injection
  pattern; skill use is read from the self-report, visible steps and notebook choices.
  Eval copies are recognised by their recorded `SKILL.md` sha256.
- Eval: `check_outcome.py` reads the served model version's signature, sends the fixed
  cells in that declared format (mixed lengths), POSTs inference controls under `params`
  (the SDK's `extra_params` are dropped by the scoring server), asks the index only for
  columns it holds, and records the endpoint's workload type and AI Gateway flags.
  `expectations.json` metadata now asks for REST body `params` instead of `extra_params`.
- Eval: the scoring notebook scores repeat attempts as extra samples, skips discarded
  records and pairs that were not run; the protocol preflight accepts "not available:
  <reason>".
- Eval scorers: no infinite recursion on `x = x + ...`; `a if cond else b` is followed in
  the AI Gateway check; two more known false negatives listed; citation test matches the
  quote in the cited file instead of the exact line number.
- Eval material moved out of `tests/` (which now holds only CI unit tests) into
  `eval/`. Task IDs renamed: oss-001 → teddy-001, oss-002 → teddy-002, oss-005 →
  geneformer-001, oss-006 → geneformer-002 (`SKILL.md` mentions updated; recorded
  Sep–Oct 2026 files keep the old IDs).

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
