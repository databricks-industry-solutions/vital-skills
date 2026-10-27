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
  `install_skill_for_run.py` (skill off/on, hidden codes per installed skill, backups,
  run records with environment, pre-filled resource names and protocol fields),
  `check_outcome.py` (outcome checks on the live deployment before teardown),
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
- `eval/runs_20260924/`: the Sep 24 – Oct 7, 2026 TEDDY evaluation kept as a frozen record
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
  Sep–Oct 2026 files keep the old IDs).

### Fixed
- Inference controls for custom PyFunc endpoints: the guidance to pass them as
  `w.serving_endpoints.query(extra_params=...)` (added 2026-09-08 in `f9cecf5`) was
  wrong — the SDK writes them to a body key `"extra_params"` that the MLflow scoring
  server drops, so `predict()` silently ran with defaults. Measured 2026-10-27 on a
  TEDDY endpoint (identical embeddings vs. changed embeddings under body `"params"`).
  The Sep–Oct 2026 with-skill notebooks followed the old advice and only checked that
  the call succeeded; the TEDDY ground-truth notebook already sent body `"params"`.
  `SKILL.md` (Troubleshooting #13, Best Practices §1, pitfall #3, Guardrail 5),
  `teddy.md` and `scgpt.md` now send controls in the REST body `"params"` field
  (`api_client.do` helper + HTTP JSON), require string defaults in the signature when
  sending strings, and make the smoke test prove a non-default value changes the
  output. scGPT is marked inferred, not live-tested.
- `teddy.md`: the AI Search step no longer hardcodes a Genesis Workbench endpoint
  name; the agent confirms the endpoint name with the user and reuses an existing
  endpoint only if the user confirms.
- `teddy.md`: `vocab.txt` size corrected from ~60 K to 43,808 lines (25,424 Ensembl
  IDs plus special tokens) for 70M at HF revision `8c880aa`.
- `teddy.md`: license note records both declarations (model card Apache-2.0,
  package `pyproject.toml` MIT, no `LICENSE` file) and asks the user to confirm with
  the model owner before redistribution.
- `teddy.md` endpoint-creation example passes `name=` to `EndpointCoreConfigInput`
  (required since databricks-sdk 0.71.0; all four 2026-10-27 runs hit the
  `missing ... 'name'` `TypeError`); `SKILL.md` Troubleshooting #27 notes it.
- `scgpt.md`: wrapper example casts string params (`int(...)`, explicit boolean
  parse), handles `params=None`, and wraps `pd.read_json` input in `io.StringIO`.

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
