# Changelog — open-weight-models skill

All notable changes to the `open-weight-models` skill are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/). The skill's
frontmatter `version` is the source of truth for releases.

## [Unreleased]

### Added
- TRIAL, not adopted: `eval/practical_checks.py` — goal-based "practical" checks that
  accept any valid way of reaching each goal (parsed code + recorded outcome), with
  tests in `tests/test_practical_checks.py`, and
  `eval/models/teddy/runs_20261026/PRACTICAL_TRIAL.md` comparing them with the current
  scores. Not wired into `score_open_weight_models.py`; score files and `RESULTS.md`
  unchanged. Revised after review: Xet setting must come before the download (or
  recorded evidence), cross-cell upstream-vs-candidate comparison for fidelity, commit-SHA
  revision and non-empty persisted licence, outcome split by severity, GPU serving as a
  separately reported guideline, and endpoint smoke-test and existing-resource checks.
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
  (Optional in databricks-sdk <=0.70.0, required from 0.71.0, so passing it works on
  both; all four 2026-10-27 runs hit the `missing ... 'name'` `TypeError`);
  `SKILL.md` Troubleshooting #27 notes it. The TEDDY signature example is now
  explicit (`infer_signature(..., params=default_params)` with string defaults) and
  the checklist validates the params schema.
- `scgpt.md`: wrapper example casts every documented string param
  (`filter_gene_by_counts`, `normalize_total`, `subset_hvg`, `binning`, explicit
  boolean parse), handles `params=None`, and wraps `pd.read_json` input in
  `io.StringIO`. Its inference-control test now uses vocabulary gene names and
  `subset_hvg="20"` so the returned gene set must shrink (inferred, not live-tested).

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
