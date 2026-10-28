# TEDDY retest, 2026-10-27 — results

Genie Code with its built-in skills **plus** the open-weight-models skill (`with_skill`)
against Genie Code with its built-in skills **only** (`baseline`), on two tasks:
**teddy-001** (deploy TEDDY-70M as a serving endpoint) and **teddy-002** (deploy it and
build a nearest-neighbour cell search). Raw counts only — no percentages, no single score.

**Words used here.** CPU = central processing unit and GPU = graphics processing unit
(the two kinds of serving hardware; `GPU_SMALL` is Model Serving's smallest GPU size, A10 is
the GPU model). SDK = software development kit (here the Databricks Python SDK). REST = the
plain web interface of an endpoint; "REST body" = the request text sent to it. ID =
identifier. UTC = coordinated universal time. UC = Unity Catalog (where models are registered). AI Search = Databricks AI Search,
formerly Vector Search (the index used for nearest-neighbour cell lookup). AI Gateway = the serving endpoint
setting that logs requests to an inference table and tracks usage. HF = Hugging Face (where
TEDDY's weights live). Sub-check = one pass/fail line of a scorer. Severity: **critical** =
the endpoint does not work or gives wrong embeddings; **major** = works but unsafe,
ungoverned or not reproducible; **minor** = robustness or style. "Shared input format" =
the request layout `adata_sparsematrix` + `adata_obs` + `adata_var` that `teddy.md`,
`scgpt.md` and Genesis Workbench all use.

## The runs

| Run | Task | Notebook | Counted as |
|---|---|---|---|
| baseline | teddy-001 | `baseline/teddy-001_attempt2.py` | primary baseline sample |
| baseline | teddy-001 | `baseline/teddy-001_attempt1.py` | second baseline sample (see caveats) |
| baseline | teddy-002 | `baseline/teddy-002.py` | baseline |
| with_skill | teddy-001 | `with_skill/teddy-001.py` | with skill |
| with_skill | teddy-002 | `with_skill/teddy-002.py` | with skill |

A sixth record (`old_prompt_baseline`/teddy-001) is a setup record for a run that never
happened; it is marked discarded ("setup record only; run order changed"). The
replication pair (the Sep–Oct 2026 prompts) was not run.

**How scored:** the scorers in `eval/scorers.py`, run with `mlflow.genai.evaluate`,
`compare_runs.extract_scores` and `compare_runs.main`, following the same steps as
`eval/score_open_weight_models.py` but on a laptop with a local MLflow store (mlflow
3.17.2). Score files: `scores/`. The outcome checks were not re-measured: every endpoint
was deleted after the runs; they come from the run records (see "Outcome").

**The Sep–Oct 2026 checks are flawed, so their corrected version is the result.** Twelve of
the replicated Sep–Oct 2026 sub-checks match literal text from `teddy.md` instead of
judging their goal: a valid alternative fails (`tempfile.mkdtemp()` fails "uses /tmp", an
explicit `ModelSignature(...)` fails "infer_signature"), and a mention in markdown, a
download filter or the quoted task prompt passes. `eval/corrected_checks.py` judges each
of them by its intent (list and rules: `eval/README.md`, "Corrected set"); every count
below uses the corrected results (`scores/<run>_corrected.json`). The unchanged
("verbatim") results stay in `scores/<run>_replicated.json` as a historical record; where
they differ, a cell shows the verbatim result as a ⚑ flag with the reason. The Sep–Oct 2026
runs were re-scored the same way (`../../../runs_20260924/scores_corrected/`, new files; the
recorded scores are kept as reported), so each table gives one line comparing them.

## Severity table — failed checks per run

| Task | Run | Critical failed | Major failed | Minor failed | Outcome passed | Shared input format |
|---|---|---|---|---|---|---|
| teddy-001 | baseline (attempt 2) | 0 | 5 | 3 | no | no |
| teddy-001 | baseline (attempt 1) | 1 | 8 | 2 | no | no |
| teddy-001 | with_skill | 0 | 0 | 1 | yes | yes |
| teddy-002 | baseline | 0 | 5 | 4 | yes | no |
| teddy-002 | with_skill | 0 | 0 | 0 | yes | yes |

Counts cover the corrected replicated checks, the new checks and the outcome checks. The
shared-input-format check is a convention, shown in its own column and not counted as a
failure. Before the correction (verbatim logic) the same table read 5 / 6 / 2, 3 / 8 / 2,
0 / 0 / 0, 4 / 5 / 2 and 0 / 0 / 1: every baseline critical failure but one came from a
flawed check. The one left, baseline attempt 1, is real: its endpoint failed on
mixed-length batches (see "Outcome").

Sep–Oct as originally reported (flawed logic), replicated checks only, critical / major /
minor failed: teddy-001 baseline 4 / 2 / 2, with skill 0 / 0 / 0; teddy-002 baseline
1 / 2 / 1, with skill 0 / 0 / 0. Sep–Oct re-scored with corrected logic: teddy-001 baseline
5 / 3 / 3, with skill 0 / 0 / 1; teddy-002 baseline 3 / 3 / 3, with skill 0 / 0 / 2.

## teddy-001 — deploy

Passed / total sub-checks per group:

| Group | baseline attempt 1 | baseline attempt 2 | with_skill |
|---|---|---|---|
| Replicated Sep–Oct 2026 checks (corrected) | 23 / 29 | 25 / 29 | 28 / 29 |
| New checks | 2 / 5 | 2 / 5 | 5 / 5 |
| Outcome | 2 / 4 | 3 / 4 | 4 / 4 |
| Convention: shared input format | 0 / 1 | 0 / 1 | 1 / 1 |

Sep–Oct as originally reported (flawed logic), replicated checks: baseline 21 / 29, with
skill 29 / 29. Sep–Oct re-scored with corrected logic: baseline 18 / 29, with skill 28 / 29.
(The verbatim logic gave this round 21 / 29, 20 / 29 and 29 / 29.)

Every sub-check that failed in at least one run, or where the verbatim result differs:

| Sub-check | Severity | attempt 1 | attempt 2 | with_skill |
|---|---|---|---|---|
| `download_staging/uses_tmp` | critical | pass (⚑ verbatim: fail — no literal '/tmp'; downloaded to a Unity Catalog Volume, which works on serverless) | pass (⚑ verbatim: fail — same: a Volume) | pass |
| `download_staging/hf_xet_disabled` | critical | pass (⚑ verbatim: fail — workaround not used, but the download completed: files loaded at line 102, endpoint READY serving the registered version) | pass (⚑ verbatim: fail — same: files loaded at line 155, endpoint READY) | pass |
| `download_staging/deps_pinned` | critical | pass | pass (⚑ verbatim: fail — `transformers>=4.41,<5` excludes 5.x; serving pins the installed 4.57.6) | pass |
| `model_registration/infer_signature` | critical | pass | pass (⚑ verbatim: fail — explicit `ModelSignature(...)`, not the word `infer_signature`) | pass |
| `forbidden_patterns/transformers` | critical | pass | pass (⚑ verbatim: fail — the text `transformers>=` is in `>=4.41,<5`, which excludes 5.x) | pass |
| `pyfunc_quality/real_gene_ids` | major | fail | pass (⚑ verbatim: fail — input built with TEDDY's own `tokenize()` on its sample data; no literal `vocab.txt`) | pass |
| `model_registration/pip_reqs_from_source` | minor | pass | fail (⚑ verbatim: pass — matched `pyproject.toml` in a download filter; the serving pins are the installed versions, e.g. transformers 4.57.6 against 4.41.0 in TEDDY's `pyproject.toml`) | pass |
| `endpoint_deployment/ai_gateway_config` | major | fail | fail | pass |
| `smoke_tests/registration_smoke` | major | fail | pass | pass |
| `smoke_tests/endpoint_query` | major | fail | pass | pass |
| `pyfunc_quality/sys_modules_purge` | minor | fail | fail | pass |
| `endpoint_deployment/run_go_gate` | minor | fail | fail | fail (⚑ verbatim: pass — the gate exists, but the `run_go` widget defaults to "true" (line 12), so the deploy runs unless the user turns it off) |
| `new_checks/ai_gateway/inference_table` | major | fail | fail | pass |
| `new_checks/ai_gateway/usage_tracking` | major | fail | fail | pass |
| `new_checks/uc_provenance/provenance_tags` | major | fail | fail | pass |
| `outcome/returns_valid_embeddings` | critical | fail | pass | pass |
| `outcome/accepts_expression_input` | major | fail | fail | pass |
| `convention/shared_single_cell_format` | (convention) | no | no | yes |

Sep–Oct as originally reported (flawed logic), these sub-checks: baseline failed
`uses_tmp`, `hf_xet_disabled`, `deps_pinned`, `sys_modules_purge`, `artifacts_declared`,
`pip_reqs_from_source`, `ai_gateway_config`, `registration_smoke`; with skill none. Sep–Oct
re-scored with corrected logic: baseline also fails `real_gene_ids`, `run_go_gate` and
`forbidden_patterns/transformers` (11 in all); with skill fails `run_go_gate` (its deploy
is behind a `deploy_endpoint` widget whose default is not set in the notebook, so it cannot
be shown to default to off).

⚑ = the verbatim (Sep–Oct 2026) logic gave the other result; the reason says why it was wrong.

## teddy-002 — deploy + nearest-neighbour cell search

| Group | baseline | with_skill |
|---|---|---|
| Replicated Sep–Oct 2026 checks (corrected) | 26 / 31 | 31 / 31 |
| New checks | 3 / 7 | 7 / 7 |
| Outcome | 7 / 7 | 7 / 7 |
| Convention: shared input format | 0 / 1 | 1 / 1 |

Sep–Oct as originally reported (flawed logic), replicated checks: baseline 27 / 31, with
skill 31 / 31 (the second with-skill run, "v2"). Sep–Oct re-scored with corrected logic:
baseline 22 / 31, with skill 29 / 31. (The verbatim logic gave this round 24 / 31 and
30 / 31.)

| Sub-check | Severity | baseline | with_skill |
|---|---|---|---|
| `download_staging/uses_tmp` | critical | pass (⚑ verbatim: fail — `tempfile.mkdtemp()`, under /tmp with no literal path) | pass |
| `download_staging/hf_xet_disabled` | critical | pass (⚑ verbatim: fail — workaround not used, but the download completed: files loaded at line 98, endpoint READY serving the registered version) | pass |
| `download_staging/deps_pinned` | critical | pass (⚑ verbatim: fail — no `transformers` on the model path: the encoder is re-implemented in plain PyTorch) | pass |
| `model_registration/infer_signature` | critical | pass (⚑ verbatim: fail — explicit `ModelSignature(...)`, not the word `infer_signature`) | pass |
| `endpoint_deployment/ai_gateway_config` | major | fail | pass |
| `pyfunc_quality/sys_modules_purge` | minor | fail | pass |
| `endpoint_deployment/run_go_gate` | minor | fail | pass |
| `model_registration/pip_reqs_from_source` | minor | fail (⚑ verbatim: pass — matched `pyproject.toml` in a download filter; the serving pins are the installed versions, not TEDDY's) | pass (⚑ verbatim: fail — literal `pyproject.toml` required; run wrote "pyproject"; its pins agree with TEDDY's `pyproject.toml`) |
| `ai_search/graceful_skip` | minor | fail (⚑ verbatim: pass — matched "TABLE IF NOT EXISTS" in the table's CREATE statement; nothing guards a missing table) | pass |
| `new_checks/ai_gateway/inference_table` | major | fail | pass |
| `new_checks/ai_gateway/usage_tracking` | major | fail | pass |
| `new_checks/uc_provenance/provenance_tags` | major | fail | pass |
| `new_checks/existing_resources/table` | major | fail | pass |
| `convention/shared_single_cell_format` | (convention) | no | yes |

Sep–Oct as originally reported (flawed logic), these sub-checks: baseline failed
`pip_reqs_from_source`, `sdk_enums`, `ai_gateway_config`, `registration_smoke`; with skill
none. Sep–Oct re-scored with corrected logic: baseline fails `hf_xet_disabled`,
`deps_pinned`, `forbidden_patterns/transformers`, `real_gene_ids` (its input example is a
hand-typed list of gene IDs, not derived from the vocabulary or real data),
`ai_gateway_config`, `registration_smoke`, `pip_reqs_from_source`, `run_go_gate`,
`graceful_skip` (9; `sdk_enums` now passes: no workload type was passed as a string); with
skill fails `run_go_gate` and `graceful_skip`.

⚑ = the verbatim (Sep–Oct 2026) logic gave the other result; the reason says why it was wrong.

**Severity labels after the correction:** unchanged for every corrected sub-check. Each
corrected check fails for the same consequence its label describes: a download that does
not complete or lands where serverless cannot write, a model that loads `transformers`
5.x, a model without a signature (Unity Catalog refuses it) or a workload type the SDK
rejects is critical; a made-up input example or an endpoint without an inference table is
major; dependency provenance, the run gate and the missing-table guard are minor.

## Outcome — did what each run built work?

Measured before teardown. The checks that `check_outcome.py` could not measure fairly at
the time (it sent only the `teddy.md` request layout, and its search query asked for
columns the index did not hold) were measured by hand by the orchestrator; each result
carries the provenance "measured by orchestrator 2026-10-27, before teardown" in
`run_records.json`, and the tool's original output is kept there under
`outcome_superseded`.

| Check | b/001 att. 1 | b/001 att. 2 | b/002 | ws/002 | ws/001 |
|---|---|---|---|---|---|
| Endpoint ready | pass | pass | pass | pass | pass |
| Serves the recorded model version | pass | pass | pass | pass | pass |
| Valid embeddings in its own declared format (one finite 512-dimension vector per cell) | **fail** — equal-length batches work; mixed lengths fail ("index out of range": the padding value −100 is used as a token ID) | pass (equal and mixed lengths) | pass (gene IDs + counts; checked by hand on 2 of the 3 fixture cells, not with mixed lengths) | pass | pass |
| Takes cell expression + gene IDs (not pre-tokenized token IDs) | **fail** (token IDs; tokenization left to the client) | **fail** (same) | pass | pass | pass |
| AI Search endpoint online | — | — | pass | pass | — |
| Index ready | — | — | pass (3,538 rows) | pass (1,000 rows) | — |
| Nearest-neighbour query returns a cell type | — | — | pass | pass | — |

**Convention check, reported separately — not a sign that anything is broken:**

| | b/001 att. 1 | b/001 att. 2 | b/002 | ws/002 | ws/001 |
|---|---|---|---|---|---|
| Uses the shared input format | no | no | no | yes | yes |

The baselines each chose a valid input format of their own; only the with-skill runs used
the one shared with `scgpt.md` and Genesis Workbench.

**Endpoint settings** (from the endpoint configs, recorded per run under `endpoint_config`):

| | b/001 att. 1 | b/001 att. 2 | b/002 | ws/002 | ws/001 |
|---|---|---|---|---|---|
| Workload type | CPU | CPU (Small) | CPU (Small) | GPU_SMALL | GPU_SMALL |
| Scale to zero | on | on | on | on | on |
| AI Gateway inference table | off | off | off | on | on |
| AI Gateway usage tracking | off | off | off | on | on |

The AI Gateway rows agree with the new `ai_gateway` checks on the notebook code (fail for
all baselines, pass for both with-skill runs). The task guidelines ask for GPU_SMALL; the
baselines chose CPU (baseline/teddy-002 measured about 15 s per cell on CPU).

## Evidence that the skill was used

Hidden codes are no longer used as proof (see "Caveats"). Evidence is read from the
self-report, the chat's visible steps and skill-specific choices in the notebook:

| | with_skill/teddy-002 | with_skill/teddy-001 |
|---|---|---|
| Self-report | open-weight-models (`SKILL.md`, `teddy.md`) plus built-in model-serving, vector-search, model-serving-ui | open-weight-models (`SKILL.md`, `references/models/teddy.md`) plus built-in model-serving-ui |
| Visible steps | Read Skill File, Fetch Omitted Content, … | Read Skill File (`SKILL.md`, `teddy.md`), … ; declined the marker as possibly injected |
| `teddy.md` choices in the notebook | shared input format, `HF_HUB_DISABLE_XET`, `transformers==4.41.0`, `sys.modules` purge, `run_go` gate, AI Gateway inference table + usage tracking, recorded HF revision | the same set |
| Hidden code in notebook | the `teddy.md` code (the installed copy asked for it); no `SKILL.md` code | none — declined the instruction as possibly injected |

No baseline self-report names open-weight-models (built-in model-serving,
model-serving-ui, vector-search and diagnose-error only), and the Skills panel showed "No
skills found" before every baseline run. None of the five notebooks contains
`HF_HUB_DISABLE_XET`, the shared input format or a `run_go` gate unless the skill was on.

with_skill/teddy-001 also found a problem in the skill: on the live endpoint, SDK
`extra_params` were ignored by the scoring server — inference controls must go under the
REST body key `params`, which contradicts `SKILL.md` troubleshooting #13 and `teddy.md`
(being fixed on a separate branch; `check_outcome.py` now posts `params`). It also noted a
licence mismatch upstream (model card Apache-2.0, `pyproject` MIT).

## Environment

- Genie Code as shipped: built-in skills always on (not visible or switchable). No
  personal or workspace skills (Skills panel: "No skills found" for baselines; only
  open-weight-models for with-skill runs).
- A temporary 3-line instructions file (179 bytes, sha256 `1630f63a…`) in all five runs.
  baseline/teddy-002's record shows the user's full file (11,229 bytes) because it was
  fingerprinted at setup (03:55 UTC), before the file was replaced; the temporary file
  was active during that run — every later record fingerprints it, the notebook was
  created at 05:56 UTC, and Genie Code's pre-run memory-check reply quoted a line found
  only in the temporary file ("check with you before deleting anything"). Recorded under
  `instructions_at_run` in that record; the setup fingerprint is kept unchanged.
- Notebooks on serverless GPU (A10), serverless environment v6 (the Sep–Oct 2026 runs used
  AI Runtime v5).
- Genie Code memories cleared before each run.
- Each run had its own resources under schema `skills_owm_202610`; all endpoints are now
  deleted.

## Caveats

1. **One sample per arm**, except baseline/teddy-001 (two attempts). Directional, not
   statistical.
2. **baseline/teddy-001 attempt 1:** the prompt is the standard teddy-001 request,
   identical word for word; only the resource names and download path differ — they
   carried the `_001_with_skill` suffix from the run guide before per-run names were
   finalised. The skill was off, so this is a valid baseline sample. Attempt 2 used the
   `_001_baseline` names and is the primary sample.
3. **Runs 4 and 5 (with_skill/teddy-002 and with_skill/teddy-001) overlapped** and shared
   one skill install, so no files were replaced under a running chat.
4. **Outcome partly measured by hand before teardown** (see "Outcome"); it cannot be
   re-measured.
5. **Hidden-code marker retired.** The installer used to add to `SKILL.md` and `teddy.md`
   an instruction to write a code into the notebook. with_skill/teddy-002 included the
   `teddy.md` code; with_skill/teddy-001 declined it as "possibly injected" — correctly: an
   instruction planted in a skill file to make the agent emit an unrelated marker is a
   prompt-injection pattern. Markers are now off by default (`--hidden-codes` to opt in,
   with that caveat); skill use is read from the self-report, visible steps and notebook
   choices.
6. **The checks were written from the skill's own content, so they may favour it.** The
   Sep–Oct 2026 sub-checks were literal-text matches; the corrected checks (now the result,
   see "How scored") judge their goal instead, and every ⚑ cell above names what the
   verbatim logic got wrong in that run. Limits of the correction:
   - `hf_xet_disabled` and `infer_signature` may credit a run with recorded evidence
     (download used, endpoint READY serving the registered version). The Sep–Oct 2026 runs
     have no run records, so only their code could be judged: a Sep–Oct baseline whose
     download worked without the workaround still fails.
   - `pip_reqs_from_source` compares serving pins with TEDDY's `pyproject.toml` at commit
     `8c880aa` (read once, read-only; pins in `../family.json` `upstream_dependencies`). A pin
     to the version installed at run time (`transformers=={transformers.__version__}`)
     records the environment, not the source, so it fails this minor check.
   - `pyfunc_quality/sys_modules_purge` is a `teddy.md` / `SKILL.md` convention (minor),
     kept with its Sep–Oct logic; a notebook can work without it.
   Not false negatives: the AI Gateway (inference table + usage tracking) and provenance
   (license + Hugging Face revision tags) checks are valid, recorded criteria. They come
   from the skill (`SKILL.md:325`, `teddy.md:533`) and the answer key, not from the
   prompts — neither task prompt asks for them — so they measure whether Genie Code
   applies these practices unprompted, which is part of what the skill is meant to add.
   Fixed in the new checks during scoring: a config written as
   `AiGatewayConfig(...) if FLAG else None` was not followed, so with_skill/teddy-002
   first failed both `ai_gateway` checks although its endpoint had both enabled.
7. **Outcome group restructured after the runs** to be fair to any declared input format;
   the shared format is now a separate convention check. The earlier layout (teddy.md
   request only) failed every baseline endpoint on format alone.

## Things that look off

- **Resolved:** baseline/teddy-002's setup-time instructions fingerprint (11,229 bytes) is
  stale; the 179-byte temporary file was active during the run (see "Environment").
- The baselines chose CPU serving; the task guidelines say GPU_SMALL. No check scores the
  workload type directly (only the endpoint-settings table above shows it).
