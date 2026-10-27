# TEDDY retest, 2026-10-27 — results (draft for review)

Genie Code with its built-in skills **plus** the open-weight-models skill (`with_skill`)
against Genie Code with its built-in skills **only** (`baseline`), on two tasks:
**teddy-001** (deploy TEDDY-70M as a serving endpoint) and **teddy-002** (deploy it and
build a nearest-neighbour cell search). Raw counts only — no percentages, no single score.

**Words used here.** UC = Unity Catalog (where models are registered). VS = Vector Search
(the index used for nearest-neighbour cell lookup). AI Gateway = the serving endpoint
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

## Severity table — failed checks per run

| Task | Run | Critical failed | Major failed | Minor failed | Outcome passed | Shared input format |
|---|---|---|---|---|---|---|
| teddy-001 | baseline (attempt 2) | 5 | 6 | 2 | no | no |
| teddy-001 | baseline (attempt 1) | 3 | 8 | 2 | no | no |
| teddy-001 | with_skill | 0 | 0 | 0 | yes | yes |
| teddy-002 | baseline | 4 | 5 | 2 | yes | no |
| teddy-002 | with_skill | 0 | 0 | 1 | yes | yes |

Counts cover the replicated, new and outcome checks. The shared-input-format check is a
convention, shown in its own column and not counted as a failure. Several baseline
failures are false negatives of the checks — read the list under "Caveats" before reading
the counts as "baseline broke".

## teddy-001 — deploy

Passed / total sub-checks per group:

| Group | baseline attempt 1 | baseline attempt 2 | with_skill |
|---|---|---|---|
| Replicated Sep–Oct 2026 checks | 21 / 29 | 20 / 29 | 29 / 29 |
| New checks | 2 / 5 | 2 / 5 | 5 / 5 |
| Outcome | 2 / 4 | 3 / 4 | 4 / 4 |
| Convention: shared input format | 0 / 1 | 0 / 1 | 1 / 1 |

Every sub-check that failed in at least one run:

| Sub-check | Severity | attempt 1 | attempt 2 | with_skill |
|---|---|---|---|---|
| `download_staging/uses_tmp` | critical | fail ⚑ | fail ⚑ | pass |
| `download_staging/hf_xet_disabled` | critical | fail ⚑ | fail ⚑ | pass |
| `download_staging/deps_pinned` | critical | pass | fail ⚑ | pass |
| `model_registration/infer_signature` | critical | pass | fail ⚑ | pass |
| `forbidden_patterns/transformers` | critical | pass | fail ⚑ | pass |
| `pyfunc_quality/real_gene_ids` | major | fail | fail ⚑ | pass |
| `endpoint_deployment/ai_gateway_config` | major | fail | fail | pass |
| `smoke_tests/registration_smoke` | major | fail | pass | pass |
| `smoke_tests/endpoint_query` | major | fail | pass | pass |
| `pyfunc_quality/sys_modules_purge` | minor | fail | fail | pass |
| `endpoint_deployment/run_go_gate` | minor | fail | fail | pass |
| `new_checks/ai_gateway/inference_table` | major | fail | fail | pass |
| `new_checks/ai_gateway/usage_tracking` | major | fail | fail | pass |
| `new_checks/uc_provenance/provenance_tags` | major | fail | fail | pass |
| `outcome/returns_valid_embeddings` | critical | fail | pass | pass |
| `outcome/accepts_expression_input` | major | fail | fail | pass |
| `convention/shared_single_cell_format` | (convention) | no | no | yes |

⚑ = a known or likely false negative of the check (see "Caveats").

## teddy-002 — deploy + nearest-neighbour cell search

| Group | baseline | with_skill |
|---|---|---|
| Replicated Sep–Oct 2026 checks | 24 / 31 | 30 / 31 |
| New checks | 3 / 7 | 7 / 7 |
| Outcome | 7 / 7 | 7 / 7 |
| Convention: shared input format | 0 / 1 | 1 / 1 |

| Sub-check | Severity | baseline | with_skill |
|---|---|---|---|
| `download_staging/uses_tmp` | critical | fail ⚑ | pass |
| `download_staging/hf_xet_disabled` | critical | fail ⚑ | pass |
| `download_staging/deps_pinned` | critical | fail ⚑ | pass |
| `model_registration/infer_signature` | critical | fail ⚑ | pass |
| `endpoint_deployment/ai_gateway_config` | major | fail | pass |
| `pyfunc_quality/sys_modules_purge` | minor | fail | pass |
| `endpoint_deployment/run_go_gate` | minor | fail | pass |
| `model_registration/pip_reqs_from_source` | minor | pass | fail ⚑ |
| `new_checks/ai_gateway/inference_table` | major | fail | pass |
| `new_checks/ai_gateway/usage_tracking` | major | fail | pass |
| `new_checks/uc_provenance/provenance_tags` | major | fail | pass |
| `new_checks/existing_resources/table` | major | fail | pass |
| `convention/shared_single_cell_format` | (convention) | no | yes |

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
| Valid embeddings in its own declared format (one finite 512-dimension vector per cell) | **fail** — equal-length batches work; mixed lengths fail ("index out of range": the padding value −100 is used as a token ID) | pass (equal and mixed lengths) | pass (gene IDs + counts) | pass | pass |
| Takes cell expression + gene IDs (not pre-tokenized token IDs) | **fail** (token IDs; tokenization left to the client) | **fail** (same) | pass | pass | pass |
| VS endpoint online | — | — | pass | pass | — |
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
- A temporary 3-line instructions file (179 bytes, fingerprinted in each record) for four
  runs — see "Things that look off" for baseline/teddy-002.
- Notebooks on serverless GPU (A10), serverless environment v6 (the Sep–Oct 2026 runs used
  AI Runtime v5).
- Genie Code memories cleared before each run.
- Each run had its own resources under schema `skills_owm_202610`; all endpoints are now
  deleted.

## Caveats

1. **One sample per arm**, except baseline/teddy-001 (two attempts). Directional, not
   statistical.
2. **baseline/teddy-001 attempt 1: the run guide supplied the wrong prompt** (the planned
   with-skill teddy-001 prompt, with `_001_with_skill` names); wording is otherwise
   identical to the teddy-001 request. The skill was off; only the names differ. Attempt 2
   used the right names and is the primary sample.
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
6. **The checks were written from the skill's own content, so they may favour it.** Known
   false negatives in these runs (marked ⚑ above):
   - `download_staging/uses_tmp`: the baselines used `tempfile.mkdtemp()` (writes under
     `/tmp`, no literal path).
   - `download_staging/hf_xet_disabled`: a TEDDY-specific download workaround from
     `teddy.md`; the baselines downloaded without it.
   - `download_staging/deps_pinned` and `forbidden_patterns/transformers`: attempt 2
     installed `transformers>=4.41,<5` and logged the exact running version
     with the model (`transformers=={transformers.__version__}`; 4.57.6 per the run notes); baseline/teddy-002 re-implemented the
     encoder in plain PyTorch and needs no `transformers` at all.
   - `model_registration/infer_signature`: both remaining baselines wrote an explicit
     `ModelSignature(...)` instead of calling `infer_signature`.
   - `pyfunc_quality/real_gene_ids`: attempt 2 used TEDDY's own tokenization pipeline
     without naming `vocab.txt` (attempt 1's test input really was random token IDs, `torch.randint`).
   - `model_registration/pip_reqs_from_source`: with_skill/teddy-002 cites "TEDDY's
     pyproject" but not the literal `pyproject.toml`.
   - `pyfunc_quality/sys_modules_purge` and `endpoint_deployment/run_go_gate` are
     `teddy.md` / `SKILL.md` conventions (minor); a notebook can work without them.
   Fixed in the new checks during scoring: a config written as
   `AiGatewayConfig(...) if FLAG else None` was not followed, so with_skill/teddy-002
   first failed both `ai_gateway` checks although its endpoint had both enabled.
7. **Outcome group restructured after the runs** to be fair to any declared input format;
   the shared format is now a separate convention check. The earlier layout (teddy.md
   request only) failed every baseline endpoint on format alone.

## Things that look off

- **baseline/teddy-002's record fingerprints the user's full instructions file**
  (11,229 bytes) at setup time, not the 179-byte temporary file of the other four runs.
  If the temporary file was put in place after setup, this is only a stale fingerprint;
  if not, that run saw the full instructions (which ask to delete existing endpoints and
  model versions). Worth confirming before the results are shared.
- The baselines chose CPU serving; the task guidelines say GPU_SMALL. No check scores the
  workload type directly (only the endpoint-settings table above shows it).
