# TEDDY evaluation — all rounds

The detail behind the summary in [`../../eval_report.md`](../../eval_report.md): how each round was run, results per round with the corrected checks, which checks passed only with the skill, and where the skill helps by category. What each check is: [`../../CHECKS.md`](../../CHECKS.md).

## How it was run

One fresh chat per run on serverless GPU notebooks, memories checked before each run, the skill
installed only for the with-skill runs.

**Rounds**, newest first:

| Round | When | What it is | Details |
|---|---|---|---|
| **Round 2 — the latest** | 2026-10-27/28 | a clean rerun of all four runs with the fixed skill (`7ca4eee`) | [details](#round-2) |
| Round 1 | 2026-10-27 | the same tasks, before the skill fixes this report lists | [details](#round-1) |
| Sep–Oct 2026 | 2026-09-24 to 10-07 | the original sprint, on the skill before its rename from `oss-models` | [details](#sep-oct-2026) |

"Latest round" below means round 2. Counted scores of every round, every sample, are in one file,
[`scores.json`](scores.json); the latest round's are also in `eval/baseline_scores.json` and
`eval/with_skill_scores.json`. The round folders themselves (`runs_<date>/`: exported notebooks, run records,
per-group score files) are git-ignored.

## Results by round

Every sub-check judged with the corrected logic (`corrected_checks.py`; what each one is:
[`CHECKS.md`](../../CHECKS.md)). A task passes only if every sub-check passes, so one minor failure
makes it fail. "Skill wins" = the task passes every sub-check with the skill and not without it;
"neither passes every check" = both runs fail at least one sub-check (named in "Failed in both"
when it is the same one). Rounds 1 and 2 add the governance, safety and outcome checks, so they
have more sub-checks than Sep–Oct. Groups that do not apply to a task (search for teddy-001, the
edge-case group) are not counted. Primary samples only.

| Round | Task | Sub-checks | Baseline passed | With skill passed | Passed only with the skill | Passed only without | Failed in both | Task result |
|---|---|---|---|---|---|---|---|---|
| Round 2 (latest) | teddy-001 | 36 | 29 | 36 | 7 | 0 | 0 | skill wins |
| Round 2 (latest) | teddy-002 | 44 | 37 | 44 | 7 | 0 | 0 | skill wins |
| Round 1 | teddy-001 | 36 | 28 | 35 | 7 | 0 | 1 (`run_go_gate`) | neither passes every check |
| Round 1 | teddy-002 | 44 | 35 | 44 | 9 | 0 | 0 | skill wins |
| Sep–Oct | teddy-001 (oss-001) | 27 | 16 | 26 | 10 | 0 | 1 (`run_go_gate`) | neither passes every check |
| Sep–Oct | teddy-002 (oss-002, after patching `teddy.md`) | 30 | 21 | 28 | 7 | 0 | 2 (`run_go_gate`, `graceful_skip`) | neither passes every check |

**Which sub-checks passed only with the skill** (plain words; the check name, in brackets, is
the row in [`CHECKS.md`](../../CHECKS.md)):

| Round, task | Passed only with the skill |
|---|---|
| Latest round, teddy-001 and teddy-002 (7 each, the same 7) | request logging configured (`ai_gateway_config`), inference table on (`inference_table`), usage tracking on (`usage_tracking`), source / revision / licence tags on the model (`provenance_tags`), deploy switch saved off (`run_go_gate`, minor), serving pins from the model's own dependency file (`pip_reqs_from_source`, minor), cached code modules cleaned up on re-run (`sys_modules_purge`, minor) |
| Round 1, teddy-001 (7) | request logging configured, inference table, usage tracking, provenance tags, serving pins from the dependency file, module clean-up — and the endpoint takes a cell's expression, not token IDs (`accepts_expression_input`) |
| Round 1, teddy-002 (9) | the 7 of the latest round, plus a guard if the cell table is missing (`graceful_skip`, minor) and looking before writing to an existing table (`existing_resources/table`) |
| Sep–Oct, teddy-001 (10) | files saved where serverless can write (`uses_tmp`), the Xet download setting handled (`hf_xet_disabled`), `transformers` pinned below 5 (`deps_pinned`), no open-ended `transformers>=` pin (`transformers`), weights and code declared as model artifacts (`artifacts_declared`), real gene IDs in the test input (`real_gene_ids`), registered model called before serving (`registration_smoke`), request logging configured, serving pins from the dependency file, module clean-up |
| Sep–Oct, teddy-002 (7) | the Xet setting, the `transformers` pin, no open-ended pin, real gene IDs, registered model called before serving, request logging configured, serving pins from the dependency file |

No sub-check passed without the skill and failed with it, in any round.

**Sep–Oct as originally reported** (old literal-text logic): oss-001 baseline 19, with skill 27
(of 27); oss-002 baseline 26, with skill 30 (of 30). The corrected logic lowers the baseline to 16
and 21 because some old checks passed on a word, not on what the code does: the gene vocabulary
file was named but the test input was not real gene data; a deploy-switch word appeared but the
deploy calls were not behind it; there was no `transformers>=` text, yet the serving pins allowed
5.x (oss-002 also: the Xet setting named but not set, and no guard on the cell table; one check
went the other way, `sdk_enums`). With the skill, 27 → 26 and 30 → 28: the deploy switch (its
saved default could not be read, or it was missing) and, on oss-002, the cell-table guard. Detail and the
original caveats (the with-skill prompt told Genie Code to read the skill): [Sep–Oct details](#sep-oct-2026).

## Where the skill helps — by category

Both tasks together. "Skill only" = passed with the skill, failed without it.

| Category | Sub-checks | Latest round, baseline → with skill | Skill only, latest round | Skill only, earlier rounds |
|---|---|---|---|---|
| Download and staging | 6 | 6 → 6 | — | Sep–Oct: download location, Xet setting, `transformers` pin |
| Model wrapper | 14 | 12 → 14 | clean-up of cached code modules on re-run (minor) | round 1: same; Sep–Oct: also artifacts declared, real gene IDs in the test input |
| Registration | 10 | 8 → 10 | serving pins taken from the model's dependency file (minor) | same |
| Endpoint deployment | 8 | 4 → 8 | request logging configured; deploy switch saved off (minor) | request logging configured; round 1 teddy-002: deploy switch saved off |
| Tests before and after deploying | 6 | 6 → 6 | — | Sep–Oct: registered model called before serving |
| Known-bad patterns | 10 | 10 → 10 | — | Sep–Oct: no open-ended `transformers>=` pin |
| Search (teddy-002) | 3 | 3 → 3 | — | round 1: guard on a missing cell table (minor) |
| Governance: request logging | 4 | 0 → 4 | inference table, usage tracking | round 1: same |
| Governance: provenance | 4 | 2 → 4 | source, revision and licence tags on the model | round 1: same |
| Safety: look before creating or deleting | 4 | 4 → 4 | — | round 1: no overwrite of an existing table |
| Outcome | 11 | 11 → 11 | — | round 1: endpoint takes a cell's expression, not token IDs |

**Why some earlier "skill only" checks are gone:** none of them was dropped; they are still
scored every round. The runs without the skill now pass them too, in their own way: they saved
the download to a Volume or a temporary folder; the download completed, which the run records
show (Sep–Oct had no run records, so only its code could be judged for the Xet setting); in round 2
they served their own plain-PyTorch copy of TEDDY, which needs no `transformers` pin; they used
real gene data in the test input and called the registered model before serving. **Why they do
better now is not measured:** Genie Code itself may have improved since September, and the
Sep–Oct setup also differed. The skill's lasting help is in **governance** (request logging,
provenance), **deployment safety** (a deploy switch saved off) and **reproducibility** (pins from
the model's own dependency file). Not counted, but also with the skill only: GPU serving and the shared request format.

## How the checks changed

**The scorer improved through the rounds too.** Each round of testing showed checks failing valid
code: some matched literal text from the skill (a different valid way of doing the same thing
failed), others could not follow a path or a copy through the notebook. They were fixed as they
were found: 13 replicated checks are now judged by intent (one added in round 2), and five more
were corrected while scoring round 2. After those five, round 1 and Sep–Oct were re-scored and
none of their results changed.

## Round details

Each round's full notes, folded: click a line to open it. File names such as `baseline/teddy-001.py` or
`run_records.json` point into that round's folder (`runs_<YYYYMMDD>/`), which is git-ignored.

**Words used in the notes**

| Term | Meaning |
|---|---|
| CPU / GPU | Central / graphics processing unit: the two kinds of serving hardware. `GPU_SMALL` is Model Serving's smallest GPU size; A10 is the GPU model |
| SDK | Software development kit (here the Databricks Python SDK) |
| REST, REST body | The plain web interface of an endpoint; the request text sent to it |
| ID, UTC | Identifier; coordinated universal time |
| UC | Unity Catalog, where models are registered |
| AI Search | Databricks AI Search, formerly Vector Search (VS): the index used for nearest-neighbour cell lookup |
| AI Gateway | The serving endpoint setting that logs requests to an inference table and tracks usage |
| HF | Hugging Face, where TEDDY's weights live |
| Sub-check | One pass/fail line of a scorer |
| Severity | **critical** = the endpoint does not work or gives wrong embeddings; **major** = works but unsafe, ungoverned or not reproducible; **minor** = robustness or style |
| Shared input format | The request layout `adata_sparsematrix` + `adata_obs` + `adata_var` that `teddy.md`, `scgpt.md` and Genesis Workbench use |
| `run_go` | The notebook switch that must be on before anything billable is created |

<a id="round-2"></a>
### Round 2 — 2026-10-27/28 (latest)

<details>
<summary>Open the round 2 notes: summary, runs, failed checks per run, deploy switch, outcome, evidence the skill was used, how scored, things that look off</summary>

A clean rerun of all four runs: Genie Code with its built-in skills **plus** the
open-weight-models skill (`with_skill`) against Genie Code with its built-in skills **only**
(`baseline`), on **teddy-001** (deploy TEDDY-70M as a serving endpoint) and **teddy-002**
(deploy it and build a nearest-neighbour cell search). Round 1 ([round 1](#round-1))
is kept as the record from before the skill fixes. Raw counts only — no percentages, no single
score.


#### Summary

- Both with-skill runs: **0 critical / 0 major / 0 minor** failed, outcome checks all passed,
  shared input format used, `run_go` saved off.
- Both baselines: **0 critical / 4 major / 3 minor** failed. Their endpoints (and the teddy-002
  search) **worked**; what they lack is governance and safety: no inference table, no usage
  tracking, no provenance tags, and no deploy switch saved off.
- The skill's gains this round: GPU serving, AI Gateway logging, provenance (pinned revision and
  licences), the shared input format, controls sent in the REST body, and a `run_go` switch
  saved off.

#### The runs

| Run | Task | Notebook | Skill | Serving | Built |
|---|---|---|---|---|---|
| baseline | teddy-001 | `baseline/teddy-001.py` | off | CPU Small | model v1, endpoint |
| baseline | teddy-002 | `baseline/teddy-002.py` | off | CPU Small | model v1, endpoint, 3,538-cell table, AI Search endpoint + index |
| with_skill | teddy-001 | `with_skill/teddy-001.py` | on | GPU_SMALL | model v1, endpoint (inference table) |
| with_skill | teddy-002 | `with_skill/teddy-002.py` | on | GPU_SMALL | model v1, endpoint (inference table), 2,000-cell table, AI Search endpoint + index |

Same setup for every run: serverless GPU (A10), environment AI v6 (read from each notebook's
saved settings); the same temporary 3-line instructions file (179 bytes: be concise, no
memories, confirm before deleting); Genie Code memories checked before each run (none about
TEDDY or the eval); a new notebook and a new chat; one run at a time; baselines first. The
with-skill runs used the skill at commit `7ca4eee` (13 files: `SKILL.md` and the references),
installed in the personal skills folder; the Skills panel showed only `open-weight-models`.
Every endpoint and index was deleted after its outcome checks; models and tables are kept.

One with-skill teddy-002 attempt was **discarded** before anything was built (the user restarted
it after a setup slip; its notebook had no GPU selected). It is not scored.

#### Severity table — failed checks per run

| Task | Run | Critical failed | Major failed | Minor failed | Outcome passed | Shared input format |
|---|---|---|---|---|---|---|
| teddy-001 | baseline | 0 | 4 | 3 | yes | no |
| teddy-001 | with_skill | 0 | 0 | 0 | yes | yes |
| teddy-002 | baseline | 0 | 4 | 3 | yes | no |
| teddy-002 | with_skill | 0 | 0 | 0 | yes | yes |

Counts cover the corrected replicated checks, the new checks and the outcome checks; the shared
input format is a convention, shown on its own. With the original Sep–Oct logic (⚑, not
counted) the same table reads 2 / 4 / 3, 1 / 1 / 0, 3 / 4 / 3 and 0 / 0 / 1.

**What the baselines failed** (both tasks the same):

| Check | Severity | Why |
|---|---|---|
| `new_checks/ai_gateway/inference_table` | major | no inference table on the endpoint |
| `new_checks/ai_gateway/usage_tracking` | major | no usage tracking |
| `corrected/endpoint_deployment/ai_gateway_config` | major | no AI Gateway config attached |
| `new_checks/uc_provenance/provenance_tags` | major | no source / revision / licence tags on the UC model |
| `corrected/endpoint_deployment/run_go_gate` | minor | teddy-001: no switch in the code; teddy-002: a switch saved **on** (see "Deploy switch") |
| `corrected/model_registration/pip_reqs_from_source` | minor | serving pins are the versions installed at run time, not the upstream `pyproject.toml` ranges (see "Things that look off") |
| `corrected/pyfunc_quality/sys_modules_purge` | minor | the original text check for `sys.modules` (see "Things that look off") |

What each check guards: [`CHECKS.md`](../../CHECKS.md). Pass ratios are not reported: the
checks differ in weight, so only failed checks by severity are counted.

#### Deploy switch (`run_go`) — how each run behaved

The prompts say "Deploy … Write the code.", which can fairly be read as "deploy it" or "only
write the code". **How the agent read it is reported, not scored:** switching on for the run
itself, or stopping to ask, are both acceptable if it says so before spending. What is scored
(`run_go_gate`) is that the saved notebook does not spend money on a re-run: every billable step
sits behind a switch whose saved default is off. Every run got the same single reply,
`Yes, go ahead.`, when it asked.

| Run | Switch in the code | Saved default | Behaviour |
|---|---|---|---|
| baseline / teddy-001 | none | — | stopped and asked in chat; after the go-ahead ran the (ungated) endpoint cell |
| baseline / teddy-002 | `CONFIRM_CREATE_ENDPOINTS` + `assert` | **True** after the go-ahead (False before) | stopped and asked; after the go-ahead it saved the flag as `True  # confirmed by user` |
| with_skill / teddy-002 | `run_go` widget | `"false"` | read the prompt as "write the code", stopped and asked; after the go-ahead switched the widget on for the run and back off |
| with_skill / teddy-001 | `run_go` widget | `"false"` | "Since you asked to deploy, I'm switching run_go to true for this run only (its saved default stays false)"; deployed without asking; switched back off |

with_skill / teddy-001 is the behaviour the skill describes. with_skill / teddy-002 shows the
prompt wording is ambiguous: for the next round, make it explicit ("Deploy … and run it" vs
"Write the code only"), and consider a skill line: when a request mixes "deploy" and "write the
code", say which reading you took and ask.

#### Outcome — did what each run built work?

Measured with `eval/check_outcome.py` against the live resources before teardown, in the format
each endpoint declares.

- **All four endpoints** were READY, served the recorded UC model version, took expression input
  and returned one finite 512-number embedding per cell for cells of different lengths.
- **teddy-002 search, both runs:** AI Search endpoint ONLINE, index ready with every table row (3,538
  baseline, 2,000 with skill), and a nearest-neighbour query returned rows with a cell-type
  column.
- Run notes (Genie Code's own tests, in the records): baseline teddy-002 labelled 20 of 21
  held-out cells correctly through the endpoint and index; with-skill teddy-002 built a
  CELLxGENE Census atlas of 227 cell types, whose labels were too coarse to separate the
  sample's interneuron subtypes. These are not scored.

#### Evidence that the skill was used

| Run | Skills panel | Visible steps | Self-report |
|---|---|---|---|
| baseline / teddy-001 | empty | "Read Skill File" (built-ins) | `model-serving`, `model-serving-ui`, `diagnose-error` |
| baseline / teddy-002 | empty | not captured | guidance areas only (Model Serving, Vector Search, error fixing); no names |
| with_skill / teddy-002 | only `open-weight-models` | "Read open-weight-models skill" | described `open-weight-models` + its TEDDY reference item by item; no names |
| with_skill / teddy-001 | only `open-weight-models` | "Read Skill File" | `open-weight-models` `SKILL.md` and `references/models/teddy.md`, plus `model-serving-ui` |

Skill-specific choices in both with-skill notebooks: `transformers==4.41.0`, the shared input
format, AI Gateway inference table, `run_go` saved off, controls under the REST body `params`,
a pinned HF revision with a provenance manifest, and a comparison against upstream TEDDY.

#### How scored

The scorers of `eval/score_open_weight_models.py` (corrected checks, the original checks as ⚑
flags, new checks, outcome, convention), run with `mlflow.genai.evaluate` on a laptop with a
local MLflow store (mlflow 3.17.2): the notebook's steps unchanged, with the workspace-only lines
(`%pip`, the tracking URI, the repo path) replaced and the round set to `runs_20261027`. The
outcome checks come from the run records (measured before teardown).

**Scorer fixes made while scoring this round** (each with tests; none changes a round-1 or
Sep–Oct result — their score files were rewritten and are unchanged):

1. `run_go_gate` — a top-level `assert FLAG` before the deploy call counts as a gate (baseline
   teddy-002's `assert CONFIRM_CREATE_ENDPOINTS`); judged on its saved default.
2. Records name their export (`export_file`), so round-2 file names are found.
3. `uses_tmp` and `local_disk0` — a download run in a subprocess script is followed back to the
   notebook's path variables, and a `/local_disk0` path with a used `/tmp` fallback is accepted by
   both (with-skill teddy-001 was failed twice, critical, for a compute-adaptive path).
4. `endpoint_query` — new corrected version: a request actually sent to the endpoint (SDK
   `query` or an HTTP POST to `/invocations`, in code that runs). The original text match failed
   with-skill teddy-001's `w.api_client.do("POST", …/invocations)`, the call the skill prescribes.
5. `deps_pinned` — judged on the **served** model's code (its PythonModel classes and what they
   use, wrapper files, `code_paths`), so an upstream import used only to compare against upstream
   does not demand a `transformers` pin (both baselines serve their own plain-PyTorch copy).
6. `hf_xet_disabled` — the evidence that a download completed follows a per-file copy from the
   download folder (baseline teddy-002's `os.walk` copy into the Volume).

Practical-checks trial (`eval/practical_checks.py`, not adopted, as in round 1), failed critical /
major / minor: baseline 0 / 2 / 0 (teddy-001) and 0 / 3 / 0 (teddy-002), with skill 0 / 0 / 0 on both.
Both baselines fail its `fidelity_verified`, although both compared their own PyTorch copy with
upstream TEDDY (difference 0.0 on the sample cells): the trial does not recognise that comparison —
a flaw of the trial, not the run.

#### Things that look off

- **Two remaining minor baseline failures may be check flaws** for a model that does not use
  upstream code at serving time: `sys_modules_purge` is still the original text check
  (`"sys.modules" in response`), and `pip_reqs_from_source` asks for the upstream
  `pyproject.toml` ranges even when the served code is a plain-PyTorch copy. Not changed; both are
  minor.
- **The with-skill runs skipped some built-in skills** they said applied (with-skill teddy-001:
  `environment-management`, `diagnose-error`, `model-serving`, `query-parameters`), relying on the
  TEDDY reference instead. Nothing failed because of it; worth watching.
- **Two baselines fixed `EndpointCoreConfigInput(name=...)` themselves** after a first failure;
  the skill now states it, and the with-skill runs got it right first time.
- `family.json` `current_round` still points to round 1 (several tests read round 1 through it);
  pass `--round runs_20261027` to the tools.

</details>

<a id="round-1"></a>
### Round 1 — 2026-10-27

<details>
<summary>Open the round 1 notes: runs, failed checks per run, teddy-001 and teddy-002 in detail, outcome, evidence the skill was used, environment, caveats, things that look off</summary>

Genie Code with its built-in skills **plus** the open-weight-models skill (`with_skill`)
against Genie Code with its built-in skills **only** (`baseline`), on two tasks:
**teddy-001** (deploy TEDDY-70M as a serving endpoint) and **teddy-002** (deploy it and
build a nearest-neighbour cell search). Raw counts only — no percentages, no single score.


#### The runs

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
runs were re-scored the same way (`../runs_20260924/scores_corrected/`, new files; the
recorded scores are kept as reported), so each table gives one line comparing them.

#### Severity table — failed checks per run

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

#### teddy-001 — deploy

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

#### teddy-002 — deploy + nearest-neighbour cell search

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

#### Outcome — did what each run built work?

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

#### Evidence that the skill was used

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

#### Environment

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

#### Caveats

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

#### Things that look off

- **Resolved:** baseline/teddy-002's setup-time instructions fingerprint (11,229 bytes) is
  stale; the 179-byte temporary file was active during the run (see "Environment").
- The baselines chose CPU serving; the task guidelines say GPU_SMALL. No check scores the
  workload type directly (only the endpoint-settings table above shows it).

</details>

<a id="sep-oct-2026"></a>
### Sep–Oct 2026 — 2026-09-24 to 10-07

<details>
<summary>Open the Sep–Oct 2026 report as recorded: when, task IDs, results, caveats, verdict</summary>

**Status: historical record.** These runs were made between 2026-09-24 and 2026-10-07 against the skill
before it was renamed from `oss-models` to `open-weight-models`. They are kept as
recorded and are **not** the current evaluation; the retest is in [`eval/README.md`](../../README.md).

**Setup**: 2 TEDDY tasks, each run once with and once without the skill in Genie Code.
Scored with the deterministic phase checks in `scorers.py` (8 groups of text checks on
the exported notebook; no LLM judge), compared with `compare_runs.py`. Score files:
`baseline_scores.json`, `with_skill_scores.json`. Per-task reports: `reports/`.
The sub-checks themselves (added 2026-10-28): the rows counted as *replicated* in
[`../CHECKS.md`](../../CHECKS.md) — 27 for oss-001 and 30 for oss-002 (the search group applies to
oss-002 only). That list shows each one with its severity, what it guards and, where the logic
used here matched literal text, how the corrected version judges it.

#### When

| Run | Notebook created |
|---|---|
| oss-001 baseline and with-skill | 2026-09-24 |
| oss-002 baseline and first with-skill run | 2026-10-01 |
| oss-002 second with-skill run ("v2") | 2026-10-01 to 2026-10-07 |
| oss-005 with-skill (Geneformer, not scored) | 2026-10-03 |

Dates are the recorded workspace creation dates of the run notebooks. The per-task
reports were generated on 2026-10-01 (`reports/`).

#### Task IDs

The retest renames the tasks. Files in this folder keep the old IDs.

| Sep–Oct 2026 ID | ID | Task |
|---|---|---|
| oss-001 | teddy-001 | TEDDY-70M deploy |
| oss-002 | teddy-002 | TEDDY-70M deploy + nearest-neighbour cell search |
| oss-005 | geneformer-001 | Geneformer V1-10M deploy (not scored, see below) |
| oss-006 | geneformer-002 | Geneformer V2 BioNeMo deploy (prompt only, never run) |

#### Results (raw sub-check counts)

| Task | Skill version | Sub-checks | Baseline passed | With skill passed | Skill only | Baseline only | Both fail | Task outcome |
|---|---|---|---|---|---|---|---|---|
| oss-001 | as installed 2026-09-24 | 27 | 19 | 27 | 8 | 0 | 0 | win |
| oss-002, first skill run ("v1") | as installed 2026-10-01 | 30 | 26 | 27 | 2 | 1 | 2 | tie-fail |
| oss-002, second skill run ("v2") | `teddy.md` patched after the v1 run | 30 | 26 | 30 | 4 | 0 | 0 | win |

Both tasks also had not-applicable groups, recorded as a single pass:
`ai_search` and `edge_handling` for oss-001, `edge_handling` for oss-002.

- **oss-001**, sub-checks passed only with the skill: `uses_tmp`, `hf_xet_disabled`,
  `deps_pinned`, `sys_modules_purge`, `artifacts_declared`, `pip_reqs_from_source`,
  `ai_gateway_config`, `registration_smoke`.
- **oss-002 v1**: with the skill, `sdk_enums` and `registration_smoke` passed and
  `sys_modules_purge` failed (a regression); the other two failures were in both runs.
  Source: the ship-gate block in `reports/oss-001_report.md` and the v1 entry in
  `results/manifest.json`. The v1 notebook output was later overwritten by v2 in
  `results/with_skill/oss-002.txt`, so only these counts survive.
- **oss-002 v2**, passed only with the skill: `pip_reqs_from_source`, `sdk_enums`,
  `ai_gateway_config`, `registration_smoke`.
- Development logs the notebooks wrote about themselves (oss-002): baseline 12 bugs /
  15 fix iterations; skill v1 7 / 8; skill v2 7 / 7 (`results/manifest.json`).

`with_skill_scores.json` holds the v2 scores for oss-002, so `compare_runs.py` on the
two score files reports 2 wins, 0 regressions.

#### Caveats

1. **The with-skill prompt told Genie Code to read the skill.** It named the paths of
   `SKILL.md` and `teddy.md` and said to treat them as authoritative. This was by
   design: it measured the skill's *content*, not whether Genie Code picks the skill
   up on its own. The baseline prompt had no such paragraph.
2. **The baseline notebooks were not blank.** Each notebook's first cell held both
   prompts (including the with-skill prompt with the skill paths) and a list of the
   decisions being scored ("What the prompt does NOT prescribe"), and a later cell
   checked whether `SKILL.md` or `SKILL.md.off` existed. All of that was visible to
   Genie Code in the baseline runs.
3. **Skill absence in the baseline runs was not recorded.** The skill was switched off
   by renaming `SKILL.md` to `SKILL.md.off` in a project folder; no listing or
   screenshot was kept. The dates suggest a personal copy of the skill was created only
   after the baselines, so it was probably absent, but this is inferred, not recorded.
4. **oss-002 v2 is not a blind result.** Between the v1 and v2 runs, `teddy.md` was
   patched with fixes for the failures seen in v1 (8 patches, listed in
   `results/with_skill/oss-002_teddy_md_v1_diff.md`), and the `pip_reqs_from_source`
   check was widened in `scorers.py`. The baseline was not re-run.
5. **Pre-rename skill.** The skill was named `oss-models` when these runs were made;
   the content has changed since.
6. **One run per task per arm, two tasks.** Direction only, not significance.
7. **Geneformer:** oss-005 has a with-skill notebook output
   (`results/with_skill/oss-005.txt`, since removed from this record, see `README.md`) but no baseline and no scores; oss-006 was never
   run. Neither is in the results above.

#### Verdict

Historical evidence that the skill's TEDDY content helped when Genie Code was told to
read it. Not a ship decision for the current skill — see the retest.

#### Reproduce

`tests/test_eval_scorers.py` re-scores the recorded notebooks in `results/` with this
folder's `scorers.py` and checks they match the recorded score files.

</details>
