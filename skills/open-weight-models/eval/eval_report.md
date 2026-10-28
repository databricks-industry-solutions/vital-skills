# Evaluation Report — open-weight-models Skill

## Takeaways (TEDDY, the model family evaluated so far)

1. **Without the skill, Genie Code can already deploy a working TEDDY endpoint** (and the cell
   search). The skill is not what makes it work.
2. **With the skill, the endpoint is one a team can keep.** It runs on a GPU, logs its requests,
   records where the model came from (revision, licences), uses the same request format as the
   other single-cell models, and the saved notebook does not create billable resources again on
   a re-run. The runs without the skill missed these every time.
3. **The evaluation found real problems in the skill and they are fixed:** six, for example
   settings that a custom model silently ignored. The second round shows the fixes work.
4. **Limits:** one run per task per arm, driven by hand in the workspace, so read this as
   direction, not statistics; automated runs are the next step (see Verdict). The checks were
   corrected along the way (some had failed valid code); the numbers below use the corrected
   checks.

Genie Code with its built-in skills **plus** this skill, against built-ins only, on the same
plain request. One section per model family; each round's full results are in its
`models/<family>/runs_<date>/RESULTS.md`, and every check is explained in
[`CHECKS.md`](CHECKS.md). Protocol and scorers: [`README.md`](README.md). Raw counts only — no
percentages, no single score.

**Words used here.** GPU / CPU = graphics / central processing unit (serving hardware). SDK =
software development kit. REST body = the request text sent to an endpoint. UC = Unity Catalog.
AI Gateway = the endpoint setting that logs requests to an inference table and tracks usage.
`run_go` = the notebook switch that must be on before anything billable is created.

**Severity of a failed check:** 🔴 critical = does not work or gives wrong results ·
🟠 major = works, but unsafe, ungoverned or not reproducible · 🟡 minor = robustness or style.
Every check, with its severity and why it matters: [`CHECKS.md`](CHECKS.md).

## At a glance

| Family | Latest round | Tasks | Failed checks with the skill (each task) | Failed checks without the skill (each task) | Endpoint worked? | Verdict |
|---|---|---|---|---|---|---|
| TEDDY | round 2 (2026-10-27/28) | teddy-001, teddy-002 | none | 🟠 4 · 🟡 3 | yes, with and without | ship with follow-ups |
| Geneformer | — | geneformer-001, -002 planned | — | — | — | pending (not yet evaluated) |

**What the TEDDY runs without the skill failed** (latest round, the same 7 on both tasks):

| | Check | Why it matters |
|---|---|---|
| 🟠 | request logging configured on the endpoint (`ai_gateway_config`) | requests and responses are not logged, so the endpoint is not governed |
| 🟠 | inference table on (`inference_table`) | no record of what was sent and returned, so nothing to audit |
| 🟠 | usage tracking on (`usage_tracking`) | no record of who used the endpoint and how much |
| 🟠 | source, revision and licence tags on the model (`provenance_tags`) | which upstream version and licence the model came from is not recorded, so it is not reproducible |
| 🟡 | deploy switch saved off (`run_go_gate`) | re-running the saved notebook can create billable resources again |
| 🟡 | serving pins from the model's own dependency file (`pip_reqs_from_source`) | the pins record what was installed that day, not what the model needs |
| 🟡 | cached code modules cleaned up on re-run (`sys_modules_purge`) | re-running cells out of order can serve stale code |

## TEDDY

### Tasks

| ID (Sep–Oct ID) | The request, as pasted (names and paths per run, from `evalset.json`) | What a run has to build | Sub-checks | Difficulty |
|---|---|---|---|---|
| teddy-001 (oss-001) | "Deploy TEDDY-70M as a Model Serving endpoint on Databricks. The model is at https://huggingface.co/Merck/TEDDY … Use these names: model …, serving endpoint …, and store downloaded files under … Write the code and explain your decisions." | a UC-registered TEDDY-70M model and a serving endpoint that returns one 512-number embedding per cell | 38 | hard |
| teddy-002 (oss-002) | "Deploy TEDDY-70M as a Model Serving endpoint and enable nearest-neighbor cell-type search on Databricks. … Use these names: model …, serving endpoint …, Vector Search endpoint …, embeddings table …, index … Write the code." | the same, plus a table of reference-cell embeddings, a Vector Search endpoint and index, and a query that returns the nearest cells' types | 45 | hard |
| geneformer-001 (oss-005), geneformer-002 (oss-006) | Geneformer V1-10M deploy; Geneformer V2 (BioNeMo) deploy | — | — | planned |

Sub-checks per task are listed in [`CHECKS.md`](CHECKS.md) (the outcome check needs the endpoint
— and for teddy-002 the search — to work when measured).

### How it was run

One fresh chat per run on serverless GPU notebooks, memories checked before each run, the skill
installed only for the with-skill runs.

**Rounds**, newest first:

| Round | When | What it is | Detail |
|---|---|---|---|
| **Round 2 — the latest** | 2026-10-27/28 | a clean rerun of all four runs with the fixed skill (`7ca4eee`) | [`runs_20261027/RESULTS.md`](models/teddy/runs_20261027/RESULTS.md) |
| Round 1 | 2026-10-27 | the same tasks, before the skill fixes this report lists | [`runs_20261026/RESULTS.md`](models/teddy/runs_20261026/RESULTS.md) |
| Sep–Oct 2026 | 2026-09-24 to 10-07 | the original sprint, on the skill before its rename from `oss-models` | [`runs_20260924/eval_report.md`](runs_20260924/eval_report.md) |

"Latest round" below means round 2.

### What the skill changed, beyond the numbers

With and without the skill, Genie Code built a TEDDY endpoint that works. What the skill changes
is how that endpoint is built and whether it is safe to keep:

| Without the skill | With the skill |
|---|---|
| Served on CPU (1.7–15 s per cell in the runs' notes) | Served on `GPU_SMALL`, the size the model reference recommends |
| No record of requests or usage | AI Gateway inference table and usage tracking on |
| No record of where the model came from | Pinned upstream revision, both licences and checksums recorded |
| Its own request format per run | The shared single-cell request format, the same across models |
| A re-run of the saved notebook can create billable resources again | `run_go` saved off; switched on only for the run that asked to deploy |
| Settings sent where a custom model may ignore them (round 1, both arms) | Settings sent in the REST body, with a test that a changed value changes the output |
| Hits the `name=` SDK error and fixes it mid-run | Gets it right the first time |

The two with-skill runs also checked the wrapper against TEDDY's own code before deploying,
and rejected requests whose genes TEDDY does not know.

### Results by round

Every sub-check judged with the corrected logic (`corrected_checks.py`; what each one is:
[`CHECKS.md`](CHECKS.md)). A task passes only if every sub-check passes, so one minor failure
makes it fail. "Skill wins" = the task passes every sub-check with the skill and not without it;
"neither passes every check" = both runs fail at least one sub-check (named in "Failed in both"
when it is the same one). Rounds 1 and 2 add the governance, safety and outcome checks, so they
have more sub-checks than Sep–Oct. Primary samples only.

| Round | Task | Sub-checks | Baseline passed | With skill passed | Passed only with the skill | Passed only without | Failed in both | Task result |
|---|---|---|---|---|---|---|---|---|
| Round 2 (latest) | teddy-001 | 38 | 31 | 38 | 7 | 0 | 0 | skill wins |
| Round 2 (latest) | teddy-002 | 45 | 38 | 45 | 7 | 0 | 0 | skill wins |
| Round 1 | teddy-001 | 38 | 30 | 37 | 7 | 0 | 1 (`run_go_gate`) | neither passes every check |
| Round 1 | teddy-002 | 45 | 36 | 45 | 9 | 0 | 0 | skill wins |
| Sep–Oct | teddy-001 (oss-001) | 29 | 18 | 28 | 10 | 0 | 1 (`run_go_gate`) | neither passes every check |
| Sep–Oct | teddy-002 (oss-002, after patching `teddy.md`) | 31 | 22 | 29 | 7 | 0 | 2 (`run_go_gate`, `graceful_skip`) | neither passes every check |

**Which sub-checks passed only with the skill** (plain words; the check name, in brackets, is
the row in [`CHECKS.md`](CHECKS.md)):

| Round, task | Passed only with the skill |
|---|---|
| Latest round, teddy-001 and teddy-002 (7 each, the same 7) | request logging configured (`ai_gateway_config`), inference table on (`inference_table`), usage tracking on (`usage_tracking`), source / revision / licence tags on the model (`provenance_tags`), deploy switch saved off (`run_go_gate`, minor), serving pins from the model's own dependency file (`pip_reqs_from_source`, minor), cached code modules cleaned up on re-run (`sys_modules_purge`, minor) |
| Round 1, teddy-001 (7) | request logging configured, inference table, usage tracking, provenance tags, serving pins from the dependency file, module clean-up — and the endpoint takes a cell's expression, not token IDs (`accepts_expression_input`) |
| Round 1, teddy-002 (9) | the 7 of the latest round, plus a guard if the cell table is missing (`graceful_skip`, minor) and looking before writing to an existing table (`existing_resources/table`) |
| Sep–Oct, teddy-001 (10) | files saved where serverless can write (`uses_tmp`), the Xet download setting handled (`hf_xet_disabled`), `transformers` pinned below 5 (`deps_pinned`), no open-ended `transformers>=` pin (`transformers`), weights and code declared as model artifacts (`artifacts_declared`), real gene IDs in the test input (`real_gene_ids`), registered model called before serving (`registration_smoke`), request logging configured, serving pins from the dependency file, module clean-up |
| Sep–Oct, teddy-002 (7) | the Xet setting, the `transformers` pin, no open-ended pin, real gene IDs, registered model called before serving, request logging configured, serving pins from the dependency file |

No sub-check passed without the skill and failed with it, in any round. Sep–Oct as originally
reported (the old literal-text logic, and a with-skill prompt that told Genie Code to read the
skill): [`runs_20260924/eval_report.md`](runs_20260924/eval_report.md).

### Where the skill helps — by category

Both tasks together. "Skill only" = passed with the skill, failed without it.

| Category | Sub-checks | Latest round, baseline → with skill | Skill only, latest round | Skill only, earlier rounds |
|---|---|---|---|---|
| Download and staging | 6 | 6 → 6 | — | Sep–Oct: download location, Xet setting, `transformers` pin |
| Model wrapper | 14 | 12 → 14 | clean-up of cached code modules on re-run (minor) | round 1: same; Sep–Oct: also artifacts declared, real gene IDs in the test input |
| Registration | 10 | 8 → 10 | serving pins taken from the model's dependency file (minor) | same |
| Endpoint deployment | 8 | 4 → 8 | request logging configured; deploy switch saved off (minor) | request logging configured; round 1 teddy-002: deploy switch saved off |
| Tests before and after deploying | 6 | 6 → 6 | — | Sep–Oct: registered model called before serving |
| Known-bad patterns | 10 | 10 → 10 | — | Sep–Oct: no open-ended `transformers>=` pin |
| Edge cases | 2 | 2 → 2 | — | — |
| Search (teddy-002) | 4 | 4 → 4 | — | round 1: guard on a missing cell table (minor) |
| Governance: request logging | 4 | 0 → 4 | inference table, usage tracking | round 1: same |
| Governance: provenance | 4 | 2 → 4 | source, revision and licence tags on the model | round 1: same |
| Safety: look before creating or deleting | 4 | 4 → 4 | — | round 1: no overwrite of an existing table |
| Outcome | 11 | 11 → 11 | — | round 1: endpoint takes a cell's expression, not token IDs |

**Reading it:** without the skill, Genie Code passed the download, test and known-bad-pattern
checks in rounds 1 and 2 that it failed in Sep–Oct (the Sep–Oct setup also differed). The
skill's lasting help is in **governance** (request logging, provenance), **deployment safety**
(a deploy switch saved off) and **reproducibility** (pins from the model's own dependency file).
Not counted, but also with the skill only: GPU serving and the shared request format.

### Problems found in the skill, and where they are fixed

| Problem | Found | Fixed in | Seen in the latest round |
|---|---|---|---|
| Inference controls sent with the SDK's `extra_params`, which a custom model silently ignores | with skill, round 1 | `SKILL.md` Troubleshooting #13; `teddy.md` "Sending inference controls" | both with-skill runs sent them in the REST body and showed a changed value changes the output |
| Wrapper compared only with itself, not with upstream TEDDY | with skill, round 1 | `teddy.md` "Upstream-parity test" | both matched upstream (cosine ≥ 0.99999) |
| `EndpointCoreConfigInput` needs `name=` (databricks-sdk 0.71.0+) | most round-1 runs | the skill's examples | with-skill runs right first time; both baselines hit the error |
| Genes outside TEDDY's vocabulary mapped to `<unk>` (upstream drops them) | `teddy.md`, round 1 | `teddy.md` "Out-of-vocabulary (OOV) genes" | both with-skill models reject a request with no known gene |
| Drop and recreate an index on a dimension mismatch | `teddy.md`, round 1 | `SKILL.md` Best Practices #15 (write additively, ask before replacing) | nothing replaced in either with-skill run |
| Licence ambiguous upstream: model card Apache-2.0, bundled package MIT | round 1 | the skill records both and asks the user to confirm | both with-skill runs recorded both |
| "Deploy … Write the code." read as code only, so it stopped to ask | — | prompt wording, next round | with skill teddy-002; reported, not scored |

The scorer had problems too: literal-text matches and path-tracing gaps that failed valid code.
13 replicated checks are now judged by intent (one added in round 2) and five more were corrected
while scoring round 2; earlier rounds' results are unchanged (round 2 `RESULTS.md`, "How scored").

### Verdict

**Ship with follow-ups.** In the latest round, with the skill both tasks passed every sub-check and
without it neither did, and no sub-check got worse with the skill (in any round). That meets the
repo's ship rule (`skill-eval`: the skill must win at least one task and make none worse). The
endpoints worked with and without the skill, so the gain is in how they are built, not whether
they work. Follow-ups, in order:
1. **Automate the runs.** Hand-driven runs in the workspace UI were slow and error-prone (wrong
   chat tab, compute not selected, a stale prompt path); each slip cost a rerun. The Genie Code
   CLI harness in `benchmarks/biomnibench/` already switches skills and memories by config and
   runs replicates unattended; adapt it for these tasks, with outcome checks and teardown
   scripted per run, and report it next to the workspace runs (as `accelerators` does).
2. Reword the prompts so "deploy" and "write the code only" are distinct.
3. Geneformer.

### Limitations

- One sample per (run, task) in each round: directional, not statistical.
- Checks come from the skill's own references, so they can favour it; the corrected checks and the
  practical-checks trial (reported in each round) judge goals instead of text.
- Built-in skills are always on, so the result is the skill's value on top of them.
- Runs are driven by hand in the workspace UI; each round's `RESULTS.md` records the setup and any
  slips.

## Geneformer

**Pending.** Not evaluated yet (geneformer-001 / -002 planned). The Sep–Oct 2026 Geneformer
notebooks and one with-skill output are kept, unscored, in `runs_20260924/`.

## Reproduce

```bash
uv run --isolated --with pytest --with-requirements skills/open-weight-models/tests/requirements.txt \
    python -m pytest -q skills/open-weight-models/tests
uv run --isolated --with-requirements skills/open-weight-models/tests/requirements.txt \
    python skills/open-weight-models/eval/practical_checks.py --round runs_20261027
```

Score files: `models/<family>/runs_<date>/scores/`; run records (environment, outcome, skill-use
evidence): `models/<family>/runs_<date>/run_records.json`.

Task list: "TEDDY → Tasks" above.
