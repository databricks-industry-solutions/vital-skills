# Evaluation Report — open-weight-models Skill

Genie Code with its built-in skills **plus** this skill, against built-ins only, on the same
plain request. One section per model family; each round's full results are in its
`models/<family>/runs_<date>/RESULTS.md`, and every check is explained in
[`CHECKS.md`](CHECKS.md). Protocol and scorers: [`README.md`](README.md). Raw counts only — no
percentages, no single score.

**Words used here.** GPU / CPU = graphics / central processing unit (serving hardware). SDK =
software development kit. REST body = the request text sent to an endpoint. UC = Unity Catalog.
AI Gateway = the endpoint setting that logs requests to an inference table and tracks usage.
`run_go` = the notebook switch that must be on before anything billable is created. Severity:
**critical** = does not work or gives wrong results; **major** = works but unsafe, ungoverned or
not reproducible; **minor** = robustness or style. "Failed c / m / m" = failed critical / major /
minor checks (counted checks only, [`CHECKS.md`](CHECKS.md)).

## At a glance

| Family | Latest round | Tasks | With skill, failed c / m / m | Baseline, failed c / m / m | Outcome checks | Verdict |
|---|---|---|---|---|---|---|
| TEDDY | round 2 (`runs_20261027`) | teddy-001, teddy-002 | 0 / 0 / 0 and 0 / 0 / 0 | 0 / 4 / 3 and 0 / 4 / 3 | all passed, both arms | ship with follow-ups |
| Geneformer | — | geneformer-001, -002 planned | — | — | — | not evaluated |

## TEDDY

Two tasks: **teddy-001** (deploy TEDDY-70M as a serving endpoint) and **teddy-002** (deploy it
and build a nearest-neighbour cell search). One fresh chat per run on serverless GPU notebooks,
memories checked before each run, the skill installed only for the with-skill runs. Detail:
[round 2 `RESULTS.md`](models/teddy/runs_20261027/RESULTS.md) (current),
[round 1 `RESULTS.md`](models/teddy/runs_20261026/RESULTS.md) (before the skill fixes),
[Sep–Oct 2026](runs_20260924/eval_report.md) (historical, pre-rename skill).

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

### Rounds

Primary sample per (run, task); repeat and discarded attempts are in each round's `RESULTS.md`.

| Round | Skill | teddy-001 baseline → with skill | teddy-002 baseline → with skill |
|---|---|---|---|
| Sep–Oct 2026 | `oss-models` (pre-rename) | 5 / 3 / 3 → 0 / 0 / 1 | 3 / 3 / 3 → 0 / 0 / 2 |
| Round 1 (`runs_20261026`) | before the fixes below | 0 / 5 / 3 → 0 / 0 / 1 | 0 / 5 / 4 → 0 / 0 / 0 |
| Round 2 (`runs_20261027`) | after the fixes (`7ca4eee`) | 0 / 4 / 3 → 0 / 0 / 0 | 0 / 4 / 3 → 0 / 0 / 0 |

Sep–Oct is re-scored with the corrected checks (replicated checks only; the original numbers and
their caveats are in its report: the with-skill prompt told Genie Code to read the skill). Every
endpoint built in rounds 1 and 2 with the skill worked; every baseline endpoint in round 2 worked
too.

### Failed checks by category

Counted checks that failed, primary samples, both tasks unless a task is named; "every round" =
Sep–Oct, round 1 and round 2. Major or critical unless marked minor.

| Category | What went wrong without the skill | With the skill |
|---|---|---|
| Download and staging | Sep–Oct: files saved where serverless cannot write (teddy-001); the Xet download setting not disabled; `transformers` not pinned below 5 | — |
| Model wrapper | rounds 1 and 2, and Sep–Oct teddy-001: no clean-up of cached code modules on re-run (minor); Sep–Oct: weights or code not declared as model artifacts (teddy-001), made-up gene IDs in the test input | — |
| Registration | every round: serving versions pinned to whatever was installed, not to the model's own dependency file (minor) | — |
| Endpoint deployment | every round: no request logging configured; no deploy switch saved off (minor) | Sep–Oct (both tasks) and round 1 (teddy-001 only): deploy switch not saved off (minor) |
| Tests | Sep–Oct: registered model never loaded and called before serving | — |
| Known-bad patterns | Sep–Oct: an open-ended `transformers>=` pin that installs the breaking 5.x | — |
| Search (teddy-002) | Sep–Oct and round 1: no guard if the cell table is missing (minor) | Sep–Oct: same (minor) |
| Governance | rounds 1 and 2: no inference table, no usage tracking, no source / revision / licence tags on the model | — |
| Safety | round 1: an existing table overwritten without looking (teddy-002) | — |
| Outcome | round 1: endpoint accepts only token IDs, not a cell's expression (teddy-001) | — |

The check behind each phrase, with its severity: [`CHECKS.md`](CHECKS.md). Governance, safety
and outcome checks did not exist in Sep–Oct 2026.

Not counted, but seen in every baseline: serving on CPU (the task guidelines suggest
`GPU_SMALL`) and a request format of its own instead of the shared single-cell format.

### Problems found in the skill, and where they are fixed

| Problem | Found | Fixed in | Round 2 |
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

The ship gate (skill-eval default) asks, per group, for at least one task-level win and no
regression.

| Group (round 2) | teddy-001 | teddy-002 | Gate |
|---|---|---|---|
| Corrected replicated checks | win | win | met |
| New checks | win | win | met |
| Outcome | tie, both worked | tie, both worked | no regression; no win to show, because the baselines' endpoints also worked |

**Ship with follow-ups.** No regression in any group in either round; round 1 also won on
outcome (a baseline endpoint that took only token IDs). Follow-ups: reword the prompts so
"deploy" and "write the code only" are distinct; more than one sample per run (the Genie Code
CLI harness in `benchmarks/biomnibench/` is the model for unattended runs); Geneformer.

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

| ID | Sep–Oct 2026 ID | Task |
|---|---|---|
| teddy-001 | oss-001 | TEDDY-70M deploy |
| teddy-002 | oss-002 | TEDDY-70M deploy + nearest-neighbour cell search |
| geneformer-001 | oss-005 | Geneformer V1-10M deploy (later) |
| geneformer-002 | oss-006 | Geneformer V2 BioNeMo deploy (later) |
