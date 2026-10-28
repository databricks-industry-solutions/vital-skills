# TEDDY evaluation — all rounds

The detail behind the summary in [`../../eval_report.md`](../../eval_report.md): how each round was run, results per round with the corrected checks, which checks passed only with the skill, and where the skill helps by category. What each check is: [`../../CHECKS.md`](../../CHECKS.md).

## How it was run

One fresh chat per run on serverless GPU notebooks, memories checked before each run, the skill
installed only for the with-skill runs.

**Rounds**, newest first:

| Round | When | What it is | Detail |
|---|---|---|---|
| **Round 2 — the latest** | 2026-10-27/28 | a clean rerun of all four runs with the fixed skill (`7ca4eee`) | [`runs_20261027/RESULTS.md`](runs_20261027/RESULTS.md) |
| Round 1 | 2026-10-27 | the same tasks, before the skill fixes this report lists | [`runs_20261026/RESULTS.md`](runs_20261026/RESULTS.md) |
| Sep–Oct 2026 | 2026-09-24 to 10-07 | the original sprint, on the skill before its rename from `oss-models` | [`runs_20260924/eval_report.md`](../../runs_20260924/eval_report.md) |

"Latest round" below means round 2.

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
saved default could not be read, or it was missing) and, on oss-002, the cell-table guard. Detail
and the original caveats (the with-skill prompt told Genie Code to read the skill):
[`runs_20260924/eval_report.md`](../../runs_20260924/eval_report.md).

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
none of their results changed (round 2 `RESULTS.md`, "How scored").
