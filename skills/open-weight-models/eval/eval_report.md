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

**Severity of a failed check** (every check, with its severity and why it matters:
[`CHECKS.md`](CHECKS.md)):

| | Severity | Means |
|---|---|---|
| 🔴 | critical | does not work, or gives wrong results |
| 🟠 | major | works, but unsafe, ungoverned or not reproducible |
| 🟡 | minor | robustness or style |

## At a glance

**TEDDY** — latest round (round 2, 2026-10-27/28), tasks teddy-001 and teddy-002. Verdict: **ship
with follow-ups**.

| | Without the skill | With the skill |
|---|---|---|
| Endpoint (and, for teddy-002, the cell search) worked | yes | yes |
| Failed checks, each task | 🟠 4 · 🟡 3 (listed below) | none |
| Served on | CPU (1.7–15 s per cell in the runs' notes) | `GPU_SMALL`, the size the model reference recommends |
| Requests and usage logged | no | yes (AI Gateway inference table and usage tracking) |
| Where the model came from recorded | no | yes (pinned upstream revision, both licences, checksums) |
| Request format | its own, different each run | the shared single-cell format, the same across models |
| Re-running the saved notebook | can create billable resources again | creates nothing until `run_go` is switched on |
| Checked against TEDDY's own code | yes in the latest round | yes, before deploying |
| The `name=` SDK error | hit it, fixed it mid-run | right the first time |

**Geneformer** — pending (not yet evaluated; geneformer-001 and -002 planned).

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
| teddy-001 (oss-001) | "Deploy TEDDY-70M as a Model Serving endpoint on Databricks. The model is at https://huggingface.co/Merck/TEDDY … Use these names: model …, serving endpoint …, and store downloaded files under … Write the code and explain your decisions." | a UC-registered TEDDY-70M model and a serving endpoint that returns one 512-number embedding per cell | 36 | hard |
| teddy-002 (oss-002) | "Deploy TEDDY-70M as a Model Serving endpoint and enable nearest-neighbor cell-type search on Databricks. … Use these names: model …, serving endpoint …, Vector Search endpoint …, embeddings table …, index … Write the code." | the same, plus a table of reference-cell embeddings, a Vector Search endpoint and index, and a query that returns the nearest cells' types | 44 | hard |
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

### Results by round

Every sub-check judged with the corrected logic (`corrected_checks.py`; what each one is:
[`CHECKS.md`](CHECKS.md)). A task passes only if every sub-check passes, so one minor failure
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
real gene data in the test input and called the registered model before serving. Why they do
better now is not measured: Genie Code itself may have improved since September, and the
Sep–Oct setup also differed. The skill's lasting help is in **governance** (request logging,
provenance), **deployment safety** (a deploy switch saved off) and **reproducibility** (pins from
the model's own dependency file). Not counted, but also with the skill only: GPU serving and the shared request format.

### Problems found in the skill, and where they are fixed

Ordered by how much each one matters for deploying and serving.

| | Problem | Effect on deploying / serving | Found | Fixed in | Seen in the latest round |
|---|---|---|---|---|---|
| 🔴 | `EndpointCoreConfigInput` needs `name=` (databricks-sdk 0.71.0+) | endpoint creation fails with an error until it is added | most round-1 runs | the skill's examples | with-skill runs right first time; both baselines hit the error and fixed it |
| 🔴 | Genes outside TEDDY's vocabulary mapped to `<unk>` (upstream drops them) | the endpoint works but its embeddings differ from what TEDDY produces | `teddy.md`, round 1 | `teddy.md` "Out-of-vocabulary (OOV) genes" | both with-skill models drop them and reject a request with no known gene |
| 🟠 | Inference controls sent with the SDK's `extra_params`, which a custom model silently ignores | the call succeeds but a requested setting (e.g. `max_seq_len`) is not applied, with no error | with skill, round 1 | `SKILL.md` Troubleshooting #13; `teddy.md` "Sending inference controls" | both sent them in the REST body and showed a changed value changes the output |
| 🟠 | Wrapper compared only with itself, not with upstream TEDDY | a wrapper that drifts from TEDDY would deploy without anyone noticing | with skill, round 1 | `teddy.md` "Upstream-parity test" | both matched upstream (cosine ≥ 0.99999) |
| 🟠 | Drop and recreate an index on a dimension mismatch | serving unaffected; risk of deleting someone's existing index | `teddy.md`, round 1 | `SKILL.md` Best Practices #15 (write additively, ask before replacing) | nothing replaced in either with-skill run |
| 🟡 | Licence ambiguous upstream: model card Apache-2.0, bundled package MIT | none technically; matters before redistributing the model | round 1 | the skill records both and asks the user to confirm | both with-skill runs recorded both |
| 🟡 | "Deploy … Write the code." read as code only, so it stopped to ask | none; one extra confirmation | with skill teddy-002, latest round | prompt wording, next round | reported, not scored |

**The scorer improved through the rounds too.** Each round of testing showed checks failing valid
code: some matched literal text from the skill (a different valid way of doing the same thing
failed), others could not follow a path or a copy through the notebook. They were fixed as they
were found: 13 replicated checks are now judged by intent (one added in round 2), and five more
were corrected while scoring round 2. After those five, round 1 and Sep–Oct were re-scored and
none of their results changed (round 2 `RESULTS.md`, "How scored").

### Verdict

**Ship with follow-ups.** In the latest round, with the skill both tasks passed every sub-check and
without it neither did, and no sub-check got worse with the skill (in any round). That meets the
repo's ship rule (`skill-eval`: the skill must win at least one task and make none worse). The
endpoints worked with and without the skill, so the gain is in how they are built, not whether
they work. Follow-ups, in order:
1. **Automate the runs.** Hand-driven runs in the workspace UI were slow and error-prone (wrong
   chat tab, compute not selected, a stale prompt path); each slip cost a rerun. Genie Code's
   memories also had to be checked and cleared by hand before every run: they are saved on the
   service side with no off switch, and in Sep–Oct they carried eval details into later chats. The Genie Code
   CLI harness in `benchmarks/biomnibench/` already does the setup by script: it runs
   `genie exec` once per task, arm and repeat in a fresh folder, with a config profile that turns
   memories and personal skills off, copies the skills at a pinned commit into that folder for the
   skills arm, and logs which skills were opened (its `README.md`, "How it works"). Adapt it for
   these tasks, with outcome checks and teardown scripted per run, and report it next to the
   workspace runs (as `accelerators` does). The CLI works from a local folder, not a serverless
   GPU notebook, so it complements the workspace runs rather than replacing them.
2. Reword the prompts so "deploy" and "write the code only" are distinct — not done yet: the prompts
   in `evalset.json` were kept word for word so rounds 1 and 2 stay comparable, and the skill has
   no line yet for a request that mixes the two.
3. Geneformer.

### Limitations

- **One sample per (run, task) in each round, run by hand:** directional, not statistical. Manual
  runs in the workspace UI are slow and prone to the same mistakes repeating (wrong chat tab,
  compute not selected, a stale prompt, a go-ahead timed wrong, memories to clear before each run);
  several runs had to be redone. Running the tasks through the Genie Code CLI (follow-up 1 above)
  would allow repeats and remove those steps: its config profile switches skills and memories off
  per run.
- Checks come from the skill's own references, so they can favour it; the corrected checks and the
  practical-checks trial (reported in each round) judge goals instead of text.
- Built-in skills are always on, so the result is the skill's value on top of them.
- Each round's `RESULTS.md` records the setup and any slips.

## Geneformer

**Pending.** Not evaluated yet (geneformer-001 / -002 planned). Starting point, as for TEDDY:
the hand-built, working reference notebooks the model reference was written from —
`gt_geneformer` (Hugging Face path) and `gt_geneformer_bionemo` (NVIDIA BioNeMo path), named in
`references/models/geneformer.md` — from which the tasks and checks are drawn; the runs must not
be able to read them. The Sep–Oct 2026 Geneformer notebooks and one with-skill output are kept,
unscored, in `runs_20260924/`.

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
