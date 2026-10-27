# Evaluation Report — open-weight-models Skill

**Words used here.** CPU = central processing unit and GPU = graphics processing unit (the
two kinds of serving hardware; `GPU_SMALL` is Model Serving's smallest GPU size). SDK =
software development kit (the Databricks Python SDK). REST = the plain web interface of an
endpoint; "REST body" = the request text sent to it. UC = Unity Catalog (where models are
registered). HF = Hugging Face (where TEDDY's files live). VS = Vector Search, the index
used for nearest-neighbour cell lookup. OOV = out-of-vocabulary (a gene the model's
vocabulary does not contain). AI Gateway = the endpoint setting that logs requests to an
inference table and tracks usage. Sub-check = one pass/fail line of a scorer. Severity:
**critical** = the endpoint does not work or gives wrong embeddings; **major** = works but
unsafe, ungoverned, unverified or not reproducible; **minor** = robustness or style.
"Shared input format" = the request layout `adata_sparsematrix` + `adata_obs` + `adata_var`
used by `teddy.md`, `scgpt.md` and Genesis Workbench. Raw counts only throughout — no
percentages, no single score.

**Summary.** One model family has been evaluated: **TEDDY** (2026-10-27, two tasks,
Genie Code with its built-in skills plus this skill against built-ins only). With the skill,
both tasks built a working GPU endpoint with request logging, provenance and the shared
input format; teddy-001 had no failed checks and teddy-002 one minor failure. The
baselines also built working endpoints in their own formats, but on CPU, without AI
Gateway logging, and (for teddy-001) taking pre-tokenized token IDs instead of cell
expression. Several baseline failures are false negatives of checks written from the
skill's own text; a goal-based rescoring (a trial, reported separately) keeps the
direction. The runs also found six problems in the skill itself, now fixed in the skill
(see "Findings for the skill"). **Geneformer** is not evaluated yet. An earlier
Sep–Oct 2026 TEDDY round is kept as a historical record.

**Setup (TEDDY round).** Two tasks — **teddy-001** (deploy TEDDY-70M as a serving
endpoint) and **teddy-002** (deploy it and build a nearest-neighbour cell search) — each
run in one fresh Genie Code chat per arm on serverless GPU notebooks (environment v6),
same plain request in both arms, every run with its own resources under schema
`skills_owm_202610`. Scored with the deterministic checks in `scorers.py` through
`mlflow.genai.evaluate` (no LLM judge), compared with skill-eval's `compare_runs`. Detail:
[`models/teddy/runs_20261026/RESULTS.md`](models/teddy/runs_20261026/RESULTS.md) (all
sub-checks, outcome evidence, skill-use evidence),
[`PRACTICAL_TRIAL.md`](models/teddy/runs_20261026/PRACTICAL_TRIAL.md) (practical trial),
[round README](models/teddy/runs_20261026/README.md) (deviations from plan), protocol in
[`README.md`](README.md).

**Numbers re-derived.** Every count below was recomputed from the committed score files in
`models/teddy/runs_20261026/scores/` (per group, and failed checks per severity from the
severities in `scorers.py`) and matches `RESULTS.md`; the practical numbers are copied from
`PRACTICAL_TRIAL.md` as merged at `4e100e7`.

## Results — TEDDY, 2026-10-27

Runs: **b/001 a2** = baseline teddy-001 attempt 2 (primary baseline sample), **b/001 a1** =
attempt 1 (second sample, same prompt word for word, different resource names — see
Limitations), **b/002** = baseline teddy-002, **ws/001** and **ws/002** = with skill.

### Task outcomes per check group (ship gate)

A task passes a group only if every check in that group passes. Standard pair, primary
samples; attempt 1 shown alongside (it gives the same outcomes).

| Group | teddy-001 baseline (a2 / a1) | teddy-001 with skill | teddy-001 outcome | teddy-002 baseline | teddy-002 with skill | teddy-002 outcome |
|---|---|---|---|---|---|---|
| Replicated Sep–Oct 2026 checks ‡ | fail / fail | pass | **win** | fail | fail | tie-fail |
| New checks | fail / fail | pass | **win** | fail | pass | **win** |
| Outcome | fail / fail | pass | **win** | pass | pass | tie-pass |

Wins: replicated 1, new checks 2, outcome 1. Regressions: 0 in every group.

### Replicated Sep–Oct 2026 checks ‡ — passed / total

| Task | b/001 a2 | b/001 a1 | b/002 | ws/001 | ws/002 |
|---|---|---|---|---|---|
| teddy-001 | 20 / 29 | 21 / 29 | — | 29 / 29 | — |
| teddy-002 | — | — | 24 / 31 | — | 30 / 31 |

‡ **Will change.** These checks are being switched to corrected logic (the old verbatim
logic kept as a flag) and the Sep–Oct 2026 runs re-scored; this table, the replicated row
above and the severity table below will be updated when that lands.

### New checks — passed / total

AI Gateway inference table and usage tracking on one attached config, UC registration
under a three-part name, licence and HF revision tags, and a look for existing resources
before creating them.

| Task | b/001 a2 | b/001 a1 | b/002 | ws/001 | ws/002 |
|---|---|---|---|---|---|
| teddy-001 | 2 / 5 | 2 / 5 | — | 5 / 5 | — |
| teddy-002 | — | — | 3 / 7 | — | 7 / 7 |

### Outcome — did the endpoint work? Passed / total

Measured on the live deployments before teardown, each request in the endpoint's own
declared input format.

| Task | b/001 a2 | b/001 a1 | b/002 | ws/001 | ws/002 |
|---|---|---|---|---|---|
| teddy-001 | 3 / 4 | 2 / 4 | — | 4 / 4 | — |
| teddy-002 | — | — | 7 / 7 | — | 7 / 7 |

Baseline failures: b/001 a2 and a1 take pre-tokenized token IDs, leaving TEDDY's
tokenization to every client; a1 also returns errors for mixed-length batches (its padding
value −100 is used as a token ID).

**Convention check, reported separately (not a failure):** uses the shared input format —
b/001 a2 no, a1 no, b/002 no, ws/001 yes, ws/002 yes. Each baseline chose a valid format of
its own.

### Failed checks per run, by severity ‡

Replicated + new + outcome checks; the convention check is not counted.

| Task | Run | Critical failed | Major failed | Minor failed | Outcome passed |
|---|---|---|---|---|---|
| teddy-001 | b/001 a2 (primary) | 5 | 6 | 2 | no |
| teddy-001 | b/001 a1 | 3 | 8 | 2 | no |
| teddy-001 | ws/001 | 0 | 0 | 0 | yes |
| teddy-002 | b/002 | 4 | 5 | 2 | yes |
| teddy-002 | ws/002 | 0 | 0 | 1 | yes |

Several of the baseline critical failures are known or likely false negatives (marked ⚑
in `RESULTS.md`): the baselines reached the same goal another valid way — `tempfile`
instead of a literal `/tmp`, no Xet workaround yet a completed download, `ModelSignature`
instead of `infer_signature`, `transformers<5` or no `transformers` at all instead of
`==4.41.0`. ws/002's one minor failure is also one (it cites "TEDDY's pyproject", not the
literal `pyproject.toml`).

### Practical checks — trial, separately reported

Goal-based checks that accept any valid way of reaching each goal (`eval/practical_checks.py`;
not adopted, not used for the gate). Passed / total and failed critical / major / minor:

| Task | Run | Practical passed | Practical failed (crit / major / minor) |
|---|---|---|---|
| teddy-001 | b/001 a2 (primary) | 10 / 12 | 0 / 2 / 0 |
| teddy-001 | b/001 a1 | 6 / 12 | 1 / 5 / 0 |
| teddy-001 | ws/001 | 12 / 12 | 0 / 0 / 0 |
| teddy-002 | b/002 | 9 / 13 | 0 / 4 / 0 |
| teddy-002 | ws/002 | 12 / 13 | 0 / 1 / 0 |

Under these, the false negatives above disappear; what remains is real: a2 fails on what
its endpoint accepts (token IDs) and governance (no AI Gateway logging); b/002 on no
comparison with upstream TEDDY, a table created without looking first, no revision pin, and
governance; ws/002 on no comparison with upstream TEDDY (see "Findings").

### GPU guideline (reported apart, not counted)

The task guidelines suggest `GPU_SMALL`; the prompts do not ask for it. Both with-skill
endpoints ran on `GPU_SMALL`; all three baselines served on CPU, and their endpoints
worked. On CPU, b/002's endpoint took about 15 s per cell (run notes) — fine for a smoke
test, slow for any real batch.

### How skill use was evidenced

From the chat's self-report, its visible steps and skill-specific choices in the notebook —
not from hidden markers. Both with-skill chats named open-weight-models (`SKILL.md`,
`teddy.md`) in their self-report and showed "Read Skill File" steps; both notebooks used
`teddy.md`-only choices (`HF_HUB_DISABLE_XET`, `transformers==4.41.0`, the shared input
format, a `run_go` gate, a `sys.modules` purge). No baseline self-report named the skill,
the Skills panel showed "No skills found" before every baseline, and no baseline notebook
contains any of those choices. The hidden-code marker used before is retired: ws/001
declined it as possibly injected, correctly — a planted "write this code" instruction in a
skill file is a prompt-injection pattern.

## Findings for the skill

Found in these runs; all six are now fixed in the skill (`CHANGELOG.md`).

1. **SDK `extra_params` are silently dropped.** The skill advised
   `w.serving_endpoints.query(extra_params=...)` for inference controls. The SDK sends them
   under the body key `"extra_params"`, which the MLflow scoring server ignores, so the call
   succeeds and `predict()` runs with defaults. ws/001 found it with a bad-pooling test
   (run notes); measured on its endpoint: `extra_params` gave embeddings identical to the
   default (cosine 1.0), the REST body `"params"` key changed them (cosine 0.53–0.64).
   ws/002 followed the old advice and only checked that the call succeeded. Fix: controls go
   in the REST body `"params"` field, and a test must show a non-default value changes the
   output.
2. **`EndpointCoreConfigInput` needs `name=` from databricks-sdk 0.71.0** (serverless
   environment v6 ships 0.122.0). Four of the five runs' notes record the
   `missing 1 required positional argument: 'name'` error and a fix. The skill's examples
   now pass `name=`, which also works on older SDKs.
3. **Licence ambiguity upstream.** At HF revision `8c880aa` the TEDDY model card declares
   Apache-2.0, the bundled package's `pyproject.toml` declares MIT, and there is no
   `LICENSE` file (noted by ws/001). The skill now records both and asks the user to
   confirm with the model owner before redistribution.
4. **Upstream-parity gap.** ws/001 compared its wrapper with TEDDY's own tokenization
   (`_build_batch_tensors` with `GeneTokenizer`) and measured cosine 1.000000 per cell;
   ws/002 compared the endpoint only with its own wrapper, which proves the deployment, not
   that the wrapper reproduces TEDDY. The skill's validation step asked only for "a small
   known-input regression test". Now required: an upstream-parity test (same tokens, cosine
   ~1.0 against the model's own preprocessing and inference), with the TEDDY functions named
   in `teddy.md`.
5. **OOV handling and normalisation.** `teddy.md` mapped OOV genes to `<unk>`. Checked in
   the upstream source at revision `8c880aa`: TEDDY drops OOV genes before choosing the
   top-K (`tokenization.py:129-146`, `:230`; it raises when fewer than 10% are in the
   vocabulary), drops zero-expression genes (`include_zero_genes=False`, `:261-264`), and
   its pretraining preprocessing normalises each cell to 10,000 counts then divides each
   gene by its TEDDY median (`preprocess.py:272-306`; settings in the model card,
   `README.md:136-155`). `teddy.md` now matches, as ws/001's wrapper already did.
6. **Safe writes.** b/002 overwrote its cells table (`insertInto(..., overwrite=True)`)
   without checking what was there, and `teddy.md` advised dropping and recreating an index
   on a dimension mismatch. The skill now appends or merges into existing tables and asks
   before deleting, overwriting or recreating anything the run did not create.

## Failure taxonomy

| failure mode | count | arm | status |
|---|---|---|---|
| served on CPU, not the suggested `GPU_SMALL` | 3 runs (a2, a1, b/002) | baseline | eliminated by skill (both with-skill runs on GPU) |
| no AI Gateway inference table / usage tracking | 3 runs | baseline | eliminated by skill |
| endpoint takes pre-tokenized token IDs, not cell expression | 2 runs (a2, a1) | baseline | eliminated by skill |
| no comparison with upstream TEDDY | 2 runs (a1, b/002) | baseline | eliminated in ws/001; still seen in ws/002 (next row) |
| no comparison with upstream TEDDY | 1 run (ws/002) | with skill | fixed in skill — upstream-parity test now required (finding 4); confirm on next round |
| inference controls sent via SDK `extra_params` (silently ignored), checked only for success | 1 run (ws/002) | with skill | fixed in skill (finding 1) |
| upstream revision not pinned to a commit | 2 runs (a1, b/002) | baseline | eliminated by skill |
| existing table overwritten without looking first | 1 run (b/002) | baseline | eliminated in ws/002; skill now also requires append/merge or confirmation (finding 6) |
| broken embeddings for mixed-length batches | 1 run (a1) | baseline | eliminated by skill |
| endpoint never queried by the run | 1 run (a1) | baseline | eliminated by skill |
| check false negatives (valid alternative not recognised) | 6 baseline sub-checks, 1 with-skill (⚑ in `RESULTS.md`) | both | open — practical checks trialled; replicated checks being corrected ‡ |

No with-skill regression at task level.

## Verdict

**Ship gate met; ship with follow-ups.** On the standard pair the skill has at least one
task-level win and zero regressions in each gated group (replicated 1 win, new checks 2,
outcome 1), which is the skill-eval default gate. Follow-ups: re-state the replicated
numbers once the corrected checks land (‡), and confirm in the next round that the fixes
made after these runs (findings 1–6) hold — the tested skill content predates them.

## Limitations

- **Single samples.** One run per task per arm, except baseline teddy-001 (two attempts).
  Directional evidence, not statistical.
- **Checks derived from the skill's own content**, so they can favour it; known false
  negatives are marked ⚑ in `RESULTS.md`, and the practical trial was written after seeing
  these five notebooks.
- **Outcome evidence partly measured by hand before teardown** (the outcome tool then sent
  only the `teddy.md` request format and asked the index for columns it did not hold);
  every endpoint is deleted, so it cannot be re-measured.
- **Runs 4 and 5 (ws/002 and ws/001) overlapped and shared one skill install**, so no files
  were replaced under a running chat.
- **Baseline teddy-001 attempt 1** used resource names carrying a `_001_with_skill` suffix
  from the run guide before per-run names were final; the prompt was identical and the
  skill was off, so it is a valid second baseline sample. Attempt 2 is primary.
- **Instructions file: a temporary 3-line version** (179 bytes) was active in all five runs
  instead of the user's full file; b/002's setup record shows the full file because it was
  fingerprinted before the swap (resolved in `RESULTS.md`, "Environment").
- **Genie Code memories cleared** before each run. Built-in skills are always on and cannot
  be switched off, so the result is the skill's value on top of them.
- **Replication pair not run** (the Sep–Oct 2026 prompts in this environment).

## Reproduce

Run protocol, scorers and ship gate: [`README.md`](README.md). Re-check the committed
scores and the trial:

```bash
uv run --isolated --with pytest --with-requirements skills/open-weight-models/tests/requirements.txt \
    python -m pytest -q skills/open-weight-models/tests
uv run --isolated --with-requirements skills/open-weight-models/tests/requirements.txt \
    python skills/open-weight-models/eval/practical_checks.py
```

Score files: `models/teddy/runs_20261026/scores/`; run records (environment, outcome,
skill-use evidence): `models/teddy/runs_20261026/run_records.json`.

### Task IDs

| ID | Sep–Oct 2026 ID | Task |
|---|---|---|
| teddy-001 | oss-001 | TEDDY-70M deploy |
| teddy-002 | oss-002 | TEDDY-70M deploy + nearest-neighbour cell search |
| geneformer-001 | oss-005 | Geneformer V1-10M deploy (later) |
| geneformer-002 | oss-006 | Geneformer V2 BioNeMo deploy (later) |

## Sep–Oct 2026 (historical)

TEDDY runs of Sep 24 – Oct 7, 2026, on the skill before its rename from `oss-models`.
Kept as recorded; not the current evaluation. Raw sub-check counts; dates, detail and
caveats in [`runs_20260924/eval_report.md`](runs_20260924/eval_report.md). ‡ These may also
change when the corrected replicated checks re-score these runs.

| Task | Sub-checks | Baseline passed | With skill passed | Outcome |
|---|---|---|---|---|
| oss-001 | 27 | 19 | 27 | win |
| oss-002, first skill run | 30 | 26 | 27 | tie-fail (2 skill-only passes, 1 regression) |
| oss-002, after patching `teddy.md` | 30 | 26 | 30 | win |

Main caveats: the with-skill prompt told Genie Code to read `SKILL.md` and `teddy.md`; the
baseline notebooks showed both prompts and the scored topics; skill absence in the baseline
was not recorded; the second oss-002 run used a `teddy.md` patched with the first run's
failures; all on the pre-rename skill.

## Geneformer

**Pending.** Not evaluated yet (geneformer-001 / -002 planned). The Sep–Oct 2026 Geneformer
notebooks and the one with-skill output are kept, unscored, in `runs_20260924/`.
