# Evaluation Report — open-weight-models skill

One section per model family. Protocol, scorers and ship gate: [`README.md`](README.md).

**Baseline environment (all families):** Genie Code as shipped — built-in skills always
on and not visible in the UI — with no personal or workspace skills, plus the user's
instructions file (fingerprinted per run). The June AI Dev Kit user skills present during
the Sep 24 – Oct 7, 2026 runs were removed before the reruns, for both pairs.

## TEDDY

**Status: standard pair run on 2026-10-27 — results draft for review:**
[`models/teddy/runs_20261026/RESULTS.md`](models/teddy/runs_20261026/RESULTS.md).
Built-ins + this skill against built-ins only, one fresh chat per task per run, the skill
installed without `eval/`, a recorded Skills panel, self-report and visible steps. Every
run created its own resources under schema `skills_owm_202610` (names in
`models/teddy/family.json`). The replication pair (Sep–Oct 2026 prompts) was not run.

Failed checks per run (raw counts; the shared-input-format convention is shown apart):

| Task | Run | Critical failed | Major failed | Minor failed | Outcome passed | Shared input format |
|---|---|---|---|---|---|---|
| teddy-001 | baseline (attempt 2) | 5 | 6 | 2 | no | no |
| teddy-001 | baseline (attempt 1) | 3 | 8 | 2 | no | no |
| teddy-001 | with_skill | 0 | 0 | 0 | yes | yes |
| teddy-002 | baseline | 4 | 5 | 2 | yes | no |
| teddy-002 | with_skill | 0 | 0 | 1 | yes | yes |

Several baseline failures are false negatives of checks written from the skill's own
content; `RESULTS.md` lists them, the outcome evidence, the skill-use evidence and the
caveats (one sample per arm except teddy-001 baseline; outcome partly measured by hand
before teardown).

### Task IDs

| Retest ID | Sep–Oct 2026 ID | Task |
|---|---|---|
| teddy-001 | oss-001 | TEDDY-70M deploy |
| teddy-002 | oss-002 | TEDDY-70M deploy + nearest-neighbour cell search |
| geneformer-001 | oss-005 | Geneformer V1-10M deploy (later) |
| geneformer-002 | oss-006 | Geneformer V2 BioNeMo deploy (later) |

### Previous runs (Sep 24 – Oct 7, 2026), historical

Raw sub-check counts from the replicated checks; dates, full detail and caveats in
[`runs_20260924/eval_report.md`](runs_20260924/eval_report.md).

| Task | Sub-checks | Baseline passed | With skill passed | Outcome |
|---|---|---|---|---|
| oss-001 | 27 | 19 | 27 | win |
| oss-002, first skill run | 30 | 26 | 27 | tie-fail (2 skill-only passes, 1 regression) |
| oss-002, after patching `teddy.md` | 30 | 26 | 30 | win |

Main caveats: the with-skill prompt told Genie Code to read `SKILL.md` and `teddy.md`;
the baseline notebooks showed both prompts and the scored topics; skill absence in the
baseline was not recorded; the second oss-002 run used a `teddy.md` patched with the
first run's failures; all on the pre-rename skill.

### Verdict

None yet — the 2026-10-27 results are a draft pending review (`RESULTS.md`).

### Limitations

Two tasks, one run each per arm (two baseline attempts for teddy-001): directional
evidence, not statistical.

## Geneformer

Not evaluated yet (geneformer-001 / -002 planned). The Sep–Oct 2026 Geneformer notebooks
and the one with-skill output are kept, unscored, in `runs_20260924/`.
