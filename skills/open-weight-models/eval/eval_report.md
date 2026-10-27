# Evaluation Report — open-weight-models skill

One section per model family. Protocol, scorers and ship gate: [`README.md`](README.md).

**Baseline environment (all families):** Genie Code as shipped — built-in skills always
on and not visible in the UI — with no personal or workspace skills, plus the user's
instructions file (fingerprinted per run). The June AI Dev Kit user skills present during
the Sep 24 – Oct 7, 2026 runs were removed before the reruns, for both pairs.

## TEDDY

**Status: retest pending** (round `models/teddy/runs_20261026/`). Built-in skills + this
skill against built-in skills only, a fresh chat per task per run, the skill installed
without `eval/`, a recorded Skills panel, and hidden codes showing which skill files
Genie Code read. Every run creates its own resources under schema `skills_owm_202610`
(names in `models/teddy/family.json`). Two pairs:

- **standard** — `baseline` vs `with_skill`, the `evalset.json` query with the run's
  names: the evaluation and the ship gate. Run first: teddy-002 (the MVP), then teddy-001;
- **replication** — `old_prompt_baseline` vs `old_prompt_with_skill`, the recorded
  Sep–Oct 2026 prompts with this round's names: the replication of the results below
  (optional).

Each run is scored in groups — the replicated Sep–Oct 2026 checks, the new checks, and an
outcome group measured on the live deployment before teardown (endpoint ready, valid
512-dimension embeddings for a fixed request, working search for teddy-002) — and every
failed check is counted by severity (critical / major / minor) next to whether the run's
outcome passed. Score files appear in `models/teddy/runs_20261026/scores/` after the runs.

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

None yet — pending the retest.

### Limitations

Two tasks, one run each per arm: directional evidence, not statistical.

## Geneformer

Not evaluated yet (geneformer-001 / -002 planned). The Sep–Oct 2026 Geneformer notebooks
and the one with-skill output are kept, unscored, in `runs_20260924/`.
