# Evaluation Report — open-weight-models skill

**Status: retest pending.** The TEDDY tasks will be re-run under the protocol in
[`README.md`](README.md): built-in skills + this skill against built-in skills only, a
fresh chat per task per run, the skill installed without `eval/`, a recorded Skills
panel, and hidden codes showing which skill files Genie Code read. Two pairs:

- **standard** — `baseline` vs `with_skill`, plain `evalset.json` query: the evaluation
  and the ship gate;
- **replication** — `old_prompt_baseline` vs `old_prompt_with_skill`, the recorded
  2026-10 prompts: the replication of the results below.

**Baseline environment:** Genie Code as shipped — built-in skills always on and not
visible in the UI — with no personal or workspace skills, plus the user's instructions
file (fingerprinted per run). The June AI Dev Kit user skills present during the 2026-10
runs were removed before the reruns, for both pairs.

Each run is scored in groups — the replicated 2026-10 checks, the new checks, and an
outcome group measured on the live deployment before teardown (endpoint ready, valid
512-dimension embeddings for a fixed request, working search for teddy-002) — and every
failed check is counted by severity (critical / major / minor) next to whether the run's
outcome passed.

The score files (`baseline_scores.json`, `with_skill_scores.json`,
`old_prompt_baseline_scores.json`, `old_prompt_with_skill_scores.json`) appear after
those runs.

## Task IDs

| Retest ID | 2026-10 ID | Task |
|---|---|---|
| teddy-001 | oss-001 | TEDDY-70M deploy |
| teddy-002 | oss-002 | TEDDY-70M deploy + nearest-neighbour cell search |
| geneformer-001 | oss-005 | Geneformer V1-10M deploy (later) |
| geneformer-002 | oss-006 | Geneformer V2 BioNeMo deploy (later) |

## Previous runs (2026-10), historical

Raw sub-check counts from the replicated checks; full detail and caveats in
[`previous_runs_2026-10/eval_report.md`](previous_runs_2026-10/eval_report.md).

| Task | Sub-checks | Baseline passed | With skill passed | Outcome |
|---|---|---|---|---|
| oss-001 | 27 | 19 | 27 | win |
| oss-002, first skill run | 30 | 26 | 27 | tie-fail (2 skill-only passes, 1 regression) |
| oss-002, after patching `teddy.md` | 30 | 26 | 30 | win |

Main caveats: the with-skill prompt told Genie Code to read `SKILL.md` and `teddy.md`;
the baseline notebooks showed both prompts and the scored topics; skill absence in the
baseline was not recorded; the second oss-002 run used a `teddy.md` patched with the
first run's failures; all on the pre-rename skill.

## Verdict

None yet — pending the retest.

## Limitations

Two tasks, one run each per arm: directional evidence, not statistical.
