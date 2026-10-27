# Evaluation Report — open-weight-models skill

**Status: retest pending.** The TEDDY tasks will be re-run under the protocol in
[`README.md`](README.md): built-in skills + this skill against built-in skills only, a
fresh chat per task per run, the skill installed without `eval/`, a recorded skill
list, and hidden codes showing which skill files Genie Code read. The score files
(`baseline_scores.json`, `with_skill_scores.json`, `with_skill_old_prompt_scores.json`)
appear after those runs.

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
