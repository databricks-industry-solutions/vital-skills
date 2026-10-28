# Evaluation Report — open-weight-models skill, previous runs (Sep–Oct 2026), historical

**Status: historical record.** These runs were made between 2026-09-24 and 2026-10-07 against the skill
before it was renamed from `oss-models` to `open-weight-models`. They are kept as
recorded and are **not** the current evaluation; the retest is in [`../`](../README.md).

**Setup**: 2 TEDDY tasks, each run once with and once without the skill in Genie Code.
Scored with the deterministic phase checks in `scorers.py` (8 groups of text checks on
the exported notebook; no LLM judge), compared with `compare_runs.py`. Score files:
`baseline_scores.json`, `with_skill_scores.json`. Per-task reports: `reports/`.
The sub-checks themselves (added 2026-10-28): the rows counted as *replicated* in
[`../CHECKS.md`](../CHECKS.md) — 27 for oss-001 and 30 for oss-002 (the search group applies to
oss-002 only). That list shows each one with its severity, what it guards and, where the logic
used here matched literal text, how the corrected version judges it.

## When

| Run | Notebook created |
|---|---|
| oss-001 baseline and with-skill | 2026-09-24 |
| oss-002 baseline and first with-skill run | 2026-10-01 |
| oss-002 second with-skill run ("v2") | 2026-10-01 to 2026-10-07 |
| oss-005 with-skill (Geneformer, not scored) | 2026-10-03 |

Dates are the recorded workspace creation dates of the run notebooks. The per-task
reports were generated on 2026-10-01 (`reports/`).

## Task IDs

The retest renames the tasks. Files in this folder keep the old IDs.

| Sep–Oct 2026 ID | ID | Task |
|---|---|---|
| oss-001 | teddy-001 | TEDDY-70M deploy |
| oss-002 | teddy-002 | TEDDY-70M deploy + nearest-neighbour cell search |
| oss-005 | geneformer-001 | Geneformer V1-10M deploy (not scored, see below) |
| oss-006 | geneformer-002 | Geneformer V2 BioNeMo deploy (prompt only, never run) |

## Results (raw sub-check counts)

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

## Caveats

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
   (`results/with_skill/oss-005.txt`) but no baseline and no scores; oss-006 was never
   run. Neither is in the results above.

## Verdict

Historical evidence that the skill's TEDDY content helped when Genie Code was told to
read it. Not a ship decision for the current skill — see the retest.

## Reproduce

`tests/test_eval_scorers.py` re-scores the recorded notebooks in `results/` with this
folder's `scorers.py` and checks they match the recorded score files.
