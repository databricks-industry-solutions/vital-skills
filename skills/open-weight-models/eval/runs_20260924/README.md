# Previous runs (Sep 24 – Oct 7, 2026) — frozen record

Folder `runs_20260924/` (named for the first run date; it was `tests/`, then
`previous_runs_2026-10/`). The current round is under `../models/teddy/runs_20261026/`.

The first evaluation of this skill, made between 2026-09-24 and 2026-10-07 when it was
still called `oss-models`. **Frozen: do not edit these files.** Recorded evidence keeps its old task
IDs (`oss-001`, `oss-002`, …); the ID mapping and results are in
[`eval_report.md`](eval_report.md). The harness for the current retest lives one level up, in
[`../`](../README.md).

## What was run

| Run | Notebook created |
|---|---|
| oss-001 baseline and with-skill | 2026-09-24 |
| oss-002 baseline and first with-skill run | 2026-10-01 |
| oss-002 second with-skill run ("v2") | 2026-10-01 to 2026-10-07 |
| oss-005 with-skill (Geneformer, not scored) | 2026-10-03 |

Dates are the recorded workspace creation dates of the run notebooks. The per-task
reports were generated on 2026-10-01 (`reports/`).

- **Tasks:** oss-001 (TEDDY-70M deploy) and oss-002 (TEDDY-70M deploy + cell search),
  one baseline and one with-skill run each, plus a second with-skill run of oss-002
  ("v2") after `teddy.md` was patched. oss-005 (Geneformer V1) has a with-skill output
  only; oss-006 (Geneformer V2 BioNeMo) was prepared but never run.
- **Skill off/on:** by renaming `SKILL.md` to `SKILL.md.off` in a project copy of the
  skill (`<project-folder>/.assistant/skills/oss-models/`).
- **Prompts:** each arm had its own prompt, pasted from the first cell of its notebook.
  The with-skill prompt told Genie Code to read `SKILL.md` and the model reference
  first ("forced read"); the baseline prompt did not. Both asked for named volumes,
  models and endpoints, a re-runnable notebook, and an export of the notebook source.
- **Scoring:** `score_open_weight_models_20260924.py` (then `03_eval_rubric_and_compare`)
  scored the exported notebooks with `scorers.py` and compared them with
  `compare_runs.py`.

Caveats (forced-read prompt, non-blank baseline notebooks, unrecorded skill absence,
v2 patched after v1, pre-rename skill) are listed in [`eval_report.md`](eval_report.md).

## Files

Standard eval layout, built from the old `tests/` files. Every file is the recorded
original, moved or renamed only, except the four files marked *split*, *new* or
*scrubbed*.

| File | What it is |
|---|---|
| `evalset.json`, `expectations.json` | *split*: the request and answer-key fields of oss-001, oss-002 and oss-005 from `evalset_combined_original.json`, values identical |
| `evalset_combined_original.json` | the original combined evalset (v3.0.0, 9 tasks — also the task backlog) |
| `scorers.py` | the Sep–Oct 2026 scorers, unchanged |
| `compare_runs.py` | the Sep–Oct 2026 copy of the skill-eval comparison script, unchanged |
| `score_open_weight_models_20260924.py` | the Sep–Oct 2026 scoring notebook (was `03_eval_rubric_and_compare.py`), unchanged |
| `baseline_scores.json`, `with_skill_scores.json` | recorded scores (were `results/baseline_scores.json`, `results/skill_scores.json`) |
| `reports/oss-00{1,2}_report.md` | per-task reports written by the scoring notebook |
| `eval_report.md`, `README.md` | *new*: summary and this file |
| `results/{baseline,with_skill,with_skill_v2}/*.txt` | exported notebook sources that were scored; `manifest.json` run index; `oss-002_teddy_md_v1_diff.md` (the v1→v2 `teddy.md` patches) |
| `oss-001_*`, `oss-002_*` | the notebook pairs, as prepared (prompt cell + skill check) |
| `oss-005_*`, `oss-006_*`, `results/with_skill/oss-005.txt` | *scrubbed*: Geneformer notebooks and the oss-005 with-skill output, added from a workspace backup; only the user name, project folder and catalog were replaced by `<workspace-user>`, `<project-folder>`, `<catalog>` |
| `00_oss_models_test_plan.py`, `02_eval_prompt_templates.py` | test plan and prompt library (the latter includes later prompt edits not used in the recorded runs) |
| `_ground_truth/` | reference implementation notebooks |
| `README_20260924_protocol.md` | the original `tests/README.md` |

## The old scoring notebook no longer runs as is

`score_open_weight_models_20260924.py` resolves paths from the folder it runs in, so it
still imports this folder's `scorers.py` and `compare_runs.py` and never points at
`tests/`. But it expects the Sep–Oct 2026 workspace layout, which this folder no longer has:

- `evalset.json` in the combined `{"tasks": [...]}` form (now `evalset_combined_original.json`; `evalset.json` is the split list);
- `results/baseline_scores.json` and `results/skill_scores.json` (now `baseline_scores.json`, `with_skill_scores.json`);
- `results/reports/` (now `reports/`);
- the skill under `.assistant/skills/open-weight-models/` next to it.

Its logic was not edited. To re-check the recorded scores, run
`tests/test_eval_scorers.py`, which re-scores `results/` with this `scorers.py`.

## Note added 2026-10-27: these checks were flawed; corrected re-scores alongside

Everything above, and every file listed in "Files", is kept exactly as recorded. This
note is the only addition.

The Sep–Oct 2026 checks in `scorers.py` match literal text instead of judging their goal,
so the recorded numbers are partly wrong in both directions. Examples from these runs:
`graceful_skip` passed both oss-002 notebooks because the task prompt, quoted in the first
cell, contains "table … not … exist"; `hf_xet_disabled` passed the oss-002 baseline because
its development log says `HF_HUB_DISABLE_XET` was "Not set"; `real_gene_ids` passed the
oss-001 baseline because `vocab.txt` is in a download file list, while its input example is
hand-written token IDs. The full list of flawed sub-checks and the corrected rules are in
[`../README.md`](../README.md) ("Corrected set").

- **Recorded numbers are kept as reported** (`baseline_scores.json`,
  `with_skill_scores.json`, `reports/`, `eval_report.md`); `tests/test_eval_scorers.py`
  still proves they came from this `scorers.py`.
- **Corrected re-scores are provided alongside**, as new files in `scores_corrected/`
  (`baseline_corrected.json`, `with_skill_corrected.json`), written by
  `../corrected_checks.py --sep-oct --write` from the same `results/<arm>/<task>.txt`
  exports. These runs have no run records, so only their code could be judged.

Replicated sub-checks passed, as originally reported → re-scored with corrected logic:

| Task | Baseline | With skill |
|---|---|---|
| oss-001 (teddy-001) | 21 / 29 → 18 / 29 | 29 / 29 → 29 / 29 |
| oss-002 (teddy-002) | 27 / 31 → 23 / 31 | 31 / 31 → 29 / 31 (the "v2" run) |

Counts as in the score files (a not-applicable group counts as one pass). The direction is
unchanged: the with-skill runs pass more sub-checks than the baselines on both tasks.
