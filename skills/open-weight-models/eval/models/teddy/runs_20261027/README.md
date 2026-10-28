# TEDDY runs, round 2 (2026-10-27/28)

A clean rerun of all four runs of round 1 (`../runs_20261026/`), with the updated skill. Same
layout as round 1 (see its README); tools take `--round runs_20261027`. Run IDs: `baseline`,
`with_skill`; task IDs: `teddy-001`, `teddy-002`. Resource names end in `_r2`; the prompts are in
each record's `prompt_used`.

- `run_records.json` — one record per (run, task), written by `eval/install_skill_for_run.py`,
  outcome results added by `eval/check_outcome.py`. Each record names its export in
  `export_file`. One record (with_skill / teddy-002, attempt 1) is discarded, with its reason.
- `<run_id>/<task_id>.py` — exported notebook source, after the run.
- `<run_id>/<task_id>_before_go.py` — the same notebook exported just before the user's
  `Yes, go ahead.` (only for runs that stopped to ask), so the saved defaults before the
  go-ahead are on record.
- `with_skill/teddy-002_discarded_attempt1.py` — the discarded attempt, kept for the record and
  not scored.
- `scores/` — written by the steps of `eval/score_open_weight_models.py`, run on a laptop with a
  local MLflow store (see `RESULTS.md`, "How scored"). Same files as round 1:
  `<run_id>_scores.json`, `_corrected.json` (counted), `_replicated.json` (original Sep–Oct
  logic, ⚑ only), `_new_checks.json`, `_outcome.json`, `_convention.json`.
- `RESULTS.md` — results of this round.

Screenshots (Skills panel, compute, chat steps) were shared in the chat during the runs and are
described in each record (`skills_panel`, `visible_steps`); they are not stored here.
