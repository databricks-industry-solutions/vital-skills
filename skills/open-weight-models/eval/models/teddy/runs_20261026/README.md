# TEDDY runs, round of 2026-10-26

Filled in during the runs (see `../../../README.md`, run protocol). Run IDs: `baseline`,
`with_skill`, `old_prompt_baseline`, `old_prompt_with_skill`; task IDs: `teddy-001`,
`teddy-002`. Resource names per run: `../family.json` → `names`.

- `run_records.json` — written by `eval/install_skill_for_run.py`, one record per (run,
  task); outcome results added by `eval/check_outcome.py`.
- `<run_id>/<task_id>.py` — exported notebook source.
- `<run_id>/<task_id>_skills_panel*.png` — Customizations → Skills panel (User and All
  skills tabs) before the run; `<run_id>/<task_id>_steps.png` — the chat's visible steps
  after the run.
- `<run_id>/backup/<task_id>-<timestamp>/` — local backup of an installed skill copy that
  `--replace-existing` deleted (not committed; see `.gitignore`).
- `scores/` — written by `eval/score_open_weight_models.py`:
  - `<run_id>_scores.json` — every metric of every task for that run;
  - `<run_id>_replicated.json` — the Sep–Oct 2026 checks;
  - `<run_id>_new_checks.json` — the new checks (`new_checks/…`);
  - `<run_id>_outcome.json` — the outcome group (`outcome/…`, measured before teardown);
  - `<run_id>_judge.json` — the optional guideline judge (`judge/guidelines`), only when
    `USE_GUIDELINE_JUDGE = True`.
