# Retest outputs

Filled in during the runs (see `../README.md`, run protocol):

- `run_records.json` — written by `../install_skill_for_run.py`, one record per (run, task).
- `<run_id>/<task_id>.py` — exported notebook source, for `run_id` in `baseline`,
  `with_skill`, `old_prompt_baseline`, `old_prompt_with_skill` and `task_id` in
  `teddy-001`, `teddy-002`.
- `<run_id>/backup/<task_id>-<timestamp>/` — local backup of an installed skill copy that
  `--replace-existing` deleted (not committed; see `.gitignore`).
- `<run_id>/<task_id>_skills_panel*.png` — Customizations → Skills panel (User and All
  skills tabs) before the run; `<run_id>/<task_id>_steps.png` — the chat's visible steps
  after the run.
- `scores/` — per-run score files split into the replicated set and the new checks,
  written by `../score_open_weight_models.py`.
