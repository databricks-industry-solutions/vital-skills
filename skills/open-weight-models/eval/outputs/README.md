# Retest outputs

Filled in during the runs (see `../README.md`, run protocol):

- `run_records.json` — written by `../install_skill_for_run.py`, one record per (run, task).
- `<run_id>/<task_id>.py` — exported notebook source, for `run_id` in `baseline`,
  `with_skill`, `with_skill_old_prompt` and `task_id` in `teddy-001`, `teddy-002`.
- `<run_id>/<task_id>_skills.png` — screenshot of the `@` skill list before the run.
- `scores/` — per-run score files split into the replicated set and the new checks,
  written by `../score_open_weight_models.py`.
