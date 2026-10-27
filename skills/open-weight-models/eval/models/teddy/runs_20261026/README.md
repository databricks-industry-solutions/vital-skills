# TEDDY runs, round of 2026-10-26

Filled in during the runs (see `../../../README.md`, run protocol). Run IDs: `baseline`,
`with_skill`, `old_prompt_baseline`, `old_prompt_with_skill`; task IDs: `teddy-001`,
`teddy-002`. Resource names per run: `../family.json` → `names`.

- `run_records.json` — written by `eval/install_skill_for_run.py`, one record per (run,
  task); outcome results added by `eval/check_outcome.py`.
- `<run_id>/<task_id>.py` — exported notebook source (`<task_id>_attempt<n>.py` when a
  (run, task) was repeated; its record has `"attempt": n`).
- `RESULTS.md` — results of this round.
- `<run_id>/<task_id>_skills_panel*.png` — Customizations → Skills panel (User and All
  skills tabs) before the run; `<run_id>/<task_id>_steps.png` — the chat's visible steps
  after the run.
- `<run_id>/backup/<task_id>-<timestamp>/` — local backup of an installed skill copy that
  `--replace-existing` deleted (not committed; see `.gitignore`).
- `scores/` — written by `eval/score_open_weight_models.py` (this round: by a local copy
  of the same steps, see `RESULTS.md`); an extra attempt as `<run_id>_attempt_<n>_…`:
  - `<run_id>_scores.json` — every metric of every task for that run;
  - `<run_id>_replicated.json` — the Sep–Oct 2026 checks;
  - `<run_id>_new_checks.json` — the new checks (`new_checks/…`);
  - `<run_id>_outcome.json` — the outcome group (`outcome/…`, measured before teardown);
  - `<run_id>_convention.json` — the shared-input-format convention check (`convention/…`);
  - `<run_id>_judge.json` — the optional guideline judge (`judge/guidelines`), only when
    `USE_GUIDELINE_JUDGE = True`.

## Deviations from plan

1. **`baseline`/teddy-001 attempt 1 (2026-10-27): resource names only.** The prompt is the
   standard teddy-001 request, identical word for word; only the resource names and
   download path differ — they carried the `_001_with_skill` suffix from the run
   guide before per-run names were finalised. The skill was off, so this is a valid
   baseline sample. The run used model
   `<catalog>.skills_owm_202610.owm_teddy_model_001_with_skill`, endpoint
   `owm_teddy_ep_001_with_skill` and files
   `/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/001_with_skill/`. Its record holds
   the actual names in `resources_created`, the pasted text in `prompt_used`, and a note.
   It was kept as attempt 1 (`baseline/teddy-001_attempt1.py`) and the run repeated with
   the `_001_baseline` names as attempt 2 (`baseline/teddy-001_attempt2.py`, the primary
   sample).
2. **`with_skill`/teddy-001 therefore uses suffix `skill_on`** (model
   `owm_teddy_model_001_skill_on`, endpoint `owm_teddy_ep_001_skill_on`, files
   `001_skill_on/`), set in `../family.json` → `run_suffix_overrides`, so it cannot clash
   with the names above. teddy-002 names are unchanged.
3. **Run order:** all three skill-off runs before both with-skill runs. Record setup
   times (UTC, `run_records.json` `timestamp`, written when the skill state was set):
   `baseline`/teddy-002 03:55, `baseline`/teddy-001 attempt 1 05:14, attempt 2 07:05,
   `with_skill`/teddy-002 07:54, `with_skill`/teddy-001 08:27. The first record (`old_prompt_baseline`/teddy-001, skill off)
   was a setup record for a run that never happened; it is marked discarded ("setup record
   only; run order changed").
4. **`with_skill`/teddy-002 and `with_skill`/teddy-001 overlapped** and shared one skill
   install (same hidden codes) so that no files were replaced under a running chat.
5. **Hidden codes were installed for both with-skill runs** (the default at the time).
   `with_skill`/teddy-002 wrote the `teddy.md` code into its notebook; `with_skill`/teddy-001
   declined the marker as possibly injected. Hidden codes are now off by default and are
   not used as proof of skill use (`eval/README.md`, step 1).
6. **Outcome measured partly by hand before teardown.** `check_outcome.py` then sent only
   the `teddy.md`-format request, which the baselines' own (valid) formats rejected, and
   its nearest-neighbour query asked for table columns the with-skill index did not hold.
   The orchestrator measured the affected checks by hand before teardown; those results
   are in each record's `"outcome"` with their provenance, and the original tool output is
   kept under `"outcome_superseded"`. Both tool problems are fixed in `check_outcome.py`.
