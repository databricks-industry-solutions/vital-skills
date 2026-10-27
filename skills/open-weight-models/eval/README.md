# open-weight-models — TEDDY retest

Paired evaluation of the `open-weight-models` skill on two TEDDY tasks (teddy-001,
teddy-002). Geneformer tasks come later.

**What is compared:** Genie Code with its built-in skills **plus** this skill, against
Genie Code with its built-in skills **only**. Built-in skills cannot be disabled, so the
result is the extra value of this skill on top of them.

**Three runs per task:**

| Run (`run_id`) | Skill | What is pasted into the chat |
|---|---|---|
| `baseline` | off | the task's `query` from `evalset.json` |
| `with_skill` | on | the same `query`, nothing else |
| `with_skill_old_prompt` | on | the task's block from `old_prompt_with_skill.md` |

`with_skill` tests whether Genie Code finds and uses the skill on its own.
`with_skill_old_prompt` exists only to replicate the 2026-10 results: that prompt told
Genie Code to read `SKILL.md` and `teddy.md` before writing code. The 2026-10 record is
in [`previous_runs_2026-10/`](previous_runs_2026-10/README.md).

## Files

| File | What it is | Who may read it during a run |
|---|---|---|
| `evalset.json` | task requests: `task_id`, model, variant, capability, `dataset`, `query` | the person running the eval |
| `answer_sheet.md` | fixed replies if Genie Code asks a question | the person running the eval |
| `old_prompt_with_skill.md` | 2026-10 forced-read prompts (`with_skill_old_prompt` only) | the person running the eval |
| `install_skill_for_run.py` | laptop script: skill off/on + run record | the person running the eval |
| `expectations.json` | answer key: difficulty, expected facts, checks, new checks | scoring only |
| `scorers.py` | MLflow scorers (exports `scorers`) | scoring only |
| `score_open_weight_models.py` | Databricks scoring notebook | scoring only |
| `outputs/<run_id>/<task_id>.py` | exported notebooks (after the runs) | scoring only |
| `outputs/run_records.json` | one record per run: skill state, folder listings, hidden codes, commit | scoring only |
| `<run_id>_scores.json`, `outputs/scores/` | per-task scores (after scoring) | — |
| `eval_report.md` | results | — |
| `README_environments.md` | known Genie Code runtime snapshots | — |

Genie Code must never see `eval/`. The skill is installed by
`install_skill_for_run.py`, which copies only `SKILL.md` and `references/` (plus
`assets/` and `scripts/` if they exist). Do not copy the repo to the workspace until all
runs are done (step 7).

## Run protocol

Order: all `baseline` runs first, then `with_skill`, then `with_skill_old_prompt`, so
nothing about the skill is in a chat or memory before the baseline is done. One task at
a time, one fresh chat per task per run.

Checklist for **each** (run, task):

1. **Set the skill off or on** from the repo root:
   ```bash
   python3 skills/open-weight-models/eval/install_skill_for_run.py --profile <profile> \
       --user <workspace-user> --state off|on --run-id <run_id> --task-id <task_id>
   ```
   `off` for `baseline`, `on` for both with-skill runs. Add `--dry-run` first to see
   the commands. The script stops if another copy of the skill (for example an old
   `oss-models` folder) is installed; move or delete it and run again. It appends a
   record to `outputs/run_records.json`; for `on` it prints the two hidden codes.
2. **Clear memory.** In Genie Code, ask it to delete any memories about this
   evaluation, TEDDY, or the open-weight-models skill. Close that chat.
3. **Check the skill list.** Hard refresh the browser (Cmd+Shift+R), type `@` in a new
   chat, and screenshot the list of skills. Save it as
   `outputs/<run_id>/<task_id>_skills.png`. For `baseline`, open-weight-models must not
   be listed; for the with-skill runs it must be.
4. **Fresh notebook and chat.** Create a blank notebook in the run folder
   (`/Workspace/Users/<workspace-user>/teddy-eval/<run_id>/<task_id>`) and open a new
   Genie Code chat on it.
5. **Paste the prompt verbatim** — the `query` from `evalset.json`, or for
   `with_skill_old_prompt` the block from `old_prompt_with_skill.md` (placeholders
   filled). Answer questions **only** from `answer_sheet.md`. Do not steer, hint or
   correct. Let it run to the end.
6. **Export and record.** Export the notebook as source to
   `outputs/<run_id>/<task_id>.py`:
   ```bash
   databricks workspace export /Workspace/Users/<workspace-user>/teddy-eval/<run_id>/<task_id> \
       --format SOURCE --file skills/open-weight-models/eval/outputs/<run_id>/<task_id>.py -p <profile>
   ```
   Then check the hidden codes:
   ```bash
   python3 skills/open-weight-models/eval/install_skill_for_run.py --run-id <run_id> \
       --task-id <task_id> --check-notebook skills/open-weight-models/eval/outputs/<run_id>/<task_id>.py
   ```
   Note anything unusual (questions asked and answers given, crashes, how long it ran)
   in the record's entry in `outputs/run_records.json` under `"notes"`.
7. **Only after all six runs:** copy the repo to a workspace folder **outside**
   `.assistant/skills/` (for example a Git folder at
   `/Workspace/Users/<workspace-user>/<repo-folder>`), set the paths at the top of
   `score_open_weight_models.py`, and run it.

A with-skill run whose notebook shows neither code did not pick up the skill. Keep it
and report it — that is a result, not a failed run.

## Scorers

All checks are deterministic text checks on the exported notebook (comments and
markdown are ignored where noted in `scorers.py`). No LLM judge. Each sub-check is its
own pass/fail metric.

**Replicated set** — the 2026-10 checks, logic unchanged, so the counts can be put next
to the 2026-10 counts. 8 groups: `download_staging` (temp path, XET off, pinned
`transformers`), `pyfunc_quality` (PyFunc class, imports, `load_context`, `sys.modules`
purge, `io.StringIO`, `artifacts=`, real gene IDs from `vocab.txt`),
`model_registration` (signature, input example, pip requirements, 70M variant),
`endpoint_deployment` (workload-type enum, AI Gateway inference table, scale to zero,
run gate), `smoke_tests` (registration, endpoint query, output shape), `ai_search`
(teddy-002 only: index, embedding dimension, skip if the table is missing),
`edge_handling` (not applicable to TEDDY), `forbidden_patterns` (`/local_disk0`,
`transformers>=`, fake gene IDs, deprecated `AutoCaptureConfigInput`, string
workload type). teddy-001 has 27 sub-checks (+2 not-applicable groups), teddy-002 has
30 (+1). `tests/test_eval_scorers.py` checks that these reproduce the recorded
2026-10 scores from the recorded 2026-10 notebooks. Metric names match the 2026-10
files except forbidden-pattern names, which MLflow does not allow as written: use
`scorers.mlflow_metric_name()` on an old key to get the new name.

**New checks** (metric names start with `new_checks/`, scored and compared separately):

| Check | Passes when the notebook code… | Tasks |
|---|---|---|
| `ai_gateway/inference_table`, `ai_gateway/usage_tracking` | sets both on the endpoint's AI Gateway (`AiGatewayInferenceTableConfig` + `AiGatewayUsageTrackingConfig`, or the same fields in a REST body / `put_ai_gateway`) | both |
| `uc_provenance/uc_registration`, `license_tag`, `hf_revision_tag` | registers to Unity Catalog and records the license and the Hugging Face revision as tags | both |
| `existing_resources/endpoint` (`/table`, `/index` for teddy-002) | looks for an existing endpoint / table / index before the first create or delete call | both |

## Scoring and ship gate

`score_open_weight_models.py` evaluates each (run, task) with `mlflow.genai.evaluate`,
extracts scores with `compare_runs.extract_scores()`, writes `<run_id>_scores.json`
(all metrics) and `outputs/scores/<run_id>_{replicated,new_checks}.json`, and runs
`compare_runs.main()` four times: baseline vs `with_skill` and baseline vs
`with_skill_old_prompt`, each for the replicated set and for the new checks.

A task passes a set only if every sub-check in that set passes. Ship gate (skill-eval
default): at least one task-level win and zero regressions, judged on `with_skill`
against `baseline`, replicated set and new checks each. `with_skill_old_prompt` is
reported for comparison with 2026-10, not used for the gate. Two tasks show direction,
not statistical significance; report raw sub-check counts, never percentages.

## Reproducing locally

```bash
uv run --isolated --with pytest --with-requirements skills/open-weight-models/tests/requirements.txt \
    python -m pytest -q skills/open-weight-models/tests
```
