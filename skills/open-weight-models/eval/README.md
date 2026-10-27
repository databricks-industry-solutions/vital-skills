# open-weight-models — TEDDY retest

Paired evaluation of the `open-weight-models` skill on two TEDDY tasks (teddy-001,
teddy-002). Geneformer tasks come later.

**What is compared:** Genie Code with its built-in skills **plus** this skill, against
Genie Code with its built-in skills **only**. Built-in skills cannot be disabled, so the
result is the extra value of this skill on top of them.

**Four runs per task, in two pairs.** Each pair is compared only within itself.

| Pair | Run (`run_id`) | Skill | What is pasted into the chat |
|---|---|---|---|
| standard | `baseline` | off | the task's `query` from `evalset.json` |
| standard | `with_skill` | on | the same `query`, nothing else |
| replication | `old_prompt_baseline` | off | the task's block from `old_prompt_baseline.md` |
| replication | `old_prompt_with_skill` | on | the task's block from `old_prompt_with_skill.md` |

- The **standard pair** is the evaluation: does Genie Code find and use the skill on
  its own, from a plain request?
- The **replication pair** is the replication of the 2026-10 results. It reuses the
  2026-10 prompts, as recorded: the baseline prompt and the with-skill prompt that told
  Genie Code to read `SKILL.md` and `teddy.md` first. Only the skill file paths and the
  export path were changed (listed at the top of each file). The 2026-10 record is in
  [`previous_runs_2026-10/`](previous_runs_2026-10/README.md).

**Environment per pair.** The replication pair is run with the user's skills as already
installed (as in 2026-10). The standard pair may be run after the user has removed
deprecated copies of other skills (for example old AI Dev Kit copies). Every run record
therefore stores the environment Genie Code could see (step 1), and each pair is compared
only within itself — never a run of one pair against a run of the other.

> [!WARNING]
> **The replication prompts replace the 2026-10 resources.** They name the same volume,
> Unity Catalog model and endpoint as the 2026-10 runs, and tell Genie Code to delete
> any pre-existing endpoint and model versions on the first run. Running them in the
> workspace and catalog used in 2026-10 deletes or overwrites those resources. Use
> another catalog for `<catalog>`, or accept that before pasting.

## Files

| File | What it is | Who may read it during a run |
|---|---|---|
| `evalset.json` | task requests: `task_id`, `dataset`, `query` only | the person running the eval |
| `answer_sheet.md` | fixed replies if Genie Code asks a question | the person running the eval |
| `old_prompt_baseline.md`, `old_prompt_with_skill.md` | 2026-10 prompts (replication pair only) | the person running the eval |
| `install_skill_for_run.py` | laptop script: skill off/on + run record | the person running the eval |
| `expectations.json` | answer key: difficulty, model family, variant, capability, expected facts, checks, new checks | scoring only |
| `scorers.py` | MLflow scorers (exports `scorers`) | scoring only |
| `score_open_weight_models.py` | Databricks scoring notebook | scoring only |
| `outputs/<run_id>/<task_id>.py` | exported notebooks (after the runs) | scoring only |
| `outputs/run_records.json` | one record per run: skill state, environment, hidden codes, backup, commit, discarded / reason, notes | scoring only |
| `<run_id>_scores.json`, `outputs/scores/` | per-task scores (after scoring) | — |
| `eval_report.md` | results | — |
| `README_environments.md` | known Genie Code runtime snapshots | — |

Genie Code must never see `eval/`. The skill is installed by
`install_skill_for_run.py`, which copies only `SKILL.md` and `references/` (plus
`assets/` and `scripts/` if they exist). Do not copy the repo to the workspace until all
runs are done (step 7).

## Run protocol

Order: both skill-off runs first (`baseline`, then `old_prompt_baseline`), then
`with_skill`, then `old_prompt_with_skill`, so nothing about the skill is in a chat or
memory before the skill-off runs are done. One task at a time, one fresh chat per task
per run.

Each run's notebooks live in its own run folder:
`/Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/<run_id>/`
(the replication prompts also export there).

Checklist for **each** (run, task):

1. **Set the skill off or on** from the repo root:
   ```bash
   python3 skills/open-weight-models/eval/install_skill_for_run.py --profile <profile> \
       --user <workspace-user> --state off|on --run-id <run_id> --task-id <task_id>
   ```
   `off` for the two skill-off runs, `on` for the two with-skill runs. Add `--dry-run`
   first to see the commands. For `on`, the script first builds and checks the copy
   locally (hidden codes inserted, no `eval/` or `tests/`), then installs it.
   - It stops if another copy of the skill (for example an old `oss-models` folder) is
     installed; move or delete it and run again.
   - It stops if `.assistant/skills/open-weight-models` already exists — for example
     the copy from the previous with-skill task, or the user's own install before the
     eval. To replace it, add `--replace-existing`: the script exports the existing copy
     to `outputs/<run_id>/backup/<task_id>-<timestamp>/` (not committed), then deletes
     it, then installs the new copy. If creating the folder or the import fails, it
     prints the commands that restore the backup.
   - It appends a record to `outputs/run_records.json` with the environment Genie Code
     will see: the user skills folder (each `SKILL.md` with size and sha256), the
     sha256 of `/Workspace/Users/<workspace-user>/.assistant_instructions.md` if present,
     and whether `/Workspace/.assistant/skills` and
     `/Workspace/.assistant_workspace_instructions.md` exist. For `on` it prints the two
     hidden codes.
2. **Clear memory.** In Genie Code, ask it to delete any memories about this
   evaluation, TEDDY, or the open-weight-models skill. Close that chat.
3. **Check the skill list.** Hard refresh the browser (Cmd+Shift+R), type `@` in a new
   chat, and screenshot the list of skills. Save it as
   `outputs/<run_id>/<task_id>_skills.png`. For skill-off runs, open-weight-models must
   not be listed; for with-skill runs it must be.
4. **Fresh notebook and chat.** Create a blank notebook `<task_id>` in the run folder
   and open a new Genie Code chat on it.
5. **Paste the prompt verbatim** — the `query` from `evalset.json` for the standard
   pair, or the task's block from `old_prompt_baseline.md` / `old_prompt_with_skill.md`
   for the replication pair (placeholders filled). Answer questions **only** from
   `answer_sheet.md`. Do not steer, hint or correct. Let it run to the end.
6. **Export and record.** Export the notebook as source to
   `outputs/<run_id>/<task_id>.py`:
   ```bash
   databricks workspace export \
       /Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/<run_id>/<task_id> \
       --format SOURCE --file skills/open-weight-models/eval/outputs/<run_id>/<task_id>.py -p <profile>
   ```
   Then check the hidden codes:
   ```bash
   python3 skills/open-weight-models/eval/install_skill_for_run.py --run-id <run_id> \
       --task-id <task_id> --check-notebook skills/open-weight-models/eval/outputs/<run_id>/<task_id>.py
   ```
   Write anything unusual (questions asked and answers given, crashes, how long it ran)
   into that run's `"notes"` field in `outputs/run_records.json`.
7. **Check for leaks.** Read the chat's visible steps. If Genie Code opened, searched or
   listed any file or folder whose path contains `oss-0`, `gt_`, `hls-skills-tests`,
   `previous_runs_2026-10` or `/eval/`, the run saw earlier results or the answer key:
   set `"discarded": true` and `"discard_reason"` (which path, which step) in its
   record, delete its exported notebook, and redo the run from step 1 in a fresh chat
   (`skills/skill-eval/references/combined-job.md:16-39`: a contaminated context is
   discarded and restarted, never "unlearned"). Discarded runs are not scored.

After all eight (run, task) sessions:

8. **Score.** Copy the repo to a workspace folder
   **outside** `.assistant/skills/` (for example a Git folder at
   `/Workspace/Users/<workspace-user>/<repo-folder>`), set the paths at the top of
   `score_open_weight_models.py`, and run it.
9. **Put the user's setup back.** Run
   ```bash
   python3 skills/open-weight-models/eval/install_skill_for_run.py --profile <profile> \
       --user <workspace-user> --restore-original
   ```
   This deletes the eval copy (it carries the hidden codes) and, if the user had the
   skill installed before the eval, re-imports that original install from its backup —
   the first backup whose record says `"backup_of": "pre-eval install"` (a copy without
   the hidden codes). If there is none, the user had no install before the eval and the
   skill is left removed. Add `--dry-run` first to see which applies.

A with-skill run whose notebook shows neither code did not pick up the skill. Keep it
and report it — that is a result, not a failed run.

## Scorers

All checks are deterministic checks on the exported notebook. No LLM judge. Each
sub-check is its own pass/fail metric. The replicated set and `provenance_tags` /
`existing_resources` use text patterns (comments and markdown ignored where noted in
`scorers.py`). `ai_gateway` and `uc_registry` parse each Python cell with Python's
`ast` module (`# MAGIC` cells and `%`/`!` lines removed); a cell that does not parse is
checked with the older text patterns instead, and the result's rationale says so.

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

**New checks** (metric names start with `new_checks/`, scored and compared separately;
each result's rationale says why it passed or failed):

| Check | Passes when the notebook code… | Tasks |
|---|---|---|
| `ai_gateway/inference_table`, `ai_gateway/usage_tracking` | attaches an AI Gateway config to a serving endpoint — `ai_gateway=` on `serving_endpoints.create` / `create_and_wait`, a `put_ai_gateway(...)` call, or an `"ai_gateway"` key in a REST body sent to a `serving-endpoints` URL (or the body of `PUT .../ai-gateway`) — in which the inference table config and the usage tracking config each have `enabled` set to `True`. The config may be built in variables (plain or annotated assignment, any cell), as SDK classes (keyword or positional), `from_dict`, dict literals or `dict(...)`; `enabled` may be a variable holding `True`. Imports alone, or a config never attached, do not count. | both |
| `uc_provenance/uc_registry` | actually registers a model in Unity Catalog: a registration call — `log_model(..., registered_model_name=...)`, `mlflow.register_model(...)`, or `MlflowClient().create_registered_model` / `create_model_version` — whose model name is three-part (`catalog.schema.model`, resolved through variables and f-strings), **or** `mlflow.set_registry_uri("databricks-uc")` plus a registration whose name is only known at run time (for example from a widget). Fails: the registry URI with no registration; a registration under a resolvable two-part name, even with the UC URI (Unity Catalog only accepts three-part names). | both |
| `uc_provenance/provenance_tags` | records the license **and** the Hugging Face revision as tags — run tags, logged-model tags or UC model tags all count (`teddy.md` asks for MLflow tags); the rationale names which kind was found | both |
| `existing_resources/endpoint` (`/table`, `/index` for teddy-002) | looks for an existing endpoint / table / index before the first create or delete call | both |

## Scoring and ship gate

`score_open_weight_models.py` evaluates each (run, task) with `mlflow.genai.evaluate`,
logging as the row's `query` the prompt actually pasted in that run (the evalset query,
or the `old_prompt_*.md` text with placeholders as written); it stops if the latest
record of any (run, task) is marked discarded (redo that run first);
extracts scores with `compare_runs.extract_scores()`, writes `baseline_scores.json`,
`with_skill_scores.json`, `old_prompt_baseline_scores.json` and
`old_prompt_with_skill_scores.json` (all metrics) and
`outputs/scores/<run_id>_{replicated,new_checks}.json`, and runs `compare_runs.main()`
within each pair only — `baseline` vs `with_skill`, `old_prompt_baseline` vs
`old_prompt_with_skill` — for the replicated set and for the new checks.

A task passes a set only if every sub-check in that set passes. Ship gate (skill-eval
default): at least one task-level win and zero regressions, judged on the standard pair,
replicated set and new checks each. The replication pair is reported next to the
2026-10 results, not used for the gate. Two tasks show direction, not statistical
significance; report raw sub-check counts, never percentages.

## Reproducing locally

```bash
uv run --isolated --with pytest --with-requirements skills/open-weight-models/tests/requirements.txt \
    python -m pytest -q skills/open-weight-models/tests
```
