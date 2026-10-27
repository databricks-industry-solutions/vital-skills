# open-weight-models — TEDDY retest

Paired evaluation of the `open-weight-models` skill on two TEDDY tasks (teddy-001,
teddy-002). Geneformer tasks come later.

**What is compared:** Genie Code with its built-in skills **plus** this skill, against
Genie Code with its built-in skills **only**. Built-in skills cannot be disabled, so the
result is the extra value of this skill on top of them.

**Baseline environment:** Genie Code as shipped — built-in skills always on and not
visible in the UI — with no personal or workspace skills, plus the user's instructions
file (`.assistant_instructions.md`, fingerprinted in every run record). The June AI Dev
Kit user skills that were installed during the 2026-10 runs were removed before the
reruns, for both pairs.

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

**Environment per pair.** Both pairs run in the baseline environment above (the AI Dev
Kit user skills from 2026-10 removed), so the replication pair is *not* the 2026-10
environment. Every run record stores the environment Genie Code could see (step 1) and
the Skills panel as observed (step 3), and each pair is compared only within itself —
never a run of one pair against a run of the other.

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
| `check_outcome.py`, `fixtures/teddy_request.json` | laptop script: outcome checks on what a run built, with a fixed TEDDY request | the person running the eval (never pasted into Genie Code) |
| `expectations.json` | answer key: difficulty, model family, variant, capability, expected facts, checks, new checks, outcome checks, per-check severity / source / environment dependence | scoring only |
| `scorers.py` | MLflow scorers (exports `scorers`) | scoring only |
| `score_open_weight_models.py` | Databricks scoring notebook | scoring only |
| `outputs/<run_id>/<task_id>.py` | exported notebooks (after the runs) | scoring only |
| `outputs/run_records.json` | one record per run: skill state, environment, hidden codes (incl. extra skills), backup, commit, Skills panel, visible steps, skill self-report, compute, resources created, outcome, discarded / reason, notes | scoring only |
| `<run_id>_scores.json`, `outputs/scores/` | per-task scores (after scoring) | — |
| `eval_report.md` | results | — |
| `README_environments.md` | known Genie Code runtime snapshots | — |

Genie Code must never see `eval/`. The skill is installed by
`install_skill_for_run.py`, which copies only `SKILL.md` and `references/` (plus
`assets/` and `scripts/` if they exist). Do not copy the repo to the workspace until all
runs are done (step 13).

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
   - Default: open-weight-models is the only skill installed. If a run needs other skills
     too, pass each with `--extra-skill-dir <skill folder>` (repeatable): each is staged
     the same way (no `eval/` or `tests/`), gets its own hidden code in its `SKILL.md`
     (`# skill-proof-<name>: <code>`), is recorded under `extra_skills`, and is checked in
     step 6. Pass the same extra skills in both runs of a pair.
2. **Clear memory.** In Genie Code, ask it to delete any memories about this
   evaluation, TEDDY, or the open-weight-models skill. Close that chat.
3. **Check the Skills panel — before pasting the prompt.** Hard refresh the browser
   (Cmd+Shift+R), open Genie Code → **Customizations → Skills**, and screenshot both the
   **User** and the **All skills** tabs (save as `outputs/<run_id>/<task_id>_skills_panel.png`;
   add `_user` / `_all` if they need two images). Record the filename(s) and the skills
   listed on each tab in the record's `skills_panel` (`screenshot`,
   `observed_user_tab`, `observed_all_skills_tab`). Expected (`skills_panel.expected`,
   written by step 1): skill-off runs show no skills ("No skills found"); with-skill runs
   show only `open-weight-models` (plus any `--extra-skill-dir` skills). **If what you
   see differs from expected, stop and fix the installs before running** — built-in
   skills are not listed here, so anything shown is a personal or workspace skill.
4. **Fresh notebook and chat.** Create a blank notebook `<task_id>` in the run folder
   and open a new Genie Code chat on it.
5. **Run.** Paste the prompt verbatim — the `query` from `evalset.json` for the
   standard pair, or the task's block from `old_prompt_baseline.md` /
   `old_prompt_with_skill.md` for the replication pair (placeholders filled). Answer
   questions **only** from `answer_sheet.md`. Do not steer, hint or correct. Let it run
   to the end.
6. **Export the notebook** as source to `outputs/<run_id>/<task_id>.py`:
   ```bash
   databricks workspace export \
       /Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/<run_id>/<task_id> \
       --format SOURCE --file skills/open-weight-models/eval/outputs/<run_id>/<task_id>.py -p <profile>
   ```
   Then check the hidden codes (every installed skill has one; they are the proof of
   which skill files Genie Code read):
   ```bash
   python3 skills/open-weight-models/eval/install_skill_for_run.py --run-id <run_id> \
       --task-id <task_id> --check-notebook skills/open-weight-models/eval/outputs/<run_id>/<task_id>.py
   ```
7. **Visible steps and skill self-report — after the task is complete and exported.**
   - Copy the chat's visible step list (for example "Searched documentation", file
     reads) into the record's `visible_steps.text` and save a screenshot
     (`outputs/<run_id>/<task_id>_steps.png`, filename in `visible_steps.screenshot`).
     Step 11 (leak check) reads this list.
   - **Standard pair only:** send this as a separate message, identical in both runs:
     "List every skill from the Skill Registry that you loaded or consulted while
     developing this notebook." Record the reply verbatim in
     `skills_self_report.text`. It is sent only after the task is done, so it cannot
     steer the work.
   - **Replication pair:** skip the follow-up — the 2026-10 prompts already end with
     this request (for example
     `previous_runs_2026-10/results/with_skill/oss-001.txt:23`). Record the answer
     Genie Code gave in the notebook or chat instead (`skills_self_report.source` is
     pre-filled accordingly).
   - The self-report is **Genie Code's own claim, not verified**. The hidden codes
     (step 6) remain the proof that open-weight-models was read.
8. **Fill `resources_created` and `compute`** in the run's record in
   `outputs/run_records.json`, from the notebook: `serving_endpoint`, `uc_model`
   (three-part name), `uc_model_version`, and for teddy-002 `delta_table`,
   `vector_search_endpoint`, `vector_search_index`; `compute.type` and
   `compute.runtime` from the notebook's runtime cell or the compute it ran on (for
   example "serverless GPU (GPU_1xA10)", "AI Runtime v5").
9. **Outcome checks — before any teardown.** From the laptop (needs
   `databricks-sdk`, verified with 0.157.0):
   ```bash
   python3 skills/open-weight-models/eval/check_outcome.py --profile <profile> \
       --run-id <run_id> --task-id <task_id> [--reference-endpoint <name>]
   ```
   It checks the endpoint is READY and serves the recorded model version, sends the
   fixed request in `fixtures/teddy_request.json` (3 cells × 100 real 70M-vocab genes,
   built per the `teddy.md` serving contract) and checks one finite, non-zero,
   512-dimensional embedding per cell (512 = `d_model` in the 70M `config.json`); for
   teddy-002 it checks the Vector Search endpoint is ONLINE, the index is ready, and a
   nearest-neighbour query with a fixture embedding returns rows with a cell-type
   column. Results go into the record's `"outcome"`. `--reference-endpoint` (optional)
   adds the per-cell cosine similarity against another TEDDY endpoint, as information
   only — no pass/fail threshold is defined.
10. **Tear down what the run created** — and only that, using the names in
   `resources_created`:
   ```bash
   databricks serving-endpoints delete <serving_endpoint> -p <profile>
   databricks vector-search-indexes delete-index <vector_search_index> -p <profile>        # teddy-002
   databricks vector-search-endpoints delete-endpoint <vector_search_endpoint> -p <profile> # teddy-002, if the run created it
   databricks tables delete <delta_table> -p <profile>                                       # teddy-002
   databricks registered-models delete <uc_model> -p <profile>
   ```
   Also drop the AI Gateway inference/usage tables the endpoint created, if any.
   Tear down the replication pair's resources too: its prompts reuse the 2026-10 names
   (see the warning above).
11. **Check for leaks.** Read the chat's visible steps (step 7). If Genie Code opened, searched
    or listed any file or folder whose path contains `oss-0`, `gt_`, `hls-skills-tests`,
    `previous_runs_2026-10` or `/eval/`, the run saw earlier results or the answer key:
    set `"discarded": true` and `"discard_reason"` (which path, which step) in its
    record, delete its exported notebook, and redo the run from step 1 in a fresh chat
    (`skills/skill-eval/references/combined-job.md:16-39`: a contaminated context is
    discarded and restarted, never "unlearned"). Discarded runs are not scored.
12. **Record.** Write anything unusual (questions asked and answers given, crashes,
    how long it ran) into the record's `"notes"`.

After all eight (run, task) sessions:

13. **Score.** Copy the repo to a workspace folder **outside** `.assistant/skills/`
    (for example a Git folder at `/Workspace/Users/<workspace-user>/<repo-folder>`),
    set the paths at the top of `score_open_weight_models.py`, and run it.
14. **Put the user's setup back.** Run
    ```bash
    python3 skills/open-weight-models/eval/install_skill_for_run.py --profile <profile> \
        --user <workspace-user> --restore-original
    ```
    This deletes the eval copy (it carries the hidden codes) and, if the user had the
    skill installed before the eval, re-imports that original install from its backup —
    the first backup whose record says `"backup_of": "pre-eval install"` (a copy without
    the hidden codes). If there is none, the user had no install before the eval and the
    skill is left removed. Add `--dry-run` first to see which applies. Extra skills
    (`--extra-skill-dir`) are not restored by this command: delete them, or re-import
    their recorded `extra_skills[].backup`, by hand.

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

**Outcome group** (metric names start with `outcome/`): did what the run built actually
work? Measured live by `check_outcome.py` (step 9, before teardown), stored in the run
record, and read by the scorer — nothing is re-measured at scoring time.

| Check | Passes when | Tasks |
|---|---|---|
| `endpoint_ready`, `serves_recorded_version` | the endpoint is READY and serves the recorded UC model + version | both |
| `one_embedding_per_cell`, `embeddings_finite`, `embedding_dim` | the fixture request returns one embedding per cell, all finite and non-zero, each of dimension 512 | both |
| `vs_endpoint_online`, `index_ready` | the Vector Search endpoint is ONLINE and the index ready | teddy-002 |
| `nn_query_returns_rows`, `nn_has_cell_type_column` | a nearest-neighbour query with a fixture embedding returns rows that include a cell-type column | teddy-002 |

**Optional, off by default:** `--reference-endpoint` cosine similarity (information
only), and an LLM judge over each task's `guidelines` (`guideline_judges()`,
skill-eval `references/scorer-pack.md`: binary, `feedback_value_type=Literal["yes","no"]`;
metric `judge/guidelines`; set `USE_GUIDELINE_JUDGE = True` in the scoring notebook; the
judge model must differ from the model that wrote the notebooks).

## Severity and environment dependence

Every check in every group has an entry in `expectations.json` → `"checks"`:

- `severity`: **critical** — prevents a working endpoint or yields wrong / mixed
  embeddings; **major** — works but unsafe, ungoverned or not reproducible (no
  inference table, no provenance, deletes without looking, unvalidated);
  **minor** — robustness or style.
- `source`: the `SKILL.md` / `teddy.md` line the check rests on (required for critical
  and major), and `why`.
- `depends_on`: `model` (true because of TEDDY itself — `HF_HUB_DISABLE_XET` because the
  Hugging Face repo uses Xet, `teddy.md:130`; `transformers==4.41.0` because of the
  model code, `teddy.md:177`; the 512 dimension), `compute` (`/tmp` vs `/local_disk0`,
  `teddy.md:160`), or `none`; with `assumes` naming the environment. The task's
  `environment_assumed` is serverless GPU notebook compute (GPU_1xA10) on AI Runtime
  v5 with Model Serving GPU_SMALL (`README_environments.md`). Each run records its
  actual compute and runtime (step 8), so a check that depends on compute can be read
  against what the run used.

The scoring notebook prints, per pair, run and task, the **raw counts of failed checks
per severity** and whether the **outcome group passed** — no percentages and no single
score — so every failure can be read against whether the run actually worked.

### Known false negatives in the replicated set

The replicated checks are kept unchanged so they stay comparable with 2026-10. These
valid alternatives fail them anyway; read such failures by hand.
`tests/test_eval_scorers.py::test_known_false_negatives` keeps this list true.

| Check | Valid alternative that fails | Where it is documented |
|---|---|---|
| `download_staging/deps_pinned`, `forbidden_patterns/transformers` | a range pin `transformers>=4.30.0,<4.52.0` plus the runtime shim | `SKILL.md:242` (`teddy.md:170` asks for the exact pin — the skill is inconsistent) |
| `model_registration/pip_reqs_from_source` | copying the pins without naming `pyproject.toml` / requirements | `teddy.md:165-173` |
| `pyfunc_quality/real_gene_ids` | a payload built from `GeneTokenizer.from_pretrained(model_dir)` without naming `vocab.txt` | `teddy.md:126` |
| `download_staging/uses_tmp` | `tempfile.mkdtemp()` (writes under /tmp, no literal) | — |
| `forbidden_patterns/local_disk0` | compute-adaptive code that tries `/local_disk0` and falls back to `/tmp` | 2026-10 ground-truth pattern (`previous_runs_2026-10/00_oss_models_test_plan.py`) |
| `endpoint_deployment/ai_gateway_config` | an inference table set in a REST body (`"inference_table_config": {...}`) | `SKILL.md:325` (the new `ai_gateway` checks accept it) |
| `endpoint_deployment/sdk_enums` | REST deployment with `"workload_type": "GPU_SMALL"` (valid for the REST API) | — |
| `ai_search/graceful_skip` | `try/except AnalysisException` around reading the table | — |
| `endpoint_deployment/run_go_gate` | a widget guard with another name (e.g. `deploy`) | `SKILL.md:251` suggests `run_go` |

Checked and **not** a false negative: the subprocess download in `teddy.md:137-154`
contains `HF_HUB_DISABLE_XET` and passes `download_staging/hf_xet_disabled`.

## Scoring and ship gate

`score_open_weight_models.py` evaluates each (run, task) with `mlflow.genai.evaluate`,
logging as the row's `query` the prompt actually pasted in that run (the evalset query,
or the `old_prompt_*.md` text with placeholders as written); it stops if the latest
record of any (run, task) is marked discarded or has no recorded outcome. It extracts
scores with `compare_runs.extract_scores()`, writes `baseline_scores.json`,
`with_skill_scores.json`, `old_prompt_baseline_scores.json` and
`old_prompt_with_skill_scores.json` (all metrics) and
`outputs/scores/<run_id>_<group>.json`, runs `compare_runs.main()` within each pair
only — `baseline` vs `with_skill`, `old_prompt_baseline` vs `old_prompt_with_skill` —
once per group (replicated, new checks, outcome, and the judge if enabled), and prints
the severity table.

A task passes a group only if every check in that group passes. Ship gate (skill-eval
default): at least one task-level win and zero regressions, judged on the standard pair,
for the replicated set, the new checks and the outcome group each. The replication pair
is reported next to the 2026-10 results, not used for the gate. Two tasks show
direction, not statistical significance; report raw counts, never percentages.

## Reproducing locally

```bash
uv run --isolated --with pytest --with-requirements skills/open-weight-models/tests/requirements.txt \
    python -m pytest -q skills/open-weight-models/tests
```
