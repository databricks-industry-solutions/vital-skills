# open-weight-models — skill evaluation

This folder evaluates the `open-weight-models` skill: the same tasks are run in Genie Code with and without
the skill, and the two results are compared.

- **Shared harness, one folder per model family.** The scripts here work for any model; each family's
  settings and rounds live under `models/<family>/`.
- **TEDDY** (tasks teddy-001 and teddy-002) has two rounds: round 1 before the skill fixes, round 2 a clean
  rerun with the updated skill. Geneformer tasks come later.
- **Results:** [`eval_report.md`](eval_report.md) (one page), [`models/teddy/ROUNDS.md`](models/teddy/ROUNDS.md)
  (every round side by side, with links to each round's notes), [`models/teddy/scores.json`](models/teddy/scores.json)
  (every round's counted scores).

**Words used here**

| Term | Meaning |
|---|---|
| Built-in skills | Skills compiled into Genie Code; always on, not listed in the UI, cannot be disabled |
| Run | One Genie Code session on one task, skill off (`baseline`) or on (`with_skill`) |
| Round | One full set of runs, kept locally in a `runs_<YYYYMMDD>/` folder |
| Run record | The JSON entry the laptop scripts write for each run (`run_records.json`) |
| Sub-check | One pass/fail line of a scorer |
| UC | Unity Catalog |
| AI Search | Databricks AI Search, formerly Vector Search (VS); some names below still say Vector Search |
| Xet | Hugging Face's newer storage backend; TEDDY's download needs it turned off |
| LLM judge | A language model that grades free text; optional here and off by default |

## Folder layout

```text
eval/
  README.md, evalset.json, expectations.json, scorers.py, score_open_weight_models.py,
  install_skill_for_run.py, check_outcome.py, family_settings.py   shared
  eval_report.md                                                   summary, one section per model family
  baseline_scores.json, with_skill_scores.json                     latest round's counted scores (skill-eval format)
  CHECKS.md, list_checks.py                                        every scored check (generated)
  models/teddy/
    family.json            names per run, outcome settings (dim 512, which tasks have search)
    fixtures/teddy_request.json, answer_sheet.md
    old_prompts.json       the Sep–Oct 2026 prompts of both replication runs, with this round's names
    runs_20261026/         round 1: run_records.json, exported notebooks, screenshots, scores   (git-ignored)
    runs_20261027/         round 2 (same layout; tools take --round runs_20261027)              (git-ignored)
    runs_20260924/         the frozen Sep 24 – Oct 7, 2026 TEDDY sprint (old task IDs)          (git-ignored)
    ROUNDS.md              all TEDDY rounds side by side; each round's full notes folded at the end
    scores.json            counted scores of every round and sample, in one file
  ground_truth/            the hand-built reference notebooks the model references were written from
```

- **In git:** the scripts, the answer key, `eval_report.md`, and per family `ROUNDS.md`, the round notes and
  `scores.json`, plus the two score files.
- **Not in git:** the `runs_<date>/` round folders. The tools write them locally during a round.
- **What the repo keeps of each round:** its notes (the round's `RESULTS.md`, folded under "Round details" in
  `models/<family>/ROUNDS.md`) and its counted scores of every sample (in `models/<family>/scores.json`, written
  by the scoring notebook).

## What is compared

- **With the skill:** Genie Code with its built-in skills **plus** this skill.
- **Without the skill (the baseline):** Genie Code with its built-in skills **only**.
- Built-in skills cannot be disabled, so the result is the extra value of this skill on top of them.

**The baseline environment** is Genie Code as shipped:

- built-in skills on (always), and no personal or workspace skills;
- the user's instructions file (`.assistant_instructions.md`), whose fingerprint is stored in every run record;
- the June AI Dev Kit user skills, installed during the Sep–Oct 2026 runs, were removed before the reruns, for
  both pairs.

## Four runs per task, in two pairs

Each pair is compared only within itself.

| Pair | Run (`run_id`) | Skill | What is pasted into the chat |
|---|---|---|---|
| standard | `baseline` | off | the task's `query` from `evalset.json`, names rendered for the run |
| standard | `with_skill` | on | the same `query`, names rendered for the run, nothing else |
| replication | `old_prompt_baseline` | off | the task's prompt under `old_prompt_baseline` in `models/teddy/old_prompts.json` |
| replication | `old_prompt_with_skill` | on | the task's prompt under `old_prompt_with_skill` in `models/teddy/old_prompts.json` |

- **Standard pair — the evaluation.** Does Genie Code find and use the skill on its own, from a plain request?
- **Replication pair — the Sep–Oct 2026 results again.** It reuses the Sep–Oct 2026 prompts as recorded: the
  baseline prompt, and the with-skill prompt that told Genie Code to read `SKILL.md` and `teddy.md` first.
  Only the resource names and paths, the skill file paths and the export path were changed (listed at the
  top of each file). The Sep 24 – Oct 7, 2026 notes and scores are in
  [`models/teddy/ROUNDS.md`](models/teddy/ROUNDS.md) (Sep–Oct 2026, under "Round details").
- **Same environment for both pairs.** Both run in the baseline environment above (AI Dev Kit user skills
  removed), so the replication pair is *not* the Sep–Oct 2026 environment. Every run record stores what
  Genie Code could see (step 1) and the Skills panel as observed (step 3). A run of one pair is never
  compared with a run of the other.

> [!NOTE]
> The replication prompts still tell Genie Code to delete any pre-existing endpoint and model versions on
> the first run (recorded wording), but they now name only this run's own resources in `skills_owm_202610`,
> so they no longer touch anything from Sep–Oct 2026. The user's instructions file has a similar "delete
> existing endpoint + model versions" rule that applies to every run.

### Names: every run creates its own resources

All resources live under schema `skills_owm_202610` (owm = open-weight-models; managed volume
`owm_teddy_files`). Each name carries the run, with `<run>` one of `baseline`, `with_skill`, `old_baseline`,
`old_with_skill` (`models/teddy/family.json` → `names`):

| Resource | teddy-001 | teddy-002 |
|---|---|---|
| serving endpoint | `owm_teddy_ep_001_<run>` | `owm_teddy_ep_002_<run>` |
| registered model | `<catalog>.skills_owm_202610.owm_teddy_model_001_<run>` | `<catalog>.skills_owm_202610.owm_teddy_model_002_<run>` |
| Vector Search endpoint | — | `owm_teddy_vs_002_<run>` |
| embeddings table | — | `<catalog>.skills_owm_202610.owm_teddy_cells_002_<run>` |
| Vector Search index | — | `<catalog>.skills_owm_202610.owm_teddy_index_002_<run>` |
| download folder | `/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/001_<run>/` | `/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/002_<run>/` |

- **One exception:** teddy-001 `with_skill` uses `<run>` = `skill_on` (`run_suffix_overrides`). The first
  `baseline`/teddy-001 run had already used the `_001_with_skill` names, which came from the run guide before
  per-run names were finalised. Its prompt is the standard teddy-001 request, word for word, and the skill was
  off, so it is a valid baseline sample (see the round README).
- **If a run used other names or another prompt,** record what was actually used in its run record:
  `prompt_used`, `resources_created`, and a line in `notes`. `check_outcome.py` reads `resources_created`, and
  the scoring notebook logs `prompt_used` when it is set.

### Prompts

- `evalset.json` stores each request with a `{run}` placeholder in one added sentence ("Use these names: …",
  right after the catalog/schema sentence). The rest is the Sep–Oct 2026 wording, with the schema changed.
- Print the exact prompt for a run:

  ```bash
  python3 skills/open-weight-models/eval/family_settings.py prompt --run-id <run_id> --task-id <task_id>
  ```

- The scoring notebook logs this rendered prompt as each row's query.

## Files

| File | What it is | Who may read it during a run |
|---|---|---|
| `evalset.json` | task requests: `task_id`, `dataset`, `query` (with `{run}`) only | the person running the eval |
| `family_settings.py`, `models/<family>/family.json` | per-family names, prompt rendering, outcome settings | the person running the eval |
| `models/teddy/answer_sheet.md` | fixed replies if Genie Code asks a question | the person running the eval |
| `models/teddy/old_prompts.json` | Sep–Oct 2026 prompts with this round's names, both replication runs, with their source and notes (replication pair only) | the person running the eval |
| `install_skill_for_run.py` | laptop script: skill off/on + run record | the person running the eval |
| `check_outcome.py`, `models/teddy/fixtures/teddy_request.json` | laptop script: outcome checks on what a run built, with the family's fixed request | the person running the eval (never pasted into Genie Code) |
| `expectations.json` | answer key: difficulty, model family, variant, capability, expected facts, checks, new checks, outcome checks, per-check severity / source / environment dependence | scoring only |
| `scorers.py` | MLflow scorers (exports `scorers`) | scoring only |
| `score_open_weight_models.py` | Databricks scoring notebook | scoring only |
| `models/teddy/runs_<YYYYMMDD>/<run_id>/<task_id>.py` | exported notebooks (after the runs; the record's `export_file` names the file when set, else `<task_id>_attempt<n>.py` for a run repeated as attempts) | scoring only |
| `models/teddy/runs_<YYYYMMDD>/run_records.json` | one record per run: skill state, environment, hidden codes if opted in, backup, commit, Skills panel, visible steps, skill self-report, compute, resources created, outcome, endpoint config, attempt, discarded / reason, notes | scoring only |
| `models/teddy/runs_<YYYYMMDD>/RESULTS.md` | results of that round | — |
| `models/teddy/runs_<YYYYMMDD>/scores/` | per-task scores (after scoring) | — |
| `baseline_scores.json`, `with_skill_scores.json` | latest round's counted scores (corrected checks, new checks, outcome) per task, in skill-eval's format; read by `compare_runs.py`; updated by the scoring notebook | — |
| `models/<family>/scores.json` | every round's counted scores (every sample, attempts included) in one file; each round names its notes section in `ROUNDS.md` | — |
| `eval_report.md` | results summary: one row per model family, links to `models/teddy/ROUNDS.md` | — |
| `models/teddy/ROUNDS.md` | TEDDY across all rounds: how each was run, results per round, checks passed only with the skill, where the skill helps by category | — |
| `CHECKS.md` | every scored check per model family: severity and what it guards (generated by `list_checks.py`; `tests/test_list_checks.py` keeps it current) | — |
| `README_environments.md` | known Genie Code runtime snapshots | — |

> [!IMPORTANT]
> Genie Code must never see `eval/`. `install_skill_for_run.py` installs only `SKILL.md` and `references/`
> (plus `assets/` and `scripts/` if they exist). Do not copy the repo to the workspace until all runs are
> done (step 13).

## Run protocol

**Order of the runs**

- Both skill-off runs come before any with-skill run. Then, optionally, the replication pair:
  `old_prompt_baseline`, then `old_prompt_with_skill`.
- A skill-off run never follows a with-skill run in the same pair, so nothing about the skill is in a chat or
  in memory before its baseline. Between a with-skill run and the next skill-off run, step 1 (`--state off`)
  removes the skill and step 2 clears memory.
- One task at a time, one fresh chat per task per run.
- **What was run so far:** round `runs_20261026`, the standard pair, on 2026-10-27 (the replication pair was
  not run; deviations from plan are in that round's README). Round `runs_20261027` reran all four runs on
  2026-10-27/28 with the updated skill. Notes for both: linked from [`models/teddy/ROUNDS.md`](models/teddy/ROUNDS.md).

**Where the notebooks live in the workspace:** one folder per run,
`/Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-<date>/<run_id>/`, and one
`teddy-eval-<date>` folder per round (round 1: `teddy-eval-2026-10-26`, round 2: `teddy-eval-2026-10-27`). The
replication prompts export there too.

### Checklist for each (run, task)

1. **Set the skill off or on**, from the repo root:
   ```bash
   python3 skills/open-weight-models/eval/install_skill_for_run.py --profile <profile> \
       --user <workspace-user> --state off|on --run-id <run_id> --task-id <task_id>
   ```
   - `off` for the two skill-off runs, `on` for the two with-skill runs. Add `--dry-run` first to see the
     commands.
   - For `on`, the script first builds and checks the copy locally (the skill's own files, unchanged; no
     `eval/` or `tests/`), then installs it.
   - **It stops if another copy of the skill is installed** (for example an old `oss-models` folder). Move or
     delete it and run again.
   - **It stops if `.assistant/skills/open-weight-models` already exists** — for example the copy from the
     previous with-skill task, or the user's own install before the eval. To replace it, add
     `--replace-existing`. The script then exports the existing copy to
     `models/teddy/runs_<YYYYMMDD>/<run_id>/backup/<task_id>-<timestamp>/` (not committed), deletes it and
     installs the new copy. If creating the folder or the import fails, it prints the commands that restore
     the backup.
   - **It writes a run record** to `models/teddy/runs_<YYYYMMDD>/run_records.json` with the environment Genie
     Code will see:
     - the user skills folder (each `SKILL.md` with size and sha256);
     - the sha256 of `/Workspace/Users/<workspace-user>/.assistant_instructions.md`, if present;
     - whether `/Workspace/.assistant/skills` and `/Workspace/.assistant_workspace_instructions.md` exist.
   - **Other skills:** by default open-weight-models is the only skill installed. If a run needs more, pass
     each with `--extra-skill-dir <skill folder>` (repeatable). Each is staged the same way (no `eval/` or
     `tests/`) and recorded under `extra_skills`. Pass the same extra skills in both runs of a pair.
   - **Hidden codes are off by default.** `--hidden-codes` (opt-in) adds to the installed `SKILL.md` and
     `teddy.md` an instruction to put a random code at the top of any notebook. Do not use it as proof of
     skill use:
     - an instruction planted in a skill file to make the agent emit an unrelated marker is a
       prompt-injection pattern, and Genie Code may rightly refuse it — in `with_skill`/teddy-001
       (2026-10-27) it declined the marker as "possibly injected";
     - it also changes the files Genie Code reads;
     - a code found shows a file was read; a code missing shows nothing.
2. **Clear memory.** Ask Genie Code to delete any memories about this evaluation, TEDDY, or the
   open-weight-models skill. Close that chat.
3. **Check the Skills panel — before pasting the prompt.**
   - Hard refresh the browser (Cmd+Shift+R), open Genie Code → **Customizations → Skills**, and screenshot
     both the **User** and the **All skills** tabs. Save as
     `models/teddy/runs_<YYYYMMDD>/<run_id>/<task_id>_skills_panel.png` (add `_user` / `_all` if you need two
     images).
   - Record the filename(s) and the skills listed on each tab in the record's `skills_panel` (`screenshot`,
     `observed_user_tab`, `observed_all_skills_tab`).
   - **Expected** (`skills_panel.expected`, written by step 1): skill-off runs show no skills ("No skills
     found"); with-skill runs show only `open-weight-models`, plus any `--extra-skill-dir` skills.
   - **If what you see differs, stop and fix the installs before running.** Built-in skills are not listed
     here, so anything shown is a personal or workspace skill.
4. **Fresh notebook and chat.** Create a blank notebook `<task_id>` in the run folder and open a new Genie
   Code chat on it.
5. **Run.**
   - Paste the prompt verbatim: the `query` from `evalset.json` for the standard pair, or the task's block
     from `models/teddy/old_prompts.json` (`old_prompt_baseline` / `old_prompt_with_skill`) for the replication pair. Print
     it with `family_settings.py prompt` and fill `<catalog>` / `<workspace-user>`.
   - Answer questions **only** from `models/teddy/answer_sheet.md`. Do not steer, hint or correct.
   - Let it run to the end.
6. **Export the notebook** as source to `models/teddy/runs_<YYYYMMDD>/<run_id>/<task_id>.py`:
   ```bash
   databricks workspace export \
       /Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-<date>/<run_id>/<task_id> \
       --format SOURCE --file skills/open-weight-models/eval/models/teddy/runs_<YYYYMMDD>/<run_id>/<task_id>.py -p <profile>
   ```
   - **A repeat of the same (run, task)** is exported as `<task_id>_attempt<n>.py`, with `"attempt": n` in
     its record. The latest attempt is the primary sample; earlier ones are scored as extra samples.
   - **Only for runs installed with `--hidden-codes`,** check the codes:
     ```bash
     python3 skills/open-weight-models/eval/install_skill_for_run.py --run-id <run_id> \
         --task-id <task_id> --check-notebook skills/open-weight-models/eval/models/teddy/runs_<YYYYMMDD>/<run_id>/<task_id>.py
     ```
7. **Visible steps and skill self-report — after the task is complete and exported.**
   - **Visible steps:** copy the chat's visible step list (for example "Searched documentation", file reads)
     into the record's `visible_steps.text`, and save a screenshot
     (`models/teddy/runs_<YYYYMMDD>/<run_id>/<task_id>_steps.png`, filename in `visible_steps.screenshot`).
     Step 11 (leak check) reads this list.
   - **Self-report, standard pair only:** send this as a separate message, identical in both runs: "List
     every skill from the Skill Registry that you loaded or consulted while developing this notebook."
     Record the reply verbatim in `skills_self_report.text`. It is sent only after the task is done, so it
     cannot steer the work.
   - **Self-report, replication pair:** skip the follow-up — the Sep–Oct 2026 prompts already end with this
     request (for example line 23 of the recorded `results/with_skill/oss-001.txt` in the Sep–Oct round
     folder). Record the answer Genie Code gave in the notebook or chat instead (`skills_self_report.source`
     is pre-filled accordingly).
   - **Evidence of skill use** comes from three sources together, none of them proof alone:
     - the self-report (Genie Code's own claim);
     - the visible steps (for example "Read Skill File" on `SKILL.md` / `teddy.md`);
     - skill-specific choices in the notebook (for example the `teddy.md` input format, the pinned Hugging
       Face revision, the `run_go` gate).

     Write them up per run in the round's results.
8. **Check `resources_created`, and fill `uc_model_version` and `compute`** in the run's record in
   `models/teddy/runs_<YYYYMMDD>/run_records.json`.
   - Step 1 pre-fills `resources_created` with the names the prompt asked for (`family.json`, with
     `<catalog>` left as a placeholder). Correct any name the notebook actually used differently.
   - Add `uc_model_version` from the notebook.
   - Fill `compute.type` and `compute.runtime` from the notebook's runtime cell or the compute it ran on (for
     example "serverless GPU (GPU_1xA10)", "AI Runtime v5").
9. **Outcome checks — before any teardown.** From the laptop (needs `databricks-sdk`, verified with 0.157.0,
   and `mlflow-skinny` to read the signature):
   ```bash
   python3 skills/open-weight-models/eval/check_outcome.py --profile <profile> \
       --run-id <run_id> --task-id <task_id> --catalog <catalog> [--reference-endpoint <name>]
   ```
   `--catalog` fills the `<catalog>` placeholder at call time; the record keeps the placeholder. The script:
   - checks the endpoint is READY and serves the recorded model version, and reads the input signature of
     that registered version;
   - sends the fixed cells of `models/teddy/fixtures/teddy_request.json` (3 cells × 100 real 70M-vocabulary
     genes, of different lengths once zero counts are dropped) **in the format the endpoint declares** — the
     shared `teddy.md` format, gene identifiers + values per cell, or pre-tokenized token IDs per cell — so a
     valid format of the run's own is not penalised;
   - checks one finite, non-zero, 512-dimensional embedding per cell (512 = `d_model` in the 70M
     `config.json`), whether the endpoint takes cell expression rather than token IDs, and, separately,
     whether it uses the shared format;
   - POSTs to `/serving-endpoints/<name>/invocations` with inference controls under `"params"` (the key the
     MLflow scoring server reads; `serving_endpoints.query(extra_params=...)` is silently dropped);
   - for teddy-002, checks that the AI Search (Vector Search) endpoint is ONLINE, the index is ready, and a
     nearest-neighbour query with a fixture embedding — asking only for columns the index holds — returns
     rows with a cell-type column.

   Results go into the record's `"outcome"`; the endpoint's workload type, scale-to-zero and AI Gateway flags
   go into `"endpoint_config"`. A check that cannot be measured (no signature, an unrecognised format) is
   recorded as `null` with the reason — measure it by hand before teardown. `--reference-endpoint`
   (optional) adds the per-cell cosine similarity against another TEDDY endpoint, as information only; no
   pass/fail threshold is defined.
10. **Tear down what the run created** — and only that, using the names in `resources_created`:
    ```bash
    databricks serving-endpoints delete <serving_endpoint> -p <profile>
    databricks vector-search-indexes delete-index <vector_search_index> -p <profile>        # teddy-002
    databricks vector-search-endpoints delete-endpoint <vector_search_endpoint> -p <profile> # teddy-002, if the run created it
    databricks tables delete <delta_table> -p <profile>                                       # teddy-002
    databricks registered-models delete <uc_model> -p <profile>
    ```
    - Also drop the AI Gateway inference and usage tables the endpoint created, if any, and empty the run's
      download folder (`/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/00N_<run>/`).
    - Every name is unique to the run, so teardown cannot touch another run or the Sep–Oct 2026 resources.
11. **Check for leaks.** Read the chat's visible steps (step 7).
    - If Genie Code opened, searched or listed any file or folder whose path contains `oss-0`, `gt_`,
      `hls-skills-tests`, `runs_20260924` or `/eval/`, the run saw earlier results or the answer key.
    - Then set `"discarded": true` and `"discard_reason"` (which path, which step) in its record, delete its
      exported notebook, and redo the run from step 1 in a fresh chat. A contaminated context is discarded
      and restarted, never "unlearned" (`skills/skill-eval/references/combined-job.md:16-39`). Discarded runs
      are not scored.
12. **Record.** Write anything unusual into the record's `"notes"`: questions asked and answers given,
    crashes, how long it ran.

### After all (run, task) sessions

Four sessions for the standard pair, eight with the replication pair.

13. **Score.** Copy the repo to a workspace folder **outside** `.assistant/skills/` (for example a Git folder
    at `/Workspace/Users/<workspace-user>/<repo-folder>`), set the paths at the top of
    `score_open_weight_models.py`, and run it.
14. **Put the user's setup back:**
    ```bash
    python3 skills/open-weight-models/eval/install_skill_for_run.py --profile <profile> \
        --user <workspace-user> --restore-original
    ```
    - **It checks first and stops without deleting anything** unless the installed copy is one the eval
      installed (its `SKILL.md` sha256 matches an install fingerprinted in the run records, or it carries a
      `--hidden-codes` marker) and the recorded pre-eval backup (if any) exists locally.
    - **Then it deletes the eval copy.** If the user had the skill installed before the eval, it re-imports
      that install from its backup — the first backup whose record says `"backup_of": "pre-eval install"`. If
      there is none, the user had no install before the eval and the skill is left removed. Add `--dry-run`
      first to see which applies.
    - **Extra skills** (`--extra-skill-dir`) are not restored by this command: delete them, or re-import their
      recorded `extra_skills[].backup`, by hand.

A with-skill run with no evidence of skill use (self-report, visible steps, notebook choices) did not pick up
the skill. Keep it and report it — that is a result, not a failed run.

## Scorers

**How every check works**

- All checks are deterministic checks on the exported notebook; no LLM judge by default.
- Each sub-check is its own pass/fail metric.
- The replicated set, `provenance_tags` and `existing_resources` use text patterns (comments and markdown are
  ignored where `scorers.py` says so).
- `ai_gateway` and `uc_registry` parse each Python cell with Python's `ast` module (`# MAGIC` cells and
  `%` / `!` lines removed). A cell that does not parse is checked with the older text patterns instead, and
  the result's rationale says so.

### Replicated set ("verbatim") — kept as a historical record only

The Sep–Oct 2026 checks with their logic unchanged, so they reproduce the Sep–Oct 2026 counts as originally
reported. Nothing is counted from them.

- **8 groups:**
  - `download_staging`: temp path, Xet off, pinned `transformers`;
  - `pyfunc_quality`: PyFunc class, imports, `load_context`, `sys.modules` purge, `io.StringIO`,
    `artifacts=`, real gene IDs from `vocab.txt`;
  - `model_registration`: signature, input example, pip requirements, 70M variant;
  - `endpoint_deployment`: workload-type enum, AI Gateway inference table, scale to zero, run gate;
  - `smoke_tests`: registration, endpoint query, output shape;
  - `ai_search` (teddy-002 only): index, embedding dimension, skip if the table is missing;
  - `edge_handling`: not applicable to TEDDY;
  - `forbidden_patterns`: `/local_disk0`, `transformers>=`, fake gene IDs, deprecated
    `AutoCaptureConfigInput`, string workload type.
- **Counts:** teddy-001 has 27 sub-checks (+2 not-applicable groups), teddy-002 has 30 (+1).
- **Tests:** `tests/test_eval_scorers.py` checks that these reproduce the recorded Sep–Oct 2026 scores from
  the recorded Sep–Oct 2026 notebooks.
- **Metric names** match the Sep–Oct 2026 files, except forbidden-pattern names, which MLflow does not allow
  as written: `scorers.mlflow_metric_name()` turns an old key into the new name.
- **Why it is only a record:** thirteen of its sub-checks match literal text from `teddy.md` instead of
  judging their goal (twelve found on the Sep–Oct records, `smoke_tests/endpoint_query` on round 2). That
  makes them wrong both ways:
  - a notebook that reaches the goal another valid way fails (e.g. `tempfile.mkdtemp()` fails `uses_tmp`);
  - a mention in markdown, a comment, a download filter or the task prompt quoted in the first cell passes
    (e.g. "HF_HUB_DISABLE_XET … Not set" passes `hf_xet_disabled`).

### Corrected set — the scorer of record

`corrected_checks.py`, metric names `corrected/<phase>/<check>`.

- **Same sub-checks,** with each of the thirteen flawed ones judged by its intent; the others keep the
  Sep–Oct logic.
- **How:** it reads the notebook's Python cells with `ast`, reusing the reviewed goal logic of
  `practical_checks.py` (download location, Xet, the `transformers` pin, the signature). For two checks it
  also uses the measured outcome (endpoint READY, serving the registered version).
- **Severities** are those of the sub-check it corrects (`corrected_checks.SEVERITY_NOTE`).
- **Used for:** the headline counts, the severity table and the ship gate. Wherever the verbatim result
  differs, reports show it as a ⚑ flag with the reason.
- **Sep–Oct re-scored:** `corrected_checks.py --sep-oct --write` re-scored the Sep–Oct 2026 exports into the
  Sep–Oct round folder's `scores_corrected/` (new files; the recorded files are untouched), so rounds are
  compared corrected against corrected.

| Sub-check | Verbatim flaw | Corrected: passes when |
|---|---|---|
| `download_staging/uses_tmp` | any `/tmp` text passes; `tempfile` / a Volume fails | the download (and any staging copy) goes to `/tmp`, a `tempfile` folder or a Volume, never `/local_disk0` |
| `download_staging/hf_xet_disabled` | the name anywhere passes (even "not set"); a completed download without it fails | Xet disabled with a disabling value before the download, or recorded evidence the download completed: a later load reads from the download's destination (or a copy of it; paths followed through joins, copies and function parameters, never line order alone) and the endpoint was READY serving the registered version |
| `download_staging/deps_pinned` | needs `transformers==<digit>`, fails any `>=` | `transformers` 5.x excluded from the serving requirements, or no `transformers` on the model path |
| `forbidden_patterns/transformers>=` | fails `>=4.41,<5` | no `transformers` requirement allows 5.x |
| `model_registration/infer_signature` | the word only: `ModelSignature(...)` fails, a comment passes | `log_model` gets a signature built by `infer_signature` or `ModelSignature` (or another, with recorded serving evidence) |
| `pyfunc_quality/real_gene_ids` | the text `vocab.txt` (a download list passes) | the logged input example / endpoint payload derives by data flow from real gene data: the vocabulary, a real `.h5ad` file, an upstream tokenizer call, or files that tokenizer wrote (read back from its save path); a typed-in ID list does not count |
| `model_registration/pip_reqs_from_source` | the text `pyproject.toml` (a download filter passes) | the notebook reads the MODEL's dependency file (under the downloaded snapshot or a copy) and its parsed contents flow into `pip_requirements`, or the serving pins agree with it (`models/teddy/family.json` `upstream_dependencies`, TEDDY `pyproject.toml` at commit `8c880aa`) |
| `forbidden_patterns//local_disk0` | any `/local_disk0` text in code, even a message or a fallback | no download, copy or write to `/local_disk0`, unless the path variable gets an alternative (`/tmp`, `tempfile`, Volume) in the try handler / other `if` branch and that variable is what is used (a try with any handler, even `except: pass`, is not a fallback) |
| `endpoint_deployment/sdk_enums` | needs the enum name; a REST body fails | no plain-string `workload_type` passed to an SDK constructor (keywords or `**kwargs`); a string only in a body actually sent to the REST API |
| `endpoint_deployment/ai_gateway_config` | the class name (even an import); a REST body fails | an inference table enabled on an attached AI Gateway config, no `AutoCaptureConfigInput` |
| `ai_search/graceful_skip` | words such as "table … not … exist" anywhere (the quoted prompt, `CREATE TABLE IF NOT EXISTS`) | an existence check on the table the index is built from whose missing branch does not raise, or the read of that table inside try/except that does not re-raise |
| `endpoint_deployment/run_go_gate` | the names `run_go` / `SKIP_DEPLOY` anywhere | every serving-endpoint deploy call runs only when a user flag (widget, notebook / environment parameter, or True/False constant) is on — in its flag-on branch, or after a top-level `if` that stops the notebook when the flag is off — and the flag defaults to off (widget default `"false"`, constant `False`); a flag set from endpoint state is idempotency, not a gate, and a default that cannot be read from the code fails |

The Sep–Oct 2026 runs have no run records, so the alternatives that rely on a record (Xet, signature) cannot
be credited for them.

### New checks

Metric names start with `new_checks/`; they are scored and compared separately, and each result's rationale
says why it passed or failed.

| Check | Passes when the notebook code… | Tasks |
|---|---|---|
| `ai_gateway/inference_table`, `ai_gateway/usage_tracking` | attaches an AI Gateway config to a serving endpoint — `ai_gateway=` on `serving_endpoints.create` / `create_and_wait`, a `put_ai_gateway(...)` call, or an `"ai_gateway"` key in a REST body sent to a `serving-endpoints` URL (or the body of `PUT .../ai-gateway`) — in which the inference table config and the usage tracking config each have `enabled` set to `True`. Both must be enabled in **one** attached config — an inference table on one endpoint and usage tracking on another does not pass both. The `create` / `create_and_wait` / `put_ai_gateway` receiver must be a WorkspaceClient's `serving_endpoints` (`w.serving_endpoints`, `WorkspaceClient(...).serving_endpoints`, or a name assigned from one); `helper.put_ai_gateway(...)` does not count. The config may be built in variables (plain or annotated assignment, any cell), as SDK classes (keyword or positional), `from_dict`, dict literals or `dict(...)`; `enabled` may be a variable holding `True`. Imports alone, or a config never attached, do not count. | both |
| `uc_provenance/uc_registry` | actually registers a model in Unity Catalog: an MLflow registration call — `mlflow.<flavor>.log_model(..., registered_model_name=...)`, `mlflow.register_model(...)`, or `create_registered_model` / `create_model_version` on an object from `MlflowClient(...)` (`mlflow.MlflowClient`, `mlflow.tracking.MlflowClient`; imports and assigned names are followed) — whose model name is three-part (`catalog.schema.model`, resolved through variables and f-strings), **or** `mlflow.set_registry_uri("databricks-uc")` plus a registration whose name is only known at run time (for example from a widget). Fails: the registry URI with no registration; a registration under a resolvable two-part name, even with the UC URI (Unity Catalog only accepts three-part names); look-alike calls on other objects (`custom.log_model`, `factory.create_registered_model`). Names passed through `**kwargs` are read when the dict is a literal or `dict(...)` in the notebook; a `**kwargs` built elsewhere hides the name, so the check is **indeterminate** and scored as not passed (the rationale says "indeterminate"; read that notebook by hand). | both |
| `uc_provenance/provenance_tags` | records the license **and** the Hugging Face revision as tags — run tags, logged-model tags or UC model tags all count (`teddy.md` asks for MLflow tags); the rationale names which kind was found | both |
| `existing_resources/endpoint` (`/table`, `/index` for teddy-002) | looks for an existing endpoint / table / index before the first create or delete call | both |

### Outcome group

Metric names start with `outcome/`. Did what the run built actually work?

- Measured live by `check_outcome.py` (step 9, before teardown) and stored in the run record; the scorer
  reads it, and nothing is re-measured at scoring time.
- Requests use the endpoint's own declared input format.

| Check | Passes when | Tasks |
|---|---|---|
| `endpoint_ready`, `serves_recorded_version` | the endpoint is READY and serves the recorded UC model + version | both |
| `returns_valid_embeddings` | the fixed cells, in the declared format and of different lengths in one request, get one finite, non-zero, 512-dimension embedding each | both |
| `accepts_expression_input` | the endpoint takes cell expression + gene identifiers, not pre-tokenized token IDs (which leave TEDDY's tokenization to every client) | both |
| `vs_endpoint_online`, `index_ready` | the Vector Search endpoint is ONLINE and the index ready | teddy-002 |
| `nn_returns_cell_type` | a nearest-neighbour query with a fixture embedding, asking only for columns the index holds, returns rows that include a cell-type column | teddy-002 |

### Convention check

Metric name `convention/shared_single_cell_format`, read from the same record, scored and compared on its own.

- It passes when the endpoint takes the shared single-cell format (`adata_sparsematrix` + `adata_obs` +
  `adata_var`) of `teddy.md`, also used by `scgpt.md` and Genesis Workbench.
- It is a convention, not correctness: shown in its own column of the severity table and not counted as a
  failure.

### Optional, off by default

- `--reference-endpoint` cosine similarity (information only).
- An LLM judge over each task's `guidelines` (`guideline_judges()`, skill-eval `references/scorer-pack.md`):
  binary, `feedback_value_type=Literal["yes","no"]`, metric `judge/guidelines`. Turn it on with
  `USE_GUIDELINE_JUDGE = True` in the scoring notebook; the judge model must differ from the model that wrote
  the notebooks.

## Severity and environment dependence

Every check in every group has an entry in `expectations.json` → `"checks"`:

- **`severity`:**
  - **critical** — prevents a working endpoint, or yields wrong or mixed embeddings;
  - **major** — works, but unsafe, ungoverned or not reproducible (no inference table, no provenance,
    deletes without looking, unvalidated);
  - **minor** — robustness or style.
- **`source`:** the `SKILL.md` / `teddy.md` line the check rests on (required for critical and major), its
  `quote`, and `why`. The test finds the quote anywhere in the cited file and warns with the line it found if
  the line number has drifted, so editing the skill does not break the answer key.
- **`depends_on`:** why the check holds —
  - `model`: because of TEDDY itself (`HF_HUB_DISABLE_XET` because the Hugging Face repo uses Xet,
    `teddy.md:130`; `transformers==4.41.0` because of the model code, `teddy.md:177`; the 512 dimension);
  - `compute`: because of the compute (`/tmp` vs `/local_disk0`, `teddy.md:160`);
  - `none`.

  With `assumes` naming the environment. The task's `environment_assumed` is serverless GPU notebook compute
  (GPU_1xA10) on AI Runtime v5, with Model Serving GPU_SMALL (`README_environments.md`). Each run records its
  actual compute and runtime (step 8), so a check that depends on compute can be read against what the run
  used.

**What the scoring notebook prints** per pair, run (and extra attempt) and task:

- the raw count of failed checks per severity;
- whether the outcome group passed;
- the convention check, on its own.

No percentages and no single score, so every failure can be read against whether the run actually worked.

### Known false negatives in the verbatim replicated set

The verbatim checks are kept unchanged so they keep reproducing the Sep–Oct 2026 numbers. These valid
alternatives fail them; the corrected set (above) accepts them. `tests/test_eval_scorers.py::test_known_false_negatives`
keeps this list true for the verbatim logic, and `tests/test_corrected_checks.py` tests the corrected one.

| Check | Valid alternative that fails | Where it is documented |
|---|---|---|
| `model_registration/pip_reqs_from_source` | copying the pins without naming `pyproject.toml` / requirements | `teddy.md:165-173` |
| `pyfunc_quality/real_gene_ids` | a payload built from `GeneTokenizer.from_pretrained(model_dir)` without naming `vocab.txt` | `teddy.md:126` |
| `download_staging/uses_tmp` | `tempfile.mkdtemp()` (writes under /tmp, no literal) | — |
| `forbidden_patterns/local_disk0` | compute-adaptive code that tries `/local_disk0` and falls back to `/tmp` | Sep–Oct 2026 ground-truth pattern (`models/teddy/runs_20260924/00_oss_models_test_plan.py`) |
| `endpoint_deployment/ai_gateway_config` | an inference table set in a REST body (`"inference_table_config": {...}`) | `SKILL.md:325` (the new `ai_gateway` checks accept it) |
| `endpoint_deployment/sdk_enums` | REST deployment with `"workload_type": "GPU_SMALL"` (valid for the REST API) | — |
| `ai_search/graceful_skip` | `try/except AnalysisException` around reading the table | — |
| `endpoint_deployment/run_go_gate` | a widget guard with another name (e.g. `deploy`) | `SKILL.md:251` suggests `run_go` |
| `model_registration/infer_signature` | an explicit `ModelSignature(inputs=Schema([...]))` (seen in the 2026-10-27 baselines) | — |
| `download_staging/deps_pinned` | no `transformers` dependency at all, e.g. the encoder re-implemented in plain PyTorch (2026-10-27 `baseline`/teddy-002) | — |

Checked and **not** false negatives:

- **The subprocess download in `teddy.md:137-154`** contains `HF_HUB_DISABLE_XET` and passes
  `download_staging/hf_xet_disabled`.
- **A range pin such as `transformers>=4.30.0,<4.52.0`** fails the verbatim `download_staging/deps_pinned`
  and `forbidden_patterns/transformers`.
  - That range in `SKILL.md:242` is the BioNeMo Geneformer `get_head_mask` workaround (`SKILL.md:241-242`);
    `teddy.md:170` asks for exactly `transformers==4.41.0`.
  - The corrected checks pass any range that excludes 5.x (the failure `teddy.md:177` describes):
    baseline/teddy-001 attempt 2 served TEDDY with `transformers` 4.57.6.

## Scoring and ship gate

**What `score_open_weight_models.py` does**

1. **Checks the records first.** It scores only the pairs that were run (a pair whose records are all
   discarded is skipped), leaves discarded records out, and stops if a counted record has no recorded
   outcome or a protocol gap.
2. **Evaluates each (run, task)** with `mlflow.genai.evaluate`. Each row's `query` is the prompt actually
   pasted in that run: the evalset query, or the `old_prompts.json` text with placeholders as written.
3. **Writes the round's score files** (local, in the git-ignored round folder), using
   `compare_runs.extract_scores()`: `models/teddy/runs_<YYYYMMDD>/scores/<run_id>_scores.json` (all metrics;
   extra attempts as `<run_id>_attempt_<n>_scores.json`) and `<run_id>_<group>.json`.
4. **Compares within each pair only** with `compare_runs.main()` — `baseline` vs `with_skill`, then
   `old_prompt_baseline` vs `old_prompt_with_skill`, primary samples — once per group (corrected, verbatim
   replicated, new checks, outcome, convention, and the judge if enabled).
5. **Prints the severity table** (corrected, new checks and outcome counted).
6. **Updates the files kept in git:**
   - `eval/baseline_scores.json` and `eval/with_skill_scores.json`: the standard pair's primary samples,
     counted groups only (`scorers.update_summary`; other families' tasks in the files are kept);
   - `models/<family>/scores.json`: the round's counted scores, every sample, under `rounds` → the round
     folder's name; other rounds in the file are kept (`scorers.update_family_scores`).

   Once the round's `RESULTS.md` is final, add it to `models/<family>/ROUNDS.md` under "Round details", folded
   like the others, and set the round's `notes` in `scores.json` to its section.

**Ship gate** (skill-eval default)

- A task passes a group only if every check in that group passes.
- The skill ships with at least one task-level win and zero regressions, judged on the standard pair, for the
  corrected set, the new checks and the outcome group each.
- The replication pair is reported next to the Sep–Oct 2026 results, not used for the gate.
- Two tasks show direction, not statistical significance: report raw counts, never percentages.

## Reproducing locally

```bash
uv run --isolated --with pytest --with-requirements skills/open-weight-models/tests/requirements.txt \
    python -m pytest -q skills/open-weight-models/tests
```

Tests that replay a round's records (`runs_<date>/`) run only where those folders are present, and are
skipped elsewhere (for example on CI), with the reason shown.
