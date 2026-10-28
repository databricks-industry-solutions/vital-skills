# Databricks notebook source
# MAGIC %md
# MAGIC # open-weight-models — TEDDY retest scoring
# MAGIC
# MAGIC Scores the exported Genie Code notebooks for teddy-001 and teddy-002 in four runs,
# MAGIC forming two pairs that are compared only within themselves:
# MAGIC
# MAGIC | Pair | Run | Skill | Prompt |
# MAGIC |---|---|---|---|
# MAGIC | standard | `baseline` | built-in skills only | `evalset.json` query, names rendered for the run |
# MAGIC | standard | `with_skill` | built-in skills + open-weight-models | `evalset.json` query, names rendered for the run |
# MAGIC | replication | `old_prompt_baseline` | built-in skills only | `models/teddy/old_prompts.json` → `old_prompt_baseline` (Sep–Oct 2026 baseline prompt) |
# MAGIC | replication | `old_prompt_with_skill` | built-in skills + open-weight-models | `models/teddy/old_prompts.json` → `old_prompt_with_skill` (Sep–Oct 2026 forced-read prompt) |
# MAGIC
# MAGIC Scorers come from `corrected_checks.py` and `scorers.py`; extraction and the paired comparison from
# MAGIC `skills/skill-eval/scripts/compare_runs.py`. Groups are compared separately: the
# MAGIC replicated Sep–Oct 2026 checks judged by intent (`corrected/…`, the primary result; the
# MAGIC unchanged Sep–Oct logic is logged too, as `replicated`, a historical record only), the new
# MAGIC checks, the outcome checks (measured live by
# MAGIC `check_outcome.py` before teardown and read from the run records), and the optional
# MAGIC guideline judge — plus the convention check (shared single-cell input format), reported
# MAGIC on its own. Run only after every run is exported to
# MAGIC `eval/models/<family>/runs_<date>/<run>/<task>.py` (`<task>_attempt<n>.py` for a run
# MAGIC repeated as attempts), its outcome recorded, and the repo copied to the workspace (see `README.md`).
# MAGIC
# MAGIC Each task is evaluated on its own: the two tasks have different sub-checks
# MAGIC (teddy-002 adds AI Search and table/index checks), and `extract_scores()` needs every
# MAGIC row of one evaluation to carry the same metrics.

# COMMAND ----------

# MAGIC %pip install "mlflow[databricks]>=3.5.0" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import sys
from pathlib import Path

import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/<workspace-user>/skill-eval-open-weight-models")

# A checkout outside .assistant/skills/, so eval/ is never installed as part of the skill
SKILLS_DIR = Path("/Workspace/Users/<workspace-user>/<repo-folder>/skills")
EVAL_DIR = SKILLS_DIR / "open-weight-models" / "eval"
SCRIPTS_DIR = SKILLS_DIR / "skill-eval" / "scripts"
for directory in (EVAL_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scorers import (
    GROUPS, SUMMARY_FILES, guideline_judges, metric_group, run_prompts, scorers as all_scorers, severity_table,
    update_family_scores, update_summary,
)
from check_outcome import outcome_scores
from corrected_checks import corrected_scorers
from compare_runs import extract_scores, main as compare_main
from family_settings import round_dir
from install_skill_for_run import all_codes, check_notebook, export_file, load_records, protocol_gaps, run_samples

FAMILY = "teddy"
ROUND = None   # runs_<date> folder under models/<family>/; None = family.json current_round
RUN_DIR = round_dir(FAMILY, ROUND, EVAL_DIR)
ALL_PAIRS = [("baseline", "with_skill"), ("old_prompt_baseline", "old_prompt_with_skill")]
SCORES_DIR = RUN_DIR / "scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)

# Optional LLM judge over the task guidelines (reported as its own group). Off by default;
# the judge model must differ from the model that wrote the notebooks.
USE_GUIDELINE_JUDGE = False
JUDGE_MODEL = "databricks:/databricks-gpt-5-mini"
run_scorers = corrected_scorers + all_scorers + (guideline_judges(JUDGE_MODEL) if USE_GUIDELINE_JUDGE else [])

# COMMAND ----------

with open(EVAL_DIR / "evalset.json", encoding="utf-8") as handle:
    evalset = json.load(handle)
with open(EVAL_DIR / "expectations.json", encoding="utf-8") as handle:
    expectations_list = json.load(handle)

task_ids = [task["task_id"] for task in evalset]
# The prompt actually pasted in each run: evalset query (standard pair) or the
# Sep–Oct 2026 prompt text from models/<family>/old_prompts.json (replication pair), with
# the run's names filled in; this is what each eval row logs as its query.
prompts = run_prompts(FAMILY, EVAL_DIR)
expectations_by_task = {}
for task in expectations_list:
    merged = dict(task["expectations"])
    merged.update(task.get("deterministic_checks", {}))
    expectations_by_task[task["task_id"]] = merged

# Score only the pairs that were run (the replication pair is optional): a pair counts once
# any of its runs has a record that is not discarded. Every (run, task) of a counted pair
# then needs one. The latest record of a (run, task) is its primary sample; earlier records
# (repeat attempts, `"attempt": n`) are scored too, as extra samples "<run> attempt <n>".
records = load_records(RUN_DIR / "run_records.json")
PAIRS = [pair for pair in ALL_PAIRS
         if any(run_samples(records, run, task_id) for run in pair for task_id in task_ids)]
RUNS = [run for pair in PAIRS for run in pair]
samples = {}                                          # sample label -> {task_id: record}
for run in RUNS:
    for task_id in task_ids:
        found = run_samples(records, run, task_id)
        assert found, f"{run}/{task_id}: no record that is not discarded"
        samples.setdefault(run, {})[task_id] = found[-1]
        for extra in found[:-1]:
            label = f"{run} attempt {extra.get('attempt') or extra['timestamp']}"
            samples.setdefault(label, {})[task_id] = extra
sources = {label: {tid: (RUN_DIR / export_file(r)).read_text(encoding="utf-8") for tid, r in by_task.items()}
           for label, by_task in samples.items()}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Skill use
# MAGIC Evidence that a with-skill run used the skill: Genie Code's self-report, the chat's
# MAGIC visible steps (Read Skill File on `SKILL.md` / `teddy.md`), and skill-specific choices
# MAGIC in the notebook — read together, none alone is proof. Hidden codes are only present for
# MAGIC runs installed with `--hidden-codes` (off by default: a planted marker instruction is a
# MAGIC prompt-injection pattern Genie Code may rightly refuse, so a missing code shows nothing).
# MAGIC Discarded records are not scored. A run without a recorded outcome (README step 9,
# MAGIC `check_outcome.py`) stops the notebook, as does one whose Skills panel screenshot,
# MAGIC visible steps or self-report is empty (each may say "not available" only with a reason).

# COMMAND ----------

outcomes = {}
for label, by_task in samples.items():
    for task_id, record in by_task.items():
        gaps = protocol_gaps(record)
        assert not gaps, f"{label}/{task_id}: fill the run record first: {gaps}"
        outcomes[(label, task_id)] = outcome_scores(record)
        print(f"{label}/{task_id} (skill {record['state']}, {record['compute']})")
        print("  self-report:", record["skills_self_report"]["text"])
        print("  visible steps:", record["visible_steps"]["text"])
        if all_codes(record):
            print("  hidden codes (opt-in, not proof):", check_notebook(sources[label][task_id], all_codes(record)))

# COMMAND ----------

scores = {label: {} for label in samples}
for label, by_task in samples.items():
    for task_id, record in by_task.items():
        run = record["run_id"]
        rows = [
            {
                # the prompt actually pasted: the record's prompt_used if it deviated, else the rendered one
                "inputs": {"task_id": task_id, "run_id": run,
                           "query": record.get("prompt_used") or prompts[run][task_id]},
                "outputs": {"notebook_source": sources[label][task_id], "outcome": outcomes[(label, task_id)]},
                "expectations": expectations_by_task[task_id],
            }
        ]
        with mlflow.start_run(run_name=f"{label}/{task_id}"):
            result = mlflow.genai.evaluate(data=rows, scorers=run_scorers)
        scores[label].update(extract_scores(result, rows))

# COMMAND ----------

for label in samples:
    stem = label.replace(" ", "_")                     # e.g. baseline_attempt_1
    with open(SCORES_DIR / f"{stem}_scores.json", "w", encoding="utf-8") as handle:
        json.dump(scores[label], handle, indent=2)
    for group in GROUPS:
        group_scores = {
            tid: {m: v for m, v in metrics.items() if metric_group(m) == group}
            for tid, metrics in scores[label].items()
        }
        if any(group_scores.values()):
            with open(SCORES_DIR / f"{stem}_{group}.json", "w", encoding="utf-8") as handle:
                json.dump(group_scores, handle, indent=2)

# The summary pair kept in the repo (eval/baseline_scores.json, eval/with_skill_scores.json): this round's
# primary baseline and with-skill samples, counted groups only. The round folder itself is git-ignored.
for run in SUMMARY_FILES:
    if run in scores:
        print(f"updated {update_summary(EVAL_DIR, run, scores[run])}")
# This round's counted scores, every sample, in the family's scores.json, kept in the repo next to the round's
# notes (once final, add the round folder's RESULTS.md to models/<family>/ROUNDS.md, folded, under "Round details").
print("updated", update_family_scores(RUN_DIR.parent / "scores.json", RUN_DIR.name, scores))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paired comparisons
# MAGIC Within each pair only: `baseline` vs `with_skill`, then `old_prompt_baseline` vs
# MAGIC `old_prompt_with_skill` (the replication of Sep–Oct 2026). One comparison per group:
# MAGIC corrected checks, verbatim replicated checks (historical record), new checks, outcome,
# MAGIC convention, and the judge if enabled. Primary samples only; extra attempts are in the
# MAGIC severity table and their own score files.

# COMMAND ----------

for reference, candidate in PAIRS:
    for group in [g for g in GROUPS if (SCORES_DIR / f"{reference}_{g}.json").exists()]:
        print(f"\n===== {reference} vs {candidate} — {group} =====")
        compare_main(
            [
                str(SCORES_DIR / f"{reference}_{group}.json"),
                str(SCORES_DIR / f"{candidate}_{group}.json"),
                "--difficulty",
                str(EVAL_DIR / "expectations.json"),
                "--format",
                "text",
            ]
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Failed checks by severity
# MAGIC Raw counts per pair, run (and extra attempt) and task (severities from
# MAGIC `expectations.json` "checks"; corrected checks, new checks and outcome — the verbatim
# MAGIC replicated checks are not counted), next to whether the outcome group passed and, on its
# MAGIC own, whether the endpoint uses the shared single-cell input format (not counted as a
# MAGIC failure) — read each failure against whether the run actually worked. No percentages,
# MAGIC no single score.

# COMMAND ----------

print(severity_table(scores, PAIRS))
