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
# MAGIC | replication | `old_prompt_baseline` | built-in skills only | `models/teddy/old_prompt_baseline.md` (Sep–Oct 2026 baseline prompt) |
# MAGIC | replication | `old_prompt_with_skill` | built-in skills + open-weight-models | `models/teddy/old_prompt_with_skill.md` (Sep–Oct 2026 forced-read prompt) |
# MAGIC
# MAGIC Scorers come from `scorers.py`; extraction and the paired comparison from
# MAGIC `skills/skill-eval/scripts/compare_runs.py`. Groups are compared separately: the
# MAGIC replicated Sep–Oct 2026 checks, the new checks, the outcome checks (measured live by
# MAGIC `check_outcome.py` before teardown and read from the run records), and the optional
# MAGIC guideline judge. Run only after every run is exported to
# MAGIC `eval/models/<family>/runs_<date>/<run>/<task>.py`,
# MAGIC its outcome recorded, and the repo copied to the workspace (see `README.md`).
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
    GROUPS, guideline_judges, metric_group, run_prompts, scorers as all_scorers, severity_table,
)
from check_outcome import outcome_scores
from compare_runs import extract_scores, main as compare_main
from family_settings import round_dir
from install_skill_for_run import all_codes, check_notebook, latest_record, load_records, protocol_gaps

FAMILY = "teddy"
ROUND = None   # runs_<date> folder under models/<family>/; None = family.json current_round
RUN_DIR = round_dir(FAMILY, ROUND, EVAL_DIR)
# Score only the pairs that were run (the replication pair is optional).
PAIRS = [("baseline", "with_skill"), ("old_prompt_baseline", "old_prompt_with_skill")]
RUNS = [run for pair in PAIRS for run in pair]
SCORES_DIR = RUN_DIR / "scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)

# Optional LLM judge over the task guidelines (reported as its own group). Off by default;
# the judge model must differ from the model that wrote the notebooks.
USE_GUIDELINE_JUDGE = False
JUDGE_MODEL = "databricks:/databricks-gpt-5-mini"
run_scorers = all_scorers + (guideline_judges(JUDGE_MODEL) if USE_GUIDELINE_JUDGE else [])

# COMMAND ----------

with open(EVAL_DIR / "evalset.json", encoding="utf-8") as handle:
    evalset = json.load(handle)
with open(EVAL_DIR / "expectations.json", encoding="utf-8") as handle:
    expectations_list = json.load(handle)

task_ids = [task["task_id"] for task in evalset]
# The prompt actually pasted in each run: evalset query (standard pair) or the
# Sep–Oct 2026 prompt text from models/<family>/old_prompt_*.md (replication pair), with
# the run's names filled in; this is what each eval row logs as its query.
prompts = run_prompts(FAMILY, EVAL_DIR)
expectations_by_task = {}
for task in expectations_list:
    merged = dict(task["expectations"])
    merged.update(task.get("deterministic_checks", {}))
    expectations_by_task[task["task_id"]] = merged

sources = {
    run: {task_id: (RUN_DIR / run / f"{task_id}.py").read_text(encoding="utf-8") for task_id in task_ids}
    for run in RUNS
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Skill pick-up
# MAGIC `install_skill_for_run.py` hid one code in the installed `SKILL.md` and one in
# MAGIC `references/models/teddy.md`. A code in a notebook means Genie Code read that file.
# MAGIC Baseline runs record no codes; with-skill runs should show them. A with-skill run
# MAGIC that shows neither code did not pick the skill up — report it, do not drop it.
# MAGIC A run marked `discarded` (README step 11) stops the notebook: redo it before scoring.
# MAGIC So does a run without a recorded outcome (README step 9, `check_outcome.py`), and a run
# MAGIC whose Skills panel screenshot, visible steps or self-report is empty (each may say
# MAGIC "not available" only with a note explaining why).

# COMMAND ----------

records = load_records(RUN_DIR / "run_records.json")
outcomes = {}
for run in RUNS:
    for task_id in task_ids:
        record = latest_record(records, run, task_id)
        assert not record.get("discarded"), f"{run}/{task_id} was discarded ({record['discard_reason']}): redo it first"
        gaps = protocol_gaps(record)
        assert not gaps, f"{run}/{task_id}: fill the run record first: {gaps}"
        outcomes[(run, task_id)] = outcome_scores(record)
        print(run, task_id, record["state"], record["compute"], check_notebook(sources[run][task_id], all_codes(record)))

# COMMAND ----------

scores = {run: {} for run in RUNS}
for run in RUNS:
    for task_id in task_ids:
        rows = [
            {
                "inputs": {"task_id": task_id, "run_id": run, "query": prompts[run][task_id]},
                "outputs": {"notebook_source": sources[run][task_id], "outcome": outcomes[(run, task_id)]},
                "expectations": expectations_by_task[task_id],
            }
        ]
        with mlflow.start_run(run_name=f"{run}/{task_id}"):
            result = mlflow.genai.evaluate(data=rows, scorers=run_scorers)
        scores[run].update(extract_scores(result, rows))

# COMMAND ----------

for run in RUNS:
    with open(SCORES_DIR / f"{run}_scores.json", "w", encoding="utf-8") as handle:
        json.dump(scores[run], handle, indent=2)
    for group in GROUPS:
        group_scores = {
            tid: {m: v for m, v in metrics.items() if metric_group(m) == group}
            for tid, metrics in scores[run].items()
        }
        if any(group_scores.values()):
            with open(SCORES_DIR / f"{run}_{group}.json", "w", encoding="utf-8") as handle:
                json.dump(group_scores, handle, indent=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paired comparisons
# MAGIC Within each pair only: `baseline` vs `with_skill`, then `old_prompt_baseline` vs
# MAGIC `old_prompt_with_skill` (the replication of Sep–Oct 2026). One comparison per group:
# MAGIC replicated checks, new checks, outcome, and the judge if enabled.

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
# MAGIC Raw counts per pair, run and task (severities from `expectations.json` "checks"),
# MAGIC next to whether the outcome group passed — read each failure against whether the
# MAGIC run actually worked. No percentages, no single score.

# COMMAND ----------

print(severity_table(scores, PAIRS))
