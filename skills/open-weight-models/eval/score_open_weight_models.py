# Databricks notebook source
# MAGIC %md
# MAGIC # open-weight-models — TEDDY retest scoring
# MAGIC
# MAGIC Scores the exported Genie Code notebooks for teddy-001 and teddy-002 in three runs:
# MAGIC
# MAGIC | Run | Skill | Prompt |
# MAGIC |---|---|---|
# MAGIC | `baseline` | built-in skills only | `evalset.json` query |
# MAGIC | `with_skill` | built-in skills + open-weight-models | `evalset.json` query |
# MAGIC | `with_skill_old_prompt` | built-in skills + open-weight-models | `old_prompt_with_skill.md` (2026-10 forced-read prompt) |
# MAGIC
# MAGIC Scorers come from `scorers.py`; extraction and the paired comparison from
# MAGIC `skills/skill-eval/scripts/compare_runs.py`. The replicated 2026-10 checks and the
# MAGIC new checks are compared separately. Run only after every run is exported to
# MAGIC `eval/outputs/<run>/<task>.py` and the repo is copied to the workspace (see `README.md`).
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

from scorers import scorers as all_scorers, NEW_CHECK_PREFIX
from compare_runs import extract_scores, main as compare_main
from install_skill_for_run import check_notebook, latest_record, load_records

RUNS = ["baseline", "with_skill", "with_skill_old_prompt"]
OUTPUTS_DIR = EVAL_DIR / "outputs"
SCORES_DIR = OUTPUTS_DIR / "scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)

# COMMAND ----------

with open(EVAL_DIR / "evalset.json", encoding="utf-8") as handle:
    evalset = json.load(handle)
with open(EVAL_DIR / "expectations.json", encoding="utf-8") as handle:
    expectations_list = json.load(handle)

queries = {task["task_id"]: task["query"] for task in evalset}
task_ids = [task["task_id"] for task in evalset]
expectations_by_task = {}
for task in expectations_list:
    merged = dict(task["expectations"])
    merged.update(task.get("deterministic_checks", {}))
    expectations_by_task[task["task_id"]] = merged

sources = {
    run: {task_id: (OUTPUTS_DIR / run / f"{task_id}.py").read_text(encoding="utf-8") for task_id in task_ids}
    for run in RUNS
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Skill pick-up
# MAGIC `install_skill_for_run.py` hid one code in the installed `SKILL.md` and one in
# MAGIC `references/models/teddy.md`. A code in a notebook means Genie Code read that file.
# MAGIC Baseline runs record no codes; with-skill runs should show them. A with-skill run
# MAGIC that shows neither code did not pick the skill up — report it, do not drop it.

# COMMAND ----------

records = load_records(OUTPUTS_DIR / "run_records.json")
for run in RUNS:
    for task_id in task_ids:
        record = latest_record(records, run, task_id)
        print(run, task_id, record["state"], check_notebook(sources[run][task_id], record["codes"]))

# COMMAND ----------

scores = {run: {} for run in RUNS}
for run in RUNS:
    for task_id in task_ids:
        rows = [
            {
                "inputs": {"task_id": task_id, "query": queries[task_id]},
                "outputs": {"notebook_source": sources[run][task_id]},
                "expectations": expectations_by_task[task_id],
            }
        ]
        with mlflow.start_run(run_name=f"{run}/{task_id}"):
            result = mlflow.genai.evaluate(data=rows, scorers=all_scorers)
        scores[run].update(extract_scores(result, rows))

# COMMAND ----------

for run in RUNS:
    with open(EVAL_DIR / f"{run}_scores.json", "w", encoding="utf-8") as handle:
        json.dump(scores[run], handle, indent=2)
    groups = {
        "replicated": {
            tid: {m: v for m, v in metrics.items() if not m.startswith(NEW_CHECK_PREFIX)}
            for tid, metrics in scores[run].items()
        },
        "new_checks": {
            tid: {m: v for m, v in metrics.items() if m.startswith(NEW_CHECK_PREFIX)}
            for tid, metrics in scores[run].items()
        },
    }
    for group, group_scores in groups.items():
        with open(SCORES_DIR / f"{run}_{group}.json", "w", encoding="utf-8") as handle:
            json.dump(group_scores, handle, indent=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paired comparisons
# MAGIC Baseline vs each with-skill run, replicated checks first, then the new checks.

# COMMAND ----------

for candidate in ("with_skill", "with_skill_old_prompt"):
    for group in ("replicated", "new_checks"):
        print(f"\n===== baseline vs {candidate} — {group} =====")
        compare_main(
            [
                str(SCORES_DIR / f"baseline_{group}.json"),
                str(SCORES_DIR / f"{candidate}_{group}.json"),
                "--difficulty",
                str(EVAL_DIR / "expectations.json"),
                "--format",
                "text",
            ]
        )
