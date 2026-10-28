#!/usr/bin/env python3
"""Per-model-family eval settings (models/<family>/family.json) and the prompts each run pastes.

Standard library only, so the laptop scripts (install_skill_for_run.py, check_outcome.py),
the scorers and the scoring notebook all read the same settings.

  python3 skills/open-weight-models/eval/family_settings.py prompt --run-id baseline --task-id teddy-002

prints the exact prompt to paste for that run (placeholders such as <catalog> left as
written; fill them the same way in every run).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_FAMILY = "teddy"
RUN_IDS = ("baseline", "with_skill", "old_prompt_baseline", "old_prompt_with_skill")
STANDARD_RUNS = ("baseline", "with_skill")


def family_dir(family: str = DEFAULT_FAMILY, eval_dir: Path = EVAL_DIR) -> Path:
    return Path(eval_dir) / "models" / family


def load_family(family: str = DEFAULT_FAMILY, eval_dir: Path = EVAL_DIR) -> dict:
    with open(family_dir(family, eval_dir) / "family.json", encoding="utf-8") as handle:
        return json.load(handle)


def round_dir(family: str = DEFAULT_FAMILY, round_name: str | None = None, eval_dir: Path = EVAL_DIR) -> Path:
    """models/<family>/runs_<date>/ for this round (default: family.json current_round)."""
    return family_dir(family, eval_dir) / (round_name or load_family(family, eval_dir)["current_round"])


def run_suffix(settings: dict, run_id: str, task_id: str | None = None) -> str:
    """The <run> part of resource names: baseline, with_skill, old_baseline, old_with_skill,
    unless family.json run_suffix_overrides sets another one for this task."""
    override = settings.get("run_suffix_overrides", {}).get(task_id, {}).get(run_id)
    return override or settings["run_suffix"][run_id]


def resource_names(settings: dict, run_id: str, task_id: str) -> dict:
    """{"uc_model", "serving_endpoint", "files", ...} for one run of one task."""
    suffix = run_suffix(settings, run_id, task_id)
    return {key: value.replace("{run}", suffix) for key, value in settings["names"][task_id].items()}


def outcome_checks(settings: dict, task_id: str) -> list:
    """Outcome check names for a task: the common checks, plus search checks for search tasks."""
    out = settings["outcome"]
    return list(out["checks_all_tasks"]) + (
        list(out["checks_vector_search"]) if task_id in out["vector_search_tasks"] else [])


def convention_checks(settings: dict) -> list:
    """Convention check names (reported apart from the outcome group): e.g. whether the
    endpoint takes the shared single-cell input format of the family's reference notes."""
    return list(settings["outcome"].get("convention_checks", []))


def render_query(query: str, settings: dict, run_id: str, task_id: str | None = None) -> str:
    """An evalset.json query with {run} replaced by the run's name suffix."""
    return query.replace("{run}", run_suffix(settings, run_id, task_id))


def old_prompt_substitutions(settings: dict, run_id: str, task_id: str) -> list:
    """[(recorded text, new text)] for an old prompt: resource names/paths from family.json."""
    names = resource_names(settings, run_id, task_id)
    return [(old, new.format(**names)) for old, new in settings["old_prompt_substitutions"][run_id][task_id]]


_RECORDED_EXPORT = re.compile(
    r"/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/results/\w+/oss-00\d\.txt")


def rewrite_old_prompt(recorded: str, settings: dict, run_id: str, task_id: str) -> str:
    """The recorded Sep–Oct 2026 prompt with only the documented changes: resource names and
    paths (family.json), the skill folder, and the export path (the new run folder)."""
    text = recorded
    for old, new in old_prompt_substitutions(settings, run_id, task_id):
        if text.count(old) != 1:
            raise ValueError(f"{run_id}/{task_id}: expected {old!r} exactly once, found {text.count(old)}")
        text = text.replace(old, new)
    text = text.replace(settings["recorded_skill_folder"], settings["installed_skill_folder"])
    exports = _RECORDED_EXPORT.findall(text)
    if len(exports) != 1:
        raise ValueError(f"{run_id}/{task_id}: expected one recorded export path, found {exports}")
    return _RECORDED_EXPORT.sub(f"{settings['workspace_run_folder']}/{run_id}/{task_id}.txt", text)


def old_prompt_blocks(settings: dict, run_id: str, family: str = DEFAULT_FAMILY, eval_dir: Path = EVAL_DIR) -> dict:
    """{task_id: prompt} for an old-prompt run, from models/<family>/old_prompts.json (family.json old_prompts_file)."""
    path = family_dir(family, eval_dir) / settings["old_prompts_file"]
    with open(path, encoding="utf-8") as handle:
        return {task_id: entry["prompt"] for task_id, entry in json.load(handle)[run_id]["prompts"].items()}


def run_prompts(family: str = DEFAULT_FAMILY, eval_dir: Path = EVAL_DIR) -> dict:
    """{run_id: {task_id: the prompt pasted in that run}}."""
    settings = load_family(family, eval_dir)
    with open(Path(eval_dir) / "evalset.json", encoding="utf-8") as handle:
        queries = {t["task_id"]: t["query"] for t in json.load(handle)}
    prompts = {run: {tid: render_query(q, settings, run, tid) for tid, q in queries.items()} for run in STANDARD_RUNS}
    for run in settings["old_prompts"]:
        blocks = old_prompt_blocks(settings, run, family, eval_dir)
        if set(blocks) != set(queries):
            raise ValueError(f"{run}: prompts for {sorted(blocks)}, evalset has {sorted(queries)}")
        prompts[run] = blocks
    return prompts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prompt = sub.add_parser("prompt", help="print the prompt to paste for one run of one task")
    prompt.add_argument("--run-id", required=True, choices=RUN_IDS)
    prompt.add_argument("--task-id", required=True)
    prompt.add_argument("--family", default=DEFAULT_FAMILY)
    names = sub.add_parser("names", help="print the resource names for one run of one task")
    names.add_argument("--run-id", required=True, choices=RUN_IDS)
    names.add_argument("--task-id", required=True)
    names.add_argument("--family", default=DEFAULT_FAMILY)
    args = parser.parse_args(argv)
    if args.command == "prompt":
        print(run_prompts(args.family)[args.run_id][args.task_id])
    else:
        print(json.dumps(resource_names(load_family(args.family), args.run_id, args.task_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
