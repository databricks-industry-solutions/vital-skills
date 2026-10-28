#!/usr/bin/env python3
"""Turn the open-weight-models skill off or on in a user's Genie Code skills folder for one eval run.

Run from a laptop with the Databricks CLI before each run (see eval/README.md).
Target: /Workspace/Users/<user>/.assistant/skills/open-weight-models.

  ON   first stages a copy locally — SKILL.md + references/ (+ assets/, scripts/ if
       present; never eval/ or tests/), unchanged — validates it before anything touches
       the workspace, then imports it to the target.
  OFF  leaves the target absent.
  BOTH refuse to run if another copy of the skill is installed (an oss-models folder,
       or any folder whose SKILL.md is named open-weight-models, in the user or the
       workspace skills folder). If the target already exists they stop, unless
       --replace-existing is given: then the target is first exported to
       <records folder>/<run-id>/backup/<task-id>-<timestamp>/, then deleted (and, for ON,
       the staged copy imported; if the import fails, the restore command is printed).
       Finally a record is appended to eval/models/<family>/runs_<date>/run_records.json
       (default: family.json current_round; --family / --round / --records): the environment
       (both skills-folder listings, size + sha256 of each user-folder SKILL.md and of
       the user's .assistant_instructions.md, whether /Workspace/.assistant/skills and
       /Workspace/.assistant_workspace_instructions.md exist), the codes (if any), the backup
       path and whether it holds the user's "pre-eval install" or an "eval copy" (an
       installed SKILL.md whose sha256 matches an install recorded by this script, or that
       carries a marker line), the skill's git commit, a UTC timestamp,
       the fields filled in by hand after the run ("compute", "resources_created",
       "discarded", "discard_reason", "notes"), and "outcome", written by
       check_outcome.py.

  --restore-original (after all runs) deletes the eval copy and re-imports the earliest
  backup of the user's pre-eval install, or, if there is none, leaves the skill removed.

  --extra-skill-dir <skill folder> (repeatable; default none) also installs another skill
  for the run, staged the same way, recorded under "extra_skills". Extra skills are not
  handled by --restore-original: remove or restore them by hand from their recorded backups.

  --hidden-codes (opt-in, OFF by default) appends to the staged SKILL.md and
  references/models/teddy.md (and each extra skill's SKILL.md) a "## Notebook marker"
  section asking Genie Code to put a random code at the top of any notebook it creates.
  Not proof of skill use: an instruction planted in a skill file to make the agent emit an
  unrelated marker is a prompt-injection pattern, and Genie Code may rightly refuse it (it
  did in with_skill/teddy-001, 2026-10-27, flagging the text as possibly injected). It also
  changes what Genie Code reads. A code in a notebook shows the file was read; a missing code
  shows nothing. Use skill-use evidence instead: the self-report, the chat's visible steps
  and skill-specific choices in the notebook (eval/README.md).

  --check-notebook <exported .py> reports which codes from the latest record for
  --run-id/--task-id appear in that notebook (runs installed with --hidden-codes only).
  Nothing is sent to the workspace.

Records hold no user name, profile or host: the user folder is written as
/Workspace/Users/<workspace-user>, so run_records.json can be committed.

Examples:
  python3 skills/open-weight-models/eval/install_skill_for_run.py --profile <profile> \\
      --user <workspace-user> --state off --run-id baseline --task-id teddy-001 --dry-run
  python3 skills/open-weight-models/eval/install_skill_for_run.py --run-id with_skill \\
      --task-id teddy-001 --check-notebook skills/open-weight-models/eval/models/teddy/runs_20261026/with_skill/teddy-001.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SKILL_NAME = "open-weight-models"
OLD_SKILL_NAME = "oss-models"
WORKSPACE_SKILLS = "/Workspace/.assistant/skills"
WORKSPACE_INSTRUCTIONS = "/Workspace/.assistant_workspace_instructions.md"
USER_PLACEHOLDER = "<workspace-user>"
INSTALLED_PARTS = ("SKILL.md", "references", "assets", "scripts")
SKIPPED_NAMES = {"__pycache__", ".DS_Store"}
# File (relative to the skill root) -> name of the comment it asks for.
MARKED_FILES = {
    "SKILL.md": "skill-proof",
    "references/models/teddy.md": "skill-proof-teddy",
}
PRE_EVAL, EVAL_COPY = "pre-eval install", "eval copy"
SELF_REPORT_PROMPT = ("List every skill from the Skill Registry that you loaded or consulted while "
                      "developing this notebook.")
SELF_REPORT_FOLLOW_UP = "follow-up message after the task"
SELF_REPORT_IN_NOTEBOOK = "in-notebook answer (the replication prompt already asks)"
# What a run created, filled in by hand for check_outcome.py (same keys there).
RESOURCE_KEYS = ("serving_endpoint", "uc_model", "uc_model_version", "delta_table",
                 "vector_search_endpoint", "vector_search_index")
DEFAULT_SKILL_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import family_settings  # noqa: E402 — same folder

NOT_AVAILABLE = "not available"   # allowed for a protocol field only together with a note


def default_records(family: str = family_settings.DEFAULT_FAMILY, round_name: str | None = None) -> Path:
    """models/<family>/<round>/run_records.json (round: family.json current_round)."""
    return family_settings.round_dir(family, round_name) / "run_records.json"


def expected_resources(run_id: str, task_id: str, family: str = family_settings.DEFAULT_FAMILY) -> dict:
    """resources_created pre-filled with the names the run's prompt asks for (family.json);
    uc_model_version and any name the task does not use are left empty."""
    names = {}
    try:
        names = family_settings.resource_names(family_settings.load_family(family), run_id, task_id)
    except KeyError:
        pass
    return {key: names.get(key, "") for key in RESOURCE_KEYS}


def user_skills_dir(user: str) -> str:
    return f"/Workspace/Users/{user}/.assistant/skills"


def user_instructions(user: str) -> str:
    return f"/Workspace/Users/{user}/.assistant_instructions.md"


def target_dir(user: str) -> str:
    return f"{user_skills_dir(user)}/{SKILL_NAME}"


def redact_user(path: str, user: str) -> str:
    return path.replace(f"/Users/{user}", f"/Users/{USER_PLACEHOLDER}")


# ---- staging + codes (pure, unit-tested) ------------------------------------

def stage_skill(skill_dir: Path, dest: Path) -> list[str]:
    """Copy the installable parts of the skill into dest. Returns the copied files, relative and sorted."""
    if not (skill_dir / "SKILL.md").is_file():
        raise FileNotFoundError(f"{skill_dir}/SKILL.md not found")
    for part in INSTALLED_PARTS:
        src = skill_dir / part
        if src.is_file():
            shutil.copy2(src, dest / part)
        elif src.is_dir():
            shutil.copytree(src, dest / part, ignore=shutil.ignore_patterns(*SKIPPED_NAMES))
    return sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())


def make_codes() -> dict[str, str]:
    return {rel: secrets.token_hex(4) for rel in MARKED_FILES}


def marker_line(rel: str, code: str) -> str:
    return f"# {MARKED_FILES[rel]}: {code}"


def extra_marker_line(name: str, code: str) -> str:
    """Marker asked for by an extra skill's staged SKILL.md."""
    return f"# skill-proof-{name}: {code}"


def append_marker(path: Path, line: str) -> None:
    """Append the notebook-marker instruction asking for `line` to a staged file."""
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    text += f"\n## Notebook marker\n\nAt the top of any notebook you create, add the comment `{line}`.\n"
    path.write_text(text, encoding="utf-8")


def insert_codes(staged: Path, codes: dict[str, str]) -> None:
    """Append the notebook-marker instruction for each code to its staged file."""
    for rel, code in codes.items():
        append_marker(staged / rel, marker_line(rel, code))


def stage_extra_skill(skill_dir: Path, dest_root: Path, hidden_code: bool = False) -> tuple:
    """Stage an extra skill (installed alongside open-weight-models for a run); with
    `hidden_code`, its SKILL.md gets its own marker. Returns (name, staged folder, code or None)."""
    skill_dir = Path(skill_dir)
    name = skill_dir.name
    skill_md = skill_dir / "SKILL.md"
    declared = skill_md_name(skill_md.read_text(encoding="utf-8")) if skill_md.is_file() else None
    if declared != name:
        raise ValueError(f"{skill_dir}: SKILL.md name {declared!r} must match the folder name {name!r}")
    if name in (SKILL_NAME, OLD_SKILL_NAME):
        raise ValueError(f"{name} is not an extra skill")
    staged = Path(dest_root) / name
    staged.mkdir()
    stage_skill(skill_dir, staged)
    code = secrets.token_hex(4) if hidden_code else None
    if code:
        append_marker(staged / "SKILL.md", extra_marker_line(name, code))
    validate_staged(staged, {"SKILL.md": code} if code else {}, line_for=lambda rel, c: extra_marker_line(name, c))
    return name, staged, code


def validate_staged(staged: Path, codes: dict[str, str], line_for=None) -> None:
    """Raise if the staged copy holds eval/ or tests/, lacks SKILL.md, or lacks a marker.
    `line_for(rel, code)` gives the expected marker line (default: open-weight-models markers)."""
    line_for = line_for or marker_line
    if not (staged / "SKILL.md").is_file():
        raise ValueError(f"{staged}: SKILL.md missing from the staged copy")
    leaked = sorted(str(p.relative_to(staged)) for p in staged.rglob("*")
                    if p.relative_to(staged).parts[0] in ("eval", "tests"))
    if leaked:
        raise ValueError(f"staged copy must not contain eval/ or tests/: {leaked}")
    for rel, code in codes.items():
        if line_for(rel, code) not in (staged / rel).read_text(encoding="utf-8"):
            raise ValueError(f"marker for {rel} missing from the staged copy")


def check_notebook(text: str, codes: dict[str, str]) -> dict[str, bool]:
    """{marked file: True if its code appears in the notebook text}. Empty for skill-off runs."""
    return {rel: code in text for rel, code in codes.items()}


def all_codes(record: dict) -> dict[str, str]:
    """Every hidden code of a run: open-weight-models files plus "<extra skill>/SKILL.md"
    (empty unless the run was installed with --hidden-codes)."""
    return {**record["codes"], **{f"{e['name']}/SKILL.md": e["code"]
                                  for e in record.get("extra_skills", []) if e.get("code")}}


def expected_panel(state: str, extra_names: list) -> list:
    """Skills the Customizations → Skills panel should list for the run (built-in skills are not shown)."""
    return sorted(([SKILL_NAME] if state == "on" else []) + list(extra_names))


# ---- records (pure, unit-tested) ---------------------------------------------

def load_records(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON array of run records")
    return records


def append_record(path: Path, record: dict) -> None:
    path = Path(path)
    records = load_records(path)
    records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")


def latest_record(records: list[dict], run_id: str, task_id: str) -> dict:
    matches = [r for r in records if r["run_id"] == run_id and r["task_id"] == task_id]
    if not matches:
        raise LookupError(f"no run record for run_id={run_id!r} task_id={task_id!r}")
    return matches[-1]


def run_samples(records: list[dict], run_id: str, task_id: str) -> list[dict]:
    """Records of (run, task) that count, oldest first: discarded ones left out. The last is
    the primary sample; earlier ones (repeat attempts) are extra samples."""
    return [r for r in records if r["run_id"] == run_id and r["task_id"] == task_id and not r.get("discarded")]


def export_file(record: dict) -> str:
    """The exported notebook of a run, relative to the round folder: the record's "export_file" when
    set (round 2 on), else the round-1 naming — <run_id>/<task_id>.py, or
    <run_id>/<task_id>_attempt<n>.py when the record has "attempt"."""
    if record.get("export_file"):
        return record["export_file"]
    attempt = record.get("attempt")
    return f"{record['run_id']}/{record['task_id']}" + (f"_attempt{attempt}" if attempt else "") + ".py"


def build_record(*, run_id, task_id, state, user, codes, environment, skill_commit, backup=None,
                 backup_of=None, extra_skills=(), timestamp=None, family=family_settings.DEFAULT_FAMILY,
                 hidden_codes=False) -> dict:
    """One run record. `skills_panel`, `visible_steps`, `skills_self_report`, `compute`,
    `resources_created`, `discarded`, `discard_reason` and `notes` are filled in by hand
    around the run; `outcome` by check_outcome.py."""
    extra_skills = [dict(e) for e in extra_skills]
    return {
        "run_id": run_id,
        "task_id": task_id,
        "state": state,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skill_commit": skill_commit,
        "target": redact_user(target_dir(user), user),
        "hidden_codes": hidden_codes,                  # --hidden-codes given (off by default; not proof)
        "codes": codes,
        "backup": str(backup) if backup else None,
        "backup_of": backup_of,
        "extra_skills": extra_skills,                  # [{"name", "code", "backup", "backup_of"}]
        "environment": json.loads(redact_user(json.dumps(environment), user)),
        # Filled in by hand around the run (README steps 3, 7, 8, 11, 12):
        "skills_panel": {                              # Customizations → Skills, before pasting the prompt
            "screenshot": "",
            "expected": expected_panel(state, [e["name"] for e in extra_skills]),
            "observed_user_tab": [],
            "observed_all_skills_tab": [],
        },
        "visible_steps": {"text": "", "screenshot": ""},  # the chat's visible step list, after the run
        "skills_self_report": {                        # Genie Code's own answer; read with visible_steps + the notebook
            "source": SELF_REPORT_IN_NOTEBOOK if run_id.startswith("old_prompt") else SELF_REPORT_FOLLOW_UP,
            "text": "",
        },
        "compute": {"type": "", "runtime": ""},       # e.g. "serverless GPU (GPU_1xA10)", "AI Runtime v5"
        "resources_created": expected_resources(run_id, task_id, family),  # check against the notebook
        "prompt_used": None,                           # set only if the pasted prompt differs from the rendered one
        "outcome": None,                               # written by check_outcome.py
        "discarded": False,
        "discard_reason": "",
        "notes": "",
    }


def protocol_gaps(record: dict) -> list:
    """Hand-filled protocol fields still empty in a run record. A field may say "not
    available" only with a reason — after it in the field ("not available: <why>") or in
    `notes`. The scoring notebook refuses runs with gaps."""
    fields = {
        "skills_panel.screenshot": (record.get("skills_panel") or {}).get("screenshot", ""),
        "visible_steps.text": (record.get("visible_steps") or {}).get("text", ""),
        "skills_self_report.text": (record.get("skills_self_report") or {}).get("text", ""),
    }
    gaps = []
    for name, value in fields.items():
        value = str(value).strip()
        if not value:
            gaps.append(f"{name} is empty")
        elif value.lower().startswith(NOT_AVAILABLE):
            reason = value[len(NOT_AVAILABLE):].strip(" :;,.-")
            if not reason and not str(record.get("notes", "")).strip():
                gaps.append(f"{name} is '{NOT_AVAILABLE}' but neither it nor notes say why")
    return gaps


def first_backup(records: list[dict]) -> str | None:
    """The earliest backup of the user's own pre-eval install (relative to the records
    folder), if there was one. Backups of the eval's own copies are skipped."""
    return next((r["backup"] for r in records if r.get("backup") and r.get("backup_of") == PRE_EVAL), None)


def eval_install_hashes(records: list[dict], name: str = SKILL_NAME) -> set:
    """sha256 of every SKILL.md of `name` that this script installed, as fingerprinted in the
    records' environment (collected right after the install): open-weight-models in "on"
    runs, an extra skill in the runs that list it under extra_skills."""
    hashes = set()
    for record in records:
        managed = name == SKILL_NAME and record.get("state") == "on" or \
            name in {e.get("name") for e in record.get("extra_skills") or []}
        if not managed:
            continue
        for entry in (record.get("environment") or {}).get("user_skills") or []:
            if entry.get("name") == name and entry.get("skill_md"):
                hashes.add(entry["skill_md"]["sha256"])
    return hashes


def is_eval_copy(skill_md: bytes | None, name: str = SKILL_NAME, records: list[dict] = ()) -> bool:
    """True if an installed SKILL.md is one this script installed: its sha256 matches an
    install recorded in `records`, or it carries a --hidden-codes marker. (Without markers the
    eval copy is byte-identical to the skill at that commit, so the records decide.)"""
    if skill_md is None:
        return False
    label = MARKED_FILES["SKILL.md"] if name == SKILL_NAME else f"skill-proof-{name}"
    return (f"# {label}: ".encode() in skill_md
            or hashlib.sha256(skill_md).hexdigest() in eval_install_hashes(records, name))


def fingerprint(data: bytes | None) -> dict | None:
    return None if data is None else {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


# ---- listings + conflicts (pure, unit-tested) --------------------------------

def parse_listing(raw: str) -> list[dict]:
    """`databricks workspace list -o json` output -> [{"name", "object_type"}], sorted by name."""
    data = json.loads(raw) if raw.strip() else []
    if isinstance(data, dict):
        data = data.get("objects", [])
    return sorted(
        ({"name": item["path"].rstrip("/").rsplit("/", 1)[-1], "object_type": item.get("object_type")}
         for item in data),
        key=lambda e: e["name"],
    )


def skill_md_name(text: str) -> str | None:
    """The `name:` value from a SKILL.md frontmatter block, if any."""
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not match:
        return None
    name = re.search(r"^name:\s*['\"]?([^'\"\n]+?)['\"]?\s*$", match.group(1), re.M)
    return name.group(1) if name else None


def find_conflicts(parent: str, entries: list[dict] | None, names: dict[str, str | None],
                   allowed) -> list[str]:
    """Other installed copies of the skill under `parent`.

    `names` maps entry name -> its SKILL.md name. `allowed` is the folder name (or set of
    names) this script manages in `parent`; every other copy is a conflict.
    """
    allowed = {allowed} if isinstance(allowed, str) else set(allowed or ())
    conflicts = []
    for entry in entries or []:
        name = entry["name"]
        if name in allowed:
            continue
        if name in (SKILL_NAME, OLD_SKILL_NAME) or names.get(name) == SKILL_NAME:
            conflicts.append(f"{parent}/{name}")
    return conflicts


# ---- CLI side effects ---------------------------------------------------------

class Cli:
    """Runs `databricks` commands; with dry_run it only prints them."""

    def __init__(self, profile: str, dry_run: bool):
        self.profile = profile
        self.dry_run = dry_run

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess | None:
        cmd = ["databricks", *args, "-p", self.profile]
        print("$ " + " ".join(cmd))
        if self.dry_run:
            return None
        result = subprocess.run(cmd, capture_output=True, text=True)
        if check and result.returncode != 0:
            raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}")
        return result

    def list_dir(self, path: str) -> list[dict] | None:
        """Entries of a workspace folder; None if it does not exist (or in dry run)."""
        result = self.run("workspace", "list", path, "-o", "json", check=False)
        if result is None:
            return None
        if result.returncode != 0:
            if "RESOURCE_DOES_NOT_EXIST" in result.stderr or "doesn't exist" in result.stderr:
                return None
            raise RuntimeError(f"listing {path} failed:\n{result.stderr}")
        return parse_listing(result.stdout)

    def export_bytes(self, path: str) -> bytes | None:
        """Content of a workspace file; None if it does not exist (or in dry run)."""
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "file"
            result = self.run("workspace", "export", path, "--format", "AUTO", "--file", str(local), check=False)
            if result is None or result.returncode != 0 or not local.exists():
                return None
            return local.read_bytes()

    def exists(self, path: str) -> bool | None:
        """Whether a workspace object exists; None in dry run."""
        result = self.run("workspace", "get-status", path, check=False)
        return None if result is None else result.returncode == 0

    def read_skill_name(self, folder: str) -> str | None:
        data = self.export_bytes(f"{folder}/SKILL.md")
        return None if data is None else skill_md_name(data.decode("utf-8", errors="replace"))


def skill_commit(skill_dir: Path) -> str:
    def git(*args):
        return subprocess.run(["git", "-C", str(skill_dir), *args], capture_output=True, text=True)

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return "unknown"
    dirty = git("status", "--porcelain", "--", ".").stdout.strip()
    return head.stdout.strip() + ("+uncommitted" if dirty else "")


def collect_listings(cli: Cli, user: str) -> dict[str, list[dict] | None]:
    return {WORKSPACE_SKILLS: cli.list_dir(WORKSPACE_SKILLS), user_skills_dir(user): cli.list_dir(user_skills_dir(user))}


def collect_environment(cli: Cli, user: str) -> dict:
    """What Genie Code could see at run time: both skills folders (user-folder SKILL.md files
    fingerprinted), the user's .assistant_instructions.md, and the workspace-level files."""
    listings = collect_listings(cli, user)
    user_entries = []
    for entry in listings[user_skills_dir(user)] or []:
        item = dict(entry)
        if entry["object_type"] == "DIRECTORY":
            item["skill_md"] = fingerprint(cli.export_bytes(f"{user_skills_dir(user)}/{entry['name']}/SKILL.md"))
        user_entries.append(item)
    return {
        "user_skills": user_entries if listings[user_skills_dir(user)] is not None else None,
        "user_instructions": fingerprint(cli.export_bytes(user_instructions(user))),
        "workspace_skills_exists": listings[WORKSPACE_SKILLS] is not None,
        "workspace_skills": listings[WORKSPACE_SKILLS],
        "workspace_instructions_exists": cli.exists(WORKSPACE_INSTRUCTIONS),
    }


def conflicts_for(cli: Cli, listings: dict[str, list[dict] | None], user: str, extra_names=()) -> list[str]:
    found = []
    for parent, entries in listings.items():
        allowed = {SKILL_NAME, *extra_names} if parent == user_skills_dir(user) else set()
        names = {e["name"]: cli.read_skill_name(f"{parent}/{e['name']}")
                 for e in entries or [] if e["object_type"] == "DIRECTORY" and e["name"] not in allowed}
        found += find_conflicts(parent, entries, names, allowed)
    return found


def backup_dir(records: Path, run_id: str, task_id: str, stamp: str) -> Path:
    """Local folder for a backup of the installed skill: <records folder>/<run-id>/backup/<task-id>-<stamp>."""
    return Path(records).parent / run_id / "backup" / f"{task_id}-{stamp}"


def set_state(args, cli: Cli | None = None) -> int:
    cli = cli or Cli(args.profile, args.dry_run)
    codes: dict[str, str] = {}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.TemporaryDirectory() as tmp:
        # 1. Stage and validate locally before touching the workspace.
        staged = Path(tmp) / SKILL_NAME
        if args.state == "on":
            staged.mkdir()
            files = stage_skill(Path(args.skill_dir), staged)
            if args.hidden_codes:
                codes = make_codes()
                insert_codes(staged, codes)
            validate_staged(staged, codes)
            print(f"staged {len(files)} files: {', '.join(files)}")
        extras_root = Path(tmp) / "extra"
        extras_root.mkdir()
        extras = [stage_extra_skill(Path(d), extras_root, args.hidden_codes) for d in args.extra_skill_dir]
        extra_names = [name for name, _, _ in extras]
        if len(set(extra_names)) != len(extra_names):
            raise ValueError(f"extra skills given twice: {extra_names}")

        # 2. Look at what is installed; stop on other copies of the skill.
        listings = collect_listings(cli, args.user)
        if args.dry_run:
            print("(dry run: listings and the duplicate-skill check are skipped)")
        else:
            conflicts = conflicts_for(cli, listings, args.user, extra_names)
            if conflicts:
                print("Refusing to continue: another copy of the skill is installed and would leak into this run:")
                for path in conflicts:
                    print(f"  {path}")
                print("Move or delete those folders in the workspace UI (or ask their owner), then re-run.")
                return 2

        # 3. An existing install is only replaced with --replace-existing, after a local backup.
        installed = {e["name"] for e in listings[user_skills_dir(args.user)] or []}
        names = [SKILL_NAME] + extra_names
        blocking = [n for n in names if n in installed]
        if blocking and not args.replace_existing:
            for n in blocking:
                print(f"Refusing to continue: {user_skills_dir(args.user)}/{n} already exists.")
            print("Re-run with --replace-existing to back it up locally, delete it, and continue.")
            return 3
        backups = {}                                   # name -> (local backup folder, backup_of)
        records = load_records(Path(args.records))
        for n in names:
            if n in installed or (args.dry_run and args.replace_existing):
                folder = f"{user_skills_dir(args.user)}/{n}"
                backup = backup_dir(Path(args.records), args.run_id, args.task_id, stamp)
                backup = backup if n == SKILL_NAME else backup / "extra" / n
                if args.dry_run:
                    print(f"(dry run: the next commands run only if {n} is installed)")
                else:
                    backup.parent.mkdir(parents=True, exist_ok=True)
                installed_md = cli.export_bytes(f"{folder}/SKILL.md")
                backups[n] = (backup, EVAL_COPY if is_eval_copy(installed_md, n, records) else PRE_EVAL)
                cli.run("workspace", "export-dir", folder, str(backup))
                cli.run("workspace", "delete", folder, "--recursive")

        # 4. Install the staged copies.
        to_install = ([(SKILL_NAME, staged)] if args.state == "on" else []) + [(n, d) for n, d, _ in extras]
        if to_install:
            try:
                cli.run("workspace", "mkdirs", user_skills_dir(args.user))
                for n, folder in to_install:
                    cli.run("workspace", "import-dir", str(folder), f"{user_skills_dir(args.user)}/{n}", "--overwrite")
            except RuntimeError as err:
                print(f"Install failed: {err}")
                if backups:
                    print("To restore the previous install:")
                    print(f"  databricks workspace mkdirs {user_skills_dir(args.user)} -p {args.profile}")
                    for n, (backup, _) in backups.items():
                        print(f"  databricks workspace import-dir {backup} {user_skills_dir(args.user)}/{n} "
                              f"--overwrite -p {args.profile}")
                return 1
            for rel, code in codes.items():
                print(f"code for {rel}: {marker_line(rel, code)}")
            for n, _, code in extras:
                if code:
                    print(f"code for {n}/SKILL.md: {extra_marker_line(n, code)}")

    if args.dry_run:
        print("(dry run: no record written)")
        return 0
    records_dir = Path(args.records).parent
    main_backup, main_backup_of = backups.get(SKILL_NAME, (None, None))
    record = build_record(
        run_id=args.run_id, task_id=args.task_id, state=args.state, user=args.user, codes=codes,
        environment=collect_environment(cli, args.user), skill_commit=skill_commit(Path(args.skill_dir)),
        backup=main_backup.relative_to(records_dir) if main_backup else None, backup_of=main_backup_of,
        family=args.family, hidden_codes=args.hidden_codes,
        extra_skills=[{"name": n, "code": code,
                       "backup": str(backups[n][0].relative_to(records_dir)) if n in backups else None,
                       "backup_of": backups[n][1] if n in backups else None} for n, _, code in extras],
    )
    append_record(Path(args.records), record)
    print(f"record appended to {args.records}")
    return 0


def restore_original(args, cli: Cli | None = None) -> int:
    """After all runs: put back the install that existed before the eval, or leave the skill removed."""
    cli = cli or Cli(args.profile, args.dry_run)
    target = target_dir(args.user)
    records = load_records(Path(args.records))
    recorded = first_backup(records)
    backup = str(Path(args.records).parent / recorded) if recorded else None
    # Check everything before deleting anything.
    if backup and not Path(backup).is_dir() and not args.dry_run:
        print(f"Refusing to continue: the recorded pre-eval backup {backup} is missing locally.")
        print("Nothing was deleted. Restore that folder (or fix the record) first.")
        return 1
    entries = cli.list_dir(user_skills_dir(args.user)) or []
    installed = args.dry_run or any(e["name"] == SKILL_NAME for e in entries)
    if installed and not args.dry_run and not is_eval_copy(cli.export_bytes(f"{target}/SKILL.md"), records=records):
        print(f"Refusing to continue: {target} is not a copy this script installed (its SKILL.md matches "
              "no recorded eval install and has no marker; perhaps the user reinstalled the skill). "
              "Nothing was deleted.")
        return 1
    if installed:
        cli.run("workspace", "delete", target, "--recursive")    # the eval copy
    if recorded is None:
        print("No install existed before the eval (no backup recorded): the skill is left removed.")
        return 0
    cli.run("workspace", "mkdirs", user_skills_dir(args.user))
    cli.run("workspace", "import-dir", backup, target, "--overwrite")
    print(f"Restored the pre-eval install from {backup}.")
    return 0


def check_mode(args) -> int:
    record = latest_record(load_records(Path(args.records)), args.run_id, args.task_id)
    found = check_notebook(Path(args.check_notebook).read_text(encoding="utf-8"), all_codes(record))
    print(f"{args.run_id}/{args.task_id} (skill {record['state']}, recorded {record['timestamp']})")
    if not found:
        print("  no codes recorded (skill off, or installed without --hidden-codes)")
    for rel, present in found.items():
        print(f"  {rel}: {'code found' if present else 'code NOT found'}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", help="Databricks CLI profile")
    parser.add_argument("--user", help="workspace user name, e.g. someone@example.com")
    parser.add_argument("--state", choices=("on", "off"))
    parser.add_argument("--run-id", help="baseline | with_skill | old_prompt_baseline | old_prompt_with_skill")
    parser.add_argument("--task-id", help="e.g. teddy-001")
    parser.add_argument("--skill-dir", default=str(DEFAULT_SKILL_DIR))
    parser.add_argument("--family", default=family_settings.DEFAULT_FAMILY)
    parser.add_argument("--round", help="runs_<date> folder (default: family.json current_round)")
    parser.add_argument("--records", help="default: eval/models/<family>/<round>/run_records.json")
    parser.add_argument("--extra-skill-dir", action="append", default=[], metavar="SKILL_DIR",
                        help="also install this skill for the run (repeatable); default: none")
    parser.add_argument("--hidden-codes", action="store_true",
                        help="opt-in, off by default: append a notebook-marker instruction with a random code "
                             "to the installed skill files. Not proof of skill use (a planted instruction is a "
                             "prompt-injection pattern that Genie Code may rightly refuse); see the docstring")
    parser.add_argument("--replace-existing", action="store_true",
                        help="if the skill is already installed, back it up to "
                             "<records folder>/<run-id>/backup/ and delete it before continuing")
    parser.add_argument("--dry-run", action="store_true", help="print the CLI commands, run nothing")
    parser.add_argument("--check-notebook", metavar="EXPORTED_PY",
                        help="report which recorded codes appear in this exported notebook")
    parser.add_argument("--restore-original", action="store_true",
                        help="after all runs: delete the eval copy and restore the pre-eval install "
                             "from the first recorded backup (or leave the skill removed if there was none)")
    args = parser.parse_args(argv)
    args.records = args.records or str(default_records(args.family, args.round))
    if args.check_notebook:
        needed = (("--run-id", args.run_id), ("--task-id", args.task_id))
    elif args.restore_original:
        needed = (("--profile", args.profile), ("--user", args.user))
    else:
        needed = (("--profile", args.profile), ("--user", args.user), ("--state", args.state),
                  ("--run-id", args.run_id), ("--task-id", args.task_id))
    missing = [flag for flag, value in needed if not value]
    if missing:
        parser.error(f"{', '.join(missing)} required for this mode")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.check_notebook:
        return check_mode(args)
    if args.restore_original:
        return restore_original(args)
    return set_state(args)


if __name__ == "__main__":
    sys.exit(main())
