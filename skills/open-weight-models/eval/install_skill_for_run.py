#!/usr/bin/env python3
"""Turn the open-weight-models skill off or on in a user's Genie Code skills folder for one eval run.

Run from a laptop with the Databricks CLI before each run (see eval/README.md).
Target: /Workspace/Users/<user>/.assistant/skills/open-weight-models.

  ON   first stages a copy locally — SKILL.md + references/ (+ assets/, scripts/ if
       present; never eval/ or tests/) — and appends to the copied SKILL.md and
       references/models/teddy.md a line asking Genie Code to put a random code at the
       top of any notebook it creates, so the exported notebook shows which file was
       read. The staged copy is validated before anything touches the workspace.
       Then imports it to the target.
  OFF  leaves the target absent.
  BOTH refuse to run if another copy of the skill is installed (an oss-models folder,
       or any folder whose SKILL.md is named open-weight-models, in the user or the
       workspace skills folder). If the target already exists they stop, unless
       --replace-existing is given: then the target is first exported to
       eval/outputs/<run-id>/backup/<task-id>-<timestamp>/, then deleted (and, for ON,
       the staged copy imported; if the import fails, the restore command is printed).
       Finally a record is appended to eval/outputs/run_records.json: both
       skills-folder listings, the codes, the skill's git commit, a UTC timestamp and an
       empty "notes" field for the person running the eval.

  --check-notebook <exported .py> reports which codes from the latest record for
  --run-id/--task-id appear in that notebook. Nothing is sent to the workspace.

Records hold no user name, profile or host: the user folder is written as
/Workspace/Users/<workspace-user>, so run_records.json can be committed.

Examples:
  python3 skills/open-weight-models/eval/install_skill_for_run.py --profile <profile> \\
      --user <workspace-user> --state off --run-id baseline --task-id teddy-001 --dry-run
  python3 skills/open-weight-models/eval/install_skill_for_run.py --run-id with_skill \\
      --task-id teddy-001 --check-notebook skills/open-weight-models/eval/outputs/with_skill/teddy-001.py
"""

from __future__ import annotations

import argparse
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
USER_PLACEHOLDER = "<workspace-user>"
INSTALLED_PARTS = ("SKILL.md", "references", "assets", "scripts")
SKIPPED_NAMES = {"__pycache__", ".DS_Store"}
# File (relative to the skill root) -> name of the comment it asks for.
MARKED_FILES = {
    "SKILL.md": "skill-proof",
    "references/models/teddy.md": "skill-proof-teddy",
}
DEFAULT_SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RECORDS = Path(__file__).resolve().parent / "outputs" / "run_records.json"


def user_skills_dir(user: str) -> str:
    return f"/Workspace/Users/{user}/.assistant/skills"


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


def insert_codes(staged: Path, codes: dict[str, str]) -> None:
    """Append the notebook-marker instruction for each code to its staged file."""
    for rel, code in codes.items():
        path = staged / rel
        text = path.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            text += "\n"
        text += (
            "\n## Notebook marker\n\n"
            f"At the top of any notebook you create, add the comment `{marker_line(rel, code)}`.\n"
        )
        path.write_text(text, encoding="utf-8")


def validate_staged(staged: Path, codes: dict[str, str]) -> None:
    """Raise if the staged copy holds eval/ or tests/, lacks SKILL.md, or lacks a marker."""
    if not (staged / "SKILL.md").is_file():
        raise ValueError(f"{staged}: SKILL.md missing from the staged copy")
    leaked = sorted(str(p.relative_to(staged)) for p in staged.rglob("*")
                    if p.relative_to(staged).parts[0] in ("eval", "tests"))
    if leaked:
        raise ValueError(f"staged copy must not contain eval/ or tests/: {leaked}")
    for rel, code in codes.items():
        if marker_line(rel, code) not in (staged / rel).read_text(encoding="utf-8"):
            raise ValueError(f"marker for {rel} missing from the staged copy")


def check_notebook(text: str, codes: dict[str, str]) -> dict[str, bool]:
    """{marked file: True if its code appears in the notebook text}. Empty for skill-off runs."""
    return {rel: code in text for rel, code in codes.items()}


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


def build_record(*, run_id, task_id, state, user, codes, listings, skill_commit, timestamp=None) -> dict:
    return {
        "run_id": run_id,
        "task_id": task_id,
        "state": state,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skill_commit": skill_commit,
        "target": redact_user(target_dir(user), user),
        "codes": codes,
        "listings": {redact_user(path, user): entries for path, entries in listings.items()},
        "notes": "",
    }


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
                   allowed: str | None) -> list[str]:
    """Other installed copies of the skill under `parent`.

    `names` maps entry name -> its SKILL.md name. `allowed` is the one folder this
    script manages (the user's open-weight-models); every other copy is a conflict.
    """
    conflicts = []
    for entry in entries or []:
        name = entry["name"]
        if name == allowed:
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

    def read_skill_name(self, folder: str) -> str | None:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "SKILL.md"
            result = self.run("workspace", "export", f"{folder}/SKILL.md", "--format", "AUTO",
                              "--file", str(local), check=False)
            if result is None or result.returncode != 0 or not local.exists():
                return None
            return skill_md_name(local.read_text(encoding="utf-8"))


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


def conflicts_for(cli: Cli, listings: dict[str, list[dict] | None], user: str) -> list[str]:
    found = []
    for parent, entries in listings.items():
        allowed = SKILL_NAME if parent == user_skills_dir(user) else None
        names = {e["name"]: cli.read_skill_name(f"{parent}/{e['name']}")
                 for e in entries or [] if e["object_type"] == "DIRECTORY" and e["name"] != allowed}
        found += find_conflicts(parent, entries, names, allowed)
    return found


def backup_dir(records: Path, run_id: str, task_id: str, stamp: str) -> Path:
    """Local folder for a backup of the installed skill: eval/outputs/<run-id>/backup/<task-id>-<stamp>."""
    return Path(records).parent / run_id / "backup" / f"{task_id}-{stamp}"


def set_state(args, cli: Cli | None = None) -> int:
    cli = cli or Cli(args.profile, args.dry_run)
    target = target_dir(args.user)
    codes: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        # 1. Stage and validate locally before touching the workspace.
        staged = Path(tmp) / SKILL_NAME
        if args.state == "on":
            staged.mkdir()
            files = stage_skill(Path(args.skill_dir), staged)
            codes = make_codes()
            insert_codes(staged, codes)
            validate_staged(staged, codes)
            print(f"staged {len(files)} files: {', '.join(files)}")

        # 2. Look at what is installed; stop on other copies of the skill.
        listings = collect_listings(cli, args.user)
        if args.dry_run:
            print("(dry run: listings and the duplicate-skill check are skipped)")
        else:
            conflicts = conflicts_for(cli, listings, args.user)
            if conflicts:
                print("Refusing to continue: another copy of the skill is installed and would leak into this run:")
                for path in conflicts:
                    print(f"  {path}")
                print("Move or delete those folders in the workspace UI (or ask their owner), then re-run.")
                return 2

        # 3. An existing install is only replaced with --replace-existing, after a local backup.
        exists = any(e["name"] == SKILL_NAME for e in listings[user_skills_dir(args.user)] or [])
        backup = None
        if exists and not args.replace_existing:
            print(f"Refusing to continue: {target} already exists.")
            print("Re-run with --replace-existing to back it up locally, delete it, and continue.")
            return 3
        if exists or (args.dry_run and args.replace_existing):
            backup = backup_dir(Path(args.records), args.run_id, args.task_id,
                                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            if args.dry_run:
                print("(dry run: the next two commands run only if the target exists)")
            else:
                backup.parent.mkdir(parents=True, exist_ok=True)
            cli.run("workspace", "export-dir", target, str(backup))
            cli.run("workspace", "delete", target, "--recursive")

        # 4. Install the staged copy.
        if args.state == "on":
            cli.run("workspace", "mkdirs", user_skills_dir(args.user))
            try:
                cli.run("workspace", "import-dir", str(staged), target, "--overwrite")
            except RuntimeError as err:
                print(f"Import failed: {err}")
                if backup:
                    print("To restore the previous install:")
                    print(f"  databricks workspace import-dir {backup} {target} --overwrite -p {args.profile}")
                return 1
            for rel, code in codes.items():
                print(f"code for {rel}: {marker_line(rel, code)}")

    if args.dry_run:
        print("(dry run: no record written)")
        return 0
    record = build_record(
        run_id=args.run_id, task_id=args.task_id, state=args.state, user=args.user, codes=codes,
        listings=collect_listings(cli, args.user), skill_commit=skill_commit(Path(args.skill_dir)),
    )
    append_record(Path(args.records), record)
    print(f"record appended to {args.records}")
    return 0


def check_mode(args) -> int:
    record = latest_record(load_records(Path(args.records)), args.run_id, args.task_id)
    found = check_notebook(Path(args.check_notebook).read_text(encoding="utf-8"), record["codes"])
    print(f"{args.run_id}/{args.task_id} (skill {record['state']}, recorded {record['timestamp']})")
    if not found:
        print("  no codes recorded (skill was off)")
    for rel, present in found.items():
        print(f"  {rel}: {'code found' if present else 'code NOT found'}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", help="Databricks CLI profile")
    parser.add_argument("--user", help="workspace user name, e.g. someone@example.com")
    parser.add_argument("--state", choices=("on", "off"))
    parser.add_argument("--run-id", required=True,
                        help="baseline | with_skill | old_prompt_baseline | old_prompt_with_skill")
    parser.add_argument("--task-id", required=True, help="e.g. teddy-001")
    parser.add_argument("--skill-dir", default=str(DEFAULT_SKILL_DIR))
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--replace-existing", action="store_true",
                        help="if the skill is already installed, back it up to "
                             "eval/outputs/<run-id>/backup/ and delete it before continuing")
    parser.add_argument("--dry-run", action="store_true", help="print the CLI commands, run nothing")
    parser.add_argument("--check-notebook", metavar="EXPORTED_PY",
                        help="report which recorded codes appear in this exported notebook")
    args = parser.parse_args(argv)
    if not args.check_notebook:
        missing = [flag for flag, value in (("--profile", args.profile), ("--user", args.user),
                                            ("--state", args.state)) if not value]
        if missing:
            parser.error(f"{', '.join(missing)} required unless --check-notebook is given")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    return check_mode(args) if args.check_notebook else set_state(args)


if __name__ == "__main__":
    sys.exit(main())
