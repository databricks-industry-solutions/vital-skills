"""Offline tests for eval/install_skill_for_run.py. No Databricks CLI calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_DIR / "eval"))

import install_skill_for_run as inst  # noqa: E402

# Run folders (exported notebooks, run records, per-run scores) are not kept in the repository: they are
# git-ignored and stay with whoever ran the round. Tests that replay them run where the folders are
# present and are skipped elsewhere, e.g. on CI.
SEP_OCT = SKILL_DIR / "eval" / "models" / "teddy" / "runs_20260924"
ROUND_1 = SKILL_DIR / "eval" / "models" / "teddy" / "runs_20261026"
NEEDS_SEP_OCT = pytest.mark.skipif(not SEP_OCT.is_dir(),
                                   reason="run records not in this checkout (runs_20260924 is kept outside the repository)")
NEEDS_ROUND_1 = pytest.mark.skipif(not ROUND_1.is_dir(),
                                   reason="run records not in this checkout (runs_20261026 is kept outside the repository)")


@pytest.fixture()
def fake_skill(tmp_path):
    skill = tmp_path / "open-weight-models"
    (skill / "references" / "models").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: open-weight-models\n---\n# Skill\n")
    (skill / "references" / "models" / "teddy.md").write_text("# TEDDY")
    (skill / "references" / "__pycache__").mkdir()
    (skill / "references" / "__pycache__" / "x.pyc").write_bytes(b"")
    for folder in ("eval", "tests"):
        (skill / folder).mkdir()
        (skill / folder / "expectations.json").write_text("{}")
    (skill / "CHANGELOG.md").write_text("changes")
    return skill


def test_staging_copies_only_installable_parts(fake_skill, tmp_path):
    dest = tmp_path / "staged"
    dest.mkdir()
    files = inst.stage_skill(fake_skill, dest)
    assert files == ["SKILL.md", "references/models/teddy.md"]
    assert not (dest / "eval").exists() and not (dest / "tests").exists()


def test_staging_includes_assets_and_scripts_when_present(fake_skill, tmp_path):
    (fake_skill / "scripts").mkdir()
    (fake_skill / "scripts" / "helper.py").write_text("")
    (fake_skill / "assets").mkdir()
    (fake_skill / "assets" / "a.json").write_text("{}")
    dest = tmp_path / "staged"
    dest.mkdir()
    assert "scripts/helper.py" in inst.stage_skill(fake_skill, dest)
    assert (dest / "assets" / "a.json").exists()


def test_staging_requires_skill_md(tmp_path):
    with pytest.raises(FileNotFoundError):
        inst.stage_skill(tmp_path, tmp_path)


def test_insert_codes_appends_one_distinct_marker_per_file(fake_skill, tmp_path):
    dest = tmp_path / "staged"
    dest.mkdir()
    inst.stage_skill(fake_skill, dest)
    codes = inst.make_codes()
    assert set(codes) == set(inst.MARKED_FILES) and len(set(codes.values())) == 2
    inst.insert_codes(dest, codes)
    skill_md = (dest / "SKILL.md").read_text()
    teddy_md = (dest / "references/models/teddy.md").read_text()
    assert skill_md.startswith("---\nname: open-weight-models\n---")  # frontmatter untouched
    assert f"`# skill-proof: {codes['SKILL.md']}`" in skill_md
    assert f"`# skill-proof-teddy: {codes['references/models/teddy.md']}`" in teddy_md
    assert codes["references/models/teddy.md"] not in skill_md
    # the source skill is never modified
    assert "skill-proof" not in (fake_skill / "SKILL.md").read_text()


def test_check_notebook_reports_each_code():
    codes = {"SKILL.md": "aaaa1111", "references/models/teddy.md": "bbbb2222"}
    text = "# Databricks notebook source\n# skill-proof: aaaa1111\nprint(1)\n"
    assert inst.check_notebook(text, codes) == {"SKILL.md": True, "references/models/teddy.md": False}
    assert inst.check_notebook(text, {}) == {}


def test_records_append_and_latest(tmp_path):
    path = tmp_path / "outputs" / "run_records.json"
    assert inst.load_records(path) == []
    first = inst.build_record(run_id="baseline", task_id="teddy-001", state="off", user="someone@example.com",
                              codes={}, environment={}, skill_commit="abc", timestamp="t1")
    second = dict(first, timestamp="t2")
    inst.append_record(path, first)
    inst.append_record(path, second)
    records = json.loads(path.read_text())
    assert [r["timestamp"] for r in records] == ["t1", "t2"]
    assert inst.latest_record(records, "baseline", "teddy-001")["timestamp"] == "t2"
    with pytest.raises(LookupError):
        inst.latest_record(records, "with_skill", "teddy-001")


def test_record_holds_no_user_name():
    record = inst.build_record(
        run_id="with_skill", task_id="teddy-002", state="on", user="someone@example.com",
        codes={"SKILL.md": "c"}, skill_commit="abc",
        environment={"note": f"{inst.user_skills_dir('someone@example.com')}/open-weight-models"},
    )
    assert "someone@example.com" not in json.dumps(record)
    assert record["target"] == "/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models"


def test_parse_listing_and_skill_name():
    raw = json.dumps([
        {"path": "/Workspace/Users/u/.assistant/skills/oss-models", "object_type": "DIRECTORY"},
        {"path": "/Workspace/Users/u/.assistant/skills/bulk-rnaseq", "object_type": "DIRECTORY"},
    ])
    assert [e["name"] for e in inst.parse_listing(raw)] == ["bulk-rnaseq", "oss-models"]
    assert inst.parse_listing("") == []
    assert inst.skill_md_name("---\nname: open-weight-models\nversion: 1\n---\n") == "open-weight-models"
    assert inst.skill_md_name("---\nname: \"x\"\n---\n") == "x"
    assert inst.skill_md_name("# no frontmatter") is None


def test_find_conflicts():
    parent = "/Workspace/Users/u/.assistant/skills"
    entries = [{"name": n, "object_type": "DIRECTORY"} for n in
               ("open-weight-models", "oss-models", "my-copy", "bulk-rnaseq")]
    names = {"my-copy": "open-weight-models", "bulk-rnaseq": "bulk-rnaseq"}
    assert inst.find_conflicts(parent, entries, names, allowed="open-weight-models") == [
        f"{parent}/oss-models", f"{parent}/my-copy"]
    # workspace-level folder: even an open-weight-models copy is a conflict
    assert inst.find_conflicts(inst.WORKSPACE_SKILLS, entries[:1], {}, allowed=None) == [
        f"{inst.WORKSPACE_SKILLS}/open-weight-models"]
    assert inst.find_conflicts(parent, None, {}, allowed="open-weight-models") == []


def test_args_require_profile_user_state_unless_checking():
    with pytest.raises(SystemExit):
        inst.parse_args(["--run-id", "baseline", "--task-id", "teddy-001"])
    with pytest.raises(SystemExit):
        inst.parse_args(["--profile", "P", "--user", "u", "--state", "on"])  # run/task ids missing
    args = inst.parse_args(["--run-id", "with_skill", "--task-id", "teddy-001", "--check-notebook", "nb.py"])
    assert args.check_notebook == "nb.py"


def test_validate_staged_rejects_leaks_and_missing_markers(fake_skill, tmp_path):
    dest = tmp_path / "staged"
    dest.mkdir()
    inst.stage_skill(fake_skill, dest)
    codes = inst.make_codes()
    with pytest.raises(ValueError, match="marker"):
        inst.validate_staged(dest, codes)
    inst.insert_codes(dest, codes)
    inst.validate_staged(dest, codes)
    (dest / "eval").mkdir()
    (dest / "eval" / "expectations.json").write_text("{}")
    with pytest.raises(ValueError, match="eval/ or tests/"):
        inst.validate_staged(dest, codes)


def test_record_has_hand_filled_fields():
    record = inst.build_record(run_id="baseline", task_id="teddy-001", state="off", user="u",
                               codes={}, environment={}, skill_commit="abc")
    assert (record["notes"], record["discarded"], record["discard_reason"], record["backup"], record["backup_of"]) == (
        "", False, "", None, None)
    assert record["compute"] == {"type": "", "runtime": ""} and record["outcome"] is None
    assert set(record["resources_created"]) == set(inst.RESOURCE_KEYS)


def test_first_backup_and_fingerprint():
    records = [{"backup": None}, {"backup": "with_skill/backup/a", "backup_of": inst.EVAL_COPY},
               {"backup": "baseline/backup/teddy-001-x", "backup_of": inst.PRE_EVAL}]
    assert inst.first_backup(records) == "baseline/backup/teddy-001-x"
    assert inst.first_backup(records[:2]) is None  # only eval copies were backed up
    assert inst.is_eval_copy(b"---\nname: x\n---\n## Notebook marker\n`# skill-proof: ab12`\n")
    assert not inst.is_eval_copy(b"---\nname: open-weight-models\n---\n") and not inst.is_eval_copy(None)
    assert inst.fingerprint(b"abc") == {
        "bytes": 3, "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}
    assert inst.fingerprint(None) is None


# ---- set_state ordering, CLI mocked -------------------------------------------

USER = "someone@example.com"
TARGET = inst.target_dir(USER)


class FakeCli(inst.Cli):
    """Records every command; serves canned listings and files; can fail a step."""

    def __init__(self, installed: bool, fail: str | None = None, files: dict | None = None):
        super().__init__("PROFILE", dry_run=False)
        self.installed = installed
        self.fail = fail
        self.files = files or {}
        self.calls = []

    def run(self, *args, check=True):
        import subprocess

        self.calls.append(args)
        done = subprocess.CompletedProcess(args, 0, "", "")
        missing = subprocess.CompletedProcess(args, 1, "", "RESOURCE_DOES_NOT_EXIST")
        if args[1] == self.fail:
            raise RuntimeError(f"{self.fail} failed")
        if args[:2] == ("workspace", "list"):
            if args[2] == inst.WORKSPACE_SKILLS:
                return missing
            entries = [{"path": TARGET, "object_type": "DIRECTORY"}] if self.installed else []
            entries.append({"path": f"{inst.user_skills_dir(USER)}/bulk-rnaseq", "object_type": "DIRECTORY"})
            return subprocess.CompletedProcess(args, 0, json.dumps(entries), "")
        if args[:2] == ("workspace", "export"):
            if args[2] not in self.files:
                return missing
            Path(args[args.index("--file") + 1]).write_bytes(self.files[args[2]])
            return done
        if args[:2] == ("workspace", "get-status"):
            return done if args[2] in self.files else missing
        return done

    def mutations(self):
        return [c[1] for c in self.calls if c[1] in ("export-dir", "delete", "mkdirs", "import-dir")]


def _args(fake_skill, tmp_path, state="on", replace=False, hidden_codes=False):
    return inst.parse_args(
        ["--profile", "PROFILE", "--user", USER, "--state", state, "--run-id", "with_skill",
         "--task-id", "teddy-001", "--skill-dir", str(fake_skill),
         "--records", str(tmp_path / "outputs" / "run_records.json")]
        + (["--replace-existing"] if replace else [])
        + (["--hidden-codes"] if hidden_codes else [])
    )


@pytest.fixture()
def no_git(monkeypatch):
    monkeypatch.setattr(inst, "skill_commit", lambda _: "abc")


def test_on_fresh_install_imports_without_deleting(fake_skill, tmp_path, no_git):
    cli = FakeCli(installed=False)
    assert inst.set_state(_args(fake_skill, tmp_path), cli) == 0
    assert cli.mutations() == ["mkdirs", "import-dir"]
    record = inst.load_records(tmp_path / "outputs" / "run_records.json")[-1]
    assert record["codes"] == {} and record["hidden_codes"] is False and record["notes"] == ""


def test_hidden_codes_are_off_by_default_and_opt_in(fake_skill, tmp_path, no_git, monkeypatch):
    """Default install: the staged files are the skill's own, byte for byte (no planted
    marker instruction). --hidden-codes adds the markers and records the codes."""
    staged_texts = {}

    def capture(staged, codes, line_for=None):
        staged_texts.update({rel: (staged / rel).read_text() for rel in inst.MARKED_FILES})
    monkeypatch.setattr(inst, "validate_staged", capture)
    assert inst.set_state(_args(fake_skill, tmp_path), FakeCli(installed=False)) == 0
    assert staged_texts == {rel: (fake_skill / rel).read_text() for rel in inst.MARKED_FILES}
    monkeypatch.undo()
    assert inst.set_state(_args(fake_skill, tmp_path, hidden_codes=True), FakeCli(installed=False)) == 0
    record = inst.load_records(tmp_path / "outputs" / "run_records.json")[-1]
    assert set(record["codes"]) == set(inst.MARKED_FILES) and record["hidden_codes"] is True


def test_eval_copy_without_marker_is_recognised_from_the_records():
    skill_md = b"---\nname: open-weight-models\n---\n# Skill\n"
    on = {"state": "on", "extra_skills": [], "environment": {"user_skills": [
        {"name": "open-weight-models", "skill_md": inst.fingerprint(skill_md)}]}}
    off = {"state": "off", "extra_skills": [], "environment": {"user_skills": [
        {"name": "bulk-rnaseq", "skill_md": inst.fingerprint(b"other")}]}}
    assert inst.is_eval_copy(skill_md, records=[off, on])
    assert not inst.is_eval_copy(skill_md, records=[off])          # before any eval install
    assert not inst.is_eval_copy(b"other", "bulk-rnaseq", records=[off])  # not installed by the eval
    extra = {"state": "off", "extra_skills": [{"name": "bulk-rnaseq"}], "environment": off["environment"]}
    assert inst.is_eval_copy(b"other", "bulk-rnaseq", records=[extra])


def test_existing_target_stops_without_flag(fake_skill, tmp_path, no_git):
    cli = FakeCli(installed=True)
    assert inst.set_state(_args(fake_skill, tmp_path), cli) == 3
    assert cli.mutations() == []
    assert not (tmp_path / "outputs" / "run_records.json").exists()


@pytest.mark.parametrize("state, expected", [
    ("on", ["export-dir", "delete", "mkdirs", "import-dir"]),
    ("off", ["export-dir", "delete"]),
])
def test_replace_existing_backs_up_before_delete(fake_skill, tmp_path, no_git, state, expected):
    cli = FakeCli(installed=True)
    assert inst.set_state(_args(fake_skill, tmp_path, state=state, replace=True), cli) == 0
    assert cli.mutations() == expected
    export = next(c for c in cli.calls if c[1] == "export-dir")
    assert export[2] == TARGET
    assert Path(export[3]).parent == tmp_path / "outputs" / "with_skill" / "backup"


def test_staging_failure_touches_nothing(tmp_path, no_git):
    cli = FakeCli(installed=True)
    with pytest.raises(FileNotFoundError):
        inst.set_state(_args(tmp_path / "missing-skill", tmp_path, replace=True), cli)
    assert cli.calls == []


@pytest.mark.parametrize("step", ["mkdirs", "import-dir"])
def test_install_failure_prints_restore(fake_skill, tmp_path, no_git, capsys, step):
    cli = FakeCli(installed=True, fail=step)
    assert inst.set_state(_args(fake_skill, tmp_path, replace=True), cli) == 1
    out = capsys.readouterr().out
    assert "To restore the previous install" in out and "import-dir" in out and TARGET in out
    assert not (tmp_path / "outputs" / "run_records.json").exists()


def test_record_holds_environment_and_relative_backup(fake_skill, tmp_path, no_git):
    files = {
        f"{inst.user_skills_dir(USER)}/bulk-rnaseq/SKILL.md": b"---\nname: bulk-rnaseq\n---\n",
        inst.user_instructions(USER): b"be nice",
    }
    cli = FakeCli(installed=True, files=files)
    assert inst.set_state(_args(fake_skill, tmp_path, replace=True), cli) == 0
    record = inst.load_records(tmp_path / "outputs" / "run_records.json")[-1]
    env = record["environment"]
    bulk = next(e for e in env["user_skills"] if e["name"] == "bulk-rnaseq")
    assert bulk["skill_md"] == inst.fingerprint(files[f"{inst.user_skills_dir(USER)}/bulk-rnaseq/SKILL.md"])
    assert env["user_instructions"] == inst.fingerprint(b"be nice")
    assert env["workspace_skills_exists"] is False and env["workspace_instructions_exists"] is False
    assert record["backup"].startswith("with_skill/backup/teddy-001-")  # relative: no local paths
    assert record["backup_of"] == inst.PRE_EVAL  # the installed SKILL.md had no marker
    assert USER not in json.dumps(record) and str(tmp_path) not in json.dumps(record)


EVAL_COPY_FILES = {f"{TARGET}/SKILL.md": b"---\nname: open-weight-models\n---\n\n## Notebook marker\n\n`# skill-proof: ab12cd34`\n"}


def _restore_args(tmp_path):
    return inst.parse_args(["--profile", "PROFILE", "--user", USER, "--restore-original",
                            "--records", str(tmp_path / "outputs" / "run_records.json")])


def test_restore_original_reimports_first_backup(tmp_path):
    records = tmp_path / "outputs" / "run_records.json"
    (tmp_path / "outputs" / "baseline" / "backup" / "teddy-001-x").mkdir(parents=True)
    inst.append_record(records, {"run_id": "baseline", "task_id": "teddy-001",
                                 "backup": "baseline/backup/teddy-001-x", "backup_of": inst.PRE_EVAL})
    inst.append_record(records, {"run_id": "with_skill", "task_id": "teddy-002",
                                 "backup": "with_skill/backup/y", "backup_of": inst.EVAL_COPY})
    cli = FakeCli(installed=True, files=EVAL_COPY_FILES)
    assert inst.restore_original(_restore_args(tmp_path), cli) == 0
    assert cli.mutations() == ["delete", "mkdirs", "import-dir"]
    imported = next(c for c in cli.calls if c[1] == "import-dir")
    assert imported[2] == str(tmp_path / "outputs" / "baseline" / "backup" / "teddy-001-x") and imported[3] == TARGET


def test_restore_original_without_backup_leaves_removed(tmp_path, capsys):
    inst.append_record(tmp_path / "outputs" / "run_records.json", {"run_id": "baseline", "backup": None})
    cli = FakeCli(installed=True, files=EVAL_COPY_FILES)
    assert inst.restore_original(_restore_args(tmp_path), cli) == 0
    assert cli.mutations() == ["delete"]
    assert "left removed" in capsys.readouterr().out


def test_backup_of_eval_copy_is_labelled(fake_skill, tmp_path, no_git):
    files = {f"{TARGET}/SKILL.md": b"---\nname: open-weight-models\n---\n\n## Notebook marker\n\n`# skill-proof: ab12cd34`\n"}
    cli = FakeCli(installed=True, files=files)
    assert inst.set_state(_args(fake_skill, tmp_path, replace=True), cli) == 0
    record = inst.load_records(tmp_path / "outputs" / "run_records.json")[-1]
    assert record["backup_of"] == inst.EVAL_COPY
    assert inst.first_backup([record]) is None


# ---- extra skills, skills panel, visible steps, self-report -------------------------

def _extra_skill(tmp_path, name="bulk-rnaseq"):
    skill = tmp_path / "extras" / name
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\n---\n# {name}\n")
    (skill / "references" / "r.md").write_text("ref")
    (skill / "eval").mkdir()
    (skill / "eval" / "expectations.json").write_text("{}")
    return skill


def test_stage_extra_skill_gets_its_own_marker(tmp_path):
    root = tmp_path / "staged"
    root.mkdir()
    name, staged, code = inst.stage_extra_skill(_extra_skill(tmp_path), root, hidden_code=True)
    assert name == "bulk-rnaseq" and not (staged / "eval").exists()
    assert f"`# skill-proof-bulk-rnaseq: {code}`" in (staged / "SKILL.md").read_text()
    assert inst.is_eval_copy((staged / "SKILL.md").read_bytes(), "bulk-rnaseq")
    assert not inst.is_eval_copy((staged / "SKILL.md").read_bytes())  # not an open-weight-models marker
    plain_root = tmp_path / "plain"
    plain_root.mkdir()
    _, plain, none = inst.stage_extra_skill(_extra_skill(tmp_path / "again"), plain_root)
    assert none is None and "Notebook marker" not in (plain / "SKILL.md").read_text()


def test_stage_extra_skill_rejects_bad_names(tmp_path):
    skill = _extra_skill(tmp_path, "my-skill")
    (skill / "SKILL.md").write_text("---\nname: other\n---\n")
    with pytest.raises(ValueError, match="must match the folder name"):
        inst.stage_extra_skill(skill, tmp_path)
    with pytest.raises(ValueError, match="not an extra skill"):
        inst.stage_extra_skill(_extra_skill(tmp_path, "oss-models"), tmp_path)


def test_all_codes_and_expected_panel():
    record = {"codes": {"SKILL.md": "a", "references/models/teddy.md": "b"},
              "extra_skills": [{"name": "bulk-rnaseq", "code": "c"}]}
    assert inst.all_codes(record) == {"SKILL.md": "a", "references/models/teddy.md": "b", "bulk-rnaseq/SKILL.md": "c"}
    assert inst.all_codes({"codes": {}}) == {}
    assert inst.all_codes({"codes": {}, "extra_skills": [{"name": "bulk-rnaseq", "code": None}]}) == {}
    assert inst.expected_panel("off", []) == []
    assert inst.expected_panel("on", []) == ["open-weight-models"]
    assert inst.expected_panel("on", ["bulk-rnaseq"]) == ["bulk-rnaseq", "open-weight-models"]


def test_record_protocol_fields():
    record = inst.build_record(run_id="baseline", task_id="teddy-001", state="off", user="u",
                               codes={}, environment={}, skill_commit="abc")
    assert record["skills_panel"] == {"screenshot": "", "expected": [], "observed_user_tab": [],
                                      "observed_all_skills_tab": []}
    assert record["visible_steps"] == {"text": "", "screenshot": ""}
    assert record["skills_self_report"] == {"source": inst.SELF_REPORT_FOLLOW_UP, "text": ""}
    assert record["extra_skills"] == []
    old = inst.build_record(run_id="old_prompt_with_skill", task_id="teddy-001", state="on", user="u",
                            codes={"SKILL.md": "a"}, environment={}, skill_commit="abc")
    assert old["skills_self_report"]["source"] == inst.SELF_REPORT_IN_NOTEBOOK
    assert old["skills_panel"]["expected"] == ["open-weight-models"]


@NEEDS_SEP_OCT
def test_self_report_prompt_is_the_sep_oct_2026_line():
    recorded = (SEP_OCT / "results" / "with_skill" / "oss-001.txt").read_text()
    line = recorded.splitlines()[22]
    assert line == "# MAGIC > At the end, l" + inst.SELF_REPORT_PROMPT[1:]


def test_extra_skill_installed_and_recorded(fake_skill, tmp_path, no_git):
    extra = _extra_skill(tmp_path, "cohort-builder")  # not listed as installed by FakeCli
    args = inst.parse_args(
        ["--profile", "PROFILE", "--user", USER, "--state", "on", "--run-id", "with_skill", "--task-id", "teddy-001",
         "--skill-dir", str(fake_skill), "--records", str(tmp_path / "outputs" / "run_records.json"),
         "--extra-skill-dir", str(extra), "--hidden-codes"])
    cli = FakeCli(installed=False)
    assert inst.set_state(args, cli) == 0
    imports = [c[3] for c in cli.calls if c[1] == "import-dir"]
    assert imports == [TARGET, f"{inst.user_skills_dir(USER)}/cohort-builder"]
    record = inst.load_records(tmp_path / "outputs" / "run_records.json")[-1]
    (entry,) = record["extra_skills"]
    assert entry["name"] == "cohort-builder" and entry["backup"] is None
    assert record["skills_panel"]["expected"] == ["cohort-builder", "open-weight-models"]
    notebook = tmp_path / "nb.py"
    notebook.write_text(f"# skill-proof-cohort-builder: {entry['code']}\n")
    assert inst.check_notebook(notebook.read_text(), inst.all_codes(record)) == {
        "SKILL.md": False, "references/models/teddy.md": False, "cohort-builder/SKILL.md": True}


def test_installed_extra_skill_needs_replace_and_is_backed_up(fake_skill, tmp_path, no_git):
    extra = _extra_skill(tmp_path)
    base = ["--profile", "PROFILE", "--user", USER, "--state", "off", "--run-id", "baseline", "--task-id",
            "teddy-001", "--skill-dir", str(fake_skill), "--records", str(tmp_path / "outputs" / "run_records.json"),
            "--extra-skill-dir", str(extra)]
    cli = FakeCli(installed=False)  # FakeCli always lists bulk-rnaseq as installed
    assert inst.set_state(inst.parse_args(base), cli) == 3
    assert cli.mutations() == []
    cli = FakeCli(installed=False)
    assert inst.set_state(inst.parse_args(base + ["--replace-existing"]), cli) == 0
    assert cli.mutations() == ["export-dir", "delete", "mkdirs", "import-dir"]
    (entry,) = inst.load_records(tmp_path / "outputs" / "run_records.json")[-1]["extra_skills"]
    assert entry["backup"].startswith("baseline/backup/teddy-001-") and entry["backup"].endswith("/extra/bulk-rnaseq")
    assert entry["backup_of"] == inst.PRE_EVAL


def test_restore_original_refuses_a_copy_without_marker(tmp_path, capsys):
    inst.append_record(tmp_path / "outputs" / "run_records.json", {"run_id": "baseline", "backup": None})
    user_copy = {f"{TARGET}/SKILL.md": b"---\nname: open-weight-models\n---\n# the user's own install\n"}
    cli = FakeCli(installed=True, files=user_copy)
    assert inst.restore_original(_restore_args(tmp_path), cli) == 1
    assert cli.mutations() == []
    assert "not a copy this script installed" in capsys.readouterr().out


def test_restore_original_deletes_an_unmarked_eval_copy_it_recorded(tmp_path):
    installed = b"---\nname: open-weight-models\n---\n# Skill\n"   # no marker: default install
    inst.append_record(tmp_path / "outputs" / "run_records.json", {
        "run_id": "with_skill", "state": "on", "backup": None, "extra_skills": [],
        "environment": {"user_skills": [{"name": inst.SKILL_NAME, "skill_md": inst.fingerprint(installed)}]}})
    cli = FakeCli(installed=True, files={f"{TARGET}/SKILL.md": installed})
    assert inst.restore_original(_restore_args(tmp_path), cli) == 0
    assert cli.mutations() == ["delete"]


def test_restore_original_refuses_when_recorded_backup_is_missing(tmp_path, capsys):
    inst.append_record(tmp_path / "outputs" / "run_records.json",
                       {"run_id": "baseline", "backup": "baseline/backup/gone", "backup_of": inst.PRE_EVAL})
    cli = FakeCli(installed=True, files=EVAL_COPY_FILES)
    assert inst.restore_original(_restore_args(tmp_path), cli) == 1
    assert cli.mutations() == [] and cli.calls == []
    assert "missing locally" in capsys.readouterr().out


# ---- family layout, pre-filled names, protocol preflight ------------------------------

def test_default_records_follow_family_round():
    assert inst.default_records() == SKILL_DIR / "eval" / "models" / "teddy" / "runs_20261026" / "run_records.json"
    args = inst.parse_args(["--run-id", "baseline", "--task-id", "teddy-002", "--check-notebook", "nb.py"])
    assert args.records == str(inst.default_records())
    args = inst.parse_args(["--run-id", "baseline", "--task-id", "teddy-002", "--check-notebook", "nb.py",
                            "--round", "runs_20991231"])
    assert args.records.endswith("models/teddy/runs_20991231/run_records.json")


def test_resources_prefilled_from_family_names():
    two = inst.expected_resources("baseline", "teddy-002")
    assert two == {
        "serving_endpoint": "owm_teddy_ep_002_baseline",
        "uc_model": "<catalog>.skills_owm_202610.owm_teddy_model_002_baseline",
        "uc_model_version": "",
        "delta_table": "<catalog>.skills_owm_202610.owm_teddy_cells_002_baseline",
        "vector_search_endpoint": "owm_teddy_vs_002_baseline",
        "vector_search_index": "<catalog>.skills_owm_202610.owm_teddy_index_002_baseline",
    }
    one = inst.expected_resources("old_prompt_with_skill", "teddy-001")
    assert one["serving_endpoint"] == "owm_teddy_ep_001_old_with_skill" and one["delta_table"] == ""
    assert inst.expected_resources("baseline", "unknown-task") == {k: "" for k in inst.RESOURCE_KEYS}


def _filled(**over):
    record = {"skills_panel": {"screenshot": "a.png"}, "visible_steps": {"text": "Searched docs"},
              "skills_self_report": {"text": "machine-learning"}, "notes": ""}
    record.update(over)
    return record


def test_protocol_gaps():
    assert inst.protocol_gaps(_filled()) == []
    assert inst.protocol_gaps(_filled(visible_steps={"text": ""})) == ["visible_steps.text is empty"]
    no_note = _filled(skills_self_report={"text": "not available"})
    assert inst.protocol_gaps(no_note) == ["skills_self_report.text is 'not available' but neither it nor notes say why"]
    assert inst.protocol_gaps({**no_note, "notes": "chat closed before the follow-up"}) == []
    inline = _filled(visible_steps={"text": "not available: chat steps were not captured"})
    assert inst.protocol_gaps(inline) == []
    fresh = inst.build_record(run_id="baseline", task_id="teddy-002", state="off", user="u",
                              codes={}, environment={}, skill_commit="abc")
    assert len(inst.protocol_gaps(fresh)) == 3


def test_prompt_used_and_skill_on_suffix():
    record = inst.build_record(run_id="baseline", task_id="teddy-001", state="off", user="u",
                               codes={}, environment={}, skill_commit="abc")
    assert record["prompt_used"] is None
    assert inst.expected_resources("with_skill", "teddy-001")["serving_endpoint"] == "owm_teddy_ep_001_skill_on"
    assert inst.expected_resources("with_skill", "teddy-002")["serving_endpoint"] == "owm_teddy_ep_002_with_skill"


@NEEDS_ROUND_1
def test_live_round_records_deviation_is_recorded():
    records = inst.load_records(inst.default_records())
    first_001 = next(r for r in records if (r["run_id"], r["task_id"]) == ("baseline", "teddy-001"))
    assert first_001["resources_created"]["serving_endpoint"] == "owm_teddy_ep_001_with_skill"
    assert "owm_teddy_ep_001_with_skill" in first_001["prompt_used"] and "{run}" not in first_001["prompt_used"]
    assert "Deviation" in first_001["notes"]


@NEEDS_ROUND_1
def test_live_round_passes_the_scoring_preflight():
    """Every counted 2026-10-27 record has its protocol fields and an outcome; the stray
    setup record is discarded; the two baseline/teddy-001 attempts map to their own exports."""
    records = inst.load_records(inst.default_records())
    stray = next(r for r in records if r["run_id"] == "old_prompt_baseline")
    assert stray["discarded"] and stray["discard_reason"]
    folder = inst.default_records().parent
    for run in ("baseline", "with_skill"):
        for task in ("teddy-001", "teddy-002"):
            samples = inst.run_samples(records, run, task)
            assert samples, (run, task)
            for record in samples:
                assert inst.protocol_gaps(record) == [], (run, task)
                assert record["outcome"], (run, task)
                assert (folder / inst.export_file(record)).is_file(), inst.export_file(record)
    attempts = inst.run_samples(records, "baseline", "teddy-001")
    assert [inst.export_file(r) for r in attempts] == ["baseline/teddy-001_attempt1.py", "baseline/teddy-001_attempt2.py"]
    assert inst.export_file({"run_id": "with_skill", "task_id": "teddy-002"}) == "with_skill/teddy-002.py"


def test_export_file_prefers_the_recorded_name(tmp_path):
    """Round 2 records name their export; a record with "attempt" but no "export_file" keeps the
    round-1 naming. A small fake round folder proves both are found."""
    import json
    records = [
        {"run_id": "baseline", "task_id": "teddy-002", "attempt": 1, "export_file": "baseline/teddy-002.py"},
        {"run_id": "baseline", "task_id": "teddy-001", "attempt": 2},
    ]
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline" / "teddy-002.py").write_text("# Databricks notebook source\n")
    (tmp_path / "baseline" / "teddy-001_attempt2.py").write_text("# Databricks notebook source\n")
    (tmp_path / "run_records.json").write_text(json.dumps(records))
    for record in inst.load_records(tmp_path / "run_records.json"):
        assert (tmp_path / inst.export_file(record)).is_file(), inst.export_file(record)
    assert [inst.export_file(r) for r in records] == ["baseline/teddy-002.py", "baseline/teddy-001_attempt2.py"]
