"""Offline tests for eval/install_skill_for_run.py. No Databricks CLI calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_DIR / "eval"))

import install_skill_for_run as inst  # noqa: E402


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
                              codes={}, listings={"/Workspace/Users/someone@example.com/.assistant/skills": None},
                              skill_commit="abc", timestamp="t1")
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
        listings={inst.WORKSPACE_SKILLS: [], inst.user_skills_dir("someone@example.com"): [
            {"name": "open-weight-models", "object_type": "DIRECTORY"}]},
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
    args = inst.parse_args(["--run-id", "with_skill", "--task-id", "teddy-001", "--check-notebook", "nb.py"])
    assert args.check_notebook == "nb.py"
