"""eval/CHECKS.md is generated from the scorers' own sources; it must match them."""

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

import list_checks  # noqa: E402


def test_checks_md_is_current():
    assert (EVAL_DIR / "CHECKS.md").read_text(encoding="utf-8") == list_checks.render(), (
        "CHECKS.md is stale: run python3 skills/open-weight-models/eval/list_checks.py --write")


def test_every_corrected_check_is_listed():
    text = list_checks.render()
    for key in list_checks.corrected_rules():
        assert f"`{key}`" in text, key
