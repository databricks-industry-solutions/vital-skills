"""Unit tests for scripts/html_report.py (no network, no notebook)."""
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import html_report as hr  # noqa: E402


def _spec(**extra):
    spec = {
        "title": "A finding",
        "authors": ["A. Researcher", "B. Researcher"],
        "summary": "What was done and the main result.",
        "sections": [
            {"title": "Results", "blocks": [{"p": "Plain text."}]},
            {"title": "Methods", "blocks": [{"p": "How it was done."}]},
            {"title": "Code availability", "blocks": [{"p": "Not publicly released."}]},
            {"title": "Data availability", "blocks": [{"p": "CELLxGENE Census release 2025-11-08."}]},
        ],
    }
    spec.update(extra)
    return spec


def test_complete_report_has_no_problems(tmp_path):
    out = hr.build_report(_spec(), tmp_path / "r.html")
    page = Path(out["path"]).read_text(encoding="utf-8")
    assert out["problems"] == []
    assert out["sections"] == 4 and out["bytes"] == len(page.encode("utf-8"))
    assert "window.print()" in page and 'class="notice"' in page
    assert "A. Researcher, B. Researcher" in page


def test_missing_sections_are_reported(tmp_path):
    spec = _spec(sections=[{"title": "Results", "blocks": [{"p": "TODO"}]}])
    problems = hr.build_report(spec, tmp_path / "r.html")["problems"]
    assert "section 'Methods' missing" in problems
    assert "section 'Data availability' missing" in problems
    assert any("TODO" in p for p in problems)


def test_text_is_escaped_and_only_http_links_are_made():
    spec = _spec(sections=[{"title": "Results", "blocks": [
        {"p": "<script>alert(1)</script> **bold** `x<y` [paper](https://arxiv.org/abs/2503.03485) "
              "[bad](javascript:alert(1))"}]}])
    page = hr.render_html(spec)
    assert "<script>alert" not in page and "&lt;script&gt;" in page
    assert "<strong>bold</strong>" in page and "<code>x&lt;y</code>" in page
    assert '<a href="https://arxiv.org/abs/2503.03485">paper</a>' in page
    assert 'href="javascript' not in page


def test_figures_are_embedded_and_need_a_caption(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    png = tmp_path / "f.png"
    fig.savefig(png, format="png")
    blocks = [{"figure": fig, "caption": "A line."}, {"figure": str(png), "caption": "The same line."}]
    out = hr.build_report(_spec(sections=_spec()["sections"] + [{"title": "Figures", "blocks": blocks}]),
                          tmp_path / "r.html")
    page = Path(out["path"]).read_text(encoding="utf-8")
    assert out["figures"] == 2 and page.count('src="data:image/png;base64,') == 2
    assert "Figure 2." in page and out["problems"] == []
    with pytest.raises(ValueError, match="caption"):
        hr.render_html(_spec(sections=[{"title": "F", "blocks": [{"figure": fig}]}]))
    with pytest.raises(ValueError, match="not a PNG"):
        hr.render_html(_spec(sections=[{"title": "F", "blocks": [{"figure": b"GIF89a", "caption": "x"}]}]))
    with pytest.raises(ValueError, match="existing PNG path"):
        hr.render_html(_spec(sections=[{"title": "F", "blocks": [{"figure": "https://x/y.png", "caption": "x"}]}]))
    plt.close(fig)


def test_tables_and_checks():
    df = pd.DataFrame({"type": ["<b>", "astrocyte"], "cells": [10, 12]})
    checks = [{"check": "Fidelity", "threshold": "cosine >= 0.999", "result": "min 0.999999", "passed": True},
              {"check": "Recall", "threshold": ">= 0.95", "result": "0.90", "passed": False},
              {"check": "Accuracy", "result": "0.75", "status": "reported"}]
    page = hr.render_html(_spec(sections=[{"title": "Results", "blocks": [
        {"table": df, "caption": "Cells per type."}, {"checks": checks}]}]))
    assert "&lt;b&gt;" in page and "<caption><strong>Table 1.</strong> Cells per type.</caption>" in page
    assert '<td class="pass">PASS</td>' in page and '<td class="fail">FAIL</td>' in page
    assert '<td class="">reported</td>' in page
    with pytest.raises(ValueError, match="max 200"):
        hr.render_html(_spec(sections=[{"title": "T", "blocks": [{"table": pd.DataFrame({"a": range(201)})}]}]))


def test_invalid_specs_raise():
    with pytest.raises(ValueError, match="title"):
        hr.render_html({"sections": [{"title": "x"}]})
    with pytest.raises(ValueError, match="at least one section"):
        hr.render_html({"title": "t"})
    with pytest.raises(ValueError, match="exactly one"):
        hr.render_html(_spec(sections=[{"title": "x", "blocks": [{"p": "a", "bullets": ["b"]}]}]))


def test_existing_file_is_not_replaced_without_overwrite(tmp_path):
    path = tmp_path / "r.html"
    hr.build_report(_spec(), path)
    with pytest.raises(FileExistsError):
        hr.build_report(_spec(), path)
    assert hr.build_report(_spec(title="New"), path, overwrite=True)["problems"] == []
    assert "<h1>New</h1>" in path.read_text(encoding="utf-8")


def test_optional_date_notice_and_footer():
    page = hr.render_html(_spec(date="2026-11-09", notice="Draft for the lab.", footer="Made by hand."))
    assert "2026-11-09" in page and "Draft for the lab." in page and hr.DEFAULT_NOTICE not in page
    assert '<footer class="report">Made by hand.</footer>' in page
    assert hr.DEFAULT_NOTICE in hr.render_html(_spec())


def test_check_report_flags_external_files_and_size():
    page = hr.render_html(_spec())
    assert hr.check_report(page) == []
    bad = page.replace("</main>", '<img src="https://example.org/x.png"></main>')
    assert any("loads a file by URL" in p for p in hr.check_report(bad))
    assert any("MB" in p for p in hr.check_report(page, max_mb=0.0001))
    assert "no Print / Save as PDF button" in hr.check_report(hr.render_html(_spec(), print_button=False))
