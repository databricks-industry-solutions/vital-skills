---
name: html-report
description: >-
  Write a scientific report of a notebook's analysis (summary, methods, key results, figures, checks,
  code and data availability) as one self-contained HTML file: figures embedded as base64 PNGs, nothing
  loaded from the network, and a Print / Save as PDF button with print styles for clean pages. No LaTeX
  or install needed. Use when the user asks to write up results, make a report, a shareable or printable
  summary, or a PDF without LaTeX, from a Databricks notebook. For a typeset LaTeX PDF use report-builder.
category: Others
summary: Turn a notebook's scientific work into a self-contained, printable HTML report (no LaTeX).
version: 0.1.0
author: May Merkle Tan
license: Databricks License
---

# HTML report: notebook results as one shareable, printable file

## Overview

Turns the work in a notebook into a short scientific report that opens in any browser, can be emailed
or stored on a volume as one file, and prints to PDF from the browser. It follows the same content
ideas as report-builder (methods detailed enough to reproduce, results limited to findings, numbers
only from the notebook) but needs no LaTeX engine and no downloads.

It runs on the **execution model**: build a spec from the notebook and call
`scripts/html_report.py` (`build_report`). Do not hand-write the report HTML; the script escapes all
text, embeds figures, adds the print button and styles, and checks the result.

## When to Use

- "Write up these results" / "make a report of this notebook" / "summarise this run as a report".
- "I need something I can share or print" / "give me a PDF" when LaTeX isn't wanted or available.
- A scientific analysis (single-cell, genomics, imaging, ML evaluation, model deployment) that ended
  with figures, tables and checks.

Not for: a typeset journal-style PDF (report-builder) or dashboards.

## Prerequisites

- A notebook with the results: figures (matplotlib Figures or saved PNGs), tables (pandas DataFrames)
  and check values.
- Python with `pandas` (for tables) and `matplotlib` (for Figure objects). The script uses only these,
  the standard library and, for the notebook preview, IPython; it installs and downloads nothing.
- Write access to the output folder (a Volume or workspace path).

## Workflow

1. **Read the notebook.** Cells, outputs, saved figures and tables. Note every number you will use and
   the cell it came from. Never estimate, round differently or invent a number; if a value is not in
   the notebook, compute it in a new cell or leave it out.
2. **Ask only what you can't find:** the output folder (default: the run's volume folder) and the
   author line. Use the notebook's own title and context for everything else.
3. **Get software versions by running code** on the notebook compute (`importlib.metadata.version`)
   for the packages the notebook imports, plus Python and the Databricks runtime or environment.
4. **Collect figures as PNGs:** pass matplotlib Figure objects, or PNG files already saved by the
   notebook. Re-render only if no saved figure exists. One idea per figure, with a full-sentence
   caption.
5. **Write the spec** following `references/report-content.md` (section order, what goes where, the
   availability rules).
6. **Build:** `out = html_report.build_report(spec, "<folder>/<name>.html")`. It refuses to replace an
   existing file: pick a new name, or ask the user before passing `overwrite=True`.
7. **Fix every item in `out["problems"]`** (external links, TODO placeholders, missing sections, size)
   and rebuild. If a TODO must stay because the information is truly unknown, keep it and list it for
   the user.
8. **Preview** with `html_report.display_report(out["path"])` and check it reads well.
9. **Reply** with the path, size, figure and table counts, any remaining TODOs, and the PDF options:
   - "Open the file in a browser and click **Print / Save as PDF** (or Ctrl/Cmd+P, then Save as PDF)."
   - Printing from the notebook preview prints the notebook, not the report: open the saved file.
   - "For a typeset LaTeX PDF instead, I can use report-builder." (only if it is installed)

## Quick Start

```python
import sys; sys.path.insert(0, "/Workspace/Users/<me>/.assistant/skills/html-report/scripts")
import html_report as hr

spec = {
    "title": "<Descriptive title of the finding or run>",
    "authors": "<Name>",
    "summary": "<150 words or fewer: what was done, how, the main result.>",
    "sections": [
        {"title": "Introduction", "blocks": [{"p": "..."}]},
        {"title": "Results", "blocks": [
            {"subheading": "<Finding 1>"}, {"p": "..."},
            {"figure": fig1, "caption": "<What it shows, axes, key takeaway.>"},
            {"table": df_summary, "caption": "..."}]},
        {"title": "Discussion", "blocks": [{"bullets": ["..."]}, {"subheading": "Limitations"},
                                           {"bullets": ["..."]}]},
        {"title": "Methods", "blocks": [
            {"subheading": "Data sources"}, {"p": "..."}, {"path": "/Volumes/..."},
            {"subheading": "Checks"}, {"checks": [
                {"check": "Fidelity", "threshold": "cosine >= 0.999", "result": "min 0.999999", "passed": True}]},
            {"subheading": "Software"}, {"table": versions_df}]},
        {"title": "Code availability", "blocks": [{"p": "..."}]},
        {"title": "Data availability", "blocks": [{"p": "..."}]},
        {"title": "References", "blocks": [{"bullets": ["..."]}]},
    ],
}
out = hr.build_report(spec, "/Volumes/<catalog>/<schema>/<volume>/<folder>/report.html")
print(out)          # {path, bytes, sections, figures, tables, problems}
hr.display_report(out["path"])
```

Block types: `p`, `subheading`, `bullets`, `figure` (+ `caption`, optional `width`, `alt`), `table`
(DataFrame or list of dicts, + `caption`, `max_rows`), `checks` (list of `{check, threshold, result,
passed}`; omit `passed` and set `status` for reported-only or pending checks), `path`. Inline markup in
any text: `**bold**`, `` `code` ``, `[text](https://...)`. Everything else is escaped.
Optional top-level keys: `date` (default today), `notice` (replaces the review-notice text; the notice
itself stays), `footer` (default "Generated <UTC time>.").

## Key Parameters

| Parameter | Default | Effect |
| --- | --- | --- |
| `dpi` (`build_report`, `render_html`) | 150 | Resolution of embedded Figure objects; lower it (e.g. 120) for many figures |
| `overwrite` (`build_report`) | `False` | An existing file is never replaced unless this is `True` (ask the user first) |
| `print_button` | `True` | Adds the Print / Save as PDF button (hidden when printing) |
| `max_rows` (table block) | 200 | Longer tables raise; summarise them instead |
| `max_mb` (`check_report`, via `build_report`) | 10.0 | Larger files are listed in `problems` |

## Expected Outputs

- One self-contained `.html` file at the path given to `build_report`.
- A dict `{path, bytes, sections, figures, tables, problems}`; `problems` lists anything
  `check_report` found (files loaded by URL, TODO text, missing Methods / Code availability / Data
  availability sections, size).

## Guardrails

- **Numbers only from the notebook.** Every number in the text, tables and captions must trace to a
  notebook output. If two cells disagree, say so instead of picking one.
- **Review notice stays.** The script adds "generated by Genie Code with a researcher; review before
  sharing" at the top. Do not remove it unless the user asks.
- **Self-contained.** No external CSS, fonts, scripts or images. Links to papers and repos are fine.
- **Code and data availability name public sources** (repo and commit, Hugging Face repo and
  revision, DOI or arXiv ID, accession, data release). When code or data is workspace-only, say so in a
  plain sentence. No TODO placeholders unless the information is truly unknown; list any left.
- **Size:** keep the file under about 10 MB (`dpi=150` default; use 120 for many figures).
- **Sensitive data:** report aggregates, not row-level patient or participant data. If the source
  holds PHI, ask before including any individual-level values.
- **Don't overwrite** an existing report without asking.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `problems` lists "loads a file by URL" | A figure was given as a URL; pass a Figure or a local PNG path instead |
| File too large | Lower `dpi`, reduce figure size, or split very large tables (`max_rows`) |
| "figure data is not a PNG" | Save the figure as PNG (`fig.savefig(path, format="png")`) or pass the Figure |
| Print splits a figure | Make the figure shorter; print styles keep a figure on one page only if it fits |
| Button prints the notebook | The report was printed from the notebook preview; open the saved file |

## References

- `references/report-content.md`: section order, what goes in each section, and an example of
  sufficient Methods.
- `scripts/html_report.py`: `build_report` (render, check and write), `render_html`, `check_report`,
  `display_report`.
- Adapted from report-builder (this repository, `skills/report-builder`, by Peter Hawkins): the same content
  rules (methods detailed enough to reproduce, findings-only results, numbers only from the notebook), as
  HTML instead of a typeset LaTeX PDF.
- Printing to PDF from a browser uses the CSS print rules (`@media print`, `break-inside`):
  https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/At-rules/@media and
  https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/break-inside
