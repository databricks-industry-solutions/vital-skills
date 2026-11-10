# Changelog — html-report skill

All notable changes to the `html-report` skill are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/). The skill's
frontmatter `version` is the source of truth for releases.

## [Unreleased]

### Added
- 2026-11-09: first version (0.1.0). `scripts/html_report.py` turns a spec built from a notebook (summary,
  sections of text, figures, tables, checks, paths) into one self-contained HTML file: text escaped,
  figures embedded as base64 PNGs, nothing loaded from the network, a Print / Save as PDF button and
  print styles. `build_report` refuses to replace an existing file and returns the problems
  `check_report` finds (files loaded by URL, TODO text, missing Methods / Code availability / Data
  availability, size). Optional top-level keys `date`, `notice`, `footer`. `references/report-content.md`
  gives the section order and an example of sufficient Methods. Unit tests in `tests/`.

### Attribution
- Adapted from the `report-builder` skill (Peter Hawkins).
