# Evaluation Report — open-weight-models Skill

**What was tested.** Genie Code deploying TEDDY-70M, an open-weight single-cell model, on
Databricks — once with this skill and once without it (built-in skills on in both), with the same
plain request. Two tasks: **teddy-001** deploys TEDDY as a serving endpoint that turns a cell's
gene counts into an embedding; **teddy-002** also builds a nearest-neighbour search that labels a
cell by the types of its closest reference cells. Latest round: 2026-10-27/28, one run per task
with and one without the skill, serverless GPU notebooks. Geneformer: evaluation pending.

**In one line:** both ways, Genie Code built endpoints that work. With the skill they were also
logged, traceable to their source, and safe to re-run; without it, they were not.

## Results — latest round

| Task | Without the skill | With the skill | Result |
|---|---|---|---|
| teddy-001: deploy TEDDY | endpoint works; **7 checks failed** (4 major, 3 minor) | endpoint works; all 36 checks passed | skill wins |
| teddy-002: deploy + cell search | endpoint and search work; **7 checks failed** (4 major, 3 minor) | endpoint and search work; all 44 checks passed | skill wins |

No check passed without the skill and failed with it. What every check is, with its severity:
[`CHECKS.md`](CHECKS.md).

## What the skill adds

The 7 checks the runs without the skill failed — the same 7 on both tasks:

| Severity | What was missing without the skill | Why it matters |
|---|---|---|
| 🟠 major | request logging on the endpoint (AI Gateway) | requests are not logged, so the endpoint is not governed |
| 🟠 major | an inference table | no record of what was sent and returned, nothing to audit |
| 🟠 major | usage tracking | no record of who used the endpoint and how much |
| 🟠 major | source, revision and licence tags on the model | which upstream version and licence it came from is not recorded |
| 🟡 minor | a deploy switch saved off (`run_go`) | re-running the saved notebook can create billable resources again |
| 🟡 minor | serving pins from the model's own dependency file | pins record what was installed that day, not what the model needs |
| 🟡 minor | clean-up of cached code on re-run | re-running cells out of order can serve stale code |

Also with the skill only (not counted as checks): serving on a GPU instead of CPU, and the shared
single-cell request format used across this skill's models.

🔴 critical = does not work or gives wrong results · 🟠 major = works, but unsafe, ungoverned or
not reproducible · 🟡 minor = robustness or style.

## What the evaluation fixed in the skill

Problems the runs found in the skill itself, all fixed before the latest round, which confirmed
the fixes ([`../CHANGELOG.md`](../CHANGELOG.md)):

- 🔴 Endpoint creation needs `name=` in newer SDKs — now in every example.
- 🔴 Genes TEDDY does not know were kept as "unknown"; TEDDY drops them — now dropped.
- 🟠 Settings sent with the SDK's `extra_params` were silently ignored — now sent in the request
  body, with a test that a changed setting changes the output.
- 🟠 The wrapper was only checked against itself — now checked against TEDDY's own code.
- 🟠 An index could be deleted and rebuilt on a mismatch — now nothing is replaced without asking.
- 🟡 The upstream licence is ambiguous (Apache-2.0 vs MIT) — both are recorded, the user confirms.

## Verdict

**Ship, with follow-ups.** The skill won both tasks and made nothing worse, which meets the repo's
ship rule (`skill-eval`: at least one win, no regressions).

## Limitations and next steps

- **One run per task, by hand.** Read the result as direction, not statistics. Manual runs were
  slow and error-prone (memories to clear, compute to select, chat tabs, prompts); several had to
  be redone. Next: run the tasks with the Genie Code CLI harness in `benchmarks/biomnibench/`,
  which sets all of that up by script and allows repeats.
- **The checks come from the skill's own references,** so they can favour it. They were corrected
  where they failed valid code; see "How the checks changed" in [`ROUNDS`](models/teddy/ROUNDS.md).
- **The request "Deploy … Write the code." is ambiguous.** Next round: reword it.
- **Geneformer:** evaluation pending.

## More detail

- **All rounds** (Sep–Oct 2026, round 1, round 2), results per round, and where the skill helps by
  category: [`models/teddy/ROUNDS.md`](models/teddy/ROUNDS.md).
- **Each round's notes and counted scores:** folded at the end of `ROUNDS.md` ("Round details"), and
  [`models/teddy/scores.json`](models/teddy/scores.json) (every round in one file).
- **Every check:** [`CHECKS.md`](CHECKS.md). **Protocol and scorers:** [`README.md`](README.md).

## Reproduce

```bash
uv run --isolated --with pytest --with-requirements skills/open-weight-models/tests/requirements.txt \
    python -m pytest -q skills/open-weight-models/tests
```

Score summary (latest round, counted checks; read by `skills/skill-eval/scripts/compare_runs.py`):
[`baseline_scores.json`](baseline_scores.json), [`with_skill_scores.json`](with_skill_scores.json):

```bash
python3 skills/skill-eval/scripts/compare_runs.py skills/open-weight-models/eval/baseline_scores.json \
    skills/open-weight-models/eval/with_skill_scores.json --difficulty skills/open-weight-models/eval/expectations.json
```
