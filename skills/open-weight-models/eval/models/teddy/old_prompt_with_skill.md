# Old forced-read with-skill prompt (run `old_prompt_with_skill`)

The Sep–Oct 2026 with-skill prompts, taken from the prompt cell of the recorded with-skill
notebooks (the text that was actually pasted). They tell Genie Code to read `SKILL.md` and
`teddy.md` first. With `old_prompt_baseline.md` this is the replication pair. Paste the block
for the task into a fresh Genie Code chat, with the skill **on**.

Only the resource names, the skill folder and the export path were changed, so the run
creates its own resources under schema `skills_owm_202610` (names in `family.json`) and
does not touch anything from Sep–Oct 2026. Everything else — the re-run instructions,
the `gt_` instruction (there are no `gt_` files any more), the self-report line — is the
recorded text.

## Differences from Sep–Oct 2026

Line numbers count lines inside each prompt block below. The same substitutions are in
`family.json` (`old_prompt_substitutions`); `tests/test_eval_scorers.py` checks that applying
them to the recorded prompt gives exactly the block below.

| Task | Line | Sep–Oct 2026 | Now |
|---|---|---|---|
| teddy-001 | 1 | `skills` | `skills_owm_202610` |
| teddy-001 | 1 | `/Volumes/<catalog>/skills/test_with/models/teddy` | `/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/001_old_with_skill/` |
| teddy-001 | 1 | `<catalog>.skills.teddy_70m` | `<catalog>.skills_owm_202610.owm_teddy_model_001_old_with_skill` |
| teddy-001 | 1 | `teddy_70m` | `owm_teddy_ep_001_old_with_skill` |
| teddy-001 | 4 | `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/.assistant/skills/oss-models/…` | `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/…` |
| teddy-001 | 5 | `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/.assistant/skills/oss-models/…` | `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/…` |
| teddy-001 | 13 | `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/results/with_skill/oss-001.txt` | `/Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/old_prompt_with_skill/teddy-001.txt` |
| teddy-002 | 1 | `skills` | `skills_owm_202610` |
| teddy-002 | 1 | `/Volumes/<catalog>/skills/test_with/models/teddy` | `/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/002_old_with_skill/` |
| teddy-002 | 1 | `<catalog>.skills.teddy_70m_v2` | `<catalog>.skills_owm_202610.owm_teddy_model_002_old_with_skill` |
| teddy-002 | 1 | `teddy-70m-v2-embedder` | `owm_teddy_ep_002_old_with_skill` |
| teddy-002 | 1 | `<catalog>.skills.teddy_cells_70m_v2` | `<catalog>.skills_owm_202610.owm_teddy_cells_002_old_with_skill` |
| teddy-002 | 1 | `<catalog>.skills.teddy_cell_index_70m_v2` | `<catalog>.skills_owm_202610.owm_teddy_index_002_old_with_skill` |
| teddy-002 | 1 | `teddy-70m-v2-search-endpoint` | `owm_teddy_vs_002_old_with_skill` |
| teddy-002 | 4 | `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/.assistant/skills/oss-models/…` | `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/…` |
| teddy-002 | 5 | `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/.assistant/skills/oss-models/…` | `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/…` |
| teddy-002 | 15 | `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/results/with_skill_v2/oss-002.txt` | `/Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/old_prompt_with_skill/teddy-002.txt` |

The skill paths pointed at the Sep–Oct 2026 project copy of the skill (then still named `oss-models`).
teddy-002 is the prompt of the Sep–Oct 2026 second ("v2") skill run, the one whose scores are recorded.
`runs_20260924/02_eval_prompt_templates.py` holds a later edit of these prompts that adds "Runtime capture" (and, for oss-001, "Development log") paragraphs; those were not in the recorded runs, so they are not used here.

`<workspace-user>` and `<catalog>` are placeholders: replace them with the real values before pasting, the same way in every run. Whatever Genie Code exports, also export the notebook yourself to `models/teddy/runs_20261026/old_prompt_with_skill/<task>.py` (README step 6).

## teddy-001 (formerly oss-001)

Source: prompt cell of `runs_20260924/results/with_skill/oss-001.txt`.

```text
Deploy TEDDY-70M as a Model Serving endpoint on Databricks. The model is at https://huggingface.co/Merck/TEDDY (Apache-2.0, paper: arxiv 2503.03485). Use catalog `<catalog>`, schema `skills_owm_202610`. Store downloaded artifacts under `/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/001_old_with_skill/`. Register the model as `<catalog>.skills_owm_202610.owm_teddy_model_001_old_with_skill` and name the endpoint `owm_teddy_ep_001_old_with_skill`. Make the notebook fully re-runnable: on first run, delete any pre-existing endpoint and UC model versions; on subsequent runs, skip cleanup if the endpoint is already in READY state (only tear down if it's in a FAILED state or doesn't exist). If a deployment fails, diagnose the error from endpoint events/logs, fix the root cause, and retry — don't leave the notebook in a broken state. Write the code and explain your decisions.

**Before writing any code**, read the project's HLS model-deployment skill and TEDDY model reference using `readAssetById` (type `file`) at these workspace paths:
- `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/SKILL.md`
- `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/references/models/teddy.md`

**Treat these skill references as authoritative.** When a skill specifies a concrete implementation pattern (e.g., a particular MLflow logging method, a dependency version, a wrapper structure), use that exact pattern — do not substitute your own approach even if it seems simpler or more familiar. Before writing each code cell, re-check the relevant skill section to ensure your implementation matches. If you deviate from a skill recommendation, state why explicitly in a code comment.

**Important**: Do NOT read, search, or reference any notebook or file whose name starts with `gt_` in this project — those are ground-truth references and reading them would contaminate the evaluation.

At the end, list every skill from the Skill Registry that you loaded or consulted while developing this notebook.

**Final step**: Export this notebook's source (Databricks `.py` format) to `/Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/old_prompt_with_skill/teddy-001.txt` using the Workspace REST API (`GET /api/2.0/workspace/export` with `format=SOURCE`, then base64-decode and write to the output path).
```

## teddy-002 (formerly oss-002)

Source: prompt cell of `runs_20260924/results/with_skill/oss-002.txt`.

```text
Deploy TEDDY-70M as a Model Serving endpoint and enable nearest-neighbor cell-type search on Databricks. The model is at https://huggingface.co/Merck/TEDDY (Apache-2.0, paper: arxiv 2503.03485). Use catalog `<catalog>`, schema `skills_owm_202610`. Store downloaded artifacts under `/Volumes/<catalog>/skills_owm_202610/owm_teddy_files/002_old_with_skill/`. Register the model as `<catalog>.skills_owm_202610.owm_teddy_model_002_old_with_skill` and name the endpoint `owm_teddy_ep_002_old_with_skill`. Create a Delta table `<catalog>.skills_owm_202610.owm_teddy_cells_002_old_with_skill` of reference cell embeddings and a Vector Search index `<catalog>.skills_owm_202610.owm_teddy_index_002_old_with_skill` on endpoint `owm_teddy_vs_002_old_with_skill` for nearest-neighbor lookup. Make the notebook fully re-runnable: on first run, delete any pre-existing endpoint and UC model versions; on subsequent runs, skip cleanup if the endpoint is already in READY state (only tear down if it's in a FAILED state or doesn't exist). If a deployment fails, diagnose the error from endpoint events/logs, fix the root cause, and retry — don't leave the notebook in a broken state. Write the code and explain your decisions.

**Before writing any code**, read the project's HLS model-deployment skill and TEDDY model reference using `readAssetById` (type `file`) at these workspace paths:
- `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/SKILL.md`
- `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/references/models/teddy.md`

**Treat these skill references as authoritative.** When a skill specifies a concrete implementation pattern (e.g., a particular MLflow logging method, a dependency version, a wrapper structure, an AI Search index spec), use that exact pattern — do not substitute your own approach even if it seems simpler or more familiar. Before writing each code cell, re-check the relevant skill section to ensure your implementation matches. If you deviate from a skill recommendation, state why explicitly in a code comment.

**Important**: Do NOT read, search, or reference any notebook or file whose name starts with `gt_` in this project — those are ground-truth references and reading them would contaminate the evaluation.

At the end, list every skill from the Skill Registry that you loaded or consulted while developing this notebook.

**Development log (required)**: Before the export step, add a markdown cell titled "Development Log" that documents every code fix, debug iteration, and reversal you made during development. For each fix, record: (1) the cell number, (2) the error message, (3) the root cause, and (4) how many attempts it took to resolve. Include a summary table with: total unique bugs encountered, total fix iterations, number of cells requiring fixes, model versions created, and any known limitations (e.g. synthetic vs real data). This is eval metadata for comparing the baseline and skill arms — do not skip it.

**Final step**: Export this notebook's source (Databricks `.py` format) to `/Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/old_prompt_with_skill/teddy-002.txt` using the Workspace REST API (`GET /api/2.0/workspace/export` with `format=SOURCE`, then base64-decode and write to the output path).
```
