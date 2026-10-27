# Old forced-read with-skill prompt (run `with_skill_old_prompt`)

The 2026-10 with-skill prompts, copied verbatim from the prompt cell of the recorded
with-skill notebooks. Paste the block for the task into the fresh Genie Code chat
instead of the `evalset.json` query. This run exists only to replicate the 2026-10
results; the plain `with_skill` run is the one that measures whether Genie Code picks
the skill up on its own.

## What changed from the recorded prompt

Only the two skill file paths, which pointed at the 2026-10 project copy of the skill
(then still named `oss-models`):

```diff
- /Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/.assistant/skills/oss-models/SKILL.md
- /Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/.assistant/skills/oss-models/references/models/teddy.md
+ /Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/SKILL.md
+ /Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/references/models/teddy.md
```

Everything else is unchanged, including the parts that no longer match the current
protocol — keep them, they are what was measured:

- the volume, model and endpoint names given in the prompt (teddy-002 uses the names of the 2026-10 "skill v2" run, `teddy_70m_v2`);
- the `gt_` instruction (there are no `gt_` files any more);
- the **Final step** export path under `PROJECTS/hls-skills-tests/<project-folder>/results/...`. Genie Code may try to write there; whatever it does, export the notebook yourself to `eval/outputs/with_skill_old_prompt/<task>.py` (README step 6).

`<workspace-user>` and `<catalog>` are placeholders: replace them with the real values before pasting, the same way in every run.

Source cells, for checking: the prompt cell of `previous_runs_2026-10/results/with_skill/oss-001.txt` (teddy-001) and of `previous_runs_2026-10/results/with_skill/oss-002.txt` (teddy-002, the run whose scores are recorded). `previous_runs_2026-10/02_eval_prompt_templates.py` holds a later edit of the same prompts that adds "Runtime capture" (and, for oss-001, "Development log") paragraphs; those were not in the recorded runs, so they are not used here.

## teddy-001 (formerly oss-001)

```text
Deploy TEDDY-70M as a Model Serving endpoint on Databricks. The model is at https://huggingface.co/Merck/TEDDY (Apache-2.0, paper: arxiv 2503.03485). Use catalog `<catalog>`, schema `skills`. Store downloaded artifacts under `/Volumes/<catalog>/skills/test_with/models/teddy`. Register the model as `<catalog>.skills.teddy_70m` and name the endpoint `teddy_70m`. Make the notebook fully re-runnable: on first run, delete any pre-existing endpoint and UC model versions; on subsequent runs, skip cleanup if the endpoint is already in READY state (only tear down if it's in a FAILED state or doesn't exist). If a deployment fails, diagnose the error from endpoint events/logs, fix the root cause, and retry — don't leave the notebook in a broken state. Write the code and explain your decisions.

**Before writing any code**, read the project's HLS model-deployment skill and TEDDY model reference using `readAssetById` (type `file`) at these workspace paths:
- `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/SKILL.md`
- `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/references/models/teddy.md`

**Treat these skill references as authoritative.** When a skill specifies a concrete implementation pattern (e.g., a particular MLflow logging method, a dependency version, a wrapper structure), use that exact pattern — do not substitute your own approach even if it seems simpler or more familiar. Before writing each code cell, re-check the relevant skill section to ensure your implementation matches. If you deviate from a skill recommendation, state why explicitly in a code comment.

**Important**: Do NOT read, search, or reference any notebook or file whose name starts with `gt_` in this project — those are ground-truth references and reading them would contaminate the evaluation.

At the end, list every skill from the Skill Registry that you loaded or consulted while developing this notebook.

**Final step**: Export this notebook's source (Databricks `.py` format) to `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/results/with_skill/oss-001.txt` using the Workspace REST API (`GET /api/2.0/workspace/export` with `format=SOURCE`, then base64-decode and write to the output path).
```

## teddy-002 (formerly oss-002)

```text
Deploy TEDDY-70M as a Model Serving endpoint and enable nearest-neighbor cell-type search on Databricks. The model is at https://huggingface.co/Merck/TEDDY (Apache-2.0, paper: arxiv 2503.03485). Use catalog `<catalog>`, schema `skills`. Store downloaded artifacts under `/Volumes/<catalog>/skills/test_with/models/teddy`. Register the model as `<catalog>.skills.teddy_70m_v2` and name the endpoint `teddy-70m-v2-embedder`. Create a Delta table `<catalog>.skills.teddy_cells_70m_v2` of reference cell embeddings and a Vector Search index `<catalog>.skills.teddy_cell_index_70m_v2` on endpoint `teddy-70m-v2-search-endpoint` for nearest-neighbor lookup. Make the notebook fully re-runnable: on first run, delete any pre-existing endpoint and UC model versions; on subsequent runs, skip cleanup if the endpoint is already in READY state (only tear down if it's in a FAILED state or doesn't exist). If a deployment fails, diagnose the error from endpoint events/logs, fix the root cause, and retry — don't leave the notebook in a broken state. Write the code and explain your decisions.

**Before writing any code**, read the project's HLS model-deployment skill and TEDDY model reference using `readAssetById` (type `file`) at these workspace paths:
- `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/SKILL.md`
- `/Workspace/Users/<workspace-user>/.assistant/skills/open-weight-models/references/models/teddy.md`

**Treat these skill references as authoritative.** When a skill specifies a concrete implementation pattern (e.g., a particular MLflow logging method, a dependency version, a wrapper structure, an AI Search index spec), use that exact pattern — do not substitute your own approach even if it seems simpler or more familiar. Before writing each code cell, re-check the relevant skill section to ensure your implementation matches. If you deviate from a skill recommendation, state why explicitly in a code comment.

**Important**: Do NOT read, search, or reference any notebook or file whose name starts with `gt_` in this project — those are ground-truth references and reading them would contaminate the evaluation.

At the end, list every skill from the Skill Registry that you loaded or consulted while developing this notebook.

**Development log (required)**: Before the export step, add a markdown cell titled "Development Log" that documents every code fix, debug iteration, and reversal you made during development. For each fix, record: (1) the cell number, (2) the error message, (3) the root cause, and (4) how many attempts it took to resolve. Include a summary table with: total unique bugs encountered, total fix iterations, number of cells requiring fixes, model versions created, and any known limitations (e.g. synthetic vs real data). This is eval metadata for comparing the baseline and skill arms — do not skip it.

**Final step**: Export this notebook's source (Databricks `.py` format) to `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/results/with_skill_v2/oss-002.txt` using the Workspace REST API (`GET /api/2.0/workspace/export` with `format=SOURCE`, then base64-decode and write to the output path).
```
