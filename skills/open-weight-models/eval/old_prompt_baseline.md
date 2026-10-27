# Old baseline prompt (run `old_prompt_baseline`)

The 2026-10 baseline prompts, copied verbatim from the prompt cell of the recorded baseline
notebooks (the text that was actually pasted). With `old_prompt_with_skill.md` this is the
replication pair: it reproduces the 2026-10 comparison. Paste the block for the task into a
fresh Genie Code chat, with the skill **off**.

> **Warning — these prompts replace the 2026-10 resources.** They name the same volume, Unity Catalog model and endpoint as the 2026-10 runs and tell Genie Code to delete any pre-existing endpoint and model versions on the first run. Running them in the 2026-10 workspace deletes or overwrites those 2026-10 resources. Use a different catalog, or accept that, before you paste.

## Differences from 2026-10

Only these lines were changed (line numbers count lines inside each prompt block below). Everything else — volume, model and endpoint names, the re-run instructions, the `gt_` instruction (there are no `gt_` files any more) — is the recorded text.

| Task | Line | 2026-10 | Now |
|---|---|---|---|
| teddy-001 | 7 | `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/results/baseline/oss-001.txt` | `/Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/old_prompt_baseline/teddy-001.txt` |
| teddy-002 | 9 | `/Workspace/Users/<workspace-user>/PROJECTS/hls-skills-tests/<project-folder>/results/baseline/oss-002.txt` | `/Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/old_prompt_baseline/teddy-002.txt` |


`<workspace-user>` and `<catalog>` are placeholders: replace them with the real values before pasting, the same way in every run. Whatever Genie Code exports, also export the notebook yourself to `eval/outputs/old_prompt_baseline/<task>.py` (README step 6).

## teddy-001 (formerly oss-001)

Source: prompt cell of `previous_runs_2026-10/results/baseline/oss-001.txt`.

```text
Deploy TEDDY-70M as a Model Serving endpoint on Databricks. The model is at https://huggingface.co/Merck/TEDDY (Apache-2.0, paper: arxiv 2503.03485). Use catalog `<catalog>`, schema `skills`. Store downloaded artifacts under `/Volumes/<catalog>/skills/test_without/models/teddy`. Register the model as `<catalog>.skills.teddy_70m_baseline` and name the endpoint `teddy_70m_baseline`. Make the notebook fully re-runnable: on first run, delete any pre-existing endpoint and UC model versions; on subsequent runs, skip cleanup if the endpoint is already in READY state (only tear down if it's in a FAILED state or doesn't exist). If a deployment fails, diagnose the error from endpoint events/logs, fix the root cause, and retry — don't leave the notebook in a broken state. Write the code and explain your decisions.

**Important**: Do NOT read, search, or reference any notebook or file whose name starts with `gt_` in this project — those are ground-truth references and reading them would contaminate the evaluation.

At the end, list every skill from the Skill Registry that you loaded or consulted while developing this notebook.

**Final step**: Export this notebook's source (Databricks `.py` format) to `/Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/old_prompt_baseline/teddy-001.txt` using the Workspace REST API (`GET /api/2.0/workspace/export` with `format=SOURCE`, then base64-decode and write to the output path).
```

## teddy-002 (formerly oss-002)

Source: prompt cell of `previous_runs_2026-10/results/baseline/oss-002.txt`.

```text
Deploy TEDDY-70M as a Model Serving endpoint and enable nearest-neighbor cell-type search on Databricks. The model is at https://huggingface.co/Merck/TEDDY (Apache-2.0, paper: arxiv 2503.03485). Use catalog `<catalog>`, schema `skills`. Store downloaded artifacts under `/Volumes/<catalog>/skills/test_without/models/teddy`. Register the model as `<catalog>.skills.teddy_70m_vs_baseline` and name the endpoint `teddy-70m-vs-baseline`. Create a Delta table of reference cell embeddings and a Vector Search index for nearest-neighbor lookup. Make the notebook fully re-runnable: on first run, delete any pre-existing endpoint and UC model versions; on subsequent runs, skip cleanup if the endpoint is already in READY state (only tear down if it's in a FAILED state or doesn't exist). If a deployment fails, diagnose the error from endpoint events/logs, fix the root cause, and retry — don't leave the notebook in a broken state. Write the code and explain your decisions.

**Important**: Do NOT read, search, or reference any notebook or file whose name starts with `gt_` in this project — those are ground-truth references and reading them would contaminate the evaluation.

At the end, list every skill from the Skill Registry that you loaded or consulted while developing this notebook.

**Development log (required)**: Before the export step, add a markdown cell titled "Development Log" that documents every code fix, debug iteration, and reversal you made during development. For each fix, record: (1) the cell number, (2) the error message, (3) the root cause, and (4) how many attempts it took to resolve. Include a summary table with: total unique bugs encountered, total fix iterations, number of cells requiring fixes, model versions created, and any known limitations (e.g. synthetic vs real data). This is eval metadata for comparing the baseline and skill arms — do not skip it.

**Final step**: Export this notebook's source (Databricks `.py` format) to `/Workspace/Users/<workspace-user>/PROJECTS/genie-code-skills/teddy-eval-2026-10/old_prompt_baseline/teddy-002.txt` using the Workspace REST API (`GET /api/2.0/workspace/export` with `format=SOURCE`, then base64-decode and write to the output path).
```
