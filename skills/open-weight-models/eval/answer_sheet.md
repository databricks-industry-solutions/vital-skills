# Answer sheet — fixed replies for every run

Give one of these answers **only if Genie Code asks**. Never volunteer them, never add
anything else, and give the same answer in every run (`baseline`, `with_skill`,
`old_prompt_baseline`, `old_prompt_with_skill`). These are preferences and settings, not how-to guidance: if a
question is about *how* to do something ("which transformers version?", "how do I
handle the tokenizer?"), reply **"Use your judgement."**

Fill the placeholders once, before the first run, and keep them identical across runs:
`<catalog>` (the catalog named in the query), `<run-id>` (`baseline` or `with_skill`, so
the two standard runs do not collide).

| If Genie Code asks about | Reply |
|---|---|
| Catalog / schema | "Catalog `<catalog>`, schema `skills`." (as in the query) |
| Volume / where to store downloaded files | "Use the volume `/Volumes/<catalog>/skills/eval_<run-id>/`; create it if it does not exist." |
| Model name in Unity Catalog | "`<catalog>.skills.teddy_70m_<run-id>`" |
| Endpoint name | "`teddy-70m-<run-id>`" |
| Table / index names (teddy-002) | "Prefix them with `teddy_70m_<run-id>_`." |
| Model variant | "70M." (as in the query) |
| GPU / compute tier for serving | "Use whatever GPU tier you think is right for the 70M model." |
| Notebook compute | "Use the compute you think is right." |
| Inference tables / usage tracking / AI Gateway | "Yes." |
| Existing endpoints, models, tables or indexes to reuse | "There are no existing resources to reuse; create new ones." |
| Whether to delete / tear down at the end | "Your choice; I will clean up anything left over." |
| Reference cell data for search (teddy-002) | "Use any public reference data you think is suitable." |
| Permission to run cells / install packages | "Yes." |
| One notebook or several | "One notebook." |
| Whether it may look at earlier notebooks, results or eval files | "No — start from scratch." |
| Anything else | "Use your judgement." |

**Skill self-report (not an answer to a question).** In the standard pair only, after
the task is complete and the notebook exported, send this as a separate message,
identical in both runs: "List every skill from the Skill Registry that you loaded or
consulted while developing this notebook." The replication prompts already end with this
request, so do not send it there. Record the reply as Genie Code's self-report (README
step 7); it is not verified.

For the replication pair (`old_prompt_baseline`, `old_prompt_with_skill`) the prompts
already name the volume, model and endpoint and say how to handle existing ones; if Genie
Code asks about any of those, answer "Use the names and instructions in the prompt." —
not the rows above. All other rows apply unchanged.
