# Answer sheet — fixed replies for every run

Give one of these answers **only if Genie Code asks**. Never volunteer them, never add
anything else, and give the same answer in every run (baseline, with_skill,
with_skill_old_prompt). These are preferences and settings, not how-to guidance: if a
question is about *how* to do something ("which transformers version?", "how do I
handle the tokenizer?"), reply **"Use your judgement."**

Fill the placeholders once, before the first run, and keep them identical across runs:
`<catalog>` (the catalog named in the query), `<run-id>` (`baseline`, `with_skill` or
`with_skill_old_prompt`, so runs do not collide).

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
| Anything else | "Use your judgement." |

For the `with_skill_old_prompt` run the prompt itself already names the volume, model and
endpoint; if Genie Code asks about those, answer "Use the names in the prompt."
