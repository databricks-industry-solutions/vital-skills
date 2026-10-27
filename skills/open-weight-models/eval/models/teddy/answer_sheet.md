# Answer sheet — fixed replies for every run

Give one of these answers **only if Genie Code asks**. Never volunteer them, never add
anything else, and give the same answer in every run (`baseline`, `with_skill`,
`old_prompt_baseline`, `old_prompt_with_skill`). These are preferences and settings, not how-to guidance: if a
question is about *how* to do something ("which transformers version?", "how do I
handle the tokenizer?"), reply **"Use your judgement."**

Every prompt (both pairs) now names all of the run's resources — schema
`skills_owm_202610`, model, endpoint, Vector Search endpoint, table, index and download
folder (`models/teddy/family.json` → `names`). If Genie Code asks about any of them, reply
**"Use the names in the prompt."** Fill `<catalog>` the same way in every run.

| If Genie Code asks about | Reply |
|---|---|
| Catalog / schema / volume / download folder / model / endpoint / table / index names | "Use the names in the prompt." |
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

The replication prompts also say how to handle existing endpoints and model versions
(recorded wording); if asked about that, answer "Use the instructions in the prompt."
