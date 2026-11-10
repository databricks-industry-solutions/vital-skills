# What goes in the report

Section order (use these titles; `check_report` looks for Methods, Code availability and Data
availability):

1. **Title, authors, date.** Title states the finding or the run ("TEDDY-70M served on Databricks and
   searched against 990 Census brain cells"), not "Report".
2. **Review notice** (added by the script).
3. **Summary**, 150 words or fewer: what was analysed, the approach, the main result with its number.
4. **Introduction**, 1-3 short paragraphs: the question, why it matters, the approach.
5. **Results: findings only.** Each subsection title is a finding ("Neighbours share the query's cell
   type 83% of the time, against 12% by chance"), not a pipeline step ("Index built"). Each finding gets
   2-4 sentences and its figure or table. If nothing notable emerged, one paragraph pointing to the
   figures is enough.
6. **Discussion**: interpretation in bullets, then **Limitations** (data size, label quality, what
   the analysis cannot show).
7. **Methods**, detailed enough to reproduce, in prose (not code):
   - *Data sources*: full paths or table names, sizes, release or version, filters and why.
   - *Preprocessing*: every step and parameter in a sentence, with counts before and after; steps
     considered and skipped, and why.
   - *Analysis or model*: the method by its exact name, every setting that matters (k, pooling,
     resolution, batch size), the compute used.
   - *Checks*: a `checks` block with each threshold and PASS/FAIL; reported-only values marked as such;
     any check that could not run, and why. QC and validation figures go here, not in Results.
   - *Timings and resources* when the run created billable resources: each resource, its state, and
     whether it is still billing.
   - *Software*: a versions table from `importlib.metadata` (never typed from memory).
8. **Code availability**: the notebook path and any wrapper or script paths; public code by repo and
   commit or release. Workspace-only code: say so plainly ("The notebook and wrapper are not publicly
   released.").
9. **Data availability**: public data by release and identifier (e.g. CELLxGENE Census release and
   dataset IDs, GEO/ENA accession, Hugging Face repo and revision); derived tables and files by path.
10. **References**: papers and model cards actually used, with links (DOI, arXiv).

## Writing style

- Plain, factual sentences; passive voice is fine in Methods.
- Numbers with their unit and their comparison ("0.999999 against a 0.999 threshold").
- Captions are full sentences: what is plotted, axes or colours, the takeaway.
- Keep the report short: most notebook runs fit in 2-5 printed pages.

## Example of insufficient vs sufficient Methods

Insufficient: "Cells were sampled from Census, embedded and indexed."

Sufficient: "1,000 human brain cells were sampled from CELLxGENE Census release 2025-11-08 (primary
data, 10x 3' v3), 125 from each of the 8 most common cell types after TEDDY's pretraining QC. Ten cells
(2 from each of 5 types) were held out as queries; the other 990 were embedded with the registered
model on the notebook GPU (mean pooling, 10 cells per request) and written to a Delta table synced to a
Vector Search index."
