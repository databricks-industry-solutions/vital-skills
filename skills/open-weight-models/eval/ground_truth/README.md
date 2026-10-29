# Ground-truth notebooks

Hand-built notebooks that deploy each model end to end on Databricks. The model references in
`../../references/models/` were written from them (for example `teddy.md` names `gt_teddy`, and
`geneformer.md` names `gt_geneformer` and `gt_geneformer_bionemo`). They are the starting point for a
model family's evaluation: its tasks and checks are drawn from what these notebooks do. Like the rest
of `eval/`, they are not published with the skill, and an evaluation run must not be able to read
them (the run protocol's leak check looks for `gt_` paths).

| Notebook | Model | Status |
|---|---|---|
| `gt_teddy.py` | TEDDY-70M | working; the TEDDY evaluation (`../models/teddy/`) is drawn from it |
| `gt_geneformer.py` | Geneformer V1-10M (Hugging Face path) | working; evaluation pending |
| `gt_geneformer_bionemo.py` | Geneformer V2 (NVIDIA BioNeMo path) | working; evaluation pending |
| `gt_scgpt_todo.py`, `gt_scimilarity_todo.py` | scGPT, SCimilarity | to-do stubs |

Moved here from `tests/_ground_truth/` (on `main` before the eval reorganisation) by way of
`runs_20260924/_ground_truth/`, so all model families share one copy.
