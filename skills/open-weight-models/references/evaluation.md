# Evaluation and stress testing

This reference has two parts: how to validate a deployed model and any chained parts (index, second endpoint), and how to evaluate this skill itself against no skills and the generic Databricks ML skills.

## Validating a deployed model: five layers

These checks are guidance: they help users confirm they can use the model appropriately. None of them blocks anyone from trying the model; report each result in one line and carry on. Parity checks alone can't show fitness for the task: a wrapper can match upstream exactly and still serve the wrong output (for example the wrong pooling of a transformer's outputs).

| Layer | Question | Who runs it, when |
| --- | --- | --- |
| 1. Fidelity | Does the served model give the same output as upstream code on real samples, and is that output the representation the paper or tutorial uses (pooling, layer)? | Each deployment; minutes |
| 2. Contract | Do the signature and malformed-input errors behave as stated? Does every documented control change the output? Is a cell's output the same alone and in a batch? | Each deployment; minutes |
| 3. Chained parts | Is each chained part complete and correct? For an index: every row present, found by search, recall vs exact search, filters return nothing they exclude | When an index is built; minutes |
| 4. Reproduce a claim | Does it reproduce one result from its paper or official tutorial, on pinned public data? | Skill maintainers, once per model revision, size or wrapper output. Record the result in the model file's "Validation record" so users can see what was checked; if none is recorded, say so and continue |
| 5. Field-style use | Does it hold up the way people use it: new studies, data after the training cut-off, a simple baseline, sensitivity to settings? | Optional, for users with a scientific question |

Layer 4 (reference test, from the model file's "Intended use and reference test"):

* Use the paper's protocol and output representation, on its data or a stated stand-in (prefer data after the training cut-off).
* Check the metric code on a known answer first (labels as the embedding ~1; random ~0).
* If the served output scores far below the paper's representation, it's a deployment defect, not a model limit.
* Run it before creating a serving endpoint when you can: it needs only the registered model on a notebook GPU, so a wrong output representation shows up before paying for serving. (TEDDY: a first-token output scored coarse NMI 0.51 vs 0.95 for mean pooling, and every layer 1-3 check had passed.) Then confirm on the endpoint that it gives the same vectors (layer 1).

Layer 5 (field-style use):

* Split by study (CELLxGENE: collection), and report data that may be in the training corpus separately.
* Report a simple baseline (single-cell: log-normalised counts, 2,000 variable genes, 50 PCs). If it wins, say so.
* Record the settings that define the result (neighbours, metric, pooling) and check it doesn't hinge on one.

Within-sample accuracy alone overstates what a new study will get; report layers 4-5 next to it.

## Evaluating the skill

Use this part to compare this skill (Vital-Skills `open-weight-models`) with no skills and with the generic Databricks ML skills.

## Configurations

* A: generic Databricks skills plus this skill
* B: skills disabled
* C: generic skills enabled but this skill disabled

The desired result for open-weight health and life sciences (HLS) model cases is A passes while B and C fail or omit important model-specific decisions. If C passes, this skill may be redundant for that case.

## Scenario groups

Test at least:

* custom PyFunc packaging for a Hugging Face checkpoint
* Git or Zenodo code/weight provenance
* model signature and `input_example`
* optional controls through SDK and HTTP
* AnnData, sparse matrix, FASTA, YAML, and structure inputs
* UC registration and versioned manifests
* Model Serving versus Jobs selection
* GPU/database/cache requirements
* AI Gateway and inference-table data minimization
* each model reference and an unknown-model fallback
* the model's reference test (layer 4) and a cross-study split with a baseline (layer 5)
* negative cases that should route to generic ML guidance

## Scoring dimensions

Score each run for:

* trigger precision
* correct reuse of generic guidance
* model-specific correctness
* explicit assumptions and open questions
* serving/Jobs decision quality
* signature and payload completeness
* provenance and license handling
* safety and data-governance awareness
* reproducibility and validation coverage (all five layers above)

Run each prompt multiple times. Keep prompts, configuration, agent version, skill revision, and results in a version-controlled evaluation set. Update the relevant model reference when a test exposes a model-specific gap; update the core skill only when the shared routing or contract changes.
