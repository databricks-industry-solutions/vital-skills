# HLS model reference template

Copy this file to `references/models/<model-family>.md` for a new model. Replace every placeholder and remove sections that do not apply.

## Identity

* Model family:
* Upstream code URL:
* Model card or documentation URL:
* Weight/checkpoint URL:
* Reviewed date:
* Upstream revision:
* Code license:
* Weight and database terms:

## What this model does

State the supported task in one paragraph. Distinguish research capability from clinical or regulatory validation.

## Inputs and outputs

Document the native inputs, the recommended Databricks transport representation, an `input_example`, the signature shape, and the output serialization. Include feature order and preprocessing requirements.

## Artifacts and runtime

List:

* code packages and pinned versions
* checkpoints and tokenizers
* reference databases
* cache directories
* CPU, memory, GPU, and storage expectations
* whether multiple processes or containers are required
* whether runtime internet access is prohibited or unnecessary

## Wrapper boundary

Describe `load_context`, request normalization, prediction, postprocessing, and failure behavior. State whether optional controls are explicit model inputs or tested `params`.

## Deployment recommendation

Choose Model Serving, Jobs, or hybrid. Explain the decision using latency, resource, request-size, database, and output-size constraints. Gate costly steps (endpoint, deployment, index) behind a `run_go` widget whose saved default is always `"false"` (SKILL.md Troubleshooting #14):

* Request asks to deploy: say in one short line that the gate is off by default and that, because they asked, you will switch it on for this run and create <endpoint name, GPU tier, any index> unless they stop you — then proceed (or tell the user how to switch it on if you cannot).
* Request says both "deploy" and "write the code": treat it as a deploy request (the code is written either way) and say so in the same one line before switching it on.
* Request asks only for code: leave the gate off and say how to turn it on.
* Anything beyond the request (larger GPU tier, extra endpoints, deleting or replacing resources): stop and ask first.

## Registration and observability

Describe the Unity Catalog model artifact and versioning approach. If AI Gateway inference tables or usage tracking are appropriate, specify what is safe to log and what must be redacted or represented by a governed URI.

## Validation

Include:

* import and offline-startup checks
* upstream-parity case: a few real samples through the model's own upstream preprocessing and inference functions (name them, with module paths) and through the wrapper — same token/input sets (order can differ on near-tied values), output cosine ~1.0 against a stated threshold
* signature and payload tests
* malformed-input tests
* resource-limit tests
* target-runtime deployment test
* model-specific scientific sanity checks

## Open questions

List uncertainties that require upstream or platform exploration before implementation.
