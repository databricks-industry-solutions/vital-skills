# TRIAL — practical rescore of the 2026-10-27 TEDDY runs (not adopted)

**Status: trial for review only.** Nothing here changes the official results: the
replicated checks, the new checks, the outcome checks, the score files in `scores/` and
`RESULTS.md` are unchanged. The practical checks live in `eval/practical_checks.py`
(tests: `tests/test_practical_checks.py`) and are **not** wired into
`eval/score_open_weight_models.py`. Raw counts only — no percentages, no single score.
This is the second version of the trial; "What changed since the first version" (below)
lists the review fixes.

**Words used here.** CPU = central processing unit and GPU = graphics processing unit (the
two kinds of serving hardware; `GPU_SMALL` is Model Serving's smallest GPU size). UC =
Unity Catalog (where models are registered). HF = Hugging Face (where TEDDY's files live);
"Xet" = HF's newer file-storage backend, which `teddy.md` says breaks downloads on
serverless unless turned off with `HF_HUB_DISABLE_XET` in a clean process. SHA = the
40-character hexadecimal identifier of one commit (one fixed version of the repository).
AST = abstract syntax tree: the notebook's Python is parsed as code (not searched as text),
so comments and look-alike strings do not count. AI Gateway = the endpoint setting that
logs requests to an inference table and tracks usage. "Upstream" = TEDDY's own published
code (the `teddy` Python package in the HF repository). "Candidate" = what the run built:
its wrapper's `predict`, the logged model, or its endpoint. "Run record" = the run's entry
in `run_records.json`, holding what was measured on the live endpoint before it was
deleted. Severity: **critical** = does not work or gives wrong embeddings; **major** =
works but unsafe, ungoverned, unverified or not reproducible; **minor** = robustness or
style. A **guideline** = a task guideline the prompts do not ask for, reported on its own
and not counted as a failure (like the shared-input-format convention check).

Run labels: **b/001 a2** = `baseline/teddy-001_attempt2.py` (primary baseline sample),
**b/001 a1** = `baseline/teddy-001_attempt1.py` (second sample), **b/002** =
`baseline/teddy-002.py`, **ws/001** = `with_skill/teddy-001.py`, **ws/002** =
`with_skill/teddy-002.py`. Line numbers below are lines of those files.

## What "practical" means

The current checks were written from the skill's own text (`teddy.md`, `SKILL.md`), so
several of them look for one specific way of doing a step — the literal `/tmp`, the
`HF_HUB_DISABLE_XET` workaround, the string `infer_signature`, `transformers==4.41.0`. A
baseline that reached the same goal another valid way fails them (the ⚑ false negatives
in `RESULTS.md`). Each practical check instead names a **goal** and accepts **any valid
way** of reaching it. Goals about what the code does are read from the parsed code, with
values followed across cells; goals about what the endpoint did are read from the outcome
in the run record.

## Headline — current checks vs practical checks, per run

Current = the committed score files (`scores/<run>_scores.json`), unchanged. Failed counts
by severity: critical / major / minor.

| Task | Run | Replicated passed | New passed | Outcome passed | Current failed (crit / major / minor) | Practical passed | Practical failed (crit / major / minor) | GPU guideline (not counted) |
|---|---|---|---|---|---|---|---|---|
| teddy-001 | b/001 a2 (primary) | 20 / 29 | 2 / 5 | 3 / 4 | 5 / 6 / 2 | 10 / 12 | 0 / 2 / 0 | no (CPU) |
| teddy-001 | b/001 a1 | 21 / 29 | 2 / 5 | 2 / 4 | 3 / 8 / 2 | 6 / 12 | 1 / 5 / 0 | no (CPU) |
| teddy-001 | ws/001 | 29 / 29 | 5 / 5 | 4 / 4 | 0 / 0 / 0 | 12 / 12 | 0 / 0 / 0 | yes (GPU_SMALL) |
| teddy-002 | b/002 | 24 / 31 | 3 / 7 | 7 / 7 | 4 / 5 / 2 | 9 / 13 | 0 / 4 / 0 | no (CPU) |
| teddy-002 | ws/002 | 30 / 31 | 7 / 7 | 7 / 7 | 0 / 0 / 1 | 12 / 13 | 0 / 1 / 0 | yes (GPU_SMALL) |

The convention check (shared single-cell input format) is left out of both sides, as in
`RESULTS.md`. teddy-002 has one more practical check than teddy-001
(`similarity_search_works`).

Read in one line: under the practical checks, the primary baseline (a2) fails only on
**what its endpoint accepts** (pre-tokenized token IDs) and **governance** (no AI Gateway
logging); b/002 fails on **verification, safety and provenance** (no comparison with
upstream code, table created without looking first, no revision pin) and governance; a1,
the weakest run, also returns broken embeddings for mixed-length batches and never tests
its endpoint. ws/002 fails one check (`fidelity_verified`).

## The practical checks

Metric names start with `practical/`. "Code" = read from the parsed notebook; "record" =
read from the run record.

| Check | Severity | Goal | Accepted ways | Fails when | Evidence |
|---|---|---|---|---|---|
| `download_location_ok` | critical | Model files are downloaded and staged where serverless compute can write | download destination: a `/tmp` path; `tempfile.mkdtemp()` / `TemporaryDirectory()`; a UC Volume path (`/Volumes/...`); the same as the destination of a download run in a subprocess or shell command | `/local_disk0` as the download destination or as the destination of a `shutil` copy/move of the files (staging); a destination that is none of the accepted ones, including the default HF cache (not stated in the code). Other `/local_disk0` strings (messages, unrelated paths) do not count | code |
| `xet_download_ok` | critical | The HF download of the TEDDY repository completes | the download runs in a subprocess whose `env=` sets `HF_HUB_DISABLE_XET`, or whose script sets it before importing `huggingface_hub` (`teddy.md`, "Hugging Face download"); a `%sh` / `!` `hf download` with it set on the line or exported earlier in the cell; in the notebook, set before the first `huggingface_hub` import and before the download (`teddy.md` warns this is not reliable on AI Runtime); **or recorded evidence** the download completed: a later cell loads files it wrote (`from_pretrained`, safetensors `load_file`, `load_state_dict`, `log_model(artifacts=...)`) **and** the run record shows the endpoint READY serving the registered version | no download call; the variable set only after the download or import; a subprocess without it; and no recorded evidence | code + record |
| `model_loads_with_pinned_deps` | critical | The served model loads: transformers 5.x excluded or not used, and the endpoint reached READY | transformers pinned so 5.x is excluded (`==4.41.0`, `==4.x`, `<5`, `~=4.x`) in the serving requirements, or a serving pin to the installed version (`transformers=={transformers.__version__}`) when `%pip` limits it to `<5`; **or** a model path that does not use transformers (plain-PyTorch re-implementation) | a spec that allows 5.x; transformers used but never pinned; endpoint not READY (record `outcome/endpoint_ready`) | code + record |
| `fidelity_verified` | major | The candidate's embeddings were compared against an upstream reference | a comparison (cosine, `allclose` / `isclose`, largest absolute difference) whose **one side derives from upstream code** — a call into the `teddy` package (e.g. its tokenizer or tokenization functions), or a direct forward pass of the upstream model (`<x>.model(...)`) in a notebook that loads the upstream package and defines no model module (`nn.Module`) of its own — **and whose other side comes from the candidate** (`.predict(...)` or an endpoint query). Values are followed through assignments, loops, tuple unpacking and the notebook's own functions, across cells | no such comparison; two candidate outputs compared with each other (endpoint vs the same wrapper); a re-implementation compared only with itself (a riskier valid alternative — this is a scientific-risk check); an upstream call that does not feed a side of the comparison | code |
| `signature_and_example` | critical | The logged model has a signature and an input example | `log_model(signature=..., input_example=...)` with the signature built by `infer_signature(...)` or `ModelSignature(...)`; another signature expression only if the run record shows the endpoint READY serving the registered version (registration, which needs a signature, succeeded) | no `log_model` call with both; a `None` signature; an unrecognised one without that recorded evidence | code (+ record) |
| `registered_in_uc` | major | The model is registered in UC | the new check's logic, unchanged (`scorers.uc_provenance()["uc_registry"]`) | as that check | code |
| `serves_valid_embeddings` | critical | The endpoint returns valid embeddings | record `outcome/returns_valid_embeddings` (in the endpoint's own declared format) | not passed or not measured | record |
| `serves_expression_input` | major | The endpoint embeds a cell from its expression, not pre-tokenized token IDs | record `outcome/accepts_expression_input` | not passed or not measured | record |
| `endpoint_smoke_tested` | major | The run queried its own endpoint and checked the result | an endpoint query (`serving_endpoints.query`, or an HTTP POST to `.../invocations`) whose result feeds an `assert`, a comparison (`==`, `<`, ...) or a numeric comparison (cosine, `allclose`, absolute difference); values followed across cells and through notebook functions | no endpoint query; the result only printed or displayed | code |
| `existing_resources_checked` | major | The run looked for an existing endpoint / table / index before creating or deleting it | the new check's logic, unchanged (`scorers.existing_resources()`), for the resources listed for the task in `expectations.json` (teddy-001: endpoint; teddy-002: endpoint, table, index) | as that check, for any listed resource | code |
| `governance_logging` | major | AI Gateway inference table **and** usage tracking enabled | the new checks' logic, unchanged (`scorers.ai_gateway_logging()`: both on one attached config); the record's endpoint settings are quoted in the reason | as those checks | code |
| `provenance_recorded` | major | Upstream revision pinned to a commit and the licence recorded in a persisted record | revision: the HF download uses a 40-character commit SHA, or the commit resolved at download time (`model_info(...).sha`) when that resolved value is also recorded under a revision / commit / sha key in a persisted sink; licence: a non-empty value (not empty, `None`, unknown, n/a) under a licence key in a persisted sink — MLflow params or tags (`set_tag(s)`, `log_param(s)`, `start_run` / `log_model` tags), registered-model or model-version tags, or a manifest file (`json.dump` to a file, `mlflow.log_dict`) | no revision; a branch or tag name (`main`, `dev`); an unresolved expression; a resolved commit never recorded; a licence that is missing, empty or unknown, or only in comments, markdown or a free-text model description | code |
| `similarity_search_works` | critical | teddy-002 only: index ready and a nearest-neighbour query returns a cell-type column | record `outcome/index_ready` **and** `outcome/nn_returns_cell_type` | either not passed or not measured | record |

**Guideline, reported on its own (metric `guideline/gpu_serving`, not counted):** served on
a GPU workload, as the task guidelines suggest (`GPU_SMALL`). Read from the run record's
endpoint settings only — the code is never read, so a `GPU_*` reference that was never
deployed does not count; no record = not measured. The prompts do not ask for a GPU and
the CPU endpoints worked. Context: on CPU, b/002's endpoint took about 15 s per cell (run
record notes).

## Per-check results and why

| Check | b/001 a2 | b/001 a1 | b/002 | ws/001 | ws/002 |
|---|---|---|---|---|---|
| `download_location_ok` | pass — UC Volume (`snapshot_download(local_dir=VOLUME_DIR)`, :44–50; :28) | pass — UC Volume (:60–64; :21) | pass — `tempfile.mkdtemp()` (:54–55), then copied to the Volume (:57) | pass — subprocess download to `/tmp/teddy_001` (:26, :42–55), copied to the Volume (:58) | pass — subprocess download to `/tmp/teddy_002` (:53, :90–106), copied to the Volume (:113) |
| `xet_download_ok` | pass — no Xet setting; **recorded evidence**: weights loaded with `from_pretrained` (:155) after the download (:44), record: endpoint READY serving v1 | pass — recorded evidence: `from_pretrained` from the Volume (:102) after the download (:60), record READY | pass — recorded evidence: safetensors `load_file` (:98) after the download (:55), record READY | pass — `HF_HUB_DISABLE_XET` in the subprocess `env=` dict (:44, passed at :55) | pass — `env.update({"HF_HUB_DISABLE_XET": ...})` (:105) on the subprocess env (:106) |
| `model_loads_with_pinned_deps` | pass — `%pip "transformers>=4.41,<5"` (:18); serving pin = installed version (:258; 4.57.6 per run notes); READY | pass — `transformers==4.41.0` (:40, :310); READY | pass — plain-PyTorch re-implementation (:74–112), no transformers in the serving requirements (:312–318); READY | pass — `==4.41.0` (:105, :389); READY | pass — `==4.41.0` (:15, :63); READY |
| `fidelity_verified` | pass — :226 compares the wrapper's `predict` output (`pred`, :219) with a direct forward pass of the upstream TeddyGModel on each cell alone (:225), on cells tokenized with TEDDY's own `preprocess` + `tokenize` (:64–101) | **fail** — no comparison at all; only a forward pass on random token IDs (:120–134) | **fail** — own encoder (:74–112); its one comparison (:326) has the candidate `predict` on one side and the notebook's own encoder, not upstream code, on the other | pass — :365 cosine of the wrapper's `predict` output (`E`, :324–325) with a forward pass on token IDs from TEDDY's `_build_batch_tensors` (:345, :355–364) | **fail** — its comparison (:583) is endpoint vs the same wrapper in the notebook (both candidate); the wrapper's preprocessing "mirrors upstream" (:172, :242–261) but is never compared with TEDDY's own tokenization; held-out 10-nearest-neighbour cell-type accuracy (:834) is a downstream sanity signal, not parity |
| `signature_and_example` | pass — `ModelSignature` (:242–249) + `input_example` (:271–277) | pass — `infer_signature` (:293) + :306 | pass — `ModelSignature` (:304) + :335 | pass — `infer_signature` (:381) + :407 | pass — `infer_signature` (:395) + :405 |
| `registered_in_uc` | pass (:305) | pass (:359) | pass (:359) | pass (:409) | pass (:444) |
| `serves_valid_embeddings` | pass — record: one finite 512-value vector per cell, equal and mixed lengths | **fail** — record: mixed-length batches fail (padding value −100 used as a token ID) | pass — record | pass — record | pass — record |
| `serves_expression_input` | **fail** — record: takes pre-tokenized token IDs | **fail** — record: token IDs | pass — record: gene IDs + raw counts | pass — record | pass — record |
| `endpoint_smoke_tested` | pass — queries the endpoint (:385) and computes the largest difference from the local result (:392) | **fail** — the endpoint is never queried | pass — queries it inside `predict_cell_type` (:538) and checks predicted vs true subtype (:566) | pass — queries it (REST helper :530, SDK query :532) and compares with the local result by cosine (:536), with asserts (:539–540) | pass — queries it (:579), cosine vs local + assert (:583–585) |
| `existing_resources_checked` | pass — endpoint looked up first (:339) | pass — endpoint looked up first (:442) | **fail** — table (the existing check's rule): the create (:373) comes before any existence check; endpoint and index pass. See caveat 11 | pass | pass — endpoint, table, index all checked first |
| `governance_logging` | **fail** — endpoint created without `ai_gateway` (:353); record: both off | **fail** — (:465); record: both off | **fail** — (:439); record: both off | pass — `ai_gateway=AiGatewayConfig(...)` (:481); record: both on | pass — (:499–522); record: both on |
| `provenance_recorded` | pass — revision `8c880aa…` (40-character SHA, :32) used for the download (:46); licence `apache-2.0` in MLflow run params (:269–270) | **fail** — no revision (:60–64); licence only in markdown (:6) | **fail** — no revision (:55); licence in run params (:338), so half met | pass — commit resolved with `model_info(...).sha` (:50–52, in the subprocess at :55) and recorded (manifest :81–97, run params/tags :398–400); licence in the manifest (:87, :97) and run tags (:399) | pass — commit resolved (:97–99, subprocess :106) and recorded (manifest :116–131, tags :408–411); licence in the manifest (:123, :131), run tags (:409), `log_dict` (:413), model-version tags (:455–457) |
| `similarity_search_works` | — | — | pass — record: index ready (3,538 rows), query returns `cell_type` | — | pass — record: index ready (1,000 rows), query returns `cell_type` |
| *guideline:* GPU serving (not counted) | no — record: CPU | no — record: CPU | no — record: CPU; about 15 s per cell | yes — record: `GPU_SMALL` | yes — record: `GPU_SMALL` |

`python3 skills/open-weight-models/eval/practical_checks.py` prints the same results with
the reason and line numbers the code found.

## Which current failures disappear, and what is new

Current failures (convention check left out) and what happens to each under the
practical set. ⚑ = marked as a known or likely false negative in `RESULTS.md`.

| Current sub-check | Failed in | Under practical | |
|---|---|---|---|
| `download_staging/uses_tmp` ⚑ | a2, a1, b/002 | `download_location_ok` passes for all three | **gone** — false negative fixed |
| `download_staging/hf_xet_disabled` ⚑ | a2, a1, b/002 | `xet_download_ok` passes on recorded evidence that the download completed | **gone** — false negative fixed |
| `download_staging/deps_pinned` ⚑ | a2, b/002 | `model_loads_with_pinned_deps` passes | **gone** — false negative fixed |
| `forbidden_patterns/transformers` ⚑ | a2 | `model_loads_with_pinned_deps` passes (`>=4.41,<5` excludes 5.x) | **gone** — false negative fixed |
| `model_registration/infer_signature` ⚑ | a2, b/002 | `signature_and_example` passes (`ModelSignature`) | **gone** — false negative fixed |
| `pyfunc_quality/real_gene_ids` ⚑ (a2), real (a1) | a2, a1 | no practical counterpart; the closest goal is `fidelity_verified` (a2 pass, a1 fail) | gone — not scored |
| `model_registration/pip_reqs_from_source` ⚑ | ws/002 | no practical counterpart | gone — not scored |
| `pyfunc_quality/sys_modules_purge`, `endpoint_deployment/run_go_gate` (minor, `teddy.md` / `SKILL.md` conventions) | a2, a1, b/002 | no practical counterpart | gone — not scored |
| `new_checks/uc_provenance/provenance_tags` | a2, a1, b/002 | `provenance_recorded`: **a2 now passes** (licence in run params counts as a persisted record; its revision is a commit SHA); a1 and b/002 still fail | partly gone |
| `endpoint_deployment/ai_gateway_config`, `new_checks/ai_gateway/inference_table`, `new_checks/ai_gateway/usage_tracking` | a2, a1, b/002 | `governance_logging` fails for the same three | **kept**, as one check instead of three |
| `outcome/returns_valid_embeddings` (critical) | a1 | `serves_valid_embeddings` (critical) fails for a1 | **kept**, same severity |
| `outcome/accepts_expression_input` (major) | a2, a1 | `serves_expression_input` (major) fails for both | **kept**, same severity |
| `smoke_tests/registration_smoke`, `smoke_tests/endpoint_query` | a1 | `endpoint_smoke_tested` fails for a1 (the registration smoke test has no practical counterpart) | **kept** (endpoint part) |
| `new_checks/existing_resources/table` | b/002 | `existing_resources_checked` fails for b/002 | **kept** |

All five ⚑ false negatives that have a practical counterpart disappear; the other two ⚑
(`real_gene_ids` for a2, `pip_reqs_from_source`) disappear only because the practical set
does not score them.

**New failures** the practical set introduces: `fidelity_verified` (major) for a1, b/002
— and **ws/002**, the with-skill run, whose preprocessing is its own code (described as
mirroring upstream) and is never compared with TEDDY's tokenization. ws/001 and a2 pass.
GPU serving is no longer a failure: it is the separate guideline column (CPU for all three
baselines).

## What changed since the first version of this trial (`ff3a64b`)

Fixes from review; results per run before → after (practical passed / total, failed
critical / major).

| Change | Effect on these runs |
|---|---|
| `xet_download_ok`: `HF_HUB_DISABLE_XET` must be set **before** the download / `huggingface_hub` import, or in the subprocess `env=` / script before its import; a bare subprocess no longer passes; otherwise recorded evidence is needed (later file load **and** record: endpoint READY serving the registered version) | none — ws/001 and ws/002 set it in the subprocess env; the baselines pass on recorded evidence |
| `fidelity_verified`: follows values across cells; one side of a comparison must derive from upstream code and the other from the candidate (an upstream call merely in the same cell no longer counts) | none — a2 and ws/001 still pass with a real upstream-vs-candidate comparison; a1, b/002, ws/002 still fail |
| `provenance_recorded`: revision must be a 40-character SHA or a resolved commit that is also recorded; licence must be a non-empty value in MLflow params/tags, registry tags or a manifest (a model description no longer counts) | none — ws/001's description no longer counts, but its manifest and run tags do |
| `serves_real_cell_data` split into `serves_valid_embeddings` (critical) and `serves_expression_input` (major), matching `expectations.json` | a2's token-ID input is now **major** (was counted critical): a2 failed critical 1 → 0 |
| `gpu_serving` moved out of the score into a separate guideline, read from the recorded workload type only (no code fallback) | the three baselines lose one major failure each |
| `download_location_ok`: only `/local_disk0` as a download destination or staging copy fails, not any `/local_disk0` string | none |
| `signature_and_example`: needs `infer_signature(...)` / `ModelSignature(...)`, or recorded READY evidence for another expression | none |
| Added `endpoint_smoke_tested` and `existing_resources_checked` (the two findings the first version dropped) | a1 fails `endpoint_smoke_tested`; b/002 fails `existing_resources_checked` |

| Run | First version | This version |
|---|---|---|
| b/001 a2 | 7 / 10, failed 1 / 2 | 10 / 12, failed 0 / 2 |
| b/001 a1 | 5 / 10, failed 1 / 4 | 6 / 12, failed 1 / 5 |
| ws/001 | 10 / 10, failed 0 / 0 | 12 / 12, failed 0 / 0 |
| b/002 | 7 / 11, failed 0 / 4 | 9 / 13, failed 0 / 4 |
| ws/002 | 10 / 11, failed 0 / 1 | 12 / 13, failed 0 / 1 |

## Caveats

1. **Trial, one sample per arm** (two for baseline/teddy-001). Directional, not
   statistical. Nothing here is adopted.
2. **The practical set still leaves some current checks out:** the registration smoke test,
   the shape check, real gene IDs and the style conventions (`sys.modules` purge, `run_go`
   gate). None changes a run's result here except as listed above, but adopting the set
   would stop measuring them.
3. **"Download completed" is recorded evidence of a sort, not a download log.** It combines
   a later cell that loads the files with the run record's endpoint READY + registered
   version. The baselines downloaded without the Xet workaround and still finished, so the
   Xet failure `teddy.md` warns about did not happen in these runs; whether Xet still breaks
   on serverless was not tested here.
4. **Value-following is an approximation.** A name used at a line takes its latest
   assignment at or before that line (inside a notebook function, a name with no earlier
   assignment takes all of them); branches of an `if` statement are not tracked separately,
   and values passed through function arguments or object attributes (other than calls to
   the notebook's own functions) are not followed. The tests pin the review's false
   positive (`teddy.version()` next to an unrelated comparison) and a cross-cell
   comparison, but a parity check written in another shape could still be missed.
5. **A resolved commit counts as pinned when any revision / commit / sha key is recorded**
   (ws/001, ws/002); the check does not prove the recorded value is the same one the
   download used — reading the notebooks, it is (both read it back from the download).
6. **`fidelity_verified` treats a direct forward of the upstream model as a reference**
   (a2) when the notebook loads the upstream package and defines no model module of its
   own; it does not check which weights that model loaded. `check_outcome.py
   --reference-endpoint` can record a cosine against a reference deployment, but no run
   recorded one.
7. **Pinning to the installed version** (a2: `transformers=={transformers.__version__}`)
   is accepted because `%pip` limited the install to `<5`; the served version was 4.57.6
   (run notes), not the 4.41.0 TEDDY declares — it worked here, but it is a looser pin.
8. **Licence in MLflow run params counts as persisted** (a2, b/002). The current
   `provenance_tags` check counts only tags; that is the whole difference for a2.
9. **Outcome evidence was partly measured by hand before teardown** (`RESULTS.md`,
   "Outcome") and cannot be re-measured; the record-based checks inherit that.
10. **The practical checks were written after seeing these five notebooks** (and revised
    after review). The tests pin passing and failing cases for each accepted alternative,
    but the set has not been tried on a fresh round.
11. **b/002's table failure comes from the existing check's rule; the real risk is
    elsewhere.** That check reads `CREATE TABLE IF NOT EXISTS` (:373) as a create with no
    earlier look, although `IF NOT EXISTS` is itself safe. The real risk comes later:
    `insertInto(CELLS_TABLE, overwrite=True)` (:404) replaces the rows of a table that
    already existed without looking first, which the existing check does not see. The
    result (fail) stands either way; the reason differs.

## How to reproduce

```
uv run --isolated --with-requirements skills/open-weight-models/tests/requirements.txt \
  python skills/open-weight-models/eval/practical_checks.py
uv run --isolated --with pytest --with-requirements skills/open-weight-models/tests/requirements.txt \
  python -m pytest -q skills/open-weight-models/tests/test_practical_checks.py
```

`test_trial_results_on_committed_exports` pins the per-run pass/fail sets and the GPU
guideline reported above.
