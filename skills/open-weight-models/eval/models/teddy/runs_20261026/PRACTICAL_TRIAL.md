# TRIAL — practical rescore of the 2026-10-27 TEDDY runs (not adopted)

**Status: trial for review only.** Nothing here changes the official results: the
replicated checks, the new checks, the outcome checks, the score files in `scores/` and
`RESULTS.md` are unchanged. The practical checks live in `eval/practical_checks.py`
(tests: `tests/test_practical_checks.py`) and are **not** wired into
`eval/score_open_weight_models.py`. Raw counts only — no percentages, no single score.

**Words used here.** CPU = central processing unit and GPU = graphics processing unit (the
two kinds of serving hardware; `GPU_SMALL` is Model Serving's smallest GPU size). UC =
Unity Catalog (where models are registered). HF = Hugging Face (where TEDDY's files live);
"Xet" = HF's newer file-storage backend, which `teddy.md` says breaks downloads on
serverless unless turned off with `HF_HUB_DISABLE_XET`. AST = abstract syntax tree: the
notebook's Python is parsed as code (not searched as text), so comments and look-alike
strings do not count. AI Gateway = the endpoint setting that logs requests to an
inference table and tracks usage. "Upstream" = TEDDY's own published code (the `teddy`
Python package in the HF repository). "Run record" = the run's entry in
`run_records.json`, holding what was measured on the live endpoint before it was deleted.
Severity: **critical** = does not work or gives wrong embeddings; **major** = works but
unsafe, ungoverned, unverified or not reproducible; **minor** = robustness or style.

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
way** of reaching it. Goals about what the code does are read from the parsed code; goals
about what the endpoint did are read from the outcome in the run record.

## Headline — current checks vs practical checks, per run

Current = the committed score files (`scores/<run>_scores.json`), unchanged. Failed counts
by severity: critical / major / minor.

| Task | Run | Replicated passed | New passed | Outcome passed | Current failed (crit / major / minor) | Practical passed | Practical failed (crit / major / minor) |
|---|---|---|---|---|---|---|---|
| teddy-001 | b/001 a2 (primary) | 20 / 29 | 2 / 5 | 3 / 4 | 5 / 6 / 2 | 7 / 10 | 1 / 2 / 0 |
| teddy-001 | b/001 a1 | 21 / 29 | 2 / 5 | 2 / 4 | 3 / 8 / 2 | 5 / 10 | 1 / 4 / 0 |
| teddy-001 | ws/001 | 29 / 29 | 5 / 5 | 4 / 4 | 0 / 0 / 0 | 10 / 10 | 0 / 0 / 0 |
| teddy-002 | b/002 | 24 / 31 | 3 / 7 | 7 / 7 | 4 / 5 / 2 | 7 / 11 | 0 / 4 / 0 |
| teddy-002 | ws/002 | 30 / 31 | 7 / 7 | 7 / 7 | 0 / 0 / 1 | 10 / 11 | 0 / 1 / 0 |

The convention check (shared single-cell input format) is left out of both sides, as in
`RESULTS.md`. teddy-002 has one more practical check than teddy-001
(`similarity_search_works`).

Read in one line: under the practical checks the baselines' remaining failures are about
**governance, hardware and verification** (no AI Gateway logging, CPU serving, no
comparison against upstream code, missing revision pin) and about **what the endpoint
accepts** (b/001's token-ID input) — not about how they downloaded or pinned. ws/002
gains one failure (`fidelity_verified`).

## The practical checks

Metric names start with `practical/`. "Code" = read from the parsed notebook; "record" =
read from the run record.

| Check | Severity | Goal | Accepted ways | Fails when | Evidence |
|---|---|---|---|---|---|
| `download_location_ok` | critical | Model files go to a location that exists on serverless | a `/tmp` path; `tempfile.mkdtemp()` / `TemporaryDirectory()`; a UC Volume path (`/Volumes/...`); the same as the destination of a download run in a subprocess or shell command | any `/local_disk0` path in the code; a destination that is none of these, including the default HF cache (not stated in the code) | code |
| `xet_download_ok` | critical | The HF download of the TEDDY repository completes | `HF_HUB_DISABLE_XET` set anywhere in the code; the download run in a subprocess / clean environment (including a `%sh` or `!` `hf download` command); **or** evidence it completed: a later cell loads files it wrote (`from_pretrained`, safetensors `load_file`, `load_state_dict`, `log_model(artifacts=...)`) **and** the endpoint built from them reached READY (record) | none of these | code + record |
| `model_loads_with_pinned_deps` | critical | The served model loads: transformers 5.x excluded or not used, and the endpoint reached READY | transformers pinned so 5.x is excluded (`==4.41.0`, `==4.x`, `<5`, `~=4.x`) in the serving requirements, or a serving pin to the installed version (`transformers=={transformers.__version__}`) when `%pip` limits it to `<5`; **or** a model path that does not use transformers (plain-PyTorch re-implementation) | a spec that allows 5.x; transformers used but never pinned; endpoint not READY (record `outcome/endpoint_ready`) | code + record |
| `fidelity_verified` | major | The embeddings were compared against an upstream reference | a cell that compares outputs (cosine, `allclose`, largest absolute difference) with a reference computed by the upstream package's own code (e.g. TEDDY's tokenizer or tokenization functions); or a cell that compares the wrapper with a direct forward pass of the upstream model (`<x>.model(...)`), in a notebook that loads the upstream package and defines no model module (`nn.Module`) of its own | no such comparison; a re-implementation compared only with itself (a riskier valid alternative — this is a scientific-risk check); endpoint-vs-notebook checks of the same wrapper | code |
| `signature_and_example` | critical | The logged model has a signature and an input example | `log_model(signature=..., input_example=...)` with the signature from `infer_signature(...)`, an explicit `ModelSignature(...)` or any other value | no `log_model` call passes both | code |
| `registered_in_uc` | major | The model is registered in UC | the new check's logic, unchanged (`scorers.uc_provenance()["uc_registry"]`) | as that check | code |
| `serves_real_cell_data` | critical | The endpoint embeds a cell from its expression and returns valid embeddings | record `outcome/accepts_expression_input` **and** `outcome/returns_valid_embeddings`, each in the endpoint's own declared format | either not passed or not measured | record |
| `gpu_serving` | major | Served on a GPU workload (task guidelines: `GPU_SMALL`) | record `endpoint_config`: every served entity has a `GPU_*` workload type; without that record, `ServingModelWorkloadType.GPU_*` in the code | a CPU workload — on CPU, b/002's endpoint took about 15 s per cell (run record notes) | record (code fallback) |
| `governance_logging` | major | AI Gateway inference table **and** usage tracking enabled | the new checks' logic, unchanged (`scorers.ai_gateway_logging()`: both on one attached config); the record's endpoint settings are quoted in the reason | as those checks | code |
| `provenance_recorded` | major | Upstream revision pinned and licence recorded durably | revision: the HF download uses an explicit revision (a commit identifier, or the one resolved with `model_info`), not `main`; licence: a licence key or value in **any** durable record — MLflow run tags or params, logged-model tags/metadata, registered-model or model-version tags, the registered-model description, or a provenance manifest file (`json.dump` / `log_dict`) | no pinned revision, or the licence only in comments/markdown | code |
| `similarity_search_works` | critical | teddy-002 only: index ready and a nearest-neighbour query returns a cell-type column | record `outcome/index_ready` **and** `outcome/nn_returns_cell_type` | either not passed or not measured | record |

## Per-check results and why

| Check | b/001 a2 | b/001 a1 | b/002 | ws/001 | ws/002 |
|---|---|---|---|---|---|
| `download_location_ok` | pass — UC Volume (`snapshot_download(local_dir=VOLUME_DIR)`, :44–50; :28) | pass — UC Volume (:60–64; :21) | pass — `tempfile.mkdtemp()`, then copied to the Volume (:54–57) | pass — subprocess download to `/tmp/teddy_001` (:26, :42–55), then the Volume (:58) | pass — subprocess download to `/tmp/teddy_002` (:53, :90–106), then the Volume (:113) |
| `xet_download_ok` | pass — no Xet setting; download completed: weights size asserted (:52–53), loaded with `from_pretrained` (:155), endpoint READY | pass — completed: `from_pretrained` from the Volume (:102–103), endpoint READY | pass — completed: safetensors `load_file` (:98), endpoint READY | pass — `HF_HUB_DISABLE_XET` (:44, :47) in a subprocess (:55) | pass — `HF_HUB_DISABLE_XET` (:93, :105) in a subprocess (:106) |
| `model_loads_with_pinned_deps` | pass — `%pip "transformers>=4.41,<5"` (:18); serving pin = installed version (:258; 4.57.6 per run notes); READY | pass — `transformers==4.41.0` (:40, :310); READY | pass — plain-PyTorch re-implementation (:74–112), no transformers in the serving requirements (:312–318); READY | pass — `==4.41.0` (:105, :389); READY | pass — `==4.41.0` (:15, :63); READY |
| `fidelity_verified` | pass — each cell run alone through the upstream TeddyGModel vs the batched wrapper, largest difference < 1e-3 (:221–229), on cells tokenized with TEDDY's own `preprocess` + `tokenize` (:64–101) | **fail** — only a forward pass on random token IDs (:120–134); nothing compared | **fail** — own encoder (:74–112) compared only with itself (:326); the held-out-donor nearest-neighbour accuracy (:209–227) is a biology sanity signal, not a parity check | pass — "Known-answer check vs upstream tokenization": cosine > 0.9999 against TEDDY's `_build_batch_tensors` path (:334–367) | **fail** — compares the endpoint with the same wrapper in the notebook (:583–585, :822–825); the wrapper's preprocessing "mirrors upstream" (:172, :242–261) but is never compared with TEDDY's own tokenization; held-out 10-nearest-neighbour cell-type accuracy (:834) is a downstream sanity signal |
| `signature_and_example` | pass — `ModelSignature` (:242–249) + `input_example` (:277) | pass — `infer_signature` (:293) + :306 | pass — `ModelSignature` (:304) + :335 | pass — `infer_signature` (:381) + :407 | pass — `infer_signature` (:395) + :405 |
| `registered_in_uc` | pass (:305) | pass (:359) | pass (:359) | pass (:409) | pass (:444) |
| `serves_real_cell_data` | **fail** — record: takes pre-tokenized token IDs (`accepts_expression_input` fail); embeddings valid | **fail** — record: token IDs, and mixed-length batches fail (padding value −100 used as a token ID) | pass — record: gene IDs + raw counts in, valid 512-value embeddings out | pass — record | pass — record |
| `gpu_serving` | **fail** — record: CPU (code :326) | **fail** — record: CPU (:397) | **fail** — record: CPU (:419); about 15 s per cell | pass — record: `GPU_SMALL` (:465) | pass — record: `GPU_SMALL` (:480) |
| `governance_logging` | **fail** — endpoint created without `ai_gateway` (:353); record: both off | **fail** — (:465); record: both off | **fail** — (:439); record: both off | pass — `ai_gateway=AiGatewayConfig(...)` (:481); record: both on | pass — (:499–522); record: both on |
| `provenance_recorded` | pass — revision `8c880aa…` pinned (:32, :46); licence in MLflow run params (:269–270) | **fail** — no revision (:60–64); licence only in markdown (:6) | **fail** — no revision (:55); licence in run params (:338), so half met | pass — revision resolved with `model_info` and pinned (:50–52, :55); licence in manifest (:87, :97), run tags (:399), registered-model description (:414) | pass — revision resolved and pinned (:97–99, :106); licence in manifest (:123, :131), run tags (:408–409), `log_dict` (:413), description (:450–453), model-version tags (:455–457) |
| `similarity_search_works` | — | — | pass — record: index ready (3,538 rows), query returns `cell_type` | — | pass — record: index ready (1,000 rows), query returns `cell_type` |

`python3 skills/open-weight-models/eval/practical_checks.py` prints the same results with
the reason and line numbers the code found.

## Which current failures disappear, and what is new

Current failures (convention check left out) and what happens to each under the
practical set. ⚑ = marked as a known or likely false negative in `RESULTS.md`.

| Current sub-check | Failed in | Under practical | |
|---|---|---|---|
| `download_staging/uses_tmp` ⚑ | a2, a1, b/002 | `download_location_ok` passes for all three | **gone** — false negative fixed |
| `download_staging/hf_xet_disabled` ⚑ | a2, a1, b/002 | `xet_download_ok` passes (download completed) | **gone** — false negative fixed |
| `download_staging/deps_pinned` ⚑ | a2, b/002 | `model_loads_with_pinned_deps` passes | **gone** — false negative fixed |
| `forbidden_patterns/transformers` ⚑ | a2 | `model_loads_with_pinned_deps` passes (`>=4.41,<5` excludes 5.x) | **gone** — false negative fixed |
| `model_registration/infer_signature` ⚑ | a2, b/002 | `signature_and_example` passes (`ModelSignature`) | **gone** — false negative fixed |
| `pyfunc_quality/real_gene_ids` ⚑ (a2), real (a1) | a2, a1 | no practical counterpart; the goal behind it is closest to `fidelity_verified` (a2 pass, a1 fail) | gone — not scored |
| `model_registration/pip_reqs_from_source` ⚑ | ws/002 | no practical counterpart | gone — not scored |
| `pyfunc_quality/sys_modules_purge`, `endpoint_deployment/run_go_gate` (minor, `teddy.md` / `SKILL.md` conventions) | a2, a1, b/002 | no practical counterpart | gone — not scored |
| `new_checks/uc_provenance/provenance_tags` | a2, a1, b/002 | `provenance_recorded`: **a2 now passes** (licence in run params counts as durable); a1 and b/002 still fail | partly gone |
| `endpoint_deployment/ai_gateway_config`, `new_checks/ai_gateway/inference_table`, `new_checks/ai_gateway/usage_tracking` | a2, a1, b/002 | `governance_logging` fails for the same three | **kept**, as one check instead of three |
| `outcome/accepts_expression_input`; `outcome/returns_valid_embeddings` | a2, a1; a1 | `serves_real_cell_data` fails for a2 and a1 | **kept**, as one check, now critical (see below) |
| `smoke_tests/registration_smoke`, `smoke_tests/endpoint_query` | a1 | no practical counterpart | gone — **a real finding lost** (a1 never loads the registered model or queries its endpoint) |
| `new_checks/existing_resources/table` | b/002 | no practical counterpart | gone — **a real finding lost** (b/002 creates its table without looking for an existing one) |

All five ⚑ false negatives that have a practical counterpart disappear; the other two ⚑
(`real_gene_ids` for a2, `pip_reqs_from_source`) disappear only because the practical set
does not score them.

**New failures** the practical set introduces:

- `gpu_serving` (major): a2, a1 and b/002 served on CPU, which the task guidelines do not
  ask for (`GPU_SMALL`); `RESULTS.md` showed this only in the endpoint-settings table.
- `fidelity_verified` (major): a1, b/002 — and **ws/002**, the with-skill run, whose
  preprocessing is its own code (described as mirroring upstream) and is never compared
  with TEDDY's tokenization. ws/001 and a2 pass.

**Severity change:** a2's token-ID input was a major failure
(`outcome/accepts_expression_input`); folded into `serves_real_cell_data` it counts as
critical. That accounts for a2's one critical practical failure.

## Caveats

1. **Trial, one sample per arm** (two for baseline/teddy-001). Directional, not
   statistical. Nothing here is adopted.
2. **The practical set is narrower than the current set.** It drops the smoke-test,
   existing-resource and style checks, so two real baseline findings (above) are no
   longer counted. Adopting it as-is would trade false negatives for missed findings;
   keeping those checks next to it would avoid that.
3. **"Download completed" is read from code plus the endpoint outcome, not from logs.**
   The evidence route says a later cell loads files from the download and the endpoint
   built from them reached READY. The baselines downloaded without the Xet workaround and
   still finished, so the Xet failure `teddy.md` warns about did not happen in these runs;
   whether Xet still breaks on serverless was not tested here.
4. **`fidelity_verified` is a heuristic.** It looks, cell by cell, for a numeric
   comparison next to a call into the `teddy` package or a direct forward pass of the
   upstream model. A parity check written in another shape (e.g. a helper in another
   cell) would be missed; a comparison that uses upstream code for something unrelated
   would pass. `check_outcome.py --reference-endpoint` can record a cosine against a
   reference deployment, but no run recorded one, so no outcome evidence was available.
5. **Pinning to the installed version** (a2: `transformers=={transformers.__version__}`)
   is accepted because `%pip` limited the install to `<5`; the served version was 4.57.6
   (run notes), not the 4.41.0 TEDDY declares — it worked here, but it is a looser pin.
6. **Licence in MLflow run params counts as durable** (a2, b/002). The current
   `provenance_tags` check counts only tags; that is the whole difference for a2.
7. **Outcome evidence was partly measured by hand before teardown** (`RESULTS.md`,
   "Outcome") and cannot be re-measured; the practical record-based checks inherit that.
8. **The practical checks were written after seeing these five notebooks.** The tests
   pin both passing and failing cases for each accepted alternative, but the set has not
   been tried on a fresh round.

## How to reproduce

```
uv run --isolated --with-requirements skills/open-weight-models/tests/requirements.txt \
  python skills/open-weight-models/eval/practical_checks.py
uv run --isolated --with pytest --with-requirements skills/open-weight-models/tests/requirements.txt \
  python -m pytest -q skills/open-weight-models/tests/test_practical_checks.py
```

`test_trial_results_on_committed_exports` pins the per-run pass/fail sets reported above.
