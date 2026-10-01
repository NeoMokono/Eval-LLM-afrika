# Build prompt: AfriGemma evaluation framework

> **How to use:** put the architecture doc in the repo at `docs/EVAL_ARCHITECTURE.md` (or change the path below), open the repo in VS Code, and paste everything under the line into the Claude agent. It works in phases and stops for review after each one, so you stay in control of the design calls.

---

You are building `afrigemma-eval`, the evaluation suite for **AfriGemma**, a supervised fine-tuning (SFT) project at DSFSI (University of Pretoria) that fine-tunes Gemma models (Gemma-2-2B / Gemma-4-E2B) for South African languages. The suite has to benchmark our own checkpoints against reference models, and later against models released by other teams, **reproducibly and without re-running work it has already done**, and then write a Markdown report of insights plus a JSON file of slide-ready data.

The design is already agreed. **`docs/EVAL_ARCHITECTURE.md` is the spec — read it end to end before you write any code.** This prompt adds the implementation detail, the invariants the code must enforce, and the build order. If the two ever conflict, stop and ask me; don't pick one quietly.

## How I want you to work

1. **Plan first.** Before writing code, read the spec and reply with: your understanding in 10 bullets or fewer, the module list with a one-line responsibility for each, anything in the spec you think is ambiguous or wrong, and questions. Wait for my go-ahead.
2. **Build in the phases below, one at a time.** At the end of each phase: tests pass, `ruff` is clean, and you give me a short summary (what exists, what's stubbed, what you decided and why). Then stop and wait.
3. **Never make up dataset facts.** HuggingFace config names, split names, column names, label sets and licences have to come from the dataset itself (load it, or call the `datasets-server` API: `/splits`, `/first-rows`). If you can't verify something, leave a `# TODO(verify):` with what's missing and tell me. Don't guess.
4. **Don't start expensive jobs without asking.** No full-benchmark GPU runs and no multi-GB checkpoint downloads unless I say so. Development and tests use a mock backend and a tiny random model.
5. **Small, reviewable commits**, one logical change each, with a clear message.
6. Python 3.11+, type hints throughout, Pydantic v2 for config, pytest for tests, `ruff` for lint and format. Keep dependencies light and pin them in `pyproject.toml`.

## Non-negotiable invariants (each one gets a test)

These are what separate this suite from "a script that prints numbers". Every one needs at least one test that fails if it's violated.

1. **Base is required; the plain baseline is optional.** `models.yaml` must contain exactly one `role: base`. `plain_baseline` is zero or one entry. If it's missing, everything still runs, comparisons go against base only, and every "beats our own recipe" statement in the report is labelled **"not yet verified: no plain baseline"**. The code must never quietly treat base as if it were the plain baseline.
2. **Cached results are reused.** A (model, benchmark) pair whose fingerprint matches a `manifest.json` entry with `status: complete` is loaded from disk and never re-inferred. This matters most for `base` and `plain_baseline`: adding a new target checkpoint costs only that checkpoint's compute.
3. **Reruns only happen for a reason.** Something runs only if (a) there's no cache entry, (b) the fingerprint changed, or (c) the pair is listed under `force_rerun` in `run.yaml`, which requires a non-empty `reason`, or `force_rerun_all: true`. The reason is logged in the manifest and shown in the report's changelog.
4. **Training loss never selects a model.** Nothing under `eval/` or `report/` imports `wandb` or reads training-loss or learning-curve fields. Enforce this with a test that statically scans those packages (AST or grep) for `wandb` imports and for identifiers like `train_loss`, `eval_loss` or `loss_curve`. Ranking and "best checkpoint" logic may only read benchmark metrics.
5. **Scores are stored per language, never pre-averaged.** The store holds one record per (model, benchmark, language). Macro-averages are computed at report time, with every language weighted equally, and always shown next to the per-language table, never on their own.
6. **Chance level travels with every score.** Every classification record stores `chance_level`. A result within `near_chance_margin` of chance is shown but flagged and **excluded from ranking and "winner" statements**.
7. **Contamination status is first-class.** Every record carries `bucket` (`seen_task` / `held_out` / `retention`) and `decontamination_status` (`verified` / `unverified` / `not_applicable`). Unverified results appear in the report under a visible warning and are never used in headline claims.
8. **The report and slide data can't disagree.** Both are rendered from the same in-memory dataframes built by `report/tables.py`. Test that every number in `slide_data.json` appears in the Markdown report.

## Fingerprint and caching details

The fingerprint is a SHA-256 over a canonical JSON of:

- **Checkpoint identity.** Don't hash multi-GB weights on every run. For `hf://` checkpoints, use the resolved commit SHA of the requested revision. For local or `s3://` paths, hash `config.json` and the safetensors index, plus each weight file's name, size and ETag or mtime. Cache that identity hash per path so it's cheap to recompute.
- Benchmark id plus pinned version/revision.
- Prompt template id plus a hash of the template text.
- Shot count, decoding settings and max new tokens.
- `harness_version` (read from the package version, not typed by hand).

Manifest key: `(model_id, checkpoint_hash, benchmark_id, benchmark_version, config_fingerprint)` → record path, status (`running` / `complete` / `failed`), timestamps and the rerun reason if there was one. Write the manifest atomically (write a temp file, then rename) and mark entries `running` before inference starts, so a crashed run doesn't leave a fake cache hit behind. Per-example predictions go to `results/runs/<run_id>/<benchmark>/<lang>/predictions.parquet`.

The CLI gets a `plan` command that prints, for every (model, benchmark) pair, whether it will be **CACHED**, **RUN (new)**, **RUN (fingerprint changed: <which field>)** or **RUN (forced: <reason>)**, and runs nothing. I'll use it before every real run.

## Pipeline (what `afrigemma-eval run` does)

1. Load and validate `models.yaml`, `benchmarks.yaml` and `run.yaml` (Pydantic). Fail fast with readable errors.
2. Resolve the roles: base required, plain baseline optional, targets, externals.
3. Fingerprint every (model, benchmark) pair and build the run plan against the manifest.
4. For each pair that needs to run: run the matching runner, score it per language, and write records and predictions.
5. Load the cached records for everything else.
6. Merge them into one results table, tagged by role.
7. Build the tables, run the insight rules and build the confusion matrices.
8. Render `reports/<date>_<run_id>_report.md` and `reports/<date>_<run_id>_slide_data.json`.

## Benchmarks: what each runner needs

Verify every config, split, column and label name against the dataset before you use it. Use **lm-evaluation-harness** task definitions where a benchmark is already supported there and the language configs match. Otherwise write a small custom runner behind the same interface. Our languages are `nso, sot, tsn, xho, zul, tso, ssw, afr` (plus `ven` and `nbl` where a dataset has them). Build a single language-code normaliser (for example `swa`/`swh`, and `nso` vs `nso_Latn`) so no language gets counted twice.

| id | Type | Scoring | Notes |
|---|---|---|---|
| `afrixnli` | classification, 3 labels | accuracy, chance 1/3 | masakhane/afrixnli |
| `afrimmlu` | classification, 4 options | accuracy, chance 0.25 | masakhane/afrimmlu |
| `afrimgsm` | generation | exact match on the extracted final number | masakhane/afrimgsm; log how many outputs couldn't be parsed |
| `sib200` | classification, 7 topics | accuracy + macro-F1, chance 1/7 | Davlan/sib200 |
| `belebele` | classification, 4 options | accuracy, chance 0.25 | facebook/belebele |
| `injongo` | intent classification + slot filling | intent accuracy, slot F1 | masakhane/InjongoIntent |
| `flores_plus` | generation (eng→xx) | chrF++ (sacrebleu, `word_order=2`); COMET optional behind a flag | openlanguagedata/flores_plus, **gated**: read the HF token from an env var and fail with a clear message if it's missing |
| `afriscience_mt` | generation (eng→xx) | chrF++ | dsfsi/afriscience_mt, `corpus` config; filter rows on `lang_pair` (e.g. `eng-nso`, `eng-zul`). Only nso and zul among our languages |
| `afrimcqa` | multimodal MCQ | accuracy, chance 0.25 | Atnafu/Afri-MCQA. **Scaffold only, disabled by default.** It needs image (and audio) input that our text models don't take. Register it with `enabled: false` and a text-only-ablation TODO. CC-BY-NC-4.0 |
| `english_mmlu` | classification | accuracy, chance 0.25 | cais/mmlu, retention bucket |
| `english_perplexity` | perplexity | token-level PPL | held-out English text, retention bucket; path configurable |

For multiple-choice benchmarks, score with **log-likelihood over the options** by default (lm-eval style). Offer generate-then-parse as a config option, and when it's used, report the unparseable count; those count as wrong. Every record also stores a 95% bootstrap confidence interval computed from the per-example predictions (1,000 resamples by default, fixed seed).

## Report requirements

Follow the spec's section order: header, insights, headline table, per-language tables, confusion matrices, English retention, data-quality and contamination notes, appendix.

- **Insights come first, as bullets ordered by severity**, using the spec's eight ordered rules. Each rule is a small pure function `(results_df, run_config) -> list[Insight]`, where `Insight` has `severity`, `rule_id`, `text` and `evidence` (the table rows behind it). Every insight bullet cites its numbers.
- **The headline table** shows each capability's macro-average for every model, with a Δ vs base and a Δ vs plain baseline (or "n/a: no plain baseline").
- **Per-language tables** have one table per benchmark, show the chance level, mark near-chance cells, and show CIs as `score ± half-width`.
- **Confusion matrices** are computed for every classification run (sklearn) and saved as Markdown tables in the run folder. They're **embedded in the report only when a trigger fires**: one class taking more than X% of predictions, an answer-position bias (e.g. option A picked more than Y%), or a language whose confusion pattern is an outlier. Otherwise the appendix links to them. The thresholds X/Y live in `run.yaml`.
- **The appendix** holds the full resolved config snapshot, so the run can be reproduced from the report alone.
- **`slide_data.json`** holds headline rows (model × capability with both deltas), per-benchmark per-language rows sorted by delta, and trade-off pairs as x/y. Data only, no images.

These thresholds are **still open decisions**. Put them in `run.yaml` with the defaults below and a `# TODO: team to agree` comment: `noise_threshold: 1.0` (points), `retention_tolerance` (leave `null`; when it's unset the report says "tolerance not set" instead of passing or failing anything), `near_chance_margin: 0.02`, and the confusion triggers `class_collapse_pct: 0.6` and `position_bias_pct: 0.5`.

## Decontamination (keep it simple for v1)

`eval/decontamination/check.py` takes a path to the SFT training data (JSONL/parquet, configurable text fields) and a benchmark's test split, runs a normalised 13-gram overlap check per language, and writes a status file per (benchmark, training-data version). Records pick up `verified` or `unverified` from that file. The training-data version is an explicit field in `run.yaml` (`sft_data_version`). When it changes, affected records go `unverified` until the check is re-run. It **isn't** part of the model fingerprint, so this alone doesn't force re-inference.

## Build phases

- **Phase 0: Plan.** Read the spec, reply with the plan and questions, and stop.
- **Phase 1: Skeleton and config.** Set up the repo layout from the spec, `pyproject.toml`, and the Pydantic schemas for the three YAML files with the role rules (invariant 1). Include example configs matching the spec. Write the tests for config validation.
- **Phase 2: Fingerprint, manifest and the `plan` command.** Build the checkpoint-identity hashing, fingerprinting, atomic manifest and the rerun-decision logic. Test all four plan outcomes, `force_rerun` without a reason being rejected, and a crashed `running` entry not counting as a cache hit.
- **Phase 3: Model backend and runners.** Define a `ModelBackend` interface (`loglikelihood`, `generate`, `perplexity`) with a `MockBackend` (deterministic, scriptable) and an HF Transformers backend (vLLM optional behind an extra). Build the classification, generation and perplexity runners. Wire up the benchmarks, verifying each one's configs. Test on the mock backend plus one smoke test using `hf-internal-testing/tiny-random-GemmaForCausalLM` (or whatever tiny Gemma-family model exists; verify it) on a handful of examples.
- **Phase 4: Scoring and storage.** Build accuracy, macro-F1, exact-match with parsing, chrF++, slot F1, perplexity, bootstrap CIs, the results store (parquet) and confusion matrices. Test each metric against hand-computed values.
- **Phase 5: Decontamination.** Build the n-gram check, status files and propagation into records. Test with a synthetic leaked example.
- **Phase 6: Tables, insights and report.** Build `tables.py`, the eight insight rules, the Jinja2 template, `render_md.py` and `slide_export.py`. Add golden-file tests on a fixture results table, the plain-baseline-missing variant, and the report-vs-slide-data consistency test (invariant 8).
- **Phase 7: CLI and end-to-end.** `afrigemma-eval plan|run|report --config configs/run.yaml`, plus `--force-rerun model:benchmark --reason "..."`. Add an end-to-end test on the mock backend that **runs twice and asserts the second run performs zero inferences**, then adds a new target and asserts only that target runs. Then run the invariant-4 lint test.
- **Phase 8: Docs.** A README with quickstart, how to add a model, how to add a benchmark and how to force a rerun, plus a `CLAUDE.md` at the repo root that restates the eight invariants and the "never make up dataset facts" rule, so future agent sessions keep to them.

## Definition of done

- `afrigemma-eval plan` on a fresh repo shows everything as RUN (new). After a run, it shows everything as CACHED.
- Deleting the `plain_baseline` entry from `models.yaml` still gives a complete report, with the "not yet verified" labelling.
- Adding one target model re-runs only that model.
- The report opens with ordered insight bullets, and every number in it can be traced to a stored record.
- All eight invariants have passing tests, and `pytest` plus `ruff check` are clean.
- Every dataset detail in the code was verified, or is marked `TODO(verify)` and listed in your final summary.

Start with Phase 0.