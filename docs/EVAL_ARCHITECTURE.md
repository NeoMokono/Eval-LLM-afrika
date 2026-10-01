# AfriGemma Eval Framework — Architecture & Plan

Sep 30, 2026

## Design principles this framework encodes

- **Two reference points, not one.** Every comparison runs against the untrained base checkpoint (always) and your current plain SFT baseline (once it exists) — a change has to beat your own recipe, not just beat doing nothing.
- **Contamination is first-class.** Every result carries a seen-task/held-out bucket and a decontamination status; anything without a passing check is marked unverified and never quoted as final.
- **Loss never selects a winner.** Training loss stays in wandb for debugging; the code path that ranks or picks a checkpoint never reads it.
- **Chance level travels with every score**, and benchmarks that land near chance are reported but excluded from ranking decisions.
- **Per-language first, averages second.** Every table starts at the language level; a macro-average is a rollup shown alongside it, never the only number.
- **English retention rides next to every African-language number**, not as a separate afterthought check.
- **Reruns are explicit.** An identical (model, benchmark, config) triple is served from cache; a rerun happens only when something in that triple actually changed, or you ask for it by name.

## Reference models: two-point comparison with base-only fallback

Every run compares up to two reference points, plus however many models you're actually evaluating.

**`base`** — the untrained checkpoint (Gemma-2-2B or Gemma-4-E2B, matching whatever you're fine-tuning). Required; every run compares against it.

**`plain_baseline`** — your current default SFT recipe (uniform language sampling, no special interventions). Optional. When it's in the config, every ablation gets scored against it too. When it's missing, the pipeline runs off `base` alone, and the report marks any "beats our own recipe" claim as not yet verified instead of silently comparing against base only.

**`target`** — the checkpoint(s) you're actually evaluating this run: one entry per scaling-ladder rung (50M / 150M / 450M tokens), or any ablation checkpoint.

**`external`** — released models from other teams (AfriqueGemma-12B, open multilingual models) once you want to benchmark against them. Same pipeline, same benchmarks, tagged separately so the report can group "ours" vs "competitive" on its own.

```yaml
reference_models:
  - id: gemma-4-e2b-base
    role: base
    checkpoint: hf://google/gemma-4-e2b
  - id: afrigemma-plain-baseline
    role: plain_baseline        # delete this entry if it doesn't exist yet
    checkpoint: s3://afrigemma/plain-baseline/checkpoint-final

target_models:
  - id: afrigemma-rung2-150M
    role: target
    checkpoint: s3://afrigemma/rung2/checkpoint-final

external_models:
  - id: afriquegemma-12b
    role: external
    checkpoint: hf://some-org/afriquegemma-12b
```

Deleting the `plain_baseline` entry is the intended way to say "we don't have one yet" — nothing else in the config needs to change.

## Repository layout

```text
afrigemma-eval/
  configs/
    models.yaml              # reference + target + external model registry
    benchmarks.yaml          # benchmark definitions: version, languages, metric, chance level
    run.yaml                 # which models x benchmarks to run this invocation, rerun overrides
  eval/
    registry.py              # loads and validates the three config files
    fingerprint.py           # builds the cache key for a (model, benchmark) run
    manifest.py              # reads/writes results/manifest.json
    runners/
      classification_runner.py   # AfriXNLI, AfriMMLU, SIB-200, Belebele, INJONGO-intent
      generation_runner.py       # AfriMGSM, FLORES+, AfriScience MT
      perplexity_runner.py       # English perplexity (retention)
    decontamination/
      check.py               # seen-task / held-out bucketing, overlap check against training data
    scoring/
      accuracy.py
      chrf.py
      slot_f1.py              # INJONGO slot filling
      confusion.py            # confusion matrix / answer-position-bias builder
    storage/
      results_store.py        # reads/writes per-example + aggregate result records
  report/
    insights.py               # rule-based insight generator, ordered by severity
    tables.py                 # per-language + rollup comparison tables
    render_md.py              # renders reports/<run_id>_report.md from a template
    slide_export.py           # writes reports/<run_id>_slide_data.json
    templates/
      report.md.j2
  results/
    runs/<run_id>/...         # raw per-example predictions, one folder per completed run
    manifest.json             # index: (model, benchmark, fingerprint) -> result path + status
  reports/
    <date>_<run_id>_report.md
    <date>_<run_id>_slide_data.json
  cli.py                       # afrigemma-eval run --config configs/run.yaml
```

## Configuration schema

Three config files drive a run.

**`benchmarks.yaml`** — one entry per benchmark, matching what's documented on the Evaluation Framework and Benchmarking protocol pages:

```yaml
benchmarks:
  - id: afrixnli
    version: masakhane/afrixnli@main
    type: classification
    languages: [sot, xho, zul]
    metric: accuracy
    chance_level: 0.33
    bucket: held_out          # or seen_task — set per your current SFT data mix
  - id: afrimmlu
    version: masakhane/afrimmlu@main
    type: classification
    languages: [sot, xho, zul]
    metric: accuracy
    chance_level: 0.25
    bucket: held_out
  - id: afrimgsm
    version: masakhane/afrimgsm@main
    type: generation
    languages: [xho, zul]
    metric: accuracy
    chance_level: 0.0
    bucket: held_out
  - id: flores_plus
    version: openlanguagedata/flores_plus@devtest
    type: generation
    languages: [nso, sot, tsn, xho, zul, tso, ssw, afr]
    metric: chrf++
    bucket: held_out
  - id: english_mmlu
    version: cais/mmlu@main
    type: classification
    languages: [eng]
    metric: accuracy
    chance_level: 0.25
    bucket: retention
  - id: english_perplexity
    version: internal/english-heldout@v1
    type: perplexity
    languages: [eng]
    metric: perplexity
    bucket: retention
```

**`run.yaml`** — which models × benchmarks to actually run this invocation, plus rerun control:

```yaml
run_id: 2026-09-30-rung2-vs-baseline
prompting:
  shots: [0, 5]
  decoding: greedy
harness_version: 0.3.1
models: [gemma-4-e2b-base, afrigemma-plain-baseline, afrigemma-rung2-150M]
benchmarks: [afrixnli, afrimmlu, afrimgsm, sib200, english_mmlu, english_perplexity]
force_rerun_all: false
force_rerun:                     # optional, explicit overrides
  - model: afrigemma-rung2-150M
    benchmark: afrimmlu
    reason: "re-ran decontamination on the SFT mix, needs fresh numbers"
```

The `force_rerun` list is the manual override: name a specific (model, benchmark) pair and it reruns regardless of cache, with a required `reason` string logged into the manifest and shown in the report's changelog.

## Caching and rerun logic

Every (model, benchmark) pair gets a fingerprint before anything runs: a hash of everything that could change the result. Match an existing entry in `results/manifest.json` and the pipeline skips straight to loading the cached scores; no match, or an explicit `force_rerun`, and it queues the run.

**What goes into the fingerprint:**

| Component | Why it's in there |
| --- | --- |
| Model checkpoint id/hash | a new checkpoint is obviously a new result |
| Benchmark id + version | a benchmark update (new split, fixed labels) invalidates old scores |
| Prompt template version | wording changes can move scores by several points |
| Few-shot count + decoding settings | zero-shot and 5-shot aren't comparable, greedy vs sampled aren't either |
| Harness version | a scoring-logic change should trigger a rerun even if nothing else did |

**Three ways a run gets triggered:**

- No cache entry exists for that fingerprint (first time seeing this model/benchmark/config combination).
- The fingerprint changed (any of the five inputs above changed) — automatic, nothing to configure.
- It's listed under `force_rerun` in `run.yaml` — for cases the fingerprint can't see, like a training-data change that doesn't touch the checkpoint id, or you just want fresh numbers to double-check something.

Anything not triggered by one of these three is served from cache, including your reference models. Once `gemma-4-e2b-base` and `afrigemma-plain-baseline` have results on file, adding a third target checkpoint to `run.yaml` costs you exactly the compute for that one new model — the two reference rows come back from `results/manifest.json` untouched.

## Pipeline: end-to-end run flow

&#91;embedded content: eval pipeline — 8 stages, 1 cache-hit branch\]

A fingerprint covers the model checkpoint, benchmark version, prompt template, shot count, decoding settings and harness version; an unchanged fingerprint with a manifest entry skips straight to the merge step, so re-running a report after adding one new target checkpoint only pays for that checkpoint's benchmarks.

## Results storage schema

Each completed (model, benchmark, language) triple is one record, stored once in `results/runs/<run_id>/` and indexed in `results/manifest.json`.

| Field | Example | Purpose |
| --- | --- | --- |
| `run_id` | `2026-09-30-rung2-vs-baseline` | groups everything produced by one invocation |
| `model_id`, `model_role` | `afrigemma-rung2-150M`, `target` | which model, and whether it's base / plain\_baseline / target / external |
| `checkpoint_hash` | sha256 of the checkpoint files | detects a silently-changed checkpoint behind the same id |
| `benchmark_id`, `benchmark_version` | `afrimmlu`, `masakhane/afrimmlu@main` | which benchmark and which snapshot of it |
| `language` | `zul` | per-language, never pre-averaged |
| `config_fingerprint` | hash of prompt template + shots + decoding + harness version | the cache key from Caching and rerun logic |
| `bucket` | `held_out` | seen-task vs held-out, from `benchmarks.yaml` |
| `decontamination_status` | `verified` / `unverified` / `not_applicable` | whether the overlap check passed for this training-data version |
| `metric_name`, `metric_value`, `n_examples`, `chance_level` | `accuracy`, `0.341`, `500`, `0.25` | the score itself, with its denominator and baseline |
| `seed` | `0` | for when repeat-seed runs get added |
| `raw_predictions_path` | `runs/<run_id>/afrimmlu/zul/predictions.parquet` | per-example predictions, used for confusion matrices and error review |
| `completed_at`, `harness_version` | timestamp, `0.3.1` | provenance |

The manifest itself is just a lookup: `(model_id, checkpoint_hash, benchmark_id, benchmark_version, config_fingerprint) → record path`. Anything not in this schema (loss curves, GPU hours, wandb run links) stays in wandb — keeping it out of the eval store is part of what makes the loss/selection separation enforceable.

## Report generator: the Markdown document

`render_md.py` produces one `reports/<run_id>_report.md` per invocation, built in this order:

| Section | Content | Built from |
| --- | --- | --- |
| Header | run id, date, models compared, which reference points were available | `run.yaml` + manifest |
| Insights | ordered bullets, most severe first | `insights.py` |
| Headline comparison table | per-capability rollup vs base and vs plain baseline (when present) | `tables.py` |
| Per-language tables | one table per benchmark, every language, chance level shown | `tables.py` |
| Confusion matrices | only for benchmarks/models where a check flags a systematic pattern | `scoring/confusion.py` |
| English retention | English MMLU + English perplexity, before/after | `tables.py` |
| Data-quality and contamination notes | unverified results, near-chance flags, anything skipped from cache | `insights.py` |
| Appendix | full config used (models, benchmarks, prompting), so the run is reproducible from the doc alone | `run.yaml` snapshot |

Confusion matrices are the one section that's conditional: they're generated for every classification benchmark run (AfriXNLI, AfriMMLU, SIB-200, Belebele, INJONGO intent), but only embedded in the report when `insights.py` flags something worth showing — a class collapse, a position bias (the model always picking option A), or a language scoring far off the rest. Otherwise they're written to the run folder and linked from the appendix instead of padding out the document.

## Insight-generation rules

`insights.py` runs a fixed, ordered set of checks over the aggregated results and turns whatever fires into a bullet. Bullets render grouped by severity, most important first, so the reader doesn't have to hunt through per-language detail for the one regression that matters.

1. **Contamination and data integrity.** Any result with `decontamination_status: unverified`, or a benchmark bucket that changed since the last run without a recheck.
2. **Regressions past the noise threshold.** Any capability where the target scores worse than the reference point it should beat (plain baseline if present, else base) by more than the configured threshold (default \~1 point, per benchmark).
3. **Headline capability results.** The per-capability deltas vs both reference points — this is the section that answers "did the change work."
4. **English retention.** English MMLU and English perplexity, flagged if the drop exceeds the tolerance set before the run.
5. **Near-chance flags.** Any benchmark scoring within a set margin of its chance level, named as excluded from ranking rather than silently dropped.
6. **Per-language outliers.** The best- and worst-performing language per benchmark, especially where a language moves opposite to the average.
7. **Trade-offs.** Paired findings where one capability improved at another's expense (translation up, knowledge down, in the style of the DSFSI draft's headline finding).
8. **Data-quality notes.** Anything caught during scoring that isn't a model-quality issue — unparsable outputs, duplicate answer options, translation artifacts spotted in the examples.

Each rule is a small function that takes the results table and returns zero or more insight objects; adding a new check later means adding one function, not rewriting the report.

## Slide-data export

Alongside the Markdown report, `slide_export.py` writes `reports/<run_id>_slide_data.json`: the same numbers the report used, reshaped for a deck instead of a document.

- **Headline slide data:** one row per model × capability, with both reference-point deltas, ready to drop into a bar or dot chart.
- **Per-language slide data:** one table per benchmark, already sorted by delta, so a "biggest movers" slide doesn't need rebuilding from the full per-language table.
- **Trade-off slide data:** the paired capability comparisons insight rule 7 generates, as x/y pairs for a scatter or two-axis chart.
- **Chart-ready, not chart-made:** this file holds rows and labels, not rendered images, so whoever builds the deck chooses the chart type for their audience instead of inheriting one baked into a PNG.

This is a straight JSON export from the same dataframes `tables.py` builds for the Markdown report, so the two outputs can't drift apart from the same run.

## Tech stack recommendation

| Layer | Suggested tool | Why |
| --- | --- | --- |
| Model inference | vLLM or Transformers, whichever your training stack already uses | keeps one dependency set across training and eval |
| Task definitions | lm-evaluation-harness task configs where a benchmark is already supported, custom runners for AfriMCQA/AfriScience MT/INJONGO if not | avoids rebuilding what's already maintained upstream |
| Config | Pydantic models over the YAML files | catches a malformed config before a run starts, not after |
| Results store | Parquet files + a small manifest.json (or SQLite if the manifest grows past a few thousand rows) | fast to query, no server to run |
| Confusion matrices | scikit-learn's `confusion_matrix` + a small renderer | standard, well-tested, nothing custom to maintain |
| Report templating | Jinja2 → Markdown | keeps `render_md.py` a template plus a data dict, easy to restyle later |
| CLI | Typer or argparse | `afrigemma-eval run --config configs/run.yaml`, `--force-rerun model:benchmark` |

None of this is load-bearing — the caching, reference-point and insight logic above works the same regardless of which libraries fill these slots. Pick what's least new to the team.

## Open decisions before implementation

- [ ] Confirm the noise/CI threshold used in insight rule 2 (currently proposed at \~1 point; needs agreement, per the Evaluation Framework page).
- ~1 point is okay
- [ ] Decide whether AfriMCQA and AfriScience MT get added to `benchmarks.yaml` for this first version, or come later.
- they should be added
- [ ] Settle the English-retention tolerance (rule 4) — how much drop is acceptable before it's flagged.
- if it drops by 3 points
- [ ] Confirm licences for SIB-200, AfriMMLU and INJONGO before any result leaves the team (flagged as inconsistent across sources on the benchmark subpages).
- [ ] Decide who signs off on `plain_baseline` before it's added to `models.yaml` — once it exists, it becomes the default comparison point for every future ablation.
- I sign off the plain_baseline
- [ ] Pick the confusion-matrix trigger thresholds — what counts as "a systematic pattern worth showing."
- at your discretion
