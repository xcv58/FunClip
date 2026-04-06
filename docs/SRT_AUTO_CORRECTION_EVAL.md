# SRT Auto-Correction Evaluation Plan

## Goal

Compare candidate LLMs for SRT auto-correction using the kinds of subtitles FunClip actually produces, and choose the best default based on quality, safety, latency, and cost.

## What "good" looks like

For this task, a strong model should:

- Preserve valid SRT structure exactly.
- Keep the same subtitle count and timestamps.
- Fix obvious ASR mistakes with minimal edits.
- Avoid rewriting lines that are already acceptable.
- Preserve names, brands, technical terms, and code-switching when uncertain.
- Return stable results across repeated runs.

## Evaluation dataset

Build a small labeled set before benchmarking models.

- `clean_pass/`: already-good SRTs that should remain almost unchanged.
- `obvious_errors/`: clear ASR mistakes, typos, punctuation problems, spacing problems.
- `ambiguous_errors/`: cases where the model should often leave text unchanged.
- `mixed_language/`: Chinese plus English brand names, people names, product names, URLs.
- `long_context/`: longer subtitle files with repeated terminology and speaker/topic continuity.
- `edge_cases/`: malformed punctuation, empty lines, repeated indices, weird spacing, OCR-like noise.

Recommended starting size:

- 50 to 100 files total.
- 10 to 20 files per bucket.
- Include both "should change" and "should not change" examples.

For each file, keep:

- `input.srt`
- `gold.srt`
- `notes.md`

`gold.srt` should reflect the exact preferred output. `notes.md` should explain tricky choices so future reviewers stay consistent.

## Candidate models

Start with a short list:

- `gpt-4o-mini`
- `gpt-5-mini`
- one or two additional provider/model options via LiteLLM

Keep prompt text fixed while comparing models. Only change one variable at a time.

## Metrics

### Hard-gate metrics

These should fail the sample outright if broken:

- SRT parse success.
- Same segment count as input.
- Same index numbers as input.
- Same timestamps as input.

### Quality metrics

Track these per file and in aggregate:

- Exact match against `gold.srt`.
- Character edit distance to `gold.srt`.
- Precision of edits:
  proportion of model edits that match desired edits.
- Recall of edits:
  proportion of desired edits that the model actually makes.
- Over-edit rate:
  how often the model changes text that should have stayed unchanged.
- Under-edit rate:
  how often the model misses obvious corrections.

### Operational metrics

- Median latency per file.
- p95 latency.
- Input tokens.
- Output tokens.
- Estimated cost per file.
- Failure rate and retry rate.

## Human review rubric

For files that are not exact matches, review with a small rubric:

- `5`: perfect or clearly acceptable.
- `4`: minor differences, still production-safe.
- `3`: mixed quality, would need spot review.
- `2`: unsafe or over-edited.
- `1`: unusable.

Review dimensions:

- correctness
- minimal-edit discipline
- terminology preservation
- readability
- structural safety

## Recommended decision rule

Use a two-stage decision:

1. Eliminate any model that fails structural hard gates above a small threshold.
2. Among the remaining models, choose the one with the best quality/cost tradeoff.

A practical default rule:

- prefer the cheaper model if it is within 2 to 3 percentage points on exact match and clearly similar on human review.
- prefer the stronger model only if it materially reduces under-correction or over-editing on real samples.

## Suggested repo workflow

1. Add `eval/srt_correction_cases/` with the labeled fixtures.
2. Add a runner that:
   - sends each `input.srt` to a selected model
   - validates structure
   - stores raw outputs
   - computes metrics
3. Add a report script that outputs:
   - per-model summary table
   - per-file failures
   - worst disagreement cases for manual review

## Resumable benchmark runner

Use the cached runner to compare models without rerunning completed results:

```bash
venv/bin/python scripts/eval_srt_correction.py \
  --suite-name baseline_v1 \
  --models gpt-4o-mini gpt-5-mini gpt-5.4-mini \
  --repeats 1
```

Results are stored under:

- `eval/srt_correction_runs/baseline_v1/models/<model>/cases/<case_id>/repeat_01/corrected.srt`
- `eval/srt_correction_runs/baseline_v1/models/<model>/cases/<case_id>/repeat_01/result.json`
- `eval/srt_correction_runs/baseline_v1/summary.json`
- `eval/srt_correction_runs/baseline_v1/results.csv`
- `eval/srt_correction_runs/baseline_v1/report.md`

Important behavior:

- completed `model x case x repeat` results are skipped automatically on later runs
- adding a new model only evaluates the missing model folders
- rerunning the same suite name rebuilds summaries from all stored results already on disk
- use `--force` only if you explicitly want to overwrite cached results

Recommended incremental workflow:

```bash
# First run the cheaper baseline pair
venv/bin/python scripts/eval_srt_correction.py \
  --suite-name baseline_v1 \
  --models gpt-4o-mini gpt-5-mini

# Later add one more model without rerunning the completed cells
venv/bin/python scripts/eval_srt_correction.py \
  --suite-name baseline_v1 \
  --models gpt-5.4-mini
```

## Fixture builder

If you already have:

- the original audio/video file
- the final corrected SRT you want to treat as gold

you can generate the raw pre-correction SRT fixture with:

```bash
python3 scripts/build_srt_eval_case.py \
  --media /absolute/path/input.mp4 \
  --gold-srt /absolute/path/final.srt \
  --case-id sample_case_001
```

That creates:

- `eval/srt_correction_cases/sample_case_001/original_transcribed.srt`
- `eval/srt_correction_cases/sample_case_001/final_traditional.srt`
- `eval/srt_correction_cases/sample_case_001/raw_text.txt`
- `eval/srt_correction_cases/sample_case_001/meta.json`

Use `--copy-media` if you also want the source media stored with the case.

## First benchmark to run

Run this small experiment first:

- same prompt
- same dataset
- 3 repeated runs per file for each model
- compare `gpt-4o-mini` vs `gpt-5-mini`

That will tell us:

- whether `gpt-5-mini` actually improves correction quality for our data
- whether the improvement is large enough to justify higher output-token cost
- whether either model is too unstable on borderline cases

## What not to optimize for

- generic chatbot quality
- coding benchmarks
- multimodal features
- maximum context window alone

This task is narrow. Structural obedience and minimal, accurate editing matter more than broad intelligence claims.
