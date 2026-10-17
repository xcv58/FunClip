# FunClip subtitle pipeline reference cases

This suite evaluates local quality diagnostics, contextual Traditional Chinese
conversion and conservative meaning-preservation guards. Recognition models
and the provider response schemas remain unchanged.

`manifest.json` pins nine final subtitles, three raw ASR files and two historical
intermediate files by SHA256. Paths are relative to a separately supplied archive
root. All source files are read-only references. Embedded excerpts preserve the
selected cue IDs, timestamps and body text; assembling excerpts may change only
the blank separators and final newline. Synthetic controls are labelled separately.

The 38 cases comprise:

- 14 timing/repetition cases, including long sparse captions, short responses,
  a long performance gap, English resegmentation and equivalent numeric wording
- 6 contextual Traditional Chinese conversion cases
- 15 meaning-guard cases, including the two historical price/comparison failures
- 3 unresolved audio questions excluded from correctness scoring

Run the offline replay from the repository root:

```sh
funclip-env/bin/python scripts/eval_subtitle_pipeline.py \
  --references-root '/Volumes/Crucial X10/srt' \
  --output eval/model_migration_runs/2026-10-17/meaning_safeguards/reference_results.json
```

Omit `--references-root` to run the embedded cases without the external volume.
When a root is supplied, any hash or excerpt mismatch fails evaluation. Report
output inside the archive is rejected. The runner calls neither ASR nor an LLM.

Quality reports are advisory. A clear timing/repetition report does not certify
speech content or translation accuracy. Gaps do not prove missing speech. A short
caption or repeated phrase may be legitimate; diagnostics never remove it.
Historical punctuation remains in reference excerpts. The final display formatter
continues enforcing punctuation-free output after conversion and validation.

Production API changes are additive: transcription and correction return
`quality_report`; file translation returns per-file `quality_reports`; Traditional
text translation attaches a report to each item. UI status text shows a short
review hint. No existing required arguments or return tuple sizes are changed.

## Meaning guard scope

Correction checks recognized numeric values and units, negation, approximation,
and literal Chinese comparison targets before display formatting. Translation
checks recognized numbers and explicit units per cue, catching a price moved
into another cue even when IDs and timestamps are unchanged. Supported number
normalization includes Chinese/English numeric words, decimals, percentages and
simple spoken ranges and separated countdown digits. Ordinary uses of one,
first and both can be nonnumeric. Comparison checks tolerate grammar particles
after supported comparative adjectives while protecting the literal target.

Violations use the existing bounded validation retry. If a corrected candidate
still fails, the operation fails without publishing a complete output SRT.
API metadata describes the enabled checks without claiming full content approval.
English translation prompts and repair requests ask for full words and natural
possessive rephrasing, reducing avoidable apostrophe-formatting failures. The
local punctuation-free formatter remains the final enforcement step.

These are conservative text guards. They cannot establish what unclear audio
actually says, recover facts absent from ASR, or verify general word/meaning
alignment. Vague quantities, implicit counts and legitimate paraphrases can
require human review. Historical human-edited translations may intentionally
condense wording or move content across cues; keep them intact as references,
even when their layout differs from the production cue-preserving contract.
