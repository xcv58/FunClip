# FunClip subtitle pipeline reference cases

This first delivery evaluates local quality diagnostics and contextual
Traditional Chinese conversion. It does not change recognition models or claim
that semantic correction and translation guards have been implemented.

`manifest.json` pins nine final subtitles, three raw ASR files and two historical
intermediate files by SHA256. Paths are relative to a separately supplied archive
root. All source files are read-only references. Embedded excerpts preserve the
selected cue IDs, timestamps and body text; assembling excerpts may change only
the blank separators and final newline. Synthetic controls are labelled separately.

The 25 cases comprise:

- 14 timing/repetition cases, including long sparse captions, short responses,
  a long performance gap, English resegmentation and equivalent numeric wording
- 6 contextual Traditional Chinese conversion cases
- 2 historical semantic failures catalogued for the next delivery, not scored here
- 3 unresolved audio questions excluded from correctness scoring

Run the offline replay from the repository root:

```sh
funclip-env/bin/python scripts/eval_subtitle_pipeline.py \
  --references-root '/Volumes/Crucial X10/srt' \
  --output eval/model_migration_runs/2026-10-17/first_delivery/reference_results.json
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
