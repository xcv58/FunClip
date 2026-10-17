# Numeric protection reference cases

These are verbatim ASR evidence copies from two local FIRE audit clips, with
the original payload hashes pinned in `manifest.json`. They are recognition
evidence, not corrected/final subtitles. All final SRT references stay unchanged.

- `fire-110-136`: native punctuation separates 百分之八十 from the standalone 三.
- `fire-319-344`: native punctuation separates 两百万 from 美元. The recognized
  一万五百美元 remains an unresolved acoustic error; this fixture must not rewrite it.
- Eleven synthetic translation controls reproduce the prior audit's eight
  valid equivalent expressions and three invalid numeric changes. The stored
  `actual_accept` fields record the prior guard's outcome, not the expected result.

Run the offline fixture replay:

```sh
funclip-env/bin/python scripts/eval_numeric_spans.py --output /private/tmp/numeric-replay
```

For the existing local audit, add `--audit-root` and optionally `--live-audio`.
The latter uses the existing ASR/VAD/punctuation weights and the two local WAV
derivatives. No LLM request or reference modification is performed. Review
diagnostics refer to original ASR sentence IDs. Passing these selected cases
does not establish a full-transcript accuracy improvement.
