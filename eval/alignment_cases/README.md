# Meaning alignment controls

All 16 cases in `manifest.json` are synthetic. They test four blocking content
changes, three advisory discrepancies, seven valid translations/corrections,
one unsupported meaning change and one already-protected wrong percentage.

Two valid pronoun/paraphrase controls intentionally receive review warnings.
An unsupported meaning change intentionally escapes this bounded lexical check.
These expose false alarms and limits rather than claiming general semantic accuracy.

The tests cover the real request/parser integration and final cross-batch check:

```sh
funclip-env/bin/python -m unittest funclip.test.test_meaning_alignment funclip.test.test_subtitle_readability funclip.test.test_gradio_alignment_readability
```

For an existing local two-video manifest and frozen saved outputs:

```sh
funclip-env/bin/python scripts/eval_alignment_readability.py --manifest /absolute/path/to/manifest.json --output /absolute/path/to/benchmark-results
```

The benchmark runner uploads nothing. Add `--live-audio` for a fresh pass using
the exact cached current ASR/VAD/punctuation models and retained WAV derivatives.
