# Chinese subtitle segmentation

Chinese transcription now renders long ASR sentences as shorter SRT cues before
AI correction. This is a selective adaptation of the timestamp-based splitter in
[upstream FunClip](https://github.com/modelscope/FunClip/blob/36acf72e2add3c4af33fd35d373574a5dee96482/funclip/videoclipper.py#L26-L111).
It introduces no dependency, recognition-model or endpoint changes.

## Rendering policy

- Target at most 8 seconds and 30 tokens per cue. A token is one Chinese character
  or one non-Chinese word. A single token longer than 8 seconds retains its actual
  timing rather than receiving invented intermediate timestamps.
- Prefer sentence punctuation, then phrase punctuation, then pauses of at least
  300 ms. Search the latter half of the available cue to avoid tiny fragments;
  fall back to a token boundary when no natural break is available. This is a
  deterministic heuristic, not linguistic word segmentation.
- Slice the original text internally, preserving punctuation and whitespace. Opening quote
  and bracket marks stay with the following text. Each resulting cue starts at
  its first token's ASR start and ends at its last token's ASR end.
- Split only Chinese string sentences whose token count matches their timestamp
  count and whose timestamp pairs are finite, positive-duration, nonnegative,
  ordered and non-overlapping. Keep an unaligned sentence intact; do not
  interpolate word times. Non-string sentences retain the existing rendering.
- Never split a recognized numeric phrase internally, including its sign, ratio
  prefix, decimal digits, fraction, scale or unit. Prefer a preceding safe boundary;
  if the numeric phrase itself exceeds a target, keep its real timing intact.
- Before long-cue splitting, repair two narrowly supported native sentence
  boundaries: a ratio ending in 十 followed by a standalone final Chinese digit
  (百分之八十 / 三), and an amount followed by an explicit currency name
  (两百万 / 美元). Move only that digit/unit prefix into a rendering copy.
  Require matching raw tokens, valid aligned timing, the same speaker and a gap
  of 0–120 ms. Separate counts, ambiguous digit lists, missing alignment, speaker
  changes and longer pauses stay unchanged and receive review diagnostics.

`VideoClipper.recog` enables this policy for Chinese in the shared transcription
path used by Gradio, its API and Streamlit. English recognition retains its
existing rendering. Correction of an uploaded SRT retains the supplied cues;
it does not have ASR word timings from which to derive new boundaries.

## Evidence and correction

The splitter creates temporary render copies. ASR text, raw text, complete token
timestamps, sentence data and speaker clipping state remain intact. In the API
evidence, `sentence_info` therefore remains the original ASR sentence list, while
`original_srt` is the segmented ASR SRT before display formatting. Their counts
may differ. Its hash covers that diagnostic SRT. `rendered_srt` and its hash
identify the punctuation-free transcription output.

Chinese recognition also provides `correction_source_srt` and its hash. This
punctuated SRT uses the protected rendering layout; `original_srt` and the native
sentence list remain available independently. The `numeric_span_report` records
adjustments and unresolved boundaries using **original ASR sentence IDs**, not
final rendered cue IDs. Its `content_verified` is always false: restored layout
does not establish the correctness of a recognized value.

Correction still changes only cue text and preserves the IDs, timing lines and
text-line counts of the rendered SRT. Translation and chapters receive the same
SRT through the existing pipeline. The splitter does not correct recognition
mistakes in names or values.

The subsequent [punctuation-free display policy](PUNCTUATION_FREE_SUBTITLES.md)
formats final cue bodies locally after segmentation and correction validation.
Original punctuation continues to inform splitting and remains in ASR evidence.

## Validation and activation

The numeric protection delivery has pinned local ASR fixtures and meaning
controls in `eval/numeric_span_cases`. Run the deterministic audit with
`funclip-env/bin/python scripts/eval_numeric_spans.py --output /private/tmp/numeric-replay`.
With an existing local audit, add `--audit-root <audit-directory> --live-audio`
to replay its saved payloads and run the two existing WAV derivatives through
the unchanged cached ASR/VAD/punctuation models. No LLM API is called.

The running launch-agent process needs a restart to load this source change.
This change does not restart or redeploy the service automatically.
