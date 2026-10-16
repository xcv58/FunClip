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
- Slice the original text, preserving punctuation and whitespace. Opening quote
  and bracket marks stay with the following text. Each resulting cue starts at
  its first token's ASR start and ends at its last token's ASR end.
- Split only Chinese string sentences whose token count matches their timestamp
  count and whose timestamp pairs are finite, positive-duration, nonnegative,
  ordered and non-overlapping. Keep an unaligned sentence intact; do not
  interpolate word times. Non-string sentences retain the existing rendering.

`VideoClipper.recog` enables this policy for Chinese in the shared transcription
path used by Gradio, its API and Streamlit. English recognition retains its
existing rendering. Correction of an uploaded SRT retains the supplied cues;
it does not have ASR word timings from which to derive new boundaries.

## Evidence and correction

The splitter creates temporary render copies. ASR text, raw text, complete token
timestamps, sentence data and speaker clipping state remain intact. In the API
evidence, `sentence_info` therefore remains the original ASR sentence list, while
`original_srt` is the newly rendered, pre-correction SRT. Their counts may differ.
The existing SRT hash covers that rendered SRT.

Correction still changes only cue text and preserves the IDs, timing lines and
text-line counts of the rendered SRT. Translation and chapters receive the same
SRT through the existing pipeline. The splitter does not correct recognition
mistakes in names or words.

## Validation and activation

Static Python syntax parsing and whitespace/diff checks passed. Test runs and
media/API experiments were skipped at the user's request; runtime subtitle
quality has not been evaluated for this change.

The running launch-agent process needs a restart to load this source change.
This change does not restart or redeploy the service automatically.
