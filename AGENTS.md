# Repository instructions

## Subtitle text style

Final displayed and exported subtitle body text must not contain punctuation
marks 标点符号. This applies to Simplified Chinese, Traditional Chinese and
English subtitles produced by this repository.

- When a separator is needed, use one ordinary space U+0020. Collapse repeated
  separator spaces within each text line and trim spaces at line edges. Preserve
  required text line breaks and cue boundaries.
- Do not add commas, periods, question marks, exclamation marks, colons,
  semicolons, quotation marks, brackets, dashes, ellipses or decorative separators
  to final subtitle text. Use spaces or punctuation-free wording instead.
- Preserve meaning. Never blindly delete punctuation inside numbers, negative
  values, percentages, contractions, names, URLs or other meaningful expressions.
  Use an unambiguous punctuation-free equivalent where possible, such as 三点五
  for 3.5 or do not for don't. Flag expressions that cannot be safely formatted
  without changing their meaning for review.
- Apply the display style locally after timestamp-based segmentation and AI
  correction validation. Internal punctuation may still inform phrase boundaries.
  A prompt instruction alone does not enforce the final output style.
- Retain original ASR text, uploaded source files and recognition evidence
  verbatim. Do not strip punctuation from these records or rewrite historical
  artifacts to match the display style.
- Preserve SRT syntax exactly, including cue IDs, timestamp punctuation, the
  --> delimiter and blank lines between cues. Do not create empty subtitle cues
  or blank text lines by removing punctuation-only content; flag these for review.

This rule concerns subtitle body text. It does not prohibit punctuation required
by source code, JSON, API contracts, configuration, documentation or diagnostic
records.
