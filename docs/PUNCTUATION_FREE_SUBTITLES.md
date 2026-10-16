# Punctuation-free subtitle output

Final subtitle body text uses ordinary spaces instead of punctuation. The shared
local formatter in `funclip/subtitle_style.py` enforces `punctuation-free-v1` for
transcription, correction, Traditional Chinese SRT conversion, English translation
and caption rendering. It does not rely on a model obeying a style instruction.

## Body text and meaning

Formatting keeps existing body line counts, trims line edges and collapses
separator spaces. Ordinary punctuation becomes a space. Supported numeric forms
and unambiguous English contractions receive meaning-preserving spellings.

| Input body text | Formatted body text |
| --- | --- |
| 你好，七美！ | 你好 七美 |
| 降低-3.5% | 降低负百分之3点5 |
| 價格是1,234.5 | 價格是1234點5 |
| I'm ready, don't worry. | I am ready do not worry |
| Down -3.5 % | Down negative 3 point 5 percent |
| U.S. market | US market |

The formatter does not guess at ambiguous possessives or contractions such as
`Jenny's` or `he's`, URLs, email addresses, currency symbols, dates, times, ranges,
fractions, formulas or punctuated identifiers. Those expressions require a
worded rendition or review. Unsupported symbols, controls and punctuation-only
lines also require review. No empty cue or blank body line is published as a
substitute. English line wrapping and the existing two-line, 42-character limit
are enforced after number/contraction expansion.

These checks preserve mechanical structure and implement a conservative display
policy; they do not certify the semantic accuracy of recognition or translation.

## Structure, evidence and failures

- SRT IDs, timing punctuation, the `-->` delimiter and cue boundaries remain
  unchanged. File conversion preserves original line endings, including mixed
  CRLF/LF, and a leading BOM.
- Raw ASR text, sentences and token timestamps remain available. Segmentation
  still uses original punctuation before final body formatting. Correction uses
  original recognition punctuation when that evidence is available.
- Recognition evidence `original_srt` and its hash describe the pre-formatting
  ASR SRT. New `rendered_srt` and `rendered_srt_sha256` fields identify the
  punctuation-free transcription output. Original upload/source artifacts and
  diagnostic originals are not rewritten.
- If transcription formatting requires review, Gradio fails the operation and
  retains original recognition evidence in the JSON artifact named in the error.
  It does not label the event as no speech. Media cleanup retains its existing
  behavior. Correction and translation use their existing bounded validation
  repair attempts and publish no partial result after unresolved failure.
- Caption clipping retains original recognition state. Partial clipping of
  punctuation-sensitive numbers or contractions, or unaligned token text, fails
  for review rather than silently dropping meaningful characters.

## Validation and activation

Test runs, media experiments and provider calls are skipped at the user's
request. Static syntax and diff checks are recorded separately from runtime
acceptance. The service must be restarted to load this implementation; pushing
the source does not restart the existing launch-agent process.
