# Audio review queue and glossary controls

Review candidates never rewrite the selected ASR result. Existing final SRTs
remain unchanged. Punctuation-free subtitle formatting and meaning guards still apply.

## Glossary selection

- **None** disables glossary use.
- **Reference only** is the UI default. A selected glossary is available to
  correction and translation, without ASR hotwords or a native hotword-parser call.
- **Recognition hints with comparison** applies validated native hotwords and
  runs an additional unbiased recognition of the same audio. The hinted result
  stays selected. Both name changes and changes outside confirmed names enter review.

Legacy `/transcribe_with_glossary` clients retain their one-pass hints behavior.
Use a review endpoint to choose the mode explicitly. Reference mode reports
`hotwords_checked: false`; an empty unsupported list then does not imply support.

Profile fields distinguish Simplified/Traditional spellings, `spoken_forms`
spelling variants, distinct `spoken_aliases`, printed `menu_name`, English wording
and provenance. Correction cannot exchange a distinct spoken alias for a
canonical label. Traditional/Simplified conversion of the alias remains valid.
Unsupported decoder spellings can still serve as translation references; they
do not prove a word was spoken.

## Review behavior

Candidates cover money/percentages/negations, uncertain short speech turns,
known name confusions, unresolved numeric boundaries and hinted/unbiased changes.
Ordinary repeated acknowledgments such as 对 or 谢谢 are not removed or treated
as unclear speech solely by this queue.

Optional alternate-window checks use the same ASR object and cached weights.
They select money/percentage/negation passages with five seconds of context on
each side when available. A request gets at most six windows and 120 seconds of
additional audio. Failed and skipped checks are explicit. Agreement never
certifies accuracy, and the input may lack enough context to reproduce a known
discrepancy. Alternatives never replace primary recognition or write its native
output files. No financial calculation is used to infer a spoken amount.

Items carry original ASR sentence IDs, media-relative milliseconds, reason,
priority and evidence. Raw ASR and token timings remain verbatim. The JSON queue
and ZIP report include derived audio with 2.5 seconds of surrounding context,
bounded to 30 seconds per item. The limit is 200 items; omissions are explicit.
Download the ZIP to keep HTML and audio together.

Gradio and Streamlit expose an audio player and decisions: unresolved, keep
original or needs editing. Decisions change review metadata only, never SRT text
or `content_verified`. Bundle IDs scope decisions to a request even for identical
recognition results. Changing the glossary/mode/context control invalidates
in-flight UI publication through the existing revision guard.

Service-generated review folders expire after an hour without use. Cleanup on
subsequent creation removes only registered process-owned derivatives; the
process cache is temporary too. Persist downloads for durable review. Source
archives and benchmark reports are outside this cleanup.

## API

Existing endpoint input lists remain unchanged. Additive endpoints:

| Endpoint | Inputs in order |
| --- | --- |
| `/transcribe_with_review` | file_path glossary_id glossary_mode review_context |
| `/submit_transcribe_and_correct_with_review` | file_path api_key model_name custom_model base_url return_traditional glossary_id glossary_mode review_context |

`glossary_mode` is `reference` or `hints`; `review_context` is a boolean. Use
`glossary_id=none` to disable the glossary. API profiles are named catalog entries.
Invalid controls fail before async media staging.

Responses add `review_queue`, `review_queue_file_path` and `review_bundle_file_path`.
Recognition evidence separately retains native `review_evidence.baseline` and
`review_evidence.context_checks` when requested. Async jobs snapshot controls and
keep the queue with transcription, including on correction failure. Processing
completion does not approve unresolved review items.

## Validation

Pinned cases in `eval/review_queue_cases` cover the 10500/10100 context conflict,
时间/时辣, place/restaurant names, Network/net worth, unsupported Malay spelling
and unresolved service speech. Audit-only profiles retain the frozen hotwords;
added confusions and distinct aliases are review annotations. They are not newly
registered production profiles or an independent vocabulary accuracy test.

```sh
funclip-env/bin/python scripts/eval_review_queue.py --output /private/tmp/review-fixtures
```

Add `--audit-root` and `--live-audio` for the existing local audit to replay 75
saved payloads and compare reference/hints on eight WAVs. The runner verifies 24
final SRTs, 73 audited SRTs and 14 model files against pinned hashes. No LLM API
or remote media upload is used. These checks validate review behavior and evidence
preservation, not overall ASR accuracy.
