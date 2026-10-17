# Meaning alignment and Chinese readability

These checks improve pipeline validation and review. They do not change the ASR
model, certify spoken content, rewrite reference SRTs, or automatically retime cues.

## Meaning alignment

`funclip/meaning_alignment.py` provides a bounded bilingual catalog of places,
concrete objects, people and actions. Simplified/Traditional variants and supported
English synonyms are normalized locally. Country names inside currency expressions
are distinguished from locations; uppercase US is distinguished from the pronoun us.

The existing numeric, unit, negation, comparison and glossary guards remain active.
The new checks add:

- A unique supported concept lost from one cue and introduced into another blocks
  publication under the current one-source-cue-per-output-cue contract. Repeated
  concepts and ambiguous references remain advisory.
- An explicit buy/sell reversal blocks publication. Other missing concepts,
  unrecognized paraphrases and possible additions request review.
- Chinese correction also checks literal removed phrases. An unambiguous phrase
  transferred into another cue blocks publication; other substantial removals are
  advisory, allowing localized ASR typo correction.
- Validation runs within batches and again over the complete result, so a transfer
  across a batch boundary cannot bypass the check. In-batch failures use the existing
  bounded retry. A final cross-batch failure returns no complete subtitle artifact.

The model response schema remains text-only. A locally extracted
`alignment_reference` is added to the input, without accepting provider-reported
coverage or changing the configured model. Source material remains untrusted data.

Each coverage row binds the output cue to its source ID, original media interval
and source/output text hashes. `semantic_coverage_verified` and `content_verified`
remain false. This proves provenance, not completeness of meaning. Vocabulary is
bounded; novel names, general paraphrases and unsupported meanings can escape the
check. Valid pronouns and paraphrases can receive advisory false alarms.

Correction APIs add `meaning_alignment`. English translation APIs add per-file
`meaning_alignment_reports`; the UI exposes downloadable alignment JSON. Correction
diffs and statuses show review warnings. Existing API input lists are unchanged.
Same-named uploads receive distinct translated-output and alignment-report filenames.
Streamlit shows correction alignment details too. Uploaded/edited cue layouts are
preserved. Regrouping with a different source coverage map is not implemented.

## Chinese readability

`build_srt_quality_report` is now version 2 and includes a `readability` subreport.
Default diagnostic references are 9 reading units per second, 16 units per line,
and two lines. Callers can configure `chinese_max_cps` and
`chinese_max_line_units`; invalid nonpositive or nonfinite values are rejected.

Full-width/Han characters count one unit. Half-width letters, digits, spaces and
punctuation count half a unit; line breaks do not count. English-only cues are not
evaluated against Chinese speed limits. Punctuation is still removed or safely
verbalized by the existing final subtitle formatter, independently of diagnostics.

Warnings cover Chinese reading speed, line length and line count. They appear in
the audio review queue with displayed cue IDs separately identified from native
ASR sentence IDs. Raw recognition evidence is untouched. Both UIs can replay them.

Grouping proposals preserve both cue texts on separate lines. They require a gap
of at most 120 ms, no overlap, at most eight seconds combined, acceptable combined
reading speed and line width, and no explicitly different speakers. Unknown speakers
remain marked as unavailable. Every proposal requires audio, speaker and phrasing
confirmation and has `applied: false`. Decisions record metadata only. No caption
is merged, shortened, moved over silence or stripped of a meaningful number/name.

## Validation and interpretation

Synthetic behavior cases live in `eval/alignment_cases`. The local benchmark runner
`scripts/eval_alignment_readability.py` reads a frozen manifest and saved responses.
Its optional `--live-audio` uses the same cached Paraformer, VAD and punctuation
models offline. The runner itself makes no external API requests.

The two-video benchmark uses 槟城一日游 小娘惹 and 第一次台湾旅游, with 32 windows.
Source finals are editorial references rather than audio-certified verbatim truth.
The initial catalog/threshold snapshot is retained. A first-pass US/currency
normalization issue was fixed after inspecting results, and the same saved outputs
were replayed; this is disclosed rather than presented as untouched held-out accuracy.

Review counts are not confirmed error counts. Passing code tests, matching model
outputs and provider schema compliance do not establish an overall quality gain.
Manual editing time requires a separate human timed study.

References for design only:

- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Netflix Chinese Simplified guide](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215986007-Chinese-Simplified-Timed-Text-Style-Guide)
- [Netflix reading-speed counting](https://partnerhelp.netflixstudios.com/hc/en-us/articles/115001352212-How-is-reading-speed-measured-Do-punctuation-and-spaces-count)
- [A-TASC and Chinese-aware subtitle evaluation](https://aclanthology.org/2025.acl-long.157/)
