# SRT Reference Script Plan

## Goal

Add an optional reference-script-assisted SRT correction mode without changing
the existing no-script workflow by default.

This is an additive feature. If no reference script is provided, the current
behavior should remain exactly the same.

## Why Full Script First

For this repo's current usage, a full-script V1 is acceptable.

- Existing SRT-only correction runs already use roughly `6.6k-8.6k` prompt
  tokens on the local eval set.
- Typical videos are under 30 minutes.
- Token cost and context window do not currently look like the main blockers.

The main risk is not prompt size. The main risk is that the LLM may over-trust
the script and "correct" subtitles toward words that were never actually
spoken.

Because of that, V1 should optimize for regression safety, not retrieval or
prompt compression.

## Design Principles

- Keep the current no-script correction path unchanged.
- Use the reference script only when the user explicitly provides one.
- Treat the script as soft guidance, not ground truth.
- Assume the script may be partial, stale, paraphrased, or from a different
  cut.
- Preserve SRT structure exactly.
- Prefer conservative fallback behavior over aggressive script-driven rewrites.

## Non-Goals For V1

- No retrieval or chunk-level script matching.
- No alignment pipeline between script sentences and subtitle segments.
- No automatic script-to-hotword extraction yet.
- No change to the default UI flow for users who do not provide a script.

## Proposed V1 Approach

### 1. Preserve the existing path

Keep the current correction entry points intact:

- `funclip/llm/srt_corrector.py`
- `gradio_app.py`

If `reference_script` is absent, call the current no-script logic and prompt.

### 2. Add an optional reference script input

Support an optional `reference_script` input in the correction UI and API.

Accepted forms can be:

- pasted text
- uploaded `.txt`
- uploaded `.md`

This input must be optional and must not affect current callers unless it is
explicitly populated.

### 3. Add a script-assisted prompt variant

When a reference script is provided, use a separate prompt variant with the same
structural constraints as the current prompt.

The prompt must state:

- the script is optional guidance only
- the script may be incomplete or mismatched
- do not force subtitles to match the script
- only copy from the script when the spoken content clearly matches
- if uncertain, preserve the subtitle text

### 4. Keep hard structural rules unchanged

Script-assisted mode must still enforce:

- valid SRT output only
- same subtitle count
- same subtitle indices
- same timestamps
- no merge or split of segments

### 5. Add a conservative fallback

Preferred V1 behavior in script mode:

1. Run baseline no-script correction.
2. Run script-assisted correction.
3. Compare the two outputs.
4. If the script-assisted output appears too aggressive or structurally unsafe,
   return the baseline result instead.

This keeps the new feature isolated and lowers the risk of harming production
results on partial or stale scripts.

### 6. Add lightweight acceptance heuristics

Before accepting the script-assisted output, validate:

- SRT parses successfully
- same block count as input
- same timestamps as input
- edit distance is not wildly larger than the baseline correction
- no obvious mass rewrite behavior

V1 heuristics should stay simple and interpretable.

## Implementation Checklist

### Prompt And Core Logic

- [ ] Keep the current prompt builder as the no-script baseline in
      `funclip/llm/srt_corrector.py`.
- [ ] Add a second prompt builder for script-assisted correction.
- [ ] Extend `request_srt_correction(...)` to accept
      `reference_script: str | None = None`.
- [ ] Route to the existing prompt when `reference_script` is empty.
- [ ] Route to the script-assisted prompt when `reference_script` is present.
- [ ] Keep the current return shape compatible with existing callers.

### Validation And Fallback

- [ ] Add structural validation helpers for corrected SRT output.
- [ ] Add a simple aggressiveness check comparing baseline vs script-assisted
      output.
- [ ] In script mode, prefer the baseline result when validation fails or the
      script-assisted result looks over-rewritten.
- [ ] Log which path was chosen for later evaluation.

### UI

- [ ] Add optional reference script input to the transcription correction tab in
      `gradio_app.py`.
- [ ] Add optional reference script input to the SRT translator correction tab
      in `gradio_app.py`.
- [ ] Keep the current correction controls and defaults unchanged.
- [ ] Make it visually clear that the script is optional.
- [ ] Add concise help text explaining that the script may be partial and is not
      treated as ground truth.

### API

- [ ] Extend the correction API surface to accept an optional
      `reference_script`.
- [ ] Keep existing API calls backward-compatible.
- [ ] Update `docs/LLM_TOOL_API.md` with one example request that includes a
      reference script.

### Evaluation

- [ ] Extend the eval workflow to support optional reference-script inputs per
      case.
- [ ] Add cases for:
      exact-match script,
      partial script,
      stale script,
      mismatched script,
      and no-script baseline parity.
- [ ] Compare baseline no-script correction vs full-script-assisted correction on
      the existing fixture set.
- [ ] Track cases where script assistance improves names and terms.
- [ ] Track cases where script assistance causes over-correction.

### Fixture Format

Optional V1 convention for eval cases:

- `original_transcribed.srt`
- `final_traditional.srt`
- `raw_text.txt`
- `reference_script.txt`
- `meta.json`

`reference_script.txt` should be optional so existing cases remain valid.

### Rollout

- [ ] Ship behind an optional UI field first.
- [ ] Treat it as an advanced/manual feature initially.
- [ ] Keep retrieval/chunk matching deferred to V2 unless prompt size or
      alignment becomes a real issue.

## Suggested File Touch Points

- `funclip/llm/srt_corrector.py`
  Add optional script-aware prompting and fallback orchestration.
- `gradio_app.py`
  Add optional input controls and plumb the new parameter through the current
  correction actions.
- `docs/LLM_TOOL_API.md`
  Document the optional API argument.
- `scripts/eval_srt_correction.py`
  Add reference-script-aware evaluation support.
- `scripts/build_srt_eval_case.py`
  Optionally support copying a provided reference script into the case folder.

## Acceptance Criteria

- No-script correction results are unchanged when no reference script is
  supplied.
- Script-assisted mode improves obvious names, entities, and terminology on
  matching scripts.
- Partial or stale scripts do not cause large rewrite regressions.
- Structural SRT safety remains intact.
- Existing API and UI flows remain backward-compatible.

## Deferred V2 Ideas

- Retrieve only relevant script snippets instead of sending the full script.
- Align script windows to subtitle chunks.
- Derive ASR hotwords from the script and feed them into FunASR before LLM
  correction.
- Add more formal rewrite-detection metrics to the eval pipeline.
