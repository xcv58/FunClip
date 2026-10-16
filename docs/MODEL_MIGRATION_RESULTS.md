# GPT-6 Luna rollout — October 16, 2026

FunClip now defaults to `gpt-6-luna` for correction, English translation and
YouTube chapters. Gradio, Streamlit, hidden API inputs and background workers
share the same configuration. Explicit model selections remain supported.

The installed LiteLLM adapter routes Luna through `openai/gpt-6-luna`, sets
`reasoning_effort="none"` and uses `max_completion_tokens` for bounded requests.
This preserves the existing text-task role and Chat Completions interface.

## Correction and evidence

The model returns cue text in JSON. FunClip retains the input IDs, timing lines,
cue boundaries and number of text lines. Missing, extra, duplicate or reordered
IDs, malformed JSON, empty text, refusals and incomplete responses fail local
validation. Each batch has one repair attempt; unresolved failures publish no
partial corrected file. English translation also reconstructs SRT structure
locally and retains its existing two-line layout limit.

Transcription retains raw ASR text, available timestamps and sentence data, the
original SRT and hashes. Audio arrays are excluded. Async status can return the
transcription after correction fails, while uploaded-media cleanup still runs.
See [the API contract](LLM_TOOL_API.md) for fields, limits and failure behavior.

## Evaluation

Five approved Chinese correction inputs containing 1,353 source cues ran twice
per model through the new correction function.

| Measure | GPT-4o mini | GPT-6 Luna |
| --- | ---: | ---: |
| Full-file runs preserving all IDs and timestamps | 10/10 | 10/10 |
| Invalid batch responses rejected and repaired | 27 | 0 |
| Median time per file | 54.35 s | 24.41 s |
| Estimated cost for ten runs | $0.04927 | $0.02713 |
| Generated controls matching expected reading text | 3/4 | 4/4 |

This is a small sample. Historical edited references contain deletion,
rewriting and resegmentation, so similarity to them is not correction accuracy.
Chinese wording, names, numbers and English terms still require review.
Dense English cues can exceed the existing layout limit and fail validation;
the migration does not change segmentation or recognition models.

Raw subtitle inputs, responses and detailed local reports are retained under
the ignored `eval/model_migration_runs/` directory. They are not part of this
release commit. The evaluator is available as `scripts/eval_model_migration.py`;
each real-data run needs authorization for its exact external-transfer scope.

## Validation and service operation

Before rollout, all 142 FunClip tests passed using the service's
`funclip-env/bin/python`. The youtube-helper caller migration passed its full
185-test validation. Direct OpenAI smoke checks for correction, translation and
chapters also passed with the default model omitted from each call.

The existing launch agent is `gui/501/com.funclip.service`. It runs
`gradio_app.py` from this checkout with `funclip-env/bin/python` on port 7860.
After confirming the queue is idle, the service can be restarted with:

```sh
launchctl kickstart -k gui/$(id -u)/com.funclip.service
```

Verify the local health, hosted API schema and Luna defaults after restart,
then run synthetic media through youtube-helper. A successful restart alone
does not establish hosted pipeline acceptance. Retain the previous commit for
rollback if startup or hosted acceptance fails.

Compatibility settings were checked against the official
[GPT-6 migration guide](https://developers.openai.com/api/docs/guides/latest-model#migration-quickstart)
and [Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).
