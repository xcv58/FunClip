---
name: jenny-tv-srt-tools
description: Call FunClip Gradio API endpoints from automation scripts with gradio_client. Use when Codex/Claude or other tools need to interact with deployed FunClip service (https://srt.xcv58.xyz/) for media transcription, SRT AI auto-correction, Traditional Chinese conversion, or English translation.
---

# FunClip Gradio API Client

## Overview
Call FunClip features through stable Gradio API endpoints with `gradio_client`.
Use this skill for machine-to-machine access instead of manual UI clicking.

## Workflow

1. Install `gradio_client` in the caller environment.
2. Create a client for `https://srt.xcv58.xyz/`.
3. Run `client.view_api(all_endpoints=True)` to confirm endpoint names and input shapes.
4. Call the endpoint by `api_name`.
5. For file inputs, always wrap paths with `handle_file(...)`.

## Recommended Workflows

### 1) Audio/Video -> SRT -> AI Correction

1. Call `/transcribe` with one media file (`.mp3`, `.wav`, `.m4a`, `.mp4`, etc.).
2. Read `srt_content` from the response.
3. Call `/srt_correct` with that `srt_content`.
4. Use `corrected_srt` (and optional Traditional output) for downstream steps.

### 2) Chinese SRT -> Traditional Chinese SRT -> AI Correction

1. Call `/srt_translate_traditional` with one or more `.srt` files.
2. Use the returned translated output for review/downloading.
3. For final cleanup, call `/srt_correct` on the translated SRT content.
4. Prefer this flow when you want fast Traditional conversion first, then quality polish.

## Endpoints

- `/transcribe`
Purpose: Transcribe one audio/video file and return markdown text + SRT.
Input: one media file.

- `/srt_correct`
Purpose: AI-correct SRT content text and optionally return Traditional Chinese output path.
Inputs: `srt_content`, `api_key`, `model_name`, `custom_model`, `base_url`, `return_traditional`.

- `/srt_translate_traditional`
Purpose: Convert uploaded SRT file(s) to Traditional Chinese.
Input: one or more `.srt` files.

- `/srt_translate_english`
Purpose: Translate uploaded SRT file(s) to English via LLM.
Inputs: one or more `.srt` files plus LLM settings.

## Input Rules

- Use absolute file paths for `handle_file(...)`.
- Pass file lists for endpoints expecting `file_count="multiple"` even when sending one file.
- Set `model_name="Custom"` only when `custom_model` is provided.
- Pass empty string for `api_key` only if server-side `OPENAI_API_KEY` is configured.

## Output Handling

- Read `status` fields first.
- Use returned SRT text for downstream workflows when available.
- Treat returned download paths as server-side paths exposed by Gradio.

## Error Handling

- On quota/rate-limit errors, retry with another key/model or back off.
- On schema mismatch, rerun `view_api(all_endpoints=True)` and align argument order.

## References

- For full copy-paste examples, read [`references/gradio_client_examples.md`](references/gradio_client_examples.md).
