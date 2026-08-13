# FunClip LLM Tool API (Gradio Client)

Use this guide to call FunClip features from automation tools (Codex/Claude scripts, CI jobs, etc.) via `gradio_client`.

Target URL:
- `https://srt.xcv58.xyz/`

## 1) Install and authenticate

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip gradio_client
cloudflared access login https://srt.xcv58.xyz/
```

## 2) Build an authenticated client

```python
import os
import subprocess
from gradio_client import Client

api_url = "https://srt.xcv58.xyz/"
token = os.getenv("CF_ACCESS_TOKEN") or subprocess.check_output(
    ["cloudflared", "access", "token", "-app", api_url],
    text=True,
).strip()

client = Client(api_url, headers={"CF-Access-Token": token})
```

## 3) Discover endpoints

```python
print(client.view_api(all_endpoints=True))
```

## 4) Common sync calls

```python
from gradio_client import handle_file

# 1) Transcribe audio/video -> text + srt
transcribe_res = client.predict(
    handle_file("/absolute/path/input.mp3"),
    api_name="/transcribe",
)
print(transcribe_res["status"])
srt_text = transcribe_res["srt_content"]

# 2) AI auto-correct SRT text
correct_res = client.predict(
    srt_text,         # srt_content
    "sk-xxx",         # api_key (or "" to use server-side OPENAI_API_KEY)
    "gpt-4o-mini",    # model_name
    "",               # custom_model (used only when model_name == "Custom")
    "",               # base_url
    True,             # return_traditional
    api_name="/srt_correct",
)
print(correct_res["status"])

# 3) Translate SRT file(s) to Traditional Chinese with text JSON
trad_text_res = client.predict(
    [handle_file("/absolute/path/a.srt")],
    api_name="/srt_translate_traditional_text",
)
print(trad_text_res["status"])
translated_srt = trad_text_res.get("translated_srt", "")
if not translated_srt and trad_text_res.get("items"):
    translated_srt = trad_text_res["items"][0].get("translated_srt", "")

# 4) Translate SRT file(s) to English (LLM)
en_res = client.predict(
    [handle_file("/absolute/path/a.srt")],  # srt_files
    "sk-xxx",                               # api_key
    "gpt-4o-mini",                          # model_name
    "",                                     # custom_model
    "",                                     # base_url
    api_name="/srt_translate_english",
)
print(en_res["download_path"])

# 5) Generate validated Traditional Chinese YouTube chapters from SRT text
chapter_res = client.predict(
    translated_srt or srt_text,  # srt_content
    "Auto",                     # density: Concise, Auto, or Detailed
    "Optional video subject",   # video_context
    "sk-xxx",                   # api_key (or "" for the server-side key)
    "gpt-4o-mini",              # model_name
    "",                         # custom_model
    "",                         # base_url
    0,                          # video_duration_seconds (0 uses SRT extent)
    api_name="/youtube_chapters",
)
print(chapter_res["chapters_text"])
```

## 6) Async submit + poll (recommended for long jobs)

```python
from time import sleep
from gradio_client import handle_file

submit_res = client.predict(
    handle_file("/absolute/path/input.mp3"),
    "",             # api_key
    "gpt-4o-mini",  # model_name
    "",             # custom_model
    "",             # base_url
    True,           # return_traditional
    api_name="/submit_transcribe_and_correct",
)

job_id = submit_res["job_id"]
while True:
    status_res = client.predict(job_id, True, api_name="/async_job_status")
    print(status_res["status"], status_res.get("stage"), status_res.get("eta_seconds"))
    if status_res["status"] in {"completed", "failed", "not_found"}:
        break
    sleep(2)
```

Notes:
- `download_path` and `corrected_traditional_file_path` are Gradio-served artifacts and may not be directly readable on the caller filesystem.
- `/srt_correct` also returns `corrected_traditional_srt` when `return_traditional` is true, so callers do not need to read the temporary artifact before generating chapters.
- `/youtube_chapters` returns paste-ready `chapters_text` plus a structured `chapters` list. It enforces at least three ascending timestamps, a minimum duration of 10 seconds per chapter, and specific non-duplicate Traditional Chinese titles without generic/filler-only labels or invisible control characters. Pass a finite positive `video_duration_seconds` value of at most 12 hours when the video is longer than its final subtitle cue; API value zero uses the SRT extent as a conservative proxy. It strictly rejects (rather than silently repairing) malformed titles and out-of-order, duplicate, over-dense, over-limit, or too-short chapter candidates. Edited text must begin with the exact ASCII timestamp `00:00` and stays within the same 24-chapter bound.
- Chapter generation accepts at most 100,000 SRT characters, 300,000 UTF-8 bytes, 3,000 subtitle cues, and 500 characters per cue. Optional video context is limited to 2,000 characters and 6,000 UTF-8 bytes. Model/provider identifiers are canonical ASCII LiteLLM identifiers of at most 200 characters. The model response is capped at 4,096 tokens, 20,000 characters, and 60,000 UTF-8 bytes before JSON parsing. LLM requests default to a 60-second timeout, one retry, and a shared concurrency limit of two; operators can adjust these with `FUNCLIP_CHAPTER_LLM_TIMEOUT_SECONDS`, `FUNCLIP_CHAPTER_LLM_MAX_RETRIES`, and `FUNCLIP_CHAPTER_CONCURRENCY_LIMIT`.
- Local upload mode accepts exactly one UTF-8 `.srt`. The browser reads its bounded bytes locally and sends base64 content, which is decoded and bound server-side to the exact session/revision; generation never dereferences a browser-supplied server cache path.
- Source preflight rejects primarily non-Chinese transcripts—including any Japanese kana and aggregate Japanese-specific Han or lexical evidence—subtitle sets that contain only ordinal chapter labels, filler phrases, or repeated low-information Chinese text, and controls, embedded BOMs, invisible direction markers, unsupported structural/timing suffixes, emoji, markup, or prompt delimiters anywhere in the raw SRT or optional context before any LLM request. One or more contiguous UTF-8 BOMs are accepted only at the very start of an SRT. Generated and edited titles also reject Japanese variants and Shinjitai instead of treating them as Traditional Chinese while preserving legitimate Traditional proper names.
- The Chapters tab offers copy in the editor and exposes a separate download only after validation succeeds. Actual duration, source identity, and artifact handles remain authoritative server-side per session; browser-returned hidden state cannot extend the video duration or remove another session's artifact. It clears the prior download as soon as editing begins and clears previous chapters whenever the source, generation settings, or upload changes. Slow producer/generation/edit callbacks stage results privately; a serialized, bounded per-session revision-aware commit prevents a late callback from overwriting newer subtitle state, edits, or chapter artifacts. The service also uses a finite global Gradio queue plus pre-queue admission guards for one outstanding UI generation per session and one public API generation per transport peer; ownership tokens and Gradio cancellation hooks release those reservations on every normal terminal path. Queue-rejected queued/publication leases expire after `FUNCLIP_CHAPTER_ADMISSION_TTL_SECONDS` (five minutes by default), and their exact unpublished candidate is discarded, so a failed dependent submission cannot strand the session. Editor input instead uses retained-latest coalescing: every retained edit advances the session revision, older edit candidates commit stale, and only the newest validated text can publish its status and download. Stale generation revisions are rejected again immediately before any paid model request.
- Chapter downloads are created directly in Gradio's served cache, and the exact served path remains a process-owned artifact. Replacement or invalidation deletes that prior exact artifact; an independent reaper retries failed deletions and expires abandoned artifacts after `FUNCLIP_CHAPTER_ARTIFACT_TTL_SECONDS` (one hour by default), normal shutdown retries remaining owned artifacts, and capacity reservations keep the ownership registry capped at 2,048 paths.
- When passing a custom `base_url`, callers must also pass their own `api_key`; the service never forwards its server-side key to a caller-selected URL. The server-side OpenAI key is restricted to models LiteLLM identifies as OpenAI. Operators using aliases on the configured `OPENAI_BASE_URL` gateway must opt in each exact alias through the comma-separated `FUNCLIP_TRUSTED_LLM_MODELS` environment variable. Gemini, Anthropic, and other provider selections require a caller-supplied key unless the operator deliberately maps and allowlists that exact gateway alias.
- For endpoints that expect multiple files, pass a list even for one file.
- `/submit_transcribe_and_correct` copies an upload into a private staging file, then deletes both that stage and the exact Gradio-cached input before publishing either `completed` or `failed`. A cleanup error fails the job instead of reporting a clean terminal result.
- The async cleanup accepts only regular, nonsymlink files contained by Gradio's configured upload root. It never deletes an arbitrary caller-supplied path.
- Gradio additionally checks hourly for cache files older than one hour and clears its cache when the service shuts down. These are fallback bounds for abandoned/non-async uploads; the async endpoint performs terminal cleanup immediately.
