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
```

## 5) Async submit + poll (recommended for long jobs)

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
- For endpoints that expect multiple files, pass a list even for one file.
