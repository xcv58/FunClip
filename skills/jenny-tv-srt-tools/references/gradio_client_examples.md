# Gradio Client Examples

Target URL:
- `https://srt.xcv58.xyz/`

Install:
```shell
pip install gradio_client
```

Inspect endpoints:
```python
from gradio_client import Client

client = Client("https://srt.xcv58.xyz/")
print(client.view_api(all_endpoints=True))
```

Call endpoints:
```python
from gradio_client import Client, handle_file

client = Client("https://srt.xcv58.xyz/")

# 1) Transcribe audio/video -> text + srt
transcribe_res = client.predict(
    handle_file("/absolute/path/input.mp4"),
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

# 3) Translate SRT file(s) to Traditional Chinese
trad_res = client.predict(
    [handle_file("/absolute/path/a.srt")],
    api_name="/srt_translate_traditional",
)
print(trad_res["download_path"])

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
