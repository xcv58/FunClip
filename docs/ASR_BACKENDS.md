# Optional ASR backends

Paraformer remains the default. Qwen3 ASR 1.7B is an optional experimental
Chinese subtitle backend on Apple Silicon. Selection affects that request only,
including its glossary comparison and alternate audio context checks.

## Switching models

Choose **Speech recognition model** in Gradio or Streamlit, then start processing.
Existing results keep their recognition evidence when the selector changes.
All existing API endpoints retain their input shapes and use Paraformer.
Two additive endpoints accept a backend explicitly:

| Endpoint | Ordered inputs |
| --- | --- |
| `/transcribe_with_backend` | `file_path`, `asr_backend`, `glossary_id`, `glossary_mode`, `review_context` |
| `/submit_transcribe_and_correct_with_backend` | `file_path`, `api_key`, `model_name`, `custom_model`, `base_url`, `return_traditional`, `asr_backend`, `glossary_id`, `glossary_mode`, `review_context` |

`asr_backend` is `paraformer` or `qwen3-asr-1.7b`. The LLM `model_name` controls
correction separately. Async polling and glossary controls follow
[LLM_TOOL_API.md](LLM_TOOL_API.md). Unsupported backend IDs are rejected before
staging an async upload.

```python
from gradio_client import Client, handle_file
client = Client("http://127.0.0.1:7860")
result = client.predict(
    handle_file("episode.wav"), "qwen3-asr-1.7b", "penghu-food", "reference", False,
    api_name="/transcribe_with_backend",
)
```

## Optional local setup

The default environment needs no new packages or checkpoints. Qwen runs in an
isolated subprocess so its MLX and Transformers dependencies cannot change the
existing FunASR environment. Use Python 3.12 on Apple Silicon and a disk with
sufficient free space:

```sh
python3.12 -m venv /path/to/qwen-env
/path/to/qwen-env/bin/pip install -r requirements-qwen.txt
```

Download complete original `Qwen/Qwen3-ASR-1.7B` and
`Qwen/Qwen3-ForcedAligner-0.6B` checkpoints separately, including configuration
and tokenizer files. The backend accepts local directories and never downloads
models. Its validated aligner configuration uses `timestamp_segment_time` of
80 milliseconds. The worker checks optional package versions before inference.

Set these server environment variables before launching Gradio or Streamlit:

```sh
export FUNCLIP_QWEN_PYTHON=/path/to/qwen-env/bin/python
export FUNCLIP_QWEN_MODEL=/path/to/Qwen3-ASR-1.7B
export FUNCLIP_QWEN_ALIGNER=/path/to/Qwen3-ForcedAligner-0.6B
funclip-env/bin/python gradio_app.py
```

This machine uses a self-contained internal runtime bundle:

```text
~/.local/share/FunClip/qwen3-asr-1.7b/
  bin/python       launcher with bundle-local caches and offline settings
  python/          private Python 3.12 interpreter and standard library
  env/             validated Python packages
  models/qwen/     Qwen3-ASR-1.7B checkpoint
  models/aligner/  Qwen3-ForcedAligner-0.6B checkpoint
```

```sh
export FUNCLIP_QWEN_PYTHON="$HOME/.local/share/FunClip/qwen3-asr-1.7b/bin/python"
export FUNCLIP_QWEN_MODEL="$HOME/.local/share/FunClip/qwen3-asr-1.7b/models/qwen"
export FUNCLIP_QWEN_ALIGNER="$HOME/.local/share/FunClip/qwen3-asr-1.7b/models/aligner"
```

The bundle includes its own interpreter rather than linking to Codex's shared
Python runtime. It does not require the external benchmark drive. The launcher
keeps model and library caches inside the bundle and enables offline loading.
The FunClip application and macOS system frameworks remain host requirements.
`SHA256SUMS.json` and `BUNDLE.json` retain migration integrity evidence. Existing
external benchmark copies remain available as rollback data.

Missing paths cause an actionable setup error; there is no silent fallback.
Launch agents must receive these variables explicitly when this backend is
enabled for the hosted service. Source edits do not restart an already running
service. Paraformer remains the default backend.

## Recognition and timing contract

1. Reuse the loaded Paraformer's FSMN VAD weights with a request-local copy of
   its configuration. Require ordered positive in-bounds speech regions. Empty
   VAD output returns no speech without launching Qwen.
2. Recognize and force-align each VAD region locally. Models load once per
   request; long regions use the package's internal chunking. Confirmed glossary
   words become Qwen context in hints mode. Reference mode supplies no ASR bias.
   Hints are not vocabulary validation or proof that a word was spoken.
3. Require exact lexical coverage, finite ordered endpoints, audio bounds and
   nontruncated generation. English subwords, overlaps and zero duration
   characters become indivisible spans using original measured endpoints. Each
   span must contain a real positive native interval; point observations alone
   cannot establish duration. No character times are interpolated. Milliseconds
   are rounded to SRT resolution. Up to one 80 ms lattice step of final audio
   overhang may be trimmed to the measured audio boundary and recorded. Larger
   errors stop the job for review.
4. Segment only on atomic and existing numeric boundaries, then apply the
   existing punctuation-free formatter. Unsafe names or expressions still
   require review. Numeric boundary repair requiring individual token identity
   is conservatively unavailable with atomic timing. Numeric phrases inside
   a sentence remain protected.

`backend_native` retains original VAD output and every original Qwen result,
including text, point timestamps, language and chunk diagnostics. It is never
rewritten for display. Standard `sentence_info` and `timestamp` fields contain
the adapter's span view, explicitly identified by
`asr_backend.timestamp_unit = "atomic text span"`. `timestamp_adapter` records
grouping and boundary trims. Adapter failures retain native evidence and accept
no partial SRT.

The review queue identifies grouped timing. Hint and context comparisons use
enclosing measured spans as review windows without claiming character precision.
Qwen currently supports subtitle generation only; text-based clipping and
speaker diarization require Paraformer. Raw ASR punctuation remains intact;
only displayed and exported subtitle bodies follow the punctuation-free style.

The worker uses private temporary files, offline library settings, a 30 minute
timeout and a bounded result size. Its macOS subprocess also runs with network
access denied. Provider credentials are removed from the environment. Temporary
audio and worker files are removed afterward. Final SRTs and original media are
never modified.

## Extending and validating

Implement `generate(wave, **options)` returning the recognition payload and an
optional `recognition_hotwords(...)` context manager. Register an ID, label and
factory in `funclip/asr_backends.py`; both selectors use that registry. Keep
selection request scoped, validate timing, retain native evidence and test
speech, no speech, invalid alignment and glossary comparisons. Keep optional
imports out of startup and retain the existing default.

```sh
funclip-env/bin/python -m unittest discover -s funclip/test -q
```

Private benchmark replays and fresh audio results stay under ignored
`eval/model_migration_runs/`. Structural timing validity is not acoustic
accuracy; disagreement with edited final SRTs is not a human-verified error rate.
