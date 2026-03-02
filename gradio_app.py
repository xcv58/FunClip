import gradio as gr
import os
import sys
import time
import soundfile as sf
import difflib
import re
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile, TemporaryDirectory
import tempfile
import zipfile

load_dotenv()

# --- 1. SETUP PATHS & IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
funclip_dir = os.path.join(current_dir, "funclip")
if funclip_dir not in sys.path:
    sys.path.append(funclip_dir)

try:
    from funclip.videoclipper import VideoClipper
    from funasr import AutoModel
    from funclip.llm.srt_corrector import correct_srt_content
    from funclip.llm.chinese_converter import convert_to_traditional
    from funclip.llm.srt_translator import translate_srt_to_english
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- 2. LOAD MODEL (Global Scope) ---
print("Loading FunASR Model...")
funasr_model = AutoModel(
    model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    # Note: spk_model removed - not needed when sd_switch='no' and causes MPS float64 issues
    device="mps",  # Apple Silicon GPU acceleration (use "cuda:0" for NVIDIA GPUs)
)
print("✅ AI Model Ready")

# --- 3. STATE ---
# Note: Per-session state is now handled via gr.State() components
# to support concurrent users without data conflicts

# --- 4. PROCESSING FUNCTIONS ---

def process_media(file_path, session_state, progress=gr.Progress()):
    """Process audio/video file and return transcription results."""
    if not file_path:
        gr.Warning("Please upload a file first.")
        return None, None, None, "No file uploaded", session_state
    
    start_time = time.time()
    
    # Get original filename for later use (stored in session state)
    original_name = os.path.basename(file_path)
    base_name = os.path.splitext(original_name)[0]
    session_state = session_state.copy() if session_state else {}
    session_state["original_filename"] = base_name
    
    # Determine file type
    _, ext = os.path.splitext(file_path)
    is_video = ext.lower() in ['.mp4', '.avi', '.mkv', '.mov']
    
    # Get duration based on file type
    duration_sec = 0
    try:
        if is_video:
            # Use subprocess to get video duration via ffprobe
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                 '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                duration_sec = float(result.stdout.strip())
        else:
            # Use librosa for audio files
            import librosa
            duration_sec = librosa.get_duration(path=file_path)
    except Exception as e:
        print(f"Could not determine duration: {e}")
        duration_sec = 0
    
    # Initialize Clipper
    audio_clipper = VideoClipper(funasr_model)
    audio_clipper.lang = 'zh'
    
    # Process based on file type (reuse is_video from earlier)
    if is_video:
        with TemporaryDirectory() as temp_dir:
            res_text, res_srt, _ = audio_clipper.video_recog(
                file_path, sd_switch='no', output_dir=temp_dir
            )
    else:
        import librosa
        wav, sr = librosa.load(file_path, sr=16000)
        res_text, res_srt, _ = audio_clipper.recog((sr, wav), sd_switch='no')
    
    # Calculate processing time
    end_time = time.time()
    total_time = end_time - start_time
    speed_x = duration_sec / total_time if total_time > 0 and duration_sec > 0 else 0
    
    # Store SRT for correction feature (in session state)
    session_state["original_srt"] = res_srt
    
    # Save SRT to temp file with proper filename (use system temp dir for Gradio compatibility)
    srt_filename = f"{base_name}.srt"
    temp_dir = tempfile.gettempdir()
    srt_path = os.path.join(temp_dir, srt_filename)
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(res_srt)
    
    # Format status message
    status_msg = f"✅ Completed in {total_time:.2f}s ({speed_x:.1f}x speed)"
    
    # Format text with proper line breaks for Markdown display
    formatted_text = res_text.replace("\n", "\n\n") if res_text else ""
    
    return formatted_text, res_srt, srt_path, status_msg, session_state


def get_api_key_status():
    """Check if system API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return "✅ System API Key detected"
    return "⚠️ No System API Key found"


def resolve_llm_config(api_key, model_name, custom_model, base_url):
    """Resolve effective LLM config from user input and environment."""
    if model_name == "Custom":
        if not custom_model or not custom_model.strip():
            raise gr.Error("Custom model is selected, but model name is empty. Please enter a custom model name.")
    effective_model = custom_model if model_name == "Custom" else model_name
    eff_api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
    eff_base_url = base_url if base_url else os.getenv("OPENAI_BASE_URL")

    if not eff_api_key:
        raise gr.Error("No API Key provided. Please enter an API key or set OPENAI_API_KEY in your .env file.")

    return eff_api_key, eff_base_url, effective_model


def _normalize_error_message(error):
    """Remove noisy provider-specific prefixes and keep message compact."""
    message = str(error or "").strip()
    if not message:
        return "Unknown error."

    # Keep only the first line to avoid dumping stack-like details to UI.
    message = message.splitlines()[0].strip()

    # Strip duplicated wrapper prefixes like:
    # "litellm.RateLimitError: RateLimitError: OpenAIException - ..."
    patterns = [
        r"^(litellm\.)?[A-Za-z_]+Error:\s*",
        r"^OpenAIException\s*-\s*",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            cleaned = re.sub(pattern, "", message, count=1).strip()
            if cleaned != message:
                message = cleaned
                changed = True

    return message


def format_llm_error(error, operation_name):
    """Return a concise, actionable UI message for common LLM failures."""
    raw_message = _normalize_error_message(error)
    raw_lower = raw_message.lower()

    if (
        "insufficient_quota" in raw_lower
        or "exceeded your current quota" in raw_lower
        or ("quota" in raw_lower and "billing" in raw_lower)
    ):
        return (
            f"{operation_name} failed: API quota/billing limit reached.\n"
            "Action: add credits/billing, use another API key, or switch provider/model."
        )

    if (
        "ratelimit" in raw_lower
        or "rate limit" in raw_lower
        or "too many requests" in raw_lower
        or "error code: 429" in raw_lower
    ):
        return (
            f"{operation_name} failed: request rate limit reached.\n"
            "Action: wait a moment and retry, or switch to a different model/provider."
        )

    if (
        "invalid_api_key" in raw_lower
        or "incorrect api key" in raw_lower
        or "authentication" in raw_lower
        or "unauthorized" in raw_lower
        or "error code: 401" in raw_lower
    ):
        return (
            f"{operation_name} failed: API key is invalid or unauthorized.\n"
            "Action: verify API key, base URL, and provider permissions."
        )

    if (
        "model_not_found" in raw_lower
        or "model not found" in raw_lower
        or ("does not exist" in raw_lower and "model" in raw_lower)
        or "error code: 404" in raw_lower
    ):
        return (
            f"{operation_name} failed: model is unavailable for this provider/key.\n"
            "Action: choose another model or confirm your provider supports the selected model."
        )

    if (
        "timed out" in raw_lower
        or "timeout" in raw_lower
        or "connection" in raw_lower
        or "network" in raw_lower
        or "dns" in raw_lower
    ):
        return (
            f"{operation_name} failed: network/API timeout.\n"
            "Action: check network/base URL and retry."
        )

    return f"{operation_name} failed: {raw_message}"


def normalize_srt_file_input(srt_files):
    """Normalize Gradio file input to a list of file paths."""
    if not srt_files:
        return []
    if isinstance(srt_files, str):
        return [srt_files]
    if isinstance(srt_files, list):
        return srt_files
    return [srt_files]


def build_download_path(file_paths, zip_filename):
    """Return single file path or a zip path if multiple files are present."""
    if not file_paths:
        raise gr.Error("No files available for download.")
    if len(file_paths) == 1:
        return file_paths[0]

    zip_path = os.path.join(os.path.dirname(file_paths[0]), zip_filename)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for p in file_paths:
            zf.write(p, os.path.basename(p))
    return zip_path


def run_llm_correction_for_content(original_srt, base_name, api_key, model_name, custom_model, base_url):
    """Run AI correction for one SRT content string."""
    start_time = time.time()

    if not original_srt or not original_srt.strip():
        raise gr.Error("No SRT content found for AI correction.")

    eff_api_key, eff_base_url, effective_model = resolve_llm_config(
        api_key=api_key,
        model_name=model_name,
        custom_model=custom_model,
        base_url=base_url
    )

    try:
        corrected_srt = correct_srt_content(
            srt_content=original_srt,
            api_key=eff_api_key,
            base_url=eff_base_url,
            model=effective_model
        )
    except Exception as e:
        raise gr.Error(format_llm_error(e, "AI auto correction"))

    diff_html = generate_diff_html(original_srt, corrected_srt)
    traditional_srt = convert_to_traditional(corrected_srt)

    safe_base = base_name if base_name else "subtitles"
    temp_dir = tempfile.gettempdir()

    orig_path = os.path.join(temp_dir, f"{safe_base}.srt")
    with open(orig_path, 'w', encoding='utf-8') as f:
        f.write(original_srt)

    corr_path = os.path.join(temp_dir, f"corrected_{safe_base}.srt")
    with open(corr_path, 'w', encoding='utf-8') as f:
        f.write(corrected_srt)

    trad_path = os.path.join(temp_dir, f"corrected_{safe_base}_traditional.srt")
    with open(trad_path, 'w', encoding='utf-8') as f:
        f.write(traditional_srt)

    elapsed_time = time.time() - start_time
    status_msg = f"✅ Correction completed in {elapsed_time:.2f}s"
    return original_srt, corrected_srt, diff_html, orig_path, corr_path, trad_path, status_msg


def run_llm_correction(original_srt, api_key, model_name, custom_model, base_url, session_state, progress=gr.Progress()):
    """Run LLM-based SRT correction from transcription pipeline content."""
    base_name = session_state.get("original_filename", "subtitles") if session_state else "subtitles"
    return run_llm_correction_for_content(
        original_srt=original_srt,
        base_name=base_name,
        api_key=api_key,
        model_name=model_name,
        custom_model=custom_model,
        base_url=base_url
    )


def generate_diff_html(original, corrected):
    """Generate styled HTML diff view."""
    diff = difflib.HtmlDiff().make_file(
        original.splitlines(),
        corrected.splitlines(),
        fromdesc='Original',
        todesc='Corrected',
        context=True,
        numlines=3
    )
    
    # Add custom styling to make it fit nicely
    custom_css = """
    <style>
        table.diff { width: 100%; font-family: monospace; font-size: 12px; }
        .diff_header { background-color: #e0e0e0; }
        .diff_next { background-color: #c0c0c0; }
        .diff_add { background-color: #aaffaa; }
        .diff_chg { background-color: #ffff77; }
        .diff_sub { background-color: #ffaaaa; }
        td { padding: 2px 8px; white-space: pre-wrap; word-wrap: break-word; }
    </style>
    """
    
    return custom_css + diff


def update_custom_model_visibility(model_choice):
    """Show/hide custom model input based on dropdown selection."""
    return gr.update(visible=(model_choice == "Custom"))


# --- 5. GRADIO UI LAYOUT ---

def translate_srt_to_traditional(srt_files):
    """Translate one or more SRT files to Traditional Chinese (step 1, fast)."""
    srt_paths = normalize_srt_file_input(srt_files)
    if not srt_paths:
        raise gr.Error("Please upload an SRT file first.")

    temp_dir = tempfile.mkdtemp()
    output_paths = []
    last_original = ""
    last_translated = ""

    for srt_file in srt_paths:
        with open(srt_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        if not srt_content.strip():
            continue

        traditional_srt = convert_to_traditional(srt_content)

        original_name = os.path.basename(srt_file)
        base_name = os.path.splitext(original_name)[0]
        trad_filename = f"{base_name}_traditional.srt"
        trad_path = os.path.join(temp_dir, trad_filename)
        with open(trad_path, 'w', encoding='utf-8') as f:
            f.write(traditional_srt)

        output_paths.append(trad_path)
        last_original = srt_content
        last_translated = traditional_srt

    if not output_paths:
        raise gr.Error("All uploaded SRT files are empty.")

    if len(output_paths) == 1:
        preview_original = last_original
        preview_translated = last_translated
        download_path = output_paths[0]
    else:
        preview_original = f"Translated {len(output_paths)} files to Traditional Chinese."
        preview_translated = last_translated
        download_path = build_download_path(output_paths, "traditional_chinese_srts.zip")

    translator_state = {
        "latest_output_paths": output_paths,
        "latest_output_kind": "traditional"
    }
    return preview_original, preview_translated, download_path, translator_state


def translate_srt_to_english_fn(srt_files, api_key, model_name, custom_model, base_url):
    """Translate one or more SRT files from Simplified Chinese to English using LLM."""
    srt_paths = normalize_srt_file_input(srt_files)
    if not srt_paths:
        raise gr.Error("Please upload an SRT file first.")

    eff_api_key, eff_base_url, effective_model = resolve_llm_config(
        api_key=api_key,
        model_name=model_name,
        custom_model=custom_model,
        base_url=base_url
    )

    temp_dir = tempfile.mkdtemp()
    output_paths = []
    last_original = ""
    last_translated = ""

    for srt_file in srt_paths:
        with open(srt_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        if not srt_content.strip():
            continue

        try:
            english_srt = translate_srt_to_english(
                srt_content=srt_content,
                api_key=eff_api_key,
                base_url=eff_base_url,
                model=effective_model
            )
        except Exception as e:
            file_name = os.path.basename(srt_file)
            raise gr.Error(format_llm_error(e, f"English translation ({file_name})"))

        original_name = os.path.basename(srt_file)
        base_name = os.path.splitext(original_name)[0]
        eng_filename = f"{base_name}_english.srt"
        eng_path = os.path.join(temp_dir, eng_filename)
        with open(eng_path, 'w', encoding='utf-8') as f:
            f.write(english_srt)

        output_paths.append(eng_path)
        last_original = srt_content
        last_translated = english_srt

    if not output_paths:
        raise gr.Error("All uploaded SRT files are empty.")

    if len(output_paths) == 1:
        preview_original = last_original
        preview_translated = last_translated
        download_path = output_paths[0]
    else:
        preview_original = f"Translated {len(output_paths)} files to English."
        preview_translated = last_translated
        download_path = build_download_path(output_paths, "english_srts.zip")

    translator_state = {
        "latest_output_paths": output_paths,
        "latest_output_kind": "english"
    }
    return preview_original, preview_translated, download_path, translator_state


def update_srt_upload_visibility(source_choice):
    """Show upload component only when 'Upload SRT file(s)' source is selected."""
    return gr.update(visible=(source_choice == "Upload SRT file(s)"))


def resolve_stream_or_upload_srt(source_choice, stream_content, stream_base_name, upload_srt_file, stream_error_message):
    """Resolve correction input as one SRT content from stream or uploaded file."""
    if source_choice == "Upload SRT file(s)":
        srt_paths = normalize_srt_file_input(upload_srt_file)
        if not srt_paths:
            raise gr.Error("Please upload an SRT file first.")
        srt_path = srt_paths[0]
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        if not srt_content.strip():
            raise gr.Error("Uploaded SRT file is empty.")
        base_name = os.path.splitext(os.path.basename(srt_path))[0]
        return srt_content, base_name

    if not stream_content or not stream_content.strip():
        raise gr.Error(stream_error_message)

    base_name = stream_base_name if stream_base_name else "subtitles"
    return stream_content, base_name


def resolve_translator_correction_files(source_choice, translator_state, uploaded_srt_files):
    """Resolve translator correction input files from stream output or uploaded files."""
    if source_choice == "Use output from previous step":
        stream_paths = translator_state.get("latest_output_paths", []) if translator_state else []
        if not stream_paths:
            raise gr.Error("No translated output found above. Run translation first or upload SRT file(s).")
        return stream_paths

    uploaded_paths = normalize_srt_file_input(uploaded_srt_files)
    if not uploaded_paths:
        raise gr.Error("Please upload SRT file(s) for AI correction.")
    return uploaded_paths


def run_llm_correction_for_files(srt_files, api_key, model_name, custom_model, base_url):
    """Run AI correction for one or more SRT files and return preview/diff/downloads."""
    srt_paths = normalize_srt_file_input(srt_files)
    if not srt_paths:
        raise gr.Error("Please provide SRT file(s) for AI correction.")

    eff_api_key, eff_base_url, effective_model = resolve_llm_config(
        api_key=api_key,
        model_name=model_name,
        custom_model=custom_model,
        base_url=base_url
    )

    start_time = time.time()
    temp_dir = tempfile.mkdtemp()
    original_paths = []
    corrected_paths = []
    last_original = ""
    last_corrected = ""
    seen_names = {}

    for srt_path in srt_paths:
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        if not srt_content.strip():
            continue

        try:
            corrected_srt = correct_srt_content(
                srt_content=srt_content,
                api_key=eff_api_key,
                base_url=eff_base_url,
                model=effective_model
            )
        except Exception as e:
            file_name = os.path.basename(srt_path)
            raise gr.Error(format_llm_error(e, f"AI auto correction ({file_name})"))

        base_name = os.path.splitext(os.path.basename(srt_path))[0]
        name_count = seen_names.get(base_name, 0)
        seen_names[base_name] = name_count + 1
        safe_name = base_name if name_count == 0 else f"{base_name}_{name_count + 1}"

        original_output_path = os.path.join(temp_dir, f"{safe_name}.srt")
        corrected_output_path = os.path.join(temp_dir, f"{safe_name}_ai_corrected.srt")

        with open(original_output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        with open(corrected_output_path, 'w', encoding='utf-8') as f:
            f.write(corrected_srt)

        original_paths.append(original_output_path)
        corrected_paths.append(corrected_output_path)
        last_original = srt_content
        last_corrected = corrected_srt

    if not corrected_paths:
        raise gr.Error("All provided SRT files are empty.")

    diff_html = generate_diff_html(last_original, last_corrected)
    original_download = build_download_path(original_paths, "original_srts_for_correction.zip")
    corrected_download = build_download_path(corrected_paths, "ai_corrected_srts.zip")
    elapsed = time.time() - start_time
    status_msg = f"✅ AI correction completed for {len(corrected_paths)} file(s) in {elapsed:.2f}s"
    return last_original, last_corrected, diff_html, original_download, corrected_download, status_msg


def safe_translate_traditional_wrapper(srt_files):
    """Safe wrapper for step-1 Traditional translation with UI-friendly status."""
    try:
        preview_original, preview_translated, download_path, translator_state = translate_srt_to_traditional(srt_files)
        status_msg = "✅ Translation to Traditional Chinese completed."
        return preview_original, preview_translated, download_path, translator_state, status_msg
    except gr.Error as e:
        return gr.update(), gr.update(), gr.update(), {}, f"❌ {_normalize_error_message(e)}"
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), {}, f"❌ {format_llm_error(e, 'Traditional Chinese translation')}"


def safe_translate_english_wrapper(srt_files, api_key, model_name, custom_model, base_url):
    """Safe wrapper for English translation with UI-friendly status."""
    try:
        preview_original, preview_translated, download_path, translator_state = translate_srt_to_english_fn(
            srt_files, api_key, model_name, custom_model, base_url
        )
        status_msg = "✅ English translation completed."
        return preview_original, preview_translated, download_path, translator_state, status_msg
    except gr.Error as e:
        return gr.update(), gr.update(), gr.update(), {}, f"❌ {_normalize_error_message(e)}"
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), {}, f"❌ {format_llm_error(e, 'English translation')}"


def update_translated_output_hint(translator_state):
    """Show what stream output is available for translator AI correction."""
    if not translator_state or not translator_state.get("latest_output_paths"):
        return gr.update(value="No translated output cached yet. Run translation above or switch to upload mode.")

    kind = translator_state.get("latest_output_kind", "translated")
    count = len(translator_state.get("latest_output_paths", []))
    return gr.update(value=f"Using latest {kind} output from above ({count} file(s)).")


with gr.Blocks(
    title="FunClip Pro - Gradio Edition",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple"
    ),
    css="""
    /* Make entire download file row clickable */
    .download-file tr.file {
        cursor: pointer !important;
    }
    .download-file tr.file:hover {
        background-color: var(--block-background-fill) !important;
    }
    .download-file tr.file td.download a {
        position: absolute !important;
        inset: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        padding-right: var(--size-2-5) !important;
        z-index: 1 !important;
    }
    .download-file tr.file {
        position: relative !important;
    }
    """
) as demo:
    
    # Header
    gr.Markdown(
        """
        # ✂️ FunClip Service
        ### AI-Powered Audio/Video Transcription & Subtitle Generation
        """
    )
    
    # Model Status
    gr.Markdown("✅ **AI Model Ready**")
    
    # Per-session state to store filename and SRT (isolated per user session)
    session_state = gr.State(value={})
    
    with gr.Tabs():
        # --- TAB 1: TRANSCRIPTION ---
        with gr.Tab("🎬 Transcription"):
            # --- MAIN PROCESSING SECTION ---
            with gr.Row():
                # Left Column: Input
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload Media")
                    input_file = gr.File(
                        label="Audio/Video File",
                        file_types=["audio", "video"],
                        file_count="single"
                    )
                    
                    # Media Preview
                    with gr.Group():
                        gr.Markdown("**Preview**")
                        video_preview = gr.Video(
                            label="Video Preview",
                            visible=False,
                            height=300
                        )
                        audio_preview = gr.Audio(
                            label="Audio Preview",
                            visible=False
                        )
                    
                    process_btn = gr.Button(
                        "🚀 Start Processing",
                        variant="primary",
                        size="lg"
                    )
                    
                    # Status Display
                    status_display = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False,
                        lines=1
                    )
                
                # Right Column: Results
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Results")
                    
                    gr.Markdown("**Recognized Text**")
                    output_text = gr.Markdown(
                        value="*Transcription will appear here...*",
                        elem_id="recognized-text"
                    )
                    
                    with gr.Group():
                        output_srt = gr.TextArea(
                            label="SRT Subtitles",
                            interactive=False,
                            lines=8,
                            placeholder="SRT subtitles will appear here..."
                        )
                        download_srt = gr.File(
                            label="📥 Download SRT",
                            interactive=False,
                            elem_classes=["download-file"]
                        )
            
            # Update preview based on file type
            def update_preview(file_path):
                if not file_path:
                    return gr.update(visible=False, value=None), gr.update(visible=False, value=None)
                
                _, ext = os.path.splitext(file_path)
                if ext.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
                    return gr.update(visible=True, value=file_path), gr.update(visible=False, value=None)
                else:
                    return gr.update(visible=False, value=None), gr.update(visible=True, value=file_path)
            
            input_file.change(
                fn=update_preview,
                inputs=[input_file],
                outputs=[video_preview, audio_preview]
            )
            
            # Note: process_btn click handler is connected below, after correct_btn is defined
            
            # --- LLM CORRECTION SECTION (Same page, below results) ---
            gr.Markdown("---")  # Divider
            gr.Markdown("## 🤖 AI Auto Correction")
            gr.Markdown("Use LLM to automatically fix typos, recognition errors, and improve subtitle quality.")
            gr.Markdown("Input can come from transcription output above or from an uploaded SRT file.")

            transcription_correction_source = gr.Radio(
                choices=["Use output from previous step", "Upload SRT file(s)"],
                value="Use output from previous step",
                label="Correction Input Source"
            )
            transcription_correction_upload = gr.File(
                label="Upload SRT File(s) for Correction",
                file_types=[".srt"],
                file_count="single",
                visible=False
            )

            transcription_correction_source.change(
                fn=update_srt_upload_visibility,
                inputs=[transcription_correction_source],
                outputs=[transcription_correction_upload]
            )
            
            # LLM Settings in collapsible accordion
            with gr.Accordion("⚙️ LLM Settings", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        api_key_input = gr.Textbox(
                            label="API Key (OpenAI/Compatible)",
                            placeholder="sk-... (leave empty to use system key)",
                            type="password"
                        )
                        api_key_status = gr.Markdown(get_api_key_status())
                    
                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            choices=["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash", "Custom"],
                            value="gpt-4o-mini",
                            label="Model",
                            allow_custom_value=False
                        )
                        custom_model_input = gr.Textbox(
                            label="Custom Model Name",
                            placeholder="e.g., claude-3-haiku-20240307",
                            visible=False
                        )
                
                base_url_input = gr.Textbox(
                    label="Base URL (Optional)",
                    placeholder="e.g., https://api.moonshot.cn/v1",
                    value=os.getenv("OPENAI_BASE_URL", "")
                )
            
            # Show/hide custom model input
            model_dropdown.change(
                fn=update_custom_model_visibility,
                inputs=[model_dropdown],
                outputs=[custom_model_input]
            )
            
            correct_btn = gr.Button(
                "✨ Run Auto Correction",
                variant="primary",
                size="lg",
                interactive=True
            )
            
            # Now connect the process_btn click handler (after correct_btn is defined)
            process_btn.click(
                fn=lambda: gr.update(interactive=False, value="⏳ Processing..."),
                outputs=[process_btn]
            ).then(
                fn=process_media,
                inputs=[input_file, session_state],
                outputs=[output_text, output_srt, download_srt, status_display, session_state]
            ).then(
                fn=lambda: gr.update(interactive=True, value="🚀 Start Processing"),
                outputs=[process_btn]
            )
            
            # Results Section (only shows after correction is run)
            with gr.Group(visible=False) as correction_results:
                gr.Markdown("### 📊 Correction Results")
                
                correction_status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                    lines=1
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Original**")
                        original_display = gr.TextArea(
                            label="Original SRT",
                            interactive=False,
                            lines=10
                        )
                        download_original = gr.File(
                            label="📥 Download Original",
                            interactive=False,
                            elem_classes=["download-file"]
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("**Corrected**")
                        corrected_display = gr.TextArea(
                            label="Corrected SRT",
                            interactive=False,
                            lines=10
                        )
                        with gr.Row():
                            download_corrected = gr.File(
                                label="📥 Download Corrected",
                                interactive=False,
                                elem_classes=["download-file"]
                            )
                            download_traditional = gr.File(
                                label="📥 Download Corrected (繁體)",
                                interactive=False,
                                elem_classes=["download-file"]
                            )
                
                # Diff View
                with gr.Accordion("🔍 Detailed Diff View", open=True):
                    diff_view = gr.HTML()
            
            # Function to run correction and show results
            def run_correction_and_show(api_key, model_name, custom_model, base_url, state, source_choice, uploaded_srt):
                stream_srt = state.get("original_srt", "") if state else ""
                stream_base_name = state.get("original_filename", "subtitles") if state else "subtitles"

                original_srt, base_name = resolve_stream_or_upload_srt(
                    source_choice=source_choice,
                    stream_content=stream_srt,
                    stream_base_name=stream_base_name,
                    upload_srt_file=uploaded_srt,
                    stream_error_message="No transcription SRT found. Process a media file first or upload SRT file(s)."
                )

                original, corrected, diff_html, orig_path, corr_path, trad_path, status_msg = run_llm_correction_for_content(
                    original_srt=original_srt,
                    base_name=base_name,
                    api_key=api_key,
                    model_name=model_name,
                    custom_model=custom_model,
                    base_url=base_url
                )
                
                # Return results and make results group visible
                return (
                    gr.update(visible=True),  # Show results group
                    status_msg,
                    original,
                    corrected,
                    diff_html,
                    orig_path,
                    corr_path,
                    trad_path
                )
            
            # Connect LLM Logic with button state management and error handling
            def safe_correction_wrapper(api_key, model_name, custom_model, base_url, state, source_choice, uploaded_srt):
                """Wrapper that catches errors and returns them along with a flag."""
                try:
                    result = run_correction_and_show(
                        api_key, model_name, custom_model, base_url, state, source_choice, uploaded_srt
                    )
                    return result
                except gr.Error as e:
                    return (
                        gr.update(visible=True),
                        f"❌ {_normalize_error_message(e)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update()
                    )
                except Exception as e:
                    return (
                        gr.update(visible=True),
                        f"❌ {format_llm_error(e, 'AI auto correction')}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update()
                    )
            
            correct_btn.click(
                fn=lambda: gr.update(interactive=False, value="⏳ Correcting..."),
                outputs=[correct_btn]
            ).then(
                fn=safe_correction_wrapper,
                inputs=[
                    api_key_input,
                    model_dropdown,
                    custom_model_input,
                    base_url_input,
                    session_state,
                    transcription_correction_source,
                    transcription_correction_upload
                ],
                outputs=[correction_results, correction_status, original_display, corrected_display, diff_view, download_original, download_corrected, download_traditional]
            ).then(
                fn=lambda: gr.update(interactive=True, value="✨ Run Auto Correction"),
                outputs=[correct_btn]
            )
        
        # --- TAB 2: SRT TRANSLATOR ---
        with gr.Tab("🔤 SRT Translator"):
            gr.Markdown("### 📄 SRT Translation Tools")
            gr.Markdown("Upload one or more SRT files in Simplified Chinese and translate them to Traditional Chinese or English.")
            translator_state = gr.State(value={})

            with gr.Row():
                # Left Column: Upload & Actions
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload SRT File")
                    srt_input_file = gr.File(
                        label="SRT File(s)",
                        file_types=[".srt"],
                        file_count="multiple"
                    )

                    translate_btn = gr.Button(
                        "🔄 Translate to Traditional Chinese (繁體)",
                        variant="primary",
                        size="lg"
                    )

                    gr.Markdown("---")

                    translate_en_btn = gr.Button(
                        "🌐 Translate to English",
                        variant="primary",
                        size="lg"
                    )

                    translator_status = gr.Textbox(
                        label="Translation Status",
                        value="Ready",
                        interactive=False,
                        lines=2
                    )

                    with gr.Accordion("⚙️ LLM Settings (for English translation)", open=False):
                        srt_api_key_input = gr.Textbox(
                            label="API Key",
                            placeholder="sk-... (leave empty to use system key)",
                            type="password"
                        )
                        srt_model_dropdown = gr.Dropdown(
                            choices=["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash", "Custom"],
                            value="gpt-4o-mini",
                            label="Model",
                            allow_custom_value=False
                        )
                        srt_custom_model_input = gr.Textbox(
                            label="Custom Model Name",
                            placeholder="e.g., claude-3-haiku-20240307",
                            visible=False
                        )
                        srt_base_url_input = gr.Textbox(
                            label="Base URL (Optional)",
                            placeholder="e.g., https://api.moonshot.cn/v1",
                            value=os.getenv("OPENAI_BASE_URL", "")
                        )

                    srt_model_dropdown.change(
                        fn=update_custom_model_visibility,
                        inputs=[srt_model_dropdown],
                        outputs=[srt_custom_model_input]
                    )

                    gr.Markdown("### 📥 Download")
                    download_translated_srt = gr.File(
                        label="Download Translated SRT",
                        interactive=False,
                        elem_classes=["download-file"]
                    )

                # Right Column: Preview
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Preview")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("**Original (简体)**")
                            original_srt_preview = gr.TextArea(
                                label="Original SRT",
                                interactive=False,
                                lines=15,
                                placeholder="Original content will appear here..."
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("**Translated**")
                            translated_srt_preview = gr.TextArea(
                                label="Translated SRT",
                                interactive=False,
                                lines=15,
                                placeholder="Translated content will appear here..."
                            )

            # --- TRANSLATOR AI CORRECTION SECTION ---
            gr.Markdown("---")
            gr.Markdown("## 🤖 AI Auto Correction")
            gr.Markdown("Use translated output from above, or upload SRT file(s) directly for correction.")

            translator_correction_source = gr.Radio(
                choices=["Use output from previous step", "Upload SRT file(s)"],
                value="Use output from previous step",
                label="Correction Input Source"
            )

            translator_stream_hint = gr.Textbox(
                label="Stream Input",
                value="No translated output cached yet. Run translation above or switch to upload mode.",
                interactive=False,
                lines=2
            )

            translator_correction_upload = gr.File(
                label="Upload SRT File(s) for Correction",
                file_types=[".srt"],
                file_count="multiple",
                visible=False
            )

            translator_correction_source.change(
                fn=update_srt_upload_visibility,
                inputs=[translator_correction_source],
                outputs=[translator_correction_upload]
            )

            with gr.Accordion("⚙️ LLM Settings (for AI correction)", open=False):
                translator_corr_api_key_input = gr.Textbox(
                    label="API Key",
                    placeholder="sk-... (leave empty to use system key)",
                    type="password"
                )
                translator_corr_model_dropdown = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash", "Custom"],
                    value="gpt-4o-mini",
                    label="Model",
                    allow_custom_value=False
                )
                translator_corr_custom_model_input = gr.Textbox(
                    label="Custom Model Name",
                    placeholder="e.g., claude-3-haiku-20240307",
                    visible=False
                )
                translator_corr_base_url_input = gr.Textbox(
                    label="Base URL (Optional)",
                    placeholder="e.g., https://api.moonshot.cn/v1",
                    value=os.getenv("OPENAI_BASE_URL", "")
                )

            translator_corr_model_dropdown.change(
                fn=update_custom_model_visibility,
                inputs=[translator_corr_model_dropdown],
                outputs=[translator_corr_custom_model_input]
            )

            translator_correct_btn = gr.Button(
                "✨ Run Auto Correction",
                variant="primary",
                size="lg"
            )

            with gr.Group(visible=False) as translator_correction_results:
                gr.Markdown("### 📊 Correction Results")
                translator_correction_status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                    lines=1
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Original**")
                        translator_original_display = gr.TextArea(
                            label="Original SRT",
                            interactive=False,
                            lines=10
                        )
                        translator_download_original = gr.File(
                            label="📥 Download Original",
                            interactive=False,
                            elem_classes=["download-file"]
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("**Corrected**")
                        translator_corrected_display = gr.TextArea(
                            label="Corrected SRT",
                            interactive=False,
                            lines=10
                        )
                        translator_download_corrected = gr.File(
                            label="📥 Download Corrected",
                            interactive=False,
                            elem_classes=["download-file"]
                        )

                with gr.Accordion("🔍 Detailed Diff View", open=True):
                    translator_diff_view = gr.HTML()

            # Connect Traditional Chinese translate button
            translate_btn.click(
                fn=lambda: (
                    gr.update(interactive=False, value="⏳ Translating..."),
                    gr.update(value="⏳ Translating to Traditional Chinese...")
                ),
                outputs=[translate_btn, translator_status]
            ).then(
                fn=safe_translate_traditional_wrapper,
                inputs=[srt_input_file],
                outputs=[original_srt_preview, translated_srt_preview, download_translated_srt, translator_state, translator_status]
            ).then(
                fn=lambda: gr.update(interactive=True, value="🔄 Translate to Traditional Chinese (繁體)"),
                outputs=[translate_btn]
            ).then(
                fn=update_translated_output_hint,
                inputs=[translator_state],
                outputs=[translator_stream_hint]
            )

            # Connect English translate button
            translate_en_btn.click(
                fn=lambda: (
                    gr.update(interactive=False, value="⏳ Translating to English..."),
                    gr.update(value="⏳ Translating to English...")
                ),
                outputs=[translate_en_btn, translator_status]
            ).then(
                fn=safe_translate_english_wrapper,
                inputs=[srt_input_file, srt_api_key_input, srt_model_dropdown, srt_custom_model_input, srt_base_url_input],
                outputs=[original_srt_preview, translated_srt_preview, download_translated_srt, translator_state, translator_status]
            ).then(
                fn=lambda: gr.update(interactive=True, value="🌐 Translate to English"),
                outputs=[translate_en_btn]
            ).then(
                fn=update_translated_output_hint,
                inputs=[translator_state],
                outputs=[translator_stream_hint]
            )

            # Function to run translator correction and show results
            def run_translator_correction_and_show(source_choice, uploaded_srt_files, state, api_key, model_name, custom_model, base_url):
                correction_files = resolve_translator_correction_files(
                    source_choice=source_choice,
                    translator_state=state,
                    uploaded_srt_files=uploaded_srt_files
                )
                original, corrected, diff_html, orig_download, corr_download, status_msg = run_llm_correction_for_files(
                    srt_files=correction_files,
                    api_key=api_key,
                    model_name=model_name,
                    custom_model=custom_model,
                    base_url=base_url
                )
                return (
                    gr.update(visible=True),
                    status_msg,
                    original,
                    corrected,
                    diff_html,
                    orig_download,
                    corr_download
                )

            def safe_translator_correction_wrapper(source_choice, uploaded_srt_files, state, api_key, model_name, custom_model, base_url):
                try:
                    return run_translator_correction_and_show(
                        source_choice, uploaded_srt_files, state, api_key, model_name, custom_model, base_url
                    )
                except gr.Error as e:
                    return (
                        gr.update(visible=True),
                        f"❌ {_normalize_error_message(e)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update()
                    )
                except Exception as e:
                    return (
                        gr.update(visible=True),
                        f"❌ {format_llm_error(e, 'Translator AI auto correction')}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update()
                    )

            translator_correct_btn.click(
                fn=lambda: gr.update(interactive=False, value="⏳ Correcting..."),
                outputs=[translator_correct_btn]
            ).then(
                fn=safe_translator_correction_wrapper,
                inputs=[
                    translator_correction_source,
                    translator_correction_upload,
                    translator_state,
                    translator_corr_api_key_input,
                    translator_corr_model_dropdown,
                    translator_corr_custom_model_input,
                    translator_corr_base_url_input
                ],
                outputs=[
                    translator_correction_results,
                    translator_correction_status,
                    translator_original_display,
                    translator_corrected_display,
                    translator_diff_view,
                    translator_download_original,
                    translator_download_corrected
                ]
            ).then(
                fn=lambda: gr.update(interactive=True, value="✨ Run Auto Correction"),
                outputs=[translator_correct_btn]
            )

    # Footer
    gr.Markdown(
        """
        ---
        **FunClip Pro** - Powered by FunASR & Gradio | 
        [GitHub](https://github.com/alibaba-damo-academy/FunClip)
        """
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=["/tmp", tempfile.gettempdir()])
