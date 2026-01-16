import gradio as gr
import os
import sys
import time
import soundfile as sf
import difflib
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile, TemporaryDirectory
import tempfile

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
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- 2. LOAD MODEL (Global Scope) ---
print("Loading FunASR Model...")
funasr_model = AutoModel(
    model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
)
print("✅ AI Model Ready")

# --- 3. GLOBAL STATE ---
# Store the original filename and SRT for use in corrections
current_state = {
    "original_filename": None,
    "original_srt": None
}

# --- 4. PROCESSING FUNCTIONS ---

def process_media(file_path, progress=gr.Progress()):
    """Process audio/video file and return transcription results."""
    if not file_path:
        gr.Warning("Please upload a file first.")
        return None, None, None, "No file uploaded"
    
    start_time = time.time()
    
    # Get original filename for later use
    original_name = os.path.basename(file_path)
    base_name = os.path.splitext(original_name)[0]
    current_state["original_filename"] = base_name
    
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
    
    # Store SRT for correction feature
    current_state["original_srt"] = res_srt
    
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
    
    return formatted_text, res_srt, srt_path, status_msg


def get_api_key_status():
    """Check if system API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return "✅ System API Key detected"
    return "⚠️ No System API Key found"


def run_llm_correction(original_srt, api_key, model_name, custom_model, base_url, progress=gr.Progress()):
    """Run LLM-based SRT correction."""
    if not original_srt or not original_srt.strip():
        raise gr.Error("No SRT content found. Please process a video first.")
    
    # Handle model selection
    effective_model = custom_model if model_name == "Custom" else model_name
    
    # Handle ENV variables if input is empty
    eff_api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
    eff_base_url = base_url if base_url else os.getenv("OPENAI_BASE_URL")
    
    if not eff_api_key:
        raise gr.Error("No API Key provided. Please enter an API key or set OPENAI_API_KEY in your .env file.")
    
    try:
        corrected_srt = correct_srt_content(
            srt_content=original_srt,
            api_key=eff_api_key,
            base_url=eff_base_url,
            model=effective_model
        )
    except Exception as e:
        raise gr.Error(f"LLM Error: {str(e)}")
    
    # Generate HTML Diff
    diff_html = generate_diff_html(original_srt, corrected_srt)
    
    # Convert corrected SRT to Traditional Chinese
    traditional_srt = convert_to_traditional(corrected_srt)
    
    # Save SRT files to system temp directory for Gradio compatibility
    base_name = current_state.get("original_filename", "subtitles")
    temp_dir = tempfile.gettempdir()
    
    orig_filename = f"{base_name}.srt"
    orig_path = os.path.join(temp_dir, orig_filename)
    with open(orig_path, 'w', encoding='utf-8') as f:
        f.write(original_srt)
    
    # Save Corrected SRT (Simplified) to temp file
    corr_filename = f"corrected_{base_name}.srt"
    corr_path = os.path.join(temp_dir, corr_filename)
    with open(corr_path, 'w', encoding='utf-8') as f:
        f.write(corrected_srt)
    
    # Save Corrected SRT (Traditional) to temp file
    trad_filename = f"corrected_{base_name}_traditional.srt"
    trad_path = os.path.join(temp_dir, trad_filename)
    with open(trad_path, 'w', encoding='utf-8') as f:
        f.write(traditional_srt)
    
    return original_srt, corrected_srt, diff_html, orig_path, corr_path, trad_path


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

def translate_srt_to_traditional(srt_file):
    """Translate an SRT file to Traditional Chinese."""
    if not srt_file:
        raise gr.Error("Please upload an SRT file first.")
    
    # Read the SRT content
    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    if not srt_content.strip():
        raise gr.Error("The uploaded SRT file is empty.")
    
    # Convert to Traditional Chinese
    traditional_srt = convert_to_traditional(srt_content)
    
    # Save to temp file with proper filename
    original_name = os.path.basename(srt_file)
    base_name = os.path.splitext(original_name)[0]
    temp_dir = tempfile.gettempdir()
    
    trad_filename = f"{base_name}_traditional.srt"
    trad_path = os.path.join(temp_dir, trad_filename)
    with open(trad_path, 'w', encoding='utf-8') as f:
        f.write(traditional_srt)
    
    return srt_content, traditional_srt, trad_path


with gr.Blocks(
    title="FunClip Pro - Gradio Edition",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple"
    )
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
                            interactive=False
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
            
            # Hidden state to store SRT for correction (auto-filled from transcription)
            correction_source_srt = gr.State(value="")
            
            correct_btn = gr.Button(
                "✨ Run Auto Correction",
                variant="primary",
                size="lg",
                interactive=False  # Disabled until SRT is ready
            )
            
            # Now connect the process_btn click handler (after correct_btn is defined)
            process_btn.click(
                fn=lambda: gr.update(interactive=False, value="⏳ Processing..."),
                outputs=[process_btn]
            ).then(
                fn=process_media,
                inputs=[input_file],
                outputs=[output_text, output_srt, download_srt, status_display]
            ).then(
                fn=lambda: (gr.update(interactive=True, value="🚀 Start Processing"), gr.update(interactive=True)),
                outputs=[process_btn, correct_btn]
            )
            
            # Results Section (only shows after correction is run)
            with gr.Group(visible=False) as correction_results:
                gr.Markdown("### 📊 Correction Results")
                
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
                            interactive=False
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
                                interactive=False
                            )
                            download_traditional = gr.File(
                                label="📥 Download Corrected (繁體)",
                                interactive=False
                            )
                
                # Diff View
                with gr.Accordion("🔍 Detailed Diff View", open=True):
                    diff_view = gr.HTML()
            
            # Function to run correction and show results
            def run_correction_and_show(api_key, model_name, custom_model, base_url):
                # Use stored SRT from transcription
                original_srt = current_state.get("original_srt", "")
                
                if not original_srt:
                    raise gr.Error("No SRT content found. Please process a media file first.")
                
                original, corrected, diff_html, orig_path, corr_path, trad_path = run_llm_correction(
                    original_srt, api_key, model_name, custom_model, base_url
                )
                
                # Return results and make results group visible
                return (
                    gr.update(visible=True),  # Show results group
                    original,
                    corrected,
                    diff_html,
                    orig_path,
                    corr_path,
                    trad_path
                )
            
            # Connect LLM Logic with button state management and error handling
            def safe_correction_wrapper(api_key, model_name, custom_model, base_url):
                """Wrapper that catches errors and returns them along with a flag."""
                try:
                    result = run_correction_and_show(api_key, model_name, custom_model, base_url)
                    return result
                except gr.Error:
                    # Re-raise Gradio errors to show in UI
                    raise
                except Exception as e:
                    raise gr.Error(f"Correction failed: {str(e)}")
            
            correct_btn.click(
                fn=lambda: gr.update(interactive=False, value="⏳ Correcting..."),
                outputs=[correct_btn]
            ).then(
                fn=safe_correction_wrapper,
                inputs=[api_key_input, model_dropdown, custom_model_input, base_url_input],
                outputs=[correction_results, original_display, corrected_display, diff_view, download_original, download_corrected, download_traditional]
            ).then(
                fn=lambda: gr.update(interactive=True, value="✨ Run Auto Correction"),
                outputs=[correct_btn]
            )
        
        # --- TAB 2: SRT TRANSLATOR ---
        with gr.Tab("🔤 SRT Translator"):
            gr.Markdown("### 📄 Translate SRT to Traditional Chinese (繁體中文)")
            gr.Markdown("Upload an SRT file in Simplified Chinese and convert it to Traditional Chinese.")
            
            with gr.Row():
                # Left Column: Upload
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload SRT File")
                    srt_input_file = gr.File(
                        label="SRT File",
                        file_types=[".srt"],
                        file_count="single"
                    )
                    
                    translate_btn = gr.Button(
                        "🔄 Translate to Traditional Chinese",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown("### 📥 Download")
                    download_translated_srt = gr.File(
                        label="Download Translated SRT (繁體)",
                        interactive=False
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
                            gr.Markdown("**Translated (繁體)**")
                            translated_srt_preview = gr.TextArea(
                                label="Translated SRT",
                                interactive=False,
                                lines=15,
                                placeholder="Translated content will appear here..."
                            )
            
            # Connect translate button
            translate_btn.click(
                fn=lambda: gr.update(interactive=False, value="⏳ Translating..."),
                outputs=[translate_btn]
            ).then(
                fn=translate_srt_to_traditional,
                inputs=[srt_input_file],
                outputs=[original_srt_preview, translated_srt_preview, download_translated_srt]
            ).then(
                fn=lambda: gr.update(interactive=True, value="🔄 Translate to Traditional Chinese"),
                outputs=[translate_btn]
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
