import streamlit as st
import os
import sys
import time
import threading
import soundfile as sf
import streamlit.components.v1 as components
from dotenv import load_dotenv
from tempfile import TemporaryDirectory

load_dotenv()

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
funclip_dir = os.path.join(current_dir, "funclip")
if funclip_dir not in sys.path:
    sys.path.append(funclip_dir)

# --- 2. IMPORT MODULES ---
try:
    from funclip.videoclipper import VideoClipper
    from funasr import AutoModel
    from funclip.llm.srt_corrector import correct_srt_content
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# --- 3. HELPER: RUN IN BACKGROUND ---
class AsyncTask:
    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.result = None
        self.error = None
        self.finished = False
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
        try:
            self.result = self.target(*self.args)
        except Exception as e:
            self.error = e
        finally:
            self.finished = True

# --- 4. LOAD MODEL (CACHED) ---
@st.cache_resource(show_spinner="Loading AI Models... (First run takes ~30s)")
def load_models():
    print("Loading FunASR Model...")
    return AutoModel(
        model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
    )

# --- 5. HELPER: RESET STATE ---
def reset_state():
    """Clears results if a new file is uploaded."""
    keys_to_clear = ['res_text', 'res_srt', 'srt_filename', 'processing_done', 'res_srt_corrected']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# --- 6. MAIN APP ---
st.set_page_config(page_title="FunClip Pro", page_icon="✂️", layout="wide")
st.title("FunClip Service ✂️")

try:
    funasr_model = load_models()
    st.success("✅ AI Model Ready")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Attach callback to clear results when file changes
uploaded_file = st.file_uploader(
    "Upload Audio/Video", 
    type=["mp3", "wav", "mp4", "m4a", "mov"],
    on_change=reset_state
)

if uploaded_file is not None:
    # --- MEDIA PREVIEW ---
    # Display audio or video player based on file type
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type in ['mp4', 'mov', 'avi', 'mkv']:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)

    status_container = st.container()
    
    # --- PROCESSING BUTTON ---
    if st.button("Start Processing", type="primary"):
        
        # Reset state explicitly on new run
        reset_state()
        
        with TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, uploaded_file.name)
            
            # --- PHASE 1: PREPARATION ---
            with status_container.status("📂 Preparing file...", expanded=True) as status:
                st.write("Saving uploaded file...")
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    audio_info = sf.info(input_path)
                    duration_sec = audio_info.duration
                    st.write(f"**Audio Duration:** {duration_sec:.2f} seconds")
                except:
                    duration_sec = 0
                    st.write("Could not determine duration.")

                st.write("Initializing Clipper...")
                audio_clipper = VideoClipper(funasr_model)
                audio_clipper.lang = 'zh'
                
                status.update(label="✅ Preparation Complete", state="complete", expanded=False)

            # --- PHASE 2: INFERENCE ---
            timer_placeholder = st.empty()
            
            def run_inference():
                _, ext = os.path.splitext(uploaded_file.name)
                if ext.lower() in ['.mp4', '.avi', '.mkv', '.mov']:
                    return audio_clipper.video_recog(input_path, sd_switch='no', output_dir=temp_dir)
                else:
                    import librosa
                    wav, sr = librosa.load(input_path, sr=16000)
                    return audio_clipper.recog((sr, wav), sd_switch='no')

            start_time = time.time()
            task = AsyncTask(run_inference, ())

            while not task.finished:
                elapsed = time.time() - start_time
                timer_placeholder.metric(label="⏳ Processing Time", value=f"{elapsed:.1f} s", delta="Running...")
                time.sleep(0.1)
            
            # --- PHASE 3: FINISHED ---
            end_time = time.time()
            total_time = end_time - start_time
            
            if task.error:
                timer_placeholder.error(f"Error: {task.error}")
                st.error(task.error)
            else:
                # Save results to Session State so they persist!
                res_text, res_srt, state = task.result
                base_name = os.path.splitext(uploaded_file.name)[0]
                
                st.session_state['res_text'] = res_text
                st.session_state['res_srt'] = res_srt
                st.session_state['srt_filename'] = f"{base_name}.srt"
                st.session_state['processing_done'] = True
                
                # Show speed stats briefly
                speed_x = duration_sec / total_time if total_time > 0 else 0
                timer_placeholder.metric(
                    label="✅ Finished In", 
                    value=f"{total_time:.2f} s", 
                    delta=f"{speed_x:.1f}x Speed"
                )

    # --- DISPLAY RESULT (Outside the button block) ---
    # This block runs even after you click download, because it checks session_state
    if st.session_state.get('processing_done'):
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recognized Text")
            st.text_area("Content", st.session_state['res_text'], height=400)
        
        with col2:
            st.subheader("SRT Subtitles")
            filename = st.session_state['srt_filename']
            
            # Download button right above the preview
            st.download_button(
                label=f"⬇️ Download {filename}",
                data=st.session_state['res_srt'],
                file_name=filename,
                mime="text/plain",
                type="primary"
            )
            
            st.text_area("SRT Content", st.session_state['res_srt'], height=350, label_visibility="collapsed")

        # --- 7. AI AUTO CORRECTION ---
        st.divider()
        st.header("🤖 AI Auto Correction")
        
        with st.expander("LLM Settings", expanded=False):
            api_key_env = os.getenv("OPENAI_API_KEY")
            base_url_env = os.getenv("OPENAI_BASE_URL", "")
            
            c1, c2 = st.columns(2)
            with c1:
                # Do not pre-fill value with env var to avoid leaking it in the UI
                help_text = "Enter your own key to override the system default."
                if api_key_env:
                    help_text += " (System key is currently active)"
                
                api_key_input = st.text_input(
                    "API Key (OpenAI/Compatible)", 
                    value="", 
                    type="password", 
                    help=help_text, 
                    placeholder="sk-..."
                )
                
                # Visual indicator if system key is available
                if api_key_env and not api_key_input:
                    st.caption("✅ System API Key detected")
                elif not api_key_env and not api_key_input:
                    st.caption("⚠️ No System API Key found")

            with c2:
                model_options = ["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash", "Custom"]
                selected_model = st.selectbox("Model Name", options=model_options, index=0, help="Select the LLM model to use")
                
                if selected_model == "Custom":
                    model_input = st.text_input("Enter Custom Model Name", value="gpt-4o-mini")
                else:
                    model_input = selected_model
            
            base_url_input = st.text_input("Base URL (Optional)", value=base_url_env, help="e.g. https://api.moonshot.cn/v1")

        if st.button("✨ Run Auto Correction"):
            target_srt = st.session_state['res_srt']
            # Use user input if provided, otherwise fall back to env (handled by litellm/backend)
            effective_api_key = api_key_input if api_key_input else None
            effective_base_url = base_url_input if base_url_input else None
            
            # Simple validation
            if not effective_api_key and not api_key_env:
                st.warning("⚠️ No API Key detected in .env or input field. The request might fail unless your provider doesn't need one.")

            with st.spinner("🤖 AI is correcting subtitles... This may take a moment."):
                try:
                    corrected_text = correct_srt_content(
                        srt_content=target_srt,
                        api_key=effective_api_key,
                        base_url=effective_base_url,
                        model=model_input
                    )
                    st.session_state['res_srt_corrected'] = corrected_text
                    st.success("✅ Correction Complete!")
                except Exception as e:
                    st.error(f"❌ Correction failed: {str(e)}")

        # Display Corrected Results & Diff
        if 'res_srt_corrected' in st.session_state:
            st.divider()
            st.subheader("📝 Correction Results")
            
            # 1. Download Buttons (Side by Side)
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button(
                    label="⬇️ Download Original SRT",
                    data=st.session_state['res_srt'],
                    file_name=st.session_state['srt_filename'],
                    mime="text/plain"
                )
            with d_col2:
                corrected_filename = "corrected_" + st.session_state['srt_filename']
                st.download_button(
                    label="⬇️ Download Corrected SRT",
                    data=st.session_state['res_srt_corrected'],
                    file_name=corrected_filename,
                    mime="text/plain",
                    type="primary"
                )

            # 2. Side-by-Side Text Areas
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.info("Original")
                st.text_area("Original Content", st.session_state['res_srt'], height=400, label_visibility="collapsed")
            with comp_col2:
                st.success("Corrected")
                st.text_area("Corrected Content", st.session_state['res_srt_corrected'], height=400, label_visibility="collapsed")

            # 3. HTML Diff View
            with st.expander("🔍 Detailed Diff View", expanded=True):
                import difflib
                
                # Generate HTML Diff
                original_lines = st.session_state['res_srt'].splitlines()
                corrected_lines = st.session_state['res_srt_corrected'].splitlines()
                
                diff = difflib.HtmlDiff().make_file(
                    original_lines, 
                    corrected_lines, 
                    fromdesc='Original', 
                    todesc='Corrected',
                    context=True,
                    numlines=3
                )
                
                # Custom CSS to make it fit nicely in Streamlit
                # We inject it via html component
                components.html(diff, height=600, scrolling=True)