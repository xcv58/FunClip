import streamlit as st
import os
import sys
import time
import threading
import soundfile as sf
from tempfile import TemporaryDirectory

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
funclip_dir = os.path.join(current_dir, "funclip")
if funclip_dir not in sys.path:
    sys.path.append(funclip_dir)

# --- 2. IMPORT MODULES ---
try:
    from funclip.videoclipper import VideoClipper
    from funasr import AutoModel
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
    keys_to_clear = ['res_text', 'res_srt', 'srt_filename', 'processing_done']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# --- 6. MAIN APP ---
st.set_page_config(page_title="FunClip Pro", page_icon="✂️")
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
            st.text_area("Content", st.session_state['res_text'], height=200)
        
        with col2:
            st.subheader("Download")
            filename = st.session_state['srt_filename']
            st.success(f"Generated: **{filename}**")
            
            st.download_button(
                label=f"⬇️ Download {filename}",
                data=st.session_state['res_srt'],
                file_name=filename,
                mime="text/plain",
                type="primary"
            )
            
            st.divider()
            st.subheader("SRT Preview")
            st.text_area("SRT Content", st.session_state['res_srt'], height=200)