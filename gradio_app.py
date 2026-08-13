import gradio as gr
from gradio.processing_utils import get_upload_folder
import atexit
import base64
import binascii
from contextlib import contextmanager
import os
import sys
import time
import soundfile as sf
import difflib
import math
import re
import shutil
import threading
import uuid
from collections import OrderedDict
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile, TemporaryDirectory
import tempfile
import zipfile
from litellm import get_llm_provider

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
    from funclip.llm.chinese_converter import (
        convert_to_traditional,
        normalize_traditional_for_validation,
    )
    from funclip.llm.srt_translator import translate_srt_to_english
    from funclip.llm.youtube_chapters import (
        ChapterGenerationError,
        MAX_LLM_MODEL_IDENTIFIER_CHARACTERS,
        MAX_SRT_UTF8_BYTES,
        MAX_VIDEO_DURATION_MS,
        generate_youtube_chapters,
        normalize_video_duration_ms,
        parse_and_validate_chapters_text,
        parse_srt,
        render_chapters_text,
        update_latest_traditional_source,
        validate_llm_model_identifier as validate_chapter_model_identifier,
        validate_chinese_transcript,
    )
    from funclip.service_retention import (
        MediaCleanupError,
        UploadPathError,
        cleanup_media_files,
        run_with_media_cleanup,
        validate_gradio_upload_path,
    )
    from funclip.service_contract import no_speech_result, speech_result
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
ASYNC_JOB_TTL_SECONDS = int(os.getenv("FUNCLIP_ASYNC_JOB_TTL_SECONDS", "86400"))
ASYNC_JOBS = {}
ASYNC_JOBS_LOCK = threading.Lock()
ASYNC_JOB_EXEC_LOCK = threading.Lock()
CHAPTER_REVISION_REGISTRY_LIMIT = 2_048
CHAPTER_REVISIONS = OrderedDict()
CHAPTER_REVISIONS_LOCK = threading.Lock()
CHAPTER_SESSION_RUNTIME = OrderedDict()
CHAPTER_SESSION_UPLOADS = OrderedDict()
CHAPTER_SESSION_UPLOAD_LIMIT = 128
CHAPTER_SESSION_UPLOAD_TTL_SECONDS = 24 * 60 * 60
STAGED_CHAPTER_OUTPUTS = OrderedDict()
STAGED_CHAPTER_OUTPUTS_LOCK = threading.Lock()
STAGED_CHAPTER_OUTPUTS_LIMIT = 2_048
STAGED_CHAPTER_OUTPUT_RESERVATIONS = OrderedDict()
CHAPTER_ARTIFACT_TTL_SECONDS = int(
    os.getenv("FUNCLIP_CHAPTER_ARTIFACT_TTL_SECONDS", "3600")
)
CHAPTER_ARTIFACT_REGISTRY_LIMIT = 2_048
CHAPTER_ARTIFACT_CACHE_DIR = get_upload_folder()
CHAPTER_PUBLICATION_CONCURRENCY_ID = "youtube_chapter_publication"
CHAPTER_GENERATION_CONCURRENCY_LIMIT = max(
    1, int(os.getenv("FUNCLIP_CHAPTER_CONCURRENCY_LIMIT", "2"))
)
CHAPTER_QUEUE_MAX_SIZE = 64
CHAPTER_ADMISSION_REGISTRY_LIMIT = CHAPTER_QUEUE_MAX_SIZE * 2
CHAPTER_ADMISSION_TTL_SECONDS = max(
    30, int(os.getenv("FUNCLIP_CHAPTER_ADMISSION_TTL_SECONDS", "300"))
)
CHAPTER_ADMISSIONS = OrderedDict()
CHAPTER_ADMISSIONS_LOCK = threading.Lock()
OWNED_CHAPTER_ARTIFACTS = OrderedDict()
OWNED_CHAPTER_ARTIFACTS_LOCK = threading.Lock()
CHAPTER_ARTIFACT_RESERVATIONS = 0
CHAPTER_ARTIFACT_REAPER_STOP = threading.Event()
PORTABLE_FILENAME_COMPONENT_BYTES = 240
TEMPFILE_RANDOM_SUFFIX_RESERVE_BYTES = 16
MAX_SAFE_ARTIFACT_BASE_BYTES = 160

_CHAPTER_UI_ACTION_ADMISSION = "ui_action"
_CHAPTER_UI_GENERATION_ADMISSION = _CHAPTER_UI_ACTION_ADMISSION
_CHAPTER_API_GENERATION_ADMISSION = "api_generation"
_CHAPTER_EDIT_ADMISSION = _CHAPTER_UI_ACTION_ADMISSION


def _chapter_session_key(request):
    """Return Gradio's opaque per-browser-session key, when available."""
    return str(getattr(request, "session_hash", "") or "").strip()


def _chapter_client_host(request):
    """Return the transport peer used to bound stateless public API callers."""
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if host is None and isinstance(client, (tuple, list)) and client:
        host = client[0]
    return str(host or "").strip()


def _chapter_admission_identity(channel, request):
    """Use server-observed identities, with API calls bounded by transport peer."""
    session_key = _chapter_session_key(request)
    client_host = _chapter_client_host(request)
    if channel == _CHAPTER_API_GENERATION_ADMISSION:
        return f"client:{client_host or session_key or 'anonymous'}"
    return f"session:{session_key or client_host or 'anonymous'}"


def _chapter_request_admission_store(request, *, create=False):
    """Return request-local admission tokens shared by validator and execution."""
    raw_request = getattr(request, "request", None) or request
    state = getattr(raw_request, "state", None)
    owner = state if state is not None else raw_request
    if owner is None:
        return None
    attribute = "_funclip_chapter_admission_tokens"
    store = getattr(owner, attribute, None)
    if store is None and create:
        store = {}
        try:
            setattr(owner, attribute, store)
        except (AttributeError, TypeError):
            return None
    return store if isinstance(store, dict) else None


def _set_chapter_request_admission_token(request, channel, token):
    store = _chapter_request_admission_store(request, create=True)
    if store is None:
        return False
    store[channel] = token
    return True


def _get_chapter_request_admission_token(request, channel):
    store = _chapter_request_admission_store(request)
    if store is None:
        return None
    token = store.get(channel)
    return token if isinstance(token, str) and token else None


def _set_chapter_request_revision(request, channel, revision):
    store = _chapter_request_admission_store(request, create=True)
    if store is None:
        return False
    store[("revision", channel)] = revision
    return True


def _get_chapter_request_revision(request, channel):
    store = _chapter_request_admission_store(request)
    if store is None:
        return None
    revision = store.get(("revision", channel))
    return revision if isinstance(revision, int) and revision >= 0 else None


def _reserve_chapter_admission(channel, request):
    """Reserve one pending chapter action before Gradio appends it to the queue."""
    cleanup_expired_chapter_admissions()
    admission_key = (channel, _chapter_admission_identity(channel, request))
    now = time.time()
    with CHAPTER_ADMISSIONS_LOCK:
        if admission_key in CHAPTER_ADMISSIONS:
            return False
        if len(CHAPTER_ADMISSIONS) >= CHAPTER_ADMISSION_REGISTRY_LIMIT:
            return False
        token = uuid.uuid4().hex
        CHAPTER_ADMISSIONS[admission_key] = {
            "state": "queued",
            "updated_at": now,
            "token": token,
            "session_key": _chapter_session_key(request),
            "client_host": _chapter_client_host(request),
        }
    if _set_chapter_request_admission_token(request, channel, token):
        return True
    release_chapter_admission(channel, request, token)
    return False


def _claim_chapter_admission(channel, request, expected_token=None):
    """Claim a validator reservation, or atomically admit a direct invocation."""
    cleanup_expired_chapter_admissions()
    admission_key = (channel, _chapter_admission_identity(channel, request))
    request_token = expected_token or _get_chapter_request_admission_token(
        request, channel
    )
    now = time.time()
    with CHAPTER_ADMISSIONS_LOCK:
        admission = CHAPTER_ADMISSIONS.get(admission_key)
        if admission is not None:
            if (
                admission["state"] != "queued"
                or not request_token
                or admission["token"] != request_token
            ):
                return None
            admission["state"] = "running"
            admission["updated_at"] = now
            CHAPTER_ADMISSIONS.move_to_end(admission_key)
            return admission["token"]
        if request_token or len(CHAPTER_ADMISSIONS) >= CHAPTER_ADMISSION_REGISTRY_LIMIT:
            return None
        token = uuid.uuid4().hex
        CHAPTER_ADMISSIONS[admission_key] = {
            "state": "running",
            "updated_at": now,
            "token": token,
            "session_key": _chapter_session_key(request),
            "client_host": _chapter_client_host(request),
        }
        _set_chapter_request_admission_token(request, channel, token)
        return token


def release_chapter_admission(channel, request, token=None):
    """Release only the requesting session/client's admission reservation."""
    admission_key = (channel, _chapter_admission_identity(channel, request))
    with CHAPTER_ADMISSIONS_LOCK:
        admission = CHAPTER_ADMISSIONS.get(admission_key)
        if admission is None:
            return False
        if token is not None and admission["token"] != token:
            return False
        if token is None and admission["state"] != "queued":
            return False
        return CHAPTER_ADMISSIONS.pop(admission_key, None) is not None


def release_cancelled_chapter_admissions(admission_tokens=()):
    """Cancel only exact reservations snapshotted before queue mutation."""
    released = 0
    with CHAPTER_ADMISSIONS_LOCK:
        for admission_key, token in set(admission_tokens):
            admission = CHAPTER_ADMISSIONS.get(admission_key)
            if admission is None or admission["token"] != token:
                continue
            if admission["state"] == "running":
                admission["cancelled"] = True
            elif admission["state"] in {"queued", "publishing"}:
                CHAPTER_ADMISSIONS.pop(admission_key, None)
                released += 1
    return released


def _snapshot_queued_chapter_admissions(*, session_key=None, keys=()):
    """Capture immutable cancellable key/token pairs before queue mutation."""
    exact_keys = set(keys)
    snapshot = set()
    with CHAPTER_ADMISSIONS_LOCK:
        for admission_key, admission in CHAPTER_ADMISSIONS.items():
            if admission["state"] not in {"queued", "publishing"}:
                continue
            if admission_key in exact_keys or (
                session_key
                and admission.get("session_key") == session_key
            ):
                snapshot.add((admission_key, admission["token"]))
    return snapshot


def cleanup_expired_chapter_admissions(now=None):
    """Release queue-rejected leases and discard only their exact candidate."""
    current_time = time.time() if now is None else now
    expired_candidates = []
    expired_count = 0
    with CHAPTER_ADMISSIONS_LOCK:
        for admission_key, admission in list(CHAPTER_ADMISSIONS.items()):
            if admission.get("state") not in {"queued", "publishing"}:
                continue
            updated_at = admission.get("updated_at")
            if (
                not isinstance(updated_at, (int, float))
                or isinstance(updated_at, bool)
                or current_time - updated_at < CHAPTER_ADMISSION_TTL_SECONDS
            ):
                continue
            removed = CHAPTER_ADMISSIONS.pop(admission_key, None)
            if removed is None:
                continue
            expired_count += 1
            candidate = removed.get("candidate")
            if isinstance(candidate, dict):
                expired_candidates.append(candidate)
    for candidate in expired_candidates:
        discard_revisioned_output_candidate(candidate)
    return expired_count


def mark_chapter_admission_publishing(channel, request, token, candidate=None):
    """Retain one exact UI lease until its dependent commit is applied."""
    admission_key = (channel, _chapter_admission_identity(channel, request))
    with CHAPTER_ADMISSIONS_LOCK:
        admission = CHAPTER_ADMISSIONS.get(admission_key)
        if (
            admission is None
            or admission["token"] != token
            or admission["state"] != "running"
        ):
            return False
        if admission.get("cancelled"):
            CHAPTER_ADMISSIONS.pop(admission_key, None)
            return False
        admission["state"] = "publishing"
        admission["updated_at"] = time.time()
        if isinstance(candidate, dict):
            admission["candidate"] = {
                "revision": candidate.get("revision"),
                "session_key": candidate.get("session_key"),
                "channel": candidate.get("channel"),
            }
        return True


def _chapter_event_admission_token(event, channel):
    """Read the immutable token carried by this exact Gradio event."""
    return _get_chapter_request_admission_token(event.request, channel)


def _chapter_candidate_from_queue_event(queue, event):
    """Read a server-side staged candidate from a commit/finalizer event."""
    body = getattr(event, "data", None)
    raw_values = getattr(body, "data", None)
    raw_values = raw_values if isinstance(raw_values, list) else []
    for index, component in enumerate(event.fn.inputs):
        value = raw_values[index] if index < len(raw_values) else None
        if getattr(component, "stateful", False) and event.session_hash:
            try:
                session_state = queue.blocks.state_holder[event.session_hash]
                value = session_state[component._id]
            except (KeyError, TypeError, AttributeError):
                value = None
        if isinstance(value, dict) and value.get("admission_token"):
            return value
    return None


def install_chapter_queue_cleanup(queue):
    """Bind admission cleanup to Gradio's queued-event cancellation lifecycle."""
    original_clean_events = queue.clean_events

    async def clean_events_with_chapter_admission(
        *, session_hash=None, event_id=None
    ):
        admission_tokens = set()
        candidates_to_discard = []
        queued_or_active_events = [
            event
            for event_queue in queue.event_queue_per_concurrency_id.values()
            for event in event_queue.queue
        ]
        queued_or_active_events.extend(
            event
            for job in queue.active_jobs
            if job
            for event in job
        )
        for event in queued_or_active_events:
            matches_session = (
                session_hash is not None
                and event.session_hash == session_hash
            )
            matches_event = event_id is not None and event._id == event_id
            if matches_session or matches_event:
                validator = event.fn.validator
                if validator is validate_api_chapter_admission:
                    channel = _CHAPTER_API_GENERATION_ADMISSION
                    identity = (
                        _chapter_client_host(event.request)
                        or event.session_hash
                        or "anonymous"
                    )
                    admission_key = (channel, f"client:{identity}")
                elif validator in {
                    validate_ui_chapter_admission,
                    validate_chapter_edit_admission,
                }:
                    channel = _CHAPTER_UI_ACTION_ADMISSION
                    identity = (
                        event.session_hash
                        or _chapter_client_host(event.request)
                        or "anonymous"
                    )
                    admission_key = (channel, f"session:{identity}")
                elif event.fn.fn in {
                    commit_chapter_generation_outputs,
                    commit_chapter_edit_outputs,
                    finalize_chapter_publication,
                }:
                    channel = _CHAPTER_UI_ACTION_ADMISSION
                    identity = (
                        event.session_hash
                        or _chapter_client_host(event.request)
                        or "anonymous"
                    )
                    admission_key = (channel, f"session:{identity}")
                    candidate = _chapter_candidate_from_queue_event(
                        queue, event
                    )
                    if candidate is not None:
                        candidates_to_discard.append(candidate)
                        token = candidate.get("admission_token")
                        if isinstance(token, str) and token:
                            admission_tokens.add((admission_key, token))
                    continue
                else:
                    continue
                token = _chapter_event_admission_token(event, channel)
                if token:
                    admission_tokens.add((admission_key, token))
        if session_hash is not None:
            # Snapshotting before await prevents delayed session cleanup from
            # releasing a replacement event's token.
            admission_tokens.update(
                _snapshot_queued_chapter_admissions(session_key=session_hash)
            )
        try:
            await original_clean_events(
                session_hash=session_hash, event_id=event_id
            )
        finally:
            release_cancelled_chapter_admissions(admission_tokens)
            for candidate in candidates_to_discard:
                discard_revisioned_output_candidate(candidate)
            if session_hash is not None:
                discard_staged_chapter_outputs_for_session(
                    session_hash
                )

    queue.clean_events = clean_events_with_chapter_admission
    queue.chapter_admission_cleanup_installed = True


@contextmanager
def chapter_admission_execution(channel, request, expected_token=None):
    """Guarantee release after success, validation failure, timeout, or error."""
    token = _claim_chapter_admission(channel, request, expected_token)
    if token is None:
        raise gr.Error(
            "A chapter request is already pending for this session. Please wait for it to finish."
        )
    try:
        yield
    finally:
        release_chapter_admission(channel, request, token)


def _chapter_admission_validation(channel, request, input_count):
    admitted = _reserve_chapter_admission(channel, request)
    message = (
        ""
        if admitted
        else "A chapter request is already pending. Please wait for it to finish."
    )
    results = [gr.validate(admitted, message)]
    results.extend(gr.validate(True, "") for _ in range(input_count - 1))
    return tuple(results)


def validate_ui_chapter_admission(
    _current_revision,
    _source_choice,
    _session_state,
    _uploaded_srt_file,
    _density,
    _video_context,
    _api_key,
    _model_name,
    _custom_model,
    _base_url,
    _video_duration_seconds,
    _previous_output,
    _previous_download,
    _previous_result,
    request: gr.Request = None,
):
    channel = _CHAPTER_UI_GENERATION_ADMISSION
    validation = list(
        _chapter_admission_validation(channel, request, 14)
    )
    if not validation[0]["is_valid"]:
        return tuple(validation)
    token = _get_chapter_request_admission_token(request, channel)
    try:
        revision = advance_chapter_revision(_current_revision, request)
    except Exception as exc:
        release_chapter_admission(channel, request, token)
        validation[0] = gr.validate(False, _normalize_error_message(exc))
        return tuple(validation)
    if not _set_chapter_request_revision(request, channel, revision):
        release_chapter_admission(channel, request, token)
        validation[0] = gr.validate(
            False, "Unable to authorize this chapter request."
        )
    return tuple(validation)


def validate_chapter_edit_admission(
    _current_revision,
    _chapters_text,
    _previous_result,
    request: gr.Request = None,
):
    channel = _CHAPTER_EDIT_ADMISSION
    validation = list(
        _chapter_admission_validation(channel, request, 3)
    )
    if not validation[0]["is_valid"]:
        return tuple(validation)
    token = _get_chapter_request_admission_token(request, channel)
    try:
        revision = advance_chapter_revision(_current_revision, request)
    except Exception as exc:
        release_chapter_admission(channel, request, token)
        validation[0] = gr.validate(False, _normalize_error_message(exc))
        return tuple(validation)
    if not _set_chapter_request_revision(request, channel, revision):
        release_chapter_admission(channel, request, token)
        validation[0] = gr.validate(
            False, "Unable to authorize this chapter edit."
        )
    return tuple(validation)


def validate_api_chapter_admission(
    _srt_content,
    _density,
    _video_context,
    _api_key,
    _model_name,
    _custom_model,
    _base_url,
    _video_duration_seconds,
    request: gr.Request = None,
):
    return _chapter_admission_validation(
        _CHAPTER_API_GENERATION_ADMISSION, request, 8
    )


def _decode_local_chapter_upload_event(event):
    """Decode browser-read bytes without dereferencing a browser-supplied path."""
    data = getattr(event, "_data", None)
    if not isinstance(data, dict):
        raise gr.Error("Please choose one local Chinese SRT file.")
    if data.get("error"):
        raise gr.Error(str(data["error"]))
    file_name = str(data.get("name", "") or "").strip()
    encoded = data.get("data_base64", "")
    if not file_name and not encoded:
        return None
    if not file_name.lower().endswith(".srt"):
        raise gr.Error("Please choose exactly one .srt subtitle file.")
    if not isinstance(encoded, str):
        raise gr.Error("The uploaded SRT payload is invalid.")
    max_encoded_length = ((MAX_SRT_UTF8_BYTES + 2) // 3) * 4
    if len(encoded) > max_encoded_length:
        raise gr.Error(
            f"The uploaded SRT exceeds the {MAX_SRT_UTF8_BYTES:,}-byte UTF-8 limit."
        )
    try:
        raw_content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise gr.Error("The uploaded SRT payload is invalid.") from exc
    if len(raw_content) > MAX_SRT_UTF8_BYTES:
        raise gr.Error(
            f"The uploaded SRT exceeds the {MAX_SRT_UTF8_BYTES:,}-byte UTF-8 limit."
        )
    try:
        content = raw_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise gr.Error("The uploaded SRT must be UTF-8 encoded.") from exc
    if not content.strip():
        raise gr.Error("The uploaded SRT file is empty.")
    base_name = sanitize_base_name(
        os.path.splitext(os.path.basename(file_name))[0]
    )
    return content, base_name


def _purge_expired_chapter_uploads_locked(now):
    for session_key, upload in list(CHAPTER_SESSION_UPLOADS.items()):
        if now - upload["updated_at"] >= CHAPTER_SESSION_UPLOAD_TTL_SECONDS:
            CHAPTER_SESSION_UPLOADS.pop(session_key, None)


def begin_chapter_upload_revision(
    current_revision,
    previous_output=None,
    previous_download=None,
    previous_result=None,
    event: gr.EventData = None,
    request: gr.Request = None,
):
    """Bind browser-read SRT bytes to this session while invalidating old output."""
    session_key = _chapter_session_key(request)
    if not session_key:
        raise gr.Error("A browser session is required for local SRT uploads.")
    decode_error = None
    try:
        decoded = _decode_local_chapter_upload_event(event)
    except gr.Error as exc:
        decoded = None
        decode_error = _normalize_error_message(exc)
    now = time.time()
    with CHAPTER_REVISIONS_LOCK:
        _purge_expired_chapter_uploads_locked(now)
        if (
            decoded is not None
            and session_key not in CHAPTER_SESSION_UPLOADS
            and len(CHAPTER_SESSION_UPLOADS) >= CHAPTER_SESSION_UPLOAD_LIMIT
        ):
            raise gr.Error(
                "Too many active chapter uploads. Please try again after an existing upload expires."
            )
    revision, output, download, status, result = begin_chapter_artifact_revision(
        current_revision,
        previous_output,
        previous_download,
        previous_result,
        request,
    )
    with CHAPTER_REVISIONS_LOCK:
        if CHAPTER_REVISIONS.get(session_key) != revision:
            raise gr.Error("The selected SRT was superseded before it could be bound.")
        if decoded is None:
            CHAPTER_SESSION_UPLOADS.pop(session_key, None)
        else:
            content, base_name = decoded
            CHAPTER_SESSION_UPLOADS[session_key] = {
                "revision": revision,
                "content": content,
                "base_name": base_name,
                "updated_at": now,
            }
            CHAPTER_SESSION_UPLOADS.move_to_end(session_key)
    if decode_error:
        status = f"❌ {decode_error} The previous uploaded SRT was cleared."
        result = {
            "valid": False,
            "chapters": [],
            "chapters_text": "",
            "error": decode_error,
        }
    return revision, output, download, status, result


def advance_chapter_revision(current_revision, request: gr.Request = None):
    """Advance and record a per-session generation revision."""
    try:
        requested_revision = max(0, int(current_revision or 0))
    except (TypeError, ValueError):
        requested_revision = 0
    session_key = _chapter_session_key(request)
    discarded_outputs = []
    discarded_artifacts = []
    capacity_exhausted = False
    if session_key:
        with CHAPTER_REVISIONS_LOCK:
            now = time.time()
            is_new_session = session_key not in CHAPTER_REVISIONS
            revision = max(requested_revision, CHAPTER_REVISIONS.get(session_key, 0)) + 1
            CHAPTER_REVISIONS[session_key] = revision
            CHAPTER_REVISIONS.move_to_end(session_key)
            runtime = CHAPTER_SESSION_RUNTIME.get(session_key)
            if runtime is not None:
                runtime["revision"] = revision
                runtime["updated_at"] = now
                CHAPTER_SESSION_RUNTIME.move_to_end(session_key)
            upload = CHAPTER_SESSION_UPLOADS.get(session_key)
            if upload is not None:
                upload["revision"] = revision
                upload["updated_at"] = now
                CHAPTER_SESSION_UPLOADS.move_to_end(session_key)
            with STAGED_CHAPTER_OUTPUTS_LOCK:
                while len(CHAPTER_REVISIONS) > CHAPTER_REVISION_REGISTRY_LIMIT:
                    evicted_session_key = None
                    for candidate_session_key in CHAPTER_REVISIONS:
                        candidate_runtime = CHAPTER_SESSION_RUNTIME.get(
                            candidate_session_key
                        )
                        runtime_updated_at = (
                            candidate_runtime.get("updated_at")
                            if candidate_runtime
                            else None
                        )
                        runtime_expired = candidate_runtime is None or (
                            isinstance(runtime_updated_at, (int, float))
                            and not isinstance(runtime_updated_at, bool)
                            and now - runtime_updated_at
                            >= CHAPTER_ARTIFACT_TTL_SECONDS
                        )
                        candidate_upload = CHAPTER_SESSION_UPLOADS.get(
                            candidate_session_key
                        )
                        upload_updated_at = (
                            candidate_upload.get("updated_at")
                            if candidate_upload
                            else None
                        )
                        upload_expired = candidate_upload is None or (
                            isinstance(upload_updated_at, (int, float))
                            and not isinstance(upload_updated_at, bool)
                            and now - upload_updated_at
                            >= CHAPTER_SESSION_UPLOAD_TTL_SECONDS
                        )
                        candidate_revision = CHAPTER_REVISIONS.get(
                            candidate_session_key
                        )
                        staged_live = any(
                            candidate_key[0] == candidate_session_key
                            and candidate_key[2] == candidate_revision
                            and isinstance(created_at, (int, float))
                            and not isinstance(created_at, bool)
                            and now - created_at < CHAPTER_ARTIFACT_TTL_SECONDS
                            for candidate_key, (
                                created_at,
                                _outputs,
                            ) in STAGED_CHAPTER_OUTPUTS.items()
                        ) or any(
                            reservation["session_key"] == candidate_session_key
                            and reservation["revision"] == candidate_revision
                            for reservation in STAGED_CHAPTER_OUTPUT_RESERVATIONS.values()
                        )
                        if runtime_expired and upload_expired and not staged_live:
                            evicted_session_key = candidate_session_key
                            break
                    if evicted_session_key is None:
                        # Existing live sessions retain their authority, pending
                        # publications, and downloads. A newcomer waits for capacity.
                        if is_new_session:
                            CHAPTER_REVISIONS.pop(session_key, None)
                            capacity_exhausted = True
                        break
                    CHAPTER_REVISIONS.pop(evicted_session_key, None)
                    if is_new_session and evicted_session_key == session_key:
                        capacity_exhausted = True
                    evicted_runtime = CHAPTER_SESSION_RUNTIME.pop(
                        evicted_session_key, None
                    )
                    CHAPTER_SESSION_UPLOADS.pop(evicted_session_key, None)
                    if evicted_runtime and evicted_runtime.get("artifact_path"):
                        discarded_artifacts.append(
                            evicted_runtime["artifact_path"]
                        )
                    for candidate_key in list(STAGED_CHAPTER_OUTPUTS):
                        if candidate_key[0] == evicted_session_key:
                            discarded_outputs.append(
                                STAGED_CHAPTER_OUTPUTS.pop(candidate_key)[1]
                            )
                    for token, reservation in list(
                        STAGED_CHAPTER_OUTPUT_RESERVATIONS.items()
                    ):
                        if reservation["session_key"] == evicted_session_key:
                            STAGED_CHAPTER_OUTPUT_RESERVATIONS.pop(token, None)
                for candidate_key in list(STAGED_CHAPTER_OUTPUTS):
                    if candidate_key[0] == session_key:
                        discarded_outputs.append(
                            STAGED_CHAPTER_OUTPUTS.pop(candidate_key)[1]
                        )
                for token, reservation in list(
                    STAGED_CHAPTER_OUTPUT_RESERVATIONS.items()
                ):
                    if reservation["session_key"] == session_key:
                        STAGED_CHAPTER_OUTPUT_RESERVATIONS.pop(token, None)
    else:
        revision = requested_revision + 1
    for outputs in discarded_outputs:
        for output in outputs:
            remove_owned_chapter_artifact(output)
    for artifact_path in discarded_artifacts:
        remove_owned_chapter_artifact(artifact_path)
    if capacity_exhausted:
        raise gr.Error(
            "Too many active chapter sessions. Please try again after an existing session expires."
        )
    return revision


def chapter_revision_is_current(expected_revision, request: gr.Request = None):
    """Return false when a newer source/settings/edit event superseded this job."""
    session_key = _chapter_session_key(request)
    if not session_key or expected_revision is None:
        return True
    try:
        expected = int(expected_revision)
    except (TypeError, ValueError):
        return False
    with CHAPTER_REVISIONS_LOCK:
        return CHAPTER_REVISIONS.get(session_key) == expected


def _unchanged_outputs(count):
    """Keep newer UI/state values when an older callback finishes late."""
    return tuple(gr.update() for _ in range(count))


def _discard_stale_staged_outputs_locked(now=None):
    """Drop staged results whose server-authoritative revision has moved on."""
    now = time.time() if now is None else now
    discarded = []
    for candidate_key in list(STAGED_CHAPTER_OUTPUTS):
        session_key, _channel, revision = candidate_key
        created_at, outputs = STAGED_CHAPTER_OUTPUTS[candidate_key]
        expired = (
            not isinstance(created_at, (int, float))
            or isinstance(created_at, bool)
            or now - created_at >= CHAPTER_ARTIFACT_TTL_SECONDS
        )
        if CHAPTER_REVISIONS.get(session_key) != revision or expired:
            STAGED_CHAPTER_OUTPUTS.pop(candidate_key, None)
            discarded.append(outputs)
    for token, reservation in list(STAGED_CHAPTER_OUTPUT_RESERVATIONS.items()):
        if CHAPTER_REVISIONS.get(reservation["session_key"]) != reservation[
            "revision"
        ]:
            STAGED_CHAPTER_OUTPUT_RESERVATIONS.pop(token, None)
    return discarded


def reserve_staged_chapter_output_slot(
    expected_revision,
    request=None,
    *,
    channel,
):
    """Reserve publication capacity before generation or edit work begins."""
    session_key = _chapter_session_key(request)
    if not session_key or not channel:
        return None
    try:
        revision = int(expected_revision)
    except (TypeError, ValueError, OverflowError) as exc:
        raise gr.Error("This chapter request has an invalid revision.") from exc
    discarded_outputs = []
    token = None
    capacity_exhausted = False
    with CHAPTER_REVISIONS_LOCK:
        if CHAPTER_REVISIONS.get(session_key) != revision:
            raise gr.Error(
                "This chapter request was superseded before it started. Please try again."
            )
        with STAGED_CHAPTER_OUTPUTS_LOCK:
            discarded_outputs.extend(
                _discard_stale_staged_outputs_locked(time.time())
            )
            if (
                len(STAGED_CHAPTER_OUTPUTS)
                + len(STAGED_CHAPTER_OUTPUT_RESERVATIONS)
                >= STAGED_CHAPTER_OUTPUTS_LIMIT
            ):
                capacity_exhausted = True
            else:
                token = uuid.uuid4().hex
                STAGED_CHAPTER_OUTPUT_RESERVATIONS[token] = {
                    "session_key": session_key,
                    "channel": channel,
                    "revision": revision,
                }
    for discarded in discarded_outputs:
        for output in discarded:
            remove_owned_chapter_artifact(output)
    if capacity_exhausted:
        raise gr.Error(
            "Too many chapter results are waiting to publish. Please try again shortly."
        )
    return token


def release_staged_chapter_output_slot(token):
    if not token:
        return False
    with STAGED_CHAPTER_OUTPUTS_LOCK:
        return STAGED_CHAPTER_OUTPUT_RESERVATIONS.pop(token, None) is not None


def make_revisioned_output_candidate(
    outputs,
    expected_revision,
    request=None,
    *,
    channel="",
    staging_reservation_token=None,
):
    """Stage callback outputs for a serialized revision-aware UI commit."""
    try:
        revision = int(expected_revision)
    except (TypeError, ValueError, OverflowError):
        revision = -1
    outputs = tuple(outputs)
    session_key = _chapter_session_key(request)
    candidate = {
        "revision": revision,
        "session_key": session_key,
        "channel": channel,
    }
    if not session_key or not channel:
        candidate["outputs"] = outputs
        return candidate

    discarded_outputs = []
    stored = False
    capacity_exhausted = False
    with CHAPTER_REVISIONS_LOCK:
        with STAGED_CHAPTER_OUTPUTS_LOCK:
            discarded_outputs.extend(
                _discard_stale_staged_outputs_locked(time.time())
            )
            reservation_matches = staging_reservation_token is None
            if staging_reservation_token is not None:
                reservation = STAGED_CHAPTER_OUTPUT_RESERVATIONS.pop(
                    staging_reservation_token, None
                )
                reservation_matches = reservation == {
                    "session_key": session_key,
                    "channel": channel,
                    "revision": revision,
                }
            if (
                CHAPTER_REVISIONS.get(session_key) == revision
                and reservation_matches
            ):
                for candidate_key in list(STAGED_CHAPTER_OUTPUTS):
                    if candidate_key[:2] == (session_key, channel):
                        discarded_outputs.append(
                            STAGED_CHAPTER_OUTPUTS.pop(candidate_key)[1]
                        )
                candidate_key = (session_key, channel, revision)
                if (
                    staging_reservation_token is None
                    and candidate_key not in STAGED_CHAPTER_OUTPUTS
                    and len(STAGED_CHAPTER_OUTPUTS)
                    + len(STAGED_CHAPTER_OUTPUT_RESERVATIONS)
                    >= STAGED_CHAPTER_OUTPUTS_LIMIT
                ):
                    capacity_exhausted = True
                else:
                    STAGED_CHAPTER_OUTPUTS[candidate_key] = (
                        time.time(),
                        outputs,
                    )
                    stored = True
    if not stored:
        discarded_outputs.append(outputs)
    for discarded in discarded_outputs:
        for output in discarded:
            remove_owned_chapter_artifact(output)
    if capacity_exhausted:
        raise gr.Error(
            "Too many chapter results are waiting to publish. Please try again shortly."
        )
    return candidate


def discard_revisioned_output_candidate(candidate):
    """Discard one exact unpublished candidate and its owned artifacts."""
    outputs = None
    try:
        session_key = str(candidate.get("session_key", "") or "")
        channel = str(candidate.get("channel", "") or "")
        revision = int(candidate.get("revision"))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    if session_key and channel:
        with CHAPTER_REVISIONS_LOCK:
            with STAGED_CHAPTER_OUTPUTS_LOCK:
                stored = STAGED_CHAPTER_OUTPUTS.pop(
                    (session_key, channel, revision), None
                )
        if stored is not None:
            outputs = stored[1]
    elif isinstance(candidate, dict):
        inline_outputs = candidate.get("outputs")
        if isinstance(inline_outputs, (list, tuple)):
            outputs = inline_outputs
    if outputs is None:
        return False
    for output in outputs:
        remove_owned_chapter_artifact(output)
    return True


def discard_staged_chapter_outputs_for_session(session_key):
    """Discard pending YouTube publications for one cancelled session."""
    discarded_outputs = []
    with CHAPTER_REVISIONS_LOCK:
        with STAGED_CHAPTER_OUTPUTS_LOCK:
            for candidate_key in list(STAGED_CHAPTER_OUTPUTS):
                if (
                    candidate_key[0] == session_key
                    and candidate_key[1]
                    in {"youtube_generation", "youtube_edit"}
                ):
                    discarded_outputs.append(
                        STAGED_CHAPTER_OUTPUTS.pop(candidate_key)[1]
                    )
    for outputs in discarded_outputs:
        for output in outputs:
            remove_owned_chapter_artifact(output)
    return len(discarded_outputs)


def commit_revisioned_outputs(
    candidate,
    current_revision,
    output_count,
    channel,
    request=None,
):
    """Publish a staged result only from the current session revision.

    UI bindings run this reducer and every revision-advancing callback in the
    same single-concurrency queue. Gradio holds that queue slot through output
    postprocessing, closing the check-to-publication window.
    """
    try:
        expected_revision = int(candidate["revision"])
        visible_revision = int(current_revision)
        session_key = str(candidate.get("session_key", "") or "")
        candidate_channel = str(candidate.get("channel", "") or "")
    except (KeyError, TypeError, ValueError, OverflowError):
        return _unchanged_outputs(output_count)
    if candidate_channel != channel:
        return _unchanged_outputs(output_count)
    outputs = None
    is_current = expected_revision >= 0 and expected_revision == visible_revision
    if session_key:
        request_matches = _chapter_session_key(request) == session_key
        with CHAPTER_REVISIONS_LOCK:
            is_current = (
                request_matches
                and visible_revision >= 0
                and CHAPTER_REVISIONS.get(session_key) == visible_revision
            )
            with STAGED_CHAPTER_OUTPUTS_LOCK:
                stored = STAGED_CHAPTER_OUTPUTS.pop(
                    (session_key, channel, visible_revision), None
                )
            if stored is not None:
                outputs = stored[1]
    else:
        try:
            outputs = tuple(candidate["outputs"])
        except (KeyError, TypeError):
            is_current = False
    if outputs is None:
        is_current = False
    if not is_current or len(outputs) != output_count:
        if outputs is not None:
            for output in outputs:
                remove_owned_chapter_artifact(output)
        return _unchanged_outputs(output_count)
    if session_key and not _record_authoritative_chapter_runtime(
        session_key,
        visible_revision,
        channel,
        outputs,
    ):
        for output in outputs:
            remove_owned_chapter_artifact(output)
        return _unchanged_outputs(output_count)
    return outputs


def commit_process_outputs(candidate, current_revision, request: gr.Request = None):
    return commit_revisioned_outputs(
        candidate, current_revision, 5, "process", request
    )


def commit_correction_outputs(candidate, current_revision, request: gr.Request = None):
    return commit_revisioned_outputs(
        candidate, current_revision, 9, "correction", request
    )


def commit_traditional_translation_outputs(candidate, current_revision, request: gr.Request = None):
    return commit_revisioned_outputs(
        candidate, current_revision, 6, "traditional_translation", request
    )


def commit_english_translation_outputs(candidate, current_revision, request: gr.Request = None):
    return commit_revisioned_outputs(
        candidate, current_revision, 5, "english_translation", request
    )


def commit_translator_correction_outputs(candidate, current_revision, request: gr.Request = None):
    return commit_revisioned_outputs(
        candidate, current_revision, 8, "translator_correction", request
    )


def commit_chapter_generation_outputs(candidate, current_revision, request: gr.Request = None):
    return commit_revisioned_outputs(
        candidate, current_revision, 4, "youtube_generation", request
    )


def commit_chapter_edit_outputs(candidate, current_revision, request: gr.Request = None):
    return commit_revisioned_outputs(
        candidate, current_revision, 3, "youtube_edit", request
    )


def finalize_chapter_publication(candidate, request: gr.Request = None):
    """Release the exact UI lease only after its commit response is applied."""
    token = candidate.get("admission_token") if isinstance(candidate, dict) else None
    if isinstance(token, str) and token:
        release_chapter_admission(
            _CHAPTER_UI_ACTION_ADMISSION, request, token
        )


def sanitize_base_name(raw_name):
    """Sanitize a filename stem for safe temporary output paths."""
    candidate = (raw_name or "subtitles").strip()
    candidate = re.sub(r"[^0-9A-Za-z._-]+", "_", candidate)
    candidate = candidate.strip("._-")
    candidate = candidate[:MAX_SAFE_ARTIFACT_BASE_BYTES].rstrip("._-")
    return candidate or "subtitles"


def _bounded_temp_base_name(base_name, prefix, suffix):
    """Reserve room for tempfile randomness under portable NAME_MAX values."""
    safe_base = sanitize_base_name(base_name)
    fixed_bytes = len(prefix.encode("utf-8")) + len(suffix.encode("utf-8"))
    available_bytes = (
        PORTABLE_FILENAME_COMPONENT_BYTES
        - fixed_bytes
        - TEMPFILE_RANDOM_SUFFIX_RESERVE_BYTES
        - 1
    )
    if available_bytes < 1:
        raise RuntimeError("The internal temporary-file prefix is too long.")
    return safe_base[:available_bytes].rstrip("._-") or "subtitles"


def create_unique_srt_path(base_name, prefix="", suffix=".srt"):
    """Create a unique temp file path for SRT output."""
    safe_base = _bounded_temp_base_name(base_name, prefix, suffix)
    fd, file_path = tempfile.mkstemp(prefix=f"{prefix}{safe_base}_", suffix=suffix)
    os.close(fd)
    return file_path


def create_unique_text_path(base_name, prefix="", suffix=".txt"):
    """Create a unique temp file path for plain-text output."""
    safe_base = _bounded_temp_base_name(base_name, prefix, suffix)
    os.makedirs(CHAPTER_ARTIFACT_CACHE_DIR, exist_ok=True)
    fd, file_path = tempfile.mkstemp(
        prefix=f"{prefix}{safe_base}_",
        suffix=suffix,
        dir=CHAPTER_ARTIFACT_CACHE_DIR,
    )
    os.close(fd)
    return file_path


def _delete_owned_chapter_artifact_locked(output_path):
    """Delete one registered artifact while preserving failed paths for retry."""
    if output_path not in OWNED_CHAPTER_ARTIFACTS:
        return False
    try:
        os.remove(output_path)
    except FileNotFoundError:
        pass
    except OSError:
        return False
    OWNED_CHAPTER_ARTIFACTS.pop(output_path, None)
    return True


def _live_chapter_artifact_paths_locked(now):
    """Return live paths while the caller holds CHAPTER_REVISIONS_LOCK."""
    live_paths = set()
    for runtime in CHAPTER_SESSION_RUNTIME.values():
        artifact_path = runtime.get("artifact_path")
        updated_at = runtime.get("updated_at")
        is_live = (
            updated_at is None
            or not isinstance(updated_at, (int, float))
            or isinstance(updated_at, bool)
            or now - updated_at < CHAPTER_ARTIFACT_TTL_SECONDS
        )
        if artifact_path and is_live:
            live_paths.add(artifact_path)
    with STAGED_CHAPTER_OUTPUTS_LOCK:
        for candidate_key, (created_at, outputs) in STAGED_CHAPTER_OUTPUTS.items():
            session_key, _channel, revision = candidate_key
            if (
                CHAPTER_REVISIONS.get(session_key) != revision
                or not isinstance(created_at, (int, float))
                or isinstance(created_at, bool)
                or now - created_at >= CHAPTER_ARTIFACT_TTL_SECONDS
            ):
                continue
            live_paths.update(
                output for output in outputs if isinstance(output, str)
            )
    return live_paths


def _reserve_chapter_artifact_slot():
    """Reserve bounded registry capacity before creating another temp file."""
    global CHAPTER_ARTIFACT_RESERVATIONS
    now = time.time()
    with CHAPTER_REVISIONS_LOCK:
        live_paths = _live_chapter_artifact_paths_locked(now)
        with OWNED_CHAPTER_ARTIFACTS_LOCK:
            for owned_path, created_at in list(OWNED_CHAPTER_ARTIFACTS.items()):
                if (
                    owned_path in live_paths
                    or now - created_at < CHAPTER_ARTIFACT_TTL_SECONDS
                ):
                    continue
                _delete_owned_chapter_artifact_locked(owned_path)

            if (
                len(OWNED_CHAPTER_ARTIFACTS) + CHAPTER_ARTIFACT_RESERVATIONS
                >= CHAPTER_ARTIFACT_REGISTRY_LIMIT
            ):
                raise RuntimeError(
                    "Unable to create a chapter download because all temporary-file slots are still active."
                )
            CHAPTER_ARTIFACT_RESERVATIONS += 1


def create_owned_chapter_artifact(base_name, prefix, text):
    """Write and register a chapter artifact that this process may safely delete."""
    global CHAPTER_ARTIFACT_RESERVATIONS
    _reserve_chapter_artifact_slot()
    output_path = None
    try:
        output_path = create_unique_text_path(base_name, prefix=prefix)
        with open(output_path, 'wb') as artifact_file:
            artifact_file.write(text.encode("utf-8"))
    except Exception:
        deletion_failed = False
        if output_path is not None:
            try:
                os.remove(output_path)
            except FileNotFoundError:
                pass
            except OSError:
                deletion_failed = True
        with OWNED_CHAPTER_ARTIFACTS_LOCK:
            CHAPTER_ARTIFACT_RESERVATIONS -= 1
            if deletion_failed:
                OWNED_CHAPTER_ARTIFACTS[output_path] = time.time()
        raise
    with OWNED_CHAPTER_ARTIFACTS_LOCK:
        CHAPTER_ARTIFACT_RESERVATIONS -= 1
        OWNED_CHAPTER_ARTIFACTS[output_path] = time.time()
    return output_path


def remove_owned_chapter_artifact(result_or_path):
    """Delete only an exact chapter path previously registered by this process."""
    if isinstance(result_or_path, dict):
        output_path = result_or_path.pop("_artifact_path", None)
    else:
        output_path = result_or_path
    if not isinstance(output_path, str):
        return False
    with OWNED_CHAPTER_ARTIFACTS_LOCK:
        if output_path not in OWNED_CHAPTER_ARTIFACTS:
            return False
        return _delete_owned_chapter_artifact_locked(output_path)


def _record_authoritative_chapter_runtime(
    session_key,
    revision,
    channel,
    outputs,
):
    """Bind validated duration and the exact served artifact to one session."""
    if channel not in {"youtube_generation", "youtube_edit"}:
        return True
    result_index = 3 if channel == "youtube_generation" else 2
    artifact_index = 1 if channel == "youtube_generation" else 0
    result = outputs[result_index] if len(outputs) > result_index else None
    if not isinstance(result, dict) or not result.get("valid"):
        return True
    artifact_path = outputs[artifact_index]
    if not isinstance(artifact_path, str):
        return False
    old_artifact_path = None
    with CHAPTER_REVISIONS_LOCK:
        with OWNED_CHAPTER_ARTIFACTS_LOCK:
            if artifact_path not in OWNED_CHAPTER_ARTIFACTS:
                return False
            if CHAPTER_REVISIONS.get(session_key) != revision:
                return False
            if channel == "youtube_generation":
                try:
                    duration_ms = normalize_video_duration_ms(
                        result.get("duration_ms")
                    )
                except ChapterGenerationError:
                    return False
                previous = CHAPTER_SESSION_RUNTIME.get(session_key)
                if previous:
                    old_artifact_path = previous.get("artifact_path")
                CHAPTER_SESSION_RUNTIME[session_key] = {
                    "revision": revision,
                    "duration_ms": duration_ms,
                    "base_name": result.get("base_name", "chapters"),
                    "source": result.get("source", "chapter source"),
                    "artifact_path": artifact_path,
                    "updated_at": time.time(),
                }
            else:
                runtime = CHAPTER_SESSION_RUNTIME.get(session_key)
                if not runtime or runtime.get("revision") != revision:
                    return False
                old_artifact_path = runtime.get("artifact_path")
                runtime["artifact_path"] = artifact_path
                runtime["updated_at"] = time.time()
            CHAPTER_SESSION_RUNTIME.move_to_end(session_key)
    if old_artifact_path and old_artifact_path != artifact_path:
        remove_owned_chapter_artifact(old_artifact_path)
    return True


def _invalidate_authoritative_chapter_runtime(
    request,
    expected_revision,
    *,
    preserve_source,
):
    """Detach only the requesting session's owned artifact and trusted metadata."""
    session_key = _chapter_session_key(request)
    if not session_key:
        return None
    try:
        revision = int(expected_revision)
    except (TypeError, ValueError, OverflowError):
        return None
    artifact_path = None
    runtime_copy = None
    with CHAPTER_REVISIONS_LOCK:
        runtime = CHAPTER_SESSION_RUNTIME.get(session_key)
        if not runtime or runtime.get("revision") != revision:
            return None
        artifact_path = runtime.get("artifact_path")
        if preserve_source:
            runtime["artifact_path"] = None
            runtime["updated_at"] = time.time()
            runtime_copy = runtime.copy()
        else:
            runtime_copy = CHAPTER_SESSION_RUNTIME.pop(session_key).copy()
    if artifact_path:
        remove_owned_chapter_artifact(artifact_path)
    return runtime_copy


def _get_authoritative_chapter_runtime(request, expected_revision):
    """Read the current trusted chapter authority for one Gradio session."""
    session_key = _chapter_session_key(request)
    if not session_key:
        return None
    try:
        revision = int(expected_revision)
    except (TypeError, ValueError, OverflowError):
        return None
    with CHAPTER_REVISIONS_LOCK:
        runtime = CHAPTER_SESSION_RUNTIME.get(session_key)
        if not runtime or runtime.get("revision") != revision:
            return None
        runtime["updated_at"] = time.time()
        CHAPTER_SESSION_RUNTIME.move_to_end(session_key)
        return runtime.copy()


def cleanup_owned_chapter_artifacts(force=False):
    """Periodically remove expired artifacts even when no new request arrives."""
    now = time.time()
    removed_count = 0
    with CHAPTER_REVISIONS_LOCK:
        discarded_outputs = []
        if not force:
            with STAGED_CHAPTER_OUTPUTS_LOCK:
                discarded_outputs = _discard_stale_staged_outputs_locked(now)
        live_paths = (
            set() if force else _live_chapter_artifact_paths_locked(now)
        )
        with OWNED_CHAPTER_ARTIFACTS_LOCK:
            for outputs in discarded_outputs:
                for output in outputs:
                    if (
                        isinstance(output, str)
                        and output in OWNED_CHAPTER_ARTIFACTS
                        and _delete_owned_chapter_artifact_locked(output)
                    ):
                        removed_count += 1
            for output_path, created_at in list(OWNED_CHAPTER_ARTIFACTS.items()):
                if output_path in live_paths:
                    continue
                if not force and now - created_at < CHAPTER_ARTIFACT_TTL_SECONDS:
                    continue
                if _delete_owned_chapter_artifact_locked(output_path):
                    removed_count += 1
    return removed_count


def _chapter_artifact_reaper_loop():
    interval = max(1.0, min(60.0, CHAPTER_ARTIFACT_TTL_SECONDS / 2))
    while not CHAPTER_ARTIFACT_REAPER_STOP.wait(interval):
        cleanup_expired_chapter_admissions()
        cleanup_owned_chapter_artifacts()


def _shutdown_chapter_artifact_reaper():
    CHAPTER_ARTIFACT_REAPER_STOP.set()
    cleanup_owned_chapter_artifacts(force=True)


CHAPTER_ARTIFACT_REAPER_THREAD = threading.Thread(
    target=_chapter_artifact_reaper_loop,
    name="funclip-chapter-artifact-reaper",
    daemon=True,
)
CHAPTER_ARTIFACT_REAPER_THREAD.start()
atexit.register(_shutdown_chapter_artifact_reaper)


def get_media_duration_seconds(file_path, is_video):
    """Best-effort media duration lookup for status/ETA reporting."""
    duration_sec = 0
    try:
        if is_video:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                duration_sec = float(result.stdout.strip())
        else:
            import librosa
            duration_sec = librosa.get_duration(path=file_path)
    except Exception as e:
        print(f"Could not determine duration: {e}")
        duration_sec = 0
    return duration_sec


def estimate_transcribe_eta_seconds(duration_sec):
    """Estimate transcribe ETA from media duration."""
    if duration_sec <= 0:
        return None
    # Typical observed throughput on this setup is much faster than realtime.
    return max(5.0, duration_sec / 45.0)


def estimate_correction_eta_seconds(srt_content):
    """Estimate correction ETA from subtitle text size."""
    if not srt_content:
        return None
    # Rough heuristic tuned for network + LLM latency.
    return max(15.0, len(srt_content) / 70.0)


def prune_expired_async_jobs():
    """Drop stale jobs to keep in-memory state bounded."""
    now = time.time()
    with ASYNC_JOBS_LOCK:
        stale_ids = [
            job_id for job_id, job in ASYNC_JOBS.items()
            if now - job.get("updated_at", now) > ASYNC_JOB_TTL_SECONDS
        ]
        for job_id in stale_ids:
            ASYNC_JOBS.pop(job_id, None)


def create_async_job(operation):
    """Create a queued async job record and return its id."""
    prune_expired_async_jobs()
    now = time.time()
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "operation": operation,
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "eta_seconds": None,
        "message": "Queued",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": "",
    }
    with ASYNC_JOBS_LOCK:
        ASYNC_JOBS[job_id] = job
    return job_id


def update_async_job(job_id, **updates):
    """Patch an async job record."""
    with ASYNC_JOBS_LOCK:
        job = ASYNC_JOBS.get(job_id)
        if not job:
            return False
        job.update(updates)
        job["updated_at"] = time.time()
        return True


def set_async_stage(job_id, stage, progress, message, eta_seconds=None):
    """Update stage/progress for a running async job."""
    update_async_job(
        job_id,
        status="running",
        stage=stage,
        progress=float(max(0.0, min(1.0, progress))),
        eta_seconds=None if eta_seconds is None else max(0.0, float(eta_seconds)),
        message=message,
    )


def complete_async_job(job_id, result, message="✅ Job completed."):
    """Mark an async job as completed and store result payload."""
    now = time.time()
    update_async_job(
        job_id,
        status="completed",
        stage="completed",
        progress=1.0,
        eta_seconds=0.0,
        message=message,
        completed_at=now,
        result=result,
        error="",
    )


def fail_async_job(job_id, error_message):
    """Mark an async job as failed with normalized error details."""
    now = time.time()
    update_async_job(
        job_id,
        status="failed",
        stage="failed",
        progress=1.0,
        eta_seconds=0.0,
        message="❌ Job failed.",
        completed_at=now,
        error=error_message,
    )


def get_async_job_snapshot(job_id, include_result=False):
    """Return a public snapshot for polling clients."""
    prune_expired_async_jobs()
    with ASYNC_JOBS_LOCK:
        job = ASYNC_JOBS.get((job_id or "").strip())
        if not job:
            return {
                "job_id": (job_id or "").strip(),
                "status": "not_found",
                "message": "Job id not found or expired.",
            }

        snapshot = {
            "job_id": job.get("job_id"),
            "operation": job.get("operation"),
            "status": job.get("status"),
            "stage": job.get("stage"),
            "progress": job.get("progress"),
            "eta_seconds": job.get("eta_seconds"),
            "message": job.get("message"),
            "error": job.get("error", ""),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }
        if include_result and job.get("result") is not None:
            snapshot["result"] = job.get("result")
        return snapshot


def stage_async_uploaded_file(file_path):
    """Copy uploaded file to a stable temp path for background processing."""
    if not file_path:
        raise gr.Error("No file uploaded.")
    try:
        source = validate_gradio_upload_path(str(file_path))
    except UploadPathError as exc:
        raise gr.Error("Uploaded media is not in the managed service cache.") from exc
    base_name = sanitize_base_name(source.stem)
    ext = source.suffix
    fd, staged_path = tempfile.mkstemp(prefix=f"funclip_job_{base_name}_", suffix=ext)
    os.close(fd)
    try:
        shutil.copy2(source, staged_path)
    except Exception:
        try:
            os.remove(staged_path)
        except OSError:
            pass
        raise
    return staged_path, str(source)

# --- 4. PROCESSING FUNCTIONS ---

def process_media(
    file_path,
    session_state,
    expected_chapter_revision=None,
    progress=gr.Progress(),
    request: gr.Request = None,
):
    """Process audio/video file and return transcription results."""
    if not chapter_revision_is_current(expected_chapter_revision, request):
        return _unchanged_outputs(5)
    session_state = update_latest_traditional_source(session_state)
    if not file_path:
        gr.Warning("Please upload a file first.")
        return None, None, None, "No file uploaded", session_state
    
    start_time = time.time()
    
    # Get original filename for later use (stored in session state)
    original_name = os.path.basename(file_path)
    base_name = sanitize_base_name(os.path.splitext(original_name)[0])
    session_state["original_filename"] = base_name
    
    # Determine file type
    _, ext = os.path.splitext(file_path)
    is_video = ext.lower() in ['.mp4', '.avi', '.mkv', '.mov']
    
    duration_sec = get_media_duration_seconds(file_path, is_video)
    session_state["media_duration_seconds"] = duration_sec
    
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
    
    # Save SRT to a unique temp file path to avoid cross-request collisions.
    srt_path = create_unique_srt_path(base_name, prefix="transcribe_")
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(res_srt)
    
    # Format status message
    status_msg = f"✅ Completed in {total_time:.2f}s ({speed_x:.1f}x speed)"
    
    # Format text with proper line breaks for Markdown display
    formatted_text = res_text.replace("\n", "\n\n") if res_text else ""

    if not chapter_revision_is_current(expected_chapter_revision, request):
        try:
            os.remove(srt_path)
        except OSError:
            pass
        return _unchanged_outputs(5)
    
    return formatted_text, res_srt, srt_path, status_msg, session_state


def get_api_key_status():
    """Check if system API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return "✅ System API Key detected"
    return "⚠️ No System API Key found"


def _server_openai_key_is_allowed(model, effective_base_url, env_base_url):
    """Keep the server OpenAI credential on OpenAI or an explicit gateway alias."""
    model_key = str(model or "").strip().casefold()
    trusted_aliases = {
        alias.strip().casefold()
        for alias in os.getenv("FUNCLIP_TRUSTED_LLM_MODELS", "").split(",")
        if alias.strip()
    }
    if (
        env_base_url
        and effective_base_url == env_base_url
        and model_key in trusted_aliases
    ):
        return True
    try:
        _, provider, _, _ = get_llm_provider(model_key)
    except Exception:
        # Unknown aliases are unsafe by default. Operators can opt an exact
        # alias into their own configured gateway with FUNCLIP_TRUSTED_LLM_MODELS.
        return False
    return str(provider or "").casefold() == "openai"


def validate_llm_model_identifier(value):
    """Return one bounded canonical LiteLLM model/provider identifier."""
    try:
        return validate_chapter_model_identifier(value)
    except ChapterGenerationError as exc:
        raise gr.Error(str(exc)) from exc


def resolve_llm_config(api_key, model_name, custom_model, base_url):
    """Resolve effective LLM config from user input and environment."""
    if model_name == "Custom":
        if not custom_model or not custom_model.strip():
            raise gr.Error("Custom model is selected, but model name is empty. Please enter a custom model name.")
    effective_model = validate_llm_model_identifier(
        custom_model if model_name == "Custom" else model_name
    )
    provided_api_key = api_key.strip() if isinstance(api_key, str) else ""
    requested_base_url = base_url.strip() if isinstance(base_url, str) else ""
    env_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    env_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if requested_base_url and not provided_api_key and requested_base_url != env_base_url:
        raise gr.Error("A custom Base URL requires its own API key; the server API key cannot be forwarded there.")
    eff_api_key = provided_api_key or env_api_key
    eff_base_url = requested_base_url or env_base_url

    if not eff_api_key:
        raise gr.Error("No API Key provided. Please enter an API key or set OPENAI_API_KEY in your .env file.")
    if (
        not provided_api_key
        and not _server_openai_key_is_allowed(
            effective_model, eff_base_url, env_base_url
        )
    ):
        raise gr.Error(
            "The selected provider requires its own API key; the server OpenAI key cannot be forwarded there."
        )

    return eff_api_key, eff_base_url, effective_model


def _compact_error_message(error):
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


def _redact_llm_error_message(message, secrets=()):
    """Remove exact keys and credential-shaped fields from provider errors."""
    redacted = str(message or "")
    changed = False
    exact_secrets = set()
    for secret in (*secrets, os.getenv("OPENAI_API_KEY", "")):
        if secret is None or not str(secret):
            continue
        exact_secrets.add(str(secret))
        if str(secret).strip():
            exact_secrets.add(str(secret).strip())
    for secret in sorted(exact_secrets, key=len, reverse=True):
        if secret in redacted:
            redacted = redacted.replace(secret, "[REDACTED]")
            changed = True

    credential_patterns = (
        (
            re.compile(r"(?i)(\b(?:authorization|proxy-authorization)\s*[:=]\s*(?:bearer|basic)?\s*)[^\s,;]+"),
            r"\1[REDACTED]",
        ),
        (
            re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]+"),
            r"\1[REDACTED]",
        ),
        (
            re.compile(r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)[^&#\s]*"),
            r"\1[REDACTED]",
        ),
        (
            re.compile(r"(?i)(\b(?:api[_-]?key|access[_-]?token|token|secret|password)\s*[:=]\s*)[^\s,;]+"),
            r"\1[REDACTED]",
        ),
        (
            re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+@"),
            r"\1[REDACTED]@",
        ),
        (re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b"), "[REDACTED]"),
    )
    for pattern, replacement in credential_patterns:
        redacted, count = pattern.subn(replacement, redacted)
        changed = changed or count > 0
    return redacted, changed


def _normalize_error_message(error, secrets=()):
    """Return one compact provider message with credentials removed."""
    message, _ = _redact_llm_error_message(
        _compact_error_message(error), secrets
    )
    return message


def format_llm_error(error, operation_name, secrets=()):
    """Return a concise, actionable UI message for common LLM failures."""
    raw_message, credentials_hidden = _redact_llm_error_message(
        _compact_error_message(error), secrets
    )
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

    if credentials_hidden:
        return (
            f"{operation_name} failed: the provider returned an unsafe error; "
            "credential-bearing details were hidden."
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


def prepare_latest_traditional_source(srt_content):
    """Return canonical Traditional Chinese SRT only when it is a valid chapter source."""
    if not isinstance(srt_content, str) or not srt_content.strip():
        return ""
    candidate = convert_to_traditional(srt_content)
    try:
        validate_chinese_transcript(parse_srt(candidate))
    except ChapterGenerationError:
        return ""
    return candidate


def run_llm_correction_for_content(
    original_srt,
    base_name,
    api_key,
    model_name,
    custom_model,
    base_url,
):
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
            model=effective_model,
        )
    except Exception as e:
        raise gr.Error(format_llm_error(e, "AI auto correction"))

    diff_html = generate_diff_html(original_srt, corrected_srt)
    traditional_srt = convert_to_traditional(corrected_srt)

    safe_base = sanitize_base_name(base_name if base_name else "subtitles")
    orig_path = create_unique_srt_path(safe_base, prefix="orig_")
    with open(orig_path, 'w', encoding='utf-8') as f:
        f.write(original_srt)

    corr_path = create_unique_srt_path(safe_base, prefix="corrected_")
    with open(corr_path, 'w', encoding='utf-8') as f:
        f.write(corrected_srt)

    trad_path = create_unique_srt_path(f"{safe_base}_traditional", prefix="corrected_")
    with open(trad_path, 'w', encoding='utf-8') as f:
        f.write(traditional_srt)

    elapsed_time = time.time() - start_time
    status_msg = f"✅ Correction completed in {elapsed_time:.2f}s"
    return original_srt, corrected_srt, traditional_srt, diff_html, orig_path, corr_path, trad_path, status_msg


def run_llm_correction(original_srt, api_key, model_name, custom_model, base_url, session_state, progress=gr.Progress()):
    """Run LLM-based SRT correction from transcription pipeline content."""
    base_name = session_state.get("original_filename", "subtitles") if session_state else "subtitles"
    return run_llm_correction_for_content(
        original_srt=original_srt,
        base_name=base_name,
        api_key=api_key,
        model_name=model_name,
        custom_model=custom_model,
        base_url=base_url,
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

    has_singular_input = len(srt_paths) == 1 and len(output_paths) == 1
    translator_state = {
        "latest_output_paths": output_paths,
        "latest_output_kind": "traditional",
        "latest_output_srt": last_translated if has_singular_input else "",
        "latest_output_base_name": base_name if has_singular_input else "",
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


def run_llm_correction_for_files(
    srt_files,
    api_key,
    model_name,
    custom_model,
    base_url,
):
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
                model=effective_model,
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
    latest_traditional_srt = ""
    latest_traditional_base_name = ""
    if len(srt_paths) == 1 and len(corrected_paths) == 1:
        candidate_traditional_srt = prepare_latest_traditional_source(last_corrected)
        if candidate_traditional_srt:
            latest_traditional_srt = candidate_traditional_srt
            latest_traditional_base_name = (
                f"{os.path.splitext(os.path.basename(corrected_paths[0]))[0]}_traditional"
            )
    return (
        last_original,
        last_corrected,
        diff_html,
        original_download,
        corrected_download,
        status_msg,
        latest_traditional_srt,
        latest_traditional_base_name,
    )


def safe_translate_traditional_wrapper(
    srt_files,
    session_state,
    expected_chapter_revision=None,
    request: gr.Request = None,
):
    """Safe wrapper for step-1 Traditional translation with UI-friendly status."""
    safe_session_state = update_latest_traditional_source(session_state)
    try:
        preview_original, preview_translated, download_path, translator_state = translate_srt_to_traditional(srt_files)
        latest_source = prepare_latest_traditional_source(
            translator_state.get("latest_output_srt", "")
        )
        updated_session_state = update_latest_traditional_source(
            safe_session_state,
            latest_source,
            translator_state.get("latest_output_base_name", "") if latest_source else "",
        )
        if not chapter_revision_is_current(expected_chapter_revision, request):
            return _unchanged_outputs(6)
        status_msg = "✅ Translation to Traditional Chinese completed."
        return preview_original, preview_translated, download_path, translator_state, updated_session_state, status_msg
    except gr.Error as e:
        if not chapter_revision_is_current(expected_chapter_revision, request):
            return _unchanged_outputs(6)
        return gr.update(), gr.update(), gr.update(), {}, safe_session_state, f"❌ {_normalize_error_message(e)}"
    except Exception as e:
        if not chapter_revision_is_current(expected_chapter_revision, request):
            return _unchanged_outputs(6)
        return gr.update(), gr.update(), gr.update(), {}, safe_session_state, f"❌ {format_llm_error(e, 'Traditional Chinese translation')}"


def safe_translate_english_wrapper(
    srt_files,
    api_key,
    model_name,
    custom_model,
    base_url,
    expected_chapter_revision=None,
    request: gr.Request = None,
):
    """Safe wrapper for English translation with UI-friendly status."""
    try:
        preview_original, preview_translated, download_path, translator_state = translate_srt_to_english_fn(
            srt_files, api_key, model_name, custom_model, base_url
        )
        if not chapter_revision_is_current(expected_chapter_revision, request):
            return _unchanged_outputs(5)
        status_msg = "✅ English translation completed."
        return preview_original, preview_translated, download_path, translator_state, status_msg
    except gr.Error as e:
        if not chapter_revision_is_current(expected_chapter_revision, request):
            return _unchanged_outputs(5)
        return gr.update(), gr.update(), gr.update(), {}, f"❌ {_normalize_error_message(e)}"
    except Exception as e:
        if not chapter_revision_is_current(expected_chapter_revision, request):
            return _unchanged_outputs(5)
        return gr.update(), gr.update(), gr.update(), {}, f"❌ {format_llm_error(e, 'English translation')}"


def update_translated_output_hint(translator_state):
    """Show what stream output is available for translator AI correction."""
    if not translator_state or not translator_state.get("latest_output_paths"):
        return gr.update(value="No translated output cached yet. Run translation above or switch to upload mode.")

    kind = translator_state.get("latest_output_kind", "translated")
    count = len(translator_state.get("latest_output_paths", []))
    return gr.update(value=f"Using latest {kind} output from above ({count} file(s)).")


def resolve_chapter_srt(
    source_choice,
    session_state,
    _uploaded_srt_value,
    request: gr.Request = None,
    expected_chapter_revision=None,
):
    """Resolve one SRT source for YouTube chapter generation."""
    if source_choice == "Upload Chinese SRT":
        try:
            revision = int(expected_chapter_revision)
        except (TypeError, ValueError, OverflowError) as exc:
            raise gr.Error("The selected SRT upload has no valid session revision.") from exc
        session_key = _chapter_session_key(request)
        if not session_key:
            raise gr.Error("A browser session is required for local SRT uploads.")
        now = time.time()
        with CHAPTER_REVISIONS_LOCK:
            _purge_expired_chapter_uploads_locked(now)
            upload = CHAPTER_SESSION_UPLOADS.get(session_key)
            if (
                CHAPTER_REVISIONS.get(session_key) != revision
                or upload is None
                or upload.get("revision") != revision
            ):
                raise gr.Error(
                    "No session-bound Chinese SRT upload is available. Choose the local file again."
                )
            upload["updated_at"] = now
            CHAPTER_SESSION_UPLOADS.move_to_end(session_key)
            return upload["content"], upload["base_name"], "uploaded SRT"

    state = session_state or {}
    srt_content = state.get("latest_traditional_srt", "")
    if not isinstance(srt_content, str) or not srt_content.strip():
        raise gr.Error(
            "No finalized Traditional Chinese SRT is available. Run Traditional Chinese correction/translation first or upload an SRT."
        )
    base_name = state.get("latest_traditional_base_name", "subtitles")
    return srt_content, base_name, "latest finalized Traditional Chinese SRT"


def run_youtube_chapter_generation(
    source_choice,
    session_state,
    uploaded_srt_file,
    density,
    video_context,
    api_key,
    model_name,
    custom_model,
    base_url,
    video_duration_seconds=None,
    pre_completion_check=None,
    request: gr.Request = None,
    expected_chapter_revision=None,
):
    """Generate validated Traditional Chinese YouTube chapters for the UI."""
    srt_content, base_name, source_label = resolve_chapter_srt(
        source_choice,
        session_state,
        uploaded_srt_file,
        request,
        expected_chapter_revision,
    )
    # Bound upload/media-derived stems before the paid provider call so an
    # otherwise-valid generation cannot fail only while creating its download.
    base_name = sanitize_base_name(base_name)
    eff_api_key, eff_base_url, effective_model = resolve_llm_config(
        api_key=api_key,
        model_name=model_name,
        custom_model=custom_model,
        base_url=base_url,
    )
    video_duration_ms = None
    if video_duration_seconds not in (None, ""):
        video_duration_ms = parse_video_duration_seconds(video_duration_seconds)
    elif source_choice != "Upload Chinese SRT":
        stored_duration = (session_state or {}).get(
            "latest_traditional_video_duration_ms"
        )
        try:
            video_duration_ms = (
                normalize_video_duration_ms(stored_duration)
                if stored_duration is not None
                else None
            )
        except ChapterGenerationError as exc:
            raise gr.Error("The stored video duration is invalid.") from exc
    result = generate_youtube_chapters(
        srt_content,
        api_key=eff_api_key,
        base_url=eff_base_url,
        model=effective_model,
        density=density,
        video_context=video_context,
        title_transform=normalize_traditional_for_validation,
        video_duration_ms=video_duration_ms,
        pre_completion_check=pre_completion_check,
    )
    result["base_name"] = base_name
    result["source"] = source_label
    result["valid"] = True
    output_path = create_owned_chapter_artifact(
        base_name, "youtube_chapters_", result["chapters_text"]
    )
    result["_artifact_path"] = output_path
    duration_source = (
        "video duration" if video_duration_ms is not None else "SRT timeline extent"
    )
    status = (
        f"✅ Valid for YouTube: {len(result['chapters'])} chapters generated from {source_label}. "
        f"The first timestamp is 00:00 and every chapter is at least 10 seconds using {duration_source}."
    )
    return result["chapters_text"], output_path, status, result


def parse_video_duration_seconds(value):
    """Convert a public seconds value to bounded milliseconds for chapter validation."""
    if isinstance(value, bool):
        raise gr.Error("Video duration must be a positive finite number of seconds.")
    try:
        duration_seconds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise gr.Error("Video duration must be a positive finite number of seconds.") from exc
    if not math.isfinite(duration_seconds):
        raise gr.Error("Video duration must be a positive finite number of seconds.")
    try:
        return normalize_video_duration_ms(duration_seconds * 1000)
    except ChapterGenerationError as exc:
        raise gr.Error(
            f"Video duration must be greater than zero and no longer than {MAX_VIDEO_DURATION_MS // 3_600_000} hours."
        ) from exc


def safe_youtube_chapter_wrapper(
    source_choice,
    session_state,
    uploaded_srt_file,
    density,
    video_context,
    api_key,
    model_name,
    custom_model,
    base_url,
    video_duration_seconds=None,
    expected_chapter_revision=None,
    request: gr.Request = None,
):
    """Return UI-friendly errors and clear artifacts that belong to an older source."""
    if not chapter_revision_is_current(expected_chapter_revision, request):
        return _unchanged_outputs(4)
    try:
        result = run_youtube_chapter_generation(
            source_choice,
            session_state,
            uploaded_srt_file,
            density,
            video_context,
            api_key,
            model_name,
            custom_model,
            base_url,
            video_duration_seconds,
            pre_completion_check=lambda: chapter_revision_is_current(
                expected_chapter_revision, request
            ),
            request=request,
            expected_chapter_revision=expected_chapter_revision,
        )
        if not chapter_revision_is_current(expected_chapter_revision, request):
            remove_owned_chapter_artifact(result[3])
            return _unchanged_outputs(4)
        return result
    except (gr.Error, ChapterGenerationError) as e:
        if not chapter_revision_is_current(expected_chapter_revision, request):
            return _unchanged_outputs(4)
        message = _normalize_error_message(e, secrets=(api_key,))
        return "", None, f"❌ {message}", {
            "valid": False,
            "chapters": [],
            "chapters_text": "",
            "error": message,
        }
    except Exception as e:
        if not chapter_revision_is_current(expected_chapter_revision, request):
            return _unchanged_outputs(4)
        message = format_llm_error(
            e, "YouTube chapter generation", secrets=(api_key,)
        )
        return "", None, f"❌ {message}", {
            "valid": False,
            "chapters": [],
            "chapters_text": "",
            "error": message,
        }


def reset_chapter_artifacts(*_previous_values):
    """Invalidate chapters whenever the finalized subtitle source is replaced or cleared."""
    for previous_value in _previous_values:
        remove_owned_chapter_artifact(previous_value)
    message = "Subtitle source changed. Generate chapters again."
    return "", None, f"Ready — {message}", {
        "valid": False,
        "chapters": [],
        "chapters_text": "",
        "error": message,
    }


def reset_chapter_source_and_artifacts(session_state, *_previous_values):
    """Publish an invalidated finalized source and empty chapters before new source work."""
    return update_latest_traditional_source(session_state), *reset_chapter_artifacts()


def begin_chapter_artifact_revision(
    current_revision,
    previous_output=None,
    previous_download=None,
    previous_result=None,
    request: gr.Request = None,
):
    """Advance the race gate and invalidate existing chapter artifacts."""
    revision = advance_chapter_revision(current_revision, request)
    if _chapter_session_key(request):
        _invalidate_authoritative_chapter_runtime(
            request, revision, preserve_source=False
        )
        previous_output = previous_download = previous_result = None
    return revision, *reset_chapter_artifacts(
        previous_output,
        previous_download,
        previous_result,
    )


def begin_chapter_source_revision(
    current_revision,
    session_state,
    previous_output=None,
    previous_download=None,
    previous_result=None,
    request: gr.Request = None,
):
    """Advance the race gate and invalidate the finalized subtitle source."""
    revision = advance_chapter_revision(current_revision, request)
    if _chapter_session_key(request):
        _invalidate_authoritative_chapter_runtime(
            request, revision, preserve_source=False
        )
        previous_output = previous_download = previous_result = None
    return revision, *reset_chapter_source_and_artifacts(
        session_state,
        previous_output,
        previous_download,
        previous_result,
    )


def begin_chapter_edit_revision(
    current_revision,
    chapters_text,
    previous_result,
    request: gr.Request = None,
):
    """Invalidate the prior download before validating newly edited text."""
    revision = _get_chapter_request_revision(request, _CHAPTER_EDIT_ADMISSION)
    if revision is None:
        revision = advance_chapter_revision(current_revision, request)
    if not chapter_revision_is_current(revision, request):
        raise gr.Error(
            "This chapter edit was superseded before it started. Please try again."
        )
    runtime = _invalidate_authoritative_chapter_runtime(
        request, revision, preserve_source=True
    )
    if runtime is not None:
        result = {
            "duration_ms": runtime["duration_ms"],
            "base_name": runtime["base_name"],
            "source": runtime["source"],
        }
    elif _chapter_session_key(request):
        result = {}
    else:
        result = previous_result.copy() if isinstance(previous_result, dict) else {}
        remove_owned_chapter_artifact(result)
    current_text = chapters_text if isinstance(chapters_text, str) else ""
    result.update({
        "valid": False,
        "chapters": [],
        "chapters_text": current_text,
        "error": "Edited chapter text is being validated.",
    })
    return revision, None, "⏳ Validating edited chapter text...", result


def begin_english_translation_revision(
    current_revision,
    request: gr.Request = None,
):
    """Advance the shared translator-output gate before English translation."""
    return advance_chapter_revision(current_revision, request)


def sync_edited_chapters(
    chapters_text,
    previous_result,
    expected_chapter_revision=None,
    request: gr.Request = None,
):
    """Synchronize an edited chapter preview, validation state, and text artifact."""
    if not chapter_revision_is_current(expected_chapter_revision, request):
        return _unchanged_outputs(3)
    runtime = _invalidate_authoritative_chapter_runtime(
        request,
        expected_chapter_revision,
        preserve_source=True,
    )
    if runtime is not None:
        result = {
            "duration_ms": runtime["duration_ms"],
            "base_name": runtime["base_name"],
            "source": runtime["source"],
        }
    else:
        result = previous_result.copy() if isinstance(previous_result, dict) else {}
        if _chapter_session_key(request):
            result = {}
        else:
            remove_owned_chapter_artifact(result)
    current_text = chapters_text if isinstance(chapters_text, str) else ""

    try:
        if _chapter_session_key(request) and runtime is None:
            raise ChapterGenerationError(
                "The authoritative chapter source expired; generate chapters again before editing."
            )
        duration_ms = normalize_video_duration_ms(result.get("duration_ms", 0))
        chapters = parse_and_validate_chapters_text(
            current_text,
            duration_ms,
            traditional_transform=normalize_traditional_for_validation,
        )
    except (ChapterGenerationError, TypeError, ValueError) as exc:
        if not chapter_revision_is_current(expected_chapter_revision, request):
            return _unchanged_outputs(3)
        result.update({"valid": False, "chapters_text": current_text, "chapters": []})
        status = f"⚠️ Edited output is not valid for YouTube: {_normalize_error_message(exc)}"
        return None, status, result
    else:
        if not chapter_revision_is_current(expected_chapter_revision, request):
            return _unchanged_outputs(3)
        base_name = result.get("base_name", "chapters")
        try:
            output_path = create_owned_chapter_artifact(
                base_name, "youtube_chapters_edited_", current_text
            )
        except (OSError, RuntimeError) as exc:
            result.update({
                "valid": False,
                "chapters_text": current_text,
                "chapters": [],
                "error": _normalize_error_message(exc),
            })
            return None, f"⚠️ Unable to create the edited chapter download: {_normalize_error_message(exc)}", result
        result["_artifact_path"] = output_path
        if not chapter_revision_is_current(expected_chapter_revision, request):
            remove_owned_chapter_artifact(result)
            return _unchanged_outputs(3)
        result.update({"valid": True, "chapters_text": current_text, "chapters": chapters})
        status = f"✅ Edited output is valid for YouTube: {len(chapters)} chapters."
        return output_path, status, result


def stage_process_media_outputs(
    file_path,
    session_state,
    expected_chapter_revision,
    request: gr.Request = None,
):
    outputs = process_media(
        file_path,
        session_state,
        expected_chapter_revision=expected_chapter_revision,
        request=request,
    )
    return make_revisioned_output_candidate(
        outputs, expected_chapter_revision, request, channel="process"
    )


def stage_traditional_translation_outputs(
    srt_files,
    session_state,
    expected_chapter_revision,
    request: gr.Request = None,
):
    outputs = safe_translate_traditional_wrapper(
        srt_files,
        session_state,
        expected_chapter_revision,
        request,
    )
    return make_revisioned_output_candidate(
        outputs,
        expected_chapter_revision,
        request,
        channel="traditional_translation",
    )


def stage_english_translation_outputs(
    srt_files,
    api_key,
    model_name,
    custom_model,
    base_url,
    expected_chapter_revision,
    request: gr.Request = None,
):
    outputs = safe_translate_english_wrapper(
        srt_files,
        api_key,
        model_name,
        custom_model,
        base_url,
        expected_chapter_revision,
        request,
    )
    return make_revisioned_output_candidate(
        outputs,
        expected_chapter_revision,
        request,
        channel="english_translation",
    )


def stage_youtube_chapter_outputs(
    current_revision,
    source_choice,
    session_state,
    uploaded_srt_file,
    density,
    video_context,
    api_key,
    model_name,
    custom_model,
    base_url,
    video_duration_seconds,
    previous_output,
    previous_download,
    previous_result,
    request: gr.Request = None,
):
    """Reset, generate, and stage one exact validator-bound UI action."""
    channel = _CHAPTER_UI_GENERATION_ADMISSION
    admission_token = _claim_chapter_admission(channel, request)
    if admission_token is None:
        raise gr.Error(
            "A chapter request is already pending for this session. Please wait for it to finish."
        )
    publication_pending = False
    try:
        expected_chapter_revision = _get_chapter_request_revision(
            request, channel
        )
        if expected_chapter_revision is None or not chapter_revision_is_current(
            expected_chapter_revision, request
        ):
            raise gr.Error(
                "This chapter request was superseded before it started. Please try again."
            )
        reservation_token = reserve_staged_chapter_output_slot(
            expected_chapter_revision,
            request,
            channel="youtube_generation",
        )
        try:
            if _chapter_session_key(request):
                _invalidate_authoritative_chapter_runtime(
                    request,
                    expected_chapter_revision,
                    preserve_source=False,
                )
                previous_output = previous_download = previous_result = None
            reset_outputs = reset_chapter_artifacts(
                previous_output,
                previous_download,
                previous_result,
            )
            outputs = safe_youtube_chapter_wrapper(
                source_choice,
                session_state,
                uploaded_srt_file,
                density,
                video_context,
                api_key,
                model_name,
                custom_model,
                base_url,
                video_duration_seconds,
                expected_chapter_revision,
                request,
            )
            candidate = make_revisioned_output_candidate(
                outputs,
                expected_chapter_revision,
                request,
                channel="youtube_generation",
                staging_reservation_token=reservation_token,
            )
            candidate["admission_token"] = admission_token
            if not mark_chapter_admission_publishing(
                channel, request, admission_token, candidate
            ):
                discard_revisioned_output_candidate(candidate)
                raise gr.Error("This chapter request was cancelled.")
            publication_pending = True
            if not chapter_revision_is_current(
                expected_chapter_revision, request
            ):
                return (*_unchanged_outputs(5), candidate)
            return (
                expected_chapter_revision,
                *reset_outputs,
                candidate,
            )
        finally:
            release_staged_chapter_output_slot(reservation_token)
    finally:
        if not publication_pending:
            release_chapter_admission(channel, request, admission_token)


def stage_edited_chapter_outputs(
    current_revision,
    chapters_text,
    previous_result,
    request: gr.Request = None,
):
    """Stage a coalescible edit without dropping the retained latest input."""
    expected_chapter_revision = advance_chapter_revision(
        current_revision, request
    )
    if not _set_chapter_request_revision(
        request, _CHAPTER_EDIT_ADMISSION, expected_chapter_revision
    ):
        raise gr.Error("Unable to authorize this chapter edit.")
    reservation_token = reserve_staged_chapter_output_slot(
        expected_chapter_revision,
        request,
        channel="youtube_edit",
    )
    try:
        (
            expected_chapter_revision,
            download,
            status,
            pending_result,
        ) = begin_chapter_edit_revision(
            current_revision,
            chapters_text,
            previous_result,
            request,
        )
        outputs = sync_edited_chapters(
            chapters_text,
            pending_result,
            expected_chapter_revision,
            request,
        )
        candidate = make_revisioned_output_candidate(
            outputs,
            expected_chapter_revision,
            request,
            channel="youtube_edit",
            staging_reservation_token=reservation_token,
        )
        if not chapter_revision_is_current(
            expected_chapter_revision, request
        ):
            return (*_unchanged_outputs(4), candidate)
        return (
            expected_chapter_revision,
            download,
            status,
            pending_result,
            candidate,
        )
    finally:
        release_staged_chapter_output_slot(reservation_token)


def api_transcribe(file_path):
    """API wrapper: transcribe audio/video file and return text + SRT."""
    formatted_text, srt_content, srt_file_path, status_msg, _ = process_media(file_path, {})
    return {
        "text_markdown": formatted_text,
        "srt_content": srt_content,
        "srt_file_path": srt_file_path,
        "status": status_msg,
    }


def api_srt_correct(srt_content, api_key, model_name, custom_model, base_url, return_traditional):
    """API wrapper: AI-correct SRT text content."""
    original, corrected, traditional, diff_html, orig_path, corr_path, trad_path, status_msg = run_llm_correction_for_content(
        original_srt=srt_content,
        base_name="subtitles",
        api_key=api_key,
        model_name=model_name,
        custom_model=custom_model,
        base_url=base_url,
    )
    result = {
        "original_srt": original,
        "corrected_srt": corrected,
        "diff_html": diff_html,
        "original_file_path": orig_path,
        "corrected_file_path": corr_path,
        "status": status_msg,
    }
    if return_traditional:
        result["corrected_traditional_srt"] = traditional
        result["corrected_traditional_file_path"] = trad_path
    return result


def api_translate_srt_traditional(srt_files):
    """API wrapper: convert SRT file(s) to Traditional Chinese."""
    preview_original, preview_translated, download_path, _ = translate_srt_to_traditional(srt_files)
    return {
        "preview_original": preview_original,
        "preview_translated": preview_translated,
        "download_path": download_path,
        "status": "✅ Translation to Traditional Chinese completed.",
    }


def api_translate_srt_traditional_text(srt_files):
    """API wrapper: convert SRT file(s) to Traditional Chinese and return text payload."""
    srt_paths = normalize_srt_file_input(srt_files)
    if not srt_paths:
        raise gr.Error("Please upload SRT file(s).")

    items = []
    for srt_path in srt_paths:
        with open(srt_path, 'r', encoding='utf-8') as f:
            original_srt = f.read()
        if not original_srt.strip():
            continue

        translated_srt = convert_to_traditional(original_srt)
        items.append({
            "file_name": os.path.basename(srt_path),
            "translated_srt": translated_srt,
        })

    if not items:
        raise gr.Error("All uploaded SRT files are empty.")

    result = {
        "count": len(items),
        "items": items,
        "status": f"✅ Translation to Traditional Chinese completed for {len(items)} file(s).",
    }
    if len(items) == 1:
        result["translated_srt"] = items[0]["translated_srt"]
    return result


def api_translate_srt_english(srt_files, api_key, model_name, custom_model, base_url):
    """API wrapper: translate SRT file(s) to English via LLM."""
    preview_original, preview_translated, download_path, _ = translate_srt_to_english_fn(
        srt_files=srt_files,
        api_key=api_key,
        model_name=model_name,
        custom_model=custom_model,
        base_url=base_url,
    )
    return {
        "preview_original": preview_original,
        "preview_translated": preview_translated,
        "download_path": download_path,
        "status": "✅ English translation completed.",
    }


def api_youtube_chapters(
    srt_content,
    density,
    video_context,
    api_key,
    model_name,
    custom_model,
    base_url,
    video_duration_seconds=0,
    request: gr.Request = None,
):
    """API wrapper: generate validated Traditional Chinese YouTube chapter text."""
    with chapter_admission_execution(
        _CHAPTER_API_GENERATION_ADMISSION, request
    ):
        if not srt_content or not srt_content.strip():
            raise gr.Error("No SRT content provided.")
        eff_api_key, eff_base_url, effective_model = resolve_llm_config(
            api_key=api_key,
            model_name=model_name,
            custom_model=custom_model,
            base_url=base_url,
        )
        try:
            duration_ms = None
            if isinstance(video_duration_seconds, bool):
                duration_ms = parse_video_duration_seconds(video_duration_seconds)
            elif video_duration_seconds not in (None, "", 0, 0.0):
                duration_ms = parse_video_duration_seconds(video_duration_seconds)
            result = generate_youtube_chapters(
                srt_content,
                api_key=eff_api_key,
                base_url=eff_base_url,
                model=effective_model,
                density=density,
                video_context=video_context,
                title_transform=normalize_traditional_for_validation,
                video_duration_ms=duration_ms,
            )
        except ChapterGenerationError as exc:
            raise gr.Error(
                _normalize_error_message(exc, secrets=(api_key, eff_api_key))
            ) from None
        except Exception as exc:
            raise gr.Error(
                format_llm_error(
                    exc,
                    "YouTube chapter generation",
                    secrets=(api_key, eff_api_key),
                )
            ) from None
        return {
            **result,
            "status": f"✅ Generated {len(result['chapters'])} valid YouTube chapters.",
        }


def run_async_transcribe_and_correct_job(job_id, payload):
    """Background worker for transcribe + correct pipeline."""
    def process_job():
        with ASYNC_JOB_EXEC_LOCK:
            media_path = payload["file_path"]
            is_video = os.path.splitext(media_path)[1].lower() in ['.mp4', '.avi', '.mkv', '.mov']
            duration_sec = get_media_duration_seconds(media_path, is_video)
            set_async_stage(
                job_id,
                stage="transcribing",
                progress=0.2,
                message="Transcribing media to SRT...",
                eta_seconds=estimate_transcribe_eta_seconds(duration_sec),
            )
            transcribe_res = api_transcribe(media_path)
            srt_content = transcribe_res.get("srt_content", "")
            if not isinstance(srt_content, str):
                raise RuntimeError("Transcription returned invalid SRT content.")
            if not srt_content.strip():
                return no_speech_result(transcribe_res)

            set_async_stage(
                job_id,
                stage="correcting",
                progress=0.65,
                message="Running AI auto-correction...",
                eta_seconds=estimate_correction_eta_seconds(srt_content),
            )
            correct_res = api_srt_correct(
                srt_content=srt_content,
                api_key=payload.get("api_key", ""),
                model_name=payload.get("model_name", "gpt-4o-mini"),
                custom_model=payload.get("custom_model", ""),
                base_url=payload.get("base_url", ""),
                return_traditional=bool(payload.get("return_traditional", True)),
            )
            return speech_result(transcribe_res, correct_res)

    try:
        result = run_with_media_cleanup(
            process_job,
            staged_path=payload["file_path"],
            cached_upload_path=payload["cached_upload_path"],
        )
    except MediaCleanupError:
        fail_async_job(job_id, "Uploaded media cleanup failed.")
    except gr.Error as e:
        fail_async_job(job_id, _normalize_error_message(e))
    except Exception as e:
        fail_async_job(job_id, _normalize_error_message(e))
    else:
        complete_async_job(
            job_id,
            result=result,
            message="✅ Async transcribe + correction completed.",
        )


def run_async_srt_correct_job(job_id, payload):
    """Background worker for correction-only pipeline."""
    try:
        with ASYNC_JOB_EXEC_LOCK:
            srt_content = payload.get("srt_content", "")
            if not isinstance(srt_content, str) or not srt_content.strip():
                raise RuntimeError("No SRT content found for correction.")

            set_async_stage(
                job_id,
                stage="correcting",
                progress=0.35,
                message="Running AI auto-correction...",
                eta_seconds=estimate_correction_eta_seconds(srt_content),
            )
            correct_res = api_srt_correct(
                srt_content=srt_content,
                api_key=payload.get("api_key", ""),
                model_name=payload.get("model_name", "gpt-4o-mini"),
                custom_model=payload.get("custom_model", ""),
                base_url=payload.get("base_url", ""),
                return_traditional=bool(payload.get("return_traditional", True)),
            )
            final_srt = correct_res.get("corrected_srt") or srt_content
            complete_async_job(
                job_id,
                result={
                    "correct": correct_res,
                    "final_srt": final_srt,
                },
                message="✅ Async correction completed.",
            )
    except gr.Error as e:
        fail_async_job(job_id, _normalize_error_message(e))
    except Exception as e:
        fail_async_job(job_id, _normalize_error_message(e))


def start_async_worker(job_id, target_fn, payload):
    """Start a daemon thread for a queued async job."""
    def _runner():
        update_async_job(
            job_id,
            status="running",
            stage="starting",
            progress=0.05,
            eta_seconds=None,
            message="Starting async worker...",
            started_at=time.time(),
        )
        target_fn(job_id, payload)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()


def api_submit_transcribe_and_correct(file_path, api_key, model_name, custom_model, base_url, return_traditional):
    """Submit async transcribe+correct job and return job id for polling."""
    staged_media_path, cached_upload_path = stage_async_uploaded_file(file_path)
    try:
        job_id = create_async_job("transcribe_and_correct")
        payload = {
            "file_path": staged_media_path,
            "cached_upload_path": cached_upload_path,
            "api_key": api_key,
            "model_name": model_name,
            "custom_model": custom_model,
            "base_url": base_url,
            "return_traditional": return_traditional,
        }
        start_async_worker(job_id, run_async_transcribe_and_correct_job, payload)
    except Exception:
        try:
            cleanup_media_files(staged_media_path, cached_upload_path)
        except MediaCleanupError as cleanup_error:
            raise gr.Error("Uploaded media cleanup failed.") from cleanup_error
        raise
    snapshot = get_async_job_snapshot(job_id, include_result=False)
    snapshot["poll_api_name"] = "/async_job_status"
    return snapshot


def api_submit_srt_correct(srt_content, api_key, model_name, custom_model, base_url, return_traditional):
    """Submit async correction-only job and return job id for polling."""
    if not srt_content or not srt_content.strip():
        raise gr.Error("No SRT content provided.")

    job_id = create_async_job("srt_correct")
    payload = {
        "srt_content": srt_content,
        "api_key": api_key,
        "model_name": model_name,
        "custom_model": custom_model,
        "base_url": base_url,
        "return_traditional": return_traditional,
    }
    start_async_worker(job_id, run_async_srt_correct_job, payload)
    snapshot = get_async_job_snapshot(job_id, include_result=False)
    snapshot["poll_api_name"] = "/async_job_status"
    return snapshot


def api_async_job_status(job_id, include_result):
    """Poll async job status/result."""
    if not job_id or not str(job_id).strip():
        raise gr.Error("job_id is required.")
    return get_async_job_snapshot(job_id, include_result=bool(include_result))


with gr.Blocks(
    title="FunClip Pro - Gradio Edition",
    delete_cache=(3600, 3600),
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
    chapter_revision_state = gr.State(value=0)
    process_candidate_state = gr.State(value={})
    correction_candidate_state = gr.State(value={})
    traditional_candidate_state = gr.State(value={})
    english_candidate_state = gr.State(value={})
    translator_correction_candidate_state = gr.State(value={})
    chapter_generation_candidate_state = gr.State(value={})
    chapter_edit_candidate_state = gr.State(value={})

    # Hidden API endpoints for gradio_client consumers (Codex/Claude scripts, etc.).
    # These endpoints call stable wrappers and avoid UI-only chained button states.
    with gr.Group(visible=False):
        api_media_input = gr.File(file_types=["audio", "video"], file_count="single")
        api_srt_files_input = gr.File(file_types=[".srt"], file_count="multiple")
        api_srt_text_input = gr.TextArea()
        api_job_id_input = gr.Textbox()
        api_api_key_input = gr.Textbox(type="password")
        api_model_name_input = gr.Textbox(value="gpt-4o-mini")
        api_custom_model_input = gr.Textbox()
        api_base_url_input = gr.Textbox()
        api_chapter_density_input = gr.Textbox(value="Auto")
        api_video_context_input = gr.Textbox()
        api_video_duration_input = gr.Number(value=0)
        api_return_traditional_input = gr.Checkbox(value=True)
        api_include_result_input = gr.Checkbox(value=False)

        api_transcribe_output = gr.JSON()
        api_correct_output = gr.JSON()
        api_translate_trad_output = gr.JSON()
        api_translate_trad_text_output = gr.JSON()
        api_translate_en_output = gr.JSON()
        api_youtube_chapters_output = gr.JSON()
        api_submit_transcribe_correct_output = gr.JSON()
        api_submit_correct_output = gr.JSON()
        api_async_status_output = gr.JSON()

        api_transcribe_trigger = gr.Button("api_transcribe")
        api_correct_trigger = gr.Button("api_srt_correct")
        api_translate_trad_trigger = gr.Button("api_translate_traditional")
        api_translate_trad_text_trigger = gr.Button("api_translate_traditional_text")
        api_translate_en_trigger = gr.Button("api_translate_english")
        api_youtube_chapters_trigger = gr.Button("api_youtube_chapters")
        api_submit_transcribe_correct_trigger = gr.Button("api_submit_transcribe_and_correct")
        api_submit_correct_trigger = gr.Button("api_submit_srt_correct")
        api_async_status_trigger = gr.Button("api_async_job_status")

    api_transcribe_trigger.click(
        fn=api_transcribe,
        inputs=[api_media_input],
        outputs=[api_transcribe_output],
        api_name="transcribe",
        api_description="Transcribe one audio/video file to text and SRT."
    )

    api_correct_trigger.click(
        fn=api_srt_correct,
        inputs=[
            api_srt_text_input,
            api_api_key_input,
            api_model_name_input,
            api_custom_model_input,
            api_base_url_input,
            api_return_traditional_input,
        ],
        outputs=[api_correct_output],
        api_name="srt_correct",
        api_description="AI-correct SRT content and optionally return Traditional Chinese file path."
    )

    api_translate_trad_trigger.click(
        fn=api_translate_srt_traditional,
        inputs=[api_srt_files_input],
        outputs=[api_translate_trad_output],
        api_name="srt_translate_traditional",
        api_description="Convert uploaded SRT file(s) from Simplified Chinese to Traditional Chinese."
    )

    api_translate_trad_text_trigger.click(
        fn=api_translate_srt_traditional_text,
        inputs=[api_srt_files_input],
        outputs=[api_translate_trad_text_output],
        api_name="srt_translate_traditional_text",
        api_description="Convert uploaded SRT file(s) to Traditional Chinese and return translated text in JSON."
    )

    api_translate_en_trigger.click(
        fn=api_translate_srt_english,
        inputs=[
            api_srt_files_input,
            api_api_key_input,
            api_model_name_input,
            api_custom_model_input,
            api_base_url_input,
        ],
        outputs=[api_translate_en_output],
        api_name="srt_translate_english",
        api_description="Translate uploaded SRT file(s) to English using LLM."
    )

    api_youtube_chapters_trigger.click(
        fn=api_youtube_chapters,
        inputs=[
            api_srt_text_input,
            api_chapter_density_input,
            api_video_context_input,
            api_api_key_input,
            api_model_name_input,
            api_custom_model_input,
            api_base_url_input,
            api_video_duration_input,
        ],
        outputs=[api_youtube_chapters_output],
        api_name="youtube_chapters",
        api_description="Generate validated Traditional Chinese YouTube chapter text from SRT content, with optional video duration in seconds.",
        concurrency_limit=CHAPTER_GENERATION_CONCURRENCY_LIMIT,
        concurrency_id="youtube_chapters",
        validator=validate_api_chapter_admission,
    )

    api_submit_transcribe_correct_trigger.click(
        fn=api_submit_transcribe_and_correct,
        inputs=[
            api_media_input,
            api_api_key_input,
            api_model_name_input,
            api_custom_model_input,
            api_base_url_input,
            api_return_traditional_input,
        ],
        outputs=[api_submit_transcribe_correct_output],
        api_name="submit_transcribe_and_correct",
        api_description="Submit async transcribe+correct job. Poll job status using /async_job_status."
    )

    api_submit_correct_trigger.click(
        fn=api_submit_srt_correct,
        inputs=[
            api_srt_text_input,
            api_api_key_input,
            api_model_name_input,
            api_custom_model_input,
            api_base_url_input,
            api_return_traditional_input,
        ],
        outputs=[api_submit_correct_output],
        api_name="submit_srt_correct",
        api_description="Submit async SRT correction job. Poll job status using /async_job_status."
    )

    api_async_status_trigger.click(
        fn=api_async_job_status,
        inputs=[api_job_id_input, api_include_result_input],
        outputs=[api_async_status_output],
        api_name="async_job_status",
        api_description="Poll async job status by job_id. Set include_result=true to include completed payload."
    )
    
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
            
            input_file_change_event = input_file.change(
                fn=update_preview,
                inputs=[input_file],
                outputs=[video_preview, audio_preview],
                api_visibility="private",
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

            transcription_correction_source_change_event = transcription_correction_source.change(
                fn=update_srt_upload_visibility,
                inputs=[transcription_correction_source],
                outputs=[transcription_correction_upload],
                api_visibility="private",
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
                outputs=[custom_model_input],
                api_visibility="private",
            )
            
            correct_btn = gr.Button(
                "✨ Run Auto Correction",
                variant="primary",
                size="lg",
                interactive=True
            )
            
            # Now connect the process_btn click handler (after correct_btn is defined)
            process_start_event = process_btn.click(
                fn=lambda: gr.update(interactive=False, value="⏳ Processing..."),
                outputs=[process_btn],
                api_visibility="private",
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

                original, corrected, traditional, diff_html, orig_path, corr_path, trad_path, status_msg = run_llm_correction_for_content(
                    original_srt=original_srt,
                    base_name=base_name,
                    api_key=api_key,
                    model_name=model_name,
                    custom_model=custom_model,
                    base_url=base_url,
                )
                latest_source = prepare_latest_traditional_source(traditional)
                latest_video_duration_ms = None
                if source_choice == "Use output from previous step":
                    try:
                        latest_video_duration_ms = round(
                            float((state or {}).get("media_duration_seconds", 0)) * 1000
                        )
                    except (TypeError, ValueError):
                        latest_video_duration_ms = None
                updated_state = update_latest_traditional_source(
                    state,
                    latest_source,
                    f"{base_name}_traditional" if latest_source else "",
                    latest_video_duration_ms,
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
                    trad_path,
                    updated_state,
                )
            
            # Connect LLM Logic with button state management and error handling
            def safe_correction_wrapper(
                api_key,
                model_name,
                custom_model,
                base_url,
                state,
                source_choice,
                uploaded_srt,
                expected_chapter_revision=None,
                request: gr.Request = None,
            ):
                """Wrapper that catches errors and returns them along with a flag."""
                safe_state = update_latest_traditional_source(state)
                try:
                    result = run_correction_and_show(
                        api_key, model_name, custom_model, base_url, safe_state, source_choice, uploaded_srt
                    )
                    if not chapter_revision_is_current(expected_chapter_revision, request):
                        return _unchanged_outputs(9)
                    return result
                except gr.Error as e:
                    if not chapter_revision_is_current(expected_chapter_revision, request):
                        return _unchanged_outputs(9)
                    return (
                        gr.update(visible=True),
                        f"❌ {_normalize_error_message(e)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        safe_state,
                    )

                except Exception as e:
                    if not chapter_revision_is_current(expected_chapter_revision, request):
                        return _unchanged_outputs(9)
                    return (
                        gr.update(visible=True),
                        f"❌ {format_llm_error(e, 'AI auto correction')}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        safe_state,
                    )

            def stage_correction_outputs(
                api_key,
                model_name,
                custom_model,
                base_url,
                state,
                source_choice,
                uploaded_srt,
                expected_chapter_revision,
                request: gr.Request = None,
            ):
                outputs = safe_correction_wrapper(
                    api_key,
                    model_name,
                    custom_model,
                    base_url,
                    state,
                    source_choice,
                    uploaded_srt,
                    expected_chapter_revision,
                    request,
                )
                return make_revisioned_output_candidate(
                    outputs,
                    expected_chapter_revision,
                    request,
                    channel="correction",
                )
            
            correction_start_event = correct_btn.click(
                fn=lambda: gr.update(interactive=False, value="⏳ Correcting..."),
                outputs=[correct_btn],
                api_visibility="private",
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
                        "🌐 Translate to English (LLM, slower)",
                        variant="primary",
                        size="lg"
                    )
                    gr.Markdown("ℹ️ English translation uses LLM and can be slower for long or multiple SRT files.")

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
                        outputs=[srt_custom_model_input],
                        api_visibility="private",
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

            translator_correction_source_change_event = translator_correction_source.change(
                fn=update_srt_upload_visibility,
                inputs=[translator_correction_source],
                outputs=[translator_correction_upload],
                api_visibility="private",
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
                outputs=[translator_corr_custom_model_input],
                api_visibility="private",
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
            traditional_translation_start_event = translate_btn.click(
                fn=lambda: (
                    gr.update(interactive=False, value="⏳ Translating..."),
                    gr.update(value="⏳ Translating to Traditional Chinese...")
                ),
                outputs=[translate_btn, translator_status],
                api_visibility="private",
            )

            # Connect English translate button
            english_translation_start_event = translate_en_btn.click(
                fn=lambda: (
                    gr.update(interactive=False, value="⏳ Translating to English (LLM)..."),
                    gr.update(value="⏳ Translating to English...")
                ),
                outputs=[translate_en_btn, translator_status],
                api_visibility="private",
            )
            english_translation_revision_event = english_translation_start_event.then(
                fn=begin_english_translation_revision,
                inputs=[chapter_revision_state],
                outputs=[chapter_revision_state],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )
            english_translation_stage_event = english_translation_revision_event.then(
                fn=stage_english_translation_outputs,
                inputs=[
                    srt_input_file,
                    srt_api_key_input,
                    srt_model_dropdown,
                    srt_custom_model_input,
                    srt_base_url_input,
                    chapter_revision_state,
                ],
                outputs=[english_candidate_state],
                api_visibility="private",
            )
            english_translation_commit_event = english_translation_stage_event.then(
                fn=commit_english_translation_outputs,
                inputs=[english_candidate_state, chapter_revision_state],
                outputs=[original_srt_preview, translated_srt_preview, download_translated_srt, translator_state, translator_status],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )
            english_translation_commit_event.then(
                fn=lambda: gr.update(interactive=True, value="🌐 Translate to English (LLM, slower)"),
                outputs=[translate_en_btn],
                api_visibility="private",
            ).then(
                fn=update_translated_output_hint,
                inputs=[translator_state],
                outputs=[translator_stream_hint],
                api_visibility="private",
            )

            # Function to run translator correction and show results
            def run_translator_correction_and_show(
                source_choice,
                uploaded_srt_files,
                state,
                shared_session_state,
                api_key,
                model_name,
                custom_model,
                base_url,
            ):
                correction_files = resolve_translator_correction_files(
                    source_choice=source_choice,
                    translator_state=state,
                    uploaded_srt_files=uploaded_srt_files
                )
                (
                    original,
                    corrected,
                    diff_html,
                    orig_download,
                    corr_download,
                    status_msg,
                    latest_traditional_srt,
                    latest_traditional_base_name,
                ) = run_llm_correction_for_files(
                    srt_files=correction_files,
                    api_key=api_key,
                    model_name=model_name,
                    custom_model=custom_model,
                    base_url=base_url,
                )
                updated_session_state = update_latest_traditional_source(
                    shared_session_state,
                    latest_traditional_srt,
                    latest_traditional_base_name,
                )
                return (
                    gr.update(visible=True),
                    status_msg,
                    original,
                    corrected,
                    diff_html,
                    orig_download,
                    corr_download,
                    updated_session_state,
                )

            def safe_translator_correction_wrapper(
                source_choice,
                uploaded_srt_files,
                state,
                shared_session_state,
                api_key,
                model_name,
                custom_model,
                base_url,
                expected_chapter_revision=None,
                request: gr.Request = None,
            ):
                safe_shared_session_state = update_latest_traditional_source(
                    shared_session_state
                )
                try:
                    result = run_translator_correction_and_show(
                        source_choice,
                        uploaded_srt_files,
                        state,
                        safe_shared_session_state,
                        api_key,
                        model_name,
                        custom_model,
                        base_url,
                    )
                    if not chapter_revision_is_current(expected_chapter_revision, request):
                        return _unchanged_outputs(8)
                    return result
                except gr.Error as e:
                    if not chapter_revision_is_current(expected_chapter_revision, request):
                        return _unchanged_outputs(8)
                    return (
                        gr.update(visible=True),
                        f"❌ {_normalize_error_message(e)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        safe_shared_session_state,
                    )

                except Exception as e:
                    if not chapter_revision_is_current(expected_chapter_revision, request):
                        return _unchanged_outputs(8)
                    return (
                        gr.update(visible=True),
                        f"❌ {format_llm_error(e, 'Translator AI auto correction')}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        safe_shared_session_state,
                    )

            def stage_translator_correction_outputs(
                source_choice,
                uploaded_srt_files,
                state,
                shared_session_state,
                api_key,
                model_name,
                custom_model,
                base_url,
                expected_chapter_revision,
                request: gr.Request = None,
            ):
                outputs = safe_translator_correction_wrapper(
                    source_choice,
                    uploaded_srt_files,
                    state,
                    shared_session_state,
                    api_key,
                    model_name,
                    custom_model,
                    base_url,
                    expected_chapter_revision,
                    request,
                )
                return make_revisioned_output_candidate(
                    outputs,
                    expected_chapter_revision,
                    request,
                    channel="translator_correction",
                )

            translator_correction_start_event = translator_correct_btn.click(
                fn=lambda: gr.update(interactive=False, value="⏳ Correcting..."),
                outputs=[translator_correct_btn],
                api_visibility="private",
            )

        # --- TAB 3: YOUTUBE CHAPTERS ---
        with gr.Tab("📚 YouTube Chapters"):
            gr.Markdown("### Generate YouTube video chapters from Chinese subtitles")
            gr.Markdown(
                "Use the latest finalized Traditional Chinese SRT from the other tabs, or upload one UTF-8 Chinese SRT. "
                "The result is validated against YouTube's chapter timing requirements and can be pasted into a video description."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    chapter_source = gr.Radio(
                        choices=["Use latest finalized Traditional Chinese SRT", "Upload Chinese SRT"],
                        value="Use latest finalized Traditional Chinese SRT",
                        label="Chapter Input Source",
                    )
                    chapter_upload = gr.HTML(
                        html_template=(
                            '<label class="chapter-local-upload">'
                            '<span>Upload one Chinese SRT</span>'
                            '<input type="file" accept=".srt,application/x-subrip,text/plain">'
                            '<small class="chapter-upload-status">No local file selected</small>'
                            '</label>'
                        ),
                        css_template=(
                            ".chapter-local-upload { display: grid; gap: 0.5rem; } "
                            ".chapter-upload-status { color: var(--body-text-color-subdued); }"
                        ),
                        js_on_load=f"""
                            const input = element.querySelector('input[type="file"]');
                            const status = element.querySelector('.chapter-upload-status');
                            input.addEventListener('change', async () => {{
                                const file = input.files && input.files[0];
                                if (!file) {{
                                    status.textContent = 'No local file selected';
                                    trigger('change', {{name: '', data_base64: ''}});
                                    return;
                                }}
                                if (file.size > {MAX_SRT_UTF8_BYTES}) {{
                                    status.textContent = 'File exceeds the UTF-8 size limit';
                                    trigger('change', {{
                                        name: file.name,
                                        data_base64: '',
                                        error: 'The uploaded SRT exceeds the {MAX_SRT_UTF8_BYTES:,}-byte UTF-8 limit.'
                                    }});
                                    return;
                                }}
                                const bytes = new Uint8Array(await file.arrayBuffer());
                                let binary = '';
                                for (let offset = 0; offset < bytes.length; offset += 32768) {{
                                    binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
                                }}
                                status.textContent = file.name;
                                trigger('change', {{
                                    name: file.name,
                                    data_base64: btoa(binary)
                                }});
                            }});
                        """,
                        visible=False,
                    )
                    chapter_source_change_event = chapter_source.change(
                        fn=lambda source: gr.update(visible=(source == "Upload Chinese SRT")),
                        inputs=[chapter_source],
                        outputs=[chapter_upload],
                        api_visibility="private",
                    )

                    chapter_density = gr.Radio(
                        choices=["Concise", "Auto", "Detailed"],
                        value="Auto",
                        label="Chapter Density",
                    )
                    chapter_video_context = gr.Textbox(
                        label="Video title or subject (optional)",
                        placeholder="Helps the model name chapters more precisely",
                        lines=2,
                    )
                    chapter_video_duration = gr.Number(
                        label="Video duration in seconds (optional)",
                        info="Latest media uses its detected duration automatically; uploaded SRT otherwise uses its timeline extent.",
                        minimum=0,
                        value=None,
                    )

                    with gr.Accordion("⚙️ LLM Settings", open=False):
                        chapter_api_key_input = gr.Textbox(
                            label="API Key",
                            placeholder="sk-... (leave empty to use system key)",
                            type="password",
                        )
                        chapter_model_dropdown = gr.Dropdown(
                            choices=["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash", "Custom"],
                            value="gpt-4o-mini",
                            label="Model",
                            allow_custom_value=False,
                        )
                        chapter_custom_model_input = gr.Textbox(
                            label="Custom Model Name",
                            placeholder="e.g., claude-3-haiku-20240307",
                            visible=False,
                        )
                        chapter_base_url_input = gr.Textbox(
                            label="Base URL (Optional)",
                            placeholder="e.g., https://api.moonshot.cn/v1",
                            value=os.getenv("OPENAI_BASE_URL", ""),
                        )

                    chapter_model_change_event = chapter_model_dropdown.change(
                        fn=update_custom_model_visibility,
                        inputs=[chapter_model_dropdown],
                        outputs=[chapter_custom_model_input],
                        api_visibility="private",
                    )

                    chapter_generate_btn = gr.Button(
                        "✨ Generate YouTube Chapters",
                        variant="primary",
                        size="lg",
                    )
                    chapter_status = gr.Textbox(
                        label="Validation Status",
                        value="Ready",
                        interactive=False,
                        lines=3,
                    )

                with gr.Column(scale=1):
                    chapter_output = gr.Code(
                        label="YouTube Chapters (editable)",
                        language=None,
                        lines=16,
                        interactive=True,
                        wrap_lines=True,
                        show_line_numbers=False,
                        buttons=["copy"],
                    )
                    chapter_download = gr.File(
                        label="📥 Download chapters.txt",
                        interactive=False,
                        elem_classes=["download-file"],
                    )
                    chapter_result_json = gr.JSON(visible=False)

            chapter_edit_stage_event = chapter_output.input(
                fn=stage_edited_chapter_outputs,
                inputs=[chapter_revision_state, chapter_output, chapter_result_json],
                outputs=[
                    chapter_revision_state,
                    chapter_download,
                    chapter_status,
                    chapter_result_json,
                    chapter_edit_candidate_state,
                ],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
                trigger_mode="always_last",
            )
            chapter_edit_stage_event.then(
                fn=commit_chapter_edit_outputs,
                inputs=[chapter_edit_candidate_state, chapter_revision_state],
                outputs=[chapter_download, chapter_status, chapter_result_json],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )

            input_file_reset_event = input_file_change_event.then(
                fn=begin_chapter_source_revision,
                inputs=[chapter_revision_state, session_state, chapter_output, chapter_download, chapter_result_json],
                outputs=[
                    chapter_revision_state,
                    session_state,
                    chapter_output,
                    chapter_download,
                    chapter_status,
                    chapter_result_json,
                ],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )

            process_reset_event = process_start_event.then(
                fn=begin_chapter_source_revision,
                inputs=[chapter_revision_state, session_state, chapter_output, chapter_download, chapter_result_json],
                outputs=[
                    chapter_revision_state,
                    session_state,
                    chapter_output,
                    chapter_download,
                    chapter_status,
                    chapter_result_json,
                ],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )
            process_stage_event = process_reset_event.then(
                fn=stage_process_media_outputs,
                inputs=[input_file, session_state, chapter_revision_state],
                outputs=[process_candidate_state],
                api_visibility="private",
            )
            process_commit_event = process_stage_event.then(
                fn=commit_process_outputs,
                inputs=[process_candidate_state, chapter_revision_state],
                outputs=[output_text, output_srt, download_srt, status_display, session_state],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )
            process_commit_event.then(
                fn=lambda: gr.update(interactive=True, value="🚀 Start Processing"),
                outputs=[process_btn],
                api_visibility="private",
            )

            correction_reset_event = correction_start_event.then(
                fn=begin_chapter_source_revision,
                inputs=[chapter_revision_state, session_state, chapter_output, chapter_download, chapter_result_json],
                outputs=[
                    chapter_revision_state,
                    session_state,
                    chapter_output,
                    chapter_download,
                    chapter_status,
                    chapter_result_json,
                ],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )
            correction_stage_event = correction_reset_event.then(
                fn=stage_correction_outputs,
                inputs=[
                    api_key_input,
                    model_dropdown,
                    custom_model_input,
                    base_url_input,
                    session_state,
                    transcription_correction_source,
                    transcription_correction_upload,
                    chapter_revision_state,
                ],
                outputs=[correction_candidate_state],
                api_visibility="private",
            )
            correction_commit_event = correction_stage_event.then(
                fn=commit_correction_outputs,
                inputs=[correction_candidate_state, chapter_revision_state],
                outputs=[
                    correction_results,
                    correction_status,
                    original_display,
                    corrected_display,
                    diff_view,
                    download_original,
                    download_corrected,
                    download_traditional,
                    session_state,
                ],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )
            correction_commit_event.then(
                fn=lambda: gr.update(interactive=True, value="✨ Run Auto Correction"),
                outputs=[correct_btn],
                api_visibility="private",
            )

            traditional_translation_reset_event = traditional_translation_start_event.then(
                fn=begin_chapter_source_revision,
                inputs=[chapter_revision_state, session_state, chapter_output, chapter_download, chapter_result_json],
                outputs=[
                    chapter_revision_state,
                    session_state,
                    chapter_output,
                    chapter_download,
                    chapter_status,
                    chapter_result_json,
                ],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )
            traditional_translation_stage_event = traditional_translation_reset_event.then(
                fn=stage_traditional_translation_outputs,
                inputs=[srt_input_file, session_state, chapter_revision_state],
                outputs=[traditional_candidate_state],
                api_visibility="private",
            )
            traditional_translation_commit_event = traditional_translation_stage_event.then(
                fn=commit_traditional_translation_outputs,
                inputs=[traditional_candidate_state, chapter_revision_state],
                outputs=[
                    original_srt_preview,
                    translated_srt_preview,
                    download_translated_srt,
                    translator_state,
                    session_state,
                    translator_status,
                ],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )
            traditional_translation_commit_event.then(
                fn=lambda: gr.update(
                    interactive=True,
                    value="🔄 Translate to Traditional Chinese (繁體)",
                ),
                outputs=[translate_btn],
                api_visibility="private",
            ).then(
                fn=update_translated_output_hint,
                inputs=[translator_state],
                outputs=[translator_stream_hint],
                api_visibility="private",
            )

            translator_correction_reset_event = translator_correction_start_event.then(
                fn=begin_chapter_source_revision,
                inputs=[chapter_revision_state, session_state, chapter_output, chapter_download, chapter_result_json],
                outputs=[
                    chapter_revision_state,
                    session_state,
                    chapter_output,
                    chapter_download,
                    chapter_status,
                    chapter_result_json,
                ],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )
            translator_correction_stage_event = translator_correction_reset_event.then(
                fn=stage_translator_correction_outputs,
                inputs=[
                    translator_correction_source,
                    translator_correction_upload,
                    translator_state,
                    session_state,
                    translator_corr_api_key_input,
                    translator_corr_model_dropdown,
                    translator_corr_custom_model_input,
                    translator_corr_base_url_input,
                    chapter_revision_state,
                ],
                outputs=[translator_correction_candidate_state],
                api_visibility="private",
            )
            translator_correction_commit_event = translator_correction_stage_event.then(
                fn=commit_translator_correction_outputs,
                inputs=[translator_correction_candidate_state, chapter_revision_state],
                outputs=[
                    translator_correction_results,
                    translator_correction_status,
                    translator_original_display,
                    translator_corrected_display,
                    translator_diff_view,
                    translator_download_original,
                    translator_download_corrected,
                    session_state,
                ],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )
            translator_correction_commit_event.then(
                fn=lambda: gr.update(interactive=True, value="✨ Run Auto Correction"),
                outputs=[translator_correct_btn],
                api_visibility="private",
            )

            chapter_generation_stage_event = chapter_generate_btn.click(
                fn=stage_youtube_chapter_outputs,
                inputs=[
                    chapter_revision_state,
                    chapter_source,
                    session_state,
                    chapter_upload,
                    chapter_density,
                    chapter_video_context,
                    chapter_api_key_input,
                    chapter_model_dropdown,
                    chapter_custom_model_input,
                    chapter_base_url_input,
                    chapter_video_duration,
                    chapter_output,
                    chapter_download,
                    chapter_result_json,
                ],
                outputs=[
                    chapter_revision_state,
                    chapter_output,
                    chapter_download,
                    chapter_status,
                    chapter_result_json,
                    chapter_generation_candidate_state,
                ],
                api_visibility="private",
                trigger_mode="once",
                concurrency_limit=CHAPTER_GENERATION_CONCURRENCY_LIMIT,
                concurrency_id="youtube_chapters",
                validator=validate_ui_chapter_admission,
            )
            chapter_generation_commit_event = chapter_generation_stage_event.then(
                fn=commit_chapter_generation_outputs,
                inputs=[chapter_generation_candidate_state, chapter_revision_state],
                outputs=[chapter_output, chapter_download, chapter_status, chapter_result_json],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )
            chapter_generation_commit_event.then(
                fn=finalize_chapter_publication,
                inputs=[chapter_generation_candidate_state],
                outputs=None,
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )

            chapter_source_change_event.then(
                fn=begin_chapter_artifact_revision,
                inputs=[chapter_revision_state, chapter_output, chapter_download, chapter_result_json],
                outputs=[chapter_revision_state, chapter_output, chapter_download, chapter_status, chapter_result_json],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )

            chapter_model_change_event.then(
                fn=begin_chapter_artifact_revision,
                inputs=[chapter_revision_state, chapter_output, chapter_download, chapter_result_json],
                outputs=[chapter_revision_state, chapter_output, chapter_download, chapter_status, chapter_result_json],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )

            for chapter_setting in (
                chapter_density,
                chapter_video_context,
                chapter_video_duration,
                chapter_api_key_input,
                chapter_custom_model_input,
                chapter_base_url_input,
            ):
                chapter_setting.change(
                    fn=begin_chapter_artifact_revision,
                    inputs=[chapter_revision_state, chapter_output, chapter_download, chapter_result_json],
                    outputs=[chapter_revision_state, chapter_output, chapter_download, chapter_status, chapter_result_json],
                    api_visibility="private",
                    concurrency_limit=1,
                    concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
                )

            for producer_setting in (
                api_key_input,
                model_dropdown,
                custom_model_input,
                base_url_input,
                translator_corr_api_key_input,
                translator_corr_model_dropdown,
                translator_corr_custom_model_input,
                translator_corr_base_url_input,
                srt_api_key_input,
                srt_model_dropdown,
                srt_custom_model_input,
                srt_base_url_input,
            ):
                producer_setting.change(
                    fn=advance_chapter_revision,
                    inputs=[chapter_revision_state],
                    outputs=[chapter_revision_state],
                    api_visibility="private",
                    concurrency_limit=1,
                    concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
                )

            chapter_upload.change(
                fn=begin_chapter_upload_revision,
                inputs=[chapter_revision_state, chapter_output, chapter_download, chapter_result_json],
                outputs=[chapter_revision_state, chapter_output, chapter_download, chapter_status, chapter_result_json],
                api_visibility="private",
                concurrency_limit=1,
                concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
            )

            for source_change_component in (
                transcription_correction_upload,
                srt_input_file,
                translator_correction_upload,
            ):
                source_change_component.change(
                    fn=begin_chapter_source_revision,
                    inputs=[chapter_revision_state, session_state, chapter_output, chapter_download, chapter_result_json],
                    outputs=[
                        chapter_revision_state,
                        session_state,
                        chapter_output,
                        chapter_download,
                        chapter_status,
                        chapter_result_json,
                    ],
                    api_visibility="private",
                    concurrency_limit=1,
                    concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
                )

            for source_change_event in (
                transcription_correction_source_change_event,
                translator_correction_source_change_event,
            ):
                source_change_event.then(
                    fn=begin_chapter_source_revision,
                    inputs=[chapter_revision_state, session_state, chapter_output, chapter_download, chapter_result_json],
                    outputs=[
                        chapter_revision_state,
                        session_state,
                        chapter_output,
                        chapter_download,
                        chapter_status,
                        chapter_result_json,
                    ],
                    api_visibility="private",
                    concurrency_limit=1,
                    concurrency_id=CHAPTER_PUBLICATION_CONCURRENCY_ID,
                )

    # Footer
    gr.Markdown(
        """
        ---
        **FunClip Pro** - Powered by FunASR & Gradio | 
        [GitHub](https://github.com/alibaba-damo-academy/FunClip)
        """
    )


demo.queue(max_size=CHAPTER_QUEUE_MAX_SIZE)
install_chapter_queue_cleanup(demo._queue)


if __name__ == "__main__":
    demo.launch()
