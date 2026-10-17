"""Stable result-envelope helpers for the asynchronous service API."""

from collections.abc import Mapping
import hashlib
import json
from funclip.subtitle_style import SUBTITLE_STYLE


def recognition_evidence(text, srt_content, recognition_state):
    """Snapshot ASR evidence without retaining media arrays or unrelated session state."""
    state = recognition_state or {}

    def json_value(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if hasattr(value, "tolist"):
            return json_value(value.tolist())
        raise TypeError("Unsupported value in ASR evidence.")

    original_srt = state.get("recognition_srt", srt_content)
    evidence = {
        "version": 1,
        "text": text,
        "raw_text": json_value(state.get("recog_res_raw")),
        "timestamp": json_value(state.get("timestamp")),
        "sentence_info": json_value(state.get("sentences")),
        "available_fields": [key for key in ("recog_res_raw", "timestamp", "sentences") if key in state],
        "original_srt": original_srt,
        "original_srt_sha256": hashlib.sha256(original_srt.encode("utf-8")).hexdigest(),
        "rendered_srt": srt_content,
        "rendered_srt_sha256": hashlib.sha256(srt_content.encode("utf-8")).hexdigest(),
        "subtitle_style": SUBTITLE_STYLE,
    }
    # Round-trip creates an independent snapshot and verifies finite JSON values.
    return json.loads(json.dumps(evidence, ensure_ascii=False, allow_nan=False))


def normalized_recognition_payload(recognition_result):
    """Validate one FunASR result and normalize a proven empty recognition."""
    if (
        not isinstance(recognition_result, list)
        or len(recognition_result) != 1
        or not isinstance(recognition_result[0], Mapping)
    ):
        raise ValueError("Recognition result has an invalid shape.")
    payload = dict(recognition_result[0])
    sentences = payload.get("sentence_info")
    if sentences is None:
        speech_fields = (
            payload.get("text"),
            payload.get("raw_text"),
            payload.get("timestamp"),
        )
        if any(value for value in speech_fields):
            raise ValueError(
                "Recognition result omitted sentence timing for detected speech."
            )
        payload["sentence_info"] = []
    elif not isinstance(sentences, list):
        raise ValueError("Recognition sentence timing must be a list.")
    return payload


def no_speech_result(transcribe_result):
    """Return an explicit no-speech result after a successful empty transcription."""
    if not isinstance(transcribe_result, Mapping):
        raise TypeError("Transcription result must be a mapping.")
    srt_content = transcribe_result.get("srt_content")
    if not isinstance(srt_content, str):
        raise TypeError("Transcription SRT content must be a string.")
    if srt_content.strip():
        raise ValueError("No-speech result cannot contain subtitle cues.")
    return {
        "outcome": "no-speech",
        "transcribe": dict(transcribe_result),
        "correct": None,
        "final_srt": "",
    }


def speech_result(transcribe_result, correction_result):
    """Return an explicit speech result after transcription and correction."""
    if not isinstance(transcribe_result, Mapping):
        raise TypeError("Transcription result must be a mapping.")
    if not isinstance(correction_result, Mapping):
        raise TypeError("Correction result must be a mapping.")
    srt_content = transcribe_result.get("srt_content")
    if not isinstance(srt_content, str) or not srt_content.strip():
        raise ValueError("Speech result requires subtitle cues.")
    corrected_srt = correction_result.get("corrected_srt")
    final_srt = (
        corrected_srt
        if isinstance(corrected_srt, str) and corrected_srt.strip()
        else srt_content
    )
    result = {
        "outcome": "speech",
        "transcribe": dict(transcribe_result),
        "correct": dict(correction_result),
        "final_srt": final_srt,
    }
    if "quality_report" in correction_result:
        result["quality_report"] = correction_result["quality_report"]
    return result
