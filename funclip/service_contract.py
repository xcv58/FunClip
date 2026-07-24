"""Stable result-envelope helpers for the asynchronous service API."""

from collections.abc import Mapping


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
    return {
        "outcome": "speech",
        "transcribe": dict(transcribe_result),
        "correct": dict(correction_result),
        "final_srt": final_srt,
    }
