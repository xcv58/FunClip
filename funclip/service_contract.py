"""Stable result-envelope helpers for the asynchronous service API."""

from collections.abc import Mapping


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
    return {
        "outcome": "speech",
        "transcribe": dict(transcribe_result),
        "correct": dict(correction_result),
        "final_srt": final_srt,
    }
