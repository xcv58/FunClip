import tempfile
import unittest
from pathlib import Path

from funclip.service_contract import (
    no_speech_result,
    normalized_recognition_payload,
    speech_result,
)
from funclip.service_retention import run_with_media_cleanup


class ServiceContractTests(unittest.TestCase):
    def test_model_result_without_speech_normalizes_to_empty_sentences(self):
        payload = normalized_recognition_payload([
            {"text": "", "raw_text": "", "timestamp": []}
        ])

        self.assertEqual(payload["sentence_info"], [])

    def test_detected_speech_without_sentence_timing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "omitted sentence timing"):
            normalized_recognition_payload([
                {"text": "spoken", "raw_text": "spoken", "timestamp": [0, 1]}
            ])

    def test_empty_success_becomes_explicit_no_speech_without_correction(self):
        transcribe = {
            "srt_content": " \n",
            "language": "zh",
            "status": "completed",
        }

        result = no_speech_result(transcribe)

        self.assertEqual(result["outcome"], "no-speech")
        self.assertEqual(result["transcribe"], transcribe)
        self.assertIsNone(result["correct"])
        self.assertEqual(result["final_srt"], "")

    def test_no_speech_rejects_cues(self):
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            no_speech_result({
                "srt_content": (
                    "1\n00:00:00,000 --> 00:00:01,000\nspoken\n"
                )
            })

    def test_speech_result_is_explicit_and_keeps_corrected_srt(self):
        transcribe = {"srt_content": "raw cue", "language": "en"}
        correction = {"corrected_srt": "corrected cue"}

        result = speech_result(transcribe, correction)

        self.assertEqual(result["outcome"], "speech")
        self.assertEqual(result["transcribe"], transcribe)
        self.assertEqual(result["correct"], correction)
        self.assertEqual(result["final_srt"], "corrected cue")

    def test_no_speech_success_still_cleans_staged_and_uploaded_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upload_root = root / "gradio"
            cached = upload_root / "digest" / "source.wav"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"cached")
            staged = root / "staged.wav"
            staged.write_bytes(b"staged")

            result = run_with_media_cleanup(
                lambda: no_speech_result({"srt_content": "", "language": "zh"}),
                staged_path=staged,
                cached_upload_path=cached,
                upload_root=upload_root,
            )

            self.assertEqual(result["outcome"], "no-speech")
            self.assertFalse(staged.exists())
            self.assertFalse(cached.exists())


if __name__ == "__main__":
    unittest.main()
