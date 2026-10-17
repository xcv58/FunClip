import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

with patch("funasr.AutoModel", return_value=object()):
    import gradio_app

from funclip.llm import srt_corrector
from funclip.service_contract import recognition_evidence, speech_result
from funclip.service_retention import run_with_media_cleanup

SOURCE = "1\n00:00:00,000 --> 00:00:02,000\n今天的天汽很好\n"
CORRECTED = SOURCE.replace("天汽", "天气")


class GradioSafeCorrectionTests(unittest.TestCase):
    def test_semantic_failure_is_visible_and_creates_no_download_artifact(self):
        source = "1\n00:00:00,000 --> 00:00:02,000\n那不能进港\n"
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content=json.dumps({"corrections": [{"cue_id": 1, "text": "那能进港"}]}), refusal=None))],
            usage=None, model="test", id="test")
        with patch.object(gradio_app, "resolve_llm_config", return_value=("key", None, "test")), \
             patch.object(srt_corrector, "completion", return_value=response) as complete, \
             patch.object(gradio_app, "create_unique_srt_path") as create_file:
            with self.assertRaisesRegex(Exception, "Meaning preservation requires review"):
                gradio_app.api_srt_correct(source, "key", "test", "", "", False)
            self.assertEqual(complete.call_count, 2)
            create_file.assert_not_called()

    def test_ui_and_api_defaults_route_luna_through_the_server_key_policy(self):
        for component in (gradio_app.model_dropdown, gradio_app.srt_model_dropdown,
                          gradio_app.translator_corr_model_dropdown, gradio_app.chapter_model_dropdown,
                          gradio_app.api_model_name_input):
            with self.subTest(component=component.label):
                self.assertEqual(component.value, "gpt-6-luna")
        choices = [value for _, value in gradio_app.model_dropdown.choices]
        self.assertEqual(choices[0], "gpt-6-luna")
        self.assertIn("gpt-4o-mini", choices)
        self.assertIn("Custom", choices)
        with patch.dict(gradio_app.os.environ, {"OPENAI_API_KEY": "server-key", "OPENAI_BASE_URL": ""}):
            self.assertEqual(
                gradio_app.resolve_llm_config("", gradio_app.api_model_name_input.value, "", ""),
                ("server-key", "", "gpt-6-luna"),
            )

    def test_async_correction_defaults_to_luna_and_honors_explicit_model_selection(self):
        with patch.object(gradio_app, "ASYNC_JOBS", {}), \
             patch.object(gradio_app, "api_srt_correct", return_value={"corrected_srt": CORRECTED}) as correct:
            for extra, expected in (({}, "gpt-6-luna"), ({"model_name": "gpt-4o-mini"}, "gpt-4o-mini")):
                job_id = gradio_app.create_async_job("srt-correct")
                gradio_app.run_async_srt_correct_job(job_id, {"srt_content": SOURCE, **extra})
                self.assertEqual(correct.call_args.kwargs["model_name"], expected)
                result = gradio_app.get_async_job_snapshot(job_id, include_result=True)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["result"]["final_srt"], CORRECTED)

    def test_failed_async_correction_retains_raw_evidence_and_still_cleans_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upload_root = root / "uploads"
            cached = upload_root / "digest" / "source.wav"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"cached")
            staged = root / "staged.wav"
            staged.write_bytes(b"staged")
            transcribe = {"srt_content": SOURCE, "recognition_evidence": recognition_evidence("text", SOURCE, {})}
            with patch.object(gradio_app, "ASYNC_JOBS", {}), \
                 patch.object(gradio_app, "get_media_duration_seconds", return_value=2), \
                 patch.object(gradio_app, "api_transcribe", return_value=transcribe), \
                 patch.object(gradio_app, "api_srt_correct", side_effect=srt_corrector.SRTCorrectionError("missing cue")) as correct, \
                 patch.object(gradio_app, "run_with_media_cleanup", side_effect=lambda operation, **kwargs: run_with_media_cleanup(operation, upload_root=upload_root, **kwargs)):
                job_id = gradio_app.create_async_job("transcribe-and-correct")
                gradio_app.run_async_transcribe_and_correct_job(job_id, {
                    "file_path": str(staged), "cached_upload_path": str(cached)})
                result = gradio_app.get_async_job_snapshot(job_id, include_result=True)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(correct.call_args.kwargs["model_name"], "gpt-6-luna")
                self.assertNotIn("result", result)
                self.assertEqual(result["transcription"]["recognition_evidence"]["original_srt"], SOURCE)
                self.assertNotIn("transcription", gradio_app.get_async_job_snapshot(job_id))
            self.assertFalse(staged.exists())
            self.assertFalse(cached.exists())

    def test_recognition_evidence_is_an_independent_json_snapshot_without_audio(self):
        state = {"recog_res_raw": "今天 的 天汽 很好", "timestamp": np.array([[0, 300], [300, 800]]),
                 "sentences": [{"text": "今天的天汽很好", "start": 0, "end": 2000}],
                 "audio_input": np.zeros(16000), "api_key": "must not be retained"}
        evidence = recognition_evidence("今天的天汽很好", SOURCE, state)
        state["sentences"][0]["text"] = "changed later"
        self.assertEqual(evidence["sentence_info"][0]["text"], "今天的天汽很好")
        self.assertEqual(evidence["timestamp"], [[0, 300], [300, 800]])
        self.assertNotIn("audio_input", evidence)
        self.assertNotIn("api_key", evidence)
        self.assertEqual(evidence["original_srt"], SOURCE)
        missing = recognition_evidence("", "", {})
        self.assertEqual(missing["available_fields"], [])
        self.assertIsNone(missing["timestamp"])

    def test_transcription_api_keeps_existing_fields_and_exposes_saved_raw_evidence(self):
        state = {"recog_res_raw": "今天 的 天汽 很好", "timestamp": [[0, 300]],
                 "sentences": [{"text": "今天的天汽很好", "start": 0, "end": 2000}],
                 "audio_input": np.zeros(16000)}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(gradio_app, "get_media_duration_seconds", return_value=2), \
             patch.object(gradio_app, "VideoClipper") as clipper, \
             patch.object(gradio_app, "create_unique_srt_path", return_value=str(Path(directory) / "raw.srt")):
            clipper.return_value.video_recog.return_value = ("今天的天汽很好", SOURCE, state)
            result = gradio_app.api_transcribe(str(Path(directory) / "source.mp4"))
            self.assertTrue({"text_markdown", "srt_content", "srt_file_path", "status"} <= result.keys())
            self.assertEqual(result["srt_content"], SOURCE)
            self.assertEqual(result["quality_report"]["status"], "clear")
            self.assertFalse(result["quality_report"]["content_verified"])
            evidence_path = Path(result["recognition_evidence_file_path"])
            try:
                self.assertEqual(json.loads(evidence_path.read_text()), result["recognition_evidence"])
                combined = speech_result(result, {"corrected_srt": CORRECTED})
                self.assertEqual(combined["transcribe"]["recognition_evidence"]["original_srt"], SOURCE)
                self.assertEqual(combined["final_srt"], CORRECTED)
            finally:
                evidence_path.unlink(missing_ok=True)

    def test_correction_api_contract_is_additive_and_invalid_output_produces_no_file(self):
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content=json.dumps({"corrections": [{"cue_id": 1, "text": "今天的天气很好"}]}), refusal=None))],
            usage=None, model="test", id="test")
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(gradio_app, "resolve_llm_config", return_value=("key", None, "gpt-4o-mini")), \
             patch.object(srt_corrector, "completion", return_value=response), \
             patch.object(gradio_app, "create_unique_srt_path", side_effect=lambda name, prefix: str(Path(directory) / f"{prefix}{name}.srt")) as create_file:
            result = gradio_app.api_srt_correct(SOURCE, "key", "gpt-4o-mini", "", "", True)
            self.assertEqual(result["corrected_srt"], CORRECTED)
            self.assertEqual(result["original_srt"], SOURCE)
            self.assertTrue(result["correction_validation"]["structure_preserved"])
            self.assertEqual(result["correction_validation"]["cue_count"], 1)
            self.assertEqual(result["quality_report"]["cue_count"], 1)
            self.assertIn("corrected_traditional_srt", result)
            create_file.reset_mock()
            response.choices[0].message.content = '{"corrections": []}'
            with self.assertRaises(Exception):
                gradio_app.api_srt_correct(SOURCE, "key", "gpt-4o-mini", "", "", False)
            create_file.assert_not_called()

    def test_transcription_warning_does_not_rewrite_or_drop_the_caption(self):
        source = "125\n00:06:44,320 --> 00:10:05,130\n七龙珠之长湖夏日花火大一年级\n"
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(gradio_app, "get_media_duration_seconds", return_value=1200), \
             patch.object(gradio_app, "VideoClipper") as clipper, \
             patch.object(gradio_app, "create_unique_srt_path", return_value=str(Path(directory) / "raw.srt")):
            clipper.return_value.video_recog.return_value = ("识别原文", source, {})
            result = gradio_app.api_transcribe(str(Path(directory) / "video.mov"))
            try:
                self.assertEqual(result["srt_content"], source)
                self.assertEqual(Path(result["srt_file_path"]).read_text(), source)
                self.assertEqual(result["quality_report"]["issue_counts"], {"long_cue": 1, "sparse_long_cue": 1})
                self.assertIn("Review suggested", result["status"])
                self.assertIn("cue 125", result["status"])
            finally:
                Path(result["recognition_evidence_file_path"]).unlink(missing_ok=True)

    def test_file_translation_reports_are_per_file_and_keep_inputs_unchanged(self):
        source = "1\n00:00:00,000 --> 00:00:12,000\n郁金香花海\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.srt"
            path.write_text(source)
            result = gradio_app.api_translate_srt_traditional_text([str(path)])
            self.assertIn("鬱金香", result["translated_srt"])
            self.assertEqual(result["items"][0]["quality_report"]["issue_counts"], {"long_cue": 1})
            self.assertIn("Review suggested", result["status"])
            download = gradio_app.api_translate_srt_traditional([str(path)])
            self.assertEqual(download["quality_reports"][0]["file_name"], "source.srt")
            self.assertIn("Review suggested", download["status"])
            english = source.replace("郁金香花海", "A sea of tulips")
            with patch.object(gradio_app, "resolve_llm_config", return_value=("", "", "test")), \
                 patch.object(gradio_app, "translate_srt_to_english", return_value=english):
                translated = gradio_app.api_translate_srt_english([str(path)], "", "test", "", "")
            self.assertEqual(translated["preview_translated"], english)
            self.assertEqual(translated["quality_reports"][0]["report"]["issue_counts"], {"long_cue": 1})
            self.assertIn("Review suggested", translated["status"])
            self.assertEqual(path.read_text(), source)


if __name__ == "__main__":
    unittest.main()
