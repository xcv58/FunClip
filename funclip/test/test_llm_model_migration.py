import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from litellm import get_llm_provider

from funclip.llm import srt_corrector, srt_translator, youtube_chapters
from funclip.llm.chinese_converter import normalize_traditional_for_validation
from funclip.llm.model_config import completion_options


def response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=None,
        model="gpt-6-luna",
        id="test-response",
    )


class ModelMigrationTests(unittest.TestCase):
    def test_luna_resolves_with_the_installed_litellm_provider_registry(self):
        for model in ("gpt-6-luna", "openai/gpt-6-luna"):
            with self.subTest(model=model):
                options = completion_options(model, max_tokens=1024)
                self.assertEqual(get_llm_provider(options["model"])[1], "openai")
                self.assertEqual(options["max_completion_tokens"], 1024)
                self.assertNotIn("max_tokens", options)

    @patch("funclip.llm.srt_corrector.completion")
    def test_correction_forwards_luna_options_and_keeps_requested_model(self, completion):
        source = "1\n00:00:00,000 --> 00:00:02,000\n澎湖花火節。"
        completion.return_value = response(json.dumps({"corrections": [{"cue_id": 1, "text": "澎湖花火節。"}]}))
        result = srt_corrector.request_srt_correction(source)
        self.assertEqual(result["corrected_content"], source.replace("。", ""))
        self.assertEqual(result["requested_model"], "gpt-6-luna")
        self.assertEqual(completion.call_args.kwargs["reasoning_effort"], "none")
        self.assertIn("reasoning_effort", completion.call_args.kwargs["allowed_openai_params"])

    @patch("funclip.llm.srt_translator.completion")
    def test_luna_translation_preserves_nonsequential_cue_ids_and_timestamps(self, completion):
        source = "001\n00:00:00,000 --> 00:00:02,000\n你好\n\n7\n00:00:02.000 --> 00:00:04.000 align:start\n澎湖"
        completion.return_value = response(json.dumps({"translations": [
            {"cue_id": 1, "text": "Hello."}, {"cue_id": 7, "text": "Penghu."},
        ]}))
        translated = srt_translator.translate_srt_to_english(source)
        self.assertEqual(translated, source.replace("你好", "Hello").replace("澎湖", "Penghu"))
        self.assertIn("max_completion_tokens", completion.call_args.kwargs)
        self.assertNotIn("max_tokens", completion.call_args.kwargs)
        self.assertEqual(completion.call_args.kwargs["model"], "openai/gpt-6-luna")

    @patch("funclip.llm.youtube_chapters.completion")
    def test_luna_chapters_use_the_existing_schema_and_validation(self, completion):
        source = (
            "1\n00:00:00,000 --> 00:00:20,000\n介紹澎湖馬公散步路線和集合地點。\n\n"
            "2\n00:00:20,000 --> 00:00:40,000\n我們在觀音亭休憩區看煙火，介紹拍攝位置。\n\n"
            "3\n00:00:40,000 --> 00:01:00,000\n明天搭船去七美，出發前確認船班和風浪。"
        )
        completion.return_value = response(json.dumps({"chapters": [
            {"cue_id": 1, "title": "澎湖馬公散步路線"},
            {"cue_id": 2, "title": "觀音亭煙火拍攝位置"},
            {"cue_id": 3, "title": "七美船班與風浪確認"},
        ]}, ensure_ascii=False))
        result = youtube_chapters.generate_youtube_chapters(source)
        self.assertEqual(len(result["chapters"]), 3)
        self.assertTrue(result["chapters_text"].startswith("00:00 "))
        request = completion.call_args.kwargs
        self.assertEqual(request["response_format"], youtube_chapters.CHAPTER_RESPONSE_FORMAT)
        self.assertIn("max_completion_tokens", request)
        self.assertNotIn("max_tokens", request)
        self.assertEqual(request["model"], "openai/gpt-6-luna")

    def test_other_providers_keep_their_existing_routing_and_token_parameters(self):
        for model in ("gpt-4o-mini", "gemini/gemini-2.5-flash", "openai/custom-gateway", "ollama_chat/gemma4"):
            with self.subTest(model=model):
                self.assertEqual(completion_options(model, max_tokens=1024), {"model": model, "max_tokens": 1024})

    @patch("funclip.llm.srt_corrector.completion")
    def test_explicit_legacy_model_keeps_its_model_and_request_parameters(self, completion):
        source = "1\n00:00:00,000 --> 00:00:02,000\n你好"
        completion.return_value = response(json.dumps({"corrections": [{"cue_id": 1, "text": "你好"}]}))
        result = srt_corrector.request_srt_correction(source, model="gpt-4o-mini")
        self.assertEqual(result["requested_model"], "gpt-4o-mini")
        request = completion.call_args.kwargs
        self.assertEqual(request["model"], "gpt-4o-mini")
        self.assertIn("max_tokens", request)
        self.assertNotIn("max_completion_tokens", request)
        self.assertNotIn("reasoning_effort", request)

    def test_finalized_florida_name_is_accepted_without_allowing_simplified_titles(self):
        title = "佛羅里達 FIRE 優勢與風險"
        self.assertEqual(normalize_traditional_for_validation(title), title)
        self.assertEqual(
            youtube_chapters.validate_chapter_title(
                title, strict_plain=True,
                traditional_transform=normalize_traditional_for_validation,
            ),
            title,
        )
        for simplified in ("佛罗里达退休生活", "佛羅里達退休计划"):
            with self.subTest(title=simplified), self.assertRaises(youtube_chapters.ChapterGenerationError):
                youtube_chapters.validate_chapter_title(
                    simplified, strict_plain=True,
                    traditional_transform=normalize_traditional_for_validation,
                )


if __name__ == "__main__":
    unittest.main()
