import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from funclip.llm import srt_translator


SAMPLE_SRT = """\ufeff001\r
00:00:00,000 --> 00:00:02,000\r
猜猜我在哪\r
\r
7\r
00:00:02.000 --> 00:00:04.000 align:start\r
我在澎湖\r
"""


def model_response(translations):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {"translations": translations}, ensure_ascii=False
                    )
                )
            )
        ]
    )


class SRTTranslatorTests(unittest.TestCase):
    @patch("funclip.llm.srt_translator.completion")
    def test_price_shift_is_repaired_without_retiming_or_partial_output(self, completion):
        source = "300\n00:13:14,490 --> 00:13:16,163\n谢谢谢谢\n\n301\n00:13:27,742 --> 00:13:30,770\n最后五道菜的价格是2910\n"
        moved = [{"cue_id": 300, "text": "The five dishes cost 2910"}, {"cue_id": 301, "text": "Our first seafood meal"}]
        corrected = [{"cue_id": 300, "text": "Thank you"}, {"cue_id": 301, "text": "Five dishes cost 2910 Taiwan dollars"}]
        completion.side_effect = [model_response(moved), model_response(corrected)]
        output = srt_translator.translate_srt_to_english(source)
        self.assertEqual(completion.call_count, 2)
        cues = srt_translator.parse_translation_srt(output)
        self.assertEqual([c.index for c in cues], [300, 301])
        self.assertNotIn("2910", cues[0].text)
        self.assertIn("2910", cues[1].text)
        self.assertEqual(cues[1].start_ms, 807742)
        self.assertIn("numeric_fact_added_or_moved", completion.call_args.kwargs["messages"][-1]["content"])

    @patch("funclip.llm.srt_translator.completion")
    def test_persistent_numeric_change_fails_after_the_bounded_retry(self, completion):
        source = "1\n00:00:00,000 --> 00:00:02,000\n35米\n"
        completion.return_value = model_response([{"cue_id": 1, "text": "350 meters"}])
        with self.assertRaisesRegex(srt_translator.SRTTranslationError, "numeric_fact_missing_or_changed"):
            srt_translator.translate_srt_to_english(source)
        self.assertEqual(completion.call_count, 2)

    @patch("funclip.llm.srt_translator.completion")
    def test_ambiguous_apostrophe_is_repaired_with_full_words(self, completion):
        source = "1\n00:00:00,000 --> 00:00:02,000\n准备好了\n"
        completion.side_effect = [
            model_response([{"cue_id": 1, "text": "It's ready"}]),
            model_response([{"cue_id": 1, "text": "It is ready"}]),
        ]
        output = srt_translator.translate_srt_to_english(source)
        self.assertEqual(output, source.replace("准备好了", "It is ready").rstrip("\n"))
        self.assertEqual(completion.call_count, 2)
        self.assertIn("Use full words", completion.call_args.kwargs["messages"][-1]["content"])

    @patch("funclip.llm.srt_translator.completion")
    def test_translation_rebuilds_original_indices_and_timestamps(self, completion):
        completion.return_value = model_response(
            [
                {"cue_id": 1, "text": "Guess where I am?"},
                {"cue_id": 7, "text": "I'm in Penghu."},
            ]
        )

        translated = srt_translator.translate_srt_to_english(
            SAMPLE_SRT,
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
        )

        self.assertEqual(
            translated,
            "001\n"
            "00:00:00,000 --> 00:00:02,000\n"
            "Guess where I am\n\n"
            "7\n"
            "00:00:02.000 --> 00:00:04.000 align:start\n"
            "I am in Penghu",
        )
        request = completion.call_args.kwargs
        self.assertEqual(request["api_key"], "test-key")
        self.assertEqual(request["base_url"], "https://example.test/v1")
        self.assertEqual(request["timeout"], 120.0)
        self.assertEqual(request["num_retries"], 1)
        self.assertEqual(
            request["response_format"]["json_schema"]["schema"]["properties"]
            ["translations"]["minItems"],
            2,
        )
        system_prompt = request["messages"][0]["content"]
        self.assertIn("Penghu International Fireworks Festival", system_prompt)
        self.assertIn("Guanyinting Recreation Area", system_prompt)
        prompt_payload = json.loads(request["messages"][1]["content"])
        self.assertEqual([cue["cue_id"] for cue in prompt_payload["cues"]], [1, 7])
        self.assertEqual(
            [cue["target_characters"] for cue in prompt_payload["cues"]],
            [40, 40],
        )
        self.assertIn("aim for its target_characters", system_prompt)

    @patch("funclip.llm.srt_translator.completion")
    def test_translation_batches_with_neighbor_context(self, completion):
        srt = """1
00:00:00,000 --> 00:00:02,000
第一句

2
00:00:02,000 --> 00:00:04,000
第二句

3
00:00:04,000 --> 00:00:06,000
第三句
"""
        completion.side_effect = [
            model_response(
                [
                    {"cue_id": 1, "text": "First line."},
                    {"cue_id": 2, "text": "Second line."},
                ]
            ),
            model_response([{"cue_id": 3, "text": "Third line."}]),
        ]

        with patch.object(srt_translator, "TRANSLATION_BATCH_SIZE", 2):
            translated = srt_translator.translate_srt_to_english(srt)

        self.assertEqual(completion.call_count, 2)
        second_payload = json.loads(
            completion.call_args_list[1].kwargs["messages"][1]["content"]
        )
        self.assertEqual(second_payload["context_before"], ["第一句", "第二句"])
        self.assertEqual(second_payload["context_after"], [])
        self.assertEqual([cue["cue_id"] for cue in second_payload["cues"]], [3])
        parsed = srt_translator.parse_translation_srt(translated)
        self.assertEqual([cue.index for cue in parsed], [1, 2, 3])

    @patch("funclip.llm.srt_translator.completion")
    def test_invalid_cue_set_is_retried_then_repaired(self, completion):
        completion.side_effect = [
            model_response([{"cue_id": 1, "text": "Guess where?"}]),
            model_response(
                [
                    {"cue_id": 1, "text": "Guess where I am?"},
                    {"cue_id": 7, "text": "I'm in Penghu."},
                ]
            ),
        ]

        translated = srt_translator.translate_srt_to_english(SAMPLE_SRT)

        self.assertIn("I am in Penghu", translated)
        self.assertEqual(completion.call_count, 2)
        repair_message = completion.call_args.kwargs["messages"][-1]["content"]
        self.assertIn("returned 1 translations for 2 cues", repair_message)

    @patch("funclip.llm.srt_translator.completion")
    def test_moved_ids_are_rejected_instead_of_retimed(self, completion):
        completion.return_value = model_response(
            [
                {"cue_id": 7, "text": "I'm in Penghu."},
                {"cue_id": 1, "text": "Guess where I am?"},
            ]
        )

        with self.assertRaisesRegex(
            srt_translator.SRTTranslationError,
            r"returned cue IDs \[7, 1\]; expected \[1, 7\]",
        ):
            srt_translator.translate_srt_to_english(
                SAMPLE_SRT, validation_retries=0
            )

    @patch("funclip.llm.srt_translator.completion")
    def test_malformed_markdown_response_never_becomes_an_srt(self, completion):
        completion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="```srt\n1\n00:00:00,000 --> 00:00:02,000\nWrong\n```"
                    )
                )
            ]
        )

        with self.assertRaisesRegex(
            srt_translator.SRTTranslationError,
            "did not return valid translation JSON",
        ):
            srt_translator.translate_srt_to_english(SAMPLE_SRT)
        self.assertEqual(completion.call_count, 2)

    @patch("funclip.llm.srt_translator.completion")
    def test_timing_length_is_advisory_and_does_not_trigger_retry(self, completion):
        short_srt = """1
00:01:25,917 --> 00:01:27,100
没想到来花火节
"""
        translation = "A surprise visit to the fireworks fest."
        self.assertGreater(len(translation), 23)
        completion.return_value = model_response(
            [{"cue_id": 1, "text": translation}]
        )

        translated = srt_translator.translate_srt_to_english(short_srt)

        self.assertTrue(translated.endswith("A surprise visit to the fireworks fest"))
        self.assertEqual(completion.call_count, 1)
        prompt_payload = json.loads(completion.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(prompt_payload["cues"][0]["target_characters"], 23)

    @patch("funclip.llm.srt_translator.completion")
    def test_reasonable_long_line_is_wrapped_without_retry(self, completion):
        translation = "This place feels a little like a lively night market."
        self.assertGreater(len(translation), 42)
        completion.return_value = model_response(
            [{"cue_id": 1, "text": translation}]
        )
        single_cue = """1
00:00:00,000 --> 00:00:02,000
这边有点像夜市一样
"""

        translated = srt_translator.translate_srt_to_english(single_cue)

        translated_text = translated.split("\n", 2)[2]
        lines = translated_text.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(line) <= 42 for line in lines))
        self.assertEqual(" ".join(lines), "This place feels a little like a lively night market")
        self.assertEqual(completion.call_count, 1)

    @patch("funclip.llm.srt_translator.completion")
    def test_excessive_translation_still_fails_bounded_validation(self, completion):
        completion.return_value = model_response(
            [{"cue_id": 1, "text": "word " * 30}]
        )
        single_cue = """1
00:00:00,000 --> 00:00:02,000
测试
"""

        with self.assertRaisesRegex(
            srt_translator.SRTTranslationError,
            "too long for 2 subtitle lines",
        ):
            srt_translator.translate_srt_to_english(
                single_cue, validation_retries=0
            )

    @patch("funclip.llm.srt_translator.completion")
    def test_invalid_input_fails_before_paid_generation(self, completion):
        invalid_inputs = [
            "",
            "1\n00:00:00,000 --> 00:00:02,000\nValid\n\n"
            "1\n00:00:02,000 --> 00:00:04,000\nDuplicate\n",
            "1\n00:00:02,000 --> 00:00:01,000\nBackwards\n",
            "1\nnot a timestamp\nBroken\n",
        ]
        for invalid_srt in invalid_inputs:
            with self.subTest(invalid_srt=invalid_srt), self.assertRaises(
                srt_translator.SRTTranslationError
            ):
                srt_translator.translate_srt_to_english(invalid_srt)
        completion.assert_not_called()

    def test_response_rejects_duplicate_json_keys_and_extra_fields(self):
        cues = srt_translator.parse_translation_srt(SAMPLE_SRT)
        invalid_responses = [
            '{"translations":[],"translations":[]}',
            json.dumps(
                {
                    "translations": [
                        {"cue_id": 1, "text": "Guess where?", "extra": True},
                        {"cue_id": 7, "text": "In Penghu."},
                    ]
                }
            ),
        ]
        for response in invalid_responses:
            with self.subTest(response=response), self.assertRaises(
                srt_translator.SRTTranslationError
            ):
                srt_translator.parse_translation_response(response, cues)


if __name__ == "__main__":
    unittest.main()
