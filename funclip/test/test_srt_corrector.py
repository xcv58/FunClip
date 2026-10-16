import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import unicodedata
from unittest.mock import patch

from funclip.llm import srt_corrector as corrector

SOURCE = (
    "001\n00:00:00,000 --> 00:00:02,000\n今天的天汽很好\n阿昇也來了\n\n"
    "7\n00:00:02.000 --> 00:00:04.000 align:start\n我用 iPhone 拍攝"
)
TEXTS = ["今天的天氣很好\n阿昇也來了", "我用 iPhone 拍攝"]


def response(texts=TEXTS, ids=(1, 7), *, finish="stop", refusal=None, content=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(
            content=content if content is not None else json.dumps({"corrections": [
                {"cue_id": cue_id, "text": text} for cue_id, text in zip(ids, texts)
            ]}, ensure_ascii=False), refusal=refusal))],
        model="test-model", id="response-id", usage={"prompt_tokens": 10, "completion_tokens": 5},
    )


class SRTCorrectionTests(unittest.TestCase):
    @patch.object(corrector, "completion")
    def test_only_text_changes_with_exact_headers_and_multiline_layout(self, completion):
        completion.return_value = response()
        result = corrector.request_srt_correction(SOURCE)
        self.assertEqual(result["corrected_content"], SOURCE.replace("天汽", "天氣"))
        self.assertEqual(result["validation"]["changed_cue_ids"], [1])
        payload = json.loads(completion.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(payload["cues"][0], {"cue_id": 1, "text": "今天的天汽很好\n阿昇也來了"})
        self.assertNotIn("-->", json.dumps(payload))
        self.assertTrue(completion.call_args.kwargs["response_format"]["json_schema"]["strict"])
        self.assertEqual(completion.call_args.kwargs["model"], "openai/gpt-6-luna")

    @patch.object(corrector, "completion")
    def test_bom_crlf_and_unchanged_bytes_preserved(self, completion):
        source = "\ufeff" + SOURCE.replace("\n", "\r\n") + "\r\n"
        completion.return_value = response()
        self.assertEqual(corrector.correct_srt_content(source), source.replace("天汽", "天氣"))
        completion.return_value = response(texts=["今天的天汽很好\n阿昇也來了", TEXTS[1]])
        self.assertEqual(corrector.correct_srt_content(source), source)

    @patch.object(corrector, "completion")
    def test_rejects_missing_added_duplicate_reordered_and_boolean_ids(self, completion):
        for ids, texts in [((1,), TEXTS[:1]), ((1, 7, 8), TEXTS + ["多了"]),
                           ((1, 1), TEXTS), ((7, 1), TEXTS), ((True, 7), TEXTS)]:
            with self.subTest(ids=ids), self.assertRaises(corrector.SRTCorrectionError):
                completion.return_value = response(texts, ids)
                corrector.request_srt_correction(SOURCE, validation_retries=0)

    @patch.object(corrector, "completion")
    def test_rejects_empty_text_structural_injection_reflow_and_control_characters(self, completion):
        for text in ("", "文字\n\n3\n00:00:00,000 --> 00:00:03,000\n插入", "合併了兩行", "第一行\n第二行\x00"):
            with self.subTest(text=text), self.assertRaises(corrector.SRTCorrectionError):
                completion.return_value = response([text, TEXTS[1]])
                corrector.request_srt_correction(SOURCE, validation_retries=0)

    @patch.object(corrector, "completion")
    def test_rejects_truncation_refusal_invalid_and_duplicate_key_json(self, completion):
        invalid = [response(finish="length"), response(finish="content_filter"), response(refusal="refused"),
                   response(content="```json\n{}\n```"), response(content='{"corrections":[],"corrections":[]}'),
                   response(content='{"corrections": [], "timestamps": []}'), response(content='[]')]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(corrector.SRTCorrectionError):
                completion.return_value = value
                corrector.request_srt_correction(SOURCE, validation_retries=0)

    @patch.object(corrector, "completion")
    def test_one_bounded_repair_includes_usage_for_rejected_attempt(self, completion):
        completion.side_effect = [response(TEXTS[:1], (1,)), response()]
        result = corrector.request_srt_correction(SOURCE)
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(result["usage"], {"prompt_tokens": 20, "completion_tokens": 10})
        self.assertEqual([a["validated"] for a in result["attempts"]], [False, True])
        self.assertIn("failed local validation", completion.call_args.kwargs["messages"][-1]["content"])

    @patch.object(corrector, "completion")
    def test_batched_context_is_read_only_and_last_batch_failure_rejects_whole_file(self, completion):
        with patch.object(corrector, "CORRECTION_BATCH_SIZE", 1):
            completion.side_effect = [response(TEXTS[:1], (1,)), response([TEXTS[1]], (7,))]
            result = corrector.request_srt_correction(SOURCE)
            self.assertEqual(result["corrected_content"], SOURCE.replace("天汽", "天氣"))
            first = json.loads(completion.call_args_list[0].kwargs["messages"][1]["content"])
            second = json.loads(completion.call_args_list[1].kwargs["messages"][1]["content"])
            self.assertEqual(first["context_after"][0]["cue_id"], 7)
            self.assertEqual(second["context_before"][0]["cue_id"], 1)
            completion.side_effect = [response(TEXTS[:1], (1,)), response([], ())]
            with self.assertRaisesRegex(corrector.SRTCorrectionError, "no corrected SRT was produced"):
                corrector.request_srt_correction(SOURCE, validation_retries=0)

    @patch.object(corrector, "completion")
    def test_invalid_input_never_calls_provider(self, completion):
        for source in ("", "not SRT", SOURCE.replace("7\n00:", "001\n00:"),
                       SOURCE.replace("00:00:04.000", "00:00:01.000"), SOURCE + "\n\nbroken"):
            with self.subTest(source=source), self.assertRaises(corrector.SRTCorrectionError):
                corrector.request_srt_correction(source)
        completion.assert_not_called()

    @patch.object(corrector, "completion")
    def test_protocol_overrides_are_not_allowed(self, completion):
        for options in ({"response_format": {}}, {"stream": True}, {"n": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                corrector.request_srt_correction(SOURCE, **options)
        completion.assert_not_called()

    def test_independent_final_gate_rejects_changed_timestamps_or_missing_cues(self):
        for output in (SOURCE.replace("00:00:04.000", "00:00:05.000"), SOURCE.split("\n\n")[0]):
            with self.subTest(output=output), self.assertRaises(corrector.SRTCorrectionError):
                corrector.correction_structure_report(SOURCE, output)

    @patch.object(corrector, "completion")
    def test_copying_provider_preserves_structure_or_flags_unsafe_display_text(self, completion):
        def copy_cues(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            return response(content=json.dumps({"corrections": payload["cues"]}, ensure_ascii=False))
        completion.side_effect = copy_cues
        cases = Path(__file__).resolve().parents[2] / "eval/srt_correction_cases"
        for path in sorted(cases.glob("*/original_transcribed.srt")):
            with self.subTest(path=path):
                source = path.read_text()
                try:
                    result = corrector.request_srt_correction(source)
                except corrector.SRTCorrectionError as exc:
                    self.assertIn("formatting requires review", str(exc))
                    continue
                self.assertTrue(result["validation"]["structure_preserved"])
                _, cues = corrector.parse_correction_srt(result["corrected_content"])
                self.assertFalse(any(
                    unicodedata.category(char).startswith("P")
                    for cue in cues for char in cue.text
                ))


if __name__ == "__main__":
    unittest.main()
