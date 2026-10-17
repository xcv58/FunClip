import unittest

from funclip.llm.chinese_converter import convert_srt_to_traditional, convert_to_traditional


class ContextualTraditionalTests(unittest.TestCase):
    def test_same_character_needs_different_contextual_outputs(self):
        self.assertEqual(
            convert_to_traditional("郁金香、郁达夫、浓郁、馥郁、忧郁症"),
            "鬱金香、郁達夫、濃郁、馥郁、憂鬱症",
        )
        text = "鬱金香與郁達夫 濃郁和馥郁 憂鬱症"
        self.assertEqual(convert_to_traditional(text), text)

    def test_export_retains_headers_and_body_lines_with_local_display_policy(self):
        source = "\ufeff007\r\n00:00:03,500 --> 00:00:05,500 align:start\r\n郁金香，花海！\r\n味道浓郁\r\n"
        self.assertEqual(convert_srt_to_traditional(source), source.replace("郁金香，花海！", "鬱金香 花海").replace("浓郁", "濃郁"))


if __name__ == "__main__":
    unittest.main()
