import copy
import hashlib
import unittest

from funclip.llm.chinese_converter import convert_srt_to_traditional
from funclip.service_contract import recognition_evidence
from funclip.subtitle_style import SubtitleFormattingError, format_srt_content, format_subtitle_text
from funclip.utils.subtitle_utils import generate_srt, generate_srt_clip


class SubtitleStyleTests(unittest.TestCase):
    def test_separators_and_line_breaks(self):
        self.assertEqual(
            format_subtitle_text('  「你好」，七美！\n今天…… 出發；好嗎？ '),
            '你好 七美\n今天 出發 好嗎',
        )

    def test_numbers_keep_sign_magnitude_and_percentage(self):
        self.assertEqual(format_subtitle_text('降低-3.5%'), '降低负百分之3点5')
        self.assertEqual(format_subtitle_text('價格是1,234.5'), '價格是1234點5')
        self.assertEqual(format_subtitle_text('Down -3.5 %'), 'Down negative 3 point 5 percent')
        self.assertEqual(format_subtitle_text('Up +1,000.25‰'), 'Up positive 1000 point 25 per mille')

    def test_supported_contractions_expand_without_losing_negation(self):
        self.assertEqual(
            format_subtitle_text("I'm ready, don't worry. You're here!"),
            'I am ready do not worry You are here',
        )
        self.assertEqual(format_subtitle_text('We won’t leave.'), 'We will not leave')
        self.assertEqual(format_subtitle_text('我说don＇t worry'), '我说do not worry')
        self.assertEqual(format_subtitle_text('U.S. market'), 'US market')

    def test_ambiguous_meaning_and_empty_lines_require_review(self):
        for text in ("Jenny's video", "he's ready", 'dogs’ toys', 'example.com',
                     'mail@example.com', 'https://example.com', '网站example.com', '3/4', '12:30',
                     '2026-10-16', '3–5', '$3.5', '1,23', '.5', '--3', 'AT&T',
                     'C++', 'word\u200bword', '……', '第一行\n。'):
            with self.subTest(text=text), self.assertRaises(SubtitleFormattingError):
                format_subtitle_text(text)

    def test_srt_headers_bom_mixed_endings_and_body_line_counts_survive(self):
        source = ('\ufeff001\r\n00:00:00,000 --> 00:00:02,000 align:start\n'
                  '你好，七美！\r\n真的嗎？\r\n\r\n'
                  '7\n00:00:02.000 --> 00:00:04.000\r\nNext, stop.\n')
        expected = ('\ufeff001\r\n00:00:00,000 --> 00:00:02,000 align:start\n'
                    '你好 七美\r\n真的嗎\r\n\r\n'
                    '7\n00:00:02.000 --> 00:00:04.000\r\nNext stop\n')
        self.assertEqual(format_srt_content(source), expected)
        self.assertEqual(format_srt_content(expected), expected)

    def test_traditional_conversion_changes_bodies_only(self):
        source = '001\r\n00:00:00,000 --> 00:00:02,000\r\n这个价格是3.5%。\r\n'
        self.assertEqual(
            convert_srt_to_traditional(source),
            '001\r\n00:00:00,000 --> 00:00:02,000\r\n這個價格是百分之3點5\r\n',
        )

    def test_transcription_renderer_retains_independent_original_evidence(self):
        sentences = [{'text': '你好，七美！', 'timestamp': [[0, 200], [200, 400], [400, 600], [600, 800]]}]
        original = copy.deepcopy(sentences)
        raw = generate_srt(sentences, format_output=False)
        rendered = generate_srt(sentences)
        evidence = recognition_evidence('你好，七美！', rendered, {
            'sentences': sentences, 'recognition_srt': raw,
        })
        self.assertIn('你好 七美', rendered)
        self.assertIn('你好，七美！', evidence['original_srt'])
        self.assertEqual(evidence['rendered_srt'], rendered)
        self.assertEqual(evidence['rendered_srt_sha256'], hashlib.sha256(rendered.encode()).hexdigest())
        self.assertEqual(sentences, original)

    def test_full_caption_clip_preserves_numeric_meaning_and_source_state(self):
        sentences = [{'text': '收益3.5%', 'timestamp': [[1000, 1500], [1500, 2000], [2000, 2500]]}]
        original = copy.deepcopy(sentences)
        srt, _, _ = generate_srt_clip(sentences, 1, 3)
        self.assertIn('收益百分之3点5', srt)
        self.assertEqual(sentences, original)
        with self.assertRaises(SubtitleFormattingError):
            generate_srt_clip(sentences, 1.2, 2.5)
        self.assertEqual(sentences, original)
