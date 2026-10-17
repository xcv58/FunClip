import copy
import json
import unittest

from funclip.subtitle_quality import build_srt_quality_report
from funclip.subtitle_readability import reading_units, chinese_readability_report
from funclip.review_queue import build_review_queue, include_readability_review, decide_review_item


def cue(i, text, start, end, **kwargs):
    return dict(cue_id=i, text=text, start_ms=start, end_ms=end, **kwargs)


class ChineseReadabilityTests(unittest.TestCase):
    def test_mixed_script_counting_spaces_supplementary_han_and_line_breaks(self):
        self.assertEqual(reading_units('红𫊻 AI 83\n'), 5)
        self.assertEqual(reading_units('你\r\n好'), 2)

    def test_exact_threshold_and_configuration(self):
        rows = [cue(1, '一二三四五六七八九', 0, 1000)]
        self.assertEqual(chinese_readability_report(rows)['issues'], [])
        self.assertIn('chinese_reading_speed', chinese_readability_report(rows, max_cps=8)['issue_counts'])
        for value in (False, 0, -1, float('nan'), float('inf'), '9'):
            with self.assertRaises(ValueError):
                chinese_readability_report(rows, max_cps=value)

    def test_english_only_is_not_measured_against_chinese_speed(self):
        self.assertEqual(chinese_readability_report([cue(1, 'A rather long English subtitle', 0, 1000)])['issues'], [])

    def test_safe_grouping_is_a_proposal_and_never_modifies_input(self):
        rows = [cue(7, '现在', 0, 150), cue(19, '我们走进博物馆', 200, 2000)]
        before = copy.deepcopy(rows)
        report = chinese_readability_report(rows)
        self.assertEqual(rows, before)
        proposal = report['grouping_proposals'][0]
        self.assertEqual(proposal['source_cue_ids'], [7, 19])
        self.assertFalse(proposal['applied'])
        self.assertTrue(proposal['requires_audio_and_speaker_confirmation'])
        self.assertEqual(report['applied_changes'], 0)

    def test_different_speakers_pauses_overlap_and_high_combined_speed_are_not_grouped(self):
        pairs = [[cue(1, '现在', 0, 150, speaker=0), cue(2, '这里很好', 200, 2000, speaker=1)],
                 [cue(1, '现在', 0, 150), cue(2, '这里很好', 1000, 2000)],
                 [cue(1, '现在', 0, 300), cue(2, '这里很好', 250, 2000)],
                 [cue(1, '现在我们走进博物馆', 0, 150), cue(2, '这里的建筑很漂亮', 200, 400)]]
        for rows in pairs:
            with self.subTest(rows=rows):
                self.assertEqual(chinese_readability_report(rows)['grouping_proposals'], [])

    def test_line_length_and_two_line_checks_are_reported_without_rewrite(self):
        rows = [cue(1, '这里有一些很长很长很长很长很长的字幕', 0, 5000), cue(2, '甲\n乙\n丙', 5000, 6000)]
        report = chinese_readability_report(rows)
        self.assertIn('chinese_line_length', report['issue_counts'])
        self.assertIn('chinese_line_count', report['issue_counts'])

    def test_quality_report_and_audio_queue_keep_display_ids_distinct(self):
        srt = '19\n00:00:01,000 --> 00:00:01,300\n我们今天来到博物馆\n'
        evidence = {'raw_text': '我们今天来到博物馆', 'timestamp': [], 'sentence_info': [{'text': '我们今天来到博物馆', 'start': 1000, 'end': 1300}]}
        queue = build_review_queue(evidence)
        old = copy.deepcopy(queue)
        combined = include_readability_review(queue, build_srt_quality_report(srt), evidence)
        self.assertEqual(queue, old)
        item = next(i for i in combined['items'] if i['code'] == 'chinese_reading_speed')
        self.assertEqual(item['display_cue_ids'], [19])
        self.assertEqual(item['sentence_ids'], [1])
        self.assertEqual(item['display_text'], '我们今天来到博物馆')
        changed = decide_review_item(combined, item['id'], 'needs_edit')
        self.assertFalse(changed['content_verified'])
        json.dumps(changed, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
