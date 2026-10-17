import hashlib
import json
import unittest
from unittest.mock import patch

from funclip.meaning_alignment import (meaning_alignment_report, srt_alignment_report,
    validate_meaning_alignment, MeaningAlignmentError, alignment_prompt_context)
from funclip.llm import srt_translator, srt_corrector
from funclip.test.test_srt_translator import model_response
from funclip.test.test_srt_corrector import response


def srt(rows):
    return '\n\n'.join(f'{i}\n00:00:0{n},000 --> 00:00:0{n + 1},000\n{t}' for n, (i, t) in enumerate(rows))


class MeaningAlignmentTests(unittest.TestCase):
    def test_unique_place_movement_binds_source_time_and_text_hash(self):
        source, target = [(7, '今天我们在台湾'), (19, '天气很好')], [(7, 'The weather is nice'), (19, 'We are in Taiwan today')]
        report = srt_alignment_report(srt(source), srt(target), mode='translation')
        self.assertEqual(report['status'], 'blocked')
        issue = next(i for i in report['issues'] if i['code'] == 'anchor_moved_between_cues')
        self.assertEqual((issue['cue_id'], issue['candidate_cue_ids'], issue['start_ms'], issue['end_ms']), (7, [19], 0, 1000))
        row = report['coverage'][0]
        self.assertEqual(row['source_cue_ids'], [7])
        self.assertEqual(row['source_text_sha256'], hashlib.sha256(source[0][1].encode()).hexdigest())
        self.assertFalse(row['semantic_coverage_verified'])
        json.dumps(report, allow_nan=False)

    def test_missing_anchor_and_contextual_pronoun_are_advisory(self):
        report = validate_meaning_alignment([(1, '我们来到槟城的博物馆')], [(1, 'We arrived here')], mode='translation')
        self.assertEqual(report['status'], 'needs_review')
        self.assertTrue(all(i['severity'] == 'review' for i in report['issues']))

    def test_traditional_spelling_and_supported_synonyms_are_preserved(self):
        report = validate_meaning_alignment([(1, '我們在臺灣搭計程車去機場')],
            [(1, 'We take a cab to the airport in Taiwan')], mode='translation')
        self.assertEqual(report['issues'], [])

    def test_us_pronoun_is_not_a_united_states_anchor(self):
        report = meaning_alignment_report([(1, '他们在等我们')], [(1, 'They are waiting for us')], mode='translation')
        self.assertEqual(report['issues'], [])

    def test_uppercase_us_location_and_currency_names_are_distinct(self):
        for source, target in [('我们从美国出发', 'We left the US'),
                               ('十五美元和三百台币', 'fifteen US dollars and three hundred Taiwan dollars'),
                               ('新加坡元', 'Singapore dollars'),
                               ('台湾使用台币', 'Taiwan uses Taiwan dollars')]:
            with self.subTest(source=source):
                report = meaning_alignment_report([(1, source)], [(1, target)], mode='translation')
                self.assertEqual(report['issues'], [])

    def test_repeated_places_are_ambiguous_rather_than_blocked(self):
        report = validate_meaning_alignment([(1, '台湾'), (2, '台湾'), (3, '这里')],
            [(1, 'Here'), (2, 'Taiwan'), (3, 'Taiwan')], mode='translation')
        self.assertNotEqual(report['status'], 'blocked')

    def test_explicit_action_reversal_is_blocked(self):
        with self.assertRaises(MeaningAlignmentError) as caught:
            validate_meaning_alignment([(1, '我买了这件衣服')], [(1, 'I sold this outfit')], mode='translation')
        self.assertIn('action_reversed', str(caught.exception))

    def test_unknown_vocabulary_is_not_certified(self):
        report = meaning_alignment_report([(1, '七彩烟霞')], [(1, 'Something else')], mode='translation')
        self.assertEqual(report['coverage'][0]['supported_source_anchors'], [])
        self.assertFalse(report['content_verified'])
        self.assertFalse(report['coverage'][0]['semantic_coverage_verified'])

    def test_literal_chinese_phrase_movement_is_blocked(self):
        with self.assertRaises(MeaningAlignmentError) as caught:
            validate_meaning_alignment([(1, '她今天要去看展览'), (2, '天气真好')],
                [(1, '她今天'), (2, '要去看展览天气真好')], mode='correction')
        self.assertIn('literal_content_moved', str(caught.exception))

    def test_typo_and_punctuation_changes_remain_allowed(self):
        report = validate_meaning_alignment([(1, '今天的天汽很好。')], [(1, '今天的天气很好')], mode='correction')
        self.assertEqual(report['issues'], [])

    def test_bad_pairing_or_timing_is_never_accepted_as_coverage(self):
        with self.assertRaises(ValueError):
            meaning_alignment_report([(1, '台湾')], [(2, 'Taiwan')], mode='translation')
        with self.assertRaises(ValueError):
            srt_alignment_report(srt([(1, '台湾')]), srt([(1, 'Taiwan')]).replace('00:00:01,000', '00:00:02,000'), mode='translation')

    def test_prompt_reference_is_extracted_locally(self):
        self.assertEqual(alignment_prompt_context([(7, '去机场')]), [{'cue_id': 7, 'concepts': ['airport']}])

    @patch.object(srt_translator, 'completion')
    def test_new_alignment_failure_uses_existing_bounded_repair(self, completion):
        source = srt([(1, '我们到了台湾'), (2, '风景真美')])
        completion.side_effect = [model_response([{'cue_id': 1, 'text': 'The view is nice'}, {'cue_id': 2, 'text': 'We are in Taiwan'}]),
                                 model_response([{'cue_id': 1, 'text': 'We are in Taiwan'}, {'cue_id': 2, 'text': 'The view is nice'}])]
        result = srt_translator.translate_srt_to_english(source)
        self.assertEqual(completion.call_count, 2)
        self.assertIn('Taiwan', result)
        self.assertIn('anchor_moved_between_cues', completion.call_args.kwargs['messages'][-1]['content'])

    @patch.object(srt_translator, 'completion')
    def test_final_check_catches_a_move_across_batches(self, completion):
        completion.side_effect = [model_response([{'cue_id': 1, 'text': 'We are here'}]),
                                  model_response([{'cue_id': 2, 'text': 'Nice view in Taiwan'}])]
        with patch.object(srt_translator, 'TRANSLATION_BATCH_SIZE', 1), self.assertRaisesRegex(srt_translator.SRTTranslationError, 'anchor_moved_between_cues'):
            srt_translator.translate_srt_to_english(srt([(1, '我们到了台湾'), (2, '风景真美')]))
        self.assertEqual(completion.call_count, 2)

    @patch.object(srt_corrector, 'completion')
    def test_correction_retries_action_reversal_and_returns_provenance(self, completion):
        source = srt([(1, '我买了衣服')])
        completion.side_effect = [response(['我卖了衣服'], (1,)), response(['我买了衣服'], (1,))]
        result = srt_corrector.request_srt_correction(source)
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(result['corrected_content'], source)
        self.assertEqual(result['validation']['meaning_alignment']['coverage'][0]['source_cue_ids'], [1])


if __name__ == '__main__':
    unittest.main()
