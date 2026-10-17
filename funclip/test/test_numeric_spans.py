import copy
import hashlib
import json
from pathlib import Path
import unittest

from funclip.numeric_spans import protect_numeric_sentences, numeric_text_spans
from funclip.utils.subtitle_alignment import TOKEN
from funclip.utils.subtitle_segmentation import split_long_chinese_sentence
from funclip.utils.subtitle_utils import generate_srt
from funclip.subtitle_meaning import check_meaning_preservation, MeaningPreservationError


FIXTURES = Path(__file__).resolve().parents[2] / 'eval/numeric_span_cases'


def aligned_sentences(*texts, gap=0):
    sentences, tokens, times, at = [], [], [], 0
    for text in texts:
        timing = []
        for token in TOKEN.finditer(text):
            tokens.append(token.group())
            timing.append([at, at + 100])
            at += 100
        sentences.append({'text': text, 'start': timing[0][0], 'end': timing[-1][1],
                          'timestamp': timing, 'spk': 0})
        times.extend(timing)
        at += gap
    return {'sentence_info': sentences, 'raw_text': ' '.join(tokens), 'timestamp': times}


def protect(evidence):
    return protect_numeric_sentences(evidence['sentence_info'], raw_text=evidence['raw_text'],
                                     timestamps=evidence['timestamp'])


class NumericSpanTests(unittest.TestCase):
    def test_audit_boundaries_are_restored_without_changing_raw_evidence(self):
        manifest = json.loads((FIXTURES / 'manifest.json').read_text())
        for case in manifest['cases']:
            with self.subTest(case=case['id']):
                source = (FIXTURES / case['evidence']).read_bytes()
                self.assertEqual(hashlib.sha256(source).hexdigest(), case['evidence_sha256'])
                evidence = json.loads(source)
                original = copy.deepcopy(evidence)
                rendered, report = protect(evidence)
                self.assertEqual(evidence, original)
                self.assertEqual(len(report['adjustments']), 1)
                self.assertEqual(report['adjustments'][0]['kind'], case['expected_kind'])
                self.assertIn(case['expected_text'], generate_srt(rendered, split_long_chinese=True))
                self.assertEqual([p for s in rendered for p in s['timestamp']], evidence['timestamp'])
                self.assertEqual([t.group() for s in rendered for t in TOKEN.finditer(s['text'])],
                                 [t.group() for t in TOKEN.finditer(evidence['raw_text'])])
                if case['id'] == 'fire-319-344':
                    self.assertIn('一万五百美元', ''.join(s['text'] for s in rendered))
                    self.assertNotIn('一万零一百', ''.join(s['text'] for s in rendered))

    def test_ambiguous_counts_pauses_speakers_and_alignment_require_review(self):
        cases = [aligned_sentences('百分之八十。', '三个人。'),
                 aligned_sentences('百分之八十。', '三维空间。'),
                 aligned_sentences('五。', '四。', '三。'),
                 aligned_sentences('百分之八十。', '三。', gap=500),
                 aligned_sentences('两百万。', '美元一共。')]
        cases[-1]['sentence_info'][1]['spk'] = 1
        missing = aligned_sentences('百分之八十。', '三。')
        missing['sentence_info'][1].pop('timestamp')
        cases.append(missing)
        mismatch = aligned_sentences('两百万。', '美元一共。')
        mismatch['raw_text'] = mismatch['raw_text'].replace('两', '三')
        cases.append(mismatch)
        empty = aligned_sentences('两百万。', '美元。')
        empty['raw_text'] = ''
        cases.append(empty)
        for evidence in cases:
            with self.subTest(evidence=evidence):
                rendered, report = protect(evidence)
                self.assertEqual(rendered, evidence['sentence_info'])
                self.assertEqual(report['adjustments'], [])
                self.assertTrue(report['issues'])
                self.assertEqual(report['status'], 'needs_review')

    def test_only_currency_prefix_moves_and_render_is_idempotent(self):
        for left, right in [('两百万。', '美元福州房子。'), ('200。', 'USD today。')]:
            with self.subTest(left=left):
                evidence = aligned_sentences(left, right)
                rendered, report = protect(evidence)
                self.assertEqual(len(report['adjustments']), 1)
                self.assertNotIn('福州', rendered[0]['text'])
                self.assertNotIn('today', rendered[0]['text'])
                self.assertEqual(rendered[1]['start'], rendered[1]['timestamp'][0][0])
                again, report2 = protect_numeric_sentences(rendered, raw_text=evidence['raw_text'],
                                                           timestamps=evidence['timestamp'])
                self.assertEqual(again, rendered)
                self.assertEqual(report2['adjustments'], [])

    def test_long_sentence_never_splits_a_value_from_its_digits_or_unit(self):
        for number in ['百分之八十三', '两百万美元', '负百分之三点五', '三分之一', '2025年', '五六十万美元']:
            for prefix_size in [24, 28, 29, 30]:
                with self.subTest(number=number, prefix_size=prefix_size):
                    text = '说' * prefix_size + number + '结束' * 10
                    sentence = aligned_sentences(text)['sentence_info'][0]
                    before = copy.deepcopy(sentence)
                    parts = split_long_chinese_sentence(sentence)
                    self.assertEqual(sentence, before)
                    self.assertEqual(''.join(s['text'] for s in parts), text)
                    self.assertTrue(any(number in part['text'] for part in parts))
                    spans = numeric_text_spans(text)
                    offset = 0
                    for part in parts[:-1]:
                        offset += len(part['text'])
                        self.assertFalse(any(a < offset < b for a, b in spans))
        # A long numeric phrase keeps its actual timing even beyond the display target.
        text = '百分之八十三'
        sentence = aligned_sentences(text)['sentence_info'][0]
        sentence['timestamp'] = [[i * 2000, (i + 1) * 2000] for i in range(len(text))]
        self.assertEqual(len(split_long_chinese_sentence(sentence)), 1)

    def test_audit_equivalents_pass_and_wrong_values_still_fail(self):
        manifest = json.loads((FIXTURES / 'manifest.json').read_text())
        for case in manifest['equivalent_controls']:
            with self.subTest(source=case['source'], target=case['translation']):
                if case['expected_accept']:
                    check_meaning_preservation([(1, case['source'])], [(1, case['translation'])], mode='translation')
                else:
                    with self.assertRaises(MeaningPreservationError):
                        check_meaning_preservation([(1, case['source'])], [(1, case['translation'])], mode='translation')

    def test_equivalents_do_not_relax_dates_durations_currency_or_endpoints(self):
        accepted = [('五六十万美元', 'five to six hundred thousand US dollars'),
                    ('五六十万美元', '500000 USD to 600000 USD'),
                    ('持续2025年', 'for 2025 years'),
                    ('2025年', 'in 2025'), ('三分之一', 'a third of it'),
                    ('第二天', 'the following day')]
        rejected = [('三分之一', 'a third party'), ('三分之一', 'a quarter'),
                    ('四分之一', 'a third'), ('第二天', 'two days later'),
                    ('第二天', 'the same day'), ('第二天', 'the previous day'),
                    ('第二天', 'the third day'), ('2025年', 'in 2026'),
                    ('2025年', '2025 US dollars'), ('持续2025年', 'in 2025'),
                    ('2025美元', 'in 2025'), ('30年', 'a 30-day mortgage'),
                    ('五六十万美元', '500000 to 600000 Singapore dollars'),
                    ('五六十万美元', '50000 to 60000 US dollars'),
                    ('五六十万美元', '500000 to 700000 US dollars'),
                    ('五六十万美元', '600000 US dollars'),
                    ('10100美元', '10500 US dollars'), ('83%', '80 percent')]
        for source, target in accepted:
            with self.subTest(source=source, target=target):
                check_meaning_preservation([(1, source)], [(1, target)], mode='translation')
        for source, target in rejected:
            with self.subTest(source=source, target=target), self.assertRaises(MeaningPreservationError):
                check_meaning_preservation([(1, source)], [(1, target)], mode='translation')


if __name__ == '__main__':
    unittest.main()
