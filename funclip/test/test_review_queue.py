import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import soundfile as sf
from funclip.glossary import parse_glossary, recognition_hotwords, validate_glossary_correction, GlossaryError
from funclip.review_queue import (build_review_queue, decide_review_item, write_review_bundle,
                                  context_targets, MAX_CONTEXT_CHECKS, MAX_CONTEXT_AUDIO_MS)
from funclip import review_queue
from funclip.test.test_glossary import native_model
from funclip.test.test_numeric_spans import aligned_sentences

FIXTURES = Path(__file__).resolve().parents[2] / 'eval/review_queue_cases'


def fixture(case):
    return json.loads((FIXTURES / case).read_text())


class ReviewQueueTests(unittest.TestCase):
    def test_eight_pinned_audit_cases_surface_without_mutating_evidence(self):
        profiles = fixture('profiles.json')
        for case in fixture('manifest.json')['cases']:
            with self.subTest(case=case['id']):
                data = (FIXTURES / case['fixture']).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), case['sha256'])
                payload = json.loads(data)
                before = copy.deepcopy(payload)
                q = build_review_queue(payload['evidence'], glossary=profiles[case['episode']],
                    baseline=payload.get('baseline'), context_checks=payload.get('context_checks', []),
                    glossary_evidence=payload.get('glossary_evidence'))
                self.assertEqual(payload, before)
                self.assertTrue(set(case['expected_codes']) <= q['issue_counts'].keys())
                self.assertTrue(all(i['decision'] == 'unresolved' and not i['content_verified'] for i in q['items']))
                if case['id'] == 'fire-319-344':
                    conflict = next(i for i in q['items'] if i['code'] == 'context_fact_disagreement')
                    self.assertIn(('10500', 'USD'), conflict['original_facts']['numbers'])
                    self.assertIn(('10100', 'USD'), conflict['alternate_facts']['numbers'])
                if case['id'] == 'soup-150-172':
                    drift = next(i for i in q['items'] if i['code'] == 'asr_hint_off_target_change')
                    self.assertEqual((drift['before'], drift['after']), ('间', '辣'))

    def test_reference_mode_has_no_decoder_hints_or_parser_access(self):
        with recognition_hotwords(object(), 'penghu-food', mode='reference') as (hints, evidence):
            self.assertEqual(hints, '')
            self.assertEqual(evidence['active_hotwords'], [])
            self.assertFalse(evidence['hotwords_checked'])
            self.assertEqual(evidence['profile']['id'], 'penghu-food')
        with self.assertRaises(GlossaryError):
            with recognition_hotwords(native_model(), 'penghu-food', mode='wrong'):
                pass

    def test_distinct_spoken_alias_is_retained_with_traditional_conversion(self):
        p = parse_glossary(fixture('profiles.json')['durian'])
        validate_glossary_correction([(1, '这是榴莲蜜')], [(1, '這是榴蓮蜜')], p)
        for source, target in [('榴莲蜜', '尖不辣'), ('尖不辣', '榴莲蜜')]:
            with self.subTest(source=source), self.assertRaisesRegex(GlossaryError, 'distinct spoken alias'):
                validate_glossary_correction([(1, source)], [(1, target)], p)

    def test_equivalent_context_facts_and_unchanged_hints_are_not_disagreements(self):
        original = aligned_sentences('价格一百美元。')
        equivalent = aligned_sentences('价格100美元。')
        # Keep tokens inside the same explicitly supplied time coverage.
        original['sentence_info'][0]['end'] = 2000
        q = build_review_queue(original, baseline=copy.deepcopy(original), context_checks=[
            {'start_ms': 0, 'end_ms': 2000, 'sentence_ids': [1], 'evidence': equivalent}])
        self.assertNotIn('context_fact_disagreement', q['issue_counts'])
        self.assertNotIn('asr_hint_off_target_change', q['issue_counts'])
        self.assertNotIn('glossary_name_changed', q['issue_counts'])

    def test_repeated_acknowledgements_are_not_removed_or_queued_as_unclear_words(self):
        evidence = aligned_sentences('谢谢谢谢。', '好。', '谢谢软件。')
        original = copy.deepcopy(evidence)
        q = build_review_queue(evidence)
        turns = [i for i in q['items'] if i['code'] == 'short_turn_review']
        self.assertEqual([i['sentence_ids'] for i in turns], [[3]])
        self.assertEqual(evidence, original)

    def test_context_check_budget_and_skipped_coverage_are_explicit(self):
        evidence = aligned_sentences(*(['价格100美元。'] * 100), gap=20_000)
        targets, skipped = context_targets(evidence, 3_000_000)
        self.assertLessEqual(len(targets), MAX_CONTEXT_CHECKS)
        self.assertLessEqual(sum(t['end_ms'] - t['start_ms'] for t in targets), MAX_CONTEXT_AUDIO_MS)
        self.assertTrue(skipped)
        self.assertEqual(context_targets(aligned_sentences('天气很好。'), 2000), ([], []))

    def test_missing_alignment_and_failed_checks_do_not_approve_anything(self):
        evidence = aligned_sentences('价格100美元。')
        bad = dict(evidence, timestamp=[])
        q = build_review_queue(evidence, baseline=bad, context_checks=[
            {'start_ms': 0, 'end_ms': 2000, 'sentence_ids': [1], 'error': 'failed'}])
        self.assertIn('hint_comparison_unavailable', q['issue_counts'])
        self.assertIn('context_check_unavailable', q['issue_counts'])
        self.assertFalse(q['content_verified'])

    def test_audio_bundle_is_replayable_and_decisions_never_edit_source(self):
        evidence = aligned_sentences('价格100美元。')
        q = build_review_queue(evidence)
        source = copy.deepcopy(q)
        with tempfile.TemporaryDirectory() as directory:
            audio = np.linspace(-0.2, 0.2, 64000, dtype=np.float32)
            bundle, path, archive = write_review_bundle(q, audio, 16000, directory)
            self.assertEqual(q, source)
            item = bundle['items'][0]
            data, sr = sf.read(item['audio_file'])
            self.assertEqual(sr, 16000)
            self.assertEqual(len(data), round((item['audio_end_ms'] - item['audio_start_ms']) * 16))
            self.assertTrue(np.allclose(data, audio[:len(data)], atol=1/32768))
            with zipfile.ZipFile(archive) as z:
                self.assertIn('review.html', z.namelist())
                self.assertIn(Path(item['audio_file']).name, z.namelist())
            decided = decide_review_item(bundle, item['id'], 'keep_original')
            self.assertEqual(bundle['items'][0]['decision'], 'unresolved')
            self.assertEqual(decided['items'][0]['decision'], 'keep_original')
            self.assertFalse(decided['items'][0]['content_verified'])
            with self.assertRaises(ValueError):
                decide_review_item(bundle, 'different-request', 'keep_original')
            with self.assertRaises(ValueError):
                decide_review_item(bundle, item['id'], 'auto_verified')

    def test_expiration_removes_only_owned_review_derivatives(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(review_queue._OWNED_DIRECTORIES, {}, clear=True):
            untouched = Path(directory) / 'source.srt'
            untouched.write_text('original reference')
            owned = review_queue.create_review_directory()
            (Path(owned) / 'review.json').write_text('{}')
            review_queue._OWNED_DIRECTORIES[owned] = 0
            self.assertEqual(review_queue.cleanup_review_artifacts(now=3601), 1)
            self.assertFalse(Path(owned).exists())
            self.assertEqual(untouched.read_text(), 'original reference')


if __name__ == '__main__':
    unittest.main()
