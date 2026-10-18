import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from funclip.asr_backends import (ASRBackendError, DEFAULT_ASR_BACKEND, QWEN_BACKEND,
                                  QwenBackend, backend_choices, qwen_settings, select_backend)
from funclip.asr_timestamps import TimestampAdapterError, adapt_qwen_alignment
from funclip.service_contract import recognition_evidence
from funclip.utils.subtitle_alignment import aligned_units
from funclip.utils.subtitle_utils import generate_srt
from funclip.videoclipper import VideoClipper


def native(text='澎湖海鲜。', pairs=((0, 0), (0, .4), (.4, .8), (.8, 1))):
    return {'text': text, 'segments': [dict(text=t, start=a, end=b)
            for t, (a, b) in zip(text.rstrip('。'), pairs)],
            'truncated': False, 'chunks': [{'truncated': False}]}


class TimestampAdapterTests(unittest.TestCase):
    def test_zero_duration_is_grouped_without_estimated_character_times(self):
        source = native()
        before = copy.deepcopy(source)
        sentence, report = adapt_qwen_alignment(source, 1000, offset_ms=2000)
        self.assertEqual(sentence['text'], source['text'])
        self.assertEqual(sentence['timestamp'], [[2000, 2400], [2400, 2800], [2800, 3000]])
        self.assertEqual(sentence['timestamp_spans'], [[0, 2], [2, 3], [3, 4]])
        self.assertEqual([t.group() for t in aligned_units(sentence['text'], sentence['timestamp'], sentence['timestamp_spans'])], ['澎湖', '海', '鲜'])
        self.assertEqual(report['merged_segments'], 1)
        self.assertFalse(report['content_verified'])
        self.assertEqual(source, before)

    def test_middle_and_trailing_points_stay_in_atomic_spans(self):
        sentence, _ = adapt_qwen_alignment(native(pairs=((0, .4), (.4, .4), (.6, 1), (1, 1))), 1100)
        self.assertEqual(sentence['timestamp_spans'], [[0, 2], [2, 4]])
        self.assertEqual(sentence['timestamp'], [[0, 400], [600, 1000]])

    def test_leading_distinct_points_wait_for_a_real_interval(self):
        sentence, _ = adapt_qwen_alignment(native(pairs=((0, 0), (.2, .2), (.4, .8), (.8, 1))), 1000)
        self.assertEqual(sentence['timestamp'], [[0, 800], [800, 1000]])
        self.assertEqual(sentence['timestamp_spans'], [[0, 3], [3, 4]])

    def test_overlaps_and_english_subwords_cannot_be_cut(self):
        value = {'text': 'Hello world', 'segments': [dict(text='Hel', start=0, end=.3),
                 dict(text='lo', start=.2, end=.5), dict(text='world', start=.4, end=1)]}
        sentence, report = adapt_qwen_alignment(value, 1000)
        self.assertEqual(sentence['timestamp_spans'], [[0, 11]])
        self.assertEqual(sentence['timestamp'], [[0, 1000]])

    def test_only_small_final_lattice_overhang_is_trimmed(self):
        value = native(pairs=((0, .2), (.2, .4), (.4, .6), (.6, 1.04)))
        sentence, report = adapt_qwen_alignment(value, 1000)
        self.assertEqual(sentence['timestamp'][-1][1], 1000)
        self.assertEqual(report['audio_boundary_trims'], 1)
        value['segments'][-1]['end'] = 1.081
        with self.assertRaises(TimestampAdapterError):
            adapt_qwen_alignment(value, 1000)

    def test_bad_alignment_is_rejected(self):
        bad = []
        value = native(); value['segments'].pop(); bad.append(value)
        value = native(); value['segments'][0]['text'] = '台'; bad.append(value)
        value = native(); value['truncated'] = True; bad.append(value)
        value = native(); value['chunks'][0]['truncated'] = True; bad.append(value)
        for a, b in [(-.1, 0), (.2, .1), (math.nan, 0), (0, math.inf), (False, .2), (1, 1)]:
            value = native(); value['segments'][0].update(start=a, end=b); bad.append(value)
        value = native(); value['segments'][3]['start'] = .1; bad.append(value)
        value = native(); value['segments'][2]['end'] = .1; bad.append(value)
        # Point observations alone do not prove a duration even at distinct times.
        value = native(pairs=((0, 0), (.2, .2), (.4, .4), (.8, .8))); bad.append(value)
        for value in bad:
            with self.subTest(value=value), self.assertRaises(TimestampAdapterError):
                adapt_qwen_alignment(value, 1000)

    def test_empty_is_distinct_from_missing_alignment(self):
        self.assertIsNone(adapt_qwen_alignment({'text': '', 'segments': []}, 1000)[0])
        for value in [{'text': '嗯'}, {'text': '。'}, {'text': '', 'segments': [{'text': '嗯'}]}]:
            with self.assertRaises(TimestampAdapterError):
                adapt_qwen_alignment(value, 1000)

    def test_segmentation_retains_atomic_boundaries_and_all_literal_text(self):
        text = '测试很长字幕' * 10
        segments = [dict(text=t, start=i * .2, end=(i + 1) * .2) for i, t in enumerate(text)]
        segments[0]['end'] = 0
        value = {'text': text, 'segments': segments}
        sentence, _ = adapt_qwen_alignment(value, len(text) * 200)
        from funclip.utils.subtitle_segmentation import split_long_chinese_sentence
        chunks = split_long_chinese_sentence(sentence)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(''.join(c['text'] for c in chunks), text)
        endpoints = {t for pair in sentence['timestamp'] for t in pair}
        for chunk in chunks:
            self.assertTrue(aligned_units(chunk['text'], chunk['timestamp'], chunk['timestamp_spans']))
            self.assertIn(chunk['start'], endpoints)
            self.assertIn(chunk['end'], endpoints)


class BackendTests(unittest.TestCase):
    def vad_model(self, regions=None):
        model = SimpleNamespace(vad_model=object(), vad_kwargs={'device': 'mps', 'cache': {}})
        def infer(wave, **options):
            options['kwargs']['cache']['modified'] = True
            return [{'value': regions if regions is not None else [[100, 1100]]}]
        model.inference = infer
        return model

    def test_selection_is_request_scoped_and_default_is_identity(self):
        default = object()
        self.assertIs(select_backend(DEFAULT_ASR_BACKEND, default), default)
        a = select_backend(QWEN_BACKEND, default)
        b = select_backend(QWEN_BACKEND, default)
        self.assertIsNot(a, b)
        self.assertIs(a.paraformer, default)
        self.assertEqual(backend_choices()[0][1], DEFAULT_ASR_BACKEND)
        for value in ['unknown', '/tmp/model', None, ['paraformer']]:
            with self.assertRaises(ValueError):
                select_backend(value, default)

    def test_existing_vad_gates_before_worker_and_keeps_original_config(self):
        model = self.vad_model([])
        backend = QwenBackend(model, runner=lambda *args: self.fail('No speech must not load Qwen'))
        result = backend.generate(np.zeros(32000))[0]
        self.assertEqual(result['sentence_info'], [])
        self.assertEqual(result['backend_native']['vad'], [{'value': []}])
        self.assertEqual(model.vad_kwargs['cache'], {})

    def test_offsets_and_native_evidence_survive_the_whole_subtitle_pipeline(self):
        source = native()
        backend = QwenBackend(self.vad_model(), runner=lambda *args: [copy.deepcopy(source)])
        clipper = VideoClipper(backend); clipper.lang = 'zh'
        text, srt, state = clipper.recog((16000, np.zeros(32000)))
        self.assertIn('澎湖海鲜', srt)
        self.assertIn('00:00:00,100 --> 00:00:01,100', srt)
        self.assertEqual(state['backend_native']['regions'], [source])
        self.assertEqual(state['asr_backend']['id'], QWEN_BACKEND)
        self.assertIn('atomic_timing_span', state['review_queue']['issue_counts'])
        evidence = recognition_evidence(text, srt, state)
        self.assertEqual(evidence['backend_native']['regions'], [source])
        evidence['backend_native']['regions'][0]['text'] = 'changed'
        self.assertEqual(state['backend_native']['regions'][0], source)
        with self.assertRaisesRegex(ValueError, 'subtitle generation only'):
            clipper.clip('澎湖', 0, 0, state)
        with self.assertRaisesRegex(ValueError, 'subtitle generation only'):
            clipper.video_clip('澎湖', 0, 0, state)

    def test_alignment_failure_carries_native_evidence_without_a_partial_srt(self):
        value = native(); value['truncated'] = True
        backend = QwenBackend(self.vad_model(), runner=lambda *args: [value])
        with self.assertRaises(TimestampAdapterError) as caught:
            backend.generate(np.zeros(32000))
        self.assertEqual(caught.exception.recognition_state['backend_native']['regions'], [value])
        self.assertEqual(caught.exception.recognition_state['recognition_srt'], '')

    def test_qwen_glossary_context_and_unbiased_comparison(self):
        calls = []
        def worker(wave, regions, context):
            calls.append(context)
            return [native()]
        clipper = VideoClipper(QwenBackend(self.vad_model(), runner=worker)); clipper.lang = 'zh'
        _, _, state = clipper.recog((16000, np.zeros(32000)), glossary='penghu-food', glossary_mode='hints', compare_hints=True)
        self.assertTrue(calls[0])
        self.assertEqual(calls[1], '')
        self.assertFalse(state['glossary']['hotwords_checked'])
        self.assertEqual(state['review_queue']['hint_comparison'], 'completed')
        self.assertNotIn('hint_comparison_unavailable', state['review_queue']['issue_counts'])
        self.assertIn('backend_native', state['review_evidence']['baseline'])
        calls.clear()
        clipper.recog((16000, np.zeros(32000)), glossary='penghu-food', glossary_mode='reference', compare_hints=True)
        self.assertEqual(calls, [''])

    def test_unsafe_vad_and_unsupported_modes_never_invoke_worker(self):
        for regions in [[[0, 3000]], [[100, 100]], [[-1, 10]], [[100, 200], [150, 300]], [[True, 100]]]:
            backend = QwenBackend(self.vad_model(regions), runner=lambda *args: self.fail('invalid VAD'))
            with self.subTest(regions=regions), self.assertRaises(ASRBackendError):
                backend.generate(np.zeros(32000))
        backend = QwenBackend(self.vad_model(), runner=lambda *args: self.fail('unsupported mode'))
        for kwargs in [dict(return_spk_res=True), dict(en_post_proc=True)]:
            with self.assertRaises(ASRBackendError):
                backend.generate(np.zeros(32000), **kwargs)

    def test_optional_configuration_is_explicit_and_keeps_venv_executable(self):
        with patch.dict('os.environ', {}, clear=True), self.assertRaisesRegex(ASRBackendError, 'not been configured'):
            qwen_settings()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'python').symlink_to('/usr/bin/true')
            (root / 'qwen').mkdir(); (root / 'aligner').mkdir()
            settings = {'FUNCLIP_QWEN_PYTHON': str(root / 'python'), 'FUNCLIP_QWEN_MODEL': str(root / 'qwen'),
                        'FUNCLIP_QWEN_ALIGNER': str(root / 'aligner')}
            with patch.dict('os.environ', settings):
                self.assertEqual(qwen_settings()[0], str(root / 'python'))


if __name__ == '__main__':
    unittest.main()
