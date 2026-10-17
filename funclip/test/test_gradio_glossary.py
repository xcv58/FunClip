from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from gradio.helpers import special_args
from funclip.test.test_gradio_srt_correction import gradio_app
from funclip.test.test_glossary import native_model, response, srt
from funclip.glossary import resolve_glossary, glossary_report
from funclip.llm import srt_corrector, srt_translator
from funclip.service_contract import recognition_evidence
from funclip.videoclipper import VideoClipper


class GradioGlossaryTests(unittest.TestCase):
    def test_legacy_endpoint_input_counts_and_additive_profiles(self):
        endpoints = gradio_app.demo.get_api_info()['named_endpoints']
        legacy = {'/transcribe': 1, '/srt_correct': 6, '/srt_translate_traditional': 1,
                  '/srt_translate_traditional_text': 1, '/srt_translate_english': 5,
                  '/youtube_chapters': 8, '/submit_transcribe_and_correct': 6,
                  '/submit_srt_correct': 6, '/async_job_status': 2}
        for name, count in legacy.items():
            self.assertEqual(len(endpoints[name]['parameters']), count, name)
        for name, count in {'/transcribe_with_glossary': 2, '/srt_correct_with_glossary': 7,
                            '/srt_translate_english_with_glossary': 6,
                            '/submit_transcribe_and_correct_with_glossary': 7,
                            '/submit_srt_correct_with_glossary': 7}.items():
            self.assertEqual(len(endpoints[name]['parameters']), count)
            self.assertEqual(endpoints[name]['parameters'][-1]['parameter_name'], 'glossary_id')
        self.assertEqual(gradio_app.glossary_selector.value, 'none')

    def test_ui_workflows_capture_selected_profile_after_request_injection(self):
        names = {'stage_process_media_outputs', 'stage_correction_outputs',
                 'stage_english_translation_outputs', 'stage_translator_correction_outputs'}
        found = set()
        for binding in gradio_app.demo.fns.values():
            name = getattr(binding.fn, '__name__', '')
            if name in names:
                found.add(name)
                position = binding.inputs.index(gradio_app.glossary_selector)
                values = [None] * len(binding.inputs)
                values[position] = 'penghu-food'
                request = gradio_app.gr.Request(session_hash='glossary-test')
                values, *_ = special_args(binding.fn, values, request=request)
                args = inspect.signature(binding.fn).bind(*values).arguments
                self.assertEqual(args['glossary_id'], 'penghu-food')
                self.assertIs(args['request'], request)
        self.assertEqual(found, names)

    def test_transcription_api_passes_profile_and_retains_evidence_verbatim(self):
        source = srt('冰卷')
        profile = resolve_glossary('penghu-food')
        report = glossary_report([(1, '冰卷')], profile)
        state = {'recognition_srt': srt('冰卷。'), 'recog_res_raw': '冰 卷',
                 'sentences': [{'text': '冰卷。', 'start': 0, 'end': 1000}],
                 'glossary': {'profile': profile.metadata(), 'unsupported_hotwords': ['红蟳']},
                 'glossary_report': report}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(gradio_app, 'get_media_duration_seconds', return_value=1), \
             patch.object(gradio_app, 'VideoClipper') as clipper, \
             patch.object(gradio_app, 'create_unique_srt_path', side_effect=lambda name, prefix: str(Path(directory) / f'{prefix}{name}.srt')):
            clipper.return_value.video_recog.return_value = ('冰卷。', source, state)
            result = gradio_app.api_transcribe_with_glossary('source.mp4', 'penghu-food')
            self.assertIs(clipper.return_value.video_recog.call_args.kwargs['glossary'], profile)
            self.assertEqual(result['recognition_evidence']['original_srt'], srt('冰卷。'))
            self.assertEqual(result['recognition_evidence']['raw_text'], '冰 卷')
            self.assertEqual(result['srt_content'], source)
            self.assertEqual(result['glossary_report'], report)
            self.assertEqual(json.loads(Path(result['recognition_evidence_file_path']).read_text()), result['recognition_evidence'])

    def test_correction_api_and_translation_api_forward_profile_and_enforce_wording(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(gradio_app, 'resolve_llm_config', return_value=('key', None, 'test')), \
             patch.object(gradio_app, 'create_unique_srt_path', side_effect=lambda name, prefix: str(Path(directory) / f'{prefix}{name}.srt')), \
             patch.object(srt_corrector, 'completion', return_value=response('corrections', '美人腿')):
            result = gradio_app.api_srt_correct_with_glossary(srt('美人腿'), 'key', 'test', '', '', True, 'penghu-food')
            self.assertEqual(result['corrected_srt'], srt('美人腿'))
            self.assertEqual(result['glossary_report']['profile']['id'], 'penghu-food')
            file = Path(directory) / 'input.srt'
            file.write_text(srt('红𫊻粥'))
            with patch.object(srt_translator, 'completion', return_value=response('translations', 'Roe crab porridge')):
                result = gradio_app.api_translate_srt_english_with_glossary([str(file)], 'key', 'test', '', '', 'penghu-food')
                self.assertEqual(result['preview_translated'], srt('Roe crab porridge'))
                self.assertEqual(result['glossary_reports'][0]['report']['issues'], [])

    def test_invalid_profile_fails_before_media_staging(self):
        with patch.object(gradio_app, 'stage_async_uploaded_file') as stage:
            with self.assertRaises(ValueError):
                gradio_app.api_submit_transcribe_and_correct_with_glossary('source.wav', '', 'test', '', '', True, '../bad')
            stage.assert_not_called()

    def test_async_submission_snapshots_profile_without_changing_legacy_payload(self):
        with patch.object(gradio_app, 'ASYNC_JOBS', {}), \
             patch.object(gradio_app, 'start_async_worker') as worker:
            gradio_app.api_submit_srt_correct_with_glossary(srt('美人腿'), '', 'test', '', '', True, 'penghu-food')
            self.assertEqual(worker.call_args.args[2]['glossary_id'], 'penghu-food')
            gradio_app.api_submit_srt_correct(srt('美人腿'), '', 'test', '', '', True)
            self.assertNotIn('glossary_id', worker.call_args.args[2])

    def test_async_worker_keeps_profile_through_recognition_and_correction(self):
        with patch.object(gradio_app, 'ASYNC_JOBS', {}), \
             patch.object(gradio_app, 'get_media_duration_seconds', return_value=1), \
             patch.object(gradio_app, 'run_with_media_cleanup', side_effect=lambda operation, **kw: operation()), \
             patch.object(gradio_app, 'api_transcribe', return_value={'srt_content': srt('美人腿')}) as transcribe, \
             patch.object(gradio_app, 'api_srt_correct', return_value={'corrected_srt': srt('美人腿')}) as correct:
            job = gradio_app.create_async_job('transcribe_and_correct')
            gradio_app.run_async_transcribe_and_correct_job(job, {'file_path': 'source.wav',
                'cached_upload_path': 'cached.wav', 'glossary_id': 'penghu-food'})
            self.assertEqual(transcribe.call_args.kwargs['glossary_id'], 'penghu-food')
            self.assertEqual(correct.call_args.kwargs['glossary_id'], 'penghu-food')
            self.assertEqual(gradio_app.get_async_job_snapshot(job)['status'], 'completed')

    def test_parallel_recognition_does_not_exchange_profile_or_mutate_raw_evidence(self):
        model = native_model()
        active = []
        observations = []
        def generate(data, **kwargs):
            # A shared native decoder would otherwise exchange hotword state.
            self.assertEqual(active, [])
            active.append(kwargs['hotword'])
            words = Path(kwargs['hotword']).read_text()
            time.sleep(0.01)
            self.assertEqual(active, [kwargs['hotword']])
            observations.append(words)
            active.clear()
            text = '冰卷。' if '冰卷' in words else '花穹。'
            return [{'text': text, 'raw_text': text, 'timestamp': [[0, 500], [500, 1000]],
                     'sentence_info': [{'text': text, 'start': 0, 'end': 1000,
                                        'timestamp': [[0, 500], [500, 1000]]}]}]
        model.generate = generate
        def run(profile):
            clipper = VideoClipper(model)
            clipper.lang = 'zh'
            text, rendered, state = clipper.recog((16000, np.zeros(16000, dtype=np.float32)), glossary=profile)
            return recognition_evidence(text, rendered, state)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run, ['penghu-food', 'singapore-gardens']))
        self.assertEqual(len(observations), 2)
        self.assertTrue(any('冰卷' in words and '花穹' not in words for words in observations))
        self.assertTrue(any('花穹' in words and '冰卷' not in words for words in observations))
        self.assertIn('冰卷。', results[0]['original_srt'])
        self.assertNotIn('。', results[0]['rendered_srt'])
        self.assertFalse(Path(model.model.files[0]).exists())


if __name__ == '__main__':
    unittest.main()
