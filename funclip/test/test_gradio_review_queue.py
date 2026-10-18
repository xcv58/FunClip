import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from gradio.routes import App
from funclip.test.test_gradio_srt_correction import gradio_app
from funclip.test.test_glossary import native_model
from funclip.test.test_review_queue import fixture
from funclip.glossary import resolve_glossary
from funclip.videoclipper import VideoClipper


class GradioReviewQueueTests(unittest.TestCase):
    def make_model(self):
        item = fixture('soup-150-172.json')
        profile = fixture('profiles.json')['soup']
        model = native_model()
        calls = []
        def generate(wave, **kwargs):
            calls.append((len(wave), kwargs['hotword']))
            return [copy.deepcopy(item['evidence'] if kwargs['hotword'] else item['baseline'])]
        model.generate = generate
        return model, profile, item, calls

    def test_reference_one_pass_and_hints_comparison_keep_the_selected_recognition(self):
        model, profile, item, calls = self.make_model()
        clipper = VideoClipper(model)
        clipper.lang = 'zh'
        wave = np.zeros(22 * 16000, dtype=np.float32)
        text, rendered, state = clipper.recog((16000, wave), glossary=profile, glossary_mode='reference', compare_hints=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], '')
        self.assertEqual(model.model.files, [])
        self.assertEqual(state['sentences'], item['baseline']['sentence_info'])
        self.assertEqual(state['glossary']['mode'], 'reference')
        self.assertEqual(state['review_queue']['hint_comparison'], 'not_requested')
        calls.clear()
        text, rendered, state = clipper.recog((16000, wave), glossary=profile, glossary_mode='hints', compare_hints=True)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][1])
        self.assertEqual(calls[1][1], '')
        self.assertEqual(state['sentences'], item['evidence']['sentence_info'])
        self.assertEqual(state['review_evidence']['baseline']['sentence_info'], item['baseline']['sentence_info'])
        self.assertIn('时辣太久了', rendered)
        self.assertIn('asr_hint_off_target_change', state['review_queue']['issue_counts'])
        self.assertEqual(state['review_queue']['hint_comparison'], 'completed')
        self.assertFalse(Path(calls[0][1]).exists())

    def test_optional_baseline_failure_preserves_selected_result_and_reports_failure(self):
        model, profile, item, calls = self.make_model()
        def generate(wave, **kw):
            if not kw['hotword']:
                raise RuntimeError('optional decoder failed')
            return [copy.deepcopy(item['evidence'])]
        model.generate = generate
        clipper = VideoClipper(model)
        clipper.lang = 'zh'
        text, rendered, state = clipper.recog((16000, np.zeros(22 * 16000)), glossary=profile, compare_hints=True)
        self.assertEqual(state['sentences'], item['evidence']['sentence_info'])
        self.assertEqual(state['review_queue']['hint_comparison'], 'failed')
        self.assertIn('hint_comparison_unavailable', state['review_queue']['issue_counts'])

    def test_context_checks_use_same_model_and_do_not_replace_primary(self):
        item = fixture('fire-319-344.json')
        calls = []
        output_dirs = []
        model = native_model()
        def generate(wave, **kw):
            calls.append((len(wave), kw['hotword']))
            output_dirs.append(kw['output_dir'])
            return [copy.deepcopy(item['evidence'])]
        model.generate = generate
        clipper = VideoClipper(model)
        clipper.lang = 'zh'
        text, rendered, state = clipper.recog((16000, np.zeros(25 * 16000)), review_context=True, output_dir='primary-evidence')
        self.assertGreater(len(calls), 1)
        self.assertLessEqual(len(calls), 7)
        self.assertLessEqual(sum(n for n, _ in calls[1:]), 120 * 16000)
        self.assertEqual(state['sentences'], item['evidence']['sentence_info'])
        self.assertIn('一万五百美元', rendered)
        self.assertEqual(state['review_queue']['context_checks'], len(calls) - 1)
        self.assertEqual(output_dirs[0], 'primary-evidence')
        self.assertTrue(all(path is None for path in output_dirs[1:]))

    def test_transcription_review_artifacts_and_decision_keep_evidence_and_srt(self):
        model, profile, item, calls = self.make_model()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(gradio_app, 'funasr_model', model), \
             patch.object(gradio_app, 'get_media_duration_seconds', return_value=22), \
             patch('librosa.load', return_value=(np.zeros(22 * 16000), 16000)), \
             patch.object(gradio_app, 'create_unique_srt_path', return_value=str(Path(directory) / 'generated.srt')):
            # Named production profile selects the same control path; fake native
            # inference provides a pinned auditable contrast, not an ASR score.
            result = gradio_app.api_transcribe_with_review('source.wav', 'penghu-food', 'hints', False)
            queue = result['review_queue']
            self.assertTrue(Path(result['review_bundle_file_path']).exists())
            self.assertEqual(json.loads(Path(result['review_queue_file_path']).read_text()), queue)
            state = {'original_srt': result['srt_content'], 'recognition_evidence': result['recognition_evidence'],
                     'review_queue': queue, 'review_queue_file_path': result['review_queue_file_path'],
                     'review_bundle_file_path': result['review_bundle_file_path']}
            before = copy.deepcopy(state)
            chosen = next(i for i in queue['items'] if i.get('audio_file'))
            file, detail = gradio_app.preview_review_item(chosen['id'], state)
            self.assertTrue(Path(file).exists())
            self.assertIn('<pre ', detail)
            self.assertTrue(all(f['meta']['_type'] == 'gradio.FileData' for f in result['review_files']))
            # Regression for JSON responses returning unservable server paths.
            # Exercise Gradio's real file route without a listening socket.
            app = App.create_app(gradio_app.demo)
            with TestClient(app) as client:
                for file in result['review_files']:
                    response = client.get('/gradio_api/file=' + file['path'])
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.content, Path(file['path']).read_bytes())
            changed = gradio_app.record_review_decision(chosen['id'], 'needs_edit', state)
            self.assertEqual(state, before)
            self.assertEqual(changed['original_srt'], before['original_srt'])
            self.assertEqual(changed['recognition_evidence'], before['recognition_evidence'])
            self.assertEqual(next(i for i in changed['review_queue']['items'] if i['id'] == chosen['id'])['decision'], 'needs_edit')
            self.assertEqual(gradio_app.preview_review_item('another-request', state), (None, ''))
            with self.assertRaises(Exception):
                gradio_app.record_review_decision('another-request', 'keep_original', state)

    def test_new_endpoint_contracts_and_ui_capture_mode_and_context(self):
        endpoints = gradio_app.demo.get_api_info()['named_endpoints']
        self.assertEqual(len(endpoints['/transcribe']['parameters']), 1)
        self.assertEqual(len(endpoints['/transcribe_with_glossary']['parameters']), 2)
        self.assertEqual(len(endpoints['/transcribe_with_review']['parameters']), 4)
        self.assertEqual(len(endpoints['/submit_transcribe_and_correct_with_review']['parameters']), 9)
        stage = next(b for b in gradio_app.demo.fns.values() if getattr(b.fn, '__name__', '') == 'stage_process_media_outputs')
        self.assertIs(stage.inputs[-3], gradio_app.glossary_mode_selector)
        self.assertIs(stage.inputs[-2], gradio_app.review_context_selector)
        self.assertIs(stage.inputs[-1], gradio_app.asr_backend_selector)
        self.assertEqual(gradio_app.glossary_mode_selector.value, 'reference')
        self.assertFalse(gradio_app.review_context_selector.value)

    def test_async_snapshots_mode_context_and_reuses_glossary_for_correction(self):
        with patch.object(gradio_app, 'ASYNC_JOBS', {}), \
             patch.object(gradio_app, 'stage_async_uploaded_file', return_value=('stage.wav', 'cached.wav')), \
             patch.object(gradio_app, 'start_async_worker') as worker:
            gradio_app.api_submit_transcribe_and_correct_with_review('source.wav', '', 'test', '', '', False,
                                                                   'penghu-food', 'reference', True)
            payload = worker.call_args.args[2]
            self.assertEqual(payload['glossary_mode'], 'reference')
            self.assertTrue(payload['review_context'])
        with patch.object(gradio_app, 'ASYNC_JOBS', {}), \
             patch.object(gradio_app, 'get_media_duration_seconds', return_value=1), \
             patch.object(gradio_app, 'run_with_media_cleanup', side_effect=lambda operation, **kw: operation()), \
             patch.object(gradio_app, 'api_transcribe', return_value={'srt_content': 'source', 'review_queue': {'items': []}}) as transcribe, \
             patch.object(gradio_app, 'api_srt_correct', return_value={'corrected_srt': 'corrected'}) as correct:
            job = gradio_app.create_async_job('transcribe_and_correct')
            gradio_app.run_async_transcribe_and_correct_job(job, payload)
            self.assertEqual(transcribe.call_args.kwargs['glossary_mode'], 'reference')
            self.assertTrue(transcribe.call_args.kwargs['review_context'])
            self.assertEqual(correct.call_args.kwargs['glossary_id'], 'penghu-food')
            self.assertIn('review_queue', gradio_app.get_async_job_snapshot(job, include_result=True)['result']['transcribe'])

    def test_invalid_review_controls_fail_before_staging(self):
        with patch.object(gradio_app, 'stage_async_uploaded_file') as stage:
            for mode, retry in [('wrong', False), ('reference', 'yes')]:
                with self.subTest(mode=mode, retry=retry), self.assertRaises((ValueError, gradio_app.gr.Error)):
                    gradio_app.api_submit_transcribe_and_correct_with_review('source.wav', '', 'test', '', '', False,
                                                                           'penghu-food', mode, retry)
            stage.assert_not_called()


if __name__ == '__main__':
    unittest.main()
