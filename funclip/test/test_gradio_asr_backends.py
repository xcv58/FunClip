import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from funclip.test.test_gradio_srt_correction import gradio_app
from funclip.asr_backends import DEFAULT_ASR_BACKEND, QWEN_BACKEND, QwenBackend
from funclip.service_contract import recognition_evidence


class GradioBackendTests(unittest.TestCase):
    def test_additive_endpoints_and_ui_keep_legacy_defaults(self):
        endpoints = gradio_app.demo.get_api_info()['named_endpoints']
        for name, count in [('transcribe', 1), ('transcribe_with_review', 4),
                            ('submit_transcribe_and_correct', 6),
                            ('submit_transcribe_and_correct_with_review', 9),
                            ('transcribe_with_backend', 5),
                            ('submit_transcribe_and_correct_with_backend', 10)]:
            self.assertEqual(len(endpoints['/' + name]['parameters']), count)
        self.assertEqual(gradio_app.asr_backend_selector.value, DEFAULT_ASR_BACKEND)
        stage = next(b for b in gradio_app.demo.fns.values() if getattr(b.fn, '__name__', '') == 'stage_process_media_outputs')
        self.assertIs(stage.inputs[-1], gradio_app.asr_backend_selector)

    def test_sync_forwards_request_selection_and_legacy_omits_it(self):
        output = ('markdown', 'srt', 'path', 'status', {'asr_backend': {'id': QWEN_BACKEND}})
        with patch.object(gradio_app, 'process_media', return_value=output) as process:
            result = gradio_app.api_transcribe_with_backend('source.wav', QWEN_BACKEND, 'none', 'reference', False)
            self.assertEqual(process.call_args.kwargs['asr_backend'], QWEN_BACKEND)
            self.assertEqual(result['asr_backend']['id'], QWEN_BACKEND)
            gradio_app.api_transcribe('source.wav')
            self.assertNotIn('asr_backend', process.call_args.kwargs)

    def test_async_snapshots_and_worker_forwards_backend(self):
        with patch.object(gradio_app, 'ASYNC_JOBS', {}), \
             patch.object(gradio_app, 'stage_async_uploaded_file', return_value=('stage.wav', 'cached.wav')), \
             patch.object(gradio_app, 'start_async_worker') as worker:
            gradio_app.api_submit_transcribe_and_correct_with_backend('source.wav', '', 'test', '', '', False,
                                                                    QWEN_BACKEND, 'none', 'reference', False)
            payload = worker.call_args.args[2]
            self.assertEqual(payload['asr_backend'], QWEN_BACKEND)
        with patch.object(gradio_app, 'ASYNC_JOBS', {}), \
             patch.object(gradio_app, 'get_media_duration_seconds', return_value=1), \
             patch.object(gradio_app, 'run_with_media_cleanup', side_effect=lambda operation, **kw: operation()), \
             patch.object(gradio_app, 'api_transcribe', return_value={'srt_content': ''}) as transcribe:
            job = gradio_app.create_async_job('transcribe_and_correct')
            gradio_app.run_async_transcribe_and_correct_job(job, payload)
            self.assertEqual(transcribe.call_args.kwargs['asr_backend'], QWEN_BACKEND)
            self.assertEqual(gradio_app.get_async_job_snapshot(job, include_result=True)['result']['outcome'], 'no-speech')

    def test_invalid_backend_rejected_before_staging(self):
        with patch.object(gradio_app, 'stage_async_uploaded_file') as stage:
            with self.assertRaises(ValueError):
                gradio_app.api_submit_transcribe_and_correct_with_backend('source.wav', '', 'test', '', '', False,
                                                                        'arbitrary/path', 'none', 'reference', False)
            stage.assert_not_called()

    def test_timing_failure_saves_original_native_evidence_before_returning_error(self):
        native = {'text': '嗯。', 'segments': [{'text': '嗯', 'start': 0, 'end': 0}]}
        model = SimpleNamespace(vad_model=object(), vad_kwargs={},
                                inference=lambda *args, **kwargs: [{'value': [[0, 1000]]}])
        backend = QwenBackend(model, runner=lambda *args: [native])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'evidence.json'
            def save(text, srt, state, name):
                evidence = recognition_evidence(text, srt, state)
                target.write_text(json.dumps(evidence))
                return evidence, str(target)
            with patch.object(gradio_app, 'select_backend', return_value=backend), \
                 patch.object(gradio_app, 'get_media_duration_seconds', return_value=1), \
                 patch('librosa.load', return_value=(np.zeros(16000), 16000)), \
                 patch.object(gradio_app, 'save_recognition_evidence', side_effect=save), \
                 patch.object(gradio_app, 'create_unique_srt_path') as output:
                with self.assertRaisesRegex(gradio_app.gr.Error, 'Original ASR evidence retained'):
                    gradio_app.api_transcribe_with_backend('source.wav', QWEN_BACKEND, 'none', 'reference', False)
                self.assertEqual(json.loads(target.read_text())['backend_native']['regions'], [native])
                self.assertEqual(json.loads(target.read_text())['rendered_srt'], '')
                output.assert_not_called()


if __name__ == '__main__':
    unittest.main()
