import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from funclip.test.test_gradio_srt_correction import gradio_app
from funclip.test.test_numeric_spans import FIXTURES
from funclip.service_contract import recognition_evidence, recognition_correction_source
from funclip.videoclipper import VideoClipper


class GradioNumericSpanTests(unittest.TestCase):
    def transcribe_fixture(self):
        evidence = json.loads((FIXTURES / 'fire-110-136.json').read_text())
        native = dict(evidence, text=''.join(s['text'] for s in evidence['sentence_info']))
        before = copy.deepcopy(native)
        model = SimpleNamespace(generate=lambda *a, **kw: [native])
        clipper = VideoClipper(model)
        clipper.lang = 'zh'
        text, rendered, state = clipper.recog((16000, np.zeros(160, dtype=np.float32)))
        self.assertEqual(native, before)
        snapshot = recognition_evidence(text, rendered, state)
        self.assertEqual(snapshot['raw_text'], evidence['raw_text'])
        self.assertEqual(snapshot['sentence_info'], evidence['sentence_info'])
        self.assertIn('百分之八十三', rendered)
        self.assertNotIn('百分之八十三', snapshot['original_srt'])
        self.assertIn('百分之八十三', snapshot['correction_source_srt'])
        self.assertEqual(snapshot['correction_source_srt_sha256'], hashlib.sha256(
            snapshot['correction_source_srt'].encode()).hexdigest())
        self.assertEqual(len(snapshot['numeric_span_report']['adjustments']), 1)
        return text, rendered, state, snapshot

    def test_transcription_api_retains_native_evidence_and_exposes_boundary_report(self):
        text, rendered, state, snapshot = self.transcribe_fixture()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(gradio_app, 'get_media_duration_seconds', return_value=26), \
             patch.object(gradio_app, 'VideoClipper') as clipper, \
             patch.object(gradio_app, 'create_unique_srt_path', side_effect=lambda name, prefix: str(Path(directory) / f'{prefix}{name}.srt')):
            clipper.return_value.video_recog.return_value = (text, rendered, state)
            result = gradio_app.api_transcribe('source.mp4')
            self.assertEqual(result['recognition_evidence'], snapshot)
            self.assertEqual(json.loads(Path(result['recognition_evidence_file_path']).read_text()), snapshot)
            self.assertEqual(result['numeric_span_report'], snapshot['numeric_span_report'])
            self.assertIn('Numeric boundaries restored: 1', result['status'])
            state['numeric_span_report']['adjustments'].clear()
            self.assertEqual(len(snapshot['numeric_span_report']['adjustments']), 1)

    def test_correction_uses_protected_layout_in_direct_ui_and_async_paths(self):
        _, rendered, _, evidence = self.transcribe_fixture()
        protected = evidence['correction_source_srt']
        state = {'original_srt': rendered, 'recognition_evidence': evidence}
        result_tuple = (protected, rendered, '', '', '', '', '', 'done')
        with patch.object(gradio_app, 'run_llm_correction_for_content', return_value=result_tuple) as correct:
            gradio_app.run_llm_correction(rendered, '', 'test', '', '', state)
            self.assertEqual(correct.call_args.kwargs['original_srt'], protected)
            # Exercise the UI callback itself through its registered stage closure.
            stage = next(b.fn for b in gradio_app.demo.fns.values()
                         if getattr(b.fn, '__name__', '') == 'stage_correction_outputs')
            captured = inspect.getclosurevars(stage)
            safe = {**captured.globals, **captured.nonlocals}['safe_correction_wrapper']
            captured = inspect.getclosurevars(safe)
            ui = {**captured.globals, **captured.nonlocals}['run_correction_and_show']
            ui('', 'test', '', '', state, 'Use output from previous step', None)
            self.assertEqual(correct.call_args.kwargs['original_srt'], protected)
            with tempfile.TemporaryDirectory() as directory:
                uploaded = Path(directory) / 'input.srt'
                uploaded.write_text(rendered)
                ui('', 'test', '', '', state, 'Upload SRT file(s)', [str(uploaded)])
                self.assertEqual(correct.call_args.kwargs['original_srt'], rendered)
                self.assertEqual(uploaded.read_text(), rendered)
        with patch.object(gradio_app, 'ASYNC_JOBS', {}), \
             patch.object(gradio_app, 'get_media_duration_seconds', return_value=26), \
             patch.object(gradio_app, 'run_with_media_cleanup', side_effect=lambda operation, **kw: operation()), \
             patch.object(gradio_app, 'api_transcribe', return_value={'srt_content': rendered, 'recognition_evidence': evidence}), \
             patch.object(gradio_app, 'api_srt_correct', return_value={'corrected_srt': rendered}) as correct:
            job = gradio_app.create_async_job('transcribe_and_correct')
            gradio_app.run_async_transcribe_and_correct_job(job, {'file_path': 'source.wav', 'cached_upload_path': 'cached.wav'})
            self.assertEqual(correct.call_args.kwargs['srt_content'], protected)
            self.assertEqual(gradio_app.get_async_job_snapshot(job)['status'], 'completed')

    def test_edited_upload_and_legacy_evidence_keep_appropriate_source(self):
        self.assertEqual(recognition_correction_source('edited', {'rendered_srt': 'display',
                         'correction_source_srt': 'protected'}), 'edited')
        self.assertEqual(recognition_correction_source('display', {'rendered_srt': 'display',
                         'original_srt': 'legacy'}), 'legacy')
        self.assertEqual(recognition_correction_source('upload', {}), 'upload')


if __name__ == '__main__':
    unittest.main()
