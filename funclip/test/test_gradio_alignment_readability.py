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
from funclip.test.test_srt_translator import model_response
from funclip.test.test_srt_corrector import response
from funclip.llm import srt_corrector, srt_translator


class AlignmentReadabilityIntegrationTests(unittest.TestCase):
    def test_same_named_uploads_keep_distinct_outputs_and_provenance_files(self):
        with tempfile.TemporaryDirectory() as d:
            first, second = Path(d) / 'a/source.srt', Path(d) / 'b/source.srt'
            for file, cue_id in [(first, 7), (second, 19)]:
                file.parent.mkdir()
                file.write_text(f'{cue_id}\n00:00:01,000 --> 00:00:03,000\n我们到了机场\n')
            with patch.object(gradio_app, 'translate_srt_to_english', side_effect=lambda srt_content, **kw: srt_content.replace('我们到了机场', 'We are here')), \
                 patch.object(gradio_app, 'resolve_llm_config', return_value=(None, None, 'test')):
                _, _, _, state = gradio_app.translate_srt_to_english_fn([str(first), str(second)], '', 'test', '', '')
            self.assertEqual(len(set(state['latest_output_paths'])), 2)
            self.assertEqual(len(set(state['alignment_report_paths'])), 2)
            for path, cue_id in zip(state['alignment_report_paths'], [7, 19]):
                self.assertEqual(json.loads(Path(path).read_text())['coverage'][0]['source_cue_ids'], [cue_id])

    def test_transcription_exports_playable_readability_candidates(self):
        text = '我们今天来到博物馆'
        times = [[i * 30, (i + 1) * 30] for i in range(len(text))]
        native = dict(text=text, raw_text=' '.join(text), timestamp=times,
            sentence_info=[dict(text=text, start=0, end=len(text) * 30, timestamp=times)])
        model = native_model()
        model.generate = lambda *a, **k: [copy.deepcopy(native)]
        with patch.object(gradio_app, 'funasr_model', model), \
             patch.object(gradio_app, 'get_media_duration_seconds', return_value=3), \
             patch('librosa.load', return_value=(np.zeros(3 * 16000), 16000)):
            result = gradio_app.api_transcribe_with_review('synthetic.wav', 'none', 'reference', False)
        candidate = next(i for i in result['review_queue']['items'] if i['code'] == 'chinese_reading_speed')
        self.assertTrue(Path(candidate['audio_file']).is_file())
        self.assertEqual(result['recognition_evidence']['raw_text'], native['raw_text'])
        self.assertEqual(result['recognition_evidence']['sentence_info'], native['sentence_info'])
        self.assertIn('chinese_reading_speed', result['quality_report']['readability']['issue_counts'])
        self.assertEqual(result['quality_report']['readability']['applied_changes'], 0)

    def test_translation_metadata_and_ui_file_are_based_on_source_cue_time(self):
        source = '19\n00:00:03,000 --> 00:00:05,000\n我们到了机场\n'
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'source.srt'
            path.write_text(source)
            with patch.object(srt_translator, 'completion', return_value=model_response([{'cue_id': 19, 'text': 'We are here'}])), \
                 patch.object(gradio_app, 'resolve_llm_config', return_value=(None, None, 'test')):
                output = gradio_app.api_translate_srt_english([str(path)], '', 'test', '', '')
                _, _, _, state = gradio_app.translate_srt_to_english_fn([str(path)], '', 'test', '', '')
            report = output['meaning_alignment_reports'][0]['report']
            self.assertIn('Meaning alignment needs review', output['status'])
            self.assertEqual(report['issues'][0]['code'], 'source_anchor_missing')
            self.assertEqual(report['coverage'][0]['start_ms'], 3000)
            self.assertEqual(report['coverage'][0]['source_cue_ids'], [19])
            file_path = Path(state['alignment_report_paths'][0])
            self.assertEqual(json.loads(file_path.read_text()), report)
            component = gradio_app.alignment_download
            files = component.postprocess([str(file_path)])
            cached = component.move_resource_to_block_cache(files[0].path)
            with TestClient(App.create_app(gradio_app.demo)) as client:
                served = client.get('/gradio_api/file=' + cached)
                self.assertEqual(served.status_code, 200)
                self.assertEqual(served.json(), report)

    def test_correction_api_returns_local_provenance_and_visible_warning(self):
        source = '7\n00:00:01,000 --> 00:00:03,000\n我们到了机场\n'
        with patch.object(srt_corrector, 'completion', return_value=response(['我们到了这里'], (7,))), \
             patch.object(gradio_app, 'resolve_llm_config', return_value=(None, None, 'test')):
            result = gradio_app.api_srt_correct(source, '', 'test', '', '', False)
        self.assertIn('Meaning alignment needs review', result['status'])
        self.assertIn('Meaning alignment review', result['diff_html'])
        self.assertFalse(result['meaning_alignment']['coverage'][0]['semantic_coverage_verified'])
        self.assertEqual(result['meaning_alignment']['coverage'][0]['start_ms'], 1000)


if __name__ == '__main__':
    unittest.main()
