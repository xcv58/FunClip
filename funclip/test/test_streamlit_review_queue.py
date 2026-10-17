import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest
from funclip.test.test_gradio_srt_correction import gradio_app
from funclip.test.test_review_queue import fixture
from funclip.review_queue import build_review_queue
from funclip.utils.subtitle_utils import generate_srt
from funclip import videoclipper


class StreamlitReviewQueueTests(unittest.TestCase):
    def test_controls_audio_review_and_decision_preserve_subtitle_content(self):
        data = fixture('soup-150-172.json')
        queue = build_review_queue(data['evidence'], glossary=fixture('profiles.json')['soup'], baseline=data['baseline'])
        selected = data['evidence']
        original = generate_srt(selected['sentence_info'], split_long_chinese=True, format_output=False)
        rendered = generate_srt(selected['sentence_info'], split_long_chinese=True)
        wave = np.zeros(22 * 16000, dtype=np.float32)
        state = {'recognition_srt': original, 'correction_source_srt': original,
                 'review_queue': queue, 'recognition_sample_rate': 16000, 'audio_input': (16000, wave)}
        upload = io.BytesIO()
        upload.name = 'fixture.wav'
        sf.write(upload, wave, 16000, format='WAV')
        upload.seek(0)
        with patch('streamlit.file_uploader', return_value=upload), \
             patch('funasr.AutoModel', return_value=object()), \
             patch.object(videoclipper, 'VideoClipper') as clipper, \
             patch('librosa.load', return_value=(wave, 16000)):
            clipper.return_value.recog.return_value = (selected['text'], rendered, state)
            app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / 'app.py')).run()
            self.assertEqual(next(s for s in app.selectbox if s.label == 'Glossary use').value, 'reference')
            next(s for s in app.selectbox if s.label == 'Episode glossary').set_value('penghu-food')
            next(s for s in app.selectbox if s.label == 'Glossary use').set_value('hints')
            next(s for s in app.checkbox if s.label == 'Compare risky passages with another audio window').check()
            next(b for b in app.button if b.label == 'Start Processing').click().run()
            self.assertEqual(list(app.exception), [])
            options = clipper.return_value.recog.call_args.kwargs
            self.assertEqual(options['glossary'].id, 'penghu-food')
            self.assertEqual(options['glossary_mode'], 'hints')
            self.assertTrue(options['compare_hints'])
            self.assertTrue(options['review_context'])
            before = app.session_state['res_srt']
            self.assertTrue(Path(app.session_state['review_bundle_path']).exists())
            next(r for r in app.radio if r.label == 'Your decision').set_value('needs_edit')
            next(b for b in app.button if b.label == 'Record decision').click().run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(app.session_state['res_srt'], before)
            self.assertEqual(app.session_state['recognition_srt'], original)
            self.assertTrue(any(i['decision'] == 'needs_edit' for i in app.session_state['review_queue']['items']))
            stored = json.loads(Path(app.session_state['review_queue_path']).read_text())
            self.assertEqual(stored, app.session_state['review_queue'])


if __name__ == '__main__':
    unittest.main()
