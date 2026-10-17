import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest
from funclip.numeric_spans import protect_numeric_sentences
from funclip.test.test_numeric_spans import FIXTURES
from funclip.utils.subtitle_utils import generate_srt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from funclip import videoclipper


class StreamlitNumericSpanTests(unittest.TestCase):
    def test_processing_and_correction_keep_the_protected_layout(self):
        native = json.loads((FIXTURES / 'fire-110-136.json').read_text())
        protected, report = protect_numeric_sentences(native['sentence_info'], raw_text=native['raw_text'],
                                                     timestamps=native['timestamp'])
        original = generate_srt(native['sentence_info'], split_long_chinese=True, format_output=False)
        correction = generate_srt(protected, split_long_chinese=True, format_output=False)
        rendered = generate_srt(protected, split_long_chinese=True)
        state = {'recognition_srt': original, 'correction_source_srt': correction, 'numeric_span_report': report}
        upload = io.BytesIO()
        upload.name = 'fixture.wav'
        sf.write(upload, np.zeros(16000), 16000, format='WAV')
        upload.seek(0)
        with patch('streamlit.file_uploader', return_value=upload), \
             patch('funasr.AutoModel', return_value=object()), \
             patch.object(videoclipper, 'VideoClipper') as clipper, \
             patch('librosa.load', return_value=(np.zeros(16000), 16000)), \
             patch('funclip.llm.srt_corrector.correct_srt_content', return_value=rendered) as correct:
            clipper.return_value.recog.return_value = ('native text', rendered, state)
            app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / 'app.py')).run()
            next(b for b in app.button if b.label == 'Start Processing').click().run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(app.session_state['recognition_srt'], original)
            self.assertEqual(app.session_state['correction_source_srt'], correction)
            self.assertTrue(any('Numeric boundaries restored: 1' in c.value for c in app.caption))
            next(b for b in app.button if b.label == '✨ Run Auto Correction').click().run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(correct.call_args.kwargs['srt_content'], correction)
            self.assertEqual(app.session_state['res_srt_corrected'], rendered)


if __name__ == '__main__':
    unittest.main()
