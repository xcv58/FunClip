"""Request scoped ASR selection. Optional runtimes never load at app startup."""

from contextlib import contextmanager
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from funclip.asr_timestamps import adapt_qwen_alignment, TimestampAdapterError
from funclip.glossary import GlossaryError, resolve_glossary, validate_glossary_mode


DEFAULT_ASR_BACKEND = 'paraformer'
QWEN_BACKEND = 'qwen3-asr-1.7b'


class ASRBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendSpec:
    id: str
    label: str
    experimental: bool = False


BACKENDS = {
    DEFAULT_ASR_BACKEND: BackendSpec(DEFAULT_ASR_BACKEND, 'Paraformer current default'),
    QWEN_BACKEND: BackendSpec(QWEN_BACKEND, 'Qwen3 ASR 1.7B experimental', True),
}


def backend_choices():
    return [(s.label, s.id) for s in BACKENDS.values()]


def validate_backend(value):
    if not isinstance(value, str) or value not in BACKENDS:
        raise ValueError('Choose a supported ASR backend: ' + ', '.join(BACKENDS))
    return value


def select_backend(value, paraformer):
    validate_backend(value)
    return BACKEND_FACTORIES[value](paraformer)


def qwen_settings():
    names = ('FUNCLIP_QWEN_PYTHON', 'FUNCLIP_QWEN_MODEL', 'FUNCLIP_QWEN_ALIGNER')
    values = [os.environ.get(name, '') for name in names]
    if not all(values):
        raise ASRBackendError('Qwen is optional and has not been configured. Set ' + ', '.join(names) + '. See docs/ASR_BACKENDS.md')
    paths = [Path(v).expanduser().resolve() for v in values]
    if not paths[0].is_file() or not os.access(paths[0], os.X_OK) or not all(p.is_dir() for p in paths[1:]):
        raise ASRBackendError('Qwen runtime or local checkpoints are unavailable. Check the configured paths and mounted drive. See docs/ASR_BACKENDS.md')
    # Keep the venv executable path; resolving a venv symlink loses its packages.
    return [str(Path(values[0]).expanduser().absolute()), *map(str, paths[1:])]


class QwenBackend:
    backend_id = QWEN_BACKEND

    def __init__(self, paraformer, *, runner=None):
        self.paraformer = paraformer
        self.runner = runner or self._run_worker

    @contextmanager
    def recognition_hotwords(self, glossary, legacy_hotwords='', *, mode='hints'):
        validate_glossary_mode(mode)
        profile = resolve_glossary(glossary)
        if legacy_hotwords:
            raise GlossaryError('Qwen accepts confirmed episode glossaries only; legacy hotword paths are unsupported.')
        words = list(dict.fromkeys(w for e in profile.entries for w in e.asr_hotwords)) if profile and mode == 'hints' else []
        evidence = None if profile is None else {
            'profile': profile.metadata(), 'mode': mode, 'active_hotwords': words,
            'unsupported_hotwords': [], 'hotwords_checked': False,
            'policy': 'Confirmed glossary words as Qwen context; no vocabulary or acoustic correctness claim',
        }
        yield ' '.join(words), evidence

    def _vad(self, wave):
        model = self.paraformer
        if getattr(model, 'vad_model', None) is None:
            raise ASRBackendError('Qwen requires the existing FSMN VAD; it is unavailable.')
        # Reuse the exact loaded VAD weights and configuration. Never call the
        # Paraformer decoder and never mutate shared inference kwargs.
        return model.inference(wave, model=model.vad_model,
                               kwargs=copy.deepcopy(model.vad_kwargs), cache={}, is_final=True)

    def generate(self, wave, **options):
        if options.get('return_spk_res') or options.get('en_post_proc'):
            raise ASRBackendError('Experimental Qwen currently supports Chinese subtitles without speaker diarization.')
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim != 1 or not np.isfinite(wave).all() or not len(wave):
            raise ASRBackendError('Qwen needs finite nonempty mono audio at 16 kHz.')
        vad = self._vad(wave)
        if not isinstance(vad, list) or len(vad) != 1 or not isinstance(vad[0], dict) or not isinstance(vad[0].get('value'), list):
            raise ASRBackendError('Existing VAD returned an invalid result.')
        regions, previous = [], 0
        duration = len(wave) / 16
        for pair in vad[0]['value']:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in pair)):
                raise ASRBackendError('Existing VAD returned invalid endpoints.')
            start, end = pair
            if start < previous or end <= start or end > duration + 1:
                raise ASRBackendError('Existing VAD endpoints are out of order or outside audio.')
            regions.append([start, end])
            previous = end
        native = self.runner(wave, regions, options.get('hotword', '')) if regions else []
        if not isinstance(native, list) or len(native) != len(regions):
            raise ASRBackendError('Qwen worker returned the wrong number of audio regions.')
        sentences, reports = [], []
        metadata = {'id': self.backend_id, 'experimental': True, 'vad': 'existing FSMN',
                    'timestamp_unit': 'atomic text span', 'runtime': 'mlx-qwen3-asr 0.4.4 float16'}
        retained_native = {'vad': copy.deepcopy(vad), 'regions': native}
        try:
            for output, (start, end) in zip(native, regions):
                sentence, report = adapt_qwen_alignment(output, end - start, offset_ms=start)
                if sentence:
                    sentences.append(sentence)
                reports.append(report)
        except TimestampAdapterError as exc:
            # Formatting failure still gives the service an inspectable original
            # native result. No partial SRT is accepted after alignment failure.
            exc.recognition_state = {'asr_backend': metadata, 'backend_native': retained_native,
                'timestamp_adapter': {'version': 1, 'error': str(exc)},
                'recognition_text': ' '.join(n.get('text', '') for n in native), 'recognition_srt': ''}
            raise
        text = ' '.join(s['text'] for s in sentences)
        return [{'text': text, 'raw_text': text,
                 'timestamp': [t for s in sentences for t in s['timestamp']], 'sentence_info': sentences,
                 'asr_backend': metadata,
                 'backend_native': retained_native,
                 'timestamp_adapter': {'version': 1, 'regions': reports, 'content_verified': False}}]

    @staticmethod
    def _run_worker(wave, regions, context):
        python, model, aligner = qwen_settings()
        with tempfile.TemporaryDirectory(prefix='funclip_qwen_') as folder:
            root = Path(folder)
            np.save(root / 'audio.npy', wave, allow_pickle=False)
            request = {'version': 1, 'model': model, 'aligner': aligner,
                       'regions_ms': regions, 'context': context}
            (root / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            command = [python, '-I', str(Path(__file__).with_name('qwen_worker.py')), str(root)]
            if sys.platform == 'darwin':
                # A local transcription backend has no reason to access a network.
                command = ['/usr/bin/sandbox-exec', '-p', '(version 1)(allow default)(deny network*)', *command]
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
            # Worker needs model paths, not the web app's provider credentials.
            for key in list(env):
                if any(s in key.upper() for s in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD')) and key != 'HF_HUB_DISABLE_IMPLICIT_TOKEN':
                    del env[key]
            with (root / 'worker.log').open('wb') as log:
                try:
                    result = subprocess.run(command, env=env, stdout=log, stderr=log, timeout=1800)
                except subprocess.TimeoutExpired as exc:
                    raise ASRBackendError('Qwen timed out after 30 minutes; no subtitle output was accepted.') from exc
            output = root / 'result.json'
            if result.returncode or not output.is_file():
                raise ASRBackendError('Qwen worker failed. Verify the optional runtime and checkpoint versions in docs/ASR_BACKENDS.md. No subtitle output was accepted.')
            if output.stat().st_size > 64 * 1024 * 1024:
                raise ASRBackendError('Qwen worker result exceeds the size limit.')
            return json.loads(output.read_text(encoding='utf-8'))


BACKEND_FACTORIES = {DEFAULT_ASR_BACKEND: lambda model: model, QWEN_BACKEND: QwenBackend}
