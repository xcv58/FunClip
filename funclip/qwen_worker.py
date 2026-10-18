"""Isolated optional MLX runtime; launched by asr_backends, never imported by UI."""

import dataclasses
import json
import os
from pathlib import Path
import sys
from importlib.metadata import version


def main(folder):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    if version('mlx-qwen3-asr') != '0.4.4' or version('mlx') != '0.32.4':
        raise RuntimeError('Use the validated optional runtime versions')
    import mlx.core as mx
    from mlx_qwen3_asr import load_model, transcribe, ForcedAligner

    root = Path(folder)
    request = json.loads((root / 'request.json').read_text(encoding='utf-8'))
    if request['version'] != 1:
        raise ValueError('Unsupported worker protocol')
    config = json.loads((Path(request['aligner']) / 'config.json').read_text())
    if config.get('timestamp_segment_time') != 80:
        raise ValueError('Unsupported forced aligner timestamp resolution')
    wave = np.load(root / 'audio.npy', allow_pickle=False)
    model, _ = load_model(request['model'], dtype=mx.float16)
    aligner = ForcedAligner(request['aligner'], dtype=mx.float16, backend='mlx')
    results = []
    for start, end in request['regions_ms']:
        value = transcribe((wave[round(start * 16):round(end * 16)], 16000),
                           model=model, context=request['context'], language='Chinese',
                           return_timestamps=True, return_chunks=True, forced_aligner=aligner,
                           dtype=mx.float16)
        mx.synchronize()
        results.append(dataclasses.asdict(value))
    (root / 'result.json').write_text(json.dumps(results, ensure_ascii=False, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
