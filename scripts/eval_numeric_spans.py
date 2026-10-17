#!/usr/bin/env python3
"""Offline numeric-boundary and equivalent-expression audit replay.

Optionally replay all cached audit payloads and re-run the two selected local
WAVs with the existing cached ASR/VAD/punctuation weights. No LLM API is called.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'funclip')]
from funclip.numeric_spans import protect_numeric_sentences
from funclip.service_contract import recognition_evidence
from funclip.subtitle_meaning import check_meaning_preservation, MeaningPreservationError
from funclip.utils.subtitle_alignment import TOKEN
from funclip.utils.subtitle_utils import generate_srt
from scripts.eval_glossary_audio import MODEL_IDS, sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def replay(evidence):
    before = copy.deepcopy(evidence)
    native = evidence['sentence_info']
    rendered, report = protect_numeric_sentences(native, raw_text=evidence['raw_text'],
                                                timestamps=evidence['timestamp'])
    assert evidence == before, 'Raw evidence was modified'
    assert [p for s in native for p in s['timestamp']] == [p for s in rendered for p in s['timestamp']]
    assert [t.group() for s in native for t in TOKEN.finditer(s['text'])] == [
        t.group() for s in rendered for t in TOKEN.finditer(s['text'])]
    return {'report': report, 'raw_evidence_unchanged': True,
            'srt': generate_srt(rendered, split_long_chinese=True),
            'correction_source_srt': generate_srt(rendered, split_long_chinese=True, format_output=False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'eval/numeric_span_cases/manifest.json')
    parser.add_argument('--audit-root', type=Path)
    parser.add_argument('--references-root', type=Path, default=Path('/Volumes/Crucial X10/srt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live-audio', action='store_true')
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = parser.parse_args()
    out, references = args.output.resolve(), args.references_root.resolve()
    if out.is_relative_to(references) or (args.audit_root and out.is_relative_to(args.audit_root.resolve())):
        parser.error('Output must be outside the unchanged reference and audit directories')
    if args.live_audio and not args.audit_root:
        parser.error('--live-audio requires --audit-root with existing WAV derivatives')
    manifest = json.loads(args.manifest.read_text())
    results = {'started_at': datetime.now(timezone.utc).isoformat(), 'content_accuracy_certified': False,
               'new_llm_requests': 0, 'fixtures': [], 'equivalents': [], 'cached_replays': [],
               'historical_translations': [], 'live_audio': []}
    source_hashes = {str(p): sha(p) for p in (ROOT / 'funclip/numeric_spans.py',
        ROOT / 'funclip/subtitle_meaning.py', ROOT / 'funclip/utils/subtitle_segmentation.py',
        ROOT / 'funclip/videoclipper.py', ROOT / 'funclip/service_contract.py', ROOT / 'gradio_app.py',
        ROOT / 'app.py')}
    save(out / 'code_sha256.json', source_hashes)
    for case in manifest['cases']:
        path = (args.manifest.parent / case['evidence']).resolve()
        if not path.is_relative_to(args.manifest.parent.resolve()) or sha(path) != case['evidence_sha256']:
            raise ValueError('Fixture path/hash mismatch')
        result = replay(json.loads(path.read_text()))
        assert len(result['report']['adjustments']) == case['expected_adjustments']
        assert result['report']['adjustments'][0]['kind'] == case['expected_kind']
        assert case['expected_text'] in result['srt']
        save(out / 'fixtures' / f"{case['id']}.json", result)
        results['fixtures'].append({'id': case['id'], 'passed': True, 'adjustments': result['report']['adjustments']})
    for case in manifest['equivalent_controls']:
        try:
            check_meaning_preservation([(1, case['source'])], [(1, case['translation'])], mode='translation')
            accepted, issues = True, []
        except MeaningPreservationError as exc:
            accepted, issues = False, exc.issues
        assert accepted == case['expected_accept'], case
        results['equivalents'].append(dict(case, previous_accept=case['actual_accept'], actual_accept=accepted, issues=issues))
    if args.audit_root:
        for path in sorted((args.audit_root / 'asr').glob('*.json')):
            if not path.name.endswith(('.baseline.json', '.glossary.json', '.context.json')):
                continue
            payload = json.loads(path.read_text())
            result = replay(payload['evidence'])
            save(out / 'cached' / path.name, result)
            results['cached_replays'].append({'id': path.stem, 'raw_evidence_unchanged': True,
                'display_changed': result['srt'] != payload['rendered_srt'],
                'adjustments': result['report']['adjustments'], 'issues': result['report']['issues']})
        from funclip.llm.srt_corrector import parse_correction_srt
        history = json.loads((args.audit_root / 'historical_translation_audit.json').read_text())
        for item in history:
            _, source = parse_correction_srt((references / item['source_path']).read_text())
            _, target = parse_correction_srt((references / item['generated_path']).read_text())
            try:
                check_meaning_preservation([(c.cue_id, c.text) for c in source],
                                           [(c.cue_id, c.text) for c in target], mode='translation')
                issues = []
            except MeaningPreservationError as exc:
                issues = exc.issues
            results['historical_translations'].append({'episode': item['episode'],
                'source_path': item['source_path'], 'generated_path': item['generated_path'],
                'cue_count': len(source), 'previous_issues': item['meaning_issues'], 'current_issues': issues,
                'not_an_error_count': True})
    baseline = json.loads((args.audit_root / 'integrity_before.json').read_text()) if args.audit_root else None
    integrity = {}
    if baseline:
        for field in ['final_srt_sha256', 'selected_srt_sha256', 'model_sha256']:
            paths = {str(references / p) if field != 'model_sha256' else p: expected
                     for p, expected in baseline[field].items()}
            assert all(sha(Path(p)) == h for p, h in paths.items()), f'Changed {field}'
            integrity[field] = paths
        save(out / 'integrity_before.json', integrity)
    if args.live_audio:
        os.environ['MODELSCOPE_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        from funasr import AutoModel
        import soundfile as sf
        from funclip.videoclipper import VideoClipper
        model_paths = [Path.home() / '.cache/modelscope/hub/models' / name for name in MODEL_IDS]
        if not all(p.is_dir() for p in model_paths):
            raise ValueError('Existing cached production models are required')
        model = AutoModel(model=str(model_paths[0]), vad_model=str(model_paths[1]), punc_model=str(model_paths[2]),
            device=args.device, disable_update=True, trust_remote_code=False, log_level='ERROR', disable_pbar=True)
        clipper = VideoClipper(model)
        clipper.lang = 'zh'
        for case in manifest['cases']:
            path = args.audit_root / 'audio' / (case['id'] + '.wav')
            wav, sr = sf.read(path, dtype='float32')
            assert sr == 16000 and wav.ndim == 1
            started = time.perf_counter()
            text, rendered, state = clipper.recog((sr, wav))
            evidence = recognition_evidence(text, rendered, state)
            old = json.loads((args.audit_root / 'asr' / (case['id'] + '.baseline.json')).read_text())['evidence']
            same_raw = all(evidence[k] == old[k] for k in ['raw_text', 'timestamp', 'sentence_info'])
            assert case['expected_text'] in rendered, f'Expected protected number absent: {case["id"]}'
            assert state['sentences'] == evidence['sentence_info']
            record = {'id': case['id'], 'audio_sha256': sha(path), 'seconds': len(wav) / sr,
                      'elapsed_seconds': time.perf_counter() - started, 'model_ids': MODEL_IDS,
                      'device': args.device, 'same_raw_as_audit': same_raw, 'evidence': evidence}
            save(out / 'live_audio' / (case['id'] + '.json'), record)
            results['live_audio'].append({k: v for k, v in record.items() if k != 'evidence'})
            print(case['id'], 'live current-model protection passed', 'same raw', same_raw, flush=True)
    if baseline:
        assert all(sha(Path(p)) == h for paths in integrity.values() for p, h in paths.items())
        results['integrity'] = {'all_unchanged': True, **{field: len(paths) for field, paths in integrity.items()}}
        save(out / 'integrity_after.json', integrity)
    assert source_hashes == {p: sha(Path(p)) for p in source_hashes}
    results['finished_at'] = datetime.now(timezone.utc).isoformat()
    save(out / 'results.json', results)
    print(json.dumps({'fixtures': len(results['fixtures']), 'equivalents': len(results['equivalents']),
                      'cached_replays': len(results['cached_replays']), 'live_audio': len(results['live_audio']),
                      'integrity': results.get('integrity')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
