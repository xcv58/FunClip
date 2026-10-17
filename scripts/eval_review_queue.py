#!/usr/bin/env python3
"""Replay review cases and optionally run local audio with fixed cached models."""
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
from funclip.glossary import parse_glossary
from funclip.review_queue import build_review_queue, write_review_bundle
from funclip.service_contract import recognition_evidence
from scripts.eval_glossary_audio import MODEL_IDS, sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'eval/review_queue_cases/manifest.json')
    parser.add_argument('--audit-root', type=Path)
    parser.add_argument('--references-root', type=Path, default=Path('/Volumes/Crucial X10/srt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live-audio', action='store_true')
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = parser.parse_args()
    out, refs = args.output.resolve(), args.references_root.resolve()
    if out.is_relative_to(refs) or out.is_relative_to(args.manifest.parent.resolve()) or (
            args.audit_root and out.is_relative_to(args.audit_root.resolve())):
        parser.error('Output must be outside unchanged reference audit and fixture directories')
    if args.live_audio and not args.audit_root:
        parser.error('--live-audio requires the existing local audit WAVs')
    manifest = json.loads(args.manifest.read_text())
    profile_file = args.manifest.parent / 'profiles.json'
    assert sha(profile_file) == manifest['profiles_sha256']
    profiles = {k: parse_glossary(v) for k, v in json.loads(profile_file.read_text()).items()}
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'new_llm_requests': 0,
              'content_accuracy_certified': False, 'audit_cases': [], 'cached_coverage': [], 'live_audio': [],
              'scope': manifest['scope'], 'profile_notes': manifest['profile_notes']}
    code = {str(ROOT / f): sha(ROOT / f) for f in ['funclip/review_queue.py', 'funclip/glossary.py',
        'funclip/videoclipper.py', 'funclip/service_contract.py', 'funclip/subtitle_meaning.py', 'gradio_app.py', 'app.py']}
    save(out / 'code_sha256.json', code)
    baseline = json.loads((args.audit_root / 'integrity_before.json').read_text()) if args.audit_root else None
    integrity = {}
    if baseline:
        for field in ['final_srt_sha256', 'selected_srt_sha256', 'model_sha256']:
            paths = {str(refs / p) if field != 'model_sha256' else p: h for p, h in baseline[field].items()}
            assert all(sha(Path(p)) == h for p, h in paths.items()), field
            integrity[field] = paths
        save(out / 'integrity_before.json', integrity)
    if args.audit_root:
        import soundfile as sf
    for case in manifest['cases']:
        path = (args.manifest.parent / case['fixture']).resolve()
        assert path.is_relative_to(args.manifest.parent.resolve()) and sha(path) == case['sha256']
        data = json.loads(path.read_text())
        before = copy.deepcopy(data)
        q = build_review_queue(data['evidence'], glossary=profiles[case['episode']], baseline=data.get('baseline'),
            context_checks=data.get('context_checks', []), glossary_evidence=data.get('glossary_evidence'))
        assert data == before
        assert set(case['expected_codes']) <= q['issue_counts'].keys(), case['id']
        save(out / 'cases' / case['id'] / 'queue_before_audio.json', q)
        if args.audit_root:
            audio = args.audit_root / 'audio' / case['audio']
            audit_manifest = json.loads((args.audit_root / 'manifest.json').read_text())
            expected = audit_manifest['derived_sha256']['audio/' + case['audio']]
            assert sha(audio) == expected
            wave, sr = sf.read(audio, dtype='float32')
            assert sr == 16000 and wave.ndim == 1
            q, _, _ = write_review_bundle(q, wave, sr, out / 'cases' / case['id'])
        report['audit_cases'].append({'id': case['id'], 'passed': True, 'issue_counts': q['issue_counts'],
            'expected_codes': case['expected_codes'], 'raw_evidence_unchanged': True,
            'playable_items': sum(bool(i.get('audio_file')) for i in q['items'])})
    if args.audit_root:
        for path in sorted((args.audit_root / 'asr').glob('*.json')):
            if not path.name.endswith(('.baseline.json', '.glossary.json', '.context.json')):
                continue
            item = json.loads(path.read_text())
            e = item['evidence']
            old = copy.deepcopy(e)
            name = path.name.replace('.glossary.json', '.baseline.json')
            base_path = path.with_name(name)
            comparison = json.loads(base_path.read_text())['evidence'] if path.name.endswith('.glossary.json') else None
            q = build_review_queue(e, glossary=profiles[item['episode']], baseline=comparison,
                                   glossary_evidence=e.get('glossary'))
            assert e == old
            report['cached_coverage'].append({'id': path.stem, 'raw_evidence_unchanged': True,
                'item_count': len(q['items']), 'omitted_items': q['omitted_items'], 'issue_counts': q['issue_counts']})
    if args.live_audio:
        os.environ['MODELSCOPE_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        from funasr import AutoModel
        from funclip.videoclipper import VideoClipper
        model_paths = [Path.home() / '.cache/modelscope/hub/models' / s for s in MODEL_IDS]
        if not all(p.is_dir() for p in model_paths):
            raise ValueError('Existing cached ASR VAD and punctuation models are required')
        model = AutoModel(model=str(model_paths[0]), vad_model=str(model_paths[1]), punc_model=str(model_paths[2]),
            device=args.device, disable_update=True, trust_remote_code=False, log_level='ERROR', disable_pbar=True)
        clipper = VideoClipper(model)
        clipper.lang = 'zh'
        for case in manifest['cases']:
            path = args.audit_root / 'audio' / case['audio']
            wave, sr = sf.read(path, dtype='float32')
            results = {}
            for mode in ['reference', 'hints']:
                start = time.perf_counter()
                text, srt, state = clipper.recog((sr, wave), glossary=profiles[case['episode']],
                    glossary_mode=mode, compare_hints=mode == 'hints')
                e = recognition_evidence(text, srt, state)
                old_path = args.audit_root / 'asr' / (case['id'] + ('.baseline.json' if mode == 'reference' else '.glossary.json'))
                old = json.loads(old_path.read_text())['evidence']
                same = all(e[k] == old[k] for k in ['raw_text', 'timestamp', 'sentence_info'])
                assert same, f'Primary recognition drift: {case["id"]} {mode}'
                if mode == 'reference':
                    assert e['glossary']['active_hotwords'] == []
                else:
                    base = state['review_evidence']['baseline']
                    assert all(base[k] == results['reference']['evidence'][k] for k in ['raw_text', 'timestamp', 'sentence_info'])
                q, _, _ = write_review_bundle(state['review_queue'], wave, sr, out / 'live_audio' / case['id'] / mode)
                result = {'evidence': e, 'queue': q, 'elapsed_seconds': time.perf_counter() - start,
                          'same_primary_as_audit': same, 'audio_sha256': sha(path), 'seconds': len(wave) / sr}
                save(out / 'live_audio' / case['id'] / (mode + '.json'), result)
                results[mode] = result
                print(case['id'], mode, 'same primary', same, 'candidates', len(q['items']), flush=True)
            report['live_audio'].append({'id': case['id'], 'seconds': len(wave) / sr,
                'reference_no_hints': True, 'both_primaries_match_audit': True,
                'unbiased_comparison_matches_reference': True,
                'hint_issues': results['hints']['queue']['issue_counts']})
        # Exercise the actual bounded alternate-window path on the wider source,
        # independently of the deterministic cached 10500/10100 comparison.
        path = args.audit_root / 'audio/fire-310-355.context.wav'
        wave, sr = sf.read(path, dtype='float32')
        text, srt, state = clipper.recog((sr, wave), glossary=profiles['fire'], glossary_mode='reference', review_context=True)
        e = recognition_evidence(text, srt, state)
        old = json.loads((args.audit_root / 'asr/fire-310-355.context.json').read_text())['evidence']
        assert all(e[k] == old[k] for k in ['raw_text', 'timestamp', 'sentence_info'])
        checks = state['review_evidence']['context_checks']
        assert 0 < len(checks) <= 6
        assert sum(c['end_ms'] - c['start_ms'] for c in checks) <= 120_000
        q, _, _ = write_review_bundle(state['review_queue'], wave, sr, out / 'live_audio/context_check')
        save(out / 'live_audio/context_check.json', {'evidence': e, 'queue': q, 'same_primary_as_audit': True})
        report['live_context'] = {'same_primary_as_audit': True, 'check_count': len(checks),
            'additional_audio_seconds': sum(c['end_ms'] - c['start_ms'] for c in checks) / 1000,
            'issue_counts': q['issue_counts']}
    if baseline:
        assert all(sha(Path(p)) == h for paths in integrity.values() for p, h in paths.items())
        save(out / 'integrity_after.json', integrity)
        report['integrity'] = {'all_unchanged': True, **{field: len(paths) for field, paths in integrity.items()}}
    assert all(sha(Path(p)) == h for p, h in code.items())
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    save(out / 'results.json', report)
    print(json.dumps({'audit_cases_passed': len(report['audit_cases']), 'cached_payloads': len(report['cached_coverage']),
                      'live_cases': len(report['live_audio']), 'live_context': report.get('live_context'),
                      'integrity': report.get('integrity')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
