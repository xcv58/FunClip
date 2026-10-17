#!/usr/bin/env python3
"""Replay a frozen two-video manifest and optionally run the current local ASR.

No API uploads are performed here. Live translation responses, if separately
authorized, are read from their saved per-phase JSON records.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'funclip')]
from funclip.meaning_alignment import srt_alignment_report, meaning_alignment_report, ANCHOR_CATALOG_SHA256
from funclip.subtitle_meaning import check_meaning_preservation, MeaningPreservationError
from funclip.subtitle_quality import build_srt_quality_report
from funclip.subtitle_style import SubtitleFormattingError
from funclip.service_contract import recognition_evidence
from scripts.eval_glossary_audio import MODEL_IDS, sha


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--live-audio', action='store_true')
    p.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    input_root, out = args.manifest.parent.resolve(), args.output.resolve()
    episodes = {e['key']: e for e in manifest['episodes']}
    references = Path(next(iter(episodes.values()))['video']).parent.parent.resolve()
    if out.is_relative_to(references) or out == input_root / 'inputs' or out.is_relative_to(input_root / 'inputs'):
        p.error('Output must be outside source references and frozen excerpts')
    integrity = json.loads((input_root / 'integrity_before.json').read_text())
    archive = json.loads((input_root / 'archive_hashes_before.json').read_text())
    assert all(sha(Path(path)) == value for paths in integrity.values() for path, value in paths.items())
    assert all(sha(Path(path)) == value for path, value in archive.items())
    definition = json.loads((input_root / 'frozen_definition.json').read_text())
    assert definition['anchor_catalog_sha256'] == ANCHOR_CATALOG_SHA256
    assert all(sha(Path(path)) == value for path, value in definition['source_sha256'].items())
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(args.manifest),
              'baseline_commit': manifest['baseline_commit'], 'model_ids': MODEL_IDS,
              'content_accuracy_certified': False, 'audio': [], 'translations': [], 'episodes': [],
              'anchor_catalog_sha256': ANCHOR_CATALOG_SHA256,
              'limits': ['Selected excerpts are not a random population sample',
                         'Finals are read-only editorial references rather than audio-certified verbatim truth',
                         'Review candidate counts and lexical matches are not confirmed error rates',
                         'Readability changes here are diagnostics and proposals; no retiming is applied',
                         'LLM phases are one sample per excerpt and can vary between runs',
                         'Manual correction time is not measured without a human editing study']}
    controls = json.loads((ROOT / 'eval/alignment_cases/manifest.json').read_text())
    report['synthetic_controls'] = []
    for case in controls['cases']:
        alignment = meaning_alignment_report(case['source'], case['candidate'], mode=case['mode'])
        try:
            check_meaning_preservation(case['source'], case['candidate'], mode=case['mode'])
            old_accept = True
        except MeaningPreservationError:
            old_accept = False
        assert alignment['status'] == case['expected_alignment_status'], case['id']
        report['synthetic_controls'].append(dict(case, baseline_numeric_guard_accepts=old_accept,
            alignment_status=alignment['status'], issues=alignment['issues'], passed=True))
    if args.live_audio:
        os.environ['MODELSCOPE_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        import soundfile as sf
        from funasr import AutoModel
        from funclip.videoclipper import VideoClipper
        paths = [Path.home() / '.cache/modelscope/hub/models' / x for x in MODEL_IDS]
        model = AutoModel(model=str(paths[0]), vad_model=str(paths[1]), punc_model=str(paths[2]),
            device=args.device, disable_update=True, trust_remote_code=False, log_level='ERROR', disable_pbar=True)
        clipper = VideoClipper(model)
        clipper.lang = 'zh'
        for index, case in enumerate(manifest['cases'], 1):
            before = json.loads((input_root / 'baseline_audio' / (case['id'] + '.json')).read_text())
            audio = Path(case['audio'])
            assert sha(audio) == before['audio_sha256']
            waveform, rate = sf.read(audio, dtype='float32')
            assert rate == 16000 and waveform.ndim == 1
            began = time.monotonic()
            try:
                text, srt, state = clipper.recog((rate, waveform), glossary_mode='reference')
                error = None
            except SubtitleFormattingError as exc:
                state, srt, error = exc.recognition_state, '', str(exc)
                text = state.get('recognition_text', '')
            after = {'id': case['id'], 'srt': srt, 'formatting_error': error,
                     'evidence': recognition_evidence(text, srt, state), 'quality_report': build_srt_quality_report(srt),
                     'review_queue': state.get('review_queue'), 'elapsed_seconds': time.monotonic() - began}
            save(out / 'after_audio' / (case['id'] + '.json'), after)
            print(f"Audio {index}/{len(manifest['cases'])} {case['id']}", flush=True)
    for case in manifest['cases']:
        before = json.loads((input_root / 'baseline_audio' / (case['id'] + '.json')).read_text())
        after_path = out / 'after_audio' / (case['id'] + '.json')
        if after_path.exists():
            after = json.loads(after_path.read_text())
            # Re-evaluate only derived diagnostics when replaying saved ASR.
            # The retained recognition evidence and rendered SRT stay verbatim.
            current_quality = build_srt_quality_report(after['srt'])
            keys = ['raw_text', 'text', 'timestamp', 'sentence_info']
            report['audio'].append({'id': case['id'], 'seconds': before['seconds'],
                'native_evidence_identical': all(before['evidence'].get(k) == after['evidence'].get(k) for k in keys),
                'rendered_srt_identical': before['srt'] == after['srt'], 'formatting_error': after['formatting_error'],
                'readability_counts': current_quality['readability']['issue_counts'],
                'grouping_proposals': len(current_quality['readability']['grouping_proposals'])})
        source_path = Path(case['input_srt'])
        assert sha(source_path) == case['input_sha256']
        source = source_path.read_text()
        row = {'id': case['id'], 'phases': {}}
        for phase in ['baseline', 'after']:
            path = input_root / 'live_translation' / phase / (case['id'] + '.json')
            if not path.exists():
                continue
            value = json.loads(path.read_text())
            item = {'status': value['status'], 'requests': len(value['requests']), 'elapsed_seconds': value['elapsed_seconds']}
            if value['status'] == 'completed':
                alignment = srt_alignment_report(source, value['srt'], mode='translation')
                save(out / 'translation_alignment' / phase / (case['id'] + '.json'), alignment)
                item.update(alignment_status=alignment['status'], issue_counts=alignment['issue_counts'],
                            reviewed_cues=sorted({i['cue_id'] for i in alignment['issues']}),
                            blocking_cues=sorted({i['cue_id'] for i in alignment['issues'] if i['severity'] == 'error'}))
            row['phases'][phase] = item
        report['translations'].append(row)
    for episode in episodes.values():
        source = Path(episode['source_srt']).read_text()
        final_english = Path(episode['final_english']).read_text()
        readability = build_srt_quality_report(source)
        save(out / 'episode_readability' / (episode['key'] + '.json'), readability)
        alignment = srt_alignment_report(source, final_english, mode='translation')
        save(out / 'episode_alignment' / (episode['key'] + '.json'), alignment)
        report['episodes'].append({'id': episode['key'], 'name': episode['name'],
            'source_cues': readability['cue_count'], 'readability_counts': readability['readability']['issue_counts'],
            'grouping_proposals': len(readability['readability']['grouping_proposals']),
            'reference_alignment_counts': alignment['issue_counts']})
    request_total = sum(i['requests'] for row in report['translations'] for i in row['phases'].values())
    assert request_total <= manifest['maximum_provider_requests']
    report['summary'] = {'audio_windows': len(report['audio']), 'audio_seconds': sum(i['seconds'] for i in report['audio']),
        'native_outputs_identical': sum(i['native_evidence_identical'] for i in report['audio']),
        'rendered_outputs_identical': sum(i['rendered_srt_identical'] for i in report['audio']),
        'translation_provider_requests': request_total,
        'translation_jobs': {phase: dict(Counter(row['phases'][phase]['status'] for row in report['translations'] if phase in row['phases'])) for phase in ['baseline', 'after']}}
    actual = {kind: {path: sha(Path(path)) for path in paths} for kind, paths in integrity.items()}
    assert actual == integrity
    assert all(sha(Path(path)) == value for path, value in archive.items())
    save(out / 'integrity_after.json', actual)
    report['integrity'] = {'unchanged': True, 'all_archive_srts': len(archive), **{k: len(v) for k, v in integrity.items()}}
    save(out / 'results.json', report)
    print(json.dumps(report['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
