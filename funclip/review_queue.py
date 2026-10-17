"""Local audio-backed review candidates, with no subtitle edits or truth claims."""

from collections import Counter
import copy
from difflib import SequenceMatcher
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile

from funclip.glossary import glossary_report, resolve_glossary
from funclip.subtitle_meaning import review_facts
from funclip.utils.subtitle_alignment import aligned_tokens

MAX_REVIEW_ITEMS = 200
MAX_COMPARE_TOKENS = 20_000
MAX_CONTEXT_CHECKS = 6
MAX_CONTEXT_AUDIO_MS = 120_000
CONTEXT_PADDING_MS = 5_000
AUDIO_PADDING_MS = 2_500
REVIEW_ARTIFACT_TTL_SECONDS = 3600
_ARTIFACT_LOCK = threading.RLock()
_ARTIFACT_CACHE = None
_OWNED_DIRECTORIES = {}
_MONEY_RATIOS = {'money', 'NTD', 'USD', 'SGD', 'CNY', 'JPY', 'percent', 'permille'}
_PRIORITY = {'high': 0, 'medium': 1, 'low': 2}
REVIEW_LABELS = {
    'fact_check_suggested': 'Check amount percentage or negation',
    'short_turn_review': 'Check a short speech turn',
    'possible_asr_confusion': 'Check a possible name confusion',
    'numeric_boundary_review': 'Check a number boundary',
    'unsupported_hotword': 'Requested spelling unavailable to ASR',
    'hint_comparison_unavailable': 'Comparison unavailable',
    'glossary_name_changed': 'Name changed with recognition hints',
    'asr_hint_off_target_change': 'Other words changed with recognition hints',
    'context_check_unavailable': 'Alternate audio check unavailable',
    'context_fact_disagreement': 'Facts differ with another audio window',
    'chinese_reading_speed': 'Check Chinese reading speed',
    'chinese_line_length': 'Check Chinese line length',
    'chinese_line_count': 'Check Chinese text lines',
    'readability_grouping_candidate': 'Review a possible phrase grouping',
}


def cleanup_review_artifacts(*, now=None):
    """Expire only this process's generated review folders, never source paths."""
    now = time.time() if now is None else now
    removed = 0
    with _ARTIFACT_LOCK:
        for path, used_at in list(_OWNED_DIRECTORIES.items()):
            if now - used_at >= REVIEW_ARTIFACT_TTL_SECONDS:
                try:
                    shutil.rmtree(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    continue
                del _OWNED_DIRECTORIES[path]
                removed += 1
    return removed


def create_review_directory():
    global _ARTIFACT_CACHE
    with _ARTIFACT_LOCK:
        cleanup_review_artifacts()
        if _ARTIFACT_CACHE is None:
            _ARTIFACT_CACHE = tempfile.TemporaryDirectory(prefix='funclip_review_cache_')
        path = tempfile.mkdtemp(prefix='result_', dir=_ARTIFACT_CACHE.name)
        _OWNED_DIRECTORIES[path] = time.time()
        return path


def touch_review_artifacts(directory):
    with _ARTIFACT_LOCK:
        path = str(Path(directory))
        if path in _OWNED_DIRECTORIES:
            _OWNED_DIRECTORIES[path] = time.time()


def review_time(milliseconds):
    if milliseconds is None:
        return 'Glossary profile'
    total = int(milliseconds)
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}.{millis:03}'


def review_details(item):
    def readable(text):
        return re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', str(text))
    result = [item['reason'], f"Decision: {item['decision']}"]
    labels = {'original_text': 'Original recognition', 'before': 'Comparison recognition',
              'after': 'Alternative recognition', 'baseline_context': 'Without hints',
              'hinted_context': 'With hints', 'candidate': 'Reference spelling'}
    labels.update(display_text='Displayed subtitle', proposed_text='Unapplied grouping proposal')
    if item['code'] in {'glossary_name_changed', 'asr_hint_off_target_change'}:
        labels.update(before='Without hints', after='Selected recognition with hints')
    elif item['code'] == 'context_fact_disagreement':
        labels.update(before='Selected recognition', after='Alternate audio window')
    for key, label in labels.items():
        if key in item:
            result.append(f"{label}: {readable(item[key])}")
    return '\n\n'.join(result)


def recognition_view(payload):
    """Independent native recognition evidence, excluding media and session state."""
    return copy.deepcopy({k: payload.get(k) for k in ('text', 'raw_text', 'timestamp', 'sentence_info')})


def _interval(sentence):
    start, end = sentence.get('start'), sentence.get('end')
    if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in (start, end)):
        return None, None
    return (start, end) if 0 <= start < end else (None, None)


def _tokens(evidence):
    text = evidence.get('raw_text') or ''
    times = evidence.get('timestamp')
    tokens = aligned_tokens(text, times)
    if len(tokens) > MAX_COMPARE_TOKENS:
        return []
    return [(t.group(), a, b, t.start(), t.end()) for t, (a, b) in zip(tokens, times if tokens else [])]


def _excerpt(evidence, start, end, *, offset=0):
    tokens = [t for t in _tokens(evidence) if t[1] + offset < end and t[2] + offset > start]
    if not tokens:
        return ''
    # Slice the retained raw record rather than inventing timing or text.
    return evidence['raw_text'][tokens[0][3]:tokens[-1][4]]


def _readable_facts(text):
    # Native raw_text has spaces between Chinese tokens. Join only adjacent Han
    # characters for fact parsing; the raw excerpt itself remains verbatim.
    compact = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    return review_facts(compact)


def context_targets(evidence, duration_ms):
    """Bounded alternate windows for explicit money/ratio/negation review."""
    targets, coverage, spent = [], [], 0
    for index, sentence in enumerate(evidence.get('sentence_info') or [], 1):
        start, end = _interval(sentence)
        if start is None:
            continue
        facts = review_facts(sentence.get('text', ''))
        if not (any(unit in _MONEY_RATIOS for _, unit in facts['numbers']) or facts['negations']):
            continue
        a, b = max(0, start - CONTEXT_PADDING_MS), min(duration_ms, end + CONTEXT_PADDING_MS)
        if b <= a or b - a > 30_000:
            continue
        existing = next((w for w in targets if w['start_ms'] <= start and end <= w['end_ms']), None)
        if existing:
            existing['sentence_ids'].append(index)
            continue
        if len(targets) == MAX_CONTEXT_CHECKS or spent + b - a > MAX_CONTEXT_AUDIO_MS:
            coverage.append(index)
            continue
        targets.append({'start_ms': a, 'end_ms': b, 'sentence_ids': [index]})
        spent += b - a
    return targets, coverage


def build_review_queue(evidence, *, glossary=None, baseline=None, context_checks=(),
                       numeric_report=None, glossary_evidence=None, comparison_error=None,
                       context_skipped=()):
    profile = resolve_glossary(glossary)
    sentences = evidence.get('sentence_info') or []
    items = []

    def add(code, reason, start=None, end=None, *, priority='medium', ids=(), **details):
        items.append({'code': code, 'reason': reason, 'priority': priority,
                      'start_ms': start, 'end_ms': end, 'sentence_ids': list(ids),
                      'decision': 'unresolved', 'content_verified': False, **details})

    for index, sentence in enumerate(sentences, 1):
        text = sentence.get('text', '')
        start, end = _interval(sentence)
        facts = review_facts(text)
        if any(unit in _MONEY_RATIOS for _, unit in facts['numbers']) or facts['negations']:
            add('fact_check_suggested', 'Check the spoken amount percentage or negation', start, end,
                ids=[index], original_text=text, facts=facts)
        compact = ''.join(c for c in text if c.isalnum())
        acknowledgement = re.fullmatch(r'(?:(?:谢谢|对|好|是|嗯|哦|啊|哈|可以|没错|不客气))+', compact)
        if start is not None and end - start <= 3000 and 2 <= len(compact) <= 18 and not acknowledgement:
            add('short_turn_review', 'Short speech turn may need speaker or wording review', start, end,
                priority='low', ids=[index], original_text=text)
    terminology = glossary_report([(i, s.get('text', '')) for i, s in enumerate(sentences, 1)], profile)
    for issue in terminology['issues']:
        index = issue['cue_id']
        sentence = sentences[index - 1]
        a, b = _interval(sentence)
        add('possible_asr_confusion', issue['action'], a, b, priority='high', ids=[index],
            original_text=sentence['text'], candidate=issue['candidate'], term_id=issue['term_id'])
    for issue in (numeric_report or {}).get('issues', []):
        ids = issue['sentence_ids']
        selected = [sentences[i - 1] for i in ids]
        a, _ = _interval(selected[0])
        _, b = _interval(selected[-1])
        add('numeric_boundary_review', 'Numeric boundary could not be safely joined', a, b,
            priority='high', ids=ids, original_text=''.join(s['text'] for s in selected),
            boundary_code=issue['code'])
    for word in (glossary_evidence or {}).get('unsupported_hotwords', []):
        add('unsupported_hotword', 'ASR cannot encode this requested spelling; this does not prove it was spoken',
            candidate=word)
    if baseline is not None:
        left, right = _tokens(baseline), _tokens(evidence)
        if (not left or not right) and ((baseline.get('raw_text') or '') or (evidence.get('raw_text') or '')):
            add('hint_comparison_unavailable', 'Token timing is unavailable or exceeds the comparison limit')
        else:
            old_text, new_text = baseline.get('raw_text') or '', evidence.get('raw_text') or ''
            def name_positions(text):
                # Compare in the same token space as raw ASR. Terms must match a
                # whole native token sequence; names never become replacements.
                from funclip.utils.subtitle_alignment import TOKEN
                values = [t.group().casefold() for t in TOKEN.finditer(text)]
                positions = set()
                if profile:
                    for entry in profile.entries:
                        for form in entry.forms:
                            needle = [t.group().casefold() for t in TOKEN.finditer(form)]
                            for i in range(len(values) - len(needle) + 1):
                                if needle and values[i:i + len(needle)] == needle:
                                    positions.update(range(i, i + len(needle)))
                return positions
            old_names, new_names = name_positions(old_text), name_positions(new_text)
            matcher = SequenceMatcher(a=[t[0].casefold() for t in left], b=[t[0].casefold() for t in right], autojunk=False)
            for tag, a, b, c, d in matcher.get_opcodes():
                if tag == 'equal':
                    continue
                changed = left[a:b] + right[c:d]
                start, end = min(t[1] for t in changed), max(t[2] for t in changed)
                before = old_text[left[a][3]:left[b - 1][4]] if b > a else ''
                after = new_text[right[c][3]:right[d - 1][4]] if d > c else ''
                inside_name = bool(set(range(a, b)) & old_names or set(range(c, d)) & new_names)
                code = 'glossary_name_changed' if inside_name else 'asr_hint_off_target_change'
                add(code, 'Recognition differs with ASR hints; listen before accepting the change', start, end,
                    priority='medium' if inside_name else 'high', before=before, after=after,
                    baseline_context=_excerpt(baseline, max(0, start - 1500), end + 1500),
                    hinted_context=_excerpt(evidence, max(0, start - 1500), end + 1500))
    if comparison_error:
        add('hint_comparison_unavailable', 'The optional baseline recognition failed; the selected recognition is retained')
    for check in context_checks:
        if check.get('error'):
            add('context_check_unavailable', 'Alternate-window recognition failed; original recognition is retained',
                check['start_ms'], check['end_ms'], ids=check['sentence_ids'])
            continue
        for index in check['sentence_ids']:
            sentence = sentences[index - 1]
            a, b = _interval(sentence)
            original = _excerpt(evidence, max(0, a - 150), b + 150)
            alternative = _excerpt(check['evidence'], max(0, a - 150), b + 150, offset=check['start_ms'])
            if not original or not alternative:
                add('context_check_unavailable', 'No aligned alternate excerpt covers this passage', a, b, ids=[index])
                continue
            before, after = _readable_facts(original), _readable_facts(alternative)
            if before != after:
                add('context_fact_disagreement', 'The same ASR model gives different facts with another audio window',
                    a, b, priority='high', ids=[index], before=original, after=alternative,
                    original_facts=before, alternate_facts=after,
                    context_start_ms=check['start_ms'], context_end_ms=check['end_ms'])
    specific_ids = {index for item in items if item['code'] in {'context_fact_disagreement', 'possible_asr_confusion', 'numeric_boundary_review'}
                    for index in item['sentence_ids']}
    items = [i for i in items if i['code'] not in {'fact_check_suggested', 'short_turn_review'}
             or not set(i['sentence_ids']) & specific_ids]
    items.sort(key=lambda x: (_PRIORITY[x['priority']], x['start_ms'] is None, x['start_ms'] or 0, x['code']))
    total = len(items)
    source_hash = hashlib.sha256(json.dumps(recognition_view(evidence), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    items = items[:MAX_REVIEW_ITEMS]
    for n, item in enumerate(items, 1):
        item['id'] = f'{source_hash[:12]}-{n}'
    return {'version': 1, 'basis': 'original_asr_sentence_ids_and_media_milliseconds',
            'source_sha256': source_hash, 'content_verified': False,
            'status': 'needs_review' if items else 'no_candidates', 'items': items,
            'total_detected': total, 'omitted_items': total - len(items),
            'issue_counts': dict(Counter(x['code'] for x in items)),
            'hint_comparison': 'failed' if comparison_error else 'completed' if baseline is not None else 'not_requested',
            'context_checks': len(context_checks), 'context_skipped_sentence_ids': list(context_skipped),
            'glossary': profile.metadata() if profile else None,
            'reference_terms': [{'id': e.id, 'spoken_forms': list(e.spoken_forms), 'spoken_aliases': list(e.spoken_aliases), 'menu_name': e.menu_name,
                                 'zh_hans': e.zh_hans, 'zh_hant': e.zh_hant, 'english': e.english,
                                 'evidence': e.evidence} for e in profile.entries] if profile else [],
            'limits': ['Candidates and model agreement do not certify spoken content',
                       'ASR alternatives never replace the selected transcript automatically',
                       'Context checks and item counts are bounded; skipped items are explicit']}


def decide_review_item(queue, item_id, decision):
    if decision not in {'unresolved', 'keep_original', 'needs_edit'}:
        raise ValueError('Unknown review decision')
    result = copy.deepcopy(queue)
    item = next((i for i in result['items'] if i['id'] == item_id), None)
    if item is None:
        raise ValueError('Review item does not belong to this result')
    item['decision'] = decision
    decisions = [i['decision'] for i in result['items']]
    result['status'] = ('needs_edit' if 'needs_edit' in decisions else
                        'needs_review' if 'unresolved' in decisions else
                        'partial_review' if result.get('omitted_items') else 'reviewed')
    return result


def include_readability_review(queue, quality_report, evidence):
    """Add display-cue diagnostics without confusing them with ASR sentence IDs."""
    result = copy.deepcopy(queue)
    items = result['items']
    readability = quality_report.get('readability', {})
    for issue in readability.get('issues', []):
        a, b = issue['start_ms'], issue['end_ms']
        native_ids = [i for i, s in enumerate(evidence.get('sentence_info') or [], 1)
                      if (interval := _interval(s))[0] is not None and interval[0] < b and interval[1] > a]
        items.append({'id': f"{result['source_sha256'][:12]}-reading-{len(items)}",
            'code': issue['code'], 'reason': issue['message'], 'priority': 'medium', 'start_ms': a, 'end_ms': b,
            'sentence_ids': native_ids, 'display_cue_ids': issue['cue_ids'], 'decision': 'unresolved',
            'content_verified': False, 'display_text': issue.get('display_text', ''), 'readability': issue})
    for proposal in readability.get('grouping_proposals', []):
        items.append({'id': f"{result['source_sha256'][:12]}-group-{len(items)}",
            'code': 'readability_grouping_candidate', 'reason': proposal['reason'], 'priority': 'low',
            'start_ms': proposal['start_ms'], 'end_ms': proposal['end_ms'], 'sentence_ids': [],
            'display_cue_ids': proposal['source_cue_ids'], 'proposed_text': proposal['proposed_text'],
            'proposal': proposal, 'decision': 'unresolved', 'content_verified': False})
    items.sort(key=lambda x: (_PRIORITY[x['priority']], x['start_ms'] is None, x['start_ms'] or 0, x['code']))
    result['total_detected'] = len(items) + queue.get('omitted_items', 0)
    result['items'] = items[:MAX_REVIEW_ITEMS]
    result['omitted_items'] = result['total_detected'] - len(result['items'])
    result['issue_counts'] = dict(Counter(i['code'] for i in result['items']))
    result['status'] = 'needs_review' if result['items'] else 'no_candidates'
    result['readability'] = {'thresholds': readability.get('thresholds', {}), 'applied_changes': 0}
    return result


def write_review_bundle(queue, audio, sample_rate, directory):
    """Write derived review audio and a portable local report, never source files."""
    import soundfile as sf
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = copy.deepcopy(queue)
    result['bundle_id'] = uuid.uuid4().hex
    for item in result['items']:
        # A decision from an old upload must not match an identical ASR result
        # produced by a later request. Source hashes remain stable provenance.
        item['id'] = result['bundle_id'][:12] + '-' + item['id']
        if audio is None or item['start_ms'] is None:
            continue
        a = max(0, int(item['start_ms'] - AUDIO_PADDING_MS))
        b = min(round(len(audio) * 1000 / sample_rate), int(item['end_ms'] + AUDIO_PADDING_MS))
        if b <= a:
            continue
        # Bound playback artifacts without stretching or guessing source timing.
        b = min(b, a + 30_000)
        path = directory / (item['id'] + '.wav')
        sf.write(path, audio[round(a * sample_rate / 1000):round(b * sample_rate / 1000)], sample_rate, subtype='PCM_16')
        item.update(audio_file=str(path), audio_start_ms=a, audio_end_ms=b)
    return save_review_bundle(result, directory)


def save_review_bundle(queue, directory):
    directory = Path(directory)
    path = directory / 'review.json'
    path.write_text(json.dumps(queue, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    body = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>FunClip subtitle review</title><style>body{font:16px/1.6 system-ui;max-width:950px;margin:auto;padding:24px}article{border:1px solid #ddd;border-radius:10px;padding:18px;margin:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}audio{max-width:100%}</style>',
            '<h1>Subtitle review</h1><p>Listen before deciding. ASR alternatives and agreement are not verification. These decisions do not edit subtitles.</p>']
    for item in queue['items']:
        body.append(f'<article><h2>{escape(REVIEW_LABELS[item["code"]])}</h2><p>{review_time(item["start_ms"])}–{review_time(item["end_ms"])} · {escape(item["priority"])}</p><pre>{escape(review_details(item))}</pre>')
        if item.get('audio_file'):
            body.append(f'<audio controls preload="none" src="{escape(Path(item["audio_file"]).name, quote=True)}"></audio>')
        body.append('</article>')
    (directory / 'review.html').write_text(''.join(body))
    bundle = directory / 'review.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        active = {directory / 'review.html', path}
        for item in queue['items']:
            if item.get('audio_file'):
                audio = Path(item['audio_file'])
                if audio.parent.resolve() == directory.resolve():
                    active.add(audio)
        for file in sorted(active):
            archive.write(file, file.name)
    return queue, str(path), str(bundle)
