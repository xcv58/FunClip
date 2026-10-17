"""Chinese reading-load diagnostics and unapplied grouping proposals."""

from collections import Counter
import math
from numbers import Real
import unicodedata

DEFAULT_CHINESE_CPS = 9.0
DEFAULT_CHINESE_LINE_UNITS = 16.0


def has_han(text):
    return any(0x3400 <= ord(c) <= 0x4DBF or 0x4E00 <= ord(c) <= 0x9FFF
               or 0xF900 <= ord(c) <= 0xFAFF or 0x20000 <= ord(c) <= 0x323AF for c in text)


def reading_units(text):
    """Full-width characters count one; half-width letters/spaces count half.

    Line breaks are layout, not reading characters. This metric is a stated
    diagnostic convention, not a claim that all viewers read at the same speed.
    """
    return sum(1 if unicodedata.east_asian_width(c) in {'W', 'F'} or has_han(c) else 0.5
               for c in text if c not in '\r\n' and not unicodedata.category(c).startswith('C'))


def chinese_readability_report(cues, *, max_cps=DEFAULT_CHINESE_CPS,
                               max_line_units=DEFAULT_CHINESE_LINE_UNITS):
    for value in (max_cps, max_line_units):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError('Reading limits must be finite positive numbers')
    metrics, issues, proposals = [], [], []
    for cue in cues:
        if not has_han(cue['text']) or cue['end_ms'] <= cue['start_ms']:
            continue
        units = reading_units(cue['text'])
        cps = units * 1000 / (cue['end_ms'] - cue['start_ms'])
        lines = [reading_units(line) for line in cue['text'].splitlines()]
        metric = {'cue_id': cue['cue_id'], 'start_ms': cue['start_ms'], 'end_ms': cue['end_ms'],
                  'reading_units': units, 'characters_per_second': round(cps, 3), 'line_units': lines}
        metrics.append(metric)
        context = {'cue_ids': [cue['cue_id']], 'start_ms': cue['start_ms'], 'end_ms': cue['end_ms'],
                   'severity': 'review', 'display_text': cue['text']}
        if cps > max_cps + 1e-9:
            issues.append(dict(context, code='chinese_reading_speed',
                message=f'Chinese caption reading load {cps:.1f} units per second exceeds the configured {max_cps:g} reference',
                reading_units=units, characters_per_second=round(cps, 3)))
        if any(n > max_line_units for n in lines):
            issues.append(dict(context, code='chinese_line_length',
                message=f'Chinese caption exceeds the configured {max_line_units:g} reading units on a line', line_units=lines))
        if len(lines) > 2:
            issues.append(dict(context, code='chinese_line_count', message='Chinese caption exceeds two text lines'))
    risk_ids = {i['cue_ids'][0] for i in issues if i['code'] == 'chinese_reading_speed'}
    for first, second in zip(cues, cues[1:]):
        a, b = first['start_ms'], second['end_ms']
        gap = second['start_ms'] - first['end_ms']
        if (first['cue_id'] == second['cue_id'] or not 0 <= gap <= 120 or not 0 < b - a <= 8000
                or not has_han(first['text']) or not has_han(second['text'])):
            continue
        speakers = (first.get('speaker'), second.get('speaker'))
        if all(s is not None for s in speakers) and speakers[0] != speakers[1]:
            continue
        if (first['cue_id'] not in risk_ids and second['cue_id'] not in risk_ids
                and first['end_ms'] - first['start_ms'] >= 700 and second['end_ms'] - second['start_ms'] >= 700):
            continue
        texts = [first['text'].strip(), second['text'].strip()]
        if (not all(texts) or any('\n' in t or '\r' in t for t in texts)
                or any(reading_units(t) > max_line_units for t in texts)):
            continue
        proposed = '\n'.join(texts)
        cps = reading_units(proposed) * 1000 / (b - a)
        if cps > max_cps + 1e-9:
            continue
        proposals.append({'source_cue_ids': [first['cue_id'], second['cue_id']], 'start_ms': a, 'end_ms': b,
                          'proposed_text': proposed, 'characters_per_second': round(cps, 3), 'applied': False,
                          'speaker_evidence': 'matching_explicit_ids' if all(s is not None for s in speakers) else 'unavailable',
                          'requires_audio_and_speaker_confirmation': True,
                          'reason': 'Possible grouping for a short or crowded adjacent caption; confirm phrasing and speaker first'})
    return {'version': 1, 'content_verified': False, 'thresholds': {'chinese_max_cps': max_cps,
            'chinese_max_line_units': max_line_units, 'max_lines': 2, 'max_group_gap_ms': 120},
            'counting': 'Full-width and Han count 1; half-width letters digits spaces and punctuation count 0.5; line breaks do not count',
            'metrics': metrics, 'issues': issues, 'issue_counts': dict(Counter(i['code'] for i in issues)),
            'grouping_proposals': proposals, 'applied_changes': 0,
            'limits': ['Reading load is advisory and not proof of a content error',
                       'No word removal punctuation insertion automatic merging or retiming',
                       'Unknown speakers require listening; explicit different speakers are never proposed together']}
