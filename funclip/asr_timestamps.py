"""Validate forced alignment and retain indivisible, measured text spans.

No character times are estimated. Zero length tokens, overlapping tokens and
subword pieces are joined into atomic spans using their original endpoints.
"""

import math
from numbers import Real
from collections.abc import Mapping

from funclip.utils.subtitle_alignment import TOKEN
from funclip.subtitle_style import SubtitleFormattingError


class TimestampAdapterError(SubtitleFormattingError):
    pass


def _letters(text):
    return ''.join(c.casefold() for c in text if c.isalnum())


def adapt_qwen_alignment(native, duration_ms, *, offset_ms=0):
    """Return a renderable sentence and diagnostics; never modify native data.

    Native seconds are rounded to the nearest millisecond, matching SRT's
    resolution. Text coverage, endpoints and monotonicity must be proven first.
    A span containing only point timestamps cannot establish subtitle duration.
    """
    def reject(reason):
        raise TimestampAdapterError(f'Qwen timing requires review: {reason}')

    if not isinstance(native, Mapping):
        reject('invalid native result')
    text = native.get('text')
    if not isinstance(text, str):
        reject('invalid transcript')
    chunks = native.get('chunks') or []
    if not isinstance(chunks, list) or any(not isinstance(c, Mapping) for c in chunks):
        reject('invalid chunk diagnostics')
    if (native.get('truncated') or native.get('finish_reason') in ('length', 'max_tokens')
        or any(c.get('truncated') or c.get('finish_reason') in ('length', 'max_tokens') for c in chunks)):
        reject('generation was truncated')
    tokens = list(TOKEN.finditer(text))
    segments = native.get('segments') or []
    if not tokens:
        if text.strip() or segments:
            reject('no lexical text')
        return None, {'version': 1, 'atomic_spans': 0, 'merged_segments': 0}
    if (not isinstance(segments, list) or not segments
        or any(not isinstance(s, Mapping) or not isinstance(s.get('text'), str) for s in segments)):
        reject('missing forced alignment')
    if (isinstance(duration_ms, bool) or not isinstance(duration_ms, Real) or not math.isfinite(duration_ms) or duration_ms <= 0
        or isinstance(offset_ms, bool) or not isinstance(offset_ms, Real) or not math.isfinite(offset_ms) or offset_ms < 0):
        reject('invalid audio duration')
    # Track lexical token boundaries in a punctuation independent string.
    ends, position = {}, 0
    for index, token in enumerate(tokens):
        position += len(_letters(token.group()))
        ends[position] = index + 1
    if ''.join(_letters(s.get('text', '')) for s in segments) != _letters(text):
        reject('alignment text does not cover transcript exactly')
    groups, pending, position, first_token, trimmed = [], [], 0, 0, 0
    previous_start, previous_end = -1, -1
    for segment in segments:
        values = [segment.get('start'), segment.get('end')]
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
            reject('non finite endpoint')
        a, b = values
        # ForcedAligner quantizes to an 80 ms lattice. Only its final lattice
        # overhang may be trimmed to the measured audio boundary.
        if a < 0 or b < a or a * 1000 >= duration_ms or b * 1000 > duration_ms + 80:
            reject('endpoint outside audio or reversed')
        if a < previous_start or b < previous_end:
            reject('alignment runs backwards')
        previous_start, previous_end = a, b
        if not _letters(segment['text']):
            reject('alignment segment has no lexical text')
        trimmed += int(b * 1000 > duration_ms)
        pending.append((round(a * 1000), round(min(b * 1000, duration_ms))))
        position += len(_letters(segment['text']))
        if position not in ends:
            continue  # English subwords cannot become subtitle cut points.
        last_token = ends[position]
        start, end = pending[0][0], pending[-1][1]
        measured = any(b > a for a, b in pending)
        candidate = [first_token, last_token, start, end, len(pending), measured]
        if groups and (start < groups[-1][3] or end <= start or not groups[-1][5]):
            previous = groups.pop()
            candidate = [previous[0], last_token, min(start, previous[2]), max(end, previous[3]), previous[4] + len(pending), previous[5] or measured]
        groups.append(candidate)
        pending, first_token = [], last_token
    # A leading point group gets its duration only from its next measured span.
    if len(groups) > 1 and groups[0][3] <= groups[0][2]:
        first, second = groups[:2]
        groups[:2] = [[first[0], second[1], first[2], second[3], first[4] + second[4], first[5] or second[5]]]
    if pending or first_token != len(tokens) or any(g[3] <= g[2] or not g[5] for g in groups):
        reject('no positive measured duration for a text span')
    spans = [[tokens[a].start(), tokens[b - 1].end()] for a, b, *_ in groups]
    timestamps = [[g[2] + offset_ms, g[3] + offset_ms] for g in groups]
    sentence = {'text': text, 'timestamp': timestamps, 'timestamp_spans': spans,
                'start': timestamps[0][0], 'end': timestamps[-1][1]}
    report = {'version': 1, 'policy': 'atomic measured spans without estimated character times',
              'atomic_spans': len(groups), 'native_segments': len(segments),
              'merged_segments': len(segments) - len(groups),
              'audio_boundary_trims': trimmed,
              'content_verified': False}
    return sentence, report
