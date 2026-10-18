"""Render shorter Chinese cues using only aligned ASR token timestamps.

Adapted from FunClip's timestamp-based splitting in videoclipper.py at upstream
36acf72e2add3c4af33fd35d373574a5dee96482. Keep literal text and prefer punctuation
or pauses instead of reconstructing text from punctuation-free tokens.
"""

import re

from .subtitle_alignment import aligned_units, TOKEN
from funclip.numeric_spans import numeric_breaks


MAX_SUBTITLE_DURATION_MS = 8000
MAX_SUBTITLE_TOKENS = 30
PAUSE_BREAK_MS = 300

_CHINESE = re.compile(r"[\u4e00-\u9fff]")
_OPENING_PUNCTUATION = frozenset("（([{【《〈「『“‘")


def _text_boundaries(text, tokens):
    """Attach closing punctuation to the preceding cue and opening marks to the next."""
    boundaries = [0]
    for current, following in zip(tokens, tokens[1:]):
        boundary = following.start()
        for offset in range(current.end(), following.start()):
            if text[offset] in _OPENING_PUNCTUATION:
                boundary = offset
                break
        boundaries.append(boundary)
    boundaries.append(len(text))
    return boundaries


def _preferred_end(text, tokens, timestamps, boundaries, start, limit, forbidden):
    # Search the latter half of the available cue to avoid tiny fragments.
    first_candidate = start + max(1, (limit - start + 1) // 2)
    best_end, best_score = limit, (0, limit)
    for end in range(first_candidate, limit + 1):
        if end in forbidden:
            continue
        punctuation = text[tokens[end - 1].end():boundaries[end]]
        if any(mark in punctuation for mark in "。！？!?；;"):
            strength = 3
        elif any(mark in punctuation for mark in "，,、：:"):
            strength = 2
        elif timestamps[end][0] - timestamps[end - 1][1] >= PAUSE_BREAK_MS:
            strength = 1
        else:
            strength = 0
        score = (strength, end)
        if score > best_score:
            best_end, best_score = end, score
    return best_end


def split_long_chinese_sentence(sentence):
    """Return render-only sentence copies; never mutate recognition evidence.

    A token must have exactly one ordered, non-overlapping timestamp pair. If the
    alignment is unavailable or ambiguous, retain the original sentence. Never
    estimate word times by distributing a sentence's duration over its text.
    Non-string and non-Chinese sentences also retain their existing behavior.
    """
    text = sentence.get("text")
    if not isinstance(text, str) or not _CHINESE.search(text):
        return [sentence]
    timestamps = sentence.get("timestamp")
    tokens = aligned_units(text, timestamps, sentence.get('timestamp_spans'))
    if len(tokens) < 2:
        return [sentence]
    if (
        len(TOKEN.findall(text)) <= MAX_SUBTITLE_TOKENS
        and timestamps[-1][1] - timestamps[0][0] <= MAX_SUBTITLE_DURATION_MS
    ):
        return [sentence]

    boundaries = _text_boundaries(text, tokens)
    forbidden = numeric_breaks(text, tokens)
    chunks = []
    start = 0
    while start < len(tokens):
        # A single token may exceed the duration limit; retain its real timing.
        limit = start + 1
        while limit < len(tokens) and sum(len(TOKEN.findall(t.group())) for t in tokens[start:limit + 1]) <= MAX_SUBTITLE_TOKENS:
            if timestamps[limit][1] - timestamps[start][0] > MAX_SUBTITLE_DURATION_MS:
                break
            limit += 1
        # Prefer an earlier safe boundary; a numeric span itself may exceed a
        # display target, in which case retain its real timing rather than split it.
        safe_limit = limit
        while safe_limit > start and safe_limit in forbidden:
            safe_limit -= 1
        if safe_limit == start:
            safe_limit = limit
            while safe_limit < len(tokens) and safe_limit in forbidden:
                safe_limit += 1
        limit = safe_limit
        end = (
            limit if limit == len(tokens)
            else _preferred_end(text, tokens, timestamps, boundaries, start, limit, forbidden)
        )
        chunk = dict(sentence)
        chunk.update(
            text=text[boundaries[start]:boundaries[end]],
            timestamp=timestamps[start:end],
            start=timestamps[start][0],
            end=timestamps[end - 1][1],
        )
        if 'timestamp_spans' in sentence:
            chunk['timestamp_spans'] = [[t.start() - boundaries[start], t.end() - boundaries[start]]
                                        for t in tokens[start:end]]
        chunks.append(chunk)
        start = end
    return chunks
