"""Token offsets and verified ASR timing, shared by render-only transforms."""

import math
from numbers import Real
import re
from dataclasses import dataclass

TOKEN = re.compile(r"[\u4e00-\u9fff]|[^\W\u4e00-\u9fff]+(?:[-'’][^\W\u4e00-\u9fff]+)*")


def aligned_tokens(text, timestamps):
    tokens = list(TOKEN.finditer(text))
    if not isinstance(timestamps, (list, tuple)) or len(tokens) != len(timestamps):
        return []
    previous_end = 0
    for pair in timestamps:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return []
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in pair):
            return []
        start, end = pair
        if start < previous_end or end <= start:
            return []
        previous_end = end
    return tokens


@dataclass(frozen=True)
class TextSpan:
    text: str
    begin: int
    finish: int

    def start(self):
        return self.begin

    def end(self):
        return self.finish

    def group(self):
        return self.text[self.begin:self.finish]


def aligned_units(text, timestamps, spans=None):
    """Strict timing for lexical tokens or explicitly adapted atomic text spans."""
    if spans is None:
        return aligned_tokens(text, timestamps)
    if not isinstance(spans, list) or not isinstance(timestamps, list) or len(spans) != len(timestamps):
        return []
    tokens, units, index = list(TOKEN.finditer(text)), [], 0
    previous_end = 0
    for pair, timing in zip(spans, timestamps):
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
            or any(type(v) is not int for v in pair) or index >= len(tokens)):
            return []
        start, end = pair
        if start != tokens[index].start():
            return []
        while index < len(tokens) and tokens[index].end() < end:
            index += 1
        if index >= len(tokens) or end != tokens[index].end():
            return []
        if (not isinstance(timing, (list, tuple)) or len(timing) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in timing)):
            return []
        a, b = timing
        if a < previous_end or b <= a:
            return []
        previous_end = b
        units.append(TextSpan(text, start, end))
        index += 1
    return units if index == len(tokens) else []
