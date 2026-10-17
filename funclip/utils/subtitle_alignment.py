"""Token offsets and verified ASR timing, shared by render-only transforms."""

import math
from numbers import Real
import re

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
