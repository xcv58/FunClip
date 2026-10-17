"""Protect aligned numeric phrases for display without rewriting ASR evidence.

Native punctuation can divide a recognized number across sentence boundaries.
Only a narrow, supported boundary repair is automatic. Ambiguous boundaries
remain intact and are reported for review; this module never corrects acoustics.
"""

import copy
import re

from funclip.utils.subtitle_alignment import TOKEN, aligned_tokens

NUMERIC_SPAN_VERSION = 1
MAX_JOIN_GAP_MS = 120
_CN = "零〇一二两兩三四五六七八九十百千万萬亿億"
_DIGIT = "一二两兩三四五六七八九"
_CURRENCY = r"新台币|新臺幣|人民币|人民幣|美元|美金|台币|臺幣|新币|新幣|日元|港币|港幣|US dollars|USD|NTD|SGD"
_UNIT = rf"{_CURRENCY}|摄氏度|攝氏度|公里|千米|公斤|千克|小时|小時|分钟|分鐘|年|天|米|秒|元|块|塊|个|個|人|份|倍|percent|per cent|years?|days?|meters?|metres?"
_INTEGER = rf"(?:\d+(?:,\d{{3}})*|[{_CN}]+)"
_VALUE = rf"{_INTEGER}(?:[.点點](?:\d+|[零〇{_DIGIT}]+))?(?:万|萬|亿|億)?"
_SPAN = re.compile(
    rf"[负負正+−-]?(?:(?:百分之|千分之)|(?:NT\$|US\$|\$))?[负負正+−-]?"
    rf"(?:[{_CN}]+分之[{_CN}]+|{_VALUE})"
    rf"(?:[%‰]|\s*(?:{_UNIT})(?![a-zA-Z]))?", re.I,
)
_END_PUNCT = re.compile(r"[。！？!?；;，,\.]+\s*$")
_CURRENCY_START = re.compile(rf"(?:万|萬|亿|億)?(?:{_CURRENCY})(?![a-zA-Z])", re.I)
_INDEPENDENT_UNIT = re.compile(rf"(?:{_UNIT}|只|条|條|次|岁|歲)")


def numeric_text_spans(text):
    """Offsets of literal numeric phrases; no value inference or rewriting."""
    return [(m.start(), m.end()) for m in _SPAN.finditer(text)]


def numeric_breaks(text, tokens):
    """Token break indices that would cut a numeric phrase or its unit."""
    spans = numeric_text_spans(text)
    return {i for i in range(1, len(tokens))
            if any(a < tokens[i].start() < b for a, b in spans)}


def protect_numeric_sentences(sentences, *, raw_text="", timestamps=None):
    """Return rendering copies and a report using original sentence IDs.

    Automatic cases are a trailing Chinese digit after a ratio ending in 十,
    and an explicit currency unit stranded at the next sentence's start. Both
    need raw-token identity, ordered timing, a gap <=120 ms, and equal speakers.
    Bare digit lists, missing alignment and speaker changes are never joined.
    """
    rendered = copy.deepcopy(list(sentences))
    origins = [[i] for i in range(1, len(rendered) + 1)]
    report = {"version": NUMERIC_SPAN_VERSION, "content_verified": False,
              "basis": "original_asr_sentence_ids", "adjustments": [], "issues": []}
    raw_tokens = aligned_tokens(raw_text, timestamps) if isinstance(raw_text, str) else []
    raw_map = {(a, b): token.group().casefold()
               for token, (a, b) in zip(raw_tokens, timestamps if raw_tokens else [])}
    if len(raw_map) != len(raw_tokens):
        raw_map = {}  # Duplicate timing cannot establish token identity.

    i = 0
    while i + 1 < len(rendered):
        left, right = rendered[i:i + 2]
        lt, rt = left.get("text"), right.get("text")
        if not isinstance(lt, str) or not isinstance(rt, str):
            i += 1
            continue
        punctuation = _END_PUNCT.search(lt)
        left_core = lt[:punctuation.start()] if punctuation else lt.rstrip()
        right_start = rt.lstrip()
        ratio = bool(re.search(rf"(?:百分之|千分之)[{_CN}]*十$", left_core)
                     and re.match(rf"[{_DIGIT}](?![{_CN}点點\d])", right_start))
        currency = bool(re.search(rf"[\d{_CN}]$", left_core) and _CURRENCY_START.match(right_start))
        bare = bool(re.search(rf"[\d{_CN}]$", left_core)
                    and re.match(rf"[\d{_CN}]", right_start))
        if not (ratio or currency or bare):
            i += 1
            continue
        separator = " " if re.search(r"[a-zA-Z0-9]$", left_core) and re.match(r"[a-zA-Z]", rt) else ""
        joined = left_core + separator + rt
        boundary = len(left_core + separator)
        crossing = next(((a, b) for a, b in numeric_text_spans(joined) if a < boundary < b), None)
        # With leading spaces, the currency may begin at the join after whitespace.
        if crossing is None:
            i += 1
            continue
        left_tokens = aligned_tokens(lt, left.get("timestamp"))
        right_tokens = aligned_tokens(rt, right.get("timestamp"))
        gap = None
        if left_tokens and right_tokens:
            gap = right["timestamp"][0][0] - left["timestamp"][-1][1]
        # A following independent count must not turn 80 percent + 3 people into 83 percent.
        separate_count = ratio and (_INDEPENDENT_UNIT.match(right_start[1:])
                                   or not re.fullmatch(rf"[{_DIGIT}][。！？!?；;，,.\s]*", right_start))
        reason = None
        if not (ratio or currency) or separate_count:
            reason = "ambiguous_numeric_boundary"
        elif left.get("spk") != right.get("spk"):
            reason = "numeric_boundary_speaker_change"
        elif gap is None or not 0 <= gap <= MAX_JOIN_GAP_MS:
            reason = "numeric_boundary_timing_requires_review"
        else:
            combined = list(left["timestamp"]) + list(right["timestamp"])
            aligned = aligned_tokens(joined, combined)
            if (not aligned or not raw_map or any(raw_map.get(tuple(pair)) != t.group().casefold()
                                                  for t, pair in zip(aligned, combined))):
                reason = "numeric_boundary_alignment_requires_review"
        ids = list(dict.fromkeys(origins[i] + origins[i + 1]))
        if reason:
            report["issues"].append({"code": reason, "sentence_ids": ids,
                                     "left_text": lt, "right_text": rt, "gap_ms": gap})
            i += 1
            continue

        prefix_end = crossing[1] - boundary
        moved = sum(t.start() < prefix_end for t in right_tokens)
        split_char = right_tokens[moved].start() if moved < len(right_tokens) else len(rt)
        prefix = rt[:split_char]
        # Move the misplaced stop after the restored value/unit if it has none.
        ending = punctuation.group().strip() if punctuation and not _END_PUNCT.search(prefix) else ""
        left["text"] = left_core + separator + prefix.rstrip() + ending
        left["timestamp"] = list(left["timestamp"]) + list(right["timestamp"][:moved])
        left["end"] = left["timestamp"][-1][1]
        origins[i] = ids
        report["adjustments"].append({"code": "numeric_span_boundary_restored", "sentence_ids": ids,
                                       "kind": "ratio" if ratio else "currency", "moved_prefix": prefix,
                                       "start_ms": left["timestamp"][0][0], "end_ms": left["end"]})
        if moved < len(right_tokens):
            right["text"] = rt[split_char:]
            right["timestamp"] = list(right["timestamp"][moved:])
            right["start"] = right["timestamp"][0][0]
            i += 1
        else:
            del rendered[i + 1]
            del origins[i + 1]
    report["protected_span_count"] = sum(len(numeric_text_spans(s.get("text", "")))
                                         for s in rendered if isinstance(s.get("text"), str))
    report["status"] = "needs_review" if report["issues"] else "clear"
    return rendered, report


def numeric_span_status_suffix(report):
    if not report:
        return ""
    restored, issues = len(report["adjustments"]), report["issues"]
    message = f" Numeric boundaries restored: {restored}" if restored else ""
    if issues:
        ids = sorted({i for issue in issues for i in issue['sentence_ids']})[:6]
        message += f" Numeric boundary review suggested: {len(issues)} item(s) at ASR sentences {ids}"
    return message
