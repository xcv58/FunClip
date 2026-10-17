"""Local, advisory SRT diagnostics. Never rewrite text or certify speech content."""

from collections import Counter
import hashlib
import math
from numbers import Real
import re
from funclip.subtitle_readability import chinese_readability_report, DEFAULT_CHINESE_CPS, DEFAULT_CHINESE_LINE_UNITS


QUALITY_REPORT_VERSION = 2
MAX_CUE_DURATION_MS = 8_000
SHORT_CUE_DURATION_MS = 700
SPARSE_CUE_MIN_DURATION_MS = 15_000
MIN_CONTENT_CHARACTERS_PER_SECOND = 1
MAX_REPORT_CHARACTERS = 100_000
MAX_REPORT_CUES = 3_000
_TIMING = re.compile(
    r"^([0-9]{1,6}):([0-5][0-9]):([0-5][0-9])[,.]([0-9]{3})"
    r"[ \t]*-->[ \t]*"
    r"([0-9]{1,6}):([0-5][0-9]):([0-5][0-9])[,.]([0-9]{3})(?:[ \t]+.*)?$"
)
_REPEATED_TEXT = re.compile(r"(.{2,40}?)\1{2,}")


def _milliseconds(values):
    hours, minutes, seconds, millis = map(int, values)
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def _content(text):
    # Includes non-BMP Han characters such as 𫊻. Punctuation is evidence, but
    # does not contribute to text density or repetition diagnostics.
    return "".join(c.casefold() for c in text if c.isalnum())


def build_srt_quality_report(srt_content, *, media_duration_seconds=None,
                             chinese_max_cps=DEFAULT_CHINESE_CPS, chinese_max_line_units=DEFAULT_CHINESE_LINE_UNITS):
    """Report timing/repetition risks; an empty issue list is not content approval.

    Unlike export validators, this parser retains diagnostics for malformed cues
    rather than raising on the first one. All checks are read-only and local.
    Gaps are intentionally not classified as missing speech or music.
    """
    if not isinstance(srt_content, str):
        raise TypeError("Subtitle content must be a string.")
    media_end = None
    if media_duration_seconds is not None:
        if (isinstance(media_duration_seconds, bool)
                or not isinstance(media_duration_seconds, Real)
                or not math.isfinite(media_duration_seconds)
                or media_duration_seconds < 0):
            raise ValueError("Media duration must be finite nonnegative seconds.")
        media_end = round(media_duration_seconds * 1000)
    report = {
        "version": QUALITY_REPORT_VERSION,
        "scope": "srt-timing-and-repetition",
        "source_sha256": hashlib.sha256(srt_content.encode("utf-8")).hexdigest(),
        "content_verified": False,
        "media_duration_ms": media_end,
        "cue_count": 0,
        "status": "clear",
        "issues": [],
        "thresholds": {
            "long_cue_ms": MAX_CUE_DURATION_MS,
            "short_cue_ms": SHORT_CUE_DURATION_MS,
            "sparse_min_duration_ms": SPARSE_CUE_MIN_DURATION_MS,
            "sparse_content_characters_per_second": MIN_CONTENT_CHARACTERS_PER_SECOND,
        },
    }

    def flag(code, message, *, cue_ids=(), start_ms=None, end_ms=None,
             severity="review", block=None):
        item = {"code": code, "severity": severity, "cue_ids": list(cue_ids),
                "start_ms": start_ms, "end_ms": end_ms, "message": message}
        if block is not None:
            item["block"] = block
        report["issues"].append(item)

    readable_cues = []
    if len(srt_content) > MAX_REPORT_CHARACTERS:
        flag("analysis_limit", "Subtitle exceeds the quality report size limit.", severity="error")
    else:
        normalized = srt_content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
        blocks = [b for b in re.split(r"\n[ \t]*\n+", normalized) if b.strip()]
        report["cue_count"] = len(blocks)
        if len(blocks) > MAX_REPORT_CUES:
            flag("analysis_limit", "Subtitle exceeds the quality report cue limit.", severity="error")
            blocks = []
        seen = set()
        latest_end, latest_id, previous_start = 0, None, 0
        repeated = []

        def flush_repeated():
            if len(repeated) >= 3:
                flag("repeated_adjacent_cues", "Repeated consecutive captions need review; they may be genuine speech.",
                     cue_ids=[c[0] for c in repeated], start_ms=repeated[0][1], end_ms=repeated[-1][2])
            repeated.clear()

        for number, block in enumerate(blocks, start=1):
            lines = block.strip("\n").split("\n")
            cue_id = lines[0].strip()
            timing = _TIMING.fullmatch(lines[1].strip()) if len(lines) >= 2 else None
            if (len(lines) < 3 or not cue_id.isascii() or not cue_id.isdecimal()
                    or len(cue_id) > 12 or int(cue_id) < 1 or timing is None):
                flush_repeated()
                flag("malformed_cue", "Cue needs an ID, valid SRT timestamps and body text.", severity="error", block=number)
                continue
            cue_id = int(cue_id)
            start, end = _milliseconds(timing.groups()[:4]), _milliseconds(timing.groups()[4:])
            context = {"cue_ids": [cue_id], "start_ms": start, "end_ms": end}
            text = "\n".join(lines[2:])
            if cue_id in seen:
                flag("duplicate_cue_id", "Cue ID occurs more than once.", severity="error", **context)
            seen.add(cue_id)
            if any(not line.strip() for line in lines[2:]):
                flag("empty_body", "Cue has an empty text line.", severity="error", **context)
            duration = end - start
            if duration > 0:
                readable_cues.append({'cue_id': cue_id, 'start_ms': start, 'end_ms': end, 'text': text})
            if duration <= 0:
                flag("nonpositive_duration", "Cue must end after it starts.", severity="error", **context)
            else:
                if duration > MAX_CUE_DURATION_MS:
                    flag("long_cue", "Caption exceeds eight seconds; review its speech boundaries.", **context)
                if duration < SHORT_CUE_DURATION_MS:
                    flag("short_cue", "Caption is shorter than 700 milliseconds and may flash too quickly.", **context)
                if (duration >= SPARSE_CUE_MIN_DURATION_MS
                        and len(_content(text)) * 1000 < duration * MIN_CONTENT_CHARACTERS_PER_SECOND):
                    flag("sparse_long_cue", "Very little text spans a long interval; verify the audio rather than spreading words across it.", **context)
            if start < previous_start:
                flag("out_of_order", "Cue start times are out of order.", severity="error", **context)
            if start < latest_end and latest_id is not None:
                flag("overlap", "Cue overlaps an earlier cue.", severity="error",
                     cue_ids=[latest_id, cue_id], start_ms=start, end_ms=end)
            previous_start = start
            if end > latest_end:
                latest_end, latest_id = end, cue_id
            if media_end is not None and end > media_end:
                flag("outside_media", "Cue extends past the supplied media duration.", severity="error", **context)
            compact = _content(text)
            # Bound each repetition check, including on malformed oversized bodies.
            if len(compact) <= 500:
                match = _REPEATED_TEXT.search(compact)
                if match and len(match.group()) >= len(compact) * 0.8:
                    flag("repeated_text", "A repeated phrase dominates this caption; verify it against the audio.", **context)
            else:
                flag("body_analysis_limit", "Caption is too large for repetition analysis.", **context)
            if repeated and (compact != repeated[-1][3] or start - repeated[-1][2] > 1000):
                flush_repeated()
            if compact:
                repeated.append((cue_id, start, end, compact))
            else:
                flush_repeated()
        flush_repeated()
    report['readability'] = chinese_readability_report(readable_cues, max_cps=chinese_max_cps,
        max_line_units=chinese_max_line_units)
    report['issues'].extend(report['readability']['issues'])
    report['thresholds'].update(report['readability']['thresholds'])
    report["issue_counts"] = dict(Counter(i["code"] for i in report["issues"]))
    if report["issues"]:
        report["status"] = "needs_review"
    return report


def quality_status_suffix(reports):
    """A short UI hint; full per-cue reports remain available in API metadata."""
    issues = [i for report in reports for i in report["issues"]]
    if not issues:
        return ""
    examples = []
    for issue in issues[:3]:
        cue_ids = issue["cue_ids"]
        location = "cue " + "/".join(map(str, cue_ids[:3])) if cue_ids else "subtitle"
        if len(cue_ids) > 3:
            location += " and others"
        examples.append(f"{location}: {issue['message']}")
    return f" Review suggested for {len(issues)} finding(s). " + " ".join(examples)
