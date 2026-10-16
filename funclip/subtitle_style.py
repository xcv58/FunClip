"""Local punctuation-free formatting for subtitle bodies, never SRT syntax."""

import re
import unicodedata
from functools import lru_cache


SUBTITLE_STYLE = "punctuation-free-v1"


class SubtitleFormattingError(ValueError):
    """A subtitle needs review before it can be rendered without punctuation."""


_CHINESE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_APOSTROPHIZED_WORD = re.compile(r"(?<![A-Za-z])[A-Za-z]+(?:['’‘ʼʻ＇‛][A-Za-z]+)+(?![A-Za-z])")
_CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "shan't": "shall not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "couldn't": "could not", "wouldn't": "would not",
    "shouldn't": "should not", "mustn't": "must not", "needn't": "need not",
    "i'm": "I am", "you're": "you are", "we're": "we are", "they're": "they are",
    "i've": "I have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "I will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "it'll": "it will", "we'll": "we will", "they'll": "they will",
    "could've": "could have", "would've": "would have", "should've": "should have",
    "must've": "must have", "might've": "might have", "let's": "let us",
    "o'clock": "o clock", "ma'am": "madam", "y'all": "you all",
}
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.．])(?:(?P<sign>[+\-−])[ \t]*)?"
    r"(?P<number>\d+(?:[,，]\d+)*(?:[.．]\d+)?)"
    r"(?:[ \t]*(?P<unit>[%％‰]))?"
)
_URL_OR_EMAIL = re.compile(
    r"(?:https?|ftp)://\S+|(?<![A-Za-z0-9_])www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.I
)
_DOMAIN_OR_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z][A-Za-z0-9-]*)+(?![A-Za-z0-9_-])"
)
_NUMERIC_OPERATOR = re.compile(r"\d\s*[-−–—+*/／⁄÷×:：·=]\s*\d")
_DECORATIVE_SYMBOLS = frozenset("|¦★☆●○■□◆◇▶▷►▸")


def _expand_contraction(match):
    word = match.group().translate(str.maketrans({char: "'" for char in "’‘ʼʻ＇‛"}))
    expanded = _CONTRACTIONS.get(word.lower())
    if expanded is None:
        raise SubtitleFormattingError(
            "Ambiguous apostrophized word or name requires review or a meaning-preserving "
            "rewrite in full words without apostrophes"
        )
    if word.isupper():
        return expanded.upper()
    if word[0].isupper():
        return expanded[0].upper() + expanded[1:]
    return expanded


def _format_number(match, *, language):
    number = match.group("number")
    number = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in number)
    number = number.replace("，", ",").replace("．", ".")
    whole, dot, fraction = number.partition(".")
    if "," in whole:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", whole) is None:
            raise SubtitleFormattingError("Ambiguous numeric separators require review or a worded rendition")
        whole = whole.replace(",", "")
    chinese = language != "en"
    point = ("點" if language == "zh-Hant" else "点") if chinese else " point "
    rendered = whole + (point + fraction if dot else "")
    unit = match.group("unit")
    if unit:
        if chinese:
            rendered = ("千分之" if unit == "‰" else "百分之") + rendered
        else:
            rendered += " per mille" if unit == "‰" else " percent"
    sign = match.group("sign")
    if sign:
        if chinese:
            rendered = (("負" if language == "zh-Hant" else "负") if sign != "+" else "正") + rendered
        else:
            rendered = ("negative " if sign != "+" else "positive ") + rendered
    return rendered


def _format_line(line, *, language):
    if _URL_OR_EMAIL.search(line):
        raise SubtitleFormattingError("URL or email address requires a worded rendition for review")
    # Dotted uppercase initialisms are unambiguous; arbitrary dotted names are not.
    line = re.sub(r"(?<![A-Za-z])(?:[A-Z]\.){2,}", lambda match: match.group().replace(".", ""), line)
    if _DOMAIN_OR_IDENTIFIER.search(line):
        raise SubtitleFormattingError("Dotted name or identifier requires a worded rendition for review")
    if (
        _NUMERIC_OPERATOR.search(line)
        or re.search(r"[A-Za-z][-−]\d|(?<=\w)[/／\\_](?=\w)|[@#]\w|\w&\w", line)
        or re.search(r"[+\-−]\s*[+\-−]\s*\d", line)
        or re.search(r"(?<!\d)[.．]\d", line)
    ):
        raise SubtitleFormattingError("Date time range formula or identifier requires a worded rendition for review")
    if re.search(r"(?<![A-Za-z])[A-Za-z]+s['’‘ʼʻ＇‛](?=\s|$)", line):
        raise SubtitleFormattingError("Possible possessive apostrophe requires review or a rewrite in full words")
    line = _APOSTROPHIZED_WORD.sub(_expand_contraction, line)
    line = _NUMBER.sub(lambda match: _format_number(match, language=language), line)
    if re.search(r"\d\s*[.,，．]\s*\d", line):
        raise SubtitleFormattingError("Ambiguous numeric punctuation requires a worded rendition for review")
    if any(mark in line for mark in "%％‰"):
        raise SubtitleFormattingError("Percentage sign without an unambiguous value requires review")
    line = line.replace("&", " 和 " if language != "en" else " and ")
    characters = []
    for char in line:
        category = unicodedata.category(char)
        if char in " \t" or category == "Zs":
            characters.append(" ")
        elif category.startswith("P") or char in _DECORATIVE_SYMBOLS:
            characters.append(" ")
        elif category.startswith(("C", "S")) or category in {"Zl", "Zp"} or char in "ʼʻ":
            raise SubtitleFormattingError("Unsupported symbol or control character requires review or a worded rendition")
        else:
            characters.append(char)
    formatted = " ".join("".join(characters).split())
    if not formatted:
        raise SubtitleFormattingError("Punctuation-only or empty subtitle line requires review")
    return formatted


def format_subtitle_text(text, *, language=None):
    """Format each body line without changing its line count or inventing wording."""
    if not isinstance(text, str) or not text:
        raise SubtitleFormattingError("Empty subtitle text requires review")
    if language is None:
        language = "en"
        if _CHINESE.search(text):
            language = "zh-Hant" if _simplifier().convert(text) != text else "zh"
    if language not in {"zh", "zh-Hant", "en"}:
        raise ValueError("Unsupported subtitle formatting language")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(_format_line(line, language=language) for line in text.split("\n"))


@lru_cache(maxsize=1)
def _simplifier():
    from opencc import OpenCC
    return OpenCC("t2s")


def format_srt_content(source, *, language=None, text_transform=None):
    """Format only validated body lines, preserving every header and line ending.

    Import the shared structure validator at call time so lightweight text
    rendering does not import the LLM stack and correction can use this module.
    """
    from funclip.llm.srt_corrector import parse_correction_srt, correction_structure_report

    _, cues = parse_correction_srt(source)
    replacements = {}
    for cue in cues:
        text = text_transform(cue.text) if text_transform else cue.text
        try:
            formatted = format_subtitle_text(text, language=language)
        except SubtitleFormattingError as exc:
            raise SubtitleFormattingError(f"Cue {cue.cue_id} requires review: {exc}") from exc
        original_lines, formatted_lines = cue.text.split("\n"), formatted.split("\n")
        if len(original_lines) != len(formatted_lines):
            raise SubtitleFormattingError(f"Cue {cue.cue_id} formatting changed its line count")
        offset = cue.text_start
        for original, replacement in zip(original_lines, formatted_lines):
            replacements[offset] = replacement
            offset += len(original) + 1

    parts, offset = [], 0
    for match in re.finditer(r"[^\r\n]+(?:\r\n|\r|\n|$)|(?:\r\n|\r|\n)", source):
        raw = match.group()
        ending = "\r\n" if raw.endswith("\r\n") else raw[-1:] if raw.endswith(("\r", "\n")) else ""
        body = raw[:-len(ending)] if ending else raw
        parts.append(replacements.pop(offset, body) + ending)
        offset += len(body) + bool(ending)
    if replacements:
        raise SubtitleFormattingError("Could not preserve the SRT body layout during formatting")
    formatted = "".join(parts)
    correction_structure_report(source, formatted)
    return formatted
