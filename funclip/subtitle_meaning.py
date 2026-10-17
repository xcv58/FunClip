"""Conservative local guards for supported facts, never acoustic truth checks."""

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
import re
import unicodedata


MEANING_GUARD_VERSION = 1
_DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
_SMALL = {"十": 10, "百": 100, "千": 1000}
_LARGE = {"万": 10_000, "亿": 100_000_000}
_CN = "零〇一二两三四五六七八九十百千万亿"
_NUMBER = re.compile(
    rf"(?P<leading>[+\-负正])?(?P<ratio>百分之|千分之)?(?P<sign>[+\-负正])?"
    rf"(?P<number>(?:\d+(?:,\d{{3}})*|[{_CN}]+)(?:[.点](?:\d+|[零〇一二两三四五六七八九]+))?)"
    r"(?P<scale>万|亿)?\s*(?P<symbol>[%‰])?"
)
_EN_VALUES = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
_EN_VALUES.update(dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10))))
_EN_VALUES.update({"both": 2, "noon": 12})
_EN_SCALE = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
_EN_SCALE["century"] = 100
_ORDINALS = dict(zip("first second third fourth fifth sixth seventh eighth ninth tenth".split(), range(1, 11)))
_WORDS = "|".join((*_EN_VALUES, *_EN_SCALE))
_EN_NUMBER = re.compile(
    rf"(?<![a-z])(?:{_WORDS})(?:[ -]+(?:and +)?(?:{_WORDS}))*"
    rf"(?: +point +(?:{_WORDS})(?:[ -]+(?:{_WORDS}))*)?(?![a-z])"
)
_ORDINAL = re.compile(r"(?<![a-z])(?:" + "|".join(_ORDINALS) + r")(?![a-z])")
_UNITS = {
    "new taiwan dollars": "NTD", "taiwan dollars": "NTD", "ntd": "NTD", "新台币": "NTD", "台币": "NTD",
    "us dollars": "USD", "usd": "USD", "美元": "USD", "美金": "USD",
    "singapore dollars": "SGD", "sgd": "SGD", "新币": "SGD",
    "人民币": "CNY", "cny": "CNY", "rmb": "CNY", "日元": "JPY", "jpy": "JPY",
    "dollars": "money", "dollar": "money", "元": "money", "块钱": "money",
    "percent": "percent", "per cent": "percent", "per mille": "permille",
    "kilometers": "km", "kilometres": "km", "公里": "km", "千米": "km", "km": "km",
    "meters": "m", "metres": "m", "meter": "m", "metre": "m", "米": "m", "m": "m",
    "degrees": "degree", "degree": "degree", "摄氏度": "degree", "度": "degree",
    "hours": "hour", "hour": "hour", "小时": "hour", "个小时": "hour",
    "minutes": "minute", "minute": "minute", "分钟": "minute",
    "seconds": "second", "second": "second", "秒": "second",
    "kilograms": "kg", "kilogram": "kg", "kg": "kg", "公斤": "kg", "千克": "kg",
    "years": "year", "year": "year", "年": "year", "days": "day", "day": "day", "天": "day",
    "people": "count", "persons": "count", "dishes": "count", "times": "count",
    **{s: "count" for s in ("个人", "人", "个", "位", "座", "间", "条", "道", "只", "台", "顿", "次", "倍", "家", "块", "份", "件", "代", "边", "字", "点")},
}
_UNIT = re.compile("(?:" + "|".join(re.escape(u) + (r"(?![a-z])" if u.isascii() else "")
                                     for u in sorted(_UNITS, key=len, reverse=True)) + ")")
_MONEY = frozenset({"money", "NTD", "USD", "SGD", "CNY", "JPY"})
_NEGATION = re.compile(
    r"不能|无法|没法|不可以|不得|不许|不准|不必|不用|无需|不要|不|没有|没|无效|未曾|未能|尚未|"
    r"别(?=去|用|吃|说|再|动|拿|买|忘|担心)|勿|"
    r"(?<![a-z])(?:cannot|can't|couldn't|won't|wouldn't|shouldn't|don't|doesn't|didn't|"
    r"isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|mustn't|not|never|without|no)(?![a-z])"
)
_COMPARISON = re.compile(r"比(?!如|方|较|例|赛|喻|对)|相比|相较|相对于")
_COMPARISON_OPERATORS = r"凉爽|凉快|暖和|便宜|高|低|多|少|大|小|快|慢|贵"
_APPROX = re.compile(r"大约|大概|约|左右|将近|(?<![a-z])(?:about|approximately|roughly|around)(?![a-z])")


class MeaningPreservationError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        locations = ", ".join(f"cue {i['cue_id']} {i['code']}" for i in issues[:4])
        super().__init__("Meaning preservation requires review: " + locations)


@lru_cache(maxsize=1)
def _simplifier():
    from opencc import OpenCC
    return OpenCC("t2s")


def _normalize(text):
    value = _simplifier().convert(unicodedata.normalize("NFKC", text)).lower().replace("’", "'").replace("−", "-")
    value = re.sub(r"(\d+)\s+point\s+(\d+)", r"\1.\2", value)
    return " ".join(value.split())


def _zh_integer(text):
    if not any(c in _SMALL or c in _LARGE for c in text):
        return int("".join(str(_DIGITS[c]) for c in text))
    total = section = digit = 0
    for c in text:
        if c in _DIGITS:
            digit = _DIGITS[c]
        elif c in _SMALL:
            section += (digit or 1) * _SMALL[c]
            digit = 0
        else:
            section += digit
            if c == "亿":
                total = (total + section) * _LARGE[c]
            else:
                total += section * _LARGE[c]
            section = digit = 0
    return total + section + digit


def _zh_decimal(text):
    whole, dot, fraction = text.replace(".", "点").partition("点")
    integer = int(whole.replace(",", "")) if whole[0].isdigit() else _zh_integer(whole)
    return Decimal(str(integer) + ("." + "".join(c if c.isdigit() else str(_DIGITS[c]) for c in fraction) if dot else ""))


def _english_decimal(text):
    whole, dot, fraction = text.replace("-", " ").partition(" point ")
    total = group = 0
    for word in whole.split():
        if word in _EN_VALUES:
            group += _EN_VALUES[word]
        elif word == "hundred":
            group = (group or 1) * 100
        elif word in _EN_SCALE:
            total += (group or 1) * _EN_SCALE[word]
            group = 0
    value = str(total + group)
    if dot:
        words = fraction.split()
        if any(w not in _EN_VALUES or _EN_VALUES[w] > 9 for w in words):
            raise ValueError("Unsupported spoken decimal fraction.")
        value += "." + "".join(str(_EN_VALUES[w]) for w in words)
    return Decimal(value)


def _unit(text, start, end):
    suffix = text[end:].lstrip()
    match = _UNIT.match(suffix)
    unit = _UNITS[match.group()] if match else None
    finish = end + len(text[end:]) - len(suffix) + len(match.group()) if match else end
    prefix = text[max(0, start - 8):start].rstrip()
    for marker, currency in (("nt$", "NTD"), ("us$", "USD"), ("sg$", "SGD"), ("s$", "SGD"), ("$", "money"), ("usd", "USD"), ("ntd", "NTD")):
        if prefix.endswith(marker):
            unit = currency
            break
    return unit, finish


@dataclass(frozen=True)
class NumericFact:
    value: Decimal
    unit: str | None
    optional: bool = False


def _numbers(text):
    # This idiom describes continuity, not a count of one year and four seasons.
    text = _normalize(text).replace("一年四季", "常年")
    text = text.replace("一天一天", "逐日").replace("这一天", "当天")
    text = text.replace("二合一", "two in one").replace("立体", "3d").replace("three-dimensional", "3d")
    text = re.sub(r"([一二两三四五六七八九]) +([一二两三四五六七八九])(?=个|天|年|小时)", r"\1\2", text)
    text = re.sub(r"([一二两三四五六七八九])分熟", lambda m: str(_DIGITS[m[1]] * 10) + "%", text)
    text = re.sub(rf"([{_CN}]+|\d+)(块|元|小时|天|米)半", lambda m: str(_zh_decimal(m[1]) + Decimal('0.5')) + m[2], text)
    text = re.sub(r"\b(" + "|".join(_EN_VALUES) + r") and a half\b", lambda m: str(Decimal(_EN_VALUES[m[1]]) + Decimal('0.5')), text)
    text = re.sub(r"几十(?=年|个|米|次)", "数十", text)
    # Adjacent spoken range endpoints share the following scale.
    def english_range(m):
        scale = _EN_SCALE[m[3]]
        return f"{_EN_VALUES[m[1]] * scale} or {_EN_VALUES[m[2]] * scale}"
    text = re.sub(r"\b(" + "|".join(_EN_VALUES) + r") (?:or|to) (" + "|".join(_EN_VALUES) + r") (hundred|thousand|million)\b", english_range, text)
    facts, occupied = [], []

    def add(start, end, value, unit=None, optional=False):
        prefix = text[max(0, start - 12):start]
        if re.search(r"(?:negative|minus)\s*$", prefix):
            value = -abs(value)
        detected, finish = _unit(text, start, end)
        unit = unit or detected
        facts.append(NumericFact(value, unit, optional and unit in {None, "count"}))
        occupied.append((start, finish))

    # A separated digit list is a sequence of facts, not a compound number.
    # Boundaries prevent matching the trailing digits inside 三十五、四十.
    digit_list = rf"(?<![{_CN}])[零〇一二两三四五六七八九](?:[、，, ]+[零〇一二两三四五六七八九])+(?![{_CN}])"
    for match in re.finditer(digit_list, text):
        for digit in re.finditer(r"[零〇一二两三四五六七八九]", match.group()):
            add(match.start() + digit.start(), match.start() + digit.end(), Decimal(_DIGITS[digit.group()]))

    # A written Chinese fraction and its ordinary English wording are equivalent.
    for match in re.finditer(rf"([{_CN}]+)分之([{_CN}]+)", text):
        if match[1] in {"百", "千"}:
            continue  # Percentage/per-mille expressions retain their explicit unit.
        denominator = _zh_integer(match[1])
        if not denominator:
            raise ValueError("Zero denominator requires review.")
        add(match.start(), match.end(), Decimal(_zh_integer(match[2])) / denominator)
    fraction_words = "|".join(_EN_VALUES)
    denominators = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4, "fourth": 4, "fourths": 4}
    for match in re.finditer(r"\b(" + fraction_words + r")[ -]+(" + "|".join(denominators) + r")\b", text):
        add(match.start(), match.end(), Decimal(_EN_VALUES[match[1]]) / denominators[match[2]])
    for match in _NUMBER.finditer(text):
        if any(match.start() < b and match.end() > a for a, b in occupied):
            continue
        number = match["number"]
        if text[max(0, match.start()-1):match.start()] in {"几", "数"}:
            continue  # Vague quantities are outside the exact-numeric guard.
        value = _zh_decimal(number)
        end = match.end()
        scale = match["scale"]
        tail_scale = re.match(r"\s*(thousand|million|billion)(?![a-z])", text[end:])
        if scale:
            value *= _LARGE[scale]
        elif tail_scale:
            value *= _EN_SCALE[tail_scale[1]]
            end += tail_scale.end()
        if (match["leading"] or match["sign"]) in {"-", "负"}:
            value = -value
        unit = {"百分之": "percent", "千分之": "permille"}.get(match["ratio"])
        unit = unit or {"%": "percent", "‰": "permille"}.get(match["symbol"])
        detected, _ = _unit(text, match.start(), end)
        ordinal = text[max(0, match.start()-1):match.start()] == "第"
        if len(number) == 1 and not number.isdigit() and detected is None and unit is None and not ordinal and text.strip() != number and text[end:end+1] not in {"多", "余"}:
            continue  # Bare 一/七 in 一点/七龙珠 are not proven numerical facts.
        # 四五个小时 and 两三百个 describe ranges, not 45 or 2300.
        range_match = re.fullmatch(r"([一二两三四五六七八九])([一二两三四五六七八九])([百千万]?)", number)
        if range_match and detected is not None and not ordinal:
            scale_value = {"": 1, "百": 100, "千": 1000, "万": 10000}[range_match[3]]
            facts.append(NumericFact(Decimal(_DIGITS[range_match[1]] * scale_value), None))
            add(match.start(), end, Decimal(_DIGITS[range_match[2]] * scale_value), unit)
        else:
            dimension = value == 3 and text[end:end+1] == "d"
            add(match.start(), end, value, unit, optional=(value == 1 and not number[0].isdigit()) or dimension)
    for pattern, converter in ((_EN_NUMBER, _english_decimal), (_ORDINAL, lambda w: Decimal(_ORDINALS[w]))):
        for match in pattern.finditer(text):
            if any(match.start() < b and match.end() > a for a, b in occupied):
                continue
            words = match.group().replace("-", " ").split()
            if pattern is _EN_NUMBER and len(words) > 1 and all(w in _EN_VALUES and _EN_VALUES[w] < 10 for w in words):
                for word in re.finditer(r"[a-z]+", match.group()):
                    add(match.start() + word.start(), match.start() + word.end(), Decimal(_EN_VALUES[word.group()]))
                continue
            descriptive_number = bool(re.match(r"-[a-z]", text[match.end():]))
            add(match.start(), match.end(), converter(match.group()), optional=match.group() in {"one", "first", "both", "noon"} or descriptive_number)
    # An explicit singular measurement can be naturally translated as "a year".
    for match in re.finditer(r"(?<![a-z])(?:a|an)\s+(?:year|day|hour|minute|second|meter|metre)(?![a-z])", text):
        if not any(match.start() < b and match.end() > a for a, b in occupied):
            _, unit_word = match.group().split()
            facts.append(NumericFact(Decimal(1), _UNITS[unit_word]))
    return facts


def _compatible(source, candidate):
    if source.value != candidate.value:
        return False
    if source.unit in {None, "count"}:
        return True
    if source.unit in _MONEY and candidate.unit in _MONEY:
        return source.unit == candidate.unit or "money" in {source.unit, candidate.unit}
    return source.unit == candidate.unit


def _negations(text):
    result = []
    for m in _NEGATION.finditer(_normalize(text)):
        word = m.group()
        result.append("cannot" if word in {"不能", "无法", "没法", "不可以", "不得", "不许", "不准", "cannot", "can't"}
                      else "need_not" if word in {"不必", "不用", "无需"} else "not")
    return Counter(result)


def _comparisons(text):
    text = _normalize(text)
    signatures = []
    for m in _COMPARISON.finditer(text):
        parts = re.split(rf"(还要|还|更|{_COMPARISON_OPERATORS}|[，。！？\n])", text[m.end():], maxsplit=1)
        target = "".join(c for c in parts[0] if c.isalnum() and c != "的")
        remainder = "".join(parts[1:])
        operator = re.match(rf"(?:还要|还|更)*(?:不)?({_COMPARISON_OPERATORS})", remainder)
        direction = operator[1] if operator else ""
        signatures.append((target, direction))
    return signatures


def check_meaning_preservation(source_cues, candidate_cues, *, mode):
    """Raise on supported fact changes, before display formatting.

    Cue pairs are (integer ID, body text). Translation checks recognized numeric
    values and explicit units per cue. Correction additionally protects negation,
    approximation and literal Chinese comparison targets. This is deliberately
    conservative; genuine uncertain ASR repairs may require human review.
    """
    if mode not in {"correction", "translation"}:
        raise ValueError("Unsupported meaning guard mode.")
    if [c[0] for c in source_cues] != [c[0] for c in candidate_cues]:
        raise ValueError("Meaning guard requires corresponding source cue IDs.")
    issues = []
    for (cue_id, source), (_, candidate) in zip(source_cues, candidate_cues):
        if not isinstance(source, str) or not isinstance(candidate, str):
            raise ValueError("Meaning guard requires text bodies.")

        def flag(code):
            issues.append({"cue_id": cue_id, "code": code})

        try:
            before, after = _numbers(source), _numbers(candidate)
        except (ValueError, ArithmeticError):
            flag("unsupported_numeric_expression")
            continue
        remaining = list(after)
        for fact in sorted(before, key=lambda f: f.unit in {None, "count"}):
            match = next((i for i, value in enumerate(remaining) if _compatible(fact, value)), None)
            if match is not None:
                remaining.pop(match)
            elif not (mode == "translation" and fact.optional):
                flag("numeric_fact_missing_or_changed")
        if any(not (mode == "translation" and f.optional) for f in remaining):
            flag("numeric_fact_added_or_moved")
        if mode == "correction":
            if _negations(source) != _negations(candidate):
                flag("negation_changed")
            if _comparisons(source) != _comparisons(candidate):
                flag("comparison_target_changed")
            if before and len(_APPROX.findall(_normalize(source))) != len(_APPROX.findall(_normalize(candidate))):
                flag("approximation_changed")
    if issues:
        raise MeaningPreservationError(issues)
    return meaning_guard_scope(mode)


def meaning_guard_scope(mode):
    return {"version": MEANING_GUARD_VERSION, "content_verified": False,
            "checks": ["recognized_numeric_values_and_units_per_cue"] +
                      (["negation", "approximation", "literal_chinese_comparison_target"] if mode == "correction" else []),
            "limits": ["No audio verification or general semantic certification",
                       "Vague quantities and contextual/idiomatic number meanings are not fully assessed",
                       "Conservative checks can require review of legitimate paraphrases"]}
