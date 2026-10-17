"""Conservative local content anchors and cue provenance, never semantic proof."""

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib
from html import escape
import json
import re
import unicodedata


ALIGNMENT_VERSION = 1
MAX_ALIGNMENT_CUES = 3000
MAX_ALIGNMENT_CHARACTERS = 100_000
# Deliberately bounded vocabulary. Missing an anchor is advisory because valid
# translations may use pronouns, idioms, synonyms, or a different word order.
# Only a paired relocation or an explicit buy/sell reversal blocks publication.
_ANCHORS = [
    ('taiwan', 'place', r'台湾', r'\btaiwan(?:ese)?\b'),
    ('taipei', 'place', r'台北', r'\btaipei\b'),
    ('penghu', 'place', r'澎湖', r'\b(?:penghu|pescadores)\b'),
    ('singapore', 'place', r'新加坡', r'\bsingapore(?:an)?\b'),
    ('malaysia', 'place', r'马来西亚', r'\bmalaysia(?:n)?\b'),
    ('penang', 'place', r'槟城', r'\bpenang\b'),
    ('usa', 'place', r'美国', r'\b(?:america(?:n)?|united states|usa|u\.s\.?|u s)\b'),
    ('new-york', 'place', r'纽约', r'\bnew york\b'),
    ('china', 'place', r'中国(?:大陆)?', r'\b(?:china|mainland china)\b'),
    ('japan', 'place', r'日本', r'\bjapan(?:ese)?\b'),
    ('thailand', 'place', r'泰国', r'\b(?:thailand|thai)\b'),
    ('fujian', 'place', r'福建', r'\bfujian\b'),
    ('hong-kong', 'place', r'香港', r'\bhong kong\b'),
    ('airport', 'place-type', r'机场', r'\bairports?\b'),
    ('hotel', 'place-type', r'酒店|旅馆', r'\b(?:hotels?|inns?)\b'),
    ('guesthouse', 'place-type', r'民宿', r'\b(?:guest\s*houses?|homestays?)\b'),
    ('museum', 'place-type', r'博物馆', r'\b(?:museums?|heritage houses?)\b'),
    ('temple', 'place-type', r'寺庙|庙宇|寺院', r'\btemples?\b'),
    ('night-market', 'place-type', r'夜市', r'\bnight markets?\b'),
    ('beach', 'place-type', r'海滩|沙滩', r'\bbeaches?\b'),
    ('bridge', 'object', r'桥', r'\bbridges?\b'),
    ('airplane', 'object', r'飞机', r'\b(?:airplanes?|aeroplanes?|aircraft|planes?)\b'),
    ('train', 'object', r'火车', r'\btrains?\b'),
    ('metro', 'object', r'捷运|地铁', r'\b(?:metro|subway|mrt)\b'),
    ('ferry', 'object', r'轮渡|渡轮', r'\bferr(?:y|ies)\b'),
    ('bus', 'object', r'公车|巴士|公交车', r'\b(?:bus|buses|shuttles?)\b'),
    ('taxi', 'object', r'计程车|出租车', r'\b(?:taxis?|cabs?)\b'),
    ('luggage', 'object', r'行李', r'\b(?:luggage|baggage|suitcases?)\b'),
    ('coffee', 'object', r'咖啡', r'\bcoffee\b'),
    ('durian', 'object', r'榴莲', r'\bdurians?\b'),
    ('shoes', 'object', r'鞋|拖鞋', r'\b(?:shoes?|slippers?|footwear)\b'),
    ('clothes', 'object', r'衣服|服装|服饰|裙子', r'\b(?:clothes|clothing|costumes?|outfits?|dresses?|skirts?|attire)\b'),
    ('buy', 'action', r'购买|买(?:了|过|到|票|衣|东西)', r'\b(?:buy|buys|buying|bought|purchas(?:e|es|ed|ing))\b'),
    ('sell', 'action', r'出售|卖(?:了|过|东西|咖啡|饮料)', r'\b(?:sell|sells|selling|sold)\b'),
    ('eat', 'action', r'品尝|尝过|尝了|吃(?:了|过|到|饭|完)', r'\b(?:eat|eats|eating|ate|eaten|tast(?:e|es|ed|ing)|tri(?:ed|es)|try|trying)\b|\bhad (?:a |some |the )?(?:meal|lunch|dinner|breakfast|food)\b'),
    ('drink', 'action', r'喝(?:了|过|到|咖啡|水|茶)', r'\b(?:drink|drinks|drinking|drank|drunk|sip|sips|sipped|sipping)\b'),
    ('pay', 'action', r'付钱|付款|缴费|支付', r'\b(?:pay|pays|paying|paid|payment)\b'),
    ('take-photo', 'action', r'拍照|拍摄|拍下', r'\b(?:tak(?:e|es|ing)|took|shoot|shooting|shot)\b.{0,30}\b(?:photos?|pictures?)\b|\bphotograph(?:ed|ing|s)?\b'),
    ('arrive', 'action', r'到达|抵达', r'\b(?:arriv(?:e|es|ed|ing)|reach(?:ed|es|ing)?)\b'),
    ('leave', 'action', r'离开|出发', r'\b(?:leav(?:e|es|ing)|left|depart(?:ed|s|ing)?)\b'),
    ('rain', 'event', r'下雨|雨天', r'\brain(?:s|ed|ing|y)?\b'),
    ('mother', 'person', r'妈妈|母亲', r'\b(?:mother|mom|mum)\b'),
    ('father', 'person', r'爸爸|父亲', r'\b(?:father|dad)\b'),
    ('grandmother', 'person', r'奶奶|外婆|祖母', r'\b(?:grandmother|grandma|granny)\b'),
]
_COMPILED = [(key, kind, re.compile(zh), re.compile(en, re.I)) for key, kind, zh, en in _ANCHORS]
ANCHOR_CATALOG_SHA256 = hashlib.sha256(json.dumps(_ANCHORS, ensure_ascii=False).encode()).hexdigest()


class MeaningAlignmentError(ValueError):
    def __init__(self, report):
        self.report = report
        issues = [i for i in report['issues'] if i['severity'] == 'error']
        super().__init__('Meaning alignment requires review: ' + ', '.join(
            f"cue {i['cue_id']} {i['code']}" for i in issues[:8]))


@lru_cache(maxsize=1)
def _converter():
    from opencc import OpenCC
    return OpenCC('t2s')


def _normalize(text):
    return _converter().convert(unicodedata.normalize('NFKC', text)).casefold()


def _plain(text):
    return ''.join(c for c in _normalize(text) if c.isalnum())


def _anchors(text):
    raw = unicodedata.normalize('NFKC', text)
    text = _normalize(text)
    # A currency's country adjective is not a claim that an event happened there.
    text = re.sub(r'\b(?:u\.s\.?|us|united states|taiwan(?:ese)?|singapore)[\s-]+(?:dollars?|currency|cents?)\b',
                  'currency', text)
    text = re.sub(r'(?:新加坡|台湾)(?:元|币)', '货币', text)
    values = {key: kind for key, kind, zh, en in _COMPILED if zh.search(text) or en.search(text)}
    # Preserve case for this abbreviation: lowercase "us" is an English pronoun.
    if re.search(r'\bUS\b(?![\s-]+(?i:dollars?|currency|cents?))', raw):
        values['usa'] = 'place'
    return values


def meaning_alignment_report(source_cues, candidate_cues, *, mode, source_times=None):
    """Analyze exact cue pairs; no lexical match or empty report certifies meaning.

    Caller owns structure validation. A coverage row binds text hashes to its
    source cue/time, explicitly marked as structural rather than semantic proof.
    """
    if mode not in {'translation', 'correction'}:
        raise ValueError('Choose translation or correction alignment mode')
    source, target = list(source_cues), list(candidate_cues)
    if (len(source) > MAX_ALIGNMENT_CUES or sum(len(t) for _, t in source + target) > MAX_ALIGNMENT_CHARACTERS * 2
            or [i for i, _ in source] != [i for i, _ in target]
            or len({i for i, _ in source}) != len(source)):
        raise ValueError('Alignment requires bounded corresponding unique cue IDs')
    left, right = {i: _anchors(t) for i, t in source}, {i: _anchors(t) for i, t in target}
    source_counts = Counter(k for row in left.values() for k in row)
    target_counts = Counter(k for row in right.values() for k in row)
    missing, extra = defaultdict(list), defaultdict(list)
    for cue_id, _ in source:
        for anchor in left[cue_id].keys() - right[cue_id].keys():
            missing[anchor].append(cue_id)
        for anchor in right[cue_id].keys() - left[cue_id].keys():
            extra[anchor].append(cue_id)
    issues, coverage = [], []
    times = source_times or {}
    source_plain = {i: _plain(t) for i, t in source} if mode == 'correction' else {}
    target_plain = {i: _plain(t) for i, t in target} if mode == 'correction' else {}

    def add(code, cue_id, *, severity='review', **details):
        a, b = times.get(cue_id, (None, None))
        issues.append({'code': code, 'cue_id': cue_id, 'source_cue_ids': [cue_id],
                       'start_ms': a, 'end_ms': b, 'severity': severity, **details})

    for (cue_id, before), (_, after) in zip(source, target):
        a, b = times.get(cue_id, (None, None))
        coverage.append({'output_cue_id': cue_id, 'source_cue_ids': [cue_id],
                         'start_ms': a, 'end_ms': b, 'source_text_sha256': hashlib.sha256(before.encode()).hexdigest(),
                         'output_text_sha256': hashlib.sha256(after.encode()).hexdigest(),
                         'supported_source_anchors': sorted(left[cue_id]),
                         'matched_anchors': sorted(left[cue_id].keys() & right[cue_id].keys()),
                         'semantic_coverage_verified': False})
        for anchor in sorted(left[cue_id].keys() - right[cue_id].keys()):
            recipients = extra.get(anchor, [])
            # Require paired loss/addition with a unique donor and recipient.
            # Repeated references and ambiguous wording only receive review.
            moved = (len(missing[anchor]) == len(recipients) == 1
                     and source_counts[anchor] == target_counts[anchor] == 1)
            add('anchor_moved_between_cues' if moved else 'source_anchor_missing', cue_id,
                severity='error' if moved else 'review', anchor=anchor, kind=left[cue_id][anchor],
                candidate_cue_ids=recipients, reason='Check where this source meaning appears in the output')
        for anchor in sorted(right[cue_id].keys() - left[cue_id].keys()):
            if not missing.get(anchor):
                add('output_anchor_needs_review', cue_id, anchor=anchor, kind=right[cue_id][anchor],
                    reason='Output introduces a supported concept absent from this source cue')
        if (set(left[cue_id]) & {'buy', 'sell'} in ({'buy'}, {'sell'})
                and set(right[cue_id]) & {'buy', 'sell'} in ({'buy'}, {'sell'})
                and (set(left[cue_id]) & {'buy', 'sell'}) != (set(right[cue_id]) & {'buy', 'sell'})):
            add('action_reversed', cue_id, severity='error', reason='Buying and selling were exchanged')
        if mode == 'correction':
            first, second = source_plain[cue_id], target_plain[cue_id]
            for tag, a1, b1, a2, b2 in SequenceMatcher(a=first, b=second, autojunk=False).get_opcodes():
                removed = first[a1:b1]
                if tag == 'equal' or len(removed) < 4:
                    continue
                recipients = [other_id for other_id, text in target_plain.items() if other_id != cue_id
                              and removed in text and removed not in source_plain[other_id]]
                donors = [other_id for other_id, text in source_plain.items() if removed in text]
                moved = len(recipients) == len(donors) == 1
                add('literal_content_moved' if moved else 'correction_content_removed', cue_id,
                    severity='error' if moved else 'review', original_fragment=before,
                    candidate_cue_ids=recipients, reason='Review removed or moved source wording')
    return {'version': ALIGNMENT_VERSION, 'normalization_revision': 2, 'mode': mode, 'content_verified': False,
            'anchor_catalog_sha256': ANCHOR_CATALOG_SHA256, 'issues': issues,
            'issue_counts': dict(Counter(i['code'] for i in issues)), 'coverage': coverage,
            'status': 'blocked' if any(i['severity'] == 'error' for i in issues) else 'needs_review' if issues else 'no_supported_discrepancy',
            'limits': ['Vocabulary is bounded and lexical matches are not a semantic judgment',
                       'Missing anchors can be legitimate paraphrases and require review',
                       'Coverage rows bind source text and time but do not prove all meaning was preserved',
                       'No regrouping or automatic source replacement is authorized by this report']}


def validate_meaning_alignment(source_cues, candidate_cues, *, mode):
    report = meaning_alignment_report(source_cues, candidate_cues, mode=mode)
    if report['status'] == 'blocked':
        raise MeaningAlignmentError(report)
    return report


def alignment_prompt_context(cues):
    """Locally extracted reference concepts, not provider-reported coverage."""
    return [{'cue_id': cue_id, 'concepts': sorted(values)} for cue_id, text in cues
            if (values := _anchors(text))]


def srt_alignment_report(source_srt, candidate_srt, *, mode):
    from funclip.llm.srt_translator import parse_translation_srt
    before, after = parse_translation_srt(source_srt), parse_translation_srt(candidate_srt)
    if [(c.index_line, c.timing_line) for c in before] != [(c.index_line, c.timing_line) for c in after]:
        raise ValueError('Alignment requires unchanged source IDs and times')
    return meaning_alignment_report([(c.index, c.text) for c in before], [(c.index, c.text) for c in after],
        mode=mode, source_times={c.index: (c.start_ms, c.end_ms) for c in before})


def alignment_status_suffix(reports):
    issues = [i for report in reports for i in report.get('issues', [])]
    if not issues:
        return ''
    return f" Meaning alignment needs review in {len({i['cue_id'] for i in issues})} cue(s)" + ''.join(
        f" cue {i['cue_id']} {i['code']}" for i in issues[:3])


def alignment_html(report):
    if not report['issues']:
        return ''
    body = ''.join(f"<li>Cue {escape(str(i['cue_id']))} {escape(i['code'])}</li>" for i in report['issues'][:30])
    return '<section><h3>Meaning alignment review</h3><p>These are review candidates and do not certify content</p><ul>' + body + '</ul></section>'
