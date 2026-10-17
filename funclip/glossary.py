"""Request-scoped terminology; decoding hints never rewrite recognition evidence."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unicodedata


CATALOG = Path(__file__).with_name("glossaries")


class GlossaryError(ValueError):
    """Invalid glossary or terminology requiring source review."""


@dataclass(frozen=True)
class GlossaryEntry:
    id: str
    zh_hans: str
    zh_hant: str
    english: str
    evidence: str
    spoken_forms: tuple = ()
    spoken_aliases: tuple = ()
    menu_name: str = ""
    asr_hotwords: tuple = ()
    asr_confusions: tuple = ()
    english_variants: tuple = ()

    @property
    def forms(self):
        return tuple(dict.fromkeys(x for x in (self.zh_hans, self.zh_hant, self.menu_name, *self.spoken_forms, *self.spoken_aliases) if x))


@dataclass(frozen=True)
class Glossary:
    id: str
    label: str
    entries: tuple
    sha256: str

    def metadata(self):
        return {"id": self.id, "sha256": self.sha256, "entry_count": len(self.entries)}


def _text(value, label, *, limit=240, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()) or len(value) > limit:
        raise GlossaryError(f"Invalid glossary {label}.")
    if any(unicodedata.category(c).startswith("C") or c in "\r\n" for c in value):
        raise GlossaryError(f"Invalid glossary {label} control character.")
    return value.strip()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value):
        raise GlossaryError("Invalid glossary identifier.")
    return value


def parse_glossary(data):
    if not isinstance(data, dict) or set(data) != {"version", "id", "label", "entries"} or type(data["version"]) is not int or data["version"] != 1:
        raise GlossaryError("Expected a version 1 glossary with id label and entries.")
    if not isinstance(data["entries"], list) or not 1 <= len(data["entries"]) <= 48:
        raise GlossaryError("A glossary must have between 1 and 48 entries.")
    entries, seen = [], set()
    required = {"id", "zh_hans", "zh_hant", "english", "evidence"}
    optional = {"spoken_forms", "spoken_aliases", "menu_name", "asr_hotwords", "asr_confusions", "english_variants"}
    for raw in data["entries"]:
        if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - required - optional:
            raise GlossaryError("Invalid glossary entry fields.")
        entry_id = _identifier(raw["id"])
        if entry_id in seen:
            raise GlossaryError("Duplicate glossary entry identifier.")
        seen.add(entry_id)
        fields = {key: _text(raw[key], key) for key in required - {"id"}}
        fields["menu_name"] = _text(raw.get("menu_name", ""), "menu_name", empty=True)
        for key in optional - {"menu_name"}:
            values = raw.get(key, [])
            if not isinstance(values, list) or len(values) > 12:
                raise GlossaryError(f"Invalid glossary {key} list.")
            fields[key] = tuple(dict.fromkeys(_text(v, key, limit=100) for v in values))
        from funclip.subtitle_style import format_subtitle_text
        for phrase in (fields["english"], *fields["english_variants"]):
            if format_subtitle_text(phrase, language="en") != phrase:
                raise GlossaryError("Glossary English wording must already be punctuation free.")
        entries.append(GlossaryEntry(id=entry_id, **fields))
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return Glossary(_identifier(data["id"]), _text(data["label"], "label", limit=80), tuple(entries), hashlib.sha256(canonical.encode()).hexdigest())


@lru_cache(maxsize=16)
def _load_named(name):
    name = _identifier(name)
    path = CATALOG / (name + ".json")
    if not path.is_file():
        raise GlossaryError("Unknown glossary profile.")
    return parse_glossary(json.loads(path.read_text(encoding="utf-8")))


def resolve_glossary(value=None):
    if value is None or (isinstance(value, str) and value in {"", "none"}):
        return None
    if isinstance(value, Glossary):
        return value
    return parse_glossary(value) if isinstance(value, dict) else _load_named(value)


def glossary_choices():
    return [("None", "none")] + [(_load_named(p.stem).label, p.stem) for p in sorted(CATALOG.glob("*.json"))]


GLOSSARY_MODES = ("reference", "hints")


def validate_glossary_mode(mode):
    if not isinstance(mode, str) or mode not in GLOSSARY_MODES:
        raise GlossaryError("Choose reference or hints for glossary use.")
    return mode


def _contains(text, phrase):
    if re.search(r"[A-Za-z]", phrase):
        return bool(re.search(r"(?<![A-Za-z])" + re.escape(phrase) + r"(?![A-Za-z])", text, re.I))
    return phrase in text


def _matches(text, entry):
    return (any(_contains(text, form) for form in entry.forms)
            or any(_contains(_alias_normalizer().convert(text), _alias_normalizer().convert(form))
                   for form in entry.spoken_aliases))


def glossary_context(cues, glossary):
    profile = resolve_glossary(glossary)
    if profile is None:
        return []
    text = "\n".join(body for _, body in cues)
    return [{"term_id": e.id, "zh_hans": e.zh_hans, "zh_hant": e.zh_hant,
             "spoken_forms": list(e.spoken_forms), "spoken_aliases": list(e.spoken_aliases), "menu_name": e.menu_name,
             "english": e.english, "english_variants": list(e.english_variants)}
            for e in profile.entries if _matches(text, e)]


def glossary_report(cues, glossary, *, english=False):
    profile = resolve_glossary(glossary)
    if profile is None:
        return {"profile": None, "issues": [], "content_verified": False}
    issues = []
    for cue_id, text in cues:
        for e in profile.entries:
            for confusion in e.asr_confusions:
                if _contains(text, confusion):
                    issues.append({"code": "possible_asr_confusion", "cue_id": cue_id, "observed": confusion,
                                   "candidate": e.zh_hans, "term_id": e.id, "action": "Verify the audio before changing this name"})
        if english and any(_is_cjk(c) for c in text):
            issues.append({"code": "untranslated_chinese_text", "cue_id": cue_id, "action": "Review the English name including rare Chinese characters"})
    return {"profile": profile.metadata(), "issues": issues, "content_verified": False}


def _is_cjk(char):
    return any(a <= ord(char) <= b for a, b in ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x323AF)))


def validate_glossary_correction(source_cues, candidate_cues, glossary):
    profile = resolve_glossary(glossary)
    if profile is None:
        return
    for (cue_id, source), (_, target) in zip(source_cues, candidate_cues):
        for e in profile.entries:
            for alias in e.spoken_aliases:
                normalizer = _alias_normalizer()
                if _contains(normalizer.convert(source), normalizer.convert(alias)) != _contains(normalizer.convert(target), normalizer.convert(alias)):
                    raise GlossaryError(f"Cue {cue_id} changed distinct spoken alias {e.id}; retain the words that were spoken.")
            if e.menu_name and e.menu_name not in (e.zh_hans, e.zh_hant, *e.spoken_forms):
                if _contains(source, e.menu_name) != _contains(target, e.menu_name):
                    raise GlossaryError(f"Cue {cue_id} substituted a printed menu name for the spoken form {e.id}.")
            if _matches(source, e) and not _matches(target, e):
                raise GlossaryError(f"Cue {cue_id} removed confirmed glossary term {e.id}.")
            if not _matches(source, e) and _matches(target, e):
                raise GlossaryError(f"Cue {cue_id} added glossary term {e.id}; verify audio rather than replacing a suspected homophone.")


@lru_cache(maxsize=1)
def _alias_normalizer():
    from opencc import OpenCC
    return OpenCC('t2s')


def validate_glossary_translation(source_cues, candidate_cues, glossary):
    profile = resolve_glossary(glossary)
    if profile is None:
        return
    def english_form(text):
        return " " + re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip() + " "
    for (cue_id, source), (_, target) in zip(source_cues, candidate_cues):
        for e in profile.entries:
            present = any(english_form(p) in english_form(target) for p in (e.english, *e.english_variants))
            if _matches(source, e) != present:
                raise GlossaryError(f"Cue {cue_id} changed glossary term {e.id}; use its confirmed English wording only in its source cue.")


@contextmanager
def recognition_hotwords(model, glossary, legacy_hotwords="", *, mode="hints"):
    """Use the installed model's exact parser; never fetch a user-supplied URL."""
    validate_glossary_mode(mode)
    profile = resolve_glossary(glossary)
    if profile is None:
        yield legacy_hotwords, None
        return
    if legacy_hotwords:
        raise GlossaryError("Choose a glossary or legacy hotwords, not both.")
    if mode == "reference":
        yield "", {"profile": profile.metadata(), "mode": mode,
                   "active_hotwords": [], "unsupported_hotwords": [],
                   "hotwords_checked": False,
                   "policy": "Reference for correction and translation only; no ASR bias"}
        return
    kwargs = getattr(model, "kwargs", {})
    parser = getattr(getattr(model, "model", None), "generate_hotwords_list", None)
    tokens = kwargs.get("token_list") if isinstance(kwargs, dict) else None
    if not callable(parser) or not tokens or "<unk>" not in tokens:
        raise GlossaryError("The current model cannot validate glossary hotwords.")
    unknown_id = list(tokens).index("<unk>")
    accepted, rejected = [], []
    with tempfile.TemporaryDirectory(prefix="funclip_hotwords_") as folder:
        path = Path(folder) / "hotwords.txt"
        for word in dict.fromkeys(w for e in profile.entries for w in e.asr_hotwords):
            path.write_text(word + "\n", encoding="utf-8")
            ids = parser(str(path), tokenizer=kwargs["tokenizer"], frontend=kwargs["frontend"])
            if not ids or not ids[0] or unknown_id in ids[0]:
                rejected.append(word)
            else:
                accepted.append(word)
        path.write_text("\n".join(accepted) + "\n", encoding="utf-8")
        yield str(path) if accepted else "", {"profile": profile.metadata(), "mode": mode,
                "hotwords_checked": True, "active_hotwords": accepted,
                "unsupported_hotwords": rejected, "policy": "Decoding bias only; unsupported spellings require review"}
