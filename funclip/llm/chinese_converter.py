"""
Chinese character conversion utility using OpenCC.
Converts Simplified Chinese to Traditional Chinese.
"""

from opencc import OpenCC

POST_CONVERSION_OVERRIDES = {
    "喫": "吃",
    "鬱": "郁",
}
S2T_CONVERTER = OpenCC('s2t')

# These characters are shared Traditional variants in ordinary Taiwanese or
# proper-name usage even though OpenCC's one-way s2t mapping prefers another
# glyph. Keeping them here avoids treating conversion preference as script
# identity (for example 台灣, 吃飯, and 郁達夫).
TRADITIONAL_SHARED_VARIANT_PAIRS = frozenset({
    ("台", "臺"),
    ("吃", "喫"),
})
TRADITIONAL_PROTECTED_TERMS = frozenset({
    "恒生銀行",
    "郁達夫",
    "馥郁",
    "濃郁",
    "郁郁",
})


def convert_to_traditional_script(text: str) -> str:
    """Convert Simplified forms without applying stylistic display overrides."""
    if not text:
        return text
    return S2T_CONVERTER.convert(text)


def normalize_traditional_for_validation(text: str) -> str:
    """Phrase-convert Simplified forms while preserving accepted shared variants."""
    if not text:
        return text
    converted = S2T_CONVERTER.convert(text)
    if len(converted) != len(text):
        return converted
    normalized = list(converted)
    protected_positions = set()
    for term in TRADITIONAL_PROTECTED_TERMS:
        start = text.find(term)
        while start >= 0:
            protected_positions.update(range(start, start + len(term)))
            start = text.find(term, start + 1)
    for position, (original, replacement) in enumerate(zip(text, converted)):
        if (
            (original, replacement) in TRADITIONAL_SHARED_VARIANT_PAIRS
            or position in protected_positions
        ):
            normalized[position] = original
    return "".join(normalized)


def canonicalize_traditional_for_comparison(text: str) -> str:
    """Canonicalize accepted Traditional variants only for equality checks."""
    if not text:
        return text
    return S2T_CONVERTER.convert(text)


def convert_to_traditional(text: str) -> str:
    """
    Convert Simplified Chinese text to Traditional Chinese.
    
    Args:
        text: Input text in Simplified Chinese
        
    Returns:
        Text converted to Traditional Chinese
    """
    if not text:
        return text

    converted_text = convert_to_traditional_script(text)
    for source, target in POST_CONVERSION_OVERRIDES.items():
        converted_text = converted_text.replace(source, target)
    return converted_text
