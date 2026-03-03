"""
Chinese character conversion utility using OpenCC.
Converts Simplified Chinese to Traditional Chinese.
"""

from opencc import OpenCC

# Keep selected terms in common modern usage after OpenCC conversion.
POST_CONVERSION_OVERRIDES = {
    "喫": "吃",
    "鬱": "郁",
}


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
    
    # s2t = Simplified to Traditional
    converter = OpenCC('s2t')
    converted_text = converter.convert(text)

    for source, target in POST_CONVERSION_OVERRIDES.items():
        converted_text = converted_text.replace(source, target)

    return converted_text
