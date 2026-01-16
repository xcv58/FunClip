"""
Chinese character conversion utility using OpenCC.
Converts Simplified Chinese to Traditional Chinese.
"""

from opencc import OpenCC


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
    return converter.convert(text)
