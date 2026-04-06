import os
import logging
from litellm import completion
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


def build_correction_prompt():
    """Build one unified correction prompt for all SRT correction scenarios."""
    return (
        "You are a professional subtitle editor. This is ERROR CORRECTION, not rewriting.\n"
        "Correct only clear errors while strictly preserving SRT structure and original wording as much as possible.\n\n"
        "Global rules:\n"
        "1. Output MUST be valid SRT content only. No markdown, no comments, no explanations.\n"
        "2. Keep EXACTLY the same number of subtitle segments as input.\n"
        "3. Keep subtitle index numbers exactly unchanged and in the same order.\n"
        "4. Keep ALL timestamps exactly unchanged.\n"
        "5. Preserve blank-line boundaries between subtitle segments. Do not merge/split segments.\n"
        "6. Confidence threshold policy:\n"
        "- High confidence: fix the error.\n"
        "- Medium or low confidence: keep the original text unchanged.\n"
        "7. Minimal edit policy: prefer minimal localized edits only.\n"
        "8. Do NOT shorten, paraphrase, or rewrite for style/readability. Only correct errors.\n"
        "9. Do NOT translate. Keep the original language/script intent of each subtitle line.\n"
        "10. Use spaces instead of commas where appropriate for better readability "
        "(e.g., between clauses or phrases where a pause is natural but a comma feels too heavy).\n"
        "11. Fix obvious typos, ASR recognition errors, punctuation issues, and minor grammar mistakes where confidence is high.\n"
        "12. Correct names/terms only when context is clear; preserve proper nouns and technical terms if uncertain.\n"
        "13. Fix obvious mixed-language artifacts/noise, but keep intentional foreign terms and brand names.\n"
        "14. If in doubt, preserve the original text exactly."
    )


def correct_srt_content(srt_content, api_key=None, base_url=None, model="gpt-4o-mini"):
    """
    Corrects typos and mixed language errors in SRT content using an LLM.
    
    Args:
        srt_content (str): The raw SRT content to correct.
        api_key (str, optional): API key for the LLM provider. Defaults to None (uses env var).
        base_url (str, optional): Base URL for the LLM provider. Defaults to None.
        model (str, optional): Model to use. Defaults to "gpt-4o-mini".
        
    Returns:
        str: The corrected SRT content.
    """
    system_prompt = build_correction_prompt()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": srt_content}
    ]
    
    try:
        logging.info(
            "Sending SRT correction request to LLM "
            f"(Model: {model})"
        )
        
        # litellm handles reading api_key from os.environ if not passed explicitly,
        # but if we pass it explicitly it uses that.
        kwargs = {
            "model": model,
            "messages": messages,
        }
        
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
            
        response = completion(**kwargs)
        
        corrected_content = response.choices[0].message.content
        return corrected_content.strip()
        
    except Exception as e:
        logging.error(f"Error during SRT correction: {e}")
        raise e
