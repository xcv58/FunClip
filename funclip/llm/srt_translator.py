import os
import logging
from litellm import completion
from dotenv import load_dotenv

load_dotenv()


def translate_srt_to_english(srt_content, api_key=None, base_url=None, model="gpt-5-mini"):
    """
    Translate SRT subtitles from Simplified Chinese to English using an LLM.

    Args:
        srt_content (str): The SRT content in Simplified Chinese.
        api_key (str, optional): API key for the LLM provider.
        base_url (str, optional): Base URL for the LLM provider.
        model (str, optional): Model to use.

    Returns:
        str: The translated SRT content in English.
    """

    system_prompt = (
        "You are an expert subtitle translator specializing in Chinese-to-English translation. "
        "Translate the provided SRT subtitles from Simplified Chinese to natural, fluent English.\n\n"
        "Rules:\n"
        "1. The output MUST have EXACTLY the same number of subtitle segments as the input. "
        "Each segment index (1, 2, 3, …) in the input must appear exactly once in the output, in the same order. "
        "Do NOT merge, split, skip, or add any segments. "
        "If the output risks being too long, use shorter and more concise translations rather than dropping segments.\n"
        "2. Translate each subtitle segment's text from Chinese to English.\n"
        "3. Keep translations CONCISE — each segment will be displayed as video subtitles. "
        "Aim for roughly the same duration feel: short segments stay short, longer ones can be slightly longer, "
        "but never exceed ~12 words per line. Split into two lines if needed.\n"
        "4. Do NOT change timestamps or subtitle index numbers.\n"
        "5. Preserve the original meaning and tone. Prefer clear, natural English over literal translation.\n"
        "6. For proper nouns, brand names, or technical terms already in English, keep them as-is.\n"
        "7. Output ONLY the translated SRT content. No markdown formatting (like ```srt), no comments, no explanations.\n"
        "8. Maintain natural spoken English rhythm suitable for on-screen reading."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": srt_content}
    ]

    try:
        logging.info(f"Sending SRT translation request to LLM (Model: {model})")

        kwargs = {
            "model": model,
            "messages": messages,
        }

        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        response = completion(**kwargs)

        translated_content = response.choices[0].message.content
        return translated_content.strip()

    except Exception as e:
        logging.error(f"Error during SRT translation: {e}")
        raise e
