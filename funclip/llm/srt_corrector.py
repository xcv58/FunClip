import os
import logging
from litellm import completion
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

def correct_srt_content(srt_content, api_key=None, base_url=None, model="gpt-4o-mini"):
    """
    Corrects typos and mixed language errors in SRT content using an LLM.
    
    Args:
        srt_content (str): The raw SRT content to correct.
        api_key (str, optional): API key for the LLM provider. Defaults to None (uses env var).
        base_url (str, optional): Base URL for the LLM provider. Defaults to None.
        model (str, optional): Model to use. Defaults to "gpt-3.5-turbo".
        
    Returns:
        str: The corrected SRT content.
    """
    
    # Construct the prompt
    system_prompt = (
        "You are a professional subtitle editor. Your task is to correct typos and fix mixed language errors "
        "in the provided SRT subtitles. \n"
        "Rules:\n"
        "1. Fix obvious typos and character recognition errors.\n"
        "2. Correct location names, famous people's names, and specific terminologies/conventions that may have been phonetic misinterpretations (e.g., correcting 'Fun Clip' to 'FunClip' if appropriate context, or correcting city names).\n"
        "3. Fix mixed language issues (e.g., if a sentence is primarily Chinese but contains random English words "
        "that are likely recognition errors, correct them to Chinese. If the English is intentional/technical, preserve it).\n"
        "4. Do NOT translate the entire text. Keep the original language structure.\n"
        "5. Do NOT change the timestamps or the subtitle index numbers at all.\n"
        "6. Output ONLY the corrected SRT content. Do NOT include any markdown formatting (like ```srt), "
        "comments, or explanations."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": srt_content}
    ]
    
    try:
        logging.info(f"Sending SRT correction request to LLM (Model: {model})")
        
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
