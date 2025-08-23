from src.utils import config
from src.llm.openai_handler import OpenAIHandler
from src.llm.gemini_handler import GeminiHandler

def LLMHandler():
    """
    Factory function that returns the appropriate LLM handler based on the provider
    specified in the configuration.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == 'openai':
        return OpenAIHandler()
    elif provider == 'gemini':
        return GeminiHandler()
    else:
        raise ValueError(
            f"Invalid LLM_PROVIDER: '{provider}'. "
            "Please choose 'openai' or 'gemini' in your .env file."
        )