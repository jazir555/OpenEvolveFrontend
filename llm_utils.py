
import requests
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple

# Optional imports with fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


async def initialize_llm_client(
    api_config: Dict[str, Any],
    default_models: Dict[str, str],
    logger: Optional[logging.Logger] = None,
    verbose_output: bool = False
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Initialize an LLM client based on available API keys.
    
    Args:
        api_config: Configuration dict with API keys
        default_models: Dict of default models for each provider
        logger: Optional logger for output
        verbose_output: Whether to print verbose messages
        
    Returns:
        Tuple of (client, client_type) or (None, None) if no client available
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Try Anthropic first
    if ANTHROPIC_AVAILABLE and api_config.get("anthropic", {}).get("api_key"):
        try:
            client = anthropic.AsyncAnthropic(
                api_key=api_config["anthropic"]["api_key"]
            )
            if verbose_output:
                logger.info("Initialized Anthropic client")
            return client, "anthropic"
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic client: {e}")
    
    # Try OpenAI second
    if OPENAI_AVAILABLE and api_config.get("openai", {}).get("api_key"):
        try:
            client = openai.AsyncOpenAI(
                api_key=api_config["openai"]["api_key"],
                base_url=api_config["openai"].get("base_url", "https://api.openai.com/v1")
            )
            if verbose_output:
                logger.info("Initialized OpenAI client")
            return client, "openai"
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
    
    # Try Google third
    if GOOGLE_AVAILABLE and api_config.get("google", {}).get("api_key"):
        try:
            genai.configure(api_key=api_config["google"]["api_key"])
            client = genai
            if verbose_output:
                logger.info("Initialized Google client")
            return client, "google"
        except Exception as e:
            logger.warning(f"Failed to initialize Google client: {e}")
    
    logger.warning("No LLM client available - check API configuration")
    return None, None


def get_model_for_task(
    task_type: str,
    default_models: Dict[str, str],
    client_type: Optional[str] = None
) -> str:
    """
    Get the appropriate model for a task type.
    
    Args:
        task_type: Type of task (e.g., 'analysis', 'generation', 'coding')
        default_models: Dict of default models for each provider
        client_type: Type of client ('anthropic', 'openai', 'google')
        
    Returns:
        Model name to use
    """
    if client_type and client_type in default_models:
        return default_models[client_type]
    
    # Fallback to any available model
    for provider, model in default_models.items():
        if model:
            return model
    
    return "gpt-3.5-turbo"  # Ultimate fallback


async def call_llm(
    client: Any,
    client_type: str,
    prompt: str,
    model: str,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Call an LLM with the given prompt.
    
    Args:
        client: Initialized LLM client
        client_type: Type of client
        prompt: Prompt to send
        model: Model to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation
        logger: Optional logger
        
    Returns:
        Generated text or None if failed
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        if client_type == "anthropic":
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
            
        elif client_type == "openai":
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
            
        elif client_type == "google":
            model_obj = client.GenerativeModel(model)
            response = await model_obj.generate_content_async(prompt)
            return response.text
            
        else:
            logger.warning(f"Unknown client type: {client_type}")
            return None
            
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None
