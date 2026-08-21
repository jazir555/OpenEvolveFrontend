from __future__ import annotations


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


def _request_openai_compatible_chat(
    api_key: str,
    base_url: Optional[str],
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Optional[str]:
    """
    Make a request to an OpenAI-compatible chat API.

    This function is a shared utility used by workflow_engine and OpenEvolve integration.
    It accepts a broad set of keyword arguments and forwards the supported ones.
    """
    base_url = base_url or "https://api.openai.com/v1"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if seed is not None:
        payload["seed"] = seed

    if "stop_sequences" in kwargs and kwargs["stop_sequences"]:
        payload["stop"] = kwargs["stop_sequences"]
    if "stop" in kwargs and kwargs["stop"]:
        payload["stop"] = kwargs["stop"]

    # Optional OpenAI-compatible fields
    optional_fields = [
        "n",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "response_format",
        "stream",
        "user",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "metadata",
    ]
    for field_name in optional_fields:
        if field_name in kwargs and kwargs[field_name] is not None:
            payload[field_name] = kwargs[field_name]

    if kwargs.get("response_json_format") and "response_format" not in payload:
        payload["response_format"] = {"type": "json_object"}

    if payload.get("max_tokens") is None and kwargs.get("max_output_tokens"):
        payload["max_tokens"] = kwargs["max_output_tokens"]

    try:
        if OPENAI_AVAILABLE:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(**payload)
            return response.choices[0].message.content
    except Exception as e:
        logging.getLogger(__name__).warning(
            "OpenAI client failed, falling back to requests: %s", e
        )

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=kwargs.get("timeout", 120),
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]


def _compose_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """Compose chat messages from system and user prompts."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    return messages
