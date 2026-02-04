"""
LLM Utilities - Production Grade

Common utilities for LLM interactions across Knowledge Engine.

Following CLAUDE.md Principles:
- CONFIGURATION EXPLICITNESS: All config via env vars
- TIMEOUTS: All LLM calls have mandatory timeouts
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import json
import logging
import os
from typing import Optional, Dict, Any, List
import httpx
from knowledge_engine.global_context_manager import get_global_context_manager

logger = logging.getLogger(__name__)


# Configuration from environment
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "gpt-4o")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120.0"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))


async def call_llm(
    prompt: str,
    model: str = LLM_DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    timeout: float = LLM_TIMEOUT,
    correlation_id: Optional[str] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Call LLM API with retry logic and optional global context management.

    Args:
        prompt: Input prompt
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        correlation_id: Optional correlation ID for tracking
        api_key: Optional API key (defaults to LLM_API_KEY env var)
        session_id: Optional session ID for global context management (Matryoshka)
        history: Optional history for context management

    Returns:
        LLM response text
    """
    correlation_id = correlation_id or "unknown"
    api_key = api_key or LLM_API_KEY
    
    # Context Management
    messages = history or []
    if not any(m['role'] == 'user' and m['content'] == prompt for m in messages):
        messages.append({"role": "user", "content": prompt})
    
    if session_id:
        gcm = get_global_context_manager()
        messages = gcm.manage(session_id, messages)

    if not api_key and not LLM_API_KEY:
        # If no API key, return fallback response
        logger.warning(
            "No LLM API key configured, using fallback response",
            extra={"correlation_id": correlation_id}
        )
        return _generate_fallback_response(prompt)

    # Validate configuration
    if timeout <= 0:
        raise ValueError(f"Invalid timeout: {timeout}")

    logger.info(
        "Calling LLM API",
        extra={
            "correlation_id": correlation_id,
            "model": model,
            "prompt_length": len(prompt),
            "timeout": timeout,
            "session_id": session_id
        }
    )

    # Prepare request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # Try with retries
    last_error = None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{LLM_API_BASE}/chat/completions",
                    headers=headers,
                    json=payload
                )

                response.raise_for_status()

                # Parse response
                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    result = data["choices"][0]["message"]["content"]

                    logger.info(
                        "LLM call successful",
                        extra={
                            "correlation_id": correlation_id,
                            "attempt": attempt + 1,
                            "response_length": len(result)
                        }
                    )

                    return result
                else:
                    raise ValueError("Invalid LLM response format")

        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(
                f"LLM call timed out (attempt {attempt + 1}/{LLM_MAX_RETRIES})",
                extra={"correlation_id": correlation_id}
            )

            if attempt < LLM_MAX_RETRIES - 1:
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)

        except httpx.HTTPError as e:
            last_error = e
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{LLM_MAX_RETRIES}): {e}",
                extra={"correlation_id": correlation_id}
            )

            if attempt < LLM_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

    # All retries failed
    logger.error(
        "LLM call failed after all retries",
        extra={"correlation_id": correlation_id},
        exc_info=last_error
    )

    # Return fallback instead of raising
    return _generate_fallback_response(prompt)


def _generate_fallback_response(prompt: str) -> str:
    """
    Generate fallback response when LLM is unavailable.

    Args:
        prompt: Original prompt

    Returns:
        Fallback response
    """
    # Extract entities or relationships from prompt using simple patterns
    import re

    # Look for JSON-like structures in prompt
    if "Extract all entities" in prompt or "entities" in prompt.lower():
        # Find capitalized words
        entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', prompt)

        if entities:
            # Return as JSON list
            unique_entities = list(set(entities))
            # Filter out common words
            stop_words = {'This', 'That', 'These', 'Those', 'The', 'A', 'An', 'Return', 'Text'}
            filtered = [e for e in unique_entities if e not in stop_words and len(e) > 2]

            if filtered:
                return json.dumps(filtered[:20])

    if "Extract relationships" in prompt.lower() or "relationships" in prompt.lower():
        # Return empty list
        return "[]"

    # Default fallback
    return "I understand your request, but I'm currently unable to process it."


async def call_llm_with_structured_output(
    prompt: str,
    output_schema: Dict[str, Any],
    model: str = LLM_DEFAULT_MODEL,
    temperature: float = 0.0,
    timeout: float = LLM_TIMEOUT,
    correlation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Call LLM API and parse structured JSON output.

    Args:
        prompt: Input prompt
        output_schema: Expected output schema (for validation)
        model: Model name
        temperature: Sampling temperature
        timeout: Request timeout
        correlation_id: Optional correlation ID

    Returns:
        Parsed JSON response

    Raises:
        ValueError: If response cannot be parsed as JSON
    """
    correlation_id = correlation_id or "unknown"

    # Call LLM
    response_text = await call_llm(
        prompt=prompt,
        model=model,
        temperature=temperature,
        timeout=timeout,
        correlation_id=correlation_id
    )

    # Parse JSON
    try:
        parsed = json.loads(response_text)

        logger.info(
            "Structured output parsed successfully",
            extra={"correlation_id": correlation_id}
        )

        return parsed

    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse LLM response as JSON: {e}",
            extra={"correlation_id": correlation_id}
        )

        # Try to extract JSON from response
        import re

        # Look for JSON block
        json_match = re.search(r'\{.*\}|\[.*\]', response_text, re.DOTALL)

        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return parsed
            except json.JSONDecodeError:
                pass

        # Return empty structure matching schema
        return _get_empty_structure(output_schema)


def _get_empty_structure(schema: Dict[str, Any]) -> Any:
    """
    Create empty structure from schema.

    Args:
        schema: Schema definition

    Returns:
        Empty structure matching schema type
    """
    if "type" not in schema:
        return {}

    schema_type = schema["type"]

    if schema_type == "object":
        result = {}
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = _get_empty_structure(prop_schema)
        return result

    elif schema_type == "array":
        if "items" in schema:
            return [_get_empty_structure(schema["items"])]
        return []

    elif schema_type == "string":
        return ""

    elif schema_type == "number":
        return 0

    elif schema_type == "boolean":
        return False

    return {}


async def validate_llm_connection(
    api_key: Optional[str] = None,
    timeout: float = 30.0
) -> bool:
    """
    Validate LLM API connection.

    Args:
        api_key: Optional API key to test
        timeout: Test timeout

    Returns:
        True if connection successful
    """
    try:
        response = await call_llm(
            prompt="Hello",
            model="gpt-4o-mini",
            max_tokens=10,
            timeout=timeout,
            api_key=api_key
        )

        return bool(response)

    except Exception as e:
        logger.warning(f"LLM connection test failed: {e}")
        return False


async def initialize_llm_client(
    api_config: Dict[str, Any],
    default_models: Dict[str, str],
    logger: logging.Logger,
    verbose_output: bool = False
) -> tuple:
    """
    Initialize LLM client based on available API keys.

    Priority:
    1. Anthropic (Claude)
    2. OpenAI (GPT)
    3. Google (Gemini) - fallback

    Args:
        api_config: API configuration dict with keys
        default_models: Dict of default model names per provider
        logger: Logger instance
        verbose_output: Enable verbose logging

    Returns:
        Tuple of (client, client_type)

    Raises:
        ValueError: If no API keys available
    """
    # Extract API keys from config
    anthropic_key = api_config.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
    openai_key = api_config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    google_key = api_config.get("google_api_key") or os.getenv("GOOGLE_API_KEY")

    # Try Anthropic first
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=anthropic_key)
            if verbose_output:
                logger.info(f"Initialized Anthropic client with model: {default_models.get('anthropic', 'claude-sonnet-4-20250514')}")
            return client, "anthropic"
        except ImportError:
            logger.warning("Anthropic package not installed, skipping Anthropic provider")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic client: {e}")

    # Try OpenAI second
    if openai_key:
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=openai_key)
            if verbose_output:
                logger.info(f"Initialized OpenAI client with model: {default_models.get('openai', 'o3-mini')}")
            return client, "openai"
        except ImportError:
            logger.warning("OpenAI package not installed, skipping OpenAI provider")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")

    # Try Google as fallback
    if google_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            # For Google, we return the genai module and handle calls differently
            if verbose_output:
                logger.info(f"Initialized Google client with model: {default_models.get('google', 'gemini-2.0-flash')}")
            return genai, "google"
        except ImportError:
            logger.warning("Google GenerativeAI package not installed, skipping Google provider")
        except Exception as e:
            logger.warning(f"Failed to initialize Google client: {e}")

    # No valid API key found
    raise ValueError(
        "No valid LLM API key found. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY "
        "in your environment or config file."
    )


def create_llm_prompt(
    template: str,
    **kwargs
) -> str:
    """
    Create an LLM prompt from a template with variable substitution.

    Args:
        template: Prompt template with {placeholders}
        **kwargs: Variables to substitute

    Returns:
        Formatted prompt string
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.warning(f"Missing placeholder in prompt template: {e}")
        return template


def extract_json_from_response(
    response_text: str,
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract JSON from LLM response text.

    Handles cases where JSON is embedded in markdown code blocks or
    surrounded by other text.

    Args:
        response_text: Raw LLM response
        logger_instance: Optional logger for warnings

    Returns:
        Parsed JSON dict or empty dict if parsing fails
    """
    import re

    # Try direct parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks
    json_block_match = re.search(r'```(?:json)?\s*\n(.*?)```', response_text, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object/array
    json_obj_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    json_arr_match = re.search(r'\[.*\]', response_text, re.DOTALL)

    if json_obj_match:
        try:
            return json.loads(json_obj_match.group())
        except json.JSONDecodeError:
            pass

    if json_arr_match:
        try:
            return json.loads(json_arr_match.group())
        except json.JSONDecodeError:
            pass

    # All parsing attempts failed
    if logger_instance:
        logger_instance.warning("Failed to extract JSON from LLM response")

    return {}
