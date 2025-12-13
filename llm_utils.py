
import requests
from typing import Dict, Any, Optional, List
import asyncio
import logging
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

def _compose_messages(system_message: str, user_message: str) -> List[Dict[str, str]]:
    """Helper function to compose messages in the format expected by OpenAI-compatible chat APIs.

    Args:
        system_message (str): The system message to set the context or role of the AI.
        user_message (str): The user's message or prompt.

    Returns:
        List[Dict[str, str]]: A list of message dictionaries.
    """
    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

def _request_openai_compatible_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    extra_headers: Optional[Dict[str, str]] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 4096,
    seed: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    response_format: Optional[Dict[str, str]] = None,
    stream: Optional[bool] = None,
    user: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_retries: int = 5,
    timeout: int = 120,
    organization: Optional[str] = None,
    response_model: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    system_fingerprint: Optional[str] = None,
    deployment_id: Optional[str] = None,
    encoding_format: Optional[str] = None,
    max_input_tokens: Optional[int] = None,
    stop_token: Optional[str] = None,
    best_of: Optional[int] = None,
    logprobs_offset: Optional[int] = None,
    suffix: Optional[str] = None,
    presence_penalty_range: Optional[List[float]] = None,
    frequency_penalty_range: Optional[List[float]] = None,
    stop_token_id: Optional[int] = None,
    response_json_format: Optional[bool] = None,
    max_output_tokens: Optional[int] = None,
    stream_options: Optional[Dict[str, Any]] = None,
    logprobs_type: Optional[str] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    length_penalty: Optional[float] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature_fallback: Optional[float] = None,
    top_p_fallback: Optional[float] = None,
    max_time: Optional[int] = None,
    return_full_text: Optional[bool] = None,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Makes a request to an OpenAI-compatible API endpoint for chat completions.
    """
    try:
        import openai
        # Initialize OpenAI client with organization and base_url
        client_params = {"api_key": api_key, "base_url": base_url}
        if organization: client_params["organization"] = organization
        client = openai.OpenAI(**client_params)
        
        completion_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "seed": seed,
            "stop": stop_sequences if stop_sequences else stop_token,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "response_format": response_format,
            "stream": stream,
            "user": user,
            "tools": tools,
            "tool_choice": tool_choice,
            "system_fingerprint": system_fingerprint,
            "best_of": best_of,
            "logit_bias": None, # Not directly exposed in ModelConfig yet, but can be added
            "suffix": suffix,
            "stop_token_id": stop_token_id,
            "max_output_tokens": max_output_tokens,
            "stream_options": stream_options,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "temperature_fallback": temperature_fallback,
            "top_p_fallback": top_p_fallback,
            "max_time": max_time,
            "return_full_text": return_full_text,
        }
        # Filter out None values to avoid sending them to the API if not specified
        completion_params = {k: v for k, v in completion_params.items() if v is not None}

        response = client.chat.completions.create(**completion_params)
        
        return response.choices[0].message.content
        
    except ImportError:
        # If openai package is not available, try using requests
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "seed": seed,
            "stop": stop_sequences if stop_sequences else stop_token,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "response_format": response_format,
            "stream": stream,
            "user": user,
            "tools": tools,
            "tool_choice": tool_choice,
            "system_fingerprint": system_fingerprint,
            "best_of": best_of,
            "logit_bias": None, # Not directly exposed in ModelConfig yet
            "suffix": suffix,
            "stop_token_id": stop_token_id,
            "max_output_tokens": max_output_tokens,
            "stream_options": stream_options,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
            "num_beams: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature_fallback: Optional[float] = None,
    top_p_fallback: Optional[float] = None,
    max_time: Optional[int] = None,
    return_full_text: Optional[bool] = None,
        }
        # Filter out None values
        data = {k: v for k, v in data.items() if v is not None}
            
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except Exception as e:
        print(f"Error making API request: {e}")
        return None


def call_openevolve_cascade(content: str, api_key: str, thresholds: List[float]) -> Dict[str, Any]:
    """Call OpenEvolve with cascade evaluation"""
    try:
        from openevolve_client import OpenEvolveClient
        
        client = OpenEvolveClient(api_key=api_key)
        
        result = client.evolve(
            content=content,
            evolution_mode="standard",
            max_iterations=10,
            population_size=20,
            cascade_evaluation=True,
            cascade_thresholds=thresholds
        )
        
        return result
    except Exception as e:
        return {'error': str(e)}

def call_openevolve_ensemble(content: str, api_key: str, num_models: int) -> Dict[str, Any]:
    """Call OpenEvolve with ensemble of models"""
    try:
        from openevolve_client import OpenEvolveClient
        
        client = OpenEvolveClient(api_key=api_key)
        
        result = client.evolve(
            content=content,
            evolution_mode="standard",
            max_iterations=5,
            population_size=num_models,
            temperature=0.7
        )
        
        return result
    except Exception as e:
        return {'error': str(e)}

async def initialize_llm_client(
    api_config: Dict[str, Any],
    default_models: Dict[str, str],
    logger: logging.Logger,
    verbose_output: bool = False
) -> tuple[Any, str]:
    """
    Initializes and returns an LLM client (Anthropic or OpenAI) based on API key availability.
    Tests the connection before returning.
    """
    # Check which API has available key and try that first
    anthropic_key = api_config.get("anthropic", {}).get("api_key", "")
    openai_key = api_config.get("openai", {}).get("api_key", "")

    # Try Anthropic API first if key is available
    if anthropic_key and anthropic_key.strip():
        try:
            client = AsyncAnthropic(api_key=anthropic_key)
            # Test connection with default model from config
            await client.messages.create(
                model=default_models["anthropic"],
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}],
            )
            if verbose_output:
                logger.info(
                    f"Using Anthropic API with model: {default_models['anthropic']}"
                )
            return client, "anthropic"
        except Exception as e:
            logger.warning(f"Anthropic API unavailable: {e}")

    # Try OpenAI API if Anthropic failed or key not available
    if openai_key and openai_key.strip():
        try:
            # Handle custom base_url if specified
            openai_provider_config = api_config.get("openai", {})
            base_url = openai_provider_config.get("base_url")

            if base_url:
                client = AsyncOpenAI(api_key=openai_key, base_url=base_url)
            else:
                client = AsyncOpenAI(api_key=openai_key)

            # Test connection with default model from config
            await client.chat.completions.create(
                model=default_models["openai"],
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}],
            )
            if verbose_output:
                logger.info(
                    f"Using OpenAI API with model: {default_models['openai']}"
                )
                if base_url:
                    logger.info(f"Using custom base URL: {base_url}")
            return client, "openai"
        except Exception as e:
            logger.warning(f"OpenAI API unavailable: {e}")

    raise ValueError(
        "No available LLM API - please check your API keys in configuration"
    )
