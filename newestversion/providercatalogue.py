"""
Provider Catalogue for OpenEvolve
"""

import streamlit as st
import requests
from typing import List, Dict, Any, Optional
from openevolve_integration import OpenEvolveAPI


def get_openrouter_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch models from OpenRouter and return their details."""
    openrouter_config = PROVIDERS["openrouter"]
    models_endpoint = openrouter_config["models_endpoint"]
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(models_endpoint, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        models = data.get("data", []) if isinstance(data, dict) else []
        return models
    except Exception as e:
        st.warning(f"Could not fetch models from OpenRouter: {e}")
        return []


def _parse_price_per_million(price: Any) -> Optional[float]:
    """Parses price per million tokens from various formats."""
    if price is None:
        return None
    if isinstance(price, (int, float)):
        return float(price)
    if isinstance(price, str):
        price = price.lower().replace("$", "").strip()
        if "/m" in price:
            price = price.replace("/m", "").strip()
        try:
            return float(price)
        except ValueError:
            pass
    return None


# Helper function for OpenAI-style loaders
def _openai_style_loader(url: str, api_key: Optional[str] = None) -> List[str]:
    """Generic loader for OpenAI-style APIs."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        models = data.get("data", []) if isinstance(data, dict) else []
        return [
            model["id"] for model in models if isinstance(model, dict) and "id" in model
        ]
    except Exception as e:
        st.warning(f"Could not fetch models from {url}: {e}")
        return []


# Specific loaders for providers that don't follow OpenAI-style APIs
def _groq_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Groq models."""
    # Groq doesn't have a models endpoint, so we'll return a predefined list
    return [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "llama-3.1-405b-reasoning",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama-guard-3-8b",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "gemma2-9b-it",
    ]


def _together_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Together models."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(
            "https://api.together.xyz/v1/models", headers=headers, timeout=10
        )
        response.raise_for_status()
        data = response.json()
        models = (
            data
            if isinstance(data, list)
            else data.get("data", [])
            if isinstance(data, dict)
            else []
        )
        return [
            model["id"] for model in models if isinstance(model, dict) and "id" in model
        ]
    except Exception as e:
        st.warning(f"Could not fetch Together models: {e}")
        # Return a predefined list if API fails
        return [
            "meta-llama/Llama-3-8b-chat-hf",
            "meta-llama/Llama-3-70b-chat-hf",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ]


def _fireworks_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Fireworks models."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(
            "https://api.fireworks.ai/inference/v1/models", headers=headers, timeout=10
        )
        response.raise_for_status()
        data = response.json()
        models = (
            data
            if isinstance(data, list)
            else data.get("data", [])
            if isinstance(data, dict)
            else []
        )
        return [
            model["id"] for model in models if isinstance(model, dict) and "id" in model
        ]
    except Exception as e:
        st.warning(f"Could not fetch Fireworks models: {e}")
        # Return a predefined list if API fails
        return [
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "accounts/fireworks/models/llama-v3p1-405b-instruct",
            "accounts/fireworks/models/mixtral-8x7b-instruct",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
        ]


def _moonshot_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Moonshot models."""
    # Moonshot doesn't have a public models endpoint, so we'll return a predefined list
    return ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]


def _baichuan_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Baichuan models."""
    # Baichuan doesn't have a public models endpoint, so we'll return a predefined list
    return [
        "Baichuan2-Turbo",
        "Baichuan2-Turbo-192k",
        "Baichuan3-Turbo",
        "Baichuan3-Turbo-128k",
        "Baichuan4",
    ]


def _zhipu_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Zhipu models."""
    # Zhipu doesn't have a public models endpoint, so we'll return a predefined list
    return [
        "glm-4-plus",
        "glm-4-0520",
        "glm-4",
        "glm-4-air",
        "glm-4-airx",
        "glm-4-long",
        "glm-4-flash",
        "glm-4v",
        "glm-4v-plus",
    ]


def _minimax_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Minimax models."""
    # Minimax doesn't have a public models endpoint, so we'll return a predefined list
    return ["abab6.5s-chat", "abab6.5-chat", "abab6-chat"]


def _yi_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Yi models."""
    # Yi doesn't have a public models endpoint, so we'll return a predefined list
    return [
        "yi-lightning",
        "yi-large",
        "yi-medium",
        "yi-medium-200k",
        "yi-spark",
        "yi-large-rag",
        "yi-large-turbo",
        "yi-large-preview",
    ]


def _deepseek_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for DeepSeek models."""
    # DeepSeek doesn't have a public models endpoint, so we'll return a predefined list
    return ["deepseek-chat", "deepseek-coder"]


def fetch_providers_from_backend(api: OpenEvolveAPI) -> Dict[str, Any]:
    """Fetch the list of available providers from the backend."""
    try:
        response = api.get("/providers")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching providers from backend: {e}")
        return {}


PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "api_base": "https://api.openai.com/v1",
        "models_endpoint": "https://api.openai.com/v1/models",
        "loader": _openai_style_loader,
        "default_model": "gpt-4o",
    },
    "anthropic": {
        "name": "Anthropic",
        "api_base": "https://api.anthropic.com/v1",
        "models_endpoint": "https://api.anthropic.com/v1/models",
        "loader": _openai_style_loader,
        "default_model": "claude-3-opus-20240229",
    },
    "google": {
        "name": "Google",
        "api_base": "https://generativelanguage.googleapis.com/v1beta",
        "models_endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
        "loader": _openai_style_loader,
        "default_model": "gemini-pro",
    },
    "openrouter": {
        "name": "OpenRouter",
        "api_base": "https://openrouter.ai/api/v1",
        "models_endpoint": "https://openrouter.ai/api/v1/models",
        "loader": _openai_style_loader,
        "default_model": "mistralai/mistral-7b-instruct",
    },
    "groq": {
        "name": "Groq",
        "api_base": "https://api.groq.com/openai/v1",
        "models_endpoint": None,  # Groq doesn't have a models endpoint
        "loader": _groq_loader,
        "default_model": "llama3-8b-8192",
    },
    "together": {
        "name": "Together AI",
        "api_base": "https://api.together.xyz/v1",
        "models_endpoint": "https://api.together.xyz/v1/models",
        "loader": _together_loader,
        "default_model": "meta-llama/Llama-3-8b-chat-hf",
    },
    "fireworks": {
        "name": "Fireworks AI",
        "api_base": "https://api.fireworks.ai/inference/v1",
        "models_endpoint": "https://api.fireworks.ai/inference/v1/models",
        "loader": _fireworks_loader,
        "default_model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    },
    "moonshot": {
        "name": "Moonshot AI",
        "api_base": "https://api.moonshot.cn/v1",
        "models_endpoint": None,
        "loader": _moonshot_loader,
        "default_model": "moonshot-v1-8k",
    },
    "baichuan": {
        "name": "Baichuan AI",
        "api_base": "https://api.baichuan-ai.com/v1",
        "models_endpoint": None,
        "loader": _baichuan_loader,
        "default_model": "Baichuan2-Turbo",
    },
    "zhipu": {
        "name": "Zhipu AI",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "models_endpoint": None,
        "loader": _zhipu_loader,
        "default_model": "glm-4",
    },
    "minimax": {
        "name": "Minimax",
        "api_base": "https://api.minimax.chat/v1",
        "models_endpoint": None,
        "loader": _minimax_loader,
        "default_model": "abab6.5s-chat",
    },
    "yi": {
        "name": "01.AI (Yi)",
        "api_base": "https://api.01.ai/v1",
        "models_endpoint": None,
        "loader": _yi_loader,
        "default_model": "yi-large",
    },
    "deepseek": {
        "name": "DeepSeek",
        "api_base": "https://api.deepseek.com/v1",
        "models_endpoint": None,
        "loader": _deepseek_loader,
        "default_model": "deepseek-chat",
    },
}

PROVIDERS_CACHE = {}


def get_providers(api: OpenEvolveAPI) -> Dict[str, Any]:
    """Get the list of available providers, from cache or backend."""
    if not PROVIDERS_CACHE:
        backend_providers = fetch_providers_from_backend(api)
        if backend_providers:
            PROVIDERS_CACHE.update(backend_providers)
        else:
            # Fallback to hardcoded providers if backend fails
            PROVIDERS_CACHE.update(PROVIDERS)
    return PROVIDERS_CACHE
