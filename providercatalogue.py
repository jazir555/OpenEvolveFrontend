"""
Provider Catalogue for OpenEvolve
"""

import streamlit as st
import requests
from typing import List, Dict, Any, Optional


# Helper function for OpenAI-style loaders
def _openai_style_loader(url: str, api_key: Optional[str] = None) -> List[str]:
    """Generic loader for OpenAI-style APIs."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        models = data.get("data", []) if isinstance(data, dict) else []
        return [model["id"] for model in models if isinstance(model, dict) and "id" in model]
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
        "gemma2-9b-it"
    ]


def _together_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Together models."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get("https://api.together.xyz/v1/models", headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        models = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
        return [model["id"] for model in models if isinstance(model, dict) and "id" in model]
    except Exception as e:
        st.warning(f"Could not fetch Together models: {e}")
        # Return a predefined list if API fails
        return [
            "meta-llama/Llama-3-8b-chat-hf",
            "meta-llama/Llama-3-70b-chat-hf",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-Instruct-v0.1"
        ]


def _fireworks_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Fireworks models."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get("https://api.fireworks.ai/inference/v1/models", headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        models = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
        return [model["id"] for model in models if isinstance(model, dict) and "id" in model]
    except Exception as e:
        st.warning(f"Could not fetch Fireworks models: {e}")
        # Return a predefined list if API fails
        return [
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "accounts/fireworks/models/llama-v3p1-405b-instruct",
            "accounts/fireworks/models/mixtral-8x7b-instruct",
            "accounts/fireworks/models/qwen2p5-72b-instruct"
        ]


def _moonshot_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Moonshot models."""
    # Moonshot doesn't have a public models endpoint, so we'll return a predefined list
    return [
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k"
    ]


def _baichuan_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Baichuan models."""
    # Baichuan doesn't have a public models endpoint, so we'll return a predefined list
    return [
        "Baichuan2-Turbo",
        "Baichuan2-Turbo-192k",
        "Baichuan3-Turbo",
        "Baichuan3-Turbo-128k",
        "Baichuan4"
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
        "glm-4v-plus"
    ]


def _minimax_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Minimax models."""
    # Minimax doesn't have a public models endpoint, so we'll return a predefined list
    return [
        "abab6.5s-chat",
        "abab6.5-chat",
        "abab6-chat"
    ]


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
        "yi-large-preview"
    ]


def _deepseek_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for DeepSeek models."""
    # DeepSeek doesn't have a public models endpoint, so we'll return a predefined list
    return [
        "deepseek-chat",
        "deepseek-coder"
    ]


def fetch_providers_from_backend(api: OpenEvolveAPI) -> Dict[str, Any]:
    """Fetch the list of available providers from the backend."""
    try:
        response = api.get("/providers")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching providers from backend: {e}")
        return {}

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