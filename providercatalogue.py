import streamlit as st
"""
Provider Catalogue for OpenEvolve
"""

import requests
from typing import List, Dict, Any, Optional
from openevolve_integration import OpenEvolveAPI

# Optional imports with fallbacks
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None
    BOTO3_AVAILABLE = False

try:
    import google.cloud.aiplatform as aiplatform
    GOOGLE_AIP_AVAILABLE = True
except ImportError:
    aiplatform = None
    GOOGLE_AIP_AVAILABLE = False

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    NVIDIA_AVAILABLE = True
except ImportError:
    ChatNVIDIA = None
    NVIDIA_AVAILABLE = False

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    replicate = None
    REPLICATE_AVAILABLE = False

try:
    from aleph_alpha_client import Client
    ALEPH_ALPHA_AVAILABLE = True
except ImportError:
    Client = None
    ALEPH_ALPHA_AVAILABLE = False

try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    runpod = None
    RUNPOD_AVAILABLE = False

import subprocess
import json


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _bedrock_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Amazon Bedrock models, attempting to use boto3 with a static fallback."""
    if not BOTO3_AVAILABLE:
        st.warning("boto3 not available. Cannot fetch models from Amazon Bedrock. Falling back to static list.")
        return [
            "anthropic.claude-v2",
            "anthropic.claude-v2:1", 
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "meta.llama2-13b-chat-v1",
            "meta.llama2-70b-chat-v1",
            "meta.llama3-8b-instruct-v1:0",
            "meta.llama3-70b-instruct-v1:0",
            "mistral.mistral-7b-instruct-v0:2",
            "mistral.mixtral-8x7b-instruct-v0:1",
            "mistral.mistral-large-2402-v1:0",
        ]
    try:
        # Bedrock API key is typically managed via AWS credentials, not passed directly.
        # The api_key parameter here might be used for region or other config if needed.
        bedrock_runtime = boto3.client(
            service_name="bedrock",
            region_name="us-east-1", # Default region, can be made configurable
            # aws_access_key_id=...,
            # aws_secret_access_key=...,
        )
        response = bedrock_runtime.list_foundation_models()
        models = [
            model["modelId"]
            for model in response["modelSummaries"]
            if "modelId" in model
        ]
        if models:
            return models
    except Exception as e:
        st.warning(f"Could not fetch models from Amazon Bedrock using boto3: {e}. Falling back to static list.")
    # Fallback to a static list if boto3 fails or is not configured
    return [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
        "amazon.titan-embed-text-v1",
        "cohere.command-text-v14",
        "cohere.command-light-text-v14",
        "cohere.embed-english-v3",
        "cohere.embed-multilingual-v3",
        "meta.llama2-13b-chat-v1",
        "meta.llama2-70b-chat-v1",
        "stability.stable-diffusion-xl-v1",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _vertex_ai_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Google Vertex AI models, attempting to use the SDK with a static fallback."""
    try:
        # Assuming project_id and location are configured elsewhere or can be passed.
        # For simplicity, using hardcoded defaults or environment variables.
        project_id = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
        location = os.environ.get("GCP_LOCATION", "us-central1")

        aiplatform.init(project=project_id, location=location)
        models = aiplatform.Model.list()
        model_ids = [model.display_name for model in models if model.display_name]
        if model_ids:
            return model_ids
    except Exception as e:
        st.warning(f"Could not fetch models from Google Vertex AI using SDK: {e}. Falling back to static list.")
    # Fallback to a static list if SDK fails or is not configured
    return [
        "gemini-pro",
        "gemini-pro-vision",
        "text-bison",
        "chat-bison",
        "code-bison",
        "codechat-bison",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _nvidia_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for NVIDIA AI Foundation Models, attempting to use the SDK with a static fallback."""
    try:
        # API key can be passed directly or set as an environment variable NVIDIA_API_KEY
        chat_nvidia = ChatNVIDIA(nvidia_api_key=api_key) if api_key else ChatNVIDIA()
        available_models = chat_nvidia.available_models
        model_ids = [model.id for model in available_models if model.id]
        if model_ids:
            return model_ids
    except Exception as e:
        st.warning(f"Could not fetch models from NVIDIA AI Foundation Models using SDK: {e}. Falling back to static list.")
    # Fallback to a static list if SDK fails or is not configured
    return [
        "nv-llama2-70b",
        "nv-mistral-7b",
        "nv-gemma-7b",
        "nv-mixtral-8x7b",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _replicate_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Replicate models, attempting to use the SDK with a static fallback."""
    try:
        # API key can be passed directly or set as an environment variable REPLICATE_API_TOKEN
        if api_key:
            os.environ["REPLICATE_API_TOKEN"] = api_key
        
        all_models = []
        for page in replicate.paginate(replicate.models.list):
            for model in page:
                all_models.append(f"{model.owner}/{model.name}") # Replicate models are typically owner/name
        
        if all_models:
            return all_models
    except Exception as e:
        st.warning(f"Could not fetch models from Replicate using SDK: {e}. Falling back to static list.")
    # Fallback to a static list if SDK fails or is not configured
    return [
        "meta/llama-2-70b-chat",
        "stability-ai/stable-diffusion",
        "andreasjansson/blip-2",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _aleph_alpha_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Aleph Alpha models, attempting to use the SDK with a static fallback."""
    try:
        # API key can be passed directly or set as an environment variable AA_TOKEN or ALEPH_ALPHA_API_KEY
        client = Client(token=api_key) if api_key else Client(token=os.environ.get("AA_TOKEN") or os.environ.get("ALEPH_ALPHA_API_KEY"))
        model_settings = client.get_model_settings()
        model_ids = list(model_settings.keys())
        if model_ids:
            return model_ids
    except Exception as e:
        st.warning(f"Could not fetch models from Aleph Alpha using SDK: {e}. Falling back to static list.")
    # Fallback to a static list if SDK fails or is not configured
    return [
        "luminous-extended",
        "luminous-base",
        "luminous-supreme",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _ai21_static_loader(api_key: Optional[str] = None) -> List[str]:
    """Static loader for AI21 Labs models, as no dynamic models endpoint is available.
    Models are based on available documentation.
    """
    return [
        "jamba-mini",
        "jamba-large",
        "j2-mid",
        "j2-ultra",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _baseten_static_loader(api_key: Optional[str] = None) -> List[str]:
    """Static loader for Baseten models, as no dynamic models endpoint is available.
    Models are based on common deployments.
    """
    return [
        "llama-2-7b-chat",
        "stable-diffusion-v1-5",
        "whisper",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _runpod_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for RunPod models, attempting to use the SDK with a static fallback."""
    try:
        # API key can be passed directly or set as an environment variable RUNPOD_API_KEY
        if api_key:
            runpod.api_key = api_key
        else:
            runpod.api_key = os.getenv("RUNPOD_API_KEY")

        if not runpod.api_key:
            raise ValueError("RunPod API key not found. Please set the RUNPOD_API_KEY environment variable or pass it directly.")

        endpoints = runpod.get_endpoints()
        model_ids = []
        for endpoint in endpoints:
            if "modelName" in endpoint:
                model_ids.append(endpoint["modelName"])
            elif "name" in endpoint:
                model_ids.append(endpoint["name"])
        
        if model_ids:
            return model_ids
    except Exception as e:
        st.warning(f"Could not fetch models from RunPod using SDK: {e}. Falling back to static list.")
    # Fallback to a static list if SDK fails or is not configured
    return [
        "llama2-70b-chat",
        "stable-diffusion-v1-5",
        "whisper",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _ollama_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Ollama models, making an HTTP GET request to its API to list models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [
            model["name"]
            for model in data.get("models", [])
            if isinstance(model, dict) and "name" in model
        ]
        if models:
            return models
    except Exception as e:
        st.warning(f"Could not fetch models from Ollama: {e}. Falling back to static list.")
    # Fallback to a static list if API fails or is not running
    return [
        "llama2",
        "mistral",
        "gemma",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _llama_cpp_static_loader(api_key: Optional[str] = None) -> List[str]:
    """Static loader for llama.cpp models, as its /models endpoint typically returns only the currently loaded model.
    Models are based on common llama.cpp deployments.
    """
    return [
        "llama-2-7b-chat.Q4_0.gguf",
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "gemma-7b-it.Q4_K_M.gguf",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _vllm_static_loader(api_key: Optional[str] = None) -> List[str]:
    """Static loader for vLLM models, as no dynamic models endpoint is available.
    Models are based on common vLLM deployments.
    """
    return [
        "llama-2-7b-chat",
        "mistral-7b-instruct",
        "gemma-7b-it",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _modal_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Modal Labs models, attempting to use the CLI with a static fallback."""
    try:
        # Modal CLI requires authentication, typically via `modal token new`
        # We're not passing API key directly here, assuming CLI is configured.
        result = subprocess.run(
            ["modal", "app", "list", "--json"],
            capture_output=True,
            text=True,
            check=True
        )
        apps_data = json.loads(result.stdout)
        model_ids = [app.get("name") for app in apps_data if app.get("name")]
        if model_ids:
            return model_ids
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        st.warning(f"Could not fetch models from Modal Labs using CLI: {e}. Falling back to static list.")
    # Fallback to a static list if CLI fails or is not configured
    return [
        "modal-llama-2-70b-chat",
        "modal-stable-diffusion",
    ]


@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _bedrock_loader(api_key: Optional[str] = None) -> List[str]:
    """Loader for Amazon Bedrock models, attempting to use boto3 with a static fallback."""
    try:
        # Bedrock API key is typically managed via AWS credentials, not passed directly.
        # The api_key parameter here might be used for region or other config if needed.
        bedrock_runtime = boto3.client(
            service_name="bedrock",
            region_name="us-east-1", # Default region, can be made configurable
            # aws_access_key_id=...,
            # aws_secret_access_key=...,
        )
        response = bedrock_runtime.list_foundation_models()
        models = [
            model["modelId"]
            for model in response["modelSummaries"]
            if "modelId" in model
        ]
        if models:
            return models
    except Exception as e:
        st.warning(f"Could not fetch models from Amazon Bedrock using boto3: {e}. Falling back to static list.")
    # Fallback to a static list if boto3 fails or is not configured
    return [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
        "amazon.titan-embed-text-v1",
        "cohere.command-text-v14",
        "cohere.command-light-text-v14",
        "cohere.embed-english-v3",
        "cohere.embed-multilingual-v3",
        "meta.llama2-13b-chat-v1",
        "meta.llama2-70b-chat-v1",
        "stability.stable-diffusion-xl-v1",
    ]
import streamlit as st
import requests
from typing import List, Dict, Any, Optional
from openevolve_integration import OpenEvolveAPI

# Optional imports with fallbacks (duplicate import section, keeping consistency)
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None
    BOTO3_AVAILABLE = False

try:
    import google.cloud.aiplatform as aiplatform
    GOOGLE_AIP_AVAILABLE = True
except ImportError:
    aiplatform = None
    GOOGLE_AIP_AVAILABLE = False

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    NVIDIA_AVAILABLE = True
except ImportError:
    ChatNVIDIA = None
    NVIDIA_AVAILABLE = False

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    replicate = None
    REPLICATE_AVAILABLE = False

try:
    from aleph_alpha_client import Client
    ALEPH_ALPHA_AVAILABLE = True
except ImportError:
    Client = None
    ALEPH_ALPHA_AVAILABLE = False

try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    runpod = None
    RUNPOD_AVAILABLE = False

import subprocess
import json
import sys
print(f"DEBUG: sys.path in providercatalogue.py: {sys.path}")


@st.cache_data(ttl=3600) # Cache the result for 1 hour
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
@st.cache_data(ttl=3600) # Cache the result for 1 hour
def _openai_style_loader(url: str, api_key: Optional[str] = None) -> List[str]:
    """Generic loader for OpenAI-style APIs."""
    if not url:
        st.warning("Could not fetch models: Model endpoint URL is missing or invalid.")
        return []
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

@st.cache_data(ttl=3600) # Cache the result for 1 hour
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


@st.cache_data(ttl=3600) # Cache the result for 1 hour
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







def _minimax_static_loader(api_key: Optional[str] = None) -> List[str]:
    """Static loader for Minimax models, as no dynamic models endpoint is available.
    Models are based on available documentation.
    """
    return ["abab6.5s-chat", "abab6.5-chat", "abab6-chat"]


def _baichuan_static_loader(api_key: Optional[str] = None) -> List[str]:
    """Static loader for Baichuan models, as no dynamic models endpoint is available.
    Models are based on Baichuan Intelligent documentation.
    """
    return [
        "Baichuan-M2",
        "Baichuan4-Turbo",
        "Baichuan4-Air",
        "Baichuan4",
        "Baichuan3-Turbo",
        "Baichuan3-Turbo-128k",
        "Baichuan2-Turbo",
        # "Baichuan2-Turbo-192k" is currently offline according to documentation
    ]

def _generic_loader(api_key: Optional[str] = None) -> List[str]:
    """Generic loader for models without a public models endpoint."""
    return ["default-model-1", "default-model-2", "default-model-3"]



def fetch_providers_from_backend(api: OpenEvolveAPI) -> Dict[str, Any]:
    """Fetch the list of available providers from the backend."""
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
        "models_endpoint": "https://api.groq.com/openai/v1/models",
        "loader": _openai_style_loader,
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
        "models_endpoint": "https://api.moonshot.cn/v1/models",
        "loader": _openai_style_loader,
        "default_model": "moonshot-v1-8k",
    },
    "baichuan": {
        "name": "Baichuan AI",
        "api_base": "https://api.baichuan-ai.com/v1",
        "models_endpoint": None, # No standard models endpoint found, using static list.
        "loader": _baichuan_static_loader,
        "default_model": "Baichuan2-Turbo",
    },
    "zhipu": {
        "name": "Zhipu AI",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "models_endpoint": "https://open.bigmodel.cn/api/paas/v4/models",
        "loader": _openai_style_loader,
        "default_model": "glm-4",
    },
    "minimax": {
        "name": "Minimax",
        "api_base": "https://api.minimax.chat/v1",
        "models_endpoint": None, # No standard models endpoint found, using static list.
        "loader": _minimax_static_loader,
        "default_model": "abab6.5s-chat",
    },
    "yi": {
        "name": "01.AI (Yi)",
        "api_base": "https://api.01.ai/v1",
        "models_endpoint": "https://api.01.ai/v1/models",
        "loader": _openai_style_loader,
        "default_model": "yi-large",
    },
    "deepseek": {
        "name": "DeepSeek",
        "api_base": "https://api.deepseek.com/v1",
        "models_endpoint": "https://api.deepseek.com/v1/models",
        "loader": _openai_style_loader,
        "default_model": "deepseek-chat",
    },
    "azure_openai": {
        "name": "Azure OpenAI",
        "api_base": "YOUR_AZURE_ENDPOINT",
        "models_endpoint": "YOUR_AZURE_ENDPOINT/openai/deployments?api-version=2023-05-15",
        "loader": _openai_style_loader,
        "default_model": "gpt-4",
    },
    "perplexity": {
        "name": "Perplexity AI",
        "api_base": "https://api.perplexity.ai",
        "models_endpoint": "https://api.perplexity.ai/models",
        "loader": _openai_style_loader,
        "default_model": "llama-3-sonar-small-32k-online",
    },
    "cohere": {
        "name": "Cohere",
        "api_base": "https://api.cohere.ai",
        "models_endpoint": "https://api.cohere.ai/v1/models",
        "loader": _openai_style_loader,
        "default_model": "command-r-plus",
    },
    "mistral": {
        "name": "Mistral AI",
        "api_base": "https://api.mistral.ai/v1",
        "models_endpoint": "https://api.mistral.ai/v1/models",
        "loader": _openai_style_loader,
        "default_model": "mistral-large-latest",
    },
    "anyscale": {
        "name": "Anyscale Endpoints",
        "api_base": "https://api.endpoints.anyscale.com/v1",
        "models_endpoint": "https://api.endpoints.anyscale.com/v1/models",
        "loader": _openai_style_loader,
        "default_model": "mistralai/Mistral-7B-Instruct-v0.1",
    },
    "databricks": {
        "name": "Databricks",
        "api_base": "https://dbc-YOUR_WORKSPACE_ID.cloud.databricks.com/serving-endpoints",
        "models_endpoint": "https://dbc-YOUR_WORKSPACE_ID.cloud.databricks.com/api/2.0/serving-endpoints",
        "loader": _openai_style_loader,
        "default_model": "databricks-llama-2-70b-chat",
    },
    "novita": {
        "name": "Novita AI",
        "api_base": "https://api.novita.ai/v3/openai",
        "models_endpoint": "https://api.novita.ai/v3/openai/models",
        "loader": _openai_style_loader,
        "default_model": "gpt-3.5-turbo",
    },
    "deepinfra": {
        "name": "DeepInfra",
        "api_base": "https://api.deepinfra.com/v1/openai",
        "models_endpoint": "https://api.deepinfra.com/v1/openai/models",
        "loader": _openai_style_loader,
        "default_model": "meta-llama/Llama-2-70b-chat-hf",
    },
    "tii": {
        "name": "TII Falcon",
        "api_base": "https://api.tii.ae/v1",
        "models_endpoint": "https://api.tii.ae/v1/models",
        "loader": _openai_style_loader,
        "default_model": "falcon-180b-chat",
    },
    "huggingface": {
        "name": "Hugging Face",
        "api_base": "https://api-inference.huggingface.co/models",
        "models_endpoint": "https://api-inference.huggingface.co/models", # This might need a custom loader for specific HF models
        "loader": _openai_style_loader, # Placeholder, might need custom logic
        "default_model": "HuggingFaceH4/zephyr-7b-beta",
    },
    "amazon_bedrock": {
        "name": "Amazon Bedrock",
        "api_base": "https://bedrock.us-east-1.amazonaws.com",
        "models_endpoint": None,  # Bedrock requires a more complex API for model listing (e.g., boto3), not a simple /models endpoint.
        "loader": _bedrock_loader,
        "default_model": "anthropic.claude-3-sonnet-20240229-v1:0",
    },
    "google_vertex_ai": {
        "name": "Google Vertex AI",
        "api_base": "https://us-central1-aiplatform.googleapis.com",
        "models_endpoint": None,  # Vertex AI requires a more complex API for model listing, using SDK.
        "loader": _vertex_ai_loader,
        "default_model": "gemini-pro",
    },
    "nvidia": {
        "name": "NVIDIA AI Foundation Models",
        "api_base": "https://api.nvcf.nvidia.com/v2/nvcf/infer",
        "models_endpoint": None,  # NVIDIA AI Foundation Models require a specific API for model listing, using SDK.
        "loader": _nvidia_loader,
        "default_model": "nv-llama2-70b",
    },
    "replicate": {
        "name": "Replicate",
        "api_base": "https://api.replicate.com/v1",
        "models_endpoint": None,  # Replicate API requires listing models by owner/name, using SDK.
        "loader": _replicate_loader,
        "default_model": "meta/llama-2-70b-chat",
    },
    "aleph_alpha": {
        "name": "Aleph Alpha",
        "api_base": "https://api.aleph-alpha.com",
        "models_endpoint": None,  # Aleph Alpha API does not have a simple /models endpoint, using SDK.
        "loader": _aleph_alpha_loader,
        "default_model": "luminous-extended",
    },
    "ai21": {
        "name": "AI21 Labs",
        "api_base": "https://api.ai21.com/studio/v1",
        "models_endpoint": None,  # AI21 Labs API does not have a simple /models endpoint, using static list.
        "loader": _ai21_static_loader,
        "default_model": "j2-ultra",
    },
    "baseten": {
        "name": "Baseten",
        "api_base": "https://model.baseten.co",
        "models_endpoint": None,  # Baseten API does not have a simple /models endpoint, using static list.
        "loader": _baseten_static_loader,
        "default_model": "llama-2-7b-chat",
    },
    "runpod": {
        "name": "RunPod",
        "api_base": "https://api.runpod.ai/v2",
        "models_endpoint": None,  # RunPod API requires SDK for model listing.
        "loader": _runpod_loader,
        "default_model": "llama2-70b-chat",
    },
    "modal": {
        "name": "Modal Labs",
        "api_base": "https://api.modal.com/v1",
        "models_endpoint": None,  # Modal Labs API requires CLI for model listing.
        "loader": _modal_loader,
        "default_model": "modal-llama-2-70b-chat",
    },
    "vllm": {
        "name": "vLLM",
        "api_base": "http://localhost:8000/v1", # Assuming local vLLM instance
        "models_endpoint": None,  # vLLM typically runs locally and model listing depends on its setup, using static list.
        "loader": _vllm_static_loader,
        "default_model": "llama-2-7b-chat",
    },
    "llama_cpp": {
        "name": "llama.cpp",
        "api_base": "http://localhost:8080/v1", # Assuming local llama.cpp instance
        "models_endpoint": None,  # llama.cpp typically runs locally and model listing depends on its setup, using static list.
        "loader": _llama_cpp_static_loader,
        "default_model": "llama-2-7b-chat.Q4_0.gguf",
    },
    "ollama": {
        "name": "Ollama",
        "api_base": "http://localhost:11434/api",
        "models_endpoint": "http://localhost:11434/api/tags",  # Ollama API endpoint for listing models.
        "loader": _ollama_loader,
        "default_model": "llama2",
    },
        "local_llm": {
            "name": "Local LLM (Generic)",
            "api_base": "http://localhost:8000/v1",
            "models_endpoint": None,  # Local LLM model listing is highly dependent on the specific local setup and cannot be dynamically fetched generically.
            "loader": _generic_loader,
            "default_model": "local-model",
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
