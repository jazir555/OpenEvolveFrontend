"""
OpenEvolve Protocol Improver - A Streamlit application for protocol improvement using LLMs.

This module provides a comprehensive interface for improving protocols and standard operating
procedures (SOPs) through two main approaches:

1. Evolution-based Improvement: Uses a single LLM provider to iteratively refine protocols
2. Adversarial Testing: Employs multiple LLM providers in a red team/blue team approach
   to identify vulnerabilities and generate hardened protocols

Key Features:
- Support for 34+ LLM providers including OpenAI, Anthropic, Google Gemini, and more
- Real-time protocol evaluation and improvement
- Cost estimation and token tracking
- Thread-safe operations for concurrent model evaluation
- Comprehensive logging and result visualization

The application uses Streamlit for the web interface and provides both single-provider
evolution and multi-provider adversarial testing capabilities.
"""

import functools
import json
import math
import os
import random
import re
import threading
import time
import traceback
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import streamlit as st
except ImportError:
    print("streamlit package not found. Please install it with 'pip install streamlit'")
    st = None

try:
    import requests
except ImportError:
    if st is not None:
        st.error("requests package not found. Please install it with 'pip install requests'")
    else:
        print("requests package not found. Please install it with 'pip install requests'")
    requests = None

# Graceful fallback for streamlit_tags
try:
    from streamlit_tags import st_tags
    HAS_STREAMLIT_TAGS = True
except ImportError:
    HAS_STREAMLIT_TAGS = False
    st.error("streamlit_tags package not found. Please install it with 'pip install streamlit-tags' for full functionality.")

# ------------------------------------------------------------------
# 0. Streamlit page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="OpenEvolve Protocol Improver (34 providers + Adversarial Testing)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Constants and Prompts
# ------------------------------------------------------------------

JSON_RE = re.compile(r"\{[\s\S]*\}")
FENCE_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
def safe_int(x, d=0):
    return d if x is None or not isinstance(x, (int, float, str)) or str(x) == '' else int(float(x))

def safe_float(x, d=0.0):
    return d if x is None or not isinstance(x, (int, float, str)) or str(x) == '' else float(x)

# Define adversarial prompts as constants for clarity and maintainability.
APPROVAL_PROMPT = (
    "You are a security auditor. Your task is to evaluate the provided Standard Operating Procedure (SOP). "
    "Return a single, STRICT JSON object and nothing else. The JSON object must have the following keys:\n"
    '{"verdict": "APPROVED" | "REJECTED", "score": 0-100, "reasons": ["..."], "notes": "..."}\n'
    "- `verdict`: Your final judgment. 'APPROVED' if the SOP is robust, clear, and secure. 'REJECTED' otherwise.\n"
    "- `score`: An integer from 0 to 100 reflecting the overall quality, security, and robustness to abuse.\n"
    "- `reasons`: A list of concise strings explaining the key factors behind your verdict and score.\n"
    "- `notes`: Any additional commentary or observations.\n"
    "Do not include any text, explanation, or formatting outside of the single JSON object."
)

RED_TEAM_CRITIQUE_PROMPT = (
    "You are an uncompromising red-team security auditor. Your goal is to find every possible flaw in the provided SOP. "
    "Analyze it for vulnerabilities, logical gaps, ambiguities, edge cases, undefined responsibilities, missing "
    "preconditions, unsafe defaults, and potential paths for abuse or malicious exploitation. "
    "Return a single, STRICT JSON object and nothing else with the following structure:\n"
    '{"issues": [{"title": "...", "severity": "low|medium|high|critical", "category": "...", "detail": "...", '
    '"reproduction": "...", "exploit_paths": ["..."], "mitigations": ["..."]}], '
    '"summary": "...", "overall_risk": "low|medium|high|critical"}'
)

BLUE_TEAM_PATCH_PROMPT = (
    "You are a meticulous blue-team security engineer. Your task is to fix the vulnerabilities identified in the "
    "provided critiques and produce a hardened, fully rewritten SOP. Incorporate the critiques to make the protocol "
    "explicit, verifiable, and enforce the principle of least privilege. Add preconditions, acceptance tests, rollback "
    "steps, monitoring, auditability, and incident response procedures where applicable. "
    "Return a single, STRICT JSON object and nothing else with the following keys:\n"
    '{"sop": "<the complete improved SOP in Markdown>", "changelog": ["..."], "residual_risks": ["..."], '
    '"mitigation_matrix": [{"issue": "...", "fix": "...", "status": "resolved|mitigated|wontfix"}]}'
)

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _now_ms() -> int:
    """Get current time in milliseconds since epoch.
    
    Returns:
        int: Current timestamp in milliseconds
    """
    return int(time.time() * 1000)

def _rand_jitter_ms(base: int = 250, spread: int = 500) -> float:
    """Generate random jitter time in seconds for retry backoff.
    
    Args:
        base: Base milliseconds value
        spread: Random spread in milliseconds
        
    Returns:
        float: Random jitter time in seconds
    """
    return (base + random.randint(0, spread)) / 1000.0

def _hash_text(s: str) -> str:
    """Generate a short hash of text for comparison purposes.
    
    Args:
        s: Text to hash
        
    Returns:
        str: First 16 characters of SHA256 hash
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def _approx_tokens(txt: str) -> int:
    """Approximate token count using character length.
    
    Args:
        txt: Text to estimate tokens for
        
    Returns:
        int: Estimated token count (minimum 1)
    """
    # A conservative approximation: 1 token ~ 4 chars in English. Clamp to a minimum of 1.
    return max(1, math.ceil(len(txt) / 4))

def _safe_json_loads(s: str) -> Optional[dict]:
    """Safely parse JSON string, returning None on failure.
    
    Args:
        s: JSON string to parse
        
    Returns:
        Optional[dict]: Parsed JSON object or None if invalid
    """
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None

def _extract_json_block(txt: str) -> Optional[dict]:
    """Extract JSON object from text, handling fenced code blocks.
    
    Args:
        txt: Text containing potential JSON
        
    Returns:
        Optional[dict]: Extracted JSON object or None if not found
    """
    if not txt or not isinstance(txt, str):
        return None
    # Try to find a JSON block fenced with ```json
    m = FENCE_RE.search(txt)
    if m:
        cand = _safe_json_loads(m.group(1).strip())
        if cand is not None:
            return cand
    # If not found, try to find any substring that looks like a JSON object
    m2 = JSON_RE.search(txt)
    if m2:
        cand = _safe_json_loads(m2.group(0))
        if cand is not None:
            return cand
    return None

def _clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))

def _parse_price_per_million(v: Any) -> Optional[float]:
    """
    Safely parses OpenRouter pricing fields (e.g., "0.15", 0.15, "N/A")
    into a float representing price per 1M tokens, or None if invalid.
    """
    if v is None or isinstance(v, (dict, list)):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def _cost_estimate(prompt_toks: int, completion_toks: int, ppm_prompt: Optional[float], ppm_comp: Optional[float]) -> float:
    """Calculates the estimated cost for a given number of tokens and prices per million."""
    cost = 0.0
    if ppm_prompt is not None and isinstance(ppm_prompt, (int, float)):
        cost += (prompt_toks / 1_000_000.0) * ppm_prompt
    if ppm_comp is not None and isinstance(ppm_comp, (int, float)):
        cost += (completion_toks / 1_000_000.0) * ppm_comp
    return cost

# ------------------------------------------------------------------
# 1. Generic helper – fetch model lists with tiny caching layer
# ------------------------------------------------------------------

@functools.lru_cache(maxsize=128)
def _cached_get(url: str, bearer: str | None = None, timeout: int = 10) -> dict | list:
    headers = {}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ------------------------------------------------------------------
# 2. Loader helpers for each provider that exposes an endpoint
# ------------------------------------------------------------------

def _openai_style_loader(url: str, api_key: str | None = None) -> List[str]:
    try:
        data = _cached_get(url, bearer=api_key)
        # Handle cases where data is a list (e.g., Together) or a dict with a 'data' key
        # (e.g., OpenAI)
        models_list = data if isinstance(data, list) else data.get("data", [])
        return sorted([m["id"] for m in models_list
                      if isinstance(m, dict) and "id" in m])
    except Exception as e:
        st.error(f"Error fetching models from {url}: {e}")
        return []

def _together_loader(api_key: str | None = None) -> List[str]:
    try:
        data = _cached_get("https://api.together.xyz/v1/models", bearer=api_key)
        return sorted([m["id"] for m in data
                      if isinstance(m, dict) and m.get("display_type") != "image"
                      and "id" in m])
    except Exception as e:
        st.error(f"Error fetching Together AI models: {e}")
        return []

def _fireworks_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.fireworks.ai/inference/v1/models", api_key)

def _groq_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.groq.com/openai/v1/models", api_key)

def _deepseek_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.deepseek.com/v1/models", api_key)

def _moonshot_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.moonshot.cn/v1/models", api_key)

def _baichuan_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.baichuan-ai.com/v1/models", api_key)

def _zhipu_loader(api_key: str | None = None) -> List[str]:
    # Zhipu's API might return models without an 'id', so we filter defensively
    try:
        data = _cached_get("https://open.bigmodel.cn/api/paas/v4/models",
                          bearer=api_key)
        return sorted([m["id"] for m in data.get("data", [])
                      if isinstance(m, dict) and "id" in m])
    except Exception as e:
        st.error(f"Error fetching Zhipu AI models: {e}")
        return []

def _minimax_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.minimax.chat/v1/models", api_key)

def _yi_loader(api_key: str | None = None) -> List[str]:
    return _openai_style_loader("https://api.lingyiwanwu.com/v1/models", api_key)


# ------------------------------------------------------------------
# 3. Central provider catalogue
# ------------------------------------------------------------------

PROVIDERS: dict[str, dict] = {
    # OpenAI official ------------------------------------------------
    "OpenAI": {
        "base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env": "OPENAI_API_KEY",
        "loader": lambda api_key=None: _openai_style_loader("https://api.openai.com/v1/models", api_key),
    },
    # Azure
    "Azure-OpenAI": {
        "base": "https://<your-resource>.openai.azure.com/openai/deployments/<deployment-name>",
        "model": "gpt-4o-mini", # This is often ignored as it's part of the deployment
        "env": "AZURE_OPENAI_API_KEY",
        "omit_model_in_payload": True, # Azure includes model in URL, not body
    },
    # Anthropic
    "Anthropic": {
        "base": "https://api.anthropic.com/v1", # Note: Anthropic has a non-OpenAI-compatible API
        "model": "claude-3-haiku-20240307",
        "env": "ANTHROPIC_API_KEY",
    },
    # Google Gemini
    "Google (Gemini)": {
        "base": "https://generativelanguage.googleapis.com/v1beta", # Note: Non-OpenAI-compatible API
        "model": "gemini-1.5-flash",
        "env": "GOOGLE_API_KEY",
    },
    # Mistral
    "Mistral": {
        "base": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
        "env": "MISTRAL_API_KEY",
        "loader": lambda api_key=None: _openai_style_loader("https://api.mistral.ai/v1/models", api_key),
    },
    # Cohere
    "Cohere": {
        "base": "https://api.cohere.ai/v1", # Note: Non-OpenAI-compatible API
        "model": "command-r-plus",
        "env": "COHERE_API_KEY",
    },
    # Perplexity
    "Perplexity": {
        "base": "https://api.perplexity.ai",
        "model": "llama-3.1-sonar-small-128k-online",
        "env": "PERPLEXITY_API_KEY",
        "loader": lambda api_key=None: _openai_style_loader("https://api.perplexity.ai/models", api_key),
    },
    # Groq
    "Groq": {
        "base": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "env": "GROQ_API_KEY",
        "loader": _groq_loader,
    },
    # Databricks
    "Databricks": {
        "base": "https://<workspace>.cloud.databricks.com/serving-endpoints", # Note: Custom API structure
        "model": "databricks-dbrx-instruct",
        "env": "DATABRICKS_TOKEN",
    },
    # Together AI
    "Together": {
        "base": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "env": "TOGETHER_API_KEY",
        "loader": _together_loader,
    },
    # Fireworks
    "Fireworks": {
        "base": "https://api.fireworks.ai/inference/v1",
        "model": "accounts/fireworks/models/llama-v3-8b-instruct",
        "env": "FIREWORKS_API_KEY",
        "loader": _fireworks_loader,
    },
    # Replicate
    "Replicate": {
        "base": "https://api.replicate.com/v1", # Note: Non-OpenAI-compatible API
        "model": "meta/meta-llama-3-8b-instruct",
        "env": "REPLICATE_API_TOKEN",
    },
    # Anyscale Endpoints
    "Anyscale": {
        "base": "https://api.endpoints.anyscale.com/v1",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "env": "ANYSCALE_API_KEY",
        "loader": lambda api_key=None: _openai_style_loader("https://api.endpoints.anyscale.com/v1/models", api_key),
    },
    # OpenRouter
    "OpenRouter": {
        "base": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "env": "OPENROUTER_API_KEY",
        "loader": lambda api_key=None: _openai_style_loader("https://openrouter.ai/api/v1/models", api_key),
    },
    # DeepInfra
    "DeepInfra": {
        "base": "https://api.deepinfra.com/v1/openai",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "env": "DEEPINFRA_API_KEY",
        "loader": lambda api_key=None: _openai_style_loader("https://api.deepinfra.com/v1/models", api_key),
    },
    # OctoAI
    "OctoAI": {
        "base": "https://text.octoai.run/v1",
        "model": "meta-llama-3-8b-instruct",
        "env": "OCTOAI_TOKEN",
    },
    # AI21
    "AI21": {
        "base": "https://api.ai21.com/studio/v1", # Note: Non-OpenAI-compatible API
        "model": "jamba-instruct",
        "env": "AI21_API_KEY",
    },
    # Aleph-Alpha
    "AlephAlpha": {
        "base": "https://api.aleph-alpha.com", # Note: Non-OpenAI-compatible API
        "model": "luminous-supreme-control",
        "env": "ALEPH_ALPHA_API_KEY",
    },
    # Bedrock variants
    "Bedrock-Claude": {
        "base": "https://bedrock-runtime.<region>.amazonaws.com", # Note: AWS Signature v4 auth needed
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "env": "AWS_SECRET_ACCESS_KEY",
    },
    "Bedrock-Titan": {
        "base": "https://bedrock-runtime.<region>.amazonaws.com",
        "model": "amazon.titan-text-express-v1",
        "env": "AWS_SECRET_ACCESS_KEY",
    },
    "Bedrock-Cohere": {
        "base": "https://bedrock-runtime.<region>.amazonaws.com",
        "model": "cohere.command-text-v14",
        "env": "AWS_SECRET_ACCESS_KEY",
    },
    "Bedrock-Jurassic": {
        "base": "https://bedrock-runtime.<region>.amazonaws.com",
        "model": "ai21.j2-ultra-v1",
        "env": "AWS_SECRET_ACCESS_KEY",
    },
    "Bedrock-Llama": {
        "base": "https://bedrock-runtime.<region>.amazonaws.com",
        "model": "meta.llama3-1-8b-instruct-v1:0",
        "env": "AWS_SECRET_ACCESS_KEY",
    },
    # Hugging Face Inference
    "HuggingFace": {
        "base": "https://api-inference.huggingface.co/models", # Note: Non-OpenAI-compatible API
        "model": "microsoft/DialoGPT-medium",
        "env": "HF_API_KEY",
    },
    # Local / self-hosted
    "Ollama": {
        "base": "http://localhost:11434/v1",
        "model": "llama3.1",
        "env": None,
    },
    "LM-Studio": {
        "base": "http://localhost:1234/v1",
        "model": "local-model",
        "env": None,
    },
    "vLLM": {
        "base": "http://localhost:8000/v1",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "env": None,
    },
    # SageMaker
    "SageMaker": {
        "base": "https://runtime.sagemaker.<region>.amazonaws.com/endpoints/<endpoint>/invocations", # AWS Sig v4
        "model": "jumpstart-dft-meta-textgeneration-llama-3-1-8b",
        "env": "AWS_SECRET_ACCESS_KEY",
    },
    # Cloudflare Workers AI
    "Cloudflare": {
        "base": "https://api.cloudflare.com/client/v4/accounts/<account>/ai/run", # Non-OpenAI format
        "model": "@cf/meta/llama-3.1-8b-instruct-awq",
        "env": "CLOUDFLARE_API_TOKEN",
    },
    # Vertex AI
    "VertexAI": {
        "base": "https://<region>-aiplatform.googleapis.com/v1/projects/<project>/locations/<region>/publishers/google/models", # Non-OpenAI
        "model": "gemini-1.5-flash",
        "env": "GOOGLE_APPLICATION_CREDENTIALS",
    },
    # Chinese providers
    "Moonshot": {
        "base": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "env": "MOONSHOT_API_KEY",
        "loader": _moonshot_loader,
    },
    "Baichuan": {
        "base": "https://api.baichuan-ai.com/v1",
        "model": "Baichuan3-Turbo",
        "env": "BAICHUAN_API_KEY",
        "loader": _baichuan_loader,
    },
    "Zhipu": {
        "base": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4",
        "env": "ZHIPU_API_KEY",
        "loader": _zhipu_loader,
    },
    "MiniMax": {
        "base": "https://api.minimax.chat/v1",
        "model": "abab6.5-chat",
        "env": "MINIMAX_API_KEY",
        "loader": _minimax_loader,
    },
    "Yi": {
        "base": "https://api.lingyiwanwu.com/v1",
        "model": "yi-large",
        "env": "YI_API_KEY",
        "loader": _yi_loader,
    },
    "DeepSeek": {
        "base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env": "DEEPSEEK_API_KEY",
        "loader": _deepseek_loader,
    },
    # Bring-your-own
    "Custom": {"base": "", "model": "", "env": None},
}


# ------------------------------------------------------------------
# 4. Session-state helpers
# ------------------------------------------------------------------

# Thread lock for safely updating shared session state from background threads.
# Use a lock to ensure thread-safe initialization of the session state lock
if "thread_lock" not in st.session_state:
    with threading.Lock():
        # Double-checked locking pattern to ensure thread safety
        if "thread_lock" not in st.session_state:
            st.session_state.thread_lock = threading.Lock()

DEFAULTS = {
    "provider": "OpenAI",
    "api_key": "",
    "base_url": PROVIDERS["OpenAI"]["base"],
    "model": PROVIDERS["OpenAI"]["model"],
    "extra_headers": "{}",
    "max_tokens": 4096,
    "temperature": 0.7,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "seed": "",
    "protocol_text": "",
    "system_prompt": "You are an assistant that makes the draft airtight, precise, and foolproof.",
    "evaluator_system_prompt": "You are a trivial evaluator that accepts everything.",
    "max_iterations": 20,
    "population_size": 1,
    "num_islands": 1,
    "checkpoint_interval": 5,
    "elite_ratio": 1.0,
    "exploration_ratio": 0.0,
    "exploitation_ratio": 0.0,
    "archive_size": 0,
    "evolution_running": False,
    "evolution_log": [],
    "evolution_current_best": "",
    "evolution_stop_flag": False,
    "openrouter_key": "",
    "red_team_models": [],
    "blue_team_models": [],
    "adversarial_running": False,
    "adversarial_results": {},
    "adversarial_status_message": "Idle.",
    "adversarial_log": [],
    "adversarial_stop_flag": False,
    "adversarial_cost_estimate_usd": 0.0,
    "adversarial_total_tokens_prompt": 0,
    "adversarial_total_tokens_completion": 0,
    "adversarial_min_iter": 3,
    "adversarial_max_iter": 10,
    "adversarial_confidence": 95,
    "adversarial_max_tokens": 8000,
    "adversarial_max_workers": 6,
    "adversarial_force_json": True,
    "adversarial_seed": "",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_defaults():
    p = st.session_state.provider
    if p in PROVIDERS:
        st.session_state.base_url = PROVIDERS[p].get("base", "")
        st.session_state.model = PROVIDERS[p].get("model") or ""
    st.session_state.api_key = ""
    st.session_state.extra_headers = "{}"


# ------------------------------------------------------------------
# 5. Adversarial Testing Functions (robust)
# ------------------------------------------------------------------

MODEL_META_BY_ID: Dict[str, Dict[str, Any]] = {}
MODEL_META_LOCK = threading.Lock()

@st.cache_data(ttl=600)
def get_openrouter_models(api_key: str) -> List[Dict]:
    """Fetch available models from OpenRouter (cached)."""
    if not api_key:
        return []
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        models = data.get("data", []) if isinstance(data, dict) else []
        return models
    except Exception as e:
        st.warning(f"Could not fetch OpenRouter models: {e}")
        return []

def _compose_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages

def _request_openrouter_chat(
    api_key: str,
    model_id: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    force_json: bool = False,
    seed: Optional[int] = None,
    req_timeout: int = 60,
    max_retries: int = 5,
) -> Tuple[str, int, int, float]:
    """
    Robust OpenRouter chat call with exponential backoff, jitter, and cost/token estimation.
    Returns: (content, prompt_tokens, completion_tokens, cost)
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/google/gemini-pro-builder", # Recommended by OpenRouter
        "X-Title": "OpenEvolve Protocol Improver", # Recommended by OpenRouter
    }
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": _clamp(temperature, 0.0, 2.0),
        "max_tokens": max_tokens,
        "top_p": _clamp(top_p, 0.0, 1.0),
        "frequency_penalty": _clamp(frequency_penalty, -2.0, 2.0),
        "presence_penalty": _clamp(presence_penalty, -2.0, 2.0),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=req_timeout)
            if r.status_code == 400:
                # HTTP 400 Bad Request - client error, don't retry
                last_err = Exception(f"HTTP 400 Bad Request: {r.text[:200]}...")
                break  # Break out of retry loop for client errors
            if r.status_code in {429, 500, 502, 503, 504}:
                sleep_s = (2 ** attempt) + _rand_jitter_ms()
                time.sleep(sleep_s)
                last_err = Exception(f"Transient error {r.status_code}: Retrying...")
                continue
            r.raise_for_status()
            data = r.json()
            
            # Safely access the first choice to prevent IndexError if "choices" is an empty list.
            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                content = choice.get("message", {}).get("content", "")
            else:
                content = ""

            usage = data.get("usage", {})
            p_tok = safe_int(usage.get("prompt_tokens"), _approx_tokens(json.dumps(messages)))
            c_tok = safe_int(usage.get("completion_tokens"), _approx_tokens(content or ""))

            with MODEL_META_LOCK:
                meta = MODEL_META_BY_ID.get(model_id, {})
            pricing = meta.get("pricing", {})
            ppm_prompt = _parse_price_per_million(pricing.get("prompt"))
            ppm_comp = _parse_price_per_million(pricing.get("completion"))
            cost = _cost_estimate(p_tok, c_tok, ppm_prompt, ppm_comp)
            return content or "", p_tok, c_tok, cost
        except Exception as e:
            last_err = e
            sleep_s = (2 ** attempt) + _rand_jitter_ms()
            time.sleep(sleep_s)
    raise RuntimeError(f"Request failed for {model_id} after {max_retries} attempts: {last_err}")

def analyze_with_model(
    api_key: str,
    model_id: str,
    sop: str,
    config: Dict,
    system_prompt: str,
    user_suffix: str = "",
    force_json: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyzes an SOP with a specific model, handling context limits and returning structured results.
    """
    try:
        max_tokens = safe_int(config.get("max_tokens"), 8000)
        user_prompt = f"Here is the Standard Operating Procedure (SOP):\n\n---\n\n{sop}\n\n---\n\n{user_suffix}"
        full_prompt_text = system_prompt + user_prompt

        with MODEL_META_LOCK:
            meta = MODEL_META_BY_ID.get(model_id, {})
        context_len = safe_int(meta.get("context_length"), 8192)
        prompt_toks_est = _approx_tokens(full_prompt_text)

        if prompt_toks_est + max_tokens >= context_len:
            err_msg = (f"ERROR[{model_id}]: Estimated prompt tokens ({prompt_toks_est}) + max_tokens ({max_tokens}) "
                       f"exceeds context window ({context_len}). Skipping.")
            return {"ok": False, "text": err_msg, "json": None, "ptoks": 0, "ctoks": 0, "cost": 0.0, "model_id": model_id}

        content, p_tok, c_tok, cost = _request_openrouter_chat(
            api_key=api_key, model_id=model_id,
            messages=_compose_messages(system_prompt, user_prompt),
            temperature=safe_float(config.get("temperature"), 0.7),
            top_p=safe_float(config.get("top_p"), 1.0),
            frequency_penalty=safe_float(config.get("frequency_penalty"), 0.0),
            presence_penalty=safe_float(config.get("presence_penalty"), 0.0),
            max_tokens=max_tokens, force_json=force_json, seed=seed,
        )
        json_content = _extract_json_block(content)
        return {"ok": True, "text": content, "json": json_content, "ptoks": p_tok, "ctoks": c_tok, "cost": cost, "model_id": model_id}
    except Exception as e:
        return {"ok": False, "text": f"ERROR[{model_id}]: {e}", "json": None, "ptoks": 0, "ctoks": 0, "cost": 0.0, "model_id": model_id}

def _safe_list(d: dict, key: str) -> List[Any]:
    v = d.get(key)
    return v if isinstance(v, list) else []

def _severity_rank(sev: str) -> int:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return order.get(str(sev).lower(), 0)

def _merge_consensus_sop(base_sop: str, blue_patches: List[dict]) -> Tuple[str, dict]:
    """
    Selects the best patch from the blue team based on coverage, resolution, and quality.
    """
    valid_patches = [p for p in blue_patches if p and (p.get("patch_json") or {}).get("sop", "").strip()]
    if not valid_patches:
        return base_sop, {"reason": "no_valid_patches_received", "score": -1}

    scored = []
    for patch in valid_patches:
        patch_json = patch.get("patch_json", {})
        sop_text = patch_json.get("sop", "").strip()

        mm = _safe_list(patch_json, "mitigation_matrix")
        residual = _safe_list(patch_json, "residual_risks")

        resolved = sum(1 for r in mm if str(r.get("status", "")).lower() == "resolved")
        mitigated = sum(1 for r in mm if str(r.get("status", "")).lower() == "mitigated")

        # Score based on resolved issues, then mitigated, penalize for residuals, and use length as tie-breaker
        coverage_score = (resolved * 2) + mitigated
        final_score = coverage_score - (len(residual) * 2)

        scored.append((final_score, resolved, len(sop_text), sop_text, patch.get("model")))

    if not scored:
        return base_sop, {"reason": "all_patches_were_empty_or_invalid", "score": -1}

    # Sort by score, then resolved count, then SOP length
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    best_score, best_resolved, _, best_sop, best_model = scored[0]
    diagnostics = {"reason": "best_patch_selected", "score": best_score, "resolved": best_resolved, "model": best_model}
    return best_sop, diagnostics

def _aggregate_red_risk(critiques: List[dict]) -> Dict[str, Any]:
    """Computes an aggregate risk score from all red-team critiques."""
    sev_weight = {"low": 1, "medium": 3, "high": 6, "critical": 12}
    total_weight, issue_count = 0, 0
    categories = {}

    valid_critiques = [c.get("critique_json") for c in critiques if c and c.get("critique_json")]

    for critique in valid_critiques:
        for issue in _safe_list(critique, "issues"):
            sev = str(issue.get("severity", "low")).lower()
            weight = sev_weight.get(sev, 1)
            total_weight += weight
            issue_count += 1
            cat = str(issue.get("category", "uncategorized")).lower()
            categories[cat] = categories.get(cat, 0) + weight

    avg_weight = (total_weight / max(1, issue_count)) if issue_count > 0 else 0
    return {"total_weight": total_weight, "avg_issue_weight": avg_weight, "categories": categories, "count": issue_count}

def _collect_model_configs(model_ids: List[str], max_tokens: int) -> Dict[str, Dict[str, Any]]:
    return {
        model_id: {
            "temperature": st.session_state.get(f"temp_{model_id}", 0.7),
            "top_p": st.session_state.get(f"topp_{model_id}", 1.0),
            "frequency_penalty": st.session_state.get(f"freqpen_{model_id}", 0.0),
            "presence_penalty": st.session_state.get(f"prespen_{model_id}", 0.0),
            "max_tokens": max_tokens,
        } for model_id in model_ids
    }

def _update_adv_log_and_status(msg: str):
    """Thread-safe way to update logs and status message."""
    with st.session_state.thread_lock:
        st.session_state.adversarial_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        st.session_state.adversarial_status_message = msg

def _update_adv_counters(ptoks: int, ctoks: int, cost: float):
    """Thread-safe way to update token and cost counters."""
    with st.session_state.thread_lock:
        st.session_state.adversarial_total_tokens_prompt += ptoks
        st.session_state.adversarial_total_tokens_completion += ctoks
        st.session_state.adversarial_cost_estimate_usd += cost

def check_approval_rate(
    api_key: str, red_team_models: List[str], sop_markdown: str, model_configs: Dict,
    seed: Optional[int], max_workers: int
) -> Dict[str, Any]:
    """Asks all red-team models for a final verdict on the SOP."""
    votes, scores, approved = [], [], 0
    total_ptoks, total_ctoks, total_cost = 0, 0, 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_model = {
            ex.submit(
                analyze_with_model, api_key, model_id, sop_markdown,
                model_configs.get(model_id, {}), APPROVAL_PROMPT, force_json=True, seed=seed
            ): model_id for model_id in red_team_models
        }
        for future in as_completed(future_to_model):
            model_id = future_to_model[future]
            res = future.result()
            total_ptoks += res["ptoks"]; total_ctoks += res["ctoks"]; total_cost += res["cost"]
            if res.get("ok") and res.get("json"):
                j = res["json"]
                verdict = str(j.get("verdict", "REJECTED")).upper()
                score = _clamp(safe_int(j.get("score"), 0), 0, 100)
                if verdict == "APPROVED":
                    approved += 1
                scores.append(score)
                votes.append({"model": model_id, "verdict": verdict, "score": score, "reasons": _safe_list(j, "reasons")})
            else:
                votes.append({"model": model_id, "verdict": "ERROR", "score": 0, "reasons": [res.get("text")]})

    rate = (approved / max(1, len(red_team_models))) * 100.0
    avg_score = (sum(scores) / max(1, len(scores))) if scores else 0
    return {"approval_rate": rate, "avg_score": avg_score, "votes": votes, "prompt_tokens": total_ptoks, "completion_tokens": total_ctoks, "cost": total_cost}

def run_adversarial_testing():
    """Main logic for the adversarial testing loop, designed to be run in a background thread."""
    try:
        # --- Initialization ---
        api_key = st.session_state.openrouter_key
        red_team = list(st.session_state.red_team_models or [])
        blue_team = list(st.session_state.blue_team_models or [])
        min_iter, max_iter = st.session_state.adversarial_min_iter, st.session_state.adversarial_max_iter
        confidence = st.session_state.adversarial_confidence
        max_tokens = st.session_state.adversarial_max_tokens
        json_mode = st.session_state.adversarial_force_json
        max_workers = st.session_state.adversarial_max_workers
        seed_str = str(st.session_state.adversarial_seed or "").strip()
        seed = None
        if seed_str:
            try:
                seed = int(float(seed_str))  # Handle floats by truncating to int
            except (ValueError, TypeError):
                pass  # Invalid input, keep seed as None

        with st.session_state.thread_lock:
            st.session_state.adversarial_log = []
            st.session_state.adversarial_stop_flag = False
            st.session_state.adversarial_total_tokens_prompt = 0
            st.session_state.adversarial_total_tokens_completion = 0
            st.session_state.adversarial_cost_estimate_usd = 0.0

        model_configs = _collect_model_configs(red_team + blue_team, max_tokens)
        current_sop = st.session_state.protocol_text
        base_hash = _hash_text(current_sop)
        results, sop_hashes = [], [base_hash]
        iteration, approval_rate = 0, 0.0

        _update_adv_log_and_status(f"Start: {len(red_team)} red / {len(blue_team)} blue | seed={seed} | base_hash={base_hash}")

        # --- Main Loop ---
        while iteration < max_iter and not st.session_state.adversarial_stop_flag:
            iteration += 1
            _update_adv_log_and_status(f"Iteration {iteration}/{max_iter}: Starting red team analysis.")

            # --- RED TEAM: CRITIQUES ---
            critiques_raw = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(analyze_with_model, api_key, m, current_sop,
                                    model_configs.get(m,{}), RED_TEAM_CRITIQUE_PROMPT,
                                    force_json=json_mode, seed=seed): m for m in red_team}
                for fut in as_completed(futures):
                    res = fut.result()
                    _update_adv_counters(res['ptoks'], res['ctoks'], res['cost'])
                    if not res.get("ok") or not res.get("json"):
                        _update_adv_log_and_status(f"🔴 {res['model_id']}: Invalid response. Details: {res.get('text', 'N/A')}")
                    critiques_raw.append({"model": res['model_id'], "critique_json": res.get("json"), "raw_text": res.get("text")})

            agg_risk = _aggregate_red_risk(critiques_raw)
            if agg_risk['count'] == 0:
                _update_adv_log_and_status(f"Iteration {iteration}: Red team found no exploitable issues. Checking for approval.")
            else:
                _update_adv_log_and_status(f"Iteration {iteration}: Red team found {agg_risk['count']} issues. Starting blue team patching.")

            # --- BLUE TEAM: PATCHING ---
            blue_patches_raw = []
            valid_critiques_json = [c['critique_json'] for c in critiques_raw if c.get('critique_json')]
            critique_block = json.dumps({"critiques": valid_critiques_json}, ensure_ascii=False, indent=2)

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(analyze_with_model, api_key, m, current_sop,
                                    model_configs.get(m,{}), BLUE_TEAM_PATCH_PROMPT,
                                    user_suffix="\n\nCRITIQUES TO ADDRESS:\n" + critique_block,
                                    force_json=True, seed=seed): m for m in blue_team}
                for fut in as_completed(futures):
                    res = fut.result()
                    _update_adv_counters(res['ptoks'], res['ctoks'], res['cost'])
                    if not res.get("ok") or not res.get("json") or not res.get("json", {}).get("sop","").strip():
                         _update_adv_log_and_status(f"🔵 {res['model_id']}: Invalid or empty patch received. Details: {res.get('text', 'N/A')}")
                    blue_patches_raw.append({"model": res['model_id'], "patch_json": res.get("json"), "raw_text": res.get("text")})

            next_sop, consensus_diag = _merge_consensus_sop(current_sop, blue_patches_raw)
            _update_adv_log_and_status(f"Iteration {iteration}: Consensus SOP generated (Best patch from '{consensus_diag.get('model', 'N/A')}'). Starting approval check.")

            # --- APPROVAL CHECK ---
            eval_res = check_approval_rate(api_key, red_team, next_sop, model_configs, seed, max_workers)
            approval_rate = eval_res["approval_rate"]
            _update_adv_counters(eval_res['prompt_tokens'], eval_res['completion_tokens'], eval_res['cost'])
            _update_adv_log_and_status(f"Iteration {iteration}: Approval rate: {approval_rate:.1f}%, Avg Score: {eval_res['avg_score']:.1f}")

            results.append({
                "iteration": iteration, "critiques": critiques_raw, "patches": blue_patches_raw,
                "current_sop": next_sop, "approval_check": eval_res, "agg_risk": agg_risk, "consensus": consensus_diag
            })
            current_sop = next_sop

            # --- Stagnation Check ---
            current_hash = _hash_text(current_sop)
            if len(sop_hashes) > 1 and current_hash == sop_hashes[-1] and current_hash == sop_hashes[-2]:
                _update_adv_log_and_status("⚠️ Stagnation detected: SOP has not changed for 2 iterations. Consider adjusting models or temperature.")
            sop_hashes.append(current_hash)

            if iteration >= min_iter and approval_rate >= confidence:
                _update_adv_log_and_status(f"✅ Success! Confidence threshold of {confidence}% reached after {iteration} iterations.")
                break
        # --- End of Loop ---

        if st.session_state.adversarial_stop_flag:
            _update_adv_log_and_status("⏹️ Process stopped by user.")
        elif iteration >= max_iter:
            _update_adv_log_and_status(f"🏁 Reached max iterations ({max_iter}). Final approval rate: {approval_rate:.1f}%")

        with st.session_state.thread_lock:
            st.session_state.adversarial_results = {
                "final_sop": current_sop, "iterations": results, "final_approval_rate": approval_rate,
                "cost_estimate_usd": st.session_state.adversarial_cost_estimate_usd,
                "tokens": {"prompt": st.session_state.adversarial_total_tokens_prompt, "completion": st.session_state.adversarial_total_tokens_completion},
                "log": list(st.session_state.adversarial_log), "seed": seed, "base_hash": base_hash
            }
            st.session_state.protocol_text = current_sop
            st.session_state.adversarial_running = False

    except Exception as e:
        # --- Global Error Handler ---
        tb_str = traceback.format_exc()
        error_message = f"💥 A critical error occurred: {e}\n{tb_str}"
        _update_adv_log_and_status(error_message)
        with st.session_state.thread_lock:
            st.session_state.adversarial_running = False
            if 'adversarial_results' not in st.session_state or not st.session_state.adversarial_results:
                st.session_state.adversarial_results = {}
            st.session_state.adversarial_results["critical_error"] = error_message
            # Ensure error is visible in UI by storing a simplified message
            st.session_state.adversarial_status_message = f"Error: {str(e)[:100]}..."


# ------------------------------------------------------------------
# 6. Sidebar – every provider + every knob
# ------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Provider Configuration")
    st.caption("Controls the 'Evolution' tab. Adversarial Testing always uses OpenRouter.")

    @st.cache_data(ttl=300)
    def _query_models(provider_key: str, api_key: str = ""):
        if provider_key not in PROVIDERS:
            return []
        loader = PROVIDERS[provider_key].get("loader")
        if loader:
            with st.spinner(f"Fetching models for {provider_key}..."):
                models = loader(api_key or None)
                if models:
                    return models
        # Fallback to the default model for the provider if loader fails or doesn't exist
        default_model = PROVIDERS[provider_key].get("model")
        return [default_model] if default_model else []

    provider = st.selectbox(
        "Provider", list(PROVIDERS.keys()), key="provider", on_change=reset_defaults,
        help="Choose the backend for the Evolution tab. Adversarial Testing always uses OpenRouter."
    )

    api_key_for_loader = st.session_state.openrouter_key if provider == "OpenRouter" else st.session_state.api_key
    model_options = _query_models(provider, api_key_for_loader)

    model_idx = 0
    if st.session_state.model in model_options:
        model_idx = model_options.index(st.session_state.model)
    elif model_options:
        st.session_state.model = model_options[0]  # Default to first in list if previous selection is invalid

    st.selectbox("Model", model_options, index=model_idx, key="model")
    st.text_input("API Key", type="password", key="api_key",
                 help=f"Leave empty to use env var: {PROVIDERS.get(provider, {}).get('env', 'Not specified')}")
    st.text_input("Base URL", key="base_url", help="Endpoint for chat/completions.")
    st.number_input("Max tokens", min_value=1, max_value=128_000, step=1, key="max_tokens")
    st.slider("Temperature", 0.0, 2.0, key="temperature")
    st.slider("Top-p", 0.0, 1.0, key="top_p")
    st.slider("Frequency penalty", -2.0, 2.0, key="frequency_penalty")
    st.slider("Presence penalty", -2.0, 2.0, key="presence_penalty")
    st.text_input("Seed (optional)", key="seed", help="Integer for deterministic sampling.")
    st.text_area("Extra headers (JSON dict)", height=80, key="extra_headers")
    st.markdown("---")
    st.subheader("Evolution Parameters")
    st.number_input("Max iterations", 1, 1000, key="max_iterations")
    st.number_input("Checkpoint interval", 1, 100, key="checkpoint_interval")
    # Disabled params from original code, kept for UI consistency
    st.number_input("Population size", 1, 100, key="population_size", disabled=True)
    st.number_input("Num islands", 1, 10, key="num_islands", disabled=True)
    st.number_input("Elite ratio", 0.0, 1.0, key="elite_ratio", disabled=True)
    st.number_input("Exploration ratio", 0.0, 1.0, key="exploration_ratio", disabled=True)
    st.number_input("Exploitation ratio", 0.0, 1.0, key="exploitation_ratio", disabled=True)
    st.number_input("Archive size", 0, 1000, key="archive_size", disabled=True)
    st.markdown("---")
    st.text_area("System prompt", height=120, key="system_prompt")
    st.text_area("Evaluator system prompt", height=120, key="evaluator_system_prompt", disabled=True)
    st.button("Reset to provider defaults", on_click=reset_defaults, use_container_width=True)

# ------------------------------------------------------------------
# 7. Main layout with tabs
# ------------------------------------------------------------------

st.title("🧬 OpenEvolve Protocol Improver")
tab1, tab2 = st.tabs(["Evolution", "Adversarial Testing"])

with tab1:
    st.text_area("Paste your draft protocol / procedure here:", height=300, key="protocol_text", disabled=st.session_state.adversarial_running)
    c1, c2 = st.columns(2)
    run_button = c1.button("🚀 Start Evolution", type="primary", disabled=st.session_state.evolution_running, use_container_width=True)
    stop_button = c2.button("⏹️ Stop Evolution", disabled=not st.session_state.evolution_running, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("📄 Current Best Protocol")
        proto_out = st.empty()
    with right:
        st.subheader("🔍 Logs")
        log_out = st.empty()

    # Display the current state from the session state
    with st.session_state.thread_lock:
        current_log = "\n".join(st.session_state.evolution_log)
        current_protocol = st.session_state.evolution_current_best or st.session_state.protocol_text

    log_out.code(current_log, language="text")
    proto_out.code(current_protocol, language="markdown")

    # If evolution is running, sleep for 1 second and then rerun to update the UI
    if st.session_state.evolution_running:
        time.sleep(1)
        st.rerun()

def render_adversarial_testing_tab():
    st.header("🔴🔵 Adversarial Testing with Multi-LLM Consensus")
    st.subheader("OpenRouter Configuration")
    openrouter_key = st.text_input("OpenRouter API Key", type="password", key="openrouter_key")
    if not openrouter_key:
        st.info("Enter your OpenRouter API key to enable model selection and testing.")
        return

    models = get_openrouter_models(openrouter_key)
    # Update global model metadata with thread safety
    for m in models:
        if isinstance(m, dict) and (mid := m.get("id")):
            with MODEL_META_LOCK:
                MODEL_META_BY_ID[mid] = m
    if not models:
        st.error("No models fetched. Check your OpenRouter key and connection.")
        return

    model_options = sorted([
        f"{m['id']} (Ctx: {m.get('context_length', 'N/A')}, "
        f"In: ${_parse_price_per_million(m.get('pricing', {}).get('prompt')) or 'N/A'}/M, "
        f"Out: ${_parse_price_per_million(m.get('pricing', {}).get('completion')) or 'N/A'}/M)"
        for m in models if isinstance(m, dict) and "id" in m
    ])

    st.subheader("Model Selection")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔴 Red Team (Critics)")
        if HAS_STREAMLIT_TAGS:
            red_team_selected_full = st_tags(
                label="Select models to find flaws.", text="Search models...", value=st.session_state.red_team_models,
                suggestions=model_options, key="red_team_select"
            )
            # Robust model ID extraction from descriptive string
            red_team_models = []
            for m in red_team_selected_full:
                # Extract model ID by splitting on first occurrence of " (" or using entire string
                # if not found
                if " (" in m:
                    model_id = m.split(" (")[0].strip()
                else:
                    model_id = m.strip()
                if model_id:
                    red_team_models.append(model_id)
            st.session_state.red_team_models = sorted(list(set(red_team_models)))
        else:
            st.warning("streamlit_tags not available. Using text input for model selection.")
            red_team_input = st.text_input("Enter Red Team models (comma-separated):", value=",".join(st.session_state.red_team_models))
            st.session_state.red_team_models = sorted(list(set([model.strip() for model in red_team_input.split(",") if model.strip()])))
    with col2:
        st.markdown("#### 🔵 Blue Team (Fixers)")
        if HAS_STREAMLIT_TAGS:
            blue_team_selected_full = st_tags(
                label="Select models to patch flaws.", text="Search models...", value=st.session_state.blue_team_models,
                suggestions=model_options, key="blue_team_select"
            )
            # Robust model ID extraction from descriptive string
            blue_team_models = []
            for m in blue_team_selected_full:
                # Extract model ID by splitting on first occurrence of " (" or using entire string
                # if not found
                if " (" in m:
                    model_id = m.split(" (")[0].strip()
                else:
                    model_id = m.strip()
                if model_id:
                    blue_team_models.append(model_id)
            st.session_state.blue_team_models = sorted(list(set(blue_team_models)))
        else:
            st.warning("streamlit_tags not available. Using text input for model selection.")
            blue_team_input = st.text_input("Enter Blue Team models (comma-separated):", value=",".join(st.session_state.blue_team_models))
            st.session_state.blue_team_models = sorted(list(set([model.strip() for model in blue_team_input.split(",") if model.strip()])))

    st.subheader("Testing Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input("Min iterations", 1, 50, key="adversarial_min_iter")
        st.number_input("Max iterations", 1, 200, key="adversarial_max_iter")
    with c2:
        st.slider("Confidence threshold (%)", 50, 100, key="adversarial_confidence", help="Stop if this % of Red Team approves the SOP.")
    with c3:
        st.number_input("Max tokens per model", 1000, 100000, key="adversarial_max_tokens")
        st.number_input("Max parallel workers", 1, 24, key="adversarial_max_workers")
    with c4:
        st.toggle("Force JSON mode", key="adversarial_force_json", help="Use model's built-in JSON mode if available. Increases reliability.")
        st.text_input("Deterministic seed", key="adversarial_seed", help="Integer for reproducible runs.")

    all_models = sorted(list(set(st.session_state.red_team_models + st.session_state.blue_team_models)))
    if all_models:
        with st.expander("Per-Model Configuration"):
            for model_id in all_models:
                st.markdown(f"**{model_id}**")
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.slider(f"Temp##{model_id}", 0.0, 2.0, 0.7, 0.1, key=f"temp_{model_id}")
                cc2.slider(f"Top-P##{model_id}", 0.0, 1.0, 1.0, 0.1, key=f"topp_{model_id}")
                cc3.slider(f"Freq Pen##{model_id}", -2.0, 2.0, 0.0, 0.1, key=f"freqpen_{model_id}")
                cc4.slider(f"Pres Pen##{model_id}", -2.0, 2.0, 0.0, 0.1, key=f"prespen_{model_id}")

    st.markdown("---")
    b1, b2, b3 = st.columns(3)
    if b1.button("🚀 Start Adversarial Testing", type="primary", use_container_width=True, disabled=st.session_state.adversarial_running):
        if not st.session_state.red_team_models or not st.session_state.blue_team_models:
            st.error("Select at least one Red Team and one Blue Team model.")
        elif not st.session_state.protocol_text.strip():
            st.error("Enter a protocol to test in the Evolution tab.")
        else:
            st.session_state.adversarial_running = True
            threading.Thread(target=run_adversarial_testing, daemon=True).start()
            # Removed unnecessary rerun - the UI will update automatically through session state changes

    if b2.button("⏹️ Stop Testing", use_container_width=True, disabled=not st.session_state.adversarial_running):
        st.session_state.adversarial_stop_flag = True
        st.warning("Stop signal sent. The current iteration will finish.")

    if b3.button("🧹 Clear Results", use_container_width=True):
        st.session_state.adversarial_results = {}
        st.session_state.adversarial_log = []
        st.session_state.adversarial_cost_estimate_usd = 0.0
        st.session_state.adversarial_total_tokens_prompt = 0
        st.session_state.adversarial_total_tokens_completion = 0
        # Removed unnecessary rerun - the UI will update automatically through session state changes

    # Live metrics and results display
    st.markdown("---")
    if st.session_state.adversarial_running:
        st.status(st.session_state.adversarial_status_message, expanded=True)

    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Estimated Cost (USD)", f"${st.session_state.adversarial_cost_estimate_usd:,.4f}")
    lc2.metric("Prompt Tokens", f"{st.session_state.adversarial_total_tokens_prompt:,}")
    lc3.metric("Completion Tokens", f"{st.session_state.adversarial_total_tokens_completion:,}")

    if st.session_state.adversarial_log:
        with st.expander("Live Log", expanded=not st.session_state.adversarial_results):
            st.code("\n".join(st.session_state.adversarial_log), language='text')

    if r := st.session_state.get("adversarial_results"):
        st.subheader("Results")
        if "critical_error" in r:
            st.error(f"The run failed with a critical error:")
            st.code(r['critical_error'])

        final_rate = r.get('final_approval_rate', 0)
        if final_rate >= st.session_state.adversarial_confidence:
            st.success(f"Final approval rate: {final_rate:.1f}% (Threshold: {st.session_state.adversarial_confidence}%)")
        else:
            st.warning(f"Final approval rate: {final_rate:.1f}% (Threshold: {st.session_state.adversarial_confidence}%)")

        with st.expander("View final hardened SOP"):
            st.code(r.get("final_sop", ""), language="markdown")
        with st.expander("View iteration-by-iteration details"):
            for iteration in r.get("iterations", []):
                st.markdown(f"### Iteration {iteration['iteration']}")
                ac = iteration.get("approval_check", {})
                st.write(f"Approval: **{ac.get('approval_rate', 0):.1f}%** | Avg Score: **{ac.get('avg_score', 0):.1f}**")
                st.json({"critiques": iteration.get("critiques",[]), "patches": iteration.get("patches",[])})

with tab2:
    render_adversarial_testing_tab()

# ------------------------------------------------------------------
# 8. Run logic for Evolution tab (Self-Contained Implementation)
# ------------------------------------------------------------------

def _request_openai_compatible_chat(
    api_key: str, base_url: str, model: str, messages: List, extra_headers: Dict,
    temperature: float, top_p: float, frequency_penalty: float, presence_penalty: float,
    max_tokens: int, seed: Optional[int], req_timeout: int = 60, max_retries: int = 5,
    provider: str = "OpenAI"
) -> str:
    # Define providers with non-OpenAI compatible APIs that are not supported
    UNSUPPORTED_PROVIDERS = {
        "Anthropic", "Google (Gemini)", "Cohere", "Replicate", "AI21", "AlephAlpha",
        "Bedrock-Claude", "Bedrock-Titan", "Bedrock-Cohere", "Bedrock-Jurassic", "Bedrock-Llama",
        "HuggingFace", "Cloudflare", "VertexAI", "Databricks", "SageMaker"
    }
    
    if provider in UNSUPPORTED_PROVIDERS:
        raise RuntimeError(f"Provider '{provider}' uses a non-OpenAI compatible API and is not supported in the Evolution tab. Please use an OpenAI-compatible provider like OpenAI, Azure-OpenAI, Mistral, etc.")
    url = base_url.rstrip('/') + "/chat/completions"
    headers = {"Content-Type": "application/json", **extra_headers}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
        "top_p": top_p, "frequency_penalty": frequency_penalty, "presence_penalty": presence_penalty,
    }
    if PROVIDERS.get(provider, {}).get("omit_model_in_payload"):
        payload.pop("model", None)
    if seed is not None:
        payload["seed"] = int(seed)

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=req_timeout)
            
            # Handle rate limiting and server errors with retry
            if r.status_code in {429, 500, 502, 503, 504}:
                sleep_s = (2 ** attempt) + _rand_jitter_ms()
                time.sleep(sleep_s)
                last_err = Exception(f"HTTP {r.status_code}: {r.text}")
                continue
                
            r.raise_for_status()
            
            # Handle non-JSON responses
            try:
                data = r.json()
            except json.JSONDecodeError as e:
                last_err = Exception(f"Invalid JSON response: {e} - Response: {r.text[:200]}...")
                time.sleep((2 ** attempt) + _rand_jitter_ms())
                continue
                
            # Safely access the response structure
            if not isinstance(data, dict):
                last_err = Exception(f"Unexpected response format: {type(data)} - Response: {data}")
                time.sleep((2 ** attempt) + _rand_jitter_ms())
                continue
                
            # Handle API-specific error formats
            if "error" in data:
                error_msg = data["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                last_err = Exception(f"API error: {error_msg}")
                time.sleep((2 ** attempt) + _rand_jitter_ms())
                continue
                
            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                content = choice.get("message", {}).get("content", "")
                if content is not None:
                    return content
                else:
                    last_err = Exception("Empty content in response choice")
            else:
                last_err = Exception("No choices in response")
                
            # If we get here, there was an issue with the response structure
            time.sleep((2 ** attempt) + _rand_jitter_ms())
                
        except requests.exceptions.ConnectionError as e:
            last_err = Exception(f"Connection error: {e}")
            time.sleep((2 ** attempt) + _rand_jitter_ms())
        except requests.exceptions.Timeout as e:
            last_err = Exception(f"Request timeout: {e}")
            time.sleep((2 ** attempt) + _rand_jitter_ms())
        except requests.exceptions.RequestException as e:
            last_err = Exception(f"Request failed: {e}")
            time.sleep((2 ** attempt) + _rand_jitter_ms())
        except Exception as e:
            last_err = e
            time.sleep((2 ** attempt) + _rand_jitter_ms())
            
    raise RuntimeError(f"Request failed after {max_retries} attempts for model {model}: {last_err}")

def run_evolution_internal():
    try:
        with st.session_state.thread_lock:
            st.session_state.evolution_log = []
            st.session_state.evolution_stop_flag = False
        current_protocol = st.session_state.protocol_text
        st.session_state.evolution_current_best = current_protocol

        def log_msg(msg):
            with st.session_state.thread_lock:
                st.session_state.evolution_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        log_msg(f"Starting evolution process with {st.session_state.provider}/{st.session_state.model}...")
        try:
            extra_hdrs = json.loads(st.session_state.extra_headers or "{}")
            if not isinstance(extra_hdrs, dict): raise json.JSONDecodeError("JSON is not a dictionary.", "", 0)
        except (ValueError, TypeError):
            log_msg("⚠️ Invalid Extra Headers JSON. Must be a dictionary. Using empty dict.")
            extra_hdrs = {}

        seed_str = str(st.session_state.seed or "").strip()
        seed = None
        if seed_str:
            try:
                seed = int(float(seed_str))  # Handle floats by truncating to int
            except (ValueError, TypeError):
                pass  # Invalid input, keep seed as None

        api_key_to_use = st.session_state.api_key
        provider_info = PROVIDERS.get(st.session_state.provider, {})
        if not api_key_to_use and (env_var := provider_info.get("env")):
            api_key_to_use = os.environ.get(env_var, "")
            if api_key_to_use: log_msg(f"Using API key from env var {env_var}.")

        if not api_key_to_use and provider_info.get("env"):
             log_msg(f"⚠️ No API key provided in UI or in env var {provider_info.get('env')}. Requests may fail.")

        consecutive_failures = 0
        max_consecutive_failures = 3  # Stop after 3 consecutive failures
        for i in range(st.session_state.max_iterations):
            if st.session_state.evolution_stop_flag:
                log_msg("Evolution stopped by user.")
                break
            log_msg(f"--- Iteration {i+1}/{st.session_state.max_iterations} ---")
            try:
                messages = _compose_messages(
                    st.session_state.system_prompt,
                    f"Current draft:\n\n---\n{current_protocol}\n---\n\nImprove it based on your instructions."
                )
                improved_protocol = _request_openai_compatible_chat(
                    api_key=api_key_to_use, base_url=st.session_state.base_url, model=st.session_state.model,
                    messages=messages, extra_headers=extra_hdrs, temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p, frequency_penalty=st.session_state.frequency_penalty,
                    presence_penalty=st.session_state.presence_penalty, max_tokens=st.session_state.max_tokens, seed=seed,
                    provider=st.session_state.provider
                )
                if improved_protocol and len(improved_protocol.strip()) > len(current_protocol) * 0.7:
                    log_msg(f"✅ Iteration {i+1} successful. Length: {len(improved_protocol.strip())}")
                    current_protocol = improved_protocol.strip()
                    st.session_state.evolution_current_best = current_protocol
                    consecutive_failures = 0  # Reset failure counter on success
                else:
                    log_msg(f"⚠️ Iteration {i+1} result rejected (too short or empty). Length: {len(improved_protocol.strip() if improved_protocol else '')}")
                    consecutive_failures += 1
                if (i + 1) % st.session_state.checkpoint_interval == 0:
                    log_msg(f"💾 Checkpoint at iteration {i+1}.")
            except Exception as e:
                log_msg(f"❌ ERROR in iteration {i+1}: {e}")
                consecutive_failures += 1
                time.sleep(2)
            
            # Stop if too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                log_msg(f"🛑 Stopping evolution due to {consecutive_failures} consecutive failures.")
                break

        log_msg("Evolution finished.")
        st.session_state.protocol_text = current_protocol  # Update the main protocol text
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        log_msg(f"💥 A critical error occurred in the evolution thread: {e}\n{tb_str}")
    finally:
        st.session_state.evolution_running = False

if stop_button:
    st.session_state.evolution_stop_flag = True
    st.warning("Stop signal sent. Evolution will stop after the current iteration.")
    # No rerun here, let the loop handle the UI update

if run_button:
    if st.session_state.protocol_text.strip():
        # Thread safety check to prevent multiple concurrent evolution threads
        if st.session_state.evolution_running:
            st.warning("Evolution is already running. Please wait for it to complete or stop it first.")
        else:
            st.session_state.evolution_running = True
            threading.Thread(target=run_evolution_internal, daemon=True).start()
            st.rerun()
    else:
        st.warning("Please paste a protocol before starting evolution.")
