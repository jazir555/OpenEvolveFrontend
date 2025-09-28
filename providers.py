from typing import Dict, Any

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "OpenAI": {"base": "https://api.openai.com/v1", "model": "gpt-4o"},
    "Anthropic": {
        "base": "https://api.anthropic.com/v1",
        "model": "claude-3-opus-20240229",
    },
    "Google": {
        "base": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-1.5-pro",
    },
    "OpenRouter": {"base": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o"},
}
