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