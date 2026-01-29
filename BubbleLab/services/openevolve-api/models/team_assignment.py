"""
Flexible Team LLM Assignment System

Allows arbitrary LLM/vLLM assignment to any team (blue/red/judge) with
unified credential management from OpenEvolve config and BubbleLab credentials.

Key Features:
- Arbitrary LLM assignment to teams
- vLLM distinction for visual workflows
- Unified credential management
- Credential verification
- Team composition flexibility
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Literal
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    TOGETHER = "together"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    OPENAI_LIKE = "openai-like"  # Compatible APIs (vLLM, Ollama, etc.)
    CUSTOM = "custom"


class LLMCapability(str, Enum):
    """LLM capabilities"""
    TEXT = "text"  # Text generation only
    VISION = "vision"  # Multimodal (vLLM)
    CODE = "code"  # Code generation optimized
    MATH = "math"  # Mathematical reasoning
    REASONING = "reasoning"  # Complex reasoning
    TOOL_USE = "tool_use"  # Function calling
    AGENTIC = "agentic"  # Agentic behavior


class LLMModel(BaseModel):
    """
    LLM model definition with capabilities
    """
    provider: LLMProvider
    model_id: str = Field(..., description="Model identifier (e.g., 'gpt-4', 'claude-3-opus')")
    name: str = Field(..., description="Human-readable model name")
    capabilities: List[LLMCapability] = Field(
        default_factory=lambda: [LLMCapability.TEXT],
        description="Model capabilities"
    )
    max_tokens: int = Field(default=4096, ge=1, le=1000000)
    supports_streaming: bool = Field(default=True)
    supports_function_calling: bool = Field(default=False)
    is_vision: bool = Field(default=False, description="Is this a vision/multimodal model?")

    # Optional pricing
    input_price_per_1k: Optional[float] = Field(None, description="Input price per 1K tokens")
    output_price_per_1k: Optional[float] = Field(None, description="Output price per 1K tokens")

    @validator('capabilities')
    def validate_capability_consistency(cls, v, values):
        """Ensure vision capability is set if is_vision is True"""
        if values.get('is_vision') and LLMCapability.VISION not in v:
            v.append(LLMCapability.VISION)
        return v


class CredentialSource(str, Enum):
    """Where credentials come from"""
    OPENEVOLVE_CONFIG = "openevolve_config"  # From OpenEvolve .env/config
    BUBBLELAB_CREDENTIALS = "bubblelab_credentials"  # From BubbleLab credentials API
    USER_PROVIDED = "user_provided"  # Provided at runtime


class LLMCredential(BaseModel):
    """
    LLM credentials with verification status
    """
    credential_id: str
    provider: LLMProvider
    api_key: str = Field(..., description="Encrypted API key")
    source: CredentialSource
    verified: bool = Field(default=False)
    verified_at: Optional[str] = None  # ISO 8601 timestamp
    last_used: Optional[str] = None  # ISO 8601 timestamp
    model_permissions: List[str] = Field(
        default_factory=list,
        description="Models this credential can access"
    )

    # Optional additional auth parameters
    api_base: Optional[str] = Field(None, description="Custom API base URL (for OpenAI-compatible APIs)")
    organization_id: Optional[str] = Field(None, description="Organization ID (for OpenAI)")
    project_id: Optional[str] = Field(None, description="Project ID (for Google/Vertex)")
    region: Optional[str] = Field(None, description="Region (for some providers)")


class TeamRole(str, Enum):
    """Standard team roles"""
    BLUE_TEAM = "blue"  # Generates solutions
    RED_TEAM = "red"  # Attacks/evaluates solutions
    JUDGE = "judge"  # Evaluates and decides
    OBSERVER = "observer"  # Watches and learns
    ARBITER = "arbitrator"  # Resolves disagreements


class TeamMemberLLM(BaseModel):
    """
    LLM assigned to a team
    """
    member_id: str
    llm: LLMModel
    credential_id: Optional[str] = Field(None, description="Credential to use (None = use default)")
    role: TeamRole = Field(..., description="Team role")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=1000000)

    # Optional personality/prompt customization
    system_prompt: Optional[str] = Field(
        None,
        description="Custom system prompt for this LLM in this team context"
    )
    personality: Optional[str] = Field(
        None,
        description="Personality preset (e.g., 'critical', 'creative', 'conservative')"
    )

    # Performance tracking
    total_requests: int = Field(default=0)
    successful_requests: int = Field(default=0)
    average_latency_ms: Optional[float] = Field(None)


class Team(BaseModel):
    """
    A team with multiple LLM members
    """
    team_id: str
    name: str
    description: Optional[str] = None
    members: List[TeamMemberLLM] = Field(
        default_factory=list,
        description="LLMs assigned to this team"
    )

    # Team strategy
    voting_strategy: Literal["consensus", "majority", "weighted", "leader_decides"] = "consensus"
    quorum_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fraction of team needed for decision"
    )

    # Team composition rules
    require_vision_for_design: bool = Field(
        default=True,
        description="Require at least one vLLM for design tasks"
    )
    require_diverse_providers: bool = Field(
        default=False,
        description="Require LLMs from different providers"
    )


class TeamAssignmentRequest(BaseModel):
    """
    Request to assign LLM to a team
    """
    team_id: str
    llm_provider: LLMProvider
    llm_model_id: str
    role: TeamRole
    temperature: float = 0.7
    max_tokens: int = 4096
    credential_id: Optional[str] = None
    system_prompt: Optional[str] = None


class CredentialVerificationRequest(BaseModel):
    """
    Request to verify LLM credentials
    """
    credential_id: Optional[str] = Field(None, description="Existing credential to verify")
    provider: LLMProvider
    api_key: str = Field(..., description="API key to verify")
    api_base: Optional[str] = Field(None, description="Custom API base URL")
    model_to_test: Optional[str] = Field(None, description="Model to test credential with")


class CredentialVerificationResponse(BaseModel):
    """
    Result of credential verification
    """
    verified: bool
    credential_id: Optional[str] = None
    message: str
    test_model: Optional[str] = None
    latency_ms: Optional[float] = None
    available_models: Optional[List[str]] = None


# Predefined LLM catalog with capabilities
PREDEFINED_LLM_MODELS = {
    # OpenAI
    "gpt-4": LLMModel(
        provider=LLMProvider.OPENAI,
        model_id="gpt-4",
        name="GPT-4",
        capabilities=[LLMCapability.TEXT, LLMCapability.CODE, LLMCapability.REASONING],
        supports_function_calling=True,
        input_price_per_1k=0.03,
        output_price_per_1k=0.06,
    ),
    "gpt-4-turbo": LLMModel(
        provider=LLMProvider.OPENAI,
        model_id="gpt-4-turbo-preview",
        name="GPT-4 Turbo",
        capabilities=[LLMCapability.TEXT, LLMCapability.CODE, LLMCapability.REASONING, LLMCapability.VISION],
        is_vision=True,
        supports_function_calling=True,
        input_price_per_1k=0.01,
        output_price_per_1k=0.03,
    ),
    "gpt-4-vision": LLMModel(
        provider=LLMProvider.OPENAI,
        model_id="gpt-4-vision-preview",
        name="GPT-4 Vision",
        capabilities=[LLMCapability.TEXT, LLMCapability.VISION],
        is_vision=True,
        input_price_per_1k=0.01,
        output_price_per_1k=0.03,
    ),
    "gpt-3.5-turbo": LLMModel(
        provider=LLMProvider.OPENAI,
        model_id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        capabilities=[LLMCapability.TEXT, LLMCapability.CODE],
        supports_function_calling=True,
        input_price_per_1k=0.0005,
        output_price_per_1k=0.0015,
    ),

    # Anthropic
    "claude-3-opus": LLMModel(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-3-opus-20240229",
        name="Claude 3 Opus",
        capabilities=[LLMCapability.TEXT, LLMCapability.CODE, LLMCapability.REASONING, LLMCapability.VISION],
        is_vision=True,
        max_tokens=200000,
        supports_function_calling=True,  # Claude 3 supports tool use
        input_price_per_1k=0.015,
        output_price_per_1k=0.075,
    ),
    "claude-3-sonnet": LLMModel(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-3-sonnet-20240229",
        name="Claude 3 Sonnet",
        capabilities=[LLMCapability.TEXT, LLMCapability.CODE, LLMCapability.VISION],
        is_vision=True,
        max_tokens=200000,
        input_price_per_1k=0.003,
        output_price_per_1k=0.015,
    ),
    "claude-3-haiku": LLMModel(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-3-haiku-20240307",
        name="Claude 3 Haiku",
        capabilities=[LLMCapability.TEXT, LLMCapability.CODE, LLMCapability.VISION],
        is_vision=True,
        max_tokens=200000,
        input_price_per_1k=0.00025,
        output_price_per_1k=0.00125,
    ),

    # Google
    "gemini-pro": LLMModel(
        provider=LLMProvider.GOOGLE,
        model_id="gemini-pro",
        name="Gemini Pro",
        capabilities=[LLMCapability.TEXT, LLMCapability.REASONING, LLMCapability.VISION],
        is_vision=True,
        input_price_per_1k=0.00025,
        output_price_per_1k=0.0005,
    ),
    "gemini-ultra": LLMModel(
        provider=LLMProvider.GOOGLE,
        model_id="gemini-ultra",
        name="Gemini Ultra",
        capabilities=[LLMCapability.TEXT, LLMCapability.CODE, LLMCapability.REASONING, LLMCapability.VISION],
        is_vision=True,
        input_price_per_1k=0.00175,
        output_price_per_1k=0.0021,
    ),

    # Groq (Fast inference)
    "llama3-70b-groq": LLMModel(
        provider=LLMProvider.GROQ,
        model_id="llama3-70b-8192",
        name="Llama 3 70B (Groq)",
        capabilities=[LLMCapability.TEXT, LLMCapability.CODE, LLMCapability.REASONING],
        supports_streaming=True,
        input_price_per_1k=0.0000?  # Groq pricing varies
        output_price_per_1k=0.0000?,
    ),

    # DeepSeek
    "deepseek-coder": LLMModel(
        provider=LLMProvider.DEEPSEEK,
        model_id="deepseek-coder",
        name="DeepSeek Coder",
        capabilities=[LLMCapability.TEXT, LLMCapability.CODE],
        max_tokens=4096,
    ),

    # OpenRouter (aggregator)
    "openrouter-mix": LLMModel(
        provider=LLMProvider.OPENROUTER,
        model_id="anthropic/claude-3-opus",
        name="Claude 3 Opus (via OpenRouter)",
        capabilities=[LLMCapability.TEXT, LLMCapability.CODE, LLMCapability.REASONING],
    ),
}


def get_available_llms(
    provider: Optional[LLMProvider] = None,
    capability: Optional[LLMCapability] = None,
    vision_only: bool = False,
) -> List[LLMModel]:
    """
    Get available LLMs with optional filtering

    Args:
        provider: Filter by provider
        capability: Filter by capability
        vision_only: Only return vLLMs

    Returns:
        List of available LLM models
    """
    llms = list(PREDEFINED_LLM_MODELS.values())

    if provider:
        llms = [llm for llm in llms if llm.provider == provider]

    if capability:
        llms = [llm for llm in llms if capability in llm.capabilities]

    if vision_only:
        llms = [llm for llm in llms if llm.is_vision]

    return llms


def get_vision_llms() -> List[LLMModel]:
    """Get all vLLMs (models with vision capability)"""
    return get_available_llms(vision_only=True)


def group_llms_by_capability() -> Dict[str, List[LLMModel]]:
    """
    Group LLMs by capability for UI organization

    Returns:
        Dict mapping capability names to lists of LLMs
    """
    groups = {}

    for llm in PREDEFINED_LLM_MODELS.values():
        if llm.is_vision:
            key = "Vision/Multimodal (vLLM)"
        elif LLMCapability.CODE in llm.capabilities:
            key = "Code Generation"
        elif LLMCapability.REASONING in llm.capabilities:
            key = "Reasoning & Analysis"
        else:
            key = "General Purpose"

        if key not in groups:
            groups[key] = []
        groups[key].append(llm)

    return groups
