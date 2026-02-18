"""Deterministic LLM integration stack (8-layer framework)."""

from .deps import ensure_local_dependencies

ensure_local_dependencies()

from .backends import BackendAdapter, BackendCapabilities, CallableLLM, CloudBackend, LocalBackend
from .consensus import ConsensusEngine
from .distributed import DistributedDeterminismCoordinator
from .examples import (
    CustomerSupportAgent,
    DeterministicCodeGenerator,
    LearningCustomerSupport,
    ScientificReasoningPipeline,
    TemporalKnowledgeLayer,
)
from .layers import (
    AtomicTask,
    ConstrainedGenerator,
    ContentValidator,
    ContentGeneration,
    ContextManager,
    DataRetrieval,
    DecompositionAdapter,
    FormalVerificationLayer,
    KnowledgeAdapter,
    LagrangeFilter,
    MatryoshkaClient,
    OptimizedWorkflow,
    ReproducibilityLayer,
    StreamingConstrainedGenerator,
    SmartContextManager,
)
from .llm import LLMConfig, BaseLLM, OpenAIChatLLM, AnthropicLLM, GoogleLLM, HFLocalLLM, build_llm
from .monitoring import CloudLLMMonitor, cloud_consensus, detect_divergence
from .multimodal import MultiModalDeterministicGenerator, MultiModalGenerator
from .pipeline import (
    DeterminismConfig,
    DeterminismResult,
    DeterministicPipeline,
    EnhancedDeterministicPipeline,
    FullDeterminismStack,
    HybridDeterministicSystem,
    ProductionDeterministicSystem,
    UltraDeterministicPipeline,
    generate_with_full_verification,
    verified_generation,
    verified_response,
)
from .routing import IntelligentModelRouter
from .security import SecurityLayer
from .utils import deterministic_seed, hash_prompt, similarity

__all__ = [
    "BackendAdapter",
    "BackendCapabilities",
    "CallableLLM",
    "CloudBackend",
    "LocalBackend",
    "ConsensusEngine",
    "DistributedDeterminismCoordinator",
    "CustomerSupportAgent",
    "DeterministicCodeGenerator",
    "LearningCustomerSupport",
    "ScientificReasoningPipeline",
    "TemporalKnowledgeLayer",
    "ConstrainedGenerator",
    "ContentValidator",
    "ContentGeneration",
    "ContextManager",
    "DataRetrieval",
    "DecompositionAdapter",
    "AtomicTask",
    "FormalVerificationLayer",
    "KnowledgeAdapter",
    "LagrangeFilter",
    "MatryoshkaClient",
    "OptimizedWorkflow",
    "ReproducibilityLayer",
    "StreamingConstrainedGenerator",
    "SmartContextManager",
    "CloudLLMMonitor",
    "cloud_consensus",
    "detect_divergence",
    "MultiModalDeterministicGenerator",
    "MultiModalGenerator",
    "DeterminismConfig",
    "DeterminismResult",
    "DeterministicPipeline",
    "EnhancedDeterministicPipeline",
    "FullDeterminismStack",
    "HybridDeterministicSystem",
    "ProductionDeterministicSystem",
    "UltraDeterministicPipeline",
    "generate_with_full_verification",
    "verified_generation",
    "verified_response",
    "IntelligentModelRouter",
    "SecurityLayer",
    "LLMConfig",
    "BaseLLM",
    "OpenAIChatLLM",
    "AnthropicLLM",
    "GoogleLLM",
    "HFLocalLLM",
    "build_llm",
    "deterministic_seed",
    "hash_prompt",
    "similarity",
]
