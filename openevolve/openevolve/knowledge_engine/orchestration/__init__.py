"""
Knowledge Engine Orchestration Module

Provides unified orchestration of all Knowledge Engine integrations with:
- Self-healing capabilities that learn from failures
- Adaptive learning from every execution
- Component coordination and gap coverage
- Domain-specific presets (finance, chemistry, healthcare, research)
- Component skip/disable capabilities
- Configurable pipeline architecture
- Continuous improvement through feedback loops
- MCP server for Model Context Protocol integration

The orchestration system creates a cohesive, self-learning system where:
1. Failures are learning opportunities
2. Components cover each other's gaps
3. The system adapts and improves over time
4. Feedback drives continuous optimization
5. Multiple healing strategies ensure robustness

Example:
    from knowledge_engine.orchestration import (
        # Self-healing orchestrators
        SelfHealingOrchestrator,
        create_self_healing_finance_orchestrator,
        create_self_healing_chemistry_orchestrator,
        
        # Base orchestrators
        KnowledgeOrchestrator,
        create_finance_orchestrator,
        create_chemistry_orchestrator,
        
        # Learning and adaptation
        LearningEngine,
        ComponentCoordinator,
        AdaptiveOrchestratorIntegration,
        
        # MCP Server
        KnowledgeEngineMCPHandler,
        create_mcp_server,
    )
    
    # Create a self-healing orchestrator
    orchestrator = create_self_healing_finance_orchestrator(
        learning_storage_path="finance_learning.json"
    )
    
    # Process data - the orchestrator learns from each execution
    result = orchestrator.process({
        'text': 'Financial data...',
        'data_type': 'financial_report'
    })
    
    # If components fail, the orchestrator automatically:
    # - Retries with adjusted configuration
    # - Substitutes alternative components
    # - Uses fallback pipelines
    # - Records lessons learned
"""

# Base orchestrator
from .knowledge_orchestrator import (
    KnowledgeOrchestrator,
    OrchestratorConfig,
    DomainPresets,
    DomainType,
    ComponentType,
    ComponentConfig,
    PipelineStage,
    create_finance_orchestrator,
    create_chemistry_orchestrator,
    create_healthcare_orchestrator,
    create_research_orchestrator,
    create_minimal_orchestrator,
)

# Self-healing orchestrator
from .self_healing_orchestrator import (
    SelfHealingOrchestrator,
    FailureType,
    HealingStrategy,
    FailureEvent,
    HealingAction,
    ComponentSubstitutionMatrix,
    create_self_healing_finance_orchestrator,
    create_self_healing_chemistry_orchestrator,
    create_self_healing_healthcare_orchestrator,
    create_self_healing_research_orchestrator,
)

# Learning engine
from .learning_engine import (
    LearningEngine,
    LearningExperience,
    ComponentProfile,
    PipelinePattern,
)

# Component coordination
from .component_coordination import (
    ComponentCoordinator,
    ComponentCapabilityRegistry,
    ComponentCapabilities,
    CapabilityType,
    GapType,
    GapFillingAssignment,
    CoordinationContext,
    analyze_pipeline_gaps,
)

# Feedback and improvement
from .feedback_loop import (
    FeedbackCollector,
    ContinuousImprovementEngine,
    AdaptiveOrchestratorIntegration,
    FeedbackType,
    ImprovementArea,
    ImprovementExperiment,
    create_adaptive_orchestrator,
)

# Circuit breaker
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitBreakerOpenError,
    CircuitState,
    CircuitStats,
    get_circuit_breaker,
    circuit_breaker,
)

# Safe evaluation
from .safe_eval import (
    SafeExpressionEvaluator,
    safe_eval,
    ConditionEvaluator,
)

# Async orchestrator
from .async_orchestrator import (
    AsyncKnowledgeOrchestrator,
    AsyncSelfHealingOrchestrator,
    create_async_finance_orchestrator,
    create_async_chemistry_orchestrator,
    create_async_healthcare_orchestrator,
    create_async_research_orchestrator,
    create_async_self_healing_finance_orchestrator,
    create_async_self_healing_chemistry_orchestrator,
    create_async_self_healing_healthcare_orchestrator,
    create_async_self_healing_research_orchestrator,
)

# Domain classifier (adaptive)
from .domain_classifier import (
    DomainClassifier,
    DomainCategory,
    ContentType,
    ClassificationResult,
    classify_input,
)

# Global learning (cross-user)
from .global_learning_engine import (
    GlobalLearningEngine,
    GlobalPattern,
    KnowledgeEntry,
    get_global_learning_engine,
)

# Gauntlet integration (validation)
from .gauntlet_integration import (
    GauntletIntegration,
    GauntletTest,
    TestExecution,
    TestType,
    TestResult,
)

# Import from parent orchestration.py for backward compatibility
import importlib.util
import sys
from pathlib import Path
_orch_py_path = Path(__file__).parent.parent / "orchestration.py"
if _orch_py_path.exists():
    _orch_spec = importlib.util.spec_from_file_location("_orchestration_py", _orch_py_path)
    _orch_py = importlib.util.module_from_spec(_orch_spec)
    sys.modules["_orchestration_py"] = _orch_py
    _orch_spec.loader.exec_module(_orch_py)
    
    KnowledgeEngine = _orch_py.KnowledgeEngine
    ProcessingResult = _orch_py.ProcessingResult
    QueryResult = _orch_py.QueryResult
    create_knowledge_engine = _orch_py.create_knowledge_engine
    
    del _orch_spec, _orch_py_path

# Adaptive orchestrator (ULTIMATE - auto-classifying, globally learning)
from .adaptive_orchestrator import (
    AdaptiveOrchestrator,
    AdaptiveConfig,
    create_adaptive_orchestrator,
)

# Integrated orchestrator (production-ready)
from .integrated_orchestrator import (
    IntegratedOrchestrator,
    ExecutionContext,
    OrchestratorResult,
    create_integrated_finance_orchestrator,
    create_integrated_chemistry_orchestrator,
    create_integrated_research_orchestrator,
)

# MCP Server
from .mcp_server import (
    KnowledgeEngineMCPHandler,
    create_mcp_server,
)

__all__ = [
    # From orchestration.py (backward compatibility)
    'KnowledgeEngine',
    'ProcessingResult',
    'QueryResult',
    'create_knowledge_engine',
    
    # Base orchestrator
    'KnowledgeOrchestrator',
    'OrchestratorConfig',
    'DomainPresets',
    'DomainType',
    'ComponentType',
    'ComponentConfig',
    'PipelineStage',
    'create_finance_orchestrator',
    'create_chemistry_orchestrator',
    'create_healthcare_orchestrator',
    'create_research_orchestrator',
    'create_minimal_orchestrator',
    
    # Self-healing
    'SelfHealingOrchestrator',
    'FailureType',
    'HealingStrategy',
    'FailureEvent',
    'HealingAction',
    'ComponentSubstitutionMatrix',
    'create_self_healing_finance_orchestrator',
    'create_self_healing_chemistry_orchestrator',
    'create_self_healing_healthcare_orchestrator',
    'create_self_healing_research_orchestrator',
    
    # Learning
    'LearningEngine',
    'LearningExperience',
    'ComponentProfile',
    'PipelinePattern',
    
    # Coordination
    'ComponentCoordinator',
    'ComponentCapabilityRegistry',
    'ComponentCapabilities',
    'CapabilityType',
    'GapType',
    'GapFillingAssignment',
    'CoordinationContext',
    'analyze_pipeline_gaps',
    
    # Feedback
    'FeedbackCollector',
    'ContinuousImprovementEngine',
    'AdaptiveOrchestratorIntegration',
    'FeedbackType',
    'ImprovementArea',
    'ImprovementExperiment',
    'create_adaptive_orchestrator',
    
    # Circuit breaker
    'CircuitBreaker',
    'CircuitBreakerRegistry',
    'CircuitBreakerOpenError',
    'CircuitState',
    'CircuitStats',
    'get_circuit_breaker',
    'circuit_breaker',
    
    # Safe evaluation
    'SafeExpressionEvaluator',
    'safe_eval',
    'ConditionEvaluator',
    
    # Async orchestrator
    'AsyncKnowledgeOrchestrator',
    'AsyncSelfHealingOrchestrator',
    'create_async_finance_orchestrator',
    'create_async_chemistry_orchestrator',
    'create_async_healthcare_orchestrator',
    'create_async_research_orchestrator',
    'create_async_self_healing_finance_orchestrator',
    'create_async_self_healing_chemistry_orchestrator',
    'create_async_self_healing_healthcare_orchestrator',
    'create_async_self_healing_research_orchestrator',
    
    # Domain classifier (adaptive)
    'DomainClassifier',
    'DomainCategory',
    'ContentType',
    'ClassificationResult',
    'classify_input',
    
    # Global learning (cross-user)
    'GlobalLearningEngine',
    'GlobalPattern',
    'KnowledgeEntry',
    'get_global_learning_engine',
    
    # Gauntlet integration (validation)
    'GauntletIntegration',
    'GauntletTest',
    'TestExecution',
    'TestType',
    'TestResult',
    
    # Adaptive orchestrator (ULTIMATE)
    'AdaptiveOrchestrator',
    'AdaptiveConfig',
    'create_adaptive_orchestrator',
    
    # Integrated orchestrator
    'IntegratedOrchestrator',
    'ExecutionContext',
    'OrchestratorResult',
    'create_integrated_finance_orchestrator',
    'create_integrated_chemistry_orchestrator',
    'create_integrated_research_orchestrator',
    
    # MCP Server
    'KnowledgeEngineMCPHandler',
    'create_mcp_server',
]
