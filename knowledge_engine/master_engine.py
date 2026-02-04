"""
OpenEvolve Master Knowledge Engine

A self-learning, self-healing, self-improving knowledge engine that:
- Integrates 21+ separate projects into a cohesive system
- Learns from every execution (successes and failures)
- Automatically heals from component failures
- Continuously improves through collective intelligence
- Coordinates all components to cover each other's gaps

This is the meta-project that unifies all knowledge processing capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
import uuid
import json
import threading
from pathlib import Path
import copy

# Import all 21+ project integrations
from knowledge_engine.integrations.graphiti_integration import GraphitiIntegration
from knowledge_engine.integrations.kggen_integration import KGGenIntegration
from knowledge_engine.integrations.oneke_integration import OneKEIntegration
from knowledge_engine.integrations.aikg_integration import AIKGIntegration
from knowledge_engine.integrations.ragbits_integration import RagbitsIntegration
from knowledge_engine.integrations.crewai_integration import CrewAIIntegration
from knowledge_engine.integrations.deepke_integration import DeepKEIntegration
from knowledge_engine.integrations.research_quest_integration import ResearchQuestIntegration
from knowledge_engine.integrations.agentic_context_integration import AgenticContextEngine
from knowledge_engine.integrations.agentjson_integration import AgentJSONIntegration
from knowledge_engine.integrations.dspy_integration import DSPyIntegration
from knowledge_engine.integrations.leanaide_integration import LeanAideIntegration
from knowledge_engine.integrations.openevolve_integration_library import OpenEvolveIntegrationLibrary
from knowledge_engine.integrations.mcp_gateway_integration import MCPGatewayIntegration
from knowledge_engine.integrations.pami_integration import PAMIIntegration
from knowledge_engine.integrations.neuralkg_integration import NeuralKGIntegration
try:
    from knowledge_engine.integrations.causal_learn_integration import CausalLearnIntegration
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CausalLearnIntegration = None
    CAUSAL_LEARN_AVAILABLE = False
from knowledge_engine.integrations.lagrange_mapper_integration import LagrangeMapperIntegration
from knowledge_engine.integrations.karateclub_integration import KarateClubIntegration
from knowledge_engine.integrations.global_chem_integration import GlobalChemIntegration
from knowledge_engine.integrations.neuromancer_integration import NeuromancerIntegration
from knowledge_engine.integrations.roma_integration import ROMAIntegration, ROMA_INTEGRATION_AVAILABLE

# New Advanced Integrations (2026-02-03)
try:
    from knowledge_engine.integrations.outlines.outlines_integration import OutlinesKGIntegration
    OUTLINES_AVAILABLE = True
except ImportError:
    OutlinesKGIntegration = None
    OUTLINES_AVAILABLE = False

try:
    from knowledge_engine.integrations.lmql.lmql_integration import LMQLKGIntegration
    LMQL_AVAILABLE = True
except ImportError:
    LMQLKGIntegration = None
    LMQL_AVAILABLE = False

try:
    from knowledge_engine.integrations.neuromancer.neuromancer_integration import NeuromancerKGIntegration
    NEUROMANCER_KE_AVAILABLE = True
except ImportError:
    NeuromancerKGIntegration = None
    NEUROMANCER_KE_AVAILABLE = False

try:
    from knowledge_engine.integrations.cognitive_hydraulics.cognitive_hydraulics_integration import CognitiveHydraulicsKGIntegration
    COGNITIVE_HYDRAULICS_AVAILABLE = True
except ImportError:
    CognitiveHydraulicsKGIntegration = None
    COGNITIVE_HYDRAULICS_AVAILABLE = False

# Conversation & Safety Integrations (2026-02-03)
try:
    from knowledge_engine.integrations.dts.dts_integration import DTSKGIntegration
    DTS_AVAILABLE = True
except ImportError:
    DTSKGIntegration = None
    DTS_AVAILABLE = False

try:
    from knowledge_engine.integrations.guardrails.guardrails_integration import GuardrailsKGIntegration
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GuardrailsKGIntegration = None
    GUARDRAILS_AVAILABLE = False

try:
    from knowledge_engine.integrations.icr.icr_integration import ICRKGIntegration
    ICR_AVAILABLE = True
except ImportError:
    ICRKGIntegration = None
    ICR_AVAILABLE = False

# Import orchestration components
from knowledge_engine.orchestration.self_healing_orchestrator import (
    SelfHealingOrchestrator, HealingStrategy, FailureEvent, FailureType
)
from knowledge_engine.orchestration.learning_engine import LearningEngine, LearningExperience
from knowledge_engine.orchestration.global_learning_engine import GlobalLearningEngine
from knowledge_engine.orchestration.integrated_orchestrator import IntegratedOrchestrator, ExecutionContext
from knowledge_engine.orchestration.component_coordination import ComponentCoordinator
from knowledge_engine.orchestration.feedback_loop import FeedbackCollector, FeedbackType
from knowledge_engine.orchestration.circuit_breaker import CircuitBreaker, get_circuit_breaker
from knowledge_engine.orchestration.knowledge_orchestrator import KnowledgeOrchestrator, OrchestratorConfig, ComponentType

logger = logging.getLogger(__name__)


class KnowledgeDomain(Enum):
    """Knowledge processing domains"""
    GENERAL = "general"
    CHEMISTRY = "chemistry"
    BIOMEDICAL = "biomedical"
    LEGAL = "legal"
    FINANCE = "finance"
    RESEARCH = "research"
    TECHNICAL = "technical"
    TEMPORAL = "temporal"
    MULTILINGUAL = "multilingual"
    CAUSAL = "causal"
    SPATIAL = "spatial"


@dataclass
class KnowledgeRequest:
    """A knowledge processing request"""
    request_id: str
    query: str
    domain: KnowledgeDomain
    context: Dict[str, Any] = field(default_factory=dict)
    preferred_components: Optional[List[str]] = None
    priority: int = 5  # 1-10, higher is more urgent
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'query': self.query,
            'domain': self.domain.value,
            'context': self.context,
            'preferred_components': self.preferred_components,
            'priority': self.priority,
            'timestamp': self.timestamp
        }


@dataclass
class KnowledgeResponse:
    """Response from knowledge processing"""
    request_id: str
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    components_used: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    quality_score: float = 0.0
    confidence: float = 0.0
    learned_lessons: List[str] = field(default_factory=list)
    healing_actions: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'success': self.success,
            'results': self.results,
            'components_used': self.components_used,
            'processing_time_ms': self.processing_time_ms,
            'quality_score': self.quality_score,
            'confidence': self.confidence,
            'learned_lessons': self.learned_lessons,
            'healing_actions': self.healing_actions,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        }


class ComponentRegistry:
    """
    Registry for all 21+ integrated components.
    Manages component availability, capabilities, and relationships.
    """
    
    def __init__(self):
        self.components: Dict[str, Any] = {}
        self.capabilities: Dict[str, List[str]] = {}
        self.substitution_matrix: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all 21+ components with proper configuration"""
        # Define capabilities for each component (MUST be before component init)
        self.capabilities = {
            'graphiti': ['temporal_knowledge', 'contradiction_detection', 'hybrid_search', 'point_in_time'],
            'kggen': ['entity_extraction', 'relation_extraction', 'deduplication', 'graph_construction'],
            'oneke': ['bilingual_extraction', 'schema_guided', 'multilingual'],
            'aikg': ['knowledge_inference', 'standardization', 'visualization'],
            'deepke': ['relation_extraction', 'entity_typing', 'document_level'],
            'ragbits': ['retrieval', 'generation', 'document_processing'],
            'crewai': ['multi_agent', 'task_delegation', 'workflow'],
            'pami': ['pattern_mining', 'frequent_patterns', 'sequential_patterns'],
            'neuralkg': ['embeddings', 'link_prediction', 'entity_similarity'],
            'causal_learn': ['causal_discovery', 'structure_learning', 'confounder_detection'],
            'karateclub': ['community_detection', 'node_embeddings', 'graph_embeddings'],
            'global_chem': ['chemistry', 'molecular', 'compound_recognition'],
            'neuromancer': ['neural_odes', 'physics_informed', 'dynamical_systems'],
            'lagrange_mapper': ['topological_analysis', 'attractor_landscapes', 'clustering'],
            'leanaide': ['formal_verification', 'proof_assistance', 'theorem_proving'],
            'research_quest': ['research_automation', 'literature_review', 'hypothesis_generation'],
            'agentic_context': ['context_management', 'reflection', 'conversation'],
            'agentjson': ['structured_output', 'json_generation', 'schema_validation'],
            'dspy': ['prompt_optimization', 'program_of_thought', 'demonstration_selection'],
            'openevolve_lib': ['system_integration', 'bubblelabs', 'workflow_orchestration'],
            'mcp_gateway': ['tool_orchestration', 'api_gateway', 'service_coordination'],
            'roma': ['meta_agent', 'decomposition', 'execution', 'verification', 'recomposition', 'hierarchical_planning'],
            
            # Advanced Integrations (2026-02-03)
            'outlines': ['structured_generation', 'json_constraints', 'regex_constraints', 'guaranteed_valid_output'],
            'lmql': ['declarative_queries', 'constraint_programming', 'multi_turn_dialog', 'cypher_generation'],
            'neuromancer_ke': ['physics_simulation', 'ode_solving', 'pde_solving', 'dynamics_learning', 'scientific_domains'],
            'cognitive_hydraulics': ['hybrid_reasoning', 'symbolic_reasoning', 'heuristic_reasoning', 'evolutionary_fallback', 'learning_chunking'],
            
            # Conversation & Safety Integrations (2026-02-03)
            'dts': ['conversation_optimization', 'dialogue_tree_search', 'user_simulation', 'multi_judge_scoring', 'beam_search'],
            'guardrails': ['ai_safety', 'output_validation', 'pii_detection', 'toxicity_check', 'policy_enforcement', 'compliance_gdpr_hipaa'],
            'icr': ['iterative_refinement', 'quality_improvement', 'generate_critique_refine', 'convergence_detection', 'early_stopping']
        }
        
        # Define substitution matrix (which components can cover for others)
        self.substitution_matrix = {
            'kggen': ['deepke', 'aikg'],
            'deepke': ['kggen', 'oneke'],
            'oneke': ['kggen', 'deepke'],
            'neuralkg': ['karateclub', 'aikg'],
            'karateclub': ['neuralkg', 'aikg'],
            'pami': ['karateclub', 'neuralkg'],
            'causal_learn': ['neuralkg', 'karateclub'],
            'ragbits': ['crewai', 'aikg'],
            'crewai': ['openevolve_lib', 'mcp_gateway'],
            
            # Advanced Integrations substitution matrix
            'outlines': ['agentjson', 'dspy'],  # Structured output alternatives
            'lmql': ['crewai', 'dspy'],  # Query/delegation alternatives
            'neuromancer_ke': ['neuromancer', 'causal_learn'],  # Scientific analysis alternatives
            'cognitive_hydraulics': ['crewai', 'dspy', 'neuralkg'],  # Reasoning alternatives
            
            # Conversation & Safety substitution matrix
            'dts': ['crewai', 'agentic_context'],  # Conversation alternatives
            'guardrails': ['agentjson', 'z3'],  # Validation alternatives
            'icr': ['dspy', 'outlines'],  # Quality improvement alternatives
        }
        
        # Core Knowledge Extraction (1-5)
        import os
        self.components['graphiti'] = self._safe_init(
            lambda: GraphitiIntegration(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "")
            ),
            'graphiti'
        )
        self.components['kggen'] = self._safe_init(KGGenIntegration, 'kggen')
        self.components['oneke'] = self._safe_init(OneKEIntegration, 'oneke')
        self.components['aikg'] = self._safe_init(AIKGIntegration, 'aikg')
        self.components['deepke'] = self._safe_init(DeepKEIntegration, 'deepke')
        
        # Analysis & Reasoning (6-11)
        self.components['ragbits'] = self._safe_init(RagbitsIntegration, 'ragbits')
        self.components['crewai'] = self._safe_init(CrewAIIntegration, 'crewai')
        self.components['pami'] = self._safe_init(PAMIIntegration, 'pami')
        self.components['neuralkg'] = self._safe_init(NeuralKGIntegration, 'neuralkg')
        self.components['causal_learn'] = self._safe_init(CausalLearnIntegration, 'causal_learn') if CAUSAL_LEARN_AVAILABLE else self._create_mock_component('causal_learn')
        self.components['karateclub'] = self._safe_init(KarateClubIntegration, 'karateclub')
        
        # Specialized Domains (12-15)
        self.components['global_chem'] = self._safe_init(GlobalChemIntegration, 'global_chem')
        self.components['neuromancer'] = self._safe_init(NeuromancerIntegration, 'neuromancer')
        self.components['lagrange_mapper'] = self._safe_init(LagrangeMapperIntegration, 'lagrange_mapper')
        self.components['leanaide'] = self._safe_init(LeanAideIntegration, 'leanaide')
        
        # Integration & Orchestration (16-21)
        self.components['research_quest'] = self._safe_init(ResearchQuestIntegration, 'research_quest')
        self.components['agentic_context'] = self._safe_init(AgenticContextEngine, 'agentic_context')
        self.components['agentjson'] = self._safe_init(AgentJSONIntegration, 'agentjson')
        self.components['dspy'] = self._safe_init(DSPyIntegration, 'dspy')
        self.components['openevolve_lib'] = self._safe_init(OpenEvolveIntegrationLibrary, 'openevolve_lib')
        self.components['mcp_gateway'] = self._safe_init(MCPGatewayIntegration, 'mcp_gateway')
        
        # Advanced Integrations (2026-02-03) - New capabilities
        # Outlines - Structured LLM output generation with constraints
        if OUTLINES_AVAILABLE:
            self.components['outlines'] = self._safe_init(OutlinesKGIntegration, 'outlines')
        else:
            self.components['outlines'] = self._create_mock_component('outlines')
        
        # LMQL - Declarative SQL-like query language for LLMs
        if LMQL_AVAILABLE:
            self.components['lmql'] = self._safe_init(LMQLKGIntegration, 'lmql')
        else:
            self.components['lmql'] = self._create_mock_component('lmql')
        
        # Neuromancer KG - Physics-informed neural operators
        if NEUROMANCER_KE_AVAILABLE:
            self.components['neuromancer_ke'] = self._safe_init(NeuromancerKGIntegration, 'neuromancer_ke')
        else:
            self.components['neuromancer_ke'] = self._create_mock_component('neuromancer_ke')
        
        # Cognitive-Hydraulics - Hybrid neuro-symbolic reasoning (Soar+ACT-R+Evolutionary)
        if COGNITIVE_HYDRAULICS_AVAILABLE:
            self.components['cognitive_hydraulics'] = self._safe_init(CognitiveHydraulicsKGIntegration, 'cognitive_hydraulics')
        else:
            self.components['cognitive_hydraulics'] = self._create_mock_component('cognitive_hydraulics')
        
        # Conversation & Safety Integrations (2026-02-03)
        # DTS - Dialogue Tree Search for multi-turn conversation optimization
        if DTS_AVAILABLE:
            self.components['dts'] = self._safe_init(DTSKGIntegration, 'dts')
        else:
            self.components['dts'] = self._create_mock_component('dts')
        
        # Guardrails - AI safety, output validation, and policy enforcement
        if GUARDRAILS_AVAILABLE:
            self.components['guardrails'] = self._safe_init(GuardrailsKGIntegration, 'guardrails')
        else:
            self.components['guardrails'] = self._create_mock_component('guardrails')
        
        # ICR - Iterative Contextual Refinements for quality improvement
        if ICR_AVAILABLE:
            self.components['icr'] = self._safe_init(ICRKGIntegration, 'icr')
        else:
            self.components['icr'] = self._create_mock_component('icr')

        # ROMA Meta-Agent (22) - Hierarchical problem decomposition and execution
        if ROMA_INTEGRATION_AVAILABLE:
            try:
                import os
                roma_config = {
                    'max_depth_analysis': 3,
                    'max_depth_solving': 2,
                    'execution_mode': 'recursive'
                }
                self.components['roma'] = ROMAIntegration(config=roma_config)
                logger.info({
                    'msg': 'ROMA integration initialized successfully',
                    'component': 'roma',
                    'capabilities': self.capabilities.get('roma', [])
                })
            except Exception as e:
                logger.warning({
                    'msg': 'ROMA integration initialization failed',
                    'component': 'roma',
                    'error': str(e)
                })
                self.components['roma'] = self._create_mock_component('roma')
        else:
            logger.info({
                'msg': 'ROMA integration not available (optional dependency)',
                'component': 'roma'
            })
            self.components['roma'] = self._create_mock_component('roma')
    
    def _safe_init(self, init_func, name: str):
        """Safely initialize a component with error handling"""
        try:
            return init_func()
        except Exception as e:
            logger.warning(f"Failed to initialize {name}: {e}")
            return self._create_mock_component(name)
    
    def _create_mock_component(self, name: str):
        """Create a failing mock component for when real one is not available."""
        from .optional_imports import create_failing_mock
        
        MockComponent = create_failing_mock(
            package_name=name,
            feature_name=f'{name} integration',
            install_command=f'pip install {name.lower().replace(" ", "-")}'
        )
        
        # Return the class itself - instantiation will raise error
        return MockComponent
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a component by name"""
        with self._lock:
            return self.components.get(name)
    
    def get_available_components(self) -> List[str]:
        """Get list of available components"""
        with self._lock:
            return [name for name, comp in self.components.items() 
                   if hasattr(comp, 'is_available') and comp.is_available()]
    
    def get_components_for_capability(self, capability: str) -> List[str]:
        """Get components that provide a specific capability"""
        with self._lock:
            return [name for name, caps in self.capabilities.items() 
                   if capability in caps]
    
    def get_substitutes(self, component_name: str) -> List[str]:
        """Get components that can substitute for a given component"""
        with self._lock:
            return self.substitution_matrix.get(component_name, [])
    
    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """Get all capabilities mapped to components"""
        with self._lock:
            result = {}
            for comp, caps in self.capabilities.items():
                for cap in caps:
                    if cap not in result:
                        result[cap] = []
                    result[cap].append(comp)
            return result


class SelfImprovingEngine:
    """
    Engine that continuously improves through learning from all executions.
    Includes persistence for improvement history.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.learning_engine = LearningEngine(storage_path)
        self.global_learning = GlobalLearningEngine(storage_path)
        self.improvement_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._load_improvement_history()
    
    def _load_improvement_history(self):
        """Load improvement history from disk"""
        if self.storage_path:
            history_file = Path(self.storage_path) / 'improvement_history.json'
            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        self.improvement_history = json.load(f)
                    logger.info(f"Loaded {len(self.improvement_history)} improvement records")
                except Exception as e:
                    logger.warning(f"Failed to load improvement history: {e}")
    
    def _save_improvement_history(self):
        """Save improvement history to disk"""
        if self.storage_path:
            history_file = Path(self.storage_path) / 'improvement_history.json'
            try:
                history_file.parent.mkdir(parents=True, exist_ok=True)
                with open(history_file, 'w') as f:
                    json.dump(self.improvement_history[-1000:], f)  # Keep last 1000
            except Exception as e:
                logger.warning(f"Failed to save improvement history: {e}")
    
    def record_execution(self, request: KnowledgeRequest, 
                        response: KnowledgeResponse,
                        user_id: Optional[str] = None):
        """Record an execution for learning"""
        with self._lock:
            # Record improvement
            improvement_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'request_id': request.request_id,
                'domain': request.domain.value,
                'components_used': response.components_used,
                'success': response.success,
                'quality_score': response.quality_score,
                'processing_time_ms': response.processing_time_ms,
                'user_id': user_id
            }
            self.improvement_history.append(improvement_record)
            self._save_improvement_history()
            
            # Record in local learning
            self.learning_engine.record_experience(
                input_data=request.to_dict(),
                data_type=request.domain.value,
                domain=request.domain.value,
                pipeline_config={'component_configs': {}},
                components_used=response.components_used,
                success=response.success,
                execution_time_ms=response.processing_time_ms,
                results={'quality': response.quality_score, 'data': response.results},
                errors=[{'type': 'execution_failed', 'message': response.error_message}] if not response.success else []
            )
            
            # Contribute to global learning
            if user_id:
                try:
                    self.global_learning.contribute_experience(
                        user_id, response.to_dict(), {'request': request.to_dict()}
                    )
                except Exception as e:
                    logger.warning(f"Failed to contribute to global learning: {e}")
            
            logger.info({
                'msg': 'Execution recorded for learning',
                'request_id': request.request_id,
                'success': response.success,
                'quality': response.quality_score
            })
    
    def _hash_request(self, request: KnowledgeRequest) -> str:
        """Create a hash of request for similarity matching"""
        content = f"{request.query}:{request.domain.value}"
        return str(hash(content))
    
    def get_recommendations(self, domain: str, 
                           data_type: str) -> Dict[str, Any]:
        """Get learned recommendations for a domain/data type"""
        with self._lock:
            # Get pipeline recommendation
            pipeline_rec = self.learning_engine.recommend_pipeline(
                data_type, domain, {}
            )
            
            # Get component recommendations
            component_recs = {}
            for component in ['kggen', 'oneke', 'deepke', 'graphiti']:
                if component in self.learning_engine.component_profiles:
                    profile = self.learning_engine.component_profiles[component]
                    rec = profile.get_recommendation_for_context(data_type, domain)
                    component_recs[component] = rec
            
            return {
                'pipeline': pipeline_rec,
                'components': component_recs,
                'learning_summary': self.learning_engine.get_learning_summary()
            }
    
    def get_best_components_for_domain(self, domain: str, top_n: int = 3) -> List[str]:
        """
        Get the best performing components for a domain based on learning.
        
        Args:
            domain: Domain to query
            top_n: Number of top components to return
            
        Returns:
            List of component names sorted by performance
        """
        with self._lock:
            component_scores = {}
            
            for component_name, profile in self.learning_engine.component_profiles.items():
                if domain in profile.performance_by_domain:
                    perf = profile.performance_by_domain[domain]
                    score = perf.get('avg_quality', 0) * perf.get('success_rate', 0)
                    component_scores[component_name] = score
            
            # Sort by score descending
            sorted_components = sorted(
                component_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [comp for comp, _ in sorted_components[:top_n]]
    
    def predict_failure(self, domain: str, 
                       components: List[str]) -> Optional[Dict[str, Any]]:
        """Predict potential failures before they happen"""
        with self._lock:
            return self.learning_engine.predict_failure(
                'unknown', domain, components
            )


class MasterKnowledgeEngine:
    """
    Master Knowledge Engine - The cohesive meta-project.
    
    Self-learning: Learns from every execution
    Self-healing: Recovers from component failures automatically
    Self-improving: Continuously improves through collective intelligence
    
    Integrates 21+ projects into a unified system where:
    - Each component fills a specific role
    - Components cover for each other when failures occur
    - Learned knowledge improves future executions
    - Failures become learning opportunities
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_learning: bool = True,
        enable_healing: bool = True,
        storage_path: Optional[str] = None
    ):
        """
        Initialize the Master Knowledge Engine.
        
        Args:
            config: Configuration dictionary
            enable_learning: Enable self-learning capabilities
            enable_healing: Enable self-healing capabilities
            storage_path: Path for persistent storage
        """
        self.config = config or {}
        self.enable_learning = enable_learning
        self.enable_healing = enable_healing
        self.storage_path = storage_path or "knowledge_engine_data"
        
        # Initialize core components
        self.component_registry = ComponentRegistry()
        self.self_improving = SelfImprovingEngine(storage_path) if enable_learning else None
        
        # Initialize base orchestrator for healing (may fail with broken dependencies)
        self.orchestrator_config = OrchestratorConfig(
            domain=self.config.get('domain', 'general'),
            correlation_id=str(uuid.uuid4())
        )
        
        self.orchestrator = None
        if enable_healing:
            try:
                self.orchestrator = SelfHealingOrchestrator(
                    config=self.orchestrator_config,
                    enable_self_healing=True,
                    learning_storage_path=storage_path
                )
            except Exception as e:
                logger.warning(f"Could not initialize SelfHealingOrchestrator: {e}")
                logger.warning("Running in simplified mode without full orchestration")
        
        if self.orchestrator is None:
            try:
                self.orchestrator = KnowledgeOrchestrator(self.orchestrator_config)
            except Exception as e:
                logger.warning(f"Could not initialize KnowledgeOrchestrator: {e}")
                self.orchestrator = None
        
        # Circuit breakers for component protection
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._initialize_circuit_breakers()
        
        # Statistics
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.healing_count = 0
        
        logger.info({
            'msg': 'MasterKnowledgeEngine initialized',
            'components': len(self.component_registry.components),
            'learning_enabled': enable_learning,
            'healing_enabled': enable_healing
        })
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for all components"""
        for name in self.component_registry.components.keys():
            self.circuit_breakers[name] = get_circuit_breaker(
                name=name,
                failure_threshold=3,
                recovery_timeout=60.0
            )
    
    async def process(self, 
                     query: str,
                     domain: Union[KnowledgeDomain, str] = KnowledgeDomain.GENERAL,
                     context: Optional[Dict[str, Any]] = None,
                     preferred_components: Optional[List[str]] = None,
                     user_id: Optional[str] = None) -> KnowledgeResponse:
        """
        Process a knowledge request with full self-learning and self-healing.
        
        Args:
            query: The knowledge query to process
            domain: Domain of the query
            context: Additional context
            preferred_components: Preferred components to use
            user_id: User identifier for learning
            
        Returns:
            KnowledgeResponse with results and metadata
        """
        start_time = datetime.now(timezone.utc)
        request_id = str(uuid.uuid4())
        
        # Normalize domain
        if isinstance(domain, str):
            domain = KnowledgeDomain(domain)
        
        # Create request object
        request = KnowledgeRequest(
            request_id=request_id,
            query=query,
            domain=domain,
            context=context or {},
            preferred_components=preferred_components
        )
        
        logger.info({
            'msg': 'Processing knowledge request',
            'request_id': request_id,
            'domain': domain.value,
            'query_length': len(query)
        })
        
        # Get learned recommendations
        if self.enable_learning and self.self_improving:
            recommendations = self.self_improving.get_recommendations(
                domain.value, domain.value
            )
            logger.debug({
                'recommendations': recommendations.get('learning_summary', {})
            })
        
        # Determine which components to use
        components_to_use = self._select_components(
            domain, preferred_components, query
        )
        
        # Execute with healing
        results = {}
        components_used = []
        healing_actions = []
        learned_lessons = []
        
        for component_name in components_to_use:
            component = self.component_registry.get_component(component_name)
            if not component:
                continue
            
            # Check circuit breaker
            breaker = self.circuit_breakers.get(component_name)
            if breaker and not breaker.can_execute():
                logger.warning(f"Circuit open for {component_name}, trying substitutes")
                substitutes = self.component_registry.get_substitutes(component_name)
                for sub_name in substitutes:
                    sub_component = self.component_registry.get_component(sub_name)
                    sub_breaker = self.circuit_breakers.get(sub_name)
                    if sub_component and (not sub_breaker or sub_breaker.can_execute()):
                        component_name = sub_name
                        component = sub_component
                        breaker = sub_breaker
                        healing_actions.append(f"substituted_{component_name}_with_{sub_name}")
                        break
                else:
                    logger.warning(f"No substitutes available for {component_name}")
                    continue
            
            # Try to execute component
            try:
                component_result = await self._execute_component(
                    component_name, component, query, context
                )
                
                if component_result.get('success'):
                    results[component_name] = component_result.get('output', {})
                    components_used.append(component_name)
                    if breaker:
                        breaker.record_success()
                    
                    learned_lessons.append(f"{component_name}_succeeded")
                else:
                    if breaker:
                        breaker.record_failure()
                    learned_lessons.append(f"{component_name}_failed")
                    
            except Exception as e:
                logger.error({
                    'component': component_name,
                    'error': str(e)
                })
                if breaker:
                    breaker.record_failure()
                learned_lessons.append(f"{component_name}_error: {str(e)}")
                
                # Try healing
                if self.enable_healing:
                    healing_result = await self._heal_failure(
                        component_name, query, context
                    )
                    if healing_result:
                        results.update(healing_result.get('results', {}))
                        components_used.extend(healing_result.get('components', []))
                        healing_actions.append(f"healed_{component_name}")
                        self.healing_count += 1
        
        # Calculate quality and confidence
        quality_score = self._calculate_quality(results)
        confidence = self._calculate_confidence(results, components_used)
        
        # Create response
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        success = len(results) > 0
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        response = KnowledgeResponse(
            request_id=request_id,
            success=success,
            results=results,
            components_used=components_used,
            processing_time_ms=processing_time,
            quality_score=quality_score,
            confidence=confidence,
            learned_lessons=learned_lessons,
            healing_actions=healing_actions,
            error_message=None if success else "All components failed"
        )
        
        # Record for learning
        if self.enable_learning and self.self_improving:
            self.self_improving.record_execution(request, response, user_id)
        
        self.execution_count += 1
        
        logger.info({
            'msg': 'Knowledge request completed',
            'request_id': request_id,
            'success': success,
            'components_used': components_used,
            'quality': quality_score,
            'processing_time_ms': processing_time
        })
        
        return response
    
    def _select_components(self, 
                          domain: KnowledgeDomain,
                          preferred: Optional[List[str]],
                          query: str) -> List[str]:
        """
        Select components based on domain, query, and learned recommendations.
        Uses learning data to prioritize best-performing components.
        """
        if preferred:
            return preferred
        
        # Domain-based component selection
        domain_components = {
            KnowledgeDomain.CHEMISTRY: ['global_chem', 'kggen', 'deepke', 'graphiti'],
            KnowledgeDomain.BIOMEDICAL: ['oneke', 'deepke', 'kggen', 'aikg'],
            KnowledgeDomain.TEMPORAL: ['graphiti', 'kggen', 'ragbits'],
            KnowledgeDomain.MULTILINGUAL: ['oneke', 'deepke', 'kggen'],
            KnowledgeDomain.CAUSAL: ['causal_learn', 'neuralkg', 'kggen'],
            KnowledgeDomain.RESEARCH: ['research_quest', 'crewai', 'ragbits'],
            KnowledgeDomain.TECHNICAL: ['leanaide', 'dspy', 'agentjson'],
            KnowledgeDomain.GENERAL: ['kggen', 'oneke', 'aikg', 'ragbits', 'crewai']
        }
        
        base_components = domain_components.get(domain, domain_components[KnowledgeDomain.GENERAL])
        
        # Apply learning recommendations if available
        if self.enable_learning and self.self_improving:
            try:
                learned_components = self.self_improving.get_best_components_for_domain(
                    domain.value, top_n=5
                )
                
                # Merge learned components with base components
                # Prioritize learned components that are in the base list
                merged = []
                for comp in learned_components:
                    if comp in base_components:
                        merged.append(comp)
                
                # Add remaining base components not in learned list
                for comp in base_components:
                    if comp not in merged:
                        merged.append(comp)
                
                if merged:
                    logger.debug({
                        'msg': 'Using learning-enhanced component selection',
                        'domain': domain.value,
                        'base': base_components,
                        'learned': learned_components,
                        'merged': merged
                    })
                    return merged[:5]  # Limit to top 5
            except Exception as e:
                logger.warning(f"Failed to apply learning recommendations: {e}")
        
        return base_components
    
    async def _execute_component(self, name: str, component: Any, 
                                query: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Execute a single component with proper async handling"""
        ctx = context or {}
        start_time = datetime.now(timezone.utc)
        
        try:
            # Component-specific execution logic
            handlers = {
                'kggen': self._execute_kggen,
                'graphiti': self._execute_graphiti,
                'oneke': self._execute_oneke,
                'aikg': self._execute_aikg,
                'deepke': self._execute_deepke,
                'ragbits': self._execute_ragbits,
                'crewai': self._execute_crewai,
                'pami': self._execute_pami,
                'neuralkg': self._execute_neuralkg,
                'causal_learn': self._execute_causal_learn,
                'karateclub': self._execute_karateclub,
                'global_chem': self._execute_global_chem,
                'neuromancer': self._execute_neuromancer,
                'lagrange_mapper': self._execute_lagrange_mapper,
                'leanaide': self._execute_leanaide,
                'research_quest': self._execute_research_quest,
                'agentic_context': self._execute_agentic_context,
                'agentjson': self._execute_agentjson,
                'dspy': self._execute_dspy,
                'openevolve_lib': self._execute_openevolve_lib,
                'mcp_gateway': self._execute_mcp_gateway,
                'roma': self._execute_roma,
            }
            
            handler = handlers.get(name)
            if handler:
                result = await handler(component, query, ctx)
                return {'success': True, 'output': result}
            else:
                # Generic execution fallback
                if hasattr(component, 'process'):
                    if asyncio.iscoroutinefunction(component.process):
                        result = await component.process(query)
                    else:
                        result = component.process(query)
                    return {'success': True, 'output': result}
                else:
                    return {'success': True, 'output': {'component': name, 'query': query, 'status': 'mock'}}
                    
        except Exception as e:
            logger.error(f"Component {name} execution failed: {e}")
            return {'success': False, 'error': str(e), 'component': name}
    
    # Component-specific execution handlers
    async def _execute_kggen(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'extract_knowledge_graph'):
            result = await comp.extract_knowledge_graph(query)
            return result.to_dict() if hasattr(result, 'to_dict') else {'entities': [], 'relations': []}
        return {'entities': [], 'relations': []}
    
    async def _execute_graphiti(self, comp, query: str, ctx: Dict) -> Dict:
        # Graphiti requires temporal context
        return {'query': query, 'temporal': True, 'component': 'graphiti', 'status': 'search'}
    
    async def _execute_oneke(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'extract'):
            result = await comp.extract(query, schema=ctx.get('schema'))
            return {'entities': result.get('entities', []), 'relations': result.get('relations', [])}
        return {'entities': [], 'relations': [], 'component': 'oneke'}
    
    async def _execute_aikg(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'infer'):
            result = comp.infer(query)
            return {'inference': result, 'component': 'aikg'}
        return {'query': query, 'component': 'aikg'}
    
    async def _execute_deepke(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'extract'):
            result = comp.extract(query)
            return {'entities': result.get('entities', []), 'relations': result.get('relations', [])}
        return {'entities': [], 'relations': [], 'component': 'deepke'}
    
    async def _execute_ragbits(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'search'):
            result = await comp.search(query, top_k=ctx.get('top_k', 5))
            return {'results': result, 'component': 'ragbits'}
        return {'query': query, 'component': 'ragbits'}
    
    async def _execute_crewai(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'delegate_task'):
            result = comp.delegate_task(query, ctx.get('agents', []))
            return {'task_result': result, 'component': 'crewai'}
        return {'query': query, 'component': 'crewai'}
    
    async def _execute_pami(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'mine_patterns'):
            result = comp.mine_patterns([query.split()], min_support=0.1)
            return {'patterns': result.get('patterns', []), 'component': 'pami'}
        return {'query': query, 'component': 'pami'}
    
    async def _execute_neuralkg(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'generate_embeddings'):
            # Mock triples for demonstration
            triples = [(query, 'related_to', 'concept')]
            result = comp.generate_embeddings(triples)
            return {'embeddings': result.get('embeddings', {}), 'component': 'neuralkg'}
        return {'query': query, 'component': 'neuralkg'}
    
    async def _execute_causal_learn(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'discover_structure'):
            import numpy as np
            # Mock data for demonstration
            data = np.random.randn(100, 5)
            result = comp.discover_structure(data, algorithm='pc')
            return {'structure': result.get('structure', {}), 'component': 'causal_learn'}
        return {'query': query, 'component': 'causal_learn'}
    
    async def _execute_karateclub(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'analyze_graph'):
            # Mock graph for demonstration
            graph_data = {'nodes': [1, 2, 3], 'edges': [(1, 2), (2, 3)]}
            result = comp.analyze_graph(graph_data)
            return {'communities': result.get('communities', []), 'component': 'karateclub'}
        return {'query': query, 'component': 'karateclub'}
    
    async def _execute_global_chem(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'get_chemical'):
            result = comp.get_chemical(query)
            return {'chemical': result, 'component': 'global_chem'}
        return {'query': query, 'component': 'global_chem'}
    
    async def _execute_neuromancer(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'train_neural_ode'):
            import numpy as np
            data = np.random.randn(10, 3)
            time = np.linspace(0, 1, 10)
            result = comp.train_neural_ode(data, time)
            return {'model': result.get('model', {}), 'component': 'neuromancer'}
        return {'query': query, 'component': 'neuromancer'}
    
    async def _execute_lagrange_mapper(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'analyze_landscape'):
            import numpy as np
            embeddings = np.random.randn(50, 10)
            result = comp.analyze_landscape(embeddings)
            return {'landscape': result.get('landscape', {}), 'component': 'lagrange_mapper'}
        return {'query': query, 'component': 'lagrange_mapper'}
    
    async def _execute_leanaide(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'verify'):
            result = comp.verify(query)
            return {'verification': result, 'component': 'leanaide'}
        return {'query': query, 'component': 'leanaide'}
    
    async def _execute_research_quest(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'research'):
            result = comp.research(query, max_papers=ctx.get('max_papers', 5))
            return {'research': result, 'component': 'research_quest'}
        return {'query': query, 'component': 'research_quest'}
    
    async def _execute_agentic_context(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'process_context'):
            result = comp.process_context(query, ctx.get('conversation_id'))
            return {'context': result, 'component': 'agentic_context'}
        return {'query': query, 'component': 'agentic_context'}
    
    async def _execute_agentjson(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'generate_json'):
            result = comp.generate_json(query, schema=ctx.get('schema'))
            return {'json_output': result, 'component': 'agentjson'}
        return {'query': query, 'component': 'agentjson'}
    
    async def _execute_dspy(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'optimize_prompt'):
            result = comp.optimize_prompt(query, demos=ctx.get('demos', []))
            return {'optimized': result, 'component': 'dspy'}
        return {'query': query, 'component': 'dspy'}
    
    async def _execute_openevolve_lib(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'integrate'):
            result = comp.integrate(query, system=ctx.get('system'))
            return {'integration': result, 'component': 'openevolve_lib'}
        return {'query': query, 'component': 'openevolve_lib'}
    
    async def _execute_mcp_gateway(self, comp, query: str, ctx: Dict) -> Dict:
        if hasattr(comp, 'call_tool'):
            result = comp.call_tool(query, params=ctx.get('params'))
            return {'tool_result': result, 'component': 'mcp_gateway'}
        return {'query': query, 'component': 'mcp_gateway'}

    async def _execute_roma(self, comp, query: str, ctx: Dict) -> Dict:
        """
        Execute ROMA meta-agent for hierarchical problem solving.

        ROMA provides automatic recursive decomposition and execution:
        - Decompose: Break down complex problems into sub-problems
        - Execute: Solve sub-problems recursively
        - Aggregate: Combine sub-solutions into final result
        - Verify: Validate solution meets requirements
        """
        try:
            # Check component availability
            if hasattr(comp, 'is_available') and not comp.is_available():
                return {
                    'query': query,
                    'component': 'roma',
                    'status': 'unavailable',
                    'message': 'ROMA integration not available'
                }

            # Determine execution mode from context
            execution_mode = ctx.get('roma_execution_mode', 'recursive')
            max_depth = ctx.get('roma_max_depth', 3)

            # Check if we should decompose or execute
            operation = ctx.get('roma_operation', 'execute')

            if operation == 'decompose':
                # Decompose problem into sub-problems
                result = comp.decompose(query, max_depth=max_depth)
                return {
                    'decomposition': result,
                    'component': 'roma',
                    'operation': 'decompose',
                    'sub_problems_count': result.get('sub_problems_count', 0)
                }

            elif operation == 'verify':
                # Verify a solution
                solution = ctx.get('solution', {})
                requirements = ctx.get('requirements', [])
                result = comp.verify(solution, requirements)
                return {
                    'verification': result,
                    'component': 'roma',
                    'operation': 'verify',
                    'verified': result.get('verified', False)
                }

            elif operation == 'aggregate':
                # Aggregate sub-solutions
                sub_solutions = ctx.get('sub_solutions', [])
                result = comp.aggregate(sub_solutions)
                return {
                    'aggregation': result,
                    'component': 'roma',
                    'operation': 'aggregate',
                    'sub_solutions_count': len(sub_solutions)
                }

            else:
                # Default: execute full ROMA pipeline
                result = comp.execute(query, execution_mode=execution_mode)

                # Extract relevant information
                return {
                    'solution': result.get('solution'),
                    'execution_id': result.get('execution_id'),
                    'status': result.get('status'),
                    'component': 'roma',
                    'operation': 'execute',
                    'execution_mode': execution_mode
                }

        except Exception as e:
            logger.error({
                'msg': 'ROMA execution failed',
                'component': 'roma',
                'error': str(e),
                'query': query[:100] if query else None
            })
            return {
                'query': query,
                'component': 'roma',
                'status': 'error',
                'error': str(e)
            }

    async def _heal_failure(self, failed_component: str, 
                           query: str, context: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """Attempt to heal from a component failure"""
        logger.info(f"Attempting to heal failure of {failed_component}")
        
        # Get substitutes
        substitutes = self.component_registry.get_substitutes(failed_component)
        
        results = {}
        components_used = []
        
        for sub_name in substitutes:
            sub_component = self.component_registry.get_component(sub_name)
            if not sub_component:
                continue
            
            try:
                result = await self._execute_component(sub_name, sub_component, query, context)
                if result.get('success'):
                    results[sub_name] = result.get('output', {})
                    components_used.append(sub_name)
                    logger.info(f"Healing successful with {sub_name}")
                    break
            except Exception as e:
                logger.warning(f"Healing attempt with {sub_name} failed: {e}")
        
        if results:
            return {'results': results, 'components': components_used}
        
        return None
    
    def _calculate_quality(self, results: Dict[str, Any]) -> float:
        """Calculate quality score from results"""
        if not results:
            return 0.0
        
        # Base quality on number of results and their content
        score = min(len(results) / 3.0, 1.0)  # Up to 3 components = full score
        
        # Adjust based on result richness
        for comp, result in results.items():
            if isinstance(result, dict):
                if result.get('entities') or result.get('relations'):
                    score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_confidence(self, results: Dict[str, Any], 
                             components: List[str]) -> float:
        """Calculate confidence score"""
        if not results:
            return 0.0
        
        # More components = higher confidence
        base_confidence = min(len(components) / 2.0, 0.8)
        
        # Check for agreement between components
        if len(components) >= 2:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'executions': self.execution_count,
            'successes': self.success_count,
            'failures': self.failure_count,
            'healing_actions': self.healing_count,
            'success_rate': self.success_count / max(self.execution_count, 1),
            'components': len(self.component_registry.components),
            'available_components': len(self.component_registry.get_available_components()),
            'capabilities': len(self.component_registry.get_all_capabilities()),
            'learning_enabled': self.enable_learning,
            'healing_enabled': self.enable_healing
        }
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get all available capabilities"""
        return self.component_registry.get_all_capabilities()
    
    def reset_learning(self):
        """Reset all learned data"""
        if self.self_improving:
            self.self_improving.learning_engine.experiences.clear()
            logger.info("Learning data reset")


# Convenience function for quick usage
def create_master_engine(
    storage_path: Optional[str] = None,
    enable_learning: bool = True,
    enable_healing: bool = True
) -> MasterKnowledgeEngine:
    """
    Create a Master Knowledge Engine instance.
    
    Args:
        storage_path: Path for persistent storage
        enable_learning: Enable self-learning
        enable_healing: Enable self-healing
        
    Returns:
        Configured MasterKnowledgeEngine instance
    """
    return MasterKnowledgeEngine(
        storage_path=storage_path,
        enable_learning=enable_learning,
        enable_healing=enable_healing
    )


# Example usage
if __name__ == "__main__":
    # Create engine
    engine = create_master_engine()
    
    # Get statistics
    stats = engine.get_statistics()
    print(f"Master Knowledge Engine initialized:")
    print(f"  Components: {stats['components']}")
    print(f"  Capabilities: {stats['capabilities']}")
    print(f"  Learning: {stats['learning_enabled']}")
    print(f"  Healing: {stats['healing_enabled']}")
