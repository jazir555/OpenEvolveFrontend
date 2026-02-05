"""
Unified Knowledge Graph Integration Hub for OpenEvolve

This module provides a centralized hub for all knowledge graph integrations,
enabling seamless access to:

**Extraction & Processing:**
- DeepKE, OneKE, KG-Gen, AI-Knowledge-Graph - Entity and relation extraction
- GlobalChem - Chemical knowledge graph operations

**Embedding & Reasoning:**
- NeuralKG - Neural knowledge graph embeddings
- Causal-Learn - Causal discovery and analysis
- KarateClub - Graph analysis and community detection
- Cognitive-Hydraulics - Hybrid neuro-symbolic reasoning (Soar+ACT-R)

**Query & Generation:**
- Graphiti - Temporal knowledge graph queries
- Outlines - Structured LLM output generation with constraints
- LMQL - Declarative SQL-like queries for LLMs

**Visualization & Simulation:**
- PyGraphistry - GPU-accelerated graph visualization
- Neuromancer - Physics-informed neural operators for simulation

**Conversation & Safety:**
- DTS - Dialogue Tree Search for multi-turn conversation optimization
- Guardrails - AI safety, output validation, and policy enforcement
- ICR - Iterative Contextual Refinements for quality improvement

Business Logic:
    - Unified interface for all KG operations
    - Intelligent routing to appropriate integration based on task type
    - Cross-integration data flow and enrichment
    - Fallback mechanisms for high availability
    - Comprehensive audit logging and metrics

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                 UnifiedKGIntegrationHub                      │
    └──────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    ┌───▼────┐       ┌────▼─────┐      ┌────▼──────┐
    │Extract │       │ Embed    │      │ Analyze   │
    │   &    │       │  &       │      │  &        │
    │ Process│       │ Reason   │      │ Visualize │
    └─┬─────┬┘       └─┬──────┬─┘      └─┬───────┬─┘
      │     │          │      │          │       │
      ▼     ▼          ▼      ▼          ▼       ▼
   DeepKE  OneKE   NeuralKG KG-Gen  KarateClub  PyGraphistry
   KG-Gen  AIKG                     Causal-Learn
   GlobalChem                     Cognitive-Hydraulics  Neuromancer
   Graphiti

    ┌─────────────────────────────────────────────────────────────┐
    │                  Advanced Capabilities                       │
    ├──────────────────┬──────────────────┬───────────────────────┤
    │   Outlines       │     LMQL         │   Neuromancer         │
    │ (Structured      │ (Declarative     │ (Physics-Informed     │
    │  Generation)     │  Queries)        │  Simulation)          │
    └──────────────────┴──────────────────┴───────────────────────┘
    ┌─────────────────────────────────────────────────────────────┐
    │           Cognitive-Hydraulics (Hybrid Reasoning)            │
    │     Soar (System 2) + ACT-R (System 1) + Evolutionary       │
    │        U = P×G - C - HistoryPenalty + Noise(s)              │
    └─────────────────────────────────────────────────────────────┘
    ┌──────────────────────────┬──────────────────────────────────┐
    │      DTS                 │  Guardrails    │  ICR            │
    │ (Conversation            │  (Safety &     │  (Iterative     │
    │  Optimization)           │   Validation)  │   Refinement)   │
    └──────────────────────────┴──────────────────────────────────┘

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
import json
import uuid
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class KGSource(Enum):
    """Sources of knowledge graph data."""
    DEEPKE = "deepke"
    ONEKE = "oneke"
    KG_GEN = "kggen"
    AIKG = "aikg"
    GLOBAL_CHEM = "global_chem"
    NEURALKG = "neuralkg"
    CAUSAL_LEARN = "causal_learn"
    GRAPHITI = "graphiti"
    NEUROMANCER = "neuromancer"
    COGNITIVE_HYDRAULICS = "cognitive_hydraulics"
    USER = "user"
    OPENEVOLVE = "openevolve"


@dataclass
class KnowledgeTriple:
    """Represents a knowledge triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: Optional[KGSource] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert triple to dictionary."""
        data = asdict(self)
        if self.source:
            data['source'] = self.source.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeTriple':
        """Create triple from dictionary."""
        source = None
        if 'source' in data and data['source']:
            try:
                source = KGSource(data['source'])
            except ValueError:
                pass
        
        return cls(
            subject=data['subject'],
            predicate=data['predicate'],
            object=data.get('object', data.get('obj', '')),
            confidence=data.get('confidence', 1.0),
            timestamp=data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            source=source,
            metadata=data.get('metadata', {})
        )


class IntegrationStatus(Enum):
    """Status of an integration."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass
class UnifiedKGConfig:
    """Configuration for the Unified KG Integration Hub."""
    enable_deepke: bool = True
    enable_oneke: bool = True
    enable_kg_gen: bool = True
    enable_neuralkg: bool = True
    enable_karateclub: bool = True
    enable_graphiti: bool = True
    enable_global_chem: bool = True
    enable_causal_learn: bool = True
    enable_pygraphistry: bool = True
    enable_outlines: bool = True
    enable_lmql: bool = True
    enable_neuromancer: bool = True
    enable_cognitive_hydraulics: bool = True
    enable_dts: bool = True
    enable_guardrails: bool = True
    enable_icr: bool = True
    enable_lagrange_mapper: bool = True
    enable_openevolve: bool = True
    enable_z3: bool = True
    enable_pami: bool = True
    default_backend: str = "memory"
    storage_config: Dict[str, Any] = field(default_factory=dict)
    llm_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Standardized result for knowledge extraction."""
    success: bool
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    triples: List[KnowledgeTriple] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'entities': self.entities,
            'relations': self.relations,
            'triples': [t.to_dict() for t in self.triples],
            'metadata': self.metadata,
            'error': self.error
        }


@dataclass
class AnalysisResult:
    """Standardized result for knowledge analysis."""
    success: bool
    findings: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'findings': self.findings,
            'summary': self.summary,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'error': self.error
        }


class IntegrationRegistry:
    """Registry for managing knowledge graph integrations."""
    
    def __init__(self):
        self._integrations: Dict[str, Any] = {}
        self._status: Dict[str, IntegrationStatus] = {}
    
    def register(self, name: str, integration: Any, status: IntegrationStatus = IntegrationStatus.AVAILABLE):
        self._integrations[name] = integration
        self._status[name] = status
        logger.info(f"Registered integration: {name} (status: {status.value})")
    
    def get_integration(self, name: str) -> Optional[Any]:
        return self._integrations.get(name)
    
    def get_status(self, name: str) -> IntegrationStatus:
        return self._status.get(name, IntegrationStatus.UNAVAILABLE)
    
    def list_integrations(self) -> List[str]:
        return list(self._integrations.keys())


class KGOperationType(Enum):
    """Types of knowledge graph operations."""
    ENTITY_EXTRACTION = auto()
    RELATION_EXTRACTION = auto()
    KNOWLEDGE_EMBEDDING = auto()
    LINK_PREDICTION = auto()
    GRAPH_ANALYSIS = auto()
    COMMUNITY_DETECTION = auto()
    CAUSAL_DISCOVERY = auto()
    VISUALIZATION = auto()
    TEMPORAL_QUERY = auto()
    CHEMICAL_ANALYSIS = auto()
    ENTITY_STANDARDIZATION = auto()
    KNOWLEDGE_INFERENCE = auto()
    STRUCTURED_GENERATION = auto()    # Outlines: constrained LLM outputs
    DECLARATIVE_QUERY = auto()        # LMQL: SQL-like LLM queries
    PHYSICS_SIMULATION = auto()       # Neuromancer: physics-informed reasoning
    HYBRID_REASONING = auto()         # Cognitive-Hydraulics: Soar+ACT-R reasoning
    CONVERSATION_OPTIMIZATION = auto() # DTS: dialogue tree search
    SAFETY_VALIDATION = auto()        # Guardrails: AI safety checks
    ITERATIVE_REFINEMENT = auto()     # ICR: contextual refinements
    TOPOLOGICAL_ANALYSIS = auto()     # Lagrange Mapper: attractor landscapes
    EPISODIC_RECORDING = auto()       # Chronicle/Graphiti: episodic memory recording


@dataclass
class IntegrationHealth:
    """Health status of an integration."""
    name: str
    status: IntegrationStatus
    latency_ms: float = 0.0
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_count: int = 0
    success_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'latency_ms': self.latency_ms,
            'last_check': self.last_check.isoformat(),
            'error_count': self.error_count,
            'success_count': self.success_count,
            'details': self.details
        }


@dataclass
class KGOperationResult:
    """Result of a knowledge graph operation."""
    success: bool
    operation_type: KGOperationType
    integration_used: str
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'operation_type': self.operation_type.name,
            'integration_used': self.integration_used,
            'data': self.data if not isinstance(self.data, (bytes, bytearray)) else '<binary>',
            'metadata': self.metadata,
            'errors': self.errors,
            'processing_time_ms': self.processing_time_ms,
            'timestamp': self.timestamp.isoformat()
        }


class UnifiedKGIntegrationHub:
    """
    Centralized hub for all knowledge graph integrations.
    
    This hub provides:
    1. Unified API for all KG operations
    2. Intelligent task routing to best integration
    3. Cross-integration data enrichment
    4. Health monitoring and fallback handling
    5. Comprehensive audit logging
    
    Example:
        >>> hub = UnifiedKGIntegrationHub()
        >>> await hub.initialize()
        >>> 
        >>> # Extract entities from text
        >>> result = await hub.extract_entities(
        ...     text="Apple Inc. was founded by Steve Jobs.",
        ...     method="deepke"
        ... )
        >>> 
        >>> # Generate embeddings
        >>> embedding_result = await hub.generate_embeddings(
        ...     triples=[("Apple", "founded_by", "Steve Jobs")]
        ... )
        >>> 
        >>> # Visualize knowledge graph
        >>> viz_result = await hub.visualize_graph(nodes, edges)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Unified KG Integration Hub.
        
        Args:
            config: Configuration dictionary for all integrations
        """
        self.config = config or {}
        self._integrations: Dict[str, Any] = {}
        self._health_status: Dict[str, IntegrationHealth] = {}
        self._initialized = False
        self.chronicle = None
        self.temporal_manager = None
        
        # Operation routing map: operation_type -> list of integrations (ordered by preference)
        self._routing_map: Dict[KGOperationType, List[str]] = {
            KGOperationType.ENTITY_EXTRACTION: ['deepke', 'oneke', 'kggen', 'aikg'],
            KGOperationType.RELATION_EXTRACTION: ['deepke', 'oneke', 'kggen', 'aikg'],
            KGOperationType.KNOWLEDGE_EMBEDDING: ['neuralkg', 'karateclub'],
            KGOperationType.LINK_PREDICTION: ['neuralkg'],
            KGOperationType.GRAPH_ANALYSIS: ['karateclub', 'pygraphistry'],
            KGOperationType.COMMUNITY_DETECTION: ['karateclub'],
            KGOperationType.CAUSAL_DISCOVERY: ['causal_learn'],
            KGOperationType.VISUALIZATION: ['pygraphistry'],
            KGOperationType.TEMPORAL_QUERY: ['graphiti'],
            KGOperationType.CHEMICAL_ANALYSIS: ['global_chem'],
            KGOperationType.ENTITY_STANDARDIZATION: ['aikg'],
            KGOperationType.KNOWLEDGE_INFERENCE: ['aikg', 'kggen'],
            KGOperationType.STRUCTURED_GENERATION: ['outlines'],
            KGOperationType.DECLARATIVE_QUERY: ['lmql'],
            KGOperationType.PHYSICS_SIMULATION: ['neuromancer'],
            KGOperationType.HYBRID_REASONING: ['cognitive_hydraulics'],
            KGOperationType.CONVERSATION_OPTIMIZATION: ['dts'],
            KGOperationType.SAFETY_VALIDATION: ['guardrails'],
            KGOperationType.ITERATIVE_REFINEMENT: ['icr'],
            KGOperationType.TOPOLOGICAL_ANALYSIS: ['lagrange_mapper'],
            KGOperationType.EPISODIC_RECORDING: ['chronicle', 'graphiti']
        }
        
        logger.info({
            "msg": "UnifiedKGIntegrationHub initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def initialize(self) -> bool:
        """
        Initialize all integrations and check health.
        
        Returns:
            True if at least one integration initialized successfully
        """
        if self._initialized:
            return True
        
        logger.info({
            "msg": "Initializing UnifiedKGIntegrationHub",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Initialize all integrations
        await self._initialize_deepke()
        await self._initialize_neuralkg()
        await self._initialize_karateclub()
        await self._initialize_kggen()
        await self._initialize_oneke()
        await self._initialize_aikg()
        await self._initialize_graphiti()
        await self._initialize_global_chem()
        await self._initialize_causal_learn()
        await self._initialize_pygraphistry()
        await self._initialize_outlines()
        await self._initialize_lmql()
        await self._initialize_neuromancer()
        await self._initialize_cognitive_hydraulics()
        await self._initialize_dts()
        await self._initialize_guardrails()
        await self._initialize_icr()
        await self._initialize_lagrange_mapper()
        
        # Initialize specialized managers
        await self._initialize_chronicle()
        await self._initialize_temporal()
        
        self._initialized = True
        
        # Log initialization results
        available = [name for name, health in self._health_status.items() 
                    if health.status == IntegrationStatus.AVAILABLE]
        
        logger.info({
            "msg": "UnifiedKGIntegrationHub initialization complete",
            "available_integrations": available,
            "total_integrations": len(self._integrations),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return len(available) > 0
    
    async def _initialize_deepke(self):
        """Initialize DeepKE integration."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.deepke_integration import DeepKEIntegration
            integration = DeepKEIntegration(self.config.get('deepke', {}))
            available = integration.is_available()
            
            self._integrations['deepke'] = integration
            self._health_status['deepke'] = IntegrationHealth(
                name='deepke',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={'description': 'Deep Knowledge Extraction'}
            )
        except Exception as e:
            self._health_status['deepke'] = IntegrationHealth(
                name='deepke',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_neuralkg(self):
        """Initialize NeuralKG integration."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.neuralkg_integration import NeuralKGIntegration
            integration = NeuralKGIntegration(self.config.get('neuralkg', {}))
            available = integration.is_available()
            
            self._integrations['neuralkg'] = integration
            self._health_status['neuralkg'] = IntegrationHealth(
                name='neuralkg',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={'description': 'Neural Knowledge Graph Embeddings'}
            )
        except Exception as e:
            self._health_status['neuralkg'] = IntegrationHealth(
                name='neuralkg',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_karateclub(self):
        """Initialize KarateClub integration."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.karateclub_integration import KarateClubIntegration
            integration = KarateClubIntegration(self.config.get('karateclub', {}))
            available = integration.is_available()
            
            self._integrations['karateclub'] = integration
            self._health_status['karateclub'] = IntegrationHealth(
                name='karateclub',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={'description': 'Graph Analysis and Community Detection'}
            )
        except Exception as e:
            self._health_status['karateclub'] = IntegrationHealth(
                name='karateclub',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_kggen(self):
        """Initialize KG-Gen integration."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.kggen_integration import KGGenIntegration
            integration = KGGenIntegration()
            available = True  # KGGen uses LLM, always available
            
            self._integrations['kggen'] = integration
            self._health_status['kggen'] = IntegrationHealth(
                name='kggen',
                status=IntegrationStatus.AVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={'description': 'LLM-based Knowledge Graph Generation'}
            )
        except Exception as e:
            self._health_status['kggen'] = IntegrationHealth(
                name='kggen',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_oneke(self):
        """Initialize OneKE integration."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.oneke_integration import OneKEIntegration
            integration = OneKEIntegration(self.config.get('oneke', {}))
            available = integration.is_available()
            
            self._integrations['oneke'] = integration
            self._health_status['oneke'] = IntegrationHealth(
                name='oneke',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={'description': 'OneKE Bilingual Knowledge Extraction'}
            )
        except Exception as e:
            self._health_status['oneke'] = IntegrationHealth(
                name='oneke',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_aikg(self):
        """Initialize AI-Knowledge-Graph integration."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.aikg_integration import AIKGIntegration
            integration = AIKGIntegration(self.config.get('aikg', {}))
            available = True  # AIKG uses internal components
            
            self._integrations['aikg'] = integration
            self._health_status['aikg'] = IntegrationHealth(
                name='aikg',
                status=IntegrationStatus.AVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={'description': 'AI-driven Knowledge Graph Processing'}
            )
        except Exception as e:
            self._health_status['aikg'] = IntegrationHealth(
                name='aikg',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_graphiti(self):
        """Initialize Graphiti integration."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.graphiti_integration import GraphitiIntegration
            
            # Graphiti requires Neo4j connection
            neo4j_config = self.config.get('neo4j', {})
            integration = GraphitiIntegration(
                uri=neo4j_config.get('uri', 'bolt://localhost:7687'),
                user=neo4j_config.get('user', 'neo4j'),
                password=neo4j_config.get('password', '')
            )
            
            self._integrations['graphiti'] = integration
            self._health_status['graphiti'] = IntegrationHealth(
                name='graphiti',
                status=IntegrationStatus.AVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={'description': 'Temporal Knowledge Graph'}
            )
        except Exception as e:
            self._health_status['graphiti'] = IntegrationHealth(
                name='graphiti',
                status=IntegrationStatus.UNAVAILABLE,
                error_count=1,
                details={'error': str(e), 'note': 'Requires Neo4j connection'}
            )
    
    async def _initialize_global_chem(self):
        """Initialize GlobalChem integration."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.global_chem_integration import GlobalChemIntegration
            integration = GlobalChemIntegration(self.config.get('global_chem', {}))
            available = integration.is_available()
            
            self._integrations['global_chem'] = integration
            self._health_status['global_chem'] = IntegrationHealth(
                name='global_chem',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={'description': 'Chemical Knowledge Graph'}
            )
        except Exception as e:
            self._health_status['global_chem'] = IntegrationHealth(
                name='global_chem',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_causal_learn(self):
        """Initialize Causal-Learn integration from SSOT."""
        start_time = datetime.now(timezone.utc)
        try:
            # Try to import from SSOT (integrations/causal_learn/) first
            try:
                from integrations.causal_learn import (
                    CausalLearnAdapter, 
                    CausalDiscoveryBridge,
                    CAUSAL_LEARN_AVAILABLE
                )
                from .integrations.causal_learn_integration import CausalLearnIntegration
                
                # Use the wrapper which delegates to SSOT
                integration = CausalLearnIntegration(self.config.get('causal_learn', {}))
                
                # Initialize if available
                if CAUSAL_LEARN_AVAILABLE:
                    await integration.initialize()
                    available = True
                else:
                    available = False
                
                self._integrations['causal_learn'] = integration
                self._health_status['causal_learn'] = IntegrationHealth(
                    name='causal_learn',
                    status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                    latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    details={
                        'description': 'Causal Discovery and Analysis (SSOT: integrations/causal_learn/)',
                        'ssot_available': CAUSAL_LEARN_AVAILABLE,
                        'adapter': 'CausalLearnAdapter',
                        'bridge': 'CausalDiscoveryBridge'
                    }
                )
            except ImportError as e:
                # Fall back to wrapper-only
                from .integrations.causal_learn_integration import CausalLearnIntegration
                integration = CausalLearnIntegration()
                self._integrations['causal_learn'] = integration
                self._health_status['causal_learn'] = IntegrationHealth(
                    name='causal_learn',
                    status=IntegrationStatus.DEGRADED,
                    latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    details={
                        'description': 'Causal Discovery (SSOT not available, using fallback)',
                        'error': str(e)
                    }
                )
        except Exception as e:
            self._health_status['causal_learn'] = IntegrationHealth(
                name='causal_learn',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e), 'phase': 'initialization'}
            )
    
    async def _initialize_pygraphistry(self):
        """Initialize PyGraphistry integration."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.pygraphistry_integration import PyGraphistryIntegration
            integration = PyGraphistryIntegration(
                api_key=self.config.get('pygraphistry', {}).get('api_key'),
                config=self.config.get('pygraphistry', {})
            )
            available = integration.is_available()
            
            self._integrations['pygraphistry'] = integration
            self._health_status['pygraphistry'] = IntegrationHealth(
                name='pygraphistry',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={'description': 'GPU-Accelerated Graph Visualization'}
            )
        except Exception as e:
            self._health_status['pygraphistry'] = IntegrationHealth(
                name='pygraphistry',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_outlines(self):
        """Initialize Outlines integration for structured generation."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.outlines.outlines_integration import OutlinesKGIntegration
            integration = OutlinesKGIntegration(self.config.get('outlines', {}))
            available = integration.is_available()
            
            self._integrations['outlines'] = integration
            self._health_status['outlines'] = IntegrationHealth(
                name='outlines',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={
                    'description': 'Structured LLM Output Generation with Regex/JSON Constraints',
                    'features': ['json_schema', 'regex_pattern', 'choices', 'batch_generation'],
                    'ssot_location': 'integrations/outlines/'
                }
            )
        except Exception as e:
            self._health_status['outlines'] = IntegrationHealth(
                name='outlines',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_lmql(self):
        """Initialize LMQL integration for declarative queries."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.lmql.lmql_integration import LMQLKGIntegration
            integration = LMQLKGIntegration(self.config.get('lmql', {}))
            available = integration.is_available()
            
            self._integrations['lmql'] = integration
            self._health_status['lmql'] = IntegrationHealth(
                name='lmql',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={
                    'description': 'SQL-like Query Language for LLMs with Constraint Programming',
                    'features': ['declarative_queries', 'constraint_checking', 'multi_turn', 'cypher_generation'],
                    'ssot_location': 'integrations/lmql/'
                }
            )
        except Exception as e:
            self._health_status['lmql'] = IntegrationHealth(
                name='lmql',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_neuromancer(self):
        """Initialize Neuromancer integration for physics-informed reasoning."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.neuromancer.neuromancer_integration import NeuromancerKGIntegration
            integration = NeuromancerKGIntegration(self.config.get('neuromancer', {}))
            available = integration.is_available()
            
            self._integrations['neuromancer'] = integration
            self._health_status['neuromancer'] = IntegrationHealth(
                name='neuromancer',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={
                    'description': 'Neural Operators for Physics-Informed Knowledge Graphs',
                    'features': ['ode_solving', 'pde_solving', 'dynamics_learning', 'physics_constraints', 'simulation'],
                    'ssot_location': 'integrations/neuromancer/',
                    'domains': ['climate', 'fluids', 'mechanics', 'chemical', 'biological']
                }
            )
        except Exception as e:
            self._health_status['neuromancer'] = IntegrationHealth(
                name='neuromancer',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_cognitive_hydraulics(self):
        """Initialize Cognitive-Hydraulics integration for hybrid reasoning."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.cognitive_hydraulics.cognitive_hydraulics_integration import CognitiveHydraulicsKGIntegration
            integration = CognitiveHydraulicsKGIntegration(self.config.get('cognitive_hydraulics', {}))
            available = integration.is_available()
            
            self._integrations['cognitive_hydraulics'] = integration
            self._health_status['cognitive_hydraulics'] = IntegrationHealth(
                name='cognitive_hydraulics',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={
                    'description': 'Hybrid Neuro-Symbolic Reasoning (Soar+ACT-R+Evolutionary)',
                    'features': [
                        'system2_symbolic_reasoning',
                        'system1_heuristic_reasoning',
                        'pressure_valve_switching',
                        'evolutionary_fallback',
                        'chunking_learning'
                    ],
                    'ssot_location': 'integrations/cognitive_hydraulics/',
                    'equation': 'U = P×G - C - HistoryPenalty + Noise',
                    'systems': ['Soar', 'ACT-R', 'Evolutionary']
                }
            )
        except Exception as e:
            self._health_status['cognitive_hydraulics'] = IntegrationHealth(
                name='cognitive_hydraulics',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_dts(self):
        """Initialize DTS integration for conversation optimization."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.dts.dts_integration import DTSKGIntegration
            integration = DTSKGIntegration(self.config.get('dts', {}))
            available = integration.is_available()
            
            self._integrations['dts'] = integration
            self._health_status['dts'] = IntegrationHealth(
                name='dts',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={
                    'description': 'Dialogue Tree Search - Multi-turn conversation optimization',
                    'features': [
                        'beam_search',
                        'user_simulation',
                        'multi_judge_scoring',
                        'strategy_optimization',
                        'conversation_trees'
                    ],
                    'ssot_location': 'integrations/dts/',
                    'algorithm': 'Parallel beam search with backpropagation'
                }
            )
        except Exception as e:
            self._health_status['dts'] = IntegrationHealth(
                name='dts',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_guardrails(self):
        """Initialize Guardrails integration for AI safety."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.guardrails.guardrails_integration import GuardrailsKGIntegration
            integration = GuardrailsKGIntegration(self.config.get('guardrails', {}))
            available = integration.is_available()
            
            self._integrations['guardrails'] = integration
            self._health_status['guardrails'] = IntegrationHealth(
                name='guardrails',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={
                    'description': 'AI Safety and Output Validation',
                    'features': [
                        'output_validation',
                        'pii_detection',
                        'toxicity_check',
                        'safety_policies',
                        'compliance_gdpr_hipaa'
                    ],
                    'ssot_location': 'integrations/guardrails/',
                    'validators': 10,
                    'safety_levels': ['STRICT', 'MODERATE', 'PERMISSIVE']
                }
            )
        except Exception as e:
            self._health_status['guardrails'] = IntegrationHealth(
                name='guardrails',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_icr(self):
        """Initialize ICR integration for iterative refinement."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.icr.icr_integration import ICRKGIntegration
            integration = ICRKGIntegration(self.config.get('icr', {}))
            available = integration.is_available()
            
            self._integrations['icr'] = integration
            self._health_status['icr'] = IntegrationHealth(
                name='icr',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={
                    'description': 'Iterative Contextual Refinements',
                    'features': [
                        'generate_critique_refine_loop',
                        'quality_judgment',
                        'convergence_detection',
                        'early_stopping',
                        'multi_criteria_evaluation'
                    ],
                    'ssot_location': 'integrations/icr/',
                    'max_iterations': 5,
                    'quality_threshold': 0.9
                }
            )
        except Exception as e:
            self._health_status['icr'] = IntegrationHealth(
                name='icr',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def _initialize_lagrange_mapper(self):
        """Initialize Lagrange Mapper integration for topological analysis."""
        start_time = datetime.now(timezone.utc)
        try:
            from .integrations.lagrange_mapper_integration import LagrangeMapperIntegration
            integration = LagrangeMapperIntegration(self.config.get('lagrange_mapper', {}))
            available = integration.is_available()
            
            self._integrations['lagrange_mapper'] = integration
            self._health_status['lagrange_mapper'] = IntegrationHealth(
                name='lagrange_mapper',
                status=IntegrationStatus.AVAILABLE if available else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={
                    'description': 'Topological Data Analysis and Attractor Landscape Mapping',
                    'features': [
                        'attractor_landscape_analysis',
                        'clustering',
                        'dimensionality_reduction',
                        'knowledge_topology_analysis',
                        'landscape_transition_detection',
                        'basin_of_attraction_computation'
                    ],
                    'ssot_location': 'knowledge_engine/integrations/lagrange_mapper_integration.py',
                    'dependencies': ['numpy', 'scikit-learn'],
                    'applications': [
                        'embedding_space_analysis',
                        'knowledge_graph_topology',
                        'concept_landscape_mapping',
                        'temporal_evolution_tracking'
                    ]
                }
            )
        except Exception as e:
            self._health_status['lagrange_mapper'] = IntegrationHealth(
                name='lagrange_mapper',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    # ============ Public API Methods ============
    
    async def extract_entities(
        self,
        text: str,
        method: Optional[str] = None,
        domain: Optional[str] = None
    ) -> KGOperationResult:
        """
        Extract entities from text using the best available integration.
        
        Args:
            text: Input text
            method: Specific method to use ('deepke', 'oneke', 'kggen', 'aikg')
            domain: Domain hint (e.g., 'chemical', 'biomedical')
            
        Returns:
            KGOperationResult with extracted entities
        """
        start_time = datetime.now(timezone.utc)
        
        # Route to appropriate integration
        if domain == 'chemical' and 'global_chem' in self._integrations:
            integration = self._integrations['global_chem']
            try:
                entities = integration._adapter.recognize_chemical_entities(text)
                return KGOperationResult(
                    success=True,
                    operation_type=KGOperationType.ENTITY_EXTRACTION,
                    integration_used='global_chem',
                    data={'entities': entities},
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"GlobalChem extraction failed: {e}")
        
        # Try specified method or route through standard pipeline
        if method and method in self._integrations:
            return await self._extract_with_method(text, method, start_time)
        
        # Auto-route to best available
        for method_name in self._routing_map[KGOperationType.ENTITY_EXTRACTION]:
            if method_name in self._integrations:
                result = await self._extract_with_method(text, method_name, start_time)
                if result.success:
                    return result
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.ENTITY_EXTRACTION,
            integration_used='none',
            errors=['No suitable extraction method available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def _extract_with_method(
        self,
        text: str,
        method: str,
        start_time: datetime
    ) -> KGOperationResult:
        """Extract entities using specified method."""
        try:
            integration = self._integrations[method]
            
            if method == 'deepke':
                result = integration.extract_entities(text)
            elif method == 'oneke':
                result = integration.extract(text)
            elif method == 'kggen':
                result = integration.extract_graph(text)
            elif method == 'aikg':
                result = await integration.process_knowledge_graph(text)
            else:
                return KGOperationResult(
                    success=False,
                    operation_type=KGOperationType.ENTITY_EXTRACTION,
                    integration_used=method,
                    errors=[f'Unknown method: {method}']
                )
            
            return KGOperationResult(
                success=True,
                operation_type=KGOperationType.ENTITY_EXTRACTION,
                integration_used=method,
                data=result if isinstance(result, dict) else result.to_dict() if hasattr(result, 'to_dict') else result,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.ENTITY_EXTRACTION,
                integration_used=method,
                errors=[str(e)],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
    
    async def generate_embeddings(
        self,
        triples: List[Tuple[str, str, str]],
        model: str = 'transe'
    ) -> KGOperationResult:
        """
        Generate knowledge graph embeddings.
        
        Args:
            triples: List of (head, relation, tail) triples
            model: Embedding model to use
            
        Returns:
            KGOperationResult with embeddings
        """
        start_time = datetime.now(timezone.utc)
        
        if 'neuralkg' in self._integrations:
            try:
                integration = self._integrations['neuralkg']
                result = integration.generate_embeddings(triples, model)
                
                return KGOperationResult(
                    success=result.get('status') == 'success',
                    operation_type=KGOperationType.KNOWLEDGE_EMBEDDING,
                    integration_used='neuralkg',
                    data=result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"NeuralKG embedding failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.KNOWLEDGE_EMBEDDING,
            integration_used='none',
            errors=['No embedding method available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def analyze_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        analysis_type: str = 'general'
    ) -> KGOperationResult:
        """
        Analyze graph structure.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            analysis_type: Type of analysis ('general', 'communities', 'metrics')
            
        Returns:
            KGOperationResult with analysis results
        """
        start_time = datetime.now(timezone.utc)
        
        if analysis_type == 'communities' and 'karateclub' in self._integrations:
            try:
                integration = self._integrations['karateclub']
                result = integration.detect_communities({'nodes': nodes, 'edges': edges})
                
                return KGOperationResult(
                    success=True,
                    operation_type=KGOperationType.COMMUNITY_DETECTION,
                    integration_used='karateclub',
                    data=result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"KarateClub analysis failed: {e}")
        
        # Fall back to PyGraphistry for general analysis
        if 'pygraphistry' in self._integrations:
            try:
                integration = self._integrations['pygraphistry']
                metrics = integration.analyze_graph(nodes, edges)
                
                return KGOperationResult(
                    success=True,
                    operation_type=KGOperationType.GRAPH_ANALYSIS,
                    integration_used='pygraphistry',
                    data=metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"PyGraphistry analysis failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.GRAPH_ANALYSIS,
            integration_used='none',
            errors=['No analysis method available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def discover_causal_structure(
        self,
        data: Any,
        algorithm: str = 'pc',
        variable_names: Optional[List[str]] = None,
        alpha: float = 0.05,
        **kwargs
    ) -> KGOperationResult:
        """
        Discover causal structure from data.
        
        Args:
            data: Data matrix or DataFrame (numpy array or list of lists)
            algorithm: Causal discovery algorithm ('pc', 'fci', 'ges', 'direct_lingam', 'ica_lingam')
            variable_names: Optional names for variables
            alpha: Significance level for independence tests
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            KGOperationResult with causal graph
        """
        start_time = datetime.now(timezone.utc)
        
        if 'causal_learn' in self._integrations:
            try:
                integration = self._integrations['causal_learn']
                
                # Initialize if needed
                if hasattr(integration, 'initialize') and not integration.is_available():
                    await integration.initialize()
                
                # Run discovery
                result = integration.discover_structure(
                    data=data,
                    algorithm=algorithm,
                    variable_names=variable_names,
                    alpha=alpha,
                    **kwargs
                )
                
                return KGOperationResult(
                    success=result.get('status') == 'success',
                    operation_type=KGOperationType.CAUSAL_DISCOVERY,
                    integration_used='causal_learn',
                    data=result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    metadata={
                        'algorithm': algorithm,
                        'alpha': alpha,
                        'variable_count': len(variable_names) if variable_names else 'auto'
                    }
                )
            except Exception as e:
                logger.error(f"Causal-learn discovery failed: {e}")
                return KGOperationResult(
                    success=False,
                    operation_type=KGOperationType.CAUSAL_DISCOVERY,
                    integration_used='causal_learn',
                    errors=[str(e)],
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.CAUSAL_DISCOVERY,
            integration_used='none',
            errors=['Causal-learn not available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def visualize_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        title: Optional[str] = None
    ) -> KGOperationResult:
        """
        Generate graph visualization.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            title: Visualization title
            
        Returns:
            KGOperationResult with visualization URL/path
        """
        start_time = datetime.now(timezone.utc)
        
        if 'pygraphistry' in self._integrations:
            try:
                integration = self._integrations['pygraphistry']
                result = integration.visualize_knowledge_graph(nodes, edges)
                
                return KGOperationResult(
                    success=result.status == 'success',
                    operation_type=KGOperationType.VISUALIZATION,
                    integration_used='pygraphistry',
                    data=result.to_dict() if hasattr(result, 'to_dict') else result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"PyGraphistry visualization failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.VISUALIZATION,
            integration_used='none',
            errors=['PyGraphistry not available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def query_temporal_knowledge(
        self,
        query: str,
        timestamp: Optional[datetime] = None
    ) -> KGOperationResult:
        """
        Query temporal knowledge graph.
        
        Args:
            query: Search query
            timestamp: Point in time (None for current)
            
        Returns:
            KGOperationResult with query results
        """
        start_time = datetime.now(timezone.utc)
        
        if 'graphiti' in self._integrations:
            try:
                integration = self._integrations['graphiti']
                
                if timestamp:
                    results = await integration.query_at_point_in_time(query, timestamp)
                else:
                    results = await integration.search_with_temporal_filters(query)
                
                return KGOperationResult(
                    success=True,
                    operation_type=KGOperationType.TEMPORAL_QUERY,
                    integration_used='graphiti',
                    data=[r.to_dict() if hasattr(r, 'to_dict') else r for r in results],
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"Graphiti query failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.TEMPORAL_QUERY,
            integration_used='none',
            errors=['Graphiti not available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def analyze_chemical(
        self,
        identifier: str,
        analysis_type: str = 'properties'
    ) -> KGOperationResult:
        """
        Analyze chemical compound.
        
        Args:
            identifier: Chemical name or SMILES
            analysis_type: Type of analysis
            
        Returns:
            KGOperationResult with chemical analysis
        """
        start_time = datetime.now(timezone.utc)
        
        if 'global_chem' in self._integrations:
            try:
                integration = self._integrations['global_chem']
                adapter = integration._adapter
                
                if analysis_type == 'properties':
                    result = adapter.get_chemical_properties(identifier)
                elif analysis_type == 'search':
                    result = adapter.search_chemicals(identifier)
                else:
                    result = adapter.get_chemical_by_name(identifier)
                
                return KGOperationResult(
                    success=True,
                    operation_type=KGOperationType.CHEMICAL_ANALYSIS,
                    integration_used='global_chem',
                    data=result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"GlobalChem analysis failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.CHEMICAL_ANALYSIS,
            integration_used='none',
            errors=['GlobalChem not available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def structured_generate(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        method: str = 'json'
    ) -> KGOperationResult:
        """
        Generate structured output using Outlines constraints.
        
        Args:
            prompt: Input prompt
            output_schema: JSON schema or regex pattern for output
            method: 'json', 'regex', or 'choices'
            
        Returns:
            KGOperationResult with structured output
        """
        start_time = datetime.now(timezone.utc)
        
        if 'outlines' in self._integrations:
            try:
                integration = self._integrations['outlines']
                
                if method == 'json':
                    result = await integration.extract_entities_constrained(
                        text=prompt,
                        entity_types=output_schema.get('entity_types', ['entity'])
                    )
                elif method == 'regex':
                    result = await integration.generate_cypher_constrained(
                        schema_desc=str(output_schema),
                        query_intent=prompt
                    )
                else:
                    result = await integration.validate_kg_structure(
                        kg_data={'prompt': prompt, 'schema': output_schema}
                    )
                
                return KGOperationResult(
                    success=True,
                    operation_type=KGOperationType.STRUCTURED_GENERATION,
                    integration_used='outlines',
                    data=result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"Outlines structured generation failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.STRUCTURED_GENERATION,
            integration_used='none',
            errors=['Outlines not available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def declarative_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        query_type: str = 'entities'
    ) -> KGOperationResult:
        """
        Execute declarative LMQL query.
        
        Args:
            query: LMQL query string
            context: Query context variables
            query_type: Type of query ('entities', 'relations', 'cypher', 'multi_hop')
            
        Returns:
            KGOperationResult with query results
        """
        start_time = datetime.now(timezone.utc)
        
        if 'lmql' in self._integrations:
            try:
                integration = self._integrations['lmql']
                
                if query_type == 'entities':
                    result = await integration.query_entities(
                        query_str=query,
                        filters=context or {}
                    )
                elif query_type == 'relations':
                    result = await integration.query_relations(
                        entity_ids=context.get('entity_ids', []),
                        relation_types=context.get('relation_types', [])
                    )
                elif query_type == 'cypher':
                    result = await integration.generate_cypher(
                        natural_language_query=query,
                        schema_description=context.get('schema', '')
                    )
                elif query_type == 'multi_hop':
                    result = await integration.multi_hop_query(
                        start_entity=context.get('start_entity', ''),
                        query_path=context.get('path', [])
                    )
                else:
                    result = await integration.explain_query(query_str=query)
                
                return KGOperationResult(
                    success=result is not None,
                    operation_type=KGOperationType.DECLARATIVE_QUERY,
                    integration_used='lmql',
                    data=result.to_dict() if hasattr(result, 'to_dict') else result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"LMQL query failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.DECLARATIVE_QUERY,
            integration_used='none',
            errors=['LMQL not available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def physics_simulate(
        self,
        system_description: Dict[str, Any],
        simulation_type: str = 'ode',
        time_horizon: float = 10.0
    ) -> KGOperationResult:
        """
        Run physics-informed simulation using Neuromancer.
        
        Args:
            system_description: System entities and relationships
            simulation_type: 'ode', 'pde', 'dynamics', or 'what_if'
            time_horizon: Simulation time horizon
            
        Returns:
            KGOperationResult with simulation results
        """
        start_time = datetime.now(timezone.utc)
        
        if 'neuromancer' in self._integrations:
            try:
                integration = self._integrations['neuromancer']
                
                if simulation_type == 'ode':
                    result = await integration.infer_temporal_dynamics(
                        entity_id=system_description.get('entity_id', ''),
                        property_name=system_description.get('property', 'state'),
                        horizon=int(time_horizon)
                    )
                elif simulation_type == 'what_if':
                    result = await integration.simulate_what_if(
                        scenario=system_description,
                        constraints=system_description.get('constraints', [])
                    )
                elif simulation_type == 'dynamics':
                    result = await integration.calibrate_from_observations(
                        entity_id=system_description.get('entity_id', ''),
                        observations=system_description.get('observations', [])
                    )
                else:
                    result = await integration.validate_physical_laws(
                        kg_subgraph=system_description,
                        domain=system_description.get('domain', 'mechanics')
                    )
                
                return KGOperationResult(
                    success=result is not None,
                    operation_type=KGOperationType.PHYSICS_SIMULATION,
                    integration_used='neuromancer',
                    data=result.to_dict() if hasattr(result, 'to_dict') else result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"Neuromancer simulation failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.PHYSICS_SIMULATION,
            integration_used='none',
            errors=['Neuromancer not available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def hybrid_reasoning(
        self,
        problem: Dict[str, Any],
        goal: str,
        reasoning_mode: str = 'auto'
    ) -> KGOperationResult:
        """
        Execute hybrid neuro-symbolic reasoning using Cognitive-Hydraulics.
        
        Combines Soar (System 2 - symbolic) + ACT-R (System 1 - heuristic) +
        Evolutionary fallback with automatic pressure-based switching.
        
        Args:
            problem: Problem description with context
            goal: Goal to achieve
            reasoning_mode: 'soar', 'actr', 'evolutionary', or 'auto' (default)
            
        Returns:
            KGOperationResult with reasoning result and explanation
        """
        start_time = datetime.now(timezone.utc)
        
        if 'cognitive_hydraulics' in self._integrations:
            try:
                integration = self._integrations['cognitive_hydraulics']
                
                if reasoning_mode == 'auto':
                    # Let pressure valve decide
                    result = await integration.solve_kg_problem(
                        problem_description=problem,
                        goal=goal
                    )
                elif reasoning_mode == 'soar':
                    # Force symbolic reasoning
                    result = await integration.reason_about_graph(
                        kg_subgraph=problem.get('kg', {}),
                        query=goal
                    )
                elif reasoning_mode == 'actr':
                    # Force heuristic reasoning
                    result = await integration.infer_relationship(
                        entity1=problem.get('entity1'),
                        entity2=problem.get('entity2')
                    )
                else:
                    # Evolutionary fallback
                    result = await integration.optimize_query_plan(
                        query=goal
                    )
                
                return KGOperationResult(
                    success=result is not None,
                    operation_type=KGOperationType.HYBRID_REASONING,
                    integration_used='cognitive_hydraulics',
                    data=result.to_dict() if hasattr(result, 'to_dict') else result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"Cognitive-Hydraulics reasoning failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.HYBRID_REASONING,
            integration_used='none',
            errors=['Cognitive-Hydraulics not available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def optimize_conversation(
        self,
        initial_context: str,
        goal: str,
        rounds: int = 3,
        beam_width: int = 5
    ) -> KGOperationResult:
        """
        Optimize multi-turn conversation using Dialogue Tree Search (DTS).
        
        Explores conversation strategies in parallel, simulates user reactions,
        scores trajectories, and prunes underperformers.
        
        Args:
            initial_context: Starting conversation context
            goal: Conversation goal
            rounds: Number of optimization rounds
            beam_width: Number of conversation branches to maintain
            
        Returns:
            KGOperationResult with optimized conversation tree
        """
        start_time = datetime.now(timezone.utc)
        
        if 'dts' in self._integrations:
            try:
                integration = self._integrations['dts']
                result = await integration.optimize_kg_query_dialog(
                    context=initial_context,
                    user_goal=goal
                )
                
                return KGOperationResult(
                    success=result is not None,
                    operation_type=KGOperationType.CONVERSATION_OPTIMIZATION,
                    integration_used='dts',
                    data=result.to_dict() if hasattr(result, 'to_dict') else result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"DTS conversation optimization failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.CONVERSATION_OPTIMIZATION,
            integration_used='none',
            errors=['DTS not available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def validate_safety(
        self,
        content: str,
        content_type: str = 'output',
        safety_level: str = 'MODERATE'
    ) -> KGOperationResult:
        """
        Validate content safety using Guardrails.
        
        Checks for PII, toxicity, policy violations, and schema compliance.
        
        Args:
            content: Content to validate
            content_type: 'input' or 'output'
            safety_level: 'STRICT', 'MODERATE', or 'PERMISSIVE'
            
        Returns:
            KGOperationResult with validation results
        """
        start_time = datetime.now(timezone.utc)
        
        if 'guardrails' in self._integrations:
            try:
                integration = self._integrations['guardrails']
                
                if content_type == 'input':
                    result = await integration.sanitize_kg_input(content)
                else:
                    result = await integration.validate_kg_output(
                        output=content,
                        schema={}
                    )
                
                return KGOperationResult(
                    success=result is not None,
                    operation_type=KGOperationType.SAFETY_VALIDATION,
                    integration_used='guardrails',
                    data=result.to_dict() if hasattr(result, 'to_dict') else result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"Guardrails validation failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.SAFETY_VALIDATION,
            integration_used='none',
            errors=['Guardrails not available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def refine_iteratively(
        self,
        initial_output: str,
        goal: str,
        max_iterations: int = 5,
        quality_threshold: float = 0.9
    ) -> KGOperationResult:
        """
        Refine output iteratively using ICR (Iterative Contextual Refinements).
        
        Generates critique, applies improvements, and judges quality until
        threshold is met or max iterations reached.
        
        Args:
            initial_output: Initial output to refine
            goal: Quality goal description
            max_iterations: Maximum refinement iterations
            quality_threshold: Quality score threshold (0-1)
            
        Returns:
            KGOperationResult with refined output
        """
        start_time = datetime.now(timezone.utc)
        
        if 'icr' in self._integrations:
            try:
                integration = self._integrations['icr']
                result = await integration.refine_kg_extraction(
                    initial_extraction={'content': initial_output, 'goal': goal}
                )
                
                return KGOperationResult(
                    success=result is not None,
                    operation_type=KGOperationType.ITERATIVE_REFINEMENT,
                    integration_used='icr',
                    data=result.to_dict() if hasattr(result, 'to_dict') else result,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            except Exception as e:
                logger.error(f"ICR refinement failed: {e}")
        
        return KGOperationResult(
            success=False,
            operation_type=KGOperationType.ITERATIVE_REFINEMENT,
            integration_used='none',
            errors=['ICR not available'],
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    async def analyze_topological_landscape(
        self,
        embeddings: Any,
        labels: Optional[List[str]] = None,
        n_clusters: int = 8,
        reduction_method: str = 'pca',
        reduction_dims: int = 2,
        analysis_type: str = 'landscape'
    ) -> KGOperationResult:
        """
        Analyze topological landscape of embeddings using Lagrange Mapper.
        
        Identifies attractors, clusters, and basins of attraction in knowledge
        embedding spaces. Useful for understanding concept landscapes and
        knowledge topology.
        
        Args:
            embeddings: Embedding matrix (n_samples x n_features) or list of vectors
            labels: Optional labels for each embedding point
            n_clusters: Number of clusters to identify (default: 8)
            reduction_method: Dimensionality reduction ('pca', 'tsne', or 'none')
            reduction_dims: Dimensions to reduce to for visualization (default: 2)
            analysis_type: Type of analysis ('landscape', 'topology', 'transitions', 'basins')
            
        Returns:
            KGOperationResult with landscape analysis containing:
                - cluster_labels: Assigned cluster for each point
                - cluster_centers: Centroid of each cluster
                - attractors: Attractor strengths and properties
                - reduced_embeddings: 2D/3D coordinates for visualization
                - clusters: Detailed cluster statistics
                
        Example:
            >>> # Analyze knowledge embedding landscape
            >>> embeddings = np.random.randn(100, 128)  # 100 knowledge items
            >>> result = await hub.analyze_topological_landscape(
            ...     embeddings=embeddings,
            ...     labels=[f'concept_{i}' for i in range(100)],
            ...     n_clusters=5,
            ...     analysis_type='landscape'
            ... )
            >>> print(f"Found {len(result.data['attractors'])} attractors")
        """
        start_time = datetime.now(timezone.utc)
        
        if 'lagrange_mapper' not in self._integrations:
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.TOPOLOGICAL_ANALYSIS,
                integration_used='none',
                errors=['Lagrange Mapper not available'],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
        
        try:
            import numpy as np
            integration = self._integrations['lagrange_mapper']
            
            # Convert embeddings to numpy array if needed
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Perform analysis based on type
            if analysis_type == 'landscape':
                result = integration.analyze_landscape(
                    embeddings=embeddings,
                    labels=labels,
                    n_clusters=n_clusters
                )
            elif analysis_type == 'topology':
                # Convert embeddings to graph format for topology analysis
                graph_data = {
                    'nodes': [{'id': labels[i] if labels else f'node_{i}'} for i in range(len(embeddings))],
                    'edges': []  # Would need adjacency info for full topology
                }
                if hasattr(integration._analyzer, 'analyze_knowledge_topology'):
                    result = integration._analyzer.analyze_knowledge_topology(graph_data)
                else:
                    result = integration.analyze_landscape(embeddings, labels, n_clusters)
            elif analysis_type == 'basins':
                # First get landscape, then compute basins
                landscape = integration.analyze_landscape(embeddings, labels, n_clusters)
                if landscape.get('status') == 'success':
                    centers = np.array(landscape['landscape']['cluster_centers'])
                    result = integration._analyzer.find_attractor_basins(embeddings, centers)
                else:
                    result = landscape
            else:
                result = integration.analyze_landscape(embeddings, labels, n_clusters)
            
            return KGOperationResult(
                success=result.get('status') == 'success',
                operation_type=KGOperationType.TOPOLOGICAL_ANALYSIS,
                integration_used='lagrange_mapper',
                data=result,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error(f"Lagrange Mapper analysis failed: {e}")
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.TOPOLOGICAL_ANALYSIS,
                integration_used='lagrange_mapper',
                errors=[str(e)],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

    async def _initialize_chronicle(self):
        """Initialize episodic chronicle memory."""
        start_time = datetime.now(timezone.utc)
        try:
            from .chronicle.chronicle import Chronicle
            storage_path = self.config.get('chronicle', {}).get('storage_path')
            self.chronicle = Chronicle(storage_path=storage_path)
            
            self._health_status['chronicle'] = IntegrationHealth(
                name='chronicle',
                status=IntegrationStatus.AVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details={'description': 'Episodic Chronicle Memory'}
            )
        except Exception as e:
            logger.error(f"Failed to initialize chronicle: {e}")
            self._health_status['chronicle'] = IntegrationHealth(
                name='chronicle',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )

    async def _initialize_temporal(self):
        """Initialize temporal knowledge manager (Graphiti)."""
        start_time = datetime.now(timezone.utc)
        try:
            from .temporal import TemporalKnowledgeManager
            self.temporal_manager = TemporalKnowledgeManager()
            
            status = self.temporal_manager.get_status()
            self._health_status['temporal_manager'] = IntegrationHealth(
                name='temporal_manager',
                status=IntegrationStatus.AVAILABLE if status['available'] else IntegrationStatus.UNAVAILABLE,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                details=status
            )
        except Exception as e:
            logger.error(f"Failed to initialize temporal manager: {e}")
            self._health_status['temporal_manager'] = IntegrationHealth(
                name='temporal_manager',
                status=IntegrationStatus.ERROR,
                error_count=1,
                details={'error': str(e)}
            )
    
    async def detect_landscape_transitions(
        self,
        embeddings_t1: Any,
        embeddings_t2: Any,
        labels: Optional[List[str]] = None
    ) -> KGOperationResult:
        """
        Detect transitions in knowledge landscape between two time points.
        
        Compares embedding landscapes at two different times to identify:
        - Created attractors (new knowledge clusters)
        - Destroyed attractors (disappeared concepts)
        - Persisted attractors (stable knowledge)
        - Strength changes (evolving importance)
        
        Args:
            embeddings_t1: Embeddings at time t1
            embeddings_t2: Embeddings at time t2
            labels: Optional labels for embedding points
            
        Returns:
            KGOperationResult with transition analysis
        """
        start_time = datetime.now(timezone.utc)
        
        if 'lagrange_mapper' not in self._integrations:
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.TOPOLOGICAL_ANALYSIS,
                integration_used='none',
                errors=['Lagrange Mapper not available'],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
        
        try:
            import numpy as np
            integration = self._integrations['lagrange_mapper']
            
            # Convert to numpy arrays
            if not isinstance(embeddings_t1, np.ndarray):
                embeddings_t1 = np.array(embeddings_t1)
            if not isinstance(embeddings_t2, np.ndarray):
                embeddings_t2 = np.array(embeddings_t2)
            
            # Detect transitions
            if hasattr(integration._analyzer, 'detect_landscape_transitions'):
                result = integration._analyzer.detect_landscape_transitions(
                    embeddings_t1, embeddings_t2
                )
            else:
                # Fallback: analyze both and compare
                landscape1 = integration.analyze_landscape(embeddings_t1, labels)
                landscape2 = integration.analyze_landscape(embeddings_t2, labels)
                result = {
                    'status': 'success',
                    'transitions': {
                        'attractors_t1': len(landscape1.get('landscape', {}).get('attractors', [])),
                        'attractors_t2': len(landscape2.get('landscape', {}).get('attractors', [])),
                        'comparison': 'manual_comparison_required'
                    }
                }
            
            return KGOperationResult(
                success=result.get('status') == 'success',
                operation_type=KGOperationType.TOPOLOGICAL_ANALYSIS,
                integration_used='lagrange_mapper',
                data=result,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error(f"Landscape transition detection failed: {e}")
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.TOPOLOGICAL_ANALYSIS,
                integration_used='lagrange_mapper',
                errors=[str(e)],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
    
    async def extract_relations(
        self,
        text: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        relation_types: Optional[List[str]] = None,
        extractor: str = 'deepke'
    ) -> KGOperationResult:
        """
        Extract relations from text using specified extractor.
        
        Args:
            text: Input text to extract relations from
            entities: Optional pre-extracted entities to use as relation endpoints
            relation_types: Optional list of relation types to extract
            extractor: Extractor to use ('deepke', 'oneke', 'kggen')
            
        Returns:
            KGOperationResult with extracted relations
        """
        start_time = datetime.now(timezone.utc)
        
        if extractor not in self._integrations:
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.RELATION_EXTRACTION,
                integration_used='none',
                errors=[f"Extractor {extractor} not found"],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

    async def record_episode(
        self,
        name: str,
        content: str,
        agent: str = "workflow",
        episode_type: str = "action",
        context: Optional[Dict[str, Any]] = None,
        outcome: Optional[str] = None,
        lesson_learned: Optional[str] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> KGOperationResult:
        """
        Record a knowledge episode in episodic memory (Chronicle and Graphiti).
        
        Args:
            name: Name of the episode
            content: Detailed body of the episode
            agent: Agent or component recording the episode
            episode_type: Type of episode (action, decision, observation, etc.)
            context: Additional structured context
            outcome: Outcome of the episode
            lesson_learned: Knowledge gained from the episode
            tags: Categorization tags
            session_id: Optional session identifier
            
        Returns:
            KGOperationResult with recorded IDs
        """
        start_time = datetime.now(timezone.utc)
        ids = {}
        errors = []
        
        # 1. Record in local Chronicle (Narrative Memory)
        if self.chronicle:
            try:
                from .chronicle.chronicle import EpisodeType as ChronEpisodeType
                try:
                    c_type = ChronEpisodeType(episode_type.lower())
                except ValueError:
                    c_type = ChronEpisodeType.ACTION
                    
                ep_id = self.chronicle.record_episode(
                    agent=agent,
                    action=name,
                    episode_type=c_type,
                    context={**(context or {}), "content": content},
                    outcome=outcome,
                    lesson_learned=lesson_learned,
                    tags=tags,
                    session_id=session_id
                )
                ids['chronicle_id'] = ep_id
            except Exception as e:
                errors.append(f"Chronicle recording failed: {e}")
        
        # 2. Record in temporal Graph (Graphiti)
        if self.temporal_manager and self.temporal_manager.available:
            try:
                # Combine name and content for Graphiti if needed
                graphiti_content = f"{name}\n\n{content}"
                if lesson_learned:
                    graphiti_content += f"\n\nLesson Learned: {lesson_learned}"
                if outcome:
                    graphiti_content += f"\n\nOutcome: {outcome}"
                    
                g_id = await self.temporal_manager.record_episode(
                    name=name,
                    content=graphiti_content,
                    source=agent
                )
                if g_id:
                    ids['graphiti_uuid'] = g_id
            except Exception as e:
                errors.append(f"Temporal graph recording failed: {e}")
                
        return KGOperationResult(
            success=len(ids) > 0,
            operation_type=KGOperationType.EPISODIC_RECORDING,
            integration_used='chronicle+graphiti',
            data=ids,
            errors=errors,
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
        
        try:
            integration = self._integrations[extractor]
            
            # Call appropriate method based on extractor
            if extractor == 'deepke' and hasattr(integration, 'extract_relations'):
                result = await integration.extract_relations(text, entities, relation_types)
            elif extractor == 'oneke' and hasattr(integration, 'extract_relations'):
                result = await integration.extract_relations(text, entities, relation_types)
            elif extractor == 'kggen' and hasattr(integration, 'generate_from_text'):
                # KG-Gen generates both entities and relations
                kg_result = await integration.generate_from_text(text)
                result = {'relations': kg_result.get('relations', [])}
            else:
                # Generic fallback
                result = {'relations': [], 'text': text}
            
            return KGOperationResult(
                success=True,
                operation_type=KGOperationType.RELATION_EXTRACTION,
                integration_used=extractor,
                data=result,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.RELATION_EXTRACTION,
                integration_used=extractor,
                errors=[str(e)],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
    
    async def generate_knowledge_graph(
        self,
        documents: List[str],
        schema_hints: Optional[Dict[str, Any]] = None,
        extraction_method: str = 'auto'
    ) -> KGOperationResult:
        """
        Generate a complete knowledge graph from documents.
        
        Args:
            documents: List of text documents to process
            schema_hints: Optional schema hints for entity/relation types
            extraction_method: Method to use ('auto', 'kggen', 'oneke', 'ensemble')
            
        Returns:
            KGOperationResult with generated knowledge graph
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Use KG-Gen if available and method is appropriate
            if 'kggen' in self._integrations and extraction_method in ('auto', 'kggen'):
                integration = self._integrations['kggen']
                if hasattr(integration, 'generate_from_documents'):
                    result = await integration.generate_from_documents(documents, schema_hints)
                    
                    return KGOperationResult(
                        success=True,
                        operation_type=KGOperationType.KNOWLEDGE_EMBEDDING,
                        integration_used='kggen',
                        data=result,
                        processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    )
            
            # Fallback: extract from each document and merge
            all_entities = []
            all_relations = []
            
            for doc in documents:
                # Extract entities
                entity_result = await self.extract_entities(doc)
                if entity_result.success:
                    all_entities.extend(entity_result.data.get('entities', []))
                
                # Extract relations
                relation_result = await self.extract_relations(doc)
                if relation_result.success:
                    all_relations.extend(relation_result.data.get('relations', []))
            
            # Deduplicate
            seen_entities = set()
            unique_entities = []
            for e in all_entities:
                key = e.get('name', str(e))
                if key not in seen_entities:
                    seen_entities.add(key)
                    unique_entities.append(e)
            
            seen_relations = set()
            unique_relations = []
            for r in all_relations:
                key = (r.get('head', ''), r.get('relation', ''), r.get('tail', ''))
                if key not in seen_relations:
                    seen_relations.add(key)
                    unique_relations.append(r)
            
            result = {
                'entities': unique_entities,
                'relations': unique_relations,
                'source_documents': len(documents)
            }
            
            return KGOperationResult(
                success=True,
                operation_type=KGOperationType.KNOWLEDGE_EMBEDDING,
                integration_used='ensemble',
                data=result,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error(f"Knowledge graph generation failed: {e}")
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.KNOWLEDGE_EMBEDDING,
                integration_used='none',
                errors=[str(e)],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
    
    async def mine_patterns(
        self,
        graph_data: Dict[str, Any],
        algorithm: str = 'gspan',
        min_support: float = 0.1,
        pattern_type: str = 'subgraph'
    ) -> KGOperationResult:
        """
        Mine patterns from knowledge graph using PAMI.
        
        Args:
            graph_data: Knowledge graph with nodes and edges
            algorithm: Pattern mining algorithm ('gspan', 'fsg', 'ffsm')
            min_support: Minimum support threshold (0-1)
            pattern_type: Type of pattern ('subgraph', 'sequential', 'periodic')
            
        Returns:
            KGOperationResult with mined patterns
        """
        start_time = datetime.now(timezone.utc)
        
        if 'pami' not in self._integrations:
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.GRAPH_ANALYSIS,
                integration_used='none',
                errors=['PAMI integration not available'],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
        
        try:
            integration = self._integrations['pami']
            
            # Convert graph to format expected by PAMI
            if hasattr(integration, 'mine_patterns'):
                result = await integration.mine_patterns(
                    graph_data,
                    algorithm=algorithm,
                    min_support=min_support,
                    pattern_type=pattern_type
                )
            else:
                # Fallback: return empty result
                result = {
                    'patterns': [],
                    'algorithm': algorithm,
                    'min_support': min_support
                }
            
            return KGOperationResult(
                success=True,
                operation_type=KGOperationType.GRAPH_ANALYSIS,
                integration_used='pami',
                data=result,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error(f"Pattern mining failed: {e}")
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.GRAPH_ANALYSIS,
                integration_used='pami',
                errors=[str(e)],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
    
    async def standardize_entities(
        self,
        entities: List[Dict[str, Any]],
        target_schema: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.85
    ) -> KGOperationResult:
        """
        Standardize entities to a common schema using AI-KG.
        
        Args:
            entities: List of entities to standardize
            target_schema: Optional target schema to standardize to
            similarity_threshold: Threshold for entity deduplication
            
        Returns:
            KGOperationResult with standardized entities
        """
        start_time = datetime.now(timezone.utc)
        
        # Try AI-KG integration first
        if 'aikg' in self._integrations:
            try:
                integration = self._integrations['aikg']
                
                if hasattr(integration, 'standardize_entities'):
                    result = await integration.standardize_entities(
                        entities,
                        target_schema=target_schema,
                        similarity_threshold=similarity_threshold
                    )
                    
                    return KGOperationResult(
                        success=True,
                        operation_type=KGOperationType.ENTITY_STANDARDIZATION,
                        integration_used='aikg',
                        data=result,
                        processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    )
            except Exception as e:
                logger.warning(f"AI-KG standardization failed: {e}")
        
        # Fallback: basic deduplication
        try:
            seen = {}
            standardized = []
            
            for entity in entities:
                name = entity.get('name', '').lower().strip()
                entity_type = entity.get('type', 'Unknown')
                
                # Create signature
                sig = (name, entity_type)
                
                if sig not in seen:
                    seen[sig] = entity
                    standardized.append(entity)
                else:
                    # Merge with existing
                    existing = seen[sig]
                    # Keep the one with more properties
                    if len(entity) > len(existing):
                        seen[sig] = entity
                        standardized = [e for e in standardized if (e.get('name', '').lower().strip(), e.get('type', 'Unknown')) != sig]
                        standardized.append(entity)
            
            result = {
                'entities': standardized,
                'original_count': len(entities),
                'standardized_count': len(standardized),
                'duplicates_removed': len(entities) - len(standardized)
            }
            
            return KGOperationResult(
                success=True,
                operation_type=KGOperationType.ENTITY_STANDARDIZATION,
                integration_used='fallback',
                data=result,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error(f"Entity standardization failed: {e}")
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.ENTITY_STANDARDIZATION,
                integration_used='none',
                errors=[str(e)],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
    
    async def infer_knowledge(
        self,
        graph_data: Dict[str, Any],
        inference_type: str = 'transitive',
        max_hops: int = 2
    ) -> KGOperationResult:
        """
        Infer new knowledge from existing graph using AI-KG.
        
        Args:
            graph_data: Knowledge graph with nodes and edges
            inference_type: Type of inference ('transitive', 'symmetric', 'inverse')
            max_hops: Maximum hops for transitive inference
            
        Returns:
            KGOperationResult with inferred knowledge
        """
        start_time = datetime.now(timezone.utc)
        
        # Try AI-KG integration
        if 'aikg' in self._integrations:
            try:
                integration = self._integrations['aikg']
                
                if hasattr(integration, 'infer_relations'):
                    result = await integration.infer_relations(
                        graph_data,
                        inference_type=inference_type,
                        max_hops=max_hops
                    )
                    
                    return KGOperationResult(
                        success=True,
                        operation_type=KGOperationType.KNOWLEDGE_INFERENCE,
                        integration_used='aikg',
                        data=result,
                        processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    )
            except Exception as e:
                logger.warning(f"AI-KG inference failed: {e}")
        
        # Fallback: basic transitive inference
        try:
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            inferred_edges = []
            
            if inference_type == 'transitive':
                # Build adjacency list
                adj = {}
                for edge in edges:
                    src = edge.get('source')
                    tgt = edge.get('target')
                    rel = edge.get('relation')
                    if src and tgt:
                        if src not in adj:
                            adj[src] = []
                        adj[src].append((tgt, rel))
                
                # Find transitive paths
                for start_node in adj:
                    visited = {start_node: 0}
                    queue = [(start_node, 0, [])]
                    
                    while queue:
                        current, hops, path = queue.pop(0)
                        
                        if hops >= max_hops:
                            continue
                        
                        if current in adj:
                            for neighbor, rel in adj[current]:
                                if neighbor not in visited or visited[neighbor] > hops + 1:
                                    visited[neighbor] = hops + 1
                                    new_path = path + [(current, rel, neighbor)]
                                    queue.append((neighbor, hops + 1, new_path))
                                    
                                    # Add inferred edge if path length > 1
                                    if len(new_path) > 1:
                                        inferred_edges.append({
                                            'source': start_node,
                                            'target': neighbor,
                                            'relation': f'inferred_{hops+1}_hop',
                                            'confidence': 0.7 ** (hops + 1),
                                            'inferred': True,
                                            'path': new_path
                                        })
            
            result = {
                'original_edges': len(edges),
                'inferred_edges': len(inferred_edges),
                'edges': inferred_edges,
                'inference_type': inference_type
            }
            
            return KGOperationResult(
                success=True,
                operation_type=KGOperationType.KNOWLEDGE_INFERENCE,
                integration_used='fallback',
                data=result,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error(f"Knowledge inference failed: {e}")
            return KGOperationResult(
                success=False,
                operation_type=KGOperationType.KNOWLEDGE_INFERENCE,
                integration_used='none',
                errors=[str(e)],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
    
    # ============ Health and Monitoring ============
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all integrations.
        
        Returns:
            Dictionary with health status for all integrations
        """
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'integrations': {name: health.to_dict() for name, health in self._health_status.items()},
            'summary': {
                'total': len(self._health_status),
                'available': sum(1 for h in self._health_status.values() if h.status == IntegrationStatus.AVAILABLE),
                'unavailable': sum(1 for h in self._health_status.values() if h.status == IntegrationStatus.UNAVAILABLE),
                'error': sum(1 for h in self._health_status.values() if h.status == IntegrationStatus.ERROR)
            }
        }
    
    def get_available_integrations(self) -> List[str]:
        """
        Get list of available integration names.
        
        Returns:
            List of available integration names
        """
        return [
            name for name, health in self._health_status.items()
            if health.status == IntegrationStatus.AVAILABLE
        ]
    
    async def execute_pipeline(
        self,
        text: str,
        pipeline_config: List[Dict[str, Any]]
    ) -> List[KGOperationResult]:
        """
        Execute a multi-step processing pipeline.
        
        Args:
            text: Input text
            pipeline_config: List of pipeline steps
            
        Returns:
            List of operation results
        """
        results = []
        current_data = text
        
        for step in pipeline_config:
            operation = step.get('operation')
            params = step.get('params', {})
            
            if operation == 'extract':
                result = await self.extract_entities(current_data, **params)
                current_data = result.data
            elif operation == 'embed':
                result = await self.generate_embeddings(current_data, **params)
            elif operation == 'analyze':
                result = await self.analyze_graph(current_data.get('nodes', []), current_data.get('edges', []), **params)
            elif operation == 'visualize':
                result = await self.visualize_graph(
                    current_data.get('nodes', []),
                    current_data.get('edges', []),
                    **params
                )
            else:
                result = KGOperationResult(
                    success=False,
                    operation_type=KGOperationType.ENTITY_EXTRACTION,
                    integration_used='none',
                    errors=[f'Unknown operation: {operation}']
                )
            
            results.append(result)
            
            if not result.success and step.get('stop_on_error', True):
                break
        
        return results


# Convenience functions for direct usage
async def extract_knowledge(
    text: str,
    method: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick knowledge extraction from text.
    
    Args:
        text: Input text
        method: Extraction method
        
    Returns:
        Extraction results
    """
    hub = UnifiedKGIntegrationHub()
    await hub.initialize()
    result = await hub.extract_entities(text, method)
    return result.to_dict()


async def visualize_knowledge_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick knowledge graph visualization.
    
    Args:
        nodes: Graph nodes
        edges: Graph edges
        title: Visualization title
        
    Returns:
        Visualization results
    """
    hub = UnifiedKGIntegrationHub()
    await hub.initialize()
    result = await hub.visualize_graph(nodes, edges, title)
    return result.to_dict()


async def create_unified_hub(config: Optional[UnifiedKGConfig] = None) -> UnifiedKGIntegrationHub:
    """
    Factory function to create and initialize a UnifiedKGIntegrationHub.
    
    Args:
        config: Optional configuration
        
    Returns:
        Initialized UnifiedKGIntegrationHub
    """
    hub = UnifiedKGIntegrationHub(config.__dict__ if config else None)
    await hub.initialize()
    return hub


async def quick_extract(text: str, method: Optional[str] = None) -> List[KnowledgeTriple]:
    """
    Quickly extract knowledge triples from text.
    
    Args:
        text: Input text
        method: Optional extraction method
        
    Returns:
        List of extracted KnowledgeTriples
    """
    hub = await create_unified_hub()
    result = await hub.extract_entities(text, method)
    
    triples = []
    if result.success and isinstance(result.data, dict):
        # Handle different result formats from different methods
        extracted = result.data.get('entities', []) or result.data.get('triples', [])
        for item in extracted:
            if isinstance(item, dict):
                triples.append(KnowledgeTriple.from_dict({
                    **item,
                    'source': result.integration_used
                }))
    
    return triples


__all__ = [
    'UnifiedKGIntegrationHub',
    'UnifiedKGConfig',
    'KnowledgeTriple',
    'KGSource',
    'KGOperationType',
    'KGOperationResult',
    'IntegrationHealth',
    'IntegrationStatus',
    'create_unified_hub',
    'quick_extract',
    'extract_knowledge',
    'visualize_knowledge_graph',
    'analyze_causal_relationships'
]
