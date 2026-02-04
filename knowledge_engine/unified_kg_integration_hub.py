"""
Unified Knowledge Graph Integration Hub for OpenEvolve

This module provides a centralized hub for all knowledge graph integrations,
enabling seamless access to DeepKE, NeuralKG, KarateClub, KG-Gen, OneKE,
AI-Knowledge-Graph, Graphiti, GlobalChem, Causal-Learn, and PyGraphistry.

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
   GlobalChem
   Graphiti

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
import json
import uuid
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


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


class IntegrationStatus(Enum):
    """Status of an integration."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    ERROR = "error"


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
            KGOperationType.KNOWLEDGE_INFERENCE: ['aikg', 'kggen']
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


async def analyze_causal_relationships(
    data: Any,
    algorithm: str = 'pc'
) -> Dict[str, Any]:
    """
    Quick causal discovery from data.
    
    Args:
        data: Data matrix
        algorithm: Causal discovery algorithm
        
    Returns:
        Causal graph results
    """
    hub = UnifiedKGIntegrationHub()
    await hub.initialize()
    result = await hub.discover_causal_structure(data, algorithm)
    return result.to_dict()


__all__ = [
    'UnifiedKGIntegrationHub',
    'KGOperationType',
    'KGOperationResult',
    'IntegrationHealth',
    'IntegrationStatus',
    'extract_knowledge',
    'visualize_knowledge_graph',
    'analyze_causal_relationships'
]
