"""
Global Knowledge Graph Orchestrator

A unified system that combines ALL knowledge graph project integrations into a
single cohesive framework. This orchestrator leverages the unique capabilities
of each integration to provide comprehensive KG processing.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │              GlobalKGOrchestrator                           │
    │         (Unified Interface for ALL KG Operations)           │
    └──────────────────────┬──────────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
    ┌──▼────┐        ┌────▼─────┐       ┌────▼──────┐
    │Extract│        │ Reason   │       │  Safety   │
    └──┬────┘        └────┬─────┘       └─────┬─────┘
       │                  │                   │
   ┌───┴──────────────────┴───────────────────┴───┐
   │         25+ Integrated Projects              │
   ├──────────────────────────────────────────────┤
   │ Extraction: DeepKE, OneKE, KG-Gen, AIKG     │
   │ Reasoning: NeuralKG, Causal-Learn, KarateClub│
   │ Temporal: Graphiti                           │
   │ Safety: Guardrails                           │
   │ Conversation: DTS                            │
   │ Refinement: ICR                              │
   │ Physics: Neuromancer                         │
   │ Hybrid: Cognitive-Hydraulics                 │
   │ Structured: Outlines, LMQL                   │
   │ Visualization: PyGraphistry                  │
   │ Chemical: GlobalChem                         │
   │ And more...                                  │
   └──────────────────────────────────────────────┘

Usage:
    >>> orchestrator = GlobalKGOrchestrator()
    >>> await orchestrator.initialize()
    >>> 
    >>> # Comprehensive KG extraction with all safety checks
    >>> result = await orchestrator.extract_and_validate(
    ...     text="Apple Inc. was founded by Steve Jobs...",
    ...     extractors=['deepke', 'oneke'],
    ...     enable_guardrails=True
    ... )
    >>> 
    >>> # Multi-turn conversation with optimization
    >>> conversation = await orchestrator.optimize_dialog(
    ...     context="Customer inquiry about products",
    ...     goal="Provide helpful response"
    ... )
    >>> 
    >>> # Hybrid reasoning with physics validation
    >>> result = await orchestrator.reason_with_physics(
    ...     problem=kg_data,
    ...     validate_physics=True
    ... )
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto

from knowledge_engine.unified_kg_integration_hub import (
    UnifiedKGIntegrationHub,
    KGOperationType
)

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Stages in the KG processing pipeline."""
    EXTRACTION = auto()
    VALIDATION = auto()
    REFINEMENT = auto()
    ENRICHMENT = auto()
    SAFETY_CHECK = auto()
    CONVERSATION = auto()
    REASONING = auto()
    SIMULATION = auto()
    ANALYSIS = auto()


@dataclass
class GlobalKGConfig:
    """Configuration for Global KG Orchestrator."""
    
    # Extraction settings
    primary_extractor: str = 'deepke'
    fallback_extractors: List[str] = field(default_factory=lambda: ['oneke', 'kggen'])
    extraction_confidence_threshold: float = 0.8
    
    # Safety settings
    enable_guardrails: bool = True
    safety_level: str = 'MODERATE'  # STRICT, MODERATE, PERMISSIVE
    auto_redact_pii: bool = True
    
    # Refinement settings
    enable_icr: bool = True
    icr_max_iterations: int = 3
    icr_quality_threshold: float = 0.85
    
    # Topological analysis settings (Lagrange Mapper)
    enable_lagrange_mapper: bool = True
    lagrange_n_clusters: int = 8
    lagrange_reduction_method: str = 'pca'
    lagrange_drift_threshold: float = 0.3
    
    # Conversation settings
    enable_dts: bool = True
    dts_beam_width: int = 5
    dts_max_rounds: int = 3
    
    # Reasoning settings
    enable_cognitive_hydraulics: bool = True
    default_reasoning_mode: str = 'auto'  # soar, actr, evolutionary, auto
    
    # Physics settings
    enable_neuromancer: bool = True
    physics_validation_domain: str = 'mechanics'
    
    # Structured output settings
    enable_outlines: bool = True
    default_output_format: str = 'json'
    
    # Query settings
    enable_lmql: bool = True
    
    # Parallel processing
    max_parallel_operations: int = 5
    enable_async_processing: bool = True


@dataclass
class ProcessingResult:
    """Result of a KG processing operation."""
    success: bool
    stage: ProcessingStage
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Integration usage tracking
    integrations_used: List[str] = field(default_factory=list)
    fallback_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'stage': self.stage.name,
            'data': self.data,
            'metadata': self.metadata,
            'errors': self.errors,
            'processing_time_ms': self.processing_time_ms,
            'timestamp': self.timestamp,
            'integrations_used': self.integrations_used,
            'fallback_used': self.fallback_used
        }


class GlobalKGOrchestrator:
    """
    Global Knowledge Graph Orchestrator.
    
    Combines ALL knowledge graph integrations into a unified system that
    leverages the unique capabilities of each project for comprehensive
    KG processing.
    
    Key Features:
    - Multi-extractor fusion (DeepKE + OneKE + KG-Gen + AIKG)
    - Automatic safety validation (Guardrails)
    - Iterative quality refinement (ICR)
    - Conversation optimization (DTS)
    - Hybrid reasoning (Cognitive-Hydraulics)
    - Physics-informed validation (Neuromancer)
    - Structured output guarantees (Outlines)
    - Declarative querying (LMQL)
    
    Example:
        >>> orchestrator = GlobalKGOrchestrator()
        >>> await orchestrator.initialize()
        >>> 
        >>> # Comprehensive extraction with all checks
        >>> result = await orchestrator.extract_comprehensive(
        ...     text="...",
        ...     enable_guardrails=True,
        ...     enable_icr=True
        ... )
    """
    
    def __init__(self, config: Optional[GlobalKGConfig] = None):
        """
        Initialize the Global KG Orchestrator.
        
        Args:
            config: Configuration object. Uses defaults if not provided.
        """
        self.config = config or GlobalKGConfig()
        self.hub = UnifiedKGIntegrationHub()
        self._initialized = False
        
        logger.info({
            'msg': 'GlobalKGOrchestrator initialized',
            'config': {
                'primary_extractor': self.config.primary_extractor,
                'enable_guardrails': self.config.enable_guardrails,
                'enable_icr': self.config.enable_icr,
                'enable_dts': self.config.enable_dts,
                'enable_cognitive_hydraulics': self.config.enable_cognitive_hydraulics
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    async def initialize(self) -> bool:
        """
        Initialize all integrations.
        
        Returns:
            True if at least one integration initialized successfully.
        """
        if self._initialized:
            return True
        
        logger.info({
            'msg': 'Initializing GlobalKGOrchestrator',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        success = await self.hub.initialize()
        self._initialized = True
        
        # Log available integrations
        health = self.hub.get_health_status()
        available = health['summary']['available']
        total = health['summary']['total']
        
        logger.info({
            'msg': 'GlobalKGOrchestrator initialization complete',
            'available_integrations': available,
            'total_integrations': total,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        return success
    
    # ============================================================================
    # COMPREHENSIVE EXTRACTION PIPELINE
    # ============================================================================
    
    async def extract_comprehensive(
        self,
        text: str,
        extractors: Optional[List[str]] = None,
        enable_guardrails: Optional[bool] = None,
        enable_icr: Optional[bool] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Extract knowledge from text using multiple extractors with validation.
        
        This method:
        1. Runs multiple extractors in parallel
        2. Merges results
        3. Validates with Guardrails (if enabled)
        4. Refines with ICR (if enabled)
        5. Ensures structured output with Outlines (if schema provided)
        
        Args:
            text: Input text to extract from
            extractors: List of extractor names. Uses config defaults if None.
            enable_guardrails: Override config for safety checks
            enable_icr: Override config for iterative refinement
            output_schema: JSON schema for structured output
            
        Returns:
            ProcessingResult with extracted entities and relations
        """
        start_time = datetime.now(timezone.utc)
        integrations_used = []
        
        # Step 1: Multi-extractor fusion
        extractors = extractors or [self.config.primary_extractor] + self.config.fallback_extractors
        extraction_results = []
        
        for extractor in extractors:
            try:
                result = await self.hub.extract_entities(text, method=extractor)
                if result.success:
                    extraction_results.append(result.data)
                    integrations_used.append(extractor)
            except Exception as e:
                logger.warning(f"Extractor {extractor} failed: {e}")
        
        if not extraction_results:
            return ProcessingResult(
                success=False,
                stage=ProcessingStage.EXTRACTION,
                errors=['All extractors failed'],
                integrations_used=integrations_used,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
        
        # Merge results (simple merge for now)
        merged_data = self._merge_extraction_results(extraction_results)
        
        # Step 2: Structured output with Outlines (if schema provided)
        if output_schema and self.config.enable_outlines:
            try:
                structured_result = await self.hub.structured_generate(
                    prompt=f"Format this extraction: {merged_data}",
                    output_schema=output_schema,
                    method='json'
                )
                if structured_result.success:
                    merged_data = structured_result.data
                    integrations_used.append('outlines')
            except Exception as e:
                logger.warning(f"Structured generation failed: {e}")
        
        # Step 3: Safety validation with Guardrails
        if (enable_guardrails if enable_guardrails is not None else self.config.enable_guardrails):
            try:
                safety_result = await self.hub.validate_safety(
                    content=str(merged_data),
                    content_type='output',
                    safety_level=self.config.safety_level
                )
                if not safety_result.success:
                    return ProcessingResult(
                        success=False,
                        stage=ProcessingStage.SAFETY_CHECK,
                        data=merged_data,
                        errors=['Safety validation failed'] + safety_result.errors,
                        integrations_used=integrations_used + ['guardrails'],
                        processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    )
                integrations_used.append('guardrails')
            except Exception as e:
                logger.warning(f"Guardrails validation failed: {e}")
        
        # Step 4: Iterative refinement with ICR
        if (enable_icr if enable_icr is not None else self.config.enable_icr):
            try:
                refined_result = await self.hub.refine_iteratively(
                    initial_output=str(merged_data),
                    goal="Improve extraction accuracy and completeness",
                    max_iterations=self.config.icr_max_iterations,
                    quality_threshold=self.config.icr_quality_threshold
                )
                if refined_result.success:
                    merged_data = refined_result.data
                    integrations_used.append('icr')
            except Exception as e:
                logger.warning(f"ICR refinement failed: {e}")
        
        return ProcessingResult(
            success=True,
            stage=ProcessingStage.EXTRACTION,
            data=merged_data,
            integrations_used=integrations_used,
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    # ============================================================================
    # CONVERSATION OPTIMIZATION
    # ============================================================================
    
    async def optimize_dialog(
        self,
        context: str,
        goal: str,
        enable_dts: Optional[bool] = None,
        enable_guardrails: Optional[bool] = None
    ) -> ProcessingResult:
        """
        Optimize multi-turn conversation using DTS.
        
        Uses Dialogue Tree Search to explore conversation strategies,
        simulate user reactions, and find optimal conversation paths.
        
        Args:
            context: Conversation context
            goal: Conversation goal
            enable_dts: Override config for DTS
            enable_guardrails: Override config for safety checks
            
        Returns:
            ProcessingResult with optimized conversation tree
        """
        start_time = datetime.now(timezone.utc)
        integrations_used = []
        
        # Step 1: Safety check on context
        if (enable_guardrails if enable_guardrails is not None else self.config.enable_guardrails):
            try:
                safety_result = await self.hub.validate_safety(
                    content=context,
                    content_type='input',
                    safety_level=self.config.safety_level
                )
                if not safety_result.success:
                    return ProcessingResult(
                        success=False,
                        stage=ProcessingStage.SAFETY_CHECK,
                        errors=['Input safety check failed'] + safety_result.errors,
                        integrations_used=['guardrails'],
                        processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    )
                integrations_used.append('guardrails')
            except Exception as e:
                logger.warning(f"Guardrails input check failed: {e}")
        
        # Step 2: Optimize conversation with DTS
        if (enable_dts if enable_dts is not None else self.config.enable_dts):
            try:
                dts_result = await self.hub.optimize_conversation(
                    initial_context=context,
                    goal=goal,
                    rounds=self.config.dts_max_rounds,
                    beam_width=self.config.dts_beam_width
                )
                if dts_result.success:
                    integrations_used.append('dts')
                    return ProcessingResult(
                        success=True,
                        stage=ProcessingStage.CONVERSATION,
                        data=dts_result.data,
                        integrations_used=integrations_used,
                        processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    )
            except Exception as e:
                logger.warning(f"DTS optimization failed: {e}")
        
        return ProcessingResult(
            success=False,
            stage=ProcessingStage.CONVERSATION,
            errors=['Conversation optimization failed'],
            integrations_used=integrations_used,
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    # ============================================================================
    # HYBRID REASONING
    # ============================================================================
    
    async def reason_with_physics(
        self,
        problem: Dict[str, Any],
        validate_physics: bool = True,
        reasoning_mode: Optional[str] = None
    ) -> ProcessingResult:
        """
        Perform hybrid reasoning with optional physics validation.
        
        Combines Cognitive-Hydraulics for hybrid reasoning with
        Neuromancer for physics-informed validation.
        
        Args:
            problem: Problem description with KG context
            validate_physics: Enable physics validation
            reasoning_mode: soar, actr, evolutionary, or auto
            
        Returns:
            ProcessingResult with reasoning output
        """
        start_time = datetime.now(timezone.utc)
        integrations_used = []
        
        # Step 1: Hybrid reasoning with Cognitive-Hydraulics
        if self.config.enable_cognitive_hydraulics:
            try:
                reasoning_result = await self.hub.hybrid_reasoning(
                    problem=problem,
                    goal=problem.get('goal', 'Solve the problem'),
                    reasoning_mode=reasoning_mode or self.config.default_reasoning_mode
                )
                if reasoning_result.success:
                    integrations_used.append('cognitive_hydraulics')
                    result_data = reasoning_result.data
                else:
                    return ProcessingResult(
                        success=False,
                        stage=ProcessingStage.REASONING,
                        errors=['Hybrid reasoning failed'] + reasoning_result.errors,
                        integrations_used=integrations_used,
                        processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    )
            except Exception as e:
                logger.warning(f"Cognitive-Hydraulics failed: {e}")
                return ProcessingResult(
                    success=False,
                    stage=ProcessingStage.REASONING,
                    errors=[str(e)],
                    integrations_used=integrations_used,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
        
        # Step 2: Physics validation with Neuromancer
        if validate_physics and self.config.enable_neuromancer:
            try:
                physics_result = await self.hub.physics_simulate(
                    system_description=problem,
                    simulation_type='validation',
                    time_horizon=10.0
                )
                if physics_result.success:
                    integrations_used.append('neuromancer')
                    # Merge physics validation into result
                    result_data = {
                        'reasoning_result': result_data,
                        'physics_validation': physics_result.data
                    }
            except Exception as e:
                logger.warning(f"Physics validation failed: {e}")
        
        return ProcessingResult(
            success=True,
            stage=ProcessingStage.REASONING,
            data=result_data,
            integrations_used=integrations_used,
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
    
    # ============================================================================
    # DECLARATIVE QUERY
    # ============================================================================
    
    async def query_kg_declarative(
        self,
        query: str,
        query_type: str = 'entities',
        context: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Query knowledge graph using declarative LMQL queries.
        
        Args:
            query: LMQL-style query string
            query_type: Type of query (entities, relations, cypher, multi_hop)
            context: Query context
            
        Returns:
            ProcessingResult with query results
        """
        start_time = datetime.now(timezone.utc)
        
        if not self.config.enable_lmql:
            return ProcessingResult(
                success=False,
                stage=ProcessingStage.REASONING,
                errors=['LMQL is disabled in configuration'],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
        
        try:
            result = await self.hub.declarative_query(
                query=query,
                context=context or {},
                query_type=query_type
            )
            
            return ProcessingResult(
                success=result.success,
                stage=ProcessingStage.REASONING,
                data=result.data,
                integrations_used=['lmql'],
                errors=result.errors,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                stage=ProcessingStage.REASONING,
                errors=[str(e)],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
    
    # ============================================================================
    # TOPOLOGICAL ANALYSIS (Lagrange Mapper)
    # ============================================================================
    
    async def analyze_knowledge_topology(
        self,
        embeddings: Any,
        labels: Optional[List[str]] = None,
        n_clusters: int = 8,
        analysis_type: str = 'landscape',
        track_transitions: bool = False,
        previous_embeddings: Optional[Any] = None
    ) -> ProcessingResult:
        """
        Analyze knowledge topology using Lagrange Mapper.
        
        Maps attractor landscapes in knowledge embedding spaces to understand
        concept clustering, stability, and evolution over time.
        
        Args:
            embeddings: Knowledge embeddings (n_samples x n_features)
            labels: Optional labels for each knowledge item
            n_clusters: Number of clusters to identify (default: 8)
            analysis_type: Type of analysis:
                - 'landscape': Full landscape analysis with attractors
                - 'topology': Graph topology analysis
                - 'basins': Basin of attraction computation
                - 'transitions': Compare with previous_embeddings
            track_transitions: Enable transition tracking if previous_embeddings provided
            previous_embeddings: Optional previous state for transition detection
            
        Returns:
            ProcessingResult with topological analysis:
                - attractors: List of attractors with strength metrics
                - clusters: Cluster assignments and statistics
                - landscape: Full landscape topology
                - transitions: Changes from previous state (if tracked)
                
        Example:
            >>> # Analyze concept landscape from knowledge graph embeddings
            >>> kg_embeddings = await orchestrator.hub.embed_graph(my_kg)
            >>> result = await orchestrator.analyze_knowledge_topology(
            ...     embeddings=kg_embeddings['embeddings'],
            ...     labels=kg_embeddings['node_labels'],
            ...     n_clusters=5,
            ...     analysis_type='landscape'
            ... )
            >>> 
            >>> # Track evolution over time
            >>> result_t2 = await orchestrator.analyze_knowledge_topology(
            ...     embeddings=new_embeddings,
            ...     previous_embeddings=old_embeddings,
            ...     track_transitions=True
            ... )
        """
        start_time = datetime.now(timezone.utc)
        integrations_used = []
        
        try:
            # Step 1: Topological landscape analysis
            landscape_result = await self.hub.analyze_topological_landscape(
                embeddings=embeddings,
                labels=labels,
                n_clusters=n_clusters,
                analysis_type=analysis_type
            )
            
            if landscape_result.success:
                integrations_used.append('lagrange_mapper')
                result_data = landscape_result.data
            else:
                return ProcessingResult(
                    success=False,
                    stage=ProcessingStage.ANALYSIS,
                    errors=['Topological analysis failed'] + landscape_result.errors,
                    integrations_used=integrations_used,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            
            # Step 2: Transition detection if requested
            if track_transitions and previous_embeddings is not None:
                try:
                    transition_result = await self.hub.detect_landscape_transitions(
                        embeddings_t1=previous_embeddings,
                        embeddings_t2=embeddings,
                        labels=labels
                    )
                    if transition_result.success:
                        result_data['transitions'] = transition_result.data
                        integrations_used.append('lagrange_mapper_transitions')
                except Exception as e:
                    logger.warning(f"Transition detection failed: {e}")
            
            return ProcessingResult(
                success=True,
                stage=ProcessingStage.ANALYSIS,
                data=result_data,
                integrations_used=integrations_used,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error(f"Knowledge topology analysis failed: {e}")
            return ProcessingResult(
                success=False,
                stage=ProcessingStage.ANALYSIS,
                errors=[str(e)],
                integrations_used=integrations_used,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
    
    async def detect_concept_drift(
        self,
        embeddings_t1: Any,
        embeddings_t2: Any,
        labels: Optional[List[str]] = None,
        drift_threshold: float = 0.3
    ) -> ProcessingResult:
        """
        Detect concept drift between two knowledge states.
        
        Uses Lagrange Mapper to identify significant changes in the knowledge
        landscape, indicating concept emergence, disappearance, or evolution.
        
        Args:
            embeddings_t1: Knowledge embeddings at time t1
            embeddings_t2: Knowledge embeddings at time t2
            labels: Optional labels for knowledge items
            drift_threshold: Threshold for significant drift detection (0-1)
            
        Returns:
            ProcessingResult with drift analysis:
                - drift_detected: Boolean indicating significant drift
                - drift_score: Overall drift metric (0-1)
                - created_concepts: New attractors in t2
                - disappeared_concepts: Attractors lost from t1
                - evolved_concepts: Attractors with significant changes
                - stability_score: Overall landscape stability (0-1)
                
        Example:
            >>> # Compare knowledge states from different time periods
            >>> result = await orchestrator.detect_concept_drift(
            ...     embeddings_t1=kg_january['embeddings'],
            ...     embeddings_t2=kg_june['embeddings'],
            ...     drift_threshold=0.25
            ... )
            >>> if result.data['drift_detected']:
            ...     print(f"Significant drift detected: {result.data['drift_score']:.2f}")
            ...     print(f"New concepts: {len(result.data['created_concepts'])}")
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Detect landscape transitions
            transition_result = await self.hub.detect_landscape_transitions(
                embeddings_t1=embeddings_t1,
                embeddings_t2=embeddings_t2,
                labels=labels
            )
            
            if not transition_result.success:
                return ProcessingResult(
                    success=False,
                    stage=ProcessingStage.ANALYSIS,
                    errors=['Transition detection failed'] + transition_result.errors,
                    processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
            
            transitions = transition_result.data.get('transitions', {})
            
            # Calculate drift metrics
            attractors_t1 = transition_result.data.get('n_attractors_t1', 1)
            attractors_t2 = transition_result.data.get('n_attractors_t2', 1)
            stability = transition_result.data.get('stability', 0.0)
            
            created = len(transitions.get('attractors_created', []))
            destroyed = len(transitions.get('attractors_destroyed', []))
            persisted = len(transitions.get('attractors_persisted', []))
            
            # Calculate drift score
            total_changes = created + destroyed
            max_attractors = max(attractors_t1, attractors_t2, 1)
            drift_score = total_changes / (2 * max_attractors)  # Normalize
            
            # Detect significant drift
            drift_detected = drift_score > drift_threshold or stability < (1 - drift_threshold)
            
            result_data = {
                'drift_detected': drift_detected,
                'drift_score': float(drift_score),
                'stability_score': float(stability),
                'created_concepts': transitions.get('attractors_created', []),
                'disappeared_concepts': transitions.get('attractors_destroyed', []),
                'evolved_concepts': transitions.get('attractors_persisted', []),
                'attractor_counts': {
                    't1': attractors_t1,
                    't2': attractors_t2,
                    'created': created,
                    'destroyed': destroyed,
                    'persisted': persisted
                }
            }
            
            return ProcessingResult(
                success=True,
                stage=ProcessingStage.ANALYSIS,
                data=result_data,
                integrations_used=['lagrange_mapper'],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error(f"Concept drift detection failed: {e}")
            return ProcessingResult(
                success=False,
                stage=ProcessingStage.ANALYSIS,
                errors=[str(e)],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _merge_extraction_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge extraction results from multiple extractors."""
        merged = {
            'entities': [],
            'relations': [],
            'sources': []
        }
        
        for i, result in enumerate(results):
            if isinstance(result, dict):
                if 'entities' in result:
                    merged['entities'].extend(result['entities'])
                if 'relations' in result:
                    merged['relations'].extend(result['relations'])
                merged['sources'].append(f'extractor_{i}')
        
        # Deduplicate entities by name
        seen = set()
        unique_entities = []
        for entity in merged['entities']:
            name = entity.get('name', '') if isinstance(entity, dict) else str(entity)
            if name and name not in seen:
                seen.add(name)
                unique_entities.append(entity)
        merged['entities'] = unique_entities
        
        return merged
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all integrations."""
        return self.hub.get_health_status()
    
    def get_available_integrations(self) -> List[str]:
        """Get list of available integration names."""
        return self.hub.get_available_integrations()


# Convenience function for quick usage
async def create_global_orchestrator(config: Optional[GlobalKGConfig] = None) -> GlobalKGOrchestrator:
    """
    Create and initialize a GlobalKGOrchestrator.
    
    Args:
        config: Optional configuration
        
    Returns:
        Initialized GlobalKGOrchestrator
    """
    orchestrator = GlobalKGOrchestrator(config)
    await orchestrator.initialize()
    return orchestrator
