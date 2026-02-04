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
