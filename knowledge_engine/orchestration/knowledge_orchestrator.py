"""
Knowledge Engine Orchestrator

A comprehensive, configurable orchestration system that ties all Knowledge Engine
integrations into a cohesive, interconnected system flow.

Features:
- Extremely configurable pipeline architecture
- Component skip/disable capabilities
- Domain-specific presets (finance, chemistry, healthcare, etc.)
- Dynamic pipeline construction
- Conditional execution based on data type and content
- Performance optimization through component selection
- Error handling and fallback mechanisms

Following CLAUDE.md principles:
- ZERO TRUST: Validate all configurations
- RUNTIME TRUTH: Check component availability at runtime
- IDEMPOTENCY: Safe to retry
- CONFIGURATION EXPLICITNESS: All config via parameters
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
import copy
import inspect

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Domain types for specialized processing"""
    GENERAL = "general"
    FINANCE = "finance"
    CHEMISTRY = "chemistry"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    ENGINEERING = "engineering"
    RESEARCH = "research"
    SOCIAL_MEDIA = "social_media"


class ComponentType(Enum):
    """Component types for the knowledge engine"""
    # Extraction Components
    DEEPKE = "deepke"
    ONEKE = "oneke"
    KG_GEN = "kg_gen"
    
    # Graph Analysis Components
    KARATE_CLUB = "karate_club"
    NEURALKG = "neuralkg"
    GRAPHITI = "graphiti"
    
    # Pattern Mining Components
    PAMI = "pami"
    
    # Causal Analysis Components
    CAUSAL_LEARN = "causal_learn"
    
    # Topological Analysis Components
    LAGRANGE_MAPPER = "lagrange_mapper"
    
    # Domain-Specific Components
    GLOBAL_CHEM = "global_chem"
    NEUROMANCER = "neuromancer"
    
    # Knowledge Management Components
    KNOWLEDGE_EXTRACTOR = "knowledge_extractor"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    KNOWLEDGE_STORAGE = "knowledge_storage"


@dataclass
class ComponentConfig:
    """Configuration for a single component"""
    enabled: bool = True
    required: bool = False
    skip_conditions: List[str] = field(default_factory=list)
    config_override: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    retry_count: int = 3
    fallback_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentConfig':
        return cls(**data)


@dataclass
class PipelineStage:
    """A single stage in the processing pipeline"""
    name: str
    component: ComponentType
    enabled: bool = True
    condition: Optional[str] = None  # Python expression for conditional execution
    depends_on: List[str] = field(default_factory=list)
    config: ComponentConfig = field(default_factory=ComponentConfig)
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        """Determine if this stage should execute based on condition"""
        if not self.enabled:
            return False
        
        if self.condition:
            try:
                # Use safe expression evaluator
                from .safe_eval import safe_eval
                # Wrap context so expressions can use 'context.get(...)' syntax
                eval_namespace = {'context': context}
                return bool(safe_eval(self.condition, eval_namespace))
            except Exception as e:
                logger.warning(f"Condition evaluation failed for {self.name}: {e}")
                return True  # Execute if condition can't be evaluated
        
        return True


@dataclass
class OrchestratorConfig:
    """Main orchestrator configuration"""
    name: str = "default_orchestrator"
    domain: DomainType = DomainType.GENERAL
    description: str = ""
    
    # Global settings
    max_workers: int = 4
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    log_level: str = "INFO"
    correlation_id: Optional[str] = None
    
    # Component configurations
    components: Dict[ComponentType, ComponentConfig] = field(default_factory=dict)
    
    # Pipeline definition
    pipeline_stages: List[PipelineStage] = field(default_factory=list)
    
    # Skip rules
    auto_skip_unused_components: bool = True
    skip_on_error: bool = False
    
    # Performance settings
    parallel_execution: bool = False
    batch_size: int = 100
    
    def __post_init__(self):
        if not self.components:
            self.components = self._default_components()
        if not self.pipeline_stages:
            self.pipeline_stages = self._default_pipeline()
    
    def _default_components(self) -> Dict[ComponentType, ComponentConfig]:
        """Default component configurations"""
        return {
            ComponentType.DEEPKE: ComponentConfig(enabled=True, required=False),
            ComponentType.KARATE_CLUB: ComponentConfig(enabled=True, required=False),
            ComponentType.PAMI: ComponentConfig(enabled=True, required=False),
            ComponentType.NEURALKG: ComponentConfig(enabled=True, required=False),
            ComponentType.CAUSAL_LEARN: ComponentConfig(enabled=True, required=False),  # Enabled for causal discovery
            ComponentType.LAGRANGE_MAPPER: ComponentConfig(enabled=True, required=False),
            ComponentType.GLOBAL_CHEM: ComponentConfig(enabled=True, required=False),
            ComponentType.NEUROMANCER: ComponentConfig(enabled=False, required=False),  # Disabled by default
        }
    
    def _default_pipeline(self) -> List[PipelineStage]:
        """Default pipeline stages"""
        return [
            PipelineStage(
                name="extract_knowledge",
                component=ComponentType.DEEPKE,
                enabled=True
            ),
            PipelineStage(
                name="build_graph",
                component=ComponentType.KG_GEN,
                enabled=True,
                depends_on=["extract_knowledge"]
            ),
            PipelineStage(
                name="analyze_communities",
                component=ComponentType.KARATE_CLUB,
                enabled=True,
                depends_on=["build_graph"]
            ),
            PipelineStage(
                name="mine_patterns",
                component=ComponentType.PAMI,
                enabled=True,
                depends_on=["extract_knowledge"]
            ),
            PipelineStage(
                name="generate_embeddings",
                component=ComponentType.NEURALKG,
                enabled=True,
                depends_on=["build_graph"]
            ),
            PipelineStage(
                name="analyze_topology",
                component=ComponentType.LAGRANGE_MAPPER,
                enabled=True,
                depends_on=["generate_embeddings"],
                condition="len(get(context, 'embeddings', [])) > 10"
            ),
            PipelineStage(
                name="discover_causal_structure",
                component=ComponentType.CAUSAL_LEARN,
                enabled=True,
                depends_on=["build_graph"],
                condition="len(get(context, 'graph_nodes', [])) > 2"
            ),
        ]
    
    def disable_component(self, component: ComponentType) -> 'OrchestratorConfig':
        """Disable a specific component"""
        if component in self.components:
            self.components[component].enabled = False
        return self
    
    def enable_component(self, component: ComponentType, required: bool = False) -> 'OrchestratorConfig':
        """Enable a specific component"""
        if component in self.components:
            self.components[component].enabled = True
            self.components[component].required = required
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'domain': self.domain.value,
            'description': self.description,
            'max_workers': self.max_workers,
            'enable_caching': self.enable_caching,
            'log_level': self.log_level,
            'components': {k.value: v.to_dict() for k, v in self.components.items()},
            'pipeline_stages': [
                {
                    'name': s.name,
                    'component': s.component.value,
                    'enabled': s.enabled,
                    'condition': s.condition,
                    'depends_on': s.depends_on,
                    'config': s.config.to_dict()
                }
                for s in self.pipeline_stages
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrchestratorConfig':
        """Create config from dictionary"""
        components = {
            ComponentType(k): ComponentConfig.from_dict(v)
            for k, v in data.get('components', {}).items()
        }
        
        pipeline_stages = [
            PipelineStage(
                name=s['name'],
                component=ComponentType(s['component']),
                enabled=s.get('enabled', True),
                condition=s.get('condition'),
                depends_on=s.get('depends_on', []),
                config=ComponentConfig.from_dict(s.get('config', {}))
            )
            for s in data.get('pipeline_stages', [])
        ]
        
        return cls(
            name=data.get('name', 'default'),
            domain=DomainType(data.get('domain', 'general')),
            description=data.get('description', ''),
            max_workers=data.get('max_workers', 4),
            enable_caching=data.get('enable_caching', True),
            log_level=data.get('log_level', 'INFO'),
            components=components,
            pipeline_stages=pipeline_stages
        )


class DomainPresets:
    """Predefined configurations for different domains"""
    
    @staticmethod
    def finance() -> OrchestratorConfig:
        """Finance domain preset - disables chemistry components"""
        config = OrchestratorConfig(
            name="finance_orchestrator",
            domain=DomainType.FINANCE,
            description="Optimized for financial data analysis"
        )
        
        # Disable chemistry-related components
        config.disable_component(ComponentType.GLOBAL_CHEM)
        config.disable_component(ComponentType.NEUROMANCER)  # Physics modeling not needed
        
        # Enable causal analysis for market causality
        config.enable_component(ComponentType.CAUSAL_LEARN, required=False)
        
        # Finance-specific pipeline
        config.pipeline_stages = [
            PipelineStage(
                name="extract_entities",
                component=ComponentType.DEEPKE,
                enabled=True
            ),
            PipelineStage(
                name="build_knowledge_graph",
                component=ComponentType.KG_GEN,
                enabled=True,
                depends_on=["extract_entities"]
            ),
            PipelineStage(
                name="detect_communities",
                component=ComponentType.KARATE_CLUB,
                enabled=True,
                depends_on=["build_knowledge_graph"]
            ),
            PipelineStage(
                name="analyze_causality",
                component=ComponentType.CAUSAL_LEARN,
                enabled=True,
                depends_on=["build_knowledge_graph"],
                condition="get(context, 'data_type') == 'time_series'"
            ),
            PipelineStage(
                name="generate_embeddings",
                component=ComponentType.NEURALKG,
                enabled=True,
                depends_on=["build_knowledge_graph"]
            ),
        ]
        
        return config
    
    @staticmethod
    def chemistry() -> OrchestratorConfig:
        """Chemistry domain preset - enables chemical analysis"""
        config = OrchestratorConfig(
            name="chemistry_orchestrator",
            domain=DomainType.CHEMISTRY,
            description="Optimized for chemical and molecular data"
        )
        
        # Enable chemistry components
        config.enable_component(ComponentType.GLOBAL_CHEM, required=True)
        config.enable_component(ComponentType.NEUROMANCER, required=False)  # For molecular dynamics
        
        # Chemistry-specific pipeline
        config.pipeline_stages = [
            PipelineStage(
                name="extract_chemical_entities",
                component=ComponentType.GLOBAL_CHEM,
                enabled=True
            ),
            PipelineStage(
                name="extract_general_entities",
                component=ComponentType.DEEPKE,
                enabled=True
            ),
            PipelineStage(
                name="build_knowledge_graph",
                component=ComponentType.KG_GEN,
                enabled=True,
                depends_on=["extract_chemical_entities", "extract_general_entities"]
            ),
            PipelineStage(
                name="analyze_structure",
                component=ComponentType.KARATE_CLUB,
                enabled=True,
                depends_on=["build_knowledge_graph"]
            ),
            PipelineStage(
                name="model_dynamics",
                component=ComponentType.NEUROMANCER,
                enabled=True,
                depends_on=["build_knowledge_graph"],
                condition="get(context, 'has_molecular_dynamics_data', False)"
            ),
            PipelineStage(
                name="generate_embeddings",
                component=ComponentType.NEURALKG,
                enabled=True,
                depends_on=["build_knowledge_graph"]
            ),
        ]
        
        return config
    
    @staticmethod
    def healthcare() -> OrchestratorConfig:
        """Healthcare domain preset"""
        config = OrchestratorConfig(
            name="healthcare_orchestrator",
            domain=DomainType.HEALTHCARE,
            description="Optimized for healthcare and medical data"
        )
        
        # Enable chemistry for drug analysis
        config.enable_component(ComponentType.GLOBAL_CHEM, required=False)
        config.enable_component(ComponentType.CAUSAL_LEARN, required=False)
        
        return config
    
    @staticmethod
    def research() -> OrchestratorConfig:
        """Research domain preset - comprehensive analysis"""
        config = OrchestratorConfig(
            name="research_orchestrator",
            domain=DomainType.RESEARCH,
            description="Comprehensive analysis for research applications"
        )
        
        # Enable all components
        for component in ComponentType:
            config.enable_component(component, required=False)
        
        return config
    
    @staticmethod
    def minimal() -> OrchestratorConfig:
        """Minimal preset - only essential components"""
        config = OrchestratorConfig(
            name="minimal_orchestrator",
            domain=DomainType.GENERAL,
            description="Minimal configuration for basic usage"
        )
        
        # Disable most components
        config.disable_component(ComponentType.CAUSAL_LEARN)
        config.disable_component(ComponentType.LAGRANGE_MAPPER)
        config.disable_component(ComponentType.GLOBAL_CHEM)
        config.disable_component(ComponentType.NEUROMANCER)
        config.disable_component(ComponentType.NEURALKG)
        config.disable_component(ComponentType.PAMI)
        
        # Minimal pipeline
        config.pipeline_stages = [
            PipelineStage(
                name="extract",
                component=ComponentType.DEEPKE,
                enabled=True
            ),
            PipelineStage(
                name="build_graph",
                component=ComponentType.KG_GEN,
                enabled=True,
                depends_on=["extract"]
            ),
        ]
        
        return config


class KnowledgeOrchestrator:
    """
    Main orchestrator for the Knowledge Engine.
    
    Manages the execution of all integrated components in a cohesive pipeline,
    with support for:
    - Dynamic pipeline construction
    - Component skip/disable
    - Domain-specific presets
    - Conditional execution
    - Error handling and fallbacks
    """
    
    def __init__(self, config: Optional[Union[OrchestratorConfig, Dict[str, Any]]] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Orchestrator configuration (OrchestratorConfig object or dict)
        """
        if config is None:
            self.config = OrchestratorConfig()
        elif isinstance(config, dict):
            # Convert dict to OrchestratorConfig
            self.config = OrchestratorConfig(**config)
        else:
            self.config = config
            
        self.components = {}
        self.cache = {}
        self.execution_history = []
        
        # Initialize components
        self._initialize_components()
        
        logger.info({
            "msg": "KnowledgeOrchestrator initialized",
            "name": self.config.name,
            "domain": self.config.domain.value,
            "enabled_components": list(self.components.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _initialize_components(self):
        """Initialize all enabled components"""
        # Import all integration modules
        try:
            from ..integrations import (
                AIKnowledgeGraphIntegrator,
                PAMIPatternMiner,
                NeuralKGEmbedder,
                CausalDiscoveryEngine,
                LagrangeAttractorAnalyzer,
                GlobalChemKnowledgeAdapter,
                NeuromancerDynamicsModeler
            )
            
            # Initialize main integrator
            self.integrator = AIKnowledgeGraphIntegrator()
            
            # Check component availability
            component_map = {
                ComponentType.DEEPKE: self.integrator.deepke_extractor,
                ComponentType.KARATE_CLUB: self.integrator.karateclub_analyzer,
                ComponentType.KG_GEN: self.integrator.kg_gen_manager,
                ComponentType.PAMI: self.integrator.pami_miner,
                ComponentType.NEURALKG: self.integrator.neuralkg_embedder,
                ComponentType.CAUSAL_LEARN: self.integrator.causal_engine,
                ComponentType.LAGRANGE_MAPPER: self.integrator.lagrange_analyzer,
                ComponentType.GLOBAL_CHEM: self.integrator.global_chem_adapter,
                ComponentType.NEUROMANCER: self.integrator.neuromancer_modeler,
            }
            
            for comp_type, instance in component_map.items():
                if comp_type in self.config.components:
                    comp_config = self.config.components[comp_type]
                    if comp_config.enabled and instance is not None:
                        self.components[comp_type] = instance
                        logger.debug(f"Initialized component: {comp_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to import integrations: {e}")
            self.integrator = None
    
    def process(
        self,
        input_data: Dict[str, Any],
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input data through the orchestrated pipeline.
        
        Args:
            input_data: Input data to process
            custom_config: Optional runtime configuration overrides
            
        Returns:
            Processing results from all enabled stages
        """
        start_time = datetime.now(timezone.utc)
        correlation_id = self.config.correlation_id or f"orch_{start_time.timestamp()}"
        
        logger.info({
            "msg": "Starting orchestrated processing",
            "correlation_id": correlation_id,
            "input_keys": list(input_data.keys()),
            "timestamp": start_time.isoformat()
        })
        
        # Build execution context
        context = {
            'input': input_data,
            'results': {},
            'correlation_id': correlation_id,
            'start_time': start_time,
            'data_type': input_data.get('data_type', 'unknown'),
            'domain': self.config.domain.value,
        }
        
        # Apply custom config if provided
        config = self.config
        if custom_config:
            config = self._apply_runtime_config(custom_config)
        
        # Execute pipeline stages
        executed_stages = []
        skipped_stages = []
        failed_stages = []
        
        for stage in config.pipeline_stages:
            stage_start = datetime.now(timezone.utc)
            
            # Check if stage should execute
            if not stage.should_execute(context):
                skipped_stages.append({
                    'name': stage.name,
                    'reason': 'condition_not_met_or_disabled'
                })
                continue
            
            # Check dependencies
            if not self._check_dependencies(stage, executed_stages):
                skipped_stages.append({
                    'name': stage.name,
                    'reason': 'dependencies_not_met'
                })
                continue
            
            # Check if component is available
            if stage.component not in self.components:
                if stage.config.required:
                    error_msg = f"Required component {stage.component.value} not available"
                    logger.error(error_msg)
                    failed_stages.append({
                        'name': stage.name,
                        'error': error_msg
                    })
                    if config.skip_on_error:
                        continue
                    else:
                        raise RuntimeError(error_msg)
                else:
                    skipped_stages.append({
                        'name': stage.name,
                        'reason': 'component_not_available'
                    })
                    continue
            
            # Execute stage
            try:
                result = self._execute_stage(stage, context)
                context['results'][stage.name] = result
                executed_stages.append({
                    'name': stage.name,
                    'duration_ms': (datetime.now(timezone.utc) - stage_start).total_seconds() * 1000
                })
                
                logger.debug({
                    "msg": f"Stage {stage.name} completed",
                    "correlation_id": correlation_id,
                    "duration_ms": (datetime.now(timezone.utc) - stage_start).total_seconds() * 1000
                })
                
            except Exception as e:
                logger.error({
                    "msg": f"Stage {stage.name} failed",
                    "correlation_id": correlation_id,
                    "error": str(e)
                })
                
                failed_stages.append({
                    'name': stage.name,
                    'error': str(e)
                })
                
                if stage.config.required and not config.skip_on_error:
                    raise
        
        # Build final result
        end_time = datetime.now(timezone.utc)
        total_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        result = {
            'status': 'success' if not failed_stages else 'partial',
            'correlation_id': correlation_id,
            'domain': self.config.domain.value,
            'orchestrator': self.config.name,
            'execution': {
                'started_at': start_time.isoformat(),
                'completed_at': end_time.isoformat(),
                'duration_ms': total_duration_ms,
                'stages_executed': len(executed_stages),
                'stages_skipped': len(skipped_stages),
                'stages_failed': len(failed_stages),
            },
            'results': context['results'],
            'executed_stages': executed_stages,
            'skipped_stages': skipped_stages,
            'failed_stages': failed_stages,
        }
        
        # Store in history
        self.execution_history.append(result)
        
        logger.info({
            "msg": "Orchestrated processing completed",
            "correlation_id": correlation_id,
            "duration_ms": total_duration_ms,
            "stages_executed": len(executed_stages),
            "timestamp": end_time.isoformat()
        })
        
        return result
    
    def _check_dependencies(self, stage: PipelineStage, executed_stages: List[Dict]) -> bool:
        """Check if all dependencies for a stage are met"""
        executed_names = {s['name'] for s in executed_stages}
        return all(dep in executed_names for dep in stage.depends_on)
    
    async def _execute_stage_with_timeout_and_retry(
        self,
        stage: PipelineStage,
        context: Dict[str, Any]
    ) -> Any:
        """
        Execute a pipeline stage with timeout and retry logic.
        
        Enforces:
        - timeout_seconds from ComponentConfig
        - retry_count from ComponentConfig
        - fallback_enabled from ComponentConfig
        """
        config = stage.config
        timeout = config.timeout_seconds
        max_retries = config.retry_count
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Execute with timeout
                return await asyncio.wait_for(
                    self._execute_stage_internal(stage, context),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                logger.warning({
                    "msg": f"Stage {stage.name} timeout (attempt {attempt + 1}/{max_retries})",
                    "stage": stage.name,
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "timeout": timeout
                })
            except Exception as e:
                last_error = str(e)
                logger.warning({
                    "msg": f"Stage {stage.name} error (attempt {attempt + 1}/{max_retries})",
                    "stage": stage.name,
                    "attempt": attempt + 1,
                    "error": str(e)
                })
            
            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)  # Max 30s wait
                await asyncio.sleep(wait_time)
        
        # All retries exhausted
        if config.fallback_enabled:
            logger.info({
                "msg": f"Using fallback for stage {stage.name}",
                "stage": stage.name
            })
            return await self._execute_fallback(stage, context)
        
        raise RuntimeError(
            f"Stage {stage.name} failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    async def _execute_fallback(self, stage: PipelineStage, context: Dict[str, Any]) -> Any:
        """Execute fallback behavior for a failed stage."""
        return {
            'status': 'fallback',
            'stage': stage.name,
            'reason': 'max_retries_exceeded',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _execute_stage_internal(self, stage: PipelineStage, context: Dict[str, Any]) -> Any:
        """Internal method to execute a single pipeline stage"""
        component = self.components.get(stage.component)
        
        if component is None:
            raise RuntimeError(f"Component {stage.component.value} not initialized")
        
        # Get input data for this stage
        input_data = context['input']
        
        # Execute based on component type
        handlers = {
            ComponentType.DEEPKE: self._handle_deepke,
            ComponentType.KARATE_CLUB: self._handle_karate_club,
            ComponentType.KG_GEN: self._handle_kg_gen,
            ComponentType.PAMI: self._handle_pami,
            ComponentType.NEURALKG: self._handle_neuralkg,
            ComponentType.CAUSAL_LEARN: self._handle_causal_learn,
            ComponentType.LAGRANGE_MAPPER: self._handle_lagrange_mapper,
            ComponentType.GLOBAL_CHEM: self._handle_global_chem,
            ComponentType.NEUROMANCER: self._handle_neuromancer,
        }
        
        handler = handlers.get(stage.component)
        if handler:
            return await handler(component, input_data, context, stage.config)

        # Component handler not found - this is a configuration error
        # Log the error and provide helpful diagnostic information
        error_msg = (
            f"No handler implemented for component {stage.component.value}. "
            f"Available handlers: {list(handlers.keys())}. "
            f"This indicates a mismatch between ComponentType enum and handler implementations."
        )
        logger.error({
            "msg": "Component handler not found",
            "component": stage.component.value,
            "stage": stage.name,
            "available_handlers": list(handlers.keys()),
            "severity": "CRITICAL"
        })

        raise NotImplementedError(error_msg)
    
    def _execute_stage(self, stage: PipelineStage, context: Dict[str, Any]) -> Any:
        """Synchronous wrapper for _execute_stage_with_timeout_and_retry"""
        # This method is kept for backward compatibility
        # It should be called from async context
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._execute_stage_with_timeout_and_retry(stage, context)
            )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(
                self._execute_stage_with_timeout_and_retry(stage, context)
            )
    
    async def _handle_deepke(self, component, input_data, context, config):
        """Handle DeepKE extraction"""
        text = input_data.get('text', '')
        if not text:
            return {'status': 'skipped', 'reason': 'no_text_input'}
        return component.extract_with_deepke(text, config.config_override)
    
    async def _handle_karate_club(self, component, input_data, context, config):
        """Handle Karate Club graph analysis"""
        graph_data = context['results'].get('build_graph', {}).get('graph')
        if not graph_data:
            graph_data = input_data.get('graph')
        if not graph_data:
            return {'status': 'skipped', 'reason': 'no_graph_data'}
        return component.analyze_graph(graph_data, config.config_override)
    
    async def _handle_kg_gen(self, component, input_data, context, config):
        """Handle KG-Gen graph generation"""
        artifacts = context['results'].get('extract_knowledge', {}).get('artifacts', [])
        if not artifacts:
            # Create simple artifacts from input
            artifacts = [{'content': input_data.get('text', '')}]
        return component.generate_and_store_knowledge_graph(artifacts, config.config_override)
    
    async def _handle_pami(self, component, input_data, context, config):
        """Handle PAMI pattern mining"""
        if component is None:
            return {'status': 'skipped', 'reason': 'component_not_available'}
        # Extract transactions from artifacts
        artifacts = context['results'].get('extract_knowledge', {}).get('artifacts', [])
        transactions = self._extract_transactions(artifacts)
        return component.mine_frequent_patterns(
            transactions=transactions,
            min_support=config.config_override.get('min_support', 0.1)
        )
    
    async def _handle_neuralkg(self, component, input_data, context, config):
        """Handle NeuralKG embeddings"""
        if component is None:
            return {'status': 'skipped', 'reason': 'component_not_available'}
        
        # Get triples from graph
        graph_data = context['results'].get('build_graph', {}).get('graph')
        if not graph_data:
            return {'status': 'skipped', 'reason': 'no_graph_data'}
        
        triples = self._extract_triples(graph_data)
        if not triples:
            return {'status': 'skipped', 'reason': 'no_triples'}
        
        return component.generate_embeddings(
            triples=triples,
            model_name=config.config_override.get('model', 'transe'),
            embedding_dim=config.config_override.get('embedding_dim', 100)
        )
    
    async def _handle_causal_learn(self, component, input_data, context, config):
        """Handle Causal-Learn discovery"""
        if component is None:
            return {'status': 'skipped', 'reason': 'component_not_available'}
        
        # Get data matrix from input or previous results
        data = input_data.get('data_matrix')
        if data is None:
            return {'status': 'skipped', 'reason': 'no_data_matrix'}
        
        import numpy as np
        return component.discover_causal_structure(
            data=np.array(data),
            variable_names=input_data.get('variable_names'),
            algorithm=config.config_override.get('algorithm', 'pc'),
            alpha=config.config_override.get('alpha', 0.05)
        )
    
    async def _handle_lagrange_mapper(self, component, input_data, context, config):
        """Handle Lagrange-Mapper analysis"""
        if component is None:
            return {'status': 'skipped', 'reason': 'component_not_available'}
        
        # Get embeddings from previous stage
        embeddings_data = context['results'].get('generate_embeddings', {}).get('embeddings', {})
        entities = embeddings_data.get('entities', {})
        
        if not entities:
            return {'status': 'skipped', 'reason': 'no_embeddings'}
        
        import numpy as np
        embeddings = np.array(list(entities.values()))
        labels = list(entities.keys())
        
        return component.analyze_embedding_landscape(
            embeddings=embeddings,
            labels=labels,
            n_clusters=config.config_override.get('n_clusters', 8)
        )
    
    async def _handle_global_chem(self, component, input_data, context, config):
        """Handle GlobalChem chemical analysis"""
        if component is None:
            return {'status': 'skipped', 'reason': 'component_not_available'}
        
        text = input_data.get('text', '')
        if not text:
            return {'status': 'skipped', 'reason': 'no_text_input'}
        
        entities = component.recognize_chemical_entities(text)
        return {
            'status': 'success',
            'entities': entities,
            'count': len(entities)
        }
    
    async def _handle_neuromancer(self, component, input_data, context, config):
        """Handle Neuromancer dynamics modeling"""
        if component is None:
            return {'status': 'skipped', 'reason': 'component_not_available'}
        
        time_series = input_data.get('time_series')
        time_points = input_data.get('time_points')
        
        if time_series is None or time_points is None:
            return {'status': 'skipped', 'reason': 'no_time_series_data'}
        
        import numpy as np
        return component.train_neural_ode(
            time_series_data=np.array(time_series),
            time_points=np.array(time_points),
            config=config.config_override
        )
    
    def _extract_transactions(self, artifacts: List[Dict]) -> List[List[str]]:
        """Extract transaction format from artifacts"""
        transactions = []
        for artifact in artifacts:
            transaction = []
            if isinstance(artifact, dict):
                for key, value in artifact.items():
                    if value:
                        transaction.append(f"{key}:{str(value)[:50]}")
            if transaction:
                transactions.append(transaction)
        return transactions if transactions else [['empty']]
    
    def _extract_triples(self, graph_data: Dict) -> List[Tuple[str, str, str]]:
        """Extract triples from graph data"""
        triples = []
        edges = graph_data.get('edges', [])
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            rel_type = edge.get('type', 'related_to')
            if source and target:
                triples.append((source, rel_type, target))
        return triples
    
    def _apply_runtime_config(self, custom_config: Dict[str, Any]) -> OrchestratorConfig:
        """Apply runtime configuration overrides"""
        config = copy.deepcopy(self.config)
        
        # Override component settings
        for comp_name, comp_settings in custom_config.get('components', {}).items():
            try:
                comp_type = ComponentType(comp_name)
                if comp_type in config.components:
                    for key, value in comp_settings.items():
                        setattr(config.components[comp_type], key, value)
            except ValueError:
                logger.warning(f"Unknown component: {comp_name}")
        
        # Override pipeline stages
        for stage_override in custom_config.get('pipeline_stages', []):
            for stage in config.pipeline_stages:
                if stage.name == stage_override.get('name'):
                    for key, value in stage_override.items():
                        if key != 'name':
                            setattr(stage, key, value)
        
        return config
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            'name': self.config.name,
            'domain': self.config.domain.value,
            'initialized_components': list(self.components.keys()),
            'configured_components': [
                k.value for k, v in self.config.components.items() if v.enabled
            ],
            'pipeline_stages': len(self.config.pipeline_stages),
            'execution_history_count': len(self.execution_history),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status including all components.
        
        This is an async version of get_status for compatibility with
        the OpenEvolveKnowledgeEngine interface.
        """
        return self.get_status()
    
    async def close(self):
        """Close all resources and connections."""
        logger.info({
            "msg": "Closing KnowledgeOrchestrator resources",
            "name": self.config.name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close any components that have close methods
        for comp_type, component in self.components.items():
            if component and hasattr(component, 'close'):
                try:
                    if asyncio.iscoroutinefunction(component.close):
                        await component.close()
                    else:
                        component.close()
                except Exception as e:
                    logger.warning(f"Error closing component {comp_type.value}: {e}")
        
        # Clear components
        self.components.clear()
        self.cache.clear()
    
    def save_config(self, path: str):
        """Save configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> 'KnowledgeOrchestrator':
        """Load configuration from file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config = OrchestratorConfig.from_dict(config_dict)
        return cls(config)


# Convenience factory functions
def create_finance_orchestrator(**kwargs) -> KnowledgeOrchestrator:
    """Create an orchestrator optimized for finance"""
    config = DomainPresets.finance()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return KnowledgeOrchestrator(config)


def create_chemistry_orchestrator(**kwargs) -> KnowledgeOrchestrator:
    """Create an orchestrator optimized for chemistry"""
    config = DomainPresets.chemistry()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return KnowledgeOrchestrator(config)


def create_healthcare_orchestrator(**kwargs) -> KnowledgeOrchestrator:
    """Create an orchestrator optimized for healthcare"""
    config = DomainPresets.healthcare()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return KnowledgeOrchestrator(config)


def create_research_orchestrator(**kwargs) -> KnowledgeOrchestrator:
    """Create an orchestrator for comprehensive research"""
    config = DomainPresets.research()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return KnowledgeOrchestrator(config)


def create_minimal_orchestrator(**kwargs) -> KnowledgeOrchestrator:
    """Create a minimal orchestrator with only essential components"""
    config = DomainPresets.minimal()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return KnowledgeOrchestrator(config)
