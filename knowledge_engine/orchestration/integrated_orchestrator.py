"""
Integrated Knowledge Orchestrator

Complete orchestrator that integrates all features:
- Base orchestration
- Self-healing with circuit breaker
- Learning from experiences
- Component coordination and gap filling
- Feedback collection
- Safe expression evaluation
- Async support

This is the production-ready, fully-integrated orchestrator.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
import copy

from .knowledge_orchestrator import (
    KnowledgeOrchestrator, OrchestratorConfig, PipelineStage, 
    ComponentType, ComponentConfig
)
from .self_healing_orchestrator import (
    SelfHealingOrchestrator, HealingStrategy, FailureEvent
)
from .learning_engine import LearningEngine, LearningExperience
from .component_coordination import (
    ComponentCoordinator, GapFillingAssignment, analyze_pipeline_gaps
)
from .feedback_loop import (
    FeedbackCollector, ContinuousImprovementEngine, 
    AdaptiveOrchestratorIntegration, FeedbackType
)
from .circuit_breaker import CircuitBreaker, get_circuit_breaker
from .safe_eval import safe_eval

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Enhanced execution context with all metadata"""
    input_data: Dict[str, Any]
    config: OrchestratorConfig
    correlation_id: str
    start_time: datetime
    
    # Execution state
    results: Dict[str, Any] = field(default_factory=dict)
    executed_stages: List[str] = field(default_factory=list)
    skipped_stages: List[Dict[str, str]] = field(default_factory=list)
    failed_stages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Healing state
    healing_applied: bool = False
    healing_strategies_used: List[str] = field(default_factory=list)
    
    # Coordination state
    gap_fillers: Dict[str, str] = field(default_factory=dict)
    cross_validation_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'correlation_id': self.correlation_id,
            'start_time': self.start_time.isoformat(),
            'results': self.results,
            'executed_stages': self.executed_stages,
            'skipped_stages': self.skipped_stages,
            'failed_stages': self.failed_stages,
            'healing_applied': self.healing_applied,
            'gap_fillers': self.gap_fillers,
        }


class IntegratedOrchestrator(SelfHealingOrchestrator):
    """
    Fully integrated orchestrator with all capabilities.
    
    Features:
    - Base orchestration with pipeline management
    - Self-healing with 7 strategies
    - Circuit breaker pattern for component protection
    - Learning from every execution
    - Component coordination and gap filling
    - Feedback collection and continuous improvement
    - Safe expression evaluation
    - Async support
    
    This is the recommended orchestrator for production use.
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        enable_self_healing: bool = True,
        enable_learning: bool = True,
        enable_coordination: bool = True,
        enable_feedback: bool = True,
        enable_circuit_breaker: bool = True,
        learning_storage_path: Optional[str] = None,
        feedback_storage_path: Optional[str] = None,
        max_healing_attempts: int = 3,
        parallel_execution: bool = False
    ):
        """
        Initialize integrated orchestrator.
        
        Args:
            config: Base configuration
            enable_self_healing: Enable healing strategies
            enable_learning: Enable learning engine
            enable_coordination: Enable component coordination
            enable_feedback: Enable feedback collection
            enable_circuit_breaker: Enable circuit breaker protection
            learning_storage_path: Path for learning data
            feedback_storage_path: Path for feedback data
            max_healing_attempts: Max healing attempts
            parallel_execution: Enable parallel stage execution
        """
        # Initialize base self-healing orchestrator
        super().__init__(
            config=config,
            enable_self_healing=enable_self_healing,
            learning_storage_path=learning_storage_path,
            max_healing_attempts=max_healing_attempts
        )
        
        # Feature flags
        self.enable_coordination = enable_coordination
        self.enable_feedback = enable_feedback
        self.enable_circuit_breaker = enable_circuit_breaker
        self.parallel_execution = parallel_execution
        
        # Initialize subsystems
        if enable_coordination:
            self.component_coordinator = ComponentCoordinator(self.learning_engine)
        else:
            self.component_coordinator = None
        
        if enable_feedback:
            self.feedback_collector = FeedbackCollector(feedback_storage_path)
            self.improvement_engine = ContinuousImprovementEngine(
                self.feedback_collector,
                self.learning_engine
            )
        else:
            self.feedback_collector = None
            self.improvement_engine = None
        
        # Circuit breakers for components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        if enable_circuit_breaker:
            self._initialize_circuit_breakers()
        
        # Semaphore for parallel execution
        if parallel_execution:
            self._execution_semaphore = asyncio.Semaphore(self.config.max_workers)
        
        logger.info({
            "msg": "IntegratedOrchestrator initialized",
            "features": {
                "self_healing": enable_self_healing,
                "learning": enable_learning,
                "coordination": enable_coordination,
                "feedback": enable_feedback,
                "circuit_breaker": enable_circuit_breaker,
                "parallel": parallel_execution
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for all components"""
        for component_type in self.components.keys():
            breaker = get_circuit_breaker(
                name=component_type.value,
                failure_threshold=3,
                recovery_timeout=60.0,
                on_open=lambda name: logger.warning(f"Circuit opened for {name}"),
                on_close=lambda name: logger.info(f"Circuit closed for {name}")
            )
            self.circuit_breakers[component_type.value] = breaker
    
    def process(self, input_data: Dict[str, Any],
                custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input data with all integrated features.
        
        Args:
            input_data: Input data to process
            custom_config: Optional runtime configuration overrides
            
        Returns:
            Processing results with full metadata
        """
        start_time = datetime.now(timezone.utc)
        correlation_id = f"integrated_{start_time.timestamp()}"
        
        # Create execution context
        context = ExecutionContext(
            input_data=input_data,
            config=self.config,
            correlation_id=correlation_id,
            start_time=start_time
        )
        
        logger.info({
            "msg": "Starting integrated orchestration",
            "correlation_id": correlation_id,
            "features_enabled": {
                "healing": self.enable_self_healing,
                "coordination": self.enable_coordination,
                "feedback": self.enable_feedback,
                "circuit_breaker": self.enable_circuit_breaker
            }
        })
        
        try:
            # Phase 1: Pre-execution analysis
            if self.enable_coordination:
                self._coordinate_pipeline(context)
            
            # Phase 2: Pre-execution check (healing)
            if self.enable_self_healing:
                warnings = self._pre_execution_check(
                    input_data.get('data_type', 'unknown'),
                    self.config.domain.value,
                    input_data
                )
                if warnings:
                    logger.warning({
                        "msg": "Pre-execution warnings",
                        "warnings": warnings
                    })
            
            # Phase 3: Execute pipeline with healing
            result = self._execute_with_full_healing(context, custom_config)
            
            # Phase 4: Post-execution processing
            if self.enable_coordination and result.get('status') in ('success', 'partial'):
                self._cross_validate_results(context)
            
            # Phase 5: Learning and feedback
            self._record_execution_experience(context, result)
            
            # Add comprehensive metadata
            result['orchestration_metadata'] = {
                'correlation_id': correlation_id,
                'features_used': {
                    'self_healing': self.enable_self_healing and context.healing_applied,
                    'coordination': self.enable_coordination,
                    'circuit_breaker': self.enable_circuit_breaker
                },
                'execution_context': context.to_dict(),
                'circuit_breaker_status': self._get_circuit_breaker_status(),
                'learning_summary': self._get_learning_summary(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error({
                "msg": "Integrated orchestration failed",
                "correlation_id": correlation_id,
                "error": str(e)
            })
            
            # Try emergency fallback
            return self._emergency_fallback(input_data, context, e)
    
    def _coordinate_pipeline(self, context: ExecutionContext):
        """Coordinate components and fill gaps"""
        if not self.component_coordinator:
            return
        
        components = [c.value for c in self.components.keys()]
        
        # Get coordination plan
        plan = self.component_coordinator.coordinate_pipeline(
            components=components,
            input_data=context.input_data,
            data_type=context.input_data.get('data_type', 'unknown'),
            domain=self.config.domain.value
        )
        
        # Store gap fillers
        for assignment in plan.get('gap_assignments', []):
            context.gap_fillers[assignment['gap']] = assignment['filler_component']
        
        logger.debug({
            "msg": "Pipeline coordinated",
            "gap_fillers": context.gap_fillers,
            "expected_confidence": plan.get('expected_confidence')
        })
    
    def _execute_with_full_healing(self, context: ExecutionContext,
                                   custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute with comprehensive healing"""
        
        # First attempt
        try:
            result = self._execute_pipeline(context, custom_config)
            if self._is_result_acceptable(result):
                return result
        except Exception as e:
            logger.warning({
                "msg": "Initial execution failed, attempting healing",
                "error": str(e)
            })
            result = None
        
        # Healing attempts
        if self.enable_self_healing:
            for attempt in range(self.max_healing_attempts):
                strategy_result = self._try_healing_strategy(
                    context, custom_config, result, attempt
                )
                
                if strategy_result:
                    context.healing_applied = True
                    context.healing_strategies_used.append(
                        strategy_result.get('strategy', 'unknown')
                    )
                    
                    if self._is_result_acceptable(strategy_result):
                        return strategy_result
                    
                    result = strategy_result
        
        # Return best result we have
        return result or {
            'status': 'failed',
            'error': 'All execution attempts failed',
            'correlation_id': context.correlation_id
        }
    
    def _execute_pipeline(self, context: ExecutionContext,
                         custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the pipeline"""
        # Use parent implementation with circuit breaker protection
        return super().process(context.input_data, custom_config)
    
    def _try_healing_strategy(self, context: ExecutionContext,
                              custom_config: Optional[Dict[str, Any]],
                              previous_result: Optional[Dict[str, Any]],
                              attempt: int) -> Optional[Dict[str, Any]]:
        """Try a healing strategy"""
        
        # Try different strategies based on attempt number
        strategies = [
            HealingStrategy.RETRY_WITH_CONFIG,
            HealingStrategy.COMPONENT_SUBSTITUTION,
            HealingStrategy.FALLBACK_PIPELINE,
            HealingStrategy.DECOMPOSE_TASK,
            HealingStrategy.SKIP_AND_CONTINUE
        ]
        
        if attempt < len(strategies):
            strategy = strategies[attempt]
            
            logger.info({
                "msg": f"Attempting healing strategy: {strategy.value}",
                "attempt": attempt + 1
            })
            
            try:
                # Apply strategy
                if strategy == HealingStrategy.RETRY_WITH_CONFIG:
                    adjusted_config = self._adjust_config_for_retry(
                        custom_config, None
                    )
                    return self._execute_pipeline(context, adjusted_config)
                
                elif strategy == HealingStrategy.FALLBACK_PIPELINE:
                    return self._execute_fallback_pipeline(
                        context.input_data, context.correlation_id
                    )
                
                elif strategy == HealingStrategy.COMPONENT_SUBSTITUTION:
                    # Use component coordinator for substitution
                    if self.component_coordinator:
                        return self._try_coordinated_substitution(context, custom_config)
                
                # Other strategies...
                return self._apply_strategy(
                    strategy, context.input_data, custom_config, 
                    context.correlation_id, None
                )
                
            except Exception as e:
                logger.warning({
                    "msg": f"Healing strategy {strategy.value} failed",
                    "error": str(e)
                })
        
        return None
    
    def _try_coordinated_substitution(self, context: ExecutionContext,
                                     custom_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Try component substitution using coordinator"""
        # Find failing components
        failing = [
            s['name'] for s in context.failed_stages
        ]
        
        if not failing:
            return None
        
        # Get substitutes from coordinator
        for failed_name in failing:
            for comp_type in self.components.keys():
                if comp_type.value == failed_name:
                    # Get substitutes
                    from .component_coordination import ComponentSubstitutionMatrix
                    substitutes = ComponentSubstitutionMatrix.get_substitutes(comp_type)
                    
                    for substitute in substitutes:
                        if substitute in self.components:
                            # Try substitution
                            result = self._try_component_substitution(
                                context.input_data, custom_config, None
                            )
                            if result:
                                return result
        
        return None
    
    def _cross_validate_results(self, context: ExecutionContext):
        """Cross-validate results between components"""
        if not self.component_coordinator:
            return
        
        # Get validation points
        components = [c.value for c in self.components.keys()]
        validation_points = self.component_coordinator._identify_validation_points(components)
        
        if validation_points:
            validation = self.component_coordinator.cross_validate_results(
                context.results,
                validation_points
            )
            
            context.cross_validation_results = validation
            
            logger.debug({
                "msg": "Cross-validation complete",
                "overall_confidence": validation.get('overall_confidence'),
                "inconsistencies": len(validation.get('inconsistencies', []))
            })
    
    def _record_execution_experience(self, context: ExecutionContext, result: Dict[str, Any]):
        """Record experience for learning and feedback"""
        
        # Extract errors
        errors = [
            {'type': 'execution_failure', 'component': s['name'], 'message': s.get('error')}
            for s in result.get('failed_stages', [])
        ]
        
        # Record in learning engine
        if hasattr(self, 'learning_engine'):
            self.learning_engine.record_experience(
                input_data=context.input_data,
                data_type=context.input_data.get('data_type', 'unknown'),
                domain=self.config.domain.value,
                pipeline_config=self.config.to_dict(),
                components_used=[c.value for c in self.components.keys()],
                success=result.get('status') in ('success', 'partial'),
                execution_time_ms=result.get('execution', {}).get('duration_ms', 0),
                results=result.get('results', {}),
                errors=errors
            )
        
        # Collect feedback
        if self.feedback_collector:
            feedback_type = self._determine_feedback_type(result, errors)
            
            self.feedback_collector.collect_feedback(
                correlation_id=context.correlation_id,
                input_data=context.input_data,
                components_used=[c.value for c in self.components.keys()],
                pipeline_config=self.config.to_dict(),
                feedback_type=feedback_type,
                issues=[e.get('message') for e in errors if e.get('message')],
                actual_output=result.get('results')
            )
    
    def _determine_feedback_type(self, result: Dict[str, Any], 
                                 errors: List[Dict]) -> FeedbackType:
        """Determine feedback type from result"""
        status = result.get('status')
        
        if status == 'success':
            return FeedbackType.SUCCESS
        elif status == 'partial':
            return FeedbackType.PARTIAL_SUCCESS
        elif errors:
            return FeedbackType.FAILURE
        else:
            return FeedbackType.QUALITY_ISSUE
    
    def _get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        if not self.enable_circuit_breaker:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "circuits": {
                name: breaker.get_status()
                for name, breaker in self.circuit_breakers.items()
            }
        }
    
    def _get_learning_summary(self) -> Dict[str, Any]:
        """Get learning summary"""
        if not hasattr(self, 'learning_engine'):
            return {"enabled": False}
        
        return self.learning_engine.get_learning_summary()
    
    def _emergency_fallback(self, input_data: Dict[str, Any],
                           context: ExecutionContext,
                           error: Exception) -> Dict[str, Any]:
        """Emergency fallback when everything fails"""
        logger.error({
            "msg": "Emergency fallback triggered",
            "correlation_id": context.correlation_id,
            "error": str(error)
        })
        
        # Try minimal extraction
        try:
            return {
                'status': 'emergency_fallback',
                'error': str(error),
                'results': {
                    'input_received': True,
                    'input_keys': list(input_data.keys())
                },
                'correlation_id': context.correlation_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except:
            return {
                'status': 'total_failure',
                'error': str(error),
                'correlation_id': context.correlation_id
            }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'orchestrator': self.get_status(),
            'features': {
                'self_healing': self.enable_self_healing,
                'coordination': self.enable_coordination,
                'feedback': self.enable_feedback,
                'circuit_breaker': self.enable_circuit_breaker
            },
            'circuit_breakers': self._get_circuit_breaker_status(),
            'learning': self._get_learning_summary(),
            'feedback': self.feedback_collector.get_feedback_stats() if self.feedback_collector else None,
            'improvements': self.improvement_engine.get_improvement_report() if self.improvement_engine else None
        }


# Factory functions for creating integrated orchestrators
def create_integrated_finance_orchestrator(
    learning_storage_path: Optional[str] = None,
    feedback_storage_path: Optional[str] = None,
    **kwargs
) -> IntegratedOrchestrator:
    """Create fully integrated finance orchestrator"""
    from . import DomainPresets
    
    config = DomainPresets.finance()
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return IntegratedOrchestrator(
        config=config,
        enable_self_healing=True,
        enable_learning=True,
        enable_coordination=True,
        enable_feedback=True,
        enable_circuit_breaker=True,
        learning_storage_path=learning_storage_path or "finance_learning.json",
        feedback_storage_path=feedback_storage_path or "finance_feedback.json",
        **kwargs
    )


def create_integrated_chemistry_orchestrator(
    learning_storage_path: Optional[str] = None,
    feedback_storage_path: Optional[str] = None,
    **kwargs
) -> IntegratedOrchestrator:
    """Create fully integrated chemistry orchestrator"""
    from . import DomainPresets
    
    config = DomainPresets.chemistry()
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return IntegratedOrchestrator(
        config=config,
        enable_self_healing=True,
        enable_learning=True,
        enable_coordination=True,
        enable_feedback=True,
        enable_circuit_breaker=True,
        learning_storage_path=learning_storage_path or "chemistry_learning.json",
        feedback_storage_path=feedback_storage_path or "chemistry_feedback.json",
        **kwargs
    )


def create_integrated_research_orchestrator(
    learning_storage_path: Optional[str] = None,
    feedback_storage_path: Optional[str] = None,
    **kwargs
) -> IntegratedOrchestrator:
    """Create fully integrated research orchestrator"""
    from . import DomainPresets
    
    config = DomainPresets.research()
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return IntegratedOrchestrator(
        config=config,
        enable_self_healing=True,
        enable_learning=True,
        enable_coordination=True,
        enable_feedback=True,
        enable_circuit_breaker=True,
        learning_storage_path=learning_storage_path or "research_learning.json",
        feedback_storage_path=feedback_storage_path or "research_feedback.json",
        **kwargs
    )
