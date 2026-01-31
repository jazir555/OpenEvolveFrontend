"""
Self-Healing Knowledge Orchestrator

An advanced orchestrator that:
1. Detects failures automatically
2. Diagnoses root causes
3. Applies healing strategies
4. Learns from healing actions
5. Prevents future similar failures
6. Coordinates components to cover each other's gaps

Healing Strategies:
- Retry with backoff
- Component substitution
- Fallback pipelines
- Configuration adjustment
- Parallel execution with race condition
- Decomposition into smaller tasks
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
import copy
import traceback

from .knowledge_orchestrator import (
    KnowledgeOrchestrator, OrchestratorConfig, 
    ComponentType, PipelineStage, ComponentConfig
)
from .learning_engine import LearningEngine, LearningExperience

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur"""
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    IMPORT_ERROR = "import_error"
    COMPONENT_UNAVAILABLE = "component_unavailable"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_FORMAT_ERROR = "data_format_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    QUALITY_THRESHOLD_NOT_MET = "quality_threshold_not_met"
    UNKNOWN = "unknown"


class HealingStrategy(Enum):
    """Available healing strategies"""
    RETRY = "retry"
    RETRY_WITH_CONFIG = "retry_with_config"
    COMPONENT_SUBSTITUTION = "component_substitution"
    FALLBACK_PIPELINE = "fallback_pipeline"
    PARALLEL_EXECUTION = "parallel_execution"
    DECOMPOSE_TASK = "decompose_task"
    SKIP_AND_CONTINUE = "skip_and_continue"
    ESCALATE = "escalate"


@dataclass
class FailureEvent:
    """Records a failure event"""
    failure_id: str
    timestamp: str
    component: str
    stage_name: str
    failure_type: FailureType
    error_message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Healing attempts
    healing_attempts: List[Dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    resolution_strategy: Optional[str] = None


@dataclass
class HealingAction:
    """A healing action taken"""
    action_id: str
    timestamp: str
    strategy: HealingStrategy
    target_component: str
    original_config: Dict[str, Any]
    modified_config: Dict[str, Any]
    success: bool = False
    result: Optional[Any] = None
    lessons_learned: List[str] = field(default_factory=list)


class ComponentSubstitutionMatrix:
    """
    Defines which components can substitute for each other.
    Used when a component fails to find alternatives.
    """
    
    SUBSTITUTIONS = {
        ComponentType.DEEPKE: [ComponentType.KG_GEN],  # KG-Gen can do basic extraction
        ComponentType.KARATE_CLUB: [ComponentType.NEURALKG],  # Both do graph analysis
        ComponentType.NEURALKG: [ComponentType.KARATE_CLUB],
        ComponentType.PAMI: [],  # No direct substitute for pattern mining
        ComponentType.CAUSAL_LEARN: [ComponentType.KARATE_CLUB],  # Graph structure instead of causal
        ComponentType.LAGRANGE_MAPPER: [ComponentType.NEURALKG],  # Embeddings analysis
        ComponentType.GLOBAL_CHEM: [],  # No substitute for chemistry
        ComponentType.NEUROMANCER: [ComponentType.CAUSAL_LEARN],  # Both do relationship modeling
    }
    
    @classmethod
    def get_substitutes(cls, component: ComponentType) -> List[ComponentType]:
        """Get substitute components for a failed component"""
        return cls.SUBSTITUTIONS.get(component, [])
    
    @classmethod
    def can_substitute(cls, failed: ComponentType, candidate: ComponentType) -> bool:
        """Check if candidate can substitute for failed component"""
        return candidate in cls.SUBSTITUTIONS.get(failed, [])


class SelfHealingOrchestrator(KnowledgeOrchestrator):
    """
    Self-healing orchestrator that learns from failures and adapts.
    
    Features:
    - Automatic failure detection and diagnosis
    - Multiple healing strategies
    - Component substitution when components fail
    - Fallback pipeline execution
    - Learning from healing actions
    - Gap coverage between components
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None,
                 enable_self_healing: bool = True,
                 learning_storage_path: Optional[str] = None,
                 max_healing_attempts: int = 3):
        """
        Initialize self-healing orchestrator.
        
        Args:
            config: Orchestrator configuration
            enable_self_healing: Whether to enable healing behaviors
            learning_storage_path: Path to persist learning data
            max_healing_attempts: Maximum healing attempts per failure
        """
        super().__init__(config)
        
        self.enable_self_healing = enable_self_healing
        self.max_healing_attempts = max_healing_attempts
        
        # Initialize learning engine
        self.learning_engine = LearningEngine(learning_storage_path)
        
        # Failure tracking
        self.failure_history: List[FailureEvent] = []
        self.healing_history: List[HealingAction] = []
        
        # Healing configuration
        self.healing_strategies = [
            HealingStrategy.RETRY,
            HealingStrategy.RETRY_WITH_CONFIG,
            HealingStrategy.COMPONENT_SUBSTITUTION,
            HealingStrategy.FALLBACK_PIPELINE,
            HealingStrategy.SKIP_AND_CONTINUE,
        ]
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_quality_score': 0.3,
            'max_execution_time_ms': 300000,  # 5 minutes
            'min_components_successful': 0.5,  # 50% of components
        }
        
        logger.info({
            "msg": "SelfHealingOrchestrator initialized",
            "self_healing_enabled": enable_self_healing,
            "max_healing_attempts": max_healing_attempts,
            "experiences": len(self.learning_engine.experiences),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def process(self, input_data: Dict[str, Any],
                custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process data with self-healing capabilities.
        
        Args:
            input_data: Input data to process
            custom_config: Optional runtime configuration
            
        Returns:
            Processing results with healing metadata
        """
        start_time = datetime.now(timezone.utc)
        correlation_id = self.config.correlation_id or f"heal_{start_time.timestamp()}"
        
        logger.info({
            "msg": "Starting self-healing orchestration",
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        # Check for learned recommendations
        data_type = input_data.get('data_type', 'unknown')
        
        # Predict potential failures before they happen
        pre_check = self._pre_execution_check(data_type, self.config.domain.value, input_data)
        if pre_check:
            logger.warning({
                "msg": "Pre-execution check warning",
                "correlation_id": correlation_id,
                "warnings": pre_check
            })
        
        # Try to get learned pipeline recommendation
        learned_pipeline = self.learning_engine.recommend_pipeline(
            data_type, self.config.domain.value, input_data
        )
        
        if learned_pipeline and not custom_config:
            logger.info({
                "msg": "Using learned pipeline configuration",
                "correlation_id": correlation_id,
                "expected_success_rate": learned_pipeline['expected_success_rate'],
                "based_on_experiences": learned_pipeline['based_on_experiences']
            })
            
            # Apply learned configuration
            custom_config = self._apply_learned_config(learned_pipeline)
        
        # Execute with healing
        execution_result = self._execute_with_healing(
            input_data, custom_config, correlation_id
        )
        
        # Analyze results and learn
        self._post_execution_learning(
            input_data, execution_result, correlation_id
        )
        
        # Add healing metadata
        execution_result['healing_metadata'] = {
            'healing_enabled': self.enable_self_healing,
            'failures_encountered': len([f for f in self.failure_history 
                                        if f.timestamp >= start_time.isoformat()]),
            'healing_actions_taken': len([h for h in self.healing_history
                                         if h.timestamp >= start_time.isoformat()]),
            'learning_summary': self.learning_engine.get_learning_summary()
        }
        
        return execution_result
    
    def _pre_execution_check(self, data_type: str, domain: str, 
                            input_data: Dict[str, Any]) -> Optional[List[Dict]]:
        """Check for potential issues before execution"""
        warnings = []
        
        # Get planned components
        planned_components = [c.value for c in self.components.keys()]
        
        # Check for predicted failures
        prediction = self.learning_engine.predict_failure(
            data_type, domain, planned_components
        )
        
        if prediction:
            warnings.extend(prediction.get('warnings', []))
        
        # Check input data characteristics
        text = input_data.get('text', '')
        if len(text) > 100000:  # Very long text
            warnings.append({
                'type': 'input_size',
                'message': 'Very long input text, consider chunking',
                'recommendation': 'Enable decompose_task strategy'
            })
        
        return warnings if warnings else None
    
    def _apply_learned_config(self, learned_pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned pipeline configuration"""
        config = {
            'pipeline_override': learned_pipeline['component_sequence'],
            'component_configs': learned_pipeline['component_configs']
        }
        return config
    
    def _execute_with_healing(self, input_data: Dict[str, Any],
                             custom_config: Optional[Dict[str, Any]],
                             correlation_id: str) -> Dict[str, Any]:
        """Execute pipeline with healing capabilities"""
        
        # First attempt - try normal execution
        try:
            result = super().process(input_data, custom_config)
            
            # Check if result quality is acceptable
            if self._is_result_acceptable(result):
                return result
            else:
                logger.warning({
                    "msg": "Result quality below threshold, attempting healing",
                    "correlation_id": correlation_id
                })
                # Continue to healing
                
        except Exception as e:
            logger.error({
                "msg": "Initial execution failed, attempting healing",
                "correlation_id": correlation_id,
                "error": str(e)
            })
            # Continue to healing
        
        # If we're here, we need healing
        if not self.enable_self_healing:
            # Return the best result we have or raise
            return result if 'result' in locals() else {
                'status': 'failed',
                'error': 'Execution failed and self-healing disabled',
                'correlation_id': correlation_id
            }
        
        # Try healing strategies
        return self._apply_healing_strategies(
            input_data, custom_config, correlation_id, result if 'result' in locals() else None
        )
    
    def _is_result_acceptable(self, result: Dict[str, Any]) -> bool:
        """Check if execution result meets quality thresholds"""
        # Check success status
        if result.get('status') != 'success' and result.get('status') != 'partial':
            return False
        
        # Check execution time
        execution_time = result.get('execution', {}).get('duration_ms', 0)
        if execution_time > self.quality_thresholds['max_execution_time_ms']:
            return False
        
        # Check stage success rate
        total_stages = result.get('execution', {}).get('stages_executed', 0)
        failed_stages = result.get('execution', {}).get('stages_failed', 0)
        
        if total_stages > 0:
            success_rate = (total_stages - failed_stages) / total_stages
            if success_rate < self.quality_thresholds['min_components_successful']:
                return False
        
        return True
    
    def _apply_healing_strategies(self, input_data: Dict[str, Any],
                                  custom_config: Optional[Dict[str, Any]],
                                  correlation_id: str,
                                  failed_result: Optional[Dict]) -> Dict[str, Any]:
        """Apply healing strategies to recover from failure"""
        
        healing_attempts = 0
        last_error = None
        
        for strategy in self.healing_strategies:
            if healing_attempts >= self.max_healing_attempts:
                break
            
            healing_attempts += 1
            
            logger.info({
                "msg": f"Attempting healing strategy: {strategy.value}",
                "correlation_id": correlation_id,
                "attempt": healing_attempts
            })
            
            try:
                healed_result = self._apply_strategy(
                    strategy, input_data, custom_config, correlation_id, last_error
                )
                
                if healed_result and self._is_result_acceptable(healed_result):
                    logger.info({
                        "msg": f"Healing successful with strategy: {strategy.value}",
                        "correlation_id": correlation_id,
                        "attempts": healing_attempts
                    })
                    
                    # Record successful healing
                    self._record_healing_action(
                        strategy, True, healed_result, input_data, correlation_id
                    )
                    
                    # Mark result as healed
                    healed_result['healed'] = True
                    healed_result['healing_strategy'] = strategy.value
                    healed_result['healing_attempts'] = healing_attempts
                    
                    return healed_result
                
            except Exception as e:
                last_error = e
                logger.warning({
                    "msg": f"Healing strategy failed: {strategy.value}",
                    "correlation_id": correlation_id,
                    "error": str(e)
                })
                
                # Record failed healing attempt
                self._record_healing_action(
                    strategy, False, None, input_data, correlation_id, str(e)
                )
        
        # All healing strategies exhausted
        logger.error({
            "msg": "All healing strategies exhausted",
            "correlation_id": correlation_id,
            "attempts": healing_attempts
        })
        
        return {
            'status': 'failed',
            'error': f'All healing strategies exhausted after {healing_attempts} attempts',
            'last_error': str(last_error) if last_error else None,
            'correlation_id': correlation_id,
            'healing_attempts': healing_attempts
        }
    
    def _apply_strategy(self, strategy: HealingStrategy, input_data: Dict[str, Any],
                       custom_config: Optional[Dict[str, Any]],
                       correlation_id: str, last_error: Optional[Exception]) -> Optional[Dict[str, Any]]:
        """Apply a specific healing strategy"""
        
        if strategy == HealingStrategy.RETRY:
            # Simple retry with same configuration
            time.sleep(1)  # Brief pause before retry
            return super().process(input_data, custom_config)
        
        elif strategy == HealingStrategy.RETRY_WITH_CONFIG:
            # Retry with adjusted configuration
            adjusted_config = self._adjust_config_for_retry(custom_config, last_error)
            return super().process(input_data, adjusted_config)
        
        elif strategy == HealingStrategy.COMPONENT_SUBSTITUTION:
            # Try substituting failed components
            return self._try_component_substitution(input_data, custom_config, last_error)
        
        elif strategy == HealingStrategy.FALLBACK_PIPELINE:
            # Use minimal fallback pipeline
            return self._execute_fallback_pipeline(input_data, correlation_id)
        
        elif strategy == HealingStrategy.PARALLEL_EXECUTION:
            # Execute multiple component options in parallel
            return self._execute_parallel_options(input_data, custom_config)
        
        elif strategy == HealingStrategy.DECOMPOSE_TASK:
            # Break task into smaller chunks
            return self._execute_decomposed(input_data, custom_config, correlation_id)
        
        elif strategy == HealingStrategy.SKIP_AND_CONTINUE:
            # Skip failed components and continue
            return self._execute_with_skips(input_data, custom_config, last_error)
        
        return None
    
    def _adjust_config_for_retry(self, custom_config: Optional[Dict[str, Any]],
                                  last_error: Optional[Exception]) -> Dict[str, Any]:
        """Adjust configuration based on error type"""
        config = copy.deepcopy(custom_config) if custom_config else {}
        
        if config is None:
            config = {}
        
        # Adjust timeouts if timeout error
        if last_error and 'timeout' in str(last_error).lower():
            if 'components' not in config:
                config['components'] = {}
            for comp in config['components']:
                config['components'][comp]['timeout_seconds'] = 120  # Increase timeout
        
        # Adjust batch size if memory error
        if last_error and 'memory' in str(last_error).lower():
            config['batch_size'] = 10  # Reduce batch size
        
        return config
    
    def _try_component_substitution(self, input_data: Dict[str, Any],
                                    custom_config: Optional[Dict[str, Any]],
                                    last_error: Optional[Exception]) -> Dict[str, Any]:
        """Try substituting failed components with alternatives"""
        
        # Identify failed component from error
        failed_component = self._extract_failed_component(last_error)
        
        if not failed_component:
            raise ValueError("Could not identify failed component for substitution")
        
        # Get substitutes
        substitutes = ComponentSubstitutionMatrix.get_substitutes(failed_component)
        
        if not substitutes:
            raise ValueError(f"No substitutes available for {failed_component.value}")
        
        # Try each substitute
        for substitute in substitutes:
            if substitute in self.components:
                logger.info({
                    "msg": f"Trying substitute: {substitute.value} for {failed_component.value}"
                })
                
                # Create modified pipeline
                modified_pipeline = self._create_substituted_pipeline(
                    failed_component, substitute
                )
                
                modified_config = copy.deepcopy(custom_config) if custom_config else {}
                modified_config['pipeline_override'] = modified_pipeline
                
                try:
                    result = super().process(input_data, modified_config)
                    
                    # Record the successful substitution
                    result['substitution_applied'] = {
                        'original': failed_component.value,
                        'substitute': substitute.value
                    }
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Substitute {substitute.value} also failed: {e}")
                    continue
        
        raise RuntimeError("All component substitutions failed")
    
    def _extract_failed_component(self, error: Optional[Exception]) -> Optional[ComponentType]:
        """Extract failed component from error"""
        if not error:
            return None
        
        error_str = str(error).lower()
        
        # Map error messages to components
        component_mapping = {
            'deepke': ComponentType.DEEPKE,
            'karate': ComponentType.KARATE_CLUB,
            'kg_gen': ComponentType.KG_GEN,
            'pami': ComponentType.PAMI,
            'neuralkg': ComponentType.NEURALKG,
            'causal': ComponentType.CAUSAL_LEARN,
            'lagrange': ComponentType.LAGRANGE_MAPPER,
            'globalchem': ComponentType.GLOBAL_CHEM,
            'neuromancer': ComponentType.NEUROMANCER,
        }
        
        for key, component in component_mapping.items():
            if key in error_str:
                return component
        
        return None
    
    def _create_substituted_pipeline(self, original: ComponentType, 
                                     substitute: ComponentType) -> List[str]:
        """Create pipeline with component substituted"""
        original_pipeline = [stage.name for stage in self.config.pipeline_stages]
        
        # Replace component in pipeline
        substituted_pipeline = []
        for stage in self.config.pipeline_stages:
            if stage.component == original:
                substituted_pipeline.append(substitute.value)
            else:
                substituted_pipeline.append(stage.component.value)
        
        return substituted_pipeline
    
    def _execute_fallback_pipeline(self, input_data: Dict[str, Any],
                                   correlation_id: str) -> Dict[str, Any]:
        """Execute minimal fallback pipeline"""
        logger.info({
            "msg": "Executing fallback pipeline",
            "correlation_id": correlation_id
        })
        
        # Create minimal config with just essential components
        fallback_config = OrchestratorConfig()
        fallback_config.components = {
            ComponentType.DEEPKE: ComponentConfig(enabled=True, required=True),
            ComponentType.KG_GEN: ComponentConfig(enabled=True, required=True),
        }
        fallback_config.pipeline_stages = [
            PipelineStage(name="extract", component=ComponentType.DEEPKE),
            PipelineStage(name="build_graph", component=ComponentType.KG_GEN, depends_on=["extract"]),
        ]
        
        # Temporarily replace config
        original_config = self.config
        self.config = fallback_config
        self._initialize_components()
        
        try:
            result = super().process(input_data, None)
            result['fallback_executed'] = True
            return result
        finally:
            # Restore original config
            self.config = original_config
            self._initialize_components()
    
    def _execute_parallel_options(self, input_data: Dict[str, Any],
                                  custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple component options in parallel"""
        # For now, this is a placeholder - would need async implementation
        logger.info("Parallel execution strategy - using first available")
        return super().process(input_data, custom_config)
    
    def _execute_decomposed(self, input_data: Dict[str, Any],
                           custom_config: Optional[Dict[str, Any]],
                           correlation_id: str) -> Dict[str, Any]:
        """Decompose large task into smaller chunks"""
        text = input_data.get('text', '')
        
        if len(text) < 10000:
            # Not large enough to decompose
            return super().process(input_data, custom_config)
        
        logger.info({
            "msg": "Decomposing large task into chunks",
            "correlation_id": correlation_id,
            "text_length": len(text)
        })
        
        # Split into chunks (simple sentence-based splitting)
        chunks = self._split_into_chunks(text, max_chunk_size=5000)
        
        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            chunk_input = copy.deepcopy(input_data)
            chunk_input['text'] = chunk
            chunk_input['chunk_index'] = i
            chunk_input['total_chunks'] = len(chunks)
            
            try:
                chunk_result = super().process(chunk_input, custom_config)
                chunk_results.append(chunk_result)
            except Exception as e:
                logger.warning(f"Chunk {i} failed: {e}")
                chunk_results.append({'status': 'failed', 'chunk_index': i})
        
        # Merge results
        merged_result = self._merge_chunk_results(chunk_results)
        merged_result['decomposed'] = True
        merged_result['chunks_processed'] = len(chunks)
        
        return merged_result
    
    def _split_into_chunks(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text into manageable chunks"""
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_size + len(sentence) > max_chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks if chunks else [text]
    
    def _merge_chunk_results(self, chunk_results: List[Dict]) -> Dict[str, Any]:
        """Merge results from multiple chunks"""
        merged = {
            'status': 'success',
            'results': {},
            'execution': {
                'stages_executed': 0,
                'stages_failed': 0
            }
        }
        
        # Merge all results
        all_entities = []
        all_relations = []
        
        for result in chunk_results:
            if result.get('status') == 'success':
                # Extract entities and relations from each chunk
                results_data = result.get('results', {})
                if 'extract_knowledge' in results_data:
                    extract_data = results_data['extract_knowledge']
                    if isinstance(extract_data, dict):
                        all_entities.extend(extract_data.get('entities', []))
                        all_relations.extend(extract_data.get('relations', []))
        
        merged['results']['merged_entities'] = all_entities
        merged['results']['merged_relations'] = all_relations
        merged['results']['chunks_successful'] = sum(1 for r in chunk_results if r.get('status') == 'success')
        
        return merged
    
    def _execute_with_skips(self, input_data: Dict[str, Any],
                           custom_config: Optional[Dict[str, Any]],
                           last_error: Optional[Exception]) -> Dict[str, Any]:
        """Execute pipeline skipping problematic components"""
        failed_component = self._extract_failed_component(last_error)
        
        if not failed_component:
            raise ValueError("Cannot identify component to skip")
        
        logger.info({
            "msg": f"Executing with {failed_component.value} skipped"
        })
        
        # Disable the failed component
        self.config.disable_component(failed_component)
        self._initialize_components()
        
        result = super().process(input_data, custom_config)
        result['components_skipped'] = [failed_component.value]
        
        return result
    
    def _record_healing_action(self, strategy: HealingStrategy, success: bool,
                               result: Optional[Dict], input_data: Dict[str, Any],
                               correlation_id: str, error: Optional[str] = None):
        """Record a healing action for learning"""
        action = HealingAction(
            action_id=f"heal_{len(self.healing_history)}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy=strategy,
            target_component="unknown",  # Would extract from context
            original_config={},
            modified_config={},
            success=success,
            result=result,
            lessons_learned=[f"Strategy {strategy.value} {'succeeded' if success else 'failed'}"]
        )
        
        self.healing_history.append(action)
    
    def _post_execution_learning(self, input_data: Dict[str, Any],
                                 result: Dict[str, Any], correlation_id: str):
        """Learn from execution results"""
        
        # Extract errors from result
        errors = []
        for failed in result.get('failed_stages', []):
            errors.append({
                'type': 'execution_failure',
                'component': failed.get('name'),
                'message': failed.get('error')
            })
        
        # Record experience
        experience = self.learning_engine.record_experience(
            input_data=input_data,
            data_type=input_data.get('data_type', 'unknown'),
            domain=self.config.domain.value,
            pipeline_config=self.config.to_dict(),
            components_used=[c.value for c in self.components.keys()],
            success=result.get('status') == 'success',
            execution_time_ms=result.get('execution', {}).get('duration_ms', 0),
            results=result.get('results', {}),
            errors=errors
        )
        
        logger.debug({
            "msg": "Learning recorded",
            "correlation_id": correlation_id,
            "experience_id": experience.experience_id,
            "lessons": len(experience.lessons_learned)
        })
    
    def get_healing_report(self) -> Dict[str, Any]:
        """Get report on healing activities"""
        return {
            'total_failures': len(self.failure_history),
            'total_healing_actions': len(self.healing_history),
            'successful_healings': len([h for h in self.healing_history if h.success]),
            'healing_success_rate': (
                len([h for h in self.healing_history if h.success]) / 
                max(len(self.healing_history), 1)
            ),
            'strategy_effectiveness': self._analyze_strategy_effectiveness(),
            'learning_summary': self.learning_engine.get_learning_summary()
        }
    
    def _analyze_strategy_effectiveness(self) -> Dict[str, float]:
        """Analyze effectiveness of healing strategies"""
        effectiveness = {}
        
        for strategy in HealingStrategy:
            strategy_actions = [h for h in self.healing_history if h.strategy == strategy]
            if strategy_actions:
                success_rate = len([h for h in strategy_actions if h.success]) / len(strategy_actions)
                effectiveness[strategy.value] = success_rate
        
        return effectiveness


# Factory functions for creating self-healing orchestrators
def create_self_healing_finance_orchestrator(
    learning_storage_path: Optional[str] = None,
    **kwargs
) -> SelfHealingOrchestrator:
    """Create a self-healing finance orchestrator"""
    from . import DomainPresets
    
    config = DomainPresets.finance()
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return SelfHealingOrchestrator(
        config=config,
        enable_self_healing=True,
        learning_storage_path=learning_storage_path
    )


def create_self_healing_chemistry_orchestrator(
    learning_storage_path: Optional[str] = None,
    **kwargs
) -> SelfHealingOrchestrator:
    """Create a self-healing chemistry orchestrator"""
    from . import DomainPresets
    
    config = DomainPresets.chemistry()
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return SelfHealingOrchestrator(
        config=config,
        enable_self_healing=True,
        learning_storage_path=learning_storage_path
    )


def create_self_healing_healthcare_orchestrator(
    learning_storage_path: Optional[str] = None,
    **kwargs
) -> SelfHealingOrchestrator:
    """Create a self-healing healthcare orchestrator"""
    from . import DomainPresets
    
    config = DomainPresets.healthcare()
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return SelfHealingOrchestrator(
        config=config,
        enable_self_healing=True,
        learning_storage_path=learning_storage_path
    )


def create_self_healing_research_orchestrator(
    learning_storage_path: Optional[str] = None,
    **kwargs
) -> SelfHealingOrchestrator:
    """Create a self-healing research orchestrator (all components)"""
    from . import DomainPresets
    
    config = DomainPresets.research()
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return SelfHealingOrchestrator(
        config=config,
        enable_self_healing=True,
        learning_storage_path=learning_storage_path
    )
