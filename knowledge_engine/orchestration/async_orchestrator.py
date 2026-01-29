"""
Async Knowledge Orchestrator

Asynchronous version of the orchestrator with:
- Parallel stage execution
- Non-blocking I/O
- Async healing strategies
- Concurrent component coordination
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
import copy

from .knowledge_orchestrator import (
    KnowledgeOrchestrator, OrchestratorConfig, PipelineStage, ComponentType
)
from .self_healing_orchestrator import SelfHealingOrchestrator, HealingStrategy
from .safe_eval import safe_eval

logger = logging.getLogger(__name__)


class AsyncKnowledgeOrchestrator(KnowledgeOrchestrator):
    """
    Async version of KnowledgeOrchestrator with parallel execution support.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        super().__init__(config)
        self._execution_semaphore = asyncio.Semaphore(self.config.max_workers)
    
    async def process(self, input_data: Dict[str, Any],
                     custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input data asynchronously with optional parallel execution.
        
        Args:
            input_data: Input data to process
            custom_config: Optional runtime configuration
            
        Returns:
            Processing results
        """
        start_time = datetime.now(timezone.utc)
        correlation_id = self.config.correlation_id or f"async_orch_{start_time.timestamp()}"
        
        logger.info({
            "msg": "Starting async orchestrated processing",
            "correlation_id": correlation_id,
            "parallel": self.config.parallel_execution,
            "timestamp": start_time.isoformat()
        })
        
        # Build context
        context = {
            'input': input_data,
            'results': {},
            'correlation_id': correlation_id,
            'start_time': start_time,
            'data_type': input_data.get('data_type', 'unknown'),
            'domain': self.config.domain.value,
        }
        
        # Apply custom config
        config = self.config
        if custom_config:
            config = self._apply_runtime_config(custom_config)
        
        # Execute pipeline
        if config.parallel_execution:
            result = await self._execute_parallel_pipeline(context, config)
        else:
            result = await self._execute_sequential_pipeline(context, config)
        
        # Add metadata
        end_time = datetime.now(timezone.utc)
        result['execution']['started_at'] = start_time.isoformat()
        result['execution']['completed_at'] = end_time.isoformat()
        result['execution']['duration_ms'] = (end_time - start_time).total_seconds() * 1000
        
        return result
    
    async def _execute_sequential_pipeline(self, context: Dict[str, Any],
                                          config: OrchestratorConfig) -> Dict[str, Any]:
        """Execute pipeline stages sequentially"""
        executed_stages = []
        skipped_stages = []
        failed_stages = []
        
        for stage in config.pipeline_stages:
            result = await self._execute_stage_async(stage, context, config)
            
            if result['status'] == 'executed':
                context['results'][stage.name] = result['data']
                executed_stages.append({
                    'name': stage.name,
                    'duration_ms': result.get('duration_ms', 0)
                })
            elif result['status'] == 'skipped':
                skipped_stages.append({
                    'name': stage.name,
                    'reason': result.get('reason', 'unknown')
                })
            else:  # failed
                failed_stages.append({
                    'name': stage.name,
                    'error': result.get('error', 'unknown')
                })
        
        return {
            'status': 'success' if not failed_stages else 'partial',
            'correlation_id': context['correlation_id'],
            'results': context['results'],
            'execution': {
                'stages_executed': len(executed_stages),
                'stages_skipped': len(skipped_stages),
                'stages_failed': len(failed_stages),
            },
            'executed_stages': executed_stages,
            'skipped_stages': skipped_stages,
            'failed_stages': failed_stages,
        }
    
    async def _execute_parallel_pipeline(self, context: Dict[str, Any],
                                        config: OrchestratorConfig) -> Dict[str, Any]:
        """Execute pipeline stages with parallelization where possible"""
        executed_stages = []
        skipped_stages = []
        failed_stages = []
        
        # Group stages by dependency level
        dependency_levels = self._build_dependency_levels(config.pipeline_stages)
        
        for level, stages in enumerate(dependency_levels):
            logger.debug({
                "msg": f"Executing dependency level {level}",
                "stage_count": len(stages)
            })
            
            # Execute stages in this level in parallel
            tasks = []
            for stage in stages:
                task = self._execute_stage_async(stage, context, config)
                tasks.append((stage, task))
            
            # Wait for all tasks in this level
            results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            # Process results
            for (stage, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    failed_stages.append({
                        'name': stage.name,
                        'error': str(result)
                    })
                elif result['status'] == 'executed':
                    context['results'][stage.name] = result['data']
                    executed_stages.append({
                        'name': stage.name,
                        'duration_ms': result.get('duration_ms', 0)
                    })
                elif result['status'] == 'skipped':
                    skipped_stages.append({
                        'name': stage.name,
                        'reason': result.get('reason', 'unknown')
                    })
        
        return {
            'status': 'success' if not failed_stages else 'partial',
            'correlation_id': context['correlation_id'],
            'results': context['results'],
            'execution': {
                'stages_executed': len(executed_stages),
                'stages_skipped': len(skipped_stages),
                'stages_failed': len(failed_stages),
            },
            'executed_stages': executed_stages,
            'skipped_stages': skipped_stages,
            'failed_stages': failed_stages,
        }
    
    def _build_dependency_levels(self, stages: List[PipelineStage]) -> List[List[PipelineStage]]:
        """Build levels of stages based on dependencies"""
        levels = []
        executed = set()
        remaining = list(stages)
        
        while remaining:
            level = []
            still_remaining = []
            
            for stage in remaining:
                # Check if all dependencies are satisfied
                deps_satisfied = all(
                    dep in executed or dep not in [s.name for s in stages]
                    for dep in stage.depends_on
                )
                
                if deps_satisfied:
                    level.append(stage)
                    executed.add(stage.name)
                else:
                    still_remaining.append(stage)
            
            if level:
                levels.append(level)
            
            if len(still_remaining) == len(remaining):
                # Deadlock - add remaining anyway
                levels.append(remaining)
                break
            
            remaining = still_remaining
        
        return levels
    
    async def _execute_stage_async(self, stage: PipelineStage, context: Dict[str, Any],
                                   config: OrchestratorConfig) -> Dict[str, Any]:
        """Execute a single stage asynchronously"""
        stage_start = datetime.now(timezone.utc)
        
        # Check if should execute
        if not stage.should_execute(context):
            return {
                'status': 'skipped',
                'reason': 'condition_not_met_or_disabled'
            }
        
        # Check dependencies
        executed_names = set(context['results'].keys())
        if not all(dep in executed_names for dep in stage.depends_on):
            return {
                'status': 'skipped',
                'reason': 'dependencies_not_met'
            }
        
        # Check component availability
        if stage.component not in self.components:
            if stage.config.required:
                return {
                    'status': 'failed',
                    'error': f"Required component {stage.component.value} not available"
                }
            else:
                return {
                    'status': 'skipped',
                    'reason': 'component_not_available'
                }
        
        # Execute with semaphore for concurrency control
        async with self._execution_semaphore:
            try:
                # Run component handler in thread pool if it's blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None,  # Default executor
                    self._execute_stage_handler,
                    stage,
                    context
                )
                
                duration_ms = (datetime.now(timezone.utc) - stage_start).total_seconds() * 1000
                
                return {
                    'status': 'executed',
                    'data': result,
                    'duration_ms': duration_ms
                }
                
            except Exception as e:
                logger.error({
                    "msg": f"Stage {stage.name} failed",
                    "error": str(e)
                })
                
                return {
                    'status': 'failed',
                    'error': str(e)
                }
    
    def _execute_stage_handler(self, stage: PipelineStage, context: Dict[str, Any]) -> Any:
        """Execute stage handler (runs in thread pool)"""
        # Use parent class implementation
        return self._execute_stage(stage, context)


class AsyncSelfHealingOrchestrator(AsyncKnowledgeOrchestrator, SelfHealingOrchestrator):
    """
    Async self-healing orchestrator with parallel execution and async healing.
    """
    
    async def process(self, input_data: Dict[str, Any],
                     custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process with async self-healing"""
        
        if not self.enable_self_healing:
            return await super().process(input_data, custom_config)
        
        start_time = datetime.now(timezone.utc)
        correlation_id = self.config.correlation_id or f"async_heal_{start_time.timestamp()}"
        
        # Pre-check
        pre_check = self._pre_execution_check(
            input_data.get('data_type', 'unknown'),
            self.config.domain.value,
            input_data
        )
        
        # Try execution with healing
        for attempt in range(self.max_healing_attempts + 1):
            try:
                result = await super().process(input_data, custom_config)
                
                if self._is_result_acceptable(result):
                    # Record success and return
                    self._post_execution_learning(input_data, result, correlation_id)
                    result['healing_metadata'] = {
                        'attempts': attempt,
                        'healed': attempt > 0
                    }
                    return result
                else:
                    # Quality issues - try healing
                    logger.warning(f"Result quality low, attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"Execution failed: {e}")
                
                if attempt < self.max_healing_attempts:
                    # Try healing
                    healed = await self._apply_healing_async(
                        input_data, custom_config, correlation_id, e
                    )
                    if healed:
                        continue
                else:
                    raise
        
        # All attempts exhausted
        return {
            'status': 'failed',
            'error': 'All healing attempts exhausted',
            'correlation_id': correlation_id
        }
    
    async def _apply_healing_async(self, input_data: Dict[str, Any],
                                   custom_config: Optional[Dict[str, Any]],
                                   correlation_id: str,
                                   error: Exception) -> bool:
        """Apply async healing strategies"""
        
        for strategy in self.healing_strategies:
            try:
                if strategy == HealingStrategy.RETRY:
                    await asyncio.sleep(1)  # Non-blocking delay
                    return True  # Will retry in next loop iteration
                
                elif strategy == HealingStrategy.RETRY_WITH_CONFIG:
                    custom_config = self._adjust_config_for_retry(custom_config, error)
                    return True
                
                elif strategy == HealingStrategy.FALLBACK_PIPELINE:
                    result = self._execute_fallback_pipeline(input_data, correlation_id)
                    if result.get('status') in ('success', 'partial'):
                        result['healed'] = True
                        result['healing_strategy'] = 'fallback_pipeline'
                        return True
                
                elif strategy == HealingStrategy.DECOMPOSE_TASK:
                    result = await self._execute_decomposed_async(
                        input_data, custom_config, correlation_id
                    )
                    if result.get('status') in ('success', 'partial'):
                        result['healed'] = True
                        result['healing_strategy'] = 'decompose_task'
                        return True
                
                # Other strategies...
                
            except Exception as heal_error:
                logger.warning(f"Healing strategy {strategy.value} failed: {heal_error}")
                continue
        
        return False
    
    async def _execute_decomposed_async(self, input_data: Dict[str, Any],
                                       custom_config: Optional[Dict[str, Any]],
                                       correlation_id: str) -> Dict[str, Any]:
        """Execute decomposed task asynchronously"""
        text = input_data.get('text', '')
        
        if len(text) < 10000:
            return await super().process(input_data, custom_config)
        
        # Split into chunks
        chunks = self._split_into_chunks(text, max_chunk_size=5000)
        
        # Process chunks concurrently
        tasks = []
        for i, chunk in enumerate(chunks):
            chunk_input = copy.deepcopy(input_data)
            chunk_input['text'] = chunk
            chunk_input['chunk_index'] = i
            chunk_input['total_chunks'] = len(chunks)
            
            task = super().process(chunk_input, custom_config)
            tasks.append(task)
        
        # Wait for all chunks
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        merged = self._merge_chunk_results([
            r if not isinstance(r, Exception) else {'status': 'failed'}
            for r in chunk_results
        ])
        
        merged['decomposed'] = True
        merged['chunks_processed'] = len(chunks)
        
        return merged


# Factory functions
async def create_async_finance_orchestrator(**kwargs) -> AsyncKnowledgeOrchestrator:
    """Create async finance orchestrator"""
    from . import DomainPresets
    config = DomainPresets.finance()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return AsyncKnowledgeOrchestrator(config)


async def create_async_chemistry_orchestrator(**kwargs) -> AsyncKnowledgeOrchestrator:
    """Create async chemistry orchestrator"""
    from . import DomainPresets
    config = DomainPresets.chemistry()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return AsyncKnowledgeOrchestrator(config)


async def create_async_healthcare_orchestrator(**kwargs) -> AsyncKnowledgeOrchestrator:
    """Create async healthcare orchestrator"""
    from . import DomainPresets
    config = DomainPresets.healthcare()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return AsyncKnowledgeOrchestrator(config)


async def create_async_research_orchestrator(**kwargs) -> AsyncKnowledgeOrchestrator:
    """Create async research orchestrator"""
    from . import DomainPresets
    config = DomainPresets.research()
    for key, value in kwargs.items():
        setattr(config, key, value)
    return AsyncKnowledgeOrchestrator(config)


async def create_async_self_healing_finance_orchestrator(
    learning_storage_path: Optional[str] = None,
    **kwargs
) -> AsyncSelfHealingOrchestrator:
    """Create async self-healing finance orchestrator"""
    from . import DomainPresets
    config = DomainPresets.finance()
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return AsyncSelfHealingOrchestrator(
        config=config,
        enable_self_healing=True,
        learning_storage_path=learning_storage_path,
        **kwargs
    )


async def create_async_self_healing_chemistry_orchestrator(
    learning_storage_path: Optional[str] = None,
    **kwargs
) -> AsyncSelfHealingOrchestrator:
    """Create async self-healing chemistry orchestrator"""
    from . import DomainPresets
    config = DomainPresets.chemistry()
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return AsyncSelfHealingOrchestrator(
        config=config,
        enable_self_healing=True,
        learning_storage_path=learning_storage_path,
        **kwargs
    )


async def create_async_self_healing_healthcare_orchestrator(
    learning_storage_path: Optional[str] = None,
    **kwargs
) -> AsyncSelfHealingOrchestrator:
    """Create async self-healing healthcare orchestrator"""
    from . import DomainPresets
    config = DomainPresets.healthcare()
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return AsyncSelfHealingOrchestrator(
        config=config,
        enable_self_healing=True,
        learning_storage_path=learning_storage_path,
        **kwargs
    )


async def create_async_self_healing_research_orchestrator(
    learning_storage_path: Optional[str] = None,
    **kwargs
) -> AsyncSelfHealingOrchestrator:
    """Create async self-healing research orchestrator"""
    from . import DomainPresets
    config = DomainPresets.research()
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return AsyncSelfHealingOrchestrator(
        config=config,
        enable_self_healing=True,
        learning_storage_path=learning_storage_path,
        **kwargs
    )
