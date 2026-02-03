"""Executor module for task execution and tool routing."""

from __future__ import annotations

# **ACTUAL INTEGRATION**: Adaptive MDAP for Executor
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

import dspy
import time
from datetime import datetime
from typing import (
    Union,
    Any,
    Optional,
    Dict,
    Mapping,
    Sequence,
    Mapping as TMapping,
    List,
)
from collections import deque

from loguru import logger

from roma_dspy.core.signatures.signatures import ExecutorSignature
from roma_dspy.types import PredictionStrategy
from roma_dspy.core.modules.base_module import BaseModule


class Executor(BaseModule):
    """Executes atomic tasks and routes to tools."""

    DEFAULT_SIGNATURE = ExecutorSignature
    MANDATORY_TOOLKIT_NAMES = ["ArtifactToolkit"]

    def __init__(
        self,
        prediction_strategy: Union[
            PredictionStrategy, str
        ] = PredictionStrategy.CHAIN_OF_THOUGHT,
        *,
        signature: Any = None,
        config: Optional[Any] = None,
        lm: Optional[dspy.LM] = None,
        model: Optional[str] = None,
        model_config: Optional[Mapping[str, Any]] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        enable_icr: bool = True,
        icr_pattern_store: Optional[Dict[str, Any]] = None,
        **strategy_kwargs: Any,
    ) -> None:
        super().__init__(
            signature=signature if signature is not None else self.DEFAULT_SIGNATURE,
            config=config,
            prediction_strategy=prediction_strategy,
            lm=lm,
            model=model,
            model_config=model_config,
            tools=tools,
            **strategy_kwargs,
        )
        
        # ICR Integration: Pattern storage and learning
        self.enable_icr = enable_icr
        self.icr_pattern_store = icr_pattern_store or {
            'execution_patterns': {},  # task_hash -> pattern list
            'task_type_patterns': {},  # task_type -> pattern list
            'tool_usage_patterns': {},  # tool_name -> pattern list
            'execution_history': deque(maxlen=500),  # Recent execution results
        }
        
        # ICR: Adaptive thresholds based on patterns
        self._adaptive_thresholds: Dict[str, float] = {}

    def forward(
        self,
        goal: str,
        *,
        context: Optional[str] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        demos: Optional[List[Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        dspy_context: Optional[Dict[str, Any]] = None,
        call_params: Optional[Dict[str, Any]] = None,
        store_pattern: bool = True,
        **call_kwargs: Any,
    ):
        """
        Synchronous forward pass with ICR pattern storage.
        """
        start_time = time.time()
        
        runtime_tools = self._merge_tools(self._tools, tools)

        ctx = dict(self._context_defaults)
        if dspy_context:
            ctx.update(dspy_context)
        ctx.setdefault("lm", self._lm)

        extra = dict(call_params or {})
        if call_kwargs:
            extra.update(call_kwargs)
        if config is not None:
            extra["config"] = config
        if runtime_tools:
            extra["tools"] = runtime_tools
        if context is not None:
            extra["context"] = context

        target_method = getattr(self._predictor, "forward", None)
        filtered = self._filter_kwargs(target_method, extra)

        with dspy.context(**ctx):
            result = self._predictor(goal=goal, **filtered)
        
        # ICR: Store execution pattern
        execution_time = time.time() - start_time
        if self.enable_icr and store_pattern:
            self._store_execution_pattern(goal, result, execution_time, context, runtime_tools)
        
        return result

    async def aforward(
        self,
        goal: str,
        *,
        context: Optional[str] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        demos: Optional[List[Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        dspy_context: Optional[Dict[str, Any]] = None,
        call_params: Optional[Dict[str, Any]] = None,
        store_pattern: bool = True,
        **call_kwargs: Any,
    ):
        """
        Async forward pass with ICR pattern storage.
        """
        start_time = time.time()
        
        # BUG FIX: Get execution-scoped tools from ExecutionContext (for toolkit-based agents)
        execution_tools = await self._get_execution_tools()
        runtime_tools = self._merge_tools(execution_tools, tools)

        # Update predictor's internal tools (for ReAct/CodeAct that don't accept tools as parameters)
        self._update_predictor_tools(runtime_tools)

        ctx = dict(self._context_defaults)
        if dspy_context:
            ctx.update(dspy_context)
        ctx.setdefault("lm", self._lm)

        extra = dict(call_params or {})
        if call_kwargs:
            extra.update(call_kwargs)
        if config is not None:
            extra["config"] = config
        if runtime_tools:
            extra["tools"] = runtime_tools
        if context is not None:
            extra["context"] = context

        method_for_filter = getattr(self._predictor, "aforward", None) or getattr(
            self._predictor, "forward", None
        )
        filtered = self._filter_kwargs(method_for_filter, extra)

        # Return raw DSPy prediction (has get_lm_usage() method)
        with dspy.context(**ctx):
            acall = getattr(self._predictor, "acall", None)
            if acall is not None and hasattr(self._predictor, "aforward"):
                result = await acall(goal=goal, **filtered)
            elif acall is not None:
                result = await acall(goal=goal, **filtered)
            else:
                result = self._predictor(goal=goal, **filtered)
        
        # ICR: Store execution pattern
        execution_time = time.time() - start_time
        if self.enable_icr and store_pattern:
            self._store_execution_pattern(goal, result, execution_time, context, runtime_tools)
        
        return result

    @classmethod
    def from_provider(
        cls,
        prediction_strategy: Union[
            PredictionStrategy, str
        ] = PredictionStrategy.CHAIN_OF_THOUGHT,
        *,
        model: str,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        enable_icr: bool = True,
        **model_config: Any,
    ) -> "Executor":
        return cls(
            prediction_strategy,
            model=model,
            model_config=model_config or None,
            tools=tools,
            enable_icr=enable_icr,
        )
    
    # ---------- ICR Integration Methods ----------
    
    def _store_execution_pattern(
        self,
        goal: str,
        result: Any,
        execution_time: float,
        context: Optional[str] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None
    ) -> None:
        """
        Store execution pattern for ICR learning.
        
        Args:
            goal: The goal being executed
            result: Execution result
            execution_time: Time taken for execution
            context: Optional context for the execution
            tools: Tools used for execution
        """
        if not self.enable_icr:
            return
        
        try:
            # Extract execution success from result
            execution_success = getattr(result, 'success', None)
            if execution_success is None:
                # Try to infer from common result attributes
                execution_success = getattr(result, 'passed', True)  # Default to success if not specified
            
            # Create task hash for pattern storage
            task_hash = self._get_task_hash(goal)
            task_type = self._classify_task_type(goal)
            
            # Extract tool usage
            tool_names = []
            if tools:
                if isinstance(tools, dict):
                    tool_names = list(tools.keys())
                else:
                    tool_names = [getattr(t, 'name', str(t)) for t in tools]
            
            pattern = {
                'timestamp': datetime.now().isoformat(),
                'task_hash': task_hash,
                'task_type': task_type,
                'goal_length': len(goal),
                'execution_success': execution_success,
                'execution_time': execution_time,
                'context_length': len(context) if context else 0,
                'tools_used': tool_names,
                'tool_count': len(tool_names),
            }
            
            # Store by task hash
            if task_hash not in self.icr_pattern_store['execution_patterns']:
                self.icr_pattern_store['execution_patterns'][task_hash] = deque(maxlen=100)
            self.icr_pattern_store['execution_patterns'][task_hash].append(pattern)
            
            # Store by task type
            if task_type not in self.icr_pattern_store['task_type_patterns']:
                self.icr_pattern_store['task_type_patterns'][task_type] = deque(maxlen=200)
            self.icr_pattern_store['task_type_patterns'][task_type].append(pattern)
            
            # Store by tool usage
            for tool_name in tool_names:
                if tool_name not in self.icr_pattern_store['tool_usage_patterns']:
                    self.icr_pattern_store['tool_usage_patterns'][tool_name] = deque(maxlen=200)
                self.icr_pattern_store['tool_usage_patterns'][tool_name].append(pattern)
            
            # Store in history
            self.icr_pattern_store['execution_history'].append(pattern)
            
            logger.debug(f"ICR pattern stored for execution: task_type={task_type}, success={execution_success}")
        except Exception as e:
            logger.warning(f"Failed to store ICR pattern: {e}")
    
    def _get_task_hash(self, goal: str) -> str:
        """Generate a hash for the task for pattern grouping."""
        # Simple hash based on task characteristics
        goal_lower = goal.lower().strip()
        # Group by first few words and length bucket
        words = goal_lower.split()[:5]
        length_bucket = len(goal) // 100
        return f"{'_'.join(words)}_{length_bucket}"
    
    def _classify_task_type(self, goal: str) -> str:
        """Classify task type for pattern grouping."""
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ['execute', 'run', 'perform', 'do']):
            return 'execution'
        elif any(word in goal_lower for word in ['calculate', 'compute', 'solve', 'find']):
            return 'calculation'
        elif any(word in goal_lower for word in ['search', 'find', 'locate', 'look up']):
            return 'search'
        elif any(word in goal_lower for word in ['create', 'make', 'generate', 'build']):
            return 'creation'
        elif any(word in goal_lower for word in ['analyze', 'examine', 'study', 'investigate']):
            return 'analysis'
        else:
            return 'general'
    
    def predict_pass_fail(
        self,
        goal: str,
        context: Optional[str] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Predict execution outcome based on ICR patterns.
        
        Args:
            goal: The goal to be executed
            context: Optional context
            tools: Tools to be used
            
        Returns:
            Dictionary with prediction details
        """
        if not self.enable_icr:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'reason': 'ICR disabled'
            }
        
        task_hash = self._get_task_hash(goal)
        task_type = self._classify_task_type(goal)
        
        # Get historical patterns for this task
        task_patterns = list(self.icr_pattern_store['execution_patterns'].get(task_hash, []))
        type_patterns = list(self.icr_pattern_store['task_type_patterns'].get(task_type, []))
        
        # Use type patterns if task-specific patterns are insufficient
        patterns = task_patterns if len(task_patterns) >= 5 else type_patterns
        
        if not patterns:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'reason': 'Insufficient historical data'
            }
        
        # Calculate success rate
        succeeded = sum(1 for p in patterns if p.get('execution_success', True))
        success_rate = succeeded / len(patterns) if patterns else 0.5
        
        # Determine confidence based on pattern count
        if len(patterns) >= 20:
            confidence = 0.9
        elif len(patterns) >= 10:
            confidence = 0.75
        elif len(patterns) >= 5:
            confidence = 0.5
        else:
            confidence = 0.25
        
        # Predict outcome
        if success_rate >= 0.8:
            prediction = 'success'
        elif success_rate >= 0.5:
            prediction = 'likely_success'
        else:
            prediction = 'likely_failure'
        
        # Calculate average execution time
        avg_execution_time = sum(p['execution_time'] for p in patterns) / len(patterns)
        
        return {
            'prediction': prediction,
            'success_probability': success_rate,
            'confidence': confidence,
            'pattern_count': len(patterns),
            'task_type': task_type,
            'average_execution_time': avg_execution_time,
            'estimated_time': avg_execution_time * (1.0 + (0.1 * (1.0 - success_rate)))  # Add time buffer for lower success rates
        }
    
    def get_icr_statistics(self) -> Dict[str, Any]:
        """
        Get ICR-related statistics.
        
        Returns:
            Dictionary with ICR statistics
        """
        if not self.enable_icr:
            return {'icr_enabled': False}
        
        # Calculate total patterns
        total_patterns = sum(
            len(patterns)
            for patterns in self.icr_pattern_store['execution_patterns'].values()
        )
        
        # Calculate overall success rate
        all_patterns = list(self.icr_pattern_store['execution_history'])
        succeeded = sum(1 for p in all_patterns if p.get('execution_success', True))
        overall_success_rate = succeeded / len(all_patterns) if all_patterns else 0.0
        
        # Calculate statistics by task type
        type_stats = {}
        for task_type, patterns in self.icr_pattern_store['task_type_patterns'].items():
            patterns_list = list(patterns)
            if patterns_list:
                type_succeeded = sum(1 for p in patterns_list if p.get('execution_success', True))
                type_stats[task_type] = {
                    'count': len(patterns_list),
                    'success_rate': type_succeeded / len(patterns_list),
                    'avg_execution_time': sum(p['execution_time'] for p in patterns_list) / len(patterns_list)
                }
        
        # Calculate statistics by tool usage
        tool_stats = {}
        for tool_name, patterns in self.icr_pattern_store['tool_usage_patterns'].items():
            patterns_list = list(patterns)
            if patterns_list:
                tool_succeeded = sum(1 for p in patterns_list if p.get('execution_success', True))
                tool_stats[tool_name] = {
                    'count': len(patterns_list),
                    'success_rate': tool_succeeded / len(patterns_list),
                    'avg_execution_time': sum(p['execution_time'] for p in patterns_list) / len(patterns_list)
                }
        
        return {
            'icr_enabled': True,
            'total_patterns': total_patterns,
            'total_tasks': len(self.icr_pattern_store['execution_patterns']),
            'overall_success_rate': overall_success_rate,
            'history_size': len(self.icr_pattern_store['execution_history']),
            'statistics_by_task_type': type_stats,
            'statistics_by_tool': tool_stats,
            'adaptive_thresholds': self._adaptive_thresholds.copy()
        }
    
    def clear_icr_patterns(self) -> None:
        """Clear all stored ICR patterns."""
        if not self.enable_icr:
            return
        
        logger.info("Clearing all ICR patterns for Executor")
        
        self.icr_pattern_store = {
            'execution_patterns': {},
            'task_type_patterns': {},
            'tool_usage_patterns': {},
            'execution_history': deque(maxlen=500),
        }
        self._adaptive_thresholds.clear()
    
    def store_icr_pattern(
        self,
        goal: str,
        result: Any,
        execution_time: float,
        context: Optional[str] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None
    ) -> None:
        """
        Public method to store ICR pattern (wrapper around _store_execution_pattern).
        
        Args:
            goal: The goal being executed
            result: Execution result
            execution_time: Time taken for execution
            context: Optional context for the execution
            tools: Tools used for execution
        """
        self._store_execution_pattern(goal, result, execution_time, context, tools)
