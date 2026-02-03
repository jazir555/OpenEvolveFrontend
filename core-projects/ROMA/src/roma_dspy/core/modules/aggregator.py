"""Aggregator module for result synthesis."""

from __future__ import annotations

# **ACTUAL INTEGRATION**: Adaptive MDAP for Aggregator
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
from typing import Union, Any, Optional, Dict, Mapping, Sequence, Mapping as TMapping
from collections import deque

from loguru import logger

from roma_dspy.core.modules.base_module import BaseModule
from roma_dspy.core.signatures.base_models.subtask import SubTask
from roma_dspy.core.signatures.signatures import AggregatorSignature
from roma_dspy.types import PredictionStrategy


class Aggregator(BaseModule):
    """Aggregates results from subtasks."""

    DEFAULT_SIGNATURE = AggregatorSignature
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
            'aggregation_patterns': {},  # goal_hash -> pattern list
            'subtask_count_patterns': {},  # subtask_count -> pattern list
            'aggregation_history': deque(maxlen=500),  # Recent aggregation results
        }
        
        # ICR: Adaptive thresholds based on patterns
        self._adaptive_thresholds: Dict[str, float] = {}

    def forward(
        self,
        original_goal: str,
        subtasks_results: Sequence[SubTask],
        *,
        context: Optional[str] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        dspy_context: Optional[Dict[str, Any]] = None,
        call_params: Optional[Dict[str, Any]] = None,
        store_pattern: bool = True,
        **call_kwargs: Any,
    ):
        """
        Synchronous forward pass with ICR pattern storage.
        
        Args:
            original_goal: Original task goal.
            subtasks_results: List of subtask results to aggregate.
            context: XML string passed to signature's context field (agent instructions).
            tools: Optional tools for this call.
            config: Optional per-call LM overrides.
            dspy_context: Dict passed into dspy.context(...) for this call (DSPy runtime config like callbacks).
            call_params: Extra kwargs to pass to predictor call.
            store_pattern: Whether to store ICR pattern.
            **call_kwargs: Additional kwargs merged into call_params.
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
            result = self._predictor(
                original_goal=original_goal,
                subtasks_results=list(subtasks_results),
                **filtered,
            )
        
        # ICR: Store aggregation pattern
        aggregation_time = time.time() - start_time
        if self.enable_icr and store_pattern:
            self._store_aggregation_pattern(original_goal, subtasks_results, result, aggregation_time, context)
        
        return result

    async def aforward(
        self,
        original_goal: str,
        subtasks_results: Sequence[SubTask],
        *,
        context: Optional[str] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        dspy_context: Optional[Dict[str, Any]] = None,
        call_params: Optional[Dict[str, Any]] = None,
        store_pattern: bool = True,
        **call_kwargs: Any,
    ):
        """
        Async forward pass with ICR pattern storage.
        Aggregate results - returns raw DSPy Prediction with get_lm_usage().
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
            payload = dict(
                original_goal=original_goal, subtasks_results=list(subtasks_results)
            )
            if acall is not None and hasattr(self._predictor, "aforward"):
                result = await acall(**payload, **filtered)
            elif acall is not None:
                result = await acall(**payload, **filtered)
            else:
                result = self._predictor(**payload, **filtered)
        
        # ICR: Store aggregation pattern
        aggregation_time = time.time() - start_time
        if self.enable_icr and store_pattern:
            self._store_aggregation_pattern(original_goal, subtasks_results, result, aggregation_time, context)
        
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
    ) -> "Aggregator":
        return cls(
            prediction_strategy,
            model=model,
            model_config=model_config or None,
            tools=tools,
            enable_icr=enable_icr,
        )
    
    # ---------- ICR Integration Methods ----------
    
    def _store_aggregation_pattern(
        self,
        original_goal: str,
        subtasks_results: Sequence[SubTask],
        result: Any,
        aggregation_time: float,
        context: Optional[str] = None
    ) -> None:
        """
        Store aggregation pattern for ICR learning.
        
        Args:
            original_goal: Original task goal
            subtasks_results: List of subtask results
            result: Aggregation result
            aggregation_time: Time taken for aggregation
            context: Optional context for the aggregation
        """
        if not self.enable_icr:
            return
        
        try:
            # Extract aggregation quality from result
            aggregation_quality = getattr(result, 'quality', None)
            if aggregation_quality is None:
                # Try to infer from common result attributes
                aggregation_quality = getattr(result, 'confidence', 0.8)  # Default to high confidence if not specified
            
            # Create goal hash for pattern storage
            goal_hash = self._get_goal_hash(original_goal)
            subtask_count = len(subtasks_results)
            subtask_count_bucket = self._get_subtask_count_bucket(subtask_count)
            
            # Count successful subtasks
            successful_subtasks = 0
            for subtask in subtasks_results:
                if hasattr(subtask, 'success'):
                    if subtask.success:
                        successful_subtasks += 1
                elif hasattr(subtask, 'result'):
                    successful_subtasks += 1  # Assume success if result exists
            
            pattern = {
                'timestamp': datetime.now().isoformat(),
                'goal_hash': goal_hash,
                'subtask_count': subtask_count,
                'subtask_count_bucket': subtask_count_bucket,
                'successful_subtasks': successful_subtasks,
                'success_rate': successful_subtasks / subtask_count if subtask_count > 0 else 0.0,
                'aggregation_quality': aggregation_quality,
                'aggregation_time': aggregation_time,
                'context_length': len(context) if context else 0,
            }
            
            # Store by goal hash
            if goal_hash not in self.icr_pattern_store['aggregation_patterns']:
                self.icr_pattern_store['aggregation_patterns'][goal_hash] = deque(maxlen=100)
            self.icr_pattern_store['aggregation_patterns'][goal_hash].append(pattern)
            
            # Store by subtask count bucket
            if subtask_count_bucket not in self.icr_pattern_store['subtask_count_patterns']:
                self.icr_pattern_store['subtask_count_patterns'][subtask_count_bucket] = deque(maxlen=200)
            self.icr_pattern_store['subtask_count_patterns'][subtask_count_bucket].append(pattern)
            
            # Store in history
            self.icr_pattern_store['aggregation_history'].append(pattern)
            
            logger.debug(f"ICR pattern stored for aggregation: subtask_count={subtask_count}, quality={aggregation_quality}")
        except Exception as e:
            logger.warning(f"Failed to store ICR pattern: {e}")
    
    def _get_goal_hash(self, goal: str) -> str:
        """Generate a hash for the goal for pattern grouping."""
        # Simple hash based on goal characteristics
        goal_lower = goal.lower().strip()
        # Group by first few words and length bucket
        words = goal_lower.split()[:5]
        length_bucket = len(goal) // 100
        return f"{'_'.join(words)}_{length_bucket}"
    
    def _get_subtask_count_bucket(self, count: int) -> str:
        """Get subtask count bucket for pattern grouping."""
        if count == 1:
            return 'single'
        elif count <= 3:
            return 'few'
        elif count <= 7:
            return 'moderate'
        else:
            return 'many'
    
    def predict_pass_fail(
        self,
        original_goal: str,
        subtasks_results: Sequence[SubTask],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict aggregation outcome based on ICR patterns.
        
        Args:
            original_goal: Original task goal
            subtasks_results: List of subtask results
            context: Optional context
            
        Returns:
            Dictionary with prediction details
        """
        if not self.enable_icr:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'reason': 'ICR disabled'
            }
        
        goal_hash = self._get_goal_hash(original_goal)
        subtask_count = len(subtasks_results)
        subtask_count_bucket = self._get_subtask_count_bucket(subtask_count)
        
        # Get historical patterns for this goal
        goal_patterns = list(self.icr_pattern_store['aggregation_patterns'].get(goal_hash, []))
        count_patterns = list(self.icr_pattern_store['subtask_count_patterns'].get(subtask_count_bucket, []))
        
        # Use count patterns if goal-specific patterns are insufficient
        patterns = goal_patterns if len(goal_patterns) >= 5 else count_patterns
        
        if not patterns:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'reason': 'Insufficient historical data'
            }
        
        # Calculate average quality
        avg_quality = sum(p['aggregation_quality'] for p in patterns) / len(patterns)
        
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
        if avg_quality >= 0.8:
            prediction = 'excellent'
        elif avg_quality >= 0.6:
            prediction = 'good'
        elif avg_quality >= 0.4:
            prediction = 'acceptable'
        else:
            prediction = 'poor'
        
        # Calculate average aggregation time
        avg_aggregation_time = sum(p['aggregation_time'] for p in patterns) / len(patterns)
        
        return {
            'prediction': prediction,
            'quality_score': avg_quality,
            'confidence': confidence,
            'pattern_count': len(patterns),
            'subtask_count_bucket': subtask_count_bucket,
            'average_aggregation_time': avg_aggregation_time,
            'estimated_time': avg_aggregation_time * (1.0 + (0.1 * (1.0 - avg_quality)))  # Add time buffer for lower quality
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
            for patterns in self.icr_pattern_store['aggregation_patterns'].values()
        )
        
        # Calculate overall average quality
        all_patterns = list(self.icr_pattern_store['aggregation_history'])
        overall_quality = sum(p['aggregation_quality'] for p in all_patterns) / len(all_patterns) if all_patterns else 0.0
        
        # Calculate statistics by subtask count bucket
        count_stats = {}
        for count_bucket, patterns in self.icr_pattern_store['subtask_count_patterns'].items():
            patterns_list = list(patterns)
            if patterns_list:
                count_stats[count_bucket] = {
                    'count': len(patterns_list),
                    'avg_quality': sum(p['aggregation_quality'] for p in patterns_list) / len(patterns_list),
                    'avg_aggregation_time': sum(p['aggregation_time'] for p in patterns_list) / len(patterns_list),
                    'avg_subtask_count': sum(p['subtask_count'] for p in patterns_list) / len(patterns_list)
                }
        
        return {
            'icr_enabled': True,
            'total_patterns': total_patterns,
            'total_goals': len(self.icr_pattern_store['aggregation_patterns']),
            'overall_quality': overall_quality,
            'history_size': len(self.icr_pattern_store['aggregation_history']),
            'statistics_by_subtask_count': count_stats,
            'adaptive_thresholds': self._adaptive_thresholds.copy()
        }
    
    def clear_icr_patterns(self) -> None:
        """Clear all stored ICR patterns."""
        if not self.enable_icr:
            return
        
        logger.info("Clearing all ICR patterns for Aggregator")
        
        self.icr_pattern_store = {
            'aggregation_patterns': {},
            'subtask_count_patterns': {},
            'aggregation_history': deque(maxlen=500),
        }
        self._adaptive_thresholds.clear()
    
    def store_icr_pattern(
        self,
        original_goal: str,
        subtasks_results: Sequence[SubTask],
        result: Any,
        aggregation_time: float,
        context: Optional[str] = None
    ) -> None:
        """
        Public method to store ICR pattern (wrapper around _store_aggregation_pattern).
        
        Args:
            original_goal: Original task goal
            subtasks_results: List of subtask results
            result: Aggregation result
            aggregation_time: Time taken for aggregation
            context: Optional context for the aggregation
        """
        self._store_aggregation_pattern(original_goal, subtasks_results, result, aggregation_time, context)
