"""Planner module for execution planning."""

from __future__ import annotations

import dspy
import time
from datetime import datetime
from typing import Union, Any, Optional, Mapping, Sequence, Mapping as TMapping
from collections import deque

from loguru import logger

from roma_dspy.core.modules.base_module import BaseModule
from roma_dspy.core.signatures.signatures import PlannerSignature
from roma_dspy.types import PredictionStrategy


class Planner(BaseModule):
    """Plans task execution strategy."""

    DEFAULT_SIGNATURE = PlannerSignature
    MANDATORY_TOOLKIT_NAMES = []

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
            'planning_patterns': {},  # task_hash -> pattern list
            'task_complexity_patterns': {},  # complexity_level -> pattern list
            'planning_history': deque(maxlen=500),  # Recent planning results
        }
        
        # ICR: Adaptive thresholds based on patterns
        self._adaptive_thresholds: Dict[str, float] = {}

    def forward(
        self,
        goal: str,
        *,
        context: Optional[str] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        demos: Optional[Sequence[Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        dspy_context: Optional[Mapping[str, Any]] = None,
        call_params: Optional[Mapping[str, Any]] = None,
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
        
        # ICR: Store planning pattern
        planning_time = time.time() - start_time
        if self.enable_icr and store_pattern:
            self._store_planning_pattern(goal, result, planning_time, context)
        
        return result

    async def aforward(
        self,
        goal: str,
        *,
        context: Optional[str] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        demos: Optional[Sequence[Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        dspy_context: Optional[Mapping[str, Any]] = None,
        call_params: Optional[Mapping[str, Any]] = None,
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

        with dspy.context(**ctx):
            acall = getattr(self._predictor, "acall", None)
            if acall is not None and hasattr(self._predictor, "aforward"):
                result = await acall(goal=goal, **filtered)
            elif acall is not None:
                result = await acall(goal=goal, **filtered)
            else:
                result = self._predictor(goal=goal, **filtered)
        
        # ICR: Store planning pattern
        planning_time = time.time() - start_time
        if self.enable_icr and store_pattern:
            self._store_planning_pattern(goal, result, planning_time, context)
        
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
    ) -> "Planner":
        return cls(
            prediction_strategy,
            model=model,
            model_config=model_config or None,
            tools=tools,
            enable_icr=enable_icr,
        )
    
    # ---------- ICR Integration Methods ----------
    
    def _store_planning_pattern(
        self,
        goal: str,
        result: Any,
        planning_time: float,
        context: Optional[str] = None
    ) -> None:
        """
        Store planning pattern for ICR learning.
        
        Args:
            goal: The goal being planned
            result: Planning result
            planning_time: Time taken for planning
            context: Optional context for the planning
        """
        if not self.enable_icr:
            return
        
        try:
            # Extract plan quality from result
            plan_quality = getattr(result, 'quality', None)
            if plan_quality is None:
                # Try to infer from common result attributes
                plan_quality = getattr(result, 'confidence', 0.8)  # Default to high confidence if not specified
            
            # Create task hash for pattern storage
            task_hash = self._get_task_hash(goal)
            complexity_level = self._estimate_complexity(goal)
            
            pattern = {
                'timestamp': datetime.now().isoformat(),
                'task_hash': task_hash,
                'complexity_level': complexity_level,
                'goal_length': len(goal),
                'plan_quality': plan_quality,
                'planning_time': planning_time,
                'context_length': len(context) if context else 0,
            }
            
            # Store by task hash
            if task_hash not in self.icr_pattern_store['planning_patterns']:
                self.icr_pattern_store['planning_patterns'][task_hash] = deque(maxlen=100)
            self.icr_pattern_store['planning_patterns'][task_hash].append(pattern)
            
            # Store by complexity level
            if complexity_level not in self.icr_pattern_store['task_complexity_patterns']:
                self.icr_pattern_store['task_complexity_patterns'][complexity_level] = deque(maxlen=200)
            self.icr_pattern_store['task_complexity_patterns'][complexity_level].append(pattern)
            
            # Store in history
            self.icr_pattern_store['planning_history'].append(pattern)
            
            logger.debug(f"ICR pattern stored for planning: complexity={complexity_level}, quality={plan_quality}")
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
    
    def _estimate_complexity(self, goal: str) -> str:
        """Estimate task complexity for pattern grouping."""
        goal_lower = goal.lower()
        
        # Count complexity indicators
        complexity_indicators = 0
        if any(word in goal_lower for word in ['complex', 'difficult', 'challenging', 'intricate']):
            complexity_indicators += 2
        if any(word in goal_lower for word in ['multiple', 'several', 'various', 'many']):
            complexity_indicators += 1
        if any(word in goal_lower for word in ['integrate', 'combine', 'merge', 'synthesize']):
            complexity_indicators += 1
        if any(word in goal_lower for word in ['optimize', 'improve', 'enhance', 'refine']):
            complexity_indicators += 1
        if len(goal) > 500:
            complexity_indicators += 1
        
        if complexity_indicators >= 4:
            return 'high'
        elif complexity_indicators >= 2:
            return 'medium'
        else:
            return 'low'
    
    def predict_pass_fail(
        self,
        goal: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict planning outcome based on ICR patterns.
        
        Args:
            goal: The goal to be planned
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
        
        task_hash = self._get_task_hash(goal)
        complexity_level = self._estimate_complexity(goal)
        
        # Get historical patterns for this task
        task_patterns = list(self.icr_pattern_store['planning_patterns'].get(task_hash, []))
        complexity_patterns = list(self.icr_pattern_store['task_complexity_patterns'].get(complexity_level, []))
        
        # Use complexity patterns if task-specific patterns are insufficient
        patterns = task_patterns if len(task_patterns) >= 5 else complexity_patterns
        
        if not patterns:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'reason': 'Insufficient historical data'
            }
        
        # Calculate average plan quality
        avg_quality = sum(p['plan_quality'] for p in patterns) / len(patterns)
        
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
        
        # Calculate average planning time
        avg_planning_time = sum(p['planning_time'] for p in patterns) / len(patterns)
        
        return {
            'prediction': prediction,
            'quality_score': avg_quality,
            'confidence': confidence,
            'pattern_count': len(patterns),
            'complexity_level': complexity_level,
            'average_planning_time': avg_planning_time,
            'estimated_time': avg_planning_time * (1.0 + (0.2 * (1.0 - avg_quality)))  # Add time buffer for lower quality
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
            for patterns in self.icr_pattern_store['planning_patterns'].values()
        )
        
        # Calculate overall average quality
        all_patterns = list(self.icr_pattern_store['planning_history'])
        overall_quality = sum(p['plan_quality'] for p in all_patterns) / len(all_patterns) if all_patterns else 0.0
        
        # Calculate statistics by complexity level
        complexity_stats = {}
        for complexity, patterns in self.icr_pattern_store['task_complexity_patterns'].items():
            patterns_list = list(patterns)
            if patterns_list:
                complexity_stats[complexity] = {
                    'count': len(patterns_list),
                    'avg_quality': sum(p['plan_quality'] for p in patterns_list) / len(patterns_list),
                    'avg_planning_time': sum(p['planning_time'] for p in patterns_list) / len(patterns_list)
                }
        
        return {
            'icr_enabled': True,
            'total_patterns': total_patterns,
            'total_tasks': len(self.icr_pattern_store['planning_patterns']),
            'overall_quality': overall_quality,
            'history_size': len(self.icr_pattern_store['planning_history']),
            'statistics_by_complexity': complexity_stats,
            'adaptive_thresholds': self._adaptive_thresholds.copy()
        }
    
    def clear_icr_patterns(self) -> None:
        """Clear all stored ICR patterns."""
        if not self.enable_icr:
            return
        
        logger.info("Clearing all ICR patterns for Planner")
        
        self.icr_pattern_store = {
            'planning_patterns': {},
            'task_complexity_patterns': {},
            'planning_history': deque(maxlen=500),
        }
        self._adaptive_thresholds.clear()
    
    def store_icr_pattern(
        self,
        goal: str,
        result: Any,
        planning_time: float,
        context: Optional[str] = None
    ) -> None:
        """
        Public method to store ICR pattern (wrapper around _store_planning_pattern).
        
        Args:
            goal: The goal being planned
            result: Planning result
            planning_time: Time taken for planning
            context: Optional context for the planning
        """
        self._store_planning_pattern(goal, result, planning_time, context)
