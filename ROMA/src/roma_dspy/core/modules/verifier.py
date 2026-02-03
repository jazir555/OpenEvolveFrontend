"""Verifier module for result validation."""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Verifier
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


from __future__ import annotations

import dspy
from typing import Union, Any, Optional, Dict, Mapping, Sequence, Mapping as TMapping

from roma_dspy.core.modules.base_module import BaseModule
from roma_dspy.core.signatures.signatures import VerifierSignature
from roma_dspy.types import PredictionStrategy


class Verifier(BaseModule):
    """Verifies task execution results."""

    DEFAULT_SIGNATURE = VerifierSignature
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
            'verification_patterns': {},  # goal_hash -> pattern list
            'goal_type_patterns': {},  # goal_type -> pattern list
            'verification_history': deque(maxlen=500),  # Recent verification results
        }
        
        # ICR: Adaptive thresholds based on patterns
        self._adaptive_thresholds: Dict[str, float] = {}

    def forward(
        self,
        goal: str,
        candidate_output: str,
        *,
        context: Optional[str] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        dspy_context: Optional[Dict[str, Any]] = None,
        call_params: Optional[Dict[str, Any]] = None,
        store_pattern: bool = True,
        **call_kwargs: Any,
    ):
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
                goal=goal, candidate_output=candidate_output, **filtered
            )
        
        # ICR: Store verification pattern
        execution_time = time.time() - start_time
        if self.enable_icr and store_pattern:
            self._store_verification_pattern(goal, candidate_output, result, execution_time, context)
        
        return result

    async def aforward(
        self,
        goal: str,
        candidate_output: str,
        *,
        context: Optional[str] = None,
        tools: Optional[Union[Sequence[Any], TMapping[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        dspy_context: Optional[Dict[str, Any]] = None,
        call_params: Optional[Dict[str, Any]] = None,
        store_pattern: bool = True,
        **call_kwargs: Any,
    ):
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
            payload = dict(goal=goal, candidate_output=candidate_output)
            if acall is not None and hasattr(self._predictor, "aforward"):
                result = await acall(**payload, **filtered)
            elif acall is not None:
                result = await acall(**payload, **filtered)
            else:
                result = self._predictor(**payload, **filtered)
        
        # ICR: Store verification pattern
        execution_time = time.time() - start_time
        if self.enable_icr and store_pattern:
            self._store_verification_pattern(goal, candidate_output, result, execution_time, context)
        
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
    ) -> "Verifier":
        return cls(
            prediction_strategy,
            model=model,
            model_config=model_config or None,
            tools=tools,
            enable_icr=enable_icr,
        )
    
    # ---------- ICR Integration Methods ----------
    
    def _store_verification_pattern(
        self,
        goal: str,
        candidate_output: str,
        result: Any,
        execution_time: float,
        context: Optional[str] = None
    ) -> None:
        """
        Store verification pattern for ICR learning.
        
        Args:
            goal: The goal being verified
            candidate_output: The output being verified
            result: Verification result
            execution_time: Time taken for verification
            context: Optional context for the verification
        """
        if not self.enable_icr:
            return
        
        try:
            # Extract verification decision from result
            verification_passed = getattr(result, 'verification_passed', None)
            if verification_passed is None:
                # Try to infer from common result attributes
                verification_passed = getattr(result, 'passed', True)  # Default to passed if not specified
            
            # Create goal hash for pattern storage
            goal_hash = self._get_goal_hash(goal)
            goal_type = self._classify_goal_type(goal)
            
            pattern = {
                'timestamp': datetime.now().isoformat(),
                'goal_hash': goal_hash,
                'goal_type': goal_type,
                'goal_length': len(goal),
                'output_length': len(candidate_output),
                'verification_passed': verification_passed,
                'execution_time': execution_time,
                'context_length': len(context) if context else 0,
            }
            
            # Store by goal hash
            if goal_hash not in self.icr_pattern_store['verification_patterns']:
                self.icr_pattern_store['verification_patterns'][goal_hash] = deque(maxlen=100)
            self.icr_pattern_store['verification_patterns'][goal_hash].append(pattern)
            
            # Store by goal type
            if goal_type not in self.icr_pattern_store['goal_type_patterns']:
                self.icr_pattern_store['goal_type_patterns'][goal_type] = deque(maxlen=200)
            self.icr_pattern_store['goal_type_patterns'][goal_type].append(pattern)
            
            # Store in history
            self.icr_pattern_store['verification_history'].append(pattern)
            
            logger.debug(f"ICR pattern stored for verification: goal_type={goal_type}, passed={verification_passed}")
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
    
    def _classify_goal_type(self, goal: str) -> str:
        """Classify the goal type for pattern grouping."""
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ['verify', 'check', 'validate', 'confirm']):
            return 'verification'
        elif any(word in goal_lower for word in ['test', 'assert', 'ensure']):
            return 'testing'
        elif any(word in goal_lower for word in ['analyze', 'evaluate', 'assess']):
            return 'analysis'
        elif any(word in goal_lower for word in ['compare', 'match', 'difference']):
            return 'comparison'
        else:
            return 'general'
    
    def predict_pass_fail(
        self,
        goal: str,
        candidate_output: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict verification outcome based on ICR patterns.
        
        Args:
            goal: The goal to be verified
            candidate_output: The output to verify
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
        
        goal_hash = self._get_goal_hash(goal)
        goal_type = self._classify_goal_type(goal)
        
        # Get historical patterns for this goal
        goal_patterns = list(self.icr_pattern_store['verification_patterns'].get(goal_hash, []))
        type_patterns = list(self.icr_pattern_store['goal_type_patterns'].get(goal_type, []))
        
        # Use type patterns if goal-specific patterns are insufficient
        patterns = goal_patterns if len(goal_patterns) >= 5 else type_patterns
        
        if not patterns:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'reason': 'Insufficient historical data'
            }
        
        # Calculate pass rate
        passed = sum(1 for p in patterns if p.get('verification_passed', True))
        pass_rate = passed / len(patterns) if patterns else 0.5
        
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
        if pass_rate >= 0.8:
            prediction = 'pass'
        elif pass_rate >= 0.5:
            prediction = 'likely_pass'
        else:
            prediction = 'likely_fail'
        
        return {
            'prediction': prediction,
            'pass_probability': pass_rate,
            'confidence': confidence,
            'pattern_count': len(patterns),
            'goal_type': goal_type,
            'average_execution_time': sum(p['execution_time'] for p in patterns) / len(patterns)
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
            for patterns in self.icr_pattern_store['verification_patterns'].values()
        )
        
        # Calculate overall pass rate
        all_patterns = list(self.icr_pattern_store['verification_history'])
        passed = sum(1 for p in all_patterns if p.get('verification_passed', True))
        overall_pass_rate = passed / len(all_patterns) if all_patterns else 0.0
        
        # Calculate statistics by goal type
        type_stats = {}
        for goal_type, patterns in self.icr_pattern_store['goal_type_patterns'].items():
            patterns_list = list(patterns)
            if patterns_list:
                type_passed = sum(1 for p in patterns_list if p.get('verification_passed', True))
                type_stats[goal_type] = {
                    'count': len(patterns_list),
                    'pass_rate': type_passed / len(patterns_list),
                    'avg_execution_time': sum(p['execution_time'] for p in patterns_list) / len(patterns_list)
                }
        
        return {
            'icr_enabled': True,
            'total_patterns': total_patterns,
            'total_goals': len(self.icr_pattern_store['verification_patterns']),
            'overall_pass_rate': overall_pass_rate,
            'history_size': len(self.icr_pattern_store['verification_history']),
            'statistics_by_goal_type': type_stats,
            'adaptive_thresholds': self._adaptive_thresholds.copy()
        }
    
    def clear_icr_patterns(self) -> None:
        """Clear all stored ICR patterns."""
        if not self.enable_icr:
            return
        
        logger.info("Clearing all ICR patterns for Verifier")
        
        self.icr_pattern_store = {
            'verification_patterns': {},
            'goal_type_patterns': {},
            'verification_history': deque(maxlen=500),
        }
        self._adaptive_thresholds.clear()
    
    def store_icr_pattern(
        self,
        goal: str,
        candidate_output: str,
        result: Any,
        execution_time: float,
        context: Optional[str] = None
    ) -> None:
        """
        Public method to store ICR pattern (wrapper around _store_verification_pattern).
        
        Args:
            goal: The goal being verified
            candidate_output: The output being verified
            result: Verification result
            execution_time: Time taken for verification
            context: Optional context for the verification
        """
        self._store_verification_pattern(goal, candidate_output, result, execution_time, context)
