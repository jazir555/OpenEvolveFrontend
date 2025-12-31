"""
Multi-Round Testing System for OpenEvolve
Implements iterative improvement through multiple testing rounds with adaptive strategies
"""

import time
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics

# Import with fallback for standalone operation
try:
    from error_handler import with_error_handling, ErrorCategory, ErrorSeverity
except ImportError:
    def with_error_handling(category=None, severity=None, fallback_value=None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in {func.__name__}: {e}")
                    return fallback_value
            return wrapper
        return decorator

    class ErrorCategory:
        PROCESSING = "processing"

    class ErrorSeverity:
        MEDIUM = "medium"


class RoundStrategy(Enum):
    """Strategies for multi-round testing"""
    PROGRESSIVE = "progressive"  # Gradually increase difficulty/complexity
    ADAPTIVE = "adaptive"  # Adapt based on previous round performance
    FIXED = "fixed"  # Same parameters for all rounds
    RANDOM = "random"  # Random variations each round
    CONVERGENT = "convergent"  # Converge towards optimal parameters


class RoundStoppingCriteria(Enum):
    """Criteria for stopping multi-round testing"""
    MAX_ROUNDS = "max_rounds"
    QUALITY_THRESHOLD = "quality_threshold"
    IMPROVEMENT_PLATEAU = "improvement_plateau"
    TIME_LIMIT = "time_limit"
    CONVERGENCE = "convergence"


@dataclass
class RoundConfiguration:
    """Configuration for a single round"""
    round_number: int
    parameters: Dict[str, Any]
    strategy_adjustments: Dict[str, Any] = field(default_factory=dict)
    expected_improvement: float = 0.0
    timeout_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundResult:
    """Result from a single round of testing"""
    round_number: int
    configuration: RoundConfiguration
    content_before: str
    content_after: str
    quality_score: float
    improvement_score: float
    execution_time: float
    metrics: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiRoundResult:
    """Result from complete multi-round testing"""
    original_content: str
    final_content: str
    total_rounds: int
    successful_rounds: int
    total_execution_time: float
    overall_improvement: float
    best_round: Optional[RoundResult]
    round_results: List[RoundResult]
    convergence_data: Dict[str, Any]
    stopping_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiRoundTester:
    """Main multi-round testing system"""
    
    def __init__(self):
        self.round_history: List[MultiRoundResult] = []
        self.adaptive_parameters: Dict[str, Any] = {}
        
    @with_error_handling(
        category="processing",
        severity="medium",
        fallback_value=None
    )
    def run_multi_round_test(
        self,
        content: str,
        test_function: Callable,
        max_rounds: int = 5,
        strategy: RoundStrategy = RoundStrategy.ADAPTIVE,
        stopping_criteria: List[RoundStoppingCriteria] = None,
        base_parameters: Dict[str, Any] = None,
        **kwargs
    ) -> MultiRoundResult:
        """
        Run multi-round testing with adaptive improvement.
        
        Args:
            content: Initial content to test/improve
            test_function: Function to run each round (should accept content and parameters)
            max_rounds: Maximum number of rounds
            strategy: Strategy for parameter adaptation
            stopping_criteria: Criteria for early stopping
            base_parameters: Base parameters for testing
            **kwargs: Additional parameters
            
        Returns:
            MultiRoundResult: Complete results from all rounds
        """
        start_time = time.time()
        
        if stopping_criteria is None:
            stopping_criteria = [RoundStoppingCriteria.MAX_ROUNDS, RoundStoppingCriteria.IMPROVEMENT_PLATEAU]
        
        if base_parameters is None:
            base_parameters = {}
        
        # Initialize tracking variables
        round_results = []
        current_content = content
        best_result = None
        stopping_reason = "max_rounds_reached"
        convergence_data = {
            'quality_scores': [],
            'improvement_scores': [],
            'parameter_evolution': []
        }
        
        # Run rounds
        for round_num in range(1, max_rounds + 1):
            # Generate round configuration
            round_config = self._generate_round_configuration(
                round_num, strategy, base_parameters, round_results
            )
            
            # Execute round
            round_result = self._execute_round(
                current_content, test_function, round_config
            )
            
            # Update tracking
            round_results.append(round_result)
            convergence_data['quality_scores'].append(round_result.quality_score)
            convergence_data['improvement_scores'].append(round_result.improvement_score)
            convergence_data['parameter_evolution'].append(round_config.parameters.copy())
            
            # Update best result
            if best_result is None or round_result.quality_score > best_result.quality_score:
                best_result = round_result
            
            # Update content if round was successful
            if round_result.success and round_result.improvement_score > 0:
                current_content = round_result.content_after
            
            # Check stopping criteria
            should_stop, stop_reason = self._check_stopping_criteria(
                stopping_criteria, round_results, round_num, max_rounds
            )
            
            if should_stop:
                stopping_reason = stop_reason
                break
        
        # Calculate overall metrics
        successful_rounds = sum(1 for r in round_results if r.success)
        total_improvement = self._calculate_total_improvement(content, current_content, round_results)
        
        # Create final result
        result = MultiRoundResult(
            original_content=content,
            final_content=current_content,
            total_rounds=len(round_results),
            successful_rounds=successful_rounds,
            total_execution_time=time.time() - start_time,
            overall_improvement=total_improvement,
            best_round=best_result,
            round_results=round_results,
            convergence_data=convergence_data,
            stopping_reason=stopping_reason,
            metadata={
                "strategy_used": strategy.value,
                "stopping_criteria": [c.value for c in stopping_criteria],
                "base_parameters": base_parameters
            }
        )
        
        # Store in history
        self.round_history.append(result)
        
        # Update adaptive parameters for future runs
        self._update_adaptive_parameters(result)
        
        return result
    
    def _generate_round_configuration(
        self,
        round_num: int,
        strategy: RoundStrategy,
        base_parameters: Dict[str, Any],
        previous_results: List[RoundResult]
    ) -> RoundConfiguration:
        """Generate configuration for a specific round"""
        
        if strategy == RoundStrategy.PROGRESSIVE:
            parameters = self._generate_progressive_parameters(round_num, base_parameters)
        elif strategy == RoundStrategy.ADAPTIVE:
            parameters = self._generate_adaptive_parameters(round_num, base_parameters, previous_results)
        elif strategy == RoundStrategy.FIXED:
            parameters = base_parameters.copy()
        elif strategy == RoundStrategy.RANDOM:
            parameters = self._generate_random_parameters(base_parameters)
        else:  # CONVERGENT
            parameters = self._generate_convergent_parameters(round_num, base_parameters, previous_results)
        
        return RoundConfiguration(
            round_number=round_num,
            parameters=parameters,
            strategy_adjustments=self._calculate_strategy_adjustments(strategy, round_num, previous_results),
            expected_improvement=self._estimate_expected_improvement(round_num, previous_results),
            metadata={
                "strategy": strategy.value,
                "generation_time": time.time()
            }
        )
    
    def _generate_progressive_parameters(self, round_num: int, base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters that progressively increase in complexity/difficulty"""
        parameters = base_parameters.copy()
        
        # Progressive scaling factors
        complexity_factor = 1.0 + (round_num - 1) * 0.2  # Increase by 20% each round
        
        # Apply progressive scaling to relevant parameters
        if 'iterations' in parameters:
            parameters['iterations'] = int(parameters.get('iterations', 10) * complexity_factor)
        
        if 'population_size' in parameters:
            parameters['population_size'] = int(parameters.get('population_size', 20) * complexity_factor)
        
        if 'mutation_rate' in parameters:
            # Decrease mutation rate as rounds progress (more focused search)
            parameters['mutation_rate'] = parameters.get('mutation_rate', 0.1) / complexity_factor
        
        if 'selection_pressure' in parameters:
            # Increase selection pressure
            parameters['selection_pressure'] = min(parameters.get('selection_pressure', 0.5) * complexity_factor, 1.0)
        
        return parameters
    
    def _generate_adaptive_parameters(
        self, round_num: int, base_parameters: Dict[str, Any], previous_results: List[RoundResult]
    ) -> Dict[str, Any]:
        """Generate parameters that adapt based on previous round performance"""
        parameters = base_parameters.copy()
        
        if not previous_results:
            return parameters
        
        # Analyze previous performance
        recent_results = previous_results[-3:]  # Look at last 3 rounds
        avg_improvement = statistics.mean([r.improvement_score for r in recent_results])
        avg_quality = statistics.mean([r.quality_score for r in recent_results])
        
        # Adaptive adjustments based on performance
        if avg_improvement < 0.1:  # Low improvement
            # Increase exploration
            if 'mutation_rate' in parameters:
                parameters['mutation_rate'] = min(parameters.get('mutation_rate', 0.1) * 1.5, 0.5)
            if 'population_size' in parameters:
                parameters['population_size'] = int(parameters.get('population_size', 20) * 1.3)
        elif avg_improvement > 0.3:  # High improvement
            # Increase exploitation
            if 'mutation_rate' in parameters:
                parameters['mutation_rate'] = max(parameters.get('mutation_rate', 0.1) * 0.7, 0.01)
            if 'selection_pressure' in parameters:
                parameters['selection_pressure'] = min(parameters.get('selection_pressure', 0.5) * 1.2, 1.0)
        
        # Quality-based adjustments
        if avg_quality > 0.8:  # High quality
            # Fine-tune parameters
            if 'iterations' in parameters:
                parameters['iterations'] = max(int(parameters.get('iterations', 10) * 0.8), 5)
        elif avg_quality < 0.5:  # Low quality
            # Increase search effort
            if 'iterations' in parameters:
                parameters['iterations'] = int(parameters.get('iterations', 10) * 1.5)
        
        return parameters
    
    def _generate_random_parameters(self, base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random parameter variations"""
        import random
        
        parameters = base_parameters.copy()
        
        # Add random variations to parameters
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                # Add ±20% random variation
                variation = random.uniform(0.8, 1.2)
                if isinstance(value, int):
                    parameters[key] = max(1, int(value * variation))
                else:
                    parameters[key] = max(0.01, value * variation)
        
        return parameters
    
    def _generate_convergent_parameters(
        self, round_num: int, base_parameters: Dict[str, Any], previous_results: List[RoundResult]
    ) -> Dict[str, Any]:
        """Generate parameters that converge towards optimal values"""
        parameters = base_parameters.copy()
        
        if len(previous_results) < 2:
            return parameters
        
        # Find best performing parameters
        best_result = max(previous_results, key=lambda r: r.quality_score)
        best_params = best_result.configuration.parameters
        
        # Converge towards best parameters
        convergence_rate = 0.3  # 30% convergence per round
        
        for key in parameters:
            if key in best_params:
                current_val = parameters[key]
                best_val = best_params[key]
                
                if isinstance(current_val, (int, float)) and isinstance(best_val, (int, float)):
                    # Linear convergence
                    new_val = current_val + (best_val - current_val) * convergence_rate
                    
                    if isinstance(current_val, int):
                        parameters[key] = int(new_val)
                    else:
                        parameters[key] = new_val
        
        return parameters
    
    def _calculate_strategy_adjustments(
        self, strategy: RoundStrategy, round_num: int, previous_results: List[RoundResult]
    ) -> Dict[str, Any]:
        """Calculate strategy-specific adjustments"""
        adjustments = {}
        
        if strategy == RoundStrategy.ADAPTIVE and previous_results:
            # Calculate performance trend
            if len(previous_results) >= 2:
                recent_scores = [r.quality_score for r in previous_results[-2:]]
                adjustments['performance_trend'] = recent_scores[-1] - recent_scores[0]
            
            # Calculate consistency
            if len(previous_results) >= 3:
                scores = [r.quality_score for r in previous_results[-3:]]
                adjustments['consistency'] = 1.0 - statistics.stdev(scores) if len(set(scores)) > 1 else 1.0
        
        adjustments['round_factor'] = round_num
        adjustments['strategy'] = strategy.value
        
        return adjustments
    
    def _estimate_expected_improvement(self, round_num: int, previous_results: List[RoundResult]) -> float:
        """Estimate expected improvement for the round"""
        if not previous_results:
            return 0.1  # Default expectation
        
        # Calculate average improvement from previous rounds
        improvements = [r.improvement_score for r in previous_results if r.success]
        
        if not improvements:
            return 0.05
        
        avg_improvement = statistics.mean(improvements)
        
        # Diminishing returns - expect less improvement in later rounds
        diminishing_factor = 1.0 / (1.0 + (round_num - 1) * 0.1)
        
        return avg_improvement * diminishing_factor
    
    def _execute_round(
        self, content: str, test_function: Callable, config: RoundConfiguration
    ) -> RoundResult:
        """Execute a single round of testing"""
        start_time = time.time()
        
        try:
            # Execute the test function with the round configuration
            result = test_function(content, **config.parameters)
            
            # Extract results (assuming test_function returns a dict with specific keys)
            if isinstance(result, dict):
                content_after = result.get('content', content)
                quality_score = result.get('quality_score', 0.0)
                metrics = result.get('metrics', {})
                success = result.get('success', True)
                error_message = result.get('error', None)
            else:
                # If result is just the improved content
                content_after = str(result) if result else content
                quality_score = self._calculate_quality_score(content, content_after)
                metrics = {}
                success = True
                error_message = None
            
            # Calculate improvement score
            improvement_score = self._calculate_improvement_score(content, content_after)
            
            return RoundResult(
                round_number=config.round_number,
                configuration=config,
                content_before=content,
                content_after=content_after,
                quality_score=quality_score,
                improvement_score=improvement_score,
                execution_time=time.time() - start_time,
                metrics=metrics,
                success=success,
                error_message=error_message,
                metadata={
                    "parameters_used": config.parameters,
                    "execution_timestamp": time.time()
                }
            )
            
        except Exception as e:
            return RoundResult(
                round_number=config.round_number,
                configuration=config,
                content_before=content,
                content_after=content,  # No change on error
                quality_score=0.0,
                improvement_score=0.0,
                execution_time=time.time() - start_time,
                metrics={},
                success=False,
                error_message=str(e),
                metadata={
                    "error_type": type(e).__name__,
                    "execution_timestamp": time.time()
                }
            )
    
    def _calculate_quality_score(self, original_content: str, improved_content: str) -> float:
        """Calculate quality score for content"""
        # Simple heuristic-based quality scoring
        score = 0.0
        
        # Length improvement (but not too much)
        length_ratio = len(improved_content) / len(original_content) if original_content else 1.0
        if 0.8 <= length_ratio <= 1.5:  # Reasonable length change
            score += 0.3
        
        # Content diversity (more unique words is generally better)
        original_words = set(original_content.lower().split())
        improved_words = set(improved_content.lower().split())
        
        if original_words:
            diversity_improvement = len(improved_words) / len(original_words)
            score += min(diversity_improvement * 0.3, 0.3)
        
        # Structure improvement (more organized content)
        original_lines = len([line for line in original_content.split('\n') if line.strip()])
        improved_lines = len([line for line in improved_content.split('\n') if line.strip()])
        
        if original_lines > 0:
            structure_score = min(improved_lines / original_lines, 2.0) * 0.2
            score += structure_score
        
        # Completeness (content should not be empty or too short)
        if len(improved_content.strip()) > 50:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_improvement_score(self, original_content: str, improved_content: str) -> float:
        """Calculate improvement score between original and improved content"""
        if not original_content or not improved_content:
            return 0.0
        
        # Calculate various improvement metrics
        improvements = []
        
        # Length-based improvement
        if len(improved_content) > len(original_content):
            length_improvement = (len(improved_content) - len(original_content)) / len(original_content)
            improvements.append(min(length_improvement, 0.5))  # Cap at 50% improvement
        
        # Vocabulary improvement
        original_words = set(original_content.lower().split())
        improved_words = set(improved_content.lower().split())
        
        if original_words:
            vocab_improvement = len(improved_words - original_words) / len(original_words)
            improvements.append(min(vocab_improvement, 0.3))
        
        # Structure improvement (more paragraphs/sections)
        original_paragraphs = len([p for p in original_content.split('\n\n') if p.strip()])
        improved_paragraphs = len([p for p in improved_content.split('\n\n') if p.strip()])
        
        if original_paragraphs > 0:
            structure_improvement = (improved_paragraphs - original_paragraphs) / original_paragraphs
            improvements.append(min(max(structure_improvement, 0), 0.2))
        
        return sum(improvements) if improvements else 0.0
    
    def _check_stopping_criteria(
        self,
        criteria: List[RoundStoppingCriteria],
        results: List[RoundResult],
        current_round: int,
        max_rounds: int
    ) -> tuple:
        """Check if any stopping criteria are met"""
        
        for criterion in criteria:
            if criterion == RoundStoppingCriteria.MAX_ROUNDS:
                if current_round >= max_rounds:
                    return True, "max_rounds_reached"
            
            elif criterion == RoundStoppingCriteria.QUALITY_THRESHOLD:
                if results and results[-1].quality_score >= 0.95:
                    return True, "quality_threshold_reached"
            
            elif criterion == RoundStoppingCriteria.IMPROVEMENT_PLATEAU:
                if len(results) >= 3:
                    recent_improvements = [r.improvement_score for r in results[-3:]]
                    if all(imp < 0.01 for imp in recent_improvements):  # Lower threshold
                        return True, "improvement_plateau_reached"
            
            elif criterion == RoundStoppingCriteria.CONVERGENCE:
                if len(results) >= 4:
                    recent_scores = [r.quality_score for r in results[-4:]]
                    if statistics.stdev(recent_scores) < 0.02:  # Very low variance
                        return True, "convergence_reached"
        
        return False, ""
    
    def _calculate_total_improvement(
        self, original_content: str, final_content: str, results: List[RoundResult]
    ) -> float:
        """Calculate total improvement across all rounds"""
        if not results:
            return 0.0
        
        # Direct improvement from original to final
        direct_improvement = self._calculate_improvement_score(original_content, final_content)
        
        # Cumulative improvement from all successful rounds
        cumulative_improvement = sum(r.improvement_score for r in results if r.success)
        
        # Average of direct and cumulative (weighted towards direct)
        total_improvement = (direct_improvement * 0.7) + (cumulative_improvement * 0.3)
        
        return min(total_improvement, 1.0)
    
    def _update_adaptive_parameters(self, result: MultiRoundResult):
        """Update adaptive parameters based on multi-round result"""
        # Store successful parameter combinations
        if result.best_round and result.best_round.success:
            best_params = result.best_round.configuration.parameters
            
            # Update adaptive parameters with successful values
            for key, value in best_params.items():
                if key not in self.adaptive_parameters:
                    self.adaptive_parameters[key] = []
                
                self.adaptive_parameters[key].append({
                    'value': value,
                    'quality_score': result.best_round.quality_score,
                    'timestamp': time.time()
                })
                
                # Keep only recent successful parameters (last 10)
                self.adaptive_parameters[key] = self.adaptive_parameters[key][-10:]
    
    def get_adaptive_recommendations(self) -> Dict[str, Any]:
        """Get parameter recommendations based on adaptive learning"""
        recommendations = {}
        
        for param_name, param_history in self.adaptive_parameters.items():
            if param_history:
                # Weight recent and high-quality parameters more
                weighted_values = []
                for entry in param_history:
                    weight = entry['quality_score']  # Quality-based weighting
                    weighted_values.extend([entry['value']] * int(weight * 10))
                
                if weighted_values:
                    if isinstance(weighted_values[0], (int, float)):
                        recommendations[param_name] = statistics.mean(weighted_values)
                    else:
                        # For non-numeric values, use most common
                        recommendations[param_name] = max(set(weighted_values), key=weighted_values.count)
        
        return recommendations
    
    def get_round_history(self) -> List[MultiRoundResult]:
        """Get history of multi-round test results"""
        return self.round_history.copy()
    
    def clear_history(self):
        """Clear round history and adaptive parameters"""
        self.round_history.clear()
        self.adaptive_parameters.clear()


# Utility functions for integration with existing systems

def create_evolution_test_function(evolution_function: Callable) -> Callable:
    """Create a test function wrapper for evolution functions"""
    def test_function(content: str, **parameters) -> Dict[str, Any]:
        try:
            result = evolution_function(content, **parameters)
            
            # Extract relevant information from evolution result
            if isinstance(result, dict):
                return {
                    'content': result.get('evolved_content', content),
                    'quality_score': result.get('quality_score', 0.0),
                    'metrics': result.get('metrics', {}),
                    'success': True
                }
            else:
                return {
                    'content': str(result) if result else content,
                    'quality_score': 0.5,  # Default score
                    'metrics': {},
                    'success': True
                }
        except Exception as e:
            return {
                'content': content,
                'quality_score': 0.0,
                'metrics': {},
                'success': False,
                'error': str(e)
            }
    
    return test_function


def create_team_test_function(team_function: Callable) -> Callable:
    """Create a test function wrapper for team-based functions"""
    def test_function(content: str, **parameters) -> Dict[str, Any]:
        try:
            result = team_function(content, **parameters)
            
            # Extract team result information
            if hasattr(result, 'improved_content'):
                improved_content = result.improved_content
            elif hasattr(result, 'content'):
                improved_content = result.content
            elif isinstance(result, dict):
                improved_content = result.get('content', content)
            else:
                improved_content = str(result) if result else content
            
            # Extract quality score
            if hasattr(result, 'overall_score'):
                quality_score = result.overall_score / 10.0  # Normalize to 0-1
            elif hasattr(result, 'quality_score'):
                quality_score = result.quality_score
            elif isinstance(result, dict):
                quality_score = result.get('quality_score', 0.5)
            else:
                quality_score = 0.5
            
            return {
                'content': improved_content,
                'quality_score': quality_score,
                'metrics': getattr(result, 'metrics', {}),
                'success': True
            }
        except Exception as e:
            return {
                'content': content,
                'quality_score': 0.0,
                'metrics': {},
                'success': False,
                'error': str(e)
            }
    
    return test_function


def run_multi_round_evolution(
    content: str,
    evolution_function: Callable,
    rounds: int = 5,
    strategy: RoundStrategy = RoundStrategy.ADAPTIVE,
    **kwargs
) -> MultiRoundResult:
    """
    Convenience function to run multi-round evolution testing.
    
    Args:
        content: Content to evolve
        evolution_function: Evolution function to use
        rounds: Number of rounds
        strategy: Round strategy
        **kwargs: Additional parameters
        
    Returns:
        MultiRoundResult: Complete multi-round results
    """
    tester = MultiRoundTester()
    test_function = create_evolution_test_function(evolution_function)
    
    return tester.run_multi_round_test(
        content=content,
        test_function=test_function,
        max_rounds=rounds,
        strategy=strategy,
        **kwargs
    )


def run_multi_round_team_testing(
    content: str,
    team_function: Callable,
    rounds: int = 3,
    strategy: RoundStrategy = RoundStrategy.PROGRESSIVE,
    **kwargs
) -> MultiRoundResult:
    """
    Convenience function to run multi-round team testing.
    
    Args:
        content: Content to test
        team_function: Team function to use
        rounds: Number of rounds
        strategy: Round strategy
        **kwargs: Additional parameters
        
    Returns:
        MultiRoundResult: Complete multi-round results
    """
    tester = MultiRoundTester()
    test_function = create_team_test_function(team_function)
    
    return tester.run_multi_round_test(
        content=content,
        test_function=test_function,
        max_rounds=rounds,
        strategy=strategy,
        **kwargs
    )