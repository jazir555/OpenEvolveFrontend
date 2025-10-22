"""
Fallback Handler - Provides fallback strategies when OpenEvolve unavailable
Handles graceful degradation and caching of fallback results
"""

import time
import logging
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class FallbackResult:
    """Result from fallback operation"""
    success: bool
    result: Any
    fallback_type: str
    cached: bool = False
    error: Optional[str] = None


class FallbackCache:
    """Caches fallback results"""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Cache result"""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_times.clear()


class FallbackHandler:
    """Handles fallback strategies when OpenEvolve unavailable"""
    
    def __init__(self):
        self.cache = FallbackCache()
        self.logger = logging.getLogger(__name__)
        self.fallback_count = 0
    
    def get_fallback_result(self, operation_type: str, input_data: Dict[str, Any]) -> Any:
        """
        Get fallback result for operation.
        
        Args:
            operation_type: Type of operation (evolution, blue_team_solution, red_team_critique, etc.)
            input_data: Input data for the operation
            
        Returns:
            Fallback result appropriate for the operation type
        """
        self.fallback_count += 1
        self.logger.info(f"Using fallback for {operation_type} (count: {self.fallback_count})")
        
        # Check cache first
        cache_key = self._generate_cache_key(operation_type, input_data)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.logger.info(f"Using cached fallback result for {operation_type}")
            return cached_result
        
        # Route to appropriate fallback
        if operation_type == "evolution":
            result = self._fallback_evolution(input_data)
        elif operation_type == "blue_team_solution":
            result = self._fallback_blue_team_solution(input_data)
        elif operation_type == "red_team_critique":
            result = self._fallback_red_team_critique(input_data)
        elif operation_type == "evaluator_assessment":
            result = self._fallback_evaluator_assessment(input_data)
        elif operation_type == "content_analysis":
            result = self._fallback_content_analysis(input_data)
        elif operation_type == "decomposition":
            result = self._fallback_decomposition(input_data)
        else:
            self.logger.warning(f"Unknown operation type: {operation_type}")
            result = self._fallback_generic(input_data)
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, operation_type: str, input_data: Dict[str, Any]) -> str:
        """Generate cache key from operation and input"""
        import hashlib
        
        # Create a stable string representation
        key_parts = [operation_type]
        
        if 'content' in input_data:
            content_hash = hashlib.md5(input_data['content'].encode()).hexdigest()[:8]
            key_parts.append(content_hash)
        
        if 'evolution_mode' in input_data:
            key_parts.append(input_data['evolution_mode'])
        
        if 'content_type' in input_data:
            key_parts.append(input_data['content_type'])
        
        return "_".join(key_parts)
    
    def _fallback_evolution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback for evolution operation"""
        content = input_data.get('content', '')
        
        return {
            'success': True,
            'best_code': content,
            'best_score': 0.5,
            'iterations_completed': 0,
            'metrics': {
                'fallback_used': True,
                'fallback_reason': 'OpenEvolve unavailable'
            }
        }
    
    def _fallback_blue_team_solution(self, input_data: Dict[str, Any]) -> Any:
        """Fallback for Blue Team solution generation"""
        from blue_team import BlueTeamAssessment, BlueTeamFix
        
        content = input_data.get('content', '')
        
        return BlueTeamAssessment(
            original_content=content,
            fixed_content=content,
            applied_fixes=[],
            fix_suggestions=[],
            assessment_summary="Fallback mode - OpenEvolve unavailable",
            overall_improvement_score=0.0,
            time_taken=0.0,
            assessment_metadata={'fallback_used': True},
            fixes_by_type={},
            fixes_by_priority={}
        )
    
    def _fallback_red_team_critique(self, input_data: Dict[str, Any]) -> Any:
        """Fallback for Red Team critique"""
        from red_team import RedTeamAssessment
        
        return RedTeamAssessment(
            findings=[],
            assessment_summary="Fallback mode - OpenEvolve unavailable",
            confidence_score=0.5,
            time_taken=0.0,
            assessment_metadata={'fallback_used': True},
            issues_by_severity={},
            issues_by_category={}
        )
    
    def _fallback_evaluator_assessment(self, input_data: Dict[str, Any]) -> Any:
        """Fallback for Evaluator assessment"""
        from evaluator_team import EvaluatorAssessment, EvaluationScore, EvaluationMetric, EvaluationConfidence
        
        return EvaluatorAssessment(
            evaluator_id="fallback",
            scores=[],
            composite_score=50.0,
            assessment_summary="Fallback mode - OpenEvolve unavailable",
            confidence_level=EvaluationConfidence.LOW,
            time_taken=0.0,
            assessment_metadata={'fallback_used': True},
            criteria_used=[],
            detailed_feedback={}
        )
    
    def _fallback_content_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback for content analysis"""
        content = input_data.get('content', '')
        
        return {
            'domain': 'general',
            'keywords': [],
            'estimated_complexity': 5,
            'potential_challenges': [],
            'required_expertise': [],
            'summary': content[:200] + '...' if len(content) > 200 else content,
            'fallback_used': True
        }
    
    def _fallback_decomposition(self, input_data: Dict[str, Any]) -> Any:
        """Fallback for decomposition"""
        from workflow_structures import DecompositionPlan, SubProblem
        
        problem_statement = input_data.get('problem_statement', '')
        analyzed_context = input_data.get('analyzed_context', {})
        
        # Create a single sub-problem as fallback
        sub_problem = SubProblem(
            id="sp_1",
            description=problem_statement,
            ai_suggested_complexity_score=5,
            dependencies=[],
            solver_team_name=None,
            red_gauntlet_name=None,
            gold_gauntlet_name=None
        )
        
        return DecompositionPlan(
            problem_statement=problem_statement,
            analyzed_context=analyzed_context,
            sub_problems=[sub_problem]
        )
    
    def _fallback_generic(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic fallback"""
        return {
            'success': False,
            'fallback_used': True,
            'error': 'OpenEvolve unavailable and no specific fallback defined'
        }
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback usage statistics"""
        return {
            'total_fallbacks': self.fallback_count,
            'cache_size': len(self.cache.cache),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # Simplified - would need to track hits/misses properly
        return 0.0
    
    def clear_cache(self):
        """Clear fallback cache"""
        self.cache.clear()
