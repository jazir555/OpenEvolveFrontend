"""
Learning Engine for Knowledge Orchestrator

A comprehensive learning system that enables the orchestrator to:
- Learn from successes and failures
- Adapt component selection based on past performance
- Build knowledge about which components work best for different data types
- Self-optimize pipeline configurations
- Transfer learning across similar tasks

Following self-healing principles:
- Every failure is a learning opportunity
- Component performance is continuously monitored
- Pipeline configurations evolve based on results
- The system gets smarter with every execution
"""

import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import statistics
import hashlib
import copy

logger = logging.getLogger(__name__)


@dataclass
class LearningExperience:
    """A single learning experience from execution"""
    experience_id: str
    timestamp: str
    input_hash: str  # Hash of input for similarity matching
    data_type: str
    domain: str
    pipeline_config: Dict[str, Any]
    components_used: List[str]
    
    # Outcomes
    success: bool
    execution_time_ms: float
    results_quality: float  # 0.0 to 1.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # Component performance details
    component_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Learning metadata
    lessons_learned: List[str] = field(default_factory=list)
    would_change: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningExperience':
        return cls(**data)


@dataclass
class ComponentProfile:
    """Performance profile for a component"""
    component_type: str
    
    # Performance metrics
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_execution_time_ms: float = 0.0
    
    # Quality metrics
    average_quality_score: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    
    # Context-specific performance
    performance_by_data_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_by_domain: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Error patterns
    error_counts: Dict[str, int] = field(default_factory=dict)
    common_errors: List[str] = field(default_factory=list)
    
    # Learning
    best_configurations: List[Dict[str, Any]] = field(default_factory=list)
    worst_configurations: List[Dict[str, Any]] = field(default_factory=list)
    
    def update(self, success: bool, execution_time_ms: float, quality_score: float,
               data_type: str, domain: str, error_type: Optional[str] = None,
               config: Optional[Dict] = None):
        """Update profile with new execution data"""
        self.total_invocations += 1
        self.total_execution_time_ms += execution_time_ms
        
        if success:
            self.successful_invocations += 1
        else:
            self.failed_invocations += 1
            if error_type:
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Update quality scores
        self.quality_scores.append(quality_score)
        if len(self.quality_scores) > 100:  # Keep last 100
            self.quality_scores = self.quality_scores[-100:]
        self.average_quality_score = statistics.mean(self.quality_scores)
        
        # Update context-specific performance
        self._update_context_performance(
            self.performance_by_data_type, data_type, success, quality_score
        )
        self._update_context_performance(
            self.performance_by_domain, domain, success, quality_score
        )
        
        # Track best/worst configurations
        if config:
            config_entry = {
                'config': config,
                'quality_score': quality_score,
                'execution_time_ms': execution_time_ms,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            if success and quality_score > 0.7:
                self.best_configurations.append(config_entry)
                self.best_configurations = self.best_configurations[-10:]  # Keep top 10
            elif not success or quality_score < 0.3:
                self.worst_configurations.append(config_entry)
                self.worst_configurations = self.worst_configurations[-10:]
    
    def _update_context_performance(self, perf_dict: Dict, key: str, 
                                    success: bool, quality_score: float):
        """Update performance metrics for a specific context"""
        if key not in perf_dict:
            perf_dict[key] = {'invocations': 0, 'successes': 0, 'avg_quality': 0.0, 'qualities': []}
        
        entry = perf_dict[key]
        entry['invocations'] += 1
        if success:
            entry['successes'] += 1
        entry['qualities'].append(quality_score)
        if len(entry['qualities']) > 50:
            entry['qualities'] = entry['qualities'][-50:]
        entry['avg_quality'] = statistics.mean(entry['qualities'])
        entry['success_rate'] = entry['successes'] / entry['invocations']
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        if self.total_invocations == 0:
            return 0.0
        return self.successful_invocations / self.total_invocations
    
    @property
    def average_execution_time_ms(self) -> float:
        """Calculate average execution time"""
        if self.total_invocations == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_invocations
    
    def get_recommendation_for_context(self, data_type: str, domain: str) -> Dict[str, Any]:
        """Get component recommendation for specific context"""
        data_type_perf = self.performance_by_data_type.get(data_type, {})
        domain_perf = self.performance_by_domain.get(domain, {})
        
        # Weight domain performance higher
        combined_score = (
            data_type_perf.get('avg_quality', 0.5) * 0.3 +
            domain_perf.get('avg_quality', 0.5) * 0.7
        )
        
        return {
            'component': self.component_type,
            'expected_success_rate': self.success_rate,
            'expected_quality': combined_score,
            'expected_execution_time_ms': self.average_execution_time_ms,
            'recommended': combined_score > 0.5 and self.success_rate > 0.7,
            'confidence': min(self.total_invocations / 10, 1.0)  # Confidence increases with experience
        }


@dataclass
class PipelinePattern:
    """A learned pattern for pipeline configurations"""
    pattern_id: str
    
    # Matching criteria
    data_type_patterns: List[str] = field(default_factory=list)
    domain_patterns: List[str] = field(default_factory=list)
    input_characteristics: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline configuration
    component_sequence: List[str] = field(default_factory=list)
    component_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance
    success_rate: float = 0.0
    average_quality: float = 0.0
    average_execution_time_ms: float = 0.0
    usage_count: int = 0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_used: Optional[str] = None
    
    def matches(self, data_type: str, domain: str, input_chars: Dict[str, Any]) -> float:
        """Calculate match score for input characteristics"""
        score = 0.0
        
        # Data type match
        if data_type in self.data_type_patterns:
            score += 0.4
        
        # Domain match
        if domain in self.domain_patterns:
            score += 0.4
        
        # Input characteristics match
        char_matches = 0
        for key, value in self.input_characteristics.items():
            if key in input_chars and input_chars[key] == value:
                char_matches += 1
        if self.input_characteristics:
            score += 0.2 * (char_matches / len(self.input_characteristics))
        
        return score


class LearningEngine:
    """
    Main learning engine for the Knowledge Orchestrator.
    
    Responsibilities:
    - Record and analyze execution experiences
    - Build component performance profiles
    - Learn optimal pipeline patterns
    - Recommend component configurations
    - Predict failures before they happen
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the learning engine.
        
        Args:
            storage_path: Path to persist learning data
        """
        self.storage_path = storage_path
        
        # Learning data
        self.experiences: List[LearningExperience] = []
        self.component_profiles: Dict[str, ComponentProfile] = {}
        self.pipeline_patterns: List[PipelinePattern] = []
        
        # Learning configuration
        self.max_experiences = 1000
        self.min_experiences_for_recommendation = 5
        self.similarity_threshold = 0.8
        
        # Performance tracking
        self.global_success_rate = 0.0
        self.global_average_quality = 0.0
        
        # Load persisted data if available
        if storage_path:
            self._load_data()
        
        logger.info({
            "msg": "LearningEngine initialized",
            "experiences_count": len(self.experiences),
            "profiles_count": len(self.component_profiles),
            "patterns_count": len(self.pipeline_patterns),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def record_experience(self, 
                         input_data: Dict[str, Any],
                         data_type: str,
                         domain: str,
                         pipeline_config: Dict[str, Any],
                         components_used: List[str],
                         success: bool,
                         execution_time_ms: float,
                         results: Dict[str, Any],
                         errors: List[Dict[str, Any]]) -> LearningExperience:
        """
        Record a learning experience from execution.
        
        Args:
            input_data: Input data processed
            data_type: Type of data
            domain: Domain preset used
            pipeline_config: Pipeline configuration
            components_used: List of components that executed
            success: Whether execution succeeded
            execution_time_ms: Execution time
            results: Execution results
            errors: List of errors encountered
            
        Returns:
            LearningExperience object
        """
        # Generate experience ID
        experience_id = f"exp_{datetime.now(timezone.utc).timestamp()}"
        
        # Hash input for similarity matching
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(results, success, errors)
        
        # Analyze component performance
        component_performance = self._analyze_component_performance(
            components_used, results, errors
        )
        
        # Generate lessons learned
        lessons = self._generate_lessons_learned(
            success, errors, component_performance, pipeline_config
        )
        
        # What would we change
        would_change = self._suggest_improvements(
            success, errors, component_performance, components_used
        )
        
        # Create experience
        experience = LearningExperience(
            experience_id=experience_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash=input_hash,
            data_type=data_type,
            domain=domain,
            pipeline_config=pipeline_config,
            components_used=components_used,
            success=success,
            execution_time_ms=execution_time_ms,
            results_quality=quality_score,
            error_type=errors[0].get('type') if errors else None,
            error_message=errors[0].get('message') if errors else None,
            component_performance=component_performance,
            lessons_learned=lessons,
            would_change=would_change
        )
        
        # Store experience
        self.experiences.append(experience)
        if len(self.experiences) > self.max_experiences:
            self.experiences = self.experiences[-self.max_experiences:]
        
        # Update component profiles
        self._update_component_profiles(experience)
        
        # Update pipeline patterns
        self._update_pipeline_patterns(experience)
        
        # Update global metrics
        self._update_global_metrics()
        
        # Persist if path set
        if self.storage_path:
            self._save_data()
        
        logger.debug({
            "msg": "Learning experience recorded",
            "experience_id": experience_id,
            "success": success,
            "quality_score": quality_score,
            "lessons_count": len(lessons)
        })
        
        return experience
    
    def _calculate_quality_score(self, results: Dict[str, Any], 
                                  success: bool, 
                                  errors: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score for execution"""
        if not success:
            return 0.0
        
        score = 1.0
        
        # Penalize for errors (even partial success)
        if errors:
            score -= len(errors) * 0.1
        
        # Check result completeness
        result_keys = list(results.keys())
        if result_keys:
            # Check for empty results
            empty_results = sum(1 for v in results.values() 
                              if v is None or (isinstance(v, dict) and not v) or
                              (isinstance(v, list) and not v))
            score -= (empty_results / len(result_keys)) * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _analyze_component_performance(self, components_used: List[str],
                                       results: Dict[str, Any],
                                       errors: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze individual component performance"""
        performance = {}
        
        for component in components_used:
            comp_errors = [e for e in errors if e.get('component') == component]
            comp_result = results.get(component, {})
            
            performance[component] = {
                'success': len(comp_errors) == 0,
                'errors': comp_errors,
                'result_keys': list(comp_result.keys()) if isinstance(comp_result, dict) else [],
                'result_size': len(str(comp_result))
            }
        
        return performance
    
    def _generate_lessons_learned(self, success: bool, errors: List[Dict[str, Any]],
                                   component_performance: Dict[str, Dict[str, Any]],
                                   pipeline_config: Dict[str, Any]) -> List[str]:
        """Generate lessons learned from execution"""
        lessons = []
        
        if not success:
            # Learn from failures
            for error in errors:
                error_type = error.get('type', 'unknown')
                component = error.get('component', 'unknown')
                
                if 'timeout' in error_type.lower():
                    lessons.append(f"Component {component} may need longer timeout")
                elif 'memory' in error_type.lower():
                    lessons.append(f"Component {component} has high memory requirements")
                elif 'import' in error_type.lower():
                    lessons.append(f"Component {component} has missing dependencies")
                else:
                    lessons.append(f"Component {component} failed with {error_type}")
        else:
            # Learn from successes
            lessons.append("Pipeline configuration successful")
            
            # Identify star performers
            for comp, perf in component_performance.items():
                if perf['success'] and len(perf.get('result_keys', [])) > 3:
                    lessons.append(f"Component {comp} produced comprehensive results")
        
        return lessons
    
    def _suggest_improvements(self, success: bool, errors: List[Dict[str, Any]],
                             component_performance: Dict[str, Dict[str, Any]],
                             components_used: List[str]) -> List[str]:
        """Suggest improvements for future runs"""
        suggestions = []
        
        # Check for failed components
        for comp, perf in component_performance.items():
            if not perf['success']:
                suggestions.append(f"Consider replacing {comp} or adding fallback")
        
        # Check for missing components that could help
        if 'causal_learn' not in components_used:
            suggestions.append("Consider adding causal_learn for relationship analysis")
        
        if 'pami' not in components_used and len(components_used) > 2:
            suggestions.append("Consider adding pami for pattern mining")
        
        return suggestions
    
    def _update_component_profiles(self, experience: LearningExperience):
        """Update component profiles with new experience"""
        for component in experience.components_used:
            if component not in self.component_profiles:
                self.component_profiles[component] = ComponentProfile(component_type=component)
            
            comp_perf = experience.component_performance.get(component, {})
            
            self.component_profiles[component].update(
                success=comp_perf.get('success', experience.success),
                execution_time_ms=experience.execution_time_ms / len(experience.components_used),
                quality_score=experience.results_quality,
                data_type=experience.data_type,
                domain=experience.domain,
                error_type=experience.error_type if not comp_perf.get('success', True) else None,
                config=experience.pipeline_config.get('components', {}).get(component)
            )
    
    def _update_pipeline_patterns(self, experience: LearningExperience):
        """Update learned pipeline patterns"""
        # Look for existing pattern
        matching_pattern = None
        for pattern in self.pipeline_patterns:
            if (experience.data_type in pattern.data_type_patterns and
                experience.domain in pattern.domain_patterns and
                pattern.component_sequence == experience.components_used):
                matching_pattern = pattern
                break
        
        if matching_pattern:
            # Update existing pattern
            matching_pattern.usage_count += 1
            matching_pattern.last_used = datetime.now(timezone.utc).isoformat()
            
            # Update success rate
            total_successes = matching_pattern.success_rate * (matching_pattern.usage_count - 1)
            if experience.success:
                total_successes += 1
            matching_pattern.success_rate = total_successes / matching_pattern.usage_count
            
            # Update quality
            matching_pattern.average_quality = (
                (matching_pattern.average_quality * (matching_pattern.usage_count - 1) +
                 experience.results_quality) / matching_pattern.usage_count
            )
            
            # Update execution time
            matching_pattern.average_execution_time_ms = (
                (matching_pattern.average_execution_time_ms * (matching_pattern.usage_count - 1) +
                 experience.execution_time_ms) / matching_pattern.usage_count
            )
        else:
            # Create new pattern
            new_pattern = PipelinePattern(
                pattern_id=f"pattern_{len(self.pipeline_patterns)}",
                data_type_patterns=[experience.data_type],
                domain_patterns=[experience.domain],
                component_sequence=experience.components_used,
                component_configs=experience.pipeline_config.get('components', {}),
                success_rate=1.0 if experience.success else 0.0,
                average_quality=experience.results_quality,
                average_execution_time_ms=experience.execution_time_ms,
                usage_count=1,
                last_used=datetime.now(timezone.utc).isoformat()
            )
            self.pipeline_patterns.append(new_pattern)
    
    def _update_global_metrics(self):
        """Update global performance metrics"""
        if not self.experiences:
            return
        
        successes = sum(1 for e in self.experiences if e.success)
        self.global_success_rate = successes / len(self.experiences)
        self.global_average_quality = statistics.mean(e.results_quality for e in self.experiences)
    
    def recommend_components(self, data_type: str, domain: str, 
                            input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend components for given input context.
        
        Args:
            data_type: Type of data to process
            domain: Domain for processing
            input_data: Input data characteristics
            
        Returns:
            List of component recommendations with scores
        """
        recommendations = []
        
        # Get recommendations from component profiles
        for component_type, profile in self.component_profiles.items():
            rec = profile.get_recommendation_for_context(data_type, domain)
            recommendations.append(rec)
        
        # Sort by expected quality
        recommendations.sort(key=lambda x: x['expected_quality'], reverse=True)
        
        # Add recommendations for untried components
        all_components = ['deepke', 'karate_club', 'kg_gen', 'pami', 
                         'neuralkg', 'causal_learn', 'lagrange_mapper',
                         'global_chem', 'neuromancer']
        tried_components = set(r['component'] for r in recommendations)
        
        for comp in all_components:
            if comp not in tried_components:
                recommendations.append({
                    'component': comp,
                    'expected_success_rate': 0.5,  # Unknown
                    'expected_quality': 0.5,
                    'expected_execution_time_ms': 0,
                    'recommended': True,  # Recommend trying
                    'confidence': 0.0,
                    'reason': 'not_yet_tried'
                })
        
        return recommendations
    
    def recommend_pipeline(self, data_type: str, domain: str,
                          input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Recommend a pipeline configuration based on learned patterns.
        
        Args:
            data_type: Type of data
            domain: Domain for processing
            input_data: Input data
            
        Returns:
            Recommended pipeline configuration or None
        """
        # Find matching patterns
        matches = []
        for pattern in self.pipeline_patterns:
            score = pattern.matches(data_type, domain, {})
            if score > 0.5:  # Threshold for matching
                matches.append((pattern, score))
        
        if not matches:
            return None
        
        # Sort by score and pattern quality
        matches.sort(key=lambda x: (x[1], x[0].success_rate, x[0].average_quality), reverse=True)
        
        best_pattern = matches[0][0]
        match_score = matches[0][1]
        
        return {
            'component_sequence': best_pattern.component_sequence,
            'component_configs': best_pattern.component_configs,
            'expected_success_rate': best_pattern.success_rate,
            'expected_quality': best_pattern.average_quality,
            'expected_execution_time_ms': best_pattern.average_execution_time_ms,
            'pattern_confidence': min(best_pattern.usage_count / 10, 1.0),
            'match_score': match_score,
            'based_on_experiences': best_pattern.usage_count
        }
    
    def predict_failure(self, data_type: str, domain: str,
                       components: List[str]) -> Optional[Dict[str, Any]]:
        """
        Predict if a configuration is likely to fail.
        
        Args:
            data_type: Type of data
            domain: Domain
            components: Planned components
            
        Returns:
            Failure prediction or None if likely to succeed
        """
        warnings = []
        
        for component in components:
            if component in self.component_profiles:
                profile = self.component_profiles[component]
                
                # Check context-specific performance
                data_perf = profile.performance_by_data_type.get(data_type, {})
                domain_perf = profile.performance_by_domain.get(domain, {})
                
                data_success_rate = data_perf.get('success_rate', profile.success_rate)
                domain_success_rate = domain_perf.get('success_rate', profile.success_rate)
                
                # Weighted success rate
                expected_success = data_success_rate * 0.3 + domain_success_rate * 0.7
                
                if expected_success < 0.3:
                    warnings.append({
                        'component': component,
                        'risk': 'high',
                        'expected_success_rate': expected_success,
                        'reason': f'Low historical success for {domain}/{data_type}'
                    })
                elif expected_success < 0.5:
                    warnings.append({
                        'component': component,
                        'risk': 'medium',
                        'expected_success_rate': expected_success,
                        'reason': f'Moderate historical success for {domain}/{data_type}'
                    })
        
        if warnings:
            return {
                'likely_to_fail': any(w['risk'] == 'high' for w in warnings),
                'warnings': warnings,
                'recommendation': 'Consider alternative components or add fallbacks'
            }
        
        return None
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress"""
        return {
            'total_experiences': len(self.experiences),
            'component_profiles': len(self.component_profiles),
            'learned_patterns': len(self.pipeline_patterns),
            'global_success_rate': self.global_success_rate,
            'global_average_quality': self.global_average_quality,
            'component_rankings': [
                {
                    'component': comp,
                    'success_rate': profile.success_rate,
                    'avg_quality': profile.average_quality_score,
                    'total_invocations': profile.total_invocations
                }
                for comp, profile in sorted(
                    self.component_profiles.items(),
                    key=lambda x: x[1].success_rate,
                    reverse=True
                )
            ],
            'top_patterns': [
                {
                    'sequence': p.component_sequence,
                    'success_rate': p.success_rate,
                    'quality': p.average_quality,
                    'usage_count': p.usage_count
                }
                for p in sorted(
                    self.pipeline_patterns,
                    key=lambda x: (x.success_rate, x.average_quality),
                    reverse=True
                )[:5]
            ]
        }
    
    def find_similar_experiences(self, input_data: Dict[str, Any], 
                                 n: int = 5) -> List[LearningExperience]:
        """Find similar past experiences"""
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()
        
        # Find experiences with matching hash
        similar = [e for e in self.experiences if e.input_hash == input_hash]
        
        # If not enough, use recent experiences
        if len(similar) < n:
            recent = [e for e in self.experiences[-20:] if e not in similar]
            similar.extend(recent)
        
        return similar[:n]
    
    def _save_data(self):
        """Persist learning data to storage"""
        try:
            data = {
                'experiences': [e.to_dict() for e in self.experiences],
                'pipeline_patterns': [asdict(p) for p in self.pipeline_patterns],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def _load_data(self):
        """Load learning data from storage"""
        try:
            import os
            if not os.path.exists(self.storage_path):
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Load experiences
            self.experiences = [
                LearningExperience.from_dict(e) 
                for e in data.get('experiences', [])
            ]
            
            logger.info({
                "msg": "Learning data loaded",
                "experiences": len(self.experiences),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to load learning data: {e}")
