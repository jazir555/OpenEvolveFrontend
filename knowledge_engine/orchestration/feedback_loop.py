"""
Feedback Loop System

Continuous improvement system that:
1. Collects feedback on execution results
2. Analyzes success/failure patterns
3. Adjusts orchestration parameters
4. Evolves pipeline configurations
5. Retrains models based on feedback
6. Implements A/B testing for configurations

The feedback loop ensures the system gets smarter over time,
adapting to the specific use cases and data patterns of the deployment.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import defaultdict
import statistics
import copy

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    QUALITY_ISSUE = "quality_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    MISSING_INFORMATION = "missing_information"
    USER_CORRECTION = "user_correction"


class ImprovementArea(Enum):
    """Areas for improvement"""
    COMPONENT_SELECTION = "component_selection"
    PIPELINE_CONFIGURATION = "pipeline_configuration"
    PARAMETER_TUNING = "parameter_tuning"
    GAP_COVERAGE = "gap_coverage"
    CROSS_VALIDATION = "cross_validation"
    ERROR_HANDLING = "error_handling"


@dataclass
class FeedbackEntry:
    """A single feedback entry"""
    feedback_id: str
    timestamp: str
    correlation_id: str
    
    # What was executed
    input_data_summary: Dict[str, Any]
    components_used: List[str]
    pipeline_config: Dict[str, Any]
    
    # Feedback details
    feedback_type: FeedbackType
    rating: Optional[int] = None  # 1-5 rating
    specific_issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # User corrections (if any)
    expected_output: Optional[Dict[str, Any]] = None
    actual_output: Optional[Dict[str, Any]] = None
    
    # Metadata
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementExperiment:
    """An A/B test experiment for improvement"""
    experiment_id: str
    created_at: str
    
    # What we're testing
    improvement_area: ImprovementArea
    hypothesis: str
    
    # Variants
    control_config: Dict[str, Any]
    treatment_config: Dict[str, Any]
    
    # Results
    control_results: List[Dict[str, Any]] = field(default_factory=list)
    treatment_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status
    status: str = "running"  # running, completed, cancelled
    winner: Optional[str] = None  # control, treatment, tie
    confidence: float = 0.0


class FeedbackCollector:
    """
    Collects and manages feedback on orchestration results.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize feedback collector.
        
        Args:
            storage_path: Path to persist feedback
        """
        self.storage_path = storage_path
        self.feedback_entries: List[FeedbackEntry] = []
        
        # Load existing feedback
        if storage_path:
            self._load_feedback()
        
        logger.info({
            "msg": "FeedbackCollector initialized",
            "entries_count": len(self.feedback_entries),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def collect_feedback(self,
                        correlation_id: str,
                        input_data: Dict[str, Any],
                        components_used: List[str],
                        pipeline_config: Dict[str, Any],
                        feedback_type: FeedbackType,
                        rating: Optional[int] = None,
                        issues: List[str] = None,
                        suggestions: List[str] = None,
                        expected_output: Dict[str, Any] = None,
                        actual_output: Dict[str, Any] = None,
                        context: Dict[str, Any] = None) -> FeedbackEntry:
        """
        Collect feedback on an execution.
        
        Args:
            correlation_id: Execution correlation ID
            input_data: Input that was processed
            components_used: Components that were used
            pipeline_config: Pipeline configuration
            feedback_type: Type of feedback
            rating: Optional 1-5 rating
            issues: Specific issues encountered
            suggestions: User suggestions
            expected_output: What was expected
            actual_output: What was produced
            context: Additional context
            
        Returns:
            FeedbackEntry
        """
        entry = FeedbackEntry(
            feedback_id=f"fb_{len(self.feedback_entries)}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id,
            input_data_summary=self._summarize_input(input_data),
            components_used=components_used,
            pipeline_config=pipeline_config,
            feedback_type=feedback_type,
            rating=rating,
            specific_issues=issues or [],
            suggestions=suggestions or [],
            expected_output=expected_output,
            actual_output=actual_output,
            context=context or {}
        )
        
        self.feedback_entries.append(entry)
        
        # Persist
        if self.storage_path:
            self._save_feedback()
        
        logger.info({
            "msg": "Feedback collected",
            "feedback_id": entry.feedback_id,
            "type": feedback_type.value,
            "rating": rating,
            "correlation_id": correlation_id
        })
        
        return entry
    
    def _summarize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of input for storage"""
        summary = {}
        
        if 'text' in input_data:
            text = input_data['text']
            summary['text_length'] = len(text)
            summary['text_preview'] = text[:200] if len(text) > 200 else text
        
        if 'data_type' in input_data:
            summary['data_type'] = input_data['data_type']
        
        # Include other scalar fields
        for key, value in input_data.items():
            if isinstance(value, (str, int, float, bool)):
                summary[key] = value
        
        return summary
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics on collected feedback"""
        if not self.feedback_entries:
            return {'total_feedback': 0}
        
        # Count by type
        type_counts = defaultdict(int)
        for entry in self.feedback_entries:
            type_counts[entry.feedback_type.value] += 1
        
        # Calculate average rating
        ratings = [e.rating for e in self.feedback_entries if e.rating is not None]
        avg_rating = statistics.mean(ratings) if ratings else None
        
        # Component performance from feedback
        component_feedback = defaultdict(lambda: {'positive': 0, 'negative': 0})
        for entry in self.feedback_entries:
            is_positive = entry.feedback_type in [FeedbackType.SUCCESS]
            for comp in entry.components_used:
                if is_positive:
                    component_feedback[comp]['positive'] += 1
                else:
                    component_feedback[comp]['negative'] += 1
        
        return {
            'total_feedback': len(self.feedback_entries),
            'by_type': dict(type_counts),
            'average_rating': avg_rating,
            'component_feedback': dict(component_feedback),
            'recent_feedback': len([
                e for e in self.feedback_entries
                if (datetime.now(timezone.utc) - datetime.fromisoformat(e.timestamp)).days < 7
            ])
        }
    
    def _save_feedback(self):
        """Persist feedback to storage"""
        try:
            data = {
                'entries': [
                    {
                        'feedback_id': e.feedback_id,
                        'timestamp': e.timestamp,
                        'correlation_id': e.correlation_id,
                        'feedback_type': e.feedback_type.value,
                        'rating': e.rating,
                        'components_used': e.components_used,
                        'specific_issues': e.specific_issues,
                        'suggestions': e.suggestions
                    }
                    for e in self.feedback_entries
                ]
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def _load_feedback(self):
        """Load feedback from storage"""
        try:
            import os
            if not os.path.exists(self.storage_path):
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct entries (simplified)
            for entry_data in data.get('entries', []):
                entry = FeedbackEntry(
                    feedback_id=entry_data['feedback_id'],
                    timestamp=entry_data['timestamp'],
                    correlation_id=entry_data['correlation_id'],
                    input_data_summary={},
                    components_used=entry_data.get('components_used', []),
                    pipeline_config={},
                    feedback_type=FeedbackType(entry_data['feedback_type']),
                    rating=entry_data.get('rating'),
                    specific_issues=entry_data.get('specific_issues', []),
                    suggestions=entry_data.get('suggestions', [])
                )
                self.feedback_entries.append(entry)
                
        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")


class ContinuousImprovementEngine:
    """
    Engine for continuous improvement based on feedback.
    """
    
    def __init__(self, feedback_collector: FeedbackCollector, 
                 learning_engine=None):
        """
        Initialize improvement engine.
        
        Args:
            feedback_collector: Feedback collector instance
            learning_engine: Optional learning engine
        """
        self.feedback_collector = feedback_collector
        self.learning_engine = learning_engine
        
        # Active experiments
        self.experiments: List[ImprovementExperiment] = []
        
        # Improvement history
        self.improvements_applied = []
        
        logger.info({
            "msg": "ContinuousImprovementEngine initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def analyze_feedback(self) -> List[Dict[str, Any]]:
        """
        Analyze feedback to identify improvement opportunities.
        
        Returns:
            List of improvement recommendations
        """
        stats = self.feedback_collector.get_feedback_stats()
        recommendations = []
        
        # Check for problematic components
        component_feedback = stats.get('component_feedback', {})
        for comp, counts in component_feedback.items():
            total = counts['positive'] + counts['negative']
            if total > 5:  # Need enough data
                success_rate = counts['positive'] / total
                if success_rate < 0.6:
                    recommendations.append({
                        'area': ImprovementArea.COMPONENT_SELECTION,
                        'priority': 'high',
                        'issue': f'Component {comp} has low success rate: {success_rate:.2%}',
                        'suggestion': f'Consider alternatives or add gap fillers for {comp}',
                        'confidence': min(total / 20, 1.0)
                    })
        
        # Check for quality issues
        type_counts = stats.get('by_type', {})
        quality_issues = type_counts.get(FeedbackType.QUALITY_ISSUE.value, 0)
        if quality_issues > 3:
            recommendations.append({
                'area': ImprovementArea.CROSS_VALIDATION,
                'priority': 'medium',
                'issue': f'{quality_issues} quality issues reported',
                'suggestion': 'Strengthen cross-validation between components',
                'confidence': min(quality_issues / 10, 1.0)
            })
        
        # Check for performance issues
        perf_issues = type_counts.get(FeedbackType.PERFORMANCE_ISSUE.value, 0)
        if perf_issues > 2:
            recommendations.append({
                'area': ImprovementArea.PIPELINE_CONFIGURATION,
                'priority': 'medium',
                'issue': f'{perf_issues} performance issues reported',
                'suggestion': 'Optimize pipeline configuration and reduce component count',
                'confidence': min(perf_issues / 5, 1.0)
            })
        
        # Check average rating
        avg_rating = stats.get('average_rating')
        if avg_rating and avg_rating < 3.0:
            recommendations.append({
                'area': ImprovementArea.PIPELINE_CONFIGURATION,
                'priority': 'high',
                'issue': f'Low average rating: {avg_rating:.1f}/5',
                'suggestion': 'Major pipeline reconfiguration recommended',
                'confidence': (3.0 - avg_rating) / 3.0
            })
        
        return recommendations
    
    def create_experiment(self, improvement_area: ImprovementArea,
                         hypothesis: str,
                         control_config: Dict[str, Any],
                         treatment_config: Dict[str, Any]) -> ImprovementExperiment:
        """
        Create an A/B test experiment.
        
        Args:
            improvement_area: Area being improved
            hypothesis: Hypothesis being tested
            control_config: Control configuration
            treatment_config: Treatment configuration
            
        Returns:
            ImprovementExperiment
        """
        experiment = ImprovementExperiment(
            experiment_id=f"exp_{len(self.experiments)}",
            created_at=datetime.now(timezone.utc).isoformat(),
            improvement_area=improvement_area,
            hypothesis=hypothesis,
            control_config=control_config,
            treatment_config=treatment_config
        )
        
        self.experiments.append(experiment)
        
        logger.info({
            "msg": "Experiment created",
            "experiment_id": experiment.experiment_id,
            "area": improvement_area.value,
            "hypothesis": hypothesis
        })
        
        return experiment
    
    def record_experiment_result(self, experiment_id: str,
                                 variant: str,  # 'control' or 'treatment'
                                 result: Dict[str, Any]):
        """
        Record a result for an experiment.
        
        Args:
            experiment_id: Experiment ID
            variant: Which variant (control/treatment)
            result: Result data
        """
        experiment = None
        for exp in self.experiments:
            if exp.experiment_id == experiment_id:
                experiment = exp
                break
        
        if not experiment:
            logger.error(f"Experiment {experiment_id} not found")
            return
        
        if variant == 'control':
            experiment.control_results.append(result)
        elif variant == 'treatment':
            experiment.treatment_results.append(result)
        
        # Check if we have enough data to determine winner
        self._evaluate_experiment(experiment)
    
    def _evaluate_experiment(self, experiment: ImprovementExperiment):
        """Evaluate experiment results to determine winner"""
        min_samples = 10
        
        if (len(experiment.control_results) < min_samples or
            len(experiment.treatment_results) < min_samples):
            return  # Not enough data
        
        # Calculate success rates
        control_success = sum(1 for r in experiment.control_results if r.get('success'))
        treatment_success = sum(1 for r in experiment.treatment_results if r.get('success'))
        
        control_rate = control_success / len(experiment.control_results)
        treatment_rate = treatment_success / len(experiment.treatment_results)
        
        # Determine winner
        if treatment_rate > control_rate + 0.1:  # 10% improvement
            experiment.winner = 'treatment'
            experiment.confidence = min(abs(treatment_rate - control_rate) * 5, 1.0)
        elif control_rate > treatment_rate + 0.1:
            experiment.winner = 'control'
            experiment.confidence = min(abs(control_rate - treatment_rate) * 5, 1.0)
        else:
            experiment.winner = 'tie'
            experiment.confidence = 0.5
        
        experiment.status = 'completed'
        
        logger.info({
            "msg": "Experiment completed",
            "experiment_id": experiment.experiment_id,
            "winner": experiment.winner,
            "confidence": experiment.confidence
        })
    
    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Get current improvement recommendations"""
        recommendations = self.analyze_feedback()
        
        # Add recommendations from completed experiments
        for exp in self.experiments:
            if exp.status == 'completed' and exp.winner == 'treatment':
                recommendations.append({
                    'area': exp.improvement_area,
                    'priority': 'high' if exp.confidence > 0.8 else 'medium',
                    'issue': exp.hypothesis,
                    'suggestion': 'Apply treatment configuration',
                    'confidence': exp.confidence,
                    'source': f"experiment_{exp.experiment_id}"
                })
        
        # Sort by priority and confidence
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: (priority_order.get(x['priority'], 3), -x['confidence']))
        
        return recommendations
    
    def apply_improvement(self, recommendation: Dict[str, Any]) -> bool:
        """
        Apply an improvement recommendation.
        
        Args:
            recommendation: Recommendation to apply
            
        Returns:
            True if applied successfully
        """
        logger.info({
            "msg": "Applying improvement",
            'area': recommendation['area'].value,
            'suggestion': recommendation['suggestion']
        })
        
        # Record the improvement
        self.improvements_applied.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'recommendation': recommendation,
            'applied': True
        })
        
        return True
    
    def get_improvement_report(self) -> Dict[str, Any]:
        """Get report on improvement activities"""
        return {
            'total_experiments': len(self.experiments),
            'active_experiments': len([e for e in self.experiments if e.status == 'running']),
            'completed_experiments': len([e for e in self.experiments if e.status == 'completed']),
            'improvements_applied': len(self.improvements_applied),
            'pending_recommendations': len(self.get_improvement_recommendations()),
            'recent_improvements': self.improvements_applied[-10:] if self.improvements_applied else []
        }


class AdaptiveOrchestratorIntegration:
    """
    Integrates feedback and improvement systems with the orchestrator.
    """
    
    def __init__(self, orchestrator, feedback_collector: FeedbackCollector,
                 improvement_engine: ContinuousImprovementEngine):
        """
        Initialize integration.
        
        Args:
            orchestrator: The orchestrator to enhance
            feedback_collector: Feedback collector
            improvement_engine: Improvement engine
        """
        self.orchestrator = orchestrator
        self.feedback_collector = feedback_collector
        self.improvement_engine = improvement_engine
        
        # Enable automatic improvement
        self.auto_improve = True
        self.improvement_interval = 100  # Apply improvements every N executions
        self.execution_count = 0
    
    def process_with_feedback(self, input_data: Dict[str, Any],
                             custom_config: Optional[Dict[str, Any]] = None,
                             collect_user_feedback: bool = False) -> Dict[str, Any]:
        """
        Process with automatic feedback collection and improvement.
        
        Args:
            input_data: Input data
            custom_config: Custom configuration
            collect_user_feedback: Whether to collect user feedback
            
        Returns:
            Processing result
        """
        self.execution_count += 1
        
        # Apply any pending improvements
        if self.auto_improve and self.execution_count % self.improvement_interval == 0:
            self._apply_pending_improvements()
        
        # Execute
        result = self.orchestrator.process(input_data, custom_config)
        
        # Auto-collect feedback based on result
        self._auto_collect_feedback(input_data, result)
        
        # Add feedback collection mechanism if requested
        if collect_user_feedback:
            result['feedback_url'] = f"/feedback/{result.get('correlation_id')}"
            result['feedback_request'] = 'Please rate this result (1-5) and provide any corrections'
        
        return result
    
    def _auto_collect_feedback(self, input_data: Dict[str, Any], result: Dict[str, Any]):
        """Automatically collect feedback based on execution result"""
        # Determine feedback type from result
        status = result.get('status')
        
        if status == 'success':
            feedback_type = FeedbackType.SUCCESS
        elif status == 'partial':
            feedback_type = FeedbackType.PARTIAL_SUCCESS
        else:
            feedback_type = FeedbackType.FAILURE
        
        # Check for quality issues
        issues = []
        if result.get('execution', {}).get('stages_failed', 0) > 0:
            issues.append('Some pipeline stages failed')
            feedback_type = FeedbackType.QUALITY_ISSUE
        
        # Check execution time
        execution_time = result.get('execution', {}).get('duration_ms', 0)
        if execution_time > 60000:  # More than 1 minute
            issues.append('Slow execution')
            if feedback_type == FeedbackType.SUCCESS:
                feedback_type = FeedbackType.PERFORMANCE_ISSUE
        
        # Collect feedback
        self.feedback_collector.collect_feedback(
            correlation_id=result.get('correlation_id', 'unknown'),
            input_data=input_data,
            components_used=[c.value for c in self.orchestrator.components.keys()],
            pipeline_config=self.orchestrator.config.to_dict(),
            feedback_type=feedback_type,
            issues=issues,
            actual_output=result.get('results')
        )
    
    def _apply_pending_improvements(self):
        """Apply pending improvement recommendations"""
        recommendations = self.improvement_engine.get_improvement_recommendations()
        
        # Apply high-confidence recommendations
        for rec in recommendations:
            if rec['priority'] == 'high' and rec['confidence'] > 0.7:
                self.improvement_engine.apply_improvement(rec)
    
    def submit_user_feedback(self, correlation_id: str, rating: int,
                           issues: List[str] = None,
                           suggestions: List[str] = None,
                           expected_output: Dict[str, Any] = None):
        """
        Submit user feedback for a specific execution.
        
        Args:
            correlation_id: Execution correlation ID
            rating: 1-5 rating
            issues: Specific issues
            suggestions: User suggestions
            expected_output: What was expected
        """
        # Find the execution in feedback history
        for entry in self.feedback_collector.feedback_entries:
            if entry.correlation_id == correlation_id:
                # Update entry with user feedback
                entry.rating = rating
                if issues:
                    entry.specific_issues.extend(issues)
                if suggestions:
                    entry.suggestions.extend(suggestions)
                if expected_output:
                    entry.expected_output = expected_output
                
                logger.info({
                    "msg": "User feedback received",
                    "correlation_id": correlation_id,
                    "rating": rating
                })
                
                # Trigger improvement analysis
                self.improvement_engine.analyze_feedback()
                
                return
        
        logger.warning(f"Execution {correlation_id} not found for feedback")
    
    def get_system_report(self) -> Dict[str, Any]:
        """Get comprehensive system report"""
        return {
            'executions': self.execution_count,
            'feedback': self.feedback_collector.get_feedback_stats(),
            'improvements': self.improvement_engine.get_improvement_report(),
            'recommendations': len(self.improvement_engine.get_improvement_recommendations())
        }


# Convenience functions
def create_adaptive_orchestrator(orchestrator, storage_path: Optional[str] = None):
    """
    Wrap an orchestrator with adaptive feedback capabilities.
    
    Args:
        orchestrator: Base orchestrator
        storage_path: Path for feedback storage
        
    Returns:
        AdaptiveOrchestratorIntegration
    """
    feedback_collector = FeedbackCollector(storage_path)
    improvement_engine = ContinuousImprovementEngine(feedback_collector)
    
    return AdaptiveOrchestratorIntegration(
        orchestrator, feedback_collector, improvement_engine
    )
