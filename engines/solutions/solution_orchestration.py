"""
Solution Orchestration Module with ICR Integration

This module orchestrates solution attempts with ICR-based quality prediction
and gauntlet outcome correlation.

ICR Integration:
- Stores solution quality patterns
- Predicts gauntlet pass probability before submission
- Recommends refinements based on ICR patterns
- Identifies high-risk solutions early
"""
from __future__ import annotations


from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, field
import logging

# ICR Integration
try:
    from icr_integration import get_icr_integration, ICRPatternType, ICRIntegration
    ICR_AVAILABLE = True
except ImportError:
    ICR_AVAILABLE = False
    get_icr_integration = None
    ICRPatternType = None
    ICRIntegration = None

logger = logging.getLogger(__name__)


@dataclass
class SolutionAttempt:
    """Represents a solution attempt."""
    solution_id: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GauntletResult:
    """Result from gauntlet validation."""
    passed: bool
    score: float
    feedback: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SolutionOrchestrator:
    """
    Orchestrates solution attempts with ICR-based quality prediction.
    
    ICR Integration:
    - Stores solution quality patterns for learning
    - Predicts gauntlet pass probability before submission
    - Recommends refinements based on ICR patterns
    - Identifies high-risk solutions early
    """

    def __init__(self, enable_icr: bool = True, enable_gauntlet: bool = True):
        """
        Initialize solution orchestrator.
        
        Args:
            enable_icr: Enable ICR pattern learning
            enable_gauntlet: Enable gauntlet outcome tracking
        """
        self.enable_icr = enable_icr and ICR_AVAILABLE
        self.enable_gauntlet = enable_gauntlet
        
        self.icr = None
        self.solution_history: List[Dict[str, Any]] = []
        self.gauntlet_correlations: Dict[str, List[Dict[str, Any]]] = {}
        
        if self.enable_icr:
            try:
                self.icr = get_icr_integration()
                if self.icr:
                    self.icr.enable()
            except Exception as e:
                logger.warning(f"Failed to initialize ICR integration: {e}")
                self.enable_icr = False
                self.icr = None

    def submit_solution(
        self,
        solution: SolutionAttempt,
        content_type: str = "code",
        complexity_score: int = 5
    ) -> Dict[str, Any]:
        """
        Submit a solution with ICR-based quality prediction.
        
        Args:
            solution: Solution attempt
            content_type: Type of content (code, text, design, etc.)
            complexity_score: Complexity score (1-10)
            
        Returns:
            Submission result with quality prediction
        """
        result = {
            "solution_id": solution.solution_id,
            "submitted": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Get ICR prediction if available
        if self.enable_icr and self.icr:
            prediction = self.predict_solution_quality(
                solution=solution,
                content_type=content_type,
                complexity_score=complexity_score
            )
            result["icr_prediction"] = prediction
            
            # Add warning if high risk
            if prediction.get("predicted_outcome") == "fail" and prediction.get("confidence", 0) > 0.7:
                result["warning"] = "High-risk solution detected - consider additional refinement"
                result["recommended_action"] = prediction.get("recommended_action")
        
        # Store in history
        self.solution_history.append({
            "solution_id": solution.solution_id,
            "content_type": content_type,
            "complexity_score": complexity_score,
            "timestamp": result["timestamp"]
        })
        
        return result

    def record_gauntlet_outcome(
        self,
        solution_id: str,
        result: GauntletResult,
        solution_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record gauntlet outcome for correlation learning.
        
        Args:
            solution_id: ID of the solution
            result: Gauntlet result
            solution_metadata: Optional solution metadata
            
        Returns:
            Pattern ID if stored, empty string if ICR not available
        """
        if not self.enable_icr or not self.icr:
            return ""
        
        # Store ICR pattern
        pattern_id = self.icr.store_pattern(
            pattern_type=ICRPatternType.GAUNTLET_OUTCOME,
            passed=result.passed,
            context={
                "solution_id": solution_id,
                "content_type": solution_metadata.get("content_type", "unknown") if solution_metadata else "unknown",
                "complexity_score": solution_metadata.get("complexity_score", 5) if solution_metadata else 5
            },
            metrics={
                "gauntlet_score": result.score,
                "feedback_count": len(result.feedback)
            }
        )
        
        # Store in gauntlet correlations
        if solution_id not in self.gauntlet_correlations:
            self.gauntlet_correlations[solution_id] = []
        
        self.gauntlet_correlations[solution_id].append({
            "result": result.passed,
            "score": result.score,
            "feedback": result.feedback,
            "timestamp": result.timestamp.isoformat(),
            "pattern_id": pattern_id
        })
        
        return pattern_id

    def predict_solution_quality(
        self,
        solution: SolutionAttempt,
        content_type: str = "code",
        complexity_score: int = 5
    ) -> Dict[str, Any]:
        """
        Predict solution quality and gauntlet pass probability.
        
        Args:
            solution: Solution attempt
            content_type: Type of content
            complexity_score: Complexity score (1-10)
            
        Returns:
            Prediction results with confidence and recommendations
        """
        if not self.enable_icr or not self.icr:
            return {
                "predicted": False,
                "reason": "ICR integration not available"
            }
        
        try:
            prediction = self.icr.predict(
                pattern_type=ICRPatternType.QUALITY_OUTCOME,
                context={
                    "content_type": content_type,
                    "complexity_score": complexity_score
                }
            )
            
            result = {
                "predicted": True,
                "predicted_outcome": prediction.predicted_outcome,
                "probability": prediction.probability,
                "confidence": prediction.confidence,
                "reason": prediction.reason,
                "pattern_count": prediction.pattern_count,
                "recommended_action": prediction.recommended_action
            }
            
            # Add refinement recommendations based on prediction
            if prediction.predicted_outcome == "fail" and prediction.confidence > 0.6:
                result["refinement_needed"] = True
                result["refinement_suggestions"] = [
                    "Review solution for edge cases",
                    "Add additional validation",
                    "Consider alternative approaches",
                    "Request peer review before submission"
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"ICR prediction failed: {e}")
            return {
                "predicted": False,
                "reason": f"Prediction error: {str(e)}"
            }

    def get_solution_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about solution attempts and gauntlet outcomes.
        
        Returns:
            Dictionary with solution statistics
        """
        if not self.solution_history:
            return {
                "total_solutions": 0,
                "message": "No solutions submitted yet"
            }
        
        total = len(self.solution_history)
        
        # Count gauntlet outcomes
        total_gauntlet_runs = sum(
            len(results) for results in self.gauntlet_correlations.values()
        )
        passed_gauntlets = sum(
            sum(1 for r in results if r.get("result", False))
            for results in self.gauntlet_correlations.values()
        )
        
        return {
            "total_solutions": total,
            "total_gauntlet_runs": total_gauntlet_runs,
            "passed_gauntlets": passed_gauntlets,
            "failed_gauntlets": total_gauntlet_runs - passed_gauntlets,
            "gauntlet_pass_rate": passed_gauntlets / total_gauntlet_runs if total_gauntlet_runs > 0 else 0.0,
            "icr_enabled": self.enable_icr,
            "gauntlet_enabled": self.enable_gauntlet
        }

    def recommend_refinements(
        self,
        solution: SolutionAttempt,
        gauntlet_feedback: List[str]
    ) -> Dict[str, Any]:
        """
        Recommend refinements based on gauntlet feedback and ICR patterns.
        
        Args:
            solution: Solution attempt
            gauntlet_feedback: Feedback from gauntlet validation
            
        Returns:
            Refinement recommendations
        """
        recommendations = {
            "solution_id": solution.solution_id,
            "recommendations": [],
            "priority": "normal"
        }
        
        # Analyze gauntlet feedback
        if gauntlet_feedback:
            if any("error" in f.lower() for f in gauntlet_feedback):
                recommendations["recommendations"].append("Fix identified errors before resubmission")
                recommendations["priority"] = "high"
            
            if any("performance" in f.lower() for f in gauntlet_feedback):
                recommendations["recommendations"].append("Optimize performance bottlenecks")
            
            if any("security" in f.lower() for f in gauntlet_feedback):
                recommendations["recommendations"].append("Address security vulnerabilities")
                recommendations["priority"] = "critical"
        
        # Add ICR-based recommendations if available
        if self.enable_icr and self.icr:
            # Check historical patterns for similar solutions
            stats = self.get_solution_statistics()
            if stats.get("gauntlet_pass_rate", 0.5) < 0.7:
                recommendations["recommendations"].append(
                    "Consider additional testing - historical pass rate is below 70%"
                )
        
        return recommendations
