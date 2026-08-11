"""
Process Optimization Module with ICR Integration

This module analyzes workflow execution and provides optimization recommendations
to improve efficiency, reduce costs, and enhance quality.

ICR Integration:
- Stores optimization patterns for learning
- Predicts optimization success probability
- Adapts recommendations based on historical outcomes
- Learns from workflow execution results
"""


from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import statistics

from workflow_structures import WorkflowState, SubProblem, DecompositionPlan

# ICR Integration
try:
    from icr_integration import (
        get_icr_integration,
        ICRPatternType,
        ICRIntegration
    )
    ICR_AVAILABLE = True
except ImportError:
    ICR_AVAILABLE = False
    get_icr_integration = None
    ICRPatternType = None
    ICRIntegration = None


class ProcessOptimizer:
    """Analyzes workflows and provides optimization recommendations.
    
    ICR Integration:
    - Stores optimization patterns for learning
    - Predicts optimization impact based on historical data
    - Adapts recommendations using ICR patterns
    """

    def __init__(self, enable_icr: bool = True):
        """
        Initialize the process optimizer.
        
        Args:
            enable_icr: Enable ICR pattern learning and prediction
        """
        self.workflow_history: List[WorkflowState] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # ICR Integration
        self.enable_icr = enable_icr and ICR_AVAILABLE
        self.icr = get_icr_integration() if self.enable_icr else None
        
        if self.enable_icr and self.icr:
            self.icr.enable()
    
    def analyze_workflow(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """
        Analyze a completed workflow and identify optimization opportunities.
        
        Args:
            workflow_state: Completed workflow state
            
        Returns:
            Analysis results with recommendations
        """
        analysis = {
            "workflow_id": workflow_state.workflow_id,
            "timestamp": datetime.now().isoformat(),
            "bottlenecks": self._identify_bottlenecks(workflow_state),
            "inefficiencies": self._identify_inefficiencies(workflow_state),
            "cost_optimization": self._analyze_cost_optimization(workflow_state),
            "quality_improvements": self._analyze_quality_improvements(workflow_state),
            "recommendations": []
        }
        
        # Generate recommendations based on analysis
        analysis["recommendations"] = self._generate_recommendations(analysis, workflow_state)
        
        # Store for historical analysis
        self.workflow_history.append(workflow_state)
        self.optimization_history.append(analysis)
        
        return analysis
    
    def _identify_bottlenecks(self, workflow_state: WorkflowState) -> List[Dict[str, Any]]:
        """Identify bottlenecks in workflow execution."""
        bottlenecks = []
        
        # Analyze sub-problem solving times
        if workflow_state.decomposition_plan:
            sub_problem_times = {}
            
            for sp in workflow_state.decomposition_plan.sub_problems:
                if sp.id in workflow_state.sub_problem_solutions:
                    solution = workflow_state.sub_problem_solutions[sp.id]
                    # Estimate time based on attempts and complexity
                    estimated_time = len(solution.critique_reports) * 60  # Rough estimate
                    sub_problem_times[sp.id] = estimated_time
            
            if sub_problem_times:
                avg_time = statistics.mean(sub_problem_times.values())
                
                # Identify sub-problems taking significantly longer
                for sp_id, time in sub_problem_times.items():
                    if time > avg_time * 2:
                        bottlenecks.append({
                            "type": "slow_sub_problem",
                            "sub_problem_id": sp_id,
                            "time": time,
                            "avg_time": avg_time,
                            "severity": "high" if time > avg_time * 3 else "medium"
                        })
        
        # Analyze refinement loops
        if workflow_state.refinement_loop_count > 2:
            bottlenecks.append({
                "type": "excessive_refinement",
                "loop_count": workflow_state.refinement_loop_count,
                "severity": "high" if workflow_state.refinement_loop_count > 5 else "medium"
            })
        
        # Analyze rejected sub-problems
        if workflow_state.rejected_sub_problems:
            bottlenecks.append({
                "type": "rejected_sub_problems",
                "count": len(workflow_state.rejected_sub_problems),
                "sub_problems": list(workflow_state.rejected_sub_problems.keys()),
                "severity": "high" if len(workflow_state.rejected_sub_problems) > 3 else "medium"
            })
        
        return bottlenecks
    
    def _identify_inefficiencies(self, workflow_state: WorkflowState) -> List[Dict[str, Any]]:
        """Identify inefficiencies in workflow execution."""
        inefficiencies = []
        
        # Check for redundant gauntlet runs
        gauntlet_runs = {}
        for critique in workflow_state.all_critique_reports:
            gauntlet_name = critique.gauntlet_name
            gauntlet_runs[gauntlet_name] = gauntlet_runs.get(gauntlet_name, 0) + 1
        
        for gauntlet_name, count in gauntlet_runs.items():
            if count > 10:
                inefficiencies.append({
                    "type": "excessive_gauntlet_runs",
                    "gauntlet_name": gauntlet_name,
                    "run_count": count,
                    "severity": "medium"
                })
        
        # Check for over-decomposition
        if workflow_state.decomposition_plan:
            num_sub_problems = len(workflow_state.decomposition_plan.sub_problems)
            if num_sub_problems > 20:
                inefficiencies.append({
                    "type": "over_decomposition",
                    "sub_problem_count": num_sub_problems,
                    "severity": "medium"
                })
            elif num_sub_problems < 2:
                inefficiencies.append({
                    "type": "under_decomposition",
                    "sub_problem_count": num_sub_problems,
                    "severity": "low"
                })
        
        # Check for unused teams
        if workflow_state.decomposition_plan:
            teams_used = set()
            for sp in workflow_state.decomposition_plan.sub_problems:
                if sp.solver_team_name:
                    teams_used.add(sp.solver_team_name)
            
            if len(teams_used) == 1 and len(workflow_state.decomposition_plan.sub_problems) > 5:
                inefficiencies.append({
                    "type": "single_team_overuse",
                    "team_name": list(teams_used)[0],
                    "severity": "low"
                })
        
        return inefficiencies
    
    def _analyze_cost_optimization(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Analyze cost optimization opportunities."""
        cost_analysis = {
            "total_api_calls": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "optimization_potential": []
        }
        
        # Estimate API calls and tokens
        if workflow_state.decomposition_plan:
            # Rough estimates
            num_sub_problems = len(workflow_state.decomposition_plan.sub_problems)
            num_critiques = len(workflow_state.all_critique_reports)
            num_verifications = len(workflow_state.all_verification_reports)
            
            estimated_api_calls = (
                1 +  # Content analysis
                1 +  # Decomposition
                num_sub_problems * 2 +  # Solution generation
                num_critiques +
                num_verifications +
                workflow_state.refinement_loop_count * 2
            )
            
            estimated_tokens = estimated_api_calls * 1500  # Rough average
            estimated_cost = estimated_tokens * 0.00002  # Rough GPT-4 pricing
            
            cost_analysis["total_api_calls"] = estimated_api_calls
            cost_analysis["total_tokens"] = estimated_tokens
            cost_analysis["estimated_cost"] = estimated_cost
        
        # Identify optimization opportunities
        if workflow_state.refinement_loop_count > 3:
            potential_savings = workflow_state.refinement_loop_count * 2 * 1500 * 0.00002
            cost_analysis["optimization_potential"].append({
                "type": "reduce_refinement_loops",
                "potential_savings": potential_savings,
                "description": "Reduce refinement loops by improving initial solution quality"
            })
        
        if len(workflow_state.all_critique_reports) > 20:
            potential_savings = (len(workflow_state.all_critique_reports) - 20) * 1500 * 0.00002
            cost_analysis["optimization_potential"].append({
                "type": "optimize_gauntlet_runs",
                "potential_savings": potential_savings,
                "description": "Optimize gauntlet configurations to reduce unnecessary runs"
            })
        
        return cost_analysis
    
    def _analyze_quality_improvements(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Analyze quality improvement opportunities."""
        quality_analysis = {
            "success_rate": 0.0,
            "avg_refinement_loops": workflow_state.refinement_loop_count,
            "improvement_opportunities": []
        }
        
        # Calculate success rate
        if workflow_state.decomposition_plan:
            total_sub_problems = len(workflow_state.decomposition_plan.sub_problems)
            solved_sub_problems = len(workflow_state.solved_sub_problem_ids)
            quality_analysis["success_rate"] = solved_sub_problems / total_sub_problems if total_sub_problems > 0 else 0.0
        
        # Identify improvement opportunities
        if quality_analysis["success_rate"] < 0.8:
            quality_analysis["improvement_opportunities"].append({
                "type": "low_success_rate",
                "current_rate": quality_analysis["success_rate"],
                "target_rate": 0.9,
                "suggestions": [
                    "Review team configurations",
                    "Adjust gauntlet strictness",
                    "Improve decomposition strategy"
                ]
            })
        
        if workflow_state.refinement_loop_count > 3:
            quality_analysis["improvement_opportunities"].append({
                "type": "excessive_refinement",
                "current_loops": workflow_state.refinement_loop_count,
                "target_loops": 2,
                "suggestions": [
                    "Improve initial solution generation",
                    "Use better solver teams",
                    "Refine gauntlet criteria"
                ]
            })
        
        # Analyze critique patterns
        common_flaws = {}
        for critique in workflow_state.all_critique_reports:
            for flaw in critique.identified_flaws:
                flaw_type = flaw.get("type", "unknown")
                common_flaws[flaw_type] = common_flaws.get(flaw_type, 0) + 1
        
        if common_flaws:
            most_common = max(common_flaws.items(), key=lambda x: x[1])
            if most_common[1] > 3:
                quality_analysis["improvement_opportunities"].append({
                    "type": "recurring_flaw",
                    "flaw_type": most_common[0],
                    "occurrence_count": most_common[1],
                    "suggestions": [
                        f"Add specific checks for {most_common[0]} in gauntlets",
                        f"Train teams to avoid {most_common[0]} issues",
                        "Update system prompts to address this pattern"
                    ]
                })
        
        return quality_analysis
    
    def _generate_recommendations(
        self,
        analysis: Dict[str, Any],
        workflow_state: WorkflowState
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Recommendations for bottlenecks
        for bottleneck in analysis["bottlenecks"]:
            if bottleneck["type"] == "slow_sub_problem":
                recommendations.append({
                    "priority": "high",
                    "category": "performance",
                    "title": f"Optimize slow sub-problem {bottleneck['sub_problem_id']}",
                    "description": f"This sub-problem took {bottleneck['time']}s, which is {bottleneck['time']/bottleneck['avg_time']:.1f}x the average.",
                    "actions": [
                        "Review sub-problem complexity",
                        "Consider breaking it into smaller sub-problems",
                        "Use a more capable solver team",
                        "Adjust gauntlet strictness"
                    ]
                })
            
            elif bottleneck["type"] == "excessive_refinement":
                recommendations.append({
                    "priority": "high",
                    "category": "quality",
                    "title": "Reduce refinement loops",
                    "description": f"Workflow required {bottleneck['loop_count']} refinement loops.",
                    "actions": [
                        "Improve initial solution quality",
                        "Review and adjust gauntlet criteria",
                        "Use more capable solver teams",
                        "Add targeted feedback parsing"
                    ]
                })
        
        # Recommendations for inefficiencies
        for inefficiency in analysis["inefficiencies"]:
            if inefficiency["type"] == "excessive_gauntlet_runs":
                recommendations.append({
                    "priority": "medium",
                    "category": "cost",
                    "title": f"Optimize {inefficiency['gauntlet_name']} gauntlet",
                    "description": f"This gauntlet ran {inefficiency['run_count']} times.",
                    "actions": [
                        "Review gauntlet configuration",
                        "Adjust strictness to reduce false negatives",
                        "Consider caching gauntlet results",
                        "Implement early stopping criteria"
                    ]
                })
            
            elif inefficiency["type"] == "over_decomposition":
                recommendations.append({
                    "priority": "medium",
                    "category": "efficiency",
                    "title": "Simplify decomposition",
                    "description": f"Problem was decomposed into {inefficiency['sub_problem_count']} sub-problems.",
                    "actions": [
                        "Review decomposition strategy",
                        "Combine related sub-problems",
                        "Adjust planner team prompts",
                        "Set maximum sub-problem limits"
                    ]
                })
        
        # Recommendations for cost optimization
        for opportunity in analysis["cost_optimization"]["optimization_potential"]:
            recommendations.append({
                "priority": "medium",
                "category": "cost",
                "title": opportunity["type"].replace("_", " ").title(),
                "description": opportunity["description"],
                "potential_savings": f"${opportunity['potential_savings']:.2f}",
                "actions": [
                    "Implement the suggested optimization",
                    "Monitor cost metrics",
                    "Set cost limits"
                ]
            })
        
        # Recommendations for quality improvements
        for opportunity in analysis["quality_improvements"]["improvement_opportunities"]:
            recommendations.append({
                "priority": "high" if opportunity["type"] == "low_success_rate" else "medium",
                "category": "quality",
                "title": opportunity["type"].replace("_", " ").title(),
                "description": f"Current: {opportunity.get('current_rate', opportunity.get('current_loops', 'N/A'))}, Target: {opportunity.get('target_rate', opportunity.get('target_loops', 'N/A'))}",
                "actions": opportunity["suggestions"]
            })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return recommendations
    
    def get_historical_trends(self) -> Dict[str, Any]:
        """Analyze trends across multiple workflows."""
        if len(self.workflow_history) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        trends = {
            "total_workflows": len(self.workflow_history),
            "avg_refinement_loops": statistics.mean([
                w.refinement_loop_count for w in self.workflow_history
            ]),
            "avg_sub_problems": statistics.mean([
                len(w.decomposition_plan.sub_problems) if w.decomposition_plan else 0
                for w in self.workflow_history
            ]),
            "success_rate_trend": self._calculate_success_rate_trend(),
            "cost_trend": self._calculate_cost_trend(),
            "common_bottlenecks": self._identify_common_bottlenecks()
        }
        
        return trends
    
    def _calculate_success_rate_trend(self) -> List[float]:
        """Calculate success rate trend over time."""
        rates = []
        for workflow in self.workflow_history:
            if workflow.decomposition_plan:
                total = len(workflow.decomposition_plan.sub_problems)
                solved = len(workflow.solved_sub_problem_ids)
                rates.append(solved / total if total > 0 else 0.0)
        return rates
    
    def _calculate_cost_trend(self) -> List[float]:
        """Calculate estimated cost trend over time."""
        costs = []
        for analysis in self.optimization_history:
            if "cost_optimization" in analysis:
                costs.append(analysis["cost_optimization"]["estimated_cost"])
        return costs
    
    def _identify_common_bottlenecks(self) -> Dict[str, int]:
        """Identify bottlenecks that appear across multiple workflows."""
        bottleneck_counts = {}
        for analysis in self.optimization_history:
            for bottleneck in analysis.get("bottlenecks", []):
                b_type = bottleneck["type"]
                bottleneck_counts[b_type] = bottleneck_counts.get(b_type, 0) + 1
        return bottleneck_counts
    
    def generate_optimization_report(self, workflow_state: WorkflowState) -> str:
        """Generate a human-readable optimization report."""
        analysis = self.analyze_workflow(workflow_state)
        
        report = []
        report.append("="*60)
        report.append("WORKFLOW OPTIMIZATION REPORT")
        report.append("="*60)
        report.append(f"Workflow ID: {analysis['workflow_id']}")
        report.append(f"Analysis Date: {analysis['timestamp']}")
        report.append("")
        
        # Bottlenecks
        if analysis["bottlenecks"]:
            report.append("BOTTLENECKS IDENTIFIED:")
            report.append("-"*60)
            for b in analysis["bottlenecks"]:
                report.append(f"  [{b['severity'].upper()}] {b['type']}")
                for key, value in b.items():
                    if key not in ["type", "severity"]:
                        report.append(f"    {key}: {value}")
            report.append("")
        
        # Cost Analysis
        report.append("COST ANALYSIS:")
        report.append("-"*60)
        cost = analysis["cost_optimization"]
        report.append(f"  Estimated API Calls: {cost['total_api_calls']}")
        report.append(f"  Estimated Tokens: {cost['total_tokens']:,}")
        report.append(f"  Estimated Cost: ${cost['estimated_cost']:.2f}")
        if cost["optimization_potential"]:
            total_savings = sum(o["potential_savings"] for o in cost["optimization_potential"])
            report.append(f"  Potential Savings: ${total_savings:.2f}")
        report.append("")
        
        # Recommendations
        if analysis["recommendations"]:
            report.append("RECOMMENDATIONS:")
            report.append("-"*60)
            for i, rec in enumerate(analysis["recommendations"][:5], 1):  # Top 5
                report.append(f"{i}. [{rec['priority'].upper()}] {rec['title']}")
                report.append(f"   {rec['description']}")
                report.append(f"   Actions:")
                for action in rec["actions"][:3]:  # Top 3 actions
                    report.append(f"     - {action}")
                report.append("")
        
        report.append("="*60)
        
        return "\n".join(report)

    # =========================================================================
    # ICR INTEGRATION METHODS
    # =========================================================================

    def store_optimization_pattern(
        self,
        workflow_state: WorkflowState,
        analysis: Dict[str, Any]
    ) -> str:
        """
        Store optimization pattern for ICR learning.
        
        Args:
            workflow_state: Workflow state
            analysis: Optimization analysis results
            
        Returns:
            Pattern ID
        """
        if not self.enable_icr or not self.icr:
            return ""
        
        # Determine if optimization was successful (based on recommendations adopted)
        recommendations_count = len(analysis.get("recommendations", []))
        bottlenecks_count = len(analysis.get("bottlenecks", []))
        
        # Success = few bottlenecks and actionable recommendations
        passed = bottlenecks_count <= 2 and recommendations_count >= 1
        
        # Build context
        context = {
            "workflow_id": workflow_state.workflow_id,
            "content_type": "workflow_optimization",
            "complexity_score": min(10, len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan else 0),
            "problem_type": analysis.get("bottlenecks", [{}])[0].get("type") if analysis.get("bottlenecks") else "none",
            "refinement_loops": workflow_state.refinement_loop_count,
            "recommendations_count": recommendations_count
        }
        
        # Build metrics
        metrics = {
            "bottleneck_count": bottlenecks_count,
            "recommendation_count": recommendations_count,
            "estimated_cost": analysis.get("cost_optimization", {}).get("estimated_cost", 0.0)
        }
        
        # Store pattern
        pattern_id = self.icr.store_pattern(
            pattern_type=ICRPatternType.OPTIMIZATION,
            passed=passed,
            context=context,
            metrics=metrics
        )
        
        return pattern_id

    def predict_optimization_success(
        self,
        workflow_state: WorkflowState
    ) -> Dict[str, Any]:
        """
        Predict optimization success based on ICR patterns.
        
        Args:
            workflow_state: Current workflow state
            
        Returns:
            Prediction results with confidence and recommendations
        """
        if not self.enable_icr or not self.icr:
            return {
                "predicted": False,
                "reason": "ICR integration not available"
            }
        
        # Build context for prediction
        context = {
            "content_type": "workflow_optimization",
            "complexity_score": min(10, len(workflow_state.decomposition_plan.sub_problems) if workflow_state.decomposition_plan else 0),
            "problem_type": "optimization",
            "refinement_loops": workflow_state.refinement_loop_count
        }
        
        # Get prediction
        prediction = self.icr.predict(
            pattern_type=ICRPatternType.OPTIMIZATION,
            context=context
        )
        
        return {
            "predicted": True,
            "predicted_outcome": prediction.predicted_outcome,
            "probability": prediction.probability,
            "confidence": prediction.confidence,
            "reason": prediction.reason,
            "pattern_count": prediction.pattern_count,
            "recommended_action": prediction.recommended_action,
            "adaptive_threshold": self.icr.get_adaptive_threshold("optimization", 0.5)
        }

    def analyze_with_icr(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """
        Analyze workflow with ICR pattern-based insights.
        
        Args:
            workflow_state: Workflow state
            
        Returns:
            Enhanced analysis with ICR insights
        """
        # Perform standard analysis
        analysis = self.analyze_workflow(workflow_state)
        
        # Store pattern for learning
        if self.enable_icr:
            self.store_optimization_pattern(workflow_state, analysis)
        
        # Get prediction for future optimizations
        prediction = self.predict_optimization_success(workflow_state)
        
        # Add ICR insights to analysis
        analysis["icr_insights"] = {
            "enabled": self.enable_icr,
            "prediction": prediction if prediction.get("predicted") else None,
            "adaptive_threshold": self.icr.get_adaptive_threshold("optimization", 0.5) if self.enable_icr else None
        }
        
        # Use ICR patterns to enhance recommendations
        if self.enable_icr and prediction.get("predicted"):
            if prediction["predicted_outcome"] == "fail" and prediction["confidence"] > 0.7:
                # High confidence failure prediction - add urgent recommendation
                analysis["recommendations"].insert(0, {
                    "priority": "critical",
                    "title": "High Risk of Optimization Failure",
                    "description": f"ICR analysis predicts optimization failure with {prediction['probability']:.1%} probability",
                    "actions": [
                        "Consider manual review before proceeding",
                        "Break down into smaller optimization targets",
                        "Review historical patterns for similar workflows"
                    ]
                })
        
        return analysis


# Global optimizer instance
_global_optimizer: Optional[ProcessOptimizer] = None


def get_process_optimizer() -> ProcessOptimizer:
    """Get or create the global process optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = ProcessOptimizer()
    return _global_optimizer
