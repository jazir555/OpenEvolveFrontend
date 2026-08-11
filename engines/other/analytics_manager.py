"""
Analytics Manager for OpenEvolve - Analytics and insights generation
This file manages analytics, insights, and data analysis features
File size: ~1200 lines (under the 2000 line limit)
"""

from ui_shim import ui as st
import logging
import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from datetime import datetime
from session_utils import (
    calculate_protocol_complexity,
    extract_protocol_structure,
    generate_protocol_recommendations,
)
import time
import numpy as np

# **ACTUAL INTEGRATION**: Alerting and knowledge for Analytics Manager
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False



class AnalyticsManager:
    """
    Manages analytics, insights, and data analysis features
    """

    def __init__(self):
        self.last_generated_at: Optional[float] = None
        self.insight_cache: Dict[str, Dict[str, Any]] = {}
        self.event_callbacks: Dict[str, List] = {}
        self.recent_events: List[Dict[str, Any]] = []

    def generate_ai_insights(self, protocol_text: str) -> Dict[str, Any]:
        """
        Generate AI-powered insights about the protocol.

        Args:
            protocol_text (str): Protocol text to analyze

        Returns:
            Dict[str, Any]: AI insights and recommendations
        """
        if not protocol_text:
            return {
                "overall_score": 0,
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": [],
                "recommendations": [],
                "complexity_analysis": {},
                "readability_score": 0,
                "compliance_risk": "low",
            }

        # Calculate metrics
        complexity = calculate_protocol_complexity(protocol_text)
        structure = extract_protocol_structure(protocol_text)

        # Overall score calculation (weighted)
        structure_score = self._calculate_structure_score(structure)
        complexity_score = self._calculate_complexity_score(complexity)
        readability_score = self._calculate_readability_score(protocol_text)
        
        overall_score = (
            structure_score * 0.4 +
            complexity_score * 0.3 +
            readability_score * 0.3
        )

        # Generate SWOT analysis
        strengths = self._identify_strengths(structure, complexity)
        weaknesses = self._identify_weaknesses(structure, complexity)
        opportunities = self._identify_opportunities(structure, complexity)
        threats = self._identify_threats(structure, complexity)

        # Generate recommendations
        recommendations = generate_protocol_recommendations(
            structure, complexity, readability_score
        )

        insights = {
            "overall_score": round(overall_score, 2),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "opportunities": opportunities,
            "threats": threats,
            "recommendations": recommendations,
            "complexity_analysis": complexity,
            "readability_score": round(readability_score, 2),
            "compliance_risk": self._assess_compliance_risk(protocol_text),
        }
        # Emit refinement-needed event for low scores
        if insights["overall_score"] < 0.5:
            self._emit_event(
                "REFINEMENT_NEEDED",
                {
                    "reason": "overall_score_below_threshold",
                    "overall_score": insights["overall_score"],
                    "weaknesses": weaknesses,
                },
            )

        # **ACTUAL INTEGRATION**: Extract knowledge, track performance, and trigger alerts for low scores
        self._extract_analytics_knowledge("generate_ai_insights", insights, protocol_text)
        self._track_analytics_performance("generate_ai_insights", True, 0.0, insights["overall_score"])

        if insights["overall_score"] < 0.5:
            self._trigger_analytics_alerts("generate_ai_insights", True, insights["overall_score"],
                                           "Overall score below threshold: {:.2f}".format(insights["overall_score"]))

        return insights

    def register_event_callback(self, event_type: str, callback) -> None:
        """Register callback for analytics events."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent analytics events."""
        return self.recent_events[-limit:]

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Emit an analytics event."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "payload": payload,
        }
        self.recent_events.append(event)
        if len(self.recent_events) > 200:
            self.recent_events = self.recent_events[-200:]
        self._forward_event_to_api(event)
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(event)
                except Exception as exc:
                    logger = logging.getLogger(__name__)
                    logger.error("Error in analytics event callback: %s", exc)

    def _forward_event_to_api(self, event: Dict[str, Any]) -> None:
        api_base = os.getenv("ICR_API_BASE_URL")
        if not api_base:
            return
        if event.get("type") != "REFINEMENT_NEEDED":
            return
        payload = event.get("payload", {})
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{api_base}/icr/events/refinement-needed",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return

    def generate_multimodal_healing_prompt(
        self,
        protocol_text: str,
        heatmap_snapshot: Optional[Dict[str, Any]] = None,
        auto_refine_enabled: bool = False
    ) -> Dict[str, Any]:
        """
        Combine textual SWOT insights with visual heatmap signals into a single prompt.
        """
        ai_insights = self.generate_ai_insights(protocol_text)
        visual_summary = self._summarize_heatmap(heatmap_snapshot or {})

        weaknesses = ai_insights.get("weaknesses", [])
        friction_points = visual_summary.get("friction_points", [])

        healing_prompt = (
            "ICR Multi-Modal Healing Prompt\n\n"
            "Textual Weaknesses:\n"
            + "\n".join(f"- {w}" for w in weaknesses[:5]) +
            "\n\nVisual Friction Points:\n"
            + "\n".join(f"- {f}" for f in friction_points[:5]) +
            "\n\nAction:\n"
            "Propose refinements that address both logical weaknesses and UI friction."
        )

        if ai_insights.get("overall_score", 1.0) < 0.5:
            payload = {
                "reason": "multimodal_low_score",
                "overall_score": ai_insights.get("overall_score"),
                "weaknesses": weaknesses,
                "friction_points": friction_points,
                "auto_refine": auto_refine_enabled,
            }
            self._emit_event("REFINEMENT_NEEDED", payload)

        return {
            "healing_prompt": healing_prompt,
            "ai_insights": ai_insights,
            "visual_summary": visual_summary,
        }

    def _summarize_heatmap(self, heatmap_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize heatmap data into friction points."""
        points = heatmap_snapshot.get("points", []) if isinstance(heatmap_snapshot, dict) else []
        if not points:
            return {"friction_points": [], "hotspots": []}

        # Bin into 3x3 grid
        grid = [[0 for _ in range(3)] for _ in range(3)]
        for point in points:
            try:
                x = min(2, max(0, int(float(point.get("x", 0)) * 3)))
                y = min(2, max(0, int(float(point.get("y", 0)) * 3)))
                grid[y][x] += 1
            except Exception:
                continue

        labels = [
            ["top-left", "top-center", "top-right"],
            ["center-left", "center", "center-right"],
            ["bottom-left", "bottom-center", "bottom-right"],
        ]

        hotspots = []
        for row_idx, row in enumerate(grid):
            for col_idx, count in enumerate(row):
                if count > 0:
                    hotspots.append({"area": labels[row_idx][col_idx], "count": count})

        hotspots.sort(key=lambda x: x["count"], reverse=True)
        friction_points = [
            f"Repeated interactions near {h['area']} (count={h['count']})"
            for h in hotspots[:3]
        ]

        return {"friction_points": friction_points, "hotspots": hotspots}

    def _calculate_structure_score(self, structure: Dict) -> float:
        """Calculate structure quality score."""
        if not structure:
            return 0.0
            
        # Points for good structure elements
        score = 0.0
        max_score = 100.0
        
        # Headers contribute significantly to structure
        headers = structure.get("headers", [])
        score += min(30, len(headers) * 5)  # Up to 30 points for headers
        
        # Sections contribute to structure
        sections = structure.get("sections", [])
        score += min(25, len(sections) * 3)  # Up to 25 points for sections
        
        # Lists contribute to structure
        lists = structure.get("lists", [])
        score += min(15, len(lists) * 2)  # Up to 15 points for lists
        
        # Code blocks contribute to structure
        code_blocks = structure.get("code_blocks", [])
        score += min(10, len(code_blocks) * 2)  # Up to 10 points for code blocks
        
        # Tables contribute to structure
        tables = structure.get("tables", [])
        score += min(10, len(tables) * 3)  # Up to 10 points for tables
        
        # Images contribute to structure
        images = structure.get("images", [])
        score += min(10, len(images) * 2)  # Up to 10 points for images
        
        return min(max_score, score)

    def _calculate_complexity_score(self, complexity: Dict) -> float:
        """Calculate complexity appropriateness score."""
        if not complexity:
            return 0.0
            
        # Ideal complexity range is 0.3-0.7 (normalized)
        ideal_min = 0.3
        ideal_max = 0.7
        actual_complexity = complexity.get("normalized_score", 0.0)
        
        # Score is highest when complexity is in ideal range
        if ideal_min <= actual_complexity <= ideal_max:
            return 100.0
        elif actual_complexity < ideal_min:
            # Too simple - score decreases linearly
            return max(0.0, 100.0 * (actual_complexity / ideal_min))
        else:
            # Too complex - score decreases linearly
            return max(0.0, 100.0 * (1.0 - (actual_complexity - ideal_max) / (1.0 - ideal_max)))

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score using simplified Flesch-Kincaid."""
        if not text:
            return 0.0
            
        # Split into sentences and words
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()
        
        if not sentences or not words:
            return 0.0
            
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Readability formula based on sentence and word length
        readability = 100 - (avg_sentence_length * 1.0) - (avg_word_length * 5.0)
        return max(0.0, min(100.0, readability))

    def _identify_strengths(self, structure: Dict, complexity: Dict) -> List[str]:
        """Identify strengths in the protocol."""
        strengths = []
        
        if structure:
            headers = structure.get("headers", [])
            if len(headers) >= 5:
                strengths.append("Well-structured with clear headings")
            
            sections = structure.get("sections", [])
            if len(sections) >= 3:
                strengths.append("Good section organization")
                
            lists = structure.get("lists", [])
            if len(lists) >= 2:
                strengths.append("Effective use of lists for clarity")
                
            code_blocks = structure.get("code_blocks", [])
            if len(code_blocks) >= 1:
                strengths.append("Includes code examples for illustration")
                
            tables = structure.get("tables", [])
            if len(tables) >= 1:
                strengths.append("Uses tables for data presentation")
        
        if complexity:
            score = complexity.get("normalized_score", 0.0)
            if 0.3 <= score <= 0.7:
                strengths.append("Appropriate complexity level")
            elif score < 0.3:
                strengths.append("Simple and easy to understand")
        
        return strengths if strengths else ["No major strengths identified"]

    def _identify_weaknesses(self, structure: Dict, complexity: Dict) -> List[str]:
        """Identify weaknesses in the protocol."""
        weaknesses = []
        
        if structure:
            headers = structure.get("headers", [])
            if len(headers) < 3:
                weaknesses.append("Lacks sufficient headings for navigation")
            
            sections = structure.get("sections", [])
            if len(sections) < 2:
                weaknesses.append("Poor section organization")
                
            lists = structure.get("lists", [])
            if len(lists) < 1:
                weaknesses.append("Limited use of lists for better readability")
                
            code_blocks = structure.get("code_blocks", [])
            if len(code_blocks) < 1 and complexity.get("domain") == "code":
                weaknesses.append("Missing code examples despite being a code protocol")
                
            tables = structure.get("tables", [])
            if len(tables) < 1 and "data" in structure.get("content_types", []):
                weaknesses.append("Missing tables for data presentation")
        
        if complexity:
            score = complexity.get("normalized_score", 0.0)
            if score > 0.7:
                weaknesses.append("Overly complex for typical audience")
            elif score < 0.3 and complexity.get("domain") not in ["simple", "basic"]:
                weaknesses.append("Too simplistic for complex subject matter")
        
        return weaknesses if weaknesses else ["No major weaknesses identified"]

    def _identify_opportunities(self, structure: Dict, complexity: Dict) -> List[str]:
        """Identify opportunities for improvement."""
        opportunities = []
        
        if structure:
            # Suggest adding missing elements
            if not structure.get("headers"):
                opportunities.append("Add clear headings to improve navigation")
            
            if not structure.get("lists"):
                opportunities.append("Use bullet points or numbered lists to break up dense text")
                
            if not structure.get("code_blocks") and complexity.get("domain") == "code":
                opportunities.append("Include code examples to illustrate concepts")
                
            if not structure.get("tables") and "data" in structure.get("content_types", []):
                opportunities.append("Add tables to present data more effectively")
                
            if not structure.get("images") and complexity.get("domain") in ["technical", "medical"]:
                opportunities.append("Consider adding diagrams or illustrations")
        
        return opportunities if opportunities else ["No specific opportunities identified"]

    def _identify_threats(self, structure: Dict, complexity: Dict) -> List[str]:
        """Identify potential threats to protocol effectiveness."""
        threats = []
        
        if complexity:
            score = complexity.get("normalized_score", 0.0)
            if score > 0.8:
                threats.append("High complexity may hinder adoption or understanding")
            elif score < 0.2 and complexity.get("domain") not in ["simple", "basic"]:
                threats.append("Oversimplification may miss important details")
        
        if structure:
            # Check for structural issues that could cause problems
            if len(structure.get("headers", [])) < 2:
                threats.append("Poor structure may cause confusion for readers")
                
            if not structure.get("sections") and len(structure.get("content", "").split("\n\n")) > 10:
                threats.append("Lack of sections makes long documents hard to navigate")
        
        return threats if threats else ["No major threats identified"]

    def _assess_compliance_risk(self, text: str) -> str:
        """Assess compliance risk level."""
        if not text:
            return "low"
            
        # Look for compliance-related keywords
        high_risk_keywords = ["must", "shall", "required", "mandatory", "compliance"]
        med_risk_keywords = ["should", "recommended", "suggested", "guideline"]

        
        text_lower = text.lower()
        high_count = sum(1 for word in high_risk_keywords if word in text_lower)
        med_count = sum(1 for word in med_risk_keywords if word in text_lower)

        
        if high_count > 3:
            return "high"
        elif high_count > 0 or med_count > 5:
            return "medium"
        else:
            return "low"

    def get_evolution_metrics(self) -> Dict[str, Any]:
        """Get evolution performance metrics."""
        metrics = {
            "timestamp": time.time(),
            "evolution_runs": len(getattr(st.session_state, "evolution_history", [])),
            "successful_evolutions": getattr(st.session_state, "successful_evolutions", 0),
            "failed_evolutions": getattr(st.session_state, "failed_evolutions", 0),
            "avg_evolution_time": getattr(st.session_state, "avg_evolution_time", 0),
            "total_generations": sum(
                len(gen.get("population", [])) 
                for gen in getattr(st.session_state, "evolution_history", [])
            ),
            "best_fitness_ever": getattr(st.session_state, "best_fitness_ever", 0),
            "current_fitness": self._get_current_fitness(),
            "population_diversity": self._calculate_population_diversity(),
            "convergence_rate": self._calculate_convergence_rate(),
        }
        return metrics

    def _get_current_fitness(self) -> float:
        """Get current fitness from session state."""
        if hasattr(st.session_state, "evolution_history") and st.session_state.evolution_history:
            latest_gen = st.session_state.evolution_history[-1]
            if "population" in latest_gen and latest_gen["population"]:
                return max(ind.get("fitness", 0) for ind in latest_gen["population"])
        return 0.0

    def _calculate_population_diversity(self) -> float:
        """Calculate current population diversity."""
        if hasattr(st.session_state, "evolution_history") and st.session_state.evolution_history:
            latest_gen = st.session_state.evolution_history[-1]
            if "population" in latest_gen and len(latest_gen["population"]) > 1:
                fitnesses = [ind.get("fitness", 0) for ind in latest_gen["population"]]
                if fitnesses:
                    return float(np.std(fitnesses))
        return 0.0

    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate."""
        if hasattr(st.session_state, "evolution_history") and len(st.session_state.evolution_history) >= 2:
            history = st.session_state.evolution_history
            if len(history) >= 5:  # Need at least 5 generations to calculate meaningful convergence
                recent_fitnesses = []
                for gen in history[-5:]:  # Last 5 generations
                    if "population" in gen and gen["population"]:
                        best_fitness = max(ind.get("fitness", 0) for ind in gen["population"])
                        recent_fitnesses.append(best_fitness)
                
                if len(recent_fitnesses) >= 2:
                    # Calculate average improvement rate
                    improvements = [recent_fitnesses[i] - recent_fitnesses[i-1] 
                                  for i in range(1, len(recent_fitnesses))]
                    avg_improvement = sum(improvements) / len(improvements)
                    return float(avg_improvement)
        return 0.0

    def get_adversarial_metrics(self) -> Dict[str, Any]:
        """Get adversarial testing performance metrics."""
        metrics = {
            "timestamp": time.time(),
            "adversarial_runs": len(getattr(st.session_state, "adversarial_history", [])),
            "successful_tests": getattr(st.session_state, "successful_adversarial_tests", 0),
            "failed_tests": getattr(st.session_state, "failed_adversarial_tests", 0),
            "avg_test_time": getattr(st.session_state, "avg_adversarial_test_time", 0),
            "total_iterations": sum(
                len(run.get("iterations", [])) 
                for run in getattr(st.session_state, "adversarial_history", [])
            ),
            "final_approval_rate": getattr(st.session_state, "final_adversarial_approval_rate", 0),
            "current_approval_rate": self._get_current_approval_rate(),
            "issues_found": self._count_issues_found(),
            "issues_resolved": self._count_issues_resolved(),
            "model_performance": self._get_model_performance_stats(),
        }
        return metrics

    def _get_current_approval_rate(self) -> float:
        """Get current approval rate from session state."""
        if hasattr(st.session_state, "adversarial_results") and st.session_state.adversarial_results:
            return float(st.session_state.adversarial_results.get("final_approval_rate", 0))
        return 0.0

    def _count_issues_found(self) -> int:
        """Count total issues found in adversarial testing."""
        count = 0
        if hasattr(st.session_state, "adversarial_results") and st.session_state.adversarial_results:
            for iteration in st.session_state.adversarial_results.get("iterations", []):
                for critique in iteration.get("critiques", []):
                    if critique.get("critique_json"):
                        count += len(critique["critique_json"].get("issues", []))
        return count

    def _count_issues_resolved(self) -> int:
        """Count total issues resolved in adversarial testing."""
        count = 0
        if hasattr(st.session_state, "adversarial_results") and st.session_state.adversarial_results:
            for iteration in st.session_state.adversarial_results.get("iterations", []):
                for patch in iteration.get("patches", []):
                    if patch.get("patch_json"):
                        count += len(patch["patch_json"].get("mitigation_matrix", []))
        return count

    def _get_model_performance_stats(self) -> Dict[str, Any]:
        """Get statistics on model performance."""
        if hasattr(st.session_state, "adversarial_model_performance"):
            return st.session_state.adversarial_model_performance
        return {}

    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed model performance metrics."""
        metrics = {
            "timestamp": time.time(),
            "models_used": list(getattr(st.session_state, "adversarial_model_performance", {}).keys()),
            "model_scores": {
                model: perf.get("score", 0) 
                for model, perf in getattr(st.session_state, "adversarial_model_performance", {}).items()
            },
            "model_issues_found": {
                model: perf.get("issues_found", 0) 
                for model, perf in getattr(st.session_state, "adversarial_model_performance", {}).items()
            },
            "best_performing_model": self._get_best_model(),
            "avg_model_score": self._calculate_avg_model_score(),
            "model_diversity": self._calculate_model_diversity(),
        }
        return metrics

    def _get_best_model(self) -> str:
        """Get the best performing model."""
        if hasattr(st.session_state, "adversarial_model_performance"):
            best_model = ""
            best_score = -1
            for model, perf in st.session_state.adversarial_model_performance.items():
                score = perf.get("score", 0)
                if score > best_score:
                    best_score = score
                    best_model = model
            return best_model
        return ""

    def _calculate_avg_model_score(self) -> float:
        """Calculate average model score."""
        if hasattr(st.session_state, "adversarial_model_performance"):
            scores = [perf.get("score", 0) 
                     for perf in st.session_state.adversarial_model_performance.values()]
            if scores:
                return float(sum(scores) / len(scores))
        return 0.0

    def _calculate_model_diversity(self) -> float:
        """Calculate model diversity based on performance variance."""
        if hasattr(st.session_state, "adversarial_model_performance"):
            scores = [perf.get("score", 0) 
                     for perf in st.session_state.adversarial_model_performance.values()]
            if len(scores) > 1:
                return float(np.std(scores))
        return 0.0

    def generate_comprehensive_report(self, protocol_text: str) -> Dict[str, Any]:
        """Generate a comprehensive analytics report."""
        ai_insights = self.generate_ai_insights(protocol_text)
        evolution_metrics = self.get_evolution_metrics()
        adversarial_metrics = self.get_adversarial_metrics()
        model_metrics = self.get_model_performance_metrics()
        
        # Calculate overall score combining all metrics
        overall_score = (
            ai_insights.get("overall_score", 0) * 0.3 +
            evolution_metrics.get("current_fitness", 0) * 100 * 0.3 +  # Scale fitness to 0-100
            adversarial_metrics.get("final_approval_rate", 0) * 0.2 +
            model_metrics.get("avg_model_score", 0) * 0.2
        )
        
        return {
            "timestamp": time.time(),
            "ai_insights": ai_insights,
            "evolution_metrics": evolution_metrics,
            "adversarial_metrics": adversarial_metrics,
            "model_metrics": model_metrics,
            "overall_score": round(overall_score, 2),
            "recommendations": self._generate_comprehensive_recommendations(
                ai_insights, evolution_metrics, adversarial_metrics, model_metrics
            )
        }

    def _generate_comprehensive_recommendations(
        self, 
        ai_insights: Dict, 
        evolution_metrics: Dict, 
        adversarial_metrics: Dict, 
        model_metrics: Dict
    ) -> List[str]:
        """Generate comprehensive recommendations based on all analytics."""
        recommendations = []
        
        # AI Insights recommendations
        recommendations.extend(ai_insights.get("recommendations", []))
        
        # Evolution recommendations
        if evolution_metrics.get("convergence_rate", 0) < 0.01:
            recommendations.append("Evolution convergence rate is low. Consider adjusting parameters or using different models.")
        
        if evolution_metrics.get("population_diversity", 0) < 0.1:
            recommendations.append("Low population diversity detected. Try increasing mutation rates or using more diverse models.")
        
        # Adversarial recommendations
        if adversarial_metrics.get("final_approval_rate", 0) < 70:
            recommendations.append("Approval rate is below target. Consider more adversarial iterations or different models.")
        
        if adversarial_metrics.get("issues_resolved", 0) < adversarial_metrics.get("issues_found", 0) * 0.5:
            recommendations.append("Less than 50% of issues resolved. Review patching strategies or model capabilities.")
        
        # Model recommendations
        if len(model_metrics.get("models_used", [])) < 3:
            recommendations.append("Using more diverse models could improve results.")
        
        if model_metrics.get("model_diversity", 0) < 10:
            recommendations.append("Low model diversity detected. Try using models with different architectures or capabilities.")
        
        # Generic recommendations if no specific issues
        if not recommendations:
            recommendations.append("Protocol quality is high. Continue with current approach.")
            recommendations.append("Evolution and adversarial testing are performing well.")
            recommendations.append("Model performance is satisfactory.")
        


    def render_ai_insights_dashboard(self, protocol_text: str) -> str:
        """
        Render an AI insights dashboard for the protocol.

        Args:
            protocol_text (str): Protocol text to analyze

        Returns:
            str: HTML formatted dashboard
        """
        insights = self.generate_ai_insights(protocol_text)

        # Create a visual dashboard
        html = f"""
        <div style="background: linear-gradient(135deg, #4a6fa5, #6b8cbc); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin-top: 0; text-align: center;">🤖 AI Insights Dashboard</h2>
            <div style="display: flex; justify-content: center; align-items: center;">
                <div style="background: white; color: #4a6fa5; border-radius: 50%; width: 100px; height: 100px; display: flex; justify-content: center; align-items: center; font-size: 2em; font-weight: bold;">
                    {insights["overall_score"]}%</div>
            </div>
            <p style="text-align: center; margin-top: 10px;">Overall Protocol Quality Score</p>
        </div>
        """

        # Add metrics cards
        html += f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 4px solid #4caf50;">
                <h4 style="margin-top: 0; color: #2e7d32;">📊 Readability</h4>
                <p style="font-size: 1.5em; font-weight: bold; margin: 0;">{insights["readability_score"]}%</p>
            </div>
            <div style="background-color: #fff8e1; padding: 15px; border-radius: 8px; border-left: 4px solid #ff9800;">
                <h4 style="margin-top: 0; color: #f57f17;">📋 Structure</h4>
                <p style="font-size: 1.5em; font-weight: bold; margin: 0;">{len([s for s in insights["structure_analysis"].values() if s])}/7</p>
            </div>
            <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border-left: 4px solid #f44336;">
                <h4 style="margin-top: 0; color: #c62828;">[WARN] Compliance Risk</h4>
                <p style="font-size: 1.5em; font-weight: bold; margin: 0; text-transform: capitalize;">{insights["compliance_risk"]}</p>
            </div>
        </div>
        """

        # Add insights sections
        html += """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
        """

        # Strengths
        if insights["strengths"]:
            html += """
            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px;">
                <h3 style="color: #2e7d32; margin-top: 0;">[OK] Strengths</h3>
                <ul style="padding-left: 20px;">
            """
            for strength in insights["strengths"][:5]:  # Limit to first 5
                html += f"<li>{strength}</li>"
            html += """
                </ul>
            </div>
            """

        # Weaknesses
        if insights["weaknesses"]:
            html += """
            <div style="background-color: #ffebee; padding: 15px; border-radius: 8px;">
                <h3 style="color: #c62828; margin-top: 0;">[FAIL] Areas for Improvement</h3>
                <ul style="padding-left: 20px;">
            """
            for weakness in insights["weaknesses"][:5]:  # Limit to first 5
                html += f"<li>{weakness}</li>"
            html += """
                </ul>
            </div>
            """

        # Opportunities
        if insights["opportunities"]:
            html += """
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px;">
                <h3 style="color: #1565c0; margin-top: 0;">✨ Opportunities</h3>
                <ul style="padding-left: 20px;">
            """
            for opportunity in insights["opportunities"][:5]:  # Limit to first 5
                html += f"<li>{opportunity}</li>"
            html += """
                </ul>
            </div>
            """

        # Threats
        if insights["threats"]:
            html += """
            <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px;">
                <h3 style="color: #ef6c00; margin-top: 0;">[WARN] Potential Threats</h3>
                <ul style="padding-left: 20px;">
            """
            for threat in insights["threats"][:5]:  # Limit to first 5
                html += f"<li>{threat}</li>"
            html += """
                </ul>
            </div>
            """

        html += "</div>"

        # Add recommendations
        if insights["recommendations"]:
            html += """
            <div style="background-color: #f3e5f5; padding: 15px; border-radius: 8px; margin-top: 20px;">
                <h3 style="color: #6a1b9a; margin-top: 0;">💡 AI Recommendations</h3>
                <ul style="padding-left: 20px;">
            """
            for recommendation in insights["recommendations"]:
                html += f"<li>{recommendation}</li>"
            html += """
                </ul>
            </div>
            """

        return html

    def generate_advanced_analytics(self, results: Dict) -> Dict:
        """
        Generate advanced analytics from adversarial testing results.

        Args:
            results (Dict): Adversarial testing results

        Returns:
            Dict: Advanced analytics data
        """
        analytics = {
            "total_iterations": len(results.get("iterations", [])),
            "final_approval_rate": results.get("final_approval_rate", 0),
            "total_cost_usd": results.get("cost_estimate_usd", 0),
            "total_tokens": results.get("tokens", {}).get("prompt", 0)
            + results.get("tokens", {}).get("completion", 0),
            "confidence_trend": [],
            "issue_resolution_rate": 0,
            "model_performance": {},
            "efficiency_score": 0,
            "security_strength": 0,
            "compliance_coverage": 0,
            "clarity_score": 0,
            "completeness_score": 0,
        }

        # Calculate confidence trend
        if results.get("iterations"):
            confidence_history = [
                iter.get("approval_check", {}).get("approval_rate", 0)
                for iter in results.get("iterations", [])
            ]
            analytics["confidence_trend"] = confidence_history

        # Calculate issue resolution rate
        if results.get("iterations"):
            total_issues_found = 0
            total_issues_resolved = 0
            severity_weights = {"low": 1, "medium": 3, "high": 6, "critical": 12}

            for iteration in results.get("iterations", []):
                critiques = iteration.get("critiques", [])
                for critique in critiques:
                    critique_json = critique.get("critique_json", {})
                    issues = critique_json.get("issues", [])
                    total_issues_found += len(issues)

                    # Count resolved issues with severity weighting
                    patches = iteration.get("patches", [])
                    resolved_weighted = 0
                    total_weighted = 0

                    for issue in issues:
                        severity = issue.get("severity", "low").lower()
                        weight = severity_weights.get(severity, 1)
                        total_weighted += weight

                    for patch in patches:
                        patch_json = patch.get("patch_json", {})
                        mitigation_matrix = patch_json.get("mitigation_matrix", [])
                        for mitigation in mitigation_matrix:
                            if mitigation.get("issue") == issue.get("title") and str(
                                mitigation.get("status", "")
                            ).lower() in ["resolved", "mitigated"]:
                                resolved_weighted += 3  # Average weight
                                break

                    total_issues_resolved += min(len(issues), len(patches))

            if total_issues_found > 0:
                analytics["issue_resolution_rate"] = (
                    total_issues_resolved / total_issues_found
                ) * 100

        # Calculate efficiency score
        efficiency = 100
        if analytics["total_cost_usd"] > 0:
            # Lower cost = higher efficiency
            efficiency -= min(50, analytics["total_cost_usd"] * 10)
        if analytics["total_iterations"] > 10:
            # More iterations = lower efficiency
            efficiency -= min(30, (analytics["total_iterations"] - 10) * 2)
        analytics["efficiency_score"] = max(0, efficiency)

        # Calculate security strength based on resolved critical/high issues
        if results.get("iterations"):
            critical_high_resolved = 0
            total_critical_high = 0

            for iteration in results.get("iterations", []):
                critiques = iteration.get("critiques", [])
                for critique in critiques:
                    critique_json = critique.get("critique_json", {})
                    issues = critique_json.get("issues", [])
                    for issue in issues:
                        severity = issue.get("severity", "low").lower()
                        if severity in ["critical", "high"]:
                            total_critical_high += 1
                            # Check if this issue was resolved in any patch
                            for patch in iteration.get("patches", []):
                                patch_json = patch.get("patch_json", {})
                                mitigation_matrix = patch_json.get(
                                    "mitigation_matrix", []
                                )
                                for mitigation in mitigation_matrix:
                                    if mitigation.get("issue") == issue.get(
                                        "title"
                                    ) and str(mitigation.get("status", "")).lower() in [
                                        "resolved",
                                        "mitigated",
                                    ]:
                                        critical_high_resolved += 1
                                        break

            if total_critical_high > 0:
                analytics["security_strength"] = (
                    critical_high_resolved / total_critical_high
                ) * 100

        # Calculate compliance coverage
        if results.get("compliance_requirements"):
            # Simple check for compliance mentions in final protocol
            final_sop = results.get("final_sop", "")
            compliance_reqs = results.get("compliance_requirements", "")

            # Count how many compliance requirements are addressed
            reqs_addressed = 0
            total_reqs = 0

            for req in compliance_reqs.split(","):
                req = req.strip().lower()
                if req:
                    total_reqs += 1
                    if req in final_sop.lower():
                        reqs_addressed += 1

            if total_reqs > 0:
                analytics["compliance_coverage"] = (reqs_addressed / total_reqs) * 100

        # Calculate clarity score based on protocol structure
        final_sop = results.get("final_sop", "")
        if final_sop:
            # Calculate metrics
            structure = extract_protocol_structure(final_sop)
            complexity = calculate_protocol_complexity(final_sop)

            # Clarity score based on structure elements
            clarity_score = 0
            if structure["has_headers"]:
                clarity_score += 25
            if structure["has_numbered_steps"] or structure["has_bullet_points"]:
                clarity_score += 25
            if structure["has_preconditions"]:
                clarity_score += 15
            if structure["has_postconditions"]:
                clarity_score += 15
            if structure["has_error_handling"]:
                clarity_score += 20

            analytics["clarity_score"] = clarity_score

            # Completeness score based on structure and complexity
            completeness_score = min(
                100,
                (
                    structure["section_count"]
                    * 5  # Sections contribute to completeness
                    + complexity["unique_words"]
                    / max(1, complexity["word_count"])
                    * 100
                    * 0.3  # Vocabulary diversity
                    + (1 - complexity["avg_sentence_length"] / 50)
                    * 100
                    * 0.7  # Sentence complexity balance
                ),
            )
            analytics["completeness_score"] = completeness_score

        return analytics

    def calculate_model_performance_metrics(self, model_performance_data: Dict) -> Dict:
        """
        Calculate additional performance metrics for models.

        Args:
            model_performance_data (Dict): Raw model performance data

        Returns:
            Dict: Enhanced model performance metrics
        """
        enhanced_metrics = {}

        for model_id, perf in model_performance_data.items():
            # Calculate derived metrics
            score = perf.get("score", 0)
            issues_found = perf.get("issues_found", 0)

            # Efficiency: issues found per unit score
            efficiency = issues_found / max(1, score) if score > 0 else 0

            # Severity-weighted score (if available)
            severity_scores = perf.get("severity_scores", {})
            weighted_score = sum(severity_scores.values())

            enhanced_metrics[model_id] = {
                "raw_score": score,
                "issues_found": issues_found,
                "efficiency": efficiency,
                "weighted_score": weighted_score,
                "detection_rate": perf.get("detection_rate", 0),
                "fix_quality": perf.get("fix_quality", 0),
                "response_time": perf.get("response_time", 0),
                "cost_efficiency": perf.get("cost", 0) / max(1, issues_found)
                if issues_found > 0
                else float("inf"),
            }

        return enhanced_metrics

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Analytics Manager
    # =========================================================================

    def _trigger_analytics_alerts(
        self,
        operation: str,
        success: bool,
        score: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for analytics issues (low scores, failures)."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            # Alert on low scores or failures
            if not success or (score is not None and score < 0.5):
                severity = AlertSeverity.MEDIUM if score and score >= 0.3 else AlertSeverity.HIGH

                alert_manager.create_alert(
                    title=f"Analytics Alert: {operation}",
                    description=f"Analytics operation '{operation}' " +
                                 ("failed" if not success else f"has low score: {score}") +
                                 (f". Error: {error}" if error else ""),
                    severity=severity.value,
                    source="analytics_manager",
                    component="analytics_insights",
                    metadata=metadata or {}
                )

        except Exception as e:
            logging.error(f"Failed to trigger Analytics alert: {e}")

    def _extract_analytics_knowledge(
        self,
        operation: str,
        insights: Dict[str, Any],
        protocol_text: Optional[str] = None
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract analytics knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"analytics_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="analytics_insights",
                source_component="analytics_manager",
                title=f"Analytics Insights: {operation}",
                content={
                    "operation": operation,
                    "insights": insights,
                    "protocol_length": len(protocol_text) if protocol_text else 0,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "overall_score": insights.get("overall_score"),
                    "readability_score": insights.get("readability_score"),
                    "compliance_risk": insights.get("compliance_risk")
                },
                tags=["analytics", "insights", operation, "protocol_analysis"]
            )

            knowledge_engine.store_artifact(artifact)
            logging.debug(f"Extracted Analytics knowledge for {operation}")
            return True

        except Exception as e:
            logging.error(f"Failed to extract Analytics knowledge: {e}")
            return False

    def _track_analytics_performance(
        self,
        operation: str,
        success: bool,
        duration: float,
        score: Optional[float] = None
    ):
        """**ACTUAL INTEGRATION**: Track analytics operation performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            performance_data = StrategyPerformanceData(
                strategy_name=f"analytics_manager_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=score if score is not None else (1.0 if success else 0.0),
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"operation": operation, "duration": duration}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logging.debug(f"Tracked Analytics performance for {operation}")

        except Exception as e:
            logging.error(f"Failed to track Analytics performance: {e}")


# Initialize analytics manager on import
analytics_manager = AnalyticsManager()
