"""
Analytics Manager for OpenEvolve - Analytics and insights generation
This file manages analytics, insights, and data analysis features
File size: ~1200 lines (under the 2000 line limit)
"""
import streamlit as st
import uuid
from datetime import datetime
from typing import Dict, List, Any
import json
import re
import threading
from .session_utils import calculate_protocol_complexity, extract_protocol_structure, generate_protocol_recommendations


class AnalyticsManager:
    """
    Manages analytics, insights, and data analysis features
    """
    
    def __init__(self):
        pass
    
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
                "compliance_risk": "low"
            }
        
        # Calculate metrics
        complexity = calculate_protocol_complexity(protocol_text)
        structure = extract_protocol_structure(protocol_text)
        
        # Overall score calculation (weighted)
        structure_score = (
            (1 if structure["has_headers"] else 0) * 0.2 +
            (1 if structure["has_numbered_steps"] or structure["has_bullet_points"] else 0) * 0.2 +
            (1 if structure["has_preconditions"] else 0) * 0.15 +
            (1 if structure["has_postconditions"] else 0) * 0.15 +
            (1 if structure["has_error_handling"] else 0) * 0.15 +
            min(structure["section_count"] / 10, 1) * 0.15
        ) * 100
        
        complexity_score = max(0, 100 - complexity["complexity_score"])
        
        overall_score = (structure_score * 0.6 + complexity_score * 0.4)
        
        # Strengths
        strengths = []
        if structure["has_headers"]:
            strengths.append("✅ Well-structured with clear headers")
        if structure["has_numbered_steps"] or structure["has_bullet_points"]:
            strengths.append("✅ Uses lists or numbered steps for clarity")
        if structure["has_preconditions"]:
            strengths.append("✅ Defines clear preconditions")
        if structure["has_postconditions"]:
            strengths.append("✅ Specifies expected outcomes")
        if structure["has_error_handling"]:
            strengths.append("✅ Includes error handling procedures")
        if complexity["unique_words"] / max(1, complexity["word_count"]) > 0.6:
            strengths.append("✅ Good vocabulary diversity")
        
        # Weaknesses
        weaknesses = []
        if not structure["has_headers"]:
            weaknesses.append("❌ Lacks clear section headers")
        if not structure["has_numbered_steps"] and not structure["has_bullet_points"]:
            weaknesses.append("❌ Could use lists or numbered steps for better readability")
        if not structure["has_preconditions"]:
            weaknesses.append("❌ Missing preconditions specification")
        if not structure["has_postconditions"]:
            weaknesses.append("❌ No defined postconditions or expected outcomes")
        if not structure["has_error_handling"]:
            weaknesses.append("❌ Lacks error handling procedures")
        if complexity["avg_sentence_length"] > 25:
            weaknesses.append("❌ Sentences are quite long (hard to read)")
        if complexity["complexity_score"] > 60:
            weaknesses.append("❌ Protocol is quite complex")
        
        # Opportunities
        opportunities = []
        if complexity["word_count"] < 500:
            opportunities.append("✨ Protocol is brief - opportunity to add more detail")
        if structure["section_count"] == 0 and complexity["word_count"] > 300:
            opportunities.append("✨ Can improve organization with section headers")
        if not structure["has_preconditions"]:
            opportunities.append("✨ Add preconditions to clarify requirements")
        if not structure["has_postconditions"]:
            opportunities.append("✨ Define postconditions to specify expected outcomes")
        if not structure["has_error_handling"]:
            opportunities.append("✨ Include error handling for robustness")
        
        # Threats (potential issues)
        threats = []
        if complexity["complexity_score"] > 70:
            threats.append("⚠️ High complexity may lead to misinterpretation")
        if complexity["avg_sentence_length"] > 30:
            threats.append("⚠️ Long sentences may reduce clarity")
        if structure["section_count"] == 0 and complexity["word_count"] > 500:
            threats.append("⚠️ Lack of sections makes long protocols hard to navigate")
        
        # Recommendations
        recommendations = generate_protocol_recommendations(protocol_text)
        
        # Readability score
        readability_score = 100 - (complexity["avg_sentence_length"] / 50 * 100)
        readability_score = max(0, min(100, readability_score))
        
        # Compliance risk assessment
        compliance_risk = "low"
        if complexity["complexity_score"] > 70:
            compliance_risk = "high"
        elif complexity["complexity_score"] > 50:
            compliance_risk = "medium"
        
        return {
            "overall_score": round(overall_score, 1),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "opportunities": opportunities,
            "threats": threats,
            "recommendations": recommendations,
            "complexity_analysis": complexity,
            "structure_analysis": structure,
            "readability_score": round(readability_score, 1),
            "compliance_risk": compliance_risk
        }
    
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
                <h4 style="margin-top: 0; color: #c62828;">⚠️ Compliance Risk</h4>
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
                <h3 style="color: #2e7d32; margin-top: 0;">✅ Strengths</h3>
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
                <h3 style="color: #c62828; margin-top: 0;">❌ Areas for Improvement</h3>
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
                <h3 style="color: #ef6c00; margin-top: 0;">⚠️ Potential Threats</h3>
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
            "total_tokens": results.get("tokens", {}).get("prompt", 0) + results.get("tokens", {}).get("completion", 0),
            "confidence_trend": [],
            "issue_resolution_rate": 0,
            "model_performance": {},
            "efficiency_score": 0,
            "security_strength": 0,
            "compliance_coverage": 0,
            "clarity_score": 0,
            "completeness_score": 0
        }
        
        # Calculate confidence trend
        if results.get("iterations"):
            confidence_history = [iter.get("approval_check", {}).get("approval_rate", 0) 
                                for iter in results.get('iterations', [])]
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
                            if (mitigation.get("issue") == issue.get("title") and 
                                str(mitigation.get("status", "")).lower() in ["resolved", "mitigated"]):
                                resolved_weighted += 3  # Average weight
                                break
                    
                    total_issues_resolved += min(len(issues), len(patches))
            
            if total_issues_found > 0:
                analytics["issue_resolution_rate"] = (total_issues_resolved / total_issues_found) * 100
        
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
                                mitigation_matrix = patch_json.get("mitigation_matrix", [])
                                for mitigation in mitigation_matrix:
                                    if (mitigation.get("issue") == issue.get("title") and 
                                        str(mitigation.get("status", "")).lower() in ["resolved", "mitigated"]):
                                        critical_high_resolved += 1
                                        break
            
            if total_critical_high > 0:
                analytics["security_strength"] = (critical_high_resolved / total_critical_high) * 100
        
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
            completeness_score = min(100, (
                structure["section_count"] * 5 +  # Sections contribute to completeness
                complexity["unique_words"] / max(1, complexity["word_count"]) * 100 * 0.3 +  # Vocabulary diversity
                (1 - complexity["avg_sentence_length"] / 50) * 100 * 0.7  # Sentence complexity balance
            ))
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
                "cost_efficiency": perf.get("cost", 0) / max(1, issues_found) if issues_found > 0 else float('inf')
            }
        
        return enhanced_metrics


# Initialize analytics manager on import
analytics_manager = AnalyticsManager()