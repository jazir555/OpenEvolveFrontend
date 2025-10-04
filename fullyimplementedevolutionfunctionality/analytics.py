from typing import Dict, Any
import re


def analyze_plan_quality(plan_text: str) -> Dict[str, Any]:
    """Analyze plan quality metrics.

    Args:
        plan_text (str): The plan to analyze

    Returns:
        Dict[str, Any]: Plan quality metrics
    """
    if not plan_text:
        return {
            "sections": 0,
            "objectives": 0,
            "milestones": 0,
            "resources": 0,
            "risks": 0,
            "dependencies": 0,
            "timeline_elements": 0,
            "quality_score": 0,
        }

    # Count sections (headers)
    sections = len(
        re.findall(r"^#{1,6}\s+|.*\n[=]{3,}|.*\n[-]{3,}", plan_text, re.MULTILINE)
    )

    # Count objectives (look for objective-related terms)
    objective_patterns = [r"\bobjectives?\b", r"\bgoals?\b", r"\bpurpose\b", r"\baim\b"]
    objectives = 0
    for pattern in objective_patterns:
        objectives += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count milestones (look for milestone-related terms)
    milestone_patterns = [
        r"\bmilestones?\b",
        r"\bdeadlines?\b",
        r"\btimelines?\b",
        r"\bschedule\b",
    ]
    milestones = 0
    for pattern in milestone_patterns:
        milestones += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count resources (look for resource-related terms)
    resource_patterns = [
        r"\bresources?\b",
        r"\bbudget\b",
        r"\bcosts?\b",
        r"\bmaterials?\b",
    ]
    resources = 0
    for pattern in resource_patterns:
        resources += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count risks (look for risk-related terms)
    risk_patterns = [
        r"\brisks?\b",
        r"\bthreats?\b",
        r"\bvulnerabilit(?:y|ies)\b",
        r"\bhazards?\b",
    ]
    risks = 0
    for pattern in risk_patterns:
        risks += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count dependencies (look for dependency-related terms)
    dependency_patterns = [
        r"\bdependenc(?:y|ies)\b",
        r"\bprerequisites?\b",
        r"\brequires?\b",
        r"\bneeds?\b",
    ]
    dependencies = 0
    for pattern in dependency_patterns:
        dependencies += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Count timeline elements (dates, time-related terms)
    timeline_patterns = [
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
        r"\bweeks?\b",
        r"\bmonths?\b",
        r"\byears?\b",
        r"\bdays?\b",
    ]
    timeline_elements = 0
    for pattern in timeline_patterns:
        timeline_elements += len(re.findall(pattern, plan_text, re.IGNORECASE))

    # Calculate quality score (simplified)
    quality_score = 50  # Start with baseline

    # Add points for completeness
    quality_score += min(20, sections * 2)  # Up to 20 points for sections
    quality_score += min(15, objectives * 3)  # Up to 15 points for objectives
    quality_score += min(10, milestones * 2)  # Up to 10 points for milestones
    quality_score += min(10, resources * 2)  # Up to 10 points for resources
    quality_score += min(10, risks * 2)  # Up to 10 points for risks
    quality_score += min(10, dependencies * 2)  # Up to 10 points for dependencies
    quality_score += min(10, timeline_elements * 2)  # Up to 10 points for timeline

    quality_score = max(0, min(100, quality_score))  # Clamp to 0-100

    return {
        "sections": sections,
        "objectives": objectives,
        "milestones": milestones,
        "resources": resources,
        "risks": risks,
        "dependencies": dependencies,
        "timeline_elements": timeline_elements,
        "quality_score": round(quality_score, 2),
    }


def generate_advanced_analytics(results: Dict) -> Dict:
    """Generate advanced analytics from adversarial testing results.

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
                            mitigation_matrix = patch_json.get("mitigation_matrix", [])
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
        # Placeholder calls for now, will be replaced with actual functions
        structure = {
            "has_headers": True,
            "has_numbered_steps": True,
            "has_preconditions": True,
            "has_postconditions": True,
            "has_error_handling": True,
            "section_count": 5,
        }
        complexity = {"word_count": 100, "unique_words": 50, "avg_sentence_length": 15}

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
                structure["section_count"] * 5  # Sections contribute to completeness
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

import streamlit as st # Import streamlit here as it's a UI function

def render_analytics_settings():
    """
    Renders the analytics settings section in the Streamlit UI.
    Allows users to configure what data to collect, how to display it, etc.
    """
    st.header("📊 Analytics Settings")
    
    st.info("Configure your analytics preferences to track and visualize application usage.")
    
    # Data collection settings
    with st.expander("📈 Data Collection Settings", expanded=True):
        st.subheader("Data Collection Preferences")
        
        # Simulate data collection settings
        collect_usage_data = st.checkbox("Enable anonymous usage data collection", value=True)
        collect_performance_data = st.checkbox("Enable performance metrics collection", value=True)
        collect_error_data = st.checkbox("Enable error reporting", value=True)
        
        if st.button("Save Analytics Settings"):
            # In a real implementation, this would save to user preferences
            st.session_state.analytics_settings = {
                "collect_usage_data": collect_usage_data,
                "collect_performance_data": collect_performance_data,
                "collect_error_data": collect_error_data
            }
            st.success("Analytics settings saved successfully!")
    
    # Report format settings
    with st.expander("📄 Report Preferences", expanded=True):
        st.subheader("Default Report Format")
        report_format = st.selectbox("Default report format", ["Markdown", "JSON", "PDF", "CSV", "Excel"])
        
        if st.button("Set Default Format"):
            st.session_state.default_report_format = report_format
            st.success(f"Default report format set to {report_format}")
    
    # Data retention settings
    with st.expander("🗂️ Data Retention", expanded=True):
        st.subheader("Data Retention Settings")
        retention_period = st.select_slider("Data retention period", 
                                          options=[7, 30, 60, 90, 180, 365], 
                                          format_func=lambda x: f"{x} days",
                                          value=90)
        
        st.write(f"Analytics data will be retained for **{retention_period} days**")
        
        if st.button("Apply Retention Policy"):
            st.session_state.data_retention_days = retention_period
            st.success(f"Data retention policy updated to {retention_period} days")
    
    # Privacy settings
    with st.expander("🔒 Privacy Settings", expanded=True):
        st.subheader("Privacy Controls")
        include_personal_info = st.checkbox("Include personal information in analytics", value=False)
        share_aggregated_data = st.checkbox("Allow sharing of aggregated, anonymized data", value=True)
        
        st.info("Personal information is never shared without your explicit consent.")
    
    # Current settings summary
    with st.expander("📋 Current Configuration", expanded=True):
        st.subheader("Current Analytics Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Usage Data Collection:** {'✅ Enabled' if collect_usage_data else '❌ Disabled'}")
            st.write(f"**Performance Data Collection:** {'✅ Enabled' if collect_performance_data else '❌ Disabled'}")
            st.write(f"**Error Reporting:** {'✅ Enabled' if collect_error_data else '❌ Disabled'}")
        with col2:
            st.write(f"**Default Report Format:** {st.session_state.get('default_report_format', 'Markdown')}")
            st.write(f"**Data Retention:** {st.session_state.get('data_retention_days', 90)} days")
            st.write(f"**Personal Info Included:** {'Yes' if include_personal_info else 'No'}")
    
    st.info("💡 Analytics help improve the application. You can change these settings anytime in this panel.")
