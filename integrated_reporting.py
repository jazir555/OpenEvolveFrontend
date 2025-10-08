"""
Reporting and Analytics for Integrated Adversarial Testing + Evolution Workflow
This module provides comprehensive reporting and analytics for the integrated workflow.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any


def generate_integrated_report(integrated_results: Dict[str, Any]) -> str:
    """
    Generate a comprehensive HTML report for the integrated workflow.
    
    Args:
        integrated_results: Dictionary containing results from the integrated workflow
        
    Returns:
        HTML report as a string
    """
    # Extract data for reporting
    initial_content_length = integrated_results.get("initial_content", "")
    final_content_length = integrated_results.get("final_content", "")
    adversarial_results = integrated_results.get("adversarial_results", {})
    evolution_results = integrated_results.get("evolution_results", {})
    evaluation_results = integrated_results.get("evaluation_results", {})
    keyword_analysis = integrated_results.get("keyword_analysis", {})
    integrated_score = integrated_results.get("integrated_score", 0.0)
    total_cost = integrated_results.get("total_cost_usd", 0.0)
    total_tokens = integrated_results.get("total_tokens", {"prompt": 0, "completion": 0})
    
    # Get adversarial metrics
    adversarial_iterations = len(adversarial_results.get("iterations", []))
    final_approval_rate = adversarial_results.get("final_approval_rate", 0.0)
    
    # Get evolution metrics
    evolution_content = evolution_results.get("final_content", "")
    
    # Get evaluation metrics
    evaluation_success = evaluation_results.get("success", False)
    evaluation_iterations = len(evaluation_results.get("iterations", [])) if evaluation_results else 0
    final_evaluation_score = evaluation_results.get("final_score", 0.0) if evaluation_results else 0.0
    consecutive_rounds_met = evaluation_results.get("consecutive_rounds_met", False) if evaluation_results else False
    
    # Create the HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Integrated Adversarial Testing + Evolution + Evaluation Report</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 40px; 
                background-color: #f8f9fa; 
                color: #333;
            }}
            h1, h2, h3 {{ 
                color: #4a6fa5; 
            }}
            .summary {{ 
                background-color: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
                margin-bottom: 20px; 
            }}
            .section {{ 
                margin: 20px 0; 
                background-color: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
            }}
            .metric {{ 
                text-align: center; 
                padding: 10px; 
                background-color: #e9ecef; 
                border-radius: 4px; 
                margin: 5px; 
                font-weight: bold;
            }}
            .approval {{ 
                color: #4caf50; 
            }}
            .rejection {{ 
                color: #f44336; 
            }}
            table {{ 
                border-collapse: collapse; 
                width: 100%; 
            }}
            th, td {{ 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left; 
            }}
            th {{ 
                background-color: #4a6fa5; 
                color: white; 
            }}
            .content-diff {{
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
                max-height: 300px;
                overflow-y: auto;
            }}
            .keyword-section {{
                background-color: #e8f4fd;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <h1>Integrated Adversarial Testing + Evolution + Evaluation Report</h1>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <p><strong>Integrated Score:</strong> {integrated_score:.2f} (0.0-1.0 scale)</p>
            <p><strong>Total Cost (USD):</strong> ${total_cost:.4f}</p>
            <p><strong>Total Tokens:</strong> {total_tokens.get('prompt', 0) + total_tokens.get('completion', 0):,} (Prompt: {total_tokens.get('prompt', 0):,}, Completion: {total_tokens.get('completion', 0):,})</p>
            <p><strong>Content Improvement:</strong> From {len(initial_content_length)} to {len(final_content_length)} characters</p>
            <p><strong>Process Success:</strong> {evaluation_success if evaluation_results else 'N/A'}</p>
        </div>
        
        <div class="section">
            <h2>Adversarial Testing Phase</h2>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                <div class="metric">
                    <div>Iterations Completed</div>
                    <div style="font-size: 24px;">{adversarial_iterations}</div>
                </div>
                <div class="metric">
                    <div>Final Approval Rate</div>
                    <div style="font-size: 24px;">{final_approval_rate:.1f}%</div>
                </div>
                <div class="metric">
                    <div>Integrated Score</div>
                    <div style="font-size: 24px;">{integrated_score:.2f}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Evolution Phase</h2>
            <p><strong>Final Evolution Content Length:</strong> {len(evolution_content)} characters</p>
        </div>
        
        <div class="section">
            <h2>Evaluation Phase</h2>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                <div class="metric">
                    <div>Iterations Completed</div>
                    <div style="font-size: 24px;">{evaluation_iterations}</div>
                </div>
                <div class="metric">
                    <div>Final Evaluation Score</div>
                    <div style="font-size: 24px;">{final_evaluation_score:.1f}%</div>
                </div>
                <div class="metric {'approval' if consecutive_rounds_met else 'rejection'}">
                    <div>Evaluator Success</div>
                    <div style="font-size: 24px;">{'✅' if consecutive_rounds_met else '❌'}</div>
                </div>
            </div>
        </div>
    """
    
    # Add keyword analysis section if available
    if keyword_analysis:
        html_report += f"""
        <div class="section">
            <h2>Keyword Analysis</h2>
            <div class="keyword-section">
                <p><strong>Keywords Found:</strong> {', '.join(keyword_analysis.get('keywords_found', []))}</p>
                <p><strong>Relevance Score:</strong> {keyword_analysis.get('relevance_score', 0.0):.2f}</p>
                <p><strong>Keyword Densities:</strong></p>
                <ul>
        """
        
        for keyword, density in keyword_analysis.get('keyword_density', {}).items():
            html_report += f"<li>{keyword}: {density:.2f}%</li>"
        
        html_report += """
                </ul>
            </div>
        </div>
        """
    
    # Complete the HTML report
    html_report += f"""
        <div class="section">
            <h2>Final Content</h2>
            <div class="content-diff">{final_content_length}</div>
        </div>
        
        <div class="section">
            <h2>Process Log</h2>
            <div class="content-diff">{chr(10).join(integrated_results.get('process_log', []))}</div>
        </div>
        
        <div class="section">
            <h2>GitHub Sync</h2>
            <p>Use this button to approve and sync the final content to GitHub:</p>
            <button type="button" onclick="approveAndSyncToGitHub()" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">✅ Approve & Sync to GitHub</button>
            <script>
                function approveAndSyncToGitHub() {{
                    // This would need to be implemented in the actual UI
                    alert('GitHub sync functionality would be implemented here.');
                }}
            </script>
        </div>
    </body>
    </html>
    """
    
    return html_report


def render_integrated_analytics_dashboard():
    """
    Render an analytics dashboard for the integrated workflow in Streamlit.
    """
    st.header("📊 Integrated Workflow Analytics Dashboard")
    
    # Check if we have integrated results to display
    if not st.session_state.get("integrated_results"):
        st.info("No integrated workflow results available yet. Run an integrated process to generate data.")
        return
    
    integrated_results = st.session_state.integrated_results
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    integrated_score = integrated_results.get("integrated_score", 0.0)
    total_cost = integrated_results.get("total_cost_usd", 0.0)
    total_tokens = integrated_results.get("total_tokens", {"prompt": 0, "completion": 0})
    total_tokens_sum = total_tokens.get("prompt", 0) + total_tokens.get("completion", 0)
    
    col1.metric("Integrated Score", f"{integrated_score:.3f}")
    col2.metric("Total Cost (USD)", f"${total_cost:.4f}")
    col3.metric("Total Tokens", f"{total_tokens_sum:,}")
    col4.metric("Content Improvement", f"{len(integrated_results.get('final_content', '')) - len(integrated_results.get('initial_content', '')):,} chars")
    
    st.divider()
    
    # Create tabs for different analytics views
    tab1, tab2, tab3 = st.tabs(["📈 Performance Metrics", "🧬 Adversarial Analysis", "🔄 Evolution Analysis"])
    
    with tab1:  # Performance Metrics
        st.subheader("Performance Overview")
        
        # Create a dataframe for performance metrics over time if available
        if st.session_state.get("integrated_adversarial_history"):
            df = pd.DataFrame(st.session_state.integrated_adversarial_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Approval rate over iterations
                if "approval_rate" in df.columns:
                    fig = px.line(df, x="iteration", y="approval_rate", title="Adversarial Approval Rate Over Iterations")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Issues vs Mitigations over iterations
                if "issues_found" in df.columns and "mitigations" in df.columns:
                    fig = px.line(df, x="iteration", y=["issues_found", "mitigations"], 
                                 title="Issues Found vs Mitigations Over Iterations",
                                 labels={"value": "Count", "variable": "Type"})
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:  # Adversarial Analysis
        st.subheader("Adversarial Testing Analysis")
        
        adversarial_results = integrated_results.get("adversarial_results", {})
        iterations = adversarial_results.get("iterations", [])
        
        if iterations:
            # Extract issue data from iterations
            issue_data = []
            for iteration in iterations:
                for critique in iteration.get("critiques", []):
                    critique_json = critique.get("critique_json", {})
                    if critique_json:
                        issues = critique_json.get("issues", [])
                        for issue in issues:
                            issue_data.append({
                                "iteration": iteration.get("iteration"),
                                "severity": issue.get("severity", "low"),
                                "category": issue.get("category", "uncategorized"),
                                "title": issue.get("title", "")
                            })
            
            if issue_data:
                df_issues = pd.DataFrame(issue_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Severity distribution
                    severity_counts = df_issues["severity"].value_counts()
                    fig = px.pie(values=severity_counts.values, names=severity_counts.index, 
                                title="Issue Severity Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Category distribution
                    category_counts = df_issues["category"].value_counts()
                    fig = px.bar(x=category_counts.index, y=category_counts.values,
                                title="Issue Category Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No issue data available from adversarial testing.")
        else:
            st.info("No adversarial testing iterations found in results.")
    
    with tab3:  # Evolution Analysis
        st.subheader("Evolution Analysis")
        
        evolution_results = integrated_results.get("evolution_results", {})
        
        if evolution_results:
            st.json(evolution_results)  # Display evolution results as JSON
        else:
            st.info("No evolution results found in integrated workflow.")
    
    st.divider()
    
    # Detailed results section
    with st.expander("📄 Full Integrated Results"):
        st.json(integrated_results)


def update_integrated_session_state(results: Dict[str, Any]):
    """
    Update the session state with integrated workflow results.
    
    Args:
        results: Dictionary containing integrated workflow results
    """
    if "integrated_results" not in st.session_state:
        st.session_state.integrated_results = {}
    
    # Update with new results
    st.session_state.integrated_results.update(results)
    
    # Update adversarial history if available in results
    if "adversarial_results" in results:
        adversarial_results = results["adversarial_results"]
        if "iterations" in adversarial_results:
            # Extract iteration data for analytics
            iteration_data = []
            for iteration in adversarial_results["iterations"]:
                iteration_data.append({
                    "iteration": iteration.get("iteration"),
                    "approval_rate": iteration.get("approval_check", {}).get("approval_rate", 0),
                    "issues_found": iteration.get("detailed_diagnostics", {}).get("critiques_summary", {}).get("issues_found", 0),
                    "mitigations": iteration.get("detailed_diagnostics", {}).get("patches_summary", {}).get("mitigation_count", 0),
                    "content_length": len(iteration.get("content_after_patch", ""))
                })
            
            st.session_state.integrated_adversarial_history = iteration_data


def export_integrated_report(integrated_results: Dict[str, Any]) -> bytes:
    """
    Export the integrated report as various formats.
    
    Args:
        integrated_results: Dictionary containing integrated workflow results
        
    Returns:
        bytes: Report in the requested format
    """
    
    # For now, return HTML report
    html_content = generate_integrated_report(integrated_results)
    return html_content.encode('utf-8')


def calculate_detailed_metrics(integrated_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate detailed metrics for the integrated workflow.
    
    Args:
        integrated_results: Dictionary containing integrated workflow results
        
    Returns:
        Dictionary containing calculated metrics
    """
    metrics = {}
    
    # Content improvement metrics
    initial_content = integrated_results.get("initial_content", "")
    final_content = integrated_results.get("final_content", "")
    
    initial_length = len(initial_content)
    final_length = len(final_content)
    
    # Improvement percentage
    if initial_length > 0:
        length_improvement_pct = ((final_length - initial_length) / initial_length) * 100
    else:
        length_improvement_pct = 0
    
    metrics["content_length_initial"] = initial_length
    metrics["content_length_final"] = final_length
    metrics["content_length_improvement_pct"] = length_improvement_pct
    
    # Adversarial testing metrics
    adversarial_results = integrated_results.get("adversarial_results", {})
    metrics["adversarial_iterations"] = len(adversarial_results.get("iterations", []))
    metrics["adversarial_final_approval_rate"] = adversarial_results.get("final_approval_rate", 0.0)
    
    # Issue resolution metrics
    total_issues = 0
    resolved_issues = 0
    
    for iteration in adversarial_results.get("iterations", []):
        for critique in iteration.get("critiques", []):
            critique_json = critique.get("critique_json", {})
            if critique_json:
                total_issues += len(critique_json.get("issues", []))
        
        for patch in iteration.get("patches", []):
            patch_json = patch.get("patch_json", {})
            if patch_json:
                mitigation_matrix = patch_json.get("mitigation_matrix", [])
                resolved_issues += len([m for m in mitigation_matrix if m.get("status", "").lower() in ["resolved", "mitigated"]])
    
    metrics["total_issues_identified"] = total_issues
    metrics["issues_resolved"] = resolved_issues
    metrics["issue_resolution_rate"] = (resolved_issues / max(1, total_issues)) * 100 if total_issues > 0 else 0
    
    # Efficiency metrics
    total_cost = integrated_results.get("total_cost_usd", 0.0)
    total_tokens = integrated_results.get("total_tokens", {"prompt": 0, "completion": 0})
    total_tokens_sum = total_tokens.get("prompt", 0) + total_tokens.get("completion", 0)
    
    metrics["total_cost_usd"] = total_cost
    metrics["total_tokens_prompt"] = total_tokens.get("prompt", 0)
    metrics["total_tokens_completion"] = total_tokens.get("completion", 0)
    metrics["total_tokens"] = total_tokens_sum
    
    # Calculate cost per content character improvement
    if final_length > initial_length and total_cost > 0:
        cost_per_char_improvement = total_cost / (final_length - initial_length)
    else:
        cost_per_char_improvement = 0
    
    metrics["cost_per_char_improvement"] = cost_per_char_improvement
    
    return metrics