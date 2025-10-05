"""
Analytics Dashboard for OpenEvolve - Visualization and reporting
This module provides the analytics dashboard UI for visualizing evolution and adversarial testing data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import time
import numpy as np


def render_analytics_dashboard():
    """Render the analytics dashboard in Streamlit."""
    st.markdown("Welcome to your Analytics Dashboard!")
    
    # Check if we have any data to display
    if not st.session_state.get("evolution_history") and not st.session_state.get("adversarial_results"):
        st.info("No analytics data available yet. Run an evolution or adversarial testing to generate data.")
        return
    
    # Create tabs for different analytics views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", 
        "🧬 Evolution Analytics", 
        "⚔️ Adversarial Analytics", 
        "🤖 Model Performance"
    ])
    
    with tab1:  # Overview
        render_overview_tab()
    
    with tab2:  # Evolution Analytics
        render_evolution_analytics_tab()
    
    with tab3:  # Adversarial Analytics
        render_adversarial_analytics_tab()
    
    with tab4:  # Model Performance
        render_model_performance_tab()


def render_overview_tab():
    """Render the overview analytics tab."""
    st.header("📊 Analytics Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Evolution metrics
    evolution_history = st.session_state.get("evolution_history", [])
    total_evolutions = len(evolution_history)
    if evolution_history:
        latest_generation = evolution_history[-1]
        population = latest_generation.get("population", [])
        if population:
            best_fitness = max(ind.get("fitness", 0) for ind in population)
        else:
            best_fitness = 0
    else:
        best_fitness = 0
    
    # Adversarial metrics
    adversarial_results = st.session_state.get("adversarial_results", {})
    adversarial_iterations = adversarial_results.get("iterations", [])
    total_adversarial_runs = len(adversarial_iterations)
    if adversarial_iterations:
        latest_iteration = adversarial_iterations[-1]
        approval_check = latest_iteration.get("approval_check", {})
        final_approval_rate = approval_check.get("approval_rate", 0)
    else:
        final_approval_rate = 0
    
    # Cost metrics
    total_cost = st.session_state.get("adversarial_cost_estimate_usd", 0) + \
                 st.session_state.get("evolution_cost_estimate_usd", 0)
    
    # Token metrics
    total_tokens = st.session_state.get("adversarial_total_tokens_prompt", 0) + \
                   st.session_state.get("adversarial_total_tokens_completion", 0) + \
                   st.session_state.get("evolution_total_tokens_prompt", 0) + \
                   st.session_state.get("evolution_total_tokens_completion", 0)
    
    col1.metric("Total Evolutions", f"{total_evolutions:,}")
    col2.metric("Best Fitness", f"{best_fitness:.4f}")
    col3.metric("Final Approval Rate", f"{final_approval_rate:.1f}%")
    col4.metric("Total Cost ($) ", f"${total_cost:.4f}")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Fitness trend
        if evolution_history:
            fitness_data = []
            generation_numbers = []
            for generation in evolution_history:
                population = generation.get("population", [])
                if population:
                    best_fitness = max(ind.get("fitness", 0) for ind in population)
                    fitness_data.append(best_fitness)
                    generation_numbers.append(generation.get("generation", 0))
            
            if fitness_data:
                df = pd.DataFrame({
                    "Generation": generation_numbers,
                    "Best Fitness": fitness_data
                })
                fig = px.line(df, x="Generation", y="Best Fitness", title="Fitness Trend")
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Approval rate trend
        if adversarial_iterations:
            approval_data = []
            iteration_numbers = []
            for iteration in adversarial_iterations:
                approval_check = iteration.get("approval_check", {})
                approval_rate = approval_check.get("approval_rate", 0)
                approval_data.append(approval_rate)
                iteration_numbers.append(iteration.get("iteration", 0))
            
            if approval_data:
                df = pd.DataFrame({
                    "Iteration": iteration_numbers,
                    "Approval Rate": approval_data
                })
                fig = px.line(df, x="Iteration", y="Approval Rate", title="Approval Rate Trend")
                st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Recent activity
    st.subheader("📝 Recent Activity")
    if evolution_history or adversarial_iterations:
        activity_log = []
        
        # Evolution activity
        for generation in evolution_history[-5:]:  # Last 5 generations
            activity_log.append({
                "Timestamp": time.strftime("%H:%M:%S", time.localtime()),
                "Activity": f"Evolution Generation {generation.get('generation', 0)}",
                "Details": f"Best fitness: {max(ind.get('fitness', 0) for ind in generation.get('population', [])):.4f}"
            })
        
        # Adversarial activity
        for iteration in adversarial_iterations[-5:]:  # Last 5 iterations
            activity_log.append({
                "Timestamp": time.strftime("%H:%M:%S", time.localtime()),
                "Activity": f"Adversarial Iteration {iteration.get('iteration', 0)}",
                "Details": f"Approval rate: {iteration.get('approval_check', {}).get('approval_rate', 0):.1f}%"
            })
        
        # Display recent activity
        if activity_log:
            df = pd.DataFrame(activity_log[-10:])  # Last 10 activities
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No recent activity to display.")


def render_evolution_analytics_tab():
    """Render the evolution analytics tab."""
    st.header("🧬 Evolution Analytics")
    
    evolution_history = st.session_state.get("evolution_history", [])
    
    if not evolution_history:
        st.info("No evolution data available yet. Run an evolution to generate data.")
        return
    
    # Evolution metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_generations = len(evolution_history)
    total_individuals = sum(len(generation.get("population", [])) for generation in evolution_history)
    
    if evolution_history:
        latest_generation = evolution_history[-1]
        population = latest_generation.get("population", [])
        if population:
            best_fitness = max(ind.get("fitness", 0) for ind in population)
            avg_fitness = sum(ind.get("fitness", 0) for ind in population) / len(population)
        else:
            best_fitness = 0
            avg_fitness = 0
    else:
        best_fitness = 0
        avg_fitness = 0
    
    col1.metric("Total Generations", f"{total_generations:,}")
    col2.metric("Total Individuals", f"{total_individuals:,}")
    col3.metric("Best Fitness", f"{best_fitness:.4f}")
    col4.metric("Avg Fitness", f"{avg_fitness:.4f}")
    
    st.divider()
    
    # Fitness trend chart
    st.subheader("📈 Fitness Trend")
    fitness_data = []
    avg_fitness_data = []
    generation_numbers = []
    
    for generation in evolution_history:
        population = generation.get("population", [])
        if population:
            best_fitness = max(ind.get("fitness", 0) for ind in population)
            avg_fitness = sum(ind.get("fitness", 0) for ind in population) / len(population)
            fitness_data.append(best_fitness)
            avg_fitness_data.append(avg_fitness)
            generation_numbers.append(generation.get("generation", 0))
    
    if fitness_data:
        df = pd.DataFrame({
            "Generation": generation_numbers,
            "Best Fitness": fitness_data,
            "Average Fitness": avg_fitness_data
        })
        fig = px.line(df, x="Generation", y=["Best Fitness", "Average Fitness"], title="Fitness Trend Over Generations")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Population diversity
    st.subheader("🌐 Population Diversity")
    diversity_data = []
    generation_numbers = []
    
    for generation in evolution_history:
        population = generation.get("population", [])
        if population and len(population) > 1:
            fitness_values = [ind.get("fitness", 0) for ind in population]
            diversity = np.std(fitness_values) if fitness_values else 0
            diversity_data.append(diversity)
            generation_numbers.append(generation.get("generation", 0))
    
    if diversity_data:
        df = pd.DataFrame({
            "Generation": generation_numbers,
            "Diversity": diversity_data
        })
        fig = px.line(df, x="Generation", y="Diversity", title="Population Diversity Over Generations")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Generation statistics
    st.subheader("📊 Generation Statistics")
    if evolution_history:
        stats_data = []
        for generation in evolution_history:
            population = generation.get("population", [])
            if population:
                fitness_values = [ind.get("fitness", 0) for ind in population]
                stats_data.append({
                    "Generation": generation.get("generation", 0),
                    "Best": max(fitness_values) if fitness_values else 0,
                    "Worst": min(fitness_values) if fitness_values else 0,
                    "Average": sum(fitness_values) / len(fitness_values) if fitness_values else 0,
                    "Std Dev": np.std(fitness_values) if fitness_values else 0,
                    "Population Size": len(population)
                })
        
        if stats_data:
            df = pd.DataFrame(stats_data)
            st.dataframe(df, use_container_width=True)


def render_adversarial_analytics_tab():
    """Render the adversarial analytics tab."""
    st.header("⚔️ Adversarial Analytics")
    
    adversarial_results = st.session_state.get("adversarial_results", {})
    adversarial_iterations = adversarial_results.get("iterations", [])
    
    if not adversarial_iterations:
        st.info("No adversarial testing data available yet. Run adversarial testing to generate data.")
        return
    
    # Adversarial metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_iterations = len(adversarial_iterations)
    if adversarial_iterations:
        latest_iteration = adversarial_iterations[-1]
        approval_check = latest_iteration.get("approval_check", {})
        final_approval_rate = approval_check.get("approval_rate", 0)
    else:
        final_approval_rate = 0
    
    total_issues = 0
    resolved_issues = 0
    for iteration in adversarial_iterations:
        critiques = iteration.get("critiques", [])
        for critique in critiques:
            critique_json = critique.get("critique_json", {})
            if critique_json:
                issues = critique_json.get("issues", [])
                total_issues += len(issues)
        
        patches = iteration.get("patches", [])
        for patch in patches:
            patch_json = patch.get("patch_json", {})
            if patch_json:
                mitigation_matrix = patch_json.get("mitigation_matrix", [])
                resolved_issues += len([m for m in mitigation_matrix if m.get("status", "").lower() in ["resolved", "mitigated"]])
    
    issue_resolution_rate = (resolved_issues / max(1, total_issues)) * 100 if total_issues > 0 else 0
    
    col1.metric("Total Iterations", f"{total_iterations:,}")
    col2.metric("Final Approval Rate", f"{final_approval_rate:.1f}%")
    col3.metric("Total Issues Found", f"{total_issues:,}")
    col4.metric("Issue Resolution Rate", f"{issue_resolution_rate:.1f}%")
    
    st.divider()
    
    # Approval rate trend
    st.subheader("📈 Approval Rate Trend")
    approval_data = []
    iteration_numbers = []
    
    for iteration in adversarial_iterations:
        approval_check = iteration.get("approval_check", {})
        approval_rate = approval_check.get("approval_rate", 0)
        approval_data.append(approval_rate)
        iteration_numbers.append(iteration.get("iteration", 0))
    
    if approval_data:
        df = pd.DataFrame({
            "Iteration": iteration_numbers,
            "Approval Rate": approval_data
        })
        fig = px.line(df, x="Iteration", y="Approval Rate", title="Approval Rate Trend Over Iterations")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Issue severity distribution
    st.subheader("⚠️ Issue Severity Distribution")
    severity_counts = {}
    
    for iteration in adversarial_iterations:
        critiques = iteration.get("critiques", [])
        for critique in critiques:
            critique_json = critique.get("critique_json", {})
            if critique_json:
                issues = critique_json.get("issues", [])
                for issue in issues:
                    severity = issue.get("severity", "low").lower()
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    if severity_counts:
        df = pd.DataFrame({
            "Severity": list(severity_counts.keys()),
            "Count": list(severity_counts.values())
        })
        fig = px.pie(df, values="Count", names="Severity", title="Issue Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Issue resolution trend
    st.subheader("✅ Issue Resolution Trend")
    resolved_data = []
    iteration_numbers = []
    
    for iteration in adversarial_iterations:
        resolved_count = 0
        patches = iteration.get("patches", [])
        for patch in patches:
            patch_json = patch.get("patch_json", {})
            if patch_json:
                mitigation_matrix = patch_json.get("mitigation_matrix", [])
                resolved_count += len([m for m in mitigation_matrix if m.get("status", "").lower() in ["resolved", "mitigated"]])
        resolved_data.append(resolved_count)
        iteration_numbers.append(iteration.get("iteration", 0))
    
    if resolved_data:
        df = pd.DataFrame({
            "Iteration": iteration_numbers,
            "Resolved Issues": resolved_data
        })
        fig = px.line(df, x="Iteration", y="Resolved Issues", title="Issues Resolved Per Iteration")
        st.plotly_chart(fig, use_container_width=True)


def render_model_performance_tab():
    """Render the model performance analytics tab."""
    st.header("🤖 Model Performance")
    
    model_performance = st.session_state.get("adversarial_model_performance", {})
    
    if not model_performance:
        st.info("No model performance data available yet. Run adversarial testing with multiple models to generate data.")
        return
    
    # Model metrics
    st.subheader("🏆 Model Rankings")
    model_data = []
    for model_id, perf_data in model_performance.items():
        model_data.append({
            "Model": model_id,
            "Score": perf_data.get("score", 0),
            "Issues Found": perf_data.get("issues_found", 0)
        })
    
    if model_data:
        df = pd.DataFrame(model_data)
        df_sorted = df.sort_values(by="Score", ascending=False)
        st.dataframe(df_sorted, use_container_width=True)
        
        # Model performance chart
        fig = px.bar(df_sorted, x="Model", y="Score", title="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Issues found chart
        fig2 = px.bar(df_sorted, x="Model", y="Issues Found", title="Issues Found by Model")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()
    
    # Model performance statistics
    st.subheader("📊 Performance Statistics")
    if model_data:
        scores = [data["Score"] for data in model_data]
        issues_found = [data["Issues Found"] for data in model_data]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Models", len(model_data))
        col2.metric("Avg Model Score", f"{np.mean(scores):.2f}")
        col3.metric("Best Model Score", f"{max(scores):.2f}")
        col4.metric("Worst Model Score", f"{min(scores):.2f}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Issues Found", sum(issues_found))
        col2.metric("Avg Issues per Model", f"{np.mean(issues_found):.1f}")
        col3.metric("Best Issues Found", max(issues_found))
        col4.metric("Worst Issues Found", min(issues_found))


def render_export_options():
    """Render export options for analytics data."""
    st.subheader("📤 Export Analytics Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as CSV"):
            st.info("CSV export functionality would be implemented here.")
    
    with col2:
        if st.button("📊 Export as Excel"):
            st.info("Excel export functionality would be implemented here.")
    
    with col3:
        if st.button("📈 Export as PDF"):
            st.info("PDF export functionality would be implemented here.")
    
    st.info("Export options will be available in a future release.")


# Test the analytics dashboard
if __name__ == "__main__":
    # This is for testing purposes only
    print("Analytics dashboard module loaded successfully.")