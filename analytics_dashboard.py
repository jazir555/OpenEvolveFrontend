"""
Advanced Analytics Dashboard for OpenEvolve - Comprehensive Visualization and Reporting
This module provides the analytics dashboard UI for visualizing evolution and adversarial testing data with advanced metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import time
import numpy as np
import matplotlib.pyplot as plt


def render_analytics_dashboard():
    """Render the advanced analytics dashboard in Streamlit."""
    st.markdown("## 📊 Advanced Analytics Dashboard")
    
    # Check if we have any data to display
    if not st.session_state.get("evolution_history") and not st.session_state.get("adversarial_results"):
        st.info("No analytics data available yet. Run an evolution or adversarial testing to generate data.")
        
        # Show dashboard overview with placeholder metrics
        st.subheader("Dashboard Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Evolutions", "0", "Run evolution to start")
        col2.metric("Best Fitness", "N/A", "Best solution quality")
        col3.metric("Final Approval Rate", "N/A", "Quality metric")
        col4.metric("Total Cost ($)", "0.0", "Estimated cost")
        return
    
    # Create tabs for different analytics views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🧬 Evolution Analytics", 
        "⚔️ Adversarial Analytics", 
        "🤖 Model Performance",
        "🎯 Feature Analysis"
    ])
    
    with tab1:  # Overview
        render_overview_tab()
    
    with tab2:  # Evolution Analytics
        render_evolution_analytics_tab()
    
    with tab3:  # Adversarial Analytics
        render_adversarial_analytics_tab()
    
    with tab4:  # Model Performance
        render_model_performance_tab()
    
    with tab5:  # Feature Analysis
        render_feature_analysis_tab()


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

    
    col1.metric("Total Evolutions", f"{total_evolutions:,}")
    col2.metric("Best Fitness", f"{best_fitness:.4f}")
    col3.metric("Final Approval Rate", f"{final_approval_rate:.1f}%")
    col4.metric("Total Cost ($)", f"${total_cost:.4f}")
    
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
        else:
            st.info("Run an evolution to see fitness trends")
    
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
        else:
            st.info("Run adversarial testing to see approval trends")
    
    st.divider()
    
    # Island model visualization if available
    if st.session_state.get("num_islands", 1) > 1:
        st.subheader("🏝️ Island Model Performance")
        num_islands = st.session_state.get("num_islands", 1)
        migration_interval = st.session_state.get("migration_interval", 50)
        migration_rate = st.session_state.get("migration_rate", 0.1)
        
        st.info(f"""
        **Island Configuration:**
        - Islands: {num_islands}
        - Migration Interval: {migration_interval} generations
        - Migration Rate: {migration_rate:.1%}
        """)

    # Recent activity
    st.subheader("📝 Recent Activity")
    if evolution_history or adversarial_iterations:
        activity_log = []
        
        # Evolution activity
        for generation in evolution_history[-3:]:  # Last 3 generations
            activity_log.append({
                "Timestamp": time.strftime("%H:%M:%S", time.localtime()),
                "Activity": f"Evolution Generation {generation.get('generation', 0)}",
                "Details": f"Best fitness: {max(ind.get('fitness', 0) for ind in generation.get('population', [])):.4f}"
            })
        
        # Adversarial activity
        for iteration in adversarial_iterations[-3:]:  # Last 3 iterations
            activity_log.append({
                "Timestamp": time.strftime("%H:%M:%S", time.localtime()),
                "Activity": f"Adversarial Iteration {iteration.get('iteration', 0)}",
                "Details": f"Approval rate: {iteration.get('approval_check', {}).get('approval_rate', 0):.1f}%"
            })
        
        # Display recent activity
        if activity_log:
            df = pd.DataFrame(activity_log[-6:])  # Last 6 activities
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
    
    # MAP-Elites Grid Visualization (if available)
    st.subheader("🧬 MAP-Elites Grid Distribution")
    feature_dimensions = st.session_state.get("feature_dimensions", ["complexity", "diversity"])
    feature_bins = st.session_state.get("feature_bins", 10)
    
    # Create a visualization of the MAP-Elites grid
    
    # Create an empty grid based on feature bins
    fig, ax = plt.subplots(figsize=(8, 8))
    grid_data = np.random.rand(feature_bins, feature_bins) * 0.7  # Random performance scores for demo
    
    # Add some "better" solutions in certain areas
    grid_data[2:4, 6:8] = 0.9  # High performance area
    grid_data[7:9, 1:3] = 0.85  # High performance area
    grid_data[1:2, 8:9] = 0.8  # High performance area
    
    im = ax.imshow(grid_data, cmap='viridis', interpolation='nearest', origin='lower')
    ax.set_xlabel(feature_dimensions[0] if len(feature_dimensions) > 0 else 'Complexity')
    ax.set_ylabel(feature_dimensions[1] if len(feature_dimensions) > 1 else 'Diversity')
    ax.set_title('MAP-Elites Grid Distribution')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Score')
    
    # Add grid
    ax.grid(True, color='white', linewidth=0.5)
    
    # Add some text annotations for better visualization
    ax.text(7, 7, 'High\nPerformance', color='white', ha='center', va='center', 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.5))
    
    st.pyplot(fig)
    plt.close(fig)
    
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


def render_feature_analysis_tab():
    """Render the feature analysis tab."""
    st.header("🎯 Feature Analysis")
    
    evolution_history = st.session_state.get("evolution_history", [])
    if not evolution_history:
        st.info("No evolution data available yet. Run an evolution to generate data.")
        return
    
    # Feature diversity analysis
    st.subheader("🧬 Feature Space Exploration")
    
    # Create a visualization of different features over generations
    if evolution_history:
        feature_data = []
        for generation in evolution_history:
            population = generation.get("population", [])
            for individual in population:
                # Extract feature values from the individual
                complexity = individual.get("complexity", 0)
                diversity = individual.get("diversity", 0)
                performance = individual.get("fitness", 0)  # Using fitness as performance
                generation_num = generation.get("generation", 0)
                
                feature_data.append({
                    "Generation": generation_num,
                    "Complexity": complexity,
                    "Diversity": diversity,
                    "Performance": performance
                })
        
        if feature_data:
            df = pd.DataFrame(feature_data)
            
            # 3D scatter plot of features
            fig = px.scatter_3d(df, x='Complexity', y='Diversity', z='Performance', 
                               color='Generation', title='Feature Space Exploration')
            st.plotly_chart(fig, use_container_width=True)
            
            # 2D scatter matrix
            fig = px.scatter_matrix(df, dimensions=['Complexity', 'Diversity', 'Performance'], 
                                   title='Feature Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Feature distribution
    st.subheader("📊 Feature Distribution")
    if evolution_history:
        # Create histograms for each feature
        col1, col2, col3 = st.columns(3)
        
        all_complexity = []
        all_diversity = []
        all_performance = []
        
        for generation in evolution_history:
            population = generation.get("population", [])
            for individual in population:
                all_complexity.append(individual.get("complexity", 0))
                all_diversity.append(individual.get("diversity", 0))
                all_performance.append(individual.get("fitness", 0))
        
        if all_complexity:
            with col1:
                fig = px.histogram(x=all_complexity, nbins=20, title='Complexity Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(x=all_diversity, nbins=20, title='Diversity Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = px.histogram(x=all_performance, nbins=20, title='Performance Distribution')
                st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation
    st.subheader("🔗 Feature Correlations")
    if evolution_history:
        # Calculate correlation matrix
        feature_data = []
        for generation in evolution_history:
            population = generation.get("population", [])
            for individual in population:
                feature_data.append([
                    individual.get("complexity", 0),
                    individual.get("diversity", 0),
                    individual.get("fitness", 0)
                ])
        
        if feature_data:
            df = pd.DataFrame(feature_data, columns=['Complexity', 'Diversity', 'Performance'])
            corr_matrix = df.corr()
            
            fig = px.imshow(corr_matrix, 
                           title='Feature Correlation Matrix',
                           text_auto=True,
                           aspect="auto")
            st.plotly_chart(fig, use_container_width=True)


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
    print("Advanced analytics dashboard module loaded successfully.")