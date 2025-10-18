"""
OpenEvolve Visualization Module for Frontend
Implements visualization tools for OpenEvolve features including MAP-Elites, evolution trees, and analytics.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


@st.cache_data
def get_evolution_data_from_db(db_path: str) -> Dict[str, Any]:
    """
    Extract evolution data from OpenEvolve database for visualization.
    This is a placeholder that would connect to the actual OpenEvolve database.
    """
    # In a real implementation, this would connect to the OpenEvolve database
    # and extract evolution data including MAP-Elites grid, population metrics, etc.
    return {
        "generations": list(range(1, 51)),
        "best_scores": np.random.uniform(0.1, 1.0, 50).tolist(),
        "average_scores": np.random.uniform(0.05, 0.8, 50).tolist(),
        "diversity_scores": np.random.uniform(0.2, 0.9, 50).tolist(),
        "feature_dimensions": ["complexity", "performance", "readability"],
        "map_elites_grid": [[np.random.uniform(0, 1) for _ in range(10)] for _ in range(10)]
    }


def plot_evolution_progression(data: Dict[str, Any]):
    """Plot evolution progression over generations."""
    if not data:
        st.warning("No evolution data available for visualization")
        return
        
    generations = data.get("generations", [])
    best_scores = data.get("best_scores", [])
    avg_scores = data.get("average_scores", [])
    diversity_scores = data.get("diversity_scores", [])
    
    if not generations or not best_scores:
        st.warning("Insufficient data for progression plot")
        return
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        "Generation": generations,
        "Best Score": best_scores,
        "Average Score": avg_scores,
        "Diversity Score": diversity_scores if len(diversity_scores) == len(generations) else [0.5]*len(generations)
    })
    
    # Melt the dataframe for Plotly
    df_melted = df.melt(id_vars=["Generation"], 
                        value_vars=["Best Score", "Average Score", "Diversity Score"],
                        var_name="Metric", 
                        value_name="Score")
    
    fig = px.line(df_melted, 
                  x="Generation", 
                  y="Score", 
                  color="Metric",
                  title="Evolution Progression Over Generations",
                  labels={"Generation": "Generation", "Score": "Score"})
    
    st.plotly_chart(fig, use_container_width=True)


def plot_map_elites_grid(data: Dict[str, Any]):
    """Plot MAP-Elites feature space visualization."""
    if not data or "map_elites_grid" not in data:
        st.warning("No MAP-Elites grid data available")
        return
        
    grid = data["map_elites_grid"]
    feature_dims = data.get("feature_dimensions", ["Dimension 1", "Dimension 2"])
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        colorscale='Viridis',
        text=np.round(grid, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Performance")
    ))
    
    fig.update_layout(
        title="MAP-Elites Grid Visualization",
        xaxis_title=feature_dims[1] if len(feature_dims) > 1 else "Feature Dimension 2",
        yaxis_title=feature_dims[0] if len(feature_dims) > 0 else "Feature Dimension 1",
        width=600,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_population_diversity(data: Dict[str, Any]):
    """Plot population diversity metrics."""
    if not data:
        st.warning("No diversity data available")
        return
        
    generations = data.get("generations", [])
    diversity_scores = data.get("diversity_scores", [])
    
    if not generations or not diversity_scores:
        st.warning("Insufficient diversity data for plotting")
        return
    
    df = pd.DataFrame({
        "Generation": generations,
        "Diversity Score": diversity_scores
    })
    
    fig = px.line(df, 
                  x="Generation", 
                  y="Diversity Score",
                  title="Population Diversity Over Generations",
                  labels={"Generation": "Generation", "Diversity Score": "Diversity"})
    
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_distribution(data: Dict[str, Any], feature_name: str):
    """Plot distribution of a specific feature."""
    sample_data = np.random.normal(0.5, 0.2, 1000)
    df = pd.DataFrame({feature_name: sample_data})
    
    fig = px.histogram(df, 
                       x=feature_name,
                       title=f"Distribution of {feature_name}",
                       nbins=30)
    
    st.plotly_chart(fig, use_container_width=True)


def render_openevolve_visualization_ui():
    """Render OpenEvolve visualization UI components in the frontend."""
    st.header("🧬 OpenEvolve Visualization Dashboard")
    
    st.markdown("""
    This dashboard visualizes OpenEvolve's advanced evolutionary features including:
    - Quality-Diversity (MAP-Elites) evolution
    - Multi-generational progression
    - Population diversity metrics
    - Feature space exploration
    """)
    
    # Sample data for demonstration
    sample_data = {
        "generations": list(range(1, 101)),
        "best_scores": [0.1 + 0.8 * (1 - np.exp(-i/50)) + np.random.normal(0, 0.05) for i in range(100)],
        "average_scores": [0.05 + 0.6 * (1 - np.exp(-i/60)) + np.random.normal(0, 0.05) for i in range(100)],
        "diversity_scores": [0.2 + 0.6 * (1 - np.exp(-i/40)) + np.random.normal(0, 0.05) for i in range(100)],
        "feature_dimensions": ["complexity", "performance"],
        "map_elites_grid": [[np.random.uniform(0, 1) for _ in range(10)] for _ in range(10)]
    }
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Evolution Progression", "🗺️ MAP-Elites Grid", "🧬 Diversity Metrics", "🔍 Feature Analysis"])
    
    with tab1:
        st.subheader("Evolution Over Generations")
        plot_evolution_progression(sample_data)
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Generation", len(sample_data["generations"]))
        with col2:
            st.metric("Best Score", f"{max(sample_data['best_scores']):.3f}")
        with col3:
            st.metric("Final Diversity", f"{sample_data['diversity_scores'][-1]:.3f}")
    
    with tab2:
        st.subheader("Quality-Diversity (MAP-Elites) Grid")
        st.info("This grid represents the quality-diversity archive where each cell corresponds to a specific combination of feature dimensions")
        plot_map_elites_grid(sample_data)
    
    with tab3:
        st.subheader("Population Diversity Analysis")
        plot_population_diversity(sample_data)
        
        st.markdown("""
        **Diversity Metrics Explained:**
        - High diversity promotes exploration of different solution strategies
        - Balancing diversity with performance leads to robust solutions
        - OpenEvolve maintains diversity through MAP-Elites and island models
        """)
    
    with tab4:
        st.subheader("Feature Space Analysis")
        
        feature_option = st.selectbox(
            "Select Feature to Analyze",
            ["Complexity", "Performance", "Readability", "Efficiency", "Robustness"]
        )
        
        plot_feature_distribution(sample_data, feature_option)
        
        st.markdown("""
        **Feature Dimensions in OpenEvolve:**
        - **Complexity**: How complex the generated solution is
        - **Performance**: How well the solution performs its task
        - **Readability**: How readable and maintainable the solution is
        - **Efficiency**: Resource usage efficiency
        - **Robustness**: How well the solution handles edge cases
        """)
    
    st.markdown("---")
    st.subheader("🚀 Advanced OpenEvolve Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Quality-Diversity Optimization:**
        - Maintains diverse, high-performing solutions
        - Explores trade-offs between competing objectives
        - Provides multiple solution pathways
        """)
        
    with col2:
        st.markdown("""
        **Island Model Evolution:**
        - Multiple populations evolving in parallel
        - Migration between populations prevents convergence
        - Better exploration of solution space
        """)
    
    st.markdown("""
    ---
    **OpenEvolve Visualization Note:** This dashboard demonstrates the visualization
    capabilities for OpenEvolve's advanced features. In a complete integration, 
    this would connect directly to the OpenEvolve database to provide real-time
    visualization of ongoing evolutionary runs.
    """)


class OpenEvolveDataProcessor:
    """Processes OpenEvolve evolution data for visualization and analysis."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir
        
    def load_evolution_history(self, evolution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load evolution history from OpenEvolve output.
        This would connect to actual OpenEvolve history in a real implementation.
        """
        return get_evolution_data_from_db(self.output_dir or "default_db")
    
    def extract_feature_trajectories(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Extract feature trajectories over evolution."""
        return {
            "trajectories": {
                "complexity": np.random.uniform(0.2, 0.9, 50).tolist(),
                "performance": np.random.uniform(0.1, 1.0, 50).tolist(),
                "diversity": np.random.uniform(0.3, 0.8, 50).tolist()
            },
            "generations": list(range(1, 51))
        }
    
    def generate_evolution_report(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive evolution report."""
        return {
            "summary": {
                "total_generations": len(history.get("generations", [])),
                "best_score": max(history.get("best_scores", [0])),
                "final_diversity": history.get("diversity_scores", [0.5])[-1],
                "improvement_rate": (max(history.get("best_scores", [0])) - min(history.get("best_scores", [100]))) / len(history.get("generations", [1]))
            },
            "feature_analysis": self.extract_feature_trajectories(history)
        }


def render_evolution_insights():
    """Render advanced evolution insights and analytics."""
    st.header("🔬 OpenEvolve Evolution Insights")
    
    processor = OpenEvolveDataProcessor()
    sample_history = processor.load_evolution_history()
    report = processor.generate_evolution_report(sample_history)
    
    # Display summary metrics
    summary = report["summary"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Generations", summary["total_generations"])
    with col2:
        st.metric("Best Score", f"{summary['best_score']:.3f}")
    with col3:
        st.metric("Final Diversity", f"{summary['final_diversity']:.3f}")
    with col4:
        st.metric("Improvement Rate", f"{summary['improvement_rate']:.3f}")
    
    # Create tabs for different analysis views
    analysis_tabs = st.tabs(["📈 General Insights", "🧬 Feature Analysis", "🎯 Performance Analysis", "🔄 Convergence Analysis"])
    
    with analysis_tabs[0]:  # General Insights
        st.subheader("Evolution Overview")
        
        # Evolution statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_performance = np.mean(sample_history["best_scores"]) if sample_history["best_scores"] else 0
            st.metric("Avg. Performance", f"{avg_performance:.3f}")
        with col2:
            max_diversity = max(sample_history["diversity_scores"]) if sample_history["diversity_scores"] else 0.5
            st.metric("Max Diversity", f"{max_diversity:.3f}")
        with col3:
            improvement_trend = (summary['best_score'] - sample_history["best_scores"][0]) / len(sample_history["generations"]) if sample_history["best_scores"] else 0
            st.metric("Avg. Improvement/Gen", f"{improvement_trend:.4f}")
        
        # Evolution progression plot
        plot_evolution_progression(sample_history)
    
    with analysis_tabs[1]:  # Feature Analysis
        st.subheader("Feature Space Exploration")
        
        # Show feature evolution trajectories
        trajectories = report["feature_analysis"]["trajectories"]
        generations = report["feature_analysis"]["generations"]
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame({
            "Generation": generations,
            "Complexity": trajectories["complexity"],
            "Performance": trajectories["performance"],
            "Diversity": trajectories["diversity"]
        })
        
        df_melted = df.melt(id_vars=["Generation"], 
                            value_vars=["Complexity", "Performance", "Diversity"],
                            var_name="Feature", 
                            value_name="Value")
        
        fig = px.line(df_melted, 
                      x="Generation", 
                      y="Value", 
                      color="Feature",
                      title="Feature Evolution Trajectories",
                      labels={"Generation": "Generation", "Value": "Feature Value"})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Feature Correlations")
        feature_corr = df[["Complexity", "Performance", "Diversity"]].corr()
        fig_corr = px.imshow(feature_corr, 
                             title="Feature Correlation Matrix",
                             text_auto=True,
                             aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with analysis_tabs[2]:  # Performance Analysis
        st.subheader("Performance Analysis")
        
        # Create performance metrics
        performance_data = {
            "Generation": sample_history["generations"],
            "Best Score": sample_history["best_scores"],
            "Average Score": sample_history["average_scores"],
            "Diversity": sample_history["diversity_scores"]
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        # Performance metrics over time
        fig_perf = px.line(perf_df, 
                          x="Generation", 
                          y=["Best Score", "Average Score"],
                          title="Performance Over Time",
                          labels={"value": "Score", "variable": "Metric"})
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Performance distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Best Score Distribution")
            best_scores = np.array(sample_history["best_scores"])
            fig_hist = px.histogram(x=best_scores, nbins=20, title="Distribution of Best Scores")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("Improvement Rate")
            if len(best_scores) > 1:
                improvements = np.diff(best_scores)
                fig_improve = px.histogram(x=improvements, nbins=20, title="Distribution of Improvements")
                st.plotly_chart(fig_improve, use_container_width=True)
    
    with analysis_tabs[3]:  # Convergence Analysis
        st.subheader("Convergence Analysis")
        
        # Moving average to identify trends
        df_perf = pd.DataFrame({
            "Generation": sample_history["generations"],
            "Best Score": sample_history["best_scores"]
        })
        
        # Calculate moving average
        window_size = max(1, len(df_perf) // 10)  # 10% of total generations
        df_perf['Moving_Avg'] = df_perf['Best Score'].rolling(window=window_size, center=True).mean()
        
        fig_convergence = px.line(df_perf,
                                 x="Generation", 
                                 y=["Best Score", "Moving_Avg"],
                                 title=f"Convergence Analysis (Moving Avg: {window_size} gens)",
                                 labels={"value": "Score", "variable": "Metric"})
        st.plotly_chart(fig_convergence, use_container_width=True)
        
        # Convergence metrics
        recent_scores = sample_history["best_scores"][-10:] if len(sample_history["best_scores"]) >= 10 else sample_history["best_scores"]
        if len(recent_scores) > 1:
            recent_std = np.std(recent_scores)
            recent_avg = np.mean(recent_scores)
            st.metric("Recent Score Stability (Std)", f"{recent_std:.4f}")
            st.metric("Recent Average Score", f"{recent_avg:.3f}")
            
            if recent_std < 0.01:  # Very low standard deviation indicates convergence
                st.info("✅ Evolution appears to have converged")
            elif recent_std < 0.05:
                st.info("🔄 Evolution may be approaching convergence")
            else:
                st.info("📈 Evolution still actively exploring")
    
    st.markdown("""
    **Insights on Evolution:**
    - **Complexity vs Performance**: Usually trade-off between complexity and performance
    - **Diversity Maintenance**: Critical for continued exploration of solution space
    - **Convergence Patterns**: Understanding when and how solutions converge
    - **Improvement Rate**: Monitoring the rate of improvement over generations
    """)


def render_advanced_diagnostics():
    """Render advanced diagnostics and algorithm discovery insights."""
    st.header("🔍 Advanced Diagnostics Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["🧠 Algorithm Discovery", "🧮 Symbolic Regression", "🤖 Neuroevolution"])
    
    with tab1:  # Algorithm Discovery
        st.subheader("Algorithm Discovery Analytics")
        
        # Sample algorithm discovery metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Discovered Approaches", 24)
        with col2:
            st.metric("Novel Algorithms", 8)
        with col3:
            st.metric("Performance Gain", "+47%")
        
        # Visualization of algorithm approaches
        approaches = ["Divide & Conquer", "Dynamic Programming", "Greedy", "Graph-Based", "Heuristic", "Probabilistic"]
        performance = [0.85, 0.92, 0.78, 0.88, 0.75, 0.82]
        
        df_approaches = pd.DataFrame({
            "Algorithm Approach": approaches,
            "Performance Score": performance
        })
        
        fig_approaches = px.bar(df_approaches, 
                               x="Algorithm Approach", 
                               y="Performance Score",
                               title="Performance by Algorithmic Approach",
                               color="Performance Score",
                               color_continuous_scale="viridis")
        st.plotly_chart(fig_approaches, use_container_width=True)
        
        st.markdown("""
        **Algorithm Discovery Insights:**
        - **Dynamic Programming** approaches showed highest performance
        - **Greedy algorithms** had lower performance but faster execution
        - **Hybrid approaches** combined benefits of multiple paradigms
        """)
    
    with tab2:  # Symbolic Regression
        st.subheader("Symbolic Regression Analytics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fitted Equations", 15)
        with col2:
            st.metric("Best R² Score", "0.994")
        with col3:
            st.metric("Avg. Complexity", 4.2)
        
        # Sample symbolic regression results
        equations = ["x² + 2x + 1", "sin(x) + 0.5x", "x³ - 2x² + x", "exp(-x/2) * cos(x)", "log(x + 1)"]
        accuracy = [0.994, 0.982, 0.978, 0.951, 0.967]
        complexity = [3, 4, 3, 5, 4]
        
        df_symbolic = pd.DataFrame({
            "Equation": equations,
            "Accuracy": accuracy,
            "Complexity": complexity
        })
        
        fig_symbolic = px.scatter(df_symbolic, 
                                 x="Complexity", 
                                 y="Accuracy", 
                                 size="Accuracy",
                                 hover_data=["Equation"],
                                 title="Accuracy vs. Complexity for Discovered Equations",
                                 color="Accuracy",
                                 color_continuous_scale="plasma")
        st.plotly_chart(fig_symbolic, use_container_width=True)
        
        st.markdown("""
        **Symbolic Regression Insights:**
        - **Simple polynomial equations** often achieve high accuracy
        - **Trigonometric functions** capture periodic patterns effectively
        - **Trade-off** between complexity and interpretability exists
        """)
    
    with tab3:  # Neuroevolution
        st.subheader("Neuroevolution Analytics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Network Architectures", 32)
        with col2:
            st.metric("Best Accuracy", "94.2%")
        with col3:
            st.metric("Parameters Optimized", "2.1M")
        
        # Sample neural network results
        architectures = ["Feedforward", "CNN", "LSTM", "Transformer", "Custom Hybrid"]
        accuracy = [0.89, 0.94, 0.91, 0.93, 0.95]
        efficiency = [0.85, 0.72, 0.68, 0.55, 0.70]  # Higher is more efficient
        
        df_neural = pd.DataFrame({
            "Architecture": architectures,
            "Accuracy": accuracy,
            "Efficiency": efficiency
        })
        
        fig_neural = px.scatter(df_neural, 
                              x="Efficiency", 
                              y="Accuracy", 
                              size="Accuracy",
                              hover_data=["Architecture"],
                              title="Accuracy vs. Efficiency for Evolved Networks",
                              color="Architecture")
        st.plotly_chart(fig_neural, use_container_width=True)
        
        st.markdown("""
        **Neuroevolution Insights:**
        - **Custom hybrid architectures** achieved highest accuracy
        - **CNNs** provided good balance of accuracy and efficiency
        - **Efficiency optimization** is crucial for deployment
        """)


def render_openevolve_advanced_ui():
    """Render the complete OpenEvolve advanced UI with all features."""
    st.header("🧬 OpenEvolve Advanced Dashboard")
    
    # Main dashboard tabs
    main_tabs = st.tabs(["📊 Evolution Dashboard", "🔍 Advanced Diagnostics", "⚙️ Configuration", "📈 Performance Metrics"])
    
    with main_tabs[0]:  # Evolution Dashboard
        render_openevolve_visualization_ui()
    
    with main_tabs[1]:  # Advanced Diagnostics
        render_advanced_diagnostics()
    
    with main_tabs[2]:  # Configuration
        st.subheader("OpenEvolve Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Core Configuration")
            max_iterations = st.number_input("Max Iterations", min_value=1, max_value=10000, value=100)
            population_size = st.number_input("Population Size", min_value=1, max_value=10000, value=100)
            num_islands = st.slider("Number of Islands", 1, 20, 5)
            archive_size = st.slider("Archive Size", 10, 1000, 100)
        
        with col2:
            st.markdown("### Evolution Strategy")
            elite_ratio = st.slider("Elite Ratio", 0.0, 1.0, 0.1, 0.01)
            exploration_ratio = st.slider("Exploration Ratio", 0.0, 1.0, 0.2, 0.01)
            exploitation_ratio = st.slider("Exploitation Ratio", 0.0, 1.0, 0.7, 0.01)
            
            # Ensure ratios sum to 1
            total_ratio = elite_ratio + exploration_ratio + exploitation_ratio
            if abs(total_ratio - 1.0) > 0.01:
                st.warning(f"Ratios sum to {total_ratio:.2f}, ideally should sum to 1.0")
        
        st.markdown("### Feature Dimensions")
        feature_dims = st.multiselect(
            "Feature Dimensions",
            ["complexity", "diversity", "performance", "efficiency", "accuracy", "readability", "robustness"],
            default=["complexity", "diversity"]
        )
        
        st.markdown("### Advanced Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            use_cascade = st.checkbox("Cascade Evaluation", value=True)
            use_artifacts = st.checkbox("Enable Artifacts", value=True)
        with col2:
            use_llm_feedback = st.checkbox("LLM Feedback", value=False)
        with col3:
            # Add any advanced options for the third column or leave it empty intentionally
            pass
        
        # Configuration summary
        st.markdown(f"""
        **Configuration Summary:**
        - Iterations: {max_iterations}, Population: {population_size}, Islands: {num_islands}
        - Archive: {archive_size}, Feature Dims: {len(feature_dims)}
        - Strategy: Elite {elite_ratio:.1f}, Explore {exploration_ratio:.1f}, Exploit {exploitation_ratio:.1f}
        - Advanced: Cascade={use_cascade}, Artifacts={use_artifacts}, LLM Feedback={use_llm_feedback}
        """)
    
    with main_tabs[3]:  # Performance Metrics
        render_evolution_insights()