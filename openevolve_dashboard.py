
"""
OpenEvolve Dashboard
A high-level overview of the OpenEvolve system.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import time
from openevolve_orchestrator import OpenEvolveOrchestrator

def render_openevolve_dashboard():
    """Render the OpenEvolve dashboard."""
    st.header("🧬 OpenEvolve Dashboard")
    st.markdown("A high-level overview of the OpenEvolve system.")

    if 'orchestrator' not in st.session_state:
        st.warning("Orchestrator not available. Please go to the Orchestrator tab to start the services.")
        return

    orchestrator = st.session_state.orchestrator

    # --- System Status ---
    st.subheader("System Status")
    # In a real application, you would get the status from the orchestrator
    # For now, we'll simulate the status
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Visualizer Service", value="Running", delta="Port 8080")
    with col2:
        st.metric(label="LLM Proxy Service", value="Running", delta="Port 8000")

    # --- Key Metrics ---
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    # These would be calculated from the evolution history and adversarial results
    total_evolutions = len(st.session_state.get("evolution_history", []))
    # peak_fitness = max((ind['fitness'] for gen in st.session_state.get("evolution_history", []) for ind in gen.get('population', [])), default=0)
    total_cost = st.session_state.get("adversarial_cost_estimate_usd", 0.0)
    col1.metric("Total Evolutions", f"{total_evolutions:,}")
    # col2.metric("Peak Fitness", f"{peak_fitness:.4f}")
    col3.metric("Total Cost (USD)", f"${total_cost:.4f}")


    # --- Active Workflows ---
    st.subheader("Active Workflows")
    active_workflows = orchestrator.get_active_workflows()
    if not active_workflows:
        st.info("No active workflows.")
    else:
        for workflow in active_workflows:
            with st.container():
                st.subheader(f"Workflow: {workflow['workflow_id']}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Type", workflow['workflow_type'])
                col2.metric("Status", workflow['status'])
                col3.metric("Progress", f"{workflow['progress'] * 100:.2f}%")
                st.progress(workflow['progress'])

    # --- Performance Charts ---
    st.subheader("Performance Charts")
    
    # Fitness Trend
    st.markdown("**Fitness Trend Over Generations**")
    if st.session_state.get("evolution_history"):
        fitness_data = []
        for gen in st.session_state.evolution_history:
            pop = gen.get('population', [])
            if pop:
                fitness_data.append({
                    'Generation': gen.get("generation", 0),
                    'Best Fitness': max(ind.get("fitness", 0) for ind in pop),
                    'Average Fitness': sum(ind.get("fitness", 0) for ind in pop) / len(pop)
                })
        if fitness_data:
            df = pd.DataFrame(fitness_data)
            fig = px.line(df, x="Generation", y=["Best Fitness", "Average Fitness"], title="Fitness Trend")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run an evolution to see fitness trends.")

    # Model Performance
    st.markdown("**Model Performance Overview**")
    model_performance = st.session_state.get("adversarial_model_performance", {})
    if model_performance:
        model_data = [{"Model": k, "Score": v.get("score", 0), "Cost": v.get("cost", 0.0)} for k, v in model_performance.items()]
        df = pd.DataFrame(model_data).sort_values(by="Score", ascending=False)
        fig = px.bar(df, x="Model", y="Score", title="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run adversarial testing to see model performance data.")

# This function is not used in the new dashboard, but we keep it for now to avoid breaking imports
def render_openevolve_workflow():
    pass

# This function is not used in the new dashboard, but we keep it for now to avoid breaking imports
def initialize_openevolve_session():
    pass