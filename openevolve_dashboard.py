"""
Comprehensive OpenEvolve Dashboard
Unified interface for all OpenEvolve features and capabilities
"""
import streamlit as st
import pandas as pd
import plotly.express as px

# Import OpenEvolve components
from openevolve_visualization import render_evolution_insights
from monitoring_system import render_comprehensive_monitoring_ui
from reporting_system import render_reporting_dashboard


def render_openevolve_dashboard():
    """Render the comprehensive OpenEvolve dashboard."""
    st.header("🧬 OpenEvolve: Advanced Evolution Platform")
    
    st.markdown("""
    ### Welcome to OpenEvolve Dashboard
    
    This dashboard provides access to all OpenEvolve research-grade evolutionary computing capabilities:
    - **Quality-Diversity Evolution** (MAP-Elites)
    - **Multi-Objective Optimization** (Pareto fronts)
    - **Adversarial Evolution** (Red Team/Blue Team)
    - **Symbolic Regression** (Mathematical discovery)
    - **Neuroevolution** (Neural architecture search)
    - **Algorithm Discovery** (Novel algorithm design)
    """)
    
    # Main navigation tabs for OpenEvolve features
    main_tabs = st.tabs([
        "🎯 Evolution Modes", 
        "📊 Live Analytics", 
        "🎛️ Configuration", 
        "📈 Performance", 
        "📋 Reports"
    ])
    
    with main_tabs[0]:  # Evolution Modes
        render_evolution_modes_tab()
    
    with main_tabs[1]:  # Live Analytics
        render_live_analytics_tab()
    
    with main_tabs[2]:  # Configuration
        render_configuration_tab()
    
    with main_tabs[3]:  # Performance
        render_performance_tab()
    
    with main_tabs[4]:  # Reports
        render_reports_tab()


def render_evolution_modes_tab():
    """Render the evolution modes selection tab."""
    st.subheader("Choose Evolution Mode")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        standard_evolution = st.button("🧬 Standard Evolution", 
                                     help="Traditional genetic programming evolution",
                                     use_container_width=True)
        quality_diversity = st.button("🎯 Quality-Diversity (MAP-Elites)", 
                                    help="Maintain diverse, high-performing solutions across feature dimensions",
                                    use_container_width=True)
        multi_objective = st.button("⚖️ Multi-Objective Optimization", 
                                  help="Optimize for multiple competing objectives simultaneously",
                                  use_container_width=True)
    
    with col2:
        adversarial = st.button("⚔️ Adversarial Evolution", 
                              help="Red Team/Blue Team approach for robustness",
                              use_container_width=True)
        symbolic_regression = st.button("🔍 Symbolic Regression", 
                                      help="Discover mathematical expressions from data",
                                      use_container_width=True)
        neuroevolution = st.button("🧠 Neuroevolution", 
                                 help="Evolve neural network architectures",
                                 use_container_width=True)
    
    with col3:
        algorithm_discovery = st.button("💡 Algorithm Discovery", 
                                      help="Discover novel algorithmic approaches",
                                      use_container_width=True)
        prompt_evolution = st.button("📝 Prompt Evolution", 
                                   help="Optimize prompts for LLMs",
                                   use_container_width=True)
        custom_evolution = st.button("🛠️ Custom Evolution", 
                                   help="Customizable evolution parameters",
                                   use_container_width=True)
    
    # Handle button clicks by setting session state
    if standard_evolution:
        st.session_state.evolution_mode = "standard"
        st.success("Standard Evolution mode selected")
    elif quality_diversity:
        st.session_state.evolution_mode = "quality_diversity"
        st.success("Quality-Diversity Evolution mode selected")
    elif multi_objective:
        st.session_state.evolution_mode = "multi_objective"
        st.success("Multi-Objective Evolution mode selected")
    elif adversarial:
        st.session_state.evolution_mode = "adversarial"
        st.success("Adversarial Evolution mode selected")
    elif symbolic_regression:
        st.session_state.evolution_mode = "symbolic_regression"
        st.success("Symbolic Regression mode selected")
    elif neuroevolution:
        st.session_state.evolution_mode = "neuroevolution"
        st.success("Neuroevolution mode selected")
    elif algorithm_discovery:
        st.session_state.evolution_mode = "algorithm_discovery"
        st.success("Algorithm Discovery mode selected")
    elif prompt_evolution:
        st.session_state.evolution_mode = "prompt_optimization"
        st.success("Prompt Evolution mode selected")
    elif custom_evolution:
        st.session_state.evolution_mode = "custom"
        st.success("Custom Evolution mode selected")


def render_live_analytics_tab():
    """Render live analytics from ongoing evolution runs."""
    st.subheader("Live Evolution Analytics")
    
    # Check if there's an active evolution run
    if st.session_state.get("evolution_running", False):
        st.info("Evolution is currently running. Live data will appear here.")
        
        # Live metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Generation", st.session_state.get("current_generation", 0))
        with col2:
            st.metric("Best Score", f"{st.session_state.get('best_score', 0.0):.3f}")
        with col3:
            st.metric("Population Size", st.session_state.get("population_size", 100))
        with col4:
            st.metric("Archive Size", st.session_state.get("archive_size", 0))
        
        # Real-time chart (simulated for now)
        st.subheader("Performance Over Time")
        progress_data = {
            "Generation": list(range(1, 11)),
            "Best Score": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.75, 0.82, 0.87, 0.91]
        }
        df = pd.DataFrame(progress_data)
        fig = px.line(df, x="Generation", y="Best Score", title="Best Score Progression")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active evolution run. Start an evolution to see live analytics.")
        
        # Display recent results if available
        if "evolution_history" in st.session_state and st.session_state.evolution_history:
            st.subheader("Recent Evolution Results")
            render_evolution_insights()


def render_configuration_tab():
    """Render advanced OpenEvolve configuration."""
    st.subheader("OpenEvolve Configuration")
    
    config_tabs = st.tabs([" Core Settings", " Island Model", " Feature Dimensions", " Advanced"])
    
    with config_tabs[0]:  # Core Settings
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.max_iterations = st.number_input(
                "Max Iterations", 
                min_value=1, 
                max_value=10000, 
                value=st.session_state.get("max_iterations", 100),
                help="Maximum number of evolutionary iterations"
            )
            st.session_state.population_size = st.number_input(
                "Population Size", 
                min_value=10, 
                max_value=10000, 
                value=st.session_state.get("population_size", 100),
                help="Size of the population in each generation"
            )
            st.session_state.temperature = st.slider(
                "LLM Temperature", 
                min_value=0.0, 
                max_value=2.0, 
                value=st.session_state.get("temperature", 0.7),
                step=0.1,
                help="Temperature for LLM generation (higher = more creative)"
            )
        
        with col2:
            st.session_state.max_tokens = st.number_input(
                "Max Tokens", 
                min_value=100, 
                max_value=32000, 
                value=st.session_state.get("max_tokens", 4096),
                help="Maximum tokens for LLM responses"
            )
            st.session_state.top_p = st.slider(
                "Top-P Sampling", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.get("top_p", 0.95),
                step=0.05,
                help="Top-P sampling parameter for generation"
            )
            st.session_state.seed = st.number_input(
                "Random Seed", 
                value=st.session_state.get("seed", 42),
                help="Seed for reproducible evolution runs"
            )
    
    with config_tabs[1]:  # Island Model
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.num_islands = st.number_input(
                "Number of Islands", 
                min_value=1, 
                max_value=20, 
                value=st.session_state.get("num_islands", 3),
                help="Number of parallel populations in island model"
            )
        with col2:
            st.session_state.migration_interval = st.number_input(
                "Migration Interval", 
                min_value=1, 
                max_value=1000, 
                value=st.session_state.get("migration_interval", 25),
                help="How often individuals migrate between islands"
            )
        with col3:
            st.session_state.migration_rate = st.slider(
                "Migration Rate", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.get("migration_rate", 0.1),
                step=0.01,
                help="Proportion of individuals that migrate"
            )
    
    with config_tabs[2]:  # Feature Dimensions
        st.session_state.feature_dimensions = st.multiselect(
            "Feature Dimensions (for MAP-Elites)",
            options=["complexity", "diversity", "performance", "readability", "efficiency", "accuracy", "robustness"],
            default=st.session_state.get("feature_dimensions", ["complexity", "diversity"]),
            help="Dimensions for quality-diversity optimization"
        )
        
        st.session_state.feature_bins = st.slider(
            "Feature Bins", 
            min_value=5, 
            max_value=50, 
            value=st.session_state.get("feature_bins", 10),
            help="Number of bins for each feature dimension in MAP-Elites"
        )
    
    with config_tabs[3]:  # Advanced
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.elite_ratio = st.slider(
                "Elite Ratio", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.get("elite_ratio", 0.1),
                step=0.01,
                help="Ratio of elite individuals preserved each generation"
            )
            st.session_state.exploration_ratio = st.slider(
                "Exploration Ratio", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.get("exploration_ratio", 0.3),
                step=0.01,
                help="Ratio of population dedicated to exploration"
            )
            st.session_state.enable_artifacts = st.checkbox(
                "Enable Artifact Feedback", 
                value=st.session_state.get("enable_artifacts", True),
                help="Enable error feedback to LLM for improved iterations"
            )
        
        with col2:
            st.session_state.cascade_evaluation = st.checkbox(
                "Cascade Evaluation", 
                value=st.session_state.get("cascade_evaluation", True),
                help="Use multi-stage testing to filter bad solutions early"
            )
            st.session_state.use_llm_feedback = st.checkbox(
                "Use LLM Feedback", 
                value=st.session_state.get("use_llm_feedback", False),
                help="Use LLM-based feedback for evolution guidance"
            )
            st.session_state.evolution_trace_enabled = st.checkbox(
                "Evolution Tracing", 
                value=st.session_state.get("evolution_trace_enabled", False),
                help="Enable detailed logging of evolution process"
            )


def render_performance_tab():
    """Render performance and monitoring features."""
    st.subheader("Performance & Monitoring")
    render_comprehensive_monitoring_ui()


def render_reports_tab():
    """Render evolution reports and analytics."""
    st.subheader("Evolution Reports & Analytics")
    render_reporting_dashboard()


def render_openevolve_control_panel():
    """Render a control panel for starting evolution runs."""
    st.subheader("Evolution Control Panel")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.session_state.evolution_content = st.text_area(
            "Content to Evolve", 
            value=st.session_state.get("evolution_content", "Enter content to evolve here..."),
            height=200,
            help="Content that will be evolved by OpenEvolve"
        )
    
    with col2:
        evolution_mode = st.selectbox(
            "Evolution Mode", 
            options=[
                "standard", "quality_diversity", "multi_objective", 
                "adversarial", "symbolic_regression", "neuroevolution", 
                "algorithm_discovery", "prompt_optimization"
            ],
            index=0,
            help="Choose the type of evolution to run"
        )
    
    with col3:
        start_evolution = st.button(
            "🚀 Start Evolution", 
            type="primary",
            use_container_width=True
        )
        stop_evolution = st.button(
            "⏹️ Stop Evolution", 
            type="secondary", 
            use_container_width=True
        )
    
    if start_evolution:
        # Validate inputs
        if not st.session_state.evolution_content.strip():
            st.error("Please enter content to evolve")
            return
        
        if not st.session_state.get("api_key"):
            st.error("Please configure your API key in the sidebar")
            return
        
        # Set up evolution parameters
        st.session_state.evolution_running = True
        st.session_state.evolution_mode = evolution_mode
        
        # Start evolution (would call the actual OpenEvolve function in a real implementation)
        with st.spinner(f"Starting {evolution_mode} evolution..."):
            # In a real implementation, this would call the OpenEvolve API
            st.success(f"{evolution_mode} evolution started successfully!")
    
    if stop_evolution:
        st.session_state.evolution_running = False
        st.info("Evolution stop signal sent")


def render_openevolve_workflow():
    """Render the main OpenEvolve workflow."""
    st.title("OpenEvolve: Advanced Evolution Platform")
    
    # Add a sidebar to the dashboard for API configuration
    with st.sidebar:
        st.header("OpenEvolve Configuration")
        
        # API Configuration
        st.session_state.openevolve_api_key = st.text_input(
            "API Key", 
            value=st.session_state.get("openevolve_api_key", ""),
            type="password",
            help="Your OpenAI-compatible API key"
        )
        
        st.session_state.openevolve_api_base = st.text_input(
            "API Base URL", 
            value=st.session_state.get("openevolve_api_base", "https://api.openai.com/v1"),
            help="Base URL for the API endpoint"
        )
        
        st.session_state.openevolve_model = st.selectbox(
            "Model", 
            options=[
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
                "claude-3-sonnet", "claude-3-haiku", 
                "gemini-1.5-pro", "gemini-1.5-flash"
            ],
            index=0,
            help="Model to use for evolution"
        )
    
    # Main dashboard content
    render_openevolve_dashboard()
    
    # Add the control panel at the bottom
    render_openevolve_control_panel()


# Initialize session state if not already done
def initialize_openevolve_session():
    """Initialize OpenEvolve session state variables."""
    default_values = {
        "evolution_running": False,
        "max_iterations": 100,
        "population_size": 100,
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 0.95,
        "seed": 42,
        "num_islands": 3,
        "migration_interval": 25,
        "migration_rate": 0.1,
        "feature_dimensions": ["complexity", "diversity"],
        "feature_bins": 10,
        "elite_ratio": 0.1,
        "exploration_ratio": 0.3,
        "enable_artifacts": True,
        "cascade_evaluation": True,
        "use_llm_feedback": False,
        "evolution_trace_enabled": False,
        "evolution_mode": "standard",
        "evolution_content": "Enter content to evolve here...",
        "openevolve_api_key": "",
        "openevolve_api_base": "https://api.openai.com/v1",
        "openevolve_model": "gpt-4o"
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Main execution
if __name__ == "__main__":
    initialize_openevolve_session()
    render_openevolve_workflow()