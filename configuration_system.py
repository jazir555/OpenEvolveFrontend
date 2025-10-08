"""
OpenEvolve Configuration Management System
Handles all configuration parameters for OpenEvolve features
"""
import streamlit as st
from typing import Dict, Any, Optional
import json
import os


class OpenEvolveConfigManager:
    """Manages all OpenEvolve configuration parameters"""
    
    def __init__(self):
        self.config_presets = {
            "default": self.get_default_config(),
            "research": self.get_research_config(),
            "production": self.get_production_config(),
            "experimental": self.get_experimental_config(),
        }
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default OpenEvolve configuration"""
        return {
            # Core parameters
            "max_iterations": 100,
            "population_size": 1000,
            "num_islands": 5,
            "migration_interval": 50,
            "migration_rate": 0.1,
            "archive_size": 100,
            "elite_ratio": 0.1,
            "exploration_ratio": 0.2,
            "exploitation_ratio": 0.7,
            "checkpoint_interval": 100,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 4096,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            
            # Feature dimensions for quality-diversity
            "feature_dimensions": ["complexity", "diversity"],
            "feature_bins": 10,
            "diversity_metric": "edit_distance",
            
            # Advanced evaluation parameters
            "enable_artifacts": True,
            "cascade_evaluation": True,
            "cascade_thresholds": [0.5, 0.75, 0.9],
            "use_llm_feedback": False,
            "llm_feedback_weight": 0.1,
            "parallel_evaluations": 4,
            "distributed": False,
            "template_dir": None,
            "num_top_programs": 3,
            "num_diverse_programs": 2,
            "use_template_stochasticity": True,
            "template_variations": {},
            "use_meta_prompting": False,
            "meta_prompt_weight": 0.1,
            "include_artifacts": True,
            "max_artifact_bytes": 20 * 1024,
            "artifact_security_filter": True,
            "early_stopping_patience": None,
            "convergence_threshold": 0.001,
            "early_stopping_metric": "combined_score",
            
            # Performance and resource management
            "memory_limit_mb": None,
            "cpu_limit": None,
            "random_seed": 42,
            "db_path": None,
            "in_memory": True,
            
            # Advanced OpenEvolve parameters
            "diff_based_evolution": True,
            "max_code_length": 10000,
            "evolution_trace_enabled": False,
            "evolution_trace_format": "jsonl",
            "evolution_trace_include_code": False,
            "evolution_trace_include_prompts": True,
            "evolution_trace_output_path": None,
            "evolution_trace_buffer_size": 10,
            "evolution_trace_compress": False,
            "log_level": "INFO",
            "log_dir": None,
            "api_timeout": 60,
            "api_retries": 3,
            "api_retry_delay": 5,
            "artifact_size_threshold": 32 * 1024,
            "cleanup_old_artifacts": True,
            "artifact_retention_days": 30,
            "diversity_reference_size": 20,
            "max_retries_eval": 3,
            "evaluator_timeout": 300,
            
            # Advanced research features
            "double_selection": True,
            "adaptive_feature_dimensions": True,
            "test_time_compute": False,
            "optillm_integration": False,
            "plugin_system": False,
            "hardware_optimization": False,
            "multi_strategy_sampling": True,
            "ring_topology": True,
            "controlled_gene_flow": True,
            "auto_diff": True,
            "symbolic_execution": False,
            "coevolutionary_approach": False,
        }
    
    def get_research_config(self) -> Dict[str, Any]:
        """Get research-focused configuration with advanced features enabled"""
        config = self.get_default_config()
        # Enhance for research purposes
        config.update({
            "num_islands": 7,
            "population_size": 2000,
            "archive_size": 200,
            "enable_artifacts": True,
            "cascade_evaluation": True,
            "use_llm_feedback": True,
            "llm_feedback_weight": 0.15,
            "parallel_evaluations": os.cpu_count() or 4,
            "evolution_trace_enabled": True,
            "evolution_trace_include_code": True,
            "early_stopping_patience": 20,
            "convergence_threshold": 0.0001,
            "double_selection": True,
            "adaptive_feature_dimensions": True,
            "multi_strategy_sampling": True,
            "test_time_compute": True,
        })
        return config
    
    def get_production_config(self) -> Dict[str, Any]:
        """Get production-optimized configuration"""
        config = self.get_default_config()
        # Optimize for production use
        config.update({
            "num_islands": 3,
            "population_size": 500,
            "archive_size": 50,
            "enable_artifacts": True,
            "cascade_evaluation": True,
            "parallel_evaluations": 2,
            "use_llm_feedback": False,
            "memory_limit_mb": 2048,
            "cpu_limit": 2.0,
            "evolution_trace_enabled": False,
            "early_stopping_patience": 10,
            "convergence_threshold": 0.001,
            "api_timeout": 30,
            "api_retries": 2,
        })
        return config
    
    def get_experimental_config(self) -> Dict[str, Any]:
        """Get experimental configuration with cutting-edge features"""
        config = self.get_default_config()
        # Enable experimental features
        config.update({
            "num_islands": 10,
            "population_size": 3000,
            "archive_size": 300,
            "enable_artifacts": True,
            "cascade_evaluation": True,
            "use_llm_feedback": True,
            "llm_feedback_weight": 0.2,
            "parallel_evaluations": os.cpu_count() or 4,
            "evolution_trace_enabled": True,
            "evolution_trace_include_prompts": True,
            "evolution_trace_include_code": True,
            "early_stopping_patience": 30,
            "convergence_threshold": 0.00001,
            "double_selection": True,
            "adaptive_feature_dimensions": True,
            "test_time_compute": True,
            "optillm_integration": True,
            "plugin_system": True,
            "hardware_optimization": True,
            "multi_strategy_sampling": True,
            "symbolic_execution": True,
            "coevolutionary_approach": True,
        })
        return config

    def apply_config_to_session(self, config_name: str = "default", custom_params: Optional[Dict[str, Any]] = None):
        """Apply configuration to Streamlit session state"""
        if config_name not in self.config_presets:
            config_name = "default"
        
        config = self.config_presets[config_name].copy()
        
        # Override with custom parameters if provided
        if custom_params:
            config.update(custom_params)
        
        # Apply to session state
        for key, value in config.items():
            st.session_state[key] = value
        
        return config
    
    def save_config(self, config: Dict[str, Any], filepath: str):
        """Save configuration to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def load_config(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load configuration from file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return None


def render_config_ui():
    """Render configuration UI in Streamlit"""
    st.header("⚙️ OpenEvolve Configuration System")
    
    # Initialize config manager
    if "config_manager" not in st.session_state:
        st.session_state.config_manager = OpenEvolveConfigManager()
    
    config_manager = st.session_state.config_manager
    
    # Configuration preset selector
    preset = st.selectbox(
        "Select Configuration Preset",
        options=["default", "research", "production", "experimental"],
        format_func=lambda x: {
            "default": "Default (Balanced)",
            "research": "Research (Advanced Features)",
            "production": "Production (Optimized)",
            "experimental": "Experimental (Cutting-Edge)"
        }.get(x, x)
    )
    
    # Display current configuration summary
    current_config = config_manager.config_presets[preset]
    with st.expander("Current Configuration Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Core Settings:**")
            st.write(f"- Max Iterations: {current_config['max_iterations']}")
            st.write(f"- Population Size: {current_config['population_size']}")
            st.write(f"- Number of Islands: {current_config['num_islands']}")
            st.write(f"- Temperature: {current_config['temperature']}")
        with col2:
            st.write("**Advanced Features:**")
            st.write(f"- Artifacts Enabled: {current_config['enable_artifacts']}")
            st.write(f"- Cascade Evaluation: {current_config['cascade_evaluation']}")
            st.write(f"- LLM Feedback: {current_config['use_llm_feedback']}")
            st.write(f"- Evolution Tracing: {current_config['evolution_trace_enabled']}")
    
    # Advanced configuration options
    with st.expander("Advanced Configuration", expanded=False):
        st.subheader("Population & Evolution Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_iterations = st.number_input(
                "Max Iterations", 
                min_value=1, 
                max_value=10000, 
                value=current_config['max_iterations']
            )
            population_size = st.number_input(
                "Population Size", 
                min_value=10, 
                max_value=10000, 
                value=current_config['population_size']
            )
            num_islands = st.number_input(
                "Number of Islands", 
                min_value=1, 
                max_value=20, 
                value=current_config['num_islands']
            )
        
        with col2:
            archive_size = st.number_input(
                "Archive Size", 
                min_value=10, 
                max_value=5000, 
                value=current_config['archive_size']
            )
            migration_interval = st.number_input(
                "Migration Interval", 
                min_value=1, 
                max_value=500, 
                value=current_config['migration_interval']
            )
            migration_rate = st.slider(
                "Migration Rate", 
                min_value=0.0, 
                max_value=1.0, 
                value=current_config['migration_rate'],
                step=0.01
            )
        
        with col3:
            elite_ratio = st.slider(
                "Elite Ratio", 
                min_value=0.0, 
                max_value=1.0, 
                value=current_config['elite_ratio'],
                step=0.01
            )
            exploration_ratio = st.slider(
                "Exploration Ratio", 
                min_value=0.0, 
                max_value=1.0, 
                value=current_config['exploration_ratio'],
                step=0.01
            )
            exploitation_ratio = st.slider(
                "Exploitation Ratio", 
                min_value=0.0, 
                max_value=1.0, 
                value=current_config['exploitation_ratio'],
                step=0.01
            )
        
        st.subheader("Generation Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=2.0, 
                value=current_config['temperature'],
                step=0.1
            )
            max_tokens = st.number_input(
                "Max Tokens", 
                min_value=100, 
                max_value=32768, 
                value=current_config['max_tokens']
            )
        
        with col2:
            top_p = st.slider(
                "Top-p", 
                min_value=0.0, 
                max_value=1.0, 
                value=current_config['top_p'],
                step=0.01
            )
            frequency_penalty = st.slider(
                "Frequency Penalty", 
                min_value=-2.0, 
                max_value=2.0, 
                value=current_config['frequency_penalty'],
                step=0.1
            )
            presence_penalty = st.slider(
                "Presence Penalty", 
                min_value=-2.0, 
                max_value=2.0, 
                value=current_config['presence_penalty'],
                step=0.1
            )
        
        st.subheader("Advanced Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enable_artifacts = st.checkbox(
                "Enable Artifacts", 
                value=current_config['enable_artifacts']
            )
            cascade_evaluation = st.checkbox(
                "Cascade Evaluation", 
                value=current_config['cascade_evaluation']
            )
            use_llm_feedback = st.checkbox(
                "Use LLM Feedback", 
                value=current_config['use_llm_feedback']
            )
            evolution_trace_enabled = st.checkbox(
                "Evolution Tracing", 
                value=current_config['evolution_trace_enabled']
            )
        
        with col2:
            double_selection = st.checkbox(
                "Double Selection", 
                value=current_config['double_selection']
            )
            adaptive_feature_dimensions = st.checkbox(
                "Adaptive Feature Dimensions", 
                value=current_config['adaptive_feature_dimensions']
            )
            multi_strategy_sampling = st.checkbox(
                "Multi-Strategy Sampling", 
                value=current_config['multi_strategy_sampling']
            )
            test_time_compute = st.checkbox(
                "Test-Time Compute", 
                value=current_config['test_time_compute']
            )
        
        with col3:
            optillm_integration = st.checkbox(
                "OptiLLM Integration", 
                value=current_config['optillm_integration']
            )
            plugin_system = st.checkbox(
                "Plugin System", 
                value=current_config['plugin_system']
            )
            hardware_optimization = st.checkbox(
                "Hardware Optimization", 
                value=current_config['hardware_optimization']
            )
            symbolic_execution = st.checkbox(
                "Symbolic Execution", 
                value=current_config['symbolic_execution']
            )
    
    # Apply configuration button
    if st.button("Apply Configuration", type="primary"):
        custom_params = {
            "max_iterations": max_iterations,
            "population_size": population_size,
            "num_islands": num_islands,
            "migration_interval": migration_interval,
            "migration_rate": migration_rate,
            "archive_size": archive_size,
            "elite_ratio": elite_ratio,
            "exploration_ratio": exploration_ratio,
            "exploitation_ratio": exploitation_ratio,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "enable_artifacts": enable_artifacts,
            "cascade_evaluation": cascade_evaluation,
            "use_llm_feedback": use_llm_feedback,
            "evolution_trace_enabled": evolution_trace_enabled,
            "double_selection": double_selection,
            "adaptive_feature_dimensions": adaptive_feature_dimensions,
            "multi_strategy_sampling": multi_strategy_sampling,
            "test_time_compute": test_time_compute,
            "optillm_integration": optillm_integration,
            "plugin_system": plugin_system,
            "hardware_optimization": hardware_optimization,
            "symbolic_execution": symbolic_execution,
        }
        
        applied_config = config_manager.apply_config_to_session(preset, custom_params)
        st.success(f"Configuration '{preset}' applied with custom parameters!")
        
        # Show what was updated
        with st.expander("Updated Configuration Parameters"):
            for key, value in applied_config.items():
                st.write(f"**{key}**: {value}")
    
    # Configuration export/import
    st.subheader("Configuration Import/Export")
    col1, col2 = st.columns(2)
    
    with col1:
        # Export current config
        if st.button("Export Current Configuration"):
            current_session_config = {k: v for k, v in st.session_state.items() 
                                    if k in current_config}
            st.download_button(
                label="Download Configuration",
                data=json.dumps(current_session_config, indent=2),
                file_name="openevolve_config.json",
                mime="application/json"
            )
    
    with col2:
        # Import config
        uploaded_file = st.file_uploader("Import Configuration", type="json")
        if uploaded_file is not None:
            try:
                imported_config = json.load(uploaded_file)
                config_manager.apply_config_to_session(custom_params=imported_config)
                st.success("Configuration imported successfully!")
            except Exception as e:
                st.error(f"Error importing configuration: {e}")


# Initialize configuration manager in session state
def initialize_config():
    if "config_manager" not in st.session_state:
        st.session_state.config_manager = OpenEvolveConfigManager()
        # Apply default configuration
        st.session_state.config_manager.apply_config_to_session()


# Run initialization if this file is executed directly
if __name__ == "__main__":
    initialize_config()
    st.write("OpenEvolve Configuration System initialized.")