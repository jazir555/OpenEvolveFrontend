"""
BubbleLabs Workflow Visualization Component

This module provides a Streamlit UI component that integrates with BubbleLabs
to visualize, interact with, and control OpenEvolve workflows.
"""

import os
import streamlit as st
import json
import html
from typing import Dict, Any, List, Optional
from uuid import uuid4
import threading
import time
import pandas as pd

from workflow_structures import Team, GauntletDefinition, WorkflowState
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from workflow_engine import run_sovereign_workflow
from api_server import team_manager, gauntlet_manager
# from session_defaults import get_default_session_values  # Import managers only # COMMENTED: Function doesn't exist
from openevolve_bubblelabs_api import openevolve_bubblelabs_integration
from parameter_sync_manager import parameter_sync_manager
from parameter_definitions import DEFAULT_PARAMETER_DEFINITIONS


# =============================================================================
# SECURITY FUNCTIONS
# =============================================================================

def escape_html(text: str) -> str:
    """
    Escape HTML special characters to prevent XSS attacks.

    Args:
        text: Text to escape

    Returns:
        HTML-escaped text
    """
    if not text:
        return ""
    return html.escape(text, quote=True)


def sanitize_user_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input to prevent XSS and injection attacks.

    Args:
        text: User input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Truncate to max length
    text = text[:max_length]

    # Escape HTML
    text = escape_html(text)

    # Remove null bytes
    text = text.replace("\x00", "")

    return text


def escape_json_for_js(data: Any) -> str:
    """
    Safely encode JSON data for insertion into JavaScript to prevent XSS.

    Args:
        data: Python data structure to encode

    Returns:
        JSON string safe for JavaScript
    """
    # Use json.dumps with ensure_ascii=True for safety
    json_str = json.dumps(data, ensure_ascii=True)

    # Escape special JavaScript characters
    json_str = json_str.replace('\\', '\\\\')
    json_str = json_str.replace('</', '<\\/')

    return json_str


class BubbleLabsWorkflowUI:
    """
    Streamlit UI component for BubbleLabs workflow integration.
    """
    
    def __init__(self):
        # Initialize managers
        self.team_manager = TeamManager()
        self.gauntlet_manager = GauntletManager()
        self.integration = openevolve_bubblelabs_integration
        self.param_sync = parameter_sync_manager

    def _render_all_openevolve_parameters(self, prefix: str = "sov"):
        """
        Render ALL 272+ OpenEvolve parameters organized by category.

        Args:
            prefix: Prefix for session state keys to avoid conflicts
        """
        st.markdown("### 🔧 Complete OpenEvolve Parameter Configuration")

        # Create tabs for each parameter category
        category_tabs = st.tabs(list(DEFAULT_PARAMETER_DEFINITIONS.keys()))

        for tab, (category, params) in zip(category_tabs, DEFAULT_PARAMETER_DEFINITIONS.items()):
            with tab:
                st.markdown(f"#### {category.replace('_', ' ').title()}")

                # Count parameters in this category
                param_count = len(params)
                st.caption(f"{param_count} parameters")

                # Render each parameter
                for param_name, param_config in params.items():
                    self._render_single_parameter(param_name, param_config, prefix, category)

    def _render_single_parameter(self, param_name: str, param_config: Dict, prefix: str, category: str):
        """
        Render a single parameter based on its type.

        Args:
            param_name: Name of the parameter
            param_config: Configuration dictionary for the parameter
            prefix: Prefix for session state keys
            category: Category the parameter belongs to
        """
        session_key = f"{prefix}_{category}_{param_name}"
        param_type = param_config.get("type", "string")
        default_value = param_config.get("default")
        description = param_config.get("description", "")
        min_value = param_config.get("min_value")
        max_value = param_config.get("max_value")
        options = param_config.get("options", [])

        # Create a unique key for Streamlit
        key = f"{session_key}_input"

        # Label with description as help text
        label = param_name.replace("_", " ").title()
        if description:
            help_text = description
        else:
            help_text = None

        # Render based on type
        if param_type == "select":
            if options:
                st.session_state[session_key] = st.selectbox(
                    label,
                    options=options,
                    index=options.index(default_value) if default_value in options else 0,
                    key=key,
                    help=help_text
                )
            else:
                st.session_state[session_key] = st.text_input(
                    label,
                    value=str(default_value) if default_value is not None else "",
                    key=key,
                    help=help_text
                )

        elif param_type == "boolean":
            st.session_state[session_key] = st.checkbox(
                label,
                value=default_value if default_value is not None else False,
                key=key,
                help=help_text
            )

        elif param_type == "integer":
            st.session_state[session_key] = st.number_input(
                label,
                value=default_value if default_value is not None else 0,
                min_value=min_value,
                max_value=max_value,
                key=key,
                help=help_text
            )

        elif param_type == "float":
            st.session_state[session_key] = st.slider(
                label,
                min_value=min_value if min_value is not None else (default_value - 1.0 if default_value else 0.0),
                max_value=max_value if max_value is not None else (default_value + 1.0 if default_value else 1.0),
                value=default_value if default_value is not None else 0.0,
                step=0.01 if (min_value or 0) % 1 == 0 or (max_value or 1) % 1 == 0 else 0.1,
                key=key,
                help=help_text
            )

        elif param_type == "list":
            # For list parameters, show a text area for JSON input
            default_str = str(default_value) if default_value else "[]"
            user_input = st.text_area(
                label,
                value=default_str,
                key=key,
                help=help_text + " (Enter as JSON list, e.g., ['item1', 'item2'])",
                height=80
            )
            # Try to parse as JSON
            try:
                import json
                parsed = json.loads(user_input)
                if isinstance(parsed, list):
                    st.session_state[session_key] = parsed
                else:
                    st.session_state[session_key] = []
            except:
                st.session_state[session_key] = default_value if default_value else []

        elif param_type == "dict":
            # For dict parameters, show a text area for JSON input
            default_str = str(default_value) if default_value else "{}"
            user_input = st.text_area(
                label,
                value=default_str,
                key=key,
                help=help_text + " (Enter as JSON dict, e.g., {'key': 'value'})",
                height=80
            )
            # Try to parse as JSON
            try:
                import json
                parsed = json.loads(user_input)
                if isinstance(parsed, dict):
                    st.session_state[session_key] = parsed
                else:
                    st.session_state[session_key] = {}
            except:
                st.session_state[session_key] = default_value if default_value else {}

        else:  # string or unknown type
            st.session_state[session_key] = st.text_input(
                label,
                value=str(default_value) if default_value is not None else "",
                key=key,
                help=help_text
            )

    def _get_all_openevolve_parameters_from_session(self, prefix: str = "sov") -> Dict[str, Any]:
        """
        Collect all OpenEvolve parameter values from session state.

        Args:
            prefix: Prefix used when rendering parameters

        Returns:
            Dictionary of all parameters organized by category
        """
        all_params = {}

        for category, params in DEFAULT_PARAMETER_DEFINITIONS.items():
            category_params = {}
            for param_name in params.keys():
                session_key = f"{prefix}_{category}_{param_name}"
                if session_key in st.session_state:
                    category_params[param_name] = st.session_state[session_key]
            all_params[category] = category_params

        return all_params

    def render_workflow_visualizer(self):
        """
        Render the workflow visualization component with comprehensive parameter controls.
        """
        st.header("🧬 OpenEvolve Workflows in BubbleLabs")
        st.markdown("""
        Visualize, interact with, and control OpenEvolve sovereign-grade decomposition workflows 
        through the BubbleLabs interface.
        """)
        
        # Create tabs for different functionality
        tabs = st.tabs(["Workflow Designer", "Active Workflows", "Workflow Control", "Global Parameters"])
        
        with tabs[0]:
            self._render_workflow_designer()
        
        with tabs[1]:
            self._render_active_workflows()
        
        with tabs[2]:
            self._render_workflow_control()
        
        with tabs[3]:
            self._render_global_parameters()
    
    def _render_global_parameters(self):
        """
        Render global OpenEvolve parameters that mirror the sidebar functionality.
        """
        st.subheader("🔧 Global Parameters")
        
        # Model configuration parameters (mirroring sidebar)
        st.markdown("### 🤖 Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            if "model" in st.session_state:
                st.session_state.model = st.selectbox(
                    "Model", 
                    options=["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet", "llama-2-70b", "llama-3-70b"],
                    index=["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet", "llama-2-70b", "llama-3-70b"].index(st.session_state.model) if st.session_state.model in ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet", "llama-2-70b", "llama-3-70b"] else 0,
                    key="bl_model"
                )
                
                st.session_state.temperature = st.slider(
                    "Temperature", 
                    min_value=0.0, 
                    max_value=2.0, 
                    value=st.session_state.get("temperature", 0.7),
                    step=0.01,
                    key="bl_temperature"
                )
                
                st.session_state.top_p = st.slider(
                    "Top P", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state.get("top_p", 1.0),
                    step=0.01,
                    key="bl_top_p"
                )
            
            # API configuration (SECURITY: Do not store API keys in session state)
            # Use environment variables or secure credential management instead
            api_key_value = os.getenv("OPENAI_API_KEY", "")

            st.markdown("**API Configuration**")
            st.info("API keys are read from environment variables for security.")
            st.caption("Set OPENAI_API_KEY environment variable to configure API access.")

            # Display masked API key if set via environment
            if api_key_value:
                st.text_input(
                    "API Key (from environment)",
                    value="*" * 20 + api_key_value[-4:],
                    type="password",
                    disabled=True,
                    key="bl_api_key_display"
                )
            else:
                st.text_input(
                    "API Key",
                    value="Not configured (set OPENAI_API_KEY environment variable)",
                    disabled=True,
                    key="bl_api_key_missing"
                )

            # Base URL (safe to store)
            if "base_url" in st.session_state:
                st.session_state.base_url = st.text_input(
                    "Base URL",
                    value=st.session_state.get("base_url", "https://api.openai.com/v1"),
                    key="bl_base_url"
                )
        
        with col2:
            if "max_tokens" in st.session_state:
                st.session_state.max_tokens = st.number_input(
                    "Max Tokens", 
                    min_value=1, 
                    max_value=32000, 
                    value=st.session_state.get("max_tokens", 4096),
                    key="bl_max_tokens"
                )
                
            if "frequency_penalty" in st.session_state:
                st.session_state.frequency_penalty = st.slider(
                    "Frequency Penalty", 
                    min_value=-2.0, 
                    max_value=2.0, 
                    value=st.session_state.get("frequency_penalty", 0.0),
                    step=0.01,
                    key="bl_frequency_penalty"
                )
                
            if "presence_penalty" in st.session_state:
                st.session_state.presence_penalty = st.slider(
                    "Presence Penalty", 
                    min_value=-2.0, 
                    max_value=2.0, 
                    value=st.session_state.get("presence_penalty", 0.0),
                    step=0.01,
                    key="bl_presence_penalty"
                )
            
            if "seed" in st.session_state:
                st.session_state.seed = st.number_input(
                    "Seed", 
                    value=st.session_state.get("seed", 42),
                    key="bl_seed"
                )
        
        # Evolution parameters
        st.markdown("### 🧬 Evolution Parameters")
        col3, col4 = st.columns(2)
        
        with col3:
            if "max_iterations" in st.session_state:
                st.session_state.max_iterations = st.number_input(
                    "Max Iterations", 
                    min_value=1, 
                    value=st.session_state.get("max_iterations", 100),
                    key="bl_max_iterations"
                )
            
            if "population_size" in st.session_state:
                st.session_state.population_size = st.number_input(
                    "Population Size", 
                    min_value=1, 
                    value=st.session_state.get("population_size", 50),
                    key="bl_population_size"
                )
        
        with col4:
            if "num_islands" in st.session_state:
                st.session_state.num_islands = st.number_input(
                    "Number of Islands", 
                    min_value=1, 
                    value=st.session_state.get("num_islands", 5),
                    key="bl_num_islands"
                )
            
            if "migration_rate" in st.session_state:
                st.session_state.migration_rate = st.slider(
                    "Migration Rate", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state.get("migration_rate", 0.1),
                    step=0.01,
                    key="bl_migration_rate"
                )
        
        # Quality Diversity parameters
        st.markdown("### 🎯 Quality Diversity Parameters")
        col5, col6 = st.columns(2)
        
        with col5:
            if "feature_dimensions" in st.session_state:
                available_dimensions = ["complexity", "diversity", "length", "readability", "performance", "security"]
                current_dims = st.session_state.get("feature_dimensions", ["complexity", "diversity"])
                
                selected_dims = st.multiselect(
                    "Feature Dimensions",
                    options=available_dimensions,
                    default=current_dims if all(d in available_dimensions for d in current_dims) else ["complexity", "diversity"],
                    key="bl_feature_dimensions"
                )
                st.session_state.feature_dimensions = selected_dims
        
        with col6:
            if "feature_bins" in st.session_state:
                st.session_state.feature_bins = st.slider(
                    "Feature Bins", 
                    min_value=2, 
                    max_value=50, 
                    value=st.session_state.get("feature_bins", 10),
                    key="bl_feature_bins"
                )
            
            if "diversity_metric" in st.session_state:
                st.session_state.diversity_metric = st.selectbox(
                    "Diversity Metric", 
                    options=["edit_distance", "ast_similarity", "ngram_overlap", "semantic_distance"],
                    index=["edit_distance", "ast_similarity", "ngram_overlap", "semantic_distance"].index(st.session_state.get("diversity_metric", "edit_distance")) if st.session_state.get("diversity_metric", "edit_distance") in ["edit_distance", "ast_similarity", "ngram_overlap", "semantic_distance"] else 0,
                    key="bl_diversity_metric"
                )
        
        # Advanced parameters
        st.markdown("### ⚙️ Advanced Parameters")
        col7, col8 = st.columns(2)
        
        with col7:
            if "early_stopping_patience" in st.session_state:
                st.session_state.early_stopping_patience = st.number_input(
                    "Early Stopping Patience", 
                    min_value=0, 
                    value=st.session_state.get("early_stopping_patience", 10),
                    key="bl_early_stopping_patience"
                )
            
            if "memory_limit_mb" in st.session_state:
                st.session_state.memory_limit_mb = st.number_input(
                    "Memory Limit (MB)", 
                    min_value=0, 
                    value=st.session_state.get("memory_limit_mb", 2048),
                    key="bl_memory_limit_mb"
                )
        
        with col8:
            if "convergence_threshold" in st.session_state:
                st.session_state.convergence_threshold = st.slider(
                    "Convergence Threshold", 
                    min_value=0.0, 
                    max_value=0.1, 
                    value=st.session_state.get("convergence_threshold", 0.01),
                    step=0.0001,
                    format="%.4f",
                    key="bl_convergence_threshold"
                )
            
            if "cpu_limit" in st.session_state:
                st.session_state.cpu_limit = st.slider(
                    "CPU Limit", 
                    min_value=0.0, 
                    max_value=4.0, 
                    value=st.session_state.get("cpu_limit", 1.0),
                    step=0.1,
                    key="bl_cpu_limit"
                )
        
        # Apply changes button
        if st.button("🔄 Apply Global Parameters", key="apply_global_params"):
            st.success("Global parameters updated successfully!")
            st.rerun()
    
    def _sync_workflow_parameters(self, workflow_state):
        """
        Sync all global parameters to the workflow state to ensure BubbleLabs has 1:1 control.
        """
        # Copy all relevant parameters from session state to workflow state
        sync_fields = [
            # Evolution parameters
            "max_iterations", "population_size", "num_islands", "migration_rate",
            # Feature and quality parameters
            "feature_dimensions", "feature_bins", "diversity_metric",
            # Advanced parameters
            "early_stopping_patience", "convergence_threshold", "memory_limit_mb", "cpu_limit"
        ]
        
        for field in sync_fields:
            if field in st.session_state:
                setattr(workflow_state, field, st.session_state[field])
        
        # Also sync model parameters
        model_fields = [
            "temperature", "top_p", "max_tokens", "frequency_penalty", 
            "presence_penalty", "seed"
        ]
        
        for field in model_fields:
            if field in st.session_state:
                setattr(workflow_state, field, st.session_state[field])
    
    def _create_openevolve_workflow_definition(self, problem_statement: str, team_config: Dict[str, str], gauntlet_config: Dict[str, str]) -> Dict[str, Any]:
        """
        Create a BubbleLabs workflow definition from OpenEvolve parameters (local implementation).
        """
        import uuid
        
        workflow_id = str(uuid.uuid4())
        
        # Create nodes for the OpenEvolve workflow
        nodes = [
            {
                "id": "content_analysis",
                "type": "content_analyzer",
                "position": {"x": 0, "y": 0},
                "data": {
                    "label": "Content Analysis",
                    "team": team_config.get("content_analyzer_team", ""),
                    "description": "Analyze the problem statement and extract structured context"
                }
            },
            {
                "id": "decomposition",
                "type": "decomposer",
                "position": {"x": 300, "y": 0},
                "data": {
                    "label": "Problem Decomposition",
                    "team": team_config.get("planner_team", ""),
                    "description": "Break down the problem into sub-problems"
                }
            },
            {
                "id": "subproblem_solver",
                "type": "solver",
                "position": {"x": 600, "y": 0},
                "data": {
                    "label": "Sub-problem Solving",
                    "team": team_config.get("solver_team", ""),
                    "gauntlet": gauntlet_config.get("sub_problem_red_gauntlet", ""),
                    "description": "Solve each sub-problem with specified gauntlet validation"
                }
            },
            {
                "id": "final_verification",
                "type": "verifier",
                "position": {"x": 900, "y": 0},
                "data": {
                    "label": "Final Verification",
                    "team": team_config.get("assembler_team", ""),
                    "gauntlet": gauntlet_config.get("final_gold_gauntlet", ""),
                    "description": "Verify the final solution with gold team gauntlet"
                }
            }
        ]
        
        # Create edges connecting the nodes
        edges = [
            {
                "id": "edge_1",
                "source": "content_analysis",
                "target": "decomposition",
                "sourceHandle": "output",
                "targetHandle": "input"
            },
            {
                "id": "edge_2",
                "source": "decomposition",
                "target": "subproblem_solver",
                "sourceHandle": "output",
                "targetHandle": "input"
            },
            {
                "id": "edge_3",
                "source": "subproblem_solver",
                "target": "final_verification",
                "sourceHandle": "output",
                "targetHandle": "input"
            }
        ]
        
        definition = {
            "id": workflow_id,
            "name": f"OpenEvolve Workflow: {problem_statement[:30]}...",
            "description": f"OpenEvolve sovereign-grade decomposition for: {problem_statement}",
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "problem_statement": problem_statement,
                "team_config": team_config,
                "gauntlet_config": gauntlet_config,
                "created_at": time.time(),
                "workflow_type": "openevolve_sovereign_decomposition"
            }
        }
        
        return definition
    
    def _render_workflow_designer(self):
        """
        Render the workflow designer interface with workflow type selection.
        """
        st.subheader("Design New Workflow")

        # Workflow type selection
        workflow_types = {
            "custom": "Custom Workflow",
            "openevolve_sovereign": "OpenEvolve Sovereign Decomposition",
            "openevolve_evolution": "OpenEvolve Evolution",
            "openevolve_adversarial": "OpenEvolve Adversarial Testing"
        }

        selected_workflow_type = st.selectbox(
            "Workflow Type",
            options=list(workflow_types.keys()),
            format_func=lambda x: workflow_types[x],
            key="workflow_type_selector"
        )

        # Input fields for workflow creation (SECURE: Sanitize input)
        problem_statement = st.text_area(
            "Problem Statement",
            placeholder="Enter the problem you want to solve...",
            height=150,
            max_chars=10000  # Limit input length
        )

        # Sanitize the input to prevent stored XSS
        if problem_statement:
            problem_statement = sanitize_user_input(problem_statement)

        # Render workflow-specific configuration
        if selected_workflow_type == "openevolve_sovereign":
            self._render_sovereign_workflow_config()
        elif selected_workflow_type == "openevolve_evolution":
            self._render_evolution_workflow_config()
        elif selected_workflow_type == "openevolve_adversarial":
            self._render_adversarial_workflow_config()
        else:
            self._render_custom_workflow_config()

        # Create workflow button
        if st.button("Create Workflow in BubbleLabs", type="primary", key="create_workflow_btn"):
            if not problem_statement.strip():
                st.error("Please enter a problem statement")
                return

            # Get workflow config based on type
            workflow_config = self._get_workflow_config_from_state(selected_workflow_type)

            # Create the workflow definition based on type
            if selected_workflow_type == "openevolve_sovereign":
                workflow_def = self._create_sovereign_workflow_definition(
                    problem_statement=problem_statement,
                    config=workflow_config
                )
            elif selected_workflow_type == "openevolve_evolution":
                workflow_def = self._create_evolution_workflow_definition(
                    problem_statement=problem_statement,
                    config=workflow_config
                )
            elif selected_workflow_type == "openevolve_adversarial":
                workflow_def = self._create_adversarial_workflow_definition(
                    problem_statement=problem_statement,
                    config=workflow_config
                )
            else:
                workflow_def = self._create_openevolve_workflow_definition(
                    problem_statement=problem_statement,
                    team_config=workflow_config.get("teams", {}),
                    gauntlet_config=workflow_config.get("gauntlets", {})
                )

            st.success(f"Workflow created successfully! ID: {workflow_def['id']}")

            # Store in session state
            if "created_workflow_defs" not in st.session_state:
                st.session_state.created_workflow_defs = []
            st.session_state.created_workflow_defs.append(workflow_def)

            # Show workflow visualization
            with st.expander("View Workflow Structure", expanded=True):
                self._display_workflow_graph(workflow_def)

            # Option to execute
            if st.button("Create and Execute Workflow Instance", key="execute_instance_btn"):
                self._create_and_execute_instance_local(workflow_def['id'], {
                    "problem_statement": problem_statement,
                    "workflow_type": selected_workflow_type,
                    **workflow_config
                })

    def _render_sovereign_workflow_config(self):
        """
        Render configuration for OpenEvolve Sovereign Decomposition workflow.
        This shows ALL 272+ configurable parameters for tweaking.
        """
        st.markdown("---")
        st.markdown("### 🔬 Sovereign Decomposition Configuration")

        # Get available teams and gauntlets
        all_teams = self.team_manager.get_all_teams()
        team_options = {team.name: team.name for team in all_teams} if all_teams else {"Default": "Default"}

        all_gauntlets = self.gauntlet_manager.get_all_gauntlets()
        gauntlet_options = {g.name: g.name for g in all_gauntlets} if all_gauntlets else {"None": "None"}

        # Use tabs to organize parameters - Teams/Gauntlets + ALL OpenEvolve parameters
        config_tabs = st.tabs([
            "Teams & Gauntlets",
            "All 272 Parameters"
        ])

        with config_tabs[0]:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 👥 Team Selection")
                st.session_state["sovereign_ca_team"] = st.selectbox(
                    "Content Analyzer Team",
                    options=list(team_options.keys()),
                    key="sov_ca_team",
                    help="Team that analyzes the problem statement"
                )
                st.session_state["sovereign_planner_team"] = st.selectbox(
                    "Planner Team",
                    options=list(team_options.keys()),
                    key="sov_planner_team",
                    help="Team that decomposes the problem into sub-problems"
                )
                st.session_state["sovereign_solver_team"] = st.selectbox(
                    "Solver Team",
                    options=list(team_options.keys()),
                    key="sov_solver_team",
                    help="Team that solves each sub-problem"
                )

            with col2:
                st.markdown("#### ⚔️ Gauntlet Selection")
                st.session_state["sovereign_sub_red"] = st.selectbox(
                    "Sub-problem Red Gauntlet",
                    options=["None"] + list(gauntlet_options.keys()),
                    key="sov_sub_red",
                    help="Red team for attacking sub-problem solutions"
                )
                st.session_state["sovereign_sub_gold"] = st.selectbox(
                    "Sub-problem Gold Gauntlet",
                    options=["None"] + list(gauntlet_options.keys()),
                    key="sov_sub_gold",
                    help="Gold team for verifying sub-problem solutions"
                )
                st.session_state["sovereign_final_red"] = st.selectbox(
                    "Final Red Gauntlet",
                    options=["None"] + list(gauntlet_options.keys()),
                    key="sov_final_red",
                    help="Red team for attacking final solution"
                )
                st.session_state["sovereign_final_gold"] = st.selectbox(
                    "Final Gold Gauntlet",
                    options=["None"] + list(gauntlet_options.keys()),
                    key="sov_final_gold",
                    help="Gold team for verifying final solution"
                )

            # Additional team selections
            col3, col4 = st.columns(2)
            with col3:
                st.session_state["sovereign_patcher_team"] = st.selectbox(
                    "Patcher Team",
                    options=list(team_options.keys()),
                    key="sov_patcher_team",
                    help="Team that refines solutions"
                )
            with col4:
                st.session_state["sovereign_assembler_team"] = st.selectbox(
                    "Assembler Team",
                    options=list(team_options.keys()),
                    key="sov_assembler_team",
                    help="Team that assembles final solution"
                )

        with config_tabs[1]:
            # Render ALL 272+ OpenEvolve parameters
            st.info("🔧 Below are ALL 272+ OpenEvolve parameters organized by category. Adjust any parameter to fine-tune your workflow.")
            self._render_all_openevolve_parameters(prefix="sov")

    def _render_evolution_workflow_config(self):
        """
        Render configuration for Evolution workflow.
        Adapted from mainlayout.py to BubbleLabs workflow model.

        NOTE: This is CONFIGURATION ONLY - no execution buttons.
        Execution happens through BubbleLabs workflow instance creation.
        """
        st.markdown("---")
        st.markdown("### 🧬 Evolution Configuration")

        # Get available teams and gauntlets
        all_teams = self.team_manager.get_all_teams()
        team_options = {team.name: team.name for team in all_teams} if all_teams else {"Default": "Default"}

        # Use tabs to organize parameters (BubbleLabs model: configuration, not execution)
        config_tabs = st.tabs([
            "Evolution Mode & Core Settings",
            "Mode-Specific Settings",
            "Advanced Features & Prompts",
            "All 272 Parameters"
        ])

        with config_tabs[0]:
            # Evolution Mode Selection (from mainlayout.py)
            st.markdown("#### 🧬 Evolution Mode")
            evolution_modes = [
                "standard", "quality_diversity", "multi_objective", "adversarial",
                "prompt_optimization", "algorithm_discovery", "symbolic_regression", "neuroevolution"
            ]
            evolution_mode_descriptions = {
                "standard": "Basic evolutionary optimization",
                "quality_diversity": "Quality-Diversity (MAP-Elites) evolution",
                "multi_objective": "Multi-objective optimization",
                "adversarial": "Red Team/Blue Team adversarial evolution",
                "prompt_optimization": "Optimize LLM prompts",
                "algorithm_discovery": "Discover novel algorithms",
                "symbolic_regression": "Discover mathematical expressions",
                "neuroevolution": "Evolve neural networks"
            }

            st.session_state["evo_mode"] = st.selectbox(
                "Select Evolution Mode",
                options=evolution_modes,
                format_func=lambda x: f"{x.replace('_', ' ').title()} - {evolution_mode_descriptions[x]}",
                key="evo_mode_select",
                help="Choose the evolutionary algorithm type"
            )

            # Team Selection (BubbleLabs: teams from TeamManager)
            st.markdown("#### 👥 Team Configuration")
            col1, col2 = st.columns(2)

            with col1:
                st.session_state["evo_ca_team"] = st.selectbox(
                    "Content Analyzer Team",
                    options=list(team_options.keys()),
                    key="evo_ca_team",
                    help="Team that analyzes the problem statement"
                )
                st.session_state["evo_planner_team"] = st.selectbox(
                    "Planner Team",
                    options=list(team_options.keys()),
                    key="evo_planner_team",
                    help="Team that plans evolution"
                )
            with col2:
                st.session_state["evo_solver_team"] = st.selectbox(
                    "Solver Team",
                    options=list(team_options.keys()),
                    key="evo_solver_team",
                    help="Team that generates solutions"
                )

            # Core Evolution Settings (from mainlayout.py)
            st.markdown("#### ⚙️ Core Evolution Settings")
            col3, col4 = st.columns(2)

            with col3:
                st.session_state["evo_max_iterations"] = st.number_input(
                    "Max Iterations",
                    min_value=1,
                    max_value=10000,
                    value=100,
                    key="evo_max_iterations",
                    help="Maximum number of evolution iterations"
                )
                st.session_state["evo_population_size"] = st.number_input(
                    "Population Size",
                    min_value=10,
                    max_value=1000,
                    value=50,
                    key="evo_population_size",
                    help="Size of the evolution population"
                )
                st.session_state["evo_generations"] = st.number_input(
                    "Generations",
                    min_value=1,
                    max_value=1000,
                    value=100,
                    key="evo_generations",
                    help="Number of generations to evolve"
                )

            with col4:
                st.session_state["evo_mutation_rate"] = st.slider(
                    "Mutation Rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    key="evo_mutation_rate",
                    help="Probability of mutation"
                )
                st.session_state["evo_num_islands"] = st.number_input(
                    "Number of Islands",
                    min_value=1,
                    max_value=20,
                    value=1,
                    key="evo_num_islands",
                    help="Island model for better exploration (1 = no islands)"
                )
                st.session_state["evo_archive_size"] = st.number_input(
                    "Archive Size",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    key="evo_archive_size",
                    help="Size of the archive for storing best solutions"
                )

            # Evolution Strategy (from mainlayout.py)
            st.markdown("#### 📊 Evolution Strategy Ratios")
            col5, col6 = st.columns(2)

            with col5:
                st.session_state["evo_elite_ratio"] = st.slider(
                    "Elite Ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    key="evo_elite_ratio",
                    help="Ratio of elite individuals to preserve"
                )
                st.session_state["evo_exploration_ratio"] = st.slider(
                    "Exploration Ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.01,
                    key="evo_exploration_ratio",
                    help="Ratio for exploration in evolution"
                )

            with col6:
                st.session_state["evo_exploitation_ratio"] = st.slider(
                    "Exploitation Ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.01,
                    key="evo_exploitation_ratio",
                    help="Ratio for exploitation in evolution"
                )

            # Show warning if ratios don't sum to 1.0 (from mainlayout.py)
            total_ratio = (
                st.session_state.get("evo_elite_ratio", 0.1) +
                st.session_state.get("evo_exploration_ratio", 0.2) +
                st.session_state.get("evo_exploitation_ratio", 0.7)
            )
            if abs(total_ratio - 1.0) > 0.01:
                st.warning(f"⚠️ Ratios sum to {total_ratio:.2f}, ideally should sum to 1.0")

        with config_tabs[1]:
            # Mode-specific settings (from mainlayout.py)
            selected_mode = st.session_state.get("evo_mode", "standard")

            # Quality-Diversity and Multi-Objective Settings
            if selected_mode in ["quality_diversity", "multi_objective"]:
                st.markdown("#### 📐 Feature Dimensions (QD/Multi-Objective)")
                available_features = ["complexity", "diversity", "performance", "efficiency", "readability", "robustness", "accuracy", "clarity"]
                st.session_state["evo_feature_dimensions"] = st.multiselect(
                    "Feature Dimensions",
                    options=available_features,
                    default=["complexity", "diversity"],
                    help="Feature dimensions for Quality-Diversity or Multi-Objective evolution",
                    key="evo_feature_dimensions"
                )
                st.session_state["evo_feature_bins"] = st.number_input(
                    "Feature Bins",
                    min_value=5,
                    max_value=50,
                    value=10,
                    help="Number of bins for each feature dimension",
                    key="evo_feature_bins"
                )

            # Multi-Objective Specific Settings
            if selected_mode == "multi_objective":
                st.markdown("#### 🎯 Multi-Objective Optimization")
                available_objectives = ["performance", "readability", "maintainability", "efficiency", "security", "robustness", "clarity", "completeness"]
                st.session_state["evo_objectives"] = st.multiselect(
                    "Objectives to Optimize",
                    options=available_objectives,
                    default=["performance", "readability"],
                    help="Objectives for multi-objective optimization",
                    key="evo_objectives"
                )

            # Adversarial Evolution Settings (from mainlayout.py)
            if selected_mode == "adversarial":
                st.markdown("#### ⚔️ Adversarial Evolution Settings")
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state["evo_attack_model"] = st.selectbox(
                        "Attack Model",
                        options=["gpt-4o", "claude-3-sonnet", "gpt-4", "gemini-1.5-pro"],
                        key="evo_attack_model",
                        help="Model for adversarial attacks (Red Team)"
                    )
                with col2:
                    st.session_state["evo_defense_model"] = st.selectbox(
                        "Defense Model",
                        options=["gpt-4o", "claude-3-opus", "gpt-4", "gemini-1.5-pro"],
                        key="evo_defense_model",
                        help="Model for adversarial defense (Blue Team)"
                    )

        with config_tabs[2]:
            # Advanced OpenEvolve Features (from mainlayout.py)
            st.markdown("#### 🔧 Advanced OpenEvolve Features")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.session_state["evo_enable_artifacts"] = st.checkbox(
                    "Enable Artifacts",
                    value=True,
                    key="evo_enable_artifacts",
                    help="Enable artifact side-channel for additional context"
                )
                st.session_state["evo_cascade_evaluation"] = st.checkbox(
                    "Cascade Evaluation",
                    value=True,
                    key="evo_cascade_evaluation",
                    help="Use cascade evaluation for efficiency"
                )
                st.session_state["evo_use_llm_feedback"] = st.checkbox(
                    "LLM Feedback",
                    value=False,
                    key="evo_use_llm_feedback",
                    help="Use LLM-based feedback for guidance"
                )

            with col2:
                st.session_state["evo_include_artifacts"] = st.checkbox(
                    "Include Artifacts",
                    value=True,
                    key="evo_include_artifacts",
                    help="Include artifacts in prompts"
                )
                st.session_state["evo_evolution_trace_enabled"] = st.checkbox(
                    "Enable Trace",
                    value=False,
                    key="evo_evolution_trace_enabled",
                    help="Enable evolution trace logging"
                )
                st.session_state["evo_diff_based_evolution"] = st.checkbox(
                    "Diff-Based Evolution",
                    value=True,
                    key="evo_diff_based_evolution",
                    help="Use diff-based evolution for efficiency"
                )

            with col3:
                st.session_state["evo_parallel_evaluations"] = st.number_input(
                    "Parallel Evaluations",
                    min_value=1,
                    max_value=16,
                    value=4,
                    key="evo_parallel_evaluations",
                    help="Number of parallel evaluations"
                )
                st.session_state["evo_checkpoint_interval"] = st.number_input(
                    "Checkpoint Interval",
                    min_value=1,
                    max_value=100,
                    value=10,
                    key="evo_checkpoint_interval",
                    help="Interval for saving checkpoints"
                )

            # Prompts Configuration (from mainlayout.py)
            st.markdown("#### 💬 Prompts Configuration")
            st.info("Configure custom prompts for the evolution process. These will be used during workflow execution.")
            col4, col5 = st.columns(2)

            with col4:
                st.session_state["evo_system_prompt"] = st.text_area(
                    "System Prompt",
                    value="You are an expert content generator and optimizer.",
                    height=150,
                    key="evo_system_prompt",
                    help="System prompt for the evolution process"
                )

            with col5:
                st.session_state["evo_evaluator_system_prompt"] = st.text_area(
                    "Evaluator System Prompt",
                    value="Evaluate the quality of this content and provide a score from 0 to 100 based on accuracy, clarity, and completeness.",
                    height=150,
                    key="evo_evaluator_system_prompt",
                    help="System prompt for the evaluator"
                )

        with config_tabs[3]:
            # Render ALL 272+ OpenEvolve parameters
            st.info("🔧 Below are ALL 272+ OpenEvolve parameters organized by category. Adjust any parameter to fine-tune your evolution workflow.")
            self._render_all_openevolve_parameters(prefix="evo")

    def _render_adversarial_workflow_config(self):
        """
        Render configuration for Adversarial Testing workflow.
        Includes ALL 272+ OpenEvolve parameters plus adversarial-specific configuration from mainlayout.py.
        """
        st.markdown("---")
        st.markdown("### ⚔️ Adversarial Testing Configuration")

        # Get available teams and gauntlets
        all_teams = self.team_manager.get_all_teams()
        team_options = {team.name: team.name for team in all_teams} if all_teams else {"Default": "Default"}

        all_gauntlets = self.gauntlet_manager.get_all_gauntlets()
        gauntlet_options = {g.name: g.name for g in all_gauntlets} if all_gauntlets else {"None": "None"}

        # Use tabs to organize parameters
        config_tabs = st.tabs([
            "Red/Blue Teams & Models",
            "Process Parameters",
            "Quality Control",
            "All 272 Parameters"
        ])

        with config_tabs[0]:
            # Content Type Selection (from mainlayout.py)
            st.markdown("#### 📋 Content Configuration")
            content_types = [
                "document_general", "code_python", "code_javascript", "code_rust",
                "prompt_engineering", "security_assessment", "compliance_check",
                "performance_optimization", "bug_detection", "documentation"
            ]
            st.session_state["adv_content_type"] = st.selectbox(
                "Content Type",
                options=content_types,
                key="adv_content_type",
                help="Type of content for adversarial testing"
            )

            # Model Configuration (from mainlayout.py)
            st.markdown("#### 🤖 Model Configuration")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### 🔴 Red Team (Attackers)")
                st.session_state["adv_red_team"] = st.selectbox(
                    "Red Team",
                    options=list(team_options.keys()),
                    key="adv_red_team",
                    help="Team that generates adversarial attacks"
                )
                st.session_state["adv_red_team_sample_size"] = st.number_input(
                    "Red Team Sample Size",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="adv_red_team_sample_size",
                    help="Number of Red Team models to use per iteration"
                )

            with col2:
                st.markdown("##### 🔵 Blue Team (Defenders)")
                st.session_state["adv_blue_team"] = st.selectbox(
                    "Blue Team",
                    options=list(team_options.keys()),
                    key="adv_blue_team",
                    help="Team that defends against attacks"
                )
                st.session_state["adv_blue_team_sample_size"] = st.number_input(
                    "Blue Team Sample Size",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="adv_blue_team_sample_size",
                    help="Number of Blue Team models to use per iteration"
                )

            # Evaluator Team Configuration (from mainlayout.py)
            st.markdown("##### 🎯 Evaluator Team")
            col3, col4 = st.columns(2)
            with col3:
                st.session_state["adv_evaluator_team"] = st.selectbox(
                    "Evaluator Team",
                    options=list(team_options.keys()),
                    key="adv_evaluator_team",
                    help="Team that evaluates final results"
                )
            with col4:
                st.session_state["adv_evaluator_sample_size"] = st.number_input(
                    "Evaluator Sample Size",
                    min_value=1,
                    max_value=10,
                    value=2,
                    key="adv_evaluator_sample_size",
                    help="Number of Evaluator models to use"
                )

            # Rotation Strategy (from mainlayout.py)
            st.markdown("#### 🔄 Rotation & Selection Strategy")
            st.session_state["adv_rotation_strategy"] = st.selectbox(
                "Model Rotation Strategy",
                options=["round_robin", "random", "performance_based", "diversity_focused"],
                key="adv_rotation_strategy",
                help="""
                - round_robin: Rotate models in fixed order
                - random: Random model selection each iteration
                - performance_based: Select best performing models
                - diversity_focused: Maximize model diversity
                """
            )

            # Core Adversarial Settings
            st.markdown("#### ⚙️ Core Adversarial Settings")
            col5, col6 = st.columns(2)
            with col5:
                st.session_state["adv_attack_strength"] = st.slider(
                    "Attack Strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    key="adv_attack_strength",
                    help="Intensity of adversarial attacks"
                )
                st.session_state["adv_defense_strength"] = st.slider(
                    "Defense Strength",
                    min_value=0.1,
                    max_value=2.0,
                    value=1.2,
                    step=0.1,
                    key="adv_defense_strength",
                    help="Strength of defensive measures"
                )
            with col6:
                st.session_state["adv_defense_strategy"] = st.selectbox(
                    "Defense Strategy",
                    options=["reactive", "proactive", "adaptive"],
                    key="adv_defense_strategy",
                    help="""
                    - reactive: Respond to attacks
                    - proactive: Anticipate attacks
                    - adaptive: Learn and adapt
                    """
                )
                st.session_state["adv_adversarial_rounds"] = st.number_input(
                    "Adversarial Rounds",
                    min_value=1,
                    max_value=100,
                    value=10,
                    key="adv_adversarial_rounds",
                    help="Number of adversarial attack/defense rounds"
                )

            # Gauntlet selection for verification
            st.markdown("#### ⚔️ Verification Gauntlets")
            col7, col8 = st.columns(2)
            with col7:
                st.session_state["adv_red_gauntlet"] = st.selectbox(
                    "Attack Verification Gauntlet",
                    options=["None"] + list(gauntlet_options.keys()),
                    key="adv_red_gauntlet",
                    help="Gauntlet to verify attack quality"
                )
            with col8:
                st.session_state["adv_blue_gauntlet"] = st.selectbox(
                    "Defense Verification Gauntlet",
                    options=["None"] + list(gauntlet_options.keys()),
                    key="adv_blue_gauntlet",
                    help="Gauntlet to verify defense quality"
                )

        with config_tabs[1]:  # Process Parameters (from mainlayout.py)
            st.markdown("#### 📊 Iteration & Threshold Management")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.session_state["adv_min_iterations"] = st.number_input(
                    "Minimum Iterations",
                    min_value=1,
                    max_value=100,
                    value=5,
                    key="adv_min_iterations",
                    help="Minimum number of adversarial iterations"
                )
                st.session_state["adv_max_iterations"] = st.number_input(
                    "Maximum Iterations",
                    min_value=1,
                    max_value=200,
                    value=50,
                    key="adv_max_iterations",
                    help="Maximum number of adversarial iterations"
                )

            with col2:
                st.session_state["adv_confidence_threshold"] = st.slider(
                    "Confidence Threshold (%)",
                    min_value=50,
                    max_value=100,
                    value=90,
                    key="adv_confidence_threshold",
                    help="Minimum acceptance level for completion"
                )
                st.session_state["adv_evaluator_threshold"] = st.slider(
                    "Evaluator Threshold",
                    min_value=50.0,
                    max_value=100.0,
                    value=90.0,
                    step=0.5,
                    key="adv_evaluator_threshold",
                    help="Minimum score for evaluator acceptance"
                )

            with col3:
                st.session_state["adv_evaluator_consecutive_rounds"] = st.number_input(
                    "Consecutive Rounds Required",
                    min_value=1,
                    max_value=10,
                    value=1,
                    key="adv_evaluator_consecutive_rounds",
                    help="Consecutive successful evaluations"
                )
                st.session_state["adv_budget_limit"] = st.number_input(
                    "Budget Limit (USD)",
                    min_value=0.0,
                    value=50.0,
                    format="%.2f",
                    key="adv_budget_limit",
                    help="Maximum budget for testing"
                )

            # Quality Control Parameters (from mainlayout.py)
            st.markdown("#### 🔍 Quality Control Parameters")
            col4, col5 = st.columns(2)
            with col4:
                st.session_state["adv_critique_depth"] = st.slider(
                    "Critique Depth Level",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key="adv_critique_depth",
                    help="Thoroughness of Red Team analysis (1=surface, 10=deep)"
                )
            with col5:
                st.session_state["adv_patch_quality"] = st.slider(
                    "Patch Quality Level",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key="adv_patch_quality",
                    help="Thoroughness of Blue Team fixes (1=basic, 10=comprehensive)"
                )

        with config_tabs[2]:  # Quality Control (from mainlayout.py)
            st.markdown("#### 🛡️ Quality Assurance & Validation")

            col1, col2 = st.columns(2)
            with col1:
                st.session_state["adv_enable_human_feedback"] = st.checkbox(
                    "Enable Human Feedback Integration",
                    value=False,
                    key="adv_enable_human_feedback",
                    help="Allow human feedback during process"
                )
                st.session_state["adv_keyword_analysis_enabled"] = st.checkbox(
                    "Enable Keyword Analysis",
                    value=True,
                    key="adv_keyword_analysis_enabled",
                    help="Analyze content for keyword presence"
                )

                if st.session_state.get("adv_keyword_analysis_enabled", True):
                    st.session_state["adv_keywords_to_target"] = st.text_area(
                        "Keywords to Target",
                        value="",
                        height=80,
                        placeholder="Enter keywords separated by commas...",
                        key="adv_keywords_to_target",
                        help="Keywords to incorporate in content"
                    )

            with col2:
                st.session_state["adv_enable_real_time_monitoring"] = st.checkbox(
                    "Enable Real-Time Monitoring",
                    value=True,
                    key="adv_enable_real_time_monitoring",
                    help="Monitor performance in real-time"
                )
                st.session_state["adv_enable_comprehensive_reporting"] = st.checkbox(
                    "Enable Comprehensive Reporting",
                    value=True,
                    key="adv_enable_comprehensive_reporting",
                    help="Generate detailed reports"
                )

            # Security & Compliance (from mainlayout.py)
            st.markdown("#### 🔒 Security & Compliance")
            col3, col4 = st.columns(2)
            with col3:
                st.session_state["adv_enable_encryption"] = st.checkbox(
                    "Enable Data Encryption",
                    value=True,
                    key="adv_enable_encryption",
                    help="Encrypt sensitive data during processing"
                )
            with col4:
                st.session_state["adv_enable_audit_trail"] = st.checkbox(
                    "Enable Audit Trail",
                    value=True,
                    key="adv_enable_audit_trail",
                    help="Maintain detailed audit trail"
                )

            # Advanced Evolution & Optimization (from mainlayout.py)
            st.markdown("#### 🧬 Advanced Evolution & Optimization")
            col5, col6 = st.columns(2)
            with col5:
                st.session_state["adv_enable_multi_objective"] = st.checkbox(
                    "Enable Multi-Objective Optimization",
                    value=False,
                    key="adv_enable_multi_objective",
                    help="Optimize for multiple objectives"
                )

                if st.session_state.get("adv_enable_multi_objective", False):
                    feature_dims = ["complexity", "diversity", "performance", "efficiency", "readability", "robustness", "accuracy", "clarity"]
                    st.session_state["adv_feature_dimensions"] = st.multiselect(
                        "Feature Dimensions",
                        options=feature_dims,
                        default=["complexity", "diversity"],
                        key="adv_feature_dimensions",
                        help="Feature dimensions for QD evolution"
                    )
                    st.session_state["adv_feature_bins"] = st.number_input(
                        "Feature Bins",
                        min_value=5,
                        max_value=50,
                        value=10,
                        key="adv_feature_bins",
                        help="Number of bins per feature dimension"
                    )

            with col6:
                st.session_state["adv_enable_data_augmentation"] = st.checkbox(
                    "Enable Data Augmentation",
                    value=False,
                    key="adv_enable_data_augmentation",
                    help="Generate adversarial examples"
                )

                if st.session_state.get("adv_enable_data_augmentation", False):
                    st.session_state["adv_augmentation_model"] = st.selectbox(
                        "Augmentation Model",
                        options=["gpt-4o", "gpt-4", "claude-3-opus", "claude-3-sonnet"],
                        index=0,
                        key="adv_augmentation_model",
                        help="Model for generating adversarial examples"
                    )
                    st.session_state["adv_augmentation_temperature"] = st.slider(
                        "Augmentation Temperature",
                        min_value=0.0,
                        max_value=2.0,
                        value=0.7,
                        step=0.1,
                        key="adv_augmentation_temperature",
                        help="Temperature for augmentation (higher=more creative)"
                    )

            # Evolution Parameters (from mainlayout.py)
            st.markdown("#### 📈 Evolution Parameters")
            col7, col8, col9 = st.columns(3)
            with col7:
                st.session_state["adv_elite_ratio"] = st.slider(
                    "Elite Ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    key="adv_elite_ratio",
                    help="Ratio of elite individuals to preserve"
                )
            with col8:
                st.session_state["adv_exploration_ratio"] = st.slider(
                    "Exploration Ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.01,
                    key="adv_exploration_ratio",
                    help="Ratio for exploration"
                )
            with col9:
                st.session_state["adv_archive_size"] = st.number_input(
                    "Archive Size",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    key="adv_archive_size",
                    help="Size of archive for best solutions"
                )

            # Custom Prompts (from mainlayout.py)
            st.markdown("#### 💬 Custom Prompts")
            st.session_state["adv_custom_red_prompt"] = st.text_area(
                "Red Team Custom Prompt",
                value="",
                height=100,
                key="adv_custom_red_prompt",
                help="Custom prompt for Red Team attacks"
            )
            st.session_state["adv_custom_blue_prompt"] = st.text_area(
                "Blue Team Custom Prompt",
                value="",
                height=100,
                key="adv_custom_blue_prompt",
                help="Custom prompt for Blue Team defense"
            )
            st.session_state["adv_custom_approval_prompt"] = st.text_area(
                "Approval Prompt",
                value="",
                height=100,
                key="adv_custom_approval_prompt",
                help="Custom prompt for final approval assessment"
            )

        with config_tabs[3]:
            # Render ALL 272+ OpenEvolve parameters
            st.info("🔧 Below are ALL 272+ OpenEvolve parameters organized by category. Adjust any parameter to fine-tune your adversarial workflow.")
            self._render_all_openevolve_parameters(prefix="adv")

    def _render_custom_workflow_config(self):
        """Render configuration for custom workflow."""
        st.markdown("---")
        st.markdown("### 🔧 Custom Workflow Configuration")

        # Get available teams and gauntlets
        all_teams = self.team_manager.get_all_teams()
        team_options = {team.name: team.name for team in all_teams} if all_teams else {"Default": "Default"}

        all_gauntlets = self.gauntlet_manager.get_all_gauntlets()
        gauntlet_options = {g.name: g.name for g in all_gauntlets} if all_gauntlets else {"None": "None"}

        st.subheader("Team Configuration")
        col1, col2 = st.columns(2)

        with col1:
            st.session_state["custom_ca_team"] = st.selectbox("Content Analyzer Team", options=list(team_options.keys()), key="custom_ca_team")
            st.session_state["custom_planner_team"] = st.selectbox("Planner Team", options=list(team_options.keys()), key="custom_planner_team")
            st.session_state["custom_solver_team"] = st.selectbox("Solver Team", options=list(team_options.keys()), key="custom_solver_team")

        with col2:
            st.session_state["custom_patcher_team"] = st.selectbox("Patcher Team", options=list(team_options.keys()), key="custom_patcher_team")
            st.session_state["custom_assembler_team"] = st.selectbox("Assembler Team", options=list(team_options.keys()), key="custom_assembler_team")

        st.subheader("Gauntlet Configuration")
        col3, col4 = st.columns(2)

        with col3:
            st.session_state["custom_sub_red"] = st.selectbox("Sub-problem Red Gauntlet", options=list(gauntlet_options.keys()), key="custom_sub_red")
            st.session_state["custom_sub_gold"] = st.selectbox("Sub-problem Gold Gauntlet", options=list(gauntlet_options.keys()), key="custom_sub_gold")

        with col4:
            st.session_state["custom_final_red"] = st.selectbox("Final Red Gauntlet", options=list(gauntlet_options.keys()), key="custom_final_red")
            st.session_state["custom_final_gold"] = st.selectbox("Final Gold Gauntlet", options=list(gauntlet_options.keys()), key="custom_final_gold")

    def _get_workflow_config_from_state(self, workflow_type: str) -> Dict[str, Any]:
        """Get workflow configuration from session state based on workflow type."""
        config = {"teams": {}, "gauntlets": {}, "openevolve_parameters": {}}

        if workflow_type == "openevolve_sovereign":
            # Teams
            config["teams"] = {
                "content_analyzer_team": st.session_state.get("sovereign_ca_team", "Default"),
                "planner_team": st.session_state.get("sovereign_planner_team", "Default"),
                "solver_team": st.session_state.get("sovereign_solver_team", "Default"),
                "patcher_team": st.session_state.get("sovereign_patcher_team", "Default"),
                "assembler_team": st.session_state.get("sovereign_assembler_team", "Default")
            }
            # Gauntlets
            config["gauntlets"] = {
                "sub_problem_red_gauntlet": st.session_state.get("sovereign_sub_red", "None"),
                "sub_problem_gold_gauntlet": st.session_state.get("sovereign_sub_gold", "None"),
                "final_red_gauntlet": st.session_state.get("sovereign_final_red", "None"),
                "final_gold_gauntlet": st.session_state.get("sovereign_final_gold", "None")
            }
            # ALL OpenEvolve parameters (272+ parameters)
            config["openevolve_parameters"] = self._get_all_openevolve_parameters_from_session(prefix="sov")

        elif workflow_type == "openevolve_evolution":
            # Teams
            config["teams"] = {
                "content_analyzer_team": st.session_state.get("evo_ca_team", "Default"),
                "planner_team": st.session_state.get("evo_planner_team", "Default"),
                "solver_team": st.session_state.get("evo_solver_team", "Default"),
                "patcher_team": st.session_state.get("evo_ca_team", "Default"),  # Use CA as patcher
                "assembler_team": st.session_state.get("evo_solver_team", "Default")  # Use solver as assembler
            }
            # Evolution-specific settings (from mainlayout.py)
            config["evolution_settings"] = {
                # Evolution mode
                "evolution_mode": st.session_state.get("evo_mode", "standard"),
                # Core settings
                "max_iterations": st.session_state.get("evo_max_iterations", 100),
                "population_size": st.session_state.get("evo_population_size", 50),
                "mutation_rate": st.session_state.get("evo_mutation_rate", 0.1),
                "generations": st.session_state.get("evo_generations", 100),
                "num_islands": st.session_state.get("evo_num_islands", 1),
                "archive_size": st.session_state.get("evo_archive_size", 100),
                # Evolution strategy
                "elite_ratio": st.session_state.get("evo_elite_ratio", 0.1),
                "exploration_ratio": st.session_state.get("evo_exploration_ratio", 0.2),
                "exploitation_ratio": st.session_state.get("evo_exploitation_ratio", 0.7),
                # Mode-specific settings
                "feature_dimensions": st.session_state.get("evo_feature_dimensions", ["complexity", "diversity"]),
                "feature_bins": st.session_state.get("evo_feature_bins", 10),
                "objectives": st.session_state.get("evo_objectives", ["performance", "readability"]),
                "adversarial_attack_model": st.session_state.get("evo_attack_model", "gpt-4"),
                "adversarial_defense_model": st.session_state.get("evo_defense_model", "gpt-4"),
                # Advanced OpenEvolve features
                "enable_artifacts": st.session_state.get("evo_enable_artifacts", True),
                "cascade_evaluation": st.session_state.get("evo_cascade_evaluation", True),
                "use_llm_feedback": st.session_state.get("evo_use_llm_feedback", False),
                "include_artifacts": st.session_state.get("evo_include_artifacts", True),
                "evolution_trace_enabled": st.session_state.get("evo_evolution_trace_enabled", False),
                "diff_based_evolution": st.session_state.get("evo_diff_based_evolution", True),
                "parallel_evaluations": st.session_state.get("evo_parallel_evaluations", 4),
                "checkpoint_interval": st.session_state.get("evo_checkpoint_interval", 10),
                # Prompts
                "system_prompt": st.session_state.get("evo_system_prompt", ""),
                "evaluator_system_prompt": st.session_state.get("evo_evaluator_system_prompt", "")
            }
            # ALL OpenEvolve parameters (272+ parameters)
            config["openevolve_parameters"] = self._get_all_openevolve_parameters_from_session(prefix="evo")

        elif workflow_type == "openevolve_adversarial":
            # Teams (Red/Blue/Evaluator)
            config["teams"] = {
                "red_team": st.session_state.get("adv_red_team", "Default"),
                "blue_team": st.session_state.get("adv_blue_team", "Default"),
                "evaluator_team": st.session_state.get("adv_evaluator_team", "Default"),
                "content_analyzer_team": st.session_state.get("adv_red_team", "Default"),  # Use red as CA
                "planner_team": st.session_state.get("adv_blue_team", "Default"),  # Use blue as planner
                "solver_team": st.session_state.get("adv_red_team", "Default"),  # Use red as solver
                "patcher_team": st.session_state.get("adv_blue_team", "Default"),  # Use blue as patcher
                "assembler_team": st.session_state.get("adv_blue_team", "Default")  # Use blue as assembler
            }
            # Gauntlets
            config["gauntlets"] = {
                "red_gauntlet": st.session_state.get("adv_red_gauntlet", "None"),
                "blue_gauntlet": st.session_state.get("adv_blue_gauntlet", "None"),
                "sub_problem_red_gauntlet": st.session_state.get("adv_red_gauntlet", "None"),
                "sub_problem_gold_gauntlet": st.session_state.get("adv_blue_gauntlet", "None"),
                "final_red_gauntlet": st.session_state.get("adv_red_gauntlet", "None"),
                "final_gold_gauntlet": st.session_state.get("adv_blue_gauntlet", "None")
            }
            # Adversarial-specific settings (from mainlayout.py)
            config["adversarial_settings"] = {
                # Content configuration
                "content_type": st.session_state.get("adv_content_type", "document_general"),
                # Model configuration
                "red_team_sample_size": st.session_state.get("adv_red_team_sample_size", 3),
                "blue_team_sample_size": st.session_state.get("adv_blue_team_sample_size", 3),
                "evaluator_sample_size": st.session_state.get("adv_evaluator_sample_size", 2),
                "rotation_strategy": st.session_state.get("adv_rotation_strategy", "round_robin"),
                # Core adversarial settings
                "attack_strength": st.session_state.get("adv_attack_strength", 0.7),
                "defense_strength": st.session_state.get("adv_defense_strength", 1.2),
                "adversarial_rounds": st.session_state.get("adv_adversarial_rounds", 10),
                "defense_strategy": st.session_state.get("adv_defense_strategy", "reactive"),
                # Process parameters
                "min_iterations": st.session_state.get("adv_min_iterations", 5),
                "max_iterations": st.session_state.get("adv_max_iterations", 50),
                "confidence_threshold": st.session_state.get("adv_confidence_threshold", 90),
                "evaluator_threshold": st.session_state.get("adv_evaluator_threshold", 90.0),
                "evaluator_consecutive_rounds": st.session_state.get("adv_evaluator_consecutive_rounds", 1),
                "budget_limit": st.session_state.get("adv_budget_limit", 50.0),
                # Quality control
                "critique_depth": st.session_state.get("adv_critique_depth", 5),
                "patch_quality": st.session_state.get("adv_patch_quality", 5),
                # Quality assurance
                "enable_human_feedback": st.session_state.get("adv_enable_human_feedback", False),
                "keyword_analysis_enabled": st.session_state.get("adv_keyword_analysis_enabled", True),
                "keywords_to_target": st.session_state.get("adv_keywords_to_target", ""),
                "enable_real_time_monitoring": st.session_state.get("adv_enable_real_time_monitoring", True),
                "enable_comprehensive_reporting": st.session_state.get("adv_enable_comprehensive_reporting", True),
                # Security & compliance
                "enable_encryption": st.session_state.get("adv_enable_encryption", True),
                "enable_audit_trail": st.session_state.get("adv_enable_audit_trail", True),
                # Advanced evolution
                "enable_multi_objective": st.session_state.get("adv_enable_multi_objective", False),
                "feature_dimensions": st.session_state.get("adv_feature_dimensions", ["complexity", "diversity"]),
                "feature_bins": st.session_state.get("adv_feature_bins", 10),
                "enable_data_augmentation": st.session_state.get("adv_enable_data_augmentation", False),
                "augmentation_model": st.session_state.get("adv_augmentation_model", "gpt-4o"),
                "augmentation_temperature": st.session_state.get("adv_augmentation_temperature", 0.7),
                # Evolution parameters
                "elite_ratio": st.session_state.get("adv_elite_ratio", 0.1),
                "exploration_ratio": st.session_state.get("adv_exploration_ratio", 0.2),
                "archive_size": st.session_state.get("adv_archive_size", 100),
                # Custom prompts
                "custom_red_prompt": st.session_state.get("adv_custom_red_prompt", ""),
                "custom_blue_prompt": st.session_state.get("adv_custom_blue_prompt", ""),
                "custom_approval_prompt": st.session_state.get("adv_custom_approval_prompt", "")
            }
            # ALL OpenEvolve parameters (272+ parameters)
            config["openevolve_parameters"] = self._get_all_openevolve_parameters_from_session(prefix="adv")

        elif workflow_type == "custom":
            config["teams"] = {
                "content_analyzer_team": st.session_state.get("custom_ca_team", "Default"),
                "planner_team": st.session_state.get("custom_planner_team", "Default"),
                "solver_team": st.session_state.get("custom_solver_team", "Default"),
                "patcher_team": st.session_state.get("custom_patcher_team", "Default"),
                "assembler_team": st.session_state.get("custom_assembler_team", "Default")
            }
            config["gauntlets"] = {
                "sub_problem_red_gauntlet": st.session_state.get("custom_sub_red", "None"),
                "sub_problem_gold_gauntlet": st.session_state.get("custom_sub_gold", "None"),
                "final_red_gauntlet": st.session_state.get("custom_final_red", "None"),
                "final_gold_gauntlet": st.session_state.get("custom_final_gold", "None")
            }

        return config

    def _create_sovereign_workflow_definition(self, problem_statement: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a BubbleLabs workflow definition for Sovereign Decomposition with ALL 272 parameters.
        """
        import uuid

        workflow_id = str(uuid.uuid4())
        teams = config.get("teams", {})
        gauntlets = config.get("gauntlets", {})
        openevolve_params = config.get("openevolve_parameters", {})

        # Extract key parameters for node configuration
        core_evolution = openevolve_params.get("core_evolution", {})
        model_config = openevolve_params.get("model_config", {})
        evaluation = openevolve_params.get("evaluation", {})

        # Create nodes with full configuration
        nodes = [
            {
                "id": "content_analysis",
                "type": "content_analyzer",
                "position": {"x": 0, "y": 0},
                "data": {
                    "label": "Content Analysis",
                    "team": teams.get("content_analyzer_team", "Default"),
                    "description": "Analyze problem statement and extract structured context",
                    "config": {
                        "model": model_config.get("model_id", core_evolution.get("model", "gpt-4")),
                        "temperature": core_evolution.get("temperature", 0.7),
                        "max_tokens": core_evolution.get("max_tokens", 2048)
                    }
                }
            },
            {
                "id": "decomposition",
                "type": "decomposer",
                "position": {"x": 300, "y": 0},
                "data": {
                    "label": "Problem Decomposition",
                    "team": teams.get("planner_team", "Default"),
                    "description": "Break down problem into sub-problems",
                    "config": {
                        "ensemble_size": evaluation.get("ensemble_size", 3)
                    }
                }
            },
            {
                "id": "subproblem_solver",
                "type": "solver",
                "position": {"x": 600, "y": 0},
                "data": {
                    "label": "Sub-problem Solving",
                    "team": teams.get("solver_team", "Default"),
                    "gauntlet": gauntlets.get("sub_problem_red_gauntlet", "None"),
                    "description": "Solve each sub-problem with gauntlet validation",
                    "config": {
                        "max_refinement_loops": core_evolution.get("max_iterations", 10),
                        "parallel": openevolve_params.get("selection", {}).get("multi_strategy_sampling", True)
                    }
                }
            },
            {
                "id": "final_verification",
                "type": "verifier",
                "position": {"x": 900, "y": 0},
                "data": {
                    "label": "Final Assembly & Verification",
                    "team": teams.get("assembler_team", "Default"),
                    "gauntlet": gauntlets.get("final_gold_gauntlet", "None"),
                    "description": "Assemble final solution with verification",
                    "config": {
                        "confidence_threshold": evaluation.get("consensus_threshold", 0.7)
                    }
                }
            }
        ]

        # Create edges
        edges = [
            {"id": "edge_1", "source": "content_analysis", "target": "decomposition", "sourceHandle": "output", "targetHandle": "input"},
            {"id": "edge_2", "source": "decomposition", "target": "subproblem_solver", "sourceHandle": "output", "targetHandle": "input"},
            {"id": "edge_3", "source": "subproblem_solver", "target": "final_verification", "sourceHandle": "output", "targetHandle": "input"}
        ]

        # Count total parameters
        total_params = sum(len(params) for params in openevolve_params.values())

        definition = {
            "id": workflow_id,
            "name": f"Sovereign Decomposition: {problem_statement[:30]}...",
            "description": f"OpenEvolve sovereign-grade decomposition for: {problem_statement}",
            "workflow_type": "openevolve_sovereign_decomposition",
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "problem_statement": problem_statement,
                "team_config": teams,
                "gauntlet_config": gauntlets,
                # ALL 272+ OpenEvolve parameters organized by category
                "openevolve_parameters": openevolve_params,
                "total_parameters": total_params,
                "created_at": time.time()
            }
        }

        return definition

    def _create_evolution_workflow_definition(self, problem_statement: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a BubbleLabs workflow definition for Evolution workflow with ALL 272 parameters.
        """
        import uuid

        workflow_id = str(uuid.uuid4())
        teams = config.get("teams", {})
        openevolve_params = config.get("openevolve_parameters", {})
        evo_settings = config.get("evolution_settings", {})

        # Extract key parameters
        core_evolution = openevolve_params.get("core_evolution", {})
        selection = openevolve_params.get("selection", {})
        evaluation = openevolve_params.get("evaluation", {})

        # Create evolution-specific nodes
        nodes = [
            {
                "id": "initialization",
                "type": "initializer",
                "position": {"x": 0, "y": 0},
                "data": {
                    "label": "Population Initialization",
                    "team": teams.get("content_analyzer_team", "Default"),
                    "description": "Initialize population for evolution",
                    "config": {
                        "population_size": evo_settings.get("population_size", 50)
                    }
                }
            },
            {
                "id": "evolution_engine",
                "type": "evolver",
                "position": {"x": 300, "y": 0},
                "data": {
                    "label": "Evolution Engine",
                    "team": teams.get("planner_team", "Default"),
                    "description": "Run evolutionary algorithm",
                    "config": {
                        "generations": evo_settings.get("generations", 100),
                        "mutation_rate": evo_settings.get("mutation_rate", 0.1),
                        "crossover_rate": selection.get("crossover_rate", 0.8),
                        "elite_ratio": selection.get("elite_ratio", 0.1)
                    }
                }
            },
            {
                "id": "evaluation",
                "type": "fitness_evaluator",
                "position": {"x": 600, "y": 0},
                "data": {
                    "label": "Fitness Evaluation",
                    "team": teams.get("solver_team", "Default"),
                    "description": "Evaluate fitness of solutions",
                    "config": {
                        "ensemble_size": evaluation.get("ensemble_size", 3),
                        "parallel_evaluations": evaluation.get("parallel_evaluations", 4)
                    }
                }
            },
            {
                "id": "selection",
                "type": "selector",
                "position": {"x": 900, "y": 0},
                "data": {
                    "label": "Selection & Reproduction",
                    "team": teams.get("assembler_team", "Default"),
                    "description": "Select best solutions for next generation",
                    "config": {
                        "tournament_size": selection.get("tournament_size", 3),
                        "selection_method": selection.get("selection_method", "tournament")
                    }
                }
            }
        ]

        # Create edges
        edges = [
            {"id": "edge_1", "source": "initialization", "target": "evolution_engine", "sourceHandle": "output", "targetHandle": "input"},
            {"id": "edge_2", "source": "evolution_engine", "target": "evaluation", "sourceHandle": "output", "targetHandle": "input"},
            {"id": "edge_3", "source": "evaluation", "target": "selection", "sourceHandle": "output", "targetHandle": "input"},
            {"id": "edge_4", "source": "selection", "target": "evolution_engine", "sourceHandle": "output", "targetHandle": "feedback"}
        ]

        # Count total parameters
        total_params = sum(len(params) for params in openevolve_params.values()) if openevolve_params else 0

        definition = {
            "id": workflow_id,
            "name": f"Evolution: {problem_statement[:30]}...",
            "description": f"OpenEvolve evolutionary optimization for: {problem_statement}",
            "workflow_type": "openevolve_evolution",
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "problem_statement": problem_statement,
                "team_config": teams,
                "evolution_settings": evo_settings,
                # ALL 272+ OpenEvolve parameters organized by category
                "openevolve_parameters": openevolve_params,
                "total_parameters": total_params,
                "created_at": time.time()
            }
        }

        return definition

    def _create_adversarial_workflow_definition(self, problem_statement: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a BubbleLabs workflow definition for Adversarial workflow with ALL 272 parameters.
        """
        import uuid

        workflow_id = str(uuid.uuid4())
        teams = config.get("teams", {})
        gauntlets = config.get("gauntlets", {})
        openevolve_params = config.get("openevolve_parameters", {})
        adv_settings = config.get("adversarial_settings", {})

        # Extract key parameters
        adversarial = openevolve_params.get("adversarial", {})
        core_evolution = openevolve_params.get("core_evolution", {})

        # Create adversarial-specific nodes
        nodes = [
            {
                "id": "initial_solution",
                "type": "generator",
                "position": {"x": 0, "y": 0},
                "data": {
                    "label": "Initial Solution",
                    "team": teams.get("content_analyzer_team", "Default"),
                    "description": "Generate initial solution to attack",
                    "config": {
                        "model": core_evolution.get("model", "gpt-4"),
                        "temperature": core_evolution.get("temperature", 0.7)
                    }
                }
            },
            {
                "id": "red_team_attack",
                "type": "attacker",
                "position": {"x": 300, "y": -100},
                "data": {
                    "label": "Red Team Attack",
                    "team": teams.get("red_team", "Default"),
                    "gauntlet": gauntlets.get("red_gauntlet", "None"),
                    "description": "Generate adversarial attacks",
                    "config": {
                        "attack_strength": adv_settings.get("attack_strength", 0.5),
                        "adversarial_rounds": adv_settings.get("adversarial_rounds", 5),
                        "adversarial_temperature": adversarial.get("adversarial_temperature", 0.8)
                    }
                }
            },
            {
                "id": "blue_team_defense",
                "type": "defender",
                "position": {"x": 300, "y": 100},
                "data": {
                    "label": "Blue Team Defense",
                    "team": teams.get("blue_team", "Default"),
                    "gauntlet": gauntlets.get("blue_gauntlet", "None"),
                    "description": "Defend against attacks",
                    "config": {
                        "defense_strength": adv_settings.get("defense_strength", 1.0),
                        "defense_strategy": adv_settings.get("defense_strategy", "reactive"),
                        "ensemble_defense": adversarial.get("ensemble_defense", True)
                    }
                }
            },
            {
                "id": "verification",
                "type": "verifier",
                "position": {"x": 600, "y": 0},
                "data": {
                    "label": "Adversarial Verification",
                    "team": teams.get("assembler_team", "Default"),
                    "description": "Verify robustness of solution",
                    "config": {
                        "robustness_metric": adversarial.get("robustness_metric", "accuracy"),
                        "perturbation_bound": adversarial.get("perturbation_bound", 0.1)
                    }
                }
            }
        ]

        # Create edges
        edges = [
            {"id": "edge_1", "source": "initial_solution", "target": "red_team_attack", "sourceHandle": "output", "targetHandle": "input"},
            {"id": "edge_2", "source": "initial_solution", "target": "blue_team_defense", "sourceHandle": "output", "targetHandle": "input"},
            {"id": "edge_3", "source": "red_team_attack", "target": "verification", "sourceHandle": "output", "targetHandle": "attack_input"},
            {"id": "edge_4", "source": "blue_team_defense", "target": "verification", "sourceHandle": "output", "targetHandle": "defense_input"},
            {"id": "edge_5", "source": "verification", "target": "red_team_attack", "sourceHandle": "output", "targetHandle": "feedback", "label": "improve"}
        ]

        # Count total parameters
        total_params = sum(len(params) for params in openevolve_params.values()) if openevolve_params else 0

        definition = {
            "id": workflow_id,
            "name": f"Adversarial Testing: {problem_statement[:30]}...",
            "description": f"OpenEvolve adversarial testing for: {problem_statement}",
            "workflow_type": "openevolve_adversarial",
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "problem_statement": problem_statement,
                "team_config": teams,
                "gauntlet_config": gauntlets,
                "adversarial_settings": adv_settings,
                # ALL 272+ OpenEvolve parameters organized by category
                "openevolve_parameters": openevolve_params,
                "total_parameters": total_params,
                "created_at": time.time()
            }
        }

        return definition

    def _display_workflow_graph(self, workflow_def: Dict[str, Any]):
        """
        Display a simplified representation of the workflow graph.
        """
        import streamlit.components.v1 as components

        # Security: Sanitize node labels to prevent DOM-based XSS
        safe_nodes = []
        for node in workflow_def.get('nodes', []):
            safe_node = node.copy()
            if 'data' in safe_node and 'label' in safe_node['data']:
                # Escape HTML in labels
                safe_node['data'] = safe_node['data'].copy()
                safe_node['data']['label'] = escape_html(safe_node['data']['label'])
            safe_nodes.append(safe_node)

        # Create a basic visualization using Mermaid.js (SECURE: Escaped data)
        mermaid_code = f"""
        graph TD
            A[Start] --> B[{safe_nodes[0]['data']['label'] if len(safe_nodes) > 0 else 'Node 1'}]
            B --> C[{safe_nodes[1]['data']['label'] if len(safe_nodes) > 1 else 'Node 2'}]
            C --> D[{safe_nodes[2]['data']['label'] if len(safe_nodes) > 2 else 'Node 3'}]
            D --> E[{safe_nodes[3]['data']['label'] if len(safe_nodes) > 3 else 'Node 4'}]
            E --> F[End]
        """

        st.markdown("### Workflow Structure")

        # Security: Escape workflow_def for JavaScript to prevent DOM-based XSS
        safe_workflow_json = escape_json_for_js({
            'id': workflow_def.get('id', ''),
            'nodes': safe_nodes
        })

        # Use escaped ID
        escaped_id = escape_html(workflow_def.get('id', 'unknown')).replace("'", "\\'")

        components.html(f"""
        <div id="mermaid-container"></div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.mermaidAPI.initialize({{
                startOnLoad: true,
                securityLevel: 'strict'
            }});

            // Safely render using escaped data
            const mermaidCode = `{mermaid_code}`;
            mermaid.render('workflow-{escaped_id}', mermaidCode, function(svgCode, bindFunctions){{
                document.getElementById("mermaid-container").innerHTML = svgCode;
            }});
        </script>
        """, height=300)
    
    def _create_and_execute_instance_local(self, definition_id: str, input_data: Dict[str, Any]):
        """
        Create and execute a workflow instance locally using OpenEvolve orchestrator patterns.
        """
        import uuid

        # Create instance
        instance_id = str(uuid.uuid4())

        # Extract configurations
        problem_statement = input_data.get("problem_statement", "Unknown problem")
        workflow_type = input_data.get("workflow_type", "custom")
        team_config = input_data.get("team_config", {})
        gauntlet_config = input_data.get("gauntlet_config", {})
        openevolve_params = input_data.get("openevolve_parameters", {})

        # Extract core parameters for direct WorkflowState fields
        core_evolution = openevolve_params.get("core_evolution", {})
        quality_diversity = openevolve_params.get("quality_diversity", {})
        evaluation = openevolve_params.get("evaluation", {})
        resource_management = openevolve_params.get("resource_management", {})
        database_storage = openevolve_params.get("database_storage", {})
        evolution_tracing = openevolve_params.get("evolution_tracing", {})
        early_stopping = openevolve_params.get("early_stopping", {})
        selection = openevolve_params.get("selection", {})
        prompt_engineering = openevolve_params.get("prompt_engineering", {})
        artifact_management = openevolve_params.get("artifact_management", {})
        distributed_processing = openevolve_params.get("distributed_processing", {})

        # Create the OpenEvolve WorkflowState following the same pattern as openevolve_orchestrator
        # Populated with ALL 272+ parameters from the UI
        workflow_state = WorkflowState(
            workflow_id=instance_id,
            workflow_type="bubblelabs_openevolve",
            problem_statement=problem_statement,
            current_stage="INITIALIZING",
            status="pending",
            # Core Evolution parameters
            max_iterations=core_evolution.get("max_iterations", 100),
            population_size=core_evolution.get("population_size", 50),
            temperature=core_evolution.get("temperature", 0.7),
            max_tokens=core_evolution.get("max_tokens", 2048),
            top_p=core_evolution.get("top_p", 1.0),
            frequency_penalty=core_evolution.get("frequency_penalty", 0.0),
            presence_penalty=core_evolution.get("presence_penalty", 0.0),
            seed=core_evolution.get("seed"),
            random_seed=core_evolution.get("random_seed", 42),
            api_timeout=core_evolution.get("api_timeout", 60),
            api_retries=core_evolution.get("api_retries", 3),
            api_retry_delay=core_evolution.get("api_retry_delay", 5.0),
            # Quality Diversity parameters
            num_islands=quality_diversity.get("archive_size", 10),
            archive_size=quality_diversity.get("archive_size", 10),
            elite_ratio=selection.get("elite_ratio", 0.1),
            exploration_ratio=selection.get("exploration_ratio", 0.2),
            exploitation_ratio=selection.get("exploitation_ratio", 0.7),
            checkpoint_interval=resource_management.get("checkpoint_interval", 10),
            feature_bins=quality_diversity.get("feature_bins", 10),
            diversity_metric=quality_diversity.get("diversity_metric", "edit_distance"),
            # Evaluation parameters
            cascade_evaluation=evaluation.get("cascade_evaluation", False),
            cascade_thresholds=evaluation.get("cascade_thresholds", [0.5, 0.75, 0.9]),
            use_llm_feedback=evaluation.get("use_llm_feedback", True),
            llm_feedback_weight=evaluation.get("llm_feedback_weight", 0.1),
            parallel_evaluations=evaluation.get("parallel_evaluations", 4),
            distributed=distributed_processing.get("distributed", False),
            num_top_programs=evaluation.get("ensemble_size", 3),
            num_diverse_programs=evaluation.get("ensemble_size", 3),
            # Prompt Engineering parameters
            use_template_stochasticity=prompt_engineering.get("template_stochasticity", True),
            use_meta_prompting=prompt_engineering.get("meta_prompting", False),
            meta_prompt_weight=0.5,
            # Artifact Management parameters
            include_artifacts=artifact_management.get("enable_artifacts", True),
            max_artifact_bytes=artifact_management.get("max_artifact_size", 20480),
            artifact_security_filter=artifact_management.get("artifact_validation", True),
            # Early Stopping parameters
            early_stopping_patience=early_stopping.get("early_stopping_patience", 10),
            convergence_threshold=early_stopping.get("min_improvement", 0.001),
            # Resource Management parameters
            memory_limit_mb=resource_management.get("memory_limit_mb", 2048),
            cpu_limit=resource_management.get("cpu_limit", 0.8),
            # Database Storage parameters
            db_path=database_storage.get("db_path", "./openevolve.db"),
            in_memory=database_storage.get("db_type") == "memory",
            # Evolution Tracing parameters
            evolution_trace_enabled=evolution_tracing.get("trace_enabled", False),
            evolution_trace_format=evolution_tracing.get("trace_format", "json"),
            evolution_trace_output_path=evolution_tracing.get("trace_file", "./trace.log"),
            evolution_trace_buffer_size=evolution_tracing.get("trace_buffer_size", 100),
            evolution_trace_compress=evolution_tracing.get("trace_compression", True),
            # MAX REFINEMENT LOOPS
            max_refinement_loops=core_evolution.get("max_iterations", 100)
        )

        # Store ALL 272+ parameters in the openevolve_parameters field for later access
        workflow_state.openevolve_parameters = openevolve_params

        # Get and assign teams and gauntlets to the workflow state
        workflow_state.content_analyzer_team = self.team_manager.get_team(team_config.get("content_analyzer_team", ""))
        workflow_state.planner_team = self.team_manager.get_team(team_config.get("planner_team", ""))
        workflow_state.solver_team = self.team_manager.get_team(team_config.get("solver_team", ""))
        workflow_state.patcher_team = self.team_manager.get_team(team_config.get("patcher_team", ""))
        workflow_state.assembler_team = self.team_manager.get_team(team_config.get("assembler_team", ""))

        workflow_state.sub_problem_red_gauntlet = self.gauntlet_manager.get_gauntlet(gauntlet_config.get("sub_problem_red_gauntlet", ""))
        workflow_state.sub_problem_gold_gauntlet = self.gauntlet_manager.get_gauntlet(gauntlet_config.get("sub_problem_gold_gauntlet", ""))
        workflow_state.final_red_gauntlet = self.gauntlet_manager.get_gauntlet(gauntlet_config.get("final_red_gauntlet", ""))
        workflow_state.final_gold_gauntlet = self.gauntlet_manager.get_gauntlet(gauntlet_config.get("final_gold_gauntlet", ""))

        # Store in session state like OpenEvolve does (using active_sovereign_workflow pattern)
        st.session_state.active_sovereign_workflow = workflow_state

        param_count = sum(len(v) for v in openevolve_params.values()) if openevolve_params else 0
        st.success(f"Workflow instance created! ID: {instance_id}")
        st.info(f"Configured with {param_count} OpenEvolve parameters across {len(openevolve_params)} categories")

        # Execute the instance in a background thread, similar to OpenEvolve orchestrator
        self._execute_workflow_instance_local(workflow_state)

        st.success("Workflow execution started successfully!")
    
    def _execute_workflow_instance_local(self, workflow_state: WorkflowState):
        """
        Execute the workflow instance locally in a background thread, following OpenEvolve orchestrator patterns.
        """
        def run_workflow():
            try:
                # Update workflow state to running (as done in OpenEvolve orchestrator)
                workflow_state.status = "running"
                workflow_state.current_stage = "content_analysis"

                # Log the full parameter configuration
                param_categories = len(workflow_state.openevolve_parameters) if workflow_state.openevolve_parameters else 0
                param_count = sum(len(v) for v in workflow_state.openevolve_parameters.values()) if workflow_state.openevolve_parameters else 0
                logger.info(f"Executing workflow {workflow_state.workflow_id} with {param_count} OpenEvolve parameters across {param_categories} categories")

                # Log each category's parameter count
                if workflow_state.openevolve_parameters:
                    for category, params in workflow_state.openevolve_parameters.items():
                        logger.info(f"  {category}: {len(params)} parameters")

                # Run the actual OpenEvolve workflow (this modifies workflow_state in place)
                # The workflow_state now contains ALL 272+ parameters
                run_sovereign_workflow(
                    workflow_state=workflow_state,
                    content_analyzer_team=workflow_state.content_analyzer_team,
                    planner_team=workflow_state.planner_team,
                    solver_team=workflow_state.solver_team,
                    patcher_team=workflow_state.patcher_team,
                    assembler_team=workflow_state.assembler_team,
                    sub_problem_red_gauntlet=workflow_state.sub_problem_red_gauntlet,
                    sub_problem_gold_gauntlet=workflow_state.sub_problem_gold_gauntlet,
                    final_red_gauntlet=workflow_state.final_red_gauntlet,
                    final_gold_gauntlet=workflow_state.final_gold_gauntlet,
                    max_refinement_loops=workflow_state.max_refinement_loops
                )

                logger.info(f"Workflow {workflow_state.workflow_id} completed with status: {workflow_state.status}")

            except Exception as e:
                # Update workflow state with error status
                workflow_state.status = "failed"
                logger.error(f"Error executing workflow {workflow_state.workflow_id}: {e}", exc_info=True)
                print(f"Error executing workflow {workflow_state.workflow_id}: {e}")

        # Start the workflow execution in a background thread
        thread = threading.Thread(target=run_workflow)
        thread.daemon = True
        thread.start()
    
    def _get_all_workflow_instances(self) -> List[Dict[str, Any]]:
        """
        Get all workflow instances from Streamlit session state, following OpenEvolve patterns.
        """
        instances = []
        
        # Get the active sovereign workflow if it exists (following OpenEvolve pattern)
        if "active_sovereign_workflow" in st.session_state:
            active_wf = st.session_state.active_sovereign_workflow
            instance = {
                "id": active_wf.workflow_id,
                "status": active_wf.status,
                "progress": active_wf.progress,
                "current_node": active_wf.current_stage,
                "created_at": active_wf.start_time,
                "updated_at": time.time(),
                "data": {
                    "problem_statement": active_wf.problem_statement
                }
            }
            instances.append(instance)
        
        # Get any other workflow instances from session state
        session_instances = st.session_state.get("active_workflow_instances", [])
        instances.extend(session_instances)
        
        return instances
    
    def _render_active_workflows(self):
        """
        Render the active workflows list.
        """
        st.subheader("Active Workflow Instances")
        
        instances = self._get_all_workflow_instances()
        
        if not instances:
            st.info("No active workflow instances found.")
            return
        
        for instance in instances:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.write(f"**ID:** `{instance['id']}`")
                
                with col2:
                    status_badge = f":{'green' if instance['status'] == 'completed' else 'orange' if instance['status'] in ['running', 'pending'] else 'red'}[{instance['status'].upper()}]"
                    st.write(f"**Status:** {status_badge}")
                
                with col3:
                    progress = instance.get('progress', 0)
                    st.write(f"**Progress:** {progress * 100:.1f}%")
                
                with col4:
                    if st.button(f"View", key=f"view_{instance['id']}"):
                        # Show detailed information for this instance
                        with st.expander(f"Details for {instance['id']}"):
                            st.json(instance)
        
        # Refresh button
        if st.button("Refresh Status"):
            st.rerun()
    
    def _control_workflow_local(self, instance_id: str, action: str):
        """
        Control a workflow instance locally, following OpenEvolve patterns.
        """
        # Check for the active sovereign workflow (primary OpenEvolve pattern)
        if "active_sovereign_workflow" in st.session_state:
            workflow_state = st.session_state.active_sovereign_workflow
            if workflow_state.workflow_id == instance_id:
                if action == "start":
                    if workflow_state.status in ["pending", "created"]:
                        workflow_state.status = "running"
                        st.success(f"Action '{action}' performed successfully")
                    else:
                        st.warning(f"Cannot start workflow in status: {workflow_state.status}")

                elif action == "pause":
                    if workflow_state.status == "running":
                        workflow_state.status = "paused"
                        workflow_state.progress = workflow_state.progress  # Preserve progress
                        st.success(f"Action '{action}' performed successfully")
                    else:
                        st.warning(f"Cannot pause workflow in status: {workflow_state.status}")

                elif action == "resume":
                    if workflow_state.status == "paused":
                        workflow_state.status = "running"
                        st.success(f"Action '{action}' performed successfully")
                    else:
                        st.warning(f"Cannot resume workflow in status: {workflow_state.status}")

                elif action == "cancel":
                    workflow_state.status = "cancelled"
                    if "active_sovereign_workflow" in st.session_state:
                        del st.session_state.active_sovereign_workflow  # Remove from session state like in openevolve_orchestrator
                    st.success(f"Action '{action}' performed successfully")

                elif action == "restart":
                    # For restart, we need to re-initialize the workflow state
                    st.warning("Restart functionality requires creating a new workflow instance.")
                
                # Refresh the UI
                st.rerun()
                return

        st.error(f"Workflow instance {instance_id} not found")

    def _render_workflow_control(self):
        """
        Render the workflow control interface, following OpenEvolve patterns.
        """
        st.subheader("Workflow Control")

        # Get all instances using the local method
        instances = self._get_all_workflow_instances()

        if not instances:
            st.info("No workflow instances available for control.")
            return

        # Create a selection for workflow instances
        instance_options = {f"{inst['id']} - {inst['status']}": inst['id'] for inst in instances}

        selected_instance_id = st.selectbox(
            "Select Workflow Instance",
            options=list(instance_options.values()),
            format_func=lambda x: [k for k, v in instance_options.items() if v == x][0]
        )

        if selected_instance_id:
            # Get the workflow state from session state (following OpenEvolve pattern)
            workflow_state = None
            if "active_sovereign_workflow" in st.session_state:
                active_wf = st.session_state.active_sovereign_workflow
                if active_wf.workflow_id == selected_instance_id:
                    workflow_state = active_wf
            
            if workflow_state:
                st.write(f"**Current Status:** :{':green' if workflow_state.status == 'completed' else ':orange' if workflow_state.status in ['running', 'pending'] else ':red'}[{workflow_state.status.upper()}]")
                st.write(f"**Progress:** {workflow_state.progress * 100:.1f}%")
                if workflow_state.current_stage:
                    st.write(f"**Current Stage:** {workflow_state.current_stage}")

            # Control actions
            st.subheader("Control Actions")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("Start", key=f"start_{selected_instance_id}"):
                    self._control_workflow_local(selected_instance_id, "start")

            with col2:
                if st.button("Pause", key=f"pause_{selected_instance_id}"):
                    self._control_workflow_local(selected_instance_id, "pause")

            with col3:
                if st.button("Resume", key=f"resume_{selected_instance_id}"):
                    self._control_workflow_local(selected_instance_id, "resume")

            with col4:
                if st.button("Cancel", key=f"cancel_{selected_instance_id}"):
                    self._control_workflow_local(selected_instance_id, "cancel")

            # Advanced controls
            st.subheader("Advanced Controls")
            col5, col6 = st.columns(2)

            with col5:
                if st.button("Restart", key=f"restart_{selected_instance_id}"):
                    self._control_workflow_local(selected_instance_id, "restart")

            # Display full instance details for local instance
            with col6:
                if st.button("Get Full Status", key=f"status_{selected_instance_id}"):
                    if "active_sovereign_workflow" in st.session_state:
                        active_wf = st.session_state.active_sovereign_workflow
                        if active_wf.workflow_id == selected_instance_id:
                            instance_data = {
                                "id": active_wf.workflow_id,
                                "status": active_wf.status,
                                "current_stage": active_wf.current_stage,
                                "progress": active_wf.progress,
                                "problem_statement": active_wf.problem_statement,
                                "start_time": active_wf.start_time,
                                "end_time": active_wf.end_time,
                                "refinement_loop_count": active_wf.refinement_loop_count
                            }
                            with st.expander("Full Instance Details"):
                                st.json(instance_data)
                        else:
                            st.error(f"Workflow instance {selected_instance_id} not found")
                    else:
                        st.error(f"Workflow instance {selected_instance_id} not found")

    def _render_parameter_sync_status(self):
        """
        Render the parameter synchronization status between Streamlit and BubbleLabs.
        """
        st.subheader("🔄 Parameter Synchronization Status")

        # Get sync status
        sync_status = self.param_sync.get_parameter_sync_status()
        sync_metrics = self.param_sync.get_sync_metrics()

        # Display sync metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Parameters", sync_metrics["total_parameters"])
        with col2:
            st.metric("Synced Parameters", sync_metrics["synced_parameters"])
        with col3:
            st.metric("Sync %", f"{sync_metrics['sync_percentage']:.1f}%")
        with col4:
            status_icon = "✅" if sync_metrics["is_fully_synced"] else "⚠️"
            status_text = "Fully Synced" if sync_metrics["is_fully_synced"] else "Partial Sync"
            st.metric("Sync Status", f"{status_icon} {status_text}")

        # Display detailed parameter sync status
        st.subheader("Parameter Details")
        
        # Create a table for parameter sync status
        param_data = []
        for param_name, status_info in sync_status["parameter_statuses"].items():
            streamlit_key = self.param_sync.parameter_mapping.get(param_name, {}).get("streamlit_key", "unknown")
            param_data.append({
                "Parameter": param_name,
                "Streamlit Key": streamlit_key,
                "Sync Status": "✅ Synced" if status_info["is_synced"] else "❌ Not Synced",
                "Validation": "✅ Valid" if status_info["validation_status"] else "❌ Invalid",
                "Value": str(status_info["streamlit_value"])[:50] + "..." if status_info["streamlit_value"] and len(str(status_info["streamlit_value"])) > 50 else str(status_info["streamlit_value"])
            })
        
        import pandas as pd
        df = pd.DataFrame(param_data)
        st.dataframe(df, use_container_width=True, height=400)

        # Sync controls
        st.subheader("Sync Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Force Resync All", key="force_resync"):
                result = self.param_sync.force_resync_all()
                st.success("Parameters resynchronized!")
                st.json(result)
        
        with col2:
            if st.button("📊 View Sync History", key="view_sync_history"):
                changes = self.param_sync.get_recent_changes(limit=20)
                if changes:
                    for change in changes[-10:]:  # Show last 10 changes
                        st.write(f"**{change.name}**: {change.old_value} → {change.new_value} ({change.source_ui}, {time.ctime(change.timestamp)})")
                else:
                    st.info("No sync history available")


# Global function to render the BubbleLabs workflow UI
def render_bubblelabs_workflow_ui():
    """
    Render the BubbleLabs workflow visualization and control UI.
    """
    ui = BubbleLabsWorkflowUI()
    ui.render_workflow_visualizer()


# For testing purposes
if __name__ == "__main__":
    st.set_page_config(page_title="BubbleLabs Workflow UI", layout="wide")
    render_bubblelabs_workflow_ui()