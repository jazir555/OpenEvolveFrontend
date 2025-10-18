"""
OpenEvolve Orchestration System
Advanced workflow orchestration for ALL OpenEvolve features
"""
import streamlit as st
import time
import threading
import os
import sys
import subprocess
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import requests # Added this line
import logging # Added this line
from ui_components import render_team_manager, render_gauntlet_designer # NEW IMPORT
from team_manager import TeamManager # NEW IMPORT
from gauntlet_manager import GauntletManager # NEW IMPORT
from template_manager import TemplateManager, WorkflowTemplate # NEW IMPORT
from workflow_engine import run_sovereign_workflow, WorkflowState # NEW IMPORT
from workflow_history_manager import WorkflowHistoryManager # NEW IMPORT


def get_project_root():
    """
    Returns the absolute path to the project's root directory.
    This is a local copy to avoid import issues in threaded contexts.
    """
    try:
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to get the project root
        return os.path.abspath(os.path.join(current_dir, os.pardir))
    except Exception as e:
        logging.error(f"Error getting project root: {e}")
        # Fallback to current working directory
        return os.getcwd()

# Import OpenEvolve components
try:
    from openevolve_integration import (
        run_unified_evolution,
        create_comprehensive_openevolve_config
    )
    from monitoring_system import EvolutionMonitor
    from reporting_system import create_evolution_report
    ORCHESTRATOR_AVAILABLE = True # Flag indicating if OpenEvolve orchestrator components are available
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    print("OpenEvolve orchestrator components not available")


class EvolutionWorkflow(Enum):
    """Different evolution workflow types"""
    STANDARD = "standard"
    QUALITY_DIVERSITY = "quality_diversity"
    MULTI_OBJECTIVE = "multi_objective"
    ADVERSARIAL = "adversarial"
    SYMBOLIC_REGRESSION = "symbolic_regression"
    NEUROEVOLUTION = "neuroevolution"
    ALGORITHM_DISCOVERY = "algorithm_discovery"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    PROBLEM_DECOMPOSITION = "problem_decomposition"
    SOVEREIGN_DECOMPOSITION = "sovereign_decomposition"


class WorkflowStage(Enum):
    """Stages of an evolution workflow"""
    INITIALIZATION = "initialization"
    CONFIGURATION = "configuration"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    COMPLETION = "completion"


@dataclass
class WorkflowState:
    """State of an evolution workflow"""
    workflow_id: str
    workflow_type: EvolutionWorkflow
    current_stage: WorkflowStage
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    metrics: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, completed, failed, cancelled
    progress: float = 0.0  # 0.0 to 1.0


class OpenEvolveOrchestrator:
    """Orchestrates complex OpenEvolve workflows, managing their lifecycle, parameters, and monitoring.
    It handles the creation, starting, stopping, and monitoring of various evolution workflow types.
    """
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowState] = {}
        self.active_workflows: List[str] = []
        self.monitor = EvolutionMonitor() if ORCHESTRATOR_AVAILABLE else None
        self.workflow_callbacks: Dict[str, List[Callable]] = {}
        self.history_manager = WorkflowHistoryManager() # Initialize history manager
        
    def create_workflow(
        self, 
        workflow_type: EvolutionWorkflow,
        parameters: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> str:
        """Creates a new evolution workflow and initializes its state.

        Args:
            workflow_type (EvolutionWorkflow): The type of evolution workflow to create.
            parameters (Dict[str, Any]): A dictionary of parameters specific to the workflow type.
            workflow_id (Optional[str]): An optional unique ID for the workflow. If None, a timestamp-based ID is generated.

        Returns:
            str: The unique ID of the created workflow.
        """
        if not workflow_id:
            workflow_id = f"workflow_{int(time.time())}"
            
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            current_stage=WorkflowStage.INITIALIZATION,
            parameters=parameters,
            results={},
            metrics={},
            start_time=time.time()
        )
        
        self.workflows[workflow_id] = workflow_state
        self.active_workflows.append(workflow_id)
        
        return workflow_id
    
    def start_workflow(self, workflow_id: str) -> bool:
        """Starts the execution of a previously created workflow in a separate thread.

        Args:
            workflow_id (str): The ID of the workflow to start.

        Returns:
            bool: True if the workflow was successfully started, False otherwise.
        """
        if workflow_id not in self.workflows:
            st.error(f"Workflow {workflow_id} not found")
            return False
            
        workflow = self.workflows[workflow_id]
        workflow.current_stage = WorkflowStage.CONFIGURATION
        workflow.status = "running"
        
        # Start workflow in a separate thread
        workflow_thread = threading.Thread(
            target=self._execute_workflow,
            args=(workflow_id,),
            daemon=True
        )
        workflow_thread.start()
        
        return True
    
    def stop_workflow(self, workflow_id: str) -> bool:
        """Stops a running workflow.

        Args:
            workflow_id (str): The ID of the workflow to stop.

        Returns:
            bool: True if the workflow was successfully stopped, False otherwise.
        """
        if workflow_id not in self.workflows:
            return False
            
        workflow = self.workflows[workflow_id]
        workflow.status = "cancelled"
        workflow.end_time = time.time()
        
        # Add to history manager
        self.history_manager.add_workflow_to_history(workflow)

        if workflow_id in self.active_workflows:
            self.active_workflows.remove(workflow_id)
            
        return True
    
    def _execute_workflow(self, workflow_id: str):
        """Executes the steps of a specific workflow, handling its stages and parameter configurations.
        This method is designed to run in a separate thread.

        Args:
            workflow_id (str): The ID of the workflow to execute.
        """
        try:
            workflow = self.workflows[workflow_id]
            
            # Configuration stage
            workflow.current_stage = WorkflowStage.CONFIGURATION
            self._notify_callbacks(workflow_id, "stage_changed", workflow.current_stage)
            
            # Create OpenEvolve configuration with ALL parameters
            config = create_comprehensive_openevolve_config(
                # Core content parameters
                content_type=workflow.parameters.get("content_type", "code_python"),
                model_configs=workflow.parameters.get("model_configs", [{"name": "gpt-4o", "weight": 1.0}]),
                api_key=workflow.parameters.get("api_key", ""),
                api_base=workflow.parameters.get("api_base", "https://api.openai.com/v1"),
                temperature=workflow.parameters.get("temperature", 0.7),
                top_p=workflow.parameters.get("top_p", 0.95),
                max_tokens=workflow.parameters.get("max_tokens", 4096),
                max_iterations=workflow.parameters.get("max_iterations", 100),
                population_size=workflow.parameters.get("population_size", 1000),
                num_islands=workflow.parameters.get("num_islands", 5),
                migration_interval=workflow.parameters.get("migration_interval", 50),
                migration_rate=workflow.parameters.get("migration_rate", 0.1),
                archive_size=workflow.parameters.get("archive_size", 100),
                elite_ratio=workflow.parameters.get("elite_ratio", 0.1),
                exploration_ratio=workflow.parameters.get("exploration_ratio", 0.2),
                exploitation_ratio=workflow.parameters.get("exploitation_ratio", 0.7),
                checkpoint_interval=workflow.parameters.get("checkpoint_interval", 100),
                feature_dimensions=workflow.parameters.get("feature_dimensions", ["complexity", "diversity"]),
                feature_bins=workflow.parameters.get("feature_bins", 10),
                diversity_metric=workflow.parameters.get("diversity_metric", "edit_distance"),
                system_message=workflow.parameters.get("system_message", None),
                evaluator_system_message=workflow.parameters.get("evaluator_system_message", None),
                
                # Advanced evaluation parameters
                enable_artifacts=workflow.parameters.get("enable_artifacts", True),
                cascade_evaluation=workflow.parameters.get("cascade_evaluation", True),
                cascade_thresholds=workflow.parameters.get("cascade_thresholds", [0.5, 0.75, 0.9]),
                use_llm_feedback=workflow.parameters.get("use_llm_feedback", False),
                llm_feedback_weight=workflow.parameters.get("llm_feedback_weight", 0.1),
                parallel_evaluations=workflow.parameters.get("parallel_evaluations", 4),
                distributed=workflow.parameters.get("distributed", False),
                template_dir=workflow.parameters.get("template_dir", None),
                num_top_programs=workflow.parameters.get("num_top_programs", 3),
                num_diverse_programs=workflow.parameters.get("num_diverse_programs", 2),
                use_template_stochasticity=workflow.parameters.get("use_template_stochasticity", True),
                template_variations=workflow.parameters.get("template_variations", None),
                use_meta_prompting=workflow.parameters.get("use_meta_prompting", False),
                meta_prompt_weight=workflow.parameters.get("meta_prompt_weight", 0.1),
                include_artifacts=workflow.parameters.get("include_artifacts", True),
                max_artifact_bytes=workflow.parameters.get("max_artifact_bytes", 20 * 1024),
                artifact_security_filter=workflow.parameters.get("artifact_security_filter", True),
                early_stopping_patience=workflow.parameters.get("early_stopping_patience", None),
                convergence_threshold=workflow.parameters.get("convergence_threshold", 0.001),
                early_stopping_metric=workflow.parameters.get("early_stopping_metric", "combined_score"),
                memory_limit_mb=workflow.parameters.get("memory_limit_mb", None),
                cpu_limit=workflow.parameters.get("cpu_limit", None),
                random_seed=workflow.parameters.get("random_seed", 42),
                db_path=workflow.parameters.get("db_path", None),
                in_memory=workflow.parameters.get("in_memory", True),
                
                # Advanced OpenEvolve parameters
                diff_based_evolution=workflow.parameters.get("diff_based_evolution", True),
                max_code_length=workflow.parameters.get("max_code_length", 10000),
                evolution_trace_enabled=workflow.parameters.get("evolution_trace_enabled", False),
                evolution_trace_format=workflow.parameters.get("evolution_trace_format", "jsonl"),
                evolution_trace_include_code=workflow.parameters.get("evolution_trace_include_code", False),
                evolution_trace_include_prompts=workflow.parameters.get("evolution_trace_include_prompts", True),
                evolution_trace_output_path=workflow.parameters.get("evolution_trace_output_path", None),
                evolution_trace_buffer_size=workflow.parameters.get("evolution_trace_buffer_size", 10),
                evolution_trace_compress=workflow.parameters.get("evolution_trace_compress", False),
                log_level=workflow.parameters.get("log_level", "INFO"),
                log_dir=workflow.parameters.get("log_dir", None),
                api_timeout=workflow.parameters.get("api_timeout", 60),
                api_retries=workflow.parameters.get("api_retries", 3),
                api_retry_delay=workflow.parameters.get("api_retry_delay", 5),
                artifact_size_threshold=workflow.parameters.get("artifact_size_threshold", 32 * 1024),
                cleanup_old_artifacts=workflow.parameters.get("cleanup_old_artifacts", True),
                artifact_retention_days=workflow.parameters.get("artifact_retention_days", 30),
                diversity_reference_size=workflow.parameters.get("diversity_reference_size", 20),
                max_retries_eval=workflow.parameters.get("max_retries_eval", 3),
                evaluator_timeout=workflow.parameters.get("evaluator_timeout", 300),
                evaluator_models=workflow.parameters.get("evaluator_models", None),
                output_dir=output_dir,
                
                # Advanced research-grade features
                double_selection=workflow.parameters.get("double_selection", True),
                adaptive_feature_dimensions=workflow.parameters.get("adaptive_feature_dimensions", True),
                test_time_compute=workflow.parameters.get("test_time_compute", False),
                optillm_integration=workflow.parameters.get("optillm_integration", False),
                plugin_system=workflow.parameters.get("plugin_system", False),
                hardware_optimization=workflow.parameters.get("hardware_optimization", False),
                multi_strategy_sampling=workflow.parameters.get("multi_strategy_sampling", True),
                ring_topology=workflow.parameters.get("ring_topology", True),
                controlled_gene_flow=workflow.parameters.get("controlled_gene_flow", True),
                auto_diff=workflow.parameters.get("auto_diff", True),
                symbolic_execution=workflow.parameters.get("symbolic_execution", False),
                coevolutionary_approach=workflow.parameters.get("coevolutionary_approach", False),
            )
            
            if not config:
                workflow.status = "failed"
                workflow.end_time = time.time()
                self._notify_callbacks(workflow_id, "workflow_failed", "Failed to create configuration")
                return
            
            # Execution stage
            workflow.current_stage = WorkflowStage.EXECUTION
            self._notify_callbacks(workflow_id, "stage_changed", workflow.current_stage)
            
            # Start monitoring if available
            if self.monitor:
                self.monitor.start_monitoring()
            
            # Define output directory for checkpoints
            checkpoint_base_dir = os.path.join(os.getcwd(), "openevolve_checkpoints")
            output_dir = os.path.join(checkpoint_base_dir, workflow_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Run evolution with ALL parameters
            result = run_unified_evolution(
            
                            content=workflow.parameters.get("content", ""),
            
                            content_type=workflow.parameters.get("content_type", "code_python"),
            
                            evolution_mode=workflow.workflow_type.value,
            
                            model_configs=workflow.parameters.get("model_configs", [{"name": "gpt-4o", "weight": 1.0}]),
            
                            api_key=workflow.parameters.get("api_key", ""),
            
                            api_base=workflow.parameters.get("api_base", "https://api.openai.com/v1"),
            
                            max_iterations=workflow.parameters.get("max_iterations", 100),
            
                            population_size=workflow.parameters.get("population_size", 1000),
            
                            system_message=workflow.parameters.get("system_message", ""),
            
                            evaluator_system_message=workflow.parameters.get("evaluator_system_message", ""),
            
                            temperature=workflow.parameters.get("temperature", 0.7),
            
                            top_p=workflow.parameters.get("top_p", 0.95),
            
                            max_tokens=workflow.parameters.get("max_tokens", 4096),
            
                            feature_dimensions=workflow.parameters.get("feature_dimensions", ["complexity", "diversity"]),
            
                            feature_bins=workflow.parameters.get("feature_bins", 10),
            
                            num_islands=workflow.parameters.get("num_islands", 5),
            
                            migration_interval=workflow.parameters.get("migration_interval", 50),
            
                            migration_rate=workflow.parameters.get("migration_rate", 0.1),
            
                            archive_size=workflow.parameters.get("archive_size", 100),
            
                            elite_ratio=workflow.parameters.get("elite_ratio", 0.1),
            
                            exploration_ratio=workflow.parameters.get("exploration_ratio", 0.2),
            
                            exploitation_ratio=workflow.parameters.get("exploitation_ratio", 0.7),
            
                            checkpoint_interval=workflow.parameters.get("checkpoint_interval", 100),
            
                            enable_artifacts=workflow.parameters.get("enable_artifacts", True),
            
                            cascade_evaluation=workflow.parameters.get("cascade_evaluation", True),
            
                            use_llm_feedback=workflow.parameters.get("use_llm_feedback", False),
            
                            llm_feedback_weight=workflow.parameters.get("llm_feedback_weight", 0.1),
            
                            evolution_trace_enabled=workflow.parameters.get("evolution_trace_enabled", False),
            
                            early_stopping_patience=workflow.parameters.get("early_stopping_patience", None),
            
                            early_stopping_metric=workflow.parameters.get("early_stopping_metric", "combined_score"),
            
                            convergence_threshold=workflow.parameters.get("convergence_threshold", 0.001),
            
                            random_seed=workflow.parameters.get("random_seed", 42),
            
                            diff_based_evolution=workflow.parameters.get("diff_based_evolution", True),
            
                            max_code_length=workflow.parameters.get("max_code_length", 10000),
            
                            diversity_metric=workflow.parameters.get("diversity_metric", "edit_distance"),
            
                            parallel_evaluations=workflow.parameters.get("parallel_evaluations", 1),
            
                            distributed=workflow.parameters.get("distributed", False),
            
                            template_dir=workflow.parameters.get("template_dir", None),
            
                            num_top_programs=workflow.parameters.get("num_top_programs", 3),
            
                            num_diverse_programs=workflow.parameters.get("num_diverse_programs", 2),
            
                            use_template_stochasticity=workflow.parameters.get("use_template_stochasticity", True),
            
                            template_variations=workflow.parameters.get("template_variations", {}),
            
                            use_meta_prompting=workflow.parameters.get("use_meta_prompting", False),
            
                            meta_prompt_weight=workflow.parameters.get("meta_prompt_weight", 0.1),
            
                            include_artifacts=workflow.parameters.get("include_artifacts", True),
            
                            max_artifact_bytes=workflow.parameters.get("max_artifact_bytes", 20 * 1024),
            
                            artifact_security_filter=workflow.parameters.get("artifact_security_filter", True),
            
                            memory_limit_mb=workflow.parameters.get("memory_limit_mb", None),
            
                            cpu_limit=workflow.parameters.get("cpu_limit", None),
            
                            db_path=workflow.parameters.get("db_path", None),
            
                            in_memory=workflow.parameters.get("in_memory", True),
            
                            log_level=workflow.parameters.get("log_level", "INFO"),
            
                            log_dir=workflow.parameters.get("log_dir", None),
            
                            api_timeout=60,
            
                            api_retries=3,
            
                            api_retry_delay=5,
            
                            artifact_size_threshold=32 * 1024,
            
                            cleanup_old_artifacts=True,
            
                            artifact_retention_days=30,
            
                            diversity_reference_size=20,
            
                            max_retries_eval=3,
            
                            evaluator_timeout=300,
            
                            evaluator_models=workflow.parameters.get("evaluator_models", None),
            
                            output_dir=output_dir, # Pass the output_dir
            
                            load_from_checkpoint=st.session_state.get("load_from_checkpoint", None), # Pass load_from_checkpoint
            
                            # Advanced research features
            
                            double_selection=workflow.parameters.get("double_selection", True),
            
                            adaptive_feature_dimensions=workflow.parameters.get("adaptive_feature_dimensions", True),
            
                            test_time_compute=workflow.parameters.get("test_time_compute", False),
            
                            optillm_integration=workflow.parameters.get("optillm_integration", False),
            
                            plugin_system=workflow.parameters.get("plugin_system", False),
            
                            hardware_optimization=workflow.parameters.get("hardware_optimization", False),
            
                            multi_strategy_sampling=workflow.parameters.get("multi_strategy_sampling", True),
            
                            ring_topology=workflow.parameters.get("ring_topology", True),
            
                            controlled_gene_flow=workflow.parameters.get("controlled_gene_flow", True),
            
                            auto_diff=workflow.parameters.get("auto_diff", True),
            
                            symbolic_execution=workflow.parameters.get("symbolic_execution", False),
            
                            coevolutionary_approach=workflow.parameters.get("coevolutionary_approach", False),
            
                        )
            
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring()
            
            # Clear the load_from_checkpoint flag after use
            if "load_from_checkpoint" in st.session_state:
                del st.session_state.load_from_checkpoint

            # Process results
            workflow.results = result or {}
            workflow.current_stage = WorkflowStage.ANALYSIS
            self._notify_callbacks(workflow_id, "stage_changed", workflow.current_stage)
            
            # Analysis stage
            if result and result.get("success"):
                workflow.metrics = result.get("metrics", {})
                workflow.progress = 1.0
                workflow.status = "completed"
                
                # Generate report
                workflow.current_stage = WorkflowStage.REPORTING
                self._notify_callbacks(workflow_id, "stage_changed", workflow.current_stage)
                
                try:
                    report = create_evolution_report(
                        run_id=workflow_id,
                        evolution_mode=workflow.workflow_type.value,
                        content_type=workflow.parameters.get("content_type", "code_python"),
                        parameters=workflow.parameters,
                        results=result,
                        metrics=result.get("metrics", {})
                    )
                    workflow.results["report"] = report
                except Exception as e:
                    print(f"Warning: Could not generate report: {e}")
            
            workflow.current_stage = WorkflowStage.COMPLETION
            workflow.end_time = time.time()
            self._notify_callbacks(workflow_id, "stage_changed", workflow.current_stage)
            self._notify_callbacks(workflow_id, "workflow_completed", workflow.results)
            
            # Add to history manager
            self.history_manager.add_workflow_to_history(workflow)

            # Remove from active workflows
            if workflow_id in self.active_workflows:
                self.active_workflows.remove(workflow_id)
                
        except Exception as e:
            workflow = self.workflows[workflow_id]
            workflow.status = "failed"
            workflow.end_time = time.time()
            workflow.results["error"] = str(e)
            self._notify_callbacks(workflow_id, "workflow_failed", str(e))
            
            # Add to history manager
            self.history_manager.add_workflow_to_history(workflow)

            if workflow_id in self.active_workflows:
                self.active_workflows.remove(workflow_id)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the current status and key metrics of a specified workflow.

        Args:
            workflow_id (str): The ID of the workflow to retrieve status for.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the workflow's status, or None if the workflow is not found.
        """
        if workflow_id not in self.workflows:
            return None
            
        workflow = self.workflows[workflow_id]
        return {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type.value,
            "current_stage": workflow.current_stage.value,
            "status": workflow.status,
            "progress": workflow.progress,
            "start_time": workflow.start_time,
            "end_time": workflow.end_time,
            "duration": (workflow.end_time or time.time()) - workflow.start_time
        }
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Retrieves the status of all currently active workflows.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing the status of an active workflow.
        """
        statuses = []
        for workflow_id in self.active_workflows:
            status = self.get_workflow_status(workflow_id)
            if status:
                statuses.append(status)
        return statuses
    
    def register_callback(self, workflow_id: str, callback: Callable):
        """Registers a callback function to be notified of events for a specific workflow.

        Args:
            workflow_id (str): The ID of the workflow to register the callback for.
            callback (Callable): The function to call when a workflow event occurs. It should accept (event: str, data: Any) as arguments.
        """
        if workflow_id not in self.workflow_callbacks:
            self.workflow_callbacks[workflow_id] = []
        self.workflow_callbacks[workflow_id].append(callback)
    
    def _notify_callbacks(self, workflow_id: str, event: str, data: Any):
        """Notifies all registered callbacks for a given workflow about a specific event.

        Args:
            workflow_id (str): The ID of the workflow whose callbacks should be notified.
            event (str): The name of the event that occurred (e.g., "stage_changed", "workflow_completed").
            data (Any): The data associated with the event.
        """
        if workflow_id in self.workflow_callbacks:
            for callback in self.workflow_callbacks[workflow_id]:
                try:
                    callback(event, data)
                except Exception as e:
                    print(f"Error in workflow callback: {e}")


def render_openevolve_orchestrator_ui():
    """Render the OpenEvolve orchestrator UI"""
    
    # Add custom CSS to maintain consistent styling and prevent UI breaking
    st.markdown("""
    <style>
    /* Ensure buttons maintain consistent styling */
    div[data-testid="stForm"] button[kind="secondary"],
    button[kind="secondary"] {
        background-color: #e0f2fe !important;
        border: 1px solid #808495 !important;
        color: var(--text-primary) !important;
    }
    
    div[data-testid="stForm"] button[kind="secondary"]:hover,
    button[kind="secondary"]:hover {
        background-color: #e0e2e6 !important;
        border: 1px solid #808495 !important;
    }
    
    div[data-testid="stForm"] button[kind="secondary"]:active,
    button[kind="secondary"]:active {
        background-color: #d0d2d6 !important;
    }
    
    /* Ensure text stays readable */
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stText"],
    p, span, div {
        color: var(--text-primary) !important;
    }
    
    /* Maintain consistent container styling */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
    .stContainer {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    /* Ensure form elements maintain consistent size */
    [data-testid="stForm"] {
        margin-bottom: 10px !important;
    }
    
    /* Override potential conflicting styles */
    [data-testid="stColumn"] button {
        min-width: unset !important;
        flex: unset !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("🤖 OpenEvolve Workflow Orchestrator")
    
    # Add description and usage information
    st.markdown("""
    **Purpose**: The OpenEvolve Workflow Orchestrator enables you to create, manage, and monitor complex evolution workflows with granular control over all parameters.
    
    **How to Use**:
    1. Navigate to the "Create Workflow" tab to define your evolution task
    2. Select a workflow type and configure parameters
    3. Monitor active workflows in the "Monitoring Panel" 
    4. View historical runs in the "History" tab
    5. Manage templates and configurations in the "Configuration" tab
    
    **Why This Matters**: Advanced evolution workflows often require fine-tuned parameters to achieve optimal results. The orchestrator provides a unified interface to control all aspects of the evolution process.
    """)
    
    if not ORCHESTRATOR_AVAILABLE:
        st.warning("OpenEvolve orchestrator components are not available. Please install the required packages.")
        return
    
    # Initialize orchestrator in session state
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = OpenEvolveOrchestrator()
    
    orchestrator = st.session_state.orchestrator
    
    # Main tabs
    tabs = st.tabs(["Create Workflow", "Monitoring Panel", "History", "Configuration"])
    
    with tabs[0]:  # Create Workflow
        render_create_workflow_tab(orchestrator)
    
    with tabs[1]:  # Monitoring
        render_monitoring_tab(orchestrator)
    
    with tabs[2]:  # History
        render_history_tab(orchestrator)
    
    with tabs[3]:  # Configuration
        render_configuration_tab(orchestrator)


def render_create_workflow_tab(orchestrator: OpenEvolveOrchestrator):
    """Renders the 'Create Workflow' tab in the Streamlit UI, allowing users to configure and start new evolution workflows.

    Args:
        orchestrator (OpenEvolveOrchestrator): The orchestrator instance to interact with.
    """
    st.subheader("Create Workflow")
    
    # Initialize managers
    team_manager = TeamManager()
    gauntlet_manager = GauntletManager()
    
    # Add usage information
    st.info("""
    **Purpose**: Create a new evolution workflow with detailed parameter configuration.
    
    **How to Use**:
    1. Select a workflow type from the dropdown (each type optimizes for different objectives)
    2. Enter the content you want to evolve
    3. Configure parameters using the expandable sections below
    4. Click 'Start Workflow' to begin the evolution process
    
    **Why This Matters**: Different evolution tasks require different configurations. The orchestrator allows you to fine-tune every aspect of the evolution process to match your specific needs.
    """)
    
    # Workflow type selection with tooltips and dynamic description
    workflow_type = st.selectbox(
        "Select Workflow Type",
        options=[wt.value for wt in EvolutionWorkflow],
        format_func=lambda x: {
            "standard": "🧬 Standard Evolution",
            "quality_diversity": "🎯 Quality-Diversity Evolution (MAP-Elites)",
            "multi_objective": "⚖️ Multi-Objective Optimization",
            "adversarial": "⚔️ Adversarial Evolution (Red Team/Blue Team)",
            "symbolic_regression": "🔍 Symbolic Regression",
            "neuroevolution": "🧠 Neuroevolution",
            "algorithm_discovery": "💡 Algorithm Discovery",
            "prompt_optimization": "📝 Prompt Optimization",
            "problem_decomposition": "🧩 Problem Decomposition",
            "sovereign_decomposition": "👑 Sovereign-Grade Decomposition"
        }.get(x, x),
        help="Workflow types define the optimization strategy: Standard (single objective), Quality-Diversity (diverse high-quality solutions), Multi-Objective (multiple competing goals), Adversarial (red team/blue team approach), etc."
    )
    
    # Add description for the selected workflow type
    workflow_descriptions = {
        "standard": "**Standard Evolution** - Optimizes a single objective function to find the best solution. Best for problems with a clear, quantifiable goal.",
        "quality_diversity": "**Quality-Diversity Evolution** - Creates a diverse set of high-quality solutions across multiple dimensions. Best for exploring solution spaces and finding multiple viable approaches.",
        "multi_objective": "**Multi-Objective Optimization** - Balances multiple competing objectives simultaneously to find the Pareto frontier of solutions. Best for problems with trade-offs between different goals.",
        "adversarial": "**Adversarial Evolution** - Uses red team/blue team approach where models compete against each other to create more robust solutions. Best for security testing and hardening.",
        "symbolic_regression": "**Symbolic Regression** - Evolves mathematical expressions to fit data or solve problems. Best for discovering mathematical relationships and formulas.",
        "neuroevolution": "**Neuroevolution** - Evolves neural network architectures and parameters. Best for machine learning model optimization.",
        "algorithm_discovery": "**Algorithm Discovery** - Discovers novel algorithms and computational approaches. Best for finding innovative problem-solving strategies.",
        "prompt_optimization": "**Prompt Optimization** - Evolves text prompts to optimize AI model performance. Best for improving LLM outputs and interactions.",
        "problem_decomposition": "**Problem Decomposition** - Breaks down a complex problem into smaller components, solves them individually, and then reassembles the solution. Best for intractable problems.",
        "sovereign_decomposition": "**Sovereign-Grade Decomposition** - The ultimate workflow for intractable problems. Features AI-assisted decomposition, manual override, multi-team gauntlets (Blue, Red, Gold), and a self-healing loop for robust, verified solutions."
    }
    
    st.markdown(f"*{workflow_descriptions.get(workflow_type, 'Select a workflow type to see its description')}*")
    
    # Content input with detailed information
    content = st.text_area(
        "Input Content",
        height=200,
        placeholder="Enter content to evolve here...",
        help="Enter the content that you want to evolve. This could be code, text, configurations, or any text-based content that can be improved through evolution."
    )
    
    # Content type selection
    content_type = st.selectbox(
        "Content Type",
        options=[
            "code_python", "code_javascript", "code_java", "code_csharp", "code_cpp", "code_go", "code_rust", "code_ruby", "code_php", "code_swift", "code_kotlin", "code_typescript", "code_sql", "code_shell", "code_assembly", "code_c", "code_clojure", "code_cobol", "code_d", "code_dart", "code_elixir", "code_erlang", "code_fsharp", "code_fortran", "code_groovy", "code_haskell", "code_julia", "code_lisp", "code_lua", "code_matlab", "code_objective_c", "code_ocaml", "code_pascal", "code_perl", "code_powershell", "code_prolog", "code_r", "code_scala", "code_scheme", "code_vbnet",
            "web_html", "web_css", "web_javascript", "web_typescript", "web_json", "web_xml", "web_graphql",
            "devops_dockerfile", "devops_kubernetes", "devops_terraform", "devops_ansible", "devops_chef", "devops_puppet", "devops_jenkinsfile", "devops_gitlab_ci", "devops_github_actions",
            "ml_jupyter_notebook", "ml_python_script", "ml_r_script", "ml_sql_query", "ml_tensorboard_log",
            "doc_plaintext", "doc_markdown", "doc_latex", "doc_rst", "doc_asciidoc", "doc_word", "doc_pdf", "doc_powerpoint", "doc_excel",
            "other_shader", "other_game_script", "other_smart_contract",
            "text_general", "text_markdown", "text_json", "text_yaml", "text_xml", "text_html", "text_css",
            "document_legal", "document_medical", "document_financial", "document_technical", "document_scientific", "document_creative", "document_business", "document_marketing", "document_educational", "document_conversational",
            "data_csv", "data_json", "data_xml", "data_parquet",
            "config_yaml", "config_json", "config_toml", "config_ini",
            "prompt", "sop", "policy", "procedure", "plan", "protocol", "documentation"
        ],
        format_func=lambda x: {
            "code_python": "🐍 Python Code",
            "code_javascript": "🌐 JavaScript Code",
            "code_java": "☕ Java Code", 
            "code_csharp": "# C# Code",
            "code_cpp": "++ C++ Code",
            "code_go": "🐹 Go Code",
            "code_rust": "🦀 Rust Code",
            "code_ruby": "💎 Ruby Code",
            "code_php": "🐘 PHP Code",
            "code_swift": "🐦 Swift Code",
            "code_kotlin": "🤖 Kotlin Code",
            "code_typescript": "📜 TypeScript Code",
            "code_sql": "💾 SQL Query",
            "code_shell": "💲 Shell Script",
            "code_assembly": "🔧 Assembly Code",
            "code_c": "🔧 C Code",
            "code_clojure": "🌀 Clojure Code",
            "code_cobol": "💼 COBOL Code",
            "code_d": " D Code",
            "code_dart": "🎯 Dart Code",
            "code_elixir": "💧 Elixir Code",
            "code_erlang": "📞 Erlang Code",
            "code_fsharp": "# F# Code",
            "code_fortran": "🔢 Fortran Code",
            "code_groovy": "🎶 Groovy Code",
            "code_haskell": "🧮 Haskell Code",
            "code_julia": "📈 Julia Code",
            "code_lisp": "🧠 Lisp Code",
            "code_lua": "🌙 Lua Code",
            "code_matlab": "🔢 MATLAB Code",
            "code_objective_c": "🍏 Objective-C Code",
            "code_ocaml": "🐫 OCaml Code",
            "code_pascal": "📝 Pascal Code",
            "code_perl": "🐪 Perl Code",
            "code_powershell": "💲 PowerShell Script",
            "code_prolog": "🧠 Prolog Code",
            "code_r": "📊 R Code",
            "code_scala": " Scala Code",
            "code_scheme": "🌀 Scheme Code",
            "code_vbnet": " VB.NET Code",
            "web_html": "🌐 HTML",
            "web_css": "🎨 CSS",
            "web_javascript": "🌐 JavaScript",
            "web_typescript": "📜 TypeScript",
            "web_json": "🕸️ JSON",
            "web_xml": "🔖 XML",
            "web_graphql": "🕸️ GraphQL",
            "devops_dockerfile": "🐳 Dockerfile",
            "devops_kubernetes": "☸️ Kubernetes YAML",
            "devops_terraform": "🏗️ Terraform",
            "devops_ansible": "📜 Ansible",
            "devops_chef": "📜 Chef",
            "devops_puppet": "📜 Puppet",
            "devops_jenkinsfile": "📜 Jenkinsfile",
            "devops_gitlab_ci": "🦊 GitLab CI",
            "devops_github_actions": "🐙 GitHub Actions",
            "ml_jupyter_notebook": "📓 Jupyter Notebook",
            "ml_python_script": "🐍 Python Script",
            "ml_r_script": "📊 R Script",
            "ml_sql_query": "💾 SQL Query",
            "ml_tensorboard_log": "📈 TensorBoard Log",
            "doc_plaintext": "📝 Plain Text",
            "doc_markdown": "📄 Markdown",
            "doc_latex": "📜 LaTeX",
            "doc_rst": "📄 reStructuredText",
            "doc_asciidoc": "📄 AsciiDoc",
            "doc_word": "📄 Word Document",
            "doc_pdf": "📄 PDF Document",
            "doc_powerpoint": "📊 PowerPoint",
            "doc_excel": "📊 Excel Spreadsheet",
            "other_shader": "🎨 Shader Code",
            "other_game_script": "🎮 Game Script",
            "other_smart_contract": "🔗 Smart Contract",
            "text_general": "📝 General Text",
            "document_legal": "⚖️ Legal Document",
            "document_medical": "⚕️ Medical Document",
            "document_financial": "💰 Financial Document",
            "document_technical": "🔧 Technical Document",
            "document_scientific": "🔬 Scientific Paper",
            "document_creative": "🎨 Creative Writing",
            "document_business": "📈 Business Plan",
            "document_marketing": "📢 Marketing Copy",
            "document_educational": "🎓 Educational Material",
            "document_conversational": "💬 Conversational AI",
            "data_csv": "📊 CSV Data",
            "data_json": "📊 JSON Data",
            "data_xml": "📊 XML Data",
            "data_parquet": "📊 Parquet Data",
            "config_yaml": "⚙️ YAML Config",
            "config_json": "⚙️ JSON Config",
            "config_toml": "⚙️ TOML Config",
            "config_ini": "⚙️ INI Config",
            "prompt": "💡 LLM Prompt",
            "sop": "📋 Standard Operating Procedure",
            "policy": "📜 Policy Document",
            "procedure": "📄 Procedure Document",
            "plan": "📝 Plan Document",
            "protocol": "📋 Protocol",
            "documentation": "📚 Documentation"
        }.get(x, x),
        help="Select the type of content being evolved. This helps the system apply appropriate evaluation and evolution strategies."
    )

    # Global Model Settings
    with st.expander("🧠 Global Model Settings", expanded=False):
        st.markdown("**Purpose**: Configure the default AI model and its generation parameters for non-Sovereign workflows.")
        st.session_state.setdefault("model", "gpt-4o")
        st.session_state.setdefault("api_key", "")
        st.session_state.setdefault("base_url", "https://api.openai.com/v1")
        st.session_state.setdefault("temperature", 0.7)
        st.session_state.setdefault("top_p", 0.95)
        st.session_state.setdefault("max_tokens", 4096)
        st.session_state.setdefault("frequency_penalty", 0.0)
        st.session_state.setdefault("presence_penalty", 0.0)
        st.session_state.setdefault("seed", 42)
        st.session_state.setdefault("stop_sequences", "")
        st.session_state.setdefault("logprobs", False)
        st.session_state.setdefault("top_logprobs", 0)
        st.session_state.setdefault("response_format", "")
        st.session_state.setdefault("stream", False)
        st.session_state.setdefault("user", "")
        st.session_state.setdefault("system_prompt", "")
        st.session_state.setdefault("evaluator_system_prompt", "")

        st.session_state.model = st.text_input("Model ID", value=st.session_state.model, key="global_model_id")
        st.session_state.api_key = st.text_input("API Key", type="password", value=st.session_state.api_key, key="global_api_key")
        st.session_state.base_url = st.text_input("API Base URL", value=st.session_state.base_url, key="global_base_url")
        st.session_state.temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=st.session_state.temperature, step=0.1, key="global_temperature")
        st.session_state.top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=st.session_state.top_p, step=0.01, key="global_top_p")
        st.session_state.max_tokens = st.number_input("Max Tokens", min_value=1, value=st.session_state.max_tokens, key="global_max_tokens")
        st.session_state.frequency_penalty = st.slider("Frequency Penalty", min_value=-2.0, max_value=2.0, value=st.session_state.frequency_penalty, step=0.01, key="global_frequency_penalty")
        st.session_state.presence_penalty = st.slider("Presence Penalty", min_value=-2.0, max_value=2.0, value=st.session_state.presence_penalty, step=0.01, key="global_presence_penalty")
        st.session_state.seed = st.number_input("Seed (Optional)", value=st.session_state.seed, key="global_seed")
        st.session_state.stop_sequences = st.text_input("Stop Sequences (comma-separated)", value=st.session_state.stop_sequences, key="global_stop_sequences")
        st.session_state.logprobs = st.checkbox("Logprobs", value=st.session_state.logprobs, key="global_logprobs")
        st.session_state.top_logprobs = st.number_input("Top Logprobs (0-5)", min_value=0, max_value=5, value=st.session_state.top_logprobs, key="global_top_logprobs")
        st.session_state.response_format = st.text_input("Response Format (JSON string, e.g., '{\"type\": \"json_object\"}')", value=st.session_state.response_format, key="global_response_format")
        st.session_state.stream = st.checkbox("Stream", value=st.session_state.stream, key="global_stream")
        st.session_state.user = st.text_input("User ID", value=st.session_state.user, key="global_user")
        st.session_state.system_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt, height=100, help="Initial system message for the AI model.")
        st.session_state.evaluator_system_prompt = st.text_area("Evaluator System Prompt", value=st.session_state.evaluator_system_prompt, height=100, help="System message for the evaluator AI model.")

    # Core Configuration with more detailed options and tooltips
    with st.expander("🎯 Core Evolution Parameters", expanded=True):
        st.markdown("**Purpose**: These parameters control the fundamental behavior of the evolution process.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_iterations = st.number_input(
                "Max Iterations", 
                min_value=1, 
                max_value=10000, 
                value=100,
                help="Maximum number of evolution iterations to run. Higher values allow more thorough exploration but take longer."
            )
            population_size = st.number_input(
                "Population Size", 
                min_value=10, 
                max_value=10000, 
                value=100,
                help="Number of individuals in each generation. Larger populations provide more diversity but require more computation."
            )
            num_islands = st.number_input(
                "Number of Islands", 
                min_value=1, 
                max_value=20, 
                value=5,
                help="Number of isolated populations (islands) that evolve separately with occasional migration. This maintains diversity."
            )
            archive_size = st.number_input(
                "Archive Size", 
                min_value=10, 
                max_value=10000, 
                value=100,
                help="Size of the archive that stores high-quality solutions found during evolution. Important for Quality-Diversity algorithms."
            )
        
        with col2:
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.7, 
                step=0.1,
                help="Controls randomness in selection. Lower values favor exploitation of high-performing solutions, higher values encourage exploration."
            )
            elite_ratio = st.slider(
                "Elite Ratio", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.1, 
                step=0.01,
                help="Proportion of top-performing individuals that automatically survive to the next generation without modification."
            )
            exploration_ratio = st.slider(
                "Exploration Ratio", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.2, 
                step=0.01,
                help="Proportion of mutations that focus on exploring new areas of the solution space."
            )
            exploitation_ratio = st.slider(
                "Exploitation Ratio", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7, 
                step=0.01,
                help="Proportion of mutations that focus on improving existing good solutions."
            )
    
    # Feature dimensions for QD and multi-objective with explanations
    with st.expander("📏 Feature Dimensions (for Quality-Diversity & Multi-Objective)", expanded=False):
        st.markdown("""
        **Purpose**: Define the dimensions used to measure diversity in Quality-Diversity algorithms.
        
        **How to Use**: Select the characteristics that define diversity in your domain. For example, for code evolution: complexity and efficiency; for text: readability and creativity.
        """)
        
        if workflow_type in ["quality_diversity", "multi_objective"]:
            feature_dimensions = st.multiselect(
                "Feature Dimensions",
                options=["complexity", "diversity", "performance", "readability", "efficiency", "accuracy", "robustness", "maintainability", "scalability", "resource_usage"],
                default=["complexity", "diversity"],
                help="Select dimensions that define diverse, high-quality solutions. These metrics will be used to measure and preserve diversity in the population."
            )
        else:
            feature_dimensions = ["complexity", "diversity"]
            st.write(f"Feature dimensions will be used based on selected workflow type: {feature_dimensions}")
    
    # Advanced features with detailed explanations
    with st.expander("⚙️ Advanced Evolution Features", expanded=False):
        st.markdown("""
        **Purpose**: Fine-tune advanced evolution strategies for complex optimization scenarios.
        
        **How to Use**: Enable features based on your specific requirements. For example, enable artifact feedback for code generation tasks, cascade evaluation for faster processing, or LLM feedback for complex evaluation criteria.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_artifacts = st.checkbox(
                "Enable Artifact Feedback", 
                value=True,
                help="Enables analysis of generated artifacts (like test outputs, execution results) to guide evolution."
            )
            cascade_evaluation = st.checkbox(
                "Enable Cascade Evaluation", 
                value=True,
                help="Uses multiple evaluation stages to quickly filter out poor solutions and reduce computational cost."
            )
            use_llm_feedback = st.checkbox(
                "Use LLM Feedback", 
                value=False,
                help="Incorporates feedback from Large Language Models for complex evaluation criteria that are hard to quantify."
            )
            evolution_trace_enabled = st.checkbox(
                "Enable Evolution Tracing", 
                value=False,
                help="Records the complete evolution process for analysis, debugging, and replay capabilities."
            )
        
        with col2:
            double_selection = st.checkbox(
                "Double Selection", 
                value=True,
                help="Applies selection pressure in both parent selection and survival selection phases."
            )
            adaptive_feature_dimensions = st.checkbox(
                "Adaptive Feature Dimensions", 
                value=True,
                help="Dynamically adjusts feature dimensions during evolution based on the current population state."
            )
            multi_strategy_sampling = st.checkbox(
                "Multi-Strategy Sampling", 
                value=True,
                help="Uses multiple mutation strategies simultaneously, selecting the most effective ones based on performance."
            )
            ring_topology = st.checkbox(
                "Ring Topology", 
                value=True,
                help="Organizes islands in a ring structure where each island only exchanges individuals with its neighbors."
            )
    
    # Research-grade features with explanations
    with st.expander("🔬 Research-Grade Features", expanded=False):
        st.markdown("""
        **Purpose**: Enable cutting-edge evolution techniques for advanced research and experimentation.
        
        **How to Use**: These features provide access to state-of-the-art evolution techniques. They may require more computational resources but can significantly improve results in complex domains.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_time_compute = st.checkbox(
                "Test-Time Compute", 
                value=False,
                help="Allows computation during evaluation that's not part of the final solution (useful for testing complex behaviors)."
            )
            optillm_integration = st.checkbox(
                "OptiLLM Integration", 
                value=False,
                help="Integrates with OptiLLM for optimized LLM usage and cost management."
            )
            plugin_system = st.checkbox(
                "Plugin System", 
                value=False,
                help="Enables loading external plugins for custom evolution strategies and evaluation functions."
            )
            hardware_optimization = st.checkbox(
                "Hardware Optimization", 
                value=False,
                help="Optimizes computation for specific hardware configurations to maximize performance."
            )
        
        with col2:
            controlled_gene_flow = st.checkbox(
                "Controlled Gene Flow", 
                value=True,
                help="Controls the flow of genetic information between populations to maintain diversity while allowing beneficial traits to spread."
            )
            auto_diff = st.checkbox(
                "Auto Diff", 
                value=True,
                help="Automatically computes differences between solutions to guide targeted mutations."
            )
            symbolic_execution = st.checkbox(
                "Symbolic Execution", 
                value=False,
                help="Uses symbolic execution to analyze program behavior without running it on concrete inputs."
            )
            coevolutionary_approach = st.checkbox(
                "Coevolutionary Approach", 
                value=False,
                help="Evolves multiple populations that influence each other's fitness, useful for competitive or collaborative scenarios."
            )
    
    # Performance optimization with explanations
    with st.expander("⚡ Performance Optimization", expanded=False):
        st.markdown("""
        **Purpose**: Configure performance parameters to optimize resource usage and processing speed.
        
        **How to Use**: Adjust these settings based on your available computational resources to maintain optimal performance without resource exhaustion.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            memory_limit_mb = st.number_input(
                "Memory Limit (MB)", 
                min_value=100, 
                max_value=32768, 
                value=2048,
                help="Maximum memory usage for the evolution process. Exceeding this limit may cause performance issues."
            )
            cpu_limit = st.number_input(
                "CPU Limit", 
                min_value=0.1, 
                max_value=32.0, 
                value=4.0, 
                step=0.1,
                help="Maximum CPU cores to use for evolution. Higher values speed up parallel operations."
            )
            parallel_evaluations = st.number_input(
                "Parallel Evaluations", 
                min_value=1, 
                max_value=32, 
                value=4,
                help="Number of solutions to evaluate simultaneously. Higher values speed up evaluation but require more resources."
            )
        
        with col2:
            max_code_length = st.number_input(
                "Max Code Length", 
                min_value=100, 
                max_value=100000, 
                value=10000,
                help="Maximum length of generated code before it's truncated. Prevents generation of excessively long solutions."
            )
            evaluator_timeout = st.number_input(
                "Evaluator Timeout (s)", 
                min_value=10, 
                max_value=3600, 
                value=300,
                help="Maximum time allowed for a single evaluation before it's considered failed."
            )
            max_retries_eval = st.number_input(
                "Max Evaluation Retries", 
                min_value=1, 
                max_value=10, 
                value=3,
                help="Number of times to retry failed evaluations before giving up on that solution."
            )
    
    # Advanced evolution trace settings with explanations
    with st.expander("📝 Evolution Trace Settings", expanded=False):
        st.markdown("""
        **Purpose**: Configure detailed logging and tracing of the evolution process for analysis and debugging.
        
        **How to Use**: Enable tracing if you need to analyze the evolution process in detail, reproduce results, or debug evolution behavior.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            evolution_trace_format = st.selectbox(
                "Trace Format", 
                options=["jsonl", "csv", "parquet"], 
                index=0,
                help="Format to store evolution traces. JSONL offers flexibility, CSV offers compatibility, Parquet offers efficiency."
            )
            evolution_trace_include_code = st.checkbox(
                "Include Code in Traces", 
                value=False,
                help="Store the actual code/text of solutions in the trace (increases file size)."
            )
            evolution_trace_include_prompts = st.checkbox(
                "Include Prompts in Traces", 
                value=True,
                help="Store the prompts used in each generation in the trace."
            )
        
        with col2:
            evolution_trace_buffer_size = st.number_input(
                "Trace Buffer Size", 
                min_value=1, 
                max_value=100, 
                value=10,
                help="Number of trace entries to buffer before writing to disk."
            )
            evolution_trace_compress = st.checkbox(
                "Compress Traces", 
                value=False,
                help="Compress trace files to save disk space."
            )
    
    # --- Sovereign-Grade Decomposition Workflow Configuration ---
    if workflow_type == "sovereign_decomposition":
        st.subheader("👑 Sovereign-Grade Workflow Configuration")
        st.info("Configure the specialized Teams and Gauntlets for each stage of the Sovereign-Grade Decomposition Workflow.")

        # Get available teams and gauntlets from managers.
        team_manager = TeamManager()
        gauntlet_manager = GauntletManager()

        # Filter teams by role for appropriate selection.
        blue_teams = [t.name for t in team_manager.get_all_teams() if t.role == "Blue"]
        red_teams = [t.name for t in team_manager.get_all_teams() if t.role == "Red"]
        gold_teams = [t.name for t in team_manager.get_all_teams() if t.role == "Gold"]

        # Filter gauntlets by the role of the team they are run by.
        # This ensures only relevant gauntlets are presented for selection.
        blue_gauntlets = [g.name for g in gauntlet_manager.get_all_gauntlets() if gauntlet_manager.get_gauntlet(g.name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Blue"]
        red_gauntlets = [g.name for g in gauntlet_manager.get_all_gauntlets() if gauntlet_manager.get_gauntlet(g.name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Red"]
        gold_gauntlets = [g.name for g in gauntlet_manager.get_all_gauntlets() if gauntlet_manager.get_gauntlet(g.name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Gold"]

        # Stage 0: Content Analysis - Select the Blue Team responsible for initial problem understanding.
        st.markdown("#### Stage 0: Content Analysis")
        content_analyzer_team_name = st.selectbox("Content Analyzer Team (Blue)", blue_teams, key="sg_content_analyzer_team")

        # Stage 1: AI-Assisted Decomposition - Select the Blue Team responsible for breaking down the problem.
        st.markdown("#### Stage 1: AI-Assisted Decomposition")
        planner_team_name = st.selectbox("Planner Team (Blue)", blue_teams, key="sg_planner_team")

        # Stage 3: Sub-Problem Solving Loop - Configure teams and gauntlets for individual sub-problem resolution.
        st.markdown("#### Stage 3: Sub-Problem Solving Loop")
        solver_team_name = st.selectbox("Solver Team (Blue)", blue_teams, key="sg_solver_team", help="Team responsible for generating initial solutions for sub-problems.")
        patcher_team_name = st.selectbox("Patcher Team (Blue)", blue_teams, key="sg_patcher_team", help="Team responsible for modifying solutions based on critique/verification reports.")
        solver_generation_gauntlet_name = st.selectbox("Solver Generation Gauntlet (Blue)", blue_gauntlets, key="sg_solver_generation_gauntlet", help="Blue Team Gauntlet defining how solvers generate solutions (e.g., single candidate, multi-candidate peer review).")
        sub_problem_red_gauntlet_name = st.selectbox("Sub-Problem Red Team Gauntlet", red_gauntlets, key="sg_sub_red_gauntlet", help="Red Team Gauntlet to critique individual sub-problem solutions.")
        sub_problem_gold_gauntlet_name = st.selectbox("Sub-Problem Gold Team Gauntlet", gold_gauntlets, key="sg_sub_gold_gauntlet", help="Gold Team Gauntlet to verify individual sub-problem solutions.")

        # Stage 4: Configurable Reassembly - Select the Blue Team for integrating verified sub-solutions.
        st.markdown("#### Stage 4: Configurable Reassembly")
        assembler_team_name = st.selectbox("Assembler Team (Blue)", blue_teams, key="sg_assembler_team", help="Team responsible for combining all verified sub-problem solutions into a final product.")

        # Stage 5: Final Verification & Self-Healing Loop - Configure gauntlets for overall solution validation.
        st.markdown("#### Stage 5: Final Verification & Self-Healing Loop")
        final_red_gauntlet_name = st.selectbox("Final Red Team Gauntlet", red_gauntlets, key="sg_final_red_gauntlet", help="Red Team Gauntlet to perform a final adversarial attack on the assembled solution.")
        final_gold_gauntlet_name = st.selectbox("Final Gold Team Gauntlet", gold_gauntlets, key="sg_final_gold_gauntlet", help="Gold Team Gauntlet to perform a holistic evaluation of the final assembled solution.")
                diversity_metric = st.selectbox("Diversity Metric per Sub-Problem", options=["edit_distance", "cosine_similarity", "jaccard_index"], index=0, key="sg_diversity_metric")

        with st.expander("⚙️ Advanced Evaluation Parameters for Sub-Problems", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                enable_artifacts = st.checkbox("Enable Artifact Feedback", value=True, key="sg_enable_artifacts")
                cascade_evaluation = st.checkbox("Enable Cascade Evaluation", value=True, key="sg_cascade_evaluation")
                cascade_thresholds_str = st.text_input("Cascade Thresholds (comma-separated floats)", value="0.5, 0.75, 0.9", key="sg_cascade_thresholds")
                use_llm_feedback = st.checkbox("Use LLM Feedback", value=False, key="sg_use_llm_feedback")
                llm_feedback_weight = st.slider("LLM Feedback Weight", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="sg_llm_feedback_weight")
                parallel_evaluations = st.number_input("Parallel Evaluations", min_value=1, value=4, key="sg_parallel_evaluations")
                distributed = st.checkbox("Distributed Evaluation", value=False, key="sg_distributed")
                num_top_programs = st.number_input("Num Top Programs for Prompt", min_value=1, value=3, key="sg_num_top_programs")
                num_diverse_programs = st.number_input("Num Diverse Programs for Prompt", min_value=1, value=2, key="sg_num_diverse_programs")
                use_template_stochasticity = st.checkbox("Use Template Stochasticity", value=True, key="sg_use_template_stochasticity")
                include_artifacts = st.checkbox("Include Artifacts in Prompts", value=True, key="sg_include_artifacts")
                max_artifact_bytes = st.number_input("Max Artifact Bytes", min_value=1024, value=20 * 1024, key="sg_max_artifact_bytes")
                artifact_security_filter = st.checkbox("Artifact Security Filter", value=True, key="sg_artifact_security_filter")
                early_stopping_patience = st.number_input("Early Stopping Patience (0 for none)", min_value=0, value=0, key="sg_early_stopping_patience")
                convergence_threshold = st.number_input("Convergence Threshold", min_value=0.0, value=0.001, format="%.3f", key="sg_convergence_threshold")
                early_stopping_metric = st.selectbox("Early Stopping Metric", options=["combined_score", "fitness", "diversity"], key="sg_early_stopping_metric")
            with col2:
                memory_limit_mb = st.number_input("Memory Limit (MB)", min_value=100, value=2048, key="sg_memory_limit_mb")
                cpu_limit = st.number_input("CPU Limit", min_value=0.1, value=4.0, step=0.1, key="sg_cpu_limit")
                random_seed = st.number_input("Random Seed", value=42, key="sg_random_seed")
                db_path = st.text_input("Database Path (leave empty for default)", value="", key="sg_db_path")
                in_memory = st.checkbox("In-Memory Database", value=True, key="sg_in_memory")
                template_dir = st.text_input("Template Directory (leave empty for default)", value="", key="sg_template_dir")
                template_variations_str = st.text_area("Template Variations (JSON string)", value="{}", key="sg_template_variations")
                use_meta_prompting = st.checkbox("Use Meta-Prompting", value=False, key="sg_use_meta_prompting")
                meta_prompt_weight = st.slider("Meta-Prompt Weight", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="sg_meta_prompt_weight")
                artifact_size_threshold = st.number_input("Artifact Size Threshold", min_value=1024, value=32 * 1024, key="sg_artifact_size_threshold")
                cleanup_old_artifacts = st.checkbox("Cleanup Old Artifacts", value=True, key="sg_cleanup_old_artifacts")
                artifact_retention_days = st.number_input("Artifact Retention Days", min_value=1, value=30, key="sg_artifact_retention_days")
                diversity_reference_size = st.number_input("Diversity Reference Size", min_value=1, value=20, key="sg_diversity_reference_size")
                max_retries_eval = st.number_input("Max Evaluation Retries", min_value=1, value=3, key="sg_max_retries_eval")
                evaluator_timeout = st.number_input("Evaluator Timeout (s)", min_value=10, value=300, key="sg_evaluator_timeout")

        with st.expander("⚙️ Advanced OpenEvolve Parameters for Sub-Problems", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                diff_based_evolution = st.checkbox("Diff-Based Evolution", value=True, key="sg_diff_based_evolution")
                max_code_length = st.number_input("Max Code Length", min_value=100, value=10000, key="sg_max_code_length")
                evolution_trace_enabled = st.checkbox("Enable Evolution Tracing", value=False, key="sg_evolution_trace_enabled")
                evolution_trace_format = st.selectbox("Evolution Trace Format", options=["jsonl", "csv", "parquet"], key="sg_evolution_trace_format")
                evolution_trace_include_code = st.checkbox("Include Code in Traces", value=False, key="sg_evolution_trace_include_code")
                evolution_trace_include_prompts = st.checkbox("Include Prompts in Traces", value=True, key="sg_evolution_trace_include_prompts")
            with col2:
                evolution_trace_output_path = st.text_input("Evolution Trace Output Path (leave empty for default)", value="", key="sg_evolution_trace_output_path")
                evolution_trace_buffer_size = st.number_input("Evolution Trace Buffer Size", min_value=1, value=10, key="sg_evolution_trace_buffer_size")
                evolution_trace_compress = st.checkbox("Compress Traces", value=False, key="sg_evolution_trace_compress")
                log_level = st.selectbox("Log Level", options=["DEBUG", "INFO", "WARNING", "ERROR"], index=1, key="sg_log_level")
                log_dir = st.text_input("Log Directory (leave empty for default)", value="", key="sg_log_dir")
                api_timeout = st.number_input("API Timeout (s)", min_value=10, value=60, key="sg_api_timeout")
                api_retries = st.number_input("API Retries", min_value=0, value=3, key="sg_api_retries")
                api_retry_delay = st.number_input("API Retry Delay (s)", min_value=1, value=5, key="sg_api_retry_delay")

        with st.expander("🔬 Research-Grade Features for Sub-Problems", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                double_selection = st.checkbox("Double Selection", value=True, key="sg_double_selection")
                adaptive_feature_dimensions = st.checkbox("Adaptive Feature Dimensions", value=True, key="sg_adaptive_feature_dimensions")
                test_time_compute = st.checkbox("Test-Time Compute", value=False, key="sg_test_time_compute")
                optillm_integration = st.checkbox("OptiLLM Integration", value=False, key="sg_optillm_integration")
                plugin_system = st.checkbox("Plugin System", value=False, key="sg_plugin_system")
                hardware_optimization = st.checkbox("Hardware Optimization", value=False, key="sg_hardware_optimization")
            with col2:
                multi_strategy_sampling = st.checkbox("Multi-Strategy Sampling", value=True, key="sg_multi_strategy_sampling")
                ring_topology = st.checkbox("Ring Topology", value=True, key="sg_ring_topology")
                controlled_gene_flow = st.checkbox("Controlled Gene Flow", value=True, key="sg_controlled_gene_flow")
                auto_diff = st.checkbox("Auto Diff", value=True, key="sg_auto_diff")
                symbolic_execution = st.checkbox("Symbolic Execution", value=False, key="sg_symbolic_execution")
                coevolutionary_approach = st.checkbox("Coevolutionary Approach", value=False, key="sg_coevolutionary_approach")

        with st.expander("⚙️ Core Evolution Parameters for Sub-Problems", expanded=False):
            st.markdown("These parameters apply to the OpenEvolve runs for individual sub-problems.")
            col1, col2 = st.columns(2)
            with col1:
                max_iterations = st.number_input("Max Iterations per Sub-Problem", min_value=1, value=100, key="sg_max_iterations")
                population_size = st.number_input("Population Size per Sub-Problem", min_value=10, value=100, key="sg_population_size")
                num_islands = st.number_input("Number of Islands per Sub-Problem", min_value=1, value=5, key="sg_num_islands")
                migration_interval = st.number_input("Migration Interval per Sub-Problem", min_value=1, value=50, key="sg_migration_interval")
                migration_rate = st.slider("Migration Rate per Sub-Problem", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="sg_migration_rate")
                archive_size = st.number_input("Archive Size per Sub-Problem", min_value=10, value=100, key="sg_archive_size")
            with col2:
                elite_ratio = st.slider("Elite Ratio per Sub-Problem", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="sg_elite_ratio")
                exploration_ratio = st.slider("Exploration Ratio per Sub-Problem", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="sg_exploration_ratio")
                exploitation_ratio = st.slider("Exploitation Ratio per Sub-Problem", min_value=0.0, max_value=1.0, value=0.7, step=0.01, key="sg_exploitation_ratio")
                checkpoint_interval = st.number_input("Checkpoint Interval per Sub-Problem", min_value=1, value=100, key="sg_checkpoint_interval")
                feature_dimensions = st.multiselect("Feature Dimensions per Sub-Problem", options=["complexity", "diversity", "performance", "readability", "efficiency", "accuracy", "robustness", "maintainability", "scalability", "resource_usage"], default=["complexity", "diversity"], key="sg_feature_dimensions")
                feature_bins = st.number_input("Feature Bins per Sub-Problem", min_value=1, value=10, key="sg_feature_bins")
                diversity_metric = st.selectbox("Diversity Metric per Sub-Problem", options=["edit_distance", "cosine_similarity", "jaccard_index"], index=0, key="sg_diversity_metric")

        # Store selected teams and gauntlets in session state for retrieval by orchestrator.
        st.session_state.sg_config = {
            "content_analyzer_team_name": content_analyzer_team_name,
            "planner_team_name": planner_team_name,
            "solver_team_name": solver_team_name,
            "patcher_team_name": patcher_team_name,
            "solver_generation_gauntlet_name": solver_generation_gauntlet_name,
            "sub_problem_red_gauntlet_name": sub_problem_red_gauntlet_name,
            "sub_problem_gold_gauntlet_name": sub_problem_gold_gauntlet_name,
            "assembler_team_name": assembler_team_name,
            "final_red_gauntlet_name": final_red_gauntlet_name,
            "final_gold_gauntlet_name": final_gold_gauntlet_name,
            "max_refinement_loops": max_refinement_loops,

            # Core Evolution Parameters
            "max_iterations": max_iterations,
            "population_size": population_size,
            "num_islands": num_islands,
            "migration_interval": migration_interval,
            "migration_rate": migration_rate,
            "archive_size": archive_size,
            "elite_ratio": elite_ratio,
            "exploration_ratio": exploration_ratio,
            "exploitation_ratio": exploitation_ratio,
            "checkpoint_interval": checkpoint_interval,
            "feature_dimensions": feature_dimensions,
            "feature_bins": feature_bins,
            "diversity_metric": diversity_metric,

            # Advanced Evaluation Parameters
            "enable_artifacts": enable_artifacts,
            "cascade_evaluation": cascade_evaluation,
            "cascade_thresholds": [float(x.strip()) for x in cascade_thresholds_str.split(',')] if cascade_thresholds_str else [],
            "use_llm_feedback": use_llm_feedback,
            "llm_feedback_weight": llm_feedback_weight,
            "parallel_evaluations": parallel_evaluations,
            "distributed": distributed,
            "template_dir": template_dir if template_dir else None,
            "num_top_programs": num_top_programs,
            "num_diverse_programs": num_diverse_programs,
            "use_template_stochasticity": use_template_stochasticity,
            "template_variations": json.loads(template_variations_str) if template_variations_str else {},
            "use_meta_prompting": use_meta_prompting,
            "meta_prompt_weight": meta_prompt_weight,
            "include_artifacts": include_artifacts,
            "max_artifact_bytes": max_artifact_bytes,
            "artifact_security_filter": artifact_security_filter,
            "early_stopping_patience": early_stopping_patience if early_stopping_patience > 0 else None,
            "convergence_threshold": convergence_threshold,
            "early_stopping_metric": early_stopping_metric,
            "memory_limit_mb": memory_limit_mb,
            "cpu_limit": cpu_limit,
            "random_seed": random_seed,
            "db_path": db_path if db_path else None,
            "in_memory": in_memory,
            "artifact_size_threshold": artifact_size_threshold,
            "cleanup_old_artifacts": cleanup_old_artifacts,
            "artifact_retention_days": artifact_retention_days,
            "diversity_reference_size": diversity_reference_size,
            "max_retries_eval": max_retries_eval,
            "evaluator_timeout": evaluator_timeout,

            # Advanced OpenEvolve Parameters
            "diff_based_evolution": diff_based_evolution,
            "max_code_length": max_code_length,
            "evolution_trace_enabled": evolution_trace_enabled,
            "evolution_trace_format": evolution_trace_format,
            "evolution_trace_include_code": evolution_trace_include_code,
            "evolution_trace_include_prompts": evolution_trace_include_prompts,
            "evolution_trace_output_path": evolution_trace_output_path if evolution_trace_output_path else None,
            "evolution_trace_buffer_size": evolution_trace_buffer_size,
            "evolution_trace_compress": evolution_trace_compress,
            "log_level": log_level,
            "log_dir": log_dir if log_dir else None,
            "api_timeout": api_timeout,
            "api_retries": api_retries,
            "api_retry_delay": api_retry_delay,

            # Research-Grade Features
            "double_selection": double_selection,
            "adaptive_feature_dimensions": adaptive_feature_dimensions,
            "test_time_compute": test_time_compute,
            "optillm_integration": optillm_integration,
            "plugin_system": plugin_system,
            "hardware_optimization": hardware_optimization,
            "multi_strategy_sampling": multi_strategy_sampling,
            "ring_topology": ring_topology,
            "controlled_gene_flow": controlled_gene_flow,
            "auto_diff": auto_diff,
            "symbolic_execution": symbolic_execution,
            "coevolutionary_approach": coevolutionary_approach,
        }
    # --- End Sovereign-Grade Decomposition Workflow Configuration ---
    
    # Start workflow button with enhanced feedback
    start_col, info_col = st.columns([1, 3])
    with start_col:
        if st.button("🚀 Start Workflow", type="primary", use_container_width=True):
            if not content.strip():
                st.error("❌ Please enter content to evolve")
                return
                
            if not st.session_state.get("api_key"):
                st.error("❌ Please configure your API key in the sidebar")
                return
            
            if workflow_type == "sovereign_decomposition":
                sg_config = st.session_state.get("sg_config")
                if not sg_config:
                    st.error("❌ Please configure the Sovereign-Grade Workflow settings.")
                    return
                
                # Retrieve Teams and Gauntlets
                team_manager = TeamManager()
                gauntlet_manager = GauntletManager()

                content_analyzer_team = team_manager.get_team(sg_config["content_analyzer_team_name"])
                planner_team = team_manager.get_team(sg_config["planner_team_name"])
                solver_team = team_manager.get_team(sg_config["solver_team_name"])
                patcher_team = team_manager.get_team(sg_config["patcher_team_name"])
                assembler_team = team_manager.get_team(sg_config["assembler_team_name"])

                sub_problem_red_gauntlet = gauntlet_manager.get_gauntlet(sg_config["sub_problem_red_gauntlet_name"])
                sub_problem_gold_gauntlet = gauntlet_manager.get_gauntlet(sg_config["sub_problem_gold_gauntlet_name"])
                final_red_gauntlet = gauntlet_manager.get_gauntlet(sg_config["final_red_gauntlet_name"])
                final_gold_gauntlet = gauntlet_manager.get_gauntlet(sg_config["final_gold_gauntlet_name"])
                
                # Basic validation for selected teams/gauntlets
                invalid_configs = []
                if not content_analyzer_team: invalid_configs.append("Content Analyzer Team")
                if not planner_team: invalid_configs.append("Planner Team")
                if not solver_team: invalid_configs.append("Solver Team")
                if not patcher_team: invalid_configs.append("Patcher Team")
                if not assembler_team: invalid_configs.append("Assembler Team")
                if not gauntlet_manager.get_gauntlet(sg_config["solver_generation_gauntlet_name"]): invalid_configs.append("Solver Generation Gauntlet")
                if not sub_problem_red_gauntlet: invalid_configs.append("Sub-Problem Red Team Gauntlet")
                if not sub_problem_gold_gauntlet: invalid_configs.append("Sub-Problem Gold Team Gauntlet")
                if not final_red_gauntlet: invalid_configs.append("Final Red Team Gauntlet")
                if not final_gold_gauntlet: invalid_configs.append("Final Gold Team Gauntlet")

                if invalid_configs:
                    st.error(f"❌ The following configurations are invalid or missing: {', '.join(invalid_configs)}. Please check your configuration.")
                    return

                # Create a new WorkflowState for the Sovereign-Grade workflow
                workflow_id = f"sg_workflow_{int(time.time())}"
                workflow_state = WorkflowState(
                    workflow_id=workflow_id,
                    current_stage="INITIALIZING",
                    workflow_type=EvolutionWorkflow.SOVEREIGN_DECOMPOSITION,
                    problem_statement=content,
                    # Store all configured teams and gauntlets in the workflow_state for easy access
                    content_analyzer_team=content_analyzer_team,
                    planner_team=planner_team,
                    solver_team=solver_team,
                    patcher_team=patcher_team,
                    solver_generation_gauntlet=gauntlet_manager.get_gauntlet(sg_config["solver_generation_gauntlet_name"]),
                    assembler_team=assembler_team,
                    sub_problem_red_gauntlet=sub_problem_red_gauntlet,
                    sub_problem_gold_gauntlet=sub_problem_gold_gauntlet,
                    final_red_gauntlet=final_red_gauntlet,
                    final_gold_gauntlet=final_gold_gauntlet,
                    max_refinement_loops=sg_config["max_refinement_loops"],

                    # Populate OpenEvolve parameters from sg_config
                    max_iterations=sg_config["max_iterations"],
                    population_size=sg_config["population_size"],
                    num_islands=sg_config["num_islands"],
                    migration_interval=sg_config["migration_interval"],
                    migration_rate=sg_config["migration_rate"],
                    archive_size=sg_config["archive_size"],
                    elite_ratio=sg_config["elite_ratio"],
                    exploration_ratio=sg_config["exploration_ratio"],
                    exploitation_ratio=sg_config["exploitation_ratio"],
                    checkpoint_interval=sg_config["checkpoint_interval"],
                    feature_dimensions=sg_config["feature_dimensions"],
                    feature_bins=sg_config["feature_bins"],
                    diversity_metric=sg_config["diversity_metric"],

                    enable_artifacts=sg_config["enable_artifacts"],
                    cascade_evaluation=sg_config["cascade_evaluation"],
                    cascade_thresholds=sg_config["cascade_thresholds"],
                    use_llm_feedback=sg_config["use_llm_feedback"],
                    llm_feedback_weight=sg_config["llm_feedback_weight"],
                    parallel_evaluations=sg_config["parallel_evaluations"],
                    distributed=sg_config["distributed"],
                    template_dir=sg_config["template_dir"],
                    num_top_programs=sg_config["num_top_programs"],
                    num_diverse_programs=sg_config["num_diverse_programs"],
                    use_template_stochasticity=sg_config["use_template_stochasticity"],
                    template_variations=sg_config["template_variations"],
                    use_meta_prompting=sg_config["use_meta_prompting"],
                    meta_prompt_weight=sg_config["meta_prompt_weight"],
                    include_artifacts=sg_config["include_artifacts"],
                    max_artifact_bytes=sg_config["max_artifact_bytes"],
                    artifact_security_filter=sg_config["artifact_security_filter"],
                    early_stopping_patience=sg_config["early_stopping_patience"],
                    convergence_threshold=sg_config["convergence_threshold"],
                    early_stopping_metric=sg_config["early_stopping_metric"],
                    memory_limit_mb=sg_config["memory_limit_mb"],
                    cpu_limit=sg_config["cpu_limit"],
                    random_seed=sg_config["random_seed"],
                    db_path=sg_config["db_path"],
                    in_memory=sg_config["in_memory"],
                    artifact_size_threshold=sg_config["artifact_size_threshold"],
                    cleanup_old_artifacts=sg_config["cleanup_old_artifacts"],
                    artifact_retention_days=sg_config["artifact_retention_days"],
                    diversity_reference_size=sg_config["diversity_reference_size"],
                    max_retries_eval=sg_config["max_retries_eval"],
                    evaluator_timeout=sg_config["evaluator_timeout"],

                    diff_based_evolution=sg_config["diff_based_evolution"],
                    max_code_length=sg_config["max_code_length"],
                    evolution_trace_enabled=sg_config["evolution_trace_enabled"],
                    evolution_trace_format=sg_config["evolution_trace_format"],
                    evolution_trace_include_code=sg_config["evolution_trace_include_code"],
                    evolution_trace_include_prompts=sg_config["evolution_trace_include_prompts"],
                    evolution_trace_output_path=sg_config["evolution_trace_output_path"],
                    evolution_trace_buffer_size=sg_config["evolution_trace_buffer_size"],
                    evolution_trace_compress=sg_config["evolution_trace_compress"],
                    log_level=sg_config["log_level"],
                    log_dir=sg_config["log_dir"],
                    api_timeout=sg_config["api_timeout"],
                    api_retries=sg_config["api_retries"],
                    api_retry_delay=sg_config["api_retry_delay"],

                    double_selection=sg_config["double_selection"],
                    adaptive_feature_dimensions=sg_config["adaptive_feature_dimensions"],
                    test_time_compute=sg_config["test_time_compute"],
                    optillm_integration=sg_config["optillm_integration"],
                    plugin_system=sg_config["plugin_system"],
                    hardware_optimization=sg_config["hardware_optimization"],
                    multi_strategy_sampling=sg_config["multi_strategy_sampling"],
                    ring_topology=sg_config["ring_topology"],
                    controlled_gene_flow=sg_config["controlled_gene_flow"],
                    auto_diff=sg_config["auto_diff"],
                    symbolic_execution=sg_config["symbolic_execution"],
                    coevolutionary_approach=sg_config["coevolutionary_approach"],
                )
                
                # Store the workflow_state in Streamlit's session state
                st.session_state.active_sovereign_workflow = workflow_state
                st.session_state.current_workflow_id = workflow_id # For monitoring
                
                st.success(f"✅ Sovereign-Grade Workflow '{workflow_id}' initialized. Starting execution...")
                # The actual execution will be triggered by Streamlit's rerun mechanism
                # when the UI renders the monitoring tab.
                st.rerun() # Trigger rerun to start execution in monitoring tab
                
            else: # Existing evolution workflows
                # Create workflow parameters with ALL OpenEvolve parameters
                parameters = {
                    # Core parameters
                    "content": content,
                    "content_type": content_type,  # Use the selected content type
                    "model_configs": [{
                        "name": st.session_state.model,
                        "weight": 1.0,
                        "api_key": st.session_state.api_key,
                        "api_base": st.session_state.base_url,
                        "temperature": st.session_state.temperature,
                        "top_p": st.session_state.top_p,
                        "max_tokens": st.session_state.max_tokens,
                        "frequency_penalty": st.session_state.frequency_penalty,
                        "presence_penalty": st.session_state.presence_penalty,
                        "seed": st.session_state.seed,
                        "stop_sequences": [s.strip() for s in st.session_state.stop_sequences.split(',')] if st.session_state.stop_sequences else None,
                        "logprobs": st.session_state.logprobs if st.session_state.logprobs else None,
                        "top_logprobs": st.session_state.top_logprobs if st.session_state.top_logprobs > 0 else None,
                        "response_format": json.loads(st.session_state.response_format) if st.session_state.response_format else None,
                        "stream": st.session_state.stream if st.session_state.stream else None,
                        "user": st.session_state.user if st.session_state.user else None
                    }],
                    "api_key": st.session_state.api_key, # Also pass top-level for backward compatibility if needed
                    "api_base": st.session_state.base_url, # Also pass top-level
                    "max_iterations": max_iterations,
                    "population_size": population_size,
                    "system_message": st.session_state.system_prompt,
                    "evaluator_system_message": st.session_state.evaluator_system_prompt,
                    "temperature": st.session_state.temperature, # Also pass top-level
                    "top_p": st.session_state.top_p, # Also pass top-level
                    "max_tokens": st.session_state.max_tokens, # Also pass top-level
                    "elite_ratio": elite_ratio,
                    "exploration_ratio": exploration_ratio,
                    "exploitation_ratio": exploitation_ratio,
                    "checkpoint_interval": 100,
                    "migration_interval": 50,
                    "migration_rate": 0.1,
                    "seed": st.session_state.seed, # Also pass top-level

                    
                    # Advanced evaluation parameters
                    "enable_artifacts": enable_artifacts,
                    "cascade_evaluation": cascade_evaluation,
                    "cascade_thresholds": [0.5, 0.75, 0.9],
                    "use_llm_feedback": use_llm_feedback,
                    "llm_feedback_weight": 0.1,
                    "parallel_evaluations": parallel_evaluations,
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
                    "memory_limit_mb": memory_limit_mb,
                    "cpu_limit": cpu_limit,
                    "random_seed": st.session_state.get("seed", 42),
                    "db_path": None,
                    "in_memory": True,
                    
                    # Advanced OpenEvolve parameters
                    "diff_based_evolution": True,
                    "max_code_length": max_code_length,
                    "evolution_trace_enabled": evolution_trace_enabled,
                    "evolution_trace_format": evolution_trace_format,
                    "evolution_trace_include_code": evolution_trace_include_code,
                    "evolution_trace_include_prompts": evolution_trace_include_prompts,
                    "evolution_trace_output_path": None,
                    "evolution_trace_buffer_size": evolution_trace_buffer_size,
                    "evolution_trace_compress": evolution_trace_compress,
                    "log_level": "INFO",
                    "log_dir": None,
                    "api_timeout": 60,
                    "api_retries": 3,
                    "api_retry_delay": 5,
                    "artifact_size_threshold": 32 * 1024,
                    "cleanup_old_artifacts": True,
                    "artifact_retention_days": 30,
                    "diversity_reference_size": 20,
                    "max_retries_eval": max_retries_eval,
                    "evaluator_timeout": evaluator_timeout,
                    "evaluator_models": None,
                    
                    # Advanced research-grade features
                    "double_selection": double_selection,
                    "adaptive_feature_dimensions": adaptive_feature_dimensions,
                    "test_time_compute": test_time_compute,
                    "optillm_integration": optillm_integration,
                    "plugin_system": plugin_system,
                    "hardware_optimization": hardware_optimization,
                    "multi_strategy_sampling": multi_strategy_sampling,
                    "ring_topology": ring_topology,
                    "controlled_gene_flow": controlled_gene_flow,
                    "auto_diff": auto_diff,
                    "symbolic_execution": symbolic_execution,
                    "coevolutionary_approach": coevolutionary_approach,
                }
                
                # Create and start workflow
                workflow_id = orchestrator.create_workflow(
                    workflow_type=EvolutionWorkflow(workflow_type),
                    parameters=parameters
                )
                
                if orchestrator.start_workflow(workflow_id):
                    st.success(f"✅ Workflow started: {workflow_id}")
                    st.session_state.current_workflow = workflow_id
                else:
                    st.error("❌ Failed to start workflow")
    
    with info_col:
        st.info("""
        **💡 Tip**: 
        - Start with default parameters for standard evolution
        - For diverse solutions, enable Quality-Diversity workflow type
        - For multiple objectives, use Multi-Objective workflow type
        - Monitor resource usage in the Monitoring Panel
        """)





def render_monitoring_tab(orchestrator: OpenEvolveOrchestrator):
    """Renders the 'Monitoring Panel' tab in the Streamlit UI, displaying real-time status and progress of active workflows.

    Args:
        orchestrator (OpenEvolveOrchestrator): The orchestrator instance to interact with.
    """
    st.subheader("Real-time Monitoring")
    
    # Add usage information
    st.info("""
    **Purpose**: Monitor active evolution workflows in real-time to track progress and performance.
    
    **How to Use**: 
    - View all currently running workflows in the table below
    - Monitor progress, stage, runtime, and status
    - Stop workflows if needed using the stop button
    - Check detailed metrics and performance indicators
    
    **Why This Matters**: Real-time monitoring allows you to track the evolution performance, identify bottlenecks, and make decisions about resource allocation and workflow continuation.
    """)

    # Check for active Sovereign-Grade Workflow
    if "active_sovereign_workflow" in st.session_state:
        workflow_state: WorkflowState = st.session_state.active_sovereign_workflow

        # Always run the workflow engine if the status is 'running'
        if workflow_state.status == "running":
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
                max_refinement_loops=workflow_state.max_refinement_loops,
                solver_generation_gauntlet=getattr(workflow_state, 'solver_generation_gauntlet', None)
            )

        st.subheader(f"👑 Sovereign-Grade Workflow: {workflow_state.workflow_id}")

        # Handle UI based on the current state
        if workflow_state.status == "awaiting_user_input" and workflow_state.current_stage == "Manual Review & Override":
            st.warning("Please review and approve the decomposition plan below to continue the workflow.")
            from ui_components import render_manual_review_panel # Import here to avoid potential circular dependencies at startup
            approval_status, approved_plan = render_manual_review_panel(workflow_state.decomposition_plan)
            
            if approval_status == "approved":
                workflow_state.decomposition_plan = approved_plan
                workflow_state.current_stage = "Sub-Problem Solving Loop"
                workflow_state.status = "running"
                st.success("Plan approved. Resuming workflow...")
                st.rerun() # Rerun to continue the workflow engine
            elif approval_status == "rejected":
                workflow_state.status = "failed"
                st.error("Workflow terminated due to plan rejection.")
                if "active_sovereign_workflow" in st.session_state:
                    del st.session_state.active_sovereign_workflow
                st.rerun()
            return # Stop further rendering in this cycle

        # Display progress and status for all other states
        st.markdown(f"**Current Stage**: `{workflow_state.current_stage}`")
        if workflow_state.current_sub_problem_id:
            st.markdown(f"**Working on Sub-Problem**: `{workflow_state.current_sub_problem_id}`")
        if workflow_state.current_gauntlet_name:
            st.markdown(f"**Running Gauntlet**: `{workflow_state.current_gauntlet_name}`")
        
        st.progress(workflow_state.progress)
        st.info(f"Status: {workflow_state.status.capitalize()}")

        if workflow_state.status == "completed":
            st.success("Workflow completed successfully!")
            st.balloons()
            del st.session_state.active_sovereign_workflow
        elif workflow_state.status == "failed":
            st.error("Workflow failed. Check logs for details.")
            del st.session_state.active_sovereign_workflow
        
        if workflow_state.status == "running":
            time.sleep(1)
            st.rerun()
        return

    # Active workflows (for traditional evolution workflows)
    active_workflows = orchestrator.get_active_workflows()
    
    if not active_workflows:
        st.info("No active workflows running. Start a new workflow in the 'Create Workflow' tab.")
        return
    
    # Display active workflows with more details
    for workflow_status in active_workflows:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"**Workflow ID**: {workflow_status['workflow_id']}")
                st.caption(f"**Type**: {workflow_status['workflow_type'].replace('_', ' ').title()}")
            
            with col2:
                progress_pct = workflow_status['progress'] * 100
                st.metric(
                    label="Progress", 
                    value=f"{progress_pct:.1f}%", 
                    delta=None
                )
                st.caption(f"**Stage**: {workflow_status['current_stage'].replace('_', ' ').title()}")
            
            with col3:
                duration = workflow_status['duration']
                st.metric(
                    label="Runtime", 
                    value=f"{duration:.1f}s",
                    help=f"Total runtime since workflow started"
                )
                st.caption(f"**Status**: {workflow_status['status'].title()}")
            
            with col4:
                if st.button("⏹️ Stop Workflow", key=f"stop_{workflow_status['workflow_id']}", type="secondary"):
                    if orchestrator.stop_workflow(workflow_status['workflow_id']):
                        st.success(f"Workflow {workflow_status['workflow_id']} stopped successfully")
                        # Instead of st.rerun(), we let the next refresh update the UI
                    else:
                        st.error(f"Failed to stop workflow {workflow_status['workflow_id']}")
            
            # Enhanced progress bar with status indicators
            st.progress(workflow_status['progress'])
            
            # Detailed status information
            if workflow_status['current_stage'] in ['configuration', 'initialization']:
                st.info(f"🔧 Initializing workflow components...")
            elif workflow_status['current_stage'] == 'execution':
                st.info(f"⚙️ Running evolution iterations...")
            elif workflow_status['current_stage'] == 'monitoring':
                st.info(f"📊 Monitoring ongoing evolution...")
            elif workflow_status['current_stage'] == 'analysis':
                st.info(f"🔍 Analyzing results...")
            elif workflow_status['current_stage'] == 'reporting':
                st.info(f"📝 Generating report...")
            elif workflow_status['current_stage'] == 'completion':
                st.info(f"✅ Finalizing workflow...")
            
            # Add additional metrics if available
            if 'metrics' in workflow_status and workflow_status['metrics']:
                with st.expander("📈 Detailed Metrics", expanded=False):
                    for key, value in list(workflow_status['metrics'].items())[:5]:  # Show first 5 metrics
                        st.write(f"{key}: {value}")


def render_history_tab(orchestrator: OpenEvolveOrchestrator):
    """Renders the 'History' tab in the Streamlit UI, displaying completed, failed, or cancelled workflows.

    Args:
        orchestrator (OpenEvolveOrchestrator): The orchestrator instance to interact with.
    """
    st.subheader("Workflow History")
    
    # Add usage information
    st.info("""
    **Purpose**: View and manage completed evolution workflows to analyze results and identify patterns.
    
    **How to Use**: 
    - Browse completed workflows in the history table
    - View details of specific workflows
    - Access results, metrics, and reports
    - Use historical data to improve future workflow configurations
    
    **Why This Matters**: Historical data provides insights into evolution patterns, parameter effectiveness, and helps with future workflow optimization.
    """)
    
    # Retrieve historical workflows from the WorkflowHistoryManager
    historical_workflows = orchestrator.history_manager.get_all_historical_workflows()
    
    if not historical_workflows:
        st.info("No completed workflows found. Run some workflows in the 'Create Workflow' tab to see history here.")
        
        # Show some example workflow types that could help guide users
        st.markdown("### 📋 Example Workflow Types to Try:")
        st.markdown("""
        - **Standard Evolution**: Optimize a single objective (e.g., code efficiency, text quality)
        - **Quality-Diversity**: Generate diverse high-quality solutions across multiple dimensions
        - **Multi-Objective**: Optimize multiple competing objectives simultaneously
        - **Adversarial**: Use red-team/blue-team approach to harden solutions
        """)
        return
    
    # Display completed workflows in a table format
    for workflow_state in historical_workflows:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"**{workflow_state.workflow_id}**")
                st.caption(f"Type: {workflow_state.workflow_type.value.replace('_', ' ').title()}")
            
            with col2:
                status_emoji = "✅" if workflow_state.status == 'completed' else "❌" if workflow_state.status == 'failed' else "🛑"
                st.metric("Status", f"{status_emoji} {workflow_state.status.title()}")
            
            with col3:
                duration = (workflow_state.end_time or time.time()) - workflow_state.start_time
                st.metric("Duration", f"{duration:.1f}s")
            
            with col4:
                start_time_str = time.strftime('%H:%M:%S', time.localtime(workflow_state.start_time))
                st.metric("Start Time", start_time_str)
            
            # Expandable section for details
            with st.expander("📊 View Details", expanded=False):
                st.write(f"**Status**: {workflow_state.status.title()}")
                st.write(f"**Duration**: {duration:.1f} seconds")
                st.write(f"**Start Time**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(workflow_state.start_time))}")
                
                if workflow_state.end_time:
                    st.write(f"**End Time**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(workflow_state.end_time))}")
                
                # For traditional workflows, parameters and metrics are stored directly in workflow_state.parameters/metrics
                if workflow_state.workflow_type != EvolutionWorkflow.SOVEREIGN_DECOMPOSITION:
                    if workflow_state.metrics:
                        st.write("**Key Metrics**: ")
                        col1, col2 = st.columns(2)
                        for i, (key, value) in enumerate(list(workflow_state.metrics.items())[:10]):  # Show first 10 metrics
                            if i % 2 == 0:
                                col1.write(f"- {key}: {value}")
                            else:
                                col2.write(f"- {key}: {value}")
                    
                    if workflow_state.results and 'error' in workflow_state.results:
                        st.error(f"**Error**: {workflow_state.results['error']}")
                    
                    with st.expander("🔧 View Parameters"):
                        params = workflow_state.parameters
                        st.json({
                            'content_type': params.get('content_type', 'Unknown'),
                            'max_iterations': params.get('max_iterations', 'Unknown'),
                            'population_size': params.get('population_size', 'Unknown'),
                            'num_islands': params.get('num_islands', 'Unknown'),
                            'temperature': params.get('temperature', 'Unknown'),
                            'workflow_type_details': 'Full parameter set stored in workflow object'
                        })
                else: # Display details for Sovereign-Grade Decomposition workflows
                    st.write("**Sovereign-Grade Workflow Details**")
                    if workflow_state.decomposition_plan:
                        st.write(f"**Problem Statement**: {workflow_state.decomposition_plan.problem_statement}")
                        st.write(f"**Analyzed Context Summary**: {workflow_state.decomposition_plan.analyzed_context.get('summary', 'N/A')}")
                        st.write(f"**Sub-Problems Solved**: {len(workflow_state.sub_problem_solutions)}/{len(workflow_state.decomposition_plan.sub_problems)}")
                        
                        with st.expander("View Decomposition Plan"):
                            for sp in workflow_state.decomposition_plan.sub_problems:
                                st.write(f"- **{sp.id}**: {sp.description}")
                                if sp.id in workflow_state.sub_problem_solutions:
                                    st.success(f"  Solution: {workflow_state.sub_problem_solutions[sp.id].content[:100]}...")
                                else:
                                    st.warning("  Solution not found or not yet solved.")
                        
                        if workflow_state.final_solution:
                            st.success(f"**Final Solution**: {workflow_state.final_solution.content[:200]}...")
                        else:
                            st.warning("Final solution not yet available.")
                    
                    if workflow_state.all_critique_reports:
                        with st.expander("View Critique Reports"):
                            for report in workflow_state.all_critique_reports:
                                st.json(dataclasses.asdict(report))
                    if workflow_state.all_verification_reports:
                        with st.expander("View Verification Reports"):
                            for report in workflow_state.all_verification_reports:
                                st.json(dataclasses.asdict(report))

def render_configuration_tab(orchestrator: OpenEvolveOrchestrator):
    """Renders the 'Configuration' tab in the Streamlit UI, allowing users to manage workflow templates, parameter presets, and global settings.

    Args:
        orchestrator (OpenEvolveOrchestrator): The orchestrator instance to interact with.
    """
    st.subheader("Configuration Management")
    
    st.info("""
    **Purpose**: Manage workflow templates, presets, and global configuration settings to streamline workflow creation.
    
    **How to Use**:
    1. Create workflow templates for common evolution scenarios
    2. Save and load parameter presets
    3. Manage global orchestrator settings
    4. Export/import configurations for sharing or backup
    
    **Why This Matters**: Configuration management allows you to standardize evolution processes, reuse effective parameter sets, and maintain consistency across multiple workflow runs.
    """)
    
    # Initialize TemplateManager
    template_manager = TemplateManager()

    # Tabs for different configuration aspects
    config_tabs = st.tabs(["Team Manager", "Gauntlet Designer", "Workflow Templates", "Parameter Presets", "Global Settings", "Import/Export"])
    
    with config_tabs[0]:  # Team Manager
        render_team_manager()
    
    with config_tabs[1]:  # Gauntlet Designer
        render_gauntlet_designer()

    with config_tabs[2]:  # Workflow Templates
        st.markdown("#### 📋 Workflow Templates")
        st.write("Create and manage templates for common workflow configurations")
        
        # Create new template
        with st.expander("Create New Template", expanded=True):
            template_name = st.text_input("Template Name", placeholder="e.g., Python Code Optimization", key="new_template_name")
            template_description = st.text_area("Description", placeholder="Describe what this template is for...", key="new_template_description")
            
            # Default parameters for the template (capturing all configurable parameters)
            default_params = {
                "model": st.session_state.get("model", "gpt-4o"),
                "api_key": st.session_state.get("api_key", ""),
                "base_url": st.session_state.get("base_url", "https://api.openai.com/v1"),
                "temperature": st.session_state.get("temperature", 0.7),
                "top_p": st.session_state.get("top_p", 0.95),
                "max_tokens": st.session_state.get("max_tokens", 4096),
                "frequency_penalty": st.session_state.get("frequency_penalty", 0.0),
                "presence_penalty": st.session_state.get("presence_penalty", 0.0),
                "seed": st.session_state.get("seed", 42),
                "stop_sequences": st.session_state.get("stop_sequences", ""),
                "logprobs": st.session_state.get("logprobs", False),
                "top_logprobs": st.session_state.get("top_logprobs", 0),
                "response_format": st.session_state.get("response_format", ""),
                "stream": st.session_state.get("stream", False),
                "user": st.session_state.get("user", ""),
                "system_prompt": st.session_state.get("system_prompt", ""),
                "evaluator_system_prompt": st.session_state.get("evaluator_system_prompt", ""),

                "max_iterations": st.session_state.get("max_iterations", 100),
                "population_size": st.session_state.get("population_size", 100),
                "num_islands": st.session_state.get("num_islands", 5),
                "archive_size": st.session_state.get("archive_size", 100),
                "elite_ratio": st.session_state.get("elite_ratio", 0.1),
                "exploration_ratio": st.session_state.get("exploration_ratio", 0.2),
                "exploitation_ratio": st.session_state.get("exploitation_ratio", 0.7),
                "checkpoint_interval": st.session_state.get("checkpoint_interval", 100),
                "feature_dimensions": st.session_state.get("feature_dimensions", ["complexity", "diversity"]),
                "feature_bins": st.session_state.get("feature_bins", 10),
                "diversity_metric": st.session_state.get("diversity_metric", "edit_distance"),

                "enable_artifacts": st.session_state.get("enable_artifacts", True),
                "cascade_evaluation": st.session_state.get("cascade_evaluation", True),
                "cascade_thresholds": st.session_state.get("cascade_thresholds", [0.5, 0.75, 0.9]),
                "use_llm_feedback": st.session_state.get("use_llm_feedback", False),
                "llm_feedback_weight": st.session_state.get("llm_feedback_weight", 0.1),
                "parallel_evaluations": st.session_state.get("parallel_evaluations", 4),
                "distributed": st.session_state.get("distributed", False),
                "template_dir": st.session_state.get("template_dir", None),
                "num_top_programs": st.session_state.get("num_top_programs", 3),
                "num_diverse_programs": st.session_state.get("num_diverse_programs", 2),
                "use_template_stochasticity": st.session_state.get("use_template_stochasticity", True),
                "template_variations": st.session_state.get("template_variations", {}),
                "use_meta_prompting": st.session_state.get("use_meta_prompting", False),
                "meta_prompt_weight": st.session_state.get("meta_prompt_weight", 0.1),
                "include_artifacts": st.session_state.get("include_artifacts", True),
                "max_artifact_bytes": st.session_state.get("max_artifact_bytes", 20 * 1024),
                "artifact_security_filter": st.session_state.get("artifact_security_filter", True),
                "early_stopping_patience": st.session_state.get("early_stopping_patience", None),
                "convergence_threshold": st.session_state.get("convergence_threshold", 0.001),
                "early_stopping_metric": st.session_state.get("early_stopping_metric", "combined_score"),
                "memory_limit_mb": st.session_state.get("memory_limit_mb", 2048),
                "cpu_limit": st.session_state.get("cpu_limit", 4.0),
                "random_seed": st.session_state.get("random_seed", 42),
                "db_path": st.session_state.get("db_path", None),
                "in_memory": st.session_state.get("in_memory", True),

                "diff_based_evolution": st.session_state.get("diff_based_evolution", True),
                "max_code_length": st.session_state.get("max_code_length", 10000),
                "evolution_trace_enabled": st.session_state.get("evolution_trace_enabled", False),
                "evolution_trace_format": st.session_state.get("evolution_trace_format", "jsonl"),
                "evolution_trace_include_code": st.session_state.get("evolution_trace_include_code", False),
                "evolution_trace_include_prompts": st.session_state.get("evolution_trace_include_prompts", True),
                "evolution_trace_output_path": st.session_state.get("evolution_trace_output_path", None),
                "evolution_trace_buffer_size": st.session_state.get("evolution_trace_buffer_size", 10),
                "evolution_trace_compress": st.session_state.get("evolution_trace_compress", False),
                "log_level": st.session_state.get("log_level", "INFO"),
                "log_dir": st.session_state.get("log_dir", None),
                "api_timeout": st.session_state.get("api_timeout", 60),
                "api_retries": st.session_state.get("api_retries", 3),
                "api_retry_delay": st.session_state.get("api_retry_delay", 5),
                "artifact_size_threshold": st.session_state.get("artifact_size_threshold", 32 * 1024),
                "cleanup_old_artifacts": st.session_state.get("cleanup_old_artifacts", True),
                "artifact_retention_days": st.session_state.get("artifact_retention_days", 30),
                "diversity_reference_size": st.session_state.get("diversity_reference_size", 20),
                "max_retries_eval": st.session_state.get("max_retries_eval", 3),
                "evaluator_timeout": st.session_state.get("evaluator_timeout", 300),

                "double_selection": st.session_state.get("double_selection", True),
                "adaptive_feature_dimensions": st.session_state.get("adaptive_feature_dimensions", True),
                "test_time_compute": st.session_state.get("test_time_compute", False),
                "optillm_integration": st.session_state.get("optillm_integration", False),
                "plugin_system": st.session_state.get("plugin_system", False),
                "hardware_optimization": st.session_state.get("hardware_optimization", False),
                "multi_strategy_sampling": st.session_state.get("multi_strategy_sampling", True),
                "ring_topology": st.session_state.get("ring_topology", True),
                "controlled_gene_flow": st.session_state.get("controlled_gene_flow", True),
                "auto_diff": st.session_state.get("auto_diff", True),
                "symbolic_execution": st.session_state.get("symbolic_execution", False),
                "coevolutionary_approach": st.session_state.get("coevolutionary_approach", False),
            }
            st.json(default_params)

            if st.button("💾 Save Template", key="save_new_template"):
                if template_name:
                    new_template = WorkflowTemplate(
                        name=template_name,
                        description=template_description,
                        parameters=default_params # Save the current default parameters
                    )
                    template_manager.save_template(new_template)
                    st.success(f"Template '{template_name}' saved successfully!")
                    st.rerun()
                else:
                    st.error("Template Name cannot be empty.")
        
        # Show existing templates
        existing_templates = template_manager.get_all_templates()
        if existing_templates:
            st.write("#### Existing Templates:")
            for template in existing_templates:
                with st.container(border=True):
                    st.write(f"**{template.name}**")
                    st.caption(f"Description: {template.description or 'No description'}")
                    
                    if st.button(f"Use Template: {template.name}", key=f"use_template_{template.name}"):
                        # Apply template parameters to session state
                        for param, value in template.parameters.items():
                            st.session_state[param] = value
                        st.success(f"Applied template '{template.name}' parameters!")
                        st.rerun()
                    
                    if st.button(f"❌ Delete: {template.name}", key=f"del_template_{template.name}"):
                        template_manager.delete_template(template.name)
                        st.success(f"Template '{template.name}' deleted successfully!")
                        st.rerun()
        else:
            st.info("No templates saved yet. Create your first template above.")
    
    with config_tabs[3]:  # Parameter Presets
        st.markdown("#### ⚙️ Parameter Presets")
        st.write("Common parameter configurations for different evolution scenarios")
        
        preset_options = {
            "Conservative": {
                "max_iterations": 50,
                "population_size": 50,
                "temperature": 0.3,
                "description": "Safe settings for initial experimentation"
            },
            "Standard": {
                "max_iterations": 100,
                "population_size": 100,
                "temperature": 0.7,
                "description": "Balanced performance and exploration"
            },
            "Aggressive": {
                "max_iterations": 200,
                "population_size": 200,
                "temperature": 1.0,
                "description": "High exploration for complex problems"
            },
            "Quality-Diversity Focus": {
                "max_iterations": 150,
                "population_size": 150,
                "temperature": 0.5,
                "num_islands": 8,
                "archive_size": 200,
                "description": "Optimized for diversity in solutions"
            }
        }
        
        for preset_name, preset_data in preset_options.items():
            with st.container(border=True):
                st.write(f"**{preset_name}**")
                st.caption(f"{preset_data['description']}")
                
                preset_params = {k: v for k, v in preset_data.items() if k != 'description'}
                param_str = ", ".join([f"{k}: {v}" for k, v in preset_params.items() if k != 'description'])
                st.write(f"Parameters: {param_str}")
                
                if st.button(f"Apply {preset_name}", key=f"apply_{preset_name.lower().replace('-', '_')}"):
                    # Apply preset parameters to session state
                    for param, value in preset_params.items():
                        st.session_state[param] = value
                    st.success(f"Applied '{preset_name}' preset!")
    
    with config_tabs[4]:  # Global Settings
        st.markdown("#### 🌐 Global Settings")
        st.write("Configure global orchestrator behavior and defaults")
        
        with st.expander("Resource Management", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                default_memory_limit = st.number_input(
                    "Default Memory Limit (MB)", 
                    min_value=100, 
                    max_value=32768, 
                    value=2048,
                    help="Default memory limit for new workflows"
                )
                default_cpu_limit = st.number_input(
                    "Default CPU Limit", 
                    min_value=0.1, 
                    max_value=32.0, 
                    value=4.0,
                    step=0.1,
                    help="Default CPU cores to allocate for new workflows"
                )
            with col2:
                default_parallel_eval = st.number_input(
                    "Default Parallel Evaluations", 
                    min_value=1, 
                    max_value=32, 
                    value=4,
                    help="Default number of parallel evaluations for new workflows"
                )
                auto_checkpoint = st.checkbox(
                    "Auto Checkpoint Enabled", 
                    value=True,
                    help="Automatically save checkpoints during evolution"
                )
        
        with st.expander("Logging & Debugging", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                default_log_level = st.selectbox(
                    "Default Log Level", 
                    options=["DEBUG", "INFO", "WARNING", "ERROR"],
                    index=1
                )
                enable_tracing = st.checkbox(
                    "Enable Evolution Tracing by Default", 
                    value=False
                )
            with col2:
                trace_format = st.selectbox(
                    "Default Trace Format", 
                    options=["jsonl", "csv", "parquet"],
                    index=0
                )
                compress_traces = st.checkbox(
                    "Compress Traces by Default", 
                    value=False
                )
        
        if st.button("💾 Save Global Settings"):
            # Store global settings in session state
            st.session_state.setdefault('global_settings', {})
            st.session_state.global_settings.update({
                'default_memory_limit': default_memory_limit,
                'default_cpu_limit': default_cpu_limit,
                'default_parallel_eval': default_parallel_eval,
                'auto_checkpoint': auto_checkpoint,
                'default_log_level': default_log_level,
                'enable_tracing': enable_tracing,
                'trace_format': trace_format,
                'compress_traces': compress_traces
            })
            st.success("Global settings saved!")
        
        # Show current global settings
        if 'global_settings' in st.session_state:
            with st.expander("Current Global Settings", expanded=False):
                settings = st.session_state.global_settings
                for key, value in settings.items():
                    st.write(f"- {key}: {value}")
    
    with config_tabs[5]:  # Import/Export
        st.markdown("#### 📤 Import/Export Configurations")
        st.write("Import or export workflow configurations for sharing or backup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Export Current Configuration")
            export_format = st.selectbox("Export Format", ["JSON", "YAML"], index=0)
            if st.button("📤 Export Configuration"):
                # Create a configuration object
                config_obj = {
                    "workflow_templates": st.session_state.get("workflow_templates", {}),
                    "global_settings": st.session_state.get("global_settings", {}),
                    "export_timestamp": time.time(),
                    "export_format": export_format.lower()
                }
                
                import json
                import yaml
                import io
                
                if export_format == "JSON":
                    config_str = json.dumps(config_obj, indent=2)
                    st.download_button(
                        label="Download JSON Configuration",
                        data=config_str,
                        file_name=f"openevolve_config_{int(time.time())}.json",
                        mime="application/json"
                    )
                else:  # YAML
                    config_str = yaml.dump(config_obj, default_flow_style=False)
                    st.download_button(
                        label="Download YAML Configuration",
                        data=config_str,
                        file_name=f"openevolve_config_{int(time.time())}.yaml",
                        mime="text/yaml"
                    )
        
        with col2:
            st.write("Import Configuration")
            uploaded_file = st.file_uploader("Choose a configuration file", type=["json", "yaml", "yml"])
            if uploaded_file is not None:
                import json
                import yaml
                
                try:
                    if uploaded_file.name.endswith('.json'):
                        config_data = json.load(uploaded_file)
                    else:  # YAML
                        config_data = yaml.safe_load(uploaded_file)
                    
                    if st.button("📥 Import Configuration"):
                        # Merge imported data with current session state
                        if "workflow_templates" in config_data:
                            st.session_state.setdefault("workflow_templates", {})
                            st.session_state.workflow_templates.update(config_data["workflow_templates"])
                        
                        if "global_settings" in config_data:
                            st.session_state.setdefault("global_settings", {})
                            st.session_state.global_settings.update(config_data["global_settings"])
                        
                        st.success("Configuration imported successfully!")
                        
                except Exception as e:
                    st.error(f"Error importing configuration: {e}")


# Initialize session state
def initialize_orchestrator_session():
    """Initializes the OpenEvolveOrchestrator instance in Streamlit's session state if it doesn't already exist.
    This ensures that the orchestrator state persists across Streamlit reruns.
    """
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = OpenEvolveOrchestrator()


# Main execution
if __name__ == "__main__":
    initialize_orchestrator_session()
    render_openevolve_orchestrator_ui()



def start_openevolve_services():
    print("start_openevolve_services called")
    """Starts necessary OpenEvolve backend services, including checking for an LLM backend and launching the visualizer.
    This function attempts to ensure all required background processes are running.
    """
    # Check if LLM backend is already running on port 8000
    try:
        # Try to connect to the LLM server health endpoint
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if response.status_code == 200:
            logging.info("LLM backend is already running on port 8000.")
            return
    except requests.exceptions.ConnectionError:
        logging.warning("LLM backend not running on port 8000.")
        logging.info("OpenEvolve requires an LLM server (like OptiLLM) to be available.")
        logging.info("Please start your LLM server on port 8000 or configure a different endpoint.")
    except requests.exceptions.Timeout:
        logging.warning("LLM backend health check timed out.")
    except Exception as e:
        logging.error(f"Error during LLM backend health check: {e}")
    
    try:
        # Define backend path and command
        backend_path = os.path.join(get_project_root(), "openevolve")
        command = [
            sys.executable,  # Use the same python that runs the frontend
            os.path.join(backend_path, "scripts", "visualizer.py"),
            "--port",
            "8080"
        ]
        
        # Get the current environment and add the project root to PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = get_project_root()

        # Use Popen to start the backend without blocking the main thread
        # Create log files in frontend directory to capture backend output for debugging
        backend_out_log = os.path.join(os.path.dirname(__file__), "backend_stdout.log")
        backend_err_log = os.path.join(os.path.dirname(__file__), "backend_stderr.log")
        
        with open(backend_out_log, "w") as stdout_file, open(backend_err_log, "w") as stderr_file:
            # We use a separate thread to run this to ensure it doesn't interfere with Streamlit's main loop
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
        
        st.session_state.openevolve_backend_process = process
        logging.info(f"OpenEvolve backend started with PID: {process.pid}")
        
        # Wait a bit for the backend to start
        time.sleep(2)
        
        # Double-check if backend is running after starting
        max_retries = 15  # Increased retry attempts
        retry_count = 0
        backend_started = False
        
        while retry_count < max_retries and not backend_started:
            try:
                response = requests.get("http://localhost:8080/", timeout=5)  # Check visualizer server
                if response.status_code == 200:
                    logging.info("OpenEvolve backend confirmed running.")
                    backend_started = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)  # Wait 2 seconds before retrying
            retry_count += 1
            
        if not backend_started:
            logging.warning("OpenEvolve backend may not have started properly. Please check backend logs.")
            logging.info("Backend logs are available at backend_stdout.log and backend_stderr.log")
            
    except Exception as e:
        logging.error(f"Failed to start OpenEvolve backend: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")

def stop_openevolve_services():
    """Stops the OpenEvolve backend process if it is running.
    This ensures a clean shutdown of background services.
    """
    if "openevolve_backend_process" in st.session_state and st.session_state.openevolve_backend_process:
        st.session_state.openevolve_backend_process.terminate()
        st.session_state.openevolve_backend_process = None
        logging.info("OpenEvolve backend process terminated.")

def restart_openevolve_services():
    """Restarts all OpenEvolve services by stopping and then starting them.
    This is useful for applying configuration changes or recovering from issues.
    """
    stop_openevolve_services()
    start_openevolve_services()
    stop_openevolve_services()
    time.sleep(2)
    start_openevolve_services()