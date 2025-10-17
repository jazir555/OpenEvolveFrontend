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
from workflow_engine import run_sovereign_workflow, WorkflowState # NEW IMPORT


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
    ORCHESTRATOR_AVAILABLE = True
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
    """Orchestrates complex OpenEvolve workflows with ALL parameters"""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowState] = {}
        self.active_workflows: List[str] = []
        self.monitor = EvolutionMonitor() if ORCHESTRATOR_AVAILABLE else None
        self.workflow_callbacks: Dict[str, List[Callable]] = {}
        
    def create_workflow(
        self, 
        workflow_type: EvolutionWorkflow,
        parameters: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> str:
        """Create a new evolution workflow"""
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
        """Start an evolution workflow"""
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
        """Stop an evolution workflow"""
        if workflow_id not in self.workflows:
            return False
            
        workflow = self.workflows[workflow_id]
        workflow.status = "cancelled"
        workflow.end_time = time.time()
        
        if workflow_id in self.active_workflows:
            self.active_workflows.remove(workflow_id)
            
        return True
    
    def _execute_workflow(self, workflow_id: str):
        """Execute the workflow steps with ALL OpenEvolve parameters"""
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
            
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                self.active_workflows.remove(workflow_id)
                
        except Exception as e:
            workflow = self.workflows[workflow_id]
            workflow.status = "failed"
            workflow.end_time = time.time()
            workflow.results["error"] = str(e)
            self._notify_callbacks(workflow_id, "workflow_failed", str(e))
            
            if workflow_id in self.active_workflows:
                self.active_workflows.remove(workflow_id)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow"""
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
        """Get all active workflows"""
        statuses = []
        for workflow_id in self.active_workflows:
            status = self.get_workflow_status(workflow_id)
            if status:
                statuses.append(status)
        return statuses
    
    def register_callback(self, workflow_id: str, callback: Callable):
        """Register a callback for workflow events"""
        if workflow_id not in self.workflow_callbacks:
            self.workflow_callbacks[workflow_id] = []
        self.workflow_callbacks[workflow_id].append(callback)
    
    def _notify_callbacks(self, workflow_id: str, event: str, data: Any):
        """Notify registered callbacks of an event"""
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
    """Render the create workflow tab"""
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

        # Get available teams and gauntlets
        all_teams = team_manager.get_all_teams()
        all_gauntlets = gauntlet_manager.get_all_gauntlets()

        blue_teams = [t.name for t in all_teams if t.role == "Blue"]
        red_teams = [t.name for t in all_teams if t.role == "Red"]
        gold_teams = [t.name for t in all_teams if t.role == "Gold"]

        # Filter gauntlets by team role
        # Note: This filtering is simplified. In a real app, you'd ensure the gauntlet's team_name actually exists and has the correct role.
        blue_gauntlets = [g.name for g in all_gauntlets if gauntlet_manager.get_gauntlet(g.name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Blue"]
        red_gauntlets = [g.name for g in all_gauntlets if gauntlet_manager.get_gauntlet(g.name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Red"]
        gold_gauntlets = [g.name for g in all_gauntlets if gauntlet_manager.get_gauntlet(g.name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name) and team_manager.get_team(gauntlet_manager.get_gauntlet(g.name).team_name).role == "Gold"]

        # Stage 0: Content Analysis
        st.markdown("#### Stage 0: Content Analysis")
        content_analyzer_team_name = st.selectbox("Content Analyzer Team (Blue)", blue_teams, key="sg_content_analyzer_team")

        # Stage 1: AI-Assisted Decomposition
        st.markdown("#### Stage 1: AI-Assisted Decomposition")
        planner_team_name = st.selectbox("Planner Team (Blue)", blue_teams, key="sg_planner_team")

        # Stage 3: Sub-Problem Solving Loop
        st.markdown("#### Stage 3: Sub-Problem Solving Loop")
        solver_team_name = st.selectbox("Solver Team (Blue)", blue_teams, key="sg_solver_team")
        patcher_team_name = st.selectbox("Patcher Team (Blue)", blue_teams, key="sg_patcher_team")
        solver_generation_gauntlet_name = st.selectbox("Solver Generation Gauntlet (Blue)", blue_gauntlets, key="sg_solver_generation_gauntlet")
        sub_problem_red_gauntlet_name = st.selectbox("Sub-Problem Red Team Gauntlet", red_gauntlets, key="sg_sub_red_gauntlet")
        sub_problem_gold_gauntlet_name = st.selectbox("Sub-Problem Gold Team Gauntlet", gold_gauntlets, key="sg_sub_gold_gauntlet")

        # Stage 4: Configurable Reassembly
        st.markdown("#### Stage 4: Configurable Reassembly")
        assembler_team_name = st.selectbox("Assembler Team (Blue)", blue_teams, key="sg_assembler_team")

        # Stage 5: Final Verification & Self-Healing Loop
        st.markdown("#### Stage 5: Final Verification & Self-Healing Loop")
        final_red_gauntlet_name = st.selectbox("Final Red Team Gauntlet", red_gauntlets, key="sg_final_red_gauntlet")
        final_gold_gauntlet_name = st.selectbox("Final Gold Team Gauntlet", gold_gauntlets, key="sg_final_gold_gauntlet")
        max_refinement_loops = st.number_input("Max Refinement Loops", min_value=1, value=3, key="sg_max_refinement_loops")

        # Store selected teams and gauntlets in session state for retrieval by orchestrator
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
                if not all([content_analyzer_team, planner_team, solver_team, patcher_team, assembler_team,
                            sub_problem_red_gauntlet, sub_problem_gold_gauntlet, final_red_gauntlet, final_gold_gauntlet]):
                    st.error("❌ One or more selected Teams/Gauntlets are invalid. Please check your configuration.")
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
                    max_refinement_loops=sg_config["max_refinement_loops"]
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
                    "model_configs": [{"name": st.session_state.get("model", "gpt-4o"), "weight": 1.0}],
                    "api_key": st.session_state.get("api_key", ""),
                    "api_base": st.session_state.get("base_url", "https://api.openai.com/v1"),
                    "max_iterations": max_iterations,
                    "population_size": population_size,
                    "num_islands": num_islands,
                    "archive_size": archive_size,
                    "feature_dimensions": feature_dimensions,
                    "feature_bins": 10,
                    "diversity_metric": "edit_distance",
                    "system_message": st.session_state.get("system_prompt", ""),
                    "evaluator_system_message": st.session_state.get("evaluator_system_prompt", ""),
                    "temperature": temperature,
                    "top_p": 0.95,
                    "max_tokens": st.session_state.get("max_tokens", 4096),
                    "elite_ratio": elite_ratio,
                    "exploration_ratio": exploration_ratio,
                    "exploitation_ratio": exploitation_ratio,
                    "checkpoint_interval": 100,
                    "migration_interval": 50,
                    "migration_rate": 0.1,
                    "seed": st.session_state.get("seed", 42),
                    
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
    """Render the monitoring tab"""
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
    if "active_sovereign_workflow" in st.session_state and st.session_state.active_sovereign_workflow.status == "running":
        workflow_state: WorkflowState = st.session_state.active_sovereign_workflow
        st.subheader(f"👑 Sovereign-Grade Workflow: {workflow_state.workflow_id}")
        
        # Call the workflow engine to continue execution
        # This is where Streamlit's rerun mechanism is crucial.
        # The run_sovereign_workflow function will update the workflow_state
        # and then rerun the script, which will re-render this monitoring tab.
        run_sovereign_workflow(
            workflow_state=workflow_state,
            problem_statement=workflow_state.problem_statement,
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
        
        # Display current status
        st.markdown(f"**Current Stage**: `{workflow_state.current_stage}`")
        if workflow_state.current_sub_problem_id:
            st.markdown(f"**Working on Sub-Problem**: `{workflow_state.current_sub_problem_id}`")
        if workflow_state.current_gauntlet_name:
            st.markdown(f"**Running Gauntlet**: `{workflow_state.current_gauntlet_name}`")
        
        st.progress(workflow_state.progress)
        st.info(f"Status: {workflow_state.status.capitalize()}")

        # Handle Manual Review & Override stage
        if workflow_state.current_stage == "Manual Review & Override" and workflow_state.status == "awaiting_user_input":
            st.warning("Please review and approve the decomposition plan below to continue the workflow.")
            approved_plan = render_manual_review_panel(workflow_state.decomposition_plan)
            if approved_plan:
                workflow_state.decomposition_plan = approved_plan
                workflow_state.current_stage = "Sub-Problem Solving Loop"
                workflow_state.status = "running"
                st.rerun()
            elif approved_plan is None: # User rejected the plan
                workflow_state.status = "failed"
                st.error("Workflow terminated due to plan rejection.")
                del st.session_state.active_sovereign_workflow # Clear active workflow
            return # Exit to prevent further execution until user input is handled

        if workflow_state.status == "completed":
            st.success("Workflow completed successfully!")
            st.balloons()
            del st.session_state.active_sovereign_workflow # Clear active workflow
        elif workflow_state.status == "failed":
            st.error("Workflow failed. Check logs for details.")
            del st.session_state.active_sovereign_workflow # Clear active workflow
        
        # Auto-rerun to continue the workflow
        if workflow_state.status == "running":
            time.sleep(1) # Small delay to make progress visible
            st.rerun()
        
        return # Exit to prevent rendering other active workflows for now

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
    """Render the history tab"""
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
    
    # Simulated history data (in a real implementation, this would come from persistent storage)
    completed_workflows = []
    
    # Get completed workflows from orchestrator (if any are already completed)
    for wf_id, wf_state in orchestrator.workflows.items():
        if wf_state.status in ['completed', 'failed', 'cancelled']:
            completed_workflows.append({
                'workflow_id': wf_id,
                'workflow_type': wf_state.workflow_type.value,
                'status': wf_state.status,
                'start_time': wf_state.start_time,
                'end_time': wf_state.end_time or time.time(),
                'duration': (wf_state.end_time or time.time()) - wf_state.start_time,
                'parameters': wf_state.parameters,
                'results': wf_state.results,
                'metrics': wf_state.metrics
            })
    
    if not completed_workflows:
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
    for workflow in completed_workflows:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"**{workflow['workflow_id']}**")
                st.caption(f"Type: {workflow['workflow_type'].replace('_', ' ').title()}")
            
            with col2:
                status_emoji = "✅" if workflow['status'] == 'completed' else "❌" if workflow['status'] == 'failed' else "🛑"
                st.metric("Status", f"{status_emoji} {workflow['status'].title()}")
            
            with col3:
                duration = workflow['duration']
                st.metric("Duration", f"{duration:.1f}s")
            
            with col4:
                start_time = time.strftime('%H:%M:%S', time.localtime(workflow['start_time']))
                st.metric("Start Time", start_time)
            
            # Expandable section for details
            with st.expander("📊 View Details", expanded=False):
                st.write(f"**Status**: {workflow['status'].title()}")
                st.write(f"**Duration**: {workflow['duration']:.1f} seconds")
                st.write(f"**Start Time**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(workflow['start_time']))}")
                
                if workflow['end_time']:
                    st.write(f"**End Time**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(workflow['end_time']))}")
                
                if workflow['metrics']:
                    st.write("**Key Metrics**: ")
                    col1, col2 = st.columns(2)
                    for i, (key, value) in enumerate(list(workflow['metrics'].items())[:10]):  # Show first 10 metrics
                        if i % 2 == 0:
                            col1.write(f"- {key}: {value}")
                        else:
                            col2.write(f"- {key}: {value}")
                
                if workflow['results'] and 'error' in workflow['results']:
                    st.error(f"**Error**: {workflow['results']['error']}")
                
                # Option to view full parameters (in a real implementation, this would load from persistent storage)
                with st.expander("🔧 View Parameters"):
                    params = workflow['parameters']
                    st.json({
                        'content_type': params.get('content_type', 'Unknown'),
                        'max_iterations': params.get('max_iterations', 'Unknown'),
                        'population_size': params.get('population_size', 'Unknown'),
                        'num_islands': params.get('num_islands', 'Unknown'),
                        'temperature': params.get('temperature', 'Unknown'),
                        'workflow_type_details': 'Full parameter set stored in workflow object'
                    })


def render_configuration_tab(orchestrator: OpenEvolveOrchestrator):
    """Render the configuration tab"""
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
            template_name = st.text_input("Template Name", placeholder="e.g., Python Code Optimization")
            template_description = st.text_area("Description", placeholder="Describe what this template is for...")
            
            # Default parameters for the template
            col1, col2 = st.columns(2)
            with col1:
                default_max_iterations = st.number_input("Default Max Iterations", value=100)
                default_population_size = st.number_input("Default Population Size", value=100)
                default_num_islands = st.number_input("Default Number of Islands", value=5)
            with col2:
                default_temperature = st.slider("Default Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
                default_elite_ratio = st.slider("Default Elite Ratio", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            
            if st.button("💾 Save Template"):
                # In a real implementation, this would save to persistent storage
                template_data = {
                    "name": template_name,
                    "description": template_description,
                    "parameters": {
                        "max_iterations": default_max_iterations,
                        "population_size": default_population_size,
                        "num_islands": default_num_islands,
                        "temperature": default_temperature,
                        "elite_ratio": default_elite_ratio
                    }
                }
                # Save template to session state (in real implementation, save to persistent storage)
                if "workflow_templates" not in st.session_state:
                    st.session_state.workflow_templates = {}
                st.session_state.workflow_templates[template_name] = template_data
                st.success(f"Template '{template_name}' saved successfully!")
        
        # Show existing templates
        if "workflow_templates" in st.session_state and st.session_state.workflow_templates:
            st.write("#### Existing Templates:")
            for name, data in st.session_state.workflow_templates.items():
                with st.container(border=True):
                    st.write(f"**{name}**")
                    st.caption(f"Description: {data.get('description', 'No description')}")
                    if st.button(f"Use Template: {name}", key=f"use_{name}"):
                        # Apply template parameters to session state
                        params = data.get('parameters', {})
                        st.session_state.setdefault('max_iterations', params.get('max_iterations', 100))
                        st.session_state.setdefault('population_size', params.get('population_size', 100))
                        st.session_state.setdefault('num_islands', params.get('num_islands', 5))
                        st.session_state.setdefault('temperature', params.get('temperature', 0.7))
                        st.session_state.setdefault('elite_ratio', params.get('elite_ratio', 0.1))
                        st.success(f"Applied template '{name}' parameters!")
                    
                    if st.button(f"❌ Delete: {name}", key=f"del_{name}"):
                        del st.session_state.workflow_templates[name]
                        st.rerun()
        else:
            st.info("No templates saved yet. Create your first template above.")
    
    with config_tabs[1]:  # Parameter Presets
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
    
    with config_tabs[2]:  # Global Settings
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
    
    with config_tabs[3]:  # Import/Export
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
    """Initialize orchestrator session state"""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = OpenEvolveOrchestrator()


# Main execution
if __name__ == "__main__":
    initialize_orchestrator_session()
    render_openevolve_orchestrator_ui()



def start_openevolve_services():
    print("start_openevolve_services called")
    """Start all OpenEvolve services."""
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
    """Stop all OpenEvolve services."""
    if "openevolve_backend_process" in st.session_state and st.session_state.openevolve_backend_process:
        st.session_state.openevolve_backend_process.terminate()
        st.session_state.openevolve_backend_process = None
        logging.info("OpenEvolve backend process terminated.")

def restart_openevolve_services():
    """Restart all OpenEvolve services."""
    stop_openevolve_services()
    time.sleep(2)
    start_openevolve_services()