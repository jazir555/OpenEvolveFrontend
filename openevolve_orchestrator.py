"""
OpenEvolve Orchestration System
Advanced workflow orchestration for ALL OpenEvolve features
"""
import streamlit as st
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

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
    st.header("🤖 OpenEvolve Workflow Orchestrator")
    
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
    
    # Workflow type selection
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
            "prompt_optimization": "📝 Prompt Optimization"
        }.get(x, x)
    )
    
    # Content input
    content = st.text_area(
        "Input Content",
        height=200,
        placeholder="Enter content to evolve here..."
    )
    
    # Advanced configuration
    with st.expander("Core Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            max_iterations = st.number_input("Max Iterations", min_value=1, max_value=10000, value=100)
            population_size = st.number_input("Population Size", min_value=10, max_value=10000, value=100)
            num_islands = st.number_input("Number of Islands", min_value=1, max_value=20, value=5)
            archive_size = st.number_input("Archive Size", min_value=10, max_value=10000, value=100)
        
        with col2:
            temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
            elite_ratio = st.slider("Elite Ratio", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            exploration_ratio = st.slider("Exploration Ratio", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
            exploitation_ratio = st.slider("Exploitation Ratio", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    
    # Feature dimensions for QD and multi-objective
    if workflow_type in ["quality_diversity", "multi_objective"]:
        feature_dimensions = st.multiselect(
            "Feature Dimensions",
            options=["complexity", "diversity", "performance", "readability", "efficiency", "accuracy", "robustness"],
            default=["complexity", "diversity"]
        )
    else:
        feature_dimensions = ["complexity", "diversity"]
    
    # Advanced features
    with st.expander("Advanced Features", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            enable_artifacts = st.checkbox("Enable Artifact Feedback", value=True)
            cascade_evaluation = st.checkbox("Enable Cascade Evaluation", value=True)
            use_llm_feedback = st.checkbox("Use LLM Feedback", value=False)
            evolution_trace_enabled = st.checkbox("Enable Evolution Tracing", value=False)
        
        with col2:
            double_selection = st.checkbox("Double Selection", value=True)
            adaptive_feature_dimensions = st.checkbox("Adaptive Feature Dimensions", value=True)
            multi_strategy_sampling = st.checkbox("Multi-Strategy Sampling", value=True)
            ring_topology = st.checkbox("Ring Topology", value=True)
    
    # Research-grade features
    with st.expander("Research-Grade Features", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            test_time_compute = st.checkbox("Test-Time Compute", value=False)
            optillm_integration = st.checkbox("OptiLLM Integration", value=False)
            plugin_system = st.checkbox("Plugin System", value=False)
            hardware_optimization = st.checkbox("Hardware Optimization", value=False)
        
        with col2:
            controlled_gene_flow = st.checkbox("Controlled Gene Flow", value=True)
            auto_diff = st.checkbox("Auto Diff", value=True)
            symbolic_execution = st.checkbox("Symbolic Execution", value=False)
            coevolutionary_approach = st.checkbox("Coevolutionary Approach", value=False)
    
    # Performance optimization
    with st.expander("Performance Optimization", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            memory_limit_mb = st.number_input("Memory Limit (MB)", min_value=100, max_value=32768, value=2048)
            cpu_limit = st.number_input("CPU Limit", min_value=0.1, max_value=32.0, value=4.0, step=0.1)
            parallel_evaluations = st.number_input("Parallel Evaluations", min_value=1, max_value=32, value=4)
        
        with col2:
            max_code_length = st.number_input("Max Code Length", min_value=100, max_value=100000, value=10000)
            evaluator_timeout = st.number_input("Evaluator Timeout (s)", min_value=10, max_value=3600, value=300)
            max_retries_eval = st.number_input("Max Evaluation Retries", min_value=1, max_value=10, value=3)
    
    # Advanced evolution trace settings
    with st.expander("Evolution Trace Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            evolution_trace_format = st.selectbox("Trace Format", options=["jsonl", "csv", "parquet"], index=0)
            evolution_trace_include_code = st.checkbox("Include Code in Traces", value=False)
            evolution_trace_include_prompts = st.checkbox("Include Prompts in Traces", value=True)
        
        with col2:
            evolution_trace_buffer_size = st.number_input("Trace Buffer Size", min_value=1, max_value=100, value=10)
            evolution_trace_compress = st.checkbox("Compress Traces", value=False)
    
    # Start workflow button
    if st.button("🚀 Start Workflow", type="primary", use_container_width=True):
        if not content.strip():
            st.error("Please enter content to evolve")
            return
            
        if not st.session_state.get("api_key"):
            st.error("Please configure your API key in the sidebar")
            return
        
        # Create workflow parameters with ALL OpenEvolve parameters
        parameters = {
            # Core parameters
            "content": content,
            "content_type": "code_python",  # Default for now
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
            st.success(f"Workflow started: {workflow_id}")
            st.session_state.current_workflow = workflow_id
        else:
            st.error("Failed to start workflow")


def render_monitoring_tab(orchestrator: OpenEvolveOrchestrator):
    """Render the monitoring tab"""
    st.subheader("Real-time Monitoring")
    
    # Active workflows
    active_workflows = orchestrator.get_active_workflows()
    
    if not active_workflows:
        st.info("No active workflows running")
        return
    
    # Display active workflows
    for workflow_status in active_workflows:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"**{workflow_status['workflow_id']}**")
                st.caption(f"Type: {workflow_status['workflow_type']}")
            
            with col2:
                st.metric("Progress", f"{workflow_status['progress']*100:.1f}%")
                st.caption(f"Stage: {workflow_status['current_stage']}")
            
            with col3:
                duration = workflow_status['duration']
                st.metric("Runtime", f"{duration:.1f} seconds")
                st.caption(f"Status: {workflow_status['status']}")
            
            with col4:
                if st.button("⏹️ Stop", key=f"stop_{workflow_status['workflow_id']}"):
                    orchestrator.stop_workflow(workflow_status['workflow_id'])
                    st.rerun()
            
            # Progress bar
            st.progress(workflow_status['progress'])


def render_history_tab(orchestrator: OpenEvolveOrchestrator):
    """Render the history tab"""
    st.subheader("Workflow History")
    
    # For now, we'll just show a message since we don't have persistent storage
    st.info("Workflow history functionality will be implemented in future versions with persistent storage and replay capabilities.")


def render_configuration_tab(orchestrator: OpenEvolveOrchestrator):
    """Render the configuration tab"""
    st.subheader("Configuration Management")
    
    st.markdown("""
    ### Workflow Template Configuration
    
    Here you can create and manage workflow templates for quickly starting common evolution tasks.
    """)
    
    # Template management would go here in a full implementation
    st.info("Workflow template management will be implemented in future versions.")


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