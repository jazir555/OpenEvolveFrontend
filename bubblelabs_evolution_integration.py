"""
BubbleLabs Integration with Evolution and Adversarial Systems

This module integrates BubbleLabs UI components with:
- Evolution engine (genetic algorithms, population evolution)
- Adversarial testing system (red team/blue team)
- MCTS evolution
- Adversarial maker integration

Key Features:
- Real-time evolution progress display
- Population diversity visualization
- Adversarial attack/defense visualization
- Fitness landscape visualization
- Evolution parameter controls
- Long-running task management with stop/resume

Author: OpenEvolve Frontend Team
"""

import streamlit as st
import time
import threading
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Evolution imports (using centralized import system)
from openevolve_imports import EVOLUTION_AVAILABLE, ADVERSARIAL_AVAILABLE
from evolution import EvolutionConfiguration
from evolution_maker_integration import (
    MAKEREvolutionEngine,
    MakerevolutionConfig,
    MakerevolutionMode,
    Individual,
    Population,
    run_maker_evolution
)

# Adversarial imports (using centralized import system)
from adversarial import AdversarialConfiguration, run_comprehensive_adversarial_testing
from adversarial_maker_integration import (
    AdversarialCoEvolution,
    MAKERRedTeamAgent,
    MDAPBlueTeamAgent,
    AdversarialMAKERConfig,
    run_maker_adversarial_testing
)

# BubbleLabs imports
from bubblelabs_ui_component import BubbleLabsWorkflowUI

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class EvolutionTaskStatus(Enum):
    """Status of evolution/adversarial tasks"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class EvolutionTask:
    """Represents a running evolution or adversarial task"""
    task_id: str
    task_type: str  # "evolution" or "adversarial"
    status: EvolutionTaskStatus = EvolutionTaskStatus.IDLE
    progress: float = 0.0
    current_generation: int = 0
    max_generations: int = 100
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)

    # Evolution-specific
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    population_diversity: float = 0.0
    fitness_history: List[float] = field(default_factory=list)

    # Adversarial-specific
    adversarial_round: int = 0
    attack_success_rate: float = 0.0
    defense_success_rate: float = 0.0
    vulnerability_count: int = 0

    # Thread control
    stop_event: Optional[threading.Event] = field(default=None)
    thread: Optional[threading.Thread] = field(default=None)

    def __post_init__(self):
        if self.stop_event is None:
            self.stop_event = threading.Event()


@dataclass
class EvolutionVisualization:
    """Visualization data for evolution progress"""
    generation: int
    best_fitness: float
    average_fitness: float
    diversity: float
    population_size: int
    timestamp: float


# =============================================================================
# MAIN INTEGRATION CLASS
# =============================================================================

class BubbleLabsEvolutionIntegration:
    """
    Main integration class for BubbleLabs + Evolution/Adversarial systems.

    Provides:
    1. Evolution workflow visualization
    2. Adversarial testing visualization
    3. Real-time progress tracking
    4. Task control (start/stop/pause/resume)
    5. Metrics dashboards
    """

    def __init__(self):
        self.bubblelabs_ui = BubbleLabsWorkflowUI()
        self.active_tasks: Dict[str, EvolutionTask] = {}
        self.task_history: List[EvolutionTask] = []

    def render_evolution_dashboard(self):
        """
        Render the main evolution dashboard with all features.
        """
        st.header("🧬 Evolution & Adversarial Testing Dashboard")

        # Create main tabs
        tabs = st.tabs([
            "Evolution Workflows",
            "Adversarial Testing",
            "Active Tasks",
            "Analytics & Metrics",
            "History & Replay"
        ])

        with tabs[0]:
            self._render_evolution_workflows()

        with tabs[1]:
            self._render_adversarial_testing()

        with tabs[2]:
            self._render_active_tasks()

        with tabs[3]:
            self._render_analytics()

        with tabs[4]:
            self._render_history()

    # =========================================================================
    # EVOLUTION WORKFLOWS
    # =========================================================================

    def _render_evolution_workflows(self):
        """Render evolution workflow configuration and execution"""
        st.subheader("🧬 Evolution Workflows")

        # Workflow type selection
        workflow_types = {
            "standard": "Standard Evolution",
            "maker_voting": "MAKER Voting Evolution",
            "mdap_decomposition": "MDAP Decomposition Evolution",
            "hybrid": "Hybrid MAKER+MDAP Evolution",
            "mcts": "MCTS Evolution"
        }

        selected_type = st.selectbox(
            "Evolution Type",
            options=list(workflow_types.keys()),
            format_func=lambda x: workflow_types[x],
            key="evo_workflow_type"
        )

        # Configuration sections
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Problem Setup")
            initial_content = st.text_area(
                "Initial Content/Program",
                placeholder="Enter initial code, prompt, or content to evolve...",
                height=200,
                key="evo_initial_content"
            )

            content_type = st.selectbox(
                "Content Type",
                options=["code", "text", "markdown", "json", "python"],
                key="evo_content_type"
            )

        with col2:
            st.markdown("### ⚙️ Evolution Parameters")
            max_generations = st.number_input(
                "Max Generations",
                min_value=1,
                max_value=1000,
                value=100,
                key="evo_max_gen"
            )

            population_size = st.number_input(
                "Population Size",
                min_value=2,
                max_value=500,
                value=20,
                key="evo_pop_size"
            )

            mutation_rate = st.slider(
                "Mutation Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                key="evo_mutation_rate"
            )

            crossover_rate = st.slider(
                "Crossover Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.01,
                key="evo_crossover_rate"
            )

        # MAKER-specific configuration
        if selected_type in ["maker_voting", "hybrid"]:
            st.markdown("### 🗳️ MAKER Voting Configuration")
            evo_col1, evo_col2 = st.columns(2)

            with evo_col1:
                enable_voting = st.checkbox(
                    "Enable MAKER Voting",
                    value=True,
                    key="evo_enable_voting"
                )
                voting_threshold = st.number_input(
                    "Voting Threshold (k)",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="First-to-ahead-by-k consensus threshold",
                    key="evo_voting_threshold"
                )

            with evo_col2:
                num_candidates = st.number_input(
                    "Number of Candidates",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="N candidates for voting (N >= 2k-1)",
                    key="evo_num_candidates"
                )
                adaptive_voting = st.checkbox(
                    "Adaptive Voting",
                    value=True,
                    help="Adjust threshold based on diversity",
                    key="evo_adaptive_voting"
                )

        # MDAP-specific configuration
        if selected_type in ["mdap_decomposition", "hybrid"]:
            st.markdown("### 🧩 MDAP Decomposition Configuration")
            mdap_col1, mdap_col2 = st.columns(2)

            with mdap_col1:
                enable_decomposition = st.checkbox(
                    "Enable Task Decomposition",
                    value=True,
                    key="evo_enable_decomp"
                )
                decomposition_depth = st.number_input(
                    "Decomposition Depth",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="evo_decomp_depth"
                )

            with mdap_col2:
                max_subtasks = st.number_input(
                    "Max Subtasks",
                    min_value=1,
                    max_value=50,
                    value=10,
                    key="evo_max_subtasks"
                )

        # Fitness function configuration
        st.markdown("### 📊 Fitness Function")
        fitness_type = st.selectbox(
            "Fitness Function Type",
            options=["custom", "length", "complexity", "readability", "performance"],
            key="evo_fitness_type"
        )

        if fitness_type == "custom":
            custom_fitness = st.text_area(
                "Custom Fitness Function (Python)",
                placeholder="def fitness_fn(content):\n    # Your fitness logic here\n    return score",
                height=150,
                key="evo_custom_fitness"
            )

        # Start evolution button
        col_start, col_info = st.columns([2, 1])

        with col_start:
            if st.button("🚀 Start Evolution", type="primary", key="start_evo_btn"):
                if not initial_content.strip():
                    st.error("Please provide initial content to evolve")
                    return

                # Create task
                task_id = f"evo_{int(time.time())}"
                task = self._create_evolution_task(
                    task_id=task_id,
                    initial_content=initial_content,
                    content_type=content_type,
                    evolution_type=selected_type,
                    config={
                        "max_generations": max_generations,
                        "population_size": population_size,
                        "mutation_rate": mutation_rate,
                        "crossover_rate": crossover_rate,
                        "fitness_type": fitness_type,
                        "enable_voting": enable_voting if selected_type in ["maker_voting", "hybrid"] else False,
                        "voting_threshold": voting_threshold if selected_type in ["maker_voting", "hybrid"] else 3,
                        "enable_decomposition": enable_decomposition if selected_type in ["mdap_decomposition", "hybrid"] else False,
                        "decomposition_depth": decomposition_depth if selected_type in ["mdap_decomposition", "hybrid"] else 3,
                    }
                )

                # Start evolution in background thread
                self._start_evolution_task(task)
                st.success(f"Evolution started! Task ID: {task_id}")

        with col_info:
            st.info(f"Active Tasks: {len([t for t in self.active_tasks.values() if t.status == EvolutionTaskStatus.RUNNING])}")
            st.info(f"Completed: {len(self.task_history)}")

    def _create_evolution_task(
        self,
        task_id: str,
        initial_content: str,
        content_type: str,
        evolution_type: str,
        config: Dict[str, Any]
    ) -> EvolutionTask:
        """Create an evolution task with configuration"""
        return EvolutionTask(
            task_id=task_id,
            task_type="evolution",
            status=EvolutionTaskStatus.IDLE,
            max_generations=config.get("max_generations", 100),
            results={
                "initial_content": initial_content,
                "content_type": content_type,
                "evolution_type": evolution_type,
                "config": config
            }
        )

    def _start_evolution_task(self, task: EvolutionTask):
        """Start evolution task in background thread"""
        task.status = EvolutionTaskStatus.RUNNING
        task.start_time = datetime.now()
        self.active_tasks[task.task_id] = task

        # Create and start thread
        task.thread = threading.Thread(
            target=self._run_evolution_worker,
            args=(task,),
            daemon=True
        )
        task.thread.start()

    def _run_evolution_worker(self, task: EvolutionTask):
        """
        Worker thread that runs the evolution process.

        This function runs in a background thread and updates the task state
        with progress information.
        """
        try:
            logger.info(f"Starting evolution worker for task {task.task_id}")

            # Extract configuration
            config = task.results.get("config", {})
            initial_content = task.results.get("initial_content", "")
            evolution_type = task.results.get("evolution_type", "standard")

            # Define fitness function
            def fitness_fn(content: str) -> float:
                """Simple fitness function - can be customized"""
                if task.stop_event.is_set():
                    return 0.0

                # Basic fitness metrics
                score = 0.0

                # Length score (prefer reasonable length)
                length = len(content)
                if 100 <= length <= 5000:
                    score += 0.3
                elif length > 0:
                    score += 0.1

                # Complexity score (prefer some complexity)
                lines = content.split('\n')
                if len(lines) >= 5:
                    score += 0.3

                # Variety score (prefer diverse content)
                unique_chars = len(set(content))
                if unique_chars > 20:
                    score += 0.2

                # Add some randomness for exploration
                import random
                score += random.random() * 0.2

                return min(score, 1.0)

            # Create MAKER evolution config if needed
            if evolution_type in ["maker_voting", "hybrid"]:
                maker_config = MakerevolutionConfig(
                    mode=MakerevolutionMode.HYBRID if evolution_type == "hybrid" else MakerevolutionMode.VOTING_ONLY,
                    enable_voting=config.get("enable_voting", True),
                    voting_threshold=config.get("voting_threshold", 3),
                    population_size=config.get("population_size", 20),
                    enable_decomposition=config.get("enable_decomposition", False),
                    decomposition_depth=config.get("decomposition_depth", 3)
                )
            else:
                maker_config = MakerevolutionConfig()

            # Run evolution
            results = run_maker_evolution(
                initial_program=initial_content,
                evaluator=fitness_fn,
                max_generations=config.get("max_generations", 100),
                config=maker_config,
                mutation_rate=config.get("mutation_rate", 0.1),
                crossover_rate=config.get("crossover_rate", 0.7)
            )

            # Check if stopped
            if task.stop_event.is_set():
                task.status = EvolutionTaskStatus.STOPPED
                logger.info(f"Evolution task {task.task_id} stopped by user")
                return

            # Update task with results
            task.status = EvolutionTaskStatus.COMPLETED
            task.end_time = datetime.now()
            task.results.update(results)

            # Extract key metrics
            task.best_fitness = results.get("best_fitness", 0.0)
            task.current_generation = results.get("generations", 0)
            task.fitness_history = results.get("fitness_history", [])

            logger.info(f"Evolution task {task.task_id} completed successfully")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Evolution task {task.task_id} failed: {e}", exc_info=True)
            task.status = EvolutionTaskStatus.FAILED
            task.end_time = datetime.now()
            task.error_message = str(e)

    # =========================================================================
    # ADVERSARIAL TESTING
    # =========================================================================

    def _render_adversarial_testing(self):
        """Render adversarial testing configuration and execution"""
        st.subheader("⚔️ Adversarial Testing")

        # Adversarial mode selection
        adversarial_modes = {
            "standard": "Standard Adversarial",
            "maker_red_team": "MAKER Red Team",
            "mdap_blue_team": "MDAP Blue Team",
            "coevolution": "Attack/Defense Coevolution",
            "maker_full": "Full MAKER+MDAP Adversarial"
        }

        selected_mode = st.selectbox(
            "Adversarial Mode",
            options=list(adversarial_modes.keys()),
            format_func=lambda x: adversarial_modes[x],
            key="adv_mode"
        )

        # Configuration
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Target Configuration")
            target_content = st.text_area(
                "Target Content to Test",
                placeholder="Enter code, prompt, or content for adversarial testing...",
                height=200,
                key="adv_target_content"
            )

            content_type = st.selectbox(
                "Content Type",
                options=["document_general", "code", "prompt", "api_response", "model_output"],
                key="adv_content_type"
            )

        with col2:
            st.markdown("### ⚙️ Adversarial Parameters")
            adversarial_rounds = st.number_input(
                "Adversarial Rounds",
                min_value=1,
                max_value=20,
                value=5,
                key="adv_rounds"
            )

            attack_strength = st.slider(
                "Attack Strength",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key="adv_attack_strength"
            )

            defense_strength = st.slider(
                "Defense Strength",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                key="adv_defense_strength"
            )

            coevolution = st.checkbox(
                "Enable Coevolution",
                value=False,
                help="Attack and defense evolve together",
                key="adv_coevolution"
            )

        # Team configuration
        st.markdown("### 👥 Team Configuration")
        team_col1, team_col2 = st.columns(2)

        with team_col1:
            red_team_size = st.number_input(
                "Red Team Size",
                min_value=1,
                max_value=10,
                value=3,
                key="adv_red_team_size"
            )

            st.caption("Red team identifies vulnerabilities and generates attacks")

        with team_col2:
            blue_team_size = st.number_input(
                "Blue Team Size",
                min_value=1,
                max_value=10,
                value=3,
                key="adv_blue_team_size"
            )

            st.caption("Blue team generates defenses and fixes")

        # MAKER/MDAP settings for adversarial
        if selected_mode in ["maker_red_team", "coevolution", "maker_full"]:
            st.markdown("### 🗳️ MAKER Red Team Settings")
            maker_col1, maker_col2 = st.columns(2)

            with maker_col1:
                red_team_voting = st.checkbox(
                    "Enable Red Team Voting",
                    value=True,
                    key="adv_red_voting"
                )
                red_consensus_threshold = st.number_input(
                    "Red Team Consensus (k)",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="adv_red_consensus"
                )

            with maker_col2:
                attack_decomposition = st.checkbox(
                    "Attack Decomposition",
                    value=True,
                    help="Break complex attacks into primitives",
                    key="adv_attack_decomp"
                )

        if selected_mode in ["mdap_blue_team", "coevolution", "maker_full"]:
            st.markdown("### 🧩 MDAP Blue Team Settings")
            mdap_col1, mdap_col2 = st.columns(2)

            with mdap_col1:
                blue_team_decomp = st.checkbox(
                    "Enable Blue Team Decomposition",
                    value=True,
                    key="adv_blue_decomp"
                )
                max_defenses = st.number_input(
                    "Max Defense Strategies",
                    min_value=1,
                    max_value=50,
                    value=10,
                    key="adv_max_defenses"
                )

            with mdap_col2:
                defense_layering = st.checkbox(
                    "Defense Layering",
                    value=True,
                    help="Apply multiple layers of defense",
                    key="adv_defense_layering"
                )

        # Start adversarial testing button
        col_start, col_info = st.columns([2, 1])

        with col_start:
            if st.button("⚔️ Start Adversarial Testing", type="primary", key="start_adv_btn"):
                if not target_content.strip():
                    st.error("Please provide target content for adversarial testing")
                    return

                # Create task
                task_id = f"adv_{int(time.time())}"
                task = self._create_adversarial_task(
                    task_id=task_id,
                    target_content=target_content,
                    content_type=content_type,
                    adversarial_mode=selected_mode,
                    config={
                        "adversarial_rounds": adversarial_rounds,
                        "attack_strength": attack_strength,
                        "defense_strength": defense_strength,
                        "coevolution": coevolution,
                        "red_team_size": red_team_size,
                        "blue_team_size": blue_team_size,
                    }
                )

                # Start adversarial testing in background
                self._start_adversarial_task(task)
                st.success(f"Adversarial testing started! Task ID: {task_id}")

        with col_info:
            st.info(f"Active Tests: {len([t for t in self.active_tasks.values() if t.status == EvolutionTaskStatus.RUNNING and t.task_type == 'adversarial'])}")

    def _create_adversarial_task(
        self,
        task_id: str,
        target_content: str,
        content_type: str,
        adversarial_mode: str,
        config: Dict[str, Any]
    ) -> EvolutionTask:
        """Create an adversarial testing task"""
        return EvolutionTask(
            task_id=task_id,
            task_type="adversarial",
            status=EvolutionTaskStatus.IDLE,
            max_generations=config.get("adversarial_rounds", 5),
            results={
                "target_content": target_content,
                "content_type": content_type,
                "adversarial_mode": adversarial_mode,
                "config": config
            }
        )

    def _start_adversarial_task(self, task: EvolutionTask):
        """Start adversarial task in background thread"""
        task.status = EvolutionTaskStatus.RUNNING
        task.start_time = datetime.now()
        self.active_tasks[task.task_id] = task

        # Create and start thread
        task.thread = threading.Thread(
            target=self._run_adversarial_worker,
            args=(task,),
            daemon=True
        )
        task.thread.start()

    def _run_adversarial_worker(self, task: EvolutionTask):
        """
        Worker thread that runs adversarial testing.

        This function runs in a background thread and updates the task state
        with adversarial testing progress.
        """
        try:
            logger.info(f"Starting adversarial worker for task {task.task_id}")

            # Extract configuration
            config = task.results.get("config", {})
            target_content = task.results.get("target_content", "")
            content_type = task.results.get("content_type", "document_general")
            adversarial_mode = task.results.get("adversarial_mode", "standard")

            # Create adversarial configuration
            adv_config = AdversarialConfiguration(
                adversarial_rounds=config.get("adversarial_rounds", 5),
                attack_strength=config.get("attack_strength", 0.5),
                defense_strategy="reactive",
                coevolutionary_approach=config.get("coevolution", False),
                red_team_sample_size=config.get("red_team_size", 3),
                blue_team_sample_size=config.get("blue_team_size", 3),
                adversarial_temperature=0.8,
                ensemble_defense=True
            )

            # Run adversarial testing
            if adversarial_mode == "maker_full":
                # Use MAKER-enhanced adversarial testing
                results = run_maker_adversarial_testing(
                    content=target_content,
                    content_type=content_type,
                    config=adv_config
                )
            else:
                # Use standard adversarial testing
                results = run_comprehensive_adversarial_testing(
                    current_content=target_content,
                    content_type=content_type,
                    config=adv_config
                )

            # Check if stopped
            if task.stop_event.is_set():
                task.status = EvolutionTaskStatus.STOPPED
                logger.info(f"Adversarial task {task.task_id} stopped by user")
                return

            # Update task with results
            task.status = EvolutionTaskStatus.COMPLETED
            task.end_time = datetime.now()
            task.results.update(results)

            # Extract key metrics
            metrics = results.get("metrics", {})
            task.attack_success_rate = metrics.get("attack_success_rate", 0.0)
            task.defense_success_rate = metrics.get("defense_success_rate", 0.0)
            task.vulnerability_count = metrics.get("vulnerability_count", 0)
            task.adversarial_round = metrics.get("total_rounds", 0)

            logger.info(f"Adversarial task {task.task_id} completed successfully")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Adversarial task {task.task_id} failed: {e}", exc_info=True)
            task.status = EvolutionTaskStatus.FAILED
            task.end_time = datetime.now()
            task.error_message = str(e)

    # =========================================================================
    # ACTIVE TASKS MONITORING
    # =========================================================================

    def _render_active_tasks(self):
        """Render active tasks with real-time progress"""
        st.subheader("🔄 Active Tasks")

        # Filter running tasks
        running_tasks = [t for t in self.active_tasks.values() if t.status == EvolutionTaskStatus.RUNNING]

        if not running_tasks:
            st.info("No active tasks. Start an evolution or adversarial test from the other tabs.")
            return

        # Display each task
        for task in running_tasks:
            with st.expander(f"📊 {task.task_type.capitalize()} Task - {task.task_id}", expanded=True):
                # Task info
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Status", task.status.value)
                    if task.start_time:
                        elapsed = (datetime.now() - task.start_time).total_seconds()
                        st.caption(f"Elapsed: {elapsed:.1f}s")

                with col2:
                    if task.task_type == "evolution":
                        st.metric("Generation", f"{task.current_generation}/{task.max_generations}")
                        st.metric("Best Fitness", f"{task.best_fitness:.4f}")
                    else:  # adversarial
                        st.metric("Round", f"{task.adversarial_round}/{task.max_generations}")
                        st.metric("Vulnerabilities", task.vulnerability_count)

                with col3:
                    # Control buttons
                    stop_btn = st.button("⏹️ Stop", key=f"stop_{task.task_id}")
                    if stop_btn:
                        self._stop_task(task)
                        st.rerun()

                # Progress visualization
                if task.task_type == "evolution" and task.fitness_history:
                    st.markdown("#### 📈 Fitness Progress")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=task.fitness_history,
                        mode='lines+markers',
                        name='Best Fitness',
                        line=dict(color='green', width=2)
                    ))
                    fig.update_layout(
                        title="Fitness Over Generations",
                        xaxis_title="Generation",
                        yaxis_title="Fitness",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Auto-refresh
                time.sleep(0.5)
                st.rerun()

    def _stop_task(self, task: EvolutionTask):
        """Stop a running task"""
        if task.stop_event:
            task.stop_event.set()
        task.status = EvolutionTaskStatus.STOPPED
        task.end_time = datetime.now()

        # Move to history
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]
        self.task_history.append(task)

        st.success(f"Task {task.task_id} stopped")

    # =========================================================================
    # ANALYTICS & METRICS
    # =========================================================================

    def _render_analytics(self):
        """Render analytics and metrics dashboard"""
        st.subheader("📊 Analytics & Metrics")

        # Get all completed tasks
        completed_tasks = [t for t in self.task_history if t.status == EvolutionTaskStatus.COMPLETED]

        if not completed_tasks:
            st.info("No completed tasks to analyze yet.")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Tasks", len(completed_tasks))

        with col2:
            evo_tasks = [t for t in completed_tasks if t.task_type == "evolution"]
            avg_fitness = sum(t.best_fitness for t in evo_tasks) / len(evo_tasks) if evo_tasks else 0
            st.metric("Avg Best Fitness", f"{avg_fitness:.4f}")

        with col3:
            adv_tasks = [t for t in completed_tasks if t.task_type == "adversarial"]
            total_vulns = sum(t.vulnerability_count for t in adv_tasks)
            st.metric("Total Vulnerabilities Found", total_vulns)

        with col4:
            st.metric("Success Rate", f"{len(completed_tasks) / (len(completed_tasks) + len([t for t in self.task_history if t.status == EvolutionTaskStatus.FAILED])) * 100:.1f}%")

        # Evolution analytics
        if evo_tasks:
            st.markdown("### 🧬 Evolution Analytics")

            # Fitness comparison
            fig = go.Figure()
            for task in evo_tasks[:5]:  # Show top 5
                fig.add_trace(go.Scatter(
                    y=task.fitness_history,
                    mode='lines',
                    name=f"{task.task_id[:8]}",
                ))

            fig.update_layout(
                title="Fitness History Comparison",
                xaxis_title="Generation",
                yaxis_title="Fitness",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Adversarial analytics
        if adv_tasks:
            st.markdown("### ⚔️ Adversarial Analytics")

            # Vulnerability distribution
            vuln_data = [
                {
                    "Task ID": t.task_id[:8],
                    "Vulnerabilities": t.vulnerability_count,
                    "Attack Success Rate": t.attack_success_rate,
                    "Defense Success Rate": t.defense_success_rate
                }
                for t in adv_tasks
            ]

            df = pd.DataFrame(vuln_data)
            st.dataframe(df, use_container_width=True)

            # Success rate comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[t.task_id[:8] for t in adv_tasks],
                y=[t.attack_success_rate for t in adv_tasks],
                name="Attack Success Rate",
                marker_color='red'
            ))
            fig.add_trace(go.Bar(
                x=[t.task_id[:8] for t in adv_tasks],
                y=[t.defense_success_rate for t in adv_tasks],
                name="Defense Success Rate",
                marker_color='blue'
            ))

            fig.update_layout(
                title="Attack vs Defense Success Rates",
                xaxis_title="Task",
                yaxis_title="Success Rate",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # HISTORY & REPLAY
    # =========================================================================

    def _render_history(self):
        """Render task history and replay functionality"""
        st.subheader("📜 Task History")

        if not self.task_history:
            st.info("No task history yet.")
            return

        # Task history table
        history_data = []
        for task in self.task_history:
            history_data.append({
                "Task ID": task.task_id,
                "Type": task.task_type.capitalize(),
                "Status": task.status.value,
                "Duration": f"{(task.end_time - task.start_time).total_seconds():.1f}s" if task.start_time and task.end_time else "N/A",
                "Result": f"{task.best_fitness:.4f}" if task.task_type == "evolution" else f"{task.vulnerability_count} vulns"
            })

        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)

        # Task details viewer
        st.markdown("### 🔍 Task Details")

        selected_task_id = st.selectbox(
            "Select Task to View",
            options=[t.task_id for t in self.task_history],
            format_func=lambda x: f"{x[:8]}... ({[t for t in self.task_history if t.task_id == x][0].task_type})",
            key="history_task_selector"
        )

        if selected_task_id:
            task = next((t for t in self.task_history if t.task_id == selected_task_id), None)
            if task:
                self._display_task_details(task)

    def _display_task_details(self, task: EvolutionTask):
        """Display detailed information about a task"""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📋 Task Information")
            st.json({
                "Task ID": task.task_id,
                "Type": task.task_type,
                "Status": task.status.value,
                "Start Time": task.start_time.isoformat() if task.start_time else None,
                "End Time": task.end_time.isoformat() if task.end_time else None,
                "Error": task.error_message
            })

        with col2:
            if task.task_type == "evolution":
                st.markdown("#### 🧬 Evolution Metrics")
                st.json({
                    "Best Fitness": task.best_fitness,
                    "Generations": task.current_generation,
                    "Max Generations": task.max_generations,
                    "Population Size": task.results.get("config", {}).get("population_size", "N/A")
                })
            else:
                st.markdown("#### ⚔️ Adversarial Metrics")
                st.json({
                    "Vulnerabilities Found": task.vulnerability_count,
                    "Attack Success Rate": task.attack_success_rate,
                    "Defense Success Rate": task.defense_success_rate,
                    "Rounds Completed": task.adversarial_round
                })

        # Results
        if task.results:
            st.markdown("#### 📊 Results")
            with st.expander("View Full Results", expanded=False):
                st.json(task.results)


# =============================================================================
# STREAMLIT PAGE FUNCTIONS
# =============================================================================

def main():
    """Main Streamlit page function"""
    st.set_page_config(
        page_title="BubbleLabs Evolution Integration",
        page_icon="🧬",
        layout="wide"
    )

    st.title("🧬 BubbleLabs x Evolution/Adversarial Integration")

    # Initialize integration
    if "evolution_integration" not in st.session_state:
        st.session_state.evolution_integration = BubbleLabsEvolutionIntegration()

    integration = st.session_state.evolution_integration

    # Render dashboard
    integration.render_evolution_dashboard()


if __name__ == "__main__":
    main()
