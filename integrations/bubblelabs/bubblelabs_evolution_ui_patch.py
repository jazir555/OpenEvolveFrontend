"""
BubbleLabs UI Evolution/Adversarial Tab Extension

This file provides extension methods to add Evolution and Adversarial
tabs to the existing BubbleLabs UI component.

Usage:
    from bubblelabs_ui_component import BubbleLabsWorkflowUI
    from bubblelabs_evolution_ui_patch import extend_bubblelabs_ui

    ui = BubbleLabsWorkflowUI()
    extend_bubblelabs_ui(ui)

Author: OpenEvolve Frontend Team
"""
from __future__ import annotations


try:
    from .ui_shim import ui
except ImportError:
    from ui_shim import ui
from typing import Optional


def extend_bubblelabs_ui(ui_instance):
    """
    Extend BubbleLabs UI with Evolution and Adversarial tabs.

    This modifies the existing BubbleLabsWorkflowUI instance to add
    evolution and adversarial testing capabilities.

    Args:
        ui_instance: Instance of BubbleLabsWorkflowUI to extend
    """

    # Store original render method
    original_render = ui_instance.render_workflow_visualizer

    def extended_render():
        """Extended render method with Evolution/Adversarial tabs"""
        ui.header("🧬 OpenEvolve Workflows in BubbleLabs")
        ui.markdown("""
        Visualize, interact with, and control OpenEvolve sovereign-grade decomposition,
        evolution, and adversarial testing workflows through the BubbleLabs interface.
        """)

        # Create tabs with new additions
        tabs = ui.tabs([
            "Workflow Designer",
            "Evolution Workflows",
            "Adversarial Testing",
            "Active Tasks",
            "Analytics",
            "Workflow Control",
            "Global Parameters"
        ])

        with tabs[0]:
            ui_instance._render_workflow_designer()

        with tabs[1]:
            _render_evolution_workflows_tab(ui_instance)

        with tabs[2]:
            _render_adversarial_testing_tab(ui_instance)

        with tabs[3]:
            _render_active_tasks_tab()

        with tabs[4]:
            _render_analytics_tab()

        with tabs[5]:
            ui_instance._render_workflow_control()

        with tabs[6]:
            ui_instance._render_global_parameters()

    # Replace render method
    ui_instance.render_workflow_visualizer = extended_render

    return ui_instance


def _render_evolution_workflows_tab(ui_instance):
    """Render Evolution Workflows tab"""
    try:
        from .bubblelabs_evolution_controls import EvolutionControlPanel, PopulationVisualizer
    except ImportError:
        from bubblelabs_evolution_controls import EvolutionControlPanel, PopulationVisualizer

    ui.subheader("🧬 Evolution Workflows")

    # Evolution type selection
    workflow_types = {
        "standard": "Standard Evolution",
        "maker_voting": "MAKER Voting Evolution",
        "mdap_decomposition": "MDAP Decomposition Evolution",
        "hybrid": "Hybrid MAKER+MDAP Evolution",
    }

    selected_type = ui.selectbox(
        "Evolution Type",
        options=list(workflow_types.keys()),
        format_func=lambda x: workflow_types[x],
        key="bl_evo_workflow_type"
    )

    # Problem setup
    col1, col2 = ui.columns(2)

    with col1:
        ui.markdown("### 🎯 Problem Setup")
        initial_content = ui.text_area(
            "Initial Content/Program",
            placeholder="Enter initial code, prompt, or content to evolve...",
            height=200,
            key="bl_evo_initial_content"
        )

        content_type = ui.selectbox(
            "Content Type",
            options=["code", "text", "markdown", "json", "python"],
            key="bl_evo_content_type"
        )

    with col2:
        # Use evolution control panel
        control_panel = EvolutionControlPanel()
        state = control_panel.render(key_prefix="bl_evo")

    # Start button
    if ui.button("🚀 Start Evolution", type="primary", key="bl_start_evo"):
        if not initial_content.strip():
            ui.error("Please provide initial content to evolve")
            return

        ui.success(f"Evolution started with {selected_type} mode!")
        ui.info("Monitor progress in the 'Active Tasks' tab")


def _render_adversarial_testing_tab(ui_instance):
    """Render Adversarial Testing tab"""
    try:
        from .bubblelabs_evolution_controls import AdversarialControlPanel
    except ImportError:
        from bubblelabs_evolution_controls import AdversarialControlPanel

    ui.subheader("⚔️ Adversarial Testing")

    # Mode selection
    adversarial_modes = {
        "standard": "Standard Adversarial",
        "maker_red_team": "MAKER Red Team",
        "mdap_blue_team": "MDAP Blue Team",
        "coevolution": "Attack/Defense Coevolution",
        "maker_full": "Full MAKER+MDAP Adversarial"
    }

    selected_mode = ui.selectbox(
        "Adversarial Mode",
        options=list(adversarial_modes.keys()),
        format_func=lambda x: adversarial_modes[x],
        key="bl_adv_mode"
    )

    # Configuration
    col1, col2 = ui.columns(2)

    with col1:
        ui.markdown("### 🎯 Target Configuration")
        target_content = ui.text_area(
            "Target Content to Test",
            placeholder="Enter code, prompt, or content for adversarial testing...",
            height=200,
            key="bl_adv_target_content"
        )

        content_type = ui.selectbox(
            "Content Type",
            options=["document_general", "code", "prompt", "api_response"],
            key="bl_adv_content_type"
        )

    with col2:
        # Use adversarial control panel
        control_panel = AdversarialControlPanel()
        config = control_panel.render(key_prefix="bl_adv")

    # Start button
    if ui.button("⚔️ Start Adversarial Testing", type="primary", key="bl_start_adv"):
        if not target_content.strip():
            ui.error("Please provide target content for adversarial testing")
            return

        ui.success(f"Adversarial testing started with {selected_mode} mode!")
        ui.info("Monitor progress in the 'Active Tasks' tab")


def _render_active_tasks_tab():
    """Render Active Tasks monitoring tab"""
    ui.subheader("🔄 Active Tasks")

    # Check for active tasks in session state
    if "evolution_tasks" not in ui.session_state:
        ui.session_state.evolution_tasks = {}

    active_tasks = ui.session_state.evolution_tasks

    if not active_tasks:
        ui.info("No active tasks. Start an evolution or adversarial test from the other tabs.")
        return

    # Display tasks
    for task_id, task_data in active_tasks.items():
        with ui.expander(f"📊 Task {task_id}", expanded=True):
            col1, col2, col3 = ui.columns(3)

            with col1:
                ui.metric("Status", task_data.get("status", "unknown"))

            with col2:
                if task_data.get("type") == "evolution":
                    ui.metric("Generation", f"{task_data.get('current_gen', 0)}/{task_data.get('max_gen', 0)}")
                else:
                    ui.metric("Round", f"{task_data.get('current_round', 0)}/{task_data.get('max_rounds', 0)}")

            with col3:
                if ui.button("⏹️ Stop", key=f"stop_{task_id}"):
                    del active_tasks[task_id]
                    ui.rerun()


def _render_analytics_tab():
    """Render Analytics dashboard tab"""
    ui.subheader("📊 Analytics & Metrics")

    # Check for task history
    if "evolution_history" not in ui.session_state:
        ui.session_state.evolution_history = []

    history = ui.session_state.evolution_history

    if not history:
        ui.info("No completed tasks to analyze yet.")
        return

    # Summary metrics
    col1, col2, col3 = ui.columns(3)

    with col1:
        evo_tasks = [t for t in history if t.get("type") == "evolution"]
        ui.metric("Evolution Tasks", len(evo_tasks))

    with col2:
        adv_tasks = [t for t in history if t.get("type") == "adversarial"]
        ui.metric("Adversarial Tasks", len(adv_tasks))

    with col3:
        success_rate = len([t for t in history if t.get("status") == "completed"])
        ui.metric("Success Rate", f"{success_rate / len(history) * 100:.1f}%")

    # Detailed results
    ui.markdown("### 📈 Performance History")

    import pandas as pd
    df = pd.DataFrame(history)

    if not df.empty:
        ui.dataframe(df, use_container_width=True)


# =============================================================================
# STANDALONE RENDER FUNCTIONS
# =============================================================================

def render_evolution_dashboard_standalone():
    """
    Render standalone evolution dashboard.

    Use this to render the evolution features independently
    of the main BubbleLabs UI.
    """
    try:
        from .bubblelabs_evolution_integration import BubbleLabsEvolutionIntegration
    except ImportError:
        from bubblelabs_evolution_integration import BubbleLabsEvolutionIntegration

    integration = BubbleLabsEvolutionIntegration()
    integration.render_evolution_dashboard()


def render_evolution_controls_standalone():
    """
    Render standalone evolution controls.

    Use this to embed evolution controls in any UI app.
    """
    try:
        from .bubblelabs_evolution_controls import EvolutionControlPanel
    except ImportError:
        from bubblelabs_evolution_controls import EvolutionControlPanel

    ui.header("🧬 Evolution Controls")

    panel = EvolutionControlPanel()
    state = panel.render()

    if ui.button("Apply Configuration"):
        ui.json(state.__dict__)


def render_adversarial_controls_standalone():
    """
    Render standalone adversarial controls.

    Use this to embed adversarial controls in any UI app.
    """
    try:
        from .bubblelabs_evolution_controls import AdversarialControlPanel
    except ImportError:
        from bubblelabs_evolution_controls import AdversarialControlPanel

    ui.header("⚔️ Adversarial Testing Controls")

    panel = AdversarialControlPanel()
    config = panel.render()

    if ui.button("Apply Configuration"):
        ui.json(config)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_extended_bubblelabs():
    """Example of using extended BubbleLabs UI"""

    # Import BubbleLabs
    try:
        from .bubblelabs_ui_component import BubbleLabsWorkflowUI
    except ImportError:
        from bubblelabs_ui_component import BubbleLabsWorkflowUI

    # Create UI instance
    ui = BubbleLabsWorkflowUI()

    # Extend with evolution features
    extend_bubblelabs_ui(ui)

    # Render extended UI
    ui.render_workflow_visualizer()


def example_embedded_evolution():
    """Example of embedding evolution controls in custom app"""

    ui.title("My Custom App")

    # Add evolution controls
    with ui.sidebar:
        render_evolution_controls_standalone()

    # Main content
    ui.markdown("## My Content")

    # Add adversarial controls
    with ui.expander("Security Testing"):
        render_adversarial_controls_standalone()


# =============================================================================
# UI PAGE FUNCTIONS
# =============================================================================

def page_extended_bubblelabs():
    """UI page for extended BubbleLabs"""
    ui.set_page_config(
        page_title="Extended BubbleLabs",
        page_icon="🧬",
        layout="wide"
    )

    example_extended_bubblelabs()


def page_evolution_standalone():
    """UI page for standalone evolution dashboard"""
    ui.set_page_config(
        page_title="Evolution Dashboard",
        page_icon="🧬",
        layout="wide"
    )

    render_evolution_dashboard_standalone()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "extended":
            page_extended_bubblelabs()
        elif mode == "evolution":
            page_evolution_standalone()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: extended, evolution")
    else:
        # Default to extended BubbleLabs
        page_extended_bubblelabs()

