"""
OpenEvolve + BubbleLabs UI Integration

This module connects the properly integrated OpenEvolve workflow manager
to the BubbleLabs UI, enabling visual workflow creation and control.

Author: OpenEvolve Team
Date: 2025-12-30
"""

import streamlit as st
import json
import time
from typing import Dict, Any, List, Optional

# Import the properly integrated workflow manager
from openevolve_workflow_manager_integrated import OpenEvolveWorkflowManager
from bubblelabs_ui_component import BubbleLabsWorkflowUI
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from bubblelabs_analytics import BubbleLabsAnalytics
from bubblelabs_crewai_bridge import BubbleLabsTicketConfig

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state for OpenEvolve workflows."""
    if 'openevolve_workflow_manager' not in st.session_state:
        st.session_state.openevolve_workflow_manager = OpenEvolveWorkflowManager(
            analytics_db_path='openevolve_analytics.db',
            enable_CREWAI=True
        )

    if 'selected_workflow_id' not in st.session_state:
        st.session_state.selected_workflow_id = None

    if 'workflow_execution_results' not in st.session_state:
        st.session_state.workflow_execution_results = {}


# =============================================================================
# OPENEVOLVE WORKFLOW UI COMPONENTS
# =============================================================================

class OpenEvolveBubbleLabsUI:
    """
    UI component for OpenEvolve + BubbleLabs integration.
    """

    def __init__(self):
        init_session_state()
        self.workflow_manager = st.session_state.openevolve_workflow_manager
        self.team_manager = TeamManager()
        self.gauntlet_manager = GauntletManager()

    def render(self):
        """Render the complete OpenEvolve + BubbleLabs UI."""
        st.title("🔬 OpenEvolve Workflow Manager")

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Create Workflow",
            "▶️ Execute Workflow",
            "📊 Monitor & Control",
            "📈 Analytics"
        ])

        with tab1:
            self._render_workflow_creation()

        with tab2:
            self._render_workflow_execution()

        with tab3:
            self._render_workflow_monitoring()

        with tab4:
            self._render_analytics_dashboard()

    # =========================================================================
    # WORKFLOW CREATION TAB
    # =========================================================================

    def _render_workflow_creation(self):
        """Render workflow creation interface."""
        st.header("Create Sovereign Decomposition Workflow")
        st.markdown("Configure and create a new OpenEvolve workflow using actual teams and gauntlets.")

        with st.form("create_workflow_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Basic Information")
                workflow_name = st.text_input(
                    "Workflow Name",
                    placeholder="My Optimization Workflow",
                    help="A descriptive name for this workflow"
                )

                problem_statement = st.text_area(
                    "Problem Statement",
                    placeholder="Describe the problem you want to solve...",
                    height=150,
                    help="Clear description of the problem to be solved"
                )

            with col2:
                st.subheader("Team Configuration")

                # Get available teams
                teams = self.team_manager.list_teams()
                team_names = [team.name for team in teams]

                if team_names:
                    content_analyzer_team = st.selectbox(
                        "Content Analyzer Team",
                        options=team_names,
                        help="Team that analyzes the problem statement (Stage 0)"
                    )

                    planner_team = st.selectbox(
                        "Planner Team",
                        options=team_names,
                        help="Team that decomposes the problem (Stage 1)"
                    )

                    solver_team = st.selectbox(
                        "Solver Team",
                        options=team_names,
                        help="Team that solves sub-problems (Stage 2)"
                    )

                    assembler_team = st.selectbox(
                        "Assembler Team",
                        options=team_names,
                        help="Team that assembles final solution (Stage 3)"
                    )
                else:
                    st.warning("No teams configured. Please create teams in the Team Manager first.")
                    st.stop()

            st.subheader("Gauntlet Configuration")

            # Get available gauntlets
            gauntlets = self.gauntlet_manager.list_gauntlets()
            gauntlet_names = [g.name for g in gauntlets]

            col3, col4 = st.columns(2)

            with col3:
                sub_problem_red_gauntlet = st.selectbox(
                    "Sub-Problem Red Team Gauntlet",
                    options=[None] + gauntlet_names,
                    help="Red team for sub-problem verification (optional)"
                )

                sub_problem_gold_gauntlet = st.selectbox(
                    "Sub-Problem Gold Team Gauntlet",
                    options=[None] + gauntlet_names,
                    help="Gold team for sub-problem verification (optional)"
                )

            with col4:
                final_red_gauntlet = st.selectbox(
                    "Final Red Team Gauntlet",
                    options=[None] + gauntlet_names,
                    help="Red team for final solution verification (optional)"
                )

                final_gold_gauntlet = st.selectbox(
                    "Final Gold Team Gauntlet",
                    options=[None] + gauntlet_names,
                    help="Gold team for final solution verification (optional)"
                )

            st.subheader("Advanced Options")

            col5, col6 = st.columns(2)

            with col5:
                mdap_enabled = st.checkbox(
                    "Enable MDAP Workflow",
                    help="Enable MDAP (Multi-Domain Analysis Pipeline) workflow"
                )

            with col6:
                maker_enabled = st.checkbox(
                    "Enable Maker Workflow",
                    help="Enable Maker workflow for code generation"
                )

            # Submit button
            submitted = st.form_submit_button("Create Workflow", type="primary")

            if submitted:
                if not workflow_name or not problem_statement:
                    st.error("❌ Workflow name and problem statement are required!")
                else:
                    self._create_workflow(
                        workflow_name=workflow_name,
                        problem_statement=problem_statement,
                        content_analyzer_team=content_analyzer_team,
                        planner_team=planner_team,
                        solver_team=solver_team,
                        assembler_team=assembler_team,
                        sub_problem_red_gauntlet=sub_problem_red_gauntlet,
                        sub_problem_gold_gauntlet=sub_problem_gold_gauntlet,
                        final_red_gauntlet=final_red_gauntlet,
                        final_gold_gauntlet=final_gold_gauntlet,
                        mdap_enabled=mdap_enabled,
                        maker_enabled=maker_enabled
                    )

    def _create_workflow(
        self,
        workflow_name: str,
        problem_statement: str,
        content_analyzer_team: str,
        planner_team: str,
        solver_team: str,
        assembler_team: str,
        sub_problem_red_gauntlet: Optional[str],
        sub_problem_gold_gauntlet: Optional[str],
        final_red_gauntlet: Optional[str],
        final_gold_gauntlet: Optional[str],
        mdap_enabled: bool,
        maker_enabled: bool
    ):
        """Create the workflow with proper error handling."""
        try:
            with st.spinner("Creating workflow..."):
                workflow_id = self.workflow_manager.create_sovereign_workflow(
                    name=workflow_name,
                    problem_statement=problem_statement,
                    content_analyzer_team=content_analyzer_team,
                    planner_team=planner_team,
                    solver_team=solver_team,
                    assembler_team=assembler_team,
                    sub_problem_red_gauntlet=sub_problem_red_gauntlet,
                    sub_problem_gold_gauntlet=sub_problem_gold_gauntlet,
                    final_red_gauntlet=final_red_gauntlet,
                    final_gold_gauntlet=final_gold_gauntlet,
                    mdap_enabled=mdap_enabled,
                    maker_enabled=maker_enabled
                )

                st.success(f"✅ Workflow created successfully! ID: `{workflow_id[:8]}...`")
                st.info(f"💡 You can now execute this workflow in the 'Execute Workflow' tab.")
                st.session_state.selected_workflow_id = workflow_id

                # Show workflow details
                with st.expander("📋 Workflow Details", expanded=False):
                    st.json({
                        "workflow_id": workflow_id,
                        "name": workflow_name,
                        "problem_statement": problem_statement,
                        "teams": {
                            "content_analyzer": content_analyzer_team,
                            "planner": planner_team,
                            "solver": solver_team,
                            "assembler": assembler_team
                        },
                        "gauntlets": {
                            "sub_problem_red": sub_problem_red_gauntlet,
                            "sub_problem_gold": sub_problem_gold_gauntlet,
                            "final_red": final_red_gauntlet,
                            "final_gold": final_gold_gauntlet
                        },
                        "advanced": {
                            "mdap_enabled": mdap_enabled,
                            "maker_enabled": maker_enabled
                        }
                    })

        except ValueError as e:
            st.error(f"❌ Error creating workflow: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

    # =========================================================================
    # WORKFLOW EXECUTION TAB
    # =========================================================================

    def _render_workflow_execution(self):
        """Render workflow execution interface."""
        st.header("Execute Workflow")
        st.markdown("Select and execute a workflow.")

        # List available workflows
        workflows = self.workflow_manager.list_workflows()

        if not workflows:
            st.info("📭 No workflows available. Please create a workflow first.")
            return

        # Workflow selection
        workflow_options = {
            f"{wf['id'][:8]}... - {wf['problem_statement'][:50]}": wf['id']
            for wf in workflows
        }

        selected = st.selectbox(
            "Select Workflow",
            options=list(workflow_options.keys()),
            help="Choose a workflow to execute"
        )

        if selected:
            workflow_id = workflow_options[selected]
            workflow_state = self.workflow_manager.workflow_states.get(workflow_id)

            # Show workflow details
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Workflow Configuration")

                if workflow_state:
                    st.write(f"**Problem:** {workflow_state.problem_statement}")
                    st.write(f"**Status:** {workflow_state.status}")
                    st.write(f"**Progress:** {workflow_state.progress*100:.1f}%")

                    if workflow_state.decomposition_plan:
                        st.write(f"**Sub-Problems:** {len(workflow_state.decomposition_plan.sub_problems)}")

            with col2:
                st.subheader("Teams")
                if workflow_state:
                    if workflow_state.content_analyzer_team:
                        st.write(f"🔵 **Content Analyzer:** {workflow_state.content_analyzer_team.name}")
                    if workflow_state.planner_team:
                        st.write(f"📋 **Planner:** {workflow_state.planner_team.name}")
                    if workflow_state.solver_team:
                        st.write(f"🔧 **Solver:** {workflow_state.solver_team.name}")
                    if workflow_state.assembler_team:
                        st.write(f"🔨 **Assembler:** {workflow_state.assembler_team.name}")

            st.divider()

            # Execute button
            col1, col2, col3 = st.columns(3)

            with col1:
                execute_button = st.button("▶️ Execute Workflow", type="primary", use_container_width=True)

            with col2:
                if workflow_state and workflow_state.status == "running":
                    st.button("⏸️ Pause", use_container_width=True)

            with col3:
                if workflow_state and workflow_state.status == "paused":
                    st.button("▶️ Resume", use_container_width=True)
                elif workflow_state and workflow_state.status in ["running", "pending", "created"]:
                    st.button("⏹️ Cancel", use_container_width=True)

            if execute_button:
                self._execute_workflow(workflow_id)

    def _execute_workflow(self, workflow_id: str):
        """Execute workflow with progress tracking."""
        try:
            # Create progress placeholder
            progress_bar = st.progress(0)
            status_text = st.empty()
            result_placeholder = st.empty()

            with st.spinner("Executing workflow... This may take several minutes."):
                # Execute workflow (synchronous for now)
                result = self.workflow_manager.execute_workflow(workflow_id)

                # Update UI based on result
                if result.success:
                    st.success(f"✅ Workflow completed successfully!")
                    st.info(f"⏱️ Execution time: {result.execution_time:.2f} seconds")

                    # Show results
                    with result_placeholder.container():
                        st.subheader("📊 Results")

                        # Tabs for different result sections
                        rtab1, rtab2, rtab3 = st.tabs(["Final Solution", "Decomposition", "Sub-Problem Solutions"])

                        with rtab1:
                            if result.result and result.result.get('final_solution'):
                                sol = result.result['final_solution']
                                st.write(f"**Solution:** {sol.solution_text}")
                                st.write(f"**Confidence:** {sol.confidence:.2%}")

                        with rtab2:
                            if result.result and result.result.get('decomposition_plan'):
                                plan = result.result['decomposition_plan']
                                st.write(f"**Sub-Problems:** {len(plan.sub_problems)}")
                                for sp in plan.sub_problems:
                                    st.write(f"- **{sp.id}**: {sp.description}")
                                    if sp.dependencies:
                                        st.write(f"  Dependencies: {', '.join(sp.dependencies)}")

                        with rtab3:
                            if result.result and result.result.get('sub_problem_solutions'):
                                solutions = result.result['sub_problem_solutions']
                                st.write(f"**Solutions:** {len(solutions)}")
                                for sp_id, sol in solutions.items():
                                    st.write(f"- **{sp_id}**: {sol.solution_text}")
                                    st.write(f"  Confidence: {sol.confidence:.2%}")

                    # Store in session state
                    st.session_state.workflow_execution_results[workflow_id] = result

                else:
                    st.error(f"❌ Workflow failed: {result.error}")

                    if result.error:
                        with st.expander("Error Details"):
                            st.code(result.error)

        except Exception as e:
            st.error(f"❌ Error executing workflow: {e}")
            st.exception(e)

    # =========================================================================
    # WORKFLOW MONITORING TAB
    # =========================================================================

    def _render_workflow_monitoring(self):
        """Render workflow monitoring and control interface."""
        st.header("Monitor & Control Workflows")
        st.markdown("Track workflow progress and control execution.")

        # List all workflows with their status
        workflows = self.workflow_manager.list_workflows()

        if not workflows:
            st.info("📭 No workflows to monitor.")
            return

        # Display workflows in a table
        for wf in workflows:
            workflow_id = wf['id']
            status = self.workflow_manager.get_workflow_status(workflow_id)

            with st.expander(f"{'🟢' if wf['status'] == 'completed' else '🔵' if wf['status'] == 'running' else '⚪'} {wf['id'][:8]}... - {wf['problem_statement'][:50]}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Status:** {wf['status']}")
                    st.write(f"**Stage:** {wf.get('current_stage', 'N/A')}")

                with col2:
                    st.write(f"**Progress:** {wf['progress']*100:.1f}%")
                    st.progress(wf['progress'])

                with col3:
                    # Control buttons
                    if wf['status'] == 'running':
                        if st.button(f"⏸️ Pause", key=f"pause_{workflow_id}"):
                            self.workflow_manager.pause_workflow(workflow_id)
                            st.rerun()

                    elif wf['status'] == 'paused':
                        if st.button(f"▶️ Resume", key=f"resume_{workflow_id}"):
                            self.workflow_manager.resume_workflow(workflow_id)
                            st.rerun()

                    if wf['status'] in ['running', 'paused', 'pending', 'created']:
                        if st.button(f"⏹️ Cancel", key=f"cancel_{workflow_id}"):
                            self.workflow_manager.cancel_workflow(workflow_id)
                            st.rerun()

                # Show detailed status
                if status:
                    with st.expander("📊 Detailed Status"):
                        st.json(status)

    # =========================================================================
    # ANALYTICS DASHBOARD TAB
    # =========================================================================

    def _render_analytics_dashboard(self):
        """Render analytics dashboard."""
        st.header("Analytics Dashboard")
        st.markdown("View workflow execution metrics and performance data.")

        if not self.workflow_manager.analytics:
            st.warning("⚠️ Analytics is not enabled. Initialize the workflow manager with analytics_db_path to enable analytics.")
            return

        analytics = self.workflow_manager.analytics

        # Get analytics summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Workflows", "0")  # Would query actual count

        with col2:
            st.metric("Total Tokens", "0")

        with col3:
            st.metric("Success Rate", "100%")

        st.divider()

        # Analytics options
        option = st.selectbox(
            "Analytics View",
            ["Workflow Summary", "Node Execution", "Provider Metrics", "Cost Analysis"]
        )

        if option == "Workflow Summary":
            self._render_workflow_summary(analytics)

        elif option == "Node Execution":
            self._render_node_execution(analytics)

        elif option == "Provider Metrics":
            self._render_provider_metrics(analytics)

        elif option == "Cost Analysis":
            self._render_cost_analysis(analytics)

    def _render_workflow_summary(self, analytics):
        """Render workflow summary analytics."""
        st.subheader("Workflow Summary")

        # Would query actual analytics data
        st.info("📊 Workflow execution summary will be displayed here")

    def _render_node_execution(self, analytics):
        """Render node execution analytics."""
        st.subheader("Node Execution Metrics")

        # Would query actual node execution data
        st.info("📊 Node-level execution metrics will be displayed here")

    def _render_provider_metrics(self, analytics):
        """Render provider metrics."""
        st.subheader("Provider Performance")

        # Would query actual provider metrics
        st.info("📊 Provider performance metrics will be displayed here")

    def _render_cost_analysis(self, analytics):
        """Render cost analysis."""
        st.subheader("Cost Analysis")

        # Would query actual cost data
        st.info("📊 Cost breakdown will be displayed here")


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_openevolve_bubblelabs_ui():
    """
    Render the complete OpenEvolve + BubbleLabs integrated UI.

    This is the main entry point for adding OpenEvolve workflow management
    to the BubbleLabs interface.
    """
    ui = OpenEvolveBubbleLabsUI()
    ui.render()


# =============================================================================
# SIDEBAR INTEGRATION
# =============================================================================

def add_openevolve_to_sidebar():
    """
    Add OpenEvolve workflow manager to the sidebar navigation.

    Call this function from main.py to add OpenEvolve to your app's sidebar.
    """
    with st.sidebar:
        st.divider()
        st.header("🔬 OpenEvolve")

        if st.page_link("app/OpenEvolve_Workflows", label="📋 Workflow Manager", use_container_width=True):
            st.page_link("app/OpenEvolve_Workflows", label="Manage OpenEvolve workflows")

        if st.page_link("app/Create_Workflow", label="➕ Create Workflow", use_container_width=True):
            st.page_link("app/Create_Workflow", label="Create a new workflow")

        if st.page_link("app/Execute_Workflow", label="▶️ Execute Workflow", use_container_width=True):
            st.page_link("app/Execute_Workflow", label="Execute a workflow")

        st.divider()


# =============================================================================
# STANDALONE APP
# =============================================================================

def main():
    """Run as standalone Streamlit app."""
    st.set_page_config(
        page_title="OpenEvolve Workflow Manager",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Render the main UI
    render_openevolve_bubblelabs_ui()

    # Add to sidebar if integrated
    add_openevolve_to_sidebar()


if __name__ == "__main__":
    main()
