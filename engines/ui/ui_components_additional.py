"""
Additional UI Components for Sovereign-Grade Decomposition Workflow

This file contains additional UI components that need to be added to ui_components.py
"""
from __future__ import annotations


from typing import Dict, Any, Optional, List
import json
from ui_shim import ui as st
import time


def render_workflow_orchestrator(
    teams: List['Team'],
    gauntlets: List['GauntletDefinition'],
    current_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Render the workflow orchestrator UI component for configuring workflow execution.

    This component allows users to:
    - Select teams for each workflow stage
    - Select gauntlets for critique and verification
    - Configure advanced workflow options
    - Set resource limits and optimization parameters
    - Enable/disable learning features
    - Configure auto-approval settings

    Args:
        teams: List of available teams
        gauntlets: List of available gauntlets
        current_config: Current configuration (for editing existing workflows)

    Returns:
        Dictionary containing the workflow configuration
    """
    st.subheader("🎛️ Workflow Orchestrator Configuration")

    if current_config is None:
        current_config = {}

    # Initialize config
    config = {
        "content_analyzer_team": current_config.get("content_analyzer_team", ""),
        "planner_team": current_config.get("planner_team", ""),
        "solver_team": current_config.get("solver_team", ""),
        "patcher_team": current_config.get("patcher_team", ""),
        "assembler_team": current_config.get("assembler_team", ""),
        "sub_problem_red_gauntlet": current_config.get("sub_problem_red_gauntlet", ""),
        "sub_problem_gold_gauntlet": current_config.get("sub_problem_gold_gauntlet", ""),
        "final_red_gauntlet": current_config.get("final_red_gauntlet", ""),
        "final_gold_gauntlet": current_config.get("final_gold_gauntlet", ""),
        "max_refinement_loops": current_config.get("max_refinement_loops", 3),
        "auto_approval_enabled": current_config.get("auto_approval_enabled", False),
        "parallel_processing_enabled": current_config.get("parallel_processing_enabled", False),
        "learning_enabled": current_config.get("learning_enabled", False),
        "resource_limits": current_config.get("resource_limits", {}),
        "mdap_enabled": current_config.get("mdap_enabled", False),
        "mdap_config": current_config.get("mdap_config", {}),
        "maker_enabled": current_config.get("maker_enabled", False),
        "maker_config": current_config.get("maker_config", {})
    }

    # Team selection section
    with st.expander("👥 Team Assignment", expanded=True):
        st.write("Select teams for each stage of the workflow:")

        # Group teams by role
        blue_teams = [t for t in teams if t.role == "Blue"]
        red_teams = [t for t in teams if t.role == "Red"]
        gold_teams = [t for t in teams if t.role == "Gold"]

        # Content Analyzer Team
        if blue_teams:
            config["content_analyzer_team"] = st.selectbox(
                "Content Analyzer Team (Stage 0)",
                options=[""] + [t.name for t in blue_teams],
                index=0 if not config.get("content_analyzer_team") else [t.name for t in blue_teams].index(config.get("content_analyzer_team")) + 1 if config.get("content_analyzer_team") in [t.name for t in blue_teams] else 0,
                help="Team responsible for analyzing the problem content"
            )

        # Planner Team
        if blue_teams:
            config["planner_team"] = st.selectbox(
                "Planner Team (Stage 1)",
                options=[""] + [t.name for t in blue_teams],
                index=0 if not config.get("planner_team") else [t.name for t in blue_teams].index(config.get("planner_team")) + 1 if config.get("planner_team") in [t.name for t in blue_teams] else 0,
                help="Team responsible for decomposing the problem"
            )

        # Solver Team
        if blue_teams:
            config["solver_team"] = st.selectbox(
                "Solver Team (Stage 3)",
                options=[""] + [t.name for t in blue_teams],
                index=0 if not config.get("solver_team") else [t.name for t in blue_teams].index(config.get("solver_team")) + 1 if config.get("solver_team") in [t.name for t in blue_teams] else 0,
                help="Team responsible for generating solutions"
            )

        # Patcher Team
        if blue_teams:
            config["patcher_team"] = st.selectbox(
                "Patcher Team (Stage 3)",
                options=[""] + [t.name for t in blue_teams],
                index=0 if not config.get("patcher_team") else [t.name for t in blue_teams].index(config.get("patcher_team")) + 1 if config.get("patcher_team") in [t.name for t in blue_teams] else 0,
                help="Team responsible for fixing rejected solutions"
            )

        # Assembler Team
        if blue_teams:
            config["assembler_team"] = st.selectbox(
                "Assembler Team (Stage 4)",
                options=[""] + [t.name for t in blue_teams],
                index=0 if not config.get("assembler_team") else [t.name for t in blue_teams].index(config.get("assembler_team")) + 1 if config.get("assembler_team") in [t.name for t in blue_teams] else 0,
                help="Team responsible for assembling the final solution"
            )

    # Gauntlet selection section
    with st.expander("⚔️ Gauntlet Assignment", expanded=True):
        st.write("Select gauntlets for critique and verification:")

        # Sub-problem Red Team Gauntlet
        if red_teams:
            config["sub_problem_red_gauntlet"] = st.selectbox(
                "Sub-Problem Red Team Gauntlet",
                options=[""] + [g.name for g in gauntlets if any(m.role == "Red" for m in red_teams if m.name == g.team_name)],
                index=0,
                help="Gauntlet for critiquing sub-problem solutions"
            )

        # Sub-problem Gold Team Gauntlet
        if gold_teams:
            config["sub_problem_gold_gauntlet"] = st.selectbox(
                "Sub-Problem Gold Team Gauntlet",
                options=[""] + [g.name for g in gauntlets if any(m.role == "Gold" for m in gold_teams if m.name == g.team_name)],
                index=0,
                help="Gauntlet for verifying sub-problem solutions"
            )

        # Final Red Team Gauntlet
        if red_teams:
            config["final_red_gauntlet"] = st.selectbox(
                "Final Red Team Gauntlet",
                options=[""] + ["Final_Red_Gauntlet"],
                index=0,
                help="Gauntlet for final adversarial testing"
            )

        # Final Gold Team Gauntlet
        if gold_teams:
            config["final_gold_gauntlet"] = st.selectbox(
                "Final Gold Team Gauntlet",
                options=[""] + ["Final_Gold_Gauntlet"],
                index=0,
                help="Gauntlet for final verification"
            )

    # Advanced configuration section
    with st.expander("🔧 Advanced Configuration"):
        col1, col2 = st.columns(2)

        with col1:
            config["max_refinement_loops"] = st.number_input(
                "Max Refinement Loops",
                min_value=1,
                max_value=10,
                value=config["max_refinement_loops"],
                help="Maximum number of self-healing iterations"
            )

            config["auto_approval_enabled"] = st.checkbox(
                "Enable Auto-Approval",
                value=config["auto_approval_enabled"],
                help="Automatically approve plans that meet criteria"
            )

        with col2:
            config["parallel_processing_enabled"] = st.checkbox(
                "Enable Parallel Processing",
                value=config["parallel_processing_enabled"],
                help="Solve independent sub-problems in parallel"
            )

            config["learning_enabled"] = st.checkbox(
                "Enable Learning",
                value=config["learning_enabled"],
                help="Extract and learn from workflow execution"
            )

        st.markdown("---")
        st.write("MDAP / MAKER Execution")
        config["mdap_enabled"] = st.checkbox(
            "Enable MDAP for solution generation",
            value=config["mdap_enabled"],
            help="Use MDAP voting and red-flagging during solution generation"
        )
        config["maker_enabled"] = st.checkbox(
            "Enable MAKER for solution generation",
            value=config["maker_enabled"],
            help="Use MAKER stepwise voting and checkpointing during solution generation"
        )
        mdap_config_text = st.text_area(
            "MDAP Config (JSON)",
            value=json.dumps(config["mdap_config"], indent=2) if config["mdap_config"] else "{}",
            height=140
        )
        maker_config_text = st.text_area(
            "MAKER Config (JSON)",
            value=json.dumps(config["maker_config"], indent=2) if config["maker_config"] else "{}",
            height=140
        )
        try:
            config["mdap_config"] = json.loads(mdap_config_text) if mdap_config_text.strip() else {}
        except json.JSONDecodeError:
            st.warning("Invalid MDAP config JSON. Using defaults.")
            config["mdap_config"] = {}
        try:
            config["maker_config"] = json.loads(maker_config_text) if maker_config_text.strip() else {}
        except json.JSONDecodeError:
            st.warning("Invalid MAKER config JSON. Using defaults.")
            config["maker_config"] = {}

    # Resource limits section
    with st.expander("💾 Resource Limits"):
        resource_limits = config["resource_limits"]

        col1, col2, col3 = st.columns(3)

        with col1:
            resource_limits["max_api_calls"] = st.number_input(
                "Max API Calls",
                min_value=0,
                value=resource_limits.get("max_api_calls", 1000),
                help="Maximum number of API calls"
            )

        with col2:
            resource_limits["max_tokens"] = st.number_input(
                "Max Tokens",
                min_value=0,
                value=resource_limits.get("max_tokens", 1000000),
                help="Maximum number of tokens"
            )

        with col3:
            resource_limits["max_cost"] = st.number_input(
                "Max Cost ($)",
                min_value=0.0,
                value=float(resource_limits.get("max_cost", 10.0)),
                step=0.1,
                help="Maximum cost in USD"
            )

        config["resource_limits"] = resource_limits

    # Configuration summary
    st.subheader("📋 Configuration Summary")

    summary_cols = st.columns(6)
    with summary_cols[0]:
        st.metric("Teams Selected", sum(1 for v in config.values() if isinstance(v, str) and v and "team" in v))
    with summary_cols[1]:
        st.metric("Gauntlets Selected", sum(1 for v in config.values() if isinstance(v, str) and v and "gauntlet" in v))
    with summary_cols[2]:
        st.metric("Max Refinements", config["max_refinement_loops"])
    with summary_cols[3]:
        st.metric("Parallel Enabled", "Yes" if config["parallel_processing_enabled"] else "No")
    with summary_cols[4]:
        st.metric("MDAP Enabled", "Yes" if config.get("mdap_enabled") else "No")
    with summary_cols[5]:
        st.metric("MAKER Enabled", "Yes" if config.get("maker_enabled") else "No")

    return config


def render_realtime_monitoring(
    workflow_state: Optional['WorkflowState'] = None
) -> None:
    """
    Render the real-time monitoring dashboard for active workflow executions.

    This component displays:
    - Progress tracking for current workflow
    - Resource usage metrics
    - Performance metrics
    - Interactive controls for workflow management
    - Alert system for issues
    - Log viewer

    Args:
        workflow_state: Current workflow state (if active)
    """
    st.subheader("📊 Real-Time Monitoring")

    if not workflow_state:
        st.info("No active workflow execution. Start a workflow to see real-time monitoring.")
        return

    # Progress section
    with st.container():
        st.write("**Progress Overview**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Current Stage",
                workflow_state.current_stage,
                help="The current stage of the workflow"
            )

        with col2:
            st.metric(
                "Progress",
                f"{workflow_state.progress * 100:.1f}%",
                delta=None,
                help="Overall workflow progress"
            )

        with col3:
            elapsed_time = time.time() - workflow_state.start_time
            st.metric(
                "Elapsed Time",
                f"{elapsed_time:.0f}s",
                help="Time since workflow started"
            )

        # Progress bar
        st.progress(workflow_state.progress)

    # Resource usage section
    with st.expander("💾 Resource Usage", expanded=True):
        resource_usage = workflow_state.resource_usage if hasattr(workflow_state, 'resource_usage') else {}

        if resource_usage:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "API Calls",
                    resource_usage.get("api_calls", 0),
                    help="Total API calls made"
                )

            with col2:
                st.metric(
                    "Tokens Used",
                    f"{resource_usage.get('tokens_used', 0):,}",
                    help="Total tokens consumed"
                )

            with col3:
                st.metric(
                    "Cost",
                    f"${resource_usage.get('estimated_cost', 0.0):.4f}",
                    help="Estimated cost in USD"
                )

            with col4:
                st.metric(
                    "Memory",
                    f"{resource_usage.get('memory_usage_mb', 0):.1f} MB",
                    help="Memory usage"
                )
        else:
            st.info("No resource usage data available yet.")

    # Performance metrics section
    with st.expander("⚡ Performance Metrics"):
        performance_metrics = workflow_state.performance_metrics if hasattr(workflow_state, 'performance_metrics') else {}

        if performance_metrics:
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Sub-Problems Solved",
                    performance_metrics.get("sub_problems_solved", 0),
                    delta=None,
                    help="Number of sub-problems successfully solved"
                )

            with col2:
                st.metric(
                    "Solution Quality",
                    f"{performance_metrics.get('avg_solution_quality', 0.0) * 100:.1f}%",
                    delta=None,
                    help="Average quality of solutions"
                )
        else:
            st.info("No performance metrics available yet.")

    # Interactive controls section
    with st.expander("🎮 Workflow Controls"):
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("⏸️ Pause Workflow", key="pause_workflow"):
                st.warning("Workflow paused (not yet implemented)")

        with col2:
            if st.button("⏹️ Stop Workflow", key="stop_workflow"):
                st.error("Workflow stopped (not yet implemented)")

        with col3:
            if st.button("🔄 Refresh", key="refresh_monitoring"):
                st.rerun()

    # Alert system section
    with st.expander("🚨 Alerts"):
        alerts = []

        # Check for issues
        if workflow_state.refinement_loop_count > 3:
            alerts.append({
                "level": "warning",
                "message": f"High refinement loop count: {workflow_state.refinement_loop_count}"
            })

        if hasattr(workflow_state, 'rejected_sub_problems') and workflow_state.rejected_sub_problems:
            alerts.append({
                "level": "error",
                "message": f"{len(workflow_state.rejected_sub_problems)} sub-problems rejected"
            })

        if alerts:
            for alert in alerts:
                if alert["level"] == "error":
                    st.error(f"[FAIL] {alert['message']}")
                elif alert["level"] == "warning":
                    st.warning(f"[WARN] {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        else:
            st.success("[OK] No alerts - Workflow running smoothly")

    # Log viewer section
    with st.expander("📜 Log Viewer"):
        st.text_area(
            "Workflow Logs",
            value="Log viewer placeholder - logs would appear here",
            height=200,
            disabled=True,
            help="Real-time workflow logs"
        )

    # Sub-problem status section
    if workflow_state.decomposition_plan and workflow_state.decomposition_plan.sub_problems:
        with st.expander("📋 Sub-Problem Status"):
            st.write("**Sub-Problem Progress:**")

            for sp in workflow_state.decomposition_plan.sub_problems:
                status_emoji = {
                    "pending": "⏳",
                    "in_progress": "🔄",
                    "solved": "[OK]",
                    "failed": "[FAIL]",
                    "requires_rework": "🔧"
                }.get(sp.status, "❓")

                st.write(f"{status_emoji} **{sp.id}**: {sp.status}")
                if sp.status != "pending":
                    st.caption(f"Complexity: {sp.ai_suggested_complexity_score}/10")


# Export list of functions to add to ui_components.py
UI_FUNCTIONS_TO_ADD = [
    render_workflow_orchestrator,
    render_realtime_monitoring,
]
