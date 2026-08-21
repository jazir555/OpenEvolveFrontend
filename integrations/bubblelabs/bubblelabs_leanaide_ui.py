"""
BubbleLabs UI Component with LeanAide Integration

This module extends BubbleLabs UI with comprehensive LeanAide functionality,
including MCTS visualization, Lean4 proof tracking, and formal verification controls.

Key Features:
    - LeanAide control panel in BubbleLabs UI
    - MCTS tree visualization nodes
    - Lean4 proof step tracking
    - MDAP voting display
    - Mathematical query interface
    - Formal verification status display

Usage:
    Add to BubbleLabs workflow UI to enable LeanAide features.

Author: OpenEvolve
Created: 2025-01-03
"""

from .ui_shim import ui as st
import json
import time
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from .bubblelabs_leanaide_integration import (
        get_leanaide_bridge,
        LeanAideTaskType,
        LeanAideIntegrationBridge,
        initialize_leanaide_integration,
        LEANAIDE_AVAILABLE,
        MCTS_AVAILABLE,
        MDAP_AVAILABLE,
        LEAN4_AVAILABLE
    )
    LEANAIDE_INTEGRATION_AVAILABLE = True
except ImportError:
    LEANAIDE_INTEGRATION_AVAILABLE = False
    print("Warning: LeanAide integration not available")


class LeanAideUIComponent:
    """
    UI UI component for LeanAide integration in BubbleLabs.

    Provides panels and controls for:
    - LeanAide task execution
    - MCTS tree visualization
    - Lean4 proof verification
    - Mathematical queries
    """

    def __init__(self):
        """Initialize LeanAide UI component."""
        if not LEANAIDE_INTEGRATION_AVAILABLE:
            st.error("LeanAide integration not available")
            return

        self.bridge = get_leanaide_bridge()
        self.initialized = False

    def render_leanaide_control_panel(self):
        """
        Render main LeanAide control panel in BubbleLabs.

        This adds a new tab to the BubbleLabs interface with LeanAide controls.
        """
        if not LEANAIDE_INTEGRATION_AVAILABLE:
            st.warning("LeanAide integration not available. Please install required dependencies.")
            return

        st.header("🧮 LeanAide Formal Verification")

        # Initialize if needed
        if not self.initialized:
            with st.spinner("Initializing LeanAide..."):
                status = initialize_leanaide_integration()
                self.initialized = True

                if not status.get("bridge_available"):
                    st.error("Failed to initialize LeanAide bridge")
                    return

        # Show status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MCTS Available", "[OK]" if MCTS_AVAILABLE else "[FAIL]")
        with col2:
            st.metric("MDAP Available", "[OK]" if MDAP_AVAILABLE else "[FAIL]")
        with col3:
            st.metric("Lean4 Available", "[OK]" if LEAN4_AVAILABLE else "[FAIL]")

        # Create tabs for different LeanAide functions
        tabs = st.tabs([
            "🔬 Theorem Proving",
            "🌳 MCTS Visualization",
            "[OK] Lean4 Verification",
            "🧮 Math Queries",
            "⚙️ Settings"
        ])

        with tabs[0]:
            self._render_theorem_proving_panel()

        with tabs[1]:
            self._render_mcts_visualization()

        with tabs[2]:
            self._render_lean4_verification()

        with tabs[3]:
            self._render_math_queries()

        with tabs[4]:
            self._render_settings()

    def _render_theorem_proving_panel(self):
        """Render theorem proving panel."""
        st.subheader("🔬 Formal Theorem Proving")

        # Theorem input
        theorem = st.text_area(
            "Theorem Statement",
            placeholder="Enter natural language theorem, e.g., 'There are infinitely many prime numbers'",
            height=100,
            key="leanaide_theorem_input"
        )

        # Theorem name (optional)
        theorem_name = st.text_input(
            "Theorem Name (Optional)",
            placeholder="e.g., infinitely_many_primes",
            key="leanaide_theorem_name"
        )

        # Action buttons
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            translate_btn = st.button("🔄 Translate to Lean", key="translate_theorem")

        with col2:
            prove_btn = st.button("📐 Generate Proof", key="generate_proof")

        with col3:
            verify_btn = st.button("[OK] Verify Code", key="verify_lean_code")

        with col4:
            mcts_btn = st.button("🌳 MCTS Search", key="mcts_search")

        # Handle actions
        if translate_btn and theorem:
            self._handle_translate_theorem(theorem, theorem_name)

        if prove_btn and theorem:
            self._handle_generate_proof(theorem, theorem_name)

        if mcts_btn and theorem:
            self._handle_mcts_search(theorem, theorem_name)

        # Display results
        if "leanaide_last_result" in st.session_state:
            self._display_leanaide_result(st.session_state.leanaide_last_result)

    def _render_mcts_visualization(self):
        """Render MCTS tree visualization."""
        st.subheader("🌳 MCTS Search Visualization")

        # Get active trees
        trees = self.bridge.get_all_trees()

        if not trees:
            st.info("No MCTS trees available. Run a proof search first.")
            return

        # Tree selector
        selected_tree_id = st.selectbox(
            "Select MCTS Tree",
            options=trees,
            format_func=lambda tid: f"Tree {tid[:8]}... ({self.bridge.get_tree(tid).theorem[:50]}...)"
        )

        if selected_tree_id:
            tree = self.bridge.get_tree(selected_tree_id)

            if tree:
                # Display tree statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Iterations", tree.iterations)
                with col2:
                    st.metric("Total Nodes", len(tree.nodes))
                with col3:
                    st.metric("Max Depth", tree.statistics.get("max_depth", 0))
                with col4:
                    st.metric("Win Rate", f"{tree.statistics.get('win_rate', 0):.3f}")

                # Agent statistics
                if tree.statistics.get("agent_statistics"):
                    st.subheader("Agent Performance")
                    agent_stats = tree.statistics["agent_statistics"]

                    # Create DataFrame for display
                    import pandas as pd
                    agent_data = []
                    for agent_id, stats in agent_stats.items():
                        agent_data.append({
                            "Agent": agent_id,
                            "Votes Cast": stats.get("votes_cast", 0),
                            "Votes Accepted": stats.get("votes_accepted", 0),
                            "Success Rate": f"{stats.get('success_rate', 0):.2%}"
                        })

                    if agent_data:
                        df = pd.DataFrame(agent_data)
                        st.dataframe(df, use_container_width=True)

                # Voting statistics
                if tree.statistics.get("voting_statistics"):
                    st.subheader("Voting Statistics")
                    voting_stats = tree.statistics["voting_statistics"]
                    cols = st.columns(len(voting_stats))
                    for i, (key, value) in enumerate(voting_stats.items()):
                        with cols[i]:
                            st.metric(key.replace("_", " ").title(), value)

                # Red flag analysis
                if tree.statistics.get("red_flag_analysis"):
                    st.subheader("🚩 Red Flag Analysis")
                    red_flags = tree.statistics["red_flag_analysis"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Red Flagged Nodes", red_flags.get("red_flagged_nodes", 0))
                    with col2:
                        st.metric("Red Flag Rate", f"{red_flags.get('red_flag_rate', 0):.2%}")

                # Best path display
                if tree.best_path:
                    st.subheader("Best Proof Path")
                    path_steps = []
                    for node_id in tree.best_path:
                        node = tree.nodes.get(node_id)
                        if node and node.action:
                            path_steps.append(f"Step {node.depth}: {node.action}")

                    for step in path_steps:
                        st.text(step)

                # JSON export
                    if st.button("📥 Export Tree JSON", key=f"export_{selected_tree_id}"):
                        st.json(tree.to_dict())

    def _render_lean4_verification(self):
        """Render Lean4 verification panel."""
        st.subheader("[OK] Lean4 Code Verification")

        # Code input
        lean_code = st.text_area(
            "Lean4 Code",
            placeholder="Enter Lean code to verify...",
            height=200,
            key="leanaide_code_input"
        )

        col1, col2 = st.columns(2)
        with col1:
            verify_btn = st.button("[OK] Verify Code", key="verify_code_btn")
        with col2:
            elaborate_btn = st.button("🔍 Elaborate", key="elaborate_code_btn")

        if verify_btn and lean_code:
            self._handle_verify_code(lean_code)

        if elaborate_btn and lean_code:
            self._handle_elaborate_code(lean_code)

        # Active proofs
        proofs = self.bridge.get_all_proofs()

        if proofs:
            st.subheader("Active Proofs")

            selected_proof_id = st.selectbox(
                "Select Proof",
                options=proofs,
                format_func=lambda pid: f"Proof {pid[:8]}..."
            )

            if selected_proof_id:
                proof = self.bridge.get_proof(selected_proof_id)

                if proof:
                    # Proof metadata
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Complete", "[OK]" if proof.is_complete else "[FAIL]")
                    with col2:
                        st.metric("Verified", "[OK]" if proof.is_verified else "[FAIL]")
                    with col3:
                        st.metric("Steps", len(proof.steps))
                    with col4:
                        st.metric("Errors", len(proof.errors))

                    # Proof steps
                    if proof.steps:
                        st.subheader("Proof Steps")
                        for step in proof.steps:
                            with st.expander(f"Step {step.step_number}: {step.tactic}"):
                                st.text(f"Goals Before: {', '.join(step.goals_before)}")
                                st.text(f"Goals After: {', '.join(step.goals_after)}")
                                if step.error_message:
                                    st.error(f"Error: {step.error_message}")
                                else:
                                    st.success("[OK] Valid")

                    # Lean code
                    if proof.lean_code:
                        st.subheader("Generated Lean Code")
                        st.code(proof.lean_code, language="lean")

                    # Errors
                    if proof.errors:
                        st.subheader("Errors")
                        for error in proof.errors:
                            st.error(error)

    def _render_math_queries(self):
        """Render math query panel."""
        st.subheader("🧮 Mathematical Queries")

        # Query input
        query = st.text_area(
            "Math Question",
            placeholder="Ask a math question, e.g., 'What is the fundamental theorem of calculus?'",
            height=80,
            key="leanaide_math_query"
        )

        # Number of answers
        n_answers = st.slider(
            "Number of Answers",
            min_value=1,
            max_value=10,
            value=3,
            key="leanaide_n_answers"
        )

        if st.button("🔍 Ask", key="math_query_btn") and query:
            self._handle_math_query(query, n_answers)

        # Display history
        history = self.bridge.get_execution_history(limit=10)
        math_queries = [r for r in history if r.task_type == LeanAideTaskType.MATH_QUERY]

        if math_queries:
            st.subheader("Recent Queries")
            for result in reversed(math_queries):
                with st.expander(f"{result.timestamp[:19]} - {result.data.get('query', '')[:50]}..."):
                    if result.success and result.data:
                        answers = result.data.get('answers', [])
                        for i, answer in enumerate(answers, 1):
                            st.text(f"Answer {i}: {answer}")
                    elif result.error:
                        st.error(result.error)

    def _render_settings(self):
        """Render LeanAide settings panel."""
        st.subheader("⚙️ LeanAide Configuration")

        # Server settings
        st.markdown("### Server Configuration")
        col1, col2 = st.columns(2)

        with col1:
            host = st.text_input(
                "LeanAide Host",
                value=self.bridge.leanaide_host,
                key="leanaide_host"
            )

        with col2:
            port = st.number_input(
                "LeanAide Port",
                value=self.bridge.leanaide_port,
                min_value=1,
                max_value=65535,
                key="leanaide_port"
            )

        # Feature toggles
        st.markdown("### Feature Toggles")
        col1, col2, col3 = st.columns(3)

        with col1:
            enable_mcts = st.checkbox(
                "Enable MCTS",
                value=self.bridge.enable_mcts,
                disabled=not MCTS_AVAILABLE
            )

        with col2:
            enable_mdap = st.checkbox(
                "Enable MDAP",
                value=self.bridge.enable_mdap,
                disabled=not MDAP_AVAILABLE
            )

        with col3:
            enable_lean4 = st.checkbox(
                "Enable Lean4",
                value=self.bridge.enable_lean4,
                disabled=not LEAN4_AVAILABLE
            )

        # MCTS parameters
        if MCTS_AVAILABLE:
            st.markdown("### MCTS Parameters")
            col1, col2, col3 = st.columns(3)

            with col1:
                max_iterations = st.number_input(
                    "Max Iterations",
                    value=1000,
                    min_value=10,
                    max_value=10000,
                    key="mcts_max_iterations"
                )

            with col2:
                time_budget = st.number_input(
                    "Time Budget (s)",
                    value=300,
                    min_value=10,
                    max_value=3600,
                    key="mcts_time_budget"
                )

            with col3:
                c_param = st.number_input(
                    "UCB C Parameter",
                    value=1.414,
                    min_value=0.1,
                    max_value=5.0,
                    step=0.01,
                    key="mcts_c_param"
                )

        # MDAP parameters
        if MDAP_AVAILABLE:
            st.markdown("### MDAP Parameters")
            col1, col2 = st.columns(2)

            with col1:
                expansion_agents = st.number_input(
                    "Expansion Agents",
                    value=3,
                    min_value=1,
                    max_value=10,
                    key="mdap_expansion_agents"
                )

            with col2:
                simulation_voters = st.number_input(
                    "Simulation Voters",
                    value=5,
                    min_value=1,
                    max_value=20,
                    key="mdap_simulation_voters"
                )

        # Status
        st.markdown("### System Status")
        status = self.bridge.get_status()
        st.json(status)

        # Apply settings
        if st.button("Apply Settings", key="apply_leanaide_settings"):
            st.success("Settings applied (requires restart)")

    # =========================================================================
    # Action Handlers
    # =========================================================================

    def _handle_translate_theorem(self, theorem: str, theorem_name: Optional[str]):
        """Handle theorem translation."""
        with st.spinner("Translating theorem..."):
            result = self.bridge.execute_task(
                LeanAideTaskType.TRANSLATE_THEOREM,
                theorem_text=theorem,
                theorem_name=theorem_name
            )
            st.session_state.leanaide_last_result = result

    def _handle_generate_proof(self, theorem: str, theorem_name: Optional[str]):
        """Handle proof generation."""
        with st.spinner("Generating proof..."):
            result = self.bridge.execute_task(
                LeanAideTaskType.GENERATE_PROOF,
                theorem_text=theorem,
                theorem_code=theorem_name
            )
            st.session_state.leanaide_last_result = result

    def _handle_mcts_search(self, theorem: str, theorem_name: Optional[str]):
        """Handle MCTS search."""
        with st.spinner("Running MCTS search (this may take a while)..."):
            result = self.bridge.execute_task(
                LeanAideTaskType.MCTS_SEARCH,
                theorem=theorem,
                theorem_name=theorem_name
            )
            st.session_state.leanaide_last_result = result

    def _handle_verify_code(self, code: str):
        """Handle code verification."""
        with st.spinner("Verifying code..."):
            result = self.bridge.execute_task(
                LeanAideTaskType.VERIFY_SOLUTION,
                code=code
            )
            st.session_state.leanaide_last_result = result

    def _handle_elaborate_code(self, code: str):
        """Handle code elaboration."""
        with st.spinner("Elaborating code..."):
            result = self.bridge.execute_task(
                LeanAideTaskType.ELABORATE_CODE,
                code=code
            )
            st.session_state.leanaide_last_result = result

    def _handle_math_query(self, query: str, n: int):
        """Handle math query."""
        with st.spinner("Processing query..."):
            result = self.bridge.execute_task(
                LeanAideTaskType.MATH_QUERY,
                query=query,
                n=n
            )
            st.session_state.leanaide_last_result = result

    def _display_leanaide_result(self, result):
        """Display LeanAide execution result."""
        st.markdown("---")
        st.markdown("### Result")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Success", "[OK]" if result.success else "[FAIL]")
        with col2:
            st.metric("Execution Time", f"{result.execution_time:.2f}s")

        if result.error:
            st.error(f"Error: {result.error}")
            return

        if result.data:
            # Display based on task type
            if result.task_type == LeanAideTaskType.TRANSLATE_THEOREM:
                if result.data.get("lean_code"):
                    st.subheader("Generated Lean Code")
                    st.code(result.data["lean_code"], language="lean")

                if result.data.get("theorem_name"):
                    st.text(f"Theorem Name: {result.data['theorem_name']}")

            elif result.task_type == LeanAideTaskType.GENERATE_PROOF:
                if result.data.get("proof_document"):
                    st.subheader("Proof Document")
                    st.markdown(result.data["proof_document"])

                if result.data.get("lean_proof"):
                    st.subheader("Lean Proof Code")
                    st.code(result.data["lean_proof"], language="lean")

            elif result.task_type == LeanAideTaskType.VERIFY_SOLUTION:
                if result.data.get("is_valid") is not None:
                    is_valid = result.data["is_valid"]
                    st.metric("Valid", "[OK] Yes" if is_valid else "[FAIL] No")

                if result.data.get("unproven_count") is not None:
                    st.metric("Unproven Obligations", result.data["unproven_count"])

                if result.data.get("sorries_after_purge"):
                    st.subheader("Unsolved Goals")
                    for sorry in result.data["sorries_after_purge"]:
                        st.text(str(sorry))

            elif result.task_type == LeanAideTaskType.MATH_QUERY:
                answers = result.data.get("answers", [])
                if answers:
                    st.subheader("Answers")
                    for i, answer in enumerate(answers, 1):
                        st.markdown(f"**Answer {i}:** {answer}")

            elif result.task_type == LeanAideTaskType.MCTS_SEARCH:
                if result.data.get("visualization_data"):
                    st.success("MCTS search complete! Check the 'MCTS Visualization' tab.")

                if result.data.get("result"):
                    mcts_result = result.data["result"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Iterations", mcts_result.get("search_iterations", 0))
                    with col2:
                        st.metric("Win Rate", f"{mcts_result.get('win_rate', 0):.3f}")
                    with col3:
                        st.metric("Confidence", f"{mcts_result.get('confidence', 0):.3f}")

        # Visualization data
        if result.visualization_data:
            st.markdown("---")
            st.markdown("### Visualization Data")
            st.json(result.visualization_data)


# =============================================================================
# Integration Helper
# =============================================================================

def add_leanaide_to_bubblelabs():
    """
    Add LeanAide tab to BubbleLabs UI.

    Call this function when setting up BubbleLabs to add LeanAide functionality.
    """
    if not LEANAIDE_INTEGRATION_AVAILABLE:
        return None

    component = LeanAideUIComponent()
    return component


def render_leanaide_in_bubblelabs():
    """
    Render LeanAide panel within BubbleLabs.

    This is a convenience function that can be called directly
    from BubbleLabs workflow code.
    """
    if not LEANAIDE_INTEGRATION_AVAILABLE:
        st.warning("LeanAide integration not available")
        return

    component = LeanAideUIComponent()
    component.render_leanaide_control_panel()


if __name__ == "__main__":
    # Test UI component
    print("LeanAide UI Component for BubbleLabs")
    print(f"Integration Available: {LEANAIDE_INTEGRATION_AVAILABLE}")
    print(f"MCTS Available: {MCTS_AVAILABLE}")
    print(f"MDAP Available: {MDAP_AVAILABLE}")
    print(f"Lean4 Available: {LEAN4_AVAILABLE}")

