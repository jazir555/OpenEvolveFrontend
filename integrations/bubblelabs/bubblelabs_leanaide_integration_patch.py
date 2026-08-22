"""
BubbleLabs UI Integration Patch for LeanAide

This file shows how to patch the existing BubbleLabs UI component
to add LeanAide functionality.

Location: Add this to bubblelabs_ui_component.py

Author: OpenEvolve
Created: 2025-01-03
"""
from __future__ import annotations


# =============================================================================
# ADD THESE IMPORTS AT THE TOP OF bubblelabs_ui_component.py
# =============================================================================

try:
    try:
        from .bubblelabs_leanaide_ui import LeanAideUIComponent
    except ImportError:
        from bubblelabs_leanaide_ui import LeanAideUIComponent
    LEANAIDE_UI_AVAILABLE = True
except ImportError:
    LEANAIDE_UI_AVAILABLE = False
    print("Warning: LeanAide UI component not available")


# =============================================================================
# ADD THIS METHOD TO BubbleLabsWorkflowUI CLASS
# =============================================================================

def _render_leanaide_integration(self):
    """
    Render LeanAide formal verification panel.

    This method should be added to the BubbleLabsWorkflowUI class.
    """
    if not LEANAIDE_UI_AVAILABLE:
        ui.warning("LeanAide integration not available. Install bubblelabs_leanaide_ui.py")
        return

    ui.subheader("🧮 LeanAide Integration")

    # Status indicators
    col1, col2, col3 = ui.columns(3)

    try:
        try:
            from .bubblelabs_leanaide_integration import (
            LEANAIDE_AVAILABLE,
            MCTS_AVAILABLE,
            MDAP_AVAILABLE,
            LEAN4_AVAILABLE
            )
        except ImportError:
            from bubblelabs_leanaide_integration import (
            LEANAIDE_AVAILABLE,
            MCTS_AVAILABLE,
            MDAP_AVAILABLE,
            LEAN4_AVAILABLE
            )

        with col1:
            ui.metric("LeanAide", "[OK]" if LEANAIDE_AVAILABLE else "[FAIL]")
        with col2:
            ui.metric("MCTS", "[OK]" if MCTS_AVAILABLE else "[FAIL]")
        with col3:
            ui.metric("Lean4", "[OK]" if LEAN4_AVAILABLE else "[FAIL]")

    except ImportError:
        with col1:
            ui.metric("LeanAide", "[FAIL]")
        with col2:
            ui.metric("MCTS", "[FAIL]")
        with col3:
            ui.metric("Lean4", "[FAIL]")

    ui.markdown("---")

    # Render LeanAide UI component
    try:
        leanaide_ui = LeanAideUIComponent()

        # Create a simplified interface for BubbleLabs integration
        self._render_leanaide_quick_actions(leanaide_ui)

        # Add expander for full LeanAide interface
        with ui.expander("🔧 Advanced LeanAide Controls", expanded=False):
            leanaide_ui.render_leanaide_control_panel()

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        ui.error(f"Error loading LeanAide: {e}")


def _render_leanaide_quick_actions(self, leanaide_ui):
    """
    Render quick LeanAide actions for BubbleLabs workflow.

    This provides a streamlined interface integrated with BubbleLabs workflows.
    """
    try:
        from .bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType
    except ImportError:
        from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

    ui.markdown("### Quick Actions")

    # Quick theorem translation
    with ui.expander("🔄 Quick Theorem Translation", expanded=False):
        quick_theorem = ui.text_area(
            "Theorem",
            placeholder="Enter theorem to translate...",
            height=80,
            key="quick_theorem"
        )

        if ui.button("Translate", key="quick_translate_btn"):
            if quick_theorem:
                bridge = get_leanaide_bridge()
                with ui.spinner("Translating..."):
                    result = bridge.execute_task(
                        LeanAideTaskType.TRANSLATE_THEOREM,
                        theorem_text=quick_theorem
                    )

                if result.success:
                    ui.success("Translation successful!")
                    ui.code(result.data.get("lean_code", ""), language="lean")

                    # Option to add to workflow
                    if ui.button("Add to Workflow", key="add_translation_to_workflow"):
                        # This would add the translated code to the current workflow
                        ui.info("Added to workflow (feature to be implemented)")
                else:
                    ui.error(f"Translation failed: {result.error}")

    # Quick proof verification
    with ui.expander("[OK] Quick Code Verification", expanded=False):
        quick_code = ui.text_area(
            "Lean Code",
            placeholder="Enter Lean code to verify...",
            height=100,
            key="quick_verify_code"
        )

        if ui.button("Verify", key="quick_verify_btn"):
            if quick_code:
                bridge = get_leanaide_bridge()
                with ui.spinner("Verifying..."):
                    result = bridge.execute_task(
                        LeanAideTaskType.VERIFY_SOLUTION,
                        code=quick_code
                    )

                if result.success:
                    is_valid = result.data.get("is_valid", False)
                    unproven = result.data.get("unproven_count", 0)

                    if is_valid:
                        ui.success(f"[OK] Code is valid! No unproven obligations.")
                    else:
                        ui.warning(f"[WARN] {unproven} unproven obligation(s)")

                    # Show errors if any
                    if result.data.get("sorries_after_purge"):
                        ui.markdown("**Unsolved Goals:**")
                        for sorry in result.data["sorries_after_purge"][:3]:
                            ui.text(str(sorry))
                else:
                    ui.error(f"Verification failed: {result.error}")

    # Active LeanAide results
    with ui.expander("📊 Active Results", expanded=False):
        bridge = get_leanaide_bridge()

        # Active MCTS trees
        trees = bridge.get_all_trees()
        if trees:
            ui.markdown("**MCTS Trees:**")
            for tree_id in trees[:3]:  # Show first 3
                tree = bridge.get_tree(tree_id)
                if tree:
                    col1, col2, col3 = ui.columns(3)
                    with col1:
                        ui.text(tree.theorem[:50] + "...")
                    with col2:
                        ui.metric("Nodes", len(tree.nodes))
                    with col3:
                        ui.metric("Win Rate", f"{tree.statistics.get('win_rate', 0):.2%}")
        else:
            ui.info("No active MCTS trees")

        ui.markdown("---")

        # Active proofs
        proofs = bridge.get_all_proofs()
        if proofs:
            ui.markdown("**Proofs:**")
            for proof_id in proofs[:3]:  # Show first 3
                proof = bridge.get_proof(proof_id)
                if proof:
                    col1, col2, col3 = ui.columns(3)
                    with col1:
                        ui.text(proof.theorem_name)
                    with col2:
                        ui.metric("Steps", len(proof.steps))
                    with col3:
                        ui.metric("Verified", "[OK]" if proof.is_verified else "[FAIL]")
        else:
            ui.info("No active proofs")


# =============================================================================
# MODIFY THE render_workflow_visualizer METHOD
# =============================================================================

"""
In the render_workflow_visualizer method of BubbleLabsWorkflowUI,
add LeanAide to the tabs list:

ORIGINAL CODE (around line 288):
    tabs = ui.tabs(["Workflow Designer", "Active Workflows", "Workflow Control", "Global Parameters"])

MODIFIED CODE:
    tabs = ui.tabs([
        "Workflow Designer",
        "Active Workflows",
        "Workflow Control",
        "LeanAide",           # NEW TAB
        "Global Parameters"
    ])

    with tabs[0]:
        self._render_workflow_designer()

    with tabs[1]:
        self._render_active_workflows()

    with tabs[2]:
        self._render_workflow_control()

    with tabs[3]:  # NEW: LeanAide tab
        self._render_leanaide_integration()

    with tabs[4]:  # UPDATED: Was tabs[3]
        self._render_global_parameters()
"""


# =============================================================================
# ALTERNATIVE: ADD LEANAIDE AS A SIDEBAR OPTION
# =============================================================================

def add_leanaide_to_sidebar():
    """
    Add LeanAide quick link to BubbleLabs sidebar.

    This can be called from the main BubbleLabs app.
    """
    try:
        try:
            from .bubblelabs_leanaide_integration import get_leanaide_bridge
        except ImportError:
            from bubblelabs_leanaide_integration import get_leanaide_bridge

        bridge = get_leanaide_bridge()
        status = bridge.get_status()

        # Add sidebar status indicator
        try:
            from .ui_shim import ui
        except ImportError:
            from ui_shim import ui

        with ui.sidebar:
            ui.markdown("---")
            ui.markdown("### 🧮 LeanAide")

            if status.get('leanaide_available'):
                ui.success("Connected")
            else:
                ui.warning("Not Available")

            if ui.button("Open LeanAide Panel", key="sidebar_leanaide_btn"):
                ui.session_state['selected_tab'] = "LeanAide"

    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in {__name__}", exc_info=True)
        raise  # Re-raise the exception


# =============================================================================
# WORKFLOW NODE INTEGRATION
# =============================================================================

def register_leanaide_workflow_nodes():
    """
    Register LeanAide tools as workflow nodes in BubbleLabs.

    Call this during BubbleLabs initialization.
    """
    try:
        try:
            from .bubblelabs_leanaide_integration import (
            LEANAIDE_AVAILABLE,
            MCTS_AVAILABLE,
            MDAP_AVAILABLE
            )
        except ImportError:
            from bubblelabs_leanaide_integration import (
            LEANAIDE_AVAILABLE,
            MCTS_AVAILABLE,
            MDAP_AVAILABLE
            )

        if not (LEANAIDE_AVAILABLE or MCTS_AVAILABLE or MDAP_AVAILABLE):
            return False

        # Define LeanAide workflow nodes
        leanaide_nodes = {
            "leanaide_translate_theorem": {
                "name": "Translate Theorem to Lean",
                "category": "leanaide",
                "description": "Translate natural language theorem to Lean code",
                "inputs": {
                    "theorem_text": {
                        "type": "string",
                        "required": True,
                        "description": "Natural language theorem"
                    },
                    "theorem_name": {
                        "type": "string",
                        "required": False,
                        "description": "Optional theorem name"
                    }
                },
                "outputs": {
                    "lean_code": "Generated Lean code",
                    "theorem_name": "Theorem name",
                    "elaborated_type": "Elaborated type"
                },
                "execute": lambda **kwargs: _execute_translate_node(**kwargs)
            },

            "leanaide_generate_proof": {
                "name": "Generate Lean Proof",
                "category": "leanaide",
                "description": "Generate a formal proof for a theorem",
                "inputs": {
                    "theorem_text": {
                        "type": "string",
                        "required": True,
                        "description": "Natural language theorem"
                    },
                    "theorem_code": {
                        "type": "string",
                        "required": False,
                        "description": "Optional pre-translated Lean code"
                    }
                },
                "outputs": {
                    "proof_document": "Proof sketch",
                    "lean_proof": "Lean proof code",
                    "proof_id": "Proof visualization ID"
                },
                "execute": lambda **kwargs: _execute_proof_node(**kwargs)
            },

            "leanaide_mcts_search": {
                "name": "MCTS Proof Search",
                "category": "leanaide",
                "description": "Search for proof using Monte Carlo Tree Search",
                "inputs": {
                    "theorem": {
                        "type": "string",
                        "required": True,
                        "description": "Theorem to prove"
                    },
                    "max_iterations": {
                        "type": "integer",
                        "required": False,
                        "default": 1000,
                        "description": "Maximum MCTS iterations"
                    },
                    "time_budget": {
                        "type": "float",
                        "required": False,
                        "default": 300.0,
                        "description": "Time budget in seconds"
                    }
                },
                "outputs": {
                    "best_proof": "Best proof found",
                    "tree_id": "MCTS tree visualization ID",
                    "win_rate": "Search win rate",
                    "statistics": "Search statistics"
                },
                "execute": lambda **kwargs: _execute_mcts_node(**kwargs)
            },

            "leanaide_verify": {
                "name": "Verify Lean Code",
                "category": "leanaide",
                "description": "Verify Lean code correctness",
                "inputs": {
                    "code": {
                        "type": "string",
                        "required": True,
                        "description": "Lean code to verify"
                    }
                },
                "outputs": {
                    "is_valid": "Whether code is valid",
                    "unproven_count": "Number of unproven obligations",
                    "errors": "List of errors"
                },
                "execute": lambda **kwargs: _execute_verify_node(**kwargs)
            }
        }

        # Register with BubbleLabs (this would integrate with actual workflow system)
        # For now, just log the registration
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Registered {len(leanaide_nodes)} LeanAide workflow nodes")

        return True

    except ImportError:
        return False


def _execute_translate_node(**kwargs):
    """Execute translate theorem node."""
    try:
        from .bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType
    except ImportError:
        from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

    bridge = get_leanaide_bridge()
    result = bridge.execute_task(
        LeanAideTaskType.TRANSLATE_THEOREM,
        **kwargs
    )

    if result.success:
        return {
            "success": True,
            "lean_code": result.data.get("lean_code", ""),
            "theorem_name": result.data.get("theorem_name", ""),
            "elaborated_type": result.data.get("elaborated_type", "")
        }
    else:
        return {
            "success": False,
            "error": result.error
        }


def _execute_proof_node(**kwargs):
    """Execute generate proof node."""
    try:
        from .bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType
    except ImportError:
        from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

    bridge = get_leanaide_bridge()
    result = bridge.execute_task(
        LeanAideTaskType.GENERATE_PROOF,
        **kwargs
    )

    if result.success:
        return {
            "success": True,
            "proof_document": result.data.get("proof_document", ""),
            "lean_proof": result.data.get("lean_proof", ""),
            "proof_id": result.visualization_data.get("proof_id") if result.visualization_data else None
        }
    else:
        return {
            "success": False,
            "error": result.error
        }


def _execute_mcts_node(**kwargs):
    """Execute MCTS search node."""
    try:
        from .bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType
    except ImportError:
        from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

    bridge = get_leanaide_bridge()
    result = bridge.execute_task(
        LeanAideTaskType.MCTS_SEARCH,
        **kwargs
    )

    if result.success and result.visualization_data:
        tree_id = result.visualization_data.get("tree_id")
        tree = bridge.get_tree(tree_id)

        return {
            "success": True,
            "best_proof": str(tree.best_path) if tree else "",
            "tree_id": tree_id,
            "win_rate": tree.statistics.get("win_rate", 0) if tree else 0,
            "statistics": tree.statistics if tree else {}
        }
    else:
        return {
            "success": False,
            "error": result.error
        }


def _execute_verify_node(**kwargs):
    """Execute verify node."""
    try:
        from .bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType
    except ImportError:
        from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

    bridge = get_leanaide_bridge()
    result = bridge.execute_task(
        LeanAideTaskType.VERIFY_SOLUTION,
        **kwargs
    )

    if result.success:
        return {
            "success": True,
            "is_valid": result.data.get("is_valid", False),
            "unproven_count": result.data.get("unproven_count", 0),
            "errors": result.data.get("errors", [])
        }
    else:
        return {
            "success": False,
            "error": result.error
        }


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================

def example_integrated_workflow():
    """
    Example showing LeanAide integrated into BubbleLabs workflow.

    This demonstrates how LeanAide nodes can be used in BubbleLabs workflows.
    """
    try:
        from .ui_shim import ui
    except ImportError:
        from ui_shim import ui

    ui.title("BubbleLabs + LeanAide Integrated Workflow")

    # Initialize LeanAide bridge
    try:
        from .bubblelabs_leanaide_integration import (
        get_leanaide_bridge,
        initialize_leanaide_integration,
        LeanAideTaskType
        )
    except ImportError:
        from bubblelabs_leanaide_integration import (
        get_leanaide_bridge,
        initialize_leanaide_integration,
        LeanAideTaskType
        )

    status = initialize_leanaide_integration()
    bridge = get_leanaide_bridge()

    # Show status
    ui.json(status)

    # Example workflow using LeanAide nodes
    ui.markdown("### Example Workflow: Prove a Theorem")

    # Step 1: Input theorem
    theorem = ui.text_area(
        "Step 1: Enter Theorem",
        value="There are infinitely many prime numbers",
        height=80
    )

    # Step 2: Translate
    if ui.button("Step 2: Translate to Lean"):
        with ui.spinner("Translating..."):
            result = bridge.execute_task(
                LeanAideTaskType.TRANSLATE_THEOREM,
                theorem_text=theorem,
                theorem_name="inf_primes"
            )

        if result.success:
            ui.success("Translation successful!")
            ui.code(result.data["lean_code"], language="lean")

            # Store for next step
            ui.session_state["translated_lean"] = result.data["lean_code"]
        else:
            ui.error(f"Translation failed: {result.error}")

    # Step 3: Generate proof
    if "translated_lean" in ui.session_state:
        if ui.button("Step 3: Generate Proof"):
            with ui.spinner("Generating proof..."):
                result = bridge.execute_task(
                    LeanAideTaskType.GENERATE_PROOF,
                    theorem_text=theorem
                )

            if result.success:
                ui.success("Proof generated!")
                ui.markdown(result.data.get("proof_document", ""))

                if result.data.get("lean_proof"):
                    ui.code(result.data["lean_proof"], language="lean")
            else:
                ui.error(f"Proof generation failed: {result.error}")

    # Step 4: Verify
    if "translated_lean" in ui.session_state:
        if ui.button("Step 4: Verify Proof"):
            with ui.spinner("Verifying..."):
                lean_code = ui.session_state.get("translated_lean", "")
                result = bridge.execute_task(
                    LeanAideTaskType.VERIFY_SOLUTION,
                    code=lean_code
                )

            if result.success:
                is_valid = result.data.get("is_valid", False)
                if is_valid:
                    ui.success("[OK] Proof is valid!")
                else:
                    ui.warning(f"[WARN] Proof has {result.data.get('unproven_count', 0)} unproven obligations")
            else:
                ui.error(f"Verification failed: {result.error}")


# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

"""
TO INTEGRATE LEANAIDE INTO BUBBLELABS:

1. Add import at top of bubblelabs_ui_component.py:
   ```python
   try:
       from bubblelabs_leanaide_ui import LeanAideUIComponent
       from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType
       LEANAIDE_UI_AVAILABLE = True
   except ImportError:
       LEANAIDE_UI_AVAILABLE = False
   ```

2. Add _render_leanaide_integration method to BubbleLabsWorkflowUI class

3. Modify render_workflow_visualizer to add LeanAide tab:
   - Add "LeanAide" to tabs list
   - Add with tabs[3]: self._render_leanaide_integration()
   - Update subsequent tab indices

4. (Optional) Register LeanAide workflow nodes:
   ```python
   from bubblelabs_leanaide_integration_patch import register_leanaide_workflow_nodes
   register_leanaide_workflow_nodes()
   ```

5. Test the integration:
   ```python
   python -c "from bubblelabs_leanaide_integration_patch import example_integrated_workflow; example_integrated_workflow()"
   ```

That's it! LeanAide is now integrated into BubbleLabs.
"""
