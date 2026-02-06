"""
Demo: OpenEvolve + BubbleLabs UI Integration

This script demonstrates the complete UI integration.

Usage:
    python demo_ui_integration.py

Note: The BubbleLab UI now lives in the TypeScript app. This demo module
keeps the legacy wiring in place via the UI shim for local smoke checks.
"""

from ui_shim import ui as st
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="OpenEvolve + BubbleLabs Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main demo application."""

    # Import UI component
    try:
        from openevolve_bubblelabs_ui import render_openevolve_bubblelabs_ui
    except ImportError as e:
        st.error(f"[FAIL] Failed to import: {e}")
        st.error("Make sure openevolve_workflow_manager_integrated.py and openevolve_bubblelabs_ui.py are in the current directory.")
        st.stop()

    # Render the UI
    render_openevolve_bubblelabs_ui()

    # Add instructions at the bottom
    st.divider()
    st.subheader("📖 Quick Start Guide")

    with st.expander("How to Use This Demo", expanded=True):
        st.markdown("""
        ### Step 1: Create a Workflow

        1. Click the **"📋 Create Workflow"** tab
        2. Enter a **workflow name** (e.g., "Optimization Workflow")
        3. Enter a **problem statement** (e.g., "How can we optimize database query performance?")
        4. **Select teams** for each stage:
           - Content Analyzer Team
           - Planner Team
           - Solver Team
           - Assembler Team
        5. (Optional) Select **gauntlets** for verification
        6. Click **"Create Workflow"**

        ### Step 2: Execute Workflow

        1. Click the **"▶️ Execute Workflow"** tab
        2. Select your workflow from the dropdown
        3. Review the workflow configuration
        4. Click **"▶️ Execute Workflow"**
        5. Wait for execution to complete
        6. View the results

        ### Step 3: Monitor Progress

        1. Click the **"📊 Monitor & Control"** tab
        2. View all workflows and their status
        3. Expand a workflow to see details
        4. Use control buttons (pause/resume/cancel)

        ### Step 4: View Analytics

        1. Click the **"📈 Analytics"** tab
        2. View workflow execution metrics
        3. Check node-level analytics
        4. Review provider performance

        ### Features

        [OK] **Uses Actual Workflow Files:**
        - workflow_structures.py (WorkflowState)
        - workflow_engine.py (run_content_analysis, run_ai_decomposition, etc.)
        - team_manager.py (TeamManager)
        - gauntlet_manager.py (GauntletManager)

        [OK] **Full Workflow Lifecycle:**
        - Create workflows from actual teams
        - Execute using actual workflow functions
        - Monitor with real-time progress
        - Control with pause/resume/cancel

        [OK] **Visual Interface:**
        - Streamlit-based UI
        - Team and gauntlet selection
        - Progress tracking
        - Results display
        """)

    st.info("💡 **Tip:** Create teams in Team Manager first, then create workflows!")

    # System status
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            from team_manager import TeamManager
            tm = TeamManager()
            teams = tm.list_teams()
            st.metric("Available Teams", len(teams))
        except:
            st.metric("Available Teams", "❓")

    with col2:
        try:
            from gauntlet_manager import GauntletManager
            gm = GauntletManager()
            gauntlets = gm.list_gauntlets()
            st.metric("Available Gauntlets", len(gauntlets))
        except:
            st.metric("Available Gauntlets", "❓")

    with col3:
        try:
            from openevolve_workflow_manager_integrated import OpenEvolveWorkflowManager
            manager = OpenEvolveWorkflowManager()
            workflows = manager.list_workflows()
            st.metric("Created Workflows", len(workflows))
        except:
            st.metric("Created Workflows", "❓")


if __name__ == "__main__":
    main()
