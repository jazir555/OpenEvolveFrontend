from __future__ import annotations

# UI Component Classes (Test Compatibility)

class UIComponentFactory:
    """Factory for creating UI components."""
    def __init__(self):
        self.components = {}
    def create_component(self, component_type, **kwargs):
        return {'type': component_type, 'props': kwargs}

class WorkflowVisualizer:
    """Visualizer for workflows."""
    def __init__(self):
        self.workflows = []
    def generate_config(self, workflow):
        return {'workflow': workflow, 'layout': 'vertical'}

class MetricsDisplay:
    """Display for metrics."""
    def __init__(self):
        self.metrics = {}
    def render(self, metrics):
        return f'<div class="metrics">Metrics: {metrics}</div>'

class CodeEditor:
    """Code editor component."""
    def __init__(self):
        self.config = {}
    def get_config(self, language='python', theme='dark'):
        return {'language': language, 'theme': theme}

class ProgressIndicator:
    """Progress indicator component."""
    def __init__(self):
        self.current = 0
        self.total = 100
    def render(self, current, total, label=''):
        percentage = (current / total * 100) if total > 0 else 0
        return f'<div class="progress">{label}: {percentage:.0f}%</div>'

class SidebarNavigation:
    """Sidebar navigation component."""
    def __init__(self):
        self.items = []
    def generate_config(self, items):
        return {'items': items, 'layout': 'sidebar'}
"""
BubbleLabs Extended UI Components

UI UI components for the extended BubbleLabs integration,
providing visualization and control for all OpenEvolve components.

License: MIT
Author: OpenEvolve Team
Date: 2026-02-03
"""

from ui_shim import ui as st
import pandas as pd
import json
import time
from typing import Dict, Any, List, Optional

# Import the extended integration
try:
    from bubblelabs_extended_integration import (
        get_extended_integration,
        initialize_extended_integration,
        get_all_integration_status,
        ComponentStatus,
    )
    EXTENDED_INTEGRATION_AVAILABLE = True
except ImportError:
    EXTENDED_INTEGRATION_AVAILABLE = False


def render_component_card(
    component_name: str,
    status: Dict[str, Any],
    expanded: bool = False
) -> None:
    """Render a component status card."""
    status_value = status.get("status", "unknown")
    
    # Status color
    if status_value == "available":
        status_color = "green"
        status_icon = "[OK]"
    elif status_value == "unavailable":
        status_color = "red"
        status_icon = "[FAIL]"
    elif status_value == "loading":
        status_color = "yellow"
        status_icon = "⟳"
    else:
        status_color = "gray"
        status_icon = "?"
    
    with st.expander(f"{status_icon} {component_name}: {status_value.upper()}", expanded=expanded):
        st.markdown(f"**Version:** {status.get('version', 'N/A')}")
        
        capabilities = status.get("capabilities", [])
        if capabilities:
            st.markdown("**Capabilities:**")
            cols = st.columns(3)
            for i, cap in enumerate(capabilities):
                cols[i % 3].badge(cap, icon="⚙️")


def render_integration_dashboard() -> None:
    """Render the main integration dashboard."""
    st.title("🔧 BubbleLabs Extended Integration")
    st.markdown("Manage and monitor all OpenEvolve component integrations")
    
    if not EXTENDED_INTEGRATION_AVAILABLE:
        st.error("Extended integration module not available")
        return
    
    # Refresh button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Get status
    status = get_all_integration_status()
    
    # Summary metrics
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Components",
            status["total_components"],
            delta=None,
        )
    with col2:
        available = status["available_components"]
        total = status["total_components"]
        st.metric(
            "Available",
            available,
            delta=f"{available}/{total} ready",
        )
    with col3:
        pct = (available / total * 100) if total > 0 else 0
        st.metric(
            "Health",
            f"{pct:.0f}%",
            delta=f"{'Good' if pct >= 70 else 'Needs Attention'}",
        )
    
    st.divider()
    
    # Component cards
    st.subheader("Component Status")
    
    for name, component in status["components"].items():
        render_component_card(name.upper(), component, expanded=False)


def render_ace_component() -> None:
    """Render ACE component controls."""
    st.header("🧠 ACE (Agentic Context Engine)")
    
    integration = get_extended_integration()
    
    # Create skillbook
    with st.form("ace_skillbook_form"):
        st.markdown("### Create Skillbook")
        name = st.text_input("Skillbook Name", placeholder="my_skillbook")
        skills_text = st.text_area(
            "Skills (JSON)",
            placeholder='[{"name": "skill1", "description": "..."}]',
            height=100,
        )
        
        if st.form_submit_button("Create Skillbook"):
            try:
                skills = json.loads(skills_text) if skills_text else []
                result = integration.ace_create_skillbook(name, skills)
                if result["success"]:
                    st.success(f"Skillbook created: {result['skillbook_id']}")
                else:
                    st.error(f"Failed: {result.get('error')}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    
    st.divider()
    
    # Extract patterns
    st.markdown("### Extract Patterns")
    with st.form("ace_patterns_form"):
        results_text = st.text_area(
            "Workflow Results (JSON)",
            placeholder='[{"result": "..."}]',
            height=100,
        )
        
        if st.form_submit_button("Extract Patterns"):
            try:
                results = json.loads(results_text) if results_text else []
                result = integration.ace_extract_patterns(results)
                if result["success"]:
                    st.success(f"Extracted {result['patterns_extracted']} patterns")
                else:
                    st.error(f"Failed: {result.get('error')}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")


def render_z3_component() -> None:
    """Render Z3 Prover component controls."""
    st.header("🔬 Z3 Prover")
    
    integration = get_extended_integration()
    
    # Constraint solver
    with st.form("z3_solver_form"):
        st.markdown("### Solve Constraints")
        
        variables_text = st.text_area(
            "Variables (JSON)",
            placeholder='[{"name": "x", "type": "Int"}]',
            height=80,
        )
        
        constraints_text = st.text_area(
            "Constraints",
            placeholder="(> x 0)\n(< x 10)",
            height=80,
        )
        
        if st.form_submit_button("Solve"):
            try:
                variables = json.loads(variables_text) if variables_text else []
                constraints = constraints_text.split("\n") if constraints_text else []
                result = integration.z3_solve_constraints(variables, constraints)
                if result["success"]:
                    st.success(f"Solver created: {result['solver_id']}")
                    st.json(result)
                else:
                    st.error(f"Failed: {result.get('error')}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    
    st.divider()
    
    # Theorem prover
    st.markdown("### Prove Theorem")
    with st.form("z3_prover_form"):
        theorem = st.text_area(
            "Theorem Statement",
            placeholder="forall x: (x > 0) => (x >= 1)",
            height=80,
        )
        
        if st.form_submit_button("Prove"):
            result = integration.z3_prove_theorem(theorem)
            if result["success"]:
                st.success(f"Theorem submitted: {result['status']}")
            else:
                st.error(f"Failed: {result.get('error')}")


def render_roma_component() -> None:
    """Render ROMA component controls."""
    st.header("🔄 ROMA (Recursive Object Model Architecture)")
    
    integration = get_extended_integration()
    
    # Analyze problem
    with st.form("roma_analyze_form"):
        st.markdown("### Analyze Problem")
        
        problem = st.text_area(
            "Problem Statement",
            placeholder="Design a system for...",
            height=100,
        )
        
        max_depth = st.slider("Max Depth", 1, 10, 3)
        
        if st.form_submit_button("Analyze"):
            result = integration.roma_analyze_problem(problem, max_depth)
            if result["success"]:
                st.success(f"Analysis started: {result['status']}")
            else:
                st.error(f"Failed: {result.get('error')}")
    
    st.divider()
    
    # Create config
    st.markdown("### Create Configuration")
    with st.form("roma_config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            mdap_k_ahead = st.number_input("MDAP K-Ahead", 2, 100, 5)
        with col2:
            roma_max_depth = st.number_input("ROMA Max Depth", 1, 20, 5)
        
        enable_red_flagging = st.checkbox("Enable Red Flagging", value=True)
        
        if st.form_submit_button("Create Config"):
            result = integration.roma_create_config(
                mdap_k_ahead=mdap_k_ahead,
                roma_max_depth=roma_max_depth,
                enable_red_flagging=enable_red_flagging,
            )
            if result["success"]:
                st.success("Configuration created")
                st.code(result.get('config', ''))
            else:
                st.error(f"Failed: {result.get('error')}")


def render_knowledge_component() -> None:
    """Render Knowledge Graph component controls."""
    st.header("📚 Knowledge Graph")
    
    integration = get_extended_integration()
    
    # Store artifact
    with st.form("knowledge_store_form"):
        st.markdown("### Store Artifact")
        
        artifact_text = st.text_area(
            "Artifact (JSON)",
            placeholder='{"type": "pattern", "content": "..."}',
            height=100,
        )
        
        if st.form_submit_button("Store"):
            try:
                artifact = json.loads(artifact_text) if artifact_text else {}
                result = integration.knowledge_store_artifact(artifact)
                if result["success"]:
                    st.success(f"Artifact stored: {result['artifact_id']}")
                else:
                    st.error(f"Failed: {result.get('error')}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    
    st.divider()
    
    # Query patterns
    st.markdown("### Query Patterns")
    with st.form("knowledge_query_form"):
        query = st.text_input("Query String", placeholder="search term...")
        
        if st.form_submit_button("Query"):
            result = integration.knowledge_query_patterns(query)
            if result["success"]:
                st.success(f"Query executed: {result['query']}")
                st.json(result)
            else:
                st.error(f"Failed: {result.get('error')}")


def render_analytics_component() -> None:
    """Render Analytics component controls."""
    st.header("📊 Analytics")
    
    integration = get_extended_integration()
    
    # Track workflow
    with st.form("analytics_track_form"):
        st.markdown("### Track Workflow")
        
        workflow_id = st.text_input("Workflow ID", placeholder="workflow_123")
        metrics_text = st.text_area(
            "Metrics (JSON)",
            placeholder='{"duration": 100, "accuracy": 0.95}',
            height=80,
        )
        
        if st.form_submit_button("Track"):
            try:
                metrics = json.loads(metrics_text) if metrics_text else {}
                result = integration.analytics_track_workflow(workflow_id, metrics)
                if result["success"]:
                    st.success(f"Workflow tracked: {result['metrics_recorded']} metrics")
                else:
                    st.error(f"Failed: {result.get('error')}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    
    st.divider()
    
    # Get dashboard
    st.markdown("### Dashboard")
    if st.button("Load Dashboard"):
        result = integration.analytics_get_dashboard()
        if result["success"]:
            st.success("Dashboard data loaded")
            st.json(result.get('dashboard_data', {}))
        else:
            st.error(f"Failed: {result.get('error')}")


def render_leanaide_component() -> None:
    """Render LeanAIDE component controls."""
    st.header("📐 LeanAIDE (Lean 4)")
    
    integration = get_extended_integration()
    
    # Prove theorem
    with st.form("leanaide_prove_form"):
        st.markdown("### Prove Theorem")
        
        theorem = st.text_area(
            "Theorem Statement (Lean 4)",
            placeholder="theorem my_theorem : ∀ n : ℕ, n ≥ 0 := by simp",
            height=100,
        )
        
        if st.form_submit_button("Prove"):
            result = integration.leanaide_prove_theorem(theorem)
            if result["success"]:
                st.success(f"Theorem submitted: {result['status']}")
            else:
                st.error(f"Failed: {result.get('error')}")


def render_ragbits_component() -> None:
    """Render Ragbits component controls."""
    st.header("📚 Ragbits (RAG + Knowledge)")
    
    integration = get_extended_integration()
    
    # Statistics
    stats_result = integration.execute_control_action("ragbits", "stats")
    if stats_result["success"]:
        stats = stats_result.get("result", {}).get("stats", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Documents", stats.get("ingested_documents", 0))
        col2.metric("Vector Store", stats.get("vector_store_type", "N/A").upper())
        col3.metric("Available", "YES" if stats.get("available") else "NO")
    
    st.divider()
    
    # Search knowledge base
    with st.form("ragbits_search_form"):
        st.markdown("### Search Knowledge Base")
        query = st.text_input("Search Query", placeholder="How to implement...")
        top_k = st.slider("Results Count", 1, 20, 5)
        
        if st.form_submit_button("Search"):
            result = integration.execute_control_action("ragbits", "search", {"query": query, "top_k": top_k})
            if result["success"]:
                search_res = result.get("result", {})
                st.success(f"Found {search_res.get('count', 0)} results")
                for i, doc in enumerate(search_res.get("results", [])):
                    with st.expander(f"Result {i+1} (Score: {doc.get('score', 0.0):.2f})"):
                        st.markdown(doc.get("content", ""))
                        if doc.get("metadata"):
                            st.json(doc.get("metadata"))
            else:
                st.error(f"Failed: {result.get('error')}")
    
    st.divider()
    
    # Ingest document (Simulated for UI)
    st.markdown("### Quick Ingest")
    with st.form("ragbits_ingest_form"):
        content = st.text_area("Content to Index", height=150)
        source = st.text_input("Source Label", value="manual_ingest")
        
        if st.form_submit_button("Index Content"):
            # In a real scenario, we'd have a specific control action for this
            # For now, we'll show a message
            st.info("Indexing requested. This feature is being wired to the backend.")


def render_bubblelab_components_tab() -> None:
    """Render the BubbleLab Components (TypeScript) integration panel.

    Bridges the Python BubbleLab UI to the ``@openevolve/bubblelab-components``
    TS package so every configuration knob is reachable. Degrades gracefully when
    the TS toolchain or build output is missing.
    """
    st.header("🔌 BubbleLab Components (TypeScript)")

    try:
        from bubblelab_components_bridge import get_bridge
    except Exception as exc:  # pragma: no cover - bridge is colocated
        st.error(f"BubbleLab Components bridge not importable: {exc}")
        return

    bridge = get_bridge()
    status = bridge.status()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Package Found", "YES" if status["available"] else "NO")
    col2.metric("Manifest", "YES" if status["has_manifest"] else "NO")
    col3.metric("Components", status["component_count"])
    col4.metric("Config Knobs", status["knob_count"])

    if status["package_dir"]:
        st.markdown(f"**Package:** `{status['package_dir']}`")
    if status.get("build_script_configured"):
        st.success("TS `tsc` build script (`build:components`) is configured.")
    else:
        st.warning("TS `build:components` script not detected in the package.")

    if not status["has_manifest"]:
        st.error(
            "Component manifest not found. The Python side cannot enumerate config "
            "knobs until the TS package is present."
        )
        return

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔨 Build TS components (tsc)", use_container_width=True):
            with st.spinner("Running `npm run build:components`..."):
                result = bridge.build()
            if result.get("success"):
                st.success("Build succeeded.")
            else:
                st.error(f"Build did not complete: {result.get('reason')}")
                if result.get("stderr"):
                    st.code(result.get("stderr", "")[:2000])
    with c2:
        if st.button("🌐 Serve built components", use_container_width=True):
            srv = bridge.serve()
            st.success(f"Serving at {srv['url']} (mode={srv['mode']})")

    st.divider()
    st.subheader("Configuration Knobs (every BubbleLab UI control)")
    for comp in bridge.get_components():
        with st.expander(f"{comp['name']}  ·  {len(comp.get('knobs', []))} knobs"):
            for knob in comp.get("knobs", []):
                ctrl = knob.get("control", "?")
                if knob.get("options"):
                    extra = f" options={knob['options']}"
                elif "min" in knob or "max" in knob:
                    extra = f" range=[{knob.get('min')}, {knob.get('max')}]"
                else:
                    extra = ""
                st.markdown(f"- `{knob['id']}` — {knob['label']} ({ctrl}){extra}")


def render_component_tab(tab_name: str) -> None:
    """Render a specific component tab."""
    if tab_name == "ace":
        render_ace_component()
    elif tab_name == "z3":
        render_z3_component()
    elif tab_name == "roma":
        render_roma_component()
    elif tab_name == "knowledge":
        render_knowledge_component()
    elif tab_name == "analytics":
        render_analytics_component()
    elif tab_name == "leanaide":
        render_leanaide_component()
    elif tab_name == "ragbits":
        render_ragbits_component()
    elif tab_name == "bubblelab_components":
        render_bubblelab_components_tab()


def render_extended_ui() -> None:
    """Render the complete extended UI."""
    st.set_page_config(
        page_title="BubbleLabs Extended Integration",
        page_icon="🔧",
        layout="wide",
    )
    
    st.title("🔧 BubbleLabs Extended Integration")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "ACE", "Z3", "ROMA", "Knowledge", "Analytics", "LeanAIDE", "Ragbits", "BubbleLab Components"],
    )
    
    # Sidebar - Status summary
    st.sidebar.divider()
    st.sidebar.markdown("### Quick Status")
    
    if EXTENDED_INTEGRATION_AVAILABLE:
        status = get_all_integration_status()
        available = status["available_components"]
        total = status["total_components"]
        st.sidebar.metric("Components Ready", f"{available}/{total}")
        
        for name, component in status["components"].items():
            status_val = component.get("status", "unknown")
            icon = "[OK]" if status_val == "available" else "[FAIL]"
            st.sidebar.text(f"{icon} {name}")
    else:
        st.sidebar.error("Integration not available")
    
    # Main content
    if page == "Dashboard":
        render_integration_dashboard()
    elif page == "ACE":
        render_ace_component()
    elif page == "Z3":
        render_z3_component()
    elif page == "ROMA":
        render_roma_component()
    elif page == "Knowledge":
        render_knowledge_component()
    elif page == "Analytics":
        render_analytics_component()
    elif page == "LeanAIDE":
        render_leanaide_component()
    elif page == "Ragbits":
        render_ragbits_component()
    elif page == "BubbleLab Components":
        render_bubblelab_components_tab()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Run the UI
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode - just check integrations
        print("Testing BubbleLabs Extended Integration...")
        
        if EXTENDED_INTEGRATION_AVAILABLE:
            print("[OK] Extended integration module available")
            
            results = initialize_extended_integration()
            print(f"\nInitialized {len(results)} components:")
            
            for name, result in results.items():
                status = "[OK]" if result["success"] else "[FAIL]"
                print(f"  {status} {name}: {result['status']}")
            
            print("\n" + "=" * 50)
            status = get_all_integration_status()
            print(f"Available: {status['available_components']}/{status['total_components']}")
        else:
            print("[FAIL] Extended integration module not available")
    else:
        # Run UI
        render_extended_ui()


# =============================================================================
# TEST COMPATIBILITY CLASS
# =============================================================================

class BubbleLabsUIComponent:
    """
    Wrapper class for test compatibility.

    This class provides a simple interface for tests to interact with BubbleLabs
    UI components without requiring the full UI infrastructure.
    """

    def __init__(self):
        """Initialize the UI component."""
        self.available = EXTENDED_INTEGRATION_AVAILABLE
        self.status = "available" if self.available else "unavailable"

    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        if self.available:
            return get_all_integration_status()
        return {
            "available_components": 0,
            "total_components": 0,
            "status": "unavailable"
        }

    def initialize(self) -> Dict[str, Any]:
        """Initialize the component."""
        if not self.available:
            return {"success": False, "status": "unavailable"}

        try:
            results = initialize_extended_integration()
            return {
                "success": True,
                "status": "available",
                "results": results
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def render(self, **kwargs) -> str:
        """Render the UI component."""
        if self.available:
            return "<div>BubbleLabs UI Component</div>"
        return "<div>Component Unavailable</div>"

