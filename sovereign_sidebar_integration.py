"""
Sovereign-Grade Problem Decomposition System - Sidebar Integration
Integrates sovereign decomposition controls into the UI sidebar.
"""

from ui_shim import ui as st
from typing import Optional, Dict, Any

from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_quality_assessment import QualityAssessor
from sovereign_refinement import RefinementCoordinator
from sovereign_knowledge_manager import KnowledgeManager


def render_sovereign_sidebar() -> Optional[Dict[str, Any]]:
    """
    Render sovereign decomposition controls in sidebar.
    
    Returns:
        Dictionary with user selections or None
    """
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Sovereign Decomposition")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Mode",
        ["Quick Decompose", "Advanced Analysis", "View History"],
        help="Select decomposition mode"
    )
    
    if mode == "Quick Decompose":
        return _render_quick_decompose()
    elif mode == "Advanced Analysis":
        return _render_advanced_analysis()
    elif mode == "View History":
        return _render_history_viewer()
    
    return None


def _render_quick_decompose() -> Optional[Dict[str, Any]]:
    """Render quick decomposition interface."""
    st.sidebar.subheader("Quick Decompose")
    
    problem_text = st.sidebar.text_area(
        "Problem Description",
        placeholder="Describe your problem...",
        height=100,
        key="quick_problem"
    )
    
    strategy = st.sidebar.selectbox(
        "Strategy",
        ["auto", "semantic", "dependency", "complexity", "hybrid"],
        help="Select decomposition strategy (auto = best match)"
    )
    
    if st.sidebar.button("🚀 Decompose", use_container_width=True):
        if problem_text:
            return {
                'mode': 'quick',
                'problem_text': problem_text,
                'strategy': strategy
            }
        else:
            st.sidebar.error("Please enter a problem description")
    
    return None


def _render_advanced_analysis() -> Optional[Dict[str, Any]]:
    """Render advanced analysis interface."""
    st.sidebar.subheader("Advanced Analysis")
    
    with st.sidebar.expander("Problem Details", expanded=True):
        title = st.text_input("Title", key="adv_title")
        description = st.text_area("Description", height=80, key="adv_desc")
        problem_type = st.selectbox(
            "Type",
            ["research", "implementation", "analysis", "optimization", "design"]
        )
    
    with st.sidebar.expander("Options"):
        strategy = st.selectbox(
            "Strategy",
            ["auto", "semantic", "dependency", "complexity", "hybrid", "research"]
        )
        
        run_gauntlets = st.checkbox("Run Validation Gauntlets", value=True)
        auto_refine = st.checkbox("Auto-refine if needed", value=True)
        max_refinement = st.slider("Max Refinement Cycles", 1, 5, 3)
    
    if st.sidebar.button("🔬 Analyze", use_container_width=True):
        if title and description:
            return {
                'mode': 'advanced',
                'title': title,
                'description': description,
                'problem_type': problem_type,
                'strategy': strategy,
                'run_gauntlets': run_gauntlets,
                'auto_refine': auto_refine,
                'max_refinement': max_refinement
            }
        else:
            st.sidebar.error("Please fill in title and description")
    
    return None


def _render_history_viewer() -> Optional[Dict[str, Any]]:
    """Render history viewing interface."""
    st.sidebar.subheader("Decomposition History")
    
    # This would connect to actual history storage
    st.sidebar.info("View past decompositions and their quality metrics")
    
    if st.sidebar.button("📊 Load History", use_container_width=True):
        return {
            'mode': 'history',
            'action': 'load'
        }
    
    return None


def render_sovereign_parameters() -> Dict[str, Any]:
    """
    Render sovereign system parameters in sidebar.
    
    Returns:
        Dictionary with parameter values
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ System Parameters")
    
    with st.sidebar.expander("Quality Thresholds"):
        coherence_threshold = st.slider(
            "Coherence",
            0.0, 1.0, 0.75, 0.05,
            key="coherence_threshold"
        )
        
        completeness_threshold = st.slider(
            "Completeness",
            0.0, 1.0, 0.80, 0.05,
            key="completeness_threshold"
        )
        
        feasibility_threshold = st.slider(
            "Feasibility",
            0.0, 1.0, 0.70, 0.05,
            key="feasibility_threshold"
        )
    
    with st.sidebar.expander("Decomposition Limits"):
        max_subproblems = st.number_input(
            "Max Sub-problems",
            min_value=2,
            max_value=20,
            value=7,
            key="max_subproblems"
        )
        
        max_complexity = st.slider(
            "Max Complexity per Sub-problem",
            1.0, 10.0, 7.0, 0.5,
            key="max_complexity"
        )
    
    with st.sidebar.expander("Performance"):
        enable_caching = st.checkbox(
            "Enable Caching",
            value=True,
            key="enable_caching"
        )
        
        parallel_processing = st.checkbox(
            "Parallel Processing",
            value=False,
            key="parallel_processing"
        )
    
    return {
        'quality_thresholds': {
            'coherence': coherence_threshold,
            'completeness': completeness_threshold,
            'feasibility': feasibility_threshold
        },
        'limits': {
            'max_subproblems': max_subproblems,
            'max_complexity': max_complexity
        },
        'performance': {
            'enable_caching': enable_caching,
            'parallel_processing': parallel_processing
        }
    }


def render_sovereign_status() -> None:
    """Render system status indicators in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("Strategies", "5")
        st.metric("Gauntlets", "4")
    
    with col2:
        st.metric("Teams", "3")
        st.metric("Active", "[OK]")
    
    # Knowledge base stats (would connect to actual data)
    with st.sidebar.expander("Knowledge Base"):
        st.metric("Patterns Learned", "0")
        st.metric("Decompositions", "0")
        st.metric("Avg Quality", "N/A")


def render_sovereign_actions() -> Optional[str]:
    """
    Render quick action buttons.
    
    Returns:
        Action name if button clicked
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📝 New", use_container_width=True, key="new_decomp"):
            return "new"
        
        if st.button("💾 Save", use_container_width=True, key="save_decomp"):
            return "save"
    
    with col2:
        if st.button("📂 Load", use_container_width=True, key="load_decomp"):
            return "load"
        
        if st.button("🗑️ Clear", use_container_width=True, key="clear_decomp"):
            return "clear"
    
    return None

