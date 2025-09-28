#!/usr/bin/env python3
"""
OpenEvolve Content Improver - Main Entry Point
A Streamlit application for content improvement using LLMs.

This is the main entry point that ties together all the components of the OpenEvolve application.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import yaml
import threading
from log_streaming import run_flask_app, log_queue

import mainlayout
# Explicitly import functions used directly in main()
from sidebar import render_sidebar
from collaboration import start_collaboration_server

try:
    # Import modules to register their functionality (suppressing F401 warnings)
    import providercatalogue  # noqa: F401
    import evolution  # noqa: F401
    import adversarial  # noqa: F401
    import integrations  # noqa: F401
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()


def show_welcome_screen():
    """Show a welcome screen for first-time users."""
    if "welcome_shown" not in st.session_state:
        st.session_state.welcome_shown = True

        st.markdown(
            """
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #4a6fa5, #6b8cbc); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white;">🧬 Welcome to OpenEvolve!</h1>
            <p style="font-size: 1.2em; color: #e0e0e0;">AI-Powered Content Evolution & Testing Platform</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🚀 Core Features")
            st.markdown("""
            - **Universal Content Support**: Works with any text-based content
            - **Intelligent Processing**: Automatic routing to appropriate processors
            - **Multi-LLM Ensemble**: Leverages multiple AI models for diverse perspectives
            - **Real-time Monitoring**: Track improvements as they happen
            """)

        with col2:
            st.markdown("### ⚔️ Advanced Testing")
            st.markdown("""
            - **Adversarial Testing**: Red team/blue team approach for hardening
            - **Multi-Model Consensus**: Uses diverse AI models for robust testing
            - **Confidence Tracking**: Statistical confidence metrics for reliability
            - **Compliance Checking**: Automatic compliance verification
            """)

        with col3:
            st.markdown("### 👥 Collaboration")
            st.markdown("""
            - **Real-time Editing**: Multiple users can collaborate simultaneously
            - **Version Control**: Complete history with branching and tagging
            - **Commenting System**: Threaded discussions with mentions
            - **Project Sharing**: Secure sharing with password protection
            """)

        st.markdown("---")

        st.info(
            "💡 **Tip**: Start with a template from our marketplace if you're unsure where to begin!"
        )

        # Quick start guide
        with st.expander("📋 Quick Start Guide", expanded=False):
            st.markdown("""
            ### Getting Started:
            1. **Configure your LLM provider** in the sidebar (OpenAI, Anthropic, Google, etc.)
            2. **Enter your content** in the main area or load a template
            3. **Choose your approach**:
               - Use **Evolution** for iterative content improvement
               - Use **Adversarial Testing** for red team/blue team hardening
            4. **Monitor progress** in real-time
            5. **Export your improved content** in your preferred format
            
            ### Content Types Supported:
            - **Code**: Python, JavaScript, Java, C++, C#, Go, Rust (and more)
            - **Documents**: Protocols, procedures, SOPs, policies
            - **General Content**: Any text-based content
            
            ### Pro Tips:
            - Enable **Adversarial Testing** after initial evolution for maximum hardening
            - Use the **Template Marketplace** for industry-specific starting points
            - Enable **Multi-Model Ensembles** for more robust improvements
            - Check the **Advanced Analytics** dashboard for detailed performance metrics
            """)


def main():
    """Main application entry point."""
    # Start the Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set session state from config
    for key, value in config["default"].items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.session_state.log_queue = log_queue

    # Start the collaboration server
    start_collaboration_server()

    # Show welcome screen for first-time users
    show_welcome_screen()

    # Render the sidebar
    render_sidebar()

    # Render the main layout
    if st.session_state.get("page") == "evaluator_uploader":
        from evaluator_uploader import render_evaluator_uploader

        render_evaluator_uploader()
    elif st.session_state.get("page") == "prompt_manager":
        from prompt_manager import render_prompt_manager

        render_prompt_manager()
    elif st.session_state.get("page") == "analytics_dashboard":
        from analytics_dashboard import render_analytics_dashboard

        render_analytics_dashboard()
    else:
        mainlayout.render_main_layout()


if __name__ == "__main__":
    main()
