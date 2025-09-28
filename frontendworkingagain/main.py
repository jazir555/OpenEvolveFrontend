#!/usr/bin/env python3
"""
OpenEvolve Content Improver - Main Entry Point
A Streamlit application for content improvement using LLMs.

This is the main entry point that ties together all the components of the OpenEvolve application.
"""

import streamlit as st
from session_utils import get_project_root
from sidebar import render_sidebar

from prompt_manager import handle_prompt_input
from analytics_dashboard import render_analytics_dashboard
from collaboration_manager import render_collaboration_section
from collaboration import start_collaboration_server
from rbac import render_rbac_settings
from template_manager import render_template_manager
from export_import_manager import render_export_import_manager
from version_control import render_version_control
from notifications import render_notifications
from suggestions import render_suggestions
from tasks import render_tasks
from validation_manager import render_validation_manager
from content_manager import render_content_manager
from providers import render_provider_settings
from analytics import render_analytics_settings
from evolution import render_evolution_settings
from adversarial import render_adversarial_settings
from log_streaming import render_log_streaming, run_flask_app, log_queue
from config_data import load_config, save_config
import os
import sys
import logging
import asyncio
import threading
import time
import requests # For health check
import subprocess
import yaml # Added import for yaml

# Configure logging for the backend launcher
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_openevolve_backend():
    backend_path = os.path.join(get_project_root(), "openevolve")
    backend_script = os.path.join(backend_path, "openevolve-run.py")
    
    # Check if backend is already running
    try:
        response = requests.get("http://localhost:8000/health", timeout=1) # Assuming a health endpoint
        if response.status_code == 200:
            logging.info("OpenEvolve backend is already running.")
            return
    except requests.exceptions.ConnectionError:
        logging.info("OpenEvolve backend not running, starting it now...")
    except requests.exceptions.Timeout:
        logging.warning("Health check timed out, assuming backend is not fully ready or not running.")
    except Exception as e:
        logging.error(f"Error during backend health check: {e}")

    command = [sys.executable, backend_script]
    
    try:
        # Use Popen to start the backend without blocking the main thread
        # stdout and stderr are redirected to files to prevent blocking and capture output
        # We use a separate thread to run this to ensure it doesn't interfere with Streamlit's main loop
        process = subprocess.Popen(
            command,
            cwd=backend_path,
            stdout=subprocess.DEVNULL, # Redirect stdout to null
            stderr=subprocess.DEVNULL, # Redirect stderr to null
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        logging.info(f"OpenEvolve backend started with PID: {process.pid}")
    except Exception as e:
        logging.error(f"Failed to start OpenEvolve backend: {e}")

# Initialize session state for backend_started if not present
if "backend_started" not in st.session_state:
    st.session_state["backend_started"] = False

# Start the backend in a separate thread to avoid blocking Streamlit's startup
if not st.session_state["backend_started"]:
    backend_thread = threading.Thread(target=start_openevolve_backend)
    backend_thread.daemon = True  # Allow the main program to exit even if the thread is still running
    backend_thread.start()
    st.session_state["backend_started"] = True


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

    if "evolution_running" not in st.session_state:
        st.session_state.evolution_running = False

    if "adversarial_running" not in st.session_state:
        st.session_state.adversarial_running = False

    if "thread_lock" not in st.session_state:
        st.session_state.thread_lock = threading.Lock()

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
