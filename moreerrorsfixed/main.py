#!/usr/bin/env python3
"""
OpenEvolve Content Improver - Main Entry Point
A Streamlit application for content improvement using LLMs.

This is the main entry point that ties together all the components of the OpenEvolve application.
"""

import streamlit as st
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

# Platform-specific asyncio configuration
if sys.platform == "win32":
    try:
        # Only set event loop policy if we're in the main thread and there's no running loop
        if (threading.current_thread() is threading.main_thread() and 
            not asyncio.get_event_loop_policy()._local._loop):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        logging.debug(f"Could not set Windows event loop policy: {e}")
        pass  # Silently ignore as this is not critical for the application

def get_project_root():
    """
    Returns the absolute path to the project's root directory.
    This is a local copy to avoid import issues in threaded contexts.
    """
    try:
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to get the project root
        return os.path.abspath(os.path.join(current_dir, os.pardir))
    except Exception as e:
        logging.error(f"Error getting project root: {e}")
        # Fallback to current working directory
        return os.getcwd()

def start_openevolve_backend():
    try:
        backend_path = os.path.join(get_project_root(), "openevolve")
        backend_script = os.path.join(backend_path, "openevolve-run.py")
        
        # Verify backend script exists
        if not os.path.exists(backend_script):
            logging.warning(f"OpenEvolve backend script not found at {backend_script}")
            return
    except Exception as e:
        logging.error(f"Error constructing backend paths: {e}")
        return
    
    # Check if backend is already running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5) # Increased timeout
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
        # Create log files in frontend directory to capture backend output for debugging
        backend_out_log = os.path.join(os.path.dirname(__file__), "backend_stdout.log")
        backend_err_log = os.path.join(os.path.dirname(__file__), "backend_stderr.log")
        
        with open(backend_out_log, "w") as stdout_file, open(backend_err_log, "w") as stderr_file:
            # We use a separate thread to run this to ensure it doesn't interfere with Streamlit's main loop
            process = subprocess.Popen(
                command,
                cwd=backend_path,
                stdout=stdout_file,
                stderr=stderr_file,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
        
        logging.info(f"OpenEvolve backend started with PID: {process.pid}")
        
        # Wait a bit for the backend to start
        time.sleep(2)
        
        # Double-check if backend is running after starting
        max_retries = 10
        retry_count = 0
        backend_started = False
        
        while retry_count < max_retries and not backend_started:
            try:
                response = requests.get("http://localhost:8000/health", timeout=3)
                if response.status_code == 200:
                    logging.info("OpenEvolve backend confirmed running.")
                    backend_started = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)  # Wait 1 second before retrying
            retry_count += 1
            
        if not backend_started:
            logging.warning("OpenEvolve backend may not have started properly. Please check backend logs.")
            
    except Exception as e:
        logging.error(f"Failed to start OpenEvolve backend: {e}")

# Import statements moved here to avoid issues with threading
from sidebar import display_sidebar
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
from mainlayout import render_main_layout
from config_data import load_config, save_config

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


@st.cache_data
def load_app_config():
    """Loads application configuration from config.yaml, caching the result."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    """Main application entry point."""


    # Start the Flask app for log streaming in a separate thread, but only once.
    if "log_streaming" not in st.session_state:
        from log_streaming import LogStreaming
        st.session_state.log_streaming = LogStreaming()
        flask_thread = threading.Thread(target=st.session_state.log_streaming.run_flask_app_in_thread, daemon=True)
        flask_thread.start()

    # Load config using cached function for performance
    config = load_app_config()

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



    # Start the collaboration server
    start_collaboration_server()

    # Show welcome screen for first-time users
    show_welcome_screen()

    # Render the sidebar
    display_sidebar()

    # Render the main layout
    if st.session_state.get("page") == "evaluator_uploader":
        from evaluator_uploader import render_evaluator_uploader

        render_evaluator_uploader()
    elif st.session_state.get("page") == "prompt_manager":
        # PromptManager is instantiated in mainlayout.py and stored in session_state
        # Ensure mainlayout.py has been rendered at least once to have prompt_manager in session_state
        if "prompt_manager" in st.session_state:
            st.session_state.prompt_manager.render_prompt_manager_ui()
        else:
            st.warning("Prompt Manager not initialized. Please navigate to the main layout first.")
    elif st.session_state.get("page") == "analytics_dashboard":
        from analytics_dashboard import render_analytics_dashboard

        render_analytics_dashboard()
    else:
        render_main_layout()


if __name__ == "__main__":
    # Initialize session state only once
    if "_session_state_initialized" not in st.session_state:
        from mainlayout import _initialize_session_state
        _initialize_session_state()
        st.session_state._session_state_initialized = True
    main()
