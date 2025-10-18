#!/usr/bin/env python3
"""
OpenEvolve Content Improver - Main Entry Point
A Streamlit application for content improvement using LLMs.

This is the main entry point that ties together all the components of the OpenEvolve application.
"""

import streamlit as st
st.set_page_config(page_title="OpenEvolve", layout="wide")

import os
import sys
import logging
import asyncio
import threading
import time
import requests # For health check
import subprocess
import signal # Added for process termination
import yaml
from openevolve_orchestrator import start_openevolve_services, stop_openevolve_services, restart_openevolve_services
from openevolve_dashboard import render_openevolve_dashboard

import queue

# Global queue for messages from background threads
backend_message_queue = queue.Queue()
# Custom CSS to style the knob and remove the focus "glow" with more stable selectors
custom_css = """
<style>
    /* Target the slider thumb (knob) with more stable selectors */
    [data-testid="stSlider"] div[role="slider"] {
        background: silver !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        /* The glow is often a box-shadow, which we'll disable on focus */
    }

    /* Style the value text inside the slider thumb */
    [data-testid="stSlider"] [data-baseweb="slider"] {
        color: var(--text-primary) !important;
    }

    /* NEW RULE: This specifically targets and removes the "glow" */
    [data-testid="stSlider"] div[role="slider"]:focus {
        box-shadow: none !important;
        outline: none !important;
        border: none !important;
    }
    
    /* Remove any focus styling on the slider container */
    [data-testid="stSlider"] [data-baseweb="slider"]:focus {
        box-shadow: none !important;
        outline: none !important;
        border: none !important;
    }
    
    /* Remove any focus styling on the slider track */
    [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child:focus {
        box-shadow: none !important;
        outline: none !important;
        border: none !important;
    }

    /* --- Updated bar styles with stable selectors --- */

    /* Target the main slider track (the full bar) */
    [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Target the "filled" portion of the slider track */
    [data-testid="stSlider"] [data-baseweb="slider"] > div:last-child {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* General UI stability rules to prevent refresh-related styling issues */
    /* Prevent buttons from changing size on refresh */
    .stButton > button {
        min-width: auto !important;
        flex: none !important;
        padding: 0.275rem 0.6rem !important;
    }
    
    /* Ensure text elements maintain readability during refreshes */
    .stMarkdown, .stText, div[data-testid="stMarkdownContainer"], p, span, div {
        color: var(--text-primary) !important;
    }
    
    /* Maintain consistent container styling */
    [data-testid="stVerticalBlockBorderWrapper"], .stContainer {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    /* Ensure form elements maintain consistent sizing */
    [data-testid="stForm"] {
        margin-bottom: 10px !important;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


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

# Import statements moved here to avoid issues with threading
# Import with error handling for missing modules
try:
    from sidebar import display_sidebar
    logging.info("Successfully imported sidebar module")
except ImportError as e:
    logging.error(f"Failed to import sidebar module: {e}")
    st.error(f"Failed to load sidebar module: {e}")
    display_sidebar = None

try:
    from collaboration import start_collaboration_server
    logging.info("Successfully imported collaboration module")
except ImportError as e:
    logging.error(f"Failed to import collaboration module: {e}")
    st.error(f"Failed to load collaboration module: {e}")
    def start_collaboration_server():
        try:
        from collaboration import CollaborationManager
        st.session_state.collaboration_manager = CollaborationManager()
        logging.info("Collaboration server module loaded successfully")
    except ImportError:
        logging.warning("Collaboration server module not available. Real-time collaboration features will be disabled.")
        st.session_state.collaboration_manager = None

try:
    from mainlayout import render_main_layout
    logging.info("Successfully imported mainlayout module")
except ImportError as e:
    logging.error(f"Failed to import mainlayout module: {e}")
    st.error(f"Failed to load mainlayout module: {e}")
    def render_main_layout():
        st.error("Main layout module not available")

# Initialize session state for backend_started if not present
if "backend_started" not in st.session_state:
    st.session_state["backend_started"] = False

# Start the backend in a separate thread to avoid blocking Streamlit's startup
if not st.session_state["backend_started"]:
    print("Starting backend thread")
    backend_thread = threading.Thread(target=start_openevolve_services)
    backend_thread.daemon = True  # Allow the main program to exit even if the thread is still running
    backend_thread.start()
    st.session_state["backend_started"] = True


def show_welcome_screen():
    """Show a welcome screen for first-time users."""
    if "welcome_shown" not in st.session_state:
        st.session_state.welcome_shown = True

        st.markdown(
            """
        <div style=\"text-align: center; padding: 20px; background: linear-gradient(135deg, #4a6fa5, #6b8cbc); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1 style=\"color: white;">🧬 Welcome to OpenEvolve!</h1>
            <p style=\"font-size: 1.2em; color: #bae6fd;">AI-Powered Content Evolution & Testing Platform</p>
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
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("Error: config.yaml not found. Please ensure it exists in the project root.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error parsing config.yaml: {e}. Please check the file format.")
        st.stop()


def main():
    """Main application entry point."""
    try:
        # Process messages from the backend queue
        while not backend_message_queue.empty():
            message_type, *message_args = backend_message_queue.get_nowait()
            if message_type == "info":
                st.info(message_args[0])
            elif message_type == "warning":
                st.warning(message_args[0])
            elif message_type == "error":
                st.error(message_args[0])
            elif message_type == "process_started":
                service_name, pid = message_args
                if "openevolve_backend_processes" not in st.session_state:
                    st.session_state.openevolve_backend_processes = {}
                st.session_state.openevolve_backend_processes[service_name] = pid # Store PID, not process object

        # Start the Flask app for log streaming in a separate thread, but only once.
        if "log_streaming" not in st.session_state:
            try:
                from log_streaming import LogStreaming
                st.session_state.log_streaming = LogStreaming()
                flask_thread = threading.Thread(target=st.session_state.log_streaming.run_flask_app_in_thread, daemon=True)
                flask_thread.start()
                logging.info("Log streaming service started successfully")
            except ImportError as e:
                logging.error(f"Failed to import log_streaming module: {e}")
                st.error(f"Failed to load log_streaming module: {e}")
            except Exception as e:
                logging.error(f"Failed to start log streaming service: {e}")

        # Load config using cached function for performance
        try:
            config = load_app_config()
        except Exception as e:
            logging.error(f"Failed to load application config: {e}")
            # Provide default config if loading fails
            config = {"default": {}}

        # Set session state from config
        for key, value in config.get("default", {}).items():
            if key not in st.session_state:
                st.session_state[key] = value

        if "evolution_running" not in st.session_state:
            st.session_state.evolution_running = False

        if "adversarial_running" not in st.session_state:
            st.session_state.adversarial_running = False

        if "thread_lock" not in st.session_state:
            st.session_state.thread_lock = threading.Lock()

        # Start the collaboration server
        if start_collaboration_server is not None:
            start_collaboration_server()
        else:
            st.warning("Collaboration server not available")

        # Show welcome screen for first-time users
        show_welcome_screen()

        # Render the sidebar if available
        if display_sidebar is not None:
            display_sidebar()
            
            # Synchronize feature dimensions from sidebar widget if it exists
            if "feature_dimensions_sidebar" in st.session_state:
                st.session_state["feature_dimensions"] = st.session_state["feature_dimensions_sidebar"]
        else:
            st.warning("Sidebar not available")

        # Render the main layout if available
        if render_main_layout is not None:
            render_main_layout()
            
            # Synchronize feature dimensions from main widget if it exists
            if "feature_dimensions_main" in st.session_state:
                st.session_state["feature_dimensions"] = st.session_state["feature_dimensions_main"]
        else:
            st.error("Main layout not available")
        
        # Ensure feature_dimensions session state is initialized if not already set
        if "feature_dimensions" not in st.session_state:
            st.session_state["feature_dimensions"] = ["complexity", "diversity"]

        # Render the OpenEvolve Dashboard
        render_openevolve_dashboard()

    except Exception as e:
        st.error(f"A critical application error occurred: {e}")
        st.exception(e) # Display full traceback for debugging


if __name__ == "__main__":
    # Initialize session state only once
    if "_session_state_initialized" not in st.session_state:
        from mainlayout import _initialize_session_state
        _initialize_session_state()
        st.session_state._session_state_initialized = True

    main()

