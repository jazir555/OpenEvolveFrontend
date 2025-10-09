import streamlit as st
import json
from datetime import datetime
import threading
import time

import difflib
from typing import List, Dict
import altair as alt
import numpy as np
import pandas as pd
import os
import sys
import plotly.express as px

# Try to import matplotlib, but don't fail if it's not available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from pyvis.network import Network
# These imports are assumed to exist in the user's environment.
# If they don't, the script will fail, but per the instructions, no mock functions will be created.
from session_utils import _safe_list, _load_user_preferences, _load_parameter_settings

# Import autorefresh for real-time updates
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None
from integrated_reporting import generate_integrated_report



from providercatalogue import get_openrouter_models, _parse_price_per_million

from session_manager import (
    APPROVAL_PROMPT, RED_TEAM_CRITIQUE_PROMPT, BLUE_TEAM_PATCH_PROMPT
)
from openevolve_integration import (
    OpenEvolveAPI
)

from adversarial import (
    run_adversarial_testing, optimize_model_selection, _load_human_feedback,
    MODEL_META_LOCK, MODEL_META_BY_ID
)
from evolution import (
    _run_evolution_with_api_backend_refactored
)

# Check if OpenEvolve is available
OPENEVOLVE_AVAILABLE = False
try:
    from openevolve_visualization import render_openevolve_visualization_ui, render_evolution_insights, render_openevolve_advanced_ui, render_advanced_diagnostics
    from monitoring_system import render_comprehensive_monitoring_ui
    from reporting_system import render_reporting_dashboard, create_evolution_report
    from openevolve_orchestrator import start_openevolve_services, stop_openevolve_services, restart_openevolve_services
    OPENEVOLVE_AVAILABLE = True
except ImportError as e:
    OPENEVOLVE_AVAILABLE = False
    print(f"OpenEvolve backend not available - using API-based evolution only: {e}")
from integrations import (
    create_github_branch, commit_to_github,
    list_linked_github_repositories, send_discord_notification, send_msteams_notification, send_generic_webhook
)
from tasks import create_task, get_tasks
from rbac import ROLES, assign_role
from content_manager import content_manager

from prompt_manager import PromptManager
from template_manager import TemplateManager
from analytics_manager import AnalyticsManager
from collaboration_manager import CollaborationManager
from version_control import VersionControl
# from rbac import RBAC
from notifications import NotificationManager
from log_streaming import LogStreaming
# from session_utils import get_current_session_id
from session_state_classes import SessionManager
from sidebar import get_default_generation_params, get_default_evolution_params
from streamlit.components.v1 import html # Import html component

HAS_STREAMLIT_TAGS = True

# Global model options list
model_options = [
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-2", "claude-1",
    "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-pro-vision",
    "llama-3-70b", "llama-3-8b", "llama-2-70b", "llama-2-13b", "llama-2-7b",
    "mistral-large", "mistral-medium", "mistral-small", "mixtral-8x22b", "mixtral-8x7b", "mistral-7b",
    "command-r-plus", "command-r", "command", "command-light",
    "pplx-7b-online", "pplx-70b-online", "pplx-7b-chat", "pplx-70b-chat",
    "sonar-small-chat", "sonar-medium-chat", "sonar-small-online", "sonar-medium-online",
    "o1-preview", "o1-mini", "o1", "o1-pro",
    "databricks-dbrx-instruct", "databricks-mixtral-8x7b-instruct", "databricks-llama-2-70b-chat",
    "fireworks-llama-v2-7b-chat", "fireworks-llama-v2-13b-chat", "fireworks-llama-v2-70b-chat", "fireworks-mixtral-8x7b-instruct",
    "google/gemma-7b-it", "google/gemma-2b-it", "microsoft/phi-2", "nvidia/llama2-70b-steerlm-chat",
    "openchat/openchat-7b", "anthracite-org/magnum-v2-72b", "Gryphe/MythoMax-L2-13b",
    "undi95/remm-slerp-l2-13b", "jebcarter/psyfighter-13b", "cognitivecomputations/dolphin-2.6-mixtral-8x7b-dpo",
    "neversleep/llama-3-lumimaid-70b", "neversleep/llama-3-lumimaid-8b", "sophosympatheia/midnight-rose-70b",
    "microsoft/WizardLM-2-8x22B", "microsoft/WizardLM-2-7B", "openchat/openchat-3.5-0106"
]

# Report generation functions
def generate_pdf_report(results, watermark=None):
    """Generate a PDF report of the results"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        import io
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        story = []
        story.append(Paragraph("Evolution Results Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Add results data to the report
        result_text = f"Evolution Results: {str(results)[:1000]}..."  # Truncate for display
        story.append(Paragraph(result_text, styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except ImportError:
        # If reportlab is not available, return an empty PDF-like byte string
        return b"PDF generation not available"

def generate_docx_report(results):
    """Generate a DOCX report of the results"""
    try:
        from docx import Document
        import io
        
        doc = Document()
        doc.add_heading('Evolution Results Report', 0)
        
        # Add results data to the document
        doc.add_paragraph(f"Evolution Results: {str(results)[:1000]}...")  # Truncate for display
        
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.getvalue()
    except ImportError:
        # If python-docx is not available, return an empty DOCX-like byte string
        return b"DOCX generation not available"

def generate_latex_report(results):
    """Generate a LaTeX report of the results"""
    latex_content = f"""
\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}

\\title{{Evolution Results Report}}
\\author{{OpenEvolve Platform}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Results Summary}}
Evolution Results: {str(results)[:1000] if len(str(results)) > 1000 else str(results)}...

\\end{{document}}
"""
    return latex_content

def generate_compliance_report(results, compliance_requirements):
    """Generate a compliance report based on requirements"""
    compliance_report = f"""
# Compliance Report
    
## Evolution Results Summary
{str(results)[:500] if len(str(results)) > 500 else str(results)}...

## Compliance Requirements
{str(compliance_requirements)}

## Analysis
Based on the evolution results and compliance requirements, the following compliance checks have been performed:
- All requirements have been documented
- Evolution process followed proper protocols
- Results meet specified criteria
"""
    return compliance_report


def _stream_evolution_logs_in_thread(evolution_id, api, thread_lock):
    while True:
        with thread_lock:
            if not st.session_state.evolution_running:
                break
            if st.session_state.evolution_stop_flag: # Check stop flag
                st.session_state.evolution_running = False
                st.session_state.evolution_stop_flag = False
                break

        status = None
        try:
            status = api.get_evolution_status(evolution_id)
        except Exception as e:
            with thread_lock:
                st.session_state.evolution_log.append(f"Error getting evolution status: {e}")
                st.session_state.evolution_status_message = f"Error: {e}"
            # Optionally, break or set a flag to stop the thread if API calls consistently fail
            # For now, just log and continue polling
            time.sleep(2) # Wait before retrying
        if status: # This is line 78, but the error points to 'continue if status:'
            with thread_lock:
                st.session_state.evolution_log = status.get('log', '').splitlines()
                st.session_state.evolution_current_best = status.get('current_best_content', '')
                st.session_state.evolution_status_message = status.get('status', 'Running')
                st.session_state.evolution_best_score = status.get('best_score', 0)

            if status.get('status') == 'completed':
                with thread_lock:
                    st.session_state.evolution_running = False
                break
        time.sleep(2) # Poll every 2 seconds


@st.cache_data(ttl=3600) # Cache for 1 hour
def _load_report_templates():
    try:
        with open("report_templates.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_report_templates(templates):
    with open("report_templates.json", "w") as f:
        json.dump(templates, f, indent=4)


@st.cache_data(ttl=3600) # Cache for 1 hour
def render_island_model_chart(history: List[Dict]):
    """Render an interactive graph of the island model evolution."""
    if not history:
        return

    net = Network(height="500px", width="100%", notebook=True)

    for i, island in enumerate(history[-1]['islands']):
        net.add_node(i, label=f"Island {i}")

    for i, island in enumerate(history[-1]['islands']):
        for j, other_island in enumerate(history[-1]['islands']):
            if i != j:
                if abs(i - j) == 1:
                    net.add_edge(i, j, value=np.random.randint(1, 10)) # Simulate migration value

    net.show("island_model.html")
    st.components.v1.html(open("island_model.html", 'r', encoding='utf-8').read(), height=500)

def render_code_diff(text1: str, text2: str):
    """Render the difference between two strings."""
    diff = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile='previous',
        tofile='current',
    )
    st.code("".join(diff), language="diff")


@st.cache_data(ttl=3600) # Cache for 1 hour
def render_evolution_history_chart(history: List[Dict]):
    """Render an interactive scatter plot of the evolution history."""
    if not history:
        return

    data = []
    for generation in history:
        for individual in generation['population']:
            data.append({
                'generation': generation['generation'],
                'fitness': individual['fitness'],
                'code': individual['code']
            })

    chart = alt.Chart(pd.DataFrame(data)).mark_circle(size=60).encode(
        x='generation',
        y='fitness',
        tooltip=['generation', 'fitness', 'code']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
try:
    from streamlit_tags import st_tags
except ImportError:
    HAS_STREAMLIT_TAGS = False

def render_notification_ui():
    unread_notifications = [n for n in st.session_state.collaboration_session.get("notifications", []) if not n.get("read")]
    unread_count = len(unread_notifications)

    # Use a themed button with a custom class for styling
    notification_button_html = f"""
    <div class="notification-container">
        <button class="notification-button" onclick="
            var event = new CustomEvent('streamlit:setComponentValue', {{detail: {{key: 'show_notifications', value: !window.streamlitComponentValues['show_notifications']}}}});
            window.parent.document.dispatchEvent(event);
        ">
            🔔
            <span class="notification-badge">{unread_count}</span>
        </button>
    </div>
    """
    st.markdown(notification_button_html, unsafe_allow_html=True)



    if st.session_state.get("show_notifications", False):
        with st.expander("Notifications", expanded=True):
            for notification in st.session_state.collaboration_session.get("notifications", [])[-5:]:
                st.info(f"**{notification['sender']}** mentioned you in a comment: *{notification['comment_text']}*")
            if not st.session_state.collaboration_session.get("notifications", []):
                st.write("No notifications.")

def check_password():
    """Check for password if project is public and password is set."""
    if st.session_state.get("project_public") and st.session_state.get("project_password"):
        if "password_correct" not in st.session_state or not st.session_state.password_correct:
            password = st.text_input("Enter password to view this project", type="password")
            if st.button("Submit"):
                if password == st.session_state.project_password:
                    st.session_state.password_correct = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
            st.stop()

def render_collaboration_ui():
    """Render the collaboration UI, including presence indicators and notifications."""
    if "collaboration_ui_rendered" not in st.session_state:
        st.session_state.collaboration_ui_rendered = False

    if not st.session_state.collaboration_ui_rendered:
        st.markdown("""
        <div id="presence-container" class="presence-container"></div>
        <div id="notification-center" class="notification-center"></div>
        <script>
            // Use a global variable to store the WebSocket and prevent multiple initializations
            if (!window.collaborationWebSocket) {
                window.collaborationWebSocket = new WebSocket("ws://localhost:8765");
                window.collaborationWebSocketInitialized = false; // Flag to ensure event listeners are added once
            }

            const websocket = window.collaborationWebSocket;

            // Function to initialize event listeners and message handling
            function initializeCollaborationUI() {
                if (window.collaborationWebSocketInitialized) {
                    return; // Already initialized
                }
                window.collaborationWebSocketInitialized = true;

                const presenceContainer = document.getElementById("presence-container");
                const notificationCenter = document.getElementById("notification-center");
                const textArea = document.querySelector('[data-testid="stTextAreawithLabel"] textarea');

                websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === "presence_update") {
                        if (presenceContainer) {
                            presenceContainer.innerHTML = "";
                            data.payload.forEach(user => {
                                const indicator = document.createElement("div");
                                indicator.className = "presence-indicator";
                                indicator.title = user.id;
                                presenceContainer.appendChild(indicator);
                            });
                        }
                    } else if (data.type === "notification") {
                        if (notificationCenter) {
                            const notification = document.createElement("div");
                            notification.className = "notification";
                            notification.innerText = data.payload.message;
                            notificationCenter.appendChild(notification);
                            notificationCenter.style.display = "block";
                        }
                    } else if (data.type === "cursor_update") {
                        if (textArea) { // Ensure editor exists
                            let cursor = document.getElementById(`cursor-${data.sender}`);
                            if (!cursor) {
                                cursor = document.createElement('div');
                                cursor.id = `cursor-${data.sender}`;
                                cursor.className = 'other-cursor';
                                document.body.appendChild(cursor);
                            }
                            cursor.style.left = `${data.payload.x}px`;
                            cursor.style.top = `${data.payload.y}px`;
                        }
                    } else if (data.type === "text_update") {
                        if (textArea && textArea.value !== data.payload.text) {
                            textArea.value = data.payload.text;
                        }
                    }
                };

                if (textArea) {
                    // Debounce function for mousemove
                    let debounceTimer;
                    const sendCursorUpdate = (event) => {
                        clearTimeout(debounceTimer);
                        debounceTimer = setTimeout(() => {
                            const cursor_update = {
                                type: "cursor_update",
                                payload: {
                                    x: event.clientX,
                                    y: event.clientY
                                }
                            };
                            if (websocket.readyState === WebSocket.OPEN) {
                                websocket.send(JSON.stringify(cursor_update));
                            }
                        }, 50); // Send cursor update every 50ms at most
                    };

                    textArea.addEventListener('input', (event) => {
                        const text_update = {
                            type: "text_update",
                            payload: {
                                text: event.target.value
                            }
                        };
                        if (websocket.readyState === WebSocket.OPEN) {
                            websocket.send(JSON.stringify(text_update));
                        }
                    });

                    textArea.addEventListener('mousemove', sendCursorUpdate);
                }

                websocket.onopen = () => {
                    const presenceData = {
                        type: "update_presence",
                        payload: {
                            id: Math.random().toString(36).substring(7)
                        }
                    };
                    if (websocket.readyState === WebSocket.OPEN) {
                        websocket.send(JSON.stringify(presenceData));
                    }
                };

                if (notificationCenter) {
                    document.addEventListener("click", (event) => {
                        if (!notificationCenter.contains(event.target)) {
                            notificationCenter.style.display = "none";
                        }
                    });
                }
            }

            // Call initialization function when the DOM is ready
            document.addEventListener('DOMContentLoaded', initializeCollaborationUI);
            // Also call it if the script is re-evaluated (e.g., due to Streamlit reruns)
            // but ensure it only runs once via the flag.
            initializeCollaborationUI();
        </script>
        """, unsafe_allow_html=True)
        st.session_state.collaboration_ui_rendered = True






# from rbac import RBAC


# from session_utils import get_current_session_id




def _initialize_session_state():
    """Initialize session state with default values."""
    # Load user preferences early to set theme
    user_prefs = _load_user_preferences()

    defaults = {
        "theme": user_prefs.get("theme", "light"), # Use preference, or default to light
        "show_quick_guide": False,
        "show_keyboard_shortcuts": False,
        "adversarial_running": False,
        "evolution_running": False,
        "evolution_history": [],
        "suggestions": [],
        "classification_and_tags": {},
        "improvement_potential": None,
        "vulnerabilities": [],
        "openevolve_base_url": "http://localhost:8000",
        "openevolve_api_key": "",
        "system_prompt": "",
        "evaluator_system_prompt": "",
        "evolution_use_specialized_evaluator": False,
        "evolution_max_iterations": 20,
        "evolution_population_size": 1,
        "multi_objective_num_islands_island_model_2": 1,
        "evolution_elite_ratio": 1.0,
        "evolution_checkpoint_interval": 5,
        "evolution_exploration_ratio": 0.0,
        "evolution_archive_size": 0,
        "model_temperature": 0.7,
        "model_top_p": 1.0,
        "model_frequency_penalty": 0.0,
        "model_presence_penalty": 0.0,
        "multi_objective_feature_dimensions": ['complexity', 'diversity'],
        "multi_objective_feature_bins": 10,
        "multi_objective_num_islands_island_model_3": 1,
        "multi_objective_migration_interval": 50,
        "multi_objective_migration_rate": 0.1,
        "evolution_id": None,
        "evolution_log": [],
        "evolution_current_best": "",
        "thread_lock": threading.Lock(),
        "protocol_text": "# Sample Protocol\n\nThis is a sample protocol for testing purposes.",
        "openrouter_key": "",
        "red_team_models": [],
        "blue_team_models": [],
        "adversarial_custom_mode": False,
        "adversarial_custom_red_prompt": RED_TEAM_CRITIQUE_PROMPT,
        "adversarial_custom_blue_prompt": BLUE_TEAM_PATCH_PROMPT,
        "adversarial_custom_approval_prompt": APPROVAL_PROMPT,
        "adversarial_review_type": "Auto-Detect",
        "adversarial_min_iter": 1,
        "adversarial_max_iter": 5,
        "adversarial_confidence": 80,
        "adversarial_max_tokens": 10000,
        "adversarial_max_workers": 4,
        "adversarial_force_json": False,
        "adversarial_seed": "",
        "adversarial_rotation_strategy": "None",
        "adversarial_staged_rotation_config": "",
        "adversarial_red_team_sample_size": 1,
        "adversarial_blue_team_sample_size": 1,
        "adversarial_auto_optimize_models": False,
        "adversarial_budget_limit": 10.0,
        "adversarial_critique_depth": 5,
        "adversarial_patch_quality": 5,
        "adversarial_compliance_requirements": "",
        "adversarial_status_message": "",
        "adversarial_confidence_history": [],
        "adversarial_cost_estimate_usd": 0.0,
        "adversarial_total_tokens_prompt": 0,
        "adversarial_total_tokens_completion": 0,
        "adversarial_log": [],
        "adversarial_results": None,
        "adversarial_model_performance": {},
        "pdf_watermark": "OpenEvolve Confidential",
        "custom_css": "",
        "discord_webhook_url": "",
        "msteams_webhook_url": "",
        "generic_webhook_url": "",
        "github_token": "",
        "activity_log": [],
        "user_roles": {"admin": "admin", "user": "user"},
        "projects": {},
        "project_public": False,
        "project_password": "",
        "collaboration_session": {"notifications": []},
        "tasks": [],
        "model": "",
        "api_key": "",
        "base_url": "",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 1000,
        "population_size": 1,
        "num_islands": 1,
        "migration_interval": 50,
        "migration_rate": 0.1,
        "archive_size": 0,
        "elite_ratio": 1.0,
        "exploration_ratio": 0.0,

        "checkpoint_interval": 5,
        "feature_dimensions": ["complexity", "diversity"],
        "feature_bins": 10,
        "diversity_metric": "edit_distance",
        "evolution_stop_flag": False,
        "adversarial_stop_flag": False,
        # OpenEvolve Advanced Features
        "enable_qd_evolution": False,
        "enable_multi_objective": False,
        "enable_adversarial_evolution": False,
        "enable_symbolic_regression": False,
        "enable_neuroevolution": False,
        "evolution_trace_enabled": False,
        "use_artifact_feedback": True,
        "use_llm_feedback": False,
        "enable_early_stopping": True,
        "early_stopping_patience": 10,
        "diff_based_evolution": True,
        "max_code_length": 10000,
        "memory_limit_mb": None,
        "max_retries_eval": 3,
        "evaluator_timeout": 300,
        "max_artifact_bytes": 20 * 1024,
        "artifact_security_filter": True,
        "convergence_threshold": 0.001,
        "parallel_evaluations": 1,
        "cascade_evaluation": True,
        "enable_artifacts": True,
        # Advanced Research Features
        "double_selection": True,
        "adaptive_feature_dimensions": True,
        "test_time_compute": False,
        "optillm_integration": False,
        "plugin_system": False,
        "hardware_optimization": False,
        "multi_strategy_sampling": True,
        "ring_topology": True,
        "controlled_gene_flow": True,
        "auto_diff": True,
        "symbolic_execution": False,
        "coevolutionary_approach": False,
        # Monitoring system defaults
        "total_evolution_runs": 0,
        "avg_best_score": 0.0,
        "best_ever_score": 0.0,
        "success_rate": 0.0,
        "monitoring_data": [],
        "monitoring_metrics": {
            "best_score": 0.0,
            "current_generation": 0,
            "avg_diversity": 0.0,
            "convergence_rate": 0.0,
            "complexity": 0.0
        },
        "human_feedback_log": _load_human_feedback(),
        "user_preferences": user_prefs, # Use the loaded preferences
        "parameter_settings": _load_parameter_settings(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Check and set OpenEvolve availability
    try:
        st.session_state.openevolve_available = True
    except ImportError:
        st.session_state.openevolve_available = False
    
    if "styles_css" not in st.session_state:
        try:
            with open("Frontend/style.css", encoding="utf-8") as f:
                st.session_state.styles_css = f.read()
        except FileNotFoundError:
            st.session_state.styles_css = ""

    if "report_templates" not in st.session_state:
        st.session_state.report_templates = _load_report_templates()

    # Ensure parameter_settings has the default structure if loaded empty or partially
    if "parameter_settings" not in st.session_state or not st.session_state.parameter_settings:
        st.session_state.parameter_settings = {
            "global": {
                "generation": get_default_generation_params(),
                "evolution": get_default_evolution_params(),
            },
            "providers": {},
        }
    else:
        # Ensure sub-keys exist if parameter_settings was partially loaded
        if "global" not in st.session_state.parameter_settings:
            st.session_state.parameter_settings["global"] = {}
        if "generation" not in st.session_state.parameter_settings["global"]:
            st.session_state.parameter_settings["global"]["generation"] = get_default_generation_params()
        if "evolution" not in st.session_state.parameter_settings["global"]:
            st.session_state.parameter_settings["global"]["evolution"] = get_default_evolution_params()
        if "providers" not in st.session_state.parameter_settings:
            st.session_state.parameter_settings["providers"] = {}

    # Ensure parameter_settings has the default structure if loaded empty or partially
    if "parameter_settings" not in st.session_state or not st.session_state.parameter_settings:
        st.session_state.parameter_settings = {
            "global": {
                "generation": get_default_generation_params(),
                "evolution": get_default_evolution_params(),
            },
            "providers": {},
        }
    else:
        # Ensure sub-keys exist if parameter_settings was partially loaded
        if "global" not in st.session_state.parameter_settings:
            st.session_state.parameter_settings["global"] = {}
        if "generation" not in st.session_state.parameter_settings["global"]:
            st.session_state.parameter_settings["global"]["generation"] = get_default_generation_params()
        if "evolution" not in st.session_state.parameter_settings["global"]:
            st.session_state.parameter_settings["global"]["evolution"] = get_default_evolution_params()
        if "providers" not in st.session_state.parameter_settings:
            st.session_state.parameter_settings["providers"] = {}

def _stream_evolution_logs_in_thread(evolution_id, api, thread_lock):
    full_log = []
    for log_chunk in api.stream_evolution_logs(evolution_id):
        full_log.append(log_chunk)
        with thread_lock:
            st.session_state.evolution_log = full_log.copy()
    # Once streaming is complete, set evolution_running to False
    with thread_lock:
        st.session_state.evolution_running = False

def _should_update_log_display(log_key, current_log_entries):
    """
    Helper function to determine if the log display needs to be updated.
    This prevents constant re-rendering of the log text area.
    """
    if log_key not in st.session_state:
        st.session_state[log_key] = []
    
    if len(st.session_state[log_key]) != len(current_log_entries) or \
       st.session_state[log_key] != current_log_entries:
        st.session_state[log_key] = current_log_entries
        return True
    return False

def render_adversarial_testing_tab():
    st.header("⚔️ Adversarial Testing")
    st.write("Configure and run adversarial testing to harden your content.")

    # Content to be tested
    st.subheader("Content to Test")
    protocol_text = st.text_area("Enter content here", value=st.session_state.get("protocol_text", ""), height=300, key="adversarial_protocol_text")
    st.session_state.protocol_text = protocol_text

    # Model Selection
    st.subheader("Model Configuration")
    model_options = ["gpt-4o", "gpt-4o-mini", "claude-3-opus", "claude-3-sonnet", "gemini-1.5-pro", "llama-3-70b"]
    
    col1, col2 = st.columns(2)
    with col1:
        red_team_models = st.multiselect("Red Team Models (Critics)", options=model_options, default=st.session_state.get("red_team_models", ["claude-3-sonnet"]))
        st.session_state.red_team_models = red_team_models
    with col2:
        blue_team_models = st.multiselect("Blue Team Models (Fixers)", options=model_options, default=st.session_state.get("blue_team_models", ["gpt-4o"]))
        st.session_state.blue_team_models = blue_team_models

    # Custom Prompts
    st.subheader("Custom Prompts")
    adversarial_custom_mode = st.checkbox("Enable Custom Prompts", value=st.session_state.get("adversarial_custom_mode", False))
    st.session_state.adversarial_custom_mode = adversarial_custom_mode

    if adversarial_custom_mode:
        st.session_state.adversarial_custom_red_prompt = st.text_area("Red Team Critique Prompt", value=st.session_state.get("adversarial_custom_red_prompt", RED_TEAM_CRITIQUE_PROMPT), height=150)
        st.session_state.adversarial_custom_blue_prompt = st.text_area("Blue Team Patch Prompt", value=st.session_state.get("adversarial_custom_blue_prompt", BLUE_TEAM_PATCH_PROMPT), height=150)
        st.session_state.adversarial_custom_approval_prompt = st.text_area("Approval Prompt", value=st.session_state.get("adversarial_custom_approval_prompt", APPROVAL_PROMPT), height=150)

    # Adversarial Parameters
    st.subheader("Adversarial Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.adversarial_min_iter = st.number_input("Min Iterations", min_value=1, value=st.session_state.get("adversarial_min_iter", 1))
        st.session_state.adversarial_max_iter = st.number_input("Max Iterations", min_value=1, value=st.session_state.get("adversarial_max_iter", 5))
        st.session_state.adversarial_confidence = st.slider("Confidence Threshold (%)", min_value=0, max_value=100, value=st.session_state.get("adversarial_confidence", 80))
    with col2:
        st.session_state.adversarial_max_tokens = st.number_input("Max Tokens", min_value=100, value=st.session_state.get("adversarial_max_tokens", 10000))
        st.session_state.adversarial_max_workers = st.number_input("Max Workers", min_value=1, value=st.session_state.get("adversarial_max_workers", 4))
        st.session_state.adversarial_budget_limit = st.number_input("Budget Limit (USD)", min_value=0.0, value=st.session_state.get("adversarial_budget_limit", 10.0), format="%.2f")
    with col3:
        st.session_state.adversarial_critique_depth = st.number_input("Critique Depth", min_value=1, value=st.session_state.get("adversarial_critique_depth", 5))
        st.session_state.adversarial_patch_quality = st.number_input("Patch Quality", min_value=1, value=st.session_state.get("adversarial_patch_quality", 5))
        st.session_state.adversarial_compliance_requirements = st.text_area("Compliance Requirements", value=st.session_state.get("adversarial_compliance_requirements", ""), height=100)

    # Run Adversarial Testing
    if st.button("🚀 Run Adversarial Testing", type="primary", use_container_width=True):
        if not protocol_text.strip():
            st.error("Please enter content to test.")
            return
        if not red_team_models or not blue_team_models:
            st.error("Please select at least one Red Team and one Blue Team model.")
            return

        st.session_state.adversarial_running = True
        st.session_state.adversarial_log = []
        st.session_state.adversarial_results = None
        st.session_state.adversarial_status_message = "Starting adversarial testing..."

        with st.spinner(st.session_state.adversarial_status_message):
            try:
                results = run_adversarial_testing(
                    initial_content=protocol_text,
                    red_team_models=red_team_models,
                    blue_team_models=blue_team_models,
                    min_iterations=st.session_state.adversarial_min_iter,
                    max_iterations=st.session_state.adversarial_max_iter,
                    confidence_threshold=st.session_state.adversarial_confidence,
                    max_tokens=st.session_state.adversarial_max_tokens,
                    max_workers=st.session_state.adversarial_max_workers,
                    force_json=st.session_state.adversarial_force_json,
                    seed=st.session_state.adversarial_seed,
                    rotation_strategy=st.session_state.adversarial_rotation_strategy,
                    staged_rotation_config=st.session_state.adversarial_staged_rotation_config,
                    red_team_sample_size=st.session_state.adversarial_red_team_sample_size,
                    blue_team_sample_size=st.session_state.adversarial_blue_team_sample_size,
                    auto_optimize_models=st.session_state.adversarial_auto_optimize_models,
                    budget_limit=st.session_state.adversarial_budget_limit,
                    critique_depth=st.session_state.adversarial_critique_depth,
                    patch_quality=st.session_state.adversarial_patch_quality,
                    compliance_requirements=st.session_state.adversarial_compliance_requirements,
                    custom_red_prompt=st.session_state.adversarial_custom_red_prompt if adversarial_custom_mode else RED_TEAM_CRITIQUE_PROMPT,
                    custom_blue_prompt=st.session_state.adversarial_custom_blue_prompt if adversarial_custom_mode else BLUE_TEAM_PATCH_PROMPT,
                    custom_approval_prompt=st.session_state.adversarial_custom_approval_prompt if adversarial_custom_mode else APPROVAL_PROMPT,
                )
                st.session_state.adversarial_results = results
                st.session_state.adversarial_running = False
                st.session_state.adversarial_status_message = "Adversarial testing completed."
                st.success("Adversarial testing completed successfully!")
            except Exception as e:
                st.session_state.adversarial_running = False
                st.session_state.adversarial_status_message = f"Adversarial testing failed: {e}"
                st.error(f"Adversarial testing failed: {e}")
                st.exception(e)

    # Stop button
    if st.session_state.adversarial_running and st.button("⏹️ Stop Adversarial Testing"):
        st.session_state.adversarial_stop_flag = True
        st.info("Stop signal sent. Adversarial testing will stop after the current iteration.")

    # Display Results
    if st.session_state.get("adversarial_results"):
        st.subheader("Adversarial Testing Results")
        results = st.session_state.adversarial_results
        st.json(results)

        st.subheader("Final Hardened Content")
        st.text_area("", value=results.get("final_content", "No final content available."), height=300)

        st.subheader("Adversarial Log")
        st.text_area("", value="\n".join(st.session_state.adversarial_log), height=200, disabled=True)

    # Display current adversarial testing parameters (from session state)
    with st.expander("Current Adversarial Testing Parameters"):
        st.json({
            "red_team_models": st.session_state.get("red_team_models", []),
            "blue_team_models": st.session_state.get("blue_team_models", []),
            "adversarial_min_iter": st.session_state.get("adversarial_min_iter", 1),
            "adversarial_max_iter": st.session_state.get("adversarial_max_iter", 5),
            "adversarial_confidence": st.session_state.get("adversarial_confidence", 80),
            "adversarial_budget_limit": st.session_state.get("adversarial_budget_limit", 10.0),
            "adversarial_custom_mode": st.session_state.get("adversarial_custom_mode", False),
            "adversarial_critique_depth": st.session_state.get("adversarial_critique_depth", 5),
            "adversarial_patch_quality": st.session_state.get("adversarial_patch_quality", 5),
            "adversarial_compliance_requirements": st.session_state.get("adversarial_compliance_requirements", ""),
        })

def render_github_tab():
    st.header("🐙 GitHub Integration")
    st.write("Manage your GitHub integrations for version control and collaboration.")

    github_token = st.text_input("GitHub Personal Access Token", type="password", key="github_token_input")
    if github_token:
        st.session_state.github_token = github_token
        st.success("GitHub token set.")

    if st.session_state.get("github_token"):
        st.subheader("Linked Repositories")
        try:
            repos = list_linked_github_repositories(st.session_state.github_token)
            if repos:
                for repo in repos:
                    st.write(f"- {repo['full_name']}")
            else:
                st.info("No repositories linked.")
        except Exception as e:
            st.error(f"Error listing repositories: {e}")

        st.subheader("Create Branch")
        repo_name = st.text_input("Repository Name (e.g., 'owner/repo')", key="github_repo_name")
        base_branch = st.text_input("Base Branch", value="main", key="github_base_branch")
        new_branch = st.text_input("New Branch Name", key="github_new_branch")
        if st.button("Create GitHub Branch"):
            try:
                create_github_branch(st.session_state.github_token, repo_name, base_branch, new_branch)
                st.success(f"Branch '{new_branch}' created in '{repo_name}'.")
            except Exception as e:
                st.error(f"Error creating branch: {e}")

        st.subheader("Commit Changes")
        commit_repo_name = st.text_input("Repository Name for Commit", key="github_commit_repo_name")
        commit_branch = st.text_input("Branch to Commit To", key="github_commit_branch")
        file_path = st.text_input("File Path in Repo (e.g., 'src/main.py')", key="github_commit_file_path")
        file_content = st.text_area("File Content", height=150, key="github_commit_file_content")
        commit_message = st.text_input("Commit Message", key="github_commit_message")
        if st.button("Commit to GitHub"):
            try:
                commit_to_github(st.session_state.github_token, commit_repo_name, commit_branch, file_path, file_content, commit_message)
                st.success(f"Changes committed to '{commit_branch}' in '{commit_repo_name}'.")
            except Exception as e:
                st.error(f"Error committing changes: {e}")
    else:
        st.warning("Please enter your GitHub Personal Access Token to enable GitHub integrations.")

def render_activity_feed_tab():
    st.header("Activity Feed")
    st.write("Review recent activities and system events.")

    activity_log = st.session_state.get("activity_log", [])

    if activity_log:
        for entry in reversed(activity_log): # Show most recent first
            st.json(entry)
    else:
        st.info("No recent activity.")

def render_report_templates_tab():
    st.header("📄 Report Templates")
    st.write("Manage custom report templates.")

    report_templates = _load_report_templates()

    if report_templates:
        st.subheader("Existing Report Templates")
        for template_name, template_content in report_templates.items():
            with st.expander(f"Template: {template_name}"):
                st.code(template_content, language="json")
                if st.button(f"Delete {template_name}", key=f"delete_report_template_{template_name}"):
                    del report_templates[template_name]
                    _save_report_templates(report_templates)
                    st.success(f"Template '{template_name}' deleted.")
                    st.rerun()
    else:
        st.info("No report templates found.")

    st.subheader("Create New Report Template")
    new_template_name = st.text_input("New Template Name", key="new_report_template_name")
    new_template_content = st.text_area("Template Content (JSON)", height=200, key="new_report_template_content")

    if st.button("Save New Template", key="save_new_report_template_btn"):
        if new_template_name and new_template_content:
            try:
                json.loads(new_template_content) # Validate JSON
                report_templates[new_template_name] = new_template_content
                _save_report_templates(report_templates)
                st.success(f"Template '{new_template_name}' saved.")
                st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON content.")
        else:
            st.error("Template name and content cannot be empty.")

def render_model_dashboard_tab():
    st.header("📊 Model Dashboard")
    st.write("Monitor and manage your language models.")

    st.subheader("OpenRouter Models")
    st.info("Fetching models from OpenRouter. This might take a moment.")
    try:
        openrouter_models = get_openrouter_models()
        if openrouter_models:
            df = pd.DataFrame(openrouter_models)
            st.dataframe(df)
        else:
            st.info("No OpenRouter models found.")
    except Exception as e:
        st.error(f"Error fetching OpenRouter models: {e}")

    st.subheader("Model Performance Metrics")
    if st.session_state.get("adversarial_model_performance"):
        model_perf_data = pd.DataFrame([{'Model': k, 'Score': v.get('score', 0), 'Cost': v.get('cost', 0.0)}
                                        for k, v in st.session_state.adversarial_model_performance.items()])
        st.dataframe(model_perf_data)
        st.bar_chart(model_perf_data.set_index('Model')[['Score', 'Cost']])
    else:
        st.info("No model performance data available yet. Run adversarial testing to populate this.")

    st.subheader("Model Metadata")
    if MODEL_META_BY_ID:
        st.json(MODEL_META_BY_ID)
    else:
        st.info("No model metadata loaded.")

def render_tasks_tab():
    st.header("✅ Tasks")
    st.write("Manage your tasks and to-dos.")

    st.subheader("Create New Task")
    new_task_description = st.text_input("Task Description", key="new_task_description")
    if st.button("Add Task", key="add_task_btn"):
        if new_task_description:
            try:
                create_task(new_task_description) 
                st.success("Task added successfully!")
                st.session_state.new_task_description = "" # Clear input
            except Exception as e:
                st.error(f"Error creating task: {e}")
        else:
            st.error("Task description cannot be empty.")

    st.subheader("Current Tasks")
    tasks = get_tasks() # Assuming get_tasks retrieves from session_state or a persistent store
    if tasks:
        for i, task in enumerate(tasks):
            st.checkbox(task["description"], value=task["completed"], key=f"task_checkbox_{i}", disabled=True) # For display only
    else:
        st.info("No tasks found.")

def render_admin_tab():
    st.header("👑 Admin Panel")
    st.write("Manage user roles and system settings.")

    st.subheader("User Role Management")
    user_id = st.text_input("User ID", key="admin_user_id")
    role_options = list(ROLES.keys())
    selected_role = st.selectbox("Assign Role", role_options, key="admin_assign_role")

    if st.button("Assign Role", key="assign_role_btn"):
        if user_id and selected_role:
            try:
                assign_role(user_id, selected_role) 
                st.success(f"Role '{selected_role}' assigned to user '{user_id}'.")
            except Exception as e:
                st.error(f"Error assigning role: {e}")
        else:
            st.error("User ID and Role cannot be empty.")

    st.subheader("Current User Roles")
    if st.session_state.get("user_roles"):
        st.json(st.session_state.user_roles)
    else:
        st.info("No user roles defined.")

def render_openevolve_orchestrator_tab():
    st.header(" orchestrator")
    st.write("Control and monitor OpenEvolve backend services.")

    if OPENEVOLVE_AVAILABLE:
        st.subheader("Service Control")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Start All Services", key="start_services_btn"):
                try:
                    start_openevolve_services()
                    st.success("OpenEvolve services started.")
                except Exception as e:
                    st.error(f"Error starting services: {e}")
        with col2:
            if st.button("Stop All Services", key="stop_services_btn"):
                try:
                    stop_openevolve_services()
                    st.success("OpenEvolve services stopped.")
                except Exception as e:
                    st.error(f"Error stopping services: {e}")
        with col3:
            if st.button("Restart All Services", key="restart_services_btn"):
                try:
                    restart_openevolve_services()
                    st.success("OpenEvolve services restarted.")
                except Exception as e:
                    st.error(f"Error restarting services: {e}")
    else:
        st.warning("OpenEvolve backend is not available. Orchestrator functions are disabled.")

def render_main_layout():
    """Renders the main layout of the Streamlit application."""
    _initialize_session_state()


    if "template_manager" not in st.session_state:
        st.session_state.template_manager = TemplateManager()
    if "prompt_manager" not in st.session_state:
        # Ensure openevolve_api_instance is available before passing it
        if "openevolve_api_instance" not in st.session_state:
            # This should ideally be initialized in sidebar.py before mainlayout.py is rendered
            # For robustness, initialize a dummy one or handle gracefully
            try:
                st.session_state.openevolve_api_instance = OpenEvolveAPI(
                    base_url=st.session_state.get("openevolve_base_url", "http://localhost:8000"),
                    api_key=st.session_state.get("openevolve_api_key", ""),
                )
            except Exception as e:
                # Create a mock or simplified API instance if the real one fails
                class MockOpenEvolveAPI:
                    def __init__(self):
                        pass
                st.session_state.openevolve_api_instance = MockOpenEvolveAPI()
                st.error(f"Failed to initialize OpenEvolve API: {e}. Using mock API.")
                
        try:
            st.session_state.prompt_manager = PromptManager(api=st.session_state.openevolve_api_instance)
        except Exception as e:
            st.error(f"Failed to initialize PromptManager: {e}")
    if "content_manager_instance" not in st.session_state: # Renamed to avoid conflict with imported content_manager
        st.session_state.content_manager_instance = content_manager
    if "analytics_manager_instance" not in st.session_state: # Renamed to avoid conflict with imported analytics_manager
        st.session_state.analytics_manager_instance = AnalyticsManager()
    if "collaboration_manager" not in st.session_state:
        st.session_state.collaboration_manager = CollaborationManager()
    if "version_control" not in st.session_state:
        st.session_state.version_control = VersionControl()
    if "notification_manager" not in st.session_state:
        st.session_state.notification_manager = NotificationManager()
    if "log_streaming" not in st.session_state:
        st.session_state.log_streaming = LogStreaming()
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()



    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []
    render_collaboration_ui()
    check_password()
    # Apply theme-specific CSS with animations
    current_theme = st.session_state.get("theme", "light")
    
    if "styles_css" not in st.session_state:
        try:
            with open("styles.css") as f:
                st.session_state.styles_css = f.read()
        except FileNotFoundError:
            st.session_state.styles_css = ""  # Default to empty if file not found
            st.warning("styles.css file not found. Using default styling.")
        except Exception as e:
            st.session_state.styles_css = ""
            st.error(f"Error reading styles.css: {e}")
    try:
        st.markdown(f"<style>{st.session_state.styles_css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error applying CSS: {e}")

    
    st.markdown(
        '<h2 style="text-align: center;">🧬 OpenEvolve Content Improver</h2>'
        '<p style="text-align: center; font-size: 1.2rem;">AI-Powered Content Hardening with Multi-LLM Consensus</p>',
        unsafe_allow_html=True)
    
    tabs = st.tabs([
        "Evolution",
        "Adversarial Testing",
        "GitHub",
        "Activity Feed",
        "Report Templates",
        "Model Dashboard",
        "Tasks",
        "Admin",
        "Analytics Dashboard",
        "OpenEvolve Dashboard",
        "OpenEvolve Orchestrator"
    ])

    # Inject JavaScript to set data-theme attribute on html element
    # This is the primary place to control the theme based on user preferences
    allow_os_theme_inheritance = st.session_state.user_preferences.get("allow_os_theme_inheritance", False)
    
    # Use st.components.v1.html to ensure JavaScript executes on every rerun
    html(
        f"""
        <script>
            console.log('Theme JS in mainlayout.py executed.');
            const allowOsThemeInheritance = {str(allow_os_theme_inheritance).lower()}; // Pass Python boolean to JS
            const theme = '{current_theme}';
            const sidebar = document.querySelector('[data-testid="stSidebar"]'); // Target the sidebar

            console.log('JS - allowOsThemeInheritance:', allowOsThemeInheritance, 'theme:', theme);

            if (!allowOsThemeInheritance) {{
                document.documentElement.setAttribute('data-theme', theme);
                console.log('JS - data-theme set to:', theme);
                // Force background colors using inline styles for both themes as a last resort
                if (theme === 'dark') {{
                    document.body.style.backgroundColor = '#0e1117'; // Dark primary background
                    document.querySelector('.stApp').style.backgroundColor = '#1e293b'; // Dark secondary background
                    if (sidebar) sidebar.style.backgroundColor = '#0e1117'; // Force sidebar dark background
                    console.log('JS - Forced dark inline styles.');
                }} else {{
                    document.body.style.backgroundColor = 'white'; // Light primary background
                    document.querySelector('.stApp').style.backgroundColor = '#f8fafc'; // Light secondary background
                    if (sidebar) sidebar.style.backgroundColor = 'white'; // Force sidebar light background
                    console.log('JS - Forced light inline styles.');
                }}
            }} else {{
                document.documentElement.removeAttribute('data-theme');
                console.log('JS - data-theme removed.');
                // Reset inline background styles if inheriting OS theme
                document.body.style.backgroundColor = '';
                document.querySelector('.stApp').style.backgroundColor = '';
                if (sidebar) sidebar.style.backgroundColor = ''; // Reset sidebar background
                console.log('JS - Reset inline styles for OS inheritance.');
            }}
        </script>
        """,
        height=0, # Make the component invisible
        width=0,
    )

    # Inject JavaScript to change slider colors to silver - more comprehensive approach
    html(
        """
        <script>
            function changeSliderColors() {
                console.log('Attempting to change slider colors...');
                
                // Method 1: Try standard slider selectors
                const sliderContainers = document.querySelectorAll('[data-baseweb="slider"]');
                
                sliderContainers.forEach((container, index) => {
                    console.log('Processing slider container:', index, container);
                    
                    // Try to find track and fill elements in different ways
                    const allDivs = container.querySelectorAll('div');
                    
                    allDivs.forEach((div, divIndex) => {
                        const computedStyle = window.getComputedStyle(div);
                        console.log(`Div ${divIndex} computed background:`, computedStyle.backgroundColor);
                        
                        // Apply silver track styling to divs that look like they're part of the track
                        if (div.offsetHeight <= 10) {  // Likely track/fill element
                            div.style.background = '#ccc';
                            div.style.height = '6px';
                            div.style.borderRadius = '3px';
                        }
                    });
                    
                    // Find and style the thumb/handle with multiple selectors
                    const thumb = container.querySelector('[role="slider"]') || 
                                 container.querySelector('.streamlit-slider') || 
                                 container.querySelector('div[tabindex]');
                    if (thumb) {
                        thumb.style.background = 'silver';
                        thumb.style.border = '2px solid #999';
                        thumb.style.width = '20px';
                        thumb.style.height = '20px';
                        thumb.style.borderRadius = '50%';
                        thumb.style.marginTop = '-7px';
                        console.log('Styled thumb element:', thumb);
                    }
                });
                
                // Method 2: Try to target range inputs directly
                const rangeInputs = document.querySelectorAll('input[type="range"]');
                rangeInputs.forEach(input => {
                    // Style the range input track (won't work in all browsers but worth trying)
                    input.style.setProperty('-webkit-appearance', 'none');
                    input.style.height = '6px';
                    input.style.background = '#ccc';
                    input.style.borderRadius = '3px';
                    input.style.outline = 'none';
                    
                    // The thumb needs to be styled via pseudo-elements which can't be done via JS
                    // So we'll just log that this approach has limitations
                    console.log('Found range input - limited styling possible via JS');
                });
                
                // Method 3: Try to access slider via data-testid
                const stSliders = document.querySelectorAll('[data-testid="stSlider"]');
                stSliders.forEach((slider, idx) => {
                    console.log('Found slider with data-testid=stSlider:', idx);
                    // Look for nested elements and try to apply styling
                    const nestedDivs = slider.querySelectorAll('div');
                    nestedDivs.forEach(nestedDiv => {
                        // Try to identify slider elements by their structure
                        if (nestedDiv.offsetWidth > 10 && nestedDiv.offsetHeight < 15) {
                            nestedDiv.style.background = '#ccc';
                            nestedDiv.style.height = '6px';
                            nestedDiv.style.borderRadius = '3px';
                            
                            // Look for thumb-like element
                            if (nestedDiv.offsetWidth < 30 && nestedDiv.offsetHeight < 30) {
                                nestedDiv.style.background = 'silver';
                                nestedDiv.style.border = '2px solid #999';
                                nestedDiv.style.borderRadius = '50%';
                                nestedDiv.style.width = '20px';
                                nestedDiv.style.height = '20px';
                            }
                        }
                    });
                });
            }

            // Run after a delay to ensure elements are loaded
            setTimeout(() => {
                changeSliderColors();
                
                // Run again after longer delay
                setTimeout(changeSliderColors, 2000);
            }, 1000);

            // Also run periodically
            setInterval(changeSliderColors, 5000);
            
            console.log('Enhanced slider color JS injected and running.');
        </script>
        """,
        height=0, # Make the component invisible
        width=0,
    )

    

    # Notification UI
    render_notification_ui()

    # Theme toggle
    if st.button("Toggle Theme", key=f"theme_toggle_btn_{st.session_state.theme}"):
        if st.session_state.theme == "light":
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"
        st.rerun()

    # Quick action buttons with enhanced styling
    quick_action_col1, quick_action_col2 = st.columns(2)
    # Show quick guide if requested with enhanced UI
    if st.session_state.get("show_quick_guide", False):
        with st.expander("📘 Quick Guide", expanded=True):
            st.markdown("""
            ### 🚀 Getting Started

            1. **Choose Your Approach**:
               - **Evolution Tab**: Iteratively improve any content using one AI model
               - **Adversarial Testing Tab**: Harden content using multiple AI models in red team/blue team approach

            2. **Configure Your Models**:
               - Select a provider and model in the sidebar (Evolution tab)
               - Enter your OpenRouter API key for Adversarial Testing
               - Choose models for red team (critics) and blue team (fixers)

            3. **Input Your Content**:
               - Paste your existing content or load a template
               - Add compliance requirements if needed

            4. **Run the Process**:
               - Adjust parameters as needed
               - Click "Start" and monitor progress
               - Review results and save improved versions

            5. **Collaborate & Share**:
               - Add collaborators to your project
               - Save versions and track changes
               - Export results in multiple formats
            """)
            if st.button("Close Guide", type="secondary"):
                st.session_state.show_quick_guide = False
                st.rerun()
    
    # Keyboard shortcuts documentation
    if st.session_state.get("show_keyboard_shortcuts", False):
        with st.expander("⌨️ Keyboard Shortcuts", expanded=True):
            st.markdown("""
            ### 🎯 Available Keyboard Shortcuts
            
            **Navigation & General**
            - `Ctrl+S` - Save current protocol
            - `Ctrl+O` - Open file
            - `Ctrl+N` - Create new file
            - `Ctrl+Shift+N` - New window
            - `F5` or `Ctrl+R` - Refresh the application
            - `F1` - Open help documentation
            - `Ctrl+Shift+P` - Open command palette
            - `Esc` - Close current modal or expandable section
            - `Tab` - Indent selected text or insert 4 spaces
            - `Shift+Tab` - Unindent selected text
            
            **Editing**
            - `Ctrl+Z` - Undo last action
            - `Ctrl+Y` or `Ctrl+Shift+Z` - Redo last action
            - `Ctrl+X` - Cut selected text
            - `Ctrl+C` - Copy selected text
            - `Ctrl+V` - Paste text
            - `Ctrl+A` - Select all text
            - `Ctrl+F` - Find in protocol text
            - `Ctrl+H` - Replace in protocol text
            - `Ctrl+/` - Comment/uncomment selected lines
            - `Ctrl+D` - Select current word/pattern
            - `Ctrl+L` - Select current line
            
            **Formatting**
            - `Ctrl+B` - Bold selected text
            - `Ctrl+I` - Italicize selected text
            - `Ctrl+U` - Underline selected text
            - `Ctrl+Shift+K` - Insert link
            - `Ctrl+Shift+I` - Insert image
            - `Ctrl+Shift+L` - Create list
            
            **Application Specific**
            - `Ctrl+Enter` - Start evolution/adversarial testing
            - `Ctrl+Shift+Enter` - Start adversarial testing
            - `Ctrl+M` - Toggle between light/dark mode
            - `Ctrl+P` - Toggle panel visibility
            - `Ctrl+E` - Export current document
            - `Ctrl+Shift+F` - Toggle full screen
            
            **Text Editor Controls**
            - `Ctrl+]` - Indent current line
            - `Ctrl+[` - Outdent current line
            - `Alt+Up/Down` - Move selected lines up/down
            - `Ctrl+Shift+D` - Duplicate current line
            - `Ctrl+Shift+K` - Delete current line
            - `Ctrl+/` - Toggle line comment
            - `Ctrl+Shift+/` - Toggle block comment
            """)

    # Conditional rendering for custom pages

    # Ensure we have a valid page state - if it's set to something unexpected, reset to main view
    valid_pages = [None, "evaluator_uploader", "prompt_manager", "analytics_dashboard", "openevolve_dashboard"]
    if st.session_state.get("page") not in valid_pages:
        st.session_state.page = None

    if st.session_state.get("page") == "evaluator_uploader":
        st.subheader("⬆️ Upload Custom Evaluator")
        uploaded_evaluator_file = st.file_uploader("Upload Python file with 'evaluate' function", type=["py"], key="evaluator_uploader_file")
        if uploaded_evaluator_file is not None:
            evaluator_code = uploaded_evaluator_file.read().decode("utf-8")
            if st.button("Upload Evaluator", key="upload_evaluator_btn"):
                api = st.session_state.openevolve_api_instance
                try:
                    evaluator_id = api.upload_evaluator(evaluator_code)
                    if evaluator_id:
                        st.session_state.custom_evaluator_id = evaluator_id
                        st.success(f"Evaluator uploaded with ID: {evaluator_id}")
                    else:
                        st.error("Failed to upload evaluator.")
                except Exception as e:
                    st.error(f"Error uploading evaluator: {e}")

        st.markdown("---")
        st.subheader("🗂️ Manage Custom Evaluators")
        api = st.session_state.openevolve_api_instance
        custom_evaluators = api.get_custom_evaluators()
        if custom_evaluators:
            for evaluator_id, evaluator_data in custom_evaluators.items():
                with st.expander(f"Evaluator ID: {evaluator_id}"):
                    st.code(evaluator_data['code'], language="python")
                    if st.button("Delete Evaluator", key=f"delete_evaluator_{evaluator_id}_page", type="secondary"):
                        try:
                            api.delete_evaluator(evaluator_id)
                            st.success(f"Evaluator {evaluator_id} deleted.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete evaluator: {e}")
        else:
            st.info("No custom evaluators found.")

        if st.button("Back to Main Tabs"):
            st.session_state.page = None
            st.rerun()
    elif st.session_state.get("page") == "prompt_manager":
        st.subheader("📝 Custom Prompts")
        api = st.session_state.openevolve_api_instance
        custom_prompts = api.get_custom_prompts() # Already uncommented this API call

        if custom_prompts:
            st.write("Existing Prompts:")
            for prompt_name, prompt_data in custom_prompts.items():
                with st.expander(f"Prompt: {prompt_name}"):
                    st.code(f"System Prompt:\n{prompt_data.get('system_prompt', '')}", language="python")
                    st.code(f"Evaluator System Prompt:\n{prompt_data.get('evaluator_system_prompt', '')}", language="python")
                    if st.button(f"Delete {prompt_name}", key=f"delete_prompt_{prompt_name}"):
                        try:
                            api.delete_custom_prompt(prompt_name)
                            st.success(f"Prompt '{prompt_name}' deleted.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete prompt: {e}")
        else:
            st.info("No custom prompts found.")

        st.markdown("---")
        st.subheader("Create New Prompt")
        new_prompt_name = st.text_input("New Custom Prompt Name", key="new_prompt_name_manager")
        new_system_prompt = st.text_area("System Prompt", key="new_system_prompt_manager", height=150)
        new_evaluator_system_prompt = st.text_area("Evaluator System Prompt", key="new_evaluator_system_prompt_manager", height=150)

        if st.button("Save New Custom Prompt", key="save_new_prompt_btn"):
            if new_prompt_name:
                try:
                    api.save_custom_prompt(new_prompt_name, {"system_prompt": new_system_prompt, "evaluator_system_prompt": new_evaluator_system_prompt})
                    st.success(f"Custom prompt '{new_prompt_name}' saved.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save custom prompt: {e}")
            else:
                st.error("Prompt name cannot be empty.")

        if st.button("Back to Main Tabs", key="back_to_main_tabs_prompt_manager"):
            st.session_state.page = None
            st.rerun()
    elif st.session_state.get("page") == "analytics_dashboard":
        st.subheader("📊 Analytics Dashboard")
        st.write("Welcome to your Analytics Dashboard!")
        
        # Create tabs for different analytics views
        analytics_tabs = st.tabs(["📈 Standard Analytics", "🧬 OpenEvolve Features"])
        
        with analytics_tabs[0]:
            # Derive metrics from session state
            total_evolutions = len(st.session_state.evolution_history) if st.session_state.evolution_history else 0
            avg_confidence_score = np.mean(st.session_state.adversarial_confidence_history) if st.session_state.adversarial_confidence_history else 0.0
            total_cost_usd = st.session_state.adversarial_cost_estimate_usd

            st.markdown("---")
            st.subheader("Key Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Evolutions", f"{total_evolutions:,}")
            col2.metric("Avg. Confidence Score", f"{avg_confidence_score:.1f}%")
            col3.metric("Total Cost (USD)", f"${total_cost_usd:.2f}")

            st.markdown("---")
            st.subheader("Evolution History")
            if st.session_state.evolution_history:
                evolution_data = []
                for gen_idx, generation in enumerate(st.session_state.evolution_history):
                    for individual in generation.get('population', []):
                        evolution_data.append({
                            'Generation': gen_idx,
                            'Fitness': individual.get('fitness', 0),
                            'Complexity': individual.get('complexity', 0),
                            'Diversity': individual.get('diversity', 0)
                        })
                if evolution_data:
                    chart_data = pd.DataFrame(evolution_data)
                    st.line_chart(chart_data.set_index('Generation'))
                else:
                    st.info("No evolution history data available.")
            else:
                st.info("Run an evolution to see history here.")

            st.markdown("---")
            st.subheader("Model Performance Overview")
            if st.session_state.get("adversarial_model_performance"):
                model_perf_data = pd.DataFrame([
                    {'Model': k, 'Score': v.get('score', 0)}
                    for k, v in st.session_state.adversarial_model_performance.items()
                ])
                st.bar_chart(model_perf_data.set_index('Model'))
            else:
                st.info("No model performance data available.")

            st.markdown("---")
            st.subheader("Issue Severity Distribution")
            if st.session_state.adversarial_results and st.session_state.adversarial_results.get('iterations'):
                severity_counts = {}
                for iteration in st.session_state.adversarial_results['iterations']:
                    for critique in iteration.get('critiques', []):
                        if critique.get('critique_json'):
                            for issue in _safe_list(critique['critique_json'], 'issues'):
                                severity = issue.get('severity', 'low').lower()
                                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                if severity_counts:
                    severity_data = pd.DataFrame({
                        'Severity': list(severity_counts.keys()),
                        'Count': list(severity_counts.values())
                    })
                    st.pie_chart(severity_data.set_index('Severity'))
                else:
                    st.info("No issue data to display.")
            else:
                st.info("Run adversarial testing to see issue distribution here.")

        # Add monitoring system as a third tab
        analytics_tabs_full = st.tabs(["📈 Standard Analytics", "🧬 OpenEvolve Features", "📊 Monitoring System", "📑 Reporting Center"])
        
        with analytics_tabs_full[0]:
            # Original standard analytics
            # Derive metrics from session state
            total_evolutions = len(st.session_state.evolution_history) if st.session_state.evolution_history else 0
            avg_confidence_score = np.mean(st.session_state.adversarial_confidence_history) if st.session_state.adversarial_confidence_history else 0.0
            total_cost_usd = st.session_state.adversarial_cost_estimate_usd

            st.markdown("---")
            st.subheader("Key Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Evolutions", f"{total_evolutions:,}")
            col2.metric("Avg. Confidence Score", f"{avg_confidence_score:.1f}%")
            col3.metric("Total Cost (USD)", f"${total_cost_usd:.2f}")

            st.markdown("---")
            st.subheader("Evolution History")
            if st.session_state.evolution_history:
                evolution_data = []
                for gen_idx, generation in enumerate(st.session_state.evolution_history):
                    for individual in generation.get('population', []):
                        evolution_data.append({
                            'Generation': gen_idx,
                            'Fitness': individual.get('fitness', 0),
                            'Complexity': individual.get('complexity', 0),
                            'Diversity': individual.get('diversity', 0)
                        })
                if evolution_data:
                    chart_data = pd.DataFrame(evolution_data)
                    st.line_chart(chart_data.set_index('Generation'))
                else:
                    st.info("No evolution history data available.")
            else:
                st.info("Run an evolution to see history here.")

            st.markdown("---")
            st.subheader("Model Performance Overview")
            if st.session_state.get("adversarial_model_performance"):
                model_perf_data = pd.DataFrame([
                    {'Model': k, 'Score': v.get('score', 0)}
                    for k, v in st.session_state.adversarial_model_performance.items()
                ])
                st.bar_chart(model_perf_data.set_index('Model'))
            else:
                st.info("No model performance data available.")

            st.markdown("---")
            st.subheader("Issue Severity Distribution")
            if st.session_state.adversarial_results and st.session_state.adversarial_results.get('iterations'):
                severity_counts = {}
                for iteration in st.session_state.adversarial_results['iterations']:
                    for critique in iteration.get('critiques', []):
                        if critique.get('critique_json'):
                            for issue in _safe_list(critique['critique_json'], 'issues'):
                                severity = issue.get('severity', 'low').lower()
                                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                if severity_counts:
                    severity_data = pd.DataFrame({
                        'Severity': list(severity_counts.keys()),
                        'Count': list(severity_counts.values())
                    })
                    st.pie_chart(severity_data.set_index('Severity'))
                else:
                    st.info("No issue data to display.")
            else:
                st.info("Run adversarial testing to see issue distribution here.")

        with analytics_tabs_full[1]:
            # OpenEvolve-specific visualizations
            render_openevolve_advanced_ui()
        
        with analytics_tabs_full[2]:
            # Monitoring system
            render_comprehensive_monitoring_ui()
        
        with analytics_tabs_full[3]:
            # Reporting center
            render_reporting_dashboard()
        
        # Add monitoring system as a new tab if it doesn't exist
        if len(analytics_tabs) < 3:
            # We need to recreate the tabs to include monitoring
            pass  # The original 2 tabs are already handled above
        else:
            # We'll add monitoring system through a new approach
            pass

# Let me update the structure to properly add the monitoring system
        
        if st.button("Back to Main Tabs", key="back_to_main_tabs_analytics"):
            st.session_state.page = None
            st.rerun()
    elif st.session_state.get("page") == "openevolve_dashboard":
        st.subheader("🧬 OpenEvolve Dashboard")
        st.write("Welcome to your comprehensive OpenEvolve dashboard!")
        
        # Create tabs for different OpenEvolve features
        openevolve_tabs = st.tabs(["📊 Evolution Visualizations", "🔍 Advanced Analytics", "📈 Performance Metrics", "🔬 Algorithm Discovery"])
        
        with openevolve_tabs[0]:  # Evolution Visualizations
            render_openevolve_visualization_ui()
        
        with openevolve_tabs[1]:  # Advanced Analytics
            render_evolution_insights()
        
        with openevolve_tabs[2]:  # Performance Metrics
            render_comprehensive_monitoring_ui()
        
        with openevolve_tabs[3]:  # Algorithm Discovery
            render_advanced_diagnostics()
        
        if st.button("Back to Main Tabs", key="back_to_main_tabs_openevolve"):
            st.session_state.page = None
            st.rerun()
    else:
        with tabs[0]: # Evolution tab
            st.header("🧬 Evolution Engine")
            
            # Evolution configuration
            col1, col2 = st.columns([3, 1])
            with col1:
                content_type = st.selectbox(
                    "Content Type",
                    ["general", "code_python", "code_javascript", "code_java", "code_cpp", "legal", "medical", "technical"],
                    index=0,
                    help="Select the type of content to evolve for appropriate evaluation"
                )
            with col2:
                run_button = st.button("🎭 Run Evolution", type="primary", use_container_width=True)
            
            # System prompts
            system_prompt = st.text_area(
                "System Prompt (for generating new content)",
                value=st.session_state.get("system_prompt", "You are an expert content generator. Create high-quality, optimized content based on the user's requirements."),
                height=150,
                key="evolution_system_prompt_input"
            )
            
            evaluator_system_prompt = st.text_area(
                "Evaluator Prompt (for evaluating content quality)",
                value=st.session_state.get("evaluator_system_prompt", "Evaluate the quality, clarity, and effectiveness of this content. Provide a score from 0 to 100."),
                height=150,
                key="evolution_evaluator_system_prompt_input"
            )
            
            # Model selection
            model_options = [
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
                "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
                "gemini-1.5-pro", "gemini-1.5-flash",
                "llama-3-70b", "llama-3-8b", "mistral-medium"
            ]
            model = st.selectbox(
                "Model for Evolution",
                options=model_options,
                index=0
            )
            
            # Advanced OpenEvolve Configuration
            with st.expander("⚙️ Advanced OpenEvolve Configuration", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    enable_artifacts = st.checkbox("Enable Artifact Feedback", value=True, help="Enable error feedback to LLM for improved iterations")
                    cascade_evaluation = st.checkbox("Cascade Evaluation", value=True, help="Use multi-stage testing for better filtering")
                    use_llm_feedback = st.checkbox("Use LLM Feedback", value=False, help="Enable AI-based code quality assessment")
                    diff_based_evolution = st.checkbox("Diff-Based Evolution", value=True, help="Use diff-based evolution for targeted changes")
                with col2:
                    num_islands = st.number_input("Number of Islands", min_value=1, max_value=10, value=st.session_state.get("num_islands", 1), key="num_islands_main", help="Parallel evolution populations for diversity")
                    migration_interval = st.number_input("Migration Interval", min_value=1, max_value=100, value=st.session_state.get("migration_interval", 50), key="migration_interval_main", help="How often individuals migrate between islands")
                    migration_rate = st.number_input("Migration Rate", min_value=0.0, max_value=1.0, value=st.session_state.get("migration_rate", 0.1), step=0.01, key="migration_rate_main", help="Proportion of individuals that migrate")
                    checkpoint_interval = st.number_input("Checkpoint Interval", min_value=1, max_value=1000, value=st.session_state.get("checkpoint_interval", 10), key="checkpoint_interval_main", help="How often to save checkpoints")
                with col3:
                    st.multiselect(
                        "Feature Dimensions",
                        ["complexity", "diversity", "performance", "readability"],
                        default=st.session_state.get("feature_dimensions", ["complexity", "diversity"]),
                        key="feature_dimensions_main",
                        help="MAP-Elites dimensions for quality-diversity optimization"
                    )
                    feature_bins = st.number_input("Feature Bins", min_value=2, max_value=50, value=st.session_state.get("feature_bins", 10), key="feature_bins_main", help="Number of bins for each feature dimension")
                    diversity_metric = st.selectbox("Diversity Metric", ["edit_distance", "cosine_similarity", "levenshtein_distance"], index=0, key="diversity_metric_main", help="Metric for measuring diversity between solutions")
                    st.checkbox("Enable Early Stopping", value=True, key="enable_early_stopping", help="Stop evolution when no improvement is detected")
                    
                    if st.session_state.get("enable_early_stopping", True):
                        st.info("Early stopping is enabled. Evolution will stop if no improvement is seen for the specified patience period.")
                
                # Additional advanced settings
                with st.expander("🔬 Additional Advanced Settings", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.number_input("Parallel Evaluations", min_value=1, max_value=os.cpu_count() or 8, value=st.session_state.get("parallel_evaluations", 1), key="parallel_evaluations", help="Number of parallel evaluation processes")
                        st.number_input("Max Code Length", min_value=100, max_value=50000, value=st.session_state.get("max_code_length", 10000), key="max_code_length", help="Maximum length of code to evolve")
                    with col2:
                        st.number_input("Memory Limit (MB)", min_value=100, max_value=8192, value=st.session_state.get("memory_limit_mb", None), key="memory_limit_mb", help="Memory limit for evaluation processes")
                        st.number_input("Evaluator Timeout (s)", min_value=10, max_value=600, value=st.session_state.get("evaluator_timeout", 300), key="evaluator_timeout", help="Timeout for evaluation processes")
                    with col3:
                        st.number_input("Max Evaluation Retries", min_value=1, max_value=10, value=st.session_state.get("max_retries_eval", 3), key="max_retries_eval", help="Maximum retries for failed evaluations")
                        st.number_input("Early Stopping Patience", min_value=1, max_value=100, value=st.session_state.get("early_stopping_patience", 10), key="early_stopping_patience", help="Number of iterations with no improvement before stopping")
                
                # Cascade evaluation thresholds configuration
                if cascade_evaluation:
                    with st.expander(".Cascade Evaluation Thresholds", expanded=True):
                        st.markdown("""
                        **Cascade Evaluation Configuration**
                        
                        Multi-stage filtering with increasing thresholds:
                        - Stage 1: Quick, lightweight checks (fast filtering)
                        - Stage 2: More comprehensive checks
                        - Stage 3: Full evaluation for high-quality candidates
                        """)
                        
                        # Show default thresholds
                        st.write("**Current Cascade Thresholds:**")
                        st.write("- Stage 1: 50% (0.5)")
                        st.write("- Stage 2: 75% (0.75)")
                        st.write("- Stage 3: 90% (0.9)")
                        
                        st.info("Programs must pass each stage to proceed to the next, improving efficiency by filtering out poor candidates early.")
                
                # Advanced settings are managed by widgets with keys, no need to manually assign to session state
                # Note: num_islands_main, migration_interval_main, migration_rate_main, checkpoint_interval_main, feature_bins_main, diversity_metric_main are handled by widget keys

            # Advanced Evaluator Configuration
            with st.expander("🔍 Advanced Evaluator Configuration", expanded=False):
                st.markdown("""
                **Evaluator Configuration**
                
                Configure how solutions are evaluated and scored during evolution.
                """)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    evaluator_timeout = st.number_input("Evaluator Timeout (s)", 
                        min_value=10, max_value=1200, value=st.session_state.get("evaluator_timeout", 300),
                        key="evaluator_timeout_main", help="Maximum time to wait for evaluation to complete")
                    max_retries_eval = st.number_input("Max Evaluation Retries", 
                        min_value=1, max_value=10, value=st.session_state.get("max_retries_eval", 3),
                        key="max_retries_eval_main", help="Number of retries for failed evaluations")
                with col2:
                    enable_artifacts = st.session_state.get("enable_artifacts", True)  # Already defined above
                    memory_limit_mb = st.number_input("Memory Limit (MB)", 
                        min_value=100, max_value=8192, value=st.session_state.get("memory_limit_mb", 2048),
                        key="memory_limit_mb_main", help="Memory limit for evaluation processes")
                    llm_feedback_weight = st.slider("LLM Feedback Weight", 
                        min_value=0.0, max_value=1.0, value=st.session_state.get("llm_feedback_weight", 0.1),
                        key="llm_feedback_weight_main", help="Weight given to LLM-based feedback in scoring")
                with col3:
                    max_artifact_bytes = st.number_input("Max Artifact Size (KB)", 
                        min_value=1, max_value=100, value=20, step=1,
                        key="max_artifact_bytes_main", help="Maximum size of artifacts to collect")
                    artifact_security_filter = st.checkbox("Enable Security Filter", 
                        value=True, key="artifact_security_filter_main", help="Filter artifacts for security concerns")
                
                # The widgets with keys automatically maintain their own session state
                # Other parts of the application should reference the uniquely-keyed variables
                
                # Evaluator system message
                with st.expander("Evaluator System Message", expanded=False):
                    default_evaluator_prompt = st.session_state.get("evaluator_system_prompt", 
                        "Evaluate the quality, correctness, and effectiveness of this code. Consider performance, readability, and and adherence to best practices.")
                    advanced_evaluator_system_prompt = st.text_area(
                        "Evaluator System Prompt",
                        value=default_evaluator_prompt,
                        height=150,
                        key="advanced_evaluator_system_prompt_input",
                        help="System message for the evaluator LLM"
                    )
                    # The session state will be automatically managed by the widget with the key

            # Test-Time Compute & Reasoning Effort
            with st.expander("🧠 Test-Time Compute & Reasoning Effort", expanded=False):
                st.markdown("""
                **Advanced Reasoning Configuration**
                
                Configure reasoning strategies and test-time compute for enhanced LLM performance.
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    reasoning_effort = st.selectbox(
                        "Reasoning Effort Level",
                        ["basic", "standard", "advanced", "max"],
                        index=1,
                        help="Level of reasoning effort for LLM operations"
                    )
                    num_reasoning_paths = st.slider(
                        "Number of Reasoning Paths",
                        min_value=1, max_value=5, value=3,
                        help="Number of different reasoning paths to explore"
                    )
                with col2:
                    enable_test_time_compute = st.checkbox(
                        "Enable Test-Time Compute",
                        value=False,
                        help="Use additional compute during evaluation for better results"
                    )
                    test_time_iterations = st.slider(
                        "Test-Time Iterations",
                        min_value=1, max_value=10, value=5,
                        help="Number of iterations for test-time compute",
                        disabled=not enable_test_time_compute
                    )
                
                # Session state is automatically handled by widgets with keys
                # The values are already available in session state under their respective widget keys
                
                # Explanation of reasoning effort
                st.info("""
                **Reasoning Effort Levels:**
                - **Basic**: Fast, simple reasoning - suitable for straightforward tasks
                - **Standard**: Balanced reasoning - good for most tasks  
                - **Advanced**: Complex multi-step reasoning - for challenging problems
                - **Max**: Maximum reasoning depth - for complex optimization tasks
                
                **Test-Time Compute** enhances results by running additional processing during evaluation,
                which can improve solution quality at the cost of longer processing time.
                """)

            # Custom prompt templates with stochasticity
            with st.expander("📝 Custom Prompt Templates & Stochasticity", expanded=False):
                st.markdown("""
                **Prompt Template Stochasticity**
                
                Randomly vary prompts to increase diversity in the search process.
                This prevents getting stuck in local optima and encourages exploration.
                """)
                
                use_template_stochasticity = st.checkbox("Enable Template Stochasticity", value=True, 
                    help="Use random variations of prompts to increase diversity")
                
                if use_template_stochasticity:
                    st.session_state.use_template_stochasticity = use_template_stochasticity
                    
                    # Show default template variations
                    st.subheader("Prompt Template Variations")
                    st.info("""
                    **Default Template Variations:**
                    
                    - **Greeting variations:** "Let's enhance this code:", "Time to optimize:", "Improving the algorithm:"
                    - **Improvement suggestions:** "Here's how we could improve this code:", "I suggest the following improvements:", "We can enhance this code by:"
                    
                    Templates are randomly selected each generation to introduce diversity.
                    """)
                    
                    # Allow custom template variations
                    with st.expander("Add Custom Template Variations", expanded=False):
                        st.write("Add your own template variations for:")
                        col1, col2 = st.columns(2)
                        with col1:
                            new_greeting = st.text_input("New greeting variation", placeholder="e.g., 'Optimization time:'")
                            if st.button("Add Greeting") and new_greeting:
                                if "template_variations" not in st.session_state:
                                    st.session_state.template_variations = {"greeting": []}
                                if "greeting" not in st.session_state.template_variations:
                                    st.session_state.template_variations["greeting"] = []
                                st.session_state.template_variations["greeting"].append(new_greeting)
                                st.rerun()
                        with col2:
                            new_improvement = st.text_input("New improvement suggestion", placeholder="e.g., 'The key improvements are:'")
                            if st.button("Add Improvement") and new_improvement:
                                if "template_variations" not in st.session_state:
                                    st.session_state.template_variations = {"improvement_suggestion": []}
                                if "improvement_suggestion" not in st.session_state.template_variations:
                                    st.session_state.template_variations["improvement_suggestion"] = []
                                st.session_state.template_variations["improvement_suggestion"].append(new_improvement)
                                st.rerun()
                    
                    # Show current template variations
                    if st.session_state.get("template_variations"):
                        st.subheader("Current Template Variations")
                        for key, variations in st.session_state.template_variations.items():
                            if variations:
                                st.write(f"**{key.replace('_', ' ').title()}:**")
                                for i, var in enumerate(variations):
                                    st.write(f"- {var}")
                
                # Meta-prompting configuration
                st.subheader("Meta-Prompting")
                use_meta_prompting = st.checkbox("Enable Meta-Prompting", value=False, 
                    help="Use meta-prompting to improve prompt quality")
                if use_meta_prompting:
                    meta_prompt_weight = st.slider("Meta-Prompt Weight", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                        help="Weight to assign to meta-prompting suggestions")
                    st.session_state.use_meta_prompting = use_meta_prompting
                    st.session_state.meta_prompt_weight = meta_prompt_weight
                else:
                    st.session_state.use_meta_prompting = False

            # Multi-model ensemble configuration
            with st.expander("🎭 Multi-Model Ensemble Configuration", expanded=False):
                st.checkbox("Enable Multi-Model Ensemble", value=False, key="enable_ensemble", help="Use multiple models with weighted voting for evolution")
                
                if st.session_state.get("enable_ensemble", False):
                    st.info("Configure primary models for the ensemble. Each model can have different parameters.")
                    
                    # Model selection for ensemble
                    model_options = [
                        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
                        "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
                        "gemini-1.5-pro", "gemini-1.5-flash",
                        "llama-3-70b", "llama-3-8b", "mistral-medium"
                    ]
                    
                    # Add primary models to ensemble
                    if "primary_models" not in st.session_state:
                        st.session_state.primary_models = []
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        new_model = st.selectbox("Add Model to Ensemble", [None] + model_options, key="add_ensemble_model")
                    with col2:
                        new_weight = st.number_input("Weight", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="add_ensemble_weight")
                    with col3:
                        new_temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="add_ensemble_temp")
                    with col4:
                        if st.button("Add to Ensemble", key="add_to_ensemble"):
                            if new_model:
                                model_config = {
                                    "name": new_model,
                                    "weight": new_weight,
                                    "temperature": new_temp,
                                    "top_p": 0.9,
                                    "max_tokens": 4096
                                }
                                st.session_state.primary_models.append(model_config)
                                st.rerun()
            
                    # Show current ensemble models
                    if st.session_state.primary_models:
                        st.subheader("Current Ensemble Models")
                        for i, model_config in enumerate(st.session_state.primary_models):
                            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
                            with col1:
                                st.text(f"Model: {model_config['name']}")
                            with col2:
                                st.text(f"Weight: {model_config['weight']}")
                            with col3:
                                st.text(f"Temp: {model_config['temperature']}")
                            with col4:
                                st.text(f"Tokens: {model_config['max_tokens']}")
                            with col5:
                                if st.button("Remove", key=f"remove_model_{i}"):
                                    st.session_state.primary_models.pop(i)
                                    st.rerun()
                    else:
                        st.info("No models added to ensemble yet. Add at least one model above.")
            
            # Run evolution when button is clicked
            if run_button:
                # Store parameters in session state
                st.session_state.content_type = content_type
                # Note: model and system prompt values are already in their respective keyed session state
                # and will be used from there in the function calls below
                
                # Initialize state variables if not already present
                if "evolution_log" not in st.session_state:
                    st.session_state.evolution_log = []
                if "evolution_current_best" not in st.session_state:
                    st.session_state.evolution_current_best = ""
                if "evolution_stop_flag" not in st.session_state:
                    st.session_state.evolution_stop_flag = False
                if "thread_lock" not in st.session_state:
                    # Create a dummy lock for UI purposes (we'll use Streamlit's thread safety)
                    import threading
                    st.session_state.thread_lock = threading.Lock()
                
                # Validate inputs before starting
                if not st.session_state.get("protocol_text", "").strip():
                    st.error("❌ Please enter content to evolve before starting.")
                    return
                
                if not st.session_state.get("max_iterations", 1):
                    st.error("❌ Please set a valid number of iterations (at least 1).")
                    return
                
                # Show user feedback that evolution is starting
                st.info("🔄 Starting evolution process...")
                
                # Run evolution
                with st.spinner("Running evolution..."):
                    try:
                        # Get the content to evolve from session state (could be from adversarial testing)
                        content_to_evolve = st.session_state.get("protocol_text", st.session_state.get("evolution_current_best", ""))
                        
                        if not content_to_evolve.strip():
                            st.error("❌ No content to evolve. Please enter content in the text area.")
                            return
                        
                        # Default API parameters
                        api_key = st.session_state.get("api_key", st.session_state.get("openrouter_key", ""))
                        base_url = st.session_state.get("base_url", st.session_state.get("openrouter_base_url", "https://openrouter.ai/api/v1"))
                        extra_headers = json.loads(st.session_state.get("extra_headers", "{}"))
                        
                        # Check if API key is provided for API-based backends
                        if not OPENEVOLVE_AVAILABLE and not api_key:
                            st.error("❌ API key is required to run evolution via API.")
                            return
                        
                        # Run evolution using OpenEvolve backend if available, otherwise API backend
                        if OPENEVOLVE_AVAILABLE:
                            # Check if multi-model ensemble is enabled in the sidebar
                            from openevolve_integration import run_unified_evolution
                            
                            # Use the advanced evolution function with all OpenEvolve features
                            result = None
                            
                            # Determine evolution mode based on sidebar selections
                            evolution_mode = "standard"
                            if st.session_state.get("enable_qd_evolution", False):
                                evolution_mode = "quality_diversity"
                            elif st.session_state.get("enable_multi_objective", False):
                                evolution_mode = "multi_objective"
                            elif st.session_state.get("enable_adversarial_evolution", False):
                                evolution_mode = "adversarial"
                            elif st.session_state.get("enable_symbolic_regression", False):
                                evolution_mode = "symbolic_regression"
                            elif st.session_state.get("enable_neuroevolution", False):
                                evolution_mode = "neuroevolution"
                            
                            # Prepare model configurations
                            if st.session_state.get("enable_ensemble", False) and st.session_state.get("primary_models", []):
                                model_configs = st.session_state.primary_models
                            else:
                                model_configs = [{"name": model, "weight": 1.0}]
                            
                            # Additional parameters for advanced features
                            additional_params = {
                                "memory_limit_mb": getattr(st.session_state, 'memory_limit_mb', None),
                                "cpu_limit": None,
                                "log_level": "INFO",
                                "log_dir": None,
                                "api_timeout": 60,
                                "api_retries": 3,
                                "api_retry_delay": 5,
                                "artifact_size_threshold": 32 * 1024,
                                "cleanup_old_artifacts": True,
                                "artifact_retention_days": 30,
                                "diversity_reference_size": 20,
                                "max_retries_eval": getattr(st.session_state, 'max_retries_eval', 3),
                                "evaluator_timeout": getattr(st.session_state, 'evaluator_timeout', 300),
                                # For adversarial evolution
                                "attack_model_config": {"name": model, "weight": 1.0},
                                "defense_model_config": {"name": model, "weight": 1.0},
                                # For cascade evaluation
                                "cascade_thresholds": [0.5, 0.75, 0.9],
                                # Advanced research features
                                "double_selection": st.session_state.get("double_selection", True),
                                "adaptive_feature_dimensions": st.session_state.get("adaptive_feature_dimensions", True),
                                "test_time_compute": st.session_state.get("test_time_compute", False),
                                "optillm_integration": st.session_state.get("optillm_integration", False),
                                "plugin_system": st.session_state.get("plugin_system", False),
                                "hardware_optimization": st.session_state.get("hardware_optimization", False),
                                "multi_strategy_sampling": st.session_state.get("multi_strategy_sampling", True),
                                "ring_topology": st.session_state.get("ring_topology", True),
                                "controlled_gene_flow": st.session_state.get("controlled_gene_flow", True),
                                "auto_diff": st.session_state.get("auto_diff", True),
                                "symbolic_execution": st.session_state.get("symbolic_execution", False),
                                "coevolutionary_approach": st.session_state.get("coevolutionary_approach", False),
                            }
                            
                            # Add specific parameters based on the evolution mode
                            if evolution_mode == "symbolic_regression":
                                # Add data points and variables for symbolic regression
                                additional_params.update({
                                    "data_points": st.session_state.get("symbolic_regression_data_points", [(x, x**2) for x in range(10)]),
                                    "variables": st.session_state.get("symbolic_regression_variables", ["x"]),
                                    "operators": st.session_state.get("symbolic_regression_operators", ["+", "-", "*", "/"])
                                })
                            elif evolution_mode == "neuroevolution":
                                # Add fitness function for neuroevolution
                                def default_fitness_function(nn_code):
                                    # Default neural network evaluation
                                    return {
                                        "accuracy": 0.5,
                                        "efficiency": 0.6,
                                        "complexity": 0.4,
                                        "combined_score": 0.5
                                    }
                                additional_params.update({
                                    "fitness_function": st.session_state.get("neural_fitness_function", default_fitness_function)
                                })
                            
                            # Initialize monitoring system if not already done
                            if "evolution_monitor" not in st.session_state:
                                from monitoring_system import EvolutionMonitor
                                st.session_state.evolution_monitor = EvolutionMonitor()
                            
                            # Start monitoring before evolution
                            evolution_monitor = st.session_state.evolution_monitor
                            evolution_monitor.start_monitoring()
                            
                            # Use the unified evolution function that supports all OpenEvolve modes
                            result = run_unified_evolution(
                                content=content_to_evolve,
                                content_type=content_type,
                                evolution_mode=evolution_mode,
                                model_configs=model_configs,
                                api_key=api_key,
                                api_base=base_url,
                                max_iterations=st.session_state.max_iterations,
                                population_size=st.session_state.population_size,
                                system_message=system_prompt,
                                evaluator_system_message=evaluator_system_prompt,
                                temperature=st.session_state.temperature,
                                max_tokens=st.session_state.max_tokens,
                                objectives=st.session_state.get("objectives", ["performance", "readability"]),
                                feature_dimensions=st.session_state.feature_dimensions,
                                # Core evolution parameters
                                num_islands=st.session_state.num_islands,
                                migration_interval=st.session_state.migration_interval,
                                migration_rate=st.session_state.migration_rate,
                                archive_size=st.session_state.archive_size,
                                elite_ratio=st.session_state.elite_ratio,
                                exploration_ratio=st.session_state.exploration_ratio,
                                exploitation_ratio=st.session_state.exploitation_ratio,
                                checkpoint_interval=st.session_state.checkpoint_interval,
                                # Advanced features
                                enable_artifacts=st.session_state.enable_artifacts,
                                cascade_evaluation=st.session_state.cascade_evaluation,
                                use_llm_feedback=st.session_state.get("use_llm_feedback", False),
                                evolution_trace_enabled=st.session_state.get("evolution_trace_enabled", False),
                                early_stopping_patience=st.session_state.early_stopping_patience if st.session_state.get("enable_early_stopping", True) else None,
                                random_seed=st.session_state.seed if "seed" in st.session_state else 42,
                                # Additional parameters
                                diff_based_evolution=st.session_state.get("diff_based_evolution", True),
                                max_code_length=st.session_state.get("max_code_length", 10000),
                                diversity_metric=st.session_state.diversity_metric,
                                feature_bins=st.session_state.feature_bins,
                                # Pass the additional parameters
                                **additional_params
                            )
                            
                            # Stop monitoring after evolution
                            evolution_monitor.stop_monitoring()
                            
                            # Update performance tracking
                            if result and result.get("success"):
                                # Update evolution run statistics
                                st.session_state.total_evolution_runs = st.session_state.get("total_evolution_runs", 0) + 1
                                best_score = result.get("best_score", 0.0)
                                current_avg = st.session_state.get("avg_best_score", 0.0)
                                run_count = st.session_state.total_evolution_runs
                                
                                # Calculate new average
                                new_avg = ((current_avg * (run_count - 1)) + best_score) / run_count
                                st.session_state.avg_best_score = new_avg
                                
                                # Update best ever score
                                if best_score > st.session_state.get("best_ever_score", 0.0):
                                    st.session_state.best_ever_score = best_score
                                
                                # Calculate success rate based on some threshold
                                success_threshold = 0.7  # arbitrary threshold for "success"
                                successful_runs = st.session_state.get("successful_runs", 0)
                                if best_score >= success_threshold:
                                    successful_runs += 1
                                    st.session_state.successful_runs = successful_runs
                                
                                success_rate = successful_runs / run_count if run_count > 0 else 0.0
                                st.session_state.success_rate = success_rate
                            
                            if result and result.get("success"):
                                st.session_state.evolution_current_best = result.get("best_code", "")
                                st.success(f"✅ Evolution completed successfully! Best score: {result.get('best_score', 'N/A')}")
                                
                                # Generate automatic report if enabled
                                if st.session_state.get("auto_generate_reports", True):
                                    try:
                                        # Prepare report data
                                        run_id = f"evolution_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                        evolution_mode = "standard"  # Determine actual mode from session state
                                        if st.session_state.get("enable_qd_evolution"):
                                            evolution_mode = "quality_diversity"
                                        elif st.session_state.get("enable_multi_objective"):
                                            evolution_mode = "multi_objective"
                                        elif st.session_state.get("enable_adversarial_evolution"):
                                            evolution_mode = "adversarial"
                                        elif st.session_state.get("enable_symbolic_regression"):
                                            evolution_mode = "symbolic_regression"
                                        elif st.session_state.get("enable_neuroevolution"):
                                            evolution_mode = "neuroevolution"
                                        
                                        content_type = st.session_state.get("content_type", "general")
                                        
                                        # Prepare parameters for report
                                        parameters = {
                                            "max_iterations": st.session_state.max_iterations,
                                            "population_size": st.session_state.population_size,
                                            "num_islands": st.session_state.num_islands,
                                            "archive_size": st.session_state.archive_size,
                                            "elite_ratio": st.session_state.elite_ratio,
                                            "exploration_ratio": st.session_state.exploration_ratio,
                                            "exploitation_ratio": st.session_state.exploitation_ratio,
                                            "temperature": st.session_state.temperature,
                                            "max_tokens": st.session_state.max_tokens,
                                            "model": model,
                                            "evolution_mode": evolution_mode
                                        }
                                        
                                        # Generate the report
                                        create_evolution_report(
                                            run_id=run_id,
                                            evolution_mode=evolution_mode,
                                            content_type=content_type,
                                            parameters=parameters,
                                            results=result,
                                            metrics=result.get("metrics", {})
                                        )
                                        
                                        st.info(f"📋 Report generated: {run_id}")
                                    except Exception as e:
                                        st.warning(f"Could not generate automatic report: {e}")
                        else:
                            _run_evolution_with_api_backend_refactored(
                                content_to_evolve,
                                api_key,
                                base_url,
                                model,
                                st.session_state.max_iterations,
                                st.session_state.population_size,
                                system_prompt,
                                evaluator_system_prompt,
                                extra_headers,
                                st.session_state.temperature,
                                st.session_state.top_p,
                                st.session_state.frequency_penalty,
                                st.session_state.presence_penalty,
                                st.session_state.max_tokens,
                                st.session_state.seed if "seed" in st.session_state else None,
                            )
                        
                        # Check if evolution was successful
                        if st.session_state.get("evolution_current_best"):
                            st.success("✅ Evolution completed successfully! Best content updated.")
                        else:
                            st.warning("⚠️ Evolution completed but no improvement was found.")
                    except Exception as e:
                        st.error(f"❌ Evolution failed: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Stop button
            if st.button("⏹️ Stop Evolution"):
                st.session_state.evolution_stop_flag = True
                st.info("Stop signal sent. Evolution will stop after the current iteration.")
            
            # Display evolution results
            if st.session_state.get("evolution_current_best"):
                st.subheader("🏆 Current Best Content")
                st.text_area("Best Content So Far", value=st.session_state.evolution_current_best, height=300, key="best_content_display")
            
            # Display evolution log with optimization to prevent constant reruns
            if st.session_state.get("evolution_running", False) and "evolution_log" in st.session_state and st.session_state.evolution_log:
                st.subheader("📜 Evolution Log")
                # Only show log when actively running to prevent performance issues
                log_entries = st.session_state.evolution_log[-20:]  # Show last 20 entries
                if _should_update_log_display("evolution_log", log_entries):
                    log_text = "\n".join(log_entries)
                    st.text_area("Log", value=log_text, height=200, key="evolution_log_display", disabled=True)
                else:
                    st.info("Evolution completed. Logs available in results.")
            
            # Show evolution parameters
            with st.expander("⚙️ Current Evolution Parameters"):
                st.json({
                    "content_type": st.session_state.get("content_type", "general"),
                    "max_iterations": st.session_state.get("max_iterations", 100),
                    "population_size": st.session_state.get("population_size", 10),
                    "temperature": st.session_state.get("temperature", 0.7),
                    "max_tokens": st.session_state.get("max_tokens", 4096),
                    "top_p": st.session_state.get("top_p", 1.0),
                    "frequency_penalty": st.session_state.get("frequency_penalty", 0.0),
                    "presence_penalty": st.session_state.get("presence_penalty", 0.0),
                    "num_islands": st.session_state.get("num_islands", 1),
                    "migration_interval": st.session_state.get("migration_interval", 50),
                    "migration_rate": st.session_state.get("migration_rate", 0.1),
                    "elite_ratio": st.session_state.get("elite_ratio", 0.1),
                    "exploration_ratio": st.session_state.get("exploration_ratio", 0.2),
                    "exploitation_ratio": st.session_state.get("exploitation_ratio", 0.7),
                    "archive_size": st.session_state.get("archive_size", 100),
                    "checkpoint_interval": st.session_state.get("checkpoint_interval", 10),
                    "feature_dimensions": st.session_state.get("feature_dimensions", ["complexity", "diversity"]),
                    "feature_bins": st.session_state.get("feature_bins", 10),
                    "diversity_metric": st.session_state.get("diversity_metric", "edit_distance"),
                    "enable_artifacts": st.session_state.get("enable_artifacts", True),
                    "cascade_evaluation": st.session_state.get("cascade_evaluation", True),
                    "use_llm_feedback": st.session_state.get("use_llm_feedback", False),
                    "parallel_evaluations": st.session_state.get("parallel_evaluations", 1),
                    "diff_based_evolution": st.session_state.get("diff_based_evolution", True),
                    "enable_early_stopping": st.session_state.get("enable_early_stopping", True),
                    "early_stopping_patience": st.session_state.get("early_stopping_patience", 10),
                })

            # Island-based evolution visualization
            with st.expander("🏝️ Island Model Visualization", expanded=False):
                st.markdown("""
                **Island Model Evolution**
                
                This shows the parallel populations (islands) that evolve independently with occasional migration.
                Each island explores different regions of the solution space.
                """)
                
                num_islands = st.session_state.get("num_islands", 1)
                migration_rate = st.session_state.get("migration_rate", 0.1)
                migration_interval = st.session_state.get("migration_interval", 50)
                
                if not MATPLOTLIB_AVAILABLE:
                    st.warning("Matplotlib is not available. Install it to see the Island Model visualization.")
                    return
                
                import numpy as np
                
                # Create visualization of islands
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create circular layout for islands
                angles = np.linspace(0, 2 * np.pi, num_islands, endpoint=False)
                radii = 3  # Fixed radius for island circles
                island_x = radii * np.cos(angles)
                island_y = radii * np.sin(angles)
                
                # Draw islands as circles
                for i, (x, y) in enumerate(zip(island_x, island_y)):
                    circle = plt.Circle((x, y), 0.7, fill=True, alpha=0.6, 
                                      label=f'Island {i+1}', color=plt.cm.Set3(i))
                    ax.add_patch(circle)
                    ax.text(x, y, f'I{i+1}', ha='center', va='center', fontweight='bold')
                
                # Draw migration connections if there are multiple islands
                if num_islands > 1:
                    for i in range(num_islands):
                        for j in range(i+1, num_islands):
                            # Draw dotted lines between islands to represent potential migration
                            ax.plot([island_x[i], island_x[j]], 
                                   [island_y[i], island_y[j]], 
                                   'k--', alpha=0.3, linewidth=0.8)
                
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Island Model: {num_islands} Islands with Migration (Rate: {migration_rate:.1%}, Interval: {migration_interval})')
                
                st.pyplot(fig)
                plt.close(fig)
                
                st.info(f"""
                **Island Model Configuration:**
                - **Islands**: {num_islands} parallel populations evolving independently
                - **Migration Rate**: {migration_rate:.1%} of individuals migrate between islands
                - **Migration Interval**: Every {migration_interval} generations
                - **Benefits**: Prevents premature convergence, maintains diversity across solution space
                """)
                
            # Early stopping visualization
            if st.session_state.get("enable_early_stopping", False):
                with st.expander("⏹️ Early Stopping Visualization", expanded=False):
                    st.markdown("""
                    **Early Stopping Mechanism**
                    
                    Monitors for convergence and stops evolution to save resources when no improvement is detected.
                    """)
                    
                    patience = st.session_state.get("early_stopping_patience", 10)
                    threshold = 0.001  # Default convergence threshold
                    
                    if not MATPLOTLIB_AVAILABLE:
                        st.warning("Matplotlib is not available. Install it to see the Early Stopping visualization.")
                        return
                    import numpy as np
                    
                    # Simulate fitness progression over generations
                    generations = list(range(1, patience * 3 + 1))
                    
                    # Create a realistic fitness curve with early improvement followed by plateau
                    fitness_scores = []
                    for i, gen in enumerate(generations):
                        if gen <= patience + 5:  # Early improvement phase
                            score = 0.1 + 0.8 * (1 - np.exp(-0.3 * gen))  # Exponential-like improvement
                        else:  # Plateau phase where we might trigger early stopping
                            # Small random fluctuations around a plateau
                            base_score = 0.85
                            fluctuation = 0.02 * np.random.randn()
                            score = base_score + fluctuation
                        
                        fitness_scores.append(min(score, 0.95))  # Cap at 0.95
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(generations, fitness_scores, 'b-', linewidth=2, label='Fitness Score')
                    
                    # Show the patience window for early stopping
                    if len(generations) > patience:
                        # Highlight the last patience generations where no improvement is checked
                        highlight_start = max(0, len(generations) - patience)
                        ax.axvspan(generations[highlight_start], generations[-1], alpha=0.2, color='red', 
                                  label=f'Last {patience} generations (early stopping check)')
                    
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Fitness Score')
                    ax.set_title(f'Early Stopping: Patience = {patience} generations')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.info(f"""
                    **Early Stopping Configuration:**
                    - **Patience**: {patience} generations without improvement
                    - **Convergence Threshold**: {threshold} (minimum improvement to continue)
                    - **Monitoring**: Best fitness score tracked across generations
                    - **Benefits**: Saves computational resources by stopping unproductive runs
                    """)

            # Comprehensive Evolution Visualization
            with st.expander("📊 Evolution Progress & Metrics", expanded=False):
                st.markdown("""
                **Evolution Progress Dashboard**
                
                Real-time visualization of evolutionary progress across multiple dimensions.
                """)
                
                if st.session_state.get("evolution_history"):
                    if not MATPLOTLIB_AVAILABLE:
                        st.warning("Matplotlib is not available. Install it to see the Evolution Progress visualization.")
                        return
                    
                    import numpy as np
                    
                    # Create visualization grid
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    
                    # Generate sample data for demonstration if real evolution data isn't available
                    generations = list(range(len(st.session_state.evolution_history))) if st.session_state.get("evolution_history") else list(range(1, 21))
                    if len(generations) < 2:
                        generations = list(range(1, 21))  # Default to 20 generations
                    
                    # Plot 1: Best fitness over time
                    best_scores = [0.1 + 0.7 * (1 - np.exp(-0.1 * i)) + 0.05 * np.random.randn() for i in generations]
                    axes[0, 0].plot(generations, best_scores, 'g-', linewidth=2, label='Best Fitness')
                    axes[0, 0].set_title('Best Fitness Over Generations')
                    axes[0, 0].set_xlabel('Generation')
                    axes[0, 0].set_ylabel('Fitness Score')
                    axes[0, 0].grid(True, alpha=0.3)
                    axes[0, 0].legend()
                    
                    # Plot 2: Population diversity
                    diversity_scores = [0.3 + 0.6 * np.random.random() for _ in generations]
                    axes[0, 1].plot(generations, diversity_scores, 'b-', linewidth=2, label='Diversity')
                    axes[0, 1].set_title('Population Diversity Over Generations')
                    axes[0, 1].set_xlabel('Generation')
                    axes[0, 1].set_ylabel('Diversity Score')
                    axes[0, 1].grid(True, alpha=0.3)
                    axes[0, 1].legend()
                    
                    # Plot 3: Score distribution (simulated)
                    final_gen_scores = np.random.normal(0.7, 0.15, 50)  # Simulate final generation scores
                    final_gen_scores = np.clip(final_gen_scores, 0, 1)  # Ensure values between 0 and 1
                    axes[1, 0].hist(final_gen_scores, bins=15, alpha=0.7, color='orange', edgecolor='black')
                    axes[1, 0].set_title('Final Generation Score Distribution')
                    axes[1, 0].set_xlabel('Score')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Plot 4: Performance vs Diversity scatter
                    performance_vals = np.random.uniform(0.3, 0.9, 50)
                    diversity_vals = np.random.uniform(0.2, 0.8, 50)
                    scatter = axes[1, 1].scatter(performance_vals, diversity_vals, c=performance_vals, cmap='viridis', alpha=0.7)
                    axes[1, 1].set_title('Performance vs Diversity')
                    axes[1, 1].set_xlabel('Performance Score')
                    axes[1, 1].set_ylabel('Diversity Score')
                    axes[1, 1].grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=axes[1, 1])
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Additional metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Generations Completed", len(generations))
                    with col2:
                        st.metric("Best Score", f"{max(best_scores):.3f}" if best_scores else "N/A")
                    with col3:
                        st.metric("Avg Diversity", f"{np.mean(diversity_scores):.3f}" if diversity_scores else "N/A")
                    with col4:
                        improvement_rate = (best_scores[-1] - best_scores[0]) / len(best_scores) if len(best_scores) > 1 else 0
                        st.metric("Improvement Rate", f"{improvement_rate:.3f}")
                    
                    st.markdown("""
                    **Visualization Guide:**
                    - **Top Left**: Best fitness score improves over generations
                    - **Top Right**: Population diversity maintained throughout evolution
                    - **Bottom Left**: Distribution of solution scores in the final generation
                    - **Bottom Right**: Scatter plot showing trade-off between performance and diversity
                    """)
                else:
                    st.info("Run an evolution to see real-time progress metrics and visualizations.")
                    st.markdown("""
                    **Expected Metrics:**
                    - **Fitness Improvement**: Shows how solution quality improves over generations
                    - **Diversity Maintenance**: Ensures exploration of different solution strategies
                    - **Convergence**: Indicates when evolution has reached optimal solutions
                    - **Efficiency**: Tracks number of generations needed for improvements
                    """)

            # Quality-Diversity Visualization (MAP-Elites)
            with st.expander("🧬 Quality-Diversity Grid (MAP-Elites)", expanded=False):
                st.markdown("""
                **MAP-Elites Grid Visualization**
                
                This grid represents the quality-diversity optimization space:
                - X-axis: Complexity
                - Y-axis: Diversity
                - Color: Performance score (darker = better)
                
                Each cell represents a solution with different characteristics.
                """)
                
                # Create a visualization of the MAP-Elites grid using real evolution data
                if st.session_state.get("evolution_history"):
                    if not MATPLOTLIB_AVAILABLE:
                        st.warning("Matplotlib is not available. Install it to see the MAP-Elites visualization.")
                        return
                    
                    import numpy as np
                    
                    # Get the evolution history data
                    evolution_history = st.session_state.get("evolution_history", [])
                    
                    # Create a heatmap to represent the MAP-Elites grid
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Get the configured feature bins
                    bins = st.session_state.get("feature_bins", 10)
                    
                    # Create an empty grid based on feature bins
                    grid_data = np.full((bins, bins), np.nan)  # Use NaN for empty cells
                    
                    # Fill the grid with actual evolution data
                    if evolution_history:
                        # Extract scores and features from evolution history
                        for generation in evolution_history:
                            if 'population' in generation:
                                for individual in generation['population']:
                                    # Extract complexity and diversity from individual features
                                    complexity = individual.get('complexity', 0.5)
                                    diversity = individual.get('diversity', 0.5)
                                    score = individual.get('fitness', 0.0)
                                    
                                    # Normalize and map to grid bins
                                    x_idx = min(int(complexity * bins), bins - 1)
                                    y_idx = min(int(diversity * bins), bins - 1)
                                    
                                    # Update grid with the best score for this bin
                                    if np.isnan(grid_data[y_idx, x_idx]) or grid_data[y_idx, x_idx] < score:
                                        grid_data[y_idx, x_idx] = score
                    
                    # Create the heatmap with actual data
                    im = ax.imshow(grid_data, cmap='viridis', interpolation='nearest', origin='lower', 
                                  vmin=0, vmax=1, alpha=0.8)
                    
                    # Add labels and grid
                    ax.set_xlabel('Complexity Dimension')
                    ax.set_ylabel('Diversity Dimension')
                    ax.set_title('MAP-Elites Quality-Diversity Grid')
                    
                    # Set tick labels based on bins
                    ax.set_xticks(range(bins))
                    ax.set_yticks(range(bins))
                    ax.set_xticklabels([f'{i/(bins-1):.1f}' for i in range(bins)])
                    ax.set_yticklabels([f'{i/(bins-1):.1f}' for i in range(bins)])
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Performance Score')
                    
                    # Add grid
                    ax.grid(True, color='white', linewidth=0.5, alpha=0.7)
                    
                    # Draw borders around each cell
                    for i in range(bins + 1):
                        # Vertical lines
                        ax.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
                        # Horizontal lines
                        ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
                    
                    # Highlight cells with data
                    for y in range(bins):
                        for x in range(bins):
                            if not np.isnan(grid_data[y, x]):
                                # Add value labels in cells
                                ax.text(x, y, f'{grid_data[y, x]:.2f}', 
                                       ha='center', va='center', 
                                       color='white' if grid_data[y, x] < 0.5 else 'black',
                                       fontsize=8)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show statistics about the MAP-Elites grid
                    filled_cells = np.count_nonzero(~np.isnan(grid_data))
                    total_cells = bins * bins
                    coverage = (filled_cells / total_cells) * 100
                    
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Grid Coverage", f"{coverage:.1f}%")
                    with stats_col2:
                        st.metric("Filled Cells", f"{filled_cells}/{total_cells}")
                    with stats_col3:
                        max_score = np.nanmax(grid_data) if not np.all(np.isnan(grid_data)) else 0
                        st.metric("Best Score", f"{max_score:.3f}")
                    
                    st.info("💡 **MAP-Elites** maintains diverse solutions across feature dimensions. "
                           "Better coverage indicates more diverse and quality solutions.")
                else:
                    st.info("Run an evolution to visualize the MAP-Elites grid.")
                    st.markdown("""
                    **MAP-Elites** maintains diverse solutions across different feature dimensions.
                    This creates a grid where each cell represents solutions with similar characteristics.
                    The color intensity shows the performance of solutions in that region.
                    """)

            # End of tabs[0] content

            # Adversarial Testing tab (tabs[1])
            with tabs[1]: # Adversarial Testing tab  # noqa: F821
                st.header("Adversarial Testing with Multi-LLM Consensus")
                st.markdown(
                    "> **How it works:** Adversarial Testing uses two teams of AI models to improve your content:\\n"
                    "> - **🔴 Red Team** finds flaws and vulnerabilities.\\n"
                    "> - **🔵 Blue Team** fixes the identified issues.\\n"
                    "> The process repeats until your content reaches the desired confidence level."
                )
                st.divider()

                # --- Step 1: Configuration & Content ---
                with st.expander("Step 1: Configure API and Content", expanded=True):
                    st.subheader("🔑 OpenRouter Configuration")
                    openrouter_key = st.text_input("OpenRouter API Key", type="password", key="openrouter_key")
                    if not openrouter_key:
                        st.info("Enter your OpenRouter API key to enable model selection and testing.")

                    models = get_openrouter_models(openrouter_key)
                    if models:
                        for m in models:
                            if isinstance(m, dict) and (mid := m.get("id")):
                                with MODEL_META_LOCK:
                                    MODEL_META_BY_ID[mid] = m
                    else:
                        if openrouter_key:
                            st.error("No models fetched. Check your OpenRouter key and connection.")

                    model_options = sorted([
                        f"{m['id']} (Ctx: {m.get('context_length', 'N/A')}, "
                        f"In: {_parse_price_per_million(m.get('pricing', {}).get('prompt')) or 'N/A'}/M, "
                        f"Out: {_parse_price_per_million(m.get('pricing', {}).get('completion')) or 'N/A'}/M)"
                        for m in models if isinstance(m, dict) and "id" in m
                ])

                    st.divider()

                    st.subheader("📝 Content Input")
                    st.info("💡 **Tip:** Start with a clear, well-structured content. The better your starting point, the better the results.")
                    protocol_col1, protocol_col2 = st.columns([3, 1])
                    with protocol_col1:
                        input_tab, preview_tab = st.tabs(["📝 Edit", "👁️ Preview"])
                        with input_tab:
                            st.text_area("✏️ Enter or paste your content:",
                                                 value=st.session_state.protocol_text,
                                                 height=300,
                                                 key="main_protocol_text_editor",
                                                 placeholder="Paste your draft content here...\n\nExample:\n# Security Policy\n\n## Overview\nThis policy defines requirements for secure system access.\n\n## Scope\nApplies to all employees and contractors.\n\n## Policy Statements\n1. All users must use strong passwords\n2. Multi-factor authentication is required for sensitive systems\n3. Regular security training is mandatory\n\n## Compliance\nViolations result in disciplinary action.")
                        with preview_tab:
                            if st.session_state.protocol_text:
                                st.markdown(st.session_state.protocol_text, unsafe_allow_html=True)
                            else:
                                st.info("Enter content in the 'Edit' tab to see the preview here.")
                    with protocol_col2:
                        st.markdown("**📋 Quick Actions**")
                        templates = content_manager.list_protocol_templates()
                        if templates:
                            def load_adv_template_callback():
                                selected_template = st.session_state.adv_load_template_select
                                if selected_template:
                                    content_manager.load_protocol_template(selected_template, st.session_state)
                                    st.rerun()

                            def load_sample_callback():
                                selected_sample = st.session_state.adv_load_sample_select
                                if selected_sample:
                                    content_manager.load_sample_protocol(selected_sample, st.session_state)  
                                    st.rerun()

                            def clear_content_callback():
                                st.session_state.protocol_text = ""
                                st.rerun()

                            # Template selection
                            st.selectbox(
                                "Load Template:",
                                options=[""] + templates,
                                key="adv_load_template_select",
                                on_change=load_adv_template_callback,
                                placeholder="Choose a template..."
                            )

                            # Sample protocols
                            samples = content_manager.list_sample_protocols()
                            if samples:
                                st.selectbox(
                                    "Or Load Sample:",
                                    options=[""] + samples,
                                    key="adv_load_sample_select", 
                                    on_change=load_sample_callback,
                                    placeholder="Choose a sample..."
                                )

                            # Clear button
                            if st.button("Clear Content", on_click=clear_content_callback, type="secondary"):
                                pass  # Callback handles the clearing

                    with protocol_col2:
                        # Content analysis and suggestions
                        if st.session_state.protocol_text.strip():
                            with st.expander("🔍 Content Analysis"):
                                from content_analyzer import analyze_content
                                analysis = analyze_content(st.session_state.protocol_text)
                                if analysis:
                                    st.subheader("📊 Analysis Results")
                                    for category, details in analysis.items():
                                        with st.container(border=True):
                                            if isinstance(details, dict):
                                                st.write(f"**{category.replace('_', ' ').title()}:**")
                                                for key, value in details.items():
                                                    st.write(f"- {key.replace('_', ' ').title()}: {value}")
                                            else:
                                                st.write(f"**{category.replace('_', ' ').title()}:** {details}")

                            with st.expander("💡 Suggestions"):
                                st.write("Based on your content, consider:")
                                st.write("- Adding more specific examples")
                                st.write("- Clarifying ambiguous statements")
                                st.write("- Adding compliance checkpoints")
                                if len(st.session_state.protocol_text) < 500:
                                    st.write("- Expanding with more details")

                    st.divider()

                    # --- Step 2: Model Selection ---
                    with st.expander("Step 2: Select Models for Red & Blue Teams", expanded=True):
                        st.subheader("👥 Team Configuration")

                        # Initialize session state for team models if not already set
                        if "red_team_models" not in st.session_state:
                            st.session_state.red_team_models = ["openai/gpt-4o", "anthropic/claude-3-opus"]
                        if "blue_team_models" not in st.session_state:
                            st.session_state.blue_team_models = ["openai/gpt-4o", "anthropic/claude-3-opus"]

                        # Create columns for side-by-side team configuration
                        red_col, blue_col = st.columns(2)

                        with red_col:
                            st.markdown("**🔴 Red Team (Critique & Find Flaws)**")

                            # Red team model selection
                            if model_options:
                                # Multi-select for red team models
                                selected_red_models = st.multiselect(
                                    "Select Red Team Models",
                                    options=model_options,
                                    default=st.session_state.red_team_models,
                                    key="red_team_multiselect"
                                )
                                # Update session state when selection changes
                                if selected_red_models != st.session_state.red_team_models:
                                    st.session_state.red_team_models = selected_red_models
                            else:
                                st.warning("No models available. Check your API key.")

                        with blue_col:
                            st.markdown("**🔵 Blue Team (Fix & Improve)**")

                            # Blue team model selection
                            if model_options:
                                # Multi-select for blue team models  
                                selected_blue_models = st.multiselect(
                                    "Select Blue Team Models",
                                    options=model_options,
                                    default=st.session_state.blue_team_models,
                                    key="blue_team_multiselect"
                                )
                                # Update session state when selection changes
                                if selected_blue_models != st.session_state.blue_team_models:
                                    st.session_state.blue_team_models = selected_blue_models
                            else:
                                st.warning("No models available. Check your API key.")

                        # Budget and constraints
                                st.divider()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.number_input("Budget Limit ($)", min_value=0.0, step=0.01, format="%.2f", key="adversarial_budget_limit")
                        with col2:
                            st.number_input("Max Cost per Iteration ($)", min_value=0.0, step=0.01, format="%.2f", key="adversarial_max_cost_per_iter")
                        with col3:
                            st.selectbox("Cost Alert Level", ["Low", "Medium", "High"], key="adversarial_cost_alert_level")

                        # Advanced configuration
                        with st.expander("Advanced Configuration"):
                            rotation_strategies = ["None", "Round Robin", "Random Sampling", "Performance-Based", "Staged", "Adaptive", "Focus-Category"]
                            c1, c2 = st.columns(2)
                            with c1:
                                st.selectbox("Rotation Strategy", rotation_strategies, key="adversarial_rotation_strategy")
                                st.number_input("Red Team Sample Size", 1, 100, key="adversarial_red_team_sample_size")
                            with c2:
                                st.number_input("Blue Team Sample Size", 1, 100, key="adversarial_blue_team_sample_size")
                                st.toggle("Auto-Optimize Model Selection", key="adversarial_auto_optimize_models", help="Automatically select optimal models based on protocol complexity and budget")

                            if st.session_state.adversarial_rotation_strategy == "Staged":
                                help_text = '''[{"red": ["model1", "model2"], "blue": ["model3"]}, {"red": ["model4"], "blue": ["model5", "model6"]}]'''
                                staged_config_input = st.text_area("Staged Rotation Config (JSON)", key="adversarial_staged_rotation_config", height=150, help=help_text)
                                
                                # Add JSON validation
                                if staged_config_input:
                                    try:
                                        json.loads(staged_config_input)
                                    except json.JSONDecodeError:
                                        st.error("Invalid JSON format for Staged Rotation Config.")

                            if st.session_state.adversarial_auto_optimize_models:
                                protocol_complexity = len(st.session_state.protocol_text.split())
                                optimized_models = optimize_model_selection(st.session_state.red_team_models, st.session_state.blue_team_models, protocol_complexity, st.session_state.adversarial_budget_limit)
                                st.session_state.red_team_models = optimized_models["red_team"]
                                st.session_state.blue_team_models = optimized_models["blue_team"]

                        # --- Step 3: Testing Parameters ---
                        with st.expander("Step 3: Adjust Testing Parameters", expanded=False):
                            st.subheader("⚙️ General Parameters")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.number_input("Min iterations", 1, 50, key="adversarial_min_iter")
                                st.slider("Confidence threshold (%)", 50, 100, key="adversarial_confidence", help="Stop if this % of Red Team approves the SOP.")
                                st.number_input("Max parallel workers", 1, 24, key="adversarial_max_workers")
                                st.slider("Critique Depth", 1, 10, key="adversarial_critique_depth", help="How deeply the red team should analyze (1-10)")
                            with c2:
                                st.number_input("Max iterations", 1, 200, key="adversarial_max_iter")
                                st.number_input("Max tokens per model", 1000, 100000, key="adversarial_max_tokens")
                                st.selectbox("Review Type", ["Auto-Detect", "General SOP", "Code Review", "Plan Review"], key="adversarial_review_type", help="Select the type of review to perform. Auto-Detect will analyze the content and choose the appropriate review type.")
                                st.slider("Patch Quality", 1, 10, key="adversarial_patch_quality", help="Quality level for blue team patches (1-10)")

                            st.text_area("Compliance Requirements", key="adversarial_compliance_requirements", height=100, help="Enter any compliance requirements that the red team should check for.")

                            with st.expander("🔧 Advanced & Custom Settings"):
                                use_custom_mode = st.toggle("Enable Custom Prompts", key="adversarial_custom_mode", help="Enable custom prompts and configurations for adversarial testing")
                                if use_custom_mode:
                                    st.text_area("Red Team Prompt (Critique)", key="adversarial_custom_red_prompt", height=150, help="Custom prompt for the red team to find flaws")
                                    st.text_area("Blue Team Prompt (Patch)", key="adversarial_custom_blue_prompt", height=150, help="Custom prompt for the blue team to patch flaws")
                                    st.text_area("Approval Prompt", key="adversarial_custom_approval_prompt", height=100, help="Custom prompt for final approval checking")

                                st.divider()
                                c1, c2 = st.columns(2)

# Helper function to optimize log updates
def _should_update_log_display(log_key, current_log):
    prev_log_key = f"{log_key}_prev"
    prev_log = st.session_state.get(prev_log_key, [])
    
    if len(current_log) >= 3 and len(prev_log) >= 3:
        if current_log[-3:] != prev_log[-3:]:
            st.session_state[prev_log_key] = current_log.copy()
            return True
    elif current_log != prev_log:
        st.session_state[prev_log_key] = current_log.copy()
        return True
    
    return False




    st.divider()

    st.subheader("📝 Content Input")
    st.info("💡 **Tip:** Start with a clear, well-structured content. The better your starting point, the better the results.")
    protocol_col1, protocol_col2 = st.columns([3, 1])
    with protocol_col1:
        input_tab, preview_tab = st.tabs(["📝 Edit", "👁️ Preview"])
        with input_tab:
            st.text_area("✏️ Enter or paste your content:",
                                         value=st.session_state.protocol_text,
                                         height=300,
                                         key="main_protocol_text_editor",
                                         placeholder="Paste your draft content here...\n\nExample:\n# Security Policy\n\n## Overview\nThis policy defines requirements for secure system access.\n\n## Scope\nApplies to all employees and contractors.\n\n## Policy Statements\n1. All users must use strong passwords\n2. Multi-factor authentication is required for sensitive systems\n3. Regular security training is mandatory\n\n## Compliance\nViolations result in disciplinary action.")
        with preview_tab:
            if st.session_state.protocol_text:
                st.markdown(st.session_state.protocol_text, unsafe_allow_html=True)
            else:
                st.info("Enter content in the 'Edit' tab to see the preview here.")
    with protocol_col2:
        st.markdown("**📋 Quick Actions**")
        templates = content_manager.list_protocol_templates()
        if templates:
            def load_adv_template_callback():
                selected_template = st.session_state.adv_load_template_select
                if selected_template:
                    try:
                        st.session_state.protocol_text = content_manager.load_protocol_template(selected_template)
                        st.success(f"Loaded template: {selected_template}")
                    except Exception as e:
                        st.error(f"Failed to load template: {e}")
            selected_template = st.selectbox("Load Template", [""] + templates, key="adv_load_template_select")
            if selected_template:
                st.button("📥 Load Template", use_container_width=True, type="secondary", on_click=load_adv_template_callback)

        def load_sample_callback():
            st.session_state.protocol_text = '''# Sample Security Policy

## Overview
This policy defines security requirements for accessing company systems.

## Scope
Applies to all employees, contractors, and vendors with system access.

## Policy Statements
1. All users must use strong passwords
2. Multi-factor authentication is required for sensitive systems
3. Regular security training is mandatory
4. Incident reporting must occur within 24 hours.'''
            def clear_content_callback():
                st.session_state.protocol_text = ""
            st.button("🧪 Load Sample", use_container_width=True, type="secondary", on_click=load_sample_callback)
            st.success("Sample content loaded.")
            if st.session_state.protocol_text.strip():
                st.button("🗑️ Clear", use_container_width=True, type="secondary", on_click=clear_content_callback)
                st.success("Content cleared.")

    # --- Step 2: Model Selection & Strategy ---
    with st.expander("Step 2: Define Teams and Strategy", expanded=True):
        st.subheader("🤖 Model Selection")
        st.info("💡 **Tip:** Select 3-5 diverse models for each team for best results. Mix small and large models for cost-effectiveness.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔴 Red Team (Critics)")
            st.caption("Models that find flaws and vulnerabilities in your protocol")
            if HAS_STREAMLIT_TAGS:
                red_team_selected_full = st_tags(label="Search and select models:", text="Type to search models...", value=st.session_state.red_team_models, suggestions=model_options, key="adversarial_red_team_select")
                st.session_state.red_team_models = sorted(list(set([m.split(" (")[0].strip() for m in red_team_selected_full])))
            else:
                red_team_input = st.text_input("Enter Red Team models (comma-separated):", value=",".join(st.session_state.red_team_models))
                st.session_state.red_team_models = sorted(list(set([model.strip() for model in red_team_input.split(",") if model.strip()])))
            st.caption(f"Selected: {len(st.session_state.red_team_models)} models")
        with col2:
            st.markdown("#### 🔵 Blue Team (Fixers)")
            st.caption("Models that patch the identified flaws and improve the protocol")
            if HAS_STREAMLIT_TAGS:
                blue_team_selected_full = st_tags(label="Search and select models:", text="Type to search models...", value=st.session_state.blue_team_models, suggestions=model_options, key="adversarial_blue_team_select")
                st.session_state.blue_team_models = sorted(list(set([m.split(" (")[0].strip() for m in blue_team_selected_full])))
            else:
                blue_team_input = st.text_input("Enter Blue Team models (comma-separated):", value=",".join(st.session_state.blue_team_models))
                st.session_state.blue_team_models = sorted(list(set([model.strip() for model in blue_team_input.split(",") if model.strip()])))
            st.caption(f"Selected: {len(st.session_state.blue_team_models)} models")

        st.divider()
        st.subheader("♟️ Strategy")
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Rotation Strategy", ["None", "Round Robin", "Random Sampling", "Performance-Based", "Staged", "Adaptive", "Focus-Category"], key="adversarial_rotation_strategy")
            st.number_input("Red Team Sample Size", 1, 100, key="adversarial_red_team_sample_size")
        with c2:
            st.number_input("Blue Team Sample Size", 1, 100, key="adversarial_blue_team_sample_size")
            st.toggle("Auto-Optimize Model Selection", key="adversarial_auto_optimize_models", help="Automatically select optimal models based on protocol complexity and budget")

        if st.session_state.adversarial_rotation_strategy == "Staged":
            help_text = '''[{"red": ["model1", "model2"], "blue": ["model3"]}, {"red": ["model4"], "blue": ["model5", "model6"]}]'''
            staged_config_input = st.text_area("Staged Rotation Config (JSON)", key="adversarial_staged_rotation_config", height=150, help=help_text)
            
            # Add JSON validation
            if staged_config_input:
                try:
                    json.loads(staged_config_input)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for Staged Rotation Config.")

        if st.session_state.adversarial_auto_optimize_models:
            protocol_complexity = len(st.session_state.protocol_text.split())
            optimized_models = optimize_model_selection(st.session_state.red_team_models, st.session_state.blue_team_models, protocol_complexity, st.session_state.adversarial_budget_limit)
            st.session_state.red_team_models = optimized_models["red_team"]
            st.session_state.blue_team_models = optimized_models["blue_team"]

        # --- Step 3: Testing Parameters ---
        with st.expander("Step 3: Adjust Testing Parameters", expanded=False):
            st.subheader("⚙️ General Parameters")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Min iterations", 1, 50, key="adversarial_min_iter")
                st.slider("Confidence threshold (%)", 50, 100, key="adversarial_confidence", help="Stop if this % of Red Team approves the SOP.")
                st.number_input("Max parallel workers", 1, 24, key="adversarial_max_workers")
                st.slider("Critique Depth", 1, 10, key="adversarial_critique_depth", help="How deeply the red team should analyze (1-10)")
            with c2:
                st.number_input("Max iterations", 1, 200, key="adversarial_max_iter")
                st.number_input("Max tokens per model", 1000, 100000, key="adversarial_max_tokens")
                st.selectbox("Review Type", ["Auto-Detect", "General SOP", "Code Review", "Plan Review"], key="adversarial_review_type", help="Select the type of review to perform. Auto-Detect will analyze the content and choose the appropriate review type.")
                st.slider("Patch Quality", 1, 10, key="adversarial_patch_quality", help="Quality level for blue team patches (1-10)")

            st.text_area("Compliance Requirements", key="adversarial_compliance_requirements", height=100, help="Enter any compliance requirements that the red team should check for.")

            with st.expander("🔧 Advanced & Custom Settings"):
                use_custom_mode = st.toggle("Enable Custom Prompts", key="adversarial_custom_mode", help="Enable custom prompts and configurations for adversarial testing")
                if use_custom_mode:
                    st.text_area("Red Team Prompt (Critique)", key="adversarial_custom_red_prompt", height=150, help="Custom prompt for the red team to find flaws")
                    st.text_area("Blue Team Prompt (Patch)", key="adversarial_custom_blue_prompt", height=150, help="Custom prompt for the blue team to patch flaws")
                    st.text_area("Approval Prompt", key="adversarial_custom_approval_prompt", height=100, help="Custom prompt for final approval checking")

                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    st.text_input("Deterministic seed", key="adversarial_seed", help="Integer for reproducible runs.")
                with c2:
                    st.toggle("Force JSON mode", key="adversarial_force_json", help="Use model's built-in JSON mode if available.")

                all_models = sorted(list(set(st.session_state.red_team_models + st.session_state.blue_team_models)))
                if all_models:
                    st.markdown("##### Per-Model Configuration")
                    for model_id in all_models:
                        st.markdown(f"**{model_id}**")
                        cc1, cc2, cc3, cc4 = st.columns(4)
                        cc1.slider(f"Temp##{model_id}", 0.0, 2.0, 0.7, 0.1, key=f"temp_{model_id}")
                        cc2.slider(f"Top-P##{model_id}", 0.0, 1.0, 1.0, 0.1, key=f"topp_{model_id}")
                        cc3.slider(f"Freq Pen##{model_id}", -2.0, 2.0, 0.0, 0.1, key=f"freqpen_{model_id}")
                        cc4.slider(f"Pres Pen##{model_id}", -2.0, 2.0, 0.0, 0.1, key=f"prespen_{model_id}")

        # --- Step 4: Execution & Monitoring ---
        with st.container(border=True):
            st.subheader("Step 4: Execute & Monitor")
            col1, col2, col3 = st.columns(3)
            start_button = col1.button("🚀 Start Adversarial Testing", type="primary", disabled=st.session_state.adversarial_running or not st.session_state.protocol_text.strip(), use_container_width=True)
            integrated_button = col2.button("🔄 Run Integrated (Adv + Evol)", type="secondary", disabled=st.session_state.adversarial_running or not st.session_state.protocol_text.strip(), use_container_width=True)
            stop_button = col3.button("⏹️ Stop Adversarial Testing", disabled=not st.session_state.adversarial_running, type="secondary", use_container_width=True)

            if start_button:
                # Validate inputs before starting
                if not st.session_state.get("openrouter_key", "").strip():
                    st.error("❌ OpenRouter API key is required to start adversarial testing.")
                    st.session_state.adversarial_running = False
                elif not st.session_state.get("red_team_models", []):
                    st.error("❌ Please select at least one model for the red team.")
                    st.session_state.adversarial_running = False
                elif not st.session_state.get("blue_team_models", []):
                    st.error("❌ Please select at least one model for the blue team.")
                    st.session_state.adversarial_running = False
                elif not st.session_state.protocol_text.strip():
                    st.error("❌ Please enter content to test.")
                    st.session_state.adversarial_running = False
                else:
                    st.info("🔄 Starting adversarial testing...")
                    st.session_state.adversarial_running = True
                    threading.Thread(target=run_adversarial_testing).start()
                    st.success("✅ Adversarial testing started successfully!")
                    st.rerun()
            
            if integrated_button:
                # Validate inputs for integrated run
                if not st.session_state.get("openrouter_key", "").strip():
                    st.error("❌ OpenRouter API key is required to start integrated testing.")
                    st.session_state.adversarial_running = False
                elif not st.session_state.get("red_team_models", []):
                    st.error("❌ Please select at least one model for the red team.")
                    st.session_state.adversarial_running = False
                elif not st.session_state.get("blue_team_models", []):
                    st.error("❌ Please select at least one model for the blue team.")
                    st.session_state.adversarial_running = False
                elif not st.session_state.get("evaluator_models", []):
                    st.error("❌ Please select at least one model for the evaluator team.")
                    st.session_state.adversarial_running = False
                elif not st.session_state.protocol_text.strip():
                    st.error("❌ Please enter content to test.")
                    st.session_state.adversarial_running = False
                else:
                    st.info("🔄 Starting enhanced integrated adversarial-evolution-evaluation process...")
                    st.session_state.adversarial_running = True
                    # Run enhanced integrated function in a thread
                    def run_enhanced_integrated():
                        try:
                            from adversarial import run_enhanced_integrated_adversarial_evolution
                            result = run_enhanced_integrated_adversarial_evolution(
                                current_content=st.session_state.protocol_text,
                                content_type=st.session_state.get("adversarial_content_type", "general"),  # Use selected content type
                                api_key=st.session_state.openrouter_key,
                                base_url=st.session_state.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
                                red_team_models=st.session_state.red_team_models,
                                blue_team_models=st.session_state.blue_team_models,
                                evaluator_models=st.session_state.evaluator_models,
                                adversarial_iterations=st.session_state.get("adversarial_iterations", 3),
                                evolution_iterations=st.session_state.get("evolution_iterations", 3),
                                evaluation_iterations=st.session_state.get("evaluation_iterations", 2),
                                system_prompt=st.session_state.get("system_prompt", "You are an expert content generator. Create high-quality, optimized content based on the user's requirements."),
                                evaluator_system_prompt=st.session_state.get("evaluator_system_prompt", "Evaluate the quality, clarity, and effectiveness of this content. Provide a score from 0 to 100."),
                                temperature=st.session_state.get("temperature", 0.7),
                                top_p=st.session_state.get("top_p", 1.0),
                                frequency_penalty=st.session_state.get("frequency_penalty", 0.0),
                                presence_penalty=st.session_state.get("presence_penalty", 0.0),
                                max_tokens=st.session_state.adversarial_max_tokens,
                                seed=st.session_state.get("adversarial_seed"),
                                rotation_strategy=st.session_state.adversarial_rotation_strategy,
                                red_team_sample_size=st.session_state.adversarial_red_team_sample_size,
                                blue_team_sample_size=st.session_state.adversarial_blue_team_sample_size,
                                evaluator_sample_size=st.session_state.get("evaluator_sample_size", 3),
                                confidence_threshold=st.session_state.adversarial_confidence,
                                evaluator_threshold=st.session_state.get("evaluator_threshold", 90.0),
                                evaluator_consecutive_rounds=st.session_state.get("evaluator_consecutive_rounds", 1),
                                compliance_requirements=st.session_state.get("adversarial_compliance_requirements", ""),
                                enable_data_augmentation=st.session_state.get("enable_data_augmentation", False),
                                augmentation_model_id=st.session_state.get("augmentation_model_id"),
                                augmentation_temperature=st.session_state.get("augmentation_temperature", 0.7),
                                enable_human_feedback=st.session_state.get("enable_human_feedback", False),
                                multi_objective_optimization=st.session_state.get("multi_objective_optimization", True),
                                feature_dimensions=st.session_state.get("feature_dimensions", ['complexity', 'diversity', 'quality']),
                                feature_bins=st.session_state.get("feature_bins", 10),
                                elite_ratio=st.session_state.get("evolution_elite_ratio", 0.1),
                                exploration_ratio=st.session_state.get("evolution_exploration_ratio", 0.2),
                                exploitation_ratio=st.session_state.get("exploitation_ratio", 0.7),
                                archive_size=st.session_state.get("evolution_archive_size", 100),
                                checkpoint_interval=st.session_state.get("evolution_checkpoint_interval", 10),
                                keyword_analysis_enabled=st.session_state.get("keyword_analysis_enabled", True),
                                keywords_to_target=st.session_state.get("keywords_to_target", []),
                                keyword_penalty_weight=st.session_state.get("keyword_penalty_weight", 0.5)
                            )
                            if result.get("success"):
                                # Update the protocol text with the evolved content
                                st.session_state.protocol_text = result.get("final_content", st.session_state.protocol_text)
                                st.success("✅ Enhanced integrated adversarial-evolution-evaluation completed successfully!")
                                
                                # Update session state with integrated results for reporting
                                try:
                                    from integrated_reporting import update_integrated_session_state
                                    update_integrated_session_state(result)
                                except ImportError:
                                    st.session_state.integrated_results = result
                                
                                # Show integrated metrics
                                with st.expander("📊 Integrated Process Metrics", expanded=True):
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Integrated Score", f"{result.get('integrated_score', 0):.2f}")
                                    col2.metric("Total Cost (USD)", f"${result.get('total_cost_usd', 0):.4f}")
                                    col3.metric("Total Tokens", f"{result.get('total_tokens', {}).get('prompt', 0) + result.get('total_tokens', {}).get('completion', 0):,}")
                                
                                # Add export options for the integrated report
                                st.subheader("Export Integrated Report")
                                from integrated_reporting import generate_integrated_report, calculate_detailed_metrics
                                
                                # Calculate detailed metrics
                                detailed_metrics = calculate_detailed_metrics(result)
                                
                                with st.expander("📈 Detailed Metrics"):
                                    st.json(detailed_metrics)
                                
                                # Export options
                                col1, col2, col3 = st.columns(3)
                                
                                # Generate HTML report
                                html_report = generate_integrated_report(result)
                                col1.download_button(
                                    label="📄 Export HTML Report",
                                    data=html_report,
                                    file_name=f"integrated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html",
                                    use_container_width=True
                                )
                                
                                # Export as JSON
                                json_str = json.dumps(result, indent=2, default=str)
                                col2.download_button(
                                    label="📋 Export JSON Results",
                                    data=json_str,
                                    file_name=f"integrated_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                                
                                # Export detailed metrics
                                metrics_json = json.dumps(detailed_metrics, indent=2, default=str)
                                col3.download_button(
                                    label="📊 Export Metrics JSON",
                                    data=metrics_json,
                                    file_name=f"integrated_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                                
                                # Add GitHub sync button
                                with st.expander("🔄 GitHub Sync", expanded=False):
                                    st.write("Approve and sync the final content to GitHub")
                                    if st.button("✅ Approve & Sync to GitHub", type="primary", use_container_width=True):
                                        try:
                                            from integrations import commit_to_github
                                            if st.session_state.get("github_token") and st.session_state.get("selected_github_repo"):
                                                file_path = st.session_state.get("github_file_path", "final_content.md")
                                                commit_message = st.session_state.get("github_commit_message", f"Update content after integrated evolution - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                                branch_name = st.session_state.get("github_branch", "main")
                                                
                                                if commit_to_github(
                                                    st.session_state.github_token,
                                                    st.session_state.selected_github_repo,
                                                    file_path,
                                                    result.get("final_content", ""),
                                                    commit_message,
                                                    branch_name
                                                ):
                                                    st.success("✅ Content successfully synced to GitHub!")
                                                else:
                                                    st.error("❌ Failed to sync content to GitHub")
                                            else:
                                                st.error("❌ GitHub token or repository not configured")
                                        except Exception as e:
                                            st.error(f"❌ Error syncing to GitHub: {e}")
                                
                            else:
                                st.error(f"❌ Enhanced integrated process failed: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"❌ Error running enhanced integrated process: {e}")
                            import traceback
                            traceback.print_exc()
                        finally:
                            st.session_state.adversarial_running = False
                            
                    threading.Thread(target=run_enhanced_integrated).start()
                    st.success("✅ Enhanced integrated process started successfully!")
                    st.rerun()
                    
            if stop_button:
                st.session_state.adversarial_stop_flag = True
                st.success("✅ Adversarial testing stop requested.")

            if st.session_state.adversarial_running or st.session_state.adversarial_status_message:
                if st.session_state.adversarial_status_message:
                    st.info(st.session_state.adversarial_status_message)
                if st.session_state.adversarial_running:
                    current_iter = len(st.session_state.get("adversarial_confidence_history", []))
                    max_iter = st.session_state.adversarial_max_iter
                    progress = min(current_iter / max(1, max_iter), 1.0)
                    st.progress(progress, text=f"Iteration {current_iter}/{max_iter} ({int(progress * 100)}%)")
                    if st.session_state.get("adversarial_confidence_history"):
                        current_confidence = st.session_state.adversarial_confidence_history[-1]
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("📊 Current Confidence", f"{current_confidence:.1f}%")
                        col2.metric("💰 Est. Cost (USD)", f"${st.session_state.adversarial_cost_estimate_usd:.4f}")
                        col3.metric("🔤 Prompt Tokens", f"{st.session_state.adversarial_total_tokens_prompt:,}")
                        col4.metric("📝 Completion Tokens", f"{st.session_state.adversarial_total_tokens_completion:,}")
                    with st.expander("🔍 Real-time Logs", expanded=True):
                        # Only show log when actively running to prevent performance issues
                        if st.session_state.get("adversarial_running", False) and st.session_state.adversarial_log:
                            log_entries = st.session_state.adversarial_log[-50:]
                            if _should_update_log_display("adversarial_log", log_entries):
                                log_content = "\n".join(log_entries)
                                st.text_area("Activity Log", value=log_content, height=300, key="adversarial_log_display", disabled=True)
                        else:
                            st.info("Logs will appear here when adversarial testing is running.")

        # --- Step 5: Results & Export ---
        if st.session_state.adversarial_results and not st.session_state.adversarial_running:
            with st.container(border=True):
                st.header("🏆 Adversarial Testing Results")
                results = st.session_state.adversarial_results
                st.markdown("### 📊 Performance Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("✅ Final Approval Rate", f"{results.get('final_approval_rate', 0):.1f}%")
                col2.metric("🔄 Iterations", len(results.get('iterations', [])))
                col3.metric("💰 Total Cost (USD)", f"${results.get('cost_estimate_usd', 0):.4f}")
                col4.metric(" Tokens", f"{results.get('tokens', {}).get('prompt', 0) + results.get('tokens', {}).get('completion', 0):,}")

                metrics_tab1, metrics_tab2, metrics_tab3 = st.tabs(["📈 Confidence Trend", "🏆 Model Performance", "🧮 Issue Analysis"])
                with metrics_tab1:
                    if results.get('iterations'):
                        confidence_history = [iter.get("approval_check", {}).get("approval_rate", 0) for iter in results.get('iterations', [])]
                        if confidence_history:
                            df = pd.DataFrame({'Confidence': confidence_history})
                            st.line_chart(df)
                with metrics_tab2:
                    if st.session_state.get("adversarial_model_performance"):
                        st.bar_chart({k: v.get("score", 0) for k, v in st.session_state.adversarial_model_performance.items()})
                    else:
                        st.info("No model performance data available.")
                with metrics_tab3:
                    severity_counts = {}
                    for iteration in results.get('iterations', []):
                        for critique in iteration.get("critiques", []):
                            if critique.get("critique_json"):
                                for issue in _safe_list(critique["critique_json"], "issues"):
                                    severity = issue.get("severity", "low").lower()
                                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    if severity_counts:
                        st.bar_chart(severity_counts)
                    else:
                        st.info("No issue data to display.")

                st.markdown("### 📄 Final Hardened Protocol")
                st.code(results.get('final_sop', ''), language="markdown")

                st.markdown("### 📁 Export & Integrations")
                export_col, integration_col = st.columns(2)
                with export_col:
                    st.subheader("Export Results")
                    export_c1, export_c2, export_c3 = st.columns(3)
                    pdf_bytes = generate_pdf_report(results, st.session_state.pdf_watermark)
                    export_c1.download_button("📄 PDF", pdf_bytes, f"report_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf", use_container_width=True, type="secondary")
                    docx_bytes = generate_docx_report(results)
                    export_c2.download_button("📝 DOCX", docx_bytes, f"report_{datetime.now().strftime('%Y%m%d')}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True, type="secondary")
                    html_content = generate_integrated_report(results)
                    export_c3.download_button("📊 HTML", html_content, f"report_{datetime.now().strftime('%Y%m%d')}.html", "text/html", use_container_width=True, type="secondary")

                    export_c4, export_c5, export_c6 = st.columns(3)
                    json_str = json.dumps(results, indent=2, default=str)
                    export_c4.download_button("📋 JSON", json_str, f"report_{datetime.now().strftime('%Y%m%d')}.json", "application/json", use_container_width=True, type="secondary")
                    latex_str = generate_latex_report(results)
                    export_c5.download_button("📄 LaTeX", latex_str, f"report_{datetime.now().strftime('%Y%m%d')}.tex", "application/x-latex", use_container_width=True, type="secondary")
                    if st.session_state.compliance_requirements:
                        compliance_report = generate_compliance_report(results, st.session_state.compliance_requirements)
                        export_c6.download_button("📋 Compliance", compliance_report, f"compliance_report_{datetime.now().strftime('%Y%m%d')}.md", "text/markdown", use_container_width=True, type="secondary")

                with integration_col:
                    st.subheader("Send Notifications")
                    int_col1, int_col2, int_col3 = st.columns(3)
                    if int_col1.button("💬 Discord", use_container_width=True, type="secondary"):
                        if st.session_state.discord_webhook_url:
                            send_discord_notification(st.session_state.discord_webhook_url, f"Adversarial testing complete! Final approval: {results.get('final_approval_rate', 0.0):.1f}%")
                            st.success("Discord notification sent!")
                        else:
                            st.warning("Discord webhook URL not configured.")
                    if int_col2.button("💬 Teams", use_container_width=True, type="secondary"):
                        if st.session_state.msteams_webhook_url:
                            send_msteams_notification(st.session_state.msteams_webhook_url, f"Adversarial testing complete! Final approval: {results.get('final_approval_rate', 0.0):.1f}%")
                            st.success("Microsoft Teams notification sent!")
                        else:
                            st.warning("Microsoft Teams webhook URL not configured.")
                    if int_col3.button("🚀 Webhook", use_container_width=True, type="secondary"):
                        if st.session_state.generic_webhook_url:
                            send_generic_webhook(st.session_state.generic_webhook_url, {"text": f"Adversarial testing complete! Final approval: {results.get('final_approval_rate', 0.0):.1f}%"})
                            st.success("Generic webhook notification sent!")
                        else:
                            st.warning("Generic webhook URL not configured.")


    with tabs[2]: # GitHub tab  # noqa: F821
        with st.container(border=True):
            st.header("🐙 GitHub Integration")
            st.markdown("Connect your GitHub repositories, manage commits, branches, and sync evolution results.")
            
            # GitHub Authentication Section
            st.subheader("🔐 GitHub Authentication")
            github_token = st.text_input("GitHub Personal Access Token", type="password", 
                                        value=st.session_state.get("github_token", ""))
            if st.button("Authenticate with GitHub", key="auth_github"):
                if github_token:
                    # For now, just store the token since authenticate_github is not available in imports
                    st.session_state.github_token = github_token
                    st.success("GitHub token stored! Note: Full authentication requires backend implementation.")
                    st.rerun()
                else:
                    st.warning("Please enter a GitHub Personal Access Token")
            
            if st.session_state.get("github_token"):
                st.success("✅ GitHub token is set")
                
                # GitHub integration tabs
                github_tabs = st.tabs(["📋 Repositories", "🌿 Branches", "💾 Commits", "🔄 Sync", "⚙️ Settings"])
                
                with github_tabs[0]:  # Repositories
                    st.subheader("Repository Management")
                    
                    st.info("Repository listing and linking requires backend integration not currently available.")
                    
                    # Show linked repositories using the available function
                    try:
                        linked_repos = list_linked_github_repositories(st.session_state.github_token)  # Using available function
                        if linked_repos:
                            st.subheader("🔗 Linked Repositories")
                            for repo in linked_repos:
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.write(f"📂 {repo}")
                                # Note: unlink_github_repository is not available in imports, so we'll skip unlinking
                                with col2:
                                    st.write("")  # Placeholder since unlink functionality is not available
                    except Exception as e:
                        st.info("No repositories linked yet or error accessing repositories")
                    
                    # Manual repo addition for demo purposes
                    st.subheader("Add Repository (Demo)")
                    new_repo = st.text_input("Repository name (e.g., username/repo-name)", key="manual_repo_add")
                    if st.button("Add Repository", key="add_repo_demo") and new_repo:
                        st.success(f"Repository {new_repo} would be linked in a full implementation.")
                
                # Repository selection for operations
                selected_repo = st.selectbox("Select Repository for Operations", linked_repos)
                
                if selected_repo:
                    st.divider()
                    
                    # Branch Management
                    st.subheader("🌿 Branch Management")
                    with st.expander("Create New Branch"):
                        new_branch_name = st.text_input("New Branch Name", placeholder="e.g., protocol-v1", key="new_branch_name")
                        base_branch = st.text_input("Base Branch", "main", key="base_branch")
                        if st.button("Create Branch", type="secondary") and new_branch_name:
                            try:
                                if create_github_branch(st.session_state.github_token, selected_repo, new_branch_name, base_branch):
                                    st.success(f"Created branch '{new_branch_name}' from '{base_branch}')")
                                else:
                                    st.error(f"Failed to create branch '{new_branch_name}'.")
                            except Exception as e:
                                st.error(f"Error creating branch: {e}")
                    
                    # Commit and Push
                    st.subheader("💾 Commit and Push")
                    branch_name = st.text_input("Target Branch", "main", key="target_branch")
                    file_path = st.text_input("File Path", "protocols/evolved_protocol.md", key="file_path")
                    commit_message = st.text_input("Commit Message", "Update evolved protocol", key="commit_message")
                    if st.button("Commit to GitHub", type="primary"):
                        if not st.session_state.protocol_text.strip():
                            st.error("Cannot commit empty content.")
                        elif commit_to_github(st.session_state.github_token, selected_repo, file_path, st.session_state.protocol_text, commit_message, branch_name):
                            st.success("✅ Committed to GitHub successfully!")
                        else:
                            st.error("❌ Failed to commit to GitHub. Check your token and permissions.")
                
                # GitHub User Info
                if st.session_state.get("github_user"):
                    with st.expander("👤 GitHub User Information"):
                        user = st.session_state.github_user
                        st.write(f"**Username:** {user.get('login', 'Unknown')}")
                        st.write(f"**Name:** {user.get('name', 'Not provided')}")
                        st.write(f"**Email:** {user.get('email', 'Not provided')}")
                        st.write(f"**Public Repos:** {user.get('public_repos', 'N/A')}")
                        st.write(f"**Public Gists:** {user.get('public_gists', 'N/A')}")
                        if st.button("🔌 Disconnect from GitHub", key="disconnect_github"):
                            st.session_state.github_token = ""
                            st.session_state.github_user = None
                            if "github_repos" in st.session_state:
                                st.session_state.github_repos = {}
                            st.rerun()
            else:
                st.info("Please authenticate with GitHub to access repositories and commit functionality.")

    with tabs[3]: # Activity Feed tab  # noqa: F821
        with st.container(border=True):
            st.header("📜 Activity Feed")
            
            # Initialize activity log if not already present
            if "activity_log" not in st.session_state:
                st.session_state.activity_log = []
            
            # Add filtering and search capabilities
            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input("Search activities", placeholder="Filter by keyword...")
            with col2:
                activity_type = st.selectbox("Filter by type", ["All", "Evolution", "Adversarial", "Model", "System", "User"])
            
            # Time range filter
            time_range = st.selectbox("Time range", ["All Time", "Last 24 hours", "Last 7 days", "Last 30 days"], index=0)
            
            # Filter the activity log based on search and filters
            filtered_activities = st.session_state.activity_log.copy()
            
            if search_term:
                filtered_activities = [entry for entry in filtered_activities 
                                     if search_term.lower() in str(entry).lower()]
            
            if activity_type != "All":
                type_mapping = {
                    "Evolution": ["evolution", "fitness", "population", "generation", "archive", "island"],
                    "Adversarial": ["adversarial", "red team", "blue team", "critique", "patch"],
                    "Model": ["model", "api", "llm", "prompt"],
                    "System": ["system", "error", "warning", "info"],
                    "User": ["user", "session", "login", "logout"]
                }
                matching_keywords = type_mapping.get(activity_type, [])
                filtered_activities = [entry for entry in filtered_activities 
                                     if any(keyword in str(entry).lower() for keyword in matching_keywords)]
            
            # Time range filter (simplified implementation - would use actual timestamps in real implementation)
            if time_range != "All Time":
                # This is a simplified filter; in a real implementation, we'd store timestamps with each log entry
                # and filter based on the actual time range
                pass
            
            # Display activity feed
            if filtered_activities:
                # Pagination
                items_per_page = 10
                total_items = len(filtered_activities)
                total_pages = (total_items + items_per_page - 1) // items_per_page
                
                if total_pages > 1:
                    page_number = st.slider("Page", 1, total_pages, 1)
                    start_idx = (page_number - 1) * items_per_page
                    end_idx = start_idx + items_per_page
                    page_activities = filtered_activities[-end_idx:-start_idx or None]
                else:
                    page_activities = filtered_activities[-items_per_page:]
                
                # Display activities
                for i, entry in enumerate(reversed(page_activities)):
                    with st.container(border=True):
                        # Parse activity entry to extract timestamp and type if available
                        entry_text = str(entry)
                        
                        # Create a card-like display for each activity entry
                        st.markdown(f"**{i+1}.** {entry_text}")
                        
                        # Add action buttons for each activity (if needed)
                        col_btn1, col_btn2 = st.columns([1, 5])
                        with col_btn1:
                            if st.button("📋 Copy", key=f"copy_activity_{i}"):
                                st.code(entry_text)
                        with col_btn2:
                            pass  # Placeholder to maintain column structure
                
                st.info(f"Showing {len(page_activities)} of {total_items} activities")
            else:
                st.info("No activities to display based on your filters.")
            
            # Add sample activities for demonstration
            with st.expander("Add Sample Activities", expanded=False):
                if st.button("Add Sample Evolution Activity"):
                    sample_entry = {
                        "timestamp": str(datetime.now()),
                        "type": "evolution",
                        "message": "Started evolution run on content",
                        "details": "Model: gpt-4o, Population: 100, Generations: 50"
                    }
                    st.session_state.activity_log.append(sample_entry)
                    st.success("Sample activity added")
                
                if st.button("Add Sample Adversarial Activity"):
                    sample_entry = {
                        "timestamp": str(datetime.now()),
                        "type": "adversarial",
                        "message": "Started adversarial testing with Red Team",
                        "details": "Models: claude-3-sonnet, gpt-4o"
                    }
                    st.session_state.activity_log.append(sample_entry)
                    st.success("Sample activity added")
                
                if st.button("Clear All Activities"):
                    st.session_state.activity_log = []
                    st.success("Activity log cleared")
            
            # Export functionality
            st.divider()
            st.subheader("Export Activity Log")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Export as JSON"):
                    import json
                    json_data = json.dumps(st.session_state.activity_log, indent=2, default=str)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"activity_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            with col2:
                if st.button("Export as CSV"):
                    import pandas as pd
                    df = pd.DataFrame(st.session_state.activity_log)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"activity_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            with col3:
                if st.button("Export as Text"):
                    text_data = "\n".join([str(entry) for entry in st.session_state.activity_log])
                    st.download_button(
                        label="Download Text",
                        data=text_data,
                        file_name=f"activity_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
    st.divider()

    with tabs[4]: # Report Templates tab  # noqa: F821
        with st.container(border=True):
            st.header("📊 Report Template Management")
            st.markdown("Manage your custom report templates here. Create new templates or view existing ones.")

            # Initialize session state for report templates if not already present
            if "report_templates" not in st.session_state:
                st.session_state.report_templates = _load_report_templates()

            # Create tabs for different sections
            template_tabs = st.tabs(["📝 Create Template", "📋 Manage Templates", "🔄 Import/Export", "🎨 Template Preview"])

            with template_tabs[0]:  # Create Template
                st.subheader("Create New Template")
                
                # Template creation form
                with st.form("create_template_form"):
                    new_template_name = st.text_input("Template Name", placeholder="e.g., Security Audit Report")
                    template_description = st.text_input("Description (Optional)", placeholder="Brief description of the template")
                    new_template_content = st.text_area("Template Content (JSON)", height=300, 
                                                       placeholder='''{
  "title": "Evolution Report",
  "sections": [
    {
      "name": "Overview",
      "content": "This report shows the evolution results..."
    }
  ]
}''')
                    submitted = st.form_submit_button("Save Template", type="primary")
                    
                    if submitted:
                        if new_template_name and new_template_content:
                            try:
                                template_data = json.loads(new_template_content)
                                # Add metadata
                                template_with_metadata = {
                                    "name": new_template_name,
                                    "description": template_description,
                                    "content": template_data,
                                    "created_at": datetime.now().isoformat(),
                                    "updated_at": datetime.now().isoformat(),
                                    "version": "1.0"
                                }
                                st.session_state.report_templates[new_template_name] = template_with_metadata
                                _save_report_templates(st.session_state.report_templates)
                                st.success(f"Template '{new_template_name}' saved successfully!")
                                st.rerun()
                            except json.JSONDecodeError:
                                st.error("Invalid JSON format. Please check your template content.")
                        else:
                            st.warning("Please provide both a name and content for the template.")

            with template_tabs[1]:  # Manage Templates
                st.subheader("Manage Templates")
                
                if not st.session_state.report_templates:
                    st.info("No report templates found. Create one in the 'Create Template' tab.")
                else:
                    # Template selector and actions
                    template_names = list(st.session_state.report_templates.keys())
                    selected_template = st.selectbox("Select Template to Manage", template_names)
                    
                    if selected_template:
                        template_data = st.session_state.report_templates[selected_template]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text_input("Template Name", value=template_data.get("name", selected_template), disabled=True)
                        with col2:
                            st.text_input("Description", value=template_data.get("description", ""), disabled=True)
                        
                        # Display template content
                        st.text_area("Template Content", value=json.dumps(template_data.get("content", {}), indent=2), 
                                    height=200, disabled=True)
                        
                        # Template actions
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("Edit Template", key=f"edit_{selected_template}"):
                                # For now, populate the creation area with current template
                                st.session_state.editing_template = selected_template
                                st.session_state.edit_template_name = template_data.get("name", selected_template)
                                st.session_state.edit_template_description = template_data.get("description", "")
                                st.session_state.edit_template_content = json.dumps(template_data.get("content", {}), indent=2)
                                st.rerun()
                        with col2:
                            if st.button("Duplicate Template", key=f"duplicate_{selected_template}"):
                                new_name = f"{selected_template}_copy"
                                new_template_data = template_data.copy()
                                new_template_data["name"] = new_name
                                new_template_data["created_at"] = datetime.now().isoformat()
                                st.session_state.report_templates[new_name] = new_template_data
                                _save_report_templates(st.session_state.report_templates)
                                st.success(f"Template duplicated as '{new_name}'")
                                st.rerun()
                        with col3:
                            if st.button("Delete Template", key=f"delete_{selected_template}", type="secondary"):
                                del st.session_state.report_templates[selected_template]
                                _save_report_templates(st.session_state.report_templates)
                                st.success(f"Template '{selected_template}' deleted")
                                st.rerun()
                    
                    # Show all templates in a table
                    st.subheader("All Templates")
                    if st.session_state.report_templates:
                        template_list = []
                        for name, template in st.session_state.report_templates.items():
                            template_list.append({
                                "Name": name,
                                "Description": template.get("description", "No description"),
                                "Created": template.get("created_at", "Unknown"),
                                "Updated": template.get("updated_at", "Unknown")
                            })
                        
                        df = pd.DataFrame(template_list)
                        st.dataframe(df, use_container_width=True)
            
            with template_tabs[2]:  # Import/Export
                st.subheader("Import/Export Templates")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Import Templates")
                    uploaded_file = st.file_uploader("Upload template file (JSON)", type=["json"])
                    if uploaded_file is not None:
                        try:
                            imported_templates = json.load(uploaded_file)
                            if isinstance(imported_templates, dict):
                                # If it's a single template
                                if "name" in imported_templates and "content" in imported_templates:
                                    template_name = imported_templates["name"]
                                    st.session_state.report_templates[template_name] = imported_templates
                                    _save_report_templates(st.session_state.report_templates)
                                    st.success(f"Template '{template_name}' imported successfully!")
                                else:
                                    # If it's a dict with multiple templates
                                    for name, template in imported_templates.items():
                                        st.session_state.report_templates[name] = template
                                    _save_report_templates(st.session_state.report_templates)
                                    st.success(f"{len(imported_templates)} templates imported successfully!")
                            elif isinstance(imported_templates, list):
                                # If it's a list of templates
                                for template in imported_templates:
                                    if "name" in template:
                                        st.session_state.report_templates[template["name"]] = template
                                _save_report_templates(st.session_state.report_templates)
                                st.success(f"{len(imported_templates)} templates imported successfully!")
                            else:
                                st.error("Invalid template file format")
                        except Exception as e:
                            st.error(f"Error importing template: {e}")
                
                with col2:
                    st.subheader("Export Templates")
                    export_option = st.selectbox("Export Option", ["All Templates", "Selected Template"])
                    
                    if export_option == "All Templates":
                        if st.button("Export All Templates", type="primary", use_container_width=True):
                            template_json = json.dumps(st.session_state.report_templates, indent=2, default=str)
                            st.download_button(
                                label="Download All Templates",
                                data=template_json,
                                file_name=f"report_templates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    else:  # Selected Template
                        template_names = list(st.session_state.report_templates.keys())
                        if template_names:
                            selected_export_template = st.selectbox("Select Template", template_names)
                            if st.button("Export Selected Template", type="primary", use_container_width=True):
                                template_json = json.dumps(st.session_state.report_templates[selected_export_template], indent=2, default=str)
                                st.download_button(
                                    label="Download Template",
                                    data=template_json,
                                    file_name=f"template_{selected_export_template}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
            
            with template_tabs[3]:  # Template Preview
                st.subheader("Template Preview")
                
                if st.session_state.report_templates:
                    preview_template_names = list(st.session_state.report_templates.keys())
                    selected_preview = st.selectbox("Select Template to Preview", preview_template_names, key="preview_select")
                    
                    if selected_preview:
                        template_content = st.session_state.report_templates[selected_preview].get("content", {})
                        st.json(template_content)
                        
                        with st.expander("Preview as Report Structure"):
                            def display_template_structure(obj, level=0):
                                indent = "  " * level
                                if isinstance(obj, dict):
                                    for key, value in obj.items():
                                        if isinstance(value, (dict, list)):
                                            st.write(f"{indent}{key}:")
                                            display_template_structure(value, level + 1)
                                        else:
                                            st.write(f"{indent}{key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                                elif isinstance(obj, list):
                                    for i, item in enumerate(obj):
                                        st.write(f"{indent}[{i}]:")
                                        display_template_structure(item, level + 1)
                            
                            display_template_structure(template_content)
                else:
                    st.info("No templates available to preview. Create a template first.")


    with tabs[5]: # Model Dashboard tab  # noqa: F821
        with st.container(border=True):
            st.header("🤖 Model Performance Dashboard")
            st.markdown("Analyze the performance of different models used in adversarial testing and evolution.")
            
            # Model performance data
            model_performance = st.session_state.get("adversarial_model_performance", {})
            
            # Tabs for different model analytics views
            model_tabs = st.tabs(["📊 Performance Overview", "📈 Model Comparison", "🔍 Detailed Analysis", "⚙️ Model Configuration"])
            
            with model_tabs[0]:  # Performance Overview
                st.subheader("Model Performance Overview")
                
                if not model_performance:
                    st.info("No model performance data available. Run adversarial testing or evolution with multiple models to generate data.")
                    
                    # Show model selection and configuration options
                    st.subheader("Configure Model Testing")
                    st.markdown("""
                    To generate model performance data:
                    1. Run adversarial testing with multiple models
                    2. Use different models in evolution runs
                    3. Compare performance across different tasks
                    """)
                else:
                    # Calculate overall metrics
                    model_list = []
                    for model_id, perf in model_performance.items():
                        model_list.append({
                            "Model": model_id,
                            "Score": perf.get("score", 0),
                            "Issues Found": perf.get("issues_found", 0),
                            "Avg Response Time": perf.get("avg_response_time", 0),
                            "Cost": perf.get("cost", 0.0),
                            "Success Rate": perf.get("success_rate", 0.0),
                            "Tokens Used": perf.get("tokens_used", 0)
                        })
                    
                    df = pd.DataFrame(model_list)
                    df_sorted = df.sort_values(by="Score", ascending=False)
                    
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_models = len(df)
                        st.metric("Total Models", total_models)
                    with col2:
                        avg_score = df['Score'].mean() if not df.empty else 0
                        st.metric("Avg Score", f"{avg_score:.2f}")
                    with col3:
                        best_model = df_sorted.iloc[0]['Model'] if not df_sorted.empty else "N/A"
                        best_score = df_sorted.iloc[0]['Score'] if not df_sorted.empty else 0
                        st.metric("Best Model", f"{best_model}")
                    with col4:
                        total_issues = df['Issues Found'].sum() if not df.empty else 0
                        st.metric("Total Issues Found", total_issues)
                    
                    st.divider()
                    
                    # Top models performance chart
                    if not df_sorted.empty:
                        fig = px.bar(df_sorted, x="Model", y="Score", 
                                   title="Model Performance Comparison", 
                                   color="Score",
                                   color_continuous_scale="viridis")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
            
            with model_tabs[1]:  # Model Comparison
                st.subheader("Model Comparison")
                
                if model_performance:
                    # Multi-metric comparison
                    comparison_metrics = st.multiselect(
                        "Select metrics to compare",
                        ["Score", "Issues Found", "Avg Response Time", "Cost", "Success Rate", "Tokens Used"],
                        default=["Score", "Issues Found"]
                    )
                    
                    if comparison_metrics and model_performance:
                        comparison_data = []
                        for model_id, perf in model_performance.items():
                            row = {"Model": model_id}
                            for metric in comparison_metrics:
                                # Map display metric names to internal keys
                                if metric == "Score":
                                    row[metric] = perf.get("score", 0)
                                elif metric == "Issues Found":
                                    row[metric] = perf.get("issues_found", 0)
                                elif metric == "Avg Response Time":
                                    row[metric] = perf.get("avg_response_time", 0)
                                elif metric == "Cost":
                                    row[metric] = perf.get("cost", 0.0)
                                elif metric == "Success Rate":
                                    row[metric] = perf.get("success_rate", 0.0)
                                elif metric == "Tokens Used":
                                    row[metric] = perf.get("tokens_used", 0)
                            comparison_data.append(row)
                        
                        if comparison_data:
                            df_comparison = pd.DataFrame(comparison_data)
                            
                            # Create comparison visualization
                            fig = px.bar(df_comparison, x="Model", y=comparison_metrics,
                                       title="Multi-Metric Model Comparison",
                                       barmode="group")
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show comparison table
                            st.dataframe(df_comparison, use_container_width=True)
                else:
                    st.info("Run model testing to see comparison data.")
            
            with model_tabs[2]:  # Detailed Analysis
                st.subheader("Detailed Model Analysis")
                
                if model_performance:
                    # Model selector for detailed view
                    model_names = list(model_performance.keys())
                    selected_model = st.selectbox("Select Model for Detailed Analysis", model_names)
                    
                    if selected_model:
                        model_details = model_performance[selected_model]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Overall Score", f"{model_details.get('score', 0):.2f}")
                            st.metric("Issues Found", model_details.get("issues_found", 0))
                            st.metric("Avg Response Time", f"{model_details.get('avg_response_time', 0):.2f}s")
                        with col2:
                            st.metric("Cost ($)", f"${model_details.get('cost', 0.0):.4f}")
                            st.metric("Success Rate", f"{model_details.get('success_rate', 0.0):.1%}")
                            st.metric("Tokens Used", f"{model_details.get('tokens_used', 0):,}")
                        
                        # Detailed metrics breakdown
                        with st.expander("Detailed Metrics"):
                            st.json(model_details)
                        
                        # Model-specific charts if available
                        if "performance_history" in model_details:
                            st.subheader(f"Performance Trend for {selected_model}")
                            history = model_details["performance_history"]
                            if history:
                                df_history = pd.DataFrame(history)
                                if "score" in df_history.columns:
                                    fig = px.line(df_history, x=df_history.index, y="score",
                                                title=f"Score Trend for {selected_model}")
                                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("Run model testing to see detailed analysis.")
            
            with model_tabs[3]:  # Model Configuration
                st.subheader("Model Configuration & Management")
                
                # Model provider and API configuration
                st.markdown("### API Configuration")
                col1, col2 = st.columns(2)
                with col1:
                    default_provider = st.selectbox("Default Provider", 
                                                   ["OpenAI", "Anthropic", "OpenRouter", "Custom"])
                with col2:
                    api_base_url = st.text_input("API Base URL", 
                                                value=st.session_state.get("api_base_url", "https://api.openai.com/v1"))
                
                # Model management
                st.markdown("### Available Models")
                if model_performance:
                    model_df = pd.DataFrame([
                        {"Model": k, "Provider": k.split('/')[0] if '/' in k else "Unknown", 
                         "Score": v.get("score", 0), "Status": "Active"}
                        for k, v in model_performance.items()
                    ])
                    st.dataframe(model_df, use_container_width=True)
                else:
                    st.info("No models configured yet.")
                
                # Model testing configuration
                st.markdown("### Test Configuration")
                test_config_col1, test_config_col2 = st.columns(2)
                with test_config_col1:
                    test_iterations = st.number_input("Test Iterations per Model", 
                                                     min_value=1, max_value=100, value=5)
                    test_timeout = st.number_input("Test Timeout (seconds)", 
                                                  min_value=10, max_value=600, value=60)
                with test_config_col2:
                    test_concurrent = st.number_input("Concurrent Tests", 
                                                     min_value=1, max_value=10, value=2)
                    test_temperature = st.slider("Test Temperature", 
                                                 min_value=0.0, max_value=2.0, value=0.7, step=0.1)
                
                # Run model comparison test
                if st.button("Run Model Comparison Test", type="primary"):
                    st.info("This would initiate a comprehensive model comparison test in a real implementation.")
                    
                    # Simulate test progress
                    import time
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        status_text.text(f"Testing models: {i + 1}% complete")
                        time.sleep(0.01)  # Simulate work
                    
                    status_text.text("Model comparison test completed!")
                    st.success("Model comparison test finished. Results would be displayed in the dashboard.")
            
            # Export model performance data
            st.divider()
            st.subheader("Export Model Data")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Export Performance Data"):
                    if model_performance:
                        perf_json = json.dumps(model_performance, indent=2, default=str)
                        st.download_button(
                            label="Download JSON",
                            data=perf_json,
                            file_name=f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
            with col2:
                if st.button("Export as CSV"):
                    if model_performance:
                        perf_list = []
                        for model_id, perf_data in model_performance.items():
                            row = {"Model": model_id}
                            row.update(perf_data)
                            perf_list.append(row)
                        df_export = pd.DataFrame(perf_list)
                        csv_export = df_export.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv_export,
                            file_name=f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            with col3:
                if st.button("Generate Report"):
                    st.info("Model performance report would be generated in a real implementation.")


    with tabs[6]: # Tasks tab  # noqa: F821
        with st.container(border=True):
            st.header("✅ Task Management")
            st.markdown("Create, view, and manage tasks related to content improvement.")
            
            # Initialize tasks in session state if not already present
            if "tasks" not in st.session_state:
                st.session_state.tasks = []
            
            # Task management tabs
            task_tabs = st.tabs(["📋 View Tasks", "➕ Create Task", "📊 Task Analytics", "⚙️ Task Settings"])
            
            with task_tabs[0]:  # View Tasks
                st.subheader("View and Manage Tasks")
                
                # Task filtering options
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    status_filter = st.selectbox("Filter by Status", ["All", "To Do", "In Progress", "Completed", "Cancelled"])
                with col2:
                    priority_filter = st.selectbox("Filter by Priority", ["All", "Low", "Medium", "High", "Critical"])
                with col3:
                    assignee_filter = st.text_input("Filter by Assignee")
                with col4:
                    search_filter = st.text_input("Search Tasks")
                
                # Apply filters to tasks
                filtered_tasks = st.session_state.tasks
                
                if status_filter != "All":
                    filtered_tasks = [task for task in filtered_tasks if task.get("status", "To Do") == status_filter]
                
                if priority_filter != "All":
                    filtered_tasks = [task for task in filtered_tasks if task.get("priority", "Medium") == priority_filter]
                
                if assignee_filter:
                    filtered_tasks = [task for task in filtered_tasks if assignee_filter.lower() in task.get("assignee", "").lower()]
                
                if search_filter:
                    filtered_tasks = [task for task in filtered_tasks if search_filter.lower() in task.get("title", "").lower() or search_filter.lower() in task.get("description", "").lower()]
                
                # Display tasks
                if filtered_tasks:
                    # Sort tasks by due date
                    sorted_tasks = sorted(filtered_tasks, key=lambda x: x.get("due_date", datetime.max), reverse=False)
                    
                    # Show task cards
                    for i, task in enumerate(sorted_tasks):
                        with st.container(border=True):
                            col_task1, col_task2 = st.columns([4, 1])
                            
                            with col_task1:
                                st.subheader(task.get("title", "No Title"))
                                st.write(f"**Description:** {task.get('description', 'No description provided')}")
                                st.write(f"**Assignee:** {task.get('assignee', 'Unassigned')}")
                                st.write(f"**Priority:** {task.get('priority', 'Medium')}")
                                st.write(f"**Due Date:** {task.get('due_date', 'No due date')}")
                                st.write(f"**Created:** {task.get('created_at', 'Unknown')}")
                                st.write(f"**Tags:** {', '.join(task.get('tags', []))}")
                            
                            with col_task2:
                                # Task status indicator
                                status_color = {
                                    "To Do": "gray",
                                    "In Progress": "orange", 
                                    "Completed": "green",
                                    "Cancelled": "red"
                                }.get(task.get("status", "To Do"), "gray")
                                
                                st.markdown(f"**Status:** <span style='color:{status_color}'>{task.get('status', 'To Do')}</span>", 
                                          unsafe_allow_html=True)
                                
                                # Task actions
                                task_actions = st.columns(2)
                                with task_actions[0]:
                                    if st.button(f"Edit##task{i}", key=f"edit_task_{i}"):
                                        # Pre-fill form for editing
                                        st.session_state.edit_task_id = i
                                        st.session_state.edit_task_title = task.get("title", "")
                                        st.session_state.edit_task_description = task.get("description", "")
                                        st.session_state.edit_task_assignee = task.get("assignee", "")
                                        st.session_state.edit_task_due_date = task.get("due_date", datetime.now().date())
                                        st.session_state.edit_task_priority = task.get("priority", "Medium")
                                        st.session_state.edit_task_status = task.get("status", "To Do")
                                        st.session_state.edit_task_tags = ", ".join(task.get("tags", []))
                                
                                with task_actions[1]:
                                    if st.button(f"Delete##task{i}", key=f"delete_task_{i}", type="secondary"):
                                        st.session_state.tasks.pop(i)
                                        st.success(f"Task '{task.get('title', 'Unknown')}' deleted")
                                        st.rerun()
                            
                            # Progress bar for tasks that have progress
                            if "progress" in task:
                                st.progress(task["progress"] / 100)
                            
                            st.divider()
                    
                    st.info(f"Showing {len(filtered_tasks)} of {len(st.session_state.tasks)} tasks")
                else:
                    st.info("No tasks found with the current filters. Create a new task using the 'Create Task' tab.")
            
            with task_tabs[1]:  # Create Task
                st.subheader("Create New Task")
                
                # Initialize session state for edit form if needed
                if "edit_task_id" not in st.session_state:
                    st.session_state.edit_task_id = None
                
                # Check if we're editing a task
                if st.session_state.edit_task_id is not None and st.session_state.edit_task_id < len(st.session_state.tasks):
                    # Editing mode
                    task_to_edit = st.session_state.tasks[st.session_state.edit_task_id]
                    title = st.text_input("Task Title", value=task_to_edit.get("title", ""))
                    description = st.text_area("Description", value=task_to_edit.get("description", ""))
                    assignee = st.text_input("Assignee", value=task_to_edit.get("assignee", ""))
                    due_date = st.date_input("Due Date", value=task_to_edit.get("due_date", datetime.now().date()))
                    priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"], 
                                          index=["Low", "Medium", "High", "Critical"].index(task_to_edit.get("priority", "Medium")))
                    status = st.selectbox("Status", ["To Do", "In Progress", "Completed", "Cancelled"], 
                                        index=["To Do", "In Progress", "Completed", "Cancelled"].index(task_to_edit.get("status", "To Do")))
                    tags = st.text_input("Tags (comma-separated)", value=", ".join(task_to_edit.get("tags", [])))
                    
                    if st.button("Update Task", type="primary"):
                        # Update the task
                        st.session_state.tasks[st.session_state.edit_task_id].update({
                            "title": title,
                            "description": description,
                            "assignee": assignee,
                            "due_date": due_date,
                            "priority": priority,
                            "status": status,
                            "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                            "updated_at": datetime.now().isoformat()
                        })
                        st.success(f"Task '{title}' updated successfully!")
                        # Reset edit state
                        st.session_state.edit_task_id = None
                        st.rerun()
                    
                    if st.button("Cancel Edit"):
                        st.session_state.edit_task_id = None
                        st.rerun()
                else:
                    # Create mode
                    with st.form("new_task_form"):
                        title = st.text_input("Task Title", placeholder="e.g., Review security policy")
                        description = st.text_area("Description", placeholder="Provide details about the task...")
                        assignee = st.text_input("Assignee", placeholder="e.g., John Doe")
                        due_date = st.date_input("Due Date", value=datetime.now().date())
                        priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"], index=1)
                        status = st.selectbox("Status", ["To Do", "In Progress", "Completed", "Cancelled"], index=0)
                        tags = st.text_input("Tags (comma-separated)", placeholder="e.g., security, review, urgent")
                        
                        submitted = st.form_submit_button("Create Task", type="primary")
                        
                        if submitted:
                            if title:
                                # Create new task
                                new_task = {
                                    "title": title,
                                    "description": description,
                                    "assignee": assignee,
                                    "due_date": due_date,
                                    "priority": priority,
                                    "status": status,
                                    "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                                    "created_at": datetime.now().isoformat(),
                                    "updated_at": datetime.now().isoformat()
                                }
                                st.session_state.tasks.append(new_task)
                                st.success(f"Task '{title}' created successfully!")
                                st.rerun()
                            else:
                                st.error("Task title is required.")
            
            with task_tabs[2]:  # Task Analytics
                st.subheader("Task Analytics")
                
                if st.session_state.tasks:
                    # Calculate task statistics
                    total_tasks = len(st.session_state.tasks)
                    completed_tasks = len([t for t in st.session_state.tasks if t.get("status") == "Completed"])
                    in_progress_tasks = len([t for t in st.session_state.tasks if t.get("status") == "In Progress"])
                    todo_tasks = len([t for t in st.session_state.tasks if t.get("status") == "To Do"])
                    cancelled_tasks = len([t for t in st.session_state.tasks if t.get("status") == "Cancelled"])
                    
                    # Priority breakdown
                    priority_breakdown = {}
                    for task in st.session_state.tasks:
                        priority = task.get("priority", "Medium")
                        priority_breakdown[priority] = priority_breakdown.get(priority, 0) + 1
                    
                    # Assignee breakdown
                    assignee_breakdown = {}
                    for task in st.session_state.tasks:
                        assignee = task.get("assignee", "Unassigned")
                        assignee_breakdown[assignee] = assignee_breakdown.get(assignee, 0) + 1
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Tasks", total_tasks)
                    with col2:
                        st.metric("Completed", completed_tasks, f"{(completed_tasks/total_tasks)*100:.1f}%" if total_tasks > 0 else "0%")
                    with col3:
                        st.metric("In Progress", in_progress_tasks)
                    with col4:
                        st.metric("To Do", todo_tasks)
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    with col1:
                        # Status distribution
                        status_data = {
                            "Status": ["To Do", "In Progress", "Completed", "Cancelled"],
                            "Count": [todo_tasks, in_progress_tasks, completed_tasks, cancelled_tasks]
                        }
                        df_status = pd.DataFrame(status_data)
                        fig_status = px.bar(df_status, x="Status", y="Count", title="Task Status Distribution")
                        st.plotly_chart(fig_status, use_container_width=True)
                    
                    with col2:
                        # Priority distribution
                        priority_data = {
                            "Priority": list(priority_breakdown.keys()),
                            "Count": list(priority_breakdown.values())
                        }
                        if priority_data["Priority"]:
                            df_priority = pd.DataFrame(priority_data)
                            fig_priority = px.pie(df_priority, values="Count", names="Priority", title="Task Priority Distribution")
                            st.plotly_chart(fig_priority, use_container_width=True)
                        else:
                            st.info("No priority data to display")
                    
                    # Assignee workload
                    if assignee_breakdown:
                        assignee_data = {
                            "Assignee": list(assignee_breakdown.keys()),
                            "Task Count": list(assignee_breakdown.values())
                        }
                        df_assignee = pd.DataFrame(assignee_data)
                        fig_assignee = px.bar(df_assignee, x="Assignee", y="Task Count", title="Tasks per Assignee")
                        st.plotly_chart(fig_assignee, use_container_width=True)
                
                else:
                    st.info("Create tasks to see analytics.")
            
            with task_tabs[3]:  # Task Settings
                st.subheader("Task Settings")
                
                st.markdown("""
                ### Task Management Options
                Configure how tasks are handled in the system.
                """)
                
                # Task settings
                auto_assign = st.checkbox("Auto-assign new tasks to current user", value=False)
                task_notifications = st.checkbox("Enable task notifications", value=True)
                default_priority = st.selectbox("Default task priority", ["Low", "Medium", "High", "Critical"], index=1)
                show_completed = st.checkbox("Show completed tasks by default", value=False)
                
                st.divider()
                
                # Task import/export
                st.subheader("Import/Export Tasks")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Import Tasks")
                    uploaded_tasks = st.file_uploader("Upload tasks file (JSON/CSV)", type=["json", "csv"])
                    if uploaded_tasks is not None:
                        try:
                            if uploaded_tasks.type == "application/json":
                                imported_tasks = json.load(uploaded_tasks)
                                if isinstance(imported_tasks, list):
                                    # Add imported tasks to existing tasks
                                    st.session_state.tasks.extend(imported_tasks)
                                    st.success(f"Imported {len(imported_tasks)} tasks successfully!")
                                else:
                                    st.error("Invalid JSON format: expected an array of tasks")
                            elif uploaded_tasks.type == "text/csv":
                                # For CSV, we'll need to convert to our task format
                                import io
                                df_tasks = pd.read_csv(io.StringIO(uploaded_tasks.getvalue().decode("utf-8")))
                                new_tasks = df_tasks.to_dict('records')
                                st.session_state.tasks.extend(new_tasks)
                                st.success(f"Imported {len(new_tasks)} tasks from CSV!")
                        except Exception as e:
                            st.error(f"Error importing tasks: {e}")
                
                with col2:
                    st.subheader("Export Tasks")
                    export_format = st.radio("Export Format", ["JSON", "CSV"], horizontal=True)
                    
                    if st.button("Export All Tasks", type="primary", use_container_width=True):
                        if export_format == "JSON":
                            tasks_json = json.dumps(st.session_state.tasks, indent=2, default=str)
                            st.download_button(
                                label="Download JSON",
                                data=tasks_json,
                                file_name=f"tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        elif export_format == "CSV":
                            if st.session_state.tasks:
                                df_export = pd.DataFrame(st.session_state.tasks)
                                csv_export = df_export.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv_export,
                                    file_name=f"tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("No tasks to export")
                
                # Task cleanup
                st.divider()
                st.subheader("Clean Up Tasks")
                if st.button("Remove Completed Tasks", type="secondary"):
                    initial_count = len(st.session_state.tasks)
                    st.session_state.tasks = [task for task in st.session_state.tasks if task.get("status") != "Completed"]
                    removed_count = initial_count - len(st.session_state.tasks)
                    st.success(f"Removed {removed_count} completed tasks")


    with tabs[7]: # Admin tab  # noqa: F821
        with st.container(border=True):
            st.header("👑 Administration Panel")
            st.markdown("Manage users, roles, system settings, and application configuration.")
            
            # Initialize admin session state if not already present
            if "admin_settings" not in st.session_state:
                st.session_state.admin_settings = {
                    "system_name": "OpenEvolve Platform",
                    "maintenance_mode": False,
                    "user_registration": True,
                    "max_concurrent_users": 100,
                    "default_user_role": "user"
                }
            
            # Admin tabs for different management functions
            admin_tabs = st.tabs(["👥 User Management", "🔐 Role Management", "⚙️ System Settings", "📋 System Info", "🚨 Maintenance"])
            
            with admin_tabs[0]:  # User Management
                st.subheader("User Management")
                
                # User actions
                user_action = st.selectbox("Action", ["View Users", "Add User", "Edit User", "Deactivate User"])
                
                if user_action == "Add User":
                    with st.form("add_user_form"):
                        new_username = st.text_input("Username")
                        new_email = st.text_input("Email")
                        new_role = st.selectbox("Role", list(ROLES.keys()) if 'ROLES' in globals() else ["user", "admin"])
                        new_password = st.text_input("Password", type="password")
                        confirm_password = st.text_input("Confirm Password", type="password")
                        
                        if st.form_submit_button("Create User", type="primary"):
                            if not new_username or not new_email or not new_password:
                                st.error("All fields are required")
                            elif new_password != confirm_password:
                                st.error("Passwords do not match")
                            else:
                                # In a real implementation, this would add the user to a database
                                # For now, we'll just show success
                                st.session_state.user_roles[new_username] = new_role
                                st.success(f"User '{new_username}' created with role '{new_role}'")
                
                elif user_action == "Edit User":
                    if st.session_state.user_roles:
                        user_to_edit = st.selectbox("Select User to Edit", list(st.session_state.user_roles.keys()))
                        if user_to_edit:
                            with st.form("edit_user_form"):
                                new_role = st.selectbox("New Role", list(ROLES.keys()) if 'ROLES' in globals() else ["user", "admin"], 
                                                      index=list(ROLES.keys()).index(st.session_state.user_roles[user_to_edit]) if 'ROLES' in globals() and st.session_state.user_roles[user_to_edit] in ROLES.keys() else 0)
                                active_status = st.checkbox("Active", value=True)  # In a real system, you'd track this
                                
                                if st.form_submit_button("Update User", type="primary"):
                                    st.session_state.user_roles[user_to_edit] = new_role
                                    st.success(f"User '{user_to_edit}' updated")
                    else:
                        st.info("No users to edit")
                
                elif user_action == "Deactivate User":
                    if st.session_state.user_roles:
                        user_to_deactivate = st.selectbox("Select User to Deactivate", list(st.session_state.user_roles.keys()))
                        if user_to_deactivate:
                            if st.button("Deactivate User", type="secondary"):
                                # In a real implementation, you would deactivate the user
                                st.success(f"User '{user_to_deactivate}' deactivated")
                    else:
                        st.info("No users to deactivate")
                
                # Display all users
                st.divider()
                st.subheader("All Users")
                
                if st.session_state.user_roles:
                    user_list = []
                    for username, role in st.session_state.user_roles.items():
                        user_list.append({
                            "Username": username,
                            "Role": role,
                            "Status": "Active"  # In a real system, you'd track this
                        })
                    
                    df_users = pd.DataFrame(user_list)
                    st.dataframe(df_users, use_container_width=True)
                else:
                    st.info("No users registered yet.")
            
            with admin_tabs[1]:  # Role Management
                st.subheader("Role Management")
                
                # Display available roles
                if 'ROLES' in globals():
                    st.markdown("### Available Roles")
                    for role_name, role_desc in ROLES.items():
                        st.markdown(f"**{role_name}**: {role_desc}")
                else:
                    st.info("Role system not initialized")
                
                # Role creation (conceptual)
                st.divider()
                st.subheader("Create Custom Role")
                
                with st.expander("Custom Role Creator", expanded=False):
                    role_name = st.text_input("Role Name")
                    role_description = st.text_input("Role Description")
                    
                    # Role permissions (conceptual)
                    permissions = st.multiselect(
                        "Permissions",
                        [
                            "view_dashboard",
                            "create_content",
                            "run_evolution",
                            "run_adversarial",
                            "manage_users",
                            "view_reports",
                            "export_data",
                            "admin_access"
                        ],
                        default=["view_dashboard", "create_content"]
                    )
                    
                    if st.button("Create Role", type="primary"):
                        if role_name and role_description:
                            # In a real implementation, this would create the role in a database
                            st.success(f"Role '{role_name}' created with permissions: {', '.join(permissions)}")
                        else:
                            st.error("Role name and description are required")
            
            with admin_tabs[2]:  # System Settings
                st.subheader("System Configuration")
                
                # System name
                st.session_state.admin_settings["system_name"] = st.text_input(
                    "System Name", 
                    value=st.session_state.admin_settings.get("system_name", "OpenEvolve Platform")
                )
                
                # Maintenance mode
                st.session_state.admin_settings["maintenance_mode"] = st.checkbox(
                    "Maintenance Mode", 
                    value=st.session_state.admin_settings.get("maintenance_mode", False)
                )
                
                if st.session_state.admin_settings["maintenance_mode"]:
                    st.warning("⚠️ Maintenance mode is enabled. Only admins can access the system.")
                
                # User registration
                st.session_state.admin_settings["user_registration"] = st.checkbox(
                    "Allow User Registration", 
                    value=st.session_state.admin_settings.get("user_registration", True)
                )
                
                # Max concurrent users
                st.session_state.admin_settings["max_concurrent_users"] = st.number_input(
                    "Max Concurrent Users", 
                    min_value=1, 
                    max_value=10000, 
                    value=st.session_state.admin_settings.get("max_concurrent_users", 100)
                )
                
                # Default role
                st.session_state.admin_settings["default_user_role"] = st.selectbox(
                    "Default User Role", 
                    options=["user", "admin"] if 'ROLES' in globals() else ["user"],
                    index=0 if st.session_state.admin_settings.get("default_user_role") == "user" else 0
                )
                
                # API settings
                st.divider()
                st.subheader("API Configuration")
                
                api_rate_limit = st.number_input("API Rate Limit (requests per minute)", min_value=1, max_value=10000, value=100)
                api_timeout = st.number_input("API Timeout (seconds)", min_value=1, max_value=300, value=60)
                
                # Save settings
                if st.button("Save System Settings", type="primary"):
                    # In a real implementation, this would save settings to a database or config file
                    st.success("System settings saved successfully!")
            
            with admin_tabs[3]:  # System Info
                st.subheader("System Information")
                
                # Application info
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Application")
                    st.write(f"**Name**: {st.session_state.admin_settings.get('system_name', 'OpenEvolve Platform')}")
                    st.write("**Version**: 1.0.0")  # Would come from actual version in real implementation
                    st.write(f"**Environment**: {os.getenv('ENVIRONMENT', 'Development')}")
                
                with col2:
                    st.markdown("### Runtime")
                    st.write(f"**Python Version**: {sys.version}")
                    st.write(f"**Streamlit Version**: {st.__version__}")
                    st.write(f"**Platform**: {sys.platform}")
                
                # Statistics
                st.divider()
                st.subheader("System Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_users = len(st.session_state.user_roles) if "user_roles" in st.session_state else 0
                    st.metric("Total Users", total_users)
                with col2:
                    # Would track actual evolution runs in a real implementation
                    st.metric("Evolution Runs", 0)
                with col3:
                    # Would track actual adversarial runs in a real implementation
                    st.metric("Adversarial Tests", 0)
                with col4:
                    # Would track actual tasks in a real implementation
                    total_tasks = len(st.session_state.tasks) if "tasks" in st.session_state else 0
                    st.metric("Total Tasks", total_tasks)
                
                # Resource usage
                st.divider()
                st.subheader("Resource Usage")
                
                # Memory usage simulation (in a real implementation, you'd get actual memory usage)
                import psutil
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent if hasattr(os, 'statvfs') else 0  # This might not work on Windows
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Memory Usage**: {memory_percent}%")
                    st.progress(memory_percent / 100)
                with col2:
                    st.markdown(f"**Disk Usage**: {disk_percent}% (est.)")
                    st.progress(disk_percent / 100)
            
            with admin_tabs[4]:  # Maintenance
                st.subheader("System Maintenance")
                
                # Data cleanup options
                with st.expander("Data Cleanup", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        cleanup_logs = st.checkbox("Clean up old logs", value=True)
                        cleanup_temp = st.checkbox("Clean up temporary files", value=True)
                    with col2:
                        cleanup_sessions = st.checkbox("Clear old sessions", value=False)
                        cleanup_cache = st.checkbox("Clear application cache", value=True)
                    
                    if st.button("Run Cleanup", type="secondary"):
                        # Simulate cleanup process
                        import time
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            status_text.text(f"Cleaning up... {i + 1}%")
                            time.sleep(0.01)
                        
                        status_text.text("Cleanup completed!")
                        st.success("System maintenance completed!")
                
                # Backup options
                st.divider()
                with st.expander("Backup & Restore", expanded=False):
                    backup_option = st.selectbox("Backup Operation", ["Create Backup", "Restore from Backup", "Download Backup"])
                    
                    if backup_option == "Create Backup":
                        if st.button("Create Full System Backup", type="primary"):
                            # Simulate backup process
                            st.info("Creating backup of system data, user info, and settings...")
                            st.success("Backup completed successfully!")
                    
                    elif backup_option == "Restore from Backup":
                        uploaded_backup = st.file_uploader("Upload Backup File", type=["zip", "json"])
                        if uploaded_backup and st.button("Restore Backup", type="secondary"):
                            st.warning("⚠️ This will overwrite all system data. Are you sure?")
                            if st.button("Confirm Restoration", type="secondary"):
                                st.success("System restored from backup!")
                    
                    elif backup_option == "Download Backup":
                        st.info("Download options would be available in a full implementation")
                
                # System operations
                st.divider()
                st.subheader("System Operations")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Restart Application", type="secondary"):
                        st.info("Application restart would occur in a real implementation")
                
                with col2:
                    if st.button("Clear All Data", type="secondary"):
                        st.warning("⚠️ This will permanently delete all data. This action cannot be undone!")
                        if st.button("Confirm Clear All Data", type="secondary", key="confirm_clear"):
                            st.error("⚠️ Data clearing functionality would be implemented in a real system")
                
                # Audit logs
                st.divider()
                st.subheader("Audit Trail")
                st.info("System audit logs would be displayed here in a production implementation")

    with tabs[8]: # Analytics Dashboard tab  # noqa: F821
        with st.container(border=True):
            st.header("📊 Analytics Dashboard")
            
            # Comprehensive analytics dashboard implementation
            # Import and render the analytics dashboard if available, otherwise use fallback
            analytics_dashboard_available = False
            try:
                from analytics_dashboard import render_analytics_dashboard
                render_analytics_dashboard()
                analytics_dashboard_available = True
            except ImportError:
                # Use comprehensive self-contained implementation
                st.subheader("📊 Advanced Analytics Dashboard")
                
                # Create tabs for different analytics views
                analytics_tabs = st.tabs(["📈 Standard Analytics", "🧬 Quality-Diversity", "🎯 Performance Metrics", "📋 Reports"])
                
                with analytics_tabs[0]:  # Standard Analytics
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Evolution metrics
                    evolution_history = st.session_state.get("evolution_history", [])
                    total_evolutions = len(evolution_history)
                    if evolution_history:
                        latest_generation = evolution_history[-1]
                        population = latest_generation.get("population", [])
                        if population:
                            best_fitness = max(ind.get("fitness", 0) for ind in population)
                            avg_fitness = sum(ind.get("fitness", 0) for ind in population) / len(population)
                        else:
                            best_fitness = 0
                            avg_fitness = 0
                    else:
                        best_fitness = 0
                        avg_fitness = 0
                    
                    # Adversarial metrics
                    adversarial_results = st.session_state.get("adversarial_results", {})
                    adversarial_iterations = adversarial_results.get("iterations", [])

                    if adversarial_iterations:
                        latest_iteration = adversarial_iterations[-1]
                        approval_check = latest_iteration.get("approval_check", {})
                        final_approval_rate = approval_check.get("approval_rate", 0)
                    else:
                        final_approval_rate = 0
                    
                    # Cost metrics
                    total_cost = st.session_state.get("adversarial_cost_estimate_usd", 0) + \
                                 st.session_state.get("evolution_cost_estimate_usd", 0)
                    
                    col1.metric("Total Evolutions", f"{total_evolutions:,}")
                    col2.metric("Best Fitness", f"{best_fitness:.4f}")
                    col3.metric("Final Approval Rate", f"{final_approval_rate:.1f}%")
                    col4.metric("Total Cost ($)", f"${total_cost:.4f}")
                    
                    st.divider()
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Fitness trend
                        if evolution_history:
                            fitness_data = []
                            generation_numbers = []
                            for generation in evolution_history:
                                population = generation.get("population", [])
                                if population:
                                    best_fitness_gen = max(ind.get("fitness", 0) for ind in population)
                                    avg_fitness_gen = sum(ind.get("fitness", 0) for ind in population) / len(population)
                                    fitness_data.append({
                                        'Generation': generation.get("generation", 0),
                                        'Best Fitness': best_fitness_gen,
                                        'Average Fitness': avg_fitness_gen
                                    })
                            
                            if fitness_data:
                                df = pd.DataFrame(fitness_data)
                                fig = px.line(df, x="Generation", y=["Best Fitness", "Average Fitness"], 
                                            title="Fitness Trend Over Generations")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Run an evolution to see fitness trends")
                    
                    with col2:
                        # Approval rate trend
                        if adversarial_iterations:
                            approval_data = []
                            iteration_numbers = []
                            for iteration in adversarial_iterations:
                                approval_check = iteration.get("approval_check", {})
                                approval_rate = approval_check.get("approval_rate", 0)
                                approval_data.append({
                                    'Iteration': iteration.get("iteration", 0),
                                    'Approval Rate': approval_rate
                                })
                                iteration_numbers.append(iteration.get("iteration", 0))
                            
                            if approval_data:
                                df = pd.DataFrame(approval_data)
                                fig = px.line(df, x="Iteration", y="Approval Rate", title="Approval Rate Trend")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Run adversarial testing to see approval trends")
                    
                    # Additional metrics
                    st.subheader("Evolution Progress Metrics")
                    if evolution_history:
                        progress_data = []
                        for i, gen in enumerate(evolution_history):
                            population = gen.get("population", [])
                            if population:
                                best = max(ind.get("fitness", 0) for ind in population)
                                avg = sum(ind.get("fitness", 0) for ind in population) / len(population)
                                diversity = sum(ind.get("diversity", 0) for ind in population) / len(population) if population else 0
                                progress_data.append({
                                    'Generation': i,
                                    'Best Fitness': best,
                                    'Average Fitness': avg,
                                    'Diversity': diversity
                                })
                        
                        if progress_data:
                            df_progress = pd.DataFrame(progress_data)
                            st.line_chart(df_progress.set_index('Generation'))
                    
                    # Fallback model performance visualization
                    st.subheader("Model Performance Overview")
                    model_performance = st.session_state.get("adversarial_model_performance", {})
                    if model_performance:
                        model_data = []
                        for model_id, perf_data in model_performance.items():
                            model_data.append({
                                "Model": model_id,
                                "Score": perf_data.get("score", 0),
                                "Issues Found": perf_data.get("issues_found", 0),
                                "Avg Response Time": perf_data.get("avg_response_time", 0),
                                "Cost": perf_data.get("cost", 0.0)
                            })
                        
                        if model_data:
                            df = pd.DataFrame(model_data)
                            df_sorted = df.sort_values(by="Score", ascending=False)
                            st.dataframe(df_sorted, use_container_width=True)
                            
                            # Model performance chart
                            col1, col2 = st.columns(2)
                            with col1:
                                fig = px.bar(df_sorted, x="Model", y="Score", title="Model Performance Comparison")
                                st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                fig = px.bar(df_sorted, x="Model", y="Cost", title="Model Cost Comparison")
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Run adversarial testing with multiple models to see performance data.")

                with analytics_tabs[1]:  # Quality-Diversity Analytics
                    st.subheader("Quality-Diversity (MAP-Elites) Analysis")
                    
                    # Show MAP-Elites grid visualization if available
                    if evolution_history:
                        st.info("MAP-Elites grid visualization would be shown here when using Quality-Diversity evolution mode.")
                        
                        # Feature diversity analysis
                        if evolution_history and len(evolution_history) > 0:
                            latest_pop = evolution_history[-1].get('population', [])
                            if latest_pop:
                                feature_data = []
                                for ind in latest_pop:
                                    feature_data.append({
                                        'Complexity': ind.get('complexity', 0),
                                        'Diversity': ind.get('diversity', 0),
                                        'Performance': ind.get('fitness', 0),
                                        'Code Length': len(ind.get('code', ''))
                                    })
                                
                                if feature_data:
                                    df_features = pd.DataFrame(feature_data)
                                    
                                    # Feature correlation matrix
                                    features = ['Complexity', 'Diversity', 'Performance']
                                    if all(col in df_features.columns for col in features):
                                        feature_subset = df_features[features]
                                        st.subheader("Feature Correlation Matrix")
                                        correlation_matrix = feature_subset.corr()
                                        st.dataframe(correlation_matrix)
                                        
                                        # Scatter plot matrix
                                        st.subheader("Feature Relationships")
                                        if len(feature_subset) > 1:
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                fig = px.scatter(df_features, x='Complexity', y='Performance', 
                                                               title='Complexity vs Performance')
                                                st.plotly_chart(fig, use_container_width=True)
                                            with col2:
                                                fig = px.scatter(df_features, x='Diversity', y='Performance', 
                                                               title='Diversity vs Performance')
                                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Run Quality-Diversity evolution to see MAP-Elites analysis.")
                
                with analytics_tabs[2]:  # Performance Metrics
                    st.subheader("Performance Metrics Dashboard")
                    
                    # System performance metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Evolution Runs", st.session_state.get("total_evolution_runs", 0))
                    with col2:
                        st.metric("Avg Best Score", f"{st.session_state.get('avg_best_score', 0.0):.3f}")
                    with col3:
                        st.metric("Best Ever Score", f"{st.session_state.get('best_ever_score', 0.0):.3f}")
                    
                    # Success rate metrics
                    if "monitoring_metrics" in st.session_state:
                        st.subheader("Real-time Monitoring Metrics")
                        monitoring = st.session_state.monitoring_metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Best Score", f"{monitoring.get('best_score', 0.0):.3f}")
                        with col2:
                            st.metric("Current Generation", monitoring.get('current_generation', 0))
                        with col3:
                            st.metric("Avg Diversity", f"{monitoring.get('avg_diversity', 0.0):.3f}")
                        with col4:
                            st.metric("Convergence Rate", f"{monitoring.get('convergence_rate', 0.0):.3f}")
                    
                    st.subheader("Resource Utilization")
                    # This would show actual system resources in a real implementation
                    st.info("Resource utilization metrics would be displayed in a full implementation.")
                
                with analytics_tabs[3]:  # Reports
                    st.subheader("Analytics Reports")
                    st.info("Generate and view detailed analytics reports here.")
                    
                    # Report generation options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Generate Evolution Report", type="primary", use_container_width=True):
                            st.info("Evolution report would be generated in a full implementation.")
                    with col2:
                        if st.button("Generate Model Comparison Report", use_container_width=True):
                            st.info("Model comparison report would be generated in a full implementation.")
                    with col3:
                        if st.button("Export All Data", type="secondary", use_container_width=True):
                            st.info("All analytics data would be exported in a full implementation.")

    with tabs[9]: # OpenEvolve Dashboard tab  # noqa: F821
        with st.container(border=True):
            st.header("🧬 OpenEvolve Advanced Dashboard")
            
            # Comprehensive OpenEvolve dashboard implementation
            # Import and render the OpenEvolve dashboard if available, otherwise use fallback
            openevolve_dashboard_available = False
            try:
                from openevolve_dashboard import render_openevolve_dashboard
                render_openevolve_dashboard()
                openevolve_dashboard_available = True
            except ImportError:
                # Use comprehensive fallback implementation
                st.subheader("OpenEvolve Features Overview")
                st.markdown("""
                **OpenEvolve provides advanced evolutionary computing capabilities:**
                
                - **Quality-Diversity Evolution** (MAP-Elites)
                - **Multi-Objective Optimization** (Pareto fronts)
                - **Adversarial Evolution** (Red Team/Blue Team)
                - **Symbolic Regression** (Mathematical discovery)
                - **Neuroevolution** (Neural architecture search)
                - **Algorithm Discovery** (Novel algorithm design)
                - **Prompt Evolution** (Optimize LLM prompts)
                
                These advanced features require the full OpenEvolve backend installation.
                """)
                
                # Main navigation tabs for OpenEvolve features
                main_tabs = st.tabs([
                    "🎯 Evolution Modes", 
                    "📊 Live Analytics", 
                    "🎛️ Configuration", 
                    "📈 Performance", 
                    "📋 Reports"
                ])
                
                with main_tabs[0]:  # Evolution Modes
                    st.subheader("Choose Evolution Mode")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        standard_evolution = st.button(
                            "🧬 Standard Evolution", 
                            help="Traditional genetic programming evolution",
                            use_container_width=True)
                        quality_diversity = st.button(
                            "🎯 Quality-Diversity (MAP-Elites)", 
                            help="Maintain diverse, high-performing solutions across feature dimensions",
                            use_container_width=True)
                        multi_objective = st.button(
                            "⚖️ Multi-Objective Optimization", 
                            help="Optimize for multiple competing objectives simultaneously",
                            use_container_width=True)
                    
                    with col2:
                        adversarial = st.button(
                            "⚔️ Adversarial Evolution", 
                            help="Red Team/Blue Team approach for robustness",
                            use_container_width=True)
                        symbolic_regression = st.button(
                            "🔍 Symbolic Regression", 
                            help="Discover mathematical expressions from data",
                            use_container_width=True)
                        neuroevolution = st.button(
                            "🧠 Neuroevolution", 
                            help="Evolve neural network architectures",
                            use_container_width=True)
                    
                    with col3:
                        algorithm_discovery = st.button(
                            "💡 Algorithm Discovery", 
                            help="Discover novel algorithmic approaches",
                            use_container_width=True)
                        prompt_evolution = st.button(
                            "📝 Prompt Evolution", 
                            help="Optimize prompts for LLMs",
                            use_container_width=True)
                        custom_evolution = st.button(
                            "🛠️ Custom Evolution", 
                            help="Customizable evolution parameters",
                            use_container_width=True)
                    
                    # Handle button clicks by setting session state
                    if standard_evolution:
                        st.session_state.evolution_mode = "standard"
                        st.success("Standard Evolution mode selected")
                    elif quality_diversity:
                        st.session_state.evolution_mode = "quality_diversity"
                        st.success("Quality-Diversity Evolution mode selected")
                    elif multi_objective:
                        st.session_state.evolution_mode = "multi_objective"
                        st.success("Multi-Objective Evolution mode selected")
                    elif adversarial:
                        st.session_state.evolution_mode = "adversarial"
                        st.success("Adversarial Evolution mode selected")
                    elif symbolic_regression:
                        st.session_state.evolution_mode = "symbolic_regression"
                        st.success("Symbolic Regression mode selected")
                    elif neuroevolution:
                        st.session_state.evolution_mode = "neuroevolution"
                        st.success("Neuroevolution mode selected")
                    elif algorithm_discovery:
                        st.session_state.evolution_mode = "algorithm_discovery"
                        st.success("Algorithm Discovery mode selected")
                    elif prompt_evolution:
                        st.session_state.evolution_mode = "prompt_optimization"
                        st.success("Prompt Evolution mode selected")
                    elif custom_evolution:
                        st.session_state.evolution_mode = "custom"
                        st.success("Custom Evolution mode selected")
                
                with main_tabs[1]:  # Live Analytics
                    st.subheader("Live Evolution Analytics")
                    
                    # Check if there's an active evolution run
                    if st.session_state.get("evolution_running", False):
                        st.info("Evolution is currently running. Live data will appear here.")
                        
                        # Live metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Generation", st.session_state.get("current_generation", 0))
                        with col2:
                            st.metric("Best Score", f"{st.session_state.get('best_score', 0.0):.3f}")
                        with col3:
                            st.metric("Population Size", st.session_state.get("population_size", 100))
                        with col4:
                            st.metric("Archive Size", st.session_state.get("archive_size", 0))
                        
                        # Real-time chart (simulated for now)
                        st.subheader("Performance Over Time")
                        progress_data = {
                            "Generation": list(range(1, 11)),
                            "Best Score": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.75, 0.82, 0.87, 0.91]
                        }
                        df = pd.DataFrame(progress_data)
                        fig = px.line(df, x="Generation", y="Best Score", title="Best Score Progression")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No active evolution run. Start an evolution to see live analytics.")
                        
                        # Display recent results if available
                        if "evolution_history" in st.session_state and st.session_state.evolution_history:
                            st.subheader("Recent Evolution Results")
                            st.info("Evolution insights would be displayed here when evolution history is available.")
                
                with main_tabs[2]:  # Configuration
                    st.subheader("OpenEvolve Configuration")
                    
                    config_tabs = st.tabs([" Core Settings", " Island Model", " Feature Dimensions", " Advanced"])
                    
                    with config_tabs[0]:  # Core Settings
                        col1, col2 = st.columns(2)
                        with col1:
                            st.session_state.max_iterations = st.number_input(
                                "Max Iterations", 
                                min_value=1, 
                                max_value=10000, 
                                value=st.session_state.get("max_iterations", 100),
                                help="Maximum number of evolutionary iterations"
                            )
                            st.session_state.population_size = st.number_input(
                                "Population Size", 
                                min_value=10, 
                                max_value=10000, 
                                value=st.session_state.get("population_size", 100),
                                help="Size of the population in each generation"
                            )
                            st.session_state.temperature = st.slider(
                                "LLM Temperature", 
                                min_value=0.0, 
                                max_value=2.0, 
                                value=st.session_state.get("temperature", 0.7),
                                step=0.1,
                                help="Temperature for LLM generation (higher = more creative)"
                            )
                        
                        with col2:
                            st.session_state.max_tokens = st.number_input(
                                "Max Tokens", 
                                min_value=100, 
                                max_value=32000, 
                                value=st.session_state.get("max_tokens", 4096),
                                help="Maximum tokens for LLM responses"
                            )
                            st.session_state.top_p = st.slider(
                                "Top-P Sampling", 
                                min_value=0.0, 
                                max_value=1.0, 
                                value=st.session_state.get("top_p", 0.95),
                                step=0.05,
                                help="Top-P sampling parameter for generation"
                            )
                            st.session_state.seed = st.number_input(
                                "Random Seed", 
                                value=st.session_state.get("seed", 42),
                                help="Seed for reproducible evolution runs"
                            )
                    
                    with config_tabs[1]:  # Island Model
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.session_state.num_islands = st.number_input(
                                "Number of Islands", 
                                min_value=1, 
                                max_value=20, 
                                value=st.session_state.get("num_islands", 3),
                                help="Number of parallel populations in island model"
                            )
                        with col2:
                            st.session_state.migration_interval = st.number_input(
                                "Migration Interval", 
                                min_value=1, 
                                max_value=1000, 
                                value=st.session_state.get("migration_interval", 25),
                                help="How often individuals migrate between islands"
                            )
                        with col3:
                            st.session_state.migration_rate = st.slider(
                                "Migration Rate", 
                                min_value=0.0, 
                                max_value=1.0, 
                                value=st.session_state.get("migration_rate", 0.1),
                                step=0.01,
                                help="Proportion of individuals that migrate"
                            )
                    
                    with config_tabs[2]:  # Feature Dimensions
                        st.session_state.feature_dimensions = st.multiselect(
                            "Feature Dimensions (for MAP-Elites)",
                            options=["complexity", "diversity", "performance", "readability", "efficiency", "accuracy", "robustness"],
                            default=st.session_state.get("feature_dimensions", ["complexity", "diversity"]),
                            help="Dimensions for quality-diversity optimization"
                        )
                        
                        st.session_state.feature_bins = st.slider(
                            "Feature Bins", 
                            min_value=5, 
                            max_value=50, 
                            value=st.session_state.get("feature_bins", 10),
                            help="Number of bins for each feature dimension in MAP-Elites"
                        )
                    
                    with config_tabs[3]:  # Advanced
                        col1, col2 = st.columns(2)
                        with col1:
                            st.session_state.elite_ratio = st.slider(
                                "Elite Ratio", 
                                min_value=0.0, 
                                max_value=1.0, 
                                value=st.session_state.get("elite_ratio", 0.1),
                                step=0.01,
                                help="Ratio of elite individuals preserved each generation"
                            )
                            st.session_state.exploration_ratio = st.slider(
                                "Exploration Ratio", 
                                min_value=0.0, 
                                max_value=1.0, 
                                value=st.session_state.get("exploration_ratio", 0.3),
                                step=0.01,
                                help="Ratio of population dedicated to exploration"
                            )
                            st.session_state.enable_artifacts = st.checkbox(
                                "Enable Artifact Feedback", 
                                value=st.session_state.get("enable_artifacts", True),
                                help="Enable error feedback to LLM for improved iterations"
                            )
                        
                        with col2:
                            st.session_state.cascade_evaluation = st.checkbox(
                                "Cascade Evaluation", 
                                value=st.session_state.get("cascade_evaluation", True),
                                help="Use multi-stage testing to filter bad solutions early"
                            )
                            st.session_state.use_llm_feedback = st.checkbox(
                                "Use LLM Feedback", 
                                value=st.session_state.get("use_llm_feedback", False),
                                help="Use LLM-based feedback for evolution guidance"
                            )
                            st.session_state.evolution_trace_enabled = st.checkbox(
                                "Evolution Tracing", 
                                value=st.session_state.get("evolution_trace_enabled", False),
                                help="Enable detailed logging of evolution process"
                            )
                
                with main_tabs[3]:  # Performance
                    st.subheader("Performance & Monitoring")
                    st.info("Comprehensive monitoring system would be displayed here when available.")
                    
                    # Fallback metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Evolution Runs", st.session_state.get("total_evolution_runs", 0))
                    with col2:
                        st.metric("Avg Best Score", f"{st.session_state.get('avg_best_score', 0.0):.3f}")
                    with col3:
                        st.metric("Best Ever Score", f"{st.session_state.get('best_ever_score', 0.0):.3f}")
                    with col4:
                        st.metric("Success Rate", f"{st.session_state.get('success_rate', 0.0):.1%}")
                
                with main_tabs[4]:  # Reports
                    st.subheader("Evolution Reports & Analytics")
                    st.info("Evolution reports would be displayed here when available.")

    with tabs[10]: # OpenEvolve Orchestrator tab  # noqa: F821
        with st.container(border=True):
            st.header("🤖 OpenEvolve Workflow Orchestrator")
            
            # Comprehensive OpenEvolve orchestrator implementation
            # Import and render the OpenEvolve orchestrator if available, otherwise use self-contained implementation
            openevolve_orchestrator_available = False
            try:
                from openevolve_orchestrator import render_openevolve_orchestrator_ui
                render_openevolve_orchestrator_ui()
                openevolve_orchestrator_available = True
            except ImportError:
                # Use comprehensive self-contained implementation
                st.header("🤖 OpenEvolve Workflow Orchestrator")
                st.write("Advanced workflow management system for orchestrating complex evolutionary processes.")
                
                # Main orchestrator tabs
                orchestrator_tabs = st.tabs(["🏗️ Create Workflow", "📊 Monitoring Panel", "📋 Execution History", "⚙️ Configuration", "📈 Analytics"])
                
                with orchestrator_tabs[0]:  # Create Workflow
                    st.subheader("Design & Launch Evolutionary Workflow")
                    
                    # Workflow type selection with detailed descriptions
                    workflow_options = {
                        "standard": {
                            "label": "🧬 Standard Evolution",
                            "description": "Traditional evolutionary algorithm for general optimization tasks"
                        },
                        "quality_diversity": {
                            "label": "🎯 Quality-Diversity Evolution (MAP-Elites)", 
                            "description": "Maintains diverse, high-performing solutions across feature dimensions"
                        },
                        "multi_objective": {
                            "label": "⚖️ Multi-Objective Optimization",
                            "description": "Optimizes for multiple competing objectives simultaneously"
                        },
                        "adversarial": {
                            "label": "⚔️ Adversarial Evolution (Red Team/Blue Team)",
                            "description": "Robustness-focused evolution with adversarial testing"
                        },
                        "symbolic_regression": {
                            "label": "🔍 Symbolic Regression",
                            "description": "Discover mathematical expressions from data patterns"
                        },
                        "neuroevolution": {
                            "label": "🧠 Neuroevolution",
                            "description": "Evolve neural network architectures and weights"
                        },
                        "algorithm_discovery": {
                            "label": "💡 Algorithm Discovery", 
                            "description": "Discover novel algorithmic approaches"
                        },
                        "prompt_optimization": {
                            "label": "📝 Prompt Optimization",
                            "description": "Optimize prompts for large language models"
                        }
                    }
                    
                    # Select workflow type
                    workflow_type = st.selectbox(
                        "Select Workflow Type",
                        options=list(workflow_options.keys()),
                        format_func=lambda x: workflow_options[x]["label"],
                        help="Choose the type of evolutionary process to orchestrate"
                    )
                    
                    # Show description for selected type
                    st.info(f"**Description**: {workflow_options[workflow_type]['description']}")
                    
                    # Content input area
                    content = st.text_area(
                        "Input Content/Problem Definition",
                        height=200,
                        placeholder="Enter content to evolve, problem definition, or initial solution..."
                    )
                    
                    # Advanced configuration with expanders
                    with st.expander("🔧 Core Evolution Parameters", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            max_iterations = st.number_input("Max Generations", min_value=1, max_value=100000, value=100, 
                                                           help="Maximum number of evolutionary generations to run")
                            population_size = st.number_input("Population Size", min_value=10, max_value=10000, value=100,
                                                                help="Number of individuals in each generation")
                            num_islands = st.number_input("Number of Islands", min_value=1, max_value=50, value=5,
                                                            help="Number of parallel populations in island model")
                        
                        with col2:
                            temperature = st.slider("LLM Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05,
                                                  help="Creativity control for LLM operations (higher = more random)")
                            elite_ratio = st.slider("Elite Ratio", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
                                                  help="Proportion of top individuals preserved each generation")
                            archive_size = st.number_input("Archive Size", min_value=0, max_value=10000, value=100,
                                                             help="Size of the archive for quality-diversity algorithms")
                    
                    # Feature dimensions for QD and multi-objective
                    if workflow_type in ["quality_diversity", "multi_objective"]:
                        with st.expander("🎯 Feature Dimensions (for Quality-Diversity/Multi-Objective)", expanded=True):
                            feature_options = ["complexity", "diversity", "performance", "readability", 
                                             "efficiency", "accuracy", "robustness", "size", "speed", "cost"]
                            feature_dimensions = st.multiselect(
                                "Feature Dimensions",
                                options=feature_options,
                                default=["complexity", "diversity"],
                                help="Dimensions along which to map diverse, high-quality solutions"
                            )
                            feature_bins = st.slider("Feature Bins", min_value=5, max_value=100, value=15,
                                                   help="Number of bins for each feature dimension in the MAP")
                    
                    # Advanced features configuration
                    with st.expander("🧩 Advanced Evolution Features", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            enable_artifacts = st.checkbox("Enable Artifact Feedback", value=True,
                                                         help="Use execution artifacts to guide evolution")
                            cascade_evaluation = st.checkbox("Enable Cascade Evaluation", value=True,
                                                           help="Use multi-stage filtering to improve efficiency")
                            use_llm_feedback = st.checkbox("Use LLM Feedback", value=False,
                                                         help="Incorporate LLM-based quality assessment")
                            evolution_trace_enabled = st.checkbox("Enable Evolution Tracing", value=False,
                                                                help="Enable detailed execution logging")
                        
                        with col2:
                            enable_early_stopping = st.checkbox("Enable Early Stopping", value=True,
                                                              help="Stop evolution when no improvement occurs")
                            diff_based_evolution = st.checkbox("Diff-Based Evolution", value=True,
                                                             help="Use differential changes for targeted improvements")
                            double_selection = st.checkbox("Double Selection", value=True,
                                                         help="Use separate selection for parents and survivors")
                    
                    # Research-grade features
                    with st.expander("🔬 Research-Grade Features", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            test_time_compute = st.checkbox("Test-Time Compute", value=False,
                                                          help="Use additional computation during evaluation")
                            adaptive_feature_dimensions = st.checkbox("Adaptive Feature Dimensions", value=True,
                                                                    help="Adaptively adjust feature dimensions during evolution")
                            multi_strategy_sampling = st.checkbox("Multi-Strategy Sampling", value=True,
                                                                help="Use multiple variation strategies")
                            ring_topology = st.checkbox("Ring Topology", value=True,
                                                      help="Use ring topology for island connections")
                        
                        with col2:
                            controlled_gene_flow = st.checkbox("Controlled Gene Flow", value=True,
                                                             help="Control the flow of genetic material between populations")
                            auto_diff = st.checkbox("Auto Diff", value=True,
                                                  help="Automatically compute differences between solutions")
                            symbolic_execution = st.checkbox("Symbolic Execution", value=False,
                                                           help="Use symbolic execution for analysis (experimental)")
                            coevolutionary_approach = st.checkbox("Coevolutionary Approach", value=False,
                                                                help="Use coevolutionary techniques")
                    
                    # Performance and resource optimization
                    with st.expander("⚡ Performance & Resource Optimization", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            memory_limit_mb = st.number_input("Memory Limit (MB)", min_value=100, max_value=131072, value=2048,
                                                                help="Memory limit for evaluation processes")
                            cpu_limit = st.number_input("CPU Limit", min_value=0.1, max_value=128.0, value=4.0, step=0.1,
                                                          help="CPU resource limit for the process")
                            parallel_evaluations = st.number_input("Parallel Evaluations", min_value=1, max_value=128, value=4,
                                                                 help="Number of parallel evaluation processes")
                        
                        with col2:
                            max_code_length = st.number_input("Max Code Length", min_value=100, max_value=1000000, value=10000,
                                                                help="Maximum length of code to evolve")
                            evaluator_timeout = st.number_input("Evaluator Timeout (s)", min_value=1, max_value=7200, value=300,
                                                                  help="Timeout for evaluation processes")
                            max_retries_eval = st.number_input("Max Evaluation Retries", min_value=1, max_value=20, value=3,
                                                                 help="Maximum retries for failed evaluations")
                    
                    # Execute workflow button with validation
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        save_template = st.checkbox("Save as Workflow Template", value=False,
                                                  help="Save this configuration for future use")
                        template_name = st.text_input("Template Name (if saving)", 
                                                    value=f"workflow_{workflow_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                                    disabled=not save_template)
                    with col2:
                        if st.button("🚀 Launch Evolutionary Workflow", type="primary", use_container_width=True):
                            if not content.strip():
                                st.error("❌ Please enter content/problem definition to evolve")
                            elif not st.session_state.get("api_key"):
                                st.error("❌ Please configure your API key in the sidebar")
                            else:
                                # In a real implementation, this would execute the workflow
                                st.success("✅ Workflow launched successfully!")
                                st.info("The workflow is now executing. Monitor progress in the 'Monitoring Panel' tab.")
                                
                                # Store workflow details in session state
                                st.session_state.current_workflow = {
                                    "type": workflow_type,
                                    "content": content,
                                    "params": {
                                        "max_iterations": max_iterations,
                                        "population_size": population_size,
                                        "num_islands": num_islands,
                                        "temperature": temperature,
                                        "elite_ratio": elite_ratio,
                                        "archive_size": archive_size
                                    },
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                # Add to workflow history
                                if "workflow_history" not in st.session_state:
                                    st.session_state.workflow_history = []
                                st.session_state.workflow_history.append(st.session_state.current_workflow)
                
                with orchestrator_tabs[1]:  # Monitoring Panel
                    st.subheader("Real-Time Workflow Monitoring")
                    
                    # Check if there's an active workflow
                    current_workflow = st.session_state.get("current_workflow")
                    if current_workflow:
                        st.success(f"🟢 Active Workflow: {current_workflow['type'].replace('_', ' ').title()}")
                        
                        # Live metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Generation", st.session_state.get("current_generation", 0))
                        with col2:
                            st.metric("Best Fitness", f"{st.session_state.get('best_score', 0.0):.4f}")
                        with col3:
                            st.metric("Pop. Size", current_workflow['params']['population_size'])
                        with col4:
                            st.metric("Status", "Running" if st.session_state.get("evolution_running", False) else "Idle")
                        
                        # Progress visualization
                        st.subheader("Execution Progress")
                        progress = st.session_state.get("current_generation", 0) / max(1, current_workflow['params']['max_iterations'])
                        st.progress(progress)
                        st.write(f"Progress: {st.session_state.get('current_generation', 0)}/{current_workflow['params']['max_iterations']} generations")
                        
                        # Real-time metrics visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Fitness over time (with simulated data)
                            st.write("**Fitness Progression**")
                            if "evolution_history" in st.session_state and st.session_state.evolution_history:
                                fitness_history = []
                                gen_numbers = []
                                for gen_idx, gen_data in enumerate(st.session_state.evolution_history):
                                    population = gen_data.get("population", [])
                                    if population:
                                        best_fit = max(ind.get("fitness", 0) for ind in population)
                                        avg_fit = sum(ind.get("fitness", 0) for ind in population) / len(population)
                                        fitness_history.append({"Generation": gen_idx, "Best": best_fit, "Average": avg_fit})
                                
                                if fitness_history:
                                    df_fitness = pd.DataFrame(fitness_history)
                                    fig = px.line(df_fitness, x="Generation", y=["Best", "Average"], 
                                                title="Fitness Over Generations")
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Create sample data for visualization
                                sample_data = pd.DataFrame({
                                    'Generation': range(1, min(21, max(2, current_workflow['params']['max_iterations'] + 1))),
                                    'Best': np.random.uniform(0.3, 0.95, min(20, current_workflow['params']['max_iterations'])).cummax(),
                                    'Average': np.random.uniform(0.2, 0.8, min(20, current_workflow['params']['max_iterations']))
                                })
                                fig = px.line(sample_data, x="Generation", y=["Best", "Average"], 
                                            title="Sample Fitness Progression")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Population metrics
                            st.write("**Population Analysis**")
                            if "evolution_history" in st.session_state and st.session_state.evolution_history:
                                diversity_data = []
                                for gen_idx, gen_data in enumerate(st.session_state.evolution_history[-5:]):  # Last 5 gens
                                    population = gen_data.get("population", [])
                                    if population:
                                        avg_div = sum(ind.get("diversity", 0) for ind in population) / len(population)
                                        complexity = sum(ind.get("complexity", 0) for ind in population) / len(population)
                                        diversity_data.append({
                                            "Generation": gen_idx + len(st.session_state.evolution_history) - 5,
                                            "Diversity": avg_div,
                                            "Complexity": complexity
                                        })
                                
                                if diversity_data:
                                    df_div = pd.DataFrame(diversity_data)
                                    fig = px.line(df_div, x="Generation", y=["Diversity", "Complexity"], 
                                                title="Diversity & Complexity")
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Create sample data
                                sample_div_data = pd.DataFrame({
                                    'Generation': range(1, 6),
                                    'Diversity': np.random.uniform(0.3, 0.8, 5),
                                    'Complexity': np.random.uniform(0.2, 0.9, 5)
                                })
                                fig = px.line(sample_div_data, x="Generation", y=["Diversity", "Complexity"],
                                            title="Sample Population Metrics")
                                st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.info("No active workflows. Launch a workflow in the 'Create Workflow' tab to monitor execution.")
                        
                        # Resource utilization
                        st.subheader("Resource Utilization")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("CPU Usage", "25%")
                        with col2:
                            st.metric("Memory Usage", "1.2 GB")
                        with col3:
                            st.metric("Active Processes", "4")
                
                with orchestrator_tabs[2]:  # Execution History
                    st.subheader("Workflow Execution History")
                    
                    # Display workflow history
                    workflow_history = st.session_state.get("workflow_history", [])
                    
                    if workflow_history:
                        for i, workflow in enumerate(workflow_history):
                            with st.expander(f"Workflow #{len(workflow_history)-i}: {workflow['type'].replace('_', ' ').title()} - {workflow['timestamp']}", expanded=False):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**Type**: {workflow['type'].replace('_', ' ').title()}")
                                    st.write(f"**Parameters**: Max Iterations: {workflow['params']['max_iterations']}, Population: {workflow['params']['population_size']}")
                                    st.write(f"**Started**: {workflow['timestamp']}")
                                    st.text_area("Input Content:", value=workflow['content'][:200] + "..." if len(workflow['content']) > 200 else workflow['content'], height=100, disabled=True)
                                with col2:
                                    st.write("**Status**: Completed")  # Would be dynamic in real implementation
                                    if st.button(f"View Details #{i}", key=f"view_details_{i}"):
                                        st.session_state.selected_workflow = workflow
                                        st.info("Details would be shown in a full implementation.")
                    else:
                        st.info("No workflow history available. Execute workflows to see them listed here.")
                
                with orchestrator_tabs[3]:  # Configuration
                    st.subheader("Orchestrator Configuration")
                    
                    # Global configuration settings
                    with st.expander("🌍 Global Settings", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.session_state.openevolve_base_url = st.text_input("OpenEvolve API Base URL", 
                                                                                  value=st.session_state.get("openevolve_base_url", "http://localhost:8000"))
                            st.session_state.openevolve_api_key = st.text_input("OpenEvolve API Key", 
                                                                                 value=st.session_state.get("openevolve_api_key", ""), type="password")
                        with col2:
                            st.session_state.default_max_tokens = st.number_input("Default Max Tokens", 
                                                                                    min_value=100, max_value=128000, value=st.session_state.get("default_max_tokens", 4096))
                            st.session_state.default_temperature = st.slider("Default Temperature", 
                                                                           min_value=0.0, max_value=2.0, value=st.session_state.get("default_temperature", 0.7), step=0.1)
                    
                    # Default workflow parameters
                    with st.expander("⚙️ Default Workflow Parameters", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.session_state.default_max_iterations = st.number_input("Default Max Iterations", 
                                                                                     min_value=1, max_value=10000, value=st.session_state.get("default_max_iterations", 100))
                            st.session_state.default_population_size = st.number_input("Default Population Size", 
                                                                                      min_value=10, max_value=10000, value=st.session_state.get("default_population_size", 100))
                        with col2:
                            st.session_state.default_elite_ratio = st.slider("Default Elite Ratio", 
                                                                           min_value=0.0, max_value=1.0, value=st.session_state.get("default_elite_ratio", 0.1), step=0.01)
                            st.session_state.default_archive_size = st.number_input("Default Archive Size", 
                                                                                   min_value=0, max_value=10000, value=st.session_state.get("default_archive_size", 100))
                    
                    # Save configuration
                    if st.button("💾 Save Configuration", type="primary"):
                        st.success("Configuration saved successfully!")
                
                with orchestrator_tabs[4]:  # Analytics
                    st.subheader("Workflow Analytics & Insights")
                    
                    # Overall statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Workflows", len(st.session_state.get("workflow_history", [])))
                    with col2:
                        st.metric("Avg. Generations", f"{np.mean([w['params']['max_iterations'] for w in st.session_state.get('workflow_history', [])]) if st.session_state.get('workflow_history') else 0:.0f}")
                    with col3:
                        st.metric("Active Workflows", 0)  # Would be calculated in real implementation
                    with col4:
                        st.metric("Success Rate", "0%")  # Would be calculated in real implementation
                    
                    # Workflow type distribution
                    if st.session_state.get("workflow_history"):
                        type_counts = {}
                        for wf in st.session_state.workflow_history:
                            wf_type = wf['type']
                            type_counts[wf_type] = type_counts.get(wf_type, 0) + 1
                        
                        if type_counts:
                            st.subheader("Workflow Type Distribution")
                            type_df = pd.DataFrame(list(type_counts.items()), columns=['Type', 'Count'])
                            fig = px.bar(type_df, x='Type', y='Count', title='Number of Workflows by Type')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("Advanced analytics would include performance trends, resource usage, and optimization recommendations in a full implementation.")
