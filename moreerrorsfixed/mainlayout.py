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
from dataclasses import asdict

from pyvis.network import Network
# These imports are assumed to exist in the user's environment.
# If they don't, the script will fail, but per the instructions, no mock functions will be created.
from session_utils import _safe_list, _load_user_preferences, _load_parameter_settings

# Import autorefresh for real-time updates
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

def _safe_list(data: dict, key: str) -> List:
    """Safely retrieves a list from a dictionary, returning an empty list if the key is not found or the value is not a list."""
    value = data.get(key)
    if isinstance(value, list):
        return value
    return []

from providercatalogue import get_openrouter_models, _parse_price_per_million

from session_manager import (
    APPROVAL_PROMPT, RED_TEAM_CRITIQUE_PROMPT, BLUE_TEAM_PATCH_PROMPT
)
from openevolve_integration import (
    OpenEvolveAPI, create_advanced_openevolve_config
)

from adversarial import (
    run_adversarial_testing, generate_html_report, generate_pdf_report, generate_docx_report,
    generate_latex_report, generate_compliance_report, optimize_model_selection, _load_human_feedback,
    MODEL_META_LOCK, MODEL_META_BY_ID
)
from integrations import (
    create_github_branch, commit_to_github,
    list_linked_github_repositories, send_discord_notification, send_msteams_notification, send_generic_webhook
)
from tasks import create_task, get_tasks, update_task
from suggestions import get_content_classification_and_tags, predict_improvement_potential, check_security_vulnerabilities
from rbac import ROLES, assign_role
from content_manager import content_manager
from analytics_manager import analytics_manager

HAS_STREAMLIT_TAGS = True

def _stream_evolution_logs_in_thread(evolution_id, api, thread_lock):
    while True:
        with thread_lock:
            if not st.session_state.evolution_running:
                break
            if st.session_state.evolution_stop_flag: # Check stop flag
                st.session_state.evolution_running = False
                st.session_state.evolution_stop_flag = False
                break

        status = api.get_evolution_status(evolution_id)
        if status:
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
            const presenceContainer = document.getElementById("presence-container");
            const notificationCenter = document.getElementById("notification-center");
            const websocket = new WebSocket("ws://localhost:8765");

            websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === "presence_update") {
                    presenceContainer.innerHTML = "";
                    data.payload.forEach(user => {
                        const indicator = document.createElement("div");
                        indicator.className = "presence-indicator";
                        indicator.title = user.id;
                        presenceContainer.appendChild(indicator);
                    });
                } else if (data.type === "notification") {
                    const notification = document.createElement("div");
                    notification.className = "notification";
                    notification.innerText = data.payload.message;
                    notificationCenter.appendChild(notification);
                    notificationCenter.style.display = "block";
                } else if (data.type === "cursor_update") {
                    const editor = document.querySelector('.stTextArea textarea');
                    let cursor = document.getElementById(`cursor-${data.sender}`);
                    if (!cursor) {
                        cursor = document.createElement('div');
                        cursor.id = `cursor-${data.sender}`;
                        cursor.className = 'other-cursor';
                        document.body.appendChild(cursor);
                    }
                    cursor.style.left = `${data.payload.x}px`;
                    cursor.style.top = `${data.payload.y}px`;
                } else if (data.type === "text_update") {
                    const editor = document.querySelector('.stTextArea textarea');
                    if (editor.value !== data.payload.text) {
                        editor.value = data.payload.text;
                    }
                }
            };

            const textArea = document.querySelector('[data-testid="stTextAreawithLabel"] textarea');
            if (textArea) {
                textArea.addEventListener('input', (event) => {
                    const text_update = {
                        type: "text_update",
                        payload: {
                            text: event.target.value
                        }
                    };
                    websocket.send(JSON.stringify(text_update));
                });

                textArea.addEventListener('mousemove', (event) => {
                    const cursor_update = {
                        type: "cursor_update",
                        payload: {
                            x: event.clientX,
                            y: event.clientY
                        }
                    };
                    websocket.send(JSON.stringify(cursor_update));
                });
            }

            websocket.onopen = () => {
                const presenceData = {
                    type: "update_presence",
                    payload: {
                        id: Math.random().toString(36).substring(7)
                    }
                };
                websocket.send(JSON.stringify(presenceData));
            };

            document.addEventListener("click", (event) => {
                if (!notificationCenter.contains(event.target)) {
                    notificationCenter.style.display = "none";
                }
            });
        </script>
        """, unsafe_allow_html=True)
        st.session_state.collaboration_ui_rendered = True

from session_manager import session_defaults
from prompt_manager import PromptManager
from template_manager import TemplateManager
from content_manager import content_manager
from analytics_manager import AnalyticsManager
from collaboration_manager import CollaborationManager
from version_control import VersionControl
# from rbac import RBAC
from notifications import NotificationManager
from log_streaming import LogStreaming
# from session_utils import get_current_session_id
from session_state_classes import State, SessionManager
from sidebar import get_default_generation_params, get_default_evolution_params

def _initialize_session_state():
    """Initialize session state with default values."""
    defaults = {
        "theme": "light",
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
        "exploitation_ratio": 0.0,
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
        "archive_size": 0,
        "elite_ratio": 1.0,
        "exploration_ratio": 0.0,
        "exploitation_ratio": 0.0,
        "checkpoint_interval": 5,
        "evolution_stop_flag": False,
        "adversarial_stop_flag": False,
        "human_feedback_log": _load_human_feedback(),
        "user_preferences": _load_user_preferences(),
        "parameter_settings": _load_parameter_settings(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if "styles_css" not in st.session_state:
        try:
            with open("styles.css") as f:
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

def _stream_evolution_logs_in_thread(evolution_id, api, thread_lock):
    full_log = []
    for log_chunk in api.stream_evolution_logs(evolution_id):
        full_log.append(log_chunk)
        with thread_lock:
            st.session_state.evolution_log = full_log.copy()
    # Once streaming is complete, set evolution_running to False
    with thread_lock:
        st.session_state.evolution_running = False

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
            st.session_state.openevolve_api_instance = OpenEvolveAPI(
                base_url=st.session_state.get("openevolve_base_url", "http://localhost:8000"),
                api_key=st.session_state.get("openevolve_api_key", ""),
            )
        st.session_state.prompt_manager = PromptManager(api=st.session_state.openevolve_api_instance)
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

    # Assign references for easier access within the function
    session_manager = st.session_state.session_manager
    prompt_manager = st.session_state.prompt_manager
    template_manager = st.session_state.template_manager
    content_manager_instance = st.session_state.content_manager_instance
    analytics_manager_instance = st.session_state.analytics_manager_instance
    collaboration_manager = st.session_state.collaboration_manager
    version_control = st.session_state.version_control

    notification_manager = st.session_state.notification_manager
    log_streaming = st.session_state.log_streaming

    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []
    render_collaboration_ui()
    check_password()
    # Apply theme-specific CSS with animations
    current_theme = st.session_state.get("theme", "light")
    
    if "styles_css" not in st.session_state:
        with open("styles.css") as f:
            st.session_state.styles_css = f.read()
    st.markdown(f"<style>{st.session_state.styles_css}</style>", unsafe_allow_html=True)

    
    st.markdown(
        '<h2 style="text-align: center;">🧬 OpenEvolve Content Improver</h2>'
        '<p style="text-align: center; font-size: 1.2rem;">AI-Powered Content Hardening with Multi-LLM Consensus</p>',
        unsafe_allow_html=True)
    # Inject JavaScript to set data-theme attribute on html element
    st.markdown(
        f"""
        <script>
            document.documentElement.setAttribute('data-theme', '{current_theme}');
        </script>
        """,
        unsafe_allow_html=True
    )

    if "styles_css" not in st.session_state:
        try:
            with open("styles.css") as f:
                st.session_state.styles_css = f.read()
        except FileNotFoundError:
            st.session_state.styles_css = ""
    st.markdown(f"<style>{st.session_state.styles_css}</style>", unsafe_allow_html=True)

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


        if st.button("Back to Main Tabs", key="back_to_main_tabs_analytics"):
            st.session_state.page = None
            st.rerun()
    else:
        tab_names = ["🧬 Evolution", "⚔️ Adversarial Testing", "🐙 GitHub", "📜 Activity Feed", "📊 Report Templates", "🤖 Model Dashboard", "✅ Tasks", "👑 Admin", "📂 Projects"]
        tabs = st.tabs(tab_names)

        with tabs[0]: # Evolution tab
            with st.container(border=True): # Wrap the entire tab content in a container
                st.header("Real-time Evolution")
                st.markdown("Iteratively improve your content using a single AI model.")

            with st.expander("📝 Content Input", expanded=True):
                # Initialize protocol_text in session_state if not already present
                if "protocol_text" not in st.session_state:
                    st.session_state.protocol_text = "# Sample Protocol\n\nThis is a sample protocol for testing purposes."

                st.text_area("Paste your draft content here:", height=300, key="protocol_text",
                             value=st.session_state.protocol_text,
                             disabled=st.session_state.evolution_running)

                templates = content_manager.list_protocol_templates()
                if templates:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        selected_template = st.selectbox("Load Template", [""] + templates, key="load_template_select")
                    with col2:
                        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                        # Define callback to update protocol text
                        def load_template_callback():
                            selected_template = st.session_state.load_template_select
                            if selected_template:
                                template_content = content_manager.load_protocol_template(selected_template)
                                st.session_state.protocol_text = template_content
                                st.success(f"Loaded template: {selected_template}")

                        if selected_template:
                            st.button("Load", key="load_template_btn", on_click=load_template_callback,
                                                                                           use_container_width=True, type="secondary")
            st.divider() # Add a divider

            with st.expander("🎮 Action Controls", expanded=True):
                c1, c2, c3 = st.columns(3)
                run_button = c1.button("🚀 Start Evolution", type="primary", disabled=st.session_state.evolution_running,
                                       use_container_width=True)
                stop_button = c2.button("⏹️ Stop Evolution", disabled=not st.session_state.evolution_running, type="secondary",
                                        use_container_width=True)
                if c3.button("🔄 Resume Evolution", use_container_width=True, type="secondary"):
                    if st.session_state.evolution_id and not st.session_state.evolution_running:
                        st.session_state.evolution_running = True
                        st.session_state.evolution_stop_flag = False # Ensure flag is reset
                        api = OpenEvolveAPI(base_url=st.session_state.openevolve_base_url, api_key=st.session_state.openevolve_api_key)
                        threading.Thread(target=_stream_evolution_logs_in_thread,
                                         args=(st.session_state.evolution_id, api, st.session_state.thread_lock)).start()
                        st.info("Evolution resumed.")
                        st.rerun()
                    elif st.session_state.evolution_running:
                        st.warning("Evolution is already running.")
                    else:
                        st.warning("No evolution to resume. Please start a new evolution.")
            st.divider() # Add a divider

            if run_button:
                st.session_state.evolution_running = True
                api = OpenEvolveAPI(base_url=st.session_state.openevolve_base_url, api_key=st.session_state.openevolve_api_key)
                config = create_advanced_openevolve_config(
                    model_name=st.session_state.model,
                    api_key=st.session_state.api_key,
                    api_base=st.session_state.base_url,
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                    max_tokens=st.session_state.max_tokens,
                    max_iterations=st.session_state.max_iterations,
                    population_size=st.session_state.population_size,
                    num_islands=st.session_state.num_islands,
                    archive_size=st.session_state.archive_size,
                    elite_ratio=st.session_state.elite_ratio,
                    exploration_ratio=st.session_state.exploration_ratio,
                    exploitation_ratio=st.session_state.exploitation_ratio,
                    checkpoint_interval=st.session_state.checkpoint_interval,
                )
                if config is not None:
                    evolution_id = api.start_evolution(config=asdict(config))
                    if evolution_id:
                        st.session_state.evolution_id = evolution_id
                        # Start log streaming in a separate thread
                        threading.Thread(target=_stream_evolution_logs_in_thread,
                                         args=(evolution_id, api, st.session_state.thread_lock)).start()
                    st.rerun()
                else:
                    st.error("Failed to create OpenEvolve configuration. Please check your settings.")

            if stop_button:
                st.session_state.evolution_stop_flag = True


            c1, c2, c3 = st.columns(3)
            with c1:
                classify_button = st.button("🏷️ Classify and Tag", use_container_width=True, type="secondary")
            with c2:
                predict_button = st.button("🔮 Predict Improvement Potential", use_container_width=True, type="secondary")
            with c3:
                security_button = st.button("🛡️ Check Security", use_container_width=True, type="secondary")
            st.divider() # Add a divider

            with st.expander("↔️ Compare Generations"):
                if not st.session_state.evolution_history:
                    st.info("Run an evolution to generate versions to compare.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        generation1 = st.selectbox("Select Generation 1", range(len(st.session_state.evolution_history)))
                    with col2:
                        generation2 = st.selectbox("Select Generation 2", range(len(st.session_state.evolution_history)))
                    if st.button("Compare", type="secondary"):
                        text1 = st.session_state.evolution_history[generation1]['population'][0]['code']
                        text2 = st.session_state.evolution_history[generation2]['population'][0]['code']
                        render_code_diff(text1, text2)
            st.divider()

            if "suggestions" in st.session_state and st.session_state.suggestions:
                with st.expander("💡 Suggestions", expanded=True):
                    for suggestion in st.session_state.suggestions:
                        st.markdown(f"- {suggestion}")
                st.divider()

            if classify_button:
                with st.spinner("Classifying and tagging..."):
                    classification_and_tags = get_content_classification_and_tags(st.session_state.protocol_text)
                    st.session_state.classification_and_tags = classification_and_tags

            if "classification_and_tags" in st.session_state and st.session_state.classification_and_tags:
                with st.expander("🏷️ Classification and Tags", expanded=True):
                    st.write(f"**Classification:** {st.session_state.classification_and_tags.get('classification')}")
                    st.write(f"**Tags:** {', '.join(st.session_state.classification_and_tags.get('tags', []))}")
                st.divider()

            if predict_button:
                with st.spinner("Predicting improvement potential..."):
                    potential = predict_improvement_potential(st.session_state.protocol_text)
                    st.session_state.improvement_potential = potential

            if "improvement_potential" in st.session_state and st.session_state.improvement_potential is not None:
                st.metric("Improvement Potential", f"{st.session_state.improvement_potential:.2%}")
                st.divider()

            if security_button:
                with st.spinner("Checking for security vulnerabilities..."):
                    vulnerabilities = check_security_vulnerabilities(st.session_state.protocol_text)
                    st.session_state.vulnerabilities = vulnerabilities

            if "vulnerabilities" in st.session_state and st.session_state.vulnerabilities:
                with st.expander("🛡️ Security Vulnerabilities", expanded=True):
                    for vulnerability in st.session_state.vulnerabilities:
                        st.warning(vulnerability)
                st.divider()





            with st.expander("⚙️ Advanced Settings", expanded=False):
                st.markdown("### 🎛️ Evolution Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("Max Iterations", 1, 200, st.session_state.max_iterations, key="max_iterations_mainlayout")
                    st.number_input("Population Size", 1, 100, st.session_state.population_size, key="population_size_mainlayout")
                    st.number_input("Number of Islands", 1, 10, st.session_state.num_islands, key="num_islands_mainlayout")
                    st.slider("Elite Ratio", 0.0, 1.0, st.session_state.elite_ratio, 0.1, key="elite_ratio_mainlayout")
                with col2:
                    st.number_input("Checkpoint Interval", 1, 100, st.session_state.checkpoint_interval, key="checkpoint_interval_mainlayout")
                    st.slider("Exploration Ratio", 0.0, 1.0, st.session_state.exploration_ratio, 0.1, key="exploration_ratio_mainlayout")
                    st.slider("Exploitation Ratio", 0.0, 1.0, st.session_state.exploitation_ratio, 0.1, key="exploitation_ratio_mainlayout")
                    st.number_input("Archive Size", 0, 100, st.session_state.archive_size, key="archive_size_mainlayout")

                st.markdown("### 🤖 Model Parameters")
                col3, col4 = st.columns(2)
                with col3:
                    st.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.1, key="temperature_mainlayout")
                    st.slider("Top-P", 0.0, 1.0, st.session_state.top_p, 0.1, key="top_p_mainlayout")
                with col4:
                    st.slider("Frequency Penalty", -2.0, 2.0, st.session_state.frequency_penalty, 0.1, key="frequency_penalty_mainlayout")
                    st.slider("Presence Penalty", -2.0, 2.0, st.session_state.presence_penalty, 0.1, key="presence_penalty_mainlayout")

                st.markdown("### 🎯 Multi-Objective Evolution")
                st.info("Define multiple objectives for the evolution. The fitness of each individual will be a vector of scores, one for each objective.")
                if HAS_STREAMLIT_TAGS:
                    st_tags(
                        label='Feature Dimensions:',
                        text='Press enter to add more',
                        value=st.session_state.feature_dimensions,
                        suggestions=['complexity', 'diversity', 'readability', 'performance', 'security', 'cost'],
                        key='feature_dimensions_mainlayout')
                st.number_input("Feature Bins", 1, 100, st.session_state.feature_bins, key="feature_bins_mainlayout")

                st.number_input("Number of Islands", 1, 10, st.session_state.num_islands, key="num_islands_multi_objective_mainlayout")
                st.slider("Migration Interval", 0, 100, st.session_state.migration_interval, key="migration_interval_mainlayout")
                st.slider("Migration Rate", 0.0, 1.0, st.session_state.migration_rate, 0.05, key="migration_rate_mainlayout")

                if st.button("Apply Advanced Settings", key="apply_advanced_settings_btn", type="primary"):
                    # Update parameter_settings with current values from session_state
                    st.session_state.parameter_settings["global"]["evolution"]["max_iterations"] = st.session_state.max_iterations_mainlayout
                    st.session_state.parameter_settings["global"]["evolution"]["population_size"] = st.session_state.population_size_mainlayout
                    st.session_state.parameter_settings["global"]["evolution"]["num_islands"] = st.session_state.num_islands_mainlayout
                    st.session_state.parameter_settings["global"]["evolution"]["elite_ratio"] = st.session_state.elite_ratio_mainlayout
                    st.session_state.parameter_settings["global"]["evolution"]["checkpoint_interval"] = st.session_state.checkpoint_interval_mainlayout
                    st.session_state.parameter_settings["global"]["evolution"]["exploration_ratio"] = st.session_state.exploration_ratio_mainlayout
                    st.session_state.parameter_settings["global"]["evolution"]["exploitation_ratio"] = st.session_state.exploitation_ratio_mainlayout
                    st.session_state.parameter_settings["global"]["evolution"]["archive_size"] = st.session_state.archive_size_mainlayout
                    st.session_state.parameter_settings["global"]["generation"]["temperature"] = st.session_state.temperature_mainlayout
                    st.session_state.parameter_settings["global"]["generation"]["top_p"] = st.session_state.top_p_mainlayout
                    st.session_state.parameter_settings["global"]["generation"]["frequency_penalty"] = st.session_state.frequency_penalty_mainlayout
                    st.session_state.parameter_settings["global"]["generation"]["presence_penalty"] = st.session_state.presence_penalty_mainlayout
                    st.session_state.parameter_settings["global"]["evolution"]["feature_dimensions"] = st.session_state.feature_dimensions_mainlayout
                    st.session_state.parameter_settings["global"]["evolution"]["feature_bins"] = st.session_state.feature_bins_mainlayout
                    st.session_state.parameter_settings["global"]["evolution"]["num_islands_multi_objective"] = st.session_state.num_islands_multi_objective_mainlayout # Corrected key
                    st.session_state.parameter_settings["global"]["evolution"]["migration_interval"] = st.session_state.migration_interval_mainlayout
                    st.session_state.parameter_settings["global"]["evolution"]["migration_rate"] = st.session_state.migration_rate_mainlayout
                    st.success("Advanced settings applied and saved to session state.")
                    st.rerun()
            st.divider()

            with st.expander("📊 Results", expanded=True):
                left, right = st.columns(2)
                with left:
                    st.subheader("📄 Current Best Content")
                    proto_out = st.empty()

                with right:
                    st.subheader("🔍 Logs")
                    log_out = st.empty()

                if st.session_state.evolution_running and st.session_state.evolution_id:
                    api = OpenEvolveAPI(base_url=st.session_state.openevolve_base_url, api_key=st.session_state.openevolve_api_key)
                    status = api.get_evolution_status(st.session_state.evolution_id)
                    if status:
                        proto_out.markdown(f"**Status:** {status['status']}\\n\\n**Best Score:** {status['best_score']}")
                        
                        # Display current log from session state
                        with st.session_state.thread_lock:
                            log_out.code("\\n".join(st.session_state.evolution_log), language="text")

                    # Check if evolution has completed (this will be updated by the thread)
                    if status.get('status') == 'completed':
                        # Only perform final actions once
                        if "evolution_finalized" not in st.session_state or not st.session_state.evolution_finalized:
                            st.session_state.evolution_finalized = True
                            previous_best = st.session_state.evolution_current_best
                            # Fetch final results after completion
                            best_solution = api.get_best_solution(st.session_state.evolution_id)
                            st.session_state.evolution_current_best = best_solution['code']
                            render_code_diff(previous_best, st.session_state.evolution_current_best)
                            history = api.get_evolution_history(st.session_state.evolution_id)
                            st.session_state.evolution_history = history
                            if history:
                                render_evolution_history_chart(history)
                                if len(history[-1].get('islands', [])) > 1:
                                    render_island_model_chart(history)
                            artifacts = api.get_artifacts(st.session_state.evolution_id)
                            if artifacts:
                                st.subheader("Artifacts")
                                for artifact in artifacts:
                                    st.download_button(artifact['name'], api.download_artifact(st.session_state.evolution_id, artifact['name']), artifact['name'])
                            st.balloons()
                            # Refresh the UI after completion
                            st.rerun()
                        else:
                            # If already finalized, just display the content
                            pass # Content is already displayed above

                    # Manual refresh button for intermediate updates
                    if st.button("🔄 Refresh Status", use_container_width=True, type="secondary"):
                        st.rerun()
                else:
                    with st.session_state.thread_lock:
                        current_log = "\\n".join(st.session_state.evolution_log)
                        current_content = st.session_state.evolution_current_best or st.session_state.protocol_text

                    log_out.code(current_log, language="text")
                    proto_out.code(current_content, language="markdown")

    with tabs[1]: # Adversarial Testing tab
        st.header("Adversarial Testing with Multi-LLM Consensus")
        st.markdown(
            "> **How it works:** Adversarial Testing uses two teams of AI models to improve your content:\n"
            "> - **🔴 Red Team** finds flaws and vulnerabilities.\n"
            "> - **🔵 Blue Team** fixes the identified issues.\n"
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
                    protocol_text = st.text_area("✏️ Enter or paste your content:",
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
                            st.session_state.protocol_text = content_manager.load_protocol_template(selected_template)
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
                if st.session_state.protocol_text.strip():
                    st.button("🗑️ Clear", use_container_width=True, type="secondary", on_click=clear_content_callback)

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
            col1, col2 = st.columns(2)
            start_button = col1.button("🚀 Start Adversarial Testing", type="primary", disabled=st.session_state.adversarial_running or not st.session_state.protocol_text.strip(), use_container_width=True)
            stop_button = col2.button("⏹️ Stop Adversarial Testing", disabled=not st.session_state.adversarial_running, type="secondary", use_container_width=True)

            if start_button:
                st.session_state.adversarial_running = True
                threading.Thread(target=run_adversarial_testing).start()
                st.rerun()
            if stop_button:
                st.session_state.adversarial_stop_flag = True

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
                        log_content = "\n".join(st.session_state.adversarial_log[-50:])
                        st.text_area("Activity Log", value=log_content, height=300, key="adversarial_log_display", disabled=True)

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
                    html_content = generate_html_report(results, st.session_state.custom_css)
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
                        else:
                            st.warning("Discord webhook URL not configured.")
                    if int_col2.button("💬 Teams", use_container_width=True, type="secondary"):
                        if st.session_state.msteams_webhook_url:
                            send_msteams_notification(st.session_state.msteams_webhook_url, f"Adversarial testing complete! Final approval: {results.get('final_approval_rate', 0.0):.1f}%")
                        else:
                            st.warning("Microsoft Teams webhook URL not configured.")
                    if int_col3.button("🚀 Webhook", use_container_width=True, type="secondary"):
                        if st.session_state.generic_webhook_url:
                            send_generic_webhook(st.session_state.generic_webhook_url, {"text": f"Adversarial testing complete! Final approval: {results.get('final_approval_rate', 0.0):.1f}%"})
                        else:
                            st.warning("Generic webhook URL not configured.")


    with tabs[2]: # GitHub tab
        with st.container(border=True):
            st.header("🐙 GitHub Integration")

            if not st.session_state.get("github_token"):
                st.warning("Please authenticate with GitHub in the sidebar first.")

            linked_repos = list_linked_github_repositories()
            if not linked_repos:
                st.info("Please link at least one GitHub repository in the sidebar to get started.")

            selected_repo = st.selectbox("Select Repository", linked_repos)

            if linked_repos:
                st.divider()

            if selected_repo:
                st.subheader("🌿 Branch Management")
                with st.expander("Create New Branch"):
                    new_branch_name = st.text_input("New Branch Name", placeholder="e.g., protocol-v1")
                    base_branch = st.text_input("Base Branch", "main")
                    if st.button("Create Branch", type="secondary") and new_branch_name:
                        if create_github_branch(st.session_state.github_token, selected_repo, new_branch_name, base_branch):
                            st.success(f"Created branch '{new_branch_name}' from '{base_branch}'")

                st.subheader("💾 Commit and Push")
                branch_name = st.text_input("Target Branch", "main")
                file_path = st.text_input("File Path", "protocols/evolved_protocol.md")
                commit_message = st.text_input("Commit Message", "Update evolved protocol")
                if st.button("Commit to GitHub", type="primary"):
                    if not st.session_state.protocol_text.strip():
                        st.error("Cannot commit empty content.")
                    elif commit_to_github(st.session_state.github_token, selected_repo, file_path, st.session_state.protocol_text, commit_message, branch_name):
                        st.success("✅ Committed to GitHub successfully!")
                    else:
                        st.error("❌ Failed to commit to GitHub. Check your token and permissions.")

    with tabs[3]: # Activity Feed tab
        with st.container(border=True):
            st.header("📜 Activity Feed")
            if st.session_state.get("activity_log"):
                for entry in reversed(st.session_state.activity_log):
                    st.markdown(f"- {entry}")
            else:
                st.info("No activity yet.")
        st.divider()

    with tabs[4]: # Report Templates tab
        with st.container(border=True):
            st.header("📊 Report Template Management")
            st.markdown("Manage your custom report templates here. Create new templates or view existing ones.")

            with st.expander("➕ Create New Template", expanded=False):
                new_template_name = st.text_input("Template Name", placeholder="e.g., Security Audit Report")
                new_template_content = st.text_area("Template Content (JSON)", height=200, placeholder="Paste your JSON template content here...")
                if st.button("Save Template", type="primary"):
                    if new_template_name and new_template_content:
                        try:
                            template_data = json.loads(new_template_content)
                            st.session_state.report_templates[new_template_name] = template_data
                            with open("report_templates.json", "w") as f:
                                json.dump(st.session_state.report_templates, f, indent=4)
                            st.success(f"Template '{new_template_name}' saved.")
                            _load_report_templates.clear()
                            st.rerun()
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format.")
                    else:
                        st.warning("Please provide a name and content for the template.")
            st.divider()

            st.subheader("📋 Existing Templates")
            if not st.session_state.report_templates:
                st.info("No report templates found. Create one above to get started!")
            else:
                st.markdown("<div class='template-grid'>", unsafe_allow_html=True)
                for template_name, template_content in st.session_state.report_templates.items():
                    st.markdown(f"""
                    <div class="template-card fade-in-up">
                        <h4 class="template-card-title">{template_name}</h4>
                        <pre class="template-content">{json.dumps(template_content, indent=2)}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


    with tabs[5]: # Model Dashboard tab
        with st.container(border=True):
            st.header("🤖 Model Performance Dashboard")
            st.markdown("Analyze the performance of different models used in adversarial testing.")

            model_performance = st.session_state.get("adversarial_model_performance")
            if not model_performance:
                st.info("No model performance data available. Run adversarial testing to generate data.")
            else:
                sorted_models = sorted(model_performance.items(), key=lambda x: x[1].get("score", 0), reverse=True)

                st.markdown("<div class='model-grid'>", unsafe_allow_html=True)
                for model_id, perf in sorted_models:
                    score = perf.get('score', 0)
                    issues = perf.get('issues_found', 0)
                    st.markdown(f"""
                    <div class="model-card fade-in-up">
                        <h4>{model_id}</h4>
                        <div class="metric">Score: <strong>{score}</strong></div>
                        <div class="metric">Issues Found: <strong>{issues}</strong></div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {min(score, 100)}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


    with tabs[6]: # Tasks tab
        with st.container(border=True):
            st.header("✅ Task Management")
            st.markdown("Create, view, and manage tasks related to content improvement.")

            with st.expander("➕ Create New Task", expanded=False):
                with st.form("new_task_form"):
                    title = st.text_input("Task Title", placeholder="e.g., Review security policy")
                    description = st.text_area("Description", placeholder="Provide details about the task...")
                    assignee = st.text_input("Assignee", placeholder="e.g., John Doe")
                    due_date = st.date_input("Due Date")
                    if st.form_submit_button("Create Task", type="primary"):
                        create_task(title, description, assignee, due_date)
                        st.success("Task created successfully!")
            st.divider()

            st.subheader("📋 Existing Tasks")
            tasks = get_tasks()
            if not tasks:
                st.info("No tasks created yet.")
            else:
                st.markdown("<div class='task-grid'>", unsafe_allow_html=True)
                for task in tasks:
                    status_color = "var(--success)" if task['status'] == "Completed" else ("var(--warning)" if task['status'] == "In Progress" else "var(--error)")
                    st.markdown(f"""
                    <div class="task-card fade-in-up">
                        <h4 class="task-card-title">{task['title']}</h4>
                        <p><strong>Status:</strong> <span style="color: {status_color};">{task['status']}</span></p>
                        <p><strong>Assignee:</strong> {task['assignee']}</p>
                        <p><strong>Due Date:</strong> {task['due_date']}</p>
                        <p>{task['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


    with tabs[7]: # Admin tab
        with st.container(border=True):
            st.header("👑 User and Role Management")
            st.markdown("Manage user accounts, assign roles, and configure access permissions.")

            with st.expander("➕ Add New User", expanded=False):
                with st.form("new_user_form"):
                    new_username = st.text_input("Username", placeholder="Enter new username")
                    new_role = st.selectbox("Role", list(ROLES.keys()), help="Select a role for the new user.")
                    if st.form_submit_button("Add User", type="primary"):
                        if new_username:
                            assign_role(new_username, new_role)
                            st.success(f"User '{new_username}' added with role '{new_role}'.")
                        else:
                            st.error("Username cannot be empty.")
            st.divider()

            st.subheader("👥 Existing Users")
            users = list(st.session_state.user_roles.keys())

            if not users:
                st.info("No users registered yet.")
            else:
                st.markdown("<div class='user-grid'>", unsafe_allow_html=True)
                for user in users:
                    st.markdown(f"""
                    <div class="user-card fade-in-up">
                        <h4 class="user-card-title">{user}</h4>
                        <p><strong>Role:</strong> <span class="user-role">{st.session_state.user_roles[user]}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    with tabs[8]: # Projects tab
        with st.container(border=True):
            st.header("📂 Project Management")
            st.markdown("Create and manage your content improvement projects.")

            with st.expander("➕ Create New Project", expanded=False):
                project_templates = content_manager.list_protocol_templates()
                selected_template = st.selectbox("Start from Template (Optional)", [""] + project_templates)
                new_project_name = st.text_input("New Project Name", placeholder="e.g., Q4 Security Policy Review")
                if st.button("Create Project", type="primary"):
                    if new_project_name:
                        if selected_template:
                            st.session_state.protocol_text = content_manager.load_protocol_template(selected_template)
                        st.session_state.project_name = new_project_name
                        st.session_state.projects[new_project_name] = {"description": "", "created_at": datetime.now().isoformat()}
                        st.success(f"Project '{new_project_name}' created.")
                        st.rerun()
                    else:
                        st.error("Project name cannot be empty.")
            st.divider()

            st.subheader("📋 Existing Projects")
            if not st.session_state.projects:
                st.info("No projects created yet.")
            else:
                st.markdown("<div class='project-grid'>", unsafe_allow_html=True)
                for project_name, project_data in st.session_state.projects.items():
                    st.markdown(f"""
                    <div class="project-card fade-in-up">
                        <h4 class="project-card-title">{project_name}</h4>
                        <p><strong>Created:</strong> {project_data.get('created_at', 'N/A')}</p>
                        <p>{project_data.get('description', 'No description provided.')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
