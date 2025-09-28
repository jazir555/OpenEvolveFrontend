import streamlit as st
import json
from datetime import datetime
import threading
import time

import difflib
from typing import List, Dict
import altair as alt
import numpy as np
from dataclasses import asdict

from pyvis.network import Network
from session_utils import _safe_list

from providercatalogue import get_openrouter_models, _parse_price_per_million

from session_manager import (
    APPROVAL_PROMPT, RED_TEAM_CRITIQUE_PROMPT, BLUE_TEAM_PATCH_PROMPT
)
from openevolve_integration import (
    OpenEvolveAPI, create_advanced_openevolve_config
)

from adversarial import (
    run_adversarial_testing, generate_html_report, generate_pdf_report, generate_docx_report,
    generate_latex_report, generate_compliance_report, optimize_model_selection,
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

@st.cache_data(ttl=3600) # Cache for 1 hour
def _load_report_templates():
    try:
        with open("report_templates.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

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
                # This is a placeholder for actual migration data
                # In a real implementation, you would get this data from the backend
                if abs(i - j) == 1:
                    net.add_edge(i, j, value=1)

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

    chart = alt.Chart(alt.Data(values=data)).mark_circle(size=60).encode(
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
    <div class="notification-button-container">
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

from session_state_classes import State, SessionManager

def _initialize_session_state():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "styles_css" not in st.session_state:
        with open("styles.css") as f:
            st.session_state.styles_css = f.read()
    if "show_quick_guide" not in st.session_state:
        st.session_state.show_quick_guide = False
    if "show_keyboard_shortcuts" not in st.session_state:
        st.session_state.show_keyboard_shortcuts = False
    if "adversarial_running" not in st.session_state:
        st.session_state.adversarial_running = False
    if "evolution_running" not in st.session_state:
        st.session_state.evolution_running = False
    if "evolution_history" not in st.session_state:
        st.session_state.evolution_history = []
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = []
    if "classification_and_tags" not in st.session_state:
        st.session_state.classification_and_tags = {}
    if "improvement_potential" not in st.session_state:
        st.session_state.improvement_potential = None
    if "vulnerabilities" not in st.session_state:
        st.session_state.vulnerabilities = []
    if "openevolve_base_url" not in st.session_state:
        st.session_state.openevolve_base_url = "http://localhost:8000"
    if "openevolve_api_key" not in st.session_state:
        st.session_state.openevolve_api_key = ""
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "evaluator_system_prompt" not in st.session_state:
        st.session_state.evaluator_system_prompt = ""
    if "evolution_use_specialized_evaluator" not in st.session_state:
        st.session_state.evolution_use_specialized_evaluator = False
    if "evolution_max_iterations" not in st.session_state:
        st.session_state.evolution_max_iterations = 20
    if "evolution_population_size" not in st.session_state:
        st.session_state.evolution_population_size = 1
    if "multi_objective_num_islands_island_model_2" not in st.session_state:
        st.session_state.multi_objective_num_islands_island_model_2 = 1
    if "evolution_elite_ratio" not in st.session_state:
        st.session_state.evolution_elite_ratio = 1.0
    if "evolution_checkpoint_interval" not in st.session_state:
        st.session_state.evolution_checkpoint_interval = 5
    if "evolution_exploration_ratio" not in st.session_state:
        st.session_state.evolution_exploration_ratio = 0.0
    if "evolution_exploitation_ratio" not in st.session_state:
        st.session_state.exploitation_ratio = 0.0
    if "evolution_archive_size" not in st.session_state:
        st.session_state.evolution_archive_size = 0
    if "model_temperature" not in st.session_state:
        st.session_state.model_temperature = 0.7
    if "model_top_p" not in st.session_state:
        st.session_state.model_top_p = 1.0
    if "model_frequency_penalty" not in st.session_state:
        st.session_state.model_frequency_penalty = 0.0
    if "model_presence_penalty" not in st.session_state:
        st.session_state.model_presence_penalty = 0.0
    if "multi_objective_feature_dimensions" not in st.session_state:
        st.session_state.multi_objective_feature_dimensions = ['complexity', 'diversity']
    if "multi_objective_feature_bins" not in st.session_state:
        st.session_state.multi_objective_feature_bins = 10
    if "multi_objective_num_islands_island_model_3" not in st.session_state:
        st.session_state.multi_objective_num_islands_island_model_3 = 1
    if "multi_objective_migration_interval" not in st.session_state:
        st.session_state.multi_objective_migration_interval = 50
    if "multi_objective_migration_rate" not in st.session_state:
        st.session_state.multi_objective_migration_rate = 0.1
    if "evolution_id" not in st.session_state:
        st.session_state.evolution_id = None
    if "evolution_log" not in st.session_state:
        st.session_state.evolution_log = []
    if "evolution_current_best" not in st.session_state:
        st.session_state.evolution_current_best = ""
    if "thread_lock" not in st.session_state:
        st.session_state.thread_lock = threading.Lock()
    if "protocol_text" not in st.session_state:
        st.session_state.protocol_text = "# Sample Protocol\n\nThis is a sample protocol for testing purposes."
    if "openrouter_key" not in st.session_state:
        st.session_state.openrouter_key = ""
    if "red_team_models" not in st.session_state:
        st.session_state.red_team_models = []
    if "blue_team_models" not in st.session_state:
        st.session_state.blue_team_models = []
    if "adversarial_custom_mode" not in st.session_state:
        st.session_state.adversarial_custom_mode = False
    if "adversarial_custom_red_prompt" not in st.session_state:
        st.session_state.adversarial_custom_red_prompt = RED_TEAM_CRITIQUE_PROMPT
    if "adversarial_custom_blue_prompt" not in st.session_state:
        st.session_state.adversarial_custom_blue_prompt = BLUE_TEAM_PATCH_PROMPT
    if "adversarial_custom_approval_prompt" not in st.session_state:
        st.session_state.adversarial_custom_approval_prompt = APPROVAL_PROMPT
    if "adversarial_review_type" not in st.session_state:
        st.session_state.adversarial_review_type = "Auto-Detect"
    if "adversarial_min_iter" not in st.session_state:
        st.session_state.adversarial_min_iter = 1
    if "adversarial_max_iter" not in st.session_state:
        st.session_state.adversarial_max_iter = 5
    if "adversarial_confidence" not in st.session_state:
        st.session_state.adversarial_confidence = 80
    if "adversarial_max_tokens" not in st.session_state:
        st.session_state.adversarial_max_tokens = 10000
    if "adversarial_max_workers" not in st.session_state:
        st.session_state.adversarial_max_workers = 4
    if "adversarial_force_json" not in st.session_state:
        st.session_state.adversarial_force_json = False
    if "adversarial_seed" not in st.session_state:
        st.session_state.adversarial_seed = ""
    if "adversarial_rotation_strategy" not in st.session_state:
        st.session_state.adversarial_rotation_strategy = "None"
    if "adversarial_staged_rotation_config" not in st.session_state:
        st.session_state.adversarial_staged_rotation_config = ""
    if "adversarial_red_team_sample_size" not in st.session_state:
        st.session_state.adversarial_red_team_sample_size = 1
    if "adversarial_blue_team_sample_size" not in st.session_state:
        st.session_state.adversarial_blue_team_sample_size = 1
    if "adversarial_auto_optimize_models" not in st.session_state:
        st.session_state.adversarial_auto_optimize_models = False
    if "adversarial_budget_limit" not in st.session_state:
        st.session_state.adversarial_budget_limit = 10.0 # Default budget limit
    if "adversarial_critique_depth" not in st.session_state:
        st.session_state.adversarial_critique_depth = 5
    if "adversarial_patch_quality" not in st.session_state:
        st.session_state.adversarial_patch_quality = 5
    if "adversarial_compliance_requirements" not in st.session_state:
        st.session_state.adversarial_compliance_requirements = ""
    if "adversarial_status_message" not in st.session_state:
        st.session_state.adversarial_status_message = ""
    if "adversarial_confidence_history" not in st.session_state:
        st.session_state.adversarial_confidence_history = []
    if "adversarial_cost_estimate_usd" not in st.session_state:
        st.session_state.adversarial_cost_estimate_usd = 0.0
    if "adversarial_total_tokens_prompt" not in st.session_state:
        st.session_state.adversarial_total_tokens_prompt = 0
    if "adversarial_total_tokens_completion" not in st.session_state:
        st.session_state.adversarial_total_tokens_completion = 0
    if "adversarial_log" not in st.session_state:
        st.session_state.adversarial_log = []
    if "adversarial_results" not in st.session_state:
        st.session_state.adversarial_results = None
    if "adversarial_model_performance" not in st.session_state:
        st.session_state.adversarial_model_performance = {}
    if "pdf_watermark" not in st.session_state:
        st.session_state.pdf_watermark = "OpenEvolve Confidential"
    if "custom_css" not in st.session_state:
        st.session_state.custom_css = ""
    if "discord_webhook_url" not in st.session_state:
        st.session_state.discord_webhook_url = ""
    if "msteams_webhook_url" not in st.session_state:
        st.session_state.msteams_webhook_url = ""
    if "generic_webhook_url" not in st.session_state:
        st.session_state.generic_webhook_url = ""
    if "github_token" not in st.session_state:
        st.session_state.github_token = ""
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []
    if "report_templates" not in st.session_state:
        st.session_state.report_templates = _load_report_templates()
    if "user_roles" not in st.session_state:
        st.session_state.user_roles = {"admin": "admin", "user": "user"} # Default roles
    if "projects" not in st.session_state:
        st.session_state.projects = {}
    if "project_public" not in st.session_state:
        st.session_state.project_public = False
    if "project_password" not in st.session_state:
        st.session_state.project_password = ""
    if "collaboration_session" not in st.session_state:
        st.session_state.collaboration_session = {"notifications": []}
    if "model" not in st.session_state:
        st.session_state.model = ""
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "base_url" not in st.session_state:
        st.session_state.base_url = ""
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "top_p" not in st.session_state:
        st.session_state.top_p = 1.0
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1000
    if "population_size" not in st.session_state:
        st.session_state.population_size = 1
    if "num_islands" not in st.session_state:
        st.session_state.num_islands = 1
    if "archive_size" not in st.session_state:
        st.session_state.archive_size = 0
    if "elite_ratio" not in st.session_state:
        st.session_state.elite_ratio = 1.0
    if "exploration_ratio" not in st.session_state:
        st.session_state.exploration_ratio = 0.0
    if "exploitation_ratio" not in st.session_state:
        st.session_state.exploitation_ratio = 0.0
    if "checkpoint_interval" not in st.session_state:
        st.session_state.checkpoint_interval = 5
    if "evolution_stop_flag" not in st.session_state:
        st.session_state.evolution_stop_flag = False
    if "adversarial_stop_flag" not in st.session_state:
        st.session_state.adversarial_stop_flag = False

def render_main_layout():

    if "template_manager" not in st.session_state:
        st.session_state.template_manager = TemplateManager()
    if "prompt_manager" not in st.session_state:
        st.session_state.prompt_manager = PromptManager()
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

    
    st.markdown('<h2 style="text-align: center; color: var(--primary-color);">🧬 OpenEvolve Content Improver</h2>', 
                unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: var(--text-color);">AI-Powered Content Hardening with Multi-LLM Consensus</p>',
        unsafe_allow_html=True)
    st.markdown("---")

    # Notification UI
    render_notification_ui()

    # Project information with enhanced UI
    st.markdown("## Adversarial Testing & Evolution-based Content Improvement")

    # Quick action buttons with enhanced styling
    quick_action_col1, quick_action_col2 = st.columns(2)
    with quick_action_col1:
        if st.button("📋 Quick Guide", key="main_layout_quick_guide_btn", use_container_width=True, type="secondary"):
            st.session_state.show_quick_guide = not st.session_state.get("show_quick_guide", False)
    with quick_action_col2:
        if st.button("⌨️ Keyboard Shortcuts", key="main_layout_keyboard_shortcuts_btn", use_container_width=True, type="secondary"):
            st.session_state.show_keyboard_shortcuts = not st.session_state.get("show_keyboard_shortcuts", False)
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
            if st.button("Close Guide"):
                st.session_state.show_quick_guide = False
                st.rerun()
    
    # Keyboard shortcuts documentation
    if st.session_state.get("show_keyboard_shortcuts", False):
        with st.expander("⌨️ Keyboard Shortcuts", expanded=True):
            st.markdown("### 🎯 Available Keyboard Shortcuts\n            \n            **Navigation & General**\n            - `Ctrl+S` - Save current protocol\n            - `Ctrl+O` - Open file\n            - `Ctrl+N` - Create new file\n            - `Ctrl+Shift+N` - New window\n            - `F5` or `Ctrl+R` - Refresh the application\n            - `F1` - Open help documentation\n            - `Ctrl+Shift+P` - Open command palette\n            - `Esc` - Close current modal or expandable section\n            - `Tab` - Indent selected text or insert 4 spaces\n            - `Shift+Tab` - Unindent selected text\n            \n            **Editing**\n            - `Ctrl+Z` - Undo last action\n            - `Ctrl+Y` or `Ctrl+Shift+Z` - Redo last action\n            - `Ctrl+X` - Cut selected text\n            - `Ctrl+C` - Copy selected text\n            - `Ctrl+V` - Paste text\n            - `Ctrl+A` - Select all text\n            - `Ctrl+F` - Find in protocol text\n            - `Ctrl+H` - Replace in protocol text\n            - `Ctrl+/` - Comment/uncomment selected lines\n            - `Ctrl+D` - Select current word/pattern\n            - `Ctrl+L` - Select current line\n            \n            **Formatting**\n            - `Ctrl+B` - Bold selected text\n            - `Ctrl+I` - Italicize selected text\n            - `Ctrl+U` - Underline selected text\n            - `Ctrl+Shift+K` - Insert link\n            - `Ctrl+Shift+I` - Insert image\n            - `Ctrl+Shift+L` - Create list\n            \n            **Application Specific**\n            - `Ctrl+Enter` - Start evolution/adversarial testing\n            - `Ctrl+Shift+Enter` - Start adversarial testing\n            - `Ctrl+M` - Toggle between light/dark mode\n            - `Ctrl+P` - Toggle panel visibility\n            - `Ctrl+E` - Export current document\n            - `Ctrl+Shift+F` - Toggle full screen\n            \n            **Text Editor Controls**\n            - `Ctrl+]` - Indent current line\n            - `Ctrl+[` - Outdent current line\n            - `Alt+Up/Down` - Move selected lines up/down\n            - `Ctrl+Shift+D` - Duplicate current line\n            - `Ctrl+Shift+K` - Delete current line\n            - `Ctrl+/` - Toggle line comment\n            - `Ctrl+Shift+/` - Toggle block comment\n            ")

    tab_names = ["Evolution", "⚔️ Adversarial Testing", "🐙 GitHub", "📜 Activity Feed", "📊 Report Templates", "🤖 Model Dashboard", "✅ Tasks", "👑 Admin", "📂 Projects"]
    tabs = st.tabs(tab_names)

    with tabs[0]: # Evolution tab
        with st.container(): # Wrap the entire tab content in a container
            st.header("Real-time Evolution Logs")

            with st.expander("📝 Content Input", expanded=True):
                st.text_area("Paste your draft content here:", height=300, key="protocol_text",
                             value="# Sample Protocol\n\nThis is a sample protocol for testing purposes.",
                             disabled=st.session_state.adversarial_running)

                templates = content_manager.list_protocol_templates()
                if templates:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        selected_template = st.selectbox("Load Template", [""] + templates, key="load_template_select")
                    with col2:
                        if selected_template and st.button("Load Selected Template", key="load_template_btn",
                                                                                           use_container_width=True, type="secondary"):
                            template_content = content_manager.load_protocol_template(selected_template)
                            st.session_state.protocol_text = template_content
                            st.success(f"Loaded template: {selected_template}")
            st.divider() # Add a divider

            with st.expander("🎮 Action Controls", expanded=True):
                c1, c2, c3 = st.columns(3)
                run_button = c1.button("🚀 Start Evolution", type="primary", disabled=st.session_state.evolution_running,
                                       use_container_width=True)
                stop_button = c2.button("⏹️ Stop Evolution", disabled=not st.session_state.evolution_running,
                                        use_container_width=True)
                c3.button("🔄 Resume Evolution", use_container_width=True, type="secondary")
            st.divider() # Add a divider

            classify_button = st.button("🏷️ Classify and Tag", use_container_width=True, type="secondary")
            predict_button = st.button("🔮 Predict Improvement Potential", use_container_width=True, type="secondary")
            security_button = st.button("🛡️ Check Security", use_container_width=True, type="secondary")
            st.divider() # Add a divider

            with st.expander("Compare Generations"):
                col1, col2 = st.columns(2)
                with col1:
                    generation1 = st.selectbox("Select Generation 1", range(len(st.session_state.evolution_history)))
                with col2:
                    generation2 = st.selectbox("Select Generation 2", range(len(st.session_state.evolution_history)))
                if st.button("Compare", type="secondary"):
                    text1 = st.session_state.evolution_history[generation1]['population'][0]['code']
                    text2 = st.session_state.evolution_history[generation2]['population'][0]['code']
                    render_code_diff(text1, text2)
            st.divider() # Add a divider

            if "suggestions" in st.session_state and st.session_state.suggestions:
                with st.expander("💡 Suggestions", expanded=True):
                    for suggestion in st.session_state.suggestions:
                        st.markdown(f"- {suggestion}")
                st.divider() # Add a divider

            if classify_button:
                with st.spinner("Classifying and tagging..."):
                    classification_and_tags = get_content_classification_and_tags(st.session_state.protocol_text)
                    st.session_state.classification_and_tags = classification_and_tags

            if "classification_and_tags" in st.session_state and st.session_state.classification_and_tags:
                with st.expander("🏷️ Classification and Tags", expanded=True):
                    st.write(f"**Classification:** {st.session_state.classification_and_tags.get('classification')}")
                    st.write(f"**Tags:** {', '.join(st.session_state.classification_and_tags.get('tags', []))}")
                st.divider() # Add a divider

            if predict_button:
                with st.spinner("Predicting improvement potential..."):
                    potential = predict_improvement_potential(st.session_state.protocol_text)
                    st.session_state.improvement_potential = potential

            if "improvement_potential" in st.session_state and st.session_state.improvement_potential is not None:
                st.metric("Improvement Potential", f"{st.session_state.improvement_potential:.2%}")
                st.divider() # Add a divider

            if security_button:
                with st.spinner("Checking for security vulnerabilities..."):
                    vulnerabilities = check_security_vulnerabilities(st.session_state.protocol_text)
                    st.session_state.vulnerabilities = vulnerabilities

            if "vulnerabilities" in st.session_state and st.session_state.vulnerabilities:
                with st.expander("🛡️ Security Vulnerabilities", expanded=True):
                    for vulnerability in st.session_state.vulnerabilities:
                        st.warning(vulnerability)
                st.divider() # Add a divider

            with st.expander("📝 Prompts"):
                api = OpenEvolveAPI(base_url=st.session_state.openevolve_base_url, api_key=st.session_state.openevolve_api_key)
                custom_prompts = api.get_custom_prompts()
                if custom_prompts:
                    selected_custom_prompt = st.selectbox("Select a custom prompt", ["None"].extend(list(custom_prompts.keys())))
                    if selected_custom_prompt != "None":
                        st.session_state.system_prompt = custom_prompts[selected_custom_prompt]['system_prompt']
                        st.session_state.evaluator_system_prompt = custom_prompts[selected_custom_prompt]['evaluator_system_prompt']
                
                st.text_area("System Prompt", key="evolution_system_prompt", height=150)
                st.text_area("Evaluator System Prompt", key="evolution_evaluator_system_prompt", height=150)
                st.checkbox("Use Specialized Evaluator", key="evolution_use_specialized_evaluator", help="Use a linter-based evaluator for more accurate code evaluation.")

                new_prompt_name = st.text_input("New Custom Prompt Name")
                if st.button("Save Custom Prompt", type="secondary"):
                    if new_prompt_name:
                        api.save_custom_prompt(new_prompt_name, {"system_prompt": st.session_state.system_prompt, "evaluator_system_prompt": st.session_state.evaluator_system_prompt})
                        st.success(f"Custom prompt '{new_prompt_name}' saved.")
                    else:
                        st.error("Prompt name cannot be empty.")
            st.divider() # Add a divider

            with st.expander("⬆️ Upload Custom Evaluator"):
                uploaded_evaluator_file = st.file_uploader("Upload Python file with 'evaluate' function", type=["py"])
                if uploaded_evaluator_file is not None:
                    evaluator_code = uploaded_evaluator_file.read().decode("utf-8")
                    api = OpenEvolveAPI(base_url=st.session_state.openevolve_base_url, api_key=st.session_state.openevolve_api_key)
                    evaluator_id = api.upload_evaluator(evaluator_code)
                    if evaluator_id:
                        st.session_state.custom_evaluator_id = evaluator_id
                        st.success(f"Evaluator uploaded with ID: {evaluator_id}")
                    else:
                        st.error("Failed to upload evaluator.")
            st.divider() # Add a divider

            with st.expander("Manage Custom Evaluators"):
                api = OpenEvolveAPI(base_url=st.session_state.openevolve_base_url, api_key=st.session_state.openevolve_api_key)
                custom_evaluators = api.get_custom_evaluators()
                if custom_evaluators:
                    for evaluator_id, evaluator_data in custom_evaluators.items():
                        with st.expander(f"Evaluator ID: {evaluator_id}"):
                            st.code(evaluator_data['code'], language="python")
                            if st.button("Delete Evaluator", key=f"delete_evaluator_{evaluator_id}", type="secondary"):
                                api.delete_evaluator(evaluator_id)
                                st.success(f"Evaluator {evaluator_id} deleted.")
                                st.rerun()
            st.divider() # Add a divider

            with st.expander("⚙️ Advanced Settings", expanded=False):
                st.markdown("### 🎛️ Evolution Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("Max Iterations", 1, 200, 20, key="evolution_max_iterations")
                    st.number_input("Population Size", 1, 100, 1, key="evolution_population_size")
                    st.number_input("Number of Islands", 1, 10, 1, key="multi_objective_num_islands_island_model_2")
                    st.slider("Elite Ratio", 0.0, 1.0, 1.0, 0.1, key="evolution_elite_ratio")
                with col2:
                    st.number_input("Checkpoint Interval", 1, 100, 5, key="evolution_checkpoint_interval")
                    st.slider("Exploration Ratio", 0.0, 1.0, 0.0, 0.1, key="evolution_exploration_ratio")
                    st.slider("Exploitation Ratio", 0.0, 1.0, 0.0, 0.1, key="evolution_exploitation_ratio")
                    st.number_input("Archive Size", 0, 100, 0, key="evolution_archive_size")
                
                st.markdown("### 🤖 Model Parameters")
                col3, col4 = st.columns(2)
                with col3:
                    st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="model_temperature")
                    st.slider("Top-P", 0.0, 1.0, 1.0, 0.1, key="model_top_p")
                with col4:
                    st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1, key="model_frequency_penalty")
                    st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1, key="model_presence_penalty")
                
                st.markdown("### 🎯 Multi-Objective Evolution")
                st.info("Define multiple objectives for the evolution. The fitness of each individual will be a vector of scores, one for each objective.")
                st_tags(
                    label='Feature Dimensions:',
                    text='Press enter to add more',
                    value=['complexity', 'diversity'],
                    key='multi_objective_feature_dimensions')
                st.number_input("Feature Bins", 1, 100, 10, key="multi_objective_feature_bins")

                st.number_input("Number of Islands", 1, 10, 1, key="multi_objective_num_islands_island_model_3")
                st.slider("Migration Interval", 0, 100, 50, key="multi_objective_migration_interval")
                st.slider("Migration Rate", 0.0, 1.0, 0.1, 0.05, key="multi_objective_migration_rate")
            st.divider() # Add a divider

            with st.expander("📊 Results", expanded=True):
                left, right = st.columns(2)
                with left:
                    st.subheader("📄 Current Best Content")
                    proto_out = st.empty()

                with right:
                    st.subheader("🔍 Logs")
                    log_out = st.empty()

                if st.session_state.evolution_running:
                    api = OpenEvolveAPI(base_url=st.session_state.openevolve_base_url, api_key=st.session_state.openevolve_api_key)
                    status = api.get_evolution_status(st.session_state.evolution_id)
                    if status:
                        proto_out.markdown(f"**Status:** {status['status']}\n\n**Best Score:** {status['best_score']}")
                        log_out.code(status['log'], language="text")
                        for log_chunk in api.stream_evolution_logs(st.session_state.evolution_id):
                            log_out.code(log_chunk, language="text")
                        if status['status'] == 'completed':
                            st.session_state.evolution_running = False
                            previous_best = st.session_state.evolution_current_best
                            best_solution = api.get_best_solution(st.session_state.evolution_id)
                            st.session_state.evolution_current_best = best_solution['code']
                            render_code_diff(previous_best, st.session_state.evolution_current_best)
                            history = api.get_evolution_history(st.session_state.evolution_id)
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
                    time.sleep(1)
                    st.rerun()
                else:
                    with st.session_state.thread_lock:
                        current_log = "\n".join(st.session_state.evolution_log)
                        current_content = st.session_state.evolution_current_best or st.session_state.protocol_text

                    log_out.code(current_content, language="markdown")
                    proto_out.code(current_content, language="markdown")

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
                    evolution_id = api.start_evolution(config=asdict(config))
                    if evolution_id:
                        st.session_state.evolution_id = evolution_id
                    st.rerun()

                if stop_button:
                    st.session_state.evolution_stop_flag = True


        with st.container(): # Wrap the entire tab content in a container
            st.header("Adversarial Testing with Multi-LLM Consensus")

            # Add a brief introduction
            st.markdown(
                "> **How it works:** Adversarial Testing uses two teams of AI models to improve your content:\n"
                "> - Red Team finds flaws and vulnerabilities\n"
                "> - Blue Team fixes the identified issues\n"
                "> The process repeats until your content reaches the desired confidence level."
            )
            st.divider() # Add a divider

            # OpenRouter Configuration
            st.subheader("🔑 OpenRouter Configuration")
            openrouter_key = st.text_input("OpenRouter API Key", type="password", key="openrouter_key")
            if not openrouter_key:
                st.info("Enter your OpenRouter API key to enable model selection and testing.")
                # return # Keep this return if you want to stop rendering the rest of the tab
            st.divider() # Add a divider

            models = get_openrouter_models(openrouter_key)
            # Update global model metadata with thread safety
            for m in models:
                if isinstance(m, dict) and (mid := m.get("id")):
                    with MODEL_META_LOCK:
                        MODEL_META_BY_ID[mid] = m
            if not models:
                st.error("No models fetched. Check your OpenRouter key and connection.")
                # return # Keep this return if you want to stop rendering the rest of the tab

            model_options = sorted([
                f"{m['id']} (Ctx: {m.get('context_length', 'N/A')}, "
                f"In: {_parse_price_per_million(m.get('pricing', {}).get('prompt')) or 'N/A'}/M, "
                f"Out: {_parse_price_per_million(m.get('pricing', {}).get('completion')) or 'N/A'}/M)"
                for m in models if isinstance(m, dict) and "id" in m
            ])

            # Protocol Templates
            st.subheader("📝 Content Input")

            # Add protocol input guidance
            st.info(
                "💡 **Tip:** Start with a clear, well-structured content. The better your starting point, the better the results.")

            # Protocol editor with enhanced features and live markdown preview
            protocol_col1, protocol_col2 = st.columns([3, 1])
            with protocol_col1:
                # Create tabs for input and preview
                input_tab, preview_tab = st.tabs(["📝 Edit", "👁️ Preview"])

                with input_tab:
                    protocol_text = st.text_area("✏️ Enter or paste your content:",
                                                 value=st.session_state.protocol_text,
                                                 height=300,
                                                 key="protocol_text_adversarial",
                                                 placeholder="Paste your draft content here...\n\nExample:\n# Security Policy\n\n## Overview\nThis policy defines requirements for secure system access.\n\n## Scope\nApplies to all employees and contractors.\n\n## Policy Statements\n1. All users must use strong passwords\n2. Multi-factor authentication is required for sensitive systems\n3. Regular security training is mandatory\n\n## Compliance\nViolations result in disciplinary action.")

                with preview_tab:
                    # Live markdown preview
                    st.markdown("### Live Preview")
                    if st.session_state.protocol_text:
                        st.markdown(st.session_state.protocol_text)
                    else:
                        st.info("Enter content in the 'Edit' tab to see the preview here.")
            with protocol_col2:
                st.markdown("**📋 Quick Actions**")

                # Template loading
                templates = content_manager.list_protocol_templates()
                if templates:
                    selected_template = st.selectbox("Load Template", [""] + templates, key="adv_load_template_select")
                    if selected_template and st.button("📥 Load Template", use_container_width=True):
                        st.session_state.protocol_text = content_manager.load_protocol_template(selected_template)

                # Sample protocol
                if st.button("🧪 Load Sample", use_container_width=True):
                    sample_protocol = """# Sample Security Policy\n\n## Overview\nThis policy defines security requirements for accessing company systems.\n\n## Scope\nApplies to all employees, contractors, and vendors with system access.\n\n## Policy Statements\n1. All users must use strong passwords\n2. Multi-factor authentication is required for sensitive systems\n3. Regular security training is mandatory\n4. Incident reporting must occur within 24 hours\n\n## Roles and Responsibilities\n- IT Security Team: Enforces policy and monitors compliance\n- Employees: Follow security practices and report incidents\n- Managers: Ensure team compliance and provide resources\n\n## Compliance\n- Audits conducted quarterly\n- Violations result in disciplinary action\n- Continuous monitoring through SIEM tools\n\n## Exceptions\n"""
                    st.session_state.protocol_text = sample_protocol

                # Clear button
                if st.session_state.protocol_text.strip() and st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.protocol_text = ""
            st.divider() # Add a divider

            # Model Selection
            st.subheader("🤖 Model Selection")

            # Add model selection guidance
            st.info(
                "💡 **Tip:** Select 3-5 diverse models for each team for best results. Mix small and large models for cost-effectiveness.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔴 Red Team (Critics)")
                st.caption("Models that find flaws and vulnerabilities in your protocol")

                if HAS_STREAMLIT_TAGS:
                    red_team_selected_full = st_tags(
                        label="Search and select models:",
                        text="Type to search models...",
                        value=st.session_state.red_team_models,
                        suggestions=model_options,
                        key="adversarial_red_team_select"
                    )
                    # Robust model ID extraction from descriptive string
                    red_team_models = []
                    for m in red_team_selected_full:
                        if " (" in m:
                            model_id = m.split(" (")[0].strip()
                        else:
                            model_id = m.strip()
                        if model_id:
                            red_team_models.append(model_id)
                    st.session_state.red_team_models = sorted(list(set(red_team_models)))
                else:
                    st.warning("streamlit_tags not available. Using text input for model selection.")
                    red_team_input = st.text_input("Enter Red Team models (comma-separated):",
                                                   value=",".join(st.session_state.red_team_models))
                    st.session_state.red_team_models = sorted(
                        list(set([model.strip() for model in red_team_input.split(",") if model.strip()])))

                # Model count indicator
                st.caption(f"Selected: {len(st.session_state.red_team_models)} models")


            with col2:
                st.markdown("#### 🔵 Blue Team (Fixers)")
                st.caption("Models that patch the identified flaws and improve the protocol")

                if HAS_STREAMLIT_TAGS:
                    blue_team_selected_full = st_tags(
                        label="Search and select models:",
                        text="Type to search models...",
                        value=st.session_state.blue_team_models,
                        suggestions=model_options,
                        key="adversarial_blue_team_select"
                    )
                    # Robust model ID extraction from descriptive string
                    blue_team_models = []
                    for m in blue_team_selected_full:
                        if " (" in m:
                            model_id = m.split(" (")[0].strip()
                        else:
                            model_id = m.strip()
                        if model_id:
                            blue_team_models.append(model_id)
                    st.session_state.blue_team_models = sorted(list(set(blue_team_models)))
                else:
                    st.warning("streamlit_tags not available. Using text input for model selection.")
                    blue_team_input = st.text_input("Enter Blue Team models (comma-separated):",
                                                    value=",".join(st.session_state.blue_team_models))
                    st.session_state.blue_team_models = sorted(
                        list(set([model.strip() for model in blue_team_input.split(",") if model.strip()])))

                # Model count indicator
                st.caption(f"Selected: {len(st.session_state.blue_team_models)} models")

            st.divider() # Add a divider

            # Testing Parameters
            st.subheader("🧪 Testing Parameters")

            # Custom Mode Toggle
            use_custom_mode = st.toggle("🔧 Use Custom Mode", key="adversarial_custom_mode",
                                        help="Enable custom prompts and configurations for adversarial testing")

            if use_custom_mode:
                with st.expander("🔧 Custom Prompts", expanded=True):
                    st.text_area("Red Team Prompt (Critique)",
                                 value=RED_TEAM_CRITIQUE_PROMPT,
                                 key="adversarial_custom_red_prompt",
                                 height=200,
                                 help="Custom prompt for the red team to find flaws in the protocol")

                    st.text_area("Blue Team Prompt (Patch)",
                                 value=BLUE_TEAM_PATCH_PROMPT,
                                 key="adversarial_custom_blue_prompt",
                                 height=200,
                                 help="Custom prompt for the blue team to patch the identified flaws")

                    st.text_area("Approval Prompt",
                                 value=APPROVAL_PROMPT,
                                 key="adversarial_custom_approval_prompt",
                                 height=150,
                                 help="Custom prompt for final approval checking")

            # Review Type Selection
            review_types = ["Auto-Detect", "General SOP", "Code Review", "Plan Review"]
            st.selectbox("Review Type", review_types, key="adversarial_review_type",
                         help="Select the type of review to perform. Auto-Detect will analyze the content and choose the appropriate review type.")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.number_input("Min iterations", 1, 50, key="adversarial_min_iter")
                st.number_input("Max iterations", 1, 200, key="adversarial_max_iter")
            with c2:
                st.slider("Confidence threshold (%)", 50, 100, key="adversarial_confidence",
                          help="Stop if this % of Red Team approves the SOP.")
            with c3:
                st.number_input("Max tokens per model", 1000, 100000, key="adversarial_max_tokens")
                st.number_input("Max parallel workers", 1, 24, key="adversarial_max_workers")
            with c4:
                st.toggle("Force JSON mode", key="adversarial_force_json",
                          help="Use model's built-in JSON mode if available. Increases reliability.")
                st.text_input("Deterministic seed", key="adversarial_seed", help="Integer for reproducible runs.")
                st.selectbox("Rotation Strategy",
                             ["None", "Round Robin", "Random Sampling", "Performance-Based", "Staged", "Adaptive",
                              "Focus-Category"], key="adversarial_rotation_strategy")
                if st.session_state.adversarial_rotation_strategy == "Staged":
                    help_text = """
[{"red": ["model1", "model2"], "blue": ["model3"]},
 {"red": ["model4"], "blue": ["model5", "model6"]}]
"""
                    st.text_area("Staged Rotation Config (JSON)", key="adversarial_staged_rotation_config", height=150, help=help_text)
                st.number_input("Red Team Sample Size", 1, 100, key="adversarial_red_team_sample_size")
                st.number_input("Blue Team Sample Size", 1, 100, key="adversarial_blue_team_sample_size")

                st.toggle("Auto-Optimize Model Selection", key="adversarial_auto_optimize_models",
                          help="Automatically select optimal models based on protocol complexity and budget")
                if st.session_state.adversarial_auto_optimize_models:
                    protocol_complexity = len(st.session_state.protocol_text.split())
                    optimized_models = optimize_model_selection(
                        st.session_state.red_team_models,
                        st.session_state.blue_team_models,
                        protocol_complexity,
                        st.session_state.adversarial_budget_limit
                    )
                    st.session_state.red_team_models = optimized_models["red_team"]
                    st.session_state.blue_team_models = optimized_models["blue_team"]

                st.markdown("### 🧠 Intelligence Settings")
                st.slider("Critique Depth", 1, 10, key="adversarial_critique_depth",
                          help="How deeply the red team should analyze the protocol (1-10)")
                st.slider("Patch Quality", 1, 10, key="adversarial_patch_quality",
                          help="Quality level for blue team patches (1-10)")
            st.divider() # Add a divider

            st.text_area("Compliance Requirements", key="adversarial_compliance_requirements", height=150,
                         help="Enter any compliance requirements that the red team should check for.")
            st.divider() # Add a divider

            all_models = sorted(list(set(st.session_state.red_team_models + st.session_state.blue_team_models)))
            if all_models:
                with st.expander("🔧 Per-Model Configuration", expanded=False):
                    for model_id in all_models:
                        st.markdown(f"**{model_id}**")
                        cc1, cc2, cc3, cc4 = st.columns(4)
                        cc1.slider(f"Temp##{model_id}", 0.0, 2.0, 0.7, 0.1, key=f"temp_{model_id}")
                        cc2.slider(f"Top-P##{model_id}", 0.0, 1.0, 1.0, 0.1, key=f"topp_{model_id}")
                        cc3.slider(f"Freq Pen##{model_id}", -2.0, 2.0, 0.0, 0.1, key=f"freqpen_{model_id}")
                        cc4.slider(f"Pres Pen##{model_id}", -2.0, 2.0, 0.0, 0.1, key=f"prespen_{model_id}")
            st.divider() # Add a divider

            # Start/Stop buttons for adversarial testing
            col1, col2, col3 = st.columns([2, 2, 1])
            start_button = col1.button("🚀 Start Adversarial Testing", type="primary",
                                       disabled=st.session_state.adversarial_running or not st.session_state.protocol_text.strip(),
                                       use_container_width=True)
            stop_button = col2.button("⏹️ Stop Adversarial Testing",
                                      disabled=not st.session_state.adversarial_running,
                                      use_container_width=True)
            st.divider() # Add a divider

            if start_button:
                st.session_state.adversarial_running = True
                threading.Thread(target=run_adversarial_testing).start()
                st.rerun()

            if stop_button:
                st.session_state.adversarial_stop_flag = True

            # Progress and status section
            if st.session_state.adversarial_running or st.session_state.adversarial_status_message:
                status_container = st.container()
                with status_container:
                    if st.session_state.adversarial_status_message:
                        status_msg = st.session_state.adversarial_status_message
                        if "Success" in status_msg or "✅" in status_msg:
                            st.success(status_msg)
                        elif "Error" in status_msg or "💥" in status_msg or "⚠️" in status_msg:
                            st.error(status_msg)
                        elif "Stop" in status_msg or "⏹️" in status_msg:
                            st.warning(status_msg)
                        else:
                            st.info(status_msg)

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
                            if st.session_state.adversarial_log:
                                log_content = "\n".join(st.session_state.adversarial_log[-50:])
                                st.text_area("Activity Log", value=log_content, height=300,
                                             key="adversarial_log_display",
                                             help="Auto-updating log of adversarial testing activities")
                            else:
                                st.info("⏳ Waiting for adversarial testing to start...")

                    if st.session_state.adversarial_results and not st.session_state.adversarial_running:
                        with st.expander("🏆 Adversarial Testing Results", expanded=True):
                            results = st.session_state.adversarial_results

                            st.markdown("### 📊 Performance Summary")

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("✅ Final Approval Rate", f"{results.get('final_approval_rate', 0):.1f}%")
                            col2.metric("🔄 Iterations Completed", len(results.get('iterations', [])))
                            col3.metric("💰 Total Cost (USD)", f"${results.get('cost_estimate_usd', 0):.4f}")
                            col4.metric("🤿 Total Tokens",
                                        f"{results.get('tokens', {}).get('prompt', 0) + results.get('tokens', {}).get('completion', 0):,}")

                            # Detailed metrics tabs
                            metrics_tab1, metrics_tab2, metrics_tab3, metrics_tab4 = st.tabs(
                                ["📈 Confidence Trend", "🏆 Model Performance", "🧮 Issue Analysis", "📊 Advanced Analytics"])

                            with metrics_tab1:
                                # Confidence trend chart
                                if results.get('iterations'):
                                    confidence_history = [iter.get("approval_check", {}).get("approval_rate", 0)
                                                          for iter in results.get('iterations', [])]
                                    if confidence_history:
                                        # Enhanced visualization
                                        import pandas as pd
                                        df = pd.DataFrame({'confidence': confidence_history})
                                        st.line_chart(df)

                                        # Trend line
                                        x = np.arange(len(confidence_history))
                                        y = np.array(confidence_history)
                                        z = np.polyfit(x, y, 1)
                                        p = np.poly1d(z)
                                        st.line_chart(pd.DataFrame({'trend': p(x)}))

                                        max_confidence = max(confidence_history)
                                        min_confidence = min(confidence_history)
                                        avg_confidence = sum(confidence_history) / len(confidence_history)

                                        st.line_chart(confidence_history)
                                        col1, col2, col3, col4 = st.columns(4)
                                        col1.metric("📈 Peak Confidence", f"{max_confidence:.1f}%")
                                        col2.metric("📉 Lowest Confidence", f"{min_confidence:.1f}%")
                                        col3.metric("📊 Average Confidence", f"{avg_confidence:.1f}%")
                                        col4.metric("📊 Final Confidence", f"{confidence_history[-1]:.1f}%")

                                        # Confidence improvement
                                        if len(confidence_history) > 1:
                                            improvement = confidence_history[-1] - confidence_history[0]
                                            if improvement > 0:
                                                st.success(f"🚀 Confidence improved by {improvement:.1f}%")
                                            elif improvement < 0:
                                                st.warning(f"⚠️ Confidence decreased by {abs(improvement):.1f}%")
                                            else:
                                                st.info("➡️ Confidence remained stable")

                            with metrics_tab2:
                                # Model performance analysis
                                if st.session_state.get("adversarial_model_performance"):
                                    model_performance = st.session_state.adversarial_model_performance
                                    st.markdown("### 🏆 Top Performing Models")

                                    # Sort models by score
                                    sorted_models = sorted(model_performance.items(), key=lambda x: x[1].get("score", 0),
                                                           reverse=True)

                                    # Display top 5 models with enhanced visualization
                                    for i, (model_id, perf) in enumerate(sorted_models[:5]):
                                        score = perf.get("score", 0)
                                        issues = perf.get("issues_found", 0)
                                        st.progress(min(score / 100, 1.0),
                                                    text=f"#{i + 1} {model_id} - Score: {score}, Issues Found: {issues}")
                                else:
                                    st.info("No model performance data available.")

                            with metrics_tab3:
                                # Issue analysis
                                if results.get('iterations'):
                                    # Aggregate issue data
                                    total_issues = 0
                                    severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
                                    category_counts = {}

                                    for iteration in results.get('iterations', []):
                                        critiques = iteration.get("critiques", [])
                                        for critique in critiques:
                                            critique_json = critique.get("critique_json", {})
                                            issues = critique_json.get("issues", [])
                                            total_issues += len(issues)

                                            for issue in issues:
                                                # Count by severity
                                                severity = issue.get("severity", "low").lower()
                                                if severity in severity_counts:
                                                    severity_counts[severity] += 1

                                                # Count by category
                                                category = issue.get("category", "uncategorized")
                                                category_counts[category] = category_counts.get(category, 0) + 1

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("### 🎯 Issue Severity Distribution")
                                        severity_counts = {}
                                        for iteration in results.get('iterations', []):
                                            for critique in iteration.get("critiques", []):
                                                if critique.get("critique_json"):
                                                    for issue in _safe_list(critique["critique_json"], "issues"):
                                                        severity = issue.get("severity", "low").lower()
                                                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                                        if severity_counts:
                                            st.bar_chart(severity_counts)
                                    with col2:
                                        st.markdown("### 📚 Issue Categories")
                                        # Show top 5 categories
                                        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                                        for category, count in sorted_categories[:5]:
                                            st.write(f"🏷️ {category}: {count}")

                                    st.metric("🔍 Total Issues Found", total_issues)
                            with metrics_tab4:
                                st.markdown("### 📊 Advanced Analytics")
                                analytics = analytics_manager.generate_advanced_analytics(results)
                                st.json(analytics)

                            st.markdown("### 📄 Final Hardened Protocol")
                            st.code(results.get('final_sop', ''), language="markdown")

                            # Export options
                            st.markdown("### 📁 Export Results")
                            st.text_input("Watermark for PDF Export", key="adversarial_pdf_watermark")
                            st.text_area("Custom CSS for HTML Export", key="adversarial_custom_css")
                            export_col1, export_col2, export_col3, export_col4 = st.columns(4)

                            with export_col1:
                                if st.button("📄 Export PDF", key="adversarial_export_pdf", use_container_width=True):
                                    if results:
                                        pdf_bytes = generate_pdf_report(results, st.session_state.pdf_watermark)
                                        st.download_button(
                                            label="📥 Download PDF",
                                            data=pdf_bytes,
                                            file_name=f"adversarial_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                    else:
                                        st.warning("No results to export.")

                            with export_col2:
                                if st.button("📝 DOCX", key="adversarial_export_docx", use_container_width=True):
                                    if results:
                                        docx_bytes = generate_docx_report(results)
                                        st.download_button(
                                            label="📥 Download DOCX",
                                            data=docx_bytes,
                                            file_name=f"adversarial_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                            use_container_width=True
                                        )
                                    else:
                                        st.warning("No results to export.")

                            with export_col3:
                                if st.button("📊 Export HTML", key="adversarial_export_html", use_container_width=True):
                                    if results:
                                        html_content = generate_html_report(results, st.session_state.custom_css)
                                        st.download_button(
                                            label="📥 Download HTML",
                                            data=html_content,
                                            file_name=f"adversarial_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                            mime="text/html",
                                            use_container_width=True
                                        )
                                    else:
                                        st.warning("No results to export.")

                            with export_col4:
                                if st.button("📋 Export JSON", key="adversarial_export_json", use_container_width=True):
                                    if results:
                                        json_str = json.dumps(results, indent=2, default=str)
                                        st.download_button(
                                            label="📥 Download JSON",
                                            data=json_str,
                                            file_name=f"adversarial_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                            mime="application/json",
                                            use_container_width=True
                                        )
                                    else:
                                        st.warning("No results to export.")
                            export_col5, = st.columns(1)
                            with export_col5:
                                if st.button("📄 Export LaTeX", key="adversarial_export_latex", use_container_width=True):
                                    if results:
                                        latex_str = generate_latex_report(results)
                                        st.download_button(
                                            label="📥 Download LaTeX",
                                            data=latex_str,
                                            file_name=f"adversarial_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
                                            mime="application/x-latex",
                                            use_container_width=True
                                        )
                                    else:
                                        st.warning("No results to export.")
                            export_col6, = st.columns(1)
                            with export_col6:
                                if st.button("💬 Send to Discord", key="adversarial_send_to_discord", use_container_width=True):
                                    if st.session_state.discord_webhook_url:
                                        message = f"Adversarial testing complete! Final approval rate: {results.get('final_approval_rate', 0.0):.1f}%"
                                        send_discord_notification(st.session_state.discord_webhook_url, message)
                                    else:
                                        st.warning("Please configure the Discord webhook URL in the sidebar.")
                            export_col7, = st.columns(1)
                            with export_col7:
                                if st.button("💬 Send to Teams", key="adversarial_send_to_teams", use_container_width=True):
                                    if st.session_state.msteams_webhook_url:
                                        message = f"Adversarial testing complete! Final approval rate: {results.get('final_approval_rate', 0.0):.1f}%"
                                        send_msteams_notification(st.session_state.msteams_webhook_url, message)
                                    else:
                                        st.warning("Please configure the Microsoft Teams webhook URL in the sidebar.")
                            export_col8, = st.columns(1)
                            with export_col8:
                                if st.button("🚀 Send Webhook", key="adversarial_send_webhook", use_container_width=True):
                                    if st.session_state.generic_webhook_url:
                                        payload = {"text": f"Adversarial testing complete! Final approval rate: {results.get('final_approval_rate', 0.0):.1f}%"}
                                        send_generic_webhook(st.session_state.generic_webhook_url, payload)
                                    else:
                                        st.warning("Please configure the generic webhook URL in the sidebar.")
                            export_col9, = st.columns(1)
                            with export_col9:
                                if st.button("📋 Generate Compliance Report", key="adversarial_generate_compliance_report", use_container_width=True):
                                    if results and st.session_state.compliance_requirements:
                                        compliance_report = generate_compliance_report(results, st.session_state.compliance_requirements)
                                        st.download_button(
                                            label="📥 Download Compliance Report",
                                            data=compliance_report,
                                            file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                            mime="text/markdown",
                                            use_container_width=True
                                        )
                                    else:
                                        st.warning("No results or compliance requirements to generate a report.")

    with tabs[2]: # GitHub tab
        with st.container():
            st.title("🐙 GitHub Integration")

            if not st.session_state.get("github_token"):
                st.warning("Please authenticate with GitHub in the sidebar first.")
                st.info("Go to the sidebar and enter your GitHub Personal Access Token to get started.")
                # st.stop() # Removed st.stop() to allow the rest of the UI to render
            st.divider()

            linked_repos = list_linked_github_repositories()
            if not linked_repos:
                st.warning("Please link at least one GitHub repository in the sidebar first.")
                st.info("Go to the sidebar, find the GitHub Integration section, and link a repository.")
                # st.stop() # Removed st.stop()
            st.divider()

            st.subheader("Select Repository")
            selected_repo = st.selectbox("Select Repository", linked_repos)
            st.divider()

            if selected_repo:
                st.subheader("🌿 Branch Management")
                with st.expander("Create New Branch"):
                    new_branch_name = st.text_input("New Branch Name", placeholder="e.g., protocol-v1")
                    base_branch = st.text_input("Base Branch", "main")
                    if st.button("Create Branch", type="secondary") and new_branch_name:
                        token = st.session_state.github_token
                        if create_github_branch(token, selected_repo, new_branch_name, base_branch):
                            st.success(f"Created branch '{new_branch_name}' from '{base_branch}'")
                st.divider()

                st.subheader("💾 Commit and Push")
                branch_name = st.text_input("Target Branch", "main")
                file_path = st.text_input("File Path", "protocols/evolved_protocol.md")
                commit_message = st.text_input("Commit Message", "Update evolved protocol")
                if st.button("Commit to GitHub", type="primary") and st.session_state.protocol_text.strip():
                    token = st.session_state.github_token
                    if commit_to_github(token, selected_repo, file_path, st.session_state.protocol_text, commit_message, branch_name):
                        st.success("✅ Committed to GitHub successfully!")
                        if "github_generations" not in st.session_state:
                            st.session_state.github_generations = []
                        st.session_state.github_generations.append({
                            "repo": selected_repo,
                            "file_path": file_path,
                            "branch": branch_name,
                            "timestamp": datetime.now().isoformat(),
                            "commit_message": commit_message
                        })
                    else:
                        st.error("❌ Failed to commit to GitHub")
                st.divider()

    with tabs[3]: # Activity Feed tab
        st.title("📜 Activity Feed")
        render_activity_feed_ui()
        st.markdown("<br>", unsafe_allow_html=True)

    with tabs[4]: # Report Templates tab
        st.title("📊 Report Templates")
        render_report_templates_ui()
        st.markdown("<br>", unsafe_allow_html=True)

    with tabs[5]: # Model Dashboard tab
        st.title("🤖 Model Dashboard")
        render_model_dashboard_ui()
        st.markdown("<br>", unsafe_allow_html=True)

    with tabs[6]: # Tasks tab
        st.title("✅ Tasks")
        render_tasks_ui()
        st.markdown("<br>", unsafe_allow_html=True)

    with tabs[7]: # Admin tab
        st.title("👑 Admin")
        render_admin_ui()
        st.markdown("<br>", unsafe_allow_html=True)
    with tabs[8]: # Projects tab
        st.title("📂 Projects")
        render_projects_tab()
        st.markdown("<br>", unsafe_allow_html=True)
def render_model_dashboard_ui():
    """Render the model comparison dashboard UI."""
    with st.container(): # Wrap the entire function content in a container
        st.subheader("Model Performance")

        if "adversarial_model_performance" not in st.session_state or not st.session_state.adversarial_model_performance:
            st.warning("No model performance data available. Run adversarial testing to generate data.")
            return

        model_performance = st.session_state.adversarial_model_performance
        sorted_models = sorted(model_performance.items(), key=lambda x: x[1].get("score", 0), reverse=True)



        st.markdown("<div class='model-dashboard-container'>", unsafe_allow_html=True)
        for model_id, perf in sorted_models:
            st.markdown(f"""
            <div class="model-card">
                <h4>{model_id}</h4>
                <p><strong>Score:</strong> {perf.get('score', 0)}</p>
                <p><strong>Issues Found:</strong> {perf.get('issues_found', 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.divider() # Add a divider at the end of the section

def render_tasks_ui():
    """Render the tasks UI."""
    with st.container():
        st.subheader("Create New Task")
        with st.form("new_task_form"):
            title = st.text_input("Title")
            description = st.text_area("Description")
            assignee = st.text_input("Assignee")
            due_date = st.date_input("Due Date")
            submitted = st.form_submit_button("Create Task")
            if submitted:
                create_task(title, description, assignee, due_date)
                st.success("Task created successfully!")
        st.divider()

        st.subheader("Tasks")
        tasks = get_tasks()



        st.markdown("<div class='task-container'>", unsafe_allow_html=True)
        for task in tasks:
            st.markdown(f"""
            <div class="task-card">
                <h4>{task['title']} ({task['status']})</h4>
                <p><strong>Description:</strong> {task['description']}</p>
                <p><strong>Assignee:</strong> {task['assignee']}</p>
                <p><strong>Due Date:</strong> {task['due_date']}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.divider() # Add a divider at the end of the section

def render_admin_ui():
    """Render the admin UI for managing users and roles."""
    with st.container():
        st.subheader("User Management")

        # Add new user form
        with st.expander("Add New User", expanded=True):
            with st.form("new_user_form"):
                st.write("Add New User")
                new_username = st.text_input("Username")
                new_role = st.selectbox("Role", list(ROLES.keys()))
                submitted = st.form_submit_button("Add User")
                if submitted:
                    if new_username:
                        assign_role(new_username, new_role)
                        st.success(f"User '{new_username}' added with role '{new_role}'.")
                    else:
                        st.error("Username cannot be empty.")
        st.divider()

        st.subheader("Existing Users")
        users = list(st.session_state.user_roles.keys())



        st.markdown("<div class='user-container'>", unsafe_allow_html=True)
        for user in users:
            st.markdown(f"""
            <div class="user-card">
                <h4>{user}</h4>
                <p><strong>Role:</strong> {st.session_state.user_roles[user]}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.divider() # Add a divider at the end of the section

def render_projects_tab():
    st.title("📂 Projects")
    with st.container():
        st.subheader("Create New Project")
        project_templates = content_manager.list_protocol_templates()
        selected_template = st.selectbox("Select a project template", [""] + project_templates)
        new_project_name = st.text_input("New Project Name")
        if st.button("Create Project") and new_project_name:
            if selected_template:
                template_content = content_manager.load_protocol_template(selected_template)
                st.session_state.protocol_text = template_content
            st.session_state.project_name = new_project_name
            st.success(f"Project '{new_project_name}' created.")
        st.divider()

        st.subheader("Manage Existing Projects")
        if "projects" not in st.session_state:
            st.session_state.projects = {}



        st.markdown("<div class='project-container'>", unsafe_allow_html=True)
        for project_name, project_data in st.session_state.projects.items():
            st.markdown(f"""
            <div class="project-card">
                <h4>{project_name}</h4>
                <p><strong>Description:</strong> {project_data.get('description', '')}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.divider() # Add a divider at the end of the section

def render_report_templates_ui():
    """Render the report templates UI."""
    st.markdown("""
    > Manage your custom report templates here. Create new templates or view existing ones.
    """)
    st.markdown("<br>", unsafe_allow_html=True)

    if "report_templates" not in st.session_state:
        st.session_state.report_templates = _load_report_templates()

    with st.container(border=True): # Create New Template Section
        st.markdown("### Create New Template")
        new_template_name = st.text_input("Template Name")
        new_template_content = st.text_area("Template Content (JSON)", height=200)
        if st.button("Save Template", type="primary"):
            if new_template_name and new_template_content:
                try:
                    template_data = json.loads(new_template_content)
                    st.session_state.report_templates[new_template_name] = template_data
                    with open("report_templates.json", "w") as f:
                        json.dump(st.session_state.report_templates, f, indent=4)
                    st.success(f"Template '{new_template_name}' saved.")
                    _load_report_templates.clear()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format.")
            else:
                st.warning("Please provide a name and content for the template.")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True): # Existing Templates Section
        st.markdown("### Existing Templates")
        st.markdown("<div class='template-container'>", unsafe_allow_html=True)
        for template_name, template_content in st.session_state.report_templates.items():
            st.markdown(f"""
            <div class="template-card">
                <h4>{template_name}</h4>
                <p>{json.dumps(template_content, indent=2)}</p>
                <button class="stButton secondary-button" onclick="
                    var event = new CustomEvent('streamlit:setComponentValue', {{detail: {{key: 'edit_template_{template_name}', value: true}}}});
                    window.parent.document.dispatchEvent(event);
                ">Edit</button>
                <button class="stButton secondary-button" onclick="
                    var event = new CustomEvent('streamlit:setComponentValue', {{detail: {{key: 'delete_template_{template_name}', value: true}}}});
                    window.parent.document.dispatchEvent(event);
                ">Delete</button>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Handle edit/delete actions (simplified for now, full implementation would involve more state management)
        for template_name in st.session_state.report_templates.keys():
            if st.session_state.get(f'edit_template_{template_name}') :
                st.session_state[f'edit_template_{template_name}'] = False # Reset
                st.info(f"Editing template: {template_name} (functionality to be implemented)")
            if st.session_state.get(f'delete_template_{template_name}') :
                st.session_state[f'delete_template_{template_name}'] = False # Reset
                del st.session_state.report_templates[template_name]
                with open("report_templates.json", "w") as f:
                    json.dump(st.session_state.report_templates, f, indent=4)
                st.success(f"Template '{template_name}' deleted.")
                _load_report_templates.clear()
                st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

def render_adversarial_testing_tab():
    st.header("Adversarial Testing with Multi-LLM Consensus")

    # Add a brief introduction
    st.markdown(
        "> **How it works:** Adversarial Testing uses two teams of AI models to improve your content:\\n"
        "> - Red Team finds flaws and vulnerabilities\\n"
        "> - Blue Team fixes the identified issues\\n"
        "> The process repeats until your content reaches the desired confidence level."
    )

    # OpenRouter Configuration
    st.subheader("🔑 OpenRouter Configuration")
    openrouter_key = st.text_input("OpenRouter API Key", type="password", key="openrouter_key")
    if not openrouter_key:
        st.info("Enter your OpenRouter API key to enable model selection and testing.")
        return
    
    # Cache models based on the openrouter_key
    if "openrouter_models" not in st.session_state or st.session_state.get("last_openrouter_key") != openrouter_key:
        models = get_openrouter_models(openrouter_key)
        if not models:
            st.error("No models fetched. Check your OpenRouter key and connection.")
            return
        st.session_state.openrouter_models = models
        st.session_state.last_openrouter_key = openrouter_key
    else:
                    models = st.session_state.openrouter_models
    
    # Update global model metadata with thread safety
    for m in models:
        if isinstance(m, dict) and (mid := m.get("id")):
            with MODEL_META_LOCK:
                MODEL_META_BY_ID[mid] = m

    model_options = sorted([
        f"{m['id']} (Ctx: {m.get('context_length', 'N/A')}, "
        f"In: {_parse_price_per_million(m.get('pricing', {}).get('prompt')) or 'N/A'}/M, "
        f"Out: {_parse_price_per_million(m.get('pricing', {}).get('completion')) or 'N/A'}/M)"
                    for m in models if isinstance(m, dict) and "id" in m
                ])
    # Protocol Templates
    st.markdown("---")
    st.subheader("📝 Content Input")

    # Add protocol input guidance
    st.info(
        "💡 **Tip:** Start with a clear, well-structured content. The better your starting point, the better the results.")

    # Protocol editor with enhanced features and live markdown preview
    protocol_col1, protocol_col2 = st.columns([3, 1])
    with protocol_col1:
        # Create tabs for input and preview
        input_tab, preview_tab = st.tabs(["📝 Edit", "👁️ Preview"])
        
        with input_tab:
            protocol_text = st.text_area("✏️ Enter or paste your content:",
                                         value=st.session_state.protocol_text,
                                         height=300,
                                         key="protocol_text_adversarial",
                                         placeholder="Paste your draft content here...\n\nExample:\n# Security Policy\n\n## Overview\nThis policy defines requirements for secure system access.\n\n## Scope\nApplies to all employees and contractors.\n\n## Policy Statements\n1. All users must use strong passwords\n2. Multi-factor authentication is required for sensitive systems\n3. Regular security training is mandatory\n\n## Compliance\nViolations result in disciplinary action.")
            st.session_state.protocol_text = protocol_text
        
        with preview_tab:
            # Live markdown preview
            st.markdown("### Live Preview")
            if st.session_state.protocol_text:
                st.markdown(st.session_state.protocol_text)
            else:
                st.info("Enter content in the 'Edit' tab to see the preview here.")
    with protocol_col2:
        st.markdown("**📋 Quick Actions**")

        # Template loading
        templates = content_manager.list_protocol_templates()
        if templates:
            selected_template = st.selectbox("Load Template", [""] + templates, key="adv_load_template_select")
            if selected_template and st.button("📥 Load Template", use_container_width=True):
                st.session_state.protocol_text = content_manager.load_protocol_template(selected_template)
                st.rerun()

        # Sample protocol
        if st.button("🧪 Load Sample", use_container_width=True):
            sample_protocol = """# Sample Security Policy

## Overview
This policy defines security requirements for accessing company systems.

## Scope
Applies to all employees, contractors, and vendors with system access.

## Policy Statements
1. All users must use strong passwords
2. Multi-factor authentication is required for sensitive systems
3. Regular security training is mandatory
4. Incident reporting must occur within 24 hours

## Roles and Responsibilities
- IT Security Team: Enforces policy and monitors compliance
- Employees: Follow security practices and report incidents
- Managers: Ensure team compliance and provide resources

## Compliance
- Audits conducted quarterly
- Violations result in disciplinary action
- Continuous monitoring through SIEM tools

## Exceptions
"""
            st.session_state.protocol_text = sample_protocol
            st.rerun()

        # Clear button
        if st.session_state.protocol_text.strip() and st.button("🗑️ Clear", use_container_width=True):
            st.session_state.protocol_text = ""
            st.rerun()

    # Model Selection
    st.markdown("---")
    st.subheader("🤖 Model Selection")

    # Add model selection guidance
    st.info(
        "💡 **Tip:** Select 3-5 diverse models for each team for best results. Mix small and large models for cost-effectiveness.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔴 Red Team (Critics)")
        st.caption("Models that find flaws and vulnerabilities in your protocol")

        if HAS_STREAMLIT_TAGS:
            red_team_selected_full = st_tags(
                label="Search and select models:",
                text="Type to search models...",
                value=st.session_state.red_team_models,
                suggestions=model_options,
                key="adversarial_red_team_select"
            )
            # Robust model ID extraction from descriptive string
            red_team_models = []
            for m in red_team_selected_full:
                if " (" in m:
                    model_id = m.split(" (")[0].strip()
                else:
                    model_id = m.strip()
                if model_id:
                    red_team_models.append(model_id)
            st.session_state.red_team_models = sorted(list(set(red_team_models)))
        else:
            st.warning("streamlit_tags not available. Using text input for model selection.")
            red_team_input = st.text_input("Enter Red Team models (comma-separated):",
                                           value=",".join(st.session_state.red_team_models))
            st.session_state.red_team_models = sorted(
                list(set([model.strip() for model in red_team_input.split(",") if model.strip()])))

        # Model count indicator
        st.caption(f"Selected: {len(st.session_state.red_team_models)} models")
        print(f"Red Team Models: {st.session_state.red_team_models}")

    with col2:
        st.markdown("#### 🔵 Blue Team (Fixers)")
        st.caption("Models that patch the identified flaws and improve the protocol")

        if HAS_STREAMLIT_TAGS:
            blue_team_selected_full = st_tags(
                label="Search and select models:",
                text="Type to search models...",
                value=st.session_state.blue_team_models,
                suggestions=model_options,
                key="adversarial_blue_team_select"
            )
            # Robust model ID extraction from descriptive string
            blue_team_models = []
            for m in blue_team_selected_full:
                if " (" in m:
                    model_id = m.split(" (")[0].strip()
                else:
                    model_id = m.strip()
                if model_id:
                    blue_team_models.append(model_id)
            st.session_state.blue_team_models = sorted(list(set(blue_team_models)))
        else:
            st.warning("streamlit_tags not available. Using text input for model selection.")
            blue_team_input = st.text_input("Enter Blue Team models (comma-separated):",
                                            value=",".join(st.session_state.blue_team_models))
            st.session_state.blue_team_models = sorted(
                list(set([model.strip() for model in blue_team_input.split(",") if model.strip()])))

        # Model count indicator
        st.caption(f"Selected: {len(st.session_state.blue_team_models)} models")
        print(f"Blue Team Models: {st.session_state.blue_team_models}")

    # Model Selection
    st.markdown("---")
    st.subheader("🤖 Model Selection")

    # Add model selection guidance
    st.info(
        "💡 **Tip:** Select 3-5 diverse models for each team for best results. Mix small and large models for cost-effectiveness.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔴 Red Team (Critics)")
        st.caption("Models that find flaws and vulnerabilities in your protocol")

        if HAS_STREAMLIT_TAGS:
            red_team_selected_full = st_tags(
                label="Search and select models:",
                text="Type to search models...",
                value=st.session_state.red_team_models,
                suggestions=model_options,
                key="red_team_select"
            )
            # Robust model ID extraction from descriptive string
            red_team_models = []
            for m in red_team_selected_full:
                if " (" in m:
                    model_id = m.split(" (")[0].strip()
                else:
                    model_id = m.strip()
                if model_id:
                    red_team_models.append(model_id)
            st.session_state.red_team_models = sorted(list(set(red_team_models)))
        else:
            st.warning("streamlit_tags not available. Using text input for model selection.")
            red_team_input = st.text_input("Enter Red Team models (comma-separated):",
                                           value=",".join(st.session_state.red_team_models))
            st.session_state.red_team_models = sorted(
                list(set([model.strip() for model in red_team_input.split(",") if model.strip()])))

        # Model count indicator
        st.caption(f"Selected: {len(st.session_state.red_team_models)} models")
        print(f"Red Team Models: {st.session_state.red_team_models}")

    with col2:
        st.markdown("#### 🔵 Blue Team (Fixers)")
        st.caption("Models that patch the identified flaws and improve the protocol")

        if HAS_STREAMLIT_TAGS:
            blue_team_selected_full = st_tags(
                label="Search and select models:",
                text="Type to search models...",
                value=st.session_state.blue_team_models,
                suggestions=model_options,
                key="blue_team_select"
            )
            # Robust model ID extraction from descriptive string
            blue_team_models = []
            for m in blue_team_selected_full:
                if " (" in m:
                    model_id = m.split(" (")[0].strip()
                else:
                    model_id = m.strip()
                if model_id:
                    blue_team_models.append(model_id)
            st.session_state.blue_team_models = sorted(list(set(blue_team_models)))
        else:
            st.warning("streamlit_tags not available. Using text input for model selection.")
            blue_team_input = st.text_input("Enter Blue Team models (comma-separated):",
                                            value=",".join(st.session_state.blue_team_models))
            st.session_state.blue_team_models = sorted(
                list(set([model.strip() for model in blue_team_input.split(",") if model.strip()])))

        # Model count indicator
        st.caption(f"Selected: {len(st.session_state.blue_team_models)} models")
        print(f"Blue Team Models: {st.session_state.blue_team_models}")

    # Testing Parameters
    st.markdown("---")
    st.subheader("🧪 Testing Parameters")

    # Custom Mode Toggle
    use_custom_mode = st.toggle("🔧 Use Custom Mode", key="adversarial_custom_mode",
                                help="Enable custom prompts and configurations for adversarial testing")

    if use_custom_mode:
        with st.expander("🔧 Custom Prompts", expanded=True):
            st.text_area("Red Team Prompt (Critique)",
                         value=RED_TEAM_CRITIQUE_PROMPT,
                         key="adversarial_custom_red_prompt",
                         height=200,
                         help="Custom prompt for the red team to find flaws in the protocol")

            st.text_area("Blue Team Prompt (Patch)",
                         value=BLUE_TEAM_PATCH_PROMPT,
                         key="adversarial_custom_blue_prompt",
                         height=200,
                         help="Custom prompt for the blue team to patch the identified flaws")

            st.text_area("Approval Prompt",
                         value=APPROVAL_PROMPT,
                         key="adversarial_custom_approval_prompt",
                         height=150,
                         help="Custom prompt for final approval checking")

    # Review Type Selection
    review_types = ["Auto-Detect", "General SOP", "Code Review", "Plan Review"]
    st.selectbox("Review Type", review_types, key="adversarial_review_type",
                 help="Select the type of review to perform. Auto-Detect will analyze the content and choose the appropriate review type.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input("Min iterations", 1, 50, key="adversarial_min_iter")
        st.number_input("Max iterations", 1, 200, key="adversarial_max_iter")
    with c2:
        st.slider("Confidence threshold (%)", 50, 100, key="adversarial_confidence",
                  help="Stop if this % of Red Team approves the SOP.")
    with c3:
        st.number_input("Max tokens per model", 1000, 100000, key="adversarial_max_tokens")
        st.number_input("Max parallel workers", 1, 24, key="adversarial_max_workers")
    with c4:
        st.toggle("Force JSON mode", key="adversarial_force_json",
                  help="Use model's built-in JSON mode if available. Increases reliability.")
        st.text_input("Deterministic seed", key="adversarial_seed", help="Integer for reproducible runs.")
        st.selectbox("Rotation Strategy",
                     ["None", "Round Robin", "Random Sampling", "Performance-Based", "Staged", "Adaptive",
                      "Focus-Category"], key="adversarial_rotation_strategy")
        if st.session_state.adversarial_rotation_strategy == "Staged":
            help_text = """
[{"red": ["model1", "model2"], "blue": ["model3"]},
 {"red": ["model4"], "blue": ["model5", "model6"]}]
"""
            st.text_area("Staged Rotation Config (JSON)", key="adversarial_staged_rotation_config", height=150, help=help_text)
        st.number_input("Red Team Sample Size", 1, 100, key="adversarial_red_team_sample_size")
        st.number_input("Blue Team Sample Size", 1, 100, key="adversarial_blue_team_sample_size")
        print(f"Min Iterations: {st.session_state.adversarial_min_iter}")
        print(f"Max Iterations: {st.session_state.adversarial_max_iter}")
        print(f"Confidence Threshold: {st.session_state.adversarial_confidence}")
        print(f"Max Tokens: {st.session_state.adversarial_max_tokens}")
        print(f"Max Parallel Workers: {st.session_state.adversarial_max_workers}")
        print(f"Force JSON Mode: {st.session_state.adversarial_force_json}")
        print(f"Deterministic Seed: {st.session_state.adversarial_seed}")
        print(f"Rotation Strategy: {st.session_state.adversarial_rotation_strategy}")
        print(f"Red Team Sample Size: {st.session_state.adversarial_red_team_sample_size}")
        print(f"Blue Team Sample Size: {st.session_state.adversarial_blue_team_sample_size}")
        st.toggle("Auto-Optimize Model Selection", key="adversarial_auto_optimize_models",
                  help="Automatically select optimal models based on protocol complexity and budget")
        if st.session_state.adversarial_auto_optimize_models:
            protocol_complexity = len(st.session_state.protocol_text.split())
            optimized_models = optimize_model_selection(
                st.session_state.red_team_models,
                st.session_state.blue_team_models,
                protocol_complexity,
                st.session_state.adversarial_budget_limit
            )
            st.session_state.red_team_models = optimized_models["red_team"]
            st.session_state.blue_team_models = optimized_models["blue_team"]

        st.markdown("### 🧠 Intelligence Settings")
        st.slider("Critique Depth", 1, 10, key="adversarial_critique_depth",
                  help="How deeply the red team should analyze the protocol (1-10)")
        st.slider("Patch Quality", 1, 10, key="adversarial_patch_quality",
                  help="Quality level for blue team patches (1-10)")

    st.text_area("Compliance Requirements", key="adversarial_compliance_requirements", height=150,
                 help="Enter any compliance requirements that the red team should check for.")

    all_models = sorted(list(set(st.session_state.red_team_models + st.session_state.blue_team_models)))
    if all_models:
        with st.expander("🔧 Per-Model Configuration", expanded=False):
            for model_id in all_models:
                st.markdown(f"**{model_id}**")
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.slider(f"Temp##{model_id}", 0.0, 2.0, 0.7, 0.1, key=f"temp_{model_id}")
                cc2.slider(f"Top-P##{model_id}", 0.0, 1.0, 1.0, 0.1, key=f"topp_{model_id}")
                cc3.slider(f"Freq Pen##{model_id}", -2.0, 2.0, 0.0, 0.1, key=f"freqpen_{model_id}")
                cc4.slider(f"Pres Pen##{model_id}", -2.0, 2.0, 0.0, 0.1, key=f"prespen_{model_id}")

    # Start/Stop buttons for adversarial testing
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])
    start_button = col1.button("🚀 Start Adversarial Testing", type="primary",
                               disabled=st.session_state.adversarial_running or not st.session_state.protocol_text.strip(),
                               use_container_width=True)
    stop_button = col2.button("⏹️ Stop Adversarial Testing",
                              disabled=not st.session_state.adversarial_running,
                              use_container_width=True)

    if start_button:
        st.session_state.adversarial_running = True
        threading.Thread(target=run_adversarial_testing).start()
        st.rerun()

    if stop_button:
        st.session_state.adversarial_stop_flag = True

    # Progress and status section
    if st.session_state.adversarial_running or st.session_state.adversarial_status_message:
        status_container = st.container()
        with status_container:
            if st.session_state.adversarial_status_message:
                status_msg = st.session_state.adversarial_status_message
                if "Success" in status_msg or "✅" in status_msg:
                    st.success(status_msg)
                elif "Error" in status_msg or "💥" in status_msg or "⚠️" in status_msg:
                    st.error(status_msg)
                elif "Stop" in status_msg or "⏹️" in status_msg:
                    st.warning(status_msg)
                else:
                    st.info(status_msg)

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
                    if st.session_state.adversarial_log:
                        log_content = "\n".join(st.session_state.adversarial_log[-50:])
                        st.text_area("Activity Log", value=log_content, height=300,
                                     key="adversarial_log_display",
                                     help="Auto-updating log of adversarial testing activities")
                    else:
                        st.info("⏳ Waiting for adversarial testing to start...")

            if st.session_state.adversarial_results and not st.session_state.adversarial_running:
                with st.expander("🏆 Adversarial Testing Results", expanded=True):
                    results = st.session_state.adversarial_results

                    st.markdown("### 📊 Performance Summary")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("✅ Final Approval Rate", f"{results.get('final_approval_rate', 0):.1f}%")
                    col2.metric("🔄 Iterations Completed", len(results.get('iterations', [])))
                    col3.metric("💰 Total Cost (USD)", f"${results.get('cost_estimate_usd', 0):.4f}")
                    col4.metric("🤿 Total Tokens",
                                f"{results.get('tokens', {}).get('prompt', 0) + results.get('tokens', {}).get('completion', 0):,}")

                    # Detailed metrics tabs
                    metrics_tab1, metrics_tab2, metrics_tab3, metrics_tab4 = st.tabs(
                        ["📈 Confidence Trend", "🏆 Model Performance", "🧮 Issue Analysis", "📊 Advanced Analytics"])

                    with metrics_tab1:
                        # Confidence trend chart
                        if results.get('iterations'):
                            confidence_history = [iter.get("approval_check", {}).get("approval_rate", 0)
                                                  for iter in results.get('iterations', [])]
                            if confidence_history:
                                # Enhanced visualization
                                import pandas as pd
                                df = pd.DataFrame({'confidence': confidence_history})
                                st.line_chart(df)

                                # Trend line
                                x = np.arange(len(confidence_history))
                                y = np.array(confidence_history)
                                z = np.polyfit(x, y, 1)
                                p = np.poly1d(z)
                                st.line_chart(pd.DataFrame({'trend': p(x)}))

                                max_confidence = max(confidence_history)
                                min_confidence = min(confidence_history)
                                avg_confidence = sum(confidence_history) / len(confidence_history)

                                st.line_chart(confidence_history)
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("📈 Peak Confidence", f"{max_confidence:.1f}%")
                                col2.metric("📉 Lowest Confidence", f"{min_confidence:.1f}%")
                                col3.metric("📊 Average Confidence", f"{avg_confidence:.1f}%")
                                col4.metric("📊 Final Confidence", f"{confidence_history[-1]:.1f}%")

                                # Confidence improvement
                                if len(confidence_history) > 1:
                                    improvement = confidence_history[-1] - confidence_history[0]
                                    if improvement > 0:
                                        st.success(f"🚀 Confidence improved by {improvement:.1f}%")
                                    elif improvement < 0:
                                        st.warning(f"⚠️ Confidence decreased by {abs(improvement):.1f}%")
                                    else:
                                        st.info("➡️ Confidence remained stable")

                    with metrics_tab2:
                        # Model performance analysis
                        if st.session_state.get("adversarial_model_performance"):
                            model_performance = st.session_state.adversarial_model_performance
                            st.markdown("### 🏆 Top Performing Models")

                            # Sort models by score
                            sorted_models = sorted(model_performance.items(), key=lambda x: x[1].get("score", 0),
                                                   reverse=True)

                            # Display top 5 models with enhanced visualization
                            for i, (model_id, perf) in enumerate(sorted_models[:5]):
                                score = perf.get("score", 0)
                                issues = perf.get("issues_found", 0)
                                st.progress(min(score / 100, 1.0), 
                                            text=f"#{i + 1} {model_id} - Score: {score}, Issues Found: {issues}")
                        else:
                            st.info("No model performance data available.")

                    with metrics_tab3:
                        # Issue analysis
                        if results.get('iterations'):
                            # Aggregate issue data
                            total_issues = 0
                            severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
                            category_counts = {}

                            for iteration in results.get('iterations', []):
                                critiques = iteration.get("critiques", [])
                                for critique in critiques:
                                    critique_json = critique.get("critique_json", {})
                                    issues = critique_json.get("issues", [])
                                    total_issues += len(issues)

                                    for issue in issues:
                                        # Count by severity
                                        severity = issue.get("severity", "low").lower()
                                        if severity in severity_counts:
                                            severity_counts[severity] += 1

                                        # Count by category
                                        category = issue.get("category", "uncategorized")
                                        category_counts[category] = category_counts.get(category, 0) + 1

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### 🎯 Issue Severity Distribution")
                                severity_counts = {}
                                for iteration in results.get('iterations', []):
                                    for critique in iteration.get("critiques", []):
                                        if critique.get("critique_json"):
                                            for issue in _safe_list(critique["critique_json"], "issues"):
                                                severity = issue.get("severity", "low").lower()
                                                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                                if severity_counts:
                                    st.bar_chart(severity_counts)
                            with col2:
                                st.markdown("### 📚 Issue Categories")
                                # Show top 5 categories
                                sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                                for category, count in sorted_categories[:5]:
                                    st.write(f"🏷️ {category}: {count}")

                            st.metric("🔍 Total Issues Found", total_issues)
                    with metrics_tab4:
                        st.markdown("### 📊 Advanced Analytics")
                        analytics = analytics_manager.generate_advanced_analytics(results)
                        st.json(analytics)

                    st.markdown("### 📄 Final Hardened Protocol")
                    st.code(results.get('final_sop', ''), language="markdown")

                    # Export options
                    st.markdown("### 📁 Export Results")
                    st.text_input("Watermark for PDF Export", key="adversarial_pdf_watermark")
                    st.text_area("Custom CSS for HTML Export", key="adversarial_custom_css")
                    export_col1, export_col2, export_col3, export_col4 = st.columns(4)
                    
                    with export_col1:
                        if st.button("📄 Export PDF", key="adversarial_export_pdf", use_container_width=True):
                            if results:
                                pdf_bytes = generate_pdf_report(results, st.session_state.pdf_watermark)
                                st.download_button(
                                    label="📥 Download PDF",
                                    data=pdf_bytes,
                                    file_name=f"adversarial_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            else:
                                st.warning("No results to export.")
                    
                    with export_col2:
                        if st.button("📝 DOCX", key="adversarial_export_docx", use_container_width=True):
                            if results:
                                docx_bytes = generate_docx_report(results)
                                st.download_button(
                                    label="📥 Download DOCX",
                                    data=docx_bytes,
                                    file_name=f"adversarial_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    use_container_width=True
                                )
                            else:
                                st.warning("No results to export.")
                    
                    with export_col3:
                        if st.button("📊 Export HTML", key="adversarial_export_html", use_container_width=True):
                            if results:
                                html_content = generate_html_report(results, st.session_state.custom_css)
                                st.download_button(
                                    label="📥 Download HTML",
                                    data=html_content,
                                    file_name=f"adversarial_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html",
                                    use_container_width=True
                                )
                            else:
                                st.warning("No results to export.")
                    
                    with export_col4:
                        if st.button("📋 Export JSON", key="adversarial_export_json", use_container_width=True):
                            if results:
                                json_str = json.dumps(results, indent=2, default=str)
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json_str,
                                    file_name=f"adversarial_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                            else:
                                st.warning("No results to export.")
                    export_col5, = st.columns(1)
                    with export_col5:
                        if st.button("📄 Export LaTeX", key="adversarial_export_latex", use_container_width=True):
                            if results:
                                latex_str = generate_latex_report(results)
                                st.download_button(
                                    label="📥 Download LaTeX",
                                    data=latex_str,
                                    file_name=f"adversarial_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
                                    mime="application/x-latex",
                                    use_container_width=True
                                )
                            else:
                                st.warning("No results to export.")
                    export_col6, = st.columns(1)
                    with export_col6:
                        if st.button("💬 Send to Discord", key="adversarial_send_to_discord", use_container_width=True):
                            if st.session_state.discord_webhook_url:
                                message = f"Adversarial testing complete! Final approval rate: {results.get('final_approval_rate', 0.0):.1f}%"
                                send_discord_notification(st.session_state.discord_webhook_url, message)
                            else:
                                st.warning("Please configure the Discord webhook URL in the sidebar.")
                    export_col7, = st.columns(1)
                    with export_col7:
                        if st.button("💬 Send to Teams", key="adversarial_send_to_teams", use_container_width=True):
                            if st.session_state.msteams_webhook_url:
                                message = f"Adversarial testing complete! Final approval rate: {results.get('final_approval_rate', 0.0):.1f}%"
                                send_msteams_notification(st.session_state.msteams_webhook_url, message)
                            else:
                                st.warning("Please configure the Microsoft Teams webhook URL in the sidebar.")
                    export_col8, = st.columns(1)
                    with export_col8:
                        if st.button("🚀 Send Webhook", key="adversarial_send_webhook", use_container_width=True):
                            if st.session_state.generic_webhook_url:
                                payload = {"text": f"Adversarial testing complete! Final approval rate: {results.get('final_approval_rate', 0.0):.1f}%"}
                                send_generic_webhook(st.session_state.generic_webhook_url, payload)
                            else:
                                st.warning("Please configure the generic webhook URL in the sidebar.")
                    export_col9, = st.columns(1)
                    with export_col9:
                        if st.button("📋 Generate Compliance Report", key="adversarial_generate_compliance_report", use_container_width=True):
                            if results and st.session_state.compliance_requirements:
                                compliance_report = generate_compliance_report(results, st.session_state.compliance_requirements)
                                st.download_button(
                                    label="📥 Download Compliance Report",
                                    data=compliance_report,
                                    file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown",
                                    use_container_width=True
                                )
                            else:
                                st.warning("No results or compliance requirements to generate a report.")

def render_report_templates_ui():
    """Render the report templates UI."""
    st.markdown("""
    > Manage your custom report templates here. Create new templates or view existing ones.
    """)
    st.markdown("<br>", unsafe_allow_html=True)

    if "report_templates" not in st.session_state:
        st.session_state.report_templates = _load_report_templates()

    with st.container(border=True): # Create New Template Section
        st.markdown("### Create New Template")
        new_template_name = st.text_input("Template Name")
        new_template_content = st.text_area("Template Content (JSON)", height=200)
        if st.button("Save Template", type="primary"):
            if new_template_name and new_template_content:
                try:
                    template_data = json.loads(new_template_content)
                    st.session_state.report_templates[new_template_name] = template_data
                    with open("report_templates.json", "w") as f:
                        json.dump(st.session_state.report_templates, f, indent=4)
                    st.success(f"Template '{new_template_name}' saved.")
                    _load_report_templates.clear()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format.")
            else:
                st.warning("Please provide a name and content for the template.")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True): # Existing Templates Section
        st.markdown("### Existing Templates")
        st.markdown("<div class='template-container'>", unsafe_allow_html=True)
        for template_name, template_content in st.session_state.report_templates.items():
            st.markdown(f"""
            <div class="template-card">
                <h4>{template_name}</h4>
                <p>{json.dumps(template_content, indent=2)}</p>
                <button class="stButton secondary-button" onclick="
                    var event = new CustomEvent('streamlit:setComponentValue', {{detail: {{key: 'edit_template_{template_name}', value: true}}}});
                    window.parent.document.dispatchEvent(event);
                ">Edit</button>
                <button class="stButton secondary-button" onclick="
                    var event = new CustomEvent('streamlit:setComponentValue', {{detail: {{key: 'delete_template_{template_name}', value: true}}}});
                    window.parent.document.dispatchEvent(event);
                ">Delete</button>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Handle edit/delete actions (simplified for now, full implementation would involve more state management)
        for template_name in st.session_state.report_templates.keys():
            if st.session_state.get(f'edit_template_{template_name}') :
                st.session_state[f'edit_template_{template_name}'] = False # Reset
                st.info(f"Editing template: {template_name} (functionality to be implemented)")
            if st.session_state.get(f'delete_template_{template_name}') :
                st.session_state[f'delete_template_{template_name}'] = False # Reset
                del st.session_state.report_templates[template_name]
                with open("report_templates.json", "w") as f:
                    json.dump(st.session_state.report_templates, f, indent=4)
                st.success(f"Template '{template_name}' deleted.")
                _load_report_templates.clear()
                st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
