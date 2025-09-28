import streamlit as st
from providercatalogue import get_providers
from session_utils import reset_defaults, save_user_preferences
from openevolve_integration import OpenEvolveAPI


def render_sidebar():
    with st.sidebar:
        # Welcome section
        st.markdown(
            """
        <div style="text-align: center; padding: 10px;">
            <h2 style="color: #4a6fa5;">🧬 OpenEvolve</h2>
            <p style="color: #666; font-size: 0.9em;">AI-Powered Content Evolution & Testing</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Quick start guide
        with st.expander("📖 Quick Start", expanded=False):
            st.markdown("""
            **Getting Started:**
            1. Configure your LLM provider
            2. Enter your content in the main area
            3. Choose between Evolution or Adversarial Testing
            4. Monitor the results
            
            **Tips:**
            - Use Adversarial Testing for security hardening
            - Start with general templates for protocols
            - Save your work with project settings
            """)

        st.markdown("---")

        st.title("⚙️ Provider Configuration")
        st.caption(
            "Controls the 'Evolution' tab. Adversarial Testing always uses OpenRouter."
        )

        st.markdown("---")

        api = OpenEvolveAPI(
            base_url=st.session_state.openevolve_base_url,
            api_key=st.session_state.openevolve_api_key,
        )
        providers = get_providers(api)
        st.selectbox(
            "Provider", list(providers.keys()), key="provider", on_change=reset_defaults
        )

        provider_info = providers[st.session_state.provider]

        st.text_input("API Key", type="password", key="api_key")
        st.text_input("Base URL", key="base_url")

        if loader := provider_info.get("loader"):
            models = loader(st.session_state.api_key)
            st.selectbox("Model", models, key="model")
        else:
            st.text_input("Model", key="model")

        st.text_area("Extra Headers (JSON)", key="extra_headers")

        st.markdown("---")
        st.subheader("Generation Parameters")
        st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="temperature")
        st.slider("Top-P", 0.0, 1.0, 1.0, 0.1, key="top_p")
        st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1, key="frequency_penalty")
        st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1, key="presence_penalty")
        st.number_input("Max Tokens", 1, 100000, 4096, key="max_tokens")
        st.number_input("Seed", value=42, key="seed")
        st.selectbox(
            "Reasoning Effort",
            ["low", "medium", "high"],
            index=1,
            key="reasoning_effort",
        )

        st.markdown("---")
        st.subheader("Evolution Parameters")
        st.number_input("Max Iterations", 1, 200, 100, key="max_iterations")
        st.number_input("Population Size", 1, 100, 10, key="population_size")
        st.number_input("Number of Islands", 1, 10, 1, key="num_islands")
        st.number_input("Migration Interval", 1, 100, 50, key="migration_interval")
        st.slider("Migration Rate", 0.0, 1.0, 0.1, 0.01, key="migration_rate")
        st.number_input("Archive Size", 0, 100, 100, key="archive_size")
        st.slider("Elite Ratio", 0.0, 1.0, 0.1, 0.01, key="elite_ratio")
        st.slider("Exploration Ratio", 0.0, 1.0, 0.2, 0.01, key="exploration_ratio")
        st.slider("Exploitation Ratio", 0.0, 1.0, 0.7, 0.01, key="exploitation_ratio")
        st.number_input("Checkpoint Interval", 1, 100, 10, key="checkpoint_interval")
        st.selectbox(
            "Language",
            [
                "python",
                "javascript",
                "java",
                "cpp",
                "csharp",
                "go",
                "rust",
                "swift",
                "kotlin",
                "typescript",
                "document",
            ],
            key="language",
        )
        st.text_input("File Suffix", value=".py", key="file_suffix")
        st.multiselect(
            "Feature Dimensions",
            ["complexity", "diversity", "readability", "performance"],
            default=["complexity", "diversity"],
            key="feature_dimensions",
        )
        st.number_input("Feature Bins", 1, 100, 10, key="feature_bins")
        st.selectbox(
            "Diversity Metric",
            ["edit_distance", "cosine_similarity", "levenshtein_distance"],
            key="diversity_metric",
        )

        st.markdown("---")
        st.subheader("Checkpointing")
        if st.button("Save Checkpoint"):
            st.session_state.save_checkpoint_triggered = True

        checkpoints = api.get_checkpoints()
        if checkpoints:
            selected_checkpoint = st.selectbox(
                "Load Checkpoint", options=checkpoints, key="selected_checkpoint"
            )
            if st.button("Load Selected Checkpoint"):
                st.session_state.load_checkpoint_triggered = True
        else:
            st.info("No checkpoints available.")

        st.markdown("---")
        st.subheader("System Prompts")
        st.text_area("System Prompt", key="system_prompt", height=200)
        st.text_area(
            "Evaluator System Prompt", key="evaluator_system_prompt", height=200
        )

        st.markdown("---")
        st.subheader("Integrations")
        st.text_input(
            "Discord Webhook URL",
            key="discord_webhook_url",
            help="Enter your Discord webhook URL to receive notifications.",
        )
        st.text_input(
            "Microsoft Teams Webhook URL",
            key="msteams_webhook_url",
            help="Enter your Microsoft Teams webhook URL to receive notifications.",
        )
        st.text_input(
            "Generic Webhook URL",
            key="generic_webhook_url",
            help="Enter a generic webhook URL to receive notifications.",
        )

        st.markdown("---")
        st.subheader("Project Settings")
        st.text_input("Project Name", key="project_name")
        st.text_area("Project Description", key="project_description")
        st.checkbox(
            "Public Project",
            key="project_public",
            help="Make your project publicly accessible.",
        )
        st.text_input(
            "Project Password",
            key="project_password",
            type="password",
            help="Password-protect your public project.",
        )
        if st.button("Share Project"):
            if st.session_state.project_public:
                st.success(f"Shareable link: /shared/{st.session_state.project_name}")
            else:
                st.warning(
                    "Project is not public. Please make the project public to share."
                )

        # Theme toggle
        st.markdown("---")
        st.subheader("🎨 Theme Settings")

        # Add a link to the evaluator uploader
        st.sidebar.header("Customization")
        if st.sidebar.button("Custom Evaluators"):
            st.session_state.page = "evaluator_uploader"
        if st.sidebar.button("Custom Prompts"):
            st.session_state.page = "prompt_manager"
        if st.sidebar.button("Analytics Dashboard"):
            st.session_state.page = "analytics_dashboard"

        # Enhanced theme toggle with better UX
        current_theme = st.session_state.get("theme", "light")
        theme_emoji = "🌙" if current_theme == "light" else "☀️"
        theme_label = f"{theme_emoji} Switch to {'Dark' if current_theme == 'light' else 'Light'} Mode"

        if st.button(theme_label, key="theme_toggle_btn", use_container_width=True):
            from sessionstate import toggle_theme

            toggle_theme()
            st.rerun()

        # Fallback theme selector
        st.caption("Alternative theme selector:")
        theme_options = ["light", "dark"]
        selected_theme = st.selectbox(
            "Select Theme",
            theme_options,
            index=theme_options.index(current_theme)
            if current_theme in theme_options
            else 0,
            label_visibility="collapsed",
        )

        if selected_theme != current_theme:
            st.session_state.theme = selected_theme
            st.session_state.user_preferences["theme"] = (
                selected_theme  # Update preferences
            )
            # Use JavaScript to apply theme changes immediately
            st.markdown(
                f"""
                <script>
                const theme = "{selected_theme}";
                document.documentElement.setAttribute('data-theme', theme);
                
                // Apply theme to Streamlit elements if possible
                if (theme === 'dark') {{
                    document.body.style.backgroundColor = '#0e1117';
                    document.querySelector('.stApp').style.backgroundColor = '#0e1117';
                }} else {{
                    document.body.style.backgroundColor = 'white';
                    document.querySelector('.stApp').style.backgroundColor = 'white';
                }}
                </script>
                """,
                unsafe_allow_html=True,
            )

        # Additional user preferences
        st.markdown("---")
        st.subheader("⚙️ User Preferences")

        # Auto-save preference
        auto_save = st.checkbox(
            "Auto-save preferences",
            value=st.session_state.user_preferences.get("auto_save", True),
        )
        st.session_state.user_preferences["auto_save"] = auto_save

        # Font size preference
        font_size = st.selectbox(
            "Font Size",
            ["small", "medium", "large"],
            index=["small", "medium", "large"].index(
                st.session_state.user_preferences.get("font_size", "medium")
            ),
        )
        st.session_state.user_preferences["font_size"] = font_size

        # Save preferences button
        if st.button("💾 Save All Preferences"):
            if save_user_preferences():
                st.success("Preferences saved successfully!")
            else:
                st.error("Failed to save preferences.")

        # Status information
        st.markdown("---")
        st.subheader("ℹ️ Status")
        st.caption(f"Active Project: {st.session_state.project_name}")
        st.caption("Content Type: Auto-detected")
        st.caption(f"Evolution Running: {st.session_state.evolution_running}")
        st.caption(f"Adversarial Running: {st.session_state.adversarial_running}")
