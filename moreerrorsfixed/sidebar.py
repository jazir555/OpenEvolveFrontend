import streamlit as st
from providercatalogue import get_providers
from session_utils import reset_defaults, save_user_preferences
from openevolve_integration import OpenEvolveAPI
import json


# It's good practice to define default parameter functions
def get_default_generation_params():
    """Returns a dictionary of default generation parameters."""
    return {
        "temperature": 0.7,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4096,
        "seed": 42,
        "reasoning_effort": "medium",
    }


def get_default_evolution_params():
    """Returns a dictionary of default evolution parameters."""
    return {
        "max_iterations": 100,
        "population_size": 10,
        "num_islands": 1,
        "migration_interval": 50,
        "migration_rate": 0.1,
        "archive_size": 100,
        "elite_ratio": 0.1,
        "exploration_ratio": 0.2,
        "exploitation_ratio": 0.7,
        "checkpoint_interval": 10,
        "language": "python",
        "file_suffix": ".py",
        "feature_dimensions": ["complexity", "diversity"],
        "feature_bins": 10,
        "diversity_metric": "edit_distance",
    }


def load_settings_for_scope():
    """
    Loads parameters into session_state for the UI based on the selected scope.
    It applies settings hierarchically: Global -> Provider -> Model.
    """
    scope = st.session_state.get("settings_scope", "Global")
    provider = st.session_state.get("provider")
    model = st.session_state.get("model")

    # Start with base defaults
    gen_params = get_default_generation_params()
    evo_params = get_default_evolution_params()

    # Layer 1: Global settings
    global_settings = st.session_state.get("parameter_settings", {}).get("global", {})
    gen_params.update(global_settings.get("generation", {}))
    evo_params.update(global_settings.get("evolution", {}))

    # Layer 2: Provider settings (if scope is Provider or Model)
    if (scope == "Provider" or scope == "Model") and provider:
        provider_settings = (
            st.session_state.get("parameter_settings", {})
            .get("providers", {})
            .get(provider, {})
        )
        if "settings" in provider_settings:
            gen_params.update(provider_settings["settings"].get("generation", {}))
            evo_params.update(provider_settings["settings"].get("evolution", {}))

    # Layer 3: Model settings (if scope is Model)
    if scope == "Model" and provider and model:
        model_settings = (
            st.session_state.get("parameter_settings", {})
            .get("providers", {})
            .get(provider, {})
            .get("models", {})
            .get(model, {})
        )
        gen_params.update(model_settings.get("generation", {}))
        evo_params.update(model_settings.get("evolution", {}))

    # Update session_state for UI widgets
    for key, value in gen_params.items():
        st.session_state[key] = value
    for key, value in evo_params.items():
        st.session_state[key] = value


def on_provider_change():
    """Handler for provider change, resets defaults and loads new settings."""
    reset_defaults()
    load_settings_for_scope()


def display_sidebar():
    # Initialize settings structures in session state if they don't exist
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {}
    if "parameter_settings" not in st.session_state:
        st.session_state.parameter_settings = {
            "global": {
                "generation": get_default_generation_params(),
                "evolution": get_default_evolution_params(),
            },
            "providers": {},
        }
    if "settings_scope" not in st.session_state:
        st.session_state.settings_scope = "Global"

    # Load initial settings into UI state
    load_settings_for_scope()

    with st.sidebar:
        st.markdown(
            '''
        <div style="text-align: center; padding: 10px;">
            <h2 style="color: #4a6fa5;">🧬 OpenEvolve</h2>
        </div>
        ''',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Quick start guide
        with st.expander("📖 Quick Start", expanded=False):
            st.markdown(
                '''
            **Getting Started:**
            1. Configure your LLM provider
            2. Enter your content in the main area
            3. Choose between Evolution or Adversarial Testing
            4. Monitor the results
            
            **Tips:**
            - Use Adversarial Testing for security hardening
            - Start with general templates for protocols
            - Save your work with project settings
            '''
            )

        st.markdown("---")

        st.title("⚙️ Provider Configuration")
        st.caption(
            "Controls the 'Evolution' tab. Adversarial Testing always uses OpenRouter."
        )

        st.markdown("---")
        if (
            "openevolve_api_instance" not in st.session_state
            or st.session_state.openevolve_api_instance.base_url
            != st.session_state.openevolve_base_url
            or st.session_state.openevolve_api_instance.api_key
            != st.session_state.openevolve_api_key
        ):
            st.session_state.openevolve_api_instance = OpenEvolveAPI(
                base_url=st.session_state.openevolve_base_url,
                api_key=st.session_state.openevolve_api_key,
            )
        api = st.session_state.openevolve_api_instance
        providers = get_providers(api)

        st.selectbox(
            "Provider",
            list(providers.keys()),
            key="provider",
            on_change=on_provider_change,
        )

        with st.form("provider_configuration_form"):
            provider_info = providers[st.session_state.provider]

            st.text_input("API Key", type="password", key="api_key")
            st.text_input("Base URL", key="base_url")

            if loader := provider_info.get("loader"):
                models = loader(st.session_state.api_key)
                st.selectbox(
                    "Model", models, key="model", on_change=load_settings_for_scope
                )
            else:
                st.text_input("Model", key="model", on_change=load_settings_for_scope)

            st.text_area("Extra Headers (JSON)", key="extra_headers")
            st.form_submit_button("Apply Provider Configuration")

        st.markdown("---")

        # SETTINGS SCOPE SELECTOR
        st.subheader("Parameter Scope")
        st.radio(
            "Settings Level",
            ["Global", "Provider", "Model"],
            key="settings_scope",
            horizontal=True,
            on_change=load_settings_for_scope,
            help="Select the scope for viewing and saving parameters. Settings are inherited from Global -> Provider -> Model.",
        )

        with st.form("generation_parameters_form"):
            st.subheader("Generation Parameters")
            st.slider("Temperature", 0.0, 2.0, key="temperature", step=0.1)
            st.slider("Top-P", 0.0, 1.0, key="top_p", step=0.1)
            st.slider(
                "Frequency Penalty", -2.0, 2.0, key="frequency_penalty", step=0.1
            )
            st.slider("Presence Penalty", -2.0, 2.0, key="presence_penalty", step=0.1)
            st.number_input("Max Tokens", 1, 100000, key="max_tokens")
            st.number_input("Seed", key="seed")
            st.selectbox(
                "Reasoning Effort",
                ["low", "medium", "high"],
                key="reasoning_effort",
            )
            if st.form_submit_button("Apply Generation Parameters"):
                scope = st.session_state.settings_scope
                provider = st.session_state.get("provider")
                model = st.session_state.get("model")

                gen_settings_to_save = {
                    "temperature": st.session_state.temperature,
                    "top_p": st.session_state.top_p,
                    "frequency_penalty": st.session_state.frequency_penalty,
                    "presence_penalty": st.session_state.presence_penalty,
                    "max_tokens": st.session_state.max_tokens,
                    "seed": st.session_state.seed,
                    "reasoning_effort": st.session_state.reasoning_effort,
                }

                if scope == "Global":
                    st.session_state.parameter_settings["global"][
                        "generation"
                    ] = gen_settings_to_save
                    st.success("Global generation parameters saved!")
                    st.rerun()
                elif scope == "Provider" and provider:
                    if (
                        provider
                        not in st.session_state.parameter_settings["providers"]
                    ):
                        st.session_state.parameter_settings["providers"][
                            provider
                        ] = {"settings": {}, "models": {}}
                    st.session_state.parameter_settings["providers"][provider][
                        "settings"
                    ]["generation"] = gen_settings_to_save
                    st.success(
                        f"Provider-level ({provider}) generation parameters saved!"
                    )
                    st.rerun()
                elif scope == "Model" and provider and model:
                    if (
                        provider
                        not in st.session_state.parameter_settings["providers"]
                    ):
                        st.session_state.parameter_settings["providers"][
                            provider
                        ] = {"settings": {}, "models": {}}
                    if (
                        model
                        not in st.session_state.parameter_settings["providers"][
                            provider
                        ]["models"]
                    ):
                        st.session_state.parameter_settings["providers"][provider][
                            "models"
                        ][model] = {}
                    st.session_state.parameter_settings["providers"][provider][
                        "models"
                    ][model]["generation"] = gen_settings_to_save
                    st.success(f"Model-level ({model}) generation parameters saved!")
                    st.rerun()
                else:
                    st.warning(
                        f"Cannot save settings for scope '{scope}'. Provider or model not selected."
                    )

        st.markdown("---")
        with st.form("evolution_parameters_form"):
            st.subheader("Evolution Parameters")
            st.number_input("Max Iterations", 1, 200, key="max_iterations")
            st.number_input("Population Size", 1, 100, key="population_size")
            st.number_input("Number of Islands", 1, 10, key="num_islands")
            st.number_input("Migration Interval", 1, 100, key="migration_interval")
            st.slider("Migration Rate", 0.0, 1.0, key="migration_rate", step=0.01)
            st.number_input("Archive Size", 0, 100, key="archive_size")
            st.slider("Elite Ratio", 0.0, 1.0, key="elite_ratio", step=0.01)
            st.slider(
                "Exploration Ratio", 0.0, 1.0, key="exploration_ratio", step=0.01
            )
            st.slider(
                "Exploitation Ratio", 0.0, 1.0, key="exploitation_ratio", step=0.01
            )
            st.number_input("Checkpoint Interval", 1, 100, key="checkpoint_interval")
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
            st.text_input("File Suffix", key="file_suffix")
            st.multiselect(
                "Feature Dimensions",
                ["complexity", "diversity", "readability", "performance"],
                key="feature_dimensions",
            )
            st.number_input("Feature Bins", 1, 100, key="feature_bins")
            st.selectbox(
                "Diversity Metric",
                ["edit_distance", "cosine_similarity", "levenshtein_distance"],
                key="diversity_metric",
            )
            if st.form_submit_button("Apply Evolution Parameters"):
                scope = st.session_state.settings_scope
                provider = st.session_state.get("provider")
                model = st.session_state.get("model")

                evo_settings_to_save = {
                    "max_iterations": st.session_state.max_iterations,
                    "population_size": st.session_state.population_size,
                    "num_islands": st.session_state.num_islands,
                    "migration_interval": st.session_state.migration_interval,
                    "migration_rate": st.session_state.migration_rate,
                    "archive_size": st.session_state.archive_size,
                    "elite_ratio": st.session_state.elite_ratio,
                    "exploration_ratio": st.session_state.exploration_ratio,
                    "exploitation_ratio": st.session_state.exploitation_ratio,
                    "checkpoint_interval": st.session_state.checkpoint_interval,
                    "language": st.session_state.language,
                    "file_suffix": st.session_state.file_suffix,
                    "feature_dimensions": st.session_state.feature_dimensions,
                    "feature_bins": st.session_state.feature_bins,
                    "diversity_metric": st.session_state.diversity_metric,
                }

                if scope == "Global":
                    st.session_state.parameter_settings["global"][
                        "evolution"
                    ] = evo_settings_to_save
                    st.success("Global evolution parameters saved!")
                    st.rerun()
                elif scope == "Provider" and provider:
                    if (
                        provider
                        not in st.session_state.parameter_settings["providers"]
                    ):
                        st.session_state.parameter_settings["providers"][
                            provider
                        ] = {"settings": {}, "models": {}}
                    st.session_state.parameter_settings["providers"][provider][
                        "settings"
                    ]["evolution"] = evo_settings_to_save
                    st.success(
                        f"Provider-level ({provider}) evolution parameters saved!"
                    )
                    st.rerun()
                elif scope == "Model" and provider and model:
                    if (
                        provider
                        not in st.session_state.parameter_settings["providers"]
                    ):
                        st.session_state.parameter_settings["providers"][
                            provider
                        ] = {"settings": {}, "models": {}}
                    if (
                        model
                        not in st.session_state.parameter_settings["providers"][
                            provider
                        ]["models"]
                    ):
                        st.session_state.parameter_settings["providers"][provider][
                            "models"
                        ][model] = {}
                    st.session_state.parameter_settings["providers"][provider][
                        "models"
                    ][model]["evolution"] = evo_settings_to_save
                    st.success(f"Model-level ({model}) evolution parameters saved!")
                    st.rerun()
                else:
                    st.warning(
                        f"Cannot save settings for scope '{scope}'. Provider or model not selected."
                    )

        st.markdown("---")
        with st.form("checkpointing_form"):
            st.subheader("Checkpointing")

            action = st.radio(
                "Action", ["Save Checkpoint", "Load Checkpoint"], key="checkpoint_action"
            )

            checkpoints = api.get_checkpoints()
            if checkpoints:
                st.selectbox(
                    "Load Checkpoint", options=checkpoints, key="selected_checkpoint"
                )
            else:
                st.info("No checkpoints available to load.")

            if st.form_submit_button("Execute Action"):
                if st.session_state.checkpoint_action == "Save Checkpoint":
                    st.session_state.save_checkpoint_triggered = True
                elif st.session_state.checkpoint_action == "Load Checkpoint":
                    if checkpoints:
                        st.session_state.load_checkpoint_triggered = True
                    else:
                        st.warning("No checkpoints available to load.")

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
        theme_label = (
            f"{theme_emoji} Switch to {'Dark' if current_theme == 'light' else 'Light'} Mode"
        )

        if st.button(theme_label, key="theme_toggle_btn", use_container_width=True):
            if st.session_state.get("theme", "light") == "light":
                st.session_state.theme = "dark"
            else:
                st.session_state.theme = "light"
            st.rerun()

        # Fallback theme selector
        st.caption("Alternative theme selector:")
        theme_options = ["light", "dark"]
        selected_theme = st.selectbox(
            "Select Theme",
            theme_options,
            index=(
                theme_options.index(current_theme)
                if current_theme in theme_options
                else 0
            ),
            label_visibility="collapsed",
        )

        if selected_theme != current_theme:
            st.session_state.theme = selected_theme
            st.session_state.user_preferences["theme"] = (
                selected_theme  # Update preferences
            )
            # Use JavaScript to apply theme changes immediately
            st.markdown(
                f'''
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
                ''',
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
