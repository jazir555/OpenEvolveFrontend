import streamlit as st
from providercatalogue import get_providers
from session_utils import reset_defaults, save_user_preferences
from openevolve_integration import OpenEvolveAPI
import json
import os
import sys
import subprocess
import threading
import requests
import time
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

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
        logger.error(f"Error getting project root: {e}")
        # Fallback to current working directory
        return os.getcwd()

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


def create_tooltip_html(label, description):
    return f"""
    <div style="display: block; margin-bottom: 5px;">
        <span style="display: inline-block; vertical-align: middle;">{label}</span>
        <span class="tooltip-container" style="vertical-align: middle;">
            <span class="question-icon">?</span>
            <span class="tooltip-text">{description}</span>
        </span>
    </div>
    """


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

        # Inject custom CSS for tooltips and other fixes
        st.markdown("""
        <style>
        .tooltip-container {
            position: relative;
            display: inline-block;
            margin-left: 5px; /* Space between label and icon */
        }

        .question-icon {
            display: inline-block;
            width: 16px;
            height: 16px;
            line-height: 16px;
            text-align: center;
            border: 1px solid #ccc;
            border-radius: 50%;
            font-size: 10px;
            font-weight: bold;
            cursor: pointer;
            color: #555;
            background-color: #f0f0f0;
        }

        .tooltip-text {
            visibility: hidden;
            max-width: 250px; /* Max width for the tooltip */
            min-width: 100px; /* Min width to prevent squishing */
            background-color: #555;
            color: #fff;
            text-align: left; /* Align text left for better readability */
            border-radius: 6px;
            padding: 5px 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position above the icon */
            right: 0; /* Align to the right edge of the .tooltip-container */
            left: auto; /* Let browser calculate left */
            transform: translateX(0); /* No horizontal translation */
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
            white-space: normal; /* Allow text to wrap */
            word-wrap: break-word; /* Break long words */
            box-sizing: border-box; /* Include padding in width calculation */
        }

        .tooltip-container:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }

        /* Fix for st.multiselect text cutoff */
        .stMultiSelect {
            width: 100% !important;
        }

        /* Fix for st.radio background and appearance */
        .stRadio > label > div {
            background-color: #f0f2f6; /* Streamlit's default light gray */
            border-radius: 0.25rem;
            padding: 0.5rem;
            border: 1px solid #ccc; /* Add a border around the group */
        }


        </style>
        """, unsafe_allow_html=True)

        # Inject JavaScript for message timeouts
        st.markdown("""
        <script>
            function setupMessageTimeout() {
                const alerts = document.querySelectorAll('[role=\"alert\"]');
                alerts.forEach(alert => {
                    // Check if a timeout has already been set for this alert
                    if (!alert.dataset.timeoutId) {
                        const timeoutId = setTimeout(() => {
                            alert.style.opacity = '0';
                            alert.style.transition = 'opacity 1s ease-out';
                            setTimeout(() => {
                                alert.remove();
                            }, 1000); // Remove after fade out
                        }, 3000); // 3 seconds before starting fade out

                        alert.dataset.timeoutId = timeoutId; // Store timeout ID to prevent duplicates
                    }
                });
            }

            // Run on initial load
            setupMessageTimeout();

            // Use a MutationObserver to detect when new alert messages are added to the DOM
            const observer = new MutationObserver((mutationsList, observer) => {
                for (const mutation of mutationsList) {
                    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                        setupMessageTimeout();
                    }
                }
            });

            // Observe the entire document body for changes
            observer.observe(document.body, { childList: true, subtree: true });
        </script>
        """, unsafe_allow_html=True)

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
        # Initialize OpenEvolve API instance
        # Connects to the OpenEvolve backend service which should be running at localhost:8000
        if (
            "openevolve_api_instance" not in st.session_state
            or st.session_state.openevolve_api_instance.base_url
            != st.session_state.openevolve_base_url
            or st.session_state.openevolve_api_instance.api_key
            != st.session_state.openevolve_api_key
        ):
            # Initialize the OpenEvolve API instance
            # This connects to the OpenEvolve backend service
            api_base_url = st.session_state.get("openevolve_base_url", "http://localhost:8000")
            api_key = st.session_state.get("openevolve_api_key", "")
            st.session_state.openevolve_api_instance = OpenEvolveAPI(
                base_url=api_base_url,
                api_key=api_key,
            )
        
        @st.cache_resource(ttl=3600, show_spinner=False) # Cache for 1 hour, disable default spinner message
        def _get_cached_providers(_api_instance): # Added underscore
            return get_providers(_api_instance)

        api = st.session_state.openevolve_api_instance
        providers = {}
        provider_keys = []
        providers_available = False

        with st.spinner("Fetching LLM providers..."):
            fetched_providers = _get_cached_providers(api)
            if fetched_providers:
                providers = fetched_providers
                provider_keys = list(providers.keys())
                providers_available = True
            else:
                st.warning("No LLM providers configured. Please add providers to proceed.")

        if providers_available:
            # Ensure st.session_state.provider is initialized and valid
            if "provider" not in st.session_state or st.session_state.provider not in provider_keys:
                st.session_state.provider = provider_keys[0] # Default to first provider

            st.markdown(create_tooltip_html("Provider", "Select the LLM provider to use for evolution. Adversarial testing always uses OpenRouter."), unsafe_allow_html=True)
            st.selectbox(
                "Provider",
                provider_keys,
                key="provider",
                on_change=on_provider_change,
                label_visibility="hidden"
            )

            try: # This is line 269
                with st.form("provider_configuration_form"):
                    provider_info = None
                    try:
                        provider_info = providers[st.session_state.provider]
                    except KeyError:
                        st.error(f"Selected provider '{st.session_state.provider}' not found. Please select a valid provider.")
                        pass

                    st.markdown(create_tooltip_html("API Key", "Your API key for the selected provider. Keep this confidential."), unsafe_allow_html=True)
                    st.text_input("API Key", type="password", key="api_key", label_visibility="hidden")
                    
                    st.markdown(create_tooltip_html("Base URL", "The base URL for the provider's API endpoint."), unsafe_allow_html=True)
                    st.text_input("Base URL", key="base_url", label_visibility="hidden")

                    if loader := provider_info.get("loader"):
                        api_key = st.session_state.get("api_key")
                        if not api_key:
                            st.warning("API Key is required to load models for this provider.")
                            st.markdown(create_tooltip_html("Model", "The name or ID of the model to use from the selected provider."), unsafe_allow_html=True)
                            st.text_input("Model", key="model", label_visibility="hidden")
                        else:
                            try:
                                models = loader(api_key)
                                if not models or not isinstance(models, list): # Check if models list is empty or not a list
                                    st.warning("No models found for this provider with the given API Key, or models data is malformed.")
                                else:
                                    st.markdown(create_tooltip_html("Model", "The specific model to use from the selected provider."), unsafe_allow_html=True)
                                    st.selectbox(
                                        "Model", models, key="model", label_visibility="hidden"
                                    )
                                    
                                    # Multi-model ensemble configuration
                                    st.markdown(create_tooltip_html("Multi-Model Ensemble", "Enable multiple models for ensemble evolution with intelligent fallback"), unsafe_allow_html=True)
                                    enable_ensemble = st.checkbox("Enable Multi-Model Ensemble", value=False, key="enable_ensemble", help="Use multiple models for enhanced evolution")
                                    
                                    if enable_ensemble:
                                        st.markdown(create_tooltip_html("Primary Models", "Primary models with higher weights for main evolution"), unsafe_allow_html=True)
                                        primary_models = st.multiselect("Primary Models", models, key="primary_models", default=[models[0]] if models else [], help="Models with higher weights for main evolution")
                                        
                                        st.markdown(create_tooltip_html("Fallback Models", "Fallback models with lower weights for intelligent fallback"), unsafe_allow_html=True)
                                        fallback_models = st.multiselect("Fallback Models", models, key="fallback_models", default=[], help="Models with lower weights for fallback when primary models fail")
                                        
                                        st.markdown(create_tooltip_html("Primary Weight", "Weight for primary models in the ensemble"), unsafe_allow_html=True)
                                        primary_weight = st.slider("Primary Weight", 0.1, 2.0, 1.0, 0.1, key="primary_weight", help="Weight for primary models")
                                        
                                        st.markdown(create_tooltip_html("Fallback Weight", "Weight for fallback models in the ensemble"), unsafe_allow_html=True)
                                        fallback_weight = st.slider("Fallback Weight", 0.0, 1.0, 0.3, 0.05, key="fallback_weight", help="Weight for fallback models")
                                        
                                        # Update session state with ensemble settings
                                        st.session_state.primary_models = primary_models
                                        st.session_state.fallback_models = fallback_models
                                        st.session_state.primary_weight = primary_weight
                                        st.session_state.fallback_weight = fallback_weight
                            except Exception as e:
                                st.error(f"Error loading models: {e}. Please check your API Key and Base URL.")
                                st.markdown(create_tooltip_html("Model", "The name or ID of the model to use from the selected provider."), unsafe_allow_html=True)
                                st.text_input("Model", key="model", label_visibility="hidden")
                    else:
                        st.markdown(create_tooltip_html("Model", "The name or ID of the model to use from the selected provider."), unsafe_allow_html=True)
                        st.text_input("Model", key="model", label_visibility="hidden")

                    st.markdown(create_tooltip_html("Extra Headers (JSON)", "Additional HTTP headers to send with API requests, in JSON format."), unsafe_allow_html=True)
                    st.text_area("Extra Headers (JSON)", key="extra_headers", label_visibility="hidden")
                    try:
                        if st.session_state.extra_headers:
                            json.loads(st.session_state.extra_headers)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format for Extra Headers.")
                    if st.form_submit_button("Apply Provider Configuration"):
                        st.success("Provider configuration applied.")
            except Exception as e:
                st.error(f"An unexpected error occurred in Provider Configuration: {e}")
        else:
            # Display a message if providers are not available, but don't stop the app
            st.info("Provider configuration is unavailable. Please ensure the backend is running and accessible.")



        # SETTINGS SCOPE SELECTOR
        st.subheader("Parameter Scope")
        st.markdown(create_tooltip_html("Settings Level", "Select the scope for viewing and saving parameters. Settings are inherited from Global -> Provider -> Model."), unsafe_allow_html=True)
        st.radio(
            "Settings Level",
            ["Global", "Provider", "Model"],
            key="settings_scope",
            horizontal=True,
            on_change=load_settings_for_scope,
            label_visibility="hidden"
        )

        with st.form("generation_parameters_form"):
            st.subheader("Generation Parameters")
            st.markdown(create_tooltip_html("Temperature", "Controls the randomness of the output. Higher values mean more creative, lower values mean more deterministic."), unsafe_allow_html=True)
            st.slider("Temperature", 0.0, 2.0, key="temperature", step=0.1, label_visibility="hidden")
            st.markdown(create_tooltip_html("Top-P", "Controls the diversity of the output by sampling from the most probable tokens whose cumulative probability exceeds top_p."), unsafe_allow_html=True)
            st.slider("Top-P", 0.0, 1.0, key="top_p", step=0.1, label_visibility="hidden")
            st.markdown(create_tooltip_html("Frequency Penalty", "Penalizes new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim."), unsafe_allow_html=True)
            st.slider(
                "Frequency Penalty", -2.0, 2.0, key="frequency_penalty", step=0.1, label_visibility="hidden"
            )
            st.markdown(create_tooltip_html("Presence Penalty", "Penalizes new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics."), unsafe_allow_html=True)
            st.slider("Presence Penalty", -2.0, 2.0, key="presence_penalty", step=0.1, label_visibility="hidden")
            st.markdown(create_tooltip_html("Max Tokens", "The maximum number of tokens to generate in the completion."), unsafe_allow_html=True)
            st.number_input("Max Tokens", 1, 100000, key="max_tokens", label_visibility="hidden")
            st.markdown(create_tooltip_html("Seed", "A seed for reproducible generation. Use the same seed to get the same output for the same input."), unsafe_allow_html=True)
            st.number_input("Seed", key="seed", label_visibility="hidden")
            st.markdown(create_tooltip_html("Reasoning Effort", "Controls the computational effort the model expends on reasoning. Higher effort may lead to better quality but slower generation."), unsafe_allow_html=True)
            st.selectbox(
                "Reasoning Effort",
                ["low", "medium", "high"],
                key="reasoning_effort",
                label_visibility="hidden"
            )
            if st.form_submit_button("Apply Generation Parameters"):
                try:
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
                    else:
                        st.warning(
                            f"Cannot save settings for scope '{scope}'. Provider or model not selected."
                        )
                except Exception as e:
                    st.error(f"Failed to apply generation parameters: {e}")
        st.markdown("---")
        with st.form("evolution_parameters_form"):
            st.subheader("Evolution Parameters")
            st.markdown(create_tooltip_html("Max Iterations", "The maximum number of evolutionary iterations to run."), unsafe_allow_html=True)
            st.number_input("Max Iterations", 1, 200, key="max_iterations", label_visibility="hidden")
            st.markdown(create_tooltip_html("Population Size", "The number of individuals (solutions) in each generation."), unsafe_allow_html=True)
            st.number_input("Population Size", 1, 100, key="population_size", label_visibility="hidden")
            st.markdown(create_tooltip_html("Number of Islands", "The number of independent evolutionary populations (islands) to maintain."), unsafe_allow_html=True)
            st.number_input("Number of Islands", 1, 10, key="num_islands", label_visibility="hidden")
            st.markdown(create_tooltip_html("Migration Interval", "How often individuals migrate between islands (in iterations)."), unsafe_allow_html=True)
            st.number_input("Migration Interval", 1, 100, key="migration_interval", label_visibility="hidden")
            st.markdown(create_tooltip_html("Migration Rate", "The proportion of individuals that migrate between islands during a migration event."), unsafe_allow_html=True)
            st.slider("Migration Rate", 0.0, 1.0, key="migration_rate", step=0.01, label_visibility="hidden")
            st.markdown(create_tooltip_html("Archive Size", "The maximum number of unique, high-performing solutions to store in the archive."), unsafe_allow_html=True)
            st.number_input("Archive Size", 0, 100, key="archive_size", label_visibility="hidden")
            st.markdown(create_tooltip_html("Elite Ratio", "The proportion of the best individuals from the current generation that are guaranteed to survive to the next generation without modification."), unsafe_allow_html=True)
            st.slider("Elite Ratio", 0.0, 1.0, key="elite_ratio", step=0.01, label_visibility="hidden")
            st.markdown(create_tooltip_html("Exploration Ratio", "The proportion of the population dedicated to exploring new solution spaces."), unsafe_allow_html=True)
            st.slider(
                "Exploration Ratio", 0.0, 1.0, key="exploration_ratio", step=0.01, label_visibility="hidden"
            )
            st.markdown(create_tooltip_html("Exploitation Ratio", "The proportion of the population dedicated to refining existing promising solutions."), unsafe_allow_html=True)
            st.slider(
                "Exploitation Ratio", 0.0, 1.0, key="exploitation_ratio", step=0.01, label_visibility="hidden"
            )
            st.markdown(create_tooltip_html("Checkpoint Interval", "How often to save the state of the evolution process (in iterations)."), unsafe_allow_html=True)
            st.number_input("Checkpoint Interval", 1, 100, key="checkpoint_interval", label_visibility="hidden")
            st.markdown(create_tooltip_html("Language", "The programming language or document type of the solutions being evolved."), unsafe_allow_html=True)
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
                label_visibility="hidden"
            )
            st.markdown(create_tooltip_html("File Suffix", "The file extension for the generated solutions (e.g., .py, .js)."), unsafe_allow_html=True)
            st.text_input("File Suffix", key="file_suffix", label_visibility="hidden")
            st.markdown(create_tooltip_html("Feature Dimensions", "The criteria used to evaluate and diversify solutions (e.g., complexity, diversity, readability, performance)."), unsafe_allow_html=True)
            st.multiselect(
                "Feature Dimensions",
                ["complexity", "diversity", "readability", "performance"],
                key="feature_dimensions_sidebar",
                label_visibility="hidden"
            )
            st.markdown(create_tooltip_html("Feature Bins", "The number of bins to use for discretizing feature dimensions in quality diversity algorithms."), unsafe_allow_html=True)
            st.number_input("Feature Bins", 1, 100, key="feature_bins", label_visibility="hidden")
            st.markdown(create_tooltip_html("Diversity Metric", "The metric used to measure the diversity between solutions."), unsafe_allow_html=True)
            st.selectbox(
                "Diversity Metric",
                ["edit_distance", "cosine_similarity", "levenshtein_distance"],
                key="diversity_metric",
                label_visibility="hidden"
            )
            if st.form_submit_button("Apply Evolution Parameters"):
                try:
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
                    else:
                        st.warning(
                            f"Cannot save settings for scope '{scope}'. Provider or model not selected."
                        )
                except Exception as e:
                    st.error(f"Failed to apply evolution parameters: {e}")
        st.markdown("---")
        with st.form("checkpointing_form"):
            st.subheader("Checkpointing")

            st.markdown(create_tooltip_html("Action", "Choose to save the current state as a checkpoint or load a previously saved checkpoint."), unsafe_allow_html=True)
            st.radio(
                "Action", ["Save Checkpoint", "Load Checkpoint"], key="checkpoint_action", label_visibility="hidden"
            )

            checkpoints = []
            with st.spinner("Fetching checkpoints..."):
                try:
                    checkpoints = api.get_checkpoints()
                except Exception as e:
                    st.error(f"Failed to retrieve checkpoints: {e}")
            if checkpoints:
                st.markdown(create_tooltip_html("Load Checkpoint", "Select a checkpoint to load."), unsafe_allow_html=True)
                st.selectbox(
                    "Load Checkpoint", options=checkpoints, key="selected_checkpoint", label_visibility="hidden"
                )
            else:
                st.info("No checkpoints available to load.")

            if st.form_submit_button("Execute Action"):
                if st.session_state.checkpoint_action == "Save Checkpoint":
                    with st.spinner("Saving checkpoint..."):
                        try:
                            api.save_checkpoint(st.session_state.evolution_id)
                            st.success("Checkpoint saved successfully!")
                        except Exception as e:
                            st.error(f"Failed to save checkpoint: {e}")
                elif st.session_state.checkpoint_action == "Load Checkpoint":
                    if st.session_state.selected_checkpoint:
                        st.session_state.load_from_checkpoint = st.session_state.selected_checkpoint
                        st.success(f"Checkpoint '{st.session_state.selected_checkpoint}' selected for loading. Start a new evolution to load it.")
                        st.rerun()
                    else:
                        st.warning("No checkpoint selected to load.")

        st.markdown("---")
        with st.form("system_prompts_form"):
            st.subheader("System Prompts")
            st.markdown(create_tooltip_html("System Prompt", "The initial prompt given to the language model to set its persona or task."), unsafe_allow_html=True)
            st.text_area("System Prompt", key="system_prompt", height=200, label_visibility="hidden")
            st.markdown(create_tooltip_html("Evaluator System Prompt", "The prompt given to the evaluator model to guide its assessment of generated solutions."), unsafe_allow_html=True)
            st.text_area(
                "Evaluator System Prompt", key="evaluator_system_prompt", height=200, label_visibility="hidden"
            )
            if st.form_submit_button("Save System Prompts"):
                try:
                    st.session_state.user_preferences["system_prompt"] = st.session_state.system_prompt
                    st.session_state.user_preferences["evaluator_system_prompt"] = st.session_state.evaluator_system_prompt
                    if save_user_preferences(st.session_state.user_preferences, st.session_state.parameter_settings):
                        st.success("System prompts saved successfully!")
                    else:
                        st.error("Failed to save system prompts.")
                except Exception as e:
                    st.error(f"An error occurred while saving prompts: {e}")

        st.markdown("---")
        with st.form("integrations_form"):
            st.subheader("Integrations")
            st.markdown(create_tooltip_html("Discord Webhook URL", "Enter your Discord webhook URL to receive notifications."), unsafe_allow_html=True)
            st.text_input(
                "Discord Webhook URL",
                key="discord_webhook_url",
                label_visibility="hidden"
            )
            st.markdown(create_tooltip_html("Microsoft Teams Webhook URL", "Enter your Microsoft Teams webhook URL to receive notifications."), unsafe_allow_html=True)
            st.text_input(
                "Microsoft Teams Webhook URL",
                key="msteams_webhook_url",
                label_visibility="hidden"
            )
            st.markdown(create_tooltip_html("Generic Webhook URL", "Enter a generic webhook URL to receive notifications."), unsafe_allow_html=True)
            st.text_input(
                "Generic Webhook URL",
                key="generic_webhook_url",
                label_visibility="hidden"
            )
            if st.form_submit_button("Save Integrations"):
                try:
                    st.session_state.user_preferences["discord_webhook_url"] = st.session_state.discord_webhook_url
                    st.session_state.user_preferences["msteams_webhook_url"] = st.session_state.msteams_webhook_url
                    st.session_state.user_preferences["generic_webhook_url"] = st.session_state.generic_webhook_url
                    if save_user_preferences(st.session_state.user_preferences, st.session_state.parameter_settings):
                        st.success("Integrations saved successfully!")
                    else:
                        st.error("Failed to save integrations.")
                except Exception as e:
                    st.error(f"An error occurred while saving integrations: {e}")

        st.markdown("---")
        st.subheader("Project Settings")
        st.markdown(create_tooltip_html("Project Name", "A unique name for your project."), unsafe_allow_html=True)
        st.text_input("Project Name", key="project_name", label_visibility="hidden")
        st.markdown(create_tooltip_html("Project Description", "A brief description of your project."), unsafe_allow_html=True)
        st.text_area("Project Description", key="project_description", label_visibility="hidden")
        st.markdown(create_tooltip_html("Public Project", "Make your project publicly accessible."), unsafe_allow_html=True)
        st.checkbox(
            "Public Project",
            key="project_public",
            label_visibility="hidden"
        )
        st.markdown(create_tooltip_html("Project Password", "Password-protect your public project."), unsafe_allow_html=True)
        st.text_input(
            "Project Password",
            key="project_password",
            type="password",
            label_visibility="hidden"
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
            st.info("Navigating to Custom Evaluators page (actual page rendering handled elsewhere).")
        if st.sidebar.button("Custom Prompts"):
            st.session_state.page = "prompt_manager"
            st.info("Navigating to Custom Prompts page (actual page rendering handled elsewhere).")
        if st.sidebar.button("Analytics Dashboard"):
            st.session_state.page = "analytics_dashboard"
            st.info("Navigating to Analytics Dashboard page (actual page rendering handled elsewhere).")

        # Enhanced theme toggle with better UX
        current_theme = st.session_state.get("theme", "light")
        theme_emoji = "🌙" if current_theme == "light" else "☀️"
        theme_label = (
            f"{theme_emoji} Switch to {'Dark' if current_theme == 'light' else 'Light'} Mode"
        )

        if st.button(theme_label, key=f"sidebar_theme_toggle_btn_{st.session_state.theme}", use_container_width=True):
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
            label_visibility="hidden"
        )

        # New checkbox for OS theme inheritance
        allow_os_theme_inheritance = st.checkbox(
            "Allow OS Theme Inheritance",
            value=st.session_state.user_preferences.get("allow_os_theme_inheritance", False),
            key="allow_os_theme_inheritance_checkbox"
        )
        # Update user preferences immediately
        st.session_state.user_preferences["allow_os_theme_inheritance"] = allow_os_theme_inheritance

        # Check if theme or inheritance setting has changed
        theme_changed = (selected_theme != current_theme)
        inheritance_setting_changed = (st.session_state.user_preferences.get("allow_os_theme_inheritance") != allow_os_theme_inheritance)

        if theme_changed or inheritance_setting_changed:
            st.session_state.theme = selected_theme
            st.session_state.user_preferences["theme"] = selected_theme  # Update preferences

            st.rerun() # Rerun to apply changes

        # Additional user preferences
        st.markdown("---")
        st.subheader("⚙️ User Preferences")

        # Auto-save preference
        st.markdown(create_tooltip_html("Auto-save preferences", "Automatically save your user preferences."), unsafe_allow_html=True)
        auto_save = st.checkbox(
            "Auto-save preferences",
            value=st.session_state.user_preferences.get("auto_save", True),
            label_visibility="hidden"
        )
        st.session_state.user_preferences["auto_save"] = auto_save

        # Font size preference
        st.markdown(create_tooltip_html("Font Size", "Adjust the font size of the application interface."), unsafe_allow_html=True)
        font_size = st.selectbox(
            "Font Size",
            ["small", "medium", "large"],
            index=["small", "medium", "large"].index(
                st.session_state.user_preferences.get("font_size", "medium")
            ),
            label_visibility="hidden"
        )
        st.session_state.user_preferences["font_size"] = font_size

        # Save preferences button
        if st.button("💾 Save All Preferences"):
            if save_user_preferences(st.session_state.user_preferences, st.session_state.parameter_settings):
                st.success("Preferences saved successfully!")
            else:
                st.error("Failed to save preferences.")

        # OpenEvolve Advanced Features
        st.markdown("---")
        st.subheader("🧬 OpenEvolve Features")
        
        # Quality-Diversity Evolution
        enable_qd_evolution = st.checkbox(
            "Quality-Diversity Evolution", 
            value=st.session_state.get("enable_qd_evolution", False),
            help="Enable Quality-Diversity evolution using MAP-Elites algorithm"
        )
        st.session_state.enable_qd_evolution = enable_qd_evolution
        
        # Multi-Objective Optimization
        enable_multi_objective = st.checkbox(
            "Multi-Objective Optimization", 
            value=st.session_state.get("enable_multi_objective", False),
            help="Optimize for multiple competing objectives simultaneously"
        )
        st.session_state.enable_multi_objective = enable_multi_objective
        
        # Adversarial Evolution
        enable_adversarial_evolution = st.checkbox(
            "Adversarial Evolution", 
            value=st.session_state.get("enable_adversarial_evolution", False),
            help="Use Red Team/Blue Team approach for robustness testing"
        )
        st.session_state.enable_adversarial_evolution = enable_adversarial_evolution

        # Advanced Evolution Modes
        col1, col2 = st.columns(2)
        with col1:
            enable_symbolic_regression = st.checkbox(
                "Symbolic Regression", 
                value=st.session_state.get("enable_symbolic_regression", False),
                help="Discover mathematical expressions that fit data points"
            )
            st.session_state.enable_symbolic_regression = enable_symbolic_regression
            
        with col2:
            enable_neuroevolution = st.checkbox(
                "Neuroevolution", 
                value=st.session_state.get("enable_neuroevolution", False),
                help="Evolve neural network architectures"
            )
            st.session_state.enable_neuroevolution = enable_neuroevolution

        # Evolutionary Architecture Options
        with st.expander("Advanced Evolutionary Options", expanded=False):
            # Evolution tracing
            evolution_trace_enabled = st.checkbox(
                "Enable Evolution Tracing", 
                value=st.session_state.get("evolution_trace_enabled", False),
                help="Log detailed evolution progress for analysis"
            )
            st.session_state.evolution_trace_enabled = evolution_trace_enabled
            
            # Artifact feedback
            use_artifact_feedback = st.checkbox(
                "Use Artifact Feedback", 
                value=st.session_state.get("use_artifact_feedback", True),
                help="Use execution artifacts for improved generation"
            )
            st.session_state.use_artifact_feedback = use_artifact_feedback
            
            # LLM feedback
            use_llm_feedback = st.checkbox(
                "Use LLM Feedback", 
                value=st.session_state.get("use_llm_feedback", False),
                help="Use LLM-based feedback for evolution guidance"
            )
            st.session_state.use_llm_feedback = use_llm_feedback
            
            # Early stopping
            enable_early_stopping = st.checkbox(
                "Enable Early Stopping", 
                value=st.session_state.get("enable_early_stopping", True),
                help="Stop evolution when convergence is detected"
            )
            st.session_state.enable_early_stopping = enable_early_stopping
            
            if enable_early_stopping:
                early_stopping_patience = st.number_input(
                    "Early Stopping Patience", 
                    min_value=1, 
                    max_value=100, 
                    value=st.session_state.get("early_stopping_patience", 10),
                    help="Number of iterations with no improvement before stopping"
                )
                st.session_state.early_stopping_patience = early_stopping_patience
                
            # Advanced research features
            st.markdown("**Advanced Research Features**")
            double_selection = st.checkbox(
                "Double Selection", 
                value=st.session_state.get("double_selection", True),
                help="Use different programs for performance vs inspiration"
            )
            st.session_state.double_selection = double_selection
            
            adaptive_feature_dimensions = st.checkbox(
                "Adaptive Feature Dimensions", 
                value=st.session_state.get("adaptive_feature_dimensions", True),
                help="Dynamically adjust feature dimensions during evolution"
            )
            st.session_state.adaptive_feature_dimensions = adaptive_feature_dimensions
            
            multi_strategy_sampling = st.checkbox(
                "Multi-Strategy Sampling", 
                value=st.session_state.get("multi_strategy_sampling", True),
                help="Use multiple sampling strategies: elite, diverse, exploratory"
            )
            st.session_state.multi_strategy_sampling = multi_strategy_sampling
            
            ring_topology = st.checkbox(
                "Ring Topology", 
                value=st.session_state.get("ring_topology", True),
                help="Use ring topology for island migration"
            )
            st.session_state.ring_topology = ring_topology
            
            coevolutionary_approach = st.checkbox(
                "Coevolutionary Approach", 
                value=st.session_state.get("coevolutionary_approach", False),
                help="Use co-evolution between different populations"
            )
            st.session_state.coevolutionary_approach = coevolutionary_approach
            
            hardware_optimization = st.checkbox(
                "Hardware Optimization", 
                value=st.session_state.get("hardware_optimization", False),
                help="Optimize specifically for target hardware"
            )
            st.session_state.hardware_optimization = hardware_optimization
            
            # Template stochasticity
            use_template_stochasticity = st.checkbox(
                "Template Stochasticity", 
                value=st.session_state.get("use_template_stochasticity", True),
                help="Use random variations of prompts to increase diversity"
            )
            st.session_state.use_template_stochasticity = use_template_stochasticity
            
            # Diff-based evolution
            diff_based_evolution = st.checkbox(
                "Diff-Based Evolution", 
                value=st.session_state.get("diff_based_evolution", True),
                help="Use diff-based evolution for targeted changes"
            )
            st.session_state.diff_based_evolution = diff_based_evolution
            
            # Cascade evaluation
            cascade_evaluation = st.checkbox(
                "Cascade Evaluation", 
                value=st.session_state.get("cascade_evaluation", True),
                help="Use multi-stage filtering with increasing thresholds"
            )
            st.session_state.cascade_evaluation = cascade_evaluation
            
            if cascade_evaluation:
                st.markdown("**Cascade Thresholds**")
                cascade_threshold_1 = st.slider(
                    "Stage 1 Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state.get("cascade_threshold_1", 0.5),
                    step=0.05,
                    help="Threshold for first stage filtering"
                )
                st.session_state.cascade_threshold_1 = cascade_threshold_1
                
                cascade_threshold_2 = st.slider(
                    "Stage 2 Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state.get("cascade_threshold_2", 0.75),
                    step=0.05,
                    help="Threshold for second stage filtering"
                )
                st.session_state.cascade_threshold_2 = cascade_threshold_2
                
                cascade_threshold_3 = st.slider(
                    "Stage 3 Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state.get("cascade_threshold_3", 0.9),
                    step=0.05,
                    help="Threshold for final stage filtering"
                )
                st.session_state.cascade_threshold_3 = cascade_threshold_3

        # Performance Optimization
        with st.expander("Performance Optimization", expanded=False):
            st.markdown("**Resource Management**")
            memory_limit_mb = st.number_input(
                "Memory Limit (MB)", 
                min_value=100, 
                max_value=32768, 
                value=st.session_state.get("memory_limit_mb", 2048),
                help="Memory limit for evaluation processes"
            )
            st.session_state.memory_limit_mb = memory_limit_mb
            
            cpu_limit = st.slider(
                "CPU Limit", 
                min_value=0.1, 
                max_value=32.0, 
                value=st.session_state.get("cpu_limit", 4.0),
                step=0.1,
                help="CPU limit for evaluation processes"
            )
            st.session_state.cpu_limit = cpu_limit
            
            parallel_evaluations = st.number_input(
                "Parallel Evaluations", 
                min_value=1, 
                max_value=32, 
                value=st.session_state.get("parallel_evaluations", 4),
                help="Number of parallel evaluation processes"
            )
            st.session_state.parallel_evaluations = parallel_evaluations
            
            max_code_length = st.number_input(
                "Max Code Length", 
                min_value=100, 
                max_value=100000, 
                value=st.session_state.get("max_code_length", 10000),
                help="Maximum length of code to evolve"
            )
            st.session_state.max_code_length = max_code_length
            
            evaluator_timeout = st.number_input(
                "Evaluator Timeout (s)", 
                min_value=10, 
                max_value=3600, 
                value=st.session_state.get("evaluator_timeout", 300),
                help="Timeout for evaluation processes"
            )
            st.session_state.evaluator_timeout = evaluator_timeout
            
            max_retries_eval = st.number_input(
                "Max Evaluation Retries", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.get("max_retries_eval", 3),
                help="Maximum retries for failed evaluations"
            )
            st.session_state.max_retries_eval = max_retries_eval

        # Add button to access visualization dashboard
        st.markdown("---")
        st.subheader("📊 Visualization & Analytics")
        if st.button("Open OpenEvolve Dashboard", key="open_openevolve_dashboard"):
            st.session_state.page = "openevolve_dashboard"
            st.rerun()

        # Status information
        st.markdown("---")
        st.subheader("ℹ️ Status")
        st.caption(f"Active Project: {st.session_state.project_name}")
        st.caption("Content Type: Auto-detected")
        st.caption(f"Evolution Running: {st.session_state.evolution_running}")
        st.caption(f"Adversarial Running: {st.session_state.adversarial_running}")
        st.caption(f"OpenEvolve Available: {st.session_state.get('openevolve_available', False)}")
        st.caption(f"Advanced Modes: QD={st.session_state.get('enable_qd_evolution', False)}, MO={st.session_state.get('enable_multi_objective', False)}")