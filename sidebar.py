# ------------------------------------------------------------------
# 6. Sidebar – every provider + every knob
# ------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Provider Configuration")
    st.caption("Controls the 'Evolution' tab. Adversarial Testing always uses OpenRouter.")

    # Add a visual separator
    st.markdown("---")

    # Provider selection section
    st.subheader("🌐 Provider Selection")


    @st.cache_data(ttl=300)
    def _query_models(provider_key: str, api_key: str = ""):
        if provider_key not in PROVIDERS:
            return []
        loader = PROVIDERS[provider_key].get("loader")
        if loader:
            with st.spinner(f"Fetching models for {provider_key}..."):
                models = loader(api_key or None)
                if models:
                    return models
        # Fallback to the default model for the provider if loader fails or doesn't exist
        default_model = PROVIDERS[provider_key].get("model")
        return [default_model] if default_model else []


    provider = st.selectbox(
        "Provider", list(PROVIDERS.keys()), key="provider", on_change=reset_defaults,
        help="Choose the backend for the Evolution tab. Adversarial Testing always uses OpenRouter."
    )

    api_key_for_loader = st.session_state.openrouter_key if provider == "OpenRouter" else st.session_state.api_key
    model_options = _query_models(provider, api_key_for_loader)

    model_idx = 0
    if st.session_state.model in model_options:
        model_idx = model_options.index(st.session_state.model)
    elif model_options:
        st.session_state.model = model_options[0]  # Default to first in list if previous selection is invalid

    st.selectbox("Model", model_options, index=model_idx, key="model")
    st.text_input("API Key", type="password", key="api_key",
                  help=f"Leave empty to use env var: {PROVIDERS.get(provider, {}).get('env', 'Not specified')}")
    st.text_input("Base URL", key="base_url", help="Endpoint for chat/completions.")

    # Model parameters section
    st.markdown("---")
    st.subheader("⚙️ Model Parameters")
    st.number_input("Max tokens", min_value=1, max_value=128_000, step=1, key="max_tokens")
    st.slider("Temperature", 0.0, 2.0, key="temperature")
    st.slider("Top-p", 0.0, 1.0, key="top_p")
    st.slider("Frequency penalty", -2.0, 2.0, key="frequency_penalty")
    st.slider("Presence penalty", -2.0, 2.0, key="presence_penalty")
    st.text_input("Seed (optional)", key="seed", help="Integer for deterministic sampling.")
    st.text_area("Extra headers (JSON dict)", height=80, key="extra_headers")

    # Evolution parameters section
    st.markdown("---")
    st.subheader("🔄 Evolution Parameters")
    st.number_input("Max iterations", 1, 1000, key="max_iterations")
    st.number_input("Checkpoint interval", 1, 100, key="checkpoint_interval")
    # Disabled params from original code, kept for UI consistency
    st.number_input("Population size", 1, 100, key="population_size", disabled=True)
    st.number_input("Num islands", 1, 10, key="num_islands", disabled=True)
    st.number_input("Elite ratio", 0.0, 1.0, key="elite_ratio", disabled=True)
    st.number_input("Exploration ratio", 0.0, 1.0, key="exploration_ratio", disabled=True)
    st.number_input("Exploitation ratio", 0.0, 1.0, key="exploitation_ratio", disabled=True)
    st.number_input("Archive size", 0, 1000, key="archive_size", disabled=True)

    # Prompts section
    st.markdown("---")
    st.subheader("💬 Prompts")
    st.text_area("System prompt", height=120, key="system_prompt")
    st.text_area("Evaluator system prompt", height=120, key="evaluator_system_prompt", disabled=True)

    # Configuration profiles section
    st.markdown("---")
    st.subheader("📋 Configuration Profiles")

    # Load profile
    profiles = list_config_profiles()
    if profiles:
        selected_profile = st.selectbox("Load Profile", [""] + profiles, key="load_profile_select")
        if selected_profile and st.button("Load Selected Profile", key="load_profile_btn"):
            if load_config_profile(selected_profile):
                st.success(f"Loaded profile: {selected_profile}")
                st.rerun()

    # Save profile
    profile_name = st.text_input("Save Current Config As", key="save_profile_name")
    if profile_name and st.button("Save Profile", key="save_profile_btn"):
        if save_config_profile(profile_name):
            st.success(f"Saved profile: {profile_name}")
            st.rerun()

    # Tutorial section
    st.markdown("---")
    st.subheader("🎓 Tutorial")
    if not st.session_state.tutorial_completed:
        if st.button("Start Tutorial"):
            st.session_state.current_tutorial_step = 1
            st.rerun()

        if st.session_state.current_tutorial_step > 0:
            tutorial_steps = [
                "Welcome to OpenEvolve! This tutorial will guide you through the main features.",
                "Step 1: Enter your protocol in the text area in the Evolution tab.",
                "Step 2: Configure your LLM provider and parameters in the sidebar.",
                "Step 3: Click 'Start Evolution' to begin improving your protocol.",
                "Step 4: Try the Adversarial Testing tab for advanced security hardening.",
                "Step 5: Use version control to track changes and collaborate with others.",
                "Tutorial completed! You're ready to use OpenEvolve."
            ]

            if st.session_state.current_tutorial_step <= len(tutorial_steps):
                st.info(tutorial_steps[st.session_state.current_tutorial_step - 1])
                col1, col2 = st.columns(2)
                if st.session_state.current_tutorial_step > 1:
                    if col1.button("Previous"):
                        st.session_state.current_tutorial_step -= 1
                        st.rerun()
                if st.session_state.current_tutorial_step < len(tutorial_steps):
                    if col2.button("Next"):
                        st.session_state.current_tutorial_step += 1
                        st.rerun()
                else:
                    if col2.button("Finish Tutorial"):
                        st.session_state.tutorial_completed = True
                        st.session_state.current_tutorial_step = 0
                        st.rerun()
    else:
        if st.button("Restart Tutorial"):
            st.session_state.tutorial_completed = False
            st.session_state.current_tutorial_step = 1
            st.rerun()

    # Reset button
    st.markdown("---")
    st.button("🔄 Reset to provider defaults", on_click=reset_defaults, use_container_width=True)
