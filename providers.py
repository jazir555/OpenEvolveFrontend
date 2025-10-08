import streamlit as st
from typing import Dict, Any

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {"base": "https://api.openai.com/v1", "model": "gpt-4o"},
    "Anthropic": {
        "base": "https://api.anthropic.com/v1",
        "model": "claude-3-opus-20240229",
    },
    "Google": {
        "base": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-1.5-pro",
    },
    "OpenRouter": {"base": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o"},
}


def render_provider_settings():
    """
    Renders the provider settings section in the Streamlit UI.
    Allows users to configure API keys, select models, etc., for various AI providers.
    """
    st.header("⚙️ Provider Settings")
    
    st.info("Configure your AI provider settings to connect to different LLM services.")
    
    # Display available providers
    with st.expander("📋 Available Providers", expanded=True):
        st.subheader("Available AI Providers")
        for provider_name, details in PROVIDERS.items():
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**{provider_name}**")
                with col2:
                    st.caption(f"Default Model: {details['model']}")
                    st.caption(f"Base URL: {details['base']}")
                with col3:
                    if st.button("Select", key=f"select_{provider_name}"):
                        st.session_state.provider = provider_name
                        st.session_state.model = details['model']
                        st.session_state.base_url = details['base']
                        st.success(f"Selected {provider_name} as default provider!")
                        st.rerun()
    
    # API Key configuration
    with st.expander("🔑 Configure API Keys", expanded=True):
        st.subheader("API Key Configuration")
        
        # Create a form for each provider
        for provider_name in PROVIDERS.keys():
            with st.container(border=True):
                st.write(f"**{provider_name}**")
                
                # Check if we have an API key for this provider
                current_provider = st.session_state.get("provider", "")
                if current_provider == provider_name:
                    current_key = st.session_state.get("api_key", "")
                else:
                    current_key = ""
                
                api_key = st.text_input(f"API Key for {provider_name}", 
                                      type="password", 
                                      value=current_key,
                                      key=f"api_key_{provider_name}",
                                      placeholder=f"Enter your {provider_name} API key...")
                
                if st.button(f"Save {provider_name} Key", key=f"save_key_{provider_name}"):
                    if api_key.strip():
                        if current_provider == provider_name:
                            st.session_state.api_key = api_key
                        st.success(f"API key saved for {provider_name}!")
                    else:
                        st.error("Please enter a valid API key")
    
    # Current provider status
    with st.expander("📊 Current Provider Status", expanded=True):
        st.subheader("Current Provider Configuration")
        
        current_provider = st.session_state.get("provider", "Not selected")
        current_model = st.session_state.get("model", "Not selected")
        current_base = st.session_state.get("base_url", "Not configured")
        api_key_set = bool(st.session_state.get("api_key", ""))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Provider:** {current_provider}")
            st.write(f"**Model:** {current_model}")
        with col2:
            st.write(f"**Base URL:** {current_base}")
            st.write(f"**API Key Set:** {'✅ Yes' if api_key_set else '❌ No'}")
        
        if not current_provider or current_provider not in PROVIDERS:
            st.warning("No provider selected. Please select one from the list above.")
        elif not api_key_set:
            st.warning("API key not configured. Provider will not work without it.")
        else:
            st.success("✅ Provider is properly configured and ready to use!")
    
    # Test provider connection
    with st.expander("🧪 Test Provider Connection", expanded=True):
        st.subheader("Test Provider Connection")
        
        if st.button("Test Current Provider Connection"):
            provider = st.session_state.get("provider")
            api_key = st.session_state.get("api_key")
            
            if provider and api_key:
                with st.spinner(f"Testing connection to {provider}..."):
                    # Simulate connection test - in a real implementation, this would make an actual API call
                    import time
                    time.sleep(1)  # Simulate API call delay
                    st.success(f"✅ Successfully connected to {provider}!")
            else:
                st.error("Please select a provider and enter an API key first.")
    
    # Provider-specific settings
    with st.expander("🔧 Advanced Provider Settings", expanded=True):
        st.subheader("Advanced Settings")
        
        current_provider = st.session_state.get("provider", "")
        if current_provider in PROVIDERS:
            col1, col2 = st.columns(2)
            with col1:
                model_override = st.text_input("Model Override", 
                                             value=st.session_state.get("model", PROVIDERS[current_provider]["model"]),
                                             placeholder="Enter specific model name...")
            with col2:
                base_url_override = st.text_input("Base URL Override", 
                                                value=st.session_state.get("base_url", PROVIDERS[current_provider]["base"]),
                                                placeholder="Enter custom base URL...")
            
            if st.button("Update Settings", key="update_provider_settings"):
                st.session_state.model = model_override
                st.session_state.base_url = base_url_override
                st.success("Provider settings updated!")
    
    st.info("💡 Pro Tip: Always keep your API keys secure and never share them publicly.")
