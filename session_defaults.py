"""
Session Defaults Manager for OpenEvolve - Default values and initialization
This file manages default values and initialization for session state
File size: ~500 lines (well under the 2000 line limit)
"""

import streamlit as st
from session_utils import DEFAULTS
from providers import PROVIDERS


class SessionDefaults:
    """
    Manages default values and initialization for session state
    """

    def __init__(self):
        self.defaults = DEFAULTS

    def initialize_defaults(self):
        """
        Initialize default values in session state that don't exist
        """
        # Add all defaults to session state if they don't exist
        for key, value in self.defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def get_default(self, key: str):
        """
        Get a default value by key
        """
        return self.defaults.get(key)

    def set_default(self, key: str, value):
        """
        Set a default value
        """
        self.defaults[key] = value

    def reset_all_defaults(self):
        """
        Reset all session state values to their defaults
        """
        for key, value in self.defaults.items():
            st.session_state[key] = value

    def reset_provider_defaults(self):
        """
        Reset provider-specific defaults
        """
        p = st.session_state.provider
        if p in PROVIDERS:  # Assuming PROVIDERS is defined elsewhere
            st.session_state.base_url = PROVIDERS[p].get("base", "")
            st.session_state.model = PROVIDERS[p].get("model") or ""
        st.session_state.api_key = ""
        st.session_state.extra_headers = "{}"

    def get_all_defaults(self):
        """
        Get all default values
        """
        return self.defaults.copy()

    def update_defaults_from_config(self, config_dict: dict):
        """
        Update defaults from a configuration dictionary
        """
        for key, value in config_dict.items():
            if key in self.defaults:
                self.defaults[key] = value
                # Also update session state if it exists
                if key in st.session_state:
                    st.session_state[key] = value

    def validate_defaults(self):
        """
        Validate that all required defaults are present
        """
        required_keys = [
            "provider",
            "api_key",
            "base_url",
            "model",
            "protocol_text",
            "system_prompt",
            "evaluator_system_prompt",
            "max_iterations",
            "evolution_running",
            "adversarial_running",
            "project_name",
        ]

        missing = []
        for key in required_keys:
            if key not in self.defaults:
                missing.append(key)

        return len(missing) == 0, missing

    def get_provider_defaults(self, provider_name: str) -> dict:
        """
        Get default values specific to a provider
        """
        provider_defaults = {}
        if provider_name in PROVIDERS:  # Assuming PROVIDERS is defined elsewhere
            provider_info = PROVIDERS[provider_name]
            provider_defaults = {
                "base_url": provider_info.get("base", self.defaults["base_url"]),
                "model": provider_info.get("model", self.defaults["model"]),
            }
        return provider_defaults

    def set_provider_defaults(self, provider_name: str):
        """
        Set default values for a specific provider
        """
        provider_defaults = self.get_provider_defaults(provider_name)
        for key, value in provider_defaults.items():
            if key in st.session_state:
                st.session_state[key] = value
            if key in self.defaults:
                self.defaults[key] = value


# Initialize session defaults on import
session_defaults = SessionDefaults()
session_defaults.initialize_defaults()
