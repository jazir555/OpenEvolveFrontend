import streamlit as st
from typing import Any, Dict


class State:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def __getitem__(self, key):
        return st.session_state[key]

    def __setitem__(self, key, value):
        st.session_state[key] = value

    def __getattr__(self, key):
        if key in st.session_state:
            return st.session_state[key]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'"
        )

    def __setattr__(self, key, value):
        st.session_state[key] = value

    def __delattr__(self, key):
        if key in st.session_state:
            del st.session_state[key]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def clear(self):
        for key in list(st.session_state.keys()):
            del st.session_state[key]

    def to_dict(self):
        return dict(st.session_state)

    def reset_defaults(self, defaults: Dict[str, Any]):
        for key, value in defaults.items():
            st.session_state[key] = value

    def sync(self):
        # This method is mostly for conceptual completeness, as Streamlit's session_state is inherently synced.
        pass
