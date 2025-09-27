import streamlit as st
import threading
from typing import Any, Callable, TypeVar, Dict, List, Optional, Tuple
from state import State

R = TypeVar("R")

class SessionManager:
    _singleton_lock = threading.Lock()
    _singleton_instance = None

    def __new__(cls, **kwargs):
        with cls._singleton_lock:
            if cls._singleton_instance is None:
                cls._singleton_instance = super().__new__(cls)
                cls._singleton_instance._state = State(**kwargs)
            return cls._singleton_instance

    def get_state(self) -> State:
        return self._state

    def function_wrapper(self, func: Callable[..., R]) -> Callable[..., R]:
        def wrapped_func(*args, **kwargs) -> R:
            # In Streamlit, session_state is automatically managed across reruns.
            # Explicit sync might not be strictly necessary for basic usage but can be useful
            # for complex scenarios or when integrating with external state management.
            result = func(*args, **kwargs)
            return result
        return wrapped_func

    def rerun(self):
        st.rerun()

    def save_user_preferences(self) -> bool:
        try:
            # Assuming user preferences are stored in st.session_state.user_preferences
            # In a real application, this would involve writing to a database or file
            st.session_state["user_preferences"] = st.session_state.get("user_preferences", {})
            return True
        except Exception as e:
            st.error(f"Error saving preferences: {e}")
            return False

    def toggle_theme(self):
        current_theme = st.session_state.get("theme", "light")
        new_theme = "dark" if current_theme == "light" else "light"
        st.session_state["theme"] = new_theme
        st.session_state.user_preferences["theme"] = new_theme
