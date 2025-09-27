import streamlit as st
import threading
from typing import Any, Callable, TypeVar, Generic

R = TypeVar("R")

class State:
    def __init__(self, **kwargs):
        self.__dict__["_state"] = kwargs
        self.__dict__["_initial_state"] = kwargs.copy()

    def __getitem__(self, key):
        return self.__dict__["_state"][key]

    def __setitem__(self, key, value):
        self.__dict__["_state"][key] = value

    def __getattr__(self, key):
        if key in self.__dict__["_state"]:
            return self.__dict__["_state"][key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self.__dict__["_state"][key] = value

    def __delattr__(self, key):
        if key in self.__dict__["_state"]:
            del self.__dict__["_state"][key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def clear(self):
        self.__dict__["_state"] = self.__dict__["_initial_state"].copy()

    def sync(self):
        for key, value in self.__dict__["_state"].items():
            if key not in st.session_state:
                st.session_state[key] = value

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
            self.sync_state_with_streamlit()
            result = func(*args, **kwargs)
            self.sync_state_with_streamlit()
            return result
        return wrapped_func

    def rerun(self):
        st.rerun()

    def sync_state_with_streamlit(self):
        for key, value in self._state.__dict__["_state"].items():
            if key not in st.session_state:
                st.session_state[key] = value
        for key in list(st.session_state.keys()):
            if key not in self._state.__dict__["_state"]:
                del st.session_state[key]

def get(**kwargs) -> State:
    session_manager = SessionManager(**kwargs)
    return session_manager.get_state()
