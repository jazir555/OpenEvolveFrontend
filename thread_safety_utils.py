"""
Thread Safety Utilities for OpenEvolve Frontend

This module provides thread-safe utilities for accessing shared mutable state,
particularly Streamlit session state which is not thread-safe by default.

Thread Safety Guidelines:
=========================
1. ALWAYS use thread_lock when accessing st.session_state in multi-threaded contexts
2. Use thread_local for thread-specific data that should not be shared
3. Use locks for global caches and shared mutable state
4. Never share mutable state between threads without proper synchronization

Usage Example:
=============
    from thread_safety_utils import get_session_state_safely, set_session_state_safely

    # Thread-safe read
    value = get_session_state_safely('my_key', default=None)

    # Thread-safe write
    set_session_state_safely('my_key', new_value)

    # Thread-safe lock context
    with get_session_lock():
        st.session_state.my_key = complex_operation()
"""

import threading
from ui_shim import ui as st
from typing import Any, Optional, Dict, List, Callable
from contextlib import contextmanager
import functools

# =============================================================================
# GLOBAL LOCKS FOR SHARED STATE
# =============================================================================

# Global lock for session state access
_session_lock: threading.RLock = threading.RLock()

# Lock for evolution-related state
_evolution_lock: threading.RLock = threading.RLock()

# Lock for adversarial-related state
_adversarial_lock: threading.RLock = threading.RLock()

# Lock for workflow-related state
_workflow_lock: threading.RLock = threading.RLock()

# Thread-local storage for thread-specific data
_thread_local = threading.local()


# =============================================================================
# SESSION STATE THREAD SAFETY UTILITIES
# =============================================================================

def get_session_lock() -> threading.RLock:
    """
    Get the global session state lock.

    This lock should be used when accessing st.session_state from multiple threads
    to prevent race conditions and data corruption.

    Returns:
        threading.RLock: Reentrant lock for session state access

    Example:
        with get_session_lock():
            value = st.session_state.some_key
    """
    return _session_lock


@contextmanager
def session_state_lock():
    """
    Context manager for thread-safe session state access.

    Usage:
        with session_state_lock():
            st.session_state.my_key = value

    Yields:
        None: Context manager for session lock
    """
    with _session_lock:
        yield


def get_session_state_safely(key: str, default: Any = None) -> Any:
    """
    Thread-safe access to session state.

    Args:
        key: Session state key to retrieve
        default: Default value if key doesn't exist

    Returns:
        The value from session state or default

    Thread Safety:
        Uses lock to prevent race conditions during concurrent access

    Example:
        value = get_session_state_safely('evolution_history', default=[])
    """
    with _session_lock:
        return st.session_state.get(key, default)


def set_session_state_safely(key: str, value: Any) -> None:
    """
    Thread-safe write to session state.

    Args:
        key: Session state key to set
        value: Value to set

    Thread Safety:
        Uses lock to prevent race conditions during concurrent writes

    Example:
        set_session_state_safely('evolution_current_best', content)
    """
    with _session_lock:
        st.session_state[key] = value


def update_session_state_safely(key: str, update_func: Callable[[Any], Any]) -> None:
    """
    Thread-safe update of session state using a function.

    This is useful for compound operations that need to be atomic.

    Args:
        key: Session state key to update
        update_func: Function that takes current value and returns new value

    Thread Safety:
        Uses lock to ensure the entire update operation is atomic

    Example:
        # Append to list atomically
        update_session_state_safely('evolution_history',
                                   lambda hist: hist + [new_entry])
    """
    with _session_lock:
        current_value = st.session_state.get(key)
        st.session_state[key] = update_func(current_value)


def delete_session_state_safely(key: str) -> None:
    """
    Thread-safe deletion from session state.

    Args:
        key: Session state key to delete

    Thread Safety:
        Uses lock to prevent race conditions during deletion
    """
    with _session_lock:
        if key in st.session_state:
            del st.session_state[key]


# =============================================================================
# EVOLUTION STATE THREAD SAFETY
# =============================================================================

@contextmanager
def evolution_state_lock():
    """
    Context manager for thread-safe evolution state access.

    Usage:
        with evolution_state_lock():
            st.session_state.evolution_history.append(entry)

    Yields:
        None: Context manager for evolution lock
    """
    with _evolution_lock:
        yield


def get_evolution_state_safely(key: str, default: Any = None) -> Any:
    """
    Thread-safe access to evolution-related session state.

    Args:
        key: Session state key to retrieve
        default: Default value if key doesn't exist

    Returns:
        The value from session state or default

    Thread Safety:
        Uses evolution-specific lock to prevent race conditions

    Example:
        history = get_evolution_state_safely('evolution_history', default=[])
    """
    with _evolution_lock, _session_lock:
        return st.session_state.get(key, default)


def set_evolution_state_safely(key: str, value: Any) -> None:
    """
    Thread-safe write to evolution-related session state.

    Args:
        key: Session state key to set
        value: Value to set

    Thread Safety:
        Uses evolution-specific lock to prevent race conditions

    Example:
        set_evolution_state_safely('evolution_current_best', content)
    """
    with _evolution_lock, _session_lock:
        st.session_state[key] = value


# =============================================================================
# ADVERSARIAL STATE THREAD SAFETY
# =============================================================================

@contextmanager
def adversarial_state_lock():
    """
    Context manager for thread-safe adversarial state access.

    Usage:
        with adversarial_state_lock():
            st.session_state.adversarial_log.append(entry)

    Yields:
        None: Context manager for adversarial lock
    """
    with _adversarial_lock:
        yield


def get_adversarial_state_safely(key: str, default: Any = None) -> Any:
    """
    Thread-safe access to adversarial-related session state.

    Args:
        key: Session state key to retrieve
        default: Default value if key doesn't exist

    Returns:
        The value from session state or default

    Thread Safety:
        Uses adversarial-specific lock to prevent race conditions

    Example:
        log = get_adversarial_state_safely('adversarial_log', default=[])
    """
    with _adversarial_lock, _session_lock:
        return st.session_state.get(key, default)


def set_adversarial_state_safely(key: str, value: Any) -> None:
    """
    Thread-safe write to adversarial-related session state.

    Args:
        key: Session state key to set
        value: Value to set

    Thread Safety:
        Uses adversarial-specific lock to prevent race conditions

    Example:
        set_adversarial_state_safely('adversarial_results', results)
    """
    with _adversarial_lock, _session_lock:
        st.session_state[key] = value


# =============================================================================
# WORKFLOW STATE THREAD SAFETY
# =============================================================================

@contextmanager
def workflow_state_lock():
    """
    Context manager for thread-safe workflow state access.

    Usage:
        with workflow_state_lock():
            st.session_state.workflow_progress = value

    Yields:
        None: Context manager for workflow lock
    """
    with _workflow_lock:
        yield


def get_workflow_state_safely(key: str, default: Any = None) -> Any:
    """
    Thread-safe access to workflow-related session state.

    Args:
        key: Session state key to retrieve
        default: Default value if key doesn't exist

    Returns:
        The value from session state or default

    Thread Safety:
        Uses workflow-specific lock to prevent race conditions
    """
    with _workflow_lock, _session_lock:
        return st.session_state.get(key, default)


def set_workflow_state_safely(key: str, value: Any) -> None:
    """
    Thread-safe write to workflow-related session state.

    Args:
        key: Session state key to set
        value: Value to set

    Thread Safety:
        Uses workflow-specific lock to prevent race conditions
    """
    with _workflow_lock, _session_lock:
        st.session_state[key] = value


# =============================================================================
# THREAD-LOCAL STORAGE UTILITIES
# =============================================================================

def get_thread_local(key: str, default: Any = None) -> Any:
    """
    Get thread-local data.

    Thread-local storage is safe for concurrent access because each thread
    has its own independent copy.

    Args:
        key: Thread-local key to retrieve
        default: Default value if key doesn't exist

    Returns:
        The value from thread-local storage or default

    Example:
        # Store thread-specific LLM client
        client = get_thread_local('llm_client')
        if not client:
            client = create_client()
            set_thread_local('llm_client', client)
    """
    return getattr(_thread_local, key, default)


def set_thread_local(key: str, value: Any) -> None:
    """
    Set thread-local data.

    Args:
        key: Thread-local key to set
        value: Value to set

    Example:
        set_thread_local('request_id', uuid.uuid4())
    """
    setattr(_thread_local, key, value)


def clear_thread_local() -> None:
    """
    Clear all thread-local data for the current thread.

    Useful for cleanup after operations complete.
    """
    keys = [k for k in dir(_thread_local) if not k.startswith('__')]
    for key in keys:
        delattr(_thread_local, key)


# =============================================================================
# DECORATORS FOR THREAD SAFETY
# =============================================================================

def with_session_lock(func: Callable) -> Callable:
    """
    Decorator to make a function thread-safe with respect to session state.

    Usage:
        @with_session_lock
        def update_evolution_history(entry):
            st.session_state.evolution_history.append(entry)

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that acquires session lock before execution
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _session_lock:
            return func(*args, **kwargs)
    return wrapper


def with_evolution_lock(func: Callable) -> Callable:
    """
    Decorator to make a function thread-safe for evolution state.

    Usage:
        @with_evolution_lock
        def update_evolution_metrics(metrics):
            st.session_state.evolution_metrics.update(metrics)

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that acquires evolution lock before execution
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _evolution_lock, _session_lock:
            return func(*args, **kwargs)
    return wrapper


def with_adversarial_lock(func: Callable) -> Callable:
    """
    Decorator to make a function thread-safe for adversarial state.

    Usage:
        @with_adversarial_lock
        def update_adversarial_log(entry):
            st.session_state.adversarial_log.append(entry)

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that acquires adversarial lock before execution
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _adversarial_lock, _session_lock:
            return func(*args, **kwargs)
    return wrapper


def with_workflow_lock(func: Callable) -> Callable:
    """
    Decorator to make a function thread-safe for workflow state.

    Usage:
        @with_workflow_lock
        def update_workflow_progress(progress):
            st.session_state.workflow_progress = progress

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that acquires workflow lock before execution
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _workflow_lock, _session_lock:
            return func(*args, **kwargs)
    return wrapper


# =============================================================================
# LIST/DICT THREAD-SAFE OPERATIONS
# =============================================================================

def append_to_session_list_safely(key: str, item: Any) -> None:
    """
    Thread-safe append to a list in session state.

    Creates the list if it doesn't exist.

    Args:
        key: Session state key for the list
        item: Item to append

    Thread Safety:
        Uses lock to ensure atomic append operation

    Example:
        append_to_session_list_safely('evolution_history', entry)
    """
    with _session_lock:
        if key not in st.session_state:
            st.session_state[key] = []
        st.session_state[key].append(item)


def update_session_dict_safely(key: str, updates: Dict[str, Any]) -> None:
    """
    Thread-safe update to a dict in session state.

    Creates the dict if it doesn't exist.

    Args:
        key: Session state key for the dict
        updates: Dictionary of updates to apply

    Thread Safety:
        Uses lock to ensure atomic update operation

    Example:
        update_session_dict_safely('evolution_metrics', {'fitness': 0.95})
    """
    with _session_lock:
        if key not in st.session_state:
            st.session_state[key] = {}
        st.session_state[key].update(updates)


def increment_session_counter_safely(key: str, delta: int = 1) -> int:
    """
    Thread-safe increment of a counter in session state.

    Args:
        key: Session state key for the counter
        delta: Amount to increment by (default: 1)

    Returns:
        The new counter value after incrementing

    Thread Safety:
        Uses lock to ensure atomic increment operation

    Example:
        count = increment_session_counter_safely('evolution_iteration')
    """
    with _session_lock:
        current = st.session_state.get(key, 0)
        new_value = current + delta
        st.session_state[key] = new_value
        return new_value


# =============================================================================
# INITIALIZATION HELPERS
# =============================================================================

def init_session_state_safely(defaults: Dict[str, Any]) -> None:
    """
    Thread-safe initialization of session state with defaults.

    Only sets keys that don't already exist.

    Args:
        defaults: Dictionary of default values for session state

    Thread Safety:
        Uses lock to ensure atomic initialization

    Example:
        init_session_state_safely({
            'evolution_history': [],
            'evolution_current_best': '',
            'evolution_iteration': 0
        })
    """
    with _session_lock:
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
