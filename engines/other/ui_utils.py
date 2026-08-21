"""
Shared utility functions for UI components.
Provides common patterns for error handling, validation, caching, and rendering.
"""
from __future__ import annotations


from ui_shim import ui as st
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar
from functools import wraps
import traceback
import time

# Set up logging
logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# Error Handling Utilities
# ============================================================================

class DataFetchError(Exception):
    """Error fetching data from backend"""
    def __init__(self, message: str):
        super().__init__(message)


class VisualizationError(Exception):
    """Error rendering visualization"""
    def __init__(self, message: str):
        super().__init__(message)


class ValidationError(Exception):
    """Error validating data"""
    def __init__(self, message: str):
        super().__init__(message)


def with_error_handling(func: Callable) -> Callable:
    """
    Decorator for consistent error handling in UI components.
    
    Catches common exceptions and displays user-friendly error messages.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DataFetchError as e:
            st.error(f"Failed to load data: {e}")
            logger.error(f"Data fetch error in {func.__name__}: {e}", exc_info=True)
        except VisualizationError as e:
            st.warning(f"Visualization error: {e}")
            logger.error(f"Visualization error in {func.__name__}: {e}", exc_info=True)
        except ValidationError as e:
            st.error(f"Validation error: {e}")
            logger.error(f"Validation error in {func.__name__}: {e}", exc_info=True)
        except Exception as e:
            st.error("An unexpected error occurred. Please try again.")
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            logger.error(traceback.format_exc())
    return wrapper


def safe_execute(func: Callable[[], T], fallback: T, error_message: str = None) -> T:
    """
    Safely execute a function and return fallback on error.
    
    Args:
        func: Function to execute
        fallback: Value to return on error
        error_message: Optional custom error message
        
    Returns:
        Result of func or fallback on error
    """
    try:
        return func()
    except Exception as e:
        if error_message:
            st.warning(error_message)
        logger.error(f"Error in safe_execute: {e}", exc_info=True)
        return fallback


# ============================================================================
# Data Validation Utilities
# ============================================================================

def validate_and_default(
    data: Any,
    validator: Callable[[Any], bool],
    default_factory: Callable[[], Any],
    warning_message: str = "Data validation failed. Using defaults."
) -> Any:
    """
    Validate data and return default if invalid.
    
    Args:
        data: Data to validate
        validator: Function that returns True if data is valid
        default_factory: Function that creates default value
        warning_message: Message to display on validation failure
        
    Returns:
        Validated data or default value
    """
    try:
        if not validator(data):
            st.warning(warning_message)
            return default_factory()
        return data
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        st.warning(warning_message)
        return default_factory()


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate that all required fields are present in data.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        
    Returns:
        True if all required fields present, False otherwise
    """
    missing = [field for field in required_fields if field not in data or data[field] is None]
    if missing:
        st.error(f"Missing required fields: {', '.join(missing)}")
        return False
    return True


# ============================================================================
# Chart Rendering Utilities
# ============================================================================

def render_chart_with_fallback(
    chart_func: Callable[[Any], None],
    data: Any,
    fallback_func: Callable[[Any], None],
    error_message: str = "Chart rendering failed. Showing table view."
) -> None:
    """
    Render chart with table fallback on error.
    
    Args:
        chart_func: Function to render chart
        data: Data to visualize
        fallback_func: Function to render fallback view
        error_message: Message to display on error
    """
    try:
        chart_func(data)
    except Exception as e:
        logger.error(f"Chart rendering error: {e}", exc_info=True)
        st.warning(error_message)
        try:
            fallback_func(data)
        except Exception as e2:
            logger.error(f"Fallback rendering error: {e2}", exc_info=True)
            st.error("Unable to display data.")


def render_table_fallback(data: Any) -> None:
    """
    Render data as a table (fallback for failed charts).
    
    Args:
        data: Data to display
    """
    if isinstance(data, (list, dict)):
        st.json(data)
    else:
        st.write(data)


# ============================================================================
# OpenEvolve Integration Utilities
# ============================================================================

def display_openevolve_metrics(metrics: Dict[str, Any]) -> None:
    """
    Display standard OpenEvolve execution metrics.
    
    Args:
        metrics: Dictionary containing OpenEvolve metrics
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Calls", metrics.get("api_calls", 0))
    
    with col2:
        st.metric("Tokens Used", f"{metrics.get('tokens', 0):,}")
    
    with col3:
        cost = metrics.get("cost", 0)
        st.metric("Cost", f"${cost:.4f}")
    
    with col4:
        iterations = metrics.get("evolution_iterations", 0)
        st.metric("Iterations", iterations)


def format_openevolve_config(config: Dict[str, Any]) -> str:
    """
    Format OpenEvolve configuration for display.
    
    Args:
        config: OpenEvolve configuration dictionary
        
    Returns:
        Formatted string representation
    """
    lines = []
    lines.append(f"Model: {config.get('model', 'N/A')}")
    lines.append(f"Evolution Mode: {config.get('evolution_mode', 'N/A')}")
    lines.append(f"Temperature: {config.get('temperature', 'N/A')}")
    lines.append(f"Max Iterations: {config.get('max_iterations', 'N/A')}")
    return "\n".join(lines)


# ============================================================================
# Session State Management Utilities
# ============================================================================

def get_or_init_state(key: str, default_factory: Callable[[], Any]) -> Any:
    """
    Get value from session state or initialize with default.
    
    Args:
        key: Session state key
        default_factory: Function that creates default value
        
    Returns:
        Value from session state
    """
    if key not in st.session_state:
        st.session_state[key] = default_factory()
    return st.session_state[key]


def update_state(key: str, value: Any) -> None:
    """
    Update session state value.
    
    Args:
        key: Session state key
        value: New value
    """
    st.session_state[key] = value


def clear_state(key: str) -> None:
    """
    Clear session state value.
    
    Args:
        key: Session state key
    """
    if key in st.session_state:
        del st.session_state[key]


# ============================================================================
# UI Component Utilities
# ============================================================================

def render_metric_card(label: str, value: Any, delta: Optional[Any] = None) -> None:
    """
    Render a metric card.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
    """
    st.metric(label, value, delta)


def render_status_badge(status: str, color_map: Optional[Dict[str, str]] = None) -> None:
    """
    Render a status badge with color.
    
    Args:
        status: Status text
        color_map: Optional mapping of status to color
    """
    if color_map is None:
        color_map = {
            "completed": "green",
            "in_progress": "orange",
            "pending": "gray",
            "failed": "red"
        }
    
    color = color_map.get(status.lower(), "gray")
    st.markdown(f":{color}[**{status}**]")


def render_progress_bar(current: int, total: int, label: str = "") -> None:
    """
    Render a progress bar.
    
    Args:
        current: Current progress value
        total: Total value
        label: Optional label
    """
    if total > 0:
        progress = current / total
        st.progress(progress, text=f"{label} {current}/{total} ({progress*100:.1f}%)")
    else:
        st.progress(0, text=f"{label} 0/0")


def render_confirmation_dialog(
    message: str,
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel"
) -> Optional[bool]:
    """
    Render a confirmation dialog.
    
    Args:
        message: Confirmation message
        confirm_label: Label for confirm button
        cancel_label: Label for cancel button
        
    Returns:
        True if confirmed, False if cancelled, None if not yet decided
    """
    st.warning(message)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(confirm_label, type="primary"):
            return True
    
    with col2:
        if st.button(cancel_label):
            return False
    
    return None


# ============================================================================
# Data Formatting Utilities
# ============================================================================

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 30m 15s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m"


def format_number(num: float, precision: int = 2) -> str:
    """
    Format number with thousands separator.
    
    Args:
        num: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    if num >= 1000000:
        return f"{num/1000000:.{precision}f}M"
    elif num >= 1000:
        return f"{num/1000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value between 0 and 1
        precision: Decimal precision
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


# ============================================================================
# Loading and Caching Utilities
# ============================================================================

def show_loading(message: str = "Loading...") -> None:
    """
    Show loading spinner with message.
    
    Args:
        message: Loading message
    """
    with st.spinner(message):
        time.sleep(0.1)


def cache_data(ttl: int = 300):
    """
    Decorator for caching data with TTL.
    
    Args:
        ttl: Time to live in seconds
        
    Returns:
        Decorated function
    """
    return st.cache_data(ttl=ttl)


# ============================================================================
# Feature Flag Utilities
# ============================================================================

FEATURE_FLAGS = {
    "analytics_dashboard": True,
    "knowledge_base": True,
    "dependency_viz": True,
    "auto_approval_ui": True,
    "batch_operations": True,
    "enhanced_monitoring": True,
    "workflow_templates": True,
}


def is_feature_enabled(feature_name: str) -> bool:
    """
    Check if a feature is enabled.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        True if enabled, False otherwise
    """
    return FEATURE_FLAGS.get(feature_name, False)


def require_feature(feature_name: str):
    """
    Decorator to require a feature to be enabled.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_feature_enabled(feature_name):
                st.warning(f"Feature '{feature_name}' is not enabled.")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator
