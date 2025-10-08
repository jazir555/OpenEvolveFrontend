"""
Main Session State Manager for OpenEvolve
This file serves as the main entry point for session state management
and combines all the modular functionality.
File size: ~200 lines (well under the 2000 line limit)
"""

from session_defaults import session_defaults
from content_manager import content_manager
from collaboration_manager import collaboration_manager
from version_control import version_control
from analytics_manager import analytics_manager
from export_import_manager import export_import_manager
from template_manager import template_manager
from validation_manager import validation_manager
from session_utils import (
    reset_defaults,
    save_user_preferences,
    toggle_theme,
    calculate_protocol_complexity,
    extract_protocol_structure,
    generate_protocol_recommendations,
    _clamp,
    _rand_jitter_ms,
    _approx_tokens,
    _cost_estimate,
    safe_int,
    safe_float,
    _safe_list,
    _extract_json_block,
    _compose_messages,
    _hash_text,
    APPROVAL_PROMPT,
    RED_TEAM_CRITIQUE_PROMPT,
    BLUE_TEAM_PATCH_PROMPT,
    CODE_REVIEW_RED_TEAM_PROMPT,
    CODE_REVIEW_BLUE_TEAM_PROMPT,
    PLAN_REVIEW_RED_TEAM_PROMPT,
    PLAN_REVIEW_BLUE_TEAM_PROMPT,
)


# Initialize all managers
def initialize_session_state():
    """
    Initialize all session state managers
    """
    session_defaults.initialize_defaults()


# Export all managers and utilities for import
__all__ = [
    # Managers
    "session_defaults",
    "content_manager",
    "collaboration_manager",
    "version_control",
    "analytics_manager",
    "export_import_manager",
    "template_manager",
    "validation_manager",
    # Utility functions
    "reset_defaults",
    "save_user_preferences",
    "toggle_theme",
    "calculate_protocol_complexity",
    "extract_protocol_structure",
    "generate_protocol_recommendations",
    "_clamp",
    "_rand_jitter_ms",
    "_approx_tokens",
    "_cost_estimate",
    "safe_int",
    "safe_float",
    "_safe_list",
    "_extract_json_block",
    "_compose_messages",
    "_hash_text",
    # Prompts
    "APPROVAL_PROMPT",
    "RED_TEAM_CRITIQUE_PROMPT",
    "BLUE_TEAM_PATCH_PROMPT",
    "CODE_REVIEW_RED_TEAM_PROMPT",
    "CODE_REVIEW_BLUE_TEAM_PROMPT",
    "PLAN_REVIEW_RED_TEAM_PROMPT",
    "PLAN_REVIEW_BLUE_TEAM_PROMPT",
]

# Initialize session state on import
initialize_session_state()
