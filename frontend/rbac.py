
import streamlit as st
from typing import List, Dict, Any

ROLES = {
    "admin": {
        "permissions": ["manage_users", "manage_roles", "manage_projects"]
    },
    "editor": {
        "permissions": ["edit_content", "add_comments"]
    },
    "viewer": {
        "permissions": ["view_content"]
    }
}

if "user_roles" not in st.session_state:
    st.session_state.user_roles = {
        "admin": "admin"
    }

def get_user_role(username: str) -> str:
    """
    Get the role of a user.
    """
    return st.session_state.user_roles.get(username, "viewer")

def has_permission(username: str, permission: str) -> bool:
    """
    Check if a user has a specific permission.
    """
    role = get_user_role(username)
    return permission in ROLES.get(role, {}).get("permissions", [])

def assign_role(username: str, role: str):
    """
    Assign a role to a user.
    """
    if role in ROLES:
        st.session_state.user_roles[username] = role
