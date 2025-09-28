import streamlit as st

ROLES = {
    "admin": {"permissions": ["manage_users", "manage_roles", "manage_projects"]},
    "editor": {"permissions": ["edit_content", "add_comments"]},
    "viewer": {"permissions": ["view_content"]},
}

if "user_roles" not in st.session_state:
    st.session_state.user_roles = {"admin": "admin"}


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

def render_rbac_settings():
    """
    Placeholder function to render the RBAC settings section in the Streamlit UI.
    This would typically allow administrators to view and manage user roles and permissions.
    """
    st.header("🔒 Role-Based Access Control (RBAC) Settings")
    st.info("RBAC settings management features are under development. Stay tuned!")
    st.subheader("Current User Roles (for demonstration)")
    for user, role in st.session_state.user_roles.items():
        st.write(f"- {user}: {role}")
