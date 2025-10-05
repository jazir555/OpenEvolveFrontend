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
    Renders the RBAC settings section in the Streamlit UI.
    Allows administrators to view and manage user roles and permissions.
    """
    st.header("🔒 Role-Based Access Control (RBAC) Settings")
    
    st.info("Manage user roles and permissions to control access to features and data.")
    
    # User management section
    with st.expander("👥 Manage Users & Roles", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Add New User")
            with st.form("add_user_form"):
                new_username = st.text_input("Username", placeholder="Enter new username")
                new_role = st.selectbox("Role", list(ROLES.keys()))
                submitted = st.form_submit_button("Add User")
                if submitted and new_username:
                    assign_role(new_username, new_role)
                    st.success(f"User '{new_username}' added with role '{new_role}'")
                    st.rerun()
        
        with col2:
            st.subheader("Current Users")
            if st.session_state.user_roles:
                for username, role in st.session_state.user_roles.items():
                    with st.container(border=True):
                        col_a, col_b, col_c = st.columns([3, 2, 1])
                        with col_a:
                            st.write(f"**{username}**")
                        with col_b:
                            st.caption(f"Role: {role}")
                        with col_c:
                            if st.button("Delete", key=f"delete_user_{username}"):
                                del st.session_state.user_roles[username]
                                st.rerun()
            else:
                st.info("No users added yet.")
    
    # Role permissions overview
    with st.expander("📋 Role Permissions", expanded=True):
        st.subheader("Role Permissions Overview")
        for role_name, role_info in ROLES.items():
            with st.container(border=True):
                st.write(f"**{role_name.title()}**")
                permissions = role_info.get("permissions", [])
                if permissions:
                    for perm in permissions:
                        st.caption(f"• {perm}")
                else:
                    st.caption("No permissions assigned")
    
    # Role assignment section
    if st.session_state.user_roles:
        with st.expander("🔄 Change User Roles", expanded=True):
            st.subheader("Change User Roles")
            users_list = list(st.session_state.user_roles.keys())
            if users_list:
                selected_user = st.selectbox("Select User", users_list)
                if selected_user:
                    current_role = get_user_role(selected_user)
                    new_role = st.selectbox("New Role", list(ROLES.keys()), 
                                          index=list(ROLES.keys()).index(current_role) 
                                          if current_role in ROLES.keys() else 0)
                    
                    if st.button(f"Update Role for {selected_user}"):
                        assign_role(selected_user, new_role)
                        st.success(f"Role for '{selected_user}' updated to '{new_role}'")
                        st.rerun()
    
    # Permissions testing
    with st.expander("🔍 Test Permissions", expanded=True):
        st.subheader("Test User Permissions")
        users_list = list(st.session_state.user_roles.keys())
        if users_list:
            test_user = st.selectbox("Select User to Test", users_list, key="test_user")
            if test_user:
                test_user_role = get_user_role(test_user)
                st.write(f"**User:** {test_user}")
                st.write(f"**Role:** {test_user_role}")
                
                st.write("**Permissions:**")
                permissions = ROLES.get(test_user_role, {}).get("permissions", [])
                for perm in permissions:
                    if st.checkbox(perm, key=f"perm_{perm}_{test_user}", disabled=True):
                        st.write(f"  ✓ {perm}")
                
                # Test specific permission
                if permissions:
                    perm_to_test = st.selectbox("Test Specific Permission", permissions, key="perm_test")
                    has_perm = has_permission(test_user, perm_to_test)
                    if has_perm:
                        st.success(f"✅ User '{test_user}' has permission '{perm_to_test}'")
                    else:
                        st.error(f"❌ User '{test_user}' does not have permission '{perm_to_test}'")
    
    # Summary
    st.subheader("📊 Access Control Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", len(st.session_state.user_roles))
    with col2:
        admin_count = sum(1 for role in st.session_state.user_roles.values() if role == "admin")
        st.metric("Admins", admin_count)
    with col3:
        editor_count = sum(1 for role in st.session_state.user_roles.values() if role == "editor")
        st.metric("Editors", editor_count)
    
    st.info("💡 RBAC helps you control who can access which features and data in your application.")
