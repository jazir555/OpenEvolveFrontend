# RBAC Enhanced System - Quick Reference Guide

## Installation

```bash
# Copy files to your project
cp rbac_enhanced.py /path/to/project/
cp rbac_enhanced_tests.py /path/to/project/
```

## 30-Second Setup

```python
from rbac_enhanced import create_rbac_system, UIRBAC, Permission

# Initialize
rbac = create_rbac_system(use_database=True)
st_rbac = UIRBAC(rbac)

# Create admin user (run once)
admin = rbac.create_user("admin", "admin@example.com", "secure_password", roles=["admin"])
```

## BubbleLab UI App Template

```python
import BubbleLab UI as st
from rbac_enhanced import create_rbac_system, UIRBAC, Permission

# Initialize
rbac = create_rbac_system()
st_rbac = UIRBAC(rbac)

# Main app
def main():
    user = st_rbac.get_current_user()

    if not user:
        st_rbac.login_form()
        return

    st.write(f"Welcome, {user.username}!")

    if st_rbac.permission_check(Permission.MANAGE_USERS):
        st_rbac.render_rbac_ui()

if __name__ == "__main__":
    main()
```

## Common Operations

### Create Users
```python
# Regular user
user = rbac.create_user("john", "john@example.com", "password123")

# User with specific roles
user = rbac.create_user(
    username="admin",
    email="admin@example.com",
    password="secure_password",
    roles=["admin"],
    is_superuser=True
)
```

### Authenticate
```python
# Native authentication
user = rbac.authenticate("username", "password")

# JWT token
token = rbac.generate_jwt_token(user, expires_in=3600)
verified_user = rbac.verify_token(token)

# API key
api_key = rbac.generate_api_key(user.user_id)
verified_user = rbac.verify_token(api_key)
```

### Check Permissions
```python
# Single permission
if rbac.has_permission(user, Permission.MANAGE_USERS):
    # Allow access

# Multiple permissions (OR)
if rbac.has_any_permission(user, [
    Permission.MANAGE_USERS,
    Permission.MANAGE_ROLES
]):
    # Allow access

# Multiple permissions (AND)
if rbac.has_all_permissions(user, [
    Permission.READ_CONTENT,
    Permission.WRITE_CONTENT
]):
    # Allow access
```

### Use Decorators
```python
@rbac.require_permission(Permission.MANAGE_USERS)
def admin_function():
    # Only executes if user has permission
    pass
```

### BubbleLab UI Permission Checks
```python
# Method 1: Direct check
if st_rbac.permission_check(Permission.MANAGE_USERS):
    st.button("Admin Button")

# Method 2: Decorator
@st_rbac.require_permission(Permission.MANAGE_USERS)
def render_admin_panel():
    st.write("Admin Panel")
```

## Available Permissions

### User Management
- `CREATE_USER` - Create new users
- `READ_USER` - View user information
- `UPDATE_USER` - Modify users
- `DELETE_USER` - Delete users
- `MANAGE_USERS` - Full user management

### Content Management
- `CREATE_CONTENT` - Create content
- `READ_CONTENT` - View content
- `UPDATE_CONTENT` - Edit content
- `DELETE_CONTENT` - Delete content
- `PUBLISH_CONTENT` - Publish content

### Project Management
- `CREATE_PROJECT` - Create projects
- `READ_PROJECT` - View projects
- `UPDATE_PROJECT` - Edit projects
- `DELETE_PROJECT` - Delete projects
- `MANAGE_PROJECT_MEMBERS` - Manage project members

### System Administration
- `SYSTEM_ADMIN` - Full system access
- `MANAGE_ROLES` - Manage roles
- `VIEW_LOGS` - View system logs
- `MANAGE_SYSTEM` - System configuration

### Analytics & API
- `VIEW_ANALYTICS` - View analytics
- `EXPORT_DATA` - Export data
- `API_ACCESS` - API read access
- `API_WRITE` - API write access

## Default Roles

### Admin
- **Permissions:** All permissions
- **Description:** Full system access
- **System Role:** Yes (cannot be deleted)

### Editor
- **Permissions:**
  - READ_CONTENT
  - CREATE_CONTENT
  - UPDATE_CONTENT
  - READ_PROJECT
- **Description:** Can edit and manage content
- **System Role:** Yes

### Viewer
- **Permissions:**
  - READ_CONTENT
  - READ_PROJECT
- **Description:** Read-only access
- **System Role:** Yes

## Storage Options

### SQLite (Recommended for Production)
```python
rbac = RBACSystem(
    storage_backend='database',
    storage_config={'db_path': 'rbac_system.db'}
)
```

### File-Based (Simple Deployments)
```python
rbac = RBACSystem(
    storage_backend='file',
    storage_config={'file_path': 'rbac_data.json'}
)
```

### Session State (Development Only)
```python
rbac = RBACSystem(
    storage_backend='session',
    storage_config={'use_session_state': True}
)
```

## Audit Logging

### Log Actions
```python
rbac.log_audit(
    user_id=user.user_id,
    action="CUSTOM_ACTION",
    resource_type="project",
    resource_id="proj_123",
    success=True,
    details={"project_name": "My Project"}
)
```

### Retrieve Logs
```python
# All logs
logs = rbac.get_audit_logs(limit=100)

# Filtered logs
logs = rbac.get_audit_logs(
    user_id=user.user_id,
    action="AUTHENTICATE",
    limit=50
)

# Display logs
for log in logs:
    print(f"{log.timestamp}: {log.action} - {log.success}")
```

## Testing

### Run All Tests
```bash
python rbac_enhanced_tests.py
```

### Run Specific Test Class
```python
import unittest
from rbac_enhanced_tests import TestRBACSystem

suite = unittest.TestLoader().loadTestsFromTestCase(TestRBACSystem)
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
```

## Environment Variables

```bash
# Database backend
export SOVEREIGN_DB_BACKEND=sqlite
export SOVEREIGN_DB_PATH=rbac_system.db

# PostgreSQL (if using)
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=rbac
export PGUSER=rbac_user
export PGPASSWORD=secure_password

# JWT secret
export RBAC_JWT_SECRET=your-secret-key-here
```

## Troubleshooting

### User Not Found
```python
# Check if user exists
user = rbac.get_user_by_username("username")
if not user:
    print("User does not exist")
```

### Authentication Fails
```python
# Verify password hash
user = rbac.get_user_by_username("username")
print(f"Password hash present: {user.password_hash is not None}")

# Check if user is active
print(f"User active: {user.is_active}")
```

### Permission Check Fails
```python
# Check user's roles
print(f"User roles: {user.role_names}")

# Check role's permissions
for role_name in user.role_names:
    role = rbac.get_role(role_name)
    print(f"Role {role_name}: {role.permissions}")
```

### Database Locked
```python
# Ensure single process for SQLite
# Or use connection pooling
```

## Best Practices

### Security
1. ✅ Always use HTTPS in production
2. ✅ Store secrets in environment variables
3. ✅ Use strong JWT secrets (32+ characters)
4. ✅ Implement rate limiting on auth endpoints
5. ✅ Regularly rotate API keys
6. ✅ Review audit logs weekly

### Performance
1. ✅ Use database storage for >100 users
2. ✅ Implement caching for permission checks
3. ✅ Use connection pooling
4. ✅ Index user_id and username columns

### Code Quality
1. ✅ Use type hints
2. ✅ Handle specific exceptions
3. ✅ Log security-relevant events
4. ✅ Write tests for new features
5. ✅ Document custom roles and permissions

## Migration from Basic RBAC

### Step 1: Install Enhanced RBAC
```python
# Old: import from rbac
# New: import from rbac_enhanced
from rbac_enhanced import create_rbac_system, UIRBAC
```

### Step 2: Initialize Storage
```python
# Old: Session state only
# New: Persistent storage
rbac = create_rbac_system(use_database=True)
```

### Step 3: Migrate Users
```python
# Old: st.session_state.user_roles
# New: Persistent users
for username, role_name in old_users.items():
    rbac.create_user(
        username=username,
        email=f"{username}@example.com",
        password="temp_password",  # User resets on first login
        roles=[role_name]
    )
```

### Step 4: Update Permission Checks
```python
# Old: has_permission(username, "permission")
# New: st_rbac.permission_check(Permission.PERMISSION_NAME)
if st_rbac.permission_check(Permission.MANAGE_USERS):
    # Protected code
```

### Step 5: Update UI
```python
# Old: Custom UI code
# New: Built-in UI
st_rbac.render_rbac_ui()  # Complete admin panel
```

## Support & Documentation

- **Full Documentation:** `rbac_enhanced_README.md`
- **Test Suite:** `rbac_enhanced_tests.py` (39 tests)
- **Enhancement Report:** `rbac_enhancement_report.md`

## License

Part of the OpenEvolve project.

---

**Version:** 2.0.0
**Status:** Production Ready
**Tests:** 39/39 Passing ✅


