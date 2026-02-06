# Enhanced RBAC System - Production-Ready Access Control

A comprehensive, production-ready Role-Based Access Control (RBAC) system for Python applications with BubbleLab UI integration.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Storage Backends](#storage-backends)
- [Authentication](#authentication)
- [Authorization](#authorization)
- [BubbleLab UI Integration](#BubbleLab UI-integration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Security Considerations](#security-considerations)
- [Migration from Basic RBAC](#migration-from-basic-rbac)

## Features

### Core Features
- ✅ **Persistent Storage** - Multiple backend options (SQLite, PostgreSQL, MySQL, File-based)
- ✅ **Multiple Authentication Backends** - Native, JWT, API Keys, OAuth (planned), LDAP (planned)
- ✅ **Role-Based Access Control** - Flexible role and permission system
- ✅ **Permission Checking** - Decorators and methods for easy integration
- ✅ **Audit Logging** - Complete audit trail of all actions
- ✅ **Type Hints** - Full type annotations throughout
- ✅ **Comprehensive Error Handling** - Specific exception types
- ✅ **Production Logging** - Structured logging with contextual information

### BubbleLab UI Integration
- ✅ Built-in login form
- ✅ Permission-based UI rendering
- ✅ Session management
- ✅ RBAC management UI

### Security Features
- ✅ Password hashing with PBKDF2
- ✅ JWT token support with expiration
- ✅ API key authentication
- ✅ Audit logging for compliance
- ✅ Superuser support
- ✅ Account activation/deactivation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (FastAPI, Flask, BubbleLab UI, or custom Python app)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      RBAC System                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Authentication│  │ Authorization│  │   Audit      │     │
│  │   Manager     │  │    Manager   │  │   Logger     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Storage Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Database   │  │  File-Based  │  │  Session     │     │
│  │  (SQLite/PG) │  │    (JSON)    │  │   (Memory)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

```bash
pip install BubbleLab UI
```

### Optional Dependencies

```bash
# For JWT authentication
pip install pyjwt

# For PostgreSQL support
pip install psycopg2-binary

# For MySQL support
pip install pymysql

# For sovereign persistence integration
# (already included in your project)
```

### Quick Install

The enhanced RBAC system is a single-file module. Simply copy `rbac_enhanced.py` to your project:

```bash
cp rbac_enhanced.py /path/to/your/project/
```

## Quick Start

### Basic Usage

```python
from rbac_enhanced import create_rbac_system, Permission

# Create RBAC system
rbac = create_rbac_system(use_database=True)

# Create an admin user
admin = rbac.create_user(
    username="admin",
    email="admin@example.com",
    password="secure_password",
    roles=["admin"]
)

# Authenticate
user = rbac.authenticate("admin", "secure_password")

# Check permissions
if rbac.has_permission(user, Permission.MANAGE_USERS):
    print("Admin can manage users")
```

### BubbleLab UI Integration

```python
import BubbleLab UI as st
from rbac_enhanced import create_rbac_system, StreamlitRBAC

# Create RBAC system
rbac = create_rbac_system()
st_rbac = StreamlitRBAC(rbac)

# Get current user or show login
user = st_rbac.get_current_user()
if not user:
    st_rbac.login_form()
    st.stop()

# Show protected content
st.write(f"Welcome, {user.username}!")

if st_rbac.permission_check(Permission.MANAGE_USERS):
    st.write("Admin panel content")
```

## Configuration

### Storage Configuration

```python
from rbac_enhanced import RBACSystem

# Using SQLite database (recommended for production)
rbac = RBACSystem(
    storage_backend='database',
    storage_config={
        'db_path': 'rbac_system.db'
    }
)

# Using file-based storage (good for simple deployments)
rbac = RBACSystem(
    storage_backend='file',
    storage_config={
        'file_path': 'rbac_data.json'
    }
)

# Using BubbleLab UI session state (development only)
rbac = RBACSystem(
    storage_backend='session',
    storage_config={
        'use_session_state': True
    }
)
```

### Authentication Configuration

```python
# With JWT support
rbac = RBACSystem(
    storage_backend='database',
    jwt_secret='your-secret-key-here'
)

# Generate JWT token for user
token = rbac.generate_jwt_token(user, expires_in=3600)

# Verify token
verified_user = rbac.verify_token(token)
```

## Storage Backends

### 1. Database Storage (Recommended)

**Advantages:**
- Persistent across sessions
- Supports multiple applications
- ACID transactions
- Better performance for large datasets

**Usage:**
```python
from rbac_enhanced import create_rbac_system

rbac = create_rbac_system(
    use_database=True,
    database_path='rbac_system.db'
)
```

### 2. File-Based Storage

**Advantages:**
- Simple to set up
- Easy to backup
- No database dependencies
- Good for single-process applications

**Usage:**
```python
rbac = RBACSystem(
    storage_backend='file',
    storage_config={'file_path': 'rbac_data.json'}
)
```

### 3. Session State Storage

**Advantages:**
- No external dependencies
- Fast for testing
- Data isolated per session

**Disadvantages:**
- Not persistent
- Not suitable for production

**Usage:**
```python
rbac = RBACSystem(
    storage_backend='session',
    storage_config={'use_session_state': True}
)
```

## Authentication

### Native Authentication (Username/Password)

```python
# Create user
user = rbac.create_user(
    username="john",
    email="john@example.com",
    password="secure_password"
)

# Authenticate
authenticated_user = rbac.authenticate("john", "secure_password")
```

### JWT Authentication

```python
# Generate token
token = rbac.generate_jwt_token(user, expires_in=3600)

# Verify token
verified_user = rbac.verify_token(token)
```

### API Key Authentication

```python
# Generate API key
api_key = rbac.generate_api_key(user.user_id)
print(f"API Key: {api_key}")  # sk-...

# Verify API key
verified_user = rbac.verify_token(api_key)
```

## Authorization

### Permission Checking

```python
from rbac_enhanced import Permission

# Check single permission
if rbac.has_permission(user, Permission.MANAGE_USERS):
    # Show admin panel
    pass

# Check if user has any of these permissions
if rbac.has_any_permission(user, [
    Permission.MANAGE_USERS,
    Permission.MANAGE_ROLES
]):
    # Show management interface
    pass

# Check if user has all permissions
if rbac.has_all_permissions(user, [
    Permission.READ_CONTENT,
    Permission.WRITE_CONTENT
]):
    # Show full content access
    pass
```

### Using Decorators

```python
from rbac_enhanced import Permission

# Protect a function
@rbac.require_permission(Permission.MANAGE_USERS)
def create_new_user(username, email):
    # This will only execute if user has permission
    pass

# Call the function
try:
    create_new_user("john", "john@example.com")
except AuthorizationError:
    print("Access denied")
```

### BubbleLab UI Permission Checks

```python
from rbac_enhanced import StreamlitRBAC, Permission

st_rbac = StreamlitRBAC(rbac)

# Method 1: Using permission_check
if st_rbac.permission_check(Permission.MANAGE_USERS):
    if st.button("Create User"):
        # Show user creation form
        pass

# Method 2: Using decorator
@st_rbac.require_permission(Permission.MANAGE_USERS)
def render_admin_panel():
    st.write("Admin Panel")
```

## BubbleLab UI Integration

### Complete BubbleLab UI App Example

```python
import BubbleLab UI as st
from rbac_enhanced import create_rbac_system, StreamlitRBAC, Permission

# Page config
st.set_page_config(page_title="My App", page_icon="🔒")

# Initialize RBAC
rbac = create_rbac_system(use_database=True)
st_rbac = StreamlitRBAC(rbac)

# Main app
def main():
    st.title("🔒 My Protected App")

    # Get current user or show login
    user = st_rbac.get_current_user()

    if not user:
        st.warning("Please login to continue")
        st_rbac.login_form()
        return

    # Show user info
    st.sidebar.write(f"Logged in as: **{user.username}**")
    if st.sidebar.button("Logout"):
        st_rbac.logout()

    # Main content
    st.write(f"Welcome, {user.full_name or user.username}!")

    # Protected content
    if st_rbac.permission_check(Permission.READ_CONTENT):
        st.write("Protected content here...")

    # Admin panel
    if st_rbac.permission_check(Permission.MANAGE_USERS):
        st_rbac.render_rbac_ui()

if __name__ == "__main__":
    main()
```

### Custom Login Page

```python
def render_custom_login():
    st.title("Welcome to My App")

    # Custom styling
    st.markdown("""
    <style>
    .stForm { border: 1px solid #ddd; padding: 20px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

    # Login form
    user = st_rbac.login_form(key="custom_login")

    if user:
        st.success(f"Welcome back, {user.username}!")
        st.rerun()
```

## API Reference

### RBACSystem Class

#### Methods

##### `create_user(username, email, password, full_name=None, roles=None, is_superuser=False)`
Create a new user account.

**Parameters:**
- `username` (str): Unique username
- `email` (str): User email address
- `password` (str): Plain text password (will be hashed)
- `full_name` (str, optional): Full name
- `roles` (List[str], optional): List of role names
- `is_superuser` (bool): Superuser flag

**Returns:** `User` object

**Raises:** `RBACError` if creation fails

##### `authenticate(username, password, backend=AuthBackend.NATIVE)`
Authenticate a user.

**Parameters:**
- `username` (str): Username
- `password` (str): Password
- `backend` (AuthBackend): Authentication backend to use

**Returns:** `User` object or `None`

##### `has_permission(user, permission)`
Check if user has a specific permission.

**Parameters:**
- `user` (User): User object
- `permission` (str | Permission): Permission to check

**Returns:** `bool`

##### `create_role(name, description, permissions, is_system_role=False)`
Create a new role.

**Parameters:**
- `name` (str): Unique role name
- `description` (str): Role description
- `permissions` (List[str]): List of permission strings
- `is_system_role` (bool): System role flag

**Returns:** `Role` object

##### `require_permission(permission)`
Decorator to require a permission for function execution.

**Parameters:**
- `permission` (str | Permission): Required permission

**Returns:** Decorator function

##### `generate_jwt_token(user, expires_in=3600)`
Generate a JWT token for a user.

**Parameters:**
- `user` (User): User object
- `expires_in` (int): Token expiration time in seconds

**Returns:** `str` JWT token or `None`

##### `generate_api_key(user_id)`
Generate an API key for a user.

**Parameters:**
- `user_id` (str): User ID

**Returns:** `str` API key or `None`

### StreamlitRBAC Class

#### Methods

##### `login_form(key="login_form")`
Render a login form in BubbleLab UI.

**Parameters:**
- `key` (str): Unique form key

**Returns:** `User` object or `None`

##### `get_current_user()`
Get the current authenticated user.

**Returns:** `User` object or `None`

##### `logout()`
Logout the current user.

##### `permission_check(permission)`
Check if current user has permission.

**Parameters:**
- `permission` (str | Permission): Permission to check

**Returns:** `bool`

##### `require_permission(permission)`
BubbleLab UI decorator to require permission.

**Parameters:**
- `permission` (str | Permission): Required permission

**Returns:** Decorator function

##### `render_rbac_ui()`
Render the complete RBAC management UI.

## Examples

### Example 1: Basic Web App Protection

```python
from rbac_enhanced import create_rbac_system, StreamlitRBAC, Permission
import BubbleLab UI as st

# Initialize
rbac = create_rbac_system()
st_rbac = StreamlitRBAC(rbac)

# Protect page
user = st_rbac.get_current_user()
if not user:
    st_rbac.login_form()
    st.stop()

st.write(f"Welcome, {user.username}!")
```

### Example 2: Admin Panel

```python
@st_rbac.require_permission(Permission.MANAGE_USERS)
def render_admin_panel():
    st.header("Admin Panel")

    # List users
    users = rbac.list_users()
    for user in users:
        st.write(f"- {user.username} ({user.email})")

render_admin_panel()
```

### Example 3: Custom Permission Check

```python
def custom_permission_check(user, action, resource):
    """Custom permission checking logic."""
    # Check user's roles
    for role_name in user.role_names:
        role = rbac.get_role(role_name)
        if role and f"{action}_{resource}" in role.permissions:
            return True

    return False

# Use in your app
if custom_permission_check(user, "read", "projects"):
    st.write("Projects data...")
```

### Example 4: Audit Logging

```python
# Log custom action
rbac.log_audit(
    user_id=user.user_id,
    action="CUSTOM_ACTION",
    resource_type="project",
    resource_id="project_123",
    success=True,
    details={
        "action": "project_created",
        "project_name": "My Project"
    }
)

# Retrieve audit logs
logs = rbac.get_audit_logs(user_id=user.user_id, limit=100)
for log in logs:
    st.write(f"{log.timestamp}: {log.action} - {log.success}")
```

## Testing

Run the comprehensive test suite:

```bash
python rbac_enhanced_tests.py
```

### Test Coverage

The test suite includes:
- Storage layer tests (database, file, session)
- Authentication tests (native, JWT, API key)
- Authorization tests (permission checking)
- User management tests
- Role management tests
- Audit logging tests
- Edge cases and error handling
- Decorator tests

### Running Individual Tests

```python
import unittest
from rbac_enhanced_tests import TestRBACSystem

# Run specific test class
suite = unittest.TestLoader().loadTestsFromTestCase(TestRBACSystem)
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
```

## Security Considerations

### Password Security
- Passwords are hashed using PBKDF2-HMAC-SHA256 with 100,000 iterations
- Salt is randomly generated for each password
- Plain text passwords are never stored

### JWT Tokens
- Tokens expire after a configurable time (default: 1 hour)
- Use HTTPS in production to prevent token interception
- Store JWT secret securely (use environment variables)

### API Keys
- API keys are generated using cryptographically secure random generation
- Keys are prefixed with `sk-` for easy identification
- Implement key rotation in production

### Audit Logging
- All authentication attempts are logged
- All authorization failures are logged
- All user/role modifications are logged
- Include IP address and user agent when possible

### Best Practices
1. **Always use HTTPS in production**
2. **Use environment variables for sensitive configuration**
3. **Implement rate limiting for authentication endpoints**
4. **Regularly rotate JWT secrets and API keys**
5. **Review audit logs regularly**
6. **Use the principle of least privilege**
7. **Test your permission model thoroughly**

## Migration from Basic RBAC

### Before (Basic RBAC)

```python
# Old rbac.py
import BubbleLab UI as st

ROLES = {
    "admin": {"permissions": ["manage_users"]},
    "viewer": {"permissions": ["view_content"]},
}

if "user_roles" not in st.session_state:
    st.session_state.user_roles = {"admin": "admin"}

def get_user_role(username: str) -> str:
    return st.session_state.user_roles.get(username, "viewer")

def has_permission(username: str, permission: str) -> bool:
    role = get_user_role(username)
    return permission in ROLES.get(role, {}).get("permissions", [])
```

### After (Enhanced RBAC)

```python
# New rbac_enhanced.py
from rbac_enhanced import create_rbac_system, StreamlitRBAC, Permission

# Initialize with persistent storage
rbac = create_rbac_system(use_database=True)
st_rbac = StreamlitRBAC(rbac)

# Get current user
user = st_rbac.get_current_user()

# Check permissions (now persistent and more powerful)
if st_rbac.permission_check(Permission.MANAGE_USERS):
    st.write("Admin panel")
```

### Migration Steps

1. **Initialize the enhanced RBAC system:**
   ```python
   rbac = create_rbac_system(use_database=True)
   ```

2. **Migrate existing users:**
   ```python
   for username, role_name in st.session_state.user_roles.items():
       try:
           rbac.create_user(
               username=username,
               email=f"{username}@example.com",
               password="temporary_password",  # User should reset
               roles=[role_name]
           )
       except:
           pass  # User might already exist
   ```

3. **Update permission checks:**
   ```python
   # Old: has_permission(username, "manage_users")
   # New: st_rbac.permission_check(Permission.MANAGE_USERS)
   ```

4. **Update UI rendering:**
   ```python
   # Old: Custom UI code
   # New: st_rbac.render_rbac_ui()
   ```

## Troubleshooting

### Issue: "User not found" error
**Solution:** Ensure the user was created successfully. Check `rbac.list_users()` to see all users.

### Issue: JWT verification fails
**Solution:** Ensure the same JWT secret is used for generation and verification. Check token expiration.

### Issue: Permissions not working
**Solution:** Verify that:
1. User has the correct roles assigned
2. Roles have the correct permissions
3. You're checking the right permission

### Issue: Database locked error
**Solution:** Ensure only one process is accessing the SQLite database at a time. Use connection pooling for multi-process scenarios.

## Performance Considerations

### Database vs File Storage
- **Database:** Better for >1000 users or high-frequency operations
- **File:** Good for <100 users and simple deployments

### Caching
Consider implementing caching for frequently accessed permissions:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_has_permission(user_id: str, permission: str) -> bool:
    user = rbac.get_user(user_id)
    return rbac.has_permission(user, permission)
```

## License

This enhanced RBAC system is part of the OpenEvolve project.

## Support

For issues, questions, or contributions, please refer to the main OpenEvolve documentation.

## Changelog

### Version 2.0.0 (Enhanced)
- Added persistent storage backends
- Multiple authentication methods
- Comprehensive audit logging
- Type hints throughout
- Production-ready error handling
- BubbleLab UI integration
- JWT and API key support
- Full test suite

### Version 1.0.0 (Basic)
- BubbleLab UI session state storage
- Basic role/permission checking
- Simple user management

