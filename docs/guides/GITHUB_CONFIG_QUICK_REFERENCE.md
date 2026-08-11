# GitHub Config - Quick Reference Guide

## Critical Fix Applied

**File:** `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\github_config.py`

**Line 375 (was 230):** Fixed base64 encoding for GitHub API

```python
# WRONG - This was the bug:
"content": content.encode("utf-8").hex()

# CORRECT - This is the fix:
encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
```

**Why:** GitHub API requires base64 encoding, not hex encoding. The bug would cause all file commits to fail.

---

## Quick Usage Examples

### Authentication

```python
from github_config import authenticate_github, check_rate_limit

# Authenticate
success = authenticate_github(token="ghp_your_token")

# Check rate limits
rate_info = check_rate_limit(token="ghp_your_token")
print(f"Remaining: {rate_info.remaining}/{rate_info.limit}")
```

### Repository Operations

```python
from github_config import (
    list_github_repositories,
    link_github_repository,
    create_github_branch,
    commit_to_github
)

# List repositories
repos = list_github_repositories(token="ghp_your_token")

# Link repository
link_github_repository(token="ghp_your_token", repo_name="user/repo")

# Create branch
create_github_branch(
    token="ghp_your_token",
    repo_name="user/repo",
    branch_name="feature-branch",
    base_branch="main"
)

# Commit file (NOW WORKS WITH BASE64 FIX)
commit_to_github(
    token="ghp_your_token",
    repo_name="user/repo",
    file_path="README.md",
    content="# My Project\n",
    commit_message="Update README",
    branch_name="main"
)
```

### Pull Request Management

```python
from github_config import (
    create_pull_request,
    update_pull_request,
    list_pull_requests,
    merge_pull_request,
    PRState
)

# Create PR
pr = create_pull_request(
    token="ghp_your_token",
    repo_name="user/repo",
    title="Add new feature",
    head_branch="feature-branch",
    base_branch="main",
    body="Description of changes"
)
print(f"Created PR: {pr.html_url}")

# Update PR
update_pull_request(
    token="ghp_your_token",
    repo_name="user/repo",
    pr_number=pr.number,
    body="Updated description"
)

# List PRs
prs = list_pull_requests(
    token="ghp_your_token",
    repo_name="user/repo",
    state=PRState.OPEN
)

# Merge PR
merge_pull_request(
    token="ghp_your_token",
    repo_name="user/repo",
    pr_number=pr.number,
    merge_method="merge"
)
```

### Webhook Management

```python
from github_config import (
    list_webhooks,
    create_webhook,
    delete_webhook,
    validate_webhook_signature
)

# Create webhook
webhook = create_webhook(
    token="ghp_your_token",
    repo_name="user/repo",
    url="https://example.com/webhook",
    secret="my-webhook-secret",
    events=["push", "pull_request"]
)

# List webhooks
webhooks = list_webhooks(token="ghp_your_token", repo_name="user/repo")

# Delete webhook
delete_webhook(
    token="ghp_your_token",
    repo_name="user/repo",
    hook_id=webhook["id"]
)

# Validate webhook signature (in your webhook handler)
is_valid = validate_webhook_signature(
    payload=request.get_data(),
    signature_header=request.headers.get("X-Hub-Signature-256"),
    webhook_secret="my-webhook-secret"
)
```

---

## Exception Handling

```python
from github_config import (
    GitHubError,
    GitHubAuthenticationError,
    GitHubRepositoryError,
    GitHubRateLimitError,
    GitHubFileError,
    GitHubPullRequestError
)

try:
    commit_to_github(...)
except GitHubAuthenticationError:
    print("Invalid token - check credentials")
except GitHubRateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except GitHubFileError as e:
    print(f"File operation failed: {e}")
except GitHubError as e:
    print(f"GitHub error: {e}")
```

---

## Key Features Added

### 1. Rate Limiting
- `check_rate_limit()` - Monitor API usage
- `handle_rate_limit()` - Automatic checking after requests
- Prevents quota exhaustion

### 2. Pull Requests
- `create_pull_request()` - Create new PRs
- `update_pull_request()` - Update title/body/state
- `list_pull_requests()` - List with filters
- `merge_pull_request()` - Merge with method selection

### 3. Webhooks
- `create_webhook()` - Create with secret
- `list_webhooks()` - List all webhooks
- `delete_webhook()` - Remove webhook
- `validate_webhook_signature()` - Security validation

### 4. Error Handling
- Custom exception hierarchy
- Specific error messages
- Proper exception propagation
- No more generic `Exception` catches

### 5. Type Safety
- Full type hints on all functions
- Data classes for structured data
- Better IDE support
- Catch errors at development time

---

## Testing

Run the verification test:

```bash
python test_github_config_fix.py
```

Expected output:
```
✓✓✓ ALL TESTS PASSED ✓✓✓
The fix correctly implements base64 encoding for GitHub API.
The old .hex() method has been replaced with base64.b64encode().
```

---

## Common Issues & Solutions

### Issue: "Failed to commit file"
**Cause:** Using old code with hex encoding
**Solution:** Ensure you're using the updated `github_config.py`

### Issue: "Rate limit exceeded"
**Cause:** Too many API calls
**Solution:** Use `check_rate_limit()` before operations

### Issue: "Webhook signature validation failed"
**Cause:** Missing or incorrect secret
**Solution:** Ensure webhook secret matches in `validate_webhook_signature()`

### Issue: "Branch already exists"
**Cause:** Trying to create duplicate branch
**Solution:** Check if branch exists first or catch `GitHubRepositoryError`

---

## Function Reference

### Core Functions
- `authenticate_github(token)` - Authenticate with GitHub
- `list_github_repositories(token)` - List accessible repos
- `link_github_repository(token, repo_name)` - Link repo to project
- `unlink_github_repository(repo_name)` - Unlink repo
- `create_github_branch(token, repo_name, branch_name, base_branch)` - Create branch
- `commit_to_github(token, repo_name, file_path, content, commit_message, branch_name)` - Commit file (FIXED)
- `sync_content_to_github(content, repo_name, file_path)` - Sync content

### Rate Limiting
- `check_rate_limit(token)` - Get rate limit info
- `handle_rate_limit(token, response)` - Check and handle limits

### Pull Requests
- `create_pull_request(token, repo_name, title, head_branch, base_branch, body, draft)` - Create PR
- `update_pull_request(token, repo_name, pr_number, title, body, state)` - Update PR
- `list_pull_requests(token, repo_name, state, head_branch, base_branch)` - List PRs
- `merge_pull_request(token, repo_name, pr_number, commit_message, merge_method)` - Merge PR

### Webhooks
- `validate_webhook_signature(payload, signature_header, webhook_secret)` - Validate webhook
- `list_webhooks(token, repo_name)` - List webhooks
- `create_webhook(token, repo_name, url, content_type, secret, events, active)` - Create webhook
- `delete_webhook(token, repo_name, hook_id)` - Delete webhook

---

## Data Classes

### RateLimitInfo
```python
@dataclass
class RateLimitInfo:
    limit: int        # Total rate limit
    remaining: int    # Remaining requests
    reset: int        # Unix timestamp of reset
    used: int         # Requests used
```

### PullRequestInfo
```python
@dataclass
class PullRequestInfo:
    number: int
    title: str
    state: str
    html_url: str
    created_at: str
    updated_at: str
    user: str
    body: Optional[str]
    base_branch: Optional[str]
    head_branch: Optional[str]
```

---

## File Statistics

**Before:** 284 lines, 7 functions, minimal error handling
**After:** 1,046 lines, 17 functions, comprehensive error handling

**Changes:**
- +762 lines of code
- +10 new functions
- +6 exception classes
- +4 data classes/enums
- Fixed 1 critical bug
- Removed 4 TODO comments

---

## Next Steps

1. Update any code using `commit_to_github()` to handle new exceptions
2. Implement webhook handlers using `validate_webhook_signature()`
3. Add rate limit checking in high-volume operations
4. Use PR functions for automated workflows
5. Run tests in your environment

---

**Last Updated:** 2026-01-22
**File:** C:\Users\mmeadow\Documents\OpenEvolve\Frontend\github_config.py
**Test:** test_github_config_fix.py
**Report:** GITHUB_CONFIG_FIX_REPORT.md
