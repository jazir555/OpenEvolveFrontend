import streamlit as st
import requests
import base64
from typing import List, Dict, Optional
from datetime import datetime


def send_discord_notification(webhook_url: str, message: str) -> bool:
    """Send a notification to a Discord webhook.

    Args:
        webhook_url (str): Discord webhook URL
        message (str): Message to send

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        payload = {"content": message}
        response = requests.post(webhook_url, json=payload, timeout=10)

        if response.status_code == 204:
            st.success("✅ Sent notification to Discord")
            return True
        else:
            st.error(
                f"❌ Failed to send Discord notification: {response.status_code} - {response.text}"
            )
            return False

    except Exception as e:
        st.error(f"❌ Error sending Discord notification: {e}")
        return False


def send_msteams_notification(webhook_url: str, message: str) -> bool:
    """Send a notification to a Microsoft Teams webhook.

    Args:
        webhook_url (str): Microsoft Teams webhook URL
        message (str): Message to send

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        payload = {"text": message}
        response = requests.post(webhook_url, json=payload, timeout=10)

        if response.status_code == 200:
            st.success("✅ Sent notification to Microsoft Teams")
            return True
        else:
            st.error(
                f"❌ Failed to send Microsoft Teams notification: {response.status_code} - {response.text}"
            )
            return False

    except Exception as e:
        st.error(f"❌ Error sending Microsoft Teams notification: {e}")
        return False


def send_generic_webhook(webhook_url: str, payload: dict) -> bool:
    """Send a payload to a generic webhook URL.

    Args:
        webhook_url (str): The webhook URL
        payload (dict): The JSON payload to send

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)

        if response.status_code >= 200 and response.status_code < 300:
            st.success("✅ Sent webhook notification")
            return True
        else:
            st.error(
                f"❌ Failed to send webhook notification: {response.status_code} - {response.text}"
            )
            return False

    except Exception as e:
        st.error(f"❌ Error sending webhook notification: {e}")
        return False


# GitHub Integration Functions
def authenticate_github(token: str) -> bool:
    """Authenticate with GitHub using a personal access token.

    Args:
        token (str): GitHub personal access token

    Returns:
        bool: True if authentication successful, False otherwise
    """
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.get(
            "https://api.github.com/user", headers=headers, timeout=10
        )

        if response.status_code == 200:
            user_data = response.json()
            st.session_state.github_user = user_data
            st.session_state.github_token = token
            st.success(f"✅ Successfully authenticated as {user_data['login']}")
            return True
        else:
            st.error(
                f"❌ GitHub authentication failed: {response.status_code} - {response.text}"
            )
            return False

    except Exception as e:
        st.error(f"❌ Error authenticating with GitHub: {e}")
        return False


def list_github_repositories(token: str) -> List[Dict]:
    """List repositories accessible to the authenticated GitHub user.

    Args:
        token (str): GitHub personal access token

    Returns:
        List[Dict]: List of repository information
    """
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        # Get user's repositories
        response = requests.get(
            "https://api.github.com/user/repos",
            headers=headers,
            params={"sort": "updated", "per_page": 100},
            timeout=15,
        )

        if response.status_code == 200:
            repos = response.json()
            return [
                {"name": repo["full_name"], "id": repo["id"], "url": repo["html_url"]}
                for repo in repos
            ]
        else:
            st.error(f"❌ Failed to fetch repositories: {response.status_code}")
            return []

    except Exception as e:
        st.error(f"❌ Error fetching repositories: {e}")
        return []


def create_github_branch(
    token: str, repository: str, branch_name: str, base_branch: str = "main"
) -> Optional[str]:
    """Create a new branch in a GitHub repository.

    Args:
        token (str): GitHub personal access token
        repository (str): Repository name (owner/repo)
        branch_name (str): Name of the new branch
        base_branch (str): Base branch to create from (default: main)

    Returns:
        Optional[str]: SHA of the new branch or None if failed
    """
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        # First, get the SHA of the base branch
        base_response = requests.get(
            f"https://api.github.com/repos/{repository}/git/refs/heads/{base_branch}",
            headers=headers,
            timeout=10,
        )

        if base_response.status_code != 200:
            st.error(
                f"❌ Failed to get base branch '{base_branch}': {base_response.status_code}"
            )
            return None

        base_sha = base_response.json()["object"]["sha"]

        # Create the new branch
        create_response = requests.post(
            f"https://api.github.com/repos/{repository}/git/refs",
            headers=headers,
            json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
            timeout=10,
        )

        if create_response.status_code == 201:
            new_branch_sha = create_response.json()["object"]["sha"]
            st.success(f"✅ Created branch '{branch_name}' in {repository}")
            return new_branch_sha
        else:
            st.error(
                f"❌ Failed to create branch '{branch_name}': {create_response.status_code} - {create_response.text}"
            )
            return None

    except Exception as e:
        st.error(f"❌ Error creating branch: {e}")
        return None


def commit_to_github(
    token: str,
    repository: str,
    file_path: str,
    content: str,
    commit_message: str,
    branch: str = "main",
) -> bool:
    """Commit content to a GitHub repository.

    Args:
        token (str): GitHub personal access token
        repository (str): Repository name (owner/repo)
        file_path (str): Path to the file in the repository
        content (str): Content to commit
        commit_message (str): Commit message
        branch (str): Branch to commit to (default: main)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        # First, check if the file already exists to get its SHA (needed for updates)
        file_sha = None
        try:
            file_response = requests.get(
                f"https://api.github.com/repos/{repository}/contents/{file_path}",
                headers=headers,
                params={"ref": branch},
                timeout=10,
            )

            if file_response.status_code == 200:
                file_sha = file_response.json()["sha"]
        except Exception:
            # File doesn\'t exist, which is fine
            pass

        # Prepare the commit payload
        commit_data = {
            "message": commit_message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch,
        }

        # If file exists, include the SHA for update
        if file_sha:
            commit_data["sha"] = file_sha

        # Make the commit
        commit_response = requests.put(
            f"https://api.github.com/repos/{repository}/contents/{file_path}",
            headers=headers,
            json=commit_data,
            timeout=15,
        )

        if commit_response.status_code in [200, 201]:
            st.success(f"✅ Committed to {repository}/{file_path} on branch '{branch}'")
            return True
        else:
            st.error(
                f"❌ Failed to commit: {commit_response.status_code} - {commit_response.text}"
            )
            return False

    except Exception as e:
        st.error(f"❌ Error committing to GitHub: {e}")
        return False


def get_github_commit_history(
    token: str, repository: str, file_path: str, branch: str = "main"
) -> List[Dict]:
    """Get commit history for a specific file in a GitHub repository.

    Args:
        token (str): GitHub personal access token
        repository (str): Repository name (owner/repo)
        file_path (str): Path to the file in the repository
        branch (str): Branch to get history from (default: main)

    Returns:
        List[Dict]: List of commit information
    """
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        response = requests.get(
            f"https://api.github.com/repos/{repository}/commits",
            headers=headers,
            params={"path": file_path, "sha": branch, "per_page": 20},
            timeout=15,
        )

        if response.status_code == 200:
            commits = response.json()
            return [
                {
                    "sha": commit["sha"][:8],
                    "message": commit["commit"]["message"],
                    "author": commit["commit"]["author"]["name"],
                    "date": commit["commit"]["author"]["date"],
                    "url": commit["html_url"],
                }
                for commit in commits
            ]
        else:
            st.error(f"❌ Failed to fetch commit history: {response.status_code}")
            return []

    except Exception as e:
        st.error(f"❌ Error fetching commit history: {e}")
        return []


# Note: This function is deprecated as GitHub functionality has been moved to the GitHub tab in mainlayout.py
# The functionality is now implemented using Streamlit native components instead of HTML/JS
def render_github_integration_ui():
    """Deprecated: GitHub integration UI. Use the GitHub tab in mainlayout.py instead."""
    import streamlit as st
    st.warning("GitHub integration is now available in the GitHub tab.")
    if st.button("Go to GitHub Tab"):
        # This requires Streamlit's experimental feature which might not work as expected
        st.info("Navigate to the GitHub tab (🐙) in the main interface")
    return ""


# GitHub Repository Integration
GITHUB_REPOS = {}


def link_github_repository(token: str, repo_name: str) -> bool:
    """Link a GitHub repository for protocol storage.

    Args:
        token (str): GitHub personal access token
        repo_name (str): Repository name (owner/repo)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        # Verify repository access
        response = requests.get(
            f"https://api.github.com/repos/{repo_name}", headers=headers, timeout=10
        )

        if response.status_code == 200:
            repo_data = response.json()
            GITHUB_REPOS[repo_name] = {
                "id": repo_data["id"],
                "name": repo_data["full_name"],
                "url": repo_data["html_url"],
                "default_branch": repo_data["default_branch"],
                "linked_at": datetime.now().isoformat(),
            }

            # Store in session state
            if "github_repos" not in st.session_state:
                st.session_state.github_repos = {}
            st.session_state.github_repos[repo_name] = GITHUB_REPOS[repo_name]

            st.success(f"✅ Linked GitHub repository: {repo_name}")
            return True
        else:
            st.error(f"❌ Failed to link repository: {response.status_code}")
            return False

    except Exception as e:
        st.error(f"❌ Error linking repository: {e}")
        return False


def unlink_github_repository(repo_name: str) -> bool:
    """Unlink a GitHub repository.

    Args:
        repo_name (str): Repository name to unlink

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if repo_name in GITHUB_REPOS:
            del GITHUB_REPOS[repo_name]

        if (
            "github_repos" in st.session_state
            and repo_name in st.session_state.github_repos
        ):
            del st.session_state.github_repos[repo_name]

        st.success(f"✅ Unlinked GitHub repository: {repo_name}")
        return True
    except Exception as e:
        st.error(f"❌ Error unlinking repository: {e}")
        return False


def list_linked_github_repositories() -> List[str]:
    """List all linked GitHub repositories.

    Returns:
        List[str]: List of linked repository names
    """
    if "github_repos" in st.session_state:
        return list(st.session_state.github_repos.keys())
    return []


def save_protocol_generation_to_github(
    repo_name: str,
    protocol_text: str,
    generation_name: str,
    branch_name: str = None,
    commit_message: str = None,
) -> bool:
    """Save a protocol generation to a GitHub repository.

    Args:
        repo_name (str): Linked GitHub repository name
        protocol_text (str): Protocol text to save
        generation_name (str): Name for this generation
        branch_name (str): Branch to save to (optional)
        commit_message (str): Custom commit message (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get repository info
        if (
            "github_repos" not in st.session_state
            or repo_name not in st.session_state.github_repos
        ):
            st.error(f"Repository '{repo_name}' is not linked")
            return False

        repo_info = st.session_state.github_repos[repo_name]
        token = st.session_state.get("github_token")

        if not token:
            st.error("GitHub token not found")
            return False

        # Determine branch
        target_branch = branch_name or repo_info["default_branch"]

        # Generate file path and commit message
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"protocols/{generation_name}_{timestamp}.md"
        commit_msg = commit_message or f"Add protocol generation: {generation_name}"

        # Create branch if needed
        if branch_name and branch_name != repo_info["default_branch"]:
            # Check if branch exists

            try:
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github.v3+json",
                }

                branch_response = requests.get(
                    f"https://api.github.com/repos/{repo_name}/branches/{branch_name}",
                    headers=headers,
                    timeout=10,
                )

                if branch_response.status_code == 200:
                    pass
                else:
                    # Create the branch
                    create_github_branch(
                        token, repo_name, branch_name, repo_info["default_branch"]
                    )
            except Exception:
                # Create the branch if it doesn't exist
                create_github_branch(
                    token, repo_name, branch_name, repo_info["default_branch"]
                )

        # Commit the protocol
        success = commit_to_github(
            token, repo_name, file_path, protocol_text, commit_msg, target_branch
        )

        if success:
            # Store generation info
            if "github_generations" not in st.session_state:
                st.session_state.github_generations = []

            st.session_state.github_generations.append(
                {
                    "repo": repo_name,
                    "file_path": file_path,
                    "branch": target_branch,
                    "generation_name": generation_name,
                    "timestamp": datetime.now().isoformat(),
                    "commit_message": commit_msg,
                }
            )

            st.success(
                f"✅ Saved protocol generation to {repo_name}/{file_path} on branch '{target_branch}'"
            )
            return True
        else:
            return False

    except Exception as e:
        st.error(f"❌ Error saving protocol generation: {e}")
        return False


def get_protocol_generations_from_github(repo_name: str) -> List[Dict]:
    """Get all protocol generations stored in a GitHub repository.

    Args:
        repo_name (str): Linked GitHub repository name

    Returns:
        List[Dict]: List of generation information
    """
    if "github_generations" in st.session_state:
        return [
            gen
            for gen in st.session_state.github_generations
            if gen["repo"] == repo_name
        ]
    return []


# Note: This function is deprecated as GitHub branching functionality has been moved to the GitHub tab in mainlayout.py
# The functionality is now implemented using Streamlit native components instead of HTML/JS
def render_github_branching_ui():
    """Deprecated: GitHub branching UI. Use the GitHub tab in mainlayout.py instead."""
    import streamlit as st
    st.warning("GitHub branching is now available in the GitHub tab.")
    if st.button("Go to GitHub Tab for Branching"):
        st.info("Navigate to the GitHub tab (🐙) in the main interface")
    return ""


# Note: This function is deprecated as remote storage functionality has been integrated into the GitHub tab in mainlayout.py
# The functionality is now implemented using Streamlit native components instead of HTML/JS
def render_remote_storage_ui():
    """Deprecated: Remote storage UI. Use the GitHub tab in mainlayout.py instead."""
    import streamlit as st
    st.warning("Remote storage is now available in the GitHub tab.")
    if st.button("Go to GitHub Tab for Remote Storage"):
        st.info("Navigate to the GitHub tab (🐙) in the main interface")
    return ""
