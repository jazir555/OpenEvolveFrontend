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
            st.error(f"❌ Failed to send Discord notification: {response.status_code} - {response.text}")
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
            st.error(f"❌ Failed to send Microsoft Teams notification: {response.status_code} - {response.text}")
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
            st.error(f"❌ Failed to send webhook notification: {response.status_code} - {response.text}")
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
            "Accept": "application/vnd.github.v3+json"
        }

        response = requests.get(
            "https://api.github.com/user",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            user_data = response.json()
            st.session_state.github_user = user_data
            st.session_state.github_token = token
            st.success(f"✅ Successfully authenticated as {user_data['login']}")
            return True
        else:
            st.error(f"❌ GitHub authentication failed: {response.status_code} - {response.text}")
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
            "Accept": "application/vnd.github.v3+json"
        }

        # Get user's repositories
        response = requests.get(
            "https://api.github.com/user/repos",
            headers=headers,
            params={"sort": "updated", "per_page": 100},
            timeout=15
        )

        if response.status_code == 200:
            repos = response.json()
            return [{"name": repo["full_name"], "id": repo["id"], "url": repo["html_url"]} for repo in repos]
        else:
            st.error(f"❌ Failed to fetch repositories: {response.status_code}")
            return []

    except Exception as e:
        st.error(f"❌ Error fetching repositories: {e}")
        return []


def create_github_branch(token: str, repository: str, branch_name: str, base_branch: str = "main") -> Optional[str]:
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
            "Accept": "application/vnd.github.v3+json"
        }

        # First, get the SHA of the base branch
        base_response = requests.get(
            f"https://api.github.com/repos/{repository}/git/refs/heads/{base_branch}",
            headers=headers,
            timeout=10
        )

        if base_response.status_code != 200:
            st.error(f"❌ Failed to get base branch '{base_branch}': {base_response.status_code}")
            return None

        base_sha = base_response.json()["object"]["sha"]

        # Create the new branch
        create_response = requests.post(
            f"https://api.github.com/repos/{repository}/git/refs",
            headers=headers,
            json={
                "ref": f"refs/heads/{branch_name}",
                "sha": base_sha
            },
            timeout=10
        )

        if create_response.status_code == 201:
            new_branch_sha = create_response.json()["object"]["sha"]
            st.success(f"✅ Created branch '{branch_name}' in {repository}")
            return new_branch_sha
        else:
            st.error(
                f"❌ Failed to create branch '{branch_name}': {create_response.status_code} - {create_response.text}")
            return None

    except Exception as e:
        st.error(f"❌ Error creating branch: {e}")
        return None


def commit_to_github(token: str, repository: str, file_path: str, content: str, commit_message: str,
                     branch: str = "main") -> bool:
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
            "Accept": "application/vnd.github.v3+json"
        }

        # First, check if the file already exists to get its SHA (needed for updates)
        file_sha = None
        try:
            file_response = requests.get(
                f"https://api.github.com/repos/{repository}/contents/{file_path}",
                headers=headers,
                params={"ref": branch},
                timeout=10
            )

            if file_response.status_code == 200:
                file_sha = file_response.json()["sha"]
        except:
            # File doesn't exist, which is fine
            pass

        # Prepare the commit payload
        commit_data = {
            "message": commit_message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch
        }

        # If file exists, include the SHA for update
        if file_sha:
            commit_data["sha"] = file_sha

        # Make the commit
        commit_response = requests.put(
            f"https://api.github.com/repos/{repository}/contents/{file_path}",
            headers=headers,
            json=commit_data,
            timeout=15
        )

        if commit_response.status_code in [200, 201]:
            st.success(f"✅ Committed to {repository}/{file_path} on branch '{branch}'")
            return True
        else:
            st.error(f"❌ Failed to commit: {commit_response.status_code} - {commit_response.text}")
            return False

    except Exception as e:
        st.error(f"❌ Error committing to GitHub: {e}")
        return False


def get_github_commit_history(token: str, repository: str, file_path: str, branch: str = "main") -> List[Dict]:
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
            "Accept": "application/vnd.github.v3+json"
        }

        response = requests.get(
            f"https://api.github.com/repos/{repository}/commits",
            headers=headers,
            params={
                "path": file_path,
                "sha": branch,
                "per_page": 20
            },
            timeout=15
        )

        if response.status_code == 200:
            commits = response.json()
            return [{
                "sha": commit["sha"][:8],
                "message": commit["commit"]["message"],
                "author": commit["commit"]["author"]["name"],
                "date": commit["commit"]["author"]["date"],
                "url": commit["html_url"]
            } for commit in commits]
        else:
            st.error(f"❌ Failed to fetch commit history: {response.status_code}")
            return []

    except Exception as e:
        st.error(f"❌ Error fetching commit history: {e}")
        return []


def render_github_integration_ui() -> str:
    """Render the GitHub integration UI.

    Returns:
        str: HTML formatted GitHub integration UI
    """
    html = """
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #4a6fa5; margin-top: 0; text-align: center;">🔗 GitHub Integration</h2>

        <!-- Authentication Section -->
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #4a6fa5; margin-top: 0;">🔐 Authentication</h3>
    """

    # Check if already authenticated
    if st.session_state.get("github_user"):
        user = st.session_state.github_user
        html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <img src="{user.get('avatar_url', '')}" alt="Avatar" style="width: 40px; height: 40px; border-radius: 50%; margin-right: 10px;">
                <div>
                    <strong style="color: #4a6fa5;">{user.get('login', 'Unknown')}</strong>
                    <div style="font-size: 0.9em; color: #666;">Authenticated with GitHub</div>
                </div>
                <button onclick="disconnectGitHub()" style="margin-left: auto; background-color: #f44336; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Disconnect
                </button>
            </div>
        """
    else:
        html += """
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">GitHub Personal Access Token</label>
                <input type="password" id="githubToken" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd;" placeholder="Enter your GitHub personal access token">
                <p style="font-size: 0.8em; color: #666; margin-top: 5px;">
                    Generate a token with 'repo' permissions at <a href="https://github.com/settings/tokens" target="_blank">GitHub Settings</a>
                </p>
            </div>
            <button onclick="authenticateGitHub()" style="background-color: #4a6fa5; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                Authenticate with GitHub
            </button>
        """

    html += """
        </div>

        <!-- Repository Selection -->
        <div id="repoSelection" style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; display: none;">
            <h3 style="color: #4a6fa5; margin-top: 0;">📂 Select Repository</h3>
            <div id="repoList">
                <!-- Repositories will be populated by JavaScript -->
            </div>
        </div>

        <!-- Branch Management -->
        <div id="branchManagement" style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; display: none;">
            <h3 style="color: #4a6fa5; margin-top: 0;">🌿 Branch Management</h3>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Create New Branch</label>
                <div style="display: flex; gap: 10px;">
                    <input type="text" id="newBranchName" placeholder="Enter branch name (e.g., protocol-improvement-v1)" style="flex: 1; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
                    <button onclick="createBranch()" style="background-color: #4a6fa5; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer;">
                        Create Branch
                    </button>
                </div>
            </div>
            <div id="branchList">
                <!-- Branches will be populated by JavaScript -->
            </div>
        </div>

        <!-- Commit and Push -->
        <div id="commitSection" style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: none;">
            <h3 style="color: #4a6fa5; margin-top: 0;">💾 Commit and Push</h3>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">File Path</label>
                <input type="text" id="filePath" value="protocols/evolved_protocol.md" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Commit Message</label>
                <input type="text" id="commitMessage" placeholder="Enter commit message" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Branch</label>
                <select id="commitBranch" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
                    <option value="main">main</option>
                    <!-- Other branches will be populated by JavaScript -->
                </select>
            </div>
            <button onclick="commitToGitHub()" style="width: 100%; background-color: #4a6fa5; color: white; border: none; padding: 12px; border-radius: 5px; cursor: pointer; font-size: 1.1em;">
                🚀 Commit and Push to GitHub
            </button>
        </div>
    </div>

    <script>
    function authenticateGitHub() {
        const token = document.getElementById('githubToken').value;
        if (!token) {
            alert('Please enter a GitHub personal access token.');
            return;
        }

        // In a real implementation, this would authenticate with GitHub
        alert('Authenticating with GitHub... (This would connect in a real implementation)');
    }

    function disconnectGitHub() {
        // In a real implementation, this would disconnect from GitHub
        alert('Disconnecting from GitHub... (This would disconnect in a real implementation)');
    }

    function createBranch() {
        const branchName = document.getElementById('newBranchName').value;
        if (!branchName) {
            alert('Please enter a branch name.');
            return;
        }

        // In a real implementation, this would create a branch
        alert(`Creating branch: ${branchName} (This would create the branch in a real implementation)`);
    }

    function commitToGitHub() {
        const filePath = document.getElementById('filePath').value;
        const commitMessage = document.getElementById('commitMessage').value;
        const branch = document.getElementById('commitBranch').value;

        if (!filePath || !commitMessage) {
            alert('Please fill in all fields.');
            return;
        }

        // In a real implementation, this would commit to GitHub
        alert(`Committing to GitHub: ${filePath} on branch ${branch} (This would commit in a real implementation)`);
    }
    </script>
    """

    return html


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
            "Accept": "application/vnd.github.v3+json"
        }

        # Verify repository access
        response = requests.get(
            f"https://api.github.com/repos/{repo_name}",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            repo_data = response.json()
            GITHUB_REPOS[repo_name] = {
                "id": repo_data["id"],
                "name": repo_data["full_name"],
                "url": repo_data["html_url"],
                "default_branch": repo_data["default_branch"],
                "linked_at": datetime.now().isoformat()
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

        if "github_repos" in st.session_state and repo_name in st.session_state.github_repos:
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


def save_protocol_generation_to_github(repo_name: str, protocol_text: str, generation_name: str,
                                       branch_name: str = None, commit_message: str = None) -> bool:
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
        if "github_repos" not in st.session_state or repo_name not in st.session_state.github_repos:
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
            branch_exists = False
            try:
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github.v3+json"
                }

                branch_response = requests.get(
                    f"https://api.github.com/repos/{repo_name}/branches/{branch_name}",
                    headers=headers,
                    timeout=10
                )

                if branch_response.status_code == 200:
                    branch_exists = True
                else:
                    # Create the branch
                    create_github_branch(token, repo_name, branch_name, repo_info["default_branch"])
            except:
                # Create the branch if it doesn't exist
                create_github_branch(token, repo_name, branch_name, repo_info["default_branch"])

        # Commit the protocol
        success = commit_to_github(token, repo_name, file_path, protocol_text, commit_msg, target_branch)

        if success:
            # Store generation info
            if "github_generations" not in st.session_state:
                st.session_state.github_generations = []

            st.session_state.github_generations.append({
                "repo": repo_name,
                "file_path": file_path,
                "branch": target_branch,
                "generation_name": generation_name,
                "timestamp": datetime.now().isoformat(),
                "commit_message": commit_msg
            })

            st.success(f"✅ Saved protocol generation to {repo_name}/{file_path} on branch '{target_branch}'")
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
        return [gen for gen in st.session_state.github_generations if gen["repo"] == repo_name]
    return []


def render_github_branching_ui() -> str:
    """Render the GitHub branching UI.

    Returns:
        str: HTML formatted branching UI
    """
    html = """
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #4a6fa5; margin-top: 0; text-align: center;">🌿 GitHub Branching</h2>

        <!-- Branch Creation -->
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #4a6fa5; margin-top: 0;">Create New Branch</h3>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Branch Name</label>
                <input type="text" id="branchName" placeholder="e.g., protocol-improvement-v1" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Base Branch</label>
                <select id="baseBranch" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
                    <option value="main">main</option>
                    <option value="develop">develop</option>
                </select>
            </div>
            <button onclick="createBranch()" style="width: 100%; background-color: #4a6fa5; color: white; border: none; padding: 12px; border-radius: 5px; cursor: pointer; font-size: 1.1em;">
                🌿 Create Branch
            </button>
        </div>

        <!-- Branch List -->
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: #4a6fa5; margin-top: 0;">Existing Branches</h3>
            <div id="branchList" style="max-height: 300px; overflow-y: auto;">
                <!-- Branches will be populated by JavaScript -->
                <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #4caf50;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #2e7d32;">main</strong>
                            <div style="font-size: 0.9em; color: #666;">Default branch</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.9em; color: #666;">Last updated: Today</div>
                            <button onclick="switchToBranch('main')" style="background-color: #4a6fa5; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; font-size: 0.8em; margin-top: 5px;">
                                Switch
                            </button>
                        </div>
                    </div>
                </div>
                <div style="background-color: #fff8e1; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #ff9800;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #f57f17;">develop</strong>
                            <div style="font-size: 0.9em; color: #666;">Development branch</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.9em; color: #666;">Last updated: Yesterday</div>
                            <button onclick="switchToBranch('develop')" style="background-color: #4a6fa5; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; font-size: 0.8em; margin-top: 5px;">
                                Switch
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
    function createBranch() {
        const branchName = document.getElementById('branchName').value;
        const baseBranch = document.getElementById('baseBranch').value;

        if (!branchName) {
            alert('Please enter a branch name.');
            return;
        }

        // In a real implementation, this would create a branch
        alert(`Creating branch '${branchName}' from '${baseBranch}' (This would create the branch in a real implementation)`);
    }

    function switchToBranch(branchName) {
        // In a real implementation, this would switch to the branch
        alert(`Switching to branch: ${branchName} (This would switch branches in a real implementation)`);
    }
    </script>
    """

    return html


def render_remote_storage_ui() -> str:
    """Render the remote storage UI for protocol generations.

    Returns:
        str: HTML formatted remote storage UI
    """
    html = """
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #4a6fa5; margin-top: 0; text-align: center;">☁️ Remote Storage</h2>

        <!-- Save Generation -->
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #4a6fa5; margin-top: 0;">Save Current Generation</h3>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Generation Name</label>
                <input type="text" id="generationName" placeholder="e.g., security-policy-v1" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Repository</label>
                <select id="storageRepo" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
                    <option value="">Select a repository</option>
                    <option value="user/repo1">user/repo1</option>
                    <option value="user/repo2">user/repo2</option>
                </select>
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Branch (Optional)</label>
                <select id="storageBranch" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
                    <option value="">Use default branch</option>
                    <option value="main">main</option>
                    <option value="develop">develop</option>
                </select>
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Commit Message (Optional)</label>
                <input type="text" id="commitMessage" placeholder="Enter commit message" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
            </div>
            <button onclick="saveGeneration()" style="width: 100%; background: linear-gradient(135deg, #4a6fa5, #6b8cbc); color: white; border: none; padding: 12px; border-radius: 5px; cursor: pointer; font-size: 1.1em; font-weight: 600;">
                ☁️ Save to Remote Storage
            </button>
        </div>

        <!-- Generation History -->
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: #4a6fa5; margin-top: 0;">Generation History</h3>
            <div id="generationHistory" style="max-height: 400px; overflow-y: auto;">
                <!-- Generations will be populated by JavaScript -->
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #2196f3;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #1565c0;">security-policy-v1</strong>
                            <div style="font-size: 0.9em; color: #666;">user/repo1 • protocols/security-policy-v1_20231201_143022.md</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.9em; color: #666;">Dec 1, 2023 • 2:30 PM</div>
                            <div style="margin-top: 5px;">
                                <button onclick="viewGeneration('user/repo1', 'protocols/security-policy-v1_20231201_143022.md')" style="background-color: #4a6fa5; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; font-size: 0.8em; margin-right: 5px;">
                                    View
                                </button>
                                <button onclick="downloadGeneration('user/repo1', 'protocols/security-policy-v1_20231201_143022.md')" style="background-color: #6b8cbc; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; font-size: 0.8em;">
                                    Download
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #4caf50;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #2e7d32;">api-security-review-v2</strong>
                            <div style="font-size: 0.9em; color: #666;">user/repo2 • protocols/api-security-review-v2_20231130_091545.md</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.9em; color: #666;">Nov 30, 2023 • 9:15 AM</div>
                            <div style="margin-top: 5px;">
                                <button onclick="viewGeneration('user/repo2', 'protocols/api-security-review-v2_20231130_091545.md')" style="background-color: #4a6fa5; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; font-size: 0.8em; margin-right: 5px;">
                                    View
                                </button>
                                <button onclick="downloadGeneration('user/repo2', 'protocols/api-security-review-v2_20231130_091545.md')" style="background-color: #6b8cbc; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; font-size: 0.8em;">
                                    Download
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
    function saveGeneration() {
        const generationName = document.getElementById('generationName').value;
        const repo = document.getElementById('storageRepo').value;
        const branch = document.getElementById('storageBranch').value;
        const commitMsg = document.getElementById('commitMessage').value;

        if (!generationName || !repo) {
            alert('Please fill in all required fields.');
            return;
        }

        // In a real implementation, this would save the generation
        alert(`Saving generation '${generationName}' to ${repo}${branch ? ` on branch ${branch}` : ''} (This would save in a real implementation)`);
    }

    function viewGeneration(repo, filePath) {
        // In a real implementation, this would view the generation
        alert(`Viewing generation: ${repo}/${filePath} (This would display the content in a real implementation)`);
    }

    function downloadGeneration(repo, filePath) {
        // In a real implementation, this would download the generation
        alert(`Downloading generation: ${repo}/${filePath} (This would download the file in a real implementation)`);
    }
    </script>
    """

    return html