"""
GitHub Integration Configuration and Utilities
This module provides utilities for integrating with GitHub repositories.
"""

import streamlit as st
import requests
from typing import Dict, Any, List
from datetime import datetime


def authenticate_github(token: str) -> bool:
    """
    Authenticate with GitHub using a personal access token.
    
    Args:
        token: GitHub personal access token
        
    Returns:
        True if authentication is successful, False otherwise
    """
    try:
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        response = requests.get("https://api.github.com/user", headers=headers)
        response.raise_for_status()
        user_data = response.json()
        
        # Store user data in session state
        st.session_state.github_user = user_data
        return True
    except Exception as e:
        st.error(f"GitHub authentication failed: {e}")
        return False


def list_github_repositories(token: str) -> List[Dict[str, Any]]:
    """
    List repositories accessible to the authenticated user.
    
    Args:
        token: GitHub personal access token
        
    Returns:
        List of repository dictionaries
    """
    try:
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Get user's repositories
        response = requests.get(
            "https://api.github.com/user/repos", 
            headers=headers,
            params={"sort": "updated", "direction": "desc"}
        )
        response.raise_for_status()
        repos = response.json()
        
        return [{"name": repo["full_name"], "id": repo["id"]} for repo in repos]
    except Exception as e:
        st.error(f"Error fetching repositories: {e}")
        return []


def list_linked_github_repositories() -> List[str]:
    """
    List repositories that are linked to the current project.
    
    Returns:
        List of linked repository names
    """
    if "linked_github_repos" not in st.session_state:
        st.session_state.linked_github_repos = []
    return st.session_state.linked_github_repos


def link_github_repository(token: str, repo_name: str) -> bool:
    """
    Link a GitHub repository to the current project.
    
    Args:
        token: GitHub personal access token
        repo_name: Full name of the repository (e.g., "username/repo")
        
    Returns:
        True if linking is successful, False otherwise
    """
    try:
        if "linked_github_repos" not in st.session_state:
            st.session_state.linked_github_repos = []
        
        # Check if already linked
        if repo_name in st.session_state.linked_github_repos:
            return True
            
        # Verify repository exists and is accessible
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        response = requests.get(f"https://api.github.com/repos/{repo_name}", headers=headers)
        response.raise_for_status()
        
        # Add to linked repositories
        st.session_state.linked_github_repos.append(repo_name)
        return True
    except Exception as e:
        st.error(f"Error linking repository {repo_name}: {e}")
        return False


def unlink_github_repository(repo_name: str) -> bool:
    """
    Unlink a GitHub repository from the current project.
    
    Args:
        repo_name: Full name of the repository to unlink
        
    Returns:
        True if unlinking is successful, False otherwise
    """
    try:
        if "linked_github_repos" in st.session_state:
            if repo_name in st.session_state.linked_github_repos:
                st.session_state.linked_github_repos.remove(repo_name)
                return True
        return False
    except Exception as e:
        st.error(f"Error unlinking repository {repo_name}: {e}")
        return False


def create_github_branch(token: str, repo_name: str, branch_name: str, base_branch: str = "main") -> bool:
    """
    Create a new branch in a GitHub repository.
    
    Args:
        token: GitHub personal access token
        repo_name: Full name of the repository (e.g., "username/repo")
        branch_name: Name of the new branch to create
        base_branch: Name of the base branch to create from (default: "main")
        
    Returns:
        True if branch creation is successful, False otherwise
    """
    try:
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Get the SHA of the base branch
        response = requests.get(
            f"https://api.github.com/repos/{repo_name}/git/refs/heads/{base_branch}",
            headers=headers
        )
        response.raise_for_status()
        base_sha = response.json()["object"]["sha"]
        
        # Create the new branch
        data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": base_sha
        }
        response = requests.post(
            f"https://api.github.com/repos/{repo_name}/git/refs",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        return True
    except Exception as e:
        st.error(f"Error creating branch {branch_name} in {repo_name}: {e}")
        return False


def commit_to_github(
    token: str,
    repo_name: str,
    file_path: str,
    content: str,
    commit_message: str,
    branch_name: str = "main"
) -> bool:
    """
    Commit content to a GitHub repository.
    
    Args:
        token: GitHub personal access token
        repo_name: Full name of the repository (e.g., "username/repo")
        file_path: Path to the file in the repository
        content: Content to commit
        commit_message: Commit message
        branch_name: Branch to commit to (default: "main")
        
    Returns:
        True if commit is successful, False otherwise
    """
    try:
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Check if file already exists to get its SHA (needed for updating)
        file_sha = None
        try:
            response = requests.get(
                f"https://api.github.com/repos/{repo_name}/contents/{file_path}",
                headers=headers,
                params={"ref": branch_name}
            )
            if response.status_code == 200:
                file_sha = response.json().get("sha")
        except Exception:
            # File doesn't exist, which is fine
            pass
        
        # Prepare commit data
        data = {
            "message": commit_message,
            "content": content.encode("utf-8").hex(),  # Encode to base64
            "branch": branch_name
        }
        
        # Add SHA if file exists
        if file_sha:
            data["sha"] = file_sha
            
        # Commit the file
        response = requests.put(
            f"https://api.github.com/repos/{repo_name}/contents/{file_path}",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        return True
    except Exception as e:
        st.error(f"Error committing to GitHub: {e}")
        return False


def sync_content_to_github(content: str, repo_name: str, file_path: str) -> bool:
    """
    Sync evolved content to a GitHub repository.
    
    Args:
        content: Content to sync
        repo_name: Full name of the repository
        file_path: Path to the file in the repository
        
    Returns:
        True if sync is successful, False otherwise
    """
    try:
        github_token = st.session_state.get("github_token")
        if not github_token:
            st.error("GitHub token not configured")
            return False
            
        commit_message = f"Update evolved content - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        branch_name = st.session_state.get("github_branch", "main")
        
        return commit_to_github(
            github_token,
            repo_name,
            file_path,
            content,
            commit_message,
            branch_name
        )
    except Exception as e:
        st.error(f"Error syncing content to GitHub: {e}")
        return False