"""
OpenEvolve Plugin Integrations Package

This package contains integration adapters for common third-party tools and services.

Available integrations:
- Jira: Issue tracking and project management
- GitHub: Code repository and project management
- Slack: Team communication and notifications
"""

from .jira_adapter import JiraAdapter
from .github_adapter import GitHubAdapter
from .slack_adapter import SlackAdapter

__all__ = [
    "JiraAdapter",
    "GitHubAdapter",
    "SlackAdapter",
]
