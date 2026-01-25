"""
OpenEvolve GitHub Integration Adapter

This adapter integrates the decomposition engine with GitHub for
repository management and issue tracking.

FEATURES:
- Create GitHub issues from sub-problems
- Create pull requests for solution implementations
- Link issues to decomposition plans
- Track progress via GitHub projects
- Sync documentation to GitHub wikis
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

from plugin_system import PluginBase, PluginMetadata
from decomposition_engine import SubProblem, DecompositionPlan

logger = logging.getLogger(__name__)


class GitHubIssueType(Enum):
    """GitHub issue types."""
    ISSUE = "issue"
    PULL_REQUEST = "pull_request"


@dataclass
class GitHubConfig:
    """GitHub integration configuration."""
    access_token: str
    repository: str  # format: "owner/repo"
    default_labels: List[str] = field(default_factory=list)
    create_projects: bool = False


@dataclass
class GitHubMapping:
    """Mapping between sub-problem and GitHub issue."""
    subproblem_id: str
    issue_number: int
    issue_type: GitHubIssueType
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class GitHubAdapter(PluginBase):
    """
    GitHub integration plugin for OpenEvolve.

    This plugin creates and manages GitHub issues and pull requests
    based on decomposition results.

    Example:
        ```python
        from plugin_integrations.github_adapter import GitHubAdapter, GitHubConfig

        config = GitHubConfig(
            access_token="your-github-token",
            repository="owner/repo"
        )

        adapter = GitHubAdapter(config)
        adapter.activate()

        # Sync decomposition plan
        adapter.sync_decomposition_plan(plan)
        ```
    """

    def __init__(self, config: Optional[GitHubConfig] = None):
        metadata = PluginMetadata(
            name="github_adapter",
            version="1.0.0",
            description="GitHub integration for repository management",
            author="OpenEvolve",
            license="MIT",
            tags=["github", "version-control", "issue-tracking"],
            category="integration"
        )

        super().__init__(metadata)

        self.config = config
        self._github_client: Optional['Github'] = None
        self._repo = None
        self._mappings: Dict[str, GitHubMapping] = {}

    def activate(self) -> bool:
        """Activate the GitHub adapter."""
        if not GITHUB_AVAILABLE:
            logger.error("PyGithub library not available. Install with: pip install PyGithub")
            return False

        if not self.config:
            logger.error("GitHub configuration not provided")
            return False

        try:
            # Initialize GitHub client
            self._github_client = Github(self.config.access_token)

            # Get repository
            self._repo = self._github_client.get_repo(self.config.repository)

            # Test access
            self._repo.get_issues(state="open", limit=1)

            logger.info(f"Connected to GitHub repository: {self.config.repository}")

            # Register hooks
            self.register_hooks()

            return super().activate()

        except Exception as e:
            logger.error(f"Failed to connect to GitHub: {e}")
            return False

    def deactivate(self) -> bool:
        """Deactivate the GitHub adapter."""
        self._github_client = None
        self._repo = None
        return super().deactivate()

    def register_hooks(self) -> None:
        """Register plugin hooks."""
        self.register_hook(
            "on_after_decompose",
            self.on_after_decompose,
            priority=100
        )

        self.register_hook(
            "on_subproblem_created",
            self.on_subproblem_created,
            priority=100
        )

        self.register_hook(
            "on_solution_assembled",
            self.on_solution_assembled,
            priority=100
        )

    def on_after_decompose(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called after decomposition.

        Creates GitHub issues for all sub-problems.
        """
        plan = context.get('plan')
        if not plan:
            return context

        try:
            self.sync_decomposition_plan(plan)
        except Exception as e:
            logger.error(f"Failed to sync decomposition plan: {e}")

        return context

    def on_subproblem_created(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called when sub-problem is created.

        Creates a GitHub issue for the sub-problem.
        """
        subproblem = context.get('subproblem')
        if not subproblem:
            return context

        try:
            issue_number = self.create_issue_from_subproblem(subproblem)
            context['github_issue_number'] = issue_number
        except Exception as e:
            logger.error(f"Failed to create GitHub issue: {e}")

        return context

    def on_solution_assembled(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called when solution is assembled.

        Creates pull request for solution.
        """
        solution = context.get('solution')
        if not solution:
            return context

        try:
            pr_number = self.create_pull_request_for_solution(solution)
            context['github_pr_number'] = pr_number
        except Exception as e:
            logger.error(f"Failed to create pull request: {e}")

        return context

    def sync_decomposition_plan(self, plan: DecompositionPlan) -> List[int]:
        """
        Sync a decomposition plan with GitHub.

        Creates a milestone and issues for each sub-problem.

        Args:
            plan: Decomposition plan

        Returns:
            List of created issue numbers
        """
        if not self._repo:
            raise RuntimeError("GitHub repository not initialized")

        issue_numbers = []

        try:
            # Create milestone for the plan
            milestone = self._create_milestone(
                title=plan.original_problem[:100],
                description=f"OpenEvolve Decomposition Plan: {plan.plan_id}"
            )

            # Create issues for each sub-problem
            for subproblem in plan.sub_problems:
                issue_number = self.create_issue_from_subproblem(
                    subproblem,
                    milestone_number=milestone.number
                )
                issue_numbers.append(issue_number)

            logger.info(f"Created {len(issue_numbers)} GitHub issues for plan {plan.plan_id}")
            return issue_numbers

        except Exception as e:
            logger.error(f"Failed to sync plan: {e}", exc_info=True)
            raise

    def _create_milestone(self, title: str, description: str) -> Any:
        """Create a GitHub milestone."""
        milestone = self._repo.create_milestone(
            title=title,
            description=description,
            state="open"
        )
        logger.info(f"Created milestone: {milestone.number}")
        return milestone

    def create_issue_from_subproblem(
        self,
        subproblem: SubProblem,
        milestone_number: Optional[int] = None
    ) -> int:
        """
        Create a GitHub issue from a sub-problem.

        Args:
            subproblem: Sub-problem to create issue from
            milestone_number: Optional milestone number

        Returns:
            Created issue number
        """
        if not self._repo:
            raise RuntimeError("GitHub repository not initialized")

        # Prepare issue title and body
        title = subproblem.title
        body = self._format_issue_body(subproblem)

        # Determine labels
        labels = self.config.default_labels.copy()
        labels.append(subproblem.problem_type.value)

        # Create issue
        issue = self._repo.create_issue(
            title=title,
            body=body,
            labels=labels,
            milestone=milestone_number
        )

        # Store mapping
        self._mappings[subproblem.id] = GitHubMapping(
            subproblem_id=subproblem.id,
            issue_number=issue.number,
            issue_type=GitHubIssueType.ISSUE
        )

        logger.info(f"Created GitHub issue #{issue.number} for sub-problem {subproblem.id}")
        return issue.number

    def _format_issue_body(self, subproblem: SubProblem) -> str:
        """Format issue body from sub-problem."""
        body = f"""## Sub-Problem Details

**ID:** {subproblem.id}
**Type:** {subproblem.problem_type.value}
**Complexity:** {subproblem.complexity_score.value:.2f}

### Description
{subproblem.description}

### Acceptance Criteria
{self._format_acceptance_criteria(subproblem)}

### Dependencies
{self._format_dependencies(subproblem)}

---
_Generated by [OpenEvolve](https://github.com/openevolve/openevolve) Decomposition Engine_
"""
        return body

    def _format_acceptance_criteria(self, subproblem: SubProblem) -> str:
        """Format acceptance criteria."""
        criteria = getattr(subproblem, 'acceptance_criteria', [])
        if not criteria:
            return "None specified"

        return "\n".join(f"- [ ] {c}" for c in criteria)

    def _format_dependencies(self, subproblem: SubProblem) -> str:
        """Format dependencies."""
        if not subproblem.dependencies:
            return "None"

        deps = []
        for dep in subproblem.dependencies:
            # Check if we have a GitHub issue for this dependency
            if dep.id in self._mappings:
                deps.append(f"- [#{self._mappings[dep.id].issue_number}] {dep.title}")
            else:
                deps.append(f"- {dep.title}")

        return "\n".join(deps) if deps else "None"

    def update_issue_state(self, issue_number: int, state: str) -> bool:
        """
        Update issue state.

        Args:
            issue_number: GitHub issue number
            state: New state (open/closed)

        Returns:
            True if successful
        """
        if not self._repo:
            return False

        try:
            issue = self._repo.get_issue(issue_number)
            issue.edit(state=state)
            logger.info(f"Updated issue #{issue_number} to {state}")
            return True
        except Exception as e:
            logger.error(f"Failed to update issue state: {e}")
            return False

    def add_comment(self, issue_number: int, comment: str) -> bool:
        """
        Add comment to issue.

        Args:
            issue_number: GitHub issue number
            comment: Comment text

        Returns:
            True if successful
        """
        if not self._repo:
            return False

        try:
            issue = self._repo.get_issue(issue_number)
            issue.create_comment(comment)
            logger.info(f"Added comment to issue #{issue_number}")
            return True
        except Exception as e:
            logger.error(f"Failed to add comment: {e}")
            return False

    def create_pull_request(
        self,
        title: str,
        body: str,
        head: str,
        base: str = "main"
    ) -> Optional[int]:
        """
        Create a pull request.

        Args:
            title: PR title
            body: PR description
            head: Branch name with changes
            base: Base branch to merge into

        Returns:
            PR number or None
        """
        if not self._repo:
            return None

        try:
            pr = self._repo.create_pull(
                title=title,
                body=body,
                head=head,
                base=base
            )
            logger.info(f"Created PR #{pr.number}")
            return pr.number
        except Exception as e:
            logger.error(f"Failed to create pull request: {e}")
            return None

    def create_pull_request_for_solution(
        self,
        solution: Dict[str, Any],
        base: str = "main"
    ) -> Optional[int]:
        """
        Create a pull request for a solution.

        Args:
            solution: Solution dictionary
            base: Base branch

        Returns:
            PR number or None
        """
        title = solution.get('title', 'Solution Implementation')
        body = solution.get('description', '')
        head = solution.get('branch', 'solution-branch')

        return self.create_pull_request(title, body, head, base)

    def link_issues(self, source_number: int, target_number: int) -> bool:
        """
        Link two issues.

        Args:
            source_number: Source issue number
            target_number: Target issue number

        Returns:
            True if successful
        """
        if not self._repo:
            return False

        try:
            source_issue = self._repo.get_issue(source_number)
            target_issue = self._repo.get_issue(target_number)

            # Add comment referencing the other issue
            source_issue.create_comment(
                f"Related to #{target_number}"
            )

            logger.info(f"Linked issue #{source_number} to #{target_number}")
            return True
        except Exception as e:
            logger.error(f"Failed to link issues: {e}")
            return False

    def get_issue(self, issue_number: int) -> Optional[Dict[str, Any]]:
        """
        Get issue details.

        Args:
            issue_number: GitHub issue number

        Returns:
            Issue details or None
        """
        if not self._repo:
            return None

        try:
            issue = self._repo.get_issue(issue_number)
            return {
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "body": issue.body,
                "assignee": issue.assignee.login if issue.assignee else None,
                "labels": [label.name for label in issue.labels]
            }
        except Exception as e:
            logger.error(f"Failed to get issue: {e}")
            return None

    def search_issues(self, query: str) -> List[Dict[str, Any]]:
        """
        Search issues using GitHub search.

        Args:
            query: Search query

        Returns:
            List of issues
        """
        if not self._repo:
            return []

        try:
            issues = self._repo.get_issues(state="open")
            return [
                {
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state
                }
                for issue in issues
            ]
        except Exception as e:
            logger.error(f"Failed to search issues: {e}")
            return []

    def create_project_board(self, name: str) -> Optional[int]:
        """
        Create a GitHub project board.

        Args:
            name: Project name

        Returns:
            Project ID or None
        """
        if not self._repo or not self.config.create_projects:
            return None

        try:
            # GitHub API v3 doesn't fully support projects, need v4
            logger.warning("Project creation requires GitHub GraphQL API")
            return None
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return None


# Factory function
def create_github_adapter(config: GitHubConfig) -> GitHubAdapter:
    """Create a GitHub adapter instance."""
    return GitHubAdapter(config)
