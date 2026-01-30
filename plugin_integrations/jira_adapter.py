"""
OpenEvolve Jira Integration Adapter

This adapter integrates the decomposition engine with Atlassian Jira for
issue tracking and project management.

FEATURES:
- Create Jira issues from sub-problems
- Update Jira issues with status changes
- Link related issues
- Sync decomposition plans with Jira epics
- Track progress via Jira workflow
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from jira import JIRA
    JIRA_AVAILABLE = True
except ImportError:
    JIRA_AVAILABLE = False

from plugin_system import PluginBase, PluginMetadata
from decomposition_engine import SubProblem, DecompositionPlan

logger = logging.getLogger(__name__)


class JiraIssueType(Enum):
    """Jira issue types."""
    EPIC = "Epic"
    STORY = "Story"
    TASK = "Task"
    SUBTASK = "Sub-task"
    BUG = "Bug"


@dataclass
class JiraConfig:
    """Jira integration configuration."""
    server_url: str
    username: str
    api_token: str
    project_key: str
    default_issue_type: JiraIssueType = JiraIssueType.STORY
    epic_link_field: Optional[str] = "customfield_10011"  # Default Epic link field
    story_point_field: Optional[str] = "customfield_10002"  # Default story points


@dataclass
class JiraMapping:
    """Mapping between sub-problem and Jira issue."""
    subproblem_id: str
    issue_key: str
    issue_type: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class JiraAdapter(PluginBase):
    """
    Jira integration plugin for OpenEvolve.

    This plugin creates and manages Jira issues based on decomposition results.

    Example:
        ```python
        from plugin_integrations.jira_adapter import JiraAdapter, JiraConfig

        config = JiraConfig(
            server_url="https://your-domain.atlassian.net",
            username="your-email@example.com",
            api_token="your-api-token",
            project_key="PROJ"
        )

        adapter = JiraAdapter(config)
        adapter.activate()

        # Sync decomposition plan
        adapter.sync_decomposition_plan(plan)
        ```
    """

    def __init__(self, config: Optional[JiraConfig] = None):
        metadata = PluginMetadata(
            name="jira_adapter",
            version="1.0.0",
            description="Jira integration for issue tracking",
            author="OpenEvolve",
            license="MIT",
            tags=["jira", "issue-tracking", "project-management"],
            category="integration"
        )

        super().__init__(metadata)

        self.config = config
        self._jira_client: Optional['JIRA'] = None
        self._mappings: Dict[str, JiraMapping] = {}

    def activate(self) -> bool:
        """Activate the Jira adapter."""
        if not JIRA_AVAILABLE:
            logger.error("Jira library not available. Install with: pip install jira")
            return False

        if not self.config:
            logger.error("Jira configuration not provided")
            return False

        try:
            # Initialize Jira client
            self._jira_client = JIRA(
                server=self.config.server_url,
                basic_auth=(self.config.username, self.config.api_token)
            )

            # Test connection
            self._jira_client.current_user()

            logger.info(f"Connected to Jira: {self.config.server_url}")

            # Register hooks
            self.register_hooks()

            return super().activate()

        except Exception as e:
            logger.error(f"Failed to connect to Jira: {e}")
            return False

    def deactivate(self) -> bool:
        """Deactivate the Jira adapter."""
        self._jira_client = None
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
            "on_workflow_complete",
            self.on_workflow_complete,
            priority=100
        )

    def on_after_decompose(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called after decomposition.

        Creates Jira issues for all sub-problems.
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

        Creates a Jira issue for the sub-problem.
        """
        subproblem = context.get('subproblem')
        if not subproblem:
            return context

        try:
            issue_key = self.create_issue_from_subproblem(subproblem)
            context['jira_issue_key'] = issue_key
        except Exception as e:
            logger.error(f"Failed to create Jira issue: {e}")

        return context

    def on_workflow_complete(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called when workflow completes.

        Updates Jira issues with completion status.
        """
        plan = context.get('plan')
        if not plan:
            return context

        try:
            self.mark_plan_complete(plan)
        except Exception as e:
            logger.error(f"Failed to mark plan complete in Jira: {e}")

        return context

    def sync_decomposition_plan(self, plan: DecompositionPlan) -> List[str]:
        """
        Sync a decomposition plan with Jira.

        Creates an Epic and linked issues for each sub-problem.

        Args:
            plan: Decomposition plan

        Returns:
            List of created issue keys
        """
        if not self._jira_client:
            raise RuntimeError("Jira client not initialized")

        issue_keys = []

        try:
            # Create Epic for the overall problem
            epic_key = self._create_epic(
                summary=plan.original_problem[:100],  # Truncate if needed
                description=self._format_epic_description(plan)
            )
            issue_keys.append(epic_key)

            # Create issues for each sub-problem
            for subproblem in plan.sub_problems:
                issue_key = self.create_issue_from_subproblem(
                    subproblem,
                    epic_key=epic_key
                )
                issue_keys.append(issue_key)

            logger.info(f"Created {len(issue_keys)} Jira issues for plan {plan.plan_id}")
            return issue_keys

        except Exception as e:
            logger.error(f"Failed to sync plan: {e}", exc_info=True)
            raise

    def _create_epic(self, summary: str, description: str) -> str:
        """Create an Epic issue."""
        epic_fields = {
            "project": {"key": self.config.project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Epic"}
        }

        # Add epic name field if available
        if hasattr(self._jira_client, 'fields'):
            epic_name_field = self._get_epic_name_field()
            if epic_name_field:
                epic_fields[epic_name_field] = summary

        epic = self._jira_client.create_issue(fields=epic_fields)
        logger.info(f"Created Epic: {epic.key}")
        return epic.key

    def _get_epic_name_field(self) -> Optional[str]:
        """Get the Epic name field for this Jira instance."""
        try:
            fields = self._jira_client.fields()
            for field in fields:
                if field['name'].lower() == 'epic name' or 'epic' in field['name'].lower():
                    return field['id']
        except (requests.RequestException, KeyError):
            pass
        return None

    def _format_epic_description(self, plan: DecompositionPlan) -> str:
        """Format Epic description from decomposition plan."""
        description = f"""*OpenEvolve Decomposition Plan*

*Plan ID:* {plan.plan_id}
*Strategy:* {plan.strategy.strategy_name}
*Created:* {plan.created_at.isoformat()}

*Problem:*
{plan.original_problem}

*Sub-problems:* {len(plan.sub_problems)}

*Quality Scores:*
- Cohesion: {plan.quality_scores.cohesion:.2f}
- Completeness: {plan.quality_scores.completeness:.2f}
- Clarity: {plan.quality_scores.clarity:.2f}

---
_Generated by OpenEvolve Decomposition Engine_
"""
        return description

    def create_issue_from_subproblem(
        self,
        subproblem: SubProblem,
        epic_key: Optional[str] = None
    ) -> str:
        """
        Create a Jira issue from a sub-problem.

        Args:
            subproblem: Sub-problem to create issue from
            epic_key: Optional epic key to link to

        Returns:
            Created issue key
        """
        if not self._jira_client:
            raise RuntimeError("Jira client not initialized")

        # Prepare issue fields
        fields = {
            "project": {"key": self.config.project_key},
            "summary": subproblem.title,
            "description": self._format_issue_description(subproblem),
            "issuetype": {"name": self.config.default_issue_type.value}
        }

        # Add Epic link if provided
        if epic_key and self.config.epic_link_field:
            fields[self.config.epic_link_field] = epic_key

        # Add story points if complexity score available
        if self.config.story_point_field and subproblem.complexity_score:
            # Convert complexity score (0-1) to story points (1-13)
            story_points = int(subproblem.complexity_score.value * 13) + 1
            fields[self.config.story_point_field] = story_points

        # Create issue
        issue = self._jira_client.create_issue(fields=fields)

        # Store mapping
        self._mappings[subproblem.id] = JiraMapping(
            subproblem_id=subproblem.id,
            issue_key=issue.key,
            issue_type=self.config.default_issue_type.value
        )

        logger.info(f"Created Jira issue {issue.key} for sub-problem {subproblem.id}")
        return issue.key

    def _format_issue_description(self, subproblem: SubProblem) -> str:
        """Format issue description from sub-problem."""
        description = f"""*Sub-Problem Details*

*ID:* {subproblem.id}
*Type:* {subproblem.problem_type.value}
*Complexity:* {subproblem.complexity_score.value:.2f}

*Description:*
{subproblem.description}

*Acceptance Criteria:*
{self._format_acceptance_criteria(subproblem)}

*Dependencies:*
{self._format_dependencies(subproblem)}

---
_Generated by OpenEvolve Decomposition Engine_
"""
        return description

    def _format_acceptance_criteria(self, subproblem: SubProblem) -> str:
        """Format acceptance criteria."""
        criteria = getattr(subproblem, 'acceptance_criteria', [])
        if not criteria:
            return "None specified"

        return "\n".join(f"- {c}" for c in criteria)

    def _format_dependencies(self, subproblem: SubProblem) -> str:
        """Format dependencies."""
        if not subproblem.dependencies:
            return "None"

        deps = []
        for dep in subproblem.dependencies:
            # Check if we have a Jira issue for this dependency
            if dep.id in self._mappings:
                deps.append(f"- {dep.title} [{self._mappings[dep.id].issue_key}]")
            else:
                deps.append(f"- {dep.title}")

        return "\n".join(deps) if deps else "None"

    def update_issue_status(self, issue_key: str, status: str) -> bool:
        """
        Update issue status.

        Args:
            issue_key: Jira issue key
            status: New status

        Returns:
            True if successful
        """
        if not self._jira_client:
            return False

        try:
            # Get available transitions
            issue = self._jira_client.issue(issue_key)
            transitions = self._jira_client.transitions(issue)

            # Find transition to target status
            transition_id = None
            for t in transitions:
                if t['name'].lower() == status.lower():
                    transition_id = t['id']
                    break

            if transition_id:
                self._jira_client.transition_issue(issue, transition_id)
                logger.info(f"Updated {issue_key} to {status}")
                return True
            else:
                logger.warning(f"No transition found to status: {status}")
                return False

        except Exception as e:
            logger.error(f"Failed to update issue status: {e}")
            return False

    def add_comment(self, issue_key: str, comment: str) -> bool:
        """
        Add comment to issue.

        Args:
            issue_key: Jira issue key
            comment: Comment text

        Returns:
            True if successful
        """
        if not self._jira_client:
            return False

        try:
            self._jira_client.add_comment(issue_key, comment)
            logger.info(f"Added comment to {issue_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to add comment: {e}")
            return False

    def link_issues(self, source_key: str, target_key: str, link_type: str = "Relates") -> bool:
        """
        Link two issues.

        Args:
            source_key: Source issue key
            target_key: Target issue key
            link_type: Type of link

        Returns:
            True if successful
        """
        if not self._jira_client:
            return False

        try:
            self._jira_client.create_issue_link(
                type=link_type,
                inwardIssue=source_key,
                outwardIssue=target_key
            )
            logger.info(f"Linked {source_key} to {target_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to link issues: {e}")
            return False

    def mark_plan_complete(self, plan: DecompositionPlan) -> None:
        """
        Mark all issues in a plan as complete.

        Args:
            plan: Decomposition plan
        """
        for subproblem in plan.sub_problems:
            if subproblem.id in self._mappings:
                issue_key = self._mappings[subproblem.id].issue_key
                self.update_issue_status(issue_key, "Done")

    def get_issue(self, issue_key: str) -> Optional[Dict[str, Any]]:
        """
        Get issue details.

        Args:
            issue_key: Jira issue key

        Returns:
            Issue details or None
        """
        if not self._jira_client:
            return None

        try:
            issue = self._jira_client.issue(issue_key)
            return {
                "key": issue.key,
                "summary": issue.fields.summary,
                "status": issue.fields.status.name,
                "description": issue.fields.description,
                "assignee": issue.fields.assignee.displayName if issue.fields.assignee else None
            }
        except Exception as e:
            logger.error(f"Failed to get issue: {e}")
            return None

    def search_issues(self, jql: str) -> List[Dict[str, Any]]:
        """
        Search issues using JQL.

        Args:
            jql: JQL query string

        Returns:
            List of issues
        """
        if not self._jira_client:
            return []

        try:
            issues = self._jira_client.search_issues(jql)
            return [
                {
                    "key": issue.key,
                    "summary": issue.fields.summary,
                    "status": issue.fields.status.name
                }
                for issue in issues
            ]
        except Exception as e:
            logger.error(f"Failed to search issues: {e}")
            return []


# Factory function
def create_jira_adapter(config: JiraConfig) -> JiraAdapter:
    """Create a Jira adapter instance."""
    return JiraAdapter(config)
