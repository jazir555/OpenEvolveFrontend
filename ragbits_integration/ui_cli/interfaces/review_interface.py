#!/usr/bin/env python
"""
Review Interface

Enhanced review interface for artifacts with inline commenting,
version comparison, and collaborative features.
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path


class ReviewStatus(Enum):
    """Review status states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class CommentType(Enum):
    """Comment types"""
    SUGGESTION = "suggestion"
    ISSUE = "issue"
    QUESTION = "question"
    APPROVAL = "approval"
    GENERAL = "general"


@dataclass
class ReviewComment:
    """Review comment on artifact"""
    comment_id: str
    author: str
    timestamp: datetime
    comment_type: CommentType
    content: str
    line_number: Optional[int] = None
    section: Optional[str] = None
    resolved: bool = False
    parent_comment_id: Optional[str] = None
    reactions: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ReviewDecision:
    """Review decision"""
    status: ReviewStatus
    summary: str
    reviewer: str
    timestamp: datetime
    conditions: List[str] = field(default_factory=list)
    approved_sections: List[str] = field(default_factory=list)
    rejected_sections: List[str] = field(default_factory=list)


@dataclass
class VersionDiff:
    """Version difference"""
    version_a: str
    version_b: str
    added_lines: List[int] = field(default_factory=list)
    removed_lines: List[int] = field(default_factory=list)
    modified_lines: List[int] = field(default_factory=list)
    unchanged_lines: List[int] = field(default_factory=list)
    diff_summary: str = ""


class ReviewInterface:
    """
    Enhanced review interface for artifacts.

    Features:
    - Inline commenting with threading
    - Version comparison and diffing
    - Collaborative review workflow
    - Approval/rejection tracking
    - Review metrics and analytics
    """

    def __init__(self, storage_manager=None, knowledge_retriever=None):
        """
        Initialize review interface.

        Args:
            storage_manager: Optional storage manager for artifact access
            knowledge_retriever: Optional knowledge retriever for context
        """
        from ragbits_integration.intermediary_storage import IntermediaryStorageManager
        from ragbits_integration.document_search import RagbitsKnowledgeRetriever

        self.storage = storage_manager
        self.retriever = knowledge_retriever

        # Review storage
        self._reviews: Dict[str, ReviewSession] = {}
        self._comments: Dict[str, List[ReviewComment]] = {}

    async def create_review_session(
        self,
        artifact_id: str,
        artifact_content: str,
        artifact_type: str,
        reviewers: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> "ReviewSession":
        """
        Create a new review session.

        Args:
            artifact_id: Artifact identifier
            artifact_content: Artifact content
            artifact_type: Type of artifact
            reviewers: List of reviewer IDs
            context: Optional context information

        Returns:
            ReviewSession object
        """
        # Generate review ID
        review_id = f"review_{artifact_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create session
        session = ReviewSession(
            review_id=review_id,
            artifact_id=artifact_id,
            artifact_content=artifact_content,
            artifact_type=artifact_type,
            reviewers=reviewers,
            context=context or {}
        )

        # Store session
        self._reviews[review_id] = session
        self._comments[review_id] = []

        # Gather context from knowledge base
        if self.retriever:
            session.knowledge_context = await self._gather_knowledge_context(
                artifact_content, artifact_type
            )

        return session

    async def add_comment(
        self,
        review_id: str,
        author: str,
        content: str,
        comment_type: CommentType = CommentType.GENERAL,
        line_number: Optional[int] = None,
        section: Optional[str] = None,
        parent_comment_id: Optional[str] = None
    ) -> ReviewComment:
        """
        Add a comment to review.

        Args:
            review_id: Review session ID
            author: Comment author
            content: Comment content
            comment_type: Type of comment
            line_number: Optional line number
            section: Optional section name
            parent_comment_id: Optional parent comment for threading

        Returns:
            ReviewComment object
        """
        if review_id not in self._reviews:
            raise ValueError(f"Review {review_id} not found")

        # Create comment
        comment = ReviewComment(
            comment_id=f"comment_{len(self._comments[review_id])}_{datetime.now().timestamp()}",
            author=author,
            timestamp=datetime.now(),
            comment_type=comment_type,
            content=content,
            line_number=line_number,
            section=section,
            parent_comment_id=parent_comment_id
        )

        # Add to comments
        self._comments[review_id].append(comment)

        # Update session
        session = self._reviews[review_id]
        session.last_updated = datetime.now()

        return comment

    async def resolve_comment(
        self,
        review_id: str,
        comment_id: str,
        resolver: str
    ) -> bool:
        """
        Resolve a comment.

        Args:
            review_id: Review session ID
            comment_id: Comment ID to resolve
            resolver: User resolving the comment

        Returns:
            True if resolved
        """
        if review_id not in self._comments:
            return False

        for comment in self._comments[review_id]:
            if comment.comment_id == comment_id:
                comment.resolved = True
                return True

        return False

    async def submit_decision(
        self,
        review_id: str,
        status: ReviewStatus,
        reviewer: str,
        summary: str,
        conditions: Optional[List[str]] = None,
        approved_sections: Optional[List[str]] = None,
        rejected_sections: Optional[List[str]] = None
    ) -> ReviewDecision:
        """
        Submit review decision.

        Args:
            review_id: Review session ID
            status: Review decision status
            reviewer: Reviewer ID
            summary: Decision summary
            conditions: Optional conditions for approval
            approved_sections: List of approved sections
            rejected_sections: List of rejected sections

        Returns:
            ReviewDecision object
        """
        if review_id not in self._reviews:
            raise ValueError(f"Review {review_id} not found")

        # Create decision
        decision = ReviewDecision(
            status=status,
            summary=summary,
            reviewer=reviewer,
            timestamp=datetime.now(),
            conditions=conditions or [],
            approved_sections=approved_sections or [],
            rejected_sections=rejected_sections or []
        )

        # Add to session
        session = self._reviews[review_id]
        session.decisions.append(decision)
        session.last_updated = datetime.now()

        # Update overall status
        session._update_overall_status()

        return decision

    async def compare_versions(
        self,
        artifact_id: str,
        version_a: str,
        version_b: str
    ) -> VersionDiff:
        """
        Compare two artifact versions.

        Args:
            artifact_id: Artifact identifier
            version_a: First version ID
            version_b: Second version ID

        Returns:
            VersionDiff object
        """
        if not self.storage:
            raise RuntimeError("Storage manager not configured")

        # Get both versions
        v_a = await self.storage.get_artifact_version(artifact_id, version_a)
        v_b = await self.storage.get_artifact_version(artifact_id, version_b)

        if not v_a or not v_b:
            raise ValueError("One or both versions not found")

        # Perform diff
        diff = self._compute_diff(v_a["content"], v_b["content"])

        return VersionDiff(
            version_a=version_a,
            version_b=version_b,
            added_lines=diff["added"],
            removed_lines=diff["removed"],
            modified_lines=diff["modified"],
            unchanged_lines=diff["unchanged"],
            diff_summary=diff["summary"]
        )

    async def get_review_summary(
        self,
        review_id: str
    ) -> Dict[str, Any]:
        """
        Get review session summary.

        Args:
            review_id: Review session ID

        Returns:
            Summary dictionary
        """
        if review_id not in self._reviews:
            raise ValueError(f"Review {review_id} not found")

        session = self._reviews[review_id]
        comments = self._comments[review_id]

        # Calculate metrics
        unresolved_comments = [c for c in comments if not c.resolved]
        issue_comments = [c for c in comments if c.comment_type == CommentType.ISSUE]

        return {
            "review_id": review_id,
            "artifact_id": session.artifact_id,
            "status": session.overall_status.value,
            "reviewers": session.reviewers,
            "total_comments": len(comments),
            "unresolved_comments": len(unresolved_comments),
            "issue_count": len(issue_comments),
            "decisions_submitted": len(session.decisions),
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat()
        }

    async def export_review_report(
        self,
        review_id: str,
        format: str = "markdown",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export review report.

        Args:
            review_id: Review session ID
            format: Export format (markdown, json, html)
            output_path: Optional output file path

        Returns:
            Report content
        """
        if review_id not in self._reviews:
            raise ValueError(f"Review {review_id} not found")

        session = self._reviews[review_id]
        comments = self._comments[review_id]

        if format == "markdown":
            report = self._generate_markdown_report(session, comments)
        elif format == "json":
            report = self._generate_json_report(session, comments)
        elif format == "html":
            report = self._generate_html_report(session, comments)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Save to file if requested
        if output_path:
            Path(output_path).write_text(report)

        return report

    async def _gather_knowledge_context(
        self,
        artifact_content: str,
        artifact_type: str
    ) -> Dict[str, Any]:
        """Gather relevant knowledge from base"""
        if not self.retriever:
            return {}

        try:
            # Find similar artifacts
            similar = await self.retriever.retrieve_similar_solutions(
                problem_description=artifact_content[:500],
                top_k=3
            )

            return {
                "similar_artifacts": [s.get("artifact_id") for s in similar],
                "patterns_found": len(similar)
            }
        except Exception as e:
            return {"error": str(e)}

    def _compute_diff(
        self,
        content_a: str,
        content_b: str
    ) -> Dict[str, Any]:
        """Compute diff between two contents"""
        lines_a = content_a.splitlines()
        lines_b = content_b.splitlines()

        # Simple line-by-line diff
        added = []
        removed = []
        modified = []
        unchanged = []

        # For now, use simple comparison
        # In production, use difflib.SequenceMatcher
        max_lines = max(len(lines_a), len(lines_b))

        for i in range(max_lines):
            line_a = lines_a[i] if i < len(lines_a) else None
            line_b = lines_b[i] if i < len(lines_b) else None

            if line_a == line_b:
                unchanged.append(i)
            elif line_a is None:
                added.append(i)
            elif line_b is None:
                removed.append(i)
            else:
                modified.append(i)

        summary = f"Added {len(added)}, removed {len(removed)}, modified {len(modified)} lines"

        return {
            "added": added,
            "removed": removed,
            "modified": modified,
            "unchanged": unchanged,
            "summary": summary
        }

    def _generate_markdown_report(
        self,
        session: "ReviewSession",
        comments: List[ReviewComment]
    ) -> str:
        """Generate markdown report"""
        lines = [
            f"# Review Report: {session.review_id}",
            "",
            f"**Artifact ID:** {session.artifact_id}",
            f"**Artifact Type:** {session.artifact_type}",
            f"**Status:** {session.overall_status.value}",
            f"**Created:** {session.created_at.isoformat()}",
            f"**Reviewers:** {', '.join(session.reviewers)}",
            "",
            "## Comments",
            ""
        ]

        # Group comments by section
        sections: Dict[str, List[ReviewComment]] = {}
        for comment in comments:
            section = comment.section or "general"
            if section not in sections:
                sections[section] = []
            sections[section].append(comment)

        # Output comments by section
        for section, section_comments in sections.items():
            lines.append(f"### {section.title()}")
            lines.append("")

            for comment in section_comments:
                status = "[OK] Resolved" if comment.resolved else "[FAIL] Open"
                lines.append(f"**{comment.comment_type.value.title()}** [{status}]")
                lines.append(f"- **Author:** {comment.author}")
                lines.append(f"- **Time:** {comment.timestamp.isoformat()}")
                if comment.line_number:
                    lines.append(f"- **Line:** {comment.line_number}")
                lines.append(f"- **Content:** {comment.content}")
                lines.append("")

        # Decisions
        if session.decisions:
            lines.append("## Decisions")
            lines.append("")

            for decision in session.decisions:
                lines.append(f"### {decision.reviewer}")
                lines.append(f"- **Status:** {decision.status.value}")
                lines.append(f"- **Time:** {decision.timestamp.isoformat()}")
                lines.append(f"- **Summary:** {decision.summary}")

                if decision.conditions:
                    lines.append(f"- **Conditions:**")
                    for cond in decision.conditions:
                        lines.append(f"  - {cond}")

                lines.append("")

        return "\n".join(lines)

    def _generate_json_report(
        self,
        session: "ReviewSession",
        comments: List[ReviewComment]
    ) -> str:
        """Generate JSON report"""
        report = {
            "review_id": session.review_id,
            "artifact_id": session.artifact_id,
            "artifact_type": session.artifact_type,
            "status": session.overall_status.value,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "reviewers": session.reviewers,
            "comments": [
                {
                    "comment_id": c.comment_id,
                    "author": c.author,
                    "timestamp": c.timestamp.isoformat(),
                    "type": c.comment_type.value,
                    "content": c.content,
                    "line_number": c.line_number,
                    "section": c.section,
                    "resolved": c.resolved
                }
                for c in comments
            ],
            "decisions": [
                {
                    "reviewer": d.reviewer,
                    "status": d.status.value,
                    "timestamp": d.timestamp.isoformat(),
                    "summary": d.summary,
                    "conditions": d.conditions,
                    "approved_sections": d.approved_sections,
                    "rejected_sections": d.rejected_sections
                }
                for d in session.decisions
            ]
        }

        return json.dumps(report, indent=2)

    def _generate_html_report(
        self,
        session: "ReviewSession",
        comments: List[ReviewComment]
    ) -> str:
        """Generate HTML report"""
        # Simple HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Review Report: {session.review_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .comment {{ border-left: 3px solid #007acc; padding: 10px; margin: 10px 0; }}
        .resolved {{ opacity: 0.6; }}
        .decision {{ background: #f5f5f5; padding: 15px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Review Report: {session.review_id}</h1>
    <p><strong>Artifact ID:</strong> {session.artifact_id}</p>
    <p><strong>Status:</strong> {session.overall_status.value}</p>
    <p><strong>Reviewers:</strong> {', '.join(session.reviewers)}</p>

    <h2>Comments ({len(comments)})</h2>
"""

        for comment in comments:
            resolved_class = "resolved" if comment.resolved else ""
            html += f"""
    <div class="comment {resolved_class}">
        <strong>{comment.comment_type.value.title()}</strong>
        by {comment.author}
        at {comment.timestamp.isoformat()}<br>
        {comment.content}
    </div>
"""

        if session.decisions:
            html += "<h2>Decisions</h2>"
            for decision in session.decisions:
                html += f"""
    <div class="decision">
        <strong>{decision.reviewer}:</strong> {decision.status.value}<br>
        {decision.summary}
    </div>
"""

        html += """
</body>
</html>
"""

        return html


@dataclass
class ReviewSession:
    """Review session data"""
    review_id: str
    artifact_id: str
    artifact_content: str
    artifact_type: str
    reviewers: List[str]
    context: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    decisions: List[ReviewDecision] = field(default_factory=list)
    overall_status: ReviewStatus = ReviewStatus.PENDING
    knowledge_context: Dict[str, Any] = field(default_factory=dict)

    def _update_overall_status(self):
        """Update overall status based on decisions"""
        if not self.decisions:
            self.overall_status = ReviewStatus.PENDING
            return

        # Check if all approved
        all_approved = all(d.status == ReviewStatus.APPROVED for d in self.decisions)
        any_rejected = any(d.status == ReviewStatus.REJECTED for d in self.decisions)

        if any_rejected:
            self.overall_status = ReviewStatus.REJECTED
        elif all_approved and len(self.decisions) == len(self.reviewers):
            self.overall_status = ReviewStatus.APPROVED
        else:
            self.overall_status = ReviewStatus.IN_PROGRESS


__all__ = ["ReviewInterface", "ReviewSession", "ReviewComment", "ReviewDecision", "ReviewStatus", "CommentType"]
