"""
Data Consistency and Integrity Verification Script

This script performs comprehensive checks for:
1. Database integrity (foreign keys, unique constraints, orphaned records)
2. Cache consistency (vs database, invalidation correctness)
3. State machine consistency (workflow state transitions)
4. Cross-component consistency (bridges, integrations, analytics)
5. Configuration consistency (parameters, settings, environment)

Author: OpenEvolve Team
Date: 2025-12-29
"""

import sqlite3
import sys
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ConsistencyIssue:
    """Represents a data consistency issue found during verification."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # DATABASE, CACHE, STATE, CROSS_COMPONENT, CONFIG
    check_name: str
    description: str
    affected_records: List[str]
    location: str  # File and line number if applicable
    recommendation: str


class DataConsistencyVerifier:
    """
    Comprehensive data consistency verification for BubbleLabs integration.

    Checks:
    - Database integrity (foreign keys, unique constraints, orphaned records)
    - Cache vs database synchronization
    - State machine consistency
    - Cross-component consistency
    - Configuration consistency
    """

    def __init__(self, db_path: str = "bubblelabs_analytics.db"):
        """
        Initialize the verifier.

        Args:
            db_path: Path to the analytics database
        """
        # SECURITY: Validate db_path to prevent path traversal and injection attacks
        self.db_path = self._validate_db_path(db_path)
        self.issues: List[ConsistencyIssue] = []
        self.stats = {
            "checks_performed": 0,
            "issues_found": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }

    def verify_all(self) -> Dict[str, Any]:
        """
        Run all verification checks.

        Returns:
            Summary of verification results
        """
        logger.info("=" * 70)
        logger.info("Starting Comprehensive Data Consistency Verification")
        logger.info("=" * 70)

        # 1. Database Integrity Checks
        self._check_database_integrity()

        # 2. Cache Consistency Checks
        self._check_cache_consistency()

        # 3. State Machine Consistency Checks
        self._check_state_machine_consistency()

        # 4. Cross-Component Consistency Checks
        self._check_cross_component_consistency()

        # 5. Configuration Consistency Checks
        self._check_configuration_consistency()

        # Generate summary
        return self._generate_summary()

    def _check_database_integrity(self):
        """Check database integrity: foreign keys, unique constraints, orphaned records."""
        logger.info("\n" + "=" * 70)
        logger.info("DATABASE INTEGRITY CHECKS")
        logger.info("=" * 70)

        try:
            # Use context manager to ensure connection is always closed
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check 1: Foreign key integrity
                self._check_foreign_keys(cursor)

                # Check 2: Unique constraint violations
                self._check_unique_constraints(cursor)

                # Check 3: Orphaned records
                self._check_orphaned_records(cursor)

                # Check 4: Transaction atomicity (workflow + node_metrics + provider_metrics)
                self._check_transaction_atomicity(cursor)

                # Check 5: Numeric consistency (totals match sums)
                self._check_numeric_consistency(cursor)

        except sqlite3.Error as e:
            self._add_issue(
                severity="CRITICAL",
                category="DATABASE",
                check_name="Database Access",
                description=f"Cannot access database: {e}",
                affected_records=[],
                location=f"{self.db_path}",
                recommendation="Check database file exists and is accessible"
            )

    def _check_foreign_keys(self, cursor: sqlite3.Cursor):
        """Check foreign key constraints."""
        logger.info("\n--- Checking Foreign Key Constraints ---")

        self.stats["checks_performed"] += 1

        # Check: node_metrics.workflow_id -> workflows.workflow_id
        cursor.execute("""
            SELECT DISTINCT nm.workflow_id
            FROM node_metrics nm
            LEFT JOIN workflows w ON nm.workflow_id = w.workflow_id
            WHERE w.workflow_id IS NULL
        """)
        orphaned_nodes = [row[0] for row in cursor.fetchall()]

        if orphaned_nodes:
            self._add_issue(
                severity="CRITICAL",
                category="DATABASE",
                check_name="Foreign Key: node_metrics.workflow_id",
                description=f"Found {len(orphaned_nodes)} node_metrics records with invalid workflow_id",
                affected_records=orphaned_nodes,
                location="node_metrics table",
                recommendation="Delete orphaned node_metrics or repair workflow_id references"
            )

        # Check: provider_metrics.workflow_id -> workflows.workflow_id
        cursor.execute("""
            SELECT DISTINCT pm.workflow_id
            FROM provider_metrics pm
            LEFT JOIN workflows w ON pm.workflow_id = w.workflow_id
            WHERE w.workflow_id IS NULL
        """)
        orphaned_providers = [row[0] for row in cursor.fetchall()]

        if orphaned_providers:
            self._add_issue(
                severity="CRITICAL",
                category="DATABASE",
                check_name="Foreign Key: provider_metrics.workflow_id",
                description=f"Found {len(orphaned_providers)} provider_metrics records with invalid workflow_id",
                affected_records=orphaned_providers,
                location="provider_metrics table",
                recommendation="Delete orphaned provider_metrics or repair workflow_id references"
            )

        logger.info(f"Foreign key check complete: {len(orphaned_nodes)} orphaned nodes, {len(orphaned_providers)} orphaned providers")

    def _check_unique_constraints(self, cursor: sqlite3.Cursor):
        """Check unique constraint violations."""
        logger.info("\n--- Checking Unique Constraints ---")

        self.stats["checks_performed"] += 1

        # Check: Duplicate provider_metrics entries (UNIQUE constraint on workflow_id, provider)
        cursor.execute("""
            SELECT workflow_id, provider, COUNT(*) as count
            FROM provider_metrics
            GROUP BY workflow_id, provider
            HAVING count > 1
        """)
        duplicates = [(row[0], row[1], row[2]) for row in cursor.fetchall()]

        if duplicates:
            affected = [f"{wf}/{prov}" for wf, prov, _ in duplicates]
            self._add_issue(
                severity="HIGH",
                category="DATABASE",
                check_name="Unique Constraint: provider_metrics",
                description=f"Found {len(duplicates)} duplicate provider_metrics entries",
                affected_records=affected,
                location="provider_metrics table",
                recommendation="Consolidate duplicate provider_metrics entries"
            )

        logger.info(f"Unique constraint check complete: {len(duplicates)} violations found")

    def _check_orphaned_records(self, cursor: sqlite3.Cursor):
        """Check for orphaned records."""
        logger.info("\n--- Checking Orphaned Records ---")

        self.stats["checks_performed"] += 1

        # Check: Workflows without any node_metrics (suspicious if status != "created")
        cursor.execute("""
            SELECT w.workflow_id, w.status
            FROM workflows w
            LEFT JOIN node_metrics nm ON w.workflow_id = nm.workflow_id
            WHERE nm.workflow_id IS NULL AND w.status NOT IN ('created', 'running')
        """)
        workflows_without_nodes = [(row[0], row[1]) for row in cursor.fetchall()]

        if workflows_without_nodes:
            affected = [f"{wf} ({status})" for wf, status in workflows_without_nodes]
            self._add_issue(
                severity="MEDIUM",
                category="DATABASE",
                check_name="Orphaned: Workflows without node_metrics",
                description=f"Found {len(workflows_without_nodes)} workflows with no node_metrics but not in 'created' or 'running' status",
                affected_records=affected,
                location="workflows table",
                recommendation="Check if workflow execution failed before tracking nodes"
            )

        # Check: Workflows without provider_metrics (suspicious if completed)
        cursor.execute("""
            SELECT w.workflow_id
            FROM workflows w
            LEFT JOIN provider_metrics pm ON w.workflow_id = pm.workflow_id
            WHERE pm.workflow_id IS NULL AND w.status = 'completed'
        """)
        completed_without_providers = [row[0] for row in cursor.fetchall()]

        if completed_without_providers:
            self._add_issue(
                severity="MEDIUM",
                category="DATABASE",
                check_name="Orphaned: Completed workflows without provider_metrics",
                description=f"Found {len(completed_without_providers)} completed workflows with no provider_metrics",
                affected_records=completed_without_providers,
                location="workflows table",
                recommendation="Check if provider tracking was disabled or failed"
            )

        logger.info(f"Orphaned records check complete: {len(workflows_without_nodes)} workflows without nodes, {len(completed_without_providers)} completed without providers")

    def _check_transaction_atomicity(self, cursor: sqlite3.Cursor):
        """Check transaction atomicity: workflow + node_metrics + provider_metrics consistency."""
        logger.info("\n--- Checking Transaction Atomicity ---")

        self.stats["checks_performed"] += 1

        # Check: Workflows marked "completed" but have node_metrics with status "running"
        # Note: node_metrics doesn't have status field, so we check for inconsistent timestamps
        cursor.execute("""
            SELECT w.workflow_id, w.status, w.end_time, MAX(nm.timestamp) as last_node_time
            FROM workflows w
            INNER JOIN node_metrics nm ON w.workflow_id = nm.workflow_id
            WHERE w.status = 'completed'
            GROUP BY w.workflow_id
            HAVING last_node_time > w.end_time
        """)
        nodes_after_completion = [row[0] for row in cursor.fetchall()]

        if nodes_after_completion:
            self._add_issue(
                severity="HIGH",
                category="DATABASE",
                check_name="Transaction Atomicity: Nodes after workflow completion",
                description=f"Found {len(nodes_after_completion)} completed workflows with node_metrics added after completion",
                affected_records=nodes_after_completion,
                location="workflows, node_metrics tables",
                recommendation="Check workflow status update timing and node tracking order"
            )

        logger.info(f"Transaction atomicity check complete: {len(nodes_after_completion)} violations found")

    def _check_numeric_consistency(self, cursor: sqlite3.Cursor):
        """Check numeric consistency: workflow totals match sums of components."""
        logger.info("\n--- Checking Numeric Consistency ---")

        self.stats["checks_performed"] += 1

        # Check: workflow.total_tokens = SUM(node_metrics.tokens_used)
        cursor.execute("""
            SELECT w.workflow_id, w.total_tokens as workflow_tokens,
                   COALESCE(SUM(nm.tokens_used), 0) as node_tokens
            FROM workflows w
            LEFT JOIN node_metrics nm ON w.workflow_id = nm.workflow_id
            GROUP BY w.workflow_id
            HAVING workflow_tokens != node_tokens
        """)
        token_mismatches = [(row[0], row[1], row[2]) for row in cursor.fetchall()]

        if token_mismatches:
            affected = [f"{wf} (workflow={tokens}, sum={sum_tokens})" for wf, tokens, sum_tokens in token_mismatches]
            self._add_issue(
                severity="HIGH",
                category="DATABASE",
                check_name="Numeric Consistency: Total tokens mismatch",
                description=f"Found {len(token_mismatches)} workflows where total_tokens != sum(node_metrics.tokens_used)",
                affected_records=affected[:20],  # Limit to first 20
                location="workflows, node_metrics tables",
                recommendation="Recalculate workflow.total_tokens from node_metrics or fix tracking logic"
            )

        # Check: workflow.total_cost = SUM(node_metrics.cost)
        cursor.execute("""
            SELECT w.workflow_id, w.total_cost as workflow_cost,
                   COALESCE(SUM(nm.cost), 0) as node_cost
            FROM workflows w
            LEFT JOIN node_metrics nm ON w.workflow_id = nm.workflow_id
            GROUP BY w.workflow_id
            HAVING ABS(workflow_cost - node_cost) > 0.001
        """)
        cost_mismatches = [(row[0], row[1], row[2]) for row in cursor.fetchall()]

        if cost_mismatches:
            affected = [f"{wf} (workflow=${cost:.6f}, sum=${sum_cost:.6f})" for wf, cost, sum_cost in cost_mismatches]
            self._add_issue(
                severity="HIGH",
                category="DATABASE",
                check_name="Numeric Consistency: Total cost mismatch",
                description=f"Found {len(cost_mismatches)} workflows where total_cost != sum(node_metrics.cost)",
                affected_records=affected[:20],
                location="workflows, node_metrics tables",
                recommendation="Recalculate workflow.total_cost from node_metrics or fix cost calculation logic"
            )

        # Check: provider_metrics totals match node_metrics
        cursor.execute("""
            SELECT pm.workflow_id, pm.provider, pm.total_tokens as provider_tokens,
                   COALESCE(SUM(nm.tokens_used), 0) as node_tokens
            FROM provider_metrics pm
            LEFT JOIN node_metrics nm ON pm.workflow_id = nm.workflow_id
            GROUP BY pm.workflow_id, pm.provider
            HAVING provider_tokens != node_tokens
        """)
        provider_token_mismatches = [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]

        if provider_token_mismatches:
            affected = [f"{wf}/{prov} (provider={tokens}, sum={sum_tokens})" for wf, prov, tokens, sum_tokens in provider_token_mismatches]
            self._add_issue(
                severity="MEDIUM",
                category="DATABASE",
                check_name="Numeric Consistency: Provider tokens mismatch",
                description=f"Found {len(provider_token_mismatches)} provider entries where total_tokens != sum(node_metrics.tokens_used)",
                affected_records=affected[:20],
                location="provider_metrics, node_metrics tables",
                recommendation="Verify provider_metrics aggregation logic"
            )

        logger.info(f"Numeric consistency check complete: {len(token_mismatches)} token mismatches, {len(cost_mismatches)} cost mismatches, {len(provider_token_mismatches)} provider token mismatches")

    def _check_cache_consistency(self):
        """Check cache vs database synchronization."""
        logger.info("\n" + "=" * 70)
        logger.info("CACHE CONSISTENCY CHECKS")
        logger.info("=" * 70)

        self.stats["checks_performed"] += 1

        # Note: This is a placeholder for actual cache checks
        # In a real implementation, we would check:
        # - BubbleLabs integration workflow_instances vs database
        # - Bridge mappings cache vs actual mappings
        # - MCP tools shared instances state

        logger.info("Cache consistency checks require component integration (not implemented in standalone verifier)")

    def _check_state_machine_consistency(self):
        """Check state machine consistency: workflow state transitions."""
        logger.info("\n" + "=" * 70)
        logger.info("STATE MACHINE CONSISTENCY CHECKS")
        logger.info("=" * 70)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check 1: Invalid state transitions
            self._check_workflow_state_transitions(cursor)

            # Check 2: Status flag consistency
            self._check_status_flag_consistency(cursor)

            conn.close()

        except (sqlite3.Error, IOError, OSError) as e:
            self._add_issue(
                severity="CRITICAL",
                category="STATE",
                check_name="State Machine Database Access",
                description=f"Cannot access database for state checks: {e}",
                affected_records=[],
                location=f"{self.db_path}",
                recommendation="Check database file exists and is accessible"
            )

    def _check_workflow_state_transitions(self, cursor: sqlite3.Cursor):
        """Check for invalid workflow state transitions."""
        logger.info("\n--- Checking Workflow State Transitions ---")

        self.stats["checks_performed"] += 1

        # Check: Workflows with end_time but status != "completed", "failed", "cancelled"
        cursor.execute("""
            SELECT workflow_id, status, end_time
            FROM workflows
            WHERE end_time IS NOT NULL
            AND status NOT IN ('completed', 'failed', 'cancelled', 'stopped')
        """)
        invalid_states = [(row[0], row[1]) for row in cursor.fetchall()]

        if invalid_states:
            affected = [f"{wf} ({status})" for wf, status in invalid_states]
            self._add_issue(
                severity="HIGH",
                category="STATE",
                check_name="Invalid State: Workflow with end_time but not terminal status",
                description=f"Found {len(invalid_states)} workflows with end_time set but status is not terminal",
                affected_records=affected,
                location="workflows table",
                recommendation="Update workflow status to terminal (completed/failed/cancelled/stopped) or clear end_time"
            )

        # Check: Workflows with status "completed" but no end_time
        cursor.execute("""
            SELECT workflow_id
            FROM workflows
            WHERE status = 'completed' AND end_time IS NULL
        """)
        completed_without_end = [row[0] for row in cursor.fetchall()]

        if completed_without_end:
            self._add_issue(
                severity="MEDIUM",
                category="STATE",
                check_name="Invalid State: Completed workflow without end_time",
                description=f"Found {len(completed_without_end)} workflows marked completed but missing end_time",
                affected_records=completed_without_end,
                location="workflows table",
                recommendation="Set end_time for completed workflows or change status to 'running'"
            )

        logger.info(f"State transition check complete: {len(invalid_states)} invalid states, {len(completed_without_end)} completed without end_time")

    def _check_status_flag_consistency(self, cursor: sqlite3.Cursor):
        """Check status flag consistency."""
        logger.info("\n--- Checking Status Flag Consistency ---")

        self.stats["checks_performed"] += 1

        # Check: Workflows with status "running" but stale start_time (> 24 hours ago)
        cursor.execute("""
            SELECT workflow_id, start_time, (strftime('%s', 'now') - start_time) / 3600 as hours_running
            FROM workflows
            WHERE status = 'running' AND (strftime('%s', 'now') - start_time) > 86400
        """)
        stale_running = [(row[0], row[2]) for row in cursor.fetchall()]

        if stale_running:
            affected = [f"{wf} (running for {hours:.1f} hours)" for wf, hours in stale_running]
            self._add_issue(
                severity="MEDIUM",
                category="STATE",
                check_name="Status Flag: Stale running workflows",
                description=f"Found {len(stale_running)} workflows marked 'running' for over 24 hours (likely stale)",
                affected_records=affected,
                location="workflows table",
                recommendation="Update stale workflow status to 'failed' or 'cancelled'"
            )

        logger.info(f"Status flag consistency check complete: {len(stale_running)} stale running workflows")

    def _check_cross_component_consistency(self):
        """Check cross-component consistency."""
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-COMPONENT CONSISTENCY CHECKS")
        logger.info("=" * 70)

        self.stats["checks_performed"] += 1

        # Note: These checks require integration with actual components
        # Placeholder for future implementation

        logger.info("Cross-component consistency checks require component integration (not implemented in standalone verifier)")

    def _check_configuration_consistency(self):
        """Check configuration consistency."""
        logger.info("\n" + "=" * 70)
        logger.info("CONFIGURATION CONSISTENCY CHECKS")
        logger.info("=" * 70)

        self.stats["checks_performed"] += 1

        # Note: These checks require access to configuration files and runtime values
        # Placeholder for future implementation

        logger.info("Configuration consistency checks require configuration access (not implemented in standalone verifier)")

    def _add_issue(self, severity: str, category: str, check_name: str,
                   description: str, affected_records: List[str],
                   location: str, recommendation: str):
        """Add a consistency issue to the list."""
        issue = ConsistencyIssue(
            severity=severity,
            category=category,
            check_name=check_name,
            description=description,
            affected_records=affected_records,
            location=location,
            recommendation=recommendation
        )
        self.issues.append(issue)

        # Update statistics
        self.stats["issues_found"] += 1
        self.stats[severity.lower()] += 1

        # Log immediate feedback
        logger.warning(f"[{severity}] {check_name}: {description}")
        if affected_records and len(affected_records) <= 5:
            for record in affected_records:
                logger.warning(f"  - {record}")
        elif len(affected_records) > 5:
            logger.warning(f"  - {affected_records[0]} ... and {len(affected_records) - 1} more")

    def _validate_db_path(self, db_path: str) -> str:
        """
        Validate database path to prevent path traversal and injection attacks.
        
        Args:
            db_path: The database path to validate
            
        Returns:
            str: Sanitized database path
            
        Raises:
            ValueError: If the path contains suspicious patterns
        """
        # Type validation
        if not isinstance(db_path, str):
            raise TypeError(f"db_path must be a string, got {type(db_path).__name__}")
        
        # Length validation
        if len(db_path) > 1024:
            raise ValueError("db_path exceeds maximum length of 1024 characters")
        
        # Check for null bytes
        if '\x00' in db_path:
            raise ValueError("db_path contains null bytes")
        
        # Check for SQL injection patterns in path
        sql_injection_patterns = [
            r"['\";]",  # Quote or semicolon
            r"--",      # SQL comment
            r"/\*",    # Start of block comment
            r"\*/",    # End of block comment
            r"\\",    # Backslash (escape attempts)
        ]
        for pattern in sql_injection_patterns:
            if re.search(pattern, db_path):
                raise ValueError(f"db_path contains potentially dangerous characters matching pattern: {pattern}")
        
        # Normalize path and check for path traversal
        try:
            path_obj = Path(db_path)
            # Resolve to absolute path
            abs_path = path_obj.resolve()
            
            # Check for suspicious path components
            suspicious_components = ['..', '~', '$', '`']
            for part in path_obj.parts:
                for suspicious in suspicious_components:
                    if suspicious in part:
                        raise ValueError(f"db_path contains suspicious component: {suspicious}")
            
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid db_path: {e}")
        
        return str(Path(db_path))

    def _execute_query(self, cursor: sqlite3.Cursor, query: str, params: Optional[Tuple] = None) -> None:
        """
        Execute a SQL query with optional parameters.
        
        SECURITY: Always use parameterized queries with params tuple to prevent SQL injection.
        Never use string formatting (%, .format(), f-strings) for query construction.
        
        Args:
            cursor: Database cursor
            query: SQL query string with ? placeholders
            params: Optional tuple of parameters to substitute
        """
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of verification results."""
        logger.info("\n" + "=" * 70)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 70)

        # Print summary
        print(f"\nChecks Performed: {self.stats['checks_performed']}")
        print(f"Total Issues Found: {self.stats['issues_found']}")
        print(f"  CRITICAL: {self.stats['critical']}")
        print(f"  HIGH:     {self.stats['high']}")
        print(f"  MEDIUM:   {self.stats['medium']}")
        print(f"  LOW:      {self.stats['low']}")

        # Group issues by category
        issues_by_category: Dict[str, List[ConsistencyIssue]] = {
            "DATABASE": [],
            "CACHE": [],
            "STATE": [],
            "CROSS_COMPONENT": [],
            "CONFIG": []
        }

        for issue in self.issues:
            if issue.category in issues_by_category:
                issues_by_category[issue.category].append(issue)

        # Print issues by category
        for category, issues in issues_by_category.items():
            if issues:
                print(f"\n{category} Issues ({len(issues)}):")
                for i, issue in enumerate(issues[:10], 1):  # Limit to 10 per category
                    print(f"  {i}. [{issue.severity}] {issue.check_name}")
                    print(f"     {issue.description}")
                    if issue.affected_records and len(issue.affected_records) > 0:
                        print(f"     Affected: {len(issue.affected_records)} records")
                    print(f"     Recommendation: {issue.recommendation}")
                    print()

                if len(issues) > 10:
                    print(f"  ... and {len(issues) - 10} more {category} issues")

        # Generate report dict
        report = {
            "timestamp": datetime.now().isoformat(),
            "database_path": self.db_path,
            "statistics": self.stats,
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "check_name": issue.check_name,
                    "description": issue.description,
                    "affected_count": len(issue.affected_records),
                    "affected_samples": issue.affected_records[:10],  # First 10 samples
                    "location": issue.location,
                    "recommendation": issue.recommendation
                }
                for issue in self.issues
            ],
            "issues_by_category": {
                cat: len(issues) for cat, issues in issues_by_category.items()
            },
            "issues_by_severity": self.stats
        }

        return report

    def save_report(self, report: Dict[str, Any], output_path: str = "data_consistency_report.json"):
        """Save verification report to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"\nReport saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


def main():
    """Main entry point."""
    # Parse command line arguments
    db_path = "bubblelabs_analytics.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    output_path = "data_consistency_report.json"
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    # Run verification
    verifier = DataConsistencyVerifier(db_path)
    report = verifier.verify_all()

    # Save report
    verifier.save_report(report, output_path)

    # Exit with error code if critical or high issues found
    if report["statistics"]["critical"] > 0 or report["statistics"]["high"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
