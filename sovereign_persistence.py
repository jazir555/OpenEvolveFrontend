"""
Sovereign-Grade Problem Decomposition System - Persistence Layer
Database schema and CRUD operations for all entities.
"""

import sqlite3
import json
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime
import logging

from sovereign_data_models import (
    ProblemDefinition, SubProblem, DecompositionPlan, SolutionAttempt,
    Pattern, TeamAssignment, Feedback, ValidationResult, QualityScores,
    DependencyGraph, ValidationCheckpoint
)
from migrations import MIGRATIONS


class SovereignDatabase:
    """Database manager for sovereign decomposition system"""
    
    def __init__(self, db_path: str = "sovereign_decomposition.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.init_database()
        self.apply_migrations()

    def apply_migrations(self):
        """Apply pending migrations to the database."""
        current_version = self.get_current_schema_version()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for version, statements in sorted(MIGRATIONS.items()):
                if version > current_version:
                    self.logger.info(f"Applying migration to version {version}")
                    for statement in statements:
                        cursor.execute(statement)
                    cursor.execute("UPDATE schema_version SET version = ?", (version,))
                    current_version = version

    def init_database(self):
        """Initialize database schema with optimizations"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            
            # Increase cache size for better performance
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            
            # Problems table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS problems (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    problem_type TEXT NOT NULL,
                    domain_context TEXT NOT NULL,
                    complexity_score TEXT NOT NULL,
                    constraints TEXT,
                    success_criteria TEXT,
                    stakeholders TEXT,
                    resources_available TEXT,
                    deadline TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Add indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_problems_type 
                ON problems(problem_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_problems_created 
                ON problems(created_at)
            """)
            
            # Sub-problems table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sub_problems (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    type TEXT NOT NULL,
                    complexity_score TEXT NOT NULL,
                    dependencies TEXT,
                    success_criteria TEXT,
                    validation_gauntlet TEXT,
                    assigned_team TEXT,
                    estimated_effort INTEGER,
                    priority INTEGER,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (parent_id) REFERENCES problems(id)
                )
            """)
            
            # Decomposition plans table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decomposition_plans (
                    id TEXT PRIMARY KEY,
                    problem_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    sub_problems TEXT,
                    dependency_graph TEXT,
                    validation_checkpoints TEXT,
                    quality_scores TEXT,
                    confidence_level REAL,
                    created_by TEXT,
                    approved_by TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (problem_id) REFERENCES problems(id)
                )
            """)
            
            # Solution attempts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS solution_attempts (
                    id TEXT PRIMARY KEY,
                    sub_problem_id TEXT NOT NULL,
                    approach TEXT NOT NULL,
                    solution_content TEXT NOT NULL,
                    team_id TEXT NOT NULL,
                    confidence_score REAL,
                    validation_results TEXT,
                    feedback TEXT,
                    status TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (sub_problem_id) REFERENCES sub_problems(id)
                )
            """)
            
            # Patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    problem_type TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    pattern_description TEXT NOT NULL,
                    success_rate REAL,
                    usage_count INTEGER,
                    avg_quality_score REAL,
                    applicable_domains TEXT,
                    created_at TEXT NOT NULL,
                    last_used TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Team assignments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS team_assignments (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    team TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    due_date TEXT,
                    status TEXT,
                    metadata TEXT
                )
            """)
            
            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    actionable INTEGER,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_problems_type ON problems(problem_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_subproblems_parent ON sub_problems(parent_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_subproblems_status ON sub_problems(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_plans_problem ON decomposition_plans(problem_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_plans_status ON decomposition_plans(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_subproblem ON solution_attempts(sub_problem_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(problem_type)")

            # Schema version table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            # Initialize schema version if not present
            cursor.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (0)")
    
    # ========================================================================
    # PROBLEM CRUD OPERATIONS
    # ========================================================================
    
    def create_problem(self, problem: ProblemDefinition) -> bool:
        """Create a new problem"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = problem.to_dict()
            cursor.execute("""
                INSERT INTO problems VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['id'], data['title'], data['description'], data['problem_type'],
                json.dumps(data['domain_context']), json.dumps(data['complexity_score']),
                json.dumps(data['constraints']), json.dumps(data['success_criteria']),
                json.dumps(data['stakeholders']), json.dumps(data['resources_available']),
                data.get('deadline'), data['created_at'], data['updated_at'],
                json.dumps(data['metadata'])
            ))
            return True
    
    def get_problem(self, problem_id: str) -> Optional[ProblemDefinition]:
        """Retrieve a problem by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM problems WHERE id = ?", (problem_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['domain_context'] = json.loads(data['domain_context'])
                data['complexity_score'] = json.loads(data['complexity_score'])
                data['constraints'] = json.loads(data['constraints'])
                data['success_criteria'] = json.loads(data['success_criteria'])
                data['stakeholders'] = json.loads(data['stakeholders'])
                data['resources_available'] = json.loads(data['resources_available'])
                data['metadata'] = json.loads(data['metadata'])
                return ProblemDefinition.from_dict(data)
            return None
    
    def update_problem(self, problem: ProblemDefinition) -> bool:
        """Update an existing problem"""
        problem.updated_at = datetime.now()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = problem.to_dict()
            cursor.execute("""
                UPDATE problems SET title=?, description=?, problem_type=?, domain_context=?,
                complexity_score=?, constraints=?, success_criteria=?, stakeholders=?,
                resources_available=?, deadline=?, updated_at=?, metadata=?
                WHERE id=?
            """, (
                data['title'], data['description'], data['problem_type'],
                json.dumps(data['domain_context']), json.dumps(data['complexity_score']),
                json.dumps(data['constraints']), json.dumps(data['success_criteria']),
                json.dumps(data['stakeholders']), json.dumps(data['resources_available']),
                data.get('deadline'), data['updated_at'], json.dumps(data['metadata']),
                data['id']
            ))
            return cursor.rowcount > 0
    
    def list_problems(self, problem_type: Optional[str] = None) -> List[ProblemDefinition]:
        """List all problems, optionally filtered by type"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if problem_type:
                cursor.execute("SELECT * FROM problems WHERE problem_type = ?", (problem_type,))
            else:
                cursor.execute("SELECT * FROM problems")
            
            problems = []
            for row in cursor.fetchall():
                data = dict(row)
                data['domain_context'] = json.loads(data['domain_context'])
                data['complexity_score'] = json.loads(data['complexity_score'])
                data['constraints'] = json.loads(data['constraints'])
                data['success_criteria'] = json.loads(data['success_criteria'])
                data['stakeholders'] = json.loads(data['stakeholders'])
                data['resources_available'] = json.loads(data['resources_available'])
                data['metadata'] = json.loads(data['metadata'])
                problems.append(ProblemDefinition.from_dict(data))
            return problems

    def create_subproblem(self, sub_problem: SubProblem) -> bool:
        """Create a new sub-problem"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = sub_problem.to_dict()
            cursor.execute("""
                INSERT INTO sub_problems VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['id'], data['parent_id'], data['title'], data['description'],
                data['type'], json.dumps(data['complexity_score']),
                json.dumps(data['dependencies']), json.dumps(data['success_criteria']),
                data['validation_gauntlet'], data.get('assigned_team'),
                data['estimated_effort'], data['priority'], data['status'],
                data['created_at'], data['updated_at'], json.dumps(data.get('metadata', {}))
            ))
            return True

    def get_subproblem(self, subproblem_id: str) -> Optional[SubProblem]:
        """Retrieve a sub-problem by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sub_problems WHERE id = ?", (subproblem_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['complexity_score'] = json.loads(data['complexity_score'])
                data['dependencies'] = json.loads(data['dependencies'])
                data['success_criteria'] = json.loads(data['success_criteria'])
                data['metadata'] = json.loads(data['metadata'])
                return SubProblem.from_dict(data)
            return None

    def update_subproblem(self, sub_problem: SubProblem) -> bool:
        """Update an existing sub-problem"""
        sub_problem.updated_at = datetime.now()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = sub_problem.to_dict()
            cursor.execute("""
                UPDATE sub_problems SET title=?, description=?, type=?, complexity_score=?,
                dependencies=?, success_criteria=?, validation_gauntlet=?, assigned_team=?,
                estimated_effort=?, priority=?, status=?, updated_at=?, metadata=?
                WHERE id=?
            """, (
                data['title'], data['description'], data['type'],
                json.dumps(data['complexity_score']),
                json.dumps(data['dependencies']), json.dumps(data['success_criteria']),
                data['validation_gauntlet'], data.get('assigned_team'),
                data['estimated_effort'], data['priority'], data['status'],
                data['updated_at'], json.dumps(data.get('metadata', {})),
                data['id']
            ))
            return cursor.rowcount > 0

    def list_subproblems(self, parent_id: str) -> List[SubProblem]:
        """List all sub-problems for a given parent problem"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sub_problems WHERE parent_id = ?", (parent_id,))
            sub_problems = []
            for row in cursor.fetchall():
                data = dict(row)
                data['complexity_score'] = json.loads(data['complexity_score'])
                data['dependencies'] = json.loads(data['dependencies'])
                data['success_criteria'] = json.loads(data['success_criteria'])
                data['metadata'] = json.loads(data['metadata'])
                sub_problems.append(SubProblem.from_dict(data))
            return sub_problems

    # ========================================================================
    # SOLUTION ATTEMPT CRUD OPERATIONS
    # ========================================================================

    def create_solution_attempt(self, attempt: SolutionAttempt) -> bool:
        """Create a new solution attempt"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = attempt.to_dict()
            cursor.execute("""
                INSERT INTO solution_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['id'], data['sub_problem_id'], data['approach'],
                data['solution_content'], data['team_id'], data['confidence_score'],
                json.dumps(data['validation_results']), json.dumps(data['feedback']),
                data['status'], data['created_at'], json.dumps(data['metadata'])
            ))
            return True

    def get_solution_attempt(self, attempt_id: str) -> Optional[SolutionAttempt]:
        """Retrieve a solution attempt by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM solution_attempts WHERE id = ?", (attempt_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['validation_results'] = json.loads(data['validation_results'])
                data['feedback'] = json.loads(data['feedback'])
                data['metadata'] = json.loads(data['metadata'])
                return SolutionAttempt.from_dict(data)
            return None

    def update_solution_attempt(self, attempt: SolutionAttempt) -> bool:
        """Update an existing solution attempt"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = attempt.to_dict()
            cursor.execute("""
                UPDATE solution_attempts SET approach=?, solution_content=?, confidence_score=?,
                validation_results=?, feedback=?, status=?, metadata=?
                WHERE id=?
            """, (
                data['approach'], data['solution_content'], data['confidence_score'],
                json.dumps(data['validation_results']), json.dumps(data['feedback']),
                data['status'], json.dumps(data['metadata']), data['id']
            ))
            return cursor.rowcount > 0

    def list_solution_attempts(self, sub_problem_id: str) -> List[SolutionAttempt]:
        """List all solution attempts for a given sub-problem"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM solution_attempts WHERE sub_problem_id = ?", (sub_problem_id,))
            attempts = []
            for row in cursor.fetchall():
                data = dict(row)
                data['validation_results'] = json.loads(data['validation_results'])
                data['feedback'] = json.loads(data['feedback'])
                data['metadata'] = json.loads(data['metadata'])
                attempts.append(SolutionAttempt.from_dict(data))
            return attempts

    # ========================================================================
    # TEAM ASSIGNMENT CRUD OPERATIONS
    # ========================================================================

    def create_team_assignment(self, assignment: TeamAssignment) -> bool:
        """Create a new team assignment"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = assignment.to_dict()
            cursor.execute("""
                INSERT INTO team_assignments VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                data['id'], data['task_id'], data['team'], data['assigned_at'],
                data.get('due_date'), data['status'], json.dumps(data['metadata'])
            ))
            return True

    def get_team_assignment(self, assignment_id: str) -> Optional[TeamAssignment]:
        """Retrieve a team assignment by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM team_assignments WHERE id = ?", (assignment_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'])
                return TeamAssignment.from_dict(data)
            return None

    def update_team_assignment(self, assignment: TeamAssignment) -> bool:
        """Update an existing team assignment"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = assignment.to_dict()
            cursor.execute("""
                UPDATE team_assignments SET due_date=?, status=?, metadata=?
                WHERE id=?
            """, (
                data.get('due_date'), data['status'], json.dumps(data['metadata']), data['id']
            ))
            return cursor.rowcount > 0

    def list_team_assignments(self, task_id: str) -> List[TeamAssignment]:
        """List all team assignments for a given task"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM team_assignments WHERE task_id = ?", (task_id,))
            assignments = []
            for row in cursor.fetchall():
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'])
                assignments.append(TeamAssignment.from_dict(data))
            return assignments

    # ========================================================================
    # FEEDBACK CRUD OPERATIONS
    # ========================================================================

    def create_feedback(self, feedback: Feedback) -> bool:
        """Create a new feedback"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = feedback.to_dict()
            cursor.execute("""
                INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['id'], data['source'], data['feedback_type'], data['content'],
                data['severity'], data['actionable'], data['timestamp'],
                json.dumps(data['metadata'])
            ))
            return True

    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """Retrieve a feedback by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'])
                return Feedback.from_dict(data)
            return None

    def list_feedback(self, source: str) -> List[Feedback]:
        """List all feedback for a given source"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feedback WHERE source = ?", (source,))
            feedback_list = []
            for row in cursor.fetchall():
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'])
                feedback_list.append(Feedback.from_dict(data))
            return feedback_list

    # ========================================================================
    # DELETE OPERATIONS
    # ========================================================================

    def delete_problem(self, problem_id: str) -> bool:
        """Delete a problem and all its related data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # First, delete related data in other tables
            cursor.execute("DELETE FROM decomposition_plans WHERE problem_id = ?", (problem_id,))
            cursor.execute("DELETE FROM sub_problems WHERE parent_id = ?", (problem_id,))
            # Finally, delete the problem itself
            cursor.execute("DELETE FROM problems WHERE id = ?", (problem_id,))
            return cursor.rowcount > 0

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a decomposition plan"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM decomposition_plans WHERE id = ?", (plan_id,))
            return cursor.rowcount > 0
    
    # ========================================================================
    # DECOMPOSITION PLAN CRUD OPERATIONS
    # ========================================================================
    
    def create_plan(self, plan: DecompositionPlan) -> bool:
        """Create a new decomposition plan"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = plan.to_dict()
            cursor.execute("""
                INSERT INTO decomposition_plans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['id'], data['problem_id'], data['strategy'],
                json.dumps(data['sub_problems']),
                json.dumps(data.get('dependency_graph')),
                json.dumps(data.get('validation_checkpoints', [])),
                json.dumps(data.get('quality_scores')),
                data['confidence_level'], data['created_by'], data.get('approved_by'),
                data['status'], data['created_at'], data['updated_at'],
                json.dumps(data['metadata'])
            ))
            return True
    
    def get_plan(self, plan_id: str) -> Optional[DecompositionPlan]:
        """Retrieve a decomposition plan by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM decomposition_plans WHERE id = ?", (plan_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['sub_problems'] = json.loads(data['sub_problems'])
                data['dependency_graph'] = json.loads(data['dependency_graph']) if data['dependency_graph'] else None
                data['validation_checkpoints'] = json.loads(data['validation_checkpoints']) if data['validation_checkpoints'] else []
                data['quality_scores'] = json.loads(data['quality_scores']) if data['quality_scores'] else None
                data['metadata'] = json.loads(data['metadata'])
                return DecompositionPlan.from_dict(data)
            return None
    
    def update_plan(self, plan: DecompositionPlan) -> bool:
        """Update an existing decomposition plan"""
        plan.updated_at = datetime.now()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = plan.to_dict()
            cursor.execute("""
                UPDATE decomposition_plans SET strategy=?, sub_problems=?, dependency_graph=?,
                validation_checkpoints=?, quality_scores=?, confidence_level=?, approved_by=?,
                status=?, updated_at=?, metadata=?
                WHERE id=?
            """, (
                data['strategy'], json.dumps(data['sub_problems']),
                json.dumps(data.get('dependency_graph')),
                json.dumps(data.get('validation_checkpoints', [])),
                json.dumps(data.get('quality_scores')),
                data['confidence_level'], data.get('approved_by'),
                data['status'], data['updated_at'], json.dumps(data['metadata']),
                data['id']
            ))
            return cursor.rowcount > 0
    
    # ========================================================================
    # PATTERN CRUD OPERATIONS
    # ========================================================================
    
    def create_pattern(self, pattern: Pattern) -> bool:
        """Create a new pattern"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = pattern.to_dict()
            cursor.execute("""
                INSERT INTO patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['id'], data['problem_type'], data['strategy'],
                data['pattern_description'], data['success_rate'],
                data['usage_count'], data['avg_quality_score'],
                json.dumps(data['applicable_domains']),
                data['created_at'], data['last_used'], json.dumps(data['metadata'])
            ))
            return True
    
    def get_patterns_by_type(self, problem_type: str) -> List[Pattern]:
        """Retrieve patterns by problem type"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patterns WHERE problem_type = ? ORDER BY success_rate DESC", (problem_type,))
            patterns = []
            for row in cursor.fetchall():
                data = dict(row)
                data['applicable_domains'] = json.loads(data['applicable_domains'])
                data['metadata'] = json.loads(data['metadata'])
                patterns.append(Pattern.from_dict(data))
            return patterns
    
    def update_pattern(self, pattern: Pattern) -> bool:
        """Update an existing pattern"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            data = pattern.to_dict()
            cursor.execute("""
                UPDATE patterns SET success_rate=?, usage_count=?, avg_quality_score=?,
                last_used=?, metadata=?
                WHERE id=?
            """, (
                data['success_rate'], data['usage_count'], data['avg_quality_score'],
                data['last_used'], json.dumps(data['metadata']), data['id']
            ))
            return cursor.rowcount > 0

    # ========================================================================
    # Database Optimization Methods
    # ========================================================================
    
    def optimize_database(self) -> Dict[str, Any]:
        """
        Optimize database performance.
        
        Returns:
            Dictionary with optimization results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Analyze tables for query optimization
            cursor.execute("ANALYZE")
            
            # Vacuum to reclaim space and defragment
            cursor.execute("VACUUM")
            
            # Get database stats
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            
            db_size = page_count * page_size
            
            return {
                'optimized': True,
                'db_size_bytes': db_size,
                'db_size_mb': db_size / (1024 * 1024),
                'page_count': page_count,
                'page_size': page_size
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count records in each table
            for table in ['problems', 'sub_problems', 'decomposition_plans', 'patterns']:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Get database size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            stats['db_size_mb'] = (page_count * page_size) / (1024 * 1024)
            
            return stats

    def get_current_schema_version(self) -> int:
        """Get the current schema version from the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
            cursor.execute("SELECT version FROM schema_version")
            result = cursor.fetchone()
            return result[0] if result else 0
    
    def batch_insert_subproblems(self, sub_problems: List[SubProblem]) -> int:
        """
        Batch insert sub-problems for better performance.
        
        Args:
            sub_problems: List of sub-problems to insert
            
        Returns:
            Number of sub-problems inserted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare batch data
            batch_data = []
            for sp in sub_problems:
                data = sp.to_dict()
                batch_data.append((
                    data['id'], data['parent_id'], data['title'], data['description'],
                    data['type'], json.dumps(data['complexity_score']),
                    json.dumps(data['dependencies']), json.dumps(data['success_criteria']),
                    data['validation_gauntlet'], data.get('assigned_team'),
                    data['estimated_effort'], data['priority'], data['status'],
                    data['created_at'], data['updated_at'], json.dumps(data.get('metadata', {}))
                ))
            
            # Batch insert
            cursor.executemany("""
                INSERT INTO sub_problems (
                    id, parent_id, title, description, type, complexity_score,
                    dependencies, success_criteria, validation_gauntlet, assigned_team,
                    estimated_effort, priority, status, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch_data)
            
            return len(batch_data)
    
    def create_indexes(self) -> None:
        """Create additional indexes for performance."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Sub-problems indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_subproblems_parent 
                ON sub_problems(parent_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_subproblems_status 
                ON sub_problems(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_subproblems_priority 
                ON sub_problems(priority DESC)
            """)
            
            # Decomposition plans indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_plans_problem 
                ON decomposition_plans(problem_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_plans_strategy 
                ON decomposition_plans(strategy)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_plans_status 
                ON decomposition_plans(status)
            """)
            
            # Patterns indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_type 
                ON patterns(problem_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_strategy 
                ON patterns(strategy)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_success 
                ON patterns(success_rate DESC)
            """)

            # Schema version table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except (sqlite3.Error, IOError, OSError):
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_plans(self, status: Optional[str] = None) -> List[DecompositionPlan]:
        """List all decomposition plans, optionally filtered by status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("""
                    SELECT * FROM decomposition_plans 
                    WHERE status = ?
                    ORDER BY created_at DESC
                """, (status,))
            else:
                cursor.execute("""
                    SELECT * FROM decomposition_plans 
                    ORDER BY created_at DESC
                """)
            
            plans = []
            for row in cursor.fetchall():
                data = dict(row)
                data['sub_problems'] = json.loads(data['sub_problems'])
                data['dependency_graph'] = json.loads(data['dependency_graph']) if data['dependency_graph'] else None
                data['validation_checkpoints'] = json.loads(data['validation_checkpoints']) if data['validation_checkpoints'] else []
                data['quality_scores'] = json.loads(data['quality_scores']) if data['quality_scores'] else None
                data['metadata'] = json.loads(data['metadata'])
                plans.append(DecompositionPlan.from_dict(data))
            return plans
