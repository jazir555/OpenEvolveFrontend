"""
Sovereign-Grade Problem Decomposition System - Database Layer

This module provides SQLite database schema, migrations, and CRUD operations
for all sovereign decomposition data models.
"""

import sqlite3
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import logging

from sovereign_data_models import (
    ProblemDefinition, SubProblem, DecompositionPlan,
    SolutionAttempt, Pattern, TeamAssignment, Feedback,
    ValidationResult, ValidationCheckpoint, QualityScores,
    DependencyGraph, generate_id
)

logger = logging.getLogger(__name__)


class SovereignDatabase:
    """Database manager for sovereign decomposition system."""
    
    def __init__(self, db_path: str = "sovereign_decomposition.db"):
        """Initialize database connection and create schema if needed."""
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_schema()
        self._create_indexes()
    
    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to database: {self.db_path}")
    
    def _create_schema(self):
        """Create database schema for all tables."""
        cursor = self.conn.cursor()
        
        # Problems table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS problems (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                domain_context TEXT NOT NULL,
                complexity_score TEXT NOT NULL,
                constraints TEXT NOT NULL,
                success_criteria TEXT NOT NULL,
                stakeholders TEXT NOT NULL,
                resources_available TEXT NOT NULL,
                deadline TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
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
                dependencies TEXT NOT NULL,
                success_criteria TEXT NOT NULL,
                validation_gauntlet TEXT NOT NULL,
                assigned_team TEXT,
                estimated_effort INTEGER DEFAULT 0,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                solution_attempts TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES problems(id)
            )
        """)
        
        # Decomposition plans table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decomposition_plans (
                id TEXT PRIMARY KEY,
                problem_id TEXT NOT NULL,
                strategy TEXT NOT NULL,
                sub_problems TEXT NOT NULL,
                dependency_graph TEXT NOT NULL,
                validation_checkpoints TEXT NOT NULL,
                quality_scores TEXT NOT NULL,
                confidence_level REAL NOT NULL,
                created_by TEXT NOT NULL,
                approved_by TEXT,
                status TEXT DEFAULT 'draft',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL,
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
                confidence_score REAL NOT NULL,
                validation_results TEXT NOT NULL,
                feedback TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY (sub_problem_id) REFERENCES sub_problems(id)
            )
        """)
        
        # Patterns table (for knowledge learning)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                problem_type TEXT NOT NULL,
                strategy TEXT NOT NULL,
                pattern_description TEXT NOT NULL,
                success_rate REAL NOT NULL,
                usage_count INTEGER NOT NULL,
                avg_quality_score REAL NOT NULL,
                applicable_domains TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used TEXT NOT NULL,
                metadata TEXT NOT NULL
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
                status TEXT DEFAULT 'assigned',
                metadata TEXT NOT NULL
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
                actionable INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        
        self.conn.commit()
        logger.info("Database schema created successfully")
    
    def _create_indexes(self):
        """Create indexes for performance optimization."""
        cursor = self.conn.cursor()
        
        # Index on problem type for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_problems_type 
            ON problems(problem_type)
        """)
        
        # Index on sub-problem status for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_subproblems_status 
            ON sub_problems(status)
        """)
        
        # Index on sub-problem parent for lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_subproblems_parent 
            ON sub_problems(parent_id)
        """)
        
        # Index on plan status for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plans_status 
            ON decomposition_plans(status)
        """)
        
        # Index on plan problem_id for lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plans_problem 
            ON decomposition_plans(problem_id)
        """)
        
        # Index on pattern problem_type for matching
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_type 
            ON patterns(problem_type)
        """)
        
        # Index on team assignments team for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_assignments_team 
            ON team_assignments(team)
        """)
        
        self.conn.commit()
        logger.info("Database indexes created successfully")
    
    # ========================================================================
    # PROBLEM CRUD OPERATIONS
    # ========================================================================
    
    def create_problem(self, problem: ProblemDefinition) -> str:
        """Create a new problem in the database."""
        cursor = self.conn.cursor()
        data = problem.to_dict()
        
        cursor.execute("""
            INSERT INTO problems (
                id, title, description, problem_type, domain_context,
                complexity_score, constraints, success_criteria, stakeholders,
                resources_available, deadline, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['id'], data['title'], data['description'], data['problem_type'],
            json.dumps(data['domain_context']), json.dumps(data['complexity_score']),
            json.dumps(data['constraints']), json.dumps(data['success_criteria']),
            json.dumps(data['stakeholders']), json.dumps(data['resources_available']),
            data['deadline'], data['created_at'], data['updated_at'],
            json.dumps(data['metadata'])
        ))
        
        self.conn.commit()
        logger.info(f"Created problem: {problem.id}")
        return problem.id
    
    def get_problem(self, problem_id: str) -> Optional[ProblemDefinition]:
        """Retrieve a problem by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM problems WHERE id = ?", (problem_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        data = dict(row)
        # Deserialize JSON fields
        data['domain_context'] = json.loads(data['domain_context'])
        data['complexity_score'] = json.loads(data['complexity_score'])
        data['constraints'] = json.loads(data['constraints'])
        data['success_criteria'] = json.loads(data['success_criteria'])
        data['stakeholders'] = json.loads(data['stakeholders'])
        data['resources_available'] = json.loads(data['resources_available'])
        data['metadata'] = json.loads(data['metadata'])
        
        return ProblemDefinition.from_dict(data)
    
    def update_problem(self, problem: ProblemDefinition) -> bool:
        """Update an existing problem."""
        cursor = self.conn.cursor()
        data = problem.to_dict()
        data['updated_at'] = datetime.now().isoformat()
        
        cursor.execute("""
            UPDATE problems SET
                title = ?, description = ?, problem_type = ?,
                domain_context = ?, complexity_score = ?,
                constraints = ?, success_criteria = ?,
                stakeholders = ?, resources_available = ?,
                deadline = ?, updated_at = ?, metadata = ?
            WHERE id = ?
        """, (
            data['title'], data['description'], data['problem_type'],
            json.dumps(data['domain_context']), json.dumps(data['complexity_score']),
            json.dumps(data['constraints']), json.dumps(data['success_criteria']),
            json.dumps(data['stakeholders']), json.dumps(data['resources_available']),
            data['deadline'], data['updated_at'], json.dumps(data['metadata']),
            data['id']
        ))
        
        self.conn.commit()
        logger.info(f"Updated problem: {problem.id}")
        return cursor.rowcount > 0
    
    def delete_problem(self, problem_id: str) -> bool:
        """Delete a problem and all related data."""
        cursor = self.conn.cursor()
        
        # Delete related sub-problems
        cursor.execute("DELETE FROM sub_problems WHERE parent_id = ?", (problem_id,))
        
        # Delete related plans
        cursor.execute("DELETE FROM decomposition_plans WHERE problem_id = ?", (problem_id,))
        
        # Delete the problem
        cursor.execute("DELETE FROM problems WHERE id = ?", (problem_id,))
        
        self.conn.commit()
        logger.info(f"Deleted problem: {problem_id}")
        return cursor.rowcount > 0
    
    def list_problems(self, problem_type: Optional[str] = None, 
                     limit: int = 100, offset: int = 0) -> List[ProblemDefinition]:
        """List problems with optional filtering."""
        cursor = self.conn.cursor()
        
        if problem_type:
            cursor.execute("""
                SELECT * FROM problems 
                WHERE problem_type = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (problem_type, limit, offset))
        else:
            cursor.execute("""
                SELECT * FROM problems 
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
        
        problems = []
        for row in cursor.fetchall():
            data = dict(row)
            # Deserialize JSON fields
            data['domain_context'] = json.loads(data['domain_context'])
            data['complexity_score'] = json.loads(data['complexity_score'])
            data['constraints'] = json.loads(data['constraints'])
            data['success_criteria'] = json.loads(data['success_criteria'])
            data['stakeholders'] = json.loads(data['stakeholders'])
            data['resources_available'] = json.loads(data['resources_available'])
            data['metadata'] = json.loads(data['metadata'])
            problems.append(ProblemDefinition.from_dict(data))
        
        return problems
    
    # ========================================================================
    # SUB-PROBLEM CRUD OPERATIONS
    # ========================================================================
    
    def create_sub_problem(self, sub_problem: SubProblem) -> str:
        """Create a new sub-problem in the database."""
        cursor = self.conn.cursor()
        data = sub_problem.to_dict()
        
        cursor.execute("""
            INSERT INTO sub_problems (
                id, parent_id, title, description, type, complexity_score,
                dependencies, success_criteria, validation_gauntlet,
                assigned_team, estimated_effort, priority, status,
                solution_attempts, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['id'], data['parent_id'], data['title'], data['description'],
            data['type'], json.dumps(data['complexity_score']),
            json.dumps(data['dependencies']), json.dumps(data['success_criteria']),
            data['validation_gauntlet'], data['assigned_team'],
            data['estimated_effort'], data['priority'], data['status'],
            json.dumps(data['solution_attempts']), data['created_at'],
            data['updated_at'], json.dumps(data['metadata'])
        ))
        
        self.conn.commit()
        logger.info(f"Created sub-problem: {sub_problem.id}")
        return sub_problem.id
    
    def get_sub_problem(self, sub_problem_id: str) -> Optional[SubProblem]:
        """Retrieve a sub-problem by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sub_problems WHERE id = ?", (sub_problem_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        data = dict(row)
        # Deserialize JSON fields
        data['complexity_score'] = json.loads(data['complexity_score'])
        data['dependencies'] = json.loads(data['dependencies'])
        data['success_criteria'] = json.loads(data['success_criteria'])
        data['solution_attempts'] = json.loads(data['solution_attempts'])
        data['metadata'] = json.loads(data['metadata'])
        
        return SubProblem.from_dict(data)
    
    def update_sub_problem(self, sub_problem: SubProblem) -> bool:
        """Update an existing sub-problem."""
        cursor = self.conn.cursor()
        data = sub_problem.to_dict()
        data['updated_at'] = datetime.now().isoformat()
        
        cursor.execute("""
            UPDATE sub_problems SET
                title = ?, description = ?, type = ?,
                complexity_score = ?, dependencies = ?,
                success_criteria = ?, validation_gauntlet = ?,
                assigned_team = ?, estimated_effort = ?,
                priority = ?, status = ?, solution_attempts = ?,
                updated_at = ?, metadata = ?
            WHERE id = ?
        """, (
            data['title'], data['description'], data['type'],
            json.dumps(data['complexity_score']), json.dumps(data['dependencies']),
            json.dumps(data['success_criteria']), data['validation_gauntlet'],
            data['assigned_team'], data['estimated_effort'], data['priority'],
            data['status'], json.dumps(data['solution_attempts']),
            data['updated_at'], json.dumps(data['metadata']), data['id']
        ))
        
        self.conn.commit()
        logger.info(f"Updated sub-problem: {sub_problem.id}")
        return cursor.rowcount > 0
    
    def list_sub_problems_by_parent(self, parent_id: str) -> List[SubProblem]:
        """List all sub-problems for a given parent problem."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM sub_problems 
            WHERE parent_id = ?
            ORDER BY priority DESC, created_at ASC
        """, (parent_id,))
        
        sub_problems = []
        for row in cursor.fetchall():
            data = dict(row)
            data['complexity_score'] = json.loads(data['complexity_score'])
            data['dependencies'] = json.loads(data['dependencies'])
            data['success_criteria'] = json.loads(data['success_criteria'])
            data['solution_attempts'] = json.loads(data['solution_attempts'])
            data['metadata'] = json.loads(data['metadata'])
            sub_problems.append(SubProblem.from_dict(data))
        
        return sub_problems
    
    # ========================================================================
    # DECOMPOSITION PLAN CRUD OPERATIONS
    # ========================================================================
    
    def create_decomposition_plan(self, plan: DecompositionPlan) -> str:
        """Create a new decomposition plan in the database."""
        cursor = self.conn.cursor()
        data = plan.to_dict()
        
        cursor.execute("""
            INSERT INTO decomposition_plans (
                id, problem_id, strategy, sub_problems, dependency_graph,
                validation_checkpoints, quality_scores, confidence_level,
                created_by, approved_by, status, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['id'], data['problem_id'], data['strategy'],
            json.dumps(data['sub_problems']), json.dumps(data['dependency_graph']),
            json.dumps(data['validation_checkpoints']), json.dumps(data['quality_scores']),
            data['confidence_level'], data['created_by'], data['approved_by'],
            data['status'], data['created_at'], data['updated_at'],
            json.dumps(data['metadata'])
        ))
        
        self.conn.commit()
        logger.info(f"Created decomposition plan: {plan.id}")
        return plan.id
    
    def get_decomposition_plan(self, plan_id: str) -> Optional[DecompositionPlan]:
        """Retrieve a decomposition plan by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM decomposition_plans WHERE id = ?", (plan_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        data = dict(row)
        # Deserialize JSON fields
        data['sub_problems'] = json.loads(data['sub_problems'])
        data['dependency_graph'] = json.loads(data['dependency_graph'])
        data['validation_checkpoints'] = json.loads(data['validation_checkpoints'])
        data['quality_scores'] = json.loads(data['quality_scores'])
        data['metadata'] = json.loads(data['metadata'])
        
        return DecompositionPlan.from_dict(data)
    
    def list_plans_by_problem(self, problem_id: str) -> List[DecompositionPlan]:
        """List all decomposition plans for a given problem."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM decomposition_plans 
            WHERE problem_id = ?
            ORDER BY created_at DESC
        """, (problem_id,))
        
        plans = []
        for row in cursor.fetchall():
            data = dict(row)
            data['sub_problems'] = json.loads(data['sub_problems'])
            data['dependency_graph'] = json.loads(data['dependency_graph'])
            data['validation_checkpoints'] = json.loads(data['validation_checkpoints'])
            data['quality_scores'] = json.loads(data['quality_scores'])
            data['metadata'] = json.loads(data['metadata'])
            plans.append(DecompositionPlan.from_dict(data))
        
        return plans
    
    # ========================================================================
    # PATTERN CRUD OPERATIONS (for knowledge learning)
    # ========================================================================
    
    def create_pattern(self, pattern: Pattern) -> str:
        """Create a new pattern in the knowledge base."""
        cursor = self.conn.cursor()
        data = pattern.to_dict()
        
        cursor.execute("""
            INSERT INTO patterns (
                id, problem_type, strategy, pattern_description,
                success_rate, usage_count, avg_quality_score,
                applicable_domains, created_at, last_used, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['id'], data['problem_type'], data['strategy'],
            data['pattern_description'], data['success_rate'],
            data['usage_count'], data['avg_quality_score'],
            json.dumps(data['applicable_domains']), data['created_at'],
            data['last_used'], json.dumps(data['metadata'])
        ))
        
        self.conn.commit()
        logger.info(f"Created pattern: {pattern.id}")
        return pattern.id
    
    def get_patterns_by_type(self, problem_type: str, limit: int = 10) -> List[Pattern]:
        """Retrieve patterns for a specific problem type."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM patterns 
            WHERE problem_type = ?
            ORDER BY success_rate DESC, usage_count DESC
            LIMIT ?
        """, (problem_type, limit))
        
        patterns = []
        for row in cursor.fetchall():
            data = dict(row)
            data['applicable_domains'] = json.loads(data['applicable_domains'])
            data['metadata'] = json.loads(data['metadata'])
            patterns.append(Pattern.from_dict(data))
        
        return patterns
    
    def update_pattern_usage(self, pattern_id: str, success: bool, quality_score: float):
        """Update pattern usage statistics."""
        cursor = self.conn.cursor()
        
        # Get current pattern
        cursor.execute("SELECT * FROM patterns WHERE id = ?", (pattern_id,))
        row = cursor.fetchone()
        if not row:
            return False
        
        data = dict(row)
        usage_count = data['usage_count'] + 1
        
        # Update success rate (running average)
        current_success_rate = data['success_rate']
        new_success_rate = ((current_success_rate * data['usage_count']) + (1.0 if success else 0.0)) / usage_count
        
        # Update quality score (running average)
        current_avg_quality = data['avg_quality_score']
        new_avg_quality = ((current_avg_quality * data['usage_count']) + quality_score) / usage_count
        
        cursor.execute("""
            UPDATE patterns SET
                usage_count = ?,
                success_rate = ?,
                avg_quality_score = ?,
                last_used = ?
            WHERE id = ?
        """, (usage_count, new_success_rate, new_avg_quality, datetime.now().isoformat(), pattern_id))
        
        self.conn.commit()
        logger.info(f"Updated pattern usage: {pattern_id}")
        return True
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
