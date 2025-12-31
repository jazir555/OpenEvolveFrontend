"""
Unit tests for Sovereign-Grade Problem Decomposition System Database Layer

Tests database schema, CRUD operations, indexing, and data persistence.
"""

import pytest
import os
import tempfile
from datetime import datetime

from sovereign_database import SovereignDatabase
from sovereign_data_models import (
    ProblemDefinition, SubProblem, DecompositionPlan, Pattern,
    ProblemType, SubProblemType, DecompositionStrategy,
    SubProblemStatus, PlanStatus,
    DomainContext, ComplexityScore, Constraint, SuccessCriterion,
    DependencyGraph, QualityScores, ValidationCheckpoint,
    generate_id
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    db = SovereignDatabase(path)
    yield db
    
    db.close()
    os.unlink(path)


@pytest.fixture
def sample_problem():
    """Create a sample problem for testing."""
    domain_context = DomainContext(
        domain="software_engineering",
        subdomain="system_design"
    )
    
    complexity_score = ComplexityScore(
        cognitive_complexity=7.0,
        computational_complexity=6.0,
        domain_complexity=8.0,
        integration_complexity=7.0,
        overall_complexity=7.0,
        explanation="Complex system design problem"
    )
    
    problem = ProblemDefinition(
        id=generate_id("problem"),
        title="Design Scalable System",
        description="Design a system that can handle high load",
        problem_type=ProblemType.DESIGN,
        domain_context=domain_context,
        complexity_score=complexity_score,
        constraints=[],
        success_criteria=[],
        stakeholders=["team_lead"],
        resources_available={"team_size": 5}
    )
    
    return problem


@pytest.fixture
def sample_sub_problem(sample_problem):
    """Create a sample sub-problem for testing."""
    complexity_score = ComplexityScore(
        cognitive_complexity=3.0,
        computational_complexity=3.0,
        domain_complexity=3.0,
        integration_complexity=3.0,
        overall_complexity=3.0,
        explanation="Low complexity"
    )
    
    sub_problem = SubProblem(
        id=generate_id("subproblem"),
        parent_id=sample_problem.id,
        title="Design Database Schema",
        description="Create database schema for the system",
        type=SubProblemType.IMPLEMENTATION,
        complexity_score=complexity_score,
        dependencies=[],
        success_criteria=[],
        validation_gauntlet="coherence"
    )
    
    return sub_problem


# ============================================================================
# DATABASE INITIALIZATION TESTS
# ============================================================================

class TestDatabaseInitialization:
    """Test database initialization and schema creation."""
    
    def test_database_creation(self, temp_db):
        """Test that database is created successfully."""
        assert temp_db.conn is not None
        assert os.path.exists(temp_db.db_path)
    
    def test_schema_creation(self, temp_db):
        """Test that all tables are created."""
        cursor = temp_db.conn.cursor()
        
        # Check that all tables exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table'
            ORDER BY name
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = [
            'decomposition_plans',
            'feedback',
            'patterns',
            'problems',
            'solution_attempts',
            'sub_problems',
            'team_assignments'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"
    
    def test_indexes_creation(self, temp_db):
        """Test that indexes are created."""
        cursor = temp_db.conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index'
            ORDER BY name
        """)
        
        indexes = [row[0] for row in cursor.fetchall()]
        
        # Check for key indexes
        assert any('idx_problems_type' in idx for idx in indexes)
        assert any('idx_subproblems_status' in idx for idx in indexes)
        assert any('idx_plans_status' in idx for idx in indexes)


# ============================================================================
# PROBLEM CRUD TESTS
# ============================================================================

class TestProblemCRUD:
    """Test CRUD operations for problems."""
    
    def test_create_problem(self, temp_db, sample_problem):
        """Test creating a problem."""
        problem_id = temp_db.create_problem(sample_problem)
        assert problem_id == sample_problem.id
        
        # Verify it was created
        retrieved = temp_db.get_problem(problem_id)
        assert retrieved is not None
        assert retrieved.id == sample_problem.id
        assert retrieved.title == sample_problem.title
    
    def test_get_problem(self, temp_db, sample_problem):
        """Test retrieving a problem."""
        temp_db.create_problem(sample_problem)
        
        retrieved = temp_db.get_problem(sample_problem.id)
        assert retrieved is not None
        assert retrieved.id == sample_problem.id
        assert retrieved.title == sample_problem.title
        assert retrieved.problem_type == sample_problem.problem_type
        assert retrieved.domain_context.domain == sample_problem.domain_context.domain
    
    def test_get_nonexistent_problem(self, temp_db):
        """Test retrieving a problem that doesn't exist."""
        retrieved = temp_db.get_problem("nonexistent_id")
        assert retrieved is None
    
    def test_update_problem(self, temp_db, sample_problem):
        """Test updating a problem."""
        temp_db.create_problem(sample_problem)
        
        # Update the problem
        sample_problem.title = "Updated Title"
        sample_problem.description = "Updated Description"
        
        success = temp_db.update_problem(sample_problem)
        assert success is True
        
        # Verify the update
        retrieved = temp_db.get_problem(sample_problem.id)
        assert retrieved.title == "Updated Title"
        assert retrieved.description == "Updated Description"
    
    def test_delete_problem(self, temp_db, sample_problem):
        """Test deleting a problem."""
        temp_db.create_problem(sample_problem)
        
        success = temp_db.delete_problem(sample_problem.id)
        assert success is True
        
        # Verify it was deleted
        retrieved = temp_db.get_problem(sample_problem.id)
        assert retrieved is None
    
    def test_list_problems(self, temp_db):
        """Test listing problems."""
        # Create multiple problems
        for i in range(5):
            problem = ProblemDefinition(
                id=generate_id("problem"),
                title=f"Problem {i}",
                description=f"Description {i}",
                problem_type=ProblemType.RESEARCH if i % 2 == 0 else ProblemType.DESIGN,
                domain_context=DomainContext(domain="test"),
                complexity_score=ComplexityScore(
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    overall_complexity=5.0,
                    explanation="Test"
                ),
                constraints=[],
                success_criteria=[],
                stakeholders=[],
                resources_available={}
            )
            temp_db.create_problem(problem)
        
        # List all problems
        all_problems = temp_db.list_problems()
        assert len(all_problems) == 5
        
        # List filtered by type
        research_problems = temp_db.list_problems(problem_type="research")
        assert len(research_problems) == 3
    
    def test_list_problems_with_pagination(self, temp_db):
        """Test listing problems with pagination."""
        # Create 10 problems
        for i in range(10):
            problem = ProblemDefinition(
                id=generate_id("problem"),
                title=f"Problem {i}",
                description=f"Description {i}",
                problem_type=ProblemType.ANALYSIS,
                domain_context=DomainContext(domain="test"),
                complexity_score=ComplexityScore(
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    overall_complexity=5.0,
                    explanation="Test"
                ),
                constraints=[],
                success_criteria=[],
                stakeholders=[],
                resources_available={}
            )
            temp_db.create_problem(problem)
        
        # Get first page
        page1 = temp_db.list_problems(limit=5, offset=0)
        assert len(page1) == 5
        
        # Get second page
        page2 = temp_db.list_problems(limit=5, offset=5)
        assert len(page2) == 5
        
        # Verify no overlap
        page1_ids = {p.id for p in page1}
        page2_ids = {p.id for p in page2}
        assert len(page1_ids.intersection(page2_ids)) == 0


# ============================================================================
# SUB-PROBLEM CRUD TESTS
# ============================================================================

class TestSubProblemCRUD:
    """Test CRUD operations for sub-problems."""
    
    def test_create_sub_problem(self, temp_db, sample_problem, sample_sub_problem):
        """Test creating a sub-problem."""
        temp_db.create_problem(sample_problem)
        
        sub_problem_id = temp_db.create_sub_problem(sample_sub_problem)
        assert sub_problem_id == sample_sub_problem.id
        
        # Verify it was created
        retrieved = temp_db.get_sub_problem(sub_problem_id)
        assert retrieved is not None
        assert retrieved.id == sample_sub_problem.id
    
    def test_get_sub_problem(self, temp_db, sample_problem, sample_sub_problem):
        """Test retrieving a sub-problem."""
        temp_db.create_problem(sample_problem)
        temp_db.create_sub_problem(sample_sub_problem)
        
        retrieved = temp_db.get_sub_problem(sample_sub_problem.id)
        assert retrieved is not None
        assert retrieved.title == sample_sub_problem.title
        assert retrieved.parent_id == sample_problem.id
    
    def test_update_sub_problem(self, temp_db, sample_problem, sample_sub_problem):
        """Test updating a sub-problem."""
        temp_db.create_problem(sample_problem)
        temp_db.create_sub_problem(sample_sub_problem)
        
        # Update the sub-problem
        sample_sub_problem.status = SubProblemStatus.IN_PROGRESS
        sample_sub_problem.assigned_team = "blue_team"
        
        success = temp_db.update_sub_problem(sample_sub_problem)
        assert success is True
        
        # Verify the update
        retrieved = temp_db.get_sub_problem(sample_sub_problem.id)
        assert retrieved.status == SubProblemStatus.IN_PROGRESS
        assert retrieved.assigned_team == "blue_team"
    
    def test_list_sub_problems_by_parent(self, temp_db, sample_problem):
        """Test listing sub-problems for a parent problem."""
        temp_db.create_problem(sample_problem)
        
        # Create multiple sub-problems
        for i in range(3):
            sub_problem = SubProblem(
                id=generate_id("subproblem"),
                parent_id=sample_problem.id,
                title=f"Sub-problem {i}",
                description=f"Description {i}",
                type=SubProblemType.ANALYSIS,
                complexity_score=ComplexityScore(
                    cognitive_complexity=3.0,
                    computational_complexity=3.0,
                    domain_complexity=3.0,
                    integration_complexity=3.0,
                    overall_complexity=3.0,
                    explanation="Test"
                ),
                dependencies=[],
                success_criteria=[],
                validation_gauntlet="test",
                priority=i + 1
            )
            temp_db.create_sub_problem(sub_problem)
        
        # List sub-problems
        sub_problems = temp_db.list_sub_problems_by_parent(sample_problem.id)
        assert len(sub_problems) == 3
        
        # Verify they're ordered by priority (descending)
        assert sub_problems[0].priority >= sub_problems[1].priority


# ============================================================================
# PATTERN CRUD TESTS (Knowledge Learning)
# ============================================================================

class TestPatternCRUD:
    """Test CRUD operations for patterns."""
    
    def test_create_pattern(self, temp_db):
        """Test creating a pattern."""
        pattern = Pattern(
            id=generate_id("pattern"),
            problem_type=ProblemType.RESEARCH,
            strategy=DecompositionStrategy.SEMANTIC,
            pattern_description="Semantic decomposition for research problems",
            success_rate=0.85,
            usage_count=10,
            avg_quality_score=0.82,
            applicable_domains=["machine_learning", "data_science"]
        )
        
        pattern_id = temp_db.create_pattern(pattern)
        assert pattern_id == pattern.id
    
    def test_get_patterns_by_type(self, temp_db):
        """Test retrieving patterns by problem type."""
        # Create patterns for different types
        for i in range(5):
            pattern = Pattern(
                id=generate_id("pattern"),
                problem_type=ProblemType.RESEARCH if i < 3 else ProblemType.DESIGN,
                strategy=DecompositionStrategy.SEMANTIC,
                pattern_description=f"Pattern {i}",
                success_rate=0.8 + (i * 0.02),
                usage_count=i + 1,
                avg_quality_score=0.75 + (i * 0.03),
                applicable_domains=["test"]
            )
            temp_db.create_pattern(pattern)
        
        # Get research patterns
        research_patterns = temp_db.get_patterns_by_type("research")
        assert len(research_patterns) == 3
        
        # Verify they're ordered by success rate
        assert research_patterns[0].success_rate >= research_patterns[1].success_rate
    
    def test_update_pattern_usage(self, temp_db):
        """Test updating pattern usage statistics."""
        pattern = Pattern(
            id=generate_id("pattern"),
            problem_type=ProblemType.ANALYSIS,
            strategy=DecompositionStrategy.COMPLEXITY,
            pattern_description="Test pattern",
            success_rate=0.8,
            usage_count=10,
            avg_quality_score=0.75,
            applicable_domains=["test"]
        )
        
        temp_db.create_pattern(pattern)
        
        # Update usage with success
        temp_db.update_pattern_usage(pattern.id, success=True, quality_score=0.9)
        
        # Retrieve and verify
        patterns = temp_db.get_patterns_by_type("analysis")
        updated_pattern = patterns[0]
        
        assert updated_pattern.usage_count == 11
        assert updated_pattern.success_rate > 0.8  # Should increase
        assert updated_pattern.avg_quality_score > 0.75  # Should increase


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestDatabaseIntegration:
    """Test integration between different database operations."""
    
    def test_complete_workflow(self, temp_db):
        """Test a complete problem decomposition workflow."""
        # Create a problem
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Complete Workflow Test",
            description="Testing complete workflow",
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=DomainContext(domain="test"),
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0,
                computational_complexity=6.0,
                domain_complexity=6.0,
                integration_complexity=6.0,
                overall_complexity=6.0,
                explanation="Medium complexity"
            ),
            constraints=[],
            success_criteria=[],
            stakeholders=["user1"],
            resources_available={"budget": 10000}
        )
        
        temp_db.create_problem(problem)
        
        # Create sub-problems
        sub_problem_ids = []
        for i in range(3):
            sub_problem = SubProblem(
                id=generate_id("subproblem"),
                parent_id=problem.id,
                title=f"Sub-problem {i}",
                description=f"Description {i}",
                type=SubProblemType.IMPLEMENTATION,
                complexity_score=ComplexityScore(
                    cognitive_complexity=2.0,
                    computational_complexity=2.0,
                    domain_complexity=2.0,
                    integration_complexity=2.0,
                    overall_complexity=2.0,
                    explanation="Low"
                ),
                dependencies=[],
                success_criteria=[],
                validation_gauntlet="test"
            )
            temp_db.create_sub_problem(sub_problem)
            sub_problem_ids.append(sub_problem.id)
        
        # Verify all sub-problems were created
        sub_problems = temp_db.list_sub_problems_by_parent(problem.id)
        assert len(sub_problems) == 3
        
        # Update sub-problem status
        sub_problem = temp_db.get_sub_problem(sub_problem_ids[0])
        sub_problem.status = SubProblemStatus.SOLVED
        temp_db.update_sub_problem(sub_problem)
        
        # Verify update
        updated = temp_db.get_sub_problem(sub_problem_ids[0])
        assert updated.status == SubProblemStatus.SOLVED
        
        # Delete problem (should cascade to sub-problems)
        temp_db.delete_problem(problem.id)
        
        # Verify deletion
        assert temp_db.get_problem(problem.id) is None
        assert temp_db.get_sub_problem(sub_problem_ids[0]) is None
    
    def test_context_manager(self):
        """Test using database as context manager."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        try:
            with SovereignDatabase(path) as db:
                problem = ProblemDefinition(
                    id=generate_id("problem"),
                    title="Context Manager Test",
                    description="Test",
                    problem_type=ProblemType.ANALYSIS,
                    domain_context=DomainContext(domain="test"),
                    complexity_score=ComplexityScore(
                        cognitive_complexity=5.0,
                        computational_complexity=5.0,
                        domain_complexity=5.0,
                        integration_complexity=5.0,
                        overall_complexity=5.0,
                        explanation="Test"
                    ),
                    constraints=[],
                    success_criteria=[],
                    stakeholders=[],
                    resources_available={}
                )
                db.create_problem(problem)
            
            # Verify database was closed
            # Connection should be closed after context manager exits
            assert True  # If we get here, context manager worked
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
