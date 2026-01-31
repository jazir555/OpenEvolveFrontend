"""
Z3 Integration Database Models

SQLAlchemy models for persisting:
- Solver results
- Proof data
- Knowledge patterns
- Performance metrics
- Configuration history

Supports SQLite, PostgreSQL, and MySQL.

Author: OpenEvolve
Created: 2026-01-31
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# SQLAlchemy imports
try:
    from sqlalchemy import (
        create_engine, Column, String, Integer, Float, Boolean, 
        DateTime, Text, JSON, ForeignKey, Index, event
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, Session
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # Create dummy base class for type hints
    class _Base:
        pass
    declarative_base = lambda: _Base

# Configure logging
logger = logging.getLogger(__name__)

Base = declarative_base()


# =============================================================================
# Result Models
# =============================================================================

class SolverResult(Base):
    """Database model for solver results."""
    __tablename__ = 'solver_results'
    
    id = Column(String(64), primary_key=True)
    operation_type = Column(String(50), nullable=False, index=True)
    problem_hash = Column(String(64), nullable=False, index=True)
    problem_statement = Column(Text)
    
    # Result data
    status = Column(String(20), nullable=False)  # sat, unsat, unknown, error
    satisfiable = Column(Boolean)
    model_data = Column(JSON)  # Variable assignments
    objective_value = Column(Float)
    
    # Metadata
    solver_used = Column(String(50))
    tactics_used = Column(JSON)  # List of tactics
    proof_data = Column(Text)  # Proof if generated
    
    # Performance
    execution_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, index=True)
    
    # Tags for categorization
    tags = Column(JSON)
    
    __table_args__ = (
        Index('idx_result_lookup', 'problem_hash', 'operation_type'),
        Index('idx_result_status', 'status', 'created_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "operation_type": self.operation_type,
            "status": self.status,
            "satisfiable": self.satisfiable,
            "model": self.model_data,
            "objective": self.objective_value,
            "solver": self.solver_used,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class TheoremProof(Base):
    """Database model for theorem proofs."""
    __tablename__ = 'theorem_proofs'
    
    id = Column(String(64), primary_key=True)
    theorem_hash = Column(String(64), nullable=False, index=True)
    theorem_statement = Column(Text, nullable=False)
    
    # Proof data
    proven = Column(Boolean, nullable=False)
    proof_steps = Column(JSON)  # Structured proof steps
    proof_tree = Column(JSON)  # Hierarchical proof tree
    raw_proof = Column(Text)  # Raw proof text
    
    # Tactics and strategy
    tactics_used = Column(JSON)
    strategy_used = Column(String(100))
    
    # Cross-verification
    z3_verified = Column(Boolean)
    lean_verified = Column(Boolean)
    verification_agreement = Column(Boolean)
    
    # Metadata
    confidence_score = Column(Float)
    domain = Column(String(100), index=True)
    
    # Performance
    execution_time_ms = Column(Float)
    elaboration_time_ms = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "proven": self.proven,
            "confidence": self.confidence_score,
            "tactics": self.tactics_used,
            "z3_verified": self.z3_verified,
            "lean_verified": self.lean_verified,
            "execution_time_ms": self.execution_time_ms
        }


# =============================================================================
# Knowledge Models
# =============================================================================

class ProofPattern(Base):
    """Database model for learned proof patterns."""
    __tablename__ = 'proof_patterns'
    
    id = Column(String(64), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Pattern data
    tactic_sequence = Column(JSON, nullable=False)
    applicable_domains = Column(JSON)
    problem_pattern = Column(Text)  # Regex or structure
    
    # Statistics
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)
    avg_execution_time_ms = Column(Float)
    
    # Quality metrics
    confidence = Column(Float, default=0.0)
    complexity_score = Column(Float)
    
    # Source
    source_proofs = Column(JSON)  # IDs of source proofs
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Versioning
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tactics": self.tactic_sequence,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "confidence": self.confidence
        }
    
    def update_success_rate(self):
        """Recalculate success rate."""
        total = self.success_count + self.failure_count
        if total > 0:
            self.success_rate = self.success_count / total


class SolutionStrategy(Base):
    """Database model for learned solution strategies."""
    __tablename__ = 'solution_strategies'
    
    id = Column(String(64), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Strategy definition
    problem_pattern = Column(Text, nullable=False)  # Matching pattern
    problem_features = Column(JSON)  # Feature requirements
    recommended_tactics = Column(JSON, nullable=False)
    solver_configuration = Column(JSON)
    prerequisites = Column(JSON)
    
    # Performance expectations
    expected_performance = Column(JSON)  # avg_time, success_rate, etc.
    
    # Statistics
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    total_attempts = Column(Integer, default=0)
    
    # Quality
    average_solving_time_ms = Column(Float)
    reliability_score = Column(Float, default=0.0)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "pattern": self.problem_pattern,
            "tactics": self.recommended_tactics,
            "success_count": self.success_count,
            "reliability": self.reliability_score
        }


class MathematicalInsight(Base):
    """Database model for extracted mathematical insights."""
    __tablename__ = 'mathematical_insights'
    
    id = Column(String(64), primary_key=True)
    category = Column(String(50), nullable=False, index=True)  # invariant, bound, relation
    statement = Column(Text, nullable=False)
    formal_representation = Column(Text)
    
    # Context
    problem_domain = Column(String(100), index=True)
    related_variables = Column(JSON)
    
    # Provenance
    derived_from = Column(JSON)  # Source problem/theorem IDs
    proof_sketch = Column(Text)
    
    # Quality
    confidence = Column(Float, default=0.0)
    verified = Column(Boolean, default=False)
    verification_method = Column(String(50))
    
    # Usage tracking
    applications = Column(JSON)  # IDs where applied
    application_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "statement": self.statement[:100] + "..." if len(self.statement) > 100 else self.statement,
            "confidence": self.confidence,
            "verified": self.verified
        }


# =============================================================================
# Performance Models
# =============================================================================

class PerformanceMetric(Base):
    """Database model for performance metrics."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Metric identification
    operation_name = Column(String(100), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    
    # Value
    value = Column(Float, nullable=False)
    unit = Column(String(20))
    
    # Context
    tags = Column(JSON)
    session_id = Column(String(64), index=True)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_metric_lookup', 'operation_name', 'metric_name', 'timestamp'),
    )


class PerformanceSnapshot(Base):
    """Database model for performance snapshots."""
    __tablename__ = 'performance_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Snapshot data
    snapshot_data = Column(JSON, nullable=False)
    
    # Summary statistics
    total_operations = Column(Integer)
    active_solvers = Column(Integer)
    queue_depth = Column(Integer)
    memory_usage_mb = Column(Float)
    cpu_percent = Column(Float)
    
    # Alert counts
    alert_count = Column(Integer)
    critical_alert_count = Column(Integer)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class PerformanceAlert(Base):
    """Database model for performance alerts."""
    __tablename__ = 'performance_alerts'
    
    id = Column(String(64), primary_key=True)
    severity = Column(String(20), nullable=False, index=True)  # info, warning, error, critical
    message = Column(Text, nullable=False)
    
    # Metric details
    metric_name = Column(String(100))
    threshold = Column(Float)
    actual_value = Column(Float)
    
    # Status
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# =============================================================================
# Cache Models
# =============================================================================

class CacheEntry(Base):
    """Database model for persistent cache."""
    __tablename__ = 'cache_entries'
    
    key = Column(String(256), primary_key=True)
    operation_type = Column(String(50), nullable=False, index=True)
    
    # Cached data
    value_blob = Column(Text)  # JSON serialized
    value_size_bytes = Column(Integer)
    
    # Metadata
    tags = Column(JSON)
    access_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, index=True)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    # Versioning
    version = Column(Integer, default=1)
    checksum = Column(String(64))
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


# =============================================================================
# Configuration Models
# =============================================================================

class ConfigurationHistory(Base):
    """Database model for configuration changes."""
    __tablename__ = 'configuration_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Change details
    config_key = Column(String(200), nullable=False)
    old_value = Column(Text)
    new_value = Column(Text)
    
    # Who made the change
    changed_by = Column(String(100))
    change_reason = Column(Text)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """
    Database connection and session management.
    
    Supports SQLite, PostgreSQL, and MySQL.
    """
    
    def __init__(self, database_url: Optional[str] = None, config: Optional[Dict] = None):
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy is required for database operations")
        
        self.database_url = database_url or self._build_url(config)
        self.engine = None
        self.SessionLocal = None
        
        self._initialize()
    
    def _build_url(self, config: Optional[Dict]) -> str:
        """Build database URL from configuration."""
        if config is None:
            return "sqlite:///./data/z3_integration.db"
        
        db_type = config.get('type', 'sqlite')
        
        if db_type == 'sqlite':
            path = config.get('sqlite', {}).get('path', './data/z3_integration.db')
            return f"sqlite:///{path}"
        
        elif db_type == 'postgresql':
            pg_config = config.get('postgresql', {})
            return (
                f"postgresql://{pg_config.get('username')}:"
                f"{pg_config.get('password')}@"
                f"{pg_config.get('host', 'localhost')}:"
                f"{pg_config.get('port', 5432)}/"
                f"{pg_config.get('database', 'z3_integration')}"
            )
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def _initialize(self):
        """Initialize database connection."""
        # Create engine with pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Create tables
        self._create_tables()
        
        logger.info(f"Database initialized: {self.database_url}")
    
    def _create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Repository Classes
# =============================================================================

class ResultRepository:
    """Repository for solver results."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_result(self, result: SolverResult) -> str:
        """Save solver result."""
        with self.db.get_session() as session:
            session.add(result)
            session.commit()
            return result.id
    
    def get_result(self, result_id: str) -> Optional[SolverResult]:
        """Get result by ID."""
        with self.db.get_session() as session:
            return session.query(SolverResult).filter(SolverResult.id == result_id).first()
    
    def find_by_problem(
        self,
        problem_hash: str,
        operation_type: Optional[str] = None
    ) -> List[SolverResult]:
        """Find results by problem hash."""
        with self.db.get_session() as session:
            query = session.query(SolverResult).filter(
                SolverResult.problem_hash == problem_hash
            )
            if operation_type:
                query = query.filter(SolverResult.operation_type == operation_type)
            return query.order_by(SolverResult.created_at.desc()).all()


class KnowledgeRepository:
    """Repository for knowledge data."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_pattern(self, pattern: ProofPattern) -> str:
        """Save proof pattern."""
        with self.db.get_session() as session:
            session.add(pattern)
            session.commit()
            return pattern.id
    
    def get_active_patterns(
        self,
        domain: Optional[str] = None,
        min_success_rate: float = 0.0
    ) -> List[ProofPattern]:
        """Get active patterns matching criteria."""
        with self.db.get_session() as session:
            query = session.query(ProofPattern).filter(
                ProofPattern.is_active == True,
                ProofPattern.success_rate >= min_success_rate
            )
            if domain:
                query = query.filter(
                    ProofPattern.applicable_domains.contains([domain])
                )
            return query.order_by(ProofPattern.success_rate.desc()).all()


# =============================================================================
# Global Instance
# =============================================================================

_db_manager: Optional[DatabaseManager] = None


def get_database_manager(
    database_url: Optional[str] = None,
    config: Optional[Dict] = None
) -> DatabaseManager:
    """Get global database manager."""
    global _db_manager
    if _db_manager is None:
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy is required for database operations")
        _db_manager = DatabaseManager(database_url, config)
    return _db_manager


# =============================================================================
# Example Usage
# =============================================================================

def example_database_usage():
    """Example: Database usage."""
    if not SQLALCHEMY_AVAILABLE:
        print("SQLAlchemy not available")
        return
    
    # Create manager
    db = DatabaseManager("sqlite:///./example.db")
    
    # Create result
    result = SolverResult(
        id="result_001",
        operation_type="constraint_solving",
        problem_hash="abc123",
        problem_statement="x > 0",
        status="sat",
        satisfiable=True,
        model_data={"x": 5},
        solver_used="z3",
        execution_time_ms=150.0
    )
    
    # Save
    repo = ResultRepository(db)
    result_id = repo.save_result(result)
    print(f"Saved result: {result_id}")
    
    # Retrieve
    retrieved = repo.get_result(result_id)
    if retrieved:
        print(f"Retrieved: {retrieved.to_dict()}")
    
    db.close()


if __name__ == "__main__":
    print("Z3 Database Models")
    print("=" * 50)
    example_database_usage()
