"""
Database Models for Mathematical Knowledge Integration

Clean SQLAlchemy models without import conflicts.
"""

from datetime import datetime
from typing import Any, Dict, Optional

# Use try/except to handle import cleanly
try:
    import sqlalchemy as sa
    from sqlalchemy.orm import declarative_base, relationship
    
    Base = declarative_base()
    
    class Z3KnowledgeBase(Base):
        """Base knowledge table."""
        __tablename__ = 'z3_knowledge_base'
        
        id = sa.Column(sa.Integer, primary_key=True)
        record_type = sa.Column(sa.String(50), nullable=False, index=True)
        record_hash = sa.Column(sa.String(64), unique=True, index=True)
        content_json = sa.Column(sa.Text)
        features_json = sa.Column(sa.Text)  # JSON as text
        metadata_json = sa.Column(sa.Text)
        source_problem = sa.Column(sa.String(500))
        problem_domain = sa.Column(sa.String(100))
        confidence = sa.Column(sa.Float, default=1.0)
        success_count = sa.Column(sa.Integer, default=0)
        failure_count = sa.Column(sa.Integer, default=0)
        created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
        updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    class Z3SolverRun(Base):
        """Solver execution records."""
        __tablename__ = 'z3_solver_runs'
        
        id = sa.Column(sa.Integer, primary_key=True)
        run_id = sa.Column(sa.String(100), unique=True, index=True)
        problem_hash = sa.Column(sa.String(64), index=True)
        problem_statement = sa.Column(sa.Text)
        result_status = sa.Column(sa.String(20))
        solving_time_ms = sa.Column(sa.Float)
        memory_usage_mb = sa.Column(sa.Float)
        tactics_used = sa.Column(sa.Text)  # JSON
        created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    
    class LeanProofRecord(Base):
        """Lean proof records."""
        __tablename__ = 'lean_proof_records'
        
        id = sa.Column(sa.Integer, primary_key=True)
        theorem_id = sa.Column(sa.String(100), unique=True, index=True)
        theorem_statement = sa.Column(sa.Text)
        proof_script = sa.Column(sa.Text)
        tactic_sequence = sa.Column(sa.Text)  # JSON
        success = sa.Column(sa.Boolean)
        execution_time_ms = sa.Column(sa.Float)
        created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    
    MODELS_AVAILABLE = True
    
    # Export main classes
    __all__ = [
        'Base',
        'Z3KnowledgeBase',
        'Z3SolverRun',
        'LeanProofRecord',
        'MODELS_AVAILABLE'
    ]
    
except ImportError:
    MODELS_AVAILABLE = False
    Base = None
    Z3KnowledgeBase = None
    Z3SolverRun = None
    LeanProofRecord = None
    
    __all__ = ['MODELS_AVAILABLE', 'Base', 'Z3KnowledgeBase', 'Z3SolverRun', 'LeanProofRecord']
