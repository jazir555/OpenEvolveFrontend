"""
SQLAlchemy Database Models for Z3 Knowledge Storage

Extends the existing knowledge engine database with Z3-specific tables
for proof patterns, constraint patterns, strategies, and mathematical insights.

Author: OpenEvolve
Created: 2026-01-31
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json

# SQLAlchemy imports
try:
    from sqlalchemy import (
        Column, Integer, String, Float, DateTime, 
        Text, JSON, ForeignKey, Index, create_engine
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # Create dummy base class for when SQLAlchemy is not available
    class DummyBase:
        pass
    declarative_base = lambda: DummyBase

Base = declarative_base()


class Z3KnowledgeEntry(Base):
    """Base table for Z3 knowledge entries."""
    __tablename__ = 'z3_knowledge_entries'
    
    id = Column(Integer, primary_key=True)
    entry_type = Column(String(50), nullable=False, index=True)  # 'proof_pattern', 'constraint', 'strategy', 'insight'
    content_hash = Column(String(64), unique=True, index=True)
    content = Column(Text)
    metadata_json = Column(JSON)
    problem_domain = Column(String(100), index=True)
    source_problem_id = Column(String(100), index=True)
    confidence = Column(Float, default=1.0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    proof_pattern = relationship("Z3ProofPattern", back_populates="knowledge_entry", uselist=False)
    constraint_pattern = relationship("Z3ConstraintPattern", back_populates="knowledge_entry", uselist=False)
    strategy = relationship("Z3Strategy", back_populates="knowledge_entry", uselist=False)
    insight = relationship("Z3MathematicalInsight", back_populates="knowledge_entry", uselist=False)
    
    __mapper_args__ = {
        'polymorphic_on': entry_type,
        'polymorphic_identity': 'base'
    }


class Z3ProofPattern(Base):
    """Stores extracted proof patterns from Z3."""
    __tablename__ = 'z3_proof_patterns'
    
    id = Column(Integer, primary_key=True)
    knowledge_entry_id = Column(Integer, ForeignKey('z3_knowledge_entries.id'))
    
    pattern_signature = Column(String(255), index=True)
    pattern_name = Column(String(200))
    description = Column(Text)
    tactic_sequence = Column(JSON)  # List of tactics
    applicable_domains = Column(JSON)  # List of domains
    proof_depth = Column(Integer)
    branching_factor = Column(Float)
    effectiveness_score = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)
    source_proofs = Column(JSON)  # List of proof IDs
    
    # Relationship
    knowledge_entry = relationship("Z3KnowledgeEntry", back_populates="proof_pattern")
    
    __table_args__ = (
        Index('idx_proof_pattern_domain', 'applicable_domains'),
        Index('idx_proof_pattern_effectiveness', 'effectiveness_score'),
    )


class Z3ConstraintPattern(Base):
    """Stores constraint patterns extracted from Z3 solving."""
    __tablename__ = 'z3_constraint_patterns'
    
    id = Column(Integer, primary_key=True)
    knowledge_entry_id = Column(Integer, ForeignKey('z3_knowledge_entries.id'))
    
    pattern_type = Column(String(50), index=True)  # 'linear', 'nonlinear', 'boolean', 'mixed', 'atomic'
    structure_template = Column(Text)
    variables_involved = Column(JSON)  # List of variable names
    complexity_score = Column(Float, default=0.0)
    frequency = Column(Integer, default=0)
    typical_solving_time = Column(Float, default=0.0)
    average_success_rate = Column(Float, default=0.0)
    constraint_examples = Column(JSON)  # List of example constraints
    
    # Relationship
    knowledge_entry = relationship("Z3KnowledgeEntry", back_populates="constraint_pattern")
    
    __table_args__ = (
        Index('idx_constraint_type', 'pattern_type'),
        Index('idx_constraint_complexity', 'complexity_score'),
    )


class Z3Strategy(Base):
    """Stores learned solution strategies."""
    __tablename__ = 'z3_strategies'
    
    id = Column(Integer, primary_key=True)
    knowledge_entry_id = Column(Integer, ForeignKey('z3_knowledge_entries.id'))
    
    strategy_name = Column(String(200))
    problem_pattern = Column(String(255), index=True)
    recommended_tactics = Column(JSON)  # List of tactics
    solver_configuration = Column(JSON)  # Solver config dict
    prerequisites = Column(JSON)  # List of prerequisites
    expected_avg_time = Column(Float)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    
    # Relationship
    knowledge_entry = relationship("Z3KnowledgeEntry", back_populates="strategy")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    __table_args__ = (
        Index('idx_strategy_pattern', 'problem_pattern'),
    )


class Z3MathematicalInsight(Base):
    """Stores mathematical insights extracted from Z3 solutions."""
    __tablename__ = 'z3_mathematical_insights'
    
    id = Column(Integer, primary_key=True)
    knowledge_entry_id = Column(Integer, ForeignKey('z3_knowledge_entries.id'))
    
    category = Column(String(50), index=True)  # 'invariant', 'bound', 'relation', 'optimization'
    statement = Column(Text)
    formal_representation = Column(Text)
    proof_sketch = Column(Text)
    confidence_score = Column(Float, default=0.0)
    derived_from = Column(JSON)  # List of source problems
    applications = Column(JSON)  # List of application contexts
    verified = Column(Integer, default=0)  # 0=unverified, 1=verified, -1=disproven
    
    # Relationship
    knowledge_entry = relationship("Z3KnowledgeEntry", back_populates="insight")
    
    __table_args__ = (
        Index('idx_insight_category', 'category'),
        Index('idx_insight_confidence', 'confidence_score'),
    )


class Z3SolverResult(Base):
    """Stores Z3 solver execution results."""
    __tablename__ = 'z3_solver_results'
    
    id = Column(Integer, primary_key=True)
    result_id = Column(String(100), unique=True, index=True)
    
    problem_hash = Column(String(64), index=True)
    problem_statement = Column(Text)
    problem_type = Column(String(50), index=True)  # 'constraint', 'theorem', 'optimization', 'smt'
    
    result_status = Column(String(20), index=True)  # 'sat', 'unsat', 'unknown', 'timeout', 'error'
    solving_time_ms = Column(Integer)
    model_data = Column(JSON)  # Model assignments
    proof_data = Column(JSON)  # Proof steps if available
    tactics_used = Column(JSON)  # List of tactics used
    solver_configuration = Column(JSON)
    
    # Statistics
    constraint_count = Column(Integer)
    variable_count = Column(Integer)
    memory_usage_mb = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_solver_result_hash', 'problem_hash'),
        Index('idx_solver_result_type', 'problem_type', 'result_status'),
    )


class Z3KnowledgeGraphNode(Base):
    """Nodes for Z3 knowledge graph."""
    __tablename__ = 'z3_kg_nodes'
    
    id = Column(Integer, primary_key=True)
    node_id = Column(String(100), unique=True, index=True)
    node_type = Column(String(50), index=True)  # 'pattern', 'strategy', 'insight', 'constraint', 'variable'
    label = Column(String(255))
    properties = Column(JSON)
    embedding = Column(JSON)  # Vector embedding for similarity search
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Z3KnowledgeGraphEdge(Base):
    """Edges for Z3 knowledge graph."""
    __tablename__ = 'z3_kg_edges'
    
    id = Column(Integer, primary_key=True)
    edge_id = Column(String(100), unique=True, index=True)
    source_node_id = Column(String(100), ForeignKey('z3_kg_nodes.node_id'), index=True)
    target_node_id = Column(String(100), ForeignKey('z3_kg_nodes.node_id'), index=True)
    relation_type = Column(String(50), index=True)
    weight = Column(Float, default=1.0)
    properties = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_kg_edge_relation', 'source_node_id', 'relation_type'),
    )


# =============================================================================
# Data Access Objects (DAOs)
# =============================================================================

@dataclass
class ProofPatternDTO:
    """Data transfer object for proof patterns."""
    pattern_id: str
    name: str
    description: str
    tactic_sequence: List[str]
    applicable_domains: List[str]
    effectiveness_score: float
    usage_count: int
    
    @classmethod
    def from_model(cls, model: Z3ProofPattern) -> 'ProofPatternDTO':
        return cls(
            pattern_id=model.pattern_signature,
            name=model.pattern_name or "",
            description=model.description or "",
            tactic_sequence=model.tactic_sequence or [],
            applicable_domains=model.applicable_domains or [],
            effectiveness_score=model.effectiveness_score or 0.0,
            usage_count=model.usage_count or 0
        )


@dataclass
class ConstraintPatternDTO:
    """Data transfer object for constraint patterns."""
    pattern_id: str
    pattern_type: str
    structure: str
    variables: List[str]
    complexity: float
    frequency: int
    typical_solving_time: float
    
    @classmethod
    def from_model(cls, model: Z3ConstraintPattern) -> 'ConstraintPatternDTO':
        return cls(
            pattern_id=str(model.id),
            pattern_type=model.pattern_type or "unknown",
            structure=model.structure_template or "",
            variables=model.variables_involved or [],
            complexity=model.complexity_score or 0.0,
            frequency=model.frequency or 0,
            typical_solving_time=model.typical_solving_time or 0.0
        )


@dataclass
class StrategyDTO:
    """Data transfer object for strategies."""
    strategy_id: str
    name: str
    problem_pattern: str
    tactics: List[str]
    success_rate: float
    avg_time: float
    
    @classmethod
    def from_model(cls, model: Z3Strategy) -> 'StrategyDTO':
        return cls(
            strategy_id=str(model.id),
            name=model.strategy_name or "",
            problem_pattern=model.problem_pattern or "",
            tactics=model.recommended_tactics or [],
            success_rate=model.success_rate,
            avg_time=model.expected_avg_time or 0.0
        )


# =============================================================================
# Migration Helper
# =============================================================================

def create_z3_tables(engine_url: str = "sqlite:///z3_knowledge.db"):
    """Create all Z3 knowledge tables."""
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy is required for database operations")
    
    engine = create_engine(engine_url)
    Base.metadata.create_all(engine)
    print(f"Z3 knowledge tables created in {engine_url}")
    return engine


def drop_z3_tables(engine_url: str = "sqlite:///z3_knowledge.db"):
    """Drop all Z3 knowledge tables."""
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy is required for database operations")
    
    engine = create_engine(engine_url)
    Base.metadata.drop_all(engine)
    print(f"Z3 knowledge tables dropped from {engine_url}")


# =============================================================================
# Example Usage
# =============================================================================

def example_models():
    """Example: Create and use Z3 knowledge models."""
    if not SQLALCHEMY_AVAILABLE:
        print("SQLAlchemy not available, skipping example")
        return
    
    # Create tables
    engine = create_z3_tables("sqlite:///example_z3_knowledge.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create a knowledge entry
    entry = Z3KnowledgeEntry(
        entry_type="strategy",
        content_hash="abc123",
        content="Strategy for linear constraints",
        metadata_json={"version": "1.0"},
        problem_domain="linear_programming",
        confidence=0.85
    )
    session.add(entry)
    session.commit()
    
    # Create a strategy
    strategy = Z3Strategy(
        knowledge_entry_id=entry.id,
        strategy_name="Linear Solver Strategy",
        problem_pattern="linear_vars_5_constraints_10",
        recommended_tactics=["simplify", "solve-eqs", "smt"],
        solver_configuration={"timeout": 30, "threads": 4},
        expected_avg_time=2.5,
        success_count=10,
        failure_count=2
    )
    session.add(strategy)
    session.commit()
    
    print(f"Created strategy with success rate: {strategy.success_rate:.1%}")
    
    # Query
    results = session.query(Z3Strategy).all()
    print(f"Found {len(results)} strategies")
    
    session.close()


if __name__ == "__main__":
    example_models()
