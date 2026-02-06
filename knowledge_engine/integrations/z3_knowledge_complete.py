"""
Complete Z3 Knowledge Integration - Production Ready

Features:
- Full database persistence with SQLAlchemy
- Comprehensive feature extraction pipeline
- Online learning with feedback loops
- Advanced proof parsing
- Conflict detection and resolution
- Redis caching integration
- Comprehensive monitoring
- Configuration management
- CAV-NLP enhanced knowledge extraction

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import logging
import hashlib
import pickle
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import re

# Configure logging
logger = logging.getLogger(__name__)

# CAV-NLP integration imports
try:
    from openevolve.unified_math_service import UnifiedMathService
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    UnifiedMathService = None
    EnhancedZ3Solver = None

# SQLAlchemy imports
try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, event, ForeignKey, Index
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, Session
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
    
    # Aliases for SQLAlchemy types
    Column = sa.Column
    Integer = sa.Integer
    String = sa.String
    Float = sa.Float
    DateTime = sa.DateTime
    Text = sa.Text
    JSON = sa.JSON
    Boolean = sa.Boolean
    
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = type('Base', (), {'metadata': type('metadata', (), {'create_all': lambda x: None})()})
    Column = lambda *args, **kwargs: None
    Integer = lambda: None
    String = lambda x: None
    Float = lambda: None
    DateTime = lambda: None
    Text = lambda: None
    JSON = lambda: None
    Boolean = lambda: None

# Redis imports
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# NumPy for numerical operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from z3_knowledge_extraction import (
    Z3KnowledgeExtractor,
    ProofPattern,
    ConstraintPattern,
    SolutionStrategy,
    MathematicalInsight
)

# Z3 imports - import specific items to avoid overwriting SQLAlchemy types
Z3_LIB_AVAILABLE = False
z3 = None

if SQLALCHEMY_AVAILABLE:
    # Save SQLAlchemy types before importing Z3
    _sa_String = String
    _sa_Integer = Integer
    _sa_Float = Float
    _sa_Bool = Boolean
    
    try:
        import z3 as _z3_module
        z3 = _z3_module
        Z3_LIB_AVAILABLE = True
        
        # Restore SQLAlchemy types
        String = _sa_String
        Integer = _sa_Integer
        Float = _sa_Float
        Boolean = _sa_Bool
        
    except ImportError:
        Z3_LIB_AVAILABLE = False
        # Types already set above

# Database Models
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class Z3KnowledgeRecord(Base):
        """Base record for all Z3 knowledge."""
        __tablename__ = 'z3_knowledge_records'
        
        id = Column(Integer, primary_key=True)
        record_type = Column(String(50), nullable=False, index=True)
        record_hash = Column(String(64), unique=True, index=True)
        content = Column(Text)
        features = Column(JSON)
        metadata_ = Column('metadata', JSON)  # Use metadata_ to avoid conflict
        source_problem = Column(String(500), index=True)
        problem_domain = Column(String(100), index=True)
        confidence = Column(Float, default=1.0)
        success_count = Column(Integer, default=0)
        failure_count = Column(Integer, default=0)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        last_accessed = Column(DateTime)
        access_count = Column(Integer, default=0)
        
        # Relationships
        proof_pattern = relationship("Z3ProofPatternRecord", back_populates="knowledge_record", uselist=False)
        constraint_pattern = relationship("Z3ConstraintPatternRecord", back_populates="knowledge_record", uselist=False)
        strategy_record = relationship("Z3StrategyRecord", back_populates="knowledge_record", uselist=False)
        insight_record = relationship("Z3InsightRecord", back_populates="knowledge_record", uselist=False)
    
    class Z3ProofPatternRecord(Base):
        """Proof pattern database record."""
        __tablename__ = 'z3_proof_patterns_db'
        
        id = Column(Integer, primary_key=True)
        knowledge_record_id = Column(Integer, ForeignKey('z3_knowledge_records.id'))
        
        pattern_signature = Column(String(255), index=True)
        tactic_sequence = Column(JSON)
        proof_tree_structure = Column(JSON)
        applicable_domains = Column(JSON)
        proof_depth = Column(Integer)
        branching_factor = Column(Float)
        effectiveness_score = Column(Float, default=0.0)
        usage_count = Column(Integer, default=0)
        average_solving_time = Column(Float)
        
        knowledge_record = relationship("Z3KnowledgeRecord", back_populates="proof_pattern")
        
        __table_args__ = (
            Index('idx_proof_effectiveness', 'effectiveness_score'),
            Index('idx_proof_usage', 'usage_count'),
        )
    
    class Z3ConstraintPatternRecord(Base):
        """Constraint pattern database record."""
        __tablename__ = 'z3_constraint_patterns_db'
        
        id = Column(Integer, primary_key=True)
        knowledge_record_id = Column(Integer, ForeignKey('z3_knowledge_records.id'))
        
        pattern_type = Column(String(50), index=True)
        structure_template = Column(Text)
        variables_pattern = Column(JSON)
        constraint_count = Column(Integer)
        complexity_score = Column(Float)
        linear_coefficients = Column(JSON)
        nonlinear_terms = Column(JSON)
        typical_solving_time = Column(Float)
        
        knowledge_record = relationship("Z3KnowledgeRecord", back_populates="constraint_pattern")
    
    class Z3StrategyRecord(Base):
        """Strategy database record."""
        __tablename__ = 'z3_strategies_db'
        
        id = Column(Integer, primary_key=True)
        knowledge_record_id = Column(Integer, ForeignKey('z3_knowledge_records.id'))
        
        strategy_name = Column(String(200))
        problem_pattern = Column(String(255), index=True)
        feature_vector = Column(JSON)
        recommended_tactics = Column(JSON)
        solver_config = Column(JSON)
        success_count = Column(Integer, default=0)
        failure_count = Column(Integer, default=0)
        avg_solving_time = Column(Float)
        avg_memory_usage = Column(Float)
        problem_types = Column(JSON)
        
        knowledge_record = relationship("Z3KnowledgeRecord", back_populates="strategy_record")
        
        @property
        def success_rate(self) -> float:
            total = self.success_count + self.failure_count
            return self.success_count / total if total > 0 else 0.0
    
    class Z3InsightRecord(Base):
        """Mathematical insight database record."""
        __tablename__ = 'z3_insights_db'
        
        id = Column(Integer, primary_key=True)
        knowledge_record_id = Column(Integer, ForeignKey('z3_knowledge_records.id'))
        
        insight_category = Column(String(50), index=True)
        statement = Column(Text)
        formal_representation = Column(Text)
        proof_sketch = Column(Text)
        confidence_score = Column(Float)
        verified = Column(Integer, default=0)
        applications = Column(JSON)
        
        knowledge_record = relationship("Z3KnowledgeRecord", back_populates="insight_record")
    
    class Z3SolverExecutionRecord(Base):
        """Solver execution history."""
        __tablename__ = 'z3_solver_executions'
        
        id = Column(Integer, primary_key=True)
        execution_id = Column(String(100), unique=True, index=True)
        problem_hash = Column(String(64), index=True)
        problem_statement = Column(Text)
        problem_type = Column(String(50), index=True)
        
        result_status = Column(String(20), index=True)
        solving_time_ms = Column(Integer)
        memory_usage_mb = Column(Float)
        
        constraints_used = Column(JSON)
        tactics_used = Column(JSON)
        strategy_id = Column(String(100))
        
        model_assignments = Column(JSON)
        proof_steps = Column(JSON)
        
        created_at = Column(DateTime, default=datetime.utcnow)
        
        __table_args__ = (
            Index('idx_exec_problem', 'problem_hash', 'result_status'),
            Index('idx_exec_time', 'solving_time_ms'),
        )
    
    class Z3KnowledgeConflict(Base):
        """Record of knowledge conflicts."""
        __tablename__ = 'z3_knowledge_conflicts'
        
        id = Column(Integer, primary_key=True)
        conflict_type = Column(String(50))
        record_1_id = Column(Integer, ForeignKey('z3_knowledge_records.id'))
        record_2_id = Column(Integer, ForeignKey('z3_knowledge_records.id'))
        conflict_description = Column(Text)
        resolution_strategy = Column(String(100))
        resolved = Column(Boolean, default=False)
        created_at = Column(DateTime, default=datetime.utcnow)
        resolved_at = Column(DateTime)


@dataclass
class ExtractedFeatures:
    """Comprehensive features extracted from solver results."""
    # Problem features
    problem_hash: str
    problem_type: str
    problem_size: int
    constraint_count: int
    variable_count: int
    
    # Structural features
    max_constraint_complexity: float
    avg_constraint_complexity: float
    linear_constraint_ratio: float
    nonlinear_constraint_count: int
    boolean_variable_count: int
    integer_variable_count: int
    real_variable_count: int
    
    # Statistical features
    constraint_density: float
    variable_connectivity: List[Tuple[str, int]]
    
    # Result features
    solving_time_ms: float
    memory_usage_mb: float
    result_status: str
    proof_depth: int
    tactic_count: int
    
    # Derived features
    difficulty_estimate: float = 0.0
    recommended_timeout: float = 30.0
    
    def to_vector(self) -> List[float]:
        """Convert features to numerical vector."""
        return [
            self.problem_size,
            self.constraint_count,
            self.variable_count,
            self.max_constraint_complexity,
            self.avg_constraint_complexity,
            self.linear_constraint_ratio,
            self.nonlinear_constraint_count,
            self.solving_time_ms,
            self.memory_usage_mb,
            self.proof_depth,
            self.tactic_count,
            self.difficulty_estimate
        ]


class FeatureExtractionPipeline:
    """Comprehensive feature extraction from Z3 problems and results."""
    
    def __init__(self):
        self.feature_cache: Dict[str, ExtractedFeatures] = {}
        self.extraction_stats = {
            "total_extractions": 0,
            "cache_hits": 0,
            "failed_extractions": 0
        }
    
    def extract_features(
        self,
        problem_statement: str,
        constraints: List[str],
        result: Any,
        proof: Optional[str] = None
    ) -> ExtractedFeatures:
        """
        Extract comprehensive features from problem and result.
        
        Args:
            problem_statement: Original problem
            constraints: List of constraint expressions
            result: Solver result
            proof: Optional proof trace
            
        Returns:
            Comprehensive extracted features
        """
        problem_hash = self._hash_problem(problem_statement)
        
        # Check cache
        if problem_hash in self.feature_cache:
            self.extraction_stats["cache_hits"] += 1
            return self.feature_cache[problem_hash]
        
        try:
            # Extract structural features
            structural = self._extract_structural_features(constraints)
            
            # Extract variable features
            variables = self._extract_variable_features(constraints)
            
            # Extract result features
            result_features = self._extract_result_features(result, proof)
            
            # Calculate difficulty
            difficulty = self._estimate_difficulty(structural, variables, result_features)
            
            features = ExtractedFeatures(
                problem_hash=problem_hash,
                problem_type=self._classify_problem_type(constraints),
                problem_size=len(problem_statement),
                constraint_count=len(constraints),
                variable_count=variables["total"],
                max_constraint_complexity=structural["max_complexity"],
                avg_constraint_complexity=structural["avg_complexity"],
                linear_constraint_ratio=structural["linear_ratio"],
                nonlinear_constraint_count=structural["nonlinear_count"],
                boolean_variable_count=variables["bool_count"],
                integer_variable_count=variables["int_count"],
                real_variable_count=variables["real_count"],
                constraint_density=structural["density"],
                variable_connectivity=variables["connectivity"],
                solving_time_ms=getattr(result, 'solving_time', 0) * 1000,
                memory_usage_mb=getattr(result, 'memory_usage', 0),
                result_status=getattr(result, 'status', 'unknown'),
                proof_depth=self._calculate_proof_depth(proof),
                tactic_count=self._count_tactics(proof),
                difficulty_estimate=difficulty,
                recommended_timeout=self._recommend_timeout(difficulty)
            )
            
            # Cache features
            self.feature_cache[problem_hash] = features
            self.extraction_stats["total_extractions"] += 1
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            self.extraction_stats["failed_extractions"] += 1
            # Return default features
            return ExtractedFeatures(
                problem_hash=problem_hash,
                problem_type="unknown",
                problem_size=len(problem_statement),
                constraint_count=len(constraints),
                variable_count=0,
                max_constraint_complexity=0.0,
                avg_constraint_complexity=0.0,
                linear_constraint_ratio=0.0,
                nonlinear_constraint_count=0,
                boolean_variable_count=0,
                integer_variable_count=0,
                real_variable_count=0,
                constraint_density=0.0,
                variable_connectivity=[],
                solving_time_ms=0.0,
                memory_usage_mb=0.0,
                result_status="error",
                proof_depth=0,
                tactic_count=0
            )
    
    def _hash_problem(self, problem: str) -> str:
        """Create hash of problem statement."""
        return hashlib.sha256(problem.encode()).hexdigest()[:32]
    
    def _extract_structural_features(self, constraints: List[str]) -> Dict[str, Any]:
        """Extract structural features from constraints."""
        complexities = []
        linear_count = 0
        nonlinear_count = 0
        
        for constraint in constraints:
            # Calculate complexity
            complexity = constraint.count('(') + constraint.count(')')
            complexity += len(re.findall(r'[\+\-\*/^]', constraint))
            complexities.append(complexity)
            
            # Check if linear
            if self._is_linear_constraint(constraint):
                linear_count += 1
            else:
                nonlinear_count += 1
        
        total = len(constraints) if constraints else 1
        
        return {
            "max_complexity": max(complexities) if complexities else 0,
            "avg_complexity": sum(complexities) / total if complexities else 0,
            "linear_ratio": linear_count / total,
            "nonlinear_count": nonlinear_count,
            "density": sum(complexities) / (total * 10) if complexities else 0
        }
    
    def _is_linear_constraint(self, constraint: str) -> bool:
        """Check if constraint is linear."""
        # Check for nonlinear operators
        nonlinear_ops = ['*', '/', '^', '**']
        has_nonlinear = any(op in constraint for op in nonlinear_ops)
        
        # Check for multiplication of variables (xy or x*y)
        var_mult = re.search(r'[a-zA-Z_]\w*\s*\*\s*[a-zA-Z_]\w*', constraint)
        
        return not (has_nonlinear or var_mult)
    
    def _extract_variable_features(self, constraints: List[str]) -> Dict[str, Any]:
        """Extract variable features."""
        all_vars = set()
        bool_count = 0
        int_count = 0
        real_count = 0
        
        var_connections = defaultdict(int)
        
        for constraint in constraints:
            # Extract variables
            vars_in_constraint = set(re.findall(r'\b[a-zA-Z_]\w*\b', constraint))
            vars_in_constraint -= {'and', 'or', 'not', 'forall', 'exists', 'assert'}
            
            all_vars.update(vars_in_constraint)
            
            # Count connections (which variables appear together)
            for v1 in vars_in_constraint:
                var_connections[v1] += 1
                for v2 in vars_in_constraint:
                    if v1 != v2:
                        var_connections[v1] += 1
            
            # Classify variables by usage
            for var in vars_in_constraint:
                if self._is_boolean_var(constraint, var):
                    bool_count += 1
                elif self._is_integer_var(constraint, var):
                    int_count += 1
                else:
                    real_count += 1
        
        return {
            "total": len(all_vars),
            "bool_count": bool_count,
            "int_count": int_count,
            "real_count": real_count,
            "connectivity": sorted(
                [(v, c) for v, c in var_connections.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def _is_boolean_var(self, constraint: str, var: str) -> bool:
        """Check if variable is used as boolean."""
        bool_contexts = ['and', 'or', 'not', '=>', 'iff', '=']
        return any(ctx in constraint.lower() for ctx in bool_contexts)
    
    def _is_integer_var(self, constraint: str, var: str) -> bool:
        """Check if variable is likely integer."""
        int_indicators = ['Int', 'int', 'Nat', 'nat', 'ℤ', 'ℕ']
        return any(ind in constraint for ind in int_indicators)
    
    def _extract_result_features(self, result: Any, proof: Optional[str]) -> Dict[str, Any]:
        """Extract features from solver result."""
        return {
            "solving_time": getattr(result, 'solving_time', 0),
            "memory_usage": getattr(result, 'memory_usage', 0),
            "status": getattr(result, 'status', 'unknown'),
            "proof_depth": self._calculate_proof_depth(proof),
            "tactic_count": self._count_tactics(proof)
        }
    
    def _calculate_proof_depth(self, proof: Optional[str]) -> int:
        """Calculate depth of proof tree."""
        if not proof:
            return 0
        # Count nesting level
        max_depth = 0
        current_depth = 0
        for char in proof:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth
    
    def _count_tactics(self, proof: Optional[str]) -> int:
        """Count number of tactics in proof."""
        if not proof:
            return 0
        # Count common tactic patterns
        tactics = re.findall(r'\b(apply|simp|rewrite|intro|exact|use|have|show|by)\b', proof)
        return len(tactics)
    
    def _classify_problem_type(self, constraints: List[str]) -> str:
        """Classify problem type from constraints."""
        constraint_text = ' '.join(constraints).lower()
        
        if 'forall' in constraint_text or 'exists' in constraint_text:
            return "quantified"
        elif any(op in constraint_text for op in ['*', 'pow', '^', '**']):
            return "nonlinear"
        elif any(op in constraint_text for op in ['+', '-']):
            return "linear"
        elif any(op in constraint_text for op in ['and', 'or', 'not']):
            return "boolean"
        else:
            return "general"
    
    def _estimate_difficulty(
        self,
        structural: Dict[str, Any],
        variables: Dict[str, Any],
        result: Dict[str, Any]
    ) -> float:
        """Estimate problem difficulty (0-1 scale)."""
        difficulty = 0.0
        
        # Based on complexity
        difficulty += min(structural["avg_complexity"] / 10, 0.3)
        
        # Based on variable count
        difficulty += min(variables["total"] / 50, 0.2)
        
        # Based on nonlinear ratio
        difficulty += structural["linear_ratio"] * 0.2
        
        # Based on solving time
        if result["solving_time"] > 0:
            difficulty += min(result["solving_time"] / 60, 0.3)
        
        return min(difficulty, 1.0)
    
    def _recommend_timeout(self, difficulty: float) -> float:
        """Recommend timeout based on difficulty."""
        if difficulty < 0.3:
            return 10.0
        elif difficulty < 0.6:
            return 30.0
        elif difficulty < 0.8:
            return 60.0
        else:
            return 300.0


class Z3KnowledgePersistence:
    """Complete persistence layer for Z3 knowledge."""
    
    def __init__(
        self,
        database_url: str = "sqlite:///z3_knowledge_complete.db",
        redis_url: Optional[str] = None
    ):
        self.database_url = database_url
        self.redis_url = redis_url
        self.engine = None
        self.Session = None
        self.redis_client = None
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize persistence layer."""
        if self._initialized:
            return
        
        # Initialize database
        if SQLALCHEMY_AVAILABLE:
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"Database initialized: {self.database_url}")
        
        # Initialize Redis
        if REDIS_AVAILABLE and self.redis_url:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis cache initialized")
        
        self._initialized = True
    
    async def store_knowledge(
        self,
        record_type: str,
        content: Dict[str, Any],
        features: ExtractedFeatures,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store knowledge record."""
        if not SQLALCHEMY_AVAILABLE:
            return ""
        
        try:
            session = self.Session()
            
            # Create record hash
            content_str = json.dumps(content, sort_keys=True)
            record_hash = hashlib.sha256(content_str.encode()).hexdigest()[:32]
            
            # Check for existing
            existing = session.query(Z3KnowledgeRecord).filter_by(
                record_hash=record_hash
            ).first()
            
            if existing:
                # Update access count
                existing.access_count += 1
                existing.last_accessed = datetime.utcnow()
                session.commit()
                return record_hash
            
            # Create new record
            record = Z3KnowledgeRecord(
                record_type=record_type,
                record_hash=record_hash,
                content=content_str,
                features=asdict(features),
                metadata=metadata or {},
                source_problem=features.problem_hash,
                problem_domain=features.problem_type,
                confidence=metadata.get("confidence", 1.0) if metadata else 1.0
            )
            
            session.add(record)
            session.flush()  # Get ID
            
            # Create type-specific record
            if record_type == "proof_pattern":
                self._store_proof_pattern(session, record.id, content)
            elif record_type == "constraint_pattern":
                self._store_constraint_pattern(session, record.id, content)
            elif record_type == "strategy":
                self._store_strategy(session, record.id, content)
            elif record_type == "insight":
                self._store_insight(session, record.id, content)
            
            session.commit()
            
            # Cache in Redis
            if self.redis_client:
                await self.redis_client.setex(
                    f"z3:knowledge:{record_hash}",
                    timedelta(hours=24),
                    pickle.dumps(content)
                )
            
            logger.info(f"Stored {record_type} knowledge: {record_hash}")
            return record_hash
            
        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
            session.rollback()
            return ""
        finally:
            session.close()
    
    def _store_proof_pattern(self, session: Session, record_id: int, content: Dict):
        """Store proof pattern details."""
        pattern = Z3ProofPatternRecord(
            knowledge_record_id=record_id,
            pattern_signature=content.get("signature", ""),
            tactic_sequence=content.get("tactics", []),
            proof_tree_structure=content.get("tree_structure", {}),
            applicable_domains=content.get("domains", []),
            proof_depth=content.get("depth", 0),
            branching_factor=content.get("branching", 1.0)
        )
        session.add(pattern)
    
    def _store_constraint_pattern(self, session: Session, record_id: int, content: Dict):
        """Store constraint pattern details."""
        pattern = Z3ConstraintPatternRecord(
            knowledge_record_id=record_id,
            pattern_type=content.get("type", "unknown"),
            structure_template=content.get("template", ""),
            variables_pattern=content.get("variables", []),
            constraint_count=content.get("count", 0),
            complexity_score=content.get("complexity", 0.0)
        )
        session.add(pattern)
    
    def _store_strategy(self, session: Session, record_id: int, content: Dict):
        """Store strategy details."""
        strategy = Z3StrategyRecord(
            knowledge_record_id=record_id,
            strategy_name=content.get("name", ""),
            problem_pattern=content.get("pattern", ""),
            feature_vector=content.get("features", []),
            recommended_tactics=content.get("tactics", []),
            solver_config=content.get("config", {})
        )
        session.add(strategy)
    
    def _store_insight(self, session: Session, record_id: int, content: Dict):
        """Store insight details."""
        insight = Z3InsightRecord(
            knowledge_record_id=record_id,
            insight_category=content.get("category", "general"),
            statement=content.get("statement", ""),
            formal_representation=content.get("formal", ""),
            confidence_score=content.get("confidence", 0.5)
        )
        session.add(insight)
    
    async def retrieve_knowledge(
        self,
        record_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve knowledge by hash."""
        # Try Redis first
        if self.redis_client:
            try:
                cached = await self.redis_client.get(f"z3:knowledge:{record_hash}")
                if cached:
                    return pickle.loads(cached)
            except Exception:
                pass
        
        # Query database
        if SQLALCHEMY_AVAILABLE and self.Session:
            try:
                session = self.Session()
                record = session.query(Z3KnowledgeRecord).filter_by(
                    record_hash=record_hash
                ).first()
                
                if record:
                    # Update access stats
                    record.access_count += 1
                    record.last_accessed = datetime.utcnow()
                    session.commit()
                    
                    return json.loads(record.content)
                    
            except Exception as e:
                logger.error(f"Failed to retrieve knowledge: {e}")
            finally:
                session.close()
        
        return None
    
    async def find_similar_knowledge(
        self,
        problem_features: ExtractedFeatures,
        record_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar knowledge using feature matching."""
        if not SQLALCHEMY_AVAILABLE or not self.Session:
            return []
        
        try:
            session = self.Session()
            
            query = session.query(Z3KnowledgeRecord)
            
            if record_type:
                query = query.filter_by(record_type=record_type)
            
            # Filter by problem domain
            query = query.filter_by(problem_domain=problem_features.problem_type)
            
            # Order by confidence and access count
            query = query.order_by(
                Z3KnowledgeRecord.confidence.desc(),
                Z3KnowledgeRecord.access_count.desc()
            )
            
            records = query.limit(top_k * 2).all()  # Get more for filtering
            
            # Calculate feature similarity
            results = []
            for record in records:
                if record.features:
                    similarity = self._calculate_feature_similarity(
                        problem_features.to_vector(),
                        record.features
                    )
                    if similarity > 0.5:  # Threshold
                        results.append({
                            "hash": record.record_hash,
                            "content": json.loads(record.content),
                            "similarity": similarity,
                            "confidence": record.confidence
                        })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar knowledge: {e}")
            return []
        finally:
            session.close()
    
    def _calculate_feature_similarity(
        self,
        features1: List[float],
        features2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between feature vectors."""
        try:
            # Extract vector from features2
            vec2 = [
                features2.get("problem_size", 0),
                features2.get("constraint_count", 0),
                features2.get("variable_count", 0),
                features2.get("max_constraint_complexity", 0),
                features2.get("avg_constraint_complexity", 0),
                features2.get("linear_constraint_ratio", 0),
                features2.get("nonlinear_constraint_count", 0),
                features2.get("solving_time_ms", 0),
                features2.get("memory_usage_mb", 0),
                features2.get("proof_depth", 0),
                features2.get("tactic_count", 0),
                features2.get("difficulty_estimate", 0)
            ]
            
            # Cosine similarity
            if NUMPY_AVAILABLE:
                v1 = np.array(features1)
                v2 = np.array(vec2)
                
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return float(np.dot(v1, v2) / (norm1 * norm2))
            else:
                # Manual calculation
                dot = sum(a * b for a, b in zip(features1, vec2))
                norm1 = sum(a * a for a in features1) ** 0.5
                norm2 = sum(b * b for b in vec2) ** 0.5
                
                return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
                
        except Exception:
            return 0.0


class ConflictDetector:
    """Detect and resolve knowledge conflicts."""
    
    def __init__(self):
        self.conflict_rules = {
            "contradictory_patterns": self._detect_contradictory_patterns,
            "inconsistent_strategies": self._detect_inconsistent_strategies,
            "conflicting_insights": self._detect_conflicting_insights
        }
    
    def detect_conflicts(
        self,
        new_knowledge: Dict[str, Any],
        existing_knowledge: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect conflicts between new and existing knowledge."""
        conflicts = []
        
        for existing in existing_knowledge:
            for rule_name, rule_func in self.conflict_rules.items():
                conflict = rule_func(new_knowledge, existing)
                if conflict:
                    conflicts.append({
                        "type": rule_name,
                        "new_record": new_knowledge.get("hash"),
                        "existing_record": existing.get("hash"),
                        "description": conflict
                    })
        
        return conflicts
    
    def _detect_contradictory_patterns(
        self,
        new: Dict[str, Any],
        existing: Dict[str, Any]
    ) -> Optional[str]:
        """Detect contradictory proof patterns."""
        # Check if patterns have same conditions but different tactics
        new_tactics = set(new.get("tactics", []))
        existing_tactics = set(existing.get("tactics", []))
        
        # High similarity in conditions but different tactics
        if (new.get("pattern_type") == existing.get("pattern_type") and
            new_tactics != existing_tactics and
            len(new_tactics & existing_tactics) < len(new_tactics | existing_tactics) / 2):
            return "Same pattern type with different tactics"
        
        return None
    
    def _detect_inconsistent_strategies(
        self,
        new: Dict[str, Any],
        existing: Dict[str, Any]
    ) -> Optional[str]:
        """Detect inconsistent strategies."""
        # Check if strategies have same pattern but different success rates
        if (new.get("problem_pattern") == existing.get("problem_pattern") and
            abs(new.get("success_rate", 0) - existing.get("success_rate", 0)) > 0.5):
            return "Same problem pattern with significantly different success rates"
        
        return None
    
    def _detect_conflicting_insights(
        self,
        new: Dict[str, Any],
        existing: Dict[str, Any]
    ) -> Optional[str]:
        """Detect conflicting mathematical insights."""
        # Check for contradictory statements
        if (new.get("category") == existing.get("category") and
            new.get("statement") != existing.get("statement") and
            self._statement_similarity(new.get("statement", ""), existing.get("statement", "")) > 0.7):
            return "Similar insights with contradictory statements"
        
        return None
    
    def _statement_similarity(self, stmt1: str, stmt2: str) -> float:
        """Calculate similarity between statements."""
        words1 = set(stmt1.lower().split())
        words2 = set(stmt2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def resolve_conflict(
        self,
        conflict: Dict[str, Any],
        new_record: Dict[str, Any],
        existing_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve a detected conflict."""
        # Default resolution: keep the one with higher confidence/success rate
        new_score = new_record.get("confidence", 0.5) * new_record.get("success_rate", 0.5)
        existing_score = existing_record.get("confidence", 0.5) * existing_record.get("success_rate", 0.5)
        
        if new_score > existing_score:
            return {"action": "replace", "keep": "new", "discard": "existing"}
        else:
            return {"action": "keep_existing", "keep": "existing", "discard": "new"}


class Z3KnowledgeManager:
    """
    Complete Z3 Knowledge Management System.
    
    Provides:
    - Full database persistence
    - Comprehensive feature extraction
    - Online learning
    - Conflict detection
    - Performance monitoring
    - CAV-NLP enhanced knowledge extraction
    """
    
    def __init__(
        self,
        database_url: str = "sqlite:///z3_knowledge_complete.db",
        redis_url: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        self.persistence = Z3KnowledgePersistence(database_url, redis_url)
        self.feature_pipeline = FeatureExtractionPipeline()
        self.conflict_detector = ConflictDetector()
        self.knowledge_extractor = Z3KnowledgeExtractor()
        
        # CAV-NLP configuration
        self.config = config or {}
        self.use_cav_nlp = self.config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        self.math_service = None
        self.enhanced_solver = None
        if self.use_cav_nlp:
            try:
                self.math_service = UnifiedMathService()
                self.enhanced_solver = EnhancedZ3Solver()
                logger.info("CAV-NLP enhanced knowledge manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP: {e}")
                self.use_cav_nlp = False
        
        # Metrics
        self.metrics = {
            "knowledge_stored": 0,
            "knowledge_retrieved": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cav_nlp_extractions": 0
        }
    
    async def initialize(self):
        """Initialize the knowledge manager."""
        await self.persistence.initialize()
        logger.info("Z3KnowledgeManager initialized")
    
    async def learn_from_solution(
        self,
        problem_statement: str,
        constraints: List[str],
        result: Any,
        proof: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Learn from a successful solution.
        
        Args:
            problem_statement: Original problem
            constraints: Problem constraints
            result: Solver result
            proof: Optional proof trace
            metadata: Additional metadata
            
        Returns:
            Learning results
        """
        try:
            # CAV-NLP enhanced formalization
            cav_nlp_result = None
            if self.use_cav_nlp and self.math_service:
                try:
                    cav_nlp_result = self.math_service.formalize(problem_statement)
                    self.metrics["cav_nlp_extractions"] += 1
                    logger.debug(f"CAV-NLP formalized: {cav_nlp_result}")
                except Exception as e:
                    logger.debug(f"CAV-NLP formalization skipped: {e}")
            
            # Extract features
            features = self.feature_pipeline.extract_features(
                problem_statement, constraints, result, proof
            )
            
            learned_items = []
            
            # Extract and store proof patterns
            if proof:
                patterns = self.knowledge_extractor.extract_proof_patterns(
                    proof, features.problem_type
                )
                for pattern in patterns:
                    content = {
                        "signature": pattern.pattern_id,
                        "tactics": pattern.tactic_sequence,
                        "domains": pattern.applicable_domains,
                        "depth": getattr(pattern, 'proof_depth', 0)
                    }
                    
                    record_hash = await self._store_with_conflict_check(
                        "proof_pattern", content, features, metadata
                    )
                    
                    if record_hash:
                        learned_items.append({"type": "proof_pattern", "hash": record_hash})
            
            # Extract and store constraint patterns
            constraint_patterns = self.knowledge_extractor.analyze_constraints(
                constraints,
                features.solving_time_ms / 1000,
                result.status == "success" if hasattr(result, 'status') else True
            )
            
            for pattern in constraint_patterns:
                content = {
                    "type": pattern.pattern_type,
                    "template": pattern.structure,
                    "variables": pattern.variables_involved,
                    "complexity": pattern.complexity_score
                }
                
                record_hash = await self._store_with_conflict_check(
                    "constraint_pattern", content, features, metadata
                )
                
                if record_hash:
                    learned_items.append({"type": "constraint_pattern", "hash": record_hash})
            
            # Learn strategy if successful
            if getattr(result, 'success', False):
                strategy = self.knowledge_extractor.learn_strategy(
                    {
                        "type": features.problem_type,
                        "constraint_count": features.constraint_count,
                        "var_count": features.variable_count
                    },
                    getattr(result, 'tactics_used', []),
                    getattr(result, 'config', {}),
                    True,
                    features.solving_time_ms / 1000
                )
                
                content = {
                    "name": strategy.name,
                    "pattern": strategy.problem_pattern,
                    "tactics": strategy.recommended_tactics,
                    "config": strategy.solver_configuration,
                    "features": features.to_vector()
                }
                
                record_hash = await self._store_with_conflict_check(
                    "strategy", content, features, metadata
                )
                
                if record_hash:
                    learned_items.append({"type": "strategy", "hash": record_hash})
            
            self.metrics["knowledge_stored"] += len(learned_items)
            
            return {
                "success": True,
                "items_learned": len(learned_items),
                "items": learned_items,
                "features": asdict(features),
                "cav_nlp_formalization": getattr(cav_nlp_result, 'code', None) if cav_nlp_result else None
            }
            
        except Exception as e:
            logger.error(f"Learning failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def extract_with_cav_nlp(self, text: str) -> Dict[str, Any]:
        """
        Extract knowledge using CAV-NLP enhancement.
        
        Args:
            text: Natural language text to formalize
            
        Returns:
            Dictionary with formalized result
        """
        if not self.use_cav_nlp or not self.math_service:
            return {"error": "CAV-NLP not available"}
        
        try:
            formalized = self.math_service.formalize(text)
            self.metrics["cav_nlp_extractions"] += 1
            return {
                "success": True,
                "original": text,
                "formalized": getattr(formalized, 'code', str(formalized)),
                "language": getattr(formalized, 'language', 'unknown'),
                "confidence": getattr(formalized, 'confidence', 0.0)
            }
        except Exception as e:
            logger.error(f"CAV-NLP extraction failed: {e}")
            return {"error": str(e)}
    
    async def _store_with_conflict_check(
        self,
        record_type: str,
        content: Dict[str, Any],
        features: ExtractedFeatures,
        metadata: Optional[Dict]
    ) -> str:
        """Store knowledge with conflict detection."""
        # Find similar existing knowledge
        similar = await self.persistence.find_similar_knowledge(
            features, record_type, top_k=5
        )
        
        # Check for conflicts
        if similar:
            conflicts = self.conflict_detector.detect_conflicts(content, similar)
            
            if conflicts:
                self.metrics["conflicts_detected"] += len(conflicts)
                
                for conflict in conflicts:
                    resolution = self.conflict_detector.resolve_conflict(
                        conflict, content, similar[0]
                    )
                    
                    if resolution["action"] == "keep_existing":
                        logger.info(f"Conflict resolved: keeping existing record")
                        return ""
                    
                    self.metrics["conflicts_resolved"] += 1
        
        # Store the knowledge
        record_hash = await self.persistence.store_knowledge(
            record_type, content, features, metadata
        )
        
        return record_hash
    
    async def find_similar_solutions(
        self,
        problem_statement: str,
        constraints: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar solutions from knowledge base."""
        # Extract features for query
        features = self.feature_pipeline.extract_features(
            problem_statement, constraints, type('Result', (), {'status': 'unknown'})()
        )
        
        # Search knowledge base
        similar = await self.persistence.find_similar_knowledge(
            features, top_k=top_k
        )
        
        self.metrics["knowledge_retrieved"] += len(similar)
        
        return similar
    
    async def get_recommended_strategy(
        self,
        problem_statement: str,
        constraints: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Get recommended strategy for problem."""
        # Extract features
        features = self.feature_pipeline.extract_features(
            problem_statement, constraints, type('Result', (), {})()
        )
        
        # Find similar strategies
        strategies = await self.persistence.find_similar_knowledge(
            features, record_type="strategy", top_k=3
        )
        
        if strategies:
            # Return best strategy
            best = max(strategies, key=lambda s: s.get("confidence", 0))
            return best["content"]
        
        # Fall back to knowledge extractor
        strategy = self.knowledge_extractor.recommend_strategy({
            "type": features.problem_type,
            "constraint_count": features.constraint_count,
            "var_count": features.variable_count
        })
        
        return strategy.to_dict() if strategy else None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get knowledge manager metrics."""
        return {
            **self.metrics,
            "feature_extraction": self.feature_pipeline.extraction_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge manager statistics (alias for get_metrics)."""
        return self.get_metrics()


# Global instance
_knowledge_manager: Optional[Z3KnowledgeManager] = None


async def get_z3_knowledge_manager() -> Z3KnowledgeManager:
    """Get global knowledge manager instance."""
    global _knowledge_manager
    if _knowledge_manager is None:
        _knowledge_manager = Z3KnowledgeManager()
        await _knowledge_manager.initialize()
    return _knowledge_manager


# Example usage
async def example_complete_integration():
    """Example: Complete integration."""
    print("Complete Z3 Knowledge Integration Example")
    print("=" * 60)
    
    manager = await get_z3_knowledge_manager()
    
    # Learn from a solution
    problem = "Linear equation system"
    constraints = ["(> x 0)", "(< x 10)", "(= y (+ x 5))"]
    
    class MockResult:
        success = True
        status = "sat"
        solving_time = 1.5
        model = type('Model', (), {'assignments': {'x': 5, 'y': 10}})()
        tactics_used = ["simplify", "solve-eqs", "smt"]
    
    result = await manager.learn_from_solution(
        problem, constraints, MockResult(),
        proof="(simplify (solve-eqs (smt)))"
    )
    
    print(f"\nLearning result:")
    print(f"  Items learned: {result['items_learned']}")
    print(f"  Features: {list(result['features'].keys())[:5]}")
    
    # Find similar solutions
    similar = await manager.find_similar_solutions(problem, constraints)
    print(f"\nSimilar solutions found: {len(similar)}")
    
    # Get metrics
    metrics = manager.get_metrics()
    print(f"\nMetrics:")
    print(f"  Knowledge stored: {metrics['knowledge_stored']}")
    print(f"  Feature extractions: {metrics['feature_extraction']['total_extractions']}")


if __name__ == "__main__":
    asyncio.run(example_complete_integration())
