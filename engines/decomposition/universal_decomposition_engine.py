"""
Universal Decomposition Engine - Industry-Agnostic Problem Decomposition System

This module implements a comprehensive decomposition system that works for any domain:
- Software Engineering
- Finance/Banking
- Scientific Research
- Healthcare
- Manufacturing
- Legal/Compliance
- Business Strategy
- Education
- And more...

Core Philosophy:
    Any complex problem can be decomposed into atomic, manageable sub-problems
    that can be solved independently and then reassembled into a complete solution.

Architecture:
    1. Problem Analysis -> Understand domain, complexity, constraints
    2. Strategy Selection -> Choose best decomposition approach
    3. Decomposition -> Break into sub-problems with dependencies
    4. Execution -> Solve each sub-problem (using appropriate teams/methods)
    5. Reassembly -> Combine solutions with conflict resolution
    6. Validation -> Verify final solution meets success criteria

Usage:
    >>> from universal_decomposition_engine import UniversalDecompositionEngine
    >>> engine = UniversalDecompositionEngine()
    >>> result = engine.decompose(
    ...     problem_statement="Build a trading risk management system",
    ...     domain="finance",
    ...     constraints=["regulatory_compliance", "real_time_processing"]
    ... )
"""
from __future__ import annotations


import logging
import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from collections import defaultdict, deque
import hashlib
try:
    from pydantic import BaseModel, Field, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    BaseModel = object  # type: ignore
    Field = lambda *args, **kwargs: None  # type: ignore
    ValidationError = Exception  # type: ignore
    PYDANTIC_AVAILABLE = False
try:
    from utils.entanglement_utils import (
        build_symbolic_entanglement_matrix,
        serialize_entanglement_matrix,
    )
except ImportError:
    import json as _json

    def build_symbolic_entanglement_matrix(sub_problems, allowed_ids=None, enforce_symmetry=True, strict=False):
        ids = list(allowed_ids) if allowed_ids is not None else [
            getattr(sp, "id", "sp_{0}".format(i)) for i, sp in enumerate(sub_problems or [])
        ]
        matrix = {sid: {oid: 1.0 if sid == oid else 0.0 for oid in ids} for sid in ids}
        symbols_by_id = {sid: set() for sid in ids}
        return matrix, symbols_by_id

    def serialize_entanglement_matrix(matrix):
        return _json.dumps(matrix, default=str)

# Configure logging
logger = logging.getLogger(__name__)

# Optional DSPy integration for structured prompting
try:
    from dspy_integration import DSPY_AVAILABLE
    import dspy
except ImportError:  # pragma: no cover - optional dependency
    DSPY_AVAILABLE = False
    dspy = None

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Universal Decomposition Engine
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

# **LEAN INTEGRATION**: Formal verification
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False
    LeanAideClient = None  # type: ignore


# ============================================================================
# ENUMS - Core Type Definitions
# ============================================================================

class ProblemDomain(Enum):
    """Supported problem domains"""
    SOFTWARE = "software"
    FINANCE = "finance"
    WEB3 = "web3"
    SCIENTIFIC = "scientific"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    LEGAL = "legal"
    BUSINESS = "business"
    EDUCATION = "education"
    GENERIC = "generic"


class DecompositionStrategy(Enum):
    """Available decomposition strategies"""
    HIERARCHICAL = "hierarchical"      # Top-down functional decomposition
    FUNCTIONAL = "functional"          # By system capabilities
    SEMANTIC = "semantic"              # By meaning and concepts
    STRUCTURAL = "structural"          # By physical/organizational structure
    DEPENDENCY = "dependency"          # Based on prerequisite relationships
    COMPLEXITY = "complexity"          # To balance cognitive load
    TEMPORAL = "temporal"              # By chronological order
    RISK_BASED = "risk_based"          # Address highest risks first
    VALUE_BASED = "value_based"        # Deliver highest value first
    HYBRID = "hybrid"                  # Adaptive combination


class SubProblemType(Enum):
    """Types of sub-problems"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    DESIGN = "design"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


class SubProblemStatus(Enum):
    """Status of sub-problem resolution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SOLVED = "solved"
    FAILED = "failed"
    BLOCKED = "blocked"
    ERROR = "error"


class AssemblyStrategy(Enum):
    """Strategies for reassembling solutions"""
    HIERARCHICAL = "hierarchical"      # Bottom-up tree assembly
    LINEAR = "linear"                  # Sequential assembly
    PARALLEL = "parallel"              # Join completed sub-solutions
    ADAPTIVE = "adaptive"              # Context-aware assembly
    ROMA_DETERMINISTIC = "roma_deterministic"  # ROMA verbatim mode
    ROMA_CREATIVE = "roma_creative"    # ROMA enhanced mode


# ============================================================================
# DATA CLASSES - Core Data Models
# ============================================================================

@dataclass
class Constraint:
    """Represents a problem constraint"""
    id: str
    description: str
    type: str  # time, resource, quality, technical, regulatory
    severity: str  # hard, soft
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuccessCriterion:
    """Defines measurable success criteria"""
    id: str
    description: str
    metric: str
    threshold: float
    validation_method: str = "automatic"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComplexityScore:
    """Multi-dimensional complexity assessment (0-10 scale)"""
    cognitive_complexity: float
    computational_complexity: float
    domain_complexity: float
    integration_complexity: float
    overall_complexity: float
    explanation: str = ""
    
    def __post_init__(self):
        """Ensure all scores are within valid range"""
        for field_name in ['cognitive_complexity', 'computational_complexity', 
                          'domain_complexity', 'integration_complexity', 'overall_complexity']:
            value = getattr(self, field_name)
            setattr(self, field_name, max(0.0, min(10.0, float(value))))


@dataclass
class ProblemDefinition:
    """Complete problem definition"""
    id: str
    title: str
    description: str
    domain: ProblemDomain
    complexity_score: ComplexityScore
    constraints: List[Constraint] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['domain'] = self.domain.value
        data['complexity_score'] = asdict(self.complexity_score)
        data['constraints'] = [c.to_dict() for c in self.constraints]
        data['success_criteria'] = [s.to_dict() for s in self.success_criteria]
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class SubProblem:
    """Atomic sub-problem"""
    id: str
    parent_id: str
    title: str
    description: str
    type: SubProblemType
    complexity_score: ComplexityScore
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    estimated_effort_hours: float = 1.0
    priority: int = 5  # 1-10
    status: SubProblemStatus = SubProblemStatus.PENDING
    domain_specific_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['type'] = self.type.value
        data['status'] = self.status.value
        return data


@dataclass
class DecompositionPlan:
    """Complete decomposition plan"""
    id: str
    original_problem: ProblemDefinition
    sub_problems: List[SubProblem]
    strategy_used: DecompositionStrategy
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    quality_score: float = 0.0
    analyzed_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'original_problem': self.original_problem.to_dict(),
            'sub_problems': [sp.to_dict() for sp in self.sub_problems],
            'strategy_used': self.strategy_used.value,
            'dependency_graph': self.dependency_graph,
            'execution_order': self.execution_order,
            'parallel_groups': self.parallel_groups,
            'quality_score': self.quality_score,
            'analyzed_context': self.analyzed_context,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class SubProblemSolution:
    """Solution for a sub-problem"""
    sub_problem_id: str
    solution_content: str
    quality_score: float
    verification_status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedSolution:
    """Final integrated solution"""
    problem_id: str
    decomposition_plan_id: str
    assembled_content: str
    sub_solutions: Dict[str, SubProblemSolution]
    assembly_strategy: AssemblyStrategy
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    conflicts_detected: List[Dict] = field(default_factory=list)
    conflicts_resolved: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# STRATEGY BASE CLASSES
# ============================================================================

class DecompositionStrategyBase(ABC):
    """Abstract base for all decomposition strategies"""
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        raise NotImplementedError("DecompositionStrategyBase.get_strategy_name must be implemented")
    
    @abstractmethod
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose problem into sub-problems"""
        raise NotImplementedError("DecompositionStrategyBase.decompose must be implemented")
    
    def generate_id(self, prefix: str = "sub") -> str:
        """Generate unique ID"""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ============================================================================
# CONCRETE DECOMPOSITION STRATEGIES
# ============================================================================

class SemanticDecomposition(DecompositionStrategyBase):
    """
    Decomposes based on semantic meaning and conceptual boundaries.
    Uses keyword extraction and conceptual clustering.
    """
    
    # Domain-specific semantic patterns
    DOMAIN_PATTERNS = {
        ProblemDomain.SOFTWARE: [
            r'(?:user interface|UI|frontend|backend|API|database|authentication|security)',
            r'(?:microservice|module|component|service|endpoint|controller)',
        ],
        ProblemDomain.FINANCE: [
            r'(?:risk assessment|portfolio|trading|compliance|regulatory|audit)',
            r'(?:market data|pricing|valuation|settlement|clearing|reporting)',
            r'(?:credit risk|market risk|operational risk|liquidity risk)',
        ],
        ProblemDomain.WEB3: [
            r'(?:smart contract|solidity|evm|foundry|forge|slither|hardhat|rust|anchor)',
            r'(?:defi|vault|oracle|flash loan|reentrancy|amm|liquidity pool|bridge)',
            r'(?:invariant|symbolic execution|exploit|audit|bug bounty|onchain)',
        ],
        ProblemDomain.HEALTHCARE: [
            r'(?:patient|diagnosis|treatment|medication|clinical|epidemiology)',
            r'(?:electronic health record|EHR|HIPAA|privacy|consent)',
        ],
        ProblemDomain.SCIENTIFIC: [
            r'(?:hypothesis|experiment|data collection|analysis|validation)',
            r'(?:theoretical|empirical|computational|simulation|modeling)',
        ],
        ProblemDomain.GENERIC: [
            r'(?:planning|execution|monitoring|review|optimization)',
            r'(?:analysis|design|implementation|testing|deployment)',
        ]
    }
    
    def get_strategy_name(self) -> str:
        return "semantic"
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose based on semantic analysis"""
        self.logger.info(f"Semantic decomposition for: {problem.title}")
        
        # Get domain-specific patterns
        patterns = self.DOMAIN_PATTERNS.get(problem.domain, self.DOMAIN_PATTERNS[ProblemDomain.GENERIC])
        
        # Extract semantic components
        components = self._extract_components(problem.description, patterns)
        
        if not components:
            # Fallback: create single sub-problem
            return [self._create_single_subproblem(problem)]
        
        sub_problems = []
        for i, component in enumerate(components):
            sp = SubProblem(
                id=self.generate_id(f"sem_{i}"),
                parent_id=problem.id,
                title=component['title'],
                description=component['description'],
                type=SubProblemType.IMPLEMENTATION,
                complexity_score=self._estimate_complexity(component, problem),
                dependencies=[],
                success_criteria=self._derive_success_criteria(component, problem),
                estimated_effort_hours=self._estimate_effort(component, problem),
                priority=5
            )
            sub_problems.append(sp)
        
        # Infer dependencies based on semantic relationships
        sub_problems = self._infer_dependencies(sub_problems)
        
        return sub_problems
    
    def _extract_components(self, description: str, patterns: List[str]) -> List[Dict]:
        """Extract semantic components from description"""
        components = []
        
        # Split by common separators
        parts = re.split(r'[,;.]|\band\b|\bor\b', description)
        
        for i, part in enumerate(parts):
            part = part.strip()
            if len(part) < 20:
                continue
            
            # Check if matches any pattern
            matches_patterns = any(re.search(pattern, part, re.IGNORECASE) for pattern in patterns)
            
            component = {
                'title': self._generate_title(part),
                'description': part,
                'keywords': self._extract_keywords(part),
                'category': 'domain_specific' if matches_patterns else 'general'
            }
            components.append(component)
        
        return components
    
    def _generate_title(self, text: str) -> str:
        """Generate a title from text"""
        words = text.split()
        if len(words) <= 5:
            return text.title()
        return ' '.join(words[:5]).title() + '...'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        stop_words = {'with', 'from', 'that', 'this', 'have', 'been', 'their', 'will'}
        return list(set(w for w in words if w not in stop_words))[:5]
    
    def _estimate_complexity(self, component: Dict, problem: ProblemDefinition) -> ComplexityScore:
        """Estimate complexity for a component"""
        base_complexity = problem.complexity_score.overall_complexity
        
        # Adjust based on description length and keyword density
        length_factor = min(len(component['description']) / 200, 1.0)
        keyword_factor = len(component['keywords']) / 5.0
        
        adjusted = base_complexity * (0.7 + 0.3 * (length_factor + keyword_factor) / 2)
        
        return ComplexityScore(
            cognitive_complexity=min(adjusted * 0.9, 10),
            computational_complexity=min(adjusted * 0.8, 10),
            domain_complexity=min(adjusted * 0.9, 10),
            integration_complexity=min(adjusted * 0.7, 10),
            overall_complexity=min(adjusted, 10),
            explanation=f"Derived from parent complexity {base_complexity} with component analysis"
        )
    
    def _estimate_effort(self, component: Dict, problem: ProblemDefinition) -> float:
        """Estimate effort in hours"""
        base_effort = problem.complexity_score.overall_complexity * 4
        if component['category'] == 'domain_specific':
            base_effort *= 1.2
        return round(base_effort, 1)
    
    def _derive_success_criteria(self, component: Dict, problem: ProblemDefinition) -> List[SuccessCriterion]:
        """Derive success criteria from parent and component"""
        criteria = []
        
        # Inherit relevant parent criteria
        for pc in problem.success_criteria:
            if any(kw in pc.description.lower() for kw in component['keywords'][:2]):
                criteria.append(SuccessCriterion(
                    id=self.generate_id("sc"),
                    description=f"Component: {pc.description}",
                    metric=pc.metric,
                    threshold=pc.threshold
                ))
        
        # Add component-specific criterion
        criteria.append(SuccessCriterion(
            id=self.generate_id("sc"),
            description=f"Complete {component['title']}",
            metric="completion",
            threshold=0.95
        ))
        
        return criteria
    
    def _infer_dependencies(self, sub_problems: List[SubProblem]) -> List[SubProblem]:
        """Infer dependencies between sub-problems"""
        # Simple heuristic: if one sub-problem's keywords appear in another's description,
        # there might be a dependency
        for i, sp1 in enumerate(sub_problems):
            for j, sp2 in enumerate(sub_problems):
                if i == j:
                    continue
                keywords1 = set(sp1.description.lower().split())
                keywords2 = set(sp2.description.lower().split())
                overlap = keywords1 & keywords2
                
                # If significant overlap, add dependency
                if len(overlap) >= 2 and sp2.complexity_score.overall_complexity > sp1.complexity_score.overall_complexity:
                    if sp2.id not in sp1.dependencies:
                        sp1.dependencies.append(sp2.id)
        
        return sub_problems
    
    def _create_single_subproblem(self, problem: ProblemDefinition) -> SubProblem:
        """Create a single sub-problem (fallback)"""
        return SubProblem(
            id=self.generate_id("sem_0"),
            parent_id=problem.id,
            title=problem.title,
            description=problem.description,
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=problem.complexity_score,
            dependencies=[],
            success_criteria=problem.success_criteria,
            estimated_effort_hours=problem.complexity_score.overall_complexity * 4
        )


class DependencyDecomposition(DecompositionStrategyBase):
    """
    Decomposes based on prerequisite relationships and dependencies.
    Creates a dependency graph that respects execution order.
    """

    if PYDANTIC_AVAILABLE:
        class DependencyEdge(BaseModel):
            """Structured dependency edge output from LLM."""
            source_id: str = Field(..., description="Sub-problem that depends on target_id")
            target_id: str = Field(..., description="Prerequisite sub-problem")
            reason: str = Field(default="", description="Short rationale for dependency")
        class DependencyEdgeList(BaseModel):
            """Structured list of dependency edges."""
            edges: List["DependencyDecomposition.DependencyEdge"] = Field(default_factory=list)
    else:  # pragma: no cover - fallback
        class DependencyEdge:  # type: ignore
            def __init__(self, source_id: str, target_id: str, reason: str = ""):
                self.source_id = source_id
                self.target_id = target_id
                self.reason = reason
        class DependencyEdgeList:  # type: ignore
            def __init__(self, edges: Optional[List["DependencyDecomposition.DependencyEdge"]] = None):
                self.edges = edges or []
    
    def get_strategy_name(self) -> str:
        return "dependency"
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose based on dependency analysis"""
        self.logger.info(f"Dependency decomposition for: {problem.title}")
        
        # Start with semantic decomposition
        semantic = SemanticDecomposition(self.llm_client)
        base_subproblems = semantic.decompose(problem)
        
        if len(base_subproblems) <= 1:
            return base_subproblems
        
        # Analyze and enhance dependencies
        if self.llm_client:
            enhanced = self._analyze_dependencies_with_llm(base_subproblems, problem)
            if enhanced:
                return enhanced
        
        # Fallback: use heuristics
        return self._apply_heuristic_dependencies(base_subproblems, problem)
    
    def _analyze_dependencies_with_llm(
        self, sub_problems: List[SubProblem], problem: ProblemDefinition
    ) -> Optional[List[SubProblem]]:
        """Use LLM to analyze dependencies if available"""
        try:
            if not PYDANTIC_AVAILABLE:
                self.logger.warning("Pydantic unavailable; skipping structured dependency analysis.")
                return self._apply_heuristic_dependencies(sub_problems, problem)

            # Build descriptions for LLM
            descriptions = []
            for i, sp in enumerate(sub_problems, 1):
                descriptions.append(
                    f"{i}. [{sp.id}] {sp.title}: {sp.description[:150]}..."
                )

            prompt = (
                "Identify prerequisite dependencies between sub-problems.\n"
                "Return ONLY structured dependency edges with fields:\n"
                "- source_id: sub-problem that depends on target_id\n"
                "- target_id: prerequisite sub-problem\n"
                "- reason: short rationale\n\n"
                f"Problem: {problem.title}\n\n"
                "Sub-Problems:\n"
                f"{chr(10).join(descriptions)}\n"
            )

            # Attempt structured response first (Instructor/DSPy/response_model)
            structured_edges: List[DependencyDecomposition.DependencyEdge] = []
            response_model = self.DependencyEdgeList

            if self.llm_client:
                if hasattr(self.llm_client, "complete_structured"):
                    try:
                        result = self.llm_client.complete_structured(
                            prompt=prompt, response_model=response_model
                        )
                        if isinstance(result, self.DependencyEdgeList):
                            structured_edges = list(result.edges or [])
                        elif isinstance(result, list):
                            structured_edges = self._parse_dependency_edges(result)
                    except Exception as exc:
                        self.logger.debug("Structured dependency call failed: %s", exc)

                if not structured_edges:
                    for method_name in ("chat", "complete", "generate"):
                        method = getattr(self.llm_client, method_name, None)
                        if not callable(method):
                            continue
                        try:
                            result = method(prompt, response_model=response_model)
                            if isinstance(result, self.DependencyEdgeList):
                                structured_edges = list(result.edges or [])
                            elif isinstance(result, list):
                                structured_edges = self._parse_dependency_edges(result)
                            elif isinstance(result, dict) and "edges" in result:
                                structured_edges = self._parse_dependency_edges(result["edges"])
                            if structured_edges:
                                break
                        except TypeError:
                            continue
                        except Exception as exc:
                            self.logger.debug("Response-model dependency call failed: %s", exc)

            if structured_edges:
                return self._apply_dependency_edges(sub_problems, structured_edges)

            # Prefer DSPy structured prompting when available
            dspy_text = None
            if DSPY_AVAILABLE and dspy is not None:
                try:
                    class DependencySignature(dspy.Signature):
                        problem_title = dspy.InputField(desc="Problem title")
                        sub_problems_text = dspy.InputField(desc="Enumerated sub-problems with ids")
                        dependencies_json = dspy.OutputField(
                            desc="JSON list of dependency edges with source_id, target_id, reason"
                        )

                    predictor = dspy.Predict(DependencySignature)
                    result = predictor(
                        problem_title=problem.title,
                        sub_problems_text="\n".join(descriptions),
                    )
                    dspy_text = getattr(result, "dependencies_json", None)
                except Exception as exc:  # pragma: no cover - DSPy optional
                    self.logger.debug("DSPy dependency analysis failed: %s", exc)

            prompt = f"""Analyze these sub-problems and identify TRUE prerequisite dependencies.

Problem: {problem.title}

Sub-Problems:
{chr(10).join(descriptions)}

Identify which sub-problems MUST be completed before others.
Return ONLY a JSON list of dependency edges (no prose, no Markdown).
Each edge must be: {{"source_id": "<dependent_id>", "target_id": "<prerequisite_id>", "reason": "<short rationale>"}}
Use the exact sub-problem ids shown above (inside brackets). List only necessary dependencies.
Example:
[
  {{"source_id": "sp_api", "target_id": "sp_auth", "reason": "API needs auth contracts"}},
  {{"source_id": "sp_ui", "target_id": "sp_api", "reason": "UI integrates API endpoints"}}
]

JSON:"""
            
            response = None
            if hasattr(self.llm_client, "generate"):
                response = self.llm_client.generate(prompt)
            elif hasattr(self.llm_client, "complete"):
                response = self.llm_client.complete(prompt)
            elif hasattr(self.llm_client, "chat"):
                response = self.llm_client.chat(prompt)
            elif callable(self.llm_client):
                response = self.llm_client(prompt)

            if not response and not dspy_text:
                return self._apply_heuristic_dependencies(sub_problems, problem)

            text = dspy_text or (response.get("text") if isinstance(response, dict) else str(response))
            edges_payload = self._extract_json_payload(text)
            edges = self._parse_dependency_edges(edges_payload)
            if not edges:
                return self._apply_heuristic_dependencies(sub_problems, problem)
            return self._apply_dependency_edges(sub_problems, edges)
            
        except (RuntimeError, ValueError, ConnectionError) as e:
            self.logger.warning(f"LLM dependency analysis failed: {e}")
            return None

    @staticmethod
    def _apply_dependency_edges(
        sub_problems: List[SubProblem],
        edges: List["DependencyDecomposition.DependencyEdge"],
    ) -> List[SubProblem]:
        """Apply dependency edges to sub-problem objects."""
        id_map = {str(i + 1): sp.id for i, sp in enumerate(sub_problems)}
        id_map.update({sp.id: sp.id for sp in sub_problems})
        subproblem_map = {sp.id: sp for sp in sub_problems}
        valid_ids = set(subproblem_map.keys())

        for sp in sub_problems:
            sp.dependencies = []
            if not isinstance(sp.metadata, dict):
                sp.metadata = {}
            sp.metadata.setdefault("dependency_reasons", {})

        for edge in edges:
            depender = id_map.get(str(edge.source_id), str(edge.source_id))
            dependency = id_map.get(str(edge.target_id), str(edge.target_id))
            if depender in valid_ids and dependency in valid_ids and dependency != depender:
                if dependency not in subproblem_map[depender].dependencies:
                    subproblem_map[depender].dependencies.append(dependency)
                reason = (edge.reason or "").strip()
                if reason:
                    subproblem_map[depender].metadata["dependency_reasons"][dependency] = reason

        return sub_problems

    def _parse_dependency_edges(self, payload: Any) -> List["DependencyDecomposition.DependencyEdge"]:
        """Validate and normalize dependency edges from JSON payload."""
        if not payload:
            return []

        if isinstance(payload, self.DependencyEdgeList):
            return list(payload.edges or [])

        if isinstance(payload, list) and payload and isinstance(payload[0], self.DependencyEdge):
            return list(payload)

        if isinstance(payload, dict) and "edges" in payload:
            payload = payload.get("edges")

        if not isinstance(payload, list):
            return []

        edges: List[DependencyDecomposition.DependencyEdge] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            if "source_id" not in normalized:
                for key in ("source", "from", "dependent", "depender", "src"):
                    if key in normalized:
                        normalized["source_id"] = normalized[key]
                        break
            if "target_id" not in normalized:
                for key in ("target", "to", "prerequisite", "depends_on", "dependency", "dest"):
                    if key in normalized:
                        normalized["target_id"] = normalized[key]
                        break
            if "reason" not in normalized:
                normalized["reason"] = ""

            try:
                if PYDANTIC_AVAILABLE and hasattr(self.DependencyEdge, "model_validate"):
                    edge = self.DependencyEdge.model_validate(normalized)
                elif PYDANTIC_AVAILABLE and hasattr(self.DependencyEdge, "parse_obj"):
                    edge = self.DependencyEdge.parse_obj(normalized)  # type: ignore[attr-defined]
                else:
                    edge = self.DependencyEdge(
                        source_id=str(normalized.get("source_id", "")),
                        target_id=str(normalized.get("target_id", "")),
                        reason=str(normalized.get("reason", "")),
                    )
                if edge.source_id and edge.target_id:
                    edges.append(edge)
            except ValidationError as exc:
                self.logger.debug("Invalid dependency edge skipped: %s", exc)
                continue

        return edges

    @staticmethod
    def _extract_json_payload(text: str) -> Any:
        """Extract JSON payload from LLM output."""
        if not text:
            return None

        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove code fences
            fence_parts = stripped.split("```")
            if len(fence_parts) >= 2:
                stripped = fence_parts[1].strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Attempt to locate JSON substring
        start_candidates = [stripped.find("["), stripped.find("{")]
        start_candidates = [idx for idx in start_candidates if idx != -1]
        if not start_candidates:
            return None
        start_idx = min(start_candidates)
        end_char = "]" if stripped[start_idx] == "[" else "}"
        end_idx = stripped.rfind(end_char)
        if end_idx == -1:
            return None
        snippet = stripped[start_idx:end_idx + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    
    def _apply_heuristic_dependencies(
        self, sub_problems: List[SubProblem], problem: ProblemDefinition
    ) -> List[SubProblem]:
        """Apply heuristic dependency rules"""
        # Keywords that indicate foundational work
        foundational_keywords = ['setup', 'configure', 'design', 'architecture', 'framework', 'base', 'core']
        
        # Identify foundational sub-problems
        foundational_ids = []
        for sp in sub_problems:
            if any(kw in sp.title.lower() or kw in sp.description.lower() for kw in foundational_keywords):
                foundational_ids.append(sp.id)
        
        # Add dependencies: non-foundational depends on foundational
        for sp in sub_problems:
            if sp.id not in foundational_ids:
                for fid in foundational_ids:
                    if fid not in sp.dependencies and fid != sp.id:
                        sp.dependencies.append(fid)
        
        return sub_problems


class ComplexityDecomposition(DecompositionStrategyBase):
    """
    Decomposes to balance cognitive load and complexity.
    Breaks down complex sub-problems into manageable chunks.
    """
    
    COMPLEXITY_THRESHOLD = 7.0  # Split if complexity > 7
    
    def get_strategy_name(self) -> str:
        return "complexity"
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose to balance complexity"""
        self.logger.info(f"Complexity decomposition for: {problem.title}")
        
        # Start with semantic decomposition
        semantic = SemanticDecomposition(self.llm_client)
        base_subproblems = semantic.decompose(problem)
        
        # Split high-complexity sub-problems
        result = []
        for sp in base_subproblems:
            if sp.complexity_score.overall_complexity > self.COMPLEXITY_THRESHOLD:
                split_subproblems = self._split_by_complexity(sp)
                result.extend(split_subproblems)
            else:
                result.append(sp)
        
        return result
    
    def _split_by_complexity(self, sub_problem: SubProblem) -> List[SubProblem]:
        """Split a complex sub-problem into simpler ones"""
        # Determine number of splits based on complexity
        complexity = sub_problem.complexity_score.overall_complexity
        num_splits = max(2, int(complexity / self.COMPLEXITY_THRESHOLD))
        
        sub_problems = []
        
        for i in range(num_splits):
            # Reduce complexity proportionally
            reduction_factor = 0.6
            
            sp = SubProblem(
                id=self.generate_id(f"cplx_{i}"),
                parent_id=sub_problem.parent_id,
                title=f"{sub_problem.title} - Part {i+1}/{num_splits}",
                description=f"Subset of: {sub_problem.description[:100]}...",
                type=sub_problem.type,
                complexity_score=ComplexityScore(
                    cognitive_complexity=sub_problem.complexity_score.cognitive_complexity * reduction_factor,
                    computational_complexity=sub_problem.complexity_score.computational_complexity * reduction_factor,
                    domain_complexity=sub_problem.complexity_score.domain_complexity * reduction_factor,
                    integration_complexity=sub_problem.complexity_score.integration_complexity * reduction_factor,
                    overall_complexity=sub_problem.complexity_score.overall_complexity * reduction_factor,
                    explanation=f"Split from complex sub-problem {sub_problem.id}"
                ),
                dependencies=[sub_problem.id] if i > 0 else [],
                estimated_effort_hours=sub_problem.estimated_effort_hours / num_splits
            )
            sub_problems.append(sp)
        
        return sub_problems


class HybridDecomposition(DecompositionStrategyBase):
    """
    Adaptive decomposition that combines multiple strategies.
    Selects and combines strategies based on problem characteristics.
    """
    
    def get_strategy_name(self) -> str:
        return "hybrid"
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose using adaptive strategy combination"""
        self.logger.info(f"Hybrid decomposition for: {problem.title}")
        
        # Determine best strategy mix
        strategies = self._select_strategies(problem)
        
        all_subproblems = []
        seen_descriptions = set()
        
        for strategy in strategies:
            subproblems = strategy.decompose(problem)
            
            # Deduplicate based on description similarity
            for sp in subproblems:
                desc_hash = hashlib.md5(sp.description.lower().encode()).hexdigest()[:16]
                if desc_hash not in seen_descriptions:
                    seen_descriptions.add(desc_hash)
                    all_subproblems.append(sp)
        
        # Merge and consolidate
        consolidated = self._consolidate_subproblems(all_subproblems)
        
        # Final dependency analysis
        final = self._finalize_dependencies(consolidated)
        
        return final
    
    def _select_strategies(self, problem: ProblemDefinition) -> List[DecompositionStrategyBase]:
        """Select appropriate strategies based on problem"""
        strategies = []
        complexity = problem.complexity_score.overall_complexity
        
        # Always use semantic
        strategies.append(SemanticDecomposition(self.llm_client))
        
        # Add dependency for complex problems
        if complexity > 5.0:
            strategies.append(DependencyDecomposition(self.llm_client))
        
        # Add complexity-based for very complex problems
        if complexity > 7.5:
            strategies.append(ComplexityDecomposition(self.llm_client))
        
        return strategies
    
    def _consolidate_subproblems(self, subproblems: List[SubProblem]) -> List[SubProblem]:
        """Merge similar sub-problems"""
        if len(subproblems) <= 5:
            return subproblems
        
        # Group by type
        by_type = defaultdict(list)
        for sp in subproblems:
            by_type[sp.type].append(sp)
        
        consolidated = []
        for sp_type, sps in by_type.items():
            if len(sps) > 3:
                # Merge similar ones
                merged = self._merge_subproblems(sps[:3])  # Merge first 3
                consolidated.append(merged)
                consolidated.extend(sps[3:])
            else:
                consolidated.extend(sps)
        
        return consolidated
    
    def _merge_subproblems(self, subproblems: List[SubProblem]) -> SubProblem:
        """Merge multiple sub-problems into one"""
        if not subproblems:
            raise ValueError("Cannot merge empty list")
        
        base = subproblems[0]
        
        merged_description = base.description + "\n\nAlso covers:\n"
        for sp in subproblems[1:]:
            merged_description += f"- {sp.title}: {sp.description[:100]}...\n"
        
        avg_complexity = sum(sp.complexity_score.overall_complexity for sp in subproblems) / len(subproblems)
        
        return SubProblem(
            id=self.generate_id("merged"),
            parent_id=base.parent_id,
            title=f"Integrated: {base.title}",
            description=merged_description,
            type=base.type,
            complexity_score=ComplexityScore(
                cognitive_complexity=avg_complexity,
                computational_complexity=avg_complexity,
                domain_complexity=avg_complexity,
                integration_complexity=avg_complexity * 1.2,
                overall_complexity=min(avg_complexity * 1.1, 10),
                explanation=f"Merged from {len(subproblems)} related sub-problems"
            ),
            dependencies=list(set(dep for sp in subproblems for dep in sp.dependencies)),
            estimated_effort_hours=sum(sp.estimated_effort_hours for sp in subproblems) * 0.8
        )
    
    def _finalize_dependencies(self, subproblems: List[SubProblem]) -> List[SubProblem]:
        """Final dependency cleanup"""
        # Ensure no circular dependencies (simplified)
        sp_ids = {sp.id for sp in subproblems}
        
        for sp in subproblems:
            # Remove dependencies to non-existent sub-problems
            sp.dependencies = [dep for dep in sp.dependencies if dep in sp_ids]
            
            # Remove self-dependencies
            if sp.id in sp.dependencies:
                sp.dependencies.remove(sp.id)
        
        return subproblems


# ============================================================================
# MAIN DECOMPOSITION ENGINE
# ============================================================================

class UniversalDecompositionEngine:
    """
    Universal decomposition engine for any problem domain.
    
    Provides a unified interface for decomposing problems from any industry
    into manageable, solvable sub-problems.
    
    Features:
        - Automatic strategy selection
        - Domain-aware decomposition
        - Dependency graph generation
        - Parallel execution planning
        - Quality assessment
    
    Example:
        >>> engine = UniversalDecompositionEngine()
        >>> 
        >>> # Software problem
        >>> result = engine.decompose(
        ...     problem_statement="Build a microservice authentication system",
        ...     domain=ProblemDomain.SOFTWARE,
        ...     constraints=["oauth2_support", "jwt_tokens"]
        ... )
        >>> 
        >>> # Finance problem
        >>> result = engine.decompose(
        ...     problem_statement="Implement real-time trading risk controls",
        ...     domain=ProblemDomain.FINANCE,
        ...     constraints=["regulatory_compliance", "sub_millisecond_latency"]
        ... )
    """
    
    # Strategy registry
    STRATEGIES = {
        DecompositionStrategy.SEMANTIC: SemanticDecomposition,
        DecompositionStrategy.DEPENDENCY: DependencyDecomposition,
        DecompositionStrategy.COMPLEXITY: ComplexityDecomposition,
        DecompositionStrategy.HYBRID: HybridDecomposition,
    }
    
    def __init__(self, llm_client: Optional[Any] = None, default_strategy: DecompositionStrategy = DecompositionStrategy.HYBRID):
        self.llm_client = llm_client
        self.default_strategy = default_strategy
        self.logger = logging.getLogger(self.__class__.__name__)
        self.decomposition_history: List[DecompositionPlan] = []

    async def verify_with_lean(self, content: str) -> Dict[str, Any]:
        """Verify content using Lean theorem prover.
        
        **LEAN INTEGRATION**: Formal verification of problem structure.
        """
        if not LEAN_AVAILABLE:
            return {"verified": False, "reason": "Lean unavailable"}
        
        try:
            lean_client = LeanAideClient()
            formalized = await lean_client.translate_thm(content)
            result = await lean_client.verify(formalized)
            
            return {
                "verified": result.success if hasattr(result, 'success') else False,
                "confidence": getattr(result, 'confidence', 0.0),
                "proof": getattr(result, 'data', {}).get('result', '') if hasattr(result, 'data') else str(result)
            }
        except Exception as e:
            self.logger.warning(f"Lean verification failed: {e}")
            return {"verified": False, "reason": str(e)}
    
    def decompose(
        self,
        problem_statement: str,
        title: Optional[str] = None,
        domain: ProblemDomain = ProblemDomain.GENERIC,
        constraints: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        strategy: Optional[DecompositionStrategy] = None,
        max_subproblems: int = 15,
        min_subproblem_size: int = 50,
        domain_artifacts: Optional[Dict[str, Any]] = None,
    ) -> DecompositionPlan:
        """
        Decompose a problem statement into sub-problems.

        Args:
            problem_statement: The problem to decompose
            title: Optional title for the problem
            domain: Problem domain (software, finance, etc.)
            constraints: List of constraint descriptions
            success_criteria: List of success criteria descriptions
            strategy: Decomposition strategy to use (auto-selected if None)
            max_subproblems: Maximum number of sub-problems to create
            min_subproblem_size: Minimum description length for sub-problems
            domain_artifacts: Optional domain-specific pre-analysis artifacts
                (e.g., Slither/Forge outputs for Web3 decomposition)

        Returns:
            DecompositionPlan with all sub-problems and dependencies
        """
        import time
        start_time_total = time.time()
        success = False
        problem_id = title or problem_statement[:50]

        try:
            start_time = datetime.now()

            self.logger.info(f"Starting decomposition: {problem_id}...")

            # Create problem definition
            problem = self._create_problem_definition(
                problem_statement=problem_statement,
                title=title,
                domain=domain,
                constraints=constraints or [],
                success_criteria=success_criteria or [],
                domain_artifacts=domain_artifacts or {},
            )

            # Select strategy
            selected_strategy = strategy or self._select_strategy(problem)
            self.logger.info(f"Using strategy: {selected_strategy.value}")

            # Get strategy instance
            strategy_class = self.STRATEGIES[selected_strategy]
            strategy_instance = strategy_class(self.llm_client)

            # Execute decomposition
            sub_problems = strategy_instance.decompose(problem)

            # Post-process
            sub_problems = self._post_process_subproblems(
                sub_problems,
                max_count=max_subproblems,
                min_size=min_subproblem_size
            )

            # Build dependency graph
            dependency_graph = self._build_dependency_graph(sub_problems)

            # Calculate execution order
            execution_order = self._calculate_execution_order(sub_problems, dependency_graph)

            # Identify parallel groups
            parallel_groups = self._identify_parallel_groups(sub_problems, dependency_graph)

            # Calculate quality score
            quality_score = self._calculate_quality_score(problem, sub_problems, dependency_graph)

            # Create plan
            plan = DecompositionPlan(
                id=self._generate_id("plan"),
                original_problem=problem,
                sub_problems=sub_problems,
                strategy_used=selected_strategy,
                dependency_graph=dependency_graph,
                execution_order=execution_order,
                parallel_groups=parallel_groups,
                quality_score=quality_score,
                metadata={
                    'decomposition_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'strategy_class': strategy_class.__name__,
                    'num_subproblems': len(sub_problems)
                }
            )

            # Apply domain-specific extensions (finance/legal/manufacturing)
            plan = self._apply_domain_extensions(plan)
            if plan.metadata.get("domain_extensions_applied"):
                plan.dependency_graph = self._build_dependency_graph(plan.sub_problems)
                plan.execution_order = self._calculate_execution_order(plan.sub_problems, plan.dependency_graph)
                plan.parallel_groups = self._identify_parallel_groups(plan.sub_problems, plan.dependency_graph)
                plan.quality_score = self._calculate_quality_score(
                    problem, plan.sub_problems, plan.dependency_graph
                )
                sub_problems = plan.sub_problems
                dependency_graph = plan.dependency_graph
                execution_order = plan.execution_order
                parallel_groups = plan.parallel_groups
                quality_score = plan.quality_score

            # Build entanglement matrix for downstream coordination
            try:
                matrix, symbols_by_id = build_symbolic_entanglement_matrix(
                    sub_problems,
                    allowed_ids=[sp.id for sp in sub_problems],
                    enforce_symmetry=True,
                    strict=False,
                )
                serialized = serialize_entanglement_matrix(matrix)
                plan.metadata["entanglement_matrix"] = serialized
                if plan.analyzed_context is None:
                    plan.analyzed_context = {}
                if isinstance(plan.analyzed_context, dict):
                    plan.analyzed_context.setdefault("domain", problem.domain.value)
                    plan.analyzed_context.setdefault(
                        "constraints", [c.description for c in problem.constraints]
                    )
                    plan.analyzed_context.setdefault(
                        "success_criteria", [c.description for c in problem.success_criteria]
                    )
                    plan.analyzed_context["entanglement_matrix"] = serialized
                for sp in sub_problems:
                    entangled_with = serialized.get(sp.id, [])
                    sp.metadata["entangled_with"] = entangled_with
                    if entangled_with and "entanglement_source" not in sp.metadata:
                        sp.metadata["entanglement_source"] = "symbolic_overlap"
                    if sp.id in symbols_by_id:
                        sp.metadata["entanglement_symbols"] = sorted(symbols_by_id.get(sp.id, set()))
            except Exception as exc:
                self.logger.warning(f"Failed to build entanglement matrix: {exc}")

            self.decomposition_history.append(plan)
            self.logger.info(f"Decomposition complete: {len(sub_problems)} sub-problems, quality={quality_score:.2f}")

            success = True
            duration = time.time() - start_time_total

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful decomposition
            self._extract_universal_decomp_knowledge("decompose", problem_id, selected_strategy, plan)
            self._track_universal_decomp_performance("decompose", True, duration, len(sub_problems))

            return plan

        except Exception as e:
            duration = time.time() - start_time_total

            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_universal_decomp_alerts("decompose", False, problem_id, str(e))
            self._track_universal_decomp_performance("decompose", False, duration, 0)

            self.logger.error(f"Error during decomposition: {e}")
            raise
    
    def _create_problem_definition(
        self,
        problem_statement: str,
        title: Optional[str],
        domain: ProblemDomain,
        constraints: List[str],
        success_criteria: List[str],
        domain_artifacts: Optional[Dict[str, Any]] = None,
    ) -> ProblemDefinition:
        """Create a problem definition from input"""
        
        # Generate title if not provided
        if not title:
            words = problem_statement.split()[:8]
            title = ' '.join(words) + ('...' if len(problem_statement.split()) > 8 else '')
        
        # Estimate complexity
        complexity = self._estimate_problem_complexity(problem_statement, domain)
        
        # Create constraints
        constraint_objects = []
        for i, c in enumerate(constraints):
            constraint_objects.append(Constraint(
                id=self._generate_id("con"),
                description=c,
                type=self._classify_constraint(c),
                severity="hard" if any(kw in c.lower() for kw in ["must", "required", "critical"]) else "soft"
            ))
        
        # Create success criteria
        criteria_objects = []
        for i, sc in enumerate(success_criteria):
            criteria_objects.append(SuccessCriterion(
                id=self._generate_id("crit"),
                description=sc,
                metric="completion",
                threshold=0.9
            ))
        
        return ProblemDefinition(
            id=self._generate_id("prob"),
            title=title,
            description=problem_statement,
            domain=domain,
            complexity_score=complexity,
            constraints=constraint_objects,
            success_criteria=criteria_objects,
            metadata={
                "domain_artifacts": domain_artifacts or {}
            }
        )
    
    def _estimate_problem_complexity(self, statement: str, domain: ProblemDomain) -> ComplexityScore:
        """Estimate problem complexity from statement"""
        
        # Base complexity on length and domain
        length_factor = min(len(statement) / 500, 2.0)
        
        # Domain-specific complexity adjustments
        domain_base = {
            ProblemDomain.SOFTWARE: 5.0,
            ProblemDomain.FINANCE: 6.0,
            ProblemDomain.WEB3: 7.2,
            ProblemDomain.SCIENTIFIC: 7.0,
            ProblemDomain.HEALTHCARE: 6.5,
            ProblemDomain.MANUFACTURING: 6.0,
            ProblemDomain.LEGAL: 6.0,
            ProblemDomain.GENERIC: 5.0
        }.get(domain, 5.0)
        
        # Keyword-based complexity
        complexity_keywords = ['real-time', 'distributed', 'scalable', 'secure', 'compliance', 'integration']
        keyword_count = sum(1 for kw in complexity_keywords if kw in statement.lower())
        keyword_factor = 1 + (keyword_count * 0.1)
        
        overall = min(10.0, domain_base * length_factor * keyword_factor / 2)
        
        return ComplexityScore(
            cognitive_complexity=min(overall * 1.1, 10),
            computational_complexity=min(overall * 0.9, 10),
            domain_complexity=min(overall * 1.0, 10),
            integration_complexity=min(overall * 1.2, 10),
            overall_complexity=overall,
            explanation=f"Estimated from domain={domain.value}, length={len(statement)}, keywords={keyword_count}"
        )
    
    def _classify_constraint(self, constraint: str) -> str:
        """Classify constraint type"""
        lower = constraint.lower()
        if any(kw in lower for kw in ['time', 'deadline', 'schedule', 'duration']):
            return 'time'
        elif any(kw in lower for kw in ['cost', 'budget', 'resource', 'money']):
            return 'resource'
        elif any(kw in lower for kw in ['quality', 'performance', 'accuracy']):
            return 'quality'
        elif any(kw in lower for kw in ['regulatory', 'compliance', 'legal', 'regulation']):
            return 'regulatory'
        return 'technical'
    
    def _select_strategy(self, problem: ProblemDefinition) -> DecompositionStrategy:
        """Select best decomposition strategy for problem"""
        complexity = problem.complexity_score.overall_complexity
        
        if complexity > 8.0:
            return DecompositionStrategy.HYBRID
        elif complexity > 6.0 and any(c.type == 'technical' for c in problem.constraints):
            return DecompositionStrategy.DEPENDENCY
        elif len(problem.description) > 1000:
            return DecompositionStrategy.SEMANTIC
        else:
            return DecompositionStrategy.HYBRID
    
    def _post_process_subproblems(
        self, 
        sub_problems: List[SubProblem],
        max_count: int,
        min_size: int
    ) -> List[SubProblem]:
        """Post-process sub-problems"""
        
        # Filter out too-small sub-problems
        filtered = [sp for sp in sub_problems if len(sp.description) >= min_size]
        
        # Limit count
        if len(filtered) > max_count:
            # Sort by priority and complexity, keep top ones
            filtered.sort(key=lambda sp: (sp.priority, sp.complexity_score.overall_complexity), reverse=True)
            filtered = filtered[:max_count]
        
        return filtered
    
    def _build_dependency_graph(self, sub_problems: List[SubProblem]) -> Dict[str, List[str]]:
        """Build dependency graph from sub-problems"""
        return {sp.id: sp.dependencies for sp in sub_problems}
    
    def _calculate_execution_order(
        self, 
        sub_problems: List[SubProblem], 
        dependency_graph: Dict[str, List[str]]
    ) -> List[str]:
        """Calculate topological execution order"""
        # Topological sort (graph maps node -> dependencies)
        in_degree = {sp.id: len(sp.dependencies) for sp in sub_problems}
        dependents: Dict[str, List[str]] = {sp.id: [] for sp in sub_problems}
        for node, deps in dependency_graph.items():
            for dep in deps:
                if dep in dependents:
                    dependents[dep].append(node)

        queue = [sp_id for sp_id, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            for sp_id in dependents.get(current, []):
                in_degree[sp_id] -= 1
                if in_degree[sp_id] == 0:
                    queue.append(sp_id)
        
        return order
    
    def _identify_parallel_groups(
        self,
        sub_problems: List[SubProblem],
        dependency_graph: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Identify groups of sub-problems that can be executed in parallel"""
        # Simple grouping: sub-problems with same dependencies can run together
        dependency_sets = defaultdict(list)
        
        for sp in sub_problems:
            dep_key = frozenset(sp.dependencies)
            dependency_sets[dep_key].append(sp.id)
        
        return list(dependency_sets.values())
    
    def _calculate_quality_score(
        self,
        problem: ProblemDefinition,
        sub_problems: List[SubProblem],
        dependency_graph: Dict[str, List[str]]
    ) -> float:
        """Calculate quality score for decomposition"""
        
        if not sub_problems:
            return 0.0
        
        scores = []
        
        # Coverage: do sub-problems cover the problem?
        problem_keywords = set(problem.description.lower().split())
        covered_keywords = set()
        for sp in sub_problems:
            covered_keywords.update(sp.description.lower().split())
        coverage = len(problem_keywords & covered_keywords) / len(problem_keywords) if problem_keywords else 1.0
        scores.append(coverage)
        
        # Balance: are sub-problems similar in complexity?
        complexities = [sp.complexity_score.overall_complexity for sp in sub_problems]
        if complexities:
            avg_complexity = sum(complexities) / len(complexities)
            variance = sum((c - avg_complexity) ** 2 for c in complexities) / len(complexities)
            balance = 1.0 - min(variance / 25, 1.0)  # Normalize
            scores.append(balance)
        
        # Dependency health: reasonable dependency structure
        dep_count = sum(len(deps) for deps in dependency_graph.values())
        avg_deps = dep_count / len(sub_problems) if sub_problems else 0
        dep_health = 1.0 - min(avg_deps / 5, 1.0)  # Penalize too many dependencies
        scores.append(dep_health)
        
        return sum(scores) / len(scores)

    def _apply_domain_extensions(self, plan: DecompositionPlan) -> DecompositionPlan:
        """Apply domain-specific extensions to the decomposition plan."""
        applied: List[str] = []
        domain = plan.original_problem.domain
        statement = plan.original_problem.description

        if domain == ProblemDomain.FINANCE or FinanceDomainExtension.is_finance_problem(statement):
            plan = FinanceDomainExtension.enhance_decomposition(plan)
            applied.append("finance")

        if domain == ProblemDomain.WEB3 or Web3DomainExtension.is_web3_problem(statement):
            plan = Web3DomainExtension.enhance_decomposition(plan)
            applied.append("web3")

        if domain == ProblemDomain.LEGAL or LegalDomainExtension.is_legal_problem(statement):
            plan = LegalDomainExtension.enhance_decomposition(plan)
            applied.append("legal")

        if domain == ProblemDomain.MANUFACTURING or ManufacturingDomainExtension.is_manufacturing_problem(statement):
            plan = ManufacturingDomainExtension.enhance_decomposition(plan)
            applied.append("manufacturing")

        if applied:
            plan.metadata.setdefault("domain_extensions_applied", [])
            plan.metadata["domain_extensions_applied"].extend(applied)
            plan.metadata["num_subproblems"] = len(plan.sub_problems)

        return plan
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"
    
    def get_decomposition_history(self) -> List[DecompositionPlan]:
        """Get history of all decompositions"""
        return self.decomposition_history.copy()

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Universal Decomposition
    # =========================================================================

    def _trigger_universal_decomp_alerts(
        self,
        operation: str,
        success: bool,
        problem_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for universal decomposition failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                alert_manager.create_alert(
                    title=f"Universal Decomposition Alert: {operation}",
                    description=f"Universal Decomposition operation '{operation}' failed" +
                                 (f" for problem '{problem_id}'" if problem_id else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=AlertSeverity.HIGH.value,
                    source="universal_decomposition_engine",
                    component="universal_decomposition",
                    metadata=metadata or {}
                )

        except Exception as e:
            self.logger.error(f"Failed to trigger Universal Decomposition alert: {e}")

    def _extract_universal_decomp_knowledge(
        self,
        operation: str,
        problem_id: str,
        strategy: 'DecompositionStrategy',
        plan: 'DecompositionPlan'
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract universal decomposition knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"univ_decomp_{operation}_{problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="universal_decomposition_execution",
                source_component="universal_decomposition_engine",
                title=f"Universal Decomposition: {operation} - {problem_id}",
                content={
                    "operation": operation,
                    "problem_id": problem_id,
                    "strategy": strategy.value if strategy else "unknown",
                    "num_subproblems": len(plan.sub_problems),
                    "quality_score": plan.quality_score,
                    "domain": plan.original_problem.domain.value if hasattr(plan.original_problem, 'domain') else "unknown",
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "parallel_groups": len(plan.parallel_groups) if hasattr(plan, 'parallel_groups') else 0,
                    "dependencies": len(plan.dependency_graph.edges) if hasattr(plan, 'dependency_graph') and hasattr(plan.dependency_graph, 'edges') else 0
                },
                tags=["universal_decomposition", operation, strategy.value if strategy else "unknown"]
            )

            knowledge_engine.store_artifact(artifact)
            self.logger.debug(f"Extracted Universal Decomposition knowledge for {problem_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to extract Universal Decomposition knowledge: {e}")
            return False

    def _track_universal_decomp_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        num_subproblems: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track universal decomposition performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            quality = 1.0 if success else 0.0

            performance_data = StrategyPerformanceData(
                strategy_name=f"univ_decomp_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "duration_seconds": duration_seconds,
                    "num_subproblems": num_subproblems
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                self.logger.debug(f"Tracked Universal Decomposition performance for {operation}")

        except Exception as e:
            self.logger.error(f"Failed to track Universal Decomposition performance: {e}")


# ============================================================================
# DOMAIN-SPECIFIC EXTENSIONS
# ============================================================================

class FinanceDomainExtension:
    """
    Finance-specific extensions for decomposition.
    
    Provides specialized handling for financial problems:
    - Regulatory compliance constraints
    - Risk management considerations
    - Trading system requirements
    - Reporting and audit trails
    """
    
    # Finance-specific sub-problem templates
    TEMPLATES = {
        'risk_management': {
            'title': 'Risk Management System',
            'description': 'Implement risk controls including position limits, exposure monitoring, and stress testing',
            'type': SubProblemType.IMPLEMENTATION,
            'success_criteria': ['Real-time risk calculation', 'Position limit enforcement', 'Stress test execution']
        },
        'compliance': {
            'title': 'Regulatory Compliance Module',
            'description': 'Ensure compliance with applicable regulations (MiFID II, Dodd-Frank, etc.)',
            'type': SubProblemType.VALIDATION,
            'success_criteria': ['Audit trail completeness', 'Report generation', 'Regulatory filing support']
        },
        'market_data': {
            'title': 'Market Data Integration',
            'description': 'Integrate real-time market data feeds with validation and normalization',
            'type': SubProblemType.INTEGRATION,
            'success_criteria': ['Sub-millisecond latency', '99.99% uptime', 'Data quality validation']
        },
        'trading_engine': {
            'title': 'Trading Engine Core',
            'description': 'Implement order management, execution algorithms, and position tracking',
            'type': SubProblemType.IMPLEMENTATION,
            'success_criteria': ['Order validation', 'Execution within SLA', 'Position reconciliation']
        },
        'reporting': {
            'title': 'Reporting and Analytics',
            'description': 'Build reporting infrastructure for P&L, risk, and regulatory reports',
            'type': SubProblemType.ANALYSIS,
            'success_criteria': ['Automated report generation', 'Data accuracy', 'Custom report support']
        }
    }
    
    REGULATORY_KEYWORDS = [
        'mifid', 'mifid ii', 'dodd-frank', 'basel', 'emir', 'sftr', 'cftc',
        'sec', 'fca', 'compliance', 'regulatory', 'audit', 'reporting'
    ]
    
    @classmethod
    def is_finance_problem(cls, problem_statement: str) -> bool:
        """Check if problem appears to be finance-related"""
        lower = problem_statement.lower()
        finance_terms = ['trading', 'risk', 'portfolio', 'market data', 'compliance', 
                        'settlement', 'clearing', 'derivative', 'equity', 'fixed income']
        return any(term in lower for term in finance_terms)
    
    @classmethod
    def enhance_decomposition(cls, plan: DecompositionPlan) -> DecompositionPlan:
        """Enhance decomposition with finance-specific considerations"""
        
        # Add regulatory compliance sub-problem if not present
        has_compliance = any(
            any(kw in sp.description.lower() for kw in cls.REGULATORY_KEYWORDS)
            for sp in plan.sub_problems
        )
        
        if not has_compliance and cls.is_finance_problem(plan.original_problem.description):
            compliance_sp = SubProblem(
                id=f"fin_compliance_{uuid.uuid4().hex[:8]}",
                parent_id=plan.original_problem.id,
                title=cls.TEMPLATES['compliance']['title'],
                description=cls.TEMPLATES['compliance']['description'],
                type=SubProblemType.VALIDATION,
                complexity_score=ComplexityScore(
                    cognitive_complexity=7.0,
                    computational_complexity=5.0,
                    domain_complexity=8.0,
                    integration_complexity=6.0,
                    overall_complexity=6.5,
                    explanation="Regulatory compliance complexity"
                ),
                dependencies=[sp.id for sp in plan.sub_problems[:2]],  # Depends on first few
                estimated_effort_hours=40
            )
            plan.sub_problems.append(compliance_sp)
        
        return plan


class LegalDomainExtension:
    """
    Legal-specific extensions for decomposition.
    
    Provides specialized handling for legal/compliance problems:
    - Jurisdiction validation
    - Clause consistency review
    - Regulatory alignment
    """

    TEMPLATES = {
        'jurisdiction_check': {
            'title': 'Jurisdiction & Applicability Check',
            'description': 'Identify applicable jurisdictions, governing law, and required legal frameworks',
            'type': SubProblemType.ANALYSIS,
            'success_criteria': ['Jurisdiction identified', 'Applicable statutes listed', 'Scope confirmed']
        },
        'clause_consistency': {
            'title': 'Clause Consistency Review',
            'description': 'Ensure clauses are coherent, non-contradictory, and aligned with definitions',
            'type': SubProblemType.VALIDATION,
            'success_criteria': ['Cross-references validated', 'Definitions consistent', 'Conflicts resolved']
        },
        'regulatory_alignment': {
            'title': 'Regulatory Alignment Review',
            'description': 'Validate obligations against relevant regulations, standards, and policies',
            'type': SubProblemType.VALIDATION,
            'success_criteria': ['Regulation mapping complete', 'Gaps identified', 'Remediation plan']
        }
    }

    LEGAL_KEYWORDS = [
        'contract', 'agreement', 'clause', 'jurisdiction', 'governing law', 'statute',
        'regulation', 'compliance', 'legal', 'litigation', 'policy', 'terms', 'liability'
    ]

    @classmethod
    def is_legal_problem(cls, problem_statement: str) -> bool:
        lower = problem_statement.lower()
        return any(term in lower for term in cls.LEGAL_KEYWORDS)

    @classmethod
    def enhance_decomposition(cls, plan: DecompositionPlan) -> DecompositionPlan:
        has_jurisdiction = any(
            'jurisdiction' in sp.description.lower() or 'governing law' in sp.description.lower()
            for sp in plan.sub_problems
        )
        has_clause_review = any(
            'clause' in sp.description.lower() or 'definition' in sp.description.lower()
            for sp in plan.sub_problems
        )

        if cls.is_legal_problem(plan.original_problem.description):
            deps = [sp.id for sp in plan.sub_problems[:2]]
            if not has_jurisdiction:
                sp = SubProblem(
                    id=f"legal_juris_{uuid.uuid4().hex[:8]}",
                    parent_id=plan.original_problem.id,
                    title=cls.TEMPLATES['jurisdiction_check']['title'],
                    description=cls.TEMPLATES['jurisdiction_check']['description'],
                    type=SubProblemType.ANALYSIS,
                    complexity_score=ComplexityScore(
                        cognitive_complexity=7.0,
                        computational_complexity=4.0,
                        domain_complexity=8.0,
                        integration_complexity=5.0,
                        overall_complexity=6.5,
                        explanation="Jurisdictional complexity"
                    ),
                    dependencies=deps,
                    estimated_effort_hours=24,
                    metadata={"domain_extension": "legal"}
                )
                plan.sub_problems.append(sp)

            if not has_clause_review:
                sp = SubProblem(
                    id=f"legal_clause_{uuid.uuid4().hex[:8]}",
                    parent_id=plan.original_problem.id,
                    title=cls.TEMPLATES['clause_consistency']['title'],
                    description=cls.TEMPLATES['clause_consistency']['description'],
                    type=SubProblemType.VALIDATION,
                    complexity_score=ComplexityScore(
                        cognitive_complexity=6.5,
                        computational_complexity=3.5,
                        domain_complexity=7.5,
                        integration_complexity=5.5,
                        overall_complexity=6.0,
                        explanation="Clause consistency complexity"
                    ),
                    dependencies=deps,
                    estimated_effort_hours=20,
                    metadata={"domain_extension": "legal"}
                )
                plan.sub_problems.append(sp)

        return plan


class ManufacturingDomainExtension:
    """
    Manufacturing-specific extensions for decomposition.
    
    Provides specialized handling for manufacturing problems:
    - Supply chain constraints
    - Physical tolerances
    - Quality control planning
    """

    TEMPLATES = {
        'supply_chain': {
            'title': 'Supply Chain Constraints',
            'description': 'Model supplier dependencies, lead times, and inventory constraints',
            'type': SubProblemType.ANALYSIS,
            'success_criteria': ['Supplier mapping complete', 'Lead time risks identified', 'Inventory model built']
        },
        'tolerance_analysis': {
            'title': 'Physical Tolerance Analysis',
            'description': 'Define and validate mechanical/production tolerances and failure modes',
            'type': SubProblemType.VALIDATION,
            'success_criteria': ['Tolerance ranges defined', 'Failure modes analyzed', 'Mitigations documented']
        },
        'quality_control': {
            'title': 'Quality Control Plan',
            'description': 'Establish QC checkpoints, sampling strategy, and acceptance criteria',
            'type': SubProblemType.TESTING,
            'success_criteria': ['QC checkpoints defined', 'Sampling plan set', 'Acceptance criteria validated']
        }
    }

    MANUFACTURING_KEYWORDS = [
        'manufacturing', 'production', 'factory', 'assembly', 'tolerance',
        'supply chain', 'inventory', 'materials', 'quality control', 'process'
    ]

    @classmethod
    def is_manufacturing_problem(cls, problem_statement: str) -> bool:
        lower = problem_statement.lower()
        return any(term in lower for term in cls.MANUFACTURING_KEYWORDS)

    @classmethod
    def enhance_decomposition(cls, plan: DecompositionPlan) -> DecompositionPlan:
        has_supply_chain = any(
            'supply chain' in sp.description.lower() or 'supplier' in sp.description.lower()
            for sp in plan.sub_problems
        )
        has_tolerance = any(
            'tolerance' in sp.description.lower() or 'failure mode' in sp.description.lower()
            for sp in plan.sub_problems
        )
        has_quality = any(
            'quality control' in sp.description.lower() or 'qc' in sp.description.lower()
            for sp in plan.sub_problems
        )

        if cls.is_manufacturing_problem(plan.original_problem.description):
            deps = [sp.id for sp in plan.sub_problems[:2]]
            if not has_supply_chain:
                sp = SubProblem(
                    id=f"mfg_supply_{uuid.uuid4().hex[:8]}",
                    parent_id=plan.original_problem.id,
                    title=cls.TEMPLATES['supply_chain']['title'],
                    description=cls.TEMPLATES['supply_chain']['description'],
                    type=SubProblemType.ANALYSIS,
                    complexity_score=ComplexityScore(
                        cognitive_complexity=6.0,
                        computational_complexity=5.0,
                        domain_complexity=6.5,
                        integration_complexity=6.0,
                        overall_complexity=6.0,
                        explanation="Supply chain constraints complexity"
                    ),
                    dependencies=deps,
                    estimated_effort_hours=28,
                    metadata={"domain_extension": "manufacturing"}
                )
                plan.sub_problems.append(sp)

            if not has_tolerance:
                sp = SubProblem(
                    id=f"mfg_tol_{uuid.uuid4().hex[:8]}",
                    parent_id=plan.original_problem.id,
                    title=cls.TEMPLATES['tolerance_analysis']['title'],
                    description=cls.TEMPLATES['tolerance_analysis']['description'],
                    type=SubProblemType.VALIDATION,
                    complexity_score=ComplexityScore(
                        cognitive_complexity=6.5,
                        computational_complexity=4.5,
                        domain_complexity=7.0,
                        integration_complexity=5.5,
                        overall_complexity=6.3,
                        explanation="Tolerance analysis complexity"
                    ),
                    dependencies=deps,
                    estimated_effort_hours=24,
                    metadata={"domain_extension": "manufacturing"}
                )
                plan.sub_problems.append(sp)

            if not has_quality:
                sp = SubProblem(
                    id=f"mfg_qc_{uuid.uuid4().hex[:8]}",
                    parent_id=plan.original_problem.id,
                    title=cls.TEMPLATES['quality_control']['title'],
                    description=cls.TEMPLATES['quality_control']['description'],
                    type=SubProblemType.TESTING,
                    complexity_score=ComplexityScore(
                        cognitive_complexity=5.5,
                        computational_complexity=4.0,
                        domain_complexity=6.0,
                        integration_complexity=5.0,
                        overall_complexity=5.6,
                        explanation="Quality control complexity"
                    ),
                    dependencies=deps,
                    estimated_effort_hours=18,
                    metadata={"domain_extension": "manufacturing"}
                )
                plan.sub_problems.append(sp)

        return plan


class Web3DomainExtension:
    """
    Web3-specific extensions for decomposition.

    Provides specialized handling for smart contract audit and exploit-generation
    workflows including:
    - Slither static analysis ingestion
    - Foundry/Forge fuzzing ingestion
    - Solidity/Rust invariant translation to Z3/Lean
    - Exploit witness synthesis
    """

    TEMPLATES = {
        "static_ingestion": {
            "title": "Static Ingestion: Slither Contract Analysis",
            "description": (
                "Run Slither static analysis to extract AST structure, detector findings, "
                "and contract dependency signals for entanglement mapping"
            ),
            "type": SubProblemType.ANALYSIS,
            "effort": 14,
            "complexity": 6.8,
        },
        "fuzz_ingestion": {
            "title": "Dynamic Ingestion: Foundry/Forge Fuzz Harness",
            "description": (
                "Execute Forge fuzz tests and property checks to surface edge-case failures, "
                "counterexample traces, and state-transition anomalies"
            ),
            "type": SubProblemType.TESTING,
            "effort": 18,
            "complexity": 7.2,
        },
        "formal_translation": {
            "title": "Formal Translation: Solidity Invariants to Z3/Lean",
            "description": (
                "Translate critical state transitions (withdraw/deposit/mint/burn) into Z3 "
                "constraints and Lean specifications for theorem-backed validation"
            ),
            "type": SubProblemType.VALIDATION,
            "effort": 22,
            "complexity": 7.6,
        },
        "exploit_synthesis": {
            "title": "Red Team Exploit Synthesis and Witness Generation",
            "description": (
                "Use symbolic execution and adversarial search to solve exploit predicates "
                "(e.g., balance drain with zero user deposit) and generate reproducible PoCs"
            ),
            "type": SubProblemType.IMPLEMENTATION,
            "effort": 24,
            "complexity": 8.2,
        },
        "patch_validation": {
            "title": "Blue/Gold Patch Validation and Replay",
            "description": (
                "Validate proposed remediations by replaying exploit traces, re-running fuzzing, "
                "and proving key invariants still hold after patching"
            ),
            "type": SubProblemType.VALIDATION,
            "effort": 16,
            "complexity": 7.0,
        },
    }

    WEB3_KEYWORDS = [
        "smart contract", "solidity", "evm", "bytecode", "defi", "web3", "onchain",
        "foundry", "forge", "slither", "hardhat", "anchor", "rust", "ink!",
        "reentrancy", "flash loan", "oracle", "bridge", "amm", "liquidity", "vault",
        "bug bounty", "audit", "exploit",
    ]

    @classmethod
    def is_web3_problem(cls, problem_statement: str) -> bool:
        lower = problem_statement.lower()
        return any(term in lower for term in cls.WEB3_KEYWORDS)

    @classmethod
    def _extract_contract_names(
        cls,
        statement: str,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        contracts: Set[str] = set()

        for match in re.findall(r"\b(?:contract|interface|library|struct)\s+([A-Z]\w+)", statement):
            contracts.add(match)

        for match in re.findall(r"\b([A-Z][a-zA-Z0-9_]{2,})\b", statement):
            if match.lower() not in {"solidity", "foundry", "slither", "defi", "evm", "z3", "lean"}:
                contracts.add(match)

        if isinstance(artifacts, dict):
            raw_contracts = artifacts.get("contracts", [])
            if isinstance(raw_contracts, list):
                for item in raw_contracts:
                    if isinstance(item, str):
                        contracts.add(item)
                    elif isinstance(item, dict):
                        name = item.get("name")
                        if isinstance(name, str):
                            contracts.add(name)
            raw_dependencies = artifacts.get("dependencies", {})
            if isinstance(raw_dependencies, dict):
                for key, values in raw_dependencies.items():
                    if isinstance(key, str):
                        contracts.add(key)
                    if isinstance(values, list):
                        for dep in values:
                            if isinstance(dep, str):
                                contracts.add(dep)

        return sorted(c for c in contracts if c and len(c) > 2)

    @classmethod
    def _extract_dependency_hints(
        cls,
        statement: str,
        contract_names: List[str],
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        dependency_map: Dict[str, Set[str]] = {c: set() for c in contract_names}

        if isinstance(artifacts, dict):
            raw_dependencies = artifacts.get("dependencies", {})
            if isinstance(raw_dependencies, dict):
                for source, deps in raw_dependencies.items():
                    if source not in dependency_map:
                        continue
                    if isinstance(deps, list):
                        for dep in deps:
                            if dep in dependency_map and dep != source:
                                dependency_map[source].add(dep)

        lower = statement.lower()
        for source in contract_names:
            source_lower = source.lower()
            if source_lower not in lower:
                continue
            for target in contract_names:
                if source == target:
                    continue
                target_lower = target.lower()
                if target_lower not in lower:
                    continue
                source_idx = lower.find(source_lower)
                target_idx = lower.find(target_lower)
                window = lower[min(source_idx, target_idx):max(source_idx, target_idx) + len(target_lower)]
                if any(term in window for term in ["depends", "uses", "calls", "oracle", "reads", "writes"]):
                    dependency_map[source].add(target)

        if not any(v for v in dependency_map.values()) and len(contract_names) >= 2:
            for idx in range(len(contract_names) - 1):
                dependency_map[contract_names[idx]].add(contract_names[idx + 1])

        return {key: sorted(list(value)) for key, value in dependency_map.items() if value}

    @classmethod
    def enhance_decomposition(cls, plan: DecompositionPlan) -> DecompositionPlan:
        statement = plan.original_problem.description
        if not cls.is_web3_problem(statement):
            return plan

        artifacts = {}
        if isinstance(plan.original_problem.metadata, dict):
            raw = plan.original_problem.metadata.get("domain_artifacts", {})
            if isinstance(raw, dict):
                artifacts = raw

        contract_names = cls._extract_contract_names(statement, artifacts)
        dependency_hints = cls._extract_dependency_hints(statement, contract_names, artifacts)
        base_dependencies = [sp.id for sp in plan.sub_problems[:2]]

        def _has_marker(marker: str) -> bool:
            return any(
                sp.metadata.get("web3_stage") == marker
                for sp in plan.sub_problems
            )

        def _append_template(template_key: str, stage: str, deps: Optional[List[str]] = None) -> Optional[str]:
            if _has_marker(stage):
                existing = next(
                    (sp.id for sp in plan.sub_problems if sp.metadata.get("web3_stage") == stage),
                    None,
                )
                return existing

            tpl = cls.TEMPLATES[template_key]
            complexity = tpl["complexity"]
            sub_problem = SubProblem(
                id=f"web3_{stage}_{uuid.uuid4().hex[:8]}",
                parent_id=plan.original_problem.id,
                title=tpl["title"],
                description=tpl["description"],
                type=tpl["type"],
                complexity_score=ComplexityScore(
                    cognitive_complexity=min(complexity + 0.3, 10.0),
                    computational_complexity=min(complexity, 10.0),
                    domain_complexity=min(complexity + 0.6, 10.0),
                    integration_complexity=min(complexity + 0.4, 10.0),
                    overall_complexity=complexity,
                    explanation=f"Web3 {stage} complexity",
                ),
                dependencies=list(deps or base_dependencies),
                estimated_effort_hours=tpl["effort"],
                metadata={
                    "domain_extension": "web3",
                    "web3_stage": stage,
                    "interface_contracts": contract_names,
                    "entanglement_symbols": contract_names,
                    "shared_symbols": contract_names + ["slither", "forge", "z3", "lean4"],
                },
            )
            plan.sub_problems.append(sub_problem)
            return sub_problem.id

        static_id = _append_template("static_ingestion", "static_ingestion")
        fuzz_id = _append_template(
            "fuzz_ingestion",
            "fuzz_ingestion",
            deps=[dep for dep in [static_id] if dep] or base_dependencies,
        )
        formal_id = _append_template(
            "formal_translation",
            "formal_translation",
            deps=[dep for dep in [static_id, fuzz_id] if dep] or base_dependencies,
        )
        exploit_id = _append_template(
            "exploit_synthesis",
            "exploit_synthesis",
            deps=[dep for dep in [fuzz_id, formal_id] if dep] or base_dependencies,
        )
        _append_template(
            "patch_validation",
            "patch_validation",
            deps=[dep for dep in [formal_id, exploit_id] if dep] or base_dependencies,
        )

        for sp in plan.sub_problems:
            symbols = set(sp.metadata.get("entanglement_symbols", []))
            text = f"{sp.title} {sp.description}".lower()
            for name in contract_names:
                if name.lower() in text:
                    symbols.add(name)
            if sp.metadata.get("domain_extension") == "web3":
                symbols.update(contract_names)
            if symbols:
                sp.metadata["entanglement_symbols"] = sorted(symbols)

        plan.metadata.setdefault("web3", {})
        plan.metadata["web3"].update({
            "contracts": contract_names,
            "dependency_hints": dependency_hints,
            "ingestion_tools": ["slither", "forge"],
            "formal_tools": ["z3", "lean4"],
        })

        return plan


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    'ProblemDomain',
    'DecompositionStrategy',
    'SubProblemType',
    'SubProblemStatus',
    'AssemblyStrategy',
    
    # Data classes
    'ProblemDefinition',
    'SubProblem',
    'DecompositionPlan',
    'SubProblemSolution',
    'IntegratedSolution',
    'Constraint',
    'SuccessCriterion',
    'ComplexityScore',
    
    # Strategies
    'DecompositionStrategyBase',
    'SemanticDecomposition',
    'DependencyDecomposition',
    'ComplexityDecomposition',
    'HybridDecomposition',
    
    # Main engine
    'UniversalDecompositionEngine',
    
    # Extensions
    'FinanceDomainExtension',
    'Web3DomainExtension',
    'LegalDomainExtension',
    'ManufacturingDomainExtension',
]


# ============================================================================
# MAIN EXECUTION (EXAMPLES)
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=" * 70)
    print("Universal Decomposition Engine - Examples")
    print("=" * 70)
    
    # Initialize engine
    engine = UniversalDecompositionEngine()
    
    # Example 1: Software Problem
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Software Engineering Problem")
    print("=" * 70)
    
    software_problem = """
    Build a scalable microservice-based authentication system with OAuth2 and JWT support.
    The system should handle 10,000 concurrent users, provide role-based access control,
    integrate with LDAP for enterprise users, and include comprehensive audit logging.
    Must comply with GDPR requirements for data privacy.
    """
    
    plan = engine.decompose(
        problem_statement=software_problem,
        title="Authentication Microservice System",
        domain=ProblemDomain.SOFTWARE,
        constraints=["OAuth2 support", "GDPR compliance", "LDAP integration"],
        success_criteria=["10K concurrent users", "sub-100ms response time"]
    )
    
    print(f"\nGenerated {len(plan.sub_problems)} sub-problems:")
    for i, sp in enumerate(plan.sub_problems, 1):
        deps = f" (depends on: {', '.join(sp.dependencies[:2])})" if sp.dependencies else ""
        print(f"  {i}. {sp.title} [complexity: {sp.complexity_score.overall_complexity:.1f}]{deps}")
    
    print(f"\nExecution Order: {plan.execution_order}")
    print(f"Parallel Groups: {len(plan.parallel_groups)}")
    print(f"Quality Score: {plan.quality_score:.2f}")
    
    # Example 2: Finance Problem
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Finance/Trading Problem")
    print("=" * 70)
    
    finance_problem = """
    Implement a real-time trading risk management system that monitors position limits,
    calculates Value at Risk (VaR) for portfolios, generates regulatory reports for MiFID II,
    and provides real-time alerts when risk thresholds are breached.
    Must handle high-frequency trading volumes with sub-millisecond latency.
    """
    
    plan = engine.decompose(
        problem_statement=finance_problem,
        title="Trading Risk Management System",
        domain=ProblemDomain.FINANCE,
        constraints=["MiFID II compliance", "sub-millisecond latency", "real-time processing"],
        success_criteria=["VaR calculation accuracy", "99.99% uptime"]
    )
    
    # Apply finance-specific enhancements
    plan = FinanceDomainExtension.enhance_decomposition(plan)
    
    print(f"\nGenerated {len(plan.sub_problems)} sub-problems:")
    for i, sp in enumerate(plan.sub_problems, 1):
        deps = f" (depends on: {', '.join(sp.dependencies[:2])})" if sp.dependencies else ""
        print(f"  {i}. {sp.title} [type: {sp.type.value}]{deps}")
    
    print(f"\nQuality Score: {plan.quality_score:.2f}")
    
    # Example 3: Scientific Problem
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Scientific Research Problem")
    print("=" * 70)
    
    science_problem = """
    Develop a machine learning pipeline for analyzing genomic sequences to identify
    disease markers. The system should handle large-scale data processing, implement
    multiple classification algorithms, perform cross-validation, and generate
    interpretable reports for biologists. Must ensure reproducibility and data privacy.
    """
    
    plan = engine.decompose(
        problem_statement=science_problem,
        title="Genomic ML Analysis Pipeline",
        domain=ProblemDomain.SCIENTIFIC,
        constraints=["data privacy (HIPAA)", "reproducibility", "large-scale processing"],
        success_criteria=["classification accuracy > 90%", "cross-validation support"]
    )
    
    print(f"\nGenerated {len(plan.sub_problems)} sub-problems:")
    for i, sp in enumerate(plan.sub_problems, 1):
        print(f"  {i}. {sp.title}")
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
