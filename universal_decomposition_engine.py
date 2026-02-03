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
    1. Problem Analysis → Understand domain, complexity, constraints
    2. Strategy Selection → Choose best decomposition approach
    3. Decomposition → Break into sub-problems with dependencies
    4. Execution → Solve each sub-problem (using appropriate teams/methods)
    5. Reassembly → Combine solutions with conflict resolution
    6. Validation → Verify final solution meets success criteria

Usage:
    >>> from universal_decomposition_engine import UniversalDecompositionEngine
    >>> engine = UniversalDecompositionEngine()
    >>> result = engine.decompose(
    ...     problem_statement="Build a trading risk management system",
    ...     domain="finance",
    ...     constraints=["regulatory_compliance", "real_time_processing"]
    ... )
"""

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
from utils.entanglement_utils import (
    build_symbolic_entanglement_matrix,
    serialize_entanglement_matrix,
)

# Configure logging
logger = logging.getLogger(__name__)

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


# ============================================================================
# ENUMS - Core Type Definitions
# ============================================================================

class ProblemDomain(Enum):
    """Supported problem domains"""
    SOFTWARE = "software"
    FINANCE = "finance"
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
            # Build descriptions for LLM
            descriptions = []
            for i, sp in enumerate(sub_problems, 1):
                descriptions.append(f"{i}. {sp.title}: {sp.description[:150]}...")
            
            prompt = f"""Analyze these sub-problems and identify TRUE prerequisite dependencies.

Problem: {problem.title}

Sub-Problems:
{chr(10).join(descriptions)}

Identify which sub-problems MUST be completed before others.
Format: "X depends on Y" (meaning X requires Y to be done first)
List only necessary dependencies, not all possible ones.

Dependencies:"""
            
            response = None
            if hasattr(self.llm_client, "generate"):
                response = self.llm_client.generate(prompt)
            elif hasattr(self.llm_client, "complete"):
                response = self.llm_client.complete(prompt)
            elif hasattr(self.llm_client, "chat"):
                response = self.llm_client.chat(prompt)
            elif callable(self.llm_client):
                response = self.llm_client(prompt)

            if not response:
                return self._apply_heuristic_dependencies(sub_problems, problem)

            text = response.get("text") if isinstance(response, dict) else str(response)
            id_map = {str(i + 1): sp.id for i, sp in enumerate(sub_problems)}
            id_map.update({sp.id: sp.id for sp in sub_problems})

            for sp in sub_problems:
                sp.dependencies = []

            pattern = re.compile(r"(\w+)\s+depends on\s+(\w+)", re.IGNORECASE)
            for line in text.splitlines():
                match = pattern.search(line.strip())
                if not match:
                    continue
                depender_raw, dep_raw = match.groups()
                depender = id_map.get(depender_raw, depender_raw)
                dependency = id_map.get(dep_raw, dep_raw)
                for sp in sub_problems:
                    if sp.id == depender and dependency != sp.id:
                        if dependency not in sp.dependencies:
                            sp.dependencies.append(dependency)

            return sub_problems
            
        except (RuntimeError, ValueError, ConnectionError) as e:
            self.logger.warning(f"LLM dependency analysis failed: {e}")
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
    
    def decompose(
        self,
        problem_statement: str,
        title: Optional[str] = None,
        domain: ProblemDomain = ProblemDomain.GENERIC,
        constraints: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        strategy: Optional[DecompositionStrategy] = None,
        max_subproblems: int = 15,
        min_subproblem_size: int = 50
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
                success_criteria=success_criteria or []
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
        success_criteria: List[str]
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
            success_criteria=criteria_objects
        )
    
    def _estimate_problem_complexity(self, statement: str, domain: ProblemDomain) -> ComplexityScore:
        """Estimate problem complexity from statement"""
        
        # Base complexity on length and domain
        length_factor = min(len(statement) / 500, 2.0)
        
        # Domain-specific complexity adjustments
        domain_base = {
            ProblemDomain.SOFTWARE: 5.0,
            ProblemDomain.FINANCE: 6.0,
            ProblemDomain.SCIENTIFIC: 7.0,
            ProblemDomain.HEALTHCARE: 6.5,
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
