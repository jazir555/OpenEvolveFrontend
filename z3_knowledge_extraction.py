"""
Z3 Knowledge Extraction and Management

Extracts structured knowledge from Z3 proofs and solutions:
- Proof patterns and tactics
- Constraint patterns
- Solution strategies
- Mathematical insights
- Knowledge graph construction
- Pattern matching and reuse

Author: OpenEvolve
Created: 2026-01-31
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Import Z3 integration
try:
    from z3prover_integration import Z3SolverResult, Z3TheoremResult
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

try:
    from z3prover_advanced import ExtractedProof, ProofStep
    Z3_ADVANCED_AVAILABLE = True
except ImportError:
    Z3_ADVANCED_AVAILABLE = False
    # Define placeholder types for type hints
    ExtractedProof = Any
    ProofStep = Any


# =============================================================================
# Knowledge Data Classes
# =============================================================================

@dataclass
class ProofPattern:
    """Pattern extracted from a proof."""
    pattern_id: str
    name: str
    description: str
    tactic_sequence: List[str]
    applicable_domains: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    usage_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source_proofs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "tactics": self.tactic_sequence,
            "domains": self.applicable_domains,
            "success_rate": f"{self.success_rate:.1%}",
            "usage_count": self.usage_count
        }


@dataclass
class ConstraintPattern:
    """Pattern in constraints."""
    pattern_id: str
    pattern_type: str  # "linear", "nonlinear", "boolean", "mixed"
    structure: str
    variables_involved: List[str]
    complexity_score: float = 0.0
    frequency: int = 0
    typical_solving_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.pattern_id,
            "type": self.pattern_type,
            "structure": self.structure,
            "variables": self.variables_involved,
            "complexity": self.complexity_score,
            "frequency": self.frequency
        }


@dataclass
class SolutionStrategy:
    """Reusable solution strategy."""
    strategy_id: str
    name: str
    problem_pattern: str
    recommended_tactics: List[str]
    solver_configuration: Dict[str, Any] = field(default_factory=dict)
    expected_performance: Dict[str, float] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.strategy_id,
            "name": self.name,
            "pattern": self.problem_pattern,
            "tactics": self.recommended_tactics,
            "success_rate": f"{self.success_rate():.1%}",
            "avg_time": self.expected_performance.get('avg_time', 0)
        }
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


@dataclass
class MathematicalInsight:
    """Insight extracted from solving process."""
    insight_id: str
    category: str  # "invariant", "bound", "relation", "optimization"
    statement: str
    formal_representation: Optional[str] = None
    proof_sketch: Optional[str] = None
    confidence: float = 0.0
    derived_from: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.insight_id,
            "category": self.category,
            "statement": self.statement,
            "confidence": f"{self.confidence:.1%}",
            "applications": len(self.applications)
        }


# =============================================================================
# Knowledge Extractor
# =============================================================================

class Z3KnowledgeExtractor:
    """
    Extracts knowledge from Z3 operations.
    
    Capabilities:
    - Proof pattern mining
    - Constraint analysis
    - Strategy learning
    - Insight extraction
    """
    
    def __init__(self):
        self.proof_patterns: Dict[str, ProofPattern] = {}
        self.constraint_patterns: Dict[str, ConstraintPattern] = {}
        self.strategies: Dict[str, SolutionStrategy] = {}
        self.insights: Dict[str, MathematicalInsight] = {}
        
        # Statistics
        self.extraction_count = 0
        self.pattern_matches = 0
    
    # =====================================================================
    # Proof Pattern Extraction
    # =====================================================================
    
    def extract_proof_patterns(
        self,
        proof: ExtractedProof,
        problem_domain: str = "general"
    ) -> List[ProofPattern]:
        """
        Extract reusable patterns from a proof.
        
        Args:
            proof: Extracted proof
            problem_domain: Domain classification
            
        Returns:
            List of extracted patterns
        """
        if not Z3_ADVANCED_AVAILABLE:
            return []
        
        patterns = []
        
        # Extract tactic sequences
        tactic_sequences = self._extract_tactic_sequences(proof.proof_steps)
        
        for seq in tactic_sequences:
            pattern_id = f"pattern_{len(self.proof_patterns)}"
            
            pattern = ProofPattern(
                pattern_id=pattern_id,
                name=f"Tactic sequence: {' -> '.join(seq[:3])}...",
                description=f"Sequence of {len(seq)} tactics",
                tactic_sequence=seq,
                applicable_domains=[problem_domain],
                source_proofs=[proof.raw_proof[:50] if proof.raw_proof else "unknown"]
            )
            
            self.proof_patterns[pattern_id] = pattern
            patterns.append(pattern)
        
        return patterns
    
    def _extract_tactic_sequences(
        self,
        steps: List[ProofStep],
        min_length: int = 2
    ) -> List[List[str]]:
        """Extract tactic sequences from proof steps."""
        sequences = []
        
        # Get all tactics
        tactics = [step.tactic for step in steps]
        
        # Extract sequences of varying lengths
        for length in range(min_length, min(len(tactics) + 1, 6)):
            for i in range(len(tactics) - length + 1):
                seq = tactics[i:i + length]
                sequences.append(seq)
        
        return sequences
    
    def find_matching_pattern(
        self,
        current_tactics: List[str]
    ) -> Optional[ProofPattern]:
        """Find pattern matching current tactic sequence."""
        best_match = None
        best_score = 0.0
        
        for pattern in self.proof_patterns.values():
            score = self._sequence_similarity(
                current_tactics,
                pattern.tactic_sequence
            )
            
            if score > best_score and score > 0.5:
                best_score = score
                best_match = pattern
        
        if best_match:
            self.pattern_matches += 1
        
        return best_match
    
    def _sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences."""
        if not seq1 or not seq2:
            return 0.0
        
        # Simple Jaccard-like similarity
        set1 = set(seq1)
        set2 = set(seq2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    # =====================================================================
    # Constraint Pattern Analysis
    # =====================================================================
    
    def analyze_constraints(
        self,
        constraints: List[str],
        solving_time: float,
        success: bool
    ) -> List[ConstraintPattern]:
        """
        Analyze and extract patterns from constraints.
        
        Args:
            constraints: List of constraint expressions
            solving_time: Time taken to solve
            success: Whether solving succeeded
            
        Returns:
            List of constraint patterns
        """
        patterns = []
        
        for constraint in constraints:
            pattern = self._classify_constraint(constraint)
            
            if pattern:
                pattern.frequency += 1
                if success:
                    # Update average solving time
                    pattern.typical_solving_time = (
                        (pattern.typical_solving_time * (pattern.frequency - 1) + solving_time)
                        / pattern.frequency
                    )
                
                patterns.append(pattern)
        
        return patterns
    
    def _classify_constraint(self, constraint: str) -> Optional[ConstraintPattern]:
        """Classify a constraint and extract its pattern."""
        pattern_id = f"constraint_{len(self.constraint_patterns)}"
        
        # Determine type
        if re.search(r'[\*\/\^]', constraint):
            pattern_type = "nonlinear"
        elif re.search(r'and|or|not', constraint, re.IGNORECASE):
            pattern_type = "boolean"
        elif re.search(r'[\+\-]', constraint):
            pattern_type = "linear"
        else:
            pattern_type = "atomic"
        
        # Extract variables
        variables = re.findall(r'\b[a-zA-Z_]\w*\b', constraint)
        variables = [v for v in variables if v not in ['and', 'or', 'not', 'assert']]
        
        # Calculate complexity
        complexity = len(variables) + constraint.count('(') + constraint.count(')')
        
        # Create or update pattern
        if pattern_type in self.constraint_patterns:
            pattern = self.constraint_patterns[pattern_type]
            pattern.frequency += 1
        else:
            pattern = ConstraintPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                structure=constraint[:50],
                variables_involved=list(set(variables)),
                complexity_score=complexity
            )
            self.constraint_patterns[pattern_id] = pattern
        
        return pattern
    
    # =====================================================================
    # Strategy Learning
    # =====================================================================
    
    def learn_strategy(
        self,
        problem_features: Dict[str, Any],
        tactics_used: List[str],
        config_used: Dict[str, Any],
        success: bool,
        solving_time: float
    ) -> SolutionStrategy:
        """
        Learn a new strategy from a successful solving attempt.
        
        Args:
            problem_features: Characteristics of the problem
            tactics_used: Tactics that were effective
            config_used: Solver configuration
            success: Whether the attempt succeeded
            solving_time: Time taken
            
        Returns:
            Learned strategy
        """
        strategy_id = f"strategy_{len(self.strategies)}"
        
        # Create problem pattern signature
        pattern_sig = self._create_problem_signature(problem_features)
        
        strategy = SolutionStrategy(
            strategy_id=strategy_id,
            name=f"Strategy for {problem_features.get('type', 'unknown')}",
            problem_pattern=pattern_sig,
            recommended_tactics=tactics_used,
            solver_configuration=config_used,
            expected_performance={
                "avg_time": solving_time,
                "success_rate": 1.0 if success else 0.0
            }
        )
        
        if success:
            strategy.success_count = 1
        else:
            strategy.failure_count = 1
        
        self.strategies[strategy_id] = strategy
        
        return strategy
    
    def _create_problem_signature(self, features: Dict[str, Any]) -> str:
        """Create a signature for problem features."""
        # Simplified signature
        parts = [
            features.get('type', 'unknown'),
            f"vars_{features.get('var_count', 0)}",
            f"constraints_{features.get('constraint_count', 0)}"
        ]
        return "_".join(parts)
    
    def recommend_strategy(
        self,
        problem_features: Dict[str, Any]
    ) -> Optional[SolutionStrategy]:
        """Recommend a strategy for a problem."""
        current_sig = self._create_problem_signature(problem_features)
        
        best_strategy = None
        best_match = 0.0
        
        for strategy in self.strategies.values():
            similarity = self._signature_similarity(
                current_sig,
                strategy.problem_pattern
            )
            
            if similarity > best_match:
                best_match = similarity
                best_strategy = strategy
        
        return best_strategy if best_match > 0.5 else None
    
    def _signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between problem signatures."""
        parts1 = set(sig1.split('_'))
        parts2 = set(sig2.split('_'))
        
        intersection = len(parts1 & parts2)
        union = len(parts1 | parts2)
        
        return intersection / union if union > 0 else 0.0
    
    # =====================================================================
    # Insight Extraction
    # =====================================================================
    
    def extract_insights(
        self,
        solution: Z3SolverResult,
        problem_statement: str
    ) -> List[MathematicalInsight]:
        """
        Extract mathematical insights from a solution.
        
        Args:
            solution: Z3 solution result
            problem_statement: Original problem
            
        Returns:
            List of insights
        """
        insights = []
        
        if not solution.model:
            return insights
        
        # Extract bounds insights
        for var_name, value in solution.model.assignments.items():
            if isinstance(value, (int, float)):
                insight_id = f"insight_bound_{len(self.insights)}"
                
                insight = MathematicalInsight(
                    insight_id=insight_id,
                    category="bound",
                    statement=f"Variable {var_name} has value {value}",
                    formal_representation=f"{var_name} = {value}",
                    confidence=0.9,
                    derived_from=[problem_statement[:50]]
                )
                
                self.insights[insight_id] = insight
                insights.append(insight)
        
        return insights
    
    def find_related_insights(
        self,
        category: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> List[MathematicalInsight]:
        """Find insights matching criteria."""
        results = []
        
        for insight in self.insights.values():
            if category and insight.category != category:
                continue
            
            if insight.confidence < min_confidence:
                continue
            
            results.append(insight)
        
        return sorted(results, key=lambda i: i.confidence, reverse=True)
    
    # =====================================================================
    # Knowledge Management
    # =====================================================================
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of extracted knowledge."""
        return {
            "proof_patterns": {
                "count": len(self.proof_patterns),
                "top_patterns": [
                    p.to_dict() for p in sorted(
                        self.proof_patterns.values(),
                        key=lambda x: x.usage_count,
                        reverse=True
                    )[:5]
                ]
            },
            "constraint_patterns": {
                "count": len(self.constraint_patterns),
                "by_type": self._count_constraint_types()
            },
            "strategies": {
                "count": len(self.strategies),
                "avg_success_rate": self._avg_strategy_success()
            },
            "insights": {
                "count": len(self.insights),
                "by_category": self._count_insight_categories()
            },
            "statistics": {
                "extractions": self.extraction_count,
                "pattern_matches": self.pattern_matches
            }
        }
    
    def _count_constraint_types(self) -> Dict[str, int]:
        """Count constraint patterns by type."""
        counts = defaultdict(int)
        for pattern in self.constraint_patterns.values():
            counts[pattern.pattern_type] += 1
        return dict(counts)
    
    def _avg_strategy_success(self) -> float:
        """Calculate average strategy success rate."""
        if not self.strategies:
            return 0.0
        
        rates = [s.success_rate() for s in self.strategies.values()]
        return sum(rates) / len(rates)
    
    def _count_insight_categories(self) -> Dict[str, int]:
        """Count insights by category."""
        counts = defaultdict(int)
        for insight in self.insights.values():
            counts[insight.category] += 1
        return dict(counts)
    
    def export_knowledge(self, format: str = "json") -> str:
        """Export knowledge base."""
        data = {
            "proof_patterns": [p.to_dict() for p in self.proof_patterns.values()],
            "constraint_patterns": [p.to_dict() for p in self.constraint_patterns.values()],
            "strategies": [s.to_dict() for s in self.strategies.values()],
            "insights": [i.to_dict() for i in self.insights.values()]
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        else:
            return str(data)
    
    def import_knowledge(self, data: Dict[str, Any]):
        """Import knowledge base."""
        # Import proof patterns
        for p_data in data.get("proof_patterns", []):
            pattern = ProofPattern(
                pattern_id=p_data["id"],
                name=p_data["name"],
                description=p_data.get("description", ""),
                tactic_sequence=p_data.get("tactics", []),
                applicable_domains=p_data.get("domains", [])
            )
            self.proof_patterns[pattern.pattern_id] = pattern
        
        # Import strategies
        for s_data in data.get("strategies", []):
            strategy = SolutionStrategy(
                strategy_id=s_data["id"],
                name=s_data["name"],
                problem_pattern=s_data.get("pattern", ""),
                recommended_tactics=s_data.get("tactics", [])
            )
            self.strategies[strategy.strategy_id] = strategy


# =============================================================================
# Global Instance
# =============================================================================

_knowledge_extractor: Optional[Z3KnowledgeExtractor] = None


def get_z3_knowledge_extractor() -> Z3KnowledgeExtractor:
    """Get global knowledge extractor."""
    global _knowledge_extractor
    if _knowledge_extractor is None:
        _knowledge_extractor = Z3KnowledgeExtractor()
    return _knowledge_extractor


# =============================================================================
# Example Usage
# =============================================================================

def example_knowledge_extraction():
    """Example: Knowledge extraction."""
    extractor = get_z3_knowledge_extractor()
    
    # Learn from a strategy
    strategy = extractor.learn_strategy(
        problem_features={
            "type": "linear",
            "var_count": 5,
            "constraint_count": 10
        },
        tactics_used=["simplify", "solve-eqs", "smt"],
        config_used={"timeout": 30, "threads": 4},
        success=True,
        solving_time=2.5
    )
    
    print(f"Learned strategy: {strategy.name}")
    print(f"Tactics: {strategy.recommended_tactics}")
    
    # Analyze constraints
    constraints = [
        "(> x 0)",
        "(< x 10)",
        "(= y (+ x 5))",
        "(> (* x y) 0)"
    ]
    
    patterns = extractor.analyze_constraints(constraints, 1.5, True)
    print(f"\nFound {len(patterns)} constraint patterns")
    for p in patterns:
        print(f"  {p.pattern_type}: complexity={p.complexity_score}")
    
    # Get summary
    summary = extractor.get_knowledge_summary()
    print(f"\nKnowledge Summary:")
    print(f"  Proof patterns: {summary['proof_patterns']['count']}")
    print(f"  Strategies: {summary['strategies']['count']}")
    print(f"  Insights: {summary['insights']['count']}")


if __name__ == "__main__":
    print("Z3 Knowledge Extraction")
    print("=" * 50)
    example_knowledge_extraction()
