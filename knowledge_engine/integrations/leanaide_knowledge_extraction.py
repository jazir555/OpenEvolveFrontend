"""
LeanAIDE Knowledge Extraction and Management

Extracts structured knowledge from LeanAIDE proof operations:
- Proof tactics and strategies
- Theorem patterns and structures
- Mathematical concepts and relationships
- Proof search patterns
- Verification insights

Integrates with OpenEvolve Knowledge Engine for unified storage.

Author: OpenEvolve
Created: 2026-01-31
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime, timezone
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Import LeanAIDE components
try:
    from leanaide_client import LeanAideClient, LeanAideResult, TaskType
    LEANAIDE_CLIENT_AVAILABLE = True
except ImportError:
    LEANAIDE_CLIENT_AVAILABLE = False
    logger.warning("LeanAIDE client not available")

try:
    from knowledge_engine.integrations.leanaide_integration import (
        LeanAideIntegration,
        LeanAideResult as KELeanAideResult
    )
    LEANAIDE_KE_AVAILABLE = True
except ImportError:
    LEANAIDE_KE_AVAILABLE = False


# =============================================================================
# Knowledge Data Classes
# =============================================================================

@dataclass
class TacticPattern:
    """Pattern of tactics used in proofs."""
    pattern_id: str
    tactic_sequence: List[str]
    applicable_goals: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    avg_proof_length: int = 0
    complexity_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source_theorems: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.pattern_id,
            "tactics": self.tactic_sequence,
            "applicable_goals": self.applicable_goals,
            "success_rate": f"{self.success_rate:.1%}",
            "avg_proof_length": self.avg_proof_length,
            "complexity": self.complexity_score
        }


@dataclass
class TheoremPattern:
    """Pattern extracted from theorem statements."""
    pattern_id: str
    pattern_type: str  # "algebraic", "logical", "arithmetic", "inductive"
    structure_template: str
    variables: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    conclusion_pattern: str = ""
    common_tactics: List[str] = field(default_factory=list)
    frequency: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.pattern_id,
            "type": self.pattern_type,
            "structure": self.structure_template,
            "variables": self.variables,
            "common_tactics": self.common_tactics,
            "frequency": self.frequency
        }


@dataclass
class ProofStrategy:
    """Reusable proof strategy."""
    strategy_id: str
    name: str
    description: str
    target_goal_pattern: str
    tactic_sequence: List[str]
    prerequisites: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    avg_proof_time: float = 0.0
    applicable_domains: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "target_pattern": self.target_goal_pattern,
            "tactics": self.tactic_sequence,
            "success_rate": f"{self.success_rate():.1%}",
            "avg_time": self.avg_proof_time
        }
    
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class MathematicalConcept:
    """Mathematical concept extracted from proofs."""
    concept_id: str
    name: str
    category: str  # "definition", "lemma", "theorem", "property"
    formal_statement: str
    natural_language: str = ""
    dependencies: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    confidence: float = 0.0
    verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.concept_id,
            "name": self.name,
            "category": self.category,
            "formal": self.formal_statement[:100],
            "confidence": f"{self.confidence:.1%}",
            "verified": self.verified
        }


@dataclass
class ProofSearchNode:
    """Node in proof search tree."""
    node_id: str
    goal_state: str
    tactic_applied: Optional[str] = None
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    depth: int = 0
    success_path: bool = False


# =============================================================================
# Knowledge Extractor
# =============================================================================

class LeanAideKnowledgeExtractor:
    """
    Extracts knowledge from LeanAIDE proof operations.
    
    Capabilities:
    - Tactic pattern mining
    - Theorem structure analysis
    - Proof search tree extraction
    - Mathematical concept identification
    - Strategy learning
    """
    
    def __init__(self):
        self.tactic_patterns: Dict[str, TacticPattern] = {}
        self.theorem_patterns: Dict[str, TheoremPattern] = {}
        self.proof_strategies: Dict[str, ProofStrategy] = {}
        self.mathematical_concepts: Dict[str, MathematicalConcept] = {}
        self.proof_search_trees: Dict[str, List[ProofSearchNode]] = {}
        
        # Statistics
        self.extraction_count = 0
        self.theorem_count = 0
        self.proof_count = 0
        
        # Pattern counters
        self._pattern_counter = 0
        
        logger.info("LeanAideKnowledgeExtractor initialized")
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique pattern ID."""
        self._pattern_counter += 1
        return f"{prefix}_{self._pattern_counter:06d}"
    
    # ========================================================================
    # Tactic Pattern Extraction
    # ========================================================================
    
    def extract_tactic_patterns(
        self,
        proof_steps: List[Dict[str, Any]],
        theorem_domain: str = "general"
    ) -> List[TacticPattern]:
        """
        Extract reusable tactic patterns from proof steps.
        
        Args:
            proof_steps: List of proof steps with tactics and goals
            theorem_domain: Domain classification
            
        Returns:
            List of extracted tactic patterns
        """
        patterns = []
        
        if not proof_steps:
            return patterns
        
        # Extract tactic sequences of varying lengths
        for length in range(2, min(len(proof_steps) + 1, 6)):
            for i in range(len(proof_steps) - length + 1):
                sequence = proof_steps[i:i + length]
                tactics = [step.get('tactic', '') for step in sequence]
                goals = [step.get('goal', '') for step in sequence]
                
                # Create pattern
                pattern_id = self._generate_id("tactic")
                pattern = TacticPattern(
                    pattern_id=pattern_id,
                    tactic_sequence=tactics,
                    applicable_goals=goals,
                    avg_proof_length=length,
                    source_theorems=[theorem_domain]
                )
                
                # Calculate complexity
                pattern.complexity_score = self._calculate_tactic_complexity(tactics)
                
                self.tactic_patterns[pattern_id] = pattern
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_tactic_complexity(self, tactics: List[str]) -> float:
        """Calculate complexity score for tactic sequence."""
        complexity = 0.0
        
        for tactic in tactics:
            # Higher complexity for tactics with arguments
            if '(' in tactic and ')' in tactic:
                complexity += 2.0
            # Higher complexity for compound tactics
            elif any(t in tactic.lower() for t in ['simp', 'rewrite', 'induction']):
                complexity += 1.5
            else:
                complexity += 1.0
        
        return complexity / len(tactics) if tactics else 0.0
    
    def find_matching_tactic_pattern(
        self,
        current_goal: str,
        available_tactics: List[str]
    ) -> Optional[TacticPattern]:
        """Find tactic pattern matching current goal."""
        best_match = None
        best_score = 0.0
        
        for pattern in self.tactic_patterns.values():
            # Check if any pattern goal matches current goal
            goal_match = any(
                self._goal_similarity(current_goal, pg) > 0.5
                for pg in pattern.applicable_goals
            )
            
            if goal_match:
                score = pattern.success_rate * (1 / (1 + pattern.complexity_score))
                if score > best_score:
                    best_score = score
                    best_match = pattern
        
        return best_match
    
    def _goal_similarity(self, goal1: str, goal2: str) -> float:
        """Calculate similarity between two goals."""
        # Simple token-based similarity
        tokens1 = set(re.findall(r'\w+', goal1.lower()))
        tokens2 = set(re.findall(r'\w+', goal2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    # ========================================================================
    # Theorem Pattern Extraction
    # ========================================================================
    
    def analyze_theorem_structure(
        self,
        theorem_statement: str,
        proof: Optional[str] = None
    ) -> TheoremPattern:
        """
        Analyze theorem structure and extract pattern.
        
        Args:
            theorem_statement: Formal theorem statement
            proof: Optional proof for tactic extraction
            
        Returns:
            Extracted theorem pattern
        """
        pattern_id = self._generate_id("theorem")
        
        # Determine pattern type
        pattern_type = self._classify_theorem_type(theorem_statement)
        
        # Extract variables
        variables = re.findall(r'\((\w+)\s*:\s*[^)]+\)', theorem_statement)
        
        # Extract hypotheses and conclusion
        hypotheses = []
        conclusion = theorem_statement
        
        if '->' in theorem_statement:
            parts = theorem_statement.split('->')
            hypotheses = [p.strip() for p in parts[:-1]]
            conclusion = parts[-1].strip()
        
        # Extract common tactics from proof
        common_tactics = []
        if proof:
            common_tactics = self._extract_tactics_from_proof(proof)
        
        pattern = TheoremPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            structure_template=self._create_structure_template(theorem_statement),
            variables=variables,
            hypotheses=hypotheses,
            conclusion_pattern=conclusion,
            common_tactics=common_tactics,
            frequency=1
        )
        
        self.theorem_patterns[pattern_id] = pattern
        self.theorem_count += 1
        
        return pattern
    
    def _classify_theorem_type(self, theorem: str) -> str:
        """Classify theorem by type."""
        theorem_lower = theorem.lower()
        
        if any(kw in theorem_lower for kw in ['forall', 'exists', '∃', '∀']):
            return "quantified"
        elif any(kw in theorem_lower for kw in ['induction', 'recursion']):
            return "inductive"
        elif any(kw in theorem_lower for kw in ['+', '-', '*', '/', 'nat', 'int']):
            return "arithmetic"
        elif any(kw in theorem_lower for kw in ['and', 'or', 'not', '->', 'implies']):
            return "logical"
        else:
            return "general"
    
    def _create_structure_template(self, theorem: str) -> str:
        """Create abstract structure template from theorem."""
        # Replace specific values with placeholders
        template = re.sub(r'\b\d+\b', 'N', theorem)
        template = re.sub(r'\b[a-z]\w*\b', 'x', template)
        return template
    
    def _extract_tactics_from_proof(self, proof: str) -> List[str]:
        """Extract tactic names from proof."""
        # Match common tactic patterns
        tactics = re.findall(r'\b([a-zA-Z_]+)\s*[<{\[]', proof)
        return list(set(tactics))[:10]  # Limit to top 10
    
    # ========================================================================
    # Proof Strategy Learning
    # ========================================================================
    
    def learn_proof_strategy(
        self,
        theorem_features: Dict[str, Any],
        tactics_used: List[str],
        proof_time: float,
        success: bool
    ) -> ProofStrategy:
        """
        Learn a new proof strategy from successful proof.
        
        Args:
            theorem_features: Characteristics of theorem
            tactics_used: Tactics that were effective
            proof_time: Time taken to prove
            success: Whether proof succeeded
            
        Returns:
            Learned strategy
        """
        strategy_id = self._generate_id("strategy")
        
        # Create goal pattern signature
        goal_pattern = self._create_goal_signature(theorem_features)
        
        strategy = ProofStrategy(
            strategy_id=strategy_id,
            name=f"Strategy for {theorem_features.get('type', 'unknown')}",
            description=f"Auto-generated strategy based on {len(tactics_used)} tactics",
            target_goal_pattern=goal_pattern,
            tactic_sequence=tactics_used,
            applicable_domains=[theorem_features.get('domain', 'general')]
        )
        
        if success:
            strategy.success_count = 1
            strategy.avg_proof_time = proof_time
        else:
            strategy.failure_count = 1
        
        self.proof_strategies[strategy_id] = strategy
        
        return strategy
    
    def _create_goal_signature(self, features: Dict[str, Any]) -> str:
        """Create signature for goal features."""
        parts = [
            features.get('type', 'unknown'),
            f"vars_{features.get('var_count', 0)}",
            f"hyps_{features.get('hypothesis_count', 0)}"
        ]
        return "_".join(parts)
    
    def recommend_strategy(
        self,
        theorem_features: Dict[str, Any]
    ) -> Optional[ProofStrategy]:
        """Recommend proof strategy for theorem."""
        current_sig = self._create_goal_signature(theorem_features)
        
        best_strategy = None
        best_match = 0.0
        
        for strategy in self.proof_strategies.values():
            similarity = self._signature_similarity(
                current_sig,
                strategy.target_goal_pattern
            )
            
            # Weight by success rate
            weighted_score = similarity * strategy.success_rate()
            
            if weighted_score > best_match:
                best_match = weighted_score
                best_strategy = strategy
        
        return best_strategy if best_match > 0.3 else None
    
    def _signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between signatures."""
        parts1 = set(sig1.split('_'))
        parts2 = set(sig2.split('_'))
        
        intersection = len(parts1 & parts2)
        union = len(parts1 | parts2)
        
        return intersection / union if union > 0 else 0.0
    
    # ========================================================================
    # Mathematical Concept Extraction
    # ========================================================================
    
    def extract_mathematical_concepts(
        self,
        theorem: str,
        proof: str,
        metadata: Optional[Dict] = None
    ) -> List[MathematicalConcept]:
        """
        Extract mathematical concepts from theorem and proof.
        
        Args:
            theorem: Theorem statement
            proof: Proof text
            metadata: Additional metadata
            
        Returns:
            List of extracted concepts
        """
        concepts = []
        
        # Extract definitions used
        definitions = re.findall(r'def\s+(\w+)', proof)
        for i, def_name in enumerate(set(definitions)):
            concept = MathematicalConcept(
                concept_id=self._generate_id("concept"),
                name=def_name,
                category="definition",
                formal_statement=f"def {def_name} := ...",
                confidence=0.8
            )
            concepts.append(concept)
            self.mathematical_concepts[concept.concept_id] = concept
        
        # Extract lemmas/theorems referenced
        lemmas = re.findall(r'(?:apply|exact|use)\s+(\w+)', proof)
        for lemma_name in set(lemmas):
            concept_id = self._generate_id("concept")
            concept = MathematicalConcept(
                concept_id=concept_id,
                name=lemma_name,
                category="lemma",
                formal_statement=f"lemma {lemma_name} := ...",
                confidence=0.7
            )
            concepts.append(concept)
            self.mathematical_concepts[concept_id] = concept
        
        return concepts
    
    # ========================================================================
    # Knowledge Management
    # ========================================================================
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of extracted knowledge."""
        return {
            "tactic_patterns": {
                "count": len(self.tactic_patterns),
                "top_patterns": [
                    p.to_dict() for p in sorted(
                        self.tactic_patterns.values(),
                        key=lambda x: x.success_rate,
                        reverse=True
                    )[:5]
                ]
            },
            "theorem_patterns": {
                "count": len(self.theorem_patterns),
                "by_type": self._count_theorem_types()
            },
            "proof_strategies": {
                "count": len(self.proof_strategies),
                "avg_success_rate": self._avg_strategy_success()
            },
            "mathematical_concepts": {
                "count": len(self.mathematical_concepts),
                "by_category": self._count_concept_categories()
            },
            "statistics": {
                "extractions": self.extraction_count,
                "theorems_analyzed": self.theorem_count,
                "proofs_processed": self.proof_count
            }
        }
    
    def _count_theorem_types(self) -> Dict[str, int]:
        """Count theorem patterns by type."""
        counts = defaultdict(int)
        for pattern in self.theorem_patterns.values():
            counts[pattern.pattern_type] += 1
        return dict(counts)
    
    def _avg_strategy_success(self) -> float:
        """Calculate average strategy success rate."""
        if not self.proof_strategies:
            return 0.0
        rates = [s.success_rate() for s in self.proof_strategies.values()]
        return sum(rates) / len(rates)
    
    def _count_concept_categories(self) -> Dict[str, int]:
        """Count concepts by category."""
        counts = defaultdict(int)
        for concept in self.mathematical_concepts.values():
            counts[concept.category] += 1
        return dict(counts)
    
    def export_knowledge(self, format: str = "json") -> str:
        """Export knowledge base."""
        data = {
            "tactic_patterns": [p.to_dict() for p in self.tactic_patterns.values()],
            "theorem_patterns": [p.to_dict() for p in self.theorem_patterns.values()],
            "proof_strategies": [s.to_dict() for s in self.proof_strategies.values()],
            "mathematical_concepts": [c.to_dict() for c in self.mathematical_concepts.values()]
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        return str(data)


# Global instance
_leanaide_knowledge_extractor: Optional[LeanAideKnowledgeExtractor] = None


def get_leanaide_knowledge_extractor() -> LeanAideKnowledgeExtractor:
    """Get global knowledge extractor."""
    global _leanaide_knowledge_extractor
    if _leanaide_knowledge_extractor is None:
        _leanaide_knowledge_extractor = LeanAideKnowledgeExtractor()
    return _leanaide_knowledge_extractor


# Example usage
def example_extraction():
    """Example: Knowledge extraction."""
    extractor = get_leanaide_knowledge_extractor()
    
    # Extract from proof steps
    proof_steps = [
        {"tactic": "intro", "goal": "forall n, n + 0 = n"},
        {"tactic": "induction", "goal": "n + 0 = n"},
        {"tactic": "simp", "goal": "0 + 0 = 0"},
        {"tactic": "rfl", "goal": "0 = 0"}
    ]
    
    patterns = extractor.extract_tactic_patterns(proof_steps, "arithmetic")
    print(f"Extracted {len(patterns)} tactic patterns")
    
    # Analyze theorem
    theorem = "theorem add_zero : forall (n : Nat), n + 0 = n := by"
    theorem_pattern = extractor.analyze_theorem_structure(theorem)
    print(f"Theorem type: {theorem_pattern.pattern_type}")
    
    # Get summary
    summary = extractor.get_knowledge_summary()
    print(f"\nKnowledge Summary:")
    print(f"  Tactic patterns: {summary['tactic_patterns']['count']}")
    print(f"  Theorem patterns: {summary['theorem_patterns']['count']}")


if __name__ == "__main__":
    example_extraction()
