"""
Similarity Result for I_mech

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class SimilarityResult:
    """
    Result of mechanistic similarity analysis
    """
    # Scores
    total_score: float              # Overall similarity (0-1)
    structural_score: float         # Graph isomorphism score
    causal_score: float             # Causal mechanism score
    semantic_score: float           # Semantic label score
    intervention_score: float       # Interventional equivalence

    # Mapping
    node_mapping: Dict[str, str]    # Source -> Target node mapping

    # Proof
    proof: Optional[str] = None     # Lean 4 proof script
    proof_verified: bool = False

    # Transferred solution
    transferred_solution: Optional[Any] = None
    validation_result: Optional[Dict] = None

    # Metadata
    timestamp: datetime = None
    computation_time: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def is_above_threshold(self, threshold: float = 0.7) -> bool:
        """Check if similarity score meets threshold"""
        return self.total_score >= threshold

    def get_confidence_interval(self) -> Tuple[float, float]:
        """
        Compute 95% confidence interval for score
        """
        margin = 0.05  # Simplified
        return (max(0, self.total_score - margin), min(1, self.total_score + margin))

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'total_score': self.total_score,
            'structural_score': self.structural_score,
            'causal_score': self.causal_score,
            'semantic_score': self.semantic_score,
            'intervention_score': self.intervention_score,
            'node_mapping': self.node_mapping,
            'proof': self.proof,
            'proof_verified': self.proof_verified,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'computation_time': self.computation_time
        }

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self):
        return (f"SimilarityResult(total={self.total_score:.3f}, "
                f"struct={self.structural_score:.3f}, "
                f"causal={self.causal_score:.3f}, "
                f"mapping_size={len(self.node_mapping)})")
