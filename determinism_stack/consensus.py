"""Advanced consensus engines for non-deterministic LLMs."""

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from .utils import similarity, extract_json

class ConsensusEngine:
    """Implements various consensus strategies for LLM outputs."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def majority_vote(self, candidates: List[str]) -> Dict[str, Any]:
        """Simple majority voting."""
        if not candidates:
            return {"status": "EMPTY", "winner": None, "agreement": 0.0}
            
        counts = Counter(candidates)
        winner, count = counts.most_common(1)[0]
        agreement = count / len(candidates)
        
        return {
            "status": "CONSENSUS" if agreement >= self.threshold else "DIVERGENT",
            "winner": winner,
            "agreement": agreement,
            "counts": dict(counts)
        }

    def similarity_consensus(self, candidates: List[str]) -> Dict[str, Any]:
        """Select candidate with highest average similarity to others."""
        if not candidates:
            return {"status": "EMPTY", "winner": None, "agreement": 0.0}
        if len(candidates) == 1:
            return {"status": "SINGLE", "winner": candidates[0], "agreement": 1.0}

        best_score = -1.0
        winner = candidates[0]
        
        all_scores = []
        for i, c1 in enumerate(candidates):
            scores = [similarity(c1, c2) for j, c2 in enumerate(candidates) if i != j]
            avg_score = sum(scores) / len(scores)
            all_scores.append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                winner = c1
        
        return {
            "status": "CONSENSUS" if best_score >= self.threshold else "DIVERGENT",
            "winner": winner,
            "agreement": best_score,
            "all_scores": all_scores
        }

    def json_consensus(self, candidates: List[str], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Consensus for JSON outputs by parsing and comparing objects."""
        parsed_objects = []
        valid_indices = []
        
        for i, c in enumerate(candidates):
            obj = extract_json(c)
            if obj is not None:
                parsed_objects.append(obj)
                valid_indices.append(i)
                
        if not parsed_objects:
            return {"status": "INVALID_JSON", "winner": None, "agreement": 0.0}
            
        # Serialize to canonical JSON for string comparison
        canonical_strings = [json.dumps(obj, sort_keys=True) for obj in parsed_objects]
        result = self.majority_vote(canonical_strings)
        
        if result["status"] == "CONSENSUS":
            result["winner"] = json.loads(result["winner"])
            return result
            
        # Fallback to similarity on original strings if JSON exact match fails
        return self.similarity_consensus([candidates[i] for i in valid_indices])
