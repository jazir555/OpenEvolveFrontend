"""
Knowledge Graph Z3 Connector

Stores Z3 proofs, theorems, and tactics in the knowledge graph for:
- Proof reuse via graph traversal
- Theorem relationship discovery
- Tactic recommendation based on graph patterns
- Mathematical knowledge extraction

Integrates with:
- ai-knowledge-graph/
- knowledge_graph_visualizer.py
- z3_knowledge_extraction.py

Author: OpenEvolve
Created: 2026-02-02
"""
from __future__ import annotations


import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


try:
    from z3prover_integration import Z3SolverResult, Z3TheoremResult
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

# CAV-NLP Integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False


@dataclass
class TheoremNode:
    """Knowledge graph node for a theorem."""
    node_id: str
    statement: str
    hash: str
    domain: str = "general"
    complexity_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": "theorem",
            "statement": self.statement[:200] + "..." if len(self.statement) > 200 else self.statement,
            "hash": self.hash,
            "domain": self.domain,
            "complexity": self.complexity_score
        }


@dataclass
class ProofNode:
    """Knowledge graph node for a proof."""
    node_id: str
    theorem_id: str
    proof_text: str
    tactics_used: List[str] = field(default_factory=list)
    is_verified: bool = False
    verification_method: str = "z3"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": "proof",
            "theorem_id": self.theorem_id,
            "tactics": self.tactics_used,
            "verified": self.is_verified
        }


@dataclass
class TacticNode:
    """Knowledge graph node for a proof tactic."""
    node_id: str
    tactic_name: str
    description: str
    applicable_domains: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": "tactic",
            "name": self.tactic_name,
            "description": self.description,
            "domains": self.applicable_domains,
            "success_rate": self.success_count / (self.success_count + self.failure_count) if (self.success_count + self.failure_count) > 0 else 0.0
        }


class KnowledgeGraphZ3Connector:
    """
    Connects Z3 proofs to the knowledge graph.
    
    Enables:
    - Storing theorems and proofs as graph entities
    - Linking related theorems
    - Recommending tactics based on graph patterns
    - Querying proof history
    """
    
    def __init__(self, use_cav_nlp: bool = True):
        """Initialize connector.
        
        Args:
            use_cav_nlp: Enable CAV-NLP enhanced query formalization
        """
        self.theorems: Dict[str, TheoremNode] = {}
        self.proofs: Dict[str, ProofNode] = {}
        self.tactics: Dict[str, TacticNode] = {}
        self.edges: List[Dict[str, str]] = []  # (source, target, relation)
        
        # CAV-NLP integration
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        self.math_service = None
        self.enhanced_solver = None
        if self.use_cav_nlp:
            try:
                self.math_service = UnifiedMathService()
                self.enhanced_solver = EnhancedZ3Solver()
                logger.info("CAV-NLP integration initialized for knowledge graph connector")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP: {e}")
                self.use_cav_nlp = False
    
    def add_theorem(
        self,
        statement: str,
        domain: str = "general",
        related_theorems: Optional[List[str]] = None
    ) -> TheoremNode:
        """Add a theorem to the knowledge graph."""
        # Generate hash and ID
        stmt_hash = hashlib.sha256(statement.encode()).hexdigest()[:16]
        theorem_id = f"theorem_{stmt_hash}"
        
        # Check if already exists
        if theorem_id in self.theorems:
            return self.theorems[theorem_id]
        
        # Create node
        theorem = TheoremNode(
            node_id=theorem_id,
            statement=statement,
            hash=stmt_hash,
            domain=domain,
            complexity_score=self._calculate_complexity(statement)
        )
        
        self.theorems[theorem_id] = theorem
        
        # Add edges to related theorems
        if related_theorems:
            for related_id in related_theorems:
                self.edges.append({
                    "source": theorem_id,
                    "target": related_id,
                    "relation": "related_to"
                })
        
        return theorem
    
    def add_proof(
        self,
        theorem_id: str,
        proof_text: str,
        tactics_used: List[str],
        is_verified: bool = True
    ) -> ProofNode:
        """Add a proof to the knowledge graph."""
        proof_hash = hashlib.sha256(proof_text.encode()).hexdigest()[:16]
        proof_id = f"proof_{proof_hash}"
        
        proof = ProofNode(
            node_id=proof_id,
            theorem_id=theorem_id,
            proof_text=proof_text,
            tactics_used=tactics_used,
            is_verified=is_verified
        )
        
        self.proofs[proof_id] = proof
        
        # Add edge from theorem to proof
        self.edges.append({
            "source": theorem_id,
            "target": proof_id,
            "relation": "has_proof"
        })
        
        # Add edges from proof to tactics
        for tactic_name in tactics_used:
            tactic_id = f"tactic_{tactic_name}"
            self.edges.append({
                "source": proof_id,
                "target": tactic_id,
                "relation": "uses_tactic"
            })
            
            # Create tactic node if not exists
            if tactic_id not in self.tactics:
                self.tactics[tactic_id] = TacticNode(
                    node_id=tactic_id,
                    tactic_name=tactic_name,
                    description=f"Tactic: {tactic_name}"
                )
            
            # Update tactic success count
            if is_verified:
                self.tactics[tactic_id].success_count += 1
        
        return proof
    
    def find_similar_theorems(self, statement: str, threshold: float = 0.8) -> List[TheoremNode]:
        """Find theorems similar to a given statement."""
        similar = []
        stmt_hash = hashlib.sha256(statement.encode()).hexdigest()[:16]
        
        for theorem in self.theorems.values():
            # Simple hash-based similarity
            if theorem.hash == stmt_hash:
                similar.append(theorem)
            # Could add more sophisticated similarity measures
        
        return similar
    
    def recommend_tactics(self, theorem_statement: str, domain: str = "general") -> List[TacticNode]:
        """Recommend tactics based on theorem characteristics."""
        recommendations = []
        
        # Find similar theorems
        similar = self.find_similar_theorems(theorem_statement)
        
        # Collect tactics used in proofs of similar theorems
        tactic_scores = {}
        
        for theorem in similar:
            # Find proofs for this theorem
            for proof in self.proofs.values():
                if proof.theorem_id == theorem.node_id:
                    for tactic_name in proof.tactics_used:
                        tactic_id = f"tactic_{tactic_name}"
                        if tactic_id not in tactic_scores:
                            tactic_scores[tactic_id] = 0
                        tactic_scores[tactic_id] += 1
        
        # Sort by score
        sorted_tactics = sorted(
            tactic_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for tactic_id, score in sorted_tactics[:5]:
            if tactic_id in self.tactics:
                recommendations.append(self.tactics[tactic_id])
        
        return recommendations
    
    def get_proof_chain(self, theorem_id: str) -> List[Dict[str, Any]]:
        """Get the chain of proofs for a theorem."""
        chain = []
        
        # Find all proofs for this theorem
        for proof in self.proofs.values():
            if proof.theorem_id == theorem_id:
                chain.append(proof.to_dict())
        
        return chain
    
    def export_graph(self) -> Dict[str, Any]:
        """Export knowledge graph as JSON."""
        return {
            "nodes": {
                "theorems": [t.to_dict() for t in self.theorems.values()],
                "proofs": [p.to_dict() for p in self.proofs.values()],
                "tactics": [t.to_dict() for t in self.tactics.values()]
            },
            "edges": self.edges,
            "metadata": {
                "total_nodes": len(self.theorems) + len(self.proofs) + len(self.tactics),
                "total_edges": len(self.edges)
            }
        }
    
    def _calculate_complexity(self, statement: str) -> float:
        """Calculate complexity score for a theorem statement."""
        # Simple heuristic: count quantifiers, operators, etc.
        complexity = 0.0
        
        # Count logical operators
        complexity += statement.count("forall") * 2
        complexity += statement.count("exists") * 2
        complexity += statement.count("and") * 0.5
        complexity += statement.count("or") * 0.5
        complexity += statement.count("implies") * 1
        
        # Count arithmetic operators
        complexity += statement.count("+") * 0.3
        complexity += statement.count("-") * 0.3
        complexity += statement.count("*") * 0.5
        
        return complexity

    def query_with_cav_nlp(self, natural_language_query: str) -> List[Dict[str, Any]]:
        """Query knowledge graph using CAV-NLP formalization.
        
        Args:
            natural_language_query: Natural language query to formalize and execute
            
        Returns:
            List of matching theorem/proof/tactic dictionaries
        """
        if self.use_cav_nlp and self.math_service:
            try:
                # Convert NL to formal query
                formalized = self.math_service.formalize(natural_language_query)
                
                # Execute formal query if code is available
                if hasattr(formalized, 'code') and formalized.code:
                    return self.query(formalized.code)
            except Exception as e:
                logger.warning(f"CAV-NLP query failed: {e}, falling back to keyword search")
        
        # Fallback to keyword-based search
        results = []
        query_lower = natural_language_query.lower()
        
        # Search theorems
        for theorem in self.theorems.values():
            if query_lower in theorem.statement.lower():
                results.append(theorem.to_dict())
        
        # Search proofs
        for proof in self.proofs.values():
            if query_lower in proof.proof_text.lower():
                results.append(proof.to_dict())
        
        # Search tactics
        for tactic in self.tactics.values():
            if query_lower in tactic.tactic_name.lower() or query_lower in tactic.description.lower():
                results.append(tactic.to_dict())
        
        return results
    
    def query(self, formal_code: str) -> List[Dict[str, Any]]:
        """Execute formal query against knowledge graph.
        
        Args:
            formal_code: Formal query code
            
        Returns:
            List of matching results
        """
        results = []
        code_lower = formal_code.lower()
        
        # Simple pattern matching on formal code
        for theorem in self.theorems.values():
            if any(term in theorem.statement.lower() for term in code_lower.split()):
                results.append(theorem.to_dict())
        
        for proof in self.proofs.values():
            if any(term in proof.proof_text.lower() for term in code_lower.split()):
                results.append(proof.to_dict())
        
        return results


def get_knowledge_graph_z3_connector():
    """Get global knowledge graph Z3 connector."""
    return KnowledgeGraphZ3Connector()


if __name__ == "__main__":
    print("Knowledge Graph Z3 Connector initialized")
