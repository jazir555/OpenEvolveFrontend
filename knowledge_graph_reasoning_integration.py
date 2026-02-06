"""
Knowledge Graph Integration with Reasoning Systems

Connects the knowledge graph to verification and decision processes,
enabling context-aware decisions and learning from history.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

# Import knowledge graph components
try:
    from bubblelabs_knowledge_integration import (
        KnowledgeGraphVisualizer,
        KnowledgeQueryInterface,
    )
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False

# Import verification engine
try:
    from verification_engine import VerificationEngine
    from expand_z3_verification import ExpandedZ3Verification
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False

# **LEAN INTEGRATION**: Formal verification with Lean alongside Z3
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False

logger = logging.getLogger(__name__)

class VerificationStatus(Enum):
    """Verification status for knowledge."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    DISPROVEN = "disproven"
    UNKNOWN = "unknown"


@dataclass
class KnowledgeVerification:
    """Verification result for knowledge."""
    entity: str
    statement: str
    status: VerificationStatus
    confidence: float
    verification_method: str
    timestamp: datetime
    metadata: Dict[str, Any]


class KnowledgeReasoningIntegration:
    """
    Integration between knowledge graph and reasoning systems.

    Enables:
    - Knowledge-based verification
    - Context-aware decision making
    - Learning from historical knowledge
    - Consistency checking across knowledge
    """

    def __init__(self):
        """Initialize knowledge reasoning integration."""
        self.kg_visualizer: Optional[KnowledgeGraphVisualizer] = None
        self.query_interface: Optional[KnowledgeQueryInterface] = None
        self.verification_engine: Optional[VerificationEngine] = None
        self.expanded_verification: Optional[ExpandedZ3Verification] = None
        self.verified_knowledge: Dict[str, KnowledgeVerification] = {}

        if KNOWLEDGE_GRAPH_AVAILABLE:
            try:
                self.kg_visualizer = KnowledgeGraphVisualizer()
                logger.info("Knowledge graph visualizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize KG visualizer: {e}")

        if VERIFICATION_AVAILABLE:
            try:
                self.verification_engine = VerificationEngine()
                self.expanded_verification = ExpandedZ3Verification()
                logger.info("Verification engines initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize verification: {e}")

    def verify_knowledge_consistency(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify knowledge graph consistency using Z3.

        Args:
            entities: List of entities
            relationships: List of relationships

        Returns:
            Verification result
        """
        if self.expanded_verification:
            return self.expanded_verification.verify_knowledge_graph_consistency(
                entities, relationships
            )
        return {
            'consistent': None,
            'status': 'VERIFICATION_UNAVAILABLE',
            'message': 'Verification not available'
        }

    def verify_statement(
        self,
        statement: str,
        context: Optional[Dict[str, Any]] = None
    ) -> KnowledgeVerification:
        """
        Verify a statement using knowledge and reasoning.

        Args:
            statement: Statement to verify
            context: Optional context

        Returns:
            Verification result
        """
        if not self.verification_engine:
            return KnowledgeVerification(
                entity="unknown",
                statement=statement,
                status=VerificationStatus.UNKNOWN,
                confidence=0.0,
                verification_method="none",
                timestamp=datetime.now(),
                metadata={'reason': 'Verification unavailable'}
            )

        try:
            # Try to verify using Z3
            result = self.verification_engine.verify_formal(
                solution={'statement': statement},
                use_z3=True,
                strategy="adaptive"
            )

            # Determine verification status
            if result.get('verified'):
                status = VerificationStatus.VERIFIED
                confidence = result.get('confidence', 0.8)
            elif result.get('status') == 'UNSAT':
                status = VerificationStatus.DISPROVEN
                confidence = 0.9
            else:
                status = VerificationStatus.UNKNOWN
                confidence = 0.5

            verification = KnowledgeVerification(
                entity=context.get('entity', 'unknown') if context else 'unknown',
                statement=statement,
                status=status,
                confidence=confidence,
                verification_method="z3_formal",
                timestamp=datetime.now(),
                metadata=result
            )

            # Cache verification
            key = self._statement_key(statement)
            self.verified_knowledge[key] = verification

            return verification

        except Exception as e:
            logger.error(f"Failed to verify statement: {e}")
            return KnowledgeVerification(
                entity="unknown",
                statement=statement,
                status=VerificationStatus.UNKNOWN,
                confidence=0.0,
                verification_method="error",
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )

    def _statement_key(self, statement: str) -> str:
        """Generate key for caching verification."""
        import hashlib
        return hashlib.md5(statement.encode()).hexdigest()

    def get_verified_knowledge(
        self,
        entity: Optional[str] = None,
        status: Optional[VerificationStatus] = None,
        min_confidence: float = 0.0
    ) -> List[KnowledgeVerification]:
        """
        Get verified knowledge matching criteria.

        Args:
            entity: Optional entity filter
            status: Optional status filter
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching verifications
        """
        results = []

        for verification in self.verified_knowledge.values():
            # Apply filters
            if entity and verification.entity != entity:
                continue
            if status and verification.status != status:
                continue
            if verification.confidence < min_confidence:
                continue

            results.append(verification)

        # Sort by confidence (highest first)
        results.sort(key=lambda v: v.confidence, reverse=True)
        return results

    def suggest_improvements(
        self,
        component: str,
        problem: str
    ) -> List[Dict[str, Any]]:
        """
        Suggest improvements based on knowledge and verification.

        Args:
            component: Component name
            problem: Problem description

        Returns:
            List of suggestions with confidence scores
        """
        suggestions = []

        # Look for similar verified problems
        verified_problems = self.get_verified_knowledge(
            status=VerificationStatus.VERIFIED,
            min_confidence=0.7
        )

        for verification in verified_problems[:10]:  # Top 10
            # Simple similarity check (could be enhanced with embeddings)
            problem_words = set(problem.lower().split())
            statement_words = set(verification.statement.lower().split())
            similarity = len(problem_words & statement_words) / max(len(problem_words), 1)

            if similarity > 0.3:  # Threshold for relevance
                suggestions.append({
                    'suggestion': verification.statement,
                    'confidence': verification.confidence * similarity,
                    'verification_status': verification.status.value,
                    'source': 'knowledge_graph',
                })

        # Sort by confidence
        suggestions.sort(key=lambda s: s['confidence'], reverse=True)
        return suggestions[:5]  # Top 5

    def check_conflicts(
        self,
        new_knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check for conflicts with existing verified knowledge.

        Args:
            new_knowledge: New knowledge to check

        Returns:
            List of conflicts found
        """
        conflicts = []

        if 'statements' in new_knowledge:
            for statement in new_knowledge['statements']:
                # Verify against existing knowledge
                verification = self.verify_statement(statement)

                # Check if contradicted by existing verified knowledge
                if verification.status == VerificationStatus.DISPROVEN:
                    conflicts.append({
                        'statement': statement,
                        'conflict_type': 'disproven',
                        'confidence': verification.confidence,
                    })

        return conflicts

    def enrich_decision_context(
        self,
        component: str,
        decision_point: str
    ) -> Dict[str, Any]:
        """
        Enrich decision context with relevant knowledge.

        Args:
            component: Component name
            decision_point: Decision point identifier

        Returns:
            Enriched context with knowledge
        """
        context = {
            'component': component,
            'decision_point': decision_point,
            'timestamp': datetime.now().isoformat(),
            'verified_knowledge': [],
            'suggestions': [],
            'conflicts': [],
        }

        # Get relevant verified knowledge
        relevant_knowledge = self.get_verified_knowledge(
            entity=component,
            min_confidence=0.6
        )
        context['verified_knowledge'] = [
            {
                'statement': v.statement,
                'status': v.status.value,
                'confidence': v.confidence,
            }
            for v in relevant_knowledge[:5]
        ]

        # Get suggestions
        suggestions = self.suggest_improvements(component, decision_point)
        context['suggestions'] = suggestions

        return context

    def learn_from_verification(
        self,
        result: Dict[str, Any],
        component: str
    ) -> bool:
        """
        Learn from verification results to improve knowledge.

        Args:
            result: Verification result
            component: Component name

        Returns:
            True if learned successfully
        """
        try:
            # Extract knowledge from verification result
            if result.get('verified'):
                # Create verification record
                if 'statement' in result or 'solution' in result:
                    statement = result.get('statement') or str(result.get('solution', ''))[:500]

                    verification = KnowledgeVerification(
                        entity=component,
                        statement=statement,
                        status=VerificationStatus.VERIFIED,
                        confidence=result.get('confidence', 0.8),
                        verification_method=result.get('method', 'verification'),
                        timestamp=datetime.now(),
                        metadata=result
                    )

                    # Cache verification
                    key = self._statement_key(statement)
                    self.verified_knowledge[key] = verification

                    logger.info(f"Learned from verification: {statement[:100]}...")
                    return True

        except Exception as e:
            logger.error(f"Failed to learn from verification: {e}")

        return False

    def export_verified_knowledge(self, filepath: str):
        """
        Export verified knowledge to file.

        Args:
            filepath: Path to export file
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_verifications': len(self.verified_knowledge),
                'verifications': [
                    {
                        'entity': v.entity,
                        'statement': v.statement,
                        'status': v.status.value,
                        'confidence': v.confidence,
                        'verification_method': v.verification_method,
                        'timestamp': v.timestamp.isoformat(),
                        'metadata': v.metadata,
                    }
                    for v in self.verified_knowledge.values()
                ]
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Exported {len(self.verified_knowledge)} verifications to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export knowledge: {e}")

    def import_verified_knowledge(self, filepath: str):
        """
        Import verified knowledge from file.

        Args:
            filepath: Path to import file
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            for v_data in data.get('verifications', []):
                verification = KnowledgeVerification(
                    entity=v_data['entity'],
                    statement=v_data['statement'],
                    status=VerificationStatus(v_data['status']),
                    confidence=v_data['confidence'],
                    verification_method=v_data['verification_method'],
                    timestamp=datetime.fromisoformat(v_data['timestamp']),
                    metadata=v_data.get('metadata', {})
                )

                key = self._statement_key(verification.statement)
                self.verified_knowledge[key] = verification

            logger.info(f"Imported {len(data.get('verifications', []))} verifications from {filepath}")

        except Exception as e:
            logger.error(f"Failed to import knowledge: {e}")


# Global instance
_knowledge_reasoning: Optional[KnowledgeReasoningIntegration] = None


def get_knowledge_reasoning() -> KnowledgeReasoningIntegration:
    """Get or create the knowledge reasoning integration singleton."""
    global _knowledge_reasoning
    if _knowledge_reasoning is None:
        _knowledge_reasoning = KnowledgeReasoningIntegration()
    return _knowledge_reasoning


__all__ = [
    'VerificationStatus',
    'KnowledgeVerification',
    'KnowledgeReasoningIntegration',
    'get_knowledge_reasoning',
]
