"""
Chronicle Memory Z3 Integration

Stores Z3 solver results in chronicle memory for:
- Retrieval of similar past problems
- Semantic search for Z3 problems
- Pattern learning from solving history
- Case-based reasoning for new problems

Integrates with:
- chronicle_memory.py
- z3_database_models.py
- knowledge_base.py
- CAV-NLP for enhanced canonicalization and retrieval

Author: OpenEvolve
Created: 2026-02-02
"""
from __future__ import annotations



import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from z3prover_integration import Z3SolverResult, Z3TheoremResult
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


try:
    from chronicle_memory import ChronicleMemory
    CHRONICLE_AVAILABLE = True
except ImportError:
    CHRONICLE_AVAILABLE = False
    logger.warning("Chronicle memory not available")

# CAV-NLP Integration
try:
    from openevolve.cav_nlp_integration.adapter import Z3LeanAideBridge, create_z3_lean_bridge
    from openevolve.cav_nlp_integration.data_structures import (
        ConstraintType,
        Z3Constraint,
        Lean4Constraint,
        CanonicalizationResult,
    )
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger.warning("CAV-NLP integration not available for chronicle memory")


@dataclass
class Z3MemoryEntry:
    """A Z3 solving entry for chronicle memory."""
    entry_id: str
    timestamp: datetime
    problem_hash: str
    problem_statement: str
    problem_type: str  # "solve", "optimize", "prove"
    result_status: str
    solution: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0
    tags: List[str] = field(default_factory=list)
    # CAV-NLP enhanced fields
    canonical_form: Optional[str] = None
    constraint_type: Optional[str] = None
    variables: List[str] = field(default_factory=list)
    mathematical_structure: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "problem_hash": self.problem_hash,
            "problem_type": self.problem_type,
            "result_status": self.result_status,
            "solution": self.solution,
            "execution_time_ms": self.execution_time_ms,
            "tags": self.tags,
            "canonical_form": self.canonical_form,
            "constraint_type": self.constraint_type,
            "variables": self.variables,
            "mathematical_structure": self.mathematical_structure
        }


class ChronicleMemoryZ3Integration:
    """
    Integrates Z3 solver with chronicle memory.
    
    Features:
    - Store Z3 solving results in chronicle
    - Semantic search for similar problems
    - Pattern matching for solution reuse
    - Case-based problem solving
    
    CAV-NLP Integration:
    - Enhanced storage with canonical forms
    - Semantic matching using CAV-NLP canonicalization
    - Better retrieval for mathematically similar problems
    """
    
    def __init__(self, chronicle: Optional['ChronicleMemory'] = None):
        self.chronicle = chronicle
        self.entries: Dict[str, Z3MemoryEntry] = {}
        self.problem_index: Dict[str, List[str]] = defaultdict(list)  # hash -> entry_ids
        self.tag_index: Dict[str, List[str]] = defaultdict(list)  # tag -> entry_ids
        
        # CAV-NLP canonical index for semantic matching
        self.canonical_index: Dict[str, List[str]] = defaultdict(list)  # canonical -> entry_ids
        
        # Initialize CAV-NLP bridge
        self.cav_nlp_bridge = None
        self._cav_nlp_available = False
        if CAV_NLP_AVAILABLE:
            try:
                self.cav_nlp_bridge = create_z3_lean_bridge()
                self._cav_nlp_available = True
                logger.info("CAV-NLP bridge initialized for chronicle memory")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP bridge: {e}")
    
    def store_result(
        self,
        problem_statement: str,
        problem_type: str,
        result: Any,
        tags: Optional[List[str]] = None
    ) -> Z3MemoryEntry:
        """
        Store a Z3 solving result in chronicle memory.
        
        Args:
            problem_statement: The problem that was solved
            problem_type: Type of problem (solve, optimize, prove)
            result: Z3SolverResult or Z3TheoremResult
            tags: Optional tags for categorization
            
        Returns:
            Z3MemoryEntry
        """
        # Generate hash
        problem_hash = hashlib.sha256(problem_statement.encode()).hexdigest()[:16]
        entry_id = f"z3_{problem_hash}_{int(datetime.utcnow().timestamp())}"
        
        # Extract result info
        result_status = "unknown"
        solution = None
        execution_time = 0.0
        
        if hasattr(result, 'status'):
            result_status = result.status.value if hasattr(result.status, 'value') else str(result.status)
        elif hasattr(result, 'proven'):
            result_status = "proven" if result.proven else "not_proven"
        
        if hasattr(result, 'model') and result.model:
            solution = result.model.assignments if hasattr(result.model, 'assignments') else str(result.model)
        
        if hasattr(result, 'execution_time'):
            execution_time = result.execution_time
        
        # CAV-NLP enhancement: determine constraint type and variables
        constraint_type = self._determine_constraint_type(problem_statement)
        variables = self._extract_variables(problem_statement)
        mathematical_structure = self._extract_structure(problem_statement)
        
        # Create entry
        entry = Z3MemoryEntry(
            entry_id=entry_id,
            timestamp=datetime.utcnow(),
            problem_hash=problem_hash,
            problem_statement=problem_statement[:500],  # Truncate for storage
            problem_type=problem_type,
            result_status=result_status,
            solution=solution,
            execution_time_ms=execution_time * 1000,
            tags=tags or [],
            constraint_type=constraint_type,
            variables=variables,
            mathematical_structure=mathematical_structure
        )
        
        # Store entry locally
        self.entries[entry_id] = entry
        self.problem_index[problem_hash].append(entry_id)
        
        # Index tags
        for tag in (tags or []):
            self.tag_index[tag].append(entry_id)
            
        # Record in chronicle if available
        if self.chronicle and CHRONICLE_AVAILABLE:
            from chronicle_memory import EventType, Outcome
            
            # Since record_event is likely async in a real system but 
            # might be called from sync code, we handle both if needed.
            # For now, we'll try to use it.
            try:
                import asyncio
                
                outcome = Outcome.SUCCESS if result_status in ["sat", "proven"] else Outcome.FAILURE
                if result_status == "unknown":
                    outcome = Outcome.UNKNOWN
                
                async def record():
                    await self.chronicle.record_event(
                        event_type=EventType.VERIFICATION_DONE,
                        action=f"z3_{problem_type}",
                        parameters=entry.to_dict(),
                        outcome=outcome,
                        narrative=f"Z3 {problem_type} completed with status {result_status}"
                    )
                
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(record())
                    else:
                        loop.run_until_complete(record())
                except RuntimeError:
                    asyncio.run(record())
            except Exception as e:
                logger.warning(f"Failed to record Z3 event in chronicle: {e}")
        
        return entry
    
    def store_with_canonicalization(
        self,
        constraint: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Z3MemoryEntry:
        """Store constraint with canonical form for better retrieval.
        
        This method uses CAV-NLP to generate a canonical form of the constraint,
        enabling semantic matching and improved retrieval of similar problems.
        
        Args:
            constraint: Constraint to store (string, Z3 expression, or dict)
            metadata: Optional metadata including:
                - problem_type: Type of problem
                - result: Z3SolverResult or Z3TheoremResult
                - tags: List of tags
                - problem_statement: Original problem statement
                
        Returns:
            Z3MemoryEntry with canonical form
        """
        metadata = metadata or {}
        problem_type = metadata.get('problem_type', 'solve')
        result = metadata.get('result')
        tags = metadata.get('tags', [])
        problem_statement = metadata.get(
            'problem_statement',
            str(constraint) if not isinstance(constraint, str) else constraint
        )
        
        # Store basic result first
        entry = self.store_result(
            problem_statement=problem_statement,
            problem_type=problem_type,
            result=result,
            tags=tags
        )
        
        # CAV-NLP enhancement: generate canonical form
        if self._cav_nlp_available and self.cav_nlp_bridge is not None:
            try:
                canonical_form = self._generate_canonical_form(constraint)
                if canonical_form:
                    entry.canonical_form = canonical_form
                    
                    # Index by canonical form for semantic matching
                    canonical_key = self._canonical_to_key(canonical_form)
                    self.canonical_index[canonical_key].append(entry.entry_id)
                    
                    logger.debug(f"Generated canonical form for entry {entry.entry_id}")
            except Exception as e:
                logger.warning(f"CAV-NLP canonicalization failed: {e}")
        
        return entry
    
    def retrieve_similar(
        self,
        query: Any,
        limit: int = 5,
        use_canonical: bool = True
    ) -> List[Z3MemoryEntry]:
        """Retrieve similar constraints using canonical forms.
        
        Uses CAV-NLP canonicalization to find mathematically similar
        problems, even if they have different surface forms.
        
        Args:
            query: Query constraint (string or Z3 expression)
            limit: Maximum number of results
            use_canonical: Whether to use CAV-NLP canonical matching
            
        Returns:
            List of similar Z3MemoryEntry objects
        """
        query_str = str(query)
        
        # First try canonical matching if CAV-NLP is available
        if use_canonical and self._cav_nlp_available and self.cav_nlp_bridge is not None:
            try:
                canonical_matches = self._retrieve_by_canonical_form(query_str, limit)
                if canonical_matches:
                    return canonical_matches
            except Exception as e:
                logger.debug(f"Canonical retrieval failed: {e}")
        
        # Fallback to traditional similarity methods
        return self._retrieve_by_traditional_methods(query_str, limit)
    
    def _retrieve_by_canonical_form(
        self,
        query_str: str,
        limit: int
    ) -> Optional[List[Z3MemoryEntry]]:
        """Retrieve entries matching the canonical form of the query."""
        try:
            # Generate canonical form for query
            query_canonical = self._generate_canonical_form(query_str)
            if not query_canonical:
                return None
            
            canonical_key = self._canonical_to_key(query_canonical)
            
            # Look for exact canonical matches
            matching_ids = self.canonical_index.get(canonical_key, [])
            
            results = []
            for entry_id in matching_ids[:limit]:
                if entry_id in self.entries:
                    results.append(self.entries[entry_id])
            
            # If no exact matches, try partial canonical matching
            if not results:
                results = self._partial_canonical_match(query_canonical, limit)
            
            return results if results else None
            
        except Exception as e:
            logger.debug(f"Canonical form retrieval error: {e}")
            return None
    
    def _partial_canonical_match(
        self,
        query_canonical: str,
        limit: int
    ) -> List[Z3MemoryEntry]:
        """Find entries with similar canonical forms."""
        results = []
        query_parts = set(query_canonical.split())
        
        for entry in self.entries.values():
            if entry.canonical_form:
                entry_parts = set(entry.canonical_form.split())
                
                # Calculate Jaccard similarity
                intersection = len(query_parts & entry_parts)
                union = len(query_parts | entry_parts)
                
                if union > 0 and intersection / union > 0.5:
                    results.append((entry, intersection / union))
        
        # Sort by similarity and return top matches
        results.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in results[:limit]]
    
    def _retrieve_by_traditional_methods(
        self,
        query_str: str,
        limit: int
    ) -> List[Z3MemoryEntry]:
        """Fallback retrieval using hash and keyword matching."""
        # Try hash matching first
        problem_hash = hashlib.sha256(query_str.encode()).hexdigest()[:16]
        matching_ids = self.problem_index.get(problem_hash, [])
        
        results = []
        for entry_id in matching_ids[:limit]:
            if entry_id in self.entries:
                results.append(self.entries[entry_id])
        
        # If no exact matches, try keyword matching
        if not results:
            keywords = self._extract_keywords(query_str)
            
            for entry in self.entries.values():
                score = self._calculate_similarity(keywords, entry)
                if score > 0.5:
                    results.append(entry)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def _generate_canonical_form(self, constraint: Any) -> Optional[str]:
        """Generate canonical form using CAV-NLP."""
        if not self._cav_nlp_available or self.cav_nlp_bridge is None:
            return None
        
        constraint_str = str(constraint)
        
        try:
            # Try canonicalizer first
            canonicalizer = getattr(self.cav_nlp_bridge, 'canonicalizer', None)
            if canonicalizer is not None:
                try:
                    if hasattr(canonicalizer, 'canonicalize_text'):
                        result = canonicalizer.canonicalize_text(constraint_str)
                        if hasattr(result, 'canonical'):
                            return result.canonical
                        return str(result)
                    elif hasattr(canonicalizer, 'canonicalize'):
                        result = canonicalizer.canonicalize(constraint_str)
                        if hasattr(result, 'canonical'):
                            return result.canonical
                        return str(result)
                except Exception as e:
                    logger.debug(f"Canonicalizer failed: {e}")
            
            # Fallback to parser canonicalization
            parser = getattr(self.cav_nlp_bridge, 'parser', None)
            if parser is not None:
                try:
                    if hasattr(parser, 'canonicalize'):
                        result = parser.canonicalize(constraint_str)
                        return str(result)
                    elif hasattr(parser, 'normalize'):
                        result = parser.normalize(constraint_str)
                        return str(result)
                except Exception as e:
                    logger.debug(f"Parser canonicalization failed: {e}")
            
            return None
            
        except Exception as e:
            logger.debug(f"CAV-NLP canonical form generation failed: {e}")
            return None
    
    def _canonical_to_key(self, canonical_form: str) -> str:
        """Convert canonical form to index key."""
        # Normalize and hash for consistent indexing
        normalized = ' '.join(canonical_form.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def find_similar_problems(
        self,
        problem_statement: str,
        limit: int = 5
    ) -> List[Z3MemoryEntry]:
        """
        Find problems similar to the given statement.
        
        Args:
            problem_statement: Problem to match
            limit: Maximum number of results
            
        Returns:
            List of similar Z3MemoryEntry objects
        """
        return self.retrieve_similar(problem_statement, limit, use_canonical=True)
    
    def search_by_tags(
        self,
        tags: List[str],
        limit: int = 10
    ) -> List[Z3MemoryEntry]:
        """Search for entries by tags."""
        matching_ids = set()
        
        for tag in tags:
            for entry_id in self.tag_index.get(tag, []):
                matching_ids.add(entry_id)
        
        results = []
        for entry_id in list(matching_ids)[:limit]:
            if entry_id in self.entries:
                results.append(self.entries[entry_id])
        
        return results
    
    def get_solution_for_problem(
        self,
        problem_statement: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached solution for a problem if available.
        
        Args:
            problem_statement: The problem to solve
            
        Returns:
            Cached solution or None
        """
        similar = self.find_similar_problems(problem_statement, limit=1)
        
        if similar and similar[0].result_status in ["sat", "proven"]:
            return similar[0].solution
        
        return None
    
    def learn_patterns(self) -> Dict[str, Any]:
        """
        Learn patterns from solving history.
        
        Returns:
            Dictionary of learned patterns
        """
        patterns = {
            "successful_approaches": defaultdict(int),
            "common_solutions": defaultdict(int),
            "problem_categories": defaultdict(int),
            "average_solve_times": defaultdict(list),
            "canonical_form_usage": defaultdict(int)
        }
        
        for entry in self.entries.values():
            # Count successful approaches
            if entry.result_status in ["sat", "proven"]:
                patterns["successful_approaches"][entry.problem_type] += 1
            
            # Categorize problems
            category = self._categorize_problem(entry.problem_statement)
            patterns["problem_categories"][category] += 1
            
            # Track solve times
            patterns["average_solve_times"][entry.problem_type].append(entry.execution_time_ms)
            
            # Track canonical form usage
            if entry.canonical_form:
                patterns["canonical_form_usage"]["has_canonical"] += 1
            else:
                patterns["canonical_form_usage"]["no_canonical"] += 1
        
        # Calculate averages
        avg_times = {}
        for prob_type, times in patterns["average_solve_times"].items():
            if times:
                avg_times[prob_type] = sum(times) / len(times)
        
        return {
            "successful_approaches": dict(patterns["successful_approaches"]),
            "problem_categories": dict(patterns["problem_categories"]),
            "average_solve_times_ms": avg_times,
            "total_entries": len(self.entries),
            "canonical_form_stats": dict(patterns["canonical_form_usage"])
        }
    
    def suggest_approach(self, problem_statement: str) -> Dict[str, Any]:
        """Suggest approach based on similar past problems."""
        similar = self.find_similar_problems(problem_statement, limit=3)
        
        if not similar:
            return {
                "suggestion": "No similar problems found",
                "recommended_approach": "standard_solving",
                "confidence": 0.0
            }
        
        # Analyze similar problems
        approaches = defaultdict(int)
        canonical_matches = sum(1 for e in similar if e.canonical_form)
        
        for entry in similar:
            approaches[entry.problem_type] += 1
        
        # Find most common approach
        best_approach = max(approaches.items(), key=lambda x: x[1])
        
        # Calculate confidence
        confidence = best_approach[1] / len(similar)
        
        # Boost confidence if canonical forms matched
        if canonical_matches > 0:
            confidence = min(confidence + 0.1 * canonical_matches, 1.0)
        
        # Get average solve time
        avg_time = sum(e.execution_time_ms for e in similar) / len(similar)
        
        return {
            "suggestion": f"Based on {len(similar)} similar problems",
            "recommended_approach": best_approach[0],
            "confidence": confidence,
            "expected_solve_time_ms": avg_time,
            "similar_problems": [e.entry_id for e in similar],
            "canonical_matches": canonical_matches
        }
    
    def _extract_keywords(self, problem_statement: str) -> List[str]:
        """Extract keywords from problem statement."""
        # Simple keyword extraction
        keywords = []
        
        # Common SMT-LIB keywords
        smt_keywords = ["assert", "declare", "check-sat", "Int", "Real", "Bool"]
        for kw in smt_keywords:
            if kw in problem_statement:
                keywords.append(kw)
        
        return keywords
    
    def _calculate_similarity(self, keywords: List[str], entry: Z3MemoryEntry) -> float:
        """Calculate similarity between keywords and entry."""
        if not keywords:
            return 0.0
        
        entry_keywords = self._extract_keywords(entry.problem_statement)
        
        matches = sum(1 for kw in keywords if kw in entry_keywords)
        return matches / len(keywords)
    
    def _categorize_problem(self, problem_statement: str) -> str:
        """Categorize a problem by its characteristics."""
        if "forall" in problem_statement or "exists" in problem_statement:
            return "quantified"
        elif "Int" in problem_statement:
            return "integer_arithmetic"
        elif "Real" in problem_statement:
            return "real_arithmetic"
        elif "BitVec" in problem_statement:
            return "bit_vector"
        else:
            return "general"
    
    def _determine_constraint_type(self, text: str) -> str:
        """Determine constraint type from text."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['forall', 'exists', '∀', '∃']):
            return "quantified"
        elif any(kw in text_lower for kw in ['array', 'select', 'store']):
            return "array"
        elif any(kw in text_lower for kw in ['bv', 'bitvec', 'extract']):
            return "bitvector"
        elif any(kw in text_lower for kw in ['*', '/', 'pow', 'exp', 'log']):
            return "nonlinear"
        elif any(kw in text_lower for kw in ['+', '-', '<', '>', '<=', '>=']):
            return "arithmetic"
        else:
            return "boolean"
    
    def _extract_variables(self, text: str) -> List[str]:
        """Extract variable names from text."""
        import re
        
        # Extract single letter variables
        matches = re.findall(r'\b[a-zA-Z]\b', text)
        
        # Filter common non-variable letters
        non_vars = {'a', 'i', 'o'}
        return [m for m in matches if m.lower() not in non_vars]
    
    def _extract_structure(self, text: str) -> Dict[str, Any]:
        """Extract mathematical structure from text."""
        text_lower = text.lower()
        
        structure = {
            "has_quantifiers": any(kw in text_lower for kw in ['forall', 'exists']),
            "has_arithmetic": any(kw in text for kw in ['+', '-', '*', '/']),
            "has_comparisons": any(kw in text for kw in ['<', '>', '=', '<=', '>=']),
            "length": len(text)
        }
        
        return structure
    
    def export_memory(self) -> Dict[str, Any]:
        """Export all memory entries."""
        return {
            "entries": [e.to_dict() for e in self.entries.values()],
            "statistics": {
                "total_entries": len(self.entries),
                "unique_problems": len(self.problem_index),
                "tags": list(self.tag_index.keys()),
                "canonical_forms": len(self.canonical_index)
            }
        }


def get_chronicle_memory_z3_integration():
    """Get global chronicle memory Z3 integration."""
    return ChronicleMemoryZ3Integration()


if __name__ == "__main__":
    print("Chronicle Memory Z3 Integration initialized")
    
    # Demo CAV-NLP integration if available
    integration = get_chronicle_memory_z3_integration()
    
    if integration._cav_nlp_available:
        print("\nCAV-NLP integration available!")
        
        # Test store_with_canonicalization
        test_constraint = "x > 0 and y > 0 implies x + y > 0"
        entry = integration.store_with_canonicalization(
            constraint=test_constraint,
            metadata={
                'problem_type': 'prove',
                'tags': ['arithmetic', 'implication']
            }
        )
        
        print(f"\nStored constraint: {test_constraint}")
        print(f"Entry ID: {entry.entry_id}")
        print(f"Canonical form: {entry.canonical_form or 'Not generated'}")
        print(f"Constraint type: {entry.constraint_type}")
        print(f"Variables: {entry.variables}")
        
        # Test retrieve_similar
        similar = integration.retrieve_similar("y > 0 and x > 0 implies y + x > 0")
        print(f"\nRetrieved {len(similar)} similar entries")
        
    else:
        print("\nCAV-NLP integration not available (graceful degradation active)")
