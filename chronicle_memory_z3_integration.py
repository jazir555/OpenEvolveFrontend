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

Author: OpenEvolve
Created: 2026-02-02
"""

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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "problem_hash": self.problem_hash,
            "problem_type": self.problem_type,
            "result_status": self.result_status,
            "solution": self.solution,
            "execution_time_ms": self.execution_time_ms,
            "tags": self.tags
        }


class ChronicleMemoryZ3Integration:
    """
    Integrates Z3 solver with chronicle memory.
    
    Features:
    - Store Z3 solving results in chronicle
    - Semantic search for similar problems
    - Pattern matching for solution reuse
    - Case-based problem solving
    """
    
    def __init__(self, chronicle: Optional['ChronicleMemory'] = None):
        self.chronicle = chronicle
        self.entries: Dict[str, Z3MemoryEntry] = {}
        self.problem_index: Dict[str, List[str]] = defaultdict(list)  # hash -> entry_ids
        self.tag_index: Dict[str, List[str]] = defaultdict(list)  # tag -> entry_ids
    
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
            tags=tags or []
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
        problem_hash = hashlib.sha256(problem_statement.encode()).hexdigest()[:16]
        
        # Find exact matches by hash
        matching_ids = self.problem_index.get(problem_hash, [])
        
        results = []
        for entry_id in matching_ids[:limit]:
            if entry_id in self.entries:
                results.append(self.entries[entry_id])
        
        # If no exact matches, try keyword matching
        if not results:
            keywords = self._extract_keywords(problem_statement)
            
            for entry in self.entries.values():
                score = self._calculate_similarity(keywords, entry)
                if score > 0.5:
                    results.append(entry)
                
                if len(results) >= limit:
                    break
        
        return results
    
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
            "average_solve_times": defaultdict(list)
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
        
        # Calculate averages
        avg_times = {}
        for prob_type, times in patterns["average_solve_times"].items():
            if times:
                avg_times[prob_type] = sum(times) / len(times)
        
        return {
            "successful_approaches": dict(patterns["successful_approaches"]),
            "problem_categories": dict(patterns["problem_categories"]),
            "average_solve_times_ms": avg_times,
            "total_entries": len(self.entries)
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
        for entry in similar:
            approaches[entry.problem_type] += 1
        
        # Find most common approach
        best_approach = max(approaches.items(), key=lambda x: x[1])
        
        # Calculate confidence
        confidence = best_approach[1] / len(similar)
        
        # Get average solve time
        avg_time = sum(e.execution_time_ms for e in similar) / len(similar)
        
        return {
            "suggestion": f"Based on {len(similar)} similar problems",
            "recommended_approach": best_approach[0],
            "confidence": confidence,
            "expected_solve_time_ms": avg_time,
            "similar_problems": [e.entry_id for e in similar]
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
    
    def export_memory(self) -> Dict[str, Any]:
        """Export all memory entries."""
        return {
            "entries": [e.to_dict() for e in self.entries.values()],
            "statistics": {
                "total_entries": len(self.entries),
                "unique_problems": len(self.problem_index),
                "tags": list(self.tag_index.keys())
            }
        }


def get_chronicle_memory_z3_integration():
    """Get global chronicle memory Z3 integration."""
    return ChronicleMemoryZ3Integration()


if __name__ == "__main__":
    print("Chronicle Memory Z3 Integration initialized")
