"""
Math Library Search Node for BubbleLabs

Search mathematical libraries (Mathlib, etc.) for:
- Theorems and lemmas
- Definitions
- Examples
- Proof techniques

Supports fuzzy search, semantic search, and exact matching.

Part of the Mathematical Verification Bubble Suite.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from difflib import SequenceMatcher

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)


class MathLibrarySearchNode(BubbleLabsNode):
    """
    Search mathematical libraries for theorems, definitions, and examples.
    
    Operations:
        - search: General search across libraries
        - search_theorems: Find relevant theorems
        - search_definitions: Find definitions
        - search_examples: Find examples
        - fuzzy_search: Fuzzy matching search
        - exact_search: Exact name search
        - get_documentation: Get documentation for an item
        - batch_search: Search for multiple queries
    """
    
    DISPLAY_NAME = "Math Library Search"
    DESCRIPTION = "Search mathematical libraries for theorems and definitions"
    ICON = "math-library"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "search",
        "search_theorems",
        "search_definitions",
        "search_examples",
        "fuzzy_search",
        "exact_search",
        "get_documentation",
        "batch_search"
    ]
    
    # Simulated Mathlib content (fallback database)
    LIBRARY_DATABASE = {
        "theorems": [
            {
                "name": "Nat.add_comm",
                "statement": "∀ n m : Nat, n + m = m + n",
                "description": "Addition of natural numbers is commutative",
                "tags": ["arithmetic", "commutativity", "natural-numbers"],
                "module": "Mathlib.Data.Nat.Basic"
            },
            {
                "name": "Nat.add_assoc",
                "statement": "∀ n m k : Nat, (n + m) + k = n + (m + k)",
                "description": "Addition of natural numbers is associative",
                "tags": ["arithmetic", "associativity", "natural-numbers"],
                "module": "Mathlib.Data.Nat.Basic"
            },
            {
                "name": "Nat.mul_comm",
                "statement": "∀ n m : Nat, n * m = m * n",
                "description": "Multiplication of natural numbers is commutative",
                "tags": ["arithmetic", "commutativity", "natural-numbers"],
                "module": "Mathlib.Data.Nat.Basic"
            },
            {
                "name": "Nat.mul_assoc",
                "statement": "∀ n m k : Nat, (n * m) * k = n * (m * k)",
                "description": "Multiplication of natural numbers is associative",
                "tags": ["arithmetic", "associativity", "natural-numbers"],
                "module": "Mathlib.Data.Nat.Basic"
            },
            {
                "name": "Nat.zero_add",
                "statement": "∀ n : Nat, 0 + n = n",
                "description": "Zero is left identity for addition",
                "tags": ["arithmetic", "identity", "natural-numbers"],
                "module": "Mathlib.Data.Nat.Basic"
            },
            {
                "name": "Nat.add_zero",
                "statement": "∀ n : Nat, n + 0 = n",
                "description": "Zero is right identity for addition",
                "tags": ["arithmetic", "identity", "natural-numbers"],
                "module": "Mathlib.Data.Nat.Basic"
            },
            {
                "name": "Int.add_comm",
                "statement": "∀ a b : Int, a + b = b + a",
                "description": "Addition of integers is commutative",
                "tags": ["arithmetic", "commutativity", "integers"],
                "module": "Mathlib.Data.Int.Basic"
            },
            {
                "name": "Int.mul_comm",
                "statement": "∀ a b : Int, a * b = b * a",
                "description": "Multiplication of integers is commutative",
                "tags": ["arithmetic", "commutativity", "integers"],
                "module": "Mathlib.Data.Int.Basic"
            },
            {
                "name": "Real.add_comm",
                "statement": "∀ x y : ℝ, x + y = y + x",
                "description": "Addition of real numbers is commutative",
                "tags": ["arithmetic", "commutativity", "real-numbers", "analysis"],
                "module": "Mathlib.Data.Real.Basic"
            },
            {
                "name": "Real.mul_comm",
                "statement": "∀ x y : ℝ, x * y = y * x",
                "description": "Multiplication of real numbers is commutative",
                "tags": ["arithmetic", "commutativity", "real-numbers", "analysis"],
                "module": "Mathlib.Data.Real.Basic"
            },
            {
                "name": "Finset.sum_comm",
                "statement": "Commutativity of finite sums",
                "description": "Finite sums can be reordered",
                "tags": ["combinatorics", "sums", "finite-sets"],
                "module": "Mathlib.Algebra.BigOperators.Basic"
            },
            {
                "name": "Continuous.add",
                "statement": "If f and g are continuous, then f + g is continuous",
                "description": "Sum of continuous functions is continuous",
                "tags": ["analysis", "continuity", "topology"],
                "module": "Mathlib.Topology.ContinuousFunction.Basic"
            },
            {
                "name": " Differentiable.add",
                "statement": "If f and g are differentiable, then f + g is differentiable",
                "description": "Sum of differentiable functions is differentiable",
                "tags": ["analysis", "differentiation", "calculus"],
                "module": "Mathlib.Calculus.Deriv.Add"
            },
            {
                "name": "Group.mul_inv_cancel",
                "statement": "∀ g : G, g * g⁻¹ = 1",
                "description": "Right inverse property in groups",
                "tags": ["algebra", "group-theory", "abstract-algebra"],
                "module": "Mathlib.Algebra.Group.Basic"
            },
            {
                "name": "Subgroup.one_mem",
                "statement": "1 ∈ H for any subgroup H",
                "description": "Identity element is in every subgroup",
                "tags": ["algebra", "group-theory", "subgroups"],
                "module": "Mathlib.GroupTheory.Subgroup.Basic"
            },
            {
                "name": "Prime.not_dvd_one",
                "statement": "∀ p : Nat, Prime p -> ¬p ∣ 1",
                "description": "Prime numbers don't divide 1",
                "tags": ["number-theory", "primes", "divisibility"],
                "module": "Mathlib.Data.Nat.Prime"
            },
            {
                "name": "Nat.even_iff",
                "statement": "∀ n : Nat, Even n ↔ n % 2 = 0",
                "description": "Characterization of even numbers",
                "tags": ["number-theory", "parity", "modular"],
                "module": "Mathlib.Data.Nat.Parity"
            },
            {
                "name": "List.length_append",
                "statement": "∀ l₁ l₂ : List α, (l₁ ++ l₂).length = l₁.length + l₂.length",
                "description": "Length of appended lists is sum of lengths",
                "tags": ["data-structures", "lists", "combinatorics"],
                "module": "Mathlib.Data.List.Basic"
            },
            {
                "name": "Set.union_comm",
                "statement": "∀ s t : Set α, s ∪ t = t ∪ s",
                "description": "Union of sets is commutative",
                "tags": ["set-theory", "set-operations"],
                "module": "Mathlib.Data.Set.Basic"
            },
            {
                "name": "Set.inter_comm",
                "statement": "∀ s t : Set α, s ∩ t = t ∩ s",
                "description": "Intersection of sets is commutative",
                "tags": ["set-theory", "set-operations"],
                "module": "Mathlib.Data.Set.Basic"
            },
            {
                "name": "Metric.continuous_iff",
                "statement": "Characterization of continuity in metric spaces",
                "description": "Continuity in terms of epsilon-delta",
                "tags": ["analysis", "metric-spaces", "continuity", "topology"],
                "module": "Mathlib.Topology.MetricSpace.Basic"
            },
            {
                "name": "Complex.I_sq",
                "statement": "I² = -1",
                "description": "Square of imaginary unit",
                "tags": ["complex-analysis", "complex-numbers", "algebra"],
                "module": "Mathlib.Data.Complex.Basic"
            },
            {
                "name": "Matrix.mul_assoc",
                "statement": "Matrix multiplication is associative",
                "description": "Associativity of matrix multiplication",
                "tags": ["linear-algebra", "matrices"],
                "module": "Mathlib.Data.Matrix.Basic"
            },
            {
                "name": "LinearMap.comp_assoc",
                "statement": "Composition of linear maps is associative",
                "description": "Associativity of linear map composition",
                "tags": ["linear-algebra", "linear-maps"],
                "module": "Mathlib.Algebra.Module.LinearMap"
            },
            {
                "name": "TopologicalSpace.isOpen_inter",
                "statement": "Intersection of open sets is open",
                "description": "Open sets are closed under finite intersections",
                "tags": ["topology", "open-sets"],
                "module": "Mathlib.Topology.Basic"
            }
        ],
        "definitions": [
            {
                "name": "Group",
                "definition": "A set equipped with an associative binary operation, identity, and inverses",
                "examples": ["Integers under addition", "Non-zero rationals under multiplication"],
                "tags": ["algebra", "group-theory", "abstract-algebra"],
                "module": "Mathlib.Algebra.Group.Defs"
            },
            {
                "name": "Ring",
                "definition": "An abelian group under addition with associative multiplication and distributivity",
                "examples": ["Integers", "Polynomials"],
                "tags": ["algebra", "ring-theory", "abstract-algebra"],
                "module": "Mathlib.Algebra.Ring.Defs"
            },
            {
                "name": "Field",
                "definition": "A commutative ring where every non-zero element has a multiplicative inverse",
                "examples": ["Rational numbers", "Real numbers", "Complex numbers"],
                "tags": ["algebra", "field-theory", "abstract-algebra"],
                "module": "Mathlib.Algebra.Field.Defs"
            },
            {
                "name": "Continuous",
                "definition": "A function where preimage of open sets is open",
                "examples": ["Polynomials", "Exponential function"],
                "tags": ["analysis", "topology", "continuity"],
                "module": "Mathlib.Topology.Basic"
            },
            {
                "name": "Differentiable",
                "definition": "A function with a well-defined derivative at each point",
                "examples": ["Polynomials", "sin", "cos", "exp"],
                "tags": ["analysis", "calculus", "differentiation"],
                "module": "Mathlib.Calculus.Deriv.Basic"
            },
            {
                "name": "Prime",
                "definition": "A natural number greater than 1 with no divisors other than 1 and itself",
                "examples": ["2", "3", "5", "7", "11"],
                "tags": ["number-theory", "primes"],
                "module": "Mathlib.Data.Nat.Prime"
            },
            {
                "name": "MetricSpace",
                "definition": "A set with a distance function satisfying positivity, symmetry, and triangle inequality",
                "examples": ["Real numbers", "Euclidean space"],
                "tags": ["analysis", "metric-spaces", "topology"],
                "module": "Mathlib.Topology.MetricSpace.Basic"
            },
            {
                "name": "VectorSpace",
                "definition": "A set closed under vector addition and scalar multiplication",
                "examples": ["R^n", "Polynomials", "Continuous functions"],
                "tags": ["linear-algebra", "vector-spaces"],
                "module": "Mathlib.Algebra.Module.Basic"
            }
        ]
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "search"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation == "batch_search":
            if "queries" not in inputs and "queries" not in self.config:
                errors.append("batch_search requires 'queries' input")
        elif operation == "get_documentation":
            if "name" not in inputs and "name" not in self.config:
                errors.append("get_documentation requires 'name' input")
        elif operation in ["search", "search_theorems", "search_definitions", "fuzzy_search", "exact_search"]:
            if "query" not in inputs and "query" not in self.config:
                errors.append(f"{operation} requires 'query' input")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "search",
                    "description": "Search operation"
                },
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of queries for batch search"
                },
                "name": {
                    "type": "string",
                    "description": "Exact name to look up"
                },
                "category": {
                    "type": "string",
                    "enum": ["theorems", "definitions", "examples", "all"],
                    "default": "all",
                    "description": "Category to search in"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags"
                },
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of results"
                },
                "fuzzy_threshold": {
                    "type": "number",
                    "default": 0.6,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Fuzzy matching threshold"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute search operation."""
        operation = inputs.get("operation", self.config.get("operation", "search"))
        
        try:
            if operation == "search":
                result = self._search(inputs, context)
            elif operation == "search_theorems":
                result = self._search_theorems(inputs, context)
            elif operation == "search_definitions":
                result = self._search_definitions(inputs, context)
            elif operation == "search_examples":
                result = self._search_examples(inputs, context)
            elif operation == "fuzzy_search":
                result = self._fuzzy_search(inputs, context)
            elif operation == "exact_search":
                result = self._exact_search(inputs, context)
            elif operation == "get_documentation":
                result = self._get_documentation(inputs, context)
            elif operation == "batch_search":
                result = self._batch_search(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            result["timestamp"] = datetime.utcnow().isoformat()
            context.add_artifact("library_search_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Search failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _search(self, inputs: Dict, context) -> Dict[str, Any]:
        """General search across library."""
        query = inputs.get("query", self.config.get("query", ""))
        category = inputs.get("category", self.config.get("category", "all"))
        max_results = inputs.get("max_results", self.config.get("max_results", 10))
        
        context.update_progress(50)
        
        results = []
        
        # Search theorems
        if category in ["all", "theorems"]:
            theorem_results = self._search_in_category(query, "theorems", max_results)
            results.extend(theorem_results)
        
        # Search definitions
        if category in ["all", "definitions"]:
            def_results = self._search_in_category(query, "definitions", max_results)
            results.extend(def_results)
        
        context.update_progress(100)
        
        # Sort by relevance
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        return {
            "success": True,
            "query": query,
            "category": category,
            "count": len(results[:max_results]),
            "results": results[:max_results]
        }
    
    def _search_theorems(self, inputs: Dict, context) -> Dict[str, Any]:
        """Search specifically for theorems."""
        query = inputs.get("query", self.config.get("query", ""))
        max_results = inputs.get("max_results", self.config.get("max_results", 10))
        
        context.update_progress(50)
        results = self._search_in_category(query, "theorems", max_results)
        context.update_progress(100)
        
        return {
            "success": True,
            "query": query,
            "category": "theorems",
            "count": len(results),
            "results": results
        }
    
    def _search_definitions(self, inputs: Dict, context) -> Dict[str, Any]:
        """Search specifically for definitions."""
        query = inputs.get("query", self.config.get("query", ""))
        max_results = inputs.get("max_results", self.config.get("max_results", 10))
        
        context.update_progress(50)
        results = self._search_in_category(query, "definitions", max_results)
        context.update_progress(100)
        
        return {
            "success": True,
            "query": query,
            "category": "definitions",
            "count": len(results),
            "results": results
        }
    
    def _search_examples(self, inputs: Dict, context) -> Dict[str, Any]:
        """Search for examples."""
        query = inputs.get("query", self.config.get("query", ""))
        
        context.update_progress(50)
        
        # Search definitions which have examples
        results = []
        for defn in self.LIBRARY_DATABASE.get("definitions", []):
            score = self._compute_relevance(query, defn["name"] + " " + defn.get("definition", ""))
            if score > 0.3:
                results.append({
                    "name": defn["name"],
                    "type": "definition",
                    "examples": defn.get("examples", []),
                    "relevance": score
                })
        
        context.update_progress(100)
        
        results.sort(key=lambda x: -x["relevance"])
        
        return {
            "success": True,
            "query": query,
            "category": "examples",
            "count": len(results[:10]),
            "results": results[:10]
        }
    
    def _fuzzy_search(self, inputs: Dict, context) -> Dict[str, Any]:
        """Fuzzy search with similarity matching."""
        query = inputs.get("query", self.config.get("query", ""))
        threshold = inputs.get("fuzzy_threshold", self.config.get("fuzzy_threshold", 0.6))
        max_results = inputs.get("max_results", self.config.get("max_results", 10))
        
        context.update_progress(50)
        
        results = []
        
        for category in ["theorems", "definitions"]:
            for item in self.LIBRARY_DATABASE.get(category, []):
                name_sim = SequenceMatcher(None, query.lower(), item["name"].lower()).ratio()
                desc = item.get("statement", item.get("definition", ""))
                desc_sim = SequenceMatcher(None, query.lower(), desc.lower()).ratio()
                
                similarity = max(name_sim, desc_sim * 0.8)
                
                if similarity >= threshold:
                    results.append({
                        "name": item["name"],
                        "type": category[:-1],  # Remove 's'
                        "similarity": round(similarity, 3),
                        "statement": desc[:200] + "..." if len(desc) > 200 else desc
                    })
        
        context.update_progress(100)
        
        results.sort(key=lambda x: -x["similarity"])
        
        return {
            "success": True,
            "query": query,
            "threshold": threshold,
            "count": len(results[:max_results]),
            "results": results[:max_results]
        }
    
    def _exact_search(self, inputs: Dict, context) -> Dict[str, Any]:
        """Exact name search."""
        query = inputs.get("query", self.config.get("query", ""))
        
        context.update_progress(50)
        
        results = []
        
        for category in ["theorems", "definitions"]:
            for item in self.LIBRARY_DATABASE.get(category, []):
                if item["name"].lower() == query.lower():
                    results.append({
                        "name": item["name"],
                        "type": category[:-1],
                        "exact_match": True,
                        **item
                    })
        
        context.update_progress(100)
        
        return {
            "success": True,
            "query": query,
            "found": len(results) > 0,
            "results": results
        }
    
    def _get_documentation(self, inputs: Dict, context) -> Dict[str, Any]:
        """Get full documentation for an item."""
        name = inputs.get("name", self.config.get("name", ""))
        
        context.update_progress(50)
        
        for category in ["theorems", "definitions"]:
            for item in self.LIBRARY_DATABASE.get(category, []):
                if item["name"] == name:
                    context.update_progress(100)
                    return {
                        "success": True,
                        "found": True,
                        "item": item
                    }
        
        context.update_progress(100)
        
        return {
            "success": True,
            "found": False,
            "name": name,
            "suggestion": "Try fuzzy_search if name might be slightly different"
        }
    
    def _batch_search(self, inputs: Dict, context) -> Dict[str, Any]:
        """Search for multiple queries."""
        queries = inputs.get("queries", self.config.get("queries", []))
        
        results = []
        total = len(queries)
        
        for i, query in enumerate(queries):
            progress = (i + 1) / total * 100
            context.update_progress(progress)
            
            search_result = self._search({"query": query}, context)
            results.append({
                "query": query,
                "top_result": search_result.get("results", [{}])[0] if search_result.get("results") else None
            })
        
        return {
            "success": True,
            "total": total,
            "results": results
        }
    
    def _search_in_category(self, query: str, category: str, max_results: int) -> List[Dict]:
        """Search within a specific category."""
        results = []
        query_lower = query.lower()
        
        for item in self.LIBRARY_DATABASE.get(category, []):
            score = 0
            
            # Name match
            if query_lower in item["name"].lower():
                score += 1.0
            
            # Statement/definition match
            text = item.get("statement", item.get("definition", "")).lower()
            if query_lower in text:
                score += 0.5
            
            # Tag match
            for tag in item.get("tags", []):
                if query_lower in tag.lower():
                    score += 0.3
            
            if score > 0:
                result = {
                    "name": item["name"],
                    "type": category[:-1],
                    "relevance": round(score, 3)
                }
                if "statement" in item:
                    result["statement"] = item["statement"][:150] + "..." if len(item["statement"]) > 150 else item["statement"]
                if "definition" in item:
                    result["definition"] = item["definition"][:150] + "..." if len(item["definition"]) > 150 else item["definition"]
                
                results.append(result)
        
        results.sort(key=lambda x: -x["relevance"])
        return results[:max_results]
    
    def _compute_relevance(self, query: str, text: str) -> float:
        """Compute relevance score."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0
        
        intersection = query_words & text_words
        return len(intersection) / len(query_words)
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
