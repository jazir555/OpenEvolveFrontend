"""
Math Tactic Recommendation Node for BubbleLabs

Recommends proof tactics based on:
- Current proof goal
- Hypotheses available
- Mathematical domain
- Proof style preferences

Integrates with LeanAide to suggest appropriate tactics.

Part of the Mathematical Verification Bubble Suite.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)


class MathTacticRecommendationNode(BubbleLabsNode):
    """
    Recommend proof tactics for mathematical goals.
    
    Operations:
        - recommend: Recommend tactics for a goal
        - recommend_for_domain: Get tactics for specific domain
        - explain_tactic: Explain how a tactic works
        - suggest_sequence: Suggest tactic sequence
        - analyze_goal: Analyze goal structure
        - compare_tactics: Compare different tactics
        - batch_recommend: Recommend tactics for multiple goals
    """
    
    DISPLAY_NAME = "Math Tactic Recommendation"
    DESCRIPTION = "Recommend proof tactics based on goal and context"
    ICON = "math-tactics"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "recommend",
        "recommend_for_domain",
        "explain_tactic",
        "suggest_sequence",
        "analyze_goal",
        "compare_tactics",
        "batch_recommend"
    ]
    
    # Tactic database with patterns and use cases
    TACTIC_DATABASE = {
        "intro": {
            "description": "Introduce assumptions from implications",
            "patterns": [r'->', r'forall', r'∀', r'assume', r'suppose'],
            "domains": ["logic", "general"],
            "difficulty": "beginner",
            "success_rate": 0.95
        },
        "apply": {
            "description": "Apply a theorem or hypothesis",
            "patterns": [r'apply', r'using', r'by'],
            "domains": ["general"],
            "difficulty": "beginner",
            "success_rate": 0.90
        },
        "simp": {
            "description": "Simplify using rewrite rules",
            "patterns": [r'simplify', r'reduce', r'=', r'calc'],
            "domains": ["algebra", "arithmetic", "general"],
            "difficulty": "beginner",
            "success_rate": 0.88
        },
        "rw": {
            "description": "Rewrite using equations",
            "patterns": [r'rewrite', r'substitute', r'replace', r'=', r'≡'],
            "domains": ["algebra", "general"],
            "difficulty": "intermediate",
            "success_rate": 0.85
        },
        "linarith": {
            "description": "Linear arithmetic solver",
            "patterns": [r'linear', r'inequality', r'<', r'>', r'≤', r'≥', r'+', r'-'],
            "domains": ["arithmetic", "algebra", "analysis"],
            "difficulty": "intermediate",
            "success_rate": 0.92
        },
        "ring": {
            "description": "Ring arithmetic solver",
            "patterns": [r'ring', r'commutative', r'associative', r'distributive', r'polynomial'],
            "domains": ["algebra", "number_theory"],
            "difficulty": "intermediate",
            "success_rate": 0.90
        },
        "field": {
            "description": "Field arithmetic solver",
            "patterns": [r'field', r'division', r'fraction', r'rational'],
            "domains": ["algebra", "analysis"],
            "difficulty": "intermediate",
            "success_rate": 0.88
        },
        "induction": {
            "description": "Proof by induction",
            "patterns": [r'induction', r'inductive', r'base case', r'step'],
            "domains": ["number_theory", "combinatorics", "discrete_math"],
            "difficulty": "advanced",
            "success_rate": 0.80
        },
        "cases": {
            "description": "Case analysis",
            "patterns": [r'cases', r'case analysis', r'disjunction', r'or', r'∨'],
            "domains": ["logic", "general"],
            "difficulty": "intermediate",
            "success_rate": 0.85
        },
        "contradiction": {
            "description": "Proof by contradiction",
            "patterns": [r'contradiction', r'contrapositive', r'not', r'¬', r'false'],
            "domains": ["logic", "general"],
            "difficulty": "intermediate",
            "success_rate": 0.82
        },
        "tauto": {
            "description": "Tautology solver for propositional logic",
            "patterns": [r'tautology', r'propositional', r'and', r'or', r'not', r'∧', r'∨', r'¬'],
            "domains": ["logic"],
            "difficulty": "beginner",
            "success_rate": 0.94
        },
        "finish": {
            "description": "Automated finishing tactic",
            "patterns": [r'finish', r'complete', r'close'],
            "domains": ["general"],
            "difficulty": "beginner",
            "success_rate": 0.75
        },
        "norm_num": {
            "description": "Normalize numerical expressions",
            "patterns": [r'number', r'numerical', r'calculate', r'compute'],
            "domains": ["arithmetic", "number_theory"],
            "difficulty": "beginner",
            "success_rate": 0.93
        },
        "calc": {
            "description": "Calculation block for transitive relations",
            "patterns": [r'calc', r'calculation', r'transitive', r'_'],
            "domains": ["general"],
            "difficulty": "intermediate",
            "success_rate": 0.87
        },
        "ext": {
            "description": "Extensionality",
            "patterns": [r'extensionality', r'equal', r'function', r'set'],
            "domains": ["set_theory", "logic", "category_theory"],
            "difficulty": "advanced",
            "success_rate": 0.78
        },
        "continuity": {
            "description": "Prove continuity",
            "patterns": [r'continuous', r'limit', r'epsilon', r'delta'],
            "domains": ["analysis", "topology"],
            "difficulty": "advanced",
            "success_rate": 0.72
        },
        "differentiability": {
            "description": "Prove differentiability",
            "patterns": [r'differentiable', r'derivative', r'smooth'],
            "domains": ["analysis", "differential_equations"],
            "difficulty": "advanced",
            "success_rate": 0.70
        },
        "measurability": {
            "description": "Prove measurability",
            "patterns": [r'measurable', r'measure', r'sigma-algebra'],
            "domains": ["probability", "analysis"],
            "difficulty": "advanced",
            "success_rate": 0.68
        }
    }
    
    # Tactic combinations for common patterns
    TACTIC_SEQUENCES = {
        "implication_chain": ["intro", "apply"],
        "equality_proof": ["rw", "simp", "ring"],
        "inequality_proof": ["linarith", "norm_num", "calc"],
        "induction_proof": ["induction", "simp", "linarith"],
        "set_equality": ["ext", "intro", "simp"],
        "continuity_proof": ["continuity", "apply", "simp"]
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "recommend"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation == "batch_recommend":
            if "goals" not in inputs and "goals" not in self.config:
                errors.append("batch_recommend requires 'goals' input")
        elif operation in ["recommend", "analyze_goal"]:
            if "goal" not in inputs and "goal" not in self.config:
                errors.append(f"{operation} requires 'goal' input")
        elif operation == "explain_tactic":
            if "tactic" not in inputs and "tactic" not in self.config:
                errors.append("explain_tactic requires 'tactic' input")
        elif operation == "compare_tactics":
            if "tactics" not in inputs and "tactics" not in self.config:
                errors.append("compare_tactics requires 'tactics' input (list)")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "recommend",
                    "description": "Tactic recommendation operation"
                },
                "goal": {
                    "type": "string",
                    "description": "Current proof goal"
                },
                "hypotheses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Available hypotheses"
                },
                "domain": {
                    "type": "string",
                    "description": "Mathematical domain"
                },
                "tactic": {
                    "type": "string",
                    "description": "Tactic name to explain"
                },
                "tactics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tactics to compare"
                },
                "goals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of goals for batch recommendation"
                },
                "skill_level": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "advanced"],
                    "default": "intermediate",
                    "description": "User skill level"
                },
                "max_recommendations": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of tactics to recommend"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute tactic recommendation operation."""
        operation = inputs.get("operation", self.config.get("operation", "recommend"))
        
        try:
            if operation == "recommend":
                result = self._recommend(inputs, context)
            elif operation == "recommend_for_domain":
                result = self._recommend_for_domain(inputs, context)
            elif operation == "explain_tactic":
                result = self._explain_tactic(inputs, context)
            elif operation == "suggest_sequence":
                result = self._suggest_sequence(inputs, context)
            elif operation == "analyze_goal":
                result = self._analyze_goal(inputs, context)
            elif operation == "compare_tactics":
                result = self._compare_tactics(inputs, context)
            elif operation == "batch_recommend":
                result = self._batch_recommend(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            result["timestamp"] = datetime.utcnow().isoformat()
            context.add_artifact("tactic_recommendation", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Tactic recommendation failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _recommend(self, inputs: Dict, context) -> Dict[str, Any]:
        """Recommend tactics for a goal."""
        goal = inputs.get("goal", self.config.get("goal", ""))
        hypotheses = inputs.get("hypotheses", self.config.get("hypotheses", []))
        domain = inputs.get("domain", self.config.get("domain", ""))
        skill_level = inputs.get("skill_level", self.config.get("skill_level", "intermediate"))
        max_rec = inputs.get("max_recommendations", self.config.get("max_recommendations", 5))
        
        context.update_progress(30)
        
        # Score tactics based on goal patterns
        scored_tactics = self._score_tactics_for_goal(goal, domain)
        
        context.update_progress(60)
        
        # Filter by skill level
        filtered = self._filter_by_skill(scored_tactics, skill_level)
        
        context.update_progress(80)
        
        # Select top recommendations
        top_tactics = sorted(filtered, key=lambda x: -x["score"])[:max_rec]
        
        context.update_progress(100)
        
        return {
            "success": True,
            "goal": goal[:100] + "..." if len(goal) > 100 else goal,
            "recommendations": top_tactics,
            "count": len(top_tactics),
            "alternative_approaches": self._suggest_alternatives(goal, top_tactics)
        }
    
    def _recommend_for_domain(self, inputs: Dict, context) -> Dict[str, Any]:
        """Get tactics suitable for a specific domain."""
        domain = inputs.get("domain", self.config.get("domain", "general"))
        
        context.update_progress(50)
        
        domain_tactics = []
        for tactic_name, info in self.TACTIC_DATABASE.items():
            if domain.lower() in info["domains"] or "general" in info["domains"]:
                domain_tactics.append({
                    "tactic": tactic_name,
                    "description": info["description"],
                    "difficulty": info["difficulty"],
                    "success_rate": info["success_rate"]
                })
        
        context.update_progress(100)
        
        # Sort by success rate
        domain_tactics.sort(key=lambda x: -x["success_rate"])
        
        return {
            "success": True,
            "domain": domain,
            "tactics": domain_tactics,
            "count": len(domain_tactics)
        }
    
    def _explain_tactic(self, inputs: Dict, context) -> Dict[str, Any]:
        """Explain how a tactic works."""
        tactic_name = inputs.get("tactic", self.config.get("tactic", ""))
        
        context.update_progress(50)
        
        info = self.TACTIC_DATABASE.get(tactic_name, {})
        
        if not info:
            return {
                "success": False,
                "error": f"Unknown tactic: {tactic_name}",
                "known_tactics": list(self.TACTIC_DATABASE.keys())
            }
        
        context.update_progress(100)
        
        return {
            "success": True,
            "tactic": tactic_name,
            "description": info["description"],
            "use_cases": info.get("patterns", []),
            "domains": info.get("domains", []),
            "difficulty": info.get("difficulty", "unknown"),
            "success_rate": info.get("success_rate", 0),
            "examples": self._get_tactic_examples(tactic_name)
        }
    
    def _suggest_sequence(self, inputs: Dict, context) -> Dict[str, Any]:
        """Suggest a sequence of tactics."""
        goal = inputs.get("goal", self.config.get("goal", ""))
        
        context.update_progress(50)
        
        # Identify pattern and suggest sequence
        sequence = self._identify_sequence_pattern(goal)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "goal": goal[:100] + "..." if len(goal) > 100 else goal,
            "suggested_sequence": sequence,
            "explanation": self._explain_sequence(sequence)
        }
    
    def _analyze_goal(self, inputs: Dict, context) -> Dict[str, Any]:
        """Analyze the structure of a goal."""
        goal = inputs.get("goal", self.config.get("goal", ""))
        
        context.update_progress(50)
        
        analysis = {
            "connectives": self._identify_connectives(goal),
            "quantifiers": self._identify_quantifiers(goal),
            "structure_type": self._classify_structure(goal),
            "suggested_approach": self._suggest_approach(goal)
        }
        
        context.update_progress(100)
        
        return {
            "success": True,
            "goal": goal[:100] + "..." if len(goal) > 100 else goal,
            "analysis": analysis
        }
    
    def _compare_tactics(self, inputs: Dict, context) -> Dict[str, Any]:
        """Compare different tactics."""
        tactics = inputs.get("tactics", self.config.get("tactics", []))
        
        context.update_progress(50)
        
        comparison = []
        for tactic in tactics:
            info = self.TACTIC_DATABASE.get(tactic, {})
            comparison.append({
                "tactic": tactic,
                "description": info.get("description", "Unknown"),
                "success_rate": info.get("success_rate", 0),
                "difficulty": info.get("difficulty", "unknown"),
                "domains": info.get("domains", [])
            })
        
        context.update_progress(100)
        
        # Sort by success rate
        comparison.sort(key=lambda x: -x["success_rate"])
        
        return {
            "success": True,
            "comparison": comparison,
            "best_for_beginners": next((t for t in comparison if t["difficulty"] == "beginner"), None),
            "highest_success": comparison[0] if comparison else None
        }
    
    def _batch_recommend(self, inputs: Dict, context) -> Dict[str, Any]:
        """Recommend tactics for multiple goals."""
        goals = inputs.get("goals", self.config.get("goals", []))
        
        results = []
        total = len(goals)
        
        for i, goal in enumerate(goals):
            progress = (i + 1) / total * 100
            context.update_progress(progress)
            
            rec = self._recommend({"goal": goal}, context)
            results.append({
                "goal": goal[:80] + "..." if len(goal) > 80 else goal,
                "recommendations": rec.get("recommendations", [])[:3]  # Top 3
            })
        
        return {
            "success": True,
            "total": total,
            "results": results
        }
    
    def _score_tactics_for_goal(self, goal: str, domain: str) -> List[Dict]:
        """Score tactics based on goal patterns."""
        goal_lower = goal.lower()
        scored = []
        
        for tactic_name, info in self.TACTIC_DATABASE.items():
            score = 0
            
            # Match patterns
            for pattern in info.get("patterns", []):
                if re.search(pattern, goal_lower):
                    score += 0.3
            
            # Domain match
            if domain.lower() in info.get("domains", []):
                score += 0.4
            elif "general" in info.get("domains", []):
                score += 0.2
            
            # Base success rate
            score += info.get("success_rate", 0) * 0.3
            
            if score > 0:
                scored.append({
                    "tactic": tactic_name,
                    "score": round(score, 3),
                    "description": info["description"],
                    "difficulty": info["difficulty"],
                    "confidence": "high" if score > 0.7 else "medium" if score > 0.4 else "low"
                })
        
        return scored
    
    def _filter_by_skill(self, tactics: List[Dict], skill_level: str) -> List[Dict]:
        """Filter tactics by skill level."""
        difficulty_order = {"beginner": 1, "intermediate": 2, "advanced": 3}
        user_level = difficulty_order.get(skill_level, 2)
        
        filtered = []
        for t in tactics:
            tactic_level = difficulty_order.get(t["difficulty"], 2)
            # Allow tactics at or below user's level, plus some slightly above
            if tactic_level <= user_level + 1:
                filtered.append(t)
        
        return filtered
    
    def _identify_sequence_pattern(self, goal: str) -> List[str]:
        """Identify common proof patterns and suggest sequences."""
        goal_lower = goal.lower()
        
        for pattern_name, sequence in self.TACTIC_SEQUENCES.items():
            # Simple pattern matching
            if pattern_name == "induction_proof" and "induction" in goal_lower:
                return sequence
            elif pattern_name == "equality_proof" and "=" in goal and "∀" in goal:
                return sequence
            elif pattern_name == "implication_chain" and "->" in goal:
                return sequence
            elif pattern_name == "inequality_proof" and any(c in goal for c in "<>"):
                return sequence
        
        # Default sequence
        return ["intro", "apply", "simp"]
    
    def _explain_sequence(self, sequence: List[str]) -> str:
        """Explain what a tactic sequence does."""
        explanations = {
            "intro": "set up the assumptions",
            "apply": "use existing theorems",
            "simp": "simplify expressions",
            "rw": "rewrite using equations",
            "induction": "prove by induction",
            "linarith": "solve linear inequalities"
        }
        
        parts = [explanations.get(t, f"apply {t}") for t in sequence]
        return f"This sequence will: {' -> '.join(parts)}"
    
    def _get_tactic_examples(self, tactic: str) -> List[str]:
        """Get example usages for a tactic."""
        examples = {
            "intro": ["intro h", "intros x y"],
            "apply": ["apply mul_comm", "apply and.intro"],
            "simp": ["simp", "simp [add_comm, mul_assoc]"],
            "rw": ["rw [h]", "rw [<- add_zero]"],
            "linarith": ["linarith", "linarith [h1, h2]"]
        }
        return examples.get(tactic, [f"{tactic} <args>"])
    
    def _suggest_alternatives(self, goal: str, primary: List[Dict]) -> List[str]:
        """Suggest alternative proof approaches."""
        alternatives = []
        
        if not any(t["tactic"] == "contradiction" for t in primary):
            alternatives.append("Try proof by contradiction")
        
        if "induction" not in [t["tactic"] for t in primary] and any(w in goal.lower() for w in ["forall", "∀", "all"]):
            alternatives.append("Consider using induction")
        
        return alternatives
    
    def _identify_connectives(self, goal: str) -> Dict[str, int]:
        """Identify logical connectives in goal."""
        return {
            "implication": len(re.findall(r'->|implies|->', goal)),
            "conjunction": len(re.findall(r'∧|and', goal)),
            "disjunction": len(re.findall(r'∨|or', goal)),
            "negation": len(re.findall(r'¬|not', goal)),
            "equivalence": len(re.findall(r'↔|iff', goal))
        }
    
    def _identify_quantifiers(self, goal: str) -> Dict[str, int]:
        """Identify quantifiers in goal."""
        return {
            "universal": len(re.findall(r'∀|forall', goal)),
            "existential": len(re.findall(r'∃|exists', goal))
        }
    
    def _classify_structure(self, goal: str) -> str:
        """Classify the structure of the goal."""
        if "∀" in goal or "forall" in goal:
            return "universal"
        elif "∃" in goal or "exists" in goal:
            return "existential"
        elif "->" in goal or "implies" in goal:
            return "implicational"
        elif "∧" in goal or "and" in goal:
            return "conjunctive"
        elif "∨" in goal or "or" in goal:
            return "disjunctive"
        else:
            return "atomic"
    
    def _suggest_approach(self, goal: str) -> str:
        """Suggest an overall approach based on goal structure."""
        structure = self._classify_structure(goal)
        
        approaches = {
            "universal": "Use intro to introduce the universal quantifier",
            "existential": "Use existsi to provide a witness",
            "implicational": "Use intro to assume the antecedent",
            "conjunctive": "Split into subgoals with constructor",
            "disjunctive": "Choose left or right with left/right tactics",
            "atomic": "Apply existing theorems or use automation"
        }
        
        return approaches.get(structure, "Analyze and apply appropriate tactic")
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
