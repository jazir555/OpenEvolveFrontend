"""
Math Problem Classification Node for BubbleLabs

Classifies mathematical problems by:
- Domain (algebra, analysis, number theory, etc.)
- Problem type (theorem, lemma, definition, conjecture)
- Difficulty level
- Required techniques
- Suitable verification approach (Lean vs Z3 vs Hybrid)

Part of the Mathematical Verification Bubble Suite.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)


class MathDomain(Enum):
    """Mathematical domains."""
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    NUMBER_THEORY = "number_theory"
    TOPOLOGY = "topology"
    GEOMETRY = "geometry"
    LOGIC = "logic"
    COMBINATORICS = "combinatorics"
    PROBABILITY = "probability"
    LINEAR_ALGEBRA = "linear_algebra"
    ABSTRACT_ALGEBRA = "abstract_algebra"
    DIFFERENTIAL_EQUATIONS = "differential_equations"
    DISCRETE_MATH = "discrete_math"
    SET_THEORY = "set_theory"
    CATEGORY_THEORY = "category_theory"
    UNKNOWN = "unknown"


class ProblemType(Enum):
    """Types of mathematical problems."""
    THEOREM = "theorem"
    LEMMA = "lemma"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    DEFINITION = "definition"
    CONJECTURE = "conjecture"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    PROOF = "proof"
    PROBLEM = "problem"


class DifficultyLevel(Enum):
    """Difficulty levels."""
    ELEMENTARY = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    RESEARCH = 4
    OPEN_PROBLEM = 5


class MathProblemClassificationNode(BubbleLabsNode):
    """
    Classify mathematical problems for optimal verification strategy.
    
    Operations:
        - classify: Full classification (domain, type, difficulty)
        - classify_domain: Determine mathematical domain
        - classify_type: Determine problem type
        - estimate_difficulty: Estimate difficulty level
        - recommend_approach: Recommend verification approach
        - analyze_complexity: Analyze problem complexity
        - batch_classify: Classify multiple problems
    """
    
    DISPLAY_NAME = "Math Problem Classification"
    DESCRIPTION = "Classify mathematical problems by domain, type, and difficulty"
    ICON = "math-classify"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "classify",
        "classify_domain",
        "classify_type",
        "estimate_difficulty",
        "recommend_approach",
        "analyze_complexity",
        "batch_classify"
    ]
    
    # Domain keywords for classification
    DOMAIN_PATTERNS = {
        MathDomain.ALGEBRA: [
            r'\bequation\b', r'\bpolynomial\b', r'\broot\b', r'\bfactor\b',
            r'\bgroup\b', r'\bring\b', r'\bfield\b', r'\bideal\b'
        ],
        MathDomain.ANALYSIS: [
            r'\bcontinu\w+\b', r'\bdifferentiable\b', r'\blimit\b', r'\bconverg\w+\b',
            r'\bintegral\b', r'\bderivative\b', r'\bsequence\b', r'\bseries\b',
            r'\breal number\b', r'\bcomplex number\b', r'\bharmonic\b'
        ],
        MathDomain.NUMBER_THEORY: [
            r'\bprime\b', r'\bdivisible\b', r'\bmodular\b', r'\bcongruent\b',
            r'\binteger\b', r'\bgcd\b', r'\blcm\b', r'\bdiophantine\b',
            r'\bfermat\b', r'\beuler\b', r'\bmodulo\b'
        ],
        MathDomain.TOPOLOGY: [
            r'\btopolog\w+\b', r'\bcompact\b', r'\bconnected\b', r'\bhausdorff\b',
            r'\bcontinuous\b', r'\bhomeomorphism\b', r'\bopen set\b', r'\bclosed set\b',
            r'\bneighborhood\b', r'\bboundary\b'
        ],
        MathDomain.GEOMETRY: [
            r'\bgeometr\w+\b', r'\btriangle\b', r'\bcircle\b', r'\bangle\b',
            r'\bpoint\b', r'\bline\b', r'\bplane\b', r'\bsurface\b',
            r'\bconvex\b', r'\bmetric\b', r'\bdistance\b'
        ],
        MathDomain.LOGIC: [
            r'\blogic\b', r'\bproposition\b', r'\bpredicate\b', r'\bquantifier\b',
            r'\baxiom\b', r'\bproof\b', r'\btheorem\b', r'\btautology\b',
            r'\bcontradiction\b', r'\bimplication\b'
        ],
        MathDomain.COMBINATORICS: [
            r'\bcombinatorics\b', r'\bpermutation\b', r'\bcombination\b', r'\bbinomial\b',
            r'\bgraph\b', r'\bcounting\b', r'\bselection\b', r'\barrangement\b'
        ],
        MathDomain.PROBABILITY: [
            r'\bprobability\b', r'\brandom\b', r'\bdistribution\b', r'\bexpected\b',
            r'\bvariance\b', r'\bstochastic\b', r'\bmarkov\b', r'\bbayes\b'
        ],
        MathDomain.LINEAR_ALGEBRA: [
            r'\bmatrix\b', r'\bvector\b', r'\beigenvalue\b', r'\beigenvector\b',
            r'\bdeterminant\b', r'\blinear\b', r'\bsubspace\b', r'\bbasis\b',
            r'\bdimension\b', r'\brank\b'
        ],
        MathDomain.ABSTRACT_ALGEBRA: [
            r'\bhomomorphism\b', r'\bisomorphism\b', r'\bautomorphism\b', r'\bsubgroup\b',
            r'\bquotient\b', r'\bmodule\b', r'\brepresentation\b'
        ],
        MathDomain.DIFFERENTIAL_EQUATIONS: [
            r'\bdifferential equation\b', r'\bode\b', r'\bpde\b', r'\bdynamical\b',
            r'\binitial value\b', r'\bboundary value\b', r'\bpartial\b'
        ],
        MathDomain.DISCRETE_MATH: [
            r'\bdiscrete\b', r'\bset\b', r'\brelation\b', r'\bfunction\b',
            r'\bposet\b', r'\blattice\b', r'\b Boolean\b'
        ],
        MathDomain.SET_THEORY: [
            r'\bset theory\b', r'\bordinal\b', r'\bcardinal\b', r'\btransfinite\b',
            r'\bzfc\b', r'\baxiom of choice\b', r'\bcontinuum\b'
        ],
        MathDomain.CATEGORY_THEORY: [
            r'\bcategory\b', r'\bfunctor\b', r'\bnatural transformation\b', r'\badjunction\b',
            r'\blimit\b', r'\bcolimit\b', r'\buniversal\b', r'\byoneda\b'
        ]
    }
    
    # Problem type patterns
    TYPE_PATTERNS = {
        ProblemType.THEOREM: [
            r'\btheorem\b', r'\bprove that\b', r'\bshow that\b', r'\bdemonstrate\b'
        ],
        ProblemType.LEMMA: [
            r'\blemma\b', r'\bauxiliary\b'
        ],
        ProblemType.PROPOSITION: [
            r'\bproposition\b'
        ],
        ProblemType.COROLLARY: [
            r'\bcorollary\b', r'\bfollows from\b'
        ],
        ProblemType.DEFINITION: [
            r'\bdefine\b', r'\bdefinition\b', r'\blet\s+\w+\s+be\b'
        ],
        ProblemType.CONJECTURE: [
            r'\bconjecture\b', r'\bconjectured\b', r'\bconjectural\b', r'\bopen problem\b'
        ],
        ProblemType.EXAMPLE: [
            r'\bexample\b', r'\bfor instance\b', r'\bconsider\b'
        ],
        ProblemType.EXERCISE: [
            r'\bexercise\b', r'\bproblem\s+\d+\b'
        ]
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict:
        """Compile regex patterns for efficiency."""
        compiled = {}
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            compiled[domain] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "classify"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation == "batch_classify":
            if "problems" not in inputs and "problems" not in self.config:
                errors.append("batch_classify requires 'problems' input")
        else:
            if "problem" not in inputs and "problem" not in self.config:
                errors.append(f"{operation} requires 'problem' input")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "classify",
                    "description": "Classification operation"
                },
                "problem": {
                    "type": "string",
                    "description": "Mathematical problem statement"
                },
                "problems": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of problems for batch classification"
                },
                "include_confidence": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include confidence scores"
                },
                "suggest_tags": {
                    "type": "boolean",
                    "default": True,
                    "description": "Suggest semantic tags"
                },
                "detailed_analysis": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include detailed complexity analysis"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute classification operation."""
        operation = inputs.get("operation", self.config.get("operation", "classify"))
        
        try:
            if operation == "classify":
                result = self._classify(inputs, context)
            elif operation == "classify_domain":
                result = self._classify_domain(inputs, context)
            elif operation == "classify_type":
                result = self._classify_type(inputs, context)
            elif operation == "estimate_difficulty":
                result = self._estimate_difficulty(inputs, context)
            elif operation == "recommend_approach":
                result = self._recommend_approach(inputs, context)
            elif operation == "analyze_complexity":
                result = self._analyze_complexity(inputs, context)
            elif operation == "batch_classify":
                result = self._batch_classify(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            result["timestamp"] = datetime.utcnow().isoformat()
            context.add_artifact("classification_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Classification failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _classify(self, inputs: Dict, context) -> Dict[str, Any]:
        """Full classification of a problem."""
        problem = inputs.get("problem", self.config.get("problem", ""))
        include_confidence = inputs.get("include_confidence", self.config.get("include_confidence", True))
        suggest_tags = inputs.get("suggest_tags", self.config.get("suggest_tags", True))
        
        context.update_progress(20)
        
        # Domain classification
        domain_scores = self._score_domains(problem)
        primary_domain = max(domain_scores, key=domain_scores.get)
        
        context.update_progress(40)
        
        # Type classification
        type_scores = self._score_types(problem)
        primary_type = max(type_scores, key=type_scores.get) if type_scores else ProblemType.PROBLEM
        
        context.update_progress(60)
        
        # Difficulty estimation
        difficulty = self._estimate_difficulty_level(problem, primary_domain)
        
        context.update_progress(80)
        
        # Generate tags
        tags = self._generate_tags(problem, primary_domain, primary_type) if suggest_tags else []
        
        context.update_progress(100)
        
        result = {
            "success": True,
            "problem": problem[:200] + "..." if len(problem) > 200 else problem,
            "classification": {
                "domain": primary_domain.value,
                "type": primary_type.value,
                "difficulty": difficulty.name,
                "difficulty_level": difficulty.value
            },
            "tags": tags
        }
        
        if include_confidence:
            result["confidence"] = {
                "domain": domain_scores[primary_domain],
                "type": type_scores.get(primary_type, 0.5)
            }
            result["alternative_domains"] = [
                {"domain": d.value, "score": s}
                for d, s in sorted(domain_scores.items(), key=lambda x: -x[1])[:3]
                if d != primary_domain
            ]
        
        return result
    
    def _classify_domain(self, inputs: Dict, context) -> Dict[str, Any]:
        """Classify only the domain."""
        problem = inputs.get("problem", self.config.get("problem", ""))
        
        context.update_progress(50)
        scores = self._score_domains(problem)
        primary = max(scores, key=scores.get)
        context.update_progress(100)
        
        return {
            "success": True,
            "domain": primary.value,
            "confidence": scores[primary],
            "all_scores": {d.value: round(s, 3) for d, s in scores.items() if s > 0}
        }
    
    def _classify_type(self, inputs: Dict, context) -> Dict[str, Any]:
        """Classify only the problem type."""
        problem = inputs.get("problem", self.config.get("problem", ""))
        
        context.update_progress(50)
        scores = self._score_types(problem)
        primary = max(scores, key=scores.get) if scores else ProblemType.PROBLEM
        context.update_progress(100)
        
        return {
            "success": True,
            "type": primary.value,
            "confidence": scores.get(primary, 0.5)
        }
    
    def _estimate_difficulty(self, inputs: Dict, context) -> Dict[str, Any]:
        """Estimate problem difficulty."""
        problem = inputs.get("problem", self.config.get("problem", ""))
        
        context.update_progress(50)
        difficulty = self._estimate_difficulty_level(problem, MathDomain.UNKNOWN)
        context.update_progress(100)
        
        return {
            "success": True,
            "difficulty": difficulty.name,
            "level": difficulty.value,
            "indicators": self._get_difficulty_indicators(problem)
        }
    
    def _recommend_approach(self, inputs: Dict, context) -> Dict[str, Any]:
        """Recommend verification approach."""
        problem = inputs.get("problem", self.config.get("problem", ""))
        
        context.update_progress(30)
        
        # Get classification
        domain_scores = self._score_domains(problem)
        primary_domain = max(domain_scores, key=domain_scores.get)
        difficulty = self._estimate_difficulty_level(problem, primary_domain)
        
        context.update_progress(60)
        
        # Recommend approach based on domain and difficulty
        recommendation = self._get_approach_recommendation(primary_domain, difficulty)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "recommendation": recommendation,
            "reasoning": f"Based on {primary_domain.value} domain and {difficulty.name} difficulty"
        }
    
    def _analyze_complexity(self, inputs: Dict, context) -> Dict[str, Any]:
        """Analyze problem complexity."""
        problem = inputs.get("problem", self.config.get("problem", ""))
        
        context.update_progress(50)
        
        analysis = {
            "statement_length": len(problem),
            "word_count": len(problem.split()),
            "formula_count": len(re.findall(r'[\$\\]', problem)),
            "quantifier_count": len(re.findall(r'\b(forall|exists|∀|∃)\b', problem)),
            "nested_depth": self._estimate_nesting_depth(problem),
            "complexity_score": self._compute_complexity_score(problem)
        }
        
        context.update_progress(100)
        
        return {
            "success": True,
            "complexity": analysis,
            "rating": "high" if analysis["complexity_score"] > 0.7 else "medium" if analysis["complexity_score"] > 0.4 else "low"
        }
    
    def _batch_classify(self, inputs: Dict, context) -> Dict[str, Any]:
        """Classify multiple problems."""
        problems = inputs.get("problems", self.config.get("problems", []))
        
        results = []
        total = len(problems)
        
        for i, problem in enumerate(problems):
            progress = (i + 1) / total * 100
            context.update_progress(progress)
            
            result = self._classify({"problem": problem}, context)
            results.append({
                "problem": problem[:100] + "..." if len(problem) > 100 else problem,
                "classification": result.get("classification", {})
            })
        
        return {
            "success": True,
            "total": total,
            "results": results
        }
    
    def _score_domains(self, problem: str) -> Dict[MathDomain, float]:
        """Score each domain based on keyword matches."""
        scores = {domain: 0.0 for domain in MathDomain}
        text_lower = problem.lower()
        
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            scores[domain] = min(score / max(len(patterns), 1) * 0.5, 1.0)
        
        # If no clear matches, mark as unknown
        if max(scores.values()) == 0:
            scores[MathDomain.UNKNOWN] = 1.0
        
        return scores
    
    def _score_types(self, problem: str) -> Dict[ProblemType, float]:
        """Score each problem type."""
        scores = {}
        text_lower = problem.lower()
        
        for ptype, patterns in self.TYPE_PATTERNS.items():
            score = sum(len(re.findall(p, text_lower)) for p in patterns)
            if score > 0:
                scores[ptype] = min(score * 0.5, 1.0)
        
        return scores
    
    def _estimate_difficulty_level(self, problem: str, domain: MathDomain) -> DifficultyLevel:
        """Estimate difficulty based on problem characteristics."""
        indicators = self._get_difficulty_indicators(problem)
        score = sum(indicators.values())
        
        if score >= 8:
            return DifficultyLevel.RESEARCH
        elif score >= 6:
            return DifficultyLevel.ADVANCED
        elif score >= 3:
            return DifficultyLevel.INTERMEDIATE
        else:
            return DifficultyLevel.ELEMENTARY
    
    def _get_difficulty_indicators(self, problem: str) -> Dict[str, int]:
        """Get difficulty indicator scores."""
        text = problem.lower()
        indicators = {
            "length": len(problem) // 500,  # Longer = harder
            "quantifiers": len(re.findall(r'\b(forall|exists|∀|∃)\b', text)),
            "nested_proofs": len(re.findall(r'\bproof\b', text)),
            "advanced_terms": len(re.findall(r'\b(universal|existential|isomorphism|homomorphism|automorphism)\b', text)),
            "multi_part": len(re.findall(r'\b(part|step|case)\s*\d+\b', text))
        }
        return indicators
    
    def _generate_tags(self, problem: str, domain: MathDomain, ptype: ProblemType) -> List[str]:
        """Generate semantic tags for the problem."""
        tags = [domain.value, ptype.value]
        
        text = problem.lower()
        
        # Add technique tags
        technique_keywords = {
            "induction": "proof-by-induction",
            "contradiction": "proof-by-contradiction",
            "cases": "case-analysis",
            "diagonalization": "diagonalization",
            "construction": "constructive-proof"
        }
        
        for keyword, tag in technique_keywords.items():
            if keyword in text:
                tags.append(tag)
        
        return list(set(tags))
    
    def _get_approach_recommendation(self, domain: MathDomain, difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Recommend verification approach based on classification."""
        # Logic/Arithmetic → Z3 first
        # Complex pure math → Lean first
        # Mixed → Hybrid
        
        if domain in [MathDomain.LOGIC, MathDomain.NUMBER_THEORY] and difficulty.value <= 2:
            return {
                "primary": "z3",
                "secondary": "lean",
                "strategy": "z3_first",
                "reasoning": "Well-suited for automated solving"
            }
        elif domain in [MathDomain.ANALYSIS, MathDomain.TOPOLOGY, MathDomain.ABSTRACT_ALGEBRA]:
            return {
                "primary": "lean",
                "secondary": "z3",
                "strategy": "lean_first",
                "reasoning": "Requires rich mathematical libraries"
            }
        else:
            return {
                "primary": "hybrid",
                "secondary": None,
                "strategy": "adaptive",
                "reasoning": "Depends on specific problem characteristics"
            }
    
    def _estimate_nesting_depth(self, problem: str) -> int:
        """Estimate maximum nesting depth of logical structure."""
        max_depth = 0
        current_depth = 0
        
        for char in problem:
            if char in '([{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ')]}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _compute_complexity_score(self, problem: str) -> float:
        """Compute overall complexity score (0-1)."""
        factors = [
            min(len(problem) / 1000, 1.0) * 0.3,  # Length
            min(self._estimate_nesting_depth(problem) / 5, 1.0) * 0.3,  # Nesting
            min(len(re.findall(r'\b(forall|exists|∀|∃)\b', problem)) / 5, 1.0) * 0.4  # Quantifiers
        ]
        return sum(factors)
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
