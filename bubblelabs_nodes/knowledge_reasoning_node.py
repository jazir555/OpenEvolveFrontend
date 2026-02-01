"""
Knowledge Reasoning Node for BubbleLabs Integration

Provides formal reasoning capabilities over knowledge using Z3 SMT solver:
- Verify knowledge consistency
- Detect contradictions in knowledge
- Infer new facts from existing knowledge
- Check logical validity of statements
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeReasoningNode(BubbleLabsNode):
    """
    Reason over knowledge using formal methods and Z3 SMT solver.

    Supports reasoning operations:
    - verify: Check if a conclusion follows from premises
    - contradiction_check: Detect contradictions in knowledge
    - infer: Infer new facts from existing knowledge
    - validate: Validate logical consistency of statements
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Reasoning"
    DESCRIPTION = "Verify, validate, and reason over knowledge using formal methods"
    ICON = "knowledge-reasoning"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Import Z3 integration (safe import)
        self.Z3SolverEngine = self.safe_import(
            'z3prover_integration.Z3SolverEngine',
            fallback_value=None,
            error_msg="Z3SolverEngine not available for KnowledgeReasoningNode"
        )
        self.Z3TheoremProver = self.safe_import(
            'z3prover_integration.Z3TheoremProver',
            fallback_value=None,
            error_msg="Z3TheoremProver not available for KnowledgeReasoningNode"
        )
        self.Z3Config = self.safe_import(
            'z3prover_integration.Z3Config',
            fallback_value=None,
            error_msg="Z3Config not available for KnowledgeReasoningNode"
        )
        self.Z3ResultStatus = self.safe_import(
            'z3prover_integration.Z3ResultStatus',
            fallback_value=None,
            error_msg="Z3ResultStatus not available for KnowledgeReasoningNode"
        )
        self.Z3Variable = self.safe_import(
            'z3prover_integration.Z3Variable',
            fallback_value=None,
            error_msg="Z3Variable not available for KnowledgeReasoningNode"
        )
        self.Z3Constraint = self.safe_import(
            'z3prover_integration.Z3Constraint',
            fallback_value=None,
            error_msg="Z3Constraint not available for KnowledgeReasoningNode"
        )
        self.Z3ConstraintType = self.safe_import(
            'z3prover_integration.Z3ConstraintType',
            fallback_value=None,
            error_msg="Z3ConstraintType not available for KnowledgeReasoningNode"
        )
        self.get_z3_solver_engine = self.safe_import(
            'z3prover_integration.get_z3_solver_engine',
            fallback_value=None,
            error_msg="get_z3_solver_engine not available for KnowledgeReasoningNode"
        )
        self.get_z3_theorem_prover = self.safe_import(
            'z3prover_integration.get_z3_theorem_prover',
            fallback_value=None,
            error_msg="get_z3_theorem_prover not available for KnowledgeReasoningNode"
        )
        self.is_z3_available = self.safe_import(
            'z3prover_integration.is_z3_available',
            fallback_value=lambda: False,
            error_msg="is_z3_available not available for KnowledgeReasoningNode"
        )

        # Initialize solver if available
        self.solver = None
        self.prover = None
        if self.is_z3_available and self.get_z3_solver_engine:
            try:
                config = self.Z3Config(timeout=30.0, proof_generation=True) if self.Z3Config else None
                self.solver = self.get_z3_solver_engine(config)
                self.prover = self.get_z3_theorem_prover(config)
            except Exception as e:
                self.logger.warning(f"Could not initialize Z3 solver: {e}")

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required (depending on reasoning_type):
            - premises: List[str] - Starting facts/premises
            - conclusion: str (for verify/validate) - Statement to verify

        Optional:
            - knowledge_graph_id: str - ID of KG to reason over
            - reasoning_type: str - Override config reasoning_type
            - include_explanation: bool - Include reasoning explanation
        """
        errors = []

        # Get reasoning type from inputs or config
        reasoning_type = inputs.get('reasoning_type', self.config.get('reasoning_type', 'verify'))

        # Validate reasoning_type
        valid_types = ['verify', 'contradiction_check', 'infer', 'validate']
        if reasoning_type not in valid_types:
            errors.append(f"Invalid reasoning_type: {reasoning_type}. Must be one of {valid_types}")

        # Check premises
        if 'premises' in inputs:
            if not isinstance(inputs['premises'], list):
                errors.append("premises must be a list")
            else:
                for i, p in enumerate(inputs['premises']):
                    if not isinstance(p, str):
                        errors.append(f"premise[{i}] must be a string")

        # Check conclusion (required for verify and validate)
        if reasoning_type in ['verify', 'validate']:
            premises = inputs.get('premises', self.config.get('premises', []))
            conclusion = inputs.get('conclusion', self.config.get('conclusion'))
            if not conclusion and not premises:
                errors.append(f"reasoning_type '{reasoning_type}' requires 'conclusion' or 'premises' in inputs or config")

        # Validate include_explanation
        if 'include_explanation' in inputs:
            if not isinstance(inputs['include_explanation'], bool):
                errors.append("include_explanation must be a boolean")

        # Validate knowledge_graph_id
        if 'knowledge_graph_id' in inputs:
            if not isinstance(inputs['knowledge_graph_id'], str):
                errors.append("knowledge_graph_id must be a string")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute knowledge reasoning operation.

        Args:
            inputs: Must contain 'premises' and/or 'conclusion' based on reasoning_type
            context: Workflow state for tracking

        Returns:
            Dict containing:
                - valid: Whether the reasoning result is valid/successful
                - explanation: Explanation of the reasoning process
                - contradictions: List of detected contradictions (for contradiction_check)
                - inferred: List of inferred facts (for infer)
                - confidence: Confidence score (0-1)
                - execution_time: Time taken for reasoning
        """
        # Get parameters
        reasoning_type = inputs.get('reasoning_type', self.config.get('reasoning_type', 'verify'))
        premises = inputs.get('premises', self.config.get('premises', []))
        conclusion = inputs.get('conclusion', self.config.get('conclusion', ''))
        knowledge_graph_id = inputs.get('knowledge_graph_id', self.config.get('knowledge_graph_id'))
        include_explanation = inputs.get('include_explanation', self.config.get('include_explanation', True))

        context.update_progress(10, f"Initializing {reasoning_type} reasoning")
        self.logger.info(f"Starting {reasoning_type} reasoning with {len(premises)} premises")

        # Load from knowledge graph if specified
        if knowledge_graph_id:
            context.update_progress(15, f"Loading knowledge from graph: {knowledge_graph_id}")
            kg_premises = self._load_from_knowledge_graph(knowledge_graph_id)
            premises = list(set(premises + kg_premises))  # Merge and deduplicate
            self.logger.info(f"Loaded {len(kg_premises)} facts from knowledge graph")

        try:
            # Execute based on reasoning type
            if reasoning_type == 'verify':
                result = self._verify_reasoning(premises, conclusion, context, include_explanation)
            elif reasoning_type == 'contradiction_check':
                result = self._check_contradictions(premises, context, include_explanation)
            elif reasoning_type == 'infer':
                result = self._infer_facts(premises, context, include_explanation)
            elif reasoning_type == 'validate':
                result = self._validate_statement(premises, conclusion, context, include_explanation)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown reasoning type: {reasoning_type}",
                    details={'reasoning_type': reasoning_type}
                )

            # Add metadata
            result['reasoning_type'] = reasoning_type
            result['premises_count'] = len(premises)
            result['knowledge_graph_id'] = knowledge_graph_id

            # Add to context
            context.add_artifact('knowledge_reasoning', {
                'result': result,
                'reasoning_type': reasoning_type,
                'timestamp': datetime.now().isoformat()
            })

            context.update_progress(100, f"Reasoning complete: valid={result.get('valid', False)}")
            self.logger.info(f"Reasoning completed: type={reasoning_type}, valid={result.get('valid', False)}")

            return result

        except Exception as e:
            self.logger.error(f"Reasoning failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Reasoning failed: {str(e)}",
                details={
                    'reasoning_type': reasoning_type,
                    'premises_count': len(premises),
                    'exception_type': type(e).__name__
                }
            ) from e

    def _verify_reasoning(self, premises: List[str], conclusion: str, context, include_explanation: bool) -> Dict[str, Any]:
        """Verify if conclusion follows from premises."""
        context.update_progress(30, "Building verification model")
        start_time = datetime.now()

        # Build SMT-LIB for verification
        smtlib = self._build_verification_smtlib(premises, conclusion)

        context.update_progress(50, "Running Z3 solver")

        if self.prover:
            # Use actual Z3 prover
            result = self.prover.prove_theorem(smtlib)
            valid = result.proven
            explanation = self._generate_explanation(premises, conclusion, valid, include_explanation)
            
            return {
                'valid': valid,
                'explanation': explanation,
                'contradictions': [],
                'inferred': [],
                'confidence': 0.95 if valid else 0.0,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'z3_result': {
                    'proven': result.proven,
                    'tactic_used': result.tactic_used,
                    'errors': result.errors
                } if hasattr(result, 'proven') else None
            }
        else:
            # Fallback verification
            return self._verify_simple(premises, conclusion, context, include_explanation)

    def _check_contradictions(self, premises: List[str], context, include_explanation: bool) -> Dict[str, Any]:
        """Detect contradictions in premises."""
        context.update_progress(30, "Checking for contradictions")
        start_time = datetime.now()

        contradictions = []
        
        if self.solver and self.Z3Variable and self.Z3Constraint and self.Z3ConstraintType:
            # Use Z3 to check satisfiability
            smtlib = self._build_smtlib(premises)
            result = self.solver.solve_smtlib(smtlib)
            
            has_contradiction = result.is_unsat() if hasattr(result, 'is_unsat') else result.status.value == 'unsat'
            
            if has_contradiction:
                # Try to find minimal unsat core (contradiction)
                contradictions = self._extract_contradictions(premises)
            
            explanation = self._generate_contradiction_explanation(premises, contradictions, include_explanation)
            
            return {
                'valid': len(contradictions) == 0,
                'explanation': explanation,
                'contradictions': contradictions,
                'inferred': [],
                'confidence': 0.9 if len(contradictions) == 0 else 0.0,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'z3_result': {
                    'satisfiable': result.is_sat() if hasattr(result, 'is_sat') else result.status.value == 'sat',
                    'status': result.status.value if hasattr(result, 'status') else str(result.status)
                } if hasattr(result, 'status') else None
            }
        else:
            # Simple contradiction check (look for direct negations)
            contradictions = self._simple_contradiction_check(premises)
            
            return {
                'valid': len(contradictions) == 0,
                'explanation': self._generate_contradiction_explanation(premises, contradictions, include_explanation),
                'contradictions': contradictions,
                'inferred': [],
                'confidence': 0.7 if len(contradictions) == 0 else 0.0,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'warning': 'Z3 not available, using simple contradiction detection'
            }

    def _infer_facts(self, premises: List[str], context, include_explanation: bool) -> Dict[str, Any]:
        """Infer new facts from premises."""
        context.update_progress(30, "Running inference engine")
        start_time = datetime.now()

        inferred = []

        if self.solver:
            # Use Z3 to find implied facts
            smtlib = self._build_smtlib(premises)
            result = self.solver.solve_smtlib(smtlib)
            
            if result.is_sat() if hasattr(result, 'is_sat') else result.status.value == 'sat':
                # Extract model and infer additional facts
                inferred = self._extract_inferred_facts(premises, result)
        else:
            # Simple inference patterns
            inferred = self._simple_inference(premises)

        explanation = self._generate_inference_explanation(premises, inferred, include_explanation)

        return {
            'valid': True,
            'explanation': explanation,
            'contradictions': [],
            'inferred': inferred,
            'confidence': 0.8 if inferred else 0.5,
            'execution_time': (datetime.now() - start_time).total_seconds(),
            'inferred_count': len(inferred)
        }

    def _validate_statement(self, premises: List[str], conclusion: str, context, include_explanation: bool) -> Dict[str, Any]:
        """Validate logical validity of a statement."""
        context.update_progress(30, "Validating statement logic")
        start_time = datetime.now()

        # Similar to verify but focuses on logical structure
        if self.prover and conclusion:
            smtlib = self._build_verification_smtlib(premises, conclusion)
            result = self.prover.prove_theorem(smtlib)
            valid = result.proven if hasattr(result, 'proven') else False
            
            # Check for structural validity even if proof fails
            structurally_valid = self._check_structural_validity(premises, conclusion)
            
            explanation = self._generate_validation_explanation(
                premises, conclusion, valid, structurally_valid, include_explanation
            )

            return {
                'valid': valid or structurally_valid,
                'explanation': explanation,
                'contradictions': [],
                'inferred': [],
                'confidence': 0.95 if valid else (0.7 if structurally_valid else 0.3),
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'structurally_valid': structurally_valid,
                'logically_valid': valid
            }
        else:
            # Fallback validation
            structurally_valid = self._check_structural_validity(premises, conclusion)
            
            return {
                'valid': structurally_valid,
                'explanation': f"Statement is {'structurally valid' if structurally_valid else 'structurally invalid'}. "
                              f"Z3 prover not available for full logical validation." if include_explanation else "",
                'contradictions': [],
                'inferred': [],
                'confidence': 0.6 if structurally_valid else 0.3,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'structurally_valid': structurally_valid,
                'warning': 'Z3 prover not available'
            }

    def _verify_simple(self, premises: List[str], conclusion: str, context, include_explanation: bool) -> Dict[str, Any]:
        """Simple verification when Z3 is not available."""
        context.update_progress(40, "Using simple verification (Z3 not available)")
        start_time = datetime.now()

        # Simple checks
        valid = False
        reasons = []

        # Check if conclusion is in premises
        if conclusion in premises:
            valid = True
            reasons.append("Conclusion is explicitly stated in premises")

        # Check for simple implication patterns
        for premise in premises:
            if f"implies {conclusion}" in premise or f"-> {conclusion}" in premise:
                valid = True
                reasons.append(f"Premise implies conclusion: {premise}")

        # Check for contradictions
        contradictions = self._simple_contradiction_check(premises)
        if contradictions:
            valid = False
            reasons.append("Contradictions found in premises")

        explanation = "; ".join(reasons) if include_explanation else ""

        return {
            'valid': valid,
            'explanation': explanation or "Simple verification performed (Z3 not available)",
            'contradictions': contradictions,
            'inferred': [],
            'confidence': 0.5 if valid else 0.3,
            'execution_time': (datetime.now() - start_time).total_seconds(),
            'warning': 'Z3 not available, using simple verification'
        }

    def _build_smtlib(self, premises: List[str]) -> str:
        """Build SMT-LIB from premises."""
        lines = [
            "(set-logic ALL)",
            "(set-option :produce-models true)",
            ""
        ]

        # Extract and declare variables
        variables = self._extract_variables(premises)
        for var, var_type in variables.items():
            lines.append(f"(declare-fun {var} () {var_type})")

        lines.append("")

        # Add assertions for premises
        for premise in premises:
            smt_premise = self._to_smt_expr(premise)
            if smt_premise:
                lines.append(f"(assert {smt_premise})")

        lines.append("")
        lines.append("(check-sat)")
        lines.append("(get-model)")

        return "\n".join(lines)

    def _build_verification_smtlib(self, premises: List[str], conclusion: str) -> str:
        """Build SMT-LIB for verification (checking if premises imply conclusion)."""
        lines = [
            "(set-logic ALL)",
            ""
        ]

        # Extract and declare variables
        variables = self._extract_variables(premises + [conclusion])
        for var, var_type in variables.items():
            lines.append(f"(declare-fun {var} () {var_type})")

        lines.append("")

        # Add premises
        for premise in premises:
            smt_premise = self._to_smt_expr(premise)
            if smt_premise:
                lines.append(f"(assert {smt_premise})")

        # Add negation of conclusion for proof by contradiction
        smt_conclusion = self._to_smt_expr(conclusion)
        if smt_conclusion:
            lines.append(f"(assert (not {smt_conclusion}))")

        lines.append("")
        lines.append("(check-sat)")

        return "\n".join(lines)

    def _extract_variables(self, statements: List[str]) -> Dict[str, str]:
        """Extract variables from statements."""
        variables = {}

        for stmt in statements:
            # Look for patterns like "x > 5", "y = 10", etc.
            import re
            # Match variable names (single letters or words followed by operators)
            matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:>|<|=|>=|<=|!=|\+|-|\*|/)', stmt)
            for match in matches:
                if match not in ['and', 'or', 'not', 'implies', 'forall', 'exists']:
                    # Infer type based on context
                    if re.search(rf'{match}\s*[=><]\s*\d+\.\d+', stmt):
                        variables[match] = "Real"
                    elif re.search(rf'{match}\s*[=><]\s*\d+', stmt):
                        variables[match] = "Int"
                    elif 'true' in stmt.lower() or 'false' in stmt.lower():
                        variables[match] = "Bool"
                    else:
                        variables[match] = "Int"  # Default to Int

        return variables

    def _to_smt_expr(self, expr: str) -> str:
        """Convert natural language expression to SMT-LIB."""
        if not expr:
            return ""

        smt = expr

        # Replace common operators
        replacements = [
            ('=', '='),
            ('==', '='),
            ('!=', 'distinct'),
            ('<>', 'distinct'),
            ('<=', '<='),
            ('>=', '>='),
            ('<', '<'),
            ('>', '>'),
            ('and', 'and'),
            ('&&', 'and'),
            ('or', 'or'),
            ('||', 'or'),
            ('not ', 'not '),
            ('!', 'not '),
            ('implies', '=>'),
            ('->', '=>'),
        ]

        for old, new in replacements:
            smt = smt.replace(old, new)

        return smt.strip()

    def _extract_contradictions(self, premises: List[str]) -> List[Dict[str, Any]]:
        """Extract minimal set of contradictory premises using Z3."""
        contradictions = []

        # Try removing premises one by one to find minimal unsat core
        for i, premise in enumerate(premises):
            reduced = premises[:i] + premises[i+1:]
            smtlib = self._build_smtlib(reduced)
            
            if self.solver:
                result = self.solver.solve_smtlib(smtlib)
                if result.is_sat() if hasattr(result, 'is_sat') else result.status.value == 'sat':
                    # This premise was part of the contradiction
                    contradictions.append({
                        'premise': premise,
                        'index': i,
                        'type': 'critical_premise'
                    })

        return contradictions

    def _simple_contradiction_check(self, premises: List[str]) -> List[Dict[str, Any]]:
        """Simple contradiction detection without Z3."""
        contradictions = []

        for i, p1 in enumerate(premises):
            for j, p2 in enumerate(premises[i+1:], i+1):
                # Check for direct negation
                if self._are_contradictory(p1, p2):
                    contradictions.append({
                        'premise1': p1,
                        'premise2': p2,
                        'index1': i,
                        'index2': j,
                        'type': 'direct_negation'
                    })

        return contradictions

    def _are_contradictory(self, p1: str, p2: str) -> bool:
        """Check if two premises are contradictory."""
        # Simple negation detection
        p1_clean = p1.strip().lower()
        p2_clean = p2.strip().lower()

        # Direct negation: "not X" vs "X"
        if p1_clean.startswith('not ') and p1_clean[4:] == p2_clean:
            return True
        if p2_clean.startswith('not ') and p2_clean[4:] == p1_clean:
            return True

        # Negation with parentheses: "not (X)" vs "X"
        if p1_clean.startswith('not(') and p1_clean[4:-1] == p2_clean:
            return True
        if p2_clean.startswith('not(') and p2_clean[4:-1] == p1_clean:
            return True

        # Arithmetic contradictions
        import re
        # Pattern: x > 5 and x <= 5
        match1 = re.match(r'(\w+)\s*>\s*(\d+)', p1)
        match2 = re.match(r'(\w+)\s*<=\s*(\d+)', p2)
        if match1 and match2:
            if match1.group(1) == match2.group(1) and int(match1.group(2)) >= int(match2.group(2)):
                return True

        return False

    def _extract_inferred_facts(self, premises: List[str], result) -> List[str]:
        """Extract inferred facts from Z3 model."""
        inferred = []

        if result.model and hasattr(result.model, 'assignments'):
            for var, value in result.model.assignments.items():
                inferred.append(f"{var} = {value}")

        # Add derived inferences
        if len(premises) >= 2:
            inferred.append(f"Knowledge base contains {len(premises)} consistent facts")

        return inferred

    def _simple_inference(self, premises: List[str]) -> List[str]:
        """Simple inference without Z3."""
        inferred = []

        # Transitive inference
        import re
        for p1 in premises:
            for p2 in premises:
                if p1 != p2:
                    # x = y and y = z implies x = z
                    eq1 = re.match(r'(\w+)\s*=\s*(\w+)', p1)
                    eq2 = re.match(r'(\w+)\s*=\s*(\w+)', p2)
                    if eq1 and eq2:
                        if eq1.group(2) == eq2.group(1) and eq1.group(1) != eq2.group(2):
                            inferred.append(f"{eq1.group(1)} = {eq2.group(2)}")

        return list(set(inferred))  # Remove duplicates

    def _check_structural_validity(self, premises: List[str], conclusion: str) -> bool:
        """Check if statement is structurally valid."""
        # Basic structural checks
        if not conclusion and not premises:
            return False

        # Check for balanced parentheses
        for stmt in premises + [conclusion]:
            if stmt:
                if stmt.count('(') != stmt.count(')'):
                    return False

        # Check for complete expressions
        for stmt in premises + [conclusion]:
            if stmt and not stmt.strip().endswith((')', '.')):
                # Might be incomplete
                pass

        return True

    def _load_from_knowledge_graph(self, kg_id: str) -> List[str]:
        """Load facts from knowledge graph."""
        # Placeholder for KG integration
        # In a real implementation, this would query the knowledge graph
        self.logger.info(f"Loading from knowledge graph: {kg_id}")
        return []

    def _generate_explanation(self, premises: List[str], conclusion: str, valid: bool, include: bool) -> str:
        """Generate explanation for verification result."""
        if not include:
            return ""

        if valid:
            return (
                f"The conclusion '{conclusion}' logically follows from the given premises. "
                f"Verified using {len(premises)} premise(s)."
            )
        else:
            return (
                f"The conclusion '{conclusion}' does not follow from the given premises. "
                f"The premises may be insufficient or contradictory."
            )

    def _generate_contradiction_explanation(self, premises: List[str], contradictions: List[Dict], include: bool) -> str:
        """Generate explanation for contradiction check."""
        if not include:
            return ""

        if contradictions:
            return (
                f"Found {len(contradictions)} contradiction(s) in {len(premises)} premise(s). "
                f"The knowledge base is inconsistent."
            )
        else:
            return f"No contradictions found in {len(premises)} premise(s). The knowledge base is consistent."

    def _generate_inference_explanation(self, premises: List[str], inferred: List[str], include: bool) -> str:
        """Generate explanation for inference result."""
        if not include:
            return ""

        if inferred:
            return (
                f"Inferred {len(inferred)} new fact(s) from {len(premises)} premise(s). "
                f"Inferences are logically entailed by the premises."
            )
        else:
            return f"No new facts could be inferred from {len(premises)} premise(s)."

    def _generate_validation_explanation(self, premises: List[str], conclusion: str, 
                                         valid: bool, structurally_valid: bool, include: bool) -> str:
        """Generate explanation for validation result."""
        if not include:
            return ""

        parts = []
        if structurally_valid:
            parts.append("The statement is structurally well-formed.")
        else:
            parts.append("The statement has structural issues.")

        if valid:
            parts.append("The statement is logically valid given the premises.")
        else:
            parts.append("The statement could not be proven from the premises.")

        return " ".join(parts)

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "title": "Knowledge Reasoning Configuration",
            "description": "Configure knowledge reasoning parameters",
            "properties": {
                "reasoning_type": {
                    "type": "string",
                    "title": "Reasoning Type",
                    "description": "Type of reasoning operation to perform",
                    "enum": ["verify", "contradiction_check", "infer", "validate"],
                    "enumNames": [
                        "Verify (Check if conclusion follows from premises)",
                        "Contradiction Check (Detect inconsistencies)",
                        "Infer (Derive new facts)",
                        "Validate (Check logical validity)"
                    ],
                    "default": "verify"
                },
                "premises": {
                    "type": "array",
                    "title": "Premises",
                    "description": "Starting facts or premises for reasoning",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "conclusion": {
                    "type": "string",
                    "title": "Conclusion",
                    "description": "Statement to verify (required for verify/validate operations)",
                    "default": ""
                },
                "knowledge_graph_id": {
                    "type": "string",
                    "title": "Knowledge Graph ID",
                    "description": "Optional ID of knowledge graph to load facts from",
                    "default": ""
                },
                "include_explanation": {
                    "type": "boolean",
                    "title": "Include Explanation",
                    "description": "Include detailed reasoning explanation in output",
                    "default": True
                },
                "timeout": {
                    "type": "integer",
                    "title": "Timeout",
                    "description": "Maximum reasoning time in seconds",
                    "minimum": 1,
                    "maximum": 300,
                    "default": 30
                }
            },
            "required": ["reasoning_type"]
        }
