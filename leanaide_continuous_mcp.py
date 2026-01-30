"""
MCP Tools for LeanAide Continuous Mathematics System

This module provides Model Context Protocol (MCP) tool wrappers for all
LeanAide continuous mathematics components, enabling seamless integration
with MCP-compatible applications and workflows.

Components wrapped:
- B.1: Continuous Math Detector
- B.2: ODE/PDE Translator
- B.3: Scientific Domain Patterns
- B.4: Verification Methods

Features:
- Unified MCP interface for all components
- Tool definitions with input/output schemas
- Error handling and validation
- Comprehensive metadata
- Easy integration with MCP servers

Author: OpenEvolve
Created: 2026-01-09
Phase: 2 - LeanAide Enhancement (Task B.5)
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

# Import all components
from continuous_math_detector import (
    ContinuousMathDetector,
    detect_continuous_math,
    MathType,
    ProblemType,
    ScientificDomain,
    MathDetectionResult,
)
from ode_pde_translator import (
    ODEPDETranslator,
    translate_to_lean4,
    Lean4TranslationResult,
    SolutionType,
)
from scientific_domain_patterns import (
    ScientificDomainPatterns,
    get_domain_patterns,
    get_equation_template,
)
from verification_methods import (
    Lean4Verifier,
    verify_lean4_code,
    verify_translation,
    VerificationResult,
    VerificationStatus,
    CheckType,
)

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# MCP Tool Definitions
# ============================================================================

class MCPToolType(Enum):
    """Types of MCP tools"""
    DETECTOR = "detector"
    TRANSLATOR = "translator"
    DOMAIN_PATTERNS = "domain_patterns"
    VERIFIER = "verifier"


@dataclass
class MCPToolInput:
    """Input schema for MCP tools"""
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


@dataclass
class MCPToolOutput:
    """Output schema for MCP tools"""
    type: str
    description: str


@dataclass
class MCPToolDefinition:
    """Complete MCP tool definition"""
    name: str
    description: str
    category: str
    input_schema: List[MCPToolInput]
    output_schema: MCPToolOutput
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MCP serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "input_schema": [
                {
                    "type": inp.type,
                    "description": inp.description,
                    "required": inp.required,
                    "default": inp.default,
                    "enum": inp.enum
                }
                for inp in self.input_schema
            ],
            "output_schema": {
                "type": self.output_schema.type,
                "description": self.output_schema.description
            },
            "examples": self.examples,
            "metadata": self.metadata
        }


@dataclass
class MCPToolResult:
    """Result from executing an MCP tool"""
    tool_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# MCP Tools Manager
# ============================================================================

class LeanAideContinuousMCP:
    """
    Manager for LeanAide Continuous Mathematics MCP tools.

    Provides unified access to all LeanAide continuous mathematics
    functionality through MCP-compatible tool wrappers.
    """

    def __init__(self):
        """Initialize all MCP tools"""
        # Initialize components
        self.detector = ContinuousMathDetector()
        self.translator = ODEPDETranslator()
        self.domain_patterns = get_domain_patterns()
        self.verifier = Lean4Verifier(enable_leanaide=False)

        # Register tools
        self.tools: Dict[str, callable] = {}
        self.tool_definitions: Dict[str, MCPToolDefinition] = {}

        self._register_tools()

        logger.info(f"LeanAide Continuous MCP initialized with {len(self.tools)} tools")

    def _register_tools(self):
        """Register all MCP tools"""

        # B.1: Detection Tools
        self._register_tool(
            name="detect_math",
            description="Detect and classify continuous mathematics in text",
            category="detection",
            handler=self._detect_math_handler,
            input_schema=[
                MCPToolInput(
                    type="string",
                    description="Text containing mathematics to detect",
                    required=True
                ),
                MCPToolInput(
                    type="boolean",
                    description="Return detailed analysis including variables and equations",
                    required=False,
                    default=False
                )
            ],
            output_schema=MCPToolOutput(
                type="object",
                description="Math detection result with type, domain, confidence, and extracted information"
            ),
            examples=[
                {
                    "input": {"text": "Solve dy/dx + y = 0"},
                    "output": {"math_type": "ODE", "domain": "GENERAL", "confidence": 0.85}
                },
                {
                    "input": {"text": "Heat equation: ∂u/∂t = α∂²u/∂x²"},
                    "output": {"math_type": "PDE", "domain": "PHYSICS", "confidence": 0.92}
                }
            ]
        )

        self._register_tool(
            name="is_ode",
            description="Check if text contains an ordinary differential equation",
            category="detection",
            handler=self._is_ode_handler,
            input_schema=[
                MCPToolInput(
                    type="string",
                    description="Text to check for ODE",
                    required=True
                )
            ],
            output_schema=MCPToolOutput(
                type="boolean",
                description="True if text contains an ODE, False otherwise"
            ),
            examples=[
                {"input": {"text": "dy/dx = x + y"}, "output": True},
                {"input": {"text": "integral of x"}, "output": False}
            ]
        )

        self._register_tool(
            name="is_pde",
            description="Check if text contains a partial differential equation",
            category="detection",
            handler=self._is_pde_handler,
            input_schema=[
                MCPToolInput(
                    type="string",
                    description="Text to check for PDE",
                    required=True
                )
            ],
            output_schema=MCPToolOutput(
                type="boolean",
                description="True if text contains a PDE, False otherwise"
            )
        )

        # B.2: Translation Tools
        self._register_tool(
            name="translate_to_lean4",
            description="Translate detected mathematics to Lean 4 formal code",
            category="translation",
            handler=self._translate_to_lean4_handler,
            input_schema=[
                MCPToolInput(
                    type="string",
                    description="Text containing mathematics to translate",
                    required=True
                ),
                MCPToolInput(
                    type="string",
                    description="Type of solution theorem (existence, uniqueness, existence_uniqueness)",
                    required=False,
                    default="existence_uniqueness",
                    enum=["existence", "uniqueness", "existence_uniqueness"]
                ),
                MCPToolInput(
                    type="boolean",
                    description="Generate proof scaffolding with tactics",
                    required=False,
                    default=True
                )
            ],
            output_schema=MCPToolOutput(
                type="object",
                description="Lean 4 translation result with code, definitions, theorems, and proof scaffolds"
            ),
            examples=[
                {
                    "input": {"text": "dy/dx + y = 0"},
                    "output": {"lean4_code": "...", "success": True}
                }
            ]
        )

        self._register_tool(
            name="translate_ode",
            description="Translate a standalone ODE to Lean 4",
            category="translation",
            handler=self._translate_ode_handler,
            input_schema=[
                MCPToolInput(
                    type="string",
                    description="ODE equation",
                    required=True
                ),
                MCPToolInput(
                    type="string",
                    description="Initial condition (e.g., 'y(0) = 1')",
                    required=False
                )
            ],
            output_schema=MCPToolOutput(
                type="object",
                description="Lean 4 code for the ODE"
            )
        )

        self._register_tool(
            name="translate_pde",
            description="Translate a standalone PDE to Lean 4",
            category="translation",
            handler=self._translate_pde_handler,
            input_schema=[
                MCPToolInput(
                    type="string",
                    description="PDE equation",
                    required=True
                ),
                MCPToolInput(
                    type="array",
                    description="Boundary conditions",
                    required=False
                )
            ],
            output_schema=MCPToolOutput(
                type="object",
                description="Lean 4 code for the PDE"
            )
        )

        # B.3: Domain Knowledge Tools
        self._register_tool(
            name="get_equation_templates",
            description="Get equation templates for a scientific domain",
            category="domain_knowledge",
            handler=self._get_equation_templates_handler,
            input_schema=[
                MCPToolInput(
                    type="string",
                    description="Scientific domain (physics, chemistry, biology, engineering, economics)",
                    required=True,
                    enum=["physics", "chemistry", "biology", "engineering", "economics"]
                ),
                MCPToolInput(
                    type="string",
                    description="Optional category filter",
                    required=False
                )
            ],
            output_schema=MCPToolOutput(
                type="array",
                description="List of equation templates with Lean 4 code"
            )
        )

        self._register_tool(
            name="get_solution_methods",
            description="Get typical solution methods for a domain",
            category="domain_knowledge",
            handler=self._get_solution_methods_handler,
            input_schema=[
                MCPToolInput(
                    type="string",
                    description="Scientific domain",
                    required=True,
                    enum=["physics", "chemistry", "biology", "engineering", "economics"]
                )
            ],
            output_schema=MCPToolOutput(
                type="array",
                description="List of solution method names"
            )
        )

        self._register_tool(
            name="recommend_solution_method",
            description="Recommend solution methods based on problem characteristics",
            category="domain_knowledge",
            handler=self._recommend_solution_method_handler,
            input_schema=[
                MCPToolInput(
                    type="string",
                    description="Scientific domain",
                    required=True,
                    enum=["physics", "chemistry", "biology", "engineering", "economics"]
                ),
                MCPToolInput(
                    type="string",
                    description="Math type (ODE, PDE, DAE, SDE)",
                    required=True,
                    enum=["ODE", "PDE", "DAE", "SDE"]
                ),
                MCPToolInput(
                    type="string",
                    description="Problem type (IVP, BVP, eigenvalue, control, optimization)",
                    required=True,
                    enum=["IVP", "BVP", "eigenvalue", "control", "optimization"]
                )
            ],
            output_schema=MCPToolOutput(
                type="array",
                description="List of recommended solution methods"
            )
        )

        # B.4: Verification Tools
        self._register_tool(
            name="verify_lean4_code",
            description="Verify Lean 4 code for syntax, types, and mathematical correctness",
            category="verification",
            handler=self._verify_lean4_code_handler,
            input_schema=[
                MCPToolInput(
                    type="string",
                    description="Lean 4 code to verify",
                    required=True
                ),
                MCPToolInput(
                    type="string",
                    description="Scientific domain for domain-specific checks",
                    required=False,
                    enum=["physics", "chemistry", "biology", "engineering", "economics"]
                )
            ],
            output_schema=MCPToolOutput(
                type="object",
                description="Verification result with status, issues, and suggestions"
            ),
            examples=[
                {
                    "input": {"code": "def test (x : Real) : Prop := x > 0"},
                    "output": {"status": "passed", "issues": []}
                }
            ]
        )

        # Combined workflow tools
        self._register_tool(
            name="complete_pipeline",
            description="Run complete pipeline: detect → translate → verify",
            category="workflow",
            handler=self._complete_pipeline_handler,
            input_schema=[
                MCPToolInput(
                    type="string",
                    description="Text containing mathematics problem",
                    required=True
                ),
                MCPToolInput(
                    type="boolean",
                    description="Verify the generated code",
                    required=False,
                    default=True
                )
            ],
            output_schema=MCPToolOutput(
                type="object",
                description="Complete pipeline result with detection, translation, and verification"
            ),
            examples=[
                {
                    "input": {"text": "Solve dy/dx + y = 0 with y(0) = 1"},
                    "output": {
                        "detection": {"math_type": "ODE"},
                        "translation": {"success": True, "lean4_code": "..."},
                        "verification": {"status": "passed"}
                    }
                }
            ]
        )

    def _register_tool(
        self,
        name: str,
        description: str,
        category: str,
        handler: callable,
        input_schema: List[MCPToolInput],
        output_schema: MCPToolOutput,
        examples: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register an MCP tool"""
        self.tools[name] = handler
        self.tool_definitions[name] = MCPToolDefinition(
            name=name,
            description=description,
            category=category,
            input_schema=input_schema,
            output_schema=output_schema,
            examples=examples or [],
            metadata=metadata or {}
        )

    # ========================================================================
    # Tool Handlers
    # ========================================================================

    def _detect_math_handler(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Handle detect_math tool"""
        try:
            text = arguments.get("text")
            detailed = arguments.get("detailed", False)

            result = self.detector.detect(text)

            data = {
                "math_type": result.math_type.value,
                "problem_type": result.problem_type.value,
                "domain": result.domain.value,
                "confidence": result.confidence,
                "equations": result.equations,
                "variables": result.variables,
                "notation": result.notation,
                "keywords": result.keywords
            }

            if detailed:
                data["metadata"] = result.metadata
                data["matched_patterns"] = getattr(result, 'matched_patterns', [])

            return MCPToolResult(
                tool_name="detect_math",
                success=True,
                data=data
            )

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in detect_math: {e}")
            return MCPToolResult(
                tool_name="detect_math",
                success=False,
                error=str(e)
            )

    def _is_ode_handler(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Handle is_ode tool"""
        try:
            text = arguments.get("text")
            result = self.detector.detect_ode(text)

            return MCPToolResult(
                tool_name="is_ode",
                success=True,
                data={"is_ode": result.math_type == MathType.ODE}
            )

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in is_ode: {e}")
            return MCPToolResult(
                tool_name="is_ode",
                success=False,
                error=str(e)
            )

    def _is_pde_handler(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Handle is_pde tool"""
        try:
            text = arguments.get("text")
            result = self.detector.detect_pde(text)

            return MCPToolResult(
                tool_name="is_pde",
                success=True,
                data={"is_pde": result.math_type == MathType.PDE}
            )

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in is_pde: {e}")
            return MCPToolResult(
                tool_name="is_pde",
                success=False,
                error=str(e)
            )

    def _translate_to_lean4_handler(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Handle translate_to_lean4 tool"""
        try:
            text = arguments.get("text")
            solution_type_str = arguments.get("solution_type", "existence_uniqueness")
            generate_proof_scaffold = arguments.get("generate_proof_scaffold", True)

            # Detect first
            detection_result = self.detector.detect(text)

            # Convert solution type
            solution_type_map = {
                "existence": SolutionType.EXISTENCE,
                "uniqueness": SolutionType.UNIQUENESS,
                "existence_uniqueness": SolutionType.EXISTENCE_UNIQUENESS
            }
            solution_type = solution_type_map.get(solution_type_str, SolutionType.EXISTENCE_UNIQUENESS)

            # Translate
            translation_result = self.translator.translate(
                detection_result,
                solution_type=solution_type,
                generate_proof_scaffold=generate_proof_scaffold
            )

            data = {
                "success": translation_result.success,
                "lean4_code": translation_result.lean4_code,
                "num_definitions": len(translation_result.definitions),
                "num_theorems": len(translation_result.theorems),
                "num_proof_scaffolds": len(translation_result.proof_scaffolds),
                "imports": translation_result.imports
            }

            if not translation_result.success:
                data["error"] = translation_result.error_message

            return MCPToolResult(
                tool_name="translate_to_lean4",
                success=translation_result.success,
                data=data
            )

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in translate_to_lean4: {e}")
            return MCPToolResult(
                tool_name="translate_to_lean4",
                success=False,
                error=str(e)
            )

    def _translate_ode_handler(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Handle translate_ode tool"""
        try:
            equation = arguments.get("equation")
            initial_condition = arguments.get("initial_condition")

            result = self.translator.translate_ode(equation, initial_condition)

            return MCPToolResult(
                tool_name="translate_ode",
                success=result.success,
                data={
                    "lean4_code": result.lean4_code,
                    "success": result.success
                }
            )

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in translate_ode: {e}")
            return MCPToolResult(
                tool_name="translate_ode",
                success=False,
                error=str(e)
            )

    def _translate_pde_handler(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Handle translate_pde tool"""
        try:
            equation = arguments.get("equation")
            boundary_conditions = arguments.get("boundary_conditions")

            result = self.translator.translate_pde(equation, boundary_conditions)

            return MCPToolResult(
                tool_name="translate_pde",
                success=result.success,
                data={
                    "lean4_code": result.lean4_code,
                    "success": result.success
                }
            )

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in translate_pde: {e}")
            return MCPToolResult(
                tool_name="translate_pde",
                success=False,
                error=str(e)
            )

    def _get_equation_templates_handler(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Handle get_equation_templates tool"""
        try:
            domain_str = arguments.get("domain")
            category = arguments.get("category")

            domain_map = {
                "physics": ScientificDomain.PHYSICS,
                "chemistry": ScientificDomain.CHEMISTRY,
                "biology": ScientificDomain.BIOLOGY,
                "engineering": ScientificDomain.ENGINEERING,
                "economics": ScientificDomain.ECONOMICS
            }
            domain = domain_map.get(domain_str)

            templates = self.domain_patterns.get_equation_templates(domain, category)

            data = [
                {
                    "name": t.name,
                    "category": t.category,
                    "equation": t.equation_pattern,
                    "description": t.description,
                    "parameters": t.parameters,
                    "solution_method": t.solution_method,
                    "has_lean4_template": t.lean4_template is not None
                }
                for t in templates
            ]

            return MCPToolResult(
                tool_name="get_equation_templates",
                success=True,
                data={"templates": data, "count": len(data)}
            )

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in get_equation_templates: {e}")
            return MCPToolResult(
                tool_name="get_equation_templates",
                success=False,
                error=str(e)
            )

    def _get_solution_methods_handler(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Handle get_solution_methods tool"""
        try:
            domain_str = arguments.get("domain")

            domain_map = {
                "physics": ScientificDomain.PHYSICS,
                "chemistry": ScientificDomain.CHEMISTRY,
                "biology": ScientificDomain.BIOLOGY,
                "engineering": ScientificDomain.ENGINEERING,
                "economics": ScientificDomain.ECONOMICS
            }
            domain = domain_map.get(domain_str)

            methods = self.domain_patterns.get_solution_methods(domain)

            return MCPToolResult(
                tool_name="get_solution_methods",
                success=True,
                data={"solution_methods": methods, "count": len(methods)}
            )

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in get_solution_methods: {e}")
            return MCPToolResult(
                tool_name="get_solution_methods",
                success=False,
                error=str(e)
            )

    def _recommend_solution_method_handler(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Handle recommend_solution_method tool"""
        try:
            domain_str = arguments.get("domain")
            math_type_str = arguments.get("math_type")
            problem_type_str = arguments.get("problem_type")

            # Map strings to enums
            domain_map = {
                "physics": ScientificDomain.PHYSICS,
                "chemistry": ScientificDomain.CHEMISTRY,
                "biology": ScientificDomain.BIOLOGY,
                "engineering": ScientificDomain.ENGINEERING,
                "economics": ScientificDomain.ECONOMICS
            }
            math_type_map = {
                "ODE": MathType.ODE,
                "PDE": MathType.PDE,
                "DAE": MathType.DAE,
                "SDE": MathType.SDE
            }
            problem_type_map = {
                "IVP": ProblemType.INITIAL_VALUE,
                "BVP": ProblemType.BOUNDARY_VALUE,
                "eigenvalue": ProblemType.EIGENVALUE,
                "control": ProblemType.CONTROL,
                "optimization": ProblemType.OPTIMIZATION
            }

            domain = domain_map.get(domain_str)
            math_type = math_type_map.get(math_type_str)
            problem_type = problem_type_map.get(problem_type_str)

            methods = self.domain_patterns.recommend_solution_method(domain, math_type, problem_type)

            return MCPToolResult(
                tool_name="recommend_solution_method",
                success=True,
                data={"recommended_methods": methods}
            )

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in recommend_solution_method: {e}")
            return MCPToolResult(
                tool_name="recommend_solution_method",
                success=False,
                error=str(e)
            )

    def _verify_lean4_code_handler(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Handle verify_lean4_code tool"""
        try:
            code = arguments.get("code")
            domain_str = arguments.get("domain")

            domain = None
            if domain_str:
                domain_map = {
                    "physics": ScientificDomain.PHYSICS,
                    "chemistry": ScientificDomain.CHEMISTRY,
                    "biology": ScientificDomain.BIOLOGY,
                    "engineering": ScientificDomain.ENGINEERING,
                    "economics": ScientificDomain.ECONOMICS
                }
                domain = domain_map.get(domain_str)

            result = self.verifier.verify_code(code, domain)

            data = {
                "status": result.overall_status.value,
                "is_valid": result.is_valid,
                "checks_performed": [c.value for c in result.checks_performed],
                "passed_checks": result.passed_checks,
                "failed_checks": result.failed_checks,
                "warnings": result.warnings,
                "verification_time": result.verification_time
            }

            if result.issues:
                data["issues"] = [issue.to_dict() for issue in result.issues]

            return MCPToolResult(
                tool_name="verify_lean4_code",
                success=True,
                data=data
            )

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in verify_lean4_code: {e}")
            return MCPToolResult(
                tool_name="verify_lean4_code",
                success=False,
                error=str(e)
            )

    def _complete_pipeline_handler(self, arguments: Dict[str, Any]) -> MCPToolResult:
        """Handle complete_pipeline tool"""
        start_time = time.time()

        try:
            text = arguments.get("text")
            verify = arguments.get("verify", True)

            # Step 1: Detect
            detection_result = self.detector.detect(text)

            detection_data = {
                "math_type": detection_result.math_type.value,
                "problem_type": detection_result.problem_type.value,
                "domain": detection_result.domain.value,
                "confidence": detection_result.confidence
            }

            # Step 2: Translate
            translation_result = self.translator.translate(detection_result)

            translation_data = {
                "success": translation_result.success,
                "num_definitions": len(translation_result.definitions),
                "num_theorems": len(translation_result.theorems)
            }

            if not translation_result.success:
                return MCPToolResult(
                    tool_name="complete_pipeline",
                    success=False,
                    data={
                        "detection": detection_data,
                        "translation": translation_data,
                        "error": translation_result.error_message
                    },
                    execution_time=time.time() - start_time
                )

            # Step 3: Verify (optional)
            verification_data = None
            if verify:
                verification_result = self.verifier.verify(translation_result, detection_result)
                verification_data = {
                    "status": verification_result.overall_status.value,
                    "is_valid": verification_result.is_valid,
                    "num_issues": len(verification_result.issues)
                }

            data = {
                "detection": detection_data,
                "translation": translation_data,
                "verification": verification_data,
                "success": translation_result.success and (not verify or verification_data["is_valid"])
            }

            return MCPToolResult(
                tool_name="complete_pipeline",
                success=True,
                data=data,
                execution_time=time.time() - start_time
            )

        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.error(f"Error in complete_pipeline: {e}")
            return MCPToolResult(
                tool_name="complete_pipeline",
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    # ========================================================================
    # Public API
    # ========================================================================

    def list_tools(self) -> List[str]:
        """List all available MCP tools"""
        return list(self.tools.keys())

    def get_tool_definition(self, tool_name: str) -> Optional[MCPToolDefinition]:
        """Get definition of a specific tool"""
        return self.tool_definitions.get(tool_name)

    def list_tool_definitions(self) -> List[MCPToolDefinition]:
        """List all tool definitions"""
        return list(self.tool_definitions.values())

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """
        Execute an MCP tool by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            MCPToolResult with execution result
        """
        if tool_name not in self.tools:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}"
            )

        handler = self.tools[tool_name]
        return handler(arguments)

    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tools in a category"""
        return [
            name for name, definition in self.tool_definitions.items()
            if definition.category == category
        ]

    def export_manifest(self) -> Dict[str, Any]:
        """
        Export MCP manifest for tool registration.

        Returns:
            Manifest dictionary with all tool definitions
        """
        return {
            "name": "leanaide_continuous_math",
            "version": "1.0.0",
            "description": "MCP tools for LeanAide continuous mathematics system",
            "categories": list(set(d.category for d in self.tool_definitions.values())),
            "tools": [d.to_dict() for d in self.tool_definitions.values()]
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def get_mcp_tools() -> LeanAideContinuousMCP:
    """Get the global MCP tools instance"""
    return LeanAideContinuousMCP()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example usage
    tools = LeanAideContinuousMCP()

    print("=" * 80)
    print("LeanAide Continuous MCP Tools - Example Usage")
    print("=" * 80)

    # List all tools
    print(f"\nAvailable Tools ({len(tools.list_tools())}):")
    for tool_name in tools.list_tools():
        definition = tools.get_tool_definition(tool_name)
        print(f"  - {tool_name}: {definition.description}")

    # Example: Complete pipeline
    print("\n" + "=" * 80)
    print("Example: Complete Pipeline")
    print("=" * 80)

    result = tools.execute_tool(
        "complete_pipeline",
        {"text": "Solve dy/dx + y = 0 with y(0) = 1", "verify": False}
    )

    print(f"\nSuccess: {result.success}")
    if result.success:
        print(f"Detection: {result.data['detection']['math_type']}")
        print(f"Domain: {result.data['detection']['domain']}")
        print(f"Confidence: {result.data['detection']['confidence']}")
        print(f"Definitions: {result.data['translation']['num_definitions']}")
        print(f"Theorems: {result.data['translation']['num_theorems']}")
    else:
        print(f"Error: {result.error}")

    # Export manifest
    print("\n" + "=" * 80)
    print("MCP Manifest Export")
    print("=" * 80)

    manifest = tools.export_manifest()
    print(f"\nManifest: {len(manifest['tools'])} tools")
    print(f"Categories: {', '.join(manifest['categories'])}")
