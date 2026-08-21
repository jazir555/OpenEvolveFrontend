"""
Conflict Detector Module for Sovereign AI System

This module provides comprehensive conflict detection capabilities for analyzing
sub-solutions generated during problem decomposition and recomposition.

Features:
- Naming conflict detection (duplicate names, shadowing)
- Logic conflict detection (contradictory logic, incompatible approaches)
- Dependency conflict detection (version mismatches, API incompatibilities)
- Severity assessment (CRITICAL, HIGH, MEDIUM, LOW)
- Automatic resolution proposals
- AST-based code analysis
- Pattern matching for conflict detection

Author: OpenEvolve AI System
Version: 1.0.0
License: MIT
"""
from __future__ import annotations


import ast
import re
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import difflib
from datetime import datetime

# **ACTUAL INTEGRATION**: Alerting and knowledge for Conflict Detector
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts that can be detected"""
    NAMING_CONFLICT = "naming_conflict"
    LOGIC_CONFLICT = "logic_conflict"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    STRUCTURAL_CONFLICT = "structural_conflict"
    API_CONFLICT = "api_conflict"
    RESOURCE_CONFLICT = "resource_conflict"
    VERSION_CONFLICT = "version_conflict"
    DATA_CONFLICT = "data_conflict"


class ConflictSeverity(Enum):
    """Severity levels for conflicts"""
    CRITICAL = "CRITICAL"  # Will cause system failure
    HIGH = "HIGH"  # Likely to cause issues
    MEDIUM = "MEDIUM"  # May cause issues in certain scenarios
    LOW = "LOW"  # Minor issues, won't affect functionality


@dataclass
class Conflict:
    """
    Represents a detected conflict between sub-solutions

    Attributes:
        conflict_type: Type of conflict detected
        severity: Severity level of the conflict
        description: Human-readable description of the conflict
        affected_solutions: List of solution IDs/names affected by the conflict
        source_locations: Locations in code where conflict occurs
        suggested_resolution: Proposed resolution for the conflict
        metadata: Additional context about the conflict
        confidence: Confidence score (0.0 to 1.0) in conflict detection
    """
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    affected_solutions: List[str]
    source_locations: List[Dict[str, Any]]
    suggested_resolution: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert conflict to dictionary representation"""
        return {
            'conflict_type': self.conflict_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'affected_solutions': self.affected_solutions,
            'source_locations': self.source_locations,
            'suggested_resolution': self.suggested_resolution,
            'metadata': self.metadata,
            'confidence': self.confidence
        }


@dataclass
class SolutionAnalysis:
    """Results from analyzing a single solution"""
    solution_id: str
    names_defined: Set[str]
    names_used: Set[str]
    imports: List[Dict[str, str]]
    function_calls: List[Dict[str, str]]
    classes_defined: Set[str]
    variables_defined: Dict[str, str]  # name -> type
    logic_patterns: List[str]
    dependencies: List[Dict[str, str]]


class ASTVisitor(ast.NodeVisitor):
    """Custom AST visitor for analyzing code structure"""

    def __init__(self, solution_id: str):
        self.solution_id = solution_id
        self.names_defined: Set[str] = set()
        self.names_used: Set[str] = set()
        self.imports: List[Dict[str, str]] = []
        self.function_calls: List[Dict[str, str]] = []
        self.classes_defined: Set[str] = set()
        self.variables_defined: Dict[str, str] = {}
        self.logic_patterns: List[str] = []
        self.dependencies: List[Dict[str, str]] = []
        self.current_function: Optional[str] = None
        self.current_class: Optional[str] = None

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements"""
        for alias in node.names:
            self.imports.append({
                'module': alias.name,
                'alias': alias.asname if alias.asname else alias.name,
                'line': node.lineno
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statements"""
        module = node.module if node.module else ''
        for alias in node.names:
            self.imports.append({
                'module': module,
                'name': alias.name,
                'alias': alias.asname if alias.asname else alias.name,
                'line': node.lineno
            })
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions"""
        func_name = node.name

        # Add to names defined
        if self.current_class:
            full_name = f"{self.current_class}.{func_name}"
        else:
            full_name = func_name

        self.names_defined.add(full_name)
        self.variables_defined[full_name] = 'function_sync'  # Mark as sync function

        # Track logic patterns
        self._analyze_function_logic(node)

        # Visit children
        old_function = self.current_function
        self.current_function = full_name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions"""
        func_name = node.name

        # Add to names defined
        if self.current_class:
            full_name = f"{self.current_class}.{func_name}"
        else:
            full_name = func_name

        self.names_defined.add(full_name)
        self.variables_defined[full_name] = 'function_async'  # Mark as async function

        # Track that this is async code
        self.logic_patterns.append('async_pattern')

        # Track logic patterns
        self._analyze_function_logic(node)

        # Visit children
        old_function = self.current_function
        self.current_function = full_name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions"""
        class_name = node.name
        self.classes_defined.add(class_name)
        self.names_defined.add(class_name)
        self.variables_defined[class_name] = 'class'

        # Track inheritance
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.dependencies.append({
                    'type': 'inheritance',
                    'target': base.id,
                    'line': node.lineno
                })

        old_class = self.current_class
        self.current_class = class_name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Name(self, node: ast.Name) -> None:
        """Visit name references"""
        if isinstance(node.ctx, ast.Load):
            self.names_used.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.names_defined.add(node.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls"""
        if isinstance(node.func, ast.Name):
            self.function_calls.append({
                'name': node.func.id,
                'line': node.lineno,
                'context': self.current_function or 'module'
            })
        elif isinstance(node.func, ast.Attribute):
            self.function_calls.append({
                'name': node.func.attr,
                'line': node.lineno,
                'context': self.current_function or 'module'
            })
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit variable assignments"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.names_defined.add(target.id)
                # Try to infer type
                if isinstance(node.value, ast.Call):
                    self.variables_defined[target.id] = 'object'
                elif isinstance(node.value, ast.Constant):
                    self.variables_defined[target.id] = type(node.value.value).__name__
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        """Visit assert statements for logic patterns"""
        self.logic_patterns.append('assertion')
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """Visit if statements for logic patterns"""
        self.logic_patterns.append('conditional')
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Visit for loops for logic patterns"""
        self.logic_patterns.append('iteration')
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        """Visit while loops for logic patterns"""
        self.logic_patterns.append('iteration')
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        """Visit try-except blocks for logic patterns"""
        self.logic_patterns.append('exception_handling')
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Visit with statements for logic patterns"""
        self.logic_patterns.append('context_manager')
        self.generic_visit(node)

    def _analyze_function_logic(self, node: ast.FunctionDef) -> None:
        """Analyze logic patterns in a function"""
        # Check for conflicting patterns
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(node))

        if has_return and has_yield:
            self.logic_patterns.append('mixed_return_yield')

        # Check for function complexity
        num_conditions = sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.While)))
        if num_conditions > 5:
            self.logic_patterns.append('complex_control_flow')


class ConflictDetector:
    """
    Main conflict detection engine

    Detects and analyzes conflicts between sub-solutions using AST analysis,
    pattern matching, and semantic analysis.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the conflict detector

        Args:
            strict_mode: If True, treat all potential conflicts as actual conflicts
        """
        self.strict_mode = strict_mode
        self.analyses: Dict[str, SolutionAnalysis] = {}

        # Patterns for detecting logic conflicts
        self.conflicting_patterns = {
            ('assert_true', 'assert_false'): 'contradictory_assertions',
            ('verify_positive', 'verify_negative'): 'opposite_verification',
            ('enable_feature', 'disable_feature'): 'feature_toggle_conflict',
            ('allow_access', 'deny_access'): 'access_control_conflict',
            ('create_resource', 'delete_resource'): 'resource_lifecycle_conflict',
        }

        # API incompatibility patterns
        self.incompatible_apis = {
            'threading': {'asyncio', 'multiprocessing'},
            'sync': {'async'},
        }

    def detect_conflicts(
        self,
        sub_solutions: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
        resource_a: Optional[Dict[str, Any]] = None,
        resource_b: Optional[Dict[str, Any]] = None
    ) -> List[Conflict]:
        """
        Detect all conflicts between sub-solutions or resources

        Args:
            sub_solutions: List of solution code strings (optional)
            metadata: Optional metadata for each solution
            resource_a: First resource dict for simple conflict detection (optional)
            resource_b: Second resource dict for simple conflict detection (optional)

        Returns:
            List of detected conflicts
        """
        # Support simple two-resource conflict detection
        if resource_a is not None and resource_b is not None:
            conflicts = []

            # Check for version conflicts
            if 'version' in resource_a and 'version' in resource_b:
                if resource_a['version'] != resource_b['version']:
                    conflicts.append(Conflict(
                        conflict_type=ConflictType.VERSION_CONFLICT,
                        severity=ConflictSeverity.HIGH,
                        description=f"Version mismatch: {resource_a['version']} vs {resource_b['version']}",
                        affected_solutions=['resource_a', 'resource_b'],
                        source_locations=[],
                        suggested_resolution={'strategy': 'resolve_version'},
                        confidence=1.0
                    ))

            # Check for data conflicts
            if 'data' in resource_a and 'data' in resource_b:
                if resource_a['data'] != resource_b['data']:
                    conflicts.append(Conflict(
                        conflict_type=ConflictType.DATA_CONFLICT,
                        severity=ConflictSeverity.MEDIUM,
                        description=f"Data mismatch between resources",
                        affected_solutions=['resource_a', 'resource_b'],
                        source_locations=[],
                        suggested_resolution={'strategy': 'merge_data'},
                        confidence=0.8
                    ))

            return conflicts

        # Standard conflict detection for sub_solutions
        if sub_solutions is None:
            return []

        logger.info(f"Starting conflict detection for {len(sub_solutions)} solutions")

        conflicts: List[Conflict] = []

        # Analyze all solutions first
        self.analyses = {}
        for idx, solution in enumerate(sub_solutions):
            solution_id = metadata[idx]['id'] if metadata and idx < len(metadata) else f"solution_{idx}"
            try:
                analysis = self._analyze_solution(solution, solution_id)
                self.analyses[solution_id] = analysis
                logger.debug(f"Analyzed solution {solution_id}: {len(analysis.names_defined)} names defined")
            except (SyntaxError, ValueError, TypeError) as e:
                logger.error(f"Failed to analyze solution {solution_id}: {e}")
                # Create minimal analysis for failed solutions
                self.analyses[solution_id] = SolutionAnalysis(
                    solution_id=solution_id,
                    names_defined=set(),
                    names_used=set(),
                    imports=[],
                    function_calls=[],
                    classes_defined=set(),
                    variables_defined={},
                    logic_patterns=[],
                    dependencies=[]
                )

        # Detect different types of conflicts
        logger.info("Detecting naming conflicts...")
        conflicts.extend(self.analyze_naming_conflicts(sub_solutions))

        logger.info("Detecting logic conflicts...")
        conflicts.extend(self.analyze_logic_conflicts(sub_solutions))

        logger.info("Detecting dependency conflicts...")
        conflicts.extend(self.analyze_dependency_conflicts(sub_solutions))

        logger.info(f"Conflict detection complete. Found {len(conflicts)} conflicts")

        # **ACTUAL INTEGRATION**: Extract knowledge, track performance, and trigger alerts
        num_critical = sum(1 for c in conflicts if c.severity == ConflictSeverity.CRITICAL)
        self._extract_conflict_knowledge("detect_conflicts", conflicts, len(sub_solutions))
        self._track_conflict_performance("detect_conflicts", True, len(conflicts), num_critical)

        # Trigger alert for critical conflicts
        if num_critical > 0:
            self._trigger_conflict_alerts(
                "detect_conflicts",
                True,
                len(conflicts),
                num_critical,
                None,
                {"critical_count": num_critical}
            )

        return conflicts

    def detect_ast_edit_conflicts(
        self,
        agent_edits: List[Dict[str, Any]]
    ) -> List[Conflict]:
        """
        Detect conflicts where multiple agents edit the same AST node with different intents.

        Args:
            agent_edits: List of edits containing at least agent_id, node_id, intent

        Returns:
            List of conflicts requiring mediation
        """
        conflicts: List[Conflict] = []
        edits_by_node: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edit in agent_edits:
            node_id = edit.get("node_id")
            if node_id:
                edits_by_node[node_id].append(edit)

        for node_id, edits in edits_by_node.items():
            intents = {e.get("intent", "") for e in edits}
            if len(intents) <= 1:
                continue

            affected_agents = [e.get("agent_id", "unknown") for e in edits]
            conflicts.append(
                Conflict(
                    conflict_type=ConflictType.STRUCTURAL_CONFLICT,
                    severity=ConflictSeverity.HIGH,
                    description=f"Agents have conflicting intents for AST node {node_id}.",
                    affected_solutions=affected_agents,
                    source_locations=[{"node_id": node_id}],
                    suggested_resolution={
                        "strategy": "nash_mediation",
                        "mediator_required": True,
                        "node_id": node_id,
                        "intents": list(intents)
                    },
                    metadata={"node_id": node_id, "edits": edits},
                    confidence=0.85
                )
            )

        return conflicts

    def _analyze_solution(self, solution_code: str, solution_id: str) -> SolutionAnalysis:
        """
        Analyze a single solution using AST

        Args:
            solution_code: Python code string
            solution_id: Identifier for the solution

        Returns:
            SolutionAnalysis object with analysis results
        """
        try:
            tree = ast.parse(solution_code)
        except SyntaxError as e:
            logger.warning(f"Syntax error in solution {solution_id}: {e}")
            # Return minimal analysis
            return SolutionAnalysis(
                solution_id=solution_id,
                names_defined=set(),
                names_used=set(),
                imports=[],
                function_calls=[],
                classes_defined=set(),
                variables_defined={},
                logic_patterns=[],
                dependencies=[]
            )

        visitor = ASTVisitor(solution_id)
        visitor.visit(tree)

        # Extract dependencies from imports
        for imp in visitor.imports:
            visitor.dependencies.append({
                'type': 'import',
                'module': imp['module'],
                'name': imp.get('name', ''),
                'line': imp['line']
            })

        return SolutionAnalysis(
            solution_id=solution_id,
            names_defined=visitor.names_defined,
            names_used=visitor.names_used,
            imports=visitor.imports,
            function_calls=visitor.function_calls,
            classes_defined=visitor.classes_defined,
            variables_defined=visitor.variables_defined,
            logic_patterns=visitor.logic_patterns,
            dependencies=visitor.dependencies
        )

    def analyze_naming_conflicts(self, solutions: List[str]) -> List[Conflict]:
        """
        Detect naming conflicts between solutions

        Types of naming conflicts:
        - Duplicate names: Same name defined in multiple solutions
        - Shadowing: Inner scope shadows outer scope
        - Inconsistent naming: Same concept, different names

        Args:
            solutions: List of solution code strings

        Returns:
            List of naming conflict objects
        """
        conflicts: List[Conflict] = []

        # Check for duplicate definitions
        name_to_solutions: Dict[str, List[str]] = defaultdict(list)
        name_to_types: Dict[str, Set[str]] = defaultdict(set)
        name_to_lines: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for solution_id, analysis in self.analyses.items():
            for name in analysis.names_defined:
                name_to_solutions[name].append(solution_id)
                name_to_types[name].add(analysis.variables_defined.get(name, 'unknown'))
                name_to_lines[name].append({
                    'solution': solution_id,
                    'type': analysis.variables_defined.get(name, 'unknown')
                })

        # Find names defined in multiple solutions
        for name, solution_list in name_to_solutions.items():
            if len(solution_list) > 1:
                # Check if types are consistent
                types = name_to_types[name]

                if len(types) > 1:
                    # Type mismatch - more severe
                    severity = ConflictSeverity.CRITICAL if 'class' in types else ConflictSeverity.HIGH
                    description = (
                        f"Name '{name}' is defined with different types across solutions: "
                        f"{', '.join(types)}. This will cause type conflicts."
                    )
                    resolution = {
                        'strategy': 'rename',
                        'suggested_names': [
                            f"{sol_id}_{name}" for sol_id in solution_list
                        ],
                        'explanation': 'Rename each instance to be solution-specific or move to shared module'
                    }
                else:
                    # Same type, different solutions - potential overwrite
                    severity = ConflictSeverity.HIGH
                    description = (
                        f"Name '{name}' is defined in multiple solutions: "
                        f"{', '.join(solution_list)}. Implementations may conflict."
                    )
                    resolution = {
                        'strategy': 'consolidate_or_prefix',
                        'options': [
                            'Consolidate into shared utility module',
                            'Prefix with solution identifier',
                            'Use namespace/package structure'
                        ]
                    }

                conflict = Conflict(
                    conflict_type=ConflictType.NAMING_CONFLICT,
                    severity=severity,
                    description=description,
                    affected_solutions=solution_list,
                    source_locations=name_to_lines[name],
                    suggested_resolution=resolution,
                    metadata={
                        'name': name,
                        'types': list(types),
                        'definition_count': len(solution_list)
                    },
                    confidence=0.95
                )
                conflicts.append(conflict)

        # Check for shadowing within solutions
        for solution_id, analysis in self.analyses.items():
            shadowing_conflicts = self._detect_shadowing(analysis)
            conflicts.extend(shadowing_conflicts)

        # Check for inconsistent naming patterns
        conflicts.extend(self._detect_inconsistent_naming())

        return conflicts

    def _detect_shadowing(self, analysis: SolutionAnalysis) -> List[Conflict]:
        """Detect variable shadowing within a solution"""
        conflicts: List[Conflict] = []

        # Check if builtin names are shadowed
        builtins = {'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool', 'object'}

        for name in analysis.names_defined:
            if name in builtins:
                conflict = Conflict(
                    conflict_type=ConflictType.NAMING_CONFLICT,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"Builtin name '{name}' is shadowed in solution {analysis.solution_id}",
                    affected_solutions=[analysis.solution_id],
                    source_locations=[{
                        'solution': analysis.solution_id,
                        'type': 'builtin_shadowing'
                    }],
                    suggested_resolution={
                        'strategy': 'rename',
                        'suggestion': f"Use a different name (e.g., {name}_data, {name}_items)"
                    },
                    metadata={'shadowed_builtin': name},
                    confidence=0.9
                )
                conflicts.append(conflict)

        return conflicts

    def _detect_inconsistent_naming(self) -> List[Conflict]:
        """Detect inconsistent naming for similar concepts across solutions"""
        conflicts: List[Conflict] = []

        # Group similar names using string similarity
        all_names = set()
        for analysis in self.analyses.values():
            all_names.update(analysis.names_defined)

        name_list = sorted(all_names)
        similar_groups: Dict[str, List[str]] = defaultdict(list)

        for i, name1 in enumerate(name_list):
            for name2 in name_list[i+1:]:
                similarity = difflib.SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
                if similarity > 0.7 and similarity < 1.0:  # Similar but not identical
                    group_key = min(name1.lower(), name2.lower())
                    similar_groups[group_key].extend([name1, name2])

        # Report potential naming inconsistencies
        for base_name, similar_names in similar_groups.items():
            similar_names = list(set(similar_names))
            if len(similar_names) > 2:
                solutions_affected = []
                for name in similar_names:
                    for sol_id, analysis in self.analyses.items():
                        if name in analysis.names_defined:
                            solutions_affected.append(sol_id)
                            break

                conflict = Conflict(
                    conflict_type=ConflictType.NAMING_CONFLICT,
                    severity=ConflictSeverity.LOW,
                    description=(
                        f"Potential naming inconsistency detected. "
                        f"Similar names found: {', '.join(similar_names)}. "
                        f"Consider standardizing naming convention."
                    ),
                    affected_solutions=list(set(solutions_affected)),
                    source_locations=[],
                    suggested_resolution={
                        'strategy': 'standardize',
                        'suggestion': f"Choose consistent name (e.g., {base_name})"
                    },
                    metadata={'similar_names': similar_names},
                    confidence=0.6
                )
                conflicts.append(conflict)

        return conflicts

    def analyze_logic_conflicts(self, solutions: List[str]) -> List[Conflict]:
        """
        Detect logic conflicts between solutions

        Types of logic conflicts:
        - Contradictory assertions/conditions
        - Opposite verification patterns
        - Incompatible control flow
        - Conflicting state management

        Args:
            solutions: List of solution code strings

        Returns:
            List of logic conflict objects
        """
        conflicts: List[Conflict] = []

        # Check for contradictory logic patterns
        for sol1_id, analysis1 in self.analyses.items():
            for sol2_id, analysis2 in self.analyses.items():
                if sol1_id >= sol2_id:
                    continue

                # Check for conflicting patterns
                conflicts.extend(
                    self._check_pattern_conflicts(analysis1, analysis2)
                )

        # Check for state management conflicts
        conflicts.extend(self._check_state_conflicts())

        # Check for control flow incompatibilities
        conflicts.extend(self._check_control_flow_conflicts())

        return conflicts

    def _check_pattern_conflicts(
        self,
        analysis1: SolutionAnalysis,
        analysis2: SolutionAnalysis
    ) -> List[Conflict]:
        """Check for conflicting logic patterns between two solutions"""
        conflicts: List[Conflict] = []

        patterns1 = set(analysis1.logic_patterns)
        patterns2 = set(analysis2.logic_patterns)

        # Check for direct contradictions based on pattern matching
        # Also check function names for contradictory patterns
        func_names1 = {name.lower() for name in analysis1.names_defined}
        func_names2 = {name.lower() for name in analysis2.names_defined}

        # Define contradictory function name patterns
        contradictory_pairs = [
            (('enable', 'activate'), ('disable', 'deactivate')),
            (('allow', 'permit'), ('deny', 'forbid', 'block')),
            (('create', 'add', 'insert'), ('delete', 'remove', 'drop')),
            (('open', 'start', 'begin'), ('close', 'stop', 'end')),
            ('positive', 'negative'),
            ('true', 'false'),
        ]

        for pair_group in contradictory_pairs:
            if isinstance(pair_group[0], tuple):
                group1, group2 = pair_group
                matches1 = any(any(p in fname for p in group1) for fname in func_names1)
                matches2 = any(any(p in fname for p in group2) for fname in func_names2)

                # Check if both groups appear across solutions
                if matches1 and matches2:
                    conflict = Conflict(
                        conflict_type=ConflictType.LOGIC_CONFLICT,
                        severity=ConflictSeverity.CRITICAL,
                        description=(
                            f"Contradictory logic patterns detected between "
                            f"{analysis1.solution_id} and {analysis2.solution_id}: "
                            f"functions suggest opposing behavior"
                        ),
                        affected_solutions=[analysis1.solution_id, analysis2.solution_id],
                        source_locations=[
                            {'solution': analysis1.solution_id, 'pattern': 'contradictory_function'},
                            {'solution': analysis2.solution_id, 'pattern': 'contradictory_function'}
                        ],
                        suggested_resolution={
                            'strategy': 'arbitrate',
                            'explanation': 'Solutions implement opposite behavior. '
                                         'One must be removed or made conditional.',
                            'options': [
                                f'Use {analysis1.solution_id} behavior',
                                f'Use {analysis2.solution_id} behavior',
                                'Add conditional logic to switch between behaviors'
                            ]
                        },
                        metadata={
                            'conflict_type': 'contradictory_functions',
                        },
                        confidence=0.75
                    )
                    conflicts.append(conflict)
            else:
                # Single pattern pair
                if pair_group[0] in str(func_names1) and pair_group[1] in str(func_names2):
                    conflict = Conflict(
                        conflict_type=ConflictType.LOGIC_CONFLICT,
                        severity=ConflictSeverity.CRITICAL,
                        description=(
                            f"Contradictory logic patterns detected between "
                            f"{analysis1.solution_id} and {analysis2.solution_id}: "
                            f"'{pair_group[0]}' vs '{pair_group[1]}'"
                        ),
                        affected_solutions=[analysis1.solution_id, analysis2.solution_id],
                        source_locations=[
                            {'solution': analysis1.solution_id, 'pattern': pair_group[0]},
                            {'solution': analysis2.solution_id, 'pattern': pair_group[1]}
                        ],
                        suggested_resolution={
                            'strategy': 'arbitrate',
                            'explanation': f'Solutions implement opposite behavior. '
                                         f'One must be removed or made conditional.',
                            'options': [
                                f'Use {analysis1.solution_id} behavior',
                                f'Use {analysis2.solution_id} behavior',
                                'Add conditional logic to switch between behaviors'
                            ]
                        },
                        metadata={
                            'conflict_type': 'contradictory_patterns',
                            'pattern1': pair_group[0],
                            'pattern2': pair_group[1]
                        },
                        confidence=0.75
                    )
                    conflicts.append(conflict)

        # Check for direct contradictions in logic_patterns
        for (pattern1, pattern2), conflict_type in self.conflicting_patterns.items():
            if pattern1 in patterns1 and pattern2 in patterns2:
                conflict = Conflict(
                    conflict_type=ConflictType.LOGIC_CONFLICT,
                    severity=ConflictSeverity.CRITICAL,
                    description=(
                        f"Contradictory logic patterns detected between "
                        f"{analysis1.solution_id} and {analysis2.solution_id}: "
                        f"'{pattern1}' vs '{pattern2}' ({conflict_type})"
                    ),
                    affected_solutions=[analysis1.solution_id, analysis2.solution_id],
                    source_locations=[
                        {'solution': analysis1.solution_id, 'pattern': pattern1},
                        {'solution': analysis2.solution_id, 'pattern': pattern2}
                    ],
                    suggested_resolution={
                        'strategy': 'arbitrate',
                        'explanation': f'Solutions implement opposite behavior for {conflict_type}. '
                                     f'One must be removed or made conditional.',
                        'options': [
                            f'Use {analysis1.solution_id} behavior',
                            f'Use {analysis2.solution_id} behavior',
                            'Add conditional logic to switch between behaviors'
                        ]
                    },
                    metadata={
                        'conflict_type': conflict_type,
                        'pattern1': pattern1,
                        'pattern2': pattern2
                    },
                    confidence=0.85
                )
                conflicts.append(conflict)

        # Check for mixed sync/async patterns
        has_async1 = any('async' in name.lower() or name.startswith('async_') for name in analysis1.names_defined)
        has_async2 = any('async' in name.lower() or name.startswith('async_') for name in analysis2.names_defined)

        # Also check for actual async function definitions in the variables_defined
        def has_async_functions(analysis: SolutionAnalysis) -> bool:
            for var_type in analysis.variables_defined.values():
                if 'async' in str(var_type).lower():
                    return True
            # Also check logic_patterns
            return 'async_pattern' in analysis.logic_patterns

        if (has_async_functions(analysis1) != has_async_functions(analysis2)) or (has_async1 != has_async2):
            conflict = Conflict(
                conflict_type=ConflictType.LOGIC_CONFLICT,
                severity=ConflictSeverity.HIGH,
                description=(
                    f"Inconsistent async/sync patterns between "
                    f"{analysis1.solution_id} and {analysis2.solution_id}. "
                    f"Mixing sync and async code can cause deadlocks or performance issues."
                ),
                affected_solutions=[analysis1.solution_id, analysis2.solution_id],
                source_locations=[
                    {'solution': analysis1.solution_id, 'pattern': 'async' if has_async1 else 'sync'},
                    {'solution': analysis2.solution_id, 'pattern': 'async' if has_async2 else 'sync'}
                ],
                suggested_resolution={
                    'strategy': 'standardize',
                    'explanation': 'Standardize on either async or sync throughout',
                    'options': [
                        'Convert all to async',
                        'Convert all to sync',
                        'Separate into async and sync modules with clear interfaces'
                    ]
                },
                metadata={'async_mismatch': True},
                confidence=0.8
            )
            conflicts.append(conflict)

        return conflicts

    def _check_state_conflicts(self) -> List[Conflict]:
        """Check for state management conflicts"""
        conflicts: List[Conflict] = []

        # Look for shared state modifications
        state_operations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for sol_id, analysis in self.analyses.items():
            for call in analysis.function_calls:
                func_name = call['name'].lower()

                # Identify state-modifying operations
                if any(op in func_name for op in ['set', 'update', 'modify', 'change', 'delete', 'add']):
                    state_operations[func_name].append({
                        'solution': sol_id,
                        'line': call['line'],
                        'context': call['context']
                    })

        # Check if same state is modified by multiple solutions
        for operation, locations in state_operations.items():
            if len(locations) > 1:
                solutions_involved = list(set(loc['solution'] for loc in locations))

                if len(solutions_involved) > 1:
                    conflict = Conflict(
                        conflict_type=ConflictType.LOGIC_CONFLICT,
                        severity=ConflictSeverity.HIGH,
                        description=(
                            f"Concurrent state modification detected. "
                            f"Multiple solutions modify state using '{operation}': "
                            f"{', '.join(solutions_involved)}. "
                            f"This may cause race conditions."
                        ),
                        affected_solutions=solutions_involved,
                        source_locations=locations,
                        suggested_resolution={
                            'strategy': 'synchronize',
                            'explanation': 'Implement proper synchronization or state management',
                            'options': [
                                'Add locks/mutexes',
                                'Use queue-based processing',
                                'Implement transaction semantics',
                                'Use immutable data structures'
                            ]
                        },
                        metadata={'operation': operation},
                        confidence=0.75
                    )
                    conflicts.append(conflict)

        return conflicts

    def _check_control_flow_conflicts(self) -> List[Conflict]:
        """Check for incompatible control flow patterns"""
        conflicts: List[Conflict] = []

        for sol_id, analysis in self.analyses.items():
            # Check for complex control flow
            complexity = analysis.logic_patterns.count('complex_control_flow')

            if complexity > 0:
                conflict = Conflict(
                    conflict_type=ConflictType.LOGIC_CONFLICT,
                    severity=ConflictSeverity.MEDIUM,
                    description=(
                        f"Solution {sol_id} has highly complex control flow "
                        f"({complexity} complex functions). "
                        f"This may conflict with simpler solutions and make integration difficult."
                    ),
                    affected_solutions=[sol_id],
                    source_locations=[{'solution': sol_id, 'complexity': complexity}],
                    suggested_resolution={
                        'strategy': 'refactor',
                        'explanation': 'Refactor complex functions into simpler, testable units',
                        'options': [
                            'Extract methods',
                            'Use strategy pattern',
                            'Implement state machine',
                            'Break into smaller functions'
                        ]
                    },
                    metadata={'complexity_score': complexity},
                    confidence=0.7
                )
                conflicts.append(conflict)

        return conflicts

    def analyze_dependency_conflicts(self, solutions: List[str]) -> List[Conflict]:
        """
        Detect dependency conflicts between solutions

        Types of dependency conflicts:
        - Version mismatches
        - API incompatibilities
        - Circular dependencies
        - Missing dependencies

        Args:
            solutions: List of solution code strings

        Returns:
            List of dependency conflict objects
        """
        conflicts: List[Conflict] = []

        # Check for API incompatibilities
        conflicts.extend(self._check_api_incompatibilities())

        # Check for circular dependencies
        conflicts.extend(self._check_circular_dependencies())

        # Check for duplicate imports with different purposes
        conflicts.extend(self._check_import_conflicts())

        # Check for conflicting module usage
        conflicts.extend(self._check_module_conflicts())

        return conflicts

    def _check_api_incompatibilities(self) -> List[Conflict]:
        """Check for incompatible API usage between solutions"""
        conflicts: List[Conflict] = []

        # Group solutions by API category
        api_usage: Dict[str, List[str]] = defaultdict(list)

        for sol_id, analysis in self.analyses.items():
            for dep in analysis.dependencies:
                if dep['type'] == 'import':
                    module = dep['module'].split('.')[0]
                    api_usage[module].append(sol_id)

        # Check for incompatible combinations
        for api1, solutions1 in api_usage.items():
            for api2, solutions2 in api_usage.items():
                if api1 >= api2:
                    continue

                # Check if these APIs are known to be incompatible
                if self._are_incompatible_apis(api1, api2):
                    affected = list(set(solutions1 + solutions2))

                    conflict = Conflict(
                        conflict_type=ConflictType.DEPENDENCY_CONFLICT,
                        severity=ConflictSeverity.HIGH,
                        description=(
                            f"Incompatible APIs detected: '{api1}' and '{api2}' "
                            f"are used together in solutions {', '.join(affected)}. "
                            f"These APIs may have conflicting requirements or behaviors."
                        ),
                        affected_solutions=affected,
                        source_locations=[
                            {'solution': s, 'api': api1} for s in solutions1
                        ] + [
                            {'solution': s, 'api': api2} for s in solutions2
                        ],
                        suggested_resolution={
                            'strategy': 'separate_or_adapter',
                            'explanation': f'{api1} and {api2} are fundamentally incompatible',
                            'options': [
                                f'Use {api1} only, remove {api2}',
                                f'Use {api2} only, remove {api1}',
                                'Create adapter layer to mediate between APIs',
                                'Run in separate processes/processes'
                            ]
                        },
                        metadata={'api1': api1, 'api2': api2},
                        confidence=0.85
                    )
                    conflicts.append(conflict)

        return conflicts

    def _are_incompatible_apis(self, api1: str, api2: str) -> bool:
        """Check if two APIs are known to be incompatible"""
        api1_lower = api1.lower()
        api2_lower = api2.lower()

        for incompat_group in self.incompatible_apis.values():
            if api1_lower in incompat_group and api2_lower in incompat_group:
                return True

        # Known incompatibilities
        known_incompatibilities = [
            ('threading', 'asyncio'),
            ('multiprocessing', 'asyncio'),
            ('tkinter', 'asyncio'),
            ('sdl2', 'asyncio'),
        ]

        for a, b in known_incompatibilities:
            if a in api1_lower and b in api2_lower:
                return True
            if b in api1_lower and a in api2_lower:
                return True

        return False

    def _check_circular_dependencies(self) -> List[Conflict]:
        """Check for circular dependencies between solutions"""
        conflicts: List[Conflict] = []

        # Build dependency graph
        graph: Dict[str, Set[str]] = {sol_id: set() for sol_id in self.analyses.keys()}

        for sol_id, analysis in self.analyses.items():
            # Check if this solution references other solutions
            for name in analysis.names_used:
                for other_sol_id, other_analysis in self.analyses.items():
                    if sol_id == other_sol_id:
                        continue

                    if name in other_analysis.names_defined:
                        graph[sol_id].add(other_sol_id)

        # Detect cycles using DFS
        visited = set()
        rec_stack = set()
        cycle_path: List[str] = []

        def has_cycle(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    result = has_cycle(neighbor, path.copy())
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            rec_stack.remove(node)
            return None

        for sol_id in graph:
            if sol_id not in visited:
                cycle = has_cycle(sol_id, [])
                if cycle:
                    conflict = Conflict(
                        conflict_type=ConflictType.DEPENDENCY_CONFLICT,
                        severity=ConflictSeverity.CRITICAL,
                        description=(
                            f"Circular dependency detected: {' -> '.join(cycle)}. "
                            f"This will cause import errors and prevent initialization."
                        ),
                        affected_solutions=list(set(cycle)),
                        source_locations=[{'solution': s} for s in cycle],
                        suggested_resolution={
                            'strategy': 'break_cycle',
                            'explanation': 'Circular dependencies must be broken',
                            'options': [
                                'Extract common code to shared module',
                                'Use dependency injection',
                                'Introduce interface/protocol',
                                'Deferred imports (import inside functions)'
                            ]
                        },
                        metadata={'cycle': cycle},
                        confidence=1.0
                    )
                    conflicts.append(conflict)

        return conflicts

    def _check_import_conflicts(self) -> List[Conflict]:
        """Check for conflicting import patterns"""
        conflicts: List[Conflict] = []

        # Group imports by module
        module_imports: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for sol_id, analysis in self.analyses.items():
            for imp in analysis.imports:
                module = imp['module']
                module_imports[module].append({
                    'solution': sol_id,
                    'line': imp['line'],
                    'alias': imp['alias']
                })

        # Check for different import styles for same module
        for module, imports in module_imports.items():
            if len(imports) > 1:
                # Check if different aliases are used
                aliases = set(imp['alias'] for imp in imports)

                if len(aliases) > 1:
                    solutions = list(set(imp['solution'] for imp in imports))

                    conflict = Conflict(
                        conflict_type=ConflictType.DEPENDENCY_CONFLICT,
                        severity=ConflictSeverity.MEDIUM,
                        description=(
                            f"Inconsistent import aliases for module '{module}': "
                            f"{', '.join(aliases)}. This may cause confusion and "
                            f"potential namespace conflicts."
                        ),
                        affected_solutions=solutions,
                        source_locations=imports,
                        suggested_resolution={
                            'strategy': 'standardize_imports',
                            'explanation': 'Use consistent import aliases across all solutions',
                            'suggestion': f"Standardize on: {module} (no alias) or choose one alias"
                        },
                        metadata={'module': module, 'aliases': list(aliases)},
                        confidence=0.8
                    )
                    conflicts.append(conflict)

        return conflicts

    def _check_module_conflicts(self) -> List[Conflict]:
        """Check for conflicting module usage patterns"""
        conflicts: List[Conflict] = []

        # Look for modules that should be centralized
        common_modules = {
            'logging', 'json', 'os', 'sys', 'pathlib',
            'datetime', 're', 'collections', 'itertools'
        }

        module_usage: Dict[str, List[str]] = defaultdict(list)

        for sol_id, analysis in self.analyses.items():
            for imp in analysis.imports:
                if imp['module'] in common_modules:
                    module_usage[imp['module']].append(sol_id)

        # Report modules used by multiple solutions
        for module, solutions in module_usage.items():
            if len(solutions) > 3:
                conflict = Conflict(
                    conflict_type=ConflictType.DEPENDENCY_CONFLICT,
                    severity=ConflictSeverity.LOW,
                    description=(
                        f"Module '{module}' is used by {len(solutions)} solutions: "
                        f"{', '.join(solutions)}. Consider centralizing common utilities."
                    ),
                    affected_solutions=solutions,
                    source_locations=[{'solution': s, 'module': module} for s in solutions],
                    suggested_resolution={
                        'strategy': 'extract_common',
                        'explanation': 'Extract common functionality to shared module',
                        'suggestion': f"Create shared utilities module for {module} usage"
                    },
                    metadata={'module': module, 'usage_count': len(solutions)},
                    confidence=0.6
                )
                conflicts.append(conflict)

        return conflicts

    def assess_conflict_severity(self, conflict: Conflict) -> str:
        """
        Assess and return the severity level of a conflict

        Args:
            conflict: Conflict object to assess

        Returns:
            Severity level as string
        """
        return conflict.severity.value

    def propose_resolution(self, conflict: Conflict) -> Dict[str, Any]:
        """
        Propose a resolution for a given conflict

        Args:
            conflict: Conflict object to resolve

        Returns:
            Dictionary containing resolution strategy and options
        """
        resolution = conflict.suggested_resolution

        # Add implementation details based on conflict type
        if conflict.conflict_type == ConflictType.NAMING_CONFLICT:
            resolution['implementation_steps'] = [
                "1. Identify all occurrences of conflicting names",
                "2. Choose naming convention (prefix, suffix, or namespace)",
                "3. Update all references consistently",
                "4. Run tests to verify no broken references"
            ]

        elif conflict.conflict_type == ConflictType.LOGIC_CONFLICT:
            resolution['implementation_steps'] = [
                "1. Analyze business logic requirements",
                "2. Determine correct behavior",
                "3. Add arbitration logic or remove conflicting implementation",
                "4. Document decision and rationale",
                "5. Add tests to prevent regression"
            ]

        elif conflict.conflict_type == ConflictType.DEPENDENCY_CONFLICT:
            resolution['implementation_steps'] = [
                "1. Review all dependencies and their versions",
                "2. Check for compatible versions or alternatives",
                "3. Update imports/usage as needed",
                "4. Test integration thoroughly",
                "5. Document dependency requirements"
            ]

        return resolution

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Conflict Detector
    # =========================================================================

    def _trigger_conflict_alerts(
        self,
        operation: str,
        success: bool,
        num_conflicts: int = 0,
        num_critical: int = 0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for conflict detection failures or critical conflicts."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            # Alert on failures or critical conflicts
            if not success or num_critical > 0:
                severity = AlertSeverity.HIGH if not success or num_critical > 0 else AlertSeverity.MEDIUM

                alert_manager.create_alert(
                    title=f"Conflict Detector Alert: {operation}",
                    description=f"Conflict detection operation '{operation}' " +
                                 ("failed" if not success else f"detected {num_critical} critical conflicts") +
                                 (f" out of {num_conflicts} total conflicts" if num_conflicts > 0 else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="conflict_detector",
                    component="conflict_analysis",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Conflict Detector alert: {e}")

    def _extract_conflict_knowledge(
        self,
        operation: str,
        conflicts: List[Conflict],
        sub_solutions_count: int
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract conflict detection knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            # Count conflicts by severity and type
            severity_counts = {}
            type_counts = {}
            for conflict in conflicts:
                severity_counts[conflict.severity.value] = severity_counts.get(conflict.severity.value, 0) + 1
                type_counts[conflict.conflict_type.value] = type_counts.get(conflict.conflict_type.value, 0) + 1

            artifact = KnowledgeArtifact(
                artifact_id=f"conflict_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="conflict_detection",
                source_component="conflict_detector",
                title=f"Conflict Detection: {operation} ({len(conflicts)} conflicts)",
                content={
                    "operation": operation,
                    "num_conflicts": len(conflicts),
                    "num_solutions": sub_solutions_count,
                    "severity_breakdown": severity_counts,
                    "type_breakdown": type_counts,
                    "num_critical": severity_counts.get('CRITICAL', 0),
                    "num_high": severity_counts.get('HIGH', 0),
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "conflict_descriptions": [c.description for c in conflicts[:10]]
                },
                tags=["conflict", "detection", operation]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted Conflict Detection knowledge for {operation}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract Conflict Detection knowledge: {e}")
            return False

    def _track_conflict_performance(
        self,
        operation: str,
        success: bool,
        num_conflicts: int = 0,
        num_critical: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track conflict detection performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            # Quality based on success and conflict ratio
            quality = 1.0 if success else 0.0
            if success:
                # More conflicts found = better detection (inverse of quality)
                # But too many critical conflicts is bad
                conflict_ratio = num_conflicts / max(len(self.analyses), 1)
                critical_penalty = min(num_critical * 0.1, 0.5)
                quality = max(1.0 - critical_penalty, 0.5)
                # Bonus for finding conflicts (detection working)
                quality = min(quality + conflict_ratio * 0.2, 1.0)
            quality = max(quality, 0.0)

            performance_data = StrategyPerformanceData(
                strategy_name=f"conflict_detector_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "num_conflicts": num_conflicts,
                    "num_critical": num_critical
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked Conflict Detection performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track Conflict Detection performance: {e}")


class ConflictReporter:
    """Generate reports from detected conflicts"""

    @staticmethod
    def generate_report(conflicts: List[Conflict], output_format: str = 'text') -> str:
        """
        Generate a formatted report of detected conflicts

        Args:
            conflicts: List of detected conflicts
            output_format: Format for report ('text', 'json', 'markdown')

        Returns:
            Formatted report string
        """
        if output_format == 'json':
            import json
            return json.dumps(
                [conflict.to_dict() for conflict in conflicts],
                indent=2
            )

        elif output_format == 'markdown':
            return ConflictReporter._generate_markdown_report(conflicts)

        else:  # text
            return ConflictReporter._generate_text_report(conflicts)

    @staticmethod
    def _generate_text_report(conflicts: List[Conflict]) -> str:
        """Generate plain text report"""
        lines = []
        lines.append("=" * 80)
        lines.append("CONFLICT DETECTION REPORT")
        lines.append("=" * 80)
        lines.append(f"Total conflicts detected: {len(conflicts)}")
        lines.append("")

        # Group by severity
        by_severity = defaultdict(list)
        for conflict in conflicts:
            by_severity[conflict.severity].append(conflict)

        for severity in [ConflictSeverity.CRITICAL, ConflictSeverity.HIGH,
                        ConflictSeverity.MEDIUM, ConflictSeverity.LOW]:
            if severity not in by_severity:
                continue

            lines.append(f"\n{severity.value} SEVERITY ({len(by_severity[severity])} conflicts)")
            lines.append("-" * 80)

            for conflict in by_severity[severity]:
                lines.append(f"\nConflict Type: {conflict.conflict_type.value}")
                lines.append(f"Description: {conflict.description}")
                lines.append(f"Affected Solutions: {', '.join(conflict.affected_solutions)}")
                lines.append(f"Confidence: {conflict.confidence:.2f}")

                if conflict.suggested_resolution:
                    lines.append(f"Resolution Strategy: {conflict.suggested_resolution.get('strategy', 'N/A')}")

                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _generate_markdown_report(conflicts: List[Conflict]) -> str:
        """Generate Markdown report"""
        lines = []
        lines.append("# Conflict Detection Report\n")
        lines.append(f"**Total Conflicts:** {len(conflicts)}\n")

        # Summary table
        by_severity = defaultdict(list)
        for conflict in conflicts:
            by_severity[conflict.severity].append(conflict)

        lines.append("## Summary\n")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for severity in [ConflictSeverity.CRITICAL, ConflictSeverity.HIGH,
                        ConflictSeverity.MEDIUM, ConflictSeverity.LOW]:
            count = len(by_severity.get(severity, []))
            lines.append(f"| {severity.value} | {count} |")

        lines.append("\n## Conflicts\n")

        # Detailed conflicts
        for conflict in conflicts:
            lines.append(f"### {conflict.severity.value}: {conflict.conflict_type.value}\n")
            lines.append(f"**Description:** {conflict.description}\n\n")
            lines.append(f"**Affected Solutions:** {', '.join(conflict.affected_solutions)}\n\n")

            if conflict.source_locations:
                lines.append("**Locations:**\n")
                for loc in conflict.source_locations:
                    lines.append(f"- {loc}\n")

            if conflict.suggested_resolution:
                lines.append(f"**Resolution Strategy:** {conflict.suggested_resolution.get('strategy', 'N/A')}\n\n")

            lines.append(f"**Confidence:** {conflict.confidence:.2f}\n")
            lines.append("---\n")

        return "\n".join(lines)


# Convenience functions
def detect_conflicts(
    sub_solutions: Optional[List[str]] = None,
    metadata: Optional[List[Dict]] = None,
    strict_mode: bool = False,
    resource_a: Optional[Dict[str, Any]] = None,
    resource_b: Optional[Dict[str, Any]] = None
) -> List[Conflict]:
    """
    Convenience function to detect conflicts

    Args:
        sub_solutions: List of solution code strings (optional)
        metadata: Optional metadata for each solution
        strict_mode: If True, treat all potential conflicts as actual conflicts
        resource_a: First resource dict for simple conflict detection (optional)
        resource_b: Second resource dict for simple conflict detection (optional)

    Returns:
        List of detected conflicts
    """
    # Support simple two-resource conflict detection
    if resource_a is not None and resource_b is not None:
        conflicts = []

        # Check for version conflicts
        if 'version' in resource_a and 'version' in resource_b:
            if resource_a['version'] != resource_b['version']:
                conflicts.append(Conflict(
                    conflict_type=ConflictType.VERSION_CONFLICT,
                    severity=ConflictSeverity.HIGH,
                    description=f"Version mismatch: {resource_a['version']} vs {resource_b['version']}",
                    affected_solutions=['resource_a', 'resource_b'],
                    source_locations=[],
                    suggested_resolution={'strategy': 'resolve_version'},
                    confidence=1.0
                ))

        # Check for data conflicts
        if 'data' in resource_a and 'data' in resource_b:
            if resource_a['data'] != resource_b['data']:
                conflicts.append(Conflict(
                    conflict_type=ConflictType.DATA_CONFLICT,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"Data mismatch between resources",
                    affected_solutions=['resource_a', 'resource_b'],
                    source_locations=[],
                    suggested_resolution={'strategy': 'merge_data'},
                    confidence=0.8
                ))

        return conflicts

    # Standard conflict detection for sub_solutions
    if sub_solutions is None:
        return []

    detector = ConflictDetector(strict_mode=strict_mode)
    return detector.detect_conflicts(sub_solutions, metadata)


def analyze_naming_conflicts(solutions: List[str]) -> List[Conflict]:
    """
    Convenience function to analyze naming conflicts

    Args:
        solutions: List of solution code strings

    Returns:
        List of naming conflicts
    """
    detector = ConflictDetector()
    detector.analyses = {
        f"solution_{i}": detector._analyze_solution(sol, f"solution_{i}")
        for i, sol in enumerate(solutions)
    }
    return detector.analyze_naming_conflicts(solutions)


def analyze_logic_conflicts(solutions: List[str]) -> List[Conflict]:
    """
    Convenience function to analyze logic conflicts

    Args:
        solutions: List of solution code strings

    Returns:
        List of logic conflicts
    """
    detector = ConflictDetector()
    detector.analyses = {
        f"solution_{i}": detector._analyze_solution(sol, f"solution_{i}")
        for i, sol in enumerate(solutions)
    }
    return detector.analyze_logic_conflicts(solutions)


def analyze_dependency_conflicts(solutions: List[str]) -> List[Conflict]:
    """
    Convenience function to analyze dependency conflicts

    Args:
        solutions: List of solution code strings

    Returns:
        List of dependency conflicts
    """
    detector = ConflictDetector()
    detector.analyses = {
        f"solution_{i}": detector._analyze_solution(sol, f"solution_{i}")
        for i, sol in enumerate(solutions)
    }
    return detector.analyze_dependency_conflicts(solutions)


def assess_conflict_severity(conflict: Conflict) -> str:
    """
    Convenience function to assess conflict severity

    Args:
        conflict: Conflict to assess

    Returns:
        Severity level as string
    """
    return conflict.severity.value


def propose_resolution(conflict: Conflict) -> Dict[str, Any]:
    """
    Convenience function to propose conflict resolution

    Args:
        conflict: Conflict to resolve

    Returns:
        Dictionary containing resolution strategy
    """
    detector = ConflictDetector()
    return detector.propose_resolution(conflict)
