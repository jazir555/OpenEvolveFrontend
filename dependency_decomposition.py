"""
Dependency Decomposition Strategy

This module implements the DependencyDecomposition class that analyzes problem
dependencies and creates a dependency-based decomposition.

The strategy identifies:
- Dependencies between sub-problems
- Prerequisite relationships
- Parallel execution opportunities
- Critical path analysis
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import DecompositionResult from problem_decomposition
try:
    from problem_decomposition import DecompositionResult, Component, DecompositionStrategy, ComponentType
except ImportError:
    # Fallback definitions if problem_decomposition is not available
    from enum import Enum
    
    class DecompositionStrategy(Enum):
        HIERARCHICAL = "hierarchical"
        FUNCTIONAL = "functional"
        SEMANTIC = "semantic"
        STRUCTURAL = "structural"
        DEPENDENCY_BASED = "dependency_based"
        COMPLEXITY_BASED = "complexity_based"
    
    class ComponentType(Enum):
        CORE_LOGIC = "core_logic"
        SUPPORTING_FUNCTION = "supporting_function"
        DATA_STRUCTURE = "data_structure"
        INTERFACE = "interface"
        CONFIGURATION = "configuration"
        DOCUMENTATION = "documentation"
        TEST_CASE = "test_case"
        ERROR_HANDLING = "error_handling"
    
    @dataclass
    class Component:
        id: str
        title: str
        content: str
        component_type: ComponentType
        complexity_score: float = 0.0
        dependencies: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)
        evolution_priority: float = 1.0
        estimated_effort: int = 1
    
    @dataclass
    class DecompositionResult:
        original_content: str
        components: List[Component]
        dependency_graph: Dict[str, List[str]]
        decomposition_strategy: DecompositionStrategy
        quality_score: float
        metadata: Dict[str, Any]
        reassembly_instructions: Dict[str, Any]

# Try to import from workflow_structures for SubProblem and DecompositionPlan
try:
    from workflow_structures import SubProblem, DecompositionPlan
    WORKFLOW_STRUCTURES_AVAILABLE = True
except ImportError:
    WORKFLOW_STRUCTURES_AVAILABLE = False
    # Define minimal implementations if not available
    @dataclass
    class SubProblem:
        id: str
        description: str
        dependencies: List[str] = field(default_factory=list)
        title: str = ""
        type: str = "general"
        complexity_score: float = 5.0
        success_criteria: List[str] = field(default_factory=list)
        priority: int = 5
        estimated_effort: int = 1
        validation_gauntlet: Optional[str] = None
        parent_id: Optional[str] = None

    @dataclass
    class DecompositionPlan:
        problem_statement: str
        analyzed_context: Dict[str, Any]
        sub_problems: List[SubProblem]

# Try to import from decomposition_strategy for base classes
try:
    from decomposition_strategy import DecompositionStrategyBase
    DECOMP_STRATEGY_AVAILABLE = True
except ImportError:
    DECOMP_STRATEGY_AVAILABLE = False
    
    class DecompositionStrategyBase(ABC):
        """Base class for decomposition strategies."""
        
        @abstractmethod
        def decompose(self, problem: Any) -> List[SubProblem]:
            """Decompose problem into sub-problems."""
            raise NotImplementedError
        
        @abstractmethod
        def get_strategy_name(self) -> str:
            """Get the name of this strategy."""
            raise NotImplementedError

# Try to import sovereign_data_models
try:
    from sovereign_data_models import (
        ProblemDefinition, DependencyGraph, generate_id
    )
    SOVEREIGN_MODELS_AVAILABLE = True
except ImportError:
    SOVEREIGN_MODELS_AVAILABLE = False
    
    @dataclass
    class ProblemDefinition:
        id: str = ""
        title: str = ""
        description: str = ""
        
    @dataclass
    class DependencyGraph:
        nodes: Dict[str, Any] = field(default_factory=dict)
        edges: Dict[str, List[str]] = field(default_factory=dict)
    
    def generate_id(prefix: str = "id") -> str:
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

# Configure logging
logger = logging.getLogger(__name__)


class DependencyDecomposition:
    """
    Decomposes problems based on dependency relationships between components.
    
    This strategy analyzes the problem to identify:
    - Prerequisite relationships between sub-problems
    - Data dependencies and resource dependencies
    - Opportunities for parallel execution
    - Critical path identification
    
    Key Features:
    - Identifies true prerequisite relationships (not just sequential)
    - Creates dependency graphs for visualization and analysis
    - Optimizes for parallel execution where possible
    - Provides fallback to sequential ordering when needed
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DependencyDecomposition strategy.
        
        Args:
            config: Optional configuration dictionary with settings such as:
                - max_subproblems: Maximum number of sub-problems to create
                - min_subproblem_size: Minimum size/complexity for a sub-problem
                - dependency_threshold: Threshold for establishing dependencies
                - use_llm: Whether to use LLM for enhanced dependency analysis
                - parallelization_preference: Preference for parallel vs sequential
        """
        self.config = config or {}
        self.max_subproblems = self.config.get('max_subproblems', 10)
        self.min_subproblem_size = self.config.get('min_subproblem_size', 1)
        self.dependency_threshold = self.config.get('dependency_threshold', 0.5)
        self.use_llm = self.config.get('use_llm', False)
        self.parallelization_preference = self.config.get('parallelization_preference', 'balanced')
        
        # Try to initialize OpenEvolve client if LLM is requested
        self.openevolve_client = None
        if self.use_llm:
            try:
                from openevolve_client import OpenEvolveClient, OPENEVOLVE_AVAILABLE
                if OPENEVOLVE_AVAILABLE:
                    self.openevolve_client = OpenEvolveClient()
                    logger.info("OpenEvolve client initialized for dependency decomposition")
            except ImportError:
                logger.warning("OpenEvolve not available, using heuristic dependency analysis")
        
        logger.info("DependencyDecomposition initialized with config: %s", self.config)
    
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return "dependency"
    
    def decompose(self, problem: Dict[str, Any]) -> DecompositionResult:
        """
        Decompose a problem based on dependency analysis.
        
        Args:
            problem: Dictionary containing problem definition with keys:
                - id: Unique problem identifier
                - title: Problem title
                - description: Problem description
                - content: Full problem content (optional)
                - constraints: List of constraints (optional)
                - requirements: List of requirements (optional)
                
        Returns:
            DecompositionResult containing:
                - original_content: The original problem description
                - components: List of Component objects representing sub-problems
                - dependency_graph: Graph showing dependencies between components
                - decomposition_strategy: The strategy used (DEPENDENCY_BASED)
                - quality_score: Quality score of the decomposition
                - metadata: Additional metadata about the decomposition
                - reassembly_instructions: Instructions for reassembling solutions
        """
        logger.info(f"Starting dependency decomposition for problem: {problem.get('id', 'unknown')}")
        
        # Extract problem information
        problem_id = problem.get('id', generate_id('problem'))
        problem_title = problem.get('title', 'Untitled Problem')
        problem_description = problem.get('description', problem.get('content', ''))
        problem_content = problem.get('content', problem_description)
        
        # Step 1: Identify potential sub-problems based on the problem structure
        sub_problems = self._identify_subproblems(problem)
        
        if not sub_problems:
            # If no sub-problems identified, create a single component
            sub_problems = [self._create_single_subproblem(problem)]
        
        # Step 2: Analyze dependencies between sub-problems
        dependency_graph = self._analyze_dependencies(sub_problems, problem)
        
        # Step 3: Try LLM-based enhancement if available
        if self.openevolve_client and len(sub_problems) > 1:
            try:
                enhanced_sub_problems = self._enhance_with_llm(sub_problems, problem)
                if enhanced_sub_problems:
                    sub_problems = enhanced_sub_problems
                    # Re-analyze dependencies after enhancement
                    dependency_graph = self._analyze_dependencies(sub_problems, problem)
                    logger.info(f"LLM enhancement applied to {len(sub_problems)} sub-problems")
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}, using heuristic analysis")
        
        # Step 4: Convert sub-problems to Components
        components = self._convert_to_components(sub_problems)
        
        # Step 5: Calculate quality score
        quality_score = self._calculate_quality_score(components, dependency_graph, problem)
        
        # Step 6: Create reassembly instructions
        reassembly_instructions = self._create_reassembly_instructions(components, dependency_graph)
        
        # Step 7: Build metadata
        metadata = {
            'problem_id': problem_id,
            'problem_title': problem_title,
            'num_components': len(components),
            'num_dependencies': sum(len(deps) for deps in dependency_graph.values()),
            'parallel_groups': self._identify_parallel_groups(dependency_graph),
            'critical_path': self._find_critical_path(dependency_graph, components),
            'strategy_config': self.config
        }
        
        # Create and return DecompositionResult
        result = DecompositionResult(
            original_content=problem_content,
            components=components,
            dependency_graph=dependency_graph,
            decomposition_strategy=DecompositionStrategy.DEPENDENCY_BASED,
            quality_score=quality_score,
            metadata=metadata,
            reassembly_instructions=reassembly_instructions
        )
        
        logger.info(f"Dependency decomposition completed: {len(components)} components, "
                   f"{metadata['num_dependencies']} dependencies, quality={quality_score:.2f}")
        
        return result
    
    def _identify_subproblems(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify potential sub-problems based on problem structure.
        
        Uses heuristics to break down the problem:
        - Section headers and numbered items
        - Sequential steps or phases
        - Independent functional areas
        - Data processing stages
        """
        content = problem.get('content', problem.get('description', ''))
        title = problem.get('title', '')
        
        sub_problems = []
        
        # Heuristic 1: Look for numbered sections (1., 2., Step 1, etc.)
        numbered_sections = re.findall(
            r'(?:^|\n)\s*(?:Step\s+|Phase\s+|Stage\s+|Part\s+)?(\d+)[.:\)]\s*([^\n]+(?:\n(?!(?:\d+|Step|Phase|Stage|Part)\s*[.:\)])[^\n]+)*)',
            content, re.IGNORECASE
        )
        
        if numbered_sections and len(numbered_sections) >= 2:
            for num, section_content in numbered_sections[:self.max_subproblems]:
                sub_problems.append({
                    'id': generate_id(f'subproblem_{num}'),
                    'title': f"Step {num}: {section_content[:50].strip()}",
                    'description': section_content.strip(),
                    'type': 'sequential',
                    'sequence_number': int(num)
                })
        
        # Heuristic 2: Look for bullet points with clear separation
        if len(sub_problems) < 2:
            bullet_points = re.findall(
                r'(?:^|\n)\s*[-**]\s+([^\n]+(?:\n(?![-**]\s)[^\n]+)*)',
                content
            )
            if len(bullet_points) >= 2:
                sub_problems = []
                for i, point in enumerate(bullet_points[:self.max_subproblems], 1):
                    sub_problems.append({
                        'id': generate_id(f'subproblem_{i}'),
                        'title': point[:60].strip(),
                        'description': point.strip(),
                        'type': 'independent',
                        'sequence_number': i
                    })
        
        # Heuristic 3: Look for keyword-based functional areas
        if len(sub_problems) < 2:
            sub_problems = self._identify_functional_areas(content)
        
        # If still no sub-problems, create default ones based on common patterns
        if len(sub_problems) < 2:
            sub_problems = self._create_default_subproblems(problem)
        
        return sub_problems
    
    def _identify_functional_areas(self, content: str) -> List[Dict[str, Any]]:
        """Identify functional areas based on keywords."""
        functional_keywords = {
            'data': ['data collection', 'data processing', 'data analysis', 'data storage'],
            'frontend': ['frontend', 'ui', 'user interface', 'client-side', 'display'],
            'backend': ['backend', 'server', 'api', 'database', 'server-side'],
            'testing': ['testing', 'validation', 'verification', 'quality assurance'],
            'deployment': ['deployment', 'production', 'release', 'distribution'],
            'design': ['design', 'architecture', 'planning', 'specification'],
            'implementation': ['implementation', 'development', 'coding', 'programming'],
            'integration': ['integration', 'connection', 'interface', 'compatibility']
        }
        
        content_lower = content.lower()
        identified_areas = []
        
        for area_type, keywords in functional_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    # Extract context around the keyword
                    match = re.search(
                        rf'[^.]*{re.escape(keyword)}[^.]*\.',
                        content_lower
                    )
                    if match:
                        identified_areas.append({
                            'type': area_type,
                            'context': match.group(0),
                            'keyword': keyword
                        })
                        break
        
        # Create sub-problems for identified areas
        sub_problems = []
        for i, area in enumerate(identified_areas[:self.max_subproblems], 1):
            sub_problems.append({
                'id': generate_id(f'subproblem_{i}'),
                'title': f"{area['type'].title()}: {area['keyword'].title()}",
                'description': area['context'].capitalize(),
                'type': area['type'],
                'sequence_number': i
            })
        
        return sub_problems
    
    def _create_default_subproblems(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create default sub-problems based on common decomposition patterns."""
        title = problem.get('title', 'Problem')
        description = problem.get('description', '')
        
        return [
            {
                'id': generate_id('subproblem_1'),
                'title': f"Analysis and Planning: {title[:40]}",
                'description': f"Analyze requirements and plan approach for: {description[:100]}",
                'type': 'analysis',
                'sequence_number': 1
            },
            {
                'id': generate_id('subproblem_2'),
                'title': f"Implementation: {title[:40]}",
                'description': f"Implement the core solution for: {description[:100]}",
                'type': 'implementation',
                'sequence_number': 2
            },
            {
                'id': generate_id('subproblem_3'),
                'title': f"Validation and Testing: {title[:40]}",
                'description': f"Validate and test the solution for: {description[:100]}",
                'type': 'validation',
                'sequence_number': 3
            }
        ]
    
    def _create_single_subproblem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Create a single sub-problem for the entire problem."""
        return {
            'id': generate_id('subproblem_single'),
            'title': problem.get('title', 'Complete Problem'),
            'description': problem.get('description', problem.get('content', '')),
            'type': 'complete',
            'sequence_number': 1
        }
    
    def _analyze_dependencies(
        self, 
        sub_problems: List[Dict[str, Any]], 
        problem: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Analyze dependencies between sub-problems.
        
        Returns a dictionary mapping component IDs to lists of dependency IDs.
        """
        dependency_graph = {}
        
        if len(sub_problems) <= 1:
            dependency_graph[sub_problems[0]['id']] = []
            return dependency_graph
        
        # Initialize empty dependency list for each sub-problem
        for sp in sub_problems:
            dependency_graph[sp['id']] = []
        
        # Heuristic 1: Sequential dependencies based on sequence numbers
        sorted_problems = sorted(sub_problems, key=lambda x: x.get('sequence_number', 0))
        
        # For sequential types, create chain dependencies
        for i, sp in enumerate(sorted_problems):
            if sp.get('type') == 'sequential' and i > 0:
                # Find the previous sequential item
                for j in range(i - 1, -1, -1):
                    if sorted_problems[j].get('type') == 'sequential':
                        dependency_graph[sp['id']].append(sorted_problems[j]['id'])
                        break
        
        # Heuristic 2: Type-based dependencies
        type_dependencies = {
            'implementation': ['design', 'analysis', 'planning'],
            'testing': ['implementation', 'development', 'coding'],
            'deployment': ['testing', 'validation', 'implementation'],
            'integration': ['implementation', 'design'],
            'validation': ['implementation', 'testing']
        }
        
        for sp in sub_problems:
            sp_type = sp.get('type', '').lower()
            if sp_type in type_dependencies:
                required_types = type_dependencies[sp_type]
                for other_sp in sub_problems:
                    if other_sp['id'] != sp['id']:
                        other_type = other_sp.get('type', '').lower()
                        if other_type in required_types:
                            if other_sp['id'] not in dependency_graph[sp['id']]:
                                dependency_graph[sp['id']].append(other_sp['id'])
        
        # Heuristic 3: Content-based dependencies (keyword matching)
        dependency_keywords = [
            'requires', 'depends on', 'needs', 'prerequisite',
            'after', 'following', 'subsequent to', 'based on'
        ]
        
        for i, sp in enumerate(sub_problems):
            desc_lower = sp.get('description', '').lower()
            for keyword in dependency_keywords:
                if keyword in desc_lower:
                    # Check if this references another sub-problem
                    for other_sp in sub_problems:
                        if other_sp['id'] != sp['id']:
                            other_title = other_sp.get('title', '').lower()
                            # Simple check: if other title appears after dependency keyword
                            idx = desc_lower.find(keyword)
                            if idx >= 0 and other_title[:20] in desc_lower[idx:idx + 100]:
                                if other_sp['id'] not in dependency_graph[sp['id']]:
                                    dependency_graph[sp['id']].append(other_sp['id'])
        
        # Ensure no circular dependencies (simple check)
        dependency_graph = self._remove_circular_dependencies(dependency_graph)
        
        return dependency_graph
    
    def _remove_circular_dependencies(
        self, 
        dependency_graph: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Remove circular dependencies from the graph."""
        # Simple cycle detection and removal
        def has_path(start: str, end: str, visited: Set[str] = None) -> bool:
            if visited is None:
                visited = set()
            if start in visited:
                return False
            visited.add(start)
            if start == end:
                return True
            for dep in dependency_graph.get(start, []):
                if has_path(dep, end, visited.copy()):
                    return True
            return False
        
        cleaned_graph = {k: list(v) for k, v in dependency_graph.items()}
        
        for node in list(cleaned_graph.keys()):
            deps = cleaned_graph[node]
            # Remove dependencies that would create cycles
            cleaned_deps = [
                dep for dep in deps 
                if not has_path(dep, node)
            ]
            cleaned_graph[node] = cleaned_deps
        
        return cleaned_graph
    
    def _enhance_with_llm(
        self, 
        sub_problems: List[Dict[str, Any]], 
        problem: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Use LLM to enhance dependency analysis."""
        if not self.openevolve_client:
            return None
        
        # Build sub-problem descriptions
        sp_descriptions = []
        for i, sp in enumerate(sub_problems, 1):
            sp_descriptions.append(
                f"{i}. {sp.get('title', 'Untitled')}\n"
                f"   Type: {sp.get('type', 'unknown')}\n"
                f"   Description: {sp.get('description', '')[:200]}..."
            )
        
        prompt = f"""Analyze these sub-problems and identify TRUE prerequisite dependencies.

PARENT PROBLEM: {problem.get('title', 'Untitled')}

SUB-PROBLEMS:
{chr(10).join(sp_descriptions)}

TASK:
For each sub-problem, identify which OTHER sub-problems must be completed FIRST (true prerequisites).
Only specify dependencies that are NECESSARY - don't create artificial sequential dependencies.
Consider:
- Does sub-problem X need outputs/results from sub-problem Y?
- Can sub-problems be worked on in parallel?
- What are the true blocking relationships?

OUTPUT FORMAT:
For each sub-problem, list its dependencies as comma-separated numbers, or "none" if independent.

1: [dependencies or "none"]
2: [dependencies or "none"]
3: [dependencies or "none"]
...

Example:
1: none
2: 1
3: 1
4: 2,3
5: none

Provide dependencies for all {len(sub_problems)} sub-problems:"""
        
        try:
            result = self.openevolve_client.evolve(
                content=prompt,
                evolution_mode="standard",
                content_type="analysis",
                max_iterations=1,
                temperature=0.2,
                max_tokens=500
            )
            
            if result.success and result.best_code:
                return self._apply_llm_dependencies(sub_problems, result.best_code)
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
        
        return None
    
    def _apply_llm_dependencies(
        self, 
        sub_problems: List[Dict[str, Any]], 
        llm_response: str
    ) -> List[Dict[str, Any]]:
        """Parse LLM dependency analysis and apply to sub-problems."""
        # Create ID mapping
        id_map = {i + 1: sp['id'] for i, sp in enumerate(sub_problems)}
        
        # Parse dependencies
        lines = llm_response.strip().split('\n')
        dependency_map = {}
        
        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
            
            try:
                num_str, deps_str = line.split(':', 1)
                num = int(num_str.strip())
                deps_str = deps_str.strip().lower()
                
                if deps_str in ['none', 'n/a', '']:
                    dependency_map[num] = []
                else:
                    # Extract dependency numbers
                    dep_nums = [int(d) for d in re.findall(r'\d+', deps_str)]
                    dependency_map[num] = [
                        id_map[d] for d in dep_nums 
                        if d in id_map and d != num
                    ]
            except Exception as e:
                logger.debug(f"Failed to parse dependency line '{line}': {e}")
                continue
        
        # Apply dependencies to sub-problems
        enhanced_sub_problems = []
        for i, sp in enumerate(sub_problems, 1):
            enhanced_sp = dict(sp)
            enhanced_sp['dependencies'] = dependency_map.get(i, [])
            enhanced_sub_problems.append(enhanced_sp)
        
        return enhanced_sub_problems
    
    def _convert_to_components(self, sub_problems: List[Dict[str, Any]]) -> List[Component]:
        """Convert sub-problem dictionaries to Component objects."""
        components = []
        
        for sp in sub_problems:
            # Map type string to ComponentType enum
            type_mapping = {
                'analysis': 'CORE_LOGIC',
                'implementation': 'CORE_LOGIC',
                'validation': 'TEST_CASE',
                'testing': 'TEST_CASE',
                'design': 'CORE_LOGIC',
                'deployment': 'SUPPORTING_FUNCTION',
                'integration': 'INTERFACE',
                'frontend': 'INTERFACE',
                'backend': 'CORE_LOGIC',
                'data': 'DATA_STRUCTURE',
                'sequential': 'CORE_LOGIC',
                'independent': 'SUPPORTING_FUNCTION',
                'complete': 'CORE_LOGIC',
                'general': 'CORE_LOGIC'
            }
            
            sp_type = sp.get('type', 'general').lower()
            component_type_str = type_mapping.get(sp_type, 'CORE_LOGIC')
            
            # Convert string to ComponentType enum
            try:
                component_type = ComponentType[component_type_str]
            except (KeyError, TypeError):
                component_type = ComponentType.CORE_LOGIC
            
            component = Component(
                id=sp['id'],
                title=sp.get('title', f"Component {sp['id']}"),
                content=sp.get('description', sp.get('title', '')),
                component_type=component_type,
                complexity_score=self._estimate_complexity(sp),
                dependencies=sp.get('dependencies', []),
                metadata={
                    'sequence_number': sp.get('sequence_number', 0),
                    'original_type': sp.get('type', 'general')
                }
            )
            components.append(component)
        
        return components
    
    def _estimate_complexity(self, sub_problem: Dict[str, Any]) -> float:
        """Estimate complexity of a sub-problem."""
        description = sub_problem.get('description', '')
        
        # Simple heuristics for complexity
        complexity = 0.5  # Base complexity
        
        # Length-based adjustment
        desc_length = len(description)
        if desc_length > 500:
            complexity += 0.2
        elif desc_length < 100:
            complexity -= 0.1
        
        # Keyword-based complexity
        complex_keywords = ['integrate', 'optimize', 'scale', 'secure', 'algorithm', 'distributed']
        simple_keywords = ['document', 'update', 'fix typo', 'rename']
        
        desc_lower = description.lower()
        for keyword in complex_keywords:
            if keyword in desc_lower:
                complexity += 0.1
        for keyword in simple_keywords:
            if keyword in desc_lower:
                complexity -= 0.1
        
        # Clamp to 0-1 range
        return max(0.0, min(1.0, complexity))
    
    def _calculate_quality_score(
        self, 
        components: List[Component], 
        dependency_graph: Dict[str, List[str]],
        problem: Dict[str, Any]
    ) -> float:
        """Calculate quality score for the decomposition."""
        if not components:
            return 0.0
        
        scores = []
        
        # Factor 1: Balance of component sizes
        if len(components) > 1:
            complexities = [c.complexity_score for c in components]
            avg_complexity = sum(complexities) / len(complexities)
            variance = sum((c - avg_complexity) ** 2 for c in complexities) / len(complexities)
            balance_score = max(0, 1.0 - variance * 4)  # Higher variance = lower score
            scores.append(balance_score)
        
        # Factor 2: Dependency graph health
        total_possible_deps = len(components) * (len(components) - 1) / 2
        if total_possible_deps > 0:
            actual_deps = sum(len(deps) for deps in dependency_graph.values())
            # Optimal is some dependencies but not too many
            optimal_deps = total_possible_deps * 0.3  # ~30% connectivity
            dep_ratio = actual_deps / optimal_deps if optimal_deps > 0 else 1.0
            dep_score = 1.0 - abs(1.0 - dep_ratio) * 0.5
            scores.append(max(0, min(1, dep_score)))
        
        # Factor 3: Coverage (problem content reflected in components)
        problem_content = problem.get('content', problem.get('description', '')).lower()
        component_content = ' '.join(c.content.lower() for c in components)
        
        # Simple coverage check - count words from problem in components
        problem_words = set(problem_content.split())
        if problem_words:
            covered_words = sum(1 for w in problem_words if w in component_content)
            coverage_score = covered_words / len(problem_words)
            scores.append(min(1.0, coverage_score * 2))  # Scale up a bit
        
        # Factor 4: Number of components (not too many, not too few)
        optimal_count = 5  # Arbitrary optimal
        count_score = 1.0 - abs(len(components) - optimal_count) / optimal_count
        scores.append(max(0, min(1, count_score)))
        
        # Calculate final score
        if scores:
            return sum(scores) / len(scores)
        return 0.5
    
    def _create_reassembly_instructions(
        self, 
        components: List[Component], 
        dependency_graph: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Create instructions for reassembling component solutions."""
        # Determine execution order (topological sort)
        execution_order = self._topological_sort(dependency_graph)
        
        # Group by parallel execution levels
        parallel_groups = self._identify_parallel_groups(dependency_graph)
        
        return {
            'execution_order': execution_order,
            'parallel_groups': parallel_groups,
            'assembly_strategy': 'dependency_ordered',
            'merge_instructions': [
                f"Execute component '{comp_id}' after all its dependencies are complete"
                for comp_id in execution_order
            ],
            'validation_checkpoints': [
                f"Verify outputs of component '{comp_id}' before proceeding"
                for comp_id in execution_order
            ]
        }
    
    def _topological_sort(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """
        Perform topological sort on the dependency graph.
        
        Returns a valid execution order respecting dependencies.
        """
        # Kahn's algorithm
        in_degree = {node: 0 for node in dependency_graph}
        for deps in dependency_graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 0  # Ensure all deps are in dict
        
        for node, deps in dependency_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] = in_degree.get(dep, 0) + 1
        
        # Find all nodes with no dependencies
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)
            
            # Find nodes that depend on this one
            for other_node, deps in dependency_graph.items():
                if node in deps:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)
        
        # Add any remaining nodes (shouldn't happen with valid DAG)
        for node in dependency_graph:
            if node not in result:
                result.append(node)
        
        return result
    
    def _identify_parallel_groups(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Identify groups of components that can be executed in parallel."""
        if not dependency_graph:
            return []
        
        # Group by topological level
        levels = {}
        
        def get_level(node: str, visited: Set[str] = None) -> int:
            if visited is None:
                visited = set()
            if node in levels:
                return levels[node]
            if node in visited:
                return 0  # Cycle detected
            visited.add(node)
            
            deps = dependency_graph.get(node, [])
            if not deps:
                levels[node] = 0
            else:
                levels[node] = max(get_level(d, visited.copy()) for d in deps) + 1
            return levels[node]
        
        # Calculate levels for all nodes
        for node in dependency_graph:
            get_level(node)
        
        # Group by level
        max_level = max(levels.values()) if levels else 0
        groups = [[] for _ in range(max_level + 1)]
        
        for node, level in levels.items():
            groups[level].append(node)
        
        # Remove empty groups
        return [g for g in groups if g]
    
    def _find_critical_path(
        self, 
        dependency_graph: Dict[str, List[str]], 
        components: List[Component]
    ) -> List[str]:
        """Find the critical path (longest path) through the dependency graph."""
        if not dependency_graph or not components:
            return []
        
        # Build reverse graph (dependencies -> dependents)
        reverse_graph = {comp.id: [] for comp in components}
        for node, deps in dependency_graph.items():
            for dep in deps:
                if dep in reverse_graph:
                    reverse_graph[dep].append(node)
        
        # Find all end nodes (no dependents)
        end_nodes = [
            comp.id for comp in components 
            if not reverse_graph.get(comp.id, [])
        ]
        
        if not end_nodes:
            end_nodes = [components[0].id]
        
        # Find longest path to each end node
        def find_path_to_start(node: str, visited: Set[str] = None) -> List[str]:
            if visited is None:
                visited = set()
            if node in visited:
                return [node]
            visited.add(node)
            
            deps = dependency_graph.get(node, [])
            if not deps:
                return [node]
            
            # Find longest dependency path
            longest_dep_path = []
            for dep in deps:
                if dep not in visited:
                    dep_path = find_path_to_start(dep, visited.copy())
                    if len(dep_path) > len(longest_dep_path):
                        longest_dep_path = dep_path
            
            return longest_dep_path + [node]
        
        # Find the longest path among all end nodes
        critical_path = []
        for end_node in end_nodes:
            path = find_path_to_start(end_node)
            if len(path) > len(critical_path):
                critical_path = path
        
        return list(reversed(critical_path))


# Convenience function for direct usage
def decompose_by_dependencies(
    problem: Dict[str, Any],
    config: Optional[Dict] = None
) -> DecompositionResult:
    """
    Convenience function to decompose a problem using dependency analysis.
    
    Args:
        problem: Problem dictionary with id, title, description, etc.
        config: Optional configuration for the decomposition
        
    Returns:
        DecompositionResult with components and dependency information
    """
    strategy = DependencyDecomposition(config)
    return strategy.decompose(problem)


# For compatibility with DecompositionStrategyBase-based systems
class DependencyDecompositionStrategy(DependencyDecomposition, DecompositionStrategyBase):
    """
    Adapter class that combines DependencyDecomposition with DecompositionStrategyBase.
    
    This allows the dependency decomposition to be used in systems expecting
    the DecompositionStrategyBase interface (e.g., decomposition_engine_backup_fix.py).
    
    Usage:
        from dependency_decomposition import DependencyDecompositionStrategy
        
        strategy = DependencyDecompositionStrategy()
        sub_problems = strategy.decompose(problem_definition)  # Returns List[SubProblem]
    """
    
    def __init__(self, openevolve_client=None, config: Optional[Dict] = None):
        """Initialize with optional OpenEvolve client."""
        merged_config = config or {}
        if openevolve_client:
            merged_config['use_llm'] = True
            # Store client reference for later use
            self._client_ref = openevolve_client
        super().__init__(merged_config)
        if openevolve_client:
            self.openevolve_client = openevolve_client
    
    def decompose(self, problem) -> List[Any]:
        """
        Decompose problem into sub-problems (DecompositionStrategyBase interface).
        
        This method adapts the DecompositionResult from the parent class to
        return a List[SubProblem] as expected by DecompositionStrategyBase.
        
        Args:
            problem: ProblemDefinition object or Dict
            
        Returns:
            List of SubProblem objects
        """
        # Convert ProblemDefinition to dict if needed
        if hasattr(problem, 'id'):
            problem_dict = {
                'id': problem.id,
                'title': getattr(problem, 'title', ''),
                'description': getattr(problem, 'description', ''),
                'content': getattr(problem, 'description', ''),
                'constraints': getattr(problem, 'constraints', []),
            }
        else:
            problem_dict = problem
        
        # Call parent decompose method to get DecompositionResult
        result = DependencyDecomposition.decompose(self, problem_dict)
        
        # Convert Components to SubProblems
        sub_problems = []
        for comp in result.components:
            # Build description including title if available
            description = comp.content
            if comp.title and comp.title not in comp.content:
                description = f"{comp.title}: {comp.content}"
            
            sub_problem = SubProblem(
                id=comp.id,
                description=description,
                dependencies=comp.dependencies,
                ai_suggested_complexity_score=int(comp.complexity_score * 10),
                acceptance_criteria=result.reassembly_instructions.get('validation_checkpoints', []),
                specific_constraints=[
                    f"type:{comp.component_type.value if hasattr(comp.component_type, 'value') else str(comp.component_type)}",
                    f"title:{comp.title}"
                ],
                estimated_resources={
                    'effort': comp.estimated_effort,
                    'complexity': comp.complexity_score,
                    'title': comp.title,
                    'component_type': comp.component_type.value if hasattr(comp.component_type, 'value') else str(comp.component_type)
                }
            )
            sub_problems.append(sub_problem)
        
        return sub_problems


# Alias for use in DecompositionEngine (matches the TODO references)
class DependencyDecompositionAdapter(DependencyDecompositionStrategy):
    """
    Adapter class for use in DecompositionEngine.
    
    This class is specifically for integration with DecompositionEngine
    in decomposition_engine_backup_fix.py. It returns List[SubProblem]
    as expected by the engine.
    
    Usage:
        from dependency_decomposition import DependencyDecompositionAdapter as DependencyDecomposition
        
        engine = DecompositionEngine()
        engine.strategies['dependency'] = DependencyDecomposition()
    """
    pass


if __name__ == "__main__":
    # Simple demonstration
    logging.basicConfig(level=logging.INFO)
    
    # Example problem
    example_problem = {
        'id': 'example_001',
        'title': 'Build a Web Application',
        'description': '''
        Build a complete web application with the following phases:
        
        1. Design the database schema and API endpoints
        2. Implement the backend API and database layer
        3. Create the frontend user interface
        4. Integrate frontend with backend API
        5. Test the complete application
        6. Deploy to production
        ''',
        'content': 'Build a complete web application with database, API, and frontend.'
    }
    
    # Create decomposition strategy
    strategy = DependencyDecomposition(config={'max_subproblems': 6})
    
    # Decompose the problem
    result = strategy.decompose(example_problem)
    
    # Print results
    print(f"\nDecomposition Result:")
    print(f"  Strategy: {result.decomposition_strategy}")
    print(f"  Quality Score: {result.quality_score:.2f}")
    print(f"  Number of Components: {len(result.components)}")
    print(f"\nDependency Graph:")
    for comp_id, deps in result.dependency_graph.items():
        print(f"  {comp_id}: {deps if deps else 'No dependencies'}")
    print(f"\nComponents:")
    for comp in result.components:
        print(f"  - {comp.id}: {comp.metadata.get('title', 'Untitled')[:50]}")
    print(f"\nParallel Execution Groups:")
    for i, group in enumerate(result.metadata.get('parallel_groups', [])):
        print(f"  Group {i+1}: {group}")
    print(f"\nCritical Path:")
    print(f"  {' -> '.join(result.metadata.get('critical_path', []))}")
