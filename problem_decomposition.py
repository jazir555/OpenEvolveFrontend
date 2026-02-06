"""
Problem Decomposition System for OpenEvolve
Implements hierarchical decomposition strategies for complex content
"""

import time
import json
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import with fallback for standalone operation
try:
    from content_analyzer import ContentAnalyzer
except ImportError:
    class ContentAnalyzer:
        """Simple mock content analyzer for problem decomposition"""
        def analyze_content(self, content: str) -> Dict[str, Any]:
            return {
                "length": len(content),
                "lines": len(content.split('\n')),
                "words": len(content.split()),
                "complexity": min(len(content) / 1000, 1.0)
            }

try:
    from error_handler import with_error_handling, ErrorCategory, ErrorSeverity
except ImportError:
    # Mock error handling for standalone operation
    def with_error_handling(category=None, severity=None, fallback_value=None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                    print(f"Error in {func.__name__}: {e}")
                    return fallback_value
            return wrapper
        return decorator

    class ErrorCategory:
        PROCESSING = "processing"

    class ErrorSeverity:
        MEDIUM = "medium"


class DecompositionStrategy(Enum):
    """Strategies for problem decomposition"""
    HIERARCHICAL = "hierarchical"
    FUNCTIONAL = "functional"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    DEPENDENCY_BASED = "dependency_based"
    COMPLEXITY_BASED = "complexity_based"


class ComponentType(Enum):
    """Types of components that can be identified"""
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
    """Represents a decomposed component"""
    id: str
    title: str
    content: str
    component_type: ComponentType
    complexity_score: float = 0.0  # 0-1 scale
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evolution_priority: float = 1.0  # Higher = evolve first
    estimated_effort: int = 1  # 1-10 scale


@dataclass
class DecompositionResult:
    """Result of problem decomposition"""
    original_content: str
    components: List[Component]
    dependency_graph: Dict[str, List[str]]
    decomposition_strategy: DecompositionStrategy
    quality_score: float
    metadata: Dict[str, Any]
    reassembly_instructions: Dict[str, Any]


@dataclass
class ReassemblyResult:
    """Result of component reassembly"""
    reassembled_content: str
    components_used: List[str]
    quality_score: float
    improvement_metrics: Dict[str, Any]
    metadata: Dict[str, Any]


class ProblemDecomposer:
    """Main problem decomposition system"""
    
    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
        self.decomposition_history: List[DecompositionResult] = []
    
    @with_error_handling(
        category="processing",
        severity="medium",
        fallback_value=None
    )
    def decompose_content(
        self,
        content: str,
        strategy: DecompositionStrategy = DecompositionStrategy.HIERARCHICAL,
        max_components: int = 10,
        min_component_size: int = 50,
        **kwargs
    ) -> DecompositionResult:
        """
        Decompose content into manageable components using specified strategy.
        
        Args:
            content: Content to decompose
            strategy: Decomposition strategy to use
            max_components: Maximum number of components to create
            min_component_size: Minimum size for a component (in characters)
            **kwargs: Additional parameters
            
        Returns:
            DecompositionResult: Decomposition results with components
        """
        start_time = time.time()
        
        # Analyze content first
        analysis = self.content_analyzer.analyze_content(content)
        
        # Choose decomposition method based on strategy
        if strategy == DecompositionStrategy.HIERARCHICAL:
            components = self._hierarchical_decomposition(
                content, analysis, max_components, min_component_size)
        elif strategy == DecompositionStrategy.FUNCTIONAL:
            components = self._functional_decomposition(
                content, analysis, max_components, min_component_size)
        elif strategy == DecompositionStrategy.SEMANTIC:
            components = self._semantic_decomposition(
                content, analysis, max_components, min_component_size)
        elif strategy == DecompositionStrategy.STRUCTURAL:
            components = self._structural_decomposition(
                content, analysis, max_components, min_component_size)
        elif strategy == DecompositionStrategy.DEPENDENCY_BASED:
            components = self._dependency_based_decomposition(
                content, analysis, max_components, min_component_size)
        else:  # COMPLEXITY_BASED
            components = self._complexity_based_decomposition(
                content, analysis, max_components, min_component_size)
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(components)
        
        # Calculate quality score
        quality_score = self._calculate_decomposition_quality(content, components)
        
        # Create reassembly instructions
        reassembly_instructions = self._create_reassembly_instructions(components, dependency_graph)
        
        # Create result
        result = DecompositionResult(
            original_content=content,
            components=components,
            dependency_graph=dependency_graph,
            decomposition_strategy=strategy,
            quality_score=quality_score,
            metadata={
                "decomposition_time": time.time() - start_time,
                "component_count": len(components),
                "avg_component_size": sum(len(c.content) for c in components) / len(components) if components else 0,
                "complexity_distribution": self._analyze_complexity_distribution(components),
                "strategy_used": strategy.value
            },
            reassembly_instructions=reassembly_instructions
        )
        
        # Store in history
        self.decomposition_history.append(result)
        
        return result
    
    def _hierarchical_decomposition(
        self, content: str, analysis: Any, max_components: int, min_component_size: int
    ) -> List[Component]:
        """Decompose content hierarchically based on major structural elements"""
        components = []
        
        # Split by major structural elements
        sections = self._identify_sections(content)
        
        for i, section in enumerate(sections):
            if len(section.strip()) >= min_component_size:
                component = Component(
                    id=f"hier_{i+1}",
                    title=f"Section {i+1}",
                    content=section.strip(),
                    component_type=self._classify_component_type(section),
                    complexity_score=self._calculate_component_complexity(section),
                    metadata={
                        "section_index": i,
                        "original_position": content.find(section)
                    }
                )
                components.append(component)
                
                if len(components) >= max_components:
                    break
        
        return components
    
    def _functional_decomposition(
        self, content: str, analysis: Any, max_components: int, min_component_size: int
    ) -> List[Component]:
        """Decompose content based on functional units (functions, classes, procedures, etc.)"""
        components = []
        
        # Identify functional units
        functional_units = self._identify_functional_units(content)
        
        for i, unit in enumerate(functional_units):
            if len(unit['content']) >= min_component_size:
                component = Component(
                    id=f"func_{i+1}",
                    title=unit['name'],
                    content=unit['content'],
                    component_type=ComponentType.CORE_LOGIC if unit['type'] == 'main' else ComponentType.SUPPORTING_FUNCTION,
                    complexity_score=self._calculate_component_complexity(unit['content']),
                    dependencies=unit.get('dependencies', []),
                    metadata={
                        "function_type": unit['type'],
                        "parameters": unit.get('parameters', [])
                    }
                )
                components.append(component)
                
                if len(components) >= max_components:
                    break
        
        return components
    
    def _semantic_decomposition(
        self, content: str, analysis: Any, max_components: int, min_component_size: int
    ) -> List[Component]:
        """Decompose content based on semantic meaning using semantic analysis"""
        components = []
        
        # Use semantic analysis to identify coherent chunks
        semantic_chunks = self._identify_semantic_chunks(content)
        
        for i, chunk in enumerate(semantic_chunks):
            if len(chunk['content']) >= min_component_size:
                component = Component(
                    id=f"sem_{i+1}",
                    title=chunk['topic'],
                    content=chunk['content'],
                    component_type=self._classify_component_type(chunk['content']),
                    complexity_score=chunk['complexity_score'],
                    metadata={
                        "semantic_topic": chunk['topic'],
                        "coherence_score": chunk['coherence']
                    }
                )
                components.append(component)
                
                if len(components) >= max_components:
                    break
        
        return components
    
    def _structural_decomposition(
        self, content: str, analysis: Any, max_components: int, min_component_size: int
    ) -> List[Component]:
        """Decompose content based on structural patterns"""
        components = []
        
        # Identify structural elements
        structural_elements = self._identify_structural_elements(content)
        
        for i, element in enumerate(structural_elements):
            if len(element['content']) >= min_component_size:
                component = Component(
                    id=f"struct_{i+1}",
                    title=element['title'],
                    content=element['content'],
                    component_type=self._map_structure_type_to_component_type(element['type']),
                    complexity_score=self._calculate_component_complexity(element['content']),
                    metadata={
                        "structural_type": element['type'],
                        "nesting_level": element.get('level', 0)
                    }
                )
                components.append(component)
                
                if len(components) >= max_components:
                    break
        
        return components
    
    def _dependency_based_decomposition(
        self, content: str, analysis: Any, max_components: int, min_component_size: int
    ) -> List[Component]:
        """Decompose content based on dependency relationships"""
        components = []
        
        # Identify dependency clusters
        dependency_clusters = self._identify_dependency_clusters(content)
        
        for i, cluster in enumerate(dependency_clusters):
            if len(cluster['content']) >= min_component_size:
                component = Component(
                    id=f"dep_{i+1}",
                    title=cluster['name'],
                    content=cluster['content'],
                    component_type=ComponentType.CORE_LOGIC if cluster['is_core'] else ComponentType.SUPPORTING_FUNCTION,
                    complexity_score=cluster['complexity'],
                    dependencies=cluster['dependencies'],
                    metadata={
                        "dependency_cluster": True,
                        "cluster_size": cluster['size']
                    }
                )
                components.append(component)
                
                if len(components) >= max_components:
                    break
        
        return components
    
    def _complexity_based_decomposition(
        self, content: str, analysis: Any, max_components: int, min_component_size: int
    ) -> List[Component]:
        """Decompose content based on complexity analysis"""
        components = []
        
        # Identify high-complexity areas that need separate attention
        complexity_regions = self._identify_complexity_regions(content)
        
        for i, region in enumerate(complexity_regions):
            if len(region['content']) >= min_component_size:
                component = Component(
                    id=f"complex_{i+1}",
                    title=f"Complex Region {i+1}",
                    content=region['content'],
                    component_type=self._classify_component_type(region['content']),
                    complexity_score=region['complexity'],
                    evolution_priority=region['complexity'],  # Higher complexity = higher priority
                    metadata={
                        "complexity_region": True,
                        "complexity_factors": region['factors']
                    }
                )
                components.append(component)
                
                if len(components) >= max_components:
                    break
        
        return components
    
    def _identify_sections(self, content: str) -> List[str]:
        """Identify major sections in content"""
        # Simple implementation - split by double newlines and headers
        sections = []
        
        # Split by headers (lines starting with #, ##, etc.)
        header_pattern = r'^#+\s+.*$'
        lines = content.split('\n')
        current_section = []
        
        for line in lines:
            if re.match(header_pattern, line) and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        # If no headers found, split by double newlines
        if len(sections) <= 1:
            sections = [s.strip() for s in content.split('\n\n') if s.strip()]
        
        return sections
    
    def _identify_functional_units(self, content: str) -> List[Dict[str, Any]]:
        """Identify functional units like functions, classes, etc."""
        units = []
        
        # Pattern for function definitions
        func_pattern = r'def\s+(\w+)\s*\([^)]*\):'
        class_pattern = r'class\s+(\w+).*?:'
        
        # Find functions
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            start = match.start()
            name = match.group(1)
            
            # Find the end of the function (simple heuristic)
            lines = content[start:].split('\n')
            func_lines = [lines[0]]
            indent_level = len(lines[0]) - len(lines[0].lstrip())
            
            for line in lines[1:]:
                if line.strip() and len(line) - len(line.lstrip()) <= indent_level and not line.startswith(' '):
                    break
                func_lines.append(line)
            
            units.append({
                'name': name,
                'type': 'function',
                'content': '\n'.join(func_lines),
                'dependencies': self._extract_dependencies('\n'.join(func_lines))
            })
        
        # Find classes
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            start = match.start()
            name = match.group(1)
            
            # Find the end of the class (simple heuristic)
            lines = content[start:].split('\n')
            class_lines = [lines[0]]
            indent_level = len(lines[0]) - len(lines[0].lstrip())
            
            for line in lines[1:]:
                if line.strip() and len(line) - len(line.lstrip()) <= indent_level and not line.startswith(' '):
                    break
                class_lines.append(line)
            
            units.append({
                'name': name,
                'type': 'class',
                'content': '\n'.join(class_lines),
                'dependencies': self._extract_dependencies('\n'.join(class_lines))
            })
        
        return units
    
    def _identify_semantic_chunks(self, content: str) -> List[Dict[str, Any]]:
        """Identify semantically coherent chunks"""
        chunks = []
        
        # Simple implementation - split by paragraphs and analyze topics
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            # Extract key topics (simple keyword extraction)
            words = re.findall(r'\b\w+\b', paragraph.lower())
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get most frequent words as topic
            if word_freq:
                topic = max(word_freq, key=word_freq.get)
            else:
                topic = f"Topic {i+1}"
            
            chunks.append({
                'topic': topic,
                'content': paragraph,
                'coherence': len(set(words)) / len(words) if words else 0,
                'complexity_score': min(len(paragraph) / 1000, 1.0)
            })
        
        return chunks
    
    def _identify_structural_elements(self, content: str) -> List[Dict[str, Any]]:
        """Identify structural elements in content"""
        elements = []
        
        # Look for various structural patterns
        patterns = {
            'header': r'^#+\s+(.+)$',
            'list_item': r'^\s*[-*+]\s+(.+)$',
            'numbered_item': r'^\s*\d+\.\s+(.+)$',
            'code_block': r'```[\s\S]*?```',
            'quote': r'^>\s+(.+)$'
        }
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            for element_type, pattern in patterns.items():
                match = re.match(pattern, line)
                if match:
                    elements.append({
                        'type': element_type,
                        'title': match.group(1) if match.groups() else f"{element_type} {i+1}",
                        'content': line,
                        'level': len(line) - len(line.lstrip())
                    })
        
        return elements
    
    def _identify_dependency_clusters(self, content: str) -> List[Dict[str, Any]]:
        """Identify clusters based on dependencies"""
        clusters = []
        
        # Simple implementation - group by import statements and function calls
        lines = content.split('\n')
        current_cluster = []
        cluster_deps = set()
        
        for line in lines:
            # Check for imports
            if 'import' in line or 'from' in line:
                cluster_deps.add(line.strip())
            
            current_cluster.append(line)
            
            # End cluster on empty line or new import
            if not line.strip() and current_cluster:
                if len(current_cluster) > 1:
                    clusters.append({
                        'name': f"Cluster {len(clusters) + 1}",
                        'content': '\n'.join(current_cluster),
                        'dependencies': list(cluster_deps),
                        'is_core': len(cluster_deps) > 2,
                        'size': len(current_cluster),
                        'complexity': min(len(current_cluster) / 50, 1.0)
                    })
                current_cluster = []
                cluster_deps = set()
        
        # Add final cluster
        if current_cluster:
            clusters.append({
                'name': f"Cluster {len(clusters) + 1}",
                'content': '\n'.join(current_cluster),
                'dependencies': list(cluster_deps),
                'is_core': len(cluster_deps) > 2,
                'size': len(current_cluster),
                'complexity': min(len(current_cluster) / 50, 1.0)
            })
        
        return clusters
    
    def _identify_complexity_regions(self, content: str) -> List[Dict[str, Any]]:
        """Identify regions of high complexity"""
        regions = []
        
        # Split content into chunks and analyze complexity
        chunk_size = 200  # characters
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            complexity_factors = []
            complexity_score = 0.0
            
            # Analyze various complexity factors
            # Nesting depth
            nesting = max(line.count('    ') for line in chunk.split('\n') if line.strip())
            if nesting > 3:
                complexity_factors.append('deep_nesting')
                complexity_score += 0.3
            
            # Number of conditions
            conditions = len(re.findall(r'\b(if|elif|while|for)\b', chunk))
            if conditions > 5:
                complexity_factors.append('many_conditions')
                complexity_score += 0.2
            
            # Long lines
            long_lines = sum(1 for line in chunk.split('\n') if len(line) > 100)
            if long_lines > 3:
                complexity_factors.append('long_lines')
                complexity_score += 0.2
            
            # Complex expressions
            complex_expr = len(re.findall(r'[(){}[\]]', chunk))
            if complex_expr > 20:
                complexity_factors.append('complex_expressions')
                complexity_score += 0.3
            
            if complexity_score > 0.5:  # High complexity threshold
                regions.append({
                    'content': chunk,
                    'complexity': min(complexity_score, 1.0),
                    'factors': complexity_factors
                })
        
        return regions
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from content"""
        dependencies = []
        
        # Look for import statements
        import_pattern = r'(?:from\s+(\w+)|import\s+(\w+))'
        for match in re.finditer(import_pattern, content):
            dep = match.group(1) or match.group(2)
            if dep:
                dependencies.append(dep)
        
        # Look for function calls
        call_pattern = r'(\w+)\s*\('
        for match in re.finditer(call_pattern, content):
            func_name = match.group(1)
            if func_name not in ['print', 'len', 'str', 'int', 'float']:  # Skip built-ins
                dependencies.append(func_name)
        
        return list(set(dependencies))
    
    def _classify_component_type(self, content: str) -> ComponentType:
        """Classify the type of a component based on its content"""
        content_lower = content.lower()
        
        if 'def ' in content or 'class ' in content:
            return ComponentType.CORE_LOGIC
        elif 'import ' in content or 'from ' in content:
            return ComponentType.SUPPORTING_FUNCTION
        elif 'config' in content_lower or 'setting' in content_lower:
            return ComponentType.CONFIGURATION
        elif 'test' in content_lower or 'assert' in content_lower:
            return ComponentType.TEST_CASE
        elif 'try:' in content or 'except' in content:
            return ComponentType.ERROR_HANDLING
        elif content.startswith('#') or '"""' in content:
            return ComponentType.DOCUMENTATION
        elif '{' in content or '[' in content:
            return ComponentType.DATA_STRUCTURE
        else:
            return ComponentType.INTERFACE
    
    def _map_structure_type_to_component_type(self, structure_type: str) -> ComponentType:
        """Map structural type to component type"""
        mapping = {
            'header': ComponentType.DOCUMENTATION,
            'list_item': ComponentType.DATA_STRUCTURE,
            'numbered_item': ComponentType.DATA_STRUCTURE,
            'code_block': ComponentType.CORE_LOGIC,
            'quote': ComponentType.DOCUMENTATION
        }
        return mapping.get(structure_type, ComponentType.INTERFACE)
    
    def _calculate_component_complexity(self, content: str) -> float:
        """Calculate complexity score for a component"""
        score = 0.0
        
        # Length factor
        score += min(len(content) / 1000, 0.3)
        
        # Nesting factor
        max_nesting = max((len(line) - len(line.lstrip())) // 4 
                         for line in content.split('\n') if line.strip())
        score += min(max_nesting / 10, 0.3)
        
        # Complexity keywords
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
        keyword_count = sum(content.lower().count(keyword) for keyword in complexity_keywords)
        score += min(keyword_count / 20, 0.4)
        
        return min(score, 1.0)
    
    def _build_dependency_graph(self, components: List[Component]) -> Dict[str, List[str]]:
        """Build dependency graph between components"""
        graph = {}
        
        for component in components:
            graph[component.id] = []
            
            # Check dependencies against other components
            for other in components:
                if other.id != component.id:
                    # Simple heuristic - check if component mentions other component's title
                    if other.title.lower() in component.content.lower():
                        graph[component.id].append(other.id)
                    
                    # Check explicit dependencies
                    for dep in component.dependencies:
                        if dep.lower() in other.title.lower() or dep.lower() in other.content.lower():
                            graph[component.id].append(other.id)
        
        return graph
    
    def test_method_after_dependency_graph(self):
        """Test method to see if this position works"""
        return "working"
    
    def _calculate_decomposition_quality(self, original_content: str, components: List[Component]) -> float:
        """Calculate quality score for decomposition"""
        if not components:
            return 0.0
        
        score = 0.0
        
        # Coverage - how much of original content is covered
        total_component_length = sum(len(c.content) for c in components)
        coverage = min(total_component_length / len(original_content), 1.0)
        score += coverage * 0.4
        
        # Balance - components should be reasonably sized
        lengths = [len(c.content) for c in components]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        balance = 1.0 / (1.0 + variance / (avg_length ** 2))
        score += balance * 0.3
        
        # Coherence - components should be internally coherent
        coherence = sum(self._calculate_component_coherence(c) for c in components) / len(components)
        score += coherence * 0.3
        
        return min(score, 1.0)
    
    def _calculate_component_coherence(self, component: Component) -> float:
        """Calculate internal coherence of a component"""
        # Simple heuristic based on repeated words and concepts
        words = re.findall(r'\b\w+\b', component.content.lower())
        if not words:
            return 0.0
        
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate coherence based on word repetition
        if not word_freq:
            return 0.0
        
        max_freq = max(word_freq.values())
        coherence = max_freq / len(words)
        return min(coherence * 5, 1.0)  # Scale up and cap at 1.0
    
    def _analyze_complexity_distribution(self, components: List[Component]) -> Dict[str, Any]:
        """Analyze complexity distribution across components"""
        if not components:
            return {}
        
        complexities = [c.complexity_score for c in components]
        return {
            'min': min(complexities),
            'max': max(complexities),
            'avg': sum(complexities) / len(complexities),
            'high_complexity_count': sum(1 for c in complexities if c > 0.7)
        }
    
    def _create_reassembly_instructions(
        self, components: List[Component], dependency_graph: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Create instructions for reassembling components"""
        return {
            'assembly_order': self._calculate_assembly_order(components, dependency_graph),
            'merge_strategies': self._suggest_merge_strategies(components),
            'validation_checks': self._create_validation_checks(components)
        }
    
    def _calculate_assembly_order(
        self, components: List[Component], dependency_graph: Dict[str, List[str]]
    ) -> List[str]:
        """Calculate optimal order for assembling components"""
        # Topological sort based on dependencies
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(component_id: str):
            if component_id in temp_visited:
                return  # Cycle detected, skip
            if component_id in visited:
                return
            
            temp_visited.add(component_id)
            for dep in dependency_graph.get(component_id, []):
                visit(dep)
            temp_visited.remove(component_id)
            visited.add(component_id)
            order.append(component_id)
        
        for component in components:
            if component.id not in visited:
                visit(component.id)
        
        return order
    
    def _suggest_merge_strategies(self, components: List[Component]) -> Dict[str, str]:
        """Suggest strategies for merging components"""
        strategies = {}
        
        for component in components:
            if component.component_type == ComponentType.CORE_LOGIC:
                strategies[component.id] = "preserve_structure"
            elif component.component_type == ComponentType.DOCUMENTATION:
                strategies[component.id] = "merge_similar"
            else:
                strategies[component.id] = "standard_merge"
        
        return strategies
    
    def _create_validation_checks(self, components: List[Component]) -> List[str]:
        """Create validation checks for reassembly"""
        checks = [
            "verify_all_components_included",
            "check_dependency_satisfaction",
            "validate_syntax_correctness",
            "ensure_functionality_preservation"
        ]
        
        # Add component-specific checks
        for component in components:
            if component.component_type == ComponentType.TEST_CASE:
                checks.append(f"run_tests_for_{component.id}")
            elif component.component_type == ComponentType.CONFIGURATION:
                checks.append(f"validate_config_{component.id}")
        
        return checks
    
    def reassemble_components(
        self, 
        components: List[Component], 
        reassembly_instructions: Dict[str, Any],
        **kwargs
    ) -> ReassemblyResult:
        """
        Reassemble components back into coherent content.
        
        Args:
            components: List of components to reassemble
            reassembly_instructions: Instructions for reassembly
            **kwargs: Additional parameters
            
        Returns:
            ReassemblyResult: Result of reassembly process
        """
        start_time = time.time()
        
        # Follow assembly order
        assembly_order = reassembly_instructions.get('assembly_order', [c.id for c in components])
        merge_strategies = reassembly_instructions.get('merge_strategies', {})
        
        # Reassemble content
        reassembled_parts = []
        components_used = []
        
        for component_id in assembly_order:
            component = next((c for c in components if c.id == component_id), None)
            if component:
                strategy = merge_strategies.get(component_id, 'standard_merge')
                
                if strategy == "preserve_structure":
                    reassembled_parts.append(component.content)
                elif strategy == "merge_similar":
                    # Simple merge - could be enhanced
                    reassembled_parts.append(component.content)
                else:  # standard_merge
                    reassembled_parts.append(component.content)
                
                components_used.append(component_id)
        
        # Join parts
        reassembled_content = '\n\n'.join(reassembled_parts)
        
        # Calculate quality metrics
        quality_score = self._calculate_reassembly_quality(
            reassembled_content, components, reassembly_instructions
        )
        
        # Calculate improvement metrics
        improvement_metrics = self._calculate_improvement_metrics(
            reassembled_content, components
        )
        
        result = ReassemblyResult(
            reassembled_content=reassembled_content,
            components_used=components_used,
            quality_score=quality_score,
            improvement_metrics=improvement_metrics,
            metadata={
                "reassembly_time": time.time() - start_time,
                "components_count": len(components_used),
                "assembly_strategy": "ordered_merge"
            }
        )
        
        return result
    
    def _calculate_reassembly_quality(
        self, 
        reassembled_content: str, 
        components: List[Component], 
        instructions: Dict[str, Any]
    ) -> float:
        """Calculate quality of reassembled content"""
        score = 0.0
        
        # Completeness - all components should be included
        total_component_content = sum(len(c.content) for c in components)
        completeness = min(len(reassembled_content) / total_component_content, 1.0)
        score += completeness * 0.4
        
        # Coherence - content should flow well
        coherence = self._calculate_content_coherence(reassembled_content)
        score += coherence * 0.3
        
        # Structure preservation
        structure_score = self._calculate_structure_preservation(reassembled_content, components)
        score += structure_score * 0.3
        
        return min(score, 1.0)
    
    def _calculate_content_coherence(self, content: str) -> float:
        """Calculate coherence of content"""
        # Simple heuristic - check for smooth transitions
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            return 1.0
        
        coherence_score = 0.0
        for i in range(len(paragraphs) - 1):
            # Check for common words between adjacent paragraphs
            words1 = set(re.findall(r'\b\w+\b', paragraphs[i].lower()))
            words2 = set(re.findall(r'\b\w+\b', paragraphs[i + 1].lower()))
            
            if words1 and words2:
                overlap = len(words1.intersection(words2))
                coherence_score += overlap / max(len(words1), len(words2))
        
        return coherence_score / (len(paragraphs) - 1)
    
    def _calculate_structure_preservation(
        self, reassembled_content: str, components: List[Component]
    ) -> float:
        """Calculate how well structure is preserved"""
        # Simple heuristic - check if component boundaries are maintained
        score = 0.0
        
        for component in components:
            if component.content.strip() in reassembled_content:
                score += 1.0
        
        return score / len(components) if components else 0.0
    
    def _calculate_improvement_metrics(
        self, reassembled_content: str, components: List[Component]
    ) -> Dict[str, Any]:
        """Calculate improvement metrics"""
        original_length = sum(len(c.content) for c in components)
        
        return {
            "length_change": len(reassembled_content) - original_length,
            "length_ratio": len(reassembled_content) / original_length if original_length > 0 else 1.0,
            "component_integration": len(components),
            "structure_improvements": self._count_structure_improvements(reassembled_content, components)
        }
    
    def _count_structure_improvements(
        self, reassembled_content: str, components: List[Component]
    ) -> int:
        """Count structural improvements made during reassembly"""
        improvements = 0
        
        # Count transitions added
        transitions = len(re.findall(r'\n\n', reassembled_content))
        improvements += transitions
        
        # Count formatting improvements
        if reassembled_content.count('\n') > sum(c.content.count('\n') for c in components):
            improvements += 1
        
        return improvements
    
    def get_decomposition_history(self) -> List[DecompositionResult]:
        """Get history of decomposition results"""
        return self.decomposition_history.copy()
    
    def reassemble_components(
        self, 
        components: List[Component], 
        reassembly_instructions: Dict[str, Any],
        **kwargs
    ) -> ReassemblyResult:
        """
        Reassemble components back into coherent content.
        
        Args:
            components: List of components to reassemble
            reassembly_instructions: Instructions for reassembly
            **kwargs: Additional parameters
            
        Returns:
            ReassemblyResult: Result of reassembly process
        """
        start_time = time.time()
        
        # Follow assembly order
        assembly_order = reassembly_instructions.get('assembly_order', [c.id for c in components])
        merge_strategies = reassembly_instructions.get('merge_strategies', {})
        
        # Reassemble content
        reassembled_parts = []
        components_used = []
        
        for component_id in assembly_order:
            component = next((c for c in components if c.id == component_id), None)
            if component:
                strategy = merge_strategies.get(component_id, 'standard_merge')
                
                if strategy == "preserve_structure":
                    reassembled_parts.append(component.content)
                elif strategy == "merge_similar":
                    reassembled_parts.append(component.content)
                else:  # standard_merge
                    reassembled_parts.append(component.content)
                
                components_used.append(component_id)
        
        # Join parts
        reassembled_content = '\n\n'.join(reassembled_parts)
        
        # Calculate basic quality score
        quality_score = 0.8 if components_used else 0.0
        
        # Calculate improvement metrics
        original_length = sum(len(c.content) for c in components)
        improvement_metrics = {
            "length_change": len(reassembled_content) - original_length,
            "length_ratio": len(reassembled_content) / original_length if original_length > 0 else 1.0,
            "component_integration": len(components),
            "structure_improvements": 1
        }
        
        result = ReassemblyResult(
            reassembled_content=reassembled_content,
            components_used=components_used,
            quality_score=quality_score,
            improvement_metrics=improvement_metrics,
            metadata={
                "reassembly_time": time.time() - start_time,
                "components_count": len(components_used),
                "assembly_strategy": "ordered_merge"
            }
        )
        
        return result
    
    def clear_history(self):
        """Clear decomposition history"""
        self.decomposition_history.clear()


# Additional utility functions for advanced decomposition features

def analyze_content_patterns(content: str) -> Dict[str, Any]:
    """Analyze patterns in content for better decomposition"""
    patterns = {
        'code_blocks': len(re.findall(r'```[\s\S]*?```', content)),
        'headers': len(re.findall(r'^#+\s+', content, re.MULTILINE)),
        'lists': len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE)),
        'functions': len(re.findall(r'def\s+\w+\s*\(', content)),
        'classes': len(re.findall(r'class\s+\w+', content)),
        'imports': len(re.findall(r'^(?:from|import)\s+', content, re.MULTILINE))
    }
    
    return patterns


def suggest_optimal_strategy(content: str) -> DecompositionStrategy:
    """Suggest optimal decomposition strategy based on content analysis"""
    patterns = analyze_content_patterns(content)
    
    # Rule-based strategy selection
    if patterns['functions'] > 3 or patterns['classes'] > 1:
        return DecompositionStrategy.FUNCTIONAL
    elif patterns['headers'] > 2:
        return DecompositionStrategy.HIERARCHICAL
    elif patterns['imports'] > 5:
        return DecompositionStrategy.DEPENDENCY_BASED
    elif len(content) > 2000:
        return DecompositionStrategy.COMPLEXITY_BASED
    else:
        return DecompositionStrategy.SEMANTIC


def create_decomposition_report(result: DecompositionResult) -> str:
    """Create a comprehensive report of decomposition results"""
    report = f"""
# Decomposition Report

## Overview
- **Strategy Used:** {result.decomposition_strategy.value}
- **Components Created:** {len(result.components)}
- **Quality Score:** {result.quality_score:.2f}
- **Processing Time:** {result.metadata.get('decomposition_time', 0):.2f}s

## Components Summary
"""
    
    for i, component in enumerate(result.components, 1):
        report += f"""
### Component {i}: {component.title}
- **Type:** {component.component_type.value}
- **Size:** {len(component.content)} characters
- **Complexity:** {component.complexity_score:.2f}
- **Dependencies:** {len(component.dependencies)}
"""
    
    report += f"""
## Dependency Graph
{json.dumps(result.dependency_graph, indent=2)}

## Quality Metrics
- **Coverage:** {result.metadata.get('avg_component_size', 0):.0f} avg chars per component
- **Complexity Distribution:** {result.metadata.get('complexity_distribution', {})}
"""
    
    return report


# Stub functions for backward compatibility

def get_recommended_strategy(problem):
    """Stub function for getting recommended strategy."""
    return None


def get_roma_integration_status():
    """Stub function for getting ROMA integration status."""
    return {'status': 'unknown'}
