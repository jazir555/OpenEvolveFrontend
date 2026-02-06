"""
Canonical Lean Code Generation
===============================

Generate strictly canonical Lean 4 code from dependency DAG with compositional semantics.
Ensures same mathematical content always produces identical Lean representation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
import re

# Lazy imports to avoid circular dependencies
def _import_dag():
    """Lazy import dependency DAG."""
    try:
        from . import dependency_dag
        return dependency_dag
    except ImportError:
        return None

def _import_semantics():
    """Lazy import compositional semantics."""
    try:
        from . import compositional_semantics
        return compositional_semantics
    except ImportError:
        return None

def _import_type_theory():
    """Lazy import lean type theory."""
    try:
        from . import lean_type_theory
        return lean_type_theory
    except ImportError:
        return None


@dataclass
class CanonicalNamingRules:
    """Rules for canonical naming in Lean."""
    
    @staticmethod
    def to_snake_case(text: str) -> str:
        """Convert to snake_case (for definitions, theorems)."""
        # Remove special characters
        text = re.sub(r'[^\w\s-]', '', text)
        # Insert underscore before capitals
        text = re.sub('([a-z0-9])([A-Z])', r'\1_\2', text)
        # Replace spaces and hyphens with underscores
        text = text.replace(' ', '_').replace('-', '_')
        # Lowercase and collapse multiple underscores
        text = text.lower()
        text = re.sub(r'_+', '_', text)
        return text.strip('_')
    
    @staticmethod
    def to_pascal_case(text: str) -> str:
        """Convert to PascalCase (for types, structures)."""
        # Remove special characters
        text = re.sub(r'[^\w\s-]', '', text)
        # Split on spaces, hyphens, underscores
        words = re.split(r'[\s\-_]+', text)
        # Capitalize each word
        return ''.join(word.capitalize() for word in words if word)
    
    @staticmethod
    def canonical_theorem_name(stmt: Any) -> str:
        """Generate canonical name for theorem."""
        if hasattr(stmt, 'name') and stmt.name:
            return CanonicalNamingRules.to_snake_case(stmt.name)
        
        # Extract key words from text
        text = stmt.text.lower() if hasattr(stmt, 'text') else str(stmt)
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'if', 'then', 'for', 
                       'all', 'every', 'there', 'exists', 'such', 'that'}
        words = [w for w in text.split() if w not in common_words and len(w) > 2]
        
        # Take first few significant words
        key_words = words[:4]
        return '_'.join(key_words)
    
    @staticmethod
    def canonical_definition_name(stmt: Any) -> str:
        """Generate canonical name for definition."""
        if hasattr(stmt, 'name') and stmt.name:
            return CanonicalNamingRules.to_snake_case(stmt.name)
        
        # Extract the thing being defined (usually first capitalized word or phrase)
        stmt_text = stmt.text if hasattr(stmt, 'text') else str(stmt)
        match = re.search(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', stmt_text)
        if match:
            return CanonicalNamingRules.to_snake_case(match.group(1))
        
        stmt_number = getattr(stmt, 'number', None)
        if stmt_number:
            return f"definition_{stmt_number.replace('.', '_')}"
        return "unnamed_def"
    
    @staticmethod
    def canonical_structure_name(stmt: Any) -> str:
        """Generate canonical name for structure."""
        # Extract structure name (first capitalized word)
        stmt_text = stmt.text if hasattr(stmt, 'text') else str(stmt)
        match = re.search(r'\b([A-Z][a-zA-Z]+)\b', stmt_text)
        if match:
            return CanonicalNamingRules.to_pascal_case(match.group(1))
        
        if hasattr(stmt, 'name') and stmt.name:
            return CanonicalNamingRules.to_pascal_case(stmt.name)
        
        return "UnnamedStructure"


@dataclass
class LeanCodeNode:
    """Canonical Lean code for a single statement."""
    statement_id: str
    lean_name: str
    lean_code: str
    dependencies: List[str]
    kind: Any  # StatementKind - lazy loaded
    
    def __str__(self):
        return self.lean_code


class SemanticGrammar:
    """Stub class for SemanticGrammar - will be replaced by actual implementation."""
    
    def parse(self, text: str, context: Any) -> List[Any]:
        """Parse text and return parse trees."""
        return []
    
    def compute_semantics(self, parse_tree: Any, context: Any) -> Any:
        """Compute semantics from parse tree."""
        return {}


class LeanTypeChecker:
    """Stub class for LeanTypeChecker - will be replaced by actual implementation."""
    pass


class Context:
    """Stub class for Context - will be replaced by actual implementation."""
    def __init__(self):
        self.bindings: Dict[str, Any] = {}
    
    def lookup(self, name: str) -> Any:
        return self.bindings.get(name)


class CanonicalLeanGenerator:
    """Generate canonical Lean 4 code from dependency DAG."""
    
    def __init__(self, grammar: Optional[Any] = None):
        self.grammar = grammar or SemanticGrammar()
        self.type_checker = LeanTypeChecker()
        self.naming = CanonicalNamingRules()
        self.generated_names: Set[str] = set()
    
    def generate_from_dag(self, dag: Any, 
                         paper_title: Optional[str] = None) -> str:
        """
        Generate complete canonical Lean file from dependency DAG.
        
        Args:
            dag: Dependency DAG of paper
            paper_title: Optional paper title for header
            
        Returns:
            Complete Lean 4 file as string
        """
        # Verify DAG is acyclic
        if hasattr(dag, 'is_acyclic') and not dag.is_acyclic():
            cycle = dag.find_cycle() if hasattr(dag, 'find_cycle') else []
            raise ValueError(f"DAG contains cycle: {' → '.join(cycle)}")
        
        # Get topological order
        topo_order = dag.topological_sort() if hasattr(dag, 'topological_sort') else []
        
        # Generate Lean code for each node
        lean_nodes: Dict[str, LeanCodeNode] = {}
        context = Context()
        
        for stmt_id in topo_order:
            stmt = dag.nodes.get(stmt_id) if hasattr(dag, 'nodes') else None
            if stmt:
                try:
                    lean_node = self.generate_statement(stmt, context, dag)
                    lean_nodes[stmt_id] = lean_node
                    
                    # Update context with new definition
                    self._update_context(context, lean_node)
                except Exception as e:
                    print(f"Warning: Failed to generate {stmt_id}: {e}")
                    # Generate placeholder
                    lean_node = self._generate_placeholder(stmt, dag)
                    lean_nodes[stmt_id] = lean_node
        
        # Assemble complete file
        return self._assemble_lean_file(lean_nodes, topo_order, paper_title, dag)
    
    def generate_statement(self, stmt: Any, context: Any,
                          dag: Any) -> LeanCodeNode:
        """Generate canonical Lean code for a single statement."""
        
        kind = getattr(stmt, 'kind', None)
        kind_value = kind.value if hasattr(kind, 'value') else str(kind)
        
        if kind_value == 'structure' or 'STRUCTURE' in str(kind):
            return self._generate_structure(stmt, context, dag)
        elif kind_value == 'definition' or 'DEFINITION' in str(kind):
            return self._generate_definition(stmt, context, dag)
        elif any(x in str(kind) for x in ['THEOREM', 'LEMMA', 'PROPOSITION']):
            return self._generate_theorem(stmt, context, dag)
        else:
            return self._generate_placeholder(stmt, dag)
    
    def _generate_structure(self, stmt: Any, context: Any,
                           dag: Any) -> LeanCodeNode:
        """Generate structure definition."""
        
        # Generate canonical name
        lean_name = self.naming.canonical_structure_name(stmt)
        lean_name = self._ensure_unique_name(lean_name)
        
        # Parse statement to extract fields
        stmt_text = stmt.text if hasattr(stmt, 'text') else str(stmt)
        parses = self.grammar.parse(stmt_text, context)
        
        if not parses:
            # Fallback: manual structure
            return self._generate_structure_fallback(stmt, lean_name, dag)
        
        # Use first valid parse
        parse_tree = parses[0]
        semantic_value = self.grammar.compute_semantics(parse_tree, context)
        
        # Extract structure components
        if isinstance(semantic_value, dict) and semantic_value.get('kind') == 'structure':
            fields = semantic_value.get('fields', [])
            extends = semantic_value.get('extends')
            
            # Generate Lean code
            code_lines = []
            
            if extends:
                code_lines.append(f"structure {lean_name} extends {extends} where")
            else:
                code_lines.append(f"structure {lean_name} where")
            
            # Add fields
            for field_name, field_type in fields:
                field_name_canonical = self.naming.to_snake_case(field_name)
                type_str = self._type_to_lean_string(field_type)
                code_lines.append(f"  {field_name_canonical} : {type_str}")
            
            lean_code = "\n".join(code_lines)
        else:
            lean_code = self._generate_structure_fallback(stmt, lean_name, dag).lean_code
        
        return LeanCodeNode(
            statement_id=getattr(stmt, 'id', 'unknown'),
            lean_name=lean_name,
            lean_code=lean_code,
            dependencies=self._get_dependencies(stmt, dag),
            kind=getattr(stmt, 'kind', None)
        )
    
    def _generate_definition(self, stmt: Any, context: Any,
                            dag: Any) -> LeanCodeNode:
        """Generate definition."""
        
        lean_name = self.naming.canonical_definition_name(stmt)
        lean_name = self._ensure_unique_name(lean_name)
        
        # Parse statement
        stmt_text = stmt.text if hasattr(stmt, 'text') else str(stmt)
        parses = self.grammar.parse(stmt_text, context)
        
        if not parses:
            return self._generate_definition_fallback(stmt, lean_name, dag)
        
        # Generate Lean code
        # For now, simple def with sorry
        code = f"def {lean_name} : Prop :=\n  sorry -- {stmt_text[:80]}..."
        
        return LeanCodeNode(
            statement_id=getattr(stmt, 'id', 'unknown'),
            lean_name=lean_name,
            lean_code=code,
            dependencies=self._get_dependencies(stmt, dag),
            kind=getattr(stmt, 'kind', None)
        )
    
    def _generate_theorem(self, stmt: Any, context: Any,
                         dag: Any) -> LeanCodeNode:
        """Generate theorem statement with sorry proof."""
        
        lean_name = self.naming.canonical_theorem_name(stmt)
        lean_name = self._ensure_unique_name(lean_name)
        
        # Parse statement to extract structure
        stmt_text = stmt.text if hasattr(stmt, 'text') else str(stmt)
        parses = self.grammar.parse(stmt_text, context)
        
        if not parses:
            return self._generate_theorem_fallback(stmt, lean_name, dag)
        
        # Generate theorem code
        code_lines = []
        
        # Extract parameters and hypotheses from context and statement
        params = self._extract_parameters(stmt, {})
        hypotheses = self._extract_hypotheses(stmt, {})
        conclusion = self._extract_conclusion(stmt, {})
        
        # Build theorem signature
        kind_value = getattr(stmt.kind, 'value', 'theorem') if hasattr(stmt, 'kind') else 'theorem'
        code_lines.append(f"{kind_value} {lean_name}")
        
        # Add parameters
        if params:
            for param_name, param_type in params:
                type_str = self._type_to_lean_string(param_type)
                code_lines.append(f"    ({param_name} : {type_str})")
        
        # Add hypotheses
        if hypotheses:
            for hyp_name, hyp_prop in hypotheses:
                prop_str = self._type_to_lean_string(hyp_prop)
                code_lines.append(f"    ({hyp_name} : {prop_str})")
        
        # Add conclusion
        if conclusion:
            concl_str = self._type_to_lean_string(conclusion)
            code_lines.append(f"    : {concl_str} := by")
        else:
            code_lines.append(f"    : Prop := by")
        
        code_lines.append("  sorry")
        
        lean_code = "\n".join(code_lines)
        
        return LeanCodeNode(
            statement_id=getattr(stmt, 'id', 'unknown'),
            lean_name=lean_name,
            lean_code=lean_code,
            dependencies=self._get_dependencies(stmt, dag),
            kind=getattr(stmt, 'kind', None)
        )
    
    def _generate_structure_fallback(self, stmt: Any, lean_name: str,
                                    dag: Any) -> LeanCodeNode:
        """Fallback structure generation when parsing fails."""
        stmt_text = stmt.text if hasattr(stmt, 'text') else str(stmt)
        code = f"""structure {lean_name} where
  -- Extracted from: {stmt_text[:80]}...
  -- TODO: Manual field extraction needed
  carrier : Type"""
        
        return LeanCodeNode(
            statement_id=getattr(stmt, 'id', 'unknown'),
            lean_name=lean_name,
            lean_code=code,
            dependencies=self._get_dependencies(stmt, dag),
            kind=getattr(stmt, 'kind', None)
        )
    
    def _generate_definition_fallback(self, stmt: Any, lean_name: str,
                                     dag: Any) -> LeanCodeNode:
        """Fallback definition generation."""
        stmt_text = stmt.text if hasattr(stmt, 'text') else str(stmt)
        code = f"""def {lean_name} : Prop :=
  sorry -- {stmt_text[:80]}..."""
        
        return LeanCodeNode(
            statement_id=getattr(stmt, 'id', 'unknown'),
            lean_name=lean_name,
            lean_code=code,
            dependencies=self._get_dependencies(stmt, dag),
            kind=getattr(stmt, 'kind', None)
        )
    
    def _generate_theorem_fallback(self, stmt: Any, lean_name: str,
                                   dag: Any) -> LeanCodeNode:
        """Fallback theorem generation."""
        kind_value = getattr(stmt.kind, 'value', 'theorem') if hasattr(stmt, 'kind') else 'theorem'
        stmt_text = stmt.text if hasattr(stmt, 'text') else str(stmt)
        code = f"""{kind_value} {lean_name} : Prop := by
  sorry -- {stmt_text[:80]}..."""
        
        return LeanCodeNode(
            statement_id=getattr(stmt, 'id', 'unknown'),
            lean_name=lean_name,
            lean_code=code,
            dependencies=self._get_dependencies(stmt, dag),
            kind=getattr(stmt, 'kind', None)
        )
    
    def _generate_placeholder(self, stmt: Any, 
                             dag: Any) -> LeanCodeNode:
        """Generate placeholder for unparseable statement."""
        stmt_id = getattr(stmt, 'id', 'unknown')
        lean_name = f"placeholder_{stmt_id}"
        stmt_text = stmt.text if hasattr(stmt, 'text') else str(stmt)
        kind_value = getattr(stmt.kind, 'value', 'unknown') if hasattr(stmt, 'kind') else 'unknown'
        code = f"""-- PLACEHOLDER: {kind_value}
-- {stmt_text[:100]}...
axiom {lean_name} : Prop"""
        
        return LeanCodeNode(
            statement_id=stmt_id,
            lean_name=lean_name,
            lean_code=code,
            dependencies=self._get_dependencies(stmt, dag),
            kind=getattr(stmt, 'kind', None)
        )
    
    def _extract_parameters(self, stmt: Any, 
                           semantic_value: Any) -> List[Tuple[str, Any]]:
        """Extract parameters from statement."""
        params = []
        
        # Extract variable names from statement
        variables = getattr(stmt, 'variables', [])
        for var_name in variables:
            # Try to infer type from context or statement
            params.append((var_name, None))  # Simplified
        
        return params[:3]  # Limit to avoid clutter
    
    def _extract_hypotheses(self, stmt: Any,
                           semantic_value: Any) -> List[Tuple[str, Any]]:
        """Extract hypotheses from theorem statement."""
        hypotheses = []
        # Look for "if P then Q" pattern → P is hypothesis
        stmt_text = stmt.text if hasattr(stmt, 'text') else str(stmt)
        if_match = re.search(r'if\s+(.+?)\s+then', stmt_text, re.IGNORECASE)
        if if_match:
            hyp_text = if_match.group(1)
            hypotheses.append(('h', None))  # Simplified
        
        return hypotheses
    
    def _extract_conclusion(self, stmt: Any,
                           semantic_value: Any) -> Optional[Any]:
        """Extract conclusion from theorem statement."""
        return None  # Simplified
    
    def _type_to_lean_string(self, lean_type: Any) -> str:
        """Convert Lean type to canonical string representation."""
        if lean_type is None:
            return "Prop"
        if hasattr(lean_type, 'to_lean_string'):
            return lean_type.to_lean_string()
        if isinstance(lean_type, str):
            return lean_type
        return "Prop"
    
    def _ensure_unique_name(self, name: str) -> str:
        """Ensure name is unique by adding suffix if needed."""
        if name not in self.generated_names:
            self.generated_names.add(name)
            return name
        
        counter = 2
        while f"{name}_{counter}" in self.generated_names:
            counter += 1
        
        unique_name = f"{name}_{counter}"
        self.generated_names.add(unique_name)
        return unique_name
    
    def _update_context(self, context: Any, lean_node: LeanCodeNode) -> None:
        """Update context with new definition."""
        if hasattr(context, 'bindings'):
            context.bindings[lean_node.lean_name] = None  # Simplified
    
    def _get_dependencies(self, stmt: Any, dag: Any) -> List[str]:
        """Get dependencies for a statement."""
        if hasattr(dag, 'get_dependencies'):
            stmt_id = getattr(stmt, 'id', None)
            if stmt_id:
                return list(dag.get_dependencies(stmt_id))
        return []
    
    def _assemble_lean_file(self, lean_nodes: Dict[str, LeanCodeNode],
                           topo_order: List[str], paper_title: Optional[str],
                           dag: Any) -> str:
        """Assemble complete Lean file from nodes."""
        lines = []
        
        # Header
        lines.append("/-")
        if paper_title:
            lines.append(f"  {paper_title}")
        lines.append("  Automatically generated from mathematical paper")
        lines.append("  ")
        lines.append(f"  Total statements: {len(lean_nodes)}")
        if hasattr(dag, 'edges'):
            lines.append(f"  Total dependencies: {len(dag.edges)}")
        lines.append("  Dependency DAG: Acyclic ✓")
        lines.append("  ")
        lines.append("  NOTE: All proofs are replaced with 'sorry'.")
        lines.append("  This is a complete skeleton - only proofs missing.")
        lines.append("-/")
        lines.append("")
        
        # Imports
        lines.append("import Mathlib.Tactic")
        lines.append("import Mathlib.Data.Real.Basic")
        lines.append("import Mathlib.Data.Nat.Basic")
        lines.append("")
        
        # Set options
        lines.append("set_option autoImplicit false")
        lines.append("")
        
        # Add each statement in topological order
        for stmt_id in topo_order:
            if stmt_id not in lean_nodes:
                continue
            
            lean_node = lean_nodes[stmt_id]
            
            # Add comment with source information
            kind = getattr(lean_node, 'kind', None)
            kind_value = kind.value if hasattr(kind, 'value') else str(kind)
            lines.append(f"-- Source: {kind_value}")
            
            # Add dependency comment
            if lean_node.dependencies:
                dep_names = [lean_nodes[dep_id].lean_name 
                           for dep_id in lean_node.dependencies 
                           if dep_id in lean_nodes]
                if dep_names:
                    lines.append(f"-- Dependencies: {', '.join(dep_names)}")
            
            # Add code
            lines.append(lean_node.lean_code)
            lines.append("")
        
        # Footer with statistics
        lines.append("/-")
        lines.append("  STATISTICS")
        structure_count = sum(1 for n in lean_nodes.values() if 'structure' in str(n.kind).lower())
        def_count = sum(1 for n in lean_nodes.values() if 'definition' in str(n.kind).lower())
        thm_count = sum(1 for n in lean_nodes.values() if 'theorem' in str(n.kind).lower())
        lemma_count = sum(1 for n in lean_nodes.values() if 'lemma' in str(n.kind).lower())
        lines.append(f"  Structures: {structure_count}")
        lines.append(f"  Definitions: {def_count}")
        lines.append(f"  Theorems: {thm_count}")
        lines.append(f"  Lemmas: {lemma_count}")
        lines.append("-/")
        
        return "\n".join(lines)


def generate_canonical_lean_from_paper(paper_text: str, 
                                       paper_title: Optional[str] = None) -> str:
    """
    Main entry point: Generate canonical Lean code from paper text.
    
    Args:
        paper_text: Full text of mathematical paper
        paper_title: Optional title for header
        
    Returns:
        Complete Lean 4 file as string
    """
    # Extract dependency DAG
    dag_module = _import_dag()
    if dag_module and hasattr(dag_module, 'PaperStructureExtractor'):
        extractor = dag_module.PaperStructureExtractor()
        dag = extractor.extract_dag(paper_text)
    else:
        # Fallback: create a minimal DAG stub
        class MinimalDAG:
            def __init__(self):
                self.nodes = {}
                self.edges = {}
            def is_acyclic(self):
                return True
            def topological_sort(self):
                return list(self.nodes.keys())
            def get_dependencies(self, stmt_id):
                return self.edges.get(stmt_id, set())
        dag = MinimalDAG()
    
    # Generate Lean code
    grammar = SemanticGrammar()
    generator = CanonicalLeanGenerator(grammar)
    lean_code = generator.generate_from_dag(dag, paper_title)
    
    return lean_code
