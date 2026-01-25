"""
Γ₁ CSP Data Models

Defines the core data structures for representing Constraint Satisfaction Problems
in the ACI system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any, Optional
import networkx as nx
from pathlib import Path


@dataclass
class Variable:
    """
    CSP Variable

    Attributes:
        name: Unique identifier for the variable
        domain: List of possible values
        metadata: Optional metadata about the variable
    """
    name: str
    domain: List[Any]
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate variable after initialization"""
        if not self.name or not self.name.strip():
            raise ValueError("Variable must have a non-empty name")
        if not self.domain:
            raise ValueError(f"Variable {self.name} must have a non-empty domain")

    def __hash__(self):
        """Make variable hashable for use in sets"""
        return hash(self.name)

    def __eq__(self, other):
        """Variable equality based on name"""
        if not isinstance(other, Variable):
            return False
        return self.name == other.name

    def domain_size(self) -> int:
        """Get domain size"""
        return len(self.domain)


@dataclass
class Constraint:
    """
    CSP Constraint

    Attributes:
        variables: List of variable names involved in this constraint
        allowed_tuples: Set of allowed value tuples
        tightness: Optional precomputed tightness (fraction of forbidden tuples)
        metadata: Optional metadata about the constraint
    """
    variables: List[str]
    allowed_tuples: Set[Tuple] = field(default_factory=set)
    tightness: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate constraint after initialization"""
        if not self.variables:
            raise ValueError("Constraint must involve at least one variable")
        if len(self.variables) != len(set(self.variables)):
            raise ValueError("Constraint cannot have duplicate variables")

    def __hash__(self):
        """Make constraint hashable for use in sets"""
        return hash(tuple(self.variables))

    def __eq__(self, other):
        """Constraint equality based on variables"""
        if not isinstance(other, Constraint):
            return False
        return self.variables == other.variables

    def arity(self) -> int:
        """Get constraint arity (number of variables)"""
        return len(self.variables)


@dataclass
class CSPInstance:
    """
    Complete CSP Instance

    Attributes:
        variables: List of variables in the CSP
        constraints: List of constraints in the CSP
        metadata: Optional metadata about the CSP
        constraint_graph: NetworkX graph of variable constraints (auto-built)
        variable_map: Dictionary mapping variable names to Variable objects (auto-built)
    """
    variables: List[Variable]
    constraints: List[Constraint]
    metadata: Dict = field(default_factory=dict)
    constraint_graph: nx.Graph = field(init=False)
    variable_map: Dict[str, Variable] = field(init=False)

    def __post_init__(self):
        """Build constraint graph and variable map after initialization"""
        self.constraint_graph = self._build_constraint_graph()
        self.variable_map = {v.name: v for v in self.variables}

    def _build_constraint_graph(self) -> nx.Graph:
        """
        Build NetworkX graph from constraints

        Creates a graph where:
        - Nodes are variables
        - Edges connect variables that share a constraint
        """
        G = nx.Graph()

        # Add all variables as nodes
        for var in self.variables:
            G.add_node(var.name, domain_size=len(var.domain))

        # Add edges for constraints
        for constraint in self.constraints:
            vars_in_constraint = constraint.variables
            # Connect all pairs in constraint (for n-ary constraints)
            for i in range(len(vars_in_constraint)):
                for j in range(i+1, len(vars_in_constraint)):
                    G.add_edge(
                        vars_in_constraint[i],
                        vars_in_constraint[j],
                        constraint=constraint
                    )

        return G

    def get_variable(self, name: str) -> Optional[Variable]:
        """Get variable by name"""
        return self.variable_map.get(name)

    def get_constraints_involving(self, var_name: str) -> List[Constraint]:
        """Get all constraints that involve a variable"""
        return [
            c for c in self.constraints
            if var_name in c.variables
        ]

    def num_variables(self) -> int:
        """Get number of variables"""
        return len(self.variables)

    def num_constraints(self) -> int:
        """Get number of constraints"""
        return len(self.constraints)

    def avg_domain_size(self) -> float:
        """Calculate average domain size"""
        if not self.variables:
            return 0.0
        return sum(v.domain_size() for v in self.variables) / len(self.variables)

    def constraint_density(self) -> float:
        """
        Calculate constraint density

        Returns fraction of possible binary constraints that exist
        """
        n = self.num_variables()
        if n < 2:
            return 0.0

        max_binary_constraints = n * (n - 1) / 2
        return min(1.0, self.num_constraints() / max_binary_constraints)

    def is_connected(self) -> bool:
        """Check if constraint graph is connected"""
        if self.constraint_graph.number_of_nodes() == 0:
            return True  # Empty graph is trivially connected
        return nx.is_connected(self.constraint_graph)

    def num_connected_components(self) -> int:
        """Get number of connected components in constraint graph"""
        if self.constraint_graph.number_of_nodes() == 0:
            return 0
        return nx.number_connected_components(self.constraint_graph)

    def tree_width_approximation(self) -> int:
        """
        Approximate tree width using minimum degree heuristic

        Returns approximation of graph tree width (lower = easier to solve)
        """
        if self.num_variables() == 0:
            return 0

        H = self.constraint_graph.copy()
        max_degree = 0

        while H.number_of_nodes() > 0:
            degrees = dict(H.degree())
            min_node = min(degrees, key=degrees.get)
            max_degree = max(max_degree, degrees[min_node])
            H.remove_node(min_node)

        return max_degree

    def __str__(self) -> str:
        """String representation"""
        return (f"CSP(n_vars={self.num_variables()}, "
                f"n_constraints={self.num_constraints()}, "
                f"avg_domain={self.avg_domain_size():.1f})")


# ============================================================================
# CSP Factory Functions
# ============================================================================

def create_test_csp(
    n_variables: int = 5,
    domain_size: int = 3,
    n_constraints: int = 4,
    constraint_tightness: float = 0.5
) -> CSPInstance:
    """
    Create a random CSP instance for testing

    Args:
        n_variables: Number of variables
        domain_size: Size of each variable's domain
        n_constraints: Number of constraints
        constraint_tightness: Fraction of forbidden tuples (0-1)

    Returns:
        CSPInstance for testing
    """
    import itertools
    import random

    # Create variables
    variables = [
        Variable(name=f"v{i}", domain=list(range(domain_size)))
        for i in range(n_variables)
    ]

    # Create random constraints
    constraints = []
    var_names = [v.name for v in variables]

    for _ in range(n_constraints):
        # Select 2 random variables for binary constraint
        vars_in_constraint = random.sample(var_names, 2)

        # Generate all possible tuples
        all_tuples = list(itertools.product(range(domain_size), repeat=2))

        # Select allowed tuples based on tightness
        n_allowed = int(len(all_tuples) * (1.0 - constraint_tightness))
        allowed_tuples = set(random.sample(all_tuples, n_allowed))

        constraint = Constraint(
            variables=vars_in_constraint,
            allowed_tuples=allowed_tuples,
            tightness=constraint_tightness
        )
        constraints.append(constraint)

    return CSPInstance(
        variables=variables,
        constraints=constraints,
        metadata={'test': True}
    )


def create_tree_csp(
    n_variables: int = 10,
    domain_size: int = 3,
    constraint_tightness: float = 0.3
) -> CSPInstance:
    """
    Create a tree-structured CSP (highly tractable)

    Args:
        n_variables: Number of variables
        domain_size: Size of each variable's domain
        constraint_tightness: Fraction of forbidden tuples

    Returns:
        Tree-structured CSPInstance
    """
    import itertools
    import random

    # Create variables
    variables = [
        Variable(name=f"v{i}", domain=list(range(domain_size)))
        for i in range(n_variables)
    ]

    # Create tree constraints (n-1 edges)
    constraints = []
    var_names = [v.name for v in variables]

    for i in range(n_variables - 1):
        vars_in_constraint = [var_names[i], var_names[i+1]]

        # Generate allowed tuples
        all_tuples = list(itertools.product(range(domain_size), repeat=2))
        n_allowed = int(len(all_tuples) * (1.0 - constraint_tightness))
        allowed_tuples = set(random.sample(all_tuples, n_allowed))

        constraint = Constraint(
            variables=vars_in_constraint,
            allowed_tuples=allowed_tuples,
            tightness=constraint_tightness
        )
        constraints.append(constraint)

    return CSPInstance(
        variables=variables,
        constraints=constraints,
        metadata={'structure': 'tree'}
    )


def create_dense_csp(
    n_variables: int = 10,
    domain_size: int = 3,
    constraint_density: float = 0.7,
    constraint_tightness: float = 0.5
) -> CSPInstance:
    """
    Create a dense CSP (challenging)

    Args:
        n_variables: Number of variables
        domain_size: Size of each variable's domain
        constraint_density: Fraction of possible constraints to create
        constraint_tightness: Fraction of forbidden tuples

    Returns:
        Dense CSPInstance
    """
    import itertools
    import random

    # Create variables
    variables = [
        Variable(name=f"v{i}", domain=list(range(domain_size)))
        for i in range(n_variables)
    ]

    # Create dense constraints
    constraints = []
    var_names = [v.name for v in variables]
    n_possible_constraints = n_variables * (n_variables - 1) // 2
    n_constraints = int(n_possible_constraints * constraint_density)

    # Generate all possible variable pairs
    var_pairs = list(itertools.combinations(var_names, 2))
    selected_pairs = random.sample(var_pairs, n_constraints)

    for vars_in_constraint in selected_pairs:
        all_tuples = list(itertools.product(range(domain_size), repeat=2))
        n_allowed = int(len(all_tuples) * (1.0 - constraint_tightness))
        allowed_tuples = set(random.sample(all_tuples, n_allowed))

        constraint = Constraint(
            variables=list(vars_in_constraint),
            allowed_tuples=allowed_tuples,
            tightness=constraint_tightness
        )
        constraints.append(constraint)

    return CSPInstance(
        variables=variables,
        constraints=constraints,
        metadata={'structure': 'dense'}
    )


if __name__ == "__main__":
    print("=" * 70)
    print("CSP Models - Demonstration")
    print("=" * 70)

    # Create test CSP
    csp = create_test_csp(n_variables=5, domain_size=3, n_constraints=4)
    print(f"\n[OK] Created test CSP: {csp}")

    # Test constraint graph
    print(f"\n[OK] Constraint graph:")
    print(f"  Nodes: {csp.constraint_graph.number_of_nodes()}")
    print(f"  Edges: {csp.constraint_graph.number_of_edges()}")
    print(f"  Connected: {csp.is_connected()}")

    # Test tree CSP
    tree_csp = create_tree_csp(n_variables=10, domain_size=3)
    print(f"\n[OK] Created tree CSP: {tree_csp}")
    print(f"  Tree width: {tree_csp.tree_width_approximation()}")

    # Test dense CSP
    dense_csp = create_dense_csp(n_variables=10, domain_size=3, constraint_density=0.7)
    print(f"\n[OK] Created dense CSP: {dense_csp}")
    print(f"  Tree width: {dense_csp.tree_width_approximation()}")

    print("\n" + "=" * 70)
    print("[OK] CSP models demonstration complete")
    print("=" * 70)
