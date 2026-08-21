"""
MCTS Coevolution Module

Implements algorithms where candidate decision trees/policies evolve genetically,
with each candidate evaluated using Monte Carlo simulations.

Key concepts:
- Genetic Programming: Evolve decision trees for theorem proving
- Monte Carlo Evaluation: Evaluate each tree with stochastic simulation
- Competitive Coevolution: Solvers and problems coevolve
- Multi-Objective: Pareto optimization of multiple criteria
"""
from __future__ import annotations


import asyncio
import random
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
import json
import copy


# ============================================================================
# Type Definitions
# ============================================================================

class NodeType(Enum):
    """Types of nodes in decision trees"""
    CONDITION = "condition"  # Branching condition
    ACTION = "action"        # Execute tactic
    SEQUENCE = "sequence"    # Execute sequentially
    LOOP = "loop"           # Repeat until condition
    PARALLEL = "parallel"    # Execute in parallel
    TRY = "try"             # Try-catch block


@dataclass
class Tactic:
    """Represents a Lean tactic"""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.5  # Estimated success rate

    def __str__(self) -> str:
        if self.parameters:
            params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            return f"{self.name} [{params}]"
        return self.name


@dataclass
class ProofContext:
    """Context for theorem proving"""
    theorem: str
    goal_state: str
    current_state: str = ""
    proof_steps: List[str] = field(default_factory=list)
    depth: int = 0
    max_depth: int = 100

    # Available information
    hypotheses: List[str] = field(default_factory=list)
    available_tactics: List[Tactic] = field(default_factory=list)

    def clone(self) -> 'ProofContext':
        """Create a copy of the context"""
        return ProofContext(
            theorem=self.theorem,
            goal_state=self.goal_state,
            current_state=self.current_state,
            proof_steps=self.proof_steps.copy(),
            depth=self.depth,
            max_depth=self.max_depth,
            hypotheses=self.hypotheses.copy(),
            available_tactics=self.available_tactics.copy()
        )


@dataclass
class ProofResult:
    """Result of attempting a proof"""
    success: bool
    proof_steps: List[str]
    final_state: str
    depth_reached: int
    time_taken: float
    error_message: str = ""

    # Quality metrics
    elegance_score: float = 0.0  # 0-1, higher is better
    simplicity_score: float = 0.0  # 0-1, higher is better

    def __post_init__(self):
        """Calculate derived metrics"""
        if self.success and self.proof_steps:
            # Simplicity: fewer steps is better
            self.simplicity_score = max(0.0, 1.0 - (len(self.proof_steps) / 100.0))
            # Elegance: combination of factors
            self.elegance_score = (self.simplicity_score + (0.5 if self.success else 0.0)) / 2


@dataclass
class DecisionResult:
    """Result of executing a decision node"""
    success: bool
    next_action: Optional[str] = None
    child_index: Optional[int] = None
    should_continue: bool = True
    message: str = ""


@dataclass
class SingleEvaluation:
    """Monte Carlo evaluation on single theorem"""
    theorem: str
    simulations: int
    success_count: int
    average_depth: float
    average_time: float
    elegance_scores: List[float]
    simplicity_scores: List[float]

    @property
    def success_rate(self) -> float:
        return self.success_count / self.simulations if self.simulations > 0 else 0.0

    @property
    def average_elegance(self) -> float:
        return statistics.mean(self.elegance_scores) if self.elegance_scores else 0.0

    @property
    def average_simplicity(self) -> float:
        return statistics.mean(self.simplicity_scores) if self.simplicity_scores else 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    tree_id: str
    evaluations: List[SingleEvaluation]
    overall_fitness: float
    success_rate: float
    average_depth: float
    average_time: float

    # Multi-objective scores
    elegance_score: float
    simplicity_score: float
    robustness: float  # Consistency across simulations


@dataclass
class TreeAnalysis:
    """Analysis of tree structure"""
    depth: int
    node_count: int
    leaf_count: int
    branching_factor: float
    node_type_distribution: Dict[str, int]
    average_subtree_size: float
    complexity_score: float


@dataclass
class TheoremVariant:
    """Variant of a theorem for competitive coevolution"""
    original_theorem: str
    modified_theorem: str
    difficulty_modifier: float  # >1 means harder
    generation: int

    def __str__(self) -> str:
        return f"{self.modified_theorem} (difficulty: {self.difficulty_modifier:.2f})"


# ============================================================================
# Decision Tree Components
# ============================================================================

class DecisionNode:
    """Node in a decision tree"""

    def __init__(
        self,
        node_type: NodeType,
        content: Any,
        children: List['DecisionNode'] = None,
        condition: Callable[[ProofContext], bool] = None,
        loop_limit: int = 10
    ):
        self.node_type = node_type
        self.content = content
        self.children = children or []
        self.condition = condition
        self.loop_limit = loop_limit
        self.node_id = str(uuid.uuid4())

    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0

    def execute(self, context: ProofContext) -> DecisionResult:
        """Execute this node"""
        if context.depth >= context.max_depth:
            return DecisionResult(
                success=False,
                should_continue=False,
                message="Max depth reached"
            )

        if self.node_type == NodeType.ACTION:
            return self._execute_action(context)
        elif self.node_type == NodeType.CONDITION:
            return self._execute_condition(context)
        elif self.node_type == NodeType.SEQUENCE:
            return self._execute_sequence(context)
        elif self.node_type == NodeType.LOOP:
            return self._execute_loop(context)
        elif self.node_type == NodeType.PARALLEL:
            return self._execute_parallel(context)
        elif self.node_type == NodeType.TRY:
            return self._execute_try(context)
        else:
            return DecisionResult(
                success=False,
                should_continue=True,
                message=f"Unknown node type: {self.node_type}"
            )

    def _execute_action(self, context: ProofContext) -> DecisionResult:
        """Execute an action node"""
        if isinstance(self.content, Tactic):
            tactic = self.content

            # Simulate tactic execution
            success_prob = tactic.success_rate + random.uniform(-0.1, 0.1)
            success_prob = max(0.0, min(1.0, success_prob))

            context.proof_steps.append(str(tactic))
            context.depth += 1

            success = random.random() < success_prob

            return DecisionResult(
                success=success,
                next_action=str(tactic),
                should_continue=success,
                message=f"Executed {tactic.name}"
            )
        else:
            # String action
            context.proof_steps.append(str(self.content))
            context.depth += 1

            return DecisionResult(
                success=True,
                next_action=str(self.content),
                should_continue=True,
                message=f"Executed {self.content}"
            )

    def _execute_condition(self, context: ProofContext) -> DecisionResult:
        """Execute a condition node"""
        if not self.condition:
            # Default: random branch
            child_index = random.randint(0, len(self.children) - 1)
        else:
            # Evaluate condition
            condition_result = self.condition(context)
            child_index = 0 if condition_result else 1 if len(self.children) > 1 else 0

        if child_index < len(self.children):
            return DecisionResult(
                success=True,
                child_index=child_index,
                should_continue=True,
                message=f"Condition evaluated, choosing child {child_index}"
            )
        else:
            return DecisionResult(
                success=False,
                should_continue=False,
                message="No valid child branch"
            )

    def _execute_sequence(self, context: ProofContext) -> DecisionResult:
        """Execute sequence node (returns first child to execute)"""
        # In sequence, we execute children one by one
        # This returns the first child; caller will continue with next
        if self.children:
            return DecisionResult(
                success=True,
                child_index=0,  # Start with first child
                should_continue=True,
                message="Starting sequence"
            )
        return DecisionResult(
            success=False,
            should_continue=False,
            message="Empty sequence"
        )

    def _execute_loop(self, context: ProofContext) -> DecisionResult:
        """Execute loop node"""
        if self.children:
            # Execute child repeatedly
            return DecisionResult(
                success=True,
                child_index=0,
                should_continue=True,
                message="Executing loop body"
            )
        return DecisionResult(
            success=False,
            should_continue=False,
            message="Empty loop"
        )

    def _execute_parallel(self, context: ProofContext) -> DecisionResult:
        """Execute parallel node (execute all children)"""
        if self.children:
            # For simplicity, execute first child
            # In real implementation, would execute all in parallel
            return DecisionResult(
                success=True,
                child_index=0,
                should_continue=True,
                message="Executing parallel branch"
            )
        return DecisionResult(
            success=False,
            should_continue=False,
            message="Empty parallel"
        )

    def _execute_try(self, context: ProofContext) -> DecisionResult:
        """Execute try-catch block"""
        if self.children:
            # First child is try block, second is catch
            return DecisionResult(
                success=True,
                child_index=0,
                should_continue=True,
                message="Executing try block"
            )
        return DecisionResult(
            success=False,
            should_continue=False,
            message="Empty try block"
        )

    def clone(self) -> 'DecisionNode':
        """Create a deep copy of this node"""
        return DecisionNode(
            node_type=self.node_type,
            content=self.content,
            children=[child.clone() for child in self.children],
            condition=self.condition,
            loop_limit=self.loop_limit
        )

    def get_subtree_nodes(self) -> Set[str]:
        """Get all node IDs in this subtree"""
        nodes = {self.node_id}
        for child in self.children:
            nodes.update(child.get_subtree_nodes())
        return nodes

    def get_depth(self) -> int:
        """Get depth of this subtree"""
        if not self.children:
            return 0
        return 1 + max(child.get_depth() for child in self.children)

    def get_node_count(self) -> int:
        """Get total nodes in this subtree"""
        count = 1
        for child in self.children:
            count += child.get_node_count()
        return count


# ============================================================================
# Decision Tree
# ============================================================================

class ProofDecisionTree:
    """Decision tree for theorem proving"""

    def __init__(
        self,
        root: DecisionNode,
        tree_id: str = None,
        generation: int = 0
    ):
        self.tree_id = tree_id or str(uuid.uuid4())
        self.root = root
        self.generation = generation
        self.depth = root.get_depth()
        self.node_count = root.get_node_count()

        # Performance metrics
        self.fitness: float = 0.0
        self.evaluation_count: int = 0
        self.success_rate: float = 0.0
        self.average_depth: float = 0.0
        self.average_time: float = 0.0
        self.elegance_score: float = 0.0
        self.simplicity_score: float = 0.0

    def evaluate(self, context: ProofContext) -> ProofResult:
        """Execute tree to prove theorem"""
        start_time = time.time()
        proof_steps = []

        current_context = context.clone()
        current_context.proof_steps = proof_steps

        result = self._traverse(self.root, current_context)

        time_taken = time.time() - start_time

        return ProofResult(
            success=result and current_context.depth > 0,
            proof_steps=proof_steps,
            final_state=current_context.current_state,
            depth_reached=current_context.depth,
            time_taken=time_taken
        )

    def _traverse(self, node: DecisionNode, context: ProofContext) -> bool:
        """Traverse tree from node"""
        if context.depth >= context.max_depth:
            return False

        result = node.execute(context)

        if not result.should_continue:
            return result.success

        # Execute children based on node type
        if node.node_type == NodeType.SEQUENCE:
            # Execute all children in sequence
            for child in node.children:
                if not self._traverse(child, context):
                    return False
            return True

        elif node.node_type == NodeType.LOOP:
            # Execute child repeatedly until condition fails or limit
            for _ in range(node.loop_limit):
                for child in node.children:
                    if not self._traverse(child, context):
                        return False
            return True

        elif node.node_type == NodeType.PARALLEL:
            # Execute all children
            all_success = True
            for child in node.children:
                if not self._traverse(child, context):
                    all_success = False
            return all_success

        elif node.node_type == NodeType.TRY:
            # Try first child, if fails try second (catch)
            for i, child in enumerate(node.children):
                success = self._traverse(child, context)
                if i == 0 and success:
                    return True
                if i == 1:
                    return success
            return False

        else:
            # CONDITION or ACTION with children
            if result.child_index is not None and result.child_index < len(node.children):
                return self._traverse(node.children[result.child_index], context)
            elif node.children:
                return self._traverse(node.children[0], context)
            else:
                return result.success

    def get_complexity(self) -> int:
        """Calculate tree complexity"""
        return self.node_count * (self.depth + 1)

    def clone(self) -> 'ProofDecisionTree':
        """Create deep copy"""
        return ProofDecisionTree(
            root=self.root.clone(),
            tree_id=str(uuid.uuid4()),
            generation=self.generation
        )

    def get_all_nodes(self) -> List[DecisionNode]:
        """Get all nodes in tree"""
        nodes = []
        self._collect_nodes(self.root, nodes)
        return nodes

    def _collect_nodes(self, node: DecisionNode, nodes: List[DecisionNode]):
        """Collect nodes recursively"""
        nodes.append(node)
        for child in node.children:
            self._collect_nodes(child, nodes)

    def get_random_node(self) -> Optional[DecisionNode]:
        """Get random node from tree"""
        nodes = self.get_all_nodes()
        return random.choice(nodes) if nodes else None

    def get_subtree_rooted_at(self, node_id: str) -> Optional[DecisionNode]:
        """Get subtree rooted at node with given ID"""
        return self._find_subtree(self.root, node_id)

    def _find_subtree(self, node: DecisionNode, node_id: str) -> Optional[DecisionNode]:
        """Find subtree recursively"""
        if node.node_id == node_id:
            return node
        for child in node.children:
            result = self._find_subtree(child, node_id)
            if result:
                return result
        return None

    def replace_subtree(
        self,
        old_node_id: str,
        new_subtree: DecisionNode
    ) -> bool:
        """Replace subtree rooted at old_node_id with new_subtree"""
        return self._replace_subtree(self.root, old_node_id, new_subtree)

    def _replace_subtree(
        self,
        node: DecisionNode,
        old_node_id: str,
        new_subtree: DecisionNode
    ) -> bool:
        """Replace subtree recursively"""
        for i, child in enumerate(node.children):
            if child.node_id == old_node_id:
                node.children[i] = new_subtree
                return True
            if self._replace_subtree(child, old_node_id, new_subtree):
                return True
        return False


# ============================================================================
# Tree Generator
# ============================================================================

class TreeGenerator:
    """Generate random decision trees"""

    def __init__(self, available_actions: List[Tactic] = None):
        self.available_actions = available_actions or self._default_actions()

    def _default_actions(self) -> List[Tactic]:
        """Get default tactic set"""
        return [
            Tactic("apply", {}, 0.6),
            Tactic("rw", {}, 0.7),
            Tactic("simp", {}, 0.8),
            Tactic("intros", {}, 0.9),
            Tactic("cases", {}, 0.6),
            Tactic("induction", {}, 0.5),
            Tactic("refine", {}, 0.4),
            Tactic("exact", {}, 0.5),
            Tactic("assumption", {}, 0.7),
            Tactic("contradiction", {}, 0.4),
        ]

    def generate_full_tree(
        self,
        max_depth: int,
        available_actions: List[Tactic] = None
    ) -> ProofDecisionTree:
        """Generate full tree to max depth"""
        actions = available_actions or self.available_actions
        root = self._generate_full_node(max_depth, 0, actions)
        return ProofDecisionTree(root, generation=0)

    def _generate_full_node(
        self,
        max_depth: int,
        current_depth: int,
        actions: List[Tactic]
    ) -> DecisionNode:
        """Generate full tree node"""
        if current_depth >= max_depth:
            # Leaf node - action
            return DecisionNode(
                node_type=NodeType.ACTION,
                content=random.choice(actions)
            )

        # Internal node - random type
        node_type = random.choice([
            NodeType.CONDITION,
            NodeType.SEQUENCE,
            NodeType.LOOP
        ])

        # Create children
        num_children = random.randint(2, 4)
        children = [
            self._generate_full_node(max_depth, current_depth + 1, actions)
            for _ in range(num_children)
        ]

        return DecisionNode(
            node_type=node_type,
            content=None,
            children=children
        )

    def generate_grow_tree(
        self,
        max_depth: int,
        min_depth: int = 1,
        available_actions: List[Tactic] = None
    ) -> ProofDecisionTree:
        """Grow tree with variable depth (Koza-style)"""
        actions = available_actions or self.available_actions
        root = self._generate_grow_node(max_depth, min_depth, 0, actions)
        return ProofDecisionTree(root, generation=0)

    def _generate_grow_node(
        self,
        max_depth: int,
        min_depth: int,
        current_depth: int,
        actions: List[Tactic]
    ) -> DecisionNode:
        """Generate grow tree node"""
        # At min depth, must continue branching
        # After min depth, can stop at any point
        must_branch = current_depth < min_depth
        can_branch = current_depth < max_depth

        if not can_branch:
            # Must be leaf
            return DecisionNode(
                node_type=NodeType.ACTION,
                content=random.choice(actions)
            )

        if must_branch or random.random() < 0.5:
            # Branching node
            node_type = random.choice([
                NodeType.CONDITION,
                NodeType.SEQUENCE,
                NodeType.LOOP,
                NodeType.PARALLEL
            ])

            num_children = random.randint(2, 3)
            children = [
                self._generate_grow_node(max_depth, min_depth, current_depth + 1, actions)
                for _ in range(num_children)
            ]

            return DecisionNode(
                node_type=node_type,
                content=None,
                children=children
            )
        else:
            # Leaf node
            return DecisionNode(
                node_type=NodeType.ACTION,
                content=random.choice(actions)
            )

    def generate_ramped_half_and_half(
        self,
        population_size: int,
        max_depth: int,
        available_actions: List[Tactic] = None
    ) -> List[ProofDecisionTree]:
        """Generate population with mix of full and grow trees"""
        population = []

        for i in range(population_size):
            # Use different depths for diversity
            depth = 2 + (i % (max_depth - 1))

            if random.random() < 0.5:
                tree = self.generate_full_tree(depth, available_actions)
            else:
                tree = self.generate_grow_tree(depth, min_depth=2, available_actions=available_actions)

            population.append(tree)

        return population


# ============================================================================
# Tree Genetic Operators
# ============================================================================

class TreeCrossover:
    """Crossover operators for decision trees"""

    def subtree_crossover(
        self,
        parent1: ProofDecisionTree,
        parent2: ProofDecisionTree,
        max_depth: int = 17
    ) -> Tuple[ProofDecisionTree, ProofDecisionTree]:
        """Exchange random subtrees"""
        # Clone parents
        child1 = parent1.clone()
        child2 = parent2.clone()

        # Select random crossover points
        node1 = child1.get_random_node()
        node2 = child2.get_random_node()

        if not node1 or not node2:
            return child1, child2

        # Check depth constraints
        depth1 = node1.get_depth()
        depth2 = node2.get_depth()

        if child1.depth - depth1 + depth2 > max_depth:
            return child1, child2
        if child2.depth - depth2 + depth1 > max_depth:
            return child1, child2

        # Swap subtrees
        temp_content = node1.content
        temp_type = node1.node_type
        temp_children = node1.children

        node1.content = node2.content
        node1.node_type = node2.node_type
        node1.children = [child.clone() for child in node2.children]

        node2.content = temp_content
        node2.node_type = temp_type
        node2.children = temp_children

        # Update metadata
        child1.depth = child1.root.get_depth()
        child1.node_count = child1.root.get_node_count()
        child2.depth = child2.root.get_depth()
        child2.node_count = child2.root.get_node_count()

        return child1, child2

    def root_crossover(
        self,
        parent1: ProofDecisionTree,
        parent2: ProofDecisionTree
    ) -> Tuple[ProofDecisionTree, ProofDecisionTree]:
        """Replace root of one with subtree of other"""
        # Select subtree from parent2
        subtree = parent2.get_random_node()

        if not subtree:
            return parent1.clone(), parent2.clone()

        # Create new trees
        child1 = ProofDecisionTree(
            root=subtree.clone(),
            generation=max(parent1.generation, parent2.generation) + 1
        )
        child2 = parent2.clone()

        return child1, child2

    def uniform_crossover(
        self,
        parent1: ProofDecisionTree,
        parent2: ProofDecisionTree,
        swap_probability: float = 0.3
    ) -> Tuple[ProofDecisionTree, ProofDecisionTree]:
        """Crossover at multiple points"""
        child1 = parent1.clone()
        child2 = parent2.clone()

        # Traverse both trees and swap matching nodes
        self._uniform_crossover_recursive(
            child1.root,
            child2.root,
            swap_probability
        )

        # Update metadata
        child1.depth = child1.root.get_depth()
        child1.node_count = child1.root.get_node_count()
        child2.depth = child2.root.get_depth()
        child2.node_count = child2.root.get_node_count()

        return child1, child2

    def _uniform_crossover_recursive(
        self,
        node1: DecisionNode,
        node2: DecisionNode,
        swap_probability: float
    ):
        """Recursively apply uniform crossover"""
        if random.random() < swap_probability:
            # Swap content and children
            node1.content, node2.content = node2.content, node1.content
            node1.node_type, node2.node_type = node2.node_type, node1.node_type

            # Swap children lists
            node1.children, node2.children = node2.children, node1.children

        # Recurse on children
        min_children = min(len(node1.children), len(node2.children))
        for i in range(min_children):
            self._uniform_crossover_recursive(
                node1.children[i],
                node2.children[i],
                swap_probability
            )


class TreeMutation:
    """Mutation operators for decision trees"""

    def __init__(self, generator: TreeGenerator):
        self.generator = generator

    def subtree_mutation(
        self,
        tree: ProofDecisionTree,
        max_subtree_size: int = 10
    ) -> ProofDecisionTree:
        """Replace random subtree with generated one"""
        mutated = tree.clone()
        node = mutated.get_random_node()

        if not node:
            return mutated

        # Generate new subtree
        max_depth = min(max_subtree_size, 5)
        new_subtree = self.generator._generate_grow_node(
            max_depth,
            2,
            0,
            self.generator.available_actions
        )

        # Replace node content
        node.content = new_subtree.content
        node.node_type = new_subtree.node_type
        node.children = new_subtree.children

        # Update metadata
        mutated.depth = mutated.root.get_depth()
        mutated.node_count = mutated.root.get_node_count()

        return mutated

    def point_mutation(
        self,
        tree: ProofDecisionTree,
        mutation_rate: float = 0.1
    ) -> ProofDecisionTree:
        """Mutate individual nodes"""
        mutated = tree.clone()
        nodes = mutated.get_all_nodes()

        for node in nodes:
            if random.random() < mutation_rate:
                if node.node_type == NodeType.ACTION:
                    # Change action
                    node.content = random.choice(self.generator.available_actions)
                elif node.node_type == NodeType.CONDITION:
                    # Could modify condition here
                    pass
                else:
                    # Structural mutation
                    if node.children and random.random() < 0.5:
                        # Reorder children
                        random.shuffle(node.children)

        return mutated

    def insert_node(
        self,
        tree: ProofDecisionTree
    ) -> ProofDecisionTree:
        """Insert new node at random position"""
        mutated = tree.clone()
        nodes = mutated.get_all_nodes()

        if not nodes:
            return mutated

        # Select random node
        target_node = random.choice(nodes)

        # Create new node
        new_node = DecisionNode(
            node_type=NodeType.SEQUENCE,
            content=None,
            children=[target_node.clone()]
        )

        # Replace target with new node (in parent's children list)
        self._replace_node_in_tree(mutated.root, target_node.node_id, new_node)

        # Update metadata
        mutated.depth = mutated.root.get_depth()
        mutated.node_count = mutated.root.get_node_count()

        return mutated

    def _replace_node_in_tree(
        self,
        parent: DecisionNode,
        target_id: str,
        new_node: DecisionNode
    ) -> bool:
        """Replace node in tree"""
        for i, child in enumerate(parent.children):
            if child.node_id == target_id:
                parent.children[i] = new_node
                return True
            if self._replace_node_in_tree(child, target_id, new_node):
                return True
        return False

    def delete_node(
        self,
        tree: ProofDecisionTree
    ) -> ProofDecisionTree:
        """Delete random node (promote children)"""
        mutated = tree.clone()

        # Don't delete root
        internal_nodes = [
            n for n in mutated.get_all_nodes()
            if n != mutated.root and n.children
        ]

        if not internal_nodes:
            return mutated

        # Select random internal node
        target_node = random.choice(internal_nodes)

        # Delete node and promote children
        self._delete_node_in_tree(mutated.root, target_node.node_id)

        # Update metadata
        mutated.depth = mutated.root.get_depth()
        mutated.node_count = mutated.root.get_node_count()

        return mutated

    def _delete_node_in_tree(
        self,
        parent: DecisionNode,
        target_id: str
    ) -> bool:
        """Delete node and promote children"""
        for i, child in enumerate(parent.children):
            if child.node_id == target_id:
                # Replace with children
                parent.children.pop(i)
                parent.children.extend(child.children)
                return True
            if self._delete_node_in_tree(child, target_id):
                return True
        return False

    def shrink_tree(
        self,
        tree: ProofDecisionTree,
        target_reduction: float = 0.2
    ) -> ProofDecisionTree:
        """Reduce tree size"""
        mutated = tree.clone()
        target_nodes = int(mutated.node_count * target_reduction)

        for _ in range(target_nodes):
            mutated = self.delete_node(mutated)

        return mutated

    def expand_tree(
        self,
        tree: ProofDecisionTree,
        target_expansion: float = 0.2
    ) -> ProofDecisionTree:
        """Increase tree size"""
        mutated = tree.clone()
        target_nodes = int(mutated.node_count * target_expansion)

        for _ in range(target_nodes):
            mutated = self.insert_node(mutated)

        return mutated


# ============================================================================
# Monte Carlo Tree Evaluator
# ============================================================================

class MCTreeEvaluator:
    """Evaluate decision trees using Monte Carlo simulation"""

    def __init__(
        self,
        simulations: int = 100,
        max_depth: int = 50
    ):
        self.simulations = simulations
        self.max_depth = max_depth
        self.executor = ThreadPoolExecutor(max_workers=4)

    def evaluate(
        self,
        tree: ProofDecisionTree,
        test_theorems: List[str]
    ) -> EvaluationResult:
        """Evaluate tree on multiple test theorems"""
        evaluations = []

        for theorem in test_theorems:
            single_eval = self.evaluate_single(tree, theorem, self.simulations)
            evaluations.append(single_eval)

        # Calculate overall metrics
        overall_fitness = statistics.mean(e.success_rate for e in evaluations)
        success_rate = overall_fitness
        average_depth = statistics.mean(e.average_depth for e in evaluations)
        average_time = statistics.mean(e.average_time for e in evaluations)
        elegance_score = statistics.mean(e.average_elegance for e in evaluations)
        simplicity_score = statistics.mean(e.average_simplicity for e in evaluations)

        # Robustness: low variance in success rates
        success_rates = [e.success_rate for e in evaluations]
        robustness = 1.0 - (statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0)
        robustness = max(0.0, robustness)

        return EvaluationResult(
            tree_id=tree.tree_id,
            evaluations=evaluations,
            overall_fitness=overall_fitness,
            success_rate=success_rate,
            average_depth=average_depth,
            average_time=average_time,
            elegance_score=elegance_score,
            simplicity_score=simplicity_score,
            robustness=robustness
        )

    def evaluate_single(
        self,
        tree: ProofDecisionTree,
        theorem: str,
        simulations: int
    ) -> SingleEvaluation:
        """Monte Carlo evaluation on single theorem"""
        results = []
        elegance_scores = []
        simplicity_scores = []

        for _ in range(simulations):
            # Create context
            context = ProofContext(
                theorem=theorem,
                goal_state=f"prove {theorem}",
                max_depth=self.max_depth
            )

            # Execute tree with random seed
            result = tree.evaluate(context)
            results.append(result)
            elegance_scores.append(result.elegance_score)
            simplicity_scores.append(result.simplicity_score)

        # Aggregate results
        success_count = sum(1 for r in results if r.success)
        average_depth = statistics.mean(r.depth_reached for r in results) if results else 0.0
        average_time = statistics.mean(r.time_taken for r in results) if results else 0.0

        return SingleEvaluation(
            theorem=theorem,
            simulations=simulations,
            success_count=success_count,
            average_depth=average_depth,
            average_time=average_time,
            elegance_scores=elegance_scores,
            simplicity_scores=simplicity_scores
        )

    async def evaluate_with_leanaide(
        self,
        tree: ProofDecisionTree,
        theorem: str,
        leanaide_client=None,
        simulations: int = 20
    ) -> EvaluationResult:
        """Evaluate with Lean formal verification (mock)"""
        # Run basic evaluation
        single_eval = self.evaluate_single(tree, theorem, simulations)

        # In real implementation, would verify with LeanAide
        # For now, simulate verification bonus
        verification_bonus = 0.0
        if single_eval.success_rate > 0.8:
            verification_bonus = 0.2

        overall_fitness = single_eval.success_rate + verification_bonus

        return EvaluationResult(
            tree_id=tree.tree_id,
            evaluations=[single_eval],
            overall_fitness=min(1.0, overall_fitness),
            success_rate=single_eval.success_rate,
            average_depth=single_eval.average_depth,
            average_time=single_eval.average_time,
            elegance_score=single_eval.average_elegance,
            simplicity_score=single_eval.average_simplicity,
            robustness=0.8  # Default robustness
        )

    def parallel_evaluate(
        self,
        population: List[ProofDecisionTree],
        test_theorems: List[str]
    ) -> List[EvaluationResult]:
        """Evaluate multiple trees in parallel"""
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(
                self.executor,
                self.evaluate,
                tree,
                test_theorems
            )
            for tree in population
        ]

        return loop.run_until_complete(asyncio.gather(*tasks))


# ============================================================================
# Selection Operators
# ============================================================================

class TournamentSelector:
    """Tournament selection for genetic programming"""

    def __init__(self, tournament_size: int = 5):
        self.tournament_size = tournament_size

    def select(
        self,
        population: List[ProofDecisionTree],
        num_select: int
    ) -> List[ProofDecisionTree]:
        """Select individuals using tournament selection"""
        selected = []

        for _ in range(num_select):
            # Random tournament participants
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            # Select best fitness
            winner = max(tournament, key=lambda t: t.fitness)
            selected.append(winner)

        return selected


class RouletteWheelSelector:
    """Roulette wheel (fitness proportionate) selection"""

    def select(
        self,
        population: List[ProofDecisionTree],
        num_select: int
    ) -> List[ProofDecisionTree]:
        """Select individuals using fitness proportionate selection"""
        # Ensure non-negative fitness
        min_fitness = min((t.fitness for t in population), default=0.0)
        adjusted_fitness = [t.fitness - min_fitness + 0.01 for t in population]
        total_fitness = sum(adjusted_fitness)

        if total_fitness == 0:
            return random.choices(population, k=num_select)

        # Calculate probabilities
        probabilities = [f / total_fitness for f in adjusted_fitness]

        # Select
        selected = random.choices(population, weights=probabilities, k=num_select)
        return selected


class RankSelector:
    """Rank-based selection"""

    def select(
        self,
        population: List[ProofDecisionTree],
        num_select: int,
        pressure: float = 1.5  # Selection pressure
    ) -> List[ProofDecisionTree]:
        """Select using rank-based selection"""
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda t: t.fitness)
        n = len(sorted_pop)

        # Calculate rank probabilities
        ranks = list(range(1, n + 1))
        total = sum(ranks)
        probabilities = [r / total for r in ranks]

        # Select (higher rank = better fitness = higher probability)
        selected = []
        for _ in range(num_select):
            idx = random.choices(range(n), weights=probabilities)[0]
            # Reverse: higher fitness should have higher probability
            selected.append(sorted_pop[n - 1 - idx])

        return selected


# ============================================================================
# Tree Coevolution Engine
# ============================================================================

class TreeCoevolution:
    """Coevolve decision trees with MC evaluation"""

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        elitism: int = 5,
        max_depth: int = 17,
        simulations: int = 100
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.max_depth = max_depth

        # Initialize components
        self.generator = TreeGenerator()
        self.crossover = TreeCrossover()
        self.mutation = TreeMutation(self.generator)
        self.evaluator = MCTreeEvaluator(simulations=simulations, max_depth=max_depth)
        self.selector = TournamentSelector(tournament_size=5)

        # Tracking
        self.history: List[Dict] = []
        self.best_tree: Optional[ProofDecisionTree] = None
        self.best_fitness: float = 0.0

    def initialize_population(
        self,
        available_actions: List[Tactic] = None
    ) -> List[ProofDecisionTree]:
        """Initialize population"""
        return self.generator.generate_ramped_half_and_half(
            self.population_size,
            self.max_depth,
            available_actions
        )

    async def coevolve(
        self,
        test_theorems: List[str],
        leanaide_client=None
    ) -> ProofDecisionTree:
        """Coevolve trees with Monte Carlo evaluation"""
        # Initialize population
        population = self.initialize_population()

        self.best_tree = None
        self.best_fitness = 0.0

        print(f"Starting coevolution: {self.population_size} trees, {self.generations} generations")
        print(f"Test theorems: {len(test_theorems)}")

        for generation in range(self.generations):
            start_time = time.time()

            # 1. Evaluate all trees with Monte Carlo
            if leanaide_client:
                # Parallel evaluation with LeanAide verification
                tasks = [
                    self.evaluator.evaluate_with_leanaide(
                        tree,
                        random.choice(test_theorems),
                        leanaide_client,
                        self.evaluator.simulations
                    )
                    for tree in population
                ]
                evaluations = await asyncio.gather(*tasks)
            else:
                # Parallel evaluation
                evaluations = self.evaluator.parallel_evaluate(population, test_theorems)

            # Assign fitness
            for tree, eval_result in zip(population, evaluations):
                tree.fitness = eval_result.overall_fitness
                tree.success_rate = eval_result.success_rate
                tree.elegance_score = eval_result.elegance_score
                tree.simplicity_score = eval_result.simplicity_score

            # 2. Track best
            current_best = max(population, key=lambda t: t.fitness)
            gen_best_fitness = current_best.fitness

            if current_best.fitness > self.best_fitness:
                self.best_tree = current_best.clone()
                self.best_fitness = current_best.fitness
                print(f"  New best fitness: {self.best_fitness:.4f}")

            # 3. Selection
            parents = self.selector.select(population, self.population_size)

            # 4. Create next generation
            next_gen = []

            # Elitism: keep best individuals
            elites = self.select_elites(population, self.elitism)
            next_gen.extend(elites)

            # Crossover and mutation
            while len(next_gen) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)

                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover.subtree_crossover(
                        parent1, parent2, self.max_depth
                    )
                    child1.generation = generation + 1
                    child2.generation = generation + 1
                    next_gen.extend([child1, child2])
                else:
                    # Clone parents
                    next_gen.extend([parent1.clone(), parent2.clone()])

            # Trim to exact size
            next_gen = next_gen[:self.population_size]

            # Mutation (skip elites)
            for i in range(self.elitism, len(next_gen)):
                if random.random() < self.mutation_rate:
                    mutated = self.mutate(next_gen[i])
                    mutated.generation = generation + 1
                    next_gen[i] = mutated

            population = next_gen

            # Statistics
            gen_time = time.time() - start_time
            avg_fitness = statistics.mean(t.fitness for t in population)

            self.history.append({
                'generation': generation,
                'best_fitness': gen_best_fitness,
                'avg_fitness': avg_fitness,
                'time': gen_time,
                'population_size': len(population)
            })

            if generation % 10 == 0 or generation == self.generations - 1:
                print(f"Generation {generation}: best={gen_best_fitness:.4f}, avg={avg_fitness:.4f}, time={gen_time:.2f}s")

        print(f"Coevolution complete. Best fitness: {self.best_fitness:.4f}")
        return self.best_tree

    def select_elites(
        self,
        population: List[ProofDecisionTree],
        num_elites: int
    ) -> List[ProofDecisionTree]:
        """Select elite individuals"""
        sorted_pop = sorted(population, key=lambda t: t.fitness, reverse=True)
        return [tree.clone() for tree in sorted_pop[:num_elites]]

    def mutate(self, tree: ProofDecisionTree) -> ProofDecisionTree:
        """Apply random mutation"""
        mutation_type = random.choice([
            'subtree', 'point', 'insert', 'delete', 'shrink', 'expand'
        ])

        if mutation_type == 'subtree':
            return self.mutation.subtree_mutation(tree)
        elif mutation_type == 'point':
            return self.mutation.point_mutation(tree)
        elif mutation_type == 'insert':
            return self.mutation.insert_node(tree)
        elif mutation_type == 'delete':
            return self.mutation.delete_node(tree)
        elif mutation_type == 'shrink':
            return self.mutation.shrink_tree(tree)
        else:  # expand
            return self.mutation.expand_tree(tree)


# ============================================================================
# Competitive Coevolution
# ============================================================================

class CompetitiveCoevolution:
    """Coevolve solver and difficulty"""

    def __init__(
        self,
        solver_pop_size: int = 50,
        problem_pop_size: int = 20,
        generations: int = 100
    ):
        self.solver_pop_size = solver_pop_size
        self.problem_pop_size = problem_pop_size
        self.generations = generations

        self.generator = TreeGenerator()
        self.evaluator = MCTreeEvaluator(simulations=50, max_depth=50)

        self.solver_population: List[ProofDecisionTree] = []
        self.problem_population: List[TheoremVariant] = []

    async def competitive_coevolve(
        self,
        initial_theorems: List[str]
    ) -> ProofDecisionTree:
        """Coevolve solvers and problems"""
        print("Starting competitive coevolution")

        # Initialize populations
        self.solver_population = self.generator.generate_ramped_half_and_half(
            self.solver_pop_size,
            15
        )

        self.problem_population = [
            TheoremVariant(
                original_theorem=thm,
                modified_theorem=thm,
                difficulty_modifier=1.0,
                generation=0
            )
            for thm in initial_theorems[:self.problem_pop_size]
        ]

        best_solver = None
        best_solver_fitness = 0.0

        for generation in range(self.generations):
            # 1. Evaluate solvers on current problems
            solver_scores = await self._evaluate_solvers()

            # 2. Track best solver
            current_best_idx = max(range(len(solver_scores)), key=lambda i: solver_scores[i])
            current_best_fitness = solver_scores[current_best_idx]

            if current_best_fitness > best_solver_fitness:
                best_solver = self.solver_population[current_best_idx].clone()
                best_solver_fitness = current_best_fitness
                print(f"  New best solver fitness: {best_solver_fitness:.4f}")

            # 3. Select and evolve solvers
            self.solver_population = self._evolve_solvers(solver_scores)

            # 4. Generate harder problem variants
            self.problem_population = self._evolve_problems(solver_scores)

            if generation % 20 == 0:
                avg_solver_score = statistics.mean(solver_scores)
                print(f"Generation {generation}: best={current_best_fitness:.4f}, avg={avg_solver_score:.4f}")

        print(f"Competitive coevolution complete. Best solver fitness: {best_solver_fitness:.4f}")
        return best_solver

    async def _evaluate_solvers(self) -> List[float]:
        """Evaluate all solvers on all problems"""
        scores = []

        for solver in self.solver_population:
            total_score = 0.0

            for problem in self.problem_population:
                # Adjust simulation count based on difficulty
                sims = max(10, int(50 / problem.difficulty_modifier))
                eval_result = self.evaluator.evaluate(
                    solver,
                    [problem.modified_theorem]
                )

                # Score adjusted by difficulty
                total_score += eval_result.success_rate * problem.difficulty_modifier

            scores.append(total_score / len(self.problem_population))

        return scores

    def _evolve_solvers(self, scores: List[float]) -> List[ProofDecisionTree]:
        """Evolve solver population"""
        # Sort by score
        sorted_solvers = [
            (solver, score)
            for solver, score in zip(self.solver_population, scores)
        ]
        sorted_solvers.sort(key=lambda x: x[1], reverse=True)

        # Elitism
        new_pop = [solver.clone() for solver, _ in sorted_solvers[:5]]

        # Generate new solvers
        crossover = TreeCrossover()
        mutation = TreeMutation(self.generator)

        while len(new_pop) < self.solver_pop_size:
            # Select from top half
            parent1, parent2 = random.choices(
                [s for s, _ in sorted_solvers[:self.solver_pop_size//2]],
                k=2
            )

            if random.random() < 0.8:
                child1, child2 = crossover.subtree_crossover(parent1, parent2)
                new_pop.extend([child1, child2])
            else:
                new_pop.append(parent1.clone())

        # Mutation
        for i in range(5, len(new_pop)):
            if random.random() < 0.15:
                new_pop[i] = mutation.subtree_mutation(new_pop[i])

        return new_pop[:self.solver_pop_size]

    def _evolve_problems(self, solver_scores: List[float]) -> List[TheoremVariant]:
        """Evolve problem population"""
        avg_solver_score = statistics.mean(solver_scores)

        # Keep hardest problems (where solvers performed worst)
        problem_difficulties = []
        for problem in self.problem_population:
            # Calculate how well solvers did on this problem
            problem_scores = []
            for solver, score in zip(self.solver_population, solver_scores):
                eval_result = self.evaluator.evaluate(solver, [problem.modified_theorem])
                problem_scores.append(eval_result.success_rate)

            avg_score = statistics.mean(problem_scores)
            difficulty = 1.0 / (avg_score + 0.01)  # Lower success = higher difficulty
            problem_difficulties.append((problem, difficulty))

        # Select hardest problems
        problem_difficulties.sort(key=lambda x: x[1], reverse=True)
        kept_problems = [p for p, _ in problem_difficulties[:self.problem_pop_size//2]]

        # Generate harder variants
        new_problems = []
        for problem in kept_problems:
            # Create harder variant
            new_modifier = problem.difficulty_modifier * 1.1
            new_theorem = self._modify_theorem(problem.original_theorem, new_modifier)

            new_problems.append(TheoremVariant(
                original_theorem=problem.original_theorem,
                modified_theorem=new_theorem,
                difficulty_modifier=new_modifier,
                generation=problem.generation + 1
            ))

        # Add new random theorems
        while len(new_problems) < self.problem_pop_size:
            new_problems.append(TheoremVariant(
                original_theorem=kept_problems[0].original_theorem,
                modified_theorem=kept_problems[0].original_theorem,
                difficulty_modifier=1.0,
                generation=0
            ))

        return new_problems

    def _modify_theorem(self, theorem: str, difficulty: float) -> str:
        """Modify theorem to adjust difficulty (mock)"""
        # In real implementation, would use actual theorem transformations
        modifiers = [
            " with additional constraints",
            " under stronger conditions",
            " with extended requirements"
        ]
        if difficulty > 1.2:
            return theorem + random.choice(modifiers)
        return theorem


# ============================================================================
# Multi-Objective Coevolution
# ============================================================================

class MultiObjectiveCoevolution:
    """Coevolve trees optimizing multiple objectives"""

    def __init__(
        self,
        objectives: List[str] = None,
        population_size: int = 100,
        generations: int = 50
    ):
        self.objectives = objectives or ["success", "speed", "elegance", "simplicity"]
        self.population_size = population_size
        self.generations = generations

        self.generator = TreeGenerator()
        self.evaluator = MCTreeEvaluator(simulations=100, max_depth=50)

    async def coevolve_multi_objective(
        self,
        test_theorems: List[str]
    ) -> List[ProofDecisionTree]:
        """Coevolve Pareto-optimal trees using NSGA-II"""
        print(f"Starting multi-objective coevolution: {self.objectives}")

        # Initialize population
        population = self.generator.generate_ramped_half_and_half(
            self.population_size,
            15
        )

        # Evaluate initial population
        population = await self._evaluate_population(population, test_theorems)

        for generation in range(self.generations):
            # 1. Non-dominated sort
            fronts = self._non_dominated_sort(population)

            # 2. Calculate crowding distance
            for front in fronts:
                self._calculate_crowding_distance(front)

            # 3. Selection (NSGA-II style)
            parents = self._nsga2_selection(population, fronts)

            # 4. Create offspring
            offspring = self._create_offspring(parents)

            # 5. Evaluate offspring
            offspring = await self._evaluate_population(offspring, test_theorems)

            # 6. Survival selection
            population = self._survival_selection(population + offspring)

            if generation % 10 == 0:
                pareto_front = fronts[0] if fronts else []
                print(f"Generation {generation}: Pareto front size = {len(pareto_front)}")

        # Return Pareto front
        fronts = self._non_dominated_sort(population)
        pareto_front = fronts[0] if fronts else population

        print(f"Multi-objective coevolution complete. Pareto front: {len(pareto_front)} solutions")
        return pareto_front

    async def _evaluate_population(
        self,
        population: List[ProofDecisionTree],
        test_theorems: List[str]
    ) -> List[ProofDecisionTree]:
        """Evaluate all individuals in population"""
        for tree in population:
            eval_result = self.evaluator.evaluate(tree, test_theorems)

            # Set multi-objective fitness
            tree.objectives = {
                'success': eval_result.success_rate,
                'speed': 1.0 - min(1.0, eval_result.average_time / 10.0),
                'elegance': eval_result.elegance_score,
                'simplicity': eval_result.simplicity_score
            }

            # Overall fitness (for compatibility)
            tree.fitness = eval_result.overall_fitness

        return population

    def _non_dominated_sort(
        self,
        population: List[ProofDecisionTree]
    ) -> List[List[ProofDecisionTree]]:
        """Non-dominated sorting (NSGA-II)"""
        fronts = []
        remaining = population.copy()

        while remaining:
            # Find current Pareto front
            current_front = []
            for i, ind1 in enumerate(remaining):
                dominated = False
                for j, ind2 in enumerate(remaining):
                    if i != j and self._dominates(ind2, ind1):
                        dominated = True
                        break
                if not dominated:
                    current_front.append(ind1)

            fronts.append(current_front)

            # Remove front from remaining
            for ind in current_front:
                remaining.remove(ind)

        return fronts

    def _dominates(
        self,
        ind1: ProofDecisionTree,
        ind2: ProofDecisionTree
    ) -> bool:
        """Check if ind1 dominates ind2"""
        obj1 = getattr(ind1, 'objectives', {})
        obj2 = getattr(ind2, 'objectives', {})

        at_least_one_better = False
        for obj in self.objectives:
            val1 = obj1.get(obj, 0.0)
            val2 = obj2.get(obj, 0.0)
            if val1 < val2:
                return False
            if val1 > val2:
                at_least_one_better = True

        return at_least_one_better

    def _calculate_crowding_distance(
        self,
        front: List[ProofDecisionTree]
    ):
        """Calculate crowding distance for diversity"""
        if not front:
            return

        n = len(front)

        # Initialize distance
        for ind in front:
            ind.crowding_distance = 0.0

        # For each objective
        for obj in self.objectives:
            # Sort by objective value
            sorted_front = sorted(front, key=lambda x: x.objectives.get(obj, 0.0))

            # Boundary points get infinite distance
            sorted_front[0].crowding_distance = float('inf')
            sorted_front[-1].crowding_distance = float('inf')

            # Calculate distance for interior points
            if n > 2:
                obj_range = sorted_front[-1].objectives.get(obj, 0.0) - sorted_front[0].objectives.get(obj, 0.0)
                if obj_range > 0:
                    for i in range(1, n - 1):
                        dist = (sorted_front[i + 1].objectives.get(obj, 0.0) -
                               sorted_front[i - 1].objectives.get(obj, 0.0)) / obj_range
                        sorted_front[i].crowding_distance += dist

    def _nsga2_selection(
        self,
        population: List[ProofDecisionTree],
        fronts: List[List[ProofDecisionTree]]
    ) -> List[ProofDecisionTree]:
        """Select parents using NSGA-II selection"""
        selected = []

        while len(selected) < self.population_size:
            # Binary tournament selection
            ind1, ind2 = random.sample(population, 2)

            # Compare rank (front index) and crowding distance
            rank1 = self._get_rank(ind1, fronts)
            rank2 = self._get_rank(ind2, fronts)

            if rank1 < rank2:
                selected.append(ind1)
            elif rank2 < rank1:
                selected.append(ind2)
            else:
                # Same front, use crowding distance
                dist1 = getattr(ind1, 'crowding_distance', 0.0)
                dist2 = getattr(ind2, 'crowding_distance', 0.0)
                selected.append(ind1 if dist1 > dist2 else ind2)

        return selected

    def _get_rank(
        self,
        individual: ProofDecisionTree,
        fronts: List[List[ProofDecisionTree]]
    ) -> int:
        """Get front rank of individual"""
        for i, front in enumerate(fronts):
            if individual in front:
                return i
        return len(fronts)

    def _create_offspring(
        self,
        parents: List[ProofDecisionTree]
    ) -> List[ProofDecisionTree]:
        """Create offspring through crossover and mutation"""
        offspring = []
        crossover = TreeCrossover()
        mutation = TreeMutation(self.generator)

        while len(offspring) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)

            if random.random() < 0.9:
                child1, child2 = crossover.subtree_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.clone(), parent2.clone()])

        # Mutation
        for i in range(len(offspring)):
            if random.random() < 0.1:
                offspring[i] = mutation.subtree_mutation(offspring[i])

        return offspring[:self.population_size]

    def _survival_selection(
        self,
        combined: List[ProofDecisionTree]
    ) -> List[ProofDecisionTree]:
        """Select survivors for next generation"""
        # Non-dominated sort
        fronts = self._non_dominated_sort(combined)

        # Fill next generation front by front
        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.population_size:
                next_gen.extend(front)
            else:
                # Need to select some from this front based on crowding distance
                self._calculate_crowding_distance(front)
                remaining = self.population_size - len(next_gen)
                sorted_by_crowding = sorted(
                    front,
                    key=lambda x: getattr(x, 'crowding_distance', 0.0),
                    reverse=True
                )
                next_gen.extend(sorted_by_crowding[:remaining])
                break

        return next_gen


# ============================================================================
# Tree Pruning and Simplification
# ============================================================================

class TreePruner:
    """Prune and simplify decision trees"""

    def prune_dead_branches(
        self,
        tree: ProofDecisionTree,
        test_cases: List[str]
    ) -> ProofDecisionTree:
        """Remove branches never reached"""
        # Analyze which nodes are reached
        reached_nodes = self._analyze_reached_nodes(tree, test_cases)

        # Prune unreached nodes
        pruned = self._prune_unreached(tree.root, reached_nodes)
        return ProofDecisionTree(root=pruned, generation=tree.generation)

    def _analyze_reached_nodes(
        self,
        tree: ProofDecisionTree,
        test_cases: List[str]
    ) -> Set[str]:
        """Analyze which nodes are reached during evaluation"""
        reached = set()

        for test in test_cases:
            context = ProofContext(
                theorem=test,
                goal_state=f"prove {test}",
                max_depth=100
            )
            self._trace_execution(tree.root, context, reached)

        return reached

    def _trace_execution(
        self,
        node: DecisionNode,
        context: ProofContext,
        reached: Set[str]
    ):
        """Trace execution and mark reached nodes"""
        reached.add(node.node_id)

        result = node.execute(context)
        if result.should_continue and result.child_index is not None:
            if result.child_index < len(node.children):
                self._trace_execution(
                    node.children[result.child_index],
                    context,
                    reached
                )

    def _prune_unreached(
        self,
        node: DecisionNode,
        reached: Set[str]
    ) -> DecisionNode:
        """Prune unreached nodes"""
        if node.node_id not in reached:
            # Don't include this node
            return None

        # Recursively prune children
        pruned_children = []
        for child in node.children:
            pruned_child = self._prune_unreached(child, reached)
            if pruned_child:
                pruned_children.append(pruned_child)

        # Create pruned node
        return DecisionNode(
            node_type=node.node_type,
            content=node.content,
            children=pruned_children,
            condition=node.condition,
            loop_limit=node.loop_limit
        )

    def simplify_conditions(
        self,
        tree: ProofDecisionTree
    ) -> ProofDecisionTree:
        """Simplify conditional logic"""
        simplified_root = self._simplify_node(tree.root)
        return ProofDecisionTree(root=simplified_root, generation=tree.generation)

    def _simplify_node(self, node: DecisionNode) -> DecisionNode:
        """Simplify a node"""
        # Recursively simplify children
        simplified_children = [
            self._simplify_node(child) for child in node.children
        ]

        # Simplify this node
        if node.node_type == NodeType.CONDITION:
            # Remove duplicate branches
            unique_children = []
            seen_contents = set()
            for child in simplified_children:
                if child and child.content not in seen_contents:
                    unique_children.append(child)
                    seen_contents.add(child.content)

            if len(unique_children) <= 1:
                # Convert to action or sequence
                if unique_children:
                    return unique_children[0]
                else:
                    return DecisionNode(
                        node_type=NodeType.ACTION,
                        content=Tactic("skip")
                    )

            return DecisionNode(
                node_type=node.node_type,
                content=node.content,
                children=unique_children,
                condition=node.condition
            )

        return DecisionNode(
            node_type=node.node_type,
            content=node.content,
            children=simplified_children,
            condition=node.condition,
            loop_limit=node.loop_limit
        )

    def merge_identical_subtrees(
        self,
        tree: ProofDecisionTree
    ) -> ProofDecisionTree:
        """Merge duplicate subtrees"""
        # Find identical subtrees
        subtree_hashes = self._hash_subtrees(tree.root)

        # Merge duplicates
        merged_root = self._merge_duplicates(tree.root, subtree_hashes)
        return ProofDecisionTree(root=merged_root, generation=tree.generation)

    def _hash_subtrees(self, node: DecisionNode) -> Dict[str, List[str]]:
        """Hash all subtrees"""
        hashes = {}
        self._collect_hashes(node, hashes)
        return hashes

    def _collect_hashes(
        self,
        node: DecisionNode,
        hashes: Dict[str, List[str]]
    ) -> str:
        """Collect subtree hashes"""
        child_hashes = [
            self._collect_hashes(child, hashes)
            for child in node.children
        ]

        # Create hash from structure
        content_str = str(node.content)
        type_str = str(node.node_type)
        children_str = ",".join(child_hashes)

        node_hash = hash(f"{type_str}:{content_str}:{children_str}")

        if node_hash not in hashes:
            hashes[node_hash] = []
        hashes[node_hash].append(node.node_id)

        return str(node_hash)

    def _merge_duplicates(
        self,
        node: DecisionNode,
        hashes: Dict[str, List[str]]
    ) -> DecisionNode:
        """Merge duplicate subtrees"""
        # This is a simplified version
        # In practice, would track which nodes have been merged
        return node.clone()


# ============================================================================
# Ensemble Methods
# ============================================================================

class TreeEnsemble:
    """Ensemble of decision trees"""

    def __init__(self, trees: List[ProofDecisionTree]):
        self.trees = trees
        self.weights: List[float] = [1.0] * len(trees)

    def majority_vote(
        self,
        context: ProofContext
    ) -> ProofResult:
        """Combine results via majority vote"""
        results = [tree.evaluate(context) for tree in self.trees]

        # Majority vote on success
        success_count = sum(1 for r in results if r.success)
        final_success = success_count > len(self.trees) / 2

        # Combine proof steps
        all_steps = []
        for result in results:
            all_steps.extend(result.proof_steps)

        # Average metrics
        avg_depth = statistics.mean(r.depth_reached for r in results)
        avg_time = statistics.mean(r.time_taken for r in results)

        return ProofResult(
            success=final_success,
            proof_steps=all_steps[:50],  # Limit steps
            final_state=context.current_state,
            depth_reached=int(avg_depth),
            time_taken=avg_time
        )

    def weighted_vote(
        self,
        context: ProofContext,
        weights: List[float] = None
    ) -> ProofResult:
        """Weighted voting"""
        if weights:
            self.weights = weights

        results = [tree.evaluate(context) for tree in self.trees]

        # Weighted success
        weighted_success = sum(
            w * (1.0 if r.success else 0.0)
            for w, r in zip(self.weights, results)
        )
        final_success = weighted_success > 0.5

        # Combine proof steps with weights
        all_steps = []
        for result, weight in zip(results, self.weights):
            if weight > 0.5:
                all_steps.extend(result.proof_steps)

        avg_depth = statistics.mean(r.depth_reached for r in results)
        avg_time = statistics.mean(r.time_taken for r in results)

        return ProofResult(
            success=final_success,
            proof_steps=all_steps[:50],
            final_state=context.current_state,
            depth_reached=int(avg_depth),
            time_taken=avg_time
        )

    def cascade(
        self,
        context: ProofContext
    ) -> ProofResult:
        """Try trees in sequence until success"""
        total_steps = []
        total_time = 0.0
        max_depth = 0

        for tree in self.trees:
            result = tree.evaluate(context)
            total_steps.extend(result.proof_steps)
            total_time += result.time_taken
            max_depth = max(max_depth, result.depth_reached)

            if result.success:
                return ProofResult(
                    success=True,
                    proof_steps=total_steps,
                    final_state=result.final_state,
                    depth_reached=max_depth,
                    time_taken=total_time
                )

        # All failed
        return ProofResult(
            success=False,
            proof_steps=total_steps,
            final_state=context.current_state,
            depth_reached=max_depth,
            time_taken=total_time
        )

    def bagging_predict(
        self,
        context: ProofContext,
        n_samples: int = None
    ) -> ProofResult:
        """Bootstrap aggregating prediction"""
        n_samples = n_samples or len(self.trees)

        # Sample trees with replacement
        sampled_trees = random.choices(self.trees, k=n_samples)
        results = [tree.evaluate(context) for tree in sampled_trees]

        # Aggregate
        success_count = sum(1 for r in results if r.success)
        final_success = success_count > n_samples / 2

        return ProofResult(
            success=final_success,
            proof_steps=results[0].proof_steps if results else [],
            final_state=context.current_state,
            depth_reached=int(statistics.mean(r.depth_reached for r in results) if results else 0),
            time_taken=statistics.mean(r.time_taken for r in results) if results else 0.0
        )


# ============================================================================
# Coevolution With LeanAide
# ============================================================================

class CoevolutionWithLeanAide:
    """Coevolution with formal verification"""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 30,
        simulations: int = 50
    ):
        self.population_size = population_size
        self.generations = generations
        self.simulations = simulations

        self.generator = TreeGenerator()
        self.evaluator = MCTreeEvaluator(simulations=simulations)

    async def coevolve_with_verification(
        self,
        test_theorems: List[str],
        leanaide_client=None,
        generations: int = 50
    ) -> ProofDecisionTree:
        """Coevolve with Lean formal verification"""
        print("Starting coevolution with Lean verification")

        # Initialize population
        population = self.generator.generate_ramped_half_and_half(
            self.population_size,
            15
        )

        best_tree = None
        best_fitness = 0.0

        for generation in range(generations):
            # Evaluate with verification
            fitness_scores = []

            for tree in population:
                # Monte Carlo evaluation
                mc_result = self.evaluator.evaluate(tree, test_theorems)

                # Formal verification bonus (simulated)
                verification_bonus = 0.0
                if leanaide_client and mc_result.success_rate > 0.5:
                    # In real implementation, would call LeanAide
                    # For now, simulate based on success rate
                    verification_bonus = mc_result.success_rate * 0.3

                # Combined fitness
                tree.fitness = mc_result.overall_fitness + verification_bonus
                tree.verified = verification_bonus > 0.0
                fitness_scores.append(tree.fitness)

            # Track best
            current_best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if fitness_scores[current_best_idx] > best_fitness:
                best_tree = population[current_best_idx].clone()
                best_fitness = fitness_scores[current_best_idx]
                print(f"  New best fitness: {best_fitness:.4f}")

            # Evolution
            population = self._evolve_population(population, fitness_scores)

            if generation % 10 == 0:
                avg_fitness = statistics.mean(fitness_scores)
                verified_count = sum(1 for t in population if getattr(t, 'verified', False))
                print(f"Generation {generation}: best={best_fitness:.4f}, avg={avg_fitness:.4f}, verified={verified_count}")

        print(f"Coevolution with verification complete. Best fitness: {best_fitness:.4f}")
        return best_tree

    def _evolve_population(
        self,
        population: List[ProofDecisionTree],
        scores: List[float]
    ) -> List[ProofDecisionTree]:
        """Evolve population using genetic operators"""
        # Sort by fitness
        sorted_pop = [
            (tree.clone(), score)
            for tree, score in zip(population, scores)
        ]
        sorted_pop.sort(key=lambda x: x[1], reverse=True)

        # Elitism
        new_pop = [tree for tree, _ in sorted_pop[:5]]

        # Generate offspring
        crossover = TreeCrossover()
        mutation = TreeMutation(self.generator)

        while len(new_pop) < self.population_size:
            # Tournament selection
            tournament = random.sample(sorted_pop[:self.population_size//2], k=3)
            parent1 = max(tournament, key=lambda x: x[1])[0]

            tournament = random.sample(sorted_pop[:self.population_size//2], k=3)
            parent2 = max(tournament, key=lambda x: x[1])[0]

            # Crossover
            if random.random() < 0.9:
                child1, child2 = crossover.subtree_crossover(parent1, parent2)
                new_pop.extend([child1, child2])
            else:
                new_pop.extend([parent1.clone(), parent2.clone()])

        # Mutation
        for i in range(5, len(new_pop)):
            if random.random() < 0.15:
                new_pop[i] = mutation.subtree_mutation(new_pop[i])

        return new_pop[:self.population_size]


# ============================================================================
# Tree Visualization
# ============================================================================

class TreeVisualizer:
    """Visualize decision trees"""

    def to_graphviz(self, tree: ProofDecisionTree) -> str:
        """Generate GraphViz representation"""
        lines = ["digraph DecisionTree {"]
        lines.append("  node [shape=box];")

        self._add_node_to_graphviz(tree.root, lines)

        lines.append("}")
        return "\n".join(lines)

    def _add_node_to_graphviz(
        self,
        node: DecisionNode,
        lines: List[str],
        parent_id: str = None,
        edge_label: str = ""
    ):
        """Add node to GraphViz output"""
        # Determine node style
        if node.node_type == NodeType.ACTION:
            style = "style=rounded,fillcolor=lightblue,"
        elif node.node_type == NodeType.CONDITION:
            style = "style=diamond,fillcolor=lightyellow,"
        else:
            style = ""

        # Create node label
        content_str = str(node.content)[:30] if node.content else node.node_type.value
        label = f"{node.node_type.value}\\n{content_str}"

        lines.append(f'  "{node.node_id}" [{style}label="{label}"];')

        # Add edge from parent
        if parent_id:
            lines.append(f'  "{parent_id}" -> "{node.node_id}" [label="{edge_label}"];')

        # Add children
        for i, child in enumerate(node.children):
            self._add_node_to_graphviz(child, lines, node.node_id, str(i))

    def to_text(self, tree: ProofDecisionTree) -> str:
        """Generate text representation"""
        lines = []
        lines.append(f"Decision Tree {tree.tree_id}")
        lines.append(f"Depth: {tree.depth}, Nodes: {tree.node_count}")
        lines.append(f"Fitness: {tree.fitness:.4f}")
        lines.append("")

        self._add_node_to_text(tree.root, lines, 0)

        return "\n".join(lines)

    def _add_node_to_text(
        self,
        node: DecisionNode,
        lines: List[str],
        depth: int
    ):
        """Add node to text representation"""
        indent = "  " * depth
        content_str = str(node.content)[:30] if node.content else "-"
        lines.append(f"{indent}{node.node_type.value}: {content_str}")

        for child in node.children:
            self._add_node_to_text(child, lines, depth + 1)

    def analyze_structure(
        self,
        tree: ProofDecisionTree
    ) -> TreeAnalysis:
        """Analyze tree structure"""
        nodes = tree.get_all_nodes()

        # Count node types
        type_counts = defaultdict(int)
        for node in nodes:
            type_counts[node.node_type.value] += 1

        # Calculate branching factor
        total_children = sum(len(node.children) for node in nodes)
        internal_nodes = [n for n in nodes if n.children]
        avg_branching = (total_children / len(internal_nodes)) if internal_nodes else 0.0

        # Calculate average subtree size
        subtree_sizes = [node.get_node_count() for node in nodes]
        avg_subtree_size = statistics.mean(subtree_sizes) if subtree_sizes else 0.0

        # Complexity score
        complexity = tree.node_count * (1 + tree.depth * 0.1)

        # Count leaves
        leaf_count = sum(1 for node in nodes if node.is_leaf())

        return TreeAnalysis(
            depth=tree.depth,
            node_count=tree.node_count,
            leaf_count=leaf_count,
            branching_factor=avg_branching,
            node_type_distribution=dict(type_counts),
            average_subtree_size=avg_subtree_size,
            complexity_score=complexity
        )


# ============================================================================
# Main Demo and Testing
# ============================================================================

async def demo_mcts_coevolution():
    """Demonstrate MCTS Coevolution"""
    print("=" * 80)
    print("MCTS COEVOLUTION DEMONSTRATION")
    print("=" * 80)

    # Sample theorems
    test_theorems = [
        "∀ n: Nat, n + 0 = n",
        "∀ a b: Nat, a + b = b + a",
        "∀ n: Nat, 2 * n = n + n",
        "∀ a b c: Nat, (a + b) + c = a + (b + c)",
        "∀ n: Nat, n ≤ n"
    ]

    print("\n1. BASIC TREE COEVOLUTION")
    print("-" * 40)

    # Basic coevolution
    basic_coevolution = TreeCoevolution(
        population_size=20,
        generations=10,
        simulations=20
    )

    best_tree = await basic_coevolution.coevolve(test_theorems)

    print(f"\nBest tree fitness: {best_tree.fitness:.4f}")
    print(f"Best tree depth: {best_tree.depth}")
    print(f"Best tree nodes: {best_tree.node_count}")

    # Visualize best tree
    visualizer = TreeVisualizer()
    print("\nBest tree structure:")
    print(visualizer.to_text(best_tree))

    print("\n2. COMPETITIVE COEVOLUTION")
    print("-" * 40)

    competitive = CompetitiveCoevolution(
        solver_pop_size=30,
        problem_pop_size=10,
        generations=20
    )

    best_solver = await competitive.competitive_coevolve(test_theorems)

    print(f"\nBest solver fitness: {best_solver.fitness:.4f}")

    print("\n3. MULTI-OBJECTIVE COEVOLUTION")
    print("-" * 40)

    multi_obj = MultiObjectiveCoevolution(
        objectives=["success", "elegance", "simplicity"],
        population_size=20,
        generations=10
    )

    pareto_front = await multi_obj.coevolve_multi_objective(test_theorems)

    print(f"\nPareto front size: {len(pareto_front)}")
    print("Pareto front solutions:")
    for i, tree in enumerate(pareto_front[:5]):
        obj = getattr(tree, 'objectives', {})
        print(f"  Solution {i+1}: success={obj.get('success', 0):.2f}, "
              f"elegance={obj.get('elegance', 0):.2f}, "
              f"simplicity={obj.get('simplicity', 0):.2f}")

    print("\n4. TREE ENSEMBLE")
    print("-" * 40)

    ensemble = TreeEnsemble(pareto_front[:5])

    test_context = ProofContext(
        theorem="∀ n: Nat, n + 0 = n",
        goal_state="prove addition identity",
        max_depth=50
    )

    result = ensemble.majority_vote(test_context)
    print(f"Ensemble result: success={result.success}, "
          f"depth={result.depth_reached}, time={result.time_taken:.3f}s")

    print("\n5. TREE PRUNING")
    print("-" * 40)

    pruner = TreePruner()
    pruned_tree = pruner.prune_dead_branches(best_tree, test_theorems)

    analysis_before = visualizer.analyze_structure(best_tree)
    analysis_after = visualizer.analyze_structure(pruned_tree)

    print(f"Before pruning: nodes={analysis_before.node_count}, depth={analysis_before.depth}")
    print(f"After pruning: nodes={analysis_after.node_count}, depth={analysis_after.depth}")
    print(f"Reduction: {analysis_before.node_count - analysis_after.node_count} nodes")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


async def demo_leanaide_integration():
    """Demonstrate LeanAide integration"""
    print("\n" + "=" * 80)
    print("LEANAIDE INTEGRATION DEMONSTRATION")
    print("=" * 80)

    test_theorems = [
        "∀ n: Nat, n + 0 = n",
        "∀ a b: Nat, a + b = b + a"
    ]

    # Coevolution with verification
    coevolution = CoevolutionWithLeanAide(
        population_size=15,
        generations=5,
        simulations=30
    )

    best_tree = await coevolution.coevolve_with_verification(
        test_theorems,
        leanaide_client=None  # Would pass actual LeanAide client
    )

    print(f"\nBest tree fitness (with verification): {best_tree.fitness:.4f}")

    print("\n" + "=" * 80)


def save_tree_to_file(tree: ProofDecisionTree, filename: str):
    """Save tree to file"""
    visualizer = TreeVisualizer()

    with open(filename, 'w') as f:
        f.write("# Decision Tree\n\n")
        f.write(f"Tree ID: {tree.tree_id}\n")
        f.write(f"Depth: {tree.depth}\n")
        f.write(f"Nodes: {tree.node_count}\n")
        f.write(f"Fitness: {tree.fitness:.4f}\n\n")

        f.write("## Structure\n\n")
        f.write(visualizer.to_text(tree))

        f.write("\n## GraphViz\n\n")
        f.write("```dot\n")
        f.write(visualizer.to_graphviz(tree))
        f.write("\n```")

        f.write("\n## Analysis\n\n")
        analysis = visualizer.analyze_structure(tree)
        f.write(f"Depth: {analysis.depth}\n")
        f.write(f"Node count: {analysis.node_count}\n")
        f.write(f"Leaf count: {analysis.leaf_count}\n")
        f.write(f"Branching factor: {analysis.branching_factor:.2f}\n")
        f.write(f"Node type distribution: {analysis.node_type_distribution}\n")
        f.write(f"Average subtree size: {analysis.average_subtree_size:.2f}\n")
        f.write(f"Complexity score: {analysis.complexity_score:.2f}\n")

    print(f"Tree saved to {filename}")


def load_tree_from_file(filename: str) -> Optional[ProofDecisionTree]:
    """Load tree from file (basic implementation)"""
    # This would need proper serialization/deserialization
    # For now, just a placeholder
    print(f"Load from {filename} - not fully implemented")
    return None


# ============================================================================
# Performance Benchmarks
# ============================================================================

async def benchmark_coevolution():
    """Benchmark coevolution performance"""
    print("\n" + "=" * 80)
    print("COEVOLUTION BENCHMARKS")
    print("=" * 80)

    test_theorems = [
        "∀ n: Nat, n + 0 = n",
        "∀ a b: Nat, a + b = b + a",
        "∀ n: Nat, 2 * n = n + n"
    ]

    configs = [
        {"pop_size": 10, "generations": 5, "sims": 10},
        {"pop_size": 20, "generations": 10, "sims": 20},
        {"pop_size": 30, "generations": 15, "sims": 30},
    ]

    results = []

    for config in configs:
        print(f"\nConfig: pop={config['pop_size']}, gen={config['generations']}, sims={config['sims']}")

        start_time = time.time()

        coevolution = TreeCoevolution(
            population_size=config['pop_size'],
            generations=config['generations'],
            simulations=config['sims']
        )

        best_tree = await coevolution.coevolve(test_theorems)

        elapsed = time.time() - start_time

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Best fitness: {best_tree.fitness:.4f}")

        results.append({
            'config': config,
            'time': elapsed,
            'fitness': best_tree.fitness
        })

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for result in results:
        cfg = result['config']
        print(f"pop={cfg['pop_size']}, gen={cfg['generations']}, sims={cfg['sims']}: "
              f"time={result['time']:.2f}s, fitness={result['fitness']:.4f}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    print("MCTS Coevolution Module")
    print("=" * 80)

    # Run demonstrations
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        asyncio.run(benchmark_coevolution())
    elif len(sys.argv) > 1 and sys.argv[1] == "leanaide":
        asyncio.run(demo_leanaide_integration())
    else:
        asyncio.run(demo_mcts_coevolution())
