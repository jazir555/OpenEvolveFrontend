"""Conversation tree data structures for DTS.

Nodes represent conversation states, edges represent strategies.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConversationNode:
    """State in conversation tree.
    
    Attributes:
        message: The conversation message content
        speaker: Who spoke ('user' or 'system')
        depth: Tree depth level
        score: Quality score (0-10)
        parent: Parent node in tree
        children: Child nodes
        metadata: Additional data (timestamp, tokens, cost)
        node_id: Unique identifier
    """
    message: str
    speaker: str = "system"  # 'user' or 'system'
    depth: int = 0
    score: float = 0.0
    parent: Optional['ConversationNode'] = None
    children: List['ConversationNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Initialize metadata with defaults if not provided."""
        if not self.metadata:
            self.metadata = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tokens": 0,
                "cost": 0.0,
            }
    
    def add_child(self, message: str, speaker: str = "system", 
                  score: float = 0.0, metadata: Optional[Dict[str, Any]] = None) -> 'ConversationNode':
        """Add a child node to this node.
        
        Args:
            message: The child message content
            speaker: Who spoke ('user' or 'system')
            score: Initial quality score
            metadata: Additional metadata
            
        Returns:
            The newly created child node
        """
        child = ConversationNode(
            message=message,
            speaker=speaker,
            depth=self.depth + 1,
            score=score,
            parent=self,
            metadata=metadata or {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tokens": 0,
                "cost": 0.0,
            }
        )
        self.children.append(child)
        return child
    
    def get_path(self) -> List['ConversationNode']:
        """Get path from root to this node.
        
        Returns:
            List of nodes from root to this node
        """
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of messages.
        
        Returns:
            List of dicts with 'speaker' and 'message' keys
        """
        path = self.get_path()
        return [{"speaker": n.speaker, "message": n.message} for n in path]
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def get_siblings(self) -> List['ConversationNode']:
        """Get sibling nodes (excluding self)."""
        if self.parent is None:
            return []
        return [c for c in self.parent.children if c.node_id != self.node_id]
    
    def update_score(self, score: float, backpropagate: bool = False) -> None:
        """Update node score.
        
        Args:
            score: New score value
            backpropagate: Whether to update parent scores
        """
        self.score = score
        if backpropagate and self.parent is not None:
            # Simple backpropagation: parent gets average of children
            child_scores = [c.score for c in self.parent.children]
            if child_scores:
                self.parent.update_score(sum(child_scores) / len(child_scores), backpropagate=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "node_id": self.node_id,
            "message": self.message,
            "speaker": self.speaker,
            "depth": self.depth,
            "score": self.score,
            "metadata": self.metadata,
            "children_count": len(self.children),
            "parent_id": self.parent.node_id if self.parent else None,
        }
    
    def __repr__(self) -> str:
        return f"ConversationNode(id={self.node_id[:8]}, depth={self.depth}, score={self.score:.2f})"


@dataclass
class ConversationTree:
    """Tree container for conversation nodes.
    
    Attributes:
        root: Root node of the tree
        branches: All paths from root to leaves
        tree_id: Unique identifier
        metadata: Tree-level metadata
    """
    root: ConversationNode
    tree_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata with defaults."""
        if not self.metadata:
            self.metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_nodes": 1,
                "max_depth": 0,
            }
    
    def add_node(self, parent: ConversationNode, message: str, 
                 speaker: str = "system", score: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None) -> ConversationNode:
        """Add a node to the tree.
        
        Args:
            parent: Parent node to attach to
            message: Message content
            speaker: Who spoke
            score: Initial score
            metadata: Node metadata
            
        Returns:
            The newly created node
        """
        child = parent.add_child(message, speaker, score, metadata)
        self.metadata["total_nodes"] = self.metadata.get("total_nodes", 0) + 1
        self.metadata["max_depth"] = max(self.metadata.get("max_depth", 0), child.depth)
        return child
    
    def get_branches(self) -> List[List[ConversationNode]]:
        """Get all paths from root to leaves.
        
        Returns:
            List of paths (each path is a list of nodes)
        """
        branches = []
        
        def traverse(node: ConversationNode, current_path: List[ConversationNode]):
            current_path = current_path + [node]
            if node.is_leaf():
                branches.append(current_path)
            else:
                for child in node.children:
                    traverse(child, current_path)
        
        traverse(self.root, [])
        return branches
    
    def get_leaves(self) -> List[ConversationNode]:
        """Get all leaf nodes."""
        leaves = []
        
        def traverse(node: ConversationNode):
            if node.is_leaf():
                leaves.append(node)
            else:
                for child in node.children:
                    traverse(child)
        
        traverse(self.root)
        return leaves
    
    def get_all_nodes(self) -> List[ConversationNode]:
        """Get all nodes in the tree."""
        nodes = []
        
        def traverse(node: ConversationNode):
            nodes.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(self.root)
        return nodes
    
    def get_path(self, node: ConversationNode) -> List[ConversationNode]:
        """Get path from root to specified node."""
        return node.get_path()
    
    def prune(self, threshold: float, keep_best_n: Optional[int] = None) -> int:
        """Prune branches below threshold score.
        
        Args:
            threshold: Minimum score to keep
            keep_best_n: If set, keep only N best branches regardless of threshold
            
        Returns:
            Number of nodes pruned
        """
        pruned_count = 0
        
        def should_prune(node: ConversationNode) -> bool:
            """Check if node should be pruned."""
            if node == self.root:
                return False
            return node.score < threshold
        
        def prune_recursive(node: ConversationNode) -> bool:
            """Prune from node downwards. Returns True if node was pruned."""
            nonlocal pruned_count
            
            if should_prune(node):
                if node.parent:
                    node.parent.children.remove(node)
                pruned_count += 1 + count_descendants(node)
                return True
            
            # Prune children
            children_to_remove = []
            for child in node.children:
                if prune_recursive(child):
                    children_to_remove.append(child)
            
            for child in children_to_remove:
                if child in node.children:
                    node.children.remove(child)
            
            return False
        
        def count_descendants(node: ConversationNode) -> int:
            """Count total descendants."""
            count = len(node.children)
            for child in node.children:
                count += count_descendants(child)
            return count
        
        # If keep_best_n is set, keep only best N leaves
        if keep_best_n is not None:
            leaves = self.get_leaves()
            if len(leaves) > keep_best_n:
                leaves.sort(key=lambda n: n.score, reverse=True)
                keep_ids = {n.node_id for n in leaves[:keep_best_n]}
                
                def prune_by_id(node: ConversationNode) -> bool:
                    nonlocal pruned_count
                    if node == self.root:
                        return False
                    if node.node_id not in keep_ids and node.node_id not in {
                        desc.node_id for leaf in leaves[:keep_best_n] 
                        for desc in leaf.get_path()
                    }:
                        if node.parent and node in node.parent.children:
                            node.parent.children.remove(node)
                        pruned_count += 1 + count_descendants(node)
                        return True
                    return False
                
                for node in self.get_all_nodes():
                    if node != self.root:
                        prune_by_id(node)
        else:
            # Prune by threshold
            for node in self.get_all_nodes():
                if node != self.root:
                    prune_recursive(node)
        
        self.metadata["pruned_count"] = self.metadata.get("pruned_count", 0) + pruned_count
        return pruned_count
    
    def backpropagate(self, leaf_node: ConversationNode) -> None:
        """Backpropagate score from leaf to root.
        
        Uses average of siblings to update parent scores.
        
        Args:
            leaf_node: Leaf node to backpropagate from
        """
        path = leaf_node.get_path()
        
        # Update scores from leaf to root
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            if node.children:
                # Average of children scores
                node.score = sum(c.score for c in node.children) / len(node.children)
    
    def get_best_path(self) -> Optional[List[ConversationNode]]:
        """Get the highest-scoring path from root to leaf.
        
        Returns:
            Best scoring path or None if tree is empty
        """
        branches = self.get_branches()
        if not branches:
            return None
        
        # Score is average of all nodes in path
        def path_score(path: List[ConversationNode]) -> float:
            return sum(n.score for n in path) / len(path)
        
        return max(branches, key=path_score)
    
    def get_best_leaf(self) -> Optional[ConversationNode]:
        """Get the highest-scoring leaf node."""
        leaves = self.get_leaves()
        if not leaves:
            return None
        return max(leaves, key=lambda n: n.score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation."""
        def node_to_dict(node: ConversationNode) -> Dict[str, Any]:
            return {
                **node.to_dict(),
                "children": [node_to_dict(c) for c in node.children]
            }
        
        return {
            "tree_id": self.tree_id,
            "metadata": self.metadata,
            "root": node_to_dict(self.root),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tree statistics."""
        nodes = self.get_all_nodes()
        leaves = self.get_leaves()
        branches = self.get_branches()
        
        scores = [n.score for n in nodes]
        
        return {
            "total_nodes": len(nodes),
            "total_leaves": len(leaves),
            "total_branches": len(branches),
            "max_depth": max((n.depth for n in nodes), default=0),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "best_score": max(scores) if scores else 0.0,
            "worst_score": min(scores) if scores else 0.0,
            "pruned_count": self.metadata.get("pruned_count", 0),
        }


@dataclass
class StrategyGenerator:
    """Generate conversation strategies.
    
    Generates diverse conversation approaches for a given context.
    """
    llm_client: Optional[Any] = None
    diversity_constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize diversity constraints."""
        if not self.diversity_constraints:
            self.diversity_constraints = {
                "min_diversity_score": 0.7,
                "max_similarity": 0.8,
                "required_styles": ["direct", "empathetic", "analytical"],
            }
    
    def generate_strategies(self, context: str, n: int = 5) -> List[str]:
        """Generate N diverse conversation strategies.
        
        Args:
            context: Conversation context/goal
            n: Number of strategies to generate
            
        Returns:
            List of strategy descriptions
        """
        # Default strategies if no LLM available
        default_strategies = [
            f"Direct approach: Address the goal '{context}' straightforwardly",
            f"Empathetic approach: Show understanding before addressing '{context}'",
            f"Analytical approach: Break down '{context}' into components",
            f"Questioning approach: Ask clarifying questions about '{context}'",
            f"Storytelling approach: Use narrative to explain '{context}'",
            f"Comparative approach: Compare '{context}' to known examples",
            f"Action-oriented approach: Focus on immediate steps for '{context}'",
            f"Educational approach: Teach concepts related to '{context}'",
        ]
        
        strategies = default_strategies[:n]
        
        # If LLM available, generate custom strategies
        if self.llm_client is not None:
            try:
                custom = self._generate_with_llm(context, n)
                if custom:
                    strategies = custom
            except Exception as e:
                logger.warning(f"LLM strategy generation failed: {e}, using defaults")
        
        return strategies[:n]
    
    def _generate_with_llm(self, context: str, n: int) -> Optional[List[str]]:
        """Generate strategies using LLM."""
        # Placeholder for LLM integration
        # In actual implementation, this would call the LLM client
        return None
    
    def ensure_diversity(self, strategies: List[str]) -> List[str]:
        """Ensure strategies are diverse using constraints.
        
        Args:
            strategies: List of strategy strings
            
        Returns:
            Filtered list ensuring diversity
        """
        if len(strategies) <= 1:
            return strategies
        
        diverse = [strategies[0]]
        
        for strategy in strategies[1:]:
            # Simple diversity check: not too similar to existing
            is_diverse = True
            for existing in diverse:
                similarity = self._calculate_similarity(strategy, existing)
                if similarity > self.diversity_constraints.get("max_similarity", 0.8):
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse.append(strategy)
        
        return diverse
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple text similarity."""
        # Simple Jaccard similarity on words
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)


# Type hint for Any
from typing import Any
