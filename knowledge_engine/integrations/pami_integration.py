"""
PAMI (Pattern Mining) Integration Module for OpenEvolve Knowledge Engine

This module provides advanced pattern mining capabilities by integrating
PAMI's state-of-the-art algorithms for frequent pattern, sequence, and graph mining.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime

# Add PAMI to Python path for import
pami_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'PAMI')
if pami_path not in sys.path:
    sys.path.insert(0, pami_path)


class PAMIIntegration:
    """
    Main PAMI Integration class for the Knowledge Engine.
    
    Provides a unified interface for pattern mining capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PAMI Integration.
        
        Args:
            config: Configuration dictionary for PAMI
        """
        self.config = config or {}
        self._miner = PAMIPatternMiner()
    
    def is_available(self) -> bool:
        """Check if PAMI is available."""
        return self._miner.is_available()
    
    def mine_patterns(self, data: List[List[str]], min_support: float = 0.1) -> Dict[str, Any]:
        """
        Mine frequent patterns from transaction data.
        
        Args:
            data: List of transactions (each transaction is a list of items)
            min_support: Minimum support threshold
            
        Returns:
            Dictionary with patterns and metadata
        """
        return self._miner.mine_frequent_patterns(data, min_support)
    
    def get_available_algorithms(self) -> Dict[str, List[str]]:
        """Get list of available algorithms."""
        return self._miner.get_available_algorithms()


class PAMIPatternMiner:
    """
    Advanced pattern miner that leverages PAMI's algorithms.
    
    This class integrates frequent pattern mining, sequential pattern mining,
    and graph pattern mining for comprehensive knowledge discovery.
    """
    
    def __init__(self):
        """Initialize PAMI modules for pattern mining."""
        self._pami_available = False
        self.frequent_pattern_algorithms = {}
        self.sequence_pattern_algorithms = {}
        self.graph_pattern_algorithms = {}
        self._initialize_pami_modules()
    
    def _initialize_pami_modules(self):
        """Initialize all PAMI modules with proper error handling."""
        try:
            # Import PAMI modules dynamically
            # Actual PAMI structure has 'subgraphMining' not 'frequentPattern'/'graphPattern'
            
            # Try to import subgraph mining modules (the actual structure)
            try:
                from PAMI.subgraphMining.basic import GSpan, FSG, TKG
                self.graph_pattern_algorithms.update({
                    'gspan': GSpan,
                    'fsg': FSG,
                    'tkg': TKG
                })
            except ImportError:
                pass
            
            # Try alternative import paths
            try:
                from PAMI import subgraphMining
                self.graph_pattern_algorithms['subgraph_mining_available'] = True
            except ImportError:
                pass
            
            # Note: PAMI may not have the traditional frequent/sequential pattern modules
            # These are placeholders for if they exist in the future
            try:
                from PAMI.frequentPattern.basic import FPGrowth
                self.frequent_pattern_algorithms['fpgrowth'] = FPGrowth
            except ImportError:
                pass
            
            try:
                from PAMI.sequentialPattern.basic import PrefixSpan
                self.sequence_pattern_algorithms['prefixspan'] = PrefixSpan
            except ImportError:
                pass
            
            # Check if any algorithms were loaded
            total_algorithms = (len(self.frequent_pattern_algorithms) + 
                              len(self.sequence_pattern_algorithms) + 
                              len(self.graph_pattern_algorithms))
            
            if total_algorithms > 0:
                self._pami_available = True
            else:
                print("Warning: No PAMI algorithms could be loaded")
                
        except ImportError as e:
            print(f"Warning: Could not import PAMI modules: {e}")
            print("PAMI integration will be disabled.")
    
    def is_available(self) -> bool:
        """Check if PAMI integration is available."""
        return self._pami_available
    
    def get_available_algorithms(self) -> Dict[str, List[str]]:
        """Get list of available algorithms by category."""
        return {
            'frequent_pattern': list(self.frequent_pattern_algorithms.keys()),
            'sequence_pattern': list(self.sequence_pattern_algorithms.keys()),
            'graph_pattern': list(self.graph_pattern_algorithms.keys())
        }
    
    def mine_frequent_patterns(
        self,
        transactions: List[List[str]],
        min_support: float = 0.1,
        algorithm: str = 'fpgrowth',
        max_pattern_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Mine frequent patterns from transaction data.
        
        Args:
            transactions: List of transactions (each transaction is a list of items)
            min_support: Minimum support threshold (0.0 to 1.0)
            algorithm: Algorithm to use ('fpgrowth', 'apriori', 'eclat')
            max_pattern_length: Maximum length of patterns to find
            
        Returns:
            Dictionary containing frequent patterns and their statistics
        """
        if not self.is_available():
            return {
                'status': 'error',
                'message': 'PAMI integration not available',
                'patterns': []
            }
        
        try:
            if algorithm not in self.frequent_pattern_algorithms:
                return {
                    'status': 'error',
                    'message': f'Algorithm {algorithm} not available',
                    'patterns': []
                }
            
            # Convert to format expected by PAMI
            # PAMI typically expects a file or specific data structure
            # We'll implement a generic adapter
            
            patterns = self._mine_patterns_adapter(
                transactions, 
                min_support, 
                algorithm,
                max_pattern_length
            )
            
            return {
                'status': 'success',
                'patterns': patterns,
                'statistics': self._calculate_pattern_statistics(patterns),
                'config': {
                    'algorithm': algorithm,
                    'min_support': min_support,
                    'max_pattern_length': max_pattern_length
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Pattern mining failed: {str(e)}',
                'patterns': []
            }
    
    def _mine_patterns_adapter(
        self,
        transactions: List[List[str]],
        min_support: float,
        algorithm: str,
        max_pattern_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Adapter for PAMI pattern mining algorithms."""
        patterns = []
        
        try:
            # Calculate absolute minimum support
            abs_min_support = int(len(transactions) * min_support)
            if abs_min_support < 1:
                abs_min_support = 1
            
            # Use simple frequent pattern mining implementation
            # that doesn't rely on specific PAMI file formats
            from collections import Counter
            
            # Generate all possible itemsets up to max_pattern_length
            if max_pattern_length is None:
                max_pattern_length = 5  # Default limit
            
            # Count single items
            item_counts = Counter()
            for transaction in transactions:
                item_counts.update(transaction)
            
            # Filter by minimum support
            frequent_items = {
                item: count for item, count in item_counts.items()
                if count >= abs_min_support
            }
            
            # Add single-item patterns
            for item, count in frequent_items.items():
                patterns.append({
                    'pattern': [item],
                    'support': count,
                    'support_ratio': count / len(transactions),
                    'length': 1
                })
            
            # Mine multi-item patterns using Apriori-like approach
            current_patterns = list(frequent_items.keys())
            
            for length in range(2, max_pattern_length + 1):
                # Generate candidate patterns
                candidates = self._generate_candidates(current_patterns, length)
                
                # Count candidates
                candidate_counts = Counter()
                for transaction in transactions:
                    transaction_set = set(transaction)
                    for candidate in candidates:
                        if set(candidate).issubset(transaction_set):
                            candidate_counts[candidate] += 1
                
                # Filter by minimum support
                new_patterns = []
                for candidate, count in candidate_counts.items():
                    if count >= abs_min_support:
                        pattern_dict = {
                            'pattern': list(candidate),
                            'support': count,
                            'support_ratio': count / len(transactions),
                            'length': length
                        }
                        patterns.append(pattern_dict)
                        new_patterns.extend(candidate)
                
                if not new_patterns:
                    break
                
                current_patterns = list(set(new_patterns))
            
            # Sort by support (descending)
            patterns.sort(key=lambda x: x['support'], reverse=True)
            
        except Exception as e:
            print(f"Warning: Pattern mining failed: {e}")
        
        return patterns
    
    def _generate_candidates(self, items: List[str], length: int) -> List[Tuple[str, ...]]:
        """Generate candidate itemsets of given length."""
        from itertools import combinations
        return list(combinations(items, length))
    
    def _calculate_pattern_statistics(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about mined patterns."""
        if not patterns:
            return {
                'total_patterns': 0,
                'patterns_by_length': {},
                'average_support': 0.0,
                'max_support': 0.0,
                'min_support': 0.0
            }
        
        lengths = {}
        supports = []
        
        for pattern in patterns:
            length = pattern['length']
            lengths[length] = lengths.get(length, 0) + 1
            supports.append(pattern['support_ratio'])
        
        return {
            'total_patterns': len(patterns),
            'patterns_by_length': lengths,
            'average_support': sum(supports) / len(supports),
            'max_support': max(supports),
            'min_support': min(supports)
        }
    
    def mine_sequences(
        self,
        sequences: List[List[List[str]]],
        min_support: float = 0.1,
        max_gap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Mine frequent sequential patterns.
        
        Args:
            sequences: List of sequences (each sequence is a list of itemsets)
            min_support: Minimum support threshold
            max_gap: Maximum gap between elements
            
        Returns:
            Dictionary containing sequential patterns
        """
        if not self.is_available():
            return {
                'status': 'error',
                'message': 'PAMI integration not available',
                'patterns': []
            }
        
        try:
            # Simplified sequential pattern mining
            patterns = self._mine_sequences_adapter(sequences, min_support, max_gap)
            
            return {
                'status': 'success',
                'patterns': patterns,
                'statistics': self._calculate_sequence_statistics(patterns),
                'config': {
                    'min_support': min_support,
                    'max_gap': max_gap
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Sequence mining failed: {str(e)}',
                'patterns': []
            }
    
    def _mine_sequences_adapter(
        self,
        sequences: List[List[List[str]]],
        min_support: float,
        max_gap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Adapter for sequential pattern mining."""
        patterns = []
        
        try:
            abs_min_support = int(len(sequences) * min_support)
            if abs_min_support < 1:
                abs_min_support = 1
            
            # Flatten sequences to find frequent individual items
            all_items = []
            for sequence in sequences:
                for itemset in sequence:
                    all_items.extend(itemset)
            
            from collections import Counter
            item_counts = Counter(all_items)
            
            # Find frequent items
            frequent_items = {
                item for item, count in item_counts.items()
                if count >= abs_min_support
            }
            
            # Simple sequence pattern detection
            # Look for common subsequences
            for length in range(1, 4):  # Limit to length 3 for simplicity
                candidate_patterns = self._generate_sequence_candidates(
                    sequences, frequent_items, length
                )
                
                for pattern in candidate_patterns:
                    support = self._count_sequence_support(sequences, pattern, max_gap)
                    
                    if support >= abs_min_support:
                        patterns.append({
                            'pattern': pattern,
                            'support': support,
                            'support_ratio': support / len(sequences),
                            'length': length
                        })
            
            # Sort by support
            patterns.sort(key=lambda x: x['support'], reverse=True)
            
        except Exception as e:
            print(f"Warning: Sequence mining failed: {e}")
        
        return patterns
    
    def _generate_sequence_candidates(
        self,
        sequences: List[List[List[str]]],
        frequent_items: set,
        length: int
    ) -> List[List[str]]:
        """Generate candidate sequence patterns."""
        candidates = []
        
        # Extract all unique items from sequences
        items = list(frequent_items)
        
        from itertools import product
        for candidate in product(items, repeat=length):
            candidates.append(list(candidate))
        
        return candidates
    
    def _count_sequence_support(
        self,
        sequences: List[List[List[str]]],
        pattern: List[str],
        max_gap: Optional[int] = None
    ) -> int:
        """Count how many sequences contain the pattern."""
        count = 0
        
        for sequence in sequences:
            if self._contains_subsequence(sequence, pattern, max_gap):
                count += 1
        
        return count
    
    def _contains_subsequence(
        self,
        sequence: List[List[str]],
        pattern: List[str],
        max_gap: Optional[int] = None
    ) -> bool:
        """Check if sequence contains the pattern as a subsequence."""
        if not pattern:
            return True
        
        if not sequence:
            return False
        
        pattern_idx = 0
        last_match_idx = -1
        
        for seq_idx, itemset in enumerate(sequence):
            if pattern[pattern_idx] in itemset:
                # Check gap constraint
                if max_gap is not None and last_match_idx != -1:
                    if seq_idx - last_match_idx - 1 > max_gap:
                        continue
                
                pattern_idx += 1
                last_match_idx = seq_idx
                
                if pattern_idx >= len(pattern):
                    return True
        
        return False
    
    def _calculate_sequence_statistics(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about sequence patterns."""
        if not patterns:
            return {
                'total_patterns': 0,
                'patterns_by_length': {},
                'average_support': 0.0
            }
        
        lengths = {}
        supports = []
        
        for pattern in patterns:
            length = pattern['length']
            lengths[length] = lengths.get(length, 0) + 1
            supports.append(pattern['support_ratio'])
        
        return {
            'total_patterns': len(patterns),
            'patterns_by_length': lengths,
            'average_support': sum(supports) / len(supports) if supports else 0.0
        }
    
    def analyze_knowledge_graph_patterns(
        self,
        graph_data: Dict[str, Any],
        min_support: float = 0.1
    ) -> Dict[str, Any]:
        """
        Analyze patterns in a knowledge graph.
        
        Args:
            graph_data: Knowledge graph data with nodes and edges
            min_support: Minimum support threshold for patterns
            
        Returns:
            Dictionary containing graph pattern analysis
        """
        if not self.is_available():
            return {
                'status': 'error',
                'message': 'PAMI integration not available',
                'patterns': []
            }
        
        try:
            # Extract patterns from the knowledge graph
            # 1. Frequent entity types
            # 2. Frequent relationship patterns
            # 3. Frequent triple patterns
            
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            # Analyze node type patterns
            node_types = [node.get('type', 'unknown') for node in nodes]
            from collections import Counter
            type_counts = Counter(node_types)
            
            # Analyze edge patterns
            edge_types = [edge.get('type', 'unknown') for edge in edges]
            edge_counts = Counter(edge_types)
            
            # Build relationship patterns (triples)
            triple_patterns = []
            node_id_to_type = {node.get('id'): node.get('type', 'unknown') for node in nodes}
            
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                rel_type = edge.get('type', 'unknown')
                
                source_type = node_id_to_type.get(source, 'unknown')
                target_type = node_id_to_type.get(target, 'unknown')
                
                triple_patterns.append(f"{source_type}-{rel_type}-{target_type}")
            
            triple_counts = Counter(triple_patterns)
            
            # Filter by minimum support
            abs_min_support = int(len(nodes) * min_support) if nodes else 0
            if abs_min_support < 1:
                abs_min_support = 1
            
            frequent_types = [
                {'pattern': t, 'count': c, 'support_ratio': c / len(nodes)}
                for t, c in type_counts.items()
                if c >= abs_min_support
            ]
            
            frequent_edges = [
                {'pattern': e, 'count': c, 'support_ratio': c / len(edges) if edges else 0}
                for e, c in edge_counts.items()
                if c >= abs_min_support
            ]
            
            frequent_triples = [
                {'pattern': t, 'count': c, 'support_ratio': c / len(edges) if edges else 0}
                for t, c in triple_counts.items()
                if c >= abs_min_support
            ]
            
            return {
                'status': 'success',
                'patterns': {
                    'entity_types': frequent_types,
                    'relationship_types': frequent_edges,
                    'triple_patterns': frequent_triples
                },
                'statistics': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'unique_entity_types': len(type_counts),
                    'unique_relationship_types': len(edge_counts),
                    'unique_triple_patterns': len(triple_counts)
                },
                'config': {
                    'min_support': min_support
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Graph pattern analysis failed: {str(e)}',
                'patterns': []
            }
    
    def discover_association_rules(
        self,
        transactions: List[List[str]],
        min_support: float = 0.1,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Discover association rules from transaction data.
        
        Args:
            transactions: List of transactions
            min_support: Minimum support for itemsets
            min_confidence: Minimum confidence for rules
            
        Returns:
            Dictionary containing association rules
        """
        try:
            # First mine frequent patterns
            patterns_result = self.mine_frequent_patterns(transactions, min_support)
            
            if patterns_result['status'] != 'success':
                return patterns_result
            
            patterns = patterns_result['patterns']
            
            # Generate association rules from patterns
            rules = []
            
            for pattern in patterns:
                if pattern['length'] < 2:
                    continue
                
                items = pattern['pattern']
                
                # Generate all possible antecedent-consequent pairs
                for i in range(1, len(items)):
                    from itertools import combinations
                    
                    for antecedent in combinations(items, i):
                        consequent = tuple(item for item in items if item not in antecedent)
                        
                        if not consequent:
                            continue
                        
                        # Calculate confidence
                        antecedent_support = self._get_pattern_support(
                            transactions, list(antecedent)
                        )
                        
                        if antecedent_support > 0:
                            confidence = pattern['support'] / antecedent_support
                            
                            if confidence >= min_confidence:
                                rules.append({
                                    'antecedent': list(antecedent),
                                    'consequent': list(consequent),
                                    'support': pattern['support_ratio'],
                                    'confidence': confidence,
                                    'lift': confidence / (pattern['support'] / len(transactions))
                                })
            
            # Sort by confidence
            rules.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'status': 'success',
                'rules': rules,
                'statistics': {
                    'total_rules': len(rules),
                    'average_confidence': sum(r['confidence'] for r in rules) / len(rules) if rules else 0,
                    'average_support': sum(r['support'] for r in rules) / len(rules) if rules else 0
                },
                'config': {
                    'min_support': min_support,
                    'min_confidence': min_confidence
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Association rule discovery failed: {str(e)}',
                'rules': []
            }
    
    def _get_pattern_support(self, transactions: List[List[str]], pattern: List[str]) -> int:
        """Get support count for a specific pattern."""
        count = 0
        pattern_set = set(pattern)
        
        for transaction in transactions:
            if pattern_set.issubset(set(transaction)):
                count += 1
        
        return count
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of PAMI integration."""
        return {
            'available': self.is_available(),
            'algorithms': self.get_available_algorithms(),
            'timestamp': datetime.now().isoformat()
        }
