"""
Unified Knowledge Extraction Integrator

This module integrates all knowledge extraction capabilities including:
- Generic Knowledge Extraction Tool
- Karate Club (graph analysis)
- PAMI (pattern mining)
- NeuralKG (KG embeddings)
- Causal-Learn (causal discovery)
- Lagrange-Mapper (topological analysis)

Provides a unified interface for the Generic Knowledge Extraction Tool to leverage
all integrated knowledge graph capabilities.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

# Add paths for all integrations
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for path in [
    os.path.join(base_path, 'karateclub'),
    os.path.join(base_path, 'PAMI'),
    os.path.join(base_path, 'NeuralKG', 'src'),
    os.path.join(base_path, 'causal-learn'),
    os.path.join(base_path, 'lagrange-mapper'),
    os.path.join(base_path, 'Generic-Knowledge-Extraction-Tool')
]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import integration modules
try:
    from .karateclub_integration import KarateClubGraphAnalyzer
except ImportError:
    KarateClubGraphAnalyzer = None

try:
    from .pami_integration import PAMIPatternMiner
except ImportError:
    PAMIPatternMiner = None

try:
    from .neuralkg_integration import NeuralKGEmbedder
except ImportError:
    NeuralKGEmbedder = None

try:
    from .causal_learn_integration import CausalDiscoveryEngine
except ImportError:
    CausalDiscoveryEngine = None

try:
    from .lagrange_mapper_integration import LagrangeAttractorAnalyzer
except ImportError:
    LagrangeAttractorAnalyzer = None


@dataclass
class ExtractionResult:
    """Standardized extraction result container."""
    status: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class UnifiedKnowledgeExtractor:
    """
    Unified knowledge extractor that integrates all available tools.
    
    This class provides a single interface to:
    - Extract structured knowledge from text
    - Analyze knowledge graphs
    - Mine patterns in data
    - Generate embeddings
    - Discover causal relationships
    - Analyze topological structures
    """
    
    def __init__(self):
        """Initialize all extraction modules."""
        self.modules = {}
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize all available extraction modules."""
        # Graph Analysis
        if KarateClubGraphAnalyzer:
            try:
                self.modules['karateclub'] = KarateClubGraphAnalyzer()
            except Exception as e:
                print(f"Warning: Could not initialize KarateClub: {e}")
        
        # Pattern Mining
        if PAMIPatternMiner:
            try:
                self.modules['pami'] = PAMIPatternMiner()
            except Exception as e:
                print(f"Warning: Could not initialize PAMI: {e}")
        
        # Knowledge Graph Embeddings
        if NeuralKGEmbedder:
            try:
                self.modules['neuralkg'] = NeuralKGEmbedder()
            except Exception as e:
                print(f"Warning: Could not initialize NeuralKG: {e}")
        
        # Causal Discovery
        if CausalDiscoveryEngine:
            try:
                self.modules['causal_learn'] = CausalDiscoveryEngine()
            except Exception as e:
                print(f"Warning: Could not initialize Causal-Learn: {e}")
        
        # Topological Analysis
        if LagrangeAttractorAnalyzer:
            try:
                self.modules['lagrange_mapper'] = LagrangeAttractorAnalyzer()
            except Exception as e:
                print(f"Warning: Could not initialize Lagrange-Mapper: {e}")
        
        print(f"UnifiedKnowledgeExtractor initialized with modules: {list(self.modules.keys())}")
    
    def get_available_modules(self) -> List[str]:
        """Get list of available modules."""
        return list(self.modules.keys())
    
    def get_module_status(self) -> Dict[str, bool]:
        """Get availability status of all modules."""
        return {
            name: module.is_available() if hasattr(module, 'is_available') else True
            for name, module in self.modules.items()
        }
    
    # ==================== Knowledge Extraction ====================
    
    def extract_from_text(
        self,
        text: str,
        extraction_type: str = 'entities_relations',
        config: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Extract structured knowledge from text.
        
        Args:
            text: Input text
            extraction_type: Type of extraction ('entities_relations', 'triples', 'patterns')
            config: Extraction configuration
            
        Returns:
            ExtractionResult with extracted knowledge
        """
        config = config or {}

        try:
            result_data = {
                'text': text,
                'extraction_type': extraction_type,
                'entities': [],
                'relations': [],
                'triples': []
            }

            # Try to use proper NLP tools
            if self._try_spacy_extraction(text, result_data, config):
                pass  # Spacy extraction succeeded
            elif self._try_transformers_extraction(text, result_data, config):
                pass  # Transformers extraction succeeded
            else:
                # Fallback to advanced rule-based extraction
                self._rule_based_extraction(text, result_data, config)

            # Extract relations and triples
            self._extract_relations(result_data, config)
            self._generate_triples(result_data, config)

            return ExtractionResult(
                status='success',
                data=result_data,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'method': result_data.get('extraction_method', 'rule_based'),
                    'entity_count': len(result_data['entities']),
                    'relation_count': len(result_data['relations']),
                    'triple_count': len(result_data['triples'])
                }
            )

        except Exception as e:
            return ExtractionResult(
                status='error',
                errors=[f'Extraction failed: {str(e)}'],
                metadata={'timestamp': datetime.now().isoformat()}
            )

    def _try_spacy_extraction(
        self,
        text: str,
        result_data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> bool:
        """Try to extract using spaCy NLP library."""
        try:
            import spacy

            # Load spaCy model
            model_name = config.get('spacy_model', 'en_core_web_sm')
            try:
                nlp = spacy.load(model_name)
            except OSError:
                # Model not installed, try to download
                import subprocess
                subprocess.run(['python', '-m', 'spacy', 'download', model_name],
                             capture_output=True, check=False)
                nlp = spacy.load(model_name)

            # Process text
            doc = nlp(text)

            # Extract entities
            for ent in doc.ents:
                result_data['entities'].append({
                    'text': ent.text,
                    'type': ent.label_,
                    'position': ent.start_char,
                    'confidence': 0.9,  # spaCy NER is generally reliable
                    'spacy_label': ent.label_
                })

            result_data['extraction_method'] = 'spacy'
            return True

        except Exception as e:
            logger.debug(f"spaCy extraction failed: {e}")
            return False

    def _try_transformers_extraction(
        self,
        text: str,
        result_data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> bool:
        """Try to extract using transformers library."""
        try:
            from transformers import pipeline

            # Use NER pipeline
            model_name = config.get('transformers_model', 'dbmdz/bert-large-cased-finetuned-conll03-english')
            ner_pipeline = pipeline('ner', model=model_name, aggregation_strategy='simple')

            # Extract entities
            entities = ner_pipeline(text)

            for ent in entities:
                result_data['entities'].append({
                    'text': ent['word'],
                    'type': ent['entity_group'],
                    'position': ent.get('start', 0),
                    'confidence': ent['score'],
                    'transformers_label': ent['entity_group']
                })

            result_data['extraction_method'] = 'transformers'
            return True

        except Exception as e:
            logger.debug(f"Transformers extraction failed: {e}")
            return False

    def _rule_based_extraction(
        self,
        text: str,
        result_data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> bool:
        """Advanced rule-based extraction with pattern matching."""
        import re

        # Define entity patterns
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'\bhttps?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w .-]*/?\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'number': r'\b\d+(?:\.\d+)?\b',
            'currency': r'\$\d+(?:\.\d{2})?\b',
            'percentage': r'\d+(?:\.\d+)?%',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }

        # Extract using patterns
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                result_data['entities'].append({
                    'text': match.group(),
                    'type': entity_type,
                    'position': match.start(),
                    'confidence': 0.85,
                    'extraction_method': 'pattern_match'
                })

        # Extract capitalized words (potential named entities)
        words = text.split()
        for i, word in enumerate(words):
            # Clean punctuation
            clean_word = word.strip('.,!?;:()"\'')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                # Skip if already extracted as a pattern
                if not any(e['text'] == clean_word for e in result_data['entities']):
                    result_data['entities'].append({
                        'text': clean_word,
                        'type': 'PROPER_NOUN',
                        'position': text.find(clean_word),
                        'confidence': 0.6,
                        'extraction_method': 'capitalization'
                    })

        # Extract noun phrases (simple version)
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) >= 2 and len(words) <= 5:
                # Look for consecutive capitalized words (likely a named entity)
                consecutive_caps = []
                for word in words:
                    if word and word[0].isupper():
                        consecutive_caps.append(word)
                    else:
                        if len(consecutive_caps) >= 2:
                            phrase = ' '.join(consecutive_caps)
                            position = text.find(phrase)
                            if position >= 0:
                                result_data['entities'].append({
                                    'text': phrase,
                                    'type': 'NOUN_PHRASE',
                                    'position': position,
                                    'confidence': 0.75,
                                    'extraction_method': 'noun_phrase'
                                })
                        consecutive_caps = []

                # Check last sequence
                if len(consecutive_caps) >= 2:
                    phrase = ' '.join(consecutive_caps)
                    position = text.find(phrase)
                    if position >= 0:
                        result_data['entities'].append({
                            'text': phrase,
                            'type': 'NOUN_PHRASE',
                            'position': position,
                            'confidence': 0.75,
                            'extraction_method': 'noun_phrase'
                        })

        result_data['extraction_method'] = 'rule_based'
        return True

    def _extract_relations(self, result_data: Dict[str, Any], config: Dict[str, Any]):
        """Extract relationships between entities."""
        entities = result_data['entities']
        text = result_data['text']

        # Simple relation extraction based on proximity and patterns
        relation_patterns = {
            ('PROPER_NOUN', 'PROPER_NOUN'): [
                (r'(\w+(?:\s+\w+)*)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(?:part of|member of|works at|employed by)\s+(\w+(?:\s+\w+)*)', 'EMPLOYED_BY'),
                (r'(\w+(?:\s+\w+)*)\s+(?:is|was)\s+(?:born in|from)\s+(\w+(?:\s+\w+)*)', 'FROM'),
                (r'(\w+(?:\s+\w+)*)\s+(?:founded|created|established)\s+(\w+(?:\s+\w+)*)', 'FOUNDER_OF'),
                (r'(\w+(?:\s+\w+)*)\s+(?:owns|possesses)\s+(\w+(?:\s+\w+)*)', 'OWNS'),
            ],
            ('PROPER_NOUN', 'ORGANIZATION'): [
                (r'(\w+(?:\s+\w+)*)\s+(?:is|was)\s+(?:CEO|CTO|CFO|president|director)\s+(?:of|at)\s+(\w+(?:\s+\w+)*)', 'EXECUTIVE_OF'),
            ],
        }

        import re

        for entity1 in entities:
            for entity2 in entities:
                if entity1['text'] == entity2['text']:
                    continue

                # Check if entities are close in text (within 50 characters)
                pos1 = entity1.get('position', 0)
                pos2 = entity2.get('position', 0)
                if abs(pos1 - pos2) > 100:
                    continue

                # Try to match relation patterns
                entity1_type = entity1.get('type', 'PROPER_NOUN')
                entity2_type = entity2.get('type', 'PROPER_NOUN')

                # Normalize types for pattern matching
                type_map = {
                    'PERSON': 'PROPER_NOUN',
                    'ORG': 'PROPER_NOUN',
                    'GPE': 'PROPER_NOUN',
                    'LOC': 'PROPER_NOUN',
                }
                entity1_type = type_map.get(entity1_type, entity1_type)
                entity2_type = type_map.get(entity2_type, entity2_type)

                key = (entity1_type, entity2_type)
                if key in relation_patterns:
                    for pattern, relation_type in relation_patterns[key]:
                        match = re.search(pattern, text[pos1:pos1+200])
                        if match:
                            result_data['relations'].append({
                                'subject': entity1['text'],
                                'object': entity2['text'],
                                'relation': relation_type,
                                'confidence': 0.7,
                                'evidence': match.group(0)
                            })

    def _generate_triples(self, result_data: Dict[str, Any], config: Dict[str, Any]):
        """Generate knowledge graph triples."""
        entities = result_data['entities']
        relations = result_data['relations']

        # Create triples from relations
        for relation in relations:
            result_data['triples'].append({
                'subject': relation['subject'],
                'predicate': relation['relation'],
                'object': relation['object'],
                'confidence': relation['confidence'],
                'source': 'extraction'
            })

        # Create self-describing triples for entities
        for entity in entities:
            result_data['triples'].append({
                'subject': entity['text'],
                'predicate': 'rdf:type',
                'object': entity['type'],
                'confidence': entity.get('confidence', 0.8),
                'source': 'entity_type'
            })

            # Add position triple
            if 'position' in entity:
                result_data['triples'].append({
                    'subject': entity['text'],
                    'predicate': 'schema:position',
                    'object': str(entity['position']),
                    'confidence': 1.0,
                    'source': 'position'
                })
    
    # ==================== Graph Analysis ====================
    
    def analyze_knowledge_graph(
        self,
        graph_data: Dict[str, Any],
        analysis_types: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Comprehensive knowledge graph analysis.
        
        Args:
            graph_data: Knowledge graph with nodes and edges
            analysis_types: List of analyses to perform
            config: Analysis configuration
            
        Returns:
            ExtractionResult with analysis results
        """
        config = config or {}
        analysis_types = analysis_types or ['community', 'embeddings', 'patterns', 'causal']
        
        results = {
            'graph_summary': self._summarize_graph(graph_data),
            'analyses': {}
        }
        
        errors = []
        
        # Community Detection (KarateClub)
        if 'community' in analysis_types and 'karateclub' in self.modules:
            try:
                karateclub = self.modules['karateclub']
                result = karateclub.analyze_graph(graph_data, config.get('karateclub_config'))
                results['analyses']['community_detection'] = result
            except Exception as e:
                errors.append(f'Community detection failed: {e}')
        
        # Pattern Mining (PAMI)
        if 'patterns' in analysis_types and 'pami' in self.modules:
            try:
                pami = self.modules['pami']
                result = pami.analyze_knowledge_graph_patterns(
                    graph_data,
                    min_support=config.get('min_support', 0.1)
                )
                results['analyses']['pattern_mining'] = result
            except Exception as e:
                errors.append(f'Pattern mining failed: {e}')
        
        # Embedding Generation (NeuralKG)
        if 'embeddings' in analysis_types and 'neuralkg' in self.modules:
            try:
                neuralkg = self.modules['neuralkg']
                triples = self._graph_to_triples(graph_data)
                if triples:
                    result = neuralkg.generate_embeddings(
                        triples,
                        model_name=config.get('embedding_model', 'transe'),
                        embedding_dim=config.get('embedding_dim', 100)
                    )
                    results['analyses']['embeddings'] = result
            except Exception as e:
                errors.append(f'Embedding generation failed: {e}')
        
        # Topological Analysis (Lagrange-Mapper)
        if 'topology' in analysis_types and 'lagrange_mapper' in self.modules:
            try:
                lagrange = self.modules['lagrange_mapper']
                result = lagrange.analyze_knowledge_topology(
                    graph_data,
                    embedding_dim=config.get('topology_embedding_dim', 50)
                )
                results['analyses']['topology'] = result
            except Exception as e:
                errors.append(f'Topological analysis failed: {e}')
        
        # Causal Discovery Analysis (Causal-Learn)
        if 'causal' in analysis_types and 'causal_learn' in self.modules:
            try:
                causal_engine = self.modules['causal_learn']
                # Convert graph edges to adjacency matrix representation for causal analysis
                nodes = graph_data.get('nodes', [])
                edges = graph_data.get('edges', [])
                
                if len(nodes) > 2 and len(edges) > 1:
                    # Generate synthetic data from graph structure for causal discovery
                    n_nodes = len(nodes)
                    node_names = [n.get('name', n.get('id', f'X{i}')) for i, n in enumerate(nodes)]
                    
                    # Create adjacency matrix from graph structure
                    adj_matrix = np.zeros((n_nodes, n_nodes))
                    for edge in edges:
                        source_idx = next((i for i, n in enumerate(nodes) 
                                         if n.get('id') == edge.get('source') or n.get('name') == edge.get('source')), None)
                        target_idx = next((i for i, n in enumerate(nodes) 
                                         if n.get('id') == edge.get('target') or n.get('name') == edge.get('target')), None)
                        if source_idx is not None and target_idx is not None:
                            adj_matrix[source_idx, target_idx] = 1
                    
                    # Run causal structure analysis on the graph structure
                    causal_result = causal_engine.analyze_causal_graph({
                        'nodes': node_names,
                        'edges': edges,
                        'adjacency_matrix': adj_matrix
                    })
                    
                    results['analyses']['causal_discovery'] = causal_result
                else:
                    results['analyses']['causal_discovery'] = {'status': 'skipped', 'reason': 'Insufficient nodes/edges'}
            except Exception as e:
                errors.append(f'Causal analysis failed: {e}')
        
        status = 'success' if not errors or results['analyses'] else 'partial'
        
        return ExtractionResult(
            status=status,
            data=results,
            errors=errors,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'analyses_performed': list(results['analyses'].keys())
            }
        )
    
    def _summarize_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for a graph."""
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        # Count node types
        node_types = {}
        for node in nodes:
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Count edge types
        edge_types = {}
        for edge in edges:
            edge_type = edge.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        return {
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'node_types': node_types,
            'edge_types': edge_types,
            'density': len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
        }
    
    def _graph_to_triples(self, graph_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """Convert graph data to triples format."""
        triples = []
        
        for edge in graph_data.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            rel_type = edge.get('type', 'related_to')
            
            if source and target:
                triples.append((source, rel_type, target))
        
        return triples
    
    # ==================== Pattern Mining ====================
    
    def mine_patterns(
        self,
        data: Union[List[List[str]], Dict[str, Any]],
        mining_type: str = 'frequent_patterns',
        config: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Mine patterns from data.
        
        Args:
            data: Data to mine (transactions or graph)
            mining_type: Type of mining ('frequent_patterns', 'sequences', 'graph_patterns', 'association_rules')
            config: Mining configuration
            
        Returns:
            ExtractionResult with mined patterns
        """
        if 'pami' not in self.modules:
            return ExtractionResult(
                status='error',
                errors=['PAMI module not available']
            )
        
        config = config or {}
        pami = self.modules['pami']
        
        try:
            if mining_type == 'frequent_patterns':
                result = pami.mine_frequent_patterns(
                    transactions=data if isinstance(data, list) else data.get('transactions', []),
                    min_support=config.get('min_support', 0.1),
                    algorithm=config.get('algorithm', 'fpgrowth')
                )
            elif mining_type == 'sequences':
                result = pami.mine_sequences(
                    sequences=data if isinstance(data, list) else data.get('sequences', []),
                    min_support=config.get('min_support', 0.1),
                    max_gap=config.get('max_gap')
                )
            elif mining_type == 'graph_patterns':
                result = pami.analyze_knowledge_graph_patterns(
                    graph_data=data if isinstance(data, dict) else {},
                    min_support=config.get('min_support', 0.1)
                )
            elif mining_type == 'association_rules':
                result = pami.discover_association_rules(
                    transactions=data if isinstance(data, list) else data.get('transactions', []),
                    min_support=config.get('min_support', 0.1),
                    min_confidence=config.get('min_confidence', 0.5)
                )
            else:
                return ExtractionResult(
                    status='error',
                    errors=[f'Unknown mining type: {mining_type}']
                )
            
            return ExtractionResult(
                status=result.get('status', 'error'),
                data=result,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'mining_type': mining_type
                }
            )
            
        except Exception as e:
            return ExtractionResult(
                status='error',
                errors=[f'Pattern mining failed: {str(e)}'],
                metadata={'timestamp': datetime.now().isoformat()}
            )
    
    # ==================== Embedding Generation ====================
    
    def generate_embeddings(
        self,
        triples: List[Tuple[str, str, str]],
        model: str = 'transe',
        config: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Generate knowledge graph embeddings.
        
        Args:
            triples: List of (head, relation, tail) triples
            model: Model to use ('transe', 'rotate', 'complex', etc.)
            config: Embedding configuration
            
        Returns:
            ExtractionResult with embeddings
        """
        if 'neuralkg' not in self.modules:
            return ExtractionResult(
                status='error',
                errors=['NeuralKG module not available']
            )
        
        config = config or {}
        neuralkg = self.modules['neuralkg']
        
        try:
            result = neuralkg.generate_embeddings(
                triples=triples,
                model_name=model,
                embedding_dim=config.get('embedding_dim', 100),
                epochs=config.get('epochs', 100),
                batch_size=config.get('batch_size', 256),
                learning_rate=config.get('learning_rate', 0.001)
            )
            
            return ExtractionResult(
                status=result.get('status', 'error'),
                data=result.get('embeddings', {}),
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            return ExtractionResult(
                status='error',
                errors=[f'Embedding generation failed: {str(e)}']
            )
    
    def predict_links(
        self,
        head: str,
        relation: str,
        candidate_tails: List[str],
        embeddings: Dict[str, Any],
        top_k: int = 10
    ) -> ExtractionResult:
        """
        Predict links using embeddings.
        
        Args:
            head: Head entity
            relation: Relation
            candidate_tails: Candidate tail entities
            embeddings: Pre-computed embeddings
            top_k: Number of top predictions
            
        Returns:
            ExtractionResult with predictions
        """
        if 'neuralkg' not in self.modules:
            return ExtractionResult(
                status='error',
                errors=['NeuralKG module not available']
            )
        
        neuralkg = self.modules['neuralkg']
        
        try:
            result = neuralkg.predict_links(
                head=head,
                relation=relation,
                candidate_tails=candidate_tails,
                embeddings=embeddings,
                top_k=top_k
            )
            
            return ExtractionResult(
                status=result.get('status', 'error'),
                data=result.get('predictions', []),
                metadata={
                    'head': head,
                    'relation': relation,
                    'top_k': top_k
                }
            )
            
        except Exception as e:
            return ExtractionResult(
                status='error',
                errors=[f'Link prediction failed: {str(e)}']
            )
    
    # ==================== Causal Discovery ====================
    
    def discover_causal_structure(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
        algorithm: str = 'pc',
        config: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Discover causal structure from data.
        
        Args:
            data: Data matrix (n_samples x n_variables)
            variable_names: Variable names
            algorithm: Algorithm ('pc', 'fci', 'ges', 'lingam', etc.)
            config: Algorithm configuration
            
        Returns:
            ExtractionResult with causal graph
        """
        if 'causal_learn' not in self.modules:
            return ExtractionResult(
                status='error',
                errors=['Causal-Learn module not available']
            )
        
        config = config or {}
        causal_engine = self.modules['causal_learn']
        
        try:
            result = causal_engine.discover_causal_structure(
                data=data,
                variable_names=variable_names,
                algorithm=algorithm,
                alpha=config.get('alpha', 0.05),
                independence_test=config.get('independence_test', 'fisherz'),
                **{k: v for k, v in config.items() if k not in ['alpha', 'independence_test']}
            )
            
            return ExtractionResult(
                status=result.get('status', 'error'),
                data=result.get('graph', {}),
                metadata={
                    'algorithm': algorithm,
                    'parameters': result.get('parameters', {})
                }
            )
            
        except Exception as e:
            return ExtractionResult(
                status='error',
                errors=[f'Causal discovery failed: {str(e)}']
            )
    
    def identify_confounders(
        self,
        graph_data: Dict[str, Any],
        target_x: str,
        target_y: str
    ) -> ExtractionResult:
        """
        Identify confounders between two variables.
        
        Args:
            graph_data: Causal graph
            target_x: First target variable
            target_y: Second target variable
            
        Returns:
            ExtractionResult with confounders
        """
        if 'causal_learn' not in self.modules:
            return ExtractionResult(
                status='error',
                errors=['Causal-Learn module not available']
            )
        
        causal_engine = self.modules['causal_learn']
        
        try:
            result = causal_engine.identify_confounders(
                graph_data=graph_data,
                target_x=target_x,
                target_y=target_y
            )
            
            return ExtractionResult(
                status=result.get('status', 'error'),
                data=result.get('confounders', {})
            )
            
        except Exception as e:
            return ExtractionResult(
                status='error',
                errors=[f'Confounder identification failed: {str(e)}']
            )
    
    # ==================== Topological Analysis ====================
    
    def analyze_embedding_landscape(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Analyze attractor landscape in embedding space.
        
        Args:
            embeddings: Embedding matrix
            labels: Optional labels
            config: Configuration
            
        Returns:
            ExtractionResult with landscape analysis
        """
        if 'lagrange_mapper' not in self.modules:
            return ExtractionResult(
                status='error',
                errors=['Lagrange-Mapper module not available']
            )
        
        config = config or {}
        lagrange = self.modules['lagrange_mapper']
        
        try:
            result = lagrange.analyze_embedding_landscape(
                embeddings=embeddings,
                labels=labels,
                n_clusters=config.get('n_clusters', 8),
                reduction_method=config.get('reduction_method', 'pca'),
                reduction_dims=config.get('reduction_dims', 2)
            )
            
            return ExtractionResult(
                status=result.get('status', 'error'),
                data=result.get('landscape', {}),
                metadata=result.get('parameters', {})
            )
            
        except Exception as e:
            return ExtractionResult(
                status='error',
                errors=[f'Landscape analysis failed: {str(e)}']
            )
    
    # ==================== Pipeline Operations ====================
    
    def run_extraction_pipeline(
        self,
        input_data: Dict[str, Any],
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Run a complete extraction pipeline.
        
        Args:
            input_data: Input data (text, graph, etc.)
            pipeline_config: Pipeline configuration
            
        Returns:
            ExtractionResult with all pipeline results
        """
        pipeline_config = pipeline_config or {}
        
        results = {
            'pipeline_stages': [],
            'stage_results': {}
        }
        errors = []
        
        # Stage 1: Text extraction (if text provided)
        if 'text' in input_data and pipeline_config.get('extract_text', True):
            result = self.extract_from_text(
                input_data['text'],
                config=pipeline_config.get('text_extraction_config')
            )
            results['stage_results']['text_extraction'] = result
            results['pipeline_stages'].append('text_extraction')
            if result.status == 'error':
                errors.extend(result.errors)
        
        # Stage 2: Graph analysis (if graph provided)
        if 'graph' in input_data and pipeline_config.get('analyze_graph', True):
            result = self.analyze_knowledge_graph(
                input_data['graph'],
                analysis_types=pipeline_config.get('analysis_types'),
                config=pipeline_config.get('graph_analysis_config')
            )
            results['stage_results']['graph_analysis'] = result
            results['pipeline_stages'].append('graph_analysis')
            if result.status == 'error':
                errors.extend(result.errors)
        
        # Stage 3: Pattern mining (if transactions provided)
        if 'transactions' in input_data and pipeline_config.get('mine_patterns', True):
            result = self.mine_patterns(
                input_data['transactions'],
                mining_type=pipeline_config.get('mining_type', 'frequent_patterns'),
                config=pipeline_config.get('pattern_mining_config')
            )
            results['stage_results']['pattern_mining'] = result
            results['pipeline_stages'].append('pattern_mining')
            if result.status == 'error':
                errors.extend(result.errors)
        
        # Stage 4: Embedding generation (if triples provided)
        if 'triples' in input_data and pipeline_config.get('generate_embeddings', True):
            result = self.generate_embeddings(
                input_data['triples'],
                model=pipeline_config.get('embedding_model', 'transe'),
                config=pipeline_config.get('embedding_config')
            )
            results['stage_results']['embeddings'] = result
            results['pipeline_stages'].append('embeddings')
            if result.status == 'error':
                errors.extend(result.errors)
        
        # Stage 5: Causal discovery (if data matrix provided)
        if 'data_matrix' in input_data and pipeline_config.get('discover_causal', False):
            result = self.discover_causal_structure(
                data=input_data['data_matrix'],
                variable_names=input_data.get('variable_names'),
                algorithm=pipeline_config.get('causal_algorithm', 'pc'),
                config=pipeline_config.get('causal_config')
            )
            results['stage_results']['causal_discovery'] = result
            results['pipeline_stages'].append('causal_discovery')
            if result.status == 'error':
                errors.extend(result.errors)
        
        status = 'success' if not errors else ('partial' if results['pipeline_stages'] else 'error')
        
        return ExtractionResult(
            status=status,
            data=results,
            errors=errors,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'stages_completed': results['pipeline_stages'],
                'modules_available': self.get_available_modules()
            }
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the extractor."""
        return {
            'available_modules': self.get_available_modules(),
            'module_status': self.get_module_status(),
            'capabilities': [
                'knowledge_extraction',
                'graph_analysis',
                'pattern_mining',
                'embedding_generation',
                'causal_discovery',
                'topological_analysis'
            ],
            'timestamp': datetime.now().isoformat()
        }


# Convenience function for quick extraction
def extract_knowledge(
    data: Dict[str, Any],
    operations: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function for quick knowledge extraction.
    
    Args:
        data: Input data dictionary
        operations: List of operations to perform
        
    Returns:
        Extraction results
    """
    extractor = UnifiedKnowledgeExtractor()
    
    result = extractor.run_extraction_pipeline(
        input_data=data,
        pipeline_config={
            'extract_text': 'text' in (operations or []),
            'analyze_graph': 'graph' in (operations or []),
            'mine_patterns': 'patterns' in (operations or []),
            'generate_embeddings': 'embeddings' in (operations or []),
            'discover_causal': 'causal' in (operations or [])
        }
    )
    
    return {
        'status': result.status,
        'data': result.data,
        'errors': result.errors,
        'metadata': result.metadata
    }
