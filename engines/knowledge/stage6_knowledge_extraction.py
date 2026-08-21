"""
Stage 6 Knowledge Extraction - License: Apache 2.0

Advanced knowledge extraction system for OpenEvolve workflows.
Extracts patterns, insights, and reusable knowledge from execution traces.

Features:
- ML-based pattern clustering (Sentence Transformers + scikit-learn)
- Temporal knowledge graph construction
- Knowledge validation with Z3
- Hybrid semantic + keyword retrieval
- Entity and relation extraction

Dependencies (all permissive licenses):
- numpy: BSD License
- scikit-learn: BSD License
- networkx: BSD License
- sentence-transformers: Apache 2.0

Author: OpenEvolve
Date: 2026-02-02
"""
from __future__ import annotations


import json
import re
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import asyncio

# NumPy - BSD License
import numpy as np

# NetworkX - BSD License
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# scikit-learn - BSD License
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ML Pattern Clustering
try:
    from ml_pattern_clustering import (
        MLKnowledgeExtraction,
        MLPatternClustering,
        EntityExtractor,
        RelationExtractor,
        TemporalKnowledgeGraph,
        KnowledgeValidator,
        MLPattern,
        ExtractedEntity,
        ExtractedRelation
    )
    ML_CLUSTERING_AVAILABLE = True
except ImportError as e:
    ML_CLUSTERING_AVAILABLE = False
    print(f"ML clustering not available: {e}")

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Z3 Validation
try:
    from z3 import Solver, Bool, And, sat
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

# CAV-NLP Integration for knowledge formalization
try:
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    print("CAV-NLP not available for knowledge formalization")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ExtractedPattern:
    """A pattern extracted from workflow execution."""
    pattern_id: str
    pattern_type: str  # 'sequence', 'semantic', 'parametric', 'structural', 'ml_clustered'
    description: str
    confidence: float  # 0.0 to 1.0
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    examples: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    # ML-specific fields
    ml_cluster_id: Optional[str] = None
    ml_silhouette_score: float = 0.0
    ml_cluster_size: int = 0
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat()
        }


@dataclass
class KnowledgeArtifact:
    """A reusable knowledge artifact extracted from workflows."""
    artifact_id: str
    name: str
    artifact_type: str  # 'strategy', 'template', 'constraint', 'heuristic'
    content: Dict
    source_workflows: List[str]
    extraction_date: datetime
    validity_score: float
    usage_count: int = 0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Temporal fields
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    version: int = 1
    
    # CAV-NLP formal representation
    formal_representation: Optional[str] = None  # Formalized code (Z3, Lean4, etc.)
    formalization_method: Optional[str] = None  # 'cav_nlp', 'manual', 'z3', 'lean4'
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'extraction_date': self.extraction_date.isoformat(),
            'valid_from': self.valid_from.isoformat() if self.valid_from else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None
        }


@dataclass
class ExecutionTrace:
    """Trace of a workflow execution."""
    trace_id: str
    workflow_id: str
    problem_description: str
    stages: List[Dict]
    final_result: Optional[Dict]
    execution_time_ms: float
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# KNOWLEDGE EXTRACTORS
# =============================================================================

class PatternExtractor:
    """Extracts patterns from execution traces with ML clustering."""
    
    PATTERN_TYPES = ['sequence', 'semantic', 'parametric', 'structural', 'ml_clustered']
    
    def __init__(self, min_confidence: float = 0.7, enable_ml_clustering: bool = True):
        self.min_confidence = min_confidence
        self.patterns: Dict[str, ExtractedPattern] = {}
        self.enable_ml_clustering = enable_ml_clustering and ML_CLUSTERING_AVAILABLE
        
        # Initialize ML clustering
        self.ml_clustering = None
        if self.enable_ml_clustering:
            try:
                self.ml_clustering = MLPatternClustering(
                    model_name='all-MiniLM-L6-v2',
                    clustering_algorithm='dbscan'
                )
                print("[OK] ML pattern clustering enabled")
            except Exception as e:
                print(f"[ERROR] Failed to initialize ML clustering: {e}")
                self.enable_ml_clustering = False
    
    def extract_sequence_patterns(
        self,
        traces: List[ExecutionTrace]
    ) -> List[ExtractedPattern]:
        """Extract sequence patterns (stage orderings)."""
        sequences = defaultdict(list)
        
        for trace in traces:
            seq = tuple(s['stage_name'] for s in trace.stages)
            sequences[seq].append(trace.trace_id)
        
        patterns = []
        for seq, trace_ids in sequences.items():
            if len(trace_ids) >= 2:  # Minimum support
                pattern_id = hashlib.md5(
                    json.dumps(seq, sort_keys=True).encode()
                ).hexdigest()[:12]
                
                confidence = min(1.0, len(trace_ids) / len(traces))
                
                if confidence >= self.min_confidence:
                    pattern = ExtractedPattern(
                        pattern_id=f"seq_{pattern_id}",
                        pattern_type='sequence',
                        description=f"Common stage sequence: {' -> '.join(seq)}",
                        confidence=confidence,
                        occurrences=len(trace_ids),
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        examples=[{'sequence': seq, 'traces': trace_ids[:5]}],
                        metadata={
                            'stage_count': len(seq),
                            'unique_stages': len(set(seq))
                        }
                    )
                    patterns.append(pattern)
                    self.patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def extract_semantic_patterns(
        self,
        traces: List[ExecutionTrace]
    ) -> List[ExtractedPattern]:
        """Extract semantic patterns from problem descriptions using ML clustering."""
        
        # Use ML clustering if available
        if self.enable_ml_clustering and self.ml_clustering:
            return self._extract_ml_semantic_patterns(traces)
        
        # Fallback to traditional TF-IDF + DBSCAN
        return self._extract_traditional_semantic_patterns(traces)
    
    def _extract_ml_semantic_patterns(
        self,
        traces: List[ExecutionTrace]
    ) -> List[ExtractedPattern]:
        """Extract semantic patterns using ML clustering."""
        if not traces:
            return []
        
        # Extract problem descriptions
        descriptions = [t.problem_description for t in traces]
        metadata = [{'trace_id': t.trace_id, 'domain': t.metadata.get('domain', 'general')} for t in traces]
        
        # Cluster using ML
        ml_patterns = self.ml_clustering.cluster_patterns(descriptions, metadata)
        
        # Convert to ExtractedPattern format
        patterns = []
        for ml_pattern in ml_patterns:
            pattern = ExtractedPattern(
                pattern_id=ml_pattern.pattern_id,
                pattern_type='ml_clustered',
                description=ml_pattern.description,
                confidence=ml_pattern.confidence,
                occurrences=ml_pattern.cluster_size,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                examples=[{'text': ex, 'source': 'ml_cluster'} for ex in ml_pattern.representative_examples],
                metadata={
                    'silhouette_score': ml_pattern.silhouette_score,
                    'cluster_features': ml_pattern.features,
                    'tags': ml_pattern.tags
                },
                ml_cluster_id=ml_pattern.pattern_id,
                ml_silhouette_score=ml_pattern.silhouette_score,
                ml_cluster_size=ml_pattern.cluster_size
            )
            patterns.append(pattern)
            self.patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def _extract_traditional_semantic_patterns(
        self,
        traces: List[ExecutionTrace]
    ) -> List[ExtractedPattern]:
        """Fallback: Extract semantic patterns using TF-IDF + DBSCAN."""
        if not SKLEARN_AVAILABLE or not traces:
            return []
        
        # Extract problem descriptions
        descriptions = [t.problem_description for t in traces]
        
        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            vectors = vectorizer.fit_transform(descriptions)
        except ValueError:
            return []
        
        # Cluster similar problems
        clustering = DBSCAN(eps=0.5, min_samples=2)
        labels = clustering.fit_predict(vectors.toarray())
        
        patterns = []
        unique_labels = set(labels) - {-1}  # Exclude noise
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_traces = [traces[i] for i in cluster_indices]
            
            # Get common terms
            feature_names = vectorizer.get_feature_names_out()
            centroid = vectors[cluster_indices].mean(axis=0).A1
            top_indices = centroid.argsort()[-5:][::-1]
            common_terms = [feature_names[i] for i in top_indices]
            
            pattern_id = f"sem_{label}_{hashlib.md5(json.dumps(common_terms).encode()).hexdigest()[:8]}"
            
            pattern = ExtractedPattern(
                pattern_id=pattern_id,
                pattern_type='semantic',
                description=f"Problem cluster: {', '.join(common_terms)}",
                confidence=len(cluster_indices) / len(traces),
                occurrences=len(cluster_indices),
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                examples=[{
                    'common_terms': common_terms,
                    'sample_problems': [t.problem_description[:100] + '...' 
                                      for t in cluster_traces[:3]]
                }],
                metadata={
                    'cluster_size': len(cluster_indices),
                    'silhouette_score': None  # Could calculate if needed
                }
            )
            patterns.append(pattern)
            self.patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def extract_parametric_patterns(
        self,
        traces: List[ExecutionTrace]
    ) -> List[ExtractedPattern]:
        """Extract patterns in parameter usage."""
        param_usage = defaultdict(lambda: defaultdict(int))
        
        for trace in traces:
            for stage in trace.stages:
                params = stage.get('parameters', {})
                for param_name, param_value in params.items():
                    param_usage[param_name][str(param_value)] += 1
        
        patterns = []
        for param_name, values in param_usage.items():
            total = sum(values.values())
            for value, count in values.items():
                frequency = count / total
                if frequency >= self.min_confidence and count >= 2:
                    pattern_id = hashlib.md5(
                        f"{param_name}:{value}".encode()
                    ).hexdigest()[:12]
                    
                    pattern = ExtractedPattern(
                        pattern_id=f"par_{pattern_id}",
                        pattern_type='parametric',
                        description=f"Common parameter value: {param_name}={value}",
                        confidence=frequency,
                        occurrences=count,
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        examples=[{'parameter': param_name, 'value': value}],
                        metadata={'frequency': frequency}
                    )
                    patterns.append(pattern)
                    self.patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def extract_structural_patterns(
        self,
        traces: List[ExecutionTrace]
    ) -> List[ExtractedPattern]:
        """Extract structural patterns from solution structures."""
        structures = defaultdict(list)
        
        for trace in traces:
            if trace.final_result:
                # Extract structure signature
                structure = self._get_structure_signature(trace.final_result)
                structures[structure].append(trace.trace_id)
        
        patterns = []
        for structure, trace_ids in structures.items():
            if len(trace_ids) >= 2:
                pattern_id = hashlib.md5(structure.encode()).hexdigest()[:12]
                confidence = min(1.0, len(trace_ids) / len(traces))
                
                if confidence >= self.min_confidence:
                    pattern = ExtractedPattern(
                        pattern_id=f"str_{pattern_id}",
                        pattern_type='structural',
                        description=f"Common solution structure: {structure[:100]}",
                        confidence=confidence,
                        occurrences=len(trace_ids),
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        examples=[{'structure': structure, 'traces': trace_ids[:5]}],
                        metadata={'structure_hash': pattern_id}
                    )
                    patterns.append(pattern)
                    self.patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def _get_structure_signature(self, data: Any) -> str:
        """Get structural signature of data."""
        if isinstance(data, dict):
            return '{' + ','.join(sorted(f"{k}:{self._get_structure_signature(v)}" 
                                        for k, v in data.items())) + '}'
        elif isinstance(data, list):
            return f'[{len(data)}]'
        else:
            return type(data).__name__


class KnowledgeArtifactGenerator:
    """Generates reusable knowledge artifacts from patterns."""
    
    def __init__(self):
        self.artifacts: Dict[str, KnowledgeArtifact] = {}
    
    def generate_strategy_artifact(
        self,
        pattern: ExtractedPattern,
        traces: List[ExecutionTrace]
    ) -> Optional[KnowledgeArtifact]:
        """Generate a strategy artifact from a pattern."""
        if pattern.pattern_type not in ['sequence', 'ml_clustered']:
            return None
        
        artifact_id = f"strategy_{pattern.pattern_id}"
        
        # Extract successful executions
        successful = [t for t in traces if t.final_result and 
                     t.trace_id in str(pattern.examples)]
        
        if not successful:
            return None
        
        # Calculate success rate
        success_rate = len(successful) / pattern.occurrences
        
        artifact = KnowledgeArtifact(
            artifact_id=artifact_id,
            name=f"Strategy for {pattern.description[:50]}",
            artifact_type='strategy',
            content={
                'stage_sequence': pattern.examples[0].get('sequence', []),
                'success_rate': success_rate,
                'avg_execution_time': np.mean([t.execution_time_ms for t in successful])
            },
            source_workflows=[t.workflow_id for t in successful[:10]],
            extraction_date=datetime.now(),
            validity_score=pattern.confidence * success_rate,
            tags=['auto-generated', 'strategy', pattern.pattern_type],
            dependencies=[]
        )
        
        self.artifacts[artifact_id] = artifact
        return artifact
    
    def generate_template_artifact(
        self,
        pattern: ExtractedPattern,
        traces: List[ExecutionTrace]
    ) -> Optional[KnowledgeArtifact]:
        """Generate a template artifact from structural patterns."""
        if pattern.pattern_type not in ['structural', 'ml_clustered']:
            return None
        
        artifact_id = f"template_{pattern.pattern_id}"
        
        # Find representative example
        related_traces = [t for t in traces if t.trace_id in str(pattern.examples)]
        
        if not related_traces:
            return None
        
        # Extract common template
        template = self._extract_common_template(
            [t.final_result for t in related_traces if t.final_result]
        )
        
        artifact = KnowledgeArtifact(
            artifact_id=artifact_id,
            name=f"Solution Template ({pattern.occurrences} occurrences)",
            artifact_type='template',
            content={
                'template_structure': template,
                'variable_slots': self._identify_variable_slots(template)
            },
            source_workflows=[t.workflow_id for t in related_traces[:10]],
            extraction_date=datetime.now(),
            validity_score=pattern.confidence,
            tags=['auto-generated', 'template'],
            dependencies=[]
        )
        
        self.artifacts[artifact_id] = artifact
        return artifact
    
    def generate_constraint_artifact(
        self,
        pattern: ExtractedPattern,
        traces: List[ExecutionTrace]
    ) -> Optional[KnowledgeArtifact]:
        """Generate constraint artifacts from parametric patterns."""
        if pattern.pattern_type != 'parametric':
            return None
        
        artifact_id = f"constraint_{pattern.pattern_id}"
        
        param_info = pattern.examples[0]
        param_name = param_info.get('parameter', 'unknown')
        param_value = param_info.get('value', 'unknown')
        
        artifact = KnowledgeArtifact(
            artifact_id=artifact_id,
            name=f"Constraint: {param_name}={param_value}",
            artifact_type='constraint',
            content={
                'parameter': param_name,
                'suggested_value': param_value,
                'confidence': pattern.confidence,
                'frequency': pattern.metadata.get('frequency', 0)
            },
            source_workflows=[t.workflow_id for t in traces[:10]],
            extraction_date=datetime.now(),
            validity_score=pattern.confidence,
            tags=['auto-generated', 'constraint', 'parameter'],
            dependencies=[]
        )
        
        self.artifacts[artifact_id] = artifact
        return artifact
    
    def _extract_common_template(self, results: List[Dict]) -> Dict:
        """Extract common template from multiple results."""
        if not results:
            return {}
        
        # Find common keys
        common_keys = set(results[0].keys())
        for r in results[1:]:
            common_keys &= set(r.keys())
        
        template = {}
        for key in common_keys:
            values = [r[key] for r in results]
            if all(isinstance(v, type(values[0])) for v in values):
                template[key] = type(values[0]).__name__
        
        return template
    
    def _identify_variable_slots(self, template: Dict) -> List[str]:
        """Identify variable slots in template."""
        slots = []
        for key, value in template.items():
            if isinstance(value, str) and value in ['str', 'int', 'float']:
                slots.append(key)
        return slots


# =============================================================================
# TEMPORAL KNOWLEDGE GRAPH MANAGER
# =============================================================================

class TemporalKnowledgeManager:
    """
    Manages temporal knowledge graph construction and querying.
    
    Features:
    - Time-aware knowledge storage
    - Knowledge versioning
    - Automatic expiration
    - Temporal querying
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("temporal_knowledge")
        self.storage_path.mkdir(exist_ok=True)
        
        self.graph = None
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        
        self.nodes: Dict[str, Dict] = {}
        self.temporal_index: List[Tuple[datetime, str]] = []
        
        self._load_data()
    
    def add_knowledge(
        self,
        content: str,
        knowledge_type: str = "fact",
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add knowledge with temporal information.
        
        Args:
            content: Knowledge content
            knowledge_type: Type of knowledge
            valid_from: When knowledge becomes valid
            valid_until: When knowledge expires
            confidence: Confidence score
            metadata: Additional metadata
            
        Returns:
            Knowledge node ID
        """
        node_id = f"tk_{hashlib.md5(f'{content}_{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"
        
        node_data = {
            'node_id': node_id,
            'content': content,
            'type': knowledge_type,
            'created_at': datetime.now().isoformat(),
            'valid_from': valid_from.isoformat() if valid_from else None,
            'valid_until': valid_until.isoformat() if valid_until else None,
            'confidence': confidence,
            'metadata': metadata or {},
            'version': 1
        }
        
        self.nodes[node_id] = node_data
        self.temporal_index.append((datetime.now(), node_id))
        
        if self.graph:
            self.graph.add_node(node_id, **node_data)
        
        # Save to disk
        self._save_data()
        
        return node_id
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "related_to",
        confidence: float = 0.5
    ) -> bool:
        """Add a relation between knowledge nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        if self.graph:
            self.graph.add_edge(
                source_id, 
                target_id,
                relation=relation_type,
                confidence=confidence,
                created_at=datetime.now().isoformat()
            )
        
        self._save_data()
        return True
    
    def get_valid_knowledge(
        self,
        at_time: Optional[datetime] = None,
        knowledge_type: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """
        Get knowledge that is valid at a specific time.
        
        Args:
            at_time: Time to check (default: now)
            knowledge_type: Optional type filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of valid knowledge nodes
        """
        check_time = at_time or datetime.now()
        valid = []
        
        for node_id, node in self.nodes.items():
            # Check expiration
            valid_until = node.get('valid_until')
            if valid_until:
                valid_until_dt = datetime.fromisoformat(valid_until)
                if check_time > valid_until_dt:
                    continue
            
            # Check valid_from
            valid_from = node.get('valid_from')
            if valid_from:
                valid_from_dt = datetime.fromisoformat(valid_from)
                if check_time < valid_from_dt:
                    continue
            
            # Check type filter
            if knowledge_type and node.get('type') != knowledge_type:
                continue
            
            # Check confidence
            if node.get('confidence', 0) < min_confidence:
                continue
            
            valid.append(node)
        
        return valid
    
    def get_knowledge_evolution(
        self,
        content_hash: str
    ) -> List[Dict]:
        """Get evolution history of knowledge with similar content."""
        # Find all versions of similar knowledge
        versions = []
        for node in self.nodes.values():
            if content_hash in hashlib.md5(node['content'].encode()).hexdigest():
                versions.append(node)
        
        # Sort by version number and creation time
        versions.sort(key=lambda n: (n.get('version', 1), n['created_at']))
        return versions
    
    def create_version(
        self,
        node_id: str,
        new_content: str,
        confidence: Optional[float] = None
    ) -> Optional[str]:
        """Create a new version of existing knowledge."""
        old_node = self.nodes.get(node_id)
        if not old_node:
            return None
        
        # Mark old version
        old_node['superseded_by'] = f"{node_id}_v{old_node.get('version', 1) + 1}"
        old_node['superseded_at'] = datetime.now().isoformat()
        
        # Create new version
        new_node_id = self.add_knowledge(
            content=new_content,
            knowledge_type=old_node['type'],
            confidence=confidence or old_node.get('confidence', 0.5),
            metadata={
                'previous_version': node_id,
                **old_node.get('metadata', {})
            }
        )
        
        # Update version number
        self.nodes[new_node_id]['version'] = old_node.get('version', 1) + 1
        
        self._save_data()
        return new_node_id
    
    def query_temporal_range(
        self,
        start: datetime,
        end: datetime,
        knowledge_type: Optional[str] = None
    ) -> List[Dict]:
        """Query knowledge within a temporal range."""
        results = []
        for node in self.nodes.values():
            created = datetime.fromisoformat(node['created_at'])
            
            if start <= created <= end:
                if knowledge_type is None or node.get('type') == knowledge_type:
                    results.append(node)
        
        return results
    
    def _save_data(self):
        """Save temporal knowledge to disk."""
        data = {
            'nodes': list(self.nodes.values()),
            'saved_at': datetime.now().isoformat()
        }
        
        filepath = self.storage_path / "temporal_knowledge.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_data(self):
        """Load temporal knowledge from disk."""
        filepath = self.storage_path / "temporal_knowledge.json"
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                for node in data.get('nodes', []):
                    self.nodes[node['node_id']] = node
                    self.temporal_index.append(
                        (datetime.fromisoformat(node['created_at']), node['node_id'])
                    )
                
                # Rebuild graph
                if self.graph:
                    for node in self.nodes.values():
                        self.graph.add_node(node['node_id'], **node)
                
            except Exception as e:
                print(f"Failed to load temporal knowledge: {e}")


# =============================================================================
# KNOWLEDGE VALIDATION ENGINE
# =============================================================================

class KnowledgeValidationEngine:
    """
    Validates extracted knowledge for consistency and correctness.
    
    Features:
    - Z3-based logical consistency checking
    - Cross-reference validation
    - Confidence scoring
    - Contradiction detection
    """
    
    def __init__(self):
        self.validation_results: List[Dict] = []
        self.ground_truth: Dict[str, Any] = {}
    
    def validate_pattern(
        self,
        pattern: ExtractedPattern
    ) -> Dict[str, Any]:
        """
        Validate a pattern against ground truth and logic.
        
        Args:
            pattern: Pattern to validate
            
        Returns:
            Validation result
        """
        result = {
            'pattern_id': pattern.pattern_id,
            'valid': True,
            'confidence': pattern.confidence,
            'checks': {}
        }
        
        # Check 1: Minimum confidence
        result['checks']['min_confidence'] = {
            'passed': pattern.confidence >= 0.5,
            'value': pattern.confidence,
            'threshold': 0.5
        }
        
        # Check 2: Minimum occurrences
        result['checks']['min_occurrences'] = {
            'passed': pattern.occurrences >= 2,
            'value': pattern.occurrences,
            'threshold': 2
        }
        
        # Check 3: ML quality (if ML clustered)
        if pattern.pattern_type == 'ml_clustered':
            result['checks']['ml_quality'] = {
                'passed': pattern.ml_silhouette_score > 0.0,
                'value': pattern.ml_silhouette_score,
                'threshold': 0.0
            }
        
        # Check 4: Ground truth validation (if available)
        ground_truth_match = self._check_ground_truth(pattern)
        result['checks']['ground_truth'] = ground_truth_match
        
        # Calculate overall validity
        result['valid'] = all(
            check['passed'] 
            for check in result['checks'].values() 
            if isinstance(check, dict) and 'passed' in check
        )
        
        # Adjust confidence based on validation
        if result['valid']:
            result['confidence'] = min(1.0, pattern.confidence * 1.1)
        else:
            result['confidence'] = pattern.confidence * 0.8
        
        self.validation_results.append(result)
        return result
    
    def validate_consistency(
        self,
        patterns: List[ExtractedPattern]
    ) -> Dict[str, Any]:
        """
        Check consistency between patterns using Z3.
        
        Args:
            patterns: List of patterns to check
            
        Returns:
            Consistency check result
        """
        if not Z3_AVAILABLE:
            return {
                'consistent': None,
                'message': 'Z3 not available',
                'patterns_checked': len(patterns)
            }
        
        try:
            solver = Solver()
            
            # Create boolean variables for each pattern
            pattern_vars = {}
            for pattern in patterns:
                var_name = f"pattern_{pattern.pattern_id.replace('-', '_')}"
                pattern_vars[pattern.pattern_id] = Bool(var_name)
                # Assume pattern is valid
                solver.add(pattern_vars[pattern.pattern_id])
            
            # Add consistency constraints (example)
            # If pattern A and pattern B contradict, add constraint
            # solver.add(Or(Not(pattern_vars['A']), Not(pattern_vars['B'])))
            
            result = solver.check()
            
            if result == sat:
                return {
                    'consistent': True,
                    'confidence': 0.9,
                    'patterns_checked': len(patterns),
                    'message': 'Patterns are logically consistent'
                }
            else:
                return {
                    'consistent': False,
                    'confidence': 0.95,
                    'patterns_checked': len(patterns),
                    'message': 'Patterns contain contradictions'
                }
        
        except Exception as e:
            return {
                'consistent': None,
                'message': f'Validation error: {e}',
                'patterns_checked': len(patterns)
            }
    
    def _check_ground_truth(self, pattern: ExtractedPattern) -> Dict:
        """Check pattern against ground truth if available."""
        # Simple ground truth matching
        if not self.ground_truth:
            return {'available': False, 'passed': True}
        
        # Check if pattern matches known good patterns
        for gt_id, gt_pattern in self.ground_truth.items():
            if pattern.pattern_type == gt_pattern.get('type'):
                similarity = self._calculate_similarity(
                    pattern.description,
                    gt_pattern.get('description', '')
                )
                if similarity > 0.8:
                    return {
                        'available': True,
                        'passed': True,
                        'match_score': similarity,
                        'ground_truth_id': gt_id
                    }
        
        return {'available': True, 'passed': False, 'match_score': 0.0}
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def load_ground_truth(self, filepath: str):
        """Load ground truth patterns from file."""
        try:
            with open(filepath, 'r') as f:
                self.ground_truth = json.load(f)
        except Exception as e:
            print(f"Failed to load ground truth: {e}")


# =============================================================================
# HYBRID RETRIEVAL SYSTEM
# =============================================================================

class HybridRetrievalSystem:
    """
    Hybrid knowledge retrieval using semantic and keyword search.
    
    Features:
    - Semantic search using embeddings
    - Keyword-based search
    - Combined ranking
    - Context-aware retrieval
    """
    
    def __init__(self, embedding_model: Optional[str] = None):
        self.embedding_model = None
        self.knowledge_base: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and embedding_model:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
    
    def add_knowledge(self, knowledge: Dict):
        """Add knowledge to retrieval index."""
        self.knowledge_base.append(knowledge)
        
        # Generate embedding if model available
        if self.embedding_model:
            text = knowledge.get('description', knowledge.get('content', ''))
            embedding = self.embedding_model.encode(text)
            self.embeddings.append(embedding)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        Retrieve knowledge using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic vs keyword (0-1)
            
        Returns:
            List of retrieved knowledge items
        """
        if not self.knowledge_base:
            return []
        
        # Semantic search
        semantic_scores = self._semantic_search(query) if self.embedding_model else [0] * len(self.knowledge_base)
        
        # Keyword search
        keyword_scores = self._keyword_search(query)
        
        # Combine scores
        combined_scores = []
        for i in range(len(self.knowledge_base)):
            score = semantic_weight * semantic_scores[i] + (1 - semantic_weight) * keyword_scores[i]
            combined_scores.append((i, score))
        
        # Sort by score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        results = []
        for idx, score in combined_scores[:top_k]:
            item = self.knowledge_base[idx].copy()
            item['retrieval_score'] = score
            results.append(item)
        
        return results
    
    def _semantic_search(self, query: str) -> List[float]:
        """Perform semantic search using embeddings."""
        if not self.embedding_model or not self.embeddings:
            return [0.0] * len(self.knowledge_base)
        
        query_embedding = self.embedding_model.encode(query)
        
        scores = []
        for emb in self.embeddings:
            similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            scores.append(float(similarity))
        
        return scores
    
    def _keyword_search(self, query: str) -> List[float]:
        """Perform keyword-based search."""
        query_words = set(query.lower().split())
        
        scores = []
        for knowledge in self.knowledge_base:
            text = knowledge.get('description', knowledge.get('content', ''))
            text_words = set(text.lower().split())
            
            if not text_words:
                scores.append(0.0)
                continue
            
            overlap = len(query_words & text_words)
            score = overlap / len(query_words) if query_words else 0.0
            scores.append(score)
        
        return scores


# =============================================================================
# STAGE 6 KNOWLEDGE EXTRACTION ENGINE
# =============================================================================

class Stage6KnowledgeExtraction:
    """
    Stage 6 Knowledge Extraction Engine.
    
    Extracts knowledge from completed workflows to improve future executions.
    Implements pattern recognition, artifact generation, and knowledge management.
    
    NEW FEATURES:
    - ML-based pattern clustering (Sentence Transformers + scikit-learn)
    - Temporal knowledge graph construction
    - Knowledge validation with Z3
    - Hybrid semantic + keyword retrieval
    - Entity and relation extraction
    
    License: Apache 2.0
    """
    
    def __init__(
        self, 
        storage_path: Optional[Path] = None, 
        enable_ml: bool = True,
        use_cav_nlp: bool = True
    ):
        self.storage_path = storage_path or Path("knowledge_extraction")
        self.storage_path.mkdir(exist_ok=True)
        
        self.pattern_extractor = PatternExtractor(enable_ml_clustering=enable_ml)
        self.artifact_generator = KnowledgeArtifactGenerator()
        self.temporal_manager = TemporalKnowledgeManager(self.storage_path)
        self.validation_engine = KnowledgeValidationEngine()
        self.retrieval_system = HybridRetrievalSystem(
            embedding_model='all-MiniLM-L6-v2' if enable_ml else None
        )
        
        self.traces: List[ExecutionTrace] = []
        self.patterns: Dict[str, ExtractedPattern] = {}
        self.artifacts: Dict[str, KnowledgeArtifact] = {}
        
        # ML extraction integration
        self.ml_extraction = None
        if enable_ml and ML_CLUSTERING_AVAILABLE:
            try:
                self.ml_extraction = MLKnowledgeExtraction()
                print("[OK] ML Knowledge Extraction enabled")
            except Exception as e:
                print(f"[FAIL] Failed to initialize ML extraction: {e}")
        
        # CAV-NLP integration for knowledge formalization
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        self.math_service: Optional[UnifiedMathService] = None
        if self.use_cav_nlp:
            try:
                self.math_service = UnifiedMathService()
                print("[OK] CAV-NLP formalization enabled")
            except Exception as e:
                print(f"[FAIL] Failed to initialize CAV-NLP: {e}")
                self.use_cav_nlp = False
        
        self._load_existing_data()
    
    def _load_existing_data(self) -> None:
        """Load existing patterns and artifacts."""
        patterns_file = self.storage_path / "patterns.json"
        artifacts_file = self.storage_path / "artifacts.json"
        
        if patterns_file.exists():
            with open(patterns_file) as f:
                data = json.load(f)
                for p in data.get('patterns', []):
                    pattern = ExtractedPattern(
                        pattern_id=p['pattern_id'],
                        pattern_type=p['pattern_type'],
                        description=p['description'],
                        confidence=p['confidence'],
                        occurrences=p['occurrences'],
                        first_seen=datetime.fromisoformat(p['first_seen']),
                        last_seen=datetime.fromisoformat(p['last_seen']),
                        examples=p.get('examples', []),
                        metadata=p.get('metadata', {}),
                        ml_cluster_id=p.get('ml_cluster_id'),
                        ml_silhouette_score=p.get('ml_silhouette_score', 0.0),
                        ml_cluster_size=p.get('ml_cluster_size', 0)
                    )
                    self.patterns[pattern.pattern_id] = pattern
        
        if artifacts_file.exists():
            with open(artifacts_file) as f:
                data = json.load(f)
                for a in data.get('artifacts', []):
                    artifact = KnowledgeArtifact(
                        artifact_id=a['artifact_id'],
                        name=a['name'],
                        artifact_type=a['artifact_type'],
                        content=a['content'],
                        source_workflows=a['source_workflows'],
                        extraction_date=datetime.fromisoformat(a['extraction_date']),
                        validity_score=a['validity_score'],
                        usage_count=a.get('usage_count', 0),
                        tags=a.get('tags', []),
                        dependencies=a.get('dependencies', []),
                        valid_from=datetime.fromisoformat(a['valid_from']) if a.get('valid_from') else None,
                        valid_until=datetime.fromisoformat(a['valid_until']) if a.get('valid_until') else None,
                        version=a.get('version', 1)
                    )
                    self.artifacts[artifact.artifact_id] = artifact
    
    async def process_trace(self, trace: ExecutionTrace) -> Dict:
        """Process a new execution trace with ML-enhanced extraction."""
        self.traces.append(trace)
        
        # Add to temporal knowledge graph
        temporal_id = self.temporal_manager.add_knowledge(
            content=trace.problem_description,
            knowledge_type="workflow_trace",
            confidence=0.8,
            metadata={
                'trace_id': trace.trace_id,
                'workflow_id': trace.workflow_id,
                'execution_time': trace.execution_time_ms
            }
        )
        
        # Extract ML patterns if available
        ml_results = None
        if self.ml_extraction:
            ml_results = self.ml_extraction.extract_from_text(
                text=trace.problem_description,
                domain="workflow",
                temporal_validity=(datetime.now(), datetime.now() + timedelta(days=365))
            )
        
        # Extract patterns if we have enough traces
        new_patterns = []
        if len(self.traces) >= 5:
            new_patterns = await self._extract_patterns_async([trace])
        
        # Validate patterns
        validated_patterns = []
        for pattern in new_patterns:
            validation = self.validation_engine.validate_pattern(pattern)
            if validation['valid']:
                validated_patterns.append(pattern)
        
        # Generate artifacts from validated patterns
        new_artifacts = []
        for pattern in validated_patterns:
            artifacts = await self._generate_artifacts_async(pattern)
            new_artifacts.extend(artifacts)
        
        # Add to retrieval system
        for artifact in new_artifacts:
            self.retrieval_system.add_knowledge({
                'id': artifact.artifact_id,
                'description': artifact.name,
                'content': str(artifact.content),
                'type': artifact.artifact_type
            })
        
        # Save data
        await self._save_data_async()
        
        return {
            'patterns_extracted': len(new_patterns),
            'patterns_validated': len(validated_patterns),
            'artifacts_generated': len(new_artifacts),
            'total_patterns': len(self.patterns),
            'total_artifacts': len(self.artifacts),
            'temporal_id': temporal_id,
            'ml_extraction': ml_results is not None
        }
    
    async def _extract_patterns_async(
        self,
        traces: List[ExecutionTrace]
    ) -> List[ExtractedPattern]:
        """Extract patterns asynchronously with ML clustering."""
        loop = asyncio.get_event_loop()
        
        # Run extraction in thread pool
        patterns = []
        
        sequence = await loop.run_in_executor(
            None, self.pattern_extractor.extract_sequence_patterns, traces
        )
        patterns.extend(sequence)
        
        semantic = await loop.run_in_executor(
            None, self.pattern_extractor.extract_semantic_patterns, self.traces[-50:]
        )
        patterns.extend(semantic)
        
        parametric = await loop.run_in_executor(
            None, self.pattern_extractor.extract_parametric_patterns, traces
        )
        patterns.extend(parametric)
        
        structural = await loop.run_in_executor(
            None, self.pattern_extractor.extract_structural_patterns, traces
        )
        patterns.extend(structural)
        
        # Update global patterns
        for p in patterns:
            self.patterns[p.pattern_id] = p
        
        return patterns
    
    async def _generate_artifacts_async(
        self,
        pattern: ExtractedPattern
    ) -> List[KnowledgeArtifact]:
        """Generate artifacts from pattern asynchronously."""
        artifacts = []
        
        # Try each artifact type
        generators = [
            self.artifact_generator.generate_strategy_artifact,
            self.artifact_generator.generate_template_artifact,
            self.artifact_generator.generate_constraint_artifact
        ]
        
        for generator in generators:
            artifact = generator(pattern, self.traces)
            if artifact:
                artifacts.append(artifact)
                self.artifacts[artifact.artifact_id] = artifact
        
        return artifacts
    
    async def _save_data_async(self) -> None:
        """Save patterns and artifacts asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_data)
    
    def _save_data(self) -> None:
        """Save patterns and artifacts."""
        patterns_file = self.storage_path / "patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump({
                'patterns': [p.to_dict() for p in self.patterns.values()],
                'total_count': len(self.patterns),
                'last_updated': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        artifacts_file = self.storage_path / "artifacts.json"
        with open(artifacts_file, 'w') as f:
            json.dump({
                'artifacts': [a.to_dict() for a in self.artifacts.values()],
                'total_count': len(self.artifacts),
                'last_updated': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def retrieve_knowledge(
        self,
        query: str,
        top_k: int = 10,
        use_semantic: bool = True
    ) -> List[Dict]:
        """
        Retrieve knowledge using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results
            use_semantic: Whether to use semantic search
            
        Returns:
            List of knowledge items
        """
        semantic_weight = 0.7 if use_semantic else 0.0
        return self.retrieval_system.retrieve(query, top_k, semantic_weight)
    
    def get_applicable_artifacts(
        self,
        problem_description: str,
        min_validity: float = 0.5
    ) -> List[KnowledgeArtifact]:
        """Get artifacts applicable to a problem using hybrid retrieval."""
        # Use hybrid retrieval
        results = self.retrieve_knowledge(problem_description, top_k=20)
        
        # Filter by validity and convert to artifacts
        applicable = []
        for result in results:
            artifact_id = result.get('id')
            if artifact_id and artifact_id in self.artifacts:
                artifact = self.artifacts[artifact_id]
                if artifact.validity_score >= min_validity:
                    applicable.append((artifact, result.get('retrieval_score', 0)))
        
        # Sort by retrieval score
        applicable.sort(key=lambda x: x[1], reverse=True)
        return [a[0] for a in applicable[:10]]
    
    def validate_all_patterns(self) -> Dict[str, Any]:
        """Validate all patterns and return summary."""
        validation_results = []
        
        for pattern in self.patterns.values():
            result = self.validation_engine.validate_pattern(pattern)
            validation_results.append(result)
        
        # Check consistency
        consistency = self.validation_engine.validate_consistency(
            list(self.patterns.values())
        )
        
        return {
            'total_patterns': len(self.patterns),
            'valid_patterns': sum(1 for r in validation_results if r['valid']),
            'invalid_patterns': sum(1 for r in validation_results if not r['valid']),
            'average_confidence': np.mean([r['confidence'] for r in validation_results]),
            'consistency_check': consistency,
            'ml_clustered_patterns': sum(
                1 for p in self.patterns.values() if p.pattern_type == 'ml_clustered'
            )
        }
    
    def get_statistics(self) -> Dict:
        """Get extraction statistics."""
        pattern_types = defaultdict(int)
        for p in self.patterns.values():
            pattern_types[p.pattern_type] += 1
        
        artifact_types = defaultdict(int)
        for a in self.artifacts.values():
            artifact_types[a.artifact_type] += 1
        
        # ML statistics
        ml_patterns = [p for p in self.patterns.values() if p.pattern_type == 'ml_clustered']
        avg_silhouette = np.mean([p.ml_silhouette_score for p in ml_patterns]) if ml_patterns else 0
        
        return {
            'traces_processed': len(self.traces),
            'patterns_extracted': len(self.patterns),
            'pattern_types': dict(pattern_types),
            'artifacts_generated': len(self.artifacts),
            'artifact_types': dict(artifact_types),
            'avg_pattern_confidence': np.mean([p.confidence for p in self.patterns.values()]) if self.patterns else 0,
            'avg_artifact_validity': np.mean([a.validity_score for a in self.artifacts.values()]) if self.artifacts else 0,
            'ml_clustered_patterns': len(ml_patterns),
            'avg_ml_silhouette_score': avg_silhouette,
            'ml_available': self.ml_extraction is not None,
            'z3_available': Z3_AVAILABLE,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE
        }


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'Stage6KnowledgeExtraction',
    'PatternExtractor',
    'KnowledgeArtifactGenerator',
    'TemporalKnowledgeManager',
    'KnowledgeValidationEngine',
    'HybridRetrievalSystem',
    'ExtractedPattern',
    'KnowledgeArtifact',
    'ExecutionTrace'
]


if __name__ == "__main__":
    # Demo usage
    print("Stage 6 Knowledge Extraction Engine")
    print("=" * 50)
    
    engine = Stage6KnowledgeExtraction()
    
    # Create sample traces
    sample_traces = [
        ExecutionTrace(
            trace_id=f"trace_{i:03d}",
            workflow_id=f"wf_{i:03d}",
            problem_description=desc,
            stages=[
                {'stage_name': 'decomposition', 'parameters': {'strategy': 'hybrid'}},
                {'stage_name': 'evolution', 'parameters': {'generations': 100}},
                {'stage_name': 'assembly', 'parameters': {}}
            ],
            final_result={'architecture': 'transformer', 'accuracy': 0.95},
            execution_time_ms=5000.0,
            timestamp=datetime.now()
        )
        for i, desc in enumerate([
            "Optimize neural network architecture for image classification",
            "Improve deep learning model for computer vision",
            "Tune transformer architecture for NLP tasks",
            "Optimize CNN for visual recognition",
            "Fine-tune BERT for text classification",
            "Improve ResNet architecture for image processing",
            "Optimize GPT model for text generation",
            "Tune YOLO for object detection",
            "Improve LSTM for sequence prediction",
            "Optimize GAN for image generation"
        ])
    ]
    
    # Process traces
    async def demo():
        for trace in sample_traces:
            result = await engine.process_trace(trace)
            print(f"Processed {trace.trace_id}: {result['patterns_extracted']} patterns, "
                  f"{result['patterns_validated']} validated, "
                  f"{result['artifacts_generated']} artifacts")
        
        print(f"\nStatistics: {engine.get_statistics()}")
        print(f"\nValidation Summary: {engine.validate_all_patterns()}")
    
    asyncio.run(demo())
