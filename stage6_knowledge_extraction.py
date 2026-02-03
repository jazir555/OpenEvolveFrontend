"""
Stage 6 Knowledge Extraction - License: Apache 2.0

Advanced knowledge extraction system for OpenEvolve workflows.
Extracts patterns, insights, and reusable knowledge from execution traces.

Dependencies (all permissive licenses):
- numpy: BSD License
- scikit-learn: BSD License
- networkx: BSD License

Author: OpenEvolve
Date: 2026-02-02
"""



import json
import re
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
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


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ExtractedPattern:
    """A pattern extracted from workflow execution."""
    pattern_id: str
    pattern_type: str  # 'sequence', 'semantic', 'parametric', 'structural'
    description: str
    confidence: float  # 0.0 to 1.0
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    examples: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
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
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'extraction_date': self.extraction_date.isoformat()
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
    """Extracts patterns from execution traces."""
    
    PATTERN_TYPES = ['sequence', 'semantic', 'parametric', 'structural']
    
    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        self.patterns: Dict[str, ExtractedPattern] = {}
    
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
                        description=f"Common stage sequence: {' → '.join(seq)}",
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
        """Extract semantic patterns from problem descriptions."""
        if not SKLEARN_AVAILABLE:
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
        if pattern.pattern_type != 'sequence':
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
        if pattern.pattern_type != 'structural':
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
# STAGE 6 KNOWLEDGE EXTRACTION ENGINE
# =============================================================================

class Stage6KnowledgeExtraction:
    """
    Stage 6 Knowledge Extraction Engine.
    
    Extracts knowledge from completed workflows to improve future executions.
    Implements pattern recognition, artifact generation, and knowledge management.
    
    License: Apache 2.0
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("knowledge_extraction")
        self.storage_path.mkdir(exist_ok=True)
        
        self.pattern_extractor = PatternExtractor()
        self.artifact_generator = KnowledgeArtifactGenerator()
        
        self.traces: List[ExecutionTrace] = []
        self.patterns: Dict[str, ExtractedPattern] = {}
        self.artifacts: Dict[str, KnowledgeArtifact] = {}
        
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
                        metadata=p.get('metadata', {})
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
                        dependencies=a.get('dependencies', [])
                    )
                    self.artifacts[artifact.artifact_id] = artifact
    
    async def process_trace(self, trace: ExecutionTrace) -> Dict:
        """Process a new execution trace."""
        self.traces.append(trace)
        
        # Extract patterns if we have enough traces
        new_patterns = []
        if len(self.traces) >= 5:
            new_patterns = await self._extract_patterns_async([trace])
        
        # Generate artifacts from new patterns
        new_artifacts = []
        for pattern in new_patterns:
            artifacts = await self._generate_artifacts_async(pattern)
            new_artifacts.extend(artifacts)
        
        # Save data
        await self._save_data_async()
        
        return {
            'patterns_extracted': len(new_patterns),
            'artifacts_generated': len(new_artifacts),
            'total_patterns': len(self.patterns),
            'total_artifacts': len(self.artifacts)
        }
    
    async def _extract_patterns_async(
        self,
        traces: List[ExecutionTrace]
    ) -> List[ExtractedPattern]:
        """Extract patterns asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run extraction in thread pool
        patterns = []
        
        sequence = await loop.run_in_executor(
            None, self.pattern_extractor.extract_sequence_patterns, traces
        )
        patterns.extend(sequence)
        
        semantic = await loop.run_in_executor(
            None, self.pattern_extractor.extract_semantic_patterns, traces
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
    
    def get_applicable_artifacts(
        self,
        problem_description: str,
        min_validity: float = 0.5
    ) -> List[KnowledgeArtifact]:
        """Get artifacts applicable to a problem."""
        applicable = []
        
        for artifact in self.artifacts.values():
            if artifact.validity_score < min_validity:
                continue
            
            # Simple relevance check
            relevance = self._calculate_relevance(artifact, problem_description)
            if relevance > 0.3:
                applicable.append((artifact, relevance))
        
        # Sort by relevance
        applicable.sort(key=lambda x: x[1], reverse=True)
        return [a[0] for a in applicable[:10]]
    
    def _calculate_relevance(
        self,
        artifact: KnowledgeArtifact,
        problem: str
    ) -> float:
        """Calculate relevance score."""
        problem_words = set(problem.lower().split())
        
        # Check artifact name
        name_words = set(artifact.name.lower().split())
        name_overlap = len(problem_words & name_words) / len(name_words | problem_words)
        
        # Check tags
        tag_matches = sum(1 for tag in artifact.tags if tag.lower() in problem.lower())
        tag_score = tag_matches / max(len(artifact.tags), 1)
        
        return (name_overlap + tag_score) / 2
    
    def get_statistics(self) -> Dict:
        """Get extraction statistics."""
        pattern_types = defaultdict(int)
        for p in self.patterns.values():
            pattern_types[p.pattern_type] += 1
        
        artifact_types = defaultdict(int)
        for a in self.artifacts.values():
            artifact_types[a.artifact_type] += 1
        
        return {
            'traces_processed': len(self.traces),
            'patterns_extracted': len(self.patterns),
            'pattern_types': dict(pattern_types),
            'artifacts_generated': len(self.artifacts),
            'artifact_types': dict(artifact_types),
            'avg_pattern_confidence': np.mean([p.confidence for p in self.patterns.values()]) if self.patterns else 0,
            'avg_artifact_validity': np.mean([a.validity_score for a in self.artifacts.values()]) if self.artifacts else 0
        }


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'Stage6KnowledgeExtraction',
    'PatternExtractor',
    'KnowledgeArtifactGenerator',
    'ExtractedPattern',
    'KnowledgeArtifact',
    'ExecutionTrace'
]


if __name__ == "__main__":
    # Demo usage
    print("Stage 6 Knowledge Extraction Engine")
    print("=" * 50)
    
    engine = Stage6KnowledgeExtraction()
    
    # Create sample trace
    sample_trace = ExecutionTrace(
        trace_id="trace_001",
        workflow_id="wf_001",
        problem_description="Optimize neural network architecture",
        stages=[
            {'stage_name': 'decomposition', 'parameters': {'strategy': 'hybrid'}},
            {'stage_name': 'evolution', 'parameters': {'generations': 100}},
            {'stage_name': 'assembly', 'parameters': {}}
        ],
        final_result={'architecture': 'transformer', 'accuracy': 0.95},
        execution_time_ms=5000.0,
        timestamp=datetime.now()
    )
    
    # Process
    async def demo():
        result = await engine.process_trace(sample_trace)
        print(f"Processing result: {result}")
        print(f"\nStatistics: {engine.get_statistics()}")
    
    asyncio.run(demo())
