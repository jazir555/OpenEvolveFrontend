"""
Knowledge Manager Module
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from workflow_structures import KnowledgeArtifact, WorkflowState, PerformanceMetrics


class KnowledgeManager:
    def __init__(self, storage_path: str = "./knowledge_base"):
        self.storage_path = storage_path
        self.artifacts_file = os.path.join(storage_path, "knowledge_artifacts.json")
        self.metrics_file = os.path.join(storage_path, "performance_metrics.json")
        os.makedirs(storage_path, exist_ok=True)
        self.artifacts: Dict[str, KnowledgeArtifact] = self._load_artifacts()
        self.metrics: List[PerformanceMetrics] = self._load_metrics()
    
    def _load_artifacts(self) -> Dict[str, KnowledgeArtifact]:
        if not os.path.exists(self.artifacts_file):
            return {}
        try:
            with open(self.artifacts_file, 'r') as f:
                data = json.load(f)
                artifacts = {}
                for artifact_id, artifact_data in data.items():
                    artifacts[artifact_id] = KnowledgeArtifact(**artifact_data)
                return artifacts
        except Exception as e:
            print(f"Error loading knowledge artifacts: {e}")
            return {}
    
    def _load_metrics(self) -> List[PerformanceMetrics]:
        if not os.path.exists(self.metrics_file):
            return []
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                return [PerformanceMetrics(**metric_data) for metric_data in data]
        except Exception as e:
            print(f"Error loading performance metrics: {e}")
            return []
    
    def _save_artifacts(self):
        try:
            data = {}
            for artifact_id, artifact in self.artifacts.items():
                data[artifact_id] = {
                    'id': artifact.id,
                    'artifact_type': artifact.artifact_type,
                    'content': artifact.content,
                    'source_workflow_id': artifact.source_workflow_id,
                    'extraction_timestamp': artifact.extraction_timestamp,
                    'domain': artifact.domain,
                    'problem_type': artifact.problem_type,
                    'usage_count': artifact.usage_count,
                    'effectiveness_score': artifact.effectiveness_score,
                    'related_artifacts': artifact.related_artifacts
                }
            with open(self.artifacts_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving knowledge artifacts: {e}")
    
    def _save_metrics(self):
        try:
            data = []
            for metric in self.metrics:
                data.append({
                    'entity_type': metric.entity_type,
                    'entity_id': metric.entity_id,
                    'metrics': metric.metrics,
                    'timestamp': metric.timestamp,
                    'domain': metric.domain,
                    'problem_type': metric.problem_type,
                    'context': metric.context
                })
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving performance metrics: {e}")
    
    def store_knowledge_artifact(self, artifact: KnowledgeArtifact):
        self.artifacts[artifact.id] = artifact
        self._save_artifacts()
    
    def retrieve_relevant_knowledge(self, problem_statement: str, domain: Optional[str] = None, problem_type: Optional[str] = None, artifact_types: Optional[List[str]] = None, limit: int = 10) -> List[KnowledgeArtifact]:
        relevant_artifacts = []
        for artifact in self.artifacts.values():
            if domain and artifact.domain and artifact.domain != domain:
                continue
            if problem_type and artifact.problem_type and artifact.problem_type != problem_type:
                continue
            if artifact_types and artifact.artifact_type not in artifact_types:
                continue
            relevance_score = self._calculate_relevance(problem_statement, artifact)
            if relevance_score > 0:
                relevant_artifacts.append((relevance_score, artifact))
        relevant_artifacts.sort(key=lambda x: x[0], reverse=True)
        return [artifact for _, artifact in relevant_artifacts[:limit]]
    
    def _calculate_relevance(self, problem_statement: str, artifact: KnowledgeArtifact) -> float:
        problem_words = set(problem_statement.lower().split())
        artifact_text = json.dumps(artifact.content).lower()
        artifact_words = set(artifact_text.split())
        intersection = problem_words.intersection(artifact_words)
        union = problem_words.union(artifact_words)
        if not union:
            return 0.0
        relevance = len(intersection) / len(union)
        relevance *= (1 + artifact.effectiveness_score)
        return relevance
    
    def update_artifact_usage(self, artifact_id: str, was_effective: bool):
        if artifact_id in self.artifacts:
            artifact = self.artifacts[artifact_id]
            artifact.usage_count += 1
            alpha = 0.3
            new_score = 1.0 if was_effective else 0.0
            artifact.effectiveness_score = (alpha * new_score + (1 - alpha) * artifact.effectiveness_score)
            self._save_artifacts()
    
    def get_all_artifacts(self) -> List[KnowledgeArtifact]:
        return list(self.artifacts.values())
