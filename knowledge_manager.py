"""
Knowledge Manager Module
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from workflow_structures import KnowledgeArtifact, WorkflowState, PerformanceMetrics
from knowledge_engine.engine import KnowledgeEngine
from knowledge_engine.core import EntityKnowledgeGraph
from ace_knowledge_artifacts import SkillbookStore, create_refinement_template


class KnowledgeManager:
    def __init__(self, storage_path: str = "./knowledge_base"):
        self.storage_path = storage_path
        self.artifacts_file = os.path.join(storage_path, "knowledge_artifacts.json")
        self.metrics_file = os.path.join(storage_path, "performance_metrics.json")
        self.entity_graph_path = os.path.join(storage_path, "entity_graph.json")
        os.makedirs(storage_path, exist_ok=True)
        self.engine = KnowledgeEngine()
        self.main_index_path = os.path.join(self.storage_path, "main_index.json")
        self.knowledge_index: Dict[str, Any] = {}
        self._load_main_index()
        self.skillbook = SkillbookStore(os.path.join(storage_path, "ace_skillbook.json"))
        self.artifacts = self._load_artifacts()
        self.metrics = self._load_metrics()

    def _load_main_index(self):
        """Loads the main knowledge index if it exists."""
        if os.path.exists(self.main_index_path):
            print(f"📂 Loading main knowledge index from {self.main_index_path}")
            self.knowledge_index = self.engine.load_index(self.main_index_path)
        else:
            print(f"⚠️ Main knowledge index not found at {self.main_index_path}. Run reindex_knowledge_base() to create it.")

    async def reindex_knowledge_base(self):
        """(Re)Indexes the entire knowledge_base directory using the KnowledgeEngine."""
        print("🧠 Re-indexing the entire knowledge base...")
        # The target structure can be generic for a full re-index
        target_structure = "General knowledge base for software and problem solving."
        
        # The indexer will create a file like 'knowledge_base_index.json'
        # inside the 'knowledge_base' directory. We want to rename it to 'main_index.json'.
        index_output_dir = os.path.join(self.storage_path, "temp_index")

        output_files = await self.engine.index_project(
            project_path=self.storage_path,
            target_structure=target_structure,
            output_dir=index_output_dir
        )

        if output_files:
            # Find the created index file and move it
            for repo_name, index_path in output_files.items():
                # Move the first found index to be the main index
                os.rename(index_path, self.main_index_path)
                print(f"✅ New main index created at: {self.main_index_path}")
                break # We only expect one
        
        # Clean up the temporary directory
        if os.path.exists(index_output_dir):
            import shutil
            shutil.rmtree(index_output_dir)
            
        # Finally, load the new index into memory
        self._load_main_index()
    
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
        except (OSError, IOError, json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Error loading knowledge artifacts: {e}")
            return {}

    def _load_entity_graph(self) -> EntityKnowledgeGraph:
        if not os.path.exists(self.entity_graph_path):
            return EntityKnowledgeGraph()
        try:
            with open(self.entity_graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EntityKnowledgeGraph.from_dict(data)
        except (OSError, IOError, json.JSONDecodeError):
            return EntityKnowledgeGraph()

    def _save_entity_graph(self, graph: EntityKnowledgeGraph) -> None:
        try:
            with open(self.entity_graph_path, "w", encoding="utf-8") as f:
                json.dump(graph.to_dict(), f, indent=2)
        except (OSError, IOError, TypeError) as e:
            print(f"Error saving entity graph: {e}")
    
    def _load_metrics(self) -> List[PerformanceMetrics]:
        if not os.path.exists(self.metrics_file):
            return []
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                return [PerformanceMetrics(**metric_data) for metric_data in data]
        except (OSError, IOError, json.JSONDecodeError, TypeError, KeyError) as e:
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
        except (OSError, IOError, TypeError) as e:
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
        except (OSError, IOError, TypeError) as e:
            print(f"Error saving performance metrics: {e}")
    
    def _convert_summary_to_artifact(self, file_summary: Dict) -> KnowledgeArtifact:
        """Converts a FileSummary dictionary from the index into a KnowledgeArtifact."""
        content_hash = hashlib.md5(file_summary.get("summary", "").encode()).hexdigest()
        artifact_id = f"indexed_{content_hash}"
        
        return KnowledgeArtifact(
            id=artifact_id,
            artifact_type="indexed_file",
            content=file_summary,
            source_workflow_id="knowledge_engine_indexer",
            domain=file_summary.get("file_type"),
            problem_type=", ".join(file_summary.get("key_concepts", [])),
            effectiveness_score=0.5 # Default score for indexed items
        )

    def store_knowledge_artifact(self, artifact: KnowledgeArtifact):
        self.artifacts[artifact.id] = artifact
        self._save_artifacts()
        
        # Also save the content to a file so it can be indexed later
        try:
            if isinstance(artifact.content, dict) or isinstance(artifact.content, list):
                file_content = json.dumps(artifact.content, indent=2)
                extension = ".json"
            elif isinstance(artifact.content, str):
                file_content = artifact.content
                extension = ".txt"
            else:
                file_content = str(artifact.content)
                extension = ".txt"

            # Sanitize the artifact ID to create a valid filename
            sanitized_id = "".join(c for c in artifact.id if c.isalnum() or c in ('-', '_')).rstrip()
            file_path = os.path.join(self.storage_path, f"{sanitized_id}{extension}")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_content)

        except (OSError, IOError, TypeError) as e:
            print(f"Warning: Could not save artifact content to file for indexing. Error: {e}")

    def record_adr(self, adr: Dict[str, Any], entity_ids: Optional[List[str]] = None) -> KnowledgeArtifact:
        """Store an ADR as a knowledge artifact and link to entities."""
        artifact = KnowledgeArtifact(
            artifact_id=adr.get("decision_id", ""),
            artifact_type="adr",
            source_workflow_id=adr.get("workflow_id", "unknown"),
            source_stage=6,
            timestamp=datetime.now(),
            confidence=adr.get("confidence", 0.9),
            title=adr.get("title", "Architecture Decision Record"),
            description=adr.get("summary", "ADR synthesized on convergence."),
            content=adr,
            metadata={"adr_path": adr.get("adr_path")}
        )
        self.store_knowledge_artifact(artifact)

        if entity_ids:
            graph = self._load_entity_graph()
            for entity_id in entity_ids:
                graph.add_decision_link(entity_id, adr.get("decision_id", ""))
            self._save_entity_graph(graph)

        return artifact

    def store_refinement_template(
        self,
        title: str,
        description: str,
        reasoning_path: List[str],
        context_signature: Dict[str, Any],
        domain: str = "general"
    ) -> None:
        """Persist a refinement template to the ACE Skillbook."""
        template = create_refinement_template(
            title=title,
            description=description,
            reasoning_path=reasoning_path,
            context_signature=context_signature,
            domain=domain
        )
        self.skillbook.add_template(template)

    
    def retrieve_relevant_knowledge(self, problem_statement: str, domain: Optional[str] = None, problem_type: Optional[str] = None, artifact_types: Optional[List[str]] = None, limit: int = 10) -> List[KnowledgeArtifact]:
        
        # --- Hybrid Search ---
        # 1. Search structured artifacts (original method)
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
        
        # 2. Search indexed files via KnowledgeEngine
        if self.knowledge_index:
            indexed_results = self.engine.query_index_by_keyword(self.knowledge_index, problem_statement)
            for summary in indexed_results:
                # Convert summary to artifact and add to list
                converted_artifact = self._convert_summary_to_artifact(summary)
                # Calculate relevance for ranking
                relevance_score = self._calculate_relevance(problem_statement, converted_artifact)
                relevant_artifacts.append((relevance_score, converted_artifact))

        # 3. Merge and rank results
        relevant_artifacts.sort(key=lambda x: x[0], reverse=True)
        
        # Remove duplicates by ID
        final_artifacts = {}
        for _, artifact in relevant_artifacts:
            if artifact.id not in final_artifacts:
                final_artifacts[artifact.id] = artifact
        
        return list(final_artifacts.values())[:limit]
    
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
    
    def get_performance_metrics(self, entity_type: Optional[str] = None, limit: Optional[int] = None) -> List[PerformanceMetrics]:
        """Get performance metrics, filtered by entity type if provided"""
        filtered_metrics = self.metrics
        if entity_type:
            filtered_metrics = [m for m in self.metrics if m.entity_type == entity_type]
        if limit:
            filtered_metrics = filtered_metrics[-limit:]  # Get most recent 'limit' metrics
        return filtered_metrics
    
    def delete_artifact(self, artifact_id: str) -> bool:
        if artifact_id in self.artifacts:
            del self.artifacts[artifact_id]
            self._save_artifacts()
            return True
        return False
    
    def clear_all_artifacts(self):
        self.artifacts = {}
        self._save_artifacts()
    
    def export_knowledge_base(self, file_path: str):
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
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_knowledge_base(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.artifacts = {}
            for artifact_id, artifact_data in data.items():
                self.artifacts[artifact_id] = KnowledgeArtifact(**artifact_data)
        self._save_artifacts()
    
    def apply_learned_patterns(self, problem_statement: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """Apply learned patterns to recommend solutions for a new problem."""
        recommendations = {
            "recommended_approaches": [],
            "similar_problems": [],
            "team_recommendations": [],
            "gauntlet_recommendations": []
        }
        
        # Look for similar problems in the knowledge base
        similar_artifacts = self.retrieve_relevant_knowledge(
            problem_statement=problem_statement,
            domain=domain,
            artifact_types=["problem_solution_mapping", "solution_pattern"],
            limit=10
        )
        
        approaches = set()
        for artifact in similar_artifacts:
            if artifact.artifact_type == "solution_pattern":
                approaches.add(artifact.content.get("solution_approach", "N/A"))
        
        for approach in approaches:
            recommendations["recommended_approaches"].append({
                "approach": approach,
                "effectiveness": 0.7,  # Default effectiveness
                "source": "multiple"
            })
        
        # Find similar problems and their solutions
        for artifact in similar_artifacts:
            if artifact.artifact_type == "problem_solution_mapping":
                recommendations["similar_problems"].append({
                    "problem": artifact.content.get("problem_statement", ""),
                    "decomposition_strategy": artifact.content.get("decomposition_strategy", {}),
                    "source": artifact.source_workflow_id
                })
        
        return recommendations
