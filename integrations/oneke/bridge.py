"""
OneKE Bridge for OpenEvolve Workflow Knowledge Extractor

This module bridges OneKE's schema-guided extraction capabilities with
OpenEvolve's workflow knowledge extractor. It provides high-level methods
for extracting domain-specific knowledge from workflow executions.

Key Capabilities:
- Physics domain knowledge extraction (GAP-2)
- Chemical entity and relation extraction
- Solution pattern extraction with schemas
- Team performance insights from workflow data
- Integration with workflow_knowledge_extractor.py
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json

from .adapter import OneKEAdapter
from ..base.extraction_interface import SchemaDefinition, ExtractionResult
try:
    from workflow_structures import WorkflowState
except ImportError:
    WorkflowState = Any

logger = logging.getLogger(__name__)

class OneKEBridge:
    """
    Bridge between OneKE and OpenEvolve workflow knowledge extractor.

    This class provides convenient methods for extracting knowledge from
    workflow executions using OneKE's schema-guided extraction.

    Attributes:
        adapter: OneKEAdapter instance
        schemas: Loaded schema definitions
        cache: Optional cache for extraction results
    """

    def __init__(self, adapter: Optional[OneKEAdapter] = None, config_path: Optional[str] = None):
        """
        Initialize the OneKE bridge.

        Args:
            adapter: Optional OneKEAdapter instance
            config_path: Optional path to OneKE config.yaml
        """
        self.adapter = adapter or OneKEAdapter(config_path)
        self.schemas: Dict[str, SchemaDefinition] = {}
        self.cache: Dict[str, ExtractionResult] = {}
        self.logger = logging.getLogger(f"{__name__}.OneKEBridge")

    async def initialize(self) -> bool:
        """
        Initialize the bridge and adapter.

        Returns:
            True if initialization successful
        """
        try:
            # Initialize adapter
            if not await self.adapter.initialize():
                return False

            # Load schemas
            await self._load_schemas()

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize OneKE bridge: {e}")
            return False

    async def _load_schemas(self) -> None:
        """Load default schemas for OpenEvolve domains."""
        schema_dir = Path(__file__).parent / 'schemas'

        if not schema_dir.exists():
            self.logger.warning(f"Schema directory not found: {schema_dir}")
            return

        for schema_file in schema_dir.glob('*.yaml'):
            try:
                schema = self.adapter.load_schema(str(schema_file))
                self.schemas[schema.name] = schema
                self.logger.info(f"Loaded schema: {schema.name}")
            except Exception as e:
                self.logger.error(f"Failed to load schema {schema_file.name}: {e}")

    async def extract_from_workflow(
        self,
        workflow: Union[WorkflowState, Dict[str, Any]],
        schemas: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict[str, ExtractionResult]:
        """
        Extract knowledge from a workflow execution.

        Args:
            workflow: WorkflowState or workflow dictionary
            schemas: Optional list of schema names to apply
            use_cache: Whether to use cached results

        Returns:
            Dictionary mapping schema names to extraction results
        """
        # Convert WorkflowState to dict if needed
        if hasattr(workflow, 'workflow_id'):
            workflow_dict = self._workflow_to_dict(workflow)
        else:
            workflow_dict = workflow

        # Determine which schemas to use
        if schemas is None:
            schemas = list(self.schemas.keys())

        # Apply schemas
        results = {}
        for schema_name in schemas:
            if schema_name not in self.schemas:
                self.logger.warning(f"Schema {schema_name} not found, skipping")
                continue

            schema = self.schemas[schema_name]

            # Check cache
            cache_key = f"{workflow_dict.get('workflow_id', 'unknown')}_{schema_name}"
            if use_cache and cache_key in self.cache:
                results[schema_name] = self.cache[cache_key]
                continue

            try:
                result = await self.adapter.extract_schema_guided(
                    text=self._workflow_to_text(workflow_dict),
                    schema=schema
                )
                results[schema_name] = result

                # Cache result
                if use_cache:
                    self.cache[cache_key] = result

            except Exception as e:
                self.logger.error(f"Failed to extract with schema {schema_name}: {e}")
                results[schema_name] = ExtractionResult(
                    extraction_type='schema',
                    entities=[], relations=[], events=[], triples=[],
                    schema=schema.__dict__,
                    confidence=0.0,
                    metadata={'error': str(e)}
                )

        return results

    async def extract_physics_knowledge(
        self,
        workflow: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract physics domain knowledge from workflow.
        """
        if 'physics_concepts' not in self.schemas:
            self.logger.warning("Physics concepts schema not loaded")
            return {}

        result = await self.extract_from_workflow(workflow, ['physics_concepts'])
        extraction = result.get('physics_concepts')

        if not extraction:
            return {}

        knowledge = {
            'concepts': [],
            'observables': [],
            'dynamics': [],
            'quantum': [],
            'confidence': extraction.confidence
        }

        for entity in extraction.entities:
            category = entity.get('category', 'concepts')
            if category in knowledge:
                knowledge[category].append(entity)

        return knowledge

    async def extract_chemistry_knowledge(
        self,
        workflow: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract chemical knowledge from workflow.
        """
        if 'chemical_entities' not in self.schemas:
            self.logger.warning("Chemical entities schema not loaded")
            return {}

        result = await self.extract_from_workflow(workflow, ['chemical_entities'])
        extraction = result.get('chemical_entities')

        if not extraction:
            return {}

        knowledge = {
            'substances': [],
            'reactions': [],
            'properties': [],
            'confidence': extraction.confidence
        }

        for entity in extraction.entities:
            category = entity.get('category', 'substances')
            if category in knowledge:
                knowledge[category].append(entity)

        return knowledge

    async def extract_relations(
        self,
        workflow: Union[WorkflowState, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relations between concepts from workflow.
        """
        if 'relations' not in self.schemas:
            self.logger.warning("Relations schema not loaded")
            return []

        result = await self.extract_from_workflow(workflow, ['relations'])
        extraction = result.get('relations')

        if not extraction:
            return []

        return extraction.relations

    async def extract_solution_patterns(
        self,
        workflow: Union[WorkflowState, Dict[str, Any]],
        domain: str = 'general'
    ) -> Dict[str, Any]:
        """
        Extract solution patterns using schema-guided extraction.
        """
        workflow_text = self._workflow_to_text(workflow)

        pattern_schema = SchemaDefinition(
            name=f'solution_patterns_{domain}',
            description=f'Solution patterns for {domain} domain',
            entity_types=[
                {'name': 'pattern', 'description': 'Solution pattern'},
                {'name': 'approach', 'description': 'High-level approach'},
                {'name': 'technique', 'description': 'Specific technique'}
            ],
            relation_types=[
                {'name': 'uses', 'description': 'Pattern uses technique'},
                {'name': 'implements', 'description': 'Approach implements pattern'}
            ]
        )

        try:
            result = await self.adapter.extract_schema_guided(
                text=workflow_text,
                schema=pattern_schema
            )

            return {
                'patterns': [e for e in result.entities if e.get('type') == 'pattern'],
                'approaches': [e for e in result.entities if e.get('type') == 'approach'],
                'techniques': [e for e in result.entities if e.get('type') == 'technique'],
                'relations': result.relations,
                'confidence': result.confidence
            }

        except Exception as e:
            self.logger.error(f"Failed to extract solution patterns: {e}")
            return {}

    async def extract_team_insights(
        self,
        workflow: Union[WorkflowState, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract team performance insights from workflow.
        """
        workflow_dict = self._workflow_to_dict(workflow)

        insights = {
            'team_composition': [],
            'velocity': None,
            'quality_metrics': {},
            'optimal_domains': []
        }

        if 'solver_team' in workflow_dict:
            team = workflow_dict['solver_team']
            insights['team_composition'] = team.get('models', [])

        if 'relations' in self.schemas:
            result = await self.extract_from_workflow(workflow, ['relations'])
            extraction = result.get('relations')

            if extraction and hasattr(extraction, 'relations'):
                insights['team_relations'] = [
                    r for r in extraction.relations
                    if 'team' in r.get('type', '').lower()
                ]

        return insights

    async def batch_extract_from_workflows(
        self,
        workflows: List[Union[WorkflowState, Dict[str, Any]]],
        schemas: Optional[List[str]] = None
    ) -> List[Dict[str, ExtractionResult]]:
        """
        Extract knowledge from multiple workflows.
        """
        tasks = [
            self.extract_from_workflow(workflow, schemas)
            for workflow in workflows
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to extract from workflow {i}: {result}")
                processed_results.append({})
            else:
                processed_results.append(result)

        return processed_results

    async def validate_integration(self) -> Dict[str, Any]:
        """
        Validate OneKE integration.
        """
        validation = await self.adapter.validate()
        bridge_checks = []

        if self.schemas:
            bridge_checks.append({
                'name': 'schemas_loaded',
                'status': 'passed',
                'count': len(self.schemas),
                'schemas': list(self.schemas.keys())
            })
        else:
            bridge_checks.append({
                'name': 'schemas_loaded',
                'status': 'failed',
                'count': 0
            })

        validation['bridge'] = bridge_checks
        return validation

    def _workflow_to_dict(self, workflow: Union[WorkflowState, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert WorkflowState to dictionary."""
        if isinstance(workflow, dict):
            return workflow

        return {
            'workflow_id': getattr(workflow, 'workflow_id', 'unknown'),
            'problem_statement': getattr(workflow, 'problem_statement', ''),
            'final_solution': str(getattr(workflow, 'final_solution', '')),
            'decomposition_plan': str(getattr(workflow, 'decomposition_plan', '')),
            'status': getattr(workflow, 'status', 'unknown'),
            'sub_problem_solutions': {
                k: str(v) for k, v in getattr(workflow, 'sub_problem_solutions', {}).items()
            }
        }

    def _workflow_to_text(self, workflow: Union[WorkflowState, Dict[str, Any]]) -> str:
        """Convert workflow to text for extraction."""
        workflow_dict = self._workflow_to_dict(workflow)

        parts = []
        if 'problem_statement' in workflow_dict:
            parts.append(f"Problem Statement:\n{workflow_dict['problem_statement']}")
        if 'final_solution' in workflow_dict and workflow_dict['final_solution']:
            parts.append(f"Final Solution:\n{workflow_dict['final_solution']}")
        if 'decomposition_plan' in workflow_dict and workflow_dict['decomposition_plan']:
            parts.append(f"Decomposition Plan:\n{workflow_dict['decomposition_plan']}")
        if 'sub_problem_solutions' in workflow_dict:
            parts.append("Sub-problem Solutions:")
            for sub_id, solution in workflow_dict['sub_problem_solutions'].items():
                parts.append(f"\n{sub_id}:\n{solution}")

        return '\n\n'.join(parts)

    async def shutdown(self) -> bool:
        """
        Shutdown the bridge and adapter.
        """
        self.cache.clear()
        return await self.adapter.shutdown()


# Convenience functions for workflow_knowledge_extractor.py integration

async def create_oneke_bridge(config_path: Optional[str] = None) -> OneKEBridge:
    """
    Create and initialize OneKE bridge.
    """
    bridge = OneKEBridge(config_path=config_path)
    await bridge.initialize()
    return bridge


async def extract_domain_knowledge(
    workflow: Union[WorkflowState, Dict[str, Any]],
    domains: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Extract domain knowledge from workflow.
    """
    if domains is None:
        domains = ['physics', 'chemistry']

    bridge = await create_oneke_bridge()
    knowledge = {}

    if 'physics' in domains:
        knowledge['physics'] = await bridge.extract_physics_knowledge(workflow)
    if 'chemistry' in domains:
        knowledge['chemistry'] = await bridge.extract_chemistry_knowledge(workflow)

    await bridge.shutdown()
    return knowledge