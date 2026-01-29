"""
Enhanced OneKE Bridge

This module provides an enhanced bridge that integrates reflection,
quality enhancement, and case-based learning into the OneKE integration.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from .bridge import OneKEBridge
from .adapter import OneKEAdapter
from .case_repository import OneKECaseRepository
from .reflection_agent import OneKEReflectionAgent
from .quality_enhancement import OneKEQualityEnhancer
from .case import Case, EnhancedResult

logger = logging.getLogger(__name__)


class EnhancedOneKEBridge(OneKEBridge):
    """
    Enhanced OneKE bridge with reflection and quality enhancement.

    This extends the base OneKEBridge with:
    - Reflection agent for self-improvement
    - Quality enhancement system
    - Case repository for learning
    - Feedback loop for continuous improvement
    """

    def __init__(
        self,
        adapter: Optional[OneKEAdapter] = None,
        config_path: Optional[str] = None,
        enhanced_config_path: Optional[str] = None
    ):
        """
        Initialize the enhanced OneKE bridge.

        Args:
            adapter: Optional OneKEAdapter instance
            config_path: Optional path to OneKE config.yaml
            enhanced_config_path: Optional path to enhanced config
        """
        super().__init__(adapter, config_path)

        # Load enhanced configuration
        self.enhanced_config = self._load_enhanced_config(enhanced_config_path)

        # Initialize components
        self.reflection_agent: Optional[OneKEReflectionAgent] = None
        self.quality_enhancer: Optional[OneKEQualityEnhancer] = None
        self.case_repository: Optional[OneKECaseRepository] = None

        self.logger = logging.getLogger(f"{__name__}.EnhancedOneKEBridge")

    async def initialize(self) -> bool:
        """
        Initialize the enhanced bridge and all components.

        Returns:
            True if initialization successful
        """
        try:
            # Initialize base bridge
            if not await super().initialize():
                return False

            # Initialize case repository
            self.case_repository = OneKECaseRepository(
                storage_path=self.enhanced_config.get(
                    'case_repository',
                    {}
                ).get('storage_path', 'data/oneke_cases.json'),
                embedding_model=self.enhanced_config.get(
                    'case_repository',
                    {}
                ).get('embedding_model'),
                auto_save=self.enhanced_config.get(
                    'case_repository',
                    {}
                ).get('auto_save', True),
                save_interval=self.enhanced_config.get(
                    'case_repository',
                    {}
                ).get('save_interval', 100)
            )

            if not await self.case_repository.initialize():
                self.logger.warning("Case repository initialization failed")
                return False

            # Initialize reflection agent
            reflection_config = self.enhanced_config.get('reflection', {})
            self.reflection_agent = OneKEReflectionAgent(
                oneke_adapter=self.adapter,
                case_repository=self.case_repository,
                reflection_iterations=reflection_config.get('iterations', 3),
                num_samples=reflection_config.get('num_samples', 3),
                temperature=reflection_config.get('temperature', 0.3)
            )

            # Initialize quality enhancer
            quality_config = self.enhanced_config.get('quality_enhancement', {})
            self.quality_enhancer = OneKEQualityEnhancer(
                reflection_agent=self.reflection_agent,
                min_quality_threshold=quality_config.get(
                    'min_quality_threshold',
                    0.7
                )
            )

            self.logger.info("Enhanced OneKE bridge initialized successfully")

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced bridge: {e}")
            return False

    async def extract_with_enhancement(
        self,
        text: str,
        schema: Union[str, Dict[str, Any]],
        domain: str = "general",
        enable_reflection: bool = True,
        enable_cases: bool = True,
        enable_validation: bool = True,
        enable_consistency: bool = True
    ) -> EnhancedResult:
        """
        Extract knowledge with full enhancement pipeline.

        Pipeline:
        1. Initial extraction (OneKE)
        2. Schema validation
        3. Reflection-based improvement
        4. Case-based retrieval
        5. Consistency checking
        6. Quality scoring
        7. Store in repository (if high quality)

        Args:
            text: Input text
            schema: Target schema (name or dict)
            domain: Domain label
            enable_reflection: Enable reflection strategy
            enable_cases: Enable case-based learning
            enable_validation: Enable validation strategy
            enable_consistency: Enable consistency checking

        Returns:
            EnhancedResult with improved extraction
        """
        try:
            self.logger.info(f"Starting enhanced extraction for domain: {domain}")

            # Step 1: Initial extraction
            if isinstance(schema, str):
                # Use schema from loaded schemas
                if schema in self.schemas:
                    schema_obj = self.schemas[schema]
                else:
                    # Create basic schema
                    from ..base.extraction_interface import SchemaDefinition
                    schema_obj = SchemaDefinition(
                        name=schema,
                        description=f"Schema for {schema}"
                    )
            else:
                # Schema is already a dict/object
                schema_obj = schema

            initial_result = await self.adapter.extract_schema_guided(
                text=text,
                schema=schema_obj
            )

            # Convert to dict format
            raw_extraction = {
                'entities': initial_result.entities,
                'relations': initial_result.relations,
                'events': initial_result.events,
                'triples': initial_result.triples,
                'confidence': initial_result.confidence
            }

            # Step 2-6: Apply enhancement strategies
            strategies = []
            if enable_reflection:
                strategies.append('reflection')
            if enable_validation:
                strategies.append('validation')
            if enable_cases:
                strategies.append('cases')
            if enable_consistency:
                strategies.append('consistency')

            enhanced_result = await self.quality_enhancer.enhance_extraction(
                raw_extraction=raw_extraction,
                text=text,
                schema=schema if isinstance(schema, str) else schema_obj.name,
                domain=domain,
                strategies=strategies
            )

            # Step 7: Store in repository if high quality
            if enhanced_result.quality_score.overall >= 0.7:
                await self._store_high_quality_case(
                    text=text,
                    extraction=enhanced_result.extraction,
                    schema=schema if isinstance(schema, str) else schema_obj.name,
                    domain=domain,
                    quality_score=enhanced_result.quality_score.overall,
                    metadata={
                        'strategies_applied': enhanced_result.strategies_applied,
                        'quality_improvement': enhanced_result.quality_improvement
                    }
                )

            return enhanced_result

        except Exception as e:
            self.logger.error(f"Enhanced extraction failed: {e}")
            # Return minimal result
            from .case import QualityScore
            neutral_quality = QualityScore(
                completeness=0.0,
                accuracy=0.0,
                consistency=0.0,
                confidence=0.0,
                overall=0.0
            )

            return EnhancedResult(
                extraction={},
                quality_score=neutral_quality,
                original_quality=neutral_quality,
                quality_improvement=0.0,
                strategies_applied=[],
                metadata={'error': str(e)}
            )

    async def extract_and_learn(
        self,
        text: str,
        schema: str,
        domain: str = "general",
        feedback: Optional[Dict[str, Any]] = None
    ) -> EnhancedResult:
        """
        Extract and learn from feedback.

        Args:
            text: Input text
            schema: Target schema
            domain: Domain label
            feedback: Human feedback on extraction quality
                - 'correctness': Correctness score (0-1)
                - 'completeness': Completeness score (0-1)
                - 'comments': Optional comments

        Process:
        1. Extract knowledge
        2. Apply enhancement
        3. If feedback provided, update case repository
        4. Track improvement over time

        Returns:
            EnhancedResult with learning metadata
        """
        try:
            self.logger.info(f"Extracting and learning for domain: {domain}")

            # Extract with enhancement
            result = await self.extract_with_enhancement(
                text=text,
                schema=schema,
                domain=domain,
                enable_reflection=True,
                enable_cases=True,
                enable_validation=True,
                enable_consistency=True
            )

            # Process feedback if provided
            if feedback:
                await self._process_feedback(
                    text=text,
                    extraction=result.extraction,
                    schema=schema,
                    domain=domain,
                    feedback=feedback,
                    base_quality=result.quality_score.overall
                )

                # Add feedback to metadata
                result.metadata['feedback_received'] = feedback
                result.metadata['learning_occurred'] = True

            return result

        except Exception as e:
            self.logger.error(f"Extract and learn failed: {e}")
            raise

    async def batch_extract_with_enhancement(
        self,
        texts: List[str],
        schema: str,
        domain: str = "general",
        enable_enhancement: bool = True
    ) -> List[EnhancedResult]:
        """
        Extract knowledge from multiple texts with enhancement.

        Args:
            texts: List of input texts
            schema: Target schema
            domain: Domain label
            enable_enhancement: Whether to apply enhancement

        Returns:
            List of EnhancedResult objects
        """
        self.logger.info(
            f"Batch extracting with enhancement ({len(texts)} texts)"
        )

        tasks = [
            self.extract_with_enhancement(
                text=text,
                schema=schema,
                domain=domain,
                enable_reflection=enable_enhancement,
                enable_cases=enable_enhancement,
                enable_validation=enable_enhancement,
                enable_consistency=enable_enhancement
            )
            for text in texts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Text {i} extraction failed: {result}")
                # Create neutral result
                from .case import QualityScore
                neutral_quality = QualityScore(
                    completeness=0.0,
                    accuracy=0.0,
                    consistency=0.0,
                    confidence=0.0,
                    overall=0.0
                )

                processed_results.append(EnhancedResult(
                    extraction={},
                    quality_score=neutral_quality,
                    original_quality=neutral_quality,
                    quality_improvement=0.0,
                    strategies_applied=[],
                    metadata={'error': str(result)}
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def get_repository_statistics(self) -> Dict[str, Any]:
        """Get case repository statistics."""
        if not self.case_repository:
            return {'error': 'Case repository not initialized'}

        stats = await self.case_repository.get_statistics()
        return stats.to_dict()

    async def export_repository(self, output_path: str) -> bool:
        """Export case repository to file."""
        if not self.case_repository:
            self.logger.error("Case repository not initialized")
            return False

        try:
            await self.case_repository.export_cases(output_path, format="json")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export repository: {e}")
            return False

    async def import_repository(self, input_path: str) -> bool:
        """Import cases from file to repository."""
        if not self.case_repository:
            self.logger.error("Case repository not initialized")
            return False

        try:
            await self.case_repository.import_cases(input_path, format="json")
            return True
        except Exception as e:
            self.logger.error(f"Failed to import repository: {e}")
            return False

    async def shutdown(self) -> bool:
        """
        Shutdown the enhanced bridge and save all data.
        """
        try:
            # Save case repository
            if self.case_repository:
                await self.case_repository.save()

            # Shutdown base bridge
            await super().shutdown()

            self.logger.info("Enhanced bridge shut down successfully")
            return True

        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")
            return False

    def _load_enhanced_config(
        self,
        config_path: Optional[str]
    ) -> Dict[str, Any]:
        """Load enhanced configuration."""
        default_config = {
            'reflection': {
                'enabled': True,
                'iterations': 3,
                'num_samples': 3,
                'temperature': 0.3
            },
            'quality_enhancement': {
                'strategies': ['reflection', 'validation', 'cases', 'consistency'],
                'min_quality_threshold': 0.7,
                'auto_refine': True
            },
            'case_repository': {
                'storage_path': 'data/oneke_cases.json',
                'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
                'min_similarity': 0.7,
                'auto_save': True,
                'save_interval': 100
            },
            'learning': {
                'enabled': True,
                'feedback_required': False,
                'learning_rate': 0.1,
                'case_limit': 10000
            }
        }

        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)

                # Merge configs
                for key in user_config:
                    if key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]

            except Exception as e:
                self.logger.warning(
                    f"Failed to load enhanced config from {config_path}: {e}"
                )

        return default_config

    async def _store_high_quality_case(
        self,
        text: str,
        extraction: dict,
        schema: str,
        domain: str,
        quality_score: float,
        metadata: dict
    ):
        """Store high-quality extraction as a case."""
        if not self.case_repository:
            return

        try:
            case = Case.create(
                input_text=text,
                extracted_data=extraction,
                schema=schema,
                domain=domain,
                quality_score=quality_score,
                metadata=metadata
            )

            await self.case_repository.add_case(case)

            self.logger.debug(
                f"Stored high-quality case {case.case_id} "
                f"(quality: {quality_score:.2f})"
            )

        except Exception as e:
            self.logger.error(f"Failed to store case: {e}")

    async def _process_feedback(
        self,
        text: str,
        extraction: dict,
        schema: str,
        domain: str,
        feedback: Dict[str, Any],
        base_quality: float
    ):
        """Process human feedback and update case repository."""
        if not self.case_repository:
            return

        try:
            # Compute adjusted quality based on feedback
            correctness = feedback.get('correctness', base_quality)
            completeness = feedback.get('completeness', base_quality)
            adjusted_quality = (correctness + completeness) / 2

            # Create or update case
            case = Case.create(
                input_text=text,
                extracted_data=extraction,
                schema=schema,
                domain=domain,
                quality_score=adjusted_quality,
                metadata={
                    'feedback': feedback,
                    'base_quality': base_quality,
                    'adjusted_quality': adjusted_quality,
                    'has_human_feedback': True
                }
            )

            await self.case_repository.add_case(case)

            self.logger.info(
                f"Processed feedback for case {case.case_id}. "
                f"Quality adjusted: {base_quality:.2f} -> {adjusted_quality:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Failed to process feedback: {e}")


# Convenience functions

async def create_enhanced_oneke_bridge(
    config_path: Optional[str] = None,
    enhanced_config_path: Optional[str] = None
) -> EnhancedOneKEBridge:
    """
    Create and initialize enhanced OneKE bridge.

    Args:
        config_path: Path to OneKE config.yaml
        enhanced_config_path: Path to enhanced config.yaml

    Returns:
        Initialized EnhancedOneKEBridge
    """
    bridge = EnhancedOneKEBridge(
        config_path=config_path,
        enhanced_config_path=enhanced_config_path
    )

    await bridge.initialize()
    return bridge


async def extract_with_quality(
    text: str,
    schema: str,
    domain: str = "general",
    enable_enhancement: bool = True
) -> EnhancedResult:
    """
    Extract knowledge with quality enhancement.

    Convenience function for quick extraction.

    Args:
        text: Input text
        schema: Target schema
        domain: Domain label
        enable_enhancement: Enable quality enhancement

    Returns:
        EnhancedResult with extraction and quality scores
    """
    bridge = await create_enhanced_oneke_bridge()

    try:
        result = await bridge.extract_with_enhancement(
            text=text,
            schema=schema,
            domain=domain,
            enable_reflection=enable_enhancement,
            enable_cases=enable_enhancement,
            enable_validation=enable_enhancement,
            enable_consistency=enable_enhancement
        )

        return result

    finally:
        await bridge.shutdown()
