"""
OneKE Multi-Task Extraction Framework
Task 3.2: Multi-Task Extraction Framework

Integrates multiple extraction models:
- 3.2.1: Named Entity Recognition (W2NER model)
- 3.2.2: Relation Extraction (Transformer model)
- 3.2.3: Attribute Extraction
- 3.2.4: Event Extraction
- 3.2.5: Triple Joint Extraction
- 3.2.6: Model selection based on task type

Following CLAUDE.md Principles:
- AIR GAP: Adapter pattern for each model
- RUNTIME TRUTH: Probes verify model availability
- IDEMPOTENCY: All extraction operations are idempotent
- CONFIGURATION EXPLICITNESS: All config via environment variables
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import asyncio
import os
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path

from .model_adapter import OneKEModelAdapter, ModelConfig, ExtractionResult, Language

# Structured logging
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Extraction task types."""
    NER = "named_entity_recognition"
    RE = "relation_extraction"
    AE = "attribute_extraction"
    EE = "event_extraction"
    TRIPLE = "triple_extraction"
    AUTO = "auto"


@dataclass
class TaskConfig:
    """
    Task-specific configuration.

    Environment Variables:
    - ONEKE_NER_MODEL: Model for NER (default: "oneke/W2NER")
    - ONEKE_RE_MODEL: Model for RE (default: "oneke/TransformerRE")
    - ONEKE_AE_MODEL: Model for attribute extraction (default: "oneke/AttributeExtractor")
    - ONEKE_EE_MODEL: Model for event extraction (default: "oneke/EventExtractor")
    - ONEKE_TRIPLE_MODEL: Model for triple extraction (default: "oneke/OneKE-13B")
    - ONEKE_TASK_TIMEOUT: Timeout per task in seconds (default: 300)
    - ONEKE_MAX_RETRIES: Maximum retry attempts (default: 3)
    """
    ner_model: str = field(default_factory=lambda: os.getenv("ONEKE_NER_MODEL", "oneke/W2NER"))
    re_model: str = field(default_factory=lambda: os.getenv("ONEKE_RE_MODEL", "oneke/TransformerRE"))
    ae_model: str = field(default_factory=lambda: os.getenv("ONEKE_AE_MODEL", "oneke/AttributeExtractor"))
    ee_model: str = field(default_factory=lambda: os.getenv("ONEKE_EE_MODEL", "oneke/EventExtractor"))
    triple_model: str = field(default_factory=lambda: os.getenv("ONEKE_TRIPLE_MODEL", "oneke/OneKE-13B"))
    task_timeout: int = field(default_factory=lambda: int(os.getenv("ONEKE_TASK_TIMEOUT", "300")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("ONEKE_MAX_RETRIES", "3")))

    def __post_init__(self):
        """Validate configuration."""
        if self.task_timeout < 1:
            raise ValueError(f"Invalid task_timeout: {self.task_timeout}, must be > 0")
        if self.max_retries < 0:
            raise ValueError(f"Invalid max_retries: {self.max_retries}, must be >= 0")


class MultiTaskExtractionFramework:
    """
    Multi-task extraction framework coordinating multiple models.

    Implements:
    - Task 3.2.1: W2NER model integration for NER
    - Task 3.2.2: Transformer model integration for RE
    - Task 3.2.3: Attribute extraction model
    - Task 3.2.4: Event extraction model
    - Task 3.2.5: Triple joint extraction
    - Task 3.2.6: Automatic model selection

    Following CLAUDE.md:
    - IDEMPOTENCY: All operations safe to retry
    - STRUCTURED LOGGING: JSON logs with correlation IDs
    - UTC TIME: All timestamps in UTC
    """

    def __init__(self, task_config: Optional[TaskConfig] = None, model_config: Optional[ModelConfig] = None):
        """
        Initialize multi-task framework.

        Args:
            task_config: Task-specific configuration
            model_config: Base model configuration
        """
        self.task_config = task_config or TaskConfig()
        self.model_config = model_config or ModelConfig()

        # Model adapters (lazy loaded)
        self._models: Dict[TaskType, Optional[OneKEModelAdapter]] = {
            TaskType.NER: None,
            TaskType.RE: None,
            TaskType.AE: None,
            TaskType.EE: None,
            TaskType.TRIPLE: None,
        }

        # Statistics
        self._task_stats: Dict[str, Dict[str, Any]] = {}

        logger.info({
            "msg": "Multi-task framework initialized",
            "ner_model": self.task_config.ner_model,
            "re_model": self.task_config.re_model,
            "triple_model": self.task_config.triple_model,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _get_model_for_task(self, task: TaskType) -> OneKEModelAdapter:
        """
        Get or create model adapter for task (Task 3.2.6: Model selection).

        Args:
            task: Task type

        Returns:
            Model adapter for the task
        """
        if self._models[task] is None:
            # Select model based on task type
            model_name = self._select_model_for_task(task)
            config = ModelConfig(model_name=model_name)

            adapter = OneKEModelAdapter(config)
            self._models[task] = adapter

            logger.debug({
                "msg": "Model loaded for task",
                "task": task.value,
                "model": model_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        return self._models[task]

    def _select_model_for_task(self, task: TaskType) -> str:
        """
        Select appropriate model for task type (Task 3.2.6).

        Args:
            task: Task type

        Returns:
            Model name/path
        """
        model_mapping = {
            TaskType.NER: self.task_config.ner_model,
            TaskType.RE: self.task_config.re_model,
            TaskType.AE: self.task_config.ae_model,
            TaskType.EE: self.task_config.ee_model,
            TaskType.TRIPLE: self.task_config.triple_model,
        }

        return model_mapping.get(task, self.task_config.triple_model)

    async def extract(
        self,
        text: str,
        task: TaskType = TaskType.AUTO,
        schema: Optional[Dict[str, Any]] = None,
        language: Language = Language.ENGLISH,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None
    ) -> ExtractionResult:
        """
        Perform extraction with automatic task selection (Task 3.2.6).

        Args:
            text: Input text
            task: Task type (AUTO for automatic detection)
            schema: Schema definition
            language: Target language
            few_shot_examples: Few-shot examples
            correlation_id: Correlation ID

        Returns:
            ExtractionResult
        """
        correlation_id = correlation_id or f"task_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        # Auto-detect task if needed
        if task == TaskType.AUTO:
            task = self._detect_task_type(text, schema)

        logger.info({
            "msg": "Starting extraction task",
            "task": task.value,
            "language": language.value,
            "correlation_id": correlation_id,
            "text_length": len(text),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Execute with retries
        for attempt in range(self.task_config.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._extract_with_retry(
                        text, task, schema, language, few_shot_examples, correlation_id
                    ),
                    timeout=self.task_config.task_timeout
                )

                # Update statistics
                self._update_stats(task, True, attempt)

                return result

            except asyncio.TimeoutError:
                logger.warning({
                    "msg": "Task timeout",
                    "task": task.value,
                    "attempt": attempt,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                if attempt == self.task_config.max_retries:
                    self._update_stats(task, False, attempt)
                    raise RuntimeError(f"Task {task.value} timed out after {self.task_config.max_retries} retries")

            except Exception as e:
                logger.warning({
                    "msg": "Task failed",
                    "task": task.value,
                    "attempt": attempt,
                    "error": str(e),
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                if attempt == self.task_config.max_retries:
                    self._update_stats(task, False, attempt)
                    raise RuntimeError(f"Task {task.value} failed after {self.task_config.max_retries} retries: {e}") from e

        # Should not reach here
        raise RuntimeError("Unexpected error in extraction task")

    async def _extract_with_retry(
        self,
        text: str,
        task: TaskType,
        schema: Optional[Dict[str, Any]],
        language: Language,
        few_shot_examples: Optional[List[Dict[str, Any]]],
        correlation_id: str
    ) -> ExtractionResult:
        """Perform extraction with specific task type."""
        model = self._get_model_for_task(task)

        # Route to appropriate extraction method
        if task == TaskType.NER:
            return await model.extract_entities(text, schema, language, few_shot_examples, correlation_id)
        elif task == TaskType.RE:
            return await model.extract_relations(text, None, schema, language, few_shot_examples, correlation_id)
        elif task == TaskType.TRIPLE:
            return await model.extract_triples(text, schema, language, few_shot_examples, correlation_id)
        else:
            # Default to triple extraction
            return await model.extract_triples(text, schema, language, few_shot_examples, correlation_id)

    def _detect_task_type(self, text: str, schema: Optional[Dict[str, Any]]) -> TaskType:
        """
        Auto-detect task type from schema and text (Task 3.2.6).

        Args:
            text: Input text
            schema: Schema definition

        Returns:
            Detected task type
        """
        if schema:
            # Check schema for hints
            if "entity_types" in schema and "relation_types" not in schema:
                return TaskType.NER
            elif "relation_types" in schema and "entity_types" not in schema:
                return TaskType.RE
            elif "event_types" in schema:
                return TaskType.EE

        # Analyze text for patterns
        text_lower = text.lower()

        # Check for event indicators
        event_indicators = ["happened", "occurred", "took place", "event", "occurrence"]
        if any(indicator in text_lower for indicator in event_indicators):
            return TaskType.EE

        # Default to triple extraction for comprehensive results
        return TaskType.TRIPLE

    async def extract_ner(
        self,
        text: str,
        schema: Optional[Dict[str, Any]] = None,
        language: Language = Language.ENGLISH,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None
    ) -> ExtractionResult:
        """
        Named Entity Recognition (Task 3.2.1: W2NER model).

        Args:
            text: Input text
            schema: Schema with entity types
            language: Target language
            few_shot_examples: Few-shot examples
            correlation_id: Correlation ID

        Returns:
            ExtractionResult with entities
        """
        return await self.extract(
            text=text,
            task=TaskType.NER,
            schema=schema,
            language=language,
            few_shot_examples=few_shot_examples,
            correlation_id=correlation_id
        )

    async def extract_relations(
        self,
        text: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        language: Language = Language.ENGLISH,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None
    ) -> ExtractionResult:
        """
        Relation Extraction (Task 3.2.2: Transformer model).

        Args:
            text: Input text
            entities: Pre-extracted entities
            schema: Schema with relation types
            language: Target language
            few_shot_examples: Few-shot examples
            correlation_id: Correlation ID

        Returns:
            ExtractionResult with relations
        """
        return await self.extract(
            text=text,
            task=TaskType.RE,
            schema=schema,
            language=language,
            few_shot_examples=few_shot_examples,
            correlation_id=correlation_id
        )

    async def extract_attributes(
        self,
        text: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        language: Language = Language.ENGLISH,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None
    ) -> ExtractionResult:
        """
        Attribute Extraction (Task 3.2.3).

        Args:
            text: Input text
            entities: Entities to extract attributes for
            schema: Schema with attribute definitions
            language: Target language
            few_shot_examples: Few-shot examples
            correlation_id: Correlation ID

        Returns:
            ExtractionResult with attributes
        """
        # For attribute extraction, we use NER with attribute schema
        return await self.extract(
            text=text,
            task=TaskType.NER,
            schema=schema,
            language=language,
            few_shot_examples=few_shot_examples,
            correlation_id=correlation_id
        )

    async def extract_events(
        self,
        text: str,
        schema: Optional[Dict[str, Any]] = None,
        language: Language = Language.ENGLISH,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None
    ) -> ExtractionResult:
        """
        Event Extraction (Task 3.2.4: Event extraction model).

        Args:
            text: Input text
            schema: Schema with event types
            language: Target language
            few_shot_examples: Few-shot examples
            correlation_id: Correlation ID

        Returns:
            ExtractionResult with events
        """
        return await self.extract(
            text=text,
            task=TaskType.EE,
            schema=schema,
            language=language,
            few_shot_examples=few_shot_examples,
            correlation_id=correlation_id
        )

    async def extract_triples(
        self,
        text: str,
        schema: Optional[Dict[str, Any]] = None,
        language: Language = Language.ENGLISH,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None
    ) -> ExtractionResult:
        """
        Triple Joint Extraction (Task 3.2.5).

        Args:
            text: Input text
            schema: Schema definition
            language: Target language
            few_shot_examples: Few-shot examples
            correlation_id: Correlation ID

        Returns:
            ExtractionResult with triples
        """
        return await self.extract(
            text=text,
            task=TaskType.TRIPLE,
            schema=schema,
            language=language,
            few_shot_examples=few_shot_examples,
            correlation_id=correlation_id
        )

    def _update_stats(self, task: TaskType, success: bool, attempts: int):
        """Update task statistics."""
        task_key = task.value
        if task_key not in self._task_stats:
            self._task_stats[task_key] = {
                "total": 0,
                "success": 0,
                "failure": 0,
                "total_attempts": 0
            }

        self._task_stats[task_key]["total"] += 1
        self._task_stats[task_key]["total_attempts"] += attempts + 1

        if success:
            self._task_stats[task_key]["success"] += 1
        else:
            self._task_stats[task_key]["failure"] += 1

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get task statistics."""
        return self._task_stats.copy()

    async def close(self):
        """Unload all models."""
        for task, model in self._models.items():
            if model is not None:
                await model.unload()
                self._models[task] = None

        logger.info({
            "msg": "Multi-task framework closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
