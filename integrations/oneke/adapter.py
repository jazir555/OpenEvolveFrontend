"""
OneKE Adapter for OpenEvolve

This adapter provides schema-guided information extraction capabilities using
the OneKE framework. It implements the ExtractionInterface for NER, RE, EE,
and Triple extraction with custom schema support.

OneKE Repository: https://github.com/zjunlp/OneKE

Integration Strategy:
- Zero modifications to OneKE source
- Adapter pattern for decoupled integration
- Support for Docker or Conda environments
- Multi-agent extraction workflows
"""

import asyncio
import json
import os
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

from ..base.extraction_interface import (
    ExtractionInterface,
    ExtractionResult,
    SchemaDefinition,
    ExtractionType,
    ConfigurationError,
    ConnectionError,
    ValidationError,
    SchemaLoadError,
    ShutdownError,
    ExtractionError
)


class OneKEAdapter(ExtractionInterface):
    """
    Adapter for OneKE schema-guided information extraction.

    This adapter provides a bridge between OpenEvolve and OneKE, supporting:
    - Named Entity Recognition (NER)
    - Relation Extraction (RE)
    - Event Extraction (EE)
    - Triple Extraction
    - Custom schema-guided extraction
    - Multi-agent workflows

    Attributes:
        config: Configuration dictionary
        oneke_path: Path to OneKE installation
        schemas: Loaded schema definitions
        is_initialized: Whether adapter has been initialized
        is_docker: Whether using Docker environment
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the OneKE adapter.

        Args:
            config_path: Optional path to config.yaml
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self.oneke_path: Optional[str] = None
        self.schemas: Dict[str, SchemaDefinition] = {}
        self.is_initialized = False
        self.is_docker = self.config.get('connection', {}).get('docker', False)
        self._process: Optional[subprocess.Popen] = None

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'project': {
                'name': 'OneKE',
                'version': '0.1.0',
                'enabled': True
            },
            'connection': {
                'model_category': 'ChatGPT',
                'model_name_or_path': 'gpt-4o-mini',
                'api_key': os.getenv('OPENAI_API_KEY'),
                'docker': False,
                'conda_env': 'oneke'
            },
            'features': {
                'ner': True,
                're': True,
                'ee': True,
                'triple': True,
                'multi_agent': True
            },
            'schemas': [
                'physics_concepts',
                'chemical_entities',
                'relations'
            ],
            'integration': {
                'auto_start': True,
                'cache_enabled': True,
                'cache_ttl': 3600,
                'fallback_on_error': True
            },
            'performance': {
                'max_workers': 4,
                'timeout': 30,
                'batch_size': 100
            }
        }

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the OneKE adapter.

        Args:
            config: Optional configuration override

        Returns:
            True if initialization successful

        Raises:
            ConfigurationError: If config is invalid
            ConnectionError: If OneKE connection fails
        """
        if config:
            self.config.update(config)

        try:
            # Validate configuration
            self._validate_config()

            # Locate OneKE installation
            self.oneke_path = self._locate_oneke()
            if not self.oneke_path:
                raise ConfigurationError(
                    "OneKE not found. Please install OneKE or set ONEKE_PATH environment variable."
                )

            # Load default schemas
            schema_dir = Path(__file__).parent / 'schemas'
            if schema_dir.exists():
                for schema_file in schema_dir.glob('*.yaml'):
                    schema_name = schema_file.stem
                    try:
                        self.schemas[schema_name] = self.load_schema(str(schema_file))
                    except Exception as e:
                        logger.warning(f"Failed to load schema {schema_name}: {e}")

            # Start OneKE server if configured
            if self.config['integration'].get('auto_start', True):
                await self._start_oneke_server()

            self.is_initialized = True
            return True

        except Exception as e:
            raise ConnectionError(f"Failed to initialize OneKE: {e}")

    def _validate_config(self) -> None:
        """Validate configuration."""
        connection = self.config.get('connection', {})
        if not connection.get('api_key') and connection.get('model_category') == 'ChatGPT':
            raise ConfigurationError(
                "OPENAI_API_KEY not set. Please set it in config or environment."
            )

        performance = self.config.get('performance', {})
        max_workers = performance.get('max_workers', 4)
        if max_workers < 1:
            raise ConfigurationError("max_workers must be at least 1")

    def _locate_oneke(self) -> Optional[str]:
        """
        Locate OneKE installation.

        Returns:
            Path to OneKE installation or None
        """
        # Check environment variable
        env_path = os.getenv('ONEKE_PATH')
        if env_path and Path(env_path).exists():
            return env_path

        # Check common locations
        possible_paths = [
            Path.cwd() / 'OneKE',
            Path.cwd().parent / 'OneKE',
            Path.home() / 'OneKE',
            Path('/opt/OneKE'),
        ]

        for path in possible_paths:
            if path.exists() and (path / 'main.py').exists():
                return str(path)

        return None

    async def _start_oneke_server(self) -> None:
        """Start OneKE server (if using server mode)."""
        if self.is_docker:
            # Docker mode - OneKE runs in container
            # Would use Docker SDK here
            pass
        else:
            # Conda/local mode - OneKE runs via subprocess
            oneke_main = Path(self.oneke_path) / 'main.py'
            if oneke_main.exists():
                # Start OneKE in background
                # This is a placeholder - actual implementation depends on OneKE API
                pass

    async def extract_ner(
        self,
        text: str,
        schema: Optional[SchemaDefinition] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Perform Named Entity Recognition extraction.

        Args:
            text: Input text
            schema: Optional schema for entity types
            **kwargs: Additional parameters

        Returns:
            ExtractionResult with entities
        """
        if not self.is_initialized:
            raise ExtractionError("Adapter not initialized. Call initialize() first.")

        if not self.config['features'].get('ner', True):
            raise ExtractionError("NER feature is not enabled")

        try:
            # Prepare extraction request
            extraction_config = {
                'task': 'NER',
                'text': text,
                'model_category': self.config['connection']['model_category'],
                'model_name_or_path': self.config['connection']['model_name_or_path'],
            }

            # Add schema if provided
            if schema:
                extraction_config['schema'] = self._schema_to_dict(schema)

            # Call OneKE (placeholder - actual API depends on OneKE)
            result = await self._call_oneke(extraction_config)

            return ExtractionResult(
                extraction_type=ExtractionType.NER,
                entities=result.get('entities', []),
                relations=[],
                events=[],
                triples=[],
                schema=schema.__dict__ if schema else {},
                confidence=result.get('confidence', 0.8),
                metadata=result.get('metadata', {}),
                raw_response=result
            )

        except Exception as e:
            if self.config['integration'].get('fallback_on_error', True):
                return self._fallback_ner(text, schema)
            raise ExtractionError(f"NER extraction failed: {e}")

    async def extract_re(
        self,
        text: str,
        schema: Optional[SchemaDefinition] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Perform Relation Extraction.

        Args:
            text: Input text
            schema: Optional schema for relation types
            **kwargs: Additional parameters

        Returns:
            ExtractionResult with relations
        """
        if not self.is_initialized:
            raise ExtractionError("Adapter not initialized. Call initialize() first.")

        if not self.config['features'].get('re', True):
            raise ExtractionError("RE feature is not enabled")

        try:
            extraction_config = {
                'task': 'RE',
                'text': text,
                'model_category': self.config['connection']['model_category'],
                'model_name_or_path': self.config['connection']['model_name_or_path'],
            }

            if schema:
                extraction_config['schema'] = self._schema_to_dict(schema)

            result = await self._call_oneke(extraction_config)

            return ExtractionResult(
                extraction_type=ExtractionType.RE,
                entities=[],
                relations=result.get('relations', []),
                events=[],
                triples=[],
                schema=schema.__dict__ if schema else {},
                confidence=result.get('confidence', 0.8),
                metadata=result.get('metadata', {}),
                raw_response=result
            )

        except Exception as e:
            if self.config['integration'].get('fallback_on_error', True):
                return self._fallback_re(text, schema)
            raise ExtractionError(f"RE extraction failed: {e}")

    async def extract_ee(
        self,
        text: str,
        schema: Optional[SchemaDefinition] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Perform Event Extraction.

        Args:
            text: Input text
            schema: Optional schema for event types
            **kwargs: Additional parameters

        Returns:
            ExtractionResult with events
        """
        if not self.is_initialized:
            raise ExtractionError("Adapter not initialized. Call initialize() first.")

        if not self.config['features'].get('ee', True):
            raise ExtractionError("EE feature is not enabled")

        try:
            extraction_config = {
                'task': 'EE',
                'text': text,
                'model_category': self.config['connection']['model_category'],
                'model_name_or_path': self.config['connection']['model_name_or_path'],
            }

            if schema:
                extraction_config['schema'] = self._schema_to_dict(schema)

            result = await self._call_oneke(extraction_config)

            return ExtractionResult(
                extraction_type=ExtractionType.EE,
                entities=[],
                relations=[],
                events=result.get('events', []),
                triples=[],
                schema=schema.__dict__ if schema else {},
                confidence=result.get('confidence', 0.8),
                metadata=result.get('metadata', {}),
                raw_response=result
            )

        except Exception as e:
            if self.config['integration'].get('fallback_on_error', True):
                return self._fallback_ee(text, schema)
            raise ExtractionError(f"EE extraction failed: {e}")

    async def extract_triple(
        self,
        text: str,
        schema: Optional[SchemaDefinition] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Perform Triple Extraction (subject-relation-object).

        Args:
            text: Input text
            schema: Optional schema for triple types
            **kwargs: Additional parameters

        Returns:
            ExtractionResult with triples
        """
        if not self.is_initialized:
            raise ExtractionError("Adapter not initialized. Call initialize() first.")

        if not self.config['features'].get('triple', True):
            raise ExtractionError("Triple feature is not enabled")

        try:
            extraction_config = {
                'task': 'Triple',
                'text': text,
                'model_category': self.config['connection']['model_category'],
                'model_name_or_path': self.config['connection']['model_name_or_path'],
            }

            if schema:
                extraction_config['schema'] = self._schema_to_dict(schema)

            result = await self._call_oneke(extraction_config)

            return ExtractionResult(
                extraction_type=ExtractionType.TRIPLE,
                entities=[],
                relations=[],
                events=[],
                triples=result.get('triples', []),
                schema=schema.__dict__ if schema else {},
                confidence=result.get('confidence', 0.8),
                metadata=result.get('metadata', {}),
                raw_response=result
            )

        except Exception as e:
            if self.config['integration'].get('fallback_on_error', True):
                return self._fallback_triple(text, schema)
            raise ExtractionError(f"Triple extraction failed: {e}")

    async def extract_schema_guided(
        self,
        text: str,
        schema: SchemaDefinition,
        **kwargs
    ) -> ExtractionResult:
        """
        Perform schema-guided extraction (most flexible method).

        Args:
            text: Input text
            schema: Schema definition
            **kwargs: Additional parameters

        Returns:
            ExtractionResult with all extracted info
        """
        if not self.is_initialized:
            raise ExtractionError("Adapter not initialized. Call initialize() first.")

        try:
            extraction_config = {
                'task': 'Schema',
                'text': text,
                'schema': self._schema_to_dict(schema),
                'model_category': self.config['connection']['model_category'],
                'model_name_or_path': self.config['connection']['model_name_or_path'],
                'multi_agent': self.config['features'].get('multi_agent', True),
            }

            result = await self._call_oneke(extraction_config)

            return ExtractionResult(
                extraction_type=ExtractionType.SCHEMA,
                entities=result.get('entities', []),
                relations=result.get('relations', []),
                events=result.get('events', []),
                triples=result.get('triples', []),
                schema=schema.__dict__,
                confidence=result.get('confidence', 0.8),
                metadata=result.get('metadata', {}),
                raw_response=result
            )

        except Exception as e:
            raise ExtractionError(f"Schema-guided extraction failed: {e}")

    async def batch_extract(
        self,
        texts: List[str],
        extraction_type: ExtractionType,
        schema: Optional[SchemaDefinition] = None,
        **kwargs
    ) -> List[ExtractionResult]:
        """
        Perform batch extraction.

        Args:
            texts: List of input texts
            extraction_type: Type of extraction
            schema: Optional schema
            **kwargs: Additional parameters

        Returns:
            List of ExtractionResult objects
        """
        if not self.is_initialized:
            raise ExtractionError("Adapter not initialized. Call initialize() first.")

        # Create tasks for parallel processing
        max_workers = self.config['performance'].get('max_workers', 4)

        if extraction_type == ExtractionType.NER:
            tasks = [self.extract_ner(text, schema, **kwargs) for text in texts]
        elif extraction_type == ExtractionType.RE:
            tasks = [self.extract_re(text, schema, **kwargs) for text in texts]
        elif extraction_type == ExtractionType.EE:
            tasks = [self.extract_ee(text, schema, **kwargs) for text in texts]
        elif extraction_type == ExtractionType.TRIPLE:
            tasks = [self.extract_triple(text, schema, **kwargs) for text in texts]
        elif extraction_type == ExtractionType.SCHEMA:
            if not schema:
                raise ValueError("Schema is required for schema-guided extraction")
            tasks = [self.extract_schema_guided(text, schema, **kwargs) for text in texts]
        else:
            raise ValueError(f"Unsupported extraction type: {extraction_type}")

        # Process in batches to respect max_workers
        results = []
        for i in range(0, len(tasks), max_workers):
            batch = tasks[i:i + max_workers]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(ExtractionResult(
                        extraction_type=extraction_type,
                        entities=[], relations=[], events=[], triples=[],
                        schema=schema.__dict__ if schema else {},
                        confidence=0.0, metadata={'error': str(result)}
                    ))
                else:
                    results.append(result)

        return results

    async def validate(self) -> Dict[str, Any]:
        """
        Validate OneKE adapter configuration and connection.

        Returns:
            Validation results
        """
        checks = []
        issues = []

        # Check configuration
        try:
            self._validate_config()
            checks.append({'name': 'configuration', 'status': 'passed'})
        except Exception as e:
            checks.append({'name': 'configuration', 'status': 'failed', 'error': str(e)})
            issues.append(f"Configuration error: {e}")

        # Check OneKE installation
        if self.oneke_path:
            checks.append({'name': 'oneke_path', 'status': 'passed', 'path': self.oneke_path})
        else:
            checks.append({'name': 'oneke_path', 'status': 'failed'})
            issues.append("OneKE installation not found")

        # Check schemas
        if self.schemas:
            checks.append({
                'name': 'schemas',
                'status': 'passed',
                'count': len(self.schemas)
            })
        else:
            checks.append({'name': 'schemas', 'status': 'warning', 'count': 0})
            issues.append("No schemas loaded")

        is_valid = all(c.get('status') == 'passed' for c in checks)

        return {
            'is_valid': is_valid,
            'checks': checks,
            'issues': issues,
            'performance': {
                'max_workers': self.config['performance'].get('max_workers', 4),
                'timeout': self.config['performance'].get('timeout', 30),
                'batch_size': self.config['performance'].get('batch_size', 100)
            }
        }

    async def shutdown(self) -> bool:
        """
        Shutdown OneKE adapter.

        Returns:
            True if successful
        """
        try:
            # Stop OneKE server if running
            if self._process:
                self._process.terminate()
                self._process.wait(timeout=5)
                self._process = None

            self.is_initialized = False
            return True

        except Exception as e:
            raise ShutdownError(f"Shutdown failed: {e}")

    def load_schema(self, schema_path: str) -> SchemaDefinition:
        """
        Load schema from YAML file.

        Args:
            schema_path: Path to schema YAML

        Returns:
            SchemaDefinition object
        """
        try:
            with open(schema_path, 'r') as f:
                schema_data = yaml.safe_load(f)

            return SchemaDefinition(
                name=schema_data['name'],
                description=schema_data.get('description', ''),
                entity_types=schema_data.get('entity_types', []),
                relation_types=schema_data.get('relation_types', []),
                event_types=schema_data.get('event_types', []),
                constraints=schema_data.get('constraints'),
                examples=schema_data.get('examples')
            )

        except Exception as e:
            raise SchemaLoadError(f"Failed to load schema from {schema_path}: {e}")

    async def extract_from_workflow(
        self,
        workflow_data: Dict[str, Any],
        schemas: List[SchemaDefinition],
        **kwargs
    ) -> Dict[str, ExtractionResult]:
        """
        Extract knowledge from workflow execution data.

        Args:
            workflow_data: Workflow state dictionary
            schemas: List of schemas to apply
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping schema names to results
        """
        # Convert workflow to text
        workflow_text = self._workflow_to_text(workflow_data)

        results = {}
        for schema in schemas:
            try:
                result = await self.extract_schema_guided(workflow_text, schema, **kwargs)
                results[schema.name] = result
            except Exception as e:
                logger.error(f"Failed to apply schema {schema.name}: {e}")
                results[schema.name] = ExtractionResult(
                    extraction_type=ExtractionType.SCHEMA,
                    entities=[], relations=[], events=[], triples=[],
                    schema=schema.__dict__,
                    confidence=0.0,
                    metadata={'error': str(e)}
                )

        return results

    # ========== Helper Methods ==========

    async def _call_oneke(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call OneKE extraction API.

        Actually calls OneKE library if available, otherwise uses
        LLM-based extraction with schema guidance.

        Args:
            config: Extraction configuration

        Returns:
            Extraction results
        """
        # Try to use actual OneKE library
        oneke_result = await self._call_actual_oneke(config)
        if oneke_result is not None:
            return oneke_result
        
        # Fallback: Use LLM-based extraction with schema guidance
        return await self._call_llm_extraction(config)
    
    async def _call_actual_oneke(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Try to call actual OneKE library.
        
        Returns:
            Extraction results or None if OneKE not available
        """
        try:
            # Try to import OneKE wrapper
            import sys
            from pathlib import Path
            
            # Add OneKE to path if it exists locally
            oneke_path = Path(__file__).parent.parent.parent / "OneKE"
            if oneke_path.exists():
                sys.path.insert(0, str(oneke_path))
            
            # Try importing OneKE
            try:
                from oneke import OneKE
            except ImportError:
                return None  # OneKE not installed
            
            # Get API key
            api_key = self.config['connection'].get('api_key')
            model_name = self.config['connection'].get('model_name_or_path', 'gpt-4o-mini')
            
            # Initialize OneKE
            if api_key:
                oneke = OneKE(
                    model_name_or_path=model_name,
                    api_key=api_key,
                    model_category="ChatGPT"
                )
            else:
                # Try local model
                oneke = OneKE(
                    model_name_or_path="zjunlp/oneke",
                    model_category="Local"
                )
            
            # Call extraction
            text = config.get('text', '')
            schema = config.get('schema', {})
            task = config.get('task', 'NER')
            
            result = oneke.extract(
                text=text,
                schema=schema,
                task=task
            )
            
            # Log actual OneKE usage
            logger.info(f"OneKE actual call successful: {len(result.get('entities', []))} entities")
            
            return {
                'entities': result.get('entities', []),
                'relations': result.get('relations', []),
                'events': result.get('events', []),
                'triples': result.get('triples', []),
                'confidence': result.get('confidence', 0.85),
                'metadata': {
                    'model': model_name,
                    'source': 'oneke_actual',
                    'task': task
                }
            }
            
        except Exception as e:
            logger.debug(f"Actual OneKE call failed: {e}")
            return None
    
    async def _call_llm_extraction(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback: Use LLM-based extraction with schema guidance.
        
        This provides real extraction using OpenAI API when OneKE library
        is not available.
        """
        import os
        import openai
        
        text = config.get('text', '')
        schema = config.get('schema', {})
        task = config.get('task', 'NER')
        
        api_key = self.config['connection'].get('api_key') or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            logger.warning("No API key available for LLM extraction")
            return self._create_fallback_response(config)
        
        try:
            client = openai.OpenAI(api_key=api_key)
            
            # Build prompt based on task and schema
            prompt = self._build_extraction_prompt(text, schema, task)
            
            response = client.chat.completions.create(
                model=self.config['connection'].get('model_name_or_path', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": "You are a knowledge extraction system. Extract structured information according to the schema provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            logger.info(f"LLM extraction successful: {len(result.get('entities', []))} entities")
            
            return {
                'entities': result.get('entities', []),
                'relations': result.get('relations', []),
                'events': result.get('events', []),
                'triples': result.get('triples', []),
                'confidence': result.get('confidence', 0.8),
                'metadata': {
                    'model': self.config['connection'].get('model_name_or_path', 'gpt-4o-mini'),
                    'source': 'llm_extraction',
                    'task': task
                }
            }
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._create_fallback_response(config)
    
    def _build_extraction_prompt(self, text: str, schema: Dict[str, Any], task: str) -> str:
        """Build extraction prompt based on schema and task."""
        entity_types = schema.get('entity_types', [])
        relation_types = schema.get('relation_types', [])
        
        prompt = f"""Extract information from the following text according to the schema.

Text: {text}

Task: {task}

Entity Types: {', '.join([et.get('name', str(et)) for et in entity_types]) if entity_types else 'Any'}
Relation Types: {', '.join([rt.get('name', str(rt)) for rt in relation_types]) if relation_types else 'Any'}

Return a JSON object with the following structure:
{{
    "entities": [
        {{
            "text": "extracted entity text",
            "type": "entity type",
            "start": 0,
            "end": 10
        }}
    ],
    "relations": [
        {{
            "head": "head entity text",
            "tail": "tail entity text",
            "type": "relation type"
        }}
    ],
    "confidence": 0.85
}}
"""
        return prompt
    
    def _create_fallback_response(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a minimal fallback response."""
        return {
            'entities': [],
            'relations': [],
            'events': [],
            'triples': [],
            'confidence': 0.0,
            'metadata': {
                'source': 'fallback',
                'error': 'OneKE not available and LLM extraction failed'
            }
        }

    def _schema_to_dict(self, schema: SchemaDefinition) -> Dict[str, Any]:
        """Convert SchemaDefinition to dictionary."""
        return {
            'name': schema.name,
            'description': schema.description,
            'entity_types': schema.entity_types,
            'relation_types': schema.relation_types,
            'event_types': schema.event_types,
            'constraints': schema.constraints,
            'examples': schema.examples
        }

    def _workflow_to_text(self, workflow_data: Dict[str, Any]) -> str:
        """Convert workflow data to text for extraction."""
        parts = []

        if 'problem_statement' in workflow_data:
            parts.append(f"Problem: {workflow_data['problem_statement']}")

        if 'final_solution' in workflow_data:
            parts.append(f"Solution: {workflow_data['final_solution']}")

        if 'decomposition_plan' in workflow_data:
            parts.append(f"Decomposition: {workflow_data['decomposition_plan']}")

        return '\n\n'.join(parts)

    def _fallback_ner(self, text: str, schema: Optional[SchemaDefinition]) -> ExtractionResult:
        """Fallback NER extraction using simple patterns."""
        import re

        entities = []
        if schema:
            for entity_type in schema.entity_types:
                pattern = entity_type.get('pattern', r'\b[A-Z][a-z]+\b')
                for match in re.finditer(pattern, text):
                    entities.append({
                        'text': match.group(),
                        'type': entity_type['name'],
                        'start': match.start(),
                        'end': match.end()
                    })

        return ExtractionResult(
            extraction_type=ExtractionType.NER,
            entities=entities,
            relations=[], events=[], triples=[],
            schema=schema.__dict__ if schema else {},
            confidence=0.5,
            metadata={'fallback': True}
        )

    def _fallback_re(self, text: str, schema: Optional[SchemaDefinition]) -> ExtractionResult:
        """Fallback relation extraction."""
        return ExtractionResult(
            extraction_type=ExtractionType.RE,
            entities=[], relations=[], events=[], triples=[],
            schema=schema.__dict__ if schema else {},
            confidence=0.0,
            metadata={'fallback': True, 'note': 'No fallback implemented for RE'}
        )

    def _fallback_ee(self, text: str, schema: Optional[SchemaDefinition]) -> ExtractionResult:
        """Fallback event extraction."""
        return ExtractionResult(
            extraction_type=ExtractionType.EE,
            entities=[], relations=[], events=[], triples=[],
            schema=schema.__dict__ if schema else {},
            confidence=0.0,
            metadata={'fallback': True, 'note': 'No fallback implemented for EE'}
        )

    def _fallback_triple(self, text: str, schema: Optional[SchemaDefinition]) -> ExtractionResult:
        """Fallback triple extraction."""
        return ExtractionResult(
            extraction_type=ExtractionType.TRIPLE,
            entities=[], relations=[], events=[], triples=[],
            schema=schema.__dict__ if schema else {},
            confidence=0.0,
            metadata={'fallback': True, 'note': 'No fallback implemented for Triple'}
        )
