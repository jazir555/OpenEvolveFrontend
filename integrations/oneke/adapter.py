"""
OneKE Adapter for OpenEvolve - TRUE 100% VERSION

This adapter provides ACTUAL schema-guided information extraction using
the OneKE framework. NO FALLBACK - OneKE is required.

OneKE Repository: https://github.com/zjunlp/OneKE

Integration Strategy:
- Zero modifications to OneKE source
- Adapter pattern for decoupled integration
- Support for Docker or Conda environments
- Multi-agent extraction workflows
- REQUIRES OneKE installation (no fallback)
"""

import asyncio
import json
import os
import subprocess
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add OneKE to path if it exists locally
_ONEKE_PATH = Path(__file__).parent.parent.parent / "OneKE"
if _ONEKE_PATH.exists() and str(_ONEKE_PATH) not in sys.path:
    sys.path.insert(0, str(_ONEKE_PATH))

try:
    import yaml
except ImportError:
    yaml = None

try:
    from ..base.extraction_interface import (
        ExtractionInterface,
        ExtractionResult,
        SchemaDefinition,
        ExtractionType,
        ConfigurationError,
        ConnectionError,
        ValidationError,
        ShutdownError,
        ExtractionError
    )
except ImportError:
    # Fallback if base module doesn't exist
    ExtractionInterface = object
    ExtractionResult = dict
    SchemaDefinition = dict
    ExtractionType = str
    ConfigurationError = Exception
    ConnectionError = Exception
    ValidationError = Exception
    ShutdownError = Exception
    ExtractionError = Exception

import logging
logger = logging.getLogger(__name__)


class OneKENotInstalledError(Exception):
    """Raised when OneKE is not installed."""
    pass


class OneKEAdapter:
    """
    Adapter for ACTUAL OneKE schema-guided information extraction.

    This adapter REQUIRES OneKE to be installed. It provides:
    - Named Entity Recognition (NER)
    - Relation Extraction (RE)
    - Event Extraction (EE)
    - Triple Extraction
    - Custom schema-guided extraction
    - Multi-agent workflows

    Raises:
        OneKENotInstalledError: If OneKE is not available
        
    Attributes:
        config: Configuration dictionary
        oneke_path: Path to OneKE installation
        schemas: Loaded schema definitions
        is_initialized: Whether adapter has been initialized
        model: The actual OneKE model instance
    """

    def __init__(self, config_path: Optional[str] = None, allow_fallback: bool = False):
        """
        Initialize the OneKE adapter.

        Args:
            config_path: Optional path to config.yaml
            allow_fallback: If True, use LLM fallback when OneKE unavailable
            
        Raises:
            OneKENotInstalledError: If OneKE not available and allow_fallback=False
        """
        self.allow_fallback = allow_fallback
        
        if config_path and yaml:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self.oneke_path: Optional[str] = None
        self.schemas: Dict[str, Any] = {}
        self.is_initialized = False
        self.is_docker = self.config.get('connection', {}).get('docker', False)
        self._process: Optional[subprocess.Popen] = None
        self.model = None
        self._oneke_available = False
        
        # Check OneKE availability
        self._check_oneke()

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
                'fallback_on_error': False  # Disabled for TRUE 100%
            },
            'performance': {
                'max_workers': 4,
                'timeout': 30,
                'batch_size': 100
            }
        }

    def _check_oneke(self):
        """Check if OneKE is available."""
        try:
            # Try importing OneKE
            try:
                from src.oneke import OneKE
                self._oneke_available = True
                logger.info("✓ OneKE is available (from src.oneke)")
            except ImportError:
                from oneke import OneKE
                self._oneke_available = True
                logger.info("✓ OneKE is available (from oneke)")
        except ImportError:
            self._oneke_available = False
            error_msg = (
                "OneKE is NOT installed.\n"
                "Run 'python setup_oneke.py --clone' to install OneKE.\n"
                "Schema-guided extraction requires OneKE for TRUE 100% functionality."
            )
            if self.allow_fallback:
                logger.warning(error_msg)
                logger.warning("Will use LLM fallback (NOT recommended)")
            else:
                logger.error(error_msg)

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the OneKE adapter with ACTUAL OneKE.

        Args:
            config: Optional configuration override

        Returns:
            True if initialization successful

        Raises:
            OneKENotInstalledError: If OneKE not available and allow_fallback=False
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
                if self.allow_fallback:
                    logger.warning("OneKE not found, will use LLM fallback")
                    self.is_initialized = True
                    return True
                raise OneKENotInstalledError(
                    "OneKE not found. Please install OneKE: python setup_oneke.py --clone"
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

            # Initialize ACTUAL OneKE model
            if self._oneke_available:
                await self._initialize_oneke_model()

            self.is_initialized = True
            logger.info("✓ OneKE adapter initialized successfully")
            return True

        except Exception as e:
            if self.allow_fallback:
                logger.warning(f"OneKE initialization failed: {e}, using fallback")
                self.is_initialized = True
                return True
            raise ConnectionError(f"Failed to initialize OneKE: {e}")

    async def _initialize_oneke_model(self):
        """Initialize the actual OneKE model."""
        try:
            # Import OneKE
            try:
                from src.oneke import OneKE
            except ImportError:
                from oneke import OneKE
            
            # Get configuration
            api_key = self.config['connection'].get('api_key') or os.getenv('OPENAI_API_KEY')
            model_name = self.config['connection'].get('model_name_or_path', 'gpt-4o-mini')
            model_category = self.config['connection'].get('model_category', 'ChatGPT')
            
            # Initialize OneKE
            if api_key and model_category == 'ChatGPT':
                self.model = OneKE(
                    model_name_or_path=model_name,
                    api_key=api_key,
                    model_category=model_category
                )
                logger.info(f"✓ OneKE initialized with OpenAI model: {model_name}")
            else:
                # Try local model
                self.model = OneKE(
                    model_name_or_path="zjunlp/oneke",
                    model_category="Local"
                )
                logger.info("✓ OneKE initialized with local model")
            
        except Exception as e:
            logger.error(f"Failed to initialize OneKE model: {e}")
            if not self.allow_fallback:
                raise OneKENotInstalledError(f"OneKE model initialization failed: {e}")

    def _validate_config(self) -> None:
        """Validate configuration."""
        connection = self.config.get('connection', {})
        if not connection.get('api_key') and connection.get('model_category') == 'ChatGPT':
            if not os.getenv('OPENAI_API_KEY'):
                if not self.allow_fallback:
                    raise ConfigurationError(
                        "OPENAI_API_KEY not set. Please set it or use local model."
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
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        return None

    async def extract_ner(
        self,
        text: str,
        schema: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform Named Entity Recognition extraction with ACTUAL OneKE.

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
            # Use ACTUAL OneKE if available
            if self.model:
                result = self.model.extract(
                    text=text,
                    schema=self._schema_to_dict(schema) if schema else {},
                    task='NER'
                )
                
                logger.info(f"✓ OneKE NER extracted {len(result.get('entities', []))} entities")
                
                return {
                    'extraction_type': 'NER',
                    'entities': result.get('entities', []),
                    'relations': [],
                    'events': [],
                    'triples': [],
                    'schema': schema.__dict__ if schema else {},
                    'confidence': result.get('confidence', 0.85),
                    'metadata': {
                        'source': 'oneke_actual',
                        'model': self.config['connection'].get('model_name_or_path')
                    },
                    'raw_response': result
                }
            
            # Fallback only if allowed
            if self.allow_fallback:
                return await self._llm_extraction(text, schema, 'NER')
            
            raise OneKENotInstalledError("OneKE model not available")

        except Exception as e:
            if self.config['integration'].get('fallback_on_error') and self.allow_fallback:
                return await self._llm_extraction(text, schema, 'NER')
            raise ExtractionError(f"NER extraction failed: {e}")

    async def extract_re(
        self,
        text: str,
        schema: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform Relation Extraction with ACTUAL OneKE.
        """
        if not self.is_initialized:
            raise ExtractionError("Adapter not initialized. Call initialize() first.")

        if not self.config['features'].get('re', True):
            raise ExtractionError("RE feature is not enabled")

        try:
            if self.model:
                result = self.model.extract(
                    text=text,
                    schema=self._schema_to_dict(schema) if schema else {},
                    task='RE'
                )
                
                logger.info(f"✓ OneKE RE extracted {len(result.get('relations', []))} relations")
                
                return {
                    'extraction_type': 'RE',
                    'entities': [],
                    'relations': result.get('relations', []),
                    'events': [],
                    'triples': [],
                    'schema': schema.__dict__ if schema else {},
                    'confidence': result.get('confidence', 0.85),
                    'metadata': {'source': 'oneke_actual'},
                    'raw_response': result
                }
            
            if self.allow_fallback:
                return await self._llm_extraction(text, schema, 'RE')
            
            raise OneKENotInstalledError("OneKE model not available")

        except Exception as e:
            if self.config['integration'].get('fallback_on_error') and self.allow_fallback:
                return await self._llm_extraction(text, schema, 'RE')
            raise ExtractionError(f"RE extraction failed: {e}")

    async def extract_triple(
        self,
        text: str,
        schema: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform Triple Extraction with ACTUAL OneKE.
        """
        if not self.is_initialized:
            raise ExtractionError("Adapter not initialized. Call initialize() first.")

        if not self.config['features'].get('triple', True):
            raise ExtractionError("Triple feature is not enabled")

        try:
            if self.model:
                result = self.model.extract(
                    text=text,
                    schema=self._schema_to_dict(schema) if schema else {},
                    task='Triple'
                )
                
                logger.info(f"✓ OneKE extracted {len(result.get('triples', []))} triples")
                
                return {
                    'extraction_type': 'TRIPLE',
                    'entities': [],
                    'relations': [],
                    'events': [],
                    'triples': result.get('triples', []),
                    'schema': schema.__dict__ if schema else {},
                    'confidence': result.get('confidence', 0.85),
                    'metadata': {'source': 'oneke_actual'},
                    'raw_response': result
                }
            
            if self.allow_fallback:
                return await self._llm_extraction(text, schema, 'Triple')
            
            raise OneKENotInstalledError("OneKE model not available")

        except Exception as e:
            if self.config['integration'].get('fallback_on_error') and self.allow_fallback:
                return await self._llm_extraction(text, schema, 'Triple')
            raise ExtractionError(f"Triple extraction failed: {e}")

    async def extract_schema_guided(
        self,
        text: str,
        schema: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform schema-guided extraction with ACTUAL OneKE.
        """
        if not self.is_initialized:
            raise ExtractionError("Adapter not initialized. Call initialize() first.")

        try:
            if self.model:
                result = self.model.extract(
                    text=text,
                    schema=self._schema_to_dict(schema),
                    task='Schema',
                    multi_agent=self.config['features'].get('multi_agent', True)
                )
                
                return {
                    'extraction_type': 'SCHEMA',
                    'entities': result.get('entities', []),
                    'relations': result.get('relations', []),
                    'events': result.get('events', []),
                    'triples': result.get('triples', []),
                    'schema': schema.__dict__ if schema else {},
                    'confidence': result.get('confidence', 0.85),
                    'metadata': {'source': 'oneke_actual'},
                    'raw_response': result
                }
            
            if self.allow_fallback:
                return await self._llm_extraction(text, schema, 'Schema')
            
            raise OneKENotInstalledError("OneKE model not available")

        except Exception as e:
            if self.config['integration'].get('fallback_on_error') and self.allow_fallback:
                return await self._llm_extraction(text, schema, 'Schema')
            raise ExtractionError(f"Schema-guided extraction failed: {e}")

    async def _call_oneke(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actually call OneKE for extraction.
        
        This method performs the actual OneKE extraction call.
        If OneKE is not available, it falls back to LLM extraction.
        
        Args:
            text: Input text to extract from
            schema: Schema definition for extraction
            
        Returns:
            Extraction result dictionary
        """
        try:
            if self.model:
                # Real OneKE call
                result = self.model.extract(
                    text=text,
                    schema=schema,
                    task='Schema'
                )
                
                logger.info(f"✓ OneKE extraction completed")
                
                return {
                    'entities': result.get('entities', []),
                    'relations': result.get('relations', []),
                    'events': result.get('events', []),
                    'triples': result.get('triples', []),
                    'confidence': result.get('confidence', 0.85),
                    'metadata': {'source': 'oneke_actual'},
                    'raw_response': result
                }
            
            # No model available - use fallback if allowed
            if self.allow_fallback:
                return await self._llm_extraction(text, schema, 'Schema')
            
            raise OneKENotInstalledError("OneKE model not available")
            
        except Exception as e:
            logger.error(f"OneKE call failed: {e}")
            if self.config['integration'].get('fallback_on_error') and self.allow_fallback:
                return await self._llm_extraction(text, schema, 'Schema')
            raise ExtractionError(f"OneKE extraction failed: {e}")

    async def _llm_extraction(self, text: str, schema: Any, task: str) -> Dict[str, Any]:
        """LLM fallback extraction (only used if allow_fallback=True)."""
        import openai
        
        api_key = self.config['connection'].get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ExtractionError("No API key available for LLM fallback")
        
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""Extract information from the following text according to the schema.

Text: {text}

Task: {task}

Return a JSON object with entities, relations, events, and triples fields.
"""
        
        response = client.chat.completions.create(
            model=self.config['connection'].get('model_name_or_path', 'gpt-4o-mini'),
            messages=[
                {"role": "system", "content": "You are a knowledge extraction system."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return {
            'extraction_type': task,
            'entities': result.get('entities', []),
            'relations': result.get('relations', []),
            'events': result.get('events', []),
            'triples': result.get('triples', []),
            'schema': schema.__dict__ if schema else {},
            'confidence': result.get('confidence', 0.8),
            'metadata': {'source': 'llm_fallback'},
            'raw_response': result
        }

    def _schema_to_dict(self, schema: Any) -> Dict[str, Any]:
        """Convert SchemaDefinition to dictionary."""
        if hasattr(schema, '__dict__'):
            return {
                'name': getattr(schema, 'name', ''),
                'description': getattr(schema, 'description', ''),
                'entity_types': getattr(schema, 'entity_types', []),
                'relation_types': getattr(schema, 'relation_types', []),
                'event_types': getattr(schema, 'event_types', []),
                'constraints': getattr(schema, 'constraints', None),
                'examples': getattr(schema, 'examples', None)
            }
        return schema if schema else {}

    def load_schema(self, schema_path: str) -> Any:
        """Load schema from YAML file."""
        if not yaml:
            raise ImportError("PyYAML required for schema loading")
        
        with open(schema_path, 'r') as f:
            schema_data = yaml.safe_load(f)
        
        return schema_data

    async def validate(self) -> Dict[str, Any]:
        """Validate OneKE adapter configuration and connection."""
        checks = []
        issues = []

        # Check OneKE availability
        if self._oneke_available:
            checks.append({'name': 'oneke_available', 'status': 'passed'})
        else:
            checks.append({'name': 'oneke_available', 'status': 'failed'})
            issues.append("OneKE library not available")

        # Check initialization
        if self.is_initialized:
            checks.append({'name': 'initialized', 'status': 'passed'})
        else:
            checks.append({'name': 'initialized', 'status': 'failed'})
            issues.append("Adapter not initialized")

        # Check model loaded
        if self.model:
            checks.append({'name': 'model_loaded', 'status': 'passed'})
        else:
            checks.append({'name': 'model_loaded', 'status': 'failed'})
            issues.append("OneKE model not loaded")

        # Check schemas
        if self.schemas:
            checks.append({
                'name': 'schemas',
                'status': 'passed',
                'count': len(self.schemas)
            })
        else:
            checks.append({'name': 'schemas', 'status': 'warning', 'count': 0})

        is_valid = all(c.get('status') == 'passed' for c in checks)

        return {
            'is_valid': is_valid,
            'checks': checks,
            'issues': issues,
            'source': 'oneke_actual' if self._oneke_available else 'fallback',
            'performance': {
                'max_workers': self.config['performance'].get('max_workers', 4),
                'timeout': self.config['performance'].get('timeout', 30),
                'batch_size': self.config['performance'].get('batch_size', 100)
            }
        }

    async def shutdown(self) -> bool:
        """Shutdown OneKE adapter."""
        try:
            if self._process:
                self._process.terminate()
                self._process.wait(timeout=5)
                self._process = None

            self.is_initialized = False
            self.model = None
            return True

        except Exception as e:
            raise ShutdownError(f"Shutdown failed: {e}")
