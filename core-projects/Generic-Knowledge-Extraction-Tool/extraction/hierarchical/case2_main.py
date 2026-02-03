#!/usr/bin/env python3
"""
Case 2 Main Orchestrator
Main entry point for Case 2 hierarchical multi-document extraction
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Case2 Main
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


import json
import logging
import os
from typing import Dict, List, Any, Optional
from .case2_core import ExtractionStrategy, Case2Config, HierarchicalExtractionResult
from .case2_strategy_generator import Case2StrategyGenerator
from .case2_model_generator import Case2ModelGenerator
from .case2_extractor import Case2SequentialExtractor

logger = logging.getLogger(__name__)

class Case2Orchestrator:
    """Main orchestrator for Case 2 hierarchical extraction"""
    
    def __init__(self, ai_client=None):
        self.ai_client = ai_client
        if not self.ai_client:
            raise ValueError("AI client is required for Case 2 orchestration")
        
        # Use the raw client directly (it's the OpenAI client from extractor.client)
        self.strategy_generator = Case2StrategyGenerator(self.ai_client)
        self.model_generator = Case2ModelGenerator(self.ai_client)
    
    def create_new_use_case(self, description: str, use_case_name: str, 
                          use_case_path: str, original_model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new Case 2 use case from text description
        Complete workflow: strategy generation -> model generation -> persistence
        """
        try:
            logger.info(f"Creating new Case 2 use case: {use_case_name}")
            
            # Step 1: Generate extraction strategy
            logger.info("Generating extraction strategy...")
            strategy = self.strategy_generator.generate_strategy(description, use_case_name)
            
            # Validate strategy
            issues = self.strategy_generator.validate_strategy(strategy)
            if issues:
                logger.warning(f"Strategy validation issues: {issues}")
            
            # Step 2: Generate hierarchical models
            logger.info("Generating hierarchical models...")
            stage_models = self.model_generator.generate_hierarchical_models(strategy, original_model_info)
            
            # Step 3: Generate stage extractors
            logger.info("Generating stage extractors...")
            stage_extractors = self.model_generator.generate_stage_extractors(strategy, stage_models, original_model_info)
            
            # Step 4: Save everything to files
            logger.info("Saving use case files...")
            self._save_use_case(strategy, stage_models, stage_extractors, use_case_path)
            
            # Step 5: Create config
            config = Case2Config.create_config(use_case_name, strategy)
            
            logger.info(f"Case 2 use case created successfully: {use_case_name}")
            return {
                'success': True,
                'use_case_name': use_case_name,
                'strategy': strategy,
                'stage_models': stage_models,
                'stage_extractors': stage_extractors,
                'config': config,
                'use_case_path': use_case_path
            }
            
        except Exception as e:
            logger.error(f"Error creating Case 2 use case: {e}")
            return {
                'success': False,
                'error': str(e),
                'use_case_name': use_case_name
            }
    
    def load_persisted_use_case(self, use_case_path: str) -> Dict[str, Any]:
        """
        Load a persisted Case 2 use case
        No regeneration needed - everything is loaded from files
        """
        try:
            logger.info(f"Loading persisted Case 2 use case: {use_case_path}")
            
            # Step 1: Load strategy from config
            config_path = os.path.join(use_case_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            strategy = Case2Config.load_strategy_from_config(config_path)
            
            # Step 2: Load stage models and extractors
            stage_models = {}
            stage_extractors = []
            
            for stage in strategy.stages:
                stage_dir = os.path.join(use_case_path, stage.stage_name)
                if os.path.exists(stage_dir):
                    # Load model
                    model_file = os.path.join(stage_dir, f"{stage.stage_name}_models.py")
                    if os.path.exists(model_file):
                        model_class = self._load_model_from_file(model_file, stage.stage_name)
                        stage_models[stage.stage_name] = model_class
                    
                    # Load prompt
                    prompt_file = os.path.join(stage_dir, f"{stage.stage_name}_prompt.py")
                    prompt = ""
                    if os.path.exists(prompt_file):
                        prompt = self._load_prompt_from_file(prompt_file)
                    
                    # Create stage extractor
                    from .case2_core import StageExtractor
                    extractor = StageExtractor(
                        stage_name=stage.stage_name,
                        stage_type=stage.stage_type,
                        document_types=stage.document_types,
                        model_class=stage_models.get(stage.stage_name),
                        prompt=prompt,
                        input_keys=stage.input_keys,
                        output_keys=stage.output_keys,
                        extraction_fields=stage.extraction_fields,
                        config_path=stage_dir
                    )
                    stage_extractors.append(extractor)
            
            logger.info(f"Loaded Case 2 use case: {strategy.use_case_name}")
            return {
                'success': True,
                'strategy': strategy,
                'stage_models': stage_models,
                'stage_extractors': stage_extractors,
                'use_case_path': use_case_path
            }
            
        except Exception as e:
            logger.error(f"Error loading persisted Case 2 use case: {e}")
            return {
                'success': False,
                'error': str(e),
                'use_case_path': use_case_path
            }
    
    def extract_documents(self, documents: List[Dict[str, Any]], 
                         use_case_path: str) -> HierarchicalExtractionResult:
        """
        Extract documents using a Case 2 use case
        Loads persisted use case and performs extraction
        """
        try:
            logger.info(f"Extracting documents using Case 2 use case: {use_case_path}")
            
            # Load persisted use case
            load_result = self.load_persisted_use_case(use_case_path)
            if not load_result['success']:
                raise ValueError(f"Failed to load use case: {load_result['error']}")
            
            strategy = load_result['strategy']
            stage_extractors = load_result['stage_extractors']
            
            # Create sequential extractor
            extractor = Case2SequentialExtractor(strategy, stage_extractors, self.ai_client)
            
            # Perform extraction
            result = extractor.extract_hierarchically(documents)
            
            logger.info(f"Case 2 extraction completed: {len(result.consolidated_results)} consolidated records")
            return result
            
        except Exception as e:
            logger.error(f"Error in Case 2 document extraction: {e}")
            raise
    
    def _save_use_case(self, strategy: ExtractionStrategy, 
                      stage_models: Dict[str, Any], 
                      stage_extractors: List[Any], 
                      use_case_path: str) -> None:
        """Save complete use case to files"""
        try:
            # Create use case directory
            os.makedirs(use_case_path, exist_ok=True)
            
            # Save main config
            config = Case2Config.create_config(strategy.use_case_name, strategy)
            config_path = os.path.join(use_case_path, "config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            # Save strategy file
            strategy_file = os.path.join(use_case_path, f"{strategy.use_case_name}_strategy.py")
            self._save_strategy_file(strategy, strategy_file)
            
            # Save each stage
            for extractor in stage_extractors:
                stage_dir = os.path.join(use_case_path, extractor.stage_name)
                os.makedirs(stage_dir, exist_ok=True)
                
                # Save model
                model_file = os.path.join(stage_dir, f"{extractor.stage_name}_models.py")
                self._save_model_file(extractor.model_class, model_file)
                
                # Save prompt
                prompt_file = os.path.join(stage_dir, f"{extractor.stage_name}_prompt.py")
                self._save_prompt_file(extractor.prompt, prompt_file)
                
                # Save stage config
                stage_config = {
                    "stage_name": extractor.stage_name,
                    "stage_type": extractor.stage_type.value,
                    "document_types": extractor.document_types,
                    "input_keys": extractor.input_keys,
                    "output_keys": extractor.output_keys,
                    "extraction_fields": extractor.extraction_fields
                }
                config_file = os.path.join(stage_dir, "config.json")
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(stage_config, f, indent=2)
            
            logger.info(f"Use case saved to: {use_case_path}")
            
        except Exception as e:
            logger.error(f"Error saving use case: {e}")
            raise
    
    def _save_strategy_file(self, strategy: ExtractionStrategy, file_path: str) -> None:
        """Save strategy to Python file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Strategy for {strategy.use_case_name}\n")
            f.write(f"# Generated by Case 2 Hierarchical Extraction\n\n")
            f.write(f"description = \"{strategy.description}\"\n")
            f.write(f"extraction_sequence = {strategy.extraction_sequence}\n")
            
            # Safely format document_classifications
            try:
                import json
                f.write(f"document_classifications = {json.dumps(strategy.document_classifications, indent=2)}\n")
            except Exception:
                f.write(f"document_classifications = {{}}\n")
    
    def _save_model_file(self, model_class: Any, file_path: str) -> None:
        """Save model class to Python file with actual executable code"""
        try:
            # Generate the actual model code
            model_code = self._generate_model_code_for_class(model_class)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Model: {model_class.__name__}\n")
                f.write(f"# Generated by Case 2 Hierarchical Extraction\n\n")
                f.write("from pydantic import BaseModel, Field\n")
                f.write("from typing import List, Optional, Dict, Any\n\n")
                f.write(model_code)
                
        except Exception as e:
            logger.error(f"Error saving model file: {e}")
            # Fallback to basic comments for debugging
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Model: {getattr(model_class, '__name__', 'UnknownModel')}\n")
                f.write(f"# Generated by Case 2 Hierarchical Extraction\n")
                f.write(f"# Error occurred during model code generation: {e}\n\n")
                try:
                    if hasattr(model_class, '__annotations__'):
                        fields = list(model_class.__annotations__.keys())
                    elif hasattr(model_class, 'model_fields'):
                        fields = list(model_class.model_fields.keys())
                    else:
                        fields = ['Unknown']
                    f.write(f"# Fields: {fields}\n")
                except Exception:
                    f.write(f"# Fields: Unknown\n")
    
    def _save_prompt_file(self, prompt: str, file_path: str) -> None:
        """Save prompt to Python file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Extraction prompt\n")
            f.write(f"# Generated by Case 2 Hierarchical Extraction\n\n")
            f.write(f"extraction_prompt = \"\"\"{prompt}\"\"\"\n")
    
    def _load_model_from_file(self, file_path: str, stage_name: str) -> Any:
        """Load model class from file by executing the saved Python code"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Model file not found: {file_path}")
                return self._create_fallback_model_for_stage(stage_name)
            
            # Read the model file
            with open(file_path, 'r', encoding='utf-8') as f:
                model_code = f.read()
            
            # Check if it's a comment-only file (old format)
            if "class " not in model_code or "BaseModel" not in model_code:
                logger.warning(f"Model file contains no executable code: {file_path}")
                return self._create_fallback_model_for_stage(stage_name)
            
            # Create namespace for executing the model
            from pydantic import BaseModel, Field
            from typing import List, Optional, Dict, Any
            
            namespace = {
                'BaseModel': BaseModel,
                'Field': Field,
                'List': List,
                'Optional': Optional,
                'Dict': Dict,
                'Any': Any,
                '__builtins__': __builtins__
            }
            
            # Execute the model code
            exec(model_code, namespace)
            
            # Find the model class
            for name, obj in namespace.items():
                if (isinstance(obj, type) and 
                    issubclass(obj, BaseModel) and 
                    obj != BaseModel and
                    not name.startswith('_')):
                    logger.info(f"Loaded model class {name} from {file_path}")
                    return obj
            
            # If no model class found, create fallback
            logger.warning(f"No model class found in {file_path}")
            return self._create_fallback_model_for_stage(stage_name)
            
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {e}")
            return self._create_fallback_model_for_stage(stage_name)
    
    def _create_fallback_model_for_stage(self, stage_name: str) -> Any:
        """Create a fallback model when loading fails"""
        try:
            # Try to load extraction fields from stage config
            stage_dir = os.path.dirname(os.path.dirname(file_path)) if 'file_path' in locals() else ""
            config_path = os.path.join(stage_dir, "config.json")
            
            extraction_fields = []
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    stage_config = json.load(f)
                    extraction_fields = stage_config.get('extraction_fields', [])
            
            # If no config found, try to infer from the main use case config
            if not extraction_fields:
                # Load from main config.json
                use_case_config_path = os.path.join(os.path.dirname(stage_dir), "config.json")
                if os.path.exists(use_case_config_path):
                    with open(use_case_config_path, 'r') as f:
                        config = json.load(f)
                        strategy = config.get('extraction_config', {}).get('extraction_strategy', {})
                        stages = strategy.get('stages', [])
                        
                        # Find this stage's fields
                        for stage in stages:
                            if stage.get('stage_name') == stage_name:
                                extraction_fields = stage.get('extraction_fields', [])
                                break
            
            # Create model with actual extraction fields
            from pydantic import BaseModel, Field
            from typing import Optional
            
            field_definitions = {}
            annotations = {}
            
            for field in extraction_fields:
                field_name = field.get('field_name', '').replace(' ', '_').replace('-', '_').lower()
                field_type = self._get_python_type_from_string(field.get('field_type', 'str'))
                description = field.get('description', 'No description')
                required = field.get('required', True)
                
                if required:
                    annotations[field_name] = field_type
                    field_definitions[field_name] = Field(description=description)
                else:
                    annotations[field_name] = Optional[field_type]
                    field_definitions[field_name] = Field(default=None, description=description)
            
            # Fallback fields if no extraction fields found
            if not annotations:
                annotations = {
                    'stage_name': str,
                    'extracted_data': str
                }
                field_definitions = {
                    'stage_name': Field(default=stage_name, description="Stage name"),
                    'extracted_data': Field(default="", description="Extracted data placeholder")
                }
            
            # Create model class
            class_name = f"{stage_name.replace(' ', '').replace('-', '')}FallbackModel"
            model_attrs = {
                '__annotations__': annotations,
                **field_definitions
            }
            
            model_class = type(class_name, (BaseModel,), model_attrs)
            logger.info(f"Created fallback model {class_name} with fields: {list(annotations.keys())}")
            return model_class
            
        except Exception as e:
            logger.error(f"Error creating fallback model: {e}")
            # Ultimate fallback
            from pydantic import BaseModel, Field
            
            class UltimateFallbackModel(BaseModel):
                stage_name: str = Field(default=stage_name, description="Stage name")
                data: str = Field(default="", description="Generic data field")
            
            return UltimateFallbackModel
    
    def _get_python_type_from_string(self, type_str: str):
        """Convert type string to Python type"""
        from typing import List, Dict, Optional, Any
        
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list[str]': List[str],
            'list': List[str],
            'dict': Dict[str, Any]
        }
        
        return type_mapping.get(type_str.lower(), str)
    
    def _load_prompt_from_file(self, file_path: str) -> str:
        """Load prompt from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract prompt from file
            import re
            prompt_match = re.search(r'extraction_prompt = """(.*?)"""', content, re.DOTALL)
            if prompt_match:
                return prompt_match.group(1)
            else:
                return "Default extraction prompt"
                
        except Exception as e:
            logger.error(f"Error loading prompt from file: {e}")
            return "Default extraction prompt"
    
    def get_use_case_info(self, use_case_path: str) -> Dict[str, Any]:
        """Get information about a Case 2 use case"""
        try:
            config_path = os.path.join(use_case_path, "config.json")
            if not os.path.exists(config_path):
                return {'error': 'Config file not found'}
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            extraction_config = config['extraction_config']
            strategy_data = extraction_config['extraction_strategy']
            
            return {
                'use_case_name': extraction_config['use_case'],
                'description': extraction_config['description'],
                'extraction_type': extraction_config['extraction_type'],
                'stages': len(strategy_data['stages']),
                'relationships': len(strategy_data['relationships']),
                'document_types': list(set([doc_type for stage in strategy_data['stages'] for doc_type in stage['document_types']])),
                'extraction_sequence': strategy_data['extraction_sequence']
            }
            
        except Exception as e:
            logger.error(f"Error getting use case info: {e}")
            return {'error': str(e)}
    
    def _generate_model_code_for_class(self, model_class: Any) -> str:
        """Generate Python code for a Pydantic model class"""
        try:
            class_name = getattr(model_class, '__name__', 'UnknownModel')
            annotations = getattr(model_class, '__annotations__', {})
            
            # Start building the model code
            lines = [f"class {class_name}(BaseModel):"]
            lines.append(f'    """Generated Pydantic model for Case 2 extraction"""')
            
            if not annotations:
                lines.append("    pass")
                return "\n".join(lines)
            
            # Process each field
            for field_name, field_type in annotations.items():
                field_info = None
                
                # Try to get field info from model_fields (Pydantic v2)
                if hasattr(model_class, 'model_fields'):
                    field_info = model_class.model_fields.get(field_name)
                
                # Generate field type string
                type_str = self._format_type_annotation_for_class(field_type)
                
                # Generate field definition
                if field_info and hasattr(field_info, 'description'):
                    description = field_info.description or "No description"
                    default = "None" if "Optional" in type_str else "..."
                    lines.append(f'    {field_name}: {type_str} = Field(default={default}, description="{description}")')
                else:
                    # Simple field without Field specification
                    lines.append(f'    {field_name}: {type_str}')
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error generating model code: {e}")
            return f"class {getattr(model_class, '__name__', 'UnknownModel')}(BaseModel):\n    '''Error generating model: {e}'''\n    pass"
    
    def _format_type_annotation_for_class(self, field_type) -> str:
        """Format type annotation for code generation"""
        try:
            # Handle string representations
            if hasattr(field_type, '__name__'):
                type_name = field_type.__name__
                if type_name in ['str', 'int', 'float', 'bool']:
                    return type_name
            
            # Handle typing generics
            type_str = str(field_type)
            
            # Clean up common type patterns
            type_str = type_str.replace('typing.', '')
            type_str = type_str.replace('<class \'', '').replace('\'>', '')
            
            return type_str
            
        except Exception:
            return "str"  # Default fallback
    
    def validate_use_case(self, use_case_path: str) -> Dict[str, Any]:
        """Validate a Case 2 use case"""
        try:
            # Load use case
            load_result = self.load_persisted_use_case(use_case_path)
            if not load_result['success']:
                return {'valid': False, 'error': load_result['error']}
            
            strategy = load_result['strategy']
            stage_extractors = load_result['stage_extractors']
            
            # Validate strategy
            issues = self.strategy_generator.validate_strategy(strategy)
            
            # Check file completeness
            missing_files = []
            for stage in strategy.stages:
                stage_dir = os.path.join(use_case_path, stage.stage_name)
                required_files = [
                    f"{stage.stage_name}_models.py",
                    f"{stage.stage_name}_prompt.py",
                    "config.json"
                ]
                
                for file_name in required_files:
                    file_path = os.path.join(stage_dir, file_name)
                    if not os.path.exists(file_path):
                        missing_files.append(file_path)
            
            return {
                'valid': len(issues) == 0 and len(missing_files) == 0,
                'strategy_issues': issues,
                'missing_files': missing_files,
                'stages': len(strategy.stages),
                'relationships': len(strategy.relationships)
            }
            
        except Exception as e:
            logger.error(f"Error validating use case: {e}")
            return {'valid': False, 'error': str(e)}
