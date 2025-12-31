#!/usr/bin/env python3
"""
Case 2 Hierarchical Model Generator
Generates multi-stage Pydantic models for hierarchical extraction
"""

import json
import logging
from typing import Dict, List, Any, Type, Tuple, Optional
from pydantic import BaseModel, Field
from .case2_core import ExtractionStrategy, ExtractionStage, StageExtractor

logger = logging.getLogger(__name__)

class Case2ModelGenerator:
    """Generates hierarchical Pydantic models for Case 2 extraction"""
    
    def __init__(self, ai_client=None):
        self.ai_client = ai_client
        if not self.ai_client:
            raise ValueError("AI client is required for model generation")
    
    def generate_hierarchical_models(self, strategy: ExtractionStrategy, original_model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Type[BaseModel]]:
        """
        Generate Pydantic models for each extraction stage
        No hard-coded patterns - purely AI-driven model generation
        """
        try:
            logger.info(f"Generating hierarchical models for strategy: {strategy.use_case_name}")
            
            stage_models = {}
            
            for stage in strategy.stages:
                logger.info(f"Generating model for stage: {stage.stage_name}")
                
                # Generate model for this stage
                model_class = self._generate_stage_model(stage, strategy, original_model_info)
                stage_models[stage.stage_name] = model_class
            
            logger.info(f"Generated {len(stage_models)} stage models")
            return stage_models
            
        except Exception as e:
            logger.error(f"Error generating hierarchical models: {e}")
            raise
    
    def _generate_stage_model(self, stage: ExtractionStage, 
                            strategy: ExtractionStrategy, 
                            original_model_info: Optional[Dict[str, Any]] = None) -> Type[BaseModel]:
        """Generate Pydantic model for a single stage"""
        
        # Create prompt for model generation
        prompt = self._create_model_generation_prompt(stage, strategy, original_model_info)
        
        try:
            # Use the adapter's generate_response method
            system_prompt = "You are an expert Python developer specializing in Pydantic models. Generate clean, well-structured Pydantic models based on the provided specifications."
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response_text = self.ai_client.generate_response(
                prompt=full_prompt,
                max_tokens=self.ai_client.max_tokens,
                temperature=0.0
            )
            
            # Extract and clean model code
            model_code = self._extract_model_code(response_text)
            
            # Create dynamic model class
            model_class = self._create_dynamic_model(stage.stage_name, model_code)
            
            return model_class
            
        except Exception as e:
            logger.error(f"Error generating model for stage {stage.stage_name}: {e}")
            # Fallback to basic model
            return self._create_fallback_model(stage)
    
    def _create_model_generation_prompt(self, stage: ExtractionStage, 
                                      strategy: ExtractionStrategy, 
                                      original_model_info: Optional[Dict[str, Any]] = None) -> str:
        """Create AI prompt for model generation"""
        
        # Build field descriptions
        field_descriptions = []
        for field in stage.extraction_fields:
            field_descriptions.append(f"- {field['field_name']} ({field['field_type']}): {field['description']}")
        
        # Build input key descriptions
        input_key_descriptions = []
        for key in stage.input_keys:
            input_key_descriptions.append(f"- {key}: Input from previous stages")
        
        # Build output key descriptions
        output_key_descriptions = []
        for key in stage.output_keys:
            output_key_descriptions.append(f"- {key}: Output for next stages")
        
        prompt = f"""
        Generate a Pydantic model for hierarchical document extraction stage.
        
        STAGE INFORMATION:
        - Stage Name: {stage.stage_name}
        - Stage Type: {stage.stage_type.value}
        - Document Types: {', '.join(stage.document_types)}
        - Description: {stage.description}
        
        EXTRACTION FIELDS:
        {chr(10).join(field_descriptions)}
        
        INPUT KEYS (from previous stages):
        {chr(10).join(input_key_descriptions) if input_key_descriptions else "- None (initial stage)"}
        
        OUTPUT KEYS (for next stages):
        {chr(10).join(output_key_descriptions) if output_key_descriptions else "- None (final stage)"}
        
        RELATIONSHIPS:
        """
        
        # Add relationship information
        for rel in strategy.relationships:
            if rel.source_stage == stage.stage_name or rel.target_stage == stage.stage_name:
                prompt += f"- {rel.source_stage} -> {rel.target_stage} via {rel.key_field}: {rel.description}\n"
        
        prompt += f"""
        
        REQUIREMENTS:
        1. Create a Pydantic BaseModel class named {stage.stage_name}Model
        2. Include all extraction fields with appropriate types and descriptions
        3. Include input keys as optional fields (they come from previous stages)
        4. Include output keys as required fields (they are produced by this stage)
        5. Add proper Field descriptions for all fields
        6. Use appropriate Pydantic types (str, int, float, bool, List[str], etc.)
        7. Add field validators if needed
        8. Make the model generic and reusable"""
        
        # Add original model field naming convention if available
        if original_model_info and 'field_names' in original_model_info:
            prompt += f"""
        
        FIELD NAMING CONVENTION:
        IMPORTANT: Use the same field naming convention as the original model to ensure consistency.
        Original model field names: {original_model_info['field_names']}
        
        For fields that correspond to the original model, use the EXACT same field names.
        For example, if the original model uses 'fieldname' (no underscore), use 'fieldname' in this model too.
        Only use snake_case for new fields that don't exist in the original model."""
        
        prompt += f"""
        
        Return only the Python code for the Pydantic model class, no explanations or markdown formatting.
        
        Example structure:
        ```python
        from pydantic import BaseModel, Field
        from typing import List, Optional
        
        class {stage.stage_name}Model(BaseModel):
            # Input keys (optional, from previous stages)
            # Output keys (required, produced by this stage)
            # Extraction fields (required, extracted from documents)
            
            class Config:
                extra = "forbid"
        ```
        """
        
        return prompt
    
    def _extract_model_code(self, response: str) -> str:
        """Extract and clean model code from AI response"""
        import re
        
        # Remove markdown code blocks
        code = re.sub(r'```python\s*\n?', '', response)
        code = re.sub(r'```\s*\n?', '', code)
        
        # Remove common AI artifacts
        code = re.sub(r'^Here\'s the.*?:\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^Here is the.*?:\n', '', code, flags=re.MULTILINE)
        
        # Clean up whitespace
        code = code.strip()
        
        return code
    
    def _create_dynamic_model(self, stage_name: str, model_code: str) -> Type[BaseModel]:
        """Create dynamic Pydantic model from generated code"""
        try:
            # Create a namespace for the model
            namespace = {
                'BaseModel': BaseModel,
                'Field': Field,
                '__builtins__': __builtins__
            }
            
            # Add typing imports
            from typing import List, Optional, Dict, Any
            namespace.update({
                'List': List,
                'Optional': Optional,
                'Dict': Dict,
                'Any': Any
            })
            
            # Execute the model code
            exec(model_code, namespace)
            
            # Find the model class
            model_class_name = f"{stage_name}Model"
            if model_class_name in namespace:
                return namespace[model_class_name]
            else:
                # Try alternative naming
                for key, value in namespace.items():
                    if isinstance(value, type) and issubclass(value, BaseModel) and key != 'BaseModel':
                        return value
                
                raise ValueError(f"Model class {model_class_name} not found in generated code")
                
        except Exception as e:
            logger.error(f"Error creating dynamic model: {e}")
            raise
    
    def _create_fallback_model(self, stage: ExtractionStage) -> Type[BaseModel]:
        """Create a basic fallback model when AI generation fails"""
        from typing import Optional
        
        # Create basic field definitions - Pydantic V2 compatible
        field_definitions = {}
        annotations = {}
        
        # Add input keys as optional fields
        for key in stage.input_keys:
            clean_key = key.replace(' ', '_').replace('-', '_').lower()
            annotations[clean_key] = Optional[str]
            field_definitions[clean_key] = Field(default=None, description=f"Input key: {key}")
        
        # Add output keys as required fields
        for key in stage.output_keys:
            clean_key = key.replace(' ', '_').replace('-', '_').lower()
            annotations[clean_key] = str
            field_definitions[clean_key] = Field(description=f"Output key: {key}")
        
        # Add extraction fields
        for field in stage.extraction_fields:
            field_name = field['field_name'].replace(' ', '_').replace('-', '_').lower()
            field_type = self._get_python_type(field['field_type'])
            annotations[field_name] = field_type
            field_definitions[field_name] = Field(description=field['description'])
        
        # Create the model class - Pydantic V2 compatible
        class_name = f"{stage.stage_name.replace(' ', '').replace('-', '')}Model"
        
        # Create model attributes
        model_attrs = {
            '__annotations__': annotations,
            **field_definitions
        }
        
        # Create the model dynamically
        model_class = type(class_name, (BaseModel,), model_attrs)
        
        return model_class
    
    def _get_python_type(self, field_type: str) -> type:
        """Convert field type string to Python type"""
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list[str]': List[str],
            'list': List[str],
            'dict': Dict[str, Any]
        }
        
        return type_mapping.get(field_type.lower(), str)
    
    def generate_stage_extractors(self, strategy: ExtractionStrategy, 
                                stage_models: Dict[str, Type[BaseModel]], 
                                original_model_info: Optional[Dict[str, Any]] = None) -> List[StageExtractor]:
        """Generate stage extractor configurations"""
        extractors = []
        
        for stage in strategy.stages:
            model_class = stage_models.get(stage.stage_name)
            if not model_class:
                logger.warning(f"No model found for stage: {stage.stage_name}")
                continue
            
            # Generate prompt for this stage
            prompt = self._generate_stage_prompt(stage, strategy, original_model_info)
            
            # Create stage extractor
            extractor = StageExtractor(
                stage_name=stage.stage_name,
                stage_type=stage.stage_type,
                document_types=stage.document_types,
                model_class=model_class,
                prompt=prompt,
                input_keys=stage.input_keys,
                output_keys=stage.output_keys,
                extraction_fields=stage.extraction_fields,
                config_path=""  # Will be set when saving
            )
            
            extractors.append(extractor)
        
        return extractors
    
    def _generate_stage_prompt(self, stage: ExtractionStage, 
                             strategy: ExtractionStrategy, 
                             original_model_info: Optional[Dict[str, Any]] = None) -> str:
        """Generate extraction prompt for a stage with full model structure and standard instructions"""
        
        # Build field descriptions
        field_descriptions = []
        for field in stage.extraction_fields:
            field_descriptions.append(f"- {field['field_name']} ({field['field_type']}): {field['description']}")
        
        # Build input key descriptions
        input_key_descriptions = []
        for key in stage.input_keys:
            input_key_descriptions.append(f"- {key}: Use this key to find related information")
        
        # Generate model class name for this stage
        stage_model_name = f"{stage.stage_name.replace(' ', '').replace('-', '')}Model"
        
        # Build Pydantic model structure for the prompt
        model_structure = self._build_model_structure_for_prompt(stage, original_model_info)
        
        prompt = f"""
TASK: {stage.stage_name}

EXTRACTION TASK:
{stage.description}

EMBEDDED PYDANTIC MODELS:
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class {stage_model_name}(BaseModel):
{model_structure}
```

CRITICAL EXTRACTION RULES:
1. ACCURACY & VERIFICATION:
   - Extract information ONLY from the provided text
   - Never fabricate, infer, or guess any information
   - Use 'n/a' for any fields where information is not explicitly stated
   - Verify all extracted data against the source text
   - Maintain exact values, dates, and numerical figures as written

2. DATA HANDLING:
   - For dates: Use DD-MM-YYYY format when possible
   - For numbers: Preserve original precision and units
   - For text: Maintain original spelling and capitalization
   - For lists: Extract all relevant items, remove duplicates
   - For enums: Choose ONLY from the specified options
   - For fields whose length exceeds 30 words, summarize into less than 30 words with key phrases separated by semi-colons

3. FIELD VALIDATION:
   - Enum fields must match one of the specified values exactly
   - List fields should contain valid, non-empty items
   - Numerical fields should be valid numbers in correct format

4. QUALITY ASSURANCE:
   - Double-check all extracted information against source
   - Ensure no information is duplicated across fields
   - Verify that field types match the expected data types
   - Confirm that all required fields are addressed
   - Validate that enum selections are from available options

OUTPUT FORMAT:
Return the extracted information as a JSON object that exactly matches the {stage_model_name} structure shown above.

CRITICAL: All JSON field names MUST be in snake_case format (lowercase with underscores) to match the model exactly.
For example: "field_name", "delivery_date", "type_designation" - NOT "Field Name", "Delivery Date", "Type Designation"

Example output structure:
```json
{{
  "field_name_1": "extracted_value_or_n/a",
  "field_name_2": ["list", "of", "values"],
  "field_name_3": "enum_option_or_n/a"
}}
```

VALIDATION CHECKLIST:
Before returning your response, verify:
- All required fields are populated or marked 'n/a'
- All enum fields contain valid options only
- All numerical fields contain valid numbers
- All date fields follow proper format
- No information is fabricated or inferred
- JSON structure matches the model exactly
- Field names match the model exactly (snake_case)
- Data types are appropriate for each field

FINAL INSTRUCTIONS:
- Process the document systematically
- Extract information field by field as specified above
- Maintain accuracy over completeness
- When in doubt, use 'n/a' rather than guessing
- Return only the JSON object with extracted data
- Ensure the output can be parsed as valid JSON

DOCUMENTS TO PROCESS:
{{documents}}

Return the extracted information as a JSON array of objects (if multiple items) or single JSON object matching the {stage_model_name} structure.
        """
        
        return prompt
    
    def _build_model_structure_for_prompt(self, stage: ExtractionStage, original_model_info: Optional[Dict[str, Any]] = None) -> str:
        """Build Pydantic model structure string for inclusion in prompts"""
        lines = []
        
        # Get original field names if available
        original_field_names = []
        if original_model_info and 'field_names' in original_model_info:
            original_field_names = original_model_info['field_names']
        
        for field in stage.extraction_fields:
            # Use original field name if it exists, otherwise convert to snake_case
            field_name = field['field_name']
            
            # Check if this field corresponds to an original field
            original_field_name = None
            for orig_name in original_field_names:
                # Normalize both field names for comparison
                normalized_field = field_name.lower().replace(' ', '').replace('-', '').replace('_', '').replace('/', '')
                normalized_orig = orig_name.lower().replace(' ', '').replace('-', '').replace('_', '').replace('/', '')
                
                # Check for exact match or semantic match
                if (normalized_field == normalized_orig or 
                    # Handle special cases like ID -> materialnumber
                    (normalized_field == 'id' and normalized_orig == 'materialnumber') or
                    # Handle Type/Part Designation -> typepartdesignation
                    ('type' in normalized_field and 'part' in normalized_field and 'designation' in normalized_field and 
                     'type' in normalized_orig and 'part' in normalized_orig and 'designation' in normalized_orig)):
                    original_field_name = orig_name
                    break
            
            # Use original field name if found, otherwise convert to snake_case
            if original_field_name:
                field_name = original_field_name
            else:
                field_name = field['field_name'].replace(' ', '_').replace('-', '_').replace('/', '_').lower()
            
            field_type = field['field_type']
            description = field['description']
            required = field.get('required', True)
            
            # Map field types to Python types for display
            type_mapping = {
                'str': 'str',
                'int': 'int',
                'float': 'float',
                'bool': 'bool',
                'list[str]': 'List[str]',
                'list': 'List[str]',
                'dict': 'Dict[str, Any]'
            }
            
            python_type = type_mapping.get(field_type.lower(), 'str')
            
            if required:
                lines.append(f'    {field_name}: {python_type} = Field(description="{description}")')
            else:
                lines.append(f'    {field_name}: Optional[{python_type}] = Field(default=None, description="{description}")')
        
        return '\n'.join(lines) if lines else '    pass'
    
    def save_models_and_prompts(self, strategy: ExtractionStrategy, 
                              stage_models: Dict[str, Type[BaseModel]], 
                              stage_extractors: List[StageExtractor], 
                              use_case_path: str) -> None:
        """Save all models and prompts to files"""
        try:
            import os
            
            # Create use case directory
            os.makedirs(use_case_path, exist_ok=True)
            
            # Save main strategy file
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
                config_file = os.path.join(stage_dir, "config.json")
                self._save_stage_config(extractor, config_file)
            
            logger.info(f"Saved models and prompts to: {use_case_path}")
            
        except Exception as e:
            logger.error(f"Error saving models and prompts: {e}")
            raise
    
    def _save_strategy_file(self, strategy: ExtractionStrategy, file_path: str) -> None:
        """Save strategy to Python file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Strategy for {strategy.use_case_name}\n")
            f.write(f"# Generated by Case 2 Hierarchical Extraction\n\n")
            f.write(f"description = \"{strategy.description}\"\n")
            f.write(f"extraction_sequence = {strategy.extraction_sequence}\n")
            f.write(f"document_classifications = {strategy.document_classifications}\n")
    
    def _save_model_file(self, model_class: Type[BaseModel], file_path: str) -> None:
        """Save model class to Python file with actual executable code"""
        try:
            # Generate the actual model code
            model_code = self._generate_model_code(model_class)
            
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
                f.write(f"# Model: {model_class.__name__}\n")
                f.write(f"# Generated by Case 2 Hierarchical Extraction\n")
                f.write(f"# Error occurred during model code generation: {e}\n\n")
                f.write(f"# Fields: {list(getattr(model_class, '__annotations__', {}).keys())}\n")
    
    def _save_prompt_file(self, prompt: str, file_path: str) -> None:
        """Save prompt to Python file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Extraction prompt\n")
            f.write(f"# Generated by Case 2 Hierarchical Extraction\n\n")
            f.write(f"extraction_prompt = \"\"\"{prompt}\"\"\"\n")
    
    def _generate_model_code(self, model_class: Type[BaseModel]) -> str:
        """Generate Python code for a Pydantic model class"""
        try:
            class_name = model_class.__name__
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
                type_str = self._format_type_annotation(field_type)
                
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
    
    def _format_type_annotation(self, field_type) -> str:
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
    
    def _save_stage_config(self, extractor: StageExtractor, file_path: str) -> None:
        """Save stage configuration to JSON file"""
        config = {
            "stage_name": extractor.stage_name,
            "stage_type": extractor.stage_type.value,
            "document_types": extractor.document_types,
            "input_keys": extractor.input_keys,
            "output_keys": extractor.output_keys,
            "extraction_fields": extractor.extraction_fields
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
