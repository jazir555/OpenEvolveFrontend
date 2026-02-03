import json
import os
import re
import tempfile
import importlib.util
import sys
from typing import Dict, Any, Type, List
from pathlib import Path
import logging
from pydantic import BaseModel
from ai.clients.claude_client import ClaudeClient
from ai.clients.openai_client import OpenAIClient
from core.text_description_parser import TextDescriptionParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelGenerator:
    """Dynamic Pydantic model generator using Claude AI or OpenAI"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Model Generator
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

    
    def __init__(self, model_selection='claude-3-5-sonnet-20241022', api_config=None):
        self.model_selection = model_selection
        self.api_config = api_config
        self.claude_client = None
        self.openai_client = None
        
        # Initialize the appropriate client based on model selection
        if 'claude' in model_selection.lower():
            self.claude_client = ClaudeClient()
        else:
            # For OpenAI models, use centralized OpenAI client
            self.openai_client = OpenAIClient(api_config=api_config)
        
        self.generated_models = {}
        self.extraction_prompt = ""
    
    def load_field_config(self, config_path: str) -> Dict[str, Any]:
        """Load field configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config['extraction_config']
        except Exception as e:
            logger.error(f"Error loading field config: {e}")
            raise
    
    def generate_models_from_config(self, config_path: str) -> tuple[Type, str]:
        """Generate Pydantic models from configuration file"""
        field_config = self.load_field_config(config_path)
        return self._generate_models_from_field_config(field_config)
    
    def generate_models_from_config_data(self, config_data: Dict[str, Any]) -> tuple[Type, str]:
        """Generate Pydantic models from configuration data directly"""
        field_config = config_data['extraction_config']
        
        # Check if this is text description mode
        if field_config.get('configuration_mode') == 'text_description':
            return self.generate_models_from_text_description(field_config)
        else:
            return self._generate_models_from_field_config(field_config)

    def generate_models_from_text_description(self, field_config: Dict[str, Any]) -> tuple[Type, str]:
        """Generate Pydantic models from text description configuration"""
        try:
            # Use the parsed fields from text description
            parsed_fields = field_config.get('parsed_fields', [])
            if not parsed_fields:
                raise ValueError("No parsed fields found in text description configuration")
            
            # Convert parsed fields to field configuration format
            field_config_for_generation = {
                'use_case': field_config.get('use_case', ''),
                'description': field_config.get('description', ''),
                'main_model_name': field_config.get('main_model_name', ''),
                'fields': parsed_fields
            }
            
            logger.info(f"Generating models from text description with {len(parsed_fields)} parsed fields")
            return self._generate_models_from_field_config(field_config_for_generation)
            
        except Exception as e:
            logger.error(f"Error generating models from text description: {e}")
            raise
    
    def _generate_models_from_field_config(self, field_config: Dict[str, Any]) -> tuple[Type, str]:
        
        # NEW: Preprocess fields using text description logic
        preprocessed_fields = self._preprocess_fields(field_config['fields'])
        
        # Create clean field config with preprocessed fields
        # Use same model name generation logic as Text Description method
        main_model_name = self._generate_model_name_from_config(field_config)
        
        clean_field_config = {
            'use_case': field_config.get('use_case', ''),
            'description': field_config.get('description', ''),
            'main_model_name': main_model_name,
            'fields': preprocessed_fields
        }
        
        # Store config for later use in prompt generation
        self.current_field_config = clean_field_config
        
        logger.info(f"Generating models for use case: {clean_field_config['use_case']}")
        logger.info(f"Preprocessed {len(preprocessed_fields)} fields")
        
        # Try AI generation first, but with immediate fallback on validation failure
        try:
            # Generate Pydantic model code using selected model with clean config
            if self.claude_client:
                model_code = self.claude_client.generate_pydantic_models(clean_field_config)
            else:
                model_code = self.openai_client.generate_pydantic_models(clean_field_config)
            
            # Validate the generated code before proceeding
            if self._validate_generated_code(model_code):
                logger.info("AI generation successful - code validation passed")
                # Generate extraction prompt using static template with embedded models
                self.extraction_prompt = self._create_static_extraction_prompt(clean_field_config, model_code)
                
                # Save and import the generated model
                main_model_class = self._create_model_from_code(model_code, clean_field_config['main_model_name'])
                
                return main_model_class, model_code
            else:
                raise ValueError("Generated code validation failed")
                
        except Exception as e:
            model_name = "Claude" if self.claude_client else "OpenAI"
            logger.warning(f"{model_name} model generation failed: {e}. Using enhanced fallback model generation.")
        
        # Use enhanced fallback model generation (more reliable)
        logger.info("Using enhanced fallback model generation for reliable results")
        model_code = self._create_fallback_model(clean_field_config)
        self.extraction_prompt = self._create_static_extraction_prompt(clean_field_config, model_code)
        
        main_model_class = self._create_model_from_code(model_code, clean_field_config['main_model_name'])
        
        return main_model_class, model_code
    
    def _validate_generated_code(self, model_code: str) -> bool:
        """Validate that generated code is syntactically correct"""
        try:
            # Clean the code first
            cleaned_code = self._clean_generated_code(model_code)
            
            # Try to compile the code
            compile(cleaned_code, '<string>', 'exec')
            
            # Additional validation: check for common validator issues
            if '@field_validator' in cleaned_code:
                # Check for incomplete validators
                lines = cleaned_code.split('\n')
                in_validator = False
                for line in lines:
                    if '@field_validator' in line:
                        in_validator = True
                    elif in_validator and line.strip().startswith('def '):
                        in_validator = False
                    elif in_validator and line.strip().startswith('if ') and ':' in line:
                        # Check if the if statement has a body
                        if_line_idx = lines.index(line)
                        next_line_idx = if_line_idx + 1
                        if next_line_idx < len(lines):
                            next_line = lines[next_line_idx].strip()
                            # If next line is empty or not indented, validator is incomplete
                            if not next_line or not next_line.startswith('    '):
                                logger.warning("Detected incomplete validator - missing body after if statement")
                                return False
            
            return True
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {e}")
            return False
        except Exception as e:
            logger.warning(f"Code validation failed: {e}")
            return False
    
    def _generate_with_openai(self, field_config: Dict[str, Any]) -> str:
        """Generate Pydantic models using OpenAI"""
        use_case = field_config.get('use_case', 'Document Analysis')
        description = field_config.get('description', 'Extract structured information from documents')
        
        prompt = f"""
You are an expert Python developer specializing in Pydantic models. Based on the provided field configuration, generate complete Pydantic models with proper imports, enums, and validation.

USE CASE CONTEXT: {use_case}
DESCRIPTION CONTEXT: {description}

Use the above context to understand the domain and purpose when generating models.

Field Configuration:
{json.dumps(field_config, indent=2)}

CRITICAL FIELD NAMING REQUIREMENTS:
1. Convert all field names from the configuration to snake_case format
2. Replace spaces and hyphens with underscores, convert to lowercase
3. DO NOT use field aliases - the Python field names must match the JSON keys exactly
4. This ensures the extracted JSON can be validated without field name mismatches

Requirements:
1. Generate all necessary imports at the top
2. Create enum classes ONLY for fields with enum_values (skip if enum_values is null)
3. Create the main Pydantic model class with snake_case field names
4. Use proper type hints (str, int, float, bool, List[enum], etc.)
5. Use field descriptions exactly as provided - do not modify or optimize them
6. Add field validators for list fields where needed
7. Handle both single enums and list[enum] types properly
8. Use proper enum inheritance (str, Enum)
9. Consider the use case context when structuring the models
10. NEVER use aliases - use consistent snake_case field names throughout

IMPORTANT: 
- Use field descriptions exactly as provided in the configuration
- Generate ONLY clean Python code with no markdown formatting
- All field names must be snake_case to match JSON output

Example structure:
```python
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import List, Optional

class Domain(str, Enum):
    HEALTHCARE = "Healthcare & wellbeing"
    AUTOMOTIVE = "Automotive"
    CONSTRUCTION = "Construction"
    MANUFACTURING = "Manufacturing"
    FINANCE = "Finance"

class AiField(str, Enum):
    GENERATIVE_AI = "Generative AI"
    MACHINE_LEARNING = "Machine learning"
    COMPUTER_VISION = "Computer vision & image processing"

class MainModel(BaseModel):
    company_name: str = Field(..., description="The name of the company")
    domain: Domain = Field(..., description="The primary industry domain")
    ai_field: AiField = Field(..., description="The primary AI field")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    skills: List[str] = Field(..., description="List of skills")

    @field_validator('skills')
    @classmethod
    def validate_skills(cls, v):
        if not isinstance(v, list):
            raise ValueError('skills must be a list')
        return v
```

CRITICAL VALIDATOR REQUIREMENTS:
1. ALL validators MUST include both the validation check AND the return statement
2. NEVER leave an if statement without a body - always include raise ValueError and return v
3. For list fields, always validate that the input is a list
4. Always return the validated value at the end of the validator

CRITICAL ENUM NAMING RULES:
1. Use clear, readable enum member names (HEALTHCARE, not HEALTHCAREW)
2. Keep enum names concise but descriptive
3. Use UPPER_CASE with underscores for multi-word concepts
4. Map readable names to full descriptive values
5. Never truncate or abbreviate enum member names randomly

CRITICAL ENUM EXTRACTION INSTRUCTIONS:
- When generating extraction prompts, ALWAYS specify that enum fields should return STRING VALUES
- Example: For AiMaturityLevel enum with LOW="low", MODERATE="moderate", HIGH="high"
- The extraction should return "moderate" NOT "MODERATE"
- This ensures Pydantic validation works correctly

Make sure all enum values are properly formatted and all field types are correct based on the configuration.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_selection,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert Python developer. Generate clean, well-structured Pydantic models based on the provided specifications."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=4000,
                temperature=0.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI model generation failed: {e}")
            raise
    
    def _create_model_from_code(self, model_code: str, main_model_name: str) -> Type:
        """Create Pydantic model class from generated code"""
        try:
            # Check if this is fallback-generated code (clean, no markdown)
            if self._is_fallback_generated_code(model_code):
                # Use code as-is for fallback-generated models
                cleaned_code = model_code
                logger.info("Using fallback-generated code without cleaning")
            else:
                # Clean AI-generated code
                cleaned_code = self._clean_generated_code(model_code)
                logger.info("Cleaned AI-generated code")
            
            logger.info(f"Generated model code:\n{cleaned_code}")
            
            # Create a temporary file with the generated code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(cleaned_code)
                temp_file_path = f.name
            
            # Test if the code is syntactically valid
            try:
                with open(temp_file_path, 'r') as f:
                    compile(f.read(), temp_file_path, 'exec')
            except SyntaxError as syntax_error:
                logger.error(f"Syntax error in generated code: {syntax_error}")
                logger.error(f"Generated code:\n{cleaned_code}")
                # Try to fix common issues and regenerate
                raise Exception(f"Invalid Python syntax in generated model code: {syntax_error}")
            
            # Import the module from the temporary file
            spec = importlib.util.spec_from_file_location("generated_models", temp_file_path)
            generated_module = importlib.util.module_from_spec(spec)
            sys.modules["generated_models"] = generated_module
            spec.loader.exec_module(generated_module)
            
            # Get the main model class
            main_model_class = getattr(generated_module, main_model_name)
            
            # Store the generated models
            self.generated_models[main_model_name] = main_model_class
            
            logger.info(f"Successfully created model class: {main_model_name}")
            
            # Clean up temporary file
            Path(temp_file_path).unlink()
            
            return main_model_class
            
        except Exception as e:
            logger.error(f"Error creating model from code: {e}")
            logger.error(f"Generated code:\n{model_code}")
            raise
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean the generated code to remove markdown formatting and fix common issues"""
        import re
        
        # Step 1: Remove all markdown code blocks aggressively
        # Remove ```python and ``` patterns
        code = re.sub(r'```\s*python\s*\n?', '', code, flags=re.IGNORECASE)
        code = re.sub(r'```\s*\n?', '', code)
        
        # Remove any remaining ``` patterns
        code = code.replace('```python', '').replace('```', '')
        
        # Step 2: Process line by line to remove markdown and non-Python content
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines at the beginning
            if not cleaned_lines and not line.strip():
                continue
            
            # Skip markdown code fence lines that might remain
            if line.strip() in ['```', '```python', '```Python']:
                continue
                
            # Skip lines that are clearly not Python code
            if line.strip().startswith(('Here is', 'Here\'s', 'The following', 'This is', 'Below is', 
                                       'This code', 'The code', 'Above', 'Note:', 'Example:')):
                continue
            
            # Skip lines that look like markdown or explanations
            if line.strip().startswith(('*', '-', '#')) and not any(keyword in line for keyword in ['=', 'Field', 'Enum', 'class', 'def']):
                continue
                
            # Include lines that are clearly Python code
            if (line.strip().startswith(('from ', 'import ', 'class ', 'def ', '    ', '@', 'if ', 'try:', 'except:', 'raise ', 'return ')) or
                (line.strip() and any(keyword in line for keyword in ['=', ':', 'Field', 'Enum', 'BaseModel', 'ValueError', 'isinstance'])) or
                line.strip() == ''):  # Keep empty lines for formatting
                cleaned_lines.append(line)
        
        # Step 3: Join and final cleanup
        cleaned_code = '\n'.join(cleaned_lines)
        
        # Step 4: Fix string literals
        cleaned_code = self._fix_string_literals(cleaned_code)
        
        # Step 5: Ensure proper imports are at the top
        if 'from pydantic import' not in cleaned_code:
            cleaned_code = 'from pydantic import BaseModel, Field, field_validator\n' + cleaned_code
        if 'from enum import Enum' not in cleaned_code and 'Enum' in cleaned_code:
            cleaned_code = 'from enum import Enum\n' + cleaned_code
        if 'from typing import' not in cleaned_code and ('List[' in cleaned_code or 'Optional[' in cleaned_code):
            cleaned_code = 'from typing import List, Optional\n' + cleaned_code
        
        # Step 6: Remove any stray markdown that might remain
        cleaned_code = re.sub(r'```[a-zA-Z]*\n?', '', cleaned_code)
        
        return cleaned_code
    
    def _is_fallback_generated_code(self, code: str) -> bool:
        """Check if code was generated by enhanced fallback (clean, no markdown)"""
        # Simple detection: fallback-generated code is clean and complete
        # AI-generated code often has markdown or incomplete validators
        
        # Check for markdown artifacts
        if '```' in code:
            return False  # Contains markdown
        
        # Check for explanatory text
        if any(text in code for text in ['Here is', 'The following', 'This code', 'Above', 'Below']):
            return False  # Contains explanatory text
        
        # Check for incomplete validators (AI often generates these)
        if '@field_validator' in code:
            # Look for incomplete validator pattern
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if '@field_validator' in line:
                    # Check if the next few lines contain complete validator
                    validator_complete = False
                    validator_section = '\n'.join(lines[i+1:i+10])
                    if 'raise ValueError' in validator_section and 'return v' in validator_section:
                        validator_complete = True
                    if not validator_complete:
                        return False  # Incomplete validator found
        
        return True  # Appears to be fallback-generated
    
    def _fix_string_literals(self, code: str) -> str:
        """Fix common string literal issues in generated code"""
        import re
        
        # Fix problematic description strings using repr() for proper escaping
        def fix_description(match):
            desc_content = match.group(1)
            # Remove escaping and use repr() for proper Python string literal
            fixed_content = desc_content.replace('\\"', '"').replace('\\\\', '\\')
            return f'description={repr(fixed_content)}'
        
        # Pattern to match description="content" (with potential escaping)
        pattern = r'description="([^"]*(?:\\"[^"]*)*)"'
        code = re.sub(pattern, fix_description, code)
        
        return code
    
    def _normalize_unicode_chars(self, text: str) -> str:
        """Normalize Unicode characters to ASCII-compatible equivalents"""
        if not text:
            return text
            
        # Replace common Unicode characters with ASCII equivalents
        replacements = {
            '\u2011': '-',  # Non-breaking hyphen → regular hyphen
            '\u2013': '-',  # En dash → regular hyphen
            '\u2014': '-',  # Em dash → regular hyphen
            '\u2018': "'",  # Left single quotation mark → apostrophe
            '\u2019': "'",  # Right single quotation mark → apostrophe
            '\u201C': '"',  # Left double quotation mark → quote
            '\u201D': '"',  # Right double quotation mark → quote
            '\u00A0': ' ',  # Non-breaking space → regular space
            '\u2026': '...',  # Horizontal ellipsis → three dots
        }
        
        normalized_text = text
        for unicode_char, ascii_char in replacements.items():
            normalized_text = normalized_text.replace(unicode_char, ascii_char)
            
        return normalized_text
    
    def _create_intelligent_enum_name(self, enum_value: str) -> str:
        """Create intelligent, readable enum member names"""
        
        # Special cases for common patterns
        special_mappings = {
            "Healthcare & wellbeing": "HEALTHCARE",
            "Cultural & creative industries": "CULTURAL",
            "Education & training": "EDUCATION", 
            "Environment & sustainability": "ENVIRONMENT",
            "Smart cities": "SMART_CITIES",
            "Transport, mobility, logistics": "TRANSPORT",
            "Travel & tourism": "TRAVEL",
            "Business development/business services": "BUSINESS",
            "Real estate & property": "REAL_ESTATE",
            "Arts & entertainment": "ARTS",
            "Computer vision & image processing": "COMPUTER_VISION",
            "Rule-based systems": "RULE_BASED",
            "Generative AI": "GENERATIVE_AI",
            "Machine learning": "MACHINE_LEARNING",
            "Predictive analytics": "PREDICTIVE_ANALYTICS"
        }
        
        # Check for exact matches first
        if enum_value in special_mappings:
            return special_mappings[enum_value]
        
        # General algorithm for other values
        # Extract key words (ignore common words)
        ignore_words = {'&', 'and', 'or', 'the', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an'}
        words = []
        
        # Split by various delimiters
        import re
        tokens = re.split(r'[,/&\-\s]+', enum_value)
        
        for token in tokens:
            clean_token = token.strip()
            if clean_token and clean_token.lower() not in ignore_words:
                # Include numbers and meaningful words
                if clean_token.isdigit() or len(clean_token) > 2:
                    words.append(clean_token)
                elif len(words) == 0:  # Always include first token even if short
                    words.append(clean_token)
        
        # Ensure we have at least some words
        if not words:
            # Fallback: take all non-ignored tokens
            tokens = re.split(r'[,/&\-\s]+', enum_value)
            words = [token.strip() for token in tokens if token.strip() and token.strip().lower() not in ignore_words]
        
        if not words:
            words = [enum_value.split()[0] if enum_value.split() else "VALUE"]  # Ultimate fallback
        
        # Create enum name
        enum_name = '_'.join(word.upper() for word in words)
        
        # Clean up the name
        enum_name = re.sub(r'[^A-Z0-9_]', '', enum_name)
        
        # Ensure it starts with a letter
        if enum_name and enum_name[0].isdigit():
            enum_name = 'OPTION_' + enum_name
        
        if not enum_name:
            enum_name = 'OTHER'
            
        return enum_name
    
    def _create_fallback_model(self, field_config: Dict[str, Any]) -> str:
        """Create an enhanced fallback Pydantic model with complete validators"""
        
        # Start with imports
        model_code = [
            "from pydantic import BaseModel, Field, field_validator",
            "from enum import Enum", 
            "from typing import List, Optional",
            ""
        ]
        
        # Create enum classes for fields that have enum_values (regardless of field_type)
        enum_classes = []
        for field in field_config.get('fields', []):
            if field.get('enum_values') and len(field['enum_values']) > 0:
                # Clean enum name - remove "Enum" suffix and use field name
                clean_field_name = field['field_name'].replace(' ', '_').replace('-', '_').title()
                if clean_field_name.endswith('_Enum'):
                    clean_field_name = clean_field_name[:-5]
                enum_name = f"{clean_field_name}"
                enum_class = [f"class {enum_name}(str, Enum):"]
                
                # Handle both list of individual values and comma-separated strings
                all_enum_values = []
                for enum_value in field['enum_values']:
                    # Normalize Unicode characters to ASCII-compatible equivalents
                    normalized_value = self._normalize_unicode_chars(enum_value)
                    
                    if ',' in normalized_value and len(field['enum_values']) == 1:
                        # This is a comma-separated string - split it intelligently
                        # Handle special cases where commas are part of the value name
                        parts = []
                        current_part = ""
                        
                        # Split by comma but handle special cases
                        tokens = normalized_value.split(',')
                        i = 0
                        while i < len(tokens):
                            token = tokens[i].strip()
                            
                            # Check for multi-word values that should stay together
                            if (token.lower() in ['transport'] and i + 2 < len(tokens) and 
                                tokens[i+1].strip().lower() == 'mobility' and 
                                tokens[i+2].strip().lower() == 'logistics'):
                                # Combine "Transport, mobility, logistics" into one value
                                parts.append(f"{token}, {tokens[i+1].strip()}, {tokens[i+2].strip()}")
                                i += 3
                            elif (token.lower().startswith('business development') and i + 1 < len(tokens)):
                                # Combine "Business development/business services"
                                parts.append(f"{token}, {tokens[i+1].strip()}")
                                i += 2
                            else:
                                parts.append(token)
                                i += 1
                        
                        all_enum_values.extend([part for part in parts if part])
                    else:
                        # This is already a proper individual value
                        all_enum_values.append(normalized_value)
                
                # Generate enum entries with intelligent naming
                for enum_value in all_enum_values:
                    # Create intelligent enum member names
                    safe_name = self._create_intelligent_enum_name(enum_value)
                    enum_class.append(f'    {safe_name} = "{enum_value}"')
                
                enum_classes.extend(enum_class)
                enum_classes.append("")
        
        model_code.extend(enum_classes)
        
        # Create main model class
        main_model_name = field_config.get('main_model_name', 'GeneratedModel')
        model_code.append(f"class {main_model_name}(BaseModel):")
        
        # Generate field definitions
        for field in field_config.get('fields', []):
            original_field_name = field.get('field_name', '')
            # Always use snake_case for field names to avoid alias issues
            field_name = original_field_name.replace(' ', '_').replace('-', '_').lower()
            field_type = field.get('field_type', 'str')
            description = self._normalize_unicode_chars(field.get('description', ''))
            required = field.get('required', True)
            
            # Map field types - auto-convert to enum if categories are provided
            if field.get('enum_values') and len(field['enum_values']) > 0:
                # Use same naming convention as enum class generation
                clean_field_name = field['field_name'].replace(' ', '_').replace('-', '_').title()
                if clean_field_name.endswith('_Enum'):
                    clean_field_name = clean_field_name[:-5]
                enum_name = f"{clean_field_name}"
                if field_type == 'list[enum]' or (field_type == 'list[str]' and field.get('enum_values')):
                    type_hint = f"List[{enum_name}]"
                else:
                    type_hint = enum_name
            elif field_type == 'list[str]':
                type_hint = "List[str]"
            elif field_type in ['int', 'float', 'bool']:
                type_hint = field_type
            else:
                type_hint = "str"
            
            if not required:
                type_hint = f"Optional[{type_hint}]"
            
            default_value = "..." if required else "None"
            
            # Use repr() to properly escape the description string
            description_repr = repr(description)
            
            # Generate field WITHOUT alias - use consistent snake_case names
            # This ensures extracted JSON keys match the model field names exactly
            model_code.append(f'    {field_name}: {type_hint} = Field({default_value}, description={description_repr})')
        
        # Generate validators for ALL list fields (both list[str] and list[enum])
        validators = []
        for field in field_config.get('fields', []):
            field_name = field.get('field_name', '').replace(' ', '_').replace('-', '_').lower()
            field_type = field.get('field_type', 'str')
            
            # Check if this is a list field (either explicitly or implicitly through enum_values)
            is_list_field = (field_type in ['list[str]', 'list[enum]'] or 
                           (field_type == 'str' and field.get('enum_values') and len(field.get('enum_values', [])) > 0))
            
            if is_list_field:
                validator = self._generate_list_validator(field_name)
                validators.extend(validator)
        
        # Add validators to the model
        if validators:
            model_code.append("")  # Add blank line before validators
            model_code.extend(validators)
        
        return '\n'.join(model_code)
    
    def _generate_list_validator(self, field_name: str) -> List[str]:
        """Generate complete validator for list fields"""
        return [
            f"    @field_validator('{field_name}')",
            f"    @classmethod",
            f"    def validate_{field_name}(cls, v):",
            f"        if not isinstance(v, list):",
            f"            raise ValueError('{field_name} must be a list')",
            f"        return v",
            ""
        ]
    
    def _create_fallback_prompt(self, field_config: Dict[str, Any], model_code: str) -> str:
        """Create a fallback extraction prompt that embeds the model code when Claude generation fails"""
        
        use_case = field_config.get('use_case', 'Document Analysis')
        description = field_config.get('description', 'Extract structured information from documents')
        additional_instructions = field_config.get('additional_instructions', '')
        main_model_name = field_config.get('main_model_name', 'GeneratedModel')
        
        # Extract field specifications from the config
        field_specs = []
        for field in field_config.get('fields', []):
            field_name = field.get('field_name', '')
            field_desc = field.get('description', '')
            field_type = field.get('field_type', 'str')
            required = field.get('required', True)
            enum_values = field.get('enum_values', [])
            
            spec = f"- {field_name} ({field_type}): {field_desc}"
            if enum_values:
                spec += f" [Options: {', '.join(enum_values)}]"
            if not required:
                spec += " [Optional]"
            field_specs.append(spec)
        
        prompt_parts = [
            f"TASK: {use_case}",
            "",
            f"CONTEXT: {description}",
            "",
            "EMBEDDED PYDANTIC MODEL:",
            "```python",
            model_code,
            "```",
            "",
            "FIELDS TO EXTRACT:",
        ]
        
        prompt_parts.extend(field_specs)
        
        prompt_parts.extend([
            "",
            "EXTRACTION RULES:",
            "- Only extract information explicitly stated in the text",
            "- Never fabricate or infer information",
            "- Use 'n/a' for missing information",
            "- Maintain accuracy over completeness",
            "- Follow the embedded model field descriptions for guidance",
        ])
        
        if additional_instructions:
            prompt_parts.extend([
                "",
                "ADDITIONAL INSTRUCTIONS:",
                additional_instructions
            ])
        
        prompt_parts.extend([
            "",
            f"OUTPUT FORMAT:",
            f"Return a JSON object matching the {main_model_name} structure shown above.",
            "All field names must match the model exactly.",
            "Use appropriate data types as defined in the model.",
            "",
            "VALIDATION:",
            "- Verify all information comes from source text",
            "- Check that no data is fabricated",
            "- Ensure JSON structure matches the embedded model"
        ])
        
        return '\n'.join(prompt_parts)
    
    def _create_static_extraction_prompt(self, field_config: Dict[str, Any], model_code: str) -> str:
        """Create a comprehensive static extraction prompt template with embedded models"""
        
        use_case = field_config.get('use_case', 'Document Analysis')
        description = field_config.get('description', 'Extract structured information from documents')
        main_model_name = field_config.get('main_model_name', 'GeneratedModel')
        
        prompt_template = f"""TASK: {use_case}

EXTRACTION TASK:
{description}

EMBEDDED PYDANTIC MODELS:
```python
{model_code}
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
   - For enums: Choose ONLY from the specified options AND return the STRING VALUE, not the enum member name
   - For the extracted fields whose length exceeds 30 words, summarize their text into less than 30 words comprising very brief key phrases separated by semi-colon.

3. FIELD VALIDATION:
   - Enum fields must match one of the specified values exactly
   - For enum fields: Return the STRING VALUE (e.g., "moderate") NOT the enum member name (e.g., "MODERATE")
   - List fields should contain valid, non-empty items
   - Numerical fields should be valid numbers in correct format

4. ENUM FIELD HANDLING (CRITICAL):
   - For enum fields, ALWAYS return the string value, NEVER the enum member name
   - Example: For AiMaturityLevel enum, return "moderate" NOT "MODERATE"
   - Example: For CompanyType enum, return "startup" NOT "STARTUP"
   - Example: For AiField enum, return "machine learning" NOT "MACHINE_LEARNING"
   - The JSON output must contain the actual string values that match the enum definitions

5. QUALITY ASSURANCE:
   - Double-check all extracted information against source
   - Ensure no information is duplicated across fields
   - Verify that field types match the expected data types
   - Confirm that all required fields are addressed
   - Validate that enum selections are from available options
   - CRITICAL: Verify enum fields return string values, not member names


OUTPUT FORMAT:
Return the extracted information as a JSON object that exactly matches the {main_model_name} structure shown above.

CRITICAL: All JSON field names MUST be in snake_case format (lowercase with underscores) to match the model exactly.
For example: "due_date", "bank_name", "car_type" - NOT "Due date", "bank name", "car type"

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
- All enum fields contain valid options only AND return STRING VALUES (not member names)
- All numerical fields contain valid numbers
- All date fields follow proper format
- No information is fabricated or inferred
- JSON structure matches the model exactly
- Field names match the model exactly (snake_case)
- Data types are appropriate for each field
- CRITICAL: Enum fields return values like "moderate", "startup", "machine learning" NOT "MODERATE", "STARTUP", "MACHINE_LEARNING"

FINAL INSTRUCTIONS:
- Process the document systematically
- Extract information field by field as specified above
- Maintain accuracy over completeness
- When in doubt, use 'n/a' rather than guessing
- Return only the JSON object with extracted data
- Ensure the output can be parsed as valid JSON"""

        return prompt_template
    
    def get_extraction_prompt(self) -> str:
        """Get the generated extraction prompt"""
        return self.extraction_prompt
    
    def get_generated_models(self) -> Dict[str, Type]:
        """Get all generated model classes"""
        return self.generated_models.copy()
    
    def save_generated_models(self, models_path: str, model_code: str):
        """Save generated models to a Python file (models only)"""
        try:
            # Clean the model code before saving to remove markdown syntax
            cleaned_code = self._clean_generated_code(model_code)
            
            with open(models_path, 'w', encoding='utf-8') as f:
                f.write("# Auto-generated Pydantic models\n")
                f.write("# Generated by Knowledge Extraction Agent\n\n")
                f.write(cleaned_code)
            
            logger.info(f"Generated models saved to: {models_path}")
            
        except Exception as e:
            logger.error(f"Error saving models to file: {e}")
            raise
    
    def save_extraction_prompt(self, prompt_path: str):
        """Save extraction prompt as reusable code with import statement"""
        try:
            # Change extension to .py for code format
            if prompt_path.endswith('.txt'):
                prompt_path = prompt_path[:-4] + '_prompt.py'
            
            # No need for import references since models are embedded in the prompt
            
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write('"""\n')
                f.write('Auto-generated Extraction Prompt with Embedded Models\n')
                f.write('Generated by Knowledge Extraction Agent\n')
                f.write('This file contains a complete, self-contained extraction prompt system.\n')
                f.write('"""\n\n')
                f.write('EXTRACTION_PROMPT = """\n')
                f.write(self.extraction_prompt.strip())
                f.write('\n"""\n\n')
                f.write('def get_extraction_prompt():\n')
                f.write('    """Return the complete extraction prompt with embedded models"""\n')
                f.write('    return EXTRACTION_PROMPT\n')
            
            logger.info(f"Extraction prompt saved as reusable code to: {prompt_path}")
            
        except Exception as e:
            logger.error(f"Error saving extraction prompt to file: {e}")
            raise
    
    def load_models_and_prompt(self, model_file_path: str, prompt_file_path: str) -> tuple[Type, str]:
        """Load models and extraction prompt from separate files"""
        try:
            # Load extraction prompt from Python file
            extraction_prompt = ""
            if os.path.exists(prompt_file_path):
                with open(prompt_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract the prompt from the Python file
                    import re
                    match = re.search(r'EXTRACTION_PROMPT = """(.*?)"""', content, re.DOTALL)
                    if match:
                        extraction_prompt = match.group(1).strip()
                    else:
                        # Fallback for text files (old format)
                        lines = content.split('\n')
                        prompt_lines = []
                        skip_header = True
                        for line in lines:
                            if skip_header and not line.startswith('#') and line.strip():
                                skip_header = False
                            if not skip_header:
                                prompt_lines.append(line)
                        extraction_prompt = '\n'.join(prompt_lines).strip()
            
            # Import the module to get the model class
            import importlib.util
            import sys
            
            spec = importlib.util.spec_from_file_location("loaded_models", model_file_path)
            loaded_module = importlib.util.module_from_spec(spec)
            sys.modules["loaded_models"] = loaded_module
            spec.loader.exec_module(loaded_module)
            
            # Find the main model class (usually the last class defined)
            model_classes = [getattr(loaded_module, name) for name in dir(loaded_module) 
                           if isinstance(getattr(loaded_module, name), type) and 
                           issubclass(getattr(loaded_module, name), BaseModel) and
                           getattr(loaded_module, name) != BaseModel]
            
            if model_classes:
                main_model = model_classes[-1]  # Assume the last one is the main model
                self.extraction_prompt = extraction_prompt
                return main_model, extraction_prompt
            else:
                raise Exception("No valid Pydantic model found in file")
                
        except Exception as e:
            logger.error(f"Error loading models from files: {e}")
            raise
    
    def load_prompt_from_file(self, prompt_file_path: str) -> str:
        """Load extraction prompt from text file"""
        try:
            with open(prompt_file_path, 'r') as f:
                content = f.read()
                # Remove the header comments to get just the prompt
                lines = content.split('\n')
                prompt_lines = []
                skip_header = True
                for line in lines:
                    if skip_header and not line.startswith('#') and line.strip():
                        skip_header = False
                    if not skip_header:
                        prompt_lines.append(line)
                return '\n'.join(prompt_lines).strip()
        except Exception as e:
            logger.error(f"Error loading prompt from file: {e}")
            raise
    
    def _preprocess_fields(self, fields: list) -> list:
        """Apply text description preprocessing to field-by-field configurations"""
        preprocessed_fields = []
        field_names = set()
        
        for field in fields:
            # Apply snake_case conversion
            field['field_name'] = self._to_snake_case(field.get('field_name', ''))
            
            # Standardize field type
            field['field_type'] = self._determine_field_type(field)
            
            # Ensure unique field names
            original_name = field['field_name']
            counter = 1
            while field['field_name'] in field_names:
                field['field_name'] = f"{original_name}_{counter}"
                counter += 1
            field_names.add(field['field_name'])
            
            # Validate and clean field
            field = self._validate_field(field)
            preprocessed_fields.append(field)
        
        return preprocessed_fields
    
    def _generate_model_name_from_config(self, field_config: Dict[str, Any]) -> str:
        """Generate model name using same logic as Text Description method"""
        # First try to use the provided model name, but convert to PascalCase
        provided_name = field_config.get('main_model_name', '')
        if provided_name:
            # Convert camelCase or snake_case to PascalCase
            # Handle camelCase: "trialInfo" -> "TrialInfo"
            if provided_name[0].islower():
                # Find the first uppercase letter and split there
                for i, char in enumerate(provided_name[1:], 1):
                    if char.isupper():
                        # Split at the uppercase letter
                        first_word = provided_name[:i].capitalize()
                        rest = provided_name[i:]
                        provided_name = first_word + rest
                        break
                else:
                    # No uppercase found, just capitalize first letter
                    provided_name = provided_name.capitalize()
            
            # Handle snake_case: "trial_info" -> "TrialInfo"
            if '_' in provided_name:
                provided_name = ''.join(word.capitalize() for word in provided_name.split('_'))
            return provided_name
        
        # If no model name provided, generate from use case (same as Text Description)
        use_case = field_config.get('use_case', '')
        if not use_case:
            return "ExtractedData"
        
        # Clean use case name and convert to PascalCase
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', use_case)
        words = cleaned.split()
        
        # Convert to PascalCase
        model_name = ''.join(word.capitalize() for word in words)
        
        # Add suffix if needed
        if not model_name.endswith(('Info', 'Data', 'Model')):
            model_name += 'Info'
        
        return model_name
    
    def _to_snake_case(self, field_name: str) -> str:
        """Convert field name to snake_case (copied from TextDescriptionParser)"""
        if not field_name:
            return ""
        
        # Remove special characters and replace with underscores
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', field_name)
        
        # Convert to snake_case - handle special cases first
        if cleaned == 'AI Field':
            return 'ai_field'
        
        # Convert to snake_case
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', cleaned).lower()
        snake_case = re.sub(r'[_\s]+', '_', snake_case)
        
        return snake_case.strip('_')
    
    def _determine_field_type(self, field: dict) -> str:
        """Determine field type (copied from TextDescriptionParser)"""
        field_type = field.get('field_type', 'str').lower()
        
        # Map common type variations
        type_mapping = {
            'string': 'str', 'text': 'str',
            'number': 'int', 'integer': 'int',
            'decimal': 'float',
            'boolean': 'bool', 'true/false': 'bool',
            'array': 'list', 'list': 'list',
            'category': 'enum', 'choice': 'enum', 'option': 'enum'
        }
        
        mapped_type = type_mapping.get(field_type, field_type)
        
        # Handle enum types
        if 'enum' in mapped_type or field.get('enum_values'):
            if 'list' in mapped_type:
                return 'list[enum]'
            return 'enum'
        
        # Handle list types
        if 'list' in mapped_type:
            return 'list[str]'
        
        return mapped_type
    
    def _validate_field(self, field: dict) -> dict:
        """Validate and clean field (adapted from TextDescriptionParser)"""
        # Ensure description is not empty
        if not field.get('description', '').strip():
            field['description'] = f"Extract {field['field_name'].replace('_', ' ')}"
        
        # Validate field type
        valid_types = ['str', 'int', 'float', 'bool', 'list[str]', 'enum', 'list[enum]']
        if field['field_type'] not in valid_types:
            field['field_type'] = 'str'  # Default fallback
        
        # Ensure enum values for enum types
        if field['field_type'] in ['enum', 'list[enum]'] and not field.get('enum_values'):
            field['field_type'] = 'str'  # Fallback to string if no enum values
        
        return field