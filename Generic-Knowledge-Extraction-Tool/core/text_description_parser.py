import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from core.text_description_client import TextDescriptionClient, OpenAITextDescriptionClient, ClaudeTextDescriptionClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParsedField(BaseModel):
    """Represents a parsed field from text description"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Text Description Parser
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

    field_name: str = Field(..., description="Field name in snake_case")
    field_type: str = Field(..., description="Data type (str, int, float, bool, list, enum)")
    description: str = Field(..., description="Field description")
    required: bool = Field(default=True, description="Whether field is required")
    enum_values: Optional[List[str]] = Field(default=None, description="Enum values if field_type is enum")

class TextDescriptionParser:
    """Parse textual descriptions and extract field information using AI"""

    def __init__(self, model_selection: str = 'claude-sonnet-4-20250514', api_config: Optional[Dict[str, Any]] = None):
        """Initialize the parser with AI client"""
        self.model_selection = model_selection
        self.api_config = api_config or {}
        
        # Initialize appropriate client
        if 'claude' in model_selection.lower():
            self.client = ClaudeTextDescriptionClient()
        else:
            self.client = OpenAITextDescriptionClient(api_config=api_config)

    def parse_extraction_description(self, description: str, use_case: str = "", context: str = "") -> Dict[str, Any]:
        """
        Parse natural language description and extract field configuration
        
        Args:
            description: Natural language description of extraction requirements
            use_case: Use case name for context
            context: Additional context about the extraction
            
        Returns:
            Dictionary containing parsed field configuration
        """
        try:
            logger.info(f"Parsing text description for use case: {use_case}")
            
            # Validate input
            if not description or not description.strip():
                raise ValueError("Description cannot be empty")
            
            # Clean and prepare description
            cleaned_description = self._clean_description(description)
            
            # Send to AI for parsing
            parsed_response = self.client.parse_description(
                description=cleaned_description,
                use_case=use_case,
                context=context
            )
            
            # Extract and validate fields
            parsed_fields = self._extract_field_metadata(parsed_response)
            
            # Validate parsed fields
            validated_fields = self._validate_parsed_fields(parsed_fields)
            
            # Convert to configuration format
            config = self._convert_to_config_format(validated_fields, use_case, description)
            
            logger.info(f"Successfully parsed {len(validated_fields)} fields from description")
            return config
            
        except Exception as e:
            logger.error(f"Error parsing description: {e}")
            raise

    def _clean_description(self, description: str) -> str:
        """Clean and normalize the input description"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', description.strip())
        
        # Ensure it ends with proper punctuation
        if not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
            
        return cleaned

    def _extract_field_metadata(self, ai_response: Dict[str, Any]) -> List[ParsedField]:
        """Extract field metadata from AI response"""
        try:
            fields_data = ai_response.get('fields', [])
            if not fields_data:
                raise ValueError("No fields found in AI response")
            
            parsed_fields = []
            for field_data in fields_data:
                # Convert field name to snake_case
                field_name = self._to_snake_case(field_data.get('field_name', ''))
                
                # Determine field type
                field_type = self._determine_field_type(field_data)
                
                # Extract enum values if applicable
                enum_values = None
                if field_type in ['enum', 'list[enum]']:
                    enum_values = field_data.get('enum_values', [])
                
                parsed_field = ParsedField(
                    field_name=field_name,
                    field_type=field_type,
                    description=field_data.get('description', ''),
                    required=field_data.get('required', True),
                    enum_values=enum_values
                )
                parsed_fields.append(parsed_field)
            
            return parsed_fields
            
        except Exception as e:
            logger.error(f"Error extracting field metadata: {e}")
            raise

    def _to_snake_case(self, field_name: str) -> str:
        """Convert field name to snake_case"""
        if not field_name:
            return ""
        
        # Remove special characters and replace with underscores
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', field_name)
        
        # Convert to snake_case
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', cleaned).lower()
        snake_case = re.sub(r'[_\s]+', '_', snake_case)
        
        return snake_case.strip('_')

    def _determine_field_type(self, field_data: Dict[str, Any]) -> str:
        """Determine the appropriate field type based on field data"""
        field_type = field_data.get('field_type', 'str').lower()
        
        # Map common type variations
        type_mapping = {
            'string': 'str',
            'text': 'str',
            'number': 'int',
            'integer': 'int',
            'decimal': 'float',
            'boolean': 'bool',
            'true/false': 'bool',
            'array': 'list',
            'list': 'list',
            'category': 'enum',
            'choice': 'enum',
            'option': 'enum'
        }
        
        mapped_type = type_mapping.get(field_type, field_type)
        
        # Handle enum types
        if 'enum' in mapped_type or field_data.get('enum_values'):
            if 'list' in mapped_type:
                return 'list[enum]'
            return 'enum'
        
        # Handle list types
        if 'list' in mapped_type:
            return 'list[str]'
        
        return mapped_type

    def _validate_parsed_fields(self, fields: List[ParsedField]) -> List[ParsedField]:
        """Validate parsed fields and fix common issues"""
        validated_fields = []
        field_names = set()
        
        for field in fields:
            # Ensure field name is unique
            original_name = field.field_name
            counter = 1
            while field.field_name in field_names:
                field.field_name = f"{original_name}_{counter}"
                counter += 1
            
            field_names.add(field.field_name)
            
            # Ensure description is not empty
            if not field.description.strip():
                field.description = f"Extract {field.field_name.replace('_', ' ')}"
            
            # Validate field type
            valid_types = ['str', 'int', 'float', 'bool', 'list[str]', 'enum', 'list[enum]']
            if field.field_type not in valid_types:
                field.field_type = 'str'  # Default fallback
            
            # Ensure enum values for enum types
            if field.field_type in ['enum', 'list[enum]'] and not field.enum_values:
                field.field_type = 'str'  # Fallback to string if no enum values
            
            validated_fields.append(field)
        
        return validated_fields

    def _convert_to_config_format(self, fields: List[ParsedField], use_case: str, description: str) -> Dict[str, Any]:
        """Convert parsed fields to configuration format matching existing structure"""
        fields_config = []
        
        for field in fields:
            field_config = {
                'field_name': field.field_name,
                'field_type': field.field_type,
                'description': field.description,
                'required': field.required,
                'enum_values': field.enum_values
            }
            fields_config.append(field_config)
        
        # Generate model name from use case
        model_name = self._generate_model_name(use_case)
        
        config = {
            'extraction_config': {
                'configuration_mode': 'text_description',
                'use_case': use_case,
                'description': description,
                'main_model_name': model_name,
                'text_description': description,
                'parsed_fields': fields_config,
                'created_at': self._get_current_timestamp(),
                'fields': fields_config  # For backward compatibility
            }
        }
        
        return config

    def _generate_model_name(self, use_case: str) -> str:
        """Generate model name from use case"""
        if not use_case:
            return "ExtractedData"
        
        # Clean use case name
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', use_case)
        words = cleaned.split()
        
        # Convert to PascalCase
        model_name = ''.join(word.capitalize() for word in words)
        
        # Add suffix if needed
        if not model_name.endswith(('Info', 'Data', 'Model')):
            model_name += 'Info'
        
        return model_name

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()

    def validate_description(self, description: str) -> Tuple[bool, List[str]]:
        """Validate text description before parsing"""
        errors = []
        
        if not description or not description.strip():
            errors.append("Description cannot be empty")
        
        if len(description.strip()) < 20:
            errors.append("Description should be at least 20 characters long")
        
        if len(description.strip()) > 2000:
            errors.append("Description should be less than 2000 characters")
        
        # Check for basic extraction keywords
        extraction_keywords = ['extract', 'get', 'find', 'identify', 'capture', 'retrieve', 'obtain']
        if not any(keyword in description.lower() for keyword in extraction_keywords):
            errors.append("Description should mention what to extract (e.g., 'extract', 'get', 'find')")
        
        return len(errors) == 0, errors

    def get_parsing_suggestions(self, description: str) -> List[str]:
        """Get suggestions for improving the description"""
        suggestions = []
        
        if len(description.strip()) < 50:
            suggestions.append("Consider providing more detail about what specific information to extract")
        
        if 'field' not in description.lower() and 'information' not in description.lower():
            suggestions.append("Mention specific fields or types of information to extract")
        
        if 'document' not in description.lower() and 'file' not in description.lower():
            suggestions.append("Specify the type of documents to process")
        
        return suggestions
