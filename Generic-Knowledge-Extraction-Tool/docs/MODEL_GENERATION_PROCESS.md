# Model Generation Process Documentation

## Overview

The Knowledge Extraction Agent supports two methods for generating Pydantic models for structured data extraction:

1. **Text Description Method** - Generate models from natural language descriptions
2. **Field-by-Field Configuration Method** - Generate models from structured field definitions

Both methods use identical underlying logic to ensure consistent, reliable model generation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Generation Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│  Input Processing → Field Preprocessing → Model Generation →     │
│  Code Validation → Fallback Generation → Model Creation          │
└─────────────────────────────────────────────────────────────────┘
```

## Method 1: Text Description

### Process Flow

```
Text Description → Parse Fields → Generate Model Name → Create Pydantic Model
```

### Key Components

#### 1. Text Description Parser (`text_description_parser.py`)

**Main Class:** `TextDescriptionParser`

**Core Method:**
```python
def parse_description(self, description: str, use_case: str = "") -> Dict[str, Any]:
    """Parse text description into structured field configuration"""
    
    # Step 1: Generate model name from use case
    model_name = self._generate_model_name(use_case)
    
    # Step 2: Parse description using AI to extract fields
    parsed_fields = self._parse_fields_with_ai(description)
    
    # Step 3: Return structured configuration
    return {
        'use_case': use_case,
        'description': description,
        'main_model_name': model_name,
        'parsed_fields': parsed_fields,
        'timestamp': self._get_current_timestamp()
    }
```

**Model Name Generation:**
```python
def _generate_model_name(self, use_case: str) -> str:
    """Generate PascalCase model name from use case"""
    if not use_case:
        return "ExtractedData"
    
    # Clean and convert to PascalCase
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', use_case)
    words = cleaned.split()
    model_name = ''.join(word.capitalize() for word in words)
    
    # Add suffix if needed
    if not model_name.endswith(('Info', 'Data', 'Model')):
        model_name += 'Info'
    
    return model_name
```

**Field Parsing with AI:**
```python
def _parse_fields_with_ai(self, description: str) -> List[Dict[str, Any]]:
    """Use AI to parse description and extract field definitions"""
    
    prompt = f"""
    Parse the following description and extract structured fields:
    
    Description: {description}
    
    Return a JSON array of field objects with:
    - field_name: descriptive name
    - field_type: string, number, boolean, array, category
    - description: what this field represents
    - enum_values: [list] if field_type is category
    """
    
    # Call AI service and parse response
    response = self.ai_client.generate_response(prompt)
    return self._parse_ai_response(response)
```

#### 2. Model Generator Integration

**Entry Point:**
```python
def generate_models_from_text_description(self, field_config: Dict[str, Any]) -> tuple[Type, str]:
    """Generate Pydantic models from text description configuration"""
    
    # Extract parsed fields from text description
    parsed_fields = field_config.get('parsed_fields', [])
    if not parsed_fields:
        raise ValueError("No parsed fields found in text description configuration")
    
    # Convert to field configuration format
    field_config_for_generation = {
        'use_case': field_config.get('use_case', ''),
        'description': field_config.get('description', ''),
        'main_model_name': field_config.get('main_model_name', ''),
        'fields': parsed_fields
    }
    
    # Use shared model generation logic
    return self._generate_models_from_field_config(field_config_for_generation)
```

## Method 2: Field-by-Field Configuration

### Process Flow

```
Field Definitions → Preprocess Fields → Generate Model Name → Create Pydantic Model
```

### Key Components

#### 1. Field Configuration Structure

**Input Format:**
```json
{
    "use_case": "Resume Extraction",
    "description": "Extract information from resumes",
    "main_model_name": "trialInfo",
    "fields": [
        {
            "field_name": "Company Name",
            "field_type": "string",
            "description": "Name of the company"
        },
        {
            "field_name": "Skills List",
            "field_type": "array",
            "description": "List of technical skills"
        },
        {
            "field_name": "Industry",
            "field_type": "category",
            "description": "Industry type",
            "enum_values": ["Tech", "Finance", "Healthcare"]
        }
    ]
}
```

#### 2. Field Preprocessing

**Core Method:**
```python
def _preprocess_fields(self, fields: list) -> list:
    """Preprocess fields using same logic as Text Description method"""
    preprocessed_fields = []
    
    for field in fields:
        # Convert field name to snake_case
        field['field_name'] = self._to_snake_case(field['field_name'])
        
        # Standardize field type
        field['field_type'] = self._determine_field_type(field)
        
        # Validate field structure
        field = self._validate_field(field)
        
        preprocessed_fields.append(field)
    
    return preprocessed_fields
```

**Snake Case Conversion:**
```python
def _to_snake_case(self, field_name: str) -> str:
    """Convert field name to snake_case"""
    if not field_name:
        return ""
    
    # Remove special characters and replace with underscores
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', field_name)
    
    # Split on spaces and join with underscores
    words = cleaned.split()
    snake_case = '_'.join(word.lower() for word in words)
    
    # Special case handling
    if snake_case == 'a_i_field':
        snake_case = 'ai_field'
    
    return snake_case
```

**Field Type Standardization:**
```python
def _determine_field_type(self, field: dict) -> str:
    """Standardize field types to Python/Pydantic types"""
    field_type = field.get('field_type', 'string').lower()
    
    type_mapping = {
        'string': 'str',
        'text': 'str',
        'number': 'int',
        'integer': 'int',
        'decimal': 'float',
        'float': 'float',
        'boolean': 'bool',
        'bool': 'bool',
        'array': 'list[str]',
        'list': 'list[str]',
        'category': 'enum'
    }
    
    return type_mapping.get(field_type, 'str')
```

**Field Validation:**
```python
def _validate_field(self, field: dict) -> dict:
    """Validate and clean field definition"""
    # Ensure description is not empty
    if not field.get('description', '').strip():
        field['description'] = f"Field: {field['field_name']}"
    
    # Validate field type
    valid_types = ['str', 'int', 'float', 'bool', 'list[str]', 'enum']
    if field['field_type'] not in valid_types:
        field['field_type'] = 'str'
    
    # Ensure enum values for category fields
    if field['field_type'] == 'enum':
        if 'enum_values' not in field or not field['enum_values']:
            field['enum_values'] = ['Option1', 'Option2']
    
    return field
```

#### 3. Model Name Generation

**Core Method:**
```python
def _generate_model_name_from_config(self, field_config: Dict[str, Any]) -> str:
    """Generate model name using same logic as Text Description method"""
    
    # Use provided model name, convert to PascalCase
    provided_name = field_config.get('main_model_name', '')
    if provided_name:
        # Convert camelCase: "trialInfo" -> "TrialInfo"
        if provided_name[0].islower():
            for i, char in enumerate(provided_name[1:], 1):
                if char.isupper():
                    first_word = provided_name[:i].capitalize()
                    rest = provided_name[i:]
                    provided_name = first_word + rest
                    break
            else:
                provided_name = provided_name.capitalize()
        
        # Handle snake_case: "trial_info" -> "TrialInfo"
        if '_' in provided_name:
            provided_name = ''.join(word.capitalize() for word in provided_name.split('_'))
        return provided_name
    
    # Generate from use case (same as Text Description)
    use_case = field_config.get('use_case', '')
    if not use_case:
        return "ExtractedData"
    
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', use_case)
    words = cleaned.split()
    model_name = ''.join(word.capitalize() for word in words)
    
    if not model_name.endswith(('Info', 'Data', 'Model')):
        model_name += 'Info'
    
    return model_name
```

## Shared Model Generation Logic

### Core Generation Method

Both methods converge on the same `_generate_models_from_field_config()` method:

```python
def _generate_models_from_field_config(self, field_config: Dict[str, Any]) -> tuple[Type, str]:
    """Generate Pydantic models from field configuration"""
    
    # Preprocess fields using text description logic
    preprocessed_fields = self._preprocess_fields(field_config['fields'])
    
    # Generate PascalCase model name
    main_model_name = self._generate_model_name_from_config(field_config)
    
    # Create clean field config
    clean_field_config = {
        'use_case': field_config.get('use_case', ''),
        'description': field_config.get('description', ''),
        'main_model_name': main_model_name,
        'fields': preprocessed_fields
    }
    
    # Try AI generation first
    try:
        if self.claude_client:
            model_code = self.claude_client.generate_pydantic_models(clean_field_config)
        else:
            model_code = self.openai_client.generate_pydantic_models(clean_field_config)
        
        # Validate generated code
        if self._validate_generated_code(model_code):
            main_model_class = self._create_model_from_code(model_code, main_model_name)
            return main_model_class, model_code
        else:
            raise ValueError("Generated code validation failed")
    
    except Exception as e:
        logger.warning(f"AI model generation failed: {e}. Using enhanced fallback model generation.")
        
        # Use enhanced fallback generation
        model_code = self._create_fallback_model(clean_field_config)
        main_model_class = self._create_model_from_code(model_code, main_model_name)
        return main_model_class, model_code
```

### AI Model Generation

**Claude Client:**
```python
def generate_pydantic_models(self, field_config: Dict[str, Any]) -> str:
    """Generate Pydantic model code using Claude"""
    
    prompt = self._build_model_generation_prompt(field_config)
    
    response = self.client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text
```

**Prompt Construction:**
```python
def _build_model_generation_prompt(self, field_config: Dict[str, Any]) -> str:
    """Build comprehensive prompt for model generation"""
    
    main_model_name = field_config.get('main_model_name', 'GeneratedModel')
    fields = field_config.get('fields', [])
    
    # Build field definitions
    field_definitions = []
    for field in fields:
        field_def = f"    {field['field_name']}: {field['field_type']} = Field(..., description='{field['description']}')"
        field_definitions.append(field_def)
    
    # Build enum classes
    enum_classes = []
    for field in fields:
        if field['field_type'] == 'enum':
            enum_name = field['field_name'].title().replace('_', '')
            enum_values = field.get('enum_values', [])
            enum_class = f"class {enum_name}(str, Enum):\n"
            for value in enum_values:
                enum_class += f"    {value.upper()} = \"{value}\"\n"
            enum_classes.append(enum_class)
    
    prompt = f"""
    Generate a complete Pydantic model with the following structure:
    
    Model Name: {main_model_name}
    
    Fields:
    {chr(10).join(field_definitions)}
    
    Requirements:
    1. Include all necessary imports
    2. Create enum classes for category fields
    3. Add complete field validators for list fields
    4. Ensure proper Python syntax
    5. Include error handling in validators
    
    Return only the Python code, no markdown formatting.
    """
    
    return prompt
```

### Enhanced Fallback Generation

**Core Method:**
```python
def _create_fallback_model(self, field_config: Dict[str, Any]) -> str:
    """Create fallback model with complete validators"""
    
    main_model_name = field_config.get('main_model_name', 'GeneratedModel')
    fields = field_config.get('fields', [])
    
    # Build imports
    imports = [
        "from pydantic import BaseModel, Field, field_validator",
        "from enum import Enum",
        "from typing import List, Optional"
    ]
    
    # Build enum classes
    enum_classes = []
    enum_imports = []
    for field in fields:
        if field['field_type'] == 'enum':
            enum_name = field['field_name'].title().replace('_', '')
            enum_values = field.get('enum_values', [])
            
            enum_class = f"class {enum_name}(str, Enum):\n"
            for value in enum_values:
                enum_class += f"    {value.upper()} = \"{value}\"\n"
            enum_classes.append(enum_class)
    
    # Build field definitions
    field_definitions = []
    for field in fields:
        field_type = field['field_type']
        if field_type == 'enum':
            enum_name = field['field_name'].title().replace('_', '')
            field_type = enum_name
        
        field_def = f"    {field['field_name']}: {field_type} = Field(..., description='{field['description']}')"
        field_definitions.append(field_def)
    
    # Build validators
    validators = []
    for field in fields:
        if field['field_type'] == 'list[str]':
            validator = self._generate_list_validator(field['field_name'])
            validators.extend(validator)
    
    # Combine all parts
    model_code = self._combine_model_parts(
        imports, enum_imports, enum_classes, 
        field_definitions, validators, main_model_name
    )
    
    return model_code
```

**List Validator Generation:**
```python
def _generate_list_validator(self, field_name: str) -> List[str]:
    """Generate complete validator for list fields"""
    validator_name = f"validate_{field_name}"
    
    return [
        f"    @field_validator('{field_name}')",
        f"    @classmethod",
        f"    def {validator_name}(cls, v):",
        f"        if not isinstance(v, list):",
        f"            raise ValueError('{field_name} must be a list')",
        f"        return v",
        ""
    ]
```

**Model Parts Combination:**
```python
def _combine_model_parts(self, imports: List[str], enum_imports: List[str], 
                         enum_classes: List[str], field_definitions: List[str], 
                         validators: List[str], main_model_name: str) -> str:
    """Combine all model parts into complete code"""
    
    model_code = []
    
    # Add imports
    model_code.extend(imports)
    model_code.append("")
    
    # Add enum classes
    if enum_classes:
        model_code.extend(enum_classes)
        model_code.append("")
    
    # Add main model class
    model_code.append(f"class {main_model_name}(BaseModel):")
    
    # Add field definitions
    if field_definitions:
        model_code.extend(field_definitions)
        model_code.append("")
    
    # Add validators
    if validators:
        model_code.extend(validators)
    
    return "\n".join(model_code)
```

### Code Validation

**Validation Method:**
```python
def _validate_generated_code(self, model_code: str) -> bool:
    """Validate generated model code"""
    try:
        # Try to compile the code
        compile(model_code, '<string>', 'exec')
        
        # Check for common validator issues
        lines = model_code.split('\n')
        for i, line in enumerate(lines):
            if '@field_validator' in line:
                # Check if validator has proper body
                validator_start = i
                validator_body_found = False
                
                for j in range(i + 1, min(i + 10, len(lines))):
                    if 'def ' in lines[j]:
                        # Check if function has proper body
                        for k in range(j + 1, min(j + 5, len(lines))):
                            if lines[k].strip() and not lines[k].startswith('    @'):
                                validator_body_found = True
                                break
                        break
                
                if not validator_body_found:
                    logger.warning(f"Incomplete validator found at line {i + 1}")
                    return False
        
        return True
    
    except SyntaxError as e:
        logger.warning(f"Syntax error in generated code: {e}")
        return False
```

### Model Creation from Code

**Core Method:**
```python
def _create_model_from_code(self, model_code: str, main_model_name: str) -> Type:
    """Create model class from generated code"""
    
    # Determine if code needs cleaning
    if not self._is_fallback_generated_code(model_code):
        model_code = self._clean_generated_code(model_code)
    
    # Create temporary file
    temp_file_path = f"temp_{uuid.uuid4().hex}.py"
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        f.write(model_code)
    
    try:
        # Import the generated module
        spec = importlib.util.spec_from_file_location("generated_models", temp_file_path)
        generated_module = importlib.util.module_from_spec(spec)
        sys.modules["generated_models"] = generated_module
        spec.loader.exec_module(generated_module)
        
        # Get the main model class
        main_model_class = getattr(generated_module, main_model_name)
        
        # Store the generated models
        self.generated_models[main_model_name] = main_model_class
        
        return main_model_class
    
    finally:
        # Clean up temporary file
        Path(temp_file_path).unlink()
```

**Code Cleaning:**
```python
def _clean_generated_code(self, code: str) -> str:
    """Clean AI-generated code to remove markdown formatting"""
    import re
    
    # Remove markdown code blocks
    code = re.sub(r'```\s*python\s*\n?', '', code, flags=re.IGNORECASE)
    code = re.sub(r'```\s*\n?', '', code)
    
    # Remove common AI artifacts
    code = re.sub(r'^Here\'s the.*?:\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'^Here is the.*?:\n', '', code, flags=re.MULTILINE)
    
    return code.strip()
```

**Fallback Code Detection:**
```python
def _is_fallback_generated_code(self, code: str) -> bool:
    """Detect if code was generated by enhanced fallback"""
    
    # Check for fallback characteristics
    has_complete_validators = True
    has_raise_valueerror = False
    has_return_v = False
    
    lines = code.split('\n')
    in_validator = False
    
    for line in lines:
        if '@field_validator' in line:
            in_validator = True
            continue
        
        if in_validator and 'def ' in line:
            # Check if this validator is complete
            validator_complete = False
            for i, check_line in enumerate(lines[lines.index(line):lines.index(line) + 10]):
                if 'raise ValueError' in check_line:
                    has_raise_valueerror = True
                if 'return v' in check_line:
                    has_return_v = True
                    validator_complete = True
                    break
            
            if not validator_complete:
                has_complete_validators = False
            in_validator = False
    
    # Fallback code has complete validators with proper error handling
    return has_complete_validators and has_raise_valueerror and has_return_v
```

## Usage Examples

### Text Description Method

```python
# Initialize parser
parser = TextDescriptionParser(ai_client)

# Parse description
description = "Extract name, skills list, and industry from resumes"
use_case = "Resume Extraction"

config = parser.parse_description(description, use_case)
# Result: {
#     'use_case': 'Resume Extraction',
#     'description': 'Extract name, skills list, and industry from resumes',
#     'main_model_name': 'ResumeExtractionInfo',
#     'parsed_fields': [
#         {'field_name': 'name', 'field_type': 'str', 'description': 'Name of the person'},
#         {'field_name': 'skills_list', 'field_type': 'list[str]', 'description': 'List of skills'},
#         {'field_name': 'industry', 'field_type': 'enum', 'description': 'Industry type', 'enum_values': ['Tech', 'Finance']}
#     ]
# }

# Generate model
generator = ModelGenerator(claude_client)
model_class, model_code = generator.generate_models_from_text_description(config)
```

### Field-by-Field Configuration Method

```python
# Define field configuration
field_config = {
    "use_case": "Resume Extraction",
    "description": "Extract information from resumes",
    "main_model_name": "trialInfo",  # Will be converted to "TrialInfo"
    "fields": [
        {
            "field_name": "Company Name",
            "field_type": "string",
            "description": "Name of the company"
        },
        {
            "field_name": "Skills List",
            "field_type": "array",
            "description": "List of technical skills"
        },
        {
            "field_name": "Industry",
            "field_type": "category",
            "description": "Industry type",
            "enum_values": ["Tech", "Finance", "Healthcare"]
        }
    ]
}

# Generate model
generator = ModelGenerator(claude_client)
model_class, model_code = generator.generate_models_from_field_config(field_config)
```

## Generated Model Example

Both methods produce identical output:

```python
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import List, Optional

class Industry(str, Enum):
    TECH = "Tech"
    FINANCE = "Finance"
    HEALTHCARE = "Healthcare"

class TrialInfo(BaseModel):
    company_name: str = Field(..., description='Name of the company')
    skills_list: List[str] = Field(..., description='List of technical skills')
    industry: Industry = Field(..., description='Industry type')

    @field_validator('skills_list')
    @classmethod
    def validate_skills_list(cls, v):
        if not isinstance(v, list):
            raise ValueError('skills_list must be a list')
        return v
```

## Key Features

### 1. **Consistent Processing**
- Both methods use identical field preprocessing
- Same model name generation logic
- Identical validation and fallback mechanisms

### 2. **Robust Error Handling**
- AI generation with immediate fallback
- Code validation before execution
- Complete validator generation

### 3. **Flexible Input Formats**
- Natural language descriptions
- Structured field definitions
- Automatic type conversion and validation

### 4. **Production Ready**
- Comprehensive error handling
- Clean code generation
- Proper Python/Pydantic conventions

## Implementation Notes

### Dependencies
- `pydantic` - Model definition and validation
- `anthropic` - Claude AI client
- `openai` - OpenAI client (alternative)
- `typing` - Type hints
- `enum` - Enumeration support
- `importlib` - Dynamic module loading

### File Structure
```
model_generator.py          # Main model generation logic
text_description_parser.py  # Text description parsing
claude_client.py          # Claude AI integration
openai_client.py          # OpenAI integration
```

### Error Handling
- Graceful fallback from AI to manual generation
- Comprehensive code validation
- Detailed logging for debugging
- Clean temporary file management

This architecture ensures reliable, consistent model generation regardless of the input method chosen.
