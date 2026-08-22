# Prompts Directory

## Purpose
Contains specialized AI prompt templates and prompt generation utilities for text description parsing and field configuration. These prompts enable natural language configuration of extraction fields, converting user descriptions into structured field definitions.

## Key Files
- `text_description_prompts.py` - Core prompt templates for parsing text descriptions into field configurations

## Architecture
```
Prompt System:
User Description → Parsing Prompts → AI Processing → Structured Fields
     ↓                   ↓              ↓              ↓
  Natural Language    Specialized    Claude/OpenAI    JSON Config
   Requirements        Prompts       Processing       Output
```

## Integration
- **Text Description Parser**: `text_description_parser.py` uses these prompts for field parsing
- **AI Clients**: Prompts are sent to Claude/OpenAI for intelligent field extraction
- **Model Generator**: Parsed fields are fed into the dynamic model generation pipeline
- **UI Application**: Enables natural language configuration in the BubbleLab (TypeScript) interface

## Patterns
1. **Template-Based Prompts**: Standardized prompt structures with variable injection
2. **Type-Aware Parsing**: Prompts guide AI to choose appropriate Pydantic field types
3. **Snake_case Conversion**: Automatic conversion of field names to Python conventions
4. **Enum Detection**: Smart identification of enumerated values from descriptions

## Dependencies
- **Input**: User text descriptions, use-case context, domain requirements
- **Output**: Structured field configurations (JSON format)
- **AI Models**: Requires Claude or OpenAI for natural language processing
- **Validation**: Results are validated before model generation

## Entry Points
1. **get_main_parsing_prompt()**: Primary function for text description parsing
2. **Field Type Guidelines**: Built-in type selection logic for various data types
3. **Template Variables**: Dynamic prompt customization based on use-case

## Common Usage Patterns

### Text Description Parsing
```python
from prompts.text_description_prompts import get_main_parsing_prompt

prompt = get_main_parsing_prompt(
    description="Extract company name, expert names, and recommendations",
    use_case="AI Consultancy Reports",
    context="Business consultancy documents"
)
```

### AI Field Configuration
```python
# Used by text_description_parser.py
parsed_fields = ai_client.generate_completion(
    prompt=get_main_parsing_prompt(user_description),
    system_message="You are a data extraction specialist"
)
```

## Prompt Features
- **Field Type Detection**: Automatically suggests appropriate types (str, int, list[str], enum)
- **Requirement Analysis**: Determines if fields are required or optional
- **Enum Value Extraction**: Identifies possible values for enumerated fields
- **Description Generation**: Creates clear field descriptions for extraction

## Advanced Capabilities
- **Context-Aware Parsing**: Adapts field suggestions based on use-case and domain
- **Multi-Language Support**: Handles descriptions in various languages
- **Complex Type Handling**: Supports nested types like `list[enum]` and complex objects
- **Validation Prompts**: Includes guidance for AI to validate field consistency

## Integration with Text Description System
This directory powers the text description configuration feature:
1. User enters natural language description
2. Prompts guide AI to parse requirements
3. Structured fields are generated automatically
4. Fields are validated and fed to model generation
5. Complete extraction configuration is created

## Customization Guidelines
- **Domain-Specific Prompts**: Add specialized prompts for different industries
- **Type Extensions**: Extend field type detection for custom data structures
- **Language Adaptation**: Modify prompts for non-English descriptions
- **Validation Enhancement**: Add more robust field validation logic