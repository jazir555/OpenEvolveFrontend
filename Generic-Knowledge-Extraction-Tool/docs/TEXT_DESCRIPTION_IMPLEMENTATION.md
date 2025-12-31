# Text Description Feature Implementation

## Overview
Added textual description functionality to generate Pydantic models from natural language descriptions, providing an alternative to field-by-field configuration.

## Architecture Changes

### New Modules Added

#### 1. `text_description_parser.py`
- **Purpose**: Core parsing logic for converting text descriptions to field configurations
- **Key Classes**: `TextDescriptionParser`, `ParsedField`
- **Dependencies**: `text_description_client`, `pydantic`
- **Main Methods**:
  - `parse_extraction_description()`: Main parsing entry point
  - `_extract_field_metadata()`: Extract fields from AI response
  - `_validate_parsed_fields()`: Validate and clean parsed fields
  - `_convert_to_config_format()`: Convert to standard config format

#### 2. `text_description_client.py`
- **Purpose**: AI communication layer for parsing descriptions
- **Key Classes**: `TextDescriptionClient` (abstract), `OpenAITextDescriptionClient`, `ClaudeTextDescriptionClient`
- **Dependencies**: `openai_client`, `claude_client`
- **Main Methods**:
  - `parse_description()`: Send description to AI and parse response
  - `_build_parsing_prompt()`: Generate AI prompts for field extraction

#### 3. `prompts/text_description_prompts.py`
- **Purpose**: Centralized prompt templates and examples
- **Key Functions**:
  - `get_main_parsing_prompt()`: Main AI prompt for field extraction
  - `get_example_descriptions()`: Pre-built examples for common use cases
  - `get_parsing_tips()`: User guidance for writing descriptions

### Modified Modules

#### 1. `ui_app.py`
**Session State Variables Added**:
```python
'configuration_mode': "fields" or "text_description"
'text_description': str
'parsed_fields': List[Dict]
'description_parsing_model': str
'parsing_in_progress': bool
```

**New Functions**:
- `parse_text_description()`: Handle AI parsing with progress indicators

**UI Changes**:
- Configuration mode selection (radio buttons)
- Text description input interface with examples
- Parsed fields preview table
- Conditional rendering of field configuration section

**Updated Functions**:
- `initialize_session_state()`: Added text description variables
- `validate_configuration()`: Added text description validation logic
- `export_configuration()`: Added text description mode data
- Model loading section: Added text description mode restoration

#### 2. `model_generator.py`
**New Methods**:
- `generate_models_from_text_description()`: Generate models from parsed fields
- Updated `generate_models_from_config_data()`: Route to appropriate generation method

**Integration**:
- Automatic detection of configuration mode
- Seamless integration with existing model generation pipeline

## Data Flow

1. **User Input**: Natural language description in text area
2. **AI Parsing**: Description sent to OpenAI/Claude for field extraction
3. **Field Processing**: AI response parsed into structured field configuration
4. **Validation**: Fields validated and cleaned (snake_case conversion, type mapping)
5. **Preview**: Parsed fields displayed in preview table
6. **Model Generation**: Standard model generation pipeline using parsed fields
7. **Extraction**: Normal extraction workflow with generated models

## Configuration Format

**Text Description Mode Config**:
```json
{
  "extraction_config": {
    "configuration_mode": "text_description",
    "text_description": "Extract company information...",
    "parsed_fields": [...],
    "description_parsing_model": "claude-sonnet-4-20250514",
    "fields": [...] // For backward compatibility
  }
}
```

## AI Prompt Engineering

**Field Extraction Prompt Structure**:
- Context-aware parsing based on use case
- Field name conversion to snake_case
- Type inference (str, int, float, bool, list, enum)
- Required/optional field detection
- Enum value extraction for categorical fields

**Supported Field Types**:
- `str`: Text fields
- `int`: Whole numbers
- `float`: Decimal numbers
- `bool`: True/false values
- `list[str]`: Arrays of text
- `enum`: Single choice from options
- `list[enum]`: Multiple choices from options

## Validation Logic

**Text Description Validation**:
- Minimum 20 characters, maximum 2000 characters
- Must contain extraction keywords (extract, get, find, etc.)
- Parsed fields must be present for model generation

**Field Validation**:
- Unique field names (snake_case conversion)
- Required descriptions
- Enum values for enum types
- Type consistency

## Error Handling

**Parsing Errors**:
- AI response validation and JSON parsing
- Fallback to manual field entry on parsing failure
- Progress indicators and user feedback

**Validation Errors**:
- Real-time character count validation
- Field-specific error messages
- Graceful degradation to field-by-field mode

## Backward Compatibility

- Existing field-based configurations continue to work
- Configuration mode detection for loading saved models
- Unified model generation pipeline
- Session state migration for existing users

## Dependencies

**New Dependencies**:
- `text_description_parser` → `text_description_client`
- `text_description_client` → `openai_client`, `claude_client`
- `ui_app` → `text_description_parser`, `prompts.text_description_prompts`
- `model_generator` → `text_description_parser`

## Testing Considerations

**Key Test Scenarios**:
- Text description parsing accuracy
- Field type inference correctness
- Configuration mode switching
- Model generation from parsed fields
- Error handling and fallback scenarios
- Backward compatibility with existing configs

## Future Enhancement Points

1. **Enhanced AI Prompts**: More sophisticated context-aware parsing
2. **Field Type Refinement**: Better type inference algorithms
3. **Batch Processing**: Parse multiple descriptions simultaneously
4. **Template Management**: User-defined description templates
5. **Validation Improvements**: More sophisticated field validation rules
6. **Performance Optimization**: Caching and response optimization

## File Structure Impact

```
Knowledge extraction agent/
├── text_description_parser.py          # NEW
├── text_description_client.py          # NEW
├── prompts/
│   └── text_description_prompts.py     # NEW
├── ui_app.py                           # MODIFIED
├── model_generator.py                  # MODIFIED
└── [existing files unchanged]
```

## Integration Points

- **UI Layer**: Configuration mode selection and text input interface
- **Parsing Layer**: AI-powered description analysis
- **Model Generation**: Seamless integration with existing pipeline
- **Storage Layer**: Configuration persistence and loading
- **Validation Layer**: Multi-level validation for both modes
