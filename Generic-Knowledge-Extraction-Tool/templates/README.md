# Use-cases Directory

## Purpose
Contains pre-configured extraction templates for common document types. Each use-case provides a complete extraction setup including field configurations, generated Pydantic models, and specialized AI prompts for specific document categories.

## Key Files Structure
Each use-case follows a standardized pattern:
- `config.json` - Complete extraction configuration with field definitions
- `{use_case_name}_models.py` - Generated Pydantic models for data validation
- `{use_case_name}_prompt.py` - Specialized extraction prompts for AI models


## Integration
- **Main Application**: Use-cases are loaded via the Streamlit UI for quick template selection
- **Model Generator**: Templates provide pre-validated configurations to `model_generator.py`
- **Document Parser**: Each use-case can specify document type optimizations
- **AI Clients**: Specialized prompts are fed to Claude/OpenAI clients for extraction

## Patterns
1. **Configuration-First**: All extraction logic starts from `config.json`
2. **AI-Generated Models**: Pydantic models are generated from field configurations
3. **Domain-Specific Prompts**: Each use-case has tailored extraction instructions
4. **Validation Pipeline**: Generated models ensure data quality and type safety

## Dependencies
- **Input**: Field configurations, document types, business rules
- **Output**: Validated extraction results, Excel exports
- **Core Systems**: `model_generator.py`, AI clients, `document_parser.py`

## Entry Points
1. **UI Selection**: Users choose templates from Streamlit dropdown
2. **Config Loading**: `ui_app.py` loads `config.json` files
3. **Template Application**: Configurations are applied to the extraction pipeline

## Common Usage Patterns

### Loading a Use-Case Template
```python
# Via Streamlit UI
selected_use_case = st.selectbox("Choose template", use_case_options)
config_path = f"Use-cases/{selected_use_case}/config.json"
```

### Creating New Use-Cases
1. Create new directory with use-case name
2. Generate `config.json` through text description or field-by-field setup
3. Run model generation to create `*_models.py` and `*_prompt.py`
4. Test extraction with sample documents

### Template Customization
- Modify field configurations in `config.json`
- Regenerate models using the main application
- Fine-tune prompts for domain-specific terminology