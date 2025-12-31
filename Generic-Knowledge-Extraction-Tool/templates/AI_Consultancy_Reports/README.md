# AI Consultancy Reports Use-Case

## Purpose
Extracts structured information from AI consultancy reports, capturing company details, expert recommendations, and business requirements for AI implementation projects.

## Key Files
- `config.json` - Complete field configuration with 13 structured fields
- `AI_Consultancy_Reports_models.py` - Generated Pydantic model for validation
- `AI_Consultancy_Reports_prompt.py` - Specialized extraction prompts for consultancy documents

## Extracted Fields
- **Company Information**: name, type (startup/established), domain, target groups
- **Consultancy Details**: expert names, customer manager, consultancy date
- **AI Requirements**: maturity level, AI field, intended AI idea
- **Service Details**: services sought, key recommendations

## Field Types & Validation
- **Enums**: Company type, AI maturity level, AI field, company domain, services sought
- **Lists**: Expert names, target groups, key recommendations
- **Strings**: Company name, customer manager, consultancy date, AI idea

## Integration
- **Generated from**: Text description using natural language parsing
- **Configuration mode**: `text_description`
- **AI Model**: claude-sonnet-4-20250514 for parsing
- **Validation**: Structured enum values for business domains and service types

## Usage Pattern
Ideal for processing business consultancy documents where structured data extraction enables:
- Client relationship management
- Service delivery tracking
- AI readiness assessment
- Recommendation analysis