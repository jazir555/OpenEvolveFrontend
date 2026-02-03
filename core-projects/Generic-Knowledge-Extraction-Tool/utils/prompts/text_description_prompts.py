"""
Text description parsing prompts for AI models
"""

def get_main_parsing_prompt(description: str, use_case: str = "", context: str = "") -> str:
    """Main prompt for parsing text descriptions into field configurations"""
    
    prompt = f"""
You are an expert data extraction specialist. Analyze the following text description and extract structured field requirements for building a Pydantic data model.

DESCRIPTION:
{description}

USE CASE: {use_case or "General data extraction"}
CONTEXT: {context or "No additional context provided"}

TASK:
Extract all fields that need to be extracted from documents based on the description. For each field, determine:

1. **field_name**: Convert to snake_case format (e.g., "Company Name" → "company_name")
2. **field_type**: Choose appropriate type from: str, int, float, bool, list[str], enum, list[enum]
3. **description**: Clear, specific description of what to extract
4. **required**: true/false (default to true unless explicitly mentioned as optional)
5. **enum_values**: Array of possible values if field_type is enum or list[enum]

FIELD TYPE GUIDELINES:
- **str**: Text fields, names, descriptions, addresses, etc.
- **int**: Whole numbers, counts, IDs, years
- **float**: Decimal numbers, percentages, prices, scores
- **bool**: True/false values, yes/no questions
- **list[str]**: Arrays of text values, multiple text entries
- **enum**: Single choice from predefined options (e.g., industry categories)
- **list[enum]**: Multiple choices from predefined options (e.g., multiple skills)

IMPORTANT RULES:
- Convert ALL field names to snake_case format
- Be conservative with field types - prefer 'str' when uncertain
- Default to required=true unless explicitly mentioned as optional
- For enum types, infer reasonable values from context
- If a field can have multiple values from a set, use 'list[enum]'
- Keep descriptions specific and actionable
- Consider the use case context when determining field importance

Return JSON in this exact format:
{{
  "fields": [
    {{
      "field_name": "company_name",
      "field_type": "str",
      "description": "The name of the company or organization",
      "required": true,
      "enum_values": null
    }},
    {{
      "field_name": "industry",
      "field_type": "enum",
      "description": "The primary industry sector",
      "required": true,
      "enum_values": ["Technology", "Healthcare", "Finance", "Manufacturing", "Education", "Retail"]
    }},
    {{
      "field_name": "revenue",
      "field_type": "float",
      "description": "Annual revenue in USD",
      "required": false,
      "enum_values": null
    }}
  ]
}}

Return only the JSON object, no additional text or explanations.
"""
    return prompt

def get_field_type_inference_prompt(field_name: str, description: str, context: str = "") -> str:
    """Prompt for inferring field type from description"""
    
    prompt = f"""
Given the field name and description, determine the most appropriate data type.

FIELD NAME: {field_name}
DESCRIPTION: {description}
CONTEXT: {context}

Choose from: str, int, float, bool, list[str], enum, list[enum]

Consider:
- Text content → str
- Whole numbers → int
- Decimal numbers → float
- True/false values → bool
- Multiple text entries → list[str]
- Single choice from options → enum
- Multiple choices from options → list[enum]

Return only the type name.
"""
    return prompt

def get_enum_values_prompt(field_name: str, description: str, context: str = "") -> str:
    """Prompt for extracting enum values"""
    
    prompt = f"""
Given the field name and description, suggest appropriate enum values.

FIELD NAME: {field_name}
DESCRIPTION: {description}
CONTEXT: {context}

Provide 3-8 relevant enum values that would be commonly found for this field.
Return as a JSON array of strings.

Example: ["Option 1", "Option 2", "Option 3"]
"""
    return prompt

def get_validation_prompt(parsed_fields: list, original_description: str) -> str:
    """Prompt for validating parsed field results"""
    
    prompt = f"""
Validate the following parsed fields against the original description.

ORIGINAL DESCRIPTION:
{original_description}

PARSED FIELDS:
{parsed_fields}

Check for:
1. Missing important fields mentioned in description
2. Incorrect field types
3. Unclear or incomplete descriptions
4. Missing enum values where needed
5. Incorrect required/optional classification

Return JSON with validation results:
{{
  "is_valid": true/false,
  "issues": ["list of issues found"],
  "suggestions": ["list of improvement suggestions"],
  "missing_fields": ["list of potentially missing fields"]
}}
"""
    return prompt

def get_example_descriptions() -> dict:
    """Get example descriptions for common use cases"""
    
    examples = {
        "Invoice Extraction": """
        Extract information from invoices including the invoice number, date, vendor name, 
        total amount, line items with descriptions and prices, tax amount, and payment terms. 
        Also capture the billing address and any special notes or conditions.
        """,
        
        "Resume/CV Parsing": """
        Extract personal information from resumes including full name, contact details (email, phone, address), 
        professional summary, work experience with company names, job titles, dates, and descriptions, 
        education details with degrees and institutions, skills list, and any certifications or awards.
        """,
        
        "Research Paper Data": """
        Extract metadata from research papers including title, authors, abstract, publication date, 
        journal name, DOI, keywords, research methodology, sample size, key findings, and references.
        """,
        
        "Medical Report Analysis": """
        Extract patient information from medical reports including patient ID, name, date of birth, 
        diagnosis codes, symptoms, medications prescribed, lab results with values and units, 
        doctor name, and treatment recommendations.
        """,
        
        "Business Document Processing": """
        Extract key business information including company name, industry sector, annual revenue, 
        number of employees, key executives, business model, target market, competitive advantages, 
        and financial metrics like profit margins and growth rate.
        """,
        
        "Contract Analysis": """
        Extract contract details including parties involved, contract type, effective dates, 
        termination clauses, payment terms, deliverables, liability limitations, governing law, 
        and any special conditions or amendments.
        """
    }
    
    return examples

def get_parsing_tips() -> list:
    """Get tips for writing effective extraction descriptions"""
    
    tips = [
        "Be specific about what information to extract - mention exact field names when possible",
        "Include context about the document type (invoices, resumes, reports, etc.)",
        "Specify data types when important (numbers, dates, categories)",
        "Mention if fields are optional or required",
        "Include examples of values when helpful",
        "Consider the business purpose of the extraction",
        "Think about edge cases and variations in data format",
        "Keep descriptions concise but comprehensive",
        "Use clear, unambiguous language",
        "Consider the target audience for the extracted data"
    ]
    
    return tips
