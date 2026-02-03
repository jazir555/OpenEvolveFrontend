import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ai.clients.openai_client import OpenAIClient
from ai.clients.claude_client import ClaudeClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDescriptionClient(ABC):
    """Abstract base class for text description parsing clients"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Text Description Client
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


    @abstractmethod
    def parse_description(self, description: str, use_case: str = "", context: str = "") -> Dict[str, Any]:
        """Parse text description and return structured field data"""
        pass

class OpenAITextDescriptionClient(TextDescriptionClient):
    """OpenAI-based text description parser"""

    def __init__(self, api_config: Optional[Dict[str, Any]] = None):
        self.openai_client = OpenAIClient(api_config=api_config)
        self.client = self.openai_client.client
        self.model = self.openai_client.model

    def parse_description(self, description: str, use_case: str = "", context: str = "") -> Dict[str, Any]:
        """Parse description using OpenAI"""
        try:
            prompt = self._build_parsing_prompt(description, use_case, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing text descriptions and extracting structured field requirements for data extraction. Return valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4000,  # Increased from 2000 to handle larger responses
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            # Parse JSON response
            response_text = response.choices[0].message.content
            parsed_data = json.loads(response_text)
            
            logger.info("Successfully parsed description using OpenAI")
            return parsed_data

        except Exception as e:
            logger.error(f"OpenAI parsing failed: {e}")
            raise

    def _build_parsing_prompt(self, description: str, use_case: str, context: str) -> str:
        """Build the parsing prompt for OpenAI"""
        prompt = f"""
Analyze the following text description and extract structured field requirements for data extraction.

DESCRIPTION:
{description}

USE CASE: {use_case or "General data extraction"}
CONTEXT: {context or "No additional context provided"}

Extract the following information and return as JSON:

1. Identify all fields that need to be extracted
2. For each field, determine:
   - field_name: Convert to snake_case format
   - field_type: Choose from str, int, float, bool, list[str], enum, list[enum]
   - description: Clear description of what to extract
   - required: true/false (default to true if not specified)
   - enum_values: Array of possible values if field_type is enum or list[enum]

IMPORTANT RULES:
- Convert all field names to snake_case (e.g., "Company Name" → "company_name")
- Use 'str' for text fields, 'int' for whole numbers, 'float' for decimals
- Use 'bool' for true/false fields
- Use 'list[str]' for arrays of text
- Use 'enum' for single choice from predefined options
- Use 'list[enum]' for multiple choices from predefined options
- If field can have multiple values from a set, use 'list[enum]'
- Default to required=true unless explicitly mentioned as optional
- Be conservative with field types - prefer 'str' when uncertain
- If there are many fields, ensure the JSON is complete and properly closed

Return JSON in this exact format:
{{
  "fields": [
    {{
      "field_name": "company_name",
      "field_type": "str",
      "description": "The name of the company",
      "required": true,
      "enum_values": null
    }},
    {{
      "field_name": "industry",
      "field_type": "enum",
      "description": "The industry sector",
      "required": true,
      "enum_values": ["Technology", "Healthcare", "Finance", "Manufacturing"]
    }}
  ]
}}

CRITICAL: Ensure the JSON response is complete with proper closing brackets and braces. Do not truncate the response.

"""
        return prompt

class ClaudeTextDescriptionClient(TextDescriptionClient):
    """Claude-based text description parser"""

    def __init__(self):
        self.claude_client = ClaudeClient()

    def parse_description(self, description: str, use_case: str = "", context: str = "") -> Dict[str, Any]:
        """Parse description using Claude with fallback for large field lists"""
        try:
            # First attempt with standard prompt
            result = self._parse_with_standard_prompt(description, use_case, context)
            return result
            
        except Exception as e:
            logger.warning(f"Standard parsing failed: {e}")
            
            # Try with chunked approach for very large descriptions
            if len(description) > 1000:  # Arbitrary threshold for "large" descriptions
                logger.info("Attempting chunked parsing for large description")
                try:
                    result = self._parse_with_chunked_approach(description, use_case, context)
                    return result
                except Exception as chunk_error:
                    logger.error(f"Chunked parsing also failed: {chunk_error}")
            
            # Re-raise the original error
            raise e
    
    def _parse_with_standard_prompt(self, description: str, use_case: str, context: str) -> Dict[str, Any]:
        """Parse using the standard prompt approach"""
        prompt = self._build_parsing_prompt(description, use_case, context)
        
        response = self.claude_client.client.messages.create(
            model=self.claude_client.model,
            max_tokens=4000,  # Increased from 2000 to handle larger responses
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON response - Claude might wrap in code blocks
        response_text = response.content[0].text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        elif response_text.startswith('```'):
            response_text = response_text[3:]
        
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Try to parse JSON
        try:
            parsed_data = json.loads(response_text)
            logger.info("Successfully parsed description using Claude")
            return parsed_data
        except json.JSONDecodeError as json_error:
            # Check if response was truncated
            if self._is_response_truncated(response_text):
                logger.warning("Response appears to be truncated, attempting to fix JSON structure")
                fixed_response = self._fix_truncated_json(response_text)
                if fixed_response:
                    try:
                        parsed_data = json.loads(fixed_response)
                        logger.info("Successfully parsed description using Claude (with JSON fix)")
                        return parsed_data
                    except json.JSONDecodeError:
                        pass
            
            # If all else fails, raise the original error
            logger.error(f"Claude JSON parsing failed: {json_error}")
            logger.error(f"Raw response: {response.content[0].text}")
            raise json_error

    def _build_parsing_prompt(self, description: str, use_case: str, context: str) -> str:
        """Build the parsing prompt for Claude"""
        prompt = f"""
Analyze the following text description and extract structured field requirements for data extraction.

DESCRIPTION:
{description}

USE CASE: {use_case or "General data extraction"}
CONTEXT: {context or "No additional context provided"}

Extract the following information and return as JSON:

1. Identify all fields that need to be extracted
2. For each field, determine:
   - field_name: Convert to snake_case format
   - field_type: Choose from str, int, float, bool, list[str], enum, list[enum]
   - description: Clear description of what to extract
   - required: true/false (default to true if not specified)
   - enum_values: Array of possible values if field_type is enum or list[enum]

IMPORTANT RULES:
- Convert all field names to snake_case (e.g., "Company Name" → "company_name")
- Use 'str' for text fields, 'int' for whole numbers, 'float' for decimals
- Use 'bool' for true/false fields
- Use 'list[str]' for arrays of text
- Use 'enum' for single choice from predefined options
- Use 'list[enum]' for multiple choices from predefined options
- If field can have multiple values from a set, use 'list[enum]'
- Default to required=true unless explicitly mentioned as optional
- Be conservative with field types - prefer 'str' when uncertain
- If there are many fields, ensure the JSON is complete and properly closed

Return JSON in this exact format:
{{
  "fields": [
    {{
      "field_name": "company_name",
      "field_type": "str",
      "description": "The name of the company",
      "required": true,
      "enum_values": null
    }},
    {{
      "field_name": "industry",
      "field_type": "enum",
      "description": "The industry sector",
      "required": true,
      "enum_values": ["Technology", "Healthcare", "Finance", "Manufacturing"]
    }}
  ]
}}

CRITICAL: Ensure the JSON response is complete with proper closing brackets and braces. Do not truncate the response.

Return only the JSON object, no additional text or explanations.
"""
        return prompt
    
    def _is_response_truncated(self, response_text: str) -> bool:
        """Check if the response appears to be truncated"""
        # Check for incomplete JSON structure
        if not response_text.strip().endswith('}'):
            return True
        
        # Check for incomplete field definitions
        if '"required": true' in response_text and not response_text.strip().endswith('}'):
            return True
        
        # Check for missing closing brackets
        open_braces = response_text.count('{')
        close_braces = response_text.count('}')
        if open_braces > close_braces:
            return True
        
        open_brackets = response_text.count('[')
        close_brackets = response_text.count(']')
        if open_brackets > close_brackets:
            return True
        
        return False
    
    def _fix_truncated_json(self, response_text: str) -> Optional[str]:
        """Attempt to fix truncated JSON by adding missing closing brackets"""
        try:
            # Count opening and closing brackets/braces
            open_braces = response_text.count('{')
            close_braces = response_text.count('}')
            open_brackets = response_text.count('[')
            close_brackets = response_text.count(']')
            
            fixed_response = response_text
            
            # Add missing closing brackets
            missing_brackets = open_brackets - close_brackets
            if missing_brackets > 0:
                fixed_response += ']' * missing_brackets
            
            # Add missing closing braces
            missing_braces = open_braces - close_braces
            if missing_braces > 0:
                fixed_response += '}' * missing_braces
            
            # If the response ends with a comma, remove it
            fixed_response = fixed_response.rstrip().rstrip(',')
            
            # Ensure proper JSON structure
            if not fixed_response.strip().endswith('}'):
                fixed_response += '}'
            
            return fixed_response
            
        except Exception as e:
            logger.error(f"Failed to fix truncated JSON: {e}")
            return None
    
    def _parse_with_chunked_approach(self, description: str, use_case: str, context: str) -> Dict[str, Any]:
        """Parse large descriptions by breaking them into smaller chunks"""
        # Split description into sentences or logical chunks
        sentences = description.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) > 500:  # Chunk size limit
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Parse each chunk separately
        all_fields = []
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Parsing chunk {i+1}/{len(chunks)}")
                chunk_result = self._parse_with_standard_prompt(chunk, f"{use_case}_chunk_{i+1}", context)
                chunk_fields = chunk_result.get('fields', [])
                all_fields.extend(chunk_fields)
            except Exception as e:
                logger.warning(f"Failed to parse chunk {i+1}: {e}")
                continue
        
        # Remove duplicate fields based on field_name
        unique_fields = []
        seen_names = set()
        for field in all_fields:
            field_name = field.get('field_name', '')
            if field_name and field_name not in seen_names:
                unique_fields.append(field)
                seen_names.add(field_name)
        
        logger.info(f"Chunked parsing completed: {len(unique_fields)} unique fields extracted")
        return {'fields': unique_fields}
