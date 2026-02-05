#!/usr/bin/env python3
"""
Document Type Classification & Routing (Case 1) Implementation
Multi-type document extraction using the same approach as single-type extraction.
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Case1 Classifier
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re
import json
import logging
import os
import shutil
from pathlib import Path
from core.text_description_parser import TextDescriptionParser
from core.model_generator import ModelGenerator


logger = logging.getLogger(__name__)

class ExtractionCase(Enum):
    CLASSIFICATION_ROUTING = "classification_routing"
    RELATIONSHIP_MAPPING = "relationship_mapping"
    CONTEXT_AWARE = "context_aware"

@dataclass
class DocumentClassification:
    document: Dict[str, Any]
    document_type: str
    confidence: float
    reasoning: str

class MultiTypeDescriptionParser:
    """Parse multi-type descriptions using the same approach as single-type extraction"""
    
    def __init__(self, ai_client=None):
        self.ai_client = ai_client
        self.extracted_document_types = {}
    
    def parse_multi_type_description(self, description: str, use_case: str = "") -> Dict[str, Any]:
        """Parse description to extract document types and their fields"""
        try:
            prompt = f"""
            Analyze this extraction description for multi-type document processing and identify document types and their specific requirements.
            
            DESCRIPTION:
            {description}
            
            USE CASE: {use_case or "Multi-type document extraction"}
            
            Extract the following information and return as JSON:
            
            1. Identify all document types mentioned (e.g., resumes, invoices, research papers, contracts, etc.)
            2. For each document type, identify the specific fields to extract
            3. For each field, determine:
               - field_name: Convert to snake_case format
               - field_type: Choose from str, int, float, bool, list[str], enum, list[enum]
               - description: Clear description of what to extract
               - required: true/false (default to true if not specified)
               - enum_values: Array of possible values if field_type is enum or list[enum]
            
            IMPORTANT RULES:
            - Convert all field names to snake_case (e.g., "Company Name" -> "company_name")
            - Use 'str' for text fields, 'int' for whole numbers, 'float' for decimals
            - Use 'bool' for true/false fields
            - Use 'list[str]' for arrays of text
            - Use 'enum' for single choice from predefined options
            - Use 'list[enum]' for multiple choices from predefined options
            - Default to required=true unless explicitly mentioned as optional
            - Be conservative with field types - prefer 'str' when uncertain
            - Only include document types explicitly mentioned in the description
            - Only include fields explicitly mentioned for each document type
            
            Return JSON in this exact format:
            {{
                "document_types": {{
                    "resume": {{
                        "fields": [
                            {{
                                "field_name": "name",
                                "field_type": "str",
                                "description": "Full name",
                                "required": true,
                                "enum_values": null
                            }},
                            {{
                                "field_name": "competence_area",
                                "field_type": "enum",
                                "description": "Main competence area",
                                "required": true,
                                "enum_values": ["computer science", "electronic engineering", "construction engineering", "electrical engineering"]
                            }}
                        ]
                    }},
                    "invoice": {{
                        "fields": [
                            {{
                                "field_name": "billing_address",
                                "field_type": "str",
                                "description": "Billing address",
                                "required": true,
                                "enum_values": null
                            }},
                            {{
                                "field_name": "amount",
                                "field_type": "float",
                                "description": "Invoice amount",
                                "required": true,
                                "enum_values": null
                            }}
                        ]
                    }}
                }}
            }}
            
            Return only the JSON object, no additional text or explanations.
            """
            
            if self.ai_client and hasattr(self.ai_client, 'chat') and hasattr(self.ai_client.chat, 'completions'):
                response = self.ai_client.chat.completions.create(
                    model=getattr(self.ai_client, 'model', 'gpt-4.1-2025-04-14'),
                    messages=[
                        {"role": "system", "content": "You are an expert at parsing extraction requirements. Extract only what is explicitly mentioned."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                self.extracted_document_types = result.get('document_types', {})
                return self.extracted_document_types
            else:
                # No fallback - LLM is required for parsing
                logger.error("AI client not available - LLM is required for description parsing")
                raise ValueError("AI client is required for description parsing. Please ensure AI client is properly configured.")
                
        except Exception as e:
            logger.error(f"Error extracting document types from description: {e}")
            raise ValueError(f"Description parsing failed: {e}")
    

class DocumentTypeClassifier:
    """Classify documents using LLM based on extracted document types"""
    
    def __init__(self, extracted_document_types: Dict[str, Dict[str, Any]], ai_client=None):
        self.extracted_document_types = extracted_document_types
        self.ai_client = ai_client
    
    def classify_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Classify documents by extracted document types using LLM"""
        # Initialize with extracted document types
        classified = {doc_type: [] for doc_type in self.extracted_document_types.keys()}
        
        for doc in documents:
            doc_type = self._classify_single_document_with_llm(doc)
            if doc_type in classified:
                classified[doc_type].append(doc)
                logger.info(f"Document {doc.get('file_name', 'unknown')} classified as {doc_type}")
            else:
                logger.warning(f"Document {doc.get('file_name', 'unknown')} could not be classified into any expected type")
        
        return classified
    
    def _classify_single_document_with_llm(self, document: Dict[str, Any]) -> str:
        """Classify document using LLM based on extracted document types"""
        try:
            if self.ai_client and hasattr(self.ai_client, 'chat') and hasattr(self.ai_client.chat, 'completions'):
                return self._ai_classify_document(document)
            else:
                # No fallback - LLM is required for classification
                logger.error("AI client not available - LLM is required for document classification")
                raise ValueError("AI client is required for document classification. Please ensure AI client is properly configured.")
        except Exception as e:
            logger.error(f"Error classifying document {document.get('file_name', 'unknown')}: {e}")
            raise ValueError(f"Document classification failed: {e}")
    
    def _ai_classify_document(self, document: Dict[str, Any]) -> str:
        """Use LLM to classify document based on extracted document types"""
        try:
            # Build document types list for the prompt
            doc_types_list = list(self.extracted_document_types.keys())
            
            prompt = f"""
            Analyze this document and classify it into one of the expected document types.
            
            EXPECTED DOCUMENT TYPES: {doc_types_list}
            
            DOCUMENT INFORMATION:
            Filename: {document.get('file_name', 'unknown')}
            Content Preview: {document.get('text_content', '')[:1500]}
            
            CLASSIFICATION TASK:
            Based on the document content and filename, determine which of the expected document types this document belongs to.
            
            Return JSON in this exact format:
            {{
                "document_type": "one_of_the_expected_types",
                "confidence": 0.95,
                "reasoning": "Brief explanation of why this document was classified as this type"
            }}
            
            IMPORTANT:
            - The document_type must be one of: {doc_types_list}
            - Provide a confidence score between 0.0 and 1.0
            - Give clear reasoning for your classification
            - Consider both filename and content when making the decision
            """
            
            response = self.ai_client.chat.completions.create(
                model=getattr(self.ai_client, 'model', 'gpt-4.1-2025-04-14'),
                messages=[
                    {"role": "system", "content": "You are an expert document classifier. Analyze documents and classify them accurately based on content and context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            doc_type = result.get('document_type', 'unknown')
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', 'No reasoning provided')
            
            # Validate that the classified type is in our expected types
            if doc_type in self.extracted_document_types:
                logger.info(f"Document {document.get('file_name', 'unknown')} classified as {doc_type} (confidence: {confidence:.2f}) - {reasoning}")
                return doc_type
            else:
                logger.error(f"LLM classified document as {doc_type} which is not in expected types: {doc_types_list}")
                raise ValueError(f"Invalid document type '{doc_type}' returned by LLM. Expected one of: {doc_types_list}")
            
        except Exception as e:
            logger.error(f"AI classification failed: {e}")
            raise ValueError(f"Document classification failed: {e}")

class MultiTypeUseCaseManager:
    """Manages multiple extractors within a single use-case"""
    
    def __init__(self, base_use_case_name: str, ai_client=None):
        self.base_use_case_name = base_use_case_name
        self.ai_client = ai_client
        self.extractors = {}  # {doc_type: extractor_info}
        self.use_case_path = f"templates/{base_use_case_name}"
        
    def create_extractors_for_document_types(self, classified_documents: Dict[str, List[Dict[str, Any]]], 
                                          extraction_description: str,
                                          extracted_document_types: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create individual extractors for each document type"""
        extractor_info = {}
        
        for doc_type, documents in classified_documents.items():
            if not documents:
                continue
                
            # Create extractor name for this document type
            extractor_name = f"{self.base_use_case_name}_{doc_type}"
            extractor_path = f"{self.use_case_path}/{doc_type}"
            
            # Create directory for this extractor
            os.makedirs(extractor_path, exist_ok=True)
            
            # Get fields for this document type from extracted types
            fields = extracted_document_types.get(doc_type, {}).get('fields', [])
            
            # Create config.json for this extractor using the same format as single-type
            config = {
                "extraction_config": {
                    "configuration_mode": "text_description",
                    "use_case": f"{self.base_use_case_name} - {doc_type.title()} Extraction",
                    "description": f"Extract {doc_type} information from {doc_type} documents",
                    "main_model_name": f"{extractor_name.title()}Info",
                    "text_description": f"Extract {doc_type} information from {doc_type} documents",
                    "parsed_fields": fields,
                    "fields": fields,  # For backward compatibility
                    "model_generation_model": "claude-sonnet-4-20250514",
                    "extraction_model": "gpt-4.1-2025-04-14",
                    "use_azure": False,
                    "parsing_method": "fast",
                    "extraction_type": "single_type",  # Each extractor is single-type
                    "document_type": doc_type
                }
            }
            
            # Save config.json
            config_path = f"{extractor_path}/config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Check if models already exist for persisted use cases
            model_filename = f"{doc_type}_models.py"
            prompt_filename = f"{doc_type}_prompt.py"
            model_path = f"{extractor_path}/{model_filename}"
            prompt_path = f"{extractor_path}/{prompt_filename}"
            
            try:
                model_generator = ModelGenerator(model_selection="claude-sonnet-4-20250514")
                
                if os.path.exists(model_path) and os.path.exists(prompt_path):
                    # Load existing models for persisted use case
                    logger.info(f"Loading existing models for {doc_type} from persisted use case")
                    pydantic_model_class, extraction_prompt = model_generator.load_models_and_prompt(model_path, prompt_path)
                    
                    extractor_info[doc_type] = {
                        'path': extractor_path,
                        'config': config,
                        'model_class': pydantic_model_class,
                        'extraction_prompt': extraction_prompt,
                        'documents': documents
                    }
                    
                    logger.info(f"Loaded existing extractor for {doc_type}: {len(documents)} documents")
                    
                else:
                    # Generate new models for new use case
                    logger.info(f"Generating new models for {doc_type}")
                    pydantic_model_class, model_code = model_generator.generate_models_from_config_data(config)
                    
                    # Save models and prompt
                    model_generator.save_generated_models(model_path, model_code)
                    model_generator.save_extraction_prompt(prompt_path)
                    
                    extractor_info[doc_type] = {
                        'path': extractor_path,
                        'config': config,
                        'model_class': pydantic_model_class,
                        'extraction_prompt': model_generator.get_extraction_prompt(),
                        'documents': documents
                    }
                    
                    logger.info(f"Created new extractor for {doc_type}: {len(documents)} documents")
                
            except Exception as e:
                logger.error(f"Failed to create/load extractor for {doc_type}: {e}")
                continue
        
        self.extractors = extractor_info
        return extractor_info
    
    def extract_from_all_types(self) -> Dict[str, Any]:
        """Extract data using all created extractors"""
        results = {}
        
        for doc_type, extractor_info in self.extractors.items():
            try:
                # Use the appropriate extractor based on the AI client
                if self.ai_client and hasattr(self.ai_client, 'chat'):
                    # Use OpenAI extractor
                    from ai.extractors.openai_extractor import OpenAIExtractor
                    extractor = OpenAIExtractor(api_config={'use_azure': False, 'api_key': 'dummy'})
                    extractor.client = self.ai_client
                    
                    # Extract data using the same approach as single-type extraction
                    doc_results = extractor.extract_batch(
                        documents=extractor_info['documents'],
                        extraction_prompt=extractor_info['extraction_prompt'],
                        model_class=extractor_info['model_class'],
                        additional_instructions=""
                    )
                    
                    results[doc_type] = {
                        'extractor_path': extractor_info['path'],
                        'documents_processed': len(extractor_info['documents']),
                        'results': doc_results
                    }
                    
                else:
                    results[doc_type] = {
                        'extractor_path': extractor_info['path'],
                        'documents_processed': len(extractor_info['documents']),
                        'error': 'AI client not available'
                    }
                    
            except Exception as e:
                logger.error(f"Error extracting from {doc_type}: {e}")
                results[doc_type] = {
                    'extractor_path': extractor_info['path'],
                    'documents_processed': len(extractor_info['documents']),
                    'error': str(e)
                }
        
        return results

class Case1Extractor:
    """Main orchestrator for Case 1: Document Type Classification & Routing"""
    
    def __init__(self, ai_client=None):
        self.ai_client = ai_client
        self.parser = MultiTypeDescriptionParser(ai_client)
    
    def extract_from_documents(self, documents: List[Dict[str, Any]], 
                             extraction_description: str, 
                             use_case_name: str) -> Dict[str, Any]:
        """Main extraction method for Case 1 using the same approach as single-type extraction"""
        try:
            # Step 1: Parse description to extract document types and fields (same as single-type)
            extracted_document_types = self.parser.parse_multi_type_description(extraction_description, use_case_name)
            
            # Step 2: Classify documents using extracted types with LLM
            classifier = DocumentTypeClassifier(extracted_document_types, self.ai_client)
            classified_documents = classifier.classify_documents(documents)
            
            # Step 3: Create multi-type use case manager
            use_case_manager = MultiTypeUseCaseManager(use_case_name, self.ai_client)
            
            # Step 4: Create extractors for each document type using the same approach as single-type
            extractor_info = use_case_manager.create_extractors_for_document_types(
                classified_documents, extraction_description, extracted_document_types
            )
            
            # Step 5: Extract data from all document types using the same approach as single-type
            results = use_case_manager.extract_from_all_types()
            
            return {
                'extraction_case': 'classification_routing',
                'classified_documents': {k: len(v) for k, v in classified_documents.items() if v},
                'extractors_created': len(extractor_info),
                'extractor_info': extractor_info,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error in Case 1 extraction: {e}")
            return {
                'error': str(e),
                'extraction_case': 'classification_routing',
                'results': {}
            }