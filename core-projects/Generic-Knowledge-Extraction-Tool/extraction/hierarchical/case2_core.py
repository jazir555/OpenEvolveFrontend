#!/usr/bin/env python3
"""
Case 2 Core Data Structures and Enums
Hierarchical multi-document extraction with relationships
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Case2 Core
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
from typing import Dict, List, Tuple, Optional, Any, Type
from dataclasses import dataclass, field
from pydantic import BaseModel
import json
import logging

logger = logging.getLogger(__name__)

class ExtractionStageType(Enum):
    """Enumeration of extraction stages in hierarchical processing"""
    INITIAL = "initial"      # First stage - extract primary keys
    SECONDARY = "secondary"   # Middle stages - use keys to extract related data
    FINAL = "final"          # Final stage - consolidate all information

class RelationshipType(Enum):
    """Types of relationships between extraction stages"""
    DIRECT_KEY_MATCH = "direct_key_match"        # Direct key matching (e.g., order_id)
    DERIVED_KEY_MATCH = "derived_key_match"      # Derived key matching (e.g., customer_id from order)
    AGGREGATION = "aggregation"                  # Aggregation of multiple records
    TRANSFORMATION = "transformation"             # Data transformation between stages

@dataclass
class RelationshipMapping:
    """Defines how data flows between extraction stages"""
    source_stage: str
    target_stage: str
    key_field: str
    relationship_type: RelationshipType
    description: str
    transformation_rules: Optional[Dict[str, Any]] = None

@dataclass
class ExtractionStage:
    """Represents a single stage in the hierarchical extraction process"""
    stage_name: str
    stage_type: ExtractionStageType
    document_types: List[str]
    key_fields: List[str] = field(default_factory=list)
    input_keys: List[str] = field(default_factory=list)  # Keys needed from previous stages
    output_keys: List[str] = field(default_factory=list)  # Keys this stage produces
    extraction_fields: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""

@dataclass
class ExtractionStrategy:
    """Complete strategy for hierarchical document extraction"""
    use_case_name: str
    description: str
    stages: List[ExtractionStage]
    relationships: List[RelationshipMapping]
    extraction_sequence: List[str]  # Ordered list of stage names
    document_classifications: Dict[str, str]  # doc_type -> stage_name mapping
    consolidation_rules: Optional[Dict[str, Any]] = None

@dataclass
class StageExtractor:
    """Configuration for a single stage extractor"""
    stage_name: str
    stage_type: ExtractionStageType
    document_types: List[str]
    model_class: Type[BaseModel]
    prompt: str
    input_keys: List[str]
    output_keys: List[str]
    extraction_fields: List[Dict[str, Any]]
    config_path: str

@dataclass
class HierarchicalExtractionResult:
    """Results from hierarchical extraction process"""
    use_case_name: str
    extraction_strategy: ExtractionStrategy
    stage_results: Dict[str, List[Dict[str, Any]]]
    relationships: Dict[str, Dict[str, Any]]
    consolidated_results: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]

class Case2Config:
    """Configuration management for Case 2 use cases"""
    
    @staticmethod
    def create_config(use_case_name: str, strategy: ExtractionStrategy, 
                     additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a complete config structure for Case 2 use case"""
        config = {
            "extraction_config": {
                "use_case": use_case_name,
                "description": strategy.description,
                "extraction_type": "multi_type_with_relationships",
                "configuration_mode": "text_description",
                "extraction_strategy": {
                    "stages": [
                        {
                            "stage_name": stage.stage_name,
                            "stage_type": stage.stage_type.value,
                            "document_types": stage.document_types,
                            "key_fields": stage.key_fields,
                            "input_keys": stage.input_keys,
                            "output_keys": stage.output_keys,
                            "extraction_fields": stage.extraction_fields,
                            "description": stage.description
                        }
                        for stage in strategy.stages
                    ],
                    "relationships": [
                        {
                            "source_stage": rel.source_stage,
                            "target_stage": rel.target_stage,
                            "key_field": rel.key_field,
                            "relationship_type": rel.relationship_type.value,
                            "description": rel.description,
                            "transformation_rules": rel.transformation_rules
                        }
                        for rel in strategy.relationships
                    ],
                    "extraction_sequence": strategy.extraction_sequence,
                    "document_classifications": strategy.document_classifications,
                    "consolidation_rules": strategy.consolidation_rules
                },
                "created_at": "",
                "text_description": strategy.description,
                "parsed_fields": [],  # Will be populated during strategy generation
                "description_parsing_model": "claude-sonnet-4-20250514",
                "parsing_method": "fast"
            }
        }
        
        if additional_config:
            config["extraction_config"].update(additional_config)
        
        return config
    
    @staticmethod
    def load_strategy_from_config(config_path: str) -> ExtractionStrategy:
        """Load extraction strategy from config file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            extraction_config = config_data['extraction_config']
            strategy_data = extraction_config.get('extraction_strategy')
            
            if strategy_data is None:
                raise ValueError("extraction_strategy is null - use case needs to be converted to Case 2 format")
            
            # Reconstruct stages
            stages = []
            for stage_data in strategy_data['stages']:
                stage = ExtractionStage(
                    stage_name=stage_data['stage_name'],
                    stage_type=ExtractionStageType(stage_data['stage_type']),
                    document_types=stage_data['document_types'],
                    key_fields=stage_data.get('key_fields', []),
                    input_keys=stage_data.get('input_keys', []),
                    output_keys=stage_data.get('output_keys', []),
                    extraction_fields=stage_data.get('extraction_fields', []),
                    description=stage_data.get('description', '')
                )
                stages.append(stage)
            
            # Reconstruct relationships
            relationships = []
            for rel_data in strategy_data['relationships']:
                relationship = RelationshipMapping(
                    source_stage=rel_data['source_stage'],
                    target_stage=rel_data['target_stage'],
                    key_field=rel_data['key_field'],
                    relationship_type=RelationshipType(rel_data['relationship_type']),
                    description=rel_data['description'],
                    transformation_rules=rel_data.get('transformation_rules')
                )
                relationships.append(relationship)
            
            # Create strategy
            strategy = ExtractionStrategy(
                use_case_name=extraction_config['use_case'],
                description=extraction_config['description'],
                stages=stages,
                relationships=relationships,
                extraction_sequence=strategy_data['extraction_sequence'],
                document_classifications=strategy_data['document_classifications'],
                consolidation_rules=strategy_data.get('consolidation_rules')
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error loading strategy from config: {e}")
            raise

class DocumentClassifier:
    """Generic document classification for Case 2"""
    
    def __init__(self, ai_client=None):
        self.ai_client = ai_client
    
    def classify_documents(self, documents: List[Dict[str, Any]], 
                          document_types: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify documents into their types using AI-based classification
        No hard-coded patterns - purely AI-driven
        """
        if not self.ai_client:
            raise ValueError("AI client is required for document classification")
        
        try:
            # Create classification prompt
            prompt = self._create_classification_prompt(documents, document_types)
            
            # Get AI classification
            response = self.ai_client.generate_response(prompt)
            
            # Parse classification results
            classified_docs = self._parse_classification_response(response, documents)
            
            return classified_docs
            
        except Exception as e:
            logger.error(f"Error in document classification: {e}")
            raise
    
    def _create_classification_prompt(self, documents: List[Dict[str, Any]], 
                                    document_types: List[str]) -> str:
        """Create AI prompt for document classification"""
        doc_types_str = ", ".join(document_types)
        
        prompt = f"""
        Analyze the following documents and classify each one into one of these document types: {doc_types_str}
        
        For each document, determine its type based on its content, structure, and purpose.
        Consider the following factors:
        - Document structure and format
        - Content type and purpose
        - Key information present
        - Document context and usage
        
        DOCUMENTS TO CLASSIFY:
        """
        
        for i, doc in enumerate(documents, 1):
            prompt += f"\n--- Document {i} ---\n"
            prompt += f"Filename: {doc.get('filename', 'Unknown')}\n"
            prompt += f"Content Preview: {doc.get('content', '')[:500]}...\n"
        
        prompt += f"""
        
        Return your classification as a JSON object with this structure:
        {{
            "classifications": [
                {{
                    "document_index": 1,
                    "document_type": "document_type_name",
                    "confidence": 0.95,
                    "reasoning": "Brief explanation of why this document is classified as this type"
                }}
            ]
        }}
        
        Ensure that:
        1. Each document is classified into exactly one of the provided types: {doc_types_str}
        2. Confidence scores are between 0.0 and 1.0
        3. Reasoning explains the classification decision
        4. All document indices are included
        """
        
        return prompt
    
    def _parse_classification_response(self, response: str, 
                                     documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Parse AI response and organize documents by type"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in classification response")
            
            classification_data = json.loads(json_match.group())
            classifications = classification_data.get('classifications', [])
            
            # Organize documents by type
            classified_docs = {}
            for classification in classifications:
                doc_index = classification['document_index'] - 1  # Convert to 0-based index
                doc_type = classification['document_type']
                
                if 0 <= doc_index < len(documents):
                    doc = documents[doc_index].copy()
                    doc['classification_confidence'] = classification['confidence']
                    doc['classification_reasoning'] = classification['reasoning']
                    
                    if doc_type not in classified_docs:
                        classified_docs[doc_type] = []
                    classified_docs[doc_type].append(doc)
            
            return classified_docs
            
        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            # Fallback: distribute documents evenly across types
            return self._fallback_classification(documents, list(set([c.get('document_type', 'unknown') for c in classifications])))
    
    def _fallback_classification(self, documents: List[Dict[str, Any]], 
                               document_types: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback classification when parsing fails"""
        classified_docs = {}
        
        # Initialize empty lists for each document type
        for doc_type in document_types:
            classified_docs[doc_type] = []
        
        # Distribute documents evenly
        for i, doc in enumerate(documents):
            doc_type = document_types[i % len(document_types)]
            doc['classification_confidence'] = 0.5
            doc['classification_reasoning'] = "Fallback classification due to parsing error"
            classified_docs[doc_type].append(doc)
        
        return classified_docs
