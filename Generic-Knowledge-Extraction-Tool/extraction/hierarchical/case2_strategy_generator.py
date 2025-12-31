#!/usr/bin/env python3
"""
Case 2 Strategy Generator
AI-based analysis of text descriptions to create hierarchical extraction strategies
"""

import json
import logging
from typing import Dict, List, Any, Optional
from .case2_core import (
    ExtractionStrategy, ExtractionStage, RelationshipMapping, 
    RelationshipType, ExtractionStageType
)

logger = logging.getLogger(__name__)

class Case2StrategyGenerator:
    """Generates hierarchical extraction strategies from text descriptions"""
    
    def __init__(self, ai_client=None):
        self.ai_client = ai_client
        if not self.ai_client:
            raise ValueError("AI client is required for strategy generation")
    
    def generate_strategy(self, description: str, use_case_name: str = "") -> ExtractionStrategy:
        """
        Generate complete extraction strategy from text description
        No hard-coded patterns - purely AI-driven analysis
        """
        try:
            logger.info(f"Generating Case 2 strategy for: {use_case_name}")
            
            # Step 1: Analyze description to identify components
            analysis = self._analyze_description(description, use_case_name)
            
            # Step 2: Generate extraction stages
            stages = self._generate_stages(analysis)
            
            # Step 3: Identify relationships between stages
            relationships = self._generate_relationships(analysis, stages)
            
            # Step 4: Determine extraction sequence
            extraction_sequence = self._determine_extraction_sequence(stages, relationships)
            
            # Step 5: Create document classifications
            document_classifications = self._create_document_classifications(stages)
            
            # Step 6: Generate consolidation rules
            consolidation_rules = self._generate_consolidation_rules(analysis, stages)
            
            # Create complete strategy
            strategy = ExtractionStrategy(
                use_case_name=use_case_name,
                description=description,
                stages=stages,
                relationships=relationships,
                extraction_sequence=extraction_sequence,
                document_classifications=document_classifications,
                consolidation_rules=consolidation_rules
            )
            
            logger.info(f"Generated strategy with {len(stages)} stages and {len(relationships)} relationships")
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            raise
    
    def _analyze_description(self, description: str, use_case_name: str) -> Dict[str, Any]:
        """Analyze text description to identify extraction components"""
        prompt = f"""
        Analyze this hierarchical document extraction description and identify all components needed for a multi-stage extraction strategy.
        
        DESCRIPTION: {description}
        USE CASE: {use_case_name or "Hierarchical Document Extraction"}
        
        Identify the following components:
        1. Document types involved
        2. Key fields that link documents together
        3. Extraction stages and their sequence
        4. Data flow between stages
        5. Final consolidation requirements
        
        Return your analysis as a JSON object with this structure:
        {{
            "document_types": [
                {{
                    "type_name": "document_type",
                    "description": "What this document type contains",
                    "key_fields": ["field1", "field2"],
                    "extraction_fields": [
                        {{
                            "field_name": "field_name",
                            "field_type": "str|int|float|bool|list",
                            "description": "What this field contains",
                            "required": true
                        }}
                    ]
                }}
            ],
            "extraction_stages": [
                {{
                    "stage_name": "stage_name",
                    "stage_type": "initial|secondary|final",
                    "document_types": ["doc_type1", "doc_type2"],
                    "description": "What this stage extracts",
                    "key_fields": ["field1", "field2"],
                    "extraction_fields": [
                        {{
                            "field_name": "field_name",
                            "field_type": "str|int|float|bool|list",
                            "description": "What this field contains",
                            "required": true
                        }}
                    ]
                }}
            ],
            "relationships": [
                {{
                    "source_stage": "stage_name",
                    "target_stage": "stage_name",
                    "key_field": "field_name",
                    "relationship_type": "direct_key_match|derived_key_match|aggregation|transformation",
                    "description": "How these stages are related"
                }}
            ],
            "extraction_sequence": ["stage1", "stage2", "stage3"],
            "consolidation_requirements": {{
                "final_output_format": "description of final output",
                "key_consolidation_fields": ["field1", "field2"],
                "aggregation_rules": "any aggregation rules needed"
            }}
        }}
        
        Guidelines:
        - Be thorough in identifying all document types and their fields
        - Clearly define the sequence of extraction stages
        - Identify all relationships and key fields that link stages
        - Consider the logical flow of information from initial documents to final consolidated results
        - Ensure each stage has clear input and output specifications
        """
        
        try:
            # Use the adapter's structured response method for JSON output
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in document processing and data extraction workflows. Return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response_text = self.ai_client.generate_structured_response(
                messages=messages,
                max_tokens=self.ai_client.max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"}  # OpenAI-specific parameter
            )
            analysis = json.loads(response_text)
            
            # Validate analysis structure
            self._validate_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing description: {e}")
            raise
    
    def _validate_analysis(self, analysis: Dict[str, Any]) -> None:
        """Validate the structure of analysis results"""
        required_keys = ['document_types', 'extraction_stages', 'relationships', 'extraction_sequence']
        
        for key in required_keys:
            if key not in analysis:
                raise ValueError(f"Missing required key in analysis: {key}")
        
        if not isinstance(analysis['document_types'], list):
            raise ValueError("document_types must be a list")
        
        if not isinstance(analysis['extraction_stages'], list):
            raise ValueError("extraction_stages must be a list")
        
        if not isinstance(analysis['relationships'], list):
            raise ValueError("relationships must be a list")
        
        if not isinstance(analysis['extraction_sequence'], list):
            raise ValueError("extraction_sequence must be a list")
    
    def _generate_stages(self, analysis: Dict[str, Any]) -> List[ExtractionStage]:
        """Generate extraction stages from analysis"""
        stages = []
        
        for stage_data in analysis['extraction_stages']:
            # Determine stage type
            stage_type_str = stage_data.get('stage_type', 'secondary')
            if stage_type_str == 'initial':
                stage_type = ExtractionStageType.INITIAL
            elif stage_type_str == 'final':
                stage_type = ExtractionStageType.FINAL
            else:
                stage_type = ExtractionStageType.SECONDARY
            
            # Create stage
            stage = ExtractionStage(
                stage_name=stage_data['stage_name'],
                stage_type=stage_type,
                document_types=stage_data['document_types'],
                key_fields=stage_data.get('key_fields', []),
                extraction_fields=stage_data.get('extraction_fields', []),
                description=stage_data.get('description', '')
            )
            
            stages.append(stage)
        
        return stages
    
    def _generate_relationships(self, analysis: Dict[str, Any], 
                              stages: List[ExtractionStage]) -> List[RelationshipMapping]:
        """Generate relationships between stages"""
        relationships = []
        
        for rel_data in analysis['relationships']:
            # Determine relationship type
            rel_type_str = rel_data.get('relationship_type', 'direct_key_match')
            try:
                relationship_type = RelationshipType(rel_type_str)
            except ValueError:
                relationship_type = RelationshipType.DIRECT_KEY_MATCH
            
            # Create relationship
            relationship = RelationshipMapping(
                source_stage=rel_data['source_stage'],
                target_stage=rel_data['target_stage'],
                key_field=rel_data['key_field'],
                relationship_type=relationship_type,
                description=rel_data.get('description', '')
            )
            
            relationships.append(relationship)
        
        return relationships
    
    def _determine_extraction_sequence(self, stages: List[ExtractionStage], 
                                     relationships: List[RelationshipMapping]) -> List[str]:
        """Determine the optimal extraction sequence"""
        # Use the sequence from analysis, but validate it
        sequence = []
        
        # Start with initial stages
        initial_stages = [stage.stage_name for stage in stages if stage.stage_type == ExtractionStageType.INITIAL]
        sequence.extend(initial_stages)
        
        # Add secondary stages based on relationships
        remaining_stages = [stage.stage_name for stage in stages if stage.stage_type == ExtractionStageType.SECONDARY]
        
        # Build sequence based on dependencies
        while remaining_stages:
            for stage_name in remaining_stages[:]:  # Copy to avoid modification during iteration
                # Check if all dependencies are satisfied
                dependencies_satisfied = True
                for rel in relationships:
                    if rel.target_stage == stage_name and rel.source_stage not in sequence:
                        dependencies_satisfied = False
                        break
                
                if dependencies_satisfied:
                    sequence.append(stage_name)
                    remaining_stages.remove(stage_name)
                    break
            else:
                # If no stage can be added, add remaining stages in order
                sequence.extend(remaining_stages)
                break
        
        # Add final stages
        final_stages = [stage.stage_name for stage in stages if stage.stage_type == ExtractionStageType.FINAL]
        sequence.extend(final_stages)
        
        return sequence
    
    def _create_document_classifications(self, stages: List[ExtractionStage]) -> Dict[str, str]:
        """Create document type to stage mapping"""
        classifications = {}
        
        for stage in stages:
            for doc_type in stage.document_types:
                classifications[doc_type] = stage.stage_name
        
        return classifications
    
    def _generate_consolidation_rules(self, analysis: Dict[str, Any], 
                                    stages: List[ExtractionStage]) -> Dict[str, Any]:
        """Generate consolidation rules for final output"""
        consolidation_data = analysis.get('consolidation_requirements', {})
        
        rules = {
            'final_output_format': consolidation_data.get('final_output_format', 'structured_data'),
            'key_consolidation_fields': consolidation_data.get('key_consolidation_fields', []),
            'aggregation_rules': consolidation_data.get('aggregation_rules', ''),
            'output_structure': 'hierarchical'  # Default for Case 2
        }
        
        return rules
    
    def validate_strategy(self, strategy: ExtractionStrategy) -> List[str]:
        """Validate generated strategy and return any issues"""
        issues = []
        
        # Check for duplicate stage names
        stage_names = [stage.stage_name for stage in strategy.stages]
        if len(stage_names) != len(set(stage_names)):
            issues.append("Duplicate stage names found")
        
        # Check for duplicate document types
        all_doc_types = []
        for stage in strategy.stages:
            all_doc_types.extend(stage.document_types)
        
        if len(all_doc_types) != len(set(all_doc_types)):
            issues.append("Document types assigned to multiple stages")
        
        # Check relationship validity
        valid_stage_names = set(stage_names)
        for rel in strategy.relationships:
            if rel.source_stage not in valid_stage_names:
                issues.append(f"Relationship references invalid source stage: {rel.source_stage}")
            if rel.target_stage not in valid_stage_names:
                issues.append(f"Relationship references invalid target stage: {rel.target_stage}")
        
        # Check extraction sequence validity
        for stage_name in strategy.extraction_sequence:
            if stage_name not in valid_stage_names:
                issues.append(f"Extraction sequence references invalid stage: {stage_name}")
        
        return issues
    
    def optimize_strategy(self, strategy: ExtractionStrategy) -> ExtractionStrategy:
        """Optimize strategy for better performance and clarity"""
        # This could include:
        # - Reordering stages for better efficiency
        # - Merging similar stages
        # - Optimizing relationship patterns
        # For now, return the strategy as-is
        return strategy
