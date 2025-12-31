#!/usr/bin/env python3
"""
Case 2 Sequential Extraction Engine
Orchestrates hierarchical multi-stage document extraction with data flow
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from .case2_core import (
    ExtractionStrategy, StageExtractor, HierarchicalExtractionResult,
    DocumentClassifier, RelationshipMapping
)
from utils.messaging_system import get_messenger

logger = logging.getLogger(__name__)

class Case2RelationshipManager:
    """Manages data flow and relationships between extraction stages"""
    
    def __init__(self):
        self.stage_results = {}
        self.relationship_cache = {}
    
    def process_stage_results(self, stage_name: str, results: List[Dict[str, Any]], 
                            relationships: List[RelationshipMapping]) -> Dict[str, Any]:
        """Process stage results and prepare data for next stages"""
        try:
            # Store results for this stage
            self.stage_results[stage_name] = results
            
            # Build relationship data
            relationship_data = {}
            
            for rel in relationships:
                if rel.source_stage == stage_name:
                    # This stage produces data for other stages
                    key_data = self._extract_key_data(results, rel.key_field)
                    relationship_data[rel.target_stage] = {
                        'key_field': rel.key_field,
                        'key_values': key_data,
                        'relationship_type': rel.relationship_type.value,
                        'description': rel.description
                    }
            
            # Cache relationship data
            self.relationship_cache[stage_name] = relationship_data
            
            return relationship_data
            
        except Exception as e:
            logger.error(f"Error processing stage results for {stage_name}: {e}")
            return {}
    
    def get_input_data_for_stage(self, target_stage: str, 
                               relationships: List[RelationshipMapping]) -> Dict[str, Any]:
        """Get input data for a target stage from previous stages"""
        input_data = {}
        
        for rel in relationships:
            if rel.target_stage == target_stage:
                source_stage_results = self.stage_results.get(rel.source_stage, [])
                if source_stage_results:
                    # Extract key values from source stage results
                    key_values = self._extract_key_data(source_stage_results, rel.key_field)
                    if key_values:
                        input_data[rel.key_field] = key_values
                        logger.info(f"Input data for {target_stage} from {rel.source_stage}: {rel.key_field} = {key_values}")
        
        return input_data
    
    def _extract_key_data(self, results: List[Dict[str, Any]], key_field: str) -> List[str]:
        """Extract key field values from stage results - find the best matching field name"""
        key_values = []
        
        for result in results:
            # First try exact match
            if key_field in result:
                key_values.append(str(result[key_field]))
                continue
                
            # If no exact match, find the best matching field from available keys
            available_keys = list(result.keys())
            matching_key = self._find_matching_key(key_field, available_keys)
            
            if matching_key:
                key_values.append(str(result[matching_key]))
        
        return key_values
    
    def _find_matching_key(self, target_key: str, available_keys: List[str]) -> str:
        """Find the best matching key from available keys without hardcoding"""
        # Direct match
        if target_key in available_keys:
            return target_key
            
        # Case-insensitive match
        target_lower = target_key.lower()
        for key in available_keys:
            if key.lower() == target_lower:
                return key
                
        # Match ignoring spaces, underscores, and case
        target_normalized = target_lower.replace(' ', '').replace('_', '').replace('-', '')
        for key in available_keys:
            key_normalized = key.lower().replace(' ', '').replace('_', '').replace('-', '')
            if key_normalized == target_normalized:
                return key
                
        # Partial match - if target contains words that match key
        target_words = set(target_lower.replace('_', ' ').replace('-', ' ').split())
        for key in available_keys:
            key_words = set(key.lower().replace('_', ' ').replace('-', ' ').split())
            if target_words.intersection(key_words):  # If they share any words
                return key
                
        return None
    
    def consolidate_results(self, strategy: ExtractionStrategy) -> List[Dict[str, Any]]:
        """Consolidate results from all stages into final output"""
        try:
            # For hierarchical extraction, the final stage contains the consolidated results
            # Find the final stage (stage_type = 'final')
            final_stage_name = None
            for stage in strategy.stages:
                if stage.stage_type.value == 'final':
                    final_stage_name = stage.stage_name
                    break
            
            if final_stage_name and final_stage_name in self.stage_results:
                consolidated = self.stage_results[final_stage_name]
                logger.info(f"Using final stage '{final_stage_name}' results as consolidated output: {len(consolidated)} records")
                return consolidated
            
            # Fallback: if no final stage, try the last stage in sequence
            if strategy.extraction_sequence and self.stage_results:
                last_stage_name = strategy.extraction_sequence[-1]
                if last_stage_name in self.stage_results:
                    consolidated = self.stage_results[last_stage_name]
                    logger.info(f"Using last stage '{last_stage_name}' results as consolidated output: {len(consolidated)} records")
                    return consolidated
            
            # Ultimate fallback: complex cross-stage consolidation (original logic)
            logger.warning("No final stage found, attempting cross-stage consolidation")
            return self._complex_consolidation(strategy)
            
        except Exception as e:
            logger.error(f"Error consolidating results: {e}")
            return []
    
    def _complex_consolidation(self, strategy: ExtractionStrategy) -> List[Dict[str, Any]]:
        """Complex cross-stage consolidation (original logic as fallback)"""
        try:
            consolidated = []
            
            # Get all unique key combinations
            key_combinations = self._get_key_combinations(strategy)
            
            for key_combo in key_combinations:
                consolidated_record = {}
                
                # Add data from each stage
                for stage in strategy.stages:
                    stage_data = self._get_stage_data_for_keys(stage.stage_name, key_combo)
                    if stage_data:
                        consolidated_record.update(stage_data)
                
                if consolidated_record:
                    consolidated.append(consolidated_record)
            
            return consolidated
            
        except Exception as e:
            logger.error(f"Error in complex consolidation: {e}")
            return []
    
    def _get_key_combinations(self, strategy: ExtractionStrategy) -> List[Dict[str, str]]:
        """Get all unique key combinations across stages"""
        key_combinations = []
        
        # Start with initial stage results
        initial_stages = [stage for stage in strategy.stages if stage.stage_type.value == 'initial']
        
        for stage in initial_stages:
            stage_results = self.stage_results.get(stage.stage_name, [])
            for result in stage_results:
                key_combo = {}
                for key_field in stage.key_fields:
                    if key_field in result:
                        key_combo[key_field] = str(result[key_field])
                
                if key_combo and key_combo not in key_combinations:
                    key_combinations.append(key_combo)
        
        return key_combinations
    
    def _get_stage_data_for_keys(self, stage_name: str, key_combo: Dict[str, str]) -> Dict[str, Any]:
        """Get stage data that matches the key combination"""
        stage_results = self.stage_results.get(stage_name, [])
        
        for result in stage_results:
            match = True
            for key_field, key_value in key_combo.items():
                if key_field not in result or str(result[key_field]) != key_value:
                    match = False
                    break
            
            if match:
                return result
        
        return {}

class Case2SequentialExtractor:
    """Main orchestrator for Case 2 hierarchical extraction"""
    
    def __init__(self, strategy: ExtractionStrategy, stage_extractors: List[StageExtractor], 
                 ai_client=None):
        self.strategy = strategy
        self.stage_extractors = stage_extractors
        self.ai_client = ai_client
        self.relationship_manager = Case2RelationshipManager()
        self.document_classifier = DocumentClassifier(ai_client)
        
        if not self.ai_client:
            raise ValueError("AI client is required for extraction")
    
    def extract_hierarchically(self, documents: List[Dict[str, Any]]) -> HierarchicalExtractionResult:
        """
        Perform complete hierarchical extraction process
        No hard-coded patterns - purely AI-driven extraction
        """
        try:
            logger.info(f"Starting hierarchical extraction for {len(documents)} documents")
            
            # Step 1: Classify documents by type
            classified_docs = self._classify_and_route_documents(documents)
            
            # Step 2: Execute stages in sequence
            stage_results = {}
            all_relationships = {}
            
            for stage_name in self.strategy.extraction_sequence:
                logger.info(f"Executing stage: {stage_name}")
                
                # Get stage extractor
                stage_extractor = self._get_stage_extractor(stage_name)
                if not stage_extractor:
                    logger.warning(f"No extractor found for stage: {stage_name}")
                    continue
                
                # Get documents for this stage
                stage_documents = self._get_documents_for_stage(stage_name, classified_docs)
                
                # Get input data from previous stages
                input_data = self.relationship_manager.get_input_data_for_stage(
                    stage_name, self.strategy.relationships
                )
                
                # Execute stage extraction
                stage_result = self._execute_stage(stage_extractor, stage_documents, input_data)
                
                # Process results and relationships
                relationship_data = self.relationship_manager.process_stage_results(
                    stage_name, stage_result, self.strategy.relationships
                )
                
                stage_results[stage_name] = stage_result
                all_relationships[stage_name] = relationship_data
            
            # Step 3: Consolidate final results
            consolidated_results = self.relationship_manager.consolidate_results(self.strategy)
            
            # Step 4: Create final result
            result = HierarchicalExtractionResult(
                use_case_name=self.strategy.use_case_name,
                extraction_strategy=self.strategy,
                stage_results=stage_results,
                relationships=all_relationships,
                consolidated_results=consolidated_results,
                processing_metadata={
                    'total_documents': len(documents),
                    'stages_executed': len(stage_results),
                    'total_extracted_records': sum(len(results) for results in stage_results.values())
                }
            )
            
            logger.info(f"Hierarchical extraction completed: {len(consolidated_results)} consolidated records")
            return result
            
        except Exception as e:
            logger.error(f"Error in hierarchical extraction: {e}")
            raise
    
    def _classify_and_route_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Classify documents and route them to appropriate stages using Case 1 approach"""
        try:
            # Get all document types from strategy
            all_doc_types = []
            for stage in self.strategy.stages:
                all_doc_types.extend(stage.document_types)
            
            unique_doc_types = list(set(all_doc_types))
            
            # Use Case 1 classification approach with raw AI client
            from extraction.case1_classifier import DocumentTypeClassifier
            
            # Create document types dict for Case 1 classifier (format: {doc_type: description})
            extracted_document_types = {doc_type: f"Documents of type {doc_type}" for doc_type in unique_doc_types}
            
            # Create Case 1 classifier - need to pass raw client, not adapter
            raw_client = getattr(self.ai_client, 'client', self.ai_client)
            if hasattr(raw_client, 'chat') and hasattr(raw_client.chat, 'completions'):
                # Pass raw client that has .chat.completions interface
                classifier = DocumentTypeClassifier(extracted_document_types, raw_client)
                classified_docs = classifier.classify_documents(documents)
                
                # IMPORTANT: Add classification_type to each document for Case 2 compatibility
                for doc_type, doc_list in classified_docs.items():
                    for doc in doc_list:
                        doc['classification_type'] = doc_type
                
                logger.info(f"Document classification completed using Case 1 approach: {len(classified_docs)} types identified")
                return classified_docs
            else:
                logger.warning("Cannot use Case 1 classifier - falling back to simple routing")
                return self._fallback_document_routing(documents)
            
        except Exception as e:
            logger.error(f"Error in document classification: {e}")
            # Fallback: distribute documents evenly
            return self._fallback_document_routing(documents)
    
    def _fallback_document_routing(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback document routing when classification fails"""
        classified_docs = {}
        
        # Get document types from strategy
        all_doc_types = []
        for stage in self.strategy.stages:
            all_doc_types.extend(stage.document_types)
        
        unique_doc_types = list(set(all_doc_types))
        
        # Distribute documents evenly
        for i, doc in enumerate(documents):
            doc_type = unique_doc_types[i % len(unique_doc_types)]
            if doc_type not in classified_docs:
                classified_docs[doc_type] = []
            
            # Add classification_type to document for Case 2 compatibility
            doc['classification_type'] = doc_type
            classified_docs[doc_type].append(doc)
        
        return classified_docs
    
    def _get_stage_extractor(self, stage_name: str) -> Optional[StageExtractor]:
        """Get stage extractor by name"""
        for extractor in self.stage_extractors:
            if extractor.stage_name == stage_name:
                return extractor
        return None
    
    def _get_documents_for_stage(self, stage_name: str, 
                                classified_docs: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Get documents assigned to a specific stage"""
        # Find stage by name
        stage = None
        for s in self.strategy.stages:
            if s.stage_name == stage_name:
                stage = s
                break
        
        if not stage:
            return []
        
        # Collect documents for this stage's document types
        stage_documents = []
        for doc_type in stage.document_types:
            if doc_type in classified_docs:
                stage_documents.extend(classified_docs[doc_type])
        
        return stage_documents
    
    def _execute_stage(self, stage_extractor: StageExtractor, 
                      documents: List[Dict[str, Any]], 
                      input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a single extraction stage"""
        try:
            if not documents:
                logger.warning(f"No documents for stage: {stage_extractor.stage_name}")
                return []
            
            # Initialize messaging system for stage progress
            messenger = get_messenger()
            
            # Prepare documents with type headers
            prepared_documents = self._prepare_documents_with_headers(documents, stage_extractor)
            
            # Create extraction prompt with input data
            full_prompt = self._create_stage_prompt(stage_extractor, prepared_documents, input_data)
            
            # Log stage execution details
            logger.info(f"Executing stage: {stage_extractor.stage_name} with {len(documents)} documents")
            
            # Show stage progress
            messenger.extraction_progress(stage_extractor.stage_name, len(documents), 0)
            
            # Execute extraction
            response = self.ai_client.generate_response(full_prompt)
            
            # Log AI response summary
            logger.info(f"Stage {stage_extractor.stage_name}: AI response received ({len(response)} chars)")
            
            # Parse results
            results = self._parse_stage_results(response, stage_extractor.model_class)
            
            # Log parsed results summary
            logger.info(f"Stage {stage_extractor.stage_name}: Parsed {len(results)} results")
            
            # Update progress with results
            messenger.extraction_progress(stage_extractor.stage_name, len(documents), len(results))
            
            logger.info(f"Stage {stage_extractor.stage_name} completed: {len(results)} records extracted")
            return results
            
        except Exception as e:
            logger.error(f"Error executing stage {stage_extractor.stage_name}: {e}")
            return []
    
    def _prepare_documents_with_headers(self, documents: List[Dict[str, Any]], 
                                       stage_extractor: StageExtractor) -> str:
        """Prepare documents with type headers for extraction"""
        prepared_text = ""
        
        for i, doc in enumerate(documents, 1):
            doc_type = doc.get('classification_type', 'unknown')
            prepared_text += f"\n--- Document {i} (Type: {doc_type}) ---\n"
            
            # Try multiple filename keys
            filename = doc.get('filename') or doc.get('file_name') or 'Unknown'
            prepared_text += f"Filename: {filename}\n"
            
            # Try multiple content keys - Case 1 uses 'text_content', Case 2 might use 'content'
            content = doc.get('content') or doc.get('text_content') or ''
            prepared_text += f"Content:\n{content}\n"
            prepared_text += f"--- End Document {i} ---\n"
        
        return prepared_text
    
    def _create_stage_prompt(self, stage_extractor: StageExtractor, 
                           prepared_documents: str, 
                           input_data: Dict[str, Any]) -> str:
        """Create complete prompt for stage extraction with proper input data integration"""
        
        # Build input data section with actionable instructions
        input_data_section = ""
        if input_data:
            input_data_section = "\n=== INPUT DATA FROM PREVIOUS STAGES ===\n"
            for key, values in input_data.items():
                value_list = ', '.join(values) if values else 'None'
                input_data_section += f"- {key}: [{value_list}]\n"
            
            # Add actionable instructions based on input data
            input_data_section += "\nUSE THIS INPUT DATA TO:\n"
            if stage_extractor.stage_type.value == 'secondary':
                input_data_section += "- Focus extraction on items that match these input values\n"
                input_data_section += "- Extract details only for the specified items above\n"
            elif stage_extractor.stage_type.value == 'final':
                input_data_section += "- Combine and align data using these key values as the linking mechanism\n"
                input_data_section += "- Ensure each output record corresponds to one of these input values\n"
            input_data_section += "\n"
        
        # Create full prompt using the stage's embedded prompt
        full_prompt = f"""
{stage_extractor.prompt}
{input_data_section}
DOCUMENTS TO PROCESS:
{prepared_documents}

IMPORTANT: Return ONLY the JSON object/array, no additional text, explanations, or markdown formatting.
        """
        
        return full_prompt
    
    def _parse_stage_results(self, response: str, model_class: type) -> List[Dict[str, Any]]:
        """Parse AI response into structured results with Pydantic validation"""
        try:
            import re
            import json
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                # Try single object
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    single_result = json.loads(json_match.group())
                    # Validate with Pydantic model
                    try:
                        validated_result = model_class(**single_result)
                        return [validated_result.model_dump()]
                    except Exception as validation_error:
                        logger.warning(f"Pydantic validation failed: {validation_error}. Using raw data.")
                        return [single_result]
                else:
                    raise ValueError("No JSON found in response")
            
            results = json.loads(json_match.group())
            
            # Validate results
            if not isinstance(results, list):
                results = [results]
            
            # Validate each result with Pydantic model
            validated_results = []
            for result in results:
                try:
                    validated_result = model_class(**result)
                    validated_results.append(validated_result.model_dump())
                except Exception as validation_error:
                    logger.warning(f"Pydantic validation failed for record: {validation_error}. Using raw data.")
                    validated_results.append(result)
            
            return validated_results
            
        except Exception as e:
            logger.error(f"Error parsing stage results: {e}")
            return []
    
    def get_extraction_summary(self, result: HierarchicalExtractionResult) -> Dict[str, Any]:
        """Get summary of extraction results"""
        summary = {
            'use_case': result.use_case_name,
            'total_documents': result.processing_metadata['total_documents'],
            'stages_executed': result.processing_metadata['stages_executed'],
            'total_records': result.processing_metadata['total_extracted_records'],
            'consolidated_records': len(result.consolidated_results),
            'stage_breakdown': {}
        }
        
        for stage_name, stage_results in result.stage_results.items():
            summary['stage_breakdown'][stage_name] = {
                'records_extracted': len(stage_results),
                'document_types': self._get_stage_document_types(stage_name)
            }
        
        return summary
    
    def _get_stage_document_types(self, stage_name: str) -> List[str]:
        """Get document types for a stage"""
        for stage in self.strategy.stages:
            if stage.stage_name == stage_name:
                return stage.document_types
        return []
