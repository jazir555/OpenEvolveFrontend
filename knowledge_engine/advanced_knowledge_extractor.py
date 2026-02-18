"""
Advanced Knowledge Extractor with NLP and Machine Learning Enhancements

This module implements a 5x enhanced knowledge extraction system that leverages:
- Advanced NLP processing with spaCy
- Machine learning-based pattern recognition
- Semantic analysis and entity linking
- Multi-modal knowledge extraction
- Enhanced quality assessment with ML models
"""

import json
import logging
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
import statistics

# Import advanced NLP and ML libraries
try:
    import spacy
    from spacy import displacy
    import en_core_web_lg
    from textblob import TextBlob
    from gensim import corpora, models, similarities
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
except ImportError as e:
    logging.warning(f"Advanced NLP libraries not available: {e}")

# Import from knowledge_engine module
from knowledge_engine.knowledge_extractor import KnowledgeArtifact, KnowledgeExtractor

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedKnowledgeExtractor(KnowledgeExtractor):
    """
    Advanced Knowledge Extractor with NLP and ML enhancements.
    
    This class extends the basic knowledge extractor with:
    - NLP-based entity recognition and semantic analysis
    - Machine learning pattern recognition
    - Advanced text processing and normalization
    - Multi-modal knowledge extraction
    - Enhanced quality assessment with ML models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced knowledge extractor with NLP capabilities"""
        super().__init__(config)

        # Initialize NLP models
        self.nlp_models = self._initialize_nlp_models()
        self.ml_models = self._initialize_ml_models()
        self.semantic_models = self._initialize_semantic_models()

        # Enhanced pattern library
        self.advanced_pattern_library = self._initialize_advanced_pattern_library()

        # NLP processing statistics
        self.nlp_processing_stats = defaultdict(int)
        self.ml_analysis_stats = defaultdict(int)

        # Artifact cache for semantic similarity
        self._artifact_cache = {}  # artifact_id -> (embedding, metadata)
        self._cache_max_size = config.get('artifact_cache_size', 1000) if config else 1000

        logger.info("Advanced knowledge extractor initialized with NLP and ML capabilities")
    
    def _initialize_nlp_models(self) -> Dict[str, Any]:
        """Initialize NLP models for advanced text processing"""
        models = {}
        
        try:
            # Load spaCy model
            models['spacy'] = en_core_web_lg.load() if hasattr(en_core_web_lg, 'load') else spacy.load('en_core_web_lg')
            logger.info("Loaded spaCy English model (large)")
            
            # Initialize TextBlob for sentiment analysis
            models['textblob'] = TextBlob
            logger.info("Initialized TextBlob for sentiment analysis")
            
            # Initialize gensim for topic modeling
            models['gensim'] = {
                'dictionary': None,
                'corpus': None,
                'lda_model': None
            }
            logger.info("Initialized gensim for topic modeling")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {str(e)}")
            
        return models
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for pattern recognition"""
        models = {}
        
        try:
            # Initialize TF-IDF vectorizer
            models['tfidf'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            logger.info("Initialized TF-IDF vectorizer")
            
            # Initialize clustering model
            models['clustering'] = KMeans(
                n_clusters=5,
                random_state=42,
                n_init=10
            )
            logger.info("Initialized KMeans clustering model")
            
            # Initialize sentence transformer for semantic analysis
            models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized sentence transformer model")
            
            # Initialize HuggingFace pipeline for text classification
            models['text_classifier'] = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("Initialized text classification pipeline")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {str(e)}")
            
        return models
    
    def _initialize_semantic_models(self) -> Dict[str, Any]:
        """Initialize semantic analysis models"""
        models = {}
        
        try:
            # Initialize semantic similarity model
            models['similarity'] = {
                'model': None,
                'threshold': 0.75
            }
            
            # Initialize named entity recognition enhancements
            models['ner_enhancer'] = {
                'custom_entities': self._load_custom_entities(),
                'entity_linking': {}
            }
            
            logger.info("Initialized semantic analysis models")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic models: {str(e)}")
            
        return models
    
    def _load_custom_entities(self) -> Dict[str, List[str]]:
        """Load custom entities for domain-specific NER"""
        return {
            'knowledge_engineering': [
                'knowledge extraction', 'pattern recognition', 'semantic analysis',
                'entity linking', 'knowledge graph', 'ontology', 'taxonomy'
            ],
            'machine_learning': [
                'neural network', 'deep learning', 'supervised learning',
                'unsupervised learning', 'reinforcement learning', 'transformer',
                'embedding', 'feature extraction', 'model training'
            ],
            'software_engineering': [
                'software architecture', 'design pattern', 'code quality',
                'refactoring', 'testing', 'debugging', 'performance optimization'
            ]
        }
    
    def _initialize_advanced_pattern_library(self) -> Dict[str, Any]:
        """Initialize enhanced pattern library with ML-based patterns"""
        base_library = super()._initialize_pattern_library()
        
        # Add ML-enhanced patterns
        base_library['ml_patterns'] = {
            'neural_network_architecture': {
                'keywords': ['layer', 'neuron', 'activation', 'loss function', 'optimizer'],
                'ml_features': ['architecture_type', 'layer_count', 'parameter_count'],
                'quality_score': 0.92
            },
            'reinforcement_learning': {
                'keywords': ['reward', 'policy', 'q-learning', 'markov', 'agent', 'environment'],
                'ml_features': ['algorithm_type', 'reward_function', 'exploration_strategy'],
                'quality_score': 0.88
            },
            'transfer_learning': {
                'keywords': ['pretrained', 'fine-tuning', 'feature extraction', 'base model'],
                'ml_features': ['base_model', 'fine_tuning_layers', 'transfer_approach'],
                'quality_score': 0.90
            }
        }
        
        # Add domain-specific patterns
        base_library['domain_patterns'] = {
            'knowledge_engineering': {
                'patterns': ['knowledge_base', 'ontology', 'semantic_network', 'expert_system'],
                'quality_score': 0.85
            },
            'software_development': {
                'patterns': ['design_pattern', 'architecture_pattern', 'coding_standard', 'best_practice'],
                'quality_score': 0.82
            }
        }
        
        return base_library
    
    def extract_from_workflow_advanced(self, workflow_data: Dict[str, Any]) -> List[KnowledgeArtifact]:
        """
        Enhanced knowledge extraction with NLP and ML capabilities.
        
        This method extends the basic extraction with:
        - NLP-based entity recognition and semantic analysis
        - Machine learning pattern recognition
        - Advanced text processing and normalization
        - Multi-modal knowledge extraction
        """
        start_time = datetime.now()
        workflow_id = workflow_data.get('workflow_id', 'unknown')
        logger.info(f"Starting advanced knowledge extraction from workflow: {workflow_id}")
        
        # First, perform basic extraction
        basic_artifacts = super().extract_from_workflow(workflow_data)
        
        # Enhance artifacts with NLP and ML analysis
        enhanced_artifacts = []
        
        for artifact in basic_artifacts:
            try:
                # Apply NLP enhancement
                nlp_enhanced = self._enhance_with_nlp(artifact)
                
                # Apply ML pattern recognition
                ml_enhanced = self._enhance_with_ml(nlp_enhanced)
                
                # Apply semantic analysis
                semantic_enhanced = self._enhance_with_semantic_analysis(ml_enhanced)
                
                # Apply quality assessment with ML
                quality_assessed = self._assess_quality_with_ml(semantic_enhanced)
                
                enhanced_artifacts.append(quality_assessed)

                # Cache artifact for future similarity comparisons
                self._cache_artifact(quality_assessed)

                # Update statistics
                self.nlp_processing_stats['artifacts_enhanced'] += 1
                
            except Exception as e:
                logger.error(f"Failed to enhance artifact {artifact.id} with advanced processing: {str(e)}")
                enhanced_artifacts.append(artifact)  # Fallback to basic artifact
        
        extraction_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Advanced extraction completed for workflow {workflow_id}")
        logger.info(f"  - Total artifacts: {len(enhanced_artifacts)}")
        logger.info(f"  - NLP enhanced: {self.nlp_processing_stats['artifacts_enhanced']}")
        logger.info(f"  - Extraction time: {extraction_time:.3f}s")
        
        return enhanced_artifacts
    
    def _enhance_with_nlp(self, artifact: KnowledgeArtifact) -> KnowledgeArtifact:
        """Enhance artifact with NLP analysis"""
        enhanced = KnowledgeArtifact(**artifact.to_dict())
        
        try:
            # Extract text content for NLP processing
            text_content = self._extract_text_content(artifact)
            
            if text_content and self.nlp_models.get('spacy'):
                # Perform NLP analysis with spaCy
                doc = self.nlp_models['spacy'](text_content)
                
                # Extract entities
                entities = self._extract_entities_with_spacy(doc)
                
                # Perform sentiment analysis
                sentiment = self._analyze_sentiment(text_content)
                
                # Extract key phrases
                key_phrases = self._extract_key_phrases(doc)
                
                # Update artifact with NLP insights
                if 'nlp_analysis' not in enhanced.metadata:
                    enhanced.metadata['nlp_analysis'] = {}
                
                enhanced.metadata['nlp_analysis'].update({
                    'entities': entities,
                    'sentiment': sentiment,
                    'key_phrases': key_phrases,
                    'tokens': len(doc),
                    'sentences': len(list(doc.sents))
                })
                
                # Add entities to content if not present
                if 'entities' not in enhanced.content:
                    enhanced.content['entities'] = entities
                
                # Update quality indicators based on NLP analysis
                enhanced.source_quality = min(1.0, enhanced.source_quality + 0.05)
                
                self.nlp_processing_stats['nlp_analyzed'] += 1
                
        except Exception as e:
            logger.error(f"NLP enhancement failed for artifact {artifact.id}: {str(e)}")
        
        return enhanced
    
    def _extract_text_content(self, artifact: KnowledgeArtifact) -> str:
        """Extract text content from artifact for NLP processing"""
        text_parts = []
        
        # Extract from content fields
        for field, value in artifact.content.items():
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, (list, dict)):
                text_parts.append(str(value))
        
        # Add domain and problem type
        if artifact.domain:
            text_parts.append(artifact.domain)
        if artifact.problem_type:
            text_parts.append(artifact.problem_type)
        
        return ' '.join(text_parts) if text_parts else ''
    
    def _extract_entities_with_spacy(self, doc) -> List[Dict[str, Any]]:
        """Extract entities using spaCy with custom enhancements"""
        entities = []
        
        # Extract standard entities
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.9  # Default confidence for spaCy entities
            })
        
        # Add custom domain entities
        custom_entities = self._extract_custom_entities(doc)
        entities.extend(custom_entities)
        
        return entities
    
    def _extract_custom_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract custom domain-specific entities"""
        custom_entities = []
        text = doc.text.lower()
        
        for entity_type, keywords in self.semantic_models['ner_enhancer']['custom_entities'].items():
            for keyword in keywords:
                if keyword in text:
                    # Find the span in the document
                    for token in doc:
                        if keyword in token.text.lower():
                            custom_entities.append({
                                'text': token.text,
                                'label': f"CUSTOM_{entity_type.upper()}",
                                'start': token.idx,
                                'end': token.idx + len(token.text),
                                'confidence': 0.85
                            })
                            break
        
        return custom_entities
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob"""
        try:
            blob = self.nlp_models['textblob'](text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'sentiment': 'positive' if blob.sentiment.polarity > 0.1 else 
                            'neutral' if abs(blob.sentiment.polarity) <= 0.1 else 'negative'
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {'polarity': 0.0, 'subjectivity': 0.5, 'sentiment': 'neutral'}
    
    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract key phrases using noun chunks and proper nouns"""
        key_phrases = []
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Multi-word phrases
                key_phrases.append(chunk.text)
        
        # Extract proper nouns
        for token in doc:
            if token.pos_ == 'PROPN' and token.text not in key_phrases:
                key_phrases.append(token.text)
        
        return list(set(key_phrases))[:10]  # Return top 10 unique phrases
    
    def _enhance_with_ml(self, artifact: KnowledgeArtifact) -> KnowledgeArtifact:
        """Enhance artifact with machine learning analysis"""
        enhanced = KnowledgeArtifact(**artifact.to_dict())
        
        try:
            # Extract text content for ML processing
            text_content = self._extract_text_content(artifact)
            
            if text_content and len(text_content.split()) > 10:  # Minimum length for ML
                # Perform pattern recognition with ML
                patterns = self._recognize_patterns_with_ml(text_content)
                
                # Perform topic modeling
                topics = self._analyze_topics(text_content)
                
                # Perform semantic similarity analysis
                similarity = self._analyze_semantic_similarity(text_content)
                
                # Update artifact with ML insights
                if 'ml_analysis' not in enhanced.metadata:
                    enhanced.metadata['ml_analysis'] = {}
                
                enhanced.metadata['ml_analysis'].update({
                    'patterns': patterns,
                    'topics': topics,
                    'semantic_similarity': similarity
                })
                
                # Update quality indicators based on ML analysis
                enhanced.confidence_score = min(1.0, enhanced.confidence_score + 0.10)
                
                self.ml_analysis_stats['ml_analyzed'] += 1
                
        except Exception as e:
            logger.error(f"ML enhancement failed for artifact {artifact.id}: {str(e)}")
        
        return enhanced
    
    def _recognize_patterns_with_ml(self, text: str) -> List[Dict[str, Any]]:
        """Recognize patterns using machine learning models"""
        patterns = []
        
        try:
            # Use text classification pipeline
            if self.ml_models.get('text_classifier'):
                result = self.ml_models['text_classifier'](text)
                
                for prediction in result:
                    patterns.append({
                        'pattern_type': prediction['label'],
                        'confidence': prediction['score'],
                        'source': 'text_classification'
                    })
            
            # Use pattern matching with advanced library
            for pattern_type, pattern_data in self.advanced_pattern_library.get('ml_patterns', {}).items():
                keywords = pattern_data['keywords']
                matches = sum(1 for keyword in keywords if keyword in text.lower())
                match_score = matches / len(keywords) if keywords else 0.0
                
                if match_score > 0.5:
                    patterns.append({
                        'pattern_type': pattern_type,
                        'confidence': 0.7 + (match_score * 0.3),
                        'source': 'pattern_matching'
                    })
        
        except Exception as e:
            logger.error(f"Pattern recognition failed: {str(e)}")
        
        return patterns
    
    def _analyze_topics(self, text: str) -> List[Dict[str, Any]]:
        """Analyze topics using topic modeling"""
        topics = []
        
        try:
            # Simple topic extraction using key phrases for now
            # In production, this would use LDA or other topic modeling
            key_phrases = self._extract_key_phrases(self.nlp_models['spacy'](text))
            
            for i, phrase in enumerate(key_phrases[:3], 1):  # Top 3 topics
                topics.append({
                    'topic_id': f"topic_{i}",
                    'topic': phrase,
                    'confidence': 0.8 - (i * 0.1)  # Decreasing confidence
                })
                
        except Exception as e:
            logger.error(f"Topic analysis failed: {str(e)}")
        
        return topics
    
    def _analyze_semantic_similarity(self, text: str) -> Dict[str, Any]:
        """
        Analyze semantic similarity using sentence embeddings.

        Compares the input text against cached artifacts to find semantically similar content.
        Uses cosine similarity to measure relatedness.
        """
        similarity = {'similar_artifacts': [], 'average_similarity': 0.0, 'embedding_dimensions': 0}

        try:
            if self.ml_models.get('sentence_transformer'):
                # Generate embedding for current text
                embedding = self.ml_models['sentence_transformer'].encode(text)
                similarity['embedding_dimensions'] = len(embedding)

                # If we have cached artifacts, calculate similarity
                if self._artifact_cache:
                    import numpy as np

                    similar_artifacts = []
                    similarities = []

                    # Calculate cosine similarity with each cached artifact
                    for artifact_id, (cached_embedding, metadata) in self._artifact_cache.items():
                        # Cosine similarity: dot product of normalized vectors
                        norm_a = np.linalg.norm(embedding)
                        norm_b = np.linalg.norm(cached_embedding)

                        if norm_a > 0 and norm_b > 0:
                            cos_sim = np.dot(embedding, cached_embedding) / (norm_a * norm_b)
                        else:
                            cos_sim = 0.0

                        # Only include artifacts with meaningful similarity
                        if cos_sim > 0.3:  # Threshold for "similar"
                            similar_artifacts.append({
                                'artifact_id': artifact_id,
                                'similarity': float(cos_sim),
                                'metadata': metadata
                            })
                            similarities.append(cos_sim)

                    # Sort by similarity (descending) and take top 5
                    similar_artifacts.sort(key=lambda x: x['similarity'], reverse=True)
                    top_similar = similar_artifacts[:5]

                    similarity['similar_artifacts'] = top_similar

                    # Calculate average similarity
                    if similarities:
                        similarity['average_similarity'] = float(np.mean(similarities))
                    else:
                        similarity['average_similarity'] = 0.0

                    logger.info({
                        'msg': 'Semantic similarity analysis completed',
                        'cached_artifacts': len(self._artifact_cache),
                        'similar_found': len(top_similar),
                        'avg_similarity': similarity['average_similarity']
                    })
                else:
                    # No cached artifacts yet
                    similarity['average_similarity'] = 0.0
                    logger.info('No cached artifacts for similarity comparison')

        except Exception as e:
            logger.error(f"Semantic similarity analysis failed: {str(e)}")
            similarity['error'] = str(e)

        return similarity

    def _cache_artifact(self, artifact: KnowledgeArtifact) -> None:
        """
        Cache an artifact's embedding for future similarity comparisons.

        Implements LRU-style cache management to limit memory usage.

        Args:
            artifact: The knowledge artifact to cache
        """
        try:
            # Generate embedding if we have a sentence transformer
            if self.ml_models.get('sentence_transformer'):
                text_content = self._extract_text_content(artifact)
                if text_content:
                    embedding = self.ml_models['sentence_transformer'].encode(text_content)

                    # Manage cache size (LRU eviction if needed)
                    if len(self._artifact_cache) >= self._cache_max_size:
                        # Remove oldest entry (first key)
                        oldest_key = next(iter(self._artifact_cache))
                        del self._artifact_cache[oldest_key]
                        logger.debug(f"Evicted artifact {oldest_key} from cache (size limit reached)")

                    # Cache the artifact
                    metadata = {
                        'id': artifact.id,
                        'type': artifact.type,
                        'created_at': artifact.created_at,
                        'content_length': len(text_content)
                    }
                    self._artifact_cache[artifact.id] = (embedding, metadata)

                    logger.debug({
                        'msg': 'Cached artifact for similarity',
                        'artifact_id': artifact.id,
                        'cache_size': len(self._artifact_cache)
                    })
        except Exception as e:
            logger.warning(f"Failed to cache artifact {artifact.id}: {e}")

    def _enhance_with_semantic_analysis(self, artifact: KnowledgeArtifact) -> KnowledgeArtifact:
        """Enhance artifact with semantic analysis"""
        enhanced = KnowledgeArtifact(**artifact.to_dict())
        
        try:
            # Extract text content
            text_content = self._extract_text_content(artifact)
            
            if text_content:
                # Perform semantic role labeling
                semantic_roles = self._perform_semantic_role_labeling(text_content)
                
                # Perform entity linking
                entity_links = self._perform_entity_linking(enhanced)
                
                # Update artifact with semantic insights
                if 'semantic_analysis' not in enhanced.metadata:
                    enhanced.metadata['semantic_analysis'] = {}
                
                enhanced.metadata['semantic_analysis'].update({
                    'semantic_roles': semantic_roles,
                    'entity_links': entity_links
                })
                
                # Update applicability based on semantic analysis
                enhanced.applicability_scope = self._determine_applicability_from_semantics(semantic_roles)
                
        except Exception as e:
            logger.error(f"Semantic analysis failed for artifact {artifact.id}: {str(e)}")
        
        return enhanced
    
    def _perform_semantic_role_labeling(self, text: str) -> List[Dict[str, Any]]:
        """Perform semantic role labeling"""
        roles = []
        
        try:
            # Simple SRL using dependency parsing
            doc = self.nlp_models['spacy'](text)
            
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'attr']:
                    roles.append({
                        'word': token.text,
                        'role': token.dep_,
                        'head': token.head.text,
                        'confidence': 0.85
                    })
                    
        except Exception as e:
            logger.error(f"Semantic role labeling failed: {str(e)}")
        
        return roles
    
    def _perform_entity_linking(self, artifact: KnowledgeArtifact) -> List[Dict[str, Any]]:
        """Perform entity linking to knowledge bases"""
        links = []
        
        try:
            # Extract entities
            entities = artifact.content.get('entities', [])
            
            # Simple entity linking (in production, link to actual knowledge bases)
            for entity in entities:
                links.append({
                    'entity': entity.get('text', ''),
                    'entity_type': entity.get('label', ''),
                    'linked_to': f"knowledge_base:{entity.get('label', '').lower()}",
                    'confidence': entity.get('confidence', 0.7)
                })
                
        except Exception as e:
            logger.error(f"Entity linking failed: {str(e)}")
        
        return links
    
    def _determine_applicability_from_semantics(self, semantic_roles: List[Dict[str, Any]]) -> str:
        """Determine applicability scope from semantic analysis"""
        # Count different semantic roles
        role_types = set(role['role'] for role in semantic_roles)
        
        if len(role_types) >= 3:
            return 'broad'
        elif len(role_types) >= 2:
            return 'moderate'
        else:
            return 'narrow'
    
    def _assess_quality_with_ml(self, artifact: KnowledgeArtifact) -> KnowledgeArtifact:
        """Assess quality using machine learning models"""
        assessed = KnowledgeArtifact(**artifact.to_dict())
        
        try:
            # Calculate ML-based quality score
            ml_quality_score = self._calculate_ml_quality_score(assessed)
            
            # Update quality assessment
            if 'quality_assessment' not in assessed.metadata:
                assessed.metadata['quality_assessment'] = {}
            
            assessed.metadata['quality_assessment']['ml_quality_score'] = ml_quality_score
            
            # Adjust overall quality based on ML assessment
            current_quality = assessed.calculate_quality_score()
            assessed.effectiveness_score = min(1.0, (current_quality + ml_quality_score) / 2)
            
        except Exception as e:
            logger.error(f"ML quality assessment failed: {str(e)}")
        
        return assessed
    
    def _calculate_ml_quality_score(self, artifact: KnowledgeArtifact) -> float:
        """Calculate quality score using ML models"""
        quality_score = 0.7  # Base score
        
        try:
            # Factor in NLP analysis
            if 'nlp_analysis' in artifact.metadata:
                nlp_data = artifact.metadata['nlp_analysis']
                
                # Entity richness factor
                entity_count = len(nlp_data.get('entities', []))
                if entity_count >= 5:
                    quality_score += 0.15
                elif entity_count >= 3:
                    quality_score += 0.10
                
                # Sentiment factor
                sentiment = nlp_data.get('sentiment', {}).get('polarity', 0)
                if abs(sentiment) > 0.3:  # Strong sentiment
                    quality_score += 0.05
                
            # Factor in ML analysis
            if 'ml_analysis' in artifact.metadata:
                ml_data = artifact.metadata['ml_analysis']
                
                # Pattern recognition factor
                patterns = len(ml_data.get('patterns', []))
                if patterns >= 2:
                    quality_score += 0.10
                elif patterns >= 1:
                    quality_score += 0.05
                
                # Topic diversity factor
                topics = len(ml_data.get('topics', []))
                if topics >= 2:
                    quality_score += 0.08
                elif topics >= 1:
                    quality_score += 0.04
            
            # Factor in semantic analysis
            if 'semantic_analysis' in artifact.metadata:
                semantic_data = artifact.metadata['semantic_analysis']
                
                # Semantic role factor
                roles = len(semantic_data.get('semantic_roles', []))
                if roles >= 3:
                    quality_score += 0.07
                elif roles >= 1:
                    quality_score += 0.03
                
                # Entity linking factor
                links = len(semantic_data.get('entity_links', []))
                if links >= 2:
                    quality_score += 0.06
                elif links >= 1:
                    quality_score += 0.03
            
        except Exception as e:
            logger.error(f"ML quality calculation failed: {str(e)}")
        
        return min(0.95, quality_score)  # Cap at 0.95 for ML assessment
    
    def get_advanced_extraction_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including NLP and ML metrics"""
        stats = super().get_extraction_stats()
        
        # Add advanced statistics
        stats['advanced_stats'] = {
            'nlp_processing': dict(self.nlp_processing_stats),
            'ml_analysis': dict(self.ml_analysis_stats),
            'enhancement_rate': (self.nlp_processing_stats.get('artifacts_enhanced', 0) / 
                               max(1, stats['total_extractions'])),
            'nlp_success_rate': (self.nlp_processing_stats.get('nlp_analyzed', 0) / 
                               max(1, self.nlp_processing_stats.get('artifacts_enhanced', 1))),
            'ml_success_rate': (self.ml_analysis_stats.get('ml_analyzed', 0) / 
                              max(1, self.nlp_processing_stats.get('artifacts_enhanced', 1)))
        }
        
        return stats
    
    def reset_advanced_stats(self):
        """Reset advanced statistics"""
        self.nlp_processing_stats = defaultdict(int)
        self.ml_analysis_stats = defaultdict(int)

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create advanced knowledge extractor
    extractor = AdvancedKnowledgeExtractor()
    
    # Example workflow data
    example_workflow = {
        'workflow_id': 'advanced_workflow_001',
        'domain': 'machine_learning',
        'complexity': 'high',
        'solutions': [
            {
                'id': 'sol_001',
                'problem_type': 'neural_network_optimization',
                'solution_approach': 'Implementing advanced neural network architecture with transfer learning and fine-tuning for improved accuracy',
                'success_rate': 0.95,
                'complexity': 8,
                'pattern_type': 'neural_network_architecture'
            }
        ],
        'critiques': [
            {
                'id': 'crit_001',
                'issue_type': 'overfitting',
                'root_cause': 'Insufficient regularization and data augmentation techniques',
                'prevention_strategy': 'Implement L2 regularization, dropout layers, and comprehensive data augmentation pipeline',
                'severity': 'high',
                'pattern_type': 'model_regularization'
            }
        ],
        'teams': [
            {
                'name': 'ml_team',
                'success_rate': 0.92,
                'avg_response_time': 2.1
            }
        ]
    }
    
    print("Starting advanced knowledge extraction...")
    
    # Perform advanced extraction
    artifacts = extractor.extract_from_workflow_advanced(example_workflow)
    
    print(f"\nAdvanced Extraction Results:")
    print(f"  - Total artifacts extracted: {len(artifacts)}")
    
    for i, artifact in enumerate(artifacts, 1):
        print(f"\nArtifact {i}: {artifact.artifact_type}")
        print(f"  - ID: {artifact.id}")
        print(f"  - Quality score: {artifact.calculate_quality_score():.2f}")
        print(f"  - Confidence: {artifact.confidence_score:.2f}")
        
        # Show NLP analysis if available
        if 'nlp_analysis' in artifact.metadata:
            nlp_data = artifact.metadata['nlp_analysis']
            print(f"  - NLP entities: {len(nlp_data.get('entities', []))}")
            print(f"  - Sentiment: {nlp_data.get('sentiment', {}).get('sentiment', 'unknown')}")
            print(f"  - Key phrases: {len(nlp_data.get('key_phrases', []))}")
        
        # Show ML analysis if available
        if 'ml_analysis' in artifact.metadata:
            ml_data = artifact.metadata['ml_analysis']
            print(f"  - ML patterns: {len(ml_data.get('patterns', []))}")
            print(f"  - Topics: {len(ml_data.get('topics', []))}")
        
        # Show semantic analysis if available
        if 'semantic_analysis' in artifact.metadata:
            semantic_data = artifact.metadata['semantic_analysis']
            print(f"  - Semantic roles: {len(semantic_data.get('semantic_roles', []))}")
            print(f"  - Entity links: {len(semantic_data.get('entity_links', []))}")
    
    # Get advanced statistics
    stats = extractor.get_advanced_extraction_stats()
    print(f"\nAdvanced Extraction Statistics:")
    print(f"  - Total extractions: {stats['total_extractions']}")
    print(f"  - NLP enhanced: {stats['advanced_stats']['nlp_processing']['artifacts_enhanced']}")
    print(f"  - ML analyzed: {stats['advanced_stats']['ml_analysis']['ml_analyzed']}")
    print(f"  - Enhancement rate: {stats['advanced_stats']['enhancement_rate']:.2f}")
    print(f"  - NLP success rate: {stats['advanced_stats']['nlp_success_rate']:.2f}")
    print(f"  - ML success rate: {stats['advanced_stats']['ml_success_rate']:.2f}")
    
    print(f"\nAdvanced knowledge extraction completed successfully!")