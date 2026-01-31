"""
ML Intelligence Layer

Machine learning powered features for the knowledge engine:
- Automatic content classification
- Intelligent recommendations
- Content summarization
- Entity extraction
- Sentiment analysis
- Duplicate detection
- Auto-tagging
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import math

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of content classification."""
    category: str
    confidence: float
    alternative_categories: List[Tuple[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "alternative_categories": self.alternative_categories
        }


@dataclass
class Entity:
    """Extracted entity from text."""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class Recommendation:
    """Recommendation for a user or item."""
    item_id: str
    score: float
    reason: str
    recommendation_type: str  # "similar", "complementary", "trending", "personalized"
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": self.score,
            "reason": self.reason,
            "recommendation_type": self.recommendation_type,
            "confidence": self.confidence
        }


class ContentClassifier:
    """
    Automatic content classification using ML techniques.
    
    Uses a combination of:
    - Keyword-based classification
    - TF-IDF similarity
    - Rule-based classification
    """
    
    def __init__(self):
        self.categories = {
            "programming": ["code", "function", "class", "variable", "python", "javascript", "api"],
            "data_science": ["machine learning", "dataset", "model", "training", "prediction", "ai"],
            "devops": ["deployment", "docker", "kubernetes", "ci/cd", "pipeline", "infrastructure"],
            "design": ["ui", "ux", "interface", "user experience", "prototype", "wireframe"],
            "business": ["strategy", "revenue", "market", "customer", "product", "growth"],
            "research": ["study", "experiment", "analysis", "hypothesis", "finding", "paper"]
        }
        
        # TF-IDF components
        self._document_frequency: Dict[str, int] = defaultdict(int)
        self._total_documents = 0
        
    def classify(self, content: str, title: Optional[str] = None) -> ClassificationResult:
        """
        Classify content into categories.
        
        Args:
            content: The content to classify
            title: Optional title (weighted more heavily)
            
        Returns:
            ClassificationResult with category and confidence
        """
        if not content:
            return ClassificationResult("uncategorized", 0.0)
        
        text = (title or "") + " " + content
        text_lower = text.lower()
        
        # Calculate scores for each category
        scores = {}
        for category, keywords in self.categories.items():
            score = 0.0
            for keyword in keywords:
                count = text_lower.count(keyword)
                if count > 0:
                    # Weight by keyword importance (could be learned)
                    score += count * (1.0 + len(keyword) / 100.0)
            scores[category] = score
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score == 0:
            return ClassificationResult("uncategorized", 0.0)
        
        # Get best category
        sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_category = sorted_categories[0]
        
        confidence = best_category[1] / total_score if total_score > 0 else 0.0
        
        # Alternative categories (with lower scores)
        alternatives = [
            (cat, score / total_score) 
            for cat, score in sorted_categories[1:3] 
            if score / total_score > 0.1
        ]
        
        return ClassificationResult(
            category=best_category[0],
            confidence=min(confidence, 1.0),
            alternative_categories=alternatives
        )
    
    def batch_classify(
        self, 
        items: List[Tuple[str, str]]  # (content, title) tuples
    ) -> List[ClassificationResult]:
        """Classify multiple items."""
        return [self.classify(content, title) for content, title in items]
    
    def add_category(self, category: str, keywords: List[str]):
        """Add a new category with keywords."""
        self.categories[category] = keywords
    
    def train_tfidf(self, documents: List[str]):
        """
        Train TF-IDF on a corpus of documents.
        This improves classification accuracy.
        """
        self._document_frequency.clear()
        self._total_documents = len(documents)
        
        for doc in documents:
            words = set(self._tokenize(doc.lower()))
            for word in words:
                self._document_frequency[word] += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())


class EntityExtractor:
    """
    Named Entity Recognition (NER) for knowledge content.
    
    Extracts entities like:
    - Person names
    - Organizations
    - Technologies
    - Dates
    - Technical terms
    """
    
    def __init__(self):
        # Entity patterns
        self.patterns = {
            "technology": [
                r'\b(?:Python|JavaScript|TypeScript|Java|C\+\+|Go|Rust|Ruby|PHP)\b',
                r'\b(?:React|Vue|Angular|Django|Flask|FastAPI|Express)\b',
                r'\b(?:Docker|Kubernetes|AWS|Azure|GCP|Terraform|Ansible)\b',
                r'\b(?:PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch)\b'
            ],
            "organization": [
                r'\b(?:Google|Microsoft|Amazon|Apple|Meta|Netflix|Spotify)\b',
                r'\b(?:GitHub|GitLab|Bitbucket|Atlassian|Jira|Confluence)\b'
            ],
            "person": [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'  # Simple name pattern
            ],
            "date": [
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b'
            ],
            "version": [
                r'\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b'
            ],
            "email": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            "url": [
                r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
            ]
        }
        
        # Compile patterns
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {
            entity_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for entity_type, patterns in self.patterns.items()
        }
    
    def extract(self, content: str) -> List[Entity]:
        """
        Extract entities from content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for entity_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(content):
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_confidence(match.group(), entity_type)
                    
                    entity = Entity(
                        text=match.group(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=confidence
                    )
                    entities.append(entity)
        
        # Sort by position
        entities.sort(key=lambda e: e.start_pos)
        
        # Remove overlapping entities (keep higher confidence)
        entities = self._remove_overlapping(entities)
        
        return entities
    
    def extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical terms from content."""
        # Patterns for technical terms
        term_patterns = [
            r'\b[a-z]+_[a-z_]+\b',  # snake_case
            r'\b[a-z]+(?:[A-Z][a-z]+)+\b',  # camelCase
            r'\b[A-Z][a-z]*(?:[A-Z][a-z]+)+\b',  # PascalCase
            r'\b[A-Z]{2,}(?:_[A-Z]+)*\b',  # CONSTANTS
        ]
        
        terms = set()
        for pattern in term_patterns:
            for match in re.finditer(pattern, content):
                terms.add(match.group())
        
        return sorted(terms)
    
    def _calculate_confidence(self, text: str, entity_type: str) -> float:
        """Calculate confidence score for an entity."""
        base_confidence = 0.7
        
        # Boost confidence for longer matches
        if len(text) > 5:
            base_confidence += 0.1
        
        # Boost for capitalized words (likely proper nouns)
        if text[0].isupper():
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _remove_overlapping(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping higher confidence ones."""
        if not entities:
            return entities
        
        # Sort by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
        
        result = []
        covered_ranges: List[Tuple[int, int]] = []
        
        for entity in sorted_entities:
            # Check if this entity overlaps with any already selected
            overlaps = False
            for start, end in covered_ranges:
                if not (entity.end_pos <= start or entity.start_pos >= end):
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(entity)
                covered_ranges.append((entity.start_pos, entity.end_pos))
        
        # Sort back by position
        result.sort(key=lambda e: e.start_pos)
        
        return result


class ContentSummarizer:
    """
    Automatic content summarization.
    
    Implements extractive summarization using:
    - Sentence scoring based on word frequency
    - Position weighting
    - Length normalization
    """
    
    def __init__(self):
        self.stop_words = set([
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "and", "but", "or", "yet", "so",
            "if", "because", "although", "though", "while", "where",
            "when", "that", "which", "who", "whom", "whose", "what"
        ])
    
    def summarize(self, content: str, num_sentences: int = 3) -> str:
        """
        Generate a summary of the content.
        
        Args:
            content: Content to summarize
            num_sentences: Number of sentences in summary
            
        Returns:
            Summary text
        """
        if not content:
            return ""
        
        # Split into sentences
        sentences = self._split_sentences(content)
        
        if len(sentences) <= num_sentences:
            return content
        
        # Score sentences
        word_freq = self._calculate_word_frequency(content)
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, word_freq, i, len(sentences))
            sentence_scores.append((score, i, sentence))
        
        # Select top sentences
        top_sentences = sorted(sentence_scores, reverse=True)[:num_sentences]
        top_sentences.sort(key=lambda x: x[1])  # Sort by original position
        
        # Join summary sentences
        summary = " ".join(sentence for _, _, sentence in top_sentences)
        
        return summary
    
    def extract_key_points(self, content: str, num_points: int = 5) -> List[str]:
        """Extract key points from content."""
        sentences = self._split_sentences(content)
        
        if len(sentences) <= num_points:
            return sentences
        
        word_freq = self._calculate_word_frequency(content)
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, word_freq, i, len(sentences))
            sentence_scores.append((score, sentence))
        
        # Get top scoring sentences
        top_sentences = sorted(sentence_scores, reverse=True)[:num_points]
        
        return [sentence for _, sentence in top_sentences]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_word_frequency(self, text: str) -> Dict[str, float]:
        """Calculate word frequency in text."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        word_counts = Counter(words)
        max_count = max(word_counts.values()) if word_counts else 1
        
        # Normalize frequencies
        return {word: count / max_count for word, count in word_counts.items()}
    
    def _score_sentence(
        self, 
        sentence: str, 
        word_freq: Dict[str, float],
        position: int,
        total_sentences: int
    ) -> float:
        """Score a sentence for importance."""
        words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        words = [w for w in words if w not in self.stop_words]
        
        if not words:
            return 0.0
        
        # Word frequency score
        freq_score = sum(word_freq.get(word, 0) for word in words) / len(words)
        
        # Position weight (first and last sentences often more important)
        if position == 0:
            position_weight = 1.5
        elif position == total_sentences - 1:
            position_weight = 1.3
        else:
            position_weight = 1.0
        
        # Length normalization (prefer medium-length sentences)
        length = len(words)
        if 10 <= length <= 25:
            length_weight = 1.0
        else:
            length_weight = 0.8
        
        return freq_score * position_weight * length_weight


class RecommendationEngine:
    """
    Intelligent recommendation system.
    
    Provides recommendations based on:
    - Content similarity
    - User behavior
    - Collaborative filtering
    - Trending items
    """
    
    def __init__(self):
        self._item_embeddings: Dict[str, List[float]] = {}
        self._user_interactions: Dict[str, List[str]] = defaultdict(list)  # user -> item_ids
        self._item_interactions: Dict[str, List[str]] = defaultdict(list)  # item -> user_ids
        self._interaction_weights = {
            "view": 1.0,
            "like": 2.0,
            "save": 3.0,
            "share": 4.0
        }
    
    def add_item_embedding(self, item_id: str, embedding: List[float]):
        """Add embedding for an item."""
        self._item_embeddings[item_id] = embedding
    
    def record_interaction(self, user_id: str, item_id: str, interaction_type: str = "view"):
        """Record a user-item interaction."""
        if item_id not in self._user_interactions[user_id]:
            self._user_interactions[user_id].append(item_id)
            self._item_interactions[item_id].append(user_id)
    
    def recommend_similar(
        self, 
        item_id: str, 
        num_recommendations: int = 5
    ) -> List[Recommendation]:
        """
        Recommend items similar to a given item.
        
        Args:
            item_id: ID of reference item
            num_recommendations: Number of recommendations
            
        Returns:
            List of recommendations
        """
        reference_embedding = self._item_embeddings.get(item_id)
        if not reference_embedding:
            return []
        
        similarities = []
        for other_id, other_embedding in self._item_embeddings.items():
            if other_id != item_id:
                similarity = self._cosine_similarity(reference_embedding, other_embedding)
                similarities.append((other_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for rec_id, score in similarities[:num_recommendations]:
            rec = Recommendation(
                item_id=rec_id,
                score=score,
                reason="Content similarity",
                recommendation_type="similar",
                confidence=score
            )
            recommendations.append(rec)
        
        return recommendations
    
    def recommend_for_user(
        self, 
        user_id: str, 
        num_recommendations: int = 5
    ) -> List[Recommendation]:
        """
        Recommend items for a specific user.
        
        Uses collaborative filtering and content-based filtering.
        """
        user_items = self._user_interactions.get(user_id, [])
        
        if not user_items:
            # New user, return trending items
            return self._get_trending_recommendations(num_recommendations)
        
        recommendations: Dict[str, float] = defaultdict(float)
        
        # Collaborative filtering: find similar users
        similar_users = self._find_similar_users(user_id)
        for similar_user, similarity in similar_users:
            for item_id in self._user_interactions[similar_user]:
                if item_id not in user_items:
                    recommendations[item_id] += similarity * 2.0
        
        # Content-based: items similar to what user liked
        for item_id in user_items:
            similar_items = self.recommend_similar(item_id, num_recommendations=3)
            for rec in similar_items:
                if rec.item_id not in user_items:
                    recommendations[rec.item_id] += rec.score
        
        # Convert to recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for rec_id, score in sorted_recs[:num_recommendations]:
            reason = "Based on your interests" if score > 1.5 else "You might find this interesting"
            result.append(Recommendation(
                item_id=rec_id,
                score=min(score, 1.0),
                reason=reason,
                recommendation_type="personalized",
                confidence=min(score / 3.0, 1.0)
            ))
        
        return result
    
    def _find_similar_users(self, user_id: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Find users with similar interaction patterns."""
        user_items = set(self._user_interactions.get(user_id, []))
        
        if not user_items:
            return []
        
        similarities = []
        for other_user, other_items in self._user_interactions.items():
            if other_user != user_id:
                other_items_set = set(other_items)
                intersection = len(user_items & other_items_set)
                union = len(user_items | other_items_set)
                
                if union > 0:
                    jaccard = intersection / union
                    if jaccard > 0.1:  # Minimum similarity threshold
                        similarities.append((other_user, jaccard))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def _get_trending_recommendations(self, num: int = 5) -> List[Recommendation]:
        """Get trending items based on recent interactions."""
        item_popularity = {
            item_id: len(users)
            for item_id, users in self._item_interactions.items()
        }
        
        sorted_items = sorted(
            item_popularity.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        max_popularity = sorted_items[0][1] if sorted_items else 1
        
        recommendations = []
        for item_id, popularity in sorted_items[:num]:
            score = popularity / max_popularity
            recommendations.append(Recommendation(
                item_id=item_id,
                score=score,
                reason="Trending now",
                recommendation_type="trending",
                confidence=score
            ))
        
        return recommendations
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class DuplicateDetector:
    """
    Detect duplicate or near-duplicate content.
    
    Uses:
    - Hash-based exact matching
    - MinHash for near-duplicate detection
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self._content_hashes: Set[str] = set()
        self._minhashes: Dict[str, Set[int]] = {}
        self._num_hashes = 20
        
    def add_content(self, content_id: str, content: str):
        """Add content for duplicate detection."""
        # Exact hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        self._content_hashes.add(content_hash)
        
        # MinHash signature
        shingles = self._get_shingles(content)
        minhash = self._compute_minhash(shingles)
        self._minhashes[content_id] = minhash
    
    def check_duplicate(self, content: str) -> Tuple[bool, Optional[str], float]:
        """
        Check if content is a duplicate.
        
        Returns:
            (is_duplicate, duplicate_id, similarity_score)
        """
        # Check exact match first
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if content_hash in self._content_hashes:
            return True, None, 1.0
        
        # Check near-duplicates using MinHash
        shingles = self._get_shingles(content)
        minhash = self._compute_minhash(shingles)
        
        best_match = None
        best_similarity = 0.0
        
        for content_id, other_minhash in self._minhashes.items():
            similarity = self._estimate_similarity(minhash, other_minhash)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = content_id
        
        if best_similarity >= self.similarity_threshold:
            return True, best_match, best_similarity
        
        return False, None, best_similarity
    
    def _get_shingles(self, content: str, k: int = 5) -> Set[str]:
        """Get k-shingles from content."""
        content = content.lower()
        content = re.sub(r'[^a-z0-9\s]', '', content)
        words = content.split()
        
        shingles = set()
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i+k])
            shingles.add(shingle)
        
        return shingles
    
    def _compute_minhash(self, shingles: Set[str]) -> Set[int]:
        """Compute MinHash signature."""
        signature = set()
        
        for i in range(self._num_hashes):
            min_hash = float('inf')
            for shingle in shingles:
                # Use hash function
                hash_val = int(hashlib.md5(f"{shingle}:{i}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, hash_val)
            signature.add(min_hash)
        
        return signature
    
    def _estimate_similarity(self, sig1: Set[int], sig2: Set[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        intersection = len(sig1 & sig2)
        return intersection / self._num_hashes


class AutoTagger:
    """
    Automatic tag generation for knowledge content.
    
    Extracts relevant tags from content using:
    - Keyword extraction
    - TF-IDF scoring
    - Domain-specific dictionaries
    """
    
    def __init__(self):
        self.domain_keywords = {
            "programming": ["python", "javascript", "coding", "development", "api"],
            "design": ["ui", "ux", "interface", "design", "prototype"],
            "data": ["analysis", "dataset", "visualization", "statistics"],
            "devops": ["deployment", "docker", "kubernetes", "ci/cd"]
        }
        
    def generate_tags(
        self, 
        content: str, 
        title: Optional[str] = None,
        max_tags: int = 10
    ) -> List[str]:
        """
        Generate tags for content.
        
        Args:
            content: Content to analyze
            title: Optional title (weighted more)
            max_tags: Maximum number of tags
            
        Returns:
            List of generated tags
        """
        text = (title or "") + " " + content
        text_lower = text.lower()
        
        # Extract potential tags
        candidates: Dict[str, float] = defaultdict(float)
        
        # 1. Domain keywords
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                count = text_lower.count(keyword)
                if count > 0:
                    candidates[keyword] += count * 1.5
        
        # 2. Capitalized terms (likely proper nouns/technical terms)
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
        for term in capitalized:
            term_lower = term.lower()
            if len(term_lower) > 2:
                candidates[term_lower] += 2.0
        
        # 3. Multi-word technical terms
        technical_patterns = [
            r'\b[a-z]+_[a-z_]+\b',  # snake_case
            r'\b[a-z]+[A-Z][a-z]+\b',  # camelCase
        ]
        for pattern in technical_patterns:
            for match in re.finditer(pattern, text):
                term = match.group().lower()
                candidates[term] += 1.5
        
        # 4. Frequent words (not stop words)
        words = re.findall(r'\b[a-z]{4,}\b', text_lower)
        stop_words = {"this", "that", "with", "from", "they", "have", "were"}
        word_counts = Counter(w for w in words if w not in stop_words)
        
        for word, count in word_counts.most_common(10):
            candidates[word] += count * 0.5
        
        # Sort by score and return top tags
        sorted_tags = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, score in sorted_tags[:max_tags]]


class MLIntelligenceEngine:
    """
    Main ML intelligence engine that combines all ML features.
    """
    
    def __init__(self):
        self.classifier = ContentClassifier()
        self.entity_extractor = EntityExtractor()
        self.summarizer = ContentSummarizer()
        self.recommendation_engine = RecommendationEngine()
        self.duplicate_detector = DuplicateDetector()
        self.auto_tagger = AutoTagger()
        
    async def analyze_content(
        self, 
        content: str, 
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive content analysis.
        
        Returns:
            Dictionary with all analysis results
        """
        # Run analysis in parallel
        classification, entities, summary, tags, duplicate_check = await asyncio.gather(
            asyncio.to_thread(self.classifier.classify, content, title),
            asyncio.to_thread(self.entity_extractor.extract, content),
            asyncio.to_thread(self.summarizer.summarize, content),
            asyncio.to_thread(self.auto_tagger.generate_tags, content, title),
            asyncio.to_thread(self.duplicate_detector.check_duplicate, content)
        )
        
        is_duplicate, duplicate_id, similarity = duplicate_check
        
        return {
            "classification": classification.to_dict(),
            "entities": [e.to_dict() for e in entities],
            "summary": summary,
            "tags": tags,
            "technical_terms": self.entity_extractor.extract_technical_terms(content),
            "key_points": self.summarizer.extract_key_points(content),
            "duplicate_info": {
                "is_duplicate": is_duplicate,
                "duplicate_id": duplicate_id,
                "similarity": similarity
            }
        }
    
    def get_recommendations(
        self, 
        user_id: Optional[str] = None,
        item_id: Optional[str] = None,
        num_recommendations: int = 5
    ) -> List[Recommendation]:
        """
        Get recommendations for a user or similar to an item.
        """
        if user_id:
            return self.recommendation_engine.recommend_for_user(user_id, num_recommendations)
        elif item_id:
            return self.recommendation_engine.recommend_similar(item_id, num_recommendations)
        else:
            # Return trending
            return self.recommendation_engine._get_trending_recommendations(num_recommendations)
    
    def record_interaction(
        self, 
        user_id: str, 
        item_id: str, 
        interaction_type: str = "view"
    ):
        """Record a user interaction for recommendations."""
        self.recommendation_engine.record_interaction(user_id, item_id, interaction_type)
    
    def add_item_for_dedup(self, item_id: str, content: str):
        """Add item to duplicate detection index."""
        self.duplicate_detector.add_content(item_id, content)
    
    def add_item_embedding(self, item_id: str, embedding: List[float]):
        """Add item embedding for recommendations."""
        self.recommendation_engine.add_item_embedding(item_id, embedding)


__all__ = [
    "MLIntelligenceEngine",
    "ContentClassifier",
    "EntityExtractor",
    "ContentSummarizer",
    "RecommendationEngine",
    "DuplicateDetector",
    "AutoTagger",
    "ClassificationResult",
    "Entity",
    "Recommendation"
]
