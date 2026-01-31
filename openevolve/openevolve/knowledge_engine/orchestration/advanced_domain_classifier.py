"""
Advanced Domain Classifier for Knowledge Engine

A production-grade, multi-modal domain classification system that uses:
1. Embedding-based semantic similarity
2. Statistical feature extraction (TF-IDF, n-grams)
3. Transformer-based classification (when available)
4. Multi-label classification for mixed content
5. Confidence calibration and uncertainty quantification
6. Active learning for continuous improvement
"""

import json
import logging
import re
import hashlib
import pickle
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import Counter, defaultdict
import statistics
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class DomainCategory(Enum):
    """Extended domain categories"""
    GENERAL = "general"
    FINANCE = "finance"
    CHEMISTRY = "chemistry"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    ENGINEERING = "engineering"
    RESEARCH = "research"
    SOCIAL_MEDIA = "social_media"
    TECHNOLOGY = "technology"
    EDUCATION = "education"
    GOVERNMENT = "government"
    NEWS = "news"
    BIOLOGY = "biology"
    PHYSICS = "physics"
    MATHEMATICS = "mathematics"
    LITERATURE = "literature"
    HISTORY = "history"
    ARTS = "arts"
    BUSINESS = "business"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    ENVIRONMENT = "environment"
    PSYCHOLOGY = "psychology"
    SOCIOLOGY = "sociology"
    PHILOSOPHY = "philosophy"
    ECONOMICS = "economics"
    ASTRONOMY = "astronomy"
    GEOGRAPHY = "geography"
    LINGUISTICS = "linguistics"
    ANTHROPOLOGY = "anthropology"


class ContentType(Enum):
    """Content type classification"""
    TEXT = "text"
    TECHNICAL_DOCUMENT = "technical_document"
    RESEARCH_PAPER = "research_paper"
    NEWS_ARTICLE = "news_article"
    SOCIAL_POST = "social_post"
    LEGAL_DOCUMENT = "legal_document"
    MEDICAL_RECORD = "medical_record"
    FINANCIAL_REPORT = "financial_report"
    EMAIL = "email"
    CHAT_LOG = "chat_log"
    CODE = "code"
    DATA_TABLE = "data_table"
    MIXED = "mixed"
    ACADEMIC_PAPER = "academic_paper"
    PATENT = "patent"
    WHITEPAPER = "whitepaper"
    BLOG_POST = "blog_post"
    FORUM_DISCUSSION = "forum_discussion"
    PRODUCT_REVIEW = "product_review"
    SURVEY_RESPONSE = "survey_response"


@dataclass
class ClassificationFeatures:
    """Extracted features from input"""
    # Statistical features
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    
    # Lexical diversity
    unique_words: int = 0
    lexical_diversity: float = 0.0  # unique_words / total_words
    
    # Domain-specific metrics
    technical_term_density: float = 0.0
    numeric_density: float = 0.0
    punctuation_density: float = 0.0
    capitalization_density: float = 0.0
    
    # N-gram features
    top_unigrams: List[str] = field(default_factory=list)
    top_bigrams: List[str] = field(default_factory=list)
    top_trigrams: List[str] = field(default_factory=list)
    
    # Structural features
    has_code_blocks: bool = False
    has_tables: bool = False
    has_headers: bool = False
    has_citations: bool = False
    has_urls: bool = False
    has_emails: bool = False
    
    # Embedding (if available)
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None


@dataclass
class DomainPrediction:
    """Single domain prediction with confidence"""
    domain: DomainCategory
    confidence: float
    raw_score: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Comprehensive classification result"""
    primary_domain: DomainCategory
    confidence: float
    calibrated_confidence: float
    secondary_domains: List[DomainPrediction] = field(default_factory=list)
    content_type: ContentType = ContentType.TEXT
    features: Optional[ClassificationFeatures] = None
    is_multi_domain: bool = False
    domain_distribution: Dict[str, float] = field(default_factory=dict)
    recommended_components: List[str] = field(default_factory=list)
    classification_methods: List[str] = field(default_factory=list)
    uncertainty_estimate: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_domain': self.primary_domain.value,
            'confidence': self.confidence,
            'calibrated_confidence': self.calibrated_confidence,
            'secondary_domains': [
                {'domain': d.domain.value, 'confidence': d.confidence, 'evidence': d.evidence}
                for d in self.secondary_domains
            ],
            'content_type': self.content_type.value,
            'is_multi_domain': self.is_multi_domain,
            'domain_distribution': self.domain_distribution,
            'recommended_components': self.recommended_components,
            'classification_methods': self.classification_methods,
            'uncertainty_estimate': self.uncertainty_estimate,
            'timestamp': self.timestamp
        }


class FeatureExtractor:
    """Extract statistical and linguistic features from text"""
    
    # Technical term patterns
    TECHNICAL_PATTERNS = {
        'finance': r'\b(?:equity|derivative|portfolio|diversification|liquidity|volatility|arbitrage|hedge|leverage|margin)\b',
        'chemistry': r'\b(?:stoichiometry|isomerism|catalysis|polymerization|hydrolysis|oxidation|reduction|electrophile|nucleophile)\b',
        'healthcare': r'\b(?:pathophysiology|pharmacokinetics|epidemiology|etiology|prognosis|comorbidity|iatrogenic)\b',
        'technology': r'\b(?:microservice|containerization|orchestration|scalability|latency|throughput|redundancy)\b',
        'legal': r'\b(?:jurisprudence|precedent|tort|negligence|liability|indemnification|arbitration|mediation)\b',
        'research': r'\b(?:methodology|hypothesis|significance|correlation|causation|validity|reliability|reproducibility)\b',
    }
    
    # Code detection patterns
    CODE_PATTERNS = [
        r'```[\s\S]*?```',  # Markdown code blocks
        r'<code>[\s\S]*?</code>',  # HTML code blocks
        r'(?:def|class|function|var|let|const|import|from)\s+\w+',
        r'[{};]\s*\n.*[{};]',  # Brace patterns
        r'(?:#|//|/\*|\*/)',  # Comments
    ]
    
    # Table detection
    TABLE_PATTERNS = [
        r'\|.*\|.*\|',  # Markdown tables
        r'<table[\s\S]*?</table>',  # HTML tables
        r'\t+',  # Tab-separated
        r'\s{3,}\w+\s{3,}\w+',  # Space-aligned
    ]
    
    def __init__(self):
        self._cache: Dict[str, ClassificationFeatures] = {}
        self._cache_lock = threading.Lock()
    
    def extract_features(self, text: str, use_cache: bool = True) -> ClassificationFeatures:
        """Extract comprehensive features from text"""
        if not text:
            return ClassificationFeatures()
        
        # Check cache
        text_hash = hashlib.md5(text[:5000].encode()).hexdigest()
        if use_cache:
            with self._cache_lock:
                if text_hash in self._cache:
                    return self._cache[text_hash]
        
        features = ClassificationFeatures()
        
        # Basic statistics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        features.word_count = len(words)
        features.char_count = len(text)
        features.sentence_count = len(sentences)
        features.avg_word_length = statistics.mean(len(w) for w in words) if words else 0
        features.avg_sentence_length = features.word_count / max(features.sentence_count, 1)
        
        # Lexical diversity
        unique_words = set(w.lower().strip('.,!?;:"()[]{}') for w in words)
        features.unique_words = len(unique_words)
        features.lexical_diversity = len(unique_words) / max(len(words), 1)
        
        # Technical term density
        tech_terms = 0
        for pattern in self.TECHNICAL_PATTERNS.values():
            tech_terms += len(re.findall(pattern, text, re.IGNORECASE))
        features.technical_term_density = tech_terms / max(len(words), 1)
        
        # Numeric and punctuation density
        numbers = len(re.findall(r'\d+\.?\d*', text))
        features.numeric_density = numbers / max(len(words), 1)
        
        punct_count = sum(1 for c in text if c in '.,!?;:()[]{}""')
        features.punctuation_density = punct_count / max(len(text), 1)
        
        caps_count = sum(1 for c in text if c.isupper())
        features.capitalization_density = caps_count / max(len(text), 1)
        
        # N-grams (top 10 by frequency)
        if len(words) >= 1:
            unigrams = Counter(w.lower().strip('.,!?;:"()[]{}') for w in words if len(w) > 3)
            features.top_unigrams = [w for w, _ in unigrams.most_common(10)]
        
        if len(words) >= 2:
            bigrams = Counter(f"{words[i].lower()} {words[i+1].lower()}" 
                            for i in range(len(words)-1))
            features.top_bigrams = [b for b, _ in bigrams.most_common(10)]
        
        if len(words) >= 3:
            trigrams = Counter(f"{words[i].lower()} {words[i+1].lower()} {words[i+2].lower()}"
                             for i in range(len(words)-2))
            features.top_trigrams = [t for t, _ in trigrams.most_common(10)]
        
        # Structural features
        features.has_code_blocks = any(re.search(p, text) for p in self.CODE_PATTERNS)
        features.has_tables = any(re.search(p, text) for p in self.TABLE_PATTERNS)
        features.has_headers = bool(re.search(r'\n#{1,6}\s+', text))
        features.has_citations = bool(re.search(r'\(\w+\s+et\s+al\.?\s*,?\s*\d{4}\)', text))
        features.has_urls = bool(re.search(r'https?://\S+', text))
        features.has_emails = bool(re.search(r'\S+@\S+\.\S+', text))
        
        # Cache result
        if use_cache:
            with self._cache_lock:
                self._cache[text_hash] = features
                # Limit cache size
                if len(self._cache) > 1000:
                    self._cache.pop(next(iter(self._cache)))
        
        return features


class EmbeddingClassifier:
    """Classifier using text embeddings and similarity"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.domain_centroids: Dict[DomainCategory, List[float]] = {}
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, embedding classification disabled")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
    
    def embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        if self.model is None:
            return None
        try:
            embedding = self.model.encode(text[:5000])  # Limit text length
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between embeddings"""
        import numpy as np
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def classify(self, text: str, features: ClassificationFeatures) -> Optional[DomainPrediction]:
        """Classify using embeddings"""
        if self.model is None:
            return None
        
        embedding = self.embed(text)
        if embedding is None:
            return None
        
        features.embedding = embedding
        features.embedding_model = self.model_name
        
        # Compare with domain centroids
        if not self.domain_centroids:
            # Initialize with default centroids (simplified)
            self._initialize_centroids()
        
        similarities = {}
        for domain, centroid in self.domain_centroids.items():
            similarity = self.compute_similarity(embedding, centroid)
            similarities[domain] = similarity
        
        if not similarities:
            return None
        
        best_domain = max(similarities.items(), key=lambda x: x[1])
        
        return DomainPrediction(
            domain=best_domain[0],
            confidence=min(best_domain[1], 1.0),
            raw_score=best_domain[1],
            evidence=[f"embedding_similarity_{self.model_name}"]
        )
    
    def _initialize_centroids(self):
        """Initialize domain centroids with representative examples"""
        # Simplified - in production, these would be learned from training data
        domain_examples = {
            DomainCategory.FINANCE: "stock market trading investment portfolio equity bond dividend earnings financial",
            DomainCategory.CHEMISTRY: "molecule compound reaction catalyst synthesis organic chemical element bond",
            DomainCategory.HEALTHCARE: "patient diagnosis treatment symptom disease medical clinical hospital medication",
            DomainCategory.TECHNOLOGY: "software hardware algorithm api database cloud server network protocol",
            DomainCategory.RESEARCH: "study experiment hypothesis data analysis methodology results conclusion",
        }
        
        for domain, example_text in domain_examples.items():
            embedding = self.embed(example_text)
            if embedding:
                self.domain_centroids[domain] = embedding


class StatisticalClassifier:
    """Classifier using statistical features and TF-IDF"""
    
    def __init__(self):
        self.domain_keywords = self._initialize_keywords()
        self.feature_weights = self._initialize_weights()
    
    def _initialize_keywords(self) -> Dict[DomainCategory, List[str]]:
        """Initialize domain keyword dictionaries"""
        return {
            DomainCategory.FINANCE: [
                'stock', 'market', 'trading', 'investment', 'portfolio', 'equity', 'bond',
                'dividend', 'earnings', 'revenue', 'profit', 'financial', 'fiscal',
                'quarter', 'sec', 'balance sheet', 'income statement', 'cash flow',
                'bull', 'bear', 'ipo', 'merger', 'acquisition', 'valuation', 'asset'
            ],
            DomainCategory.CHEMISTRY: [
                'molecule', 'compound', 'reaction', 'catalyst', 'synthesis', 'organic',
                'inorganic', 'polymer', 'acid', 'base', 'ph', 'solution', 'solvent',
                'chemical', 'element', 'periodic', 'bond', 'ion', 'mole', 'stoichiometry'
            ],
            DomainCategory.HEALTHCARE: [
                'patient', 'diagnosis', 'treatment', 'symptom', 'disease', 'condition',
                'medical', 'clinical', 'hospital', 'doctor', 'physician', 'medication',
                'drug', 'dosage', 'therapy', 'surgery', 'prognosis', 'pathology'
            ],
            DomainCategory.TECHNOLOGY: [
                'software', 'hardware', 'algorithm', 'api', 'database', 'cloud',
                'server', 'network', 'protocol', 'framework', 'library', 'module',
                'function', 'class', 'method', 'variable', 'code', 'programming'
            ],
            DomainCategory.RESEARCH: [
                'study', 'research', 'experiment', 'hypothesis', 'data', 'analysis',
                'methodology', 'results', 'discussion', 'conclusion', 'publication',
                'peer review', 'significance', 'correlation', 'causation', 'sample'
            ],
            DomainCategory.LEGAL: [
                'contract', 'agreement', 'clause', 'provision', 'liability',
                'jurisdiction', 'plaintiff', 'defendant', 'court', 'litigation',
                'arbitration', 'statute', 'regulation', 'compliance', 'legal'
            ],
            DomainCategory.BUSINESS: [
                'strategy', 'marketing', 'sales', 'customer', 'product', 'service',
                'revenue', 'growth', 'roi', 'kpi', 'metric', 'conversion', 'lead',
                'stakeholder', 'shareholder', 'b2b', 'b2c', 'startup'
            ],
            DomainCategory.NEWS: [
                'breaking', 'exclusive', 'reported', 'according to', 'sources',
                'officials', 'spokesperson', 'announced', 'confirmed', 'developing'
            ],
        }
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize feature weights for classification"""
        return {
            'keyword_match': 0.4,
            'technical_density': 0.2,
            'lexical_diversity': 0.15,
            'structural_features': 0.15,
            'content_patterns': 0.1
        }
    
    def classify(self, text: str, features: ClassificationFeatures) -> List[DomainPrediction]:
        """Classify using statistical features"""
        text_lower = text.lower()
        predictions = []
        
        for domain, keywords in self.domain_keywords.items():
            score = 0.0
            evidence = []
            
            # Keyword matching
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            keyword_score = keyword_matches / max(len(keywords), 1)
            score += keyword_score * self.feature_weights['keyword_match']
            
            if keyword_matches > 0:
                evidence.append(f"keywords:{keyword_matches}")
            
            # Technical term density
            tech_score = min(features.technical_term_density * 10, 1.0)
            score += tech_score * self.feature_weights['technical_density']
            
            # Lexical diversity (research papers tend to have higher diversity)
            diversity_score = features.lexical_diversity
            if domain in [DomainCategory.RESEARCH, DomainCategory.ACADEMIC]:
                score += diversity_score * self.feature_weights['lexical_diversity'] * 0.5
            
            # Structural features
            struct_score = 0.0
            if features.has_citations and domain == DomainCategory.RESEARCH:
                struct_score += 0.5
            if features.has_code_blocks and domain == DomainCategory.TECHNOLOGY:
                struct_score += 0.5
            if features.has_tables and domain in [DomainCategory.RESEARCH, DomainCategory.FINANCE]:
                struct_score += 0.3
            
            score += struct_score * self.feature_weights['structural_features']
            
            if score > 0.1:  # Minimum threshold
                predictions.append(DomainPrediction(
                    domain=domain,
                    confidence=min(score, 1.0),
                    raw_score=score,
                    evidence=evidence
                ))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        return predictions


class ContentTypeClassifier:
    """Classify content type based on structural and linguistic features"""
    
    PATTERNS = {
        ContentType.RESEARCH_PAPER: [
            r'\babstract\b.*\bintroduction\b',
            r'\bmethodology\b.*\bresults\b',
            r'\bdoi:\s*\d{2}\.\d{4,}',
            r'\b(?:figure|table)\s+\d+[.:]',
        ],
        ContentType.LEGAL_DOCUMENT: [
            r'\bwitness\s+whereof\b',
            r'\bterms\s+and\s+conditions\b',
            r'\bwhereas\b.*\bhereby\b',
            r'\bparty\s+of\s+the\s+(?:first|second)\s+part\b',
        ],
        ContentType.CODE: [
            r'```[\s\S]{50,}```',
            r'(?:def|class|function)\s+\w+\s*[\(:\{]',
            r'(?:import|from|require)\s+[\'"\w]',
        ],
        ContentType.EMAIL: [
            r'(?m)^from:\s*\S+@\S+',
            r'(?m)^to:\s*\S+@\S+',
            r'(?m)^subject:\s*',
            r'(?m)^date:\s*',
        ],
        ContentType.FINANCIAL_REPORT: [
            r'\b10-[kq]\b',
            r'\bannual\s+report\b.*\b(?:\d{4}|fiscal)\b',
            r'\bconsolidated\s+(?:statements|financial)\b',
            r'\bsec\s+filing\b',
        ],
    }
    
    def classify(self, text: str, features: ClassificationFeatures) -> ContentType:
        """Classify content type"""
        scores = defaultdict(float)
        
        for content_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[content_type] += 1.0
        
        # Additional heuristics
        if features.has_code_blocks:
            scores[ContentType.CODE] += 2.0
        
        if features.has_tables and scores[ContentType.RESEARCH_PAPER] > 0:
            scores[ContentType.RESEARCH_PAPER] += 1.0
        
        # Check for very short text (likely social post)
        if features.word_count < 100:
            scores[ContentType.SOCIAL_POST] += 0.5
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return ContentType.TEXT


class AdvancedDomainClassifier:
    """
    Production-grade domain classifier with multiple classification strategies.
    
    Features:
    - Multi-modal classification (embeddings, statistics, patterns)
    - Confidence calibration
    - Uncertainty quantification
    - Multi-domain detection
    - Active learning support
    """
    
    def __init__(self, 
                 use_embeddings: bool = True,
                 use_statistical: bool = True,
                 calibration_model_path: Optional[str] = None):
        """
        Initialize advanced domain classifier.
        
        Args:
            use_embeddings: Whether to use embedding-based classification
            use_statistical: Whether to use statistical classification
            calibration_model_path: Path to confidence calibration model
        """
        self.feature_extractor = FeatureExtractor()
        
        self.classifiers = {}
        if use_embeddings:
            try:
                self.classifiers['embedding'] = EmbeddingClassifier()
            except Exception as e:
                logger.warning(f"Could not initialize embedding classifier: {e}")
        
        if use_statistical:
            self.classifiers['statistical'] = StatisticalClassifier()
        
        self.content_classifier = ContentTypeClassifier()
        
        # Classification history for calibration
        self.classification_history: List[Tuple[ClassificationResult, bool]] = []
        self.calibration_model: Optional[Any] = None
        
        if calibration_model_path and Path(calibration_model_path).exists():
            self._load_calibration_model(calibration_model_path)
        
        logger.info(f"AdvancedDomainClassifier initialized with {len(self.classifiers)} classifiers")
    
    def classify(self, text: str, 
                 hint: Optional[DomainCategory] = None,
                 return_all_scores: bool = False) -> ClassificationResult:
        """
        Classify text into domain categories.
        
        Args:
            text: Input text to classify
            hint: Optional hint about the domain
            return_all_scores: Whether to return scores for all domains
            
        Returns:
            ClassificationResult with comprehensive classification data
        """
        if not text or not text.strip():
            return ClassificationResult(
                primary_domain=DomainCategory.GENERAL,
                confidence=0.0,
                calibrated_confidence=0.0,
                uncertainty_estimate=1.0
            )
        
        # Extract features
        features = self.feature_extractor.extract_features(text)
        
        # Collect predictions from all classifiers
        all_predictions: List[DomainPrediction] = []
        methods_used = []
        
        # Embedding classifier
        if 'embedding' in self.classifiers:
            pred = self.classifiers['embedding'].classify(text, features)
            if pred:
                all_predictions.append(pred)
                methods_used.append('embedding')
        
        # Statistical classifier
        if 'statistical' in self.classifiers:
            preds = self.classifiers['statistical'].classify(text, features)
            all_predictions.extend(preds)
            methods_used.append('statistical')
        
        # Aggregate predictions
        aggregated = self._aggregate_predictions(all_predictions, hint)
        
        # Classify content type
        content_type = self.content_classifier.classify(text, features)
        
        # Detect multi-domain content
        is_multi_domain = self._detect_multi_domain(aggregated)
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(aggregated, methods_used)
        
        # Calibrate confidence
        raw_confidence = aggregated[0].confidence if aggregated else 0.0
        calibrated_confidence = self._calibrate_confidence(raw_confidence, features, methods_used)
        
        # Build distribution
        distribution = {p.domain.value: p.confidence for p in aggregated}
        
        # Get recommendations
        recommendations = self._get_component_recommendations(aggregated)
        
        result = ClassificationResult(
            primary_domain=aggregated[0].domain if aggregated else DomainCategory.GENERAL,
            confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            secondary_domains=aggregated[1:] if len(aggregated) > 1 else [],
            content_type=content_type,
            features=features if return_all_scores else None,
            is_multi_domain=is_multi_domain,
            domain_distribution=distribution,
            recommended_components=recommendations,
            classification_methods=methods_used,
            uncertainty_estimate=uncertainty
        )
        
        return result
    
    def _aggregate_predictions(self, predictions: List[DomainPrediction],
                               hint: Optional[DomainCategory]) -> List[DomainPrediction]:
        """Aggregate predictions from multiple classifiers"""
        if not predictions:
            return [DomainPrediction(DomainCategory.GENERAL, 0.5, 0.5, [])]
        
        # Group by domain
        by_domain: Dict[DomainCategory, List[DomainPrediction]] = defaultdict(list)
        for pred in predictions:
            by_domain[pred.domain].append(pred)
        
        # Average confidence for each domain
        aggregated = []
        for domain, preds in by_domain.items():
            avg_confidence = statistics.mean(p.confidence for p in preds)
            avg_raw = statistics.mean(p.raw_score for p in preds)
            all_evidence = [e for p in preds for e in p.evidence]
            
            # Boost if matches hint
            if hint and domain == hint:
                avg_confidence = min(avg_confidence * 1.2, 1.0)
            
            aggregated.append(DomainPrediction(
                domain=domain,
                confidence=avg_confidence,
                raw_score=avg_raw,
                evidence=all_evidence
            ))
        
        # Sort by confidence
        aggregated.sort(key=lambda x: x.confidence, reverse=True)
        return aggregated
    
    def _detect_multi_domain(self, predictions: List[DomainPrediction]) -> bool:
        """Detect if content spans multiple domains"""
        if len(predictions) < 2:
            return False
        
        # Check if second domain has significant confidence
        if predictions[1].confidence > 0.3:
            # Check if it's close to primary
            ratio = predictions[1].confidence / max(predictions[0].confidence, 0.001)
            if ratio > 0.6:
                return True
        
        return False
    
    def _calculate_uncertainty(self, predictions: List[DomainPrediction],
                                methods_used: List[str]) -> float:
        """Calculate uncertainty estimate"""
        if not predictions:
            return 1.0
        
        # High uncertainty if low confidence
        uncertainty = 1.0 - predictions[0].confidence
        
        # Higher uncertainty if few methods
        if len(methods_used) < 2:
            uncertainty += 0.2
        
        # Higher uncertainty if close second
        if len(predictions) > 1:
            gap = predictions[0].confidence - predictions[1].confidence
            if gap < 0.2:
                uncertainty += 0.15
        
        return min(uncertainty, 1.0)
    
    def _calibrate_confidence(self, raw_confidence: float,
                              features: ClassificationFeatures,
                              methods_used: List[str]) -> float:
        """Calibrate confidence score"""
        # Temperature scaling (simplified)
        temperature = 1.0
        
        # Adjust based on text length (longer texts tend to be more confidently classified)
        if features.word_count > 500:
            temperature -= 0.1
        elif features.word_count < 50:
            temperature += 0.2
        
        # Adjust based on methods
        if len(methods_used) >= 2:
            temperature -= 0.1
        
        # Apply temperature scaling
        calibrated = raw_confidence / max(temperature, 0.5)
        return min(calibrated, 1.0)
    
    def _get_component_recommendations(self, predictions: List[DomainPrediction]) -> List[str]:
        """Get recommended components based on classification"""
        if not predictions:
            return ['deepke', 'kg_gen']
        
        primary = predictions[0].domain
        
        # Base components
        base = ['deepke', 'kg_gen', 'karate_club']
        
        # Domain-specific additions
        additions = {
            DomainCategory.CHEMISTRY: ['global_chem'],
            DomainCategory.HEALTHCARE: ['global_chem', 'causal_learn'],
            DomainCategory.FINANCE: ['causal_learn', 'pami'],
            DomainCategory.RESEARCH: ['pami', 'neuralkg', 'causal_learn'],
            DomainCategory.TECHNOLOGY: ['pami'],
            DomainCategory.BIOLOGY: ['global_chem'],
            DomainCategory.PHYSICS: ['neuromancer', 'lagrange_mapper'],
            DomainCategory.MATHEMATICS: ['causal_learn', 'lagrange_mapper'],
            DomainCategory.SOCIAL_MEDIA: ['pami'],
            DomainCategory.ENVIRONMENT: ['causal_learn', 'neuromancer'],
            DomainCategory.LEGAL: ['pami'],
            DomainCategory.BUSINESS: ['causal_learn', 'pami'],
        }
        
        components = base + additions.get(primary, [])
        
        # Add secondary domain components if multi-domain
        if len(predictions) > 1 and predictions[1].confidence > 0.3:
            secondary_additions = additions.get(predictions[1].domain, [])
            components.extend([c for c in secondary_additions if c not in components])
        
        return components
    
    def _load_calibration_model(self, path: str):
        """Load confidence calibration model"""
        try:
            with open(path, 'rb') as f:
                self.calibration_model = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load calibration model: {e}")
    
    def record_feedback(self, result: ClassificationResult, was_correct: bool):
        """Record classification feedback for learning"""
        self.classification_history.append((result, was_correct))
        
        # Keep history manageable
        if len(self.classification_history) > 10000:
            self.classification_history = self.classification_history[-5000:]
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics"""
        if not self.classification_history:
            return {"samples": 0}
        
        correct = sum(1 for _, c in self.classification_history if c)
        total = len(self.classification_history)
        
        # Bin by confidence
        bins = defaultdict(lambda: {'correct': 0, 'total': 0})
        for result, was_correct in self.classification_history:
            conf_bin = int(result.confidence * 10) / 10   # 0.1, 0.2, etc.
            bins[conf_bin]['total'] += 1
            if was_correct:
                bins[conf_bin]['correct'] += 1
        
        return {
            "samples": total,
            "accuracy": correct / total,
            "bin_accuracy": {
                f"{b:.1f}": v['correct'] / max(v['total'], 1)
                for b, v in sorted(bins.items())
            }
        }


# Convenience function
def classify_text(text: str, hint: Optional[str] = None) -> ClassificationResult:
    """Quick classification function"""
    classifier = AdvancedDomainClassifier()
    hint_enum = DomainCategory(hint.lower()) if hint else None
    return classifier.classify(text, hint_enum)
