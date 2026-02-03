"""
Domain Classifier for Adaptive Knowledge Engine

Automatically categorizes input data to determine optimal processing strategy.
Uses multiple classification methods:
1. LLM-based classification (via prompts)
2. Heuristic/rule-based classification
3. Content analysis (keywords, patterns)
4. Historical pattern matching from learning engine

The classifier adapts over time based on successful classifications.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class DomainCategory(Enum):
    """Domain categories for classification"""
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


@dataclass
class ClassificationResult:
    """Result of domain classification"""
    primary_domain: DomainCategory
    confidence: float
    secondary_domains: List[Tuple[DomainCategory, float]] = field(default_factory=list)
    content_type: ContentType = ContentType.TEXT
    features_detected: List[str] = field(default_factory=list)
    recommended_components: List[str] = field(default_factory=list)
    classification_method: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_domain': self.primary_domain.value,
            'confidence': self.confidence,
            'secondary_domains': [(d.value, c) for d, c in self.secondary_domains],
            'content_type': self.content_type.value,
            'features_detected': self.features_detected,
            'recommended_components': self.recommended_components,
            'classification_method': self.classification_method,
            'timestamp': self.timestamp
        }


class DomainClassifier:
    """
    Multi-method domain classifier for automatic input categorization.
    
    Classification methods (in order of priority):
    1. Explicit user-provided domain
    2. LLM-based classification (if available)
    3. Pattern/heuristic matching
    4. Historical learning from similar inputs
    5. Content analysis (keywords, entities)
    """
    
    # Domain-specific keyword patterns
    DOMAIN_PATTERNS = {
        DomainCategory.FINANCE: [
            r'\b(?:stock|market|trading|investment|portfolio|equity|bond|dividend|earnings|revenue|profit|loss|financial|fiscal|quarter|q[1-4]|annual report|sec filing|10-k|10-q|balance sheet|income statement|cash flow)\b',
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Dollar amounts
            r'\b(?:bull|bear|market cap|ipo|merger|acquisition)\b'
        ],
        DomainCategory.CHEMISTRY: [
            r'\b(?:molecule|compound|reaction|catalyst|synthesis|organic|inorganic|polymer|acid|base|ph|solution|solvent|chemical|element|periodic|bond|ion)\b',
            r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\b',  # Chemical formulas (simple)
            r'\b(?:smiles|inchi|cas number|molecular weight|mw)\b'
        ],
        DomainCategory.HEALTHCARE: [
            r'\b(?:patient|diagnosis|treatment|symptom|disease|condition|medical|clinical|hospital|doctor|physician|medication|drug|dosage|therapy|surgery)\b',
            r'\b(?:icd-?10|icd-?9|cpt|hl7|ehr|emr|hipaa)\b',
            r'\b(?:blood pressure|heart rate|temperature|lab result|biomarker)\b'
        ],
        DomainCategory.LEGAL: [
            r'\b(?:contract|agreement|clause|provision|liability|jurisdiction|plaintiff|defendant|court|litigation|arbitration|statute|regulation|compliance)\b',
            r'\b(?:hereinafter|whereas|witnesseth|party of the first part)\b',
            r'\b\d+\s+(?:u\.s\.c\.|c\.f\.r\.|fed\.|supp)\b'  # Legal citations
        ],
        DomainCategory.RESEARCH: [
            r'\b(?:abstract|introduction|methodology|results|discussion|conclusion|references|citation|doi|journal|publication|peer review|hypothesis|experiment)\b',
            r'\b(?:et al\.|ibid\.|op cit\.|loc cit\.)\b',
            r'\b(?:figure|table|appendix|supplementary)\s+\d+\b'
        ],
        DomainCategory.TECHNOLOGY: [
            r'\b(?:software|hardware|algorithm|api|database|cloud|server|network|protocol|framework|library|module|function|class|method|variable)\b',
            r'\b(?:python|java|javascript|typescript|go|rust|cpp|c\+|sql|nosql)\b',
            r'\b(?:github|docker|kubernetes|aws|azure|gcp|ci/cd|devops)\b'
        ],
        DomainCategory.BIOLOGY: [
            r'\b(?:gene|protein|dna|rna|cell|organism|species|genome|sequence|enzyme|metabolism|pathway|organ|tissue|ecosystem)\b',
            r'\b(?:pcr|crispr|gel electrophoresis|western blot|microscopy)\b'
        ],
        DomainCategory.PHYSICS: [
            r'\b(?:force|energy|mass|velocity|acceleration|momentum|quantum|relativity|thermodynamics|electromagnetic|particle|wave|field)\b',
            r'\b(?:newton|joule|watt|electron|photon|neutron|proton)\b'
        ],
        DomainCategory.MATHEMATICS: [
            r'\b(?:equation|theorem|proof|lemma|corollary|function|derivative|integral|matrix|vector|eigenvalue|algorithm)\b',
            r'\b(?:\$[^$]+\$|\\(?:begin\{equation\}|sum|int|frac|sqrt))',  # LaTeX math
            r'\b\d+\s*[=\+\-*/]\s*\d+\b'  # Simple equations
        ],
        DomainCategory.SOCIAL_MEDIA: [
            r'\b(?:hashtag|@\w+|#\w+|like|share|follow|post|tweet|viral|trending|influencer|engagement)\b',
            r'https?://(?:twitter|facebook|instagram|linkedin|tiktok|youtube)\.com/\S+',
            r'\b(?:emoji|gif|meme|story|reel)\b'
        ],
        DomainCategory.NEWS: [
            r'\b(?:breaking|exclusive|interview|reported by|according to|sources say|officials|spokesperson)\b',
            r'\b(?:president|minister|senator|representative|governor|mayor)\b',
            r'\b(?:yesterday|today|this morning|last night|just announced)\b'
        ],
        DomainCategory.BUSINESS: [
            r'\b(?:strategy|marketing|sales|customer|product|service|revenue|growth|roi|kpi|metric|conversion|lead|prospect)\b',
            r'\b(?:b2b|b2c|saas|startup|enterprise|stakeholder|shareholder)\b'
        ],
        DomainCategory.EDUCATION: [
            r'\b(?:course|curriculum|syllabus|assignment|exam|quiz|grade|student|teacher|professor|university|college|school)\b',
            r'\b(?:learning outcome|objective|assessment|rubric|credit hour|degree)\b'
        ],
        DomainCategory.GOVERNMENT: [
            r'\b(?:policy|legislation|regulation|agency|department|federal|state|local|public|citizen|constituent)\b',
            r'\b(?:bill|act|law|executive order|memorandum|circular)\b'
        ],
        DomainCategory.ENVIRONMENT: [
            r'\b(?:climate|carbon|emission|renewable|sustainability|conservation|biodiversity|ecosystem|pollution|green)\b',
            r'\b(?:solar|wind|hydroelectric|geothermal|biomass|recycling)\b'
        ]
    }
    
    # Content type patterns
    CONTENT_TYPE_PATTERNS = {
        ContentType.RESEARCH_PAPER: [
            r'\b(?:abstract|keywords|introduction|literature review|methodology|results|discussion|conclusion|references)\b',
            r'\b(?:doi:|arxiv:|pmid:|isbn:|issn:)\s*\S+',
            r'\b(?:figure|table)\s+\d+[.:]'
        ],
        ContentType.FINANCIAL_REPORT: [
            r'\b(?:10-k|10-q|8-k|annual report|quarterly report|sec filing|form 4|proxy statement)\b',
            r'\b(?:consolidated statements|balance sheet|income statement|cash flow|notes to financial statements)\b'
        ],
        ContentType.LEGAL_DOCUMENT: [
            r'\b(?:contract|agreement|terms and conditions|privacy policy|warranty|indemnification)\b',
            r'\b(?:witness whereof|in witness whereof|signed this|executed as of)\b'
        ],
        ContentType.CODE: [
            r'\b(?:def|class|function|var|let|const|import|from|return|if|else|for|while)\b',
            r'[{};()]\s*\n',  # Code-like punctuation
            r'\b(?:#|//|/\*|\*/)\s+'  # Comments
        ],
        ContentType.EMAIL: [
            r'\b(?:from:|to:|cc:|bcc:|subject:|date:|sent:|received:)\b',
            r'\b(?:dear|hello|hi|regards|sincerely|best|thanks|thank you),?\s*\n',
            r'https?://\S+\s+unsubscribe'  # Unsubscribe links
        ],
        ContentType.SOCIAL_POST: [
            r'[@#]\w+',
            r'\b(?:rt|retweet|like|comment|share)\b',
            r'https?://t\.co/\w+'  # Twitter links
        ]
    }
    
    def __init__(self, learning_engine=None, llm_client=None):
        """
        Initialize domain classifier.
        
        Args:
            learning_engine: Optional learning engine for historical patterns
            llm_client: Optional LLM client for AI-based classification
        """
        self.learning_engine = learning_engine
        self.llm_client = llm_client
        self.classification_history = []
        
        logger.info({
            "msg": "DomainClassifier initialized",
            "llm_available": llm_client is not None,
            "learning_available": learning_engine is not None
        })
    
    def classify(self, input_data: Dict[str, Any], 
                 use_llm: bool = True,
                 use_learning: bool = True) -> ClassificationResult:
        """
        Classify input data to determine domain and optimal processing.
        
        Args:
            input_data: Input data dictionary
            use_llm: Whether to use LLM-based classification
            use_learning: Whether to use learning engine patterns
            
        Returns:
            ClassificationResult with domain, confidence, recommendations
        """
        text = input_data.get('text', '')
        explicit_domain = input_data.get('domain')
        data_type = input_data.get('data_type', 'unknown')
        
        # Method 1: Check for explicit domain
        if explicit_domain:
            try:
                domain = DomainCategory(explicit_domain.lower())
                return ClassificationResult(
                    primary_domain=domain,
                    confidence=1.0,
                    classification_method="explicit",
                    content_type=self._detect_content_type(text),
                    recommended_components=self._get_recommended_components(domain)
                )
            except ValueError:
                pass  # Not a valid domain enum value
        
        # Method 2: LLM-based classification (if available and enabled)
        if use_llm and self.llm_client:
            llm_result = self._classify_with_llm(text)
            if llm_result and llm_result.confidence > 0.7:
                return llm_result
        
        # Method 3: Pattern-based classification
        pattern_result = self._classify_with_patterns(text)
        
        # Method 4: Learning engine pattern matching (if available)
        if use_learning and self.learning_engine:
            learning_result = self._classify_with_learning(text, data_type)
            if learning_result:
                # Combine pattern and learning results
                pattern_result = self._combine_results(pattern_result, learning_result)
        
        # Store classification for learning
        self.classification_history.append({
            'input_hash': hashlib.md5(text[:1000].encode()).hexdigest(),
            'result': pattern_result.to_dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        return pattern_result
    
    def _classify_with_llm(self, text: str) -> Optional[ClassificationResult]:
        """Use LLM to classify domain"""
        if not self.llm_client:
            return None
        
        try:
            # Construct classification prompt
            prompt = f"""Analyze the following text and classify it into ONE of these domains:
            finance, chemistry, healthcare, legal, research, technology, biology, physics, 
            mathematics, social_media, news, business, education, government, environment, 
            or general.
            
            Also classify the content type: text, technical_document, research_paper, 
            news_article, social_post, legal_document, medical_record, financial_report, 
            email, chat_log, code, data_table, or mixed.
            
            Text excerpt (first 1000 chars): {text[:1000]}
            
            Respond in JSON format:
            {{
                "domain": "the_domain",
                "confidence": 0.0 to 1.0,
                "content_type": "the_content_type",
                "key_features": ["feature1", "feature2"],
                "reasoning": "brief explanation"
            }}
            """
            
            # Call LLM (implementation depends on client interface)
            response = self.llm_client.generate(prompt)
            
            # Parse response
            if isinstance(response, str):
                result = json.loads(response)
            else:
                result = response
            
            domain = DomainCategory(result.get('domain', 'general'))
            content_type = ContentType(result.get('content_type', 'text'))
            
            return ClassificationResult(
                primary_domain=domain,
                confidence=result.get('confidence', 0.5),
                content_type=content_type,
                features_detected=result.get('key_features', []),
                classification_method="llm",
                recommended_components=self._get_recommended_components(domain)
            )
            
        except Exception as e:
            logger.warning({
                "msg": "LLM classification failed",
                "error": str(e)
            })
            return None
    
    def _classify_with_patterns(self, text: str) -> ClassificationResult:
        """Classify using regex patterns"""
        text_lower = text.lower()
        domain_scores = {}
        features_detected = []
        
        # Score each domain based on pattern matches
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                if matches > 0:
                    score += matches
                    features_detected.append(f"{domain.value}:{pattern[:30]}")
            
            if score > 0:
                domain_scores[domain] = score
        
        # Normalize scores to confidence values
        if domain_scores:
            total_score = sum(domain_scores.values())
            normalized_scores = {
                domain: score / total_score 
                for domain, score in domain_scores.items()
            }
            
            # Sort by confidence
            sorted_domains = sorted(
                normalized_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            primary_domain = sorted_domains[0][0]
            confidence = sorted_domains[0][1]
            secondary_domains = sorted_domains[1:3]
            
            # Detect content type
            content_type = self._detect_content_type(text)
            
            return ClassificationResult(
                primary_domain=primary_domain,
                confidence=confidence,
                secondary_domains=secondary_domains,
                content_type=content_type,
                features_detected=features_detected[:10],  # Top 10 features
                classification_method="pattern_matching",
                recommended_components=self._get_recommended_components(primary_domain)
            )
        
        # No patterns matched - return general
        return ClassificationResult(
            primary_domain=DomainCategory.GENERAL,
            confidence=0.3,
            content_type=ContentType.TEXT,
            classification_method="pattern_matching",
            recommended_components=self._get_recommended_components(DomainCategory.GENERAL)
        )
    
    def _classify_with_learning(self, text: str, data_type: str) -> Optional[ClassificationResult]:
        """Classify using historical learning patterns"""
        if not self.learning_engine:
            return None
        
        try:
            # Find similar experiences
            similar = self.learning_engine.find_similar_experiences(
                {'text': text, 'data_type': data_type},
                n=5
            )
            
            if not similar:
                return None
            
            # Aggregate domain from similar experiences
            domain_votes = {}
            for exp in similar:
                domain = exp.domain
                if domain not in domain_votes:
                    domain_votes[domain] = 0
                # Weight by result quality
                weight = exp.results_quality if exp.success else 0.1
                domain_votes[domain] += weight
            
            if domain_votes:
                best_domain = max(domain_votes.items(), key=lambda x: x[1])
                
                return ClassificationResult(
                    primary_domain=DomainCategory(best_domain[0]),
                    confidence=min(best_domain[1] / 2, 0.8),  # Cap at 0.8 for learning-based
                    classification_method="learning_engine",
                    recommended_components=self._get_recommended_components(
                        DomainCategory(best_domain[0])
                    )
                )
            
        except Exception as e:
            logger.warning({
                "msg": "Learning-based classification failed",
                "error": str(e)
            })
        
        return None
    
    def _combine_results(self, pattern_result: ClassificationResult,
                        learning_result: ClassificationResult) -> ClassificationResult:
        """Combine pattern and learning classification results"""
        # Weight pattern matching higher for now
        pattern_weight = 0.6
        learning_weight = 0.4
        
        if pattern_result.primary_domain == learning_result.primary_domain:
            # Agreement - boost confidence
            combined_confidence = min(
                pattern_result.confidence * pattern_weight + 
                learning_result.confidence * learning_weight + 0.1,
                1.0
            )
            
            return ClassificationResult(
                primary_domain=pattern_result.primary_domain,
                confidence=combined_confidence,
                secondary_domains=pattern_result.secondary_domains,
                content_type=pattern_result.content_type,
                features_detected=pattern_result.features_detected,
                classification_method="combined",
                recommended_components=pattern_result.recommended_components
            )
        else:
            # Disagreement - use pattern but include learning as secondary
            secondary = list(pattern_result.secondary_domains)
            secondary.append((learning_result.primary_domain, learning_result.confidence))
            secondary.sort(key=lambda x: x[1], reverse=True)
            
            return ClassificationResult(
                primary_domain=pattern_result.primary_domain,
                confidence=pattern_result.confidence * 0.9,  # Slightly reduce confidence
                secondary_domains=secondary[:3],
                content_type=pattern_result.content_type,
                features_detected=pattern_result.features_detected,
                classification_method="combined_with_disagreement",
                recommended_components=pattern_result.recommended_components
            )
    
    def _detect_content_type(self, text: str) -> ContentType:
        """Detect content type from patterns"""
        text_lower = text.lower()
        scores = {}
        
        for content_type, patterns in self.CONTENT_TYPE_PATTERNS.items():
            score = sum(
                1 for pattern in patterns 
                if re.search(pattern, text_lower, re.IGNORECASE)
            )
            if score > 0:
                scores[content_type] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return ContentType.TEXT
    
    def _get_recommended_components(self, domain: DomainCategory) -> List[str]:
        """Get recommended components for a domain"""
        # Base components for all domains
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
        }
        
        return base + additions.get(domain, [])
    
    def get_classifier_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""
        if not self.classification_history:
            return {"total_classifications": 0}
        
        domain_counts = {}
        for entry in self.classification_history:
            domain = entry['result']['primary_domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {
            "total_classifications": len(self.classification_history),
            "domain_distribution": domain_counts,
            "methods_used": list(set(
                e['result']['classification_method'] 
                for e in self.classification_history
            ))
        }


# Convenience function
def classify_input(input_data: Dict[str, Any], 
                   learning_engine=None,
                   llm_client=None) -> ClassificationResult:
    """
    Quick classification function.
    
    Args:
        input_data: Input data dictionary
        learning_engine: Optional learning engine
        llm_client: Optional LLM client
        
    Returns:
        ClassificationResult
    """
    classifier = DomainClassifier(learning_engine, llm_client)
    return classifier.classify(input_data)
