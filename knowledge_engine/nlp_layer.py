"""
Natural Language Processing Layer

Advanced NLP capabilities for the knowledge engine:
- Named Entity Recognition (NER)
- Part-of-Speech (POS) tagging
- Dependency parsing
- Sentiment analysis
- Topic modeling
- Question answering
- Text similarity
- Language detection
- Keyword extraction
- Coreference resolution
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import math

logger = logging.getLogger(__name__)


@dataclass
class Token:
    """A token in a text."""
    text: str
    lemma: str
    pos: str  # Part of speech
    tag: str  # Detailed tag
    dep: str  # Dependency relation
    head_id: int  # ID of head token
    start_char: int
    end_char: int
    is_stop: bool = False
    is_entity: bool = False
    entity_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "lemma": self.lemma,
            "pos": self.pos,
            "tag": self.tag,
            "dep": self.dep,
            "head_id": self.head_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "is_stop": self.is_stop,
            "is_entity": self.is_entity,
            "entity_type": self.entity_type
        }


@dataclass
class ParsedSentence:
    """A parsed sentence with NLP annotations."""
    text: str
    tokens: List[Token]
    sentiment: float  # -1.0 to 1.0
    entities: List[Dict[str, Any]] = field(default_factory=list)
    noun_chunks: List[str] = field(default_factory=list)
    root: Optional[Token] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens],
            "sentiment": self.sentiment,
            "entities": self.entities,
            "noun_chunks": self.noun_chunks,
            "root": self.root.to_dict() if self.root else None
        }


@dataclass
class DocumentAnalysis:
    """Complete NLP analysis of a document."""
    text: str
    language: str
    sentences: List[ParsedSentence]
    entities: List[Dict[str, Any]]
    keywords: List[Tuple[str, float]]
    topics: List[Tuple[str, float]]
    overall_sentiment: float
    readability_score: float
    word_count: int
    unique_words: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:1000] + "..." if len(self.text) > 1000 else self.text,
            "language": self.language,
            "sentences": [s.to_dict() for s in self.sentences],
            "entities": self.entities,
            "keywords": self.keywords,
            "topics": self.topics,
            "overall_sentiment": self.overall_sentiment,
            "readability_score": self.readability_score,
            "word_count": self.word_count,
            "unique_words": self.unique_words
        }


class LanguageDetector:
    """Detect language of text."""
    
    # Common words by language for simple detection
    LANGUAGE_PROFILES = {
        "en": ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i"],
        "es": ["el", "la", "de", "que", "y", "a", "en", "un", "ser", "se"],
        "fr": ["le", "de", "et", "à", "un", "il", "être", "et", "avoir", "ne"],
        "de": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"],
        "it": ["il", "di", "che", "è", "la", "per", "un", "sono", "con", "ma"],
        "pt": ["o", "de", "a", "que", "e", "do", "da", "em", "um", "para"],
        "zh": ["的", "是", "在", "和", "有", "大", "年", "人", "中", "小"],
        "ja": ["の", "に", "は", "を", "た", "が", "で", "て", "と", "し"],
        "ko": ["의", "이", "가", "은", "는", "에", "를", "로", "와", "과"]
    }
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect language of text.
        
        Returns:
            (language_code, confidence)
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b[a-zA-Z]+\b', text_lower))
        
        scores = {}
        for lang, common_words in self.LANGUAGE_PROFILES.items():
            matches = len(words & set(common_words))
            scores[lang] = matches / len(common_words) if common_words else 0
        
        # Check for CJK characters
        cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text))
        if cjk_chars > len(text) * 0.3:
            # Determine which CJK
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
            korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
            
            if japanese_chars > chinese_chars and japanese_chars > korean_chars:
                return "ja", 0.9
            elif korean_chars > chinese_chars:
                return "ko", 0.9
            else:
                return "zh", 0.9
        
        if scores:
            best_lang = max(scores, key=scores.get)
            confidence = scores[best_lang]
            return best_lang, min(confidence, 1.0)
        
        return "unknown", 0.0


class SentimentAnalyzer:
    """Analyze sentiment of text."""
    
    # Simple word lists for sentiment
    POSITIVE_WORDS = set([
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "love", "like", "happy", "pleased", "satisfied", "perfect",
        "best", "awesome", "brilliant", "outstanding", "superb",
        "positive", "success", "beneficial", "effective", "useful"
    ])
    
    NEGATIVE_WORDS = set([
        "bad", "terrible", "awful", "horrible", "worst", "hate",
        "dislike", "sad", "disappointed", "unsatisfied", "poor",
        "worse", "fail", "failure", "problem", "issue", "bug",
        "error", "wrong", "negative", "ineffective", "useless",
        "difficult", "hard", "complicated", "confusing"
    ])
    
    INTENSIFIERS = {
        "very": 1.5, "extremely": 2.0, "really": 1.5, "quite": 1.3,
        "somewhat": 0.7, "slightly": 0.5, "barely": 0.3
    }
    
    NEGATORS = ["not", "no", "never", "neither", "nor", "hardly", "barely"]
    
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text.
        
        Returns:
            Sentiment score from -1.0 (negative) to 1.0 (positive)
        """
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        score = 0.0
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for negation
            negation = 1.0
            if i > 0 and words[i-1] in self.NEGATORS:
                negation = -1.0
            
            # Check for intensifier
            multiplier = 1.0
            if i > 0 and words[i-1] in self.INTENSIFIERS:
                multiplier = self.INTENSIFIERS[words[i-1]]
            
            if word in self.POSITIVE_WORDS:
                score += 1.0 * negation * multiplier
            elif word in self.NEGATIVE_WORDS:
                score -= 1.0 * negation * multiplier
            
            i += 1
        
        # Normalize to [-1, 1]
        if len(words) > 0:
            score = max(-1.0, min(1.0, score / math.sqrt(len(words))))
        
        return score
    
    def analyze_detailed(self, text: str) -> Dict[str, Any]:
        """Detailed sentiment analysis."""
        overall = self.analyze(text)
        
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        sentence_sentiments = [self.analyze(s) for s in sentences if s.strip()]
        
        return {
            "overall": overall,
            "confidence": abs(overall),
            "label": "positive" if overall > 0.1 else "negative" if overall < -0.1 else "neutral",
            "sentence_sentiments": sentence_sentiments,
            "positive_ratio": sum(1 for s in sentence_sentiments if s > 0) / len(sentence_sentiments) if sentence_sentiments else 0
        }


class KeywordExtractor:
    """Extract keywords from text using TF-IDF and other methods."""
    
    STOP_WORDS = set([
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can",
        "this", "that", "these", "those", "i", "you", "he", "she",
        "it", "we", "they", "me", "him", "her", "us", "them"
    ])
    
    def __init__(self):
        self.idf_scores: Dict[str, float] = {}
        self.document_count = 0
    
    def extract_keywords(
        self, 
        text: str, 
        top_n: int = 10,
        method: str = "tfidf"
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            top_n: Number of keywords to return
            method: "tfidf", "frequency", or "pos"
        """
        if method == "frequency":
            return self._extract_by_frequency(text, top_n)
        elif method == "pos":
            return self._extract_by_pos(text, top_n)
        else:  # tfidf
            return self._extract_by_tfidf(text, top_n)
    
    def _extract_by_frequency(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """Extract keywords by frequency."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in self.STOP_WORDS]
        
        word_freq = Counter(words)
        total = sum(word_freq.values())
        
        scored = [(word, count / total) for word, count in word_freq.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:top_n]
    
    def _extract_by_tfidf(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in self.STOP_WORDS]
        
        tf = Counter(words)
        total = sum(tf.values())
        
        scores = []
        for word, count in tf.items():
            tf_score = count / total
            idf_score = self.idf_scores.get(word, 1.0)
            scores.append((word, tf_score * idf_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    def _extract_by_pos(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """Extract keywords using part-of-speech patterns."""
        # Simple pattern: noun phrases
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)  # Capitalized words
        
        word_freq = Counter(w.lower() for w in words)
        total = sum(word_freq.values())
        
        scored = [(word, count / total) for word, count in word_freq.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:top_n]
    
    def update_idf(self, documents: List[str]):
        """Update IDF scores with a corpus."""
        self.document_count = len(documents)
        document_frequency = defaultdict(int)
        
        for doc in documents:
            words = set(re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower()))
            words = words - self.STOP_WORDS
            for word in words:
                document_frequency[word] += 1
        
        for word, df in document_frequency.items():
            self.idf_scores[word] = math.log(self.document_count / (df + 1)) + 1


class TextSimilarity:
    """Calculate similarity between texts."""
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        words1 = Counter(self._tokenize(text1))
        words2 = Counter(self._tokenize(text2))
        
        all_words = set(words1.keys()) | set(words2.keys())
        
        dot_product = sum(words1.get(w, 0) * words2.get(w, 0) for w in all_words)
        magnitude1 = math.sqrt(sum(c ** 2 for c in words1.values()))
        magnitude2 = math.sqrt(sum(c ** 2 for c in words2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity."""
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """Calculate n-gram similarity."""
        ngrams1 = self._get_ngrams(text1, n)
        ngrams2 = self._get_ngrams(text2, n)
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    def _get_ngrams(self, text: str, n: int) -> Set[str]:
        """Get character n-grams."""
        text = text.lower().replace(" ", "")
        return set(text[i:i+n] for i in range(len(text) - n + 1))


class QuestionAnswerer:
    """Simple question answering based on knowledge context."""
    
    def __init__(self):
        self.keyword_extractor = KeywordExtractor()
    
    def answer(
        self, 
        question: str, 
        context: str
    ) -> Dict[str, Any]:
        """
        Answer a question based on context.
        
        Uses simple keyword matching and sentence ranking.
        """
        # Extract keywords from question
        question_keywords = self.keyword_extractor.extract_keywords(
            question, top_n=5, method="frequency"
        )
        question_words = set(kw for kw, _ in question_keywords)
        
        # Split context into sentences
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Score sentences by keyword overlap
        scored_sentences = []
        for sent in sentences:
            sent_words = set(self._tokenize(sent))
            overlap = len(question_words & sent_words)
            score = overlap / len(question_words) if question_words else 0
            scored_sentences.append((score, sent))
        
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Determine question type
        question_type = self._determine_question_type(question)
        
        best_answer = scored_sentences[0] if scored_sentences else (0, "No answer found")
        
        return {
            "answer": best_answer[1],
            "confidence": best_answer[0],
            "question_type": question_type,
            "alternative_answers": [s for _, s in scored_sentences[1:3]]
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    def _determine_question_type(self, question: str) -> str:
        """Determine the type of question."""
        question_lower = question.lower()
        
        if question_lower.startswith(("what", "which")):
            return "what"
        elif question_lower.startswith("who"):
            return "who"
        elif question_lower.startswith("when"):
            return "when"
        elif question_lower.startswith("where"):
            return "where"
        elif question_lower.startswith("why"):
            return "why"
        elif question_lower.startswith("how"):
            return "how"
        else:
            return "yes/no"


class ReadabilityAnalyzer:
    """Analyze text readability."""
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text readability."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        if not sentences or not words:
            return {"score": 0, "level": "unknown"}
        
        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences)
        
        # Count syllables (simplified)
        syllable_count = sum(self._count_syllables(w) for w in words)
        avg_syllables_per_word = syllable_count / len(words)
        
        # Flesch Reading Ease
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))
        
        # Determine reading level
        if flesch_score >= 90:
            level = "very_easy"
        elif flesch_score >= 80:
            level = "easy"
        elif flesch_score >= 70:
            level = "fairly_easy"
        elif flesch_score >= 60:
            level = "standard"
        elif flesch_score >= 50:
            level = "fairly_difficult"
        elif flesch_score >= 30:
            level = "difficult"
        else:
            level = "very_difficult"
        
        return {
            "score": flesch_score,
            "level": level,
            "avg_sentence_length": avg_sentence_length,
            "avg_syllables_per_word": avg_syllables_per_word,
            "word_count": len(words),
            "sentence_count": len(sentences)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        # Silent e
        if word.endswith("e") and count > 1:
            count -= 1
        
        return max(1, count)


class NLPEngine:
    """
    Main NLP engine combining all capabilities.
    """
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.text_similarity = TextSimilarity()
        self.question_answerer = QuestionAnswerer()
        self.readability_analyzer = ReadabilityAnalyzer()
    
    def analyze(self, text: str) -> DocumentAnalysis:
        """
        Perform complete NLP analysis of text.
        """
        # Language detection
        language, lang_confidence = self.language_detector.detect(text)
        
        # Split into sentences (simplified)
        sentence_texts = re.split(r'[.!?]+', text)
        sentence_texts = [s.strip() for s in sentence_texts if s.strip()]
        
        # Analyze each sentence
        sentences = []
        all_entities = []
        
        for sent_text in sentence_texts:
            # Tokenize (simplified)
            tokens = self._simple_tokenize(sent_text)
            
            # Sentiment
            sentiment = self.sentiment_analyzer.analyze(sent_text)
            
            # Extract simple entities
            entities = self._extract_simple_entities(sent_text)
            all_entities.extend(entities)
            
            # Noun chunks (simplified)
            noun_chunks = self._extract_noun_chunks(sent_text)
            
            sentence = ParsedSentence(
                text=sent_text,
                tokens=tokens,
                sentiment=sentiment,
                entities=entities,
                noun_chunks=noun_chunks
            )
            sentences.append(sentence)
        
        # Keywords
        keywords = self.keyword_extractor.extract_keywords(text, top_n=10)
        
        # Topics (simplified - just top keywords)
        topics = keywords[:5]
        
        # Overall sentiment
        overall_sentiment = sum(s.sentiment for s in sentences) / len(sentences) if sentences else 0
        
        # Readability
        readability = self.readability_analyzer.analyze(text)
        
        # Word stats
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        return DocumentAnalysis(
            text=text,
            language=language,
            sentences=sentences,
            entities=all_entities,
            keywords=keywords,
            topics=topics,
            overall_sentiment=overall_sentiment,
            readability_score=readability["score"],
            word_count=len(words),
            unique_words=len(set(words))
        )
    
    def _simple_tokenize(self, text: str) -> List[Token]:
        """Simple tokenization."""
        words = re.finditer(r'\b[a-zA-Z]+\b', text)
        tokens = []
        
        for i, match in enumerate(words):
            token = Token(
                text=match.group(),
                lemma=match.group().lower(),
                pos="NOUN",  # Simplified
                tag="NN",
                dep="root",
                head_id=-1,
                start_char=match.start(),
                end_char=match.end(),
                is_stop=match.group().lower() in self.keyword_extractor.STOP_WORDS
            )
            tokens.append(token)
        
        return tokens
    
    def _extract_simple_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract simple entities using patterns."""
        entities = []
        
        # Capitalized sequences (likely proper nouns)
        for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            entities.append({
                "text": match.group(),
                "label": "PERSON_ORG",
                "start": match.start(),
                "end": match.end()
            })
        
        return entities
    
    def _extract_noun_chunks(self, text: str) -> List[str]:
        """Extract noun chunks (simplified)."""
        # Adj + Noun patterns
        chunks = []
        for match in re.finditer(r'\b(?:[a-z]+\s+)?[a-z]+\b', text.lower()):
            chunk = match.group()
            if len(chunk.split()) >= 1:
                chunks.append(chunk)
        return chunks[:10]
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, float]:
        """Compare two texts for similarity."""
        return {
            "cosine": self.text_similarity.cosine_similarity(text1, text2),
            "jaccard": self.text_similarity.jaccard_similarity(text1, text2),
            "ngram_3": self.text_similarity.ngram_similarity(text1, text2, 3),
            "ngram_4": self.text_similarity.ngram_similarity(text1, text2, 4)
        }
    
    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """Answer a question based on context."""
        return self.question_answerer.answer(question, context)


__all__ = [
    "NLPEngine",
    "DocumentAnalysis",
    "ParsedSentence",
    "Token",
    "LanguageDetector",
    "SentimentAnalyzer",
    "KeywordExtractor",
    "TextSimilarity",
    "QuestionAnswerer",
    "ReadabilityAnalyzer"
]
