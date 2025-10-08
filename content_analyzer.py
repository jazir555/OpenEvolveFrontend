"""
Content Analyzer Component for OpenEvolve
Implements the core Content Analyzer functionality described in the ultimate explanation document.
"""
import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textstat import flesch_reading_ease, flesch_kincaid_grade
import hashlib

try:
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import wordnet
except Exception:
    # Handle cases where NLTK downloads fail
    wordnet = None

class ContentType(Enum):
    """Enumeration of content types for categorization"""
    CODE = "code"
    DOCUMENT = "document"
    PROTOCOL = "protocol"
    LEGAL = "legal"
    MEDICAL = "medical"
    TECHNICAL = "technical"
    GENERAL = "general"

@dataclass
class ContentAnalysisResult:
    """Data class for content analysis results"""
    input_parsing: Dict[str, Any]
    semantic_understanding: Dict[str, Any]
    pattern_recognition: Dict[str, Any]
    metadata_extraction: Dict[str, Any]
    overall_score: float
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]

class ContentAnalyzer:
    """Main content analysis component that implements parsing, understanding, pattern recognition and metadata extraction"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english')) if 'stopwords' in dir(nltk.corpus) else set()
        self.content_type_patterns = {
            ContentType.CODE: [
                r'def\s+\w+',  # Python function
                r'function\s+\w+',  # JavaScript/other function
                r'public|private|protected',  # Java/C# access modifiers
                r'#include',  # C/C++ includes
                r'use\s+.*?;',  # Rust/Perl includes
                r'package\s+\w+',  # Package declarations
                r'class\s+\w+',  # Class declarations
                r'import\s+.*?;',  # Import statements
                r'var\s+\w+|let\s+\w+|const\s+\w+',  # Variable declarations
            ],
            ContentType.LEGAL: [
                r'\bcontract\b', r'\bagreement\b', r'\bclause\b', r'\bparty\b',
                r'\bjurisdiction\b', r'\bliability\b', r'\bindemnity\b', r'\bconfidentiality\b',
                r'section\s+\d+', r'§', r'et al\.', r'supra', r'infra'
            ],
            ContentType.MEDICAL: [
                r'\bpatient\b', r'\bdiagnosis\b', r'\btreatment\b', r'\bsymptom\b',
                r'\bmedication\b', r'\bclinic\b', r'\bhospital\b', r'\bclinical\b',
                r'PID-', r'\bprotocol\b', r'\bprocedure\b'
            ],
            ContentType.TECHNICAL: [
                r'\bAPI\b', r'\bdatabase\b', r'\bserver\b', r'\bclient\b', r'\balgorithm\b',
                r'\bframework\b', r'\blibrary\b', r'\bfunction\b', r'\bclass\b', r'\bmethod\b',
                r'\binterface\b', r'\bendpoint\b'
            ]
        }

    def analyze_content(self, content: str) -> ContentAnalysisResult:
        """
        Analyzes content using all analyzer components
        
        Args:
            content: The content to analyze
            
        Returns:
            ContentAnalysisResult containing all analysis results
        """
        input_parsing = self.parse_content(content)
        semantic_understanding = self.understand_content(content)
        pattern_recognition = self.recognize_patterns(content)
        metadata_extraction = self.extract_metadata(content)
        
        # Calculate overall score based on various factors
        readability_score = semantic_understanding.get('readability_score', 50) / 100  # Normalize to 0-1
        complexity_score = input_parsing.get('complexity_score', 50) / 100  # Normalize to 0-1
        completeness_score = 0.8  # Default high completeness unless issues are found
        
        # Adjust completeness based on issues found
        issues_found = pattern_recognition.get('structural_issues', []) + pattern_recognition.get('logical_anomalies', [])
        if len(issues_found) > 5:
            completeness_score = 0.3
        elif len(issues_found) > 2:
            completeness_score = 0.6
        elif len(issues_found) > 0:
            completeness_score = 0.8
        
        overall_score = (readability_score * 0.3 + complexity_score * 0.3 + completeness_score * 0.4) * 100
        
        # Compile issues found
        issues = []
        for issue_list in [pattern_recognition.get('structural_issues', []), 
                          pattern_recognition.get('logical_anomalies', []),
                          pattern_recognition.get('compliance_violations', [])]:
            issues.extend(issue_list)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(issues, content)
        
        return ContentAnalysisResult(
            input_parsing=input_parsing,
            semantic_understanding=semantic_understanding,
            pattern_recognition=pattern_recognition,
            metadata_extraction=metadata_extraction,
            overall_score=overall_score,
            issues_found=issues,
            recommendations=recommendations
        )
    
    def parse_content(self, content: str) -> Dict[str, Any]:
        """
        Analyzes content structure, format, and type
        
        Args:
            content: The content to parse
            
        Returns:
            Dictionary containing parsing results
        """
        lines = content.split('\n')
        sentences = sent_tokenize(content)
        words = word_tokenize(content.lower())
        
        # Filter out punctuation
        words = [word for word in words if word.isalnum()]
        
        # Calculate statistics
        word_count = len(words)
        sentence_count = len(sentences)
        line_count = len(lines)
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Determine content complexity based on structure
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_paragraph_length = sentence_count / paragraph_count if paragraph_count > 0 else 0
        unique_word_ratio = len(set(words)) / word_count if word_count > 0 else 0
        
        # Calculate complexity score (0-100)
        complexity_score = min(100, max(0, 
            (word_count / 50) + 
            (avg_sentence_length * 2) + 
            (avg_paragraph_length * 1.5) + 
            (unique_word_ratio * 20) - 
            (sentence_count / 10)  # Penalty for too many short sentences
        ))
        
        # Identify structure elements
        has_headers = bool(re.search(r'^#+\s|^.*\n[=]{3,}|^.*\n[-]{3,}', content, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*+]\s|^[\d]+\.\s', content, re.MULTILINE))
        has_code_blocks = bool(re.search(r'```.*?```', content, re.DOTALL))
        has_tables = bool(re.search(r'\|.*\|', content))  # Simple table detection
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'line_count': line_count,
            'paragraph_count': paragraph_count,
            'avg_sentence_length': avg_sentence_length,
            'avg_paragraph_length': avg_paragraph_length,
            'unique_word_ratio': unique_word_ratio,
            'complexity_score': complexity_score,
            'has_headers': has_headers,
            'has_lists': has_lists,
            'has_code_blocks': has_code_blocks,
            'has_tables': has_tables,
            'structure_elements': {
                'headers': len(re.findall(r'^#+\s', content, re.MULTILINE)),
                'lists': len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE)),
                'code_blocks': len(re.findall(r'```', content)) // 2,
                'tables': len(re.findall(r'\|.*\|', content))
            },
            'content_format': self._determine_format(content)
        }
    
    def understand_content(self, content: str) -> Dict[str, Any]:
        """
        Identifies content domain and purpose using semantic analysis
        
        Args:
            content: The content to understand
            
        Returns:
            Dictionary containing semantic understanding results
        """
        # Determine content type
        content_type = self._classify_content_type(content)
        
        # Analyze readability
        readability_score = flesch_reading_ease(content)
        reading_grade = flesch_kincaid_grade(content)
        
        # Extract key terms (simple approach)
        sent_tokenize(content)
        words = word_tokenize(content.lower())
        meaningful_words = [w for w in words if w.isalnum() and w not in self.stop_words]
        
        # Identify key entities (simple approach - looking for capitalized words)
        entities = self._extract_entities(content)
        
        # Analyze sentiment (if possible)
        sentiment = self._estimate_sentiment(content)
        
        return {
            'content_type': content_type,
            'domain': self._determine_domain(content, content_type),
            'purpose': self._infer_purpose(content, content_type),
            'readability_score': readability_score,
            'reading_grade': reading_grade,
            'key_terms': meaningful_words[:20],  # Top 20 meaningful terms
            'entities': entities,
            'sentiment': sentiment,
            'semantic_complexity': self._calculate_semantic_complexity(content)
        }
    
    def recognize_patterns(self, content: str) -> Dict[str, Any]:
        """
        Detects structural patterns and anomalies using pattern recognition
        
        Args:
            content: The content to analyze for patterns
            
        Returns:
            Dictionary containing pattern recognition results
        """
        structural_issues = []
        logical_anomalies = []
        compliance_violations = []
        
        # Check for common structural issues
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Check for lines that are too long
            if len(line) > 120:
                structural_issues.append({
                    'type': 'line_too_long',
                    'line_number': i + 1,
                    'description': f'Line {i + 1} is too long ({len(line)} chars)',
                    'severity': 'medium'
                })
            
            # Check for inconsistent indentation
            if line.strip() and not line.startswith(' ') and i > 0:
                prev_line = lines[i-1] if i > 0 else ""
                if prev_line.strip() and prev_line.startswith(' '):
                    # Potential indentation issue
                    pass
        
        # Check for logical anomalies
        if re.search(r'does not|never|always', content, re.IGNORECASE):
            # Check for absolute statements that might be logical issues
            if re.search(r'(all|every|never|always|completely|totally) (users|people|systems|processes)', content, re.IGNORECASE):
                logical_anomalies.append({
                    'type': 'absolute_statement',
                    'description': 'Found absolute statement that might need refinement',
                    'severity': 'medium'
                })
        
        # Check for compliance issues based on content type
        content_type = self._classify_content_type(content)
        if content_type == ContentType.LEGAL:
            # Check for missing legal clauses
            required_terms = ['liability', 'disclaimer', 'warranty']
            for term in required_terms:
                if term not in content.lower():
                    compliance_violations.append({
                        'type': 'missing_legal_term',
                        'description': f'Missing potentially important legal term: {term}',
                        'severity': 'high'
                    })
        elif content_type == ContentType.MEDICAL:
            # Check for HIPAA compliance issues
            if re.search(r'patient.*?name|medical.*?record|personal.*?information', content, re.IGNORECASE):
                compliance_violations.append({
                    'type': 'potential_phi_exposure',
                    'description': 'Potential identification of protected health information',
                    'severity': 'critical'
                })
        elif content_type == ContentType.TECHNICAL:
            # Check for security issues in technical content
            if re.search(r'password|secret|token|key.*?=|api_key', content, re.IGNORECASE):
                compliance_violations.append({
                    'type': 'security_exposure',
                    'description': 'Potential exposure of sensitive credentials',
                    'severity': 'critical'
                })
        
        # Advanced pattern detection
        patterns = self._detect_advanced_patterns(content)
        
        return {
            'structural_issues': structural_issues,
            'logical_anomalies': logical_anomalies,
            'compliance_violations': compliance_violations,
            'advanced_patterns': patterns,
            'pattern_score': self._calculate_pattern_score(structural_issues, logical_anomalies, compliance_violations)
        }
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Gathers contextual information for processing
        
        Args:
            content: The content to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata
        """
        # Calculate content hash for version tracking
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Extract metadata from the content
        title = self._extract_title(content)
        authors = self._extract_authors(content)
        creation_date = self._extract_date(content)
        version = self._extract_version(content)
        tags = self._extract_tags(content)
        
        # Extract references/citations
        references = self._extract_references(content)
        
        return {
            'content_hash': content_hash,
            'title': title,
            'authors': authors,
            'creation_date': creation_date,
            'version': version,
            'tags': tags,
            'language': self._detect_language(content),
            'encoding': 'utf-8',  # Assuming UTF-8
            'references': references,
            'size_bytes': len(content.encode('utf-8')),
            'size_kilobytes': round(len(content.encode('utf-8')) / 1024, 2),
            'last_modified': None,  # Would come from file system if applicable
        }
    
    def _determine_format(self, content: str) -> str:
        """Determine the format of the content"""
        if content.strip().startswith('```'):
            return 'markdown_with_code'
        elif content.count('# ') > 0:
            return 'markdown'
        elif '<html' in content.lower() or '<body' in content.lower():
            return 'html'
        elif '<?xml' in content.lower():
            return 'xml'
        elif content.strip().startswith('{') and content.strip().endswith('}'):
            try:
                json.loads(content.strip())
                return 'json'
            except Exception:
                pass
        elif 'class ' in content or 'def ' in content or 'function ' in content:
            return 'code'
        else:
            return 'plain_text'
    
    def _classify_content_type(self, content: str) -> ContentType:
        """Classify the content type based on pattern matching"""
        content_lower = content.lower()
        
        for content_type, patterns in self.content_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return content_type
        
        return ContentType.GENERAL
    
    def _determine_domain(self, content: str, content_type: ContentType) -> str:
        """Determine the domain based on content type and content"""
        if content_type == ContentType.CODE:
            # Detect programming language
            if 'import' in content and 'def ' in content:
                return 'python'
            elif re.search(r'function\s+\w+\s*\(', content):
                return 'javascript'
            elif 'public class' in content or 'import java' in content:
                return 'java'
            elif '#include' in content:
                return 'cpp'
            else:
                return 'general_programming'
        
        elif content_type == ContentType.DOCUMENT:
            if any(keyword in content.lower() for keyword in ['policy', 'procedure', 'protocol']):
                return 'business'
            elif any(keyword in content.lower() for keyword in ['research', 'study', 'analysis']):
                return 'academic'
            else:
                return 'general_documentation'
        
        elif content_type == ContentType.PROTOCOL:
            return 'operational'
        
        else:
            return content_type.value
    
    def _infer_purpose(self, content: str, content_type: ContentType) -> str:
        """Infer the purpose of the content"""
        content_lower = content.lower()
        
        if content_type == ContentType.CODE:
            if any(keyword in content_lower for keyword in ['test', 'assert', 'check', 'verify']):
                return 'testing'
            elif any(keyword in content_lower for keyword in ['api', 'endpoint', 'client', 'server']):
                return 'interface'
            else:
                return 'implementation'
        
        elif content_type in [ContentType.DOCUMENT, ContentType.PROTOCOL]:
            if any(keyword in content_lower for keyword in ['requirements', 'specification', 'define']):
                return 'specification'
            elif any(keyword in content_lower for keyword in ['how to', 'tutorial', 'guide']):
                return 'instructional'
            elif any(keyword in content_lower for keyword in ['review', 'audit', 'check']):
                return 'evaluation'
            else:
                return 'informational'
        
        else:
            return 'general'
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract key entities from content (simplified approach)"""
        # Simple approach: extract capitalized words that appear to be entities
        sentences = sent_tokenize(content)
        entities = set()
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 2 and word.lower() not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']:
                    # Check if it's followed by other capitalized words (potential entity phrase)
                    if i + 1 < len(words) and words[i + 1][0].isupper():
                        entities.add(f"{word} {words[i + 1]}")
                    else:
                        entities.add(word)
        
        return list(entities)[:20]  # Limit to top 20 entities
    
    def _estimate_sentiment(self, content: str) -> Dict[str, float]:
        """Estimate sentiment of the content (simplified approach)"""
        # Simple keyword-based sentiment analysis
        positive_words = ['good', 'excellent', 'great', 'amazing', 'wonderful', 'perfect', 'best', 'superb', 'outstanding', 'fantastic', 'brilliant', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'disgusting', 'hate', 'hated', 'dislike', 'poor', 'worst', 'dreadful', 'atrocious']
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        total_words = len(content_lower.split())
        pos_ratio = positive_count / max(1, total_words)
        neg_ratio = negative_count / max(1, total_words)
        
        return {
            'positive_ratio': pos_ratio,
            'negative_ratio': neg_ratio,
            'neutral_ratio': 1 - (pos_ratio + neg_ratio),
            'compound_score': pos_ratio - neg_ratio
        }
    
    def _calculate_semantic_complexity(self, content: str) -> float:
        """Calculate semantic complexity of the content"""
        sentences = sent_tokenize(content)
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / max(1, len(sentences))
        
        # Complexity based on average sentence length, vocabulary diversity, and structure
        words = word_tokenize(content.lower())
        unique_words = set(word for word in words if word.isalnum())
        vocabulary_diversity = len(unique_words) / max(1, len(words))
        
        # Return normalized complexity score (0-100)
        complexity = (avg_sentence_length * 0.3 + vocabulary_diversity * 50)
        return min(100, max(0, complexity))
    
    def _detect_advanced_patterns(self, content: str) -> Dict[str, Any]:
        """Detect advanced patterns in the content"""
        patterns = {
            'conditional_logic': len(re.findall(r'if\s+.*?:\s*\n', content, re.IGNORECASE)),
            'loops': len(re.findall(r'for\s+.*?:|while\s+.*?:', content, re.IGNORECASE)),
            'functions': len(re.findall(r'def\s+\w+|function\s+\w+', content, re.IGNORECASE)),
            'classes': len(re.findall(r'class\s+\w+', content, re.IGNORECASE)),
            'data_structures': len(re.findall(r'\[.*?\]|\{.*?\}|\(.*?\)', content)),
            'nested_structures': len(re.findall(r'\[.*?\[|\{.*?\{|\(.*?\(', content)),
            'repetitive_patterns': self._find_repetitive_patterns(content),
            'inconsistencies': self._find_inconsistencies(content)
        }
        return patterns
    
    def _calculate_pattern_score(self, structural_issues: List, logical_anomalies: List, compliance_violations: List) -> float:
        """Calculate a pattern-based score (0-100, where 100 is best)"""
        score = 100 - (len(structural_issues) * 2 + len(logical_anomalies) * 5 + len(compliance_violations) * 10)
        
        return max(0, min(100, score))
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract title from content"""
        # Look for markdown-style title
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # Look for HTML title
        match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Look for document-style title (first significant line)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if lines and len(lines[0]) < 100 and not lines[0].startswith('#'):  # Reasonable length, not a header
            return lines[0]
        
        return None
    
    def _extract_authors(self, content: str) -> List[str]:
        """Extract authors from content"""
        # Look for author lines in various formats
        patterns = [
            r'author:\s*(.+)',
            r'authors:\s*(.+)',
            r'by:\s*(.+)',
            r'written by:\s*(.+)',
            r'created by:\s*(.+)'
        ]
        
        authors = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Split by comma or 'and' to separate multiple authors
                parts = re.split(r',|\sand\s', match)
                for part in parts:
                    part = part.strip()
                    if part and part not in authors:
                        authors.append(part)
        
        return authors
    
    def _extract_date(self, content: str) -> Optional[str]:
        """Extract date from content"""
        # Look for various date formats
        patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',   # YYYY/MM/DD or YYYY-MM-DD
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_version(self, content: str) -> Optional[str]:
        """Extract version from content"""
        patterns = [
            r'version:\s*v?(\d+\.\d+(?:\.\d+)?)',
            r'v(\d+\.\d+(?:\.\d+)?)',
            r'version\s+(\d+\.\d+(?:\.\d+)?)',
            r'@version\s+(\d+\.\d+(?:\.\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from content"""
        # Look for markdown-style tags or other tagging systems
        all_tags = set()
        
        # Look for hashtags
        hashtags = re.findall(r'#(\w+)', content)
        all_tags.update(hashtags)
        
        # Look for specially formatted tags
        special_tags = re.findall(r'\[(\w+)\]', content)
        all_tags.update(special_tags)
        
        # Look for inline keywords that might serve as tags
        content_lower = content.lower()
        keyword_indicators = ['category:', 'topic:', 'subject:', 'tags:']
        for indicator in keyword_indicators:
            if indicator in content_lower:
                # Extract content following the indicator
                start_idx = content_lower.find(indicator) + len(indicator)
                line = content[start_idx:].split('\n')[0].strip()
                # Split by common separators
                possible_tags = re.split(r'[,\s]+', line)
                for tag in possible_tags:
                    tag = tag.strip('[]')
                    if tag and len(tag) > 1:
                        all_tags.add(tag)
        
        return list(all_tags)
    
    def _extract_references(self, content: str) -> List[str]:
        """Extract references from content"""
        references = []
        
        # Look for URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        references.extend(urls)
        
        # Look for citations in various formats
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\b([A-Z][a-z]+ et al\., \d{4})\b',  # Smith et al., 2023
            r'\b([A-Z][a-z]+ \(\d{4}\))\b',  # Smith (2023)
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            references.extend(matches)
        
        return references[:20]  # Limit to 20 references
    
    def _detect_language(self, content: str) -> str:
        """Detect language of the content"""
        # For simplicity, assume English unless we have evidence otherwise
        # In a real implementation, this would use language detection libraries
        common_english_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have']
        words = word_tokenize(content.lower())
        english_word_count = sum(1 for word in words if word in common_english_words)
        
        # If at least 30% of the content consists of common English words, assume English
        if len(words) > 0 and english_word_count / len(words) > 0.3:
            return 'en'
        else:
            return 'unknown'
    
    def _find_repetitive_patterns(self, content: str) -> List[str]:
        """Find repetitive patterns in content"""
        lines = content.split('\n')
        line_counts = {}
        
        for line in lines:
            stripped = line.strip()
            if stripped:  # Only count non-empty lines
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
        
        # Return lines that appear more than 3 times
        repetitive = [line for line, count in line_counts.items() if count > 3]
        return repetitive
    
    def _find_inconsistencies(self, content: str) -> List[str]:
        """Find potential inconsistencies in content"""
        inconsistencies = []
        
        # Check for inconsistent capitalization of the same term
        words = word_tokenize(content)
        word_forms = {}
        
        for word in words:
            if word.isalpha():  # Only consider alphabetic words
                lower_word = word.lower()
                if lower_word not in word_forms:
                    word_forms[lower_word] = set()
                word_forms[lower_word].add(word)
        
        # Look for words that have multiple capitalization forms
        for lower_word, forms in word_forms.items():
            if len(forms) > 1:
                inconsistencies.append(f"Inconsistent capitalization of '{lower_word}': {', '.join(forms)}")
        
        return inconsistencies
    
    def generate_recommendations(self, issues: List[Dict], content: str) -> List[str]:
        """Generate recommendations based on issues found"""
        recommendations = []
        
        if not issues:
            recommendations.append("No major issues found. Content appears well-structured.")
            return recommendations
        
        # Count issue types
        issue_types = {}
        for issue in issues:
            issue_type = issue.get('type', 'unknown')
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        # Generate specific recommendations
        if 'line_too_long' in issue_types:
            recommendations.append("Consider breaking up long lines for better readability.")
        
        if 'absolute_statement' in issue_types:
            recommendations.append("Review absolute statements (always, never, all) for accuracy and nuance.")
        
        if 'missing_legal_term' in issue_types:
            recommendations.append("Consider adding important legal terms like liability and warranty clauses.")
        
        if 'potential_phi_exposure' in issue_types:
            recommendations.append("Review content for potential exposure of protected health information.")
        
        if 'security_exposure' in issue_types:
            recommendations.append("Remove or secure any exposed credentials or sensitive information.")
        
        if any('capitalization' in str(issue.get('type', '')) for issue in issues):
            recommendations.append("Standardize capitalization of terms throughout the document.")
        
        if any('format' in str(issue.get('type', '')) for issue in issues):
            recommendations.append("Ensure consistent formatting throughout the content.")
        
        # Add general recommendations based on content type
        content_type = self._classify_content_type(content)
        if content_type == ContentType.CODE:
            recommendations.append("Consider adding documentation comments and type hints for better maintainability.")
        elif content_type == ContentType.DOCUMENT:
            recommendations.append("Consider adding a table of contents and cross-references for better navigation.")
        elif content_type == ContentType.TECHNICAL:
            recommendations.append("Include more specific technical details and implementation examples.")
        
        return recommendations

# Example usage and testing
def test_content_analyzer():
    """Test function for the Content Analyzer"""
    analyzer = ContentAnalyzer()
    
    # Test with sample content
    sample_content = """
# Sample Technical Protocol

## Overview
This document describes a technical process for data validation. 
The system should always validate inputs before processing them.

## Requirements
- All user inputs must be validated
- Error handling must be implemented
- Security tokens should be stored securely

## Implementation
The function validate_input() performs validation checks.
"""
    
    result = analyzer.analyze_content(sample_content)
    
    print("Content Analysis Results:")
    print(f"Overall Score: {result.overall_score:.2f}")
    print(f"Content Type: {result.semantic_understanding['content_type'].value}")
    print(f"Word Count: {result.input_parsing['word_count']}")
    print(f"Issues Found: {len(result.issues_found)}")
    print("Recommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")
    
    return result

if __name__ == "__main__":
    test_content_analyzer()