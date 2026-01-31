"""
Input Processor Module for OpenEvolve Knowledge Engine

Provides input validation, sanitization, and preprocessing
to improve robustness and prevent hallucination.
"""

import re
import os
import sys
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class InputValidationError(Exception):
    """Raised when input fails validation"""
    pass


class CapabilityError(Exception):
    """Raised when request exceeds system capabilities"""
    pass


@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    coherence_score: float
    confidence: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    category: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'coherence_score': self.coherence_score,
            'confidence': self.confidence,
            'issues': self.issues,
            'suggestions': self.suggestions,
            'category': self.category
        }


@dataclass
class CapabilityCheck:
    """Result of capability boundary check"""
    is_feasible: bool
    capability_type: str
    confidence: float
    reasoning: str
    alternative_suggestions: List[str] = field(default_factory=list)


class InputValidator:
    """
    Validates input for semantic coherence, detects nonsensical inputs,
    and identifies potential issues before processing.
    """
    
    def __init__(self):
        # Common nonsensical patterns
        self.nonsense_patterns = [
            r'\b(colorless green|sleep furiously|invisible pink)\b',
            r'\b(square circle|round triangle|loud silence)\b',
            r'\b(frozen fire|burning ice|dry water)\b',
            r'^\s*$',  # Empty or whitespace only
            r'^[^a-zA-Z]*$',  # No letters
        ]
        
        # Contradictory requirement indicators
        self.contradiction_patterns = [
            (r'\b(detailed|comprehensive|thorough)\b', r'\b(brief|short|concise|\d+ words?)\b'),
            (r'\b(simple|basic)\b', r'\b(complex|advanced|sophisticated)\b'),
            (r'\b(quick|fast|rapid)\b', r'\b(careful|thorough|detailed)\b'),
        ]
        
        # Minimum coherence indicators
        self.coherence_indicators = [
            r'\b(the|a|an|is|are|was|were|have|has|had|do|does|did)\b',
            r'\b(what|why|how|when|where|who|which)\b',
            r'\b(analyze|explain|describe|compare|evaluate|recommend)\b',
        ]
        
        self.min_length = 10
        self.max_length = 10000
        
    def validate(self, text: str) -> ValidationResult:
        """
        Comprehensive input validation.
        
        Args:
            text: Input text to validate
            
        Returns:
            ValidationResult with validation status and issues
        """
        issues = []
        suggestions = []
        
        # Check length
        if len(text) < self.min_length:
            issues.append(f"Input too short ({len(text)} chars, minimum {self.min_length})")
            suggestions.append("Provide more context or detail")
            
        if len(text) > self.max_length:
            issues.append(f"Input too long ({len(text)} chars, maximum {self.max_length})")
            suggestions.append("Consider breaking into smaller requests")
        
        # Check for nonsensical content
        coherence_score = self._check_coherence(text)
        if coherence_score < 0.3:
            issues.append("Input appears nonsensical or lacks semantic meaning")
            suggestions.append("Rephrase with clear, meaningful language")
            
        # Check for contradictions
        contradictions = self._detect_contradictions(text)
        if contradictions:
            issues.append(f"Potential contradictory requirements detected: {contradictions}")
            suggestions.append("Clarify or prioritize your requirements")
            
        # Check for ambiguity
        ambiguity_score = self._check_ambiguity(text)
        if ambiguity_score > 0.7:
            issues.append("Input is highly ambiguous")
            suggestions.append("Be more specific about what you need")
            
        # Categorize input
        category = self._categorize_input(text)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(
            coherence_score, 
            ambiguity_score, 
            len(issues)
        )
        
        is_valid = (
            coherence_score >= 0.3 and 
            len(text) >= self.min_length and
            len(issues) < 3
        )
        
        return ValidationResult(
            is_valid=is_valid,
            coherence_score=coherence_score,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
            category=category
        )
    
    def _check_coherence(self, text: str) -> float:
        """
        Check semantic coherence of input.
        Returns score between 0 and 1.
        """
        text_lower = text.lower()
        
        # Check for nonsense patterns
        for pattern in self.nonsense_patterns:
            if re.search(pattern, text_lower):
                return 0.0
        
        # Count coherence indicators
        indicator_count = sum(
            1 for pattern in self.coherence_indicators
            if re.search(pattern, text_lower)
        )
        
        # Check for reasonable word structure
        words = text.split()
        if len(words) < 3:
            return 0.1
            
        # Average word length check (too short = gibberish)
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2:
            return 0.2
            
        # Calculate coherence score
        base_score = min(1.0, indicator_count / 3)
        length_bonus = min(0.3, len(words) / 100)
        
        return min(1.0, base_score + length_bonus)
    
    def _detect_contradictions(self, text: str) -> List[str]:
        """Detect potentially contradictory requirements"""
        contradictions = []
        text_lower = text.lower()
        
        for pattern1, pattern2 in self.contradiction_patterns:
            if re.search(pattern1, text_lower) and re.search(pattern2, text_lower):
                # Extract the contradictory terms
                match1 = re.search(pattern1, text_lower)
                match2 = re.search(pattern2, text_lower)
                if match1 and match2:
                    contradictions.append(f"'{match1.group()}' vs '{match2.group()}'")
                    
        return contradictions
    
    def _check_ambiguity(self, text: str) -> float:
        """
        Check for ambiguity in input.
        Returns score between 0 (clear) and 1 (highly ambiguous).
        """
        # Vague pronouns indicate ambiguity
        vague_indicators = ['it', 'this', 'that', 'they', 'them']
        words = text.lower().split()
        
        if not words:
            return 1.0
            
        vague_count = sum(1 for w in words if w.strip('.,!?;:"') in vague_indicators)
        vague_ratio = vague_count / len(words)
        
        # Check for missing context indicators
        context_indicators = ['because', 'since', 'as', 'for', 'about', 'regarding']
        has_context = any(ci in text.lower() for ci in context_indicators)
        
        # Short inputs are more ambiguous
        length_factor = max(0, 1 - len(words) / 50)
        
        ambiguity = vague_ratio * 0.5 + (0.3 if not has_context else 0) + length_factor * 0.2
        return min(1.0, ambiguity)
    
    def _categorize_input(self, text: str) -> str:
        """Categorize input type for routing"""
        text_lower = text.lower()
        
        # Check for question patterns
        if any(text_lower.startswith(w) for w in ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'do', 'is', 'are']):
            return 'question'
            
        # Check for analysis requests
        if any(w in text_lower for w in ['analyze', 'evaluate', 'assess', 'review', 'examine']):
            return 'analysis'
            
        # Check for creative requests
        if any(w in text_lower for w in ['write', 'create', 'generate', 'story', 'poem', 'essay']):
            return 'creative'
            
        # Check for comparison
        if any(w in text_lower for w in ['compare', 'contrast', 'difference', 'versus', 'vs']):
            return 'comparison'
            
        # Check for summarization
        if any(w in text_lower for w in ['summarize', 'summary', 'tldr', 'brief']):
            return 'summarization'
            
        return 'general'
    
    def _calculate_confidence(self, coherence: float, ambiguity: float, issue_count: int) -> float:
        """Calculate overall confidence score"""
        base = coherence * 0.6
        ambiguity_penalty = ambiguity * 0.3
        issue_penalty = min(0.3, issue_count * 0.1)
        
        return max(0.0, min(1.0, base - ambiguity_penalty - issue_penalty))


class CapabilityRegistry:
    """
    Tracks system capabilities and limitations.
    Prevents attempting impossible tasks.
    """
    
    CAPABILITIES = {
        'text_analysis': {
            'description': 'Analyze and interpret text',
            'examples': ['sentiment analysis', 'topic extraction', 'summarization'],
            'limitations': ['Cannot access real-time data', 'Cannot browse the internet']
        },
        'code_generation': {
            'description': 'Generate and review code',
            'examples': ['Python functions', 'Code review', 'Debugging assistance'],
            'limitations': ['Cannot execute code', 'Limited to common languages']
        },
        'creative_writing': {
            'description': 'Generate creative content',
            'examples': ['Stories', 'Descriptions', 'Marketing copy'],
            'limitations': ['Cannot guarantee originality', 'May need human editing']
        },
        'factual_qa': {
            'description': 'Answer factual questions',
            'examples': ['Definitions', 'Explanations', 'How-to guides'],
            'limitations': ['Knowledge cutoff date', 'May hallucinate facts', 'No real-time info']
        },
        'reasoning': {
            'description': 'Logical and analytical reasoning',
            'examples': ['Problem solving', 'Decision analysis', 'Risk assessment'],
            'limitations': ['Cannot predict future', 'Limited to provided context']
        }
    }
    
    IMPOSSIBLE_PATTERNS = [
        r'\bpredict\b.*\b(exact|precise|specific)\b.*\b(price|outcome|result|number)\b.*\b(in \d+ (years?|months?)|future)\b',
        r'\bwhat\b.*\b(am i|are you)\b.*\bthinking\b',
        r'\baccess\b.*\b(real-time|live|current|today)\b.*\b(data|information|price|weather)\b',
        r'\bhack\b.*\b(into|password|database|account)\b',
        r'\bcurrent\b.*\b(time in|weather in|price of)\b',
        r'\bexactly\b.*\b\d+\b.*\b(years?|months?|days?)\b.*\b(from now|in the future)\b',
    ]
    
    def __init__(self):
        self.validator = InputValidator()
        
    def check_feasibility(self, request: str) -> CapabilityCheck:
        """
        Check if a request is within system capabilities.
        
        Args:
            request: The user request
            
        Returns:
            CapabilityCheck with feasibility assessment
        """
        request_lower = request.lower()
        
        # Check for impossible patterns
        for pattern in self.IMPOSSIBLE_PATTERNS:
            if re.search(pattern, request_lower):
                return CapabilityCheck(
                    is_feasible=False,
                    capability_type='impossible',
                    confidence=0.9,
                    reasoning="Request asks for prediction of specific future values or access to private/real-time data",
                    alternative_suggestions=[
                        "Ask for analysis of trends instead of exact predictions",
                        "Request general guidance rather than specific future values",
                        "Ask how to find real-time information"
                    ]
                )
        
        # Check for requests beyond knowledge cutoff
        if any(phrase in request_lower for phrase in ['latest', 'most recent', 'just happened', 'today', 'yesterday']):
            # Validate if it's asking for time-sensitive info
            if re.search(r'\b(news|event|price|stock|weather|score)\b', request_lower):
                return CapabilityCheck(
                    is_feasible=False,
                    capability_type='real_time_data',
                    confidence=0.85,
                    reasoning="Request requires real-time or recent information beyond knowledge cutoff",
                    alternative_suggestions=[
                        "Check current news sources or official websites",
                        "Ask for general analysis instead of current data",
                        "Request historical context instead"
                    ]
                )
        
        # Determine best matching capability
        capability_scores = {}
        
        for cap_name, cap_info in self.CAPABILITIES.items():
            score = 0
            # Check examples
            for example in cap_info['examples']:
                example_words = example.lower().split()
                matches = sum(1 for word in example_words if word in request_lower)
                score += matches / len(example_words)
                
            # Check description keywords
            desc_words = cap_info['description'].lower().split()
            matches = sum(1 for word in desc_words if word in request_lower)
            score += matches * 0.5
            
            capability_scores[cap_name] = score
        
        # Get best match
        best_capability = max(capability_scores, key=capability_scores.get)
        best_score = capability_scores[best_capability]
        
        if best_score < 0.2:
            return CapabilityCheck(
                is_feasible=True,  # Assume feasible but uncertain
                capability_type='unknown',
                confidence=0.4,
                reasoning="Request type unclear, will attempt general response",
                alternative_suggestions=[]
            )
        
        return CapabilityCheck(
            is_feasible=True,
            capability_type=best_capability,
            confidence=min(0.95, 0.5 + best_score * 0.5),
            reasoning=f"Request matches {best_capability} capability",
            alternative_suggestions=[]
        )
    
    def get_limitations(self, capability_type: str) -> List[str]:
        """Get limitations for a specific capability"""
        if capability_type in self.CAPABILITIES:
            return self.CAPABILITIES[capability_type]['limitations']
        return ["General AI limitations apply"]


class EnhancedInputProcessor:
    """
    Main input processing class that combines validation and capability checking.
    """
    
    def __init__(self):
        self.validator = InputValidator()
        self.capability_registry = CapabilityRegistry()
        
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process input through full validation pipeline.
        
        Args:
            text: Raw user input
            
        Returns:
            Dict with processed input, validation results, and metadata
        """
        # Step 1: Basic validation
        validation = self.validator.validate(text)
        
        # Step 2: Capability check
        capability_check = self.capability_registry.check_feasibility(text)
        
        # Step 3: Determine if we should proceed
        should_proceed = validation.is_valid and capability_check.is_feasible
        
        # Step 4: Generate processed result
        result = {
            'original_input': text,
            'processed_input': self._enhance_input(text, validation, capability_check),
            'validation': validation.to_dict(),
            'capability_check': {
                'is_feasible': capability_check.is_feasible,
                'type': capability_check.capability_type,
                'confidence': capability_check.confidence,
                'reasoning': capability_check.reasoning,
                'alternatives': capability_check.alternative_suggestions
            },
            'should_proceed': should_proceed,
            'routing_info': {
                'category': validation.category,
                'capability_type': capability_check.capability_type,
                'confidence': min(validation.confidence, capability_check.confidence)
            }
        }
        
        return result
    
    def _enhance_input(self, text: str, validation: ValidationResult, 
                      capability: CapabilityCheck) -> str:
        """
        Enhance input with clarifications if needed.
        """
        enhanced = text
        
        # Add capability-specific instructions
        if capability.capability_type == 'factual_qa':
            enhanced += "\n\n(Note: If this requires information beyond my knowledge cutoff, I will indicate that.)"
            
        if capability.capability_type == 'code_generation':
            enhanced += "\n\n(Note: I will provide code examples but cannot execute them. Please test thoroughly.)"
        
        # Add clarification for ambiguous inputs
        if validation.confidence < 0.5:
            enhanced += "\n\n(I'll do my best with this request, but please let me know if you'd like me to focus on a specific aspect.)"
            
        return enhanced
    
    def get_error_message(self, validation: ValidationResult, 
                         capability: CapabilityCheck) -> str:
        """Generate appropriate error message for failed validation"""
        messages = []
        
        if not validation.is_valid:
            messages.append("I notice some issues with your request:")
            for issue in validation.issues[:3]:
                messages.append(f"  - {issue}")
        
        if not capability.is_feasible:
            messages.append(f"\nI'm not able to fulfill this request because: {capability.reasoning}")
            if capability.alternative_suggestions:
                messages.append("\nInstead, you could:")
                for suggestion in capability.alternative_suggestions[:3]:
                    messages.append(f"  • {suggestion}")
        
        return "\n".join(messages)


# Convenience function for quick validation
def validate_input(text: str) -> Dict[str, Any]:
    """Quick validation function"""
    processor = EnhancedInputProcessor()
    return processor.process(text)


if __name__ == "__main__":
    # Test the input processor
    test_inputs = [
        "Analyze the risks of investing in Tesla stock for 2025",  # Valid
        "Colorless green ideas sleep furiously",  # Nonsensical
        "Tell me about it",  # Ambiguous
        "Provide detailed analysis in exactly 5 words",  # Contradictory
        "What is the exact price of Bitcoin 5 years from now?",  # Impossible
        "What am I thinking right now?",  # Impossible
    ]
    
    processor = EnhancedInputProcessor()
    
    print("=" * 70)
    print("INPUT PROCESSOR TESTS")
    print("=" * 70)
    
    for test_input in test_inputs:
        print(f"\nInput: '{test_input}'")
        print("-" * 70)
        
        result = processor.process(test_input)
        
        print(f"Valid: {result['validation']['is_valid']}")
        print(f"Feasible: {result['capability_check']['is_feasible']}")
        print(f"Category: {result['routing_info']['category']}")
        print(f"Confidence: {result['routing_info']['confidence']:.2f}")
        
        if result['validation']['issues']:
            print(f"Issues: {result['validation']['issues']}")
            
        if not result['capability_check']['is_feasible']:
            print(f"Reason: {result['capability_check']['reasoning']}")
