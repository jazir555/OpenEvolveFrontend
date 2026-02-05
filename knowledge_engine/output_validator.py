"""
Output Validator Module for OpenEvolve Knowledge Engine

Provides post-generation validation, self-correction, and quality assurance.
"""

import re
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class ValidationError(Enum):
    """Types of validation errors"""
    MISSING_FACTS = "missing_facts"
    INCOMPLETE_SECTIONS = "incomplete_sections"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    POOR_STRUCTURE = "poor_structure"
    LOW_CONFIDENCE = "low_confidence"
    FACTUAL_ERROR = "factual_error"


@dataclass
class QualityCheck:
    """Result of quality check"""
    passed: bool
    score: float
    errors: List[ValidationError]
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'score': self.score,
            'errors': [e.value for e in self.errors],
            'details': self.details
        }


@dataclass
class CorrectionSuggestion:
    """Suggestion for correcting output"""
    issue: str
    suggestion: str
    priority: str  # 'high', 'medium', 'low'
    auto_fixable: bool


class OutputValidator:
    """
    Validates generated output against requirements.
    Identifies issues and suggests corrections.
    """
    
    def __init__(self):
        self.min_quality_threshold = 70.0
        self.max_retries = 2
        
    def validate(self, output: str, requirements: Dict[str, Any]) -> QualityCheck:
        """
        Validate output against requirements.
        
        Args:
            output: Generated text
            requirements: Dict with validation criteria
                - required_facts: List of facts that must be present
                - required_sections: List of sections/topics to cover
                - min_length: Minimum word count
                - max_length: Maximum word count
                - requires_structure: Whether structure is required
                
        Returns:
            QualityCheck with validation results
        """
        errors = []
        details = {}
        scores = []
        
        # Check 1: Required facts
        if 'required_facts' in requirements:
            fact_score, missing = self._check_facts(output, requirements['required_facts'])
            scores.append(fact_score * 0.35)  # 35% weight
            details['fact_score'] = fact_score
            details['missing_facts'] = missing
            if fact_score < 70:
                errors.append(ValidationError.MISSING_FACTS)
        
        # Check 2: Required sections/aspects
        if 'required_sections' in requirements:
            section_score, missing = self._check_sections(output, requirements['required_sections'])
            scores.append(section_score * 0.30)  # 30% weight
            details['section_score'] = section_score
            details['missing_sections'] = missing
            if section_score < 70:
                errors.append(ValidationError.INCOMPLETE_SECTIONS)
        
        # Check 3: Length requirements
        word_count = len(output.split())
        details['word_count'] = word_count
        
        if 'min_length' in requirements:
            min_words = requirements['min_length']
            if word_count < min_words:
                length_score = (word_count / min_words) * 100
                scores.append(length_score * 0.15)
                errors.append(ValidationError.TOO_SHORT)
            else:
                scores.append(15)  # Full marks
        
        if 'max_length' in requirements:
            max_words = requirements['max_length']
            if word_count > max_words:
                length_score = max(0, 100 - ((word_count - max_words) / max_words * 100))
                scores.append(length_score * 0.10)
                errors.append(ValidationError.TOO_LONG)
            else:
                scores.append(10)
        
        # Check 4: Structure
        if requirements.get('requires_structure', True):
            structure_score = self._check_structure(output)
            scores.append(structure_score * 0.10)  # 10% weight
            details['structure_score'] = structure_score
            if structure_score < 50:
                errors.append(ValidationError.POOR_STRUCTURE)
        
        # Calculate overall score
        total_score = sum(scores)
        
        return QualityCheck(
            passed=total_score >= self.min_quality_threshold and len(errors) <= 1,
            score=total_score,
            errors=errors,
            details=details
        )
    
    def _check_facts(self, output: str, required_facts: List[str]) -> Tuple[float, List[str]]:
        """Check if required facts are present"""
        output_lower = output.lower()
        found = []
        missing = []
        
        for fact in required_facts:
            if fact.lower() in output_lower:
                found.append(fact)
            else:
                missing.append(fact)
        
        score = (len(found) / len(required_facts)) * 100 if required_facts else 100
        return score, missing
    
    def _check_sections(self, output: str, required_sections: List[str]) -> Tuple[float, List[str]]:
        """Check if required sections are covered"""
        output_lower = output.lower()
        found = []
        missing = []
        
        for section in required_sections:
            # Check for section keyword or synonym
            keywords = section.lower().split()
            if any(kw in output_lower for kw in keywords):
                found.append(section)
            else:
                missing.append(section)
        
        score = (len(found) / len(required_sections)) * 100 if required_sections else 100
        return score, missing
    
    def _check_structure(self, output: str) -> float:
        """Check if output has good structure"""
        score = 100.0
        
        # Check for headers/sections
        has_headers = bool(re.search(r'#{2,}|\*\*|__|^[A-Z][a-z]+:', output, re.MULTILINE))
        if not has_headers and len(output) > 300:
            score -= 30
        
        # Check for lists
        has_lists = bool(re.search(r'^\s*[-**\d]\.', output, re.MULTILINE))
        if not has_lists and len(output) > 400:
            score -= 20
        
        # Check paragraph length
        paragraphs = output.split('\n\n')
        long_paragraphs = sum(1 for p in paragraphs if len(p.split()) > 100)
        if long_paragraphs > 2:
            score -= 15
        
        return max(0, score)
    
    def generate_suggestions(self, output: str, check: QualityCheck, 
                           requirements: Dict) -> List[CorrectionSuggestion]:
        """Generate suggestions for improving output"""
        suggestions = []
        
        for error in check.errors:
            if error == ValidationError.MISSING_FACTS:
                missing = check.details.get('missing_facts', [])
                suggestions.append(CorrectionSuggestion(
                    issue=f"Missing required facts: {', '.join(missing[:3])}",
                    suggestion=f"Add information about: {', '.join(missing)}",
                    priority='high',
                    auto_fixable=False
                ))
            
            elif error == ValidationError.INCOMPLETE_SECTIONS:
                missing = check.details.get('missing_sections', [])
                suggestions.append(CorrectionSuggestion(
                    issue=f"Missing sections: {', '.join(missing[:3])}",
                    suggestion=f"Add a section covering: {', '.join(missing)}",
                    priority='high',
                    auto_fixable=False
                ))
            
            elif error == ValidationError.TOO_SHORT:
                current = check.details.get('word_count', 0)
                target = requirements.get('min_length', 100)
                suggestions.append(CorrectionSuggestion(
                    issue=f"Output too short ({current} words, need {target})",
                    suggestion="Expand with more detail, examples, or explanations",
                    priority='medium',
                    auto_fixable=False
                ))
            
            elif error == ValidationError.TOO_LONG:
                current = check.details.get('word_count', 0)
                target = requirements.get('max_length', 500)
                suggestions.append(CorrectionSuggestion(
                    issue=f"Output too long ({current} words, max {target})",
                    suggestion="Be more concise, focus on key points only",
                    priority='medium',
                    auto_fixable=False
                ))
            
            elif error == ValidationError.POOR_STRUCTURE:
                suggestions.append(CorrectionSuggestion(
                    issue="Poor structure (no clear sections)",
                    suggestion="Add headers and use bullet points for readability",
                    priority='medium',
                    auto_fixable=False
                ))
        
        return suggestions


class SelfCorrectionLoop:
    """
    Implements self-correction by validating output and retrying if needed.
    """
    
    def __init__(self, api_caller=None):
        self.validator = OutputValidator()
        self.api_caller = api_caller
        self.max_retries = 2
        
    def generate_with_correction(self, prompt: str, requirements: Dict[str, Any],
                                 api_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate output with self-correction loop.
        
        Args:
            prompt: The input prompt
            requirements: Validation requirements
            api_params: Parameters for API call (temp, max_tokens, etc.)
            
        Returns:
            Dict with final output and metadata
        """
        attempts = []
        
        for attempt in range(self.max_retries + 1):
            # Generate output
            result = self._call_api(prompt, api_params)
            
            if not result['success']:
                return {
                    'success': False,
                    'error': result.get('error', 'API call failed'),
                    'attempts': attempts
                }
            
            output = result['content']
            
            # Validate output
            validation = self.validator.validate(output, requirements)
            
            attempt_info = {
                'attempt': attempt + 1,
                'output': output,
                'validation': validation.to_dict(),
                'quality_score': validation.score
            }
            attempts.append(attempt_info)
            
            # Check if validation passed
            if validation.passed:
                return {
                    'success': True,
                    'output': output,
                    'quality_score': validation.score,
                    'attempts': attempts,
                    'corrections_applied': attempt > 0
                }
            
            # If not passed and we have retries left, generate correction prompt
            if attempt < self.max_retries:
                suggestions = self.validator.generate_suggestions(
                    output, validation, requirements
                )
                
                prompt = self._create_correction_prompt(
                    prompt, output, validation, suggestions
                )
                
                # Slightly increase temperature for correction to encourage variation
                api_params = api_params.copy()
                api_params['temperature'] = min(0.9, api_params.get('temperature', 0.5) + 0.1)
        
        # All retries exhausted
        # Return best attempt
        best_attempt = max(attempts, key=lambda x: x['quality_score'])
        
        return {
            'success': True,  # Partial success
            'output': best_attempt['output'],
            'quality_score': best_attempt['quality_score'],
            'attempts': attempts,
            'corrections_applied': True,
            'warning': 'Maximum retries reached, returning best attempt',
            'issues': [e.value for e in validation.errors]
        }
    
    def _call_api(self, prompt: str, params: Dict) -> Dict:
        """Call API with given parameters"""
        if self.api_caller:
            return self.api_caller(prompt, **params)
        
        # Default implementation if no api_caller provided
        # In practice, this should be injected from the main system
        return {'success': False, 'error': 'No API caller provided'}
    
    def _create_correction_prompt(self, original_prompt: str, previous_output: str,
                                 validation: QualityCheck, 
                                 suggestions: List[CorrectionSuggestion]) -> str:
        """Create prompt for correction attempt"""
        
        correction_prompt = f"""{original_prompt}

---

CORRECTION NEEDED:

Previous attempt had these issues:
"""
        
        for suggestion in suggestions[:3]:  # Top 3 suggestions
            correction_prompt += f"\n- {suggestion.issue}"
            correction_prompt += f"\n  Fix: {suggestion.suggestion}"
        
        correction_prompt += f"""

Quality score: {validation.score:.1f}/100

Please provide an improved response addressing these issues."""
        
        return correction_prompt


class ConflictResolver:
    """
    Detects and resolves contradictory requirements in prompts.
    """
    
    CONFLICT_PATTERNS = [
        {
            'patterns': [r'\bdetailed\b', r'\bcomprehensive\b', r'\bthorough\b'],
            'opposite': [r'\bbrief\b', r'\bshort\b', r'\bconcise\b', r'\b\d+ words?\b'],
            'resolution': 'ask_priority',
            'message': 'Conflicting length requirements detected'
        },
        {
            'patterns': [r'\bsimple\b', r'\bbasic\b'],
            'opposite': [r'\bcomplex\b', r'\badvanced\b', r'\bsophisticated\b'],
            'resolution': 'ask_priority',
            'message': 'Conflicting complexity requirements detected'
        },
        {
            'patterns': [r'\bquick\b', r'\bfast\b', r'\brapid\b'],
            'opposite': [r'\bcareful\b', r'\bthorough\b', r'\bdetailed\b'],
            'resolution': 'explain_tradeoff',
            'message': 'Speed vs quality trade-off detected'
        }
    ]
    
    def detect_conflicts(self, prompt: str) -> List[Dict]:
        """Detect conflicts in prompt"""
        conflicts = []
        prompt_lower = prompt.lower()
        
        for pattern_def in self.CONFLICT_PATTERNS:
            has_first = any(re.search(p, prompt_lower) for p in pattern_def['patterns'])
            has_opposite = any(re.search(p, prompt_lower) for p in pattern_def['opposite'])
            
            if has_first and has_opposite:
                conflicts.append({
                    'message': pattern_def['message'],
                    'resolution': pattern_def['resolution'],
                    'patterns': pattern_def['patterns'],
                    'opposites': pattern_def['opposite']
                })
        
        return conflicts
    
    def resolve(self, prompt: str, conflicts: List[Dict]) -> Tuple[str, List[str]]:
        """
        Resolve conflicts by modifying prompt or adding clarifications.
        
        Returns:
            Tuple of (resolved_prompt, warnings)
        """
        if not conflicts:
            return prompt, []
        
        warnings = []
        resolution_note = "\n\n[Note: I've detected conflicting requirements. Prioritizing as follows:\n"
        
        for conflict in conflicts:
            if conflict['resolution'] == 'ask_priority':
                # For conflicts we can't auto-resolve, add a note
                warnings.append(conflict['message'])
                resolution_note += f"- {conflict['message']}: Prioritizing completeness\n"
            
            elif conflict['resolution'] == 'explain_tradeoff':
                warnings.append(conflict['message'])
                resolution_note += f"- {conflict['message']}: Will balance speed and quality\n"
        
        resolution_note += "]"
        
        # Add resolution note to prompt
        resolved_prompt = prompt + resolution_note
        
        return resolved_prompt, warnings


# Convenience functions
def validate_output(output: str, requirements: Dict) -> QualityCheck:
    """Quick output validation"""
    validator = OutputValidator()
    return validator.validate(output, requirements)


def detect_conflicts(prompt: str) -> List[Dict]:
    """Quick conflict detection"""
    resolver = ConflictResolver()
    return resolver.detect_conflicts(prompt)


if __name__ == "__main__":
    # Test output validator
    print("=" * 70)
    print("OUTPUT VALIDATOR TESTS")
    print("=" * 70)
    
    validator = OutputValidator()
    
    # Test 1: Good output
    good_output = """
## Analysis

The fintech startup faces several risks:

### Market Risk
High competition in the payments space.

### Financial Risk
Negative cash flow requires additional funding within 6 months.

### Regulatory Risk
Compliance with financial regulations is costly.
"""
    
    requirements = {
        'required_facts': ['risk', 'fintech', 'cash flow'],
        'required_sections': ['market', 'financial', 'regulatory'],
        'min_length': 50
    }
    
    result = validator.validate(good_output, requirements)
    print(f"\nGood output test:")
    print(f"  Passed: {result.passed}")
    print(f"  Score: {result.score:.1f}")
    print(f"  Errors: {[e.value for e in result.errors]}")
    
    # Test 2: Bad output (missing facts)
    bad_output = "This is a company. It does things."
    
    result = validator.validate(bad_output, requirements)
    print(f"\nBad output test:")
    print(f"  Passed: {result.passed}")
    print(f"  Score: {result.score:.1f}")
    print(f"  Errors: {[e.value for e in result.errors]}")
    
    # Test conflict detection
    print("\n" + "=" * 70)
    print("CONFLICT DETECTION TESTS")
    print("=" * 70)
    
    resolver = ConflictResolver()
    
    conflict_prompt = "Provide a detailed comprehensive analysis in exactly 10 words."
    conflicts = resolver.detect_conflicts(conflict_prompt)
    
    print(f"\nPrompt: '{conflict_prompt}'")
    print(f"Conflicts found: {len(conflicts)}")
    for c in conflicts:
        print(f"  - {c['message']}")
