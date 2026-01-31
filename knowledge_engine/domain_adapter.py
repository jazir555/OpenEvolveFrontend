"""
Domain Adapter Module for OpenEvolve Knowledge Engine

Provides automatic domain detection and parameter optimization
for different types of tasks.
"""

import re
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class TaskDomain(Enum):
    """Task domain categories"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    EDUCATIONAL = "educational"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"


class AudienceLevel(Enum):
    """Target audience expertise levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"
    MIXED = "mixed"


@dataclass
class DomainConfig:
    """Configuration for a specific domain"""
    domain: TaskDomain
    temperature: float
    max_tokens: int
    system_prompt: str
    style_instructions: str
    format_template: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'domain': self.domain.value,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'system_prompt': self.system_prompt,
            'style_instructions': self.style_instructions,
            'format_template': self.format_template
        }


@dataclass
class AdaptationResult:
    """Result of domain adaptation"""
    original_prompt: str
    enhanced_prompt: str
    domain: TaskDomain
    audience: AudienceLevel
    config: DomainConfig
    confidence: float


class DomainClassifier:
    """
    Automatically classifies input into task domains.
    """
    
    DOMAIN_KEYWORDS = {
        TaskDomain.ANALYTICAL: [
            'analyze', 'evaluate', 'assess', 'compare', 'contrast',
            'review', 'examine', 'investigate', 'study', 'research',
            'risk', 'benefit', 'advantage', 'disadvantage', 'pros cons',
            'performance', 'metrics', 'kpi', 'roi', 'efficiency',
            'market', 'competitive', 'strategy', 'forecast', 'trend'
        ],
        TaskDomain.CREATIVE: [
            'write', 'create', 'generate', 'compose', 'draft',
            'story', 'narrative', 'fiction', 'poem', 'song',
            'creative', 'imaginative', 'invent', 'design', 'concept',
            'brainstorm', 'ideate', 'novel', 'character', 'plot',
            'marketing', 'copy', 'slogan', 'tagline', 'campaign'
        ],
        TaskDomain.TECHNICAL: [
            'code', 'program', 'function', 'algorithm', 'debug',
            'implement', 'develop', 'software', 'application',
            'architecture', 'system', 'database', 'api', 'interface',
            'error', 'bug', 'fix', 'optimize', 'refactor',
            'python', 'javascript', 'java', 'c++', 'rust', 'go',
            'configure', 'deploy', 'integrate', 'automate'
        ],
        TaskDomain.EDUCATIONAL: [
            'explain', 'teach', 'tutorial', 'guide', 'how to',
            'learn', 'understand', 'concept', 'definition',
            'beginner', 'introduction', 'basics', 'fundamentals',
            'step by step', 'walkthrough', 'example', 'demonstrate',
            'simplify', 'clarify', 'what is', 'why does', 'how does'
        ],
        TaskDomain.CONVERSATIONAL: [
            'chat', 'talk', 'discuss', 'opinion', 'thought',
            'what do you think', 'help me', 'advice', 'suggestion',
            'recommendation', 'ideas', 'feedback', 'thoughts on'
        ]
    }
    
    AUDIENCE_INDICATORS = {
        AudienceLevel.BEGINNER: [
            'beginner', 'newbie', 'novice', 'starting out', 'just started',
            'simple terms', 'layman', 'non-technical', 'easy to understand',
            'explain like i\'m five', 'el5', 'basic', 'fundamentals',
            'don\'t know much', 'new to', 'learning', 'student'
        ],
        AudienceLevel.EXPERT: [
            'expert', 'advanced', 'technical', 'detailed', 'in-depth',
            'comprehensive', 'thorough', 'sophisticated', 'complex',
            'professional', 'experienced', 'specialist', 'researcher',
            'deep dive', 'nuanced', 'granular', 'implementation details'
        ],
        AudienceLevel.INTERMEDIATE: [
            'intermediate', 'some experience', 'familiar with',
            'practical', 'hands-on', 'working knowledge'
        ]
    }
    
    def classify(self, text: str) -> Tuple[TaskDomain, float]:
        """
        Classify text into task domain.
        
        Returns:
            Tuple of (domain, confidence_score)
        """
        text_lower = text.lower()
        scores = {}
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight multi-word matches higher
                    score += len(keyword.split())
            scores[domain] = score
        
        # Get best match
        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]
        total_score = sum(scores.values())
        
        # Calculate confidence
        if total_score == 0:
            return TaskDomain.UNKNOWN, 0.0
            
        confidence = best_score / total_score if total_score > 0 else 0
        
        # Boost confidence if strong indicators present
        if best_score >= 3:
            confidence = min(0.95, confidence * 1.2)
        
        return best_domain, confidence
    
    def detect_audience(self, text: str) -> Tuple[AudienceLevel, float]:
        """
        Detect target audience level from text.
        
        Returns:
            Tuple of (audience_level, confidence)
        """
        text_lower = text.lower()
        scores = {}
        
        for level, indicators in self.AUDIENCE_INDICATORS.items():
            score = sum(1 for ind in indicators if ind in text_lower)
            scores[level] = score
        
        best_level = max(scores, key=scores.get)
        best_score = scores[best_level]
        
        if best_score == 0:
            # Default to intermediate if no indicators
            return AudienceLevel.INTERMEDIATE, 0.5
        
        confidence = min(0.9, 0.4 + best_score * 0.15)
        return best_level, confidence


class ModeSelector:
    """
    Selects optimal parameters based on domain and audience.
    """
    
    DOMAIN_CONFIGS = {
        TaskDomain.ANALYTICAL: DomainConfig(
            domain=TaskDomain.ANALYTICAL,
            temperature=0.3,
            max_tokens=800,
            system_prompt="You are an expert analyst. Provide thorough, objective analysis with specific data points and clear reasoning.",
            style_instructions="Be thorough, objective, and data-driven. Use specific examples and avoid vague statements.",
            format_template="""## Executive Summary
[2-3 sentence overview]

## Key Findings
- [Finding 1 with evidence]
- [Finding 2 with evidence]
- [Finding 3 with evidence]

## Detailed Analysis
[In-depth analysis with data]

## Recommendations
1. [Specific, actionable recommendation]
2. [Specific, actionable recommendation]
3. [Specific, actionable recommendation]

## Risk Factors
- [Risk 1: mitigation]
- [Risk 2: mitigation]"""
        ),
        TaskDomain.CREATIVE: DomainConfig(
            domain=TaskDomain.CREATIVE,
            temperature=0.8,
            max_tokens=1200,
            system_prompt="You are a creative writer. Generate original, engaging content with vivid details and strong narrative flow.",
            style_instructions="Be creative and original. Use vivid language, strong imagery, and engaging narrative techniques.",
            format_template=None  # Flexible for creative content
        ),
        TaskDomain.TECHNICAL: DomainConfig(
            domain=TaskDomain.TECHNICAL,
            temperature=0.2,
            max_tokens=1000,
            system_prompt="You are a technical expert. Provide precise, accurate technical information with code examples where appropriate.",
            style_instructions="Be precise, accurate, and practical. Include code examples, specific configurations, and implementation details.",
            format_template="""## Overview
[Brief description]

## Solution
```[language]
[Code example]
```

## Explanation
[How it works]

## Usage
[How to use it]

## Considerations
- [Important consideration]
- [Potential issue]"""
        ),
        TaskDomain.EDUCATIONAL: DomainConfig(
            domain=TaskDomain.EDUCATIONAL,
            temperature=0.4,
            max_tokens=900,
            system_prompt="You are an excellent teacher. Explain concepts clearly using analogies and examples appropriate to the audience level.",
            style_instructions="Be clear, patient, and encouraging. Use analogies, step-by-step explanations, and check for understanding.",
            format_template="""## Concept
[What it is]

## Analogy
[Relatable comparison]

## How It Works
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Example
[Concrete example]

## Key Points to Remember
- [Point 1]
- [Point 2]"""
        ),
        TaskDomain.CONVERSATIONAL: DomainConfig(
            domain=TaskDomain.CONVERSATIONAL,
            temperature=0.6,
            max_tokens=600,
            system_prompt="You are a helpful conversational assistant. Provide friendly, helpful responses that address the user's needs.",
            style_instructions="Be friendly, conversational, and helpful. Ask clarifying questions when needed.",
            format_template=None
        ),
        TaskDomain.UNKNOWN: DomainConfig(
            domain=TaskDomain.UNKNOWN,
            temperature=0.5,
            max_tokens=700,
            system_prompt="You are a helpful assistant. Provide clear, accurate information.",
            style_instructions="Be helpful and clear. If uncertain about the request, ask for clarification.",
            format_template=None
        )
    }
    
    AUDIENCE_ADJUSTMENTS = {
        AudienceLevel.BEGINNER: {
            'temperature_modifier': 0.1,  # More consistent
            'instruction_addition': "\nUse simple language, avoid jargon, and provide plenty of examples.",
            'format_addition': "\n\n## Common Mistakes to Avoid\n- [Mistake 1]\n- [Mistake 2]"
        },
        AudienceLevel.EXPERT: {
            'temperature_modifier': -0.1,  # More focused
            'instruction_addition': "\nBe technical and detailed. Assume deep domain knowledge.",
            'format_addition': "\n\n## Advanced Considerations\n- [Advanced point]\n- [Edge case]"
        },
        AudienceLevel.INTERMEDIATE: {
            'temperature_modifier': 0,
            'instruction_addition': "",
            'format_addition': ""
        },
        AudienceLevel.MIXED: {
            'temperature_modifier': 0,
            'instruction_addition': "\nBalance technical depth with accessibility.",
            'format_addition': "\n\n## Quick Reference\n[Summary table or key points]"
        }
    }
    
    def select_config(self, domain: TaskDomain, audience: AudienceLevel) -> DomainConfig:
        """
        Get configuration for domain and audience.
        """
        base_config = self.DOMAIN_CONFIGS.get(domain, self.DOMAIN_CONFIGS[TaskDomain.UNKNOWN])
        adjustments = self.AUDIENCE_ADJUSTMENTS.get(audience, self.AUDIENCE_ADJUSTMENTS[AudienceLevel.INTERMEDIATE])
        
        # Create modified config
        adjusted_temp = max(0.0, min(1.0, base_config.temperature + adjustments['temperature_modifier']))
        
        adjusted_instructions = base_config.style_instructions + adjustments['instruction_addition']
        
        adjusted_template = base_config.format_template
        if adjusted_template and adjustments['format_addition']:
            adjusted_template += adjustments['format_addition']
        
        return DomainConfig(
            domain=domain,
            temperature=adjusted_temp,
            max_tokens=base_config.max_tokens,
            system_prompt=base_config.system_prompt,
            style_instructions=adjusted_instructions,
            format_template=adjusted_template
        )


class DomainAdapter:
    """
    Main domain adaptation class.
    Combines classification and configuration selection.
    """
    
    def __init__(self):
        self.classifier = DomainClassifier()
        self.selector = ModeSelector()
    
    def adapt(self, prompt: str) -> AdaptationResult:
        """
        Adapt prompt based on domain and audience detection.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            AdaptationResult with enhanced prompt and config
        """
        # Detect domain and audience
        domain, domain_conf = self.classifier.classify(prompt)
        audience, audience_conf = self.classifier.detect_audience(prompt)
        
        # Get configuration
        config = self.selector.select_config(domain, audience)
        
        # Enhance prompt
        enhanced = self._enhance_prompt(prompt, config)
        
        # Calculate overall confidence
        confidence = (domain_conf + audience_conf) / 2
        
        return AdaptationResult(
            original_prompt=prompt,
            enhanced_prompt=enhanced,
            domain=domain,
            audience=audience,
            config=config,
            confidence=confidence
        )
    
    def _enhance_prompt(self, prompt: str, config: DomainConfig) -> str:
        """Enhance prompt with domain-specific instructions"""
        enhanced = prompt
        
        # Add style instructions
        if config.style_instructions:
            enhanced += f"\n\n[Instruction: {config.style_instructions}]"
        
        # Add format template if available
        if config.format_template:
            enhanced += f"\n\n[Use this format:\n{config.format_template}\n]"
        
        return enhanced
    
    def get_api_params(self, adaptation: AdaptationResult) -> Dict[str, Any]:
        """Get API parameters from adaptation result"""
        return {
            'temperature': adaptation.config.temperature,
            'max_tokens': adaptation.config.max_tokens,
            'system_prompt': adaptation.config.system_prompt
        }


# Convenience function
def adapt_prompt(prompt: str) -> AdaptationResult:
    """Quick adaptation function"""
    adapter = DomainAdapter()
    return adapter.adapt(prompt)


if __name__ == "__main__":
    # Test the domain adapter
    test_prompts = [
        "Write a short story about an AI that discovers emotions",  # Creative
        "Analyze the competitive landscape for Tesla in 2024",  # Analytical
        "How do I fix a Python 'NoneType' error?",  # Technical
        "Explain blockchain to a 10-year-old",  # Educational, beginner
        "What's your opinion on AI regulation?",  # Conversational
    ]
    
    adapter = DomainAdapter()
    
    print("=" * 70)
    print("DOMAIN ADAPTER TESTS")
    print("=" * 70)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt[:60]}...'")
        print("-" * 70)
        
        result = adapter.adapt(prompt)
        
        print(f"Domain: {result.domain.value} (confidence: {result.confidence:.2f})")
        print(f"Audience: {result.audience.value}")
        print(f"Temperature: {result.config.temperature}")
        print(f"Max Tokens: {result.config.max_tokens}")
        print(f"Enhanced length: {len(result.enhanced_prompt)} chars")
