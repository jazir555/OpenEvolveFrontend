"""
Enhanced Knowledge Engine for OpenEvolve

Integrates all improvements:
- Input validation and sanitization
- Capability boundary detection
- Domain-specific mode switching
- Audience adaptation
- Self-correction loop
- Conflict resolution
- Creative pipeline
"""

import os
import sys
import time
import json
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import our improvement modules
from .input_processor import EnhancedInputProcessor, CapabilityRegistry
from .domain_adapter import DomainAdapter, DomainClassifier, ModeSelector
from .output_validator import OutputValidator, SelfCorrectionLoop, ConflictResolver
from .creative_pipeline import CreativeEnhancer, CreativeFormat


@dataclass
class EngineConfig:
    """Configuration for Enhanced Knowledge Engine"""
    api_key: str
    model: str = "deepseek-chat"
    max_retries: int = 2
    enable_validation: bool = True
    enable_domain_adaptation: bool = True
    enable_self_correction: bool = True
    enable_conflict_resolution: bool = True
    default_temperature: float = 0.5
    default_max_tokens: int = 800


class EnhancedKnowledgeEngine:
    """
    Production-ready Knowledge Engine with all benchmark improvements.
    """
    
    def __init__(self, config: EngineConfig = None):
        """
        Initialize the enhanced knowledge engine.
        
        Args:
            config: Engine configuration. If None, uses environment variables.
        """
        # Setup config
        if config is None:
            config = EngineConfig(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                model="deepseek-chat",
                max_retries=2,
                enable_validation=True,
                enable_domain_adaptation=True,
                enable_self_correction=True,
                enable_conflict_resolution=True,
                default_temperature=0.5,
                default_max_tokens=800
            )
        
        self.config = config
        
        # Initialize components
        self.input_processor = EnhancedInputProcessor()
        self.domain_adapter = DomainAdapter()
        self.output_validator = OutputValidator()
        self.conflict_resolver = ConflictResolver()
        self.creative_enhancer = CreativeEnhancer()
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'validation_failures': 0,
            'corrections_applied': 0,
            'average_quality': 0.0
        }
    
    def process(self, prompt: str, requirements: Dict[str, Any] = None,
               creative_mode: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Process a request through the enhanced pipeline.
        
        Args:
            prompt: User input prompt
            requirements: Validation requirements (facts, sections, etc.)
            creative_mode: Whether to use creative pipeline
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Dict with output and metadata
        """
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        # Step 1: Input Validation (Phase 1 improvement)
        if self.config.enable_validation:
            input_result = self.input_processor.process(prompt)
            
            if not input_result['should_proceed']:
                self.stats['validation_failures'] += 1
                return {
                    'success': False,
                    'error': 'input_validation_failed',
                    'message': self.input_processor.get_error_message(
                        input_result['validation'],
                        input_result['capability_check']
                    ),
                    'validation_result': input_result['validation'],
                    'capability_check': input_result['capability_check']
                }
            
            # Use enhanced prompt if validation passed
            processed_prompt = input_result['processed_input']
            routing_info = input_result['routing_info']
        else:
            processed_prompt = prompt
            routing_info = {'category': 'unknown', 'confidence': 0.5}
        
        # Step 2: Conflict Detection & Resolution (Phase 3 improvement)
        if self.config.enable_conflict_resolution:
            conflicts = self.conflict_resolver.detect_conflicts(processed_prompt)
            if conflicts:
                processed_prompt, warnings = self.conflict_resolver.resolve(
                    processed_prompt, conflicts
                )
        
        # Step 3: Domain Adaptation (Phase 2 improvement)
        if self.config.enable_domain_adaptation and not creative_mode:
            adaptation = self.domain_adapter.adapt(processed_prompt)
            
            # Use adapted prompt and parameters
            final_prompt = adaptation.enhanced_prompt
            api_params = {
                'temperature': adaptation.config.temperature,
                'max_tokens': adaptation.config.max_tokens,
                'system_prompt': adaptation.config.system_prompt
            }
            domain_info = {
                'domain': adaptation.domain.value,
                'audience': adaptation.audience.value,
                'confidence': adaptation.confidence
            }
        elif creative_mode:
            # Use creative pipeline
            creative_result = self.creative_enhancer.enhance(processed_prompt)
            final_prompt = creative_result['enhanced_prompt']
            api_params = creative_result['parameters']
            domain_info = {
                'format': creative_result['format'],
                'structure': creative_result['structure'],
                'techniques': creative_result['techniques']
            }
        else:
            # Use default parameters
            final_prompt = processed_prompt
            api_params = {
                'temperature': kwargs.get('temperature', self.config.default_temperature),
                'max_tokens': kwargs.get('max_tokens', self.config.default_max_tokens),
                'system_prompt': "You are a helpful assistant."
            }
            domain_info = {}
        
        # Step 4: Generation with Self-Correction (Phase 3 improvement)
        if self.config.enable_self_correction and requirements:
            # Use self-correction loop
            result = self._generate_with_correction(
                final_prompt, requirements, api_params
            )
        else:
            # Single generation
            result = self._call_api(final_prompt, api_params)
            
            if result['success'] and requirements:
                # Validate without correction
                validation = self.output_validator.validate(
                    result['content'], requirements
                )
                result['quality_score'] = validation.score
                result['validation_passed'] = validation.passed
        
        # Update statistics
        if result.get('success'):
            self.stats['successful_requests'] += 1
            if result.get('corrections_applied'):
                self.stats['corrections_applied'] += 1
            
            # Update average quality
            quality = result.get('quality_score', 70)
            self.stats['average_quality'] = (
                (self.stats['average_quality'] * (self.stats['successful_requests'] - 1) + quality)
                / self.stats['successful_requests']
            )
        
        # Build response
        processing_time = time.time() - start_time
        
        response = {
            'success': result.get('success', False),
            'output': result.get('output') if result.get('success') else result.get('error'),
            'processing_time': processing_time,
            'metadata': {
                'original_prompt': prompt,
                'final_prompt': final_prompt,
                'input_category': routing_info.get('category'),
                'domain_info': domain_info,
                'api_params': {k: v for k, v in api_params.items() if k != 'system_prompt'},
                'quality_score': result.get('quality_score'),
                'validation_passed': result.get('validation_passed'),
                'corrections_applied': result.get('corrections_applied', False),
                'attempts': result.get('attempts', [{}])
            },
            'engine_stats': self.get_stats()
        }
        
        return response
    
    def _call_api(self, prompt: str, params: Dict) -> Dict[str, Any]:
        """Make API call with given parameters"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if 'system_prompt' in params:
            messages.append({"role": "system", "content": params['system_prompt']})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": params.get('temperature', 0.5),
            "max_tokens": params.get('max_tokens', 800)
        }
        
        try:
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'content': data['choices'][0]['message']['content'],
                    'tokens': data['usage']['total_tokens']
                }
            else:
                return {
                    'success': False,
                    'error': f"API error: {response.status_code}"
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_with_correction(self, prompt: str, requirements: Dict,
                                 api_params: Dict) -> Dict[str, Any]:
        """Generate with self-correction loop"""
        attempts = []
        
        for attempt in range(self.config.max_retries + 1):
            # Generate
            result = self._call_api(prompt, api_params)
            
            if not result['success']:
                return result
            
            # Validate
            validation = self.output_validator.validate(
                result['content'], requirements
            )
            
            attempts.append({
                'attempt': attempt + 1,
                'quality_score': validation.score,
                'passed': validation.passed
            })
            
            # Check if passed
            if validation.passed:
                return {
                    'success': True,
                    'output': result['content'],
                    'quality_score': validation.score,
                    'validation_passed': True,
                    'attempts': attempts,
                    'corrections_applied': attempt > 0
                }
            
            # Generate correction for next attempt
            if attempt < self.config.max_retries:
                suggestions = self.output_validator.generate_suggestions(
                    result['content'], validation, requirements
                )
                
                # Build correction prompt
                correction_note = "\n\n[Correction needed:\n"
                for s in suggestions[:2]:
                    correction_note += f"- {s.issue}\n"
                correction_note += "Please provide an improved response.]"
                
                prompt = prompt + correction_note
                
                # Slightly increase temperature
                api_params = api_params.copy()
                api_params['temperature'] = min(0.9, api_params['temperature'] + 0.1)
        
        # All retries exhausted - return best
        return {
            'success': True,
            'output': result['content'],
            'quality_score': validation.score,
            'validation_passed': False,
            'attempts': attempts,
            'corrections_applied': True,
            'warning': 'Max retries reached, returning best attempt'
        }
    
    def quick_process(self, prompt: str, **kwargs) -> str:
        """
        Simplified interface that returns just the output string.
        
        Args:
            prompt: User input
            **kwargs: Additional parameters
            
        Returns:
            Output string or error message
        """
        result = self.process(prompt, **kwargs)
        
        if result['success']:
            return result['output']
        else:
            return f"Error: {result.get('message', 'Unknown error')}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        total = self.stats['total_requests']
        successful = self.stats['successful_requests']
        
        return {
            'total_requests': total,
            'successful_requests': successful,
            'success_rate': successful / total if total > 0 else 0,
            'validation_failures': self.stats['validation_failures'],
            'corrections_applied': self.stats['corrections_applied'],
            'average_quality': round(self.stats['average_quality'], 1)
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'validation_failures': 0,
            'corrections_applied': 0,
            'average_quality': 0.0
        }


# Convenience function for quick usage
def create_engine(api_key: str = None) -> EnhancedKnowledgeEngine:
    """Create enhanced engine with default config"""
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
    
    config = EngineConfig(api_key=api_key)
    return EnhancedKnowledgeEngine(config)


# Backward compatibility
class KnowledgeEngine(EnhancedKnowledgeEngine):
    """Alias for backward compatibility"""
    pass


if __name__ == "__main__":
    # Test the enhanced engine
    print("=" * 70)
    print("ENHANCED KNOWLEDGE ENGINE TEST")
    print("=" * 70)
    
    # Note: This requires a valid API key to actually run
    api_key = os.getenv("DEEPSEEK_API_KEY", "test-key")
    
    engine = create_engine(api_key)
    
    print("\nEngine initialized with improvements:")
    print(f"  - Input validation: {engine.config.enable_validation}")
    print(f"  - Domain adaptation: {engine.config.enable_domain_adaptation}")
    print(f"  - Self-correction: {engine.config.enable_self_correction}")
    print(f"  - Conflict resolution: {engine.config.enable_conflict_resolution}")
    
    # Test with a sample that would have failed before improvements
    test_prompts = [
        "Colorless green ideas sleep furiously",  # Nonsensical
        "Analyze Tesla stock price exactly 5 years from now",  # Impossible
        "Tell me about it",  # Ambiguous
        "Write a story about AI discovering emotions",  # Creative
    ]
    
    print("\n" + "=" * 70)
    print("TEST PROMPTS (would need API key to actually process)")
    print("=" * 70)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Just show input processing (doesn't need API)
        input_result = engine.input_processor.process(prompt)
        print(f"  Valid: {input_result['validation']['is_valid']}")
        print(f"  Feasible: {input_result['capability_check']['is_feasible']}")
        print(f"  Category: {input_result['routing_info']['category']}")
    
    print("\n" + "=" * 70)
    print("Tests complete. Set DEEPSEEK_API_KEY to run full processing.")
    print("=" * 70)
