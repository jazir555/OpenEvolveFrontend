#!/usr/bin/env python3
"""
Case 2 AI Client Adapter
Adapts existing extractors (OpenAI, Claude) to Case 2 interface without modifying them
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Case2 Ai Adapter
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class Case2AIAdapter:
    """Adapter that provides Case 2-compatible interface for existing extractors"""
    
    def __init__(self, extractor):
        """
        Initialize adapter with existing extractor
        
        Args:
            extractor: OpenAIExtractor or ClaudeExtractor instance
        """
        self.extractor = extractor
        self.extractor_type = type(extractor).__name__
        
        # Determine the underlying client based on extractor type
        if hasattr(extractor, 'client') and hasattr(extractor, 'model'):
            self.client = extractor.client
            self.model = extractor.model
        else:
            raise ValueError(f"Unsupported extractor type: {self.extractor_type}")
        
        # Expose extractor attributes that Case 2 system expects - force temperature to 0.0 for consistency
        self.max_tokens = getattr(extractor, 'max_tokens', 4000)
        self.temperature = 0.0  # Always use 0.0 for Case 2 for deterministic results
        self.api_config = getattr(extractor, 'api_config', {})
        
        # For Claude extractors, also expose model_selection
        if hasattr(extractor, 'model_selection'):
            self.model_selection = extractor.model_selection
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response compatible with Case 2 interface
        Adapts the existing extractor's functionality
        """
        try:
            if self.extractor_type == "OpenAIExtractor":
                return self._generate_openai_response(prompt, **kwargs)
            elif self.extractor_type == "ClaudeExtractor":
                return self._generate_claude_response(prompt, **kwargs)
            else:
                raise ValueError(f"Unsupported extractor type: {self.extractor_type}")
        
        except Exception as e:
            logger.error(f"Error generating response with {self.extractor_type}: {e}")
            raise
    
    def generate_structured_response(self, messages: list, **kwargs) -> str:
        """
        Generate structured response for JSON output
        Handles OpenAI's response_format parameter
        """
        try:
            if self.extractor_type == "OpenAIExtractor":
                # Set defaults from extractor if not provided
                api_kwargs = {
                    'model': self.model,
                    'messages': messages,
                    'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                    'temperature': kwargs.get('temperature', self.temperature)
                }
                
                # Add OpenAI-specific parameters
                if 'response_format' in kwargs:
                    api_kwargs['response_format'] = kwargs['response_format']
                
                response = self.client.chat.completions.create(**api_kwargs)
                return response.choices[0].message.content
                
            elif self.extractor_type == "ClaudeExtractor":
                # Claude doesn't support response_format, so we rely on prompt engineering
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature),
                    messages=[{"role": "user", "content": messages[-1]['content']}]  # Use last message
                )
                return response.content[0].text
            else:
                raise ValueError(f"Unsupported extractor type: {self.extractor_type}")
        
        except Exception as e:
            logger.error(f"Error generating structured response with {self.extractor_type}: {e}")
            raise
    
    def _generate_openai_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI client"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data extraction and document analysis specialist. Analyze documents and extract structured information according to the provided instructions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in OpenAI response generation: {e}")
            raise
    
    def _generate_claude_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Claude client"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error in Claude response generation: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the underlying model"""
        try:
            # Try to get info from the original extractor
            if hasattr(self.extractor, 'get_model_info'):
                base_info = self.extractor.get_model_info()
                base_info['adapter'] = 'Case2AIAdapter'
                return base_info
            else:
                return {
                    "model": self.model,
                    "extractor_type": self.extractor_type,
                    "adapter": "Case2AIAdapter"
                }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "extractor_type": self.extractor_type,
                "adapter": "Case2AIAdapter",
                "error": str(e)
            }
    
    def extract_data(self, *args, **kwargs):
        """Delegate extract_data calls to the original extractor"""
        if hasattr(self.extractor, 'extract_data'):
            return self.extractor.extract_data(*args, **kwargs)
        else:
            raise NotImplementedError(f"extract_data not available in {self.extractor_type}")
    
    def extract_batch(self, *args, **kwargs):
        """Delegate extract_batch calls to the original extractor"""
        if hasattr(self.extractor, 'extract_batch'):
            return self.extractor.extract_batch(*args, **kwargs)
        else:
            raise NotImplementedError(f"extract_batch not available in {self.extractor_type}")