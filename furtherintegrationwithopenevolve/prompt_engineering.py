"""
Prompt Engineering System for OpenEvolve
Implements the Prompt Engineering functionality described in the ultimate explanation document.
"""
import json
import re
import tempfile
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import jinja2
from jinja2 import Template
import copy
import hashlib
import time
from datetime import datetime

# Import OpenEvolve components for enhanced functionality
try:
    from openevolve.api import run_evolution as openevolve_run_evolution
    from openevolve.config import Config, LLMModelConfig
    from openevolve.evaluation_result import EvaluationResult
    from openevolve.evaluator import Evaluator
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available - using fallback implementation")

class PromptType(Enum):
    """Enumeration of different prompt types"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CRITIQUE = "critique"
    PATCH = "patch"
    EVALUATION = "evaluation"
    APPROVAL = "approval"

class ContentType(Enum):
    """Content types for prompt specialization"""
    CODE = "code"
    DOCUMENT = "document"
    PROTOCOL = "protocol"
    LEGAL = "legal"
    MEDICAL = "medical"
    TECHNICAL = "technical"
    GENERAL = "general"

@dataclass
class PromptTemplate:
    """Data class for prompt templates"""
    name: str
    description: str
    template: str
    variables: List[str]
    prompt_type: PromptType
    content_types: List[ContentType]
    version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""
    tags: List[str] = None

@dataclass
class PromptInstance:
    """Data class for instantiated prompts"""
    template_name: str
    rendered_prompt: str
    variables_used: Dict[str, Any]
    context: Dict[str, Any]
    rendered_at: str = ""
    prompt_type: PromptType = None

class PromptOptimizer:
    """Class for optimizing prompts based on effectiveness"""
    
    def __init__(self):
        self.effectiveness_scores = {}
        self.optimization_history = {}
    
    def evaluate_effectiveness(self, prompt_id: str, response_quality: float, processing_time: float, cost: float) -> float:
        """
        Evaluate the effectiveness of a prompt based on multiple metrics
        
        Args:
            prompt_id: Unique identifier for the prompt
            response_quality: Quality score of the response (0-1)
            processing_time: Time taken to process (in seconds)
            cost: Cost of processing
            
        Returns:
            Effectiveness score (0-1)
        """
        # Weighted effectiveness calculation
        # Higher response quality = higher effectiveness
        # Lower processing time = higher effectiveness  
        # Lower cost = higher effectiveness
        time_factor = max(0, 1 - (processing_time / 60))  # Normalize time (assuming max 60s is bad)
        cost_factor = max(0, 1 - cost)  # Normalize cost
        
        effectiveness = (response_quality * 0.6) + (time_factor * 0.2) + (cost_factor * 0.2)
        
        # Store effectiveness score
        if prompt_id not in self.effectiveness_scores:
            self.effectiveness_scores[prompt_id] = []
        self.effectiveness_scores[prompt_id].append({
            'timestamp': datetime.now().isoformat(),
            'score': effectiveness,
            'response_quality': response_quality,
            'processing_time': processing_time,
            'cost': cost
        })
        
        return effectiveness
    
    def get_best_prompts(self, prompt_type: PromptType = None, limit: int = 5) -> List[str]:
        """Get the best performing prompts"""
        if not self.effectiveness_scores:
            return []
        
        # Calculate average effectiveness for each prompt
        avg_scores = {}
        for prompt_id, scores in self.effectiveness_scores.items():
            avg_scores[prompt_id] = sum(s['score'] for s in scores) / len(scores)
        
        # Sort by average effectiveness
        sorted_prompts = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter by prompt type if specified
        if prompt_type:
            # This would require keeping track of prompt types, simplified for this example
            pass
        
        return [prompt_id for prompt_id, _ in sorted_prompts[:limit]]

class PromptManager:
    """Central manager for all prompt-related functionality"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.custom_templates: Dict[str, PromptTemplate] = {}
        self.optimizer = PromptOptimizer()
        self.context_processors: Dict[str, Callable] = {}
        self.variable_extractors: Dict[str, Callable] = {}
        
        # Load default templates
        self._load_default_templates()
        
        # Register default context processors
        self._register_default_context_processors()
        
        # Register default variable extractors
        self._register_default_variable_extractors()
    
    def _load_default_templates(self):
        """Load default prompt templates based on the existing functionality"""
        # Red Team Critique Template
        self.add_template(PromptTemplate(
            name="red_team_critique",
            description="Template for red team (critics) to find flaws and vulnerabilities",
            template="""You are a critical reviewer examining {{ content_type }} for flaws and vulnerabilities.
Your role is to identify potential problems, risks, and weaknesses from a red-team perspective.
Focus on finding:
1. Logical errors or gaps in the procedure
2. Security vulnerabilities or risks  
3. Ambiguous or unclear instructions
4. Potential for misinterpretation
5. Missing edge cases or exception handling
6. Compliance with best practices
7. Performance issues
8. Scalability concerns

Be specific and constructive in your critique.
{% if compliance_requirements %}{{ compliance_requirements }}{% endif %}

Respond in JSON format with:
- "issues": [array of issue objects, each with "title", "description", "severity" (low/medium/high/critical), and "category"]
- "overall_assessment": "string with your overall assessment"
- "suggestions": [array of improvement suggestions]

Content: {{ content }}""",
            variables=["content", "content_type", "compliance_requirements"],
            prompt_type=PromptType.CRITIQUE,
            content_types=[ContentType.CODE, ContentType.DOCUMENT, ContentType.PROTOCOL, ContentType.GENERAL]
        ))
        
        # Blue Team Patch Template
        self.add_template(PromptTemplate(
            name="blue_team_patch",
            description="Template for blue team (fixers) to address identified issues",
            template="""You are an improvement specialist tasked with fixing issues in {{ content_type }}.
A red team has identified the following issues that need to be addressed:
{{ critiques }}

Your task is to improve the content by addressing these issues while preserving its core purpose and functionality.
Update the content by incorporating fixes for the identified issues.
Focus on:
1. Resolving all identified issues
2. Maintaining or improving clarity
3. Ensuring consistency with best practices
4. Preserving original intent

Respond in JSON format with:
- "content": "the updated content with fixes applied" 
- "mitigation_matrix": [{"issue": "issue title", "status": "resolved/mitigated/acknowledged", "approach": "brief description of how it was addressed"}]
- "residual_risks": ["list of any remaining risks or concerns"]

Original Content: {{ content }}""",
            variables=["content", "critiques", "content_type"],
            prompt_type=PromptType.PATCH,
            content_types=[ContentType.CODE, ContentType.DOCUMENT, ContentType.PROTOCOL, ContentType.GENERAL]
        ))
        
        # Evaluation Template
        self.add_template(PromptTemplate(
            name="evaluator_assessment",
            description="Template for evaluator team to judge quality and fitness",
            template="""You are an evaluator assessing the quality of {{ content_type }}.
Please evaluate the provided content according to these criteria:
1. Clarity: Is the content clear and unambiguous?
2. Completeness: Does the content cover all necessary aspects?
3. Correctness: Is the content factually accurate and logically consistent?
4. Effectiveness: Does the content achieve its intended objectives?
5. Compliance: Does the content adhere to relevant standards?
6. Maintainability: Is the content easy to maintain and modify?
7. Performance: Is the content efficient in its execution (where applicable)?

Rate each criterion on a scale of 1-10, with 10 being excellent.
Provide specific feedback for each criterion.

{% if custom_requirements %}{{ custom_requirements }}{% endif %}

Content: {{ content }}

Respond in JSON format with:
- "scores": {"clarity": score, "completeness": score, "correctness": score, "effectiveness": score, "compliance": score, "maintainability": score, "performance": score}
- "overall_score": average of all scores
- "detailed_feedback": "detailed feedback for each criterion"
- "recommendations": ["list of specific recommendations for improvement"]""",
            variables=["content", "content_type", "custom_requirements"],
            prompt_type=PromptType.EVALUATION,
            content_types=[ContentType.CODE, ContentType.DOCUMENT, ContentType.PROTOCOL, ContentType.GENERAL]
        ))
        
        # Approval Template
        self.add_template(PromptTemplate(
            name="approval_check",
            description="Template for final approval verification",
            template="""You are an evaluator assessing the quality of {{ content_type }}.
Please evaluate the provided content and determine if it meets the required standards for approval.
Consider the following:
1. Has the content addressed the original requirements?
2. Is the content of sufficient quality for its intended use?
3. Are there any remaining critical issues that need to be resolved?

{% if approval_criteria %}{{ approval_criteria }}{% endif %}

Content: {{ content }}

Respond in JSON format with:
- "verdict": "APPROVED" or "REJECTED"
- "score": 0-100 (numerical score)
- "reasons": [array of brief reason strings for the verdict]
- "suggestions": [array of improvement suggestions if any]

Content: {{ content }}""",
            variables=["content", "content_type", "approval_criteria"],
            prompt_type=PromptType.APPROVAL,
            content_types=[ContentType.CODE, ContentType.DOCUMENT, ContentType.PROTOCOL, ContentType.GENERAL]
        ))
    
    def _register_default_context_processors(self):
        """Register default context processors"""
        self.context_processors['default'] = self._default_context_processor
        self.context_processors['code'] = self._code_context_processor
        self.context_processors['legal'] = self._legal_context_processor
        self.context_processors['medical'] = self._medical_context_processor
        self.context_processors['technical'] = self._technical_context_processor
    
    def _register_default_variable_extractors(self):
        """Register default variable extractors"""
        self.variable_extractors['content'] = self._extract_content_variable
        self.variable_extractors['content_type'] = self._extract_content_type_variable
        self.variable_extractors['compliance_requirements'] = self._extract_compliance_variable
    
    def _default_context_processor(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default context processor"""
        # Add default context variables
        context['timestamp'] = datetime.now().isoformat()
        context['current_date'] = datetime.now().strftime('%Y-%m-%d')
        return context
    
    def _code_context_processor(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Code-specific context processor"""
        context = self._default_context_processor(context)
        # Add code-specific context
        context['programming_paradigm'] = context.get('programming_paradigm', 'general')
        context['target_platform'] = context.get('target_platform', 'any')
        context['security_requirements'] = context.get('security_requirements', 'standard')
        return context
    
    def _legal_context_processor(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Legal-specific context processor"""
        context = self._default_context_processor(context)
        # Add legal-specific context
        context['jurisdiction'] = context.get('jurisdiction', 'generic')
        context['regulatory_framework'] = context.get('regulatory_framework', 'none')
        context['compliance_level'] = context.get('compliance_level', 'standard')
        return context
    
    def _medical_context_processor(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Medical-specific context processor"""
        context = self._default_context_processor(context)
        # Add medical-specific context
        context['medical_specialty'] = context.get('medical_specialty', 'general')
        context['patient_population'] = context.get('patient_population', 'general')
        context['regulatory_compliance'] = context.get('regulatory_compliance', 'none')
        return context
    
    def _technical_context_processor(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Technical-specific context processor"""
        context = self._default_context_processor(context)
        # Add technical-specific context
        context['technical_domain'] = context.get('technical_domain', 'general')
        context['implementation_complexity'] = context.get('implementation_complexity', 'moderate')
        return context
    
    def _extract_content_variable(self, content: str) -> str:
        """Extract content variable"""
        return content
    
    def _extract_content_type_variable(self, content_type: str) -> str:
        """Extract content type variable"""
        try:
            return ContentType(content_type).value
        except ValueError:
            return ContentType.GENERAL.value
    
    def _extract_compliance_variable(self, compliance_rules: List[str]) -> str:
        """Extract compliance variable"""
        if not compliance_rules:
            return ""
        return "Compliance requirements: " + "; ".join(compliance_rules)
    
    def add_template(self, template: PromptTemplate):
        """Add a new prompt template"""
        template.created_at = datetime.now().isoformat()
        self.templates[template.name] = template
    
    def update_template(self, template: PromptTemplate):
        """Update an existing prompt template"""
        if template.name in self.templates:
            template.updated_at = datetime.now().isoformat()
            self.templates[template.name] = template
        else:
            raise ValueError(f"Template {template.name} does not exist")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name"""
        return self.templates.get(name) or self.custom_templates.get(name)
    
    def remove_template(self, name: str):
        """Remove a template by name"""
        if name in self.templates:
            del self.templates[name]
        elif name in self.custom_templates:
            del self.custom_templates[name]
        else:
            raise ValueError(f"Template {name} does not exist")
    
    def instantiate_prompt(self, template_name: str, variables: Dict[str, Any], 
                         context: Dict[str, Any] = None) -> PromptInstance:
        """
        Instantiate a prompt template with specific variables
        
        Args:
            template_name: Name of the template to instantiate
            variables: Dictionary of variables to substitute
            context: Additional context for processing
            
        Returns:
            PromptInstance with the rendered prompt
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        # Create a copy of context and add variables
        full_context = copy.deepcopy(context) if context else {}
        full_context.update(variables)
        
        # Apply context processors based on content type
        content_type = full_context.get('content_type', 'general')
        if content_type in self.context_processors:
            full_context = self.context_processors[content_type](full_context)
        else:
            full_context = self.context_processors['default'](full_context)
        
        # Create Jinja2 template and render
        jinja_template = Template(template.template)
        rendered_prompt = jinja_template.render(**full_context)
        
        return PromptInstance(
            template_name=template_name,
            rendered_prompt=rendered_prompt,
            variables_used=variables,
            context=full_context,
            rendered_at=datetime.now().isoformat(),
            prompt_type=template.prompt_type
        )
    
    def create_dynamic_template(self, name: str, description: str, template: str,
                               variables: List[str], prompt_type: PromptType,
                               content_types: List[ContentType]) -> PromptTemplate:
        """
        Create a dynamic prompt template
        
        Args:
            name: Name of the template
            description: Description of the template
            template: Template string with Jinja2 syntax
            variables: List of required variables
            prompt_type: Type of prompt
            content_types: List of applicable content types
            
        Returns:
            Created PromptTemplate
        """
        new_template = PromptTemplate(
            name=name,
            description=description,
            template=template,
            variables=variables,
            prompt_type=prompt_type,
            content_types=content_types
        )
        
        self.custom_templates[name] = new_template
        return new_template
    
    def optimize_prompt(self, prompt_id: str, response_quality: float, 
                       processing_time: float, cost: float) -> float:
        """
        Optimize a prompt based on its effectiveness
        
        Args:
            prompt_id: ID of the prompt to optimize
            response_quality: Quality of the response (0-1)
            processing_time: Processing time in seconds
            cost: Cost of processing
            
        Returns:
            Effectiveness score
        """
        return self.optimizer.evaluate_effectiveness(
            prompt_id, response_quality, processing_time, cost
        )
    
    def get_best_prompts_for_content_type(self, content_type: ContentType, 
                                        prompt_type: PromptType = None) -> List[str]:
        """
        Get best prompts for a specific content type
        
        Args:
            content_type: Type of content
            prompt_type: Optional type of prompt
            
        Returns:
            List of best prompt names
        """
        # Find templates that match the content type
        matching_templates = []
        for name, template in self.templates.items():
            if content_type in template.content_types:
                if prompt_type is None or template.prompt_type == prompt_type:
                    matching_templates.append(name)
        
        # If optimization data exists, rank by effectiveness
        if self.optimizer.effectiveness_scores:
            effective_templates = []
            for template_name in matching_templates:
                if template_name in self.optimizer.effectiveness_scores:
                    scores = self.optimizer.effectiveness_scores[template_name]
                    avg_score = sum(s['score'] for s in scores) / len(scores)
                    effective_templates.append((template_name, avg_score))
            
            # Sort by effectiveness and return names
            effective_templates.sort(key=lambda x: x[1], reverse=True)
            return [name for name, _ in effective_templates[:10]]  # Top 10
        
        return matching_templates
    
    def validate_template_variables(self, template_name: str, variables: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate that required variables are provided for a template
        
        Args:
            template_name: Name of the template
            variables: Variables to validate
            
        Returns:
            Dictionary mapping variable names to validation status
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        validation_results = {}
        for var in template.variables:
            validation_results[var] = var in variables
        
        return validation_results
    
    def get_prompt_complexity(self, prompt_instance: PromptInstance) -> Dict[str, Any]:
        """
        Analyze the complexity of a prompt instance
        
        Args:
            prompt_instance: The prompt instance to analyze
            
        Returns:
            Complexity analysis
        """
        content = prompt_instance.rendered_prompt
        word_count = len(content.split())
        char_count = len(content)
        
        # Count complexity indicators
        question_count = len(re.findall(r'\?', content))
        exclamation_count = len(re.findall(r'!', content))
        conditional_count = len(re.findall(r'\b(if|when|provided|assuming|given)\b', content.lower()))
        list_count = len(re.findall(r'[-*]\s', content)) + len(re.findall(r'\d+\.\s', content))
        
        # Estimate cognitive load
        cognitive_load = (
            (word_count / 100) * 0.2 +  # Longer prompts = higher load
            (question_count * 0.3) +    # More questions = higher load
            (conditional_count * 0.2) + # More conditions = higher load
            (list_count * 0.1)          # More lists = slightly higher load
        )
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'question_count': question_count,
            'exclamation_count': exclamation_count,
            'conditional_count': conditional_count,
            'list_count': list_count,
            'cognitive_load_estimate': min(10, cognitive_load),  # Cap at 10
            'estimated_reading_time': word_count / 200  # Assuming 200 wpm reading speed
        }
    
    def batch_instantiate_prompts(self, template_name: str, 
                                  variable_sets: List[Dict[str, Any]], 
                                  context: Dict[str, Any] = None) -> List[PromptInstance]:
        """
        Instantiate multiple prompts using the same template but different variables
        
        Args:
            template_name: Name of the template to use
            variable_sets: List of variable dictionaries for each instantiation
            context: Common context for all instances
            
        Returns:
            List of PromptInstance objects
        """
        instances = []
        for variables in variable_sets:
            instance = self.instantiate_prompt(template_name, variables, context)
            instances.append(instance)
        return instances

class PromptEngineeringSystem:
    """Main system that coordinates all prompt engineering functionality"""
    
    def __init__(self):
        self.prompt_manager = PromptManager()
        self.evolution_strategies = {}
        self.personalization_engine = {}
        
        # Register evolution strategies
        self._register_evolution_strategies()
    
    def _register_evolution_strategies(self):
        """Register different prompt evolution strategies"""
        self.evolution_strategies['mutation'] = self._mutate_prompt
        self.evolution_strategies['crossover'] = self._crossover_prompts
        self.evolution_strategies['refinement'] = self._refine_prompt
        self.evolution_strategies['contextual_adaptation'] = self._adapt_prompt_contextually
    
    def _mutate_prompt(self, prompt_instance: PromptInstance, mutation_rate: float = 0.1) -> str:
        """
        Apply mutation to a prompt by slightly changing its content
        
        Args:
            prompt_instance: Original prompt instance
            mutation_rate: Rate of mutation (0-1)
            
        Returns:
            Mutated prompt string
        """
        # For now, we'll implement a simple mutation by adding a random note
        import random
        prompt = prompt_instance.rendered_prompt
        
        if random.random() < mutation_rate:
            addition = f"\n\nNote: This is an evolved version. Focus on {random.choice(['clarity', 'depth', 'creativity', 'accuracy'])}."
            return prompt + addition
        
        return prompt
    
    def _crossover_prompts(self, prompt1: PromptInstance, prompt2: PromptInstance, 
                          crossover_point: float = 0.5) -> str:
        """
        Combine two prompts using crossover technique
        
        Args:
            prompt1: First prompt instance
            prompt2: Second prompt instance
            crossover_point: Where to split the prompts (0-1)
            
        Returns:
            Crossed-over prompt string
        """
        text1 = prompt1.rendered_prompt
        text2 = prompt2.rendered_prompt
        
        split_point1 = int(len(text1) * crossover_point)
        split_point2 = int(len(text2) * crossover_point)
        
        return text1[:split_point1] + text2[split_point2:]
    
    def _refine_prompt(self, prompt_instance: PromptInstance, feedback: str) -> str:
        """
        Refine a prompt based on feedback
        
        Args:
            prompt_instance: Original prompt instance
            feedback: Feedback on the prompt's effectiveness
            
        Returns:
            Refined prompt string
        """
        # For now, implement a simple refinement by incorporating feedback as a note
        original_prompt = prompt_instance.rendered_prompt
        return f"{original_prompt}\n\nFeedback: {feedback}\n\nAdjusted approach: Please address the feedback provided above."
    
    def _adapt_prompt_contextually(self, prompt_instance: PromptInstance, 
                                 new_context: Dict[str, Any]) -> PromptInstance:
        """
        Adapt a prompt to a new context
        
        Args:
            prompt_instance: Original prompt instance
            new_context: New context to adapt to
            
        Returns:
            Adapted PromptInstance
        """
        # Get the template used for the original prompt
        template = self.prompt_manager.get_template(prompt_instance.template_name)
        if not template:
            return prompt_instance  # Return original if template not found
        
        # Update the context with new context
        updated_context = prompt_instance.context.copy()
        updated_context.update(new_context)
        
        # Re-render the prompt with new context
        variables = {k: v for k, v in updated_context.items() 
                    if k in template.variables}
        
        return self.prompt_manager.instantiate_prompt(
            template.name, variables, updated_context
        )
    
    def evolve_prompt(self, prompt_instance: PromptInstance, strategy: str = 'refinement',
                     api_key: Optional[str] = None,
                     model_name: str = "gpt-4o",
                     **kwargs) -> PromptInstance:
        """
        Evolve a prompt using various strategies, using OpenEvolve when available
        
        Args:
            prompt_instance: Original prompt instance to evolve
            strategy: Evolution strategy to use
            api_key: API key for OpenEvolve backend (required when using OpenEvolve)
            model_name: Model to use when using OpenEvolve
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Evolved PromptInstance
        """
        # Prioritize OpenEvolve backend when available
        if OPENEVOLVE_AVAILABLE and api_key:
            return self._evolve_prompt_with_openevolve(
                prompt_instance, strategy, api_key, model_name, **kwargs
            )
        
        # Fallback to custom implementation
        return self._evolve_prompt_custom(prompt_instance, strategy, **kwargs)
    
    def _evolve_prompt_with_openevolve(self, prompt_instance: PromptInstance, strategy: str,
                                       api_key: str, model_name: str, **kwargs) -> PromptInstance:
        """
        Evolve a prompt using OpenEvolve backend
        """
        try:
            # Create OpenEvolve configuration
            config = Config()
            
            # Configure LLM model
            llm_config = LLMModelConfig(
                name=model_name,
                api_key=api_key,
                api_base="https://api.openai.com/v1",  # Default, can be overridden
                temperature=0.7,
                max_tokens=4096,
            )
            
            config.llm.models = [llm_config]
            config.evolution.max_iterations = 1  # Just one evolution step
            config.evolution.population_size = 1  # Single prompt evolution
            
            # Create an evaluator for prompt evolution
            def prompt_evaluator(program_path: str) -> Dict[str, Any]:
                """
                Evaluator that performs prompt evolution assessment
                """
                try:
                    with open(program_path, "r", encoding='utf-8') as f:
                        prompt_content = f.read()
                    
                    # Perform basic prompt assessment
                    word_count = len(prompt_content.split())
                    
                    # Return prompt evolution metrics
                    return {
                        "score": 0.85,  # Placeholder evolution score
                        "timestamp": datetime.now().timestamp(),
                        "content_length": len(prompt_content),
                        "word_count": word_count,
                        "prompt_evolution_completed": True
                    }
                except Exception as e:
                    print(f"Error in prompt evaluator: {e}")
                    return {
                        "score": 0.0,
                        "timestamp": datetime.now().timestamp(),
                        "error": str(e)
                    }
            
            # Save prompt to temporary file for OpenEvolve
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding='utf-8') as temp_file:
                temp_file.write(prompt_instance.rendered_prompt)
                temp_file_path = temp_file.name
            
            try:
                # Run prompt evolution using OpenEvolve API
                result = openevolve_run_evolution(
                    initial_program=temp_file_path,
                    evaluator=prompt_evaluator,
                    config=config,
                    iterations=1,
                    output_dir=None,  # Use temporary directory
                    cleanup=True,
                )
                
                # Generate evolved prompt based on OpenEvolve result
                if result.best_code:
                    evolved_prompt = result.best_code
                else:
                    evolved_prompt = prompt_instance.rendered_prompt
                
                # Create new prompt instance with evolved content
                evolved_instance = PromptInstance(
                    template_name=prompt_instance.template_name,
                    rendered_prompt=evolved_prompt,
                    variables_used=prompt_instance.variables_used,
                    context=prompt_instance.context,
                    rendered_at=datetime.now().isoformat(),
                    prompt_type=prompt_instance.prompt_type
                )
                
                return evolved_instance
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        except Exception as e:
            print(f"Error using OpenEvolve backend: {e}")
            # Fallback to custom implementation
            return self._evolve_prompt_custom(prompt_instance, strategy, **kwargs)
    
    def _evolve_prompt_custom(self, prompt_instance: PromptInstance, strategy: str = 'refinement',
                              **kwargs) -> PromptInstance:
        """
        Fallback prompt evolution using custom implementation
        """
        if strategy not in self.evolution_strategies:
            raise ValueError(f"Unknown evolution strategy: {strategy}")
        
        strategy_func = self.evolution_strategies[strategy]
        
        if strategy == 'mutation':
            evolved_text = strategy_func(prompt_instance, kwargs.get('mutation_rate', 0.1))
            return PromptInstance(
                template_name=prompt_instance.template_name,
                rendered_prompt=evolved_text,
                variables_used=prompt_instance.variables_used,
                context=prompt_instance.context,
                rendered_at=datetime.now().isoformat()
            )
        elif strategy == 'crossover':
            other_prompt = kwargs.get('other_prompt')
            if not other_prompt:
                raise ValueError("Crossover strategy requires 'other_prompt' parameter")
            evolved_text = strategy_func(prompt_instance, other_prompt, kwargs.get('crossover_point', 0.5))
            return PromptInstance(
                template_name=prompt_instance.template_name,
                rendered_prompt=evolved_text,
                variables_used=prompt_instance.variables_used,
                context=prompt_instance.context,
                rendered_at=datetime.now().isoformat()
            )
        elif strategy == 'refinement':
            feedback = kwargs.get('feedback', '')
            evolved_text = strategy_func(prompt_instance, feedback)
            return PromptInstance(
                template_name=prompt_instance.template_name,
                rendered_prompt=evolved_text,
                variables_used=prompt_instance.variables_used,
                context=prompt_instance.context,
                rendered_at=datetime.now().isoformat()
            )
        elif strategy == 'contextual_adaptation':
            new_context = kwargs.get('new_context', {})
            return strategy_func(prompt_instance, new_context)
        else:
            # Default behavior
            return prompt_instance
    
    def personalize_prompt(self, prompt_instance: PromptInstance, 
                          user_profile: Dict[str, Any]) -> PromptInstance:
        """
        Personalize a prompt based on user profile
        
        Args:
            prompt_instance: Original prompt instance
            user_profile: User profile containing preferences, expertise, etc.
            
        Returns:
            Personalized PromptInstance
        """
        original_prompt = prompt_instance.rendered_prompt
        user_expertise = user_profile.get('expertise', 'intermediate')
        user_preferences = user_profile.get('preferences', {})
        
        # Adjust the prompt based on user expertise
        if user_expertise == 'beginner':
            # Add more explanatory text for beginners
            personalized_prompt = f"{original_prompt}\n\nPlease provide detailed explanations for technical concepts."
        elif user_expertise == 'expert':
            # Be more concise and technical for experts
            personalized_prompt = f"{original_prompt}\n\nProvide concise, technical responses with minimal explanation."
        else:
            # Default intermediate level
            personalized_prompt = original_prompt
        
        # Apply user preferences
        if user_preferences.get('verbose_mode', False):
            personalized_prompt += "\n\nPlease provide comprehensive responses with multiple examples."
        elif user_preferences.get('brief_mode', False):
            personalized_prompt += "\n\nKeep responses concise and to the point."
        
        if user_preferences.get('formal_tone', False):
            # Convert to more formal language if needed
            replacements = {
                'you': 'the user',
                'your': 'the user\'s',
                'please': 'kindly'
            }
            for old, new in replacements.items():
                personalized_prompt = personalized_prompt.replace(old, new)
        
        return PromptInstance(
            template_name=prompt_instance.template_name,
            rendered_prompt=personalized_prompt,
            variables_used=prompt_instance.variables_used,
            context=prompt_instance.context,
            rendered_at=datetime.now().isoformat()
        )
    
    def evaluate_prompt_effectiveness(self, prompt_instance: PromptInstance, 
                                    response: str, 
                                    metrics: Dict[str, float] = None) -> Dict[str, float]:
        """
        Evaluate the effectiveness of a prompt based on the response
        
        Args:
            prompt_instance: The prompt instance that was used
            response: The response received from the model
            metrics: Additional metrics about the interaction
            
        Returns:
            Effectiveness metrics
        """
        if metrics is None:
            metrics = {}
        
        # Calculate basic effectiveness metrics
        response_length = len(response)
        prompt_length = len(prompt_instance.rendered_prompt)
        
        # Calculate relevance (simplified - in a real system, this would be more complex)
        relevance_score = min(1.0, response_length / max(1, prompt_length * 0.5))
        
        # Calculate coherence (simplified)
        sentences = re.split(r'[.!?]+', response)
        coherent_sentences = sum(1 for s in sentences if len(s.strip()) > 3)
        coherence_score = coherent_sentences / max(1, len(sentences))
        
        # Calculate task completion (simplified)
        template_requirements = prompt_instance.context.get('requirements', [])
        completion_score = 0.5  # Default
        
        if template_requirements:
            response_lower = response.lower()
            satisfied_count = sum(1 for req in template_requirements if req.lower() in response_lower)
            completion_score = satisfied_count / max(1, len(template_requirements))
        
        # Overall effectiveness
        overall_effectiveness = (
            (relevance_score * 0.3) + 
            (coherence_score * 0.3) + 
            (completion_score * 0.4)
        )
        
        # Report effectiveness to optimizer
        prompt_id = f"{prompt_instance.template_name}_{hashlib.md5(prompt_instance.rendered_prompt.encode()).hexdigest()[:8]}"
        self.prompt_manager.optimize_prompt(
            prompt_id,
            overall_effectiveness,
            metrics.get('processing_time', 0),
            metrics.get('cost', 0)
        )
        
        return {
            'relevance_score': relevance_score,
            'coherence_score': coherence_score,
            'completion_score': completion_score,
            'overall_effectiveness': overall_effectiveness,
            'response_length': response_length,
            'prompt_length': prompt_length
        }

# Example usage and testing
def test_prompt_engineering_system():
    """Test function for the Prompt Engineering System"""
    system = PromptEngineeringSystem()
    
    # Test template creation and instantiation
    variables = {
        'content': 'def hello_world():\n    print("Hello, World!")',
        'content_type': 'code_python',
        'compliance_requirements': 'Follow security best practices'
    }
    
    context = {
        'target_audience': 'developers',
        'security_requirements': 'high'
    }
    
    # Instantiate a critique prompt
    critique_prompt = system.prompt_manager.instantiate_prompt(
        'red_team_critique', variables, context
    )
    
    print("Prompt Engineering System Test:")
    print(f"Template used: {critique_prompt.template_name}")
    print(f"Rendered prompt length: {len(critique_prompt.rendered_prompt)} characters")
    print(f"Variables used: {list(critique_prompt.variables_used.keys())}")
    
    # Test prompt evolution
    evolved_prompt = system.evolve_prompt(
        critique_prompt, 
        strategy='refinement',
        feedback="The prompt should be more specific about security vulnerabilities."
    )
    
    print(f"\nEvolved prompt length: {len(evolved_prompt.rendered_prompt)} characters")
    
    # Test effectiveness evaluation
    sample_response = """
    {
        "issues": [
            {
                "title": "Missing Input Validation",
                "description": "Function doesn't validate its inputs",
                "severity": "high",
                "category": "security"
            }
        ],
        "overall_assessment": "Function is vulnerable to injection attacks",
        "suggestions": ["Add input validation"]
    }
    """
    
    effectiveness = system.evaluate_prompt_effectiveness(
        critique_prompt, 
        sample_response, 
        {'processing_time': 2.5, 'cost': 0.05}
    )
    
    print(f"\nEffectiveness metrics: {effectiveness}")
    
    return system

if __name__ == "__main__":
    test_prompt_engineering_system()