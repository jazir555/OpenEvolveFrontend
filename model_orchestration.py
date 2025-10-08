"""
OpenEvolve Model Orchestration System
Manages multi-model coordination, load balancing, and intelligent ensemble operations
"""
import streamlit as st
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from enum import Enum
import uuid
import statistics
from datetime import datetime, timedelta
import logging

# Try to import OpenEvolve components
try:
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve not available, using basic orchestration")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRole(Enum):
    """Different roles for models in the system"""
    RED_TEAM = "red_team"  # Critics and vulnerability finders
    BLUE_TEAM = "blue_team"  # Fixers and improvers
    EVALUATOR = "evaluator"  # Judges and assessors
    GENERATOR = "generator"  # Content creators
    ANALYZER = "analyzer"  # Content analyzers
    OPTIMIZER = "optimizer"  # Performance optimizers


class ModelTeam(Enum):
    """Enumeration of model teams"""
    RED = "red"
    BLUE = "blue"
    EVALUATOR = "evaluator"


class ModelProvider(Enum):
    """Enumeration of model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"


class OrchestrationStrategy(Enum):
    """Enumeration of orchestration strategies"""
    ROUND_ROBIN = "round_robin"
    RANDOM_SAMPLING = "random_sampling"
    PERFORMANCE_BASED = "performance_based"
    STAGED = "staged"
    ADAPTIVE = "adaptive"
    FOCUS_CATEGORY = "focus_category"


class ModelConfig:
    """Configuration for a model"""
    def __init__(self, model_id: str, provider: ModelProvider, api_key: str, 
                 api_base: str, **kwargs):
        self.model_id = model_id
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        self.presence_penalty = kwargs.get("presence_penalty", 0.0)
        self.seed = kwargs.get("seed", None)
        self.team = kwargs.get("team", ModelTeam.RED)
        self.weight = kwargs.get("weight", 1.0)
        self.enabled = kwargs.get("enabled", True)
        self.performance_score = kwargs.get("performance_score", 0.0)


class ModelPerformance:
    """Performance tracking for a model"""
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.total_requests = 0
        self.success_requests = 0
        self.avg_response_time = 0.0
        self.avg_token_usage = 0
        self.avg_cost = 0.0
        self.issues_found = 0  # For red team models
        self.issues_resolved = 0  # For blue team models
        self.evaluation_score = 0.0  # For evaluator models
        self.last_used = None
        self.performance_history = []


class OrchestrationRequest:
    """Request for model orchestration"""
    def __init__(self, content: str, prompt: str, team: ModelTeam, 
                 temperature: float = 0.7, max_tokens: int = 1000,
                 custom_parameters: Optional[Dict[str, Any]] = None):
        self.content = content
        self.prompt = prompt
        self.team = team
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.custom_parameters = custom_parameters or {}


class OrchestrationResponse:
    """Response from model orchestration"""
    def __init__(self, model_id: str, response: str, success: bool,
                 response_time: float, token_usage: int, cost: float,
                 error_message: Optional[str] = None):
        self.model_id = model_id
        self.response = response
        self.success = success
        self.response_time = response_time
        self.token_usage = token_usage
        self.cost = cost
        self.error_message = error_message


class ModelRegistry:
    """Registry to track and manage models"""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.model_teams: Dict[ModelTeam, List[str]] = {
            ModelTeam.RED: [],
            ModelTeam.BLUE: [],
            ModelTeam.EVALUATOR: []
        }
        self.lock = threading.Lock()
    
    def register_model(self, config: ModelConfig) -> bool:
        """Register a new model configuration"""
        with self.lock:
            self.models[config.model_id] = config
            if config.model_id not in self.model_performance:
                self.model_performance[config.model_id] = ModelPerformance(config.model_id)
            
            # Add to appropriate team
            if config.model_id not in self.model_teams[config.team]:
                self.model_teams[config.team].append(config.model_id)
            
            logger.info(f"Registered model: {config.model_id} for team {config.team.value}")
            return True
    
    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model"""
        with self.lock:
            if model_id in self.models:
                team = self.models[model_id].team
                if model_id in self.model_teams[team]:
                    self.model_teams[team].remove(model_id)
                del self.models[model_id]
                if model_id in self.model_performance:
                    del self.model_performance[model_id]
                logger.info(f"Unregistered model: {model_id}")
                return True
            return False
    
    def get_models_by_team(self, team: ModelTeam) -> List[ModelConfig]:
        """Get all models for a specific team"""
        return [self.models[model_id] for model_id in self.model_teams[team] if model_id in self.models]
    
    def update_model_performance(self, model_id: str, response_success: bool, 
                                response_time: float, token_usage: int, 
                                cost: float, additional_metrics: Dict[str, Any] = None):
        """Update performance metrics for a model"""
        if model_id not in self.model_performance:
            self.model_performance[model_id] = ModelPerformance(model_id)
        
        perf = self.model_performance[model_id]
        perf.total_requests += 1
        
        if response_success:
            perf.success_requests += 1
            
            # Update averages
            total_responses = perf.success_requests
            perf.avg_response_time = ((perf.avg_response_time * (total_responses - 1)) + response_time) / total_responses
            perf.avg_token_usage = ((perf.avg_token_usage * (total_responses - 1)) + token_usage) / total_responses
            perf.avg_cost = ((perf.avg_cost * (total_responses - 1)) + cost) / total_responses
            
            # Update additional metrics based on team
            if additional_metrics:
                if 'issues_found' in additional_metrics:
                    perf.issues_found += additional_metrics['issues_found']
                if 'issues_resolved' in additional_metrics:
                    perf.issues_resolved += additional_metrics['issues_resolved']
                if 'evaluation_score' in additional_metrics:
                    perf.evaluation_score = additional_metrics['evaluation_score']
        
        perf.last_used = datetime.now()
        
        # Store in history
        perf.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'success': response_success,
            'response_time': response_time,
            'token_usage': token_usage,
            'cost': cost,
            **(additional_metrics or {})
        })
        
        # Keep only last 100 performance records
        if len(perf.performance_history) > 100:
            perf.performance_history = perf.performance_history[-100:]


class LoadBalancer:
    """Load balancing for model requests"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.request_counts: Dict[str, int] = {}
        self.last_request_time: Dict[str, datetime] = {}
        self.lock = threading.Lock()
        # Round robin indices for each team
        self.round_robin_indices = {
            ModelTeam.RED: 0,
            ModelTeam.BLUE: 0,
            ModelTeam.EVALUATOR: 0
        }
    
    def get_next_model(self, team: ModelTeam, strategy: OrchestrationStrategy = OrchestrationStrategy.ROUND_ROBIN) -> Optional[str]:
        """Get the next model to use based on strategy"""
        models = self.registry.get_models_by_team(team)
        enabled_models = [m for m in models if m.enabled]
        
        if not enabled_models:
            return None
        
        if strategy == OrchestrationStrategy.ROUND_ROBIN:
            return self._round_robin_selection(team, enabled_models)
        elif strategy == OrchestrationStrategy.RANDOM_SAMPLING:
            return self._random_selection(enabled_models)
        elif strategy == OrchestrationStrategy.PERFORMANCE_BASED:
            return self._performance_based_selection(team, enabled_models)
        elif strategy == OrchestrationStrategy.ADAPTIVE:
            return self._adaptive_selection(team, enabled_models)
        else:
            # Default to round robin
            return self._round_robin_selection(team, enabled_models)
    
    def _round_robin_selection(self, team: ModelTeam, models: List[ModelConfig]) -> str:
        """Select model using round-robin strategy"""
        with self.lock:
            idx = self.round_robin_indices[team]
            model_id = models[idx % len(models)].model_id
            self.round_robin_indices[team] = (idx + 1) % len(models)
            return model_id
    
    def _random_selection(self, models: List[ModelConfig]) -> str:
        """Select model randomly"""
        return random.choice(models).model_id
    
    def _performance_based_selection(self, team: ModelTeam, models: List[ModelConfig]) -> str:
        """Select model based on performance metrics"""
        best_model = None
        best_score = -1
        
        for model in models:
            perf = self.registry.model_performance.get(model.model_id)
            if perf:
                # Calculate performance score based on success rate, speed, and other factors
                success_rate = perf.success_requests / max(1, perf.total_requests)
                # Higher success rate is better, lower response time is better
                score = success_rate * 0.7 + (1 / max(0.001, perf.avg_response_time)) * 0.3
            else:
                score = 0.5  # Default score for new models
            
            if score > best_score:
                best_score = score
                best_model = model.model_id
        
        return best_model or models[0].model_id
    
    def _adaptive_selection(self, team: ModelTeam, models: List[ModelConfig]) -> str:
        """Select model using adaptive strategy based on recent performance"""
        # Use a more sophisticated adaptive approach considering recent performance
        best_model = None
        best_score = -1
        
        for model in models:
            perf = self.registry.model_performance.get(model.model_id)
            if perf and perf.performance_history:
                # Consider last 10 performance samples
                recent_history = perf.performance_history[-10:]
                
                # Calculate recent success rate
                recent_success_count = sum(1 for h in recent_history if h['success'])
                recent_success_rate = recent_success_count / max(1, len(recent_history))
                
                # Calculate average recent response time
                recent_response_times = [h['response_time'] for h in recent_history if h.get('success', True)]
                avg_recent_response_time = sum(recent_response_times) / max(1, len(recent_response_times))
                
                # Calculate score based on recent performance
                score = recent_success_rate * 0.6 + (1 / max(0.001, avg_recent_response_time)) * 0.4
            else:
                score = 0.5  # Default score
            
            if score > best_score:
                best_score = score
                best_model = model.model_id
        
        return best_model or models[0].model_id


class ModelClient:
    """Client for communicating with LLM models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.session = None  # Will be created when needed
    
    def call_model(self, prompt: str, temperature: float = None, max_tokens: int = None) -> Tuple[str, int, float]:
        """
        Call the model with the given prompt
        
        Returns:
            Tuple of (response_text, token_usage, cost)
        """
        # Use provided values or defaults
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        
        # Prepare the request based on provider
        if self.config.provider == ModelProvider.OPENAI:
            return self._call_openai_api(prompt, temp, max_tok)
        elif self.config.provider == ModelProvider.ANTHROPIC:
            return self._call_anthropic_api(prompt, temp, max_tok)
        elif self.config.provider == ModelProvider.GOOGLE:
            return self._call_google_api(prompt, temp, max_tok)
        elif self.config.provider == ModelProvider.OPENROUTER:
            return self._call_openrouter_api(prompt, temp, max_tok)
        else:
            return self._call_custom_api(prompt, temp, max_tok)
    
    def _call_openai_api(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, int, float]:
        """Call OpenAI API using existing OpenEvolve integration"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base
            )
            
            response = client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            
            # Estimate cost based on OpenAI pricing (approximate)
            cost = (prompt_tokens * 0.0000005) + (completion_tokens * 0.0000015)  # GPT-3.5-turbo pricing
            
            return content, prompt_tokens + completion_tokens, cost
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "", 0, 0.0
    
    def _call_anthropic_api(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, int, float]:
        """Call Anthropic API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.config.api_key)
            
            response = client.completions.create(
                model=self.config.model_id,
                prompt=f"Human: {prompt}\n\nAssistant:",
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p
            )
            
            content = response.completion
            # Anthropic doesn't return token count, so estimate
            estimated_tokens = len(content.split())
            cost = estimated_tokens * 0.00001  # Rough estimate
            
            return content, estimated_tokens, cost
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return "", 0, 0.0
    
    def _call_google_api(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, int, float]:
        """Call Google API"""
        # For now, we'll implement a basic structure
        logger.warning("Google API call not fully implemented in this example")
        content = f"Response to: {prompt[:100]}..."  # Simulated response
        estimated_tokens = len(content.split())
        cost = estimated_tokens * 0.000005  # Rough estimate
        return content, estimated_tokens, cost
    
    def _call_openrouter_api(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, int, float]:
        """Call OpenRouter API"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base or "https://openrouter.ai/api/v1"
            )
            
            response = client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            
            # Cost is provided by OpenRouter
            cost = usage.cost if hasattr(usage, 'cost') else 0.0
            
            return content, prompt_tokens + completion_tokens, cost
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            return "", 0, 0.0
    
    def _call_custom_api(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, int, float]:
        """Call custom API"""
        logger.warning("Custom API call not fully implemented in this example")
        content = f"Response to: {prompt[:100]}..."  # Simulated response
        estimated_tokens = len(content.split())
        cost = estimated_tokens * 0.000008  # Rough estimate
        return content, estimated_tokens, cost


class ModelOrchestrator:
    """Main orchestrator that manages model teams and requests"""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_weights: Dict[str, float] = {}
        self.role_assignments: Dict[str, ModelRole] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.selection_strategies = {
            "round_robin": self._round_robin_selection,
            "random": self._random_selection, 
            "performance_based": self._performance_based_selection,
            "staged": self._staged_selection,
            "adaptive": self._adaptive_selection,
            "focus_category": self._focus_category_selection
        }
        
        # Advanced orchestration components
        self.registry = ModelRegistry()
        self.load_balancer = LoadBalancer(self.registry)
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.active_requests: Dict[str, Dict[str, Any]] = {}
    
    def register_model(self, model_name: str, role: ModelRole, weight: float = 1.0, 
                      api_key: str = "", api_base: str = "", **config_kwargs):
        """Register a model with the orchestrator"""
        self.models[model_name] = {
            "name": model_name,
            "role": role,
            "weight": weight,
            "api_key": api_key,
            "api_base": api_base,
            **config_kwargs
        }
        self.model_weights[model_name] = weight
        self.role_assignments[model_name] = role
        self.performance_history[model_name] = []
        
        # Also register with advanced registry
        provider = ModelProvider.OPENAI  # Default, can be overridden
        if "anthropic" in model_name.lower():
            provider = ModelProvider.ANTHROPIC
        elif "claude" in model_name.lower():
            provider = ModelProvider.ANTHROPIC
        elif "gemini" in model_name.lower() or "google" in model_name.lower():
            provider = ModelProvider.GOOGLE
        elif "openrouter" in api_base.lower():
            provider = ModelProvider.OPENROUTER
            
        team_map = {
            ModelRole.RED_TEAM: ModelTeam.RED,
            ModelRole.BLUE_TEAM: ModelTeam.BLUE,
            ModelRole.EVALUATOR: ModelTeam.EVALUATOR,
            ModelRole.GENERATOR: ModelTeam.RED,  # Default to red for generators
            ModelRole.ANALYZER: ModelTeam.EVALUATOR,  # Default to evaluator for analyzers
            ModelRole.OPTIMIZER: ModelTeam.BLUE  # Default to blue for optimizers
        }
        
        model_team = team_map.get(role, ModelTeam.RED)
        
        model_config = ModelConfig(
            model_id=model_name,
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            team=model_team,
            weight=weight,
            temperature=config_kwargs.get("temperature", 0.7),
            max_tokens=config_kwargs.get("max_tokens", 4096),
            top_p=config_kwargs.get("top_p", 1.0),
            frequency_penalty=config_kwargs.get("frequency_penalty", 0.0),
            presence_penalty=config_kwargs.get("presence_penalty", 0.0)
        )
        
        self.registry.register_model(model_config)
    
    def assign_role(self, model_name: str, role: ModelRole):
        """Assign or update a model's role"""
        if model_name in self.models:
            self.role_assignments[model_name] = role
            self.models[model_name]["role"] = role
    
    def get_models_by_role(self, role: ModelRole) -> List[str]:
        """Get all models assigned to a specific role"""
        return [model for model, assigned_role in self.role_assignments.items() 
                if assigned_role == role]
    
    def execute_with_ensemble(
        self, 
        messages: List[Dict[str, str]], 
        role: ModelRole,
        selection_strategy: str = "performance_based",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        num_responses: int = 1,
        weight_adjustment: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute a request with an ensemble of models in a specific role
        """
        available_models = self.get_models_by_role(role)
        if not available_models:
            st.warning(f"No models registered for role: {role.value}")
            return []
        
        # Select models based on strategy
        selected_models = self._select_models(
            available_models, 
            selection_strategy, 
            count=min(num_responses, len(available_models))
        )
        
        responses = []
        with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
            future_to_model = {}
            for model_name in selected_models:
                model_config = self.models[model_name]
                
                future = executor.submit(
                    self._execute_single_model_request,
                    model_name=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **model_config
                )
                future_to_model[future] = model_name
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        result["source_model"] = model_name
                        result["role"] = role.value
                        responses.append(result)
                        
                        # Update performance history
                        if weight_adjustment:
                            self._update_model_performance(model_name, result)
                            
                except Exception as e:
                    st.error(f"Error with model {model_name}: {e}")
        
        return responses
    
    def execute_triad_interaction(
        self,
        content: str,
        content_type: str,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Execute a complete triad interaction with Red Team, Blue Team, and Evaluator
        """
        results = {
            "red_team_analysis": [],
            "blue_team_resolution": [],
            "evaluator_assessment": [],
            "final_decision": None
        }
        
        # Red Team Analysis
        red_team_models = self.get_models_by_role(ModelRole.RED_TEAM)
        if red_team_models:
            red_messages = [{
                "role": "user", 
                "content": f"Analyze this {content_type} content for vulnerabilities, weaknesses, and issues:\n\n{content}"
            }]
            results["red_team_analysis"] = self.execute_with_ensemble(
                messages=red_messages,
                role=ModelRole.RED_TEAM,
                selection_strategy="performance_based",
                temperature=temperature,
                max_tokens=max_tokens,
                num_responses=2
            )
        
        # Blue Team Resolution
        blue_team_models = self.get_models_by_role(ModelRole.BLUE_TEAM)
        if blue_team_models:
            # Create prompt based on red team findings
            red_findings = [r.get("response", "") for r in results["red_team_analysis"]]
            blue_prompt = f"Address these issues found in the {content_type} content:\n\n{chr(10).join(red_findings)}\n\nOriginal content: {content}"
            
            blue_messages = [{"role": "user", "content": blue_prompt}]
            results["blue_team_resolution"] = self.execute_with_ensemble(
                messages=blue_messages,
                role=ModelRole.BLUE_TEAM,
                selection_strategy="performance_based",
                temperature=temperature,
                max_tokens=max_tokens,
                num_responses=2
            )
        
        # Evaluator Assessment
        evaluator_models = self.get_models_by_role(ModelRole.EVALUATOR)
        if evaluator_models:
            evaluation_content = content
            if results["blue_team_resolution"]:
                evaluation_content = results["blue_team_resolution"][0].get("response", content)
            
            evaluation_prompt = f"Evaluate this {content_type} content for quality, correctness, and improvement:\n\n{evaluation_content}"
            evaluation_messages = [{"role": "user", "content": evaluation_prompt}]
            results["evaluator_assessment"] = self.execute_with_ensemble(
                messages=evaluation_messages,
                role=ModelRole.EVALUATOR,
                selection_strategy="performance_based",
                temperature=0.3,  # Lower temp for more consistent evaluation
                max_tokens=max_tokens,
                num_responses=1
            )
        
        # Collate final decision from evaluators
        if results["evaluator_assessment"]:
            results["final_decision"] = self._collate_evaluator_decisions(
                results["evaluator_assessment"]
            )
        
        return results
    
    def _collate_evaluator_decisions(self, evaluator_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate decisions from multiple evaluators"""
        if not evaluator_responses:
            return {"consensus": "no evaluation", "score": 0.0}
        
        # Extract scores and feedback from evaluator responses
        scores = []
        feedback = []
        
        for response in evaluator_responses:
            resp_text = response.get("response", "")
            # Try to extract score from response
            try:
                if "score" in resp_text.lower():
                    # Look for numeric score patterns
                    import re
                    score_matches = re.findall(r"score[^\d]*(\d+\.?\d*)", resp_text.lower())
                    if score_matches:
                        scores.append(float(score_matches[0]))
            except Exception:
                pass
            
            feedback.append(resp_text)
        
        # Calculate average score if available, otherwise default
        avg_score = sum(scores) / len(scores) if scores else 0.7  # Default to 0.7
        
        return {
            "consensus": f"Average score: {avg_score:.2f} across {len(evaluator_responses)} evaluators",
            "score": avg_score,
            "individual_scores": scores,
            "feedback_summary": " | ".join(feedback[:2]),  # Limit to 2 feedbacks
            "num_evaluators": len(evaluator_responses)
        }
    
    def _execute_single_model_request(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **model_config
    ) -> Optional[Dict[str, Any]]:
        """Execute a single request to a model"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=model_config.get("api_key", st.session_state.get("api_key", "")),
                base_url=model_config.get("api_base", "https://api.openai.com/v1")
            )
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30
            )
            
            return {
                "response": response.choices[0].message.content,
                "model": model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            st.error(f"Error executing request to {model_name}: {e}")
            return None
    
    def _select_models(self, available_models: List[str], strategy: str, count: int) -> List[str]:
        """Select models based on the specified strategy"""
        if strategy in self.selection_strategies:
            return self.selection_strategies[strategy](available_models, count)
        else:
            # Default to random selection
            return random.sample(available_models, min(count, len(available_models)))
    
    def _round_robin_selection(self, available_models: List[str], count: int) -> List[str]:
        """Round robin selection - cycle through models"""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected = []
        for _ in range(count):
            if available_models:
                selected.append(available_models[self._round_robin_index % len(available_models)])
                self._round_robin_index += 1
        
        return selected
    
    def _random_selection(self, available_models: List[str], count: int) -> List[str]:
        """Random selection of models"""
        return random.sample(available_models, min(count, len(available_models)))
    
    def _performance_based_selection(self, available_models: List[str], count: int) -> List[str]:
        """Select models based on historical performance"""
        if not self.performance_history:
            return self._random_selection(available_models, count)
        
        # Calculate average performance scores
        perf_scores = {}
        for model in available_models:
            history = self.performance_history.get(model, [])
            if history:
                # Average the performance metrics (simplified)
                scores = [h.get("score", 0.5) for h in history if "score" in h]
                perf_scores[model] = sum(scores) / len(scores) if scores else 0.5
            else:
                perf_scores[model] = 0.5  # Default score
        
        # Sort models by performance and select top performers
        sorted_models = sorted(perf_scores.keys(), key=lambda x: perf_scores[x], reverse=True)
        return sorted_models[:count]
    
    def _staged_selection(self, available_models: List[str], count: int) -> List[str]:
        """Staged selection where different models handle different aspects"""
        # For now, just return a random sample
        # In the future, this could implement more sophisticated staged processing
        return random.sample(available_models, min(count, len(available_models)))
    
    def _adaptive_selection(self, available_models: List[str], count: int) -> List[str]:
        """Adaptive selection based on current needs and context"""
        # For now, use performance-based approach
        return self._performance_based_selection(available_models, count)
    
    def _focus_category_selection(self, available_models: List[str], count: int) -> List[str]:
        """Select models based on specific focus categories"""
        # For now, return a balanced selection
        return random.sample(available_models, min(count, len(available_models)))
    
    def _update_model_performance(self, model_name: str, result: Dict[str, Any]):
        """Update model performance history"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        # Record performance metrics
        performance_record = {
            "timestamp": result.get("timestamp", time.time()),
            "success": result.get("response") is not None,
            "response_time": 0,  # Response time not measured here
            "tokens_used": result.get("usage", {}).get("total_tokens", 0)
        }
        
        # Add to history (keep last 50 records)
        self.performance_history[model_name].append(performance_record)
        if len(self.performance_history[model_name]) > 50:
            self.performance_history[model_name] = self.performance_history[model_name][-50:]
    
    def get_model_performance_metrics(self, model_name: str = None) -> Dict[str, Any]:
        """Get performance metrics for models"""
        if model_name:
            history = self.performance_history.get(model_name, [])
            if not history:
                return {"model": model_name, "avg_success_rate": 0, "total_requests": 0}
            
            success_count = sum(1 for record in history if record.get("success", False))
            return {
                "model": model_name,
                "avg_success_rate": success_count / len(history) if history else 0,
                "total_requests": len(history),
                "avg_tokens_used": sum(record.get("tokens_used", 0) for record in history) / len(history) if history else 0
            }
        else:
            # Return metrics for all models
            all_metrics = {}
            for model in self.models.keys():
                all_metrics[model] = self.get_model_performance_metrics(model)
            return all_metrics
    
    def load_model_configurations_from_openevolve(self, config) -> bool:
        """Load model configurations from OpenEvolve Config object"""
        if not OPENEVOLVE_AVAILABLE or not config:
            return False
        
        try:
            # Extract LLM models from OpenEvolve config 
            for llm_config in getattr(config.llm, 'models', []):
                # Map OpenEvolve team to frontend role
                role_mapping = {
                    "red_team": ModelRole.RED_TEAM,
                    "blue_team": ModelRole.BLUE_TEAM,
                    "evaluator": ModelRole.EVALUATOR,
                    "generator": ModelRole.GENERATOR,
                    "analyzer": ModelRole.ANALYZER,
                    "optimizer": ModelRole.OPTIMIZER
                }
                
                # Default to generator role
                role = role_mapping.get(getattr(llm_config, 'role', 'generator'), ModelRole.GENERATOR)
                
                self.register_model(
                    model_name=llm_config.name,
                    role=role,
                    weight=getattr(llm_config, 'weight', 1.0),
                    api_key=getattr(llm_config, 'api_key', ''),
                    api_base=getattr(llm_config, 'api_base', 'https://api.openai.com/v1'),
                    temperature=getattr(llm_config, 'temperature', 0.7),
                    max_tokens=getattr(llm_config, 'max_tokens', 4096),
                    top_p=getattr(llm_config, 'top_p', 1.0),
                    frequency_penalty=getattr(llm_config, 'frequency_penalty', 0.0),
                    presence_penalty=getattr(llm_config, 'presence_penalty', 0.0)
                )
            
            return True
        except Exception as e:
            st.error(f"Error loading OpenEvolve model configurations: {e}")
            return False
    
    def submit_request(self, request: OrchestrationRequest, strategy: OrchestrationStrategy = OrchestrationStrategy.ROUND_ROBIN) -> str:
        """
        Submit a request for processing by the appropriate team
        
        Returns:
            Request ID for tracking
        """
        request_id = str(uuid.uuid4())
        
        # Store request info
        self.active_requests[request_id] = {
            'request': request,
            'start_time': datetime.now(),
            'strategy': strategy
        }
        
        # Get the appropriate model for this request
        model_id = self.load_balancer.get_next_model(request.team, strategy)
        if not model_id:
            logger.error(f"No models available for team: {request.team.value}")
            return request_id
        
        # Get model config
        model_config = self.registry.models.get(model_id)
        if not model_config:
            logger.error(f"Model config not found for: {model_id}")
            return request_id
        
        # Update load balancer
        # self.load_balancer.update_request_count(model_id)  # Not implemented in current LoadBalancer
        
        # Execute the request in a separate thread
        future = self.executor.submit(
            self._execute_advanced_model_request,
            request_id,
            model_config,
            request
        )
        
        self.active_requests[request_id]['future'] = future
        
        return request_id
    
    def _execute_advanced_model_request(self, request_id: str, model_config: ModelConfig, 
                                        request: OrchestrationRequest) -> OrchestrationResponse:
        """Execute a single model request using advanced orchestration"""
        start_time = time.time()
        
        try:
            client = ModelClient(model_config)
            response_text, token_usage, cost = client.call_model(
                prompt=request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            response_time = time.time() - start_time
            
            # Create response object
            response = OrchestrationResponse(
                model_id=model_config.model_id,
                response=response_text,
                success=True,
                response_time=response_time,
                token_usage=token_usage,
                cost=cost
            )
            
            # Update performance metrics
            additional_metrics = request.custom_parameters or {}
            self.registry.update_model_performance(
                model_config.model_id,
                response_success=True,
                response_time=response_time,
                token_usage=token_usage,
                cost=cost,
                additional_metrics=additional_metrics
            )
            
            # Store response in active requests
            self.active_requests[request_id]['response'] = response
            
            logger.info(f"Successfully processed request {request_id} with model {model_config.model_id}")
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            
            response = OrchestrationResponse(
                model_id=model_config.model_id,
                response="",
                success=False,
                response_time=response_time,
                token_usage=0,
                cost=0.0,
                error_message=str(e)
            )
            
            # Update performance metrics for failure
            self.registry.update_model_performance(
                model_config.model_id,
                response_success=False,
                response_time=response_time,
                token_usage=0,
                cost=0.0
            )
            
            # Store response in active requests
            self.active_requests[request_id]['response'] = response
            
            logger.error(f"Failed to process request {request_id} with model {model_config.model_id}: {e}")
            
            return response
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get the status of a specific request"""
        if request_id not in self.active_requests:
            return {'status': 'not_found', 'request_id': request_id}
        
        request_info = self.active_requests[request_id]
        result = {
            'status': 'submitted',
            'request_id': request_id,
            'team': request_info['request'].team.value,
            'start_time': request_info['start_time'].isoformat(),
            'strategy': request_info['strategy'].value
        }
        
        if 'response' in request_info:
            result['status'] = 'completed'
            result['response'] = {
                'response': request_info['response'].response,
                'model': request_info['response'].model_id,
                'success': request_info['response'].success,
                'response_time': request_info['response'].response_time,
                'token_usage': request_info['response'].token_usage,
                'cost': request_info['response'].cost
            }
        elif 'future' in request_info:
            if request_info['future'].done():
                result['status'] = 'completed'
                try:
                    response_obj = request_info['future'].result()
                    result['response'] = {
                        'response': response_obj.response,
                        'model': response_obj.model_id,
                        'success': response_obj.success,
                        'response_time': response_obj.response_time,
                        'token_usage': response_obj.token_usage,
                        'cost': response_obj.cost
                    }
                except Exception as e:
                    result['status'] = 'failed'
                    result['error'] = str(e)
            else:
                result['status'] = 'processing'
        
        return result
    
    def get_team_performance_report(self, team: ModelTeam) -> Dict[str, Any]:
        """Get performance report for an entire team"""
        models = self.registry.get_models_by_team(team)
        
        team_performance = {
            'team': team.value,
            'total_models': len(models),
            'enabled_models': len([m for m in models if m.enabled]),
            'models': []
        }
        
        total_requests = 0
        total_success = 0
        total_response_time = 0
        total_tokens = 0
        total_cost = 0
        
        for model in models:
            perf = self.registry.model_performance.get(model.model_id)
            if not perf:
                continue
                
            model_stats = {
                'model_id': model.model_id,
                'success_rate': perf.success_requests / max(1, perf.total_requests),
                'avg_response_time': perf.avg_response_time,
                'avg_token_usage': perf.avg_token_usage,
                'avg_cost': perf.avg_cost,
                'total_requests': perf.total_requests,
                'total_success': perf.success_requests
            }
            
            if team == ModelTeam.RED:
                model_stats['avg_issues_found'] = perf.issues_found / max(1, perf.success_requests)
            elif team == ModelTeam.BLUE:
                model_stats['avg_issues_resolved'] = perf.issues_resolved / max(1, perf.success_requests)
            elif team == ModelTeam.EVALUATOR:
                model_stats['avg_evaluation_score'] = perf.evaluation_score
            
            team_performance['models'].append(model_stats)
            
            total_requests += perf.total_requests
            total_success += perf.success_requests
            total_response_time += perf.avg_response_time * perf.success_requests
            total_tokens += perf.avg_token_usage * perf.success_requests
            total_cost += perf.avg_cost * perf.success_requests
        
        if total_success > 0:
            team_performance['overall_success_rate'] = total_success / max(1, total_requests)
            team_performance['overall_avg_response_time'] = total_response_time / total_success
            team_performance['overall_avg_token_usage'] = total_tokens / total_success
            team_performance['overall_avg_cost'] = total_cost / total_success
        else:
            team_performance['overall_success_rate'] = 0
            team_performance['overall_avg_response_time'] = 0
            team_performance['overall_avg_token_usage'] = 0
            team_performance['overall_avg_cost'] = 0
        
        return team_performance
    
    def get_orchestration_efficiency_metrics(self) -> Dict[str, Any]:
        """Get metrics on orchestration efficiency"""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        # Calculate request distribution across teams
        team_requests = {}
        for team in ModelTeam:
            models = self.registry.get_models_by_team(team)
            team_requests[team.value] = len(models)
        
        # Calculate recent performance
        recent_successes = 0
        recent_failures = 0
        recent_response_times = []
        
        for request_info in self.active_requests.values():
            if 'response' in request_info:
                response = request_info['response']
                if response.success:
                    recent_successes += 1
                    recent_response_times.append(response.response_time)
                else:
                    recent_failures += 1
            elif request_info['start_time'] > one_hour_ago:
                # Count in-progress requests as "waiting"
                pass
        
        return {
            'total_active_requests': len(self.active_requests),
            'recent_successes': recent_successes,
            'recent_failures': recent_failures,
            'success_rate': recent_successes / max(1, recent_successes + recent_failures),
            'avg_recent_response_time': statistics.mean(recent_response_times) if recent_response_times else 0,
            'min_recent_response_time': min(recent_response_times) if recent_response_times else 0,
            'max_recent_response_time': max(recent_response_times) if recent_response_times else 0,
            'team_distribution': team_requests,
            'timestamp': now.isoformat()
        }


def create_openevolve_model_orchestrator(
    config,
    red_team_models: List[str],
    blue_team_models: List[str], 
    evaluator_models: List[str]
) -> ModelOrchestrator:
    """
    Create a model orchestrator configured with OpenEvolve settings and specialized teams
    """
    orchestrator = ModelOrchestrator()
    
    # Load base models from OpenEvolve config
    orchestrator.load_model_configurations_from_openevolve(config)
    
    # Register specialized models for each team with appropriate roles
    for model_name in red_team_models:
        orchestrator.register_model(model_name, ModelRole.RED_TEAM)
    
    for model_name in blue_team_models:
        orchestrator.register_model(model_name, ModelRole.BLUE_TEAM)
    
    for model_name in evaluator_models:
        orchestrator.register_model(model_name, ModelRole.EVALUATOR)
    
    return orchestrator


def render_model_orchestration_ui():
    """Render the model orchestration UI in Streamlit"""
    st.header("🤖 OpenEvolve Model Orchestration")
    
    # Initialize orchestrator in session state
    if "model_orchestrator" not in st.session_state:
        st.session_state.model_orchestrator = ModelOrchestrator()
    
    orchestrator = st.session_state.model_orchestrator
    
    # Model registration
    with st.expander("Model Registration", expanded=True):
        st.subheader("Register New Model")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_name = st.text_input("Model Name", placeholder="e.g., gpt-4o, claude-3-opus")
        
        with col2:
            role_options = {
                "Red Team (Critic)": ModelRole.RED_TEAM,
                "Blue Team (Fixer)": ModelRole.BLUE_TEAM, 
                "Evaluator (Judge)": ModelRole.EVALUATOR,
                "Generator (Creator)": ModelRole.GENERATOR,
                "Analyzer (Reviewer)": ModelRole.ANALYZER,
                "Optimizer (Enhancer)": ModelRole.OPTIMIZER
            }
            selected_role_label = st.selectbox("Model Role", options=list(role_options.keys()))
            selected_role = role_options[selected_role_label]
        
        with col3:
            model_weight = st.slider("Model Weight", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        
        api_key = st.text_input("API Key", type="password", placeholder="Enter API key")
        api_base = st.text_input("API Base URL", value="https://api.openai.com/v1", 
                                placeholder="Enter API base URL")
        
        if st.button("Register Model", type="secondary"):
            if model_name:
                orchestrator.register_model(
                    model_name=model_name,
                    role=selected_role,
                    weight=model_weight,
                    api_key=api_key,
                    api_base=api_base
                )
                st.success(f"Model '{model_name}' registered with role '{selected_role.value}'")
            else:
                st.error("Please provide a model name")
    
    # Model dashboard
    with st.expander("Model Dashboard", expanded=True):
        st.subheader("Registered Models")
        
        if not orchestrator.models:
            st.info("No models registered yet. Add models using the form above.")
        else:
            # Show models by role
            roles = [ModelRole.RED_TEAM, ModelRole.BLUE_TEAM, ModelRole.EVALUATOR, 
                    ModelRole.GENERATOR, ModelRole.ANALYZER, ModelRole.OPTIMIZER]
            
            for role in roles:
                models_in_role = orchestrator.get_models_by_role(role)
                if models_in_role:
                    st.markdown(f"**{role.value.replace('_', ' ').title()} Models:**")
                    for model_name in models_in_role:
                        model_info = orchestrator.models[model_name]
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.text(f"• {model_name}")
                        with col2:
                            st.text(f"Weight: {model_info['weight']}")
                        with col3:
                            if st.button(f"Details##{model_name}", key=f"details_{model_name}"):
                                st.session_state.selected_model = model_name
            
            # Show performance metrics if a model is selected
            if hasattr(st.session_state, 'selected_model') and st.session_state.selected_model:
                with st.expander(f"Performance Metrics for {st.session_state.selected_model}"):
                    metrics = orchestrator.get_model_performance_metrics(st.session_state.selected_model)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Success Rate", f"{metrics.get('avg_success_rate', 0):.1%}")
                    with col2:
                        st.metric("Requests", metrics.get('total_requests', 0))
                    with col3:
                        avg_tokens = metrics.get('avg_tokens_used', 0)
                        st.metric("Avg Tokens", f"{avg_tokens:.0f}")

    # Ensemble execution demo
    with st.expander("Ensemble Execution Demo", expanded=False):
        st.subheader("Execute with Model Ensemble")
        
        # Select role for ensemble
        role_options = {
            "Red Team (Analysis)": ModelRole.RED_TEAM,
            "Blue Team (Resolution)": ModelRole.BLUE_TEAM,
            "Evaluator (Assessment)": ModelRole.EVALUATOR
        }
        selected_role_label = st.selectbox("Select Role for Ensemble", 
                                         options=list(role_options.keys()), 
                                         key="ensemble_role")
        selected_role = role_options[selected_role_label]
        
        # Get models for selected role
        available_models = orchestrator.get_models_by_role(selected_role)
        if not available_models:
            st.warning(f"No models registered for {selected_role.value} role")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                selection_strategy = st.selectbox("Selection Strategy", 
                                                options=list(orchestrator.selection_strategies.keys()),
                                                key="strategy")
            with col2:
                num_responses = st.number_input("Number of Responses", min_value=1, 
                                              max_value=len(available_models), value=1, key="num_resp")
            with col3:
                temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, 
                                      value=0.7, step=0.1, key="temp")
            
            user_input = st.text_area("Input Content", height=150, 
                                    placeholder="Enter content to process...", key="user_input")
            
            if st.button("Execute Ensemble", type="primary", key="execute_ensemble"):
                if user_input.strip():
                    messages = [{"role": "user", "content": user_input}]
                    
                    with st.spinner(f"Executing with {selected_role.value} ensemble..."):
                        responses = orchestrator.execute_with_ensemble(
                            messages=messages,
                            role=selected_role,
                            selection_strategy=selection_strategy,
                            temperature=temperature,
                            max_tokens=2048,
                            num_responses=num_responses
                        )
                    
                    if responses:
                        st.success(f"Received {len(responses)} responses from ensemble:")
                        for i, response in enumerate(responses):
                            with st.container(border=True):
                                st.write(f"**Response {i+1}** (Model: {response.get('source_model', 'Unknown')})")
                                st.write(response.get("response", "No response"))
                                if "usage" in response:
                                    usage = response["usage"]
                                    st.caption(f"Tokens: {usage.get('total_tokens', 0)} total "
                                             f"({usage.get('prompt_tokens', 0)} in, {usage.get('completion_tokens', 0)} out)")
                    else:
                        st.error("No responses received from ensemble")
                else:
                    st.error("Please enter content to process")


# Example usage and testing
def test_model_orchestration():
    """Test function for the Model Orchestration System"""
    # Create orchestrator
    orchestrator = ModelOrchestrator()
    
    # Register sample models
    orchestrator.register_model(
        model_name="gpt-4o",
        role=ModelRole.RED_TEAM,
        weight=1.0,
        api_key="test-key",
        api_base="https://api.openai.com/v1"
    )
    
    orchestrator.register_model(
        model_name="claude-3-opus",
        role=ModelRole.BLUE_TEAM,
        weight=1.0,
        api_key="test-key",
        api_base="https://api.anthropic.com/v1"
    )
    
    orchestrator.register_model(
        model_name="gemini-1.5-pro",
        role=ModelRole.EVALUATOR,
        weight=1.0,
        api_key="test-key",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    print("Model Orchestration System Test:")
    print(f"Registered models: {list(orchestrator.models.keys())}")
    print(f"Red team models: {[m for m in orchestrator.get_models_by_role(ModelRole.RED_TEAM)]}")
    print(f"Blue team models: {[m for m in orchestrator.get_models_by_role(ModelRole.BLUE_TEAM)]}")
    print(f"Evaluator team models: {[m for m in orchestrator.get_models_by_role(ModelRole.EVALUATOR)]}")
    
    return orchestrator


if __name__ == "__main__":
    test_model_orchestration()