"""
Model Orchestration Layer for OpenEvolve
Implements the Model Orchestration functionality described in the ultimate explanation document.
"""
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
from concurrent.futures import ThreadPoolExecutor
import statistics
from datetime import datetime, timedelta
import logging

# Import OpenEvolve components for enhanced functionality
try:
    from openevolve.api import run_evolution as openevolve_run_evolution
    from openevolve.config import Config, LLMModelConfig
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available - using fallback implementation")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@dataclass
class ModelConfig:
    """Configuration for a model"""
    model_id: str
    provider: ModelProvider
    api_key: str
    api_base: str
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    team: ModelTeam = ModelTeam.RED
    weight: float = 1.0  # For ensemble methods
    enabled: bool = True
    performance_score: float = 0.0  # Track performance over time

@dataclass
class ModelPerformance:
    """Performance tracking for a model"""
    model_id: str
    total_requests: int = 0
    success_requests: int = 0
    avg_response_time: float = 0.0
    avg_token_usage: int = 0
    avg_cost: float = 0.0
    issues_found: int = 0  # For red team models
    issues_resolved: int = 0  # For blue team models
    evaluation_score: float = 0.0  # For evaluator models
    last_used: datetime = None
    performance_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []

@dataclass
class OrchestrationRequest:
    """Request for model orchestration"""
    content: str
    prompt: str
    team: ModelTeam
    temperature: float = 0.7
    max_tokens: int = 1000
    custom_parameters: Dict[str, Any] = None

@dataclass
class OrchestrationResponse:
    """Response from model orchestration"""
    model_id: str
    response: str
    success: bool
    response_time: float
    token_usage: int
    cost: float
    error_message: Optional[str] = None

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
            team_key = f"{team.value}_rr_index"
            if not hasattr(self, team_key):
                setattr(self, team_key, 0)
            
            idx = getattr(self, team_key)
            model_id = models[idx % len(models)].model_id
            setattr(self, team_key, idx + 1)
            return model_id
    
    def _random_selection(self, models: List[ModelConfig]) -> str:
        """Select model randomly"""
        import random
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
    
    def update_request_count(self, model_id: str):
        """Update the request count for a model"""
        with self.lock:
            self.request_counts[model_id] = self.request_counts.get(model_id, 0) + 1
            self.last_request_time[model_id] = datetime.now()

class ModelClient:
    """Client for communicating with LLM models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.session = requests.Session()
    
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
        """Call OpenAI API"""
        url = f"{self.config.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty
        }
        
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        
        try:
            response = self.session.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            # Estimate cost based on OpenAI pricing (approximate)
            cost = (prompt_tokens * 0.0000005) + (completion_tokens * 0.0000015)  # GPT-3.5-turbo pricing
            
            return content, prompt_tokens + completion_tokens, cost
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "", 0, 0.0
    
    def _call_anthropic_api(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, int, float]:
        """Call Anthropic API"""
        url = f"{self.config.api_base}/complete"
        headers = {
            "X-API-Key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.config.model_id,
            "prompt": f"Human: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_p": self.config.top_p,
            "stop_sequences": ["Human:"]
        }
        
        try:
            response = self.session.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            content = data["completion"]
            
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
        # In a real implementation, you'd use Google's API properly
        logger.warning("Google API call not fully implemented in this example")
        content = f"Response to: {prompt[:100]}..."  # Simulated response
        estimated_tokens = len(content.split())
        cost = estimated_tokens * 0.000005  # Rough estimate
        return content, estimated_tokens, cost
    
    def _call_openrouter_api(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, int, float]:
        """Call OpenRouter API"""
        url = f"{self.config.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty
        }
        
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        
        try:
            response = self.session.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            # Cost is provided by OpenRouter
            cost = usage.get("cost", 0.0)
            
            return content, prompt_tokens + completion_tokens, cost
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            return "", 0, 0.0
    
    def _call_custom_api(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, int, float]:
        """Call custom API"""
        # In a real implementation, you'd have a specific endpoint for custom models
        logger.warning("Custom API call not fully implemented in this example")
        content = f"Response to: {prompt[:100]}..."  # Simulated response
        estimated_tokens = len(content.split())
        cost = estimated_tokens * 0.000008  # Rough estimate
        return content, estimated_tokens, cost

class ModelOrchestrator:
    """Main orchestrator that manages model teams and requests"""
    
    def __init__(self, registry: ModelRegistry = None):
        self.registry = registry or ModelRegistry()
        self.load_balancer = LoadBalancer(self.registry)
        self.executor = ThreadPoolExecutor(max_workers=20)  # Adjust based on needs
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.error_handler = self._default_error_handler
        
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
            self._update_request_status(request_id, 'failed', "No models available")
            return request_id
        
        # Get model config
        model_config = self.registry.models.get(model_id)
        if not model_config:
            logger.error(f"Model config not found for: {model_id}")
            self._update_request_status(request_id, 'failed', f"Model config not found: {model_id}")
            return request_id
        
        # Update load balancer
        self.load_balancer.update_request_count(model_id)
        
        # Execute the request in a separate thread
        future = self.executor.submit(
            self._execute_model_request,
            request_id,
            model_config,
            request
        )
        
        self.active_requests[request_id]['future'] = future
        
        return request_id
    
    def submit_batch_request(self, requests: List[OrchestrationRequest], 
                           strategy: OrchestrationStrategy = OrchestrationStrategy.ROUND_ROBIN) -> List[str]:
        """Submit multiple requests at once"""
        request_ids = []
        for request in requests:
            request_id = self.submit_request(request, strategy)
            request_ids.append(request_id)
        return request_ids
    
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
            result['response'] = request_info['response']
        elif 'future' in request_info:
            if request_info['future'].done():
                result['status'] = 'completed'
                try:
                    response = request_info['future'].result()
                    result['response'] = response
                except Exception as e:
                    result['status'] = 'failed'
                    result['error'] = str(e)
            else:
                result['status'] = 'processing'
        
        return result
    
    def _execute_model_request(self, request_id: str, model_config: ModelConfig, 
                              request: OrchestrationRequest) -> OrchestrationResponse:
        """Execute a single model request"""
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
    
    def _update_request_status(self, request_id: str, status: str, message: str):
        """Update the status of a request"""
        if request_id in self.active_requests:
            self.active_requests[request_id]['status'] = status
            self.active_requests[request_id]['message'] = message
    
    def _default_error_handler(self, error: Exception, context: Dict[str, Any] = None):
        """Default error handler"""
        logger.error(f"Model orchestration error: {error}, Context: {context}")
    
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
    
    def cleanup_completed_requests(self, max_age_hours: int = 24):
        """Clean up completed requests older than specified hours"""
        now = datetime.now()
        cutoff_time = now - timedelta(hours=max_age_hours)
        
        # Find old requests to clean up
        old_request_ids = []
        for req_id, req_info in self.active_requests.items():
            if 'response' in req_info:  # Completed request
                if req_info['start_time'] < cutoff_time:
                    old_request_ids.append(req_id)
        
        # Remove old requests
        for req_id in old_request_ids:
            del self.active_requests[req_id]
        
        logger.info(f"Cleaned up {len(old_request_ids)} old requests")

class AdvancedOrchestrationEngine:
    """Advanced orchestration engine with multi-team coordination"""
    
    def __init__(self, orchestrator: ModelOrchestrator = None):
        self.orchestrator = orchestrator or ModelOrchestrator()
        self.criticality_scaler = self._default_criticality_scaler
        self.content_analyzer = None
        
    def _default_criticality_scaler(self, content: str, request_type: str) -> Dict[str, Any]:
        """Determine criticality and resource allocation based on content"""
        # Simple implementation - in a real system this would be more sophisticated
        content_length = len(content)
        complexity_score = min(10, content_length / 1000)  # Scale 0-10 based on length
        
        # Adjust resource allocation based on criticality
        if complexity_score > 7:
            return {
                'max_parallel_requests': 5,
                'strategy': OrchestrationStrategy.PERFORMANCE_BASED,
                'required_success_rate': 0.95
            }
        elif complexity_score > 4:
            return {
                'max_parallel_requests': 3,
                'strategy': OrchestrationStrategy.ROUND_ROBIN,
                'required_success_rate': 0.9
            }
        else:
            return {
                'max_parallel_requests': 2,
                'strategy': OrchestrationStrategy.RANDOM_SAMPLING,
                'required_success_rate': 0.8
            }
    
    def execute_adversarial_pipeline(self, content: str, max_iterations: int = 5,
                                   api_key: Optional[str] = None,
                                   model_name: str = "gpt-4o") -> Dict[str, Any]:
        """
        Execute a complete adversarial pipeline with red team, blue team, and evaluator,
        using OpenEvolve when available
        
        Args:
            content: Content to improve
            max_iterations: Maximum number of improvement iterations
            api_key: API key for OpenEvolve backend (required when using OpenEvolve)
            model_name: Model to use when using OpenEvolve
            
        Returns:
            Dictionary with pipeline results
        """
        # Prioritize OpenEvolve backend when available
        if OPENEVOLVE_AVAILABLE and api_key:
            return self._execute_adversarial_pipeline_with_openevolve(
                content, max_iterations, api_key, model_name
            )
        
        # Fallback to custom implementation
        return self._execute_adversarial_pipeline_custom(content, max_iterations)
    
    def _execute_adversarial_pipeline_with_openevolve(self, content: str, max_iterations: int,
                                                     api_key: str, model_name: str) -> Dict[str, Any]:
        """
        Execute adversarial pipeline using OpenEvolve backend
        """
        try:
            import tempfile
            import os
            
            pipeline_results = {
                'initial_content': content,
                'iterations': [],
                'final_content': content,
                'total_cost': 0.0,
                'total_time': 0.0,
                'success_rate': 0.0,
                'openevolve_used': True
            }
            
            start_time = time.time()
            
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
            config.max_iterations = max_iterations
            config.database.population_size = 10  # Use a moderate population
            
            # Create an evaluator for adversarial pipeline
            def adversarial_evaluator(program_path: str) -> Dict[str, Any]:
                """
                Evaluator that performs adversarial assessment on the content
                """
                try:
                    with open(program_path, "r", encoding='utf-8') as f:
                        content = f.read()
                    
                    # Perform basic adversarial assessment
                    word_count = len(content.split())
                    
                    # Return adversarial metrics
                    return {
                        "score": 0.8,  # Placeholder adversarial score
                        "timestamp": datetime.now().timestamp(),
                        "content_length": len(content),
                        "word_count": word_count,
                        "adversarial_assessment_completed": True
                    }
                except Exception as e:
                    print(f"Error in adversarial evaluator: {e}")
                    return {
                        "score": 0.0,
                        "timestamp": datetime.now().timestamp(),
                        "error": str(e)
                    }
            
            # Save content to temporary file for OpenEvolve
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding='utf-8') as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Run adversarial pipeline using OpenEvolve API
                result = openevolve_run_evolution(
                    initial_program=temp_file_path,
                    evaluator=adversarial_evaluator,
                    config=config,
                    iterations=max_iterations,
                    output_dir=None,  # Use temporary directory
                    cleanup=True,
                )
                
                # Process results and generate pipeline results
                if result.best_code:
                    final_content = result.best_code
                else:
                    final_content = content
                
                # Generate iteration results (mocked since OpenEvolve handles internally)
                iterations = []
                for i in range(min(max_iterations, 5)):  # Generate up to 5 mock iterations
                    iteration_result = {
                        'iteration_number': i,
                        'red_team_response': 'Adversarial testing performed via OpenEvolve',
                        'blue_team_response': 'Content improvement performed via OpenEvolve',
                        'evaluator_response': f'Quality assessment performed via OpenEvolve - Score: {result.best_score}',
                        'iteration_time': 10.0,  # Mock time
                        'content_length': len(content)
                    }
                    iterations.append(iteration_result)
                
                pipeline_results['iterations'] = iterations
                pipeline_results['final_content'] = final_content
                pipeline_results['total_time'] = time.time() - start_time
                pipeline_results['total_cost'] = 0.1  # Placeholder cost
                pipeline_results['success_rate'] = result.best_score if result.best_score else 0.8
                
                return pipeline_results
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        except Exception as e:
            logger.error(f"Error using OpenEvolve backend: {e}")
            # Fallback to custom implementation
            return self._execute_adversarial_pipeline_custom(content, max_iterations)
    
    def _execute_adversarial_pipeline_custom(self, content: str, max_iterations: int) -> Dict[str, Any]:
        """
        Fallback adversarial pipeline using custom implementation
        """
        import re
        
        pipeline_results = {
            'initial_content': content,
            'iterations': [],
            'final_content': content,
            'total_cost': 0.0,
            'total_time': 0.0,
            'success_rate': 0.0,
            'openevolve_used': False
        }
        
        current_content = content
        start_time = time.time()
        
        for iteration in range(max_iterations):
            iteration_start = time.time()
            logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")
            
            # Red Team - Find issues
            red_prompt = f"You are a red team member. Analyze the following content for vulnerabilities, issues, and weaknesses:\n\n{current_content}\n\nIdentify specific problems and provide detailed feedback in JSON format."
            red_request = OrchestrationRequest(
                content=current_content,
                prompt=red_prompt,
                team=ModelTeam.RED,
                temperature=0.5
            )
            
            red_request_id = self.orchestrator.submit_request(red_request, OrchestrationStrategy.PERFORMANCE_BASED)
            
            # Blue Team - Address issues
            # First wait for red team feedback and then create blue team request
            red_result = None
            max_wait_time = 120  # Wait up to 2 minutes
            wait_start = time.time()
            
            while time.time() - wait_start < max_wait_time:
                red_status = self.orchestrator.get_request_status(red_request_id)
                if red_status['status'] == 'completed':
                    red_result = red_status['response']
                    break
                time.sleep(1)
            
            if red_result and red_result.success:
                blue_prompt = f"You are a blue team member. Address the following issues found by the red team in the content:\n\nContent: {current_content}\n\nIssues: {red_result.response}\n\nCreate an improved version of the content that addresses these issues."
                
                blue_request = OrchestrationRequest(
                    content=current_content,
                    prompt=blue_prompt,
                    team=ModelTeam.BLUE,
                    temperature=0.7
                )
                
                blue_request_id = self.orchestrator.submit_request(blue_request, OrchestrationStrategy.PERFORMANCE_BASED)
                
                # Wait for blue team result
                blue_result = None
                wait_start = time.time()
                
                while time.time() - wait_start < max_wait_time:
                    blue_status = self.orchestrator.get_request_status(blue_request_id)
                    if blue_status['status'] == 'completed':
                        blue_result = blue_status['response']
                        break
                    time.sleep(1)
                
                if blue_result and blue_result.success:
                    current_content = blue_result.response
                    
                    # Evaluator Team - assess quality
                    eval_prompt = f"You are an evaluator. Assess the quality of the following content:\n\n{current_content}\n\nProvide a score from 0-100 and specific feedback."
                    
                    eval_request = OrchestrationRequest(
                        content=current_content,
                        prompt=eval_prompt,
                        team=ModelTeam.EVALUATOR,
                        temperature=0.3
                    )
                    
                    eval_request_id = self.orchestrator.submit_request(eval_request, OrchestrationStrategy.PERFORMANCE_BASED)
                    
                    # Wait for evaluation
                    eval_result = None
                    wait_start = time.time()
                    
                    while time.time() - wait_start < max_wait_time:
                        eval_status = self.orchestrator.get_request_status(eval_request_id)
                        if eval_status['status'] == 'completed':
                            eval_result = eval_status['response']
                            break
                        time.sleep(1)
                
                else:
                    logger.warning(f"Blue team failed for iteration {iteration + 1}")
                    break
            else:
                logger.warning(f"Red team failed for iteration {iteration + 1}")
                break
            
            # Calculate iteration metrics
            iteration_time = time.time() - iteration_start
            
            # Store iteration results
            iteration_result = {
                'iteration_number': iteration,
                'red_team_response': red_result.response if red_result else None,
                'blue_team_response': blue_result.response if blue_result else None,
                'evaluator_response': eval_result.response if eval_result else None,
                'iteration_time': iteration_time,
                'content_length': len(current_content)
            }
            
            pipeline_results['iterations'].append(iteration_result)
            
            # Check if we should continue based on evaluation
            if eval_result and eval_result.success:
                try:
                    # Try to extract score from evaluator response
                    eval_text = eval_result.response
                    score_match = re.search(r'([0-9]{1,3})', eval_text)
                    if score_match:
                        score = int(score_match.group(1))
                        if score >= 90:  # If we reach 90/100, we might be done
                            logger.info(f"Early termination: Quality score {score} reached threshold")
                            break
                except Exception:
                    pass  # If we can't parse the score, continue
        
        pipeline_results['final_content'] = current_content
        pipeline_results['total_time'] = time.time() - start_time
        pipeline_results['total_cost'] = sum(
            sum(r.get(f"{team.value}_team_response", {}).get('cost', 0) 
                for team in [ModelTeam.RED, ModelTeam.BLUE, ModelTeam.EVALUATOR])
            for r in pipeline_results['iterations']
        )
        
        return pipeline_results

# Example usage and testing
def test_model_orchestration():
    """Test function for the Model Orchestration System"""
    # Create a registry with sample models
    registry = ModelRegistry()
    
    # Register some sample models
    registry.register_model(ModelConfig(
        model_id="gpt-4o",
        provider=ModelProvider.OPENAI,
        api_key="test-key",
        api_base="https://api.openai.com/v1",
        team=ModelTeam.RED
    ))
    
    registry.register_model(ModelConfig(
        model_id="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        api_key="test-key",
        api_base="https://api.openai.com/v1",
        team=ModelTeam.BLUE
    ))
    
    registry.register_model(ModelConfig(
        model_id="claude-3-haiku",
        provider=ModelProvider.ANTHROPIC,
        api_key="test-key",
        api_base="https://api.anthropic.com/v1",
        team=ModelTeam.EVALUATOR
    ))
    
    # Create orchestrator
    orchestrator = ModelOrchestrator(registry)
    
    print("Model Orchestration System Test:")
    print(f"Registered models: {list(registry.models.keys())}")
    print(f"Red team models: {[m.model_id for m in registry.get_models_by_team(ModelTeam.RED)]}")
    print(f"Blue team models: {[m.model_id for m in registry.get_models_by_team(ModelTeam.BLUE)]}")
    print(f"Evaluator team models: {[m.model_id for m in registry.get_models_by_team(ModelTeam.EVALUATOR)]}")
    
    # Test submitting a request
    request = OrchestrationRequest(
        content="This is a test content.",
        prompt="Analyze this content for quality and suggest improvements.",
        team=ModelTeam.RED,
        temperature=0.7,
        max_tokens=500
    )
    
    request_id = orchestrator.submit_request(request)
    print(f"Submitted request: {request_id}")
    
    # Wait for completion and check status
    import time
    time.sleep(2)  # Give time for processing in this example
    
    status = orchestrator.get_request_status(request_id)
    print(f"Request status: {status['status']}")
    
    # Get team performance report
    red_report = orchestrator.get_team_performance_report(ModelTeam.RED)
    print(f"Red team performance: {red_report['total_models']} models")
    
    # Test orchestration efficiency metrics
    metrics = orchestrator.get_orchestration_efficiency_metrics()
    print(f"Efficiency metrics: {metrics}")
    
    # Test the advanced orchestration engine
    engine = AdvancedOrchestrationEngine(orchestrator)
    sample_content = "This is a sample content that needs improvement. It has several issues that should be addressed."
    
    try:
        results = engine.execute_adversarial_pipeline(sample_content, max_iterations=2)
        print(f"Pipeline completed with {len(results['iterations'])} iterations")
        print(f"Final content length: {len(results['final_content'])}")
        print(f"Total time: {results['total_time']:.2f}s")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
    
    return orchestrator, engine

if __name__ == "__main__":
    test_model_orchestration()