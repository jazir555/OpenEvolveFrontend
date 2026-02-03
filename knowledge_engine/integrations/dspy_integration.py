"""
DSPy Integration for OpenEvolve Knowledge Engine

This module provides integration with the DSPy program-of-thought prompting system,
enabling advanced reasoning and problem-solving capabilities.
"""

# Import aiohttp compatibility shim BEFORE any dspy imports
# This patches aiohttp to be compatible with litellm (used by dspy)
from knowledge_engine.aiohttp_compat import *

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import uuid


logger = logging.getLogger(__name__)


@dataclass
class DSPyResult:
    """Result of a DSPy operation."""
    success: bool
    output: Any
    reasoning: str
    metadata: Dict[str, Any]
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'output': self.output,
            'reasoning': self.reasoning,
            'metadata': self.metadata,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error
        }


class DSPyIntegration:
    """
    Integration with DSPy program-of-thought prompting system.
    
    Provides methods for:
    - Chain of thought reasoning
    - Program of thought execution
    - Multi-step problem solving
    - Reasoning trace extraction
    - Advanced prompting techniques
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DSPy integration.
        
        Args:
            config: Configuration for DSPy components
        """
        self.config = config or self._get_default_config()
        
        # Initialize DSPy components
        self.lm = None
        self.teleprompter = None
        self.modules = {}
        
        # Initialize based on configuration
        self._initialize_components()
        
        logger.info({
            "msg": "DSPyIntegration initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for DSPy integration."""
        return {
            "model": "gpt-4o",
            "api_key": None,
            "api_base": None,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_retries": 3,
            "backoff_factor": 1,
            "teleprompter": {
                "type": "BootstrapFewShot",  # BootstrapFewShot, COPRO, Ensemble, etc.
                "k": 8,  # Number of examples to bootstrap
                "max_bootstrapped_demos": 8,
                "max_labeled_demos": 8
            },
            "cot_config": {
                "max_iters": 3,
                "rationale_field": "reasoning"
            },
            "pot_config": {
                "max_iters": 3
            }
        }
    
    def _initialize_components(self):
        """Initialize DSPy components based on configuration."""
        try:
            # Import DSPy components
            import dspy
            
            # Configure DSPy with the specified model
            model_name = self.config.get("model", "gpt-4o")
            api_key = self.config.get("api_key")
            api_base = self.config.get("api_base")
            
            # Check if we have API credentials - if not, use mock
            if not api_key:
                logger.warning({
                    "msg": "No API key provided for DSPy, using mock implementation",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                self._initialize_mock_components()
                return
            
            # Initialize the language model
            if "openai" in model_name.lower() or "gpt" in model_name.lower():
                from dspy.clients.lm import OpenAILM
                self.lm = OpenAILM(
                    model=model_name,
                    api_key=api_key,
                    api_base=api_base,
                    temperature=self.config.get("temperature", 0.7),
                    max_tokens=self.config.get("max_tokens", 4096),
                    top_p=self.config.get("top_p", 1.0),
                    frequency_penalty=self.config.get("frequency_penalty", 0.0),
                    presence_penalty=self.config.get("presence_penalty", 0.0)
                )
            elif "anthropic" in model_name.lower() or "claude" in model_name.lower():
                from dspy.clients.lm import AnthropicLM
                self.lm = AnthropicLM(
                    model=model_name,
                    api_key=api_key,
                    temperature=self.config.get("temperature", 0.7),
                    max_tokens=self.config.get("max_tokens", 4096)
                )
            else:
                # Default to OpenAI-compatible interface
                from dspy.clients.lm import OpenAILM
                self.lm = OpenAILM(
                    model=model_name,
                    api_key=api_key,
                    api_base=api_base,
                    temperature=self.config.get("temperature", 0.7),
                    max_tokens=self.config.get("max_tokens", 4096)
                )
            
            # Configure DSPy with the language model
            dspy.configure(lm=self.lm)
            
            # Initialize teleprompter based on configuration
            teleprompter_type = self.config.get("teleprompter", {}).get("type", "BootstrapFewShot")
            if teleprompter_type == "BootstrapFewShot":
                from dspy.teleprompt import BootstrapFewShot
                k = self.config.get("teleprompter", {}).get("k", 8)
                self.teleprompter = BootstrapFewShot(
                    metric=lambda x, y, z: True,  # Placeholder metric
                    k=k,
                    max_bootstrapped_demos=self.config.get("teleprompter", {}).get("max_bootstrapped_demos", 8),
                    max_labeled_demos=self.config.get("teleprompter", {}).get("max_labeled_demos", 8)
                )
            else:
                # Default to BootstrapFewShot
                from dspy.teleprompt import BootstrapFewShot
                self.teleprompter = BootstrapFewShot(
                    metric=lambda x, y, z: True,
                    k=8
                )
            
            logger.info({
                "msg": "DSPy components initialized successfully",
                "model": model_name,
                "teleprompter_type": teleprompter_type,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except ImportError:
            logger.warning({
                "msg": "DSPy not available, using mock implementation",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Initialize with mock components
            self._initialize_mock_components()
        except Exception as e:
            logger.warning({
                "msg": f"Failed to initialize DSPy components: {e}, using mock implementation",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Initialize with mock components instead of raising
            self._initialize_mock_components()
    
    def _initialize_mock_components(self):
        """Initialize mock components when DSPy is not available."""
        logger.warning({
            "msg": "DSPy not available - integration will raise errors on use",
            "install": "pip install dspy-ai",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Create failing mock implementations
        from ..optional_imports import create_failing_mock
        
        MockLM = create_failing_mock(
            package_name='dspy-ai',
            feature_name='DSPy language model interface',
            install_command='pip install dspy-ai'
        )
        
        MockTeleprompter = create_failing_mock(
            package_name='dspy-ai',
            feature_name='DSPy teleprompter',
            install_command='pip install dspy-ai'
        )
        
        self.lm = None
        self.teleprompter = None
        self._mock_lm_class = MockLM
        self._mock_teleprompter_class = MockTeleprompter
    
    async def chain_of_thought(
        self,
        question: str,
        context: str = "",
        max_steps: int = 5,
        correlation_id: Optional[str] = None
    ) -> DSPyResult:
        """
        Execute chain of thought reasoning for a given question.
        
        Args:
            question: Question to reason about
            context: Context information for reasoning
            max_steps: Maximum number of reasoning steps
            correlation_id: Correlation ID for tracking
            
        Returns:
            DSPyResult with reasoning trace and answer
        """
        correlation_id = correlation_id or f"dspy_cot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting DSPy chain of thought reasoning",
            "question_length": len(question),
            "context_length": len(context),
            "max_steps": max_steps,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.lm:
                raise RuntimeError("DSPy language model not initialized")
            
            # Import DSPy components
            import dspy
            from dspy.predict.chain_of_thought import ChainOfThought
            
            # Create a simple signature for the task
            class SimpleTask(dspy.Signature):
                """A simple task that requires reasoning."""
                question = dspy.InputField(desc="The question to answer")
                context = dspy.InputField(desc="Context information", default="")
                reasoning = dspy.OutputField(desc="Step-by-step reasoning")
                answer = dspy.OutputField(desc="Final answer")
            
            # Create a Chain of Thought predictor
            cot_predictor = ChainOfThought(SimpleTask, max_iters=max_steps)
            
            # Execute the chain of thought
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: cot_predictor(question=question, context=context)
            )
            
            # Extract results
            reasoning = getattr(result, 'reasoning', 'No reasoning provided')
            answer = getattr(result, 'answer', 'No answer provided')
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            dspy_result = DSPyResult(
                success=True,
                output=answer,
                reasoning=reasoning,
                metadata={
                    "model_used": self.config.get("model", "unknown"),
                    "max_steps": max_steps,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "DSPy chain of thought reasoning completed",
                "correlation_id": correlation_id,
                "reasoning_length": len(reasoning),
                "answer_length": len(answer),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return dspy_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "DSPy chain of thought reasoning failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return DSPyResult(
                success=False,
                output=None,
                reasoning="",
                metadata={
                    "model_used": self.config.get("model", "unknown"),
                    "max_steps": max_steps,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def program_of_thought(
        self,
        question: str,
        context: str = "",
        max_iterations: int = 3,
        correlation_id: Optional[str] = None
    ) -> DSPyResult:
        """
        Execute program of thought reasoning for a given question.
        
        Args:
            question: Question to solve using program of thought
            context: Context information for solving
            max_iterations: Maximum number of iterations for code generation
            correlation_id: Correlation ID for tracking
            
        Returns:
            DSPyResult with solution and reasoning trace
        """
        correlation_id = correlation_id or f"dspy_pot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting DSPy program of thought reasoning",
            "question_length": len(question),
            "context_length": len(context),
            "max_iterations": max_iterations,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.lm:
                raise RuntimeError("DSPy language model not initialized")
            
            # Import DSPy components
            import dspy
            from dspy.predict.program_of_thought import ProgramOfThought
            
            # Create a simple signature for the task
            class MathTask(dspy.Signature):
                """A math task that can be solved with code."""
                question = dspy.InputField(desc="The math question to solve")
                context = dspy.InputField(desc="Context information", default="")
                answer = dspy.OutputField(desc="Final answer")
            
            # Create a Program of Thought predictor
            pot_predictor = ProgramOfThought(
                signature=MathTask,
                max_iters=max_iterations
            )
            
            # Execute the program of thought
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pot_predictor(question=question, context=context)
            )
            
            # Extract results
            answer = getattr(result, 'answer', 'No answer provided')
            reasoning = getattr(result, 'reasoning', 'No reasoning provided')
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            dspy_result = DSPyResult(
                success=True,
                output=answer,
                reasoning=reasoning,
                metadata={
                    "model_used": self.config.get("model", "unknown"),
                    "max_iterations": max_iterations,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "DSPy program of thought reasoning completed",
                "correlation_id": correlation_id,
                "answer_length": len(answer),
                "reasoning_length": len(reasoning),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return dspy_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "DSPy program of thought reasoning failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return DSPyResult(
                success=False,
                output=None,
                reasoning="",
                metadata={
                    "model_used": self.config.get("model", "unknown"),
                    "max_iterations": max_iterations,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def multi_step_reasoning(
        self,
        question: str,
        steps: List[str],
        context: str = "",
        correlation_id: Optional[str] = None
    ) -> DSPyResult:
        """
        Execute multi-step reasoning with specified steps.
        
        Args:
            question: Question to reason about
            steps: List of reasoning steps to execute
            context: Context information
            correlation_id: Correlation ID for tracking
            
        Returns:
            DSPyResult with final answer and reasoning trace
        """
        correlation_id = correlation_id or f"dspy_multi_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting DSPy multi-step reasoning",
            "question_length": len(question),
            "steps_count": len(steps),
            "context_length": len(context),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.lm:
                raise RuntimeError("DSPy language model not initialized")
            
            # Import DSPy components
            import dspy
            from dspy.predict.chain_of_thought import ChainOfThought
            
            # Create a signature for multi-step reasoning
            class MultiStepTask(dspy.Signature):
                """A task that requires multi-step reasoning."""
                question = dspy.InputField(desc="The question to answer")
                steps = dspy.InputField(desc="Steps to follow for reasoning")
                context = dspy.InputField(desc="Context information", default="")
                intermediate_reasoning = dspy.OutputField(desc="Reasoning for each step")
                final_answer = dspy.OutputField(desc="Final answer after all steps")
            
            # Create a Chain of Thought predictor for multi-step reasoning
            cot_predictor = ChainOfThought(MultiStepTask)
            
            # Execute the multi-step reasoning
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: cot_predictor(
                    question=question,
                    steps=str(steps),
                    context=context
                )
            )
            
            # Extract results
            intermediate_reasoning = getattr(result, 'intermediate_reasoning', 'No intermediate reasoning provided')
            final_answer = getattr(result, 'final_answer', 'No final answer provided')
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            dspy_result = DSPyResult(
                success=True,
                output=final_answer,
                reasoning=intermediate_reasoning,
                metadata={
                    "model_used": self.config.get("model", "unknown"),
                    "steps_count": len(steps),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "DSPy multi-step reasoning completed",
                "correlation_id": correlation_id,
                "intermediate_reasoning_length": len(intermediate_reasoning),
                "final_answer_length": len(final_answer),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return dspy_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "DSPy multi-step reasoning failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return DSPyResult(
                success=False,
                output=None,
                reasoning="",
                metadata={
                    "model_used": self.config.get("model", "unknown"),
                    "steps_count": len(steps),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def solve_with_signature(
        self,
        question: str,
        signature: str,
        context: str = "",
        correlation_id: Optional[str] = None
    ) -> DSPyResult:
        """
        Solve a problem using a custom DSPy signature.
        
        Args:
            question: Question to solve
            signature: DSPy signature string (e.g., "question -> answer")
            context: Context information
            correlation_id: Correlation ID for tracking
            
        Returns:
            DSPyResult with solution
        """
        correlation_id = correlation_id or f"dspy_sig_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting DSPy signature-based solving",
            "question_length": len(question),
            "signature": signature,
            "context_length": len(context),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.lm:
                raise RuntimeError("DSPy language model not initialized")
            
            # Import DSPy components
            import dspy
            from dspy.predict.predict import Predict
            
            # Create predictor with the given signature
            predictor = Predict(signature)
            
            # Parse the signature to determine input/output fields
            parts = signature.split(" -> ")
            if len(parts) != 2:
                raise ValueError("Signature must be in format 'input_field -> output_field'")
            
            input_field = parts[0].strip()
            output_field = parts[1].strip()
            
            # Prepare input arguments
            input_args = {input_field: question}
            if context:
                input_args['context'] = context
            
            # Execute the prediction
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: predictor(**input_args)
            )
            
            # Extract the output
            output = getattr(result, output_field, None)
            if output is None:
                # Try to get the output in a different way
                output = str(result)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            dspy_result = DSPyResult(
                success=True,
                output=output,
                reasoning="Signature-based prediction",
                metadata={
                    "model_used": self.config.get("model", "unknown"),
                    "signature": signature,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "DSPy signature-based solving completed",
                "correlation_id": correlation_id,
                "output_length": len(str(output)) if output else 0,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return dspy_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "DSPy signature-based solving failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return DSPyResult(
                success=False,
                output=None,
                reasoning="",
                metadata={
                    "model_used": self.config.get("model", "unknown"),
                    "signature": signature,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def optimize_module(
        self,
        module,
        trainset: List[Dict[str, Any]],
        correlation_id: Optional[str] = None
    ) -> Any:
        """
        Optimize a DSPy module using the configured teleprompter.
        
        Args:
            module: DSPy module to optimize
            trainset: Training set for optimization
            correlation_id: Correlation ID for tracking
            
        Returns:
            Optimized module
        """
        correlation_id = correlation_id or f"dspy_opt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting DSPy module optimization",
            "trainset_size": len(trainset),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.teleprompter:
                raise RuntimeError("DSPy teleprompter not initialized")
            
            # Convert trainset to DSPy examples
            import dspy
            train_examples = []
            for item in trainset:
                example = dspy.Example(**item).with_inputs(*list(item.keys()))
                train_examples.append(example)
            
            # Compile the module with the teleprompter
            optimized_module = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.teleprompter.compile(module, trainset=train_examples)
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "DSPy module optimization completed",
                "correlation_id": correlation_id,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return optimized_module
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "DSPy module optimization failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            raise
    
    async def batch_solve(
        self,
        questions: List[str],
        signature: str = "question -> answer",
        context: str = "",
        correlation_id: Optional[str] = None
    ) -> List[DSPyResult]:
        """
        Solve multiple questions in batch.
        
        Args:
            questions: List of questions to solve
            signature: DSPy signature for the task
            context: Context information
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of DSPyResult objects
        """
        correlation_id = correlation_id or f"dspy_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting DSPy batch solving",
            "questions_count": len(questions),
            "signature": signature,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Process each question in parallel
            tasks = [
                self.solve_with_signature(
                    question=q,
                    signature=signature,
                    context=context,
                    correlation_id=f"{correlation_id}_q_{i}"
                )
                for i, q in enumerate(questions)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in the results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error({
                        "msg": f"Batch item {i} solving failed",
                        "correlation_id": f"{correlation_id}_q_{i}",
                        "error": str(result)
                    })
                    processed_results.append(DSPyResult(
                        success=False,
                        output=None,
                        reasoning="",
                        metadata={"batch_index": i, "error": str(result)},
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            successful_count = sum(1 for r in processed_results if r.success)
            
            logger.info({
                "msg": "DSPy batch solving completed",
                "correlation_id": correlation_id,
                "questions_count": len(questions),
                "successful_count": successful_count,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return processed_results
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "DSPy batch solving failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Return error results for all questions
            error_results = []
            for i in range(len(questions)):
                error_results.append(DSPyResult(
                    success=False,
                    output=None,
                    reasoning="",
                    metadata={"batch_index": i, "error": str(e)},
                    processing_time_ms=processing_time_ms / len(questions) if questions else 0.0,
                    error=str(e)
                ))
            
            return error_results
    
    def get_dspy_status(self) -> Dict[str, Any]:
        """
        Get the status of the DSPy integration.
        
        Returns:
            Dictionary with integration status
        """
        return {
            "available": self.lm is not None,
            "model": self.config.get("model", "unknown"),
            "teleprompter_available": self.teleprompter is not None,
            "initialized": self.lm is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def close(self):
        """Close resources used by the integration."""
        logger.info({
            "msg": "Closing DSPy integration resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # If we have a language model with a close method, close it
        if self.lm and hasattr(self.lm, 'close'):
            try:
                if asyncio.iscoroutinefunction(self.lm.close):
                    await self.lm.close()
                else:
                    self.lm.close()
            except Exception as e:
                logger.error(f"Error closing language model: {e}")
        
        logger.info({
            "msg": "DSPy integration resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })