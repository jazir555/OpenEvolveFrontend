"""
LMQL-DSPy Integration Adapter

This module provides an adapter that allows LMQL (Language Model Query Language) 
to work with DSPy (Declarative Structured Prompt Engineering) for enhanced 
constrained reasoning and programmatic prompting.

Following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs and outputs
- ANTI-HALLUCINATION: Verify data integrity
- READ-ONLY STATE: Don't modify underlying systems' data
- IDEMPOTENCY: Safe to run multiple times
- CONFIGURATION EXPLICITNESS: All parameters configurable
- UTC: All timestamps in UTC
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timezone
import sys
import os

# Add paths to access the required modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge_engine", "integrations"))

from dspy_integration import DSPyIntegration, DSPyResult
from lmql_adapter import LMQLAdapter, Constraint, ConstraintType, ConstraintResult

logger = logging.getLogger(__name__)


class LMQLDSPyAdapter:
    """
    Adapter class that bridges LMQL with DSPy for enhanced constrained reasoning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter with configuration
        
        Args:
            config: Configuration dictionary with optional parameters
        """
        self.config = config or {}
        
        # Initialize logging
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)
        
        # Initialize the DSPy integration
        dspy_config = self.config.get('dspy_config', {})
        self.dspy_integration = DSPyIntegration(config=dspy_config)
        
        # Initialize the LMQL adapter
        lmql_config = self.config.get('lmql_config', {})
        self.lmql_adapter = LMQLAdapter(
            lmql_available=lmql_config.get('lmql_available'),
            fallback_on_error=lmql_config.get('fallback_on_error', True),
            enable_metrics=lmql_config.get('enable_metrics', True),
            default_timeout=lmql_config.get('default_timeout', 30.0)
        )
        
        self.logger.info("LMQL-DSPy Adapter initialized successfully")
    
    async def constrained_chain_of_thought(
        self,
        question: str,
        context: str = "",
        constraints: Optional[List[Constraint]] = None,
        max_steps: int = 5,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute chain of thought reasoning with LMQL constraints.
        
        Args:
            question: Question to reason about
            context: Context information for reasoning
            constraints: List of LMQL constraints to apply
            max_steps: Maximum number of reasoning steps
            correlation_id: Correlation ID for tracking
            
        Returns:
            Dictionary with reasoning results and constraint validation
        """
        self.logger.info(f"Starting constrained chain of thought for question: {question[:50]}...")
        
        start_time = datetime.now(timezone.utc)
        correlation_id = correlation_id or f"lmql_dspy_cot_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            # First, get DSPy chain of thought result
            dspy_result = await self.dspy_integration.chain_of_thought(
                question=question,
                context=context,
                max_steps=max_steps,
                correlation_id=f"{correlation_id}_dspy"
            )
            
            if not dspy_result.success:
                return {
                    'success': False,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': None,
                    'constraint_validation': {'valid': False, 'errors': ['DSPy failed']},
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id,
                    'error': dspy_result.error
                }
            
            # Apply LMQL constraints to the DSPy result
            constraints = constraints or []
            
            if constraints:
                # Use LMQL adapter to validate and potentially regenerate with constraints
                lmql_result = self.lmql_adapter.constrained_generation(
                    prompt=f"Question: {question}\nContext: {context}\n\nReasoning: {dspy_result.reasoning}\n\nAnswer: {dspy_result.output}",
                    constraints=constraints,
                    decoding="argmax",
                    max_tokens=500
                )
                
                # Validate the result against constraints
                valid, errors = self.lmql_adapter._validate_constraints(lmql_result.text, constraints)
                
                result = {
                    'success': valid,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': lmql_result.__dict__ if hasattr(lmql_result, '__dict__') else {
                        'success': lmql_result.success,
                        'text': lmql_result.text,
                        'metadata': lmql_result.metadata,
                        'error': lmql_result.error,
                        'validation_errors': lmql_result.validation_errors,
                        'fallback_used': lmql_result.fallback_used,
                        'generation_time': lmql_result.generation_time
                    },
                    'constraint_validation': {
                        'valid': valid,
                        'errors': errors,
                        'constraints_applied': [c.to_lmql_syntax() for c in constraints]
                    },
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id
                }
            else:
                # No constraints applied
                result = {
                    'success': dspy_result.success,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': None,
                    'constraint_validation': {
                        'valid': True,
                        'errors': [],
                        'constraints_applied': []
                    },
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id
                }
            
            self.logger.info(f"Constrained chain of thought completed successfully: {correlation_id}")
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.error(f"Constrained chain of thought failed: {e}")
            
            return {
                'success': False,
                'dspy_result': None,
                'lmql_result': None,
                'constraint_validation': {'valid': False, 'errors': [str(e)]},
                'processing_time_ms': processing_time_ms,
                'correlation_id': correlation_id,
                'error': str(e)
            }
    
    async def constrained_program_of_thought(
        self,
        question: str,
        context: str = "",
        constraints: Optional[List[Constraint]] = None,
        max_iterations: int = 3,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute program of thought reasoning with LMQL constraints.
        
        Args:
            question: Question to solve using program of thought
            context: Context information for solving
            constraints: List of LMQL constraints to apply
            max_iterations: Maximum number of iterations for code generation
            correlation_id: Correlation ID for tracking
            
        Returns:
            Dictionary with solution results and constraint validation
        """
        self.logger.info(f"Starting constrained program of thought for question: {question[:50]}...")
        
        start_time = datetime.now(timezone.utc)
        correlation_id = correlation_id or f"lmql_dspy_pot_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            # First, get DSPy program of thought result
            dspy_result = await self.dspy_integration.program_of_thought(
                question=question,
                context=context,
                max_iterations=max_iterations,
                correlation_id=f"{correlation_id}_dspy"
            )
            
            if not dspy_result.success:
                return {
                    'success': False,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': None,
                    'constraint_validation': {'valid': False, 'errors': ['DSPy failed']},
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id,
                    'error': dspy_result.error
                }
            
            # Apply LMQL constraints to the DSPy result
            constraints = constraints or []
            
            if constraints:
                # Use LMQL adapter to validate and potentially regenerate with constraints
                lmql_result = self.lmql_adapter.constrained_generation(
                    prompt=f"Question: {question}\nContext: {context}\n\nSolution: {dspy_result.output}",
                    constraints=constraints,
                    decoding="argmax",
                    max_tokens=500
                )
                
                # Validate the result against constraints
                valid, errors = self.lmql_adapter._validate_constraints(lmql_result.text, constraints)
                
                result = {
                    'success': valid,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': lmql_result.__dict__ if hasattr(lmql_result, '__dict__') else {
                        'success': lmql_result.success,
                        'text': lmql_result.text,
                        'metadata': lmql_result.metadata,
                        'error': lmql_result.error,
                        'validation_errors': lmql_result.validation_errors,
                        'fallback_used': lmql_result.fallback_used,
                        'generation_time': lmql_result.generation_time
                    },
                    'constraint_validation': {
                        'valid': valid,
                        'errors': errors,
                        'constraints_applied': [c.to_lmql_syntax() for c in constraints]
                    },
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id
                }
            else:
                # No constraints applied
                result = {
                    'success': dspy_result.success,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': None,
                    'constraint_validation': {
                        'valid': True,
                        'errors': [],
                        'constraints_applied': []
                    },
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id
                }
            
            self.logger.info(f"Constrained program of thought completed successfully: {correlation_id}")
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.error(f"Constrained program of thought failed: {e}")
            
            return {
                'success': False,
                'dspy_result': None,
                'lmql_result': None,
                'constraint_validation': {'valid': False, 'errors': [str(e)]},
                'processing_time_ms': processing_time_ms,
                'correlation_id': correlation_id,
                'error': str(e)
            }
    
    async def constrained_multi_step_reasoning(
        self,
        question: str,
        steps: List[str],
        context: str = "",
        constraints: Optional[List[Constraint]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute multi-step reasoning with LMQL constraints.
        
        Args:
            question: Question to reason about
            steps: List of reasoning steps to execute
            context: Context information
            constraints: List of LMQL constraints to apply
            correlation_id: Correlation ID for tracking
            
        Returns:
            Dictionary with reasoning results and constraint validation
        """
        self.logger.info(f"Starting constrained multi-step reasoning for question: {question[:50]}...")
        
        start_time = datetime.now(timezone.utc)
        correlation_id = correlation_id or f"lmql_dspy_multi_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            # First, get DSPy multi-step reasoning result
            dspy_result = await self.dspy_integration.multi_step_reasoning(
                question=question,
                steps=steps,
                context=context,
                correlation_id=f"{correlation_id}_dspy"
            )
            
            if not dspy_result.success:
                return {
                    'success': False,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': None,
                    'constraint_validation': {'valid': False, 'errors': ['DSPy failed']},
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id,
                    'error': dspy_result.error
                }
            
            # Apply LMQL constraints to the DSPy result
            constraints = constraints or []
            
            if constraints:
                # Use LMQL adapter to validate and potentially regenerate with constraints
                lmql_result = self.lmql_adapter.constrained_generation(
                    prompt=f"Question: {question}\nSteps: {steps}\nContext: {context}\n\nReasoning: {dspy_result.reasoning}\n\nAnswer: {dspy_result.output}",
                    constraints=constraints,
                    decoding="argmax",
                    max_tokens=500
                )
                
                # Validate the result against constraints
                valid, errors = self.lmql_adapter._validate_constraints(lmql_result.text, constraints)
                
                result = {
                    'success': valid,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': lmql_result.__dict__ if hasattr(lmql_result, '__dict__') else {
                        'success': lmql_result.success,
                        'text': lmql_result.text,
                        'metadata': lmql_result.metadata,
                        'error': lmql_result.error,
                        'validation_errors': lmql_result.validation_errors,
                        'fallback_used': lmql_result.fallback_used,
                        'generation_time': lmql_result.generation_time
                    },
                    'constraint_validation': {
                        'valid': valid,
                        'errors': errors,
                        'constraints_applied': [c.to_lmql_syntax() for c in constraints]
                    },
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id
                }
            else:
                # No constraints applied
                result = {
                    'success': dspy_result.success,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': None,
                    'constraint_validation': {
                        'valid': True,
                        'errors': [],
                        'constraints_applied': []
                    },
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id
                }
            
            self.logger.info(f"Constrained multi-step reasoning completed successfully: {correlation_id}")
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.error(f"Constrained multi-step reasoning failed: {e}")
            
            return {
                'success': False,
                'dspy_result': None,
                'lmql_result': None,
                'constraint_validation': {'valid': False, 'errors': [str(e)]},
                'processing_time_ms': processing_time_ms,
                'correlation_id': correlation_id,
                'error': str(e)
            }
    
    async def solve_with_constrained_signature(
        self,
        question: str,
        signature: str,
        context: str = "",
        constraints: Optional[List[Constraint]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem using a DSPy signature with LMQL constraints.
        
        Args:
            question: Question to solve
            signature: DSPy signature string (e.g., "question -> answer")
            context: Context information
            constraints: List of LMQL constraints to apply
            correlation_id: Correlation ID for tracking
            
        Returns:
            Dictionary with solution results and constraint validation
        """
        self.logger.info(f"Starting constrained signature-based solving for question: {question[:50]}...")
        
        start_time = datetime.now(timezone.utc)
        correlation_id = correlation_id or f"lmql_dspy_sig_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            # First, get DSPy signature-based result
            dspy_result = await self.dspy_integration.solve_with_signature(
                question=question,
                signature=signature,
                context=context,
                correlation_id=f"{correlation_id}_dspy"
            )
            
            if not dspy_result.success:
                return {
                    'success': False,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': None,
                    'constraint_validation': {'valid': False, 'errors': ['DSPy failed']},
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id,
                    'error': dspy_result.error
                }
            
            # Apply LMQL constraints to the DSPy result
            constraints = constraints or []
            
            if constraints:
                # Use LMQL adapter to validate and potentially regenerate with constraints
                lmql_result = self.lmql_adapter.constrained_generation(
                    prompt=f"Question: {question}\nSignature: {signature}\nContext: {context}\n\nResult: {dspy_result.output}",
                    constraints=constraints,
                    decoding="argmax",
                    max_tokens=500
                )
                
                # Validate the result against constraints
                valid, errors = self.lmql_adapter._validate_constraints(lmql_result.text, constraints)
                
                result = {
                    'success': valid,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': lmql_result.__dict__ if hasattr(lmql_result, '__dict__') else {
                        'success': lmql_result.success,
                        'text': lmql_result.text,
                        'metadata': lmql_result.metadata,
                        'error': lmql_result.error,
                        'validation_errors': lmql_result.validation_errors,
                        'fallback_used': lmql_result.fallback_used,
                        'generation_time': lmql_result.generation_time
                    },
                    'constraint_validation': {
                        'valid': valid,
                        'errors': errors,
                        'constraints_applied': [c.to_lmql_syntax() for c in constraints]
                    },
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id
                }
            else:
                # No constraints applied
                result = {
                    'success': dspy_result.success,
                    'dspy_result': dspy_result.to_dict(),
                    'lmql_result': None,
                    'constraint_validation': {
                        'valid': True,
                        'errors': [],
                        'constraints_applied': []
                    },
                    'processing_time_ms': (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    'correlation_id': correlation_id
                }
            
            self.logger.info(f"Constrained signature-based solving completed successfully: {correlation_id}")
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.error(f"Constrained signature-based solving failed: {e}")
            
            return {
                'success': False,
                'dspy_result': None,
                'lmql_result': None,
                'constraint_validation': {'valid': False, 'errors': [str(e)]},
                'processing_time_ms': processing_time_ms,
                'correlation_id': correlation_id,
                'error': str(e)
            }
    
    async def batch_solve_with_constraints(
        self,
        questions: List[str],
        signature: str = "question -> answer",
        context: str = "",
        constraints: Optional[List[Constraint]] = None,
        correlation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Solve multiple questions in batch with constraints.
        
        Args:
            questions: List of questions to solve
            signature: DSPy signature for the task
            context: Context information
            constraints: List of LMQL constraints to apply
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of dictionaries with results for each question
        """
        self.logger.info(f"Starting batch solving with constraints for {len(questions)} questions")
        
        start_time = datetime.now(timezone.utc)
        correlation_id = correlation_id or f"lmql_dspy_batch_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            # Process each question with constraints
            results = []
            for i, question in enumerate(questions):
                result = await self.solve_with_constrained_signature(
                    question=question,
                    signature=signature,
                    context=context,
                    constraints=constraints,
                    correlation_id=f"{correlation_id}_q{i}"
                )
                results.append(result)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            self.logger.info(f"Batch solving with constraints completed: {correlation_id}")
            return results
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.error(f"Batch solving with constraints failed: {e}")
            
            # Return error results for all questions
            error_results = []
            for i in range(len(questions)):
                error_results.append({
                    'success': False,
                    'dspy_result': None,
                    'lmql_result': None,
                    'constraint_validation': {'valid': False, 'errors': [str(e)]},
                    'processing_time_ms': processing_time_ms / len(questions) if questions else 0.0,
                    'correlation_id': f"{correlation_id}_q{i}",
                    'error': str(e)
                })
            
            return error_results


def create_unified_interface(adapter: LMQLDSPyAdapter):
    """
    Creates an interface function that allows unified access to both systems
    
    Args:
        adapter: Instance of LMQLDSPyAdapter
        
    Returns:
        Function that provides unified access to both systems
    """
    async def unified_query(query_type: str, **kwargs) -> Dict[str, Any]:
        """
        Interface function for unified access to LMQL-DSPy capabilities
        
        Args:
            query_type: Type of query ('constrained_cot', 'constrained_pot', 'constrained_multi', 'constrained_signature', 'batch_constrained')
            **kwargs: Query-specific parameters
            
        Returns:
            Query results
        """
        if query_type == 'constrained_cot':
            question = kwargs.get('question')
            if not question:
                raise ValueError("question is required for constrained chain of thought queries")
            return await adapter.constrained_chain_of_thought(
                question=question,
                context=kwargs.get('context', ''),
                constraints=kwargs.get('constraints'),
                max_steps=kwargs.get('max_steps', 5),
                correlation_id=kwargs.get('correlation_id')
            )
        
        elif query_type == 'constrained_pot':
            question = kwargs.get('question')
            if not question:
                raise ValueError("question is required for constrained program of thought queries")
            return await adapter.constrained_program_of_thought(
                question=question,
                context=kwargs.get('context', ''),
                constraints=kwargs.get('constraints'),
                max_iterations=kwargs.get('max_iterations', 3),
                correlation_id=kwargs.get('correlation_id')
            )
        
        elif query_type == 'constrained_multi':
            question = kwargs.get('question')
            steps = kwargs.get('steps', [])
            if not question:
                raise ValueError("question is required for constrained multi-step queries")
            if not steps:
                raise ValueError("steps are required for constrained multi-step queries")
            return await adapter.constrained_multi_step_reasoning(
                question=question,
                steps=steps,
                context=kwargs.get('context', ''),
                constraints=kwargs.get('constraints'),
                correlation_id=kwargs.get('correlation_id')
            )
        
        elif query_type == 'constrained_signature':
            question = kwargs.get('question')
            signature = kwargs.get('signature', 'question -> answer')
            if not question:
                raise ValueError("question is required for constrained signature queries")
            return await adapter.solve_with_constrained_signature(
                question=question,
                signature=signature,
                context=kwargs.get('context', ''),
                constraints=kwargs.get('constraints'),
                correlation_id=kwargs.get('correlation_id')
            )
        
        elif query_type == 'batch_constrained':
            questions = kwargs.get('questions', [])
            if not questions:
                raise ValueError("questions are required for batch constrained queries")
            return await adapter.batch_solve_with_constraints(
                questions=questions,
                signature=kwargs.get('signature', 'question -> answer'),
                context=kwargs.get('context', ''),
                constraints=kwargs.get('constraints'),
                correlation_id=kwargs.get('correlation_id')
            )
        
        else:
            raise ValueError(f"Unknown query type: {query_type}")
    
    return unified_query


# Example usage
async def main():
    # Initialize the adapter
    config = {
        'log_level': 'INFO',
        'dspy_config': {
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 4096
        },
        'lmql_config': {
            'fallback_on_error': True,
            'enable_metrics': True
        }
    }
    
    adapter = LMQLDSPyAdapter(config=config)
    
    # Create the unified interface
    unified_interface = create_unified_interface(adapter)
    
    # Example: Constrained chain of thought
    from lmql_adapter import create_list_constraint
    
    boolean_constraint = create_list_constraint("answer", ["yes", "no"])
    
    cot_result = await unified_interface(
        'constrained_cot',
        question="Is the Earth round?",
        constraints=[boolean_constraint]
    )
    print("Constrained chain of thought result:", cot_result)
    
    # Example: Constrained program of thought
    pot_result = await unified_interface(
        'constrained_pot',
        question="What is 15 * 23?",
        constraints=[create_datatype_constraint("answer", "int")]
    )
    print("Constrained program of thought result:", pot_result)
    
    # Example: Batch solving with constraints
    batch_result = await unified_interface(
        'batch_constrained',
        questions=["Is water wet?", "Is fire hot?"],
        constraints=[boolean_constraint]
    )
    print("Batch constrained result:", batch_result)


if __name__ == "__main__":
    asyncio.run(main())