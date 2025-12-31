"""
Comprehensive Logging Utility for OpenEvolve
This module provides centralized logging functionality for all OpenEvolve operations
"""

import logging
from typing import Dict, Any, Optional
import json
import functools
import time


class OpenEvolveLogger:
    """Centralized logging utility for OpenEvolve."""
    
    def __init__(self, log_file: str = "openevolve.log", log_level: int = logging.INFO):
        """Initialize the logger."""
        self.logger = logging.getLogger("OpenEvolve")
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log an error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)
    
    def log_evolution_start(self, parameters: Dict[str, Any]):
        """Log the start of an evolution process."""
        message = f"Starting evolution with parameters: {json.dumps(parameters, indent=2)}"
        self.info(message)
    
    def log_evolution_progress(self, generation: int, population_size: int, best_fitness: float, avg_fitness: float):
        """Log evolution progress."""
        message = (f"Generation {generation} completed | "
                  f"Population: {population_size} | "
                  f"Best Fitness: {best_fitness:.4f} | "
                  f"Avg Fitness: {avg_fitness:.4f}")
        self.info(message)
    
    def log_evolution_complete(self, best_score: float, generations: int, total_time: float):
        """Log completion of evolution."""
        message = (f"Evolution completed | "
                  f"Best Score: {best_score:.4f} | "
                  f"Generations: {generations} | "
                  f"Total Time: {total_time:.2f}s")
        self.info(message)
    
    def log_adversarial_start(self, parameters: Dict[str, Any]):
        """Log the start of adversarial testing."""
        message = f"Starting adversarial testing with parameters: {json.dumps(parameters, indent=2)}"
        self.info(message)
    
    def log_adversarial_progress(self, iteration: int, approval_rate: float, issues_found: int):
        """Log adversarial testing progress."""
        message = (f"Adversarial iteration {iteration} completed | "
                  f"Approval Rate: {approval_rate:.2%} | "
                  f"Issues Found: {issues_found}")
        self.info(message)
    
    def log_adversarial_complete(self, final_approval_rate: float, iterations: int, total_time: float):
        """Log completion of adversarial testing."""
        message = (f"Adversarial testing completed | "
                  f"Final Approval Rate: {final_approval_rate:.2%} | "
                  f"Iterations: {iterations} | "
                  f"Total Time: {total_time:.2f}s")
        self.info(message)
    
    def log_api_call(self, model: str, tokens_used: int, response_time: float, success: bool = True):
        """Log API call information."""
        status = "SUCCESS" if success else "FAILED"
        message = (f"API Call [{status}] | "
                  f"Model: {model} | "
                  f"Tokens: {tokens_used} | "
                  f"Response Time: {response_time:.2f}s")
        if success:
            self.info(message)
        else:
            self.error(message)
    
    def log_ensemble_decision(self, model_weights: Dict[str, float], selected_model: str):
        """Log ensemble model decision."""
        message = (f"Ensemble Decision | "
                  f"Selected Model: {selected_model} | "
                  f"Weights: {json.dumps(model_weights)}")
        self.info(message)
    
    def log_error_exception(self, error_type: str, error_message: str, stack_trace: Optional[str] = None):
        """Log exception information."""
        message = f"EXCEPTION [{error_type}]: {error_message}"
        self.error(message)
        if stack_trace:
            self.error(f"Stack Trace: {stack_trace}")


# Global logger instance
logger = OpenEvolveLogger()


def log_function_call(func):
    """Decorator to log function calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        
        # Log function entry
        logger.info(f"Calling function: {func_name}")
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Log function exit
            logger.info(f"Function {func_name} completed successfully in {execution_time:.2f}s")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Log function error
            logger.error(f"Function {func_name} failed after {execution_time:.2f}s: {str(e)}")
            raise
    
    return wrapper


def log_evolution_operation(operation_name: str):
    """Decorator to log evolution operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {operation_name}")
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.info(f"{operation_name} completed successfully in {execution_time:.2f}s")
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.error(f"{operation_name} failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


def _update_adv_log_and_status(message: str) -> None:
    """Update adversarial log and status message in a thread-safe manner."""
    import streamlit as st
    with st.session_state.thread_lock:
        if "adversarial_log" not in st.session_state:
            st.session_state.adversarial_log = []
        st.session_state.adversarial_log.append(message)
        st.session_state.adversarial_status_message = message


# Test the logger
if __name__ == "__main__":
    # Test the logger
    logger.info("Testing OpenEvolve logger")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
    
    # Test evolution logging
    logger.log_evolution_start({"population_size": 100, "generations": 50})
    logger.log_evolution_progress(10, 100, 0.85, 0.72)
    logger.log_evolution_complete(0.92, 50, 120.5)
    
    # Test adversarial logging
    logger.log_adversarial_start({"models": ["gpt-4", "claude-3"], "iterations": 10})
    logger.log_adversarial_progress(5, 0.87, 3)
    logger.log_adversarial_complete(0.95, 10, 45.2)
    
    # Test API logging
    logger.log_api_call("gpt-4", 1250, 1.34, True)
    logger.log_api_call("claude-3", 890, 2.12, False)
    
    print("Logger test completed successfully!")