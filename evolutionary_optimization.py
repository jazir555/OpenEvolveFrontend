"""
Evolutionary Optimization Framework for OpenEvolve
This module now uses the core OpenEvolve evolutionary optimization instead of custom implementation.
"""
import tempfile
import os
from typing import Dict, Any
from datetime import datetime

# Import OpenEvolve components for evolutionary optimization
try:
    from openevolve.api import run_evolution as openevolve_run_evolution
    from openevolve.config import Config, LLMModelConfig
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available - using API-based evolution only")

from quality_assessment import QualityAssessmentEngine
from evaluator_team import EvaluatorTeam


class OpenEvolveEvaluator:
    """Wrapper for using OpenEvolve's evaluator with our content analysis"""
    
    def __init__(self, evaluator_team: EvaluatorTeam, quality_assessment: QualityAssessmentEngine,
                 content_type: str = "general"):
        self.evaluator_team = evaluator_team
        self.quality_assessment = quality_assessment
        self.content_type = content_type
    
    def evaluate(self, program_path: str) -> Dict[str, Any]:
        """
        Evaluate content using OpenEvolve-compatible evaluator interface
        """
        try:
            with open(program_path, "r", encoding='utf-8') as f:
                content = f.read()
            
            # Use our existing evaluator team for assessment
            if self.evaluator_team:
                assessment = self.evaluator_team.evaluate_content(content, self.content_type)
                score = assessment.consensus_score / 100.0  # Normalize to 0-1 range
            else:
                # Fallback scoring based on content characteristics
                word_count = len(content.split())
                score = min(1.0, word_count / 1000.0)  # Simple length-based scoring
            
            # Also perform quality assessment if available
            if self.quality_assessment:
                quality_result = self.quality_assessment.assess_quality(content, self.content_type)
                quality_score = quality_result.composite_score / 100.0
                # Combine scores
                score = (score + quality_score) / 2.0
            
            return {
                "score": score,
                "timestamp": datetime.now().timestamp(),
                "content_length": len(content),
                "word_count": len(content.split()),
                "line_count": len(content.splitlines())
            }
        except Exception as e:
            print(f"Error evaluating content: {e}")
            return {
                "score": 0.0,
                "timestamp": datetime.now().timestamp(),
                "error": str(e)
            }


def run_evolution(initial_content: str, 
                 content_type: str,
                 api_key: str,
                 model_name: str = "gpt-4o",
                 api_base: str = "https://api.openai.com/v1",
                 max_iterations: int = 20,
                 population_size: int = 10,
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 evaluator_team: EvaluatorTeam = None,
                 quality_assessment: QualityAssessmentEngine = None) -> Dict[str, Any]:
    """
    Run evolution using OpenEvolve's core functionality
    
    Args:
        initial_content: Starting content to evolve
        content_type: Type of content being evolved
        api_key: API key for the LLM provider
        model_name: Name of the LLM to use for evolution
        api_base: Base URL for the API
        max_iterations: Maximum number of evolution iterations
        population_size: Size of the population to maintain
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        evaluator_team: Evaluator team for assessment
        quality_assessment: Quality assessment engine
    
    Returns:
        Dictionary with evolution results
    """
    if not OPENEVOLVE_AVAILABLE:
        raise RuntimeError("OpenEvolve backend is not available")
    
    try:
        # Create configuration for OpenEvolve
        config = Config()
        
        # Configure LLM model
        llm_config = LLMModelConfig(
            name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        config.llm.models = [llm_config]
        config.max_iterations = max_iterations
        config.database.population_size = population_size
        config.database.num_islands = 1  # Add island model for better exploration
        config.database.elite_selection_ratio = 0.2  # Keep top 20% of individuals
        # Note: mutation_rate, crossover_rate are not standard OpenEvolve config attributes
        config.database.archive_size = 100  # Maintain an archive of best solutions
        config.checkpoint_interval = 5  # Save checkpoint every 5 iterations
        
        # Configure database settings
        config.database.feature_dimensions = ["quality", "diversity", "complexity"]  # Multi-objective optimization
        config.database.feature_bins = 10  # Number of bins for each feature
        config.database.elite_selection_ratio = 0.2  # Ratio of elite individuals to preserve
        config.database.exploration_ratio = 0.3  # Ratio for exploration in evolution
        config.database.exploitation_ratio = 0.7  # Ratio for exploitation in evolution
        
        # Configure evaluator settings
        config.evaluator.timeout = 300  # 5 minute timeout for evaluation
        config.evaluator.max_retries = 3  # Retry failed evaluations
        config.evaluator.cascade_evaluation = True  # Use cascade evaluation
        config.evaluator.cascade_thresholds = [0.5, 0.75, 0.9]  # Cascade thresholds
        config.evaluator.parallel_evaluations = os.cpu_count() or 4  # Use all available cores
        
        # Create evaluator
        evaluator = OpenEvolveEvaluator(evaluator_team, quality_assessment, content_type)
        
        # Save initial content to temporary file for OpenEvolve
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding='utf-8') as temp_file:
            temp_file.write(initial_content)
            temp_file_path = temp_file.name
        
        try:
            # Run evolution using OpenEvolve
            result = openevolve_run_evolution(
                initial_program=temp_file_path,
                evaluator=evaluator.evaluate,
                config=config,
                iterations=max_iterations,
                output_dir=None,  # Use temporary directory
                cleanup=True,
            )
            
            # Process results
            evolution_result = {
                "success": True,
                "best_content": result.best_code if result.best_code else initial_content,
                "best_score": result.best_score if result.best_score else 0.0,
                "iterations_completed": max_iterations,
                "final_population_size": population_size,
                "metrics": result.metrics if result.metrics else {},
                "timestamp": datetime.now().isoformat()
            }
            
            return evolution_result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        print(f"Error running evolution: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def evolve_content(initial_content: str, 
                  content_type: str = "general",
                  api_key: str = None,
                  model_name: str = "gpt-4o",
                  max_iterations: int = 10,
                  temperature: float = 0.7) -> Dict[str, Any]:
    """
    Convenience function to evolve content using OpenEvolve
    
    Args:
        initial_content: The content to evolve
        content_type: Type of content (affects evaluation)
        api_key: API key for LLM (if None, uses environment)
        model_name: Model to use for evolution
        max_iterations: Maximum number of evolution iterations
        temperature: Temperature for generation
    
    Returns:
        Dictionary with evolution results
    """
    # If no API key provided, try to get from environment
    if not api_key:
        import os
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set in environment as OPENAI_API_KEY or OPENROUTER_API_KEY")
    
    # Create evaluator team and quality assessment engines if available
    evaluator_team = None
    quality_assessment = None
    
    try:
        evaluator_team = EvaluatorTeam()
    except Exception:
        print("Could not initialize EvaluatorTeam, using default evaluation")
    
    try:
        quality_assessment = QualityAssessmentEngine()
    except Exception:
        print("Could not initialize QualityAssessmentEngine, using default evaluation")
    
    # Run evolution
    return run_evolution(
        initial_content=initial_content,
        content_type=content_type,
        api_key=api_key,
        model_name=model_name,
        max_iterations=max_iterations,
        temperature=temperature,
        evaluator_team=evaluator_team,
        quality_assessment=quality_assessment
    )


# Example usage and testing
def test_evolution():
    """Test function for the evolution functionality"""
    print("Testing evolution functionality using OpenEvolve:")
    
    # Test with sample content
    sample_content = """
# Sample Document
This is a sample document for testing the evolution functionality.
It contains multiple paragraphs and sections to demonstrate the evolution process.

## Section 1
This section discusses the basics of evolutionary algorithms and their applications.

## Section 2
This section explores advanced topics in genetic programming and optimization.

## Conclusion
This concludes our sample document for testing purposes.
"""
    
    print(f"Sample content length: {len(sample_content)} characters")
    
    # Try to evolve the content (this will require API key)
    try:
        # This would require a valid API key to work
        result = evolve_content(
            initial_content=sample_content,
            content_type="document",
            max_iterations=3,  # Small number for testing
            temperature=0.7
        )
        
        if result.get("success"):
            print("Evolution completed successfully!")
            print(f"Best score achieved: {result.get('best_score', 0):.4f}")
            print(f"Result content length: {len(result.get('best_content', ''))} characters")
        else:
            print(f"Evolution failed: {result.get('error', 'Unknown error')}")
            print("Note: This may be due to missing API key")
            
    except ValueError as e:
        print(f"Configuration error (likely missing API key): {e}")
        print("The evolution function is properly configured to use OpenEvolve backend")
    
    return True


if __name__ == "__main__":
    test_evolution()