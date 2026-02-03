"""
Global DSPy Integration Module for OpenEvolve

This module provides a central import point for DSPy integration across the OpenEvolve system.
It defines the DSPY_AVAILABLE constant and provides common DSPy utilities.
"""

import logging

logger = logging.getLogger(__name__)

# Try to import DSPy for enhanced prompting capabilities
try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    from dspy.predict import Predict
    from dspy import Signature
    
    DSPY_AVAILABLE = True
    logger.info("DSPy successfully imported - enhanced programmatic prompting available")
    
    # Initialize a basic DSPy configuration
    def initialize_dspy(lm_name: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize DSPy with a language model.
        
        Args:
            lm_name: Name of the language model to use
            api_key: API key for the language model (if needed)
        """
        if api_key:
            import os
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Configure DSPy with the specified model
        try:
            llm = dspy.OpenAI(model=lm_name, api_key=api_key)
            dspy.configure(lm=llm)
            return llm
        except Exception as e:
            logger.warning(f"Could not configure DSPy with model {lm_name}: {e}")
            # Try a default configuration
            try:
                llm = dspy.OpenAI(model="gpt-3.5-turbo")
                dspy.configure(lm=llm)
                return llm
            except Exception as e2:
                logger.warning(f"Could not configure default DSPy model: {e2}")
                return None

except ImportError:
    # DSPy is not available, set up fallbacks
    dspy = None
    BootstrapFewShot = None
    Predict = None
    Signature = None
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using standard prompting methods")

    def initialize_dspy(lm_name: str = "gpt-4o-mini", api_key: str = None):
        """
        Fallback function when DSPy is not available.
        
        Args:
            lm_name: Name of the language model to use
            api_key: API key for the language model (if needed)
        """
        logger.warning("DSPy not available - initialize_dspy is a no-op")
        return None


# Export commonly used DSPy components
if DSPY_AVAILABLE:
    DSPyPredict = Predict
    DSPySignature = Signature
    DSPyBootstrap = BootstrapFewShot
else:
    DSPyPredict = None
    DSPySignature = None
    DSPyBootstrap = None


def get_dspy_status() -> dict:
    """
    Get the status of DSPy integration.
    
    Returns:
        Dictionary with DSPy availability status
    """
    return {
        "dspy_available": DSPY_AVAILABLE,
        "dspy_module_loaded": dspy is not None,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }


# Define common DSPy signatures that can be reused across the system
if DSPY_AVAILABLE:
    class KnowledgeExtractionSignature(Signature):
        """Signature for extracting knowledge from content."""
        content_to_analyze = dspy.InputField(desc="Content to extract knowledge from")
        extraction_context = dspy.InputField(desc="Additional context for extraction")
        extraction_type = dspy.InputField(desc="Type of extraction (comprehensive, entities, relations, patterns)")
        
        extracted_entities = dspy.OutputField(desc="JSON array of entities with name, type, and description")
        extracted_relations = dspy.OutputField(desc="JSON array of relations between entities with source, target, and relationship type")
        identified_patterns = dspy.OutputField(desc="JSON array of patterns or concepts identified in the content")
        knowledge_summary = dspy.OutputField(desc="Structured summary of extracted knowledge")
        confidence_score = dspy.OutputField(desc="Confidence in the extraction (0-100)")


    class ContentEvaluationSignature(Signature):
        """Signature for evaluating content quality."""
        content_to_evaluate = dspy.InputField(desc="Content to evaluate for quality")
        content_type = dspy.InputField(desc="Type of content (code, document, etc.)")
        evaluation_criteria = dspy.InputField(desc="List of criteria to evaluate against")
        
        overall_quality_score = dspy.OutputField(desc="Overall quality score (0-100)")
        correctness_score = dspy.OutputField(desc="Correctness score (0-100)")
        clarity_score = dspy.OutputField(desc="Clarity score (0-100)")
        completeness_score = dspy.OutputField(desc="Completeness score (0-100)")
        effectiveness_score = dspy.OutputField(desc="Effectiveness score (0-100)")
        efficiency_score = dspy.OutputField(desc="Efficiency score (0-100)")
        maintainability_score = dspy.OutputField(desc="Maintainability score (0-100)")
        robustness_score = dspy.OutputField(desc="Robustness score (0-100)")
        security_score = dspy.OutputField(desc="Security score (0-100)")
        compliance_score = dspy.OutputField(desc="Compliance score (0-100)")
        aesthetics_score = dspy.OutputField(desc="Aesthetics score (0-100)")
        detailed_feedback = dspy.OutputField(desc="Detailed feedback and suggestions")
        confidence_level = dspy.OutputField(desc="Confidence level in evaluation (low, medium, high)")


    class StrategyGenerationSignature(Signature):
        """Signature for generating evolution strategies."""
        problem_description = dspy.InputField(desc="Description of the problem to solve")
        content_type = dspy.InputField(desc="Type of content being evolved")
        evolution_mode = dspy.InputField(desc="Mode of evolution (standard, adversarial, etc.)")
        
        suggested_strategies = dspy.OutputField(desc="JSON array of suggested strategies with title and description")
        recommended_strategy = dspy.OutputField(desc="Recommended strategy to use")
        confidence_score = dspy.OutputField(desc="Confidence in the recommendation (0-100)")
        potential_risks = dspy.OutputField(desc="Potential risks with the recommended strategy")
        success_factors = dspy.OutputField(desc="Key factors for success with this strategy")


    class SolutionPatternSignature(Signature):
        """Signature for identifying solution patterns."""
        solution_attempts = dspy.InputField(desc="List of solution attempts with results")
        problem_context = dspy.InputField(desc="Context of the problem being solved")
        
        identified_patterns = dspy.OutputField(desc="JSON array of identified solution patterns")
        pattern_applicability = dspy.OutputField(desc="When each pattern is applicable")
        pattern_strengths = dspy.OutputField(desc="Strengths of each pattern")
        pattern_weaknesses = dspy.OutputField(desc="Weaknesses of each pattern")
        implementation_guidance = dspy.OutputField(desc="Guidance for implementing each pattern")


# Define a global DSPy instance that can be shared across the system
_global_dspy_instance = None

def get_global_dspy_instance():
    """
    Get or create a global DSPy instance for the system.
    
    Returns:
        DSPy instance or None if DSPy is not available
    """
    global _global_dspy_instance
    
    if not DSPY_AVAILABLE:
        return None
    
    if _global_dspy_instance is None:
        # Create a basic configuration
        _global_dspy_instance = {
            'predict': DSPyPredict,
            'signature': DSPySignature,
            'bootstrap': DSPyBootstrap
        }
    
    return _global_dspy_instance


if __name__ == "__main__":
    # Test the DSPy integration
    print(f"DSPy Available: {DSPY_AVAILABLE}")
    print(f"Status: {get_dspy_status()}")
    
    if DSPY_AVAILABLE:
        print("DSPy components loaded successfully")
        print(f"Available components: {hasattr(dspy, 'Predict')}, {hasattr(dspy, 'Signature')}")
    else:
        print("DSPy not available - using fallback methods")