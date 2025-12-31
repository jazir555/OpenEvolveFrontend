"""
Generic MAKER/MDAP - Demo Script

Demonstrates the generic MAKER framework working with different task types:
- Code generation
- Document summarization
- Text processing
- Custom optimization

Usage:
    python demo_generic_maker.py
"""

import asyncio
import logging
import re
from typing import Dict, Any

from generic_maker_integration import (
    run_generic_maker,
    GenericEvaluator,
    GenericTask,
    TaskType,
    MAKERConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


# ============================================================================
# Example Evaluators
# ============================================================================

class CodeGeneratorEvaluator(GenericEvaluator):
    """Evaluator for code generation tasks"""

    def evaluate(self, solution: str, task: GenericTask) -> float:
        """Evaluate code quality"""
        score = 0.0

        # Check for function definition
        if "def " in solution:
            score += 0.3

        # Check for docstring
        if '"""' in solution or "'''" in solution:
            score += 0.2

        # Check for error handling
        if "try:" in solution or "except" in solution:
            score += 0.2

        # Check for type hints
        if "->" in solution or ": " in solution:
            score += 0.1

        # Check for comments
        if "#" in solution:
            score += 0.1

        # Check length (prefer reasonable length)
        if 50 < len(solution) < 1000:
            score += 0.1

        return min(1.0, score)

    def get_evaluation_details(self) -> Dict[str, Any]:
        return {
            "metrics": ["function_def", "docstring", "error_handling", "type_hints", "comments", "length"],
            "description": "Evaluates Python code quality based on best practices"
        }


class DocumentSummarizerEvaluator(GenericEvaluator):
    """Evaluator for document summarization"""

    def evaluate(self, solution: str, task: GenericTask) -> float:
        """Evaluate summary quality"""
        score = 0.0

        # Check for summary indicators
        if any(word in solution.lower() for word in ["summary", "key", "main", "important"]):
            score += 0.2

        # Check for bullet points or structured output
        if "-" in solution or "*" in solution or "•" in solution:
            score += 0.2

        # Check for reasonable length
        words = solution.split()
        if 20 <= len(words) <= 100:
            score += 0.2
        elif len(words) > 0:
            score += 0.1

        # Check for complete sentences
        sentences = re.split(r'[.!?]', solution)
        if len([s for s in sentences if len(s.strip()) > 0]) >= 2:
            score += 0.2

        # Check for concise (not just copying)
        if len(solution) < len(task.description) * 0.8:
            score += 0.2

        return min(1.0, score)

    def get_evaluation_details(self) -> Dict[str, Any]:
        return {
            "metrics": ["summary_indicators", "structure", "length", "sentences", "conciseness"],
            "description": "Evaluates summary quality based on structure and conciseness"
        }


class TextProcessingEvaluator(GenericEvaluator):
    """Evaluator for text processing tasks"""

    def evaluate(self, solution: str, task: GenericTask) -> float:
        """Evaluate text processing quality"""
        score = 0.0

        # Check if result contains key terms from description
        words_in_desc = set(task.description.lower().split())
        words_in_sol = set(solution.lower().split())
        overlap = len(words_in_desc & words_in_sol)

        if overlap > 0:
            score += min(0.3, overlap / len(words_in_desc) * 0.3)

        # Check for proper formatting
        if solution[0].isupper() if solution else False:
            score += 0.2

        # Check for reasonable length
        if 10 < len(solution) < 500:
            score += 0.2

        # Check for structure (multiple parts)
        parts = solution.split("\n")
        if len(parts) >= 2:
            score += 0.2

        # Check for processing indicators
        if any(word in solution.lower() for word in ["processed", "result", "output", "final"]):
            score += 0.1

        return min(1.0, score)

    def get_evaluation_details(self) -> Dict[str, Any]:
        return {
            "metrics": ["keyword_overlap", "formatting", "length", "structure", "indicators"],
            "description": "Evaluates text processing quality"
        }


class CustomOptimizationEvaluator(GenericEvaluator):
    """Generic evaluator for custom optimization tasks"""

    def evaluate(self, solution: str, task: GenericTask) -> float:
        """Evaluate solution for custom optimization"""
        # This is a placeholder - users would implement their own logic
        score = 0.0

        # Reward longer solutions (assuming more detail is better)
        score += min(0.3, len(solution) / 1000)

        # Reward structured output
        if "\n" in solution:
            score += 0.3

        # Reward specific keywords from requirements
        for req in task.requirements:
            if req.lower() in solution.lower():
                score += 0.1

        return min(1.0, score)

    def get_evaluation_details(self) -> Dict[str, Any]:
        return {
            "metrics": ["length", "structure", "requirement_match"],
            "description": "Generic evaluator for custom optimization tasks"
        }


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_1_code_generation():
    """Demo 1: Code Generation"""
    print_section("DEMO 1: Code Generation with MAKER")

    task = "Generate a Python function to validate email addresses"

    print(f"Task: {task}")
    print("MAKER: Voting + Decomposition + Evolution")

    evaluator = CodeGeneratorEvaluator()

    config = MAKERConfig(
        enable_voting=True,
        voting_threshold=3,
        enable_decomposition=True,
        max_generations=10,
        population_size=10
    )

    result = await run_generic_maker(
        task_description=task,
        evaluator=evaluator,
        task_type=TaskType.CODE_GENERATION,
        config=config
    )

    print("\n[Result]")
    print(f"Success: {result.quality_score > 0}")
    print(f"Quality: {result.quality_score:.3f}")
    print(f"Generation: {result.generation}")
    print(f"\nSolution:\n{result.solution}")

    return result


async def demo_2_document_summarization():
    """Demo 2: Document Summarization"""
    print_section("DEMO 2: Document Summarization with MAKER")

    document = """
    Machine learning is a subset of artificial intelligence that focuses on
    building systems that can learn from data. It enables computers to
    automatically learn and improve from experience without being explicitly
    programmed. Key types include supervised learning, unsupervised learning,
    and reinforcement learning. Applications range from image recognition to
    natural language processing and autonomous vehicles.
    """

    task = f"Summarize this document: {document}"

    print(f"Task: Summarize document about machine learning")
    print("MAKER: Voting + Evolution")

    evaluator = DocumentSummarizerEvaluator()

    config = MAKERConfig(
        enable_voting=True,
        voting_threshold=2,
        enable_decomposition=False,  # Summarization doesn't need decomposition
        max_generations=8,
        population_size=8
    )

    result = await run_generic_maker(
        task_description=task,
        evaluator=evaluator,
        task_type=TaskType.TEXT_SUMMARIZATION,
        config=config
    )

    print("\n[Result]")
    print(f"Quality: {result.quality_score:.3f}")
    print(f"\nSummary:\n{result.solution}")

    return result


async def demo_3_text_processing():
    """Demo 3: Text Processing"""
    print_section("DEMO 3: Text Processing with MAKER")

    task = "Process and format the following text for readability: This is a sample text that needs to be processed and formatted properly"

    print(f"Task: Format text for readability")
    print("MAKER: Evolution only")

    evaluator = TextProcessingEvaluator()

    config = MAKERConfig(
        enable_voting=False,
        enable_decomposition=False,
        max_generations=5,
        population_size=8
    )

    result = await run_generic_maker(
        task_description=task,
        evaluator=evaluator,
        task_type=TaskType.DOCUMENT_PROCESSING,
        config=config
    )

    print("\n[Result]")
    print(f"Quality: {result.quality_score:.3f}")
    print(f"\nProcessed Text:\n{result.solution}")

    return result


async def demo_4_custom_optimization():
    """Demo 4: Custom Optimization"""
    print_section("DEMO 4: Custom Optimization with MAKER")

    task = GenericTask(
        task_id="custom_1",
        description="Design a user authentication system",
        task_type=TaskType.CUSTOM,
        requirements=["secure", "scalable", "user-friendly"]
    )

    print(f"Task: Design user authentication system")
    print("Requirements: secure, scalable, user-friendly")
    print("MAKER: Full MAKER (Voting + Decomposition + Evolution)")

    evaluator = CustomOptimizationEvaluator()

    config = MAKERConfig(
        enable_voting=True,
        voting_threshold=3,
        enable_decomposition=True,
        max_generations=12,
        population_size=12
    )

    result = await run_generic_maker(
        task_description=task.description,
        evaluator=evaluator,
        task_type=TaskType.CUSTOM,
        config=config
    )

    print("\n[Result]")
    print(f"Quality: {result.quality_score:.3f}")
    print(f"\nSolution:\n{result.solution}")

    return result


async def demo_5_comparison():
    """Demo 5: Compare different MAKER configurations"""
    print_section("DEMO 5: Configuration Comparison")

    task = "Generate a function to calculate fibonacci numbers"

    print(f"Task: {task}")
    print("Comparing different configurations...")

    evaluator = CodeGeneratorEvaluator()

    configs = [
        ("Voting Only", MAKERConfig(enable_voting=True, enable_decomposition=False, max_generations=5)),
        ("Decomposition Only", MAKERConfig(enable_voting=False, enable_decomposition=True, max_generations=5)),
        ("Full MAKER", MAKERConfig(enable_voting=True, enable_decomposition=True, max_generations=5)),
    ]

    results = []
    for name, config in configs:
        print(f"\n  Testing: {name}...")
        result = await run_generic_maker(
            task_description=task,
            evaluator=evaluator,
            task_type=TaskType.CODE_GENERATION,
            config=config
        )
        results.append((name, result))
        print(f"    Quality: {result.quality_score:.3f}, Generations: {result.generation}")

    print("\n[Summary]")
    print("  Configuration    | Quality | Generations")
    print("  -----------------|---------|------------")
    for name, result in results:
        print(f"  {name:16s} | {result.quality_score:7.3f} | {result.generation:11}")

    return results


async def demo_6_capabilities():
    """Demo 6: Check Capabilities"""
    print_section("DEMO 6: Generic MAKER Capabilities")

    from generic_maker_integration import get_generic_maker_capabilities

    capabilities = get_generic_maker_capabilities()

    print("Generic MAKER Integration Capabilities:")
    print(f"  - MAKER enabled: {capabilities.get('generic_maker_enabled', False)}")
    print(f"  - MDAP available: {capabilities.get('mdap_available', False)}")
    print(f"  - Integration status: {capabilities.get('integration_status', 'unknown')}")

    print("\n  Supported Task Types:")
    for task_type in capabilities.get('supported_task_types', []):
        print(f"    - {task_type}")

    print("\n  Features:")
    for feature, description in capabilities.get('features', {}).items():
        print(f"    - {feature}: {description}")

    if 'paper' in capabilities:
        paper = capabilities['paper']
        print(f"\n  Paper Reference:")
        print(f"    - Title: {paper.get('title', 'N/A')}")
        print(f"    - arXiv: {paper.get('arxiv', 'N/A')}")
        print(f"    - URL: {paper.get('url', 'N/A')}")

    return capabilities


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos"""
    print("\n")
    print("=" * 80)
    print("  GENERIC MAKER/MDAP - DEMONSTRATION")
    print("  Works with ANY task type - not just math proofs!")
    print("  Paper: arXiv:2511.09030 (Solving a Million-Step LLM Task with Zero Errors)")
    print("=" * 80)
    print("")

    demos = [
        ("Code Generation", demo_1_code_generation),
        ("Document Summarization", demo_2_document_summarization),
        ("Text Processing", demo_3_text_processing),
        ("Custom Optimization", demo_4_custom_optimization),
        ("Configuration Comparison", demo_5_comparison),
        ("Capabilities Check", demo_6_capabilities),
    ]

    print("Available Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all demos")
    print("")

    try:
        choice = input("Select demo (0-6, or press Enter for all): ").strip()
        if not choice:
            choice = "0"

        choice_num = int(choice)

        if choice_num == 0:
            # Run all demos
            for name, demo_func in demos:
                try:
                    await demo_func()
                except Exception as e:
                    logger.error(f"Demo {name} failed: {e}", exc_info=True)
        elif 1 <= choice_num <= len(demos):
            # Run selected demo
            name, demo_func = demos[choice_num - 1]
            await demo_func()
        else:
            print("Invalid choice")

    except ValueError:
        print("Invalid input")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

    print("\n" + "=" * 80)
    print("  DEMO COMPLETED")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - generic_maker_integration.py")
    print("  - GENERIC_MAKER_INTEGRATION_GUIDE.md")
    print("  - Paper: https://arxiv.org/abs/2511.09030")
    print("")


if __name__ == "__main__":
    asyncio.run(main())
