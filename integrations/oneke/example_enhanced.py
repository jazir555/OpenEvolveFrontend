"""
OneKE Enhanced Integration Examples

This module demonstrates how to use the enhanced OneKE integration
with reflection, quality enhancement, and case-based learning.
"""

import asyncio
import logging
from pathlib import Path

from .enhanced_bridge import (
    EnhancedOneKEBridge,
    create_enhanced_oneke_bridge,
    extract_with_quality
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example texts for different domains
EXAMPLE_TEXTS = {
    'software_engineering': """
    Python uses async/await for concurrent code execution. The async def
    syntax defines a coroutine, which can be scheduled with the asyncio
    library. Python 3.5 introduced these features to simplify asynchronous
    programming. Type hints were added in Python 3.5 to improve code
    documentation and IDE support.
    """,

    'physics': """
    Quantum entanglement is a phenomenon where quantum particles remain
    connected regardless of distance. When measuring one particle, the
    state of its entangled partner is instantly determined. Albert Einstein
    called this "spooky action at a distance." The Schrödinger equation
    describes how quantum states evolve over time.
    """,

    'mathematics': """
    Theorem: All prime numbers greater than 2 are odd. Proof: Assume there
    exists an even prime p > 2. Then p = 2k for some integer k > 1. But
    then p is divisible by 2, contradicting primality. Therefore, no even
    primes exist beyond 2. This is a proof by contradiction.
    """,

    'chemistry': """
    Photosynthesis converts carbon dioxide and water into glucose and oxygen
    using sunlight. The chemical equation is: 6CO2 + 6H2O + light energy
    → C6H12O6 + 6O2. Chlorophyll in chloroplasts absorbs light energy to
    drive this reaction. This process is fundamental to life on Earth.
    """
}


async def example_basic_extraction():
    """Example: Basic extraction with quality enhancement."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Extraction with Quality Enhancement")
    print("="*80)

    # Create bridge
    bridge = await create_enhanced_oneke_bridge()

    try:
        # Extract with enhancement
        result = await bridge.extract_with_enhancement(
            text=EXAMPLE_TEXTS['software_engineering'],
            schema='software_engineering',
            domain='software_engineering',
            enable_reflection=True,
            enable_cases=True,
            enable_validation=True,
            enable_consistency=True
        )

        # Display results
        print(f"\nQuality Score: {result.quality_score.overall:.2f}")
        print(f"Original Quality: {result.original_quality.overall:.2f}")
        print(f"Improvement: {result.quality_improvement:.2%}")
        print(f"\nStrategies Applied: {', '.join(result.strategies_applied)}")

        print(f"\nEntities Extracted: {len(result.extraction.get('entities', []))}")
        for entity in result.extraction.get('entities', [])[:5]:
            print(f"  - {entity.get('text')} ({entity.get('type')}) "
                  f"[confidence: {entity.get('confidence', 0.0):.2f}]")

        print(f"\nRelations Extracted: {len(result.extraction.get('relations', []))}")
        for relation in result.extraction.get('relations', [])[:5]:
            print(f"  - {relation.get('subject')} -> {relation.get('object')} "
                  f"({relation.get('type')})")

    finally:
        await bridge.shutdown()


async def example_extraction_with_feedback():
    """Example: Extraction with human feedback and learning."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Extraction with Feedback and Learning")
    print("="*80)

    bridge = await create_enhanced_oneke_bridge()

    try:
        # Extract with feedback
        result = await bridge.extract_and_learn(
            text=EXAMPLE_TEXTS['physics'],
            schema='physics',
            domain='physics',
            feedback={
                'correctness': 0.9,
                'completeness': 0.85,
                'comments': 'Good extraction of quantum concepts'
            }
        )

        print(f"\nQuality Score: {result.quality_score.overall:.2f}")
        print(f"Feedback Received: {result.metadata.get('feedback_received')}")
        print(f"Learning Occurred: {result.metadata.get('learning_occurred')}")

        # Check repository statistics
        stats = await bridge.get_repository_statistics()
        print(f"\nRepository Statistics:")
        print(f"  Total Cases: {stats['total_cases']}")
        print(f"  Average Quality: {stats['average_quality']:.2f}")
        print(f"  Domain Distribution: {stats['domain_distribution']}")

    finally:
        await bridge.shutdown()


async def example_batch_extraction():
    """Example: Batch extraction with enhancement."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Extraction with Enhancement")
    print("="*80)

    bridge = await create_enhanced_oneke_bridge()

    try:
        # Prepare texts
        texts = [
            EXAMPLE_TEXTS['software_engineering'],
            EXAMPLE_TEXTS['physics'],
            EXAMPLE_TEXTS['mathematics'],
            EXAMPLE_TEXTS['chemistry']
        ]

        # Batch extract
        results = await bridge.batch_extract_with_enhancement(
            texts=texts,
            schema='general',
            domain='general',
            enable_enhancement=True
        )

        # Display results
        print(f"\nProcessed {len(results)} extractions")
        for i, result in enumerate(results):
            print(f"\nExtraction {i+1}:")
            print(f"  Quality: {result.quality_score.overall:.2f}")
            print(f"  Entities: {len(result.extraction.get('entities', []))}")
            print(f"  Relations: {len(result.extraction.get('relations', []))}")
            print(f"  Strategies: {', '.join(result.strategies_applied)}")

    finally:
        await bridge.shutdown()


async def example_retrieve_similar_cases():
    """Example: Retrieve similar cases from repository."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Retrieve Similar Cases")
    print("="*80)

    bridge = await create_enhanced_oneke_bridge()

    try:
        # Add some cases first
        print("\nAdding sample cases to repository...")
        for domain, text in EXAMPLE_TEXTS.items():
            result = await bridge.extract_with_enhancement(
                text=text,
                schema=domain,
                domain=domain,
                enable_reflection=False,
                enable_cases=False,
                enable_validation=True
            )
            print(f"  Added case for {domain}: quality={result.quality_score.overall:.2f}")

        # Retrieve similar cases for a query
        query_text = "Type hints improve code documentation in Python 3.5"
        print(f"\nRetrieving cases similar to: '{query_text}'")

        from .case_repository import OneKECaseRepository
        similar = await bridge.case_repository.retrieve_similar_cases(
            query={'input_text': query_text, 'domain': 'software_engineering'},
            top_k=3,
            min_similarity=0.6
        )

        print(f"\nFound {len(similar)} similar cases:")
        for sim in similar:
            print(f"\n  Case ID: {sim.case.case_id}")
            print(f"  Similarity: {sim.similarity:.2f}")
            print(f"  Quality: {sim.case.quality_score:.2f}")
            print(f"  Match Reasons:")
            for reason in sim.match_reasons:
                print(f"    - {reason}")

    finally:
        await bridge.shutdown()


async def example_repository_management():
    """Example: Repository management (export/import)."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Repository Management")
    print("="*80)

    bridge = await create_enhanced_oneke_bridge()

    try:
        # Get initial statistics
        stats_before = await bridge.get_repository_statistics()
        print(f"\nInitial Statistics:")
        print(f"  Total Cases: {stats_before['total_cases']}")
        print(f"  Average Quality: {stats_before['average_quality']:.2f}")

        # Export repository
        export_path = "data/oneke_cases_export.json"
        print(f"\nExporting repository to {export_path}...")
        success = await bridge.export_repository(export_path)

        if success:
            print("  Export successful")

            # Import into new repository (simulate)
            print(f"\nImporting from {export_path}...")
            success = await bridge.import_repository(export_path)

            if success:
                stats_after = await bridge.get_repository_statistics()
                print(f"  Import successful")
                print(f"  Total Cases: {stats_after['total_cases']}")
                print(f"  Average Quality: {stats_after['average_quality']:.2f}")

    finally:
        await bridge.shutdown()


async def example_quality_metrics():
    """Example: Detailed quality metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Detailed Quality Metrics")
    print("="*80)

    bridge = await create_enhanced_oneke_bridge()

    try:
        # Extract with detailed metrics
        result = await bridge.extract_with_enhancement(
            text=EXAMPLE_TEXTS['mathematics'],
            schema='mathematics',
            domain='mathematics',
            enable_reflection=True,
            enable_validation=True
        )

        # Display quality breakdown
        print(f"\nQuality Breakdown:")
        print(f"  Completeness: {result.quality_score.completeness:.2f}")
        print(f"  Accuracy: {result.quality_score.accuracy:.2f}")
        print(f"  Consistency: {result.quality_score.consistency:.2f}")
        print(f"  Confidence: {result.quality_score.confidence:.2f}")
        print(f"  Overall: {result.quality_score.overall:.2f}")

        # Display metadata
        print(f"\nDetailed Metrics:")
        for key, value in result.metadata.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            elif isinstance(value, list) and len(value) <= 5:
                print(f"  {key}: {value}")

    finally:
        await bridge.shutdown()


async def example_domain_specific_extraction():
    """Example: Domain-specific extraction with custom settings."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Domain-Specific Extraction")
    print("="*80)

    bridge = await create_enhanced_oneke_bridge()

    try:
        # Extract from different domains
        for domain, text in EXAMPLE_TEXTS.items():
            print(f"\n--- Domain: {domain} ---")

            result = await bridge.extract_with_enhancement(
                text=text,
                schema=domain,
                domain=domain,
                enable_reflection=True,
                enable_cases=True,
                enable_validation=True
            )

            print(f"Quality: {result.quality_score.overall:.2f}")
            print(f"Entities: {len(result.extraction.get('entities', []))}")
            print(f"Relations: {len(result.extraction.get('relations', []))}")

            # Show top entities
            entities = result.extraction.get('entities', [])
            if entities:
                print(f"Top Entities:")
                for entity in sorted(
                    entities,
                    key=lambda e: e.get('confidence', 0.0),
                    reverse=True
                )[:3]:
                    print(f"  - {entity.get('text')}: {entity.get('confidence', 0.0):.2f}")

    finally:
        await bridge.shutdown()


async def example_quick_extraction():
    """Example: Quick extraction using convenience function."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Quick Extraction (Convenience Function)")
    print("="*80)

    # Use convenience function for quick extraction
    result = await extract_with_quality(
        text=EXAMPLE_TEXTS['chemistry'],
        schema='chemistry',
        domain='chemistry',
        enable_enhancement=True
    )

    print(f"\nQuality: {result.quality_score.overall:.2f}")
    print(f"Improvement: {result.quality_improvement:.2%}")
    print(f"\nExtracted {len(result.extraction.get('entities', []))} entities")
    print(f"Extracted {len(result.extraction.get('relations', []))} relations")


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("OneKE Enhanced Integration Examples")
    print("="*80)

    examples = [
        ("Basic Extraction", example_basic_extraction),
        ("Extraction with Feedback", example_extraction_with_feedback),
        ("Batch Extraction", example_batch_extraction),
        ("Retrieve Similar Cases", example_retrieve_similar_cases),
        ("Repository Management", example_repository_management),
        ("Quality Metrics", example_quality_metrics),
        ("Domain-Specific Extraction", example_domain_specific_extraction),
        ("Quick Extraction", example_quick_extraction)
    ]

    # Run examples
    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            logger.error(f"Example '{name}' failed: {e}", exc_info=True)

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == '__main__':
    asyncio.run(main())
