#!/bin/bash
# Probe script for KG-Gen deduplication engine
# LAW OF RUNTIME TRUTH: Verify deduplication works before using it

set -e

echo "=== KG-Gen Deduplication Engine Probe ==="

# Check if deduplication_engine module exists
echo "Checking deduplication_engine module..."
python3 -c "from knowledge_engine.integrations.kggen.deduplication_engine import DeduplicationEngine; print('✓ DeduplicationEngine import successful')"

# Check configuration validation
echo "Testing configuration validation..."
python3 -c "
from knowledge_engine.integrations.kggen.deduplication_engine import DeduplicationConfig
config = DeduplicationConfig()
config.validate()
print('✓ Configuration validation successful')
"

# Test SEMHASH deduplication
echo "Testing SEMHASH deduplication..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.deduplication_engine import SEMHASHStrategy, DeduplicationConfig

async def test():
    config = DeduplicationConfig()
    semhash = SEMHASHStrategy(config)

    entities = ['Apple', 'apple', 'APPLE', 'Google', 'Microsoft']
    unique, clusters = await semhash.deduplicate(entities, 'test-correlation')

    assert len(unique) < len(entities), 'SEMHASH did not deduplicate'
    print(f'✓ SEMHASH successful: {len(entities)} -> {len(unique)} unique')

asyncio.run(test())
"

# Test LM clustering
echo "Testing LM clustering..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.deduplication_engine import LMClusterStrategy, DeduplicationConfig

async def test():
    config = DeduplicationConfig()
    lm = LMClusterStrategy(config)

    entities = ['Apple Inc', 'Apple Corporation', 'Google LLC', 'Microsoft Corp']
    unique, clusters = await lm.deduplicate(entities, 'test-correlation')

    print(f'✓ LM clustering successful: {len(entities)} -> {len(unique)} unique, {len(clusters)} clusters')

asyncio.run(test())
"

# Test full deduplication
echo "Testing full deduplication..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.deduplication_engine import DeduplicationEngine, DeduplicationMethod

async def test():
    engine = DeduplicationEngine()

    entities = ['Apple', 'apple', 'APPLE', 'Google', 'google', 'Microsoft']
    result = await engine.deduplicate(entities, DeduplicationMethod.FULL, 'test-correlation')

    assert result.final_count < result.original_count, 'Full deduplication did not reduce count'
    assert result.reduction_rate > 0, 'Reduction rate should be positive'

    print(f'✓ Full deduplication successful: {result.original_count} -> {result.final_count} ({result.reduction_rate:.1%} reduction)')
    await engine.close()

asyncio.run(test())
"

# Test relationship deduplication
echo "Testing relationship deduplication..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.deduplication_engine import DeduplicationEngine

async def test():
    engine = DeduplicationEngine()

    relationships = [
        {'subject': 'Apple', 'predicate': 'owns', 'object': 'iOS'},
        {'subject': 'Apple', 'predicate': 'owns', 'object': 'iOS'},  # Duplicate
        {'subject': 'Google', 'predicate': 'owns', 'object': 'Android'}
    ]

    unique = await engine.deduplicate_relationships(relationships, 'test-correlation')

    assert len(unique) < len(relationships), 'Relationship deduplication did not work'
    print(f'✓ Relationship deduplication successful: {len(relationships)} -> {len(unique)} unique')
    await engine.close()

asyncio.run(test())
"

echo "=== All Deduplication Engine Probes Passed ==="
