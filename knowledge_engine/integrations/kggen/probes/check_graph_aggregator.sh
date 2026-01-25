#!/bin/bash
# Probe script for KG-Gen graph aggregator
# LAW OF RUNTIME TRUTH: Verify graph aggregator works before using it

set -e

echo "=== KG-Gen Graph Aggregator Probe ==="

# Check if graph_aggregator module exists
echo "Checking graph_aggregator module..."
python3 -c "from knowledge_engine.integrations.kggen.graph_aggregator import GraphAggregator; print('✓ GraphAggregator import successful')"

# Check configuration validation
echo "Testing configuration validation..."
python3 -c "
from knowledge_engine.integrations.kggen.graph_aggregator import GraphAggregatorConfig
config = GraphAggregatorConfig()
config.validate()
print('✓ Configuration validation successful')
"

# Test graph aggregation
echo "Testing graph aggregation..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.graph_aggregator import GraphAggregator

async def test():
    aggregator = GraphAggregator()

    graphs = [
        {
            'entities': ['Apple', 'Google'],
            'relationships': [
                {'subject': 'Apple', 'predicate': 'competes_with', 'object': 'Google'}
            ]
        },
        {
            'entities': ['Apple', 'Microsoft'],
            'relationships': [
                {'subject': 'Apple', 'predicate': 'competes_with', 'object': 'Microsoft'}
            ]
        }
    ]

    result = await aggregator.aggregate(graphs)

    assert result.total_entities == 3, f'Expected 3 entities, got {result.total_entities}'
    assert result.total_relationships == 2, f'Expected 2 relationships, got {result.total_relationships}'
    assert result.aggregated_graph.version_id, 'Version ID not generated'

    print(f'✓ Graph aggregation successful: {result.total_entities} entities, {result.total_relationships} relationships')
    await aggregator.close()

asyncio.run(test())
"

# Test graph versioning
echo "Testing graph versioning..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.graph_aggregator import GraphAggregator

async def test():
    aggregator = GraphAggregator()

    graph1 = {'entities': ['Apple'], 'relationships': []}
    result1 = await aggregator.aggregate([graph1])
    version1_id = result1.aggregated_graph.version_id

    graph2 = {'entities': ['Apple', 'Google'], 'relationships': []}
    result2 = await aggregator.aggregate([graph2])

    # Should have multiple versions
    versions = await aggregator.list_versions()

    assert len(versions) >= 2, f'Expected at least 2 versions, got {len(versions)}'

    print(f'✓ Graph versioning successful: {len(versions)} versions stored')
    await aggregator.close()

asyncio.run(test())
"

# Test graph differential comparison
echo "Testing graph differential comparison..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.graph_aggregator import GraphAggregator

async def test():
    aggregator = GraphAggregator()

    graph1 = {'entities': ['Apple', 'Google'], 'relationships': []}
    result1 = await aggregator.aggregate([graph1])

    graph2 = {'entities': ['Apple', 'Google', 'Microsoft'], 'relationships': []}
    result2 = await aggregator.aggregate([graph2])

    # Compare versions
    diff = await aggregator.compare_versions(
        result1.aggregated_graph.version_id,
        result2.aggregated_graph.version_id
    )

    assert 'Microsoft' in diff.entities_added, 'Microsoft should be in entities_added'
    assert diff.change_count > 0, 'Change count should be positive'

    print(f'✓ Graph differential comparison successful: {diff.change_count} changes detected')
    await aggregator.close()

asyncio.run(test())
"

# Test conflict resolution
echo "Testing conflict resolution..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.graph_aggregator import GraphAggregator

async def test():
    aggregator = GraphAggregator()

    # Same entity from multiple sources
    graphs = [
        {
            'entities': ['Apple'],
            'relationships': [
                {'subject': 'Apple', 'predicate': 'owns', 'object': 'iOS'}
            ]
        },
        {
            'entities': ['Apple'],
            'relationships': [
                {'subject': 'Apple', 'predicate': 'owns', 'object': 'iOS'}
            ]
        }
    ]

    result = await aggregator.aggregate(graphs)

    # Should resolve conflicts
    assert result.total_entities == 1, f'Expected 1 entity, got {result.total_entities}'
    assert result.conflicts_resolved >= 0, 'Conflicts resolved should be non-negative'

    print(f'✓ Conflict resolution successful: {result.conflicts_resolved} conflicts resolved')
    await aggregator.close()

asyncio.run(test())
"

# Test merge strategies
echo "Testing merge strategies..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.graph_aggregator import GraphAggregator, GraphAggregatorConfig

async def test():
    # Test union strategy
    config = GraphAggregatorConfig(merge_strategy='union')
    aggregator = GraphAggregator(config)

    graphs = [
        {'entities': ['Apple', 'Google'], 'relationships': []},
        {'entities': ['Apple', 'Microsoft'], 'relationships': []}
    ]

    result = await aggregator.aggregate(graphs)

    # Union should have all entities
    assert result.total_entities == 3, f'Union strategy: Expected 3 entities, got {result.total_entities}'

    print(f'✓ Merge strategy successful: union strategy produced {result.total_entities} entities')
    await aggregator.close()

asyncio.run(test())
"

echo "=== All Graph Aggregator Probes Passed ==="
