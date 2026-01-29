#!/bin/bash
# Probe script for KG-Gen extraction pipeline
# LAW OF RUNTIME TRUTH: Verify extraction pipeline works before using it

set -e

echo "=== KG-Gen Extraction Pipeline Probe ==="

# Check if extraction_pipeline module exists
echo "Checking extraction_pipeline module..."
python3 -c "from knowledge_engine.integrations.kggen.extraction_pipeline import ExtractionPipeline; print('✓ ExtractionPipeline import successful')"

# Check configuration validation
echo "Testing configuration validation..."
python3 -c "
from knowledge_engine.integrations.kggen.extraction_pipeline import PipelineConfig
config = PipelineConfig()
config.validate()
print('✓ Configuration validation successful')
"

# Test basic entity extraction
echo "Testing entity extraction..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.extraction_pipeline import ExtractionPipeline

async def test():
    pipeline = ExtractionPipeline()
    entities = pipeline._extract_entities_fallback('Apple and Google are tech companies.')
    assert len(entities) > 0, 'No entities extracted'
    print(f'✓ Entity extraction successful: {entities}')
    await pipeline.close()

asyncio.run(test())
"

# Test relation extraction
echo "Testing relation extraction..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.extraction_pipeline import ExtractionPipeline

async def test():
    pipeline = ExtractionPipeline()
    entities = ['Apple', 'Google']
    relations = pipeline._extract_relations_fallback('Apple acquired a startup. Google released Android.', entities)
    print(f'✓ Relation extraction successful: {len(relations)} relations')
    await pipeline.close()

asyncio.run(test())
"

# Test correlation ID generation
echo "Testing correlation ID generation..."
python3 -c "
from knowledge_engine.integrations.kggen.extraction_pipeline import ExtractionPipeline
pipeline = ExtractionPipeline()
correlation_id = pipeline.generate_correlation_id('test text')
assert correlation_id.startswith('kggen-'), f'Invalid correlation ID: {correlation_id}'
print(f'✓ Correlation ID generation successful: {correlation_id}')
"

echo "=== All Extraction Pipeline Probes Passed ==="
