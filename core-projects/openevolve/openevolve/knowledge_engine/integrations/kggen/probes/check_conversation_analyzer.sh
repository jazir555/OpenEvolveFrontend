#!/bin/bash
# Probe script for KG-Gen conversation analyzer
# LAW OF RUNTIME TRUTH: Verify conversation analyzer works before using it

set -e

echo "=== KG-Gen Conversation Analyzer Probe ==="

# Check if conversation_analyzer module exists
echo "Checking conversation_analyzer module..."
python3 -c "from knowledge_engine.integrations.kggen.conversation_analyzer import ConversationAnalyzer; print('✓ ConversationAnalyzer import successful')"

# Check configuration validation
echo "Testing configuration validation..."
python3 -c "
from knowledge_engine.integrations.kggen.conversation_analyzer import ConversationAnalyzerConfig
config = ConversationAnalyzerConfig()
config.validate()
print('✓ Configuration validation successful')
"

# Test message parsing
echo "Testing message parsing..."
python3 -c "
from knowledge_engine.integrations.kggen.conversation_analyzer import ConversationAnalyzer

analyzer = ConversationAnalyzer()

messages = [
    {'role': 'user', 'content': 'Hello', 'speaker_id': 'user1'},
    {'role': 'assistant', 'content': 'Hi there', 'speaker_id': 'assistant'}
]

parsed = analyzer._parse_messages(messages)

assert len(parsed) == 2, f'Expected 2 messages, got {len(parsed)}'
assert parsed[0].role == 'user', 'First message should be user'
print(f'✓ Message parsing successful: {len(parsed)} messages parsed')
"

# Test conversation analysis
echo "Testing conversation analysis..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.conversation_analyzer import ConversationAnalyzer

async def test():
    analyzer = ConversationAnalyzer()

    messages = [
        {'role': 'user', 'content': 'Tell me about Apple and Google', 'speaker_id': 'user1'},
        {'role': 'assistant', 'content': 'Both are major tech companies', 'speaker_id': 'assistant'}
    ]

    result = await analyzer.analyze(messages)

    assert result.conversation_id, 'Conversation ID not generated'
    assert result.total_speakers > 0, 'No speakers detected'
    assert result.processing_time_seconds >= 0, 'Processing time invalid'

    print(f'✓ Conversation analysis successful: {result.total_speakers} speakers, {result.total_entities} entities')
    await analyzer.close()

asyncio.run(test())
"

# Test speaker entity extraction
echo "Testing speaker entity extraction..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.conversation_analyzer import ConversationAnalyzer, Message

async def test():
    analyzer = ConversationAnalyzer()

    messages = [
        Message(role='user', content='Apple and Google are tech companies', speaker_id='user1'),
        Message(role='assistant', content='Yes, both are major companies', speaker_id='assistant')
    ]

    entities = await analyzer.entity_extractor.extract_entities(
        messages,
        speaker_id='user1',
        correlation_id='test'
    )

    assert isinstance(entities, list), 'Entities should be a list'
    print(f'✓ Speaker entity extraction successful: {len(entities)} entities extracted')
    await analyzer.close()

asyncio.run(test())
"

# Test conversation-to-knowledge-graph conversion
echo "Testing conversation-to-KG conversion..."
python3 -c "
import asyncio
from knowledge_engine.integrations.kggen.conversation_analyzer import ConversationAnalyzer

async def test():
    analyzer = ConversationAnalyzer()

    messages = [
        {'role': 'user', 'content': 'Tell me about Apple', 'speaker_id': 'user1'},
        {'role': 'assistant', 'content': 'Apple is a tech company', 'speaker_id': 'assistant'}
    ]

    result = await analyzer.analyze(messages)

    assert isinstance(result.entities, list), 'Entities should be a list'
    assert isinstance(result.relationships, list), 'Relationships should be a list'

    print(f'✓ Conversation-to-KG conversion successful: {len(result.entities)} entities, {len(result.relationships)} relationships')
    await analyzer.close()

asyncio.run(test())
"

echo "=== All Conversation Analyzer Probes Passed ==="
