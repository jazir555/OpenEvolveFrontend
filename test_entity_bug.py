"""Test script to reproduce the entity bug"""
import time
from datetime import datetime, timezone
from typing import Dict, Any
from knowledge_engine.core.entity_knowledge_graph import EntityKnowledgeGraph

def generate_entity_name(index: int, prefix: str = "entity") -> str:
    """Generate entity name"""
    return f"{prefix}_{index:06d}"

def generate_entity_type() -> str:
    """Generate random entity type"""
    types = ['Person', 'Organization', 'Location', 'Event', 'Concept', 'Document', 'Product', 'Transaction']
    index = int(time.time() * 1000) % 8
    return types[index]

def generate_attributes(entity_id: int) -> Dict[str, Any]:
    """Generate entity attributes"""
    return {
        'id': entity_id,
        'name': f"Entity_{entity_id}",
        'description': f"Test entity number {entity_id}",
        'created_at': datetime.now(timezone.utc).isoformat(),
        'value': entity_id * 10,
        'active': entity_id % 2 == 0,
        'tags': [f'tag{i}' for i in range(min(5, entity_id % 10))],
        'metadata': {'key': f'value_{entity_id}'}
    }

# Test
graph = EntityKnowledgeGraph()
success_count = 0
error_count = 0

for i in range(100):
    try:
        success = graph.add_entity(
            name=generate_entity_name(i),
            entity_type=generate_entity_type(),
            attributes=generate_attributes(i)
        )
        if success:
            success_count += 1
        else:
            error_count += 1
    except Exception as e:
        print(f"Error at entity {i}: {e}")
        import traceback
        traceback.print_exc()
        error_count += 1

print(f"Success: {success_count}, Errors: {error_count}")
