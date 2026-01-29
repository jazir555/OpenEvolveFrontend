# OneKE Integration File Structure

```
Frontend/
├── integrations/
│   ├── base/
│   │   └── extraction_interface.py          # Abstract interface for extraction implementations
│   │                                         # - ExtractionInterface (ABC)
│   │                                         # - ExtractionResult, SchemaDefinition dataclasses
│   │                                         # - ExtractionType enum
│   │                                         # - Custom exceptions
│   │
│   └── oneke/
│       ├── __init__.py                       # Package initialization
│       ├── adapter.py                        # OneKEAdapter (implements ExtractionInterface)
│       │                                     # - extract_ner(), extract_re(), extract_ee(), extract_triple()
│       │                                     # - extract_schema_guided()
│       │                                     # - batch_extract()
│       │                                     # - validate(), shutdown()
│       │                                     # - load_schema()
│       │
│       ├── bridge.py                         # OneKEBridge (workflow integration)
│       │                                     # - extract_from_workflow()
│       │                                     # - extract_physics_knowledge()
│       │                                     # - extract_chemistry_knowledge()
│       │                                     # - extract_solution_patterns()
│       │                                     # - batch_extract_from_workflows()
│       │                                     # - validate_integration()
│       │
│       ├── config.yaml                       # Configuration file
│       │                                     # - Model settings
│       │                                     # - Feature flags
│       │                                     # - Schema list
│       │                                     # - Performance tuning
│       │
│       ├── schemas/
│       │   ├── physics.yaml                  # Physics domain schema
│       │   │                                 # - Entity types (quantum_system, observable, etc.)
│       │   │                                 # - Relation types (describes, evolves_by, etc.)
│       │   │                                 # - Event types (measurement, evolution)
│       │   │                                 # - Examples and constraints
│       │   │
│       │   ├── chemistry.yaml                # Chemistry domain schema
│       │   │                                 # - Entity types (substance, reaction, etc.)
│       │   │                                 # - Relation types (contains, catalyzes, etc.)
│       │   │                                 # - Event types (chemical_reaction, synthesis)
│       │   │                                 # - Examples and constraints
│       │   │
│       │   └── relations.yaml                # General relations schema
│       │                                     # - Entity types (concept, property, constraint)
│       │                                     # - Relation types (causes, enables, requires, etc.)
│       │                                     # - Examples and constraints
│       │
│       ├── INTEGRATION_COMPLETE.md           # Integration completion report
│       └── FILE_STRUCTURE.md                # This file
│
├── docs/
│   └── integrations/
│       └── ONEKE_INTEGRATION_GUIDE.md       # Comprehensive integration guide
│                                             # - Overview and purpose
│                                             # - Technical implementation
│                                             # - Configuration reference
│                                             # - Usage examples
│                                             # - API reference
│                                             # - Testing guide
│                                             # - Troubleshooting
│                                             # - Future enhancements
│
├── tests/
│   └── integrations/
│       └── test_oneke_integration.py        # Comprehensive test suite
│                                             # - Adapter tests
│                                             # - Bridge tests
│                                             # - Domain extraction tests
│                                             # - Integration tests
│                                             # - Error handling tests
│                                             # - Performance tests
│
└── workflow_knowledge_extractor.py          # Enhanced with OneKE integration
                                            # - use_oneke parameter
                                            # - extract_domain_knowledge()
                                            # - extract_enhanced_solution_patterns()
                                            # - extract_all_knowledge_enhanced()
```

## File Summary

### Core Integration Files (11 files)

1. **integrations/base/extraction_interface.py** (280 lines)
   - Abstract base class for extraction implementations
   - Dataclasses for results and schemas
   - Type definitions and exceptions

2. **integrations/oneke/adapter.py** (580 lines)
   - OneKEAdapter class implementing ExtractionInterface
   - Async API for all extraction types
   - Schema loading and validation
   - Batch processing support

3. **integrations/oneke/bridge.py** (450 lines)
   - OneKEBridge class for workflow integration
   - Domain-specific extraction methods
   - Batch workflow processing
   - Result caching

4. **integrations/oneke/config.yaml** (60 lines)
   - Model configuration
   - Feature flags
   - Performance settings

5. **integrations/oneke/schemas/physics.yaml** (100 lines)
   - Physics domain schema
   - 7 entity types, 6 relation types, 3 event types

6. **integrations/oneke/schemas/chemistry.yaml** (100 lines)
   - Chemistry domain schema
   - 7 entity types, 6 relation types, 3 event types

7. **integrations/oneke/schemas/relations.yaml** (100 lines)
   - General relations schema
   - 3 entity types, 10 relation types

8. **integrations/oneke/__init__.py** (20 lines)
   - Package initialization
   - Public API exports

9. **docs/integrations/ONEKE_INTEGRATION_GUIDE.md** (400+ lines)
   - Comprehensive integration guide
   - All documentation needed for usage

10. **tests/integrations/test_oneke_integration.py** (400+ lines)
    - Complete test suite
    - Unit and integration tests

11. **workflow_knowledge_extractor.py** (updated)
    - Enhanced with OneKE integration
    - 160 new lines of code

### Total Lines of Code: ~2,650 lines

## Integration Points

### 1. workflow_knowledge_extractor.py
```python
from workflow_knowledge_extractor import WorkflowKnowledgeExtractor

# Enable OneKE integration
extractor = WorkflowKnowledgeExtractor(use_oneke=True)

# Extract with domain knowledge
counts = await extractor.extract_all_knowledge_enhanced(workflow)
```

### 2. Direct Adapter Usage
```python
from integrations.oneke import OneKEAdapter

adapter = OneKEAdapter()
await adapter.initialize()
result = await adapter.extract_schema_guided(text, schema)
```

### 3. Bridge Usage
```python
from integrations.oneke import OneKEBridge

bridge = OneKEBridge()
await bridge.initialize()
knowledge = await bridge.extract_physics_knowledge(workflow)
```

## Dependencies

### Required
- Python 3.8+
- PyYAML (for schema loading)
- asyncio (for async operations)

### Optional
- OneKE framework (https://github.com/zjunlp/OneKE)
- OpenAI API (for ChatGPT models)
- Docker (for containerized OneKE)

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key"
export ONEKE_PATH="/path/to/OneKE"  # Optional
```

## Testing

```bash
# Run all tests
pytest tests/integrations/test_oneke_integration.py -v

# Run specific test class
pytest tests/integrations/test_oneke_integration.py::TestPhysicsExtraction -v

# Run with coverage
pytest tests/integrations/test_oneke_integration.py --cov=integrations/oneke
```

## Documentation

- **Quick Start**: See docs/integrations/ONEKE_INTEGRATION_GUIDE.md
- **API Reference**: See docs/integrations/ONEKE_INTEGRATION_GUIDE.md (API Reference section)
- **Configuration**: See integrations/oneke/config.yaml
- **Schemas**: See integrations/oneke/schemas/

## Support

For issues or questions:
1. Check ONEKE_INTEGRATION_GUIDE.md
2. Review test examples
3. Check OneKE documentation
4. Open GitHub issue

---

**Status**: Complete
**Version**: 0.1.0
**Last Updated**: 2025-01-02
