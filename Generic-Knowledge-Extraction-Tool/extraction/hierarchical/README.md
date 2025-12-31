# Hierarchical_extraction Directory

## Purpose
Advanced multi-level extraction system (Case 2) that handles complex documents requiring hierarchical data extraction. Supports multi-document processing, document classification, and sequential extraction strategies for documents with nested or related data structures.

## Key Files
- `case2_main.py` - Main orchestrator for hierarchical extraction workflows
- `case2_core.py` - Core data structures and extraction strategy definitions
- `case2_extractor.py` - Sequential extraction engine for multi-step processing
- `case2_model_generator.py` - Dynamic model generation for hierarchical structures
- `case2_strategy_generator.py` - AI-powered extraction strategy creation
- `case2_ai_adapter.py` - AI client adapter for hierarchical processing
- `__init__.py` - Module initialization and exports

## Architecture
```
Hierarchical Extraction Pipeline:
1. Strategy Generation → Define extraction approach
2. Model Generation → Create hierarchical Pydantic models  
3. Document Classification → Identify document types
4. Sequential Extraction → Multi-pass extraction with dependencies
5. Result Consolidation → Merge and validate hierarchical results
```

### Processing Flow
```python
Case2Orchestrator → Strategy → Models → Extractor → Results
     ↓                ↓         ↓         ↓         ↓
  AI Planning    Dynamic Gen  Multi-Doc  Sequential  Validated
                              Classes   Extraction   Hierarchy
```

## Integration
- **Main Application**: Extends the core extraction system for complex use-cases
- **Use-cases**: Powers hierarchical use-cases like `LabReportExtraction` with sub-categories
- **AI Clients**: Utilizes advanced AI capabilities for strategy planning and model generation
- **Document Parser**: Integrates with document classification for type-specific processing

## Patterns
1. **Strategy-First Design**: AI generates extraction strategies before execution
2. **Hierarchical Models**: Nested Pydantic structures for complex data relationships
3. **Sequential Processing**: Multi-pass extraction with inter-document dependencies
4. **Type-Aware Extraction**: Document classification drives extraction approach

## Dependencies
- **Core Dependencies**: AI clients (Claude/OpenAI), Pydantic models, document parsers
- **Strategy Generation**: Requires AI client for intelligent extraction planning
- **Model Generation**: Depends on dynamic Pydantic model creation capabilities
- **Document Processing**: Integrates with classification and parsing systems

## Entry Points
1. **Case2Orchestrator**: Main entry point for hierarchical extraction workflows
2. **create_new_use_case()**: Creates new hierarchical extraction configurations
3. **extract_documents()**: Processes multiple documents with hierarchical structure

## Common Usage Patterns

### Hierarchical Extraction Setup
```python
from Hierarchical_extraction.case2_main import Case2Orchestrator

orchestrator = Case2Orchestrator(ai_client=claude_client)
result = orchestrator.create_new_use_case(
    description="Extract lab results with sub-test categories",
    use_case_name="ComplexLabReports"
)
```

### Multi-Document Processing
```python
# Sequential extraction with document dependencies
extraction_result = orchestrator.extract_documents(
    documents=[doc1, doc2, doc3],
    strategy=hierarchical_strategy
)
```

### Strategy Generation
```python
# AI-powered extraction strategy creation
strategy = orchestrator.strategy_generator.generate_strategy(
    description="Process purchase orders with linked BOMs",
    document_types=["PO", "BOM", "Invoice"]
)
```

## Advanced Features
- **Dynamic Strategy Generation**: AI creates extraction approaches based on document analysis
- **Cross-Document Relationships**: Links data between related documents in a set
- **Adaptive Model Creation**: Generates hierarchical Pydantic models for complex structures
- **Sequential Dependencies**: Supports extraction workflows where later steps depend on earlier results

## Integration with Core System
This module extends the main extraction pipeline for scenarios requiring:
- Multi-document sets with relationships
- Hierarchical data structures (parent-child relationships)
- Complex business logic spanning multiple document types
- Sequential processing with state management