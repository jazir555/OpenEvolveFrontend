# Self-Play Evolution Integration with Knowledgebase Specification

> **STATUS: design-only / not implemented in this distribution.** Both halves exist separately — PSV self-play in `engines/other/psv_selfplay.py` (`PSVManager`, `MathematicalProblemProposer/Solver/Verifier`) and the knowledge stack in `knowledge_engine/engine.py` (`KnowledgeEngine`), `knowledge_engine/indexer.py` (`CodeIndexer`), `knowledge_engine/enhanced_engine.py`, `knowledge_engine/orchestration.py` — but the wiring described here does not exist: greps for `knowledge` in `engines/other/psv_selfplay.py` and for `psv`/`PSVManager` under `knowledge_engine/` both return no matches.
>
> **Integration backend:** the distribution's real backend is `services/openevolve-api` (FastAPI, port 8000) which mounts all `/api/*` route groups including `/api/knowledge` (`api/knowledge.py`), fronted by the BubbleLab Hono proxy at `apps/bubblelab-api/src/routes/openevolve.ts`. There is no self-play route group.
>
> **Last reconciled: 2026-08-20**

## Overview
This specification outlines the integration of PSV (Propose, Solve, Verify) self-play functionality with the existing knowledgebase system in OpenEvolve. The integration will enable the self-play system to leverage knowledge from various sources (documents, code repositories, specifications) to enhance the propose, solve, and verify phases.

## Integration Architecture

### Knowledgebase Components Overview
The existing knowledgebase system includes:
- **KnowledgeEngine**: Main facade for knowledge operations
- **CodeIndexer**: For indexing code repositories
- **Document loaders**: For processing various document formats
- **Verification backends**: For formal verification
- **Search engines**: Elasticsearch, Bedrock, EKS knowledge bases

### Self-Play Knowledge Integration Points

## 1. Knowledge-Enhanced Proposer

### Knowledge-Augmented Specification Generation
- **Purpose**: Use knowledgebase to inform specification generation with real-world context
- **Implementation**:
  - Query knowledgebase for similar specifications or problems
  - Extract patterns and common structures from existing codebases
  - Use retrieved knowledge to generate more realistic and diverse specifications

### New Functions
```python
async def generate_knowledge_enhanced_specification(
    self_play_manager: SelfPlayEvolutionManager,
    knowledge_engine: KnowledgeEngine,
    context_query: str,
    target_difficulty: str
) -> Specification:
    """
    Generate specifications using knowledgebase context
    """
    # Query knowledgebase for relevant specifications
    # Use retrieved examples to inform new specification generation
    # Apply difficulty targeting based on solver performance
    pass

def query_specification_patterns(
    knowledge_engine: KnowledgeEngine,
    category: str,
    difficulty: str
) -> List[Dict[str, Any]]:
    """
    Query knowledgebase for specification patterns
    """
    pass
```

### Knowledge Sources for Proposer
- Indexed code repositories containing verified specifications
- Documented code patterns and best practices
- Historical self-play iterations and their outcomes
- Domain-specific knowledge from external documents

## 2. Knowledge-Enhanced Solver

### Context-Aware Solution Generation
- **Purpose**: Use knowledgebase to inform solution generation with relevant context
- **Implementation**:
  - Retrieve similar solved problems for reference
  - Access code patterns and implementation strategies
  - Leverage formal verification examples and proof techniques

### New Functions
```python
async def solve_with_knowledge_context(
    specification: Specification,
    knowledge_engine: KnowledgeEngine,
    solver_model: Any
) -> SolverResult:
    """
    Generate solution using knowledgebase context
    """
    # Retrieve similar solved problems
    # Extract relevant code patterns and techniques
    # Generate solution with contextual guidance
    pass

def retrieve_solution_patterns(
    knowledge_engine: KnowledgeEngine,
    specification: Specification
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant solution patterns from knowledgebase
    """
    pass
```

## 3. Knowledge-Enhanced Verification

### Knowledge-Based Verification Enhancement
- **Purpose**: Use knowledgebase to enhance verification process
- **Implementation**:
  - Retrieve verification patterns and common proof strategies
  - Access historical verification results for similar specifications
  - Use knowledge to suggest proof annotations or verification hints

### New Functions
```python
def enhance_verification_with_knowledge(
    code: str,
    specification: Specification,
    knowledge_engine: KnowledgeEngine
) -> Dict[str, Any]:
    """
    Enhance verification process with knowledgebase insights
    """
    # Retrieve similar verification cases
    # Apply learned verification strategies
    # Suggest proof annotations based on patterns
    pass
```

## 4. Knowledgebase Indexing for Self-Play

### Self-Play Result Indexing
- **Purpose**: Store and index self-play results for future use
- **Implementation**:
  - Index verified solutions and their specifications
  - Store difficulty assessments and solver performance
  - Track evolution of specifications over iterations

### New Indexing Functions
```python
async def index_selfplay_results(
    knowledge_engine: KnowledgeEngine,
    results: List[VerificationResult],
    iteration: int
) -> bool:
    """
    Index self-play results for future retrieval
    """
    # Create index entries for verified solutions
    # Include metadata about difficulty and performance
    # Track iteration and evolution metrics
    pass

def index_specification_evolution(
    knowledge_engine: KnowledgeEngine,
    original_spec: Specification,
    evolved_specs: List[Specification],
    iteration: int
) -> bool:
    """
    Index the evolution of specifications over time
    """
    pass
```

## 5. Knowledge Graph Integration

### Entity Knowledge Graph for Self-Play
- **Purpose**: Use knowledge graphs to track relationships between specifications, solutions, and verification results
- **Implementation**:
  - Create nodes for specifications, solutions, and verification outcomes
  - Establish relationships between related problems and solutions
  - Track difficulty progression and solver improvement

### New Graph Functions
```python
def update_entity_knowledge_graph(
    entity_graph: EntityKnowledgeGraph,
    selfplay_results: List[VerificationResult]
) -> bool:
    """
    Update knowledge graph with self-play results
    """
    # Add nodes for specifications and solutions
    # Create relationships between related entities
    # Update difficulty and performance attributes
    pass
```

## 6. Search and Retrieval Integration

### Self-Play Specific Search
- **Purpose**: Enable targeted search for self-play relevant information
- **Implementation**:
  - Search for specifications of specific difficulty levels
  - Find similar problems based on verification patterns
  - Retrieve solutions that match current solver capabilities

### New Search Functions
```python
def search_selfplay_knowledge(
    knowledge_engine: KnowledgeEngine,
    query_type: str,  # specification, solution, verification_pattern
    difficulty: Optional[str] = None,
    category: Optional[str] = None,
    solver_capability: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Search for self-play relevant knowledge
    """
    pass

def find_adaptive_specifications(
    knowledge_engine: KnowledgeEngine,
    solver_performance: Dict[str, float],
    target_difficulty: str
) -> List[Specification]:
    """
    Find specifications adapted to current solver capabilities
    """
    pass
```

## 7. Knowledgebase Configuration for Self-Play

### Enhanced Configuration Parameters
Add knowledgebase-specific parameters to self-play configuration:
```python
# Knowledgebase Integration Parameters
selfplay_knowledge_enabled: bool = True
selfplay_knowledge_retrieval_limit: int = 10
selfplay_knowledge_relevance_threshold: float = 0.7
selfplay_knowledge_index_update_freq: int = 5  # iterations
selfplay_knowledge_sources: List[str] = None  # codebases, documents, etc.
selfplay_context_window_size: int = 2048  # tokens for context
```

## 8. Workflow Integration

### Knowledge-Enhanced Self-Play Workflow
```python
async def run_knowledge_enhanced_selfplay(
    seed_specifications: List[Specification],
    knowledge_engine: KnowledgeEngine,
    config: EvolutionConfiguration
) -> Dict[str, Any]:
    """
    Execute self-play with knowledgebase integration
    """
    # Initialize with seed specifications
    # For each iteration:
    #   1. Query knowledgebase for relevant context
    #   2. Propose new specifications with knowledge guidance
    #   3. Solve with knowledge-augmented context
    #   4. Verify with knowledge-enhanced strategies
    #   5. Index results for future retrieval
    #   6. Update knowledge graph
    pass
```

## 9. Performance and Caching

### Knowledgebase Caching for Self-Play
- **Purpose**: Cache frequently accessed knowledge to improve performance
- **Implementation**:
  - Cache specification patterns and solution templates
  - Store recent verification results
  - Cache difficulty assessments for common problem types

### New Caching Functions
```python
def setup_selfplay_caching(
    knowledge_engine: KnowledgeEngine,
    cache_config: Dict[str, Any]
) -> None:
    """
    Setup caching for self-play operations
    """
    pass

def cache_specification_context(
    knowledge_engine: KnowledgeEngine,
    specification: Specification,
    context: Dict[str, Any]
) -> None:
    """
    Cache context for specification
    """
    pass
```

## 10. Monitoring and Analytics

### Knowledgebase Integration Metrics
- Track knowledge retrieval effectiveness
- Monitor knowledgebase contribution to self-play performance
- Measure knowledge graph growth and quality
- Analyze knowledge source utilization

### New Metrics Functions
```python
def collect_knowledge_integration_metrics(
    knowledge_engine: KnowledgeEngine,
    selfplay_results: Dict[str, Any]
) -> Dict[str, float]:
    """
    Collect metrics on knowledgebase integration effectiveness
    """
    pass
```

## 11. Error Handling and Fallbacks

### Knowledgebase Error Handling
- Handle knowledgebase unavailability gracefully
- Implement fallback strategies when knowledge retrieval fails
- Maintain self-play functionality without knowledgebase

### New Error Handling Functions
```python
def handle_knowledge_retrieval_failure(
    error: Exception,
    fallback_strategy: str = "default_proposal"
) -> Any:
    """
    Handle knowledgebase retrieval failures
    """
    pass
```

## 12. Security and Access Control

### Knowledgebase Access for Self-Play
- Ensure proper authentication for knowledgebase access
- Implement access controls for sensitive knowledge
- Secure verification of external knowledge sources

## Implementation Priorities

### Phase 1: Basic Integration
1. Implement knowledge retrieval for proposer enhancement
2. Add basic indexing of self-play results
3. Create simple search functions for self-play context

### Phase 2: Advanced Features
1. Implement knowledge graph integration
2. Add caching mechanisms
3. Enhance verification with knowledgebase insights

### Phase 3: Optimization
1. Performance optimization for knowledge retrieval
2. Advanced analytics and monitoring
3. Security enhancements