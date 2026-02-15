# Architecture Decision Record: Curie-GlobalChem Integration

**Status:** Accepted
**Date:** 2026-02-12
**Context:** OpenEvolve Federation - Curie + GlobalChem Integration

---

## Context

**Curie** is an AI research experimentation agent that conducts autonomous experiments. **GlobalChem** is a comprehensive chemical knowledge graph with curated chemical data, molecular properties, and related chemical information.

The integration enables Curie to leverage GlobalChem's chemistry knowledge for conducting chemistry-related experiments and research.

### Key Challenges

1. **Domain Specialization** - Chemistry requires specialized knowledge (SMILES, molecular properties)
2. **Data Integrity** - GlobalChem's curated datasets must remain unmodified (Law of Untouchable DB)
3. **API Complexity** - Multiple GlobalChem operations (search, properties, related chemicals)
4. **Validation Requirements** - Chemical data must be validated against curated sources
5. **Performance** - Chemical property calculations can be computationally expensive

### Integration Requirements

1. **Zero Trust** - All inputs and outputs must be validated
2. **Anti-Hallucination** - Data integrity verified through GlobalChem's curated datasets
3. **Read-Only State** - GlobalChem's data remains unmodified
4. **Idempotency** - Operations safe to repeat
5. **Configuration Explicitness** - All parameters configurable via environment variables
6. **UTC** - All timestamps stored in UTC

---

## Decision

### Architecture Pattern: Chemistry Knowledge Sidecar

We chose a **Chemistry Knowledge Sidecar Pattern** with the following characteristics:

```
┌─────────────────────────────────────────────────────────────┐
│                     Curie Agent                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Curie-GlobalChem Adapter (ACL)          │  │
│  │  • Chemical name search                           │  │
│  │  • Molecular property calculation                │  │
│  │  • Related chemicals discovery                       │  │
│  │  • Chemistry experiment execution                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ HTTP API
┌─────────────────────────────────────────────────────────────┐
│                  GlobalChem Core                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Chemical Knowledge Graph                   │  │
│  │  • Chemical entities (nodes)                       │  │
│  │  • Molecular relationships (edges)                  │  │
│  │  • Property calculations                           │  │
│  │  • SMILES processing                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌──────────────────────────────────────┐
        │    Curated Chemistry Data       │
        │  (Immutable, Read-Only)        │
        └──────────────────────────────────────┘
```

### Key Design Choices

1. **Adapter Location**: `/glue/adapters/curie-globalchem-integration/`
   - Isolated from core-projects (Law of Air Gap)
   - Rewritten chemistry utilities in adapter layer
   - Canonical schema at `/glue/schemas/curie-globalchem-canonical.json`

2. **Interface Strategy**: Clean Python API for Curie
   - `search`: Find chemicals by name
   - `properties`: Calculate molecular properties
   - `related`: Find related chemicals
   - `experiment`: Run chemistry-focused experiments

3. **Data Flow**:
   ```
   Input (Canonical Format)
       --> CurieGlobalChemAdapter.validate()
       --> GlobalChem API call
       --> Response validation
       --> Canonical Format
       --> Output
   ```

4. **Validation Layer**: Multi-level validation
   - Input validation (SMILES syntax, chemical name format)
   - Output validation (against curated datasets)
   - Property validation (RDKit calculations)

---

## Alternatives Considered

### Alternative 1: Direct GlobalChem Import
**Rejected**: Violates Law of Air Gap, creates tight coupling to GlobalChem internals

### Alternative 2: Separate Chemistry Service
**Rejected**: Adds unnecessary infrastructure complexity, Python API is sufficient

### Alternative 3: In-Memory Chemistry Cache
**Rejected**: Duplicates GlobalChem storage, stale data issues, memory overhead

### Alternative 4: Direct RDKit Integration
**Rejected**: Bypasses GlobalChem's curated knowledge, loses relationship data

---

## Consequences

### Positive Benefits

1. **Domain Specialization** - Curie gains expert chemistry knowledge
2. **Data Integrity** - GlobalChem's curated datasets ensure accuracy
3. **Rich Relationships** - Access to chemical relationship graph
4. **Property Calculations** - RDKit-based molecular properties
5. **Experiment Support** - Chemistry-focused experiment templates
6. **Validation** - Anti-hallucination through curated sources

### Negative Tradeoffs

1. **Read-Only** - Cannot modify GlobalChem data (Law of Untouchable DB)
2. **API Latency** - HTTP calls add 50-200ms per operation
3. **Dependency** - Requires GlobalChem service to be available
4. **Complexity** - Additional adapter layer for translation
5. **Cost** - RDKit property calculations can be expensive

### Known Limitations

1. **No Write Access** - Curie cannot add new chemicals to GlobalChem
2. **Calculation Timeout** - Complex molecular properties can take >30s
3. **SMILES Validation** - Only supports standard SMILES (not all variants)
4. **Relationship Depth** - Related chemicals limited to 2-3 hops
5. **Property Coverage** - Not all properties available for all chemicals

---

## Implementation Details

### Core Components

#### 1. CurieGlobalChemAdapter
```python
class CurieGlobalChemAdapter:
    def __init__(self, config: Dict[str, Any])
    def search(self, chemical_name: str) -> List[Chemical]
    def properties(self, smiles: str) -> MolecularProperties
    def related(self, chemical_name: str, max_results: int) -> List[Chemical]
    def experiment(self, experiment_config: dict) -> ExperimentResult
```

**Capabilities**:
- Chemical name search with fuzzy matching
- Molecular property calculation (RDKit)
- Related chemicals discovery (graph traversal)
- Chemistry experiment execution

**Example**:
```python
adapter = CurieGlobalChemAdapter(config={'log_level': 'INFO'})

# Search for chemical
chemicals = adapter.search('aspirin')
print(chemicals[0].name)  # "Aspirin"
print(chemicals[0].smiles)  # "CC(=O)OC1=CC=CC=C1C(=O)O"

# Calculate properties
props = adapter.properties('CC(=O)OC1=CC=CC=C1C(=O)O')
print(props.molecular_weight)  # 180.16
print(props.logp)  # 1.19

# Find related chemicals
related = adapter.related('aspirin', max_results=5)
print([c.name for c in related])  # ["Salicylic acid", ...]
```

#### 2. Curie Interface Factory
```python
def create_curie_interface(adapter: CurieGlobalChemAdapter) -> Callable:
    """Create Curie-compatible interface function"""
    def interface(operation: str, **kwargs) -> dict:
        if operation == 'search':
            return adapter.search(kwargs.get('chemical_name'))
        elif operation == 'properties':
            return adapter.properties(kwargs.get('smiles'))
        elif operation == 'related':
            return adapter.related(kwargs.get('chemical_name'), kwargs.get('max_results', 5))
        elif operation == 'experiment':
            return adapter.experiment(kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    return interface
```

**Usage in Curie**:
```python
import curie
from curie_globalchem_adapter import CurieGlobalChemAdapter, create_curie_interface

# Initialize chemistry interface
adapter = CurieGlobalChemAdapter()
chemistry_interface = create_curie_interface(adapter)

# Use in Curie workflow
result = chemistry_interface('search', chemical_name='aspirin')
```

### API Endpoints

| Operation | Purpose | Timeout | Retry Strategy |
|-----------|---------|---------|----------------|
| `search` | Find chemicals by name | 5s | 3 attempts, exponential backoff |
| `properties` | Calculate molecular properties | 30s | 2 attempts, linear backoff |
| `related` | Find related chemicals | 10s | 3 attempts, exponential backoff |
| `experiment` | Run chemistry experiment | 60s | No retry (stateful) |

### Canonical Schema

#### Chemical Entry
```typescript
interface Chemical {
  id: string;                    // GlobalChem UUID
  name: string;                  // Preferred chemical name
  synonyms: string[];            // Alternative names
  smiles: string;                // SMILES notation
  inchi: string;                // InChI identifier
  inchi_key: string;             // InChI Key
  molecular_weight: number;       // g/mol
  formula: string;               // Molecular formula
  created_at?: string;           // UTC ISO-8601
}
```

#### Molecular Properties
```typescript
interface MolecularProperties {
  smiles: string;                // Input SMILES
  molecular_weight: number;      // g/mol
  logp: number;                 // Partition coefficient
  hbd: number;                  // Hydrogen bond donors
  hba: number;                  // Hydrogen bond acceptors
  tpsa: number;                 // Topological polar surface area
  rotatable_bonds: number;       // Count of rotatable bonds
  created_at: string;            // UTC ISO-8601
}
```

### Configuration Requirements

#### Environment Variables
```bash
# GlobalChem Configuration
GLOBALCHEM_API_URL=http://globalchem:8000    # GlobalChem API URL
GLOBALCHEM_TIMEOUT=30                        # Default timeout (seconds)
GLOBALCHEM_MAX_RESULTS=100                    # Max results per query

# RDKit Configuration
ENABLE_RDKIT=true                             # Enable RDKit calculations
RDKIT_TIMEOUT=30                              # Property calculation timeout
RDKIT_MAX_PROPS=50                           # Max properties per batch

# Adapter Configuration
CURIE_GLOBALCHEM_HOST=curie-globalchem-adapter  # Service name
CURIE_GLOBALCHEM_PORT=8001                       # HTTP port
CURIE_GLOBALCHEM_LOG_LEVEL=INFO                    # Logging level

# Validation
GLOBALCHEM_VALIDATE_SMILES=true              # Validate SMILES syntax
GLOBALCHEM_VALIDATE_PROPERTIES=true          # Validate calculated properties
```

#### Python Configuration
```python
config = {
    "log_level": "INFO",
    "max_results": 100,
    "timeout_seconds": 30,
    "enable_rdkit": True,
    "validation": {
        "validate_smiles": True,
        "validate_properties": True
    }
}
```

---

## Gotchas

### API Quirks Discovered

1. **SMILES Case Sensitivity**:
   - GlobalChem SMILES are case-sensitive (atoms vs bonds)
   - **Gotcha**: Lowercase 'c' means aromatic carbon
   - **Solution**: Always validate with RDKit before querying

2. **Property Calculation Timeout**:
   - Complex molecules can exceed RDKit timeout
   - **Gotcha**: No reliable timeout for RDKit calculations
   - **Solution**: Use process-level timeout via `asyncio.wait_for()`

3. **Related Chemicals Depth**:
   - Graph traversal can return 1000s of related chemicals
   - **Gotcha**: No automatic depth limiting
   - **Solution**: Always set `max_results` parameter

4. **Fuzzy Matching Precision**:
   - Chemical name search uses fuzzy matching (Levenshtein)
   - **Gotcha**: Can return false positives for short names
   - **Solution**: Post-process with similarity threshold >0.8

5. **RDKit Version**:
   - Property calculations vary by RDKit version
   - **Gotcha**: `LogP` values differ between RDKit 2023.09 and 2024.03
   - **Solution**: Pin RDKit version in requirements

### Version Requirements

| Component | Minimum Version | Recommended Version | Notes |
|-----------|----------------|---------------------|-------|
| GlobalChem | 1.0.0 | latest | Curated chemistry dataset |
| RDKit | 2023.09 | 2024.03+ | 2024 fixes property bugs |
| Python | 3.10 | 3.11+ | 3.11 improves RDKit performance |

### Non-Obvious Behaviors

1. **Stereochemistry Loss**:
   - SMILES to InChI conversion loses stereochemistry
   - **Gotcha**: `C[C@H](O)O` becomes canonical form without stereo
   - **Solution**: Store original SMILES alongside InChI

2. **Tautomerism**:
   - Same chemical can have multiple SMILES representations
   - **Gotcha**: Search may miss tautomers
   - **Solution**: Normalize to canonical SMILES before search

3. **Property Calculation Failures**:
   - Some molecules fail RDKit property calculation
   - **Gotcha**: RDKit raises exception, returns no error message
   - **Solution**: Try catch with graceful degradation

4. **Related Chemicals Directionality**:
   - Graph relationships are directed (A → B)
   - **Gotcha**: `related(A)` may not include `related(B)`
   - **Solution**: Bidirectional traversal for related queries

---

## Circuit Breaker Configuration

### Timeout Values
```python
TIMEOUTS = {
    "search": 5.0,            # seconds
    "properties": 30.0,         # seconds (RDKit is slow)
    "related": 10.0,           # seconds
    "experiment": 60.0          # seconds
}
```

### Retry Strategies

#### Exponential Backoff (Default)
```python
@retry(
    attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    exponential=2.0,
    jitter=0.1
)
async def search_with_retry(...):
    ...
```

**Usage**: Chemical search, related chemicals

#### Linear Backoff (Property Calculations)
```python
@retry(
    attempts=2,
    base_delay=2.0,
    max_delay=5.0,
    exponential=1.0      # linear
)
async def properties_with_retry(...):
    ...
```

**Usage**: Molecular property calculations (expensive)

#### No Retry (Experiments)
```python
# No retry decorator
async def execute_experiment(...):
    ...
```

**Usage**: Chemistry experiments (stateful)

### Failure Thresholds

```python
CIRCUIT_BREAKER = {
    "failure_threshold": 5,        # open after 5 failures
    "success_threshold": 2,        # close after 2 successes
    "timeout": 60.0,               # open state duration (seconds)
    "half_open_max_calls": 1       # test call in half-open state
}
```

**States**:
- **CLOSED**: Normal operation
- **OPEN**: Circuit tripped, use fallback
- **HALF_OPEN**: Test if GlobalChem recovered

**Triggers**:
- 5 consecutive failures (timeout, parse error)
- 3 consecutive RDKit calculation failures
- GlobalChem service unresponsive

---

## Security Considerations

### Input Validation

#### SMILES Validation
```python
def validate_smiles(smiles: str) -> bool:
    """Validate SMILES syntax"""
    # Check length
    if len(smiles) > 500:
        raise ValueError("SMILES too long")

    # Check for invalid characters
    valid_chars = set('CNOPSFClBrI()[]=#@+-.0123456789')
    if not all(c in valid_chars for c in smiles):
        raise ValueError("Invalid characters in SMILES")

    # Validate with RDKit
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        # RDKit not available, basic validation only
        return True
```

#### Chemical Name Validation
```python
def validate_chemical_name(name: str) -> bool:
    """Sanitize chemical name input"""
    # Max length
    if len(name) > 200:
        raise ValueError("Chemical name too long")

    # Block shell commands
    shell_indicators = ["; rm", "| rm", "$(", "`"]
    if any(indicator in name for indicator in shell_indicators):
        raise ValueError("Shell commands not allowed")

    return True
```

### Data Privacy

**Chemical data is not sensitive** - Public scientific knowledge

```python
# OK to log chemical names
logger.info(f"Searching for chemical: {chemical_name}")

# OK to log SMILES
logger.info(f"Calculating properties for SMILES: {smiles}")

# OK to log results
logger.info(f"Found {len(results)} related chemicals")
```

---

## Testing Strategy

### 1. Probes (Before Implementation)

```bash
# Verify GlobalChem API
python probes/check_globalchem_api.sh

# Verify chemical search
python probes/check_chemical_search.sh

# Verify property calculations
python probes/check_property_calculations.sh
```

### 2. Contract Tests (On Every Deploy)

```bash
npm run test:contract
```

Tests validate:
- Chemical search response structure
- Molecular property calculations
- Related chemicals discovery
- SMILES validation
- Error handling

### 3. Integration Tests

```python
from curie_globalchem_adapter import CurieGlobalChemAdapter, create_curie_interface

# Initialize adapter
adapter = CurieGlobalChemAdapter(config={'log_level': 'INFO'})

# Test chemical search
chemicals = adapter.search('aspirin')
assert len(chemicals) > 0
assert chemicals[0].name == 'Aspirin'

# Test properties
props = adapter.properties('CC(=O)OC1=CC=CC=C1C(=O)O')
assert props.molecular_weight == 180.16

# Test related chemicals
related = adapter.related('aspirin', max_results=5)
assert len(related) <= 5
```

---

## Federation Constitution Compliance Checklist

- ✅ **Law of Air Gap**: No imports from `core-projects/`
- ✅ **Law of Runtime Truth**: Probes verify API before use
- ✅ **Law of Untouchable DB**: Read-only access to GlobalChem data
- ✅ **Law of Idempotency**: All operations safe to retry (search, properties)
- ✅ **Law of Configuration Explicitness**: All required env vars validated
- ✅ **Law of UTC**: All timestamps in UTC ISO-8601

---

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy adapter code
COPY . .

# Expose port
EXPOSE 8001

# Run adapter
CMD ["python", "-m", "curie_globalchem_adapter"]
```

```bash
# Build and run
docker build -t curie-globalchem-adapter .
docker run -d --name curie-globalchem curie-globalchem-adapter
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: curie-globalchem-adapter
spec:
  replicas: 3
  selector:
    matchLabels:
      app: curie-globalchem-adapter
  template:
    metadata:
      labels:
        app: curie-globalchem-adapter
    spec:
      containers:
      - name: adapter
        image: curie-globalchem-adapter:latest
        ports:
        - containerPort: 8001
        env:
        - name: GLOBALCHEM_API_URL
          value: "http://globalchem:8000"
        - name: ENABLE_RDKIT
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

---

## References

- [Curie Documentation](https://github.com/EricLBuehler/curie)
- [GlobalChem Documentation](https://github.com/Sulstice/globalchem)
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [Integration README](./README.md)
- [Federation Constitution](../../../../CLAUDE.md)

---

**Created**: 2026-02-12
**Author**: OpenEvolve Architecture Team
**Status**: Accepted, Implemented
**Last Updated**: 2026-02-12
