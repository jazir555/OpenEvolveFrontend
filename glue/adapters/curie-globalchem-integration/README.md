# Curie-GlobalChem Integration

This integration enables Curie (the AI research experimentation agent) to leverage GlobalChem's comprehensive chemical knowledge graph for conducting chemistry-related experiments.

## Architecture

The integration follows the CLAUDE.md principles:

- **Zero Trust**: All inputs and outputs are validated
- **Anti-Hallucination**: Data integrity is verified through GlobalChem's curated datasets
- **Read-Only State**: GlobalChem's data remains unmodified
- **Idempotency**: Operations are safe to repeat
- **Configuration Explicitness**: All parameters are configurable via environment variables
- **UTC**: All timestamps are stored in UTC

## Components

### 1. CurieGlobalChemAdapter
The main adapter class that bridges Curie and GlobalChem systems, providing:

- Chemical name search capabilities
- Molecular property calculation
- Related chemicals discovery
- Chemistry experiment execution

### 2. Interface Functions
Provides a clean API for Curie to access chemistry knowledge:

- `search`: Find chemicals by name
- `properties`: Calculate molecular properties
- `related`: Find related chemicals
- `experiment`: Run chemistry-focused experiments

## Usage

### Direct Usage
```python
from curie_globalchem_adapter import CurieGlobalChemAdapter, create_curie_interface

# Initialize the adapter
adapter = CurieGlobalChemAdapter(config={'log_level': 'INFO'})

# Create the interface for Curie
chemistry_interface = create_curie_interface(adapter)

# Search for a chemical
result = chemistry_interface('search', chemical_name='aspirin')
print(result)

# Calculate properties
props = chemistry_interface('properties', smiles='CC(=O)OC1=CC=CC=C1C(=O)O')
print(props)

# Find related chemicals
related = chemistry_interface('related', chemical_name='aspirin', max_results=5)
print(related)
```

### Integration with Curie
The adapter can be integrated into Curie's workflow to enable chemistry-focused experiments:

```python
import curie
from curie_globalchem_adapter import CurieGlobalChemAdapter, create_curie_interface

# Initialize the chemistry interface
adapter = CurieGlobalChemAdapter()
chemistry_interface = create_curie_interface(adapter)

# Example: Curie could use chemistry_interface to gather chemical data
# before formulating hypotheses about molecular behavior
```

## Configuration

The adapter supports the following configuration options:

- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `max_results`: Maximum number of results to return
- `timeout_seconds`: Timeout for operations
- `enable_rdkit`: Enable RDKit-based property calculations

## Testing

Run the test suite:
```bash
python test_adapter.py
```

Run the probe to verify integration:
```bash
python probes/integration_probe.py
```

## Deployment

The integration can be deployed as a container using the provided Dockerfile:

```bash
docker build -t curie-globalchem-adapter .
docker run -d --name curie-globalchem curie-globalchem-adapter
```

## Security

This integration follows security best practices:

- No direct access to GlobalChem's internal data structures
- Input validation for all queries
- Read-only access to GlobalChem's data
- Proper error handling to prevent information disclosure