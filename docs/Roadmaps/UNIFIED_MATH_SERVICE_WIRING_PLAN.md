# Unified Math Service Wiring Plan

## Overview

This document outlines the comprehensive plan for migrating all components from direct LeanAideClient usage to the new UnifiedMathService. The UnifiedMathService combines CAV-NLP formalization with LeanAide verification capabilities, providing a single interface for all mathematical operations.

## Current Architecture

### Components Using LeanAideClient Directly

1. **`adversarial_unified.py`**
   - Uses LeanAideClient for proof verification via `execute_task`
   - Located in main adversarial framework

2. **`blue_team_solver_engine.py`**
   - Uses `translate_thm_detailed()` for theorem translation
   - Part of the blue team solution engine

3. **`unified_mcp_server.py`**
   - Multiple tool registrations using various LeanAideClient methods
   - Contains server-side MCP tools

4. **`verification_methods.py`**
   - Uses LeanAideClient for verification functionality
   - Core verification logic component

5. **`bubblelabs_nodes/` directory**
   - Multiple nodes using LeanAideClient directly:
     - `math_verification_pipeline_node.py`
     - `lean_proof_checking_node.py`
     - `lean_autoformalization_node.py`

## Migration Path

### Method Mapping

| Old LeanAideClient Method | New UnifiedMathService Method | Notes |
|---------------------------|-------------------------------|-------|
| `translate_thm(text)` | `formalize(text)` | Primary formalization method |
| `translate_def(text)` | `formalize(text)` | Same method handles both |
| `translate_thm_detailed(text, name)` | `formalize(text, context)` | Enhanced with context |
| `elaborate(code)` | `elaborate(code)` | Direct replacement |
| `verify(code)` | `verify(code)` | Direct replacement |
| `execute_task(task, data)` | Various specific methods | Use appropriate method |

### Code Migration Examples

#### A. Import Changes
```python
# OLD
from leanaide_client import LeanAideClient, LeanAideConfig

# NEW  
from openevolve.unified_math_service import (
    UnifiedMathService, 
    create_unified_math_service,
    FormalizationResult,
    VerificationResult
)
```

#### B. Initialization Changes
```python
# OLD
self.leanaide_client = LeanAideClient(config)

# NEW
self.math_service = create_unified_math_service(
    use_cav_nlp=True,
    use_leanaide=True
)
```

#### C. Method Call Changes
```python
# OLD
result = await self.leanaide_client.translate_thm(text)
lean_code = result.data["lean_code"]

# NEW
result = await self.math_service.formalize(text)
lean_code = result.code
```

## Wiring Requirements

### A. Core Components Migration

#### 1. `adversarial_unified.py`
- **Current**: Uses `LeanAideClient` for proof verification via `execute_task`
- **Migration**: Replace with `UnifiedMathService.verify()` method
- **Changes needed**:
  ```python
  # Replace import
  from openevolve.unified_math_service import create_unified_math_service
  
  # Replace initialization
  self.service = create_unified_math_service()
  
  # Replace usage
  result = await self.service.verify(proof_code)
  ```

#### 2. `blue_team_solver_engine.py`
- **Current**: Uses `translate_thm_detailed()` for theorem translation
- **Migration**: Replace with `service.formalize()`
- **Changes needed**:
  ```python
  # Replace import
  from openevolve.unified_math_service import create_unified_math_service
  
  # Replace usage
  result = await service.formalize(prompt)
  lean_code = result.code
  ```

#### 3. `unified_mcp_server.py`
- **Current**: Multiple tool registrations using various LeanAideClient methods
- **Migration**: Replace all tool implementations with UnifiedMathService equivalents
- **Changes needed**:
  ```python
  # Replace all tool implementations:
  # leanaide_translate_theorem -> service.formalize()
  # leanaide_translate_definition -> service.formalize()
  # leanaide_elaborate -> service.elaborate()
  # leanaide_verify -> service.verify()
  ```

#### 4. `verification_methods.py`
- **Current**: Uses LeanAideClient for verification
- **Migration**: Replace with `service.verify()` and `service.formalize()`

### B. Node Components Migration

#### 1. `bubblelabs_nodes/` directory
- **Current**: Multiple nodes using LeanAideClient directly
- **Migration**: Each node needs to be updated to use UnifiedMathService

### C. Bridge Components

#### 1. `openevolve/leanaide_cav_nlp_bridge.py`
- **Current**: Already partially integrated with UnifiedMathService
- **Status**: Good shape, mainly needs verification

## Dependencies to Ensure

### A. CAV-NLP Integration Components
- `openevolve/cav_nlp_integration/` - Must be properly available
- All sub-components: `flexible_semantic_parsing`, `dependency_dag`, `canonical_lean_generator`

### B. LeanAide Integration Components  
- `lean4_integration.py` - For verification capabilities
- `leanaide_client.py` - For fallback capabilities

### C. Error Handling Updates
- Update error handling to work with new result types
- `FormalizationResult` vs old dictionary responses
- `VerificationResult` vs old verification responses

## Testing Requirements

### A. Unit Tests
- Update all test files that mock or use LeanAideClient
- Files like `test_leanaide_client.py`, `test_leanaide_integration.py`, etc.

### B. Integration Tests  
- Verify all workflows still function with new service
- Test end-to-end scenarios

## Configuration Updates

### A. Update Configuration Classes
- Modify any config classes that reference LeanAideClient settings
- Update to reference UnifiedMathService configuration options

### B. Environment Compatibility
- Ensure backward compatibility where needed
- Provide migration utilities (already exists in `leanaide_cav_nlp_bridge.py`)

## Migration Strategy

### Phase 1: Core Services
1. Update `adversarial_unified.py`
2. Update `blue_team_solver_engine.py` 
3. Update `verification_methods.py`

### Phase 2: Server Components
1. Update `unified_mcp_server.py` tool registrations

### Phase 3: Node Components
1. Update all files in `bubblelabs_nodes/`

### Phase 4: Testing
1. Update all test files
2. Run comprehensive integration tests

## Benefits of Migration

1. **Unified Interface**: Single service for all mathematical operations
2. **Enhanced Capabilities**: Leverages CAV-NLP formalization alongside LeanAide verification
3. **Better Maintainability**: Centralized mathematical service reduces code duplication
4. **Improved Error Handling**: Consistent result types across all operations
5. **Future-Proof**: Architecture supports additional mathematical tools and services

## Risks and Mitigation

### Risks:
- Breaking changes to existing functionality
- Potential performance differences
- Integration issues with complex workflows

### Mitigation:
- Thorough testing of all migrated components
- Gradual rollout with fallback mechanisms
- Comprehensive error handling and logging
- Maintain backward compatibility where possible during transition