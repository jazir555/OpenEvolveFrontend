<<<<<<< HEAD
# MathSolver Module

Mathematical theorem proving and SMT solving integration for Iterative Studio.

## Overview

MathSolver provides automated mathematical reasoning capabilities by integrating:
- **Z3 SMT Solver**: For constraint satisfaction, arithmetic, and finite domain problems
- **Lean Theorem Prover**: For formal proofs and logical deduction
- **Unified Solver**: Intelligent solver selection with consensus validation

## Features

- 🔢 **Multi-Solver Support**: Z3, Lean, and Auto-select modes
- 🔍 **Knowledge Base**: Search previous solutions for similar problems
- 📊 **Real-time Results**: Live updates during solving with cancellation support
- ♿ **Accessible**: Full keyboard navigation and screen reader support
- 🛡️ **Error Boundaries**: Graceful error handling without app crashes
- 📱 **Responsive**: Works on desktop and mobile devices

## API Version

**Current**: 1.1.0 (aligned with Python backend)

## Documentation

Comprehensive documentation is available:

| Document | Purpose | Time |
|----------|---------|------|
| [QUICK_START.md](./QUICK_START.md) | Get up and running | 5 min |
| [API_REFERENCE.md](./API_REFERENCE.md) | Complete API documentation | Reference |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Problem solving guide | Reference |
| [DEVELOPMENT_HISTORY.md](./DEVELOPMENT_HISTORY.md) | Development journey | 20 min |
| [IMPLEMENTATION_COMPLETE.md](./IMPLEMENTATION_COMPLETE.md) | Completion report | 10 min |
| [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md) | Guide to all docs | 5 min |

**New users**: Start with [QUICK_START.md](./QUICK_START.md)

## Installation

The MathSolver module is included in Iterative Studio. No additional installation required.

### Backend Requirements

Requires Python backend running at `http://localhost:8000` with:
- Z3 SMT solver
- Lean 4 theorem prover
- FastAPI web framework

## Quick Start

```typescript
import { MathSolverCore, MathSolverUI } from './MathSolver';

// Create core instance
const core = new MathSolverCore();

// Create a problem
const problem = core.createProblem('Prove that n² ≥ 0 for all integers n');

// Solve with auto-selected solver
const result = await core.solve({
    problem,
    preferredSolver: 'auto',
    useKnowledgeBase: true
});

console.log(result.success); // true if solved
```

## React Component Usage

```tsx
import { MathSolverUI, MathSolverErrorBoundary } from './MathSolver';

function App() {
    return (
        <MathSolverErrorBoundary onReset={() => console.log('Reset')}>
            <MathSolverUI 
                initialProblem="x² + 3x + 2 = 0"
                onClose={() => console.log('Closed')}
            />
        </MathSolverErrorBoundary>
    );
}
```

## Configuration

### Default Configuration

```typescript
const DEFAULT_CONFIG = {
    autoSelectSolver: true,
    useKnowledgeBase: true,
    consensusLevel: 'confidence',
    explainResults: true,
    maxIterations: 3,
    defaultTimeout: 300,
    enableVerification: true
};
```

### Environment Variables

- `MATH_SOLVER_API_URL`: Backend API URL (default: `http://localhost:8000`)

## Graceful Degradation

MathSolver is designed to function fully even when the knowledge engine is unavailable. The system gracefully degrades by:

1. **Continuing without knowledge base search** - Direct solving still works
2. **Falling back to heuristic strategies** - When `get_strategy` fails, local heuristics are used
3. **Disabling learning** - Solutions aren't cached but solving continues
4. **UI indicators** - Clear status showing knowledge engine availability

### Detecting Knowledge Engine Status

```typescript
import { MathSolverCore, isKnowledgeEngineAvailable } from './MathSolver';

const core = new MathSolverCore();

// Check if knowledge engine is available
const available = await core.checkKnowledgeEngineAvailability();

// Or check without network call
const status = core.getKnowledgeEngineStatus();
console.log(status.available);  // boolean
console.log(status.lastChecked); // timestamp
```

### Handling Unavailability in UI

```tsx
import { MathSolverUI } from './MathSolver';

// The MathSolverUI automatically:
// - Shows knowledge engine status in header
// - Disables knowledge base checkbox when unavailable
// - Shows visual indicators (✓/✗) next to KB option
```

## API Reference

### MathSolverCore

Main class for mathematical problem solving.

#### Methods

- `createProblem(statement, options?)`: Create a new math problem
- `solve(options)`: Solve the current problem
- `cancelSolve()`: Cancel an in-progress solve operation
- `isSolving()`: Check if a solve operation is in progress
- `exportState()`: Export current state for persistence
- `importState(state)`: Import previously exported state
- `reset()`: Clear all state
- `checkBackendHealth()`: Check backend availability
- `checkKnowledgeEngineAvailability()`: Check if knowledge engine is available
- `isKnowledgeEngineAvailable()`: Get cached knowledge engine status
- `getKnowledgeEngineStatus()`: Get detailed knowledge engine status

#### Events

Subscribe to events using `on(event, callback)`:

- `problemCreated`: Emitted when a problem is created
- `solvingStarted`: Emitted when solving begins
- `solvingCompleted`: Emitted when solving finishes
- `solvingError`: Emitted when an error occurs
- `solvingCancelled`: Emitted when solving is cancelled
- `stateImported`: Emitted when state is imported
- `stateReset`: Emitted when state is reset

### Math Tools for Agentic Mode

```typescript
import { executeMathToolCall, isMathTool } from './MathSolver';

// Check if a tool is a math tool
if (isMathTool('solve_z3')) {
    // Execute math tool
    const result = await executeMathToolCall({
        type: 'solve_z3',
        content: '(declare-fun x () Int)(assert (> x 0))(check-sat)'
    });
}
```

Available tools:
- `solve_z3`: Z3 SMT solver
- `solve_lean`: Lean theorem prover
- `solve_unified`: Unified approach with consensus
- `search_math_knowledge`: Search knowledge base
- `get_strategy`: Get solving strategy recommendation
- `formalize_problem`: Get formalization guidance
- `explain_proof`: Explain a proof in natural language
- `verify_proof`: Verify proof correctness

## Keyboard Shortcuts

- `Ctrl+Enter`: Start solving
- `Escape`: Cancel solving (when in progress)

## Accessibility

- Full keyboard navigation support
- ARIA labels on all interactive elements
- Screen reader compatible
- High contrast mode support
- Focus management

## Error Handling

MathSolver includes comprehensive error handling:

1. **Error Boundary**: Catches React rendering errors
2. **API Error Handling**: Handles backend connection issues
3. **Version Checking**: Warns about API version mismatches
4. **Retry Logic**: Automatic retry for transient failures
5. **User Feedback**: Toast notifications for all operations

## Testing

Run the test suite:

```bash
# Run all MathSolver tests
npm test -- MathSolver

# Run integration tests
npm test -- MathSolver.integration

# Run unit tests
npm test -- MathSolver.unit
```

### Test Coverage

- Module exports
- Type definitions
- Core functionality
- Utility functions
- Event system
- State management
- Tool execution
- API type alignment

## Troubleshooting

### Backend Not Available

**Symptom**: "Backend unavailable" message
**Solution**: 
1. Ensure Python backend is running at `http://localhost:8000`
2. Check backend health: `curl http://localhost:8000/health`

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed solutions.

### Version Mismatch

**Symptom**: "Version mismatch" warning
**Solution**: Update frontend or backend to matching versions

### Solve Operation Hangs

**Symptom**: Solving never completes
**Solution**: 
1. Check network connection
2. Try cancelling (Escape) and retrying
3. Check backend logs

### Knowledge Engine Unavailable

**Symptom**: "Knowledge base unavailable" message, KB checkbox disabled
**Impact**: MathSolver continues to work without self-improving capabilities
**Solution**:
1. Check if knowledge base endpoint is accessible: `curl http://localhost:8000/knowledge/stats`
2. Restart backend if needed
3. MathSolver will function normally without knowledge base - this is a graceful degradation, not a failure

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MathSolverUI  │────▶│  MathSolverCore │────▶│  MathSolverAPI  │
│   (React Component)   │     (State Mgmt)       │   (HTTP Client)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Python Backend │
                                               │  (Z3 + Lean)    │
                                               └─────────────────┘
```

## Development History

The MathSolver was developed over 15 rounds of iterative refinement. See [DEVELOPMENT_HISTORY.md](./DEVELOPMENT_HISTORY.md) for:
- Round-by-round development log
- Critical bugs fixed (including setTimeout naming conflict and circular import)
- Lessons learned
- Technical decisions

## Contributing

When adding features to MathSolver:

1. Maintain API version compatibility
2. Add tests for new functionality
3. Update TypeScript types
4. Follow existing code style
5. Add JSDoc comments
6. Update documentation (see [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md))

## Implementation Status

✅ **Production Ready** - See [IMPLEMENTATION_COMPLETE.md](./IMPLEMENTATION_COMPLETE.md) for:
- Feature checklist
- Test coverage
- Performance metrics
- Sign-off status

## License

SPDX-License-Identifier: Apache-2.0
=======
# MathSolver Module

Mathematical theorem proving and SMT solving integration for Iterative Studio.

## Overview

MathSolver provides automated mathematical reasoning capabilities by integrating:
- **Z3 SMT Solver**: For constraint satisfaction, arithmetic, and finite domain problems
- **Lean Theorem Prover**: For formal proofs and logical deduction
- **Unified Solver**: Intelligent solver selection with consensus validation

## Features

- 🔢 **Multi-Solver Support**: Z3, Lean, and Auto-select modes
- 🔍 **Knowledge Base**: Search previous solutions for similar problems
- 📊 **Real-time Results**: Live updates during solving with cancellation support
- ♿ **Accessible**: Full keyboard navigation and screen reader support
- 🛡️ **Error Boundaries**: Graceful error handling without app crashes
- 📱 **Responsive**: Works on desktop and mobile devices

## API Version

**Current**: 1.1.0 (aligned with Python backend)

## Installation

The MathSolver module is included in Iterative Studio. No additional installation required.

### Backend Requirements

Requires Python backend running at `http://localhost:8000` with:
- Z3 SMT solver
- Lean 4 theorem prover
- FastAPI web framework

## Quick Start

```typescript
import { MathSolverCore, MathSolverUI } from './MathSolver';

// Create core instance
const core = new MathSolverCore();

// Create a problem
const problem = core.createProblem('Prove that n² ≥ 0 for all integers n');

// Solve with auto-selected solver
const result = await core.solve({
    problem,
    preferredSolver: 'auto',
    useKnowledgeBase: true
});

console.log(result.success); // true if solved
```

## React Component Usage

```tsx
import { MathSolverUI, MathSolverErrorBoundary } from './MathSolver';

function App() {
    return (
        <MathSolverErrorBoundary onReset={() => console.log('Reset')}>
            <MathSolverUI 
                initialProblem="x² + 3x + 2 = 0"
                onClose={() => console.log('Closed')}
            />
        </MathSolverErrorBoundary>
    );
}
```

## Configuration

### Default Configuration

```typescript
const DEFAULT_CONFIG = {
    autoSelectSolver: true,
    useKnowledgeBase: true,
    consensusLevel: 'confidence',
    explainResults: true,
    maxIterations: 3,
    defaultTimeout: 300,
    enableVerification: true
};
```

### Environment Variables

- `MATH_SOLVER_API_URL`: Backend API URL (default: `http://localhost:8000`)

## API Reference

### MathSolverCore

Main class for mathematical problem solving.

#### Methods

- `createProblem(statement, options?)`: Create a new math problem
- `solve(options)`: Solve the current problem
- `cancelSolve()`: Cancel an in-progress solve operation
- `isSolving()`: Check if a solve operation is in progress
- `exportState()`: Export current state for persistence
- `importState(state)`: Import previously exported state
- `reset()`: Clear all state
- `checkBackendHealth()`: Check backend availability

#### Events

Subscribe to events using `on(event, callback)`:

- `problemCreated`: Emitted when a problem is created
- `solvingStarted`: Emitted when solving begins
- `solvingCompleted`: Emitted when solving finishes
- `solvingError`: Emitted when an error occurs
- `solvingCancelled`: Emitted when solving is cancelled
- `stateImported`: Emitted when state is imported
- `stateReset`: Emitted when state is reset

### Math Tools for Agentic Mode

```typescript
import { executeMathToolCall, isMathTool } from './MathSolver';

// Check if a tool is a math tool
if (isMathTool('solve_z3')) {
    // Execute math tool
    const result = await executeMathToolCall({
        type: 'solve_z3',
        content: '(declare-fun x () Int)(assert (> x 0))(check-sat)'
    });
}
```

Available tools:
- `solve_z3`: Z3 SMT solver
- `solve_lean`: Lean theorem prover
- `solve_unified`: Unified approach with consensus
- `search_math_knowledge`: Search knowledge base
- `get_strategy`: Get solving strategy recommendation
- `formalize_problem`: Get formalization guidance
- `explain_proof`: Explain a proof in natural language
- `verify_proof`: Verify proof correctness

## Keyboard Shortcuts

- `Ctrl+Enter`: Start solving
- `Escape`: Cancel solving (when in progress)

## Accessibility

- Full keyboard navigation support
- ARIA labels on all interactive elements
- Screen reader compatible
- High contrast mode support
- Focus management

## Error Handling

MathSolver includes comprehensive error handling:

1. **Error Boundary**: Catches React rendering errors
2. **API Error Handling**: Handles backend connection issues
3. **Version Checking**: Warns about API version mismatches
4. **Retry Logic**: Automatic retry for transient failures
5. **User Feedback**: Toast notifications for all operations

## Testing

Run the test suite:

```bash
# Run all MathSolver tests
npm test -- MathSolver

# Run integration tests
npm test -- MathSolver.integration

# Run unit tests
npm test -- MathSolver.unit
```

### Test Coverage

- Module exports
- Type definitions
- Core functionality
- Utility functions
- Event system
- State management
- Tool execution
- API type alignment

## Troubleshooting

### Backend Not Available

**Symptom**: "Backend unavailable" message
**Solution**: 
1. Ensure Python backend is running at `http://localhost:8000`
2. Check backend health: `curl http://localhost:8000/health`

### Version Mismatch

**Symptom**: "Version mismatch" warning
**Solution**: Update frontend or backend to matching versions

### Solve Operation Hangs

**Symptom**: Solving never completes
**Solution**: 
1. Check network connection
2. Try cancelling (Escape) and retrying
3. Check backend logs

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MathSolverUI  │────▶│  MathSolverCore │────▶│  MathSolverAPI  │
│   (React Component)   │     (State Mgmt)       │   (HTTP Client)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Python Backend │
                                               │  (Z3 + Lean)    │
                                               └─────────────────┘
```

## Contributing

When adding features to MathSolver:

1. Maintain API version compatibility
2. Add tests for new functionality
3. Update TypeScript types
4. Follow existing code style
5. Add JSDoc comments
6. Update this README

## License

SPDX-License-Identifier: Apache-2.0
>>>>>>> 5eda1a20fcb6c8612f843e21628e85c5f3699f23
