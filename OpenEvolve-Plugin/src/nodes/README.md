# OpenEvolve Core Nodes

Production-ready TypeScript implementations of core OpenEvolve workflow nodes for BubbleLab integration.

## Overview

These nodes are the fundamental building blocks for OpenEvolve workflows in BubbleLab. Each node extends the `OpenEvolveBaseNode` class and implements specific functionality for problem-solving workflows.

## Auto-Registration

All core nodes are automatically registered when you import the nodes module:

```typescript
import { DecompositionNode, NodeRegistry } from '@openevolve/plugin/nodes';

// Already registered - just create it
const node = NodeRegistry.create('Decomposition', 'my-id', {
  config: { strategy: 'semantic' }
});

// Or use the convenience function
import { getNode } from '@openevolve/plugin/nodes';
const node = getNode('Decomposition', 'my-id');
```

### Manual Registration

If you create custom nodes, register them:

```typescript
import { MyCustomNode } from './MyCustomNode';
import { NodeRegistry } from '@openevolve/plugin/nodes';

NodeRegistry.register('MyCustom', MyCustomNode, {
  source: 'custom'
});
```

## Available Nodes

### 1. DecompositionNode

**Purpose:** Break down complex problems into smaller, manageable sub-problems with dependency tracking and quality scoring.

**Category:** `analysis`

**Strategies:**
- `semantic` - Decompose by meaning and concepts
- `complexity` - Decompose by complexity levels
- `hybrid` - Combine semantic and complexity approaches

**Features:**
- Automatic dependency graph generation
- Quality metrics calculation (completeness, clarity, feasibility)
- Topological sorting for execution order
- Critical path identification
- Configurable sub-problem limits

**Input:**
```typescript
{
  problem: string;              // Required: Problem statement
  context?: string;             // Optional: Additional context
  requirements?: string[];      // Optional: Requirements list
}
```

**Output:**
```typescript
{
  subProblems: SubProblem[];
  dependencyGraph: DependencyGraph;
  qualityMetrics: {
    completeness: number;
    clarity: number;
    feasibility: number;
    overall: number;
  };
  metadata: { /* ... */ };
}
```

**Example:**
```typescript
const node = new DecompositionNode('decomp-1', {
  strategy: 'semantic',
  maxSubProblems: 10,
  qualityThreshold: 0.7
});

const result = await node.executeWithHistory(
  { problem: 'Design a scalable web application architecture' },
  { environment: 'development', timestamp: new Date() }
);

console.log(result.data.subProblems);
```

---

### 2. SolutionNode

**Purpose:** Generate high-quality solutions using multiple strategies with iterative refinement and convergence tracking.

**Category:** `generation`

**Strategies:**
- `MAKER` - Methodical, Analytical, Knowledge-driven, Efficient, Robust
- `MCTS` - Monte Carlo Tree Search with exploratory path analysis
- `Evolutionary` - Iterative improvement with variation
- `Hybrid` - Combined approach leveraging all strategies

**Features:**
- Iterative solution generation until convergence
- Quality scoring across multiple dimensions
- Convergence detection and early stopping
- Solution caching for performance
- Alternative solution generation
- Timeout protection

**Input:**
```typescript
{
  problem: string;              // Required: Problem statement
  requirements?: string[];      // Optional: Requirements
  constraints?: string[];       // Optional: Constraints
  context?: string;             // Optional: Additional context
}
```

**Output:**
```typescript
{
  bestSolution: Solution;
  allSolutions: Solution[];
  convergenceMetrics: {
    iterations: number;
    qualityHistory: number[];
    convergenceRate: number;
    converged: boolean;
    finalQuality: number;
    bestIteration: number;
  };
  metadata: { /* ... */ };
}
```

**Example:**
```typescript
const node = new SolutionNode('solution-1', {
  strategy: 'Evolutionary',
  maxIterations: 10,
  qualityThreshold: 0.8,
  generateAlternatives: true,
  numAlternatives: 3
});

const result = await node.executeWithHistory(
  {
    problem: 'Implement user authentication',
    requirements: ['Secure', 'Scalable', 'User-friendly']
  },
  { environment: 'production', timestamp: new Date() }
);

console.log('Best solution:', result.data.bestSolution);
console.log('Quality score:', result.data.bestSolution.qualityScore);
```

---

### 3. VerificationNode

**Purpose:** Verify solutions against requirements and quality standards with comprehensive reporting.

**Category:** `verification`

**Checks:**
- `requirements` - Requirements coverage validation
- `quality` - Quality standards compliance
- `completeness` - Solution completeness assessment
- `correctness` - Correctness validation
- `consistency` - Internal consistency checking
- `feasibility` - Feasibility analysis

**Features:**
- Multi-dimensional quality metrics
- Requirements coverage analysis
- Issue identification by severity
- Automated suggestion generation
- Detailed verification reports
- Configurable thresholds

**Input:**
```typescript
{
  solution: string;             // Required: Solution to verify
  requirements: string[];       // Required: Requirements to verify against
  problem?: string;             // Optional: Original problem statement
  qualityStandards?: object;    // Optional: Custom quality standards
}
```

**Output:**
```typescript
{
  solutionId: string;
  overallScore: number;
  passed: boolean;
  checks: CheckResult[];
  requirements: {
    specified: string[];
    met: string[];
    partiallyMet: string[];
    notMet: string[];
    coverage: number;
  };
  qualityMetrics: {
    completeness: number;
    correctness: number;
    clarity: number;
    consistency: number;
    feasibility: number;
  };
  issues: {
    critical: string[];
    major: string[];
    minor: string[];
  };
  suggestions: string[];
  metadata: { /* ... */ };
}
```

**Example:**
```typescript
const node = new VerificationNode('verify-1', {
  threshold: 0.7,
  checks: ['all'],
  strictMode: false,
  generateSuggestions: true
});

const result = await node.executeWithHistory(
  {
    solution: 'Implementation of OAuth 2.0 with JWT tokens...',
    requirements: ['Secure authentication', 'Token refresh mechanism']
  },
  { environment: 'production', timestamp: new Date() }
);

console.log('Verification passed:', result.data.passed);
console.log('Overall score:', result.data.overallScore);
console.log('Issues:', result.data.issues);
console.log('Suggestions:', result.data.suggestions);
```

---

## Base Class: OpenEvolveBaseNode

All nodes extend this abstract base class which provides:

- **Standardized interface** with `execute()`, `validateInputs()`, and `getParameterSchema()`
- **Execution history tracking** for debugging and analysis
- **Error handling** with automatic error result creation
- **Configuration management** with getters and setters
- **Static metadata** (DISPLAY_NAME, DESCRIPTION, ICON, CATEGORY, VERSION)
- **Protected helper methods** for common operations

**Key Methods:**

```typescript
// Main execution method (must implement)
abstract async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;

// Validate inputs before execution (must implement)
abstract validateInputs(inputs: NodeInputs): ValidationError[];

// Get JSON Schema for parameters (must implement)
abstract getParameterSchema(): ParameterSchema;

// Execute with automatic history tracking
async executeWithHistory(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;

// Configuration management
getConfig(): NodeConfig;
setConfig(config: Partial<NodeConfig>): void;

// History management
getExecutionHistory(): NodeResult[];
clearHistory(): void;
```

---

## Usage Examples

### Basic Workflow: Decompose → Solve → Verify

```typescript
import { DecompositionNode, SolutionNode, VerificationNode } from './nodes';

// Step 1: Decompose the problem
const decomposer = new DecompositionNode('decomp-1', {
  strategy: 'semantic',
  maxSubProblems: 5
});

const { data: decompResult } = await decomposer.executeWithHistory(
  { problem: 'Build a real-time chat application' },
  { environment: 'development', timestamp: new Date() }
);

console.log(`Created ${decompResult.subProblems.length} sub-problems`);

// Step 2: Generate solutions for each sub-problem
const solver = new SolutionNode('solver-1', {
  strategy: 'Evolutionary',
  maxIterations: 5
});

for (const subProblem of decompResult.subProblems) {
  const { data: solutionResult } = await solver.executeWithHistory(
    {
      problem: subProblem.description,
      requirements: subProblem.requirements
    },
    { environment: 'development', timestamp: new Date() }
  );

  console.log(`Solution for ${subProblem.id}:`, solutionResult.bestSolution.qualityScore);
}

// Step 3: Verify the best solution
const verifier = new VerificationNode('verify-1', {
  threshold: 0.8,
  checks: ['all']
});

const { data: verifyResult } = await verifier.executeWithHistory(
  {
    solution: solutionResult.bestSolution.content,
    requirements: subProblem.requirements
  },
  { environment: 'development', timestamp: new Date() }
);

if (verifyResult.passed) {
  console.log('Solution verified successfully!');
} else {
  console.log('Issues found:', verifyResult.issues);
}
```

### Using the Node Registry

```typescript
import { NodeRegistry, getNode } from './nodes';

// Check what nodes are available
const allNodes = NodeRegistry.listAll();
console.log('Available nodes:', allNodes);

// Create a node through the registry
const node = getNode('Decomposition', 'my-decomposer', {
  config: {
    strategy: 'hybrid',
    maxSubProblems: 8
  }
});

// Use the node
const result = await node.executeWithHistory(
  { problem: 'Design an API gateway' },
  { environment: 'production', timestamp: new Date() }
);
```

### Custom Node Implementation

```typescript
import { OpenEvolveBaseNode, NodeInputs, NodeResult, ExecutionContext } from './nodes';

export class CustomNode extends OpenEvolveBaseNode {
  static readonly DISPLAY_NAME = 'My Custom Node';
  static readonly DESCRIPTION = 'Does something custom';
  static readonly ICON = 'custom';
  static readonly CATEGORY = 'custom';
  static readonly VERSION = '1.0.0';

  constructor(id: string, config = {}) {
    super(id, config);
  }

  async execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult> {
    try {
      // Your logic here
      const result = { /* ... */ };
      return this.createSuccessResult(result);
    } catch (error) {
      return this.createErrorResult(error);
    }
  }

  validateInputs(inputs: NodeInputs): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!inputs.requiredField) {
      errors.push({
        field: 'requiredField',
        message: 'This field is required',
        severity: 'error'
      });
    }

    return errors;
  }

  getParameterSchema(): ParameterSchema {
    return {
      type: 'object',
      properties: {
        param1: {
          type: 'string',
          description: 'First parameter',
          default: 'default-value'
        }
      },
      required: []
    };
  }
}

// Register the custom node
NodeRegistry.register('Custom', CustomNode);
```

---

## Configuration

All nodes support runtime configuration through the constructor:

```typescript
const node = new AnyNode('id', {
  // Node-specific configuration
  parameter1: 'value1',
  parameter2: 42,
  // ...
});

// Update configuration at runtime
node.setConfig({ parameter1: 'newValue' });

// Get current configuration
const config = node.getConfig();
```

---

## Error Handling

Nodes provide comprehensive error handling:

```typescript
const result = await node.executeWithHistory(inputs, context);

if (!result.success) {
  console.error('Execution failed:', result.error);

  // Error details are in metadata
  console.error('Error metadata:', result.metadata);
} else {
  console.log('Success:', result.data);
}
```

---

## Type Safety

Full TypeScript support with strict types:

```typescript
import type {
  DecompositionResult,
  SolutionResult,
  VerificationReport,
  SubProblem,
  Solution
} from './nodes';

// Type-safe result handling
const processDecomposition = (result: DecompositionResult) => {
  result.subProblems.forEach((sp: SubProblem) => {
    console.log(`${sp.title}: ${sp.complexity}`);
  });
};
```

---

## Performance Considerations

1. **Caching:** SolutionNode implements caching to avoid regenerating identical solutions
2. **Timeout Protection:** All nodes support timeout configuration
3. **Execution History:** Automatically limited to last 100 executions
4. **Memory Management:** Unneeded history can be cleared with `clearHistory()`

---

## Best Practices

1. **Always use `executeWithHistory()`** for automatic error handling and history tracking
2. **Validate configuration** before critical workflows using `validateInputs()`
3. **Monitor convergence** in iterative processes through `convergenceMetrics`
4. **Check verification reports** thoroughly before accepting solutions
5. **Clear history periodically** in long-running applications
6. **Use appropriate strategies** for your problem domain
7. **Leverage caching** for repeated similar problems

---

## Future Enhancements

Potential additions to the core nodes:

- **BatchNode** - Process multiple problems in parallel
- **OptimizationNode** - Optimize existing solutions
- **IntegrationNode** - Integrate multiple solutions
- **LearningNode** - Learn from past executions
- **CollaborationNode** - Multi-agent solution generation

---

## License

MIT License - See project root for details.

---

## Contributing

When contributing new nodes:

1. Extend `OpenEvolveBaseNode`
2. Implement all abstract methods
3. Add comprehensive JSDoc comments
4. Include usage examples
5. Update this README
6. Register in the node registry

---

## Support

For issues, questions, or contributions, please refer to the main project repository.
