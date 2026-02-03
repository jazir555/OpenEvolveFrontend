# AST Parsing Implementation for BubbleLab

## Overview

The AST parsing and parameter extraction system for BubbleLab is **fully implemented** in `BubbleParser.ts`. This document provides a comprehensive overview of the implementation.

## Technology Stack

- **Parser**: `@typescript-eslint/typescript-estree` (v8.43.0)
- **Scope Analysis**: `@bubblelab/ts-scope-manager`
- **Language**: TypeScript/JavaScript
- **Target**: ES2022+ with full TypeScript support

## Implementation Status

### ✅ Completed Features

1. **AST Parsing**
   - Full TypeScript/JavaScript parsing using `@typescript-eslint/typescript-estree`
   - Source location tracking (line/column numbers)
   - Range tracking for substring extraction
   - Support for modern ES2022+ features

2. **Bubble Detection**
   - `new XyzBubble(...)` pattern matching
   - Support for `.action()` chaining
   - `await` expression handling
   - Anonymous bubble detection
   - Variable declaration tracking

3. **Parameter Extraction**
   - Object literal properties: `new Bubble({ foo: bar })`
   - Single variable parameter: `new Bubble(config)`
   - Spread operators: `new Bubble({ ...params })`
   - Nested objects and arrays
   - Template literals
   - Environment variables: `process.env.XYZ`
   - Member expressions: `foo.bar.baz`
   - Type annotations with non-null assertions

4. **Type Detection**
   - String literals
   - Number literals
   - Boolean literals
   - Array expressions
   - Object expressions
   - Template literals
   - Environment variables
   - Variable references
   - Complex expressions (calculated at runtime)

5. **Dependency Analysis**
   - Flat dependency extraction from bubble metadata
   - Detailed dependency graph construction
   - Circular dependency detection
   - Tool dependency tracking for AI agents
   - Per-instance dependency tracking

6. **Workflow Analysis**
   - Method invocation tracking with ordinals
   - Control flow detection (if/else, switch)
   - Loop detection (for, while, do-while)
   - Try-catch-finally blocks
   - Promise.all() parallel execution patterns
   - Transformation function identification
   - Variable declaration blocks
   - Return statement tracking

7. **Advanced Features**
   - Custom tools detection within AI agents
   - Bubble containment tracking (isInsideCustomTool)
   - Per-invocation cloning with hashed variableIds
   - Dependency graph cloning with uniqueId suffixing
   - Comment extraction for bubble documentation
   - Variable reference resolution through scope manager
   - JSON Schema generation for payload types

## Key Classes and Methods

### BubbleParser

Main parser class that orchestrates AST analysis.

#### Public Methods

```typescript
// Main entry point - parse bubbles from AST
parseBubblesFromAST(
  bubbleFactory: BubbleFactory,
  ast: TSESTree.Program,
  scopeManager: ScopeManager
): {
  bubbles: Record<number, ParsedBubbleWithInfo>;
  workflow: ParsedWorkflow;
  instanceMethodsLocation: Record<string, MethodInfo>;
}

// Extract JSON Schema for payload parameter
getPayloadJsonSchema(ast: TSESTree.Program): Record<string, unknown> | null
```

#### Private AST Traversal Methods

```typescript
// Visit all nodes in AST recursively
visitNode(node, nodes, classNameLookup, scopeManager): void

// Extract bubble from expression (handles await, action(), etc.)
extractBubbleFromExpression(expr, classNameLookup): ParsedBubbleWithInfo | null

// Parse new XyzBubble(...) expression
extractFromNewExpression(newExpr, classNameLookup): ParsedBubbleWithInfo | null

// Extract parameter value and type
extractParameterValue(expression): { value, type }
```

#### Parameter Extraction Details

The `extractParameterValue()` method handles:

```typescript
// Type detection priority:
1. TSNonNullExpression (process.env.FOO!)
2. MemberExpression (foo.bar)
3. ChainExpression (foo?.bar)
4. Identifier (variableName)
5. Literal (string, number, boolean)
6. TemplateLiteral (`foo ${bar}`)
7. ArrayExpression ([1, 2, 3])
8. ObjectExpression ({ foo: bar })
9. Complex expressions (anything else)
```

#### Dependency Graph Construction

```typescript
buildDependencyGraph(
  bubbleName: BubbleName,
  bubbleFactory: BubbleFactory,
  seen: Set<BubbleName>,
  toolsForThisNode?: BubbleName[],
  parentUniqueId: string = '',
  ordinalCounters: Map<string, number>,
  usedVariableIds: Set<number>,
  explicitVariableId?: number,
  suppressSelfSegment: boolean = false,
  instanceVariableName?: string
): DependencyGraphNode
```

Features:
- Cycle detection
- Ordinal-based numbering (e.g., `ai-agent#1`, `ai-agent#2`)
- uniqueId construction: `parent.child#ordinal`
- variableId hashing from uniqueId
- Tool dependency inclusion for AI agents

#### Workflow Analysis Methods

```typescript
buildWorkflowTree(ast, nodes, scopeManager): ParsedWorkflow

// Pattern detection:
- handle() method extraction
- Method invocation tracking
- Promise.all() parallel execution
- Control flow analysis
- Variable declaration blocks
- Transformation functions
```

#### Custom Tools Handling

```typescript
findCustomToolsInAIAgentBubbles(ast, nodes, classNameLookup): void
markBubblesInsideCustomTools(nodes): void
```

Detects AI agent custom tools and marks bubbles declared within them.

#### Per-Invocation Cloning

```typescript
cloneBubbleForInvocation(bubble, callSiteKey, bubbleSourceMap): number
cloneDependencyGraphNodeForInvocation(node, callSiteKey): DependencyGraphNode
```

Creates isolated copies of bubbles for each method call site with:
- Hashed variableIds: `hashToVariableId("421:runDentalAssistant#1")`
- Suffixed uniqueIds: `"421@runDentalAssistant#1"`
- Cloned dependency graphs

## Data Structures

### ParsedBubbleWithInfo

```typescript
interface ParsedBubbleWithInfo {
  variableId: number;                    // Unique numeric ID
  variableName: string;                  // Variable name in code
  bubbleName: BubbleName;                // 'ai-agent', 'slack', etc.
  className: string;                     // 'AIAgentBubble', 'SlackBubble', etc.
  parameters: BubbleParameter[];         // Extracted parameters
  hasAwait: boolean;                     // Uses await?
  hasActionCall: boolean;                // Has .action() call?
  nodeType: BubbleNodeType;              // 'service' | 'tool' | 'trigger'
  location: SourceLocation;              // Line/column numbers
  description?: string;                  // Comment above bubble
  dependencies?: BubbleName[];           // Flat dependency list
  dependencyGraph?: DependencyGraphNode; // Hierarchical graph
  isInsideCustomTool?: boolean;          // Inside AI agent tool?
  containingCustomToolId?: string;       // Tool ID if inside
  clonedFromVariableId?: number;         // Original ID if clone
  invocationCallSiteKey?: string;        // Call site if clone
}
```

### BubbleParameter

```typescript
interface BubbleParameter {
  name: string;                          // Parameter name
  value: string | number | boolean | ...; // Parameter value
  type: BubbleParameterType;             // 'string', 'number', 'env', etc.
  location?: SourceLocation;             // Value location in code
  source: 'object-property' | 'first-arg' | 'spread'; // Where it came from
  variableId?: number;                   // If value references a variable
}
```

### DependencyGraphNode

```typescript
interface DependencyGraphNode {
  name: BubbleName;                      // Bubble name
  uniqueId: string;                      // Hierarchical ID with ordinals
  variableId: number;                    // Numeric ID
  variableName?: string;                 // Instance variable name
  nodeType: BubbleNodeType;              // 'service' | 'tool' | 'trigger'
  dependencies: DependencyGraphNode[];   // Child dependencies
  functionCallChildren?: FunctionCallWorkflowNode[]; // Custom tool calls
}
```

## Edge Cases Handled

### 1. Invalid Syntax
- Parser throws errors for invalid TypeScript/JavaScript
- Graceful handling of missing AST nodes

### 2. Missing Parameters
- Bubbles with no parameters: `new Bubble()`
- Parameters with undefined values

### 3. Type Annotations
- Full TypeScript type annotation support
- Generic type handling
- Union and intersection types
- Indexed access types (e.g., `Registry['event']`)

### 4. Default Values
- Function parameter defaults
- Destructuring defaults
- Object property defaults

### 5. Complex Expressions
- Nested member expressions: `a.b.c.d`
- Optional chaining: `a?.b?.c`
- Non-null assertions: `process.env.FOO!`
- Template literals with interpolation

### 6. Scope Management
- Variable shadowing
- Closure variable capture
- Block-scoped variables (let/const)
- Module-level imports

## Test Coverage

The implementation includes comprehensive tests for:

1. **Basic Bubble Parsing**
   - `new Bubble({ foo: 'bar' })`
   - `new Bubble(config)`
   - `new Bubble({ ...params })`

2. **Complex Patterns**
   - Bubbles inside Promise.all()
   - Bubbles in .map() callbacks
   - Bubbles in async functions
   - Bubbles with .action() chaining

3. **Dependency Analysis**
   - Flat dependency extraction
   - Hierarchical graph construction
   - Tool dependency tracking
   - Circular dependency detection

4. **Workflow Analysis**
   - Method invocation tracking
   - Parallel execution patterns
   - Control flow structures
   - Transformation functions

5. **Edge Cases**
   - Comments above bubbles
   - Nested custom tools
   - Per-invocation cloning
   - Variable reference resolution

## Performance Considerations

1. **Caching**
   - AST is cached after first parse
   - Scope manager is reused
   - Invocation clone cache prevents duplicate clones

2. **Efficient Traversal**
   - Single-pass AST visiting
   - Early termination on bubble detection
   - Optimized node type checks

3. **Memory Management**
   - Maps for O(1) lookups
   - Sets for cycle detection
   - Sparse arrays for ordinal counting

## Future Enhancements

Potential areas for improvement:

1. **Incremental Parsing**
   - Re-parse only changed portions
   - Maintain parse tree across edits

2. **Better Error Recovery**
   - Continue parsing after errors
   - Provide error location details

3. **Advanced Type Inference**
   - Track generic type parameters
   - Infer types from usage

4. **Performance Profiling**
   - Measure parse times
   - Identify bottlenecks

## Conclusion

The AST parsing and parameter extraction system is production-ready with comprehensive coverage of:

- ✅ AST parsing with `@typescript-eslint/typescript-estree`
- ✅ Parameter extraction from all common patterns
- ✅ Dependency graph construction
- ✅ Workflow analysis
- ✅ Per-invocation cloning
- ✅ Edge case handling
- ✅ Comprehensive test coverage

The TODO comment in `index.ts` has been updated to reflect the completed implementation status.
