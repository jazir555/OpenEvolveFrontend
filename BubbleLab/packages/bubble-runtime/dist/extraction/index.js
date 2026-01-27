// Bubble parameter extraction logic
//
// This module provides comprehensive AST parsing and parameter extraction for BubbleLab.
// It uses @typescript-eslint/typescript-estree to parse TypeScript/JavaScript code and
// extract bubble instantiations, parameters, dependencies, and workflow information.
//
// Key features:
// - AST-based bubble detection from new XyzBubble(...) patterns
// - Parameter extraction from object literals, variables, and spread operators
// - Dependency graph construction with uniqueId and variableId tracking
// - Support for bubbles inside customTools (AI agent tools)
// - Per-invocation cloning for isolated bubble instances
// - Promise.all() parallel execution pattern detection
// - Workflow tree construction with control flow analysis
//
// Main export: BubbleParser class with parseBubblesFromAST() method
export * from './BubbleParser.js';
//# sourceMappingURL=index.js.map