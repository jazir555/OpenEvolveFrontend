# OpenEvolve BubbleLabs Integration - Technical Specification

## Overview

This document provides a comprehensive technical specification for integrating OpenEvolve with BubbleLabs, enabling intended end-to-end control (NOT yet implemented) of OpenEvolve workflows through the BubbleLabs UI. The integration will provide visualization, control, and management of OpenEvolve's sophisticated evolutionary computing workflows within the BubbleLabs workflow automation platform.

## Architecture Overview

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BubbleLabs UI Layer                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   Flow Studio   │  │  Bubble Nodes   │  │  Parameter      │              │
│  │   (React)      │  │   (ReactFlow)   │  │  Management     │              │
│  │                 │  │                 │  │                 │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │      API Gateway              │
                    │   (BubbleLabs API Server)     │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │      OpenEvolve Core          │
                    │      (Python Backend)         │
                    └───────────────────────────────┘
```

### Component Mapping

| OpenEvolve Component | BubbleLabs Equivalent | Purpose |
|---------------------|----------------------|---------|
| Content Analyzer | Bubble Node | Analyzes input content and extracts structured context |
| Problem Decomposer | Bubble Node | Breaks down problems into sub-problems |
| Sub-problem Solver | Bubble Node | Solves individual sub-problems |
| Final Verifier | Bubble Node | Verifies final solution quality |
| Team Manager | Flow Parameter | Manages AI team configurations |
| Gauntlet Manager | Flow Parameter | Manages evaluation criteria |
| Workflow State | ReactFlow Node | Maintains workflow execution state |

## Technical Implementation

### 1. Bubble Node Implementation

Create OpenEvolve-specific bubble nodes that map to the OpenEvolve workflow components:

#### OpenEvolve Content Analyzer Bubble

```typescript
// packages/bubble-core/src/bubbles/OpenEvolveContentAnalyzerBubble.ts
import { ServiceBubble, bubbleSchema } from '../bubble-core';
import { z } from 'zod';

export const OpenEvolveContentAnalyzerParamsSchema = z.object({
  content: z.string().describe('Content to analyze'),
  analysisType: z.enum(['text', 'code', 'document', 'protocol']).default('text'),
  language: z.string().optional().default('en'),
  analysisDepth: z.enum(['shallow', 'deep', 'comprehensive']).default('deep'),
  apiKey: z.string().describe('API key for LLM service'),
  model: z.string().optional().default('gpt-4o'),
  temperature: z.number().min(0).max(2).optional().default(0.7),
  maxTokens: z.number().optional().default(4096),
});

export class OpenEvolveContentAnalyzerBubble extends ServiceBubble<
  z.input<typeof OpenEvolveContentAnalyzerParamsSchema>,
  {
    analysis: string;
    extractedContext: Record<string, unknown>;
    contentSummary: string;
    recommendations: string[];
    confidence: number;
  }
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey';
  static readonly bubbleName = 'openevolve-content-analyzer';
  static readonly schema = OpenEvolveContentAnalyzerParamsSchema;

  protected async performAction() {
    // Make HTTP request to OpenEvolve backend
    const response = await fetch('http://localhost:8000/api/content-analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.params.apiKey}`,
      },
      body: JSON.stringify(this.params),
    });

    if (!response.ok) {
      throw new Error(`Content analysis failed: ${response.statusText}`);
    }

    return await response.json();
  }
}
```

#### OpenEvolve Problem Decomposer Bubble

```typescript
// packages/bubble-core/src/bubbles/OpenEvolveDecomposerBubble.ts
import { ServiceBubble } from '../bubble-core';
import { z } from 'zod';

export const OpenEvolveDecomposerParamsSchema = z.object({
  problemStatement: z.string().describe('Problem to decompose'),
  decompositionStrategy: z.enum(['functional', 'hierarchical', 'temporal', 'spatial']).default('functional'),
  maxSubProblems: z.number().min(1).max(20).optional().default(10),
  apiKey: z.string().describe('API key for LLM service'),
  model: z.string().optional().default('gpt-4o'),
  temperature: z.number().min(0).max(2).optional().default(0.7),
});

export class OpenEvolveDecomposerBubble extends ServiceBubble<
  z.input<typeof OpenEvolveDecomposerParamsSchema>,
  {
    subProblems: Array<{
      id: string;
      title: string;
      description: string;
      priority: number;
      dependencies: string[];
      estimatedEffort: number;
    }>;
    decompositionQuality: number;
    strategyApplied: string;
  }
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey';
  static readonly bubbleName = 'openevolve-decomposer';
  static readonly schema = OpenEvolveDecomposerParamsSchema;

  protected async performAction() {
    const response = await fetch('http://localhost:8000/api/decompose', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.params.apiKey}`,
      },
      body: JSON.stringify(this.params),
    });

    if (!response.ok) {
      throw new Error(`Problem decomposition failed: ${response.statusText}`);
    }

    return await response.json();
  }
}
```

#### OpenEvolve Solver Bubble

```typescript
// packages/bubble-core/src/bubbles/OpenEvolveSolverBubble.ts
import { ServiceBubble } from '../bubble-core';
import { z } from 'zod';

export const OpenEvolveSolverParamsSchema = z.object({
  subProblem: z.object({
    id: z.string(),
    title: z.string(),
    description: z.string(),
    priority: z.number(),
    dependencies: z.array(z.string()),
    estimatedEffort: z.number(),
  }),
  solutionApproach: z.enum(['evolution', 'algorithmic', 'ml', 'hybrid']).default('evolution'),
  qualityRequirements: z.object({
    performance: z.number().min(0).max(100).optional().default(80),
    readability: z.number().min(0).max(100).optional().default(70),
    efficiency: z.number().min(0).max(100).optional().default(75),
  }).optional().default({}),
  evolutionParams: z.object({
    maxIterations: z.number().min(1).max(1000).optional().default(100),
    populationSize: z.number().min(1).max(100).optional().default(50),
    temperature: z.number().min(0).max(2).optional().default(0.7),
    selectionPressure: z.number().min(0).max(2).optional().default(1.0),
  }).optional().default({}),
  apiKey: z.string().describe('API key for LLM service'),
  model: z.string().optional().default('gpt-4o'),
});

export class OpenEvolveSolverBubble extends ServiceBubble<
  z.input<typeof OpenEvolveSolverParamsSchema>,
  {
    solution: string;
    solutionQuality: number;
    evolutionMetrics: {
      bestFitness: number;
      avgFitness: number;
      diversity: number;
      convergence: number;
    };
    executionTime: number;
    solutionType: string;
  }
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey';
  static readonly bubbleName = 'openevolve-solver';
  static readonly schema = OpenEvolveSolverParamsSchema;

  protected async performAction() {
    const response = await fetch('http://localhost:8000/api/solve', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.params.apiKey}`,
      },
      body: JSON.stringify(this.params),
    });

    if (!response.ok) {
      throw new Error(`Problem solving failed: ${response.statusText}`);
    }

    return await response.json();
  }
}
```

#### OpenEvolve Verifier Bubble

```typescript
// packages/bubble-core/src/bubbles/OpenEvolveVerifierBubble.ts
import { ServiceBubble } from '../bubble-core';
import { z } from 'zod';

export const OpenEvolveVerifierParamsSchema = z.object({
  solution: z.string().describe('Solution to verify'),
  requirements: z.object({
    functional: z.array(z.string()).optional().default([]),
    nonFunctional: z.object({
      performance: z.object({ minThroughput: z.number().optional(), maxLatency: z.number().optional() }).optional(),
      security: z.array(z.string()).optional().default([]),
      reliability: z.number().min(0).max(100).optional().default(95),
    }).optional().default({}),
    compliance: z.array(z.string()).optional().default([]),
  }),
  verificationDepth: z.enum(['light', 'standard', 'comprehensive']).default('standard'),
  apiKey: z.string().describe('API key for LLM service'),
  model: z.string().optional().default('gpt-4o'),
  temperature: z.number().min(0).max(2).optional().default(0.3),
});

export class OpenEvolveVerifierBubble extends ServiceBubble<
  z.input<typeof OpenEvolveVerifierParamsSchema>,
  {
    verificationResult: {
      passed: boolean;
      confidence: number;
      issues: Array<{
        severity: 'critical' | 'high' | 'medium' | 'low';
        description: string;
        recommendation: string;
      }>;
      overallScore: number;
      detailedReport: string;
    };
    verificationMetrics: {
      completeness: number;
      accuracy: number;
      coverage: number;
    };
  }
> {
  static readonly service = 'openevolve';
  static readonly authType = 'apikey';
  static readonly bubbleName = 'openevolve-verifier';
  static readonly schema = OpenEvolveVerifierParamsSchema;

  protected async performAction() {
    const response = await fetch('http://localhost:8000/api/verify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.params.apiKey}`,
      },
      body: JSON.stringify(this.params),
    });

    if (!response.ok) {
      throw new Error(`Solution verification failed: ${response.statusText}`);
    }

    return await response.json();
  }
}
```

### 2. Bubble Node Registration

Register the OpenEvolve bubbles in the BubbleLabs system:

```typescript
// packages/bubble-core/src/bubbles/openevolve/index.ts
import { BubbleRegistry } from '../bubble-registry';
import { OpenEvolveContentAnalyzerBubble } from './OpenEvolveContentAnalyzerBubble';
import { OpenEvolveDecomposerBubble } from './OpenEvolveDecomposerBubble';
import { OpenEvolveSolverBubble } from './OpenEvolveSolverBubble';
import { OpenEvolveVerifierBubble } from './OpenEvolveVerifierBubble';

export function registerOpenEvolveBubbles() {
  BubbleRegistry.register('openevolve-content-analyzer', OpenEvolveContentAnalyzerBubble);
  BubbleRegistry.register('openevolve-decomposer', OpenEvolveDecomposerBubble);
  BubbleRegistry.register('openevolve-solver', OpenEvolveSolverBubble);
  BubbleRegistry.register('openevolve-verifier', OpenEvolveVerifierBubble);
}
```

### 3. OpenEvolve Workflow Bubble

Create a high-level bubble that encapsulates the entire OpenEvolve workflow:

```typescript
// packages/bubble-core/src/bubbles/OpenEvolveWorkflowBubble.ts
import { Bubble } from '../bubble-core';
import { z } from 'zod';
import { OpenEvolveContentAnalyzerBubble } from './OpenEvolveContentAnalyzerBubble';
import { OpenEvolveDecomposerBubble } from './OpenEvolveDecomposerBubble';
import { OpenEvolveSolverBubble } from './OpenEvolveSolverBubble';
import { OpenEvolveVerifierBubble } from './OpenEvolveVerifierBubble';

export const OpenEvolveWorkflowParamsSchema = z.object({
  problem: z.string().describe('The problem to solve'),
  workflowConfig: z.object({
    enableQualityDiversity: z.boolean().optional().default(false),
    enableMultiObjective: z.boolean().optional().default(false),
    enableAdversarial: z.boolean().optional().default(false),
    maxIterations: z.number().min(1).max(1000).optional().default(100),
    populationSize: z.number().min(1).max(100).optional().default(50),
    numIslands: z.number().min(1).max(10).optional().default(5),
    archiveSize: z.number().min(0).max(1000).optional().default(100),
  }).optional().default({}),
  contentAnalyzerConfig: OpenEvolveContentAnalyzerParamsSchema.partial().optional().default({}),
  decomposerConfig: OpenEvolveDecomposerParamsSchema.partial().optional().default({}),
  solverConfigs: z.array(OpenEvolveSolverParamsSchema.partial()).optional().default([]),
  verifierConfig: OpenEvolveVerifierParamsSchema.partial().optional().default({}),
  apiKey: z.string().describe('API key for LLM service'),
  model: z.string().optional().default('gpt-4o'),
});

export class OpenEvolveWorkflowBubble extends Bubble<
  z.input<typeof OpenEvolveWorkflowParamsSchema>,
  {
    finalSolution: string;
    workflowQuality: number;
    executionMetrics: {
      totalExecutionTime: number;
      totalTokensUsed: number;
      successRate: number;
      solutionQualityScore: number;
    };
    workflowSteps: Array<{
      step: string;
      status: 'completed' | 'failed' | 'skipped';
      result?: unknown;
      metrics?: Record<string, unknown>;
    }>;
  }
> {
  static readonly bubbleName = 'openevolve-full-workflow';
  static readonly schema = OpenEvolveWorkflowParamsSchema;

  protected async performAction() {
    const {
      problem,
      workflowConfig = {},
      contentAnalyzerConfig = {},
      decomposerConfig = {},
      solverConfigs = [],
      verifierConfig = {},
      apiKey,
      model
    } = this.params;

    const results = {
      workflowSteps: [] as Array<{
        step: string;
        status: 'completed' | 'failed' | 'skipped';
        result?: unknown;
        metrics?: Record<string, unknown>;
      }>,
      executionMetrics: {
        totalExecutionTime: 0,
        totalTokensUsed: 0,
        successRate: 0,
        solutionQualityScore: 0,
      }
    };

    const startTime = Date.now();

    try {
      // Step 1: Content Analysis
      const contentAnalyzer = new OpenEvolveContentAnalyzerBubble({
        ...contentAnalyzerConfig,
        content: problem,
        apiKey,
        model,
      });
      const contentAnalysis = await contentAnalyzer.action();
      results.workflowSteps.push({
        step: 'content-analysis',
        status: contentAnalysis.success ? 'completed' : 'failed',
        result: contentAnalysis.data,
      });

      if (!contentAnalysis.success) {
        throw new Error('Content analysis failed');
      }

      // Step 2: Problem Decomposition
      const decomposer = new OpenEvolveDecomposerBubble({
        ...decomposerConfig,
        problemStatement: problem,
        apiKey,
        model,
      });
      const decomposition = await decomposer.action();
      results.workflowSteps.push({
        step: 'decomposition',
        status: decomposition.success ? 'completed' : 'failed',
        result: decomposition.data,
      });

      if (!decomposition.success) {
        throw new Error('Problem decomposition failed');
      }

      // Step 3: Solve sub-problems
      const subProblems = decomposition.data?.subProblems || [];
      const solutions = [];
      for (let i = 0; i < subProblems.length; i++) {
        const solverConfig = i < solverConfigs.length ? solverConfigs[i] : {};
        const solver = new OpenEvolveSolverBubble({
          ...solverConfig,
          subProblem: subProblems[i],
          apiKey,
          model,
        });
        const solution = await solver.action();
        solutions.push(solution);
        
        results.workflowSteps.push({
          step: `solve-sub-problem-${i}`,
          status: solution.success ? 'completed' : 'failed',
          result: solution.data,
        });

        if (!solution.success) {
          throw new Error(`Sub-problem ${i} solving failed`);
        }
      }

      // Step 4: Final Verification
      const finalSolution = solutions.map(s => s.data?.solution || '').join('\n\n');
      const verifier = new OpenEvolveVerifierBubble({
        ...verifierConfig,
        solution: finalSolution,
        apiKey,
        model,
      });
      const verification = await verifier.action();
      results.workflowSteps.push({
        step: 'verification',
        status: verification.success ? 'completed' : 'failed',
        result: verification.data,
      });

      if (!verification.success) {
        throw new Error('Final verification failed');
      }

      // Calculate overall metrics
      const endTime = Date.now();
      results.executionMetrics.totalExecutionTime = (endTime - startTime) / 1000;
      
      // Calculate success rate
      const completedSteps = results.workflowSteps.filter(s => s.status === 'completed').length;
      results.executionMetrics.successRate = completedSteps / results.workflowSteps.length;
      
      // Calculate solution quality score from verification
      results.executionMetrics.solutionQualityScore = verification.data?.verificationResult?.overallScore || 0;

      return {
        finalSolution: finalSolution,
        workflowQuality: results.executionMetrics.solutionQualityScore,
        executionMetrics: results.executionMetrics,
        workflowSteps: results.workflowSteps,
      };
    } catch (error) {
      // Add error step
      results.workflowSteps.push({
        step: 'error',
        status: 'failed',
        result: { error: error.message },
      });
      
      const endTime = Date.now();
      results.executionMetrics.totalExecutionTime = (endTime - startTime) / 1000;
      
      throw new Error(`OpenEvolve workflow failed: ${error.message}`);
    }
  }
}
```

### 4. Frontend Integration Components

#### Bubble Node Component for OpenEvolve Workflows

```tsx
// apps/bubble-studio/src/components/OpenEvolveBubbleNode.tsx
import { memo } from 'react';
import type { BubbleNodeData } from './BubbleNode';
import { useExecutionStore } from '../stores/executionStore';
import { BUBBLE_COLORS, BADGE_COLORS } from './BubbleColors';
import { Handle, Position } from '@xyflow/react';

interface OpenEvolveBubbleNodeProps {
  data: BubbleNodeData;
}

function OpenEvolveBubbleNode({ data }: OpenEvolveBubbleNodeProps) {
  const { flowId, bubble, bubbleKey } = data;

  // Get execution state for this bubble
  const bubbleId = bubble.variableId ? String(bubble.variableId) : String(bubbleKey);
  const highlightedBubble = useExecutionStore(flowId, (s) => s.highlightedBubble);
  const bubbleWithError = useExecutionStore(flowId, (s) => s.bubbleWithError);
  const runningBubbles = useExecutionStore(flowId, (s) => s.runningBubbles);
  const completedBubbles = useExecutionStore(flowId, (s) => s.completedBubbles);

  const isHighlighted = highlightedBubble === bubbleKey || highlightedBubble === bubbleId;
  const hasError = bubbleWithError === bubbleId;
  const isExecuting = runningBubbles.has(bubbleId);
  const isCompleted = bubbleId in completedBubbles;

  // Determine bubble status colors
  let borderClass, bgClass, handleClass;
  
  if (isExecuting) {
    borderClass = BUBBLE_COLORS.RUNNING.border;
    bgClass = isHighlighted ? BUBBLE_COLORS.SELECTED.background : BUBBLE_COLORS.RUNNING.background;
    handleClass = BUBBLE_COLORS.RUNNING.handle;
  } else if (hasError) {
    borderClass = BUBBLE_COLORS.ERROR.border;
    bgClass = isHighlighted ? BUBBLE_COLORS.SELECTED.background : BUBBLE_COLORS.ERROR.background;
    handleClass = BUBBLE_COLORS.ERROR.handle;
  } else if (isCompleted) {
    borderClass = BUBBLE_COLORS.COMPLETED.border;
    bgClass = isHighlighted ? BUBBLE_COLORS.SELECTED.background : BUBBLE_COLORS.COMPLETED.background;
    handleClass = BUBBLE_COLORS.COMPLETED.handle;
  } else if (isHighlighted) {
    borderClass = BUBBLE_COLORS.SELECTED.border;
    bgClass = BUBBLE_COLORS.SELECTED.background;
    handleClass = BUBBLE_COLORS.SELECTED.handle;
  } else {
    borderClass = BUBBLE_COLORS.DEFAULT.border;
    bgClass = BUBBLE_COLORS.DEFAULT.background;
    handleClass = BUBBLE_COLORS.DEFAULT.handle;
  }

  return (
    <div
      className={`bg-neutral-800/90 rounded-lg border overflow-hidden transition-all duration-300 w-80 ${borderClass} ${bgClass}`}
    >
      <Handle
        type="target"
        position={Position.Left}
        id="left"
        className={`w-3 h-3 ${handleClass}`}
        style={{ left: -6 }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="right"
        className={`w-3 h-3 ${handleClass}`}
        style={{ right: -6 }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="bottom"
        className={`w-3 h-3 ${handleClass}`}
        style={{ bottom: -6 }}
      />

      <div className="p-4">
        <div className="flex items-center gap-2 mb-2">
          <div className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center">
            <span className="text-white text-sm">OE</span>
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-neutral-100 truncate">
              {bubble.variableName || bubble.bubbleName}
            </h3>
            <p className="text-xs text-neutral-400 truncate">
              {bubble.bubbleName}
            </p>
          </div>
        </div>

        {/* Parameters display */}
        {bubble.parameters.length > 0 && (
          <div className="mt-3 space-y-2">
            {bubble.parameters.map((param, idx) => (
              <div key={idx} className="text-xs">
                <span className="text-neutral-300">{param.name}:</span>
                <span className="text-neutral-400 ml-1">
                  {typeof param.value === 'string' && param.value.length > 30
                    ? `${param.value.substring(0, 27)}...`
                    : String(param.value)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default memo(OpenEvolveBubbleNode);
```

### 5. Backend API Integration

#### OpenEvolve API Proxy Service

```typescript
// apps/bubblelab-api/src/services/openevolve-service.ts
import { BubbleFlow } from '@bubblelab/shared-schemas';

interface OpenEvolveConfig {
  baseUrl: string;
  apiKey: string;
}

export class OpenEvolveService {
  private config: OpenEvolveConfig;

  constructor(config: OpenEvolveConfig) {
    this.config = config;
  }

  async executeWorkflow(flow: BubbleFlow, inputs: Record<string, unknown>) {
    // Map bubble flow to OpenEvolve workflow
    const openevolveWorkflow = this.mapFlowToOpenEvolve(flow, inputs);
    
    // Execute the workflow
    const response = await fetch(`${this.config.baseUrl}/api/execute-workflow`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({
        workflow: openevolveWorkflow,
        inputs: inputs,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenEvolve workflow execution failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async analyzeContent(content: string, options: Record<string, unknown>) {
    const response = await fetch(`${this.config.baseUrl}/api/content-analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({
        content,
        ...options,
      }),
    });

    if (!response.ok) {
      throw new Error(`Content analysis failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async decomposeProblem(problem: string, options: Record<string, unknown>) {
    const response = await fetch(`${this.config.baseUrl}/api/decompose`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({
        problemStatement: problem,
        ...options,
      }),
    });

    if (!response.ok) {
      throw new Error(`Problem decomposition failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async solveSubProblem(problem: Record<string, unknown>, options: Record<string, unknown>) {
    const response = await fetch(`${this.config.baseUrl}/api/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({
        subProblem: problem,
        ...options,
      }),
    });

    if (!response.ok) {
      throw new Error(`Problem solving failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async verifySolution(solution: string, requirements: Record<string, unknown>) {
    const response = await fetch(`${this.config.baseUrl}/api/verify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({
        solution,
        requirements,
      }),
    });

    if (!response.ok) {
      throw new Error(`Solution verification failed: ${response.statusText}`);
    }

    return await response.json();
  }

  private mapFlowToOpenEvolve(flow: BubbleFlow, inputs: Record<string, unknown>) {
    // Convert the BubbleFlow structure to OpenEvolve-compatible workflow
    // This involves mapping bubble nodes to OpenEvolve workflow steps
    
    const workflowSteps = flow.code.match(/new\s+\w+Bubble/g)?.map(step => {
      const bubbleName = step.replace('new ', '');
      return {
        type: bubbleName,
        parameters: this.extractParameters(flow.code, bubbleName),
      };
    }) || [];

    return {
      id: flow.id,
      name: flow.name,
      steps: workflowSteps,
      inputs: inputs,
    };
  }

  private extractParameters(code: string, bubbleName: string) {
    // Extract parameters from the bubble instantiation code
    // This is a simplified example - in practice you'd need more sophisticated parsing
    const regex = new RegExp(`${bubbleName}\\s*\\(\\s*({[\\s\\S]*?})\\s*\\)`, 'g');
    const match = regex.exec(code);
    if (match && match[1]) {
      try {
        return JSON.parse(match[1]);
      } catch (e) {
        console.warn(`Failed to parse parameters for ${bubbleName}:`, e);
        return {};
      }
    }
    return {};
  }
}
```

#### API Routes for OpenEvolve Integration

```typescript
// apps/bubblelab-api/src/routes/openevolve.ts
import { Hono } from 'hono';
import { OpenEvolveService } from '../services/openevolve-service';

const app = new Hono();

app.post('/openevolve/execute-workflow', async (c) => {
  try {
    const { workflow, inputs } = await c.req.json();
    
    const openevolveService = new OpenEvolveService({
      baseUrl: process.env.OPENEVOLVE_BASE_URL || 'http://localhost:8000',
      apiKey: process.env.OPENEVOLVE_API_KEY || 'your-default-key',
    });

    const result = await openevolveService.executeWorkflow(workflow, inputs);
    
    return c.json({ result, status: 'success' });
  } catch (error) {
    console.error('OpenEvolve workflow execution error:', error);
    return c.json({ error: error.message, status: 'error' }, 500);
  }
});

app.post('/openevolve/analyze-content', async (c) => {
  try {
    const { content, options } = await c.req.json();
    
    const openevolveService = new OpenEvolveService({
      baseUrl: process.env.OPENEVOLVE_BASE_URL || 'http://localhost:8000',
      apiKey: process.env.OPENEVOLVE_API_KEY || 'your-default-key',
    });

    const result = await openevolveService.analyzeContent(content, options);
    
    return c.json({ result, status: 'success' });
  } catch (error) {
    console.error('OpenEvolve content analysis error:', error);
    return c.json({ error: error.message, status: 'error' }, 500);
  }
});

app.post('/openevolve/decompose-problem', async (c) => {
  try {
    const { problem, options } = await c.req.json();
    
    const openevolveService = new OpenEvolveService({
      baseUrl: process.env.OPENEVOLVE_BASE_URL || 'http://localhost:8000',
      apiKey: process.env.OPENEVOLVE_API_KEY || 'your-default-key',
    });

    const result = await openevolveService.decomposeProblem(problem, options);
    
    return c.json({ result, status: 'success' });
  } catch (error) {
    console.error('OpenEvolve problem decomposition error:', error);
    return c.json({ error: error.message, status: 'error' }, 500);
  }
});

export default app;
```

### 6. Configuration Management

#### OpenEvolve Configuration Schema

```typescript
// packages/shared-schemas/src/openevolve-schemas.ts
import { z } from 'zod';

export const OpenEvolveConfigSchema = z.object({
  baseUrl: z.string().url('OpenEvolve base URL must be a valid URL'),
  apiKey: z.string().min(1, 'API key is required'),
  defaultModel: z.string().optional().default('gpt-4o'),
  defaultTemperature: z.number().min(0).max(2).optional().default(0.7),
  enableQualityDiversity: z.boolean().optional().default(false),
  enableMultiObjective: z.boolean().optional().default(false),
  enableAdversarial: z.boolean().optional().default(false),
  maxIterations: z.number().min(1).max(10000).optional().default(100),
  populationSize: z.number().min(1).max(1000).optional().default(50),
  numIslands: z.number().min(1).max(50).optional().default(5),
  migrationRate: z.number().min(0).max(1).optional().default(0.1),
  archiveSize: z.number().min(0).max(10000).optional().default(100),
  eliteRatio: z.number().min(0).max(1).optional().default(0.1),
  explorationRatio: z.number().min(0).max(1).optional().default(0.2),
  exploitationRatio: z.number().min(0).max(1).optional().default(0.7),
  checkpointInterval: z.number().min(1).max(1000).optional().default(10),
  featureDimensions: z.array(z.string()).optional().default(['complexity', 'diversity']),
  featureBins: z.number().min(1).max(100).optional().default(10),
  diversityMetric: z.enum(['edit_distance', 'cosine_similarity', 'levenshtein_distance']).optional().default('edit_distance'),
});

export type OpenEvolveConfig = z.infer<typeof OpenEvolveConfigSchema>;
```

### 7. Execution Monitoring and State Management

#### Bubble Execution Store Integration

```typescript
// apps/bubble-studio/src/stores/openevolveExecutionStore.ts
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

interface OpenEvolveExecutionState {
  flowId: number;
  // Evolution metrics
  bestFitness: number;
  avgFitness: number;
  diversity: number;
  generation: number;
  populationSize: number;
  // Quality-Diversity metrics
  archiveSize: number;
  coverage: number;
  // Performance metrics
  executionTime: number;
  tokensUsed: number;
  // Status
  isRunning: boolean;
  isConverged: boolean;
  error: string | null;
  // Actions
  setMetrics: (metrics: Partial<OpenEvolveExecutionState>) => void;
  startExecution: () => void;
  stopExecution: () => void;
  reset: () => void;
}

export const createOpenEvolveExecutionStore = (flowId: number) => 
  create<OpenEvolveExecutionState>()(
    subscribeWithSelector((set, get) => ({
      flowId,
      bestFitness: 0,
      avgFitness: 0,
      diversity: 0,
      generation: 0,
      populationSize: 0,
      archiveSize: 0,
      coverage: 0,
      executionTime: 0,
      tokensUsed: 0,
      isRunning: false,
      isConverged: false,
      error: null,

      setMetrics: (metrics) => set(metrics),

      startExecution: () => set({ isRunning: true, error: null }),

      stopExecution: () => set({ isRunning: false }),

      reset: () => set((state) => ({
        ...state,
        bestFitness: 0,
        avgFitness: 0,
        diversity: 0,
        generation: 0,
        archiveSize: 0,
        coverage: 0,
        executionTime: 0,
        tokensUsed: 0,
        isRunning: false,
        isConverged: false,
        error: null,
      })),
    }))
  );
```

### 8. Deployment and Configuration

#### Docker Configuration

```dockerfile
# Dockerfile for OpenEvolve-BubbleLabs Integration
FROM node:20-alpine AS bubblelab-builder

WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

# Copy monorepo files
COPY pnpm-lock.yaml ./
COPY turbo.json ./
COPY packages/ ./packages/
COPY apps/ ./apps/

# Install dependencies
RUN pnpm install --frozen-lockfile

# Build packages
RUN pnpm --filter "@bubblelab/*" build

# Build bubble-studio
RUN pnpm --filter bubble-studio build

# Build bubblelab-api
RUN pnpm --filter bubblelab-api build

# Production stage
FROM node:20-alpine AS bubblelab-prod

WORKDIR /app

RUN npm install -g pnpm

COPY --from=bubblelab-builder /app/node_modules ./node_modules
COPY --from=bubblelab-builder /app/packages ./packages
COPY --from=bubblelab-builder /app/apps ./apps

# Set environment variables
ENV NODE_ENV=production
ENV OPENEVOLVE_BASE_URL=http://openevolve-service:8000
ENV OPENEVOLVE_API_KEY=your-api-key

EXPOSE 3000 3001

CMD ["sh", "-c", "cd apps/bubblelab-api && npm run start & cd apps/bubble-studio && npm run preview"]
```

#### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: bubblelab
      POSTGRES_USER: bubblelab
      POSTGRES_PASSWORD: bubblelab
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  openevolve-service:
    image: openevolve:latest  # Your OpenEvolve service
    environment:
      - DATABASE_URL=postgresql://bubblelab:bubblelab@postgres:5432/bubblelab
    ports:
      - "8000:8000"
    depends_on:
      - postgres

  bubblelab-api:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://bubblelab:bubblelab@postgres:5432/bubblelab
      - OPENEVOLVE_BASE_URL=http://openevolve-service:8000
      - OPENEVOLVE_API_KEY=${OPENEVOLVE_API_KEY}
      - BUN_ENV=production
    ports:
      - "3001:3001"
    depends_on:
      - postgres
      - openevolve-service

  bubblelab-frontend:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - VITE_API_URL=http://localhost:3001
    ports:
      - "3000:3000"
    depends_on:
      - bubblelab-api

volumes:
  postgres_data:
```

### 9. Testing Strategy

#### Unit Tests for OpenEvolve Bubbles

```typescript
// packages/bubble-core/__tests__/openevolve-bubbles.test.ts
import { OpenEvolveContentAnalyzerBubble } from '../src/bubbles/OpenEvolveContentAnalyzerBubble';
import { OpenEvolveWorkflowBubble } from '../src/bubbles/OpenEvolveWorkflowBubble';

describe('OpenEvolve Bubbles', () => {
  describe('OpenEvolveContentAnalyzerBubble', () => {
    it('should analyze content successfully', async () => {
      const bubble = new OpenEvolveContentAnalyzerBubble({
        content: 'This is a test document to analyze',
        apiKey: 'test-api-key',
      });

      // Mock the API call
      const mockResponse = {
        analysis: 'Content analysis result',
        extractedContext: { key: 'value' },
        contentSummary: 'Test content summary',
        recommendations: ['Recommendation 1'],
        confidence: 0.95,
      };

      jest.spyOn(global, 'fetch').mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data).toEqual(mockResponse);
    });

    it('should handle API errors gracefully', async () => {
      const bubble = new OpenEvolveContentAnalyzerBubble({
        content: 'Test content',
        apiKey: 'test-api-key',
      });

      jest.spyOn(global, 'fetch').mockResolvedValue({
        ok: false,
        statusText: 'Not Found',
      } as Response);

      const result = await bubble.action();

      expect(result.success).toBe(false);
      expect(result.error).toContain('Content analysis failed');
    });
  });

  describe('OpenEvolveWorkflowBubble', () => {
    it('should execute full workflow successfully', async () => {
      const bubble = new OpenEvolveWorkflowBubble({
        problem: 'Test problem to solve',
        apiKey: 'test-api-key',
      });

      // Mock all API calls in the workflow
      jest.spyOn(global, 'fetch')
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ analysis: 'Test analysis' }),
        } as Response) // content analysis
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ subProblems: [{ id: '1', title: 'Sub-problem 1', description: 'Test', priority: 1, dependencies: [], estimatedEffort: 5 }] }),
        } as Response) // decomposition
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ solution: 'Test solution', solutionQuality: 0.95 }),
        } as Response) // solving
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ verificationResult: { passed: true, overallScore: 0.92 } }),
        } as Response); // verification

      const result = await bubble.action();

      expect(result.success).toBe(true);
      expect(result.data?.workflowSteps.length).toBeGreaterThan(0);
    });
  });
});
```

#### Integration Tests

```typescript
// apps/bubblelab-api/__tests__/openevolve-integration.test.ts
import { OpenEvolveService } from '../src/services/openevolve-service';

describe('OpenEvolve Integration', () => {
  let openevolveService: OpenEvolveService;

  beforeAll(() => {
    openevolveService = new OpenEvolveService({
      baseUrl: process.env.OPENEVOLVE_BASE_URL || 'http://localhost:8000',
      apiKey: process.env.OPENEVOLVE_API_KEY || 'test-key',
    });
  });

  it('should connect to OpenEvolve service', async () => {
    // Test connection by performing a simple content analysis
    const result = await openevolveService.analyzeContent('Test content', {});
    expect(result).toBeDefined();
  });

  it('should execute a workflow end-to-end', async () => {
    const workflow = {
      id: 1,
      name: 'test-workflow',
      steps: [
        { type: 'ContentAnalyzer', parameters: { content: 'Test content' } }
      ],
    };

    const result = await openevolveService.executeWorkflow(
      workflow as any,
      { testInput: 'value' }
    );
    
    expect(result).toBeDefined();
  });
});
```

## Implementation Roadmap

### Phase 1: Core Bubble Implementation (Week 1-2)
1. Create basic OpenEvolve bubble classes (Content Analyzer, Decomposer, Solver, Verifier)
2. Implement parameter validation schemas
3. Create API proxy service
4. Register bubbles in the BubbleLabs system
5. Basic unit tests

### Phase 2: Workflow Integration (Week 3-4)
1. Create high-level OpenEvolve workflow bubble
2. Implement state management for OpenEvolve execution
3. Add monitoring and metrics tracking
4. Create custom bubble node UI components
5. Integration tests

### Phase 3: UI Integration (Week 5-6)
1. Integrate OpenEvolve bubbles into Bubble Studio UI
2. Add parameter management for OpenEvolve-specific options
3. Create visualization for OpenEvolve workflow execution
4. Add real-time status updates
5. User documentation and examples

### Phase 4: Production Deployment (Week 7)
1. Containerize the integration
2. Set up production deployment pipeline
3. Performance optimization
4. Security hardening
5. Production testing

## Security Considerations

### API Security
- Use secure API keys with proper scopes
- Implement rate limiting for OpenEvolve API calls
- Validate all input parameters
- Encrypt sensitive data in transit and at rest

### Credential Management
- Store OpenEvolve API keys securely in BubbleLabs credential management
- Implement proper credential rotation
- Use scoped credentials with minimal required permissions

### Data Privacy
- Ensure compliance with data privacy regulations (GDPR, CCPA)
- Implement data retention policies
- Secure data transmission between BubbleLabs and OpenEvolve

## Performance Optimization

### Caching Strategy
- Cache OpenEvolve API responses when appropriate
- Cache bubble execution results for deterministic operations
- Implement connection pooling for database operations

### Resource Management
- Implement proper resource cleanup after bubble execution
- Use async/await patterns to prevent blocking operations
- Optimize API call batching where possible

### Monitoring and Observability
- Implement comprehensive logging
- Add performance metrics collection
- Create health check endpoints
- Set up alerting for critical failures

This technical specification provides a comprehensive framework for integrating OpenEvolve with BubbleLabs, enabling complete control and visualization of OpenEvolve workflows through the BubbleLabs UI. The implementation follows the existing BubbleLabs architecture patterns while providing the specialized functionality needed for OpenEvolve's evolutionary computing capabilities.