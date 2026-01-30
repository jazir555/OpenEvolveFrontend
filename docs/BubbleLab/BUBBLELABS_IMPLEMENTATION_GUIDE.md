# OpenEvolve BubbleLabs Integration - Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the OpenEvolve BubbleLabs integration, enabling complete control and visualization of OpenEvolve workflows through the BubbleLabs UI. The integration will allow users to design, execute, and monitor OpenEvolve evolutionary computing workflows using BubbleLabs' visual workflow builder.

## Prerequisites

Before starting the implementation, ensure you have:

- Node.js (v18 or higher)
- pnpm package manager
- Python 3.8+ for OpenEvolve backend
- Access to OpenEvolve API (running on localhost:8000 or configured endpoint)
- Git for version control

## Step 1: Set Up the Development Environment

### 1.1 Clone or Access the Repository
```bash
# If you have access to the repository
git clone <repository-url>
cd <repository-directory>
```

### 1.2 Install Dependencies
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend\BubbleLabs
pnpm install
```

### 1.3 Verify BubbleLabs is Running
```bash
pnpm run dev
```
This should start BubbleLabs on:
- Frontend: http://localhost:3000
- Backend: http://localhost:3001

## Step 2: Create OpenEvolve Bubble Definitions

### 2.1 Create the OpenEvolve Bubble Core Directory
```bash
mkdir -p packages/bubble-core/src/bubbles/openevolve
```

### 2.2 Create the Content Analyzer Bubble
Create `packages/bubble-core/src/bubbles/openevolve/OpenEvolveContentAnalyzerBubble.ts`:

```typescript
import { ServiceBubble } from '../bubble-core';
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

### 2.3 Create the Problem Decomposer Bubble
Create `packages/bubble-core/src/bubbles/openevolve/OpenEvolveDecomposerBubble.ts`:

```typescript
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

### 2.4 Create the Solver Bubble
Create `packages/bubble-core/src/bubbles/openevolve/OpenEvolveSolverBubble.ts`:

```typescript
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

### 2.5 Create the Verifier Bubble
Create `packages/bubble-core/src/bubbles/openevolve/OpenEvolveVerifierBubble.ts`:

```typescript
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

### 2.6 Create the Full Workflow Bubble
Create `packages/bubble-core/src/bubbles/openevolve/OpenEvolveWorkflowBubble.ts`:

```typescript
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

## Step 3: Register the OpenEvolve Bubbles

### 3.1 Create the registration file
Create `packages/bubble-core/src/bubbles/openevolve/index.ts`:

```typescript
import { BubbleRegistry } from '../bubble-registry';
import { OpenEvolveContentAnalyzerBubble } from './OpenEvolveContentAnalyzerBubble';
import { OpenEvolveDecomposerBubble } from './OpenEvolveDecomposerBubble';
import { OpenEvolveSolverBubble } from './OpenEvolveSolverBubble';
import { OpenEvolveVerifierBubble } from './OpenEvolveVerifierBubble';
import { OpenEvolveWorkflowBubble } from './OpenEvolveWorkflowBubble';

export function registerOpenEvolveBubbles() {
  BubbleRegistry.register('openevolve-content-analyzer', OpenEvolveContentAnalyzerBubble);
  BubbleRegistry.register('openevolve-decomposer', OpenEvolveDecomposerBubble);
  BubbleRegistry.register('openevolve-solver', OpenEvolveSolverBubble);
  BubbleRegistry.register('openevolve-verifier', OpenEvolveVerifierBubble);
  BubbleRegistry.register('openevolve-full-workflow', OpenEvolveWorkflowBubble);
}
```

### 3.2 Update the main bubble registration
Modify `packages/bubble-core/src/bubble-registry.ts` to include OpenEvolve bubbles:

```typescript
// Add this import at the top of the file
import { registerOpenEvolveBubbles } from './bubbles/openevolve';

// Add this call in the registerBuiltInBubbles function
export function registerBuiltInBubbles() {
  // ... existing bubble registrations ...
  
  // Register OpenEvolve bubbles
  registerOpenEvolveBubbles();
}
```

## Step 4: Create OpenEvolve-Specific UI Components

### 4.1 Create OpenEvolve Bubble Node Component
Create `apps/bubble-studio/src/components/OpenEvolveBubbleNode.tsx`:

```tsx
import { memo } from 'react';
import type { BubbleNodeData } from './BubbleNode';
import { useExecutionStore } from '../stores/executionStore';
import { BUBBLE_COLORS } from './BubbleColors';
import { Handle, Position } from '@xyflow/react';
import { CogIcon } from '@heroicons/react/24/outline';

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

### 4.2 Update the BubbleNode to handle OpenEvolve types
Update `apps/bubble-studio/src/components/BubbleNode.tsx` to include OpenEvolve node type:

```tsx
// Add OpenEvolve node type to nodeTypes
const nodeTypes = {
  bubbleNode: BubbleNode,
  inputSchemaNode: InputSchemaNode,
  cronScheduleNode: CronScheduleNode,
  openevolveBubbleNode: OpenEvolveBubbleNode, // Add this line
};
```

## Step 5: Update the API Service for OpenEvolve Integration

### 5.1 Create OpenEvolve API Service
Create `apps/bubble-studio/src/services/openevolveApi.ts`:

```typescript
import { api } from '../lib/api';

export const openevolveApi = {
  /**
   * Execute an OpenEvolve workflow
   */
  executeOpenEvolveWorkflow: async (params: {
    problem: string;
    apiKey: string;
    workflowConfig?: Record<string, unknown>;
  }) => {
    return api.post('/openevolve/execute', params);
  },

  /**
   * Analyze content using OpenEvolve
   */
  analyzeContent: async (params: {
    content: string;
    apiKey: string;
    options?: Record<string, unknown>;
  }) => {
    return api.post('/openevolve/analyze', params);
  },

  /**
   * Decompose a problem using OpenEvolve
   */
  decomposeProblem: async (params: {
    problem: string;
    apiKey: string;
    options?: Record<string, unknown>;
  }) => {
    return api.post('/openevolve/decompose', params);
  },

  /**
   * Solve a sub-problem using OpenEvolve
   */
  solveProblem: async (params: {
    subProblem: Record<string, unknown>;
    apiKey: string;
    options?: Record<string, unknown>;
  }) => {
    return api.post('/openevolve/solve', params);
  },
};
```

## Step 6: Build and Test the Integration

### 6.1 Build the BubbleLabs packages
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend\BubbleLabs
pnpm run build:core
```

### 6.2 Verify the Bubble Registration
Add a test to make sure your bubbles are registered correctly:

Create `packages/bubble-core/__tests__/openevolve-bubbles.test.ts`:

```typescript
import { BubbleRegistry } from '../src/bubble-registry';

describe('OpenEvolve Bubbles Registration', () => {
  it('should register all OpenEvolve bubbles', () => {
    const registeredBubbles = BubbleRegistry.list();
    expect(registeredBubbles).toContain('openevolve-content-analyzer');
    expect(registeredBubbles).toContain('openevolve-decomposer');
    expect(registeredBubbles).toContain('openevolve-solver');
    expect(registeredBubbles).toContain('openevolve-verifier');
    expect(registeredBubbles).toContain('openevolve-full-workflow');
  });

  it('should be able to create OpenEvolve bubbles', () => {
    const ContentAnalyzerBubble = BubbleRegistry.get('openevolve-content-analyzer');
    expect(ContentAnalyzerBubble).toBeDefined();
    
    const workflowBubble = BubbleRegistry.get('openevolve-full-workflow');
    expect(workflowBubble).toBeDefined();
  });
});
```

### 6.3 Run the tests
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend\BubbleLabs
pnpm test:core
```

## Step 7: Configure OpenEvolve API Connection

### 7.1 Create OpenEvolve Configuration
Create `apps/bubblelab-api/src/config/openevolve.ts`:

```typescript
import { OpenEvolveConfigSchema, type OpenEvolveConfig } from '@bubblelab/shared-schemas';

export function getOpenEvolveConfig(): OpenEvolveConfig {
  const config: Partial<OpenEvolveConfig> = {
    baseUrl: process.env.OPENEVOLVE_BASE_URL || 'http://localhost:8000',
    apiKey: process.env.OPENEVOLVE_API_KEY || '',
    defaultModel: process.env.OPENEVOLVE_DEFAULT_MODEL || 'gpt-4o',
    defaultTemperature: parseFloat(process.env.OPENEVOLVE_DEFAULT_TEMPERATURE || '0.7'),
    enableQualityDiversity: process.env.OPENEVOLVE_ENABLE_QUALITY_DIVERSITY === 'true',
    enableMultiObjective: process.env.OPENEVOLVE_ENABLE_MULTI_OBJECTIVE === 'true',
    enableAdversarial: process.env.OPENEVOLVE_ENABLE_ADVERSARIAL === 'true',
    maxIterations: parseInt(process.env.OPENEVOLVE_MAX_ITERATIONS || '100'),
    populationSize: parseInt(process.env.OPENEVOLVE_POPULATION_SIZE || '50'),
    numIslands: parseInt(process.env.OPENEVOLVE_NUM_ISLANDS || '5'),
    migrationRate: parseFloat(process.env.OPENEVOLVE_MIGRATION_RATE || '0.1'),
    archiveSize: parseInt(process.env.OPENEVOLVE_ARCHIVE_SIZE || '100'),
    eliteRatio: parseFloat(process.env.OPENEVOLVE_ELITE_RATIO || '0.1'),
    explorationRatio: parseFloat(process.env.OPENEVOLVE_EXPLORATION_RATIO || '0.2'),
    exploitationRatio: parseFloat(process.env.OPENEVOLVE_EXPLOITATION_RATIO || '0.7'),
    checkpointInterval: parseInt(process.env.OPENEVOLVE_CHECKPOINT_INTERVAL || '10'),
    featureDimensions: (process.env.OPENEVOLVE_FEATURE_DIMENSIONS || 'complexity,diversity').split(','),
    featureBins: parseInt(process.env.OPENEVOLVE_FEATURE_BINS || '10'),
    diversityMetric: process.env.OPENEVOLVE_DIVERSITY_METRIC as any || 'edit_distance',
  };

  return OpenEvolveConfigSchema.parse(config);
}
```

### 7.2 Create OpenEvolve Service
Create `apps/bubblelab-api/src/services/openevolve-service.ts`:

```typescript
import { getOpenEvolveConfig } from '../config/openevolve';

interface OpenEvolveRequest {
  method: string;
  endpoint: string;
  body?: any;
  headers?: Record<string, string>;
}

export class OpenEvolveService {
  private config = getOpenEvolveConfig();

  async request<T = any>({ method, endpoint, body, headers }: OpenEvolveRequest): Promise<T> {
    const url = `${this.config.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`,
        ...headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`OpenEvolve API request failed: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  async executeWorkflow(params: {
    problem: string;
    workflowConfig?: Record<string, any>;
    contentAnalyzerConfig?: Record<string, any>;
    decomposerConfig?: Record<string, any>;
    solverConfig?: Record<string, any>;
    verifierConfig?: Record<string, any>;
  }) {
    return this.request({
      method: 'POST',
      endpoint: '/api/execute-workflow',
      body: params,
    });
  }

  async analyzeContent(content: string, options?: Record<string, any>) {
    return this.request({
      method: 'POST',
      endpoint: '/api/content-analyze',
      body: {
        content,
        ...options,
      },
    });
  }

  async decomposeProblem(problem: string, options?: Record<string, any>) {
    return this.request({
      method: 'POST',
      endpoint: '/api/decompose',
      body: {
        problemStatement: problem,
        ...options,
      },
    });
  }

  async solveProblem(subProblem: Record<string, any>, options?: Record<string, any>) {
    return this.request({
      method: 'POST',
      endpoint: '/api/solve',
      body: {
        subProblem,
        ...options,
      },
    });
  }

  async verifySolution(solution: string, requirements: Record<string, any>) {
    return this.request({
      method: 'POST',
      endpoint: '/api/verify',
      body: {
        solution,
        requirements,
      },
    });
  }
}
```

## Step 8: Add OpenEvolve API Routes

### 8.1 Create the API routes
Create `apps/bubblelab-api/src/routes/openevolve.ts`:

```typescript
import { Hono } from 'hono';
import { OpenEvolveService } from '../services/openevolve-service';

const app = new Hono();

app.post('/openevolve/execute', async (c) => {
  try {
    const { problem, workflowConfig = {}, contentAnalyzerConfig, decomposerConfig, solverConfig, verifierConfig } = await c.req.json();
    
    const service = new OpenEvolveService();
    
    const result = await service.executeWorkflow({
      problem,
      workflowConfig,
      contentAnalyzerConfig,
      decomposerConfig,
      solverConfig,
      verifierConfig,
    });
    
    return c.json({ result, status: 'success' });
  } catch (error: any) {
    console.error('OpenEvolve execution error:', error);
    return c.json({ error: error.message, status: 'error' }, 500);
  }
});

app.post('/openevolve/analyze', async (c) => {
  try {
    const { content, options } = await c.req.json();
    
    const service = new OpenEvolveService();
    
    const result = await service.analyzeContent(content, options);
    
    return c.json({ result, status: 'success' });
  } catch (error: any) {
    console.error('OpenEvolve analysis error:', error);
    return c.json({ error: error.message, status: 'error' }, 500);
  }
});

app.post('/openevolve/decompose', async (c) => {
  try {
    const { problem, options } = await c.req.json();
    
    const service = new OpenEvolveService();
    
    const result = await service.decomposeProblem(problem, options);
    
    return c.json({ result, status: 'success' });
  } catch (error: any) {
    console.error('OpenEvolve decomposition error:', error);
    return c.json({ error: error.message, status: 'error' }, 500);
  }
});

app.post('/openevolve/solve', async (c) => {
  try {
    const { subProblem, options } = await c.req.json();
    
    const service = new OpenEvolveService();
    
    const result = await service.solveProblem(subProblem, options);
    
    return c.json({ result, status: 'success' });
  } catch (error: any) {
    console.error('OpenEvolve solving error:', error);
    return c.json({ error: error.message, status: 'error' }, 500);
  }
});

export default app;
```

### 8.2 Register the routes in the main API
Update `apps/bubblelab-api/src/index.ts` to include OpenEvolve routes:

```typescript
// Add this import
import openevolveRoutes from './routes/openevolve';

// Add this route registration
app.route('/', openevolveRoutes);
```

## Step 9: Start the Development Server and Test

### 9.1 Set Environment Variables
Create or update `.env` files:

In `apps/bubblelab-api/.env`:
```
OPENEVOLVE_BASE_URL=http://localhost:8000
OPENEVOLVE_API_KEY=your-openevolve-api-key
```

### 9.2 Run the Development Server
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend\BubbleLabs
pnpm run dev
```

### 9.3 Test the Integration
1. Navigate to http://localhost:3000
2. Create a new flow
3. Look for OpenEvolve bubbles in the node palette
4. Create a workflow with OpenEvolve nodes
5. Add parameters like API keys and problem statements
6. Execute the workflow

## Step 10: Create Example Workflows

### 10.1 Create an Example OpenEvolve Workflow
Create a sample workflow file that demonstrates the integration:

Create `apps/bubble-studio/src/examples/openevolve-example.ts`:

```typescript
// Example OpenEvolve workflow definition
export const openevolveExampleFlow = `
import { BubbleFlow } from '@bubblelab/bubble-core';
import { OpenEvolveWorkflowBubble } from '@bubblelab/bubble-core/bubbles/openevolve/OpenEvolveWorkflowBubble';

export class ContentOptimizationFlow extends BubbleFlow<'webhook/http'> {
  async handle(payload: { content: string; apiKey: string }) {
    // Step 1: Analyze the content
    const analysisResult = await new OpenEvolveWorkflowBubble({
      problem: payload.content,
      apiKey: payload.apiKey,
      workflowConfig: {
        maxIterations: 50,
        populationSize: 20,
        enableQualityDiversity: true,
      }
    }).action();

    return {
      optimizedContent: analysisResult.data?.finalSolution,
      qualityScore: analysisResult.data?.workflowQuality,
      executionMetrics: analysisResult.data?.executionMetrics,
    };
  }
}
`;
```

## Troubleshooting

### Common Issues and Solutions:

1. **Bubble Registration Issues:**
   - Ensure `registerOpenEvolveBubbles()` is called in the main registration function
   - Verify that the bubble names don't conflict with existing bubbles
   - Check that schema validation passes

2. **API Connection Issues:**
   - Verify that OpenEvolve service is running on the configured endpoint
   - Check API key validity
   - Confirm proper CORS configuration

3. **UI Display Issues:**
   - Ensure node types are properly registered in ReactFlow
   - Verify component imports are correct
   - Check for TypeScript compilation errors

4. **Build Issues:**
   - Run `pnpm install` to ensure all dependencies are present
   - Run `pnpm run build:core` to build the bubble packages
   - Check for any compilation errors in the build output

## Verification Steps

Once the integration is complete, verify the following:

1. ✅ OpenEvolve bubbles appear in the BubbleLabs UI
2. ✅ Bubbles can be added to workflows
3. ✅ Parameter configuration works correctly
4. ✅ Workflows can be executed successfully
5. ✅ Results are displayed properly
6. ✅ Error handling works as expected
7. ✅ Real-time status updates function correctly

This implementation guide provides a complete step-by-step approach to integrating OpenEvolve with BubbleLabs, enabling complete control and visualization of OpenEvolve workflows through the BubbleLabs UI.