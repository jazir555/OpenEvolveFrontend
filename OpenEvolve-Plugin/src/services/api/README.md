# OpenEvolve API Service Layer Documentation

## Overview

The OpenEvolve API Service Layer provides a comprehensive, type-safe interface for connecting to the OpenEvolve backend. It replaces all mock data interfaces across page components with actual API calls, supporting RESTful patterns, error handling, and React integration.

## Architecture

```
OpenEvolveAPI.ts (Main API Service)
    ├── Type Definitions (all request/response interfaces)
    ├── OpenEvolveAPI Class (singleton pattern)
    │   ├── Evolution Methods
    │   ├── Adversarial Methods
    │   ├── Knowledge Base Methods
    │   ├── Workflow Methods
    │   ├── Analytics Methods
    │   ├── Decomposition Methods
    │   └── System/Health Methods
    └── Exports: openEvolveAPI (singleton instance)

OpenEvolveAPIHooks.ts (React Integration)
    ├── Custom React Hooks
    │   ├── Data Fetching Hooks (useQuery pattern)
    │   ├── Mutation Hooks (useMutation pattern)
    │   └── Real-time Polling Hooks
    └── Automatic Loading/Error State Management

client.ts (Underlying HTTP Client)
    ├── fetch-based implementation
    ├── Timeout handling
    ├── Retry logic with exponential backoff
    ├── Authentication token injection
    └── Comprehensive error handling
```

## Installation & Setup

### 1. Environment Configuration

Create or update `.env` file:

```bash
# API Base URL (adjust based on your deployment)
VITE_API_BASE_URL=http://localhost:8000/api/v1

# Alternative: Production URL
# VITE_API_BASE_URL=https://api.openevolve.com/api/v1
```

### 2. Import the API Service

```typescript
// Import the singleton instance
import { openEvolveAPI } from '@/services/api';

// Or import specific types
import type {
  EvolutionRun,
  EvolutionConfig,
  AdversarialRun,
  KnowledgeEntry
} from '@/services/api';
```

## Usage Examples

### Evolution API

#### Fetch All Evolution Runs

```typescript
import { useEvolutionRuns } from '@/services/api';

function EvolutionList() {
  const { data: runs, isLoading, error, refetch } = useEvolutionRuns({
    status: 'running',
    limit: 10
  });

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <ul>
      {runs?.map(run => (
        <li key={run.id}>
          {run.name} - Progress: {run.progress}%
        </li>
      ))}
    </ul>
  );
}
```

#### Create Evolution Run

```typescript
import { useCreateEvolutionRun } from '@/services/api';

function CreateEvolutionForm() {
  const { mutate, isLoading, error } = useCreateEvolutionRun();

  const handleSubmit = () => {
    mutate({
      name: 'My Evolution Run',
      config: {
        populationSize: 100,
        generations: 50,
        mutationRate: 0.1,
        crossoverRate: 0.8,
        selectionMethod: 'tournament',
        elitismCount: 2,
        tournamentSize: 3,
        temperature: 0.7,
        modelId: 'gpt-4',
        mdapMakerEnabled: false,
        mdapMakerAutoSelect: true
      }
    });
  };

  return (
    <button onClick={handleSubmit} disabled={isLoading}>
      {isLoading ? 'Creating...' : 'Create Run'}
    </button>
  );
}
```

#### Control Evolution Run

```typescript
import { openEvolveAPI } from '@/services/api';

// Pause a run
async function pauseRun(runId: string) {
  try {
    const result = await openEvolveAPI.pauseEvolutionRun(runId);
    console.log('Run paused:', result);
  } catch (error) {
    console.error('Failed to pause run:', error);
  }
}

// Resume a run
async function resumeRun(runId: string) {
  try {
    const result = await openEvolveAPI.resumeEvolutionRun(runId);
    console.log('Run resumed:', result);
  } catch (error) {
    console.error('Failed to resume run:', error);
  }
}

// Stop a run
async function stopRun(runId: string) {
  try {
    const result = await openEvolveAPI.stopEvolutionRun(runId);
    console.log('Run stopped:', result);
  } catch (error) {
    console.error('Failed to stop run:', error);
  }
}
```

### Adversarial API

```typescript
import { useAdversarialRuns, useAdversarialConfig } from '@/services/api';

function AdversarialDashboard() {
  const { data: runs, isLoading } = useAdversarialRuns();
  const { data: config, updateConfig } = useAdversarialConfig();

  const handleUpdateConfig = () => {
    updateConfig({
      attackStrategy: 'pgd',
      defenseStrategy: 'robust',
      strength: 0.3
    });
  };

  return (
    <div>
      <h2>Adversarial Runs</h2>
      {runs?.map(run => (
        <div key={run.id}>{run.name}</div>
      ))}
      <button onClick={handleUpdateConfig}>Update Config</button>
    </div>
  );
}
```

### Knowledge Base API

```typescript
import {
  useKnowledgeEntries,
  useKnowledgeCategories,
  useCreateKnowledgeEntry
} from '@/services/api';

function KnowledgeBase() {
  const { data: entries, isLoading, refetch } = useKnowledgeEntries({
    status: 'published'
  });
  const { data: categories } = useKnowledgeCategories();
  const { mutate: createEntry } = useCreateKnowledgeEntry();

  const handleCreate = () => {
    createEntry({
      title: 'New Entry',
      content: 'Content here...',
      category: 'algorithms',
      tags: ['optimization', 'evolutionary'],
      status: 'published'
    }).then(() => refetch());
  };

  return (
    <div>
      <button onClick={handleCreate}>Create Entry</button>
      {entries?.map(entry => (
        <div key={entry.id}>{entry.title}</div>
      ))}
    </div>
  );
}
```

### Workflow API

```typescript
import { useWorkflows, useWorkflow } from '@/services/api';

function WorkflowManager() {
  const { data: workflows } = useWorkflows();
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const { data: workflow } = useWorkflow(selectedWorkflow || '');

  const handleRunWorkflow = async () => {
    if (!selectedWorkflow) return;

    try {
      const instance = await openEvolveAPI.runWorkflow(selectedWorkflow);
      console.log('Workflow started:', instance);
    } catch (error) {
      console.error('Failed to run workflow:', error);
    }
  };

  return (
    <div>
      <ul>
        {workflows?.map(wf => (
          <li key={wf.id} onClick={() => setSelectedWorkflow(wf.id)}>
            {wf.name}
          </li>
        ))}
      </ul>
      {workflow && (
        <div>
          <h3>{workflow.name}</h3>
          <p>{workflow.description}</p>
          <button onClick={handleRunWorkflow}>Run Workflow</button>
        </div>
      )}
    </div>
  );
}
```

### Analytics API

```typescript
import {
  useAnalyticsOverview,
  useWorkflowPerformance,
  useTeamPerformance
} from '@/services/api';

function AnalyticsDashboard() {
  const { data: overview, isLoading } = useAnalyticsOverview({
    startDate: '2024-01-01',
    endDate: '2024-12-31'
  });

  if (isLoading) return <div>Loading...</div>;

  return (
    <div>
      <h2>Analytics Overview</h2>
      <div>Total Workflows: {overview?.workflows.length}</div>
      <div>
        {overview?.workflows.map(wf => (
          <div key={wf.workflowId}>
            {wf.name}: {wf.successRate * 100}% success rate
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Decomposition API

```typescript
import {
  useDecompositionProblems,
  useSubProblems
} from '@/services/api';

function DecompositionView() {
  const { data: problems } = useDecompositionProblems();
  const [selectedProblem, setSelectedProblem] = useState<string | null>(null);
  const { data: subProblems } = useSubProblems(selectedProblem || '');

  return (
    <div>
      <h2>Decomposition Problems</h2>
      <ul>
        {problems?.map(prob => (
          <li key={prob.id} onClick={() => setSelectedProblem(prob.id)}>
            {prob.title}
          </li>
        ))}
      </ul>

      {subProblems && (
        <div>
          <h3>Sub-Problems</h3>
          <ul>
            {subProblems.map(sub => (
              <li key={sub.id}>{sub.title}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

## API Reference

### Evolution Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `getEvolutionRuns()` | `GET /evolution/runs` | Get all evolution runs |
| `getEvolutionRun(id)` | `GET /evolution/runs/:id` | Get specific run |
| `createEvolutionRun(req)` | `POST /evolution/runs` | Create new run |
| `updateEvolutionRun(id, req)` | `PATCH /evolution/runs/:id` | Update run |
| `deleteEvolutionRun(id)` | `DELETE /evolution/runs/:id` | Delete run |
| `startEvolutionRun(id)` | `POST /evolution/runs/:id/start` | Start run |
| `pauseEvolutionRun(id)` | `POST /evolution/runs/:id/pause` | Pause run |
| `resumeEvolutionRun(id)` | `POST /evolution/runs/:id/resume` | Resume run |
| `stopEvolutionRun(id)` | `POST /evolution/runs/:id/stop` | Stop run |
| `getEvolutionConfig()` | `GET /evolution/config` | Get config |
| `updateEvolutionConfig(config)` | `PUT /evolution/config` | Update config |

### Adversarial Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `getAdversarialRuns()` | `GET /adversarial/runs` | Get all runs |
| `getAdversarialRun(id)` | `GET /adversarial/runs/:id` | Get specific run |
| `createAdversarialRun(req)` | `POST /adversarial/runs` | Create new run |
| `startAdversarialRun(id)` | `POST /adversarial/runs/:id/start` | Start run |
| `pauseAdversarialRun(id)` | `POST /adversarial/runs/:id/pause` | Pause run |
| `stopAdversarialRun(id)` | `POST /adversarial/runs/:id/stop` | Stop run |
| `getAdversarialConfig()` | `GET /adversarial/config` | Get config |
| `updateAdversarialConfig(config)` | `PUT /adversarial/config` | Update config |

### Knowledge Base Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `getKnowledgeEntries(params)` | `GET /knowledge/entries` | Get entries |
| `getKnowledgeEntry(id)` | `GET /knowledge/entries/:id` | Get specific entry |
| `createKnowledgeEntry(req)` | `POST /knowledge/entries` | Create entry |
| `updateKnowledgeEntry(id, req)` | `PATCH /knowledge/entries/:id` | Update entry |
| `deleteKnowledgeEntry(id)` | `DELETE /knowledge/entries/:id` | Delete entry |
| `getKnowledgeCategories()` | `GET /knowledge/categories` | Get categories |
| `getKnowledgeStats()` | `GET /knowledge/stats` | Get statistics |
| `searchKnowledge(query)` | `GET /knowledge/search` | Search entries |

### Workflow Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `getWorkflows(params)` | `GET /workflows` | Get all workflows |
| `getWorkflow(id)` | `GET /workflows/:id` | Get specific workflow |
| `createWorkflow(req)` | `POST /workflows` | Create workflow |
| `updateWorkflow(id, req)` | `PATCH /workflows/:id` | Update workflow |
| `deleteWorkflow(id)` | `DELETE /workflows/:id` | Delete workflow |
| `publishWorkflow(id)` | `POST /workflows/:id/publish` | Publish workflow |
| `runWorkflow(id, config)` | `POST /workflows/:id/run` | Run workflow |
| `getWorkflowInstances(id)` | `GET /workflows/:id/instances` | Get instances |

### Analytics Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `getWorkflowPerformance(params)` | `GET /analytics/workflows` | Get performance |
| `getTeamPerformance(params)` | `GET /analytics/teams` | Get team metrics |
| `getGauntletPerformance(params)` | `GET /analytics/gauntlets` | Get gauntlet metrics |
| `getSolutionQuality(params)` | `GET /analytics/solutions` | Get quality metrics |
| `getAnalyticsOverview(params)` | `GET /analytics/overview` | Get overview |
| `exportAnalytics(params)` | `POST /analytics/export` | Export data |

## Type Definitions

### EvolutionConfig

```typescript
interface EvolutionConfig {
  populationSize: number;
  generations: number;
  mutationRate: number;
  crossoverRate: number;
  selectionMethod: 'tournament' | 'roulette' | 'rank' | 'uniform';
  elitismCount: number;
  tournamentSize: number;
  temperature: number;
  modelId: string;
  mdapMakerEnabled: boolean;
  mdapMakerAutoSelect: boolean;
}
```

### EvolutionRun

```typescript
interface EvolutionRun {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  generation: number;
  bestFitness: number;
  avgFitness: number;
  startTime?: string;
  endTime?: string;
  config: EvolutionConfig;
}
```

### AdversarialConfig

```typescript
interface AdversarialConfig {
  enabled: boolean;
  attackStrategy: 'fgsm' | 'pgd' | 'cw' | 'bim' | 'deepfool';
  numExamples: number;
  strength: number;
  stepSize: number;
  numSteps: number;
  defenseStrategy: 'robust' | 'certified' | 'detection' | 'randomization' | 'gradient_masking';
  robustnessThreshold: number;
  modelId: string;
  mdapMakerEnabled: boolean;
  mdapMakerAutoSelect: boolean;
}
```

See individual TypeScript files for complete type definitions.

## Error Handling

All API methods throw errors that can be caught and handled:

```typescript
try {
  const result = await openEvolveAPI.createEvolutionRun(request);
  // Handle success
} catch (error) {
  if (error.message.includes('Authentication required')) {
    // Redirect to login
  } else if (error.message.includes('permission')) {
    // Show permission error
  } else {
    // Show generic error
    console.error('API Error:', error);
  }
}
```

### React Hook Error Handling

```typescript
const { data, isLoading, error } = useEvolutionRuns();

useEffect(() => {
  if (error) {
    toast.error(`Failed to load runs: ${error.message}`);
  }
}, [error]);
```

## Best Practices

### 1. Use React Hooks for Component Integration

```typescript
// ✅ Good: Use hooks
const { data, isLoading } = useEvolutionRuns();

// ❌ Bad: Direct API calls in components
const [data, setData] = useState([]);
useEffect(() => {
  openEvolveAPI.getEvolutionRuns().then(setData);
}, []);
```

### 2. Handle Loading States

```typescript
const { data, isLoading, error } = useKnowledgeEntries();

if (isLoading) return <LoadingSpinner />;
if (error) return <ErrorMessage error={error} />;
if (!data || data.length === 0) return <EmptyState />;
```

### 3. Leverage Refetch for Data Updates

```typescript
const { data, refetch } = useEvolutionRuns();

const handleCreate = async () => {
  await createEvolutionRun(request);
  refetch(); // Refresh list after creation
};
```

### 4. Use Mutation Hooks for Actions

```typescript
const { mutate, isLoading } = useCreateEvolutionRun();

const handleSubmit = () => {
  mutate(request)
    .then(() => toast.success('Run created'))
    .catch(() => toast.error('Failed to create run'));
};
```

## Testing

### Mock API for Testing

```typescript
// __tests__/mocks/api.ts
export const mockOpenEvolveAPI = {
  getEvolutionRuns: vi.fn().mockResolvedValue(mockRuns),
  createEvolutionRun: vi.fn().mockResolvedValue(mockRun),
  // ... other methods
};

// Test file
import { mockOpenEvolveAPI } from './mocks/api';
vi.mock('@/services/api', () => ({
  openEvolveAPI: mockOpenEvolveAPI
}));
```

## Migration from Mock Data

### Before (Mock Data)

```typescript
const [runs, setRuns] = useState<EvolutionRun[]>([
  {
    id: 'run-1',
    name: 'Optimization Run #1',
    status: 'completed',
    // ... mock data
  }
]);
```

### After (API Integration)

```typescript
const { data: runs, isLoading } = useEvolutionRuns();

if (isLoading) return <div>Loading...</div>;
```

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure backend allows frontend origin
   - Check API_BASE_URL configuration

2. **Authentication Errors**
   - Verify auth token is set
   - Check token expiration
   - Ensure proper Authorization header

3. **Timeout Errors**
   - Increase timeout in client.ts
   - Check network connectivity
   - Verify backend is responsive

4. **Type Errors**
   - Ensure all types are imported from '@/services/api'
   - Check for version compatibility
   - Run `tsc --noEmit` to verify types

## Support

For issues or questions:
- Check TypeScript type definitions in `OpenEvolveAPI.ts`
- Review React hook examples in `OpenEvolveAPIHooks.ts`
- Consult existing component implementations
- See error handling in `client.ts`
