# OpenEvolve API - Quick Reference Guide

## Import Statement

```typescript
// Import everything from the central location
import {
  // API Service
  openEvolveAPI,

  // Types
  EvolutionRun,
  EvolutionConfig,
  AdversarialRun,
  KnowledgeEntry,
  WorkflowDefinition,

  // Hooks
  useEvolutionRuns,
  useAdversarialRuns,
  useKnowledgeEntries,
  useWorkflows,
  useAnalyticsOverview,
} from '@/services/api';
```

## Common Hooks Quick Reference

### Evolution Hooks
```typescript
useEvolutionRuns({ status, limit })
useEvolutionRun(runId)
useCreateEvolutionRun()
useEvolutionConfig()
```

### Adversarial Hooks
```typescript
useAdversarialRuns({ status, limit })
useAdversarialRun(runId)
useCreateAdversarialRun()
useAdversarialConfig()
```

### Knowledge Hooks
```typescript
useKnowledgeEntries({ search, category, tags, status })
useKnowledgeCategories()
useKnowledgeStats()
useCreateKnowledgeEntry()
```

### Workflow Hooks
```typescript
useWorkflows({ status, limit })
useWorkflow(workflowId)
useWorkflowInstances(workflowId)
useWorkflowTemplates()
```

### Analytics Hooks
```typescript
useWorkflowPerformance({ startDate, endDate })
useTeamPerformance({ teamIds })
useGauntletPerformance()
useSolutionQuality()
useAnalyticsOverview({ startDate, endDate })
```

### System Hooks
```typescript
useHealthStatus(pollInterval?)  // Auto-refresh every N ms
useSystemStatus(pollInterval?)  // Auto-refresh every N ms
```

## Hook Return Value Pattern

All data fetching hooks return:
```typescript
{
  data: T | null,           // The fetched data (null if loading)
  isLoading: boolean,       // True while fetching
  error: Error | null,      // Error if request failed
  refetch: () => void       // Function to refetch data
}
```

All mutation hooks return:
```typescript
{
  data: T | null,           // Result of mutation (null if not executed)
  isLoading: boolean,       // True while mutating
  error: Error | null,      // Error if mutation failed
  mutate: (params) => Promise<void>,  // Execute mutation
  reset: () => void         // Clear data and error
}
```

## Quick Examples

### Example 1: Display List of Runs
```typescript
import { useEvolutionRuns } from '@/services/api';

function EvolutionList() {
  const { data: runs, isLoading, error } = useEvolutionRuns();

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <ul>
      {runs?.map(run => (
        <li key={run.id}>{run.name} - {run.progress}%</li>
      ))}
    </ul>
  );
}
```

### Example 2: Create New Run
```typescript
import { useCreateEvolutionRun } from '@/services/api';

function CreateEvolutionButton() {
  const { mutate, isLoading } = useCreateEvolutionRun();

  const handleClick = () => {
    mutate({
      name: 'My Run',
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
    <button onClick={handleClick} disabled={isLoading}>
      {isLoading ? 'Creating...' : 'Create Evolution Run'}
    </button>
  );
}
```

### Example 3: Control Run
```typescript
import { openEvolveAPI } from '@/services/api';
import { useEvolutionRun } from '@/services/api';

function EvolutionControls({ runId }) {
  const { data: run, refetch } = useEvolutionRun(runId);

  const handlePause = async () => {
    await openEvolveAPI.pauseEvolutionRun(runId);
    refetch();
  };

  const handleResume = async () => {
    await openEvolveAPI.resumeEvolutionRun(runId);
    refetch();
  };

  const handleStop = async () => {
    await openEvolveAPI.stopEvolutionRun(runId);
    refetch();
  };

  return (
    <div>
      {run?.status === 'running' && (
        <button onClick={handlePause}>Pause</button>
      )}
      {run?.status === 'paused' && (
        <>
          <button onClick={handleResume}>Resume</button>
          <button onClick={handleStop}>Stop</button>
        </>
      )}
    </div>
  );
}
```

### Example 4: Search Knowledge Base
```typescript
import { useKnowledgeEntries } from '@/services/api';

function KnowledgeSearch({ searchQuery }) {
  const { data: entries, isLoading } = useKnowledgeEntries({
    search: searchQuery,
    status: 'published'
  });

  return (
    <div>
      {isLoading ? (
        <div>Searching...</div>
      ) : (
        <ul>
          {entries?.map(entry => (
            <li key={entry.id}>
              <h3>{entry.title}</h3>
              <p>{entry.content.substring(0, 100)}...</p>
              <div>{entry.tags.join(', ')}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

### Example 5: Real-time System Status
```typescript
import { useHealthStatus } from '@/services/api';

function SystemHealth() {
  // Poll every 5 seconds
  const { data: health } = useHealthStatus(5000);

  return (
    <div>
      <h2>System Status: {health?.status}</h2>
      <ul>
        <li>Evolution: {health?.services.evolution.status}</li>
        <li>Adversarial: {health?.services.adversarial.status}</li>
        <li>Knowledge: {health?.services.knowledge.status}</li>
      </ul>
    </div>
  );
}
```

### Example 6: Analytics Dashboard
```typescript
import { useAnalyticsOverview } from '@/services/api';

function Dashboard() {
  const { data: overview, isLoading } = useAnalyticsOverview({
    startDate: '2024-01-01',
    endDate: '2024-12-31'
  });

  if (isLoading) return <div>Loading dashboard...</div>;

  return (
    <div>
      <h2>Analytics Overview</h2>
      <div>Total Workflows: {overview?.workflows.length}</div>
      <div>
        Success Rate:{' '}
        {overview?.workflows.reduce((acc, w) => acc + w.successRate, 0) /
          overview.workflows.length}
      </div>
    </div>
  );
}
```

## Type Reference

### Status Types
```typescript
type RunStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed';
type EntryStatus = 'draft' | 'published' | 'archived';
type WorkflowStatus = 'draft' | 'published' | 'archived';
```

### Selection Methods
```typescript
type SelectionMethod = 'tournament' | 'roulette' | 'rank' | 'uniform';
```

### Attack Strategies
```typescript
type AttackStrategy = 'fgsm' | 'pgd' | 'cw' | 'bim' | 'deepfool';
```

### Defense Strategies
```typescript
type DefenseStrategy = 'robust' | 'certified' | 'detection' | 'randomization' | 'gradient_masking';
```

## Common Patterns

### Pattern 1: Conditional Fetching
```typescript
// Only fetch when runId is provided
const { data: run } = useEvolutionRun(runId || '');
```

### Pattern 2: Create and Refresh
```typescript
const { mutate } = useCreateEvolutionRun();
const { refetch } = useEvolutionRuns();

const handleCreate = () => {
  mutate(request).then(() => refetch());
};
```

### Pattern 3: Error Handling
```typescript
const { data, error, isLoading } = useEvolutionRuns();

useEffect(() => {
  if (error) {
    console.error('Failed to load runs:', error);
    toast.error(error.message);
  }
}, [error]);
```

### Pattern 4: Optimistic Updates
```typescript
const [localRuns, setLocalRuns] = useState([]);
const { data: serverRuns } = useEvolutionRuns();

// Use local state for immediate updates
useEffect(() => {
  if (serverRuns) {
    setLocalRuns(serverRuns);
  }
}, [serverRuns]);
```

## Configuration

### Environment Variables
```bash
# .env file
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

### Client Configuration
```typescript
// Already configured in client.ts
{
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
  timeout: 30000,  // 30 seconds
  retryAttempts: 3,
  retryDelay: 1000  // 1 second
}
```

## Troubleshooting

### Problem: "Authentication required"
**Solution:** Check that auth token is set in the store

### Problem: "Request timeout"
**Solution:** Increase timeout in `client.ts` or check backend responsiveness

### Problem: "CORS error"
**Solution:** Ensure backend allows frontend origin

### Problem: "Type errors"
**Solution:** Import types from `@/services/api`, not from component files

## Useful Utilities

### Date Conversion
The API returns ISO strings. Convert to Date objects:
```typescript
const startDate = new Date(run.startTime);
```

### Progress Calculation
```typescript
const percentage = Math.round(run.progress * 100) + '%';
```

### Status Colors
```typescript
const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed': return 'green';
    case 'running': return 'blue';
    case 'failed': return 'red';
    case 'paused': return 'yellow';
    default: return 'gray';
  }
};
```

## See Also

- **Complete Documentation:** `OpenEvolve-Plugin/src/services/api/README.md`
- **Implementation Summary:** `API_INTEGRATION_SUMMARY.md`
- **API Service Source:** `OpenEvolveAPI.ts`
- **React Hooks Source:** `OpenEvolveAPIHooks.ts`
