/**
 * API Integration Guide
 * OpenEvolve Frontend - BubbleLab UI
 */

# API Integration Guide

This guide explains how to integrate the BubbleLab React UI with the OpenEvolve Python backend.

---

## Architecture Overview

```
React Frontend → API Bridge (FastAPI) → Python Backend → Workflow Engine
     ↓                    ↓                      ↓
Zustand Stores       CORS/SSE              Team/Gauntlet
React Query          WebSocket             Managers
TanStack Router      Auth Middleware       LLM Integration
```

---

## Prerequisites

### Backend Requirements:
- Python 3.9+
- FastAPI
- Uvicorn
- OpenEvolve backend running on port 8000

### Frontend Requirements:
- Node.js 18+
- React 18+
- BubbleLab monorepo

---

## API Bridge Setup

The `api_bridge.py` file serves as the gateway between React and Python.

### Starting the API Bridge:

```bash
# Navigate to frontend directory
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Install Python dependencies
pip install -r api_bridge_requirements.txt

# Start the API bridge
python api_bridge.py
```

The API bridge will run on `http://localhost:8001` by default.

---

## API Endpoints

### Workflows

#### List Workflows
```typescript
GET /api/workflows
```

```typescript
// Usage
const { data: workflows } = await apiClient.getWorkflows();
```

#### Create Workflow
```typescript
POST /api/workflows
```

```typescript
// Usage
const workflow = await apiClient.createWorkflow({
  name: 'My Workflow',
  problem_statement: 'Solve this problem',
  content_type: 'math',
  teams: ['team-1'],
  gauntlets: ['gauntlet-1'],
});
```

#### Get Workflow Details
```typescript
GET /api/workflows/:id
```

#### Start Execution
```typescript
POST /api/workflows/:id/start
```

#### Pause Execution
```typescript
POST /api/workflows/:id/pause
```

#### Resume Execution
```typescript
POST /api/workflows/:id/resume
```

#### Stop Execution
```typescript
POST /api/workflows/:id/stop
```

### Teams

#### List Teams
```typescript
GET /api/teams
```

#### Create Team
```typescript
POST /api/teams
```

```typescript
await apiClient.createTeam({
  name: 'Math Solvers',
  members: [
    {
      name: 'Solver',
      model: 'gpt-4',
      temperature: 0.7,
      max_tokens: 2000,
    },
  ],
});
```

### Gauntlets

#### List Gauntlets
```typescript
GET /api/gauntlets
```

#### Create Gauntlet
```typescript
POST /api/gauntlets
```

```typescript
await apiClient.createGauntlet({
  name: 'Quality Gate',
  rounds: [
    {
      name: 'Round 1',
      quorum: 0.7,
      confidence_threshold: 0.8,
    },
  ],
});
```

---

## SSE Streaming

Real-time workflow execution updates are delivered via Server-Sent Events.

### Connecting to SSE Stream:

```typescript
import { useExecutionStream } from '@/hooks/use-execution-stream';

function WorkflowExecution({ workflowId }: { workflowId: string }) {
  const { events, connected, error } = useExecutionStream(workflowId);

  useEffect(() => {
    if (events.length > 0) {
      const latestEvent = events[events.length - 1];
      console.log('New event:', latestEvent);
    }
  }, [events]);

  return (
    <div>
      <p>Connection: {connected ? 'Connected' : 'Disconnected'}</p>
      {error && <p>Error: {error.message}</p>}
    </div>
  );
}
```

### Event Types:

```typescript
interface ExecutionEvent {
  type: 'workflow_started' | 'node_started' | 'node_completed' | 'workflow_completed' | 'error';
  workflow_id: string;
  timestamp: string;
  data: unknown;
}
```

---

## Authentication

The application uses Clerk for authentication. JWT tokens are automatically included in API requests.

### Setting Up Clerk:

1. Create a Clerk application at https://dashboard.clerk.com
2. Add your Clerk Publishable Key to `.env`:
   ```
   VITE_CLERK_PUBLISHABLE_KEY=pk_test_...
   ```
3. Wrap your app with `<ClerkProvider>` in `main.tsx`:
   ```typescript
   import { ClerkProvider } from '@clerk/clerk-react';

   root.render(
     <ClerkProvider publishableKey={import.meta.env.VITE_CLERK_PUBLISHABLE_KEY}>
       <App />
     </ClerkProvider>
   );
   ```

### Including Auth in API Requests:

The `apiClient` automatically includes the JWT token:

```typescript
// In api-client.ts
const token = await getToken();
headers.append('Authorization', `Bearer ${token}`);
```

---

## Error Handling

API errors are automatically caught and formatted:

```typescript
try {
  const workflow = await apiClient.createWorkflow(data);
} catch (error) {
  if (error instanceof ApiError) {
    console.error('API Error:', error.message);
    console.error('Status:', error.status);
    console.error('Details:', error.details);
  }
}
```

---

## React Query Integration

All API calls are wrapped in React Query hooks for automatic caching and revalidation.

### Using React Query Hooks:

```typescript
import { useWorkflows, useCreateWorkflow } from '@/hooks/use-workflows-api';

function WorkflowList() {
  const { data: workflows, isLoading, error } = useWorkflows();
  const createWorkflow = useCreateWorkflow();

  if (isLoading) return <LoadingSpinner />;
  if (error) return <Error message={error.message} />;

  return (
    <div>
      {workflows?.map((workflow) => (
        <WorkflowCard key={workflow.id} workflow={workflow} />
      ))}

      <Button onClick={() => createWorkflow.mutate({ /* data */ })}>
        Create Workflow
      </Button>
    </div>
  );
}
```

---

## Environment Configuration

### Development (.env.development):
```env
VITE_API_BASE_URL=http://localhost:8001
VITE_CLERK_PUBLISHABLE_KEY=pk_test_...
```

### Production (.env.production):
```env
VITE_API_BASE_URL=https://api.openevolve.com
VITE_CLERK_PUBLISHABLE_KEY=pk_live_...
```

---

## Troubleshooting

### CORS Errors

If you encounter CORS errors:

1. Ensure `api_bridge.py` is running
2. Check CORS middleware configuration:
   ```python
   api_bridge.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:5173"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

### SSE Connection Issues

If SSE streaming doesn't work:

1. Check browser console for errors
2. Verify the endpoint is accessible:
   ```bash
   curl http://localhost:8001/stream/workflow/test-id
   ```
3. Ensure no proxy is blocking SSE connections

### Authentication Issues

If auth isn't working:

1. Verify Clerk is configured correctly
2. Check that JWT token is being sent:
   ```typescript
   console.log('Token:', await getToken());
   ```
3. Ensure Python backend can validate Clerk JWTs

---

## Performance Optimization

### API Response Caching:

React Query automatically caches responses:

```typescript
const { data } = useQuery({
  queryKey: ['workflows'],
  queryFn: () => apiClient.getWorkflows(),
  staleTime: 5 * 60 * 1000, // 5 minutes
  cacheTime: 10 * 60 * 1000, // 10 minutes
});
```

### Request Debouncing:

Debounce user input before making API calls:

```typescript
import { useDebounce } from '@/hooks/useDebounce';

function Search() {
  const [searchTerm, setSearchTerm] = useState('');
  const debouncedSearch = useDebounce(searchTerm, 500);

  useEffect(() => {
    if (debouncedSearch) {
      // Make API call with debounced value
      searchWorkflows(debouncedSearch);
    }
  }, [debouncedSearch]);

  return <input value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} />;
}
```

---

## Testing API Integration

### Mocking API Responses:

For testing without a real backend:

```typescript
// __mocks__/api-client.ts
export const apiClient = {
  getWorkflows: jest.fn().mockResolvedValue(mockWorkflows),
  createWorkflow: jest.fn().mockResolvedValue(mockWorkflow),
  // ...
};
```

### Integration Tests:

```typescript
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

test('loads and displays workflows', async () => {
  const queryClient = new QueryClient();

  render(
    <QueryClientProvider client={queryClient}>
      <WorkflowList />
    </QueryClientProvider>
  );

  await waitFor(() => {
    expect(screen.getByText('Workflow 1')).toBeInTheDocument();
  });
});
```

---

## Next Steps

1. ✅ Start API bridge: `python api_bridge.py`
2. ✅ Configure environment variables
3. ✅ Set up Clerk authentication
4. ✅ Test API endpoints
5. ✅ Integrate SSE streaming
6. ✅ Deploy to production

---

**For more information, see:**
- [Migration Plan](./BUBBLELAB_MIGRATION_PLAN.md)
- [Component Documentation](./COMPONENT_DOCS.md)
- [Deployment Guide](./DEPLOYMENT.md)
