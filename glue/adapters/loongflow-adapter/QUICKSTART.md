# LoongFlow Adapter - Quick Start

## 5-Minute Setup

### 1. Install Dependencies
```bash
cd glue/adapters/loongflow-adapter
npm install
```

### 2. Set Required Environment Variables
```bash
export LOONGFLOW_API_URL="http://localhost:8000"
export LOONGFLOW_TIMEOUT_MS=30000
```

### 3. Build the Adapter
```bash
npm run build
```

### 4. Verify LoongFlow Structure (Probes)
```bash
cd probes
./check_api.sh
```

Expected output:
```
✅ LoongFlow source found at: ../../../../core-projects/LoongFlow
✅ PES Agent module found
✅ EvolveDatabase module found
✅ LoongFlow API probe completed successfully
```

### 5. Run Tests
```bash
cd ..
npm test
```

## Basic Usage

```typescript
import { createLoongFlowAdapter } from './src/index';

// Initialize
const adapter = createLoongFlowAdapter({
  api_url: 'http://localhost:8000',
  timeout_ms: 30000,
});

// Submit problem
const { agent_id } = await adapter.submitProblem({
  task: 'Optimize the circle packing algorithm',
  max_iterations: 100,
  target_score: 0.95,
});

// Monitor progress
const state = await adapter.getAgentState(agent_id);
console.log(`Best score: ${state.best_score}`);

// Get result
const result = await adapter.getExecutionResult(agent_id);
console.log(`Final solution: ${result.final_solution}`);
```

## Docker Deployment

```bash
# Build image
docker build -t loongflow-adapter:1.0.0 .

# Run container
docker run -d \
  -e LOONGFLOW_API_URL=http://loongflow-sidecar:8000 \
  -e LOONGFLOW_TIMEOUT_MS=30000 \
  -p 8040:8040 \
  loongflow-adapter:1.0.0
```

## Troubleshooting

### "LOONGFLOW_API_URL is required"
Set the environment variable before starting.

### "Circuit breaker is OPEN"
LoongFlow sidecar is down. Check with:
```bash
curl $LOONGFLOW_API_URL/health
```

### Probe fails with "Python not found"
Probes verify source structure, not execution. This is OK on Windows.
The actual Python execution happens in the sidecar container.

## Next Steps

- Read [README.md](./README.md) for full documentation
- Read [ADR.md](./ADR.md) for architecture decisions
- See [IMPLEMENTATION_COMPLETE.md](./IMPLEMENTATION_COMPLETE.md) for detailed report

## Support

- Federation Constitution: `../../../CLAUDE.md`
- LoongFlow Source: `../../../../core-projects/LoongFlow/`
- Issues: Create ticket in project tracker
