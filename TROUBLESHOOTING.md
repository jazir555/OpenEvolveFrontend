# Hybrid PES System Troubleshooting Guide

Common issues, symptoms, and solutions for the OpenEvolve LoongFlow PES hybrid system.

## Table of Contents

1. [Build and Compilation Issues](#build-and-compilation-issues)
2. [Test Failures](#test-failures)
3. [Deployment Issues](#deployment-issues)
4. [Runtime Errors](#runtime-errors)
5. [Performance Issues](#performance-issues)
6. [Integration Issues](#integration-issues)
7. [Network and Connectivity Issues](#network-and-connectivity-issues)
8. [Data and Schema Issues](#data-and-schema-issues)
9. [Container and Docker Issues](#container-and-docker-issues)
10. [Logging and Debugging](#logging-and-debugging)

## Build and Compilation Issues

### Issue: TypeScript Compilation Fails

**Symptom**:
```
error TS2307: Cannot find module '@/schemas/loongflow-canonical'
```

**Cause**: TypeScript path mapping not configured correctly.

**Solution**:
1. Check `tsconfig.json`:
```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}
```

2. Ensure all dependencies are installed:
```bash
npm install
```

3. Clean and rebuild:
```bash
npm run clean
npm run build
```

### Issue: Module Not Found

**Symptom**:
```
Error: Cannot find module 'zod'
```

**Cause**: Missing dependencies.

**Solution**:
```bash
cd glue/adapters/loongflow-adapter
npm install

# Or if using workspace
npm install --workspace=@loongflow/loongflow-adapter
```

### Issue: Type Errors After Schema Changes

**Symptom**:
```
error TS2345: Argument of type 'string' is not assignable to parameter of type 'WorkflowId'
```

**Cause**: Schema changed but code not updated.

**Solution**:
1. Check schema definition:
```typescript
// glue/schemas/loongflow-canonical.ts
export const WorkflowIdSchema = z.string().uuid();
```

2. Update code to match schema:
```typescript
const workflowId: z.infer<typeof WorkflowIdSchema> = 'uuid-here';
```

3. Run type checking:
```bash
npx tsc --noEmit
```

## Test Failures

### Issue: Contract Tests Fail

**Symptom**:
```
FAIL tests/contract.test.ts
  ✓ Environment is configured
  ✕ LoongFlow API responds to /health
```

**Cause**: LoongFlow core is not running or not accessible.

**Solution**:
1. Check if LoongFlow core is running:
```bash
docker ps | grep loongflow-core
```

2. If not running, start it:
```bash
cd infra
docker-compose -f docker-compose.loongflow-core.yml up -d
```

3. Verify connectivity:
```bash
curl http://localhost:8050/health
```

4. Check adapter configuration:
```bash
echo $LOONGFLOW_API_URL
# Should be: http://loongflow-core:8050
```

### Issue: Schema Validation Tests Fail

**Symptom**:
```
✕ Schema validates LoongFlow response
  Invalid schema: missing property "workflowId"
```

**Cause**: Schema mismatch between API response and schema definition.

**Solution**:
1. Check actual API response:
```bash
curl http://loongflow-core:8050/workflow/execute -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"test"}' | jq .
```

2. Compare with schema definition:
```typescript
// glue/schemas/loongflow-canonical.ts
export const LoongFlowWorkflowResponseSchema = z.object({
  workflowId: z.string().uuid(),  // Check this matches API
  status: WorkflowStatusSchema,
  // ...
});
```

3. Update schema or API to match.

### Issue: Timeout Errors in Tests

**Symptom**:
```
Timeout - Async callback was not invoked within the 30000ms timeout
```

**Cause**: Tests timing out due to slow services.

**Solution**:
1. Increase test timeout in `jest.config.js`:
```javascript
module.exports = {
  testTimeout: 60000, // Increase from 30000 to 60000
};
```

2. Or skip slow tests:
```bash
npm test -- --testNamePattern="^((?!slow).)*$"
```

## Deployment Issues

### Issue: Docker Container Fails to Start

**Symptom**:
```
docker: Error response from daemon: Container loongflow-adapter exited with code 1.
```

**Cause**: Configuration error or missing environment variables.

**Solution**:
1. Check container logs:
```bash
docker logs loongflow-adapter
```

2. Common causes:
   - Missing environment variable
   - Invalid configuration
   - Port already in use

3. Verify environment variables:
```bash
docker exec loongflow-adapter env | grep LOONGFLOW
```

4. Check if port is available:
```bash
netstat -tuln | grep 8040
```

### Issue: Kubernetes Pod Not Ready

**Symptom**:
```
kubectl get pods
NAME                    READY   STATUS             RESTARTS   AGE
loongflow-adapter-xxx   0/1     CrashLoopBackOff   5          3m
```

**Cause**: Pod failing to start.

**Solution**:
1. Check pod logs:
```bash
kubectl logs loongflow-adapter-xxx
```

2. Describe pod for events:
```bash
kubectl describe pod loongflow-adapter-xxx
```

3. Common issues:
   - Image pull error (check image name and registry)
   - ConfigMap missing (check config maps are created)
   - Resource limits too low (increase CPU/memory)

4. Check resource usage:
```bash
kubectl top pod loongflow-adapter-xxx
```

### Issue: Service Not Accessible

**Symptom**:
```
curl: (7) Failed to connect to loongflow-adapter port 8040: Connection refused
```

**Cause**: Service not exposed or network issue.

**Solution**:
1. Check service exists:
```bash
kubectl get svc loongflow-adapter
```

2. Check service endpoints:
```bash
kubectl get endpoints loongflow-adapter
```

3. If no endpoints, check pod selector:
```bash
kubectl get svc loongflow-adapter -o yaml | grep selector -A 2
```

4. For Docker Compose, check ports mapping:
```yaml
services:
  loongflow-adapter:
    ports:
      - "8040:8040"  # Check this
```

## Runtime Errors

### Issue: Circuit Breaker Open

**Symptom**:
```
Error: Circuit breaker is OPEN for service 'loongflow-core'
```

**Cause**: Too many failures to LoongFlow core.

**Solution**:
1. Check LoongFlow core health:
```bash
curl http://loongflow-core:8050/health
```

2. Wait for circuit breaker timeout (default 60 seconds):
```typescript
// Circuit breaker will close after timeout
```

3. Reset circuit breaker manually (if needed):
```typescript
circuitBreaker.reset();
```

4. Adjust circuit breaker threshold:
```typescript
const cb = new CircuitBreaker({
  threshold: 10,  // Increase from 5 to 10
  timeout: 120000 // Increase timeout
});
```

### Issue: Dead Letter Queue Growing

**Symptom**:
DLQ has thousands of failed events.

**Cause**: Systematic failure in event processing.

**Solution**:
1. Inspect DLQ:
```bash
redis-cli LRANGE dlq:events 0 10
```

2. Identify failure pattern:
```bash
redis-cli LPOP dlq:events | jq .error
```

3. Fix underlying issue:
   - Schema mismatch → Update schema
   - Service down → Restart service
   - Validation error → Fix data

4. Process DLQ:
```bash
# Replay events after fixing issue
redis-cli LRANGE dlq:events 0 -1 | while read event; do
  # Process event
  redis-cli LPOP dlq:events
done
```

### Issue: Workflow Stuck in "Running" State

**Symptom**: Workflow never completes.

**Cause**: Hanging process or deadlock.

**Solution**:
1. Check workflow timeout:
```typescript
// Ensure timeout is set
const result = await adapter.executePESWorkflow({
  query: "...",
  timeout: 300000 // 5 minutes
});
```

2. Cancel workflow:
```bash
curl -X DELETE http://loongflow-adapter:8040/workflow/abc-123
```

3. Check LoongFlow core logs:
```bash
docker logs loongflow-core --tail 100
```

4. Restart LoongFlow core if needed:
```bash
docker restart loongflow-core
```

## Performance Issues

### Issue: Slow Workflow Execution

**Symptom**: Workflows take much longer than expected.

**Cause**: Network latency, slow services, or inefficient algorithms.

**Solution**:
1. Check service response times:
```bash
curl -w "@curl-format.txt" http://loongflow-core:8050/health
```

2. Enable checkpointing to skip completed work:
```typescript
await adapter.executePESWorkflow({
  query: "...",
  enableCheckpointing: true
});
```

3. Adjust workflow parameters:
```typescript
await adapter.executePESWorkflow({
  query: "...",
  maxIterations: 3  // Reduce from 5 to 3
});
```

4. Check for resource bottlenecks:
```bash
docker stats loongflow-core loongflow-adapter
```

### Issue: High Memory Usage

**Symptom**: Container memory usage keeps growing.

**Cause**: Memory leak or insufficient limits.

**Solution**:
1. Check container memory:
```bash
docker stats loongflow-adapter --no-stream
```

2. Increase memory limit:
```yaml
services:
  loongflow-adapter:
    deploy:
      resources:
        limits:
          memory: 2G  # Increase from 1G
```

3. Restart container periodically if leak persists:
```bash
docker restart loongflow-adapter
```

4. Profile memory usage:
```bash
node --inspect loongflow-adapter
# Connect Chrome DevTools for profiling
```

## Integration Issues

### Issue: LoongFlow Core Not Responding

**Symptom**:
```
Error: connect ECONNREFUSED 127.0.0.1:8050
```

**Cause**: LoongFlow core not running or wrong URL.

**Solution**:
1. Check if LoongFlow core is running:
```bash
docker ps | grep loongflow-core
```

2. If not running, start it:
```bash
cd infra
docker-compose -f docker-compose.loongflow-core.yml up -d
```

3. Check adapter configuration:
```bash
echo $LOONGFLOW_API_URL
# Should be: http://loongflow-core:8050 (not localhost)
```

4. For local development, use localhost:
```bash
export LOONGFLOW_API_URL=http://localhost:8050
```

### Issue: OpenEvolve Adapter Connection Refused

**Symptom**: Similar to above but for OpenEvolve.

**Solution**: Same steps as above, but for OpenEvolve:
```bash
docker ps | grep openevolve
docker-compose -f docker-compose-all-adapters.yml up -d openevolve-adapter
```

### Issue: Event Bus Connection Failed

**Symptom**:
```
Error: Redis connection to event-bus:6379 failed - connect ECONNREFUSED
```

**Cause**: Redis not running.

**Solution**:
1. Check if Redis is running:
```bash
docker ps | grep redis
```

2. Start Redis:
```bash
docker-compose up -d redis
```

3. Test connection:
```bash
redis-cli -h event-bus -p 6379 ping
# Should return: PONG
```

## Network and Connectivity Issues

### Issue: Container Cannot Reach Another Container

**Symptom**: Adapter cannot reach LoongFlow core.

**Cause**: Not on same Docker network.

**Solution**:
1. Check networks:
```bash
docker network ls
```

2. Ensure services share a network:
```yaml
services:
  loongflow-core:
    networks:
      - loongflow-network
  loongflow-adapter:
    networks:
      - loongflow-network
networks:
  loongflow-network:
    driver: bridge
```

3. Restart services:
```bash
docker-compose down
docker-compose up -d
```

### Issue: DNS Resolution Fails

**Symptom**:
```
Error: getaddrinfo ENOTFOUND loongflow-core
```

**Cause**: Docker DNS not resolving service names.

**Solution**:
1. Check Docker DNS:
```bash
docker exec loongflow-adapter nslookup loongflow-core
```

2. Use container IP instead:
```bash
docker inspect loongflow-core | grep IPAddress
export LOONGFLOW_API_URL=http://<IP>:8050
```

3. Restart Docker daemon (last resort):
```bash
sudo systemctl restart docker
```

## Data and Schema Issues

### Issue: Schema Validation Error

**Symptom**:
```
Error: Schema validation failed: .workflowId must be a UUID
```

**Cause**: Data doesn't match schema.

**Solution**:
1. Check schema:
```typescript
export const WorkflowIdSchema = z.string().uuid();
```

2. Validate data before sending:
```typescript
const result = WorkflowIdSchema.safeParse(workflowId);
if (!result.success) {
  console.error('Invalid workflow ID');
}
```

3. Transform data to match schema:
```typescript
const validId = workflowId.trim(); // Remove whitespace
```

### Issue: Data Loss Between Services

**Symptom**: Some fields missing in response.

**Cause**: Schema transformation dropping fields.

**Solution**:
1. Check transformation function:
```typescript
export function toCanonical(response: LoongFlowResponse) {
  return {
    workflowId: response.workflow_id,  // Check mapping
    // ...
  };
}
```

2. Add passthrough for unknown fields:
```typescript
export const CanonicalSchema = z.object({
  // ...
}).passthrough(); // Allow additional fields
```

3. Log before and after transformation:
```typescript
console.log('Before:', response);
const canonical = toCanonical(response);
console.log('After:', canonical);
```

## Container and Docker Issues

### Issue: Image Build Fails

**Symptom**:
```
ERROR [builder] failed to solve: executor failed running [/bin/sh -c npm install]
```

**Cause**: Build error in Dockerfile.

**Solution**:
1. Check Dockerfile:
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install  # This step failing
COPY . .
RUN npm run build
```

2. Check package.json exists:
```bash
ls -la glue/adapters/loongflow-adapter/package.json
```

3. Build with no cache:
```bash
docker build --no-cache -t loongflow-adapter .
```

4. Check build logs:
```bash
docker build -t loongflow-adapter . 2>&1 | tee build.log
```

### Issue: Container Exits Immediately

**Symptom**:
```
docker run loongflow-adapter
# Container exits immediately
```

**Cause**: Main process exits or error in startup.

**Solution**:
1. Check entrypoint:
```dockerfile
ENTRYPOINT ["node", "dist/index.js"]  # Check this exists
```

2. Run with interactive shell:
```bash
docker run -it loongflow-adapter sh
# Then manually run the command
node dist/index.js
```

3. Check for missing files:
```bash
docker run loongflow-adapter ls -la dist/
```

4. Add health check to catch issues:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD node -e "require('http').get('http://localhost:8040/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"
```

## Logging and Debugging

### Enable Debug Logging

**Adapter**:
```bash
export LOG_LEVEL=debug
npm start
```

**LoongFlow Core**:
```bash
export LOONGFLOW_LOG_LEVEL=DEBUG
docker-compose up loongflow-core
```

### View Logs

**Docker Compose**:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f loongflow-adapter

# Last 100 lines
docker-compose logs --tail=100 loongflow-adapter
```

**Kubernetes**:
```bash
# Pod logs
kubectl logs -f loongflow-adapter-xxx

# All pods in deployment
kubectl logs -f -l app=loongflow-adapter
```

### Structured Log Analysis

Logs are in JSON format for easy parsing:

```bash
# Extract errors
docker-compose logs loongflow-adapter | jq 'select(.level == "error")'

# Extract by correlation ID
docker-compose logs loongflow-adapter | jq 'select(.correlation_id == "abc-123")'

# Extract slow requests
docker-compose logs loongflow-adapter | jq 'select(.duration > 1000)'
```

### Common Log Patterns

**Successful workflow**:
```json
{
  "level": "info",
  "msg": "Workflow completed",
  "workflow_id": "abc-123",
  "duration": 5000,
  "correlation_id": "xyz-789"
}
```

**Failed workflow**:
```json
{
  "level": "error",
  "msg": "Workflow execution failed",
  "workflow_id": "abc-123",
  "error": "Connection timeout",
  "correlation_id": "xyz-789"
}
```

**Circuit breaker opened**:
```json
{
  "level": "warning",
  "msg": "Circuit breaker opened",
  "service": "loongflow-core",
  "failure_count": 5,
  "threshold": 5
}
```

---

**Last Updated**: 2024-02-22
**For more help**: See [DEVELOPMENT.md](./DEVELOPMENT.md) or create an issue on GitHub
