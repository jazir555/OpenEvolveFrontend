# LeanAide Adapter

## Purpose

Anti-Corruption Layer (ACL) for LeanAide AI Assistant integration.

## Responsibilities

1. **Normalize Data**: Convert LeanAide's data format to Canonical Schema
2. **Failure Management**: Implement circuit breakers and retries
3. **API Verification**: Probe LeanAide's API to verify availability
4. **Contract Enforcement**: Ensure LeanAide doesn't break our integration

## Directory Structure

```
leanaide-adapter/
├── src/              # Adapter implementation code
├── probes/           # API discovery scripts (run BEFORE implementing)
├── tests/            # Contract tests (run on startup)
└── Dockerfile        # Containerize the adapter
```

## Protocol

### Phase 1: Probe (Discovery)

Before writing any code:

1. Create `probes/check_api.sh`
2. Execute against running LeanAide container
3. Verify API returns expected fields
4. **IF PROBE FAILS**: Do not write adapter code

### Phase 2: Implement

After successful probe:

1. Define Canonical Schema in `/glue/schemas/`
2. Write adapter with:
   - Circuit breaker (stop hammering if LeanAide is down)
   - Exponential backoff retry (with jitter)
   - Request timeout (MANDATORY: 10000ms default for AI)
   - Structured logging (correlation_id, source_service, target_service)

### Phase 3: Contract Test

Protect against upstream changes:

1. Create `tests/contract.test.ts`
2. Assert on all fields we depend on
3. Test runs on container startup
4. **IF CONTRACT FAILS**: Adapter refuses to start

## Integration Points

LeanAide provides:
- AI-powered proof assistance
- Natural language to Lean translation
- Proof strategy suggestions

We expose via Canonical Schema:
- Standardized proof request format
- Normalized AI response structure
- Unified confidence scoring

## Failure Scenarios

| Scenario | Strategy |
|----------|----------|
| Network timeout | Exponential backoff retry (max 2 attempts - AI is slow) |
| LeanAide container down | Circuit breaker opens, stop requests |
| Invalid query | Return to DLQ, don't block pipeline |
| API changed | Contract test fails, adapter refuses start |
| AI hallucination | Log to DLQ for review, continue processing |

## Configuration

Required environment variables:

```bash
LEANAIDE_API_URL=http://leanaide-core:8000
LEANAIDE_TIMEOUT_MS=10000
LEANAIDE_RETRY_MAX=2
LEANAIDE_CIRCUIT_BREAKER_THRESHOLD=5
LEANAIDE_MAX_TOKENS=4000
```

## Logs

All logs must include:
```json
{
  "correlation_id": "uuid-here",
  "source_service": "leanaide-adapter",
  "target_service": "leanaide-core",
  "msg": "Proof suggestion completed",
  "confidence_score": 0.95,
  "tokens_used": 1234,
  "duration_ms": 5678
}
```

## AI-Specific Considerations

1. **Longer Timeouts**: AI requests take longer (10s default vs 5s for normal APIs)
2. **Fewer Retries**: Don't spam expensive AI calls (max 2 retries vs 3 for normal)
3. **Token Tracking**: Log token usage for cost monitoring
4. **Confidence Scoring**: Normalize AI confidence to 0-1 scale
5. **Hallucination Handling**: Log unexpected responses to DLQ

## See Also

- LeanAide upstream documentation: [link]
- Canonical Schema: `/glue/schemas/leanaide-schema.ts`
- ADR: `/glue/adapters/leanaide-adapter/ADR.md`
