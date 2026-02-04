# Knowledge Engine Integration Best Practices

This guide provides comprehensive best practices for working with Knowledge Engine integrations, covering selection strategies, common patterns, performance optimization, error handling, and security considerations.

## Table of Contents

1. [Choosing the Right Integration](#choosing-the-right-integration)
2. [Common Patterns](#common-patterns)
3. [Performance Optimization](#performance-optimization)
4. [Error Handling Strategies](#error-handling-strategies)
5. [Security Best Practices](#security-best-practices)
6. [Testing Strategies](#testing-strategies)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Choosing the Right Integration

### Decision Tree

```
Start
  |
  v
What is your primary task?
  |
  +-- Reasoning/Problem Solving
  |    |
  |    +-- Simple reasoning -> DSPy
  |    +-- Multi-agent collaboration -> CrewAI
  |    +-- Complex decomposition -> ROMA
  |    +-- Adaptive learning -> ACE
  |
  +-- Knowledge Extraction
  |    |
  |    +-- Entity/Relation extraction -> DeepKE
  |    +-- General extraction -> OneKE
  |    +-- Document-level -> DeepKE (document mode)
  |
  +-- Knowledge Retrieval
  |    |
  |    +-- Document search -> Ragbits
  |    +-- Semantic search -> Knowledge Engine (native)
  |    +-- Graph traversal -> ROMA EKG, Graphiti
  |
  +-- Knowledge Representation
  |    |
  |    +-- Temporal knowledge -> Graphiti
  |    +-- Entity graphs -> ROMA EKG
  |    +-- Neural embeddings -> NeuralKG
  |    +-- Graph analytics -> Karate Club
  |
  +-- Mathematical/Formal Reasoning
  |    |
  |    +-- SMT solving -> Z3 Prover
  |    +-- Proof assistance -> LeanAIDE
  |    +-- Cross-system -> Unified Math Bridge
  |
  +-- Research & Discovery
  |    |
  |    +-- Literature review -> Research Quest
  |    +-- Evolutionary search -> Unified Evolution
  |    +-- Causal discovery -> CausalLearn
```

### Integration Comparison Matrix

| Integration | Primary Use | Complexity | Resource Usage | Best For |
|-------------|-------------|------------|----------------|----------|
| **DSPy** | Reasoning | Low | Medium | Step-by-step reasoning, problem solving |
| **DeepKE** | Extraction | Medium | High (GPU) | Knowledge graph construction |
| **CrewAI** | Orchestration | Medium | High | Multi-agent workflows |
| **Ragbits** | Retrieval | Low | Medium | Document search and RAG |
| **ROMA** | Decomposition | High | High | Complex problem solving |
| **ACE** | Learning | High | High | Adaptive systems |
| **Z3** | Formal Reasoning | High | Low | Theorem proving, constraints |
| **LeanAIDE** | Proof Assistance | High | Medium | Mathematical proofs |
| **Graphiti** | Temporal KG | Medium | Medium | Time-based knowledge |

### Use Case Examples

#### 1. Building a Question-Answering System

```python
from knowledge_engine.integrations import RagbitsIntegration, DSPyIntegration

# Best: Ragbits for retrieval + DSPy for reasoning
ragbits = RagbitsIntegration()
dspy = DSPyIntegration()

# Retrieve relevant documents
docs = await ragbits.search(query="What is machine learning?", top_k=5)

# Reason about the answer
answer = await dspy.chain_of_thought(
    query=f"Based on these documents: {docs}, answer: What is machine learning?"
)
```

#### 2. Building a Knowledge Graph from Documents

```python
from knowledge_engine.integrations import DeepKEIntegration, ROMAEntityExtractor

# Best: DeepKE for extraction + ROMA EKG for storage
deepke = DeepKEIntegration()
roma = ROMAEntityExtractor()

# Extract entities and relations
result = await deepke.extract_entities_relations(document)

# Store in knowledge graph
for entity in result.entities:
    await roma.add_entity(
        entity_type=entity["type"],
        name=entity["text"]
    )

for relation in result.relations:
    await roma.add_relation(
        from_entity=relation["head"],
        relation_type=relation["relation"],
        to_entity=relation["tail"]
    )
```

#### 3. Solving Complex Multi-Step Problems

```python
from knowledge_engine.integrations import ROMAIntegration, DSPyIntegration

# Best: ROMA for decomposition + DSPy for reasoning
roma = ROMAIntegration()
dspy = DSPyIntegration()

# Decompose problem
decomposition = await roma.decompose(
    problem="Design a scalable distributed system"
)

# Solve each subproblem with DSPy
solutions = []
for subproblem in decomposition.subproblems:
    solution = await dspy.chain_of_thought(
        query=subproblem["description"]
    )
    solutions.append(solution)

# Synthesize solutions
final = await roma.synthesize(solutions)
```

---

## Common Patterns

### Pattern 1: Retrieval-Augmented Generation (RAG)

```python
class RAGPipeline:
    def __init__(self):
        self.retriever = RagbitsIntegration()
        self.generator = DSPyIntegration()

    async def query(self, question: str) -> str:
        # 1. Retrieve relevant context
        context = await self.retriever.search(
            query=question,
            top_k=5
        )

        # 2. Generate answer with context
        answer = await self.generator.chain_of_thought(
            query=question,
            context={"documents": context}
        )

        return answer.output
```

### Pattern 2: Knowledge Extraction Pipeline

```python
class ExtractionPipeline:
    def __init__(self):
        self.extractor = DeepKEIntegration()
        self.graph = ROMAEntityExtractor()

    async def process(self, documents: List[str]):
        for doc in documents:
            # Extract entities and relations
            result = await self.extractor.extract_entities_relations(doc)

            # Store in knowledge graph
            await self._store_in_graph(result)

    async def _store_in_graph(self, result):
        # Store entities
        for entity in result.entities:
            await self.graph.add_entity(
                entity_type=entity["type"],
                name=entity["text"],
                properties={"confidence": entity["confidence"]}
            )

        # Store relations
        for relation in result.relations:
            await self.graph.add_relation(
                from_entity=relation["head"],
                relation_type=relation["relation"],
                to_entity=relation["tail"]
            )
```

### Pattern 3: Multi-Agent Collaboration

```python
class CollaborativeSolving:
    def __init__(self):
        self.coordinator = CrewAIIntegration()
        self.roma = ROMAIntegration()

    async def solve(self, problem: str):
        # 1. Decompose problem with ROMA
        decomposition = await self.roma.decompose(problem)

        # 2. Create agents for each subproblem
        agents = []
        tasks = []
        for subproblem in decomposition.subproblems:
            agent = self._create_agent_for_task(subproblem)
            task = self._create_task_from_subproblem(subproblem)
            agents.append(agent)
            tasks.append(task)

        # 3. Coordinate with CrewAI
        await self.coordinator.create_crew(
            crew_id="problem_solvers",
            agents=agents,
            tasks=tasks,
            process="hierarchical"
        )

        # 4. Execute and return result
        result = await self.coordinator.execute_crew(
            crew_id="problem_solvers"
        )

        return result
```

### Pattern 4: Ensemble Reasoning

```python
class EnsembleReasoner:
    def __init__(self):
        self.dspy = DSPyIntegration()
        self.roma = ROMAIntegration()
        self.ace = AgenticContextEngine()

    async def reason(self, problem: str):
        # Get multiple perspectives
        dspy_result = await self.dspy.chain_of_thought(problem)
        roma_result = await self.roma.solve(problem)
        ace_result = await self.ace.process(problem)

        # Combine results
        combined = self._combine_results([
            dspy_result,
            roma_result,
            ace_result
        ])

        return combined

    def _combine_results(self, results):
        # Voting or weighted combination
        # Implementation depends on use case
        pass
```

### Pattern 5: Incremental Knowledge Building

```python
class KnowledgeBuilder:
    def __init__(self):
        self.extractor = DeepKEIntegration()
        self.graph = ROMAEntityExtractor()
        self.ace = AgenticContextEngine()

    async def add_knowledge(self, text: str):
        # Extract new knowledge
        result = await self.extractor.extract_entities_relations(text)

        # Check for conflicts with existing knowledge
        conflicts = await self._check_conflicts(result)

        if conflicts:
            # Resolve conflicts with ACE
            resolved = await self.ace.resolve_conflicts(conflicts)
            result = self._apply_resolution(result, resolved)

        # Add to graph
        await self._store_in_graph(result)

        # Learn from new knowledge
        await self.ace.learn(result)
```

---

## Performance Optimization

### 1. Caching Strategies

```python
from functools import lru_cache
import hashlib

class CachedIntegration:
    def __init__(self, integration):
        self.integration = integration
        self.cache_enabled = True
        self.cache_ttl = 3600

    @lru_cache(maxsize=1000)
    async def cached_call(self, method: str, *args, **kwargs):
        # Create cache key from arguments
        key = self._make_cache_key(method, args, kwargs)

        # Check cache
        if self.cache_enabled:
            cached = await self._get_from_cache(key)
            if cached:
                return cached

        # Call integration
        result = await getattr(self.integration, method)(*args, **kwargs)

        # Store in cache
        if self.cache_enabled:
            await self._store_in_cache(key, result)

        return result

    def _make_cache_key(self, method, args, kwargs):
        # Create deterministic key
        data = f"{method}:{args}:{kwargs}"
        return hashlib.md5(data.encode()).hexdigest()
```

### 2. Batch Processing

```python
class BatchProcessor:
    def __init__(self, integration, batch_size=10):
        self.integration = integration
        self.batch_size = batch_size

    async def process_batch(self, items: List[Any]):
        results = []

        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]

            # Parallel processing within batch
            batch_results = await asyncio.gather(*[
                self.integration.process(item)
                for item in batch
            ])

            results.extend(batch_results)

        return results
```

### 3. Connection Pooling

```python
class PooledIntegration:
    def __init__(self, integration_class, pool_size=5):
        self.pool = asyncio.Queue(maxsize=pool_size)
        self.integration_class = integration_class

        # Initialize pool
        for _ in range(pool_size):
            integration = integration_class()
            self.pool.put_nowait(integration)

    async def process(self, item):
        # Get integration from pool
        integration = await self.pool.get()

        try:
            # Process item
            result = await integration.process(item)
            return result
        finally:
            # Return integration to pool
            self.pool.put_nowait(integration)
```

### 4. Lazy Loading

```python
class LazyIntegration:
    def __init__(self, integration_class, config=None):
        self.integration_class = integration_class
        self.config = config
        self._integration = None

    @property
    def integration(self):
        if self._integration is None:
            self._integration = self.integration_class(self.config)
        return self._integration

    async def process(self, item):
        return await self.integration.process(item)
```

### 5. Resource Monitoring

```python
import psutil
import time

class MonitoredIntegration:
    def __init__(self, integration):
        self.integration = integration
        self.max_memory = 4 * 1024 * 1024 * 1024  # 4GB
        self.max_cpu = 80  # 80%

    async def process(self, item):
        # Check resources before processing
        if not self._check_resources():
            raise ResourceWarning("Insufficient resources")

        # Monitor during processing
        start_time = time.time()
        start_memory = psutil.virtual_memory().used

        result = await self.integration.process(item)

        # Log resource usage
        elapsed = time.time() - start_time
        memory_used = psutil.virtual_memory().used - start_memory

        logger.info({
            "processing_time": elapsed,
            "memory_used": memory_used,
            "item": str(item)
        })

        return result

    def _check_resources(self):
        memory_ok = psutil.virtual_memory().available > self.max_memory
        cpu_ok = psutil.cpu_percent() < self.max_cpu
        return memory_ok and cpu_ok
```

---

## Error Handling Strategies

### 1. Graceful Degradation

```python
class ResilientIntegration:
    def __init__(self, primary, fallbacks):
        self.primary = primary
        self.fallbacks = fallbacks

    async def process(self, item):
        # Try primary
        try:
            result = await self.primary.process(item)
            if result.success:
                return result
        except Exception as e:
            logger.warning(f"Primary failed: {e}")

        # Try fallbacks in order
        for fallback in self.fallbacks:
            try:
                result = await fallback.process(item)
                if result.success:
                    logger.info(f"Fallback {fallback} succeeded")
                    return result
            except Exception as e:
                logger.warning(f"Fallback {fallback} failed: {e}")

        # All failed
        return self._get_default_result(item)

    def _get_default_result(self, item):
        # Return safe default
        return Result(success=False, output=None, error="All integrations failed")
```

### 2. Retry with Exponential Backoff

```python
import asyncio

class RetryIntegration:
    def __init__(self, integration, max_retries=3, base_delay=1.0):
        self.integration = integration
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def process(self, item):
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await self.integration.process(item)
                if result.success:
                    return result
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                # Exponential backoff
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        # All retries failed
        raise last_error
```

### 3. Circuit Breaker Pattern

```python
class CircuitBreakerIntegration:
    def __init__(self, integration, failure_threshold=5, timeout=60):
        self.integration = integration
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    async def process(self, item):
        # Check circuit state
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise CircuitBreakerOpenError("Circuit is open")

        try:
            result = await self.integration.process(item)

            # Success: reset failures and close circuit
            self.failures = 0
            self.state = "closed"
            return result

        except Exception as e:
            # Failure: increment counter
            self.failures += 1
            self.last_failure_time = time.time()

            # Open circuit if threshold reached
            if self.failures >= self.failure_threshold:
                self.state = "open"
                logger.error("Circuit breaker opened")

            raise e
```

### 4. Validation Middleware

```python
class ValidatedIntegration:
    def __init__(self, integration, validators):
        self.integration = integration
        self.validators = validators

    async def process(self, item):
        # Validate input
        for validator in self.validators.get("input", []):
            if not validator(item):
                raise ValidationError(f"Input validation failed: {validator.__name__}")

        # Process
        result = await self.integration.process(item)

        # Validate output
        for validator in self.validators.get("output", []):
            if not validator(result):
                logger.error(f"Output validation failed: {validator.__name__}")
                # Return error or raise exception based on policy

        return result
```

---

## Security Best Practices

### 1. API Key Management

```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.key = os.environ.get("ENCRYPTION_KEY")
        self.cipher = Fernet(self.key)

    def get_api_key(self, service):
        # Get encrypted key from secure storage
        encrypted_key = os.environ.get(f"{service}_API_KEY_ENCRYPTED")

        if not encrypted_key:
            raise ValueError(f"No API key found for {service}")

        # Decrypt
        decrypted = self.cipher.decrypt(encrypted_key.encode())
        return decrypted.decode()

# Usage
config = SecureConfig()
openai_key = config.get_api_key("OPENAI")
```

### 2. Input Sanitization

```python
import re

class SanitizedIntegration:
    def __init__(self, integration):
        self.integration = integration

    async def process(self, text: str):
        # Sanitize input
        sanitized = self._sanitize(text)

        # Process
        return await self.integration.process(sanitized)

    def _sanitize(self, text: str) -> str:
        # Remove potentially harmful content
        # Remove SQL injection patterns
        text = re.sub(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b)", "", text, flags=re.IGNORECASE)

        # Remove script tags
        text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)

        # Limit length
        text = text[:10000]

        return text
```

### 3. Rate Limiting

```python
import time
from collections import defaultdict

class RateLimitedIntegration:
    def __init__(self, integration, max_requests=100, window=60):
        self.integration = integration
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)

    async def process(self, item):
        # Check rate limit
        now = time.time()
        client_id = self._get_client_id(item)

        # Remove old requests outside window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window
        ]

        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.max_requests:
            raise RateLimitError("Rate limit exceeded")

        # Record request
        self.requests[client_id].append(now)

        # Process
        return await self.integration.process(item)

    def _get_client_id(self, item):
        # Extract client ID from item
        return item.get("client_id", "default")
```

### 4. Audit Logging

```python
class AuditLoggedIntegration:
    def __init__(self, integration, audit_logger):
        self.integration = integration
        self.audit_logger = audit_logger

    async def process(self, item, user_id=None):
        # Log before processing
        await self.audit_logger.log({
            "event": "integration_call",
            "integration": self.integration.__class__.__name__,
            "user_id": user_id,
            "timestamp": time.time(),
            "item": str(item)
        })

        try:
            result = await self.integration.process(item)

            # Log success
            await self.audit_logger.log({
                "event": "integration_success",
                "integration": self.integration.__class__.__name__,
                "user_id": user_id,
                "timestamp": time.time()
            })

            return result

        except Exception as e:
            # Log failure
            await self.audit_logger.log({
                "event": "integration_failure",
                "integration": self.integration.__class__.__name__,
                "user_id": user_id,
                "error": str(e),
                "timestamp": time.time()
            })

            raise e
```

---

## Testing Strategies

### 1. Unit Testing with Mocks

```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_dspy_integration():
    # Create mock integration
    mock_dspy = Mock()
    mock_dspy.chain_of_thought = AsyncMock(
        return_value=DSPyResult(
            success=True,
            output="42",
            reasoning="Step-by-step reasoning..."
        )
    )

    # Test with mock
    result = await mock_dspy.chain_of_thought("What is 6 * 7?")

    assert result.success is True
    assert result.output == "42"
    mock_dspy.chain_of_thought.assert_called_once()
```

### 2. Integration Testing with Test Doubles

```python
@pytest.mark.asyncio
async def test_rag_pipeline():
    # Use test double for retriever
    mock_retriever = Mock()
    mock_retriever.search = AsyncMock(
        return_value=["doc1", "doc2", "doc3"]
    )

    # Use test double for generator
    mock_generator = Mock()
    mock_generator.chain_of_thought = AsyncMock(
        return_value=DSPyResult(
            success=True,
            output="Test answer",
            reasoning="Test reasoning"
        )
    )

    # Create pipeline with test doubles
    pipeline = RAGPipeline()
    pipeline.retriever = mock_retriever
    pipeline.generator = mock_generator

    # Test
    result = await pipeline.query("Test question")

    assert result == "Test answer"
    mock_retriever.search.assert_called_once()
    mock_generator.chain_of_thought.assert_called_once()
```

### 3. Contract Testing

```python
def test_integration_contract():
    """Test that integration conforms to expected interface"""

    integration = DSPyIntegration()

    # Check required methods exist
    assert hasattr(integration, "chain_of_thought")
    assert hasattr(integration, "program_of_thought")

    # Check method signatures
    import inspect
    sig = inspect.signature(integration.chain_of_thought)
    params = list(sig.parameters.keys())
    assert "query" in params
    assert "context" in params
```

### 4. Performance Testing

```python
import time

@pytest.mark.asyncio
async def test_integration_performance():
    integration = DSPyIntegration()

    # Measure response time
    start = time.time()
    result = await integration.chain_of_thought("Simple question")
    elapsed = time.time() - start

    # Assert response time is acceptable
    assert elapsed < 5.0  # Should respond in less than 5 seconds

    # Measure throughput
    questions = ["q1", "q2", "q3"] * 10
    start = time.time()
    results = await asyncio.gather(*[
        integration.chain_of_thought(q) for q in questions
    ])
    elapsed = time.time() - start

    throughput = len(questions) / elapsed
    assert throughput > 0.5  # At least 0.5 questions per second
```

---

## Monitoring and Observability

### 1. Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
integration_requests = Counter(
    'integration_requests_total',
    'Total integration requests',
    ['integration', 'status']
)

integration_latency = Histogram(
    'integration_latency_seconds',
    'Integration request latency',
    ['integration']
)

integration_active = Gauge(
    'integration_active_requests',
    'Active integration requests',
    ['integration']
)

class MonitoredIntegration:
    def __init__(self, integration):
        self.integration = integration
        self.name = integration.__class__.__name__

    async def process(self, item):
        integration_active.labels(self.name).inc()

        try:
            start = time.time()

            result = await self.integration.process(item)

            latency = time.time() - start
            integration_latency.labels(self.name).observe(latency)

            if result.success:
                integration_requests.labels(self.name, 'success').inc()
            else:
                integration_requests.labels(self.name, 'failure').inc()

            return result

        finally:
            integration_active.labels(self.name).dec()
```

### 2. Distributed Tracing

```python
from opentelemetry import trace

class TracedIntegration:
    def __init__(self, integration):
        self.integration = integration
        self.tracer = trace.get_tracer(__name__)

    async def process(self, item):
        with self.tracer.start_as_current_span(
            f"{self.integration.__class__.__name__}.process"
        ) as span:
            # Add attributes
            span.set_attribute("integration", self.integration.__class__.__name__)
            span.set_attribute("item", str(item)[:100])

            # Process
            result = await self.integration.process(item)

            # Add result attributes
            span.set_attribute("success", result.success)
            if result.error:
                span.set_attribute("error", result.error)

            return result
```

---

## Anti-Patterns to Avoid

### 1. Don't Ignore Optional Dependencies

```python
# BAD: Will crash if DeepKE not installed
from knowledge_engine.integrations import DeepKEIntegration
deepke = DeepKEIntegration()

# GOOD: Check availability
from knowledge_engine.integrations import DeepKEIntegration

try:
    deepke = DeepKEIntegration()
except ImportError:
    logger.warning("DeepKE not available, using fallback")
    deepke = FallbackExtractor()
```

### 2. Don't Hard-Throw on Errors

```python
# BAD: Will crash on any error
result = await integration.process(item)
if not result.success:
    raise Exception("Processing failed")

# GOOD: Handle gracefully
result = await integration.process(item)
if not result.success:
    logger.error(f"Processing failed: {result.error}")
    # Use fallback or return default
    return get_default_result()
```

### 3. Don't Assume Success

```python
# BAD: Assume success
result = await integration.process(item)
output = result.output  # Might be None!

# GOOD: Check success
result = await integration.process(item)
if result.success:
    output = result.output
else:
    handle_error(result.error)
```

### 4. Don't Ignore Configuration

```python
# BAD: Use hardcoded config
integration = DSPyIntegration(config={
    "model": "gpt-4o",
    "api_key": "hardcoded-key"  # SECURITY RISK!
})

# GOOD: Use environment variables
import os
integration = DSPyIntegration(config={
    "model": os.getenv("MODEL", "gpt-4o"),
    "api_key": os.getenv("OPENAI_API_KEY")
})
```

### 5. Don't Forget Resource Cleanup

```python
# BAD: Resources not cleaned up
integration = DSPyIntegration()
result = await integration.process(item)
# Function ends, resources not released

# GOOD: Use context manager
async with DSPyIntegration() as integration:
    result = await integration.process(item)
# Resources automatically cleaned up
```

---

## Conclusion

Following these best practices will help you build robust, efficient, and secure systems using Knowledge Engine integrations. Key takeaways:

1. **Choose the right tool** for the job based on your use case
2. **Use common patterns** like RAG, pipelines, and orchestration
3. **Optimize performance** with caching, batching, and pooling
4. **Handle errors gracefully** with retries, fallbacks, and circuit breakers
5. **Prioritize security** with proper API key management and input validation
6. **Test thoroughly** with unit, integration, and performance tests
7. **Monitor continuously** with metrics, logs, and tracing
8. **Avoid anti-patterns** that lead to fragile systems

For more specific guidance, see the individual integration documentation.

---

**Last Updated**: 2025-02-03
**Version**: 1.0.0
