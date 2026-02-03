# Troubleshooting Guide

Common issues and solutions for the OpenEvolve Knowledge Engine.

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [Connection Issues](#connection-issues)
3. [Performance Issues](#performance-issues)
4. [Extraction Issues](#extraction-issues)
5. [Query Issues](#query-issues)
6. [Memory Issues](#memory-issues)
7. [Neo4j Issues](#neo4j-issues)

## Installation Issues

### Issue: Import Error

**Symptom**:
```
ImportError: No module named 'knowledge_engine'
```

**Solutions**:

1. **Check Python path**:
```bash
python -c "import sys; print('\n'.join(sys.path))"
```

2. **Install in development mode**:
```bash
cd /path/to/Frontend
pip install -e .
```

3. **Check virtual environment**:
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Reinstall
pip install -r requirements.txt
```

### Issue: Dependency Conflict

**Symptom**:
```
ERROR: pip's dependency resolver does not currently take into account...
```

**Solutions**:

1. **Use pip-tools**:
```bash
pip install pip-tools
pip-compile requirements.txt
pip-sync
```

2. **Create clean environment**:
```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements.txt
```

3. **Update pip**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Connection Issues

### Issue: Neo4j Connection Failed

**Symptom**:
```
Failed to initialize Graphiti bridge: Connection refused
```

**Solutions**:

1. **Check Neo4j is running**:
```bash
# Systemd
systemctl status neo4j

# Docker
docker ps | grep neo4j

# Manual
ps aux | grep neo4j
```

2. **Test connection**:
```bash
# Using neo4j-client
neo4j-client -u neo4j -p password bolt://localhost:7687

# Using telnet
telnet localhost 7687
```

3. **Check configuration**:
```yaml
# integrations/graphiti/config.yaml
neo4j:
  uri: bolt://localhost:7687  # Correct URI
  username: neo4j             # Correct username
  password: your_password     # Correct password
```

4. **Check firewall**:
```bash
# Allow Neo4j port
sudo ufw allow 7687/tcp
# or
sudo firewall-cmd --add-port=7687/tcp --permanent
```

### Issue: OpenAI API Error

**Symptom**:
```
openai.error.AuthenticationError: Incorrect API key provided
```

**Solutions**:

1. **Set environment variable**:
```bash
export OPENAI_API_KEY="sk-..."
# Windows
set OPENAI_API_KEY=sk-...
```

2. **Check in Python**:
```python
import os
print(os.getenv('OPENAI_API_KEY'))
```

3. **Test API key**:
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

## Performance Issues

### Issue: Slow Extraction

**Symptom**: Knowledge graph extraction takes too long

**Diagnosis**:
```python
import time

start = time.time()
graph = await pipeline.extract_knowledge_graph(text)
duration = time.time() - start

print(f"Extraction took {duration:.2f}s")
print(f"Text length: {len(text)} chars")
print(f"Extraction speed: {len(text)/duration:.0f} chars/s")
```

**Solutions**:

1. **Reduce chunk size**:
```yaml
# config/kggen_pipeline.yaml
default_chunk_size: 3000  # Reduce from 5000
```

2. **Increase parallel workers**:
```yaml
parallel_workers: 8  # Increase from 4
```

3. **Use caching**:
```python
# Enable caching
text_hash = hashlib.md5(text.encode()).hexdigest()
graph = await pipeline.extract_cached(text_hash)
```

4. **Use faster model**:
```yaml
stages:
  entity_extraction:
    model: openai/gpt-3.5-turbo  # Instead of gpt-4
```

### Issue: Slow Queries

**Symptom**: Queries take several seconds

**Diagnosis**:
```python
import time

start = time.time()
results = await engine.search_with_graphiti(query="test")
duration = time.time() - start

print(f"Query took {duration:.2f}s")
```

**Solutions**:

1. **Create indices**:
```cypher
# In Neo4j
CREATE INDEX entity_name FOR (e:Entity) ON (e.name);
CREATE INDEX rel_type FOR ()-[r:RELATED]->() ON (r.type);
```

2. **Use specific filters**:
```python
# Good: Specific group
results = await engine.search_with_graphiti(
    query="test",
    group_ids=["specific_group"]
)

# Bad: All knowledge
results = await engine.search_with_graphiti(
    query="test"
)
```

3. **Limit results**:
```python
results = await engine.search_with_graphiti(
    query="test",
    max_results=10  # Instead of 100
)
```

4. **Disable hybrid search**:
```python
# If not needed
results = await engine.search_with_graphiti(
    query="test",
    use_hybrid=False
)
```

## Extraction Issues

### Issue: No Entities Extracted

**Symptom**: `graph.entities` is empty

**Diagnosis**:
```python
graph = await pipeline.extract_knowledge_graph(text)
print(f"Entities: {len(graph.entities)}")
print(f"Text length: {len(text)}")

# Check entity extraction step
entities = await pipeline._extract_entities(text, "")
print(f"Raw entities: {entities}")
```

**Solutions**:

1. **Provide context**:
```python
graph = await pipeline.extract_knowledge_graph(
    text=text,
    context="This is a document about Apple Inc."  # Add context
)
```

2. **Check text quality**:
```python
# Ensure text is not empty or too short
if len(text) < 100:
    print("Text too short for extraction")

# Ensure text is not all special characters
import re
clean_text = re.sub(r'[^\w\s]', '', text)
if len(clean_text) < len(text) * 0.5:
    print("Text has too many special characters")
```

3. **Adjust extraction parameters**:
```yaml
stages:
  entity_extraction:
    temperature: 0.2  # Increase from 0.0
    max_tokens: 4000  # Ensure enough tokens
```

### Issue: Poor Quality Relationships

**Symptom**: Extracted relationships don't make sense

**Diagnosis**:
```python
graph = await pipeline.extract_knowledge_graph(text)

for subj, pred, obj in graph.relationships[:10]:
    print(f"{subj} -> {pred} -> {obj}")
```

**Solutions**:

1. **Improve context**:
```python
graph = await pipeline.extract_knowledge_graph(
    text=text,
    context="Technical documentation about software architecture"
)
```

2. **Use domain-specific model**:
```yaml
stages:
  relation_extraction:
    model: local/biobert  # For biomedical
    # or
    model: local/codebert  # For code
```

3. **Adjust temperature**:
```yaml
stages:
  relation_extraction:
    temperature: 0.1  # Slightly higher for creativity
```

## Query Issues

### Issue: No Query Results

**Symptom**: Query returns empty list

**Diagnosis**:
```python
# Check if knowledge exists
all_knowledge = await engine.get_valid_knowledge()
print(f"Total knowledge: {len(all_knowledge)}")

# Check query
results = await engine.search_with_graphiti(query="test")
print(f"Results: {len(results)}")
```

**Solutions**:

1. **Check temporal filters**:
```python
# Don't filter by time
results = await engine.search_with_graphiti(
    query="test",
    temporal_filters={"filter_type": TemporalFilter.ALL}
)
```

2. **Use broader query**:
```python
# Instead of specific term
results = await engine.search_with_graphiti(query="authentication")

# Use broader term
results = await engine.search_with_graphiti(query="security")
```

3. **Check group IDs**:
```python
# Don't filter by group
results = await engine.search_with_graphiti(
    query="test",
    group_ids=None  # All groups
)
```

### Issue: Inconsistent Results

**Symptom**: Same query returns different results

**Diagnosis**:
```python
# Run query multiple times
results1 = await engine.search_with_graphiti(query="test")
results2 = await engine.search_with_graphiti(query="test")

print(f"Results 1: {len(results1)}")
print(f"Results 2: {len(results2)}")
```

**Solutions**:

1. **Set random seed**:
```yaml
stages:
  entity_extraction:
    temperature: 0.0  # Deterministic
```

2. **Use caching**:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
async def cached_query(query: str):
    return await engine.search_with_graphiti(query)
```

3. **Disable reranking**:
```python
results = await engine.search_with_graphiti(
    query="test",
    rerank_method=RerankMethod.NONE
)
```

## Memory Issues

### Issue: Out of Memory

**Symptom**: `MemoryError` or system becomes unresponsive

**Diagnosis**:
```python
import psutil
import os

process = psutil.Process(os.getpid())
mem_info = process.memory_info()

print(f"Memory used: {mem_info.rss / 1024 / 1024:.2f} MB")
```

**Solutions**:

1. **Process in chunks**:
```python
# Instead of processing entire document
graph = await pipeline.extract_from_large_document(
    document=text,
    chunk_size=3000,  # Smaller chunks
    parallel_chunks=2  # Fewer parallel
)
```

2. **Clear cache**:
```python
# Clear LRU caches
pipeline.extract_cached.cache_clear()
```

3. **Reduce batch size**:
```yaml
neo4j_upload:
  batch_size: 50  # Reduce from 100
```

4. **Use streaming**:
```python
async def extract_streaming(text_file):
    with open(text_file, 'r') as f:
        while True:
            chunk = f.read(5000)
            if not chunk:
                break

            graph = await pipeline.extract_knowledge_graph(chunk)
            await pipeline.upload_to_neo4j(graph)

            # Free memory
            del graph
```

## Neo4j Issues

### Issue: Neo4j Out of Memory

**Symptom**: Neo4j crashes or becomes slow

**Diagnosis**:
```cypher
// Check database size
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Store file sizes")
YIELD attributes
RETURN attributes;
```

**Solutions**:

1. **Increase Neo4j memory**:
```conf
# neo4j/conf/neo4j.conf
dbms.memory.heap.initial_size=2g
dbms.memory.heap.max_size=4g
dbms.memory.pagecache.size=2g
```

2. **Clean old data**:
```cypher
// Delete old knowledge
MATCH (k:KnowledgeArtifact)
WHERE k.valid_at < datetime('2020-01-01')
DELETE k;
```

3. **Use batch operations**:
```python
# Upload in smaller batches
await pipeline.upload_to_neo4j(graph, batch_size=50)
```

### Issue: Slow Neo4j Queries

**Symptom**: Neo4j queries timeout

**Diagnosis**:
```cypher
// Profile query
PROFILE MATCH (n) RETURN n LIMIT 100;
```

**Solutions**:

1. **Create indices**:
```cypher
CREATE INDEX entity_name FOR (e:Entity) ON (e.name);
CREATE INDEX artifact_valid_at FOR (a:KnowledgeArtifact) ON (a.valid_at);
```

2. **Use query optimization**:
```cypher
// Good: Uses index
MATCH (e:Entity {name: "Apple"})
RETURN e;

// Bad: Full scan
MATCH (e:Entity)
WHERE e.name = "Apple"
RETURN e;
```

3. **Limit results**:
```cypher
MATCH (n)
RETURN n
LIMIT 100;
```

## Getting Help

If you can't resolve your issue:

1. **Check logs**:
```bash
# Application logs
tail -f openevolve.log

# Neo4j logs
tail -f neo4j/logs/neo4j.log
```

2. **Enable debug logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

3. **Report issue**:
   - GitHub: https://github.com/openevolve/frontend/issues
   - Include: Error message, logs, steps to reproduce
   - Provide: System info, Python version, configuration

## FAQ

**Q: How do I reset everything?**

A:
```bash
# Stop services
docker-compose down

# Clear Neo4j data
rm -rf neo4j/data

# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +

# Restart
docker-compose up -d
```

**Q: What's the minimum hardware requirement?**

A:
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB
- For production: 16 cores, 32 GB RAM, 500 GB SSD

**Q: Can I use without Neo4j?**

A: Yes, but temporal features will be limited:
```python
engine = TemporalKnowledgeEngine(
    enable_temporal=False  # Disable temporal features
)
```

**Q: How do I upgrade?**

A:
```bash
# Backup
docker-compose exec neo4j neo4j-admin dump

# Update
git pull origin main
pip install -r requirements.txt

# Restart
docker-compose restart
```
