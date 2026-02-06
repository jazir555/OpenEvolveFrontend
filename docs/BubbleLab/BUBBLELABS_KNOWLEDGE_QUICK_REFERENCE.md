# BubbleLabs Knowledge Integration - Quick Reference

## 🚀 Quick Start (5 minutes)

### Python (BubbleLab UI)
```bash
# 1. Install
pip install BubbleLab UI plotly networkx pandas boto3

# 2. Configure API keys in mcp_agent.secrets.yaml
# 3. Run
BubbleLab UI run bubblelabs_knowledge_integration.py
```

### TypeScript (React)
```bash
# 1. Components are in BubbleLab/apps/bubble-studio/src/components/knowledge/
# 2. Import and use
import { KnowledgeGraphViewer } from '@/components/knowledge/KnowledgeGraphViewer';
```

---

## 📁 Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `bubblelabs_knowledge_integration.py` | Main Python integration | 800+ |
| `KnowledgeGraphViewer.tsx` | React graph component | 400+ |
| `KnowledgeQueryInterface.tsx` | React query component | 500+ |
| `BUBBLELABS_KNOWLEDGE_INTEGRATION_GUIDE.md` | Complete guide | 400+ |
| `BUBBLELABS_KNOWLEDGE_INTEGRATION_SUMMARY.md` | Project summary | 400+ |

---

## 🔧 Common Tasks

### Query Knowledge Base
```python
# Python
ui = BubbleLabsKnowledgeUI()
ui.initialize_engine()
results = await ui.query_interface.query_bedrock(
    knowledge_base_id="YOUR_KB_ID",
    query_text="Your query here"
)
```

### Visualize Knowledge Graph
```python
# Python
visualizer = KnowledgeGraphVisualizer()
visualizer.build_graph_from_data(entities, relationships)
fig = visualizer.create_interactive_plot(layout='spring')
st.plotly_chart(fig)
```

```tsx
// TypeScript
<KnowledgeGraphViewer
  data={{ entities: [...], relationships: [...] }}
  onEntityClick={(entity) => console.log(entity)}
/>
```

### Extract Knowledge from Document
```python
# Python
workflow = KnowledgeExtractionWorkflow(engine)
results = await workflow.extract_from_document(
    document_path_or_url="https://arxiv.org/pdf/2301.07041"
)
entities = results['entities']
relationships = results['relationships']
```

### Unified Multi-Source Query
```python
# Python
results = await ui.query_interface.unified_query(
    query="MCTS optimization",
    sources=['bedrock', 'graphiti', 'local'],
    bedrock_kb_id="KB_ID",
    index_path="knowledge_index"
)
```

---

## 🎨 UI Components

### BubbleLab UI Tabs
1. **🔍 Query Knowledge** - Multi-source querying interface
2. **📊 Knowledge Graph** - Interactive visualization
3. **📄 Extract Knowledge** - Document processing
4. **📈 Statistics** - Metrics dashboard

### React Components
1. **KnowledgeGraphViewer** - Interactive graph with React Flow
2. **KnowledgeQueryInterface** - Multi-source query builder

---

## 🔑 Configuration Files

### `mcp_agent.secrets.yaml`
```yaml
aws:
  region_name: "us-east-1"
  access_key_id: "YOUR_KEY"
  secret_access_key: "YOUR_SECRET"

anthropic:
  api_key: "sk-ant-..."
```

### `knowledge_engine/indexer_config.yaml`
```yaml
llm:
  model_provider: "anthropic"
  temperature: 0.3
  max_tokens: 1000

paths:
  code_base_path: "code_base"
  output_dir: "indexes"
```

---

## 📊 Knowledge Sources

### Bedrock Knowledge Base
- Traditional vector search
- RAG-powered answers
- Integration with Graphiti

### Graphiti Temporal Knowledge
- Time-aware relationships
- Historical tracking
- Provenance metadata

### Local Code Indexes
- Fast keyword search
- Code structure analysis
- File-based retrieval

---

## 🎯 Key Classes

### Python
```python
KnowledgeGraphVisualizer    # Graph visualization
KnowledgeQueryInterface      # Multi-source queries
KnowledgeExtractionWorkflow  # Document processing
BubbleLabsKnowledgeUI       # Main UI component
KnowledgeEngine             # Engine facade
```

### TypeScript
```typescript
KnowledgeGraphViewer        # Graph visualization
KnowledgeQueryInterface     # Query interface
```

---

## 💡 Tips & Tricks

### Performance
- Filter large graphs by entity type
- Use pagination for results
- Enable caching in config
- Set `num_results` limit

### Visualization
- Use circular layout for small graphs
- Use force-directed for medium graphs
- Filter by confidence to reduce noise
- Search entities by name

### Extraction
- Start with small documents
- Review confidence scores
- Validate extracted entities
- Build custom extraction prompts

---

## 🐛 Troubleshooting

### Engine won't initialize
→ Check API keys in `mcp_agent.secrets.yaml`
→ Verify AWS credentials
→ Ensure LLM API keys are valid

### Graph visualization slow
→ Filter entities by type
→ Reduce node count
→ Use simpler layout
→ Check browser console

### Extraction returns empty
→ Check document format
→ Ensure LLM client initialized
→ Try smaller chunks
→ Increase max_tokens

### Query fails
→ Verify knowledge base ID
→ Check network connectivity
→ Review error logs
→ Try single source first

---

## 📈 Statistics Tracking

### Graph Metrics
- Total nodes/edges
- Graph density
- Connected components
- Average clustering

### Query Metrics
- Total queries
- Source distribution
- Average response time
- Success rate

### Extraction Metrics
- Documents processed
- Entities extracted
- Relationships found
- Average confidence

---

## 🔒 Security Best Practices

1. **Never commit** `mcp_agent.secrets.yaml`
2. Use environment variables for production
3. Rotate API keys regularly
4. Validate all user inputs
5. Limit file upload sizes
6. Sanitize knowledge graph data

---

## 📚 Documentation

- **Complete Guide:** `BUBBLELABS_KNOWLEDGE_INTEGRATION_GUIDE.md`
- **Project Summary:** `BUBBLELABS_KNOWLEDGE_INTEGRATION_SUMMARY.md`
- **This Reference:** `BUBBLELABS_KNOWLEDGE_QUICK_REFERENCE.md`

---

## 🚀 Next Steps

1. ✅ Configure API keys
2. ✅ Install dependencies
3. ✅ Run the application
4. ✅ Try example queries
5. ✅ Extract from a document
6. ✅ Visualize knowledge graph
7. ✅ Explore statistics

---

## 🎓 Learning Resources

- NetworkX: https://networkx.org/documentation/
- Plotly: https://plotly.com/python/
- React Flow: https://reactflow.dev/
- Bedrock: https://docs.aws.amazon.com/bedrock/

---

**Quick Reference v1.0** - Last Updated: 2026-01-03

