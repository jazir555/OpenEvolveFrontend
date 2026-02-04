# Knowledge Engine - Quick Start Scenarios

**One-page reference for common use cases**

---

## 🚀 5-Minute Quick Starts

### 1. Extract Knowledge from Documents
```python
from knowledge_engine.global_kg_orchestrator import create_global_orchestrator

orchestrator = create_global_orchestrator()

result = await orchestrator.extract_comprehensive(
    text="Your document text here...",
    extractors=['deepke', 'oneke'],  # Use multiple for better coverage
    enable_guardrails=True,          # Check for PII/safety
    enable_icr=True                  # Refine for accuracy
)

# Access results
entities = result.data['entities']
relations = result.data['relations']
```
**Best for**: Research papers, news articles, reports, contracts

---

### 2. Safe AI Chatbot
```python
# Plan conversation
plan = await orchestrator.optimize_dialog(
    context="Customer asking about return policy",
    goal="Resolve with satisfaction",
    enable_dts=True,
    enable_guardrails=True
)

# Generate safe response
response = await orchestrator.hub.structured_generate(
    prompt="Generate helpful response",
    output_schema={
        "response": {"type": "string"},
        "confidence": {"type": "number"}
    }
)

# Validate safety
safety = await orchestrator.hub.validate_safety(
    content=response['response'],
    validation_type='output',
    safety_level='strict'
)
```
**Best for**: Customer support, virtual assistants, help desks

---

### 3. Financial Risk Analysis
```python
# Extract entities from earnings call
result = await orchestrator.extract_comprehensive(
    text=earnings_transcript,
    entity_types=['ORG', 'MONEY', 'PERCENT', 'PRODUCT']
)

# Discover causal relationships
causal = await orchestrator.hub.discover_causal_structure(
    data=financial_metrics,
    variables=['revenue', 'expenses', 'market_sentiment', 'stock_price']
)

# Simulate scenarios
simulation = await orchestrator.hub.physics_simulate(
    system_description={'type': 'market_dynamics', ...},
    simulation_type='what_if'
)
```
**Best for**: Investment research, risk assessment, fraud detection

---

### 4. Drug Discovery Pipeline
```python
# Extract from scientific literature
result = await orchestrator.extract_comprehensive(
    text=research_paper,
    entity_types=['COMPOUND', 'PROTEIN', 'DISEASE', 'PATHWAY']
)

# Simulate molecular interactions
simulation = await orchestrator.hub.physics_simulate(
    system_description={
        'type': 'molecular_dynamics',
        'ligand': 'Candidate_Drug_42',
        'target': 'Protein_X'
    }
)

# Check compliance
safety = await orchestrator.hub.validate_safety(
    content=drug_proposal,
    validation_type='output',
    safety_level='strict'
)
```
**Best for**: Pharma research, chemical analysis, safety testing

---

### 5. Supply Chain Optimization
```python
# Mine patterns from logistics data
patterns = await orchestrator.hub.mine_patterns(
    graph_data=supply_chain_graph,
    algorithm='gspan',
    pattern_type='subgraph'
)

# Analyze risk topology
landscape = await orchestrator.analyze_knowledge_topology(
    embeddings=supplier_risk_profiles,
    n_clusters=5
)

# Detect supply chain drift
alert = await orchestrator.detect_concept_drift(
    embeddings_t1=baseline_patterns,
    embeddings_t2=current_patterns,
    drift_threshold=0.3
)
```
**Best for**: Logistics, inventory management, risk assessment

---

### 6. Legal Contract Analysis
```python
# Extract contract elements
result = await orchestrator.extract_comprehensive(
    text=contract_text,
    entity_types=['PARTY', 'OBLIGATION', 'PAYMENT_TERM', 'LIABILITY']
)

# Check compliance
compliance = await orchestrator.hub.validate_safety(
    content=contract_text,
    validation_type='compliance',
    safety_level='strict',
    policies=['GDPR', 'CCPA', 'SOX']
)

# Query similar cases
precedents = await orchestrator.hub.declarative_query(
    query="Find similar termination clauses",
    query_type='legal'
)
```
**Best for**: Contract review, compliance checking, due diligence

---

### 7. Research Literature Synthesis
```python
# Extract from papers
result = await orchestrator.extract_comprehensive(
    text=paper_text,
    extractors=['deepke', 'oneke', 'kggen']
)

# Analyze research landscape
landscape = await orchestrator.analyze_knowledge_topology(
    embeddings=paper_embeddings,
    labels=paper_titles,
    n_clusters=10
)

# Find research gaps
gaps = await orchestrator.hub.infer_knowledge(
    graph_data=current_kg,
    inference_type='transitive'
)
```
**Best for**: Literature reviews, research planning, gap analysis

---

### 8. Quality Control & Manufacturing
```python
# Mine defect patterns
patterns = await orchestrator.hub.mine_patterns(
    graph_data=quality_data,
    algorithm='sequential',
    pattern_type='defect_indicator'
)

# Physics simulation of wear
simulation = await orchestrator.hub.physics_simulate(
    system_description={
        'type': 'mechanical_degradation',
        'component': 'turbine_blade'
    }
)

# Detect new defect types
alert = await orchestrator.detect_concept_drift(
    embeddings_t1=normal_patterns,
    embeddings_t2=current_patterns
)
```
**Best for**: Manufacturing QC, predictive maintenance, defect detection

---

### 9. Cybersecurity Threat Intel
```python
# Extract IOCs from reports
result = await orchestrator.extract_comprehensive(
    text=threat_report,
    entity_types=['IOC', 'ATTACK_TECHNIQUE', 'THREAT_ACTOR']
)

# Mine attack patterns
patterns = await orchestrator.hub.mine_patterns(
    graph_data=attack_graph,
    algorithm='gspan'
)

# Detect emerging threats
emerging = await orchestrator.detect_concept_drift(
    embeddings_t1=historical_patterns,
    embeddings_t2=current_patterns,
    drift_threshold=0.25
)
```
**Best for**: SOC operations, threat hunting, incident response

---

### 10. Personalized Education
```python
# Extract curriculum concepts
result = await orchestrator.extract_comprehensive(
    text=course_material,
    entity_types=['CONCEPT', 'PREREQUISITE', 'LEARNING_OBJECTIVE']
)

# Analyze knowledge space
knowledge_map = await orchestrator.analyze_knowledge_topology(
    embeddings=concept_vectors,
    n_clusters=8
)

# Optimize learning path
path = await orchestrator.optimize_dialog(
    context="Student struggling with calculus",
    goal="Achieve mastery",
    enable_dts=True
)

# Generate personalized content
content = await orchestrator.hub.structured_generate(
    prompt="Explain derivatives to visual learner",
    output_schema={
        "explanation": "string",
        "examples": "array",
        "visualizations": "array"
    }
)
```
**Best for**: Adaptive learning, tutoring systems, curriculum design

---

## 🎯 Scenario Selector

| Your Goal | Use This | Key Integrations |
|-----------|----------|------------------|
| Extract entities from text | `extract_comprehensive()` | DeepKE, OneKE, KG-Gen |
| Build safe chatbot | `optimize_dialog()` + `validate_safety()` | DTS, Guardrails, Outlines |
| Analyze financial data | `discover_causal_structure()` + `physics_simulate()` | Causal-Learn, Neuromancer |
| Drug discovery | `extract_comprehensive()` + `physics_simulate()` | DeepKE, GlobalChem, Neuromancer |
| Check legal compliance | `validate_safety()` with compliance policies | Guardrails, OneKE |
| Mine patterns | `mine_patterns()` | PAMI |
| Find research gaps | `infer_knowledge()` + `analyze_knowledge_topology()` | AI-KG, Lagrange Mapper |
| Optimize supply chain | `detect_concept_drift()` + `mine_patterns()` | Lagrange Mapper, PAMI |
| Personalize content | `optimize_dialog()` + `structured_generate()` | DTS, Outlines |
| Ensure AI safety | `validate_safety()` + `refine_iteratively()` | Guardrails, ICR |

---

## 💡 Pro Tips

1. **Always use `enable_guardrails=True`** for production systems handling user data
2. **Use multiple extractors** (`['deepke', 'oneke', 'kggen']`) for better coverage
3. **Enable ICR** (`enable_icr=True`) when accuracy is critical
4. **Check drift regularly** to detect concept/behavior changes
5. **Use structured outputs** (`structured_generate()`) for reliable downstream processing

---

## 📚 Full Documentation

- **Detailed Scenarios**: See `SCENARIOS_AND_USE_CASES.md`
- **API Reference**: See `API_REFERENCE.md`
- **Architecture Guide**: See `COMPREHENSIVE_SYSTEM_GUIDE.md`
