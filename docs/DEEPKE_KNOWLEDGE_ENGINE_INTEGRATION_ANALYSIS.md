# DeepKE Knowledge Engine Integration Analysis

**Analysis Date:** 2025-12-31
**Analyst:** Claude Code
**Component:** OpenEvolve Knowledge Engine (Stage 6)
**Candidate:** DeepKE (Deep Learning Based Knowledge Extraction Toolkit)

---

## Executive Summary

### Recommendation: **PARTIAL INTEGRATION** (Specific Components Only)

**Verdict:** DeepKE provides valuable structured knowledge extraction capabilities that can **enhance** (but not replace) the existing Knowledge Engine implementation. Integration is recommended for **specific knowledge extraction tasks**, particularly for extracting structured artifacts from workflow execution data.

### Key Findings

| Aspect | Assessment | Details |
|--------|------------|---------|
| **Entity/Relation Extraction** | ✅ **Excellent** | DeepKE's NER/RE capabilities are mature and production-ready |
| **Event Extraction** | ✅ **Excellent** | Well-implemented EE can extract workflow events |
| **MCP Integration** | ✅ **Immediate** | DeepKE-MCP-Tools provides plug-and-play MCP server |
| **LLM Integration** | ⚠️ **Partial** | OneKE model powerful but requires 13B parameter deployment |
| **Pattern Mining** | ❌ **None** | No ML clustering for solution pattern discovery |
| **Knowledge Graph** | ❌ **None** | No graph visualization or relationship tracking |
| **Learning from Execution** | ❌ **None** | No feedback loop or learning pipeline |
| **Workflow-Specific Artifacts** | ❌ **None** | No support for Decomposition Workflow artifact types |

### Effort Estimate

- **Minimal Integration** (MCP-Tools only): 1-2 days
- **Structured Extraction Integration** (NER/RE/EE for artifacts): 5-7 days
- **Custom Schema Training** (workflow-specific artifacts): 15-20 days
- **Full Knowledge Engine Replacement**: **NOT RECOMMENDED** (insufficient coverage)

**Recommended Approach:** Hybrid integration using DeepKE for extraction + existing stack for learning/graph/search.

---

## 1. DeepKE Capabilities Summary

### 1.1 Core Knowledge Extraction Capabilities

DeepKE provides **four main extraction capabilities**, all mature and production-ready:

#### **A. Named Entity Recognition (NER)**
- **Purpose:** Extract and classify named entities (people, organizations, locations, etc.)
- **Models:**
  - Standard: BiLSTM-CRF, BERT-based
  - Advanced: W2NER (AAAI'22), LightNER (COLING'22)
  - Low-resource: Few-shot learning with prompts
  - Multimodal: Image + text entity extraction
- **Strengths:** High accuracy, multilingual support (Chinese/English)
- **Relevance to Knowledge Engine:** Extract entities from workflow solutions, critiques, verification reports

#### **B. Relation Extraction (RE)**
- **Purpose:** Extract semantic relationships between entities
- **Models:**
  - Standard: CNN-based, BiLSTM-based
  - Advanced: PRGC (ACL'21), ASP (EMNLP'22)
  - Low-resource: KnowPrompt (WWW'22)
  - Document-level: Cross-sentence relation extraction
- **Strengths:** Handles complex relation types, document-level context
- **Relevance to Knowledge Engine:** Extract relationships between solution components, problem-solution mappings

#### **C. Attribute Extraction (AE)**
- **Purpose:** Extract attributes of entities
- **Models:**
  - Standard: PCNN, BiLSTM, Transformer-based
  - Supports various attribute types
- **Strengths:** Fine-grained attribute extraction
- **Relevance to Knowledge Engine:** Extract solution metadata, configuration attributes, performance metrics

#### **D. Event Extraction (EE)**
- **Purpose:** Extract events and arguments (trigger words, roles)
- **Models:**
  - Standard: BERT-CRF, DEGREE (document-level)
  - Supports Chinese and English
- **Strengths:** Complex event structures, multi-argument extraction
- **Relevance to Knowledge Engine:** Extract workflow execution events, refinement loops, failure events

### 1.2 DeepKE-LLM Capabilities

#### **OneKE Model**
- **Size:** 13B parameters (Chinese-Alpaca-2 based)
- **Capability:** Schema-based information extraction (bilingual Chinese/English)
- **Strengths:**
  - Instruction-following for custom schemas
  - High-quality extraction on complex documents
  - Bilingual support
- **Limitations:**
  - Requires significant GPU resources (13B params)
  - Deployment complexity
  - Overkill for simple extraction tasks

#### **Training Datasets**
- **InstructIE:** 300k+ instruction dataset (bilingual)
- **IEPile:** 0.32B tokens, 2M+ instruction examples
- **KnowLM-IE:** Specialized IE instruction dataset

**Relevance:** Can fine-tune on Decomposition Workflow artifact types for custom extraction.

### 1.3 DeepKE-MCP-Tools

#### **MCP Server Implementation**
- **Status:** Production-ready, deployed at [ModelScope](https://modelscope.cn/mcp/servers/OpenKG/deepke-mcp-tools)
- **Tools Provided:**
  - `deepke_ner()`: Named entity recognition
  - `deepke_re()`: Relation extraction
  - `deepke_ae()`: Attribute extraction
  - `deepke_ee()`: Event extraction
- **Integration:** Can be called by any MCP client (including OpenEvolve's existing MCP infrastructure)
- **Effort:** Plug-and-play (1-2 hours to configure)

**Critical Advantage:** OpenEvolve already has MCP integration via `ace_mcp_tools.py` and other MCP clients. DeepKE-MCP-Tools can be integrated immediately without code changes.

---

## 2. Gap Analysis: DeepKE vs Knowledge Engine Requirements

### 2.1 Requirement Mapping Matrix

| Knowledge Engine Requirement | DeepKE Capability | Gap | Solution |
|------------------------------|-------------------|-----|----------|
| **KnowledgeArtifact Schema** | ❌ None | High | Build adapter layer to map DeepKE output to KnowledgeArtifact |
| **Solution Pattern Mining** | ❌ No ML clustering | High | Use RAGbits + custom clustering, not DeepKE |
| **Problem-Solution Mapping** | ⚠️ RE can extract | Medium | DeepKE RE for extraction + custom logic for mappings |
| **Critique Insights Extraction** | ⚠EE can extract events | Medium | DeepKE EE for critique events + custom aggregation |
| **Team Performance Tracking** | ❌ None | High | Must build custom tracker |
| **Gauntlet Effectiveness** | ❌ None | High | Must build custom analyzer |
| **Learning from Execution** | ❌ None | High | Use ACE framework (already integrated) |
| **Knowledge Graph Visualization** | ❌ None | High | Use RAGbits + NetworkX + Plotly |
| **Vector Embeddings** | ❌ None | Medium | Use RAGbits (already planned) |
| **Semantic Search** | ❌ None | Medium | Use RAGbits (already planned) |
| **Entity Knowledge Graph** | ⚠️ Extracts entities | Low | Map DeepKE entities to existing graph structure |
| **Document Ingestion** | ❌ None | None | Existing KE handles this |
| **Code Indexing** | ❌ None | None | Existing KE handles this |
| **External KB Integration** | ❌ None | None | Existing KE handles this |
| **MCP Integration** | ✅ Full | None | DeepKE-MCP-Tools plug-and-play |
| **Structured Extraction** | ✅ Full (NER/RE/AE/EE) | None | DeepKE excels here |

### 2.2 What DeepKE Fills

#### **Filled Gap #1: Structured Knowledge Extraction**
**Current State:** Existing Knowledge Engine has basic LLM-based code analysis but no structured entity/relation extraction.

**DeepKE Solution:**
- Use DeepKE NER to extract entities from workflow solutions:
  - Algorithm names
  - Data structures
  - Libraries/frameworks
  - Team members
  - Gauntlet types
- Use DeepKE RE to extract relationships:
  - Solution dependencies
  - Component relationships
  - Team assignments
  - Gauntlet application order
- Use DeepKE AE to extract attributes:
  - Solution quality scores
  - Resource utilization
  - Configuration parameters
- Use DeepKE EE to extract events:
  - Refinement loops
  - Failure events
  - Verification attempts

**Impact:** High-quality structured extraction that enhances knowledge artifact creation.

#### **Filled Gap #2: MCP-Ready Integration**
**Current State:** Knowledge Engine needs extraction capabilities that work with existing MCP infrastructure.

**DeepKE Solution:**
- DeepKE-MCP-Tools provides 4 MCP tools ready for integration
- Can be called via existing MCP clients in OpenEvolve
- No need to deploy DeepKE models separately (MCP server handles it)

**Impact:** Immediate integration with minimal effort.

#### **Filled Gap #3: Bilingual Support**
**Current State:** Knowledge Engine primarily English-focused.

**DeepKE Solution:**
- All DeepKE models support Chinese and English
- InstructIE and IEPile datasets are bilingual
- OneKE model natively bilingual

**Impact:** Can extract knowledge from multilingual workflows.

### 2.3 What DeepKE Does NOT Fill

#### **Missing #1: Solution Pattern Mining (ML Clustering)**
**Requirement:** Cluster successful solutions to discover reusable patterns.

**DeepKE Limitation:** DeepKE extracts entities/relations but does not perform ML-based clustering or pattern mining.

**Alternative:** Use scikit-learn/HDBSCAN with sentence-transformers embeddings (as recommended in requirements analysis).

#### **Missing #2: Learning from Execution Feedback**
**Requirement:** Learn from workflow execution to improve future performance.

**DeepKE Limitation:** DeepKE is an extraction toolkit, not a learning framework.

**Alternative:** Use ACE framework (already in `agentic-context-engine/`).

#### **Missing #3: Knowledge Graph Visualization**
**Requirement:** Visualize artifact relationships and knowledge graph.

**DeepKE Limitation:** DeepKE extracts entities/relations but provides no graph visualization.

**Alternative:** Use NetworkX + Plotly (as recommended in requirements analysis).

#### **Missing #4: Workflow-Specific Artifact Types**
**Requirement:** Support KnowledgeArtifact schema types (solution_pattern, problem_solution_mapping, critique_insight, team_performance, gauntlet_effectiveness).

**DeepKE Limitation:** DeepKE extracts generic entities/relations/events, not specialized artifact types.

**Alternative:** Build adapter layer to map DeepKE output to KnowledgeArtifact schema.

#### **Missing #5: Vector Embeddings & Semantic Search**
**Requirement:** Convert artifacts to vector embeddings, enable semantic search.

**DeepKE Limitation:** DeepKE does not provide vector embeddings or search.

**Alternative:** Use RAGbits (already planned for integration).

---

## 3. Integration Assessment

### 3.1 Technical Feasibility

#### **Language & Framework Compatibility**
| Aspect | DeepKE | OpenEvolve | Compatible? |
|--------|--------|------------|-------------|
| **Language** | Python 3.8+ | Python 3.8+ | ✅ Yes |
| **Deep Learning** | PyTorch | PyTorch (in some modules) | ✅ Yes |
| **Dependencies** | transformers, torch, hydra | transformers, torch, yaml | ✅ Mostly compatible |
| **MCP Protocol** | MCP server | MCP clients (ace_mcp_tools.py) | ✅ Yes |

**Verdict:** High technical compatibility. Minor dependency version adjustments may be needed.

#### **Integration Approaches**

##### **Approach 1: MCP Integration (Recommended for Quick Start)**
**How it works:**
1. Configure DeepKE-MCP-Tools server (local or remote)
2. Add DeepKE MCP server to OpenEvolve's MCP client configuration
3. Call `deepke_ner()`, `deepke_re()`, `deepke_ae()`, `deepke_ee()` from Knowledge Engine
4. Parse results and map to KnowledgeArtifact schema

**Effort:** 1-2 days
**Pros:**
- Zero code changes to DeepKE
- Isolated deployment (doesn't affect OpenEvolve environment)
- Easy to enable/disable

**Cons:**
- Network latency if using remote server
- Limited to pre-trained models (no custom fine-tuning)

**Code Example:**
```python
# In knowledge_engine/engine.py

async def extract_artifacts_with_deepke(
    self,
    workflow_execution: WorkflowExecution
) -> List[KnowledgeArtifact]:
    """Extract knowledge artifacts using DeepKE MCP tools."""

    artifacts = []

    # Extract entities from solutions using DeepKE NER
    for solution in workflow_execution.verified_solutions:
        entities = await self.mcp_client.call_tool("deepke_ner", {
            "text": solution.code,
            "schema": ["algorithm", "data_structure", "library"]
        })

        # Create KnowledgeArtifact from extracted entities
        artifact = KnowledgeArtifact(
            id=str(uuid.uuid4()),
            artifact_type="solution_pattern",
            content={"entities": entities},
            source_workflow_id=workflow_execution.id,
            extraction_timestamp=time.time(),
            effectiveness_score=solution.quality_score
        )
        artifacts.append(artifact)

    # Extract relations using DeepKE RE
    relations = await self.mcp_client.call_tool("deepke_re", {
        "text": workflow_execution.full_transcript,
        "schema": ["depends_on", "improves", "replaces"]
    })

    # Extract events using DeepKE EE
    events = await self.mcp_client.call_tool("deepke_ee", {
        "text": workflow_execution.refinement_history,
        "schema": ["refinement_loop", "failure", "verification"]
    })

    return artifacts
```

##### **Approach 2: Direct Import (Recommended for Custom Training)**
**How it works:**
1. Install DeepKE in OpenEvolve environment
2. Import DeepKE modules directly in Knowledge Engine
3. Fine-tune DeepKE models on workflow-specific data
4. Use trained models for extraction

**Effort:** 10-15 days (including fine-tuning)
**Pros:**
- Full control over models
- Can train on workflow-specific artifact types
- No network latency

**Cons:**
- Increases OpenEvolve dependency footprint
- Requires GPU resources for training/inference
- Model maintenance burden

**Code Example:**
```python
# In knowledge_engine/deepke_adapter.py

from deepke.name_entity_re.standard import InferBert
from deepke.relation_extraction.standard import load_model as load_re_model

class DeepKEExtractor:
    """Adapter for DeepKE extraction models."""

    def __init__(self, ner_model_path, re_model_path):
        self.ner_model = InferBert(ner_model_path)
        self.re_model = load_re_model(re_model_path)

    async def extract_solution_entities(
        self,
        solution_code: str
    ) -> List[Entity]:
        """Extract entities from solution code."""
        entities = self.ner_model.predict(solution_code)
        return [
            Entity(
                text=e["text"],
                type=e["type"],
                confidence=e["score"]
            )
            for e in entities
        ]

    async def extract_component_relations(
        self,
        workflow_text: str
    ) -> List[Relation]:
        """Extract relations between workflow components."""
        relations = self.re_model.predict(workflow_text)
        return [
            Relation(
                head=r["head"],
                tail=r["tail"],
                type=r["relation"],
                confidence=r["score"]
            )
            for r in relations
        ]
```

##### **Approach 3: Hybrid (Recommended for Production)**
**How it works:**
1. Use DeepKE-MCP-Tools for quick extraction (baseline)
2. Build custom adapter layer for KnowledgeArtifact mapping
3. Gradually fine-tune DeepKE models on workflow data
4. Replace MCP calls with direct imports for performance

**Effort:** 5-7 days for adapter + 10-15 days for fine-tuning
**Pros:**
- Best of both worlds
- Incremental integration
- Easy fallback to MCP

**Cons:**
- More complex architecture
- Requires maintaining both integration paths

### 3.2 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Dependency Conflicts** | Medium | Medium | Use MCP integration (isolated environment) |
| **GPU Resource Requirements** | High | Medium | Start with MCP server (cloud-hosted), evaluate local deployment later |
| **Model Accuracy on Workflow Data** | Medium | High | Fine-tune on workflow-specific examples, validate with test set |
| **Maintenance Overhead** | Low | Medium | MCP integration reduces maintenance burden |
| **Integration Complexity** | Low | Low | DeepKE has clean APIs, MCP protocol is standard |

**Overall Risk:** **Medium** (manageable with MCP approach)

---

## 4. Comparison with Existing Knowledge Engine

### 4.1 Feature Comparison

| Feature | Existing KE | DeepKE | Combined |
|---------|-------------|--------|----------|
| **Document Ingestion** | ✅ PDF, Office, text, URLs | ❌ | ✅ Use existing |
| **Code Indexing** | ✅ LLM-powered repository analysis | ❌ | ✅ Use existing |
| **Entity Knowledge Graph** | ⚠️ Basic in-memory graph | ✅ NER extraction | ✅ Enhanced with DeepKE entities |
| **External KB Integration** | ✅ Bedrock, EKS, Elasticsearch | ❌ | ✅ Use existing |
| **DeepCode Workflows** | ✅ Multi-agent research pipeline | ❌ | ✅ Use existing |
| **LLM Client Support** | ✅ Anthropic, OpenAI, Google | ⚠️ General LLM support | ✅ Use existing |
| **Named Entity Recognition** | ❌ | ✅ Multiple models | ✅ **Add DeepKE** |
| **Relation Extraction** | ❌ | ✅ Multiple models | ✅ **Add DeepKE** |
| **Attribute Extraction** | ❌ | ✅ Multiple models | ✅ **Add DeepKE** |
| **Event Extraction** | ❌ | ✅ Multiple models | ✅ **Add DeepKE** |
| **MCP Server** | ❌ | ✅ 4 tools | ✅ **Add DeepKE** |
| **Vector Embeddings** | ❌ | ❌ | ⚠️ Need RAGbits |
| **Semantic Search** | ❌ | ❌ | ⚠️ Need RAGbits |
| **Learning from Execution** | ❌ | ❌ | ⚠️ Need ACE |
| **Solution Pattern Mining** | ❌ | ❌ | ⚠️ Need custom implementation |
| **Knowledge Graph Viz** | ⚠️ Basic | ❌ | ⚠️ Need enhancement |
| **KnowledgeArtifact Schema** | ❌ | ❌ | ❌ Must implement |

**Key Insight:** DeepKE **complements** the existing Knowledge Engine rather than replacing it. The existing KE provides infrastructure (document loading, indexing, external KBs), while DeepKE provides extraction capabilities.

### 4.2 Complementarity Analysis

#### **Where DeepKE Shines**
1. **Structured Extraction:** DeepKE's NER/RE/AE/EE are state-of-the-art for knowledge extraction
2. **MCP Integration:** Ready-to-use MCP server aligns with OpenEvolve's architecture
3. **Bilingual Support:** fills gap for Chinese/English workflows
4. **Pre-trained Models:** Can use immediately without training

#### **Where Existing KE Shines**
1. **Document Processing:** PDF/Office conversion, URL downloading
2. **Code Analysis:** Repository indexing with LLM
3. **External Integrations:** Bedrock, EKS, Elasticsearch
4. **Workflow Integration:** DeepCode pipelines already integrated

#### **What's Still Missing**
1. **KnowledgeArtifact Schema:** Must be implemented regardless of DeepKE
2. **Pattern Mining:** ML clustering for solution patterns (RAGbits + scikit-learn)
3. **Learning Framework:** ACE for learning from execution
4. **Knowledge Graph Visualization:** NetworkX + Plotly
5. **UI Interface:** Knowledge base browser and management

### 4.3 Redundancy Assessment

**Is there overlap between DeepKE and existing KE?**

| Capability | Existing KE | DeepKE | Redundant? |
|------------|-------------|--------|------------|
| Document Ingestion | ✅ | ❌ | No |
| Code Indexing | ✅ | ❌ | No |
| Entity Extraction | ⚠️ Basic LLM | ✅ Specialized NER | Partial (DeepKE better) |
| Relation Extraction | ❌ | ✅ Specialized RE | No |
| Attribute Extraction | ❌ | ✅ Specialized AE | No |
| Event Extraction | ❌ | ✅ Specialized EE | No |
| External KB Integration | ✅ | ❌ | No |
| LLM Support | ✅ Multi-provider | ⚠️ General | Partial (existing better) |

**Verdict:** **Minimal redundancy**. DeepKE adds new capabilities rather than duplicating existing ones.

---

## 5. Recommendation

### 5.1 Decision: **PARTIAL INTEGRATION RECOMMENDED**

**Rationale:**
1. DeepKE fills specific extraction gaps (NER/RE/AE/EE) that enhance Knowledge Engine
2. DeepKE-MCP-Tools enables immediate integration with minimal effort
3. DeepKE does NOT replace the need for ACE, RAGbits, or custom components
4. Hybrid approach leverages strengths of both systems

### 5.2 Integration Plan

#### **Phase 1: Quick Win (Week 1-2) - MCP Integration**

**Objective:** Integrate DeepKE-MCP-Tools for basic extraction.

**Tasks:**
1. Configure DeepKE-MCP-Tools server (local or ModelScope)
2. Add MCP client configuration to `mcp_agent.secrets.yaml`
3. Implement `DeepKEExtractor` class in `knowledge_engine/deepke_adapter.py`
4. Create mapping from DeepKE output to KnowledgeArtifact schema
5. Test extraction on sample workflow execution data

**Deliverables:**
- `knowledge_engine/deepke_adapter.py` - DeepKE adapter class
- `knowledge_engine/engine.py` - Updated with DeepKE extraction methods
- Integration tests validating extraction quality

**Effort:** 5-7 days

**Success Criteria:**
- ✅ DeepKE NER extracts entities from 5 sample solutions
- ✅ DeepKE RE extracts relations from 5 sample workflows
- ✅ Extracted data maps to KnowledgeArtifact schema
- ✅ Extraction quality > 70% F1 score on validation set

#### **Phase 2: Enhanced Extraction (Week 3-5) - Custom Schema**

**Objective:** Fine-tune DeepKE models on workflow-specific artifact types.

**Tasks:**
1. Prepare training dataset from historical workflow executions
2. Define custom schema for workflow artifacts (solution_pattern, critique_insight, etc.)
3. Fine-tune DeepKE models (start with NER, then RE/EE)
4. Validate fine-tuned models on test set
5. Replace MCP calls with fine-tuned models (optional)

**Deliverables:**
- Training dataset (InstructIE format) for workflow artifacts
- Fine-tuned DeepKE models for NER (solution components)
- Fine-tuned DeepKE models for RE (workflow relations)
- Fine-tuned DeepKE models for EE (workflow events)
- Performance evaluation report

**Effort:** 15-20 days

**Success Criteria:**
- ✅ Fine-tuned NER model achieves > 80% F1 on solution component extraction
- ✅ Fine-tuned RE model achieves > 75% F1 on workflow relation extraction
- ✅ Fine-tuned EE model achieves > 70% F1 on event extraction
- ✅ Models generalize to unseen workflow types

#### **Phase 3: Production Integration (Week 6-8) - Full Pipeline**

**Objective:** Integrate DeepKE extraction into complete Knowledge Engine pipeline.

**Tasks:**
1. Implement `WorkflowKnowledgeExtractor` with DeepKE backend
2. Integrate with ACE learning pipeline
3. Add vector embeddings (RAGbits) for extracted artifacts
4. Implement knowledge graph visualization with DeepKE entities
5. Build UI for browsing extracted knowledge artifacts
6. End-to-end testing with real workflow executions

**Deliverables:**
- Complete Knowledge Engine pipeline with DeepKE extraction
- Integration with ACE for learning from execution
- Knowledge base interface for artifact browsing
- Performance benchmarks (extraction time, quality)

**Effort:** 15-20 days

**Success Criteria:**
- ✅ End-to-end extraction pipeline operational
- ✅ Knowledge artifacts extracted from 100% of workflow executions
- ✅ Extraction quality > 75% F1 on all artifact types
- ✅ Learning feedback loop improves workflow success rate by > 10%

### 5.3 Alternative Approaches

#### **Alternative 1: Defer DeepKE Integration**
**When to choose:**
- Limited development resources
- Immediate focus on other Stage 6 components (ACE, RAGbits)
- Existing LLM-based extraction deemed sufficient

**Pros:**
- Focus resources on unique components (pattern mining, learning)
- Avoid dependency complexity

**Cons:**
- Lower extraction quality
- Manual effort required for artifact creation

#### **Alternative 2: Replace Existing KE with DeepKE**
**When to choose:** ❌ **NOT RECOMMENDED**

**Why not:**
- DeepKE lacks critical KE features (document loading, code indexing, external KBs)
- DeepKE does not provide learning framework (ACE)
- DeepKE does not provide vector search (RAGbits)
- Would lose existing functionality

#### **Alternative 3: Use Only DeepKE-MCP-Tools (No Fine-tuning)**
**When to choose:**
- Quick proof-of-concept
- Limited GPU resources
- Satisfactory extraction quality with pre-trained models

**Pros:**
- Minimal integration effort
- No model maintenance

**Cons:**
- Lower accuracy on workflow-specific artifacts
- Limited customization

### 5.4 Recommended Technology Stack

```python
# knowledge_engine/deepke_adapter.py

from typing import List, Dict, Any
import asyncio
from dataclasses import dataclass

# DeepKE imports (if using direct import)
from deepke.name_entity_re.standard import InferBert
from deepke.relation_extraction.standard import load_model as load_re_model
from deepke.event_extraction.standard import load_model as load_ee_model

# Existing imports
from .core import KnowledgeArtifact, WorkflowExecution
from llm_utils import initialize_llm_client

class DeepKEExtractor:
    """
    Adapter for DeepKE knowledge extraction capabilities.

    Supports two modes:
    1. MCP mode: Call DeepKE-MCP-Tools server
    2. Direct mode: Import DeepKE models directly
    """

    def __init__(
        self,
        mode: str = "mcp",  # "mcp" or "direct"
        mcp_client=None,
        ner_model_path: str = None,
        re_model_path: str = None,
        ee_model_path: str = None
    ):
        self.mode = mode
        self.mcp_client = mcp_client

        if mode == "direct":
            self.ner_model = InferBert(ner_model_path)
            self.re_model = load_re_model(re_model_path)
            self.ee_model = load_ee_model(ee_model_path)

    async def extract_solution_pattern(
        self,
        solution_code: str,
        solution_metadata: Dict[str, Any]
    ) -> KnowledgeArtifact:
        """
        Extract solution pattern artifact using DeepKE NER + RE.

        Extracts:
        - Algorithms used
        - Data structures
        - Libraries/frameworks
        - Dependencies between components
        """
        if self.mode == "mcp":
            entities = await self.mcp_client.call_tool("deepke_ner", {
                "text": solution_code,
                "schema": ["algorithm", "data_structure", "library", "technique"]
            })

            relations = await self.mcp_client.call_tool("deepke_re", {
                "text": solution_code,
                "schema": ["uses", "depends_on", "implements", "optimizes"]
            })
        else:
            entities = self.ner_model.predict(solution_code)
            relations = self.re_model.predict(solution_code)

        return KnowledgeArtifact(
            artifact_type="solution_pattern",
            content={
                "entities": entities,
                "relations": relations,
                "metadata": solution_metadata
            }
        )

    async def extract_critique_insight(
        self,
        critique_reports: List[str],
        verification_reports: List[str]
    ) -> List[KnowledgeArtifact]:
        """
        Extract critique insights using DeepKE EE.

        Extracts:
        - Common issues identified
        - Improvement patterns
        - Failure modes
        """
        insights = []

        for critique in critique_reports:
            if self.mode == "mcp":
                events = await self.mcp_client.call_tool("deepke_ee", {
                    "text": critique,
                    "schema": ["issue_identified", "improvement_suggested", "flaw_type"]
                })
            else:
                events = self.ee_model.predict(critique)

            insight = KnowledgeArtifact(
                artifact_type="critique_insight",
                content={"events": events}
            )
            insights.append(insight)

        return insights

    async def extract_workflow_events(
        self,
        workflow_execution: WorkflowExecution
    ) -> List[KnowledgeArtifact]:
        """
        Extract workflow events using DeepKE EE.

        Extracts:
        - Refinement loops
        - Failures
        - Verification attempts
        - Resource usage spikes
        """
        full_transcript = workflow_execution.full_transcript

        if self.mode == "mcp":
            events = await self.mcp_client.call_tool("deepke_ee", {
                "text": full_transcript,
                "schema": ["refinement_loop", "failure", "verification", "resource_usage"]
            })
        else:
            events = self.ee_model.predict(full_transcript)

        return [
            KnowledgeArtifact(
                artifact_type="workflow_event",
                content={"event": event}
            )
            for event in events
        ]
```

### 5.5 Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  DECOMPOSITION WORKFLOW STAGE 6                  │
│               (Knowledge Extraction & Learning)                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   DeepKE         │  │   ACE Framework  │  │   RAGbits        │
│   Extraction     │  │   (Learning)     │  │   (Vector Store) │
│                  │  │                  │  │                  │
│ • NER            │  │ • Reflector      │  │ • Vector Embed    │
│ • RE             │  │ • SkillManager   │  │ • Semantic Search│
│ • AE             │  │ • Async Pipeline │  │ • Hybrid Search  │
│ • EE             │  │ • Deduplication  │  │ • Retrieval      │
│ • MCP-Tools      │  │                  │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Knowledge Engine       │
                    │  Orchestrator           │
                    │  (NEW LAYER)            │
                    │                         │
                    │ • DeepKEExtractor       │
                    │ • ArtifactMapper        │
                    │ • PatternMiner          │
                    │ • GraphBuilder          │
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Knowledge Base   │  │  Knowledge Graph │  │  Learning Loop   │
│                  │  │  Visualization   │  │                  │
│ • Artifacts      │  │  • NetworkX      │  │ • Decomposer     │
│ • Patterns       │  │  • Plotly        │  │ • Gauntlets      │
│ • Mappings       │  │  • D3.js         │  │ • Optimizer      │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

**Key Points:**
1. **DeepKE** fills extraction role (NER/RE/AE/EE)
2. **ACE** fills learning role (feedback, improvement)
3. **RAGbits** fills search role (vector embeddings, semantic retrieval)
4. **Knowledge Engine Orchestrator** (NEW) coordinates all components
5. **Outputs** feed back into Decomposition Workflow (stages 0-5)

---

## 6. Conclusion

### 6.1 Summary

**DeepKE is a valuable addition** to the OpenEvolve Knowledge Engine for **structured knowledge extraction**, but it is **not a complete solution** for Stage 6 requirements.

**Strengths:**
- ✅ Excellent NER/RE/AE/EE capabilities
- ✅ MCP integration ready
- ✅ Bilingual support
- ✅ Pre-trained models available

**Limitations:**
- ❌ No ML-based pattern mining
- ❌ No learning from execution
- ❌ No knowledge graph visualization
- ❌ No workflow-specific artifact types

**Recommendation:** **Integrate DeepKE for extraction tasks** while using ACE for learning and RAGbits for vector search.

### 6.2 Next Steps

1. **Review this analysis** with stakeholders to confirm approach
2. **Prototype MCP integration** (1-2 days) to validate extraction quality
3. **Evaluate extraction results** on sample workflow data
4. **Decide on fine-tuning** based on prototype results
5. **Plan integration** with ACE and RAGbits

### 6.3 Estimated Timeline

- **MCP Integration (Phase 1):** 1 week
- **Custom Fine-tuning (Phase 2):** 3-4 weeks
- **Full Pipeline (Phase 3):** 3-4 weeks
- **Total:** 7-9 weeks for complete integration

### 6.4 Final Verdict

**PARTIAL INTEGRATION RECOMMENDED**

Use DeepKE for what it does best (structured extraction) while complementing it with ACE (learning), RAGbits (search), and custom components (pattern mining, graph visualization).

**Do not attempt to replace the entire Knowledge Engine with DeepKE** - it lacks critical capabilities required by the Decomposition Workflow.

---

**Report Version:** 1.0
**Last Updated:** 2025-12-31
**Status:** Draft for Review
