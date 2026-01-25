# Phase 3: Knowledge Engine Integration Tasks

**Integration:** ai-knowledge-graph + DeepKE into OpenEvolve Knowledge Engine
**Estimated Duration:** 3 weeks (15 working days)
**Team Size:** 1 developer (part-time)
**Start Date:** TBD
**Target Date:** TBD + 3 weeks

---

## Task Overview

This document provides detailed implementation tasks for integrating both **ai-knowledge-graph** and **DeepKE** into OpenEvolve's Knowledge Engine (Stage 6 of the Decomposition Workflow).

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  OPENEVOLVE KNOWLEDGE ENGINE                    │
│                       (Stage 6)                                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   DeepKE         │  │  ai-knowledge-   │  │  ACE Framework   │
│   MCP-Tools      │  │  graph           │  │  (Existing)      │
│                  │  │                  │  │                  │
│ • deepke_ner()   │  │ • SPO Extract    │  │ • Reflector      │
│ • deepke_re()    │  │ • Entity Std.    │  │ • SkillManager   │
│ • deepke_ae()    │  │ • Rel. Inference │  │ • Async Pipeline │
│ • deepke_ee()    │  │ • PyVis Viz      │  │ • Deduplication  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Knowledge Engine       │
                    │  Orchestrator (NEW)     │
                    │                         │
                    │ • DeepKEExtractor       │
                    │ • AIKGProcessor         │
                    │ • ArtifactMapper        │
                    │ • WorkflowExtractor     │
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Knowledge Base  │  │  Knowledge Graph │  │  Learning Loop   │
│  (RAGbits)       │  │  Visualization   │  │  (ACE)           │
│                  │  │                  │  │                  │
│ • Vector Embed   │  │ • PyVis HTML     │  │ • Decomposer     │
│ • Semantic Search│  │ • Communities    │  │ • Gauntlets      │
│ • Hybrid Search  │  │ • Centrality     │  │ • Optimizer      │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## Phase 1: Quick Wins (Week 1)

**Objective:** Install both projects and validate basic extraction capabilities.

### Task 1.1: Install ai-knowledge-graph

**Effort:** 0.5 days
**Priority:** P0
**Dependencies:** None

**Steps:**
1. Clone repository (if not already present):
   ```bash
   cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
   git clone https://github.com/robert-mcdermott/ai-knowledge-graph.git
   ```
2. Install dependencies:
   ```bash
   cd ai-knowledge-graph
   pip install -r requirements.txt
   ```
3. Verify installation:
   ```bash
   python generate-graph.py --test
   ```
4. Open generated HTML to verify visualization works

**Success Criteria:**
- ✅ All dependencies installed without errors
- ✅ Sample visualization generated successfully
- ✅ HTML file opens in browser and displays interactive graph

**Deliverables:**
- Installed ai-knowledge-graph package
- Sample visualization HTML file
- Installation verification report

---

### Task 1.2: Install DeepKE MCP Server

**Effort:** 2 days
**Priority:** P0
**Dependencies:** None

**Steps:**
1. Clone repository (if not already present):
   ```bash
   cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
   git clone --depth 1 https://github.com/zjunlp/DeepKE.git
   ```
2. Setup MCP server environment:
   ```bash
   cd DeepKE/mcp-tools
   pip install uv
   uv venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   uv add "mcp[cli]" httpx openai pyyaml
   ```
3. Configure environment variables:
   ```bash
   # Edit .env file
   DEEPKE_PATH="../"
   CONDA_PY="C:/path/to/anaconda3/envs/deepke/bin/"
   ```
4. Test MCP server:
   ```bash
   python run.py
   ```
5. Verify server responds to MCP requests

**Success Criteria:**
- ✅ MCP server starts without errors
- ✅ Server responds to test requests
- ✅ All 4 tools accessible (deepke_ner, deepke_re, deepke_ae, deepke_ee)

**Deliverables:**
- Working DeepKE MCP server
- Environment configuration (.env file)
- MCP server test report

---

### Task 1.3: Test DeepKE Extraction Quality

**Effort:** 0.5 days
**Priority:** P0
**Dependencies:** Task 1.2

**Steps:**
1. Prepare sample workflow execution data
2. Test each DeepKE tool:
   - `deepke_ner()` on solution code
   - `deepke_re()` on workflow transcripts
   - `deepke_ae()` on solution metadata
   - `deepke_ee()` on refinement loops
3. Evaluate extraction quality:
   - Precision, recall, F1 score
   - Entity types extracted
   - Relation types extracted
   - Event types extracted
4. Document results

**Success Criteria:**
- ✅ All 4 tools execute successfully
- ✅ Extraction quality > 70% F1 score
- ✅ Relevant entity/relation/event types identified

**Deliverables:**
- Extraction quality evaluation report
- Sample extraction results (JSON)
- Recommendations for schema customization

---

### Task 1.4: Test ai-knowledge-graph Pipeline

**Effort:** 1 day
**Priority:** P0
**Dependencies:** Task 1.1

**Steps:**
1. Prepare sample workflow execution data
2. Test complete pipeline:
   - SPO extraction from solutions
   - Entity standardization
   - Relationship inference
   - Visualization generation
3. Evaluate each stage:
   - Extraction quality
   - Entity reduction percentage
   - Relationship increase percentage
   - Visualization performance
4. Document results

**Success Criteria:**
- ✅ All pipeline stages execute successfully
- ✅ Entity standardization reduces entities by 20-30%
- ✅ Relationship inference increases relationships by 50-100%
- ✅ Visualization renders within 1 second for 500 nodes

**Deliverables:**
- Pipeline evaluation report
- Sample knowledge graphs (HTML + JSON)
- Performance metrics

---

### Task 1.5: Integration Prototypes

**Effort:** 1 day
**Priority:** P0
**Dependencies:** Tasks 1.3, 1.4

**Steps:**
1. Create prototype DeepKE extractor:
   ```python
   # knowledge_engine/deepke_extractor.py
   class DeepKEExtractor:
       async def extract_entities(self, text: str) -> List[Entity]
       async def extract_relations(self, text: str) -> List[Relation]
       async def extract_events(self, text: str) -> List[Event]
   ```
2. Create prototype AI-KG processor:
   ```python
   # knowledge_engine/aikg_processor.py
   class AIKGProcessor:
       async def standardize_entities(self, triples: List[Triple]) -> List[Triple]
       async def infer_relationships(self, triples: List[Triple]) -> List[Triple]
       async def visualize(self, triples: List[Triple]) -> str
   ```
3. Test both with sample data
4. Document integration challenges

**Success Criteria:**
- ✅ Both prototypes functional
- ✅ Integration challenges identified
- ✅ Preliminary architecture validated

**Deliverables:**
- Prototype extractor classes
- Integration assessment report
- Architecture recommendations

---

## Phase 2: Core Integration (Week 2)

**Objective:** Build production-ready adapters and integrate with Knowledge Engine.

### Task 2.1: ai-knowledge-graph Hephaestus Bridge

**Effort:** 2 days
**Priority:** P0
**Dependencies:** Task 1.5

**Steps:**
1. Create `ai_knowledge_graph_hephaestus_bridge.py`:
   ```python
   """
   Hephaestus bridge for ai-knowledge-graph integration.
   Exposes MCP tools for SPO extraction, entity standardization, and visualization.
   """
   from mcp.server import Server
   from src.knowledge_graph.main import process_text_in_chunks
   from src.knowledge_graph.entity_standardization import (
       standardize_entities, infer_relationships
   )
   from src.knowledge_graph.visualization import visualize_knowledge_graph

   app = Server("ai-knowledge-graph")

   @app.tool()
   async def aikg_extract_spo(
       text: str,
       chunk_size: int = 200,
       overlap: int = 20
   ) -> dict:
       """Extract Subject-Predicate-Object triples from text."""
       config = load_config()
       config["chunking"]["chunk_size"] = chunk_size
       config["chunking"]["overlap"] = overlap
       triples = process_text_in_chunks(config, text)
       return {"triples": triples}

   @app.tool()
   async def aikg_standardize_entities(
       triples: list
   ) -> dict:
       """Standardize entity names across triples."""
       config = load_config()
       standardized = standardize_entities(triples, config)
       return {"triples": standardized}

   @app.tool()
   async def aikg_infer_relationships(
       triples: list
   ) -> dict:
       """Infer additional relationships between entities."""
       config = load_config()
       inferred = infer_relationships(triples, config)
       return {"triples": inferred}

   @app.tool()
   async def aikg_visualize(
       triples: list,
       output_file: str = "knowledge_graph.html"
   ) -> dict:
       """Generate interactive knowledge graph visualization."""
       config = load_config()
       stats = visualize_knowledge_graph(triples, output_file, config)
       return {"stats": stats, "output_file": output_file}
   ```

2. Register MCP tools in OpenEvolve
3. Test all tools
4. Document API

**Success Criteria:**
- ✅ All 4 tools registered and functional
- ✅ Tools accessible via MCP protocol
- ✅ Error handling implemented
- ✅ API documentation complete

**Deliverables:**
- `ai_knowledge_graph_hephaestus_bridge.py`
- MCP tool registration
- API documentation

---

### Task 2.2: DeepKE MCP Integration

**Effort:** 1 day
**Priority:** P0
**Dependencies:** Task 1.3

**Steps:**
1. Configure DeepKE MCP server in OpenEvolve:
   ```yaml
   # mcp_agent.secrets.yaml
   mcp_servers:
     deepke:
       command: "uv"
       args:
         - "--directory"
         - "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/DeepKE/mcp-tools/tools"
         - "run"
         - "server.py"
   ```
2. Create DeepKE MCP client wrapper:
   ```python
   # knowledge_engine/deepke_client.py
   class DeepKEClient:
       def __init__(self, mcp_client):
           self.mcp_client = mcp_client

       async def extract_entities(self, text: str, schema: List[str]) -> List[Entity]:
           """Extract entities using DeepKE NER."""
           result = await self.mcp_client.call_tool("deepke_ner", {
               "text": text,
               "schema": schema
           })
           return self._parse_entities(result)

       async def extract_relations(self, text: str, schema: List[str]) -> List[Relation]:
           """Extract relations using DeepKE RE."""
           result = await self.mcp_client.call_tool("deepke_re", {
               "text": text,
               "schema": schema
           })
           return self._parse_relations(result)

       async def extract_attributes(self, text: str) -> List[Attribute]:
           """Extract attributes using DeepKE AE."""
           result = await self.mcp_client.call_tool("deepke_ae", {
               "text": text
           })
           return self._parse_attributes(result)

       async def extract_events(self, text: str, schema: List[str]) -> List[Event]:
           """Extract events using DeepKE EE."""
           result = await self.mcp_client.call_tool("deepke_ee", {
               "text": text,
               "schema": schema
           })
           return self._parse_events(result)
   ```
3. Test all extraction methods
4. Validate output formats

**Success Criteria:**
- ✅ DeepKE MCP server configured
- ✅ All 4 tools accessible via client
- ✅ Output parsing functional
- ✅ Error handling implemented

**Deliverables:**
- `deepke_client.py`
- MCP configuration
- Integration tests

---

### Task 2.3: KnowledgeArtifact Adapter (ai-knowledge-graph)

**Effort:** 2 days
**Priority:** P0
**Dependencies:** Task 2.1

**Steps:**
1. Create adapter class:
   ```python
   # knowledge_engine/aikg_artifact_adapter.py
   from dataclasses import dataclass
   from typing import List, Dict, Any
   from .core import KnowledgeArtifact

   @dataclass
   class KnowledgeArtifact:
       id: str
       artifact_type: Literal["solution_pattern", "problem_solution_mapping",
                              "critique_insight", "team_performance",
                              "gauntlet_effectiveness"]
       content: Dict[str, Any]
       source_workflow_id: str
       extraction_timestamp: float
       domain: Optional[str]
       problem_type: Optional[str]
       usage_count: int
       effectiveness_score: float
       related_artifacts: List[str]

   class AIKGArtifactAdapter:
       """Map AI-KG triples to KnowledgeArtifact schema."""

       async def extract_solution_pattern(
           self,
           triples: List[Triple],
           solution_metadata: Dict[str, Any]
       ) -> KnowledgeArtifact:
           """
           Extract solution pattern from AI-KG triples.

           Maps:
           - Subject → algorithm/component
           - Predicate → relationship type
           - Object → related component
           """
           # Group triples by entity
           entities = self._group_by_entity(triples)

           # Extract solution components
           components = {
               entity: self._extract_component_relations(entity_triples)
               for entity, entity_triples in entities.items()
           }

           return KnowledgeArtifact(
               id=str(uuid.uuid4()),
               artifact_type="solution_pattern",
               content={
                   "triples": triples,
                   "components": components,
                   "metadata": solution_metadata
               },
               source_workflow_id=solution_metadata.get("workflow_id"),
               extraction_timestamp=time.time(),
               domain=solution_metadata.get("domain"),
               problem_type=solution_metadata.get("problem_type"),
               usage_count=0,
               effectiveness_score=solution_metadata.get("quality_score", 0.0),
               related_artifacts=[]
           )

       async def extract_problem_solution_mapping(
           self,
           triples: List[Triple],
           problem_context: Dict[str, Any]
       ) -> KnowledgeArtifact:
           """Extract problem-solution mapping from triples."""
           # Extract problem entities
           problem_entities = self._find_problem_entities(triples)

           # Extract solution entities
           solution_entities = self._find_solution_entities(triples)

           # Map problems to solutions
           mappings = self._create_mappings(triples, problem_entities, solution_entities)

           return KnowledgeArtifact(
               id=str(uuid.uuid4()),
               artifact_type="problem_solution_mapping",
               content={
                   "mappings": mappings,
                   "problem_context": problem_context
               },
               source_workflow_id=problem_context.get("workflow_id"),
               extraction_timestamp=time.time(),
               domain=problem_context.get("domain"),
               problem_type=problem_context.get("problem_type"),
               usage_count=0,
               effectiveness_score=0.0,
               related_artifacts=[]
           )
   ```

2. Implement mapping methods
3. Add validation logic
4. Write unit tests

**Success Criteria:**
- ✅ All artifact types mappable
- ✅ Schema validation passes
- ✅ Unit tests pass
- ✅ Integration tests pass

**Deliverables:**
- `aikg_artifact_adapter.py`
- Unit tests
- Integration tests
- Schema documentation

---

### Task 2.4: KnowledgeArtifact Adapter (DeepKE)

**Effort:** 3 days
**Priority:** P0
**Dependencies:** Task 2.2

**Steps:**
1. Create adapter class:
   ```python
   # knowledge_engine/deepke_artifact_adapter.py
   from .deepke_client import DeepKEClient
   from .core import KnowledgeArtifact

   class DeepKEArtifactAdapter:
       """Map DeepKE NER/RE/AE/EE output to KnowledgeArtifact schema."""

       def __init__(self, deepke_client: DeepKEClient):
           self.client = deepke_client

       async def extract_solution_pattern(
           self,
           solution_code: str,
           solution_metadata: Dict[str, Any]
       ) -> KnowledgeArtifact:
           """
           Extract solution pattern using DeepKE NER + RE.

           Schema:
           - Entities: algorithm, data_structure, library, technique
           - Relations: uses, depends_on, implements, optimizes
           """
           # Extract entities
           entities = await self.client.extract_entities(
               solution_code,
               schema=["algorithm", "data_structure", "library", "technique"]
           )

           # Extract relations
           relations = await self.client.extract_relations(
               solution_code,
               schema=["uses", "depends_on", "implements", "optimizes"]
           )

           return KnowledgeArtifact(
               id=str(uuid.uuid4()),
               artifact_type="solution_pattern",
               content={
                   "entities": entities,
                   "relations": relations,
                   "metadata": solution_metadata
               },
               source_workflow_id=solution_metadata.get("workflow_id"),
               extraction_timestamp=time.time(),
               domain=solution_metadata.get("domain"),
               problem_type=solution_metadata.get("problem_type"),
               usage_count=0,
               effectiveness_score=solution_metadata.get("quality_score", 0.0),
               related_artifacts=[]
           )

       async def extract_critique_insight(
           self,
           critique_reports: List[str],
           verification_reports: List[str]
       ) -> List[KnowledgeArtifact]:
           """
           Extract critique insights using DeepKE EE.

           Schema:
           - Events: issue_identified, improvement_suggested, flaw_type
           """
           insights = []

           for critique in critique_reports:
               events = await self.client.extract_events(
                   critique,
                   schema=["issue_identified", "improvement_suggested", "flaw_type"]
               )

               insight = KnowledgeArtifact(
                   id=str(uuid.uuid4()),
                   artifact_type="critique_insight",
                   content={"events": events, "critique": critique},
                   source_workflow_id="",
                   extraction_timestamp=time.time(),
                   usage_count=0,
                   effectiveness_score=0.0,
                   related_artifacts=[]
               )
               insights.append(insight)

           return insights

       async def extract_workflow_events(
           self,
           workflow_execution: WorkflowExecution
       ) -> List[KnowledgeArtifact]:
           """
           Extract workflow events using DeepKE EE.

           Schema:
           - Events: refinement_loop, failure, verification, resource_usage
           """
           full_transcript = workflow_execution.full_transcript

           events = await self.client.extract_events(
               full_transcript,
               schema=["refinement_loop", "failure", "verification", "resource_usage"]
           )

           return [
               KnowledgeArtifact(
                   id=str(uuid.uuid4()),
                   artifact_type="workflow_event",
                   content={"event": event},
                   source_workflow_id=workflow_execution.id,
                   extraction_timestamp=time.time(),
                   usage_count=0,
                   effectiveness_score=0.0,
                   related_artifacts=[]
               )
               for event in events
           ]
   ```

2. Implement extraction methods for all artifact types
3. Add schema customization
4. Write unit tests

**Success Criteria:**
- ✅ All artifact types extractable
- ✅ Schema customization functional
- ✅ Unit tests pass
- ✅ Integration tests pass

**Deliverables:**
- `deepke_artifact_adapter.py`
- Unit tests
- Integration tests
- Schema customization guide

---

### Task 2.5: Combined Extraction Pipeline

**Effort:** 2 days
**Priority:** P0
**Dependencies:** Tasks 2.3, 2.4

**Steps:**
1. Create unified pipeline:
   ```python
   # knowledge_engine/unified_extractor.py
   from .deepke_artifact_adapter import DeepKEArtifactAdapter
   from .aikg_artifact_adapter import AIKGArtifactAdapter
   from .aikg_processor import AIKGProcessor

   class UnifiedKnowledgeExtractor:
       """
       Unified extraction pipeline combining DeepKE and AI-KG.

       Pipeline:
       1. DeepKE extracts entities/relations/events (high quality)
       2. AI-KG processes and enriches (standardization, inference)
       3. Both map to KnowledgeArtifact schema
       4. AI-KG generates visualization
       """

       def __init__(
           self,
           deepke_adapter: DeepKEArtifactAdapter,
           aikg_adapter: AIKGArtifactAdapter,
           aikg_processor: AIKGProcessor
       ):
           self.deepke = deepke_adapter
           self.aikg = aikg_adapter
           self.processor = aikg_processor

       async def extract_from_workflow(
           self,
           workflow_execution: WorkflowExecution
       ) -> List[KnowledgeArtifact]:
           """
           Extract all knowledge artifacts from workflow execution.

           Returns:
               List of KnowledgeArtifact objects
           """
           artifacts = []

           # Phase 1: DeepKE extraction (high quality)
           # Extract solution patterns
           for solution in workflow_execution.verified_solutions:
               deepke_artifact = await self.deepke.extract_solution_pattern(
                   solution.code,
                   solution.metadata
               )
               artifacts.append(deepke_artifact)

           # Extract critique insights
           critique_artifacts = await self.deepke.extract_critique_insight(
               workflow_execution.critique_reports,
               workflow_execution.verification_reports
           )
           artifacts.extend(critique_artifacts)

           # Extract workflow events
           event_artifacts = await self.deepke.extract_workflow_events(
               workflow_execution
           )
           artifacts.extend(event_artifacts)

           # Phase 2: Convert to AI-KG triples
           triples = self._artifacts_to_triples(artifacts)

           # Phase 3: AI-KG enrichment
           standardized_triples = await self.processor.standardize_entities(triples)
           enriched_triples = await self.processor.infer_relationships(standardized_triples)

           # Phase 4: Map back to artifacts
           enriched_artifacts = await self.aikg.extract_solution_pattern(
               enriched_triples,
               {"workflow_id": workflow_execution.id}
           )
           artifacts.append(enriched_artifact)

           # Phase 5: Generate visualization
           viz_path = await self.processor.visualize(
               enriched_triples,
               output_file=f"workflow_{workflow_execution.id}_graph.html"
           )

           return artifacts, viz_path
   ```

2. Implement workflow integration hooks
3. Add error handling and retry logic
4. Write integration tests

**Success Criteria:**
- ✅ Unified pipeline executes end-to-end
- ✅ DeepKE and AI-KG both utilized
- ✅ Enriched artifacts generated
- ✅ Visualization created
- ✅ Integration tests pass

**Deliverables:**
- `unified_extractor.py`
- Integration tests
- Pipeline documentation

---

## Phase 3: Production Integration (Week 3)

**Objective:** Integrate with WorkflowKnowledgeExtractor and deploy to production.

### Task 3.1: WorkflowKnowledgeExtractor Integration

**Effort:** 2 days
**Priority:** P0
**Dependencies:** Task 2.5

**Steps:**
1. Extend WorkflowKnowledgeExtractor:
   ```python
   # knowledge_engine/workflow_extractor.py
   from .unified_extractor import UnifiedKnowledgeExtractor
   from .core import WorkflowExecution

   class WorkflowKnowledgeExtractor:
       """
       Extract knowledge from Decomposition Workflow execution.

       Hooks into:
       - Stage 0: AnalyzedContext
       - Stage 3: SubProblemSolutions + CritiqueReports + VerificationReports
       - Stage 5: RefinementLoops
       """

       def __init__(self, unified_extractor: UnifiedKnowledgeExtractor):
           self.extractor = unified_extractor

       async def extract_from_stage_0(
           self,
           analyzed_context: AnalyzedContext
       ) -> List[KnowledgeArtifact]:
           """Extract problem type patterns, complexity metrics."""
           # Use AI-KG SPO extraction
           triples = await self.extractor.aikg.extract_spo(
               analyzed_context.text_description
           )

           # Map to problem_solution_mapping artifacts
           artifact = await self.extractor.aikg.extract_problem_solution_mapping(
               triples,
               {
                   "workflow_id": analyzed_context.workflow_id,
                   "domain": analyzed_context.domain,
                   "problem_type": analyzed_context.problem_type
               }
           )

           return [artifact]

       async def extract_from_stage_3(
           self,
           sub_problem_solutions: Dict[str, SolutionAttempt],
           critique_reports: List[CritiqueReport],
           verification_reports: List[VerificationReport]
       ) -> List[KnowledgeArtifact]:
           """Extract solution patterns, critique insights, team performance."""
           artifacts = []

           # Extract solution patterns (DeepKE)
           for solution_id, solution in sub_problem_solutions.items():
               artifact = await self.extractor.deepke.extract_solution_pattern(
                   solution.code,
                   solution.metadata
               )
               artifacts.append(artifact)

           # Extract critique insights (DeepKE)
           critique_texts = [report.text for report in critique_reports]
           critique_artifacts = await self.extractor.deepke.extract_critique_insight(
               critique_texts,
               [report.text for report in verification_reports]
           )
           artifacts.extend(critique_artifacts)

           return artifacts

       async def extract_from_stage_5(
           self,
           refinement_loops: List[RefinementLoop]
       ) -> List[KnowledgeArtifact]:
           """Extract failure learning artifacts, prevention strategies."""
           artifacts = []

           # Extract events from refinement loops (DeepKE)
           for loop in refinement_loops:
               events = await self.extractor.deepke.extract_events(
                   loop.transcript,
                   schema=["failure", "fix_attempt", "prevention_strategy"]
               )

               artifact = KnowledgeArtifact(
                   id=str(uuid.uuid4()),
                   artifact_type="failure_learning",
                   content={"events": events, "loop": loop},
                   source_workflow_id=loop.workflow_id,
                   extraction_timestamp=time.time(),
                   usage_count=0,
                   effectiveness_score=0.0,
                   related_artifacts=[]
               )
               artifacts.append(artifact)

           return artifacts

       async def build_knowledge_base_update(
           self,
           all_artifacts: List[KnowledgeArtifact]
       ) -> KnowledgeBaseUpdate:
           """
           Prepare comprehensive update for all system components.

           Updates:
           - Decomposer with solution patterns
           - Gauntlets with effectiveness data
           - AI recommender with problem mappings
           - ML models with fine-tuning data
           """
           # Group artifacts by type
           solution_patterns = [a for a in all_artifacts if a.artifact_type == "solution_pattern"]
           problem_mappings = [a for a in all_artifacts if a.artifact_type == "problem_solution_mapping"]
           critique_insights = [a for a in all_artifacts if a.artifact_type == "critique_insight"]

           return KnowledgeBaseUpdate(
               solution_patterns=solution_patterns,
               problem_mappings=problem_mappings,
               critique_insights=critique_insights,
               timestamp=time.time()
           )
   ```

2. Add hooks to workflow stages
3. Test with real workflow executions
4. Validate artifact quality

**Success Criteria:**
- ✅ All 3 stage hooks functional
- ✅ Artifacts extracted from real workflows
- ✅ KnowledgeBaseUpdate generated
- ✅ Integration tests pass

**Deliverables:**
- `workflow_extractor.py`
- Stage integration hooks
- Integration tests

---

### Task 3.2: Vector Database Integration (RAGbits)

**Effort:** 1.5 days
**Priority:** P1
**Dependencies:** Task 3.1

**Steps:**
1. Setup RAGbits vector store
2. Create embeddings for artifacts:
   ```python
   # knowledge_engine/vector_store.py
   from ragbits.vector_store import VectorStore
   from sentence_transformers import SentenceTransformer

   class ArtifactVectorStore:
       """Vector embeddings for semantic search."""

       def __init__(self, vector_store: VectorStore):
           self.store = vector_store
           self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

       async def index_artifacts(self, artifacts: List[KnowledgeArtifact]):
           """Index artifacts in vector database."""
           for artifact in artifacts:
               # Create embedding
               text = self._artifact_to_text(artifact)
               embedding = self.encoder.encode(text)

               # Store in vector database
               await self.store.add(
                   id=artifact.id,
                   embedding=embedding,
                   metadata={
                       "artifact_type": artifact.artifact_type,
                       "domain": artifact.domain,
                       "problem_type": artifact.problem_type
                   }
               )

       async def search_similar(
           self,
           query: str,
           artifact_type: str = None,
           top_k: int = 10
       ) -> List[KnowledgeArtifact]:
           """Search for similar artifacts."""
           query_embedding = self.encoder.encode(query)

           results = await self.store.search(
               query_embedding,
               filters={"artifact_type": artifact_type} if artifact_type else None,
               top_k=top_k
           )

           return results
   ```

3. Integrate with extraction pipeline
4. Test semantic search

**Success Criteria:**
- ✅ Artifacts indexed in vector store
- ✅ Semantic search functional
- ✅ Search relevance > 70%

**Deliverables:**
- `vector_store.py`
- Vector database integration
- Search interface

---

### Task 3.3: Knowledge Base Interface

**Effort:** 1.5 days
**Priority:** P1
**Dependencies:** Task 3.2

**Steps:**
1. Extend RAGbits Chat UI
2. Add artifact browser
3. Add knowledge graph viewer
4. Add management UI (add/edit/delete)

**Success Criteria:**
- ✅ Artifact browser functional
- ✅ Knowledge graph viewer displays AI-KG visualizations
- ✅ CRUD operations work
- ✅ UI integrated with OpenEvolve dashboard

**Deliverables:**
- Knowledge base UI components
- UI integration
- User documentation

---

### Task 3.4: End-to-End Testing

**Effort:** 2 days
**Priority:** P0
**Dependencies:** Tasks 3.1, 3.2, 3.3

**Steps:**
1. Create test suite:
   - Unit tests for all components
   - Integration tests for pipeline
   - End-to-end tests with real workflows
2. Performance testing:
   - Extraction time
   - Visualization rendering time
   - Search query time
3. Quality validation:
   - Extraction accuracy (F1 score)
   - Entity standardization quality
   - Relationship inference precision
4. Document results

**Success Criteria:**
- ✅ All unit tests pass (>90% coverage)
- ✅ All integration tests pass
- ✅ End-to-end test successful
- ✅ Extraction quality > 80% F1
- ✅ Performance within acceptable limits

**Deliverables:**
- Test suite
- Performance benchmarks
- Quality validation report

---

### Task 3.5: Documentation and Deployment

**Effort:** 2 days
**Priority:** P0
**Dependencies:** Task 3.4

**Steps:**
1. Write technical documentation:
   - Architecture overview
   - API documentation
   - Integration guide
   - Troubleshooting guide
2. Write user documentation:
   - Quick start guide
   - Configuration guide
   - Best practices
3. Deploy to production:
   - Configure production environment
   - Setup monitoring
   - Run smoke tests
4. Handoff to operations team

**Success Criteria:**
- ✅ Documentation complete and reviewed
- ✅ Production deployment successful
- ✅ Monitoring operational
- ✅ Operations team trained

**Deliverables:**
- Technical documentation
- User documentation
- Deployment guide
- Monitoring dashboard

---

## Dependencies

### Critical Path

```
Task 1.1 (0.5d)
    ↓
Task 1.4 (1d)
    ↓
Task 1.5 (1d)
    ↓
    ├→ Task 2.1 (2d) → Task 2.3 (2d) ┐
    │                                ↓
    └→ Task 1.2 (2d) → Task 1.3 (0.5d) → Task 2.2 (1d) → Task 2.4 (3d) → Task 2.5 (2d)
                                                                     ↓
                                                        Task 3.1 (2d) → Task 3.2 (1.5d)
                                                                          ↓
                                                                    Task 3.3 (1.5d)
                                                                          ↓
                                                                    Task 3.4 (2d)
                                                                          ↓
                                                                    Task 3.5 (2d)
```

### Parallelization Opportunities

**Week 1:**
- Task 1.1 and Task 1.2 can run in parallel

**Week 2:**
- Task 2.1 and Task 2.2 can run in parallel (after Task 1.5)
- Task 2.3 and Task 2.4 can run in parallel (after respective bridges)

**Week 3:**
- Task 3.2 and Task 3.3 can run in parallel (after Task 3.1)

---

## Success Criteria

### Phase 1 Success (Week 1)

- ✅ ai-knowledge-graph installed and functional
- ✅ DeepKE MCP server operational
- ✅ Extraction quality validated (>70% F1)
- ✅ Prototypes demonstrate integration feasibility

### Phase 2 Success (Week 2)

- ✅ Both Hephaestus bridges functional
- ✅ KnowledgeArtifact adapters implemented
- ✅ Unified extraction pipeline operational
- ✅ Integration tests passing

### Phase 3 Success (Week 3)

- ✅ WorkflowKnowledgeExtractor integrated
- ✅ Vector database operational
- ✅ Knowledge base UI functional
- ✅ End-to-end testing complete
- ✅ Production deployment successful

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Dependency conflicts** | Medium | Medium | Use MCP isolation, test early |
| **GPU unavailable** | Low | Medium | Use DeepKE MCP server (cloud) |
| **Extraction quality insufficient** | Low | High | Fine-tune DeepKE models on workflow data |
| **Integration complexity** | Medium | Medium | Phased approach, prototype first |
| **Timeline overrun** | Medium | Medium | Prioritize P0 tasks, defer P1 |

---

## Resource Requirements

### Hardware

- **Development:**
  - CPU: 4+ cores
  - RAM: 16 GB
  - Storage: 10 GB

- **Production (DeepKE):**
  - CPU: 8+ cores
  - RAM: 32 GB
  - GPU: NVIDIA GPU with 8+ GB VRAM (recommended)
  - Storage: 50 GB

### Software

- Python 3.8+
- Conda (for DeepKE environment isolation)
- UV (for MCP server)
- PostgreSQL/MySQL (for artifact storage)
- Qdrant/Weaviate (for vector store, via RAGbits)

### Personnel

- 1 Backend Developer (part-time, 50% FTE)
- Duration: 3 weeks
- Total effort: ~15 developer-days

---

## Maintenance Plan

### Regular Maintenance

**Monthly:**
- Update dependencies
- Review extraction quality metrics
- Optimize pipeline performance

**Quarterly:**
- Fine-tune DeepKE models on new workflow data
- Update AI-KG inference rules
- Review and update documentation

### Long-term Enhancements

**6-12 months:**
- Implement SolutionPatternMiner (ML clustering)
- Build TeamPerformanceTracker
- Build GauntletEffectivenessAnalyzer
- Optimize for production scale

---

## Appendix

### A. File Structure

```
knowledge_engine/
├── core.py                          # KnowledgeArtifact schema
├── engine.py                        # Main KnowledgeEngine facade
├── deepke_client.py                 # DeepKE MCP client (NEW)
├── deepke_artifact_adapter.py       # DeepKE → KnowledgeArtifact (NEW)
├── aikg_processor.py                # AI-KG processing wrapper (NEW)
├── aikg_artifact_adapter.py         # AI-KG → KnowledgeArtifact (NEW)
├── unified_extractor.py             # Combined extraction pipeline (NEW)
├── workflow_extractor.py            # Stage 0/3/5 hooks (NEW)
├── vector_store.py                  # RAGbits integration (NEW)
└── tests/
    ├── test_deepke_adapter.py       # DeepKE adapter tests (NEW)
    ├── test_aikg_adapter.py         # AI-KG adapter tests (NEW)
    ├── test_unified_extractor.py    # Unified pipeline tests (NEW)
    └── test_workflow_extractor.py   # Workflow integration tests (NEW)

ai_knowledge_graph_hephaestus_bridge.py  # AI-KG MCP tools (NEW)
```

### B. Configuration Files

```yaml
# mcp_agent.secrets.yaml (additions)
mcp_servers:
  ai-knowledge-graph:
    command: "python"
    args:
      - "-m"
      - "ai_knowledge_graph_hephaestus_bridge"
  deepke:
    command: "uv"
    args:
      - "--directory"
      - "C:/Users/mmeadow/Documents/OpenEvolve/Frontend/DeepKE/mcp-tools/tools"
      - "run"
      - "server.py"
```

### C. Environment Variables

```bash
# ai-knowledge-graph
AIKG_MODEL="gpt-4"
AIKG_API_KEY="sk-..."
AIKG_BASE_URL="https://api.openai.com/v1"

# DeepKE
DEEPKE_PATH="../"
CONDA_PY="/path/to/anaconda3/envs/deepke/bin/"
```

---

**Document Version:** 1.0
**Last Updated:** 2025-12-31
**Status:** READY FOR IMPLEMENTATION
**Next Review:** After Phase 1 completion
