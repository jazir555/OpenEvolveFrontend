"""
Complete Integration Test for Knowledge Engine

Tests all phases (1-6) and components working together.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_engine import (
    # Phase 1: Knowledge Graph
    NodeType, EdgeType, KnowledgeNode, KnowledgeEdge, KnowledgeGraph,
    GraphCRUD, ConnectionPool, CypherQueryBuilder,
    
    # Phase 2: DeepKE
    DeepKEExtractor, ExtractedEntity, ExtractedRelation,
    
    # Phase 3: Hybrid Search
    HybridSearch, VectorSearch, GraphSearch, FusionStrategy,
    
    # Phase 4: Architectural Gaps
    SandboxManager, SandboxType, SecurityPolicy,
    VisionLanguageMonitor, VLMProvider,
    BrowserResearchAgent,
    ComplexityRouter, ModelTier,
    Chronicle, EpisodeType,
    
    # Phase 5: OpenEvolve Integration
    OpenEvolveIntegration, ProjectContext, ProjectLifecycleStage,
    
    # Phase 6: Query Interface
    NaturalLanguageQueryParser, ResultFormatter, QueryCache, FeedbackLoop,
    
    # Unified Interface
    UnifiedKnowledgeEngine
)


class Colors:
    OK = "[OK]"
    FAIL = "[FAIL]"
    INFO = "[INFO]"


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_result(name, success, details=""):
    status = Colors.OK if success else Colors.FAIL
    print(f"  {status} {name}")
    if details:
        print(f"       {details}")


async def test_phase1_knowledge_graph():
    """Test Phase 1: Core Knowledge Graph"""
    print_header("PHASE 1: Core Knowledge Graph")
    
    results = []
    
    # Test 1: Schema Creation
    try:
        from knowledge_engine.graph.schema import GraphSchema, NodeSchema, NodeType
        schema = GraphSchema(name="Test Schema")
        assert NodeType.CONCEPT in schema.node_schemas
        results.append(("Schema Creation", True))
    except Exception as e:
        results.append(("Schema Creation", False, str(e)))
    
    # Test 2: Knowledge Node
    try:
        from knowledge_engine.graph.models import NodeProperties
        node = KnowledgeNode(
            node_type=NodeType.CONCEPT,
            properties=NodeProperties(name="Test Node")
        )
        assert node.id is not None
        results.append(("Knowledge Node", True))
    except Exception as e:
        results.append(("Knowledge Node", False, str(e)))
    
    # Test 3: Knowledge Graph
    try:
        kg = KnowledgeGraph(name="Test Graph")
        node = KnowledgeNode(
            node_type=NodeType.CONCEPT,
            properties=NodeProperties(name="Node 1")
        )
        kg.add_node(node)
        assert len(kg.nodes) == 1
        results.append(("Knowledge Graph", True))
    except Exception as e:
        results.append(("Knowledge Graph", False, str(e)))
    
    # Test 4: Connection Pool (mock mode)
    try:
        pool = ConnectionPool()
        assert pool is not None
        results.append(("Connection Pool", True))
    except Exception as e:
        results.append(("Connection Pool", False, str(e)))
    
    # Test 5: Cypher Builder
    try:
        builder = CypherQueryBuilder()
        query, params = builder.match_node(node_type=NodeType.CONCEPT).return_("n").build()
        assert "MATCH" in query
        results.append(("Cypher Builder", True))
    except Exception as e:
        results.append(("Cypher Builder", False, str(e)))
    
    for name, success, *details in results:
        print_result(name, success, details[0] if details else "")
    
    return all(r[1] for r in results)


async def test_phase2_deepke():
    """Test Phase 2: DeepKE Integration"""
    print_header("PHASE 2: DeepKE Integration")
    
    results = []
    
    # Test 1: DeepKE Extractor
    try:
        extractor = DeepKEExtractor()
        text = "OpenAI is a company in San Francisco that develops GPT-4."
        result = extractor.extract(text)
        assert len(result.entities) > 0
        results.append(("Entity Extraction", True, f"Found {len(result.entities)} entities"))
    except Exception as e:
        results.append(("Entity Extraction", False, str(e)))
    
    # Test 2: Document Processing
    try:
        from knowledge_engine.deepke.pipeline import DeepKEPipeline
        pipeline = DeepKEPipeline()
        result = pipeline.extractor.extract_from_document(
            "Python is a programming language. JavaScript is used for web development.",
            chunk_size=100
        )
        assert len(result.entities) > 0
        results.append(("Document Processing", True, f"Found {len(result.entities)} entities"))
    except Exception as e:
        results.append(("Document Processing", False, str(e)))
    
    for name, success, *details in results:
        print_result(name, success, details[0] if details else "")
    
    return all(r[1] for r in results)


async def test_phase3_hybrid_search():
    """Test Phase 3: Hybrid Queries"""
    print_header("PHASE 3: Hybrid Queries")
    
    results = []
    
    # Test 1: Vector Search (mock)
    try:
        vs = VectorSearch()
        # Mock mode should work without Chroma
        results.append(("Vector Search Init", True, "Mock mode"))
    except Exception as e:
        results.append(("Vector Search Init", False, str(e)))
    
    # Test 2: Graph Search (mock)
    try:
        gs = GraphSearch()
        results.append(("Graph Search Init", True, "Mock mode"))
    except Exception as e:
        results.append(("Graph Search Init", False, str(e)))
    
    # Test 3: Hybrid Search
    try:
        hs = HybridSearch()
        results.append(("Hybrid Search Init", True))
    except Exception as e:
        results.append(("Hybrid Search Init", False, str(e)))
    
    # Test 4: Query Optimizer
    try:
        from knowledge_engine.hybrid.optimizer import QueryOptimizer
        opt = QueryOptimizer()
        query_type = opt.analyze("What is machine learning?")
        assert query_type is not None
        results.append(("Query Optimizer", True, f"Detected: {query_type.name}"))
    except Exception as e:
        results.append(("Query Optimizer", False, str(e)))
    
    for name, success, *details in results:
        print_result(name, success, details[0] if details else "")
    
    return all(r[1] for r in results)


async def test_phase4_architectural_gaps():
    """Test Phase 4: Architectural Gaps"""
    print_header("PHASE 4: Architectural Gaps")
    
    results = []
    
    # Test 1: Sandbox
    try:
        sandbox = SandboxManager()
        result = await sandbox.execute_python("print('Hello')")
        assert result.success
        results.append(("Sandbox Execution", True, f"Exit code: {result.exit_code}"))
    except Exception as e:
        results.append(("Sandbox Execution", False, str(e)))
    
    # Test 2: Vision Monitor
    try:
        vlm = VisionLanguageMonitor(provider=VLMProvider.MOCK)
        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(b'fake_image_data')
            temp_path = f.name
        analysis = await vlm.analyze_screenshot(temp_path, "What do you see?")
        import os
        os.unlink(temp_path)
        assert analysis is not None
        results.append(("Vision Monitor", True, "Mock analysis working"))
    except Exception as e:
        results.append(("Vision Monitor", False, str(e)))
    
    # Test 3: Browser Agent
    try:
        browser = BrowserResearchAgent()
        results_search = await browser.search("Python programming")
        assert results_search is not None
        results.append(("Browser Agent", True, f"Mock search returned {len(results_search)} results"))
    except Exception as e:
        results.append(("Browser Agent", False, str(e)))
    
    # Test 4: Complexity Router
    try:
        router = ComplexityRouter()
        decision = router.route("What time is it?")
        assert decision.selected_tier == ModelTier.FAST
        results.append(("Complexity Router", True, f"Routed to: {decision.selected_tier.value}"))
    except Exception as e:
        results.append(("Complexity Router", False, str(e)))
    
    # Test 5: Chronicle
    try:
        chronicle = Chronicle()
        episode_id = chronicle.record_episode(
            episode_type=EpisodeType.ACTION,
            action="Test action",
            agent="test_agent"
        )
        assert episode_id is not None
        results.append(("Chronicle", True, f"Episode: {episode_id[:20]}..."))
    except Exception as e:
        results.append(("Chronicle", False, str(e)))
    
    for name, success, *details in results:
        print_result(name, success, details[0] if details else "")
    
    return all(r[1] for r in results)


async def test_phase5_openevolve():
    """Test Phase 5: OpenEvolve Integration"""
    print_header("PHASE 5: OpenEvolve Integration")
    
    results = []
    
    # Test 1: Project Registration
    try:
        integration = OpenEvolveIntegration()
        project = ProjectContext(
            project_id="test_123",
            name="Test Project",
            description="A test project"
        )
        pid = integration.register_project(project)
        assert pid == "test_123"
        results.append(("Project Registration", True))
    except Exception as e:
        results.append(("Project Registration", False, str(e)))
    
    # Test 2: Context Injection
    try:
        query = "What is the architecture?"
        enriched = integration.inject_context(query, "test_123")
        assert "Test Project" in enriched
        results.append(("Context Injection", True, "Context enriched"))
    except Exception as e:
        results.append(("Context Injection", False, str(e)))
    
    # Test 3: Project Lifecycle
    try:
        project = integration.get_project("test_123")
        assert project.stage == ProjectLifecycleStage.INITIALIZED
        results.append(("Project Lifecycle", True, f"Stage: {project.stage.value}"))
    except Exception as e:
        results.append(("Project Lifecycle", False, str(e)))
    
    for name, success, *details in results:
        print_result(name, success, details[0] if details else "")
    
    return all(r[1] for r in results)


async def test_phase6_query_interface():
    """Test Phase 6: Query Interface"""
    print_header("PHASE 6: Query Interface")
    
    results = []
    
    # Test 1: Query Parser
    try:
        parser = NaturalLanguageQueryParser()
        parsed = parser.parse("What is machine learning?")
        assert parsed is not None
        assert len(parsed.keywords) > 0
        results.append(("Query Parser", True, f"Keywords: {parsed.keywords[:3]}"))
    except Exception as e:
        results.append(("Query Parser", False, str(e)))
    
    # Test 2: Result Formatter
    try:
        from knowledge_engine.hybrid.search import SearchResult
        from knowledge_engine.query.formatter import OutputFormat
        formatter = ResultFormatter()
        results_list = [
            SearchResult(id="1", score=0.9, content="Test result 1", source="test"),
            SearchResult(id="2", score=0.8, content="Test result 2", source="test")
        ]
        formatted = formatter.format(results_list, output_format=OutputFormat.TEXT)
        assert formatted is not None
        results.append(("Result Formatter", True, "Text format working"))
    except Exception as e:
        results.append(("Result Formatter", False, str(e)))
    
    # Test 3: Query Cache
    try:
        cache = QueryCache()
        cache.set("test_query", [{"id": "1", "result": "test"}])
        cached = cache.get("test_query")
        assert cached is not None
        results.append(("Query Cache", True, "Cache working"))
    except Exception as e:
        results.append(("Query Cache", False, str(e)))
    
    # Test 4: Feedback Loop
    try:
        from knowledge_engine.query.feedback import QueryFeedback, FeedbackType
        feedback = FeedbackLoop()
        fb = QueryFeedback(
            query="test",
            result_id="1",
            feedback_type=FeedbackType.THUMBS_UP
        )
        feedback.record_feedback(fb)
        insights = feedback.get_insights()
        assert insights is not None
        results.append(("Feedback Loop", True, f"Total feedback: {insights.get('total_feedback', 0)}"))
    except Exception as e:
        results.append(("Feedback Loop", False, str(e)))
    
    for name, success, *details in results:
        print_result(name, success, details[0] if details else "")
    
    return all(r[1] for r in results)


async def test_unified_interface():
    """Test Unified Knowledge Engine Interface"""
    print_header("UNIFIED INTERFACE")
    
    results = []
    
    # Test 1: Unified KE Init
    try:
        ke = UnifiedKnowledgeEngine(config={
            'enable_graph': False,  # Skip graph to avoid Neo4j dependency
            'enable_deepke': True,
            'enable_hybrid_search': False,
            'enable_sandbox': False,
            'enable_vision': False,
            'enable_browser': False,
            'enable_openevolve': False
        })
        status = ke.get_status()
        assert status['query_parser'] is True
        results.append(("Unified KE Init", True, f"Components: {sum(status.values())}/{len(status)}"))
    except Exception as e:
        results.append(("Unified KE Init", False, str(e)))
    
    # Test 2: Query Execution
    try:
        ke = UnifiedKnowledgeEngine(config={
            'enable_graph': False,
            'enable_deepke': True,
            'enable_hybrid_search': False,
        })
        result = await ke.query("What is Python?")
        assert result is not None
        results.append(("Query Execution", True, f"Success: {result.get('success')}"))
    except Exception as e:
        results.append(("Query Execution", False, str(e)))
    
    # Test 3: Extract
    try:
        ke = UnifiedKnowledgeEngine(config={
            'enable_graph': False,
            'enable_deepke': True,
        })
        result = await ke.extract("Python is a programming language.")
        assert result is not None
        results.append(("Extract", True, f"Success: {result.get('success')}"))
    except Exception as e:
        results.append(("Extract", False, str(e)))
    
    # Test 4: Chronicle
    try:
        ke = UnifiedKnowledgeEngine(config={'enable_graph': False})
        result = await ke.record_episode(
            episode_type=EpisodeType.SUCCESS,
            action="Test episode",
            agent="test"
        )
        assert result is not None
        results.append(("Record Episode", True, f"Episode: {result.get('episode_id', 'N/A')[:20]}..."))
    except Exception as e:
        results.append(("Record Episode", False, str(e)))
    
    for name, success, *details in results:
        print_result(name, success, details[0] if details else "")
    
    return all(r[1] for r in results)


async def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("  KNOWLEDGE ENGINE - COMPLETE INTEGRATION TEST")
    print("="*60)
    
    results = {
        "Phase 1: Knowledge Graph": await test_phase1_knowledge_graph(),
        "Phase 2: DeepKE": await test_phase2_deepke(),
        "Phase 3: Hybrid Search": await test_phase3_hybrid_search(),
        "Phase 4: Architectural Gaps": await test_phase4_architectural_gaps(),
        "Phase 5: OpenEvolve": await test_phase5_openevolve(),
        "Phase 6: Query Interface": await test_phase6_query_interface(),
        "Unified Interface": await test_unified_interface()
    }
    
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    
    for phase, passed in results.items():
        status = Colors.OK if passed else Colors.FAIL
        print(f"  {status} {phase}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "="*60)
    print(f"  RESULT: {passed}/{total} phases passed")
    if passed == total:
        print("  STATUS: ALL TESTS PASSED [OK]")
    else:
        print(f"  STATUS: {total - passed} PHASE(S) FAILED [FAIL]")
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
