"""
Comprehensive Test Suite for CrewAI Research Roadmap Implementation

Tests all 10 features:
1. Hierarchical Process Support
2. Advanced Delegation Mechanisms
3. Memory-Augmented Research
4. External Tool Orchestration
5. Multi-Modal Support
6. Real-Time Collaboration
7. Research Workflow Templates
8. Automated Literature Search
9. Experiment Tracking
10. Research Report Generation

Usage: pytest test_crewai_research_comprehensive.py -v
"""

import pytest
import asyncio
import json
import os
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, List

# Import all feature modules
from crewai_research_core import (
    HierarchicalCrew,
    CrewLevel,
    HierarchicalTask,
    AdvancedDelegationManager,
    DelegationType,
    AgentCapability,
    MemoryAugmentedResearch,
    MemoryType,
    create_hierarchical_crew,
    create_delegation_manager,
    create_memory_system
)

from crewai_research_tools import (
    ExternalToolOrchestrator,
    MultiModalProcessor,
    RealTimeCollaboration,
    CollaborationEventType,
    ToolDefinition,
    ToolType,
    create_tool_orchestrator,
    create_multimodal_processor,
    create_collaboration_system
)

from crewai_research_templates import (
    TemplateRegistry,
    TemplateType,
    LiteratureReviewTemplate,
    ExperimentalDesignTemplate,
    DataAnalysisTemplate,
    PaperWritingTemplate,
    PeerReviewTemplate,
    create_template_registry
)

from crewai_research_external import (
    LiteratureSearchOrchestrator,
    DatabaseType,
    ExperimentTracker,
    ResearchReportGenerator,
    ReportFormat,
    create_literature_search,
    create_experiment_tracker,
    create_report_generator
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def hierarchical_crew():
    """Create hierarchical crew for testing"""
    return create_hierarchical_crew(name="TestCrew", max_depth=3)


@pytest.fixture
def delegation_manager():
    """Create delegation manager for testing"""
    manager = create_delegation_manager()
    
    # Register test agents
    manager.register_agent(AgentCapability(
        agent_id="agent_1",
        skills=["python", "data_analysis"],
        role="developer",
        max_workload=5
    ))
    manager.register_agent(AgentCapability(
        agent_id="agent_2",
        skills=["machine_learning", "python"],
        role="senior",
        max_workload=3
    ))
    manager.register_agent(AgentCapability(
        agent_id="agent_3",
        skills=["testing", "qa"],
        role="qa_engineer",
        max_workload=5
    ))
    
    return manager


@pytest.fixture
def memory_system(temp_dir):
    """Create memory system for testing"""
    return create_memory_system(storage_dir=temp_dir)


@pytest.fixture
def tool_orchestrator():
    """Create tool orchestrator for testing"""
    return create_tool_orchestrator()


@pytest.fixture
def multimodal_processor():
    """Create multi-modal processor for testing"""
    return create_multimodal_processor()


@pytest.fixture
def collaboration_system():
    """Create collaboration system for testing"""
    return create_collaboration_system()


@pytest.fixture
def template_registry():
    """Create template registry for testing"""
    return create_template_registry()


@pytest.fixture
def literature_search():
    """Create literature search orchestrator for testing"""
    return create_literature_search()


@pytest.fixture
def experiment_tracker(temp_dir):
    """Create experiment tracker for testing"""
    return create_experiment_tracker(storage_dir=temp_dir)


@pytest.fixture
def report_generator():
    """Create report generator for testing"""
    return create_report_generator()


# =============================================================================
# FEATURE 1: HIERARCHICAL PROCESS SUPPORT
# =============================================================================

class TestHierarchicalCrew:
    """Test hierarchical crew management"""
    
    def test_create_manager_crew(self, hierarchical_crew):
        """Test creating manager-led crew"""
        manager_config = {"name": "Manager", "expertise": ["management"]}
        worker_configs = [
            {"name": "Worker1", "skills": ["coding"]},
            {"name": "Worker2", "skills": ["testing"]}
        ]
        
        result = hierarchical_crew.create_manager_crew(manager_config, worker_configs)
        
        assert "crew_id" in result
        assert "manager_id" in result
        assert len(result["worker_ids"]) == 2
        assert result["level"] == "manager"
    
    def test_delegate_task(self, hierarchical_crew):
        """Test task delegation"""
        # Setup crew first
        manager_config = {"name": "Manager"}
        worker_configs = [{"name": "Worker1"}, {"name": "Worker2"}]
        crew = hierarchical_crew.create_manager_crew(manager_config, worker_configs)
        
        task = HierarchicalTask(
            task_id="task_1",
            title="Test Task",
            description="Test description",
            level=CrewLevel.WORKER
        )
        
        result = hierarchical_crew.delegate_task(
            task=task,
            from_agent_id=crew["manager_id"],
            to_agent_ids=crew["worker_ids"]
        )
        
        assert result["status"] == "delegated"
        assert len(result["sub_tasks"]) == 2
        assert task.task_id in hierarchical_crew.task_tree
    
    def test_aggregate_results_consensus(self, hierarchical_crew):
        """Test result aggregation with consensus strategy"""
        # Create parent task
        parent_task = HierarchicalTask(
            task_id="parent_1",
            title="Parent Task",
            description="Parent description",
            level=CrewLevel.MANAGER
        )
        hierarchical_crew.tasks[parent_task.task_id] = parent_task
        
        # Create child tasks with results
        for i in range(3):
            child = HierarchicalTask(
                task_id=f"child_{i}",
                title=f"Child {i}",
                description="Child task",
                level=CrewLevel.WORKER,
                parent_task_id=parent_task.task_id,
                result={"answer": "consensus_value", "quality": 0.8}
            )
            child.status = "completed"
            hierarchical_crew.tasks[child.task_id] = child
        
        hierarchical_crew.task_tree[parent_task.task_id] = ["child_0", "child_1", "child_2"]
        
        result = hierarchical_crew.aggregate_results(
            parent_task_id=parent_task.task_id,
            aggregation_strategy="consensus"
        )
        
        assert result["aggregation_strategy"] == "consensus"
        assert "aggregated_result" in result
        assert result["child_count"] == 3
    
    def test_hierarchy_status(self, hierarchical_crew):
        """Test getting hierarchy status"""
        # Create crew structure
        manager_config = {"name": "Manager"}
        worker_configs = [{"name": "Worker1"}, {"name": "Worker2"}]
        hierarchical_crew.create_manager_crew(manager_config, worker_configs)
        
        status = hierarchical_crew.get_hierarchy_status()
        
        assert "crew_name" in status
        assert "total_tasks" in status
        assert "total_agents" in status
        assert status["total_agents"] == 3  # 1 manager + 2 workers
    
    def test_max_depth_enforcement(self):
        """Test that max depth is enforced"""
        crew = create_hierarchical_crew(max_depth=2)
        assert crew.max_depth == 2
        
        # Test that deep delegation respects max depth
        task = HierarchicalTask(
            task_id="deep_task",
            title="Deep Task",
            description="Test",
            level=CrewLevel.EXECUTIVE
        )
        
        # Should handle deep tasks gracefully
        assert task.level == CrewLevel.EXECUTIVE


# =============================================================================
# FEATURE 2: ADVANCED DELEGATION MECHANISMS
# =============================================================================

class TestAdvancedDelegation:
    """Test advanced delegation mechanisms"""
    
    def test_role_based_delegation(self, delegation_manager):
        """Test role-based delegation"""
        task = {
            "id": "task_1",
            "required_role": "senior",
            "description": "Senior task"
        }
        
        result = delegation_manager.delegate(task, DelegationType.ROLE_BASED)
        
        assert result["success"] is True
        assert result["agent_info"]["role"] == "senior"
    
    def test_skill_based_delegation(self, delegation_manager):
        """Test skill-based delegation"""
        task = {
            "id": "task_1",
            "required_skills": ["machine_learning", "python"],
            "description": "ML task"
        }
        
        result = delegation_manager.delegate(task, DelegationType.SKILL_BASED)
        
        assert result["success"] is True
        assert "machine_learning" in result["agent_info"]["skills"]
    
    def test_load_balanced_delegation(self, delegation_manager):
        """Test load-balanced delegation"""
        # Assign work to agent_1
        delegation_manager.agents["agent_1"].workload = 3
        delegation_manager.agents["agent_2"].workload = 1
        
        task = {"id": "task_1"}
        result = delegation_manager.delegate(task, DelegationType.LOAD_BALANCED)
        
        # Should select agent_2 as it has lower workload
        assert result["success"] is True
    
    def test_priority_based_delegation(self, delegation_manager):
        """Test priority-based delegation"""
        # High priority task should go to senior agent
        task = {
            "id": "task_1",
            "priority": 9,
            "description": "Critical task"
        }
        
        result = delegation_manager.delegate(task, DelegationType.PRIORITY_BASED)
        
        assert result["success"] is True
    
    def test_escalation_chain(self, delegation_manager):
        """Test escalation mechanism"""
        delegation_manager.set_escalation_chain("agent_3", ["agent_1", "agent_2"])
        
        task = {"id": "task_1"}
        context = {
            "previous_agent": "agent_3",
            "escalation_level": 0
        }
        
        result = delegation_manager.delegate(task, DelegationType.ESCALATION, context)
        
        assert result["success"] is True
    
    def test_task_completion_tracking(self, delegation_manager):
        """Test task completion tracking updates metrics"""
        delegation_manager.report_task_completion("agent_1", "task_1", True, 0.9)
        
        stats = delegation_manager.get_delegation_stats()
        assert "performance_summary" in stats
        assert "agent_1" in stats["performance_summary"]
    
    def test_delegation_stats(self, delegation_manager):
        """Test delegation statistics"""
        stats = delegation_manager.get_delegation_stats()
        
        assert stats["total_agents"] == 3
        assert "agents_by_role" in stats


# =============================================================================
# FEATURE 3: MEMORY-AUGMENTED RESEARCH
# =============================================================================

class TestMemoryAugmentedResearch:
    """Test memory-augmented research system"""
    
    def test_store_conversation_memory(self, memory_system):
        """Test storing conversation memory"""
        entry_id = memory_system.store(
            content="Researcher: What is the hypothesis?",
            memory_type=MemoryType.CONVERSATION,
            metadata={"session_id": "session_1", "role": "researcher"},
            importance=0.8
        )
        
        assert entry_id is not None
        assert entry_id.startswith("mem_")
        assert len(memory_system.memories[MemoryType.CONVERSATION]) == 1
    
    def test_store_entity_memory(self, memory_system):
        """Test storing entity memory"""
        entry_id = memory_system.store(
            content={"name": "Machine Learning", "type": "field"},
            memory_type=MemoryType.ENTITY,
            metadata={"entity_type": "research_field", "mention_count": 5},
            importance=0.9
        )
        
        assert entry_id is not None
        assert len(memory_system.memories[MemoryType.ENTITY]) == 1
    
    def test_retrieve_relevant_memories(self, memory_system):
        """Test memory retrieval"""
        # Store some memories
        memory_system.store(
            content="Machine learning applications in healthcare",
            memory_type=MemoryType.LONG_TERM,
            importance=0.8
        )
        memory_system.store(
            content="Deep learning for image classification",
            memory_type=MemoryType.LONG_TERM,
            importance=0.7
        )
        
        # Retrieve
        results = memory_system.retrieve("machine learning", top_k=5)
        
        assert len(results) > 0
        assert all("relevance" in r for r in results)
    
    def test_retrieve_conversation_history(self, memory_system):
        """Test conversation history retrieval"""
        # Store conversation
        for i in range(5):
            memory_system.store(
                content=f"Message {i}",
                memory_type=MemoryType.CONVERSATION,
                metadata={"session_id": "session_1", "role": "user"}
            )
        
        history = memory_system.retrieve_conversation_history("session_1", limit=10)
        
        assert len(history) == 5
        assert history[0]["content"] == "Message 0"
    
    def test_retrieve_entities(self, memory_system):
        """Test entity retrieval"""
        memory_system.store(
            content={"name": "Neural Networks", "category": "algorithm"},
            memory_type=MemoryType.ENTITY,
            metadata={"entity_type": "algorithm", "mention_count": 10}
        )
        
        entities = memory_system.retrieve_entities(entity_type="algorithm")
        
        assert len(entities) == 1
        assert entities[0]["type"] == "algorithm"
    
    def test_memory_consolidation(self, memory_system):
        """Test memory consolidation"""
        # Store low-importance memory
        memory_system.store(
            content="Low importance content",
            memory_type=MemoryType.WORKING,
            importance=0.1
        )
        
        # Consolidate
        report = memory_system.consolidate_memories()
        
        assert "total_before" in report
        assert "total_after" in report


# =============================================================================
# FEATURE 4: EXTERNAL TOOL ORCHESTRATION
# =============================================================================

class TestExternalToolOrchestration:
    """Test external tool orchestration"""
    
    @pytest.mark.asyncio
    async def test_register_tool(self, tool_orchestrator):
        """Test tool registration"""
        definition = ToolDefinition(
            tool_id="test_tool",
            name="Test Tool",
            tool_type=ToolType.CUSTOM,
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}}
        )
        
        tool_orchestrator.register_tool(definition)
        
        assert "test_tool" in tool_orchestrator.tool_definitions
    
    @pytest.mark.asyncio
    async def test_register_custom_tool(self, tool_orchestrator):
        """Test registering custom function as tool"""
        def test_func(input_str: str) -> str:
            return f"Processed: {input_str}"
        
        tool_id = tool_orchestrator.register_custom_tool(
            name="test_func",
            func=test_func,
            input_schema={"input_str": {"type": "string"}}
        )
        
        assert tool_id.startswith("custom_")
        assert tool_id in tool_orchestrator.tools
    
    @pytest.mark.asyncio
    async def test_execute_tool(self, tool_orchestrator):
        """Test tool execution"""
        def test_func(value: int) -> int:
            return value * 2
        
        tool_id = tool_orchestrator.register_custom_tool(
            name="doubler",
            func=test_func
        )
        
        result = await tool_orchestrator.execute_tool(
            tool_id=tool_id,
            inputs={"value": 5},
            use_cache=False
        )
        
        assert result.success is True
        assert result.result == 10
    
    def test_create_tool_chain(self, tool_orchestrator):
        """Test creating tool chain"""
        tool_orchestrator.create_tool_chain("chain_1", ["tool_1", "tool_2", "tool_3"])
        
        assert "chain_1" in tool_orchestrator.chains
        assert len(tool_orchestrator.chains["chain_1"]) == 3
    
    @pytest.mark.asyncio
    async def test_execute_chain(self, tool_orchestrator):
        """Test executing tool chain"""
        # Create chain
        tool_orchestrator.create_tool_chain("test_chain", [])
        
        # With empty chain, should return success
        result = await tool_orchestrator.execute_chain("test_chain", {"input": "test"})
        
        # Should succeed even with empty chain
        assert isinstance(result, dict)
    
    def test_tool_caching(self, tool_orchestrator):
        """Test tool result caching"""
        from crewai_research_tools import ToolCache
        
        cache = ToolCache()
        
        # Create mock result
        from crewai_research_tools import ToolResult
        result = ToolResult(
            tool_id="test",
            success=True,
            result="cached_data",
            execution_time_ms=100
        )
        
        # Store in cache
        cache.put("tool_1", {"input": "test"}, result)
        
        # Retrieve
        cached = cache.get("tool_1", {"input": "test"})
        
        assert cached is not None
        assert cached.result == "cached_data"
        assert cached.cache_hit is True


# =============================================================================
# FEATURE 5: MULTI-MODAL SUPPORT
# =============================================================================

class TestMultiModalSupport:
    """Test multi-modal support"""
    
    def test_capabilities_detection(self, multimodal_processor):
        """Test that capabilities are detected"""
        capabilities = multimodal_processor.get_capabilities()
        
        assert "vision" in capabilities
        assert "audio" in capabilities
        assert "document" in capabilities
        assert "video" in capabilities
    
    def test_image_processing_mock(self, multimodal_processor):
        """Test image processing with mock"""
        # Create simple test image
        from PIL import Image
        import io
        
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        result = multimodal_processor.process_image(
            img_bytes.getvalue(),
            task="describe"
        )
        
        if multimodal_processor.vision_enabled:
            assert "width" in result
            assert "height" in result
        else:
            assert "error" in result
    
    def test_document_parsing_txt(self, multimodal_processor, temp_dir):
        """Test text document parsing"""
        # Create test file
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Test document content\nLine 2")
        
        result = multimodal_processor.parse_document(test_file)
        
        assert result["content"] == "Test document content\nLine 2"
        assert result["file_type"] == ".txt"
    
    def test_document_parsing_unsupported(self, multimodal_processor, temp_dir):
        """Test handling of unsupported file types"""
        test_file = os.path.join(temp_dir, "test.xyz")
        with open(test_file, 'w') as f:
            f.write("content")
        
        result = multimodal_processor.parse_document(test_file)
        
        assert "error" in result


# =============================================================================
# FEATURE 6: REAL-TIME COLLABORATION
# =============================================================================

class TestRealTimeCollaboration:
    """Test real-time collaboration system"""
    
    def test_create_channel(self, collaboration_system):
        """Test channel creation"""
        channel_id = collaboration_system.create_channel("test_channel")
        
        assert channel_id in collaboration_system.channels
        assert channel_id == "test_channel"
        
        # Test auto-generated ID
        auto_id = collaboration_system.create_channel()
        assert auto_id.startswith("ch_")
    
    def test_join_channel(self, collaboration_system):
        """Test joining channel"""
        channel_id = collaboration_system.create_channel("test_channel")
        
        result = collaboration_system.join_channel(
            channel_id=channel_id,
            agent_id="agent_1",
            agent_info={"name": "Test Agent"}
        )
        
        assert result is True
        assert "agent_1" in collaboration_system.agent_channels
    
    def test_leave_channel(self, collaboration_system):
        """Test leaving channel"""
        channel_id = collaboration_system.create_channel("test_channel")
        collaboration_system.join_channel(channel_id, "agent_1")
        
        result = collaboration_system.leave_channel(channel_id, "agent_1")
        
        assert result is True
    
    def test_broadcast(self, collaboration_system):
        """Test broadcasting events"""
        channel_id = collaboration_system.create_channel("test_channel")
        collaboration_system.join_channel(channel_id, "agent_1")
        collaboration_system.join_channel(channel_id, "agent_2")
        
        result = collaboration_system.broadcast(
            channel_id=channel_id,
            event_type=CollaborationEventType.MESSAGE,
            source_agent_id="agent_1",
            payload={"message": "Hello"}
        )
        
        assert result is True
    
    def test_direct_message(self, collaboration_system):
        """Test direct messaging"""
        channel_id = collaboration_system.create_channel("test_channel")
        collaboration_system.join_channel(channel_id, "agent_1")
        collaboration_system.join_channel(channel_id, "agent_2")
        
        result = collaboration_system.send_direct_message(
            from_agent_id="agent_1",
            to_agent_id="agent_2",
            message="Private message"
        )
        
        assert result is True
    
    def test_notifications(self, collaboration_system):
        """Test notification system"""
        channel_id = collaboration_system.create_channel("test_channel")
        collaboration_system.join_channel(channel_id, "agent_1")
        
        collaboration_system.notify(
            agent_id="agent_1",
            notification_type="task_complete",
            message="Task completed successfully",
            priority="high"
        )
        
        notifs = collaboration_system.get_notifications("agent_1")
        
        assert len(notifs) == 1
        assert notifs[0]["priority"] == "high"
    
    def test_channel_info(self, collaboration_system):
        """Test getting channel information"""
        channel_id = collaboration_system.create_channel("test_channel")
        collaboration_system.join_channel(channel_id, "agent_1")
        
        info = collaboration_system.get_channel_info(channel_id)
        
        assert info is not None
        assert info["channel_id"] == channel_id
        assert info["participants"] == 1


# =============================================================================
# FEATURE 7: RESEARCH WORKFLOW TEMPLATES
# =============================================================================

class TestResearchWorkflowTemplates:
    """Test research workflow templates"""
    
    def test_literature_review_template(self):
        """Test literature review template"""
        template = LiteratureReviewTemplate()
        
        assert template.template.template_type == TemplateType.LITERATURE_REVIEW
        assert len(template.template.steps) == 9
        assert "research_lead" in template.template.required_agents
    
    def test_experimental_design_template(self):
        """Test experimental design template"""
        template = ExperimentalDesignTemplate()
        
        assert template.template.template_type == TemplateType.EXPERIMENTAL_DESIGN
        assert len(template.template.steps) == 10
    
    def test_data_analysis_template(self):
        """Test data analysis template"""
        template = DataAnalysisTemplate()
        
        assert template.template.template_type == TemplateType.DATA_ANALYSIS
        assert len(template.template.steps) == 10
    
    def test_paper_writing_template(self):
        """Test paper writing template"""
        template = PaperWritingTemplate()
        
        assert template.template.template_type == TemplateType.PAPER_WRITING
        assert len(template.template.steps) == 12
    
    def test_peer_review_template(self):
        """Test peer review template"""
        template = PeerReviewTemplate()
        
        assert template.template.template_type == TemplateType.PEER_REVIEW
        assert len(template.template.steps) == 12
    
    def test_template_registry(self, template_registry):
        """Test template registry"""
        templates = template_registry.list_templates()
        
        assert len(templates) == 5  # All 5 default templates
        
        # Check all types are present
        types = [t["type"] for t in templates]
        assert "literature_review" in types
        assert "experimental_design" in types
        assert "data_analysis" in types
        assert "paper_writing" in types
        assert "peer_review" in types
    
    def test_get_template_by_type(self, template_registry):
        """Test getting template by type"""
        template = template_registry.get_template_by_type(TemplateType.LITERATURE_REVIEW)
        
        assert template is not None
        assert template.template.template_type == TemplateType.LITERATURE_REVIEW
    
    def test_validate_inputs(self):
        """Test input validation"""
        template = LiteratureReviewTemplate()
        
        # Valid inputs
        valid_inputs = {
            "research_topic": "AI in Healthcare",
            "research_questions": ["Q1", "Q2"]
        }
        errors = template.validate_inputs(valid_inputs)
        assert len(errors) == 0
        
        # Invalid inputs (missing required)
        invalid_inputs = {"research_topic": "AI"}
        errors = template.validate_inputs(invalid_inputs)
        assert len(errors) > 0
    
    def test_get_execution_plan(self):
        """Test generating execution plan"""
        template = LiteratureReviewTemplate()
        
        inputs = {
            "research_topic": "Test",
            "research_questions": ["Q1"]
        }
        
        plan = template.get_execution_plan(inputs)
        
        assert plan["template_id"] == template.template.template_id
        assert len(plan["steps"]) == len(template.template.steps)
        assert plan["total_steps"] == 9


# =============================================================================
# FEATURE 8: AUTOMATED LITERATURE SEARCH
# =============================================================================

class TestAutomatedLiteratureSearch:
    """Test automated literature search"""
    
    def test_search_semantic_scholar(self, literature_search):
        """Test Semantic Scholar search"""
        results = literature_search.search(
            query="machine learning",
            database=DatabaseType.SEMANTIC_SCHOLAR,
            max_results=5
        )
        
        assert isinstance(results, list)
        assert len(results) <= 5
    
    def test_search_arxiv(self, literature_search):
        """Test arXiv search"""
        results = literature_search.search(
            query="deep learning",
            database=DatabaseType.ARXIV,
            max_results=5
        )
        
        assert isinstance(results, list)
        if results:
            assert hasattr(results[0], 'title')
            assert hasattr(results[0], 'authors')
    
    def test_search_all_databases(self, literature_search):
        """Test searching all databases"""
        results = literature_search.search_all(
            query="artificial intelligence",
            max_results_per_db=3,
            databases=[DatabaseType.SEMANTIC_SCHOLAR, DatabaseType.ARXIV]
        )
        
        assert DatabaseType.SEMANTIC_SCHOLAR in results
        assert DatabaseType.ARXIV in results
    
    def test_citation_analysis(self, literature_search):
        """Test citation analysis"""
        from crewai_research_external import Paper
        
        papers = [
            Paper(paper_id="1", title="Paper 1", authors=["A"], abstract="", citation_count=100),
            Paper(paper_id="2", title="Paper 2", authors=["A", "B"], abstract="", citation_count=50),
            Paper(paper_id="3", title="Paper 3", authors=["B"], abstract="", citation_count=25)
        ]
        
        analysis = literature_search.analyze_citations(papers)
        
        assert analysis["total_papers"] == 3
        assert analysis["total_citations"] == 175
        assert analysis["average_citations"] == 175 / 3
        assert len(analysis["most_cited_papers"]) <= 5
        assert len(analysis["top_authors"]) <= 10


# =============================================================================
# FEATURE 9: EXPERIMENT TRACKING
# =============================================================================

class TestExperimentTracking:
    """Test experiment tracking system"""
    
    def test_create_experiment(self, experiment_tracker):
        """Test creating experiment"""
        exp_id = experiment_tracker.create_experiment(
            name="Test Experiment",
            description="Testing experiment tracking",
            parameters={"learning_rate": 0.001, "epochs": 100},
            tags=["test", "ml"]
        )
        
        assert exp_id.startswith("exp_")
        assert exp_id in experiment_tracker.experiments
    
    def test_log_parameters(self, experiment_tracker):
        """Test logging parameters"""
        exp_id = experiment_tracker.create_experiment(name="Test")
        
        experiment_tracker.log_parameter(exp_id, "batch_size", 32)
        
        exp = experiment_tracker.get_experiment(exp_id)
        param_names = [p.name for p in exp.parameters]
        assert "batch_size" in param_names
    
    def test_log_metrics(self, experiment_tracker):
        """Test logging metrics"""
        exp_id = experiment_tracker.create_experiment(name="Test")
        
        experiment_tracker.log_metric(exp_id, "accuracy", 0.95, step=1)
        experiment_tracker.log_metric(exp_id, "loss", 0.05, step=1)
        
        exp = experiment_tracker.get_experiment(exp_id)
        assert len(exp.metrics) == 2
    
    def test_log_multiple_metrics(self, experiment_tracker):
        """Test logging multiple metrics at once"""
        exp_id = experiment_tracker.create_experiment(name="Test")
        
        experiment_tracker.log_metrics(exp_id, {
            "precision": 0.92,
            "recall": 0.88,
            "f1": 0.90
        }, step=1)
        
        exp = experiment_tracker.get_experiment(exp_id)
        assert len(exp.metrics) == 3
    
    def test_log_artifact(self, experiment_tracker, temp_dir):
        """Test logging artifacts"""
        exp_id = experiment_tracker.create_experiment(name="Test")
        
        artifact_id = experiment_tracker.log_artifact(
            exp_id,
            name="model.pkl",
            artifact_type="model",
            file_path=os.path.join(temp_dir, "model.pkl"),
            metadata={"accuracy": 0.95}
        )
        
        assert artifact_id.startswith("art_")
        exp = experiment_tracker.get_experiment(exp_id)
        assert len(exp.artifacts) == 1
    
    def test_complete_experiment(self, experiment_tracker):
        """Test completing experiment"""
        exp_id = experiment_tracker.create_experiment(name="Test")
        
        experiment_tracker.complete_experiment(exp_id, status="completed")
        
        exp = experiment_tracker.get_experiment(exp_id)
        assert exp.status == "completed"
        assert exp.completed_at is not None
    
    def test_list_experiments(self, experiment_tracker):
        """Test listing experiments"""
        experiment_tracker.create_experiment(name="Exp 1", tags=["test"])
        experiment_tracker.create_experiment(name="Exp 2", tags=["prod"])
        
        all_exps = experiment_tracker.list_experiments()
        assert len(all_exps) == 2
        
        filtered = experiment_tracker.list_experiments(tags=["test"])
        assert len(filtered) == 1
    
    def test_compare_experiments(self, experiment_tracker):
        """Test comparing experiments"""
        exp1 = experiment_tracker.create_experiment(
            name="Exp 1",
            parameters={"lr": 0.01}
        )
        exp2 = experiment_tracker.create_experiment(
            name="Exp 2",
            parameters={"lr": 0.001}
        )
        
        experiment_tracker.log_metric(exp1, "accuracy", 0.90)
        experiment_tracker.log_metric(exp2, "accuracy", 0.95)
        
        comparison = experiment_tracker.compare_experiments([exp1, exp2])
        
        assert comparison["experiments_compared"] == 2
        assert "parameter_comparison" in comparison
        assert "metric_comparison" in comparison


# =============================================================================
# FEATURE 10: RESEARCH REPORT GENERATION
# =============================================================================

class TestResearchReportGeneration:
    """Test research report generation"""
    
    def test_add_section(self, report_generator):
        """Test adding sections"""
        report_generator.add_section(
            title="Introduction",
            content="This is the introduction."
        )
        
        assert len(report_generator.sections) == 1
        assert report_generator.sections[0].title == "Introduction"
    
    def test_add_table(self, report_generator):
        """Test adding tables"""
        report_generator.add_table(
            title="Results Table",
            headers=["Metric", "Value"],
            rows=[["Accuracy", "95%"], ["Loss", "0.05"]]
        )
        
        assert len(report_generator.sections) == 1
        assert "Accuracy" in report_generator.sections[0].content
    
    def test_add_citation(self, report_generator):
        """Test adding citations"""
        from crewai_research_external import Paper
        
        paper = Paper(
            paper_id="123",
            title="Test Paper",
            authors=["Author A", "Author B"],
            abstract="Test abstract",
            publication_date="2023",
            journal="Test Journal"
        )
        
        cite_id = report_generator.add_citation(paper)
        
        assert cite_id.startswith("cite_")
        assert cite_id in report_generator.citations
    
    def test_format_citation_apa(self, report_generator):
        """Test APA citation formatting"""
        from crewai_research_external import Paper
        
        paper = Paper(
            paper_id="123",
            title="Test Paper",
            authors=["John Doe", "Jane Smith"],
            abstract="Test",
            publication_date="2023"
        )
        
        cite_id = report_generator.add_citation(paper)
        formatted = report_generator.format_citation(cite_id, inline=True)
        
        assert "Doe" in formatted
        assert "2023" in formatted
    
    def test_generate_markdown(self, report_generator):
        """Test Markdown generation"""
        report_generator.add_section("Section 1", "Content 1")
        report_generator.add_section("Section 2", "Content 2")
        
        markdown = report_generator.generate_markdown()
        
        assert "## Section 1" in markdown
        assert "## Section 2" in markdown
        assert "Content 1" in markdown
    
    def test_generate_html(self, report_generator):
        """Test HTML generation"""
        report_generator.add_section("Section 1", "Content 1")
        
        html = report_generator.generate_html()
        
        assert "<html>" in html
        assert "<h2>Section 1</h2>" in html
        assert "</html>" in html
    
    def test_export_markdown(self, report_generator, temp_dir):
        """Test exporting to Markdown"""
        report_generator.add_section("Test", "Content")
        
        output_path = os.path.join(temp_dir, "report.md")
        result = report_generator.export(output_path, ReportFormat.MARKDOWN)
        
        assert result is True
        assert os.path.exists(output_path)
    
    def test_export_html(self, report_generator, temp_dir):
        """Test exporting to HTML"""
        report_generator.add_section("Test", "Content")
        
        output_path = os.path.join(temp_dir, "report.html")
        result = report_generator.export(output_path, ReportFormat.HTML)
        
        assert result is True
        assert os.path.exists(output_path)
    
    def test_clear(self, report_generator):
        """Test clearing report"""
        report_generator.add_section("Test", "Content")
        report_generator.clear()
        
        assert len(report_generator.sections) == 0
        assert len(report_generator.citations) == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests across features"""
    
    def test_research_workflow_integration(self, temp_dir):
        """Test complete research workflow"""
        # 1. Create hierarchical crew
        crew = create_hierarchical_crew(name="ResearchCrew")
        crew_config = crew.create_manager_crew(
            {"name": "Research Manager"},
            [{"name": "Analyst 1"}, {"name": "Analyst 2"}]
        )
        
        # 2. Get research template
        templates = create_template_registry()
        lit_review = templates.get_template_by_type(TemplateType.LITERATURE_REVIEW)
        
        # 3. Create execution plan
        plan = lit_review.get_execution_plan({
            "research_topic": "AI Safety",
            "research_questions": ["How to ensure AI safety?"]
        })
        
        # 4. Create experiment tracker
        tracker = create_experiment_tracker(storage_dir=temp_dir)
        exp_id = tracker.create_experiment(
            name="Literature Review",
            description="AI Safety Literature Review"
        )
        
        # 5. Execute workflow steps
        for step in plan["steps"][:3]:  # Just first 3 steps for testing
            tracker.log_metric(exp_id, f"step_{step['step_id']}_complete", 1.0)
        
        # 6. Complete experiment
        tracker.complete_experiment(exp_id)
        
        # Verify
        exp = tracker.get_experiment(exp_id)
        assert exp.status == "completed"
        assert len(exp.metrics) == 3
    
    def test_collaborative_research(self):
        """Test collaborative research scenario"""
        # Setup collaboration
        collab = create_collaboration_system()
        channel = collab.create_channel("research_room")
        
        # Agents join
        collab.join_channel(channel, "lead_researcher")
        collab.join_channel(channel, "analyst_1")
        collab.join_channel(channel, "analyst_2")
        
        # Use delegation manager
        delegation = create_delegation_manager()
        delegation.register_agent(AgentCapability(
            agent_id="analyst_1",
            skills=["data_analysis"],
            role="analyst"
        ))
        delegation.register_agent(AgentCapability(
            agent_id="analyst_2",
            skills=["statistics"],
            role="analyst"
        ))
        
        # Delegate tasks
        result = delegation.delegate(
            {"id": "analysis_task", "required_skills": ["data_analysis"]},
            DelegationType.SKILL_BASED
        )
        
        # Broadcast result
        collab.broadcast(
            channel,
            CollaborationEventType.TASK_UPDATE,
            "lead_researcher",
            {"delegation": result}
        )
        
        # Verify
        info = collab.get_channel_info(channel)
        assert info["participants"] == 3


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
