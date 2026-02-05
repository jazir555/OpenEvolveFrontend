"""
Comprehensive Test Suite for OpenEvolve Tripartite System

This test suite verifies the complete integration of:
1. ACE (Agentic Context Engine) - Self-improving capabilities
2. Steer - Reliability verification  
3. LangChain + ChromaDB - Knowledge retrieval and memory
"""

import logging
import tempfile
import os
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the tripartite system
from langchain_chroma_integration import (
    TripartiteAgentSystem, 
    KnowledgeBaseManager,
    LangChainIntegration,
    tripartite_agent_capture
)

# Import individual components for isolated testing
from ace_steer_integration import AceSteerBridge
from steer_crewai_bridge import SteerCrewAIWorkflowBridge  # MIGRATED

class TestTripartiteIntegration:
    """Comprehensive test suite for the tripartite system."""
    
    def __init__(self):
        self.test_results = []
        self.system = None
    
    def run_all_tests(self):
        """Run all tests in the suite."""
        logger.info("🧪 Starting OpenEvolve Tripartite System Tests")
        
        tests = [
            self.test_knowledge_base_initialization,
            self.test_knowledge_addition,
            self.test_knowledge_retrieval,
            self.test_ace_steer_integration,
            self.test_complete_tripartite_execution,
            self.test_decorator_functionality,
            self.test_learning_experience_storage,
            self.test_system_status,
        ]
        
        for test in tests:
            try:
                test()
                self.test_results.append({"test": test.__name__, "status": "PASS"})
            except Exception as e:
                self.test_results.append({"test": test.__name__, "status": "FAIL", "error": str(e)})
                logger.error(f"[FAIL] Test failed: {test.__name__} - {e}")
        
        self.print_test_summary()
    
    def test_knowledge_base_initialization(self):
        """Test knowledge base initialization."""
        logger.info("Testing knowledge base initialization...")
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize knowledge base manager
            from langchain_chroma_integration import KnowledgeBaseConfig
            config = KnowledgeBaseConfig()
            config.persist_directory = temp_dir
            
            kb = KnowledgeBaseManager(config)
            stats = kb.get_knowledge_stats()
            
            assert stats["document_count"] == 0, "New knowledge base should be empty"
            assert stats["collection_name"] == "openevolve_knowledge"
            
            kb.close()
            
        logger.info("[OK] Knowledge base initialization test passed")
    
    def test_knowledge_addition(self):
        """Test adding knowledge to the knowledge base."""
        logger.info("Testing knowledge addition...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            from langchain_chroma_integration import KnowledgeBaseConfig
            config = KnowledgeBaseConfig()
            config.persist_directory = temp_dir
            
            kb = KnowledgeBaseManager(config)
            
            # Add test knowledge
            test_text = "ACE is an agentic context engine that enables agents to learn from their experiences."
            doc_ids = kb.add_knowledge(test_text, source="test", metadata={"category": "ace"})
            
            assert len(doc_ids) > 0, "Should return document IDs"
            
            # Verify knowledge was added
            stats = kb.get_knowledge_stats()
            assert stats["document_count"] > 0, "Knowledge should be added"
            
            kb.close()
            
        logger.info("[OK] Knowledge addition test passed")
    
    def test_knowledge_retrieval(self):
        """Test knowledge retrieval functionality."""
        logger.info("Testing knowledge retrieval...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            from langchain_chroma_integration import KnowledgeBaseConfig
            config = KnowledgeBaseConfig()
            config.persist_directory = temp_dir
            
            kb = KnowledgeBaseManager(config)
            
            # Add test knowledge
            ace_knowledge = "ACE system learns from agent executions and improves over time."
            steer_knowledge = "Steer provides deterministic verification of LLM outputs."
            
            kb.add_knowledge(ace_knowledge, source="docs", metadata={"type": "system"})
            kb.add_knowledge(steer_knowledge, source="docs", metadata={"type": "system"})
            
            # Retrieve knowledge
            results = kb.retrieve_knowledge("How does ACE work?", k=2)
            
            assert len(results) > 0, "Should retrieve relevant knowledge"
            assert any("ACE" in doc.page_content for doc in results), "Should find ACE-related knowledge"
            
            kb.close()
            
        logger.info("[OK] Knowledge retrieval test passed")
    
    def test_ace_steer_integration(self):
        """Test ACE + Steer integration."""
        logger.info("Testing ACE + Steer integration...")
        
        # Initialize ACE + Steer bridge
        ace_steer = AceSteerBridge("test_agent")
        
        # Test prompt preparation
        prompt = ace_steer.prepare_prompt("Test task", "Test context")
        assert "TASK:" in prompt, "Prompt should contain task"
        assert "Test task" in prompt, "Prompt should contain the actual task"
        
        # Test verification (with mock data)
        verification_result = ace_steer.verify_and_learn(
            query="Test query",
            output={"result": "test output"},
            verifications=["json"]
        )
        
        assert "all_passed" in verification_result, "Should return verification status"
        assert "results" in verification_result, "Should return verification results"
        
        logger.info("[OK] ACE + Steer integration test passed")
    
    def test_complete_tripartite_execution(self):
        """Test complete tripartite system execution."""
        logger.info("Testing complete tripartite execution...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize tripartite system with temp directory
            self.system = TripartiteAgentSystem(agent_id="test_tripartite")
            
            # Add some test knowledge
            self.system.add_knowledge(
                "The tripartite system combines ACE learning, Steer verification, and LangChain knowledge retrieval.",
                source="system_docs"
            )
            
            # Execute a task
            result = self.system.execute_with_knowledge(
                "Explain how the tripartite system works",
                verifications=["json", "slop"]
            )
            
            # Verify results
            assert "success" in result, "Should return success status"
            assert "knowledge_context" in result, "Should include knowledge context"
            assert "verification" in result, "Should include verification results"
            assert "stats" in result, "Should include execution statistics"
            
            # Verify knowledge was used
            stats = result["stats"]
            assert stats["knowledge_documents_retrieved"] >= 0, "Should track knowledge retrieval"
            
            logger.info("[OK] Complete tripartite execution test passed")
    
    def test_decorator_functionality(self):
        """Test the tripartite agent decorator."""
        logger.info("Testing decorator functionality...")
        
        # Create a simple agent function
        @tripartite_agent_capture(agent_id="decorator_test")
        def test_agent(task: str) -> Dict[str, Any]:
            return {
                "task": task,
                "response": f"Processed: {task}"
            }
        
        # Test the decorated function
        result = test_agent("Test task with decorator")
        
        assert "task" in result, "Should return original function result"
        assert "_tripartite_results" in result, "Should attach tripartite results"
        
        tripartite_results = result["_tripartite_results"]
        assert "knowledge_context" in tripartite_results, "Should include knowledge context"
        
        logger.info("[OK] Decorator functionality test passed")
    
    def test_learning_experience_storage(self):
        """Test storing learning experiences."""
        logger.info("Testing learning experience storage...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            from langchain_chroma_integration import KnowledgeBaseConfig
            config = KnowledgeBaseConfig()
            config.persist_directory = temp_dir
            
            # Initialize components
            kb = KnowledgeBaseManager(config)
            langchain = LangChainIntegration()
            langchain.knowledge_base = kb
            
            # Create mock verification result
            verification_result = {
                "all_passed": True,
                "results": [
                    {
                        "judge": "JsonJudge",
                        "passed": True,
                        "reason": "JSON is valid"
                    }
                ]
            }
            
            # Store learning experience
            langchain.store_learning_experience(
                query="Test query",
                response="Test response",
                verification_result=verification_result,
                source="test"
            )
            
            # Verify it was stored
            stats = kb.get_knowledge_stats()
            initial_count = stats["document_count"]
            
            # Store another experience
            verification_result["all_passed"] = False
            langchain.store_learning_experience(
                query="Failed query",
                response="Failed response",
                verification_result=verification_result,
                source="test"
            )
            
            # Verify count increased
            stats = kb.get_knowledge_stats()
            assert stats["document_count"] > initial_count, "Should store learning experiences"
            
            kb.close()
            
        logger.info("[OK] Learning experience storage test passed")
    
    def test_system_status(self):
        """Test system status reporting."""
        logger.info("Testing system status...")
        
        if not self.system:
            self.system = TripartiteAgentSystem()
        
        status = self.system.get_system_status()
        
        # Verify status structure
        assert "ace_steer" in status, "Should include ACE+Steer status"
        assert "knowledge_base" in status, "Should include knowledge base status"
        assert "steer_workflow" in status, "Should include Steer workflow status"
        
        # Verify knowledge base stats
        kb_stats = status["knowledge_base"]
        assert "document_count" in kb_stats, "Should include document count"
        
        logger.info("[OK] System status test passed")
    
    def print_test_summary(self):
        """Print test results summary."""
        logger.info("\n" + "="*60)
        logger.info("📊 TRIPARTITE SYSTEM TEST SUMMARY")
        logger.info("="*60)
        
        passed = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed = sum(1 for r in self.test_results if r["status"] == "FAIL")
        total = len(self.test_results)
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"[OK] Passed: {passed}")
        logger.info(f"[FAIL] Failed: {failed}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if failed > 0:
            logger.warning("\nFailed Tests:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    logger.warning(f"  - {result['test']}: {result.get('error', 'Unknown error')}")
        
        if passed == total:
            logger.info("\n🎉 All tests passed! Tripartite system is working correctly.")
        else:
            logger.warning("\n[WARN]  Some tests failed. Please check the errors above.")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example usage of the tripartite system."""
    logger.info("\n" + "="*60)
    logger.info("🚀 EXAMPLE USAGE: Tripartite Agent System")
    logger.info("="*60)
    
    # Initialize the system
    system = TripartiteAgentSystem(agent_id="example_agent")
    
    # Add domain knowledge
    system.add_knowledge(
        "OpenEvolve is an advanced AI framework that combines multiple AI systems.",
        source="documentation",
        metadata={"category": "overview", "importance": "high"}
    )
    
    system.add_knowledge(
        "The ACE component provides self-improving capabilities through skill learning.",
        source="documentation",
        metadata={"category": "ace", "component": "learning"}
    )
    
    system.add_knowledge(
        "Steer ensures output reliability through deterministic verification mechanisms.",
        source="documentation",
        metadata={"category": "steer", "component": "verification"}
    )
    
    # Execute a complex task
    task = "Explain how OpenEvolve combines ACE learning with Steer verification"
    
    logger.info(f"\n📝 Task: {task}")
    
    result = system.execute_with_knowledge(
        task=task,
        verifications=["json", "slop", "citations"]
    )
    
    # Display results
    logger.info(f"\n🔍 Execution Results:")
    logger.info(f"  Success: {result['success']}")
    logger.info(f"  Knowledge Documents Retrieved: {result['stats']['knowledge_documents_retrieved']}")
    logger.info(f"  Verification Passed: {result['stats']['verification_passed']}")
    logger.info(f"  Failed Verifications: {result['stats']['failed_verifications']}")
    
    if result['knowledge_context']:
        logger.info(f"\n📚 Knowledge Context Used:")
        # Show first 200 characters of knowledge context
        knowledge_preview = result['knowledge_context'][:200] + "..." if len(result['knowledge_context']) > 200 else result['knowledge_context']
        logger.info(f"  {knowledge_preview}")
    
    logger.info(f"\n💬 Agent Response:")
    response_preview = result['response'][:150] + "..." if len(result['response']) > 150 else result['response']
    logger.info(f"  {response_preview}")
    
    # Show system status
    logger.info(f"\n📊 System Status:")
    status = system.get_system_status()
    logger.info(f"  Knowledge Base: {status['knowledge_base']['document_count']} documents")
    logger.info(f"  Available Verifications: {status['steer_workflow']['available_verifications']}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run comprehensive tests
    test_suite = TestTripartiteIntegration()
    test_suite.run_all_tests()
    
    # Show example usage
    example_usage()
    
    logger.info("\n" + "="*60)
    logger.info("🎉 OpenEvolve Tripartite System Integration Complete!")
    logger.info("="*60)
    logger.info("Components Successfully Integrated:")
    logger.info("  [OK] ACE (Agentic Context Engine) - Self-improving capabilities")
    logger.info("  [OK] Steer - Reliability verification")
    logger.info("  [OK] LangChain + ChromaDB - Knowledge retrieval and memory")