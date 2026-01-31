"""
Comprehensive Tests for Advanced Knowledge Engine Features

Tests for:
- Distributed coordination
- Real-time collaboration
- ML intelligence
- Workflow automation
- Security layer
- Unified platform
"""

import asyncio
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import all modules
from distributed_coordination import (
    RaftNode, NodeState, LogEntry, LogEntryType,
    DistributedKnowledgeCoordinator, NotLeaderException
)
from realtime_collaboration import (
    PresenceManager, LockManager, OperationalTransformation,
    RealtimeCollaborationServer, CollaborationEventType, Operation
)
from ml_intelligence import (
    MLIntelligenceEngine, ContentClassifier, EntityExtractor,
    ContentSummarizer, RecommendationEngine, DuplicateDetector,
    AutoTagger
)
from workflow_automation import (
    WorkflowEngine, Workflow, Trigger, Action,
    TriggerType, ActionType, WorkflowContext
)
from security_layer import (
    SecurityManager, AccessControlManager, Permission,
    EncryptionLevel, User, AuditEvent
)


class TestRaftNode(unittest.IsolatedAsyncioTestCase):
    """Test Raft consensus implementation."""
    
    async def asyncSetUp(self):
        self.node = RaftNode(
            node_id="node-1",
            address="localhost",
            port=8001,
            peers=[("node-2", "localhost", 8002)],
            data_dir="./test_raft_data"
        )
    
    async def asyncTearDown(self):
        await self.node.stop()
    
    async def test_initial_state(self):
        """Test initial node state."""
        self.assertEqual(self.node.state, NodeState.FOLLOWER)
        self.assertEqual(self.node.node_id, "node-1")
        self.assertIsNone(self.node.current_leader)
    
    async def test_state_transitions(self):
        """Test state transitions."""
        # Start as follower
        await self.node.start()
        self.assertEqual(self.node.state, NodeState.FOLLOWER)
        
        # Test manual state change
        old_state = self.node.state
        self.node.state = NodeState.CANDIDATE
        self.assertEqual(self.node.state, NodeState.CANDIDATE)
    
    async def test_log_append(self):
        """Test log appending."""
        entry = await self.node._append_entry(
            LogEntryType.KNOWLEDGE_ADD,
            {"content": "test", "id": "item-1"}
        )
        
        self.assertEqual(entry.index, 1)
        self.assertEqual(entry.entry_type, LogEntryType.KNOWLEDGE_ADD)
        self.assertEqual(len(self.node.persistent_state.log), 1)
    
    async def test_vote_handling(self):
        """Test vote request handling."""
        request = {
            "term": 1,
            "candidate_id": "node-2",
            "last_log_index": 0,
            "last_log_term": 0
        }
        
        response = await self.node.handle_request_vote(request)
        
        self.assertIn("vote_granted", response)
        self.assertIn("term", response)


class TestPresenceManager(unittest.IsolatedAsyncioTestCase):
    """Test presence management."""
    
    async def asyncSetUp(self):
        self.manager = PresenceManager(idle_timeout=300)
        await self.manager.start()
    
    async def asyncTearDown(self):
        await self.manager.stop()
    
    async def test_user_join(self):
        """Test user joining."""
        presence = await self.manager.user_joined(
            user_id="user-1",
            user_name="Test User",
            session_id="session-1"
        )
        
        self.assertEqual(presence.user_id, "user-1")
        self.assertEqual(presence.user_name, "Test User")
        self.assertEqual(presence.status, "active")
    
    async def test_user_leave(self):
        """Test user leaving."""
        await self.manager.user_joined("user-1", "Test User", "session-1")
        presence = await self.manager.user_left("session-1")
        
        self.assertEqual(presence.user_id, "user-1")
        self.assertIsNone(await self.manager.get_presence("session-1"))
    
    async def test_current_view(self):
        """Test setting current view."""
        await self.manager.user_joined("user-1", "Test User", "session-1")
        await self.manager.set_current_view("session-1", "item-1")
        
        presence = await self.manager.get_presence("session-1")
        self.assertEqual(presence.current_view, "item-1")
        
        viewers = await self.manager.get_item_viewers("item-1")
        self.assertEqual(len(viewers), 1)


class TestLockManager(unittest.IsolatedAsyncioTestCase):
    """Test lock management."""
    
    async def asyncSetUp(self):
        self.manager = LockManager(default_ttl=60)
        await self.manager.start()
    
    async def asyncTearDown(self):
        await self.manager.stop()
    
    async def test_acquire_lock(self):
        """Test acquiring a lock."""
        success, lock = await self.manager.acquire_lock(
            "item-1", "user-1", "session-1"
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(lock)
        self.assertEqual(lock.item_id, "item-1")
    
    async def test_lock_conflict(self):
        """Test lock conflicts."""
        await self.manager.acquire_lock("item-1", "user-1", "session-1")
        
        # Another user tries to acquire
        success, lock = await self.manager.acquire_lock(
            "item-1", "user-2", "session-2"
        )
        
        self.assertFalse(success)
        self.assertIsNotNone(lock)  # Returns existing lock
    
    async def test_release_lock(self):
        """Test releasing a lock."""
        await self.manager.acquire_lock("item-1", "user-1", "session-1")
        
        success = await self.manager.release_lock("item-1", "session-1")
        self.assertTrue(success)
        
        # Lock should be gone
        lock = await self.manager.get_lock("item-1")
        self.assertIsNone(lock)


class TestOperationalTransformation(unittest.TestCase):
    """Test operational transformation."""
    
    def setUp(self):
        self.ot = OperationalTransformation()
    
    def test_transform_insert_insert(self):
        """Test transforming two insert operations."""
        op1 = Operation(
            operation_id="op1",
            user_id="user-1",
            item_id="doc-1",
            operation_type="insert",
            position=5,
            content="hello"
        )
        
        op2 = Operation(
            operation_id="op2",
            user_id="user-2",
            item_id="doc-1",
            operation_type="insert",
            position=5,
            content="world"
        )
        
        transformed1, transformed2 = self.ot.transform(op1, op2)
        
        # One of them should shift
        self.assertTrue(
            transformed1.position != 5 or transformed2.position != 5
        )


class TestMLIntelligence(unittest.TestCase):
    """Test ML intelligence features."""
    
    def test_content_classifier(self):
        """Test content classification."""
        classifier = ContentClassifier()
        
        result = classifier.classify(
            "Python is a programming language used for web development and data science",
            title="Python Programming"
        )
        
        self.assertIn(result.category, classifier.categories.keys())
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_entity_extractor(self):
        """Test entity extraction."""
        extractor = EntityExtractor()
        
        content = "Contact john.doe@example.com or visit https://example.com for Python 3.9 tutorials"
        entities = extractor.extract(content)
        
        entity_types = {e.entity_type for e in entities}
        # Should detect email, url, technology
        self.assertTrue(len(entities) > 0)
    
    def test_content_summarizer(self):
        """Test content summarization."""
        summarizer = ContentSummarizer()
        
        content = """
        Python is a high-level programming language. It was created by Guido van Rossum 
        and first released in 1991. Python is known for its simple syntax and readability. 
        It supports multiple programming paradigms. Python is widely used for web development, 
        data analysis, artificial intelligence, and scientific computing.
        """
        
        summary = summarizer.summarize(content, num_sentences=2)
        
        self.assertLess(len(summary), len(content))
        self.assertGreater(len(summary), 0)
    
    def test_recommendation_engine(self):
        """Test recommendation engine."""
        engine = RecommendationEngine()
        
        # Add item embeddings
        engine.add_item_embedding("item-1", [1.0, 0.0, 0.0])
        engine.add_item_embedding("item-2", [0.9, 0.1, 0.0])
        engine.add_item_embedding("item-3", [0.0, 1.0, 0.0])
        
        # Get similar items
        recommendations = engine.recommend_similar("item-1", num_recommendations=2)
        
        self.assertTrue(len(recommendations) > 0)
        # item-2 should be most similar to item-1
        if recommendations:
            self.assertEqual(recommendations[0].item_id, "item-2")
    
    def test_duplicate_detector(self):
        """Test duplicate detection."""
        detector = DuplicateDetector(similarity_threshold=0.8)
        
        content1 = "Python is a programming language for web development"
        detector.add_content("item-1", content1)
        
        # Exact duplicate
        is_dup, dup_id, similarity = detector.check_duplicate(content1)
        self.assertTrue(is_dup)
        
        # Similar content
        content2 = "Python is a programming language for web apps"
        is_dup, dup_id, similarity = detector.check_duplicate(content2)
        self.assertTrue(similarity > 0.5)  # Should be somewhat similar
    
    def test_auto_tagger(self):
        """Test automatic tag generation."""
        tagger = AutoTagger()
        
        content = "Python programming tutorial for beginners learning web development"
        tags = tagger.generate_tags(content, max_tags=5)
        
        self.assertTrue(len(tags) > 0)
        self.assertLessEqual(len(tags), 5)


class TestWorkflowEngine(unittest.IsolatedAsyncioTestCase):
    """Test workflow automation."""
    
    async def asyncSetUp(self):
        self.engine = WorkflowEngine()
        await self.engine.start()
    
    async def asyncTearDown(self):
        await self.engine.stop()
    
    async def test_create_workflow(self):
        """Test workflow creation."""
        workflow = self.engine.create_workflow(
            name="Test Workflow",
            description="A test workflow",
            triggers=[
                Trigger(
                    trigger_id="t1",
                    trigger_type=TriggerType.KNOWLEDGE_CREATED
                )
            ],
            actions=[
                Action(
                    action_id="a1",
                    action_type=ActionType.SEND_NOTIFICATION,
                    parameters={"message": "Test"}
                )
            ]
        )
        
        self.assertIsNotNone(workflow.workflow_id)
        self.assertEqual(workflow.name, "Test Workflow")
        self.assertTrue(workflow.enabled)
    
    async def test_process_event(self):
        """Test event processing."""
        # Create a workflow
        self.engine.create_workflow(
            name="Event Test",
            description="Test event processing",
            triggers=[
                Trigger(
                    trigger_id="t1",
                    trigger_type=TriggerType.KNOWLEDGE_CREATED,
                    conditions={"type": "text"}
                )
            ],
            actions=[
                Action(
                    action_id="a1",
                    action_type=ActionType.ADD_TAGS,
                    parameters={"tags": ["auto"]}
                )
            ]
        )
        
        # Process matching event
        event = {
            "type": "knowledge_created",
            "data": {"type": "text", "id": "item-1"}
        }
        
        await self.engine.process_event(event)
        
        # Check that workflow was triggered
        stats = self.engine.get_all_stats()
        self.assertGreater(stats["total_executions"], 0)
    
    def test_workflow_stats(self):
        """Test workflow statistics."""
        workflow = self.engine.create_workflow(
            name="Stats Test",
            description="Test stats",
            triggers=[],
            actions=[]
        )
        
        stats = self.engine.get_workflow_stats(workflow.workflow_id)
        
        self.assertIn("executions", stats)
        self.assertIn("successful", stats)


class TestSecurityLayer(unittest.TestCase):
    """Test security features."""
    
    def setUp(self):
        self.security = SecurityManager(master_key="test-key-123")
    
    def test_user_creation(self):
        """Test user creation."""
        user = self.security.access_control.create_user(
            username="testuser",
            email="test@example.com",
            roles=["editor"],
            is_admin=False
        )
        
        self.assertIsNotNone(user.user_id)
        self.assertEqual(user.username, "testuser")
        self.assertTrue(Permission.READ in user.permissions)
    
    def test_access_control(self):
        """Test access control checks."""
        # Create user
        user = self.security.access_control.create_user(
            username="user1",
            email="user1@example.com"
        )
        
        # Create policy
        self.security.access_control.create_access_policy(
            item_id="item-1",
            owner_id=user.user_id,
            allowed_users={user.user_id}
        )
        
        # Grant permission
        self.security.access_control.grant_permission(
            "item-1",
            user.user_id,
            {Permission.READ, Permission.WRITE}
        )
        
        # Check permission
        has_perm, reason = self.security.access_control.check_permission(
            user.user_id, "item-1", Permission.READ
        )
        
        self.assertTrue(has_perm)
    
    def test_encryption(self):
        """Test encryption/decryption."""
        plaintext = "Sensitive data here"
        
        encrypted = self.security.encryption.encrypt(plaintext)
        decrypted = self.security.encryption.decrypt(encrypted)
        
        self.assertEqual(decrypted, plaintext)
        self.assertNotEqual(encrypted, plaintext)
    
    def test_audit_logging(self):
        """Test audit logging."""
        event = self.security.audit_logger.log_event(
            user_id="user-1",
            action="read",
            resource_type="knowledge_item",
            resource_id="item-1",
            status="success"
        )
        
        self.assertIsNotNone(event.event_id)
        self.assertEqual(event.action, "read")
        self.assertEqual(event.status, "success")
        
        # Query events
        events = self.security.audit_logger.get_events(
            user_id="user-1",
            limit=10
        )
        
        self.assertEqual(len(events), 1)


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the complete platform."""
    
    async def asyncSetUp(self):
        # Import here to avoid circular imports
        from unified_knowledge_platform import UnifiedKnowledgePlatform
        
        self.platform = UnifiedKnowledgePlatform(
            node_id="test-node",
            address="localhost",
            port=9000,
            enable_distributed=False,  # Simplify tests
            enable_collaboration=True,
            enable_ml=True,
            enable_workflows=True,
            enable_security=True
        )
        await self.platform.initialize()
    
    async def asyncTearDown(self):
        await self.platform.shutdown()
    
    async def test_add_knowledge_with_ml(self):
        """Test adding knowledge with ML analysis."""
        item, analysis = await self.platform.add_knowledge(
            content="Python is a programming language for machine learning",
            knowledge_type=KnowledgeType.TEXT,
            user_id="user-1"
        )
        
        self.assertIsNotNone(item.id)
        self.assertIn("classification", analysis)
        self.assertIn("tags", analysis)
    
    async def test_search_with_permissions(self):
        """Test search with permission checking."""
        # Create user
        user = self.platform.create_user(
            username="searchuser",
            email="search@example.com"
        )
        
        # Add knowledge
        await self.platform.add_knowledge(
            content="Searchable content",
            knowledge_type=KnowledgeType.TEXT,
            user_id=user.user_id
        )
        
        # Search
        results = await self.platform.search(
            query="searchable",
            user_id=user.user_id
        )
        
        self.assertIsInstance(results, list)
    
    async def test_platform_stats(self):
        """Test platform statistics."""
        stats = self.platform.get_platform_stats()
        
        self.assertIn("node_id", stats)
        self.assertIn("components", stats)
        self.assertIn("knowledge_engine", stats)
        
        # Check component flags
        components = stats["components"]
        self.assertTrue(components["ml"])
        self.assertTrue(components["workflows"])
        self.assertTrue(components["security"])
        self.assertFalse(components["distributed"])  # Disabled in tests
    
    def test_health_check(self):
        """Test health check."""
        health = self.platform.health_check()
        
        self.assertEqual(health["status"], "healthy")
        self.assertIn("components", health)
        self.assertTrue(all(
            c["status"] == "healthy" 
            for c in health["components"].values()
        ))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRaftNode))
    suite.addTests(loader.loadTestsFromTestCase(TestPresenceManager))
    suite.addTests(loader.loadTestsFromTestCase(TestLockManager))
    suite.addTests(loader.loadTestsFromTestCase(TestOperationalTransformation))
    suite.addTests(loader.loadTestsFromTestCase(TestMLIntelligence))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
