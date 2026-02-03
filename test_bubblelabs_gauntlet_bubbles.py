"""
Test suite for bubblelabs_gauntlet_bubbles module.

Tests the creation and management of BubbleLab workflow bubbles for gauntlet operations.
"""

import unittest
import json
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bubblelabs_gauntlet_bubbles import (
    GauntletBubbleConfig,
    GauntletRoundBubbleConfig,
    GauntletValidationBubbleConfig,
    create_gauntlet_execution_bubble,
    create_gauntlet_round_bubble,
    create_gauntlet_validation_bubble,
    create_gauntlet_result_bubble,
    create_red_team_bubble,
    create_blue_team_bubble,
    create_gold_team_bubble,
    create_loongeval_bubble,
    create_bubble_edge,
    create_conditional_edge,
    create_feedback_edge,
    create_gauntlet_workflow_definition,
    create_3_round_gauntlet_workflow,
    update_bubble_status,
    add_bubble_result,
    serialize_bubble,
    serialize_workflow,
    export_workflow_to_json,
    create_simple_gauntlet_bubble,
    GAUNTLET_NODE_POSITIONS,
    GAUNTLET_NODE_COLORS,
    GAUNTLET_NODE_ICONS,
)


class TestGauntletBubbleConfig(unittest.TestCase):
    """Test GauntletBubbleConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GauntletBubbleConfig(
            gauntlet_name="Test Gauntlet",
            gauntlet_type="red_team",
            team_name="Test Team"
        )
        self.assertEqual(config.gauntlet_name, "Test Gauntlet")
        self.assertEqual(config.gauntlet_type, "red_team")
        self.assertEqual(config.team_name, "Test Team")
        self.assertEqual(config.timeout_seconds, 300)
        self.assertEqual(config.retry_count, 3)
        self.assertEqual(config.priority, 1)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GauntletBubbleConfig(
            gauntlet_name="Custom Gauntlet",
            gauntlet_type="gold_team",
            team_name="Gold Team",
            description="Custom description",
            timeout_seconds=600,
            retry_count=5,
            priority=2
        )
        self.assertEqual(config.timeout_seconds, 600)
        self.assertEqual(config.retry_count, 5)
        self.assertEqual(config.priority, 2)


class TestGauntletRoundBubbleConfig(unittest.TestCase):
    """Test GauntletRoundBubbleConfig dataclass."""
    
    def test_default_config(self):
        """Test default round configuration."""
        config = GauntletRoundBubbleConfig(
            round_name="Evaluation Round",
            round_order=1,
            gauntlet_types=["evaluation"]
        )
        self.assertEqual(config.round_name, "Evaluation Round")
        self.assertEqual(config.round_order, 1)
        self.assertEqual(config.pass_threshold, 0.7)
        self.assertTrue(config.requires_consensus)
    
    def test_custom_round_config(self):
        """Test custom round configuration."""
        config = GauntletRoundBubbleConfig(
            round_name="Attack Round",
            round_order=2,
            gauntlet_types=["red_team", "blue_team"],
            pass_threshold=0.8,
            requires_consensus=False,
            max_iterations=10
        )
        self.assertEqual(config.pass_threshold, 0.8)
        self.assertFalse(config.requires_consensus)
        self.assertEqual(config.max_iterations, 10)


class TestGauntletValidationBubbleConfig(unittest.TestCase):
    """Test GauntletValidationBubbleConfig dataclass."""
    
    def test_default_validation_config(self):
        """Test default validation configuration."""
        config = GauntletValidationBubbleConfig(
            validation_type="consistency",
            criteria={"relevance": 0.5, "correctness": 0.5}
        )
        self.assertEqual(config.validation_type, "consistency")
        self.assertEqual(config.required_score, 0.8)
        self.assertEqual(config.weight, 1.0)
    
    def test_custom_validation_config(self):
        """Test custom validation configuration."""
        config = GauntletValidationBubbleConfig(
            validation_type="custom",
            criteria={"metric1": 0.3, "metric2": 0.7},
            weight=2.0,
            required_score=0.9,
            feedback_mode="summary"
        )
        self.assertEqual(config.weight, 2.0)
        self.assertEqual(config.required_score, 0.9)
        self.assertEqual(config.feedback_mode, "summary")


class TestGauntletExecutionBubble(unittest.TestCase):
    """Test gauntlet execution bubble creation."""
    
    def test_create_execution_bubble(self):
        """Test basic execution bubble creation."""
        config = GauntletBubbleConfig(
            gauntlet_name="Test Gauntlet",
            gauntlet_type="red_team",
            team_name="Test Team",
            description="Test description"
        )
        bubble = create_gauntlet_execution_bubble(config)
        
        self.assertIn("id", bubble)
        self.assertEqual(bubble["type"], "gauntlet_execution")
        self.assertIn("position", bubble)
        self.assertIn("data", bubble)
        self.assertEqual(bubble["data"]["label"], "🛡️ Test Gauntlet")
        self.assertEqual(bubble["data"]["gauntlet_type"], "red_team")
        self.assertEqual(bubble["data"]["team_name"], "Test Team")
        self.assertEqual(bubble["data"]["status"], "pending")
    
    def test_custom_position(self):
        """Test bubble with custom position."""
        config = GauntletBubbleConfig(
            gauntlet_name="Position Test",
            gauntlet_type="blue_team",
            team_name="Team"
        )
        custom_pos = {"x": 100, "y": 200}
        bubble = create_gauntlet_execution_bubble(config, custom_pos)
        
        self.assertEqual(bubble["position"], custom_pos)


class TestGauntletRoundBubble(unittest.TestCase):
    """Test gauntlet round bubble creation."""
    
    def test_create_round_bubble(self):
        """Test round bubble creation."""
        config = GauntletRoundBubbleConfig(
            round_name="First Round",
            round_order=1,
            gauntlet_types=["evaluation"]
        )
        bubble = create_gauntlet_round_bubble(config)
        
        self.assertEqual(bubble["type"], "gauntlet_round")
        self.assertIn("Round 1", bubble["data"]["label"])
        self.assertEqual(bubble["data"]["round_order"], 1)
        self.assertEqual(bubble["data"]["gauntlet_types"], ["evaluation"])
    
    def test_multi_gauntlet_round(self):
        """Test round with multiple gauntlet types."""
        config = GauntletRoundBubbleConfig(
            round_name="Attack Round",
            round_order=2,
            gauntlet_types=["red_team", "blue_team"]
        )
        bubble = create_gauntlet_round_bubble(config)
        
        self.assertEqual(bubble["data"]["gauntlet_types"], ["red_team", "blue_team"])


class TestGauntletValidationBubble(unittest.TestCase):
    """Test gauntlet validation bubble creation."""
    
    def test_create_validation_bubble(self):
        """Test validation bubble creation."""
        config = GauntletValidationBubbleConfig(
            validation_type="loongeval",
            criteria={"relevance": 0.3, "correctness": 0.4, "completeness": 0.3}
        )
        bubble = create_gauntlet_validation_bubble(config)
        
        self.assertEqual(bubble["type"], "gauntlet_validation")
        self.assertEqual(bubble["data"]["validation_type"], "loongeval")
        self.assertEqual(bubble["data"]["criteria"]["relevance"], 0.3)


class TestGauntletResultBubble(unittest.TestCase):
    """Test gauntlet result bubble creation."""
    
    def test_default_result_bubble(self):
        """Test default result bubble."""
        bubble = create_gauntlet_result_bubble("Test Gauntlet")
        
        self.assertEqual(bubble["type"], "gauntlet_result")
        self.assertEqual(bubble["data"]["gauntlet_name"], "Test Gauntlet")
        self.assertEqual(bubble["data"]["status"], "pending")
    
    def test_passed_result_bubble(self):
        """Test passed result bubble."""
        bubble = create_gauntlet_result_bubble("Test Gauntlet", "passed")
        
        self.assertEqual(bubble["data"]["status"], "passed")
        self.assertEqual(bubble["data"]["node_color"], "#00B894")
    
    def test_failed_result_bubble(self):
        """Test failed result bubble."""
        bubble = create_gauntlet_result_bubble("Test Gauntlet", "failed")
        
        self.assertEqual(bubble["data"]["status"], "failed")
        self.assertEqual(bubble["data"]["node_color"], "#FF7675")


class TestTeamBubbles(unittest.TestCase):
    """Test team-specific bubble creation."""
    
    def test_red_team_bubble(self):
        """Test Red Team bubble creation."""
        bubble = create_red_team_bubble(
            team_name="Security Team",
            attack_modes=["injection", "overflow"]
        )
        
        self.assertEqual(bubble["data"]["gauntlet_type"], "red_team")
        self.assertEqual(bubble["data"]["team_name"], "Security Team")
        self.assertEqual(bubble["data"]["parameters"]["attack_modes"], ["injection", "overflow"])
    
    def test_blue_team_bubble(self):
        """Test Blue Team bubble creation."""
        bubble = create_blue_team_bubble(
            team_name="Dev Team",
            fix_types=["correctness", "performance"]
        )
        
        self.assertEqual(bubble["data"]["gauntlet_type"], "blue_team")
        self.assertEqual(bubble["data"]["parameters"]["fix_types"], ["correctness", "performance"])
    
    def test_gold_team_bubble(self):
        """Test Gold Team bubble creation."""
        bubble = create_gold_team_bubble(
            team_name="QA Team",
            verification_modes=["consistency", "completeness"]
        )
        
        self.assertEqual(bubble["data"]["gauntlet_type"], "gold_team")
        self.assertEqual(bubble["data"]["parameters"]["verification_modes"], ["consistency", "completeness"])
    
    def test_loongeval_bubble(self):
        """Test LoongFlow evaluation bubble."""
        criteria = {"accuracy": 0.5, "relevance": 0.5}
        bubble = create_loongeval_bubble(evaluation_criteria=criteria)
        
        self.assertEqual(bubble["data"]["validation_type"], "loongeval")
        self.assertEqual(bubble["data"]["criteria"]["accuracy"], 0.5)


class TestBubbleEdges(unittest.TestCase):
    """Test bubble edge creation."""
    
    def test_default_edge(self):
        """Test default edge creation."""
        edge = create_bubble_edge("node1", "node2")
        
        self.assertIn("id", edge)
        self.assertEqual(edge["source"], "node1")
        self.assertEqual(edge["target"], "node2")
        self.assertEqual(edge["sourceHandle"], "output")
        self.assertEqual(edge["targetHandle"], "input")
        self.assertEqual(edge["type"], "default")
    
    def test_conditional_edge(self):
        """Test conditional edge creation."""
        edge = create_conditional_edge("node1", "node2", "score > 0.7")
        
        self.assertEqual(edge["type"], "conditional")
        self.assertEqual(edge["label"], "score > 0.7")
    
    def test_feedback_edge(self):
        """Test feedback edge creation."""
        edge = create_feedback_edge("node1", "node2")
        
        self.assertEqual(edge["type"], "feedback")
        self.assertEqual(edge["sourceHandle"], "feedback")
        self.assertEqual(edge["targetHandle"], "input")


class TestGauntletWorkflow(unittest.TestCase):
    """Test complete gauntlet workflow creation."""
    
    def test_create_gauntlet_workflow(self):
        """Test complete workflow creation."""
        workflow = create_gauntlet_workflow_definition(
            workflow_name="Test Workflow",
            problem_statement="Test problem statement",
            gauntlet_config={
                "attack_modes": ["mode1", "mode2"],
                "include_blue_team": True,
                "max_iterations": 3
            },
            team_config={
                "red_team": "Red Team",
                "blue_team": "Blue Team",
                "gold_team": "Gold Team"
            }
        )
        
        self.assertIn("id", workflow)
        self.assertEqual(workflow["name"], "Test Workflow")
        self.assertIn("nodes", workflow)
        self.assertIn("edges", workflow)
        self.assertIn("metadata", workflow)
        
        # Check node count (input + loongeval + red + blue + gold + result = 6)
        self.assertEqual(len(workflow["nodes"]), 6)
        
        # Check edge count
        self.assertGreater(len(workflow["edges"]), 0)
    
    def test_3_round_gauntlet_workflow(self):
        """Test 3-round gauntlet workflow creation."""
        workflow = create_3_round_gauntlet_workflow(
            problem_statement="Design a REST API",
            gauntlet_name="API Gauntlet"
        )
        
        self.assertIn("nodes", workflow)
        self.assertIn("edges", workflow)
        self.assertEqual(len(workflow["nodes"]), 6)  # input + 4 team bubbles + result


class TestBubbleUpdates(unittest.TestCase):
    """Test bubble update functions."""
    
    def test_update_bubble_status(self):
        """Test status update function."""
        config = GauntletBubbleConfig(
            gauntlet_name="Test",
            gauntlet_type="red_team",
            team_name="Team"
        )
        bubble = create_gauntlet_execution_bubble(config)
        
        updated = update_bubble_status(bubble, "running")
        
        self.assertEqual(updated["data"]["status"], "running")
        self.assertEqual(updated["data"]["node_color"], "#74B9FF")
    
    def test_add_bubble_result(self):
        """Test result addition function."""
        config = GauntletBubbleConfig(
            gauntlet_name="Test",
            gauntlet_type="red_team",
            team_name="Team"
        )
        bubble = create_gauntlet_execution_bubble(config)
        
        updated = add_bubble_result(
            bubble,
            score=0.85,
            feedback="Good result",
            improvements=["Improve X", "Fix Y"]
        )
        
        self.assertEqual(updated["data"]["score"], 0.85)
        self.assertEqual(updated["data"]["feedback"], "Good result")
        self.assertEqual(updated["data"]["improvements"], ["Improve X", "Fix Y"])
        self.assertEqual(updated["data"]["status"], "partial")


class TestSerialization(unittest.TestCase):
    """Test serialization functions."""
    
    def test_serialize_bubble(self):
        """Test bubble serialization."""
        config = GauntletBubbleConfig(
            gauntlet_name="Test",
            gauntlet_type="red_team",
            team_name="Team"
        )
        bubble = create_gauntlet_execution_bubble(config)
        
        serialized = serialize_bubble(bubble)
        parsed = json.loads(serialized)
        
        self.assertEqual(parsed["id"], bubble["id"])
        self.assertEqual(parsed["type"], bubble["type"])
    
    def test_serialize_workflow(self):
        """Test workflow serialization."""
        workflow = create_3_round_gauntlet_workflow(
            problem_statement="Test",
            gauntlet_name="Test Gauntlet"
        )
        
        serialized = serialize_workflow(workflow)
        parsed = json.loads(serialized)
        
        self.assertEqual(parsed["name"], workflow["name"])
        self.assertEqual(len(parsed["nodes"]), len(workflow["nodes"]))
    
    def test_export_workflow_to_json(self):
        """Test workflow export to file."""
        workflow = create_3_round_gauntlet_workflow(
            problem_statement="Test",
            gauntlet_name="Test Gauntlet"
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            result = export_workflow_to_json(workflow, temp_path)
            self.assertTrue(result)
            
            # Verify file was created and contains valid JSON
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            self.assertEqual(loaded["name"], workflow["name"])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestSimpleBubble(unittest.TestCase):
    """Test simple bubble creation convenience function."""
    
    def test_simple_gauntlet_bubble(self):
        """Test simple bubble creation."""
        bubble = create_simple_gauntlet_bubble(
            gauntlet_type="gold_team",
            label="Final Verification",
            team="QA Team"
        )
        
        self.assertEqual(bubble["data"]["gauntlet_type"], "gold_team")
        self.assertIn("Final Verification", bubble["data"]["label"])  # Label includes icon
        self.assertEqual(bubble["data"]["team_name"], "QA Team")


class TestConstants(unittest.TestCase):
    """Test module constants."""
    
    def test_node_positions(self):
        """Test node position constants."""
        self.assertIn("start", GAUNTLET_NODE_POSITIONS)
        self.assertIn("evaluation", GAUNTLET_NODE_POSITIONS)
        self.assertIn("red_team", GAUNTLET_NODE_POSITIONS)
        self.assertIn("gold_team", GAUNTLET_NODE_POSITIONS)
        self.assertIn("result", GAUNTLET_NODE_POSITIONS)
    
    def test_node_colors(self):
        """Test node color constants."""
        self.assertIn("red_team", GAUNTLET_NODE_COLORS)
        self.assertIn("blue_team", GAUNTLET_NODE_COLORS)
        self.assertIn("gold_team", GAUNTLET_NODE_COLORS)
        self.assertEqual(GAUNTLET_NODE_COLORS["red_team"], "#FF6B6B")
    
    def test_node_icons(self):
        """Test node icon constants."""
        self.assertIn("red_team", GAUNTLET_NODE_ICONS)
        self.assertIn("blue_team", GAUNTLET_NODE_ICONS)
        self.assertIn("gold_team", GAUNTLET_NODE_ICONS)
        self.assertEqual(GAUNTLET_NODE_ICONS["red_team"], "🛡️")


if __name__ == "__main__":
    unittest.main(verbosity=2)


# =============================================================================
# Tests for Flexible Workflow Builder
# =============================================================================

class TestFlexibleWorkflowBuilder(unittest.TestCase):
    """Test flexible workflow builder for arbitrary patterns."""
    
    def test_builder_basic(self):
        """Test basic workflow builder functionality."""
        from bubblelabs_gauntlet_bubbles import (
            FlexibleWorkflowBuilder,
            BubbleDefinition,
            EdgeDefinition
        )
        
        builder = FlexibleWorkflowBuilder()
        
        # Add bubbles
        builder.add_bubble(BubbleDefinition("input", "Start"))
        builder.add_bubble(BubbleDefinition("gauntlet_execution", "Process"))
        builder.add_bubble(BubbleDefinition("gauntlet_result", "End"))
        
        # Add edges
        builder.add_edge(EdgeDefinition("Start", "Process"))
        builder.add_edge(EdgeDefinition("Process", "End"))
        
        workflow = builder.build("Test Workflow")
        
        self.assertEqual(len(workflow["nodes"]), 3)
        self.assertEqual(len(workflow["edges"]), 2)
        self.assertEqual(workflow["name"], "Test Workflow")
    
    def test_create_custom_workflow(self):
        """Test creating custom workflow from pattern."""
        from bubblelabs_gauntlet_bubbles import (
            create_custom_workflow,
            WorkflowPattern,
            BubbleDefinition,
            EdgeDefinition
        )
        
        pattern = WorkflowPattern(
            name="Custom Pipeline",
            bubbles=[
                BubbleDefinition("input", "Input"),
                BubbleDefinition("gauntlet_execution", "Step 1", team_name="Team A"),
                BubbleDefinition("gauntlet_execution", "Step 2", team_name="Team B"),
                BubbleDefinition("gauntlet_result", "Output")
            ],
            edges=[
                EdgeDefinition("Input", "Step 1"),
                EdgeDefinition("Step 1", "Step 2"),
                EdgeDefinition("Step 2", "Output")
            ]
        )
        
        workflow = create_custom_workflow(pattern)
        
        self.assertEqual(len(workflow["nodes"]), 4)
        self.assertEqual(len(workflow["edges"]), 3)
    
    def test_create_sequential_workflow(self):
        """Test creating sequential workflow from labels."""
        from bubblelabs_gauntlet_bubbles import create_sequential_workflow
        
        workflow = create_sequential_workflow(
            "Sequential Pipeline",
            ["Start", "Step 1", "Step 2", "End"],
            {"Step 1": "Blue Team", "Step 2": "Gold Team"}
        )
        
        self.assertEqual(len(workflow["nodes"]), 4)
        self.assertEqual(len(workflow["edges"]), 3)
        
        # Verify team assignments
        for node in workflow["nodes"]:
            if node["data"]["label"] == "Step 1":
                self.assertEqual(node["data"]["team_name"], "Blue Team")
            elif node["data"]["label"] == "Step 2":
                self.assertEqual(node["data"]["team_name"], "Gold Team")
    
    def test_create_branching_workflow(self):
        """Test creating workflow with branching."""
        from bubblelabs_gauntlet_bubbles import create_branching_workflow
        
        workflow = create_branching_workflow(
            "Decision Workflow",
            "Start",
            [
                {"label": "Fast Path", "condition": "priority == 'high'", "team": "Express Team"},
                {"label": "Slow Path", "condition": "priority == 'low'", "team": "Standard Team"}
            ],
            "End"
        )
        
        # Should have: Start + 2 branches + End = 5 nodes
        # But since branches share edges to End, verify structure
        self.assertGreaterEqual(len(workflow["nodes"]), 4)  # At least Start + 2 branches + End
        # Should have: Start->Fast, Start->Slow, Fast->End, Slow->End = 4 edges
        self.assertEqual(len(workflow["edges"]), 4)
        
        # Verify conditional edges exist
        conditional_edges = [e for e in workflow["edges"] if e["type"] == "conditional"]
        self.assertEqual(len(conditional_edges), 2)
    
    def test_create_loop_workflow(self):
        """Test creating workflow with loops."""
        from bubblelabs_gauntlet_bubbles import create_loop_workflow
        
        workflow = create_loop_workflow(
            "Iterative Workflow",
            ["Step 1", "Step 2", "Step 3"],
            iterations=5,
            feedback_condition="needs more work"
        )
        
        # Should have: 3 body bubbles
        self.assertEqual(len(workflow["nodes"]), 3)
        # Should have: 2 sequential + 1 feedback = 3 edges
        self.assertEqual(len(workflow["edges"]), 3)
        
        # Verify feedback edge
        feedback_edges = [e for e in workflow["edges"] if e["type"] == "feedback"]
        self.assertEqual(len(feedback_edges), 1)
    
    def test_bubble_not_found_error(self):
        """Test error handling for missing bubbles."""
        from bubblelabs_gauntlet_bubbles import (
            FlexibleWorkflowBuilder, 
            BubbleDefinition,
            EdgeDefinition
        )
        
        builder = FlexibleWorkflowBuilder()
        builder.add_bubble(BubbleDefinition("input", "Start"))
        
        with self.assertRaises(ValueError):
            builder.add_edge(EdgeDefinition("Start", "NonExistent"))


class TestBubbleDefinition(unittest.TestCase):
    """Test BubbleDefinition dataclass."""
    
    def test_default_values(self):
        """Test default values for BubbleDefinition."""
        from bubblelabs_gauntlet_bubbles import BubbleDefinition
        
        bubble = BubbleDefinition(
            bubble_type="gauntlet_execution",
            label="Test Bubble"
        )
        
        self.assertEqual(bubble.bubble_type, "gauntlet_execution")
        self.assertEqual(bubble.label, "Test Bubble")
        self.assertIsNone(bubble.position)
        self.assertEqual(bubble.team_name, "")
        self.assertEqual(bubble.parameters, {})
        self.assertEqual(bubble.node_color, "#888888")
    
    def test_custom_values(self):
        """Test custom values for BubbleDefinition."""
        from bubblelabs_gauntlet_bubbles import BubbleDefinition
        
        bubble = BubbleDefinition(
            bubble_type="custom",
            label="Custom Bubble",
            position={"x": 100, "y": 200},
            team_name="Test Team",
            parameters={"key": "value"},
            node_color="#FF0000"
        )
        
        self.assertEqual(bubble.position, {"x": 100, "y": 200})
        self.assertEqual(bubble.team_name, "Test Team")
        self.assertEqual(bubble.parameters, {"key": "value"})
        self.assertEqual(bubble.node_color, "#FF0000")


class TestEdgeDefinition(unittest.TestCase):
    """Test EdgeDefinition dataclass."""
    
    def test_default_values(self):
        """Test default values for EdgeDefinition."""
        from bubblelabs_gauntlet_bubbles import EdgeDefinition
        
        edge = EdgeDefinition(
            source_label="Source",
            target_label="Target"
        )
        
        self.assertEqual(edge.source_label, "Source")
        self.assertEqual(edge.target_label, "Target")
        self.assertEqual(edge.edge_type, "default")
        self.assertEqual(edge.condition, "")
        self.assertEqual(edge.source_handle, "output")
        self.assertEqual(edge.target_handle, "input")
    
    def test_conditional_edge(self):
        """Test conditional edge definition."""
        from bubblelabs_gauntlet_bubbles import EdgeDefinition
        
        edge = EdgeDefinition(
            source_label="Step 1",
            target_label="Step 2",
            edge_type="conditional",
            condition="score > 0.7"
        )
        
        self.assertEqual(edge.edge_type, "conditional")
        self.assertEqual(edge.condition, "score > 0.7")


class TestWorkflowPattern(unittest.TestCase):
    """Test WorkflowPattern dataclass."""
    
    def test_empty_pattern(self):
        """Test empty workflow pattern."""
        from bubblelabs_gauntlet_bubbles import WorkflowPattern
        
        pattern = WorkflowPattern(name="Empty Pattern")
        
        self.assertEqual(pattern.name, "Empty Pattern")
        self.assertEqual(pattern.description, "")
        self.assertEqual(pattern.bubbles, [])
        self.assertEqual(pattern.edges, [])
    
    def test_complete_pattern(self):
        """Test complete workflow pattern."""
        from bubblelabs_gauntlet_bubbles import (
            WorkflowPattern,
            BubbleDefinition,
            EdgeDefinition
        )
        
        pattern = WorkflowPattern(
            name="Complete Pattern",
            description="A complete workflow",
            bubbles=[
                BubbleDefinition("input", "Start"),
                BubbleDefinition("gauntlet_execution", "Process")
            ],
            edges=[
                EdgeDefinition("Start", "Process")
            ],
            metadata={"version": "1.0"}
        )
        
        self.assertEqual(len(pattern.bubbles), 2)
        self.assertEqual(len(pattern.edges), 1)
        self.assertEqual(pattern.metadata, {"version": "1.0"})
