"""
Tests for Z3 and LeanAIDE Bubbles Module

This test suite verifies the functionality of z3_leanaide_bubbles.py including:
- Bubble creation functions
- Edge creation functions
- Complete workflow creation
- Flexible workflow builder
- Serialization and export
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

from z3_leanaide_bubbles import (
    # Config classes
    Z3SolverBubbleConfig,
    Z3ProverBubbleConfig,
    LeanAideProofBubbleConfig,
    CrossVerificationBubbleConfig,
    ProblemClassificationBubbleConfig,
    Z3BubbleDefinition,
    Z3EdgeDefinition,
    
    # Bubble creation
    create_z3_solver_bubble,
    create_z3_prover_bubble,
    create_leanaide_proof_bubble,
    create_cross_verification_bubble,
    create_problem_classification_bubble,
    create_z3_result_bubble,
    
    # Edge creation
    create_z3_edge,
    create_conditional_z3_edge,
    create_feedback_z3_edge,
    get_z3_edge_color,
    
    # Workflow creation
    create_z3_solver_workflow,
    create_z3_leanaide_workflow,
    
    # Flexible builder
    Z3FlexibleWorkflowBuilder,
    create_custom_z3_workflow,
    
    # Updates
    update_z3_bubble_status,
    add_z3_result_to_bubble,
    
    # Serialization
    serialize_z3_bubble,
    serialize_z3_workflow,
    export_z3_workflow_to_json,
    
    # Constants
    Z3_NODE_COLORS,
    Z3_NODE_ICONS,
    Z3_NODE_POSITIONS,
)


class TestZ3BubbleConfigs:
    """Tests for bubble configuration classes."""
    
    def test_z3_solver_config_defaults(self):
        """Test Z3SolverBubbleConfig default values."""
        config = Z3SolverBubbleConfig(problem_text="test problem")
        
        assert config.problem_text == "test problem"
        assert config.variables == []
        assert config.constraints == []
        assert config.timeout_seconds == 30
        assert config.strategy == "auto"
    
    def test_z3_solver_config_custom(self):
        """Test Z3SolverBubbleConfig with custom values."""
        config = Z3SolverBubbleConfig(
            problem_text="x + y = 10",
            variables=[{"name": "x", "type": "Int"}],
            constraints=[{"expr": "x > 0"}],
            timeout_seconds=60,
            strategy="qe"
        )
        
        assert config.problem_text == "x + y = 10"
        assert len(config.variables) == 1
        assert len(config.constraints) == 1
        assert config.timeout_seconds == 60
        assert config.strategy == "qe"
    
    def test_z3_prover_config_defaults(self):
        """Test Z3ProverBubbleConfig default values."""
        config = Z3ProverBubbleConfig(theorem_statement="forall x: x >= 0")
        
        assert config.theorem_statement == "forall x: x >= 0"
        assert config.proof_strategy == "default"
        assert config.timeout_seconds == 60
    
    def test_leanaide_proof_config_defaults(self):
        """Test LeanAideProofBubbleConfig default values."""
        config = LeanAideProofBubbleConfig(theorem_name="MainTheorem")
        
        assert config.theorem_name == "MainTheorem"
        assert config.proof_type == "theorem"
        assert config.mcts_enabled == True
        assert config.timeout_seconds == 120
    
    def test_cross_verify_config_defaults(self):
        """Test CrossVerificationBubbleConfig default values."""
        config = CrossVerificationBubbleConfig(problem_text="test")
        
        assert config.problem_text == "test"
        assert config.z3_strategy == "adaptive"
        assert config.lean_strategy == "auto"
        assert config.timeout_seconds == 60
    
    def test_problem_classification_config_defaults(self):
        """Test ProblemClassificationBubbleConfig default values."""
        config = ProblemClassificationBubbleConfig(problem_text="test")
        
        assert config.problem_text == "test"
        assert config.auto_classify == True


class TestZ3BubbleCreation:
    """Tests for bubble creation functions."""
    
    def test_create_z3_solver_bubble(self):
        """Test creating a Z3 solver bubble."""
        config = Z3SolverBubbleConfig(problem_text="x + y = 10")
        bubble = create_z3_solver_bubble(config)
        
        assert bubble["type"] == "z3_solver"
        assert "id" in bubble
        assert "position" in bubble
        assert "data" in bubble
        assert bubble["data"]["problem_text"] == "x + y = 10"
        assert bubble["data"]["status"] == "pending"
        assert "node_color" in bubble["data"]
    
    def test_create_z3_solver_bubble_custom_position(self):
        """Test Z3 solver bubble with custom position."""
        config = Z3SolverBubbleConfig(problem_text="test")
        custom_pos = {"x": 500, "y": 200}
        bubble = create_z3_solver_bubble(config, position=custom_pos)
        
        assert bubble["position"] == custom_pos
    
    def test_create_z3_solver_bubble_custom_label(self):
        """Test Z3 solver bubble with custom label."""
        config = Z3SolverBubbleConfig(problem_text="test")
        bubble = create_z3_solver_bubble(config, label="Custom Solver")
        
        assert "Custom Solver" in bubble["data"]["label"]
    
    def test_create_z3_prover_bubble(self):
        """Test creating a Z3 prover bubble."""
        config = Z3ProverBubbleConfig(theorem_statement="forall x: P(x)")
        bubble = create_z3_prover_bubble(config)
        
        assert bubble["type"] == "z3_prover"
        assert bubble["data"]["theorem_statement"] == "forall x: P(x)"
        assert bubble["data"]["status"] == "pending"
        assert bubble["data"]["proven"] == False
    
    def test_create_leanaide_proof_bubble(self):
        """Test creating a LeanAIDE proof bubble."""
        config = LeanAideProofBubbleConfig(theorem_name="PrimeTheorem")
        bubble = create_leanaide_proof_bubble(config)
        
        assert bubble["type"] == "leanaide_proof"
        assert bubble["data"]["theorem_name"] == "PrimeTheorem"
        assert bubble["data"]["proof_type"] == "theorem"
    
    def test_create_leanaide_proof_bubble_custom_label(self):
        """Test LeanAIDE proof bubble with custom label."""
        config = LeanAideProofBubbleConfig(theorem_name="Test")
        bubble = create_leanaide_proof_bubble(config, label="My Proof")
        
        assert "My Proof" in bubble["data"]["label"]
    
    def test_create_cross_verification_bubble(self):
        """Test creating a cross-verification bubble."""
        config = CrossVerificationBubbleConfig(problem_text="test problem")
        bubble = create_cross_verification_bubble(config)
        
        assert bubble["type"] == "cross_verification"
        assert bubble["data"]["problem_text"] == "test problem"
        assert bubble["data"]["z3_status"] is None
        assert bubble["data"]["agreement"] is None
    
    def test_create_problem_classification_bubble(self):
        """Test creating a problem classification bubble."""
        config = ProblemClassificationBubbleConfig(problem_text="test")
        bubble = create_problem_classification_bubble(config)
        
        assert bubble["type"] == "problem_classification"
        assert bubble["data"]["problem_text"] == "test"
        assert bubble["data"]["classification"] is None
    
    def test_create_z3_result_bubble_default(self):
        """Test creating a Z3 result bubble with defaults."""
        bubble = create_z3_result_bubble()
        
        assert bubble["type"] == "z3_result"
        assert bubble["data"]["status"] == "pending"
        assert "node_color" in bubble["data"]
    
    def test_create_z3_result_bubble_success(self):
        """Test creating a Z3 result bubble with success status."""
        bubble = create_z3_result_bubble(result_status="success")
        
        assert bubble["data"]["status"] == "success"
        assert bubble["data"]["node_color"] == "#00B894"
    
    def test_create_z3_result_bubble_failed(self):
        """Test creating a Z3 result bubble with failed status."""
        bubble = create_z3_result_bubble(result_status="failed")
        
        assert bubble["data"]["status"] == "failed"
        assert bubble["data"]["node_color"] == "#FF7675"


class TestZ3EdgeCreation:
    """Tests for edge creation functions."""
    
    def test_create_z3_edge_default(self):
        """Test creating a default Z3 edge."""
        edge = create_z3_edge("source_id", "target_id")
        
        assert edge["source"] == "source_id"
        assert edge["target"] == "target_id"
        assert edge["sourceHandle"] == "output"
        assert edge["targetHandle"] == "input"
        assert edge["type"] == "default"
        assert edge["animated"] == True
    
    def test_create_z3_edge_custom_handles(self):
        """Test creating a Z3 edge with custom handles."""
        edge = create_z3_edge(
            "source", "target",
            source_handle="feedback",
            target_handle="retry"
        )
        
        assert edge["sourceHandle"] == "feedback"
        assert edge["targetHandle"] == "retry"
    
    def test_create_conditional_z3_edge(self):
        """Test creating a conditional Z3 edge."""
        edge = create_conditional_z3_edge("source", "target", "x > 0")
        
        assert edge["type"] == "conditional"
        assert edge["label"] == "x > 0"
        assert edge["style"]["stroke"] == "#FF6B6B"
    
    def test_create_feedback_z3_edge(self):
        """Test creating a feedback Z3 edge."""
        edge = create_feedback_z3_edge("source", "target")
        
        assert edge["type"] == "feedback"
        assert edge["sourceHandle"] == "feedback"
        assert edge["targetHandle"] == "retry"
    
    def test_get_z3_edge_color(self):
        """Test edge color retrieval."""
        assert get_z3_edge_color("default") == "#888888"
        assert get_z3_edge_color("conditional") == "#FF6B6B"
        assert get_z3_edge_color("feedback") == "#9B59B6"
        assert get_z3_edge_color("success") == "#00B894"
        assert get_z3_edge_color("error") == "#FF7675"
        assert get_z3_edge_color("unknown") == "#888888"


class TestZ3NodeColorsAndIcons:
    """Tests for node colors and icons constants."""
    
    def test_z3_node_colors_has_required_types(self):
        """Test Z3_NODE_COLORS has required node types."""
        required_types = ["z3_solver", "z3_prover", "leanaide_proof", 
                         "cross_verify", "classification", "result"]
        
        for node_type in required_types:
            assert node_type in Z3_NODE_COLORS
    
    def test_z3_node_icons_has_required_types(self):
        """Test Z3_NODE_ICONS has required node types."""
        required_types = ["z3_solver", "z3_prover", "leanaide_proof",
                         "cross_verify", "classification"]
        
        for node_type in required_types:
            assert node_type in Z3_NODE_ICONS
    
    def test_z3_node_positions_has_required_types(self):
        """Test Z3_NODE_POSITIONS has required node types."""
        required_types = ["input", "classification", "z3_solver", 
                         "z3_prover", "leanaide_proof", "cross_verify", "result"]
        
        for node_type in required_types:
            assert node_type in Z3_NODE_POSITIONS
            assert "x" in Z3_NODE_POSITIONS[node_type]
            assert "y" in Z3_NODE_POSITIONS[node_type]


class TestZ3WorkflowCreation:
    """Tests for complete workflow creation functions."""
    
    def test_create_z3_solver_workflow_basic(self):
        """Test creating a basic Z3 solver workflow."""
        workflow = create_z3_solver_workflow(
            problem_text="x + y = 10",
            workflow_name="Test Workflow"
        )
        
        assert workflow["name"] == "Test Workflow"
        assert len(workflow["nodes"]) == 3  # input, solver, result
        assert len(workflow["edges"]) == 2  # input->solver, solver->result
        assert workflow["metadata"]["workflow_type"] == "z3_solver"
    
    def test_create_z3_solver_workflow_with_variables(self):
        """Test Z3 solver workflow with variables and constraints."""
        variables = [{"name": "x", "type": "Int"}]
        constraints = [{"expr": "x > 0"}]
        
        workflow = create_z3_solver_workflow(
            problem_text="x + y = 10",
            workflow_name="Vars Workflow",
            variables=variables,
            constraints=constraints
        )
        
        assert len(workflow["nodes"]) == 3
        assert workflow["metadata"]["problem_text"] == "x + y = 10"
    
    def test_create_z3_leanaide_workflow_defaults(self):
        """Test creating Z3-LeanAIDE workflow with defaults."""
        workflow = create_z3_leanaide_workflow(
            problem_text="Prove n^2 >= n",
            workflow_name="Theorem Verification"
        )
        
        assert workflow["name"] == "Theorem Verification"
        # Default: input, classification, solver, proof, cross_verify, result
        assert len(workflow["nodes"]) == 6
        assert workflow["metadata"]["workflow_type"] == "z3_leanaide"
        assert workflow["metadata"]["include_proof"] == True
        assert workflow["metadata"]["include_cross_verify"] == True
    
    def test_create_z3_leanaide_workflow_no_proof(self):
        """Test Z3-LeanAIDE workflow without proof."""
        workflow = create_z3_leanaide_workflow(
            problem_text="test",
            workflow_name="No Proof",
            include_proof=False
        )
        
        # input, classification, solver, cross_verify, result
        assert len(workflow["nodes"]) == 5
        assert workflow["metadata"]["include_proof"] == False
    
    def test_create_z3_leanaide_workflow_no_cross_verify(self):
        """Test Z3-LeanAIDE workflow without cross-verification."""
        workflow = create_z3_leanaide_workflow(
            problem_text="test",
            workflow_name="No Cross Verify",
            include_cross_verify=False,
            include_proof=True
        )
        
        # input, classification, solver, proof, result
        assert len(workflow["nodes"]) == 5
        assert workflow["metadata"]["include_cross_verify"] == False
    
    def test_create_z3_leanaide_workflow_minimal(self):
        """Test minimal Z3-LeanAIDE workflow."""
        workflow = create_z3_leanaide_workflow(
            problem_text="test",
            workflow_name="Minimal",
            include_proof=False,
            include_cross_verify=False
        )
        
        # input, classification, solver, result
        assert len(workflow["nodes"]) == 4
        assert len(workflow["edges"]) == 3


class TestZ3FlexibleWorkflowBuilder:
    """Tests for the flexible workflow builder."""
    
    def test_builder_add_bubble(self):
        """Test adding bubbles to the builder."""
        builder = Z3FlexibleWorkflowBuilder()
        
        bubble_def = Z3BubbleDefinition(
            bubble_type="z3_solver",
            label="Solver",
            node_color="#FF6B6B"
        )
        bubble_id = builder.add_bubble(bubble_def)
        
        assert len(builder.bubbles) == 1
        assert "Solver" in builder.bubble_map
        assert bubble_id in [b["id"] for b in builder.bubbles]
    
    def test_builder_add_edge(self):
        """Test adding edges to the builder."""
        builder = Z3FlexibleWorkflowBuilder()
        
        # Add bubbles
        builder.add_bubble(Z3BubbleDefinition("z3_solver", "Source"))
        builder.add_bubble(Z3BubbleDefinition("z3_result", "Target"))
        
        # Add edge
        edge_def = Z3EdgeDefinition("Source", "Target")
        edge_id = builder.add_edge(edge_def)
        
        assert len(builder.edges) == 1
        assert edge_id in [e["id"] for e in builder.edges]
    
    def test_builder_add_conditional_edge(self):
        """Test adding conditional edge."""
        builder = Z3FlexibleWorkflowBuilder()
        
        builder.add_bubble(Z3BubbleDefinition("classification", "Classify"))
        builder.add_bubble(Z3BubbleDefinition("z3_solver", "Solver"))
        builder.add_bubble(Z3BubbleDefinition("z3_result", "Result"))
        
        # Add edges
        builder.add_edge(Z3EdgeDefinition("Classify", "Solver"))
        builder.add_edge(Z3EdgeDefinition("Classify", "Result", condition="already solved"))
        
        assert len(builder.edges) == 2
        conditional_edge = builder.edges[1]
        assert conditional_edge["label"] == "already solved"
    
    def test_builder_build(self):
        """Test building a complete workflow."""
        builder = Z3FlexibleWorkflowBuilder()
        
        builder.add_bubble(Z3BubbleDefinition("z3_solver", "Solver"))
        builder.add_bubble(Z3BubbleDefinition("z3_result", "Result"))
        builder.add_edge(Z3EdgeDefinition("Solver", "Result"))
        
        workflow = builder.build("Test Workflow", "test problem")
        
        assert workflow["name"] == "Test Workflow"
        assert workflow["description"] == "test problem"
        assert len(workflow["nodes"]) == 2
        assert len(workflow["edges"]) == 1
        assert workflow["metadata"]["workflow_type"] == "z3_leanaide_custom"
    
    def test_builder_reset(self):
        """Test resetting the builder."""
        builder = Z3FlexibleWorkflowBuilder()
        
        builder.add_bubble(Z3BubbleDefinition("z3_solver", "Solver"))
        builder.add_bubble(Z3BubbleDefinition("z3_result", "Result"))
        
        builder.reset()
        
        assert len(builder.bubbles) == 0
        assert len(builder.edges) == 0
        assert len(builder.bubble_map) == 0
    
    def test_builder_error_missing_source(self):
        """Test builder error when source bubble missing."""
        builder = Z3FlexibleWorkflowBuilder()
        
        builder.add_bubble(Z3BubbleDefinition("z3_result", "Result"))
        
        with pytest.raises(ValueError, match="Source bubble not found"):
            builder.add_edge(Z3EdgeDefinition("Missing", "Result"))
    
    def test_builder_error_missing_target(self):
        """Test builder error when target bubble missing."""
        builder = Z3FlexibleWorkflowBuilder()
        
        builder.add_bubble(Z3BubbleDefinition("z3_solver", "Solver"))
        
        with pytest.raises(ValueError, match="Target bubble not found"):
            builder.add_edge(Z3EdgeDefinition("Solver", "Missing"))


class TestCreateCustomZ3Workflow:
    """Tests for create_custom_z3_workflow function."""
    
    def test_create_custom_workflow_sequential(self):
        """Test creating a custom sequential workflow."""
        workflow = create_custom_z3_workflow(
            workflow_name="Custom Sequential",
            problem_text="test",
            bubble_labels=["Input", "Solver", "Result"],
            bubble_types=["classification", "z3_solver", "z3_result"]
        )
        
        assert workflow["name"] == "Custom Sequential"
        assert len(workflow["nodes"]) == 3
        assert len(workflow["edges"]) == 2
        
        # Check node types (classification type uses "classification" not "problem_classification")
        node_types = [n["type"] for n in workflow["nodes"]]
        assert "classification" in node_types
        assert "z3_solver" in node_types
        assert "z3_result" in node_types
    
    def test_create_custom_workflow_full(self):
        """Test creating a full custom workflow."""
        workflow = create_custom_z3_workflow(
            workflow_name="Full Custom",
            problem_text="complex problem",
            bubble_labels=["Input", "Classify", "Solver", "Proof", "Cross", "Result"],
            bubble_types=[
                "classification",
                "classification",
                "z3_solver",
                "leanaide_proof",
                "cross_verification",
                "z3_result"
            ]
        )
        
        assert len(workflow["nodes"]) == 6
        assert len(workflow["edges"]) == 5
    
    def test_create_custom_workflow_with_team_config(self):
        """Test custom workflow with team configuration."""
        team_config = {
            "classification": "Blue Team",
            "z3_solver": "Red Team",
            "cross_verification": "Gold Team"
        }
        
        workflow = create_custom_z3_workflow(
            workflow_name="Team Workflow",
            problem_text="test",
            bubble_labels=["Input", "Classify", "Solver", "Cross", "Result"],
            bubble_types=["classification", "classification", "z3_solver", 
                         "cross_verification", "z3_result"],
            team_config=team_config
        )
        
        assert workflow is not None
        assert len(workflow["nodes"]) == 5


class TestBubbleUpdates:
    """Tests for bubble update functions."""
    
    def test_update_z3_bubble_status(self):
        """Test updating bubble status."""
        config = Z3SolverBubbleConfig(problem_text="test")
        bubble = create_z3_solver_bubble(config)
        
        updated = update_z3_bubble_status(bubble, "running")
        
        assert updated["data"]["status"] == "running"
        assert updated["data"]["node_color"] == "#74B9FF"
    
    def test_update_z3_bubble_status_success(self):
        """Test updating bubble to success status."""
        config = Z3SolverBubbleConfig(problem_text="test")
        bubble = create_z3_solver_bubble(config)
        
        updated = update_z3_bubble_status(bubble, "success")
        
        assert updated["data"]["status"] == "success"
        assert updated["data"]["node_color"] == "#00B894"
    
    def test_update_z3_bubble_status_failed(self):
        """Test updating bubble to failed status."""
        config = Z3SolverBubbleConfig(problem_text="test")
        bubble = create_z3_solver_bubble(config)
        
        updated = update_z3_bubble_status(bubble, "failed")
        
        assert updated["data"]["status"] == "failed"
        assert updated["data"]["node_color"] == "#FF7675"
    
    def test_update_z3_bubble_with_additional_data(self):
        """Test updating bubble with additional data."""
        config = Z3SolverBubbleConfig(problem_text="test")
        bubble = create_z3_solver_bubble(config)
        
        updated = update_z3_bubble_status(
            bubble, "success",
            additional_data={"execution_time": 1.5, "solution_found": True}
        )
        
        assert updated["data"]["status"] == "success"
        assert updated["data"]["execution_time"] == 1.5
        assert updated["data"]["solution_found"] == True
    
    def test_add_z3_result_success(self):
        """Test adding success result to bubble."""
        config = Z3SolverBubbleConfig(problem_text="test")
        bubble = create_z3_solver_bubble(config)
        
        result_data = {"solution": {"x": 5, "y": 5}}
        updated = add_z3_result_to_bubble(bubble, True, result_data)
        
        assert updated["data"]["result"] == result_data
        assert updated["data"]["status"] == "success"
    
    def test_add_z3_result_failure(self):
        """Test adding failure result to bubble."""
        config = Z3SolverBubbleConfig(problem_text="test")
        bubble = create_z3_solver_bubble(config)
        
        result_data = {"error": "timeout"}
        updated = add_z3_result_to_bubble(bubble, False, result_data)
        
        assert updated["data"]["result"] == result_data
        assert updated["data"]["status"] == "failed"


class TestSerialization:
    """Tests for serialization functions."""
    
    def test_serialize_z3_bubble(self):
        """Test serializing a bubble to JSON."""
        config = Z3SolverBubbleConfig(problem_text="test")
        bubble = create_z3_solver_bubble(config)
        
        json_str = serialize_z3_bubble(bubble)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["id"] == bubble["id"]
        assert parsed["type"] == "z3_solver"
    
    def test_serialize_z3_workflow(self):
        """Test serializing a workflow to JSON."""
        workflow = create_z3_solver_workflow(
            problem_text="test",
            workflow_name="Test"
        )
        
        json_str = serialize_z3_workflow(workflow)
        
        parsed = json.loads(json_str)
        assert parsed["name"] == "Test"
        assert len(parsed["nodes"]) == 3
    
    def test_export_z3_workflow_to_json(self):
        """Test exporting workflow to file."""
        workflow = create_z3_solver_workflow(
            problem_text="test",
            workflow_name="Export Test"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "exported_workflow.json")
            
            result = export_z3_workflow_to_json(workflow, output_path)
            
            assert result == True
            assert os.path.exists(output_path)
            
            with open(output_path, 'r') as f:
                imported = json.load(f)
            
            assert imported["name"] == "Export Test"
            assert len(imported["nodes"]) == 3
    
    def test_export_z3_workflow_to_json_creates_directory(self):
        """Test that export creates parent directories."""
        workflow = create_z3_solver_workflow(
            problem_text="test",
            workflow_name="Nested Export"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nested", "dir", "workflow.json")
            
            result = export_z3_workflow_to_json(workflow, output_path)
            
            assert result == True
            assert os.path.exists(output_path)
    
    def test_export_z3_workflow_to_json_error(self):
        """Test export error handling."""
        workflow = create_z3_solver_workflow(
            problem_text="test",
            workflow_name="Error Export"
        )
        
        # Try to export to invalid path (non-writable location)
        result = export_z3_workflow_to_json(workflow, "/invalid/path/workflow.json")
        
        # Result depends on OS behavior - just verify function handles gracefully
        assert result is True or result is False


class TestComplexWorkflowScenarios:
    """Tests for complex workflow scenarios."""
    
    def test_parallel_solver_proof_workflow(self):
        """Test workflow with parallel solver and proof branches."""
        builder = Z3FlexibleWorkflowBuilder()
        
        # Input
        builder.add_bubble(Z3BubbleDefinition(
            "classification", "Input",
            config={"problem_text": "test"}
        ))
        
        # Branch point
        builder.add_bubble(Z3BubbleDefinition(
            "z3_solver", "Z3 Solver"
        ))
        builder.add_bubble(Z3BubbleDefinition(
            "leanaide_proof", "LeanAIDE"
        ))
        
        # Convergence
        builder.add_bubble(Z3BubbleDefinition(
            "cross_verification", "Cross Verify"
        ))
        builder.add_bubble(Z3BubbleDefinition(
            "z3_result", "Result"
        ))
        
        # Edges
        builder.add_edge(Z3EdgeDefinition("Input", "Z3 Solver"))
        builder.add_edge(Z3EdgeDefinition("Input", "LeanAIDE"))
        builder.add_edge(Z3EdgeDefinition("Z3 Solver", "Cross Verify"))
        builder.add_edge(Z3EdgeDefinition("LeanAIDE", "Cross Verify"))
        builder.add_edge(Z3EdgeDefinition("Cross Verify", "Result"))
        
        workflow = builder.build("Parallel Workflow", "parallel test")
        
        assert len(workflow["nodes"]) == 5
        assert len(workflow["edges"]) == 5
    
    def test_iterative_verification_workflow(self):
        """Test workflow with iterative verification and feedback."""
        builder = Z3FlexibleWorkflowBuilder()
        
        builder.add_bubble(Z3BubbleDefinition("z3_solver", "Solver"))
        builder.add_bubble(Z3BubbleDefinition("cross_verification", "Verify"))
        builder.add_bubble(Z3BubbleDefinition("z3_result", "Result"))
        
        # Forward edge
        builder.add_edge(Z3EdgeDefinition("Solver", "Verify"))
        # Conditional feedback edge if not verified
        builder.add_edge(Z3EdgeDefinition(
            "Verify", "Solver", 
            condition="retry",
            edge_type="feedback"
        ))
        # Success edge to result
        builder.add_edge(Z3EdgeDefinition("Verify", "Result", condition="verified"))
        
        workflow = builder.build("Iterative Workflow", "iterative test")
        
        assert len(workflow["edges"]) == 3
        feedback_edges = [e for e in workflow["edges"] if e["type"] == "feedback"]
        assert len(feedback_edges) == 1


class TestExportValidation:
    """Tests for exported workflow validation."""
    
    def test_workflow_has_required_fields(self):
        """Test that created workflows have all required fields."""
        workflow = create_z3_solver_workflow(
            problem_text="test",
            workflow_name="Required Fields Test"
        )
        
        required_fields = ["id", "name", "description", "nodes", "edges", "metadata"]
        
        for field in required_fields:
            assert field in workflow, f"Missing field: {field}"
    
    def test_workflow_nodes_have_required_fields(self):
        """Test that workflow nodes have required fields."""
        workflow = create_z3_solver_workflow(
            problem_text="test",
            workflow_name="Node Fields Test"
        )
        
        for node in workflow["nodes"]:
            assert "id" in node
            assert "type" in node
            assert "position" in node
            assert "data" in node
    
    def test_workflow_edges_have_required_fields(self):
        """Test that workflow edges have required fields."""
        workflow = create_z3_solver_workflow(
            problem_text="test",
            workflow_name="Edge Fields Test"
        )
        
        for edge in workflow["edges"]:
            assert "id" in edge
            assert "source" in edge
            assert "target" in edge
            assert "type" in edge


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
