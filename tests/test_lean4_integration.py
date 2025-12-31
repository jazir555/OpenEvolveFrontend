import unittest
from unittest import mock
import json
import os
import sys
import dataclasses

# Add the parent directory to the sys.path to allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from workflow_engine import _run_standard_gauntlet_headless # Using headless for easier testing
from workflow_structures import GauntletDefinition, GauntletRoundRule, Team, ModelConfig
from lean4_system.lean4_api import MathematicalVerificationAPI
from lean4_system.lean4_client import Lean4Client
from lean4_system.lean4_data_models import VerificationResult

# Mock the config.yaml reading
mock_config_content = {
    "default": {
        "lean_verification": {
            "lean_prover": {
                "endpoint": "http://mock-lean4-server:3000",
                "timeout": 10
            }
        }
    }
}

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.exceptions.HTTPError(f"HTTP Error: {self.status_code}")

class TestLean4Integration(unittest.TestCase):

    @mock.patch('requests.post')
    def test_lean4_client_verify_properties_success(self, mock_post):
        mock_post.return_value = MockResponse({"is_verified": True, "proof_status": "proven", "details": {}, "confidence": 0.99})
        client = Lean4Client("http://test-server:3000")
        result = client.verify_properties({"model": "test"}, ["property1"])
        self.assertTrue(result["is_verified"])
        self.assertEqual(result["proof_status"], "proven")

    @mock.patch('requests.post')
    def test_lean4_client_generate_proof_success(self, mock_post):
        mock_post.return_value = MockResponse({"success": True, "proof": "mock_proof_content"})
        client = Lean4Client("http://test-server:3000")
        proof = client.generate_proof("statement")
        self.assertEqual(proof, "mock_proof_content")

    @mock.patch('requests.post')
    def test_mathematical_verification_api_submit_request_success(self, mock_post):
        mock_post.return_value = MockResponse({"is_verified": True, "proof_status": "proven", "details": {}, "confidence": 0.99})
        api = MathematicalVerificationAPI("http://test-server:3000")
        result = api.submit_verification_request({"component": "test"}, ["property1"])
        self.assertIsInstance(result, VerificationResult)
        self.assertTrue(result.is_verified)

    @mock.patch('yaml.safe_load', return_value=mock_config_content)
    @mock.patch('builtins.open', mock.mock_open())
    @mock.patch('requests.post')
    def test_gold_team_gauntlet_with_lean4_verification_success(self, mock_post, mock_open, mock_yaml_load):
        # Mock Lean 4 verification success
        mock_post.return_value = MockResponse({"is_verified": True, "proof_status": "proven", "details": {}, "confidence": 0.99})

        # Mock LLM evaluation success
        mock_llm_response = MockResponse({"score": 0.9, "justification": "Looks good", "targeted_feedback": []})
        with mock.patch('workflow_engine._request_openai_compatible_chat', return_value=json.dumps(mock_llm_response.json())):
            
            # Define a Gold Team
            gold_team_member = ModelConfig(model_id="mock-gold-llm", api_key="mock-key")
            gold_team = Team(name="GoldTeam", role="Gold", members=[gold_team_member])

            # Define a GauntletRule with Lean 4 verification enabled
            lean4_round_rule = GauntletRoundRule(
                round_number=1,
                quorum_required_approvals=1,
                quorum_from_panel_size=1,
                min_overall_confidence=0.8,
                proof_verification_enabled=True,
                required_mathematical_properties=["correctness"],
                proof_obligation_threshold=0.9
            )
            gauntlet_def = GauntletDefinition(name="Lean4GoldGauntlet", team_name="GoldTeam", rounds=[lean4_round_rule])

            solution_content = "def add(a, b): return a + b"
            context = {"solution_id": "sol_123", "evaluation_prompt": "Verify the add function."}
            
            result = _run_standard_gauntlet_headless(solution_content, gauntlet_def, gold_team, context, logs=[])
            
            self.assertTrue(result["is_approved"])
            self.assertIn("Lean 4 verification PASSED", result["logs"][-2]) # Check log message

    @mock.patch('yaml.safe_load', return_value=mock_config_content)
    @mock.patch('builtins.open', mock.mock_open())
    @mock.patch('requests.post')
    def test_gold_team_gauntlet_with_lean4_verification_failure(self, mock_post, mock_open, mock_yaml_load):
        # Mock Lean 4 verification failure (e.g., confidence too low)
        mock_post.return_value = MockResponse({"is_verified": True, "proof_status": "proven", "details": {}, "confidence": 0.7}) # Below threshold
        
        # Mock LLM evaluation success
        mock_llm_response = MockResponse({"score": 0.9, "justification": "Looks good", "targeted_feedback": []})
        with mock.patch('workflow_engine._request_openai_compatible_chat', return_value=json.dumps(mock_llm_response.json())):
            
            # Define a Gold Team
            gold_team_member = ModelConfig(model_id="mock-gold-llm", api_key="mock-key")
            gold_team = Team(name="GoldTeam", role="Gold", members=[gold_team_member])

            # Define a GauntletRule with Lean 4 verification enabled
            lean4_round_rule = GauntletRoundRule(
                round_number=1,
                quorum_required_approvals=1,
                quorum_from_panel_size=1,
                min_overall_confidence=0.8,
                proof_verification_enabled=True,
                required_mathematical_properties=["correctness"],
                proof_obligation_threshold=0.9 # Higher than mock confidence
            )
            gauntlet_def = GauntletDefinition(name="Lean4GoldGauntlet", team_name="GoldTeam", rounds=[lean4_round_rule])

            solution_content = "def add(a, b): return a + b"
            context = {"solution_id": "sol_123", "evaluation_prompt": "Verify the add function."}
            
            result = _run_standard_gauntlet_headless(solution_content, gauntlet_def, gold_team, context, logs=[])
            
            self.assertFalse(result["is_approved"])
            self.assertIn("Lean 4 verification FAILED", result["logs"][-2]) # Check log message

    @mock.patch('yaml.safe_load', return_value=mock_config_content)
    @mock.patch('builtins.open', mock.mock_open())
    @mock.patch('requests.post')
    def test_gold_team_gauntlet_with_lean4_verification_error(self, mock_post, mock_open, mock_yaml_load):
        # Mock Lean 4 verification error (e.g., server unreachable)
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")
        
        # Mock LLM evaluation success
        mock_llm_response = MockResponse({"score": 0.9, "justification": "Looks good", "targeted_feedback": []})
        with mock.patch('workflow_engine._request_openai_compatible_chat', return_value=json.dumps(mock_llm_response.json())):
            
            # Define a Gold Team
            gold_team_member = ModelConfig(model_id="mock-gold-llm", api_key="mock-key")
            gold_team = Team(name="GoldTeam", role="Gold", members=[gold_team_member])

            # Define a GauntletRule with Lean 4 verification enabled
            lean4_round_rule = GauntletRoundRule(
                round_number=1,
                quorum_required_approvals=1,
                quorum_from_panel_size=1,
                min_overall_confidence=0.8,
                proof_verification_enabled=True,
                required_mathematical_properties=["correctness"],
                proof_obligation_threshold=0.9
            )
            gauntlet_def = GauntletDefinition(name="Lean4GoldGauntlet", team_name="GoldTeam", rounds=[lean4_round_rule])

            solution_content = "def add(a, b): return a + b"
            context = {"solution_id": "sol_123", "evaluation_prompt": "Verify the add function."}
            
            result = _run_standard_gauntlet_headless(solution_content, gauntlet_def, gold_team, context, logs=[])
            
            self.assertFalse(result["is_approved"])
            self.assertIn("Lean 4 verification encountered an ERROR", result["logs"][-2]) # Check log message

if __name__ == '__main__':
    unittest.main()
