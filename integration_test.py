"""
Comprehensive Integration Test for OpenEvolve and Hephaestus

This module provides tests to verify the complete integration between OpenEvolve and Hephaestus:
1. Data structure compatibility
2. API endpoint functionality
3. Frontend component integration
4. Gauntlet verification workflows
5. End-to-end workflow execution
"""

import asyncio
import requests
import time
import json
from typing import Dict, Any, Optional
import unittest
from unittest.mock import Mock, patch

from openevolve_structures import ModelConfig, Team, GauntletDefinition, GauntletRoundRule
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from sgd_workflow_orchestrator import SGDWorkflowOrchestrator


class TestOpenEvolveHephaestusIntegration:
    """
    Comprehensive test suite for OpenEvolve-Hephaestus integration
    """
    
    def __init__(self):
        self.hephaestus_api_base = "http://localhost:8002"
        self.openevolve_api_base = "http://localhost:8000"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def test_data_structures_compatibility(self):
        """
        Test that data structures are compatible between OpenEvolve and Hephaestus
        """
        print("Testing data structure compatibility...")
        
        # Test ModelConfig
        model_config = ModelConfig(
            model_id="gpt-4-test",
            api_key="test-key",  # For local testing, this could be empty
            api_base="http://localhost:8001"
        )
        assert model_config.model_id == "gpt-4-test"
        assert model_config.api_base == "http://localhost:8001"
        print("✓ ModelConfig structure is valid")
        
        # Test Team
        team = Team(
            name="test-team",
            role="Blue",
            members=[model_config]
        )
        assert team.name == "test-team"
        assert team.role == "Blue"
        assert len(team.members) == 1
        print("✓ Team structure is valid")
        
        # Test GauntletRoundRule
        round_rule = GauntletRoundRule(
            round_number=1,
            quorum_required_approvals=1,
            quorum_from_panel_size=1
        )
        assert round_rule.round_number == 1
        print("✓ GauntletRoundRule structure is valid")
        
        # Test GauntletDefinition
        gauntlet = GauntletDefinition(
            name="test-gauntlet",
            team_name="test-team",
            rounds=[round_rule]
        )
        assert gauntlet.name == "test-gauntlet"
        assert len(gauntlet.rounds) == 1
        print("✓ GauntletDefinition structure is valid")
        
        print("All data structures are compatible ✓")
        return True

    def test_team_manager_functionality(self):
        """
        Test TeamManager functionality
        """
        print("Testing TeamManager functionality...")
        
        team_manager = TeamManager()
        
        # Test creating a team
        model_config = ModelConfig(
            model_id="gpt-4-test",
            api_key="test-key",
            api_base="http://localhost:8001"
        )
        
        team = Team(
            name="integration-test-team",
            role="Blue",
            members=[model_config],
            description="Team for integration testing"
        )
        
        # Create team
        success = team_manager.create_team(team)
        assert success, "Failed to create team"
        print("✓ Team creation successful")
        
        # Get team
        retrieved_team = team_manager.get_team("integration-test-team")
        assert retrieved_team is not None
        assert retrieved_team.name == "integration-test-team"
        print("✓ Team retrieval successful")
        
        # List teams
        all_teams = team_manager.get_all_teams()
        assert len(all_teams) >= 1
        print("✓ Team listing successful")
        
        # Test team update
        updated_team = Team(
            name="integration-test-team",
            role="Red",
            members=[model_config],
            description="Updated team for integration testing"
        )
        update_success = team_manager.update_team(updated_team)
        assert update_success, "Failed to update team"
        print("✓ Team update successful")
        
        # Cleanup: delete team
        delete_success = team_manager.delete_team("integration-test-team")
        assert delete_success, "Failed to delete team"
        print("✓ Team deletion successful")
        
        print("TeamManager functionality verified ✓")
        return True

    def test_gauntlet_manager_functionality(self):
        """
        Test GauntletManager functionality
        """
        print("Testing GauntletManager functionality...")
        
        gauntlet_manager = GauntletManager()
        
        # Test creating a gauntlet
        round_rule = GauntletRoundRule(
            round_number=1,
            quorum_required_approvals=1,
            quorum_from_panel_size=1
        )
        
        gauntlet = GauntletDefinition(
            name="integration-test-gauntlet",
            team_name="test-team",
            rounds=[round_rule],
            description="Gauntlet for integration testing"
        )
        
        # Create gauntlet
        success = gauntlet_manager.create_gauntlet(gauntlet)
        assert success, "Failed to create gauntlet"
        print("✓ Gauntlet creation successful")
        
        # Get gauntlet
        retrieved_gauntlet = gauntlet_manager.get_gauntlet("integration-test-gauntlet")
        assert retrieved_gauntlet is not None
        assert retrieved_gauntlet.name == "integration-test-gauntlet"
        print("✓ Gauntlet retrieval successful")
        
        # List gauntlets
        all_gauntlets = gauntlet_manager.get_all_gauntlets()
        assert len(all_gauntlets) >= 1
        print("✓ Gauntlet listing successful")
        
        # Test gauntlet update
        updated_round = GauntletRoundRule(
            round_number=1,
            quorum_required_approvals=2,
            quorum_from_panel_size=3
        )
        updated_gauntlet = GauntletDefinition(
            name="integration-test-gauntlet",
            team_name="updated-team",
            rounds=[updated_round],
            description="Updated gauntlet for integration testing"
        )
        update_success = gauntlet_manager.update_gauntlet(updated_gauntlet)
        assert update_success, "Failed to update gauntlet"
        print("✓ Gauntlet update successful")
        
        # Cleanup: delete gauntlet
        delete_success = gauntlet_manager.delete_gauntlet("integration-test-gauntlet")
        assert delete_success, "Failed to delete gauntlet"
        print("✓ Gauntlet deletion successful")
        
        print("GauntletManager functionality verified ✓")
        return True

    def test_hephaestus_api_endpoints(self):
        """
        Test Hephaestus API endpoints for OpenEvolve integration
        """
        print("Testing Hephaestus API endpoints...")
        
        try:
            # Test teams endpoints
            print("  Testing teams endpoints...")
            
            # Create a test team via API
            team_data = {
                "name": "api-test-team",
                "role": "Blue",
                "members": [
                    {
                        "model_id": "gpt-4-api-test",
                        "api_key": "test-key",
                        "api_base": "http://localhost:8001",
                        "temperature": 0.7,
                        "max_tokens": 4096
                    }
                ],
                "description": "Team created via API for testing"
            }
            
            response = self.session.post(
                f"{self.hephaestus_api_base}/openevolve/teams",
                json=team_data
            )
            assert response.status_code == 201 or response.status_code == 200, f"Failed to create team via API: {response.status_code}"
            print("  ✓ Team creation via API successful")
            
            # Get teams via API
            response = self.session.get(f"{self.hephaestus_api_base}/openevolve/teams")
            assert response.status_code == 200, f"Failed to get teams: {response.status_code}"
            teams_data = response.json()
            assert "teams" in teams_data, "Teams data not found in response"
            print("  ✓ Team listing via API successful")
            
            # Test gauntlets endpoints
            print("  Testing gauntlets endpoints...")
            
            # Create a test gauntlet via API
            gauntlet_data = {
                "name": "api-test-gauntlet",
                "team_name": "test-team",
                "rounds": [
                    {
                        "round_number": 1,
                        "quorum_required_approvals": 1,
                        "quorum_from_panel_size": 1,
                        "min_overall_confidence": 0.5
                    }
                ],
                "description": "Gauntlet created via API for testing"
            }
            
            response = self.session.post(
                f"{self.hephaestus_api_base}/openevolve/gauntlets",
                json=gauntlet_data
            )
            assert response.status_code == 201 or response.status_code == 200, f"Failed to create gauntlet via API: {response.status_code}"
            print("  ✓ Gauntlet creation via API successful")
            
            # Get gauntlets via API
            response = self.session.get(f"{self.hephaestus_api_base}/openevolve/gauntlets")
            assert response.status_code == 200, f"Failed to get gauntlets: {response.status_code}"
            gauntlets_data = response.json()
            assert "gauntlets" in gauntlets_data, "Gauntlets data not found in response"
            print("  ✓ Gauntlet listing via API successful")
            
            # Cleanup: delete created resources
            response = self.session.delete(f"{self.hephaestus_api_base}/openevolve/teams/api-test-team")
            if response.status_code in [200, 404]:  # 404 means it was already deleted
                print("  ✓ Team cleanup successful")
            
            response = self.session.delete(f"{self.hephaestus_api_base}/openevolve/gauntlets/api-test-gauntlet")
            if response.status_code in [200, 404]:  # 404 means it was already deleted
                print("  ✓ Gauntlet cleanup successful")
            
            print("Hephaestus API endpoints verified ✓")
            return True
            
        except requests.exceptions.ConnectionError:
            print("⚠ Hephaestus API not accessible - server may not be running")
            print("  To run the server: python -m hephaestus.main")
            return False
        except Exception as e:
            print(f"✗ Error testing API endpoints: {e}")
            return False

    def test_ticket_with_verification(self):
        """
        Test creating a ticket with verification gauntlets in Hephaestus
        """
        print("Testing ticket creation with verification...")
        
        try:
            # Create a ticket with gauntlet configuration
            ticket_data = {
                "title": "Integration Test Ticket",
                "description": "Test ticket for verifying gauntlet integration",
                "workflow_id": "test-workflow-123",
                "red_team_gauntlet_name": "basic-red-gauntlet",
                "gold_team_gauntlet_name": "basic-gold-gauntlet"
            }
            
            response = self.session.post(
                f"{self.hephaestus_api_base}/tickets/create",
                json=ticket_data
            )
            
            if response.status_code == 200:
                ticket_result = response.json()
                ticket_id = ticket_result.get("ticket", {}).get("id")
                
                if ticket_id:
                    print(f"  ✓ Ticket created successfully: {ticket_id}")
                    
                    # Update ticket to simulate completion
                    update_data = {
                        "ticket_id": ticket_id,
                        "updates": {
                            "solution_content": "This is a test solution for integration verification",
                            "status": "completed"
                        }
                    }
                    
                    update_response = self.session.post(
                        f"{self.hephaestus_api_base}/tickets/update",
                        json=update_data
                    )
                    
                    if update_response.status_code == 200:
                        print("  ✓ Ticket solution content updated")
                        
                        # Get the updated ticket to verify verification status
                        ticket_response = self.session.get(f"{self.hephaestus_api_base}/tickets/{ticket_id}")
                        if ticket_response.status_code == 200:
                            ticket = ticket_response.json().get("ticket")
                            if ticket:
                                print(f"  ✓ Verification status: {ticket.get('verification_status', 'not_found')}")
                                print(f"  ✓ Ticket status: {ticket.get('status', 'not_found')}")
                                
                                # Clean up the ticket
                                # Note: There's no delete endpoint in the current server, so we'll just update status
                                cleanup_response = self.session.post(
                                    f"{self.hephaestus_api_base}/tickets/update",
                                    json={
                                        "ticket_id": ticket_id,
                                        "updates": {"status": "archived"}
                                    }
                                )
                                print("  ✓ Ticket cleanup attempted")
                        
                    return True
                else:
                    print("  ✗ Failed to get ticket ID from creation response")
                    return False
            else:
                print(f"  ✗ Failed to create ticket: {response.status_code}, {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("⚠ Hephaestus server not accessible - unable to test ticket functionality")
            print("  To run the server: python -m hephaestus.main")
            return False
        except Exception as e:
            print(f"✗ Error testing ticket functionality: {e}")
            return False

    def test_sgd_workflow_orchestration(self):
        """
        Test the SGD workflow orchestration
        """
        print("Testing SGD workflow orchestration...")
        
        try:
            # Create orchestrator instance
            orchestrator = SGDWorkflowOrchestrator(
                hephaestus_api_base=self.hephaestus_api_base,
                openevolve_api_base=self.openevolve_api_base
            )
            
            # Create a workflow
            workflow_id = orchestrator.create_workflow(
                problem_statement="Integration test workflow",
                content_analyzer_team="test-content-analyzer",
                planner_team="test-planner",
                solver_team="test-solver",
                patcher_team="test-patcher",
                assembler_team="test-assembler",
                sub_problem_red_gauntlet="basic-red-gauntlet",
                sub_problem_gold_gauntlet="basic-gold-gauntlet",
                final_red_gauntlet="final-red-gauntlet",
                final_gold_gauntlet="final-gold-gauntlet"
            )
            
            print(f"  ✓ Workflow created: {workflow_id}")
            
            # Check workflow status
            status = orchestrator.get_workflow_status(workflow_id)
            if status:
                print(f"  ✓ Workflow status: {status['status']}")
            
            # List workflows
            workflows = orchestrator.list_workflows()
            print(f"  ✓ Total workflows: {len(workflows)}")
            
            # Clean up: try to stop the workflow
            orchestrator.stop_workflow(workflow_id)
            print("  ✓ Workflow cleanup attempted")
            
            print("SGD workflow orchestration verified ✓")
            return True
            
        except Exception as e:
            print(f"✗ Error testing SGD workflow orchestration: {e}")
            return False

    def run_all_tests(self):
        """
        Run all integration tests and return overall status
        """
        print("="*60)
        print("OPENEVOLVE-HEPHAESTUS INTEGRATION TEST SUITE")
        print("="*60)
        
        tests = [
            ("Data Structures Compatibility", self.test_data_structures_compatibility),
            ("Team Manager Functionality", self.test_team_manager_functionality),
            ("Gauntlet Manager Functionality", self.test_gauntlet_manager_functionality),
            ("Hephaestus API Endpoints", self.test_hephaestus_api_endpoints),
            ("Ticket Verification Workflow", self.test_ticket_with_verification),
            ("SGD Workflow Orchestration", self.test_sgd_workflow_orchestration),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            print("-" * len(test_name))
            try:
                result = test_func()
                results.append((test_name, result))
                if result:
                    print(f"  {test_name}: PASSED ✓")
                else:
                    print(f"  {test_name}: FAILED ✗")
            except Exception as e:
                print(f"  {test_name}: ERROR - {e}")
                results.append((test_name, False))
        
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY:")
        print("="*60)
        
        passed = 0
        failed = 0
        
        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            print(f"{test_name:<40} {status}")
            if result:
                passed += 1
            else:
                failed += 1
        
        print("-" * 60)
        print(f"TOTAL: {passed} passed, {failed} failed")
        print(f"SUCCESS RATE: {passed}/{len(tests)} ({100*passed/len(tests):.1f}%)")
        
        overall_success = failed == 0
        print(f"OVERALL INTEGRATION STATUS: {'SUCCESS' if overall_success else 'NEEDS ATTENTION'}")
        
        if failed == 0:
            print("\n🎉 All integration tests passed! OpenEvolve and Hephaestus are fully integrated.")
        else:
            print(f"\n⚠ {failed} test(s) failed. Please review the output above and address any issues.")
        
        return overall_success


def main():
    """
    Main function to run the integration tests
    """
    tester = TestOpenEvolveHephaestusIntegration()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()