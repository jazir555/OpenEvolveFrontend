import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationVerifier")

def verify_gauntlet_system():
    logger.info("Verifying Gauntlet System...")
    try:
        from gauntlet_system import GauntletSystem, GauntletSystemConfig
        system = GauntletSystem()
        logger.info("GauntletSystem instantiated successfully.")
        
        # Test evaluation
        test_submission = {
            "content": "def add(a, b): return a + b",
            "domain": "code_python"
        }
        result = system.evaluate(test_submission)
        logger.info(f"Evaluation result: {result.get('passed', False)} (Score: {result.get('score', 0.0):.2f})")
        
        return True
    except Exception as e:
        logger.error(f"GauntletSystem verification failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def verify_workflow_engine():
    logger.info("Verifying Workflow Engine integration...")
    try:
        from unittest.mock import patch
        with patch("workflow_engine._request_openai_compatible_chat") as mock_chat:
            mock_chat.return_value = '{"score": 0.9, "justification": "Test pass", "targeted_feedback": []}'
            
            from workflow_engine import run_gauntlet_headless
            from workflow_structures import GauntletDefinition, Team, GauntletRoundRule, ModelConfig
            
            # Create minimal structures for testing
            member = ModelConfig(model_id="test-model", api_key="test-key")
            team = Team(name="test-team", role="Gold", members=[member])
            round_rule = GauntletRoundRule(
                round_number=1, 
                quorum_required_approvals=1, 
                quorum_from_panel_size=1,
                min_overall_confidence=0.5
            )
            gauntlet_def = GauntletDefinition(
                name="test-gauntlet",
                team_name="test-team",
                rounds=[round_rule],
                gauntlet_type="standard"
            )
            
            result = run_gauntlet_headless(
                solution_content="test solution",
                gauntlet_def=gauntlet_def,
                team=team,
                context={"solution_id": "test-sol"}
            )
            logger.info(f"Headless gauntlet run result: {result.get('is_approved', False)}")
            
            return True
    except Exception as e:
        logger.error(f"Workflow Engine verification failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def verify_adaptive_mdap():
    logger.info("Verifying Adaptive MDAP integration...")
    try:
        from adaptive_mdap import get_adaptive_workflow
        from workflow_structures import SubProblem
        
        sp = SubProblem(id="test-sp", description="test sub-problem")
        integration = get_adaptive_workflow()
        config = integration.get_solver_config(sp)
        logger.info(f"Adaptive MDAP config: {config}")
        
        return True
    except Exception as e:
        logger.error(f"Adaptive MDAP verification failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = True
    success &= verify_gauntlet_system()
    print("-" * 20)
    success &= verify_workflow_engine()
    print("-" * 20)
    success &= verify_adaptive_mdap()
    
    if success:
        print("\n[SUCCESS] All core integrations verified successfully.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Integration verification failed.")
        sys.exit(1)