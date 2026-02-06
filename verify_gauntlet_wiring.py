import logging
import sys
import json
from datetime import datetime

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("VerifyWiring")

def test_gauntlet_instantiation():
    try:
        from gauntlet_types import (
            create_gauntlet, list_available_gauntlets,
            AdversarialGauntlet, FormalVerificationGauntlet, 
            LogicalSandboxGauntlet, LeanVerificationGauntlet,
            EvolutionaryGauntlet
        )
        logger.info("Successfully imported gauntlet types")
        
        available = list_available_gauntlets()
        logger.info(f"Available gauntlets: {list(available.keys())}")
        
        # Test individual instantiations
        gauntlets = []
        gauntlets.append(create_gauntlet("adversarial", "test_adv"))
        gauntlets.append(create_gauntlet("formal_verification", "test_formal"))
        gauntlets.append(create_gauntlet("logical_sandbox", "test_sandbox"))
        gauntlets.append(create_gauntlet("lean_verification", "test_lean"))
        gauntlets.append(create_gauntlet("evolutionary", "test_evo"))
        gauntlets.append(create_gauntlet("physics", "test_physics"))
        gauntlets.append(create_gauntlet("web3", "test_web3"))
        
        logger.info(f"Successfully instantiated {len(gauntlets)} gauntlet types")
        
        for g in gauntlets:
            logger.info(f"  - {g.name} ({g.gauntlet_type})")
            
        return True
    except Exception as e:
        logger.error(f"Gauntlet instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_z3_integration():
    try:
        from z3prover_integration import (
            Z3SolverEngine, Z3TheoremProver, 
            DigitalTwinSandbox, SmartContractInvariantTranslator
        )
        logger.info("Successfully imported Z3 integration components")
        
        engine = Z3SolverEngine()
        prover = Z3TheoremProver()
        sandbox = DigitalTwinSandbox()
        translator = SmartContractInvariantTranslator()
        
        logger.info("Successfully instantiated all Z3 components")
        return True
    except Exception as e:
        logger.error(f"Z3 integration components instantiation failed: {e}")
        return False

def test_evolution_engine():
    try:
        from evolution import EvolutionEngine, run_evolution_loop, EvolutionConfiguration
        logger.info("Successfully imported Evolution components")
        
        engine = EvolutionEngine()
        logger.info("Successfully instantiated EvolutionEngine")
        
        config = EvolutionConfiguration()
        logger.info(f"EvolutionConfiguration default mode: {config.evolution_mode}")
        
        return True
    except Exception as e:
        logger.error(f"Evolution components instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator():
    try:
        from gauntlet_orchestrator import GauntletOrchestrator, create_all_gauntlets
        logger.info("Successfully imported Orchestrator components")
        
        orchestrator = GauntletOrchestrator()
        gauntlets = create_all_gauntlets()
        
        logger.info(f"Orchestrator created with {len(gauntlets)} gauntlets")
        return True
    except Exception as e:
        logger.error(f"Orchestrator instantiation failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting Thorough Verification of Gauntlet System and Z3 Integration")
    logger.info("=" * 70)
    
    results = {
        "Gauntlet Instantiation": test_gauntlet_instantiation(),
        "Z3 Integration": test_z3_integration(),
        "Evolution Engine": test_evolution_engine(),
        "Orchestrator": test_orchestrator()
    }
    
    logger.info("=" * 70)
    logger.info("VERIFICATION SUMMARY:")
    all_passed = True
    for test, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test:30}: {status}")
        if not result:
            all_passed = False
            
    if all_passed:
        logger.info("ALL THOROUGH CHECKS PASSED! The system is fully wired.")
        sys.exit(0)
    else:
        logger.error("SOME CHECKS FAILED. Please review the logs.")
        sys.exit(1)