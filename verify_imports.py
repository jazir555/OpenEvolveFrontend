#!/usr/bin/env python3
"""
Quick verification script - imports all critical RESE modules
"""
import sys
sys.path.insert(0, 'rese')

print("="*80)
print("RESE FRAMEWORK CRITICAL MODULE IMPORT VERIFICATION")
print("="*80)

modules = [
    # Core
    ("core.symbolic_constraint_engine", "Symbolic Constraint Engine"),
    ("core.dito_optimizer", "DITO Optimizer"),
    ("core.constraint_optimizer", "Constraint Optimizer"),
    
    # Gamma1
    ("gamma1.core.aci_calculator", "ACI Calculator"),
    ("gamma1.core.coherence_engine", "Coherence Engine"),
    
    # Phase 1
    ("phase1.tacit_assumption_miner", "Tacit Assumption Miner (Φ₁₅)"),
    ("phase1.cognitive_biases", "Cognitive Biases"),
    
    # Phase 2
    ("phase2.imech.isomorphism_validator", "I_mech Validator"),
    
    # Phase 3
    ("phase3.mcts_search", "MCTS Search"),
    ("phase3.convergence_controller", "Convergence Controller"),
    
    # Phase 4
    ("phase4.aci_reduction_validator", "ACI Reduction Validator"),
    ("phase4.predictive_model_generator", "Predictive Model Generator"),
    
    # Security
    ("security.error_handler", "Error Handler"),
    ("security.input_validator", "Input Validator"),
    
    # Integrations
    ("integrations.stage1", "Stage 1 Integration"),
    ("integrations.stage9", "Stage 9 Integration"),
]

success = 0
failed = 0

for module, name in modules:
    try:
        __import__(module)
        print(f"[OK] {name:40s} [{module}]")
        success += 1
    except Exception as e:
        print(f"[FAIL] {name:40s} [{module}]")
        print(f"  Error: {e}")
        failed += 1

print("="*80)
print(f"Results: {success} successful, {failed} failed")
print("="*80)

sys.exit(0 if failed == 0 else 1)
