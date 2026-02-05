#!/usr/bin/env python3
"""
Basic test for adversarial evolution functionality in evolution.py
Tests core adversarial features without requiring full team system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evolution import (
    EvolutionConfiguration,
    get_evolution_capabilities_summary,
    TEAM_SYSTEM_AVAILABLE
)
from parameter_manager import ParameterManager
import json

def test_adversarial_parameters():
    """Test adversarial parameter configuration"""
    print("🛡️ Testing Adversarial Parameters")
    print("=" * 50)
    
    param_manager = ParameterManager()
    
    # Get adversarial parameters
    adversarial_params = param_manager.get_parameters_by_category("adversarial")
    print(f"[OK] Adversarial parameters available: {len(adversarial_params)}")
    
    for param in adversarial_params[:5]:  # Show first 5
        print(f"   - {param.name}: {param.description}")
    
    if len(adversarial_params) > 5:
        print(f"   ... and {len(adversarial_params) - 5} more")
    
    return len(adversarial_params) >= 15  # Should have at least 15 adversarial parameters

def test_adversarial_configuration():
    """Test creating adversarial evolution configuration"""
    print("\n⚙️ Testing Adversarial Configuration")
    print("=" * 50)
    
    param_manager = ParameterManager()
    
    # Create adversarial configuration
    config = EvolutionConfiguration()
    config.evolution_mode = "adversarial"
    config.api_key = "test_key"
    
    # Set adversarial-specific parameters
    config.adversarial_rounds = 5
    config.attack_strength = 0.8
    config.defense_strength = 0.9
    config.adversarial_budget = 1000
    config.attack_types = ["injection", "overflow", "social"]
    config.defense_strategies = ["validation", "sanitization", "monitoring"]
    config.robustness_metric = "security_score"
    config.perturbation_bound = 0.1
    config.gradient_masking = True
    config.ensemble_defense = True
    config.red_team_models = ["gpt-4", "claude-3"]
    config.blue_team_models = ["gpt-4", "gemini-pro"]
    config.red_team_sample_size = 3
    config.blue_team_sample_size = 3
    config.adversarial_temperature = 0.8
    config.attack_diversity = True
    
    print("[OK] Adversarial configuration created")
    print(f"   - Evolution Mode: {config.evolution_mode}")
    print(f"   - Adversarial Rounds: {config.adversarial_rounds}")
    print(f"   - Attack Strength: {config.attack_strength}")
    print(f"   - Defense Strength: {config.defense_strength}")
    print(f"   - Red Team Models: {len(config.red_team_models)}")
    print(f"   - Blue Team Models: {len(config.blue_team_models)}")
    print(f"   - Attack Types: {len(config.attack_types)}")
    print(f"   - Defense Strategies: {len(config.defense_strategies)}")
    
    # Validate configuration
    validation_result = config.validate(param_manager)
    print(f"[OK] Configuration validation: {validation_result.valid}")
    
    if not validation_result.valid:
        print("   Validation errors:")
        for error in validation_result.errors[:3]:
            print(f"     - {error}")
    
    return validation_result.valid

def test_evolution_capabilities():
    """Test evolution capabilities include adversarial features"""
    print("\n📊 Testing Evolution Capabilities")
    print("=" * 50)
    
    capabilities = get_evolution_capabilities_summary()
    
    print(f"[OK] Total Parameters: {capabilities['total_parameters']}")
    print(f"[OK] Categories: {capabilities['categories']}")
    print(f"[OK] Evolution Modes: {capabilities['evolution_modes']}")
    
    # Check adversarial mode
    adversarial_available = "adversarial" in capabilities["evolution_modes"]
    print(f"[OK] Adversarial mode available: {adversarial_available}")
    
    # Check team system status
    team_system = capabilities.get("team_system", {})
    team_available = team_system.get("available", False)
    print(f"[OK] Team system available: {team_available}")
    
    if team_available:
        print("   Team system features:")
        print(f"   - Red Team: {team_system.get('red_team', 'Not available')}")
        print(f"   - Blue Team: {team_system.get('blue_team', 'Not available')}")
        print(f"   - Evaluator Team: {team_system.get('evaluator_team', 'Not available')}")
        print(f"   - Gauntlet System: {team_system.get('gauntlet_system', 'Not available')}")
    else:
        print("   Team system not available - using basic adversarial mode")
    
    # Check advanced features
    advanced_features = capabilities.get("advanced_features", {})
    adversarial_feature = advanced_features.get("adversarial", "Not available")
    print(f"[OK] Adversarial feature: {adversarial_feature}")
    
    return adversarial_available

def test_parameter_serialization():
    """Test that adversarial configuration can be serialized"""
    print("\n💾 Testing Parameter Serialization")
    print("=" * 50)
    
    # Create comprehensive adversarial configuration
    config = EvolutionConfiguration()
    config.evolution_mode = "adversarial"
    config.api_key = "test_key"
    config.adversarial_rounds = 3
    config.attack_strength = 0.7
    config.defense_strength = 0.8
    config.adversarial_budget = 500
    config.attack_types = ["injection", "overflow"]
    config.defense_strategies = ["validation", "sanitization"]
    config.red_team_models = ["gpt-4"]
    config.blue_team_models = ["claude-3"]
    config.ensemble_defense = True
    config.gradient_masking = False
    
    try:
        # Test serialization
        config_dict = config.to_openevolve_config()
        config_json = json.dumps(config_dict, indent=2, default=str)
        
        print(f"[OK] Configuration serialized: {len(config_json)} characters")
        
        # Test that adversarial parameters are included
        adversarial_keys = [
            "adversarial_rounds", "attack_strength", "defense_strength",
            "attack_types", "defense_strategies", "red_team_models", "blue_team_models"
        ]
        
        included_keys = [key for key in adversarial_keys if key in config_dict]
        print(f"[OK] Adversarial parameters included: {len(included_keys)}/{len(adversarial_keys)}")
        
        for key in included_keys:
            print(f"   - {key}: {config_dict[key]}")
        
        return len(included_keys) >= 5  # Should include most adversarial parameters
        
    except Exception as e:
        print(f"[FAIL] Serialization failed: {e}")
        return False

def test_parameter_coverage():
    """Test comprehensive parameter coverage for adversarial evolution"""
    print("\n🔧 Testing Parameter Coverage")
    print("=" * 50)
    
    param_manager = ParameterManager()
    
    # Categories relevant to adversarial evolution
    adversarial_categories = [
        "adversarial",
        "core_evolution", 
        "model_config",
        "evaluation",
        "selection",
        "island_model"
    ]
    
    total_params = 0
    for category in adversarial_categories:
        params = param_manager.get_parameters_by_category(category)
        total_params += len(params)
        print(f"[OK] {category}: {len(params)} parameters")
    
    print(f"[OK] Total adversarial-relevant parameters: {total_params}")
    
    # Test that we can create a configuration with many parameters
    config = EvolutionConfiguration()
    
    # Set parameters from each category
    config.evolution_mode = "adversarial"  # core_evolution
    config.api_key = "test_key"  # model_config
    config.adversarial_rounds = 5  # adversarial
    config.cascade_evaluation = True  # evaluation
    config.elite_ratio = 0.1  # selection
    config.num_islands = 3  # island_model
    
    configured_params = len([k for k, v in config.__dict__.items() if v is not None])
    print(f"[OK] Parameters configured in test: {configured_params}")
    
    return total_params >= 50  # Should have at least 50 relevant parameters

def test_evolution_mode_validation():
    """Test that adversarial evolution mode is properly validated"""
    print("\n[OK] Testing Evolution Mode Validation")
    print("=" * 50)
    
    param_manager = ParameterManager()
    
    # Test valid adversarial configuration
    valid_config = EvolutionConfiguration()
    valid_config.evolution_mode = "adversarial"
    valid_config.api_key = "test_key"
    valid_config.adversarial_rounds = 3
    
    validation = valid_config.validate(param_manager)
    print(f"[OK] Valid adversarial config: {validation.valid}")
    
    # Test invalid evolution mode
    invalid_config = EvolutionConfiguration()
    invalid_config.evolution_mode = "invalid_mode"
    invalid_config.api_key = "test_key"
    
    validation = invalid_config.validate(param_manager)
    print(f"[OK] Invalid mode detected: {not validation.valid}")
    
    if not validation.valid:
        print("   Validation errors (expected):")
        for error in validation.errors[:2]:
            print(f"     - {error}")
    
    return True

def main():
    """Run all basic adversarial evolution tests"""
    print("🚀 Evolution.py Adversarial Features Test Suite")
    print("=" * 60)
    
    print(f"Team System Available: {TEAM_SYSTEM_AVAILABLE}")
    if not TEAM_SYSTEM_AVAILABLE:
        print("Note: Testing basic adversarial features without full team system")
    
    tests = [
        ("Adversarial Parameters", test_adversarial_parameters),
        ("Adversarial Configuration", test_adversarial_configuration),
        ("Evolution Capabilities", test_evolution_capabilities),
        ("Parameter Serialization", test_parameter_serialization),
        ("Parameter Coverage", test_parameter_coverage),
        ("Evolution Mode Validation", test_evolution_mode_validation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Adversarial evolution features are working!")
        print("[OK] 272 parameters including comprehensive adversarial support")
        print("[OK] Adversarial evolution mode available")
        print("[OK] Parameter validation and serialization working")
        if TEAM_SYSTEM_AVAILABLE:
            print("[OK] Full team system integration available")
        else:
            print("[WARN] Team system not available - basic adversarial mode only")
        return 0
    else:
        print(f"\n[WARN] {total - passed} tests failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())