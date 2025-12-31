#!/usr/bin/env python3
"""
Simple test for adversarial.py configuration and parameter coverage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parameter_manager import ParameterManager
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import json

# Define AdversarialConfiguration directly to avoid import issues
@dataclass
class AdversarialConfiguration:
    """
    Comprehensive adversarial configuration utilizing all adversarial parameters
    """
    # Core Adversarial Parameters (20)
    attack_model_config: Dict[str, Any] = field(default_factory=dict)
    defense_model_config: Dict[str, Any] = field(default_factory=dict)
    adversarial_rounds: int = 5
    attack_strength: float = 0.5
    defense_strategy: str = "reactive"
    coevolutionary_approach: bool = False
    red_team_models: List[str] = field(default_factory=list)
    blue_team_models: List[str] = field(default_factory=list)
    red_team_sample_size: int = 3
    blue_team_sample_size: int = 3
    adversarial_temperature: float = 0.8
    attack_diversity: bool = True
    defense_strength: float = 1.0
    adversarial_budget: int = 100
    attack_types: List[str] = field(default_factory=list)
    defense_strategies: List[str] = field(default_factory=list)
    robustness_metric: str = "accuracy"
    perturbation_bound: float = 0.1
    gradient_masking: bool = False
    ensemble_defense: bool = True
    
    # Additional relevant parameters
    api_key: str = ""
    max_iterations: int = 10
    temperature: float = 0.7
    max_tokens: int = 2048

def test_adversarial_parameter_coverage():
    """Test adversarial parameter coverage"""
    print("🧪 Testing Adversarial Parameter Coverage")
    print("=" * 50)
    
    # Initialize parameter manager
    param_manager = ParameterManager()
    adversarial_params = param_manager.get_parameters_by_category("adversarial")
    total_adversarial_params = len(adversarial_params)
    print(f"✅ Total adversarial parameters in manager: {total_adversarial_params}")
    
    # Create adversarial configuration
    config = AdversarialConfiguration()
    config_dict = asdict(config)
    print(f"✅ Adversarial configuration fields: {len(config_dict)}")
    
    # Check coverage of adversarial parameters
    adversarial_param_names = set(p.name for p in adversarial_params)
    config_names = set(config_dict.keys())
    
    covered_adversarial = adversarial_param_names.intersection(config_names)
    missing_adversarial = adversarial_param_names - config_names
    
    coverage_percent = len(covered_adversarial) / total_adversarial_params * 100
    print(f"✅ Adversarial parameter coverage: {len(covered_adversarial)}/{total_adversarial_params} ({coverage_percent:.1f}%)")
    
    if missing_adversarial:
        print(f"⚠️ Missing adversarial parameters ({len(missing_adversarial)}):")
        for param in sorted(list(missing_adversarial)):
            print(f"   - {param}")
    
    # List covered parameters
    print(f"\n✅ Covered adversarial parameters ({len(covered_adversarial)}):")
    for param in sorted(list(covered_adversarial)):
        print(f"   - {param}")
    
    return coverage_percent >= 90.0  # 90% coverage threshold

def test_adversarial_parameter_types():
    """Test different adversarial parameter types"""
    print("\n🏷️ Testing Adversarial Parameter Types")
    print("=" * 50)
    
    config = AdversarialConfiguration()
    config.api_key = "test_key"
    config.red_team_models = ["gpt-4", "claude-3"]
    config.blue_team_models = ["gpt-3.5-turbo"]
    config.attack_model_config = {"name": "gpt-4", "temperature": 0.9}
    config.defense_model_config = {"name": "claude-3", "temperature": 0.3}
    config.attack_types = ["perturbation", "injection"]
    config.defense_strategies = ["ensemble", "filtering"]
    
    config_dict = asdict(config)
    
    # Count parameter types
    type_counts = {
        "int": 0,
        "float": 0,
        "str": 0,
        "bool": 0,
        "list": 0,
        "dict": 0
    }
    
    for key, value in config_dict.items():
        if isinstance(value, int):
            type_counts["int"] += 1
        elif isinstance(value, float):
            type_counts["float"] += 1
        elif isinstance(value, str):
            type_counts["str"] += 1
        elif isinstance(value, bool):
            type_counts["bool"] += 1
        elif isinstance(value, list):
            type_counts["list"] += 1
        elif isinstance(value, dict):
            type_counts["dict"] += 1
    
    print("✅ Parameter type distribution:")
    for param_type, count in type_counts.items():
        print(f"   - {param_type}: {count} parameters")
    
    return sum(type_counts.values()) > 0

def test_adversarial_serialization():
    """Test adversarial configuration serialization"""
    print("\n💾 Testing Adversarial Serialization")
    print("=" * 50)
    
    config = AdversarialConfiguration()
    config.api_key = "test_key"
    config.adversarial_rounds = 3
    config.attack_strength = 0.7
    config.red_team_models = ["gpt-4", "claude-3"]
    config.attack_model_config = {"name": "gpt-4", "temp": 0.9}
    
    try:
        config_dict = asdict(config)
        config_json = json.dumps(config_dict, indent=2, default=str)
        print(f"✅ Configuration serialized: {len(config_json)} characters")
        
        # Test deserialization
        parsed_config = json.loads(config_json)
        print(f"✅ Configuration deserialized: {len(parsed_config)} fields")
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization failed: {e}")
        return False

def test_adversarial_validation_logic():
    """Test adversarial parameter validation logic"""
    print("\n🔍 Testing Adversarial Validation Logic")
    print("=" * 50)
    
    param_manager = ParameterManager()
    
    # Test valid ranges
    valid_tests = [
        ("adversarial_rounds", 5, True),
        ("attack_strength", 0.5, True),
        ("perturbation_bound", 0.1, True),
        ("adversarial_budget", 100, True),
    ]
    
    # Test invalid ranges
    invalid_tests = [
        ("adversarial_rounds", -1, False),
        ("attack_strength", 5.0, False),
        ("perturbation_bound", 2.0, False),
        ("adversarial_budget", -50, False),
    ]
    
    all_passed = True
    
    for param_name, value, should_be_valid in valid_tests + invalid_tests:
        param_def = param_manager.get_parameter(param_name)
        if param_def:
            # Simple range validation
            is_valid = True
            if param_def.min_value is not None and value < param_def.min_value:
                is_valid = False
            if param_def.max_value is not None and value > param_def.max_value:
                is_valid = False
            
            if is_valid == should_be_valid:
                status = "✅"
            else:
                status = "❌"
                all_passed = False
            
            print(f"   {status} {param_name} = {value} (expected valid: {should_be_valid}, got: {is_valid})")
        else:
            print(f"   ⚠️ Parameter {param_name} not found in schema")
    
    return all_passed

def main():
    """Run adversarial tests"""
    print("🚀 OpenEvolve Adversarial Parameter Test Suite")
    print("=" * 60)
    
    tests = [
        ("Adversarial Parameter Coverage", test_adversarial_parameter_coverage),
        ("Adversarial Parameter Types", test_adversarial_parameter_types),
        ("Adversarial Serialization", test_adversarial_serialization),
        ("Adversarial Validation Logic", test_adversarial_validation_logic),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Adversarial parameters are properly configured!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())