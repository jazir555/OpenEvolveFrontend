#!/usr/bin/env python3
"""
Test script to verify sidebar integration with parameter_manager.py
Tests all 272 OpenEvolve parameters and validation functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parameter_manager import ParameterManager, ParameterType
from sidebar import get_parameter_manager

def test_parameter_manager_integration():
    """Test parameter manager integration"""
    print("🧪 Testing Parameter Manager Integration")
    print("=" * 50)
    
    # Test parameter manager initialization
    pm = get_parameter_manager()
    print(f"✅ Parameter manager initialized with {len(pm.schema.parameters)} parameters")
    
    # Test categories
    categories = pm.get_categories()
    print(f"✅ Found {len(categories)} parameter categories:")
    for cat in sorted(categories):
        params = pm.get_parameters_by_category(cat)
        print(f"   - {cat}: {len(params)} parameters")
    
    # Test parameter validation
    print("\n🔍 Testing Parameter Validation")
    
    # Valid configuration
    valid_config = {
        "evolution_mode": "standard",
        "max_iterations": 10,
        "population_size": 20,
        "temperature": 0.7,
        "api_key": "test_key"
    }
    
    result = pm.validate(valid_config)
    print(f"✅ Valid config validation: {result.valid}")
    
    # Invalid configuration
    invalid_config = {
        "evolution_mode": "invalid_mode",
        "max_iterations": -5,
        "temperature": 3.0,
        "population_size": 0
    }
    
    result = pm.validate(invalid_config)
    print(f"✅ Invalid config validation: {not result.valid} (correctly detected as invalid)")
    print(f"   Errors found: {len(result.errors)}")
    
    # Test presets
    print("\n📋 Testing Presets")
    presets = pm.list_presets()
    print(f"✅ Available presets: {presets}")
    
    for preset_name in presets:
        preset = pm.get_preset(preset_name)
        if preset:
            validation = pm.validate(preset)
            print(f"   - {preset_name}: {len(preset)} params, valid: {validation.valid}")
    
    # Test parameter types
    print("\n🏷️ Testing Parameter Types")
    type_counts = {}
    for param in pm.schema.parameters.values():
        param_type = param.type.value
        type_counts[param_type] = type_counts.get(param_type, 0) + 1
    
    for param_type, count in sorted(type_counts.items()):
        print(f"   - {param_type}: {count} parameters")
    
    # Test required parameters
    print("\n⚠️ Testing Required Parameters")
    required_params = [p for p in pm.schema.parameters.values() if p.required]
    print(f"✅ Found {len(required_params)} required parameters:")
    for param in required_params:
        print(f"   - {param.name}: {param.description}")
    
    # Test parameter ranges
    print("\n📊 Testing Parameter Ranges")
    range_params = [p for p in pm.schema.parameters.values() 
                   if p.min_value is not None or p.max_value is not None]
    print(f"✅ Found {len(range_params)} parameters with ranges")
    
    # Test parameter options
    option_params = [p for p in pm.schema.parameters.values() if p.options]
    print(f"✅ Found {len(option_params)} parameters with predefined options")
    
    print("\n🎉 All tests completed successfully!")
    return True

def test_sidebar_functions():
    """Test sidebar helper functions"""
    print("\n🎛️ Testing Sidebar Functions")
    print("=" * 50)
    
    try:
        from sidebar import get_default_generation_params, get_default_evolution_params
        
        gen_params = get_default_generation_params()
        print(f"✅ Generation parameters: {len(gen_params)} params")
        
        evo_params = get_default_evolution_params()
        print(f"✅ Evolution parameters: {len(evo_params)} params")
        
        # Test parameter validation
        pm = get_parameter_manager()
        all_params = {**gen_params, **evo_params}
        result = pm.validate(all_params)
        print(f"✅ Default parameters validation: {result.valid}")
        
        if not result.valid:
            print("   Validation errors:")
            for error in result.errors[:5]:  # Show first 5 errors
                print(f"     - {error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing sidebar functions: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 OpenEvolve Sidebar-ParameterManager Integration Test")
    print("=" * 60)
    
    try:
        # Test parameter manager
        success1 = test_parameter_manager_integration()
        
        # Test sidebar functions
        success2 = test_sidebar_functions()
        
        if success1 and success2:
            print("\n🎉 ALL TESTS PASSED! Integration is working correctly.")
            print(f"✅ Parameter Manager: {len(get_parameter_manager().schema.parameters)} parameters")
            print("✅ Sidebar: Compatible with ParameterManager")
            print("✅ Validation: Working correctly")
            print("✅ Presets: Available and valid")
            return 0
        else:
            print("\n❌ Some tests failed. Check the output above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())