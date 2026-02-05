#!/usr/bin/env python3
"""
Test script to verify OpenEvolve configuration is working
"""

def test_openevolve_configuration():
    """Test that OpenEvolve configuration works properly"""
    try:
        from openevolve_client import OpenEvolveClient
        
        print("[OK] OpenEvolveClient imported successfully!")
        
        # Test basic client initialization
        client = OpenEvolveClient()
        print(f"[OK] Client initialized, available: {client.available}")
        
        # Test configuration creation with fallback
        try:
            config = client._prepare_config(
                evolution_mode='standard',
                content_type='general',
                evaluator=None
            )
            print("[OK] Fallback configuration created successfully!")
            print(f"   - LLM models configured: {len(config.llm.models)}")
            print(f"   - Evolution mode: {config.evolution_mode}")
        except Exception as e:
            print(f"[FAIL] Fallback configuration failed: {e}")
            return False
        
        # Test configuration with API key
        try:
            config_with_key = client._prepare_config(
                evolution_mode='standard',
                content_type='general',
                evaluator=None,
                api_key='test-key',
                model_name='gpt-3.5-turbo'
            )
            print("[OK] Configuration with API key created successfully!")
            print(f"   - Model name: {config_with_key.llm.models[0].name}")
            print(f"   - API key set: {'test-key' in config_with_key.llm.models[0].api_key}")
        except Exception as e:
            print(f"[FAIL] Configuration with API key failed: {e}")
            return False
        
        # Test validated configuration creation
        try:
            validated_config = client.create_config_with_validation(
                api_key='test-key',
                model_name='gpt-4',
                evolution_mode='quality_diversity',
                max_iterations=5,
                population_size=10
            )
            print("[OK] Validated configuration created successfully!")
            print(f"   - Evolution mode: {validated_config.evolution_mode}")
            print(f"   - Max iterations: {validated_config.max_iterations}")
            print(f"   - Population size: {validated_config.database.population_size}")
        except Exception as e:
            print(f"[FAIL] Validated configuration failed: {e}")
            return False
        
        # Test configuration validation without API key
        try:
            client.create_config_with_validation()
            print("[FAIL] Should have failed without API key!")
            return False
        except ValueError as e:
            print(f"[OK] Properly rejected configuration without API key: {e}")
        except Exception as e:
            print(f"[FAIL] Unexpected error: {e}")
            return False
        
        print("[OK] OpenEvolve configuration system is working!")
        return True
        
    except Exception as e:
        print(f"[FAIL] OpenEvolve configuration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_openevolve_configuration()
    if success:
        print("\n🎉 OpenEvolve configuration fix successful! Critical blocker #2 RESOLVED!")
    else:
        print("\n💥 OpenEvolve configuration still has issues")