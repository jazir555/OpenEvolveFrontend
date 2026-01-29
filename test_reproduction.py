<<<<<<< HEAD
"""
Minimal reproduction test for PyTorch/Transformers crash on Windows
Tests the meta model loading issue
"""
import sys
import traceback

def test_meta_device_issue():
    """Test if the issue is related to meta device loading"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Test 1: Simple model creation
        print("\n=== Test 1: Simple model creation ===")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("gpt2")
        print(f"Config loaded successfully: {config.model_type}")

        # Test 2: Model without weights (test meta device)
        print("\n=== Test 2: Model with meta device ===")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            device_map="meta",
            trust_remote_code=False
        )
        print(f"Meta model created: {type(model)}")

        # Test 3: Actual model loading with low memory
        print("\n=== Test 3: Actual model loading (low_mem) ===")
        model2 = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            low_cpu_mem_usage=True,
            trust_remote_code=False
        )
        print(f"Model loaded successfully: {type(model2)}")
        print("[PASS] All tests passed!")

    except Exception as e:
        print(f"\n[FAIL] Error occurred: {e}")
        traceback.print_exc()
        return False

    return True

def test_alternative_loading():
    """Test alternative loading strategies"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\n=== Test 4: Safe loading without meta device ===")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("Tokenizer loaded")

        # Load without low_cpu_mem_usage which causes the issue
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            low_cpu_mem_usage=False,  # This is the key fix
            trust_remote_code=False
        )
        print(f"Model loaded successfully: {type(model)}")
        print("[PASS] Alternative loading works!")

        return True

    except Exception as e:
        print(f"\n[FAIL] Alternative loading failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch/Transformers Crash Reproduction Test")
    print("=" * 60)

    # Run tests
    test1_passed = test_meta_device_issue()
    test2_passed = test_alternative_loading()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Meta device test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Alternative loading test: {'PASS' if test2_passed else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if (test1_passed or test2_passed) else 1)
=======
"""
Minimal reproduction test for PyTorch/Transformers crash on Windows
Tests the meta model loading issue
"""
import sys
import traceback

def test_meta_device_issue():
    """Test if the issue is related to meta device loading"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Test 1: Simple model creation
        print("\n=== Test 1: Simple model creation ===")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("gpt2")
        print(f"Config loaded successfully: {config.model_type}")

        # Test 2: Model without weights (test meta device)
        print("\n=== Test 2: Model with meta device ===")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            device_map="meta",
            trust_remote_code=False
        )
        print(f"Meta model created: {type(model)}")

        # Test 3: Actual model loading with low memory
        print("\n=== Test 3: Actual model loading (low_mem) ===")
        model2 = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            low_cpu_mem_usage=True,
            trust_remote_code=False
        )
        print(f"Model loaded successfully: {type(model2)}")
        print("[PASS] All tests passed!")

    except Exception as e:
        print(f"\n[FAIL] Error occurred: {e}")
        traceback.print_exc()
        return False

    return True

def test_alternative_loading():
    """Test alternative loading strategies"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\n=== Test 4: Safe loading without meta device ===")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("Tokenizer loaded")

        # Load without low_cpu_mem_usage which causes the issue
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            low_cpu_mem_usage=False,  # This is the key fix
            trust_remote_code=False
        )
        print(f"Model loaded successfully: {type(model)}")
        print("[PASS] Alternative loading works!")

        return True

    except Exception as e:
        print(f"\n[FAIL] Alternative loading failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch/Transformers Crash Reproduction Test")
    print("=" * 60)

    # Run tests
    test1_passed = test_meta_device_issue()
    test2_passed = test_alternative_loading()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Meta device test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Alternative loading test: {'PASS' if test2_passed else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if (test1_passed or test2_passed) else 1)
>>>>>>> 1cb9c5e35 (update)
