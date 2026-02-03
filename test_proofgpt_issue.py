"""
Test the specific proofGPT model that causes the crash
"""
import sys
import traceback

def test_proofgpt_original():
    """Test original loading method that crashes"""
    try:
        print("\n=== Test 1: Original proofGPT loading ===")
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("hoskinson-center/proofGPT-v0.1-6.7B")
        print("Tokenizer loaded")

        print("Loading model (this may crash)...")
        model = AutoModelForCausalLM.from_pretrained("hoskinson-center/proofGPT-v0.1-6.7B")
        print(f"Model loaded: {type(model)}")
        print("[PASS] Original loading works")
        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()
        return False

def test_proofgpt_safe():
    """Test safe loading method"""
    try:
        print("\n=== Test 2: Safe proofGPT loading ===")
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("hoskinson-center/proofGPT-v0.1-6.7B")
        print("Tokenizer loaded")

        print("Loading model with low_cpu_mem_usage=False...")
        model = AutoModelForCausalLM.from_pretrained(
            "hoskinson-center/proofGPT-v0.1-6.7B",
            low_cpu_mem_usage=False  # Disable meta device loading
        )
        print(f"Model loaded: {type(model)}")
        print("[PASS] Safe loading works")
        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()
        return False

def test_proofgpt_torch_dtype():
    """Test with explicit torch dtype"""
    try:
        print("\n=== Test 3: proofGPT with explicit dtype ===")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("hoskinson-center/proofGPT-v0.1-6.7B")
        print("Tokenizer loaded")

        print("Loading model with torch_dtype=float32...")
        model = AutoModelForCausalLM.from_pretrained(
            "hoskinson-center/proofGPT-v0.1-6.7B",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False
        )
        print(f"Model loaded: {type(model)}")
        print("[PASS] Explicit dtype loading works")
        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("proofGPT Model Loading Test")
    print("=" * 60)

    # Run tests
    test1_passed = test_proofgpt_original()
    test2_passed = test_proofgpt_safe()
    test3_passed = test_proofgpt_torch_dtype()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Original loading: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Safe loading: {'PASS' if test2_passed else 'FAIL'}")
    print(f"  Explicit dtype loading: {'PASS' if test3_passed else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if (test1_passed or test2_passed or test3_passed) else 1)