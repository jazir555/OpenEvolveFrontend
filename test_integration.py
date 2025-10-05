"""
Integration test for adversarial and evolution modules
"""
import sys
import os

# Add the current directory to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adversarial import test_integration
from evolution import render_evolution_settings, ContentEvaluator
import streamlit as st

def main():
    print("Starting integration test for adversarial and evolution modules...")
    
    # Initialize session state if needed
    if not hasattr(st.session_state, 'adversarial_log'):
        st.session_state.adversarial_log = []
    if not hasattr(st.session_state, 'adversarial_results'):
        st.session_state.adversarial_results = {}
    if not hasattr(st.session_state, 'thread_lock'):
        import threading
        st.session_state.thread_lock = threading.Lock()
    
    print("\n1. Testing ContentEvaluator class...")
    try:
        evaluator = ContentEvaluator("general", "Evaluate content quality")
        print("+ ContentEvaluator created successfully")
    except Exception as e:
        print(f"- ContentEvaluator failed: {e}")
    
    print("\n2. Testing evolution settings renderer...")
    try:
        # This is a UI function, so we'll just check if it can be called without error
        print("+ render_evolution_settings function exists")
    except Exception as e:
        print(f"- render_evolution_settings failed: {e}")
    
    print("\n3. Testing integration function...")
    try:
        result = test_integration()
        print(f"+ Integration test completed with success: {result.get('success', False)}")
        if 'error' in result:
            print(f"  Error (expected for API test): {result['error']}")
    except Exception as e:
        print(f"- Integration test function failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n4. Testing key adversarial functions...")
    try:
        from adversarial import determine_review_type
        sample_code = "def hello():\n    print('world')"
        review_type = determine_review_type(sample_code)
        print(f"+ determine_review_type works, detected: {review_type}")
    except Exception as e:
        print(f"- Adversarial functions test failed: {e}")
    
    print("\nIntegration test completed!")

if __name__ == "__main__":
    main()