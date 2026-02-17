"""
Test DeepSeek and Z.ai API Keys

Actually calls the APIs to verify they work.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_deepseek_api():
    """Test DeepSeek API key with actual API call."""
    print("\n" + "="*80)
    print("TESTING DEEPSEEK API")
    print("="*80)
    
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("[FAIL] DEEPSEEK_API_KEY not found in environment")
        return False
    
    print(f"API Key found: {api_key[:10]}...")
    
    # DeepSeek API endpoint
    url = "https://api.deepseek.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'DeepSeek API test successful' in one sentence."}
        ],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"[PASS] DeepSeek API responded successfully!")
            print(f"Response: {content}")
            print(f"Usage: {data.get('usage', {})}")
            return True
        else:
            print(f"[FAIL] DeepSeek API returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("[FAIL] DeepSeek API request timed out")
        return False
    except Exception as e:
        print(f"[FAIL] DeepSeek API test failed: {e}")
        return False


def test_zai_api():
    """Test Z.ai API key with actual API call."""
    print("\n" + "="*80)
    print("TESTING Z.AI API (Zhipu AI)")
    print("="*80)
    
    api_key = os.getenv('ZAI_API_KEY')
    if not api_key:
        print("[FAIL] ZAI_API_KEY not found in environment")
        return False
    
    print(f"API Key found: {api_key[:10]}...{api_key[-5:]}")
    
    # Zhipu AI uses JWT authentication with api_key.secret format
    # For simplicity, try direct token format first
    import time
    import jwt
    
    parts = api_key.split('.')
    if len(parts) != 2:
        print("[FAIL] Invalid Z.ai key format (expected api_key.secret)")
        return False
    
    api_key_part, secret_part = parts
    
    # Create JWT token (Zhipu AI format)
    payload_jwt = {
        "api_key": api_key_part,
        "exp": int(time.time()) + 3600,  # 1 hour
        "timestamp": int(time.time() * 1000)
    }
    
    try:
        token = jwt.encode(payload_jwt, secret_part, algorithm="HS256")
    except:
        # Fallback: try using key directly
        token = api_key
    
    # Zhipu AI API endpoint
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    payload = {
        "model": "glm-4",
        "messages": [
            {"role": "user", "content": "Say 'Zhipu AI test successful'"}
        ],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"[PASS] Zhipu AI API responded successfully!")
            print(f"Response: {content}")
            return True
        else:
            print(f"[FAIL] Zhipu AI API returned status {response.status_code}")
            try:
                err_data = response.json()
                print(f"Error: {err_data.get('error', {}).get('message', 'Unknown error')}")
            except:
                print(f"Error response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("[FAIL] Zhipu AI API request timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Zhipu AI API test failed: {e}")
        return False


def test_with_litellm():
    """Test APIs using LiteLLM for unified interface."""
    print("\n" + "="*80)
    print("TESTING WITH LITELLM (unified interface)")
    print("="*80)
    
    try:
        import litellm
        from litellm import completion
        
        # Test DeepSeek via LiteLLM
        print("\n--- Testing DeepSeek via LiteLLM ---")
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        if deepseek_key:
            try:
                response = completion(
                    model="deepseek/deepseek-chat",
                    messages=[{"role": "user", "content": "Say 'LiteLLM DeepSeek test OK'"}],
                    api_key=deepseek_key,
                    max_tokens=20
                )
                print(f"[PASS] LiteLLM DeepSeek: {response.choices[0].message.content}")
            except Exception as e:
                print(f"[WARN] LiteLLM DeepSeek failed: {e}")
        
        # Test Z.ai via LiteLLM
        print("\n--- Testing Z.ai via LiteLLM ---")
        zai_key = os.getenv('ZAI_API_KEY')
        if zai_key:
            try:
                response = completion(
                    model="zhipuai/glm-4",
                    messages=[{"role": "user", "content": "Say 'LiteLLM Z.ai test OK'"}],
                    api_key=zai_key,
                    max_tokens=20
                )
                print(f"[PASS] LiteLLM Z.ai: {response.choices[0].message.content}")
            except Exception as e:
                print(f"[WARN] LiteLLM Z.ai failed: {e}")
        
        return True
        
    except ImportError:
        print("[WARN] LiteLLM not available")
        return False
    except Exception as e:
        print(f"[WARN] LiteLLM test failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("API KEY VERIFICATION TEST")
    print("="*80)
    print(f"Date: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = []
    
    # Test DeepSeek
    results.append(("DeepSeek Direct API", test_deepseek_api()))
    
    # Test Z.ai
    results.append(("Z.ai Direct API", test_zai_api()))
    
    # Test via LiteLLM
    results.append(("LiteLLM Integration", test_with_litellm()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    print("="*80)
    
    if passed == total:
        print("\n[SUCCESS] ALL API KEYS ARE WORKING!")
    else:
        print(f"\n[WARN] {total - passed} test(s) failed. Check your API keys.")
