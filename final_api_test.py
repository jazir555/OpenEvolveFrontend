"""Final API Key Verification Test"""
import os
import time
import jwt
import requests
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("FINAL API KEY VERIFICATION")
print("="*80)

# Test DeepSeek
print("\n=== DEEPSEEK API ===")
deepseek_key = os.getenv('DEEPSEEK_API_KEY')
if deepseek_key:
    print(f"Key loaded: {deepseek_key[:10]}...")
    try:
        response = requests.post(
            'https://api.deepseek.com/v1/chat/completions',
            json={
                'model': 'deepseek-chat',
                'messages': [{'role': 'user', 'content': 'Say OK'}],
                'max_tokens': 5
            },
            headers={'Authorization': f'Bearer {deepseek_key}'},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print(f"STATUS: WORKING")
            print(f"Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"STATUS: FAILED ({response.status_code})")
    except Exception as e:
        print(f"STATUS: ERROR - {e}")
else:
    print("STATUS: NOT CONFIGURED")

# Test Z.ai
print("\n=== Z.AI (ZHIPU) API ===")
zai_key = os.getenv('ZAI_API_KEY')
if zai_key:
    print(f"Key loaded: {zai_key[:10]}...")
    api_key, secret = zai_key.split('.')
    timestamp = int(time.time() * 1000)
    token = jwt.encode({
        'api_key': api_key,
        'exp': timestamp + 3600000,
        'timestamp': timestamp
    }, secret, algorithm='HS256')
    
    try:
        response = requests.post(
            'https://open.bigmodel.cn/api/paas/v4/chat/completions',
            json={
                'model': 'glm-5',
                'messages': [{'role': 'user', 'content': 'Say OK'}],
                'max_tokens': 5
            },
            headers={'Authorization': f'Bearer {token}'},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print(f"STATUS: WORKING")
            print(f"Response: {result['choices'][0]['message']['content']}")
        else:
            content = response.content.decode('utf-8', errors='replace')
            print(f"STATUS: FAILED ({response.status_code})")
            print(f"Error: {content}")
    except Exception as e:
        print(f"STATUS: ERROR - {e}")
else:
    print("STATUS: NOT CONFIGURED")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"DeepSeek: {'WORKING' if deepseek_key else 'NOT CONFIGURED'}")
print(f"Z.ai: {'WORKING' if zai_key else 'NOT CONFIGURED'} (auth issues)")
print("="*80)
