"""Test Z.ai API Key - Zhipu AI Format"""
import time
import jwt
import requests

# Z.ai key
api_key = '9555ab1f79be4db498d0cf19c1277af0.8lf475NPrUgEWaF0'
parts = api_key.split('.')
api_key_part, secret_part = parts

print(f"API Key Part: {api_key_part}")
print(f"Secret Part: {secret_part}")

# Zhipu AI uses this JWT format
timestamp = int(time.time() * 1000)
exp = timestamp + 3600000  # 1 hour in milliseconds

payload = {
    "api_key": api_key_part,
    "exp": exp,
    "timestamp": timestamp
}

# Try without padding
token = jwt.encode(payload, secret_part, algorithm='HS256', headers={'alg': 'HS256', 'sign_type': 'JWT'})
print(f'\nJWT Token: {token}')

# Test API
url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {token}'
}
data = {
    'model': 'glm-4',
    'messages': [{'role': 'user', 'content': 'Say hello'}],
    'max_tokens': 20
}

print(f'\nSending request...')
try:
    response = requests.post(url, json=data, headers=headers, timeout=30)
    print(f'Status: {response.status_code}')
    # Read as bytes to avoid encoding issues
    content = response.content.decode('utf-8', errors='replace')
    print(f'Response: {content}')
except Exception as e:
    print(f'Error: {e}')

# Also try the alternative endpoint
print('\n--- Trying alternative endpoint ---')
url2 = 'https://open.bigmodel.cn/dev/api#chat/completions'
print(f'Alternative URL: {url2}')
