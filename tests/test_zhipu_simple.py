"""Test Zhipu AI (Z.ai) API"""
import time
import jwt
import requests

key = '9555ab1f79be4db498d0cf19c1277af0.8lf475NPrUgEWaF0'
api_key, secret = key.split('.')

print('Testing Zhipu AI API')
print(f'API Key: {api_key}')
print(f'Secret: {secret}')

# JWT Auth
timestamp = int(time.time() * 1000)
payload = {
    'api_key': api_key,
    'exp': timestamp + 3600000,
    'timestamp': timestamp
}
token = jwt.encode(payload, secret, algorithm='HS256')

url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
data = {'model': 'glm-5', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 20}

print(f'\nSending request to {url}')
print(f'Token: {token[:50]}...')

try:
    response = requests.post(url, json=data, headers=headers, timeout=30)
    print(f'Status: {response.status_code}')
    print(f'Content: {response.content}')
except Exception as e:
    print(f'Exception: {e}')
